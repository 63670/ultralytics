# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export YOLOv8-style detectors with the nine-output layout used by Rockchip's model zoo."""
from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules import ContentAwareUpsample, Detect


class ContentAwareUpsample4D(nn.Module):
    """Reassemble CAFR with 4-D shifted features, without materializing an unfolded neighborhood tensor."""

    def __init__(self, source: ContentAwareUpsample):
        """Reuse the learned kernel predictor from a private model copy."""
        super().__init__()
        self.kernel_size = source.kernel_size
        self.scale_factor = source.scale_factor
        if self.scale_factor != 2 or self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("CAFR 4D reassembly requires an odd positive kernel and scale_factor=2.")
        self.compress = source.compress
        self.kernel_encoder = source.kernel_encoder
        for attr in ("i", "f", "type", "np"):
            if hasattr(source, attr):
                setattr(self, attr, getattr(source, attr))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the same k*k weighted neighbors using only BCHW feature tensors."""
        weights = F.pixel_shuffle(self.kernel_encoder(self.compress(x)), self.scale_factor).softmax(dim=1)
        upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        height, width = upsampled.shape[-2:]
        padding = self.scale_factor * (self.kernel_size // 2)
        padded = F.pad(upsampled, (padding, padding, padding, padding))
        result = None
        for row in range(self.kernel_size):
            for column in range(self.kernel_size):
                top, left = row * self.scale_factor, column * self.scale_factor
                index = row * self.kernel_size + column
                neighbor = padded[:, :, top : top + height, left : left + width]
                weighted = neighbor * weights[:, index : index + 1]
                result = weighted if result is None else result + weighted
        return result


class RockchipDetectOutputs(nn.Module):
    """Emit raw box distributions, sigmoid class scores, and clipped score sums per scale."""

    def __init__(self, head: Detect):
        """Reuse trained prediction convolutions, without DFL or box decoding."""
        super().__init__()
        self.box_heads = head.cv2
        self.class_heads = head.cv3
        for attr in ("i", "f", "type", "np"):
            if hasattr(head, attr):
                setattr(self, attr, getattr(head, attr))

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Match airockchip/ultralytics_yolov8: box, class, score_sum at strides 8, 16, 32."""
        outputs = []
        for feature, box_head, class_head in zip(features, self.box_heads, self.class_heads):
            scores = class_head(feature).sigmoid()
            outputs.extend((box_head(feature), scores, scores.sum(dim=1, keepdim=True).clamp(0, 1)))
        return tuple(outputs)


def prepare_rkopt_model(model: nn.Module, cafr_mode: str = "original") -> nn.Module:
    """Prepare a private FP32 CPU copy; preserve custom backbone forwards and the original model."""
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or head.nl != 3:
        raise ValueError("RKOPT requires a three-scale, non-end-to-end Detect head (no segment/pose/OBB subclasses).")
    if cafr_mode not in {"original", "streaming"}:
        raise ValueError("cafr_mode must be 'original' or 'streaming'.")
    model = deepcopy(model).cpu().float().eval()
    model.requires_grad_(False)
    model.fuse(verbose=False)
    if cafr_mode == "streaming":
        for parent in tuple(model.modules()):
            for name, child in tuple(parent.named_children()):
                if type(child) is ContentAwareUpsample:
                    setattr(parent, name, ContentAwareUpsample4D(child))
    # Keep custom C2f.forward methods: an inherited forward_split can bypass DME/WWTE computations.
    model.model[-1] = RockchipDetectOutputs(model.model[-1])
    return model


def export_rkopt_onnx(
    model: nn.Module, output: str | Path, imgsz=(640, 640), opset: int = 12, cafr_mode: str = "original"
) -> dict:
    """Write static batch-1 ONNX with a checked nine-output contract; conversion/CPU postprocessing stay external."""
    import onnx

    if len(imgsz) != 2 or any(size < 32 or size % 32 for size in imgsz):
        raise ValueError("imgsz must contain two positive multiples of 32.")
    if not 12 <= opset <= 19:
        raise ValueError("Use an opset from 12 to 19; Rockchip's reference export uses 12.")
    output = Path(output)
    if output.suffix.lower() != ".onnx":
        raise ValueError("output must end with .onnx.")
    prepared = prepare_rkopt_model(model, cafr_mode=cafr_mode)
    image = torch.zeros(1, 3, *imgsz)
    with torch.no_grad():
        predictions = prepared(image)
    names = [f"p{scale}_{kind}" for scale in (3, 4, 5) for kind in ("box", "class", "score_sum")]
    output.parent.mkdir(parents=True, exist_ok=True)
    options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(
        prepared, image, str(output), opset_version=opset, do_constant_folding=True,
        input_names=["images"], output_names=names, **options,
    )
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    if len(graph.graph.output) != 9:
        raise RuntimeError("RKOPT export must have nine outputs.")
    info = {
        "format": "rockchip_yolov8_raw_9_outputs",
        "input": {"name": "images", "shape": list(image.shape), "dtype": "float32", "range": [0, 1]},
        "outputs": [{"name": name, "shape": list(t.shape)} for name, t in zip(names, predictions)],
        "names": getattr(model, "names", {}),
        "reg_max": model.model[-1].reg_max,
        "opset": opset,
        "cafr_mode": cafr_mode,
        "rewritten_cafr_modules": [
            name for name, m in prepared.named_modules() if isinstance(m, ContentAwareUpsample4D)
        ],
        "postprocess": "External DFL, box decode, threshold filtering and NMS (RKNN Model Zoo YOLOv8 layout).",
    }
    for key in ("format", "names", "reg_max", "postprocess", "cafr_mode", "rewritten_cafr_modules"):
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, json.dumps(info[key], ensure_ascii=False)
    onnx.save(graph, str(output))
    return info
