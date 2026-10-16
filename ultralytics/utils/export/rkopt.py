# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Export YOLOv8-style detectors with the nine-output layout used by Rockchip's model zoo."""
from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.modules import Detect


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


def prepare_rkopt_model(model: nn.Module) -> nn.Module:
    """Prepare a private FP32 CPU copy; preserve custom backbone forwards and the original model."""
    head = model.model[-1]
    if type(head) is not Detect or head.end2end or head.nl != 3:
        raise ValueError("RKOPT requires a three-scale, non-end-to-end Detect head (no segment/pose/OBB subclasses).")
    model = deepcopy(model).cpu().float().eval()
    model.requires_grad_(False)
    model.fuse(verbose=False)
    # Keep custom C2f.forward methods: an inherited forward_split can bypass DME/WWTE computations.
    model.model[-1] = RockchipDetectOutputs(model.model[-1])
    return model


def export_rkopt_onnx(model: nn.Module, output: str | Path, imgsz=(640, 640), opset: int = 12) -> dict:
    """Write static batch-1 ONNX with a checked nine-output contract; conversion/CPU postprocessing stay external."""
    import onnx

    if len(imgsz) != 2 or any(size < 32 or size % 32 for size in imgsz):
        raise ValueError("imgsz must contain two positive multiples of 32.")
    if not 12 <= opset <= 19:
        raise ValueError("Use an opset from 12 to 19; Rockchip's reference export uses 12.")
    output = Path(output)
    if output.suffix.lower() != ".onnx":
        raise ValueError("output must end with .onnx.")
    prepared = prepare_rkopt_model(model)
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
        "postprocess": "External DFL, box decode, threshold filtering and NMS (RKNN Model Zoo YOLOv8 layout).",
    }
    for key in ("format", "names", "reg_max", "postprocess"):
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, json.dumps(info[key], ensure_ascii=False)
    onnx.save(graph, str(output))
    return info
