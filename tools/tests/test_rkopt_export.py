"""Check Rockchip output parity, real ONNX execution, and preservation of custom C2f forwards."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tools.export_rkopt_onnx import main as export_main
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.modules import C2fDirectional, ContentAwareUpsample, Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.export.rkopt import (
    ContentAwareUpsample4D,
    RockchipDetectOutputs,
    export_rkopt_onnx,
    prepare_rkopt_model,
)
from ultralytics.utils.tal import dist2bbox, make_anchors

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "ultralytics/cfg/models/v8"


class RKOptTests(unittest.TestCase):
    """Use synthetic models and inputs; no downloaded weights, calibration data, or NPU are required."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_external_dfl_decode_matches_original_head(self):
        head = Detect(nc=3, ch=(16, 32, 64)).eval()
        head.stride = torch.tensor([8.0, 16.0, 32.0])
        features = [torch.randn(2, c, h, h) for c, h in ((16, 8), (32, 4), (64, 2))]
        with torch.no_grad():
            expected = head(features)[0]
            raw = RockchipDetectOutputs(head)(features)
            logits = torch.cat([raw[i].flatten(2) for i in (0, 3, 6)], dim=2)
            probabilities = logits.reshape(2, 4, head.reg_max, -1).softmax(dim=2)
            bins = torch.arange(head.reg_max).reshape(1, 1, head.reg_max, 1)
            distances = (probabilities * bins).sum(dim=2)
            anchors, strides = make_anchors(features, head.stride, 0.5)
            boxes = dist2bbox(distances, anchors.T.unsqueeze(0), xywh=True, dim=1) * strides.T
            scores = torch.cat([raw[i].flatten(2) for i in (1, 4, 7)], dim=2)
            torch.testing.assert_close(torch.cat((boxes, scores), dim=1), expected)
            for index in (0, 3, 6):
                torch.testing.assert_close(raw[index + 2], raw[index + 1].sum(dim=1, keepdim=True).clamp(0, 1))

    def test_rkopt_onnx_runs_with_custom_modules_and_no_head_dfl(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("baseline", "directional-carafe", "axis-residual"):
                with self.subTest(model=suffix):
                    model = DetectionModel(str(CONFIGS / f"yolov8n-fabric-{suffix}.yaml"), verbose=False).eval()
                    original_head = model.model[-1]
                    image = torch.rand(1, 3, 64, 64)
                    with torch.no_grad():
                        expected = [t.numpy() for t in prepare_rkopt_model(model)(image)]
                    output = Path(directory) / f"{suffix}.onnx"
                    info = export_rkopt_onnx(model, output, imgsz=(64, 64))
                    self.assertIs(model.model[-1], original_head)
                    self.assertIs(type(model.model[-1]), Detect)
                    self.assertEqual(len(info["outputs"]), 9)
                    graph = onnx.load(str(output))
                    self.assertFalse(any("dfl" in item.name for item in graph.graph.initializer))
                    self.assertFalse(any(node.op_type == "NonMaxSuppression" for node in graph.graph.node))
                    if suffix == "directional-carafe":
                        # CAFR's Softmax stays in the backbone/neck; only detection-head DFL is removed.
                        self.assertTrue(any(node.op_type == "Softmax" for node in graph.graph.node))
                    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                        None, {"images": image.numpy()}
                    )
                    for expected_tensor, actual_tensor in zip(expected, actual):
                        np.testing.assert_allclose(actual_tensor, expected_tensor, atol=2e-5, rtol=1e-4)

    def test_standard_export_retains_directional_forward(self):
        api = YOLO(str(CONFIGS / "yolov8n-fabric-directional.yaml"))
        model = api.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            model.pt_path = str(Path(directory) / "directional.pt")
            image = torch.rand(1, 3, 64, 64)
            with torch.no_grad():
                expected = model(image)[0].numpy()
            exporter = Exporter(overrides={"format": "onnx", "imgsz": 64, "batch": 1, "opset": 12, "simplify": False})
            output = exporter(model=model)
            for module in exporter.model.modules():
                if isinstance(module, C2fDirectional):
                    self.assertIs(module.forward.__func__, C2fDirectional.forward)
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                None, {"images": image.numpy()}
            )[0]
            np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)

    def test_cli_loads_custom_checkpoint_and_writes_640_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "custom.pt"
            output = Path(directory) / "custom-rkopt.onnx"
            YOLO(str(CONFIGS / "yolov8n-fabric-directional-carafe.yaml")).save(str(weights))
            with patch("sys.argv", ["export_rkopt_onnx.py", "--weights", str(weights), "--output", str(output)]):
                export_main()
            info = json.loads(output.with_suffix(".rkopt.json").read_text(encoding="utf-8"))
            self.assertEqual(info["input"]["shape"], [1, 3, 640, 640])
            self.assertEqual(info["outputs"][0]["shape"], [1, 64, 80, 80])
            self.assertEqual(info["outputs"][1]["shape"], [1, 3, 80, 80])
            self.assertEqual(info["outputs"][8]["shape"], [1, 1, 20, 20])
            self.assertEqual(len(info["weights_sha256"]), 64)
            onnx.checker.check_model(onnx.load(str(output)))

    def test_streaming_reassembly_matches_outputs_and_gradients_at_borders(self):
        for kernel in (1, 3, 5):
            for shape in ((1, 4, 1, 2), (2, 4, 5, 7)):
                with self.subTest(kernel=kernel, shape=shape):
                    source = ContentAwareUpsample(4, kernel).double().eval()
                    streaming = ContentAwareUpsample4D(source)
                    image = torch.randn(*shape, dtype=torch.double, requires_grad=True)
                    expected = source(image)
                    actual = streaming(image)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                    upstream = torch.randn_like(expected)
                    parameters = (image, *source.parameters())
                    reference_gradients = torch.autograd.grad((expected * upstream).sum(), parameters)
                    actual_gradients = torch.autograd.grad((actual * upstream).sum(), parameters)
                    for reference, gradient in zip(reference_gradients, actual_gradients):
                        torch.testing.assert_close(gradient, reference, rtol=1e-9, atol=1e-9)

    def test_streaming_onnx_matches_original_model_without_large_rank_features(self):
        model = DetectionModel(str(CONFIGS / "yolov8n-fabric-directional-carafe.yaml"), verbose=False).eval()
        image = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            expected = [tensor.numpy() for tensor in prepare_rkopt_model(model)(image)]
            streamed = prepare_rkopt_model(model, cafr_mode="streaming")
            for reference, tensor in zip(expected, streamed(image)):
                np.testing.assert_allclose(tensor.numpy(), reference, atol=2e-5, rtol=1e-4)
        self.assertIs(type(model.model[10]), ContentAwareUpsample)
        self.assertIs(type(model.model[13]), ContentAwareUpsample)
        self.assertIs(type(streamed.model[10]), ContentAwareUpsample4D)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "streaming.onnx"
            info = export_rkopt_onnx(model, output, imgsz=(64, 96), cafr_mode="streaming")
            self.assertEqual(info["rewritten_cafr_modules"], ["model.10", "model.13"])
            graph = onnx.shape_inference.infer_shapes(onnx.load(str(output))).graph
            ranks = {item.name: len(item.type.tensor_type.shape.dim) for item in graph.value_info}
            for node in graph.node:
                if "/model.10/" in node.name or "/model.13/" in node.name:
                    for name in node.output:
                        if name in ranks:
                            self.assertLessEqual(ranks[name], 4, f"High-rank CAFR tensor: {node.name}")
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                None, {"images": image.numpy()}
            )
            for reference, tensor in zip(expected, actual):
                np.testing.assert_allclose(tensor, reference, atol=2e-5, rtol=1e-4)

    def test_streaming_cli_loads_existing_weights_and_records_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "custom.pt"
            YOLO(str(CONFIGS / "yolov8n-fabric-directional-carafe.yaml")).save(str(weights))
            with patch("sys.argv", ["export_rkopt_onnx.py", "--weights", str(weights), "--cafr-mode", "streaming"]):
                export_main()
            output = weights.with_name("custom-rkopt-cafr4d.onnx")
            info = json.loads(output.with_suffix(".rkopt.json").read_text(encoding="utf-8"))
            self.assertEqual(info["cafr_mode"], "streaming")
            self.assertEqual(info["rewritten_cafr_modules"], ["model.10", "model.13"])
            self.assertEqual(info["input"]["shape"], [1, 3, 640, 640])
            self.assertEqual(len(onnx.load(str(output)).graph.output), 9)
            image = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
            with torch.no_grad():
                original = prepare_rkopt_model(YOLO(str(weights)).model)(torch.from_numpy(image))
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"images": image})
            for reference, tensor in zip(original, actual):
                np.testing.assert_allclose(tensor, reference.numpy(), atol=2e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
