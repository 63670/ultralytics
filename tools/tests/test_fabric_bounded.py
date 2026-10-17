"""Check architecture constraints, first-step gradients, and real RKOPT export for the bounded candidates."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from tools.export_rkopt_onnx import main as export_main
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, C2fAxisBounded, GatedInterpolationUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.export.rkopt import export_rkopt_onnx, prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
CONFIG = CONFIGS / "yolov8n-fabric-axis-bounded-interp.yaml"
CANDIDATES = ("yolov8n-fabric-axis-interp.yaml", "yolov8n-fabric-axis-bounded-cafr-lite.yaml", CONFIG.name)


class BoundedFabricTests(unittest.TestCase):
    """Use synthetic features and targets; real-data accuracy and NPU speed require separate experiments."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_interpolation_initial_mapping_is_seed_independent(self):
        image = torch.randn(2, 8, 5, 7)
        expected = 0.95 * F.interpolate(image, scale_factor=2, mode="nearest") + 0.05 * F.interpolate(
            image, scale_factor=2, mode="bilinear", align_corners=False
        )
        outputs = []
        for seed in [0, 17, 123]:
            torch.manual_seed(seed)
            module = GatedInterpolationUpsample(8)
            outputs.append(module(image))
            torch.testing.assert_close(outputs[-1], expected, atol=3e-7, rtol=1e-6)
        for value in outputs[1:]:
            torch.testing.assert_close(value, outputs[0], atol=0, rtol=0)

    def test_interpolation_is_convex_and_preserves_constant_features(self):
        module = GatedInterpolationUpsample(8).double()
        image = torch.randn(2, 8, 5, 7, dtype=torch.double)
        nearest = F.interpolate(image, scale_factor=2, mode="nearest")
        linear = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
        with torch.no_grad():
            module.gate_conv.weight.normal_(0, 20)
            module.gate_conv.bias.fill_(10)
        actual = module(image)
        self.assertTrue((actual >= torch.minimum(nearest, linear) - 1e-12).all())
        self.assertTrue((actual <= torch.maximum(nearest, linear) + 1e-12).all())
        constant = torch.randn(2, 8, 1, 1, dtype=torch.double).expand(2, 8, 5, 7)
        torch.testing.assert_close(module(constant), F.interpolate(constant, scale_factor=2), atol=1e-12, rtol=0)

    def test_axial_initial_mapping_preserves_loaded_c2f(self):
        source = C2f(16, 32, n=2, shortcut=True).eval()
        module = C2fAxisBounded(16, 32, n=2, shortcut=True, partial=0.5).eval()
        missing, unexpected = module.load_state_dict(source.state_dict(), strict=False)
        self.assertEqual(set(missing), {"row.weight", "column.weight"})
        self.assertFalse(unexpected)
        image = torch.randn(2, 16, 9, 13)
        torch.testing.assert_close(module(image), source(image), atol=0, rtol=0)
        with torch.no_grad():
            module.row.weight.normal_()
            module.column.weight.normal_()
        torch.testing.assert_close(module(image), module.forward_split(image))

    def test_axial_modulation_is_bounded_and_constant_neutral(self):
        module = C2fAxisBounded(16, 32, partial=0.5).double()
        with torch.no_grad():
            module.row.weight.normal_(0, 100)
            module.column.weight.normal_(0, 100)
        image = torch.randn(2, module.active, 9, 13, dtype=torch.double)
        self.assertTrue(((module._modulate(image) - image).abs() <= 0.25 * image.abs() + 1e-12).all())
        constant = torch.randn(2, module.active, 1, 1, dtype=torch.double).expand(2, module.active, 9, 13)
        torch.testing.assert_close(module._modulate(constant), constant, atol=1e-12, rtol=0)

    def test_all_candidates_retain_baseline_parameter_shapes_and_detection_scales(self):
        data = YAML.load(CONFIGS / "yolov8n-fabric-axis-residual.yaml")
        for layer in data["backbone"]:
            if layer[2] == "C2fAxisResidual":
                layer[2], layer[3] = "C2f", layer[3][:2]
        source = DetectionModel(data, verbose=False).eval()
        for filename in CANDIDATES:
            with self.subTest(model=filename):
                model = DetectionModel(str(CONFIGS / filename), verbose=False).eval()
                model.load(source, verbose=False)
                for name, tensor in source.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                with torch.no_grad():
                    self.assertEqual(tuple(model(torch.rand(1, 3, 128, 160))[0].shape), (1, 7, 420))

    def test_real_detection_loss_updates_new_filters_and_gates_on_first_step(self):
        model = DetectionModel(str(CONFIG), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 160, 160),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for index in (4, 6):
            for weight in (model.model[index].row.weight, model.model[index].column.weight):
                self.assertTrue(torch.isfinite(weight.grad).all())
                self.assertGreater(weight.grad.abs().sum().item(), 0)
        for index in (10, 13):
            weight = model.model[index].gate_conv.weight
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)

    def test_nonzero_corrections_survive_split_fusion_and_actual_onnx_execution(self):
        model = DetectionModel(str(CONFIG), verbose=False).eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, C2fAxisBounded):
                    module.row.weight.normal_(0, 0.2)
                    module.column.weight.normal_(0, 0.2)
                elif isinstance(module, GatedInterpolationUpsample):
                    module.gate_conv.weight.normal_(0, 0.1)
                    module.gate_conv.bias.fill_(-1.5)
        image = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            expected_prediction = model(image)[0]
            expected = [tensor.numpy() for tensor in prepare_rkopt_model(model)(image)]
            for module in model.modules():
                if isinstance(module, C2f):
                    module.forward = module.forward_split
            torch.testing.assert_close(model(image)[0], expected_prediction)
            fused = copy.deepcopy(model).fuse(verbose=False)
            torch.testing.assert_close(fused(image)[0], expected_prediction, atol=1e-3, rtol=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bounded.onnx"
            info = export_rkopt_onnx(model, output, imgsz=(64, 96))
            self.assertEqual(info["gated_interpolation_modules"], ["model.10", "model.13"])
            self.assertEqual(info["native_cafr_lite_modules"], [])
            graph = onnx.load(str(output)).graph
            types = [node.op_type for node in graph.node]
            self.assertNotIn("Softmax", types)
            self.assertNotIn("GridSample", types)
            modes = [
                a.s.decode() for node in graph.node if node.op_type == "Resize" for a in node.attribute if a.name == "mode"
            ]
            self.assertEqual(modes.count("linear"), 2)
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                None, {"images": image.numpy()}
            )
            for reference, tensor in zip(expected, actual):
                np.testing.assert_allclose(tensor, reference, atol=2e-5, rtol=1e-4)

    def test_export_cli_loads_checkpoint_with_new_module_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "bounded.pt"
            output = Path(directory) / "bounded.onnx"
            model = YOLO(str(CONFIG))
            with torch.no_grad():
                for module in model.model.modules():
                    if isinstance(module, C2fAxisBounded):
                        module.row.weight.fill_(0.1)
                        module.column.weight.fill_(-0.1)
            model.save(str(weights))
            loaded = YOLO(str(weights)).model
            self.assertIsInstance(loaded.model[4], C2fAxisBounded)
            self.assertIsInstance(loaded.model[10], GatedInterpolationUpsample)
            args = ["export_rkopt_onnx.py", "--weights", str(weights), "--output", str(output), "--imgsz", "64"]
            with patch("sys.argv", args):
                export_main()
            graph = onnx.load(str(output)).graph
            modes = [a.s.decode() for n in graph.node if n.op_type == "Resize" for a in n.attribute if a.name == "mode"]
            self.assertEqual(modes.count("linear"), 2)
            self.assertTrue(output.with_suffix(".rkopt.json").is_file())


if __name__ == "__main__":
    unittest.main()
