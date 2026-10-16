"""Validate the low-channel CAFR replacement before any real-data training or board benchmark."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F

from ultralytics.nn.modules import ContentAwareUpsampleLite
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.export.rkopt import export_rkopt_onnx, prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
CONFIG = CONFIGS / "yolov8n-fabric-axis-cafr-lite.yaml"


class CAFRLiteTests(unittest.TestCase):
    """Check compact aggregation, exact zero-gate initialization, trainability and actual ONNX execution."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_compact_reassembly_matches_unfold_reference(self):
        module = ContentAwareUpsampleLite(32, 3, 4).double().eval()
        module.gain.data.fill_(0.3)
        image = torch.randn(2, 32, 3, 5, dtype=torch.double, requires_grad=True)
        actual = module(image)
        latent = module.compress(image)
        weights = F.pixel_shuffle(module.kernel_encoder(latent), 2).softmax(dim=1)
        patches = F.unfold(latent, 3, padding=1).reshape(2, module.hidden * 9, 3, 5)
        patches = F.interpolate(patches, scale_factor=2, mode="nearest").reshape(2, module.hidden, 9, 6, 10)
        refined = (patches * weights.unsqueeze(1)).sum(dim=2)
        expected = F.interpolate(image, scale_factor=2, mode="nearest") + module.gain * module.project(refined)
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        actual_gradient = torch.autograd.grad(actual.square().sum(), image)[0]
        expected_gradient = torch.autograd.grad(expected.square().sum(), image)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-9, rtol=1e-9)

    def test_model_starts_as_axis_residual_and_transfers_all_existing_weights(self):
        source = DetectionModel(str(CONFIGS / "yolov8n-fabric-axis-residual.yaml"), verbose=False).eval()
        model = DetectionModel(str(CONFIG), verbose=False).eval()
        with torch.no_grad():
            source.model[4].gamma.fill_(0.2)
            source.model[6].gamma.fill_(0.2)
        model.load(source, verbose=False)
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)
        self.assertEqual([model.model[i].hidden for i in (10, 13)], [16, 8])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            torch.testing.assert_close(model(image)[0], source(image)[0], atol=0, rtol=0)

    def test_real_detection_loss_opens_gates_then_updates_kernel_predictors(self):
        model = DetectionModel(str(CONFIG), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 160, 160),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            for index in (10, 13):
                module = model.model[index]
                self.assertTrue(torch.isfinite(module.gain.grad).all())
                self.assertGreater(module.gain.grad.abs().sum().item(), 0)
                gradient = module.kernel_encoder.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                if step == 0:
                    self.assertEqual(gradient.abs().sum().item(), 0)
                else:
                    self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()

    def test_nonzero_residuals_are_exported_and_execute_in_onnx(self):
        model = DetectionModel(str(CONFIG), verbose=False).eval()
        with torch.no_grad():
            for index in (10, 13):
                model.model[index].gain.fill_(0.2)
        image = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            expected = [tensor.numpy() for tensor in prepare_rkopt_model(model)(image)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lite.onnx"
            info = export_rkopt_onnx(model, output, imgsz=(64, 96))
            self.assertEqual(info["native_cafr_lite_modules"], ["model.10", "model.13"])
            graph = onnx.load(str(output)).graph
            self.assertEqual(sum(node.op_type == "Softmax" for node in graph.node), 2)
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                None, {"images": image.numpy()}
            )
            for reference, tensor in zip(expected, actual):
                np.testing.assert_allclose(tensor, reference, atol=2e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
