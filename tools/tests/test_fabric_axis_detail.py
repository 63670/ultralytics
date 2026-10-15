"""Validate the experimental fabric detector with real PyTorch layers and detection loss."""
from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2f, C2fDetailFusion, C2fPartialAxis
from ultralytics.nn.modules.block import PartialAxisBottleneck
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ultralytics/cfg/models/v8/yolov8n-fabric-axis-detail.yaml"
BASELINE = ROOT / "ultralytics/cfg/models/v8/yolov8n-fabric-baseline.yaml"


class FabricAxisDetailTests(unittest.TestCase):
    """Exercise global context, shallow-detail gradients, detector loss, and export-time split behavior."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_axis_context_connects_distant_pixels(self):
        for enabled in (False, True):
            block = PartialAxisBottleneck(16, context=enabled).eval()
            image = torch.randn(1, 16, 12, 24, requires_grad=True)
            gradient = torch.autograd.grad(block(image)[0, :, 5, 3].sum(), image)[0]
            distant = gradient[0, :4, 5, 20].abs().sum().item()
            if enabled:
                self.assertGreater(distant, 0)
            else:
                self.assertEqual(distant, 0)

    def test_parser_strides_parameter_budget_and_weight_transfer(self):
        model = DetectionModel(str(CONFIG), verbose=False)
        baseline = DetectionModel(str(BASELINE), verbose=False)
        self.assertIsInstance(model.model[4], C2fPartialAxis)
        self.assertIsInstance(model.model[15], C2fDetailFusion)
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        self.assertLess(sum(p.numel() for p in model.parameters()), sum(p.numel() for p in baseline.parameters()))
        with torch.no_grad():
            baseline.model[4].cv1.conv.weight.fill_(0.125)
            baseline.model[15].cv2.conv.weight.fill_(0.25)
        model.load(baseline, verbose=False)
        torch.testing.assert_close(model.model[4].cv1.conv.weight, baseline.model[4].cv1.conv.weight)
        torch.testing.assert_close(model.model[15].cv2.conv.weight, baseline.model[15].cv2.conv.weight)

    def test_detection_loss_updates_both_new_paths(self):
        model = DetectionModel(str(CONFIG), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 160, 160),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        tracked = [model.model[4].m[0].row.weight, model.model[15].detail[0].conv.weight]
        before = [p.detach().clone() for p in tracked]
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        for parameter in tracked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        for parameter, previous in zip(tracked, before):
            self.assertFalse(torch.equal(parameter, previous))

    def test_inference_and_export_split_preserve_detail_fusion(self):
        model = DetectionModel(str(CONFIG), verbose=False).eval()
        image = torch.rand(1, 3, 640, 640)
        with torch.no_grad():
            expected = model(image)[0]
            self.assertEqual(tuple(expected.shape), (1, 7, 8400))
            for module in model.modules():
                if isinstance(module, C2f):
                    module.forward = module.forward_split
            actual = model(image)[0]
            torch.testing.assert_close(actual, expected)
            rectangular = model(torch.rand(1, 3, 256, 320))[0]
            self.assertEqual(tuple(rectangular.shape), (1, 7, 1680))
            fused = copy.deepcopy(model).fuse(verbose=False)
            torch.testing.assert_close(fused(image)[0], actual, atol=1e-3, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
