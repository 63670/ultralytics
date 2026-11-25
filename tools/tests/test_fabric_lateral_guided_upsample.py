"""Verify directional guidance and its integration into the ribbon-attention detector."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import LateralGuidedUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
RIBBON = CONFIGS / "yolov8n-fabric-ribbon-attention.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-ribbon-guided-upsample.yaml"


class FabricLateralGuidedUpsampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_axes_are_independent_and_constant_fields_are_preserved(self):
        upsample = LateralGuidedUpsample(1, 1, guide_channels=1).double()
        coarse = torch.tensor([[[[0.0, 4.0], [8.0, 12.0]]]], dtype=torch.double)
        lateral = torch.randn(1, 1, 4, 4, dtype=torch.double)
        with torch.no_grad():
            base = F.interpolate(coarse, scale_factor=2, mode="nearest")
            torch.testing.assert_close(upsample([coarse, lateral]), base, atol=0, rtol=0)
            upsample.direction.bias[0] = torch.atanh(torch.tensor(0.5, dtype=torch.double))
            horizontal = upsample([coarse, lateral])
            torch.testing.assert_close(horizontal[0, 0, 0], torch.tensor([0.0, 0.25, 3.75, 4.0], dtype=torch.double))
            torch.testing.assert_close(horizontal[..., 0, :], horizontal[..., 1, :])

            upsample.direction.bias[1] = upsample.direction.bias[0]
            upsample.direction.bias[0] = 0
            vertical = upsample([coarse, lateral])
            torch.testing.assert_close(vertical[0, 0, :, 0], torch.tensor([0.0, 0.5, 7.5, 8.0], dtype=torch.double))
            torch.testing.assert_close(vertical[..., 0], vertical[..., 1])
            torch.testing.assert_close(upsample([torch.ones_like(coarse), lateral]), torch.ones_like(base))

    def test_lateral_input_changes_the_requested_high_resolution_cell(self):
        upsample = LateralGuidedUpsample(1, 1, guide_channels=1).eval()
        coarse = torch.tensor([[[[0.0, 4.0], [0.0, 4.0]]]])
        lateral = torch.zeros(1, 1, 4, 4)
        with torch.no_grad():
            upsample.lateral_guide.weight.fill_(1.0)
            upsample.direction.weight[0, 1, 1, 1] = 1.0
            before = upsample([coarse, lateral])
            lateral[..., 1, 1] = 1.0
            change = upsample([coarse, lateral]) - before
        self.assertGreater(change[..., 1, 1].item(), 0.0)
        self.assertEqual(change.count_nonzero().item(), 1)

    def test_combination_reuses_ribbon_weights_and_starts_from_its_predictions(self):
        torch.manual_seed(0)
        reference_cfg, target_cfg = YAML.load(RIBBON), YAML.load(COMBINATION)
        for index in (0, 3):
            target_cfg["head"][index] = reference_cfg["head"][index]
        self.assertEqual(target_cfg, reference_cfg)

        reference = DetectionModel(str(RIBBON), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        layers = [(i, layer.f) for i, layer in enumerate(model.model) if isinstance(layer, LateralGuidedUpsample)]
        self.assertEqual(layers, [(10, [-1, 6]), (13, [-1, 4])])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        with torch.no_grad():
            for index in (4, 6):
                reference.model[index].project.weight.normal_(0, 0.1)
            model.load(reference, verbose=False)
            image = torch.rand(1, 3, 128, 160)
            torch.testing.assert_close(model(image)[0], reference(image)[0], atol=0, rtol=0)

    def test_detection_loss_updates_both_gates_and_guidance_inputs(self):
        torch.manual_seed(0)
        model = DetectionModel(str(COMBINATION), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 128, 160),
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
                self.assertGreater(module.direction.weight.grad.abs().sum().item(), 0.0)
                for guide in (module.coarse_guide, module.lateral_guide):
                    gradient = guide.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
