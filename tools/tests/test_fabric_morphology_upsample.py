"""Verify per-location reconstruction support and its ribbon-detector integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import MorphologyAdaptiveUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
RIBBON = CONFIGS / "yolov8n-fabric-ribbon-attention.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-ribbon-morphology-upsample.yaml"


class FabricMorphologyUpsampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_point_local_and_context_support_have_distinct_reconstruction(self):
        module = MorphologyAdaptiveUpsample(1, 1, guide_channels=1)
        coarse = torch.zeros(1, 1, 3, 4)
        coarse[..., 1, 1] = 1.0
        lateral = torch.randn(1, 1, 6, 8)
        size = lateral.shape[-2:]
        nearest = F.interpolate(coarse, size=size, mode="nearest")
        local = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
        pooled = F.avg_pool2d(coarse, 3, stride=1, padding=1, count_include_pad=False)
        context = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        with torch.no_grad():
            torch.testing.assert_close(module([coarse, lateral]), nearest, atol=0, rtol=0)
            module.coarse_guide.weight.fill_(1.0)
            module.project.weight.fill_(1.0)
            module.support.weight.zero_()
            for index, expected in enumerate((nearest, local, context)):
                module.support.bias.fill_(-100)
                module.support.bias[index] = 100
                torch.testing.assert_close(module([coarse, lateral]), expected)
                torch.testing.assert_close(module([torch.ones_like(coarse), lateral]), torch.ones_like(nearest))
        self.assertNotEqual(local[..., 2, 2].item(), context[..., 2, 2].item())

    def test_different_neighborhood_shapes_with_same_center_select_different_support(self):
        module = MorphologyAdaptiveUpsample(1, 1, guide_channels=1)
        coarse = torch.zeros(1, 1, 5, 5)
        coarse[..., 2, 2] = 1.0
        narrow = torch.zeros(1, 1, 10, 10)
        narrow[..., 4, 2:7] = 1.0
        compact = torch.zeros_like(narrow)
        compact[..., 2:7, 2:7] = 1.0
        with torch.no_grad():
            module.coarse_guide.weight.fill_(1.0)
            module.lateral_guide.weight.fill_(1.0)
            module.project.weight.fill_(1.0)
            module.support.weight.zero_()
            module.support.bias.zero_()
            module.support.weight[1, 2, 1, 1] = 6.0
            module.support.weight[2, 3, 1, 1] = 6.0
            narrow_output = module([coarse, narrow])
            compact_output = module([coarse, compact])
        self.assertEqual(narrow[..., 4, 4].item(), compact[..., 4, 4].item())
        self.assertGreater((narrow_output - compact_output)[..., 4, 4].item(), 0.1)

    def test_combination_changes_only_upsampling_and_starts_from_ribbon_predictions(self):
        torch.manual_seed(0)
        reference_cfg, target_cfg = YAML.load(RIBBON), YAML.load(COMBINATION)
        for index in (0, 3):
            target_cfg["head"][index] = reference_cfg["head"][index]
        self.assertEqual(target_cfg, reference_cfg)

        reference = DetectionModel(str(RIBBON), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        layers = [(i, layer.f) for i, layer in enumerate(model.model) if isinstance(layer, MorphologyAdaptiveUpsample)]
        self.assertEqual(layers, [(10, [-1, 6]), (13, [-1, 4])])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        with torch.no_grad():
            for index in (4, 6):
                reference.model[index].project.weight.normal_(0, 0.1)
            model.load(reference, verbose=False)
            image = torch.rand(1, 3, 128, 160)
            torch.testing.assert_close(model(image)[0], reference(image)[0], atol=0, rtol=0)

    def test_detection_loss_updates_both_scale_selectors(self):
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
                self.assertGreater(module.project.weight.grad.abs().sum().item(), 0.0)
                for layer in (module.support, module.coarse_guide, module.lateral_guide):
                    gradient = layer.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
