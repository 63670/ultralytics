"""Exercise bright and dark contrast and its use in the fabric detector stem."""

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import LuminanceStem
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-luminance-phase.yaml"


class FabricLuminanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_faint_bright_and_dark_spots_have_separate_polarities(self):
        image = torch.full((1, 3, 96, 128), 0.5)
        image[..., 48:50, 40:42] += 0.04
        image[..., 48:50, 80:82] -= 0.04
        stem = LuminanceStem(3, 16)
        contrast = stem._contrast(image)
        for index in (0, 2):
            self.assertGreater(contrast[0, index, 24, 20].item(), 0.25)
            self.assertGreater(contrast[0, index + 1, 24, 40].item(), 0.25)
            self.assertEqual(contrast[0, index + 1, 24, 20].item(), 0.0)
            self.assertEqual(contrast[0, index, 24, 40].item(), 0.0)
        torch.testing.assert_close(contrast, stem._contrast(image + 0.1), atol=1e-4, rtol=0)
        torch.testing.assert_close(contrast, LuminanceStem(1, 16)._contrast(image[:, :1]), atol=1e-4, rtol=0)

    def test_large_neighborhood_can_detect_a_wide_brightness_change(self):
        image = torch.full((1, 1, 96, 128), 0.5)
        image[..., 32:64, 48:80] += 0.04
        contrast = LuminanceStem(1, 16)._contrast(image)
        self.assertAlmostEqual(contrast[0, 0, 24, 32].item(), 0.0, places=5)
        self.assertGreater(contrast[0, 2, 24, 32].item(), 1.5)

    def test_flat_ramp_and_single_step_do_not_create_bright_or_dark_regions(self):
        stem = LuminanceStem(1, 16)
        ramp = torch.linspace(0.2, 0.7, 96).view(1, 1, 1, -1).expand(1, 1, 96, 96)
        step = (torch.arange(96).view(1, 1, 1, -1) >= 48).expand_as(ramp) * 0.4 + 0.3
        for image in (torch.full_like(ramp, 0.4), ramp, step, step.transpose(-2, -1)):
            contrast = stem._contrast(image)
            torch.testing.assert_close(contrast, torch.zeros_like(contrast), atol=2e-5, rtol=0)

    def test_config_preserves_saru_weights_and_luminance_survives_fusion(self):
        torch.manual_seed(0)
        source_cfg = YAML.load(PHASE)
        target_cfg = YAML.load(COMBINATION)
        target_cfg["backbone"][0] = source_cfg["backbone"][0]
        self.assertEqual(target_cfg, source_cfg)

        reference = DetectionModel(str(PHASE), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        self.assertEqual([i for i, layer in enumerate(model.model) if isinstance(layer, LuminanceStem)], [0])
        image = torch.rand(1, 3, 96, 128)
        with torch.no_grad():
            model.load(reference, verbose=False)
            baseline = reference(image)[0]
            torch.testing.assert_close(model(image)[0], baseline, rtol=0, atol=0)
            self.assertEqual(model.stride.tolist(), [8, 16, 32])
            model.model[0].project.weight.normal_(0, 0.1)
            expected = model(image)[0]
            self.assertGreater((expected - baseline).abs().max().item(), 0.0)
            fused = copy.deepcopy(model).fuse(verbose=False)
            torch.testing.assert_close(fused(image)[0], expected, atol=1e-3, rtol=1e-4)
            torch.testing.assert_close(fused.model[0](image), model.model[0](image))

    def test_detection_loss_updates_the_zero_initialized_contrast_projection(self):
        torch.manual_seed(0)
        model = DetectionModel(str(COMBINATION), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 96, 128),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        gradient = model.model[0].project.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0.0)
