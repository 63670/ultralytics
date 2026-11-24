"""Check per-image period selection, repeat subtraction and P2-to-P3 fusion."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fPeriodicResidual
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-periodic-residual-phase.yaml"


class FabricPeriodicResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_periods_are_selected_per_image_and_aligned_texture_cancels_at_borders(self):
        position = torch.arange(80)
        fabric = torch.stack((position % 4, position % 5)).float().reshape(2, 1, 1, -1)
        weights = C2fPeriodicResidual._period_weights(fabric)
        self.assertEqual(weights[:, :, 0, 0].argmax(dim=1).tolist(), [2, 3])
        self.assertTrue((weights.sum(dim=1) > 0.5).all())
        self.assertEqual(weights.count_nonzero().item(), 2)
        residual = C2fPeriodicResidual._repeat_residual(fabric, weights)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0, rtol=0)

        fabric[..., 40] += 1.0
        weights = C2fPeriodicResidual._period_weights(fabric)
        residual = C2fPeriodicResidual._repeat_residual(fabric, weights)
        self.assertTrue((residual[..., 40] > 0.5).all())
        self.assertEqual(residual[..., :34].count_nonzero().item(), 0)

    def test_smooth_textures_and_insufficient_overlap_are_neutral(self):
        for image in (torch.ones(1, 2, 3, 80), torch.linspace(0, 1, 80).view(1, 1, 1, -1), torch.rand(2, 3, 5, 3)):
            weights = C2fPeriodicResidual._period_weights(image)
            self.assertEqual(weights.count_nonzero().item(), 0)
            residual = C2fPeriodicResidual._repeat_residual(image, weights)
            torch.testing.assert_close(residual, torch.zeros_like(image), atol=0, rtol=0)

    def test_config_preserves_saru_and_initial_predictions(self):
        torch.manual_seed(0)
        source_cfg = YAML.load(PHASE)
        target_cfg = YAML.load(COMBINATION)
        target_cfg["head"][5] = source_cfg["head"][5]
        self.assertEqual(target_cfg, source_cfg)

        source = DetectionModel(str(PHASE), verbose=False).eval()
        target = DetectionModel(str(COMBINATION), verbose=False).eval()
        self.assertEqual([i for i, layer in enumerate(target.model) if isinstance(layer, C2fPeriodicResidual)], [15])
        block = target.model[15]
        self.assertEqual(block.f, [-1, 2])
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            for index in (10, 13):
                source.model[index].correction.weight.normal_(0, 0.1)
            target.load(source, verbose=False)
            torch.testing.assert_close(target(image)[0], source(image)[0], atol=0, rtol=0)
            self.assertEqual(target.stride.tolist(), [8, 16, 32])
            block.project.weight.normal_(0, 0.1)
            expected = target(image)[0]
            block.forward = block.forward_split
            torch.testing.assert_close(target(image)[0], expected)

    def test_detection_loss_trains_the_shallow_residual_path(self):
        torch.manual_seed(0)
        model = DetectionModel(str(COMBINATION), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        image = torch.rand(2, 3, 128, 160) * 0.2
        image += (torch.arange(160) % 16 >= 8).float().view(1, 1, 1, -1) * 0.8
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        block = model.model[15]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            self.assertGreater(block.project.weight.grad.abs().sum().item(), 0)
            for layer in (block.compress, block.downsample):
                gradient = layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
