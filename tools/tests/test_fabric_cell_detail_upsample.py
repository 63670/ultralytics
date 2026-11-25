"""Verify mean-preserving detail reconstruction and its ribbon-detector integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ultralytics.nn.modules import CellDetailUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
RIBBON = CONFIGS / "yolov8n-fabric-ribbon-attention.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-ribbon-cell-detail.yaml"


class FabricCellDetailUpsampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_cell_mean_can_restore_different_subpixel_shapes(self):
        module = CellDetailUpsample(1, 1, detail_channels=1).double()
        coarse = torch.full((1, 1, 1, 2), 3.0, dtype=torch.double)
        lateral = torch.tensor([[[[2.0, 2.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]], dtype=torch.double)
        with torch.no_grad():
            module.lateral_detail.weight.fill_(1.0)
            module.project.weight.fill_(1.0)
            module.gate.weight.zero_()
            module.gate.bias.zero_()
            output = module([coarse, lateral])
        expected = torch.tensor([[[[3.5, 3.5, 4.5, 2.5], [2.5, 2.5, 2.5, 2.5]]]], dtype=torch.double)
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(F.avg_pool2d(output, 2, 2), coarse, atol=0, rtol=0)

    def test_learned_gates_preserve_the_mean_and_reject_constant_cell_details(self):
        torch.manual_seed(0)
        module = CellDetailUpsample(5, 3, detail_channels=4).double()
        coarse = torch.rand(2, 5, 5, 7, dtype=torch.double)
        lateral = torch.rand(2, 3, 10, 14, dtype=torch.double)
        with torch.no_grad():
            module.project.weight.normal_(0, 0.5)
            output = module([coarse, lateral])
            base = F.interpolate(coarse, scale_factor=2, mode="nearest")
            self.assertGreater((output - base).abs().max().item(), 0.01)
            torch.testing.assert_close(F.avg_pool2d(output, 2, 2), coarse)
            constant_cells = F.interpolate(F.avg_pool2d(lateral, 2, 2), scale_factor=2, mode="nearest")
            torch.testing.assert_close(module([coarse, constant_cells]), base)

    def test_combination_changes_only_upsampling_and_starts_from_ribbon_predictions(self):
        torch.manual_seed(0)
        reference_cfg, target_cfg = YAML.load(RIBBON), YAML.load(COMBINATION)
        for index in (0, 3):
            target_cfg["head"][index] = reference_cfg["head"][index]
        self.assertEqual(target_cfg, reference_cfg)

        reference = DetectionModel(str(RIBBON), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        layers = [(i, layer.f) for i, layer in enumerate(model.model) if isinstance(layer, CellDetailUpsample)]
        self.assertEqual(layers, [(10, [-1, 6]), (13, [-1, 4])])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        with torch.no_grad():
            for index in (4, 6):
                reference.model[index].project.weight.normal_(0, 0.1)
            model.load(reference, verbose=False)
            image = torch.rand(1, 3, 128, 160)
            torch.testing.assert_close(model(image)[0], reference(image)[0], atol=0, rtol=0)

    def test_detection_loss_trains_both_detail_projections(self):
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
                for layer in (module.gate, module.coarse_guide, module.lateral_detail):
                    gradient = layer.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
