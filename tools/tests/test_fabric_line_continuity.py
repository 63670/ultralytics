"""Check line support on controlled features and in the SARU detector."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fLineContinuity
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-line-continuity-phase.yaml"


class FabricLineContinuityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_line_support_bridges_a_gap_and_preserves_an_endpoint(self):
        evidence = torch.zeros(1, 1, 7, 25)
        evidence[..., 3, 2:23] = 1.0
        evidence[..., 3, 12] = 0.0
        support = C2fLineContinuity._line_support(evidence)
        self.assertEqual(support[..., 3, 12].item(), 1.0)
        self.assertEqual(support[..., 3, 2].item(), 1.0)
        self.assertEqual(support[..., 3, 1].item(), 0.0)

        evidence.zero_()
        evidence[..., 3, [2, 3, 5]] = 1.0
        support = C2fLineContinuity._line_support(evidence)
        self.assertEqual(support[..., 3, 2].item(), 0.5)

    def test_flanks_suppress_parallel_texture_and_edges_do_not_wrap(self):
        flat = torch.ones(1, 1, 7, 25)
        self.assertEqual(C2fLineContinuity._line_support(flat).count_nonzero().item(), 0)

        repeated = torch.zeros_like(flat)
        repeated[..., 1::2, :] = 1.0
        self.assertEqual(C2fLineContinuity._line_support(repeated)[..., 3, :].count_nonzero().item(), 0)

        sparse = torch.zeros_like(flat)
        sparse[..., 3, [0, 18, 22, 24]] = 1.0
        self.assertEqual(C2fLineContinuity._line_support(sparse)[..., 3, 0].item(), 0.0)

    def test_both_directions_receive_support(self):
        block = C2fLineContinuity(2, 2, n=0, channels_per_direction=1).eval()
        with torch.no_grad():
            identity = torch.eye(2).reshape(2, 2, 1, 1)
            block.evidence.weight.copy_(identity)
            block.project.weight.copy_(identity)
            lines = torch.full((1, 2, 21, 25), -3.0)
            lines[:, 0, 10, :] = 3.0
            lines[:, 1, :, 12] = 3.0
            correction = block._add_continuity(lines) - lines

        torch.testing.assert_close(correction[:, :, 10, 12], torch.ones(1, 2), rtol=0, atol=0)
        self.assertEqual(correction[:, 0, 0].count_nonzero().item(), 0)
        self.assertEqual(correction[:, 1, :, 0].count_nonzero().item(), 0)

    def test_config_keeps_saru_paths_and_zero_initial_predictions(self):
        torch.manual_seed(0)
        phase_cfg = YAML.load(PHASE)
        combo_cfg = YAML.load(COMBINATION)
        for index in (2, 5):
            combo_cfg["head"][index] = phase_cfg["head"][index]
        self.assertEqual(combo_cfg, phase_cfg)

        reference = DetectionModel(str(PHASE), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        self.assertEqual([i for i, layer in enumerate(model.model) if isinstance(layer, C2fLineContinuity)], [12, 15])
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            for index in (10, 13):
                reference.model[index].correction.weight.normal_(0, 0.1)
            model.load(reference, verbose=False)
            torch.testing.assert_close(model(image)[0], reference(image)[0], rtol=0, atol=0)
            self.assertEqual(model.stride.tolist(), [8, 16, 32])

            for index in (12, 15):
                model.model[index].project.weight.normal_(0, 0.1)
            expected = model(image)[0]
            for index in (12, 15):
                block = model.model[index]
                block.forward = block.forward_split
            torch.testing.assert_close(model(image)[0], expected)

    def test_detection_loss_trains_both_line_projections(self):
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
            for index in (12, 15):
                layer = model.model[index]
                self.assertGreater(layer.project.weight.grad.abs().sum().item(), 0.0)
                gradient = layer.evidence.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
