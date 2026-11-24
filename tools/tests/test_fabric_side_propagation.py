"""Check long-range, gated propagation and regression-only integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import SidePropagationDetect
from ultralytics.nn.modules.head import _SidePropagation
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.export.rkopt import prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-side-propagation-phase.yaml"


class FabricSidePropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parallel_scan_reaches_63_cells_and_cannot_skip_a_closed_link(self):
        branch = _SidePropagation(1, 16)
        value = torch.zeros(1, 1, 2, 100)
        value[..., 0] = 1.0
        retention = torch.full_like(value, 0.97)
        retention[..., 0] = 0.0
        result = branch._scan(value, retention, False)
        torch.testing.assert_close(result[..., 63], torch.full_like(result[..., 63], 0.97**63))
        self.assertEqual(result[..., 64:].count_nonzero().item(), 0)
        reverse = branch._scan(value.flip(-1), retention.flip(-1), True)
        torch.testing.assert_close(result, reverse.flip(-1))

        retention[..., 31] = 0.0
        result = branch._scan(value, retention, False)
        self.assertEqual(result[..., 31:].count_nonzero().item(), 0)

    def test_affinity_reduces_distant_influence_across_a_feature_transition(self):
        branch = _SidePropagation(1, 16)
        influence = []
        for separated in (False, True):
            value = torch.zeros(1, 2, 1, 80)
            if separated:
                value[:, 1, :, 24:] = 8.0
            value.requires_grad_(True)
            output = branch._propagate(value, branch.retention_bias[0], False)
            derivative = torch.autograd.grad(output[0, 0, 0, 48], value)[0]
            influence.append(derivative[0, 0, 0, 0].item())
        self.assertGreater(influence[0], 0.1)
        self.assertLess(influence[1], influence[0] * 0.001)

    def test_candidate_preserves_pretrained_mapping_and_modifies_only_p3_p4_boxes(self):
        torch.manual_seed(0)
        original_cfg, candidate_cfg = YAML.load(PHASE), YAML.load(COMBINATION)
        candidate_cfg["head"][-1] = original_cfg["head"][-1]
        self.assertEqual(candidate_cfg, original_cfg)

        reference = DetectionModel(str(PHASE), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        head = model.model[-1]
        self.assertIsInstance(head, SidePropagationDetect)
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        image = torch.rand(1, 3, 96, 128)
        with torch.no_grad():
            model.load(reference, verbose=False)
            decoded, initial = model(image)
            torch.testing.assert_close(decoded, reference(image)[0], atol=0, rtol=0)

            features = [torch.rand_like(feature) for feature in initial["feats"]]
            _, baseline = head(features)
            for propagation in head.side_propagation:
                propagation.project.weight.normal_(0, 0.2)
            _, modified = head(features)
            torch.testing.assert_close(modified["scores"], baseline["scores"], atol=0, rtol=0)
            sizes = [feature.shape[-2] * feature.shape[-1] for feature in modified["feats"]]
            difference = modified["boxes"] - baseline["boxes"]
            p3, p4, p5 = difference.split(sizes, dim=-1)
            self.assertGreater(p3.abs().max().item(), 0)
            self.assertGreater(p4.abs().max().item(), 0)
            self.assertEqual(p5.count_nonzero().item(), 0)

            raw = prepare_rkopt_model(model).model[-1](features)
            self.assertEqual(len(raw), 9)
            boxes = torch.cat([value.flatten(2) for value in raw[0::3]], dim=-1)
            scores = torch.cat([value.flatten(2) for value in raw[1::3]], dim=-1)
            torch.testing.assert_close(boxes, modified["boxes"], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(scores, modified["scores"].sigmoid())

    def test_detection_loss_updates_both_propagation_branches(self):
        torch.manual_seed(0)
        model = DetectionModel(str(COMBINATION), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 160, 192),
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
            for branch in model.model[-1].side_propagation:
                self.assertGreater(branch.project.weight.grad.abs().sum().item(), 0.0)
                for parameter in (branch.value.weight, branch.retention_bias):
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertEqual(parameter.grad.count_nonzero().item() > 0, step == 1)
            optimizer.step()
