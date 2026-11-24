"""Verify long-axis coverage, DFL viability, and the training-only model configuration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.tal import LongAxisTaskAlignedAssigner, TaskAlignedAssigner

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-long-axis-assignment-phase.yaml"


class FabricLongAxisAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_horizontal_and_vertical_boxes_cover_three_thirds_within_the_topk_budget(self):
        standard = TaskAlignedAssigner(topk=4, num_classes=1)
        long_axis = LongAxisTaskAlignedAssigner(topk=4, num_classes=1, reg_max=16)
        positions = torch.tensor([8.0, 24.0, 40.0, 48.0, 56.0, 64.0, 72.0, 80.0, 96.0, 104.0, 120.0])
        anchors = torch.stack((positions, torch.full_like(positions, 20.0)), dim=1)
        strides = torch.full((len(anchors), 1), 8.0)
        scores = torch.tensor([0.90, 0.80, 0.70, 0.95, 0.98, 0.99, 0.97, 0.91, 0.75, 0.85, 0.90]).view(1, -1, 1)
        gt_boxes = torch.tensor([[[0.0, 16.0, 128.0, 24.0]]])
        labels = torch.zeros(1, 1, 1)
        mask = torch.ones_like(labels, dtype=torch.bool)

        for point_order, box_order in (([0, 1], [0, 1, 2, 3]), ([1, 0], [1, 0, 3, 2])):
            truth = gt_boxes[..., box_order]
            points = anchors[:, point_order]
            predictions = truth.expand(-1, len(points), -1)
            normal = standard(scores, predictions, points, labels, truth, mask)[3]
            covered = long_axis(scores, predictions, points, labels, truth, mask, anchor_strides=strides)[3]
            self.assertEqual(positions[normal[0]].tolist(), [48.0, 56.0, 64.0, 72.0])
            self.assertEqual(positions[covered[0]].tolist(), [24.0, 56.0, 64.0, 104.0])

    def test_low_quality_endpoints_are_skipped_and_round_boxes_use_baseline_assignment(self):
        standard = TaskAlignedAssigner(topk=4, num_classes=1)
        long_axis = LongAxisTaskAlignedAssigner(topk=4, num_classes=1, reg_max=16)
        anchors = torch.tensor([[24.0, 20.0], [48.0, 20.0], [56.0, 20.0], [64.0, 20.0], [72.0, 20.0], [104.0, 20.0]])
        strides = torch.full((len(anchors), 1), 8.0)
        scores = torch.ones(1, len(anchors), 1)
        labels = torch.zeros(1, 1, 1)
        mask = torch.ones_like(labels, dtype=torch.bool)
        for box in ([0.0, 16.0, 128.0, 24.0], [0.0, 0.0, 128.0, 80.0]):
            truth = torch.tensor(box).reshape(1, 1, 4)
            predictions = truth.expand(-1, len(anchors), -1).clone()
            predictions[:, [0, -1]] = torch.tensor([0.0, 16.0, 32.0, 24.0])
            expected = standard(scores, predictions, anchors, labels, truth, mask)
            actual = long_axis(scores, predictions, anchors, labels, truth, mask, anchor_strides=strides)
            for actual_tensor, expected_tensor in zip(actual, expected):
                torch.testing.assert_close(actual_tensor, expected_tensor)

    def test_config_changes_assignment_only_and_detection_loss_remains_finite(self):
        torch.manual_seed(0)
        base_cfg, combo_cfg = YAML.load(PHASE), YAML.load(COMBINATION)
        self.assertIs(combo_cfg.pop("long_axis_assignment"), True)
        self.assertEqual(base_cfg, combo_cfg)
        reference = DetectionModel(str(PHASE), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        model.load(reference, verbose=False)
        image = torch.rand(2, 3, 128, 160)
        with torch.no_grad():
            torch.testing.assert_close(model(image)[0], reference(image)[0], atol=0, rtol=0)

        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        self.assertIsInstance(model.init_criterion().assigner, LongAxisTaskAlignedAssigner)
        model.train()
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertTrue(torch.isfinite(model.model[0].conv.weight.grad).all())
