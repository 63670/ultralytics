"""Check content-routed global context and its single-stage YOLOv8 integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fStateContext
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
BASELINE = CONFIGS / "yolov8n-fabric-baseline.yaml"
CANDIDATE = CONFIGS / "yolov8n-fabric-state-context.yaml"


class FabricStateContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_remote_region_reaches_queries_off_its_row_and_column(self):
        block = C2fStateContext(4, 4, n=0, states=2).eval()
        with torch.no_grad():
            block.route.weight.zero_()
            block.route_spatial.weight.zero_()
            block.route.weight[0, 0] = 4.0
            block.route.weight[2, 1] = 4.0
            block.route.weight[3, 1] = -4.0
            block.state_reduce.weight.zero_()
            block.state_reduce.weight[0, 0] = 1.0
            block.state_expand.weight[0, 0] = 1.0

            feature = torch.zeros(1, 4, 9, 13)
            feature[0, 1, 0, :2] = torch.tensor([1.0, -1.0])
            before = block._add_context(feature)
            feature[0, 0, -1, -1] = 2.0
            change = block._add_context(feature) - before
        self.assertGreater(change[0, 0, 0, 0].item(), 1.0)
        self.assertLess(change[0, 0, 0, 1].item(), 0.05)

    def test_candidate_changes_only_backbone_p4_and_preserves_initial_predictions(self):
        torch.manual_seed(0)
        base_cfg, candidate_cfg = YAML.load(BASELINE), YAML.load(CANDIDATE)
        candidate_cfg["backbone"][6] = base_cfg["backbone"][6]
        self.assertEqual(candidate_cfg, base_cfg)

        baseline = DetectionModel(str(BASELINE), verbose=False).eval()
        model = DetectionModel(str(CANDIDATE), verbose=False).eval()
        block = model.model[6]
        self.assertIsInstance(block, C2fStateContext)
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            model.load(baseline, verbose=False)
            torch.testing.assert_close(model(image)[0], baseline(image)[0], rtol=0, atol=0)
            block.state_expand.weight.normal_(0, 0.1)
            feats = torch.rand(1, 128, 8, 10)
            torch.testing.assert_close(block(feats), block.forward_split(feats))

    def test_detection_loss_updates_state_projection_and_routing(self):
        torch.manual_seed(0)
        model = DetectionModel(str(CANDIDATE), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 128, 160),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        block = model.model[6]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            self.assertGreater(block.state_expand.weight.grad.abs().sum().item(), 0)
            for layer in (block.route, block.route_spatial, block.state_reduce):
                gradient = layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
