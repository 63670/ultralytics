"""Check content-routed global context and its YOLOv8 backbone integration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fStateContext
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
BASELINE = CONFIGS / "yolov8n-fabric-baseline.yaml"
P4 = CONFIGS / "yolov8n-fabric-state-context.yaml"
P3_P4 = CONFIGS / "yolov8n-fabric-state-context-p3-p4.yaml"


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

    def test_candidates_change_only_the_expected_backbone_stages_and_preserve_initial_predictions(self):
        torch.manual_seed(0)
        base_cfg = YAML.load(BASELINE)
        baseline = DetectionModel(str(BASELINE), verbose=False).eval()
        image = torch.rand(1, 3, 128, 160)
        for config, stages in ((P4, [6]), (P3_P4, [4, 6])):
            with self.subTest(config=config.name), torch.no_grad():
                candidate_cfg = YAML.load(config)
                for stage in stages:
                    candidate_cfg["backbone"][stage] = base_cfg["backbone"][stage]
                self.assertEqual(candidate_cfg, base_cfg)

                model = DetectionModel(str(config), verbose=False).eval()
                blocks = [(i, layer) for i, layer in enumerate(model.model) if isinstance(layer, C2fStateContext)]
                self.assertEqual([i for i, _ in blocks], stages)
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                model.load(baseline, verbose=False)
                torch.testing.assert_close(model(image)[0], baseline(image)[0], rtol=0, atol=0)

                for _, block in blocks:
                    self.assertEqual(block.route.out_channels, 16)
                    block.state_expand.weight.normal_(0, 0.1)
                    features = torch.rand(1, block.cv1.conv.in_channels, 8, 10)
                    torch.testing.assert_close(block(features), block.forward_split(features))

    def test_detection_loss_updates_both_state_projections_and_routing(self):
        torch.manual_seed(0)
        model = DetectionModel(str(P3_P4), verbose=False).train()
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
            for stage in (4, 6):
                block = model.model[stage]
                self.assertGreater(block.state_expand.weight.grad.abs().sum().item(), 0)
                for layer in (block.route, block.route_spatial, block.state_reduce):
                    gradient = layer.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
