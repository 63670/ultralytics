"""Check nonlocal attention and the fabric detector configurations that use it."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fPooledContext
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
MODEL_PREFIX = "yolov8n-fabric-"


class FabricPooledContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_distant_region_changes_a_query_outside_its_row_and_column(self):
        block = C2fPooledContext(2, 2, n=0, context_channels=1, pool_size=5).eval()
        with torch.no_grad():
            block.query.weight.fill_(1)
            block.key_value.weight.fill_(1)
            block.project.weight.fill_(1)
            features = torch.zeros(1, 2, 8, 9)
            features[0, 0, 0, 0] = 1.0
            before = block._add_context(features) - features
            features[0, 0, 5:, 5:] = 1.0
            after = block._add_context(features) - features
            change = after - before

        self.assertGreater(change[0, 0, 0, 0].item(), change[0, 0, 0, 1].item())
        self.assertAlmostEqual(change[0, 0, 0, 1].item(), 0.25)

    def test_configs_only_replace_p4_and_preserve_loaded_predictions(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 96, 128)
        variants = (("baseline", "pooled-context"), ("phase", "pooled-context-phase"))
        for reference_name, candidate_name in variants:
            with self.subTest(candidate=candidate_name), torch.no_grad():
                reference_path = CONFIGS / f"{MODEL_PREFIX}{reference_name}.yaml"
                candidate_path = CONFIGS / f"{MODEL_PREFIX}{candidate_name}.yaml"
                reference_cfg = YAML.load(reference_path)
                candidate_cfg = YAML.load(candidate_path)
                candidate_cfg["backbone"][6] = reference_cfg["backbone"][6]
                self.assertEqual(candidate_cfg, reference_cfg)

                reference = DetectionModel(str(reference_path), verbose=False).eval()
                candidate = DetectionModel(str(candidate_path), verbose=False).eval()
                if reference_name == "phase":
                    for index in (10, 13):
                        reference.model[index].correction.weight.normal_(0, 0.1)
                candidate.load(reference, verbose=False)
                self.assertEqual([i for i, block in enumerate(candidate.model) if isinstance(block, C2fPooledContext)], [6])
                self.assertEqual(candidate.stride.tolist(), [8, 16, 32])
                torch.testing.assert_close(candidate(image)[0], reference(image)[0], rtol=0, atol=0)

                block = candidate.model[6]
                block.project.weight.normal_(0, 0.1)
                features = torch.rand(1, 128, 6, 8)
                torch.testing.assert_close(block(features), block.forward_split(features))

    def test_detection_loss_trains_context_projection_then_attention(self):
        torch.manual_seed(0)
        model = DetectionModel(str(CONFIGS / f"{MODEL_PREFIX}pooled-context-phase.yaml"), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 96, 128),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        context = model.model[6]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            self.assertGreater(context.project.weight.grad.abs().sum().item(), 0)
            for projection in (context.query, context.key_value):
                gradient = projection.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                if step == 0:
                    self.assertEqual(gradient.count_nonzero().item(), 0)
                else:
                    self.assertGreater(gradient.abs().sum().item(), 0)
            optimizer.step()
