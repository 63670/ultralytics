"""Check remote-tile masking and the supervision path in the P3/P4 reference experiment."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fCrossScaleContext
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"


def config(name):
    return CONFIGS / f"yolov8n-fabric-{name}.yaml"


class FabricCrossScaleContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gate_responds_to_remote_deviation_and_excludes_aligned_tile(self):
        block = C2fCrossScaleContext(2, 2, n=0, c_ref=1, context_channels=1, pool_size=2).eval()
        with torch.no_grad():
            block.local.weight.zero_()
            block.local.weight[0, 0] = 1
            block.reference.weight.fill_(1)
            block.project.weight.fill_(1)
            feature = torch.full((1, 2, 4, 10), 0.5)
            reference = torch.full((1, 1, 2, 5), 0.5)
            torch.testing.assert_close(block._add_context(feature, reference), feature, rtol=0, atol=0)

            reference[..., :2] = 4.0
            aligned_only = block._add_context(feature, reference)
            torch.testing.assert_close(aligned_only[..., :4], feature[..., :4], rtol=0, atol=0)
            self.assertTrue((aligned_only[..., 4:] > feature[..., 4:]).all())

            reference[..., :2] = 0.5
            reference[..., -1:] = 1.0
            edge_tile = block._add_context(feature, reference)
            self.assertTrue((edge_tile[..., :8] > feature[..., :8]).all())
            torch.testing.assert_close(edge_tile[..., -2:], feature[..., -2:], rtol=0, atol=0)

            single_tile = block._add_context(feature[..., :4], reference[..., :2])
            torch.testing.assert_close(single_tile, feature[..., :4], rtol=0, atol=0)
            block.exclude_aligned = False
            included = block._add_context(feature, reference)
            self.assertTrue((included[..., -2:] > feature[..., -2:]).all())
            block.exclude_aligned = True
            block.deviation = False
            additive = block._add_context(feature, torch.full_like(reference, 0.5))
            torch.testing.assert_close(additive, feature + 0.5, rtol=0, atol=0)

    def test_retrieval_prefers_matching_remote_texture(self):
        block = C2fCrossScaleContext(2, 2, n=0, c_ref=2, context_channels=2, pool_size=2, deviation=False).eval()
        with torch.no_grad():
            for projection in (block.local, block.reference, block.project):
                projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            feature = torch.zeros(1, 2, 4, 12)
            feature[:, 0] = 1.0
            reference = torch.zeros(1, 2, 2, 6)
            reference[:, 0, :, 2:4] = 1.0
            reference[:, 1, :, 4:6] = 1.0
            retrieved = block._add_context(feature, reference) - feature
            expected = torch.tensor([4.0, 0.0]).softmax(dim=0)
            torch.testing.assert_close(retrieved[0, :, 0, 0], expected)

    def test_detector_configs_preserve_base_paths_weights_and_strides(self):
        torch.manual_seed(0)
        image = torch.rand(1, 3, 160, 192)
        variants = (
            ("baseline", "cross-scale-context", True, True),
            ("phase", "cross-scale-context-phase", True, True),
            ("phase", "cross-scale-add-phase", False, True),
            ("phase", "cross-scale-unmasked-phase", True, False),
        )
        for source, candidate, deviation, exclude_aligned in variants:
            with self.subTest(candidate=candidate), torch.no_grad():
                reference_cfg = YAML.load(config(source))
                candidate_cfg = YAML.load(config(candidate))
                candidate_cfg["head"][5] = reference_cfg["head"][5]
                self.assertEqual(candidate_cfg, reference_cfg)

                reference = DetectionModel(str(config(source)), verbose=False).eval()
                model = DetectionModel(str(config(candidate)), verbose=False).eval()
                model.load(reference, verbose=False)
                block = model.model[15]
                self.assertIsInstance(block, C2fCrossScaleContext)
                self.assertEqual(block.f, [-1, 12])
                self.assertEqual(block.deviation, deviation)
                self.assertEqual(block.exclude_aligned, exclude_aligned)
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                torch.testing.assert_close(model(image)[0], reference(image)[0], rtol=0, atol=0)

                block.project.weight.normal_(0, 0.1)
                expected = model(image)[0]
                block.forward = block.forward_split
                torch.testing.assert_close(model(image)[0], expected)

    def test_detection_loss_updates_both_projections_after_zero_start(self):
        torch.manual_seed(0)
        model = DetectionModel(str(config("cross-scale-context-phase")), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 160, 160),
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
            self.assertGreater(block.project.weight.grad.abs().sum().item(), 0.0)
            for projection in (block.local, block.reference):
                gradient = projection.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
