"""Check position-preserving long-range attention and the P3/P4 detector configuration."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import C2fRibbonAttention
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
BASELINE = CONFIGS / "yolov8n-fabric-baseline.yaml"
CANDIDATE = CONFIGS / "yolov8n-fabric-ribbon-attention.yaml"


class FabricRibbonAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_remote_token_is_read_at_its_exact_phase_and_row_band(self):
        block = C2fRibbonAttention(4, 4, n=0, head_dim=1, band=2, step=4)
        qkv = torch.zeros(2, 3, 5, 11)
        qkv[:, 0, 0, 0] = 1.0
        qkv[:, 1, 1, 8] = 6.0
        qkv[:, 2, 1, 8] = torch.tensor([2.0, 3.0])
        qkv[:, 1:, 1, 9] = 100.0  # Neighboring column phase.
        qkv[:, 1:, 2, 8] = 100.0  # Neighboring row band.
        context = block._ribbon_context(qkv)
        expected = torch.tensor([6.0, 0.0, 0.0, 0.0, 0.0, 0.0]).softmax(0)[0] * torch.tensor([2.0, 3.0])
        torch.testing.assert_close(context[:, 0, 0, 0], expected)
        self.assertEqual(tuple(context.shape), (2, 1, 5, 11))

    def test_padding_does_not_dilute_valid_values_or_produce_nan(self):
        block = C2fRibbonAttention(4, 4, n=0, head_dim=1, band=2, step=8)
        for shape in ((5, 11), (3, 1), (1, 3)):
            qkv = torch.zeros(2, 3, *shape)
            qkv[:, 2] = 1.0
            actual = block._ribbon_context(qkv)
            torch.testing.assert_close(actual, torch.ones_like(actual))

    def test_only_p3_p4_change_and_zero_start_matches_baseline(self):
        torch.manual_seed(0)
        base_cfg, candidate_cfg = YAML.load(BASELINE), YAML.load(CANDIDATE)
        for index in (4, 6):
            candidate_cfg["backbone"][index] = base_cfg["backbone"][index]
        self.assertEqual(candidate_cfg, base_cfg)

        baseline = DetectionModel(str(BASELINE), verbose=False).eval()
        model = DetectionModel(str(CANDIDATE), verbose=False).eval()
        self.assertEqual([i for i, layer in enumerate(model.model) if isinstance(layer, C2fRibbonAttention)], [4, 6])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            model.load(baseline, verbose=False)
            torch.testing.assert_close(model(image)[0], baseline(image)[0], rtol=0, atol=0)
            for index in (4, 6):
                block = model.model[index]
                block.project.weight.normal_(0, 0.1)
                features = torch.rand(2, block.cv1.conv.in_channels, 9, 13)
                torch.testing.assert_close(block(features), block.forward_split(features))

    def test_both_backbone_blocks_receive_gradients_after_zero_start(self):
        torch.manual_seed(0)
        model = DetectionModel(str(CANDIDATE), verbose=False).train()
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
            for index in (4, 6):
                block = model.model[index]
                self.assertGreater(block.project.weight.grad.abs().sum().item(), 0)
                for layer in (block.qkv, block.local):
                    gradient = layer.weight.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
