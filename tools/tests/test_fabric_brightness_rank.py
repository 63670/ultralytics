"""Check local rank, classification-only correction, and raw Rockchip outputs."""

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import BrightnessRankDetect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.export.rkopt import prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"
COMBINATION = CONFIGS / "yolov8n-fabric-brightness-rank-phase.yaml"


class FabricBrightnessRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rank_separates_local_extremes_and_rejects_a_single_outlier(self):
        texture = torch.full((1, 1, 17, 23), 0.5)
        texture[..., 8, 6] += 0.06
        texture[..., 8, 16] -= 0.06
        texture[..., 11, 11] = 20.0
        rank = BrightnessRankDetect._local_rank(texture)
        self.assertGreater(rank[0, 0, 8, 6].item(), 0.1)
        self.assertGreater(rank[0, 1, 8, 16].item(), 0.1)
        self.assertEqual(rank[0, 1, 8, 6].item(), 0)
        self.assertEqual(rank[0, 0, 8, 16].item(), 0)
        self.assertEqual(rank[0, :, 8, 11].count_nonzero().item(), 0)
        self.assertEqual(rank[..., :3, :].count_nonzero().item(), 0)
        torch.testing.assert_close(rank, BrightnessRankDetect._local_rank(texture + 0.1), atol=1e-6, rtol=0)
        torch.testing.assert_close(rank, BrightnessRankDetect._local_rank(texture * 0.4), atol=1e-6, rtol=0)

    def test_model_changes_only_p3_scores_and_keeps_rkopt_nine_output_contract(self):
        torch.manual_seed(0)
        original_cfg, candidate_cfg = YAML.load(PHASE), YAML.load(COMBINATION)
        candidate_cfg["head"][-1] = original_cfg["head"][-1]
        self.assertEqual(candidate_cfg, original_cfg)

        baseline = DetectionModel(str(PHASE), verbose=False).eval()
        model = DetectionModel(str(COMBINATION), verbose=False).eval()
        head = model.model[-1]
        self.assertIsInstance(head, BrightnessRankDetect)
        self.assertEqual(head.f, [15, 18, 21, 2])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        image = torch.rand(1, 3, 96, 128)
        with torch.no_grad():
            model.load(baseline, verbose=False)
            output, native = model(image)
            torch.testing.assert_close(output, baseline(image)[0], atol=0, rtol=0)

            head.rank_project.weight.normal_(0, 0.2)
            _, modified = model(image)
            torch.testing.assert_close(modified["boxes"], native["boxes"], atol=0, rtol=0)
            p3_cells = modified["feats"][0].shape[-2] * modified["feats"][0].shape[-1]
            self.assertGreater((modified["scores"][..., :p3_cells] - native["scores"][..., :p3_cells]).abs().max().item(), 0)
            torch.testing.assert_close(modified["scores"][..., p3_cells:], native["scores"][..., p3_cells:], atol=0, rtol=0)

            raw = prepare_rkopt_model(model)(image)
            self.assertEqual(len(raw), 9)
            boxes = torch.cat([value.flatten(2) for value in raw[0::3]], dim=-1)
            scores = torch.cat([value.flatten(2) for value in raw[1::3]], dim=-1)
            torch.testing.assert_close(boxes, modified["boxes"], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(scores, modified["scores"].sigmoid())

    def test_detection_loss_updates_rank_parameters(self):
        torch.manual_seed(0)
        model = DetectionModel(str(COMBINATION), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 96, 128),
            "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
            "cls": torch.tensor([[0.0], [1.0], [2.0]]),
            "bboxes": torch.tensor([[0.5, 0.3, 0.65, 0.05], [0.3, 0.5, 0.05, 0.65], [0.7, 0.7, 0.15, 0.15]]),
        }
        head = model.model[-1]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            self.assertGreater(head.rank_project.weight.grad.abs().sum().item(), 0)
            for layer in (head.rank_features, head.rank_gate):
                gradient = layer.weight.grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertEqual(gradient.count_nonzero().item() > 0, step == 1)
            optimizer.step()
