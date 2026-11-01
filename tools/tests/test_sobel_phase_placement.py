"""Verify that moving Sobel after top-down fusion removes its direct input path into Phase."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, C2fSobel, PhaseAwareUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.export.rkopt import export_rkopt_onnx, prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
NECK = CONFIGS / "yolov8n-fabric-phase-sobel-neck.yaml"
CONTROL = CONFIGS / "yolov8n-fabric-sobel-p3-phase.yaml"
PHASE = CONFIGS / "yolov8n-fabric-phase.yaml"


class SobelPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    @staticmethod
    def phase_outputs(model, image):
        outputs = {}
        handles = []
        for index in (10, 13):
            def capture(module, inputs, output, index=index):
                outputs[index] = output.detach().clone()
            handles.append(model.model[index].register_forward_hook(capture))
        try:
            with torch.no_grad():
                prediction = model(image)[0].clone()
        finally:
            for handle in handles:
                handle.remove()
        return prediction, outputs

    def test_both_candidates_transfer_phase_baseline_exactly_with_equal_new_parameter_counts(self):
        source = DetectionModel(str(PHASE), verbose=False).eval()
        with torch.no_grad():
            for index in (10, 13):
                source.model[index].correction.weight.normal_(0, 0.1)
                source.model[index].phase.weight.normal_(0, 0.1)
        image = torch.rand(2, 3, 96, 128)
        with torch.no_grad():
            reference = source(image)[0]
            counts = []
            for config, sobel_index in ((NECK, 15), (CONTROL, 4)):
                model = DetectionModel(str(config), verbose=False).eval()
                model.load(source, verbose=False)
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                torch.testing.assert_close(model(image)[0], reference, atol=0, rtol=0)
                self.assertEqual([i for i, m in enumerate(model.model) if isinstance(m, C2fSobel)], [sobel_index])
                self.assertEqual([i for i, m in enumerate(model.model) if isinstance(m, PhaseAwareUpsample)], [10, 13])
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                counts.append(sum(p.numel() for p in model.parameters()))
            self.assertEqual(counts[0], counts[1])
            self.assertEqual(counts[0] - sum(p.numel() for p in source.parameters()), 144)

    def test_nonzero_neck_sobel_changes_predictions_without_changing_either_phase_output(self):
        model = DetectionModel(str(NECK), verbose=False).eval()
        image = torch.rand(2, 3, 96, 128)
        original, phase_before = self.phase_outputs(model, image)
        with torch.no_grad():
            model.model[15].edge_mix.weight.normal_(0, 0.4)
            model.model[15].selector.weight.normal_(0, 0.2)
        changed, phase_after = self.phase_outputs(model, image)
        for index in (10, 13):
            torch.testing.assert_close(phase_after[index], phase_before[index], atol=0, rtol=0)
        self.assertGreater((original - changed).abs().max().item(), 0)

    def test_backbone_control_still_changes_both_phase_outputs(self):
        model = DetectionModel(str(CONTROL), verbose=False).eval()
        image = torch.rand(2, 3, 96, 128)
        _, before = self.phase_outputs(model, image)
        with torch.no_grad():
            model.model[4].edge_mix.weight.normal_(0, 0.4)
        _, after = self.phase_outputs(model, image)
        for index in (10, 13):
            self.assertGreater((before[index] - after[index]).abs().max().item(), 0)

    def test_real_detection_loss_trains_the_relocated_sobel_and_phase(self):
        model = DetectionModel(str(NECK), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 96, 96),
            "batch_idx": torch.tensor([0., 0., 1.]),
            "cls": torch.tensor([[0.], [1.], [2.]]),
            "bboxes": torch.tensor([[.5, .3, .65, .05], [.3, .5, .05, .65], [.7, .7, .15, .15]]),
        }
        fixed = {name: value.clone() for name, value in model.named_buffers() if "sobel_" in name}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            sobel = model.model[15]
            self.assertGreater(sobel.edge_mix.weight.grad.abs().sum().item(), 0)
            if step:
                self.assertGreater(sobel.selector.weight.grad.abs().sum().item(), 0)
            for index in (10, 13):
                self.assertGreater(model.model[index].correction.weight.grad.abs().sum().item(), 0)
                if step:
                    self.assertGreater(model.model[index].phase.weight.grad.abs().sum().item(), 0)
            optimizer.step()
        for name, value in fixed.items():
            torch.testing.assert_close(dict(model.named_buffers())[name], value, atol=0, rtol=0)

    def test_nonzero_corrections_survive_split_fusion_checkpoint_and_onnx(self):
        model = DetectionModel(str(NECK), verbose=False).eval()
        with torch.no_grad():
            model.model[15].edge_mix.weight.normal_(0, 0.1)
            model.model[15].selector.weight.normal_(0, 0.1)
            for index in (10, 13):
                model.model[index].correction.weight.normal_(0, 0.1)
                model.model[index].phase.weight.normal_(0, 0.1)
        image = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            reference = model(image)[0]
            for layer in model.modules():
                if isinstance(layer, C2f):
                    layer.forward = layer.forward_split
            torch.testing.assert_close(model(image)[0], reference)
            torch.testing.assert_close(copy.deepcopy(model).fuse(verbose=False)(image)[0], reference, atol=1e-3, rtol=1e-4)
            expected = [t.numpy() for t in prepare_rkopt_model(model)(image)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "phase-sobel-neck.onnx"
            export_rkopt_onnx(model, output, imgsz=(64, 96))
            types = {node.op_type for node in onnx.load(str(output)).graph.node}
            self.assertTrue({"Abs", "Conv", "DepthToSpace"} <= types)
            self.assertFalse({"Softmax", "GridSample", "Gather", "GatherND", "GatherElements"} & types)
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
                None, {"images": image.numpy()}
            )
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=1e-4)
            wrapper = YOLO(str(NECK))
            wrapper.model.load(model, verbose=False)
            checkpoint = Path(directory) / "candidate.pt"
            wrapper.save(str(checkpoint))
            torch.testing.assert_close(YOLO(str(checkpoint)).model.eval()(image)[0], reference, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
