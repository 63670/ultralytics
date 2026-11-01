"""Check Sobel priors, baseline transfer, gradients and actual ONNX execution for the new candidate."""
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
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import C2f, C2fSobel, PhaseAwareUpsample
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.export.rkopt import export_rkopt_onnx, prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / 'ultralytics/cfg/models/v8'
CONFIG = CONFIGS / 'yolov8n-fabric-sobel-phase.yaml'
VARIANTS = (CONFIG.name, 'yolov8n-fabric-sobel.yaml', 'yolov8n-fabric-phase.yaml')


class SobelPhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_sobel_filters_are_fixed_and_respond_to_perpendicular_steps(self):
        module = C2fSobel(16, 32, n=2).double().eval()
        parameters = dict(module.named_parameters())
        self.assertNotIn('sobel_x.weight', parameters)
        self.assertNotIn('sobel_y.weight', parameters)
        vertical = torch.zeros(1, module.active, 9, 11, dtype=torch.double)
        vertical[:, :, :, 5:] = 1
        gx, gy = module.sobel_x(vertical), module.sobel_y(vertical)
        self.assertGreater(gx[:, :, 1:-1, 4:6].abs().sum().item(), 0)
        self.assertEqual(gy[:, :, 1:-1, 1:-1].abs().sum().item(), 0)
        horizontal = vertical.transpose(2, 3)
        self.assertEqual(module.sobel_x(horizontal)[:, :, 1:-1, 1:-1].abs().sum().item(), 0)
        self.assertGreater(module.sobel_y(horizontal)[:, :, 4:6, 1:-1].abs().sum().item(), 0)

    def test_c2f_initial_mapping_and_nonzero_split_forward_match(self):
        for partial in (0.25, 1.0):
            with self.subTest(partial=partial):
                baseline = C2f(16, 32, n=2, shortcut=True).eval()
                module = C2fSobel(16, 32, n=2, shortcut=True, partial=partial).eval()
                module.load_state_dict(baseline.state_dict(), strict=False)
                x = torch.randn(2, 16, 9, 13)
                torch.testing.assert_close(module(x), baseline(x), atol=0, rtol=0)
                with torch.no_grad():
                    module.edge_mix.weight.normal_(0, 0.1)
                    module.selector.weight.normal_(0, 0.1)
                torch.testing.assert_close(module(x), module.forward_split(x))

    def test_phase_weights_can_reconstruct_four_distinct_subpixels(self):
        module = PhaseAwareUpsample(1).double().eval()
        x = torch.ones(2, 1, 3, 5, dtype=torch.double)
        torch.testing.assert_close(module(x), F.interpolate(x, scale_factor=2), atol=0, rtol=0)
        with torch.no_grad():
            module.correction.weight[0, 0, 1, 1] = 2
            module.phase.bias.copy_(torch.tensor([-3., -1., 1., 3.], dtype=torch.double))
        y = module(x)
        self.assertEqual(tuple(y.shape), (2, 1, 6, 10))
        torch.testing.assert_close(y[0, 0, :2, :2], torch.tensor([[1., 5/3], [7/3, 3.]], dtype=torch.double))

    def test_configs_preserve_all_baseline_weights_and_prediction_scales(self):
        source = DetectionModel(str(CONFIGS / 'yolov8n-fabric-baseline.yaml'), verbose=False).eval()
        x = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            reference = source(x)[0]
            for filename in VARIANTS:
                with self.subTest(config=filename):
                    model = DetectionModel(str(CONFIGS / filename), verbose=False).eval()
                    model.load(source, verbose=False)
                    for name, tensor in source.state_dict().items():
                        torch.testing.assert_close(model.state_dict()[name], tensor, atol=0, rtol=0)
                    torch.testing.assert_close(model(x)[0], reference, atol=0, rtol=0)
                    self.assertEqual(model.stride.tolist(), [8, 16, 32])
        model = DetectionModel(str(CONFIG), verbose=False)
        self.assertEqual([i for i, layer in enumerate(model.model) if isinstance(layer, C2fSobel)], [4, 6])
        self.assertEqual([i for i, layer in enumerate(model.model) if isinstance(layer, PhaseAwareUpsample)], [10, 13])

    def test_detection_loss_trains_corrections_then_selectors_without_changing_sobel(self):
        model = DetectionModel(str(CONFIG), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            'img': torch.rand(2, 3, 96, 96),
            'batch_idx': torch.tensor([0., 0., 1.]),
            'cls': torch.tensor([[0.], [1.], [2.]]),
            'bboxes': torch.tensor([[.5, .3, .65, .05], [.3, .5, .05, .65], [.7, .7, .15, .15]]),
        }
        fixed = {name: t.clone() for name, t in model.named_buffers() if 'sobel_' in name}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            for layer in (model.model[4], model.model[6]):
                self.assertGreater(layer.edge_mix.weight.grad.abs().sum().item(), 0)
                if step:
                    self.assertGreater(layer.selector.weight.grad.abs().sum().item(), 0)
            for layer in (model.model[10], model.model[13]):
                self.assertGreater(layer.correction.weight.grad.abs().sum().item(), 0)
                if step:
                    self.assertGreater(layer.phase.weight.grad.abs().sum().item(), 0)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            optimizer.step()
        for name, value in fixed.items():
            torch.testing.assert_close(dict(model.named_buffers())[name], value, atol=0, rtol=0)

    def test_nonzero_paths_survive_checkpoint_fusion_and_rkopt_onnx(self):
        model = DetectionModel(str(CONFIG), verbose=False).eval()
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, C2fSobel):
                    layer.edge_mix.weight.normal_(0, 0.1)
                    layer.selector.weight.normal_(0, 0.1)
                elif isinstance(layer, PhaseAwareUpsample):
                    layer.correction.weight.normal_(0, 0.1)
                    layer.phase.weight.normal_(0, 0.1)
        x = torch.rand(1, 3, 64, 96)
        with torch.no_grad():
            reference = model(x)[0]
            for layer in model.modules():
                if isinstance(layer, C2f):
                    layer.forward = layer.forward_split
            torch.testing.assert_close(model(x)[0], reference)
            torch.testing.assert_close(copy.deepcopy(model).fuse(verbose=False)(x)[0], reference, atol=1e-3, rtol=1e-4)
            expected = [t.numpy() for t in prepare_rkopt_model(model)(x)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'sobel-phase.onnx'
            export_rkopt_onnx(model, output, imgsz=(64, 96))
            graph = onnx.load(str(output)).graph
            types = {node.op_type for node in graph.node}
            self.assertTrue({'Abs', 'Conv', 'DepthToSpace'} <= types)
            self.assertFalse({'Softmax', 'GridSample', 'Gather', 'GatherND', 'GatherElements'} & types)
            self.assertTrue(all(len(v.type.tensor_type.shape.dim) <= 4 for v in (*graph.input, *graph.output)))
            actual = ort.InferenceSession(str(output), providers=['CPUExecutionProvider']).run(None, {'images': x.numpy()})
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=1e-4)
            wrapper = YOLO(str(CONFIG))
            wrapper.model.load(model, verbose=False)
            checkpoint = Path(directory) / 'candidate.pt'
            wrapper.save(str(checkpoint))
            loaded = YOLO(str(checkpoint)).model.eval()
            torch.testing.assert_close(loaded(x)[0], reference, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
