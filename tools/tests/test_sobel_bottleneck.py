"""Check progressive Sobel bottlenecks without changing original C2f paths or old checkpoint behavior."""
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
from ultralytics.nn.modules import Bottleneck, BottleneckSobel, C2f, C2fSobelBottleneck, PhaseAwareUpsample, SobelResidual
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.export.rkopt import export_rkopt_onnx, prepare_rkopt_model

CONFIGS = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/v8"
COMBINED = CONFIGS / "yolov8n-fabric-sobel-bottleneck-phase.yaml"
SOBEL_ONLY = CONFIGS / "yolov8n-fabric-sobel-bottleneck.yaml"


class SobelBottleneckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(0)

    def test_bottleneck_keeps_the_original_shortcut_and_convolutions_at_initialization(self):
        for c1, c2, shortcut in [(16, 16, True), (16, 16, False), (16, 32, True)]:
            with self.subTest(c1=c1, c2=c2, shortcut=shortcut):
                source = Bottleneck(c1, c2, shortcut).eval()
                model = BottleneckSobel(c1, c2, shortcut).eval()
                model.load_state_dict(source.state_dict(), strict=False)
                image = torch.randn(2, c1, 9, 13)
                torch.testing.assert_close(model(image), source(image), atol=0, rtol=0)
                self.assertEqual(model.add, source.add)
                self.assertEqual(model.sobel.channels, c2)
                self.assertNotIn("sobel.sobel_x.weight", dict(model.named_parameters()))
                self.assertNotIn("sobel.sobel_y.weight", dict(model.named_parameters()))

    def test_every_refined_bottleneck_output_is_passed_to_the_next_bottleneck(self):
        source = C2f(16, 32, n=3, shortcut=True).eval()
        model = C2fSobelBottleneck(16, 32, n=3, shortcut=True).eval()
        model.load_state_dict(source.state_dict(), strict=False)
        image = torch.randn(2, 16, 9, 13)
        torch.testing.assert_close(model(image), source(image), atol=0, rtol=0)
        inputs, outputs, counts, handles = {}, {}, [0, 0, 0], []
        for index, block in enumerate(model.m):
            with torch.no_grad():
                block.sobel.edge_mix.weight.normal_(0, 0.1)
                block.sobel.selector.weight.normal_(0, 0.1)

            def before(module, value, index=index):
                inputs[index] = value[0].detach().clone()

            def after(module, value, output, index=index):
                outputs[index] = output.detach().clone()

            def count(module, value, output, index=index):
                counts[index] += 1

            handles.extend([
                block.register_forward_pre_hook(before), block.register_forward_hook(after),
                block.sobel.register_forward_hook(count),
            ])
        model(image)
        for handle in handles:
            handle.remove()
        self.assertEqual(counts, [1, 1, 1])
        torch.testing.assert_close(inputs[1], outputs[0], atol=0, rtol=0)
        torch.testing.assert_close(inputs[2], outputs[1], atol=0, rtol=0)
        torch.testing.assert_close(model(image), model.forward_split(image))

    def test_configs_transfer_all_baseline_weights_and_have_no_p5_sobel(self):
        for config, reference in [(COMBINED, "phase"), (SOBEL_ONLY, "baseline")]:
            with self.subTest(config=config.name):
                source = DetectionModel(str(CONFIGS / f"yolov8n-fabric-{reference}.yaml"), verbose=False).eval()
                model = DetectionModel(str(config), verbose=False).eval()
                model.load(source, verbose=False)
                for name, value in source.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
                self.assertEqual([i for i, m in enumerate(model.model) if isinstance(m, C2fSobelBottleneck)], [4, 6])
                self.assertEqual([len(model.model[i].m) for i in (4, 6)], [2, 2])
                self.assertEqual(type(model.model[8]), C2f)
                self.assertEqual(sum(isinstance(m, SobelResidual) for m in model.modules()), 4)
                self.assertEqual([m.channels for m in model.modules() if isinstance(m, SobelResidual)], [32, 32, 64, 64])
                self.assertEqual(model.stride.tolist(), [8, 16, 32])
                self.assertEqual(sum(isinstance(m, PhaseAwareUpsample) for m in model.modules()), 2 if reference == "phase" else 0)
                image = torch.rand(1, 3, 64, 96)
                with torch.no_grad():
                    torch.testing.assert_close(model(image)[0], source(image)[0], atol=0, rtol=0)

    def test_detection_loss_updates_each_sobel_projection_then_each_selector(self):
        model = DetectionModel(str(COMBINED), verbose=False).train()
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        batch = {
            "img": torch.rand(2, 3, 96, 96),
            "batch_idx": torch.tensor([0., 0., 1.]),
            "cls": torch.tensor([[0.], [1.], [2.]]),
            "bboxes": torch.tensor([[.5, .3, .65, .05], [.3, .5, .05, .65], [.7, .7, .15, .15]]),
        }
        fixed = {name: value.clone() for name, value in model.named_buffers() if "sobel_x.weight" in name or "sobel_y.weight" in name}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            loss.sum().backward()
            for block in model.modules():
                if isinstance(block, SobelResidual):
                    self.assertGreater(block.edge_mix.weight.grad.abs().sum().item(), 0)
                    if step:
                        self.assertGreater(block.selector.weight.grad.abs().sum().item(), 0)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        for name, value in fixed.items():
            torch.testing.assert_close(dict(model.named_buffers())[name], value, atol=0, rtol=0)

    def test_nonzero_refinement_survives_split_fusion_checkpoint_and_onnx(self):
        model = DetectionModel(str(COMBINED), verbose=False).eval()
        with torch.no_grad():
            for block in model.modules():
                if isinstance(block, SobelResidual):
                    block.edge_mix.weight.normal_(0, 0.1)
                    block.selector.weight.normal_(0, 0.1)
                elif isinstance(block, PhaseAwareUpsample):
                    block.correction.weight.normal_(0, 0.1)
                    block.phase.weight.normal_(0, 0.1)
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
            output = Path(directory) / "sobel-bottleneck.onnx"
            export_rkopt_onnx(model, output, imgsz=(64, 96))
            graph = onnx.load(str(output)).graph
            self.assertEqual(sum(n.op_type == "Abs" for n in graph.node), 8)
            actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"images": image.numpy()})
            for a, b in zip(expected, actual):
                np.testing.assert_allclose(a, b, atol=2e-5, rtol=1e-4)
            wrapper = YOLO(str(COMBINED))
            wrapper.model.load(model, verbose=False)
            checkpoint = Path(directory) / "candidate.pt"
            wrapper.save(str(checkpoint))
            torch.testing.assert_close(YOLO(str(checkpoint)).model.eval()(image)[0], reference, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
