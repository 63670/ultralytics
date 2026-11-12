"""Check isolation, reference operator behavior, and gradients before costly training."""
import copy
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C2f, C2fAxisResidual, GatedInterpolationUpsample
from ultralytics.nn.modules.comparisons import (
    CoordinateAttention, C2fCoordinateAttention, CARAFEUpsample, C2fECA, C2fCBAM,
    EfficientChannelAttention, CBAMReference, DySampleUpsample,
)
from ultralytics.utils import YAML

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"ultralytics/cfg/models/v8"

class TargetedComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_configs_change_only_the_selected_sites(self):
        for filename,reference,section,sites in [
            ("yolov8n-fabric-coordinate-attention.yaml","yolov8n-fabric-axis-residual.yaml","backbone",[4,6]),
            ("yolov8n-fabric-bilinear.yaml","yolov8n-fabric-baseline.yaml","head",[0,3]),
            ("yolov8n-fabric-carafe-reference.yaml","yolov8n-fabric-baseline.yaml","head",[0,3]),
            ("yolov8n-fabric-eca.yaml","yolov8n-fabric-axis-residual.yaml","backbone",[4,6]),
            ("yolov8n-fabric-cbam.yaml","yolov8n-fabric-axis-residual.yaml","backbone",[4,6]),
            ("yolov8n-fabric-dysample.yaml","yolov8n-fabric-baseline.yaml","head",[0,3]),
        ]:
            base=YAML.load(CFG/reference)
            other=YAML.load(CFG/filename)
            for key in ("nc","scales"): self.assertEqual(other[key],base[key])
            for key in ("backbone","head"):
                self.assertEqual(len(base[key]),len(other[key]))
                for i,(a,b) in enumerate(zip(base[key],other[key])):
                    self.assertEqual(a==b,not(key==section and i in sites))

    def test_ca_site_and_export_split_preserve_original_weight_paths(self):
        base=C2f(32,64,n=2,shortcut=True).eval()
        block=C2fCoordinateAttention(32,64,n=2,shortcut=True).eval()
        block.load_state_dict(base.state_dict(),strict=False)
        for key,value in base.state_dict().items(): torch.testing.assert_close(block.state_dict()[key],value)
        x=torch.randn(2,32,11,13,requires_grad=True)
        y=block(x)
        torch.testing.assert_close(y,block.forward_split(x))
        y.square().mean().backward()
        self.assertGreater(block.ca.height.weight.grad.abs().sum().item(),0)
        self.assertEqual(y.shape,(2,64,11,13))

    def test_uniform_carafe_equals_local_average_then_nearest(self):
        module=CARAFEUpsample(8)
        torch.nn.init.zeros_(module.content_encoder.weight)
        x=torch.randn(2,8,7,9,requires_grad=True)
        y=module(x)
        expected=F.interpolate(F.avg_pool2d(x,5,1,2),scale_factor=2,mode="nearest")
        torch.testing.assert_close(y,expected,atol=1e-6,rtol=1e-5)
        y.square().mean().backward()
        self.assertGreater(module.content_encoder.weight.grad.abs().sum().item(),0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_eca_and_cbam_keep_pretrained_c2f_paths_and_split_forward(self):
        base=C2f(32,64,n=2,shortcut=True).eval()
        for block_type in (C2fECA,C2fCBAM):
            with self.subTest(block=block_type.__name__):
                block=block_type(32,64,n=2,shortcut=True).eval()
                missing=block.load_state_dict(base.state_dict(),strict=False)
                self.assertFalse(missing.unexpected_keys)
                self.assertTrue(all(key.startswith("attention.") for key in missing.missing_keys))
                for key,value in base.state_dict().items():
                    torch.testing.assert_close(block.state_dict()[key],value)
                x=torch.randn(2,32,11,13,requires_grad=True)
                y=block(x)
                torch.testing.assert_close(y,block.forward_split(x))
                y.square().mean().backward()
                self.assertEqual(y.shape,(2,64,11,13))
                gradients=[p.grad for p in block.attention.parameters()]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)

    def test_reference_attention_has_both_cbam_descriptors_and_eca_default(self):
        eca=EfficientChannelAttention(32)
        self.assertEqual(sum(p.numel() for p in eca.parameters()),3)
        cbam=CBAMReference(32).eval()
        calls=[]
        handle=cbam.mlp.register_forward_pre_hook(lambda m,args:calls.append(args[0].detach().clone()))
        x=torch.randn(2,32,7,9)
        cbam(x)
        handle.remove()
        self.assertEqual(len(calls),2)
        torch.testing.assert_close(calls[0],x.mean((2,3)))
        torch.testing.assert_close(calls[1],x.amax((2,3)))
        self.assertEqual(sum(p.numel() for p in cbam.parameters()),262)

    def test_zero_offset_dysample_matches_bilinear_on_rectangles(self):
        for shape in ((2,8,5,7),(1,16,1,3),(2,8,3,1)):
            with self.subTest(shape=shape):
                module=DySampleUpsample(shape[1])
                torch.nn.init.zeros_(module.offset.weight)
                x=torch.randn(*shape,requires_grad=True)
                y=module(x)
                expected=F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)
                torch.testing.assert_close(y,expected,atol=1e-6,rtol=1e-5)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(module.offset.weight.grad.abs().sum().item(),0)

    def test_dysample_groups_preserve_channels_and_batches(self):
        module=DySampleUpsample(8,groups=4)
        constants=torch.arange(16,dtype=torch.float32).reshape(2,8,1,1)
        x=constants.expand(2,8,5,7)
        torch.testing.assert_close(module(x),constants.expand(2,8,10,14))
        with self.assertRaises(ValueError):
            DySampleUpsample(6,groups=4)

    def test_complexity_includes_new_operators_and_preserves_the_input_model(self):
        from tools.run_mcgi_targeted import complexity
        for name in ("eca","cbam","dysample"):
            with self.subTest(model=name):
                model=DetectionModel(str(CFG/f"yolov8n-fabric-{name}.yaml"),verbose=False).eval()
                original={k:v.detach().clone() for k,v in model.state_dict().items()}
                stats=complexity(model)
                self.assertEqual(stats["params"],sum(p.numel() for p in model.parameters()))
                expected_fused=copy.deepcopy(model).fuse(verbose=False)
                self.assertEqual(stats["params_fused"],sum(p.numel() for p in expected_fused.parameters()))
                self.assertGreater(stats["comparison_arithmetic_gflops"],0)
                self.assertGreater(stats["gflops"],stats["conv_gflops"])
                for key,value in original.items():
                    torch.testing.assert_close(model.state_dict()[key],value)

    def test_all_detectors_forward_backward_and_fuse(self):
        for name in ["coordinate-attention","bilinear","carafe-reference","eca","cbam","dysample"]:
            model=DetectionModel(str(CFG/f"yolov8n-fabric-{name}.yaml"),verbose=False).eval()
            x=torch.randn(1,3,128,160)
            prediction=model(x)[0]
            self.assertEqual(prediction.shape,(1,7,420))
            prediction.square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            with torch.no_grad():
                fused=copy.deepcopy(model).fuse(verbose=False)
                torch.testing.assert_close(fused(x)[0],prediction,atol=1e-3,rtol=1e-4)

if __name__=="__main__": unittest.main()
