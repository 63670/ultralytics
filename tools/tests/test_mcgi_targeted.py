"""Check isolation, reference operator behavior, and gradients before costly training."""
import copy
import unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C2f, C2fAxisResidual, GatedInterpolationUpsample
from ultralytics.nn.modules.comparisons import CoordinateAttention, C2fCoordinateAttention, CARAFEUpsample
from ultralytics.utils import YAML

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"ultralytics/cfg/models/v8"

class TargetedComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_configs_change_only_the_selected_sites(self):
        base=YAML.load(CFG/"yolov8n-fabric-axis-interp.yaml")
        for filename,section,sites in [
            ("yolov8n-fabric-ca-interp.yaml","backbone",[4,6]),
            ("yolov8n-fabric-axis-bilinear.yaml","head",[0,3]),
            ("yolov8n-fabric-axis-carafe-reference.yaml","head",[0,3]),
        ]:
            other=YAML.load(CFG/filename)
            for key in ("nc","scales"): self.assertEqual(other[key],base[key])
            for key in ("backbone","head"):
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

    def test_all_detectors_forward_backward_and_fuse(self):
        for name in ["ca-interp","axis-bilinear","axis-carafe-reference"]:
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
