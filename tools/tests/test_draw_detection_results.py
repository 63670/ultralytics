"""Verify full-dataset PNG-only drawing and one-time baseline reuse."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from tools.draw_detection_results import main


class DrawDetectionResultsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dataset = self.root / "data"
        for split in ("train", "val", "test"):
            images = self.dataset / "images" / split / "nested"
            images.mkdir(parents=True)
            Image.new("RGB", (120, 80), "white").save(images / "one.jpg")
        self.common = ["--dataset", str(self.dataset), "--device", "cpu"]

    def test_own_weights_draw_every_split_without_a_manifest(self):
        weights = self.root / "run" / "weights" / "best.pt"
        weights.parent.mkdir(parents=True)
        weights.touch()
        extra = self.dataset / "images" / "train" / "nested" / "one.png"
        Image.new("RGB", (120, 80), "white").save(extra)
        records = [{"class_id": 0, "xyxy": [15, 25, 65, 55], "score": 0.8}]
        predictor = Mock(return_value=records)
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("tools.draw_detection_results.build_ultralytics_predictor", return_value=(predictor, ["defect"])):
                self.assertEqual(main(["--weights", str(weights), *self.common]), 0)
        output = weights.parent.parent / "detection-results"
        rendered = sorted(path for path in output.rglob("*") if path.is_file())
        self.assertEqual(len(rendered), 4)
        self.assertEqual({path.suffix for path in rendered}, {".png"})
        self.assertTrue((output / "train" / "nested" / "one.png.png").is_file())
        with Image.open(output / "test" / "nested" / "one.jpg.png") as image:
            self.assertEqual(image.size, (120, 80))
            self.assertEqual(image.getpixel((15, 55)), (255, 68, 68))
        self.assertEqual(predictor.call_count, 4)

    def test_baselines_resume_missing_images_and_exclude_experimental_entries(self):
        weights = self.root / "baseline.pt"
        weights.touch()
        manifest = self.root / "models.json"
        manifest.write_text(json.dumps({
            "classes": ["row", "col", "hole"],
            "models": [
                {"name": "Public", "backend": "ultralytics", "weights": str(weights)},
                {"name": "Proposed", "backend": "ultralytics", "baseline": False},
            ],
        }), encoding="utf-8")
        output = self.root / "baselines"
        args = ["--baselines", "--manifest", str(manifest), "--output", str(output), *self.common]
        predictor = Mock(return_value=[])
        with contextlib.redirect_stdout(io.StringIO()):
            with patch("tools.draw_detection_results.build_predictor", return_value=(predictor, "mock")) as build:
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
                build.assert_called_once()
                self.assertEqual(predictor.call_count, 3)

                missing = output / "Public" / "val" / "nested" / "one.jpg.png"
                missing.unlink()
                self.assertEqual(main(args), 0)
                self.assertEqual(predictor.call_count, 4)
                self.assertEqual(main([*args, "--overwrite"]), 0)
                self.assertEqual(predictor.call_count, 7)
        rendered = sorted(path for path in output.rglob("*") if path.is_file())
        self.assertEqual(len(rendered), 3)
        self.assertEqual({path.suffix for path in rendered}, {".png"})
        self.assertTrue(missing.is_file())
        self.assertFalse((output / "Proposed").exists())

    def test_baseline_backends_can_run_in_separate_environments(self):
        manifest = self.root / "models.json"
        manifest.write_text(json.dumps({
            "classes": ["row"],
            "models": [{"name": "DETR", "backend": "mmdet"}, {"name": "RT-DETR", "backend": "rtdetrv2"}],
        }), encoding="utf-8")
        with patch("tools.draw_detection_results.subprocess.run", return_value=Mock(returncode=0)) as run:
            self.assertEqual(main([
                "--baselines", "--manifest", str(manifest), "--output", str(self.root / "results"),
                "--backend-env", "mmdet=mmdet", "--backend-env", "rtdetrv2=rtdetr", "--overwrite",
                *self.common,
            ]), 0)
        self.assertEqual(run.call_count, 2)
        for call, backend, environment in zip(run.call_args_list, ("mmdet", "rtdetrv2"), ("mmdet", "rtdetr")):
            command = call.args[0]
            self.assertEqual(command[1:6], ["run", "--no-capture-output", "-n", environment, "python"])
            self.assertEqual(command[command.index("--backend") + 1], backend)
            self.assertIn("--overwrite", command)
            self.assertNotIn("--backend-env", command)
