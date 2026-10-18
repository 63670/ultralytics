"""Check GT coordinates, class alignment and complete per-image comparison exports without a GPU."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.export_detection_comparison import class_mapping, gt_boxes, main, predictions


class DetectionComparisonTests(unittest.TestCase):
    def test_gt_coordinates_and_empty_background(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"labels.txt"
            path.write_text("1 0.5 0.25 0.2 0.1\n", encoding="utf-8")
            box = gt_boxes(path, (1000, 800), ["row", "col", "hole"])[0]
            self.assertEqual(box["xyxy"], [400, 160, 600, 240])
            self.assertEqual(box["class_id"], 1)
            path.write_text("")
            self.assertEqual(gt_boxes(path, (1000, 800), ["row"]), [])
            with self.assertRaises(FileNotFoundError):
                gt_boxes(path.parent/"missing.txt", (1000, 800), ["row"])

    def test_mapping_threshold_order_and_empty_detections(self):
        mapping = class_mapping({0: "hole", 1: "row", 2: "col"}, ["row", "col", "hole"])
        self.assertEqual(mapping, [2, 0, 1])
        rows = predictions([[1, 2, 3, 4]]*3, [0.2, 0.8, 0.5], [0, 1, 2], mapping, 0.25, 1)
        self.assertEqual([(r["class_id"], r["score"]) for r in rows], [(0, 0.8)])
        self.assertEqual(predictions([], [], [], mapping, 0.25, 100), [])
        with self.assertRaises(ValueError):
            class_mapping(["cat"], ["row"])

    def test_reject_nonfinite_and_bad_annotations(self):
        with self.assertRaises(ValueError):
            predictions([[0, 0, float("nan"), 1]], [0.9], [0], [0], 0.25, 100)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"bad.txt"
            for text in ("0 .5 .5 2 .1", "0 .5 .5 .1 .1 .9", "2 .5 .5 .1 .1"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    gt_boxes(path, (100, 100), ["row"])

    def test_export_all_splits_and_compose_including_no_detections(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset, output = root/"data", root/"output"
            for split in ("train", "val", "test"):
                images, labels = dataset/"images"/split, dataset/"labels"/split
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                Image.new("RGB", (120, 80), "gray").save(images/"one.jpg")
                (labels/"one.txt").write_text("0 .5 .5 .4 .2\n")
            weights = root/"fake.pt"
            weights.write_bytes(b"fake checkpoint")
            manifest = root/"models.json"
            manifest.write_text(json.dumps({"classes": ["row", "col", "hole"], "models": [
                {"name": "Empty", "backend": "ultralytics", "weights": str(weights)}]}))
            common = ["--dataset", str(dataset), "--output", str(output), "--manifest", str(manifest)]
            main(["--backend", "gt"]+common)
            with patch("tools.export_detection_comparison.build_predictor", return_value=(lambda _: [], "fake")):
                main(["--backend", "ultralytics"]+common)
            main(["--backend", "compose", "--panel-width", "120"]+common)
            for split in ("train", "val", "test"):
                self.assertTrue((output/"Empty"/split/"one.jpg.png").is_file())
                data = json.loads((output/"Empty"/split/"one.jpg.json").read_text())
                self.assertEqual(data["detections"], [])
                with Image.open(output/"comparison"/split/"one.jpg.png") as image:
                    self.assertEqual(image.size, (240, 120))
            (output/"Empty"/"test"/"one.jpg.png").unlink()
            with self.assertRaises(FileNotFoundError):
                main(["--backend", "compose", "--splits", "test"]+common)
            Image.new("RGB", (120, 80), "white").save(dataset/"images"/"train"/"one.jpg")
            with self.assertRaises(ValueError):
                main(["--backend", "compose", "--splits", "train"]+common)


if __name__ == "__main__":
    unittest.main()
