"""Standalone regression tests: python -m unittest scripts.test_split_hybrid_dataset."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.split_hybrid_dataset import load_source, prepare_dataset, read_json, split_records


class HybridSplitTests(unittest.TestCase):
    """Check matching exports, background retention, reproducibility and input validation."""

    def setUp(self):
        """Build ten images including one background absent from COCO and duplicate annotation IDs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "output"
        for directory in ("images", "yolo_labels", "coco_labels"):
            (self.source / directory).mkdir(parents=True)
        (self.source / "classes.txt").write_text("row\nhole\n", encoding="utf-8")
        coco = {
            "info": {"description": "test"},
            "categories": [{"id": 9, "name": "hole"}, {"id": 3, "name": "row"}],
            "images": [], "annotations": [],
        }
        for index in range(10):
            name = f"sample_{index}.png"
            Image.new("RGB", (100, 80), (index, 0, 0)).save(self.source / "images" / name)
            (self.source / "yolo_labels" / f"sample_{index}.txt").write_text(
                "0 0.300000 0.437500 0.400000 0.375000\n" if index else "", encoding="utf-8"
            )
            if index:
                coco["images"].append({"id": index, "file_name": name, "width": 100, "height": 80})
                coco["annotations"].append({
                    "id": 0, "image_id": index, "category_id": 3,
                    "bbox": [10, 20, 40, 30], "area": 1200, "iscrowd": 0, "segmentation": [],
                })
        self.coco_path = self.source / "coco_labels" / "coco_detection.json"
        self.coco_path.write_text(json.dumps(coco), encoding="utf-8")

    def test_matching_exports(self):
        """Every image and box has the same content and split in both output formats."""
        summary = prepare_dataset(self.source, self.output)
        self.assertEqual([part["images"] for part in summary["splits"].values()], [7, 1, 2])
        seen, annotation_ids = set(), set()
        for split, part in summary["splits"].items():
            coco = read_json(self.output / "coco" / "annotations" / f"instances_{split}2017.json")
            self.assertEqual([category["name"] for category in coco["categories"]], ["row", "hole"])
            names = {image["file_name"] for image in coco["images"]}
            self.assertEqual(names, set(part["files"]))
            self.assertFalse(names & seen)
            seen.update(names)
            annotations = {annotation["image_id"]: annotation for annotation in coco["annotations"]}
            for image in coco["images"]:
                name = image["file_name"]
                original = (self.source / "images" / name).read_bytes()
                for path in (self.output / "yolo" / "images" / split / name,
                             self.output / "coco" / f"{split}2017" / name):
                    self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), hashlib.sha256(original).digest())
                text = (self.output / "yolo" / "labels" / split / Path(name).with_suffix(".txt")).read_text()
                if name == "sample_0.png":
                    self.assertEqual(text, "")
                    self.assertNotIn(image["id"], annotations)
                else:
                    row = list(map(float, text.split()))
                    annotation = annotations[image["id"]]
                    self.assertEqual(row[0], 0)
                    self.assertEqual(annotation["category_id"], 3)
                    cx, cy, width, height = row[1:]
                    box = [(cx - width / 2) * 100, (cy - height / 2) * 80, width * 100, height * 80]
                    for actual, expected in zip(box, annotation["bbox"]):
                        self.assertAlmostEqual(actual, expected)
                    self.assertNotIn(annotation["id"], annotation_ids)
                    annotation_ids.add(annotation["id"])
        self.assertEqual(len(seen), 10)
        self.assertEqual(len(annotation_ids), 9)
        self.assertEqual(read_json(self.output / "split_manifest.json"), summary)

    def test_reproducible_split_and_rounding(self):
        """The same seed repeats exactly; integer allocations cover small and full-size datasets."""
        records = [{"name": str(index)} for index in range(534)]
        first = split_records(records, 42)
        self.assertEqual(first, split_records(records, 42))
        self.assertNotEqual(first, split_records(records, 43))
        self.assertEqual([len(part) for part in first.values()], [374, 53, 107])
        for count in range(30):
            self.assertEqual(sum(map(len, split_records(records[:count], 42).values())), count)

    def test_refuses_overwrite(self):
        """Existing output is preserved intact."""
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.source, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_mismatching_labels_fail_before_export(self):
        """A box disagreement cannot silently produce different annotations."""
        (self.source / "yolo_labels" / "sample_1.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        with self.assertRaisesRegex(ValueError, "labels disagree"):
            prepare_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_label_is_not_a_background(self):
        """Only explicit empty files qualify as negative images."""
        (self.source / "yolo_labels" / "sample_0.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            load_source(self.source)

    def test_nonempty_label_absent_from_coco(self):
        """Unmatched positive images fail instead of being silently dropped."""
        (self.source / "yolo_labels" / "sample_0.txt").write_text("0 0.3 0.4375 0.4 0.375\n")
        with self.assertRaisesRegex(ValueError, "no matching COCO"):
            load_source(self.source)

    def test_invalid_dimensions_and_boxes(self):
        """Source dimensions and bounding boxes must describe the actual image."""
        original = read_json(self.coco_path)
        for field, value, message in (("width", 99, "dimensions"), ("bbox", [90, 20, 40, 30], "outside")):
            coco = json.loads(json.dumps(original))
            target = coco["images"][0] if field == "width" else coco["annotations"][0]
            target[field] = value
            self.coco_path.write_text(json.dumps(coco))
            with self.assertRaisesRegex(ValueError, message):
                load_source(self.source)

    def test_rejects_overlapping_source_and_output(self):
        """An output inside the source cannot be picked up as input on a later run."""
        with self.assertRaisesRegex(ValueError, "contain each other"):
            prepare_dataset(self.source, self.source / "output")


if __name__ == "__main__":
    unittest.main()
