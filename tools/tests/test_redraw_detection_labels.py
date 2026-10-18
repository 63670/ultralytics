"""Check clean-source redraws and saved-detection integrity without SSH or model inference."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper.redraw_detection_labels import label_rect, load_font, overlap, read_bundle, redraw


class RedrawDetectionLabelTests(unittest.TestCase):
    def test_auto_split_read_and_source_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root/"sample.jpg"
            Image.new("RGB", (100, 80), "gray").save(original)
            record = {"image": str(original), "size": [100, 80],
                      "image_sha256": hashlib.sha256(original.read_bytes()).hexdigest(), "detections": []}
            for model in ("GT", "YOLOv8n"):
                target = root/model/"train"
                target.mkdir(parents=True)
                (target/"sample.jpg.json").write_text(json.dumps(record))
                (target.parent/"manifest.json").write_text(json.dumps({"classes": ["row", "col", "hole"]}))
            request = {"source": str(root), "models": {"GT": "GT", "yolov8n": "YOLOv8n"},
                       "images": ["sample.jpg.png"], "split": "auto"}
            bundle = read_bundle(request)
            self.assertEqual(bundle["images"]["sample.jpg.png"]["split"], "train")
            self.assertEqual(len(bundle["models"]), 2)
            original.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                read_bundle(request)

    def test_text_scores_and_box_records_preserved(self):
        image = Image.new("RGB", (400, 300), (100, 100, 100))
        records = [{"class_id": 0, "xyxy": [30.2, 80.1, 160.4, 100.7], "score": .847},
                   {"class_id": 0, "xyxy": [32.3, 82.1, 170.8, 103.5], "score": None}]
        saved = copy.deepcopy(records)
        result, labels = redraw(image, records, ["row"], load_font(24))
        self.assertEqual(records, saved)
        self.assertEqual([r["text"] for r in labels], ["row 0.85", "row"])
        self.assertEqual(result.getpixel((399, 299)), image.getpixel((399, 299)))
        self.assertEqual(overlap(labels[0]["rectangle"], labels[1]["rectangle"]), 0)
        _, hidden = redraw(image, records, ["row"], load_font(24), show_conf=False)
        self.assertEqual([r["text"] for r in hidden], ["row", "row"])

    def test_edge_placement_stays_inside_image(self):
        occupied = []
        for _ in range(4):
            rect = label_rect((170, 0, 190, 199), (100, 30), (200, 200), occupied, [])
            self.assertTrue(0 <= rect[0] < rect[2] <= 200)
            self.assertTrue(0 <= rect[1] < rect[3] <= 200)
            self.assertTrue(all(overlap(rect, other) == 0 for other in occupied))
            occupied.append(rect)


if __name__ == "__main__":
    unittest.main()
