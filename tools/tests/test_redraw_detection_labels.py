"""Check clean-source redraws and saved-detection integrity without SSH or model inference."""
import copy
import base64
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from unittest.mock import patch

from PIL import Image

from paper.redraw_detection_labels import (
    MODELS, label_rect, load_font, main, overlap, read_bundle, redraw, render_bundle,
)


class RedrawDetectionLabelTests(unittest.TestCase):
    def test_server_defaults_do_not_require_a_local_selection_file(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            source = Path(temp)/"exports"
            with patch("paper.redraw_detection_labels.fetch", return_value={"images": {}}) as fetch_mock, \
                    patch("paper.redraw_detection_labels.render_bundle") as render_mock:
                main(["--source", str(source), "--model-font-size", "36"])
            request, host, _ = fetch_mock.call_args.args
            self.assertIsNone(host)
            self.assertEqual(request["images"], [
                "VFDS66RK26040002_20260801_195949765.jpg.png",
                "VFDS66RK26040002_20260809_022619641.jpg.png",
                "VFDS66RK26040002_20260727_193808156.jpg.png",
                "VFDS66RK26040002_20260805_002859637.jpg.png",
                "VFDS66RK26040002_20260727_200936881.jpg.png",
            ])
            args = render_mock.call_args.args[2]
            self.assertEqual(args.output_dir, source/"paper_figures")
            self.assertEqual(args.model_font_size, 36)

    def test_final_figure_row_order_and_independent_footer_font(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            filenames = ["second.jpg.png", "first.jpg.png"]
            bundle = {"images": {}, "models": {}}
            for filename, color in zip(filenames, ("red", "blue")):
                stream = io.BytesIO()
                Image.new("RGB", (400, 300), color).save(stream, format="PNG")
                content = stream.getvalue()
                bundle["images"][filename] = {"content_base64": base64.b64encode(content).decode(),
                    "sha256": hashlib.sha256(content).hexdigest(), "source": filename, "split": "test"}
            for alias, name in MODELS.items():
                bundle["models"][alias] = {"name": name, "classes": ["row", "col", "hole"],
                    "items": {filename: {"size": [400, 300], "detections": []} for filename in filenames}}
            args = Namespace(work=root/"work", output_dir=root/"figures", font_size=24, font=None,
                             model_font_size=18, model_font=None, no_conf=False, line_width=4, gap=10)
            render_bundle(bundle, filenames, args)
            for suffix in ("", "_baseline"):
                figure = root/"figures"/f"qualitative_large_labels{suffix}.png"
                layout = json.loads(figure.with_suffix(".layout.json").read_text())
                self.assertEqual(layout["rows"], filenames)
                self.assertFalse(layout["panel_letters"])
                self.assertEqual(layout["model_font_size"], 18)
                with Image.open(figure) as image:
                    self.assertEqual(image.getpixel((20, 20)), (255, 0, 0))
                    self.assertEqual(image.getpixel((20, 330)), (0, 0, 255))

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
