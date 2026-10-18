"""Guard sample matching and pixel preservation in paper comparison composition."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from paper.compose_detection_figure import collect_rows, main


class ComposeDetectionFigureTests(unittest.TestCase):
    def test_missing_or_extra_images_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("GT", "ours"):
                (root/name).mkdir()
                Image.new("RGB", (64, 40)).save(root/name/"one.png")
            Image.new("RGB", (64, 40)).save(root/"ours"/"two.png")
            with self.assertRaises(ValueError):
                collect_rows(root, ["GT", "ours"])
            self.assertEqual(collect_rows(root, ["GT", "ours"], ["one.png"]), ["one.png"])
            with self.assertRaises(ValueError):
                collect_rows(root, ["GT", "ours"], ["absent.png"])

    def test_pixels_order_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            for name in ("GT", "ours"):
                (root/name).mkdir()
                for filename, color in (("a.png", "red"), ("b.png", "blue")):
                    Image.new("RGB", (160, 100), color).save(root/name/filename)
            output = root/"final.png"
            main(["--input", str(root), "--output", str(output), "--columns", "ours", "GT",
                  "--images", "b.png", "a.png", "--font-size", "12", "--gap", "10"])
            with Image.open(output) as image:
                self.assertEqual(image.size, (330, 246))
                self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
                self.assertEqual(image.getpixel((170, 110)), (255, 0, 0))
                self.assertEqual(image.getpixel((165, 5)), (255, 255, 255))
            metadata = json.loads(output.with_suffix(".layout.json").read_text())
            self.assertEqual(metadata["rows"], ["b.png", "a.png"])
            self.assertEqual(metadata["columns"], ["ours", "GT"])
            self.assertEqual(len(metadata["sources"]), 4)


if __name__ == "__main__":
    unittest.main()
