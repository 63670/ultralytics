"""Check timing accounting and input validation without requiring CUDA or model downloads."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tools.benchmark_torch_fps import digest, image_files, measure, parse_args


class FakeCuda:
    def __init__(self):
        self.clock = 0.0
        self.syncs = 0

    def synchronize(self):
        self.syncs += 1

    def Event(self, enable_timing):
        assert enable_timing
        cuda = self

        class Event:
            def record(self):
                self.value = cuda.clock

            def elapsed_time(self, other):
                return other.value - self.value

        return Event()


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.inference = False

    @contextlib.contextmanager
    def inference_mode(self):
        self.inference = True
        try:
            yield
        finally:
            self.inference = False


class TorchFpsTests(unittest.TestCase):
    def test_warmup_excluded_and_inputs_cycled_in_inference_mode(self):
        torch = FakeTorch()
        seen = []

        def forward(duration):
            self.assertTrue(torch.inference)
            seen.append(duration)
            torch.cuda.clock += duration

        result = measure(forward, [2.0, 4.0], warmup=5, iterations=4, repeat=3, torch=torch)
        self.assertEqual(seen, [2.0, 4.0] * 8 + [2.0])
        self.assertEqual(result["timed_forwards"], 12)
        self.assertEqual(result["round_mean_ms"], [3.0, 3.0, 3.0])
        self.assertAlmostEqual(result["fps"], 1000 / 3)
        self.assertEqual(torch.cuda.syncs, 4)
        self.assertFalse(torch.inference)

    def test_nonpositive_or_nonfinite_timing_rejected(self):
        for duration in (0, float("nan"), float("inf")):
            torch = FakeTorch()

            def forward(_):
                torch.cuda.clock += duration

            with self.subTest(duration=duration), self.assertRaises(RuntimeError):
                measure(forward, [0], 0, 1, 1, torch)

    def test_missing_inputs_or_invalid_counts_rejected(self):
        for batches, warmup, iterations, repeat in (([], 1, 1, 1), ([0], -1, 1, 1), ([0], 0, 0, 1)):
            with self.assertRaises(ValueError):
                measure(lambda _: None, batches, warmup, iterations, repeat, FakeTorch())

    def test_images_have_deterministic_order_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("b.jpg", "a.png", "c.JPG", "labels.txt"):
                (root / name).write_bytes(b"test")
            self.assertEqual([p.name for p in image_files(root, 2)], ["a.png", "b.jpg"])
            self.assertEqual(digest(root / "a.png"), digest(root / "b.jpg"))

    def test_protocol_defaults_and_argument_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "weights.pt"
            checkpoint.touch()
            base = ["--weights", str(checkpoint), "--images", str(root)]
            args = parse_args(base)
            self.assertEqual((args.imgsz, args.warmup, args.iterations, args.repeat), (640, 50, 200, 5))
            self.assertTrue(args.fuse)
            self.assertFalse(parse_args(base + ["--no-fuse"]).fuse)
            for extra in (["--warmup", "-1"], ["--backend", "mmdet"], ["--output", str(checkpoint)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args(base + extra)


if __name__ == "__main__":
    unittest.main()
