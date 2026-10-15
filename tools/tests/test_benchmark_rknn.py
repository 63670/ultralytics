"""Exercise connected-board benchmark control flow without requiring an RKNN SDK or board."""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SPEC = importlib.util.spec_from_file_location("benchmark_rknn", Path(__file__).parents[1] / "benchmark_rknn.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    """Check device selection, report handling, cleanup and input protection."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "model.rknn"
        self.model.write_bytes(b"test model, not an executable RKNN graph")
        self.factory = MagicMock()
        self.runtime = self.factory.return_value
        self.runtime.load_rknn.return_value = 0
        self.runtime.init_runtime.return_value = 0

    def args(self, *extra):
        return benchmark.parse_args(["--model", str(self.model), *extra])

    def test_device_timing_reports_exclude_warmup(self):
        output = Path(self.temp.name) / "results" / "perf.txt"
        self.runtime.eval_perf.side_effect = ["discard me", "Total Time(us): 62500", "Total Time(us): 63000"]
        args = self.args("--repeat", "2", "--device-id", "board-123", "--output", str(output))
        with patch("builtins.print"):
            report = benchmark.run_benchmark(args, self.factory)
        self.runtime.init_runtime.assert_called_once_with(target="rk3566", perf_debug=False, device_id="board-123")
        self.assertEqual(self.runtime.eval_perf.call_args_list[0].kwargs, {"is_print": False})
        self.assertEqual(self.runtime.eval_perf.call_args_list[1].kwargs, {"is_print": True})
        self.assertNotIn("discard me", report)
        self.assertIn("Total Time(us): 62500", report)
        self.assertEqual(output.read_text(encoding="utf-8"), report)
        self.runtime.inference.assert_not_called()
        self.runtime.build.assert_not_called()
        self.runtime.release.assert_called_once()

    def test_profile_is_explicit(self):
        self.runtime.eval_perf.return_value = "per-layer performance report"
        with patch("builtins.print"):
            benchmark.run_benchmark(self.args("--profile", "--warmup", "0", "--repeat", "1"), self.factory)
        self.runtime.init_runtime.assert_called_once_with(target="rk3566", perf_debug=True)

    def test_connection_failure_stops_and_releases(self):
        self.runtime.init_runtime.return_value = -1
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "init_runtime failed"):
            benchmark.run_benchmark(self.args(), self.factory)
        self.runtime.eval_perf.assert_not_called()
        self.runtime.release.assert_called_once()

    def test_failed_evaluation_is_not_saved_as_success(self):
        output = Path(self.temp.name) / "perf.txt"
        self.runtime.eval_perf.return_value = None
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "performance report"):
            benchmark.run_benchmark(self.args("--output", str(output)), self.factory)
        self.assertFalse(output.exists())
        self.runtime.release.assert_called_once()

    def test_output_cannot_overwrite_model(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            self.args("--output", str(self.model))
        self.assertTrue(self.model.read_bytes().startswith(b"test model"))


if __name__ == "__main__":
    unittest.main()
