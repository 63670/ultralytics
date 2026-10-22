# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Exercise the experiment shell runner with a fake YOLO CLI, without training or downloading weights."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.sh"
MOCK_YOLO = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\t' "$@" >> "$MOCK_LOG"
printf '\n' >> "$MOCK_LOG"
project='' name='' seed=''
for argument in "$@"; do
  case "$argument" in
    project=*) project="${argument#project=}" ;;
    name=*) name="${argument#name=}" ;;
    seed=*) seed="${argument#seed=}" ;;
  esac
done
if [[ "$2" == train ]]; then
  if [[ -n "${MOCK_FAIL_TRAIN_SEED:-}" && "$seed" == "$MOCK_FAIL_TRAIN_SEED" ]]; then
    exit 17
  fi
  mkdir -p "$project/$name/weights"
  touch "$project/$name/weights/last.pt" "$project/$name/weights/epoch1.pt"
  if [[ "${MOCK_NO_BEST:-0}" != 1 ]]; then
    touch "$project/$name/weights/best.pt"
  fi
else
  if [[ "${MOCK_FAIL_VAL:-0}" == 1 ]]; then
    exit 23
  fi
  mkdir -p "$project/$name"
  printf 'mock test metrics\n'
fi
"""


def bash_path(path):
    """Express a native path for Bash, including Git Bash's drive mount convention."""
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" and path.drive else value


class TestRunExperiment(unittest.TestCase):
    """Validate argument routing, independent seed runs, and checkpoint safety."""

    @classmethod
    def setUpClass(cls):
        """Find native Bash or Git Bash, avoiding the legacy Windows WSL launcher."""
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        cls.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash is required for shell-runner tests")

    def setUp(self):
        """Create isolated output directories and a replacement YOLO executable."""
        self.temp = tempfile.TemporaryDirectory(prefix="yolo-runner-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "mock bin"
        self.bin_dir.mkdir()
        fake_yolo = self.bin_dir / "yolo"
        fake_yolo.write_bytes(MOCK_YOLO.encode("utf-8"))
        fake_yolo.chmod(0o755)
        self.runs = self.root / "runs with spaces"
        self.log = self.root / "calls.tsv"
        self.env = os.environ.copy()
        self.env.update(
            RUNS_DIR=bash_path(self.runs),
            DATA_YAML=bash_path(self.root / "fabric data.yaml"),
            MOCK_BIN=bash_path(self.bin_dir),
            MOCK_LOG=bash_path(self.log),
        )

    def run_script(self, *args, **extra_env):
        """Invoke the real runner while intercepting its training and validation commands."""
        env = {**self.env, **extra_env}
        return subprocess.run(
            [
                self.bash,
                "--noprofile",
                "--norc",
                "-c",
                'export PATH="$MOCK_BIN:/usr/bin:$PATH"; exec "$BASH" "$@"',
                "runner-test",
                bash_path(SCRIPT),
                *args,
            ],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

    def calls(self):
        """Read individual argument arrays recorded by the replacement CLI."""
        if not self.log.exists():
            return []
        return [line.rstrip("\t").split("\t") for line in self.log.read_text(encoding="utf-8").splitlines()]

    def assert_success(self, result):
        """Include shell output when a run fails unexpectedly."""
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_multiple_seeds(self):
        """Each seed trains and tests its own checkpoint and keeps separate logs."""
        result = self.run_script("--device", "1", "--seeds", "0,1,2", "yolov10n.pt", "yolov10n")
        self.assert_success(result)
        calls = self.calls()
        self.assertEqual([call[:2] for call in calls], [["detect", action] for action in ["train", "val"] * 3])
        for index, seed in enumerate(["0", "1", "2"]):
            train, val = calls[index * 2 : index * 2 + 2]
            run_dir = self.runs / "yolov10n" / seed
            self.assertIn(f"project={bash_path(self.runs / 'yolov10n')}", train)
            self.assertIn(f"name={seed}", train)
            self.assertIn(f"seed={seed}", train)
            self.assertIn("device=1", train)
            self.assertIn(f"model={bash_path(run_dir / 'weights' / 'best.pt')}", val)
            self.assertIn(f"project={bash_path(run_dir)}", val)
            self.assertIn("split=test", val)
            self.assertTrue((run_dir / "test").is_dir())
            self.assertTrue((run_dir / "test_metrics.log").is_file())
            self.assertEqual([path.name for path in (run_dir / "weights").iterdir()], ["best.pt"])
            self.assertIn("last.pt", (run_dir / "deleted_checkpoints.log").read_text(encoding="utf-8"))

    def test_legacy_seed_argument(self):
        """Existing positional commands and trailing seed= still use the legacy directory."""
        result = self.run_script("yolov8n.pt", "legacy/seed0", "seed=0", "epochs=1")
        self.assert_success(result)
        train, val = self.calls()
        self.assertIn("name=legacy/seed0", train)
        self.assertIn("seed=0", train)
        self.assertIn("epochs=1", train)
        self.assertIn("device=0", train)
        self.assertIn(f"project={bash_path(self.runs / 'legacy' / 'seed0')}", val)

    def test_single_seed_and_shared_overrides(self):
        """--seed creates the nested layout and shared options apply to train and test."""
        result = self.run_script(
            "--seed", "7", "--device", "0", "yolov10n.pt", "model",
            "batch=8", "imgsz=320", "data=other.yaml", "workers=2", "max_det=20", "device=cpu",
        )
        self.assert_success(result)
        train, val = self.calls()
        self.assertIn("seed=7", train)
        for argument in ["batch=8", "imgsz=320", "data=other.yaml", "workers=2", "max_det=20", "device=cpu"]:
            self.assertIn(argument, train)
            self.assertIn(argument, val)
        self.assertTrue((self.runs / "model" / "7" / "weights" / "best.pt").is_file())

    def test_preflight_rejects_existing_later_seed(self):
        """A conflict at any requested seed is detected before training the first one."""
        existing = self.runs / "model" / "1"
        existing.mkdir(parents=True)
        marker = existing / "keep.txt"
        marker.write_text("user data", encoding="utf-8")
        result = self.run_script("--seeds", "0,1,2", "yolov10n.pt", "model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to overwrite", result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.runs / "model" / "0").exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), "user data")

    def test_invalid_arguments(self):
        """Malformed lists, conflicting seed sources, and output overrides fail before training."""
        cases = [
            ("--seeds", "0,0", "model.pt", "model"),
            ("--seeds", "0,", "model.pt", "model"),
            ("--seeds", "-1", "model.pt", "model"),
            ("--seeds", "00", "model.pt", "model"),
            ("--seeds", "4294967296", "model.pt", "model"),
            ("--seed", "0,1", "model.pt", "model"),
            ("--seeds", "0", "--seed", "1", "model.pt", "model"),
            ("--seeds", "0,1", "model.pt", "model", "seed=2"),
            ("--device",),
            ("--seeds",),
            ("--unknown", "0", "model.pt", "model"),
            ("model.pt", "model", "project=other"),
            ("model.pt", "model", "name=other"),
            ("model.pt", "model", "resume=True"),
            ("model.pt", "model", "--seed", "0"),
            ("model.pt", "../outside"),
        ]
        for args in cases:
            with self.subTest(args=args):
                result = self.run_script(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertEqual(self.calls(), [])

    def test_train_failure_stops_remaining_seeds(self):
        """Training failures stop the sequence, without cleaning later or failed runs."""
        result = self.run_script(
            "--seeds", "0,1,2", "model.pt", "model", MOCK_FAIL_TRAIN_SEED="1"
        )
        self.assertEqual(result.returncode, 17)
        self.assertEqual(
            [call[:2] for call in self.calls()],
            [["detect", "train"], ["detect", "val"], ["detect", "train"]],
        )
        self.assertFalse((self.runs / "model" / "2").exists())

    def test_test_failure_preserves_checkpoints(self):
        """Do not delete last.pt or proceed to another seed when testing fails."""
        result = self.run_script("--seeds", "0,1", "model.pt", "model", MOCK_FAIL_VAL="1")
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.calls()), 2)
        self.assertTrue((self.runs / "model" / "0" / "weights" / "last.pt").is_file())
        self.assertFalse((self.runs / "model" / "1").exists())

    def test_missing_best_checkpoint(self):
        """A nominally successful training run must produce best.pt before testing or cleanup."""
        result = self.run_script("--seed", "0", "model.pt", "model", MOCK_NO_BEST="1")
        self.assertEqual(result.returncode, 1)
        self.assertIn("best checkpoint was not found", result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertTrue((self.runs / "model" / "0" / "weights" / "last.pt").is_file())

    def test_help(self):
        """Document the multi-seed layout and existing checkpoint cleanup policy."""
        result = self.run_script("--help")
        self.assert_success(result)
        self.assertIn("--seeds 0,1,2", result.stdout)
        self.assertIn("last.pt is deleted", result.stdout)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
