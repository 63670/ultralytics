"""Prevent costly YAML training runs with accidentally unspecified initial weights."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run_experiment.sh"
MOCK = r'''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$MOCK_LOG"
project='' name=''
for arg in "$@"; do
  case "$arg" in
    project=*) project="${arg#project=}" ;;
    name=*) name="${arg#name=}" ;;
  esac
done
mkdir -p "$project/$name/weights"
touch "$project/$name/weights/best.pt"
'''


def shell_path(path):
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" and path.drive else value


class ExperimentPretrainedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        cls.bash = str(candidate) if os.name == "nt" and candidate.is_file() else shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash is required")

    def invoke(self, model, *args):
        with tempfile.TemporaryDirectory(prefix="yolo-pretrained-") as directory:
            root = Path(directory)
            executable = root / "yolo"
            executable.write_bytes(MOCK.encode())
            executable.chmod(0o755)
            log = root / "calls.txt"
            env = {**os.environ, "MOCK_LOG": shell_path(log), "MOCK_BIN": shell_path(root),
                   "RUNS_DIR": shell_path(root / "runs")}
            result = subprocess.run(
                [self.bash, "--noprofile", "--norc", "-c",
                 'export PATH="$MOCK_BIN:/usr/bin:$PATH"; exec "$BASH" "$@"',
                 "runner", shell_path(SCRIPT), "--seed", "0", model, "experiment", *args],
                env=env, capture_output=True, text=True, encoding="utf-8", timeout=30,
            )
            return result, log.read_text(encoding="utf-8") if log.exists() else ""

    def test_yaml_without_explicit_weights_never_launches_training(self):
        for args in ((), ("pretrained=True",), ("pretrained=true",), ("pretrained=",)):
            with self.subTest(args=args):
                result, calls = self.invoke("custom.yaml", *args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("YAML models require", result.stderr)
                self.assertEqual(calls, "")

    def test_yaml_explicit_checkpoint_reaches_training_only(self):
        result, calls = self.invoke("custom.yaml", "pretrained=yolov8n.pt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        train, val = calls.splitlines()
        self.assertIn("pretrained=yolov8n.pt", train)
        self.assertNotIn("pretrained=", val)

    def test_yaml_allows_intentional_random_initialization(self):
        result, calls = self.invoke("custom.yaml", "pretrained=False")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("pretrained=False", calls.splitlines()[0])

    def test_checkpoint_entry_remains_supported(self):
        result, calls = self.invoke("yolov8n.pt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls.splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
