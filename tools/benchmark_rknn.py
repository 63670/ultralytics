#!/usr/bin/env python3
"""Evaluate an existing RKNN model on a connected RK3566, without application postprocessing."""
from __future__ import annotations

import argparse
import hashlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    """Validate arguments before importing the optional Rockchip SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Existing .rknn model compiled for the target.")
    parser.add_argument("--target", default="rk3566", choices=["rk3566", "rk3568", "rk3588", "rk3576"])
    parser.add_argument("--device-id", help="Connected device ID; omit when only one board is connected.")
    parser.add_argument("--warmup", type=int, default=1, help="Discard this many eval_perf calls (default: 1).")
    parser.add_argument("--repeat", type=int, default=5, help="Retain this many eval_perf reports (default: 5).")
    parser.add_argument("--profile", action="store_true", help="Enable per-layer profiling in a separate diagnostic run.")
    parser.add_argument("--output", type=Path, help="Save SDK-returned performance reports as UTF-8 text.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose SDK logs.")
    args = parser.parse_args(argv)
    if not args.model.is_file() or args.model.suffix.lower() != ".rknn":
        parser.error("--model must be an existing .rknn file (not an ONNX file or an export directory).")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("--warmup must be >= 0 and --repeat must be >= 1.")
    if args.output and args.output.resolve() == args.model.resolve():
        parser.error("--output must not overwrite --model.")
    return args


def check_return(ret, operation: str) -> None:
    """Stop if the SDK could not load the model or connect to the board."""
    if ret != 0:
        raise RuntimeError(f"{operation} failed (return code: {ret}).")


def run_benchmark(args: argparse.Namespace, rknn_factory) -> str:
    """Collect the SDK's device-side timing, never timing the PC-to-board API call."""
    try:
        toolkit_version = version("rknn-toolkit2")
    except PackageNotFoundError:
        toolkit_version = "unknown"
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    header = (
        f"Model: {args.model.resolve()}\nSHA256: {digest}\nToolkit2: {toolkit_version}\n"
        f"Target: {args.target}\nDevice: {args.device_id or 'SDK auto-selection'}\n"
        f"perf_debug: {args.profile}\nDiscarded eval_perf calls: {args.warmup}\n"
        f"Retained eval_perf calls: {args.repeat}\n"
        "Timing source: RKNN eval_perf device-side report; not host wall-clock time.\n"
    )
    print(header, flush=True)
    if args.profile:
        print("Diagnostic profiling is enabled. Use a separate run without --profile for timing comparisons.")
    reports = [header]
    rknn = rknn_factory(verbose=args.verbose)
    try:
        check_return(rknn.load_rknn(str(args.model)), "load_rknn")
        runtime = {"target": args.target, "perf_debug": args.profile}
        if args.device_id:
            runtime["device_id"] = args.device_id
        check_return(rknn.init_runtime(**runtime), "init_runtime")
        # eval_perf runs the model internally; it needs neither images nor inference(inputs=...).
        # Counts here are API calls, not a promise about the SDK's internal number of frames.
        for index in range(args.warmup + args.repeat):
            retained = index >= args.warmup
            label = f"Report {index - args.warmup + 1}/{args.repeat}" if retained else "Warmup (discarded)"
            print(f"--> {label}", flush=True)
            report = rknn.eval_perf(is_print=retained)
            if not isinstance(report, str) or not report.strip():
                raise RuntimeError(f"eval_perf did not return a performance report: {report!r}")
            if retained:
                reports.append(f"\n--- {label} ---\n{report}\n")
    finally:
        rknn.release()
    result = "".join(reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Saved SDK reports to {args.output}")
    return result


def main() -> int:
    args = parse_args()
    try:
        from rknn.api import RKNN
    except ImportError:
        print("RKNN-Toolkit2 is required on the PC host. See tools/benchmark_rknn.md for setup.", file=sys.stderr)
        return 1
    try:
        run_benchmark(args, RKNN)
    except (OSError, RuntimeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
