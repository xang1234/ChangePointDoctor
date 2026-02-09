#!/usr/bin/env python3
"""Collect and gate benchmark regressions for CI."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkMetric:
    name: str
    runtime_seconds: float
    max_rss_kib: int


_TIME_ELAPSED_RE = re.compile(r"^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(.+)\s*$")
_MAX_RSS_RE = re.compile(r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$")


def _parse_elapsed_seconds(value: str) -> float:
    raw = value.strip()
    parts = raw.split(":")

    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds

    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"unsupported elapsed time format: {value!r}")


def _extract_metrics(time_output: str) -> tuple[float, int]:
    elapsed_seconds: float | None = None
    max_rss_kib: int | None = None

    for line in time_output.splitlines():
        elapsed_match = _TIME_ELAPSED_RE.match(line)
        if elapsed_match is not None:
            elapsed_seconds = _parse_elapsed_seconds(elapsed_match.group(1))
            continue

        rss_match = _MAX_RSS_RE.match(line)
        if rss_match is not None:
            max_rss_kib = int(rss_match.group(1))
            continue

    if elapsed_seconds is None:
        raise ValueError("missing elapsed wall-clock time in /usr/bin/time -v output")
    if max_rss_kib is None:
        raise ValueError("missing max RSS in /usr/bin/time -v output")

    return elapsed_seconds, max_rss_kib


def _run_benchmark(workspace: Path, bench_name: str) -> BenchmarkMetric:
    command = [
        "/usr/bin/time",
        "-v",
        "cargo",
        "bench",
        "-p",
        "cpd-bench",
        "--bench",
        bench_name,
        "--",
        "--noplot",
    ]

    process = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "benchmark command failed for "
            f"{bench_name} with exit code {process.returncode}\n"
            f"stdout:\n{process.stdout}\n"
            f"stderr:\n{process.stderr}"
        )

    runtime_seconds, max_rss_kib = _extract_metrics(process.stderr)
    return BenchmarkMetric(
        name=bench_name,
        runtime_seconds=runtime_seconds,
        max_rss_kib=max_rss_kib,
    )


def collect_metrics(workspace: Path, benches: list[str]) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for bench in benches:
        metric = _run_benchmark(workspace=workspace, bench_name=bench)
        metrics.append(
            {
                "name": metric.name,
                "runtime_seconds": metric.runtime_seconds,
                "max_rss_kib": metric.max_rss_kib,
            }
        )

    return {
        "metadata": {
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "workspace": str(workspace),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "benchmarks": metrics,
    }


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected JSON object in {path}")
    return parsed


def _extract_benchmark_map(payload: dict[str, Any], label: str) -> dict[str, BenchmarkMetric]:
    data = payload.get("benchmarks")
    if not isinstance(data, list):
        raise ValueError(f"{label} is missing 'benchmarks' list")

    out: dict[str, BenchmarkMetric] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"{label} contains non-object benchmark entry")

        name = item.get("name")
        runtime = item.get("runtime_seconds")
        max_rss = item.get("max_rss_kib")

        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} benchmark entry missing non-empty 'name'")
        if not isinstance(runtime, (int, float)):
            raise ValueError(f"{label} benchmark {name} missing numeric runtime_seconds")
        if not isinstance(max_rss, int):
            raise ValueError(f"{label} benchmark {name} missing integer max_rss_kib")
        if runtime <= 0:
            raise ValueError(f"{label} benchmark {name} has non-positive runtime_seconds")
        if max_rss <= 0:
            raise ValueError(f"{label} benchmark {name} has non-positive max_rss_kib")

        out[name] = BenchmarkMetric(
            name=name,
            runtime_seconds=float(runtime),
            max_rss_kib=max_rss,
        )

    return out


def compare_metrics(
    baseline_payload: dict[str, Any],
    current_payload: dict[str, Any],
    max_runtime_regression_pct: float,
    max_rss_regression_pct: float,
) -> list[str]:
    baseline = _extract_benchmark_map(baseline_payload, "baseline")
    current = _extract_benchmark_map(current_payload, "current")

    failures: list[str] = []

    for bench_name, current_metric in current.items():
        baseline_metric = baseline.get(bench_name)
        if baseline_metric is None:
            failures.append(f"missing baseline metric for benchmark '{bench_name}'")
            continue

        runtime_regression = (
            (current_metric.runtime_seconds - baseline_metric.runtime_seconds)
            / baseline_metric.runtime_seconds
            * 100.0
        )
        rss_regression = (
            (current_metric.max_rss_kib - baseline_metric.max_rss_kib)
            / baseline_metric.max_rss_kib
            * 100.0
        )

        if runtime_regression > max_runtime_regression_pct:
            failures.append(
                f"{bench_name}: runtime regression {runtime_regression:.2f}% exceeds "
                f"threshold {max_runtime_regression_pct:.2f}% "
                f"(baseline={baseline_metric.runtime_seconds:.3f}s, "
                f"current={current_metric.runtime_seconds:.3f}s)"
            )

        if rss_regression > max_rss_regression_pct:
            failures.append(
                f"{bench_name}: RSS regression {rss_regression:.2f}% exceeds "
                f"threshold {max_rss_regression_pct:.2f}% "
                f"(baseline={baseline_metric.max_rss_kib} KiB, "
                f"current={current_metric.max_rss_kib} KiB)"
            )

    return failures


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark collection and regression gate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Run benchmarks and collect metrics.")
    collect_parser.add_argument("--workspace", required=True, help="Workspace directory.")
    collect_parser.add_argument("--out", required=True, help="Output JSON path.")
    collect_parser.add_argument(
        "--bench",
        action="append",
        required=True,
        help="Benchmark name to run. Repeat for multiple benchmarks.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare current metrics to baseline.")
    compare_parser.add_argument("--baseline", required=True, help="Baseline JSON path.")
    compare_parser.add_argument("--current", required=True, help="Current JSON path.")
    compare_parser.add_argument(
        "--max-runtime-regression-pct",
        type=float,
        required=True,
        help="Maximum allowed runtime regression percentage.",
    )
    compare_parser.add_argument(
        "--max-rss-regression-pct",
        type=float,
        required=True,
        help="Maximum allowed RSS regression percentage.",
    )

    args = parser.parse_args(argv)

    try:
        if args.command == "collect":
            workspace = Path(args.workspace).resolve()
            out_path = Path(args.out)
            if not workspace.exists() or not workspace.is_dir():
                raise ValueError(f"workspace does not exist or is not a directory: {workspace}")
            if not Path("/usr/bin/time").exists():
                raise ValueError("/usr/bin/time is required for collect mode")

            payload = collect_metrics(workspace=workspace, benches=args.bench)
            _write_json(out_path, payload)

            print(f"Wrote benchmark metrics to {out_path}")
            for row in payload["benchmarks"]:
                print(
                    f"- {row['name']}: runtime={row['runtime_seconds']:.3f}s, "
                    f"max_rss={row['max_rss_kib']} KiB"
                )
            return 0

        if args.command == "compare":
            baseline_path = Path(args.baseline)
            current_path = Path(args.current)

            baseline_payload = _load_json(baseline_path)
            current_payload = _load_json(current_path)

            failures = compare_metrics(
                baseline_payload=baseline_payload,
                current_payload=current_payload,
                max_runtime_regression_pct=args.max_runtime_regression_pct,
                max_rss_regression_pct=args.max_rss_regression_pct,
            )

            if failures:
                print("BLOCK: benchmark regressions detected:", file=sys.stderr)
                for failure in failures:
                    print(f"  - {failure}", file=sys.stderr)
                return 1

            print("PASS: benchmark regressions are within configured thresholds")
            return 0

        raise ValueError(f"unsupported command: {args.command}")

    except Exception as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
