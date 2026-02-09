import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_gate  # noqa: E402


def _payload(entries):
    return {
        "metadata": {"source": "unit-test"},
        "benchmarks": entries,
    }


def _entry(name, runtime_seconds, max_rss_kib):
    return {
        "name": name,
        "runtime_seconds": runtime_seconds,
        "max_rss_kib": max_rss_kib,
    }


class BenchmarkGateTests(unittest.TestCase):
    def test_compare_pass_within_thresholds(self):
        baseline = _payload([_entry("offline_pelt", 10.0, 1000)])
        current = _payload([_entry("offline_pelt", 10.9, 1140)])

        failures = benchmark_gate.compare_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )

        self.assertEqual(failures, [])

    def test_compare_fails_runtime_regression(self):
        baseline = _payload([_entry("offline_pelt", 10.0, 1000)])
        current = _payload([_entry("offline_pelt", 11.1, 1000)])

        failures = benchmark_gate.compare_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("runtime regression", failures[0])

    def test_compare_fails_rss_regression(self):
        baseline = _payload([_entry("offline_pelt", 10.0, 1000)])
        current = _payload([_entry("offline_pelt", 10.0, 1200)])

        failures = benchmark_gate.compare_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("RSS regression", failures[0])

    def test_compare_fails_on_missing_baseline_bench(self):
        baseline = _payload([_entry("offline_pelt", 10.0, 1000)])
        current = _payload([
            _entry("offline_pelt", 10.0, 1000),
            _entry("offline_binseg", 10.0, 1000),
        ])

        failures = benchmark_gate.compare_metrics(
            baseline_payload=baseline,
            current_payload=current,
            max_runtime_regression_pct=10.0,
            max_rss_regression_pct=15.0,
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("missing baseline metric", failures[0])

    def test_main_compare_missing_baseline_file_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            current_path = tmp_path / "current.json"
            current_path.write_text(
                json.dumps(_payload([_entry("offline_pelt", 10.0, 1000)])),
                encoding="utf-8",
            )

            with io.StringIO() as stderr_buffer, redirect_stderr(stderr_buffer):
                exit_code = benchmark_gate.main(
                    [
                        "compare",
                        "--baseline",
                        str(tmp_path / "does-not-exist.json"),
                        "--current",
                        str(current_path),
                        "--max-runtime-regression-pct",
                        "10",
                        "--max-rss-regression-pct",
                        "15",
                    ]
                )

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
