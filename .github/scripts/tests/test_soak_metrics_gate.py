import json
import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import soak_metrics_gate  # noqa: E402


def _thresholds_payload():
    return {
        "schema_version": 1,
        "profiles": {
            "nightly_1h": {
                "min_updates_per_sec": 1000.0,
                "max_cancellation_latency_p95_ms": 2000,
                "max_rss_kib": 2_500_000,
                "max_rss_slope_kib_per_hr": 50_000.0,
            },
            "weekly_24h": {
                "min_updates_per_sec": 250.0,
                "max_cancellation_latency_p95_ms": 2000,
                "max_rss_kib": 3_000_000,
                "max_rss_slope_kib_per_hr": 15_000.0,
            },
        },
    }


def _metrics_payload(**overrides):
    payload = {
        "scenario": "soak_profile_gate_metrics_are_emitted",
        "profile": "nightly_1h",
        "target_runtime_seconds": 3600,
        "updates_per_sec": 1200.0,
        "cancellation_latency_ms": 150,
        "cancellation_latency_p50_ms": 100,
        "cancellation_latency_p95_ms": 150,
        "checkpoint_roundtrip_count": 100,
        "max_rss_kib": 800_000,
        "rss_slope_kib_per_hr": 3_000.0,
        "alert_flip_count": 200,
    }
    payload.update(overrides)
    return payload


class SoakMetricsGateTests(unittest.TestCase):
    def test_load_thresholds_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.json"
            path.write_text(
                json.dumps({"schema_version": 2, "profiles": {}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema_version"):
                soak_metrics_gate.load_thresholds(path)

    def test_load_metrics_falls_back_to_legacy_cancellation_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            payload = _metrics_payload(cancellation_latency_p95_ms=None)
            path.write_text(json.dumps(payload), encoding="utf-8")

            metrics = soak_metrics_gate.load_metrics(path)
            self.assertEqual(metrics.cancellation_latency_p95_ms, 150)

    def test_evaluate_metrics_passes_when_within_thresholds(self):
        thresholds = soak_metrics_gate.SoakThresholds(
            min_updates_per_sec=1000.0,
            max_cancellation_latency_p95_ms=2000,
            max_rss_kib=2_500_000,
            max_rss_slope_kib_per_hr=50_000.0,
        )
        metrics = soak_metrics_gate.SoakMetrics(
            profile="nightly_1h",
            updates_per_sec=1200.0,
            cancellation_latency_p95_ms=150,
            max_rss_kib=800_000,
            rss_slope_kib_per_hr=3_000.0,
        )

        failures = soak_metrics_gate.evaluate_metrics(metrics, thresholds)
        self.assertEqual(failures, [])

    def test_evaluate_metrics_reports_each_threshold_breach(self):
        thresholds = soak_metrics_gate.SoakThresholds(
            min_updates_per_sec=1000.0,
            max_cancellation_latency_p95_ms=2000,
            max_rss_kib=2_500_000,
            max_rss_slope_kib_per_hr=50_000.0,
        )
        metrics = soak_metrics_gate.SoakMetrics(
            profile="nightly_1h",
            updates_per_sec=800.0,
            cancellation_latency_p95_ms=3000,
            max_rss_kib=3_500_000,
            rss_slope_kib_per_hr=80_000.0,
        )

        failures = soak_metrics_gate.evaluate_metrics(metrics, thresholds)
        self.assertEqual(len(failures), 4)

    def test_main_blocks_when_profile_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            thresholds_path = Path(tmp) / "thresholds.json"
            metrics_path = Path(tmp) / "metrics.json"
            thresholds_path.write_text(json.dumps(_thresholds_payload()), encoding="utf-8")
            metrics_path.write_text(
                json.dumps(_metrics_payload(profile="unknown_profile")), encoding="utf-8"
            )

            exit_code = soak_metrics_gate.main(
                [
                    "--thresholds",
                    str(thresholds_path),
                    "--metrics",
                    str(metrics_path),
                ]
            )
            self.assertEqual(exit_code, 1)

    def test_main_passes_with_valid_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            thresholds_path = Path(tmp) / "thresholds.json"
            metrics_path = Path(tmp) / "metrics.json"
            thresholds_path.write_text(json.dumps(_thresholds_payload()), encoding="utf-8")
            metrics_path.write_text(json.dumps(_metrics_payload()), encoding="utf-8")

            exit_code = soak_metrics_gate.main(
                [
                    "--thresholds",
                    str(thresholds_path),
                    "--metrics",
                    str(metrics_path),
                ]
            )
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
