#!/usr/bin/env python3
"""Gate BOCPD soak metrics artifacts against profile thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SoakThresholds:
    min_updates_per_sec: float | None
    max_cancellation_latency_p95_ms: int | None
    max_rss_kib: int | None
    max_rss_slope_kib_per_hr: float | None


@dataclass(frozen=True)
class SoakMetrics:
    profile: str
    updates_per_sec: float
    cancellation_latency_p95_ms: int | None
    max_rss_kib: int | None
    rss_slope_kib_per_hr: float | None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _parse_optional_positive_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not float(value) > 0.0:
        raise ValueError(f"{label} must be a positive number when set; got {value!r}")
    return float(value)


def _parse_optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer when set; got {value!r}")
    return value


def load_thresholds(path: Path) -> dict[str, SoakThresholds]:
    payload = _load_json(path)

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version={schema_version!r}; expected 1")

    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("thresholds file requires non-empty 'profiles' object")

    out: dict[str, SoakThresholds] = {}
    for profile, raw_thresholds in raw_profiles.items():
        if not isinstance(profile, str) or not profile:
            raise ValueError(f"invalid profile key: {profile!r}")
        if not isinstance(raw_thresholds, dict):
            raise ValueError(f"profile {profile!r} thresholds must be an object")

        out[profile] = SoakThresholds(
            min_updates_per_sec=_parse_optional_positive_float(
                raw_thresholds.get("min_updates_per_sec"),
                f"profiles.{profile}.min_updates_per_sec",
            ),
            max_cancellation_latency_p95_ms=_parse_optional_positive_int(
                raw_thresholds.get("max_cancellation_latency_p95_ms"),
                f"profiles.{profile}.max_cancellation_latency_p95_ms",
            ),
            max_rss_kib=_parse_optional_positive_int(
                raw_thresholds.get("max_rss_kib"),
                f"profiles.{profile}.max_rss_kib",
            ),
            max_rss_slope_kib_per_hr=_parse_optional_positive_float(
                raw_thresholds.get("max_rss_slope_kib_per_hr"),
                f"profiles.{profile}.max_rss_slope_kib_per_hr",
            ),
        )

    return out


def load_metrics(path: Path) -> SoakMetrics:
    payload = _load_json(path)

    profile = payload.get("profile")
    if not isinstance(profile, str) or not profile:
        raise ValueError("metrics payload requires non-empty 'profile'")

    updates_per_sec = payload.get("updates_per_sec")
    if not isinstance(updates_per_sec, (int, float)) or not float(updates_per_sec) > 0.0:
        raise ValueError("metrics payload requires positive numeric 'updates_per_sec'")

    cancellation_latency_p95_ms = payload.get("cancellation_latency_p95_ms")
    if cancellation_latency_p95_ms is None:
        cancellation_latency_p95_ms = payload.get("cancellation_latency_ms")

    if cancellation_latency_p95_ms is not None:
        if (
            not isinstance(cancellation_latency_p95_ms, int)
            or cancellation_latency_p95_ms < 0
        ):
            raise ValueError(
                "metrics 'cancellation_latency_p95_ms' (or fallback 'cancellation_latency_ms') "
                f"must be a non-negative integer; got {cancellation_latency_p95_ms!r}"
            )

    max_rss_kib = payload.get("max_rss_kib")
    if max_rss_kib is not None and (not isinstance(max_rss_kib, int) or max_rss_kib <= 0):
        raise ValueError(f"metrics 'max_rss_kib' must be positive int or null; got {max_rss_kib!r}")

    rss_slope_kib_per_hr = payload.get("rss_slope_kib_per_hr")
    if rss_slope_kib_per_hr is not None and (
        not isinstance(rss_slope_kib_per_hr, (int, float))
        or not float(rss_slope_kib_per_hr) == float(rss_slope_kib_per_hr)
    ):
        raise ValueError(
            "metrics 'rss_slope_kib_per_hr' must be numeric or null; "
            f"got {rss_slope_kib_per_hr!r}"
        )

    return SoakMetrics(
        profile=profile,
        updates_per_sec=float(updates_per_sec),
        cancellation_latency_p95_ms=cancellation_latency_p95_ms,
        max_rss_kib=max_rss_kib,
        rss_slope_kib_per_hr=float(rss_slope_kib_per_hr)
        if rss_slope_kib_per_hr is not None
        else None,
    )


def evaluate_metrics(metrics: SoakMetrics, thresholds: SoakThresholds) -> list[str]:
    failures: list[str] = []

    if (
        thresholds.min_updates_per_sec is not None
        and metrics.updates_per_sec < thresholds.min_updates_per_sec
    ):
        failures.append(
            "updates_per_sec below threshold: "
            f"observed={metrics.updates_per_sec:.3f}, "
            f"min={thresholds.min_updates_per_sec:.3f}"
        )

    if thresholds.max_cancellation_latency_p95_ms is not None:
        observed = metrics.cancellation_latency_p95_ms
        if observed is None:
            failures.append("missing cancellation latency metric for configured p95 threshold")
        elif observed > thresholds.max_cancellation_latency_p95_ms:
            failures.append(
                "cancellation_latency_p95_ms above threshold: "
                f"observed={observed}, max={thresholds.max_cancellation_latency_p95_ms}"
            )

    if thresholds.max_rss_kib is not None:
        observed = metrics.max_rss_kib
        if observed is None:
            failures.append("missing max_rss_kib metric for configured threshold")
        elif observed > thresholds.max_rss_kib:
            failures.append(
                "max_rss_kib above threshold: "
                f"observed={observed}, max={thresholds.max_rss_kib}"
            )

    if thresholds.max_rss_slope_kib_per_hr is not None:
        observed = metrics.rss_slope_kib_per_hr
        if observed is None:
            failures.append("missing rss_slope_kib_per_hr metric for configured threshold")
        elif observed > thresholds.max_rss_slope_kib_per_hr:
            failures.append(
                "rss_slope_kib_per_hr above threshold: "
                f"observed={observed:.3f}, max={thresholds.max_rss_slope_kib_per_hr:.3f}"
            )

    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate BOCPD soak metrics against profile thresholds.")
    parser.add_argument(
        "--thresholds",
        required=True,
        help="Path to soak thresholds JSON file.",
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to soak metrics JSON artifact.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Profile override (defaults to profile in metrics artifact).",
    )

    args = parser.parse_args(argv)

    try:
        thresholds_by_profile = load_thresholds(Path(args.thresholds))
        metrics = load_metrics(Path(args.metrics))

        profile = args.profile or metrics.profile
        thresholds = thresholds_by_profile.get(profile)
        if thresholds is None:
            known = ", ".join(sorted(thresholds_by_profile))
            raise ValueError(
                f"profile {profile!r} not present in thresholds file (known: {known})"
            )

        failures = evaluate_metrics(metrics, thresholds)
        if failures:
            print("BLOCK: soak metric thresholds violated:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1

        print(f"PASS: soak metrics satisfy thresholds for profile {profile}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCK: failed to evaluate soak metrics: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
