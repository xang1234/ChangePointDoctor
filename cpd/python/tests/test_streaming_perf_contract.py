import json
import os
import time
from pathlib import Path
from statistics import median

import numpy as np

import cpd

PERF_BATCH_SIZES = (1, 8, 16, 64, 4096)
WARMUP_ROUNDS = 4
MEASURE_ROUNDS = 12
OUTLIER_FACTOR = 4.0
OUTLIER_RETRY_WARMUP_ROUNDS = 2
OUTLIER_RETRY_MEASURE_ROUNDS = 8
PERF_ENFORCE_ENV = "CPD_PY_STREAMING_PERF_ENFORCE"
PERF_REPORT_ENV = "CPD_PY_STREAMING_PERF_REPORT_OUT"


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.lower() in {"1", "true", "yes", "on"}


def _step_signal(n: int) -> np.ndarray:
    values = np.zeros(n, dtype=np.float64)
    values[n // 2 :] = 4.0
    return values


def _collect_pair_timings(
    *, batch_size: int, warmup_rounds: int, measure_rounds: int
) -> tuple[list[float], list[float]]:
    values = _step_signal(batch_size)
    single_ms: list[float] = []
    batch_ms: list[float] = []
    total_rounds = warmup_rounds + measure_rounds
    for idx in range(total_rounds):
        detector = cpd.Cusum(threshold=8.0, alert_policy={"threshold": 0.95})
        t0 = time.perf_counter()
        for x_t in values:
            detector.update(float(x_t))
        t1 = time.perf_counter()

        detector = cpd.Cusum(threshold=8.0, alert_policy={"threshold": 0.95})
        t2 = time.perf_counter()
        detector.update_many(values)
        t3 = time.perf_counter()

        if idx >= warmup_rounds:
            single_ms.append((t1 - t0) * 1_000.0)
            batch_ms.append((t3 - t2) * 1_000.0)
    return single_ms, batch_ms


def _has_outlier(samples: list[float]) -> bool:
    med = median(samples)
    if med <= 0.0:
        return False
    return max(samples) > (med * OUTLIER_FACTOR)


def _measure_pair(batch_size: int) -> dict:
    single_ms, batch_ms = _collect_pair_timings(
        batch_size=batch_size,
        warmup_rounds=WARMUP_ROUNDS,
        measure_rounds=MEASURE_ROUNDS,
    )

    retried_for_outlier = False
    if _has_outlier(single_ms) or _has_outlier(batch_ms):
        retried_for_outlier = True
        extra_single, extra_batch = _collect_pair_timings(
            batch_size=batch_size,
            warmup_rounds=OUTLIER_RETRY_WARMUP_ROUNDS,
            measure_rounds=OUTLIER_RETRY_MEASURE_ROUNDS,
        )
        single_ms.extend(extra_single)
        batch_ms.extend(extra_batch)

    single_median_ms = median(single_ms)
    batch_median_ms = median(batch_ms)
    speedup = single_median_ms / batch_median_ms
    return {
        "batch_size": batch_size,
        "single_median_ms": single_median_ms,
        "batch_median_ms": batch_median_ms,
        "speedup_vs_single": speedup,
        "single_rounds": len(single_ms),
        "batch_rounds": len(batch_ms),
        "retried_for_outlier": retried_for_outlier,
    }


def test_update_many_perf_contract_snapshot() -> None:
    enforce = _env_flag(PERF_ENFORCE_ENV)
    rows = [_measure_pair(size) for size in PERF_BATCH_SIZES]
    by_size = {int(row["batch_size"]): row for row in rows}
    report = {
        "scenario": "python_streaming_update_vs_update_many",
        "detector": "Cusum",
        "metric": "median_ms",
        "warmup_rounds": WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
        "outlier_factor": OUTLIER_FACTOR,
        "outlier_retry_warmup_rounds": OUTLIER_RETRY_WARMUP_ROUNDS,
        "outlier_retry_measure_rounds": OUTLIER_RETRY_MEASURE_ROUNDS,
        "results": rows,
        "enforce": enforce,
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    report_out = os.getenv(PERF_REPORT_ENV)
    if report_out:
        Path(report_out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    # Always-on checks: broad sanity for crossover behavior.
    assert by_size[1]["speedup_vs_single"] < 0.8
    assert by_size[8]["speedup_vs_single"] < 1.05
    assert by_size[16]["speedup_vs_single"] > 1.0
    assert by_size[64]["speedup_vs_single"] > 1.05
    assert by_size[4096]["speedup_vs_single"] > 1.2

    # Optional stricter gates for local perf validation runs.
    if enforce:
        assert by_size[1]["speedup_vs_single"] < 0.6
        assert by_size[8]["speedup_vs_single"] < 1.0
        assert by_size[16]["speedup_vs_single"] > 1.05
        assert by_size[64]["speedup_vs_single"] > 1.2
        assert by_size[4096]["speedup_vs_single"] > 1.4
