# BOCPD Online Performance SLOs (MVP-C)

This document defines the BOCPD online performance contract for `CPD-45f.1`.

## Protected Scenario

- Detector: `BocpdDetector`
- Observation model: Gaussian (`ObservationModel::default()`)
- Dimensionality: `d=1`
- Hazard: constant (`p_change = 1/200`)
- State bound: `max_run_length = 2000`
- Pruning: enabled (`log_prob_threshold = -20.0`)
- Warmup updates: `2500`
- Measured updates: `12000`
- Signal profile: deterministic multi-regime signal with smooth intra-regime wobble and regular regime shifts every `96` points (representative steady-state production mix)

## SLO Thresholds

- `p99_update_us <= 75`
- `updates_per_sec >= 150000`
- `p95_run_length_mode >= 64` (occupancy guard to avoid degenerate always-reset traces)

Definitions:

- `p99_update_us` is the p99 of per-update `processing_latency_us` emitted by `OnlineStepResult`.
- `updates_per_sec` is measured as `measured_updates / elapsed_wall_time_seconds` over the measurement window.

## Local Commands

Observe-only run (never fails on SLO):

```bash
cd cpd
cargo test -p cpd-online --test perf bocpd_gaussian_perf_contract -- --exact --nocapture
```

Enforced run (fails if SLOs are violated):

```bash
cd cpd
CPD_ONLINE_PERF_ENFORCE=1 cargo test -p cpd-online --test perf bocpd_gaussian_perf_contract --release -- --exact --nocapture
```

Emit JSON metrics artifact:

```bash
cd cpd
CPD_ONLINE_PERF_METRICS_OUT=/tmp/bocpd-perf.json cargo test -p cpd-online --test perf bocpd_gaussian_perf_contract -- --exact --nocapture
```

## CI Enforcement Policy

SLO enforcement is nightly-only. The nightly workflow runs the perf scenario in release mode with:

- `CPD_ONLINE_PERF_ENFORCE=1`
- `CPD_ONLINE_PERF_METRICS_OUT=<artifact path>`

PR checks remain non-blocking for this SLO to reduce platform-driven performance flakiness while still preserving nightly regression protection.
