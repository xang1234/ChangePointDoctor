# `cpd-rs` vs `ruptures` Parity Contract

This document defines the MVP-B differential parity suite for issue `CPD-l15.4`.

## Scope

The parity suite compares `cpd-rs` and `ruptures` on a curated corpus and evaluates:

- breakpoint exact and tolerance-adjusted agreement,
- segmentation-cost agreement (for exact breakpoint matches),
- runtime telemetry (informational only).

## Corpus Layout

Manifest path:
`/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python/tests/fixtures/parity/corpus_manifest.v1.json`

Category minimums (current corpus has 56 total cases):

- `synthetic_step`: 18
- `synthetic_volatility`: 8
- `heavy_tailed`: 8
- `autocorrelated`: 6
- `trending`: 6
- `missing_gap`: 4
- `real_world_subset`: 6

Real-world vendored slices live at:
`/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python/tests/fixtures/parity/real_world/`

Attribution metadata:
`/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python/tests/fixtures/parity/real_world/ATTRIBUTION.md`

## Tolerance Rules

- Default breakpoint tolerance is `±2` indices.
- `exact_match`: exact equality of `change_points`.
- `tolerant_match`: one-to-one matching under tolerance for all breakpoints.
- `tolerant_jaccard`: tolerance-adjusted Jaccard similarity.

## Cost Parity Rule

Independent evaluator (not detector internals):

- `l2`: sum of segment SSE against segment mean.
- `normal`: `m * ln(max(var, eps_floor))` per segment, with
  `eps_floor = np.finfo(np.float64).eps * 1e6`.

Gate:

- for exact-match cases only, relative error must be `<= 1e-6`.
- tolerant-only matches report cost deltas but do not fail on cost.

## Missing-Data Contract

Missing-gap cases are compared after deterministic preprocessing in the parity harness:

1. forward-fill interior NaNs,
2. backward-fill leading NaNs,
3. fallback remaining NaNs to `0.0` (degenerate all-NaN defense).

This suite does **not** validate native missing-policy APIs.

## CI Profiles

- `smoke` profile (PR): representative subset with fast runtime.
- `full` profile (nightly): full 56-case corpus with artifact report.

Environment variables:

- `CPD_PARITY_PROFILE=smoke|full`
- `CPD_PARITY_REPORT_OUT=/path/to/report.json` (optional)

## Thresholds

Tolerant pass-rate gates:

- `pelt:l2` >= `0.95`
- `binseg:l2` >= `0.90`
- `pelt:normal` >= `0.90`
- `binseg:normal` >= `0.90`
