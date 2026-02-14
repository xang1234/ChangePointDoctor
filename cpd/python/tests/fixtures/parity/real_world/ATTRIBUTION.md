# Real-World Slice Attribution

These fixtures are small, vendored snippets shaped after public CPD benchmark patterns
(mean shifts, trend breaks, and heteroskedastic regimes). They are stored directly in-repo
for offline and reproducible CI parity checks.

## Provenance

- `public_benchmark_slice_01.json` to `public_benchmark_slice_06.json`
  - Origin: benchmark-inspired synthetic derivatives curated for CI stability.
  - Purpose: deterministic regression slices for `cpd-rs` vs `ruptures` parity.
  - License note: marked as CC-BY-compatible synthetic derivatives; no upstream raw data copied.

## Usage Contract

- Fixtures are immutable inputs for parity tests under
  `/Users/admin/Documents/Work/claude-doctor-changepoint/cpd/python/tests/test_ruptures_parity.py`.
- Any fixture update must preserve deterministic behavior and include rationale in PR notes.
