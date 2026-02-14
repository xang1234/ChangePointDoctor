# Reproducibility Modes and Deterministic Contracts

This document defines the reproducibility contract for `ReproMode` and how to
interpret expected drift across runs, environments, and platforms.

## Scope

- Applies to offline detection APIs and diagnostics metadata.
- Covers `strict`, `balanced` (default), and `fast` modes.
- Covers both current implementation behavior and intended long-term contract.

## Mode Contract Matrix

| Mode | Guaranteed | Not guaranteed | Performance impact | When to use |
| --- | --- | --- | --- | --- |
| `strict` | Target: bitwise-identical results on the same machine, toolchain, and CPU feature set. Deterministic numeric accumulation paths are used in supported cost models. | Bitwise identity across different target triples/architectures/compilers. | Highest overhead of the three modes due to deterministic numeric reductions. | Regulated or audit-heavy workflows requiring maximum run-to-run reproducibility. |
| `balanced` (default) | Deterministic breakpoint sets on the same target triple with the same inputs/configuration and execution settings. | Bitwise identity across platforms or compiler/toolchain upgrades. | Baseline production profile. | Default for most production workloads. |
| `fast` | Throughput-first execution with valid outputs and practical segmentation quality. | Deterministic floating-point ordering and bitwise-identical scores. | Lowest overhead, highest throughput target. | Large-scale or latency-sensitive workloads where small numeric drift is acceptable. |

## Strict Mode

`strict` is the highest reproducibility setting. The goal is bitwise-identical
outputs when all of the following match:

- machine and architecture,
- toolchain and build settings,
- available CPU features.

Current implementation status:

- `CostL2Mean` and `CostNormalMeanVar` use compensated/Kahan prefix accumulation
  under `ReproMode::Strict`.
- This is the deterministic-numerics path currently implemented in code for
  strict mode.

Tradeoff:

- strict mode may run slower because deterministic compensated reductions are
  more expensive than non-compensated accumulation.

## Balanced Mode (Default)

`balanced` is the default (`ExecutionContext` and Python API defaults).

Contract:

- On the same target triple, with the same effective config and execution
  settings, breakpoint sets are expected to be stable.

Current implementation status:

- For current MVP-A offline detectors (`pelt`, `binseg`), code paths are
  deterministic and do not use RNG-based branching.
- `balanced` and `fast` currently share the same numeric accumulation path in
  implemented cost models; they are separated as contract modes for future
  behavior/performance evolution.

## Fast Mode

`fast` is the throughput-oriented mode.

Contract:

- Allows implementation strategies that can change floating-point operation
  ordering and hardware-specific numeric behavior while preserving practical
  segmentation quality.

Current implementation status:

- `fast` is accepted and propagated in diagnostics and API surfaces.
- In current MVP-A implementations, `fast` and `balanced` behave similarly in
  the core numeric path; future implementations may diverge to prioritize speed.

## Per-Cost-Model Score Tolerances

For exact-change-point matches in parity validation, score agreement currently
uses this relative-error gate:

| Cost model | Score tolerance (exact-change-point cases) |
| --- | --- |
| `l2` | `<= 1e-6` relative error |
| `normal` | `<= 1e-6` relative error |

Breakpoint agreement is evaluated separately using tolerance-adjusted
segmentation criteria:

- default breakpoint tolerance: `±2` indices,
- tolerant pass-rate thresholds:
  - `pelt:l2 >= 0.95`
  - `binseg:l2 >= 0.90`
  - `pelt:normal >= 0.90`
  - `binseg:normal >= 0.90`

These thresholds are used for cross-implementation/platform comparability
instead of bitwise-equal floating-point scores.

## Cross-Platform Reproducibility

Bitwise-identical results across platforms are generally unrealistic due to:

- floating-point instruction and rounding differences,
- SIMD feature differences,
- compiler/codegen differences,
- math library/backend differences.

What is guaranteed instead:

- contract-level comparability through breakpoint tolerance and segmentation
  agreement thresholds,
- explicit diagnostics metadata to help explain environment-dependent drift.

## Configuration Examples

Python:

```python
import cpd

result_strict = cpd.detect_offline(x, detector="pelt", cost="l2", repro_mode="strict")
result_balanced = cpd.detect_offline(x, detector="pelt", cost="l2", repro_mode="balanced")
result_fast = cpd.detect_offline(x, detector="pelt", cost="l2", repro_mode="fast")
```

Rust:

```rust
use cpd_core::{Constraints, ExecutionContext, ReproMode};

let constraints = Constraints::default();
let ctx_strict = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Strict);
let ctx_balanced = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Balanced);
let ctx_fast = ExecutionContext::new(&constraints).with_repro_mode(ReproMode::Fast);
```

## FAQ

### Why did my results change when I upgraded?

Check these first:

- `engine_version` in diagnostics (runtime/library version changed),
- `repro_mode` (mode changed from strict/balanced/fast),
- algorithm/cost model combination (`algorithm`, `cost_model`),
- target/platform/toolchain differences (for example ARM vs x86, compiler
  updates, BLAS/runtime differences).

Even with the same mode, upgrades can change low-level numerics or optimization
paths. Compare breakpoints with tolerance-based criteria when evaluating
cross-version drift.

### Why do I get different scores on ARM vs x86?

This is expected for floating-point-heavy workloads. ARM and x86 can differ in:

- available SIMD instructions,
- codegen decisions,
- floating-point reduction ordering.

Treat cross-platform comparability primarily as a segmentation agreement problem
using tolerance-adjusted breakpoint checks and documented parity thresholds,
rather than exact score bitwise identity.
