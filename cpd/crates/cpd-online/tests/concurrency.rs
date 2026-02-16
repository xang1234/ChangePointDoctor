// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[path = "support/soak_harness.rs"]
mod soak_harness;

use cpd_core::{BudgetMode, Constraints, ExecutionContext, OnlineDetector};
use cpd_online::{AlertPolicy, BocpdConfig, BocpdDetector, ConstantHazard, HazardSpec};
use soak_harness::{
    HarnessConfig, SoakProfile, measure_cancellation_latency_quantiles_ms, run_soak,
};
use std::thread;

fn make_bocpd_detector() -> BocpdDetector {
    BocpdDetector::new(BocpdConfig {
        hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 120.0).expect("valid hazard")),
        max_run_length: 512,
        log_prob_threshold: Some(-30.0),
        alert_policy: AlertPolicy::compatibility(0.55),
        ..BocpdConfig::default()
    })
    .expect("BOCPD config should be valid")
}

#[test]
fn multi_instance_threaded_runs_keep_state_isolated() {
    const THREADS: usize = 4;
    let config = HarnessConfig {
        steps: 800,
        checkpoint_every: 50,
        sleep_per_step_ms: 0,
        enforce_target_runtime: false,
        target_runtime_seconds: SoakProfile::PrSmoke.target_runtime_seconds(),
    };

    let mut workers = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        let config = config.clone();
        workers.push(thread::spawn(move || {
            let constraints = Constraints::default();
            let ctx = ExecutionContext::new(&constraints);
            let mut detector = make_bocpd_detector();
            let metrics = run_soak(&mut detector, &config, SoakProfile::PrSmoke, &ctx)
                .expect("threaded BOCPD soak run should succeed");

            (detector.save_state().t, metrics)
        }));
    }

    for worker in workers {
        let (updates_seen, metrics) = worker.join().expect("thread should join cleanly");
        assert_eq!(updates_seen, config.steps);
        assert_eq!(
            metrics.checkpoint_roundtrip_count,
            config.steps / config.checkpoint_every
        );
        assert!(metrics.updates_per_sec > 0.0);
    }
}

#[test]
fn concurrent_cancellation_latency_is_bounded() {
    let latencies = measure_cancellation_latency_quantiles_ms(make_bocpd_detector, 9, 30)
        .expect("cancellation latency measurements should succeed");

    assert!(latencies.p50_ms <= latencies.p95_ms);
    assert!(
        latencies.p95_ms <= 2_000,
        "expected prompt cancellation, got p95={}ms",
        latencies.p95_ms
    );
}

#[test]
fn concurrent_budget_enforcement_is_deterministic() {
    const THREADS: usize = 4;
    const BUDGET_LIMIT: usize = 50;

    let mut workers = Vec::with_capacity(THREADS);
    for _ in 0..THREADS {
        workers.push(thread::spawn(move || {
            let constraints = Constraints {
                max_cost_evals: Some(BUDGET_LIMIT),
                ..Constraints::default()
            };
            let ctx = ExecutionContext::new(&constraints).with_budget_mode(BudgetMode::HardFail);
            let mut detector = make_bocpd_detector();

            let mut ok_updates = 0usize;
            loop {
                match detector.update(&[ok_updates as f64], None, &ctx) {
                    Ok(_) => ok_updates += 1,
                    Err(err) => return (ok_updates, err.to_string()),
                }
            }
        }));
    }

    for worker in workers {
        let (ok_updates, err_msg) = worker.join().expect("worker should join cleanly");
        assert_eq!(ok_updates, BUDGET_LIMIT);
        assert!(
            err_msg.contains("constraints.max_cost_evals exceeded"),
            "unexpected budget error: {err_msg}"
        );
    }
}
