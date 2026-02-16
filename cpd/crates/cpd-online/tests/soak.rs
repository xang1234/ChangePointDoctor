// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

#[path = "support/soak_harness.rs"]
mod soak_harness;

use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
use cpd_online::{AlertPolicy, BocpdConfig, BocpdDetector, ConstantHazard, HazardSpec};
use soak_harness::{
    HarnessConfig, SoakProfile, attach_cancellation_quantiles, emit_metrics_json_if_requested,
    enforce_runtime_from_env, measure_cancellation_latency_quantiles_ms, profile_from_env,
    run_soak,
};

fn make_bocpd_detector() -> BocpdDetector {
    BocpdDetector::new(BocpdConfig {
        hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 150.0).expect("valid hazard")),
        max_run_length: 512,
        log_prob_threshold: Some(-30.0),
        alert_policy: AlertPolicy::compatibility(0.55),
        ..BocpdConfig::default()
    })
    .expect("BOCPD config should be valid")
}

fn deterministic_signal(step: usize) -> f64 {
    if (step / 250).is_multiple_of(2) {
        (step as f64 * 0.03).sin() * 0.2
    } else {
        4.0 + (step as f64 * 0.02).cos() * 0.2
    }
}

#[test]
fn soak_checkpoint_restore_equivalence_is_stable() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut baseline = make_bocpd_detector();
    let mut with_roundtrip = make_bocpd_detector();

    const STEPS: usize = 1_000;
    const CHECKPOINT_EVERY: usize = 25;

    for step in 0..STEPS {
        let x = deterministic_signal(step);
        let baseline_step = baseline
            .update(&[x], None, &ctx)
            .expect("baseline update should succeed");
        let roundtrip_step = with_roundtrip
            .update(&[x], None, &ctx)
            .expect("roundtrip update should succeed");

        if (step + 1).is_multiple_of(CHECKPOINT_EVERY) {
            let saved = with_roundtrip.save_state();
            with_roundtrip.load_state(&saved);
        }

        assert!(
            (baseline_step.p_change - roundtrip_step.p_change).abs() < 1e-12,
            "p_change mismatch at step={step}"
        );
        assert_eq!(baseline_step.alert, roundtrip_step.alert);
        assert_eq!(
            baseline_step.run_length_mode,
            roundtrip_step.run_length_mode
        );
    }
}

#[test]
fn soak_checkpoint_restore_roundtrip_metrics_are_reported() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = make_bocpd_detector();

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 1_000,
            checkpoint_every: 50,
            sleep_per_step_ms: 0,
            enforce_target_runtime: false,
            target_runtime_seconds: SoakProfile::PrSmoke.target_runtime_seconds(),
        },
        SoakProfile::PrSmoke,
        &ctx,
    )
    .expect("soak run should succeed");

    assert_eq!(detector.save_state().t, 1_000);
    assert_eq!(metrics.checkpoint_roundtrip_count, 20);
    assert!(metrics.updates_per_sec > 0.0);
}

#[test]
fn soak_reports_alert_stability_markers() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = make_bocpd_detector();

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 2_000,
            checkpoint_every: 20,
            sleep_per_step_ms: 0,
            enforce_target_runtime: false,
            target_runtime_seconds: SoakProfile::PrSmoke.target_runtime_seconds(),
        },
        SoakProfile::PrSmoke,
        &ctx,
    )
    .expect("soak run should succeed");

    assert!(metrics.alert_flip_count > 0);
    assert!(metrics.alert_flip_count < 2_000);
}

#[test]
fn soak_rss_metrics_are_well_formed_when_available() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let mut detector = make_bocpd_detector();

    let metrics = run_soak(
        &mut detector,
        &HarnessConfig {
            steps: 600,
            checkpoint_every: 30,
            sleep_per_step_ms: 0,
            enforce_target_runtime: false,
            target_runtime_seconds: SoakProfile::PrSmoke.target_runtime_seconds(),
        },
        SoakProfile::PrSmoke,
        &ctx,
    )
    .expect("soak run should succeed");

    if let Some(max_rss_kib) = metrics.max_rss_kib {
        assert!(max_rss_kib > 0);
    }
    if let Some(rss_slope_kib_per_hr) = metrics.rss_slope_kib_per_hr {
        assert!(rss_slope_kib_per_hr.is_finite());
    }
}

#[test]
fn soak_profile_gate_metrics_are_emitted() {
    let constraints = Constraints::default();
    let ctx = ExecutionContext::new(&constraints);
    let profile = profile_from_env();
    let mut config = HarnessConfig::for_profile(profile);
    config.enforce_target_runtime = enforce_runtime_from_env();

    let mut detector = make_bocpd_detector();
    let mut metrics =
        run_soak(&mut detector, &config, profile, &ctx).expect("profile soak run should succeed");

    let latency_quantiles = measure_cancellation_latency_quantiles_ms(make_bocpd_detector, 9, 30)
        .expect("cancellation latency measurement should succeed");
    attach_cancellation_quantiles(&mut metrics, &latency_quantiles);

    emit_metrics_json_if_requested("soak_profile_gate_metrics_are_emitted", profile, &metrics)
        .expect("metrics artifact emission should not fail when configured");
}

#[test]
fn soak_profile_runtime_contract_is_stable() {
    assert_eq!(SoakProfile::PrSmoke.target_runtime_seconds(), 120);
    assert_eq!(SoakProfile::Nightly1h.target_runtime_seconds(), 3_600);
    assert_eq!(SoakProfile::Weekly24h.target_runtime_seconds(), 86_400);

    let profile = profile_from_env();
    let config = HarnessConfig::for_profile(profile);
    assert!(config.steps > 0);
    assert!(config.checkpoint_every > 0);
}
