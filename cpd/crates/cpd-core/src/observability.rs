// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Optional callback for reporting algorithm progress in `[0.0, 1.0]`.
pub trait ProgressSink: Send + Sync {
    fn on_progress(&self, fraction: f32);
}

/// Optional sink for low-overhead scalar telemetry.
pub trait TelemetrySink: Send + Sync {
    fn record_scalar(&self, key: &'static str, value: f64);
}
