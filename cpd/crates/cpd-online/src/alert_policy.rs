// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::CpdError;

/// Shared online-alert policy used across detectors.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlertPolicy {
    /// Candidate event when `p_change >= threshold`.
    pub threshold: f64,
    /// Re-arm when `p_change < threshold - hysteresis`.
    pub hysteresis: f64,
    /// Minimum number of processed steps between alert emissions.
    pub cooldown_steps: usize,
    /// Minimum number of processed samples (`t + 1`) before alerting.
    pub min_run_length: usize,
}

impl AlertPolicy {
    pub const fn new(
        threshold: f64,
        hysteresis: f64,
        cooldown_steps: usize,
        min_run_length: usize,
    ) -> Self {
        Self {
            threshold,
            hysteresis,
            cooldown_steps,
            min_run_length,
        }
    }

    pub const fn compatibility(threshold: f64) -> Self {
        Self::new(threshold, 0.0, 0, 0)
    }

    pub fn validate(&self) -> Result<(), CpdError> {
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err(CpdError::invalid_input(format!(
                "alert policy threshold must be finite and in [0,1]; got {}",
                self.threshold
            )));
        }

        if !self.hysteresis.is_finite() || self.hysteresis < 0.0 || self.hysteresis > self.threshold
        {
            return Err(CpdError::invalid_input(format!(
                "alert policy hysteresis must be finite and in [0, threshold]; got {} with threshold {}",
                self.hysteresis, self.threshold
            )));
        }

        Ok(())
    }
}

impl Default for AlertPolicy {
    fn default() -> Self {
        Self::compatibility(0.5)
    }
}

/// Serialized alert-gate runtime state that survives checkpoint/restore.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AlertGateState {
    pub policy: AlertPolicy,
    pub armed: bool,
    pub last_alert_t: Option<usize>,
}

impl AlertGateState {
    pub(crate) fn new(policy: AlertPolicy) -> Self {
        Self {
            policy,
            armed: true,
            last_alert_t: None,
        }
    }

    pub(crate) fn validate(&self, t: usize) -> Result<(), CpdError> {
        self.policy.validate()?;
        if let Some(last_alert_t) = self.last_alert_t
            && last_alert_t >= t
        {
            return Err(CpdError::invalid_input(format!(
                "alert gate last_alert_t={} must be < t={}",
                last_alert_t, t
            )));
        }
        Ok(())
    }

    pub(crate) fn maybe_emit(&mut self, p_change: f64, t: usize) -> bool {
        let rearm_threshold = self.policy.threshold - self.policy.hysteresis;
        if p_change < rearm_threshold {
            self.armed = true;
        }

        let is_candidate = p_change >= self.policy.threshold;
        let cooldown_ok = self
            .last_alert_t
            .is_none_or(|last| t.saturating_sub(last) >= self.policy.cooldown_steps);
        let min_run_ok = t.saturating_add(1) >= self.policy.min_run_length;

        let emit = is_candidate && self.armed && cooldown_ok && min_run_ok;
        if emit {
            self.armed = false;
            self.last_alert_t = Some(t);
        }
        emit
    }
}

impl Default for AlertGateState {
    fn default() -> Self {
        Self::new(AlertPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::{AlertGateState, AlertPolicy};

    #[test]
    fn hysteresis_prevents_flapping_until_rearm_boundary() {
        let mut gate = AlertGateState::new(AlertPolicy::new(0.7, 0.1, 0, 0));
        assert!(
            gate.maybe_emit(0.8, 0),
            "first threshold crossing should alert"
        );
        assert!(
            !gate.maybe_emit(0.72, 1),
            "staying above threshold must not re-alert before rearm"
        );
        assert!(
            !gate.maybe_emit(0.65, 2),
            "must stay disarmed until p_change < threshold - hysteresis"
        );
        assert!(
            !gate.maybe_emit(0.59, 3),
            "drop below rearm threshold should re-arm but not alert on same sample"
        );
        assert!(
            gate.maybe_emit(0.71, 4),
            "after re-arm, next threshold crossing should alert"
        );
    }

    #[test]
    fn cooldown_blocks_duplicate_alerts_between_events() {
        let mut gate = AlertGateState::new(AlertPolicy::new(0.5, 0.0, 3, 0));
        assert!(gate.maybe_emit(0.6, 0));
        assert!(!gate.maybe_emit(0.1, 1));
        assert!(!gate.maybe_emit(0.6, 2), "still in cooldown");
        assert!(!gate.maybe_emit(0.1, 3));
        assert!(gate.maybe_emit(0.6, 4), "cooldown expired");
    }

    #[test]
    fn min_run_length_suppresses_startup_alerts() {
        let mut gate = AlertGateState::new(AlertPolicy::new(0.2, 0.0, 0, 4));
        assert!(!gate.maybe_emit(0.9, 0));
        assert!(!gate.maybe_emit(0.9, 1));
        assert!(!gate.maybe_emit(0.9, 2));
        assert!(!gate.maybe_emit(0.1, 3));
        assert!(gate.maybe_emit(0.9, 4));
    }
}
