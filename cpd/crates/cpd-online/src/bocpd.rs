// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{
    CpdError, ExecutionContext, OnlineDetector, OnlineStepResult, log_add_exp, log_sum_exp,
};
use std::f64::consts::PI;
use std::time::Instant;

/// Hazard-function contract used by BOCPD run-length transitions.
pub trait HazardFunction: Send + Sync {
    fn log_hazard(&self, r: usize) -> f64;
    fn log_survival(&self, r: usize) -> f64;
}

/// Constant hazard: `h(r) = p_change`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstantHazard {
    p_change: f64,
}

impl ConstantHazard {
    pub fn new(p_change: f64) -> Result<Self, CpdError> {
        if !(p_change.is_finite() && 0.0 < p_change && p_change < 1.0) {
            return Err(CpdError::invalid_input(format!(
                "constant hazard p_change must be finite and in (0,1); got {p_change}"
            )));
        }
        Ok(Self { p_change })
    }
}

impl HazardFunction for ConstantHazard {
    fn log_hazard(&self, _r: usize) -> f64 {
        self.p_change.ln()
    }

    fn log_survival(&self, _r: usize) -> f64 {
        (1.0 - self.p_change).ln()
    }
}

/// Geometric hazard parameterized by mean run length.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometricHazard {
    mean_run_length: f64,
    p_change: f64,
}

impl GeometricHazard {
    pub fn new(mean_run_length: f64) -> Result<Self, CpdError> {
        if !mean_run_length.is_finite() || mean_run_length <= 1.0 {
            return Err(CpdError::invalid_input(format!(
                "geometric hazard mean_run_length must be finite and > 1; got {mean_run_length}"
            )));
        }

        let p_change = 1.0 / mean_run_length;
        Ok(Self {
            mean_run_length,
            p_change,
        })
    }

    pub fn mean_run_length(&self) -> f64 {
        self.mean_run_length
    }
}

impl HazardFunction for GeometricHazard {
    fn log_hazard(&self, _r: usize) -> f64 {
        self.p_change.ln()
    }

    fn log_survival(&self, _r: usize) -> f64 {
        (1.0 - self.p_change).ln()
    }
}

/// Built-in hazard variants for BOCPD.
#[derive(Clone, Debug, PartialEq)]
pub enum HazardSpec {
    Constant(ConstantHazard),
    Geometric(GeometricHazard),
}

impl Default for HazardSpec {
    fn default() -> Self {
        Self::Constant(ConstantHazard::new(1.0 / 200.0).expect("default hazard must be valid"))
    }
}

impl HazardFunction for HazardSpec {
    fn log_hazard(&self, r: usize) -> f64 {
        match self {
            Self::Constant(h) => h.log_hazard(r),
            Self::Geometric(h) => h.log_hazard(r),
        }
    }

    fn log_survival(&self, r: usize) -> f64 {
        match self {
            Self::Constant(h) => h.log_survival(r),
            Self::Geometric(h) => h.log_survival(r),
        }
    }
}

/// Normal-Inverse-Gamma prior for Gaussian observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianNigPrior {
    pub mu0: f64,
    pub kappa0: f64,
    pub alpha0: f64,
    pub beta0: f64,
}

impl Default for GaussianNigPrior {
    fn default() -> Self {
        Self {
            mu0: 0.0,
            kappa0: 1.0,
            alpha0: 1.0,
            beta0: 1.0,
        }
    }
}

impl GaussianNigPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.mu0.is_finite() {
            return Err(CpdError::invalid_input("gaussian prior mu0 must be finite"));
        }
        if !self.kappa0.is_finite() || self.kappa0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior kappa0 must be finite and > 0",
            ));
        }
        if !self.alpha0.is_finite() || self.alpha0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior alpha0 must be finite and > 0",
            ));
        }
        if !self.beta0.is_finite() || self.beta0 <= 0.0 {
            return Err(CpdError::invalid_input(
                "gaussian prior beta0 must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Gamma prior for Poisson-rate observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct PoissonGammaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for PoissonGammaPrior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl PoissonGammaPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(CpdError::invalid_input(
                "poisson prior alpha must be finite and > 0",
            ));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CpdError::invalid_input(
                "poisson prior beta must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Beta prior for Bernoulli observation model.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BernoulliBetaPrior {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for BernoulliBetaPrior {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl BernoulliBetaPrior {
    fn validate(&self) -> Result<(), CpdError> {
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(CpdError::invalid_input(
                "bernoulli prior alpha must be finite and > 0",
            ));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(CpdError::invalid_input(
                "bernoulli prior beta must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Observation model variants for BOCPD.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationModel {
    Gaussian { prior: GaussianNigPrior },
    Poisson { prior: PoissonGammaPrior },
    Bernoulli { prior: BernoulliBetaPrior },
}

impl Default for ObservationModel {
    fn default() -> Self {
        Self::Gaussian {
            prior: GaussianNigPrior::default(),
        }
    }
}

impl ObservationModel {
    fn validate(&self) -> Result<(), CpdError> {
        match self {
            Self::Gaussian { prior } => prior.validate(),
            Self::Poisson { prior } => prior.validate(),
            Self::Bernoulli { prior } => prior.validate(),
        }
    }

    fn prior_stats(&self) -> ObservationStats {
        match self {
            Self::Gaussian { .. } => ObservationStats::Gaussian {
                n: 0,
                sum: 0.0,
                sum_sq: 0.0,
            },
            Self::Poisson { .. } => ObservationStats::Poisson { n: 0, sum: 0 },
            Self::Bernoulli { .. } => ObservationStats::Bernoulli { n: 0, ones: 0 },
        }
    }

    fn update_stats(
        &self,
        current: &ObservationStats,
        x: f64,
    ) -> Result<ObservationStats, CpdError> {
        match (self, current) {
            (Self::Gaussian { .. }, ObservationStats::Gaussian { n, sum, sum_sq }) => {
                if !x.is_finite() {
                    return Err(CpdError::invalid_input(
                        "gaussian observation must be finite",
                    ));
                }
                Ok(ObservationStats::Gaussian {
                    n: n.saturating_add(1),
                    sum: *sum + x,
                    sum_sq: *sum_sq + x * x,
                })
            }
            (Self::Poisson { .. }, ObservationStats::Poisson { n, sum }) => {
                let value = parse_non_negative_count(x)?;
                Ok(ObservationStats::Poisson {
                    n: n.saturating_add(1),
                    sum: sum.saturating_add(value),
                })
            }
            (Self::Bernoulli { .. }, ObservationStats::Bernoulli { n, ones }) => {
                let value = parse_bernoulli(x)?;
                Ok(ObservationStats::Bernoulli {
                    n: n.saturating_add(1),
                    ones: ones.saturating_add(u64::from(value)),
                })
            }
            _ => Err(CpdError::numerical_issue(
                "observation stats variant mismatch with configured model",
            )),
        }
    }

    fn log_predictive(&self, current: &ObservationStats, x: f64) -> Result<f64, CpdError> {
        match (self, current) {
            (Self::Gaussian { prior }, ObservationStats::Gaussian { n, sum, sum_sq }) => {
                gaussian_log_predictive(prior, *n, *sum, *sum_sq, x)
            }
            (Self::Poisson { prior }, ObservationStats::Poisson { n, sum }) => {
                poisson_log_predictive(prior, *n, *sum, x)
            }
            (Self::Bernoulli { prior }, ObservationStats::Bernoulli { n, ones }) => {
                bernoulli_log_predictive(prior, *n, *ones, x)
            }
            _ => Err(CpdError::numerical_issue(
                "observation stats variant mismatch with configured model",
            )),
        }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub enum ObservationStats {
    Gaussian { n: usize, sum: f64, sum_sq: f64 },
    Poisson { n: usize, sum: u64 },
    Bernoulli { n: usize, ones: u64 },
}

/// Serializable BOCPD state for checkpoint/restore.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct BocpdState {
    pub t: usize,
    pub watermark_ns: Option<i64>,
    pub log_run_probs: Vec<f64>,
    pub run_stats: Vec<ObservationStats>,
}

impl BocpdState {
    fn new(observation: &ObservationModel) -> Self {
        Self {
            t: 0,
            watermark_ns: None,
            log_run_probs: vec![0.0],
            run_stats: vec![observation.prior_stats()],
        }
    }

    fn validate(&self) -> Result<(), CpdError> {
        if self.log_run_probs.is_empty() {
            return Err(CpdError::invalid_input(
                "bocpd state requires at least one run-length probability",
            ));
        }
        if self.log_run_probs.len() != self.run_stats.len() {
            return Err(CpdError::invalid_input(format!(
                "bocpd state length mismatch: log_run_probs={}, run_stats={}",
                self.log_run_probs.len(),
                self.run_stats.len()
            )));
        }
        Ok(())
    }
}

/// BOCPD configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct BocpdConfig {
    pub hazard: HazardSpec,
    pub observation: ObservationModel,
    pub max_run_length: usize,
    /// Relative log-probability threshold (must be <= 0). Entries below `max + threshold` are pruned.
    pub log_prob_threshold: Option<f64>,
    pub alert_threshold: f64,
}

impl Default for BocpdConfig {
    fn default() -> Self {
        Self {
            hazard: HazardSpec::default(),
            observation: ObservationModel::default(),
            max_run_length: 2_000,
            log_prob_threshold: Some(-35.0),
            alert_threshold: 0.5,
        }
    }
}

impl BocpdConfig {
    fn validate(&self) -> Result<(), CpdError> {
        self.observation.validate()?;

        if self.max_run_length < 1 {
            return Err(CpdError::invalid_input(
                "max_run_length must be >= 1 for bounded BOCPD state",
            ));
        }

        if let Some(threshold) = self.log_prob_threshold
            && (!threshold.is_finite() || threshold > 0.0)
        {
            return Err(CpdError::invalid_input(
                "log_prob_threshold must be finite and <= 0",
            ));
        }

        if !self.alert_threshold.is_finite() || !(0.0..=1.0).contains(&self.alert_threshold) {
            return Err(CpdError::invalid_input(
                "alert_threshold must be finite and in [0,1]",
            ));
        }

        Ok(())
    }
}

/// Bayesian Online Change Point Detection implementation.
#[derive(Clone, Debug)]
pub struct BocpdDetector {
    config: BocpdConfig,
    state: BocpdState,
}

impl BocpdDetector {
    pub fn new(config: BocpdConfig) -> Result<Self, CpdError> {
        config.validate()?;
        let state = BocpdState::new(&config.observation);
        Ok(Self { config, state })
    }

    pub fn config(&self) -> &BocpdConfig {
        &self.config
    }

    pub fn state(&self) -> &BocpdState {
        &self.state
    }
}

impl OnlineDetector for BocpdDetector {
    type State = BocpdState;

    fn reset(&mut self) {
        self.state = BocpdState::new(&self.config.observation);
    }

    fn update(
        &mut self,
        x_t: &[f64],
        t_ns: Option<i64>,
        ctx: &ExecutionContext<'_>,
    ) -> Result<OnlineStepResult, CpdError> {
        ctx.check_cancelled()?;
        let _ = ctx.check_cost_eval_budget(self.state.t.saturating_add(1))?;

        if x_t.len() != 1 {
            return Err(CpdError::invalid_input(format!(
                "BOCPD currently supports univariate updates only; got d={} (expected 1)",
                x_t.len()
            )));
        }

        let x = x_t[0];
        if !x.is_finite() {
            return Err(CpdError::invalid_input(
                "BOCPD observation must be finite for update",
            ));
        }

        self.state.validate()?;

        let started_at = Instant::now();
        let prior_stats = self.config.observation.prior_stats();
        let prev_len = self.state.log_run_probs.len();
        let hard_cap = self.config.max_run_length.saturating_add(1);
        let candidate_len = prev_len.saturating_add(1);
        let keep_len = candidate_len.min(hard_cap).max(1);

        let mut next_log_probs = vec![f64::NEG_INFINITY; keep_len];
        let mut next_stats = vec![prior_stats.clone(); keep_len];

        let mut cp_mass = f64::NEG_INFINITY;
        let log_pred_reset = self.config.observation.log_predictive(&prior_stats, x)?;

        for run_length in 0..prev_len {
            let log_prev = self.state.log_run_probs[run_length];
            let log_pred_growth = self
                .config
                .observation
                .log_predictive(&self.state.run_stats[run_length], x)?;

            let cp_term = log_prev + self.config.hazard.log_hazard(run_length) + log_pred_reset;
            cp_mass = log_add_exp(cp_mass, cp_term);

            let next_run_length = run_length + 1;
            let growth_term =
                log_prev + self.config.hazard.log_survival(run_length) + log_pred_growth;

            if next_run_length < keep_len {
                next_log_probs[next_run_length] = growth_term;
                next_stats[next_run_length] = self
                    .config
                    .observation
                    .update_stats(&self.state.run_stats[run_length], x)?;
            } else {
                // Truncation redistributes overflow mass to run_length=0.
                cp_mass = log_add_exp(cp_mass, growth_term);
            }
        }

        next_log_probs[0] = cp_mass;
        next_stats[0] = self.config.observation.update_stats(&prior_stats, x)?;

        if let Some(threshold) = self.config.log_prob_threshold {
            let max_log_prob = next_log_probs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let cutoff = max_log_prob + threshold;

            let mut keep = next_log_probs.len();
            while keep > 1 && next_log_probs[keep - 1] < cutoff {
                keep -= 1;
            }
            next_log_probs.truncate(keep);
            next_stats.truncate(keep);
        }

        normalize_log_probs(&mut next_log_probs)?;

        let run_length_mode = argmax_index(&next_log_probs);
        let run_length_mean = run_length_expectation(&next_log_probs);

        let mut p_change = next_log_probs[0].exp();
        if !p_change.is_finite() {
            return Err(CpdError::numerical_issue(
                "BOCPD p_change became non-finite after normalization",
            ));
        }
        p_change = p_change.clamp(0.0, 1.0);

        let alert = p_change >= self.config.alert_threshold;

        self.state.t = self.state.t.saturating_add(1);
        if let Some(ts) = t_ns {
            self.state.watermark_ns = Some(self.state.watermark_ns.map_or(ts, |w| w.max(ts)));
        }
        self.state.log_run_probs = next_log_probs;
        self.state.run_stats = next_stats;

        Ok(OnlineStepResult {
            t: self.state.t.saturating_sub(1),
            p_change,
            alert,
            alert_reason: alert.then(|| {
                format!(
                    "bocpd p_change {:.6} >= threshold {:.6}",
                    p_change, self.config.alert_threshold
                )
            }),
            run_length_mode,
            run_length_mean,
            processing_latency_us: Some(started_at.elapsed().as_micros() as u64),
        })
    }

    fn save_state(&self) -> Self::State {
        self.state.clone()
    }

    fn load_state(&mut self, state: &Self::State) {
        self.state = state.clone();
    }
}

fn normalize_log_probs(log_probs: &mut [f64]) -> Result<(), CpdError> {
    let normalizer = log_sum_exp(log_probs);
    if !normalizer.is_finite() {
        return Err(CpdError::numerical_issue(
            "BOCPD normalization failed (non-finite log_sum_exp)",
        ));
    }

    for value in log_probs {
        *value -= normalizer;
        if value.is_nan() {
            return Err(CpdError::numerical_issue(
                "BOCPD normalization produced NaN run-length log probability",
            ));
        }
    }

    Ok(())
}

fn argmax_index(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn run_length_expectation(log_probs: &[f64]) -> f64 {
    log_probs
        .iter()
        .enumerate()
        .map(|(idx, log_prob)| idx as f64 * log_prob.exp())
        .sum()
}

fn gaussian_log_predictive(
    prior: &GaussianNigPrior,
    n: usize,
    sum: f64,
    sum_sq: f64,
    x: f64,
) -> Result<f64, CpdError> {
    if !x.is_finite() {
        return Err(CpdError::invalid_input(
            "gaussian observation must be finite",
        ));
    }

    let n_f64 = n as f64;
    let kappa_n = prior.kappa0 + n_f64;
    if !(kappa_n.is_finite() && kappa_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian posterior kappa became non-finite or non-positive",
        ));
    }

    let mu_n = (prior.kappa0 * prior.mu0 + sum) / kappa_n;

    let centered_sse = if n == 0 {
        0.0
    } else {
        let mean = sum / n_f64;
        let raw = sum_sq - n_f64 * mean * mean;
        if raw <= 0.0 { 0.0 } else { raw }
    };

    let alpha_n = prior.alpha0 + 0.5 * n_f64;
    let shrinkage = if n == 0 {
        0.0
    } else {
        let mean = sum / n_f64;
        prior.kappa0 * n_f64 * (mean - prior.mu0).powi(2) / (2.0 * kappa_n)
    };
    let beta_n = prior.beta0 + 0.5 * centered_sse + shrinkage;

    if !(alpha_n.is_finite() && alpha_n > 0.0 && beta_n.is_finite() && beta_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian posterior alpha/beta became non-finite or non-positive",
        ));
    }

    let nu = 2.0 * alpha_n;
    let scale_sq = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n);
    if !(nu.is_finite() && nu > 0.0 && scale_sq.is_finite() && scale_sq > 0.0) {
        return Err(CpdError::numerical_issue(
            "gaussian predictive scale became non-finite or non-positive",
        ));
    }

    let z = (x - mu_n).powi(2) / (nu * scale_sq);
    let log_norm =
        ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu.ln() + PI.ln() + scale_sq.ln());
    let log_tail = -0.5 * (nu + 1.0) * (1.0 + z).ln();
    let out = log_norm + log_tail;

    if !out.is_finite() {
        return Err(CpdError::numerical_issue(
            "gaussian predictive log-likelihood became non-finite",
        ));
    }

    Ok(out)
}

fn poisson_log_predictive(
    prior: &PoissonGammaPrior,
    n: usize,
    sum: u64,
    x: f64,
) -> Result<f64, CpdError> {
    let count = parse_non_negative_count(x)?;
    let posterior_alpha = prior.alpha + (sum as f64);
    let posterior_beta = prior.beta + (n as f64);

    if !(posterior_alpha.is_finite() && posterior_alpha > 0.0) {
        return Err(CpdError::numerical_issue(
            "poisson posterior alpha became non-finite or non-positive",
        ));
    }
    if !(posterior_beta.is_finite() && posterior_beta > 0.0) {
        return Err(CpdError::numerical_issue(
            "poisson posterior beta became non-finite or non-positive",
        ));
    }

    let count_f64 = count as f64;
    let out =
        ln_gamma(posterior_alpha + count_f64) - ln_gamma(posterior_alpha) - ln_factorial(count)
            + posterior_alpha * (posterior_beta / (posterior_beta + 1.0)).ln()
            + count_f64 * (1.0 / (posterior_beta + 1.0)).ln();

    if !out.is_finite() {
        return Err(CpdError::numerical_issue(
            "poisson predictive log-likelihood became non-finite",
        ));
    }

    Ok(out)
}

fn bernoulli_log_predictive(
    prior: &BernoulliBetaPrior,
    n: usize,
    ones: u64,
    x: f64,
) -> Result<f64, CpdError> {
    let bit = parse_bernoulli(x)?;
    let alpha_n = prior.alpha + (ones as f64);
    let beta_n = prior.beta + ((n as u64).saturating_sub(ones) as f64);

    if !(alpha_n.is_finite() && alpha_n > 0.0 && beta_n.is_finite() && beta_n > 0.0) {
        return Err(CpdError::numerical_issue(
            "bernoulli posterior alpha/beta became non-finite or non-positive",
        ));
    }

    let denom = alpha_n + beta_n;
    let prob = if bit { alpha_n / denom } else { beta_n / denom };

    if !(prob.is_finite() && prob > 0.0) {
        return Err(CpdError::numerical_issue(
            "bernoulli predictive probability became non-finite or non-positive",
        ));
    }

    Ok(prob.ln())
}

fn parse_non_negative_count(x: f64) -> Result<u64, CpdError> {
    if !x.is_finite() || x < 0.0 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation must be finite and >= 0; got {x}"
        )));
    }

    let rounded = x.round();
    if (rounded - x).abs() > 1e-9 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation must be an integer-valued count; got {x}"
        )));
    }

    if rounded > u64::MAX as f64 {
        return Err(CpdError::invalid_input(format!(
            "poisson observation exceeds u64 range; got {x}"
        )));
    }

    Ok(rounded as u64)
}

fn parse_bernoulli(x: f64) -> Result<bool, CpdError> {
    if (x - 0.0).abs() <= 1e-12 {
        return Ok(false);
    }
    if (x - 1.0).abs() <= 1e-12 {
        return Ok(true);
    }
    Err(CpdError::invalid_input(format!(
        "bernoulli observation must be exactly 0 or 1; got {x}"
    )))
}

fn ln_factorial(n: u64) -> f64 {
    ln_gamma((n + 1) as f64)
}

fn ln_gamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z <= 0.0 || !z.is_finite() {
        return f64::NAN;
    }

    if z < 0.5 {
        let sin_term = (PI * z).sin();
        if sin_term == 0.0 {
            return f64::INFINITY;
        }
        return PI.ln() - sin_term.ln() - ln_gamma(1.0 - z);
    }

    let x = z - 1.0;
    let mut acc = COEFFS[0];
    for (idx, coeff) in COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (x + idx as f64);
    }

    let t = x + 7.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + acc.ln()
}

#[cfg(test)]
mod tests {
    use super::{
        BocpdConfig, BocpdDetector, BocpdState, ConstantHazard, GeometricHazard, HazardSpec,
        ObservationModel,
    };
    use cpd_core::{Constraints, ExecutionContext, OnlineDetector};
    use std::sync::OnceLock;

    fn ctx() -> ExecutionContext<'static> {
        static CONSTRAINTS: OnceLock<Constraints> = OnceLock::new();
        let constraints = CONSTRAINTS.get_or_init(Constraints::default);
        ExecutionContext::new(constraints)
    }

    fn probs_from_log_probs(log_probs: &[f64]) -> Vec<f64> {
        log_probs.iter().map(|value| value.exp()).collect()
    }

    #[test]
    fn known_posterior_first_step_matches_closed_form() {
        let hazard_p = 0.2;
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(hazard_p).expect("valid hazard")),
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            max_run_length: 32,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        detector
            .update(&[1.0], None, &ctx())
            .expect("update should succeed");

        let probs = probs_from_log_probs(&detector.state().log_run_probs);
        let expected = [hazard_p, 1.0 - hazard_p];

        assert_eq!(probs.len(), expected.len());
        for (idx, (observed, expected)) in probs.iter().zip(expected).enumerate() {
            assert!(
                (observed - expected).abs() < 1e-12,
                "posterior mismatch at run_length={idx}: observed={observed}, expected={expected}",
            );
        }
    }

    #[test]
    fn known_posterior_two_step_bernoulli_matches_closed_form() {
        let hazard_p = 0.2;
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(hazard_p).expect("valid hazard")),
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            max_run_length: 32,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        detector
            .update(&[1.0], None, &ctx())
            .expect("first update should succeed");
        let step = detector
            .update(&[1.0], None, &ctx())
            .expect("second update should succeed");

        let normalizer = 4.0 - hazard_p;
        let expected = [
            (3.0 * hazard_p) / normalizer,
            (4.0 * hazard_p * (1.0 - hazard_p)) / normalizer,
            (4.0 * (1.0 - hazard_p) * (1.0 - hazard_p)) / normalizer,
        ];

        let probs = probs_from_log_probs(&detector.state().log_run_probs);
        assert_eq!(probs.len(), expected.len());
        for (idx, (observed, expected)) in probs.iter().zip(expected).enumerate() {
            assert!(
                (observed - expected).abs() < 1e-12,
                "posterior mismatch at run_length={idx}: observed={observed}, expected={expected}",
            );
        }

        assert!(
            (step.p_change - expected[0]).abs() < 1e-12,
            "step p_change mismatch: observed={}, expected={}",
            step.p_change,
            expected[0]
        );
    }

    #[test]
    fn constant_series_keeps_change_probability_low() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            max_run_length: 256,
            alert_threshold: 0.7,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        let mut tail_sum = 0.0;
        let mut tail_n = 0usize;
        for step in 0..300 {
            let result = detector
                .update(&[0.0], None, &ctx())
                .expect("update should succeed");
            if step >= 80 {
                tail_sum += result.p_change;
                tail_n += 1;
            }
        }

        let tail_mean = tail_sum / tail_n as f64;
        assert!(tail_mean < 0.2, "tail mean p_change too high: {tail_mean}");
    }

    #[test]
    fn step_shift_produces_change_probability_spike() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            hazard: HazardSpec::Constant(ConstantHazard::new(1.0 / 80.0).expect("valid hazard")),
            max_run_length: 256,
            ..BocpdConfig::default()
        })
        .expect("config should be valid");

        let mut best_idx = 0usize;
        let mut best_prob = 0.0;

        for step in 0..240 {
            let x = if step < 120 { 0.0 } else { 6.0 };
            let result = detector
                .update(&[x], None, &ctx())
                .expect("update should succeed");
            if result.p_change > best_prob {
                best_prob = result.p_change;
                best_idx = step;
            }
        }

        assert!(
            best_prob > 0.25,
            "expected spike; observed p_change={best_prob}"
        );
        assert!(
            (105..=135).contains(&best_idx),
            "expected spike near changepoint; best_idx={best_idx}"
        );
    }

    #[test]
    fn checkpoint_restore_roundtrip_is_equivalent() {
        let mut baseline = BocpdDetector::new(BocpdConfig::default()).expect("valid config");
        let mut first = BocpdDetector::new(BocpdConfig::default()).expect("valid config");

        for i in 0..120 {
            let x = ((i as f64) * 0.07).sin();
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = first
                .update(&[x], None, &ctx())
                .expect("first update should succeed");
            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
        }

        let saved: BocpdState = first.save_state();
        let mut restored = BocpdDetector::new(BocpdConfig::default()).expect("valid config");
        restored.load_state(&saved);

        for i in 120..260 {
            let x = if i % 53 < 11 {
                4.0
            } else {
                ((i as f64) * 0.03).cos()
            };
            let lhs = baseline
                .update(&[x], None, &ctx())
                .expect("baseline update should succeed");
            let rhs = restored
                .update(&[x], None, &ctx())
                .expect("restored update should succeed");

            assert!((lhs.p_change - rhs.p_change).abs() < 1e-12);
            assert_eq!(lhs.alert, rhs.alert);
            assert_eq!(lhs.run_length_mode, rhs.run_length_mode);
        }
    }

    #[test]
    fn max_run_length_bounds_state_size() {
        let mut detector = BocpdDetector::new(BocpdConfig {
            max_run_length: 16,
            log_prob_threshold: None,
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for i in 0..512 {
            detector
                .update(&[(i as f64) * 0.001], None, &ctx())
                .expect("update should succeed");
            assert!(detector.state().log_run_probs.len() <= 17);
        }
    }

    #[test]
    fn poisson_and_bernoulli_models_update_successfully() {
        let mut poisson = BocpdDetector::new(BocpdConfig {
            observation: ObservationModel::Poisson {
                prior: super::PoissonGammaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for x in [0.0, 1.0, 2.0, 3.0, 1.0, 0.0] {
            poisson
                .update(&[x], None, &ctx())
                .expect("poisson update should succeed");
        }

        let mut bernoulli = BocpdDetector::new(BocpdConfig {
            observation: ObservationModel::Bernoulli {
                prior: super::BernoulliBetaPrior {
                    alpha: 1.0,
                    beta: 1.0,
                },
            },
            ..BocpdConfig::default()
        })
        .expect("valid config");

        for x in [0.0, 1.0, 1.0, 0.0, 1.0] {
            bernoulli
                .update(&[x], None, &ctx())
                .expect("bernoulli update should succeed");
        }

        let err = bernoulli
            .update(&[0.2], None, &ctx())
            .expect_err("non-binary observation should fail");
        assert!(
            err.to_string().contains("bernoulli observation"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn geometric_hazard_and_threshold_validation() {
        assert!(GeometricHazard::new(200.0).is_ok());
        assert!(GeometricHazard::new(1.0).is_err());

        let config = BocpdConfig {
            hazard: HazardSpec::Geometric(
                GeometricHazard::new(120.0).expect("mean run length should be valid"),
            ),
            log_prob_threshold: Some(0.1),
            ..BocpdConfig::default()
        };

        let err = BocpdDetector::new(config).expect_err("positive threshold should fail");
        assert!(err.to_string().contains("log_prob_threshold"));
    }
}
