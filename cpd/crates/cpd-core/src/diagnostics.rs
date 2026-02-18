// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use crate::repro::ReproMode;
use std::borrow::Cow;

/// Diagnostics schema version for change-point run metadata.
pub const DIAGNOSTICS_SCHEMA_VERSION: u32 = 1;

/// Build provenance captured at compile time and optionally enriched by adapters.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildInfo {
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub git_sha: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub git_dirty: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub rustc_version: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub target_triple: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub profile: Option<String>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub features: Option<Vec<String>>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub abi: Option<String>,
}

impl BuildInfo {
    pub fn from_compile_time_env() -> Option<Self> {
        let build = Self {
            git_sha: option_env!("CPD_BUILD_GIT_SHA")
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty()),
            git_dirty: option_env!("CPD_BUILD_GIT_DIRTY").and_then(|value| match value {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }),
            rustc_version: option_env!("CPD_BUILD_RUSTC_VERSION")
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty()),
            target_triple: option_env!("CPD_BUILD_TARGET_TRIPLE")
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty()),
            profile: option_env!("CPD_BUILD_PROFILE")
                .map(ToString::to_string)
                .filter(|value| !value.trim().is_empty()),
            features: None,
            abi: None,
        };
        if build.is_empty() { None } else { Some(build) }
    }

    pub fn is_empty(&self) -> bool {
        self.git_sha.is_none()
            && self.git_dirty.is_none()
            && self.rustc_version.is_none()
            && self.target_triple.is_none()
            && self.profile.is_none()
            && self.features.is_none()
            && self.abi.is_none()
    }
}

/// Counters that summarize pruning effectiveness during a run.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PruningStats {
    pub candidates_considered: usize,
    pub candidates_pruned: usize,
}

/// Structured diagnostics captured from a detector execution.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostics {
    pub n: usize,
    pub d: usize,
    pub schema_version: u32,
    pub engine_version: Option<String>,
    pub runtime_ms: Option<u64>,
    pub notes: Vec<String>,
    pub warnings: Vec<String>,
    pub algorithm: Cow<'static, str>,
    pub cost_model: Cow<'static, str>,
    pub seed: Option<u64>,
    pub repro_mode: ReproMode,
    pub thread_count: Option<usize>,
    pub blas_backend: Option<String>,
    pub cpu_features: Option<Vec<String>>,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    pub build: Option<BuildInfo>,
    #[cfg(feature = "serde")]
    pub params_json: Option<serde_json::Value>,
    pub pruning_stats: Option<PruningStats>,
    pub missing_policy_applied: Option<String>,
    pub missing_fraction: Option<f64>,
    pub effective_sample_count: Option<usize>,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Self {
            n: 0,
            d: 0,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            runtime_ms: None,
            notes: vec![],
            warnings: vec![],
            algorithm: Cow::Borrowed(""),
            cost_model: Cow::Borrowed(""),
            seed: None,
            repro_mode: ReproMode::Balanced,
            thread_count: None,
            blas_backend: None,
            cpu_features: None,
            build: BuildInfo::from_compile_time_env(),
            #[cfg(feature = "serde")]
            params_json: None,
            pruning_stats: None,
            missing_policy_applied: None,
            missing_fraction: None,
            effective_sample_count: None,
        }
    }
}

fn normalized_features(features: Option<Vec<String>>) -> Option<Vec<String>> {
    let mut features = features?;
    features.retain(|feature| !feature.trim().is_empty());
    if features.is_empty() {
        return None;
    }
    features.sort_unstable();
    features.dedup();
    Some(features)
}

/// Ensures diagnostics carry build provenance and adapter context when available.
pub fn enrich_diagnostics_build(
    diagnostics: &mut Diagnostics,
    features: Option<Vec<String>>,
    abi: Option<String>,
) {
    let mut build = diagnostics
        .build
        .take()
        .or_else(BuildInfo::from_compile_time_env)
        .unwrap_or_default();

    if let Some(features) = normalized_features(features) {
        build.features = Some(features);
    }
    if let Some(abi) = abi.filter(|value| !value.trim().is_empty()) {
        build.abi = Some(abi);
    }

    diagnostics.build = if build.is_empty() { None } else { Some(build) };
}

#[cfg(test)]
mod tests {
    use super::{DIAGNOSTICS_SCHEMA_VERSION, Diagnostics, PruningStats, enrich_diagnostics_build};
    use crate::ReproMode;
    use std::borrow::Cow;

    #[test]
    fn diagnostics_default_sets_schema_and_engine_version() {
        let diagnostics = Diagnostics::default();
        assert_eq!(diagnostics.schema_version, DIAGNOSTICS_SCHEMA_VERSION);
        assert_eq!(
            diagnostics.engine_version,
            Some(env!("CARGO_PKG_VERSION").to_string())
        );
    }

    #[test]
    fn diagnostics_default_sets_expected_empty_and_none_fields() {
        let diagnostics = Diagnostics::default();
        assert_eq!(diagnostics.n, 0);
        assert_eq!(diagnostics.d, 0);
        assert_eq!(diagnostics.algorithm, Cow::Borrowed(""));
        assert_eq!(diagnostics.cost_model, Cow::Borrowed(""));
        assert_eq!(diagnostics.repro_mode, ReproMode::Balanced);
        assert!(diagnostics.runtime_ms.is_none());
        assert!(diagnostics.notes.is_empty());
        assert!(diagnostics.warnings.is_empty());
        assert!(diagnostics.seed.is_none());
        assert!(diagnostics.thread_count.is_none());
        assert!(diagnostics.blas_backend.is_none());
        assert!(diagnostics.cpu_features.is_none());
        assert!(diagnostics.pruning_stats.is_none());
        assert!(diagnostics.missing_policy_applied.is_none());
        assert!(diagnostics.missing_fraction.is_none());
        assert!(diagnostics.effective_sample_count.is_none());
        if let Some(build) = diagnostics.build {
            assert!(!build.is_empty());
        }
    }

    #[test]
    fn enrich_build_context_adds_features_and_abi() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.build = None;
        enrich_diagnostics_build(
            &mut diagnostics,
            Some(vec![
                "serde".to_string(),
                "".to_string(),
                "serde".to_string(),
                "preprocess".to_string(),
            ]),
            Some("abi3-py39".to_string()),
        );

        let build = diagnostics
            .build
            .as_ref()
            .expect("adapter context should create build info");
        assert_eq!(
            build.features.as_ref(),
            Some(&vec!["preprocess".to_string(), "serde".to_string()])
        );
        assert_eq!(build.abi.as_deref(), Some("abi3-py39"));
    }

    #[test]
    fn pruning_stats_fields_roundtrip_in_memory() {
        let stats = PruningStats {
            candidates_considered: 10_000,
            candidates_pruned: 9_500,
        };
        let copied = stats.clone();
        assert_eq!(copied, stats);
        assert_eq!(copied.candidates_considered, 10_000);
        assert_eq!(copied.candidates_pruned, 9_500);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn diagnostics_serde_roundtrip_preserves_all_fields() {
        let diagnostics = Diagnostics {
            n: 1_024,
            d: 4,
            schema_version: DIAGNOSTICS_SCHEMA_VERSION,
            engine_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            runtime_ms: Some(125),
            notes: vec!["candidate set canonicalized".to_string()],
            warnings: vec!["using approximate cache".to_string()],
            algorithm: Cow::Owned("pelt".to_string()),
            cost_model: Cow::Owned("l2".to_string()),
            seed: Some(42),
            repro_mode: ReproMode::Balanced,
            thread_count: Some(8),
            blas_backend: Some("openblas".to_string()),
            cpu_features: Some(vec!["avx2".to_string(), "fma".to_string()]),
            build: Some(super::BuildInfo {
                git_sha: Some("abc123".to_string()),
                git_dirty: Some(false),
                rustc_version: Some("rustc 1.85.0".to_string()),
                target_triple: Some("x86_64-unknown-linux-gnu".to_string()),
                profile: Some("release".to_string()),
                features: Some(vec!["serde".to_string()]),
                abi: Some("abi3-py39".to_string()),
            }),
            params_json: Some(serde_json::json!({
                "jump": 5,
                "min_segment_len": 20
            })),
            pruning_stats: Some(PruningStats {
                candidates_considered: 9_000,
                candidates_pruned: 7_500,
            }),
            missing_policy_applied: Some("Ignore".to_string()),
            missing_fraction: Some(0.0125),
            effective_sample_count: Some(1_011),
        };

        let encoded = serde_json::to_string(&diagnostics).expect("diagnostics should serialize");
        let decoded: Diagnostics =
            serde_json::from_str(&encoded).expect("diagnostics should deserialize");
        assert_eq!(decoded, diagnostics);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn diagnostics_serialization_omits_build_when_absent() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.build = None;
        let encoded =
            serde_json::to_value(&diagnostics).expect("diagnostics should serialize to value");
        assert!(encoded.get("build").is_none());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn pruning_stats_serde_roundtrip() {
        let stats = PruningStats {
            candidates_considered: 1_000,
            candidates_pruned: 840,
        };
        let encoded = serde_json::to_string(&stats).expect("pruning stats should serialize");
        let decoded: PruningStats =
            serde_json::from_str(&encoded).expect("pruning stats should deserialize");
        assert_eq!(decoded, stats);
    }
}
