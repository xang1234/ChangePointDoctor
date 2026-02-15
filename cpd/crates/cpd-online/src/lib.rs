// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

pub mod bocpd;

pub use bocpd::{
    BernoulliBetaPrior, BocpdConfig, BocpdDetector, BocpdState, ConstantHazard, GaussianNigPrior,
    GeometricHazard, HazardFunction, HazardSpec, ObservationModel, ObservationStats,
    PoissonGammaPrior,
};

/// Online detector namespace.
pub fn crate_name() -> &'static str {
    let _ = (cpd_core::crate_name(), cpd_costs::crate_name());
    "cpd-online"
}
