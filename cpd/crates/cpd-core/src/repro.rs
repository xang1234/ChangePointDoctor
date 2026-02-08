// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

/// Reproducibility mode used to control determinism/performance trade-offs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReproMode {
    Strict,
    #[default]
    Balanced,
    Fast,
}

#[cfg(test)]
mod tests {
    use super::ReproMode;

    #[test]
    fn repro_mode_default_is_balanced() {
        assert_eq!(ReproMode::default(), ReproMode::Balanced);
    }
}
