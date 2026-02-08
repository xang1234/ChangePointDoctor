// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Cooperative cancellation token shared across algorithm loops.
#[derive(Clone, Debug)]
pub struct CancelToken(pub Arc<AtomicBool>);

impl CancelToken {
    /// Creates a new token in the non-cancelled state.
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    /// Requests cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Returns true when cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget handling mode for runtime guardrails.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetMode {
    HardFail,
    SoftDegrade,
}

/// Result of checking an execution budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetStatus {
    WithinBudget,
    ExceededSoftDegrade,
}

#[cfg(test)]
mod tests {
    use super::CancelToken;

    #[test]
    fn cancel_token_new_starts_not_cancelled() {
        let token = CancelToken::new();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn cancel_token_cancel_sets_cancelled_state() {
        let token = CancelToken::new();
        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn cancel_token_clones_share_state() {
        let token = CancelToken::new();
        let clone = token.clone();

        clone.cancel();
        assert!(token.is_cancelled());
        assert!(clone.is_cancelled());
    }
}
