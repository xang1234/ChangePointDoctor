#![no_main]

use cpd_core::{
    Constraints, DTypeView, ExecutionContext, MemoryLayout, MissingPolicy, OfflineDetector,
    Penalty, Stopping, TimeIndex, TimeSeriesView,
};
use cpd_costs::CostL2Mean;
use cpd_offline::{Pelt, PeltConfig};
use libfuzzer_sys::fuzz_target;

fn decode_values(data: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(data.len() / 8);

    for chunk in data.chunks_exact(8).take(512) {
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(chunk);
        let raw = f64::from_le_bytes(bytes);
        if raw.is_finite() {
            values.push(raw.clamp(-1.0e6, 1.0e6));
        } else {
            values.push(0.0);
        }
    }

    values
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let mut values = decode_values(data);
    if values.len() < 4 {
        return;
    }
    if values.len() > 512 {
        values.truncate(512);
    }

    let n = values.len();
    let min_segment_len = (usize::from(data[0]) % 8) + 1;
    let jump = (usize::from(data[1]) % 8) + 1;
    let max_change_points = usize::from(data[2]) % 16;
    let penalty_value = (f64::from(data[3]) / 8.0) + 0.1;

    let constraints = Constraints {
        min_segment_len: min_segment_len.min(n.saturating_sub(1)).max(1),
        jump: jump.min(n.saturating_sub(1)).max(1),
        max_change_points: Some(max_change_points),
        ..Constraints::default()
    };

    let view = match TimeSeriesView::new(
        DTypeView::F64(&values),
        n,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    ) {
        Ok(view) => view,
        Err(_) => return,
    };

    let detector = match Pelt::new(
        CostL2Mean::default(),
        PeltConfig {
            stopping: Stopping::Penalized(Penalty::Manual(penalty_value)),
            params_per_segment: 2,
            cancel_check_every: 128,
        },
    ) {
        Ok(detector) => detector,
        Err(_) => return,
    };

    let context = ExecutionContext::new(&constraints);
    let _ = detector.detect(&view, &context);
});
