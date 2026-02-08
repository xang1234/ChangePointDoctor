// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]

use cpd_core::{CachePolicy, DTypeView, MemoryLayout, MissingPolicy, TimeIndex, TimeSeriesView};
use cpd_costs::{CostL2Mean, CostModel};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 1_000_000;
const QUERY_COUNT: usize = 1_000_000;

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn generate_queries(n: usize, count: usize) -> Vec<(usize, usize)> {
    let mut queries = Vec::with_capacity(count);
    let mut state = 0xfeed_f00d_dead_beef_u64;

    for _ in 0..count {
        let a = (lcg_next(&mut state) as usize) % n;
        let b = (lcg_next(&mut state) as usize) % n;
        let start = a.min(b);
        let mut end = a.max(b) + 1;
        if start == end {
            end = (start + 1).min(n);
        }
        queries.push((start, end));
    }

    queries
}

fn benchmark_cost_l2(c: &mut Criterion) {
    let values: Vec<f64> = (0..N)
        .map(|idx| {
            let x = idx as f64;
            x.sin() + x.cos() * 0.1
        })
        .collect();

    let view = TimeSeriesView::new(
        DTypeView::F64(&values),
        N,
        1,
        MemoryLayout::CContiguous,
        None,
        TimeIndex::None,
        MissingPolicy::Error,
    )
    .expect("benchmark view should be valid");

    let model = CostL2Mean::default();

    let mut group = c.benchmark_group("cost_l2");

    group.bench_function("l2_precompute_n1e6_d1_balanced", |b| {
        b.iter(|| {
            let _cache = model
                .precompute(black_box(&view), black_box(&CachePolicy::Full))
                .expect("precompute should succeed");
        })
    });

    let cache = model
        .precompute(&view, &CachePolicy::Full)
        .expect("precompute should succeed");
    let queries = generate_queries(N, QUERY_COUNT);
    let mut out_costs = vec![0.0; queries.len()];

    group.bench_function("l2_segment_queries_n1e6_d1_1m", |b| {
        b.iter(|| {
            model.segment_cost_batch(
                black_box(&cache),
                black_box(&queries),
                black_box(&mut out_costs),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_cost_l2);
criterion_main!(benches);
