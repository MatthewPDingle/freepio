//! GPU-vs-CPU equivalence for the preflop engine. These tests need a CUDA
//! machine (RTX 3090 etc.) — run there before trusting GPU output:
//!
//!     cargo test --release --features gpu --test preflop_gpu -- --test-threads=1
//!
//! The GPU engine was written to mirror the CPU traversal exactly; any
//! disagreement beyond float-order noise is a GPU bug.
#![cfg(feature = "gpu")]

use solver::preflop::equity::{class_index, EquityTable, NUM_CLASSES};
use solver::preflop::gpu::PreflopGpu;
use solver::preflop::{PreflopConfig, PreflopSolver};
use std::sync::{Arc, OnceLock};

fn table() -> Arc<EquityTable> {
    static T: OnceLock<Arc<EquityTable>> = OnceLock::new();
    T.get_or_init(|| Arc::new(EquityTable::build(4000))).clone()
}

fn hu25() -> PreflopConfig {
    PreflopConfig {
        positions: vec!["SB".into(), "BB".into()],
        stack: 25.0,
        posts: vec![0.5, 1.0],
        ante: 0.0,
        limp: true,
        open_raises: vec![2.0, 2.5],
        raise_mults: vec![3.0],
        max_raises: 3,
        add_allin: true,
        allin_threshold: 0.85,
        rake_pct: 5.0,
        rake_cap: 1.0,
        no_flop_no_drop: true,
        realization: "static".into(),
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
    }
}

/// Same game, same iteration count, CPU vs GPU: every action node's average
/// strategy must agree wherever the answer is DECISIVE, and the BR gaps
/// must match. (Trajectory-exact equality is impossible between two f32
/// implementations with different summation orders: near-zero regret
/// crossings flip differently and mixing-region frequencies fork while
/// both remain equally converged — verified 2026-07-07 on an RTX 4090:
/// per-iteration arena diffs stay at float noise (<1e-4) for the first
/// iterations, then knife-edge classes drift apart.)
#[test]
fn gpu_matches_cpu() {
    let eq = table();
    let mut cpu = PreflopSolver::new(hu25(), eq.clone()).unwrap();
    let mut gs = PreflopSolver::new(hu25(), eq).unwrap();
    cpu.prune = false;
    gs.prune = false;
    let mut g = PreflopGpu::new(&gs, 8_000).expect("gpu init");

    // Short horizon: per-iteration math must mirror to float noise before
    // chaos has room to amplify.
    for _ in 0..5 {
        cpu.iterate();
        g.iterate(&mut gs).unwrap();
    }
    g.sync_to_cpu(&mut gs).unwrap();
    let (cr, _) = cpu.arena_snapshot();
    let (gr, _) = gs.arena_snapshot();
    let mut worst_r = 0f32;
    for (x, y) in cr.iter().zip(gr.iter()) {
        worst_r = worst_r.max((x - y).abs());
    }
    assert!(
        worst_r < 1e-3,
        "per-iteration regret math diverges beyond float noise: {worst_r}"
    );

    // Long horizon: both must CONVERGE to the same answer — identical pure
    // decisions, matching gaps — even though mixing frequencies may fork.
    for _ in 0..295 {
        cpu.iterate();
    }
    for _ in 0..295 {
        g.iterate(&mut gs).unwrap();
    }
    g.sync_to_cpu(&mut gs).unwrap();
    assert_eq!(cpu.iteration, gs.iteration);

    let (mut decisive, mut clash) = (0u32, 0u32);
    for i in 0..cpu.nodes.len() {
        if cpu.nodes[i].kind != 0 {
            continue;
        }
        let a = cpu.average_strategy(i);
        let b = gs.average_strategy(i);
        for (x, y) in a.iter().zip(b.iter()) {
            if *x > 0.9 || *x < 0.1 {
                decisive += 1;
                if (x - y).abs() > 0.1 {
                    clash += 1;
                }
            }
        }
    }
    assert!(decisive > 1000, "sanity: expected many decisive entries, got {decisive}");
    assert!(
        (clash as f64) < decisive as f64 * 0.002,
        "pure decisions disagree: {clash} of {decisive} decisive entries"
    );

    let gc: f64 = cpu.br_gaps().iter().sum();
    let gg: f64 = gs.br_gaps().iter().sum();
    assert!(
        (gc - gg).abs() < 0.02,
        "BR gap mismatch: cpu {gc} vs gpu-synced {gg}"
    );

    // the device's own gap computation must agree with the CPU's on the
    // synced arenas
    let (gaps_dev, _evs) = g.gaps_and_evs().unwrap();
    let gd: f64 = gaps_dev.iter().sum();
    assert!(
        (gd - gg).abs() < 0.02,
        "device gap computation disagrees: {gd} vs {gg}"
    );
}

/// The 10bb push/fold anchors must hold when solved entirely on the GPU.
#[test]
fn gpu_push_fold_anchors() {
    let eq = table();
    let cfg = PreflopConfig {
        positions: vec!["SB".into(), "BB".into()],
        stack: 10.0,
        posts: vec![0.5, 1.0],
        ante: 0.0,
        limp: false,
        open_raises: vec![],
        raise_mults: vec![],
        max_raises: 1,
        add_allin: true,
        allin_threshold: 0.85,
        rake_pct: 0.0,
        rake_cap: 0.0,
        no_flop_no_drop: true,
        realization: "raw".into(),
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
    };
    let mut s = PreflopSolver::new(cfg, eq).unwrap();
    let mut g = PreflopGpu::new(&s, 8_000).expect("gpu init");
    for _ in 0..800 {
        g.iterate(&mut s).unwrap();
    }
    g.sync_to_cpu(&mut s).unwrap();

    let sb = s.average_strategy(0);
    let jam = s.nodes[0].actions.iter().position(|a| a.kind == "jam").unwrap();
    let bb_idx = s.child(0, jam);
    let bb = s.average_strategy(bb_idx);
    let call = s.nodes[bb_idx].actions.iter().position(|a| a.kind == "call").unwrap();
    let aa = class_index(12, 12, false);
    let seven_deuce = class_index(5, 0, false);
    assert!(sb[jam * NUM_CLASSES + aa] > 0.99, "AA must jam");
    assert!(bb[call * NUM_CLASSES + aa] > 0.99, "AA must call");
    assert!(bb[call * NUM_CLASSES + seven_deuce] < 0.05, "72o must fold");
    let gap: f64 = s.br_gaps().iter().sum();
    assert!(gap < 0.02, "GPU solve should converge: gap {gap} bb");
}
