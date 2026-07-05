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
    }
}

/// Same game, same iteration count, CPU vs GPU: every action node's average
/// strategy must agree within float-order noise, and the BR gaps must match.
#[test]
fn gpu_matches_cpu() {
    let eq = table();
    let mut cpu = PreflopSolver::new(hu25(), eq.clone()).unwrap();
    let mut gs = PreflopSolver::new(hu25(), eq).unwrap();
    // the GPU mirrors the UNPRUNED traversal bit-for-bit
    cpu.prune = false;
    gs.prune = false;
    let mut g = PreflopGpu::new(&gs, 8_000).expect("gpu init");

    for _ in 0..300 {
        cpu.iterate();
    }
    for _ in 0..300 {
        g.iterate(&mut gs).unwrap();
    }
    g.sync_to_cpu(&mut gs).unwrap();
    assert_eq!(cpu.iteration, gs.iteration);

    let mut worst = 0f32;
    for i in 0..cpu.nodes.len() {
        if cpu.nodes[i].kind != 0 {
            continue;
        }
        let a = cpu.average_strategy(i);
        let b = gs.average_strategy(i);
        for (x, y) in a.iter().zip(b.iter()) {
            worst = worst.max((x - y).abs());
        }
    }
    assert!(
        worst < 0.05,
        "CPU and GPU average strategies diverge: max |diff| = {worst}"
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
