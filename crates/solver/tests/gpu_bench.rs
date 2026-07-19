#![cfg(feature = "gpu")]
use solver::preflop::equity::EquityTable;
use solver::preflop::{gpu::PreflopGpu, PreflopConfig, PreflopSolver};
use std::sync::Arc;
use std::time::Instant;

#[test]
#[ignore] // manual benchmark: cargo test --release --features gpu --test gpu_bench -- --ignored --nocapture
fn bench_gpu_six_max() {
    let eq = Arc::new(EquityTable::build(20000));
    let cfg = PreflopConfig {
        positions: vec!["UTG".into(), "HJ".into(), "CO".into(), "BTN".into(), "SB".into(), "BB".into()],
        stack: 100.0,
        posts: vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0],
        ante: 0.0,
        limp: true,
        open_raises: vec![2.5, 4.0],
        raise_mults: vec![3.0],
        max_raises: 3,
        add_allin: false,
        allin_threshold: 0.85,
        rake_pct: 5.0,
        rake_cap: 3.0,
        no_flop_no_drop: true,
        realization: "static".into(),
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
    };
    let mut cpu = PreflopSolver::new(cfg.clone(), eq.clone()).unwrap();
    println!("nodes: {}", cpu.nodes.len());
    let t0 = Instant::now();
    for _ in 0..30 { cpu.iterate(); }
    println!("CPU (pruned, {} threads): {:.2} it/s", rayon::current_num_threads(), 30.0 / t0.elapsed().as_secs_f64());

    let mut gs = PreflopSolver::new(cfg, eq).unwrap();
    let mut g = PreflopGpu::new(&gs, 20_000).expect("gpu init");
    for _ in 0..5 { g.iterate(&mut gs).unwrap(); } // warm
    let t0 = Instant::now();
    for _ in 0..30 { g.iterate(&mut gs).unwrap(); }
    println!("GPU 4090: {:.2} it/s", 30.0 / t0.elapsed().as_secs_f64());
}
