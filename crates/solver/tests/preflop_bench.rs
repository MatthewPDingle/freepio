//! Manual benchmark for regret-based pruning (not part of the suite):
//! cargo test --release --test preflop_bench -- --ignored --nocapture

use solver::preflop::equity::EquityTable;
use solver::preflop::{PreflopConfig, PreflopSolver};
use std::sync::Arc;
use std::time::Instant;

#[test]
#[ignore]
fn bench_pruning_six_max() {
    let eq = Arc::new(EquityTable::load_or_build("cache/preflop_eq169.bin", 20000));
    let cfg = PreflopConfig {
        positions: vec![
            "UTG".into(),
            "HJ".into(),
            "CO".into(),
            "BTN".into(),
            "SB".into(),
            "BB".into(),
        ],
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
    for prune in [false, true] {
        let mut s = PreflopSolver::new(cfg.clone(), eq.clone()).unwrap();
        s.prune = prune;
        // warm both runs identically past the pruning warmup
        let warm: u32 = std::env::var("BENCH_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
        for _ in 0..warm {
            s.iterate();
        }
        let t0 = Instant::now();
        for _ in 0..60 {
            s.iterate();
        }
        let dt = t0.elapsed().as_secs_f64();
        let gaps = s.br_gaps();
        println!(
            "prune={prune}: {:.2} it/s ({:.2}s / 60 iters) · gap_total {:.4}",
            60.0 / dt,
            dt,
            gaps.iter().sum::<f64>()
        );
    }
}
