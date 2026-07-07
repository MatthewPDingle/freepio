//! M5 Phase A: realization observations are sane and the flop enumeration
//! is the known combinatorial result.

use solver::tree::{parse_sizes, StreetSizing, TreeConfig};
use solver::{Solver, Spot, SpotConfig};
use std::sync::Arc;

#[test]
fn canonical_flops_are_1755() {
    let flops = solver::cards::canonical_flops();
    assert_eq!(flops.len(), 1755);
    assert_eq!(flops.iter().map(|x| x.1 as u64).sum::<u64>(), 22100);
    assert_eq!(flops, solver::cards::canonical_flops(), "must be deterministic");
}

fn sizing(bet: &str, raise: &str) -> StreetSizing {
    StreetSizing {
        bet: parse_sizes(bet).unwrap(),
        raise: parse_sizes(raise).unwrap(),
        donk: vec![],
    }
}

#[test]
fn realization_observations_are_sane() {
    // IDENTICAL ranges both sides: any realization asymmetry is then purely
    // positional, so "IP over-realizes" must hold. (With asymmetric ranges
    // the range-advantage side can over-realize instead — that's a feature
    // of the data, not a test invariant.)
    let range = "22+,A2s+,K9s+,QTs+,JTs,T9s,98s,87s,76s,ATo+,KJo+,QJo";
    let s = |_p: usize| [sizing("50", "100"), sizing("66", ""), sizing("66", "")];
    let spot = Spot::new(SpotConfig {
        board: "Ks7h2d".to_string(),
        range_oop: range.to_string(),
        range_ip: range.to_string(),
        tree: TreeConfig {
            starting_pot: 10.0,
            effective_stack: 40.0,
            oop: s(0),
            ip: s(1),
            max_raises: 2,
            ..Default::default()
        },
    })
    .unwrap();
    let mut solver = Solver::new(Arc::new(spot));
    for _ in 0..500 {
        solver.iterate();
    }
    let obs = solver.realization_observations().unwrap();
    assert!(obs.len() > 20, "expected observations for both players, got {}", obs.len());
    assert!(obs.iter().any(|o| o.player == 0) && obs.iter().any(|o| o.player == 1));
    for o in &obs {
        // bottom-of-range hands legitimately realize ~0 (check-fold ≈ fold
        // EV = 0 in pot-share) — that spread IS what M5 calibrates
        assert!(o.r_obs.is_finite() && o.r_obs > -0.05 && o.r_obs < 3.5,
            "wild r_obs {} for {} (player {})", o.r_obs, o.label, o.player);
        assert!((o.spr - 4.0).abs() < 1e-9);
        assert_eq!(o.board, "Ks7h2d");
    }
    // position premium: IP's reach-weighted mean realization beats OOP's
    let mean = |p: u8| {
        let (mut n, mut d) = (0f64, 0f64);
        for o in obs.iter().filter(|o| o.player == p) {
            n += o.r_obs * o.reach;
            d += o.reach;
        }
        n / d
    };
    let (moop, mip) = (mean(0), mean(1));
    assert!(
        mip > moop,
        "IP should over-realize vs OOP: OOP {moop:.3} IP {mip:.3}"
    );
}
