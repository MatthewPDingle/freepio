//! Load compatibility for pre-validation .gto saves.
//!
//! A save written before the raise-only-size build validation existed can
//! carry a "2.5x" token in a consumed bet/donk list: the builder of that era
//! silently dropped the size, so the saved arenas were sized against the
//! dropped-size tree. The load path must rebuild that exact tree (leniently)
//! and load the file, while the strict build path keeps rejecting the same
//! config for NEW builds.

use solver::tree::{parse_sizes, StreetSizing, TreeConfig};
use solver::{Solver, Spot, SpotConfig};
use std::sync::Arc;

fn sizing(bet: &str, raise: &str) -> StreetSizing {
    StreetSizing {
        bet: parse_sizes(bet).unwrap(),
        raise: parse_sizes(raise).unwrap(),
        donk: vec![],
    }
}

/// Regression: pre-fix save whose stored config still contains the 'x' token
/// in the OOP flop bet list (the finding's "2.5x, 50" scenario). Built the
/// OLD way — the spot is constructed WITHOUT the token, exactly as the
/// pre-fix builder ended up doing, then the token is injected back into the
/// stored config JSON to make the file byte-equivalent to a real pre-fix
/// save.
#[test]
fn pre_fix_save_with_raise_multiple_in_bet_list_still_loads() {
    // 1. Solve + save what the pre-fix builder actually produced for a user
    //    who typed "2.5x, 50": a tree with just the "50".
    let config = SpotConfig {
        board: "Th9h2c".to_string(),
        range_oop: "QQ,JJ,TT,99".to_string(),
        range_ip: "AA,KK".to_string(),
        tree: TreeConfig {
            starting_pot: 60.0,
            effective_stack: 200.0,
            oop: [sizing("50", "60"), sizing("50", ""), sizing("50", "")],
            ip: [sizing("50", "60"), sizing("50", ""), sizing("50", "")],
            ..Default::default()
        },
    };
    let spot = Spot::new(config).unwrap();
    let mut solver = Solver::new(Arc::new(spot));
    for _ in 0..20 {
        solver.iterate();
    }
    let e_before = solver.exploitability();
    let path = std::env::temp_dir().join("gto_test_prefix_prevmult.gto");
    let path_str = path.to_str().unwrap();
    solver.save(path_str).unwrap();

    // 2. Doctor the stored config into what a pre-fix save really holds: the
    //    bet list still CONTAINS the 2.5x token ahead of the 50.
    let bytes = std::fs::read(path_str).unwrap();
    let nl = 10 + bytes[10..].iter().position(|&b| b == b'\n').unwrap();
    let mut hdr: serde_json::Value = serde_json::from_slice(&bytes[10..nl]).unwrap();
    hdr["config"]["tree"]["oop"][0]["bet"]
        .as_array_mut()
        .unwrap()
        .insert(0, serde_json::json!({ "PrevMult": 2.5 }));
    let mut doctored = bytes[..10].to_vec();
    doctored.extend_from_slice(serde_json::to_string(&hdr).unwrap().as_bytes());
    doctored.extend_from_slice(&bytes[nl..]);
    std::fs::write(path_str, doctored).unwrap();

    // 3. The strict path must keep rejecting this config for new builds...
    let stored_config: SpotConfig = serde_json::from_value(hdr["config"].clone()).unwrap();
    let err = Spot::new(stored_config.clone())
        .err()
        .expect("strict build must reject the stored config");
    assert!(
        err.contains("2.5x"),
        "strict rejection should name the token: {err}"
    );

    // 4. ...but the load path rebuilds leniently: the arena length checks
    //    pass and the solve comes back intact.
    let loaded = Solver::load(path_str).expect("pre-fix save must keep loading");
    assert_eq!(loaded.iteration, solver.iteration);
    let e_after = loaded.exploitability();
    assert!(
        (e_before - e_after).abs() < 1e-9,
        "loaded solve differs: {e_before} vs {e_after}"
    );

    // 5. The capped lenient constructor (the save-vetting path) must agree
    //    with the real load on the tree layout.
    let vetted = Spot::new_lenient_with_limit(stored_config, Some(10_000_000))
        .expect("lenient vetting rebuild must succeed");
    assert_eq!(vetted.tree.data_size, loaded.spot.tree.data_size);
    assert_eq!(vetted.tree.nodes.len(), loaded.spot.tree.nodes.len());

    std::fs::remove_file(path).ok();
}
