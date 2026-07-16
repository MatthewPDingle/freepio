//! Regression tests for lock validation on the save/load path.
//!
//! A save whose header parses but whose lock entries are malformed used to be
//! installed unvalidated; the panic then fired at the first query (strategy
//! copy, ensure_symmetric, GPU lock table) — after the caller had already
//! discarded its previous session. Load must instead return a clear Err.

use solver::save::validate_locks;
use solver::tree::{parse_sizes, StreetSizing, TreeConfig, KIND_ACTION};
use solver::{LockMode, Solver, Spot, SpotConfig};
use std::sync::Arc;

fn sizing(bet: &str, raise: &str) -> StreetSizing {
    StreetSizing {
        bet: parse_sizes(bet).unwrap(),
        raise: parse_sizes(raise).unwrap(),
        donk: vec![],
    }
}

/// Tiny river-only clairvoyance spot: cheap to build and solve.
fn tiny_config() -> SpotConfig {
    SpotConfig {
        board: "QcJc9c3d2d".to_string(),
        range_oop: "AcKc,8h7h".to_string(),
        range_ip: "QdQh".to_string(),
        tree: TreeConfig {
            starting_pot: 100.0,
            effective_stack: 100.0,
            oop: [sizing("", ""), sizing("", ""), sizing("100", "")],
            ip: [sizing("", ""), sizing("", ""), sizing("", "")],
            ..Default::default()
        },
    }
}

/// Build, briefly solve, and save the tiny spot; returns (solver, path).
fn solved_save(file: &str) -> (Solver, String) {
    let spot = Spot::new(tiny_config()).unwrap();
    let mut solver = Solver::new(Arc::new(spot));
    for _ in 0..50 {
        solver.iterate();
    }
    let path = std::env::temp_dir().join(file);
    let path = path.to_str().unwrap().to_string();
    solver.save(&path).unwrap();
    (solver, path)
}

/// Load that must fail; returns the error (Solver has no Debug for expect_err).
fn load_err(path: &str, what: &str) -> String {
    match Solver::load(path) {
        Ok(_) => panic!("{what}: load unexpectedly succeeded"),
        Err(e) => e,
    }
}

/// Rewrite the save header's "locks" field in place (the header is a JSON
/// line between the magic and the arenas; the arenas are kept verbatim).
fn doctor_locks(path: &str, locks: serde_json::Value) {
    let bytes = std::fs::read(path).unwrap();
    const MAGIC_LEN: usize = 10; // b"GTOSOLVE2\n"
    let nl = MAGIC_LEN + bytes[MAGIC_LEN..].iter().position(|&b| b == b'\n').unwrap();
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[MAGIC_LEN..nl]).unwrap();
    header["locks"] = locks;
    let mut out = bytes[..MAGIC_LEN].to_vec();
    out.extend_from_slice(serde_json::to_string(&header).unwrap().as_bytes());
    out.extend_from_slice(&bytes[nl..]); // '\n' + arenas, untouched
    std::fs::write(path, out).unwrap();
}

/// Well-formed locks still round-trip, and the loaded solver serves queries.
#[test]
fn lock_survives_save_load_roundtrip() {
    let spot = Spot::new(tiny_config()).unwrap();
    let mut solver = Solver::new(Arc::new(spot));
    for _ in 0..50 {
        solver.iterate();
    }
    solver
        .lock_node(&[], LockMode::Freeze, "root".into())
        .unwrap();
    let path = std::env::temp_dir().join("gto_test_lock_roundtrip.gto");
    let path = path.to_str().unwrap().to_string();
    solver.save(&path).unwrap();

    let loaded = Solver::load(&path).unwrap();
    assert_eq!(loaded.locks.len(), 1);
    for (idx, sigma) in &solver.locks {
        assert_eq!(loaded.locks.get(idx), Some(sigma));
    }
    // the previously panicking consumers must work on the loaded solver
    let view = loaded.node_view(&[]).unwrap();
    assert!(view.locked, "root lock lost in roundtrip");
    loaded.exploit_view(&[], 0).unwrap();
    std::fs::remove_file(path).ok();
}

/// An out-of-range node index must be refused at load, not panic at first use.
#[test]
fn load_rejects_out_of_range_lock_index() {
    let (_solver, path) = solved_save("gto_test_lock_oob.gto");
    doctor_locks(&path, serde_json::json!([[999_999, [0.25, 0.25, 0.25, 0.25]]]));
    let err = load_err(&path, "out-of-range lock index");
    assert!(
        err.contains("lock") && err.contains("out of range"),
        "unexpected error: {err}"
    );
    std::fs::remove_file(path).ok();
}

/// A lock on a non-action (terminal/chance) node must be refused at load.
#[test]
fn load_rejects_non_action_node_lock() {
    let (solver, path) = solved_save("gto_test_lock_nonaction.gto");
    let idx = solver
        .spot
        .tree
        .nodes
        .iter()
        .position(|n| n.kind != KIND_ACTION)
        .expect("tree has terminal nodes") as u32;
    doctor_locks(&path, serde_json::json!([[idx, [0.5, 0.5]]]));
    let err = load_err(&path, "non-action lock");
    assert!(
        err.contains("not an action node"),
        "unexpected error: {err}"
    );
    std::fs::remove_file(path).ok();
}

/// A sigma with the wrong length (the copy_from_slice panic) must be refused.
#[test]
fn load_rejects_wrong_sigma_length() {
    let (_solver, path) = solved_save("gto_test_lock_shortsigma.gto");
    doctor_locks(&path, serde_json::json!([[0, [0.5]]]));
    let err = load_err(&path, "short lock sigma");
    assert!(
        err.contains("expected") && err.contains("lock"),
        "unexpected error: {err}"
    );
    std::fs::remove_file(path).ok();
}

/// Negative frequencies must be refused even when the shape is right.
#[test]
fn load_rejects_negative_lock_value() {
    let (solver, path) = solved_save("gto_test_lock_negative.gto");
    let root = &solver.spot.tree.nodes[0];
    assert_eq!(root.kind, KIND_ACTION);
    let n = root.num_children as usize * solver.spot.hands[root.player as usize].len();
    let mut sigma = vec![0.5f32; n];
    sigma[0] = -1.0;
    doctor_locks(&path, serde_json::json!([[0, sigma]]));
    let err = load_err(&path, "negative lock value");
    assert!(
        err.contains("finite") || err.contains("invalid"),
        "unexpected error: {err}"
    );
    std::fs::remove_file(path).ok();
}

/// Non-finite values can't ride through JSON, but a caller-supplied vector
/// can carry them: validate_locks (the function the server's peek path uses)
/// must reject NaN/inf directly.
#[test]
fn validate_locks_rejects_non_finite() {
    let spot = Spot::new(tiny_config()).unwrap();
    let root = &spot.tree.nodes[0];
    let n = root.num_children as usize * spot.hands[root.player as usize].len();
    assert!(validate_locks(&spot, &[(0, vec![0.5f32; n])]).is_ok());
    assert!(validate_locks(&spot, &[(0, vec![f32::NAN; n])]).is_err());
    assert!(validate_locks(&spot, &[(0, vec![f32::INFINITY; n])]).is_err());
}
