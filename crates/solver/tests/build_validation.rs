//! Build-time validation: misconfigured sizes, out-of-range rake, and
//! over-budget trees must be rejected with clear errors instead of silently
//! solving an unintended tree (or OOMing mid-build).

use solver::tree::{parse_sizes, StreetSizing, TreeConfig};
use solver::{Spot, SpotConfig};

fn sizing(bet: &str, raise: &str) -> StreetSizing {
    StreetSizing {
        bet: parse_sizes(bet).unwrap(),
        raise: parse_sizes(raise).unwrap(),
        donk: vec![],
    }
}

fn flop_config() -> SpotConfig {
    SpotConfig {
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
    }
}

/// A raise-multiple in a bet list used to be silently dropped at tree build
/// (the c-bet the user configured just vanished); it must now be a build
/// error naming the offending field and token.
#[test]
fn raise_multiple_in_bet_list_is_a_build_error() {
    let mut config = flop_config();
    config.tree.oop[0].bet = parse_sizes("2.5x, 50").unwrap();
    let err = Spot::new(config).err().expect("expected a build error");
    assert!(
        err.contains("flop OOP bet") && err.contains("2.5x"),
        "error should name the field and token: {err}"
    );
}

/// Same for donk lists: with add_allin set, a donk list holding only "2.5x"
/// used to silently degrade to all-in-only donks.
#[test]
fn raise_multiple_in_donk_list_is_a_build_error() {
    let mut config = flop_config();
    config.tree.add_allin = true;
    config.tree.oop[1].donk = parse_sizes("3x").unwrap();
    let err = Spot::new(config).err().expect("expected a build error");
    assert!(
        err.contains("turn OOP donk") && err.contains("3x"),
        "error should name the field and token: {err}"
    );
}

/// Positive control: multiples stay valid in raise lists.
#[test]
fn raise_multiples_in_raise_lists_still_build() {
    let mut config = flop_config();
    config.tree.oop[0].raise = parse_sizes("2.5x").unwrap();
    config.tree.ip[0].raise = parse_sizes("3x, a").unwrap();
    Spot::new(config).expect("multiples are valid raise sizes");
}

/// Lists below the root street are never consumed, so junk there must not
/// block a turn/river solve that previously built fine.
#[test]
fn raise_multiple_below_root_street_is_ignored() {
    let mut config = flop_config();
    config.board = "Th9h2c5d".to_string(); // turn root
    config.tree.oop[0].bet = parse_sizes("2.5x").unwrap(); // unused flop list
    Spot::new(config).expect("flop sizing is unused for a turn root");
}

/// IP can never donk (donking is OOP leading into the aggressor), so the IP
/// donk list is never consumed and must not fail the build.
#[test]
fn raise_multiple_in_ip_donk_list_is_ignored() {
    let mut config = flop_config();
    config.tree.ip[1].donk = parse_sizes("2.5x").unwrap();
    Spot::new(config).expect("IP donk list is never consumed");
}

/// Lenient (save-load) builds must reproduce the legacy builder exactly: the
/// raise-only size is dropped silently and the tree layout is byte-identical
/// to a build without it — pre-fix .gto arenas were sized against that tree.
#[test]
fn lenient_build_matches_legacy_dropped_size_layout() {
    let mut with_token = flop_config();
    with_token.tree.oop[0].bet = parse_sizes("2.5x, 50").unwrap();
    let legacy = flop_config(); // same config, minus the dropped "2.5x"

    assert!(
        Spot::new(with_token.clone()).is_err(),
        "strict build must keep rejecting the raise-only bet size"
    );
    let lenient = Spot::new_lenient(with_token).expect("lenient build must succeed");
    let legacy = Spot::new(legacy).unwrap();
    assert_eq!(lenient.tree.data_size, legacy.tree.data_size);
    assert_eq!(lenient.tree.children, legacy.tree.children);
    assert_eq!(lenient.tree.actions, legacy.tree.actions);
    assert_eq!(
        serde_json::to_string(&lenient.tree.nodes).unwrap(),
        serde_json::to_string(&legacy.tree.nodes).unwrap(),
        "lenient rebuild must be byte-identical to the legacy dropped-size tree"
    );
}

/// The donk-list variant of the same invariant. Legacy subtlety: with
/// add_allin set, a donk list holding only "3x" degraded to all-in-only
/// donks (the list counted as non-empty, so the all-in was still offered) —
/// NOT to no donks at all. The lenient build must match that, i.e. equal a
/// donk list of just "a".
#[test]
fn lenient_build_matches_legacy_allin_only_donks() {
    let mut with_token = flop_config();
    with_token.tree.add_allin = true;
    with_token.tree.oop[1].donk = parse_sizes("3x").unwrap();
    let mut legacy = flop_config();
    legacy.tree.add_allin = true;
    legacy.tree.oop[1].donk = parse_sizes("a").unwrap();

    assert!(Spot::new(with_token.clone()).is_err());
    let lenient = Spot::new_lenient(with_token).expect("lenient build must succeed");
    let legacy = Spot::new(legacy).unwrap();
    assert_eq!(lenient.tree.data_size, legacy.tree.data_size);
    assert_eq!(lenient.tree.children, legacy.tree.children);
    assert_eq!(lenient.tree.actions, legacy.tree.actions);
}

/// Lenient mode relaxes only the sizing check — the node budget must still
/// abort an oversized lenient rebuild (peek_save vets under a memory cap).
#[test]
fn lenient_build_still_enforces_node_budget() {
    let mut config = flop_config();
    config.tree.oop[0].bet = parse_sizes("2.5x, 50").unwrap();
    let err = Spot::new_lenient_with_limit(config, Some(10))
        .err()
        .expect("expected a budget error");
    assert!(err.contains("max_nodes"), "error should name the budget: {err}");
}

/// Negative rake used to pass validation and pay the winner more than the
/// pot at every terminal.
#[test]
fn negative_rake_pct_is_rejected() {
    let mut config = flop_config();
    config.tree.rake_pct = -0.05;
    let err = Spot::new(config).err().expect("expected a build error");
    assert!(err.contains("rake_pct"), "error should name rake_pct: {err}");
}

#[test]
fn negative_rake_cap_is_rejected() {
    let mut config = flop_config();
    config.tree.rake_pct = 0.05;
    config.tree.rake_cap = -1.0;
    let err = Spot::new(config).err().expect("expected a build error");
    assert!(err.contains("rake_cap"), "error should name rake_cap: {err}");
}

/// Positive control: an ordinary rake config still builds.
#[test]
fn valid_rake_still_builds() {
    let mut config = flop_config();
    config.tree.rake_pct = 0.05;
    config.tree.rake_cap = 3.0;
    Spot::new(config).expect("normal rake config must build");
}

/// The node budget must abort the build with a clear error instead of
/// allocating the whole oversized tree first.
#[test]
fn max_nodes_budget_aborts_oversized_build() {
    let config = flop_config();
    // sanity: unlimited build succeeds and is comfortably over the budget
    let full = Spot::new(config.clone()).unwrap();
    assert!(full.tree.nodes.len() > 10);
    let err = Spot::new_with_limit(config, Some(10)).err().expect("expected a build error");
    assert!(
        err.contains("max_nodes") && err.contains("10"),
        "error should name the budget: {err}"
    );
}

/// A budget the tree fits under must not change the build at all.
#[test]
fn max_nodes_budget_admits_trees_within_it() {
    let config = flop_config();
    let n = Spot::new(config.clone()).unwrap().tree.nodes.len();
    let spot = Spot::new_with_limit(config, Some(n + 1000)).expect("within budget");
    assert_eq!(spot.tree.nodes.len(), n, "budget must not change the tree");
}
