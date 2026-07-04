//! Preflop solver validation: CFR vs an independent push/fold oracle,
//! structural sanity of the action grammar, and model invariants.

use solver::preflop::equity::{class_prob, EquityTable, NUM_CLASSES};
use solver::preflop::{PreflopConfig, PreflopSolver};
use std::sync::{Arc, OnceLock};

fn table() -> Arc<EquityTable> {
    static T: OnceLock<Arc<EquityTable>> = OnceLock::new();
    T.get_or_init(|| Arc::new(EquityTable::build(4000))).clone()
}

fn hu_push_fold_config(stack: f64) -> PreflopConfig {
    PreflopConfig {
        positions: vec!["SB".into(), "BB".into()],
        stack,
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
    }
}

/// Independent Nash oracle for heads-up jam/fold at `stack` bb, using the
/// SAME equity table and the same mean-field assumptions as the solver.
/// Fictitious play (best response vs the opponent's AVERAGED strategy),
/// which converges in two-player zero-sum games — pure alternating best
/// responses cycle around the indifference boundary. Returns the averaged
/// (jam, call) frequencies plus each class's final decision margin in bb
/// against the averaged opponent strategy.
fn push_fold_oracle(eq: &EquityTable, stack: f64) -> (Vec<f32>, Vec<f32>, Vec<f64>, Vec<f64>) {
    let prob: Vec<f64> = (0..NUM_CLASSES).map(|h| class_prob(h) as f64).collect();
    let total: f64 = prob.iter().sum();
    let mut jam_mix = vec![1f64; NUM_CLASSES];
    let mut call_mix = vec![0f64; NUM_CLASSES];
    let (mut jam_margin, mut call_margin) = (vec![0f64; NUM_CLASSES], vec![0f64; NUM_CLASSES]);
    for t in 1..=4000u32 {
        // BB best response vs averaged jam range: call risks (stack-1) more
        // to win 2*stack total; folding loses the posted blind.
        let mut dist: Vec<f32> = (0..NUM_CLASSES)
            .map(|j| (prob[j] * jam_mix[j]) as f32)
            .collect();
        let s: f32 = dist.iter().sum();
        if s > 0.0 {
            dist.iter_mut().for_each(|d| *d /= s);
        }
        let mut call_br = vec![0f64; NUM_CLASSES];
        for h in 0..NUM_CLASSES {
            let e = eq.eq_vs_dist(h, &dist) as f64;
            call_margin[h] = (e * 2.0 * stack - stack) - (-1.0);
            call_br[h] = if call_margin[h] > 0.0 { 1.0 } else { 0.0 };
        }
        // SB best response vs averaged call range.
        let p_call: f64 = (0..NUM_CLASSES)
            .map(|j| prob[j] * call_mix[j])
            .sum::<f64>()
            / total;
        let mut cdist: Vec<f32> = (0..NUM_CLASSES)
            .map(|j| (prob[j] * call_mix[j]) as f32)
            .collect();
        let cs: f32 = cdist.iter().sum();
        if cs > 0.0 {
            cdist.iter_mut().for_each(|d| *d /= cs);
        }
        let mut jam_br = vec![0f64; NUM_CLASSES];
        for h in 0..NUM_CLASSES {
            let e = eq.eq_vs_dist(h, &cdist) as f64;
            let ev_jam = (1.0 - p_call) * 1.0 + p_call * (e * 2.0 * stack - stack);
            jam_margin[h] = ev_jam - (-0.5);
            jam_br[h] = if jam_margin[h] > 0.0 { 1.0 } else { 0.0 };
        }
        // fictitious-play averaging
        let w = 1.0 / t as f64;
        for h in 0..NUM_CLASSES {
            call_mix[h] += w * (call_br[h] - call_mix[h]);
            jam_mix[h] += w * (jam_br[h] - jam_mix[h]);
        }
    }
    (
        jam_mix.iter().map(|&x| x as f32).collect(),
        call_mix.iter().map(|&x| x as f32).collect(),
        jam_margin,
        call_margin,
    )
}

/// CFR must reproduce the oracle's push/fold equilibrium for every class
/// whose decision margin is clear (mixing at the indifference boundary is
/// expected and excluded).
#[test]
fn hu_push_fold_matches_oracle() {
    let eq = table();
    let stack = 10.0;
    let mut s = PreflopSolver::new(hu_push_fold_config(stack), eq.clone()).unwrap();
    // tree: SB [Fold, All-in] -> BB [Fold, Call]
    assert_eq!(s.nodes[0].actions.len(), 2, "SB should have fold/jam");
    for _ in 0..4000 {
        s.iterate();
    }

    let (jam, call, jam_margin, call_margin) = push_fold_oracle(&eq, stack);
    let sb = s.average_strategy(0);
    let sb_jam = s.nodes[0]
        .actions
        .iter()
        .position(|a| a.kind == "jam")
        .unwrap();
    let bb_idx = s.child(0, sb_jam);
    let bb = s.average_strategy(bb_idx);
    let bb_call_a = s.nodes[bb_idx]
        .actions
        .iter()
        .position(|a| a.kind == "call")
        .unwrap();
    let sb_jam_a = s.nodes[0]
        .actions
        .iter()
        .position(|a| a.kind == "jam")
        .unwrap();

    let (mut checked, mut skipped) = (0, 0);
    for h in 0..NUM_CLASSES {
        // SB decision
        if jam_margin[h].abs() > 0.10 {
            let f = sb[sb_jam_a * NUM_CLASSES + h];
            let want = jam[h] > 0.5;
            assert!(
                if want { f > 0.85 } else { f < 0.15 },
                "SB class {} ({}): oracle jam={} margin={:.3}bb, CFR jam freq={:.3}",
                h,
                solver::preflop::equity::class_label(h),
                want,
                jam_margin[h],
                f
            );
            checked += 1;
        } else {
            skipped += 1;
        }
        // BB decision
        if call_margin[h].abs() > 0.10 {
            let f = bb[bb_call_a * NUM_CLASSES + h];
            let want = call[h] > 0.5;
            assert!(
                if want { f > 0.85 } else { f < 0.15 },
                "BB class {} ({}): oracle call={} margin={:.3}bb, CFR call freq={:.3}",
                h,
                solver::preflop::equity::class_label(h),
                want,
                call_margin[h],
                f
            );
        }
    }
    assert!(
        checked > 120,
        "oracle should give clear answers for most classes, got {checked} (skipped {skipped})"
    );

    // sanity anchors: at 10bb, AA always jams and always calls; 72o folds to a jam
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let seven_deuce = solver::preflop::equity::class_index(5, 0, false);
    assert!(sb[sb_jam_a * NUM_CLASSES + aa] > 0.99);
    assert!(bb[bb_call_a * NUM_CLASSES + aa] > 0.99);
    assert!(bb[bb_call_a * NUM_CLASSES + seven_deuce] < 0.05);

    // zero-sum (no rake): total EV across players ~ 0
    let evs = s.evs();
    let total: f64 = evs.iter().sum();
    assert!(total.abs() < 0.01, "EVs should sum to ~0, got {total}");

    // convergence: BR gaps small
    let gaps = s.br_gaps();
    for (p, g) in gaps.iter().enumerate() {
        assert!(*g < 0.02, "BR gap for player {p} too big: {g} bb");
    }
}

/// Full grammar: 6-max with limps, an open size and a 3-bet builds a legal
/// tree, converges in the model, and conserves chips (minus rake when on).
#[test]
fn six_max_limp_tree_sanity() {
    let eq = table();
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
        open_raises: vec![2.5],
        raise_mults: vec![3.0],
        max_raises: 3,
        add_allin: false,
        allin_threshold: 0.85,
        rake_pct: 0.0,
        rake_cap: 0.0,
        no_flop_no_drop: true,
        realization: "static".into(),
    };
    let mut s = PreflopSolver::new(cfg, eq.clone()).unwrap();
    let action_nodes = s.nodes.iter().filter(|n| n.kind == 0).count();
    assert!(action_nodes > 100, "tree suspiciously small: {action_nodes}");

    // UTG's root actions must include limp (limp=true), a raise and fold
    let kinds: Vec<&str> = s.nodes[0].actions.iter().map(|a| a.kind.as_str()).collect();
    assert!(kinds.contains(&"fold") && kinds.contains(&"call") && kinds.contains(&"raise"));

    for _ in 0..100 {
        s.iterate();
    }
    let evs = s.evs();
    let total: f64 = evs.iter().sum();
    assert!(
        total.abs() < 0.06,
        "chips should be conserved without rake, got sum {total} ({evs:?})"
    );
    let g1: f64 = s.br_gaps().iter().sum();
    for _ in 0..200 {
        s.iterate();
    }
    let g2: f64 = s.br_gaps().iter().sum();
    assert!(g2 < g1, "BR gap should shrink with iterations: {g1} -> {g2}");
}

/// Rake drains EV: with rake on, total EV goes negative (and not absurdly).
#[test]
fn rake_drains_total_ev() {
    let eq = table();
    let mut cfg = hu_push_fold_config(20.0);
    cfg.open_raises = vec![2.5];
    cfg.max_raises = 2;
    cfg.limp = true;
    cfg.rake_pct = 5.0;
    cfg.rake_cap = 3.0;
    cfg.no_flop_no_drop = true;
    let mut s = PreflopSolver::new(cfg, eq).unwrap();
    for _ in 0..400 {
        s.iterate();
    }
    let total: f64 = s.evs().iter().sum();
    assert!(total < 0.0, "rake should make the game net negative, got {total}");
    assert!(total > -1.0, "rake drain implausibly large: {total}");
}

/// The size estimator must agree exactly with what the builder builds —
/// they share the enumeration logic, and this pins that they stay shared.
#[test]
fn estimate_matches_build() {
    let eq = table();
    let mut cfgs = vec![hu_push_fold_config(10.0)];
    let mut six = hu_push_fold_config(100.0);
    six.positions = vec![
        "UTG".into(), "HJ".into(), "CO".into(), "BTN".into(), "SB".into(), "BB".into(),
    ];
    six.posts = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0];
    six.limp = true;
    six.open_raises = vec![2.5];
    six.raise_mults = vec![3.0];
    six.max_raises = 2;
    six.add_allin = false;
    cfgs.push(six);
    for cfg in cfgs {
        let est = solver::preflop::estimate_tree(&cfg).unwrap();
        let s = PreflopSolver::new(cfg, eq.clone()).unwrap();
        assert!(!est.truncated);
        assert_eq!(est.nodes as usize, s.nodes.len(), "node count mismatch");
        assert_eq!(
            est.action_nodes as usize,
            s.nodes.iter().filter(|n| n.kind == 0).count(),
            "action node mismatch"
        );
        assert!((est.arena_len as f64 * 8.0 / 1e6 - s.arena_mb()).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Player profiles (P1): buckets, seat modes, behavioral anchors
// ---------------------------------------------------------------------------

use solver::preflop::{
    BucketPolicy, SeatProfile, BUCKET_SQUEEZE, BUCKET_UNOPENED, BUCKET_VS_3BET, BUCKET_VS_LIMPS,
    BUCKET_VS_RAISE, NUM_BUCKETS,
};

fn hu_limp_config() -> PreflopConfig {
    PreflopConfig {
        positions: vec!["SB".into(), "BB".into()],
        stack: 25.0,
        posts: vec![0.5, 1.0],
        ante: 0.0,
        limp: true,
        open_raises: vec![2.5, 5.0],
        raise_mults: vec![3.0],
        max_raises: 3,
        add_allin: false,
        allin_threshold: 0.85,
        rake_pct: 0.0,
        rake_cap: 0.0,
        no_flop_no_drop: true,
        realization: "static".into(),
    }
}

fn flat_policy(call: f32, raise: f32) -> BucketPolicy {
    BucketPolicy {
        call: vec![call; NUM_CLASSES],
        raise: vec![raise; NUM_CLASSES],
        jam: vec![0.0; NUM_CLASSES],
        raise_size: "max".into(),
    }
}

fn profile_with(bucket: u8, pol: BucketPolicy, name: &str) -> SeatProfile {
    let mut buckets: Vec<Option<BucketPolicy>> = vec![None; NUM_BUCKETS];
    buckets[bucket as usize] = Some(pol);
    SeatProfile { name: name.into(), buckets }
}

fn agg_freq(s: &PreflopSolver, node: usize, act_pred: impl Fn(&str) -> bool) -> f64 {
    let sigma = s.average_strategy(node);
    let nd = &s.nodes[node];
    let (mut num, mut den) = (0f64, 0f64);
    for h in 0..NUM_CLASSES {
        let w = class_prob(h) as f64;
        den += w;
        for (a, act) in nd.actions.iter().enumerate() {
            if act_pred(&act.kind) {
                num += w * sigma[a * NUM_CLASSES + h] as f64;
            }
        }
    }
    num / den
}

/// Situation buckets tag the tree the way a player would describe spots.
#[test]
fn buckets_are_tagged_correctly() {
    let eq = table();
    let s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    assert_eq!(s.nodes[0].bucket, BUCKET_UNOPENED);
    let limp = s.nodes[0].actions.iter().position(|a| a.kind == "call").unwrap();
    let open = s.nodes[0].actions.iter().position(|a| a.kind == "raise").unwrap();
    assert_eq!(s.nodes[s.child(0, limp)].bucket, BUCKET_VS_LIMPS);
    let vs_raise = s.child(0, open);
    assert_eq!(s.nodes[vs_raise].bucket, BUCKET_VS_RAISE);
    let threebet = s.nodes[vs_raise].actions.iter().position(|a| a.kind == "raise").unwrap();
    assert_eq!(s.nodes[s.child(vs_raise, threebet)].bucket, BUCKET_VS_3BET);

    // squeeze needs 3 players: BTN opens, SB calls, BB faces raise + caller
    let mut cfg3 = hu_limp_config();
    cfg3.positions = vec!["BTN".into(), "SB".into(), "BB".into()];
    cfg3.posts = vec![0.0, 0.5, 1.0];
    let s3 = PreflopSolver::new(cfg3, eq).unwrap();
    let open3 = s3.nodes[0].actions.iter().position(|a| a.kind == "raise").unwrap();
    let n1 = s3.child(0, open3); // SB facing the raise
    assert_eq!(s3.nodes[n1].bucket, BUCKET_VS_RAISE);
    let call3 = s3.nodes[n1].actions.iter().position(|a| a.kind == "call").unwrap();
    let n2 = s3.child(n1, call3); // BB facing raise + cold caller
    assert_eq!(s3.nodes[n2].bucket, BUCKET_SQUEEZE);
}

/// Anchor: against a BB who NEVER 3-bets, the SB attacks more (raise+limp
/// pressure up, and specifically the raise frequency should not shrink).
#[test]
fn never_threebettor_gets_attacked_wider() {
    let eq = table();
    let mut base = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    for _ in 0..400 {
        base.iterate();
    }
    let base_raise = agg_freq(&base, 0, |k| k == "raise" || k == "jam");

    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    let pol = flat_policy(0.5, 0.0); // calls half of everything, raises nothing
    s.set_table(
        vec![false, false],
        vec![None, Some(profile_with(BUCKET_VS_RAISE, pol, "never-3bets"))],
    )
    .unwrap();
    for _ in 0..400 {
        s.iterate();
    }
    let vs_nit_raise = agg_freq(&s, 0, |k| k == "raise" || k == "jam");
    assert!(
        vs_nit_raise > base_raise + 0.02,
        "SB should open wider vs a never-3-bettor: {base_raise:.3} -> {vs_nit_raise:.3}"
    );
}

/// Anchor: an OMC who only raises AA/KK (max size) gets respect — the BB
/// mostly folds QQ to the raise and never folds AA.
#[test]
fn omc_raises_get_respect() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let kk = solver::preflop::equity::class_index(11, 11, false);
    let qq = solver::preflop::equity::class_index(10, 10, false);
    let mut pol = flat_policy(0.0, 0.0);
    pol.raise[aa] = 1.0;
    pol.raise[kk] = 1.0;
    // limps all pairs below KK and strong suited stuff; folds the rest
    for h in 0..NUM_CLASSES {
        if h != aa && h != kk {
            let (hi, lo, suited) = solver::preflop::equity::class_parts(h);
            if hi == lo || (suited && hi >= 8) {
                pol.call[h] = 1.0;
            }
        }
    }
    s.set_table(
        vec![false, false],
        vec![Some(profile_with(BUCKET_UNOPENED, pol, "OMC")), None],
    )
    .unwrap();
    for _ in 0..500 {
        s.iterate();
    }
    // SB's aggregate open-raise frequency == AA+KK share of the deck
    let raise_freq = agg_freq(&s, 0, |k| k == "raise");
    assert!(
        (raise_freq - 12.0 / 1326.0).abs() < 0.004,
        "OMC raise freq should be ~0.9%, got {raise_freq:.4}"
    );
    // BB facing the max-size raise: QQ mostly folds, AA never does
    let nd0 = &s.nodes[0];
    let raises: Vec<(f64, usize)> = nd0
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.kind == "raise")
        .map(|(i, a)| (a.to, i))
        .collect();
    let max_raise = raises
        .iter()
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap()
        .1;
    let bb_node = s.child(0, max_raise);
    let bb = s.average_strategy(bb_node);
    let fold_a = s.nodes[bb_node].actions.iter().position(|a| a.kind == "fold").unwrap();
    assert!(
        bb[fold_a * NUM_CLASSES + qq] > 0.5,
        "QQ should mostly fold to the OMC raise, folds {:.3}",
        bb[fold_a * NUM_CLASSES + qq]
    );
    assert!(
        bb[fold_a * NUM_CLASSES + aa] < 0.05,
        "AA should never fold to the OMC raise"
    );
}

/// Anchor: a whale who never folds bleeds, and exploiting him raises the
/// other seat's EV vs baseline.
#[test]
fn whale_bleeds_and_gets_exploited() {
    let eq = table();
    let mut base = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    for _ in 0..400 {
        base.iterate();
    }
    let base_ev_sb = base.evs()[0];

    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    let mut buckets: Vec<Option<BucketPolicy>> = vec![None; NUM_BUCKETS];
    for b in 0..NUM_BUCKETS {
        buckets[b] = Some(flat_policy(1.0, 0.0)); // never folds, never raises
    }
    s.set_table(
        vec![false, false],
        vec![None, Some(SeatProfile { name: "whale".into(), buckets })],
    )
    .unwrap();
    for _ in 0..400 {
        s.iterate();
    }
    let ev_sb = s.evs()[0];
    assert!(
        ev_sb > base_ev_sb + 0.15,
        "SB should exploit the whale: EV {base_ev_sb:.3} -> {ev_sb:.3}"
    );
    let bleed = s.br_gaps()[1];
    assert!(bleed > 0.3, "the whale should bleed plainly, got {bleed:.3} bb");
}

/// Frozen seats keep their exact average strategy while others keep moving.
#[test]
fn frozen_seat_stops_adapting() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    let open = s.nodes[0].actions.iter().position(|a| a.kind == "raise").unwrap();
    let bb_node = s.child(0, open);
    let before = s.average_strategy(bb_node);
    let sb_before = s.average_strategy(0);
    s.set_table(vec![false, true], vec![None, None]).unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    let after = s.average_strategy(bb_node);
    let sb_after = s.average_strategy(0);
    let bb_moved = before
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let sb_moved = sb_before
        .iter()
        .zip(sb_after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(bb_moved < 1e-4, "frozen BB moved: {bb_moved}");
    assert!(sb_moved > 1e-3, "live SB should keep adapting, moved {sb_moved}");
}

/// Point locks pin a single node and unlock cleanly.
#[test]
fn point_lock_roundtrip() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..100 {
        s.iterate();
    }
    let before = s.average_strategy(0);
    let pol = flat_policy(0.0, 1.0); // raise everything
    s.lock_point(&[], Some(pol)).unwrap();
    let locked = s.average_strategy(0);
    let raise_mass: f32 = s.nodes[0]
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.kind == "raise")
        .map(|(a, _)| locked[a * NUM_CLASSES])
        .sum();
    assert!(raise_mass > 0.99, "point lock should force raising, got {raise_mass}");
    assert!(s.has_overrides());
    assert!(s.unlock_point(&[]).unwrap());
    let unlocked = s.average_strategy(0);
    let diff = before
        .iter()
        .zip(unlocked.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff < 1e-6, "unlock should restore the solver strategy");
}

/// P2: equilibrium-distortion generation hits its stat targets and puts the
/// right hands in the right slices.
#[test]
fn generated_profiles_match_stats()  {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..400 {
        s.iterate();
    }
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let seven_deuce = solver::preflop::equity::class_index(5, 0, false);

    // a whale: 60/8, mostly position-blind
    let whale_stats = solver::preflop::archetypes()
        .into_iter()
        .find(|(n, _)| n.starts_with("Whale"))
        .unwrap()
        .1;
    let (whale, implied) = s.generate_profile(1, &whale_stats, "whale").unwrap();
    assert!((implied.vpip - 60.0).abs() < 3.0, "implied vpip {}", implied.vpip);
    assert!((implied.pfr - 8.0).abs() < 2.0, "implied pfr {}", implied.pfr);
    let b0 = whale.buckets[BUCKET_UNOPENED as usize].as_ref().unwrap();
    let cont_aa = b0.call[aa] + b0.raise[aa] + b0.jam[aa];
    assert!(cont_aa > 0.99, "AA must be in a 60% range");

    // a nit/OMC: tiny pfr -> premiums raise, junk folds
    let nit_stats = solver::preflop::archetypes()
        .into_iter()
        .find(|(n, _)| n.starts_with("Nit"))
        .unwrap()
        .1;
    let (nit, ni) = s.generate_profile(1, &nit_stats, "nit").unwrap();
    assert!((ni.vpip - 12.0).abs() < 2.5, "implied vpip {}", ni.vpip);
    let b0 = nit.buckets[BUCKET_UNOPENED as usize].as_ref().unwrap();
    assert!(b0.raise[aa] > 0.9, "AA must be in the OMC's raising range");
    let cont_72 = b0.call[seven_deuce] + b0.raise[seven_deuce] + b0.jam[seven_deuce];
    assert!(cont_72 < 0.05, "72o must be out of a 12% range, got {cont_72}");

    // applying a generated profile keeps the solver healthy
    s.set_table(vec![false, false], vec![None, Some(whale)]).unwrap();
    for _ in 0..100 {
        s.iterate();
    }
    let evs = s.evs();
    assert!(evs[0] > 0.0, "SB should profit vs the generated whale: {evs:?}");
}
