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
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
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
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
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
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
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
    SeatProfile { name: name.into(), buckets, postflop: None }
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
    let vs_3bet = s.child(vs_raise, threebet);
    assert_eq!(s.nodes[vs_3bet].bucket, BUCKET_VS_3BET);
    // ...and the SAME bucket at every deeper re-raise: fold-to-3bet+ covers
    // 4-bets, 5-bets, jams (at 25bb the 4-bet clips to a jam — still a raise)
    let fourbet = s.nodes[vs_3bet]
        .actions
        .iter()
        .position(|a| a.kind == "raise" || a.kind == "jam")
        .unwrap();
    assert_eq!(s.nodes[s.child(vs_3bet, fourbet)].bucket, BUCKET_VS_3BET);

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
        vec![None, Some(SeatProfile { name: "whale".into(), buckets, postflop: None })],
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

    // The whale's defend-vs-raise range must be HUMAN-shaped, not
    // equilibrium-polarized: dominated broadways in (fish call Q9o),
    // low suited junk out (fish don't call raises with 53s).
    let bvr = whale.buckets[BUCKET_VS_RAISE as usize].as_ref().unwrap();
    let q9o = solver::preflop::equity::class_index(10, 7, false);
    let three2s = solver::preflop::equity::class_index(1, 0, true);
    let cont = |b: &solver::preflop::BucketPolicy, h: usize| b.call[h] + b.raise[h] + b.jam[h];
    assert!(
        cont(bvr, q9o) > 0.9,
        "a whale calls raises with Q9o, got {}",
        cont(bvr, q9o)
    );
    assert!(
        cont(bvr, three2s) < 0.1,
        "even a whale's 55% defend excludes 32s, got {}",
        cont(bvr, three2s)
    );

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

/// A bucket the baseline never reaches has no equilibrium ordering to
/// distort — its propensities are exact zeros (or float noise once limps
/// decay out of a converged multiway solve), and ranking those used to fill
/// ranges in class-index order: 22, 32o, ... — bottom half of the grid in,
/// every playable hand out. Generation must fall back to card appeal.
/// Repro: no-limp game => VS_LIMPS mass is exactly zero for every seat, and
/// a LOW-naiveté player (whose ordering leans hardest on the equilibrium).
#[test]
fn unreached_bucket_falls_back_to_card_appeal() {
    let eq = table();
    let cfg = PreflopConfig {
        positions: vec!["BTN".into(), "SB".into(), "BB".into()],
        stack: 40.0,
        posts: vec![0.0, 0.5, 1.0],
        ante: 0.0,
        limp: false,
        open_raises: vec![2.5],
        raise_mults: vec![3.0],
        max_raises: 2,
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
    for _ in 0..300 {
        s.iterate();
    }
    let tag_stats = solver::preflop::archetypes()
        .into_iter()
        .find(|(n, _)| n.starts_with("TAG"))
        .unwrap()
        .1;
    let (tag, _) = s.generate_profile(0, &tag_stats, "tag").unwrap();
    let bvl = tag.buckets[BUCKET_VS_LIMPS as usize].as_ref().unwrap();
    let cont = |h: usize| bvl.call[h] + bvl.raise[h] + bvl.jam[h];
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let aks = solver::preflop::equity::class_index(12, 11, true);
    let seven_deuce = solver::preflop::equity::class_index(5, 0, false);
    let three_two = solver::preflop::equity::class_index(1, 0, false);
    assert!(cont(aa) > 0.99, "AA must head an appeal-ordered range, got {}", cont(aa));
    assert!(cont(aks) > 0.99, "AKs must be in a ~22% range, got {}", cont(aks));
    assert!(cont(seven_deuce) < 0.05, "72o out, got {}", cont(seven_deuce));
    assert!(cont(three_two) < 0.05, "32o out, got {}", cont(three_two));

    // A nit at the same data-less seat: defend-vs-raise must come out
    // TIGHTER than unopened (not wider — the old table-numerator over
    // own-denominator scaling gave a 12-VPIP nit a 21% defend), premiums
    // in, and no low-suited-junk polarization borrowed from BB defense.
    let nit_stats = solver::preflop::archetypes()
        .into_iter()
        .find(|(n, _)| n.starts_with("Nit"))
        .unwrap()
        .1;
    let (nit, ni) = s.generate_profile(0, &nit_stats, "nit").unwrap();
    assert!(
        ni.cont_vs_raise < nit_stats.vpip && ni.cont_vs_raise > 2.0,
        "nit defend-vs-raise should be tighter than VPIP {}, got {}",
        nit_stats.vpip,
        ni.cont_vs_raise
    );
    let nvr = nit.buckets[BUCKET_VS_RAISE as usize].as_ref().unwrap();
    let ncont = |h: usize| nvr.call[h] + nvr.raise[h] + nvr.jam[h];
    let qq = solver::preflop::equity::class_index(10, 10, false);
    let six_two_s = solver::preflop::equity::class_index(4, 0, true);
    assert!(ncont(qq) > 0.99, "QQ continues vs a raise, got {}", ncont(qq));
    assert!(ncont(six_two_s) < 0.05, "62s must not be in a nit's defend, got {}", ncont(six_two_s));
}

/// Saved games restore the WHOLE session bit-for-bit: config, solver state,
/// seat models, and the solve can continue from where it stopped.
#[test]
fn game_save_load_roundtrip() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    for _ in 0..120 {
        s.iterate();
    }
    let whale_stats = solver::preflop::archetypes()
        .into_iter()
        .find(|(n, _)| n.starts_with("Whale"))
        .unwrap()
        .1;
    let (prof, _) = s.generate_profile(1, &whale_stats, "whale").unwrap();
    s.set_table(vec![false, false], vec![None, Some(prof)]).unwrap();
    for _ in 0..30 {
        s.iterate();
    }
    let path = std::env::temp_dir().join("gtopen_pf_roundtrip.gtop");
    let path = path.to_str().unwrap().to_string();
    s.save_game(&path).unwrap();
    let l = PreflopSolver::load_game(&path, eq).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(l.iteration, s.iteration);
    assert!(l.seat_profiles[1].is_some(), "seat model must ride along");
    let a = s.average_strategy(0);
    let b = l.average_strategy(0);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-7, "root strategy must survive the roundtrip");
    }
    // resuming the solve from the loaded state works and stays sane
    let mut l = l;
    for _ in 0..20 {
        l.iterate();
    }
    assert!(l.br_gaps().iter().all(|g| g.is_finite()));
}

/// M5 Phase D: the calibrated realization table loads, its monotonicities
/// hold, the solver uses it end-to-end, and all-in anchors are untouched.
#[test]
fn calibrated_realization_works() {
    use solver::preflop::RealizationFit;
    let fit = RealizationFit::load("../../cache/realization_fit.json")
        .expect("shipped fit table");
    let ci = solver::preflop::equity::class_index;
    // aggressor+IP beats defender+OOP at the same holdings
    let m_def = fit.seat_mult(-0.5, 8.0, 0.5, -0.5);
    let m_agg = fit.seat_mult(0.5, 8.0, 0.5, 0.5);
    let t9s = ci(8, 7, true);
    assert!(fit.eval(m_agg, t9s) > fit.eval(m_def, t9s));
    // playability + the v1 postmortem orderings
    let m0 = fit.seat_mult(0.0, 8.0, 0.5, 0.0);
    let (s76, o76, s72) = (ci(5, 4, true), ci(5, 4, false), ci(5, 0, true));
    assert!(fit.eval(m0, s76) > fit.eval(m0, o76));
    assert!(fit.eval(m0, s76) > fit.eval(m0, s72));
    let (a9o, s32) = (ci(7, 12, false), ci(1, 0, true));
    assert!(fit.eval(m_def, a9o) >= fit.eval(m_def, s32), "A9o must not crater below 32s");
    // clip bounds hold at extremes
    for k in [ci(12, 12, false), ci(0, 0, false), ci(5, 0, false)] {
        let r = fit.eval(fit.seat_mult(0.5, 80.0, 0.9, 0.5), k);
        assert!((0.2..=2.5).contains(&r), "R out of clip: {r}");
    }

    // end-to-end: calibrated solves converge and diverge from static
    let eq = table();
    let mut cal_cfg = hu_limp_config();
    cal_cfg.realization = "calibrated".into();
    let mut cal = PreflopSolver::new(cal_cfg, eq.clone()).unwrap();
    assert!(cal.fit.is_some(), "fit must load (cwd fallback path)");
    let mut sta = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    for _ in 0..300 {
        cal.iterate();
        sta.iterate();
    }
    assert!(cal.br_gaps().iter().all(|g| g.is_finite() && *g < 1.0));
    let (a, b) = (cal.average_strategy(0), sta.average_strategy(0));
    let diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    assert!(diff > 0.02, "calibrated should reshape strategies, max diff {diff}");

    // push/fold anchors: all-in terminals bypass R entirely
    let mut pf = hu_push_fold_config(10.0);
    pf.realization = "calibrated".into();
    let mut s = PreflopSolver::new(pf, eq).unwrap();
    for _ in 0..2000 {
        s.iterate();
    }
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let jam_a = s.nodes[0].actions.iter().position(|a| a.kind == "jam").unwrap();
    let sb = s.average_strategy(0);
    assert!(sb[jam_a * NUM_CLASSES + aa] > 0.99, "AA still jams at 10bb");
}

// ---------------------------------------------------------------------------
// Regression tests (review 2026-07-16)
// ---------------------------------------------------------------------------

/// Heads-up the SB IS the button and acts LAST postflop: the BB is OOP.
/// Exports must assign positions accordingly and the static realization
/// premium must go to the SB. 3+ players keep the standard order.
#[test]
fn hu_postflop_position_bb_is_oop() {
    let eq = table();
    let s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    assert_eq!(s.postflop_order(), vec![1, 0], "HU: BB first (OOP), SB/button last");
    // limp, check -> flop terminal
    let limp = s.nodes[0].actions.iter().position(|a| a.kind == "call").unwrap();
    let bb_node = s.child(0, limp);
    let check = s.nodes[bb_node].actions.iter().position(|a| a.kind == "check").unwrap();
    let term = s.child(bb_node, check);
    assert_eq!(s.nodes[term].kind, 2, "expected a pot_share terminal");
    let ex = s.export_spot(&[limp, check]).unwrap();
    assert_eq!(ex.oop_pos, "BB");
    assert_eq!(ex.ip_pos, "SB");
    // static realization: IP premium to the SB (seat 0), discount to the BB
    let nd = &s.nodes[term];
    assert!(
        nd.r[0] > 1.0 && nd.r[1] < 1.0,
        "IP premium must go to the SB/button, got r = {:?}",
        nd.r
    );
    assert!(
        nd.posf[0] > 0.0 && nd.posf[1] < 0.0,
        "SB acts last: posf = {:?}",
        nd.posf
    );
    // 3-handed keeps blinds-first order: SB, BB, then BTN
    let mut cfg3 = hu_limp_config();
    cfg3.positions = vec!["BTN".into(), "SB".into(), "BB".into()];
    cfg3.posts = vec![0.0, 0.5, 1.0];
    let s3 = PreflopSolver::new(cfg3, eq).unwrap();
    assert_eq!(s3.postflop_order(), vec![1, 2, 0]);
}

/// A frozen seat's pinned average must survive any number of solve cycles:
/// the DCFR strategy-sum discount may not touch frozen blocks, or the pins
/// decay below the uniform-fallback floor after a couple of hero switches.
#[test]
fn frozen_average_survives_hero_cycles() {
    let eq = table();
    let mut cfg = hu_push_fold_config(10.0);
    cfg.positions = vec!["BTN".into(), "SB".into(), "BB".into()];
    cfg.posts = vec![0.0, 0.5, 1.0];
    let mut s = PreflopSolver::new(cfg, eq).unwrap();
    for _ in 0..300 {
        s.iterate();
    }
    let node = (0..s.nodes.len())
        .find(|&i| s.nodes[i].kind == 0 && s.nodes[i].actor == 2)
        .unwrap();
    let pinned = s.average_strategy(node);
    // two hero cycles; seat 2 stays frozen through both, and each cycle
    // restarts iteration at 0 where the discount decays hardest
    s.set_hero(Some(0)).unwrap();
    for _ in 0..1500 {
        s.iterate();
    }
    s.set_hero(Some(1)).unwrap();
    for _ in 0..1500 {
        s.iterate();
    }
    let after = s.average_strategy(node);
    let moved = pinned
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        moved < 1e-4,
        "frozen seat's pinned average moved by {moved} across hero cycles"
    );
    let na = s.nodes[node].actions.len();
    let uni = 1.0 / na as f32;
    let max_dev = after.iter().map(|x| (x - uni).abs()).fold(0f32, f32::max);
    assert!(max_dev > 0.3, "frozen average flipped to ~uniform (max dev {max_dev})");
}

/// Leaving hero mode restores the frozen flags the table had before it —
/// explicitly pinned seats must stay pinned.
#[test]
fn hero_off_restores_explicit_frozen_seats() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    s.set_table(vec![false, true], vec![None, None]).unwrap();
    s.set_hero(Some(0)).unwrap();
    for _ in 0..100 {
        s.iterate();
    }
    s.set_hero(None).unwrap();
    assert_eq!(
        s.seat_frozen,
        vec![false, true],
        "hero off must restore the explicitly frozen seat"
    );
}

/// REGRESSION: an explicitly frozen seat must be REFUSED as hero. Hero entry
/// zeroes the hero's strategy sums ("converges fresh") and hero exit restores
/// the pre-hero frozen flags, so pin BB -> set_hero(BB) -> hero off used to
/// re-freeze BB over zeroed sums — the seat then played uniform random
/// forever while labeled frozen (the frozen-block discount skip preserves
/// the zeros), bypassing the uniform-pin guards in set_table and set_hero.
#[test]
fn frozen_seat_refused_as_hero() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_push_fold_config(10.0), eq).unwrap();
    for _ in 0..300 {
        s.iterate();
    }
    // pin BB (seat 1) "as solved", then solve on so learning state is live
    s.set_table(vec![false, true], vec![None, None]).unwrap();
    for _ in 0..50 {
        s.iterate();
    }
    let node = (0..s.nodes.len())
        .find(|&i| s.nodes[i].kind == 0 && s.nodes[i].actor == 1)
        .unwrap();
    let pinned = s.average_strategy(node);
    let iter_before = s.iteration;
    // the confirmed failure sequence starts here — it must be refused
    let err = s.set_hero(Some(1)).unwrap_err();
    assert!(err.contains("frozen"), "refusal must name the freeze, got: {err}");
    // ...and refused BEFORE any state is touched: no hero, flags intact,
    // learning not reset
    assert_eq!(s.seat_frozen, vec![false, true]);
    assert_eq!(s.iteration, iter_before, "refusal must not reset learning");
    s.set_hero(None).unwrap();
    assert_eq!(s.seat_frozen, vec![false, true]);
    // the pinned average survives further solving, and is nowhere near the
    // uniform fallback the old sequence left behind
    for _ in 0..300 {
        s.iterate();
    }
    let after = s.average_strategy(node);
    let moved = pinned
        .iter()
        .zip(after.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(moved < 1e-4, "pinned average moved by {moved}");
    let na = s.nodes[node].actions.len();
    let uni = 1.0 / na as f32;
    let max_dev = after.iter().map(|x| (x - uni).abs()).fold(0f32, f32::max);
    assert!(
        max_dev > 0.3,
        "frozen seat plays ~uniform (max dev {max_dev}) — its pinned average was wiped"
    );
    // the live seat is still a legal hero; while hero mode is active every
    // villain is hero-frozen, so the guard must consult the TABLE's flags:
    // switching hero onto the pinned seat is still refused, switching back
    // off still restores the pin
    s.set_hero(Some(0)).unwrap();
    let err = s.set_hero(Some(1)).unwrap_err();
    assert!(err.contains("frozen"), "hero switch onto the pinned seat, got: {err}");
    assert_eq!(s.hero, Some(0), "failed switch must leave hero mode as it was");
    s.set_hero(None).unwrap();
    assert_eq!(s.seat_frozen, vec![false, true]);
}

/// The one exemption from the frozen-hero refusal: a FULLY ruled seat. Its
/// profile forces every node, so its strategy sums never matter — the
/// hero-entry reset cannot corrupt its play (same exemption set_table's
/// uniform-pin guard makes).
#[test]
fn fully_ruled_frozen_seat_allowed_as_hero() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    let mut buckets: Vec<Option<BucketPolicy>> = vec![None; NUM_BUCKETS];
    for b in 0..NUM_BUCKETS {
        buckets[b] = Some(flat_policy(1.0, 0.0)); // never folds, never raises
    }
    s.set_table(
        vec![true, false],
        vec![Some(SeatProfile { name: "station".into(), buckets, postflop: None }), None],
    )
    .unwrap();
    for _ in 0..50 {
        s.iterate();
    }
    s.set_hero(Some(0)).unwrap();
    for _ in 0..50 {
        s.iterate();
    }
    s.set_hero(None).unwrap();
    assert_eq!(s.seat_frozen, vec![true, false]);
    // the profile rules the seat again — no uniform fallback possible
    let sigma = s.average_strategy(0);
    let pass = s.nodes[0]
        .actions
        .iter()
        .position(|a| a.kind == "check" || a.kind == "call")
        .unwrap();
    let aa = solver::preflop::equity::class_index(12, 12, false);
    assert!(
        (sigma[pass * NUM_CLASSES + aa] - 1.0).abs() < 1e-6,
        "profile must rule the restored frozen seat"
    );
}

/// Hero mode on a fully-ruled seat computes that seat's FREE max exploit:
/// the hero is exempt from its own profile (otherwise the solve is a no-op
/// that returns the profile itself), and the profile snaps back when hero
/// mode ends.
#[test]
fn hero_on_ruled_seat_learns_free_exploit() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    let mut buckets: Vec<Option<BucketPolicy>> = vec![None; NUM_BUCKETS];
    for b in 0..NUM_BUCKETS {
        buckets[b] = Some(flat_policy(1.0, 0.0)); // never folds, never raises
    }
    s.set_table(
        vec![false, false],
        vec![Some(SeatProfile { name: "whale".into(), buckets, postflop: None }), None],
    )
    .unwrap();
    for _ in 0..200 {
        s.iterate();
    }
    let ruled = s.average_strategy(0);
    s.set_hero(Some(0)).unwrap();
    assert_eq!(s.iteration, 0, "hero entry resets the exploit solve");
    for _ in 0..200 {
        s.iterate();
    }
    let exploit = s.average_strategy(0);
    let diff = ruled
        .iter()
        .zip(exploit.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(diff > 0.2, "hero must escape its own profile, max diff {diff}");
    let aa = solver::preflop::equity::class_index(12, 12, false);
    let aggr_aa: f32 = s.nodes[0]
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.kind == "raise" || a.kind == "jam")
        .map(|(a, _)| exploit[a * NUM_CLASSES + aa])
        .sum();
    assert!(aggr_aa > 0.5, "the exploit raises AA, got {aggr_aa}");
    // hero off: the profile rules the seat again
    s.set_hero(None).unwrap();
    let back = s.average_strategy(0);
    let aggr_back: f32 = s.nodes[0]
        .actions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.kind == "raise" || a.kind == "jam")
        .map(|(a, _)| back[a * NUM_CLASSES + aa])
        .sum();
    assert!(aggr_back < 0.01, "profile must rule again after hero off");
}

/// rake_cap = 0 means UNCAPPED (the documented convention, shared with the
/// postflop engine and the UI tooltip) — not "no rake".
#[test]
fn rake_cap_zero_is_uncapped() {
    let eq = table();
    let mut cfg = hu_push_fold_config(20.0);
    cfg.open_raises = vec![2.5];
    cfg.max_raises = 2;
    cfg.limp = true;
    cfg.rake_pct = 10.0;
    cfg.rake_cap = 0.0;
    let mut raked = PreflopSolver::new(cfg.clone(), eq.clone()).unwrap();
    cfg.rake_pct = 0.0;
    let mut free = PreflopSolver::new(cfg, eq).unwrap();
    for _ in 0..400 {
        raked.iterate();
        free.iterate();
    }
    let free_total: f64 = free.evs().iter().sum();
    let raked_total: f64 = raked.evs().iter().sum();
    assert!(free_total.abs() < 0.05, "no-rake game conserves chips, got {free_total}");
    assert!(
        raked_total < -0.05,
        "rake_pct=10 with cap 0 must charge rake (0 = uncapped), got {raked_total}"
    );
}

/// Locking a node "as currently solved" before any solve would pin it to a
/// uniform random sigma — it must be rejected like set_table/set_hero are.
#[test]
fn lock_point_before_solve_is_rejected() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq).unwrap();
    assert!(s.lock_point(&[], None).is_err(), "as-solved lock needs a solve first");
    assert!(!s.has_overrides());
    // an explicit policy is fine at iteration 0
    s.lock_point(&[], Some(flat_policy(1.0, 0.0))).unwrap();
    assert!(s.has_overrides());
}

/// A malformed clip/mult_clip array in the fit table must be a load ERROR
/// (the caller falls back to static realization), never a panic.
#[test]
fn malformed_fit_clip_is_err_not_panic() {
    use solver::preflop::RealizationFit;
    let mut v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("../../cache/realization_fit.json").expect("shipped fit table"),
    )
    .unwrap();
    v["mult_clip"] = serde_json::json!([0.8]); // truncated write / hand edit
    let path = std::env::temp_dir().join("gtopen_bad_fit.json");
    std::fs::write(&path, v.to_string()).unwrap();
    let r = RealizationFit::load(path.to_str().unwrap());
    std::fs::remove_file(&path).ok();
    assert!(r.is_err(), "1-element mult_clip must be a load error");
}

// ===== config economics validation =====

/// Impossible economics must be rejected at build with an error naming the
/// offending field: negative rake minted chips at every raked terminal,
/// rake >= 100% made effective pots negative, NaN slipped through every
/// range comparison, over-stack opens invested chips that don't exist.
#[test]
fn impossible_economics_are_rejected() {
    let eq = table();
    let reject = |what: &str, needle: &str, mutate: &dyn Fn(&mut PreflopConfig)| {
        let mut cfg = hu_limp_config();
        mutate(&mut cfg);
        match PreflopSolver::new(cfg, eq.clone()) {
            Ok(_) => panic!("{what}: config unexpectedly accepted"),
            Err(e) => assert!(e.contains(needle), "{what}: error should name {needle}: {e}"),
        }
    };
    reject("negative rake", "rake_pct", &|c| c.rake_pct = -5.0);
    reject("rake at 100%", "rake_pct", &|c| c.rake_pct = 100.0);
    reject("NaN rake", "rake_pct", &|c| c.rake_pct = f64::NAN);
    reject("negative rake cap", "rake_cap", &|c| c.rake_cap = -1.0);
    reject("NaN stack", "stack", &|c| c.stack = f64::NAN);
    reject("infinite stack", "stack", &|c| c.stack = f64::INFINITY);
    reject("negative ante", "ante", &|c| c.ante = -0.25);
    reject("NaN post", "posts", &|c| c.posts = vec![0.5, f64::NAN]);
    reject("negative post", "posts", &|c| c.posts = vec![-0.5, 1.0]);
    reject("negative open", "open_raises", &|c| c.open_raises = vec![-2.5]);
    reject("zero open", "open_raises", &|c| c.open_raises = vec![0.0]);
    reject("infinite open", "open_raises", &|c| c.open_raises = vec![2.5, f64::INFINITY]);
    reject("over-stack open", "open_raises", &|c| c.open_raises = vec![250.0]);
    reject("zero raise mult", "raise_mults", &|c| c.raise_mults = vec![0.0]);
    reject("NaN raise mult", "raise_mults", &|c| c.raise_mults = vec![f64::NAN]);
    reject("zero all-in threshold", "allin_threshold", &|c| c.allin_threshold = 0.0);
    reject("threshold above 1", "allin_threshold", &|c| c.allin_threshold = 1.5);
    reject("NaN threshold", "allin_threshold", &|c| c.allin_threshold = f64::NAN);
}

/// Boundary values are all VALID (0% rake, uncapped, jam threshold 1.0) and
/// so are sub-minimum opens: "any raise sizes" is an advertised study
/// feature — the builder recomputes last_raise from the actual open and
/// clamps every RE-raise up to a legal increment, so a 1.5bb open is an
/// unusual config, not a broken one.
#[test]
fn boundary_and_study_configs_still_build() {
    let eq = table();
    let mut cfg = hu_limp_config();
    cfg.rake_pct = 0.0;
    cfg.rake_cap = 0.0;
    cfg.allin_threshold = 1.0;
    PreflopSolver::new(cfg, eq.clone()).expect("boundary values are valid");

    let mut cfg = hu_limp_config();
    cfg.open_raises = vec![1.5];
    let s = PreflopSolver::new(cfg, eq).expect("sub-min opens are a valid study config");
    assert!(
        s.nodes[0]
            .actions
            .iter()
            .any(|a| a.kind == "raise" && (a.to - 1.5).abs() < 1e-9),
        "the 1.5bb open must be offered at the root"
    );
}

/// Opens at/below the biggest blind are silently dropped by the action
/// builder; with no limp and no all-in that would build a fold-only root.
#[test]
fn all_dropped_opens_are_rejected() {
    let eq = table();
    let mut cfg = hu_limp_config();
    cfg.limp = false;
    cfg.open_raises = vec![1.0]; // == BB: never offered
    match PreflopSolver::new(cfg, eq) {
        Ok(_) => panic!("fold-only config unexpectedly accepted"),
        Err(e) => assert!(e.contains("opening"), "unexpected error: {e}"),
    }
}

// ===== save/load session-state validation =====

/// Rewrite one field of a .gtop header in place (the header is a JSON line
/// between the magic and the arenas; the arenas are kept verbatim).
fn doctor_pf_header(path: &str, field: &str, value: serde_json::Value) {
    let bytes = std::fs::read(path).unwrap();
    const MAGIC_LEN: usize = 12; // b"GTOPREFLOP1\n"
    let nl = MAGIC_LEN + bytes[MAGIC_LEN..].iter().position(|&b| b == b'\n').unwrap();
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[MAGIC_LEN..nl]).unwrap();
    header[field] = value;
    let mut out = bytes[..MAGIC_LEN].to_vec();
    out.extend_from_slice(serde_json::to_string(&header).unwrap().as_bytes());
    out.extend_from_slice(&bytes[nl..]); // '\n' + arenas, untouched
    std::fs::write(path, out).unwrap();
}

/// A malformed point_locks entry used to be installed unvalidated; the
/// panic then fired in the traversal's copy_from_slice at the first solve
/// step or query — under the server's session mutex, wedging the lab.
/// Load must return a clear Err instead.
#[test]
fn load_rejects_malformed_point_locks() {
    let eq = table();
    let s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    let path = std::env::temp_dir().join("gtopen_pf_badlocks.gtop");
    let path = path.to_str().unwrap().to_string();

    let reject = |locks: serde_json::Value, needle: &str, what: &str| {
        s.save_game(&path).unwrap();
        doctor_pf_header(&path, "point_locks", locks);
        match PreflopSolver::load_game(&path, eq.clone()) {
            Ok(_) => panic!("{what}: load unexpectedly succeeded"),
            Err(e) => assert!(e.contains(needle), "{what}: unexpected error: {e}"),
        }
    };
    // the review's reproducer: a 1-entry sigma at the root
    reject(serde_json::json!([[0, [1.0]]]), "expected", "short sigma");
    reject(serde_json::json!([[999_999, [1.0]]]), "out of range", "oob node index");
    let na = s.nodes[0].actions.len();
    let mut sigma = vec![0.5f32; na * NUM_CLASSES];
    sigma[0] = -1.0;
    reject(serde_json::json!([[0, sigma]]), "finite", "negative frequency");
    // a lock on a terminal: 0 actions x 169 classes = 0 entries, so only
    // the node-kind check stands between an empty sigma and the consumers
    let term = s
        .nodes
        .iter()
        .position(|n| n.actions.is_empty())
        .expect("tree has terminals");
    reject(
        serde_json::json!([[term, Vec::<f32>::new()]]),
        "not an action node",
        "terminal lock",
    );

    // control: a WELL-FORMED lock still loads, and the consumer that used
    // to panic serves it verbatim
    s.save_game(&path).unwrap();
    let uniform = vec![1.0f32 / na as f32; na * NUM_CLASSES];
    doctor_pf_header(&path, "point_locks", serde_json::json!([[0, uniform.clone()]]));
    let l = PreflopSolver::load_game(&path, eq).unwrap();
    assert_eq!(l.average_strategy(0), uniform, "the lock must be served verbatim");
    std::fs::remove_file(&path).ok();
}

/// Hero-mode state rides the same header and gets the same scrutiny: the
/// hero seat indexes per-seat arrays, and pre-hero frozen flags are
/// installed as-is.
#[test]
fn load_rejects_bad_hero_state() {
    let eq = table();
    let s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    let path = std::env::temp_dir().join("gtopen_pf_badhero.gtop");
    let path = path.to_str().unwrap().to_string();

    s.save_game(&path).unwrap();
    doctor_pf_header(&path, "hero", serde_json::json!(7));
    match PreflopSolver::load_game(&path, eq.clone()) {
        Ok(_) => panic!("hero seat 7 of 2: load unexpectedly succeeded"),
        Err(e) => assert!(e.contains("hero"), "unexpected error: {e}"),
    }

    s.save_game(&path).unwrap();
    doctor_pf_header(&path, "pre_hero_frozen", serde_json::json!([true]));
    match PreflopSolver::load_game(&path, eq) {
        Ok(_) => panic!("1 pre-hero flag for 2 seats: load unexpectedly succeeded"),
        Err(e) => assert!(e.contains("pre-hero"), "unexpected error: {e}"),
    }
    std::fs::remove_file(&path).ok();
}

/// A failed save must leave the previous save intact: save_game used to
/// File::create (truncate) the destination first, so disk-full or a kill
/// mid-write destroyed the only copy. It now stages into `{path}.tmp` and
/// renames; blocking the staging path simulates the write failure.
#[test]
fn failed_save_leaves_previous_game_intact() {
    let eq = table();
    let mut s = PreflopSolver::new(hu_limp_config(), eq.clone()).unwrap();
    for _ in 0..10 {
        s.iterate();
    }
    let dir = std::env::temp_dir().join("gtopen_pf_atomic");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("game.gtop");
    let path = path.to_str().unwrap().to_string();
    let tmp = format!("{path}.tmp");

    s.save_game(&path).unwrap();
    assert!(
        !std::path::Path::new(&tmp).exists(),
        "staging file must not survive a successful save"
    );
    let before = std::fs::read(&path).unwrap();

    // a directory at the staging path makes its File::create fail — the
    // same failure point as a full disk
    std::fs::create_dir(&tmp).unwrap();
    for _ in 0..5 {
        s.iterate();
    }
    assert!(s.save_game(&path).is_err(), "blocked staging path must fail the save");
    assert_eq!(
        before,
        std::fs::read(&path).unwrap(),
        "failed save must leave the previous save byte-identical"
    );
    std::fs::remove_dir(&tmp).unwrap();

    // the next save recovers, replaces the old file and loads
    s.save_game(&path).unwrap();
    let l = PreflopSolver::load_game(&path, eq).unwrap();
    assert_eq!(l.iteration, s.iteration);
    std::fs::remove_dir_all(&dir).ok();
}

/// call_only_seats: the masked seat's nodes never offer a raise or jam, while
/// other seats' menus are untouched and the masked seat can still limp/call.
#[test]
fn call_only_seat_never_raises() {
    let eq = table();
    let cfg = PreflopConfig {
        positions: vec!["BTN".into(), "SB".into(), "BB".into()],
        stack: 100.0,
        posts: vec![0.0, 0.5, 1.0],
        ante: 0.0,
        limp: true,
        open_raises: vec![4.0],
        raise_mults: vec![3.0],
        max_raises: 4,
        add_allin: false,
        allin_threshold: 0.85,
        rake_pct: 0.0,
        rake_cap: 0.0,
        no_flop_no_drop: true,
        realization: "raw".into(),
        call_only_seats: vec![0],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
    };
    let s = PreflopSolver::new(cfg.clone(), eq.clone()).unwrap();
    let (mut masked_nodes, mut others_raise) = (0, 0);
    for nd in &s.nodes {
        if nd.kind != 0 {
            continue;
        }
        let has_raise = nd.actions.iter().any(|a| a.kind == "raise" || a.kind == "jam");
        if nd.actor == 0 {
            masked_nodes += 1;
            assert!(!has_raise, "call-only seat offered a raise: {:?}",
                nd.actions.iter().map(|a| a.label.clone()).collect::<Vec<_>>());
            assert!(nd.actions.iter().any(|a| a.kind == "call" || a.kind == "check"),
                "call-only seat should still be able to limp/call/check");
        } else if has_raise {
            others_raise += 1;
        }
    }
    assert!(masked_nodes > 0, "masked seat never acted");
    assert!(others_raise > 0, "unmasked seats lost their raises");

    // out-of-range index is refused
    let mut bad = cfg;
    bad.call_only_seats = vec![7];
    let err = match PreflopSolver::new(bad, eq) { Err(e) => e, Ok(_) => panic!("bad index accepted") };
    assert!(err.contains("call_only_seats"));
}

/// Per-seat size menus: the overridden seat sees its own sizes while every
/// other seat keeps the global menu.
#[test]
fn per_seat_size_menus() {
    let eq = table();
    let mut cfg = PreflopConfig {
        positions: vec!["BTN".into(), "SB".into(), "BB".into()],
        stack: 100.0,
        posts: vec![0.0, 0.5, 1.0],
        ante: 0.0,
        limp: true,
        open_raises: vec![4.0],
        raise_mults: vec![3.0],
        max_raises: 3,
        add_allin: false,
        allin_threshold: 0.85,
        rake_pct: 0.0,
        rake_cap: 0.0,
        no_flop_no_drop: true,
        realization: "raw".into(),
        call_only_seats: vec![],
        open_raises_by_seat: None,
        raise_mults_by_seat: None,
    };
    cfg.open_raises_by_seat = Some(vec![vec![2.5, 3.0, 5.0], vec![], vec![]]);
    cfg.raise_mults_by_seat = Some(vec![vec![2.5, 4.0], vec![], vec![]]);
    let s = PreflopSolver::new(cfg.clone(), eq.clone()).unwrap();
    let mut seen_btn_opens: Vec<f64> = vec![];
    let mut seen_other_opens: Vec<f64> = vec![];
    for nd in &s.nodes {
        if nd.kind != 0 {
            continue;
        }
        for a in &nd.actions {
            if a.kind == "raise" {
                let owed_zero_open = a.label.starts_with("Raise");
                if owed_zero_open {
                    if nd.actor == 0 {
                        seen_btn_opens.push(a.to);
                    } else {
                        seen_other_opens.push(a.to);
                    }
                }
            }
        }
    }
    seen_btn_opens.sort_by(f64::total_cmp);
    seen_btn_opens.dedup();
    seen_other_opens.sort_by(f64::total_cmp);
    seen_other_opens.dedup();
    assert_eq!(seen_btn_opens, vec![2.5, 3.0, 5.0], "override seat menu");
    assert_eq!(seen_other_opens, vec![4.0], "global menu untouched");

    // wrong length rejected
    let mut bad = cfg;
    bad.open_raises_by_seat = Some(vec![vec![2.5]]);
    let err = match PreflopSolver::new(bad, eq) { Err(e) => e, Ok(_) => panic!("bad len accepted") };
    assert!(err.contains("open_raises_by_seat"));
}
