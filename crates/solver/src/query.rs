//! Node inspection for the strategy browser: walk a path of actions/cards
//! from the root, then report per-hand strategies, EVs and equities.

use crate::cards::*;
use crate::cfr::Solver;
use crate::game::Dealt;
use crate::tree::{
    Action, KIND_ACTION, KIND_CHANCE, KIND_TERM_FOLD, KIND_TERM_SHOWDOWN, SENTINEL,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum PathStep {
    #[serde(rename = "action")]
    Action { index: usize },
    #[serde(rename = "card")]
    Card { card: String },
}

/// How to build a node lock. See `Solver::lock_node`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum LockMode {
    /// Freeze the solved (GTO) strategy as-is.
    Freeze,
    /// Target overall (reach-weighted) action frequencies for the whole range.
    Range { freqs: Vec<f32> },
    /// Set exact frequencies for specific combos; other hands keep current.
    Hands { edits: Vec<HandEdit> },
}

#[derive(Debug, Clone, Deserialize)]
pub struct HandEdit {
    pub combo: String,
    pub freqs: Vec<f32>,
}

/// Validate and normalize a per-action frequency vector to sum 1.
fn normalize_freqs(freqs: &[f32], na: usize) -> Result<Vec<f32>, String> {
    if freqs.len() != na {
        return Err(format!("expected {na} action frequencies, got {}", freqs.len()));
    }
    if freqs.iter().any(|&x| x < 0.0 || !x.is_finite()) {
        return Err("frequencies must be finite and >= 0".to_string());
    }
    let sum: f32 = freqs.iter().sum();
    if sum <= 1e-9 {
        return Err("at least one frequency must be positive".to_string());
    }
    Ok(freqs.iter().map(|&x| x / sum).collect())
}

/// Rake per-action multipliers (iterative proportional fitting) so that, after
/// scaling each hand's strategy and renormalizing, the reach-weighted aggregate
/// frequency of each action matches `target`. Actions with a zero target are
/// driven out; actions the solved strategy never uses can't be forced up
/// (no probability mass to scale) — those stay near zero (use per-hand edits).
fn rake_to_target(sigma: &mut [f32], na: usize, nh: usize, reach: &[f32], target: &[f32]) {
    let mut m = vec![0f64; na];
    for a in 0..na {
        if target[a] > 1e-9 {
            m[a] = 1.0;
        }
    }
    // Hands with no mass on any targeted action can't be raked: assign them
    // the target distribution itself, so zero-target actions stay at zero
    // (uniform-over-everything would leak forbidden actions back in).
    for i in 0..nh {
        let mut mass = 0f64;
        for a in 0..na {
            if target[a] > 1e-9 {
                mass += sigma[a * nh + i] as f64;
            }
        }
        if mass <= 1e-12 {
            for a in 0..na {
                sigma[a * nh + i] = target[a];
            }
        }
    }
    for _ in 0..400 {
        let mut agg = vec![0f64; na];
        let mut total = 0f64;
        for i in 0..nh {
            let w = reach[i] as f64;
            if w <= 0.0 {
                continue;
            }
            let mut denom = 0f64;
            for a in 0..na {
                denom += m[a] * sigma[a * nh + i] as f64;
            }
            if denom <= 1e-12 {
                continue;
            }
            total += w;
            for a in 0..na {
                agg[a] += w * m[a] * sigma[a * nh + i] as f64 / denom;
            }
        }
        if total <= 1e-12 {
            break;
        }
        let mut max_err = 0f64;
        for a in 0..na {
            if target[a] <= 1e-9 {
                continue;
            }
            let f = agg[a] / total;
            max_err = max_err.max((f - target[a] as f64).abs());
            if f > 1e-9 {
                m[a] *= target[a] as f64 / f;
            }
        }
        // keep multipliers bounded
        let msum: f64 = m.iter().sum();
        if msum > 1e-12 {
            let scale = na as f64 / msum;
            for a in 0..na {
                m[a] *= scale;
            }
        }
        if max_err < 1e-4 {
            break;
        }
    }
    for i in 0..nh {
        let mut s = 0f64;
        for a in 0..na {
            let v = m[a] * sigma[a * nh + i] as f64;
            sigma[a * nh + i] = v as f32;
            s += v;
        }
        if s > 1e-12 {
            for a in 0..na {
                sigma[a * nh + i] = (sigma[a * nh + i] as f64 / s) as f32;
            }
        } else {
            // unreachable after the pre-pass above, but keep the fallback
            // consistent: distribute by target, never onto zeroed actions
            for a in 0..na {
                sigma[a * nh + i] = target[a];
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ActionView {
    pub label: String,
    pub kind: String,
    pub amount: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct HandView {
    pub combo: String,
    pub c1: u8,
    pub c2: u8,
    /// Range weight at the root.
    pub weight: f32,
    /// Reach probability product (range weight x strategy path).
    pub reach: f32,
    /// Card-removal-adjusted opponent reach mass (the per-hand EV normalizer).
    /// Range-average EVs must weight by reach x valid, or the invariant
    /// EV_OOP + EV_IP = pot breaks.
    pub valid: f32,
    /// Equity vs opponent reach range (NaN -> null in JSON).
    pub eq: Option<f32>,
    /// EV in pot-share convention (chips owned of final pot, incl. future bets).
    pub ev: Option<f32>,
    /// Acting player only: average strategy per action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<Vec<f32>>,
    /// Acting player only: EV per action.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evs: Option<Vec<Option<f32>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PlayerView {
    pub hands: Vec<HandView>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeView {
    pub node_type: String,
    pub street: u8,
    pub board: Vec<String>,
    pub pot: f64,
    pub put: [f64; 2],
    pub player: Option<u8>,
    pub actions: Vec<ActionView>,
    /// Chance nodes: cards that can come next.
    pub available_cards: Option<Vec<String>>,
    pub players: [PlayerView; 2],
    pub locked: bool,
    /// One entry per node along the path, plus the current node (chosen=None).
    pub history: Vec<HistoryStep>,
}

/// Per-hand best-response data for the exploit ("max exploit") view.
#[derive(Debug, Clone, Serialize)]
pub struct ExploitHandView {
    pub combo: String,
    pub c1: u8,
    pub c2: u8,
    pub weight: f32,
    pub reach: f32,
    /// Card-removal-adjusted opponent reach mass (the per-hand EV normalizer).
    /// Aggregates of br_ev/cur_ev/gain/evs must weight by reach x valid, same
    /// convention as `HandView::valid`.
    pub valid: f32,
    /// EV of best-responding from here on (pot-share convention).
    pub br_ev: Option<f32>,
    /// EV of the current average strategy (same convention).
    pub cur_ev: Option<f32>,
    /// br_ev - cur_ev: what this hand gains by max-exploiting.
    pub gain: Option<f32>,
    /// Exploiter-to-act nodes only: the best response (one-hot, ties split;
    /// the lock itself at locked nodes, where deviating is not allowed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub br_strategy: Option<Vec<f32>>,
    /// Exploiter-to-act nodes only: EV per action under BR continuation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evs: Option<Vec<Option<f32>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExploitView {
    pub node_type: String,
    pub exploiter: u8,
    /// Actor at this node (action nodes only).
    pub player: Option<u8>,
    pub actions: Vec<ActionView>,
    pub locked: bool,
    pub hands: Vec<ExploitHandView>,
    /// Reach x valid weighted average EV under best response / current
    /// strategy, and the difference (chips) — the headline "how exploitable
    /// is this node". Same weighting as every other range-average EV, so the
    /// levels line up with the strategy-mode EVs at the same node.
    pub avg_br_ev: Option<f32>,
    pub avg_cur_ev: Option<f32>,
    pub avg_gain: Option<f32>,
}

/// One stop along the walked line, for the action-history bar.
#[derive(Debug, Clone, Serialize)]
pub struct HistoryStep {
    /// "action" | "card" | "terminal"
    pub kind: String,
    pub player: Option<u8>,
    /// Actor's remaining stack when this node is reached.
    pub stack: f64,
    pub pot: f64,
    pub street: u8,
    pub actions: Vec<ActionView>,
    /// Index of the action taken from here (None for the current node).
    pub chosen: Option<usize>,
    /// Card dealt at this step (card steps only; None = not yet picked).
    pub card: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunoutRow {
    pub card: String,
    /// Action frequencies at the action node following this card (empty for
    /// all-in runouts with no further action).
    pub freqs: Vec<f32>,
    /// Average equity for [OOP, IP] on this runout.
    pub eq: [Option<f32>; 2],
    /// Reach-weighted average EV for [OOP, IP] (pot-share convention), so
    /// the report can rank runouts by EV and derive EQR = EV / (pot x EQ).
    pub ev: [Option<f32>; 2],
}

#[derive(Debug, Clone, Serialize)]
pub struct RunoutsReport {
    pub actions: Vec<ActionView>,
    pub player: Option<u8>,
    /// Pot at the runout nodes (same for every card), for EQR.
    pub pot: f64,
    pub rows: Vec<RunoutRow>,
}

pub struct Walk {
    pub node_idx: u32,
    pub reach: [Vec<f32>; 2],
    pub dealt: Dealt,
}

impl Solver {
    /// Walk a path from the root, multiplying strategy reach along the way.
    pub fn walk_path(&self, path: &[PathStep]) -> Result<Walk, String> {
        let spot = &*self.spot;
        let mut node_idx = 0u32;
        let mut reach = [spot.weights[0].clone(), spot.weights[1].clone()];
        let mut dealt = Dealt::default();
        for step in path {
            let node = &spot.tree.nodes[node_idx as usize];
            match (node.kind, step) {
                (KIND_ACTION, PathStep::Action { index }) => {
                    let na = node.num_children as usize;
                    if *index >= na {
                        return Err(format!("action index {index} out of range"));
                    }
                    let p = node.player as usize;
                    let nh = spot.hands[p].len();
                    let sigma = self.average_strategy(node_idx, node);
                    for i in 0..nh {
                        reach[p][i] *= sigma[index * nh + i];
                    }
                    node_idx = spot.tree.children[node.children_start as usize + index];
                }
                (KIND_CHANCE, PathStep::Card { card }) => {
                    let c = card_from_str(card)?;
                    let child = spot.tree.children[node.children_start as usize + c as usize];
                    if child == SENTINEL || dealt.contains(c) {
                        return Err(format!("card {card} not available here"));
                    }
                    let cm = card_mask(c);
                    for p in 0..2 {
                        for (i, h) in spot.hands[p].iter().enumerate() {
                            if h.mask & cm != 0 {
                                reach[p][i] = 0.0;
                            }
                        }
                    }
                    dealt = dealt.push(c);
                    node_idx = child;
                }
                _ => return Err("path step does not match node type".to_string()),
            }
        }
        Ok(Walk {
            node_idx,
            reach,
            dealt,
        })
    }

    /// Labeled views of an action node's actions (empty for other node kinds).
    fn action_views(&self, node_idx: u32) -> Vec<ActionView> {
        let spot = &*self.spot;
        let node = &spot.tree.nodes[node_idx as usize];
        if node.kind != KIND_ACTION {
            return Vec::new();
        }
        let pot = node.put[0] + node.put[1];
        let me = node.player as usize;
        (0..node.num_children as usize)
            .map(|a| {
                let action = spot.tree.actions[node.actions_start as usize + a];
                // An aggressive action is all-in iff the child node leaves the
                // actor with no stack behind.
                let all_in = match action {
                    Action::Bet(_) | Action::Raise(_) => {
                        let child_idx = spot.tree.children[node.children_start as usize + a];
                        let child = &spot.tree.nodes[child_idx as usize];
                        let stack_after = spot.tree.config.effective_stack
                            - (child.put[me] - spot.tree.config.starting_pot / 2.0);
                        stack_after.abs() < 1e-6
                    }
                    _ => false,
                };
                ActionView {
                    label: action.label(pot, all_in),
                    kind: match action {
                        Action::Fold => "fold",
                        Action::Check => "check",
                        Action::Call(_) => "call",
                        Action::Bet(_) => "bet",
                        Action::Raise(_) => "raise",
                    }
                    .to_string(),
                    amount: match action {
                        Action::Call(x) | Action::Bet(x) | Action::Raise(x) => x,
                        _ => 0.0,
                    },
                }
            })
            .collect()
    }

    /// One HistoryStep per node along the path, plus the current node.
    fn build_history(&self, path: &[PathStep]) -> Result<Vec<HistoryStep>, String> {
        let spot = &*self.spot;
        let config = &spot.tree.config;
        let stack_of = |node: &crate::tree::Node, p: usize| {
            config.effective_stack - (node.put[p] - config.starting_pot / 2.0)
        };
        let mut history = Vec::with_capacity(path.len() + 1);
        let mut idx = 0u32;
        for step in path {
            let node = &spot.tree.nodes[idx as usize];
            match (node.kind, step) {
                (KIND_ACTION, PathStep::Action { index }) => {
                    history.push(HistoryStep {
                        kind: "action".to_string(),
                        player: Some(node.player),
                        stack: stack_of(node, node.player as usize),
                        pot: node.put[0] + node.put[1],
                        street: node.street,
                        actions: self.action_views(idx),
                        chosen: Some(*index),
                        card: None,
                    });
                    idx = spot.tree.children[node.children_start as usize + *index];
                }
                (KIND_CHANCE, PathStep::Card { card }) => {
                    let c = card_from_str(card)?;
                    history.push(HistoryStep {
                        kind: "card".to_string(),
                        player: None,
                        stack: 0.0,
                        pot: node.put[0] + node.put[1],
                        street: node.street,
                        actions: Vec::new(),
                        chosen: None,
                        card: Some(card_to_string(c)),
                    });
                    idx = spot.tree.children[node.children_start as usize + c as usize];
                }
                _ => return Err("path step does not match node type".to_string()),
            }
        }
        let node = &spot.tree.nodes[idx as usize];
        history.push(HistoryStep {
            kind: match node.kind {
                KIND_ACTION => "action",
                KIND_CHANCE => "card",
                _ => "terminal",
            }
            .to_string(),
            player: if node.kind == KIND_ACTION {
                Some(node.player)
            } else {
                None
            },
            stack: if node.kind == KIND_ACTION {
                stack_of(node, node.player as usize)
            } else {
                0.0
            },
            pot: node.put[0] + node.put[1],
            street: node.street,
            actions: self.action_views(idx),
            chosen: None,
            card: None,
        });
        Ok(history)
    }

    /// Compatible opponent reach sum for each of p's hands (normalizer for EVs).
    fn valid_opp_sum(&self, p: usize, reach_o: &[f32]) -> Vec<f64> {
        let spot = &*self.spot;
        let mut t = 0f64;
        let mut s = [0f64; 52];
        for (j, h) in spot.hands[1 - p].iter().enumerate() {
            let r = reach_o[j] as f64;
            t += r;
            s[h.c1 as usize] += r;
            s[h.c2 as usize] += r;
        }
        spot.hands[p]
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let same = spot.same_combo[p][i];
                let same_r = if same != SENTINEL {
                    reach_o[same as usize] as f64
                } else {
                    0.0
                };
                t - s[h.c1 as usize] - s[h.c2 as usize] + same_r
            })
            .collect()
    }

    pub fn node_view(&self, path: &[PathStep]) -> Result<NodeView, String> {
        let spot = &*self.spot;
        let walk = self.walk_path(path)?;
        let node = &spot.tree.nodes[walk.node_idx as usize];
        let mut board: Vec<String> = spot.board.iter().map(|&c| card_to_string(c)).collect();
        for i in 0..walk.dealt.len as usize {
            board.push(card_to_string(walk.dealt.cards[i]));
        }
        let pot = node.put[0] + node.put[1];

        let node_type = match node.kind {
            KIND_ACTION => "action",
            KIND_CHANCE => "chance",
            KIND_TERM_FOLD => "terminal_fold",
            KIND_TERM_SHOWDOWN => "terminal_showdown",
            _ => "?",
        }
        .to_string();

        let actions = self.action_views(walk.node_idx);

        let available_cards = if node.kind == KIND_CHANCE {
            let mut v = Vec::new();
            for c in 0..52u8 {
                let child = spot.tree.children[node.children_start as usize + c as usize];
                if child != SENTINEL && !walk.dealt.contains(c) {
                    v.push(card_to_string(c));
                }
            }
            Some(v)
        } else {
            None
        };

        // Per-player hand data.
        let mut players: Vec<PlayerView> = Vec::with_capacity(2);
        for p in 0..2usize {
            let nh = spot.hands[p].len();
            let reach_o = &walk.reach[1 - p];
            let valid = self.valid_opp_sum(p, reach_o);
            let eq = self.equity(p, reach_o, walk.dealt);

            // EV via average-strategy traversal from this node.
            let is_actor = node.kind == KIND_ACTION && node.player as usize == p;
            let (ev_total, evs_per_action, sigma) = if is_actor {
                let na = node.num_children as usize;
                let sigma = self.average_strategy(walk.node_idx, node);
                let mut evs: Vec<Vec<Option<f32>>> = vec![vec![None; na]; nh];
                let mut total = vec![0f64; nh];
                for a in 0..na {
                    let child = spot.tree.children[node.children_start as usize + a];
                    let cfv = self.traverse_avg(child, p, reach_o, walk.dealt);
                    for i in 0..nh {
                        if valid[i] > 1e-9 {
                            let ev = cfv[i] as f64 / valid[i] + node.put[p];
                            evs[i][a] = Some(ev as f32);
                            total[i] += sigma[a * nh + i] as f64 * ev;
                        }
                    }
                }
                (
                    total
                        .iter()
                        .zip(valid.iter())
                        .map(|(&t, &v)| if v > 1e-9 { Some(t as f32) } else { None })
                        .collect::<Vec<_>>(),
                    Some(evs),
                    Some(sigma),
                )
            } else if node.kind == KIND_TERM_FOLD || node.kind == KIND_TERM_SHOWDOWN {
                let cfv = self.traverse_avg(walk.node_idx, p, reach_o, walk.dealt);
                (
                    cfv.iter()
                        .zip(valid.iter())
                        .map(|(&c, &v)| {
                            if v > 1e-9 {
                                Some((c as f64 / v + node.put[p]) as f32)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    None,
                    None,
                )
            } else {
                let cfv = self.traverse_avg(walk.node_idx, p, reach_o, walk.dealt);
                (
                    cfv.iter()
                        .zip(valid.iter())
                        .map(|(&c, &v)| {
                            if v > 1e-9 {
                                Some((c as f64 / v + node.put[p]) as f32)
                            } else {
                                None
                            }
                        })
                        .collect(),
                    None,
                    None,
                )
            };

            let hands = (0..nh)
                .map(|i| {
                    let h = &spot.hands[p][i];
                    HandView {
                        combo: combo_to_string(h.c1, h.c2),
                        c1: h.c1,
                        c2: h.c2,
                        weight: h.weight,
                        reach: walk.reach[p][i],
                        valid: valid[i] as f32,
                        eq: if eq[i].is_nan() { None } else { Some(eq[i]) },
                        ev: ev_total[i],
                        strategy: sigma.as_ref().map(|s| {
                            (0..node.num_children as usize)
                                .map(|a| s[a * nh + i])
                                .collect()
                        }),
                        evs: evs_per_action.as_ref().map(|e| e[i].clone()),
                    }
                })
                .collect();
            players.push(PlayerView { hands });
        }

        let players: [PlayerView; 2] = players.try_into().map_err(|_| "internal".to_string())?;

        Ok(NodeView {
            node_type,
            street: node.street,
            board,
            pot,
            put: node.put,
            player: if node.kind == KIND_ACTION {
                Some(node.player)
            } else {
                None
            },
            actions,
            available_cards,
            players,
            locked: {
                // the lock lives on the orbit representative's node
                let canon = self.canonical_path(path);
                let idx = self.walk_path(&canon).map(|w| w.node_idx).unwrap_or(walk.node_idx);
                self.locks.contains_key(&idx)
            },
            history: self.build_history(path)?,
        })
    }

    /// Map every card step to its suit-isomorphism orbit representative, so
    /// operations land on the branch CFR actually traverses.
    pub fn canonical_path(&self, path: &[PathStep]) -> Vec<PathStep> {
        self.canonical_path_with_perm(path).0
    }

    /// Canonicalize card steps to orbit representatives, tracking the COMPOSED
    /// suit permutation that maps the browsed branch onto the canonical one.
    /// Later steps must canonicalize the already-permuted card (not the
    /// original), mirroring exactly how `symmetrize` nests `copy_branch` per
    /// street; the greedy strictly-less scan below matches its tie-breaking.
    /// The returned suit map lets callers translate hand identities between
    /// the browsed branch and the canonical one (identity when nothing moved).
    pub fn canonical_path_with_perm(&self, path: &[PathStep]) -> (Vec<PathStep>, [u8; 4]) {
        let mut composed: [u8; 4] = [0, 1, 2, 3];
        if !self.use_isomorphism || self.spot.suit_perms.len() < 2 {
            return (path.to_vec(), composed);
        }
        let mut dealt = Dealt::default();
        let out = path
            .iter()
            .map(|s| match s {
                PathStep::Card { card } => match card_from_str(card) {
                    Ok(c) => {
                        // the card as CFR sees it on the canonical branch so far
                        let cm = crate::cards::permute_card(c, &composed);
                        let valid = self.spot.perms_fixing(&dealt);
                        let mut rep = cm;
                        let mut k_rep = None;
                        for &k in &valid {
                            let pc = crate::cards::permute_card(cm, &self.spot.suit_perms[k]);
                            if pc < rep {
                                rep = pc;
                                k_rep = Some(k);
                            }
                        }
                        if let Some(k) = k_rep {
                            let pm = &self.spot.suit_perms[k];
                            for s in composed.iter_mut() {
                                *s = pm[*s as usize];
                            }
                        }
                        dealt = dealt.push(rep);
                        PathStep::Card {
                            card: card_to_string(rep),
                        }
                    }
                    Err(_) => s.clone(),
                },
                a => a.clone(),
            })
            .collect();
        (out, composed)
    }

    /// Lock a node's strategy. Each mode recomputes the locked strategy from a
    /// stable base, so re-applying with new values never compounds.
    pub fn lock_node(
        &mut self,
        path: &[PathStep],
        mode: LockMode,
        label: String,
    ) -> Result<(), String> {
        // lock the orbit representative: that's the branch the solver visits,
        // and the lock then applies to all isomorphic runouts at once
        let (canon, perm) = self.canonical_path_with_perm(path);
        let walk = self.walk_path(&canon)?;
        let node = &self.spot.tree.nodes[walk.node_idx as usize];
        if node.kind != KIND_ACTION {
            return Err("only action nodes can be locked".to_string());
        }
        let na = node.num_children as usize;
        let p = node.player as usize;
        let nh = self.spot.hands[p].len();

        let sigma = match mode {
            // Lock the solved (GTO) strategy exactly.
            LockMode::Freeze => {
                let mut s = vec![0f32; na * nh];
                self.solved_strategy_into(node, &mut s);
                s
            }
            // Rake the SOLVED strategy so the reach-weighted aggregate action
            // frequencies match the targets (must sum to ~1). Always from the
            // solved base, so re-locking is idempotent.
            LockMode::Range { freqs } => {
                let target = normalize_freqs(&freqs, na)?;
                let mut s = vec![0f32; na * nh];
                self.solved_strategy_into(node, &mut s);
                rake_to_target(&mut s, na, nh, &walk.reach[p], &target);
                s
            }
            // Override specific hands' frequencies on top of the CURRENT
            // strategy (the lock if one exists, else solved). Absolute
            // assignment, so successive per-hand edits accumulate cleanly.
            LockMode::Hands { edits } => {
                let mut s = self.average_strategy(walk.node_idx, node);
                for edit in &edits {
                    let target = normalize_freqs(&edit.freqs, na)?;
                    let combo = parse_cards(&edit.combo)?;
                    if combo.len() != 2 {
                        return Err(format!("expected a 2-card combo, got {:?}", edit.combo));
                    }
                    // the combo named by the user lives on the browsed branch;
                    // on the canonical branch it is the suit-permuted combo
                    let c1 = crate::cards::permute_card(combo[0], &perm);
                    let c2 = crate::cards::permute_card(combo[1], &perm);
                    let idx = self.spot.hands[p]
                        .iter()
                        .position(|h| {
                            (h.c1 == c1 && h.c2 == c2) || (h.c1 == c2 && h.c2 == c1)
                        })
                        .ok_or_else(|| {
                            format!("{} is not in {}'s range here", edit.combo,
                                if p == 0 { "OOP" } else { "IP" })
                        })?;
                    for a in 0..na {
                        s[a * nh + idx] = target[a];
                    }
                }
                s
            }
        };

        self.locks.insert(walk.node_idx, sigma);
        self.lock_labels.insert(walk.node_idx, label);
        // locks live outside the arenas: isomorphic siblings only see them
        // once ensure_symmetric re-materializes the branch copies
        self.mark_sym_dirty();
        Ok(())
    }

    pub fn unlock_node(&mut self, path: &[PathStep]) -> Result<bool, String> {
        let path = &self.canonical_path(path);
        let walk = self.walk_path(path)?;
        self.lock_labels.remove(&walk.node_idx);
        let removed = self.locks.remove(&walk.node_idx).is_some();
        if removed {
            self.mark_sym_dirty();
        }
        Ok(removed)
    }

    pub fn clear_locks(&mut self) {
        if !self.locks.is_empty() {
            self.mark_sym_dirty();
        }
        self.locks.clear();
        self.lock_labels.clear();
    }

    pub fn list_locks(&self) -> Vec<String> {
        let mut v: Vec<String> = self.lock_labels.values().cloned().collect();
        v.sort();
        v
    }

    /// Best-response ("max exploit") view at a node: what each of the
    /// exploiter's hands gains by deviating to a true best response against
    /// the opponent's CURRENT strategy — locks included, which is the point:
    /// lock villain to a pool tendency, then read off what beats it. The
    /// exploiter's own locked nodes are constraints the best responder cannot
    /// deviate at (mirroring `br_into`). The path is canonicalized and the
    /// per-hand results mapped back through the suit permutation, so browsing
    /// an isomorphic runout reports the browsed branch's own combos.
    pub fn exploit_view(&self, path: &[PathStep], p: usize) -> Result<ExploitView, String> {
        if p > 1 {
            return Err("exploiter must be 0 (OOP) or 1 (IP)".to_string());
        }
        let spot = &*self.spot;
        let (canon, perm) = self.canonical_path_with_perm(path);
        let walk = self.walk_path(&canon)?;
        let node = &spot.tree.nodes[walk.node_idx as usize];
        let nh = spot.hands[p].len();
        let reach_o = &walk.reach[1 - p];
        let valid = self.valid_opp_sum(p, reach_o);

        // canonical-branch hand index for each browsed-branch hand
        // (identity table when the path needed no remapping)
        let k = spot
            .suit_perms
            .iter()
            .position(|pm| *pm == perm)
            .unwrap_or(0);
        let tbl = &spot.hand_perm[p][k];

        let is_actor = node.kind == KIND_ACTION && node.player as usize == p;
        let na = if node.kind == KIND_ACTION {
            node.num_children as usize
        } else {
            0
        };
        let lock = self.locks.get(&walk.node_idx);

        // Per-action BR continuation values at actor nodes, then the per-hand
        // BR value here (argmax, or the lock-weighted sum when constrained).
        let mut acfv: Vec<Vec<f32>> = Vec::new();
        let br_cfv: Vec<f32> = if is_actor {
            for a in 0..na {
                let child = spot.tree.children[node.children_start as usize + a];
                acfv.push(self.traverse_br(child, p, reach_o, walk.dealt));
            }
            (0..nh)
                .map(|i| match lock {
                    Some(l) => (0..na).map(|a| l[a * nh + i] * acfv[a][i]).sum(),
                    None => (0..na)
                        .map(|a| acfv[a][i])
                        .fold(f32::NEG_INFINITY, f32::max),
                })
                .collect()
        } else {
            self.traverse_br(walk.node_idx, p, reach_o, walk.dealt)
        };
        let cur_cfv = self.traverse_avg(walk.node_idx, p, reach_o, walk.dealt);

        let put = node.put[p];
        let ev_of = |cfv: f32, i: usize| -> Option<f32> {
            if valid[i] > 1e-9 {
                Some((cfv as f64 / valid[i] + put) as f32)
            } else {
                None
            }
        };

        let hands: Vec<ExploitHandView> = (0..nh)
            .map(|i| {
                let h = &spot.hands[p][i];
                let j = tbl[i] as usize;
                let br_ev = ev_of(br_cfv[j], j);
                let cur_ev = ev_of(cur_cfv[j], j);
                let gain = match (br_ev, cur_ev) {
                    (Some(b), Some(c)) => Some(b - c),
                    _ => None,
                };
                let (br_strategy, evs) = if is_actor {
                    let strat: Vec<f32> = match lock {
                        Some(l) => (0..na).map(|a| l[a * nh + j]).collect(),
                        None => {
                            let m = (0..na)
                                .map(|a| acfv[a][j])
                                .fold(f32::NEG_INFINITY, f32::max);
                            let eps = (m.abs() * 1e-5).max(1e-5);
                            let win: Vec<bool> =
                                (0..na).map(|a| m - acfv[a][j] <= eps).collect();
                            let w = win.iter().filter(|&&x| x).count().max(1) as f32;
                            (0..na)
                                .map(|a| if win[a] { 1.0 / w } else { 0.0 })
                                .collect()
                        }
                    };
                    let evs = (0..na).map(|a| ev_of(acfv[a][j], j)).collect();
                    (Some(strat), Some(evs))
                } else {
                    (None, None)
                };
                ExploitHandView {
                    combo: combo_to_string(h.c1, h.c2),
                    c1: h.c1,
                    c2: h.c2,
                    weight: h.weight,
                    reach: walk.reach[p][j],
                    valid: valid[j] as f32,
                    br_ev,
                    cur_ev,
                    gain,
                    br_strategy,
                    evs,
                }
            })
            .collect();

        // Banner averages weight by reach x valid: per-hand EVs are
        // normalized by valid, so only this weighting aggregates them back to
        // a true range EV (the reach x valid convention used everywhere else).
        let mut wsum = 0f64;
        let (mut b, mut c) = (0f64, 0f64);
        for h in &hands {
            if let (Some(be), Some(ce)) = (h.br_ev, h.cur_ev) {
                let w = h.reach as f64 * h.valid as f64;
                if w > 0.0 {
                    wsum += w;
                    b += w * be as f64;
                    c += w * ce as f64;
                }
            }
        }
        let (avg_br_ev, avg_cur_ev, avg_gain) = if wsum > 1e-12 {
            (
                Some((b / wsum) as f32),
                Some((c / wsum) as f32),
                Some(((b - c) / wsum) as f32),
            )
        } else {
            (None, None, None)
        };

        Ok(ExploitView {
            node_type: match node.kind {
                KIND_ACTION => "action",
                KIND_CHANCE => "chance",
                KIND_TERM_FOLD => "terminal_fold",
                KIND_TERM_SHOWDOWN => "terminal_showdown",
                _ => "?",
            }
            .to_string(),
            exploiter: p as u8,
            player: if node.kind == KIND_ACTION {
                Some(node.player)
            } else {
                None
            },
            actions: self.action_views(walk.node_idx),
            locked: self.locks.contains_key(&walk.node_idx),
            hands,
            avg_br_ev,
            avg_cur_ev,
            avg_gain,
        })
    }

    /// Per-card strategy/equity overview at a chance node.
    pub fn runouts(&self, path: &[PathStep]) -> Result<RunoutsReport, String> {
        let spot = &*self.spot;
        let walk = self.walk_path(path)?;
        let node = &spot.tree.nodes[walk.node_idx as usize];
        if node.kind != KIND_CHANCE {
            return Err("runouts report requires a chance (card) node".to_string());
        }
        let mut actions: Vec<ActionView> = Vec::new();
        let mut player: Option<u8> = None;
        let mut rows: Vec<RunoutRow> = Vec::new();
        let mut pot = 0f64;

        for c in (0..52u8).rev() {
            let child_idx = spot.tree.children[node.children_start as usize + c as usize];
            if child_idx == SENTINEL || walk.dealt.contains(c) {
                continue;
            }
            let child = &spot.tree.nodes[child_idx as usize];
            let dealt2 = walk.dealt.push(c);
            let cm = card_mask(c);
            // filter reach for both players
            let mut reach = [walk.reach[0].clone(), walk.reach[1].clone()];
            for p in 0..2 {
                for (i, h) in spot.hands[p].iter().enumerate() {
                    if h.mask & cm != 0 {
                        reach[p][i] = 0.0;
                    }
                }
            }
            let mut freqs: Vec<f32> = Vec::new();
            if child.kind == KIND_ACTION {
                let p = child.player as usize;
                player = Some(child.player);
                let na = child.num_children as usize;
                if actions.is_empty() {
                    actions = self.action_views(child_idx);
                }
                let nh = spot.hands[p].len();
                let sigma = self.average_strategy(child_idx, child);
                let mut total = 0f64;
                let mut sums = vec![0f64; na];
                for i in 0..nh {
                    let r = reach[p][i] as f64;
                    total += r;
                    for a in 0..na {
                        sums[a] += r * sigma[a * nh + i] as f64;
                    }
                }
                freqs = sums
                    .iter()
                    .map(|&s| if total > 1e-12 { (s / total) as f32 } else { 0.0 })
                    .collect();
            }
            // average equities and average-strategy EVs on this runout (EVs
            // in the node_view convention: cfv normalized by non-blocking
            // opponent mass, plus own contribution => pot-share). Averages
            // weight by reach x valid (pair mass): per-hand values are
            // normalized by that hand's valid mass, so only this weighting
            // keeps EV_OOP + EV_IP = pot.
            let mut eq_avg: [Option<f32>; 2] = [None, None];
            let mut ev_avg: [Option<f32>; 2] = [None, None];
            for p in 0..2 {
                let valid = self.valid_opp_sum(p, &reach[1 - p]);
                let eq = self.equity(p, &reach[1 - p], dealt2);
                let mut n = 0f64;
                let mut d = 0f64;
                for (i, &e) in eq.iter().enumerate() {
                    if !e.is_nan() {
                        let w = reach[p][i] as f64 * valid[i];
                        n += w * e as f64;
                        d += w;
                    }
                }
                eq_avg[p] = if d > 1e-12 {
                    Some((n / d) as f32)
                } else {
                    None
                };
                let cfv = self.traverse_avg(child_idx, p, &reach[1 - p], dealt2);
                let (mut n, mut d) = (0f64, 0f64);
                for i in 0..spot.hands[p].len() {
                    if valid[i] > 1e-9 && reach[p][i] > 0.0 {
                        let ev = cfv[i] as f64 / valid[i] + child.put[p];
                        let w = reach[p][i] as f64 * valid[i];
                        n += w * ev;
                        d += w;
                    }
                }
                ev_avg[p] = if d > 1e-12 { Some((n / d) as f32) } else { None };
            }
            if pot == 0.0 {
                pot = child.put[0] + child.put[1];
            }
            rows.push(RunoutRow {
                card: card_to_string(c),
                freqs,
                eq: eq_avg,
                ev: ev_avg,
            });
        }
        Ok(RunoutsReport {
            actions,
            player,
            pot,
            rows,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A hand whose entire solved mass sits on zero-target actions can't be
    /// raked; it must be assigned the target distribution — never uniform
    /// over all actions, which would leak forbidden actions into the lock.
    #[test]
    fn rake_assigns_target_to_unrakeable_hands() {
        // 2 actions x 3 hands; hand 2 is pure on action 1 (the forbidden one)
        let mut sigma = vec![
            0.6, 0.3, 0.0, // action 0
            0.4, 0.7, 1.0, // action 1
        ];
        let reach = [1.0f32, 1.0, 1.0];
        let target = [1.0f32, 0.0];
        rake_to_target(&mut sigma, 2, 3, &reach, &target);
        for i in 0..3 {
            assert!(
                (sigma[i] - 1.0).abs() < 1e-6,
                "hand {i} action0 should be 1.0, got {}",
                sigma[i]
            );
            assert!(
                sigma[3 + i].abs() < 1e-6,
                "hand {i} action1 should be 0.0, got {}",
                sigma[3 + i]
            );
        }
    }
}

// ======================= postflop player profiles ==========================
//
// The preflop player model continued past the flop: HUD stats compile into
// Range-style locks over EVERY node where the villain acts, distorting the
// SOLVED strategy (hands that bet most keep betting most — never hand-blind)
// to the stat targets. Same philosophy as the preflop generator.

fn default_bet_size() -> String {
    "min".into()
}

/// Postflop HUD stats for a modeled villain. Percent units throughout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostflopStats {
    /// Bet frequency WITH the initiative, per street [flop, turn, river]
    /// (flop = c-bet, turn/river = barrels).
    pub cbet: [f32; 3],
    /// Fold frequency facing a bet or raise, per street. Applies at every
    /// raise depth (same simplification as preflop fold-to-3bet+).
    pub fold_to_bet: [f32; 3],
    /// Raise frequency facing a bet (any street, any depth).
    #[serde(default)]
    pub raise_bet: f32,
    /// Bet frequency WITHOUT the initiative (donk / stab), any street.
    #[serde(default)]
    pub donk: f32,
    /// Which size takes the bet/raise mass: "min" | "max".
    #[serde(default = "default_bet_size")]
    pub bet_size: String,
}

/// Reach-weighted target-vs-achieved readback per situation (trust check —
/// rake can't force actions the solved strategy never uses).
#[derive(Debug, Clone, Serialize)]
pub struct ProfileLockRow {
    pub label: String,
    pub target: f32,
    pub achieved: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProfileLockSummary {
    pub locked: usize,
    pub rows: Vec<ProfileLockRow>,
}

const PROFILE_LABEL: &str = "profile:";

impl Solver {
    /// Remove every profile-generated lock (manual point-locks keep the
    /// labels the user gave them and survive). Returns how many were cleared.
    pub fn clear_profile_locks(&mut self) -> usize {
        let keys: Vec<u32> = self
            .lock_labels
            .iter()
            .filter(|(_, l)| l.starts_with(PROFILE_LABEL))
            .map(|(k, _)| *k)
            .collect();
        for k in &keys {
            self.locks.remove(k);
            self.lock_labels.remove(k);
        }
        if !keys.is_empty() {
            self.mark_sym_dirty();
        }
        keys.len()
    }

    /// Lock `villain`'s whole tree to a postflop stat profile. Walks the
    /// orbit-representative branches (the ones the solver actually visits),
    /// classifying each villain node by street / initiative / facing-bet,
    /// and rakes the SOLVED strategy to the stat targets with reaches that
    /// reflect the locks applied upstream. Manual (non-"profile:") locks
    /// take precedence and are left untouched. Re-applying first clears the
    /// previous profile locks, so it never compounds.
    ///
    /// `pf_aggressor`: who arrives with the initiative (last preflop
    /// raiser), if anyone — decides c-bet vs donk classification on the
    /// first betting round.
    pub fn lock_profile(
        &mut self,
        villain: usize,
        stats: &PostflopStats,
        pf_aggressor: Option<usize>,
    ) -> Result<ProfileLockSummary, String> {
        if villain > 1 {
            return Err("villain must be 0 (OOP) or 1 (IP)".into());
        }
        if self.iteration == 0 {
            return Err(
                "solve the spot first — the profile distorts the solved strategy".into(),
            );
        }
        self.clear_profile_locks();
        let reach: Vec<f32> = self.spot.weights[villain].clone();
        let root_mass: f32 = reach.iter().sum();
        if root_mass <= 0.0 {
            return Err("villain range is empty".into());
        }
        let eps = root_mass * 1e-6;
        let mut locked = 0usize;
        // slots: 0-2 bet-with-initiative per street, 3-5 fold-vs-bet per
        // street, 6 raise-vs-bet, 7 bet-without-initiative
        let mut acc = [(0f64, 0f64); 8];
        self.profile_walk(
            0,
            &reach,
            Dealt::default(),
            pf_aggressor,
            villain,
            stats,
            eps,
            &mut locked,
            &mut acc,
        );
        let street_name = |s: usize| ["flop", "turn", "river"][s.min(2)];
        let mut rows = Vec::new();
        for s in 0..3 {
            if acc[s].0 > 0.0 {
                rows.push(ProfileLockRow {
                    label: format!("{} bet (initiative)", street_name(s)),
                    target: stats.cbet[s],
                    achieved: (acc[s].1 / acc[s].0 * 100.0) as f32,
                });
            }
            if acc[3 + s].0 > 0.0 {
                rows.push(ProfileLockRow {
                    label: format!("{} fold vs bet", street_name(s)),
                    target: stats.fold_to_bet[s],
                    achieved: (acc[3 + s].1 / acc[3 + s].0 * 100.0) as f32,
                });
            }
        }
        if acc[6].0 > 0.0 {
            rows.push(ProfileLockRow {
                label: "raise vs bet".into(),
                target: stats.raise_bet,
                achieved: (acc[6].1 / acc[6].0 * 100.0) as f32,
            });
        }
        if acc[7].0 > 0.0 {
            rows.push(ProfileLockRow {
                label: "bet (no initiative)".into(),
                target: stats.donk,
                achieved: (acc[7].1 / acc[7].0 * 100.0) as f32,
            });
        }
        Ok(ProfileLockSummary { locked, rows })
    }

    #[allow(clippy::too_many_arguments)]
    fn profile_walk(
        &mut self,
        node_idx: u32,
        reach: &[f32],
        dealt: Dealt,
        aggressor: Option<usize>,
        villain: usize,
        stats: &PostflopStats,
        eps: f32,
        locked: &mut usize,
        acc: &mut [(f64, f64); 8],
    ) {
        let mass: f32 = reach.iter().sum();
        if mass < eps {
            return; // villain effectively never here — irrelevant to exploit
        }
        let node = self.spot.tree.nodes[node_idx as usize].clone();
        match node.kind {
            KIND_TERM_FOLD | KIND_TERM_SHOWDOWN => {}
            KIND_CHANCE => {
                // Orbit representatives only, mirroring chance_node(): those
                // are the branches CFR visits (siblings are synthesized).
                // Reach is scaled by orbit_size/divisor so downstream masses
                // are probability-weighted — otherwise every runout branch
                // counts at full weight and the cross-street summary rows
                // drown the flop. (A per-node scalar cancels inside the rake
                // itself, but the eps skip and the readback need real mass.)
                let spot = self.spot.clone();
                let cs = node.children_start as usize;
                let divisor = (46 - node.street as i32) as f32;
                let mut rep_of = [255u8; 52];
                let use_iso = self.use_isomorphism && spot.suit_perms.len() > 1;
                let valid = if use_iso { spot.perms_fixing(&dealt) } else { Vec::new() };
                for c in 0..52u8 {
                    if spot.tree.children[cs + c as usize] == SENTINEL || dealt.contains(c) {
                        continue;
                    }
                    let mut rep = c;
                    if use_iso {
                        for &k in &valid {
                            let pc = crate::cards::permute_card(c, &spot.suit_perms[k]);
                            if pc < rep {
                                rep = pc;
                            }
                        }
                    }
                    rep_of[c as usize] = rep;
                }
                let nh = spot.hands[villain].len();
                for c in 0..52u8 {
                    if rep_of[c as usize] != c {
                        continue; // not a representative (or not dealable)
                    }
                    let orbit = rep_of.iter().filter(|&&r| r == c).count() as f32;
                    let child = spot.tree.children[cs + c as usize];
                    let cm = 1u64 << c;
                    let scale = orbit / divisor;
                    let mut r = vec![0f32; nh];
                    for i in 0..nh {
                        if spot.hands[villain][i].mask & cm == 0 {
                            r[i] = reach[i] * scale;
                        }
                    }
                    self.profile_walk(
                        child, &r, dealt.push(c), aggressor, villain, stats, eps, locked,
                        acc,
                    );
                }
            }
            KIND_ACTION => {
                let na = node.num_children as usize;
                let a0 = node.actions_start as usize;
                let acts: Vec<Action> = self.spot.tree.actions[a0..a0 + na].to_vec();
                let actor = node.player as usize;
                let nh = self.spot.hands[actor].len();

                let sigma: Option<Vec<f32>> = if actor == villain {
                    Some(self.profile_lock_node(
                        node_idx, &node, &acts, reach, aggressor, villain, stats, locked,
                        acc,
                    ))
                } else {
                    None
                };

                for (a, act) in acts.iter().enumerate() {
                    let child =
                        self.spot.tree.children[node.children_start as usize + a];
                    let next_aggr = match act {
                        Action::Bet(_) | Action::Raise(_) => Some(actor),
                        _ => aggressor,
                    };
                    if let Some(sig) = &sigma {
                        let mut r = vec![0f32; nh];
                        for h in 0..nh {
                            r[h] = reach[h] * sig[a * nh + h];
                        }
                        self.profile_walk(
                            child, &r, dealt, next_aggr, villain, stats, eps, locked, acc,
                        );
                    } else {
                        self.profile_walk(
                            child, reach, dealt, next_aggr, villain, stats, eps, locked,
                            acc,
                        );
                    }
                }
            }
            _ => {}
        }
    }

    /// Build + install the lock for one villain node; returns the sigma the
    /// villain plays there (the fresh lock, or a pre-existing manual lock).
    #[allow(clippy::too_many_arguments)]
    fn profile_lock_node(
        &mut self,
        node_idx: u32,
        node: &crate::tree::Node,
        acts: &[Action],
        reach: &[f32],
        aggressor: Option<usize>,
        villain: usize,
        stats: &PostflopStats,
        locked: &mut usize,
        acc: &mut [(f64, f64); 8],
    ) -> Vec<f32> {
        let na = acts.len();
        let nh = self.spot.hands[villain].len();
        let st = (node.street as usize).min(2);

        // classify: indices of fold / passive (check|call) / aggressive acts
        let mut fold_i = None;
        let mut passive_i = None;
        let mut aggro: Vec<(usize, f64)> = Vec::new();
        for (a, act) in acts.iter().enumerate() {
            match act {
                Action::Fold => fold_i = Some(a),
                Action::Check | Action::Call(_) => passive_i = Some(a),
                Action::Bet(x) | Action::Raise(x) => aggro.push((a, *x)),
            }
        }
        let facing = fold_i.is_some();
        let pref = if stats.bet_size == "max" {
            aggro.iter().max_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).map(|x| x.0)
        } else {
            aggro.iter().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).map(|x| x.0)
        };

        // pre-existing manual lock wins (point read > profile), like preflop
        if let Some(lbl) = self.lock_labels.get(&node_idx) {
            if !lbl.starts_with(PROFILE_LABEL) {
                return self.locks[&node_idx].clone();
            }
        }

        let mut target = vec![0f32; na];
        let label;
        if facing {
            let f = (stats.fold_to_bet[st] / 100.0).clamp(0.0, 1.0);
            let mut r = (stats.raise_bet / 100.0).clamp(0.0, 1.0).min(1.0 - f);
            if pref.is_none() {
                r = 0.0; // facing an all-in: no raise exists, mass goes to call
            }
            target[fold_i.unwrap()] = f;
            if let Some(ci) = passive_i {
                target[ci] = (1.0 - f - r).max(0.0);
            }
            if let Some(pi) = pref {
                target[pi] = r;
            }
            label = format!(
                "{PROFILE_LABEL} {} fold {:.0}% / raise {:.0}%",
                ["flop", "turn", "river"][st],
                f * 100.0,
                r * 100.0
            );
        } else {
            let with_init = aggressor == Some(villain);
            let bf_pct = if with_init { stats.cbet[st] } else { stats.donk };
            let mut bf = (bf_pct / 100.0).clamp(0.0, 1.0);
            if pref.is_none() {
                bf = 0.0;
            }
            if let Some(ci) = passive_i {
                target[ci] = 1.0 - bf;
            }
            if let Some(pi) = pref {
                target[pi] = bf;
            }
            label = format!(
                "{PROFILE_LABEL} {} {} {:.0}%",
                ["flop", "turn", "river"][st],
                if with_init { "c-bet" } else { "stab" },
                bf * 100.0
            );
        }

        let mut sigma = vec![0f32; na * nh];
        self.solved_strategy_into(node, &mut sigma);
        rake_to_target(&mut sigma, na, nh, reach, &target);
        self.locks.insert(node_idx, sigma.clone());
        self.lock_labels.insert(node_idx, label);
        self.mark_sym_dirty();
        *locked += 1;

        // achieved-frequency accounting (reach-weighted)
        let mass: f64 = reach.iter().map(|&x| x as f64).sum();
        let freq_of = |set: &dyn Fn(usize) -> bool| -> f64 {
            let mut s = 0f64;
            for a in 0..na {
                if set(a) {
                    for h in 0..nh {
                        s += reach[h] as f64 * sigma[a * nh + h] as f64;
                    }
                }
            }
            s / mass.max(1e-12)
        };
        // aggro rows only count nodes where an aggressive action EXISTS —
        // a street with no raise size configured can't meet any raise
        // target, and folding that impossibility into the average misleads
        let is_aggro = |a: usize| acts[a].is_aggressive();
        if facing {
            let fi = fold_i.unwrap();
            let fold_f = freq_of(&|a| a == fi);
            acc[3 + st].0 += mass;
            acc[3 + st].1 += mass * fold_f;
            if pref.is_some() {
                let raise_f = freq_of(&is_aggro);
                acc[6].0 += mass;
                acc[6].1 += mass * raise_f;
            }
        } else if pref.is_some() {
            let bet_f = freq_of(&is_aggro);
            let slot = if aggressor == Some(villain) { st } else { 7 };
            acc[slot].0 += mass;
            acc[slot].1 += mass * bet_f;
        }
        sigma
    }
}

// ===================== M5: realization observations ========================

/// One 169-class observation at a solved flop root — the raw material for
/// calibrating the Preflop Lab's equity-realization model. r_obs =
/// EV / (pot × EQ): how much of its raw equity share the class actually
/// converts under solved postflop play (1.0 = exactly its share).
#[derive(Debug, Clone, Serialize)]
pub struct RealizationObs {
    pub board: String,
    /// 0 = OOP, 1 = IP.
    pub player: u8,
    /// Postflop acting order in [-0.5, +0.5]: OOP -0.5, IP +0.5 (the
    /// preflop model's pos_frac convention).
    pub pos_frac: f64,
    /// Effective stack / pot at the flop root.
    pub spr: f64,
    pub n_players: u8,
    /// 169-class index in the preflop lattice convention, plus its label.
    pub class: usize,
    pub label: String,
    pub reach: f64,
    pub eq: f64,
    pub ev: f64,
    pub r_obs: f64,
}

impl Solver {
    /// Reach-weighted per-class realization observations at the root.
    /// Classes with reach under 1% of the busiest class are dropped
    /// (noise), as are classes with negligible equity (r_obs explodes).
    pub fn realization_observations(&self) -> Result<Vec<RealizationObs>, String> {
        use crate::preflop::equity::{class_index, class_label};
        let view = self.node_view(&[])?;
        let tc = &self.spot.tree.config;
        let pot = tc.starting_pot;
        let spr = tc.effective_stack / pot.max(1e-9);
        let mut out = Vec::new();
        for p in 0..2usize {
            let mut w = vec![0f64; 169];
            let mut wv = vec![0f64; 169];
            let mut se = vec![0f64; 169];
            let mut sv = vec![0f64; 169];
            for h in &view.players[p].hands {
                let (eq, ev) = match (h.eq, h.ev) {
                    (Some(a), Some(b)) => (a as f64, b as f64),
                    _ => continue,
                };
                let (r1, r2) = (rank(h.c1), rank(h.c2));
                let (hi, lo) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
                let k = class_index(hi, lo, suit(h.c1) == suit(h.c2));
                // eq/ev means weight by reach x valid (per-hand values are
                // normalized by valid, so only the pair mass aggregates
                // consistently); the reach output stays plain reach mass —
                // it is the fit's observation weight, not an EV aggregate
                let pw = h.reach as f64 * h.valid as f64;
                w[k] += h.reach as f64;
                wv[k] += pw;
                se[k] += pw * eq;
                sv[k] += pw * ev;
            }
            let wmax = w.iter().cloned().fold(0.0f64, f64::max);
            for k in 0..169 {
                if w[k] <= 0.0 || w[k] < wmax * 0.01 || wv[k] <= 0.0 {
                    continue;
                }
                let (eq, ev) = (se[k] / wv[k], sv[k] / wv[k]);
                if eq < 0.02 {
                    continue;
                }
                out.push(RealizationObs {
                    board: self.spot.config.board.clone(),
                    player: p as u8,
                    pos_frac: if p == 0 { -0.5 } else { 0.5 },
                    spr,
                    n_players: 2,
                    class: k,
                    label: class_label(k),
                    reach: w[k],
                    eq,
                    ev,
                    r_obs: ev / (pot * eq),
                });
            }
        }
        Ok(out)
    }
}
