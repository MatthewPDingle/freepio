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
    /// Reach-weighted average EV under best response / current strategy, and
    /// the difference (chips) — the headline "how exploitable is this node".
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
}

#[derive(Debug, Clone, Serialize)]
pub struct RunoutsReport {
    pub actions: Vec<ActionView>,
    pub player: Option<u8>,
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
        Ok(())
    }

    pub fn unlock_node(&mut self, path: &[PathStep]) -> Result<bool, String> {
        let path = &self.canonical_path(path);
        let walk = self.walk_path(path)?;
        self.lock_labels.remove(&walk.node_idx);
        Ok(self.locks.remove(&walk.node_idx).is_some())
    }

    pub fn clear_locks(&mut self) {
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
                    br_ev,
                    cur_ev,
                    gain,
                    br_strategy,
                    evs,
                }
            })
            .collect();

        // reach-weighted averages for the banner
        let mut wsum = 0f64;
        let (mut b, mut c) = (0f64, 0f64);
        for h in &hands {
            if let (Some(be), Some(ce)) = (h.br_ev, h.cur_ev) {
                let w = h.reach as f64;
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
                    let pot = child.put[0] + child.put[1];
                    for a in 0..na {
                        let action = spot.tree.actions[child.actions_start as usize + a];
                        actions.push(ActionView {
                            label: action.label(pot, false),
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
                        });
                    }
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
            // average equities on this runout
            let mut eq_avg: [Option<f32>; 2] = [None, None];
            for p in 0..2 {
                let eq = self.equity(p, &reach[1 - p], dealt2);
                let mut n = 0f64;
                let mut d = 0f64;
                for (i, &e) in eq.iter().enumerate() {
                    if !e.is_nan() {
                        n += reach[p][i] as f64 * e as f64;
                        d += reach[p][i] as f64;
                    }
                }
                eq_avg[p] = if d > 1e-12 {
                    Some((n / d) as f32)
                } else {
                    None
                };
            }
            rows.push(RunoutRow {
                card: card_to_string(c),
                freqs,
                eq: eq_avg,
            });
        }
        Ok(RunoutsReport {
            actions,
            player,
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
