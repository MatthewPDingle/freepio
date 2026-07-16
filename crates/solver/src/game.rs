//! A `Spot` bundles everything needed to solve one postflop scenario:
//! the built tree, both players' hand lists, and precomputed river strengths.

use crate::cards::*;
use crate::evaluator::evaluate7;
use crate::range::Range;
use crate::scratch::Buf;
use crate::store::Storage;
use crate::tree::{Strictness, Tree, TreeBuilder, TreeConfig, SENTINEL};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct HandInfo {
    pub c1: Card,
    pub c2: Card,
    pub weight: f32,
    pub mask: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotConfig {
    pub board: String,
    pub range_oop: String,
    pub range_ip: String,
    pub tree: TreeConfig,
}

/// Per-river-board sorted hand strengths for both players.
/// Hands conflicting with the full 5-card board are excluded.
pub struct RiverBoardEval {
    /// (strength, index into Spot::hands[p]) sorted ascending by strength.
    pub sorted: [Vec<(u32, u16)>; 2],
}

pub struct RiverTable {
    pub(crate) root_street: u8,
    /// root_street 0: indexed lo*52+hi (lo < hi); 1: indexed by river card; 2: index 0.
    pub(crate) entries: Vec<Option<Box<RiverBoardEval>>>,
}

impl RiverTable {
    /// Linear key for a set of dealt cards (see `entries` indexing).
    pub(crate) fn key(&self, dealt: &Dealt) -> usize {
        match self.root_street {
            2 => 0,
            1 => {
                debug_assert_eq!(dealt.len, 1);
                dealt.cards[0] as usize
            }
            _ => {
                debug_assert_eq!(dealt.len, 2);
                let (a, b) = (dealt.cards[0].min(dealt.cards[1]), dealt.cards[0].max(dealt.cards[1]));
                a as usize * 52 + b as usize
            }
        }
    }

    pub fn get(&self, dealt: &Dealt) -> &RiverBoardEval {
        self.entries[self.key(dealt)].as_deref().expect("river eval missing")
    }
}

/// Board cards dealt during traversal (beyond the initial board).
#[derive(Debug, Copy, Clone, Default)]
pub struct Dealt {
    pub cards: [Card; 2],
    pub len: u8,
}

impl Dealt {
    pub fn push(mut self, c: Card) -> Self {
        self.cards[self.len as usize] = c;
        self.len += 1;
        self
    }
    pub fn contains(&self, c: Card) -> bool {
        (0..self.len as usize).any(|i| self.cards[i] == c)
    }
}

pub struct Spot {
    pub config: SpotConfig,
    pub board: Vec<Card>,
    pub board_mask: u64,
    pub tree: Tree,
    pub hands: [Vec<HandInfo>; 2],
    /// Initial range weights per hand (reach vector at the root).
    pub weights: [Vec<f32>; 2],
    /// For each hand of player p, index in the opponent's list holding the
    /// identical combo (SENTINEL if the opponent range lacks it).
    pub same_combo: [Vec<u32>; 2],
    pub river: RiverTable,
    /// Suit permutations under which the board (pointwise) and both ranges
    /// are invariant. Entry 0 is always the identity.
    pub suit_perms: Vec<[u8; 4]>,
    /// hand_perm[p][k][i] = index in hands[p] of suit_perms[k] applied to
    /// hands[p][i] (the permuted combo is guaranteed to exist in range).
    pub hand_perm: [Vec<Vec<u16>>; 2],
}

impl Spot {
    pub fn new(config: SpotConfig) -> Result<Spot, String> {
        Spot::new_with_limit(config, None)
    }

    /// [`Spot::new`] with a tree node budget: construction aborts with an
    /// error once the tree grows past `max_nodes` nodes (None = no cap), so
    /// an oversized config errors out early instead of OOMing before the
    /// caller can size-check the finished spot.
    pub fn new_with_limit(config: SpotConfig, max_nodes: Option<usize>) -> Result<Spot, String> {
        Spot::new_impl(config, max_nodes, Strictness::Strict)
    }

    /// [`Spot::new`] in [`Strictness::LenientLoad`] mode, for rebuilding a
    /// spot from a SAVED config only (never for new builds): sizing quirks
    /// the pre-validation tree builder silently dropped (a raise-only "2.5x"
    /// in a bet/donk list) are dropped again instead of rejected, so the
    /// rebuilt tree layout is byte-identical to the one the save's arenas
    /// were written against and old .gto files keep loading.
    pub fn new_lenient(config: SpotConfig) -> Result<Spot, String> {
        Spot::new_impl(config, None, Strictness::LenientLoad)
    }

    /// [`Spot::new_lenient`] with a tree node budget (see
    /// [`Spot::new_with_limit`]); for save-vetting paths that rebuild the
    /// tree under a memory cap.
    pub fn new_lenient_with_limit(
        config: SpotConfig,
        max_nodes: Option<usize>,
    ) -> Result<Spot, String> {
        Spot::new_impl(config, max_nodes, Strictness::LenientLoad)
    }

    fn new_impl(
        config: SpotConfig,
        max_nodes: Option<usize>,
        strictness: Strictness,
    ) -> Result<Spot, String> {
        // fraction-vs-percent confusion charges the full cap on every pot
        if config.tree.rake_pct >= 1.0 {
            return Err(format!(
                "rake_pct is a FRACTION of the pot (0.05 = 5%), got {} — did you pass a percent?",
                config.tree.rake_pct
            ));
        }
        // negative rake would mint chips: every terminal would pay the
        // winner more than the opponent ever put in
        if config.tree.rake_pct < 0.0 {
            return Err(format!(
                "rake_pct must be >= 0, got {}",
                config.tree.rake_pct
            ));
        }
        if config.tree.rake_cap < 0.0 {
            return Err(format!(
                "rake_cap must be >= 0, got {}",
                config.tree.rake_cap
            ));
        }
        let board = parse_cards(&config.board)?;
        if !(3..=5).contains(&board.len()) {
            return Err("board must have 3 to 5 cards".to_string());
        }
        let mut board_mask = 0u64;
        for &c in &board {
            board_mask |= card_mask(c);
        }

        let ranges = [
            Range::parse(&config.range_oop)?,
            Range::parse(&config.range_ip)?,
        ];
        let mut hands: [Vec<HandInfo>; 2] = [Vec::new(), Vec::new()];
        for p in 0..2 {
            for idx in 0..NUM_COMBOS {
                let w = ranges[p].weights[idx];
                if w <= 0.0 {
                    continue;
                }
                let (c1, c2) = combo_from_index(idx);
                let mask = card_mask(c1) | card_mask(c2);
                if mask & board_mask != 0 {
                    continue;
                }
                hands[p].push(HandInfo {
                    c1,
                    c2,
                    weight: w,
                    mask,
                });
            }
            if hands[p].is_empty() {
                return Err(format!(
                    "{} range has no combos left after board card removal",
                    if p == 0 { "OOP" } else { "IP" }
                ));
            }
            if hands[p].len() > u16::MAX as usize {
                return Err("range too large".to_string());
            }
        }

        // Cross-index identical combos.
        let mut same_combo: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
        for p in 0..2 {
            let opp = &hands[1 - p];
            let mut map = std::collections::HashMap::with_capacity(opp.len());
            for (j, h) in opp.iter().enumerate() {
                map.insert(combo_index(h.c1, h.c2), j as u32);
            }
            same_combo[p] = hands[p]
                .iter()
                .map(|h| {
                    map.get(&combo_index(h.c1, h.c2))
                        .copied()
                        .unwrap_or(SENTINEL)
                })
                .collect();
        }

        let weights = [
            hands[0].iter().map(|h| h.weight).collect::<Vec<f32>>(),
            hands[1].iter().map(|h| h.weight).collect::<Vec<f32>>(),
        ];

        let tree = TreeBuilder::build_with_options(
            &config.tree,
            &board,
            [hands[0].len(), hands[1].len()],
            max_nodes,
            strictness,
        )?;
        let river = build_river_table(&board, board_mask, &hands);

        // Suit symmetry group: permutations fixing the board pointwise and
        // leaving both ranges invariant (equal weights for permuted combos).
        let mut suit_perms: Vec<[u8; 4]> = Vec::new();
        let mut hand_perm: [Vec<Vec<u16>>; 2] = [Vec::new(), Vec::new()];
        let idx_of: [std::collections::HashMap<usize, u16>; 2] = [0, 1].map(|p| {
            hands[p]
                .iter()
                .enumerate()
                .map(|(i, h)| (combo_index(h.c1, h.c2), i as u16))
                .collect()
        });
        let mut perm4 = Vec::new();
        for a in 0..4u8 {
            for b in 0..4u8 {
                for c in 0..4u8 {
                    for d in 0..4u8 {
                        let pm = [a, b, c, d];
                        let mut seen = [false; 4];
                        if pm.iter().all(|&s| {
                            let f = !seen[s as usize];
                            seen[s as usize] = true;
                            f
                        }) {
                            perm4.push(pm);
                        }
                    }
                }
            }
        }
        'perm: for pm in perm4 {
            // board fixed pointwise (each board card maps to itself)
            for &c in &board {
                if permute_card(c, &pm) != c {
                    continue 'perm;
                }
            }
            let mut tables: [Vec<u16>; 2] = [Vec::new(), Vec::new()];
            for p in 0..2 {
                for h in hands[p].iter() {
                    let pc = combo_index(permute_card(h.c1, &pm), permute_card(h.c2, &pm));
                    match idx_of[p].get(&pc) {
                        Some(&j) if (hands[p][j as usize].weight - h.weight).abs() < 1e-6 => {
                            tables[p].push(j)
                        }
                        _ => continue 'perm,
                    }
                }
            }
            suit_perms.push(pm);
            hand_perm[0].push(tables[0].clone());
            hand_perm[1].push(tables[1].clone());
        }
        // ensure identity is entry 0
        if let Some(id) = suit_perms.iter().position(|pm| *pm == [0, 1, 2, 3]) {
            suit_perms.swap(0, id);
            hand_perm[0].swap(0, id);
            hand_perm[1].swap(0, id);
        }

        Ok(Spot {
            config,
            board,
            board_mask,
            tree,
            hands,
            weights,
            same_combo,
            river,
            suit_perms,
            hand_perm,
        })
    }

    /// Indices into suit_perms that fix every dealt card (the stabilizer of
    /// the cards revealed so far).
    pub fn perms_fixing(&self, dealt: &Dealt) -> Vec<usize> {
        self.suit_perms
            .iter()
            .enumerate()
            .filter(|(_, pm)| {
                (0..dealt.len as usize).all(|i| permute_card(dealt.cards[i], pm) == dealt.cards[i])
            })
            .map(|(k, _)| k)
            .collect()
    }

    /// Estimated memory for solver arenas, in bytes (f32 storage).
    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes_for(Storage::F32)
    }

    /// Estimated memory for solver arenas under a given storage mode.
    pub fn arena_bytes_for(&self, storage: Storage) -> u64 {
        let entries = self.tree.data_size[0] + self.tree.data_size[1];
        match storage {
            Storage::F32 => entries * 2 * 4,
            // 2 bytes per entry plus four per-node f32 scale arrays
            Storage::Compressed => entries * 2 * 2 + self.tree.nodes.len() as u64 * 16,
        }
    }

    /// Rough VRAM needed to solve this spot on the GPU: per-node hand staging
    /// buffers, f32 regret+strategy arenas, and slack for river/lock tables.
    /// Pure arithmetic (no CUDA), so the UI can show the estimate up front even
    /// when the `gpu` feature is off. Mirrors `gpu::GpuSolver`'s allocations.
    pub fn vram_estimate_bytes(&self) -> u64 {
        let n = self.tree.nodes.len() as u64;
        let nh0 = self.hands[0].len() as u64;
        let nh1 = self.hands[1].len() as u64;
        let nh_max = nh0.max(nh1);
        let staging = n * (nh0 + nh1 + nh_max) * 4;
        let arenas = (self.tree.data_size[0] + self.tree.data_size[1]) * 2 * 4;
        staging + arenas + 512 * 1024 * 1024
    }

    pub fn num_action_nodes(&self) -> usize {
        self.tree
            .nodes
            .iter()
            .filter(|n| n.kind == crate::tree::KIND_ACTION)
            .count()
    }
}

fn build_river_table(board: &[Card], board_mask: u64, hands: &[Vec<HandInfo>; 2]) -> RiverTable {
    let root_street = (board.len() - 3) as u8;

    let eval_board = |extra: &[Card]| -> Box<RiverBoardEval> {
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        full[board.len()..board.len() + extra.len()].copy_from_slice(extra);
        debug_assert_eq!(board.len() + extra.len(), 5);
        let mut extra_mask = 0u64;
        for &c in extra {
            extra_mask |= card_mask(c);
        }
        let mut sorted: [Vec<(u32, u16)>; 2] = [Vec::new(), Vec::new()];
        let mut cards7 = [0u8; 7];
        cards7[..5].copy_from_slice(&full);
        for p in 0..2 {
            let mut v: Vec<(u32, u16)> = Vec::with_capacity(hands[p].len());
            for (i, h) in hands[p].iter().enumerate() {
                if h.mask & extra_mask != 0 {
                    continue;
                }
                cards7[5] = h.c1;
                cards7[6] = h.c2;
                v.push((evaluate7(&cards7), i as u16));
            }
            v.sort_unstable();
            sorted[p] = v;
        }
        Box::new(RiverBoardEval { sorted })
    };

    let entries: Vec<Option<Box<RiverBoardEval>>> = match root_street {
        2 => vec![Some(eval_board(&[]))],
        1 => {
            let mut slots: Vec<Option<Box<RiverBoardEval>>> = (0..52).map(|_| None).collect();
            let computed: Vec<(usize, Box<RiverBoardEval>)> = (0..52u8)
                .into_par_iter()
                .filter(|&c| board_mask & card_mask(c) == 0)
                .map(|c| (c as usize, eval_board(&[c])))
                .collect();
            for (i, e) in computed {
                slots[i] = Some(e);
            }
            slots
        }
        _ => {
            let mut slots: Vec<Option<Box<RiverBoardEval>>> =
                (0..52 * 52).map(|_| None).collect();
            let pairs: Vec<(u8, u8)> = {
                let mut v = Vec::new();
                for a in 0..52u8 {
                    if board_mask & card_mask(a) != 0 {
                        continue;
                    }
                    for b in (a + 1)..52 {
                        if board_mask & card_mask(b) != 0 {
                            continue;
                        }
                        v.push((a, b));
                    }
                }
                v
            };
            let computed: Vec<(usize, Box<RiverBoardEval>)> = pairs
                .into_par_iter()
                .map(|(a, b)| (a as usize * 52 + b as usize, eval_board(&[a, b])))
                .collect();
            for (i, e) in computed {
                slots[i] = Some(e);
            }
            slots
        }
    };

    RiverTable {
        root_street,
        entries,
    }
}

// ---------------------------------------------------------------------------
// Terminal node evaluation
// ---------------------------------------------------------------------------

/// Counterfactual value at a fold terminal: `amount` per unit of compatible
/// opponent reach.
pub fn fold_cfv(
    hands_me: &[HandInfo],
    hands_opp: &[HandInfo],
    reach_opp: &[f32],
    same_combo_me: &[u32],
    amount: f32,
    out: &mut [f32],
) {
    let mut t = 0f64;
    let mut s = [0f64; 52];
    for (j, h) in hands_opp.iter().enumerate() {
        let r = reach_opp[j] as f64;
        t += r;
        s[h.c1 as usize] += r;
        s[h.c2 as usize] += r;
    }
    for (i, h) in hands_me.iter().enumerate() {
        let same = same_combo_me[i];
        let same_r = if same != SENTINEL {
            reach_opp[same as usize]
        } else {
            0.0
        };
        let valid = (t - s[h.c1 as usize] - s[h.c2 as usize]) as f32 + same_r;
        out[i] = amount * valid;
    }
}

/// Counterfactual value at a showdown. `out` must be zeroed by the caller;
/// only hands valid on this river board are written.
#[allow(clippy::too_many_arguments)]
pub fn showdown_cfv(
    eval: &RiverBoardEval,
    me: usize,
    hands_me: &[HandInfo],
    hands_opp: &[HandInfo],
    reach_opp: &[f32],
    same_combo_me: &[u32],
    win: f32,
    lose: f32,
    tie: f32,
    out: &mut [f32],
) {
    let mine = &eval.sorted[me];
    let opps = &eval.sorted[1 - me];

    let mut t_all = 0f64;
    let mut s_all = [0f64; 52];
    for &(_, j) in opps {
        let r = reach_opp[j as usize] as f64;
        let h = &hands_opp[j as usize];
        t_all += r;
        s_all[h.c1 as usize] += r;
        s_all[h.c2 as usize] += r;
    }

    // Pass 1 (ascending): strictly weaker opponent reach for each of my hands.
    let mut lower = Buf::zeroed(mine.len());
    {
        let mut t = 0f64;
        let mut s = [0f64; 52];
        let mut j = 0usize;
        for (k, &(stren, i)) in mine.iter().enumerate() {
            while j < opps.len() && opps[j].0 < stren {
                let jj = opps[j].1 as usize;
                let r = reach_opp[jj] as f64;
                let h = &hands_opp[jj];
                t += r;
                s[h.c1 as usize] += r;
                s[h.c2 as usize] += r;
                j += 1;
            }
            let h = &hands_me[i as usize];
            lower[k] = (t - s[h.c1 as usize] - s[h.c2 as usize]) as f32;
        }
    }

    // Pass 2 (descending): strictly stronger opponent reach, then finalize.
    {
        let mut t = 0f64;
        let mut s = [0f64; 52];
        let mut j = opps.len();
        for k in (0..mine.len()).rev() {
            let (stren, i) = mine[k];
            while j > 0 && opps[j - 1].0 > stren {
                j -= 1;
                let jj = opps[j].1 as usize;
                let r = reach_opp[jj] as f64;
                let h = &hands_opp[jj];
                t += r;
                s[h.c1 as usize] += r;
                s[h.c2 as usize] += r;
            }
            let h = &hands_me[i as usize];
            let higher = (t - s[h.c1 as usize] - s[h.c2 as usize]) as f32;
            let same = same_combo_me[i as usize];
            let same_r = if same != SENTINEL {
                reach_opp[same as usize]
            } else {
                0.0
            };
            let valid = (t_all - s_all[h.c1 as usize] - s_all[h.c2 as usize]) as f32 + same_r;
            let lo = lower[k];
            let ti = valid - lo - higher;
            out[i as usize] = win * lo + lose * higher + tie * ti;
        }
    }
}

/// Accumulate win/tie/valid opponent reach per hand for equity calculation.
pub fn sweep_buckets(
    eval: &RiverBoardEval,
    me: usize,
    hands_me: &[HandInfo],
    hands_opp: &[HandInfo],
    reach_opp: &[f32],
    same_combo_me: &[u32],
    acc_win: &mut [f64],
    acc_tie: &mut [f64],
    acc_valid: &mut [f64],
) {
    let mine = &eval.sorted[me];
    let opps = &eval.sorted[1 - me];

    let mut t_all = 0f64;
    let mut s_all = [0f64; 52];
    for &(_, j) in opps {
        let r = reach_opp[j as usize] as f64;
        let h = &hands_opp[j as usize];
        t_all += r;
        s_all[h.c1 as usize] += r;
        s_all[h.c2 as usize] += r;
    }

    let mut lower = vec![0f64; mine.len()];
    {
        let mut t = 0f64;
        let mut s = [0f64; 52];
        let mut j = 0usize;
        for (k, &(stren, i)) in mine.iter().enumerate() {
            while j < opps.len() && opps[j].0 < stren {
                let jj = opps[j].1 as usize;
                let r = reach_opp[jj] as f64;
                let h = &hands_opp[jj];
                t += r;
                s[h.c1 as usize] += r;
                s[h.c2 as usize] += r;
                j += 1;
            }
            let h = &hands_me[i as usize];
            lower[k] = t - s[h.c1 as usize] - s[h.c2 as usize];
        }
    }
    {
        let mut t = 0f64;
        let mut s = [0f64; 52];
        let mut j = opps.len();
        for k in (0..mine.len()).rev() {
            let (stren, i) = mine[k];
            while j > 0 && opps[j - 1].0 > stren {
                j -= 1;
                let jj = opps[j].1 as usize;
                let r = reach_opp[jj] as f64;
                let h = &hands_opp[jj];
                t += r;
                s[h.c1 as usize] += r;
                s[h.c2 as usize] += r;
            }
            let h = &hands_me[i as usize];
            let higher = t - s[h.c1 as usize] - s[h.c2 as usize];
            let same = same_combo_me[i as usize];
            let same_r = if same != SENTINEL {
                reach_opp[same as usize] as f64
            } else {
                0.0
            };
            let valid = t_all - s_all[h.c1 as usize] - s_all[h.c2 as usize] + same_r;
            let i = i as usize;
            acc_win[i] += lower[k];
            acc_tie[i] += valid - lower[k] - higher;
            acc_valid[i] += valid;
        }
    }
}
