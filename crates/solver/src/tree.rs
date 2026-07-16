//! Game tree construction from the spot configuration (board, ranges, sizes).
//!
//! Streets are indexed 0 = flop, 1 = turn, 2 = river. Players: 0 = OOP, 1 = IP.
//! All chip amounts are tracked as f64 during construction.

use serde::{Deserialize, Serialize};

pub const OOP: u8 = 0;
pub const IP: u8 = 1;
pub const SENTINEL: u32 = u32::MAX;

/// A bet size specification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BetSize {
    /// Percentage of the pot (e.g. 75.0 = 75% pot).
    PotPct(f64),
    /// Multiple of the opponent's bet (raises only), e.g. 3.0 = raise to 3x.
    PrevMult(f64),
    /// All-in.
    AllIn,
}

/// Parse a size list like "33 75" / "50, a" / "2.5x, 100".
pub fn parse_sizes(s: &str) -> Result<Vec<BetSize>, String> {
    let mut out = Vec::new();
    for tok in s.split([',', ' ', ';']) {
        let tok = tok.trim().to_ascii_lowercase();
        if tok.is_empty() {
            continue;
        }
        if tok == "a" || tok == "allin" || tok == "all-in" {
            out.push(BetSize::AllIn);
        } else if let Some(mult) = tok.strip_suffix('x') {
            let m: f64 = mult
                .parse()
                .map_err(|_| format!("invalid size token {tok:?}"))?;
            if m <= 1.0 {
                return Err(format!("raise multiple must be > 1: {tok:?}"));
            }
            out.push(BetSize::PrevMult(m));
        } else {
            let p: f64 = tok
                .parse()
                .map_err(|_| format!("invalid size token {tok:?}"))?;
            if p <= 0.0 {
                return Err(format!("bet size must be positive: {tok:?}"));
            }
            out.push(BetSize::PotPct(p));
        }
    }
    Ok(out)
}

/// Bet sizing options for one player on one street.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreetSizing {
    #[serde(default)]
    pub bet: Vec<BetSize>,
    #[serde(default)]
    pub raise: Vec<BetSize>,
    /// OOP only: lead sizes when the opponent was the last street's aggressor.
    #[serde(default)]
    pub donk: Vec<BetSize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Total chips in the pot at the root (both players' prior contributions).
    pub starting_pot: f64,
    /// Effective remaining stack behind for each player.
    pub effective_stack: f64,
    /// Rake percentage of the final pot (0.05 = 5%).
    #[serde(default)]
    pub rake_pct: f64,
    /// Rake cap in chips (0 = no cap).
    #[serde(default)]
    pub rake_cap: f64,
    /// Sizing per street for each player: [flop, turn, river].
    pub oop: [StreetSizing; 3],
    pub ip: [StreetSizing; 3],
    /// If a bet/raise commits more than this fraction of the player's total
    /// stack, it becomes all-in instead (the "all-in threshold"). 1.0
    /// disables the conversion except for exact overshoots.
    #[serde(default = "default_allin_threshold")]
    pub allin_threshold: f64,
    /// Always offer an all-in option in addition to configured sizes.
    #[serde(default)]
    pub add_allin: bool,
    /// Maximum number of raises per street (all-in always terminates anyway).
    #[serde(default = "default_max_raises")]
    pub max_raises: u8,
}

fn default_allin_threshold() -> f64 {
    0.85
}
fn default_max_raises() -> u8 {
    10
}

impl Default for TreeConfig {
    fn default() -> Self {
        TreeConfig {
            starting_pot: 60.0,
            effective_stack: 970.0,
            rake_pct: 0.0,
            rake_cap: 0.0,
            oop: Default::default(),
            ip: Default::default(),
            allin_threshold: default_allin_threshold(),
            add_allin: false,
            max_raises: default_max_raises(),
        }
    }
}

/// Player action at a decision node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "amount")]
pub enum Action {
    Fold,
    Check,
    /// Call, matching the opponent's street contribution.
    Call(f64),
    /// Bet to this street-total amount.
    Bet(f64),
    /// Raise to this street-total amount.
    Raise(f64),
}

impl Action {
    pub fn is_aggressive(&self) -> bool {
        matches!(self, Action::Bet(_) | Action::Raise(_))
    }

    pub fn label(&self, pot_before: f64, all_in: bool) -> String {
        match self {
            Action::Fold => "Fold".to_string(),
            Action::Check => "Check".to_string(),
            Action::Call(_) => "Call".to_string(),
            Action::Bet(amt) => {
                let pct = (amt / pot_before * 100.0).round();
                if all_in {
                    format!("All-in {}", trim_num(*amt))
                } else {
                    format!("Bet {} ({}%)", trim_num(*amt), pct)
                }
            }
            Action::Raise(amt) => {
                if all_in {
                    format!("All-in {}", trim_num(*amt))
                } else {
                    format!("Raise {}", trim_num(*amt))
                }
            }
        }
    }
}

fn trim_num(x: f64) -> String {
    if (x - x.round()).abs() < 1e-6 {
        format!("{}", x.round() as i64)
    } else {
        format!("{:.2}", x)
    }
}

pub const KIND_ACTION: u8 = 0;
pub const KIND_CHANCE: u8 = 1;
pub const KIND_TERM_FOLD: u8 = 2;
pub const KIND_TERM_SHOWDOWN: u8 = 3;

/// Compact node. Children / actions live in shared arenas in `Tree`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub kind: u8,
    /// Action node: acting player. Fold terminal: the player who folded.
    pub player: u8,
    /// Street of this node (0/1/2). For chance nodes: street being dealt INTO.
    pub street: u8,
    pub num_children: u8,
    /// Index into `Tree::children`. Chance nodes own 52 slots (card -> child).
    pub children_start: u32,
    /// Action nodes: index into `Tree::actions` (num_children entries).
    pub actions_start: u32,
    /// Action nodes: offset (in f32 elements) into the regret/strategy arenas.
    pub data_offset: u64,
    /// Total contribution of each player when this node is reached.
    pub put: [f64; 2],
    /// Showdown: payoff for winner (+), loser (-), and tie delta.
    /// Fold terminal: t_win = winner's net gain, t_lose = folder's net loss (negative).
    pub t_win: f64,
    pub t_lose: f64,
    pub t_tie: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Tree {
    pub config: TreeConfig,
    /// Street of the root (0 if solving from the flop, 1 from turn, 2 from river).
    pub root_street: u8,
    pub nodes: Vec<Node>,
    pub children: Vec<u32>,
    pub actions: Vec<Action>,
    /// Total f32 elements needed per data arena, per player perspective:
    /// data_size[p] counts elements of action nodes where `player == p`.
    pub data_size: [u64; 2],
}

struct BuildState {
    street: u8,
    to_act: u8,
    put: [f64; 2],
    street_bet: [f64; 2],
    last_increment: f64,
    num_raises: u8,
    /// Last street's final aggressor (carried over check-check streets).
    last_aggressor: Option<u8>,
    /// Whether the first player already checked this street.
    checked: bool,
}

/// How strictly the builder validates the sizing config.
///
/// `Strict` is for every NEW build: a raise-only size ("2.5x") in a
/// consumed bet/donk list is a build error, because the size the user
/// configured would otherwise silently vanish from the solved tree.
///
/// `LenientLoad` is for rebuilding a tree from a SAVED config only: it
/// reproduces the legacy pre-validation behavior exactly (the raise-only
/// size is silently dropped at action generation), so a .gto file written
/// before the validation existed rebuilds the byte-identical tree layout
/// its arenas were sized against and keeps loading. It relaxes only this
/// check — configs whose pre-fix solve was itself invalid (e.g. bad rake)
/// are still rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strictness {
    Strict,
    LenientLoad,
}

pub struct TreeBuilder<'a> {
    config: &'a TreeConfig,
    root_street: u8,
    /// Cards that can never appear as chance cards (the initial board).
    board_mask: u64,
    /// Abort the build once `nodes` grows past this budget (None = no cap).
    max_nodes: Option<usize>,
    nodes: Vec<Node>,
    children: Vec<u32>,
    actions: Vec<Action>,
    /// Number of hands per player; used to compute data offsets.
    num_hands: [u64; 2],
    data_size: [u64; 2],
}

impl<'a> TreeBuilder<'a> {
    pub fn build(
        config: &'a TreeConfig,
        board: &[u8],
        num_hands: [usize; 2],
    ) -> Result<Tree, String> {
        Self::build_with_limit(config, board, num_hands, None)
    }

    /// [`TreeBuilder::build`] with a node budget: the build aborts with an
    /// error once the tree grows past `max_nodes` nodes (None = no cap), so
    /// an oversized config fails fast instead of exhausting memory before
    /// any post-build size check can run.
    pub fn build_with_limit(
        config: &'a TreeConfig,
        board: &[u8],
        num_hands: [usize; 2],
        max_nodes: Option<usize>,
    ) -> Result<Tree, String> {
        Self::build_with_options(config, board, num_hands, max_nodes, Strictness::Strict)
    }

    /// [`TreeBuilder::build_with_limit`] with explicit [`Strictness`]. Every
    /// new build must use `Strict` (the wrappers above do); `LenientLoad` is
    /// reserved for rebuilding a tree from a saved config, where layout
    /// compatibility with the pre-validation builder matters more than
    /// flagging a size the legacy builder silently dropped.
    pub fn build_with_options(
        config: &'a TreeConfig,
        board: &[u8],
        num_hands: [usize; 2],
        max_nodes: Option<usize>,
        strictness: Strictness,
    ) -> Result<Tree, String> {
        if !(3..=5).contains(&board.len()) {
            return Err("board must have 3 to 5 cards".to_string());
        }
        if config.starting_pot <= 0.0 || config.effective_stack <= 0.0 {
            return Err("pot and stacks must be positive".to_string());
        }
        let root_street = (board.len() - 3) as u8;
        // Reject raise-multiple sizes ("2.5x") in bet/donk lists up front:
        // with no prior bet to multiply they are meaningless there, and they
        // used to be dropped silently, solving a tree missing a size the
        // user configured. Only lists the build can consume are checked
        // (nothing below the root street is read, and IP never donks).
        // LenientLoad skips the check: saves written before it existed carry
        // such configs, and their arenas match the tree with the size
        // dropped, which `legal_actions` still reproduces.
        if strictness == Strictness::Strict {
            for (player, streets) in [("OOP", &config.oop), ("IP", &config.ip)] {
                for street in root_street as usize..3 {
                    let sizing = &streets[street];
                    let donk_used = player == "OOP" && street > root_street as usize;
                    for (field, list, used) in [
                        ("bet", &sizing.bet, true),
                        ("donk", &sizing.donk, donk_used),
                    ] {
                        if !used {
                            continue;
                        }
                        for size in list {
                            if let BetSize::PrevMult(m) = size {
                                let street_name = ["flop", "turn", "river"][street];
                                return Err(format!(
                                    "{street_name} {player} {field}: '{m}x' is a \
                                     raise-only size — use % of pot or 'a'"
                                ));
                            }
                        }
                    }
                }
            }
        }
        let mut board_mask = 0u64;
        for &c in board {
            board_mask |= 1u64 << c;
        }
        let mut builder = TreeBuilder {
            config,
            root_street,
            board_mask,
            max_nodes,
            nodes: Vec::new(),
            children: Vec::new(),
            actions: Vec::new(),
            num_hands: [num_hands[0] as u64, num_hands[1] as u64],
            data_size: [0, 0],
        };
        let half = config.starting_pot / 2.0;
        let state = BuildState {
            street: root_street,
            to_act: OOP,
            put: [half, half],
            street_bet: [0.0, 0.0],
            last_increment: 0.0,
            num_raises: 0,
            last_aggressor: None,
            checked: false,
        };
        builder.action_node(state)?;
        Ok(Tree {
            config: config.clone(),
            root_street,
            nodes: builder.nodes,
            children: builder.children,
            actions: builder.actions,
            data_size: builder.data_size,
        })
    }

    fn push_node(&mut self, node: Node) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(node);
        idx
    }

    /// Enforce the optional node budget. Called on every growth path so an
    /// oversized config aborts mid-build instead of thrashing into OOM.
    fn check_budget(&self) -> Result<(), String> {
        match self.max_nodes {
            Some(cap) if self.nodes.len() >= cap => Err(format!(
                "tree too large: over the max_nodes budget of {cap}; \
                 reduce bet/raise sizes, max_raises, or effective stack"
            )),
            _ => Ok(()),
        }
    }

    fn sizing(&self, player: u8, street: u8) -> &StreetSizing {
        let arr = if player == OOP {
            &self.config.oop
        } else {
            &self.config.ip
        };
        &arr[street as usize]
    }

    /// Generate the list of legal actions for the given state.
    fn legal_actions(&self, st: &BuildState) -> Vec<Action> {
        let me = st.to_act as usize;
        let opp = 1 - me;
        let stack_me = self.config.effective_stack - (st.put[me] - self.config.starting_pot / 2.0);
        let mut actions = Vec::new();

        let facing = st.street_bet[opp] - st.street_bet[me];
        debug_assert!(facing >= -1e-9);

        if facing > 1e-9 {
            // Facing a bet/raise: fold, call, raise.
            actions.push(Action::Fold);
            actions.push(Action::Call(st.street_bet[opp]));
            let can_raise = stack_me > facing + 1e-9 && st.num_raises < self.config.max_raises;
            if can_raise {
                let pot_after_call = st.put[0] + st.put[1] + facing;
                let max_to = st.street_bet[me] + stack_me; // all-in "to" amount
                let mut candidates: Vec<f64> = Vec::new();
                let sizing = self.sizing(st.to_act, st.street);
                for size in &sizing.raise {
                    let to = match size {
                        BetSize::PotPct(p) => st.street_bet[opp] + p / 100.0 * pot_after_call,
                        BetSize::PrevMult(m) => st.street_bet[opp] * m,
                        BetSize::AllIn => max_to,
                    };
                    candidates.push(to);
                }
                if self.config.add_allin {
                    candidates.push(max_to);
                }
                let min_to = st.street_bet[opp] + st.last_increment.max(1e-9);
                let mut tos: Vec<f64> = Vec::new();
                for mut to in candidates {
                    if to < min_to {
                        to = min_to;
                    }
                    if to >= max_to - 1e-9 || to >= self.config.allin_threshold * max_to - 1e-9 {
                        to = max_to;
                    }
                    if to > st.street_bet[opp] + 1e-9 {
                        tos.push(to);
                    }
                }
                dedupe_amounts(&mut tos);
                for to in tos {
                    actions.push(Action::Raise(to));
                }
            }
        } else {
            // No bet yet this street: check or bet.
            actions.push(Action::Check);
            if stack_me > 1e-9 {
                let sizing = self.sizing(st.to_act, st.street);
                // Donk situation: OOP leading into the previous street's aggressor.
                let donking = st.to_act == OOP
                    && st.street > self.root_street
                    && st.last_aggressor == Some(IP);
                let size_list = if donking { &sizing.donk } else { &sizing.bet };
                let pot = st.put[0] + st.put[1];
                let max_to = stack_me;
                let mut candidates: Vec<f64> = Vec::new();
                for size in size_list {
                    let to = match size {
                        BetSize::PotPct(p) => p / 100.0 * pot,
                        // Raise-only: rejected up front in strict builds. The
                        // silent drop must stay for LenientLoad — old saves'
                        // arenas were sized against the dropped-size tree.
                        BetSize::PrevMult(_) => continue,
                        BetSize::AllIn => max_to,
                    };
                    candidates.push(to);
                }
                if self.config.add_allin && (!size_list.is_empty() || !donking) {
                    candidates.push(max_to);
                }
                let mut tos: Vec<f64> = Vec::new();
                for mut to in candidates {
                    if to <= 1e-9 {
                        continue;
                    }
                    if to >= max_to - 1e-9 || to >= self.config.allin_threshold * max_to - 1e-9 {
                        to = max_to;
                    }
                    tos.push(to);
                }
                dedupe_amounts(&mut tos);
                for to in tos {
                    actions.push(Action::Bet(to));
                }
            }
        }
        actions
    }

    fn rake(&self, pot: f64) -> f64 {
        let r = self.config.rake_pct * pot;
        if self.config.rake_cap > 0.0 {
            r.min(self.config.rake_cap)
        } else {
            r
        }
    }

    fn action_node(&mut self, st: BuildState) -> Result<u32, String> {
        self.check_budget()?;
        let actions = self.legal_actions(&st);
        let n = actions.len();
        if n == 0 || n > 250 {
            return Err(format!("invalid action count {n} during tree build"));
        }

        let actions_start = self.actions.len() as u32;
        self.actions.extend(actions.iter().copied());

        let me = st.to_act as usize;
        let data_offset = self.data_size[me];
        self.data_size[me] += (n as u64) * self.num_hands[me];

        // Reserve the node before children so parents precede children.
        let node_idx = self.push_node(Node {
            kind: KIND_ACTION,
            player: st.to_act,
            street: st.street,
            num_children: n as u8,
            children_start: 0, // patched below
            actions_start,
            data_offset,
            put: st.put,
            t_win: 0.0,
            t_lose: 0.0,
            t_tie: 0.0,
        });

        let mut child_indices = Vec::with_capacity(n);
        for action in &actions {
            let child = self.apply_action(&st, *action)?;
            child_indices.push(child);
        }
        let children_start = self.children.len() as u32;
        self.children.extend(child_indices);
        self.nodes[node_idx as usize].children_start = children_start;
        Ok(node_idx)
    }

    fn apply_action(&mut self, st: &BuildState, action: Action) -> Result<u32, String> {
        let me = st.to_act as usize;
        let opp = 1 - me;
        match action {
            Action::Fold => {
                // Folder loses what they've put in beyond the start... net loss is
                // their full contribution relative to pre-hand stack; in our EV
                // convention payoffs are measured from the node's pot, so the
                // winner gains the folder's total contribution minus rake.
                let f = st.put[me];
                let pot_raked = 2.0 * f; // matched portion of the pot
                let rake = self.rake(pot_raked);
                Ok(self.push_node(Node {
                    kind: KIND_TERM_FOLD,
                    player: st.to_act, // who folded
                    street: st.street,
                    num_children: 0,
                    children_start: 0,
                    actions_start: 0,
                    data_offset: 0,
                    put: st.put,
                    t_win: f - rake,
                    t_lose: -f,
                    t_tie: 0.0,
                }))
            }
            Action::Check => {
                if st.checked {
                    // Second check: street is over, aggressor carries over.
                    self.check_behind(st)
                } else {
                    // First check: pass action to the other player.
                    let new_st = BuildState {
                        to_act: st.to_act ^ 1,
                        checked: true,
                        put: st.put,
                        street_bet: st.street_bet,
                        last_increment: st.last_increment,
                        num_raises: st.num_raises,
                        last_aggressor: st.last_aggressor,
                        street: st.street,
                    };
                    self.action_node(new_st)
                }
            }
            Action::Call(to) => {
                let mut put = st.put;
                put[me] += to - st.street_bet[me];
                let stack_after =
                    self.config.effective_stack - (put[me] - self.config.starting_pot / 2.0);
                let opp_stack_after =
                    self.config.effective_stack - (put[opp] - self.config.starting_pot / 2.0);
                let allin = stack_after <= 1e-9 || opp_stack_after <= 1e-9;
                let aggressor = Some(st.to_act ^ 1);
                self.street_end(st.street, put, aggressor, allin)
            }
            Action::Bet(to) | Action::Raise(to) => {
                let mut put = st.put;
                let add = to - st.street_bet[me];
                put[me] += add;
                let mut street_bet = st.street_bet;
                street_bet[me] = to;
                let increment = to - st.street_bet[opp].max(st.street_bet[me]);
                let new_st = BuildState {
                    street: st.street,
                    to_act: st.to_act ^ 1,
                    put,
                    street_bet,
                    last_increment: increment.max(st.last_increment),
                    num_raises: st.num_raises + matches!(action, Action::Raise(_)) as u8,
                    last_aggressor: st.last_aggressor,
                    checked: st.checked,
                };
                self.action_node(new_st)
            }
        }
    }

    /// Handle the second check of a street (called from action_node when the
    /// IP player checks behind, or check-check ends the street).
    fn check_behind(&mut self, st: &BuildState) -> Result<u32, String> {
        let aggressor = st.last_aggressor;
        self.street_end(st.street, st.put, aggressor, false)
    }

    /// Street is over: showdown, all-in runout, or next street chance node.
    fn street_end(
        &mut self,
        street: u8,
        put: [f64; 2],
        aggressor: Option<u8>,
        allin: bool,
    ) -> Result<u32, String> {
        if street == 2 {
            return Ok(self.showdown_node(2, put));
        }
        if allin {
            // Runout: chance chain with no action until showdown.
            return self.runout_chance(street, put);
        }
        self.check_budget()?;
        // Chance node into next street, then a fresh action round.
        let next_street = street + 1;
        let node_idx = self.push_node(Node {
            kind: KIND_CHANCE,
            player: 0,
            street: next_street,
            num_children: 0,
            children_start: 0,
            actions_start: 0,
            data_offset: 0,
            put,
            t_win: 0.0,
            t_lose: 0.0,
            t_tie: 0.0,
        });
        // Known waste, kept deliberately: only the static root board is
        // excluded below, so e.g. the river chance node under turn card T
        // still builds a subtree in T's own slot even though traversal
        // (cfr, best_response, gpu) skips already-dealt cards and never
        // visits it (~2% of nodes/arena on flop solves). Excluding ancestor
        // cards would renumber every node and break existing .gto saves.
        let mut child_indices = vec![SENTINEL; 52];
        for card in 0..52u8 {
            if self.board_mask & (1 << card) != 0 {
                continue;
            }
            let child_state = BuildState {
                street: next_street,
                to_act: OOP,
                put,
                street_bet: [0.0, 0.0],
                last_increment: 0.0,
                num_raises: 0,
                last_aggressor: aggressor,
                checked: false,
            };
            let _ = card;
            let child = self.action_node(child_state)?;
            child_indices[card as usize] = child;
        }
        let children_start = self.children.len() as u32;
        self.children.extend(child_indices);
        let node = &mut self.nodes[node_idx as usize];
        node.children_start = children_start;
        node.num_children = 52;
        Ok(node_idx)
    }

    fn runout_chance(&mut self, street: u8, put: [f64; 2]) -> Result<u32, String> {
        self.check_budget()?;
        let next_street = street + 1;
        let node_idx = self.push_node(Node {
            kind: KIND_CHANCE,
            player: 0,
            street: next_street,
            num_children: 0,
            children_start: 0,
            actions_start: 0,
            data_offset: 0,
            put,
            t_win: 0.0,
            t_lose: 0.0,
            t_tie: 0.0,
        });
        let mut child_indices = vec![SENTINEL; 52];
        for card in 0..52u8 {
            if self.board_mask & (1 << card) != 0 {
                continue;
            }
            let child = if next_street == 2 {
                self.showdown_node(2, put)
            } else {
                self.runout_chance(next_street, put)?
            };
            child_indices[card as usize] = child;
        }
        let children_start = self.children.len() as u32;
        self.children.extend(child_indices);
        let node = &mut self.nodes[node_idx as usize];
        node.children_start = children_start;
        node.num_children = 52;
        Ok(node_idx)
    }

    fn showdown_node(&mut self, street: u8, put: [f64; 2]) -> u32 {
        debug_assert!((put[0] - put[1]).abs() < 1e-6);
        let pot = put[0] + put[1];
        let rake = self.rake(pot);
        self.push_node(Node {
            kind: KIND_TERM_SHOWDOWN,
            player: 0,
            street,
            num_children: 0,
            children_start: 0,
            actions_start: 0,
            data_offset: 0,
            put,
            t_win: put[0] - rake,
            t_lose: -put[0],
            t_tie: -rake / 2.0,
        })
    }
}

fn dedupe_amounts(v: &mut Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
}
