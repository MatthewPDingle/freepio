# TODO

Work queue for FREEPIO. Items are written so a fresh contributor (human or
Claude) can execute them without prior context. Read `README.md` first for
the architecture; run `cargo test --release -p solver` (all green, ~5 min on
a laptop) before and after any change.

---

## 1. M5 — Calibrated equity-realization model (the big one)

**Goal.** Replace the hand-blind heuristic R in the Preflop Lab with factors
*measured from this engine's own postflop solves*, making preflop output
empirically grounded. Design agreed 2026-07-04.

**Background / current state.**
- The Preflop Lab (`crates/solver/src/preflop/`) prices flop terminals as
  `share = pot × multiway_equity × R`. R lives in
  `PreflopSolver::realization_weights()` (`preflop/mod.rs`):
  `R = 1 + 0.16 × pos_frac × min(SPR,8)/8`, pos_frac ∈ [-0.5, +0.5] by
  postflop acting order. Class-independent — 76s and Q2o get the same R.
  Selected by `PreflopConfig.realization` (serde default `"static"`;
  `"raw"` = R≡1 also supported). This makes the model too fond of offsuit
  junk and too cool on suited/connected playability hands.
- The postflop solver already exposes everything needed to MEASURE true
  realization: per-hand `ev` and `eq` at any node via
  `Solver::node_view()` (`query.rs`, `HandView.ev/.eq`, pot-share
  convention: EV_OOP + EV_IP = pot). Observed realization for hand h at the
  root of a solved postflop spot is simply
  `R_obs(h) = ev(h) / (pot × eq(h))`   (guard eq ≈ 0).
- Batch solving exists: `solve-cli batch spot.json boards.txt [iters]
  [target]` (`crates/solver/src/bin/solve_cli.rs`) writes
  `batch_results.json`; ~10–15 s/board on an RTX 3090, minutes/board on CPU.
- Spot inputs should come from real Preflop Lab exports
  (`POST /api/preflop/export {path}` → ranges/pot/eff-stack), so the fit
  covers the spots actually studied.

**Plan.**

*Phase A — observation extraction (laptop-friendly, ~1 session).*
Extend batch mode (or add `solve-cli realization <spot.json> <boards>`)
to emit, per board × player × 169-class, one JSON line:
`{board, player, pos_frac, spr, n_players: 2, class, reach, eq, ev, r_obs}`.
Aggregate combos→class with reach weighting (see `cellAgg` in
`web/js/browse.js` for the convention). Skip classes with reach < ~1% of the
class max (noise). Output: `realization_obs.jsonl`. SPR = eff_stack/pot at
the flop root. Include the tree-size config in a header line — R is
conditional on the bet-size menu used; calibrate with the menus you study.

*Phase B — data generation (desktop 3090, 1–2 overnights, no engineering).*
~20 exported spot configs (BTNvBB SRP, BBvUTG limped, 3-bet pots, lab-line
exports at several SPRs) × ~100-flop subset each ≈ 2,000 solves ≈ 7–8 GPU-h.
Pilot first: ~50 solves on CPU to validate the pipeline end to end.

*Phase C — fitting (~1 session).* Small weighted least-squares fit (no ML
stack needed; a ~200-line Rust or Python script): predict `r_obs` from
`(pos_frac, spr_bucket[≈6], class features: pair?, suited?, gap,
high_rank)` with reach×pot weights. Deliverable: a small table file, e.g.
`cache/realization_fit.json`. Sanity: R rises with pos_frac, spreads with
SPR, suited/connected > offsuit-junk at equal equity.

*Phase D — integration + validation (~1 session).*
- `realization: "calibrated"` in `PreflopConfig`: load the fit at solver
  construction (fall back to `"static"` with a warning if the file is
  missing). NOTE: R becomes class-dependent → `terminal_value()` must apply
  R per class h, not per seat only (today `nd.r[p]` is a per-seat scalar
  computed at build time; move the class-dependent part into the h-loop or
  precompute a per-seat 169-vector at build).
- Multiway stays heuristic (postflop engine is HU-only, so 3+-way R is
  unmeasurable): keep the positional shape, state the limitation in the UI.
- Expose raw/static/calibrated as a dropdown in the lab config panel
  (`web/js/preflop_lab.js` `config()` currently hardcodes `'static'`).
- Validation: HU push/fold anchors must not move (all-in terminals bypass
  R); re-run the BB-defend-vs-2.5x threshold example and report before/after
  vs GTO Wizard's BB defend range; add a regression test asserting the
  calibrated table's monotonicities.
- Optional round 2: re-export lab spots under calibrated R and refit once
  (fixed-point loop; expected to settle immediately — R shifts thresholds
  by ~1–2 equity points).

**Effort.** ~2–3 sessions of engineering + overnight desktop compute.

---

## 2. Preflop player profiles — model & exploit real opponents (NEXT UP)

**Goal (designed 2026-07-04 with Matthew; his examples are the spec).** Model
reads like "VPIPs every hand", "OMC who only raises AA/KK at the max open
size", "never 3-bets" as per-SEAT behavioral profiles in the Preflop Lab,
lock seats to them, and exploit: re-solve so the table adapts, or freeze all
non-hero seats and solve hero's seat = a personal max-exploitation chart.

**Core design.**
- Locks are per SEAT, not per node: a profile compiles into behavior at every
  node where that seat acts. Situation buckets (stored per node at build,
  1 byte): unopened / vs limp(s) / vs raise / vs raise+caller(s) (squeeze —
  explicitly requested) / vs 3-bet+.
- Seat modes in the engine: live (normal CFR) | frozen (plays current average
  strategy, no updates — seat-level "lock as solved") | ruled (strategy from
  profile). Frozen/ruled need NO lock tables: traversal sources sigma
  differently and skips that seat's regret/strat updates (zero memory).
  Precedence later: node point-locks > profile > solver.
- Profiles are STAT-DRIVEN (HUD vocabulary): VPIP, PFR, 3bet%, fold-to-3bet%,
  squeeze%, limp-behind. Archetype presets as stat vectors: Whale 60/8/2,
  Nit/OMC 12/8/1, TAG 24/19/7, LAG 30/25/11, Maniac 55/40/20, Station.
- Range generation = EQUILIBRIUM DISTORTION (Matthew's explicit pick over a
  static ranking): baseline = the current UNLOCKED solve's average strategy.
  Per seat and bucket, summarize per-class propensities (reach-weighted over
  the bucket's nodes): continue prob c_h, raise prob r_h. Generate a
  VPIP-X profile by ranking classes by c_h (tie-break EV) and filling
  cumulative combo mass to X% (boundary class gets fractional weight);
  within the continuing range, fill the raising slice to the PFR-analog by
  r_h rank; VPIP−PFR gap = the limp/call slice (this alone produces
  passive vs aggressive shapes). Same construction per bucket from its own
  stats (3bet%, fold-to-3bet → survive-share of the opening range, etc.).
  Raise-range carries a SIZE CHOICE (min/max/jam; OMC preset = max — per
  Matthew). Positional shape is inherited (each seat distorts its own
  equilibrium); add a "position blind" 0..1 dial that interpolates toward
  the seat-average range for true fish. CONSEQUENCE: generation requires a
  baseline solve — UI prompts to solve unlocked first (natural workflow:
  build → solve → profile seats → re-solve/exploit).
- Implied-stat readback after generate/edit ("this profile ≈ 58/6/1, folds
  to 3-bet 22%") so the model can be checked against the HUD numbers.
- Free wins to preserve: ribbon chips show locked seats' real frequencies
  automatically (strategy source drives display); SEND TO POSTFLOP exports
  arriving ranges that reflect profiles (whale's wide flop range flows into
  postflop study); postflop EXPLOIT mode continues the pipeline.

**Phases.**
- P1 engine: seat modes, bucket tagging, profile→sigma compilation, tests
  with behavioral anchors (vs never-3-bettor hero opens widen; facing OMC
  raise hero folds QQ; vs whale hero's bluffs die and thin value grows).
- P2 generator: equilibrium-distortion synthesis (server-side Rust),
  archetypes, implied-stat readback; profiles stored as JSON in
  saves/profiles/ (travels via git, unlike localStorage). Interim rule
  editing via range text per bucket action.
- P3 UI: profile editor modal (stats row → GENERATE → bucket tabs →
  multi-action painting grid: palette fold/limp-call/raise@size/jam painted
  per class with mix weights — extend the SETUP RangeEditor), seat lock
  badges, ribbon lock icons, "N seats profiled" status note.
- P4 hero mode ("solve as SEAT: freeze everyone else" → CFR vs fixed
  opponents converges to hero's max exploit) + instant per-seat BR bleed
  readout ("this profile loses X bb/hand") via the existing mode-2 pass.
- P5: GPU support for seat modes (after preflop kernels are
  desktop-validated; until then profile solves fall back to CPU with a
  status note), node-level point-locks for spot-specific reads.

## 2b. Postflop player profiles (LATER — explicit Matthew request)

The same player model continued past the flop: postflop HUD stats (c-bet%,
fold-to-c-bet, WTSD, raise-c-bet...) auto-generate POSTFLOP node locks in a
spot exported from the lab — e.g. c-bet 80% → Range-lock villain's flop bet
frequency; fold-to-c-bet 60% → lock his facing-bet fold frequency; then
postflop EXPLOIT mode (already built) reads off the punishment. Design when
preflop profiles (item 2) are in use: the profile JSON should carry a
postflop-stats section from day one so one player file describes the whole
hand. Depends on: item 2's profile format; the postflop lock API
(POST /api/lock, Range mode) already suffices mechanically.

## 3. Raw/static realization toggle in the lab UI (tiny)

`web/js/preflop_lab.js` `config()` hardcodes `realization: 'static'`. Add a
select (static / raw — later calibrated, see item 1) so model sensitivity
can be A/B'd from the UI. If a decision survives both, the model isn't
deciding it. ~20 lines (config field, dropdown in `index.html` view-preflop
panel, els wiring in `app.js`).

## 4. Tier 2 — heads-up full-game preflop solver (desktop-class project)

True preflop solving for 2 players (SB vs BB): preflop street + flop chance
node fanning into a weighted canonical-flop subset (~50–95 boards), each
continuing into deliberately small postflop trees, solved as ONE game by the
existing 2-player DCFR engine (`cfr.rs`) — all convergence guarantees hold.
This is PioSolver's preflop module rebuilt on freepio. Blind-vs-blind limp
trees at any sizing, exact (no realization model). Big lift: the tree
builder (`tree.rs`) assumes a fixed board; needs a preflop street + flop
chance layer, and memory planning (each flop subtree ≈ a current spot;
×95 boards → needs small per-street size menus + the 128 GB desktop).
Validate against item 1's data and published HU charts.

## 5. UI consolidation — Browse as the only screen (design agreed 2026-07-02)

SETUP → GTO Wizard-style modal (tabs: New spot / Library of saves via
`/api/saves`); SOLVE → header strip + collapsible convergence drawer
(header solve buttons already exist); tabs removed. Pure frontend. Phase 2:
merge the preflop study panel and PREFLOP LAB ribbon into Browse's ribbon.

## 6. Smaller items

- **Preflop CUDA: VALIDATE ON THE DESKTOP** — implemented 2026-07-04
  (`preflop/gpu.rs` + `preflop/kernels.cu`: level-synchronous CFR
  mirroring the CPU exactly; server falls back to CPU + system RAM when
  the game exceeds free VRAM or CUDA errors, mid-solve included) but
  written on a GPU-less laptop: the kernels have NEVER RUN. On the 3090
  box, before trusting GPU output, run
  `cargo test --release --features gpu --test preflop_gpu -- --test-threads=1`
  (CPU-vs-GPU strategy equivalence + push/fold anchors). NVRTC compiles
  kernels at runtime, so kernel syntax errors surface there as a
  "GPU unavailable" fallback note with the compiler message.
- **Multiway all-in equity refinement**: product approximation is slightly
  pessimistic 3+-way; for POT_SHARE terminals with everyone all-in, an
  on-demand Monte-Carlo 3-way table (cached like the pairwise one in
  `preflop/equity.rs`) would make jam-heavy multiway trees near-exact.
- **EV heatmap mode for the lab grid**: per-class EVs fall out of
  `PreflopSolver::traverse(mode=1)`; wire into `paintGrid` like Browse's EV
  mode.
- **EXPLOIT mode hands panel**: the per-combo side panel in Browse still
  shows current-strategy data in EXPLOIT mode; feed it the
  `/api/exploit` payload instead.
- **Unequal stacks in the Preflop Lab**: `PreflopConfig.stack` is a single
  value (no side pots by design). Support per-seat stacks + side-pot-aware
  terminals if short-stack study becomes interesting.
- **GTO Wizard range import**: paste-parse GTOW's copy format into the
  range editor (mostly compatible with the existing text syntax already).
