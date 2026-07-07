# TODO

Work queue for GTOpen. Items are written so a fresh contributor (human or
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

*Phase A — observation extraction: DONE 2026-07-07.*
`solve-cli realization <spot.json> <boards> [iters] [target] [out.jsonl]`
solves each board and appends JSONL: a header (full config — R is
conditional on the bet-size menu), one meta line per board (iterations,
exploit%, so the fit can filter on solve quality), then per-class rows
`{board, player, pos_frac, spr, n_players, class, label, reach, eq, ev,
r_obs}` (reach-weighted combo→class aggregation; classes under 1% of the
busiest class or under 2% equity dropped). `solve-cli flops <n|all>` emits
the 1755 canonical flop classes (weights sum 22,100 — verified) or a
deterministic weighted subset. Engine fn: `Solver::realization_observations`
(query.rs). Tests: enumeration count + symmetric-range invariant (identical
ranges ⇒ IP over-realizes; asymmetric ranges legitimately flip this — the
range-advantage side over-realizes, that's data not error). Pilot (laptop
CPU, small SRP spot): AA/set r_obs ≈ 2.3-2.5, dominated KQo 0.24 on A-high,
IP mean 1.09 vs OOP 0.69 — the spread M5 fits. Spot templates + workflow in
`m5_spots/README.md`.

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
  vs a commercial solver's published BB defend range; add a regression test asserting the
  calibrated table's monotonicities.
- DESIGN OPTION (research-backed, decide during Phase D): posture-based
  leaves. A single fixed leaf value is exploitable in principle (the
  opponent can shift posture past the leaf); Brown's depth-limited-solving
  result (arXiv:1805.08195) restores soundness by letting each player
  CHOOSE among k continuation postures at the leaf (fit-or-fold /
  aggressive / balanced), each with its own measured per-class value
  vector. Phase B's batch solves can be sliced to measure posture vectors
  at no extra compute — fold this in if the calibrated-R validation shows
  posture exploits.
- Optional round 2: re-export lab spots under calibrated R and refit once
  (fixed-point loop; expected to settle immediately — R shifts thresholds
  by ~1–2 equity points).

**Effort.** ~2–3 sessions of engineering + overnight desktop compute.

---

## 2. Preflop player profiles — SHIPPED except P5 (updated 2026-07-05)

P1–P4 are live and battle-tested (full design + phase history: git log
2026-07-04..05 and the commit messages on `preflop/mod.rs`). In brief:
seat modes live/frozen/ruled; five situation buckets (unopened / vs limps /
vs raise / squeeze / vs 3-BET+, where 3-bet+ covers EVERY re-raise depth);
stat-driven generation by equilibrium distortion (VPIP/PFR/3bet/
fold-to-3bet+/squeeze, raise size min/max/jam); archetypes (Whale 60/8/2,
Nit-OMC 12/1.5/1, Station, TAG, LAG, Maniac); inline sidebar editor with
per-bucket painting grid; hero max-exploit mode; per-seat bleed readouts;
profiles saved in saves/profiles/.

Generator refinements shipped 2026-07-05 (post-original-design — the
"position blind" dial in the old design is now **naiveté** and does more):
- naiveté blends hand ORDERING (equilibrium playability ↔ raw card appeal),
  not just seat-shape — fixes polarized whale defends (Q9o in, 53s out).
- Buckets the baseline never reaches (nobody open-limps at equilibrium)
  fall back to card-appeal ordering + human-default targets instead of
  ranking float noise (which filled ranges bottom-up).
- Seats with no data of their own in a bucket lean on the table average
  for TARGET SIZE but card appeal for ORDERING (borrowing the table's
  ordering imported BB-defense polarization); tightening ratios compare
  same-source numerator/denominator so a nit's defend-vs-raise is tighter
  than its VPIP, never wider.

**Remaining (P5, as originally scoped):**
- GPU kernels for seat modes — profile solves fall back to CPU with a
  status note. UNBLOCKED 2026-07-07 (base kernels validated on a cloud
  4090); write the forced-sigma/frozen kernels and validate on the next
  cloud session or the home 3090.
- Point-lock UI for spot-specific reads — engine + API
  (`/api/preflop/lock|unlock`, precedence point-lock > profile > solver)
  already exist and are tested; needs frontend only.

## 2b. Postflop player profiles — DONE (2026-07-05)

The same player continued past the flop, end to end:
- `query::PostflopStats` (c-bet/barrel per street, fold-vs-bet per street —
  applies at every raise depth, raise-vs-bet, donk/stab, bet-size pref) —
  carried in `SeatProfile.postflop` (serde-defaulted; legacy saves load).
- `Solver::lock_profile(villain, stats, aggressor)`: walks the
  orbit-representative tree with probability-weighted reaches, classifies
  every villain node (street / initiative / facing-bet) and rakes the
  SOLVED strategy to the targets (Range-lock IPF — his natural betting
  hands keep betting, never hand-blind). Manual point-locks take
  precedence; re-apply is idempotent; returns target-vs-achieved readback.
- API: POST/DELETE /api/profile_locks; archetypes ship postflop defaults.
- UI: POSTFLOP TENDENCIES in the lab profile editor (saved with the
  profile), exports carry each flop player's stats + the preflop
  aggressor, and Browse grows a "LOCK <seat> TO <name>" toggle → RE-SOLVE
  → EXPLOIT reads the punishment.
- Tests: tests/postflop_profile.rs (targets hit, manual lock survives,
  exploit EV grows vs a 65% folder, idempotence, clear) + HTTP E2E
  (achieved == target on every row of a 3-street spot).

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
This is classic full-game preflop solving built on GTOpen. Blind-vs-blind limp
trees at any sizing, exact (no realization model). Big lift: the tree
builder (`tree.rs`) assumes a fixed board; needs a preflop street + flop
chance layer, and memory planning (each flop subtree ≈ a current spot;
×95 boards → needs small per-street size menus + the 128 GB desktop).
Validate against item 1's data and published HU charts.

## 5. RNR robustness dial for exploitation (research-backed, high value)

Hero mode and postflop EXPLOIT both compute FULL best responses against the
modeled table — maximally profitable, maximally exploitable if the read is
off. Restricted Nash Response (Johanson/Zinkevich/Bowling 2007; see also
Ganzfried & Sandholm, Safe Opponent Exploitation) fixes this with one knob:
solve hero against an opponent who plays the PROFILE with probability p and
plays FREELY (adversarially, regret-updating) with probability 1-p. p=1 is
today's max exploit; p≈0.8 gives up a little EV vs the read but stays
near-unexploitable when the read is wrong. Implementation: mixture in the
sigma source for ruled/frozen seats (preflop engine) and for profile-locked
nodes (postflop: lock sigma = p·raked + (1-p)·live), plus a slider in the
hero panel / EXPLOIT bar labeled "trust the read". Report both numbers:
EV vs the profile AND own exploitability at each p.

## 6. M6 — range-conditional leaf values via a small value net (after M5)

The DeepStack/ReBeL/commercial-AI-solver architecture, scaled to a study
tool: train a small net on GTOpen's OWN batch solves (M5 Phase B data —
inputs: both 169-class reach vectors + pot + SPR + position; outputs:
per-class EV vectors at the flop root) and use it as the Preflop Lab's
terminal evaluator ("learned" realization mode). Strictly supervised — no
self-play loop needed for study purposes. 3090 is ample. Depends on M5's
data pipeline; supersedes calibrated-R when validated.

## 7. Flop reports (SHELVED until the desktop — needs batch compute)

Phase 1 (turn/river reports inside a solved tree) shipped 2026-07-05: the
RUNOUTS REPORT gained EV and EQR metrics, texture filters
(flush/pairing/straighty/overcard/brick) and an equal-weight aggregate
row. Remaining phases are batch-compute features:
- Phase 2: in-server batch runner solving the SAME spot across a weighted
  canonical flop set (default ~95-184 flops; all 1755 as an overnight
  option), progress via status polling, report JSON in saves/reports/,
  chart + sortable table + texture filters, click-a-flop -> re-solve into
  Browse (don't store 100+ full saves). Works against profile-locked
  villains for free -> "where does the whale bleed by texture", which no
  commercial tool offers.
- Phase 3: exploit-metric columns, report library/compare, and feed M5
  Phase A/B from the same runs (the per-flop extraction IS r_obs:
  EQR = EV / (pot x EQ) per class — one pipeline, two consumers).

## 8. Smaller items

- **Preflop CUDA: VALIDATED (2026-07-07, cloud RTX 4090)** — both engines
  pass on real hardware: postflop 6/6, preflop 2/2 (blind kernels' first
  execution; the equivalence test now uses principled criteria — see
  below). Bench: 6-max limp tree 166k nodes, 48.5 it/s on a 4090 vs 3.7
  on a 54-thread CPU (~13x; ~20x the laptop). NVRTC arch is now
  device-dynamic (dd765a6), so A100/H100 rentals work too. The home 3090
  needs no separate validation — same test suite, run it once for
  confidence. Original item follows for context: implemented 2026-07-04
  (`preflop/gpu.rs` + `preflop/kernels.cu`: level-synchronous CFR
  mirroring the CPU exactly; server falls back to CPU + system RAM when
  the game exceeds free VRAM or CUDA errors, mid-solve included) but
  written on a GPU-less laptop: the kernels have NEVER RUN. On the 3090
  box, before trusting GPU output, run
  `cargo test --release --features gpu --test preflop_gpu -- --test-threads=1`
  (CPU-vs-GPU strategy equivalence + push/fold anchors). NVRTC compiles
  kernels at runtime, so kernel syntax errors surface there as a
  "GPU unavailable" fallback note with the compiler message. This run
  also UNBLOCKS item 2's P5 (GPU seat-mode kernels): plan both for the
  same desktop session.
- **Blind-seat generation wart**: in buckets where checking is free (BB
  unopened / BB vs limps), "continue" = 1.0 for every class because there
  is no fold action, which inflates the baseline continue% used for
  tightening ratios when GENERATING a profile from a blind seat (fine
  from UTG, the common case). Fix idea: treat free-check nodes as
  no-signal (exclude from bucket summaries) rather than 100%-continue.
- **Multiway all-in equity refinement**: product approximation is slightly
  pessimistic 3+-way; for POT_SHARE terminals with everyone all-in, an
  on-demand Monte-Carlo 3-way table (cached like the pairwise one in
  `preflop/equity.rs`) would make jam-heavy multiway trees near-exact.
- **EV heatmap mode for the lab grid**: per-class EVs fall out of
  `PreflopSolver::traverse(mode=1)`; wire into `paintGrid` like Browse's EV
  mode.
- **EXPLOIT mode hands panel**: the per-combo side panel in Browse still
  shows current-strategy data in EXPLOIT mode (the matrix uses the
  `handsFor()` exploit overlay; `cellHandData()` — which feeds both the
  HANDS tab and the 2026-07-05 grid-hover popup — bypasses it). Route
  `cellHandData` through `handsFor` and both fix at once; check that
  `handIdx` combo→index mapping matches the exploit payload's hand order.
- **Unequal stacks in the Preflop Lab**: `PreflopConfig.stack` is a single
  value (no side pots by design). Support per-seat stacks + side-pot-aware
  terminals if short-stack study becomes interesting.
- **External range import**: paste-parse the copy formats of popular
  solvers/trainers into the range editor (mostly compatible with the
  existing text syntax already).
- **Safe re-solve gadget for SEND TO POSTFLOP**: exporting a subgame and
  solving it in isolation is "unsafe" subgame solving (fine for study);
  add an optional maxmargin/reach-style resolve (Libratus lineage) that
  constrains the flop solution so villain can't beat his preflop EV —
  a correctness toggle for the purists.
- **Action translation (pseudo-harmonic)**: Ganzfried & Sandholm's mapping
  for off-tree bet sizes — the enabler for a future "paste a real hand
  and analyze it" flow. Naive nearest-size rounding is exploitable.
- **MCCFR sampling option for huge multiway preflop trees**: external-
  sampling blueprint pass (Pluribus-style) if 9-max configs ever outgrow
  the vectorized full traversal + RAM caps.
- **Regret-based pruning for the POSTFLOP engine**: the preflop-style
  whole-branch skip (2026-07-05) rarely fires on 1326-hand vectors (some
  hand always keeps mass on an action); needs per-hand masking or sampled
  traversal to pay off. Measure before building.
- **Size-sensitive fold-vs-bet**: postflop profiles use ONE fold-vs-bet
  number per street regardless of the size faced, but real players fold
  more to pot-sized bets than to 33% stabs. Add per-size anchors (e.g.
  fold-vs-33% / fold-vs-100%, interpolated by the actual size at each
  node) in `lock_profile`'s target construction. Same family as the
  preflop fold-to-4bet split — build either when a real opponent's reads
  demand it.
