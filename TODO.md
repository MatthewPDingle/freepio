# TODO

Work queue for GTOpen. Items are written so a fresh contributor (human or
Claude) can execute them without prior context. Read `README.md` first for
the architecture; run `cargo test --release -p solver` (all green, ~5 min on
a laptop) before and after any change.

---

## 1. M5 — Calibrated equity-realization model — **DONE 2026-07-08** 🎉

All four phases complete; `realization: "calibrated"` ships as the lab
default (dropdown: calibrated / positional / raw). Engine model:
**measured per-class realization** (169 reach-weighted, count-shrunk bases
from 91k observations of the engine's own postflop solves) × the mild
positional weight, clipped, at HU flop terminals with chips behind;
multiway/all-in terminals unchanged. Validated on a raked HU game: SB
becomes raise-or-fold (67%, no limps — modern HU theory), BB defends 50%
vs 2.5x with textbook composition (vs static's junk-loving 94%).

**Modeling postmortems worth remembering (git log has details):**
- v1 linear class features (hi/lo/gap) extrapolated catastrophically at
  preflop boundaries (folded all offsuit aces, defended all suited junk).
- v2 per-class offsets + shared equity terms: the +4 eq² implied-odds
  bonus mis-ranked mid-equity vs low-equity classes.
- v3 CAUSAL TRAP: feeding the measured initiative premium (+0.68) back
  into optimization let the solver BUY the aggressor multiplier — 100%
  open rates. Initiative/range-equity are equilibrium correlates, not
  levers; they remain in the fit output for analysis, not in the engine.
  (Applies to M6 too.)
- Remaining refinement (round 2 in the original plan): fixed-point refit
  under calibrated exports, and per-context tables once multi-context
  class coverage improves. Original design below for reference.

### (original M5 design — for reference)

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

*Phase B — data generation: DONE 2026-07-08 (cloud H100s, ~$65 all-in
including one invalidated run — percent/fraction rake postmortem in git
log 538d1c4).* 116,214 class-observations / 2,080 solved boards / 24 spots
across all six game structures (three single-size $2/2 variants, both $2/5
depths, NL10 6-max), median solve quality 0.278% pot, SPR 3–74. Validated:
position premium (IP 0.98 vs OOP 0.59), suited over offsuit at matched
equity (0.83 vs 0.61). Dropped with cause: vs-10x BB defend (the calling
range barely exists under an 11bb-cap rake — itself a finding) and the two
SPR-40+ limped spots (500s/flop to inform a region the model clamps to
SPR 8). Data: m5_spots/data/phase_b_2026-07-08.tgz; driver:
m5_spots/phase_b.py (fully scripted — reruns are one command on any
rented GPU).

*Phase C — fitting: DONE 2026-07-08.* `m5_spots/fit_phase_c.py` (weighted
least squares, reach×pot weights, board-level holdout) →
`cache/realization_fit.json`. 91,172 obs / 23 spots (mini-menu smoke rows
excluded — R is menu-conditional). Weighted R² 0.385 train / 0.404
holdout. Features (ALL evaluable at a lab terminal): pos_frac, SPR bucket
(+pos interaction), pair/suited/gap/hi/lo, own-class eq (+square), range
mean eq, initiative (±0.5, 0 for limped). **HEADLINE FINDING: initiative,
not position, drives aggregate realization** — raw mean r_obs 1.25 with
initiative vs 0.42 facing it vs 0.60 limped; controlled, pure position is
a small negative residual ("IP over-realizes" was mostly "the aggressor
does, and he's usually IP"). Also: value hands over-realize convexly in
equity (eq² +4.14 — set-miners AND nut hands both beat pot×eq), range
advantage over-realizes (+2.2/eq-point), suited +0.20, gap −0.23.
Sanity gates (7) all pass and are enforced by the script — it refuses to
write a table that violates them.

*Phase C v4 refit — DONE 2026-07-09* (the "calibrated hates A6s–ATs"
postmortem). v3 class bases were in-context means, so they encoded each
class's ROLE MIX across the 23 training spots; the per-role split showed
the real structure is a class×role interaction (ATs: 1.17 with
initiative / 0.39 facing — kicker domination; A5s: 1.28 / 0.77 —
undominated wheel outs), and classes with different mixes weren't
comparable (ATs 67% defender rows vs A5s 39% → bases 0.64 vs 1.07, a
role-inflated 1.67× premium). v4 = per-class role standardization at
the structural reference mix (facing = init = (1−limp)/2), cells shrunk
λ=15 toward IPF class×role priors, blended by α×role-coverage (single-
role classes keep in-context values — 32s "with initiative" is fiction),
equity-anchored curve (ridge over strength/suited/pair/straight-windows/
high-card) as the shrinkage target for thin/unobserved classes, and
weighted-PAVA domination chains (broadway aces, K/Q kickers,
suited≥offsuit; wheel aces deliberately unchained). 11 gates enforced.
Validated on the 4-max 150bb 7.5x 10%/11 complaint game: JJ opens 99.9%
(v3 folded it), broadway-ace gradient AQs 62/AJs 47/ATs 41/A9s 11, K5s
no longer out-opens AJs (2.5%), wheel premium bounded at 1.44×.
KNOWN RESIDUAL: pot-type mix within role (single-raised vs 3-bet pots)
still biases classes that ate 3-bet-pot defends (99 base 0.73 < 66's
0.94 → 99 limps where 66 opens). Fix = add the pot-type axis to the
standardization; needs Phase B round 2 with more 3-bet-pot spots for
cell support.

*Phase D — integration + validation (~1 session).*
- `realization: "calibrated"` in `PreflopConfig`: load the fit at solver
  construction (fall back to `"static"` with a warning if the file is
  missing). NOTE: R becomes class-dependent → `terminal_value()` must apply
  R per class h, not per seat only (today `nd.r[p]` is a per-seat scalar
  computed at build time; move the class-dependent part into the h-loop or
  precompute a per-seat 169-vector at build).
- RAKE DOUBLE-COUNT WARNING: r_obs was measured as (net-of-rake EV) /
  (GROSS pot × eq), so calibrated R already embeds the postflop rake
  drain — richer than the lab's crude min(pot×pct, cap) deduction. At
  calibrated terminals use share = pot_GROSS × eq × R and skip the
  separate rake deduction, or rake gets charged twice.
- Terminal feature values: eq per class is already computed there; range
  mean eq = reach-weighted mean of it; initiative = last preflop
  aggressor (trackable in the builder like BuildState.last_raise);
  pos_frac exists. Evaluate the dot product per (seat, class) and clip
  to the table's [0.2, 2.5].
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

## 3. Realization toggle in the lab UI — DONE 2026-07-08

Shipped with M5 Phase D: "Postflop model" dropdown (calibrated default /
positional / raw) in the lab scenario panel, carried through config,
saved games, and loads. If a decision survives all three models, the
model isn't deciding it.

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

## 7. Flop reports — DONE 2026-07-09 (phases 1-3 shipped)

Phase 1 (turn/river RUNOUTS REPORT: EV/EQR metrics, texture filters,
aggregate row) shipped 2026-07-05. Phases 2-3 shipped 2026-07-08
(a8807fd): background batch runner (`/api/reports/*`, partial saves
every 8 flops, GPU path for fresh solves), REPORTS tab (strip chart +
hitmap hover, OOP-root/IP-vs-check toggle, texture filters,
iso-weighted aggregate, sortable table, OPEN IN BROWSE re-solve), and
vs-villain reports via per-flop `lock_profile` + hero re-adapt — the
"where does the whale bleed by texture" report no commercial tool has.
Starter library generated on cloud A40s 2026-07-09 (saves/reports/,
gitignored): "2-5 150bb BB defends vs UTG 3x" (95 flops), "NL10 BB
defends vs CO 2.5x" (95), "NL10 BB is a WHALE vs CO 2.5x" (47 flops —
the locked-villain hero re-adapt is CPU-bound; regenerate at 95 on the
desktop or a high-vCPU pod when wanted). M5's realization extraction
shares this pipeline as designed. Ops gotchas live in item 8 ("Report
solver: log GPU fallback") and the memory file (SOLVER_THREADS).

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
- **Report solver: log (and pre-check) GPU fallback**: `report_solve`'s
  `if let Ok(gpu) = GpuSolver::new(..)` silently drops to CPU on init
  failure (e.g. staging+arenas > VRAM — the NL10 trees need 24-25 GB and
  wedged a 24 GB 4090 for 105 min/flop on 4 vCPUs). Log the fallback
  with the reason, and estimate staging vs free VRAM up front so a
  report run can fail fast or warn instead of silently crawling.
- **Size-sensitive fold-vs-bet**: postflop profiles use ONE fold-vs-bet
  number per street regardless of the size faced, but real players fold
  more to pot-sized bets than to 33% stabs. Add per-size anchors (e.g.
  fold-vs-33% / fold-vs-100%, interpolated by the actual size at each
  node) in `lock_profile`'s target construction. Same family as the
  preflop fold-to-4bet split — build either when a real opponent's reads
  demand it. PREFLOP HAS THE SAME BLINDNESS: the VS_3BET bucket target
  is a flat `1 − fold-to-3bet%` continue mass (mod.rs `targets[VS_3BET]`)
  regardless of re-raise SIZE (min-3-bet vs pot-sized) or DEPTH (3-bet /
  4-bet / 5-bet / jam all share it; fold-vs-jam is really much higher for
  most players), and unlike VS_RAISE/SQUEEZE it skips `gated_cont`'s
  equilibrium/naiveté blending entirely. Known artifact: hero mode with
  multiple re-raise sizes will spam the SMALLEST size since fold equity
  is size-invariant. Fix = same per-size/per-depth anchor scheme.
