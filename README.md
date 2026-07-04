# FREEPIO — a PioSolver-style GTO poker solver

A from-scratch heads-up no-limit hold'em postflop solver with a local web UI.
Rust solver core (discounted CFR), optional CUDA GPU engine, zero-install
browser frontend.

![solver tests](https://img.shields.io/badge/tests-32%20passing-success)

![Browse — mid-hand on the turn](docs/browse-midhand.png)
*Mid-hand in BROWSE: bet–call on K♠7♥2♦ to the Q♥ turn — action ribbon,
strategy matrix, equity curves, node locking, and the EXPLOIT mode toggle.*

## Quick start

```bash
./start.sh          # builds (release) and serves on :3737
# open http://127.0.0.1:3737
```

`start.sh` detects the machine: with an NVIDIA CUDA runtime present it sets
up the library path and builds the GPU engine; without one it builds
CPU-only automatically (same UI, just slower). Manual equivalent:

```bash
cargo build --release        # no GPU feature
./target/release/gto-server
```

Run from the repo root (the server serves the `web/` directory). Solves and
saves land in `./saves/`.

### GPU engine (optional, ~10× faster)

The CUDA backend needs an NVIDIA GPU and the `nvrtc` runtime compiler. The
easiest way to get `nvrtc` without a full CUDA toolkit:

```bash
pip install --target ~/.local/cuda-nvrtc nvidia-cuda-nvrtc-cu12
cargo build --release -p server --features gpu
./start.sh          # finds nvrtc and enables the GPU automatically
```

Without `LD_LIBRARY_PATH` pointing at `nvrtc` the server silently falls back
to CPU — `start.sh` handles this for you. The status JSON reports `"gpu": true`
when the engine is live.

Optional environment:

| var | default | meaning |
|---|---|---|
| `PORT` | 3737 | HTTP port |
| `SOLVER_THREADS` | physical cores | rayon worker threads (SMT hurts this workload; 12–16 is the sweet spot on a 5950X) |
| `SOLVER_COMPRESS` | 1 | `0` = full-precision f32 arenas instead of 16-bit compressed |
| `SOLVER_GPU` | 1 (if built with `gpu`) | `0` forces the CPU engine even when CUDA is available |
| `SOLVER_GPU_MEM_MB` | live free VRAM | manual VRAM cap for the GPU engine; spots over budget fall back to CPU |
| `SOLVER_MEM_MB` | 80% of free RAM (≤48 GB) | solver-arena RAM cap; bigger trees are refused at BUILD |
| `PREFLOP_EQ_SAMPLES` | 20000 | Monte-Carlo samples per hand-class pair for the Preflop Lab equity table |
| `PREFLOP_MAX_NODES` | RAM-derived | Preflop Lab tree-size limit; the lab shows a live estimate + this machine's caps before BUILD |
| `PREFLOP_MAX_ARENA_MB` | ~40% of free RAM | Preflop Lab regret/strategy memory limit (MB) |

## Workflow (mirrors PioSolver)

0. **PREFLOP LAB** (optional) — solve the preflop game first (limps, any
   sizes, 2–9 players; see below), walk the line you care about, and **SEND
   TO POSTFLOP**: both conditional ranges, pot and stacks land in SETUP, the
   spot is saved as a reusable preset, and the preflop line shows in Browse's
   ribbon ahead of the flop.
1. **SETUP** — edit both ranges on the 13×13 grid (drag-paint, weight brush,
   text syntax `AA,AKs,KQs:0.5,A5s-A2s,99-66,AhKh:0.25`, presets), pick the
   board (3 cards = flop solve, 4 = turn, 5 = river), set pot/stacks/rake and
   per-street bet/raise/donk sizes (`33 75`, `a` = all-in, `2.5x` = raise
   multiple), then **BUILD TREE**. The build reports node count and exact
   solver memory before you commit to solving.
2. **SOLVE** — set a target exploitability (% of pot; 0.3% is a typical
   "Pio-quality" target), watch the live convergence chart. Stop/resume any
   time; save/load full solves to disk.
3. **BROWSE** — walk any line with breadcrumbs. The matrix shows strategy
   stacked-bars (blue fold / green check-call / amber→red bets by size),
   EV heatmap, or equity heatmap for either player. Click a cell for the
   per-combo breakdown (strategy, reach, EQ, EV per action). At card nodes,
   pick the turn/river from the deck or open the **RUNOUTS REPORT** (strategy
   + equity for every possible next card).
4. **NODE LOCKING** — at any decision node, set aggregate action frequencies
   or exact per-hand frequencies and lock. Re-run SOLVE and the rest of the
   tree re-optimizes around your assumption ("what if villain never raises
   here?"). Verified by test: locking the bluff-catcher to always-call drives
   bluffing to zero.
5. **EXPLOIT** — the fourth matrix mode: a true best response for either
   player against the opponent's CURRENT strategy, locks included. Lock
   villain to a pool tendency (station, never-bluffs, over-folds), flip to
   EXPLOIT, and read off the max-exploit strategy plus each hand's EV gain
   over the equilibrium line — no re-solve needed. Verified by test: vs a
   locked always-caller the best response bets the nuts and never bluffs.

## Engine

- **Algorithm**: Discounted CFR (α=1.5, β=0, γ=2) with alternating updates,
  vectorized over hands. CFR+ and Predictive CFR+ (PCFR+) are selectable per
  solve (`POST /api/solve {"algorithm": "pcfr+"}` or `SOLVER_ALGO` in the CLI).
- **Compressed arenas (PioSolver-style)**: regrets stored as i16 and strategy
  sums as u16, quantized per node against the node's max magnitude — half the
  memory of f32, and faster on big trees because CFR is memory-bandwidth
  bound. Verified equivalent to f32 within 0.1% pot exploitability by test;
  save files stay full-precision f32 and load into either mode.
- **Zero-allocation traversal**: all per-node-visit scratch vectors come from
  a thread-local buffer pool instead of malloc.
- **CUDA GPU solver**: level-synchronous batched CFR with regrets/strategy
  resident in VRAM (`--features gpu`). **~9.6x a 16-core 5950X** per iteration
  on an RTX 3090 (1.25M-node tree: 62 ms/iter vs 594 ms on rainbow boards;
  suit isomorphism works on the GPU too — 37 ms/iter two-tone, 18 ms/iter
  monotone, with orbit-aware node locks). Exploitability (true best
  response, rake-aware) also runs on the GPU (~50 ms per check vs ~1.5 s on
  CPU), so progress checks are nearly free. Node locking works (mid-solve
  too); CPU storage may be f32 or i16-compressed (arenas decode on upload,
  re-encode on sync); mid-solve browse data refreshes every few checks and
  fully at stop/finish. DCFR/CFR+ only. The server uses the GPU
  automatically when built with the feature (`SOLVER_GPU=0` opts out,
  `SOLVER_GPU_MEM_MB` caps VRAM, default 20000; falls back to CPU if the spot
  doesn't fit or CUDA is unavailable). Needs `libcuda` (WSL provides it) and
  `libnvrtc` on `LD_LIBRARY_PATH` (the `nvidia-cuda-nvrtc-cu12` pip wheel
  works — no full toolkit needed). GPU tests:
  `cargo test --release --features gpu --test gpu -- --test-threads=1`.
- **Terminal evaluation**: O(n) sorted showdown sweep with exact card-removal
  (blocker) accounting via per-card prefix sums; precomputed 7-card strengths
  for every river runout.
- **Exploitability**: true best-response traversal both ways, reported as % of
  pot — the number shown is the real distance from Nash, not a proxy.
- **Parallelism**: rayon across chance branches; arenas use disjoint unsafe
  slices (each tree node's data is touched by exactly one branch).
- **Suit isomorphism**: chance branches that are suit-symmetric (given the
  board and ranges) are solved once and mirrored exactly — ~1.4x on two-tone
  flops, ~2.2x on monotone, no effect on rainbow. Exact (verified against the
  non-isomorphic solver). Zero-reach subtrees are also pruned exactly.
- **Trees**: per-street/per-player bet, raise and (OOP) donk sizes, all-in
  threshold conversion, raise caps, NL min-raise rules, rake (% + cap),
  flop/turn/river root. EVs use Pio's pot-share convention (EV OOP + EV IP =
  pot).
- **Accuracy tests** (`cargo test -p solver --release`): hand evaluator vs an
  independent reference on 20k random deals; range parser round-trips; the
  clairvoyance game converges to its known closed-form solution (bet-ratio
  1/3 bluffs, MDF 50% calls, EVs exact); equity matches brute-force
  enumeration; full flop trees reach <1% pot exploitability; node-lock
  semantics; save/load roundtrip (both storage modes); compressed-vs-f32
  exploitability equivalence; PCFR+ convergence; exploit-view best response
  vs a locked station (bets the nuts, never bluffs, BR EV dominates);
  per-hand locks land on the right combos across suit-isomorphic runouts;
  preflop CFR vs an independent fictitious-play Nash oracle on HU jam/fold,
  plus multiway limp-tree chip conservation and rake-drain direction.

## Performance

Reference spot: 100bb single-raised pot, ~260 vs ~320 combos, two bet sizes +
raises everywhere (1.25M nodes).

- **GPU (RTX 3090)**: ~60 ms/iteration, GPU best-response checks ~50 ms.
  A 1.76M-node flop solve reaches 0.3% pot in ~10 s. Batch mode clears a
  canonical flop set in under an hour.
- **CPU (Ryzen 5950X, 16 threads)**: ~0.6 s/iteration, 0.3% pot in roughly
  3–6 minutes. Simple one-size trees solve in seconds.

Memory is reported pre-solve; the server refuses trees over its RAM budget
(80% of currently available memory, never above 48 GB — `SOLVER_MEM_MB`
overrides), so a laptop rejects a workstation-sized spot instead of
thrashing. Compressed arenas roughly double what fits. SMT hurts this
workload, so the CPU engine defaults to physical cores.

## CLI

```bash
./target/release/solve-cli spot.json [max_iterations] [target_exploit_pct]
./target/release/solve-cli batch spot.json boards.txt [max_iterations] [target]
```

`spot.json` matches the `SpotConfig` JSON schema (see `bench_spot.json`).
Env: `SOLVER_STORAGE=f32|i16` (default i16), `SOLVER_ALGO=dcfr|cfr+|pcfr+`
(default dcfr), `SOLVER_ISO=0`, `SOLVER_THREADS=N`, `SOLVER_GPU=1`
(gpu-feature builds).

**Batch mode** solves the same ranges/tree across many boards (file with one
board per line, or an inline `b1,b2,..` list), prints one row per board
(iterations, exploitability, reach-weighted root EVs) and writes
`batch_results.json` — the raw material for multi-flop aggregate analysis.
On the GPU a 1.25M-node flop solves to 0.4% pot in ~12-15 s, so a full
canonical flop set is an under-an-hour job. `SOLVER_BATCH_SAVE=1` also
writes `saves/batch_<board>.gto` per board.

## API

Everything the UI does is plain JSON over HTTP — scriptable:

`POST /api/spot`, `POST /api/solve`, `POST /api/stop`, `GET /api/status`,
`POST /api/node {path}`, `POST /api/exploit {path, exploiter}` (per-hand best
response + EV gain vs the current strategy), `POST /api/lock {path, mode}`,
`POST /api/unlock`, `GET /api/locks`, `POST /api/runouts {path}`,
`POST /api/range/parse`, `POST /api/save|load {name}`, `GET /api/saves`,
`GET /api/presets`.

Preflop lab: `POST /api/preflop/spot {config}`, `POST /api/preflop/solve`,
`POST /api/preflop/stop`, `GET /api/preflop/status`,
`POST /api/preflop/node {path}` (path = action indices),
`POST /api/preflop/export {path}` (heads-up flop node → postflop spot inputs).

Path steps: `{"type":"action","index":0}` / `{"type":"card","card":"Ah"}`.
Browse deep links: `/#line=a1,a1,cQh` opens BROWSE at that node (`a<i>` =
action index, `c<card>` = dealt card).

## Preflop Lab (multiway preflop over an equity model)

The **00 · PREFLOP LAB** tab solves N-player (2–9) preflop trees exactly at
the action level — **limps, cold calls, any raise sizes, antes, rake** — the
spots GTO Wizard's fixed libraries can't express. Postflop play is priced by
a model instead of solved: at flop terminals each live player's share is
`pot × multiway-equity × R`, with R a pluggable realization factor ("raw" or
positional-vs-SPR "static"); all-in terminals are model-exact. Hands are the
169 canonical classes over a Monte-Carlo pairwise equity table (built once,
disk-cached, `PREFLOP_EQ_SAMPLES` env to tune); multiway equity uses the
product approximation (exact heads-up). Convergence is reported as per-player
best-response gaps in bb — for 3+ players CFR gives *an* equilibrium of the
model, not a unique GTO answer.

Validated: CFR reproduces an independent fictitious-play Nash oracle on
heads-up jam/fold, and hits the published 10bb push/fold ranges (SB jams
~58%, BB calls ~37%). Walk any line in the ribbon; at a heads-up flop node,
**SEND TO POSTFLOP** exports both conditional ranges + pot/stack straight
into SETUP for an exact postflop solve.

**Player profiles** model real opponents: give any seat HUD-style stats
(VPIP/PFR/3-bet/fold-to-3-bet/squeeze) or an archetype (Whale, Nit/OMC,
Station, TAG, LAG, Maniac) and its ranges are generated by **distorting the
solve's own equilibrium** — then refined on a multi-action painting grid,
saved to `saves/profiles/`, and locked in. RE-SOLVE adapts the table around
the reads; **HERO mode** freezes everyone else so your seat's re-solve
converges to a maximum-exploitation strategy, with per-seat "bleeds X bb"
readouts. Situation buckets: unopened / vs limps / vs raise / squeeze /
vs 3-bet+. Verified by test: an AA/KK-only OMC's raises get QQ folds (never
AA), a never-folding whale bleeds 2.5+ bb and flips the exploiter's EV
positive, frozen seats stay exactly put.

The lab is multi-core on the CPU (subtree-parallel CFR + player-parallel
accuracy checks) and, when built with `--features gpu`, solves on the GPU:
a level-synchronous CUDA engine mirroring the CPU math, with automatic
fallback to CPU + system RAM when the game exceeds free VRAM or CUDA
errors. First time on a GPU machine, validate the kernels:
`cargo test --release --features gpu --test preflop_gpu -- --test-threads=1`.

## Differences from PioSolver (known gaps)

- No multi-flop aggregated reports (single-board runouts reports only; batch
  solving is possible via the CLI/API).
- Preflop is solved against an equity-realization model (see Preflop Lab),
  not full-game trees; postflop solving is heads-up only. No ICM.
- Saves are not .cfr-compatible (own format).

## Layout

```
crates/solver           — engine: cards, evaluator, ranges, tree, CFR, BR, queries
crates/solver/src/preflop — multiway preflop solver + 169-class equity table
crates/server           — axum HTTP server + static hosting
web/                    — vanilla-JS frontend (no build step)
cache/                  — preflop equity table (deterministic, regenerable)
```
