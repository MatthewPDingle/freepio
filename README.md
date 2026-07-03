# FREEPIO — a PioSolver-style GTO poker solver

A from-scratch heads-up no-limit hold'em postflop solver with a local web UI.
Rust solver core (discounted CFR), optional CUDA GPU engine, zero-install
browser frontend.

![solver tests](https://img.shields.io/badge/tests-31%20passing-success)

## Quick start

```bash
./start.sh          # builds (release, GPU enabled) and serves on :3737
# open http://127.0.0.1:3737
```

`start.sh` sets up the CUDA runtime path so the GPU engine engages, building
on first run. CPU-only is just as good a starting point:

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
| `SOLVER_GPU_MEM_MB` | 20000 | VRAM budget for the GPU engine; spots over this fall back to CPU |

## Workflow (mirrors PioSolver)

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
  exploitability equivalence; PCFR+ convergence.

## Performance

Reference spot: 100bb single-raised pot, ~260 vs ~320 combos, two bet sizes +
raises everywhere (1.25M nodes).

- **GPU (RTX 3090)**: ~60 ms/iteration, GPU best-response checks ~50 ms.
  A 1.76M-node flop solve reaches 0.3% pot in ~10 s. Batch mode clears a
  canonical flop set in under an hour.
- **CPU (Ryzen 5950X, 16 threads)**: ~0.6 s/iteration, 0.3% pot in roughly
  3–6 minutes. Simple one-size trees solve in seconds.

Memory is reported pre-solve; the server refuses configurations above 48 GB
(compressed arenas roughly double what fits). SMT hurts this workload, so the
CPU engine defaults to physical cores.

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

Path steps: `{"type":"action","index":0}` / `{"type":"card","card":"Ah"}`.

## Differences from PioSolver (known gaps)

- No multi-flop aggregated reports (single-board runouts reports only; batch
  solving is possible via the CLI/API).
- No preflop solving, ICM, or multiway (same as PioSolver edge/core).
- Saves are not .cfr-compatible (own format).

## Layout

```
crates/solver   — engine: cards, evaluator, ranges, tree, CFR, BR, queries
crates/server   — axum HTTP server + static hosting
web/            — vanilla-JS frontend (no build step)
```
