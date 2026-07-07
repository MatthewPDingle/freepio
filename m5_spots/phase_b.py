#!/usr/bin/env python3
"""M5 Phase B driver: preflop lab solves -> HU flop exports -> realization runs.

Runs against a local gto-server (started automatically if :3737 is quiet).
Stages:
  spots  solve each game config in the lab (GPU when built with it), walk
         the study lines, export HU flop spots into m5_out/spots/*.json
  run    solve-cli realization for every spot x the flop subset, one output
         file per spot (resumable: finished spots are skipped), then concat
         into m5_out/realization_obs.jsonl and write m5_out/DONE
  all    both

  --pilot   NL10 game only, 2 spots, first 10 flops, loose targets
  --mini    tiny 3-max game, 2 flops - local smoke test of the plumbing
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "m5_out")
API = "http://127.0.0.1:3737"

# ---------------------------------------------------------------- games ----

P8 = ["UTG", "UTG1", "MP", "HJ", "CO", "BTN", "SB", "BB"]
P6 = ["UTG", "HJ", "CO", "BTN", "SB", "BB"]
P3 = ["BTN", "SB", "BB"]

def pf(positions, stack, opens, mults, max_raises, limp, allin, rake_pct, rake_cap):
    posts = [0.0] * (len(positions) - 2) + [0.5, 1.0]
    return {
        "positions": positions, "stack": stack, "posts": posts, "ante": 0.0,
        "limp": limp, "open_raises": opens, "raise_mults": mults,
        "max_raises": max_raises, "add_allin": allin, "allin_threshold": 0.85,
        "rake_pct": rake_pct, "rake_cap": rake_cap,
        "no_flop_no_drop": True, "realization": "static",
    }

# line steps: F=fold, C=call, K=check, ("R", to)=open to amount, RM=min re-raise
F, C, K, RM = ("fold",), ("call",), ("check",), ("raise_min",)
def R(to):
    return ("raise", to)

def eight_max_lines(open_to, limped):
    lines = {
        "utg_open_btn_call": [R(open_to), F, F, F, F, C, F, F],
        "utg_open_bb_call":  [R(open_to), F, F, F, F, F, F, C],
        "co_open_bb_call":   [F, F, F, F, R(open_to), F, F, C],
        "utg_open_btn_3bet": [R(open_to), F, F, F, F, RM, F, F, C],
    }
    if limped:
        lines["sb_limp_bb_check"] = [F, F, F, F, F, F, C, K]
    return lines

GAMES = [
    ("g22_open5",  pf(P8, 150, [5.0],  [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(5.0, True)),
    ("g22_open75", pf(P8, 150, [7.5],  [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(7.5, False)),
    ("g22_open10", pf(P8, 150, [10.0], [2.0, 3.5], 2, True, True, 10.0, 11.0),
     eight_max_lines(10.0, False)),
    ("g25_150",    pf(P8, 150, [3.0, 4.0], [2.5, 4.0], 2, True, True, 10.0, 9.0),
     eight_max_lines(3.0, True)),
    ("g25_200",    pf(P8, 200, [3.0, 4.0], [2.5, 4.0], 2, True, True, 10.0, 5.0),
     eight_max_lines(3.0, False)),
    ("nl10_6max",  pf(P6, 100, [2.5], [3.0], 3, True, False, 5.0, 3.0), {
        "co_open_bb_call":  [F, F, R(2.5), F, F, C],
        "btn_open_bb_call": [F, F, F, R(2.5), F, C],
        "btn_open_sb_3bet": [F, F, F, R(2.5), RM, F, C],
        "sb_limp_bb_check": [F, F, F, F, C, K],
    }),
]

MINI = [("mini_3max", pf(P3, 40, [2.5], [3.0], 2, True, True, 0.0, 0.0), {
    "btn_open_bb_call": [R(2.5), F, C],
    "sb_limp_bb_check": [F, C, K],
})]

# Postflop menu per Matthew: 30/75 + all-in on the flop, single size after
# (raises 3x, raise cap 2). The full two-size-every-street menu explodes to
# 100GB+ arenas on wide-range NL10 spots (pilot 2026-07-07: 8.2M nodes) —
# calibration needs realistic play, not exhaustive sizing.
def street3075():
    return {"bet": [{"PotPct": 30}, {"PotPct": 75}], "raise": [{"PrevMult": 3.0}], "donk": []}

def street75():
    return {"bet": [{"PotPct": 75}], "raise": [{"PrevMult": 3.0}], "donk": []}

def spot_config(ex, mini=False):
    # mini: one small size, no all-in — plumbing test on laptop-class RAM
    if mini:
        mini_st = lambda: {"bet": [{"PotPct": 33}], "raise": [{"PrevMult": 3.0}], "donk": []}
        streets = [mini_st(), mini_st(), mini_st()]
    else:
        streets = [street3075(), street75(), street75()]
    return {
        "board": "AhKs2d",  # overwritten per board by solve-cli
        "range_oop": ex["range_oop"], "range_ip": ex["range_ip"],
        "tree": {
            "starting_pot": ex["pot_bb"], "effective_stack": ex["eff_stack_bb"],
            # exports carry PERCENT (10.0); TreeConfig wants a FRACTION (0.10)
            "rake_pct": ex.get("rake_pct", 0.0) / 100.0, "rake_cap": ex.get("rake_cap", 0.0),
            "oop": [dict(s) for s in streets],
            "ip": [dict(s) for s in streets],
            "allin_threshold": 0.85, "add_allin": not mini, "max_raises": 2,
        },
    }

# ------------------------------------------------------------- plumbing ----

def req(method, path, body=None, timeout=1800, retries=3):
    for attempt in range(retries):
        try:
            r = urllib.request.Request(
                API + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"}, method=method)
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"  api retry {path}: {exc}", flush=True)
            time.sleep(5)

def ensure_server():
    try:
        req("GET", "/api/preflop/status", timeout=5, retries=1)
        return None
    except Exception:
        pass
    binpath = os.path.join(ROOT, "target/release/gto-server")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen([binpath], cwd=ROOT, env=env,
                            stdout=open(os.path.join(OUT, "server.log"), "ab"),
                            stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(1)
        try:
            req("GET", "/api/preflop/status", timeout=5, retries=1)
            print("server up", flush=True)
            return proc
        except Exception:
            continue
    raise RuntimeError("gto-server did not come up")

def solve_lab(cfg, iters, check, target):
    req("POST", "/api/preflop/spot", cfg)
    req("POST", "/api/preflop/solve",
        {"iterations": iters, "check_every": check, "target_gap": target})
    t0 = time.time()
    while True:
        time.sleep(5)
        st = req("GET", "/api/preflop/status")
        if st["state"] != "running":
            return st, time.time() - t0

def find_action(actions, step):
    kind = step[0]
    if kind == "raise":
        want = step[1]
        for i, a in enumerate(actions):
            amt = a.get("to", a.get("amount", 0)) or 0
            if a["kind"] in ("raise", "jam") and abs(amt - want) < 0.01:
                return i
        raise RuntimeError(f"no raise-to-{want} in {[a['label'] for a in actions]}")
    if kind == "raise_min":
        for i, a in enumerate(actions):
            if a["kind"] == "raise":
                return i  # actions are size-ordered; first = smallest
        raise RuntimeError(f"no raise in {[a['label'] for a in actions]}")
    for i, a in enumerate(actions):
        if a["kind"] == kind:
            return i
    raise RuntimeError(f"no {kind} in {[a['label'] for a in actions]}")

def walk(steps):
    path = []
    for step in steps:
        node = req("POST", "/api/preflop/node", {"path": path})
        path.append(find_action(node["actions"], step))
    return path

def stage_spots(games, iters, check, target, mini=False):
    os.makedirs(os.path.join(OUT, "spots"), exist_ok=True)
    manifest = {}
    for gname, cfg, lines in games:
        print(f"=== {gname}: solving lab game", flush=True)
        st, secs = solve_lab(cfg, iters, check, target)
        print(f"    iter {st['iteration']} gap {st.get('gap_total', -1):.4f} "
              f"[{secs:.0f}s] gpu={st.get('gpu')}", flush=True)
        for lname, steps in lines.items():
            sname = f"{gname}__{lname}"
            try:
                path = walk(steps)
                ex = req("POST", "/api/preflop/export", {"path": path})
            except Exception as exc:
                print(f"    {sname}: EXPORT FAILED: {exc}", flush=True)
                continue
            spot = spot_config(ex, mini=mini)
            fname = os.path.join(OUT, "spots", sname + ".json")
            with open(fname, "w") as f:
                json.dump(spot, f, indent=1)
            manifest[sname] = {
                "game": gname, "line": lname, "oop": ex["oop_pos"], "ip": ex["ip_pos"],
                "pot_bb": ex["pot_bb"], "eff_stack_bb": ex["eff_stack_bb"],
                "spr": ex["eff_stack_bb"] / max(ex["pot_bb"], 1e-9),
                "lab_iterations": st["iteration"], "lab_gap": st.get("gap_total"),
            }
            print(f"    {sname}: pot {ex['pot_bb']:.1f} spr "
                  f"{manifest[sname]['spr']:.1f} ({ex['oop_pos']} vs {ex['ip_pos']})",
                  flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"spots stage done: {len(manifest)} spots", flush=True)

def stage_run(flops_file, iters, target):
    spots = sorted(os.listdir(os.path.join(OUT, "spots")))
    cli = os.path.join(ROOT, "target/release/solve-cli")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    # wide-range spots (limped pots, NL10 defends) cost ~10x per flop; 40
    # flops x ~265 classes still out-samples 100 x ~42 on tight spots
    flops40 = os.path.join(ROOT, "m5_spots/flops40.txt")
    if not os.path.exists(flops40):
        with open(flops40, "w") as f:
            subprocess.check_call([cli, "flops", "40"], stdout=f)
    total_t0 = time.time()
    for i, s in enumerate(spots):
        name = s[:-5]
        obs = os.path.join(OUT, f"obs_{name}.jsonl")
        mark = obs + ".done"
        if os.path.exists(mark):
            print(f"[{i+1}/{len(spots)}] {name}: already done, skipping", flush=True)
            continue
        if os.path.exists(obs):
            os.remove(obs)  # partial from a crash: redo cleanly
        wide = ("limp" in name) or ("nl10" in name)
        f_file = flops40 if wide else flops_file
        f_target = max(target, 0.45) if wide else target
        tag = " (wide: 40 flops, 0.45%)" if wide else ""
        print(f"[{i+1}/{len(spots)}] {name}{tag}", flush=True)
        rc = subprocess.call(
            [cli, "realization", os.path.join(OUT, "spots", s), f_file,
             str(iters), str(f_target), obs],
            cwd=ROOT, env=env)
        if rc != 0:
            print(f"    FAILED rc={rc}", flush=True)
            continue
        open(mark, "w").write("ok\n")
    with open(os.path.join(OUT, "realization_obs.jsonl"), "w") as out:
        for s in spots:
            name = s[:-5]
            p = os.path.join(OUT, f"obs_{name}.jsonl")
            if os.path.exists(p + ".done"):
                out.write(open(p).read())
    open(os.path.join(OUT, "DONE"), "w").write(
        f"finished in {time.time() - total_t0:.0f}s\n")
    print("run stage done -> m5_out/realization_obs.jsonl", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["spots", "run", "all"], default="all")
    ap.add_argument("--flops", default=os.path.join(ROOT, "m5_spots/flops100.txt"))
    ap.add_argument("--iters", type=int, default=900)
    ap.add_argument("--target", type=float, default=0.3)
    ap.add_argument("--lab-iters", type=int, default=1500)
    ap.add_argument("--lab-target", type=float, default=0.008)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--mini", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    games = GAMES
    if args.mini:
        games = MINI
        args.lab_iters, args.lab_target = 300, 0.02
        args.iters, args.target = 300, 0.6
        flops = os.path.join(OUT, "flops_mini.txt")
        open(flops, "w").write("Ks7h2d\nAc6c2d\n")
        args.flops = flops
    elif args.pilot:
        # tight-range 8-max spots: they fit a 24GB card; the wide NL10
        # ranges are H100 material
        games = [g for g in GAMES if g[0] == "g22_open5"]
        games = [(g[0], g[1], {k: g[2][k] for k in ("utg_open_btn_call", "utg_open_bb_call")}) for g in games]
        args.lab_iters, args.lab_target = 800, 0.012
        args.iters, args.target = 600, 0.4
        flops = os.path.join(OUT, "flops_pilot.txt")
        subset = open(args.flops).read().split()[:10]
        open(flops, "w").write("\n".join(subset) + "\n")
        args.flops = flops

    proc = ensure_server() if args.stage in ("spots", "all") else None
    try:
        if args.stage in ("spots", "all"):
            stage_spots(games, args.lab_iters, 100, args.lab_target, mini=args.mini)
        if args.stage in ("run", "all"):
            stage_run(args.flops, args.iters, args.target)
    finally:
        if proc is not None:
            proc.terminate()

if __name__ == "__main__":
    main()
