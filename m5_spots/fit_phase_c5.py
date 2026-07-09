#!/usr/bin/env python3
"""M5 Phase C fitter (v5): class x (role, pot-type) standardization.

Extends v4 (see fit_phase_c.py) with the pot-type axis from its known
residual: within the FACING role, single-raised pots and 3-bet pots are
different games (a 3-bet-pot caller faces a QQ+/AK-dense range at low
SPR), and classes differ in how much of their facing mass came from
each (99/JJ/TT ate 3-bet-pot defends -> 0.73-0.76 bases; 66 set-mined
single-raised pots -> 0.94). v5 cells: (facing|init) x (SRP|3BP) plus
limped. Requires round-2 data (phase_b.py --round2: dense 3-bet-pot
lines + the 4-max 7.5x game) — refuses to run without real 3BP support.

Estimator (hierarchical, degrades gracefully to v4 then to in-context):
  cell mean  --lam15-->  pot-agnostic role mean  --lam15-->  IPF prior
Per class, unsupported pot-type mass folds back into the supported pot
cells (a class with no 3-bet-pot rows is standardized exactly as in v4);
single-role classes keep their in-context value. Blend alpha=0.85 x
role-coverage, equity-anchored curve shrinkage, weighted-PAVA chains
(now including big pairs AA>=KK>=QQ>=JJ>=TT>=99; 88-22 left free for
legitimate set-miner premium). Writes table version 5.

Initiative/pot-type come from the line NAME + manifest.json (oop/ip
positions), so new round-2 lines parse without hand-editing:
  "<opener>_open_<x>_call"  -> opener has initiative
  "<opener>_open_<y>_3bet"  -> y has initiative (opener called the 3-bet)
  "...limp..."              -> limped, no initiative

Usage: python3 m5_spots/fit_phase_c5.py [obs.jsonl] [out.json]
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "m5_out/realization_obs.jsonl")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "cache/realization_fit.json")

SPR_EDGES = [2.5, 5.0, 8.0, 13.0, 22.0]
NB = len(SPR_EDGES) + 1
RANKS = "23456789TJQKA"
LAMN = 40.0      # class-base shrinkage toward the curve, in rows
LAMC = 15.0      # per-level cell shrinkage, in rows
RIDGE = 200.0    # context ridge: mostly trust the bases
M_CLIP = (0.8, 1.25)
ALPHAS = [0.85, 0.75, 0.95]
CURVE_RIDGE = 5.0
CURVE_CLIP = (0.35, 1.45)
MIN_3BP_ROWS = 25        # per class: below this, 3BP mass folds into SRP
MIN_3BP_SPOTS = 8        # dataset guard: v5 needs round-2 data
MIN_3BP_TOTAL = 8000
P3_CLAMP = (0.15, 0.30)  # reference 3BP share of non-limped flops

# cells: 0 facing-SRP, 1 init-SRP, 2 facing-3BP, 3 init-3BP, 4 limped
NCELL = 5

def spr_bucket(s):
    for i, e in enumerate(SPR_EDGES):
        if s < e:
            return i
    return NB - 1

def class_of(label):
    hi, lo = RANKS.index(label[0]), RANKS.index(label[1])
    if len(label) == 2:
        return hi * 13 + lo
    return hi * 13 + lo if label[2] == "s" else lo * 13 + hi

def ci(hi, lo, suited):
    if hi == lo:
        return hi * 13 + hi
    return hi * 13 + lo if suited else lo * 13 + hi

def class_shape(k):
    r, c = divmod(k, 13)
    hi, lo = max(r, c), min(r, c)
    return hi, lo, (r > c), (r == c)

def class_combos(k):
    _, _, suited, pair = class_shape(k)
    return 6 if pair else (4 if suited else 12)

def straight_windows(hi, lo):
    if hi == lo:
        return 0
    wins = [{12, 0, 1, 2, 3}] + [set(range(s, s + 5)) for s in range(9)]
    return sum(1 for w in wins if hi in w and lo in w)

LEGACY_INIT = {  # round-1 line shapes (manifest may predate name parsing)
    "bb_call": 1, "utg_open_btn_call": 0, "utg_open_btn_3bet": 1, "sb_3bet": 0,
}

def spot_axes(sname, manifest):
    """(initiative player 0/1/None, pot type 0=SRP 1=3BP 2=limped)."""
    line = sname.split("__", 1)[1] if "__" in sname else sname
    if "limp" in line:
        return None, 2
    pot = 1 if "_3bet" in line else 0
    m = manifest.get(sname)
    if m and "_open_" in line:
        opener = line.split("_open_")[0].split("_")[-1]
        agg = line.split("_open_")[1].split("_")[0] if pot == 1 else opener
        # the aggressor is whoever made the LAST raise: the 3-bettor in a
        # 3-bet pot (opener called), the opener otherwise
        if pot == 1 and not line.endswith("_3bet"):
            raise RuntimeError(f"unexpected 3-bet line shape: {line}")
        ip = (m.get("ip") or "").lower()
        oop = (m.get("oop") or "").lower()
        if agg == ip:
            return 1, pot
        if agg == oop:
            return 0, pot
        raise RuntimeError(f"{sname}: aggressor {agg} is neither {oop} nor {ip}")
    for suffix, init in LEGACY_INIT.items():
        if line.endswith(suffix):
            return init, pot
    raise RuntimeError(f"unknown line shape: {sname}")

def load():
    fp2name = {}
    for f in glob.glob(os.path.join(os.path.dirname(OBS), "spots", "*.json")):
        sp = json.load(open(f))
        fp = (round(sp["tree"]["starting_pot"], 3),
              sp["range_oop"][:40], sp["range_ip"][:40])
        fp2name[fp] = os.path.basename(f)[:-5]
    manifest = {}
    mpath = os.path.join(os.path.dirname(OBS), "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath))
    rows, pot, cur, dropped = [], 1.0, None, 0
    with open(OBS) as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "header":
                sc = r["spot_config"]
                pot = sc["tree"]["starting_pot"]
                fp = (round(pot, 3), sc["range_oop"][:40], sc["range_ip"][:40])
                cur = fp2name.get(fp)
                if cur is None:
                    raise RuntimeError(f"header not matched: pot {pot}")
            elif r["type"] == "obs":
                if cur.startswith("mini"):
                    dropped += 1
                    continue
                rows.append((r, pot, cur))
    if dropped:
        print(f"dropped {dropped} mini-menu smoke-test rows")
    return rows, manifest

def load_strength():
    raw = open(os.path.join(ROOT, "cache/preflop_eq169.bin"), "rb").read()
    t = np.frombuffer(raw[4:], dtype="<f4").reshape(169, 169).astype(float)
    prob = np.array([class_combos(k) for k in range(169)], float) / 1326.0
    return t @ prob

def pava_desc(v, w):
    vals, wts, idx = [], [], []
    for i in range(len(v)):
        vals.append(v[i]); wts.append(w[i]); idx.append([i])
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, i2 = vals.pop(), wts.pop(), idx.pop()
            vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
            wts[-1] += w2
            idx[-1].extend(i2)
    out = np.array(v, float)
    for val, ids in zip(vals, idx):
        for i in ids:
            out[i] = val
    return out

def main():
    rows, manifest = load()
    spots = {n for _, _, n in rows}
    print(f"{len(rows)} observations from {len(spots)} spots")

    axes = {n: spot_axes(n, manifest) for n in spots}
    n3bp_spots = sum(1 for a in axes.values() if a[1] == 1)

    def cell_of(r, n):
        init, pot = axes[n]
        if pot == 2:
            return 4
        return (1 if r["player"] == init else 0) + 2 * pot

    cells_row = np.array([cell_of(r, n) for r, _, n in rows])
    n3bp_rows = int(((cells_row == 2) | (cells_row == 3)).sum())
    print(f"3-bet-pot support: {n3bp_spots} spots, {n3bp_rows} rows")
    if n3bp_spots < MIN_3BP_SPOTS or n3bp_rows < MIN_3BP_TOTAL:
        print(f"insufficient 3-bet-pot data for v5 (need >= {MIN_3BP_SPOTS} "
              f"spots and >= {MIN_3BP_TOTAL} rows — run phase_b.py --round2). "
              "Refusing; the shipped v4 table stands.")
        sys.exit(1)

    acc = defaultdict(lambda: [0.0, 0.0])
    for r, _, n in rows:
        k = (n, r["board"], r["player"])
        acc[k][0] += r["eq"] * r["reach"]
        acc[k][1] += r["reach"]
    reqs = {k: v[0] / v[1] for k, v in acc.items()}

    y = np.array([min(max(r["r_obs"], 0.0), 3.0) for r, _, _ in rows])
    wr = np.array([r["reach"] for r, _, _ in rows])
    w = np.array([r["reach"] * p for r, p, _ in rows])
    w /= w.mean()
    ks = np.array([class_of(r["label"]) for r, _, _ in rows])
    roles = np.where(cells_row == 4, 2, cells_row % 2)  # 0 facing, 1 init, 2 limp

    gmean = float(np.sum(wr * y) / np.sum(wr))
    nk = np.array([(ks == k).sum() for k in range(169)])
    wsum = np.zeros(169)
    raw = np.full(169, gmean)
    for k in range(169):
        m = ks == k
        if m.any():
            wsum[k] = wr[m].sum()
            raw[k] = float(np.sum(wr[m] * y[m]) / wsum[k])
    observed = nk > 0

    # --- IPF main effects over the 5 cells ---
    eff = raw.copy()
    rho = np.ones(NCELL)
    for _ in range(60):
        for c in range(NCELL):
            m = cells_row == c
            if m.any():
                rho[c] = np.sum(wr[m] * y[m]) / max(np.sum(wr[m] * eff[ks[m]]), 1e-12)
        rho /= np.sum(wr * rho[cells_row]) / np.sum(wr)
        for k in np.where(observed)[0]:
            m = ks == k
            eff[k] = np.sum(wr[m] * y[m]) / max(np.sum(wr[m] * rho[cells_row[m]]), 1e-12)
    print("cell multipliers (f-SRP/i-SRP/f-3BP/i-3BP/limp): "
          + " / ".join(f"{v:.3f}" for v in rho))
    # pot-agnostic role multipliers, for the mid-level prior
    rho_role = np.ones(3)
    for r3 in range(3):
        m = roles == r3
        if m.any():
            rho_role[r3] = np.sum(wr[m] * y[m]) / max(np.sum(wr[m] * eff[ks[m]]), 1e-12)

    # --- reference mix ---
    limp_share = float(np.sum(wr[roles == 2]) / np.sum(wr))
    nl = np.sum(wr[roles != 2])
    p3_data = float(np.sum(wr[(cells_row == 2) | (cells_row == 3)]) / max(nl, 1e-12))
    p3 = min(max(p3_data, P3_CLAMP[0]), P3_CLAMP[1])
    print(f"limp share {limp_share:.2f}; 3BP share of non-limped {p3_data:.2f} "
          f"-> reference {p3:.2f}")
    pi5 = np.array([(1 - limp_share) * (1 - p3) / 2, (1 - limp_share) * (1 - p3) / 2,
                    (1 - limp_share) * p3 / 2, (1 - limp_share) * p3 / 2, limp_share])

    # --- per-class standardization with per-class supported-pot renorm ---
    std = raw.copy()
    cov = np.zeros(169)
    for k in np.where(observed)[0]:
        mk = ks == k
        # mid-level: pot-agnostic role cells (v4's cells)
        mid = np.zeros(3)
        for r3 in range(3):
            m = mk & (roles == r3)
            prior = eff[k] * rho_role[r3]
            n_cell = int(m.sum())
            if n_cell:
                mc = float(np.sum(wr[m] * y[m]) / np.sum(wr[m]))
                mid[r3] = (n_cell * mc + LAMC * prior) / (n_cell + LAMC)
            else:
                mid[r3] = prior
        # leaf cells shrink toward mid-level scaled by the cell/role ratio
        leaf = np.zeros(NCELL)
        for c in range(NCELL):
            r3 = 2 if c == 4 else c % 2
            prior = mid[r3] * (rho[c] / max(rho_role[r3], 1e-12))
            m = mk & (cells_row == c)
            n_cell = int(m.sum())
            if n_cell:
                mc = float(np.sum(wr[m] * y[m]) / np.sum(wr[m]))
                leaf[c] = (n_cell * mc + LAMC * prior) / (n_cell + LAMC)
            else:
                leaf[c] = prior
        # unsupported 3BP mass folds back into the SRP cells (v4 behavior)
        pik = pi5.copy()
        if int((mk & ((cells_row == 2) | (cells_row == 3))).sum()) < MIN_3BP_ROWS:
            pik[0] += pik[2]; pik[1] += pik[3]
            pik[2] = pik[3] = 0.0
            leaf[0], leaf[1] = mid[0], mid[1]
        std[k] = float(pik @ leaf)
        sf, si = np.sum(wr[mk & (roles == 0)]), np.sum(wr[mk & (roles == 1)])
        fi = sf + si
        cov[k] = 4.0 * sf * si / (fi * fi) if fi > 0 else 0.0

    # --- curve + chains + gates + context stage: same shape as v4 ---
    strength = load_strength()
    feats = np.zeros((169, 7))
    for k in range(169):
        hi, lo, suited, pair = class_shape(k)
        s = strength[k]
        feats[k] = [1.0, s, s * s, float(suited), float(pair),
                    straight_windows(hi, lo) / 4.0, hi / 12.0]

    def build(alpha):
        a_k = alpha * cov
        base = (1.0 - a_k) * raw + a_k * std
        base *= gmean / (np.sum(wsum * base) / np.sum(wsum))
        X, bw = feats[observed], nk[observed].astype(float)
        A = (X * bw[:, None]).T @ X + CURVE_RIDGE * np.eye(7)
        A[0, 0] -= CURVE_RIDGE
        beta = np.linalg.solve(A, (X * bw[:, None]).T @ base[observed])
        g = np.clip(feats @ beta, *CURVE_CLIP)
        out = np.where(observed, (nk * base + LAMN * g) / (nk + LAMN), g)
        wgt = nk + LAMN
        lad = lambda labs: [class_of(l) for l in labs]
        chains = [lad(["AKs", "AQs", "AJs", "ATs", "A9s", "A8s", "A7s", "A6s"]),
                  lad(["AKo", "AQo", "AJo", "ATo", "A9o", "A8o", "A7o", "A6o"]),
                  lad(["A5s", "A4s", "A3s", "A2s"]), lad(["A5o", "A4o", "A3o", "A2o"]),
                  lad(["KQs", "KJs", "KTs"]), lad(["KQo", "KJo", "KTo"]),
                  lad(["QJs", "QTs"]), lad(["QJo", "QTo"]),
                  lad(["AA", "KK", "QQ", "JJ", "TT", "99"])]
        for ch in chains:
            out[ch] = pava_desc(out[ch], wgt[ch])
        for hi in range(13):
            for lo in range(hi):
                out[ci(hi, lo, False)] = min(out[ci(hi, lo, False)], out[ci(hi, lo, True)])
        return out

    def base_gates(b):
        return [
            ("wheel premium bounded (A5s <= 1.45 ATs)",
             b[class_of("A5s")] <= 1.45 * b[class_of("ATs")]),
            ("middle aces recovered (ATs >= 0.70, A9s >= 0.60)",
             b[class_of("ATs")] >= 0.70 and b[class_of("A9s")] >= 0.60),
            ("AA base over-realizes", b[class_of("AA")] > 1.0),
            ("scale preserved", abs(np.sum(wsum * b) / np.sum(wsum) - gmean) < 0.02),
            ("mid pairs vs set-miners sane (66 <= 1.15 x 99)",
             b[class_of("66")] <= 1.15 * b[class_of("99")]),
        ]

    base, alpha = None, None
    for a in ALPHAS:
        cand = build(a)
        checks = base_gates(cand)
        tag = " ".join("PASS" if p else "FAIL" for _, p in checks)
        print(f"alpha={a}: ATs {cand[class_of('ATs')]:.2f} A5s {cand[class_of('A5s')]:.2f} "
              f"99 {cand[class_of('99')]:.2f} 66 {cand[class_of('66')]:.2f}  [{tag}]")
        if base is None and all(p for _, p in checks):
            base, alpha = cand, a
    if base is None:
        print("no alpha passes the base gates — not writing the table")
        sys.exit(1)
    print(f"selected alpha={alpha}")

    def ctx_feats(r, n):
        init, pot = axes[n]
        iv = 0.0 if init is None else (0.5 if r["player"] == init else -0.5)
        x = np.zeros(2 + NB + 2)
        x[0] = 1.0
        x[1] = r["pos_frac"]
        x[2 + spr_bucket(r["spr"])] = 1.0
        x[2 + NB] = reqs[(n, r["board"], r["player"])] - 0.5
        x[3 + NB] = iv
        return x

    Xc = np.stack([ctx_feats(r, n) for r, _, n in rows])
    ratio = y / base[ks]
    A = (Xc * w[:, None]).T @ Xc + RIDGE * np.eye(Xc.shape[1])
    A[0, 0] -= RIDGE
    beta = np.linalg.solve(A, (Xc * w[:, None]).T @ ratio)
    names = ["c0", "pos"] + [f"spr{i}" for i in range(NB)] + ["range_eq", "initiative"]
    print({n: round(float(v), 4) for n, v in zip(names, beta)})

    def mult(pos, spr, req, init):
        x = np.zeros(len(beta))
        x[0] = 1.0
        x[1] = pos
        x[2 + spr_bucket(spr)] = 1.0
        x[2 + NB] = req - 0.5
        x[3 + NB] = init
        return float(np.clip(x @ beta, *M_CLIP))

    def R(lab, pos, spr, req=0.5, init=0.0):
        return float(np.clip(base[class_of(lab)] * mult(pos, spr, req, init), 0.2, 2.5))

    pred = base[ks] * np.clip(Xc @ beta, *M_CLIP)
    r2 = 1 - float(np.sum(w * (y - pred) ** 2) / np.sum(w * (y - gmean) ** 2))
    print(f"v5 weighted R²: {r2:.3f}")

    checks = [
        ("initiative premium", beta[3 + NB] > 0.03),
        ("aggressor beats defender", R("T9s", 0.5, 8, init=0.5) > R("T9s", -0.5, 8, init=-0.5)),
        ("suited > offsuit (76)", R("76s", 0, 8) > R("76o", 0, 8)),
        ("connected > gapped junk", R("76s", 0, 8) > R("72s", 0, 8)),
        ("A9o >= 32s in defend context",
         R("A9o", -0.5, 20, init=-0.5) >= R("32s", -0.5, 20, init=-0.5)),
        ("A9o never craters (v1 postmortem)", R("A9o", -0.5, 20, init=-0.5) > 0.40),
        ("AA over-realizes", R("AA", 0, 8) > 1.0),
    ] + base_gates(base)
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    if not ok:
        print("SANITY FAILURES — not writing the table")
        sys.exit(1)
    for lab in ("AKs", "AQs", "AJs", "ATs", "A9s", "A6s", "A5s", "A2s", "AA",
                "JJ", "TT", "99", "66", "JTs", "K8s", "K5s", "ATo", "KJo", "72o"):
        print(f"{lab:>4} {base[class_of(lab)]:>5.2f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({
        "version": 5,
        "spr_edges": SPR_EDGES,
        "class_base": [round(float(v), 5) for v in base],
        "ctx": {n: float(v) for n, v in zip(names, beta)},
        "mult_clip": list(M_CLIP),
        "clip": [0.2, 2.5],
        "meta": {
            "n_obs": len(rows), "r2": round(r2, 4),
            "alpha": alpha,
            "rho_cells": {"f_srp": round(float(rho[0]), 4), "i_srp": round(float(rho[1]), 4),
                          "f_3bp": round(float(rho[2]), 4), "i_3bp": round(float(rho[3]), 4),
                          "limp": round(float(rho[4]), 4)},
            "pi5": [round(float(v), 4) for v in pi5],
            "n_3bp_spots": n3bp_spots, "n_3bp_rows": n3bp_rows,
            "note": "v5 = v4 + pot-type axis: class x (role, pot-type) cells, "
                    "hierarchical shrinkage cell->role->IPF, unsupported 3BP "
                    "mass folds back to SRP per class, big-pair PAVA chain "
                    "added. Requires round-2 data (3-bet pots + 4-max game).",
        },
    }, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
