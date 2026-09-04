import os
import math
import time
import random
import tracemalloc
import itertools
import subprocess
from fractions import Fraction

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import visibility as V
import latsieve as LS

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

COL = {"enum": "#0072B2", "mark": "#D55E00", "fiber": "#009E73"}
MRK = {"enum": "o", "mark": "s", "fiber": "^"}
LST = {"enum": "-", "mark": "--", "fiber": "-."}
LBL = {"enum": "enumeration", "mark": "marking sieve", "fiber": "fiber sieve"}

SCOL = {"line": "#0072B2", "plane": "#D55E00", "ntv": "#CC79A7",
        "fib": "#009E73"}
SMRK = {"line": "o", "plane": "s", "ntv": "d", "fib": "^"}
SLST = {"line": "-", "plane": "--", "ntv": ":", "fib": "-."}
SLBL = {"line": "line sieve", "plane": "plane sieve",
        "ntv": "transition sieve", "fib": "fibered sieve"}

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 150,
})

GEOM = {3: 64, 4: 20, 5: 10, 6: 6}
RSEQ = [7, 61, 509, 4001, 32003, 250007, 1000003]
RGRID = {3: RSEQ, 4: RSEQ, 5: RSEQ, 6: RSEQ}


def log(msg):
    print(msg, flush=True)


def save_fig(fig, name):
    eps = os.path.join(FIGURES, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight")
    ok = False
    try:
        subprocess.run(["epstopdf", eps, "--outfile",
                        os.path.join(FIGURES, name + ".pdf")],
                       check=True, capture_output=True)
        ok = True
    except Exception:
        ok = False
    if not ok:
        fig.savefig(os.path.join(FIGURES, name + ".pdf"),
                    format="pdf", bbox_inches="tight")
    plt.close(fig)


def timed(fun, *args):
    t0 = time.perf_counter()
    val = fun(*args)
    t1 = time.perf_counter()
    return val, t1 - t0


def peak_mem(fun, *args):
    tracemalloc.start()
    fun(*args)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def make_box(t, side):
    return [(-side, side) for _ in range(t - 1)] + [(0, side)]


def exp_correctness():
    log("[E1] cross-validation of the three counting methods")
    random.seed(20260730)
    lines = []
    agree = True
    for trial in range(60):
        k = random.choice([2, 2, 3])
        s = random.choice([1, 2, 3])
        L = random.choice([21, 40] if k == 2 else [9, 14, 22])
        S = set()
        while len(S) < s:
            S.add(tuple(random.randint(-5, 5) for _ in range(k)))
        S = sorted(S)
        a = V.count_bruteforce(S, L)
        b = V.count_marking(S, L)
        c = V.count_exact(S, L)
        if not (a == b == c):
            agree = False
            lines.append(f"MISMATCH k={k} L={L} S={S}: {a} {b} {c}")
    lines.append(f"60 random instances, all three methods agree: {agree}")
    with open(os.path.join(RESULTS, "e1_correctness.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log("     " + lines[-1])


def exp_timing():
    log("[E2] time and space comparison for exact counting")
    grid = {
        2: ([(1, 0), (0, 1)],
            [100, 300, 1000, 3000, 10000, 30000, 100000],
            {"enum": 3000, "mark": 3000, "fiber": 100000}),
        3: ([(0, 0, 0), (1, 2, 2)],
            [30, 100, 300, 1000, 10000, 100000, 1000000],
            {"enum": 100, "mark": 300, "fiber": 1000000}),
        4: ([(0, 0, 0, 0), (1, 2, 2, 1)],
            [20, 50, 200, 1000, 10000, 100000],
            {"enum": 50, "mark": 50, "fiber": 100000}),
    }
    rows = []
    plotdata = {}
    for k, (S, Ls, caps) in grid.items():
        plotdata[k] = {m: ([], []) for m in ["enum", "mark", "fiber"]}
        for L in Ls:
            entry = {"k": k, "L": L}
            vals = {}
            for m, fun in [("enum", V.count_bruteforce),
                           ("mark", V.count_marking),
                           ("fiber", V.count_exact)]:
                if L <= caps[m]:
                    val, dt = timed(fun, S, L)
                    vals[m] = val
                    entry[m] = dt
                    plotdata[k][m][0].append(L)
                    plotdata[k][m][1].append(dt)
                else:
                    entry[m] = None
            got = set(v for v in vals.values())
            assert len(got) <= 1, (k, L, vals)
            entry["N"] = vals.get("fiber", vals.get("mark", vals.get("enum")))
            rows.append(entry)
            log(f"     k={k} L={L}: " + "  ".join(
                f"{m}={entry[m]:.3f}s" if entry[m] is not None else f"{m}=--"
                for m in ["enum", "mark", "fiber"]) + f"  N={entry['N']}")
    mems = []
    for k, S, L in [(2, [(1, 0), (0, 1)], 2000),
                    (3, [(0, 0, 0), (1, 2, 2)], 120),
                    (3, [(0, 0, 0), (1, 2, 2)], 100000),
                    (4, [(0, 0, 0, 0), (1, 2, 2, 1)], 40)]:
        pm_mark = peak_mem(V.count_marking, S, L) if L ** k <= 3 * 10 ** 7 else None
        pm_fib = peak_mem(V.count_exact, S, L)
        mems.append((k, L, pm_mark, pm_fib))
        log(f"     mem k={k} L={L}: marking={pm_mark} B  fiber={pm_fib} B")
    with open(os.path.join(RESULTS, "e2_timing.txt"), "w") as f:
        for e in rows:
            f.write(repr(e) + "\n")
        for mrow in mems:
            f.write("MEM " + repr(mrow) + "\n")
    return plotdata, rows, mems


def scan_G_np(S, k, Lmax, Dmax=350):
    G = np.zeros(Lmax + 1)
    idxs = np.arange(Lmax + 1)
    for D in range(2, Dmax + 1):
        x = D
        pr = []
        ok = True
        p = 2
        while p * p <= x:
            if x % p == 0:
                x //= p
                pr.append(p)
                if x % p == 0:
                    ok = False
                    break
            p += 1
        if not ok:
            continue
        if x > 1:
            pr.append(x)
        classes = [V.classes_mod_p(S, p) for p in pr]
        cnt = np.zeros(D, dtype=np.int64)
        nlab = 1
        for c in classes:
            nlab *= len(c)
        for combo in itertools.product(*classes):
            m = 1
            res = [0] * k
            for (p, cls) in zip(pr, combo):
                inv = pow(m, -1, p)
                for i in range(k):
                    tt = (cls[i] - res[i]) % p
                    res[i] = res[i] + m * ((tt * inv) % p)
                m *= p
            for i in range(k):
                c = res[i] % D
                cbar = c if c != 0 else D
                if cbar <= D - 1:
                    cnt[cbar:] += 1
        r = np.arange(D)
        gD = (cnt - k * nlab * r / D) * ((-1) ** len(pr)) * float(D) ** (1.0 - k)
        G += gD[idxs % D]
    return G


def exp_sd():
    log("[E3] Schnirelmann density scans and deficit certificates")
    out = []
    for k, Lmax, label in [(2, 2000, "V2"), (3, 200000, "V3")]:
        counts = V.jordan_scan_counts(k, Lmax)
        best = None
        first_def = None
        lo, hi = V.density_bounds([tuple([0] * k)], k)
        for L in range(1, Lmax + 1):
            r = Fraction(counts[L - 1], L ** k)
            if r < lo and first_def is None:
                first_def = L
            if best is None or r < best[1]:
                best = (L, r)
        out.append(f"{label}: min ratio at L={best[0]}: {float(best[1]):.9f} "
                   f"(density in [{float(lo):.9f},{float(hi):.9f}]), "
                   f"first deficit at L={first_def}, "
                   f"N(Lmin)={counts[best[0]-1]}")
        log("     " + out[-1])
    scans = [
        ([(1, 0), (0, 1)], 600),
        ([(0, 0), (1, 0), (0, 1)], 1200),
    ]
    ratio_series = {}
    for S, Lmax in scans:
        k = len(S[0])
        lo, hi = V.density_bounds(S, k)
        best = None
        ndef = 0
        first_def = None
        rs = []
        for L in range(1, Lmax + 1):
            N = V.count_exact(S, L)
            r = Fraction(N, L ** k)
            rs.append((L, float(r)))
            if r < lo:
                ndef += 1
                if first_def is None:
                    first_def = L
            if best is None or r < best[1]:
                best = (L, r, N)
        ratio_series[str(S)] = (rs, float(lo))
        Lb, rb, Nb = best
        out.append(f"S={S}: min ratio {float(rb):.9f} = {rb} at L={Lb} (N={Nb}), "
                   f"density in [{float(lo):.9f},{float(hi):.9f}], "
                   f"first deficit L={first_def}, deficits<= {Lmax}: {ndef}")
        log("     " + out[-1])

    log("[E3b] certified Schnirelmann densities")
    certified = [
        [(1, 2, 1)],
        [(2, 1, 1), (1, 2, 1)],
        [(1, 2, 1), (2, 1, 2)],
        [(1, 2, 3), (3, 2, 1)],
        [(1, 0, 0), (0, 1, 0)],
        [(1, 2, 1, 1)],
        [(1, 2, 3, 1)],
        [(0, 0, 0, 0), (1, 2, 1, 1)],
    ]
    cert_rows = []
    for S in certified:
        k = len(S[0])
        lo, hi = V.density_bounds(S, k)
        probe = [(L, Fraction(V.count_exact(S, L), L ** k))
                 for L in range(1, 400 if k == 3 else 200)]
        _, L1, info = V.certify_sd_minimum(S, probe)
        ratios = list(probe)
        have = len(probe)
        for L in range(have + 1, L1 + 1):
            ratios.append((L, Fraction(V.count_exact(S, L), L ** k)))
        best = min(ratios, key=lambda pr: pr[1])
        _, L1f, infof = V.certify_sd_minimum(S, ratios)
        assert L1f <= max(L for L, _ in ratios), (S, L1f)
        first_def = min((L for (L, r) in ratios if r < lo), default=None)
        ndef = sum(1 for (L, r) in ratios if r < lo)
        row = {"S": S, "k": k,
               "delta_lo": float(lo), "delta_hi": float(hi),
               "SD": f"{best[1].numerator}/{best[1].denominator}",
               "SD_float": float(best[1]),
               "L_min": best[0], "L1": L1f,
               "z": infof[0], "t": infof[1],
               "first_deficit": first_def, "n_deficits": ndef,
               "N_at_min": V.count_exact(S, best[0])}
        cert_rows.append(row)
        out.append(f"S={S} k={k}: SD = {best[1]} = {float(best[1]):.9f} at L={best[0]}, "
                   f"delta in [{float(lo):.9f},{float(hi):.9f}], "
                   f"certified by scan to L1={L1f} with (z,t)=({infof[0]},{infof[1]}), "
                   f"first deficit L={first_def}, deficits: {ndef}")
        log("     " + out[-1])
    S_hard = [(0, 0, 0), (-1, -1, -1), (-2, 0, -1)]
    lo, hi = V.density_bounds(S_hard, 3)
    G = scan_G_np(S_hard, 3, 100000, Dmax=350)
    cand = np.argsort(G[100:])[:3] + 100
    hard_lines = [f"S_hard={S_hard}: density in [{float(lo):.9f},{float(hi):.9f}], "
                  f"min truncated G on [100,100000] = {G[100:].min():+.5f}"]
    for L in cand:
        L = int(L)
        N = V.count_exact(S_hard, L)
        diff = Fraction(N) - lo * L ** 3
        hard_lines.append(f"   candidate L={L}: N={N}, N-lo*L^3={float(diff):+.1f} "
                          f"(no deficit)" if diff > 0 else
                          f"   candidate L={L}: N={N}, DEFICIT")
        log("     " + hard_lines[-1])
    out.extend(hard_lines)
    with open(os.path.join(RESULTS, "e3_sd.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(RESULTS, "e3_certified.txt"), "w") as f:
        for row in cert_rows:
            f.write(repr(row) + "\n")
    return ratio_series, cert_rows


def exp_bu():
    log("[E4] criterion values B_u")
    sets = [
        ("origin", [(0, 0, 0)]),
        ("mixed singleton", [(3, -7, 2)]),
        ("two positive", [(0, 0, 0), (1, 1, 1)]),
        ("chain", [(0, 0, 0), (-1, -1, -1), (-2, -2, -2), (-3, -3, -3)]),
        ("nonpositive", [(0, 0, 0), (-1, -1, -1), (-2, 0, -1)]),
        ("axis pattern", [(0, 0, 0), (-2, 0, 0), (0, -2, 0), (0, 0, -2)]),
    ]
    lines = []
    table = {}
    for name, S in sets:
        vals = []
        for u in range(0, 8):
            b, tl = V.B_criterion(S, 3, u, Dmax=400)
            vals.append(b)
        table[name] = vals
        lines.append(name + " & " + " & ".join(f"{v:+.3f}" for v in vals))
        log("     " + lines[-1])
    with open(os.path.join(RESULTS, "e4_bu.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return table


def exp_sieve_validation():
    log("[E5] validation of the sieve algorithms against exhaustive search")
    rng = random.Random(20260801)
    lines = []
    mismatch = 0
    ntrial = 0
    tot_ref = 0
    tot_ntv = 0
    for trial in range(240):
        t = rng.choice([2, 3, 4, 5])
        side = rng.choice([3, 4, 6, 8])
        r = rng.choice([5, 7, 11, 13, 17, 23, 31, 101, 211, 1009, 4001])
        lam = [rng.randrange(0, r) for _ in range(t - 1)]
        box = make_box(t, side)
        if LS.box_volume(box) > 300000:
            continue
        ntrial += 1
        ref = sorted(LS.brute_sieve(r, lam, box))
        got = {
            "line": sorted(LS.line_sieve(r, lam, box)),
            "plane": sorted(LS.plane_sieve(r, lam, box)),
            "fib": sorted(LS.fibered_sieve(r, lam, box)),
        }
        for name, val in got.items():
            if val != ref:
                mismatch += 1
                lines.append(f"MISMATCH {name} t={t} side={side} r={r} lam={lam}")
        heur = set(LS.space_sieve(r, lam, box))
        tot_ref += len(ref)
        tot_ntv += len(heur & set(ref))
    lines.append(f"{ntrial} random instances in dimensions 2 to 5; "
                 f"line, plane and fibered sieves agree with exhaustive "
                 f"search on all of them: {mismatch == 0}")
    lines.append(f"transition sieve recall on the same instances: "
                 f"{tot_ntv}/{tot_ref} = {tot_ntv/max(tot_ref,1):.4f}")
    with open(os.path.join(RESULTS, "e5_sieve_validation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for ln in lines[-2:]:
        log("     " + ln)
    return mismatch == 0


def exp_sieve_bench(reps=4):
    log("[E6] running time, work and completeness against the level")
    rng = random.Random(4242)
    rows = []
    for t in sorted(GEOM):
        side = GEOM[t]
        box = make_box(t, side)
        vol = LS.box_volume(box)
        for r in RGRID[t]:
            if r > vol:
                continue
            acc = {k: 0.0 for k in ["line", "plane", "ntv", "fib"]}
            npts = 0
            nrec = 0
            nslice = 0
            nnodes = 0
            lev = None
            for _ in range(reps):
                lam = [rng.randrange(1, r) for _ in range(t - 1)]
                fs = LS.FiberedSieve(r, lam, box)
                lev = fs.level
                res, dt = timed(fs.run)
                acc["fib"] += dt
                nslice += fs.slices
                nnodes += fs.nodes()
                exact = set(res)
                npts += len(exact)
                _, dt = timed(LS.line_sieve, r, lam, box)
                acc["line"] += dt
                _, dt = timed(LS.plane_sieve, r, lam, box)
                acc["plane"] += dt
                heur, dt = timed(LS.space_sieve, r, lam, box)
                acc["ntv"] += dt
                nrec += len(set(heur) & exact)
            row = {"t": t, "side": side, "vol": vol, "r": r, "level": lev,
                   "pts": npts / reps,
                   "expected": vol / float(r),
                   "slices": nslice / reps,
                   "nodes": nnodes / reps,
                   "recall": nrec / max(npts, 1),
                   "t_line": acc["line"] / reps,
                   "t_plane": acc["plane"] / reps,
                   "t_ntv": acc["ntv"] / reps,
                   "t_fib": acc["fib"] / reps,
                   "w_line": LS.line_sieve_work(r, None, box),
                   "w_plane": LS.plane_sieve_work(r, None, box),
                   "w_fib": LS.fibered_sieve_work(r, None, box)}
            rows.append(row)
            log("     t=%d r=%-7d level=%d pts=%.1f  line=%.3f plane=%.3f "
                "ntv=%.3f fib=%.3f  recall=%.3f"
                % (t, r, lev, row["pts"], row["t_line"], row["t_plane"],
                   row["t_ntv"], row["t_fib"], row["recall"]))
    with open(os.path.join(RESULTS, "e6_sieve_bench.txt"), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")
    return rows


def exp_scaling():
    log("[E7] output sensitivity and scaling in the region size")
    rng = random.Random(77)
    rows = []
    t = 4
    for side in [8, 12, 16, 24, 32]:
        box = make_box(t, side)
        vol = LS.box_volume(box)
        r = 32003
        acc = {"line": 0.0, "plane": 0.0, "fib": 0.0}
        npts = 0
        nsl = 0
        reps = 3
        for _ in range(reps):
            lam = [rng.randrange(1, r) for _ in range(t - 1)]
            fs = LS.FiberedSieve(r, lam, box)
            res, dt = timed(fs.run)
            acc["fib"] += dt
            nsl += fs.slices
            npts += len(res)
            _, dt = timed(LS.line_sieve, r, lam, box)
            acc["line"] += dt
            _, dt = timed(LS.plane_sieve, r, lam, box)
            acc["plane"] += dt
        rows.append({"t": t, "side": side, "vol": vol, "r": r,
                     "level": LS.sieve_level(box, r),
                     "pts": npts / reps, "slices": nsl / reps,
                     "expected": vol / float(r),
                     "t_line": acc["line"] / reps,
                     "t_plane": acc["plane"] / reps,
                     "t_fib": acc["fib"] / reps})
        log("     side=%d vol=%d pts=%.1f slices=%.1f expected=%.1f "
            "line=%.3f plane=%.3f fib=%.3f"
            % (side, vol, rows[-1]["pts"], rows[-1]["slices"],
               rows[-1]["expected"], rows[-1]["t_line"],
               rows[-1]["t_plane"], rows[-1]["t_fib"]))
    with open(os.path.join(RESULTS, "e7_scaling.txt"), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")
    return rows


def exp_memory():
    log("[E8] working memory of the relation collection")
    rng = random.Random(909)
    t = 4
    box = [(-16, 16), (-16, 16), (-16, 16), (0, 32)]
    ideals = []
    for r in LS.primes_upto(400)[10:40]:
        ideals.append((r, [rng.randrange(1, r) for _ in range(t - 1)]))
    rows = []
    h0 = None
    for sf in [4, 3, 2]:
        tracemalloc.start()
        t0 = time.perf_counter()
        h, cells, nb = LS.blocked_collection(ideals, box, sf)
        dt = time.perf_counter() - t0
        m = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        if h0 is None:
            h0 = h
        assert h == h0
        rows.append({"split": sf, "blocks": nb, "cells": cells,
                     "peak": m, "hits": h, "time": dt})
        log("     split=%d blocks=%d cells=%d peak=%d time=%.2fs"
            % (sf, nb, cells, m, dt))
    fs = LS.FiberedSieve(ideals[0][0], ideals[0][1], box)
    enum_mem = peak_mem(fs.run_count)
    rows.append({"split": 0, "blocks": 0, "cells": 0,
                 "peak": enum_mem, "hits": 0, "time": 0.0})
    log("     enumeration alone: peak=%d" % enum_mem)
    with open(os.path.join(RESULTS, "e8_memory.txt"), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")
    return rows


def exp_nodes():
    log("[E10] enumeration nodes per reported point and the shape hypothesis")
    rng = random.Random(31337)
    rows = []
    for t in sorted(GEOM):
        box = make_box(t, GEOM[t])
        vol = LS.box_volume(box)
        for r in RGRID[t]:
            if r > vol or LS.sieve_level(box, r) < 2:
                continue
            nodes = 0
            pts = 0
            ok = 0
            worst = 0.0
            reps = 6
            for _ in range(reps):
                lam = [rng.randrange(1, r) for _ in range(t - 1)]
                fs = LS.FiberedSieve(r, lam, box)
                res = fs.run()
                nodes += fs.nodes()
                pts += len(res)
                g = LS.normalised_gso(fs.ctx.B, box)
                bound = math.sqrt(fs.m)
                worst = max(worst, max(g) / bound)
                if max(g) <= bound:
                    ok += 1
            rows.append({"t": t, "r": r, "level": LS.sieve_level(box, r),
                         "vol": vol, "nodes": nodes / reps, "pts": pts / reps,
                         "expected": vol / float(r),
                         "nodes_per_expected": nodes / reps / (vol / float(r)),
                         "bound": (2 * math.sqrt(LS.sieve_level(box, r) + 1))
                         ** (LS.sieve_level(box, r) + 1),
                         "shape_ok": ok, "shape_reps": reps,
                         "worst_ratio": worst})
            log("     t=%d r=%-8d level=%d nodes/|H|r^-1 = %.2f (bound %.0f) "
                "shape hypothesis %d/%d, worst ratio %.3f"
                % (t, r, rows[-1]["level"], rows[-1]["nodes_per_expected"],
                   rows[-1]["bound"], ok, reps, worst))
    with open(os.path.join(RESULTS, "e10_nodes.txt"), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")
    return rows


YIELD_SETS = [
    ("$\\mathbf{a}=\\mathbf{0}$, $t=3$", [(0, 0, 0)], 3, "#0072B2", "-"),
    ("$\\mathbf{a}=(0,-1,0)$, $t=3$", [(1, 2, 1)], 3, "#D55E00", "--"),
    ("two anchors, $t=3$", [(2, 1, 1), (1, 2, 1)], 3, "#009E73", "-."),
    ("$\\mathbf{a}=(0,-1,0,0)$, $t=4$", [(1, 2, 1, 1)], 4, "#CC79A7", ":"),
]


def exp_yield(cert_rows):
    log("[E9] duplicate-free yield of a sieve region")
    rows = []
    for row in cert_rows:
        d = row["delta_lo"]
        rows.append({"S": row["S"], "k": row["k"], "delta": d,
                     "worst": row["SD_float"], "worst_frac": row["SD"],
                     "L_min": row["L_min"], "L1": row["L1"],
                     "shortfall": (d - row["SD_float"]) / d,
                     "n_deficient": row["n_deficits"]})
        log("     S=%s: prediction %.7f, certified worst case %.7f at L=%d, "
            "relative shortfall %.4f" % (row["S"], d, row["SD_float"],
                                         row["L_min"], rows[-1]["shortfall"]))
    series = {}
    for (lab, S, k, col, ls) in YIELD_SETS:
        lo, hi = V.density_bounds(S, k)
        Lmax = 400 if k == 3 else 160
        vals = []
        for L in range(2, Lmax + 1):
            n = V.count_exact(S, L)
            vals.append((L, n / float(L ** k)))
        series[lab] = (vals, float(lo), col, ls)
    log("[E9b] multi-centre admissible counts")
    mc = []
    for S, k, L in [([(0, 0, 0), (1, 2, 1)], 3, 200),
                    ([(0, 0, 0), (1, 2, 1), (2, 1, 2)], 3, 200),
                    ([(0, 0, 0, 0), (1, 2, 1, 1)], 4, 60)]:
        lo, hi = V.density_bounds(S, k)
        nfib, tfib = timed(V.count_exact, S, L)
        nen, ten = timed(V.count_bruteforce, S, L)
        assert nfib == nen
        mc.append({"S": S, "k": k, "L": L, "N": nfib,
                   "delta": float(lo), "ratio": nfib / float(L ** k),
                   "t_fiber": tfib, "t_enum": ten,
                   "speedup": ten / max(tfib, 1e-9)})
        log("     S=%s L=%d: N=%d ratio=%.6f delta=%.6f  fiber=%.3fs "
            "enum=%.3fs speedup=%.0f"
            % (S, L, nfib, mc[-1]["ratio"], float(lo), tfib, ten,
               mc[-1]["speedup"]))
    with open(os.path.join(RESULTS, "e9_yield.txt"), "w") as f:
        for row in rows:
            f.write(repr(row) + "\n")
        for row in mc:
            f.write("MC " + repr(row) + "\n")
    return rows, series, mc


def fig_timing(plotdata):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, k in zip(axes, [2, 3]):
        for m in ["enum", "mark", "fiber"]:
            xs, ys = plotdata[k][m]
            if xs:
                ax.loglog(xs, ys, marker=MRK[m], color=COL[m], ls=LST[m],
                          label=LBL[m], ms=4, lw=1.2)
        ax.set_xlabel("$L$")
        ax.set_ylabel("time (s)")
        ax.set_title(f"$k={k}$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_time")


def fig_gprofile():
    k = 3
    Lmax = 100000
    sets = [
        ("$S=\\{\\mathbf{0}\\}$", [(0, 0, 0)], "#0072B2", "-", "o"),
        ("$S=\\{(1,2,1)\\}$", [(1, 2, 1)], "#D55E00", "--", "s"),
        ("$S=S_{\\mathrm{h}}$",
         [(0, 0, 0), (-1, -1, -1), (-2, 0, -1)], "#009E73", "-.", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for (lab, S, col, ls, mk) in sets:
        G = scan_G_np(S, k, Lmax, Dmax=350)
        w = np.arange(601, 661)
        axes[0].plot(w, G[w], color=col, ls=ls, lw=1.0, marker=mk, ms=2.6,
                     label=lab)
        Ls = np.arange(100, Lmax + 1)
        run = np.minimum.accumulate(G[Ls])
        step = 150
        axes[1].plot(Ls[::step], run[::step], color=col, ls=ls, lw=1.3,
                     marker=mk, ms=3, markevery=50, label=lab)
    for ax, xl, yl in [(axes[0], "$L$", "$G_S(L)$"),
                       (axes[1], "$L$", "running minimum of $G_S$")]:
        ax.axhline(0.0, color="#333333", lw=1.0)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[0].set_title("window $601\\leq L\\leq 660$", fontsize=9)
    axes[1].set_title("running minimum over $[100,L]$", fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_yticks([-1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1])
    axes[0].legend(fontsize=7, loc="upper right", ncol=1)
    axes[1].legend(fontsize=7, loc="center left")
    fig.tight_layout()
    save_fig(fig, "fig_gprofile")


def fig_ratio(ratio_series):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    picks = [("[(1, 0), (0, 1)]", "$S=\\{(1,0),(0,1)\\}$, $k=2$", 300),
             ("[(0, 0), (1, 0), (0, 1)]",
              "$S=\\{\\mathbf{0},(1,0),(0,1)\\}$, $k=2$", 300)]
    for ax, (key, lab, Lcut) in zip(axes, picks):
        rs, D = ratio_series[key]
        Ls = [x for (x, _) in rs if 3 <= x <= Lcut]
        ys = [y for (x, y) in rs if 3 <= x <= Lcut]
        ax.plot(Ls, ys, color="#0072B2", lw=0.9, label="$N_S(L)/L^k$")
        ax.axhline(D, color="#D55E00", lw=1.1, ls="--", label="$\\delta(S)$")
        defs = [(x, y) for (x, y) in zip(Ls, ys) if y < D]
        if defs:
            ax.plot([d[0] for d in defs], [d[1] for d in defs], ".",
                    color="#D55E00", ms=3.2, label="deficient $L$")
        span = max(abs(y - D) for y in ys[max(0, len(ys) // 20):])
        ax.set_ylim(D - 1.5 * span, D + 1.5 * span)
        ax.set_xlabel("$L$")
        ax.set_ylabel("$N_S(L)/L^k$")
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "fig_ratio")


def fig_sieve_time(rows):
    ts = sorted(set(r["t"] for r in rows))
    fig, axes = plt.subplots(1, len(ts), figsize=(7.4, 2.35), sharey=True)
    handles = None
    for ax, t in zip(axes, ts):
        sub = [r for r in rows if r["t"] == t]
        xs = [r["r"] for r in sub]
        for key, col in [("line", "t_line"), ("plane", "t_plane"),
                         ("ntv", "t_ntv"), ("fib", "t_fib")]:
            ax.loglog(xs, [max(r[col], 1e-5) for r in sub], marker=SMRK[key],
                      color=SCOL[key], ls=SLST[key], label=SLBL[key],
                      ms=3.8, lw=1.2)
        ax.set_title("$t=%d$" % t, fontsize=9)
        ax.set_xlabel("$r$")
        ax.tick_params(labelsize=7.5)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("time (s)")
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save_fig(fig, "fig_sieve_time")


def fig_sieve_work(rows, scal):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    ax = axes[0]
    sub = [r for r in rows if r["t"] == 6]
    xs = [r["r"] for r in sub]
    ax.loglog(xs, [r["w_line"] for r in sub], marker=SMRK["line"],
              color=SCOL["line"], ls=SLST["line"], label=SLBL["line"],
              ms=3.6, lw=1.1)
    ax.loglog(xs, [r["w_plane"] for r in sub], marker=SMRK["plane"],
              color=SCOL["plane"], ls=SLST["plane"], label=SLBL["plane"],
              ms=3.6, lw=1.1)
    ax.loglog(xs, [r["w_fib"] for r in sub], marker=SMRK["fib"],
              color=SCOL["fib"], ls=SLST["fib"], label=SLBL["fib"],
              ms=3.6, lw=1.1)
    ax.loglog(xs, [r["expected"] for r in sub], color="#444444", ls=":",
              lw=1.1, label="$|H|/r$")
    ax.set_xlabel("$r$")
    ax.set_ylabel("outer passes")
    ax.set_title("$t=6$, $|H|=%d$" % sub[0]["vol"], fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax = axes[1]
    xs = [r["vol"] for r in scal]
    ax.loglog(xs, [r["t_line"] for r in scal], marker=SMRK["line"],
              color=SCOL["line"], ls=SLST["line"], label=SLBL["line"],
              ms=3.6, lw=1.1)
    ax.loglog(xs, [r["t_plane"] for r in scal], marker=SMRK["plane"],
              color=SCOL["plane"], ls=SLST["plane"], label=SLBL["plane"],
              ms=3.6, lw=1.1)
    ax.loglog(xs, [r["t_fib"] for r in scal], marker=SMRK["fib"],
              color=SCOL["fib"], ls=SLST["fib"], label=SLBL["fib"],
              ms=3.6, lw=1.1)
    ax.set_xlabel("$|H|$")
    ax.set_ylabel("time (s)")
    ax.set_title("$t=4$, $r=%d$" % scal[0]["r"], fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "fig_sieve_work")


def fig_recall(rows, mem):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.35))
    ax = axes[0]
    for t, col, mk, ls in [(3, "#0072B2", "o", "-"), (4, "#D55E00", "s", "--"),
                           (5, "#009E73", "^", "-."), (6, "#CC79A7", "d", ":")]:
        sub = sorted([r for r in rows if r["t"] == t],
                     key=lambda r: r["level"])
        ax.plot([r["level"] for r in sub], [100.0 * r["recall"] for r in sub],
                marker=mk, color=col, ls=ls, ms=4.5, lw=1.3, label="$t=%d$" % t)
    ax.axhline(100.0, color="#333333", lw=1.1)
    ax.set_ylim(45, 108)
    ax.set_xlabel("level $\\ell$")
    ax.set_ylabel("relations found (\\%)")
    ax.set_title("completeness of the transition sieve", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right", ncol=2)
    ax = axes[1]
    labels = []
    vals = []
    for row in mem:
        if row["split"] == 0:
            labels.append("enumeration")
        elif row["blocks"] == 1:
            labels.append("one array")
        else:
            labels.append("%d blocks" % row["blocks"])
        vals.append(row["peak"] / 1024.0)
    ax.bar(range(len(vals)), vals, color="#0072B2", width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 1e5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("peak memory (KiB)")
    ax.set_title("relation collection, $t=4$, $|H|=2^{20}$", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_recall")


def fig_yield(series):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for (lab, (vals, d, col, ls)) in series.items():
        xs = np.array([x for (x, y) in vals])
        ys = np.array([y / d for (x, y) in vals])
        axes[0].plot(xs, ys, color=col, ls=ls, lw=1.0, label=lab)
        axes[1].plot(xs, np.minimum.accumulate(ys), color=col, ls=ls, lw=1.3,
                     label=lab)
    for ax, ttl, yl in [(axes[0], "yield relative to the prediction",
                         "$N_S(L)/(\\delta(S)L^{t})$"),
                        (axes[1], "worst case over all sides up to $L$",
                         "running minimum")]:
        ax.axhline(1.0, color="#333333", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlabel("$L$")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=9)
    axes[0].legend(fontsize=6.2, loc="lower right")
    axes[1].legend(fontsize=6.2, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "fig_yield")


def main():
    t0 = time.perf_counter()
    exp_correctness()
    plotdata, rows, mems = exp_timing()
    ratio_series, cert_rows = exp_sd()
    exp_bu()
    exp_sieve_validation()
    bench = exp_sieve_bench()
    scal = exp_scaling()
    mem = exp_memory()
    yrows, yseries, mc = exp_yield(cert_rows)
    exp_nodes()
    fig_timing(plotdata)
    fig_gprofile()
    fig_ratio(ratio_series)
    fig_sieve_time(bench)
    fig_sieve_work(bench, scal)
    fig_recall(bench, mem)
    fig_yield(yseries)
    log(f"total {time.perf_counter()-t0:.1f}s; tables in results/, "
        f"figures in figures/")


def figures_only():
    import ast
    def load(name):
        return [ast.literal_eval(l) for l in
                open(os.path.join(RESULTS, name)) if l.strip()]
    bench = load("e6_sieve_bench.txt")
    mem = load("e8_memory.txt")
    fig_sieve_time(bench)
    fig_recall(bench, mem)
    log("regenerated fig_sieve_time and fig_recall from results/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "figures":
        figures_only()
    else:
        main()
