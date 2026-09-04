import ast
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    out = []
    with open(os.path.join(RESULTS, name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("MC "):
                out.append(("MC", ast.literal_eval(line[3:])))
            elif line.startswith("MEM "):
                out.append(("MEM", ast.literal_eval(line[4:])))
            else:
                out.append(("", ast.literal_eval(line)))
    return out


def fmt(x, d=4):
    return ("%." + str(d) + "f") % x


def table_bench():
    rows = [r for (_, r) in load("e6_sieve_bench.txt")]
    out = []
    for r in rows:
        if r["level"] < 1:
            continue
        sp = min(r["t_line"], r["t_plane"], r["t_ntv"]) / max(r["t_fib"], 1e-9)
        out.append("%d & %d & %d & %d & %s & %s & %s & %s & %.0f & %.3f \\\\"
                   % (r["t"], r["r"], r["level"], round(r["pts"]),
                      fmt(r["t_line"], 4), fmt(r["t_plane"], 4),
                      fmt(r["t_ntv"], 4), fmt(r["t_fib"], 4), sp, r["recall"]))
    return out


def table_nodes():
    rows = [r for (_, r) in load("e10_nodes.txt")]
    out = []
    for r in rows:
        out.append("%d & %d & %d & %.0f & %.0f & %.2f & %.0f & %d/%d & %.2f \\\\"
                   % (r["t"], r["r"], r["level"], r["expected"], r["nodes"],
                      r["nodes_per_expected"], r["bound"], r["shape_ok"],
                      r["shape_reps"], r["worst_ratio"]))
    return out


def table_scaling():
    rows = [r for (_, r) in load("e7_scaling.txt")]
    out = []
    for r in rows:
        out.append("%d & %d & %d & %.1f & %.0f & %.0f & %s & %s & %s \\\\"
                   % (2 * r["side"], r["vol"], r["level"], r["expected"],
                      r["pts"], r["slices"], fmt(r["t_line"], 4),
                      fmt(r["t_plane"], 4), fmt(r["t_fib"], 5)))
    return out


def table_memory():
    rows = [r for (_, r) in load("e8_memory.txt")]
    out = []
    for r in rows:
        if r["split"] == 0:
            out.append("enumeration only & --- & --- & %d & --- \\\\"
                       % r["peak"])
        else:
            out.append("%d & %d & %d & %d & %.1f \\\\"
                       % (r["split"], r["blocks"], r["cells"], r["peak"],
                          r["time"]))
    return out


def table_yield():
    rows = []
    for tag, r in load("e9_yield.txt"):
        if tag == "MC":
            continue
        rows.append(r)
    out = []
    for r in rows:
        S = ",".join("(" + ",".join(str(c) for c in v) + ")" for v in r["S"])
        out.append("$\\{%s\\}$ & %d & %.7f & %s & %.7f & %d & %d & %.2f \\\\"
                   % (S, r["k"], r["delta"], r["worst_frac"], r["worst"],
                      r["L_min"], r["L1"], 100.0 * r["shortfall"]))
    return out


def table_mc():
    out = []
    for tag, r in load("e9_yield.txt"):
        if tag != "MC":
            continue
        S = ",".join("(" + ",".join(str(c) for c in v) + ")" for v in r["S"])
        out.append("$\\{%s\\}$ & %d & %d & %d & %.6f & %.6f & %s & %s & %.0f \\\\"
                   % (S, r["k"], r["L"], r["N"], r["ratio"], r["delta"],
                      fmt(r["t_fiber"], 3), fmt(r["t_enum"], 3), r["speedup"]))
    return out


def table_counting():
    rows = [r for (_, r) in load("e2_timing.txt") if isinstance(r, dict)]
    out = []
    for r in rows:
        def g(k):
            return "---" if r[k] is None else fmt(r[k], 3)
        out.append("%d & %d & %s & %s & %s & %d \\\\"
                   % (r["k"], r["L"], g("enum"), g("mark"), g("fiber"),
                      r["N"]))
    return out


def main():
    blocks = [("bench", table_bench()), ("nodes", table_nodes()),
              ("scaling", table_scaling()), ("memory", table_memory()),
              ("yield", table_yield()), ("mc", table_mc()),
              ("counting", table_counting())]
    with open(os.path.join(RESULTS, "tables.tex"), "w") as f:
        for name, rows in blocks:
            f.write("%% ---- " + name + " ----\n")
            for r in rows:
                f.write(r + "\n")
            f.write("\n")
    for name, rows in blocks:
        print("---- " + name + " ----")
        for r in rows:
            print(r)


if __name__ == "__main__":
    main()
