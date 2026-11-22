"""Statistics, tables (results/summary.md) and figures (results/figures/*.png).
"""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {"Naive BT": "#c0392b", "naive": "#c0392b", "naive+sym": "#e67e22", "fc": "#f1c40f", "fc_mrv": "#27ae60",
       "mac": "#2980b9", "FirstFit": "#95a5a6", "Welsh-Powell": "#7f8c8d", "DSATUR": "#8e44ad",
       "B&B": "#2980b9", "Hybrid": "#16a085"}
LABEL = {"naive": "Naive BT", "naive+sym": "Naive BT + symmetry", "fc": "BT + FC", "fc_mrv": "BT + FC + MRV + degree",
         "mac": "MAC + MRV + degree"}


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def md(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in r.values) + " |")
    return "\n".join(out)


def wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if np.allclose(d, 0):
        return float("nan"), 1.0, len(a)
    s, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(s), float(p), len(a)


def savefig(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def analyse_pruning(R, F, S):
    p = pd.read_csv(os.path.join(R, "exp1_pruning.csv"))
    S.append("## Experiment 1 - Pruning effectiveness of CSP search variants\n")
    S.append(f"Decision problem at k* (satisfiable) and k*-1 (unsatisfiable); {p.seed.nunique()} seeds per size; "
             f"{len(p)} runs. 'solved' = finished within the limit.\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, q in zip(axes, ("sat", "unsat")):
        sub = p[p["query"] == q]
        for mode in ["naive", "naive+sym", "fc", "fc_mrv", "mac"]:
            g = sub[sub["mode"] == mode].groupby("n")["nodes"].median()
            if len(g):
                ax.plot(g.index, g.values, "o-", color=COL[mode], label=LABEL[mode])
        ax.set_yscale("log"); ax.set_xlabel("number of exams n")
        ax.set_title("feasible k = k*" if q == "sat" else "infeasible k = k*-1 (proof of unsatisfiability)")
    axes[0].set_ylabel("median search nodes (log)"); axes[0].legend(fontsize=7)
    savefig(fig, os.path.join(F, "fig_pruning_nodes.png"))

    tab = p.groupby(["mode", "n"]).agg(median_nodes=("nodes", "median"), solved=("status", lambda s: (s != "timeout").mean())).reset_index()
    piv = tab.pivot(index="n", columns="mode", values="median_nodes")
    piv = piv[[c for c in ["naive", "naive+sym", "fc", "fc_mrv", "mac"] if c in piv.columns]]
    S.append("**Median search nodes (sat and unsat queries pooled)**\n")
    S.append(md(piv.reset_index(), "{:,.0f}") + "\n")
    sv = tab.pivot(index="n", columns="mode", values="solved")
    sv = sv[[c for c in ["naive", "naive+sym", "fc", "fc_mrv", "mac"] if c in sv.columns]]
    S.append("**Fraction of runs solved within the time limit**\n")
    S.append(md(sv.reset_index(), "{:.2f}") + "\n")

    # pruning effectiveness vs naive, on small n where naive mostly finishes
    small = p[p["n"] <= 20]
    key = ["dataset", "n", "seed", "query"]
    base = small[small["mode"] == "naive"].set_index(key)["nodes"]
    rows = []
    for mode in ["naive+sym", "fc", "fc_mrv", "mac"]:
        cur = small[small["mode"] == mode].set_index(key)["nodes"]
        j = pd.concat([base.rename("b"), cur.rename("c")], axis=1, join="inner").dropna()
        red = 100 * (1 - j["c"].sum() / j["b"].sum())
        per = 100 * (1 - j["c"] / j["b"])
        s, pv, nn = wilcoxon(j["b"], j["c"])
        rows.append(dict(variant=LABEL[mode], pruning_pct=red, mean_per_instance_pct=per.mean(),
                         ci95=ci95(per), wilcoxon_p=pv, pairs=nn))
    pr = pd.DataFrame(rows)
    S.append("**Pruning effectiveness vs naive backtracking (instances with n <= 20; reduction of total nodes, "
             "mean per-instance reduction with 95% CI, paired Wilcoxon signed-rank test)**\n")
    S.append(md(pr, "{:.4g}") + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(pr["variant"], pr["pruning_pct"], color=[COL[m] for m in ["naive+sym", "fc", "fc_mrv", "mac"]])
    ax.set_ylim(0, 100.5); ax.set_ylabel("% fewer nodes than naive BT")
    for i, v in enumerate(pr["pruning_pct"]):
        ax.text(i, min(v + 1, 97), f"{v:.2f}%", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    savefig(fig, os.path.join(F, "fig_pruning_pct.png"))



def analyse_bounds(R, F, S):
    b = pd.read_csv(os.path.join(R, "exp4_bounds.csv"))
    S.append("## Experiment 4 - Effect of the root lower bound inside B&B\n")
    g = b.groupby(["n", "lower_bound_on"]).agg(median_nodes=("nodes", "median"), mean_time=("time", "mean"),
                                               solved=("optimal", "mean")).reset_index()
    S.append(md(g, "{:,.3f}") + "\n")
    on = b[b["lower_bound_on"]].set_index(["n", "seed"])
    off = b[~b["lower_bound_on"]].set_index(["n", "seed"])
    j = pd.concat([off["nodes"].rename("off"), on["nodes"].rename("on")], axis=1).dropna()
    s, pv, nn = wilcoxon(j["off"], j["on"])
    red = 100 * (1 - j["on"].sum() / j["off"].sum())
    S.append(f"Total node reduction from lower bound: **{red:.1f}%** (Wilcoxon p = {pv:.3g}, {nn} paired instances).\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for flag, c, lab in ((False, "#c0392b", "no root lower bound"), (True, "#2980b9", "clique + seat + big-exam bound")):
        gg = b[b["lower_bound_on"] == flag].groupby("n")["nodes"].median()
        ax.plot(gg.index, gg.values, "o-", color=c, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("median B&B nodes"); ax.legend()
    savefig(fig, os.path.join(F, "fig_bounds.png"))


def analyse_scaling(R, F, S):
    d = pd.read_csv(os.path.join(R, "exp2_scaling.csv"))
    d["ratio_lb"] = d["slots"] / d["lb"]
    S.append("## Experiment 2 - Scalability and solution quality (n = 10 ... 500)\n")
    S.append(f"{len(d)} runs; all returned timetables passed the independent validator: "
             f"**{bool(d['valid'].all())}**; total conflicts in returned timetables: **{int(d['conflicts'].sum())}**.\n")
    algos = ["FirstFit", "Welsh-Powell", "DSATUR", "B&B", "Hybrid"]
    for metric, ttl, fname, logy in (("time", "wall-clock time (s)", "fig_time_vs_n.png", True),
                                      ("slots", "slots used", "fig_slots_vs_n.png", False),
                                      ("ratio_lb", "slots / lower bound", "fig_ratio_vs_n.png", False)):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
        for ax, ds in zip(axes, ("random", "uz")):
            for a in algos:
                sub = d[(d["dataset"] == ds) & (d["algo"] == a)]
                g = sub.groupby("n")[metric].agg(["mean", lambda x: ci95(x)])
                g.columns = ["mean", "ci"]
                ax.errorbar(g.index, g["mean"], yerr=g["ci"], marker="o", ms=3, capsize=2, color=COL[a], label=a)
            ax.set_xlabel("number of exams n"); ax.set_title({"random": "random cohort model", "uz": "UZ-inspired model"}[ds])
            if logy:
                ax.set_yscale("log"); ax.set_xscale("log")
        axes[0].set_ylabel(ttl); axes[0].legend(fontsize=7)
        savefig(fig, os.path.join(F, fname))

    for ds in ("random", "uz"):
        sub = d[d["dataset"] == ds]
        rows = []
        for n in sorted(sub["n"].unique()):
            r = {"n": n, "seeds": sub[(sub.n == n) & (sub.algo == "DSATUR")].shape[0],
                 "mean_LB": sub[(sub.n == n) & (sub.algo == "DSATUR")]["lb"].mean()}
            for a in algos:
                x = sub[(sub.n == n) & (sub.algo == a)]
                r[a] = f"{x['slots'].mean():.1f} +/- {ci95(x['slots']):.1f}"
            for a in ("B&B", "Hybrid"):
                x = sub[(sub.n == n) & (sub.algo == a)]
                r[a + " proven opt."] = f"{100 * x['optimal'].mean():.0f}%"
            for a in ("B&B", "Hybrid"):
                x = sub[(sub.n == n) & (sub.algo == a)]
                r[a + " time (s)"] = f"{x['time'].mean():.2f}"
            rows.append(r)
        S.append(f"**{ds} dataset: mean slots +/- 95% CI, share of instances solved to proven optimality, mean time**\n")
        S.append(md(pd.DataFrame(rows), "{:.1f}") + "\n")

    # greedy approximation ratio (vs lower bound = upper bound on true ratio)
    rows = []
    for a in ["FirstFit", "Welsh-Powell", "DSATUR", "B&B", "Hybrid"]:
        x = d[d["algo"] == a]
        rows.append(dict(algorithm=a, mean_ratio_to_LB=x["ratio_lb"].mean(), ci95=ci95(x["ratio_lb"]),
                         max_ratio_to_LB=x["ratio_lb"].max()))
    S.append("**Empirical approximation ratio against the lower bound (an upper bound on the true ratio), all instances**\n")
    S.append(md(pd.DataFrame(rows), "{:.3f}") + "\n")

    piv = d.pivot_table(index=["dataset", "n", "seed"], columns="algo", values="slots")
    big = piv[piv.index.get_level_values("n") >= 50]
    tests = []
    for a, b in (("DSATUR", "FirstFit"), ("Hybrid", "DSATUR"), ("B&B", "DSATUR"), ("Hybrid", "B&B")):
        s, pv, nn = wilcoxon(big[a], big[b])
        diff = (big[a] - big[b])
        tests.append(dict(comparison=f"{a} vs {b} (slots)", mean_diff=diff.mean(), ci95=ci95(diff), wilcoxon_p=pv, pairs=nn))
    big2 = piv[piv.index.get_level_values("n") >= 200]
    if len(big2) > 3:
        s, pv, nn = wilcoxon(big2["Hybrid"], big2["B&B"])
        diff = big2["Hybrid"] - big2["B&B"]
        tests.append(dict(comparison="Hybrid vs B&B (slots), n >= 200", mean_diff=diff.mean(), ci95=ci95(diff), wilcoxon_p=pv, pairs=nn))
    S.append("**Paired hypothesis tests (n >= 50; negative mean difference = first algorithm uses fewer slots)**\n")
    S.append(md(pd.DataFrame(tests), "{:.4g}") + "\n")

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for a in ("B&B", "Hybrid"):
        g = d[d["algo"] == a].groupby("n")["optimal"].mean() * 100
        ax.plot(g.index, g.values, "o-", color=COL[a], label=a)
    ax.set_xscale("log"); ax.set_xlabel("n"); ax.set_ylabel("% instances proven optimal"); ax.legend()
    savefig(fig, os.path.join(F, "fig_optimal_rate.png"))


def analyse_density(R, F, S):
    d = pd.read_csv(os.path.join(R, "exp3_density.csv"))
    S.append("## Experiment 3 - Effect of conflict density (random model, n = %d)\n" % d["n"].iloc[0])
    d["gap"] = d["slots"] - d["lb"]
    rows = []
    for dens in sorted(d["density_param"].unique()):
        r = {"density param": dens, "real density": d[d["density_param"] == dens]["density"].mean()}
        for a in ("DSATUR", "B&B", "Hybrid"):
            x = d[(d["density_param"] == dens) & (d["algo"] == a)]
            r[a + " gap to LB"] = f"{x['gap'].mean():.2f} +/- {ci95(x['gap']):.2f}"
        x = d[(d["density_param"] == dens) & (d["algo"] == "Hybrid")]
        r["Hybrid proven opt."] = f"{100 * x['optimal'].mean():.0f}%"
        rows.append(r)
    S.append(md(pd.DataFrame(rows), "{:.3f}") + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for a in ("DSATUR", "B&B", "Hybrid"):
        g = d[d["algo"] == a].groupby("density_param")["gap"].agg(["mean", lambda x: ci95(x)])
        g.columns = ["m", "c"]
        ax.errorbar(g.index, g["m"], yerr=g["c"], marker="o", capsize=2, color=COL[a], label=a)
    ax.set_xlabel("conflict-graph density"); ax.set_ylabel("slots - lower bound"); ax.legend()
    savefig(fig, os.path.join(F, "fig_density.png"))


def analyse_crown(R, F, S):
    c = pd.read_csv(os.path.join(R, "exp5_crown.csv"))
    S.append("## Experiment 5 - Worst case of greedy colouring (crown graphs, optimum = 2)\n")
    S.append(md(c, "{:.1f}") + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(c["n"], c["firstfit_adversarial"], "o-", color=COL["FirstFit"], label="First-Fit, adversarial order")
    ax.plot(c["n"], c["welsh_powell"], "s--", color=COL["Welsh-Powell"], label="Welsh-Powell")
    ax.plot(c["n"], c["dsatur"], "^-", color=COL["DSATUR"], label="DSATUR")
    ax.plot(c["n"], c["optimum"], "k:", label="optimum")
    ax.set_xlabel("vertices n = 2h"); ax.set_ylabel("slots used"); ax.legend(fontsize=7)
    savefig(fig, os.path.join(F, "fig_crown.png"))


def analyse_trace(R, F, S):
    t = pd.read_csv(os.path.join(R, "exp6_trace.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for a in ("B&B", "Hybrid"):
        x = t[t["algo"] == a].sort_values("t")
        ax.step(x["t"], x["slots"], where="post", color=COL[a], label=a, lw=2)
    lb = t[t["algo"] == "lower bound"]["slots"].iloc[0]
    ax.axhline(lb, color="k", ls=":", label=f"lower bound = {lb}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("best slots found"); ax.legend()
    ax.set_title("Anytime behaviour on a hard instance")
    savefig(fig, os.path.join(F, "fig_trace.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    R = a.results
    F = os.path.join(R, "figures")
    os.makedirs(F, exist_ok=True)
    S = ["# Experimental results (auto-generated by experiments/analyze.py)\n"]
    for fn in (analyse_pruning, analyse_scaling, analyse_density, analyse_bounds, analyse_crown, analyse_trace):
        try:
            fn(R, F, S)
        except FileNotFoundError as e:
            print("skipping", fn.__name__, e)
    with open(os.path.join(R, "summary.md"), "w") as f:
        f.write("\n".join(S))
    print("wrote", os.path.join(R, "summary.md"))


if __name__ == "__main__":
    main()
