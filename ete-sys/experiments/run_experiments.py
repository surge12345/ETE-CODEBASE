"""Run all experiments and write CSVs to results/.

Every random choice is seeded, so a run is reproducible.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ete.bnb import solve_bnb
from ete.bounds import lower_bound
from ete.csp import solve_k
from ete.generators import crown_bad_order, crown_graph, random_instance, uz_instance
from ete.greedy import first_fit, make_order, solve_greedy
from ete.hybrid import solve_hybrid
from ete.validate import count_conflicts, validate_assignment


def gen(kind: str, n: int, seed: int, density: float = 0.12):
    if kind == "random":
        return random_instance(n, density=density, seed=seed)
    if kind == "uz":
        return uz_instance(n, seed=seed)
    raise ValueError(kind)


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)", flush=True)


def exp_pruning(out, quick):
    """Decision problem at k* (satisfiable) and k*-1 (unsatisfiable) for each CSP variant."""
    sizes = [10, 15, 20] if quick else [10, 15, 20, 25, 30, 40, 50, 60]
    seeds = 3 if quick else 6
    tl = 2.0 if quick else 3.0
    modes = [("naive", False), ("naive", True), ("fc", True), ("fc_mrv", True), ("mac", True)]
    rows = []
    for kind, dens in (("random", 0.3), ("uz", 0.0)):
        for n in sizes:
            for seed in range(seeds):
                inst = gen(kind, n, seed, dens if dens else 0.12)
                opt = solve_bnb(inst, time_limit=20)
                if not opt.optimal:
                    continue
                kstar = opt.num_slots
                for mode, sym in modes:
                    if mode == "naive" and n > 20:
                        continue
                    if mode == "fc" and n > 40:
                        continue
                    for kind_q, k in (("sat", kstar), ("unsat", kstar - 1)):
                        if k < 1:
                            continue
                        r = solve_k(inst, k, mode, symmetry=sym, time_limit=tl)
                        if r.status == "sat":
                            ok, _ = validate_assignment(inst, r.slot_of)
                            assert ok
                        rows.append(dict(dataset=kind, n=n, seed=seed, mode=mode + ("+sym" if (mode == "naive" and sym) else ""),
                                         query=kind_q, k=k, status=r.status, nodes=r.nodes, time=round(r.time, 5)))
        print(f"  pruning: {kind} done", flush=True)
    write(os.path.join(out, "exp1_pruning.csv"), rows)


def exp_scaling(out, quick):
    sizes = [10, 25, 50, 100] if quick else [10, 25, 50, 100, 200, 300, 500]
    T = 1.0 if quick else 5.0
    rows = []
    for kind in ("random", "uz"):
        for n in sizes:
            seeds = (3 if quick else 10) if n <= 100 else (2 if quick else 5)
            for seed in range(seeds):
                inst = gen(kind, n, seed)
                lb, parts = lower_bound(inst, 2.0)
                base = dict(dataset=kind, n=n, seed=seed, m=inst.m, density=round(inst.density, 4), lb=lb)
                for label, method in (("FirstFit", "input"), ("Welsh-Powell", "degree"), ("DSATUR", "dsatur")):
                    r = solve_greedy(inst, method)
                    ok, _ = validate_assignment(inst, r.slot_of)
                    rows.append(dict(base, algo=label, slots=r.num_slots, time=round(r.time, 5), nodes=r.nodes,
                                     optimal=False, valid=ok, conflicts=count_conflicts(inst, r.slot_of)))
                b = solve_bnb(inst, time_limit=T, lb=lb)
                ok, _ = validate_assignment(inst, b.slot_of)
                rows.append(dict(base, algo="B&B", slots=b.num_slots, time=round(b.time, 4), nodes=b.nodes,
                                 optimal=b.optimal, valid=ok, conflicts=count_conflicts(inst, b.slot_of)))
                h = solve_hybrid(inst, time_limit=T, seed=seed)
                ok, _ = validate_assignment(inst, h.slot_of)
                rows.append(dict(base, algo="Hybrid", slots=h.num_slots, time=round(h.time, 4), nodes=h.nodes,
                                 optimal=h.optimal, valid=ok, conflicts=count_conflicts(inst, h.slot_of)))
            print(f"  scaling: {kind} n={n} done", flush=True)
    write(os.path.join(out, "exp2_scaling.csv"), rows)


def exp_density(out, quick):
    """Effect of conflict density on greedy vs B&B vs hybrid (n = 100)."""
    dens = [0.05, 0.2, 0.5] if quick else [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    seeds = 3 if quick else 5
    T = 1.0 if quick else 4.0
    n = 60 if quick else 100
    rows = []
    for d in dens:
        for seed in range(seeds):
            inst = random_instance(n, density=d, seed=seed)
            lb, _ = lower_bound(inst, 2.0)
            base = dict(density_param=d, density=round(inst.density, 3), n=n, seed=seed, lb=lb)
            ds = solve_greedy(inst, "dsatur")
            rows.append(dict(base, algo="DSATUR", slots=ds.num_slots, time=ds.time, optimal=False))
            b = solve_bnb(inst, time_limit=T, lb=lb)
            rows.append(dict(base, algo="B&B", slots=b.num_slots, time=b.time, optimal=b.optimal))
            h = solve_hybrid(inst, time_limit=T, seed=seed)
            rows.append(dict(base, algo="Hybrid", slots=h.num_slots, time=h.time, optimal=h.optimal))
        print(f"  density {d} done", flush=True)
    write(os.path.join(out, "exp3_density.csv"), rows)


def exp_bounds(out, quick):
    """Pruning effectiveness of the lower bounds inside B&B (with vs without root lower bound)."""
    sizes = [20, 30, 40] if quick else [20, 30, 40, 50, 60, 80]
    seeds = 3 if quick else 8
    rows = []
    for n in sizes:
        for seed in range(seeds):
            inst = random_instance(n, density=0.3, seed=seed)
            for use_lb in (False, True):
                r = solve_bnb(inst, time_limit=5 if not quick else 2, use_lower_bound=use_lb)
                rows.append(dict(n=n, seed=seed, lower_bound_on=use_lb, nodes=r.nodes, time=round(r.time, 4),
                                 slots=r.num_slots, optimal=r.optimal))
    write(os.path.join(out, "exp4_bounds.csv"), rows)


def exp_greedy_worst(out, quick):
    """Greedy worst case: crown graphs (chromatic number 2)."""
    rows = []
    for h in ([2, 5, 10, 20] if quick else [2, 3, 5, 8, 10, 15, 20, 30, 40, 50]):
        inst = crown_graph(h)
        bad = first_fit(inst, crown_bad_order(h))
        ds = solve_greedy(inst, "dsatur")
        wp = solve_greedy(inst, "degree")
        rows.append(dict(h=h, n=2 * h, optimum=2, firstfit_adversarial=max(bad) + 1, welsh_powell=wp.num_slots,
                         dsatur=ds.num_slots, ratio_firstfit=(max(bad) + 1) / 2))
    write(os.path.join(out, "exp5_crown.csv"), rows)


def exp_traces(out, quick):
    """Anytime behaviour: best-so-far slots vs time, B&B vs hybrid, on one hard instance."""
    T = 3.0 if quick else 10.0
    inst = random_instance(80 if quick else 120, density=0.5, seed=1)
    rows = []
    b = solve_bnb(inst, time_limit=T)
    h = solve_hybrid(inst, time_limit=T, seed=1)
    for name, r in (("B&B", b), ("Hybrid", h)):
        for t, k in r.trace:
            rows.append(dict(algo=name, t=round(t, 4), slots=k))
        rows.append(dict(algo=name, t=T, slots=r.num_slots))
    rows.append(dict(algo="lower bound", t=0, slots=b.lower_bound))
    write(os.path.join(out, "exp6_trace.csv"), rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small/fast run (~1-2 min)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    exps = {"pruning": exp_pruning, "bounds": exp_bounds, "crown": exp_greedy_worst,
            "scaling": exp_scaling, "density": exp_density, "trace": exp_traces}
    t0 = time.time()
    for name, fn in exps.items():
        if a.only and name not in a.only:
            continue
        print(f"[{name}]", flush=True)
        fn(a.out, a.quick)
    print(f"all experiments finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
