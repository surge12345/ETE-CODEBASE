"""Hybrid B&B + local search.

Phase 1  DSATUR gives an incumbent (upper bound UB); root lower bound LB (clique/seat/big).
Phase 1b A node-limited B&B probe settles easy instances immediately.
Phase 2  Local search (TabuCol) repeatedly tries to find a timetable with UB-1 slots,
         starting from the incumbent with one slot dissolved. Each success lowers UB.
Phase 3  Branch-and-bound is seeded with the improved UB and the root LB; because UB is
         now tight, the bounding rule `max(used, LB) >= UB` prunes far more of the tree,
         and B&B can either prove optimality or improve the incumbent exactly.
The two components cooperate: LS supplies strong incumbents (primal side), B&B supplies
proofs/improvements (dual side). Total time is split by `ls_fraction`.
"""
from __future__ import annotations
from typing import Optional
from .bnb import solve_bnb
from .bounds import lower_bound
from .common import Clock, Result, compact, num_slots_used
from .greedy import dsatur
from .instance import Instance
from .local_search import tabucol


def solve_hybrid(inst: Instance, time_limit: float = 10.0, ls_fraction: float = 0.5, seed: int = 0,
                 ls_iters: int = 20000, probe_nodes: int = 3000) -> Result:
    clock = Clock(time_limit)
    if inst.n == 0:
        return Result("Hybrid B&B+LS", "optimal", [], 0, 0, 0.0, 0, True)
    sol = compact(dsatur(inst))
    ub = max(sol) + 1
    lb, parts = lower_bound(inst, min(1.0, time_limit * 0.1))
    trace = [(clock.elapsed(), ub)]
    # Phase 1b: cheap B&B probe - many instances are solved (or certified) within a few thousand nodes
    probe = solve_bnb(inst, node_limit=probe_nodes, initial=sol, lb=lb)
    if probe.optimal:
        trace.append((clock.elapsed(), probe.num_slots))
        return Result("Hybrid B&B+LS", "optimal", probe.slot_of, probe.num_slots, probe.nodes,
                      clock.elapsed(), lb, True, trace, extra={"ls_attempts": 0, "probe_solved": True})
    if probe.num_slots < ub:
        sol, ub = compact(probe.slot_of), probe.num_slots
        trace.append((clock.elapsed(), ub))
    ls_deadline = time_limit * ls_fraction
    attempt = 0
    while ub > lb and clock.elapsed() < ls_deadline:
        target = ub - 1
        found = tabucol(inst, target, init=sol, max_iters=ls_iters,
                        time_limit=max(0.01, ls_deadline - clock.elapsed()), seed=seed + attempt)
        attempt += 1
        if found is not None:
            sol = compact(found)
            ub = num_slots_used(sol)
            trace.append((clock.elapsed(), ub))
        else:
            if attempt >= 2:                       # LS stagnated on this target
                break
    t_bnb0 = clock.elapsed()
    res = solve_bnb(inst, time_limit=clock.remaining(), initial=sol, lb=lb)
    for t, kk in res.trace[1:]:
        trace.append((t_bnb0 + t, kk))
    best = res.slot_of
    k = res.num_slots
    trace.append((clock.elapsed(), k))
    return Result("Hybrid B&B+LS", "optimal" if res.optimal else "feasible", best, k, res.nodes,
                  clock.elapsed(), lb, res.optimal, trace, extra={"ls_attempts": attempt, **{kk: parts[kk] for kk in ("clique", "seat", "big")}})
