"""Branch-and-bound for the minimum number of slots (DSATUR-style exact search).

Search tree : choose the uncoloured exam with maximum saturation (ties: max uncoloured degree);
              branch on every feasible existing slot plus ONE new slot (no symmetric duplicates).
Bounding    : a node with `used` slots is pruned when  max(used, LB_root) >= best  where
              LB_root = max(clique, seat-count, big-exam) bound (see bounds.py).
              The search stops as soon as best == LB_root (optimality certificate).
Incumbent   : initial upper bound from DSATUR (or supplied by the hybrid).
Anytime     : with a time limit it returns the best timetable found plus the optimality flag.
"""
from __future__ import annotations
from typing import List, Optional
from .bounds import lower_bound
from .common import Clock, Result, Timeout, compact
from .greedy import dsatur
from .instance import Instance


def solve_bnb(inst: Instance, time_limit: Optional[float] = None, node_limit: Optional[int] = None,
              initial: Optional[List[int]] = None, lb: Optional[int] = None,
              use_lower_bound: bool = True, clique_time: float = 1.0) -> Result:
    clock = Clock(time_limit)
    n = inst.n
    if n == 0:
        return Result("B&B", "optimal", [], 0, 0, 0.0, 0, True)
    if initial is None:
        initial = dsatur(inst)
    initial = compact(initial)
    best_k = max(initial) + 1
    best_sol = list(initial)
    if use_lower_bound:
        if lb is None:
            lb, _ = lower_bound(inst, clique_time)
    else:
        lb = 1
    trace = [(clock.elapsed(), best_k)]
    if best_k <= lb:
        return Result("B&B", "optimal", best_sol, best_k, 0, clock.elapsed(), lb, True, trace)

    cap = inst.capacity
    size = inst.size
    adjl = inst.adj_list
    K = best_k
    cnt = [[0] * K for _ in range(n)]            # cnt[v][c] = #neighbours of v in slot c
    sat = [0] * n
    udeg = list(inst.degree)
    slot_of = [-1] * n
    load = [0] * K
    st = {"nodes": 0, "best_k": best_k, "best": best_sol, "done": False}

    def assign(v: int, c: int) -> None:
        slot_of[v] = c; load[c] += size[v]
        for u in adjl[v]:
            if cnt[u][c] == 0:
                sat[u] += 1
            cnt[u][c] += 1
            udeg[u] -= 1

    def unassign(v: int, c: int) -> None:
        slot_of[v] = -1; load[c] -= size[v]
        for u in adjl[v]:
            cnt[u][c] -= 1
            if cnt[u][c] == 0:
                sat[u] -= 1
            udeg[u] += 1

    def rec(assigned: int, used: int) -> None:
        if st["done"]:
            return
        st["nodes"] += 1
        if (node_limit is not None and st["nodes"] > node_limit) or \
                ((st["nodes"] & 255) == 0 and clock.expired()):
            raise Timeout()
        if assigned == n:
            if used < st["best_k"]:
                st["best_k"] = used
                st["best"] = list(slot_of)
                trace.append((clock.elapsed(), used))
                if used <= lb:
                    st["done"] = True
            return
        # --- bounding: any completion needs >= max(used, lb) slots; need strictly fewer than best
        if max(used, lb) >= st["best_k"]:
            return
        # --- branching variable: max saturation, then max uncoloured degree
        v, bkey = -1, None
        for x in range(n):
            if slot_of[x] < 0:
                key = (sat[x], udeg[x])
                if bkey is None or key > bkey:
                    v, bkey = x, key
        cv = cnt[v]
        for c in range(used):
            if cv[c] == 0 and load[c] + size[v] <= cap:
                assign(v, c)
                rec(assigned + 1, used)
                unassign(v, c)
                if st["done"]:
                    return
        if used + 1 < st["best_k"] and used < K:       # open a new slot only if it can still improve
            assign(v, used)
            rec(assigned + 1, used + 1)
            unassign(v, used)

    status = "optimal"
    try:
        rec(0, 0)
    except Timeout:
        status = "feasible"
    optimal = (status == "optimal")
    return Result("B&B", status, st["best"], st["best_k"], st["nodes"], clock.elapsed(), lb, optimal, trace)
