"""Lower bounds on the number of slots (clique, seat-count, large-exam)."""
from __future__ import annotations
import math
import time
from typing import List, Optional, Tuple
from .instance import Instance


def greedy_clique(inst: Instance) -> List[int]:
    """Greedy clique: repeatedly add the highest-degree vertex adjacent to all chosen ones."""
    best: List[int] = []
    order = sorted(range(inst.n), key=lambda v: -inst.degree[v])
    for start in order[: min(len(order), 30)]:
        cl = [start]
        cand = inst.adj_mask[start]
        while cand:
            # pick candidate with most neighbours inside cand
            bestv, bestc = -1, -1
            c = cand
            while c:
                v = (c & -c).bit_length() - 1
                c &= c - 1
                cnt = bin(cand & inst.adj_mask[v]).count("1")
                if cnt > bestc:
                    bestv, bestc = v, cnt
            cl.append(bestv)
            cand &= inst.adj_mask[bestv]
        if len(cl) > len(best):
            best = cl
    return best


def max_clique(inst: Instance, time_limit: Optional[float] = 2.0) -> Tuple[List[int], bool]:
    """Exact maximum clique (Tomita-style branch and bound with greedy-colouring bound).

    Returns (clique, proven_optimal).  If the time limit is hit the clique found so far is
    still a valid lower bound on the chromatic number.
    """
    best = greedy_clique(inst)
    if inst.n == 0:
        return [], True
    t0 = time.perf_counter()
    adj = inst.adj_mask
    state = {"best": list(best), "timeout": False, "cnt": 0}

    def expand(R: List[int], P: int) -> None:
        if state["timeout"]:
            return
        state["cnt"] += 1
        if (state["cnt"] & 255) == 0 and time_limit is not None and time.perf_counter() - t0 > time_limit:
            state["timeout"] = True
            return
        # greedy colouring of P gives an upper bound on clique size within P
        order: List[int] = []
        bnd: List[int] = []
        Q = P
        colour = 0
        while Q:
            colour += 1
            Qc = Q
            while Qc:
                v = (Qc & -Qc).bit_length() - 1
                Qc &= ~(1 << v) & ~adj[v]
                Q &= ~(1 << v)
                order.append(v)
                bnd.append(colour)
        for i in range(len(order) - 1, -1, -1):
            if len(R) + bnd[i] <= len(state["best"]):
                return
            v = order[i]
            R.append(v)
            newP = P & adj[v]
            if newP:
                expand(R, newP)
            elif len(R) > len(state["best"]):
                state["best"] = list(R)
            R.pop()
            P &= ~(1 << v)
            if state["timeout"]:
                return

    expand([], (1 << inst.n) - 1)
    return state["best"], not state["timeout"]


def seat_bound(inst: Instance) -> int:
    """ceil(total seats / capacity): every slot holds at most `capacity` seats."""
    if not inst.venues or inst.n == 0:
        return 1 if inst.n else 0
    return max(1, math.ceil(sum(inst.size) / inst.capacity))


def big_exam_bound(inst: Instance) -> int:
    """Exams larger than capacity/2 can never share a slot: their count is a lower bound."""
    if not inst.venues:
        return 1 if inst.n else 0
    big = sum(1 for s in inst.size if 2 * s > inst.capacity)
    return max(big, 1 if inst.n else 0)


def lower_bound(inst: Instance, clique_time: Optional[float] = 2.0) -> Tuple[int, dict]:
    """Best available lower bound and its components."""
    if inst.n == 0:
        return 0, {}
    cl, exact = max_clique(inst, clique_time)
    parts = {"clique": len(cl), "clique_exact": exact, "seat": seat_bound(inst), "big": big_exam_bound(inst),
             "clique_set": cl}
    return max(parts["clique"], parts["seat"], parts["big"]), parts
