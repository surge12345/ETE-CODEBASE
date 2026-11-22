"""CSP solvers for the k-slot *decision* problem (is there a conflict-free timetable in k slots?).

Modes (cumulative):
  naive   : chronological backtracking, static order, consistency check only
  fc      : + forward checking (domain pruning of neighbours), static order
  fc_mrv  : + MRV variable ordering, degree-heuristic tie-break
  mac     : + maintained arc consistency (singleton propagation) for the != constraints
`symmetry=True` additionally enforces value-symmetry breaking (a variable may only take an
already-used slot or the lowest unused slot) - slots are interchangeable, so this is sound.

Nodes = number of consistent (partial) assignments explored (search-tree nodes).
Worst case O(k^n * m) for naive; FC adds O(deg) work per node but prunes dead subtrees early.
"""
from __future__ import annotations
import time
from typing import List, Optional
from .common import Clock, Result, Timeout, num_slots_used
from .instance import Instance

MODES = ("naive", "fc", "fc_mrv", "mac")


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def solve_k(inst: Instance, k: int, mode: str = "fc_mrv", symmetry: Optional[bool] = None,
            node_limit: Optional[int] = None, time_limit: Optional[float] = None) -> Result:
    if mode not in MODES:
        raise ValueError(mode)
    if symmetry is None:
        symmetry = mode != "naive"
    n = inst.n
    clock = Clock(time_limit)
    name = {"naive": "Naive BT", "fc": "BT+FC", "fc_mrv": "BT+FC+MRV+deg", "mac": "MAC+MRV+deg"}[mode]
    if symmetry and mode == "naive":
        name += "+sym"
    if n == 0:
        return Result(name, "sat", [], 0, 0, 0.0)
    if k <= 0:
        return Result(name, "unsat", None, 0, 0, 0.0)
    cap = inst.capacity
    size = inst.size
    adjm = inst.adj_mask
    adjl = inst.adj_list
    slot_of = [-1] * n
    load = [0] * k
    stats = {"nodes": 0}
    full = (1 << k) - 1

    def tick() -> None:
        stats["nodes"] += 1
        if (node_limit is not None and stats["nodes"] > node_limit) or \
                ((stats["nodes"] & 1023) == 0 and clock.expired()):
            raise Timeout()

    if mode == "naive":
        slotmask = [0] * k      # bitmask of exams in each slot

        def bt(i: int, used: int) -> bool:
            if i == n:
                return True
            lim = min(k, used + 1) if symmetry else k
            for c in range(lim):
                if slotmask[c] & adjm[i] or load[c] + size[i] > cap:
                    continue
                tick()
                slot_of[i] = c; slotmask[c] |= 1 << i; load[c] += size[i]
                if bt(i + 1, max(used, c + 1)):
                    return True
                slot_of[i] = -1; slotmask[c] &= ~(1 << i); load[c] -= size[i]
            return False

        try:
            ok = bt(0, 0)
            status = "sat" if ok else "unsat"
        except Timeout:
            ok, status = False, "timeout"
        return Result(name, status, list(slot_of) if ok else None, num_slots_used(slot_of) if ok else 0,
                      stats["nodes"], clock.elapsed())

    dom = [full] * n
    udeg = list(inst.degree)       # number of unassigned neighbours
    maxsize_unassigned = max(size)
    use_mrv = mode in ("fc_mrv", "mac")
    use_mac = mode == "mac"
    static_order = list(range(n))

    def select(assigned: int) -> int:
        if not use_mrv:
            return static_order[assigned]
        best, bkey = -1, None
        for v in range(n):
            if slot_of[v] >= 0:
                continue
            key = (bin(dom[v]).count("1"), -udeg[v])
            if bkey is None or key < bkey:
                best, bkey = v, key
        return best

    def remove(u: int, c: int, trail: list, queue: list) -> bool:
        d = dom[u]
        if (d >> c) & 1:
            trail.append((u, d))
            d &= ~(1 << c)
            dom[u] = d
            if d == 0:
                return False
            if use_mac and (d & (d - 1)) == 0:
                queue.append(u)
        return True

    def propagate(v: int, c: int, trail: list) -> bool:
        queue: List[int] = []
        for u in adjl[v]:
            if slot_of[u] < 0 and not remove(u, c, trail, queue):
                return False
        rem = cap - load[c]
        if rem < maxsize_unassigned:               # capacity pruning (C2)
            for u in range(n):
                if slot_of[u] < 0 and size[u] > rem and not remove(u, c, trail, queue):
                    return False
        while queue:                                # MAC: singleton propagation
            u = queue.pop()
            d = dom[u]
            if d == 0 or (d & (d - 1)) != 0:
                continue
            val = d.bit_length() - 1
            for w in adjl[u]:
                if slot_of[w] < 0 and w != u and not remove(w, val, trail, queue):
                    return False
        return True

    def bt2(assigned: int, used: int) -> bool:
        nonlocal maxsize_unassigned
        if assigned == n:
            return True
        v = select(assigned)
        d = dom[v]
        if symmetry:
            d &= (1 << min(k, used + 1)) - 1
        for c in _bits(d):
            if load[c] + size[v] > cap:
                continue
            tick()
            trail: list = []
            slot_of[v] = c; load[c] += size[v]
            for u in adjl[v]:
                udeg[u] -= 1
            ok = propagate(v, c, trail)
            if ok and bt2(assigned + 1, max(used, c + 1)):
                return True
            for u, dd in reversed(trail):
                dom[u] = dd
            for u in adjl[v]:
                udeg[u] += 1
            slot_of[v] = -1; load[c] -= size[v]
        return False

    try:
        ok = bt2(0, 0)
        status = "sat" if ok else "unsat"
    except Timeout:
        ok, status = False, "timeout"
    return Result(name, status, list(slot_of) if ok else None, num_slots_used(slot_of) if ok else 0,
                  stats["nodes"], clock.elapsed())


def minimise_with_csp(inst: Instance, lb: int, ub: int, mode: str = "fc_mrv",
                      node_limit: Optional[int] = None, time_limit: Optional[float] = None) -> Result:
    """Find the optimum by solving the decision problem for k = lb, lb+1, ... ub-1 (ub assumed feasible)."""
    clock = Clock(time_limit)
    total_nodes = 0
    last = None
    for k in range(max(1, lb), ub):
        rem_nodes = None if node_limit is None else max(1, node_limit - total_nodes)
        r = solve_k(inst, k, mode, node_limit=rem_nodes, time_limit=clock.remaining())
        total_nodes += r.nodes
        last = r
        if r.status == "sat":
            return Result(r.algorithm, "optimal", r.slot_of, r.num_slots, total_nodes, clock.elapsed(), lb, True)
        if r.status == "timeout":
            return Result(r.algorithm, "timeout", None, 0, total_nodes, clock.elapsed(), lb, False)
    return Result(last.algorithm if last else mode, "ub_optimal", None, ub, total_nodes, clock.elapsed(), lb, True)
