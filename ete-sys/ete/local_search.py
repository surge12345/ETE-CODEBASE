"""TabuCol-style local search for the k-slot problem, extended with a soft capacity penalty.

State       : a complete assignment of exams to k slots (conflicts and overloads allowed).
Cost        : #conflicting pairs  +  (seats above capacity, summed over slots) / mean exam size.
Move        : re-slot one exam that is in conflict (or sits in an overloaded slot).
Tabu rule   : the (exam, old slot) pair is tabu for  0.6*cost + U[0,9]  iterations
              (Galinier-Hao tenure); aspiration if the move yields a new best cost.
Returns a *feasible* timetable (cost 0) or None.
"""
from __future__ import annotations
import random
import time
from typing import List, Optional
from .common import Clock
from .instance import Instance


def reduce_to_k(inst: Instance, slot_of: List[int], k: int, rng: random.Random) -> List[int]:
    """Make an assignment with <= k slots by dissolving the smallest slots (exams re-inserted
    into the slot with the fewest conflicts)."""
    cur = list(slot_of)
    used = sorted(set(cur))
    while len(used) > k:
        loads = {s: sum(inst.size[e] for e in range(inst.n) if cur[e] == s) for s in used}
        victim = min(used, key=lambda s: loads[s])
        used.remove(victim)
        for e in [e for e in range(inst.n) if cur[e] == victim]:
            best, bc = None, None
            for s in used:
                c = sum(1 for u in inst.adj_list[e] if cur[u] == s)
                c += rng.random() * 0.1
                if bc is None or c < bc:
                    best, bc = s, c
            cur[e] = best
    mp = {s: i for i, s in enumerate(sorted(set(cur)))}
    out = [mp[s] for s in cur]
    # pad if fewer than k slots (empty slots are fine for the search)
    return out


def tabucol(inst: Instance, k: int, init: Optional[List[int]] = None, max_iters: int = 200000,
            time_limit: Optional[float] = None, seed: int = 0) -> Optional[List[int]]:
    n = inst.n
    if n == 0:
        return []
    rng = random.Random(seed)
    clock = Clock(time_limit)
    cap = inst.capacity
    size = inst.size
    adjl = inst.adj_list
    mean_size = max(1.0, sum(size) / n)
    if init is None:
        init = [rng.randrange(k) for _ in range(n)]
    col = reduce_to_k(inst, init, k, rng) if max(init) + 1 > k else list(init)
    gamma = [[0] * k for _ in range(n)]
    for v in range(n):
        for u in adjl[v]:
            gamma[v][col[u]] += 1
    load = [0] * k
    for v in range(n):
        load[col[v]] += size[v]
    f = sum(gamma[v][col[v]] for v in range(n)) // 2

    def over(l: int) -> int:
        return l - cap if l > cap else 0

    ov = sum(over(l) for l in load)
    cost = f + ov / mean_size
    best_cost = cost
    tabu = [[0] * k for _ in range(n)]
    for it in range(1, max_iters + 1):
        if f == 0 and ov == 0:
            return col
        if (it & 63) == 0 and clock.expired():
            return None
        overloaded = {c for c in range(k) if load[c] > cap}
        bm, bd = None, None
        nb = 0
        for v in range(n):
            a = col[v]
            if gamma[v][a] == 0 and a not in overloaded:
                continue
            ga = gamma[v][a]
            sv = size[v]
            for b in range(k):
                if b == a:
                    continue
                d = gamma[v][b] - ga + (over(load[a] - sv) - over(load[a]) + over(load[b] + sv) - over(load[b])) / mean_size
                if tabu[v][b] > it and cost + d >= best_cost - 1e-9:
                    continue
                if bd is None or d < bd - 1e-12:
                    bm, bd, nb = (v, b), d, 1
                elif abs(d - bd) < 1e-12:
                    nb += 1
                    if rng.random() < 1.0 / nb:
                        bm = (v, b)
        if bm is None:                      # everything tabu: random conflicting move
            cands = [v for v in range(n) if gamma[v][col[v]] > 0 or col[v] in overloaded]
            if not cands:
                return col if (f == 0 and ov == 0) else None
            v = rng.choice(cands)
            b = rng.choice([c for c in range(k) if c != col[v]] or [col[v]])
            bm = (v, b)
        v, b = bm
        a = col[v]
        sv = size[v]
        f += gamma[v][b] - gamma[v][a]
        ov += over(load[a] - sv) - over(load[a]) + over(load[b] + sv) - over(load[b])
        load[a] -= sv; load[b] += sv
        col[v] = b
        for u in adjl[v]:
            gamma[u][a] -= 1
            gamma[u][b] += 1
        cost = f + ov / mean_size
        tabu[v][a] = it + int(0.6 * (f + ov / mean_size)) + rng.randint(0, 9)
        if cost < best_cost:
            best_cost = cost
    return None
