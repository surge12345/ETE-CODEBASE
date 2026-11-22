"""Greedy approximation algorithms: First-Fit (several orders), Welsh-Powell, DSATUR.

All of them respect the capacity constraint (C2): an exam goes to the lowest-numbered
slot with no conflicting exam and enough free seats.
Complexity: first-fit O(n + m + n*k) ; DSATUR O(n^2 + m).
"""
from __future__ import annotations
import random
import time
from typing import List, Optional, Sequence
from .common import Result, num_slots_used
from .instance import Instance


def first_fit(inst: Instance, order: Sequence[int]) -> List[int]:
    n = inst.n
    slot_of = [-1] * n
    load: List[int] = []
    cap = inst.capacity
    for v in order:
        forbidden = {slot_of[u] for u in inst.adj_list[v] if slot_of[u] >= 0}
        s = 0
        while True:
            if s == len(load):
                load.append(0)
            if s not in forbidden and load[s] + inst.size[v] <= cap:
                break
            s += 1
        slot_of[v] = s
        load[s] += inst.size[v]
    return slot_of


def make_order(inst: Instance, kind: str, seed: int = 0) -> List[int]:
    idx = list(range(inst.n))
    if kind == "input":
        return idx
    if kind == "degree":           # Welsh-Powell: non-increasing degree
        return sorted(idx, key=lambda v: (-inst.degree[v], v))
    if kind == "size":             # biggest exams first
        return sorted(idx, key=lambda v: (-inst.size[v], v))
    if kind == "random":
        rng = random.Random(seed)
        rng.shuffle(idx)
        return idx
    raise ValueError(kind)


def dsatur(inst: Instance) -> List[int]:
    n = inst.n
    slot_of = [-1] * n
    nbr_slots: List[set] = [set() for _ in range(n)]
    load: List[int] = []
    cap = inst.capacity
    udeg = list(inst.degree)
    todo = set(range(n))
    while todo:
        v = max(todo, key=lambda x: (len(nbr_slots[x]), udeg[x], -x))
        s = 0
        while True:
            if s == len(load):
                load.append(0)
            if s not in nbr_slots[v] and load[s] + inst.size[v] <= cap:
                break
            s += 1
        slot_of[v] = s
        load[s] += inst.size[v]
        todo.discard(v)
        for u in inst.adj_list[v]:
            if slot_of[u] < 0:
                nbr_slots[u].add(s)
                udeg[u] -= 1
    return slot_of


def solve_greedy(inst: Instance, method: str = "dsatur", seed: int = 0) -> Result:
    t0 = time.perf_counter()
    if method == "dsatur":
        slot_of = dsatur(inst)
    else:
        slot_of = first_fit(inst, make_order(inst, method, seed))
    k = num_slots_used(slot_of)
    name = {"dsatur": "DSATUR", "degree": "Welsh-Powell", "input": "FirstFit", "size": "FirstFit-size",
            "random": "FirstFit-random"}.get(method, method)
    return Result(name, "feasible", slot_of, k, nodes=inst.n, time=time.perf_counter() - t0)
