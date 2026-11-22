"""Independent solution checker (used by every test and benchmark)."""
from __future__ import annotations
from typing import List, Sequence, Tuple
from .instance import Instance


def count_conflicts(inst: Instance, slot_of: Sequence[int]) -> int:
    """Number of conflicting exam pairs placed in the same slot (-1 = unassigned, ignored)."""
    c = 0
    for u in range(inst.n):
        su = slot_of[u]
        if su < 0:
            continue
        for v in inst.adj_list[u]:
            if v > u and slot_of[v] == su:
                c += 1
    return c


def capacity_violations(inst: Instance, slot_of: Sequence[int]) -> int:
    """Total seats in excess of capacity, summed over slots."""
    load = {}
    for e, s in enumerate(slot_of):
        if s >= 0:
            load[s] = load.get(s, 0) + inst.size[e]
    return sum(max(0, l - inst.capacity) for l in load.values())


def validate_assignment(inst: Instance, slot_of: Sequence[int]) -> Tuple[bool, List[str]]:
    """Return (ok, problems). A student/lecturer clash or an over-full slot is a problem."""
    problems: List[str] = []
    if len(slot_of) != inst.n:
        return False, ["wrong length"]
    if any(s < 0 for s in slot_of):
        problems.append("unassigned exam(s)")
    for u in range(inst.n):
        for v in inst.adj_list[u]:
            if v > u and slot_of[u] == slot_of[v] and slot_of[u] >= 0:
                problems.append(f"conflict: {inst.exams[u].code} & {inst.exams[v].code} in slot {slot_of[u]}")
    cv = capacity_violations(inst, slot_of)
    if cv:
        problems.append(f"capacity exceeded by {cv} seats in total")
    return (not problems), problems
