"""Venue allocation inside a slot: best-fit-decreasing with splitting.

Because constraint (C2) guarantees total seats <= total capacity, splitting an exam
over several venues always succeeds, so allocation is a post-processing step.
"""
from __future__ import annotations
from typing import List, Tuple
from .instance import Instance, Timetable


def allocate_venues(inst: Instance, slot_of: List[int]) -> Timetable:
    k = (max(slot_of) + 1) if slot_of else 0
    tt = Timetable(list(slot_of), k)
    if not inst.venues:
        return tt
    for s, exams in tt.slots().items():
        free = list(inst.venues)
        alloc: List[Tuple[int, int, int]] = []
        for e in sorted(exams, key=lambda x: -inst.size[x]):
            need = inst.size[e]
            fits = [v for v in range(len(free)) if free[v] >= need]
            if fits:
                v = min(fits, key=lambda x: free[x])
                free[v] -= need; alloc.append((e, v, need)); continue
            for v in sorted(range(len(free)), key=lambda x: -free[x]):
                if need == 0:
                    break
                take = min(free[v], need)
                if take > 0:
                    free[v] -= take; need -= take; alloc.append((e, v, take))
            if need > 0:
                raise RuntimeError("capacity invariant broken")
        tt.venue_alloc[s] = alloc
    return tt
