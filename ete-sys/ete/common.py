"""Shared result type and helpers."""
from __future__ import annotations
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.setrecursionlimit(100000)


class Timeout(Exception):
    """Raised internally when a node or time budget is exhausted."""


@dataclass
class Result:
    algorithm: str
    status: str                      # 'sat' | 'unsat' | 'optimal' | 'feasible' | 'timeout'
    slot_of: Optional[List[int]]     # best assignment found (None if none)
    num_slots: int                   # slots used by slot_of (0 if none)
    nodes: int = 0
    time: float = 0.0
    lower_bound: int = 0
    optimal: bool = False
    trace: List[Tuple[float, int]] = field(default_factory=list)   # (elapsed s, best slots)
    extra: dict = field(default_factory=dict)


class Clock:
    def __init__(self, limit: Optional[float]):
        self.t0 = time.perf_counter()
        self.limit = limit

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def expired(self) -> bool:
        return self.limit is not None and self.elapsed() > self.limit

    def remaining(self) -> Optional[float]:
        return None if self.limit is None else max(0.0, self.limit - self.elapsed())


def num_slots_used(slot_of) -> int:
    return len(set(s for s in slot_of if s >= 0))


def compact(slot_of: List[int]) -> List[int]:
    """Renumber slots 0..k-1 in order of first use."""
    mp = {}
    out = []
    for s in slot_of:
        if s not in mp:
            mp[s] = len(mp)
        out.append(mp[s])
    return out
