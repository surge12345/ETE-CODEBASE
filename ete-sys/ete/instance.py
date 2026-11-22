"""Problem model: exams, students, lecturers, venues -> conflict graph.

Formal model:
    Variables  X_e  : the time slot assigned to exam e
    Domains    D_e  : {0, ..., k-1}
    Constraints
      (C1) conflict   : X_e != X_f whenever e and f share a student or a lecturer
      (C2) capacity   : sum of sizes of the exams in one slot <= total venue capacity
Objective: minimise the number of slots k used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass
class Exam:
    eid: int
    code: str
    size: int                      # number of enrolled students (= seats needed)
    lecturers: Tuple[int, ...] = ()


@dataclass
class Instance:
    """An exam timetabling instance.

    Parameters
    ----------
    exams : list of Exam (eid must equal list index)
    enrolments : mapping student id -> set of exam ids taken by that student
    venues : list of venue capacities (all venues are available in every slot)
    extra_edges : additional conflict edges (used by reductions from pure colouring)
    """

    exams: List[Exam]
    enrolments: Dict[int, Set[int]] = field(default_factory=dict)
    venues: List[int] = field(default_factory=list)
    extra_edges: List[Tuple[int, int]] = field(default_factory=list)
    name: str = "instance"

    def __post_init__(self) -> None:
        for i, e in enumerate(self.exams):
            if e.eid != i:
                raise ValueError("exam ids must equal their list index")
        self.n: int = len(self.exams)
        self.capacity: int = sum(self.venues) if self.venues else 10 ** 12
        self.size: List[int] = [e.size for e in self.exams]
        for e in self.exams:
            if e.size > self.capacity:
                raise ValueError(
                    f"exam {e.code} needs {e.size} seats but total capacity is {self.capacity}")
        self._build_graph()

    def _build_graph(self) -> None:
        adj: List[Set[int]] = [set() for _ in range(self.n)]
        for exams in self.enrolments.values():
            ex = sorted(exams)
            for a in range(len(ex)):
                for b in range(a + 1, len(ex)):
                    adj[ex[a]].add(ex[b]); adj[ex[b]].add(ex[a])
        by_lect: Dict[int, List[int]] = {}
        for e in self.exams:
            for l in e.lecturers:
                by_lect.setdefault(l, []).append(e.eid)
        for group in by_lect.values():
            for a in range(len(group)):
                for b in range(a + 1, len(group)):
                    adj[group[a]].add(group[b]); adj[group[b]].add(group[a])
        for u, v in self.extra_edges:
            if u != v:
                adj[u].add(v); adj[v].add(u)
        self.adj: List[Set[int]] = adj
        self.adj_list: List[List[int]] = [sorted(s) for s in adj]
        self.adj_mask: List[int] = [sum(1 << v for v in s) for s in adj]
        self.degree: List[int] = [len(s) for s in adj]
        self.m: int = sum(self.degree) // 2

    @property
    def density(self) -> float:
        return 0.0 if self.n < 2 else 2.0 * self.m / (self.n * (self.n - 1))

    @classmethod
    def from_graph(cls, n: int, edges: Sequence[Tuple[int, int]], sizes: Optional[Sequence[int]] = None,
                   capacity: Optional[int] = None, name: str = "graph") -> "Instance":
        """Pure graph-colouring instance (each edge = a conflict)."""
        sizes = list(sizes) if sizes is not None else [1] * n
        exams = [Exam(i, f"V{i}", sizes[i]) for i in range(n)]
        venues = [capacity] if capacity is not None else []
        return cls(exams, {}, venues, list(edges), name=name)

    def summary(self) -> str:
        cap = self.capacity if self.venues else "inf"
        return (f"{self.name}: n={self.n} exams, m={self.m} conflicts, density={self.density:.3f}, "
                f"students={len(self.enrolments)}, capacity={cap}")


@dataclass
class Timetable:
    """A solution: slot per exam (+ optional venue allocation)."""
    slot_of: List[int]
    num_slots: int
    venue_alloc: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)  # slot -> [(exam, venue, seats)]

    def slots(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for e, s in enumerate(self.slot_of):
            out.setdefault(s, []).append(e)
        return out
