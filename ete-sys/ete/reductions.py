"""Polynomial-time reductions used in the NP-completeness proof.

    3-SAT  <=p  3-COLOURING  <=p  EXAM TIMETABLING (k = 3 slots)

Literals are non-zero ints: +i = x_i, -i = NOT x_i.
"""
from __future__ import annotations
from itertools import product
from typing import Dict, List, Sequence, Tuple
from .instance import Exam, Instance


def sat_to_3col(n_vars: int, clauses: Sequence[Sequence[int]]) -> Tuple[int, List[Tuple[int, int]], Dict[str, int]]:
    """Build graph G(phi) that is 3-colourable iff phi is satisfiable.

    Palette triangle T,F,B; for each variable a triangle (x_i, not x_i, B); each clause
    (a v b v c) is two chained OR-gadgets whose output is joined to F and B (forcing it to T).
    """
    edges: List[Tuple[int, int]] = []
    nodes = {"T": 0, "F": 1, "B": 2}
    nxt = [3]

    def new() -> int:
        v = nxt[0]; nxt[0] += 1; return v

    T, F, B = 0, 1, 2
    edges += [(T, F), (F, B), (B, T)]
    lit_node: Dict[int, int] = {}
    for i in range(1, n_vars + 1):
        p, q = new(), new()
        lit_node[i], lit_node[-i] = p, q
        edges += [(p, q), (p, B), (q, B)]

    def or_gadget(a: int, b: int) -> int:
        p, q, o = new(), new(), new()
        edges.extend([(p, a), (q, b), (p, q), (p, o), (q, o)])
        return o

    for cl in clauses:
        lits = [lit_node[l] for l in cl]
        while len(lits) > 1:                       # fold clause of any width (3 for 3-SAT)
            a, b = lits.pop(0), lits.pop(0)
            lits.insert(0, or_gadget(a, b))
        out = lits[0]
        edges += [(out, F), (out, B)]
    return nxt[0], edges, nodes


def three_col_to_timetabling(n: int, edges: Sequence[Tuple[int, int]]) -> Instance:
    """Vertex -> exam; edge {u,v} -> one student enrolled in exactly exams u and v. Ask for 3 slots."""
    exams = [Exam(i, f"E{i}", 1) for i in range(n)]
    enrol = {j: {u, v} for j, (u, v) in enumerate(edges)}
    return Instance(exams, enrol, [], [], name="reduced")


def brute_force_sat(n_vars: int, clauses: Sequence[Sequence[int]]) -> bool:
    for bits in product([False, True], repeat=n_vars):
        if all(any((bits[abs(l) - 1] if l > 0 else not bits[abs(l) - 1]) for l in cl) for cl in clauses):
            return True
    return False
