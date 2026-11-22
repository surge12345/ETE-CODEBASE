"""Instance generators: random (cohort model) and UZ-inspired (faculty/programme/level model).

NOTE: the UZ-inspired generator is *synthetic*. It mimics the structure of a Zimbabwean
university (faculties -> programmes -> levels 1-4 -> cohorts, shared service courses,
carry-over/re-sit cohorts, lecturers teaching several courses). Venue names are
illustrative and capacities are scaled to the instance; no official UZ data is used.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Set, Tuple
from .instance import Exam, Instance

FACULTIES = ["Science", "Engineering", "Law", "Arts", "Medicine", "Commerce", "Agriculture", "Social Studies"]
VENUE_NAMES = ["Great Hall", "Science Theatre", "Engineering Hall", "Law Hall", "Medical Auditorium",
               "Commerce Hall", "Agric Hall", "Annex"]
VENUE_SHARE = [0.26, 0.16, 0.14, 0.10, 0.10, 0.09, 0.08, 0.07]


def _make_venues(total_cap: int, n_venues: int) -> List[int]:
    shares = VENUE_SHARE[:n_venues]
    s = sum(shares)
    caps = [max(1, int(total_cap * x / s)) for x in shares]
    caps[0] += total_cap - sum(caps)
    return caps


def _finalise(name: str, n: int, cohorts: Dict[int, Set[int]], cohort_size: Dict[int, int],
              lecturers: Dict[int, Tuple[int, ...]], codes: List[str], slot_factor: float,
              n_venues: int, rng: random.Random) -> Instance:
    sizes = [0] * n
    for cid, ex in cohorts.items():
        for e in ex:
            sizes[e] += cohort_size[cid]
    for e in range(n):                        # every exam has at least one candidate
        if sizes[e] == 0:
            sizes[e] = rng.randint(10, 40)
    total = sum(sizes)
    # Capacity is tuned relative to the conflict structure so that BOTH constraints matter:
    # the seat bound total/capacity is about `slot_factor` x (uncapacitated DSATUR slots).
    from .greedy import dsatur
    free = Instance([Exam(i, codes[i], sizes[i], lecturers.get(i, ())) for i in range(n)], cohorts, [], [])
    k0 = max(dsatur(free)) + 1 if n else 1
    total_cap = max(int(1.15 * max(sizes)), int(total / max(1.0, slot_factor * k0)) + 1)
    venues = _make_venues(total_cap, n_venues)
    exams = [Exam(i, codes[i], sizes[i], lecturers.get(i, ())) for i in range(n)]
    return Instance(exams, cohorts, venues, [], name=name)


def random_instance(n: int, density: float = 0.12, seed: int = 0, n_lecturers: int = 0,
                    slot_factor: float = 0.8, n_venues: int = 6) -> Instance:
    """Random cohort model.

    Cohorts (groups of 10-60 students) each take 2-6 random exams; the number of cohorts is
    chosen so that the expected conflict-graph density is about `density`.
    """
    rng = random.Random(seed)
    avg_c = 4.0
    pairs_per_cohort = avg_c * (avg_c - 1) / 2.0
    n_pairs = n * (n - 1) / 2.0
    n_coh = max(1, int(round(-math.log(max(1e-9, 1 - density)) * n_pairs / pairs_per_cohort))) if n > 1 else 1
    weights = [1.0 / (1 + 0.5 * math.log(1 + i)) for i in range(n)]   # mild popularity skew
    cohorts: Dict[int, Set[int]] = {}
    cohort_size: Dict[int, int] = {}
    for cid in range(n_coh):
        c = min(n, rng.randint(2, 6))
        cohorts[cid] = set()
        while len(cohorts[cid]) < c:
            cohorts[cid].add(rng.choices(range(n), weights)[0])
        cohort_size[cid] = rng.randint(10, 60)
    lecturers: Dict[int, Tuple[int, ...]] = {}
    L = n_lecturers or max(2, n // 2)
    for e in range(n):
        lecturers[e] = (rng.randrange(L),)
    return _finalise(f"random_n{n}_s{seed}", n, cohorts, cohort_size, lecturers,
                     [f"EX{e:03d}" for e in range(n)], slot_factor, n_venues, rng)


def uz_instance(n: int, seed: int = 0, slot_factor: float = 0.8, n_venues: int = 8) -> Instance:
    """UZ-inspired synthetic instance with `n` exams."""
    rng = random.Random(seed)
    n_units = max(1, math.ceil(n / 6))            # programme-level units (~6 core exams each)
    unit_fac = [rng.randrange(len(FACULTIES)) for _ in range(n_units)]
    unit_level = [1 + (u % 4) for u in range(n_units)]
    exams_of_unit: List[List[int]] = [[] for _ in range(n_units)]
    codes: List[str] = []
    for e in range(n):
        u = e % n_units
        exams_of_unit[u].append(e)
        codes.append(f"{FACULTIES[unit_fac[u]][:3].upper()}{unit_level[u]}{e:03d}")
    service = set(rng.sample(range(n), max(1, n // 20)))        # shared service courses
    fac_exams: Dict[int, List[int]] = {}
    for e in range(n):
        fac_exams.setdefault(unit_fac[e % n_units], []).append(e)
    cohorts: Dict[int, Set[int]] = {}
    cohort_size: Dict[int, int] = {}
    cid = 0
    for u in range(n_units):
        core = exams_of_unit[u]
        if not core:
            continue
        main = set(core)
        for _ in range(rng.randint(0, 2)):                       # electives within faculty
            main.add(rng.choice(fac_exams[unit_fac[u]]))
        if service and rng.random() < 0.7:
            main.add(rng.choice(sorted(service)))
        cohorts[cid] = main
        cohort_size[cid] = rng.randint(40, 260)
        cid += 1
        for _ in range(rng.randint(1, 2)):                       # carry-over / re-sit cohorts
            pool = list(core) + (exams_of_unit[(u - 1) % n_units] if rng.random() < 0.6 else [])
            k = min(len(pool), rng.randint(2, 5))
            cohorts[cid] = set(rng.sample(pool, k))
            cohort_size[cid] = rng.randint(4, 25)
            cid += 1
    lecturers: Dict[int, Tuple[int, ...]] = {}
    pool_by_fac: Dict[int, List[int]] = {}
    next_lect = 0
    for f, es in fac_exams.items():
        n_l = max(1, len(es) // 3)
        pool_by_fac[f] = list(range(next_lect, next_lect + n_l)); next_lect += n_l
    for e in range(n):
        f = unit_fac[e % n_units]
        lecturers[e] = (rng.choice(pool_by_fac[f]),)
    return _finalise(f"uz_n{n}_s{seed}", n, cohorts, cohort_size, lecturers, codes, slot_factor, n_venues, rng)


def crown_graph(h: int) -> Instance:
    """Crown graph on 2h vertices (K_{h,h} minus a perfect matching): chromatic number 2,
    but First-Fit with the order a1,b1,a2,b2,... uses h colours (greedy worst case)."""
    edges = [(i, h + j) for i in range(h) for j in range(h) if i != j]
    return Instance.from_graph(2 * h, edges, name=f"crown{h}")


def crown_bad_order(h: int) -> List[int]:
    order: List[int] = []
    for i in range(h):
        order += [i, h + i]
    return order
