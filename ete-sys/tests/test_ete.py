"""Unit + property tests."""
import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ete.bnb import solve_bnb
from ete.bounds import big_exam_bound, lower_bound, max_clique, seat_bound
from ete.csp import MODES, minimise_with_csp, solve_k
from ete.generators import crown_bad_order, crown_graph, random_instance, uz_instance
from ete.greedy import dsatur, first_fit, make_order, solve_greedy
from ete.hybrid import solve_hybrid
from ete.instance import Exam, Instance
from ete.local_search import tabucol
from ete.reductions import brute_force_sat, sat_to_3col, three_col_to_timetabling
from ete.validate import count_conflicts, validate_assignment
from ete.venues import allocate_venues


def cycle(n):
    return Instance.from_graph(n, [(i, (i + 1) % n) for i in range(n)], name=f"C{n}")


def complete(n):
    return Instance.from_graph(n, [(i, j) for i in range(n) for j in range(i + 1, n)], name=f"K{n}")


def petersen():
    e = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] + \
        [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return Instance.from_graph(10, e, name="Petersen")


def grotzsch():  # Mycielskian of C5: triangle-free, chromatic number 4
    e = [(i, (i + 1) % 5) for i in range(5)]
    for i in range(5):
        e += [(5 + i, (i + 1) % 5), (5 + i, (i - 1) % 5), (5 + i, 10)]
    return Instance.from_graph(11, e, name="Grotzsch")


KNOWN = [(cycle(4), 2), (cycle(5), 3), (cycle(7), 3), (complete(5), 5), (petersen(), 3), (grotzsch(), 4),
         (crown_graph(4), 2)]


class TestModel(unittest.TestCase):
    def test_conflict_graph_from_students_and_lecturers(self):
        exams = [Exam(0, "A", 2, (7,)), Exam(1, "B", 1, (7,)), Exam(2, "C", 1, (8,)), Exam(3, "D", 1, (9,))]
        inst = Instance(exams, {0: {0, 2}}, [100])
        self.assertEqual(inst.adj[0], {1, 2})       # A-B share lecturer 7, A-C share student 0
        self.assertEqual(inst.m, 2)
        self.assertEqual(inst.degree[3], 0)

    def test_exam_larger_than_capacity_rejected(self):
        with self.assertRaises(ValueError):
            Instance([Exam(0, "X", 50)], {}, [10, 10])

    def test_validator_detects_problems(self):
        inst = cycle(4)
        self.assertTrue(validate_assignment(inst, [0, 1, 0, 1])[0])
        ok, pr = validate_assignment(inst, [0, 0, 1, 1])
        self.assertFalse(ok); self.assertEqual(count_conflicts(inst, [0, 0, 1, 1]), 2)
        self.assertFalse(validate_assignment(inst, [0, 1, 0, -1])[0])

    def test_capacity_violation_detected(self):
        inst = Instance([Exam(0, "A", 6), Exam(1, "B", 6)], {}, [10])
        self.assertFalse(validate_assignment(inst, [0, 0])[0])
        self.assertTrue(validate_assignment(inst, [0, 1])[0])

    def test_empty_and_single(self):
        self.assertEqual(solve_bnb(Instance([], {}, [])).num_slots, 0)
        one = Instance([Exam(0, "A", 5)], {}, [10])
        self.assertEqual(solve_bnb(one).num_slots, 1)
        self.assertEqual(solve_k(one, 1, "mac").status, "sat")
        self.assertEqual(solve_greedy(one).num_slots, 1)

    def test_venue_allocation_respects_capacity(self):
        inst = random_instance(30, seed=2)
        r = solve_greedy(inst)
        tt = allocate_venues(inst, r.slot_of)
        for s, alloc in tt.venue_alloc.items():
            used = {}
            for e, v, seats in alloc:
                used[v] = used.get(v, 0) + seats
            for v, u in used.items():
                self.assertLessEqual(u, inst.venues[v])
            per_exam = {}
            for e, v, seats in alloc:
                per_exam[e] = per_exam.get(e, 0) + seats
            for e in tt.slots()[s]:
                self.assertEqual(per_exam[e], inst.size[e])


class TestKnownGraphs(unittest.TestCase):
    def test_bnb_chromatic_numbers(self):
        for inst, chi in KNOWN:
            r = solve_bnb(inst)
            self.assertTrue(r.optimal, inst.name); self.assertEqual(r.num_slots, chi, inst.name)
            self.assertTrue(validate_assignment(inst, r.slot_of)[0])

    def test_csp_modes_agree_with_chromatic_number(self):
        for inst, chi in KNOWN:
            for mode in MODES:
                self.assertEqual(solve_k(inst, chi, mode).status, "sat", (inst.name, mode))
                self.assertEqual(solve_k(inst, chi - 1, mode).status, "unsat", (inst.name, mode))

    def test_naive_with_and_without_symmetry(self):
        inst = petersen()
        for sym in (False, True):
            self.assertEqual(solve_k(inst, 3, "naive", symmetry=sym).status, "sat")
            self.assertEqual(solve_k(inst, 2, "naive", symmetry=sym).status, "unsat")

    def test_minimise_with_csp(self):
        inst = grotzsch()
        r = minimise_with_csp(inst, 2, dsatur_k(inst), "mac")
        self.assertEqual(r.num_slots, 4)

    def test_hybrid_and_tabucol(self):
        for inst, chi in KNOWN:
            h = solve_hybrid(inst, time_limit=3)
            self.assertEqual(h.num_slots, chi, inst.name)
        sol = tabucol(petersen(), 3, max_iters=5000, seed=1)
        self.assertIsNotNone(sol); self.assertEqual(count_conflicts(petersen(), sol), 0)
        self.assertIsNone(tabucol(complete(4), 3, max_iters=300, seed=1))

    def test_timeout_status(self):
        inst = random_instance(60, density=0.5, seed=0)
        r = solve_k(inst, 11, "naive", node_limit=1000)
        self.assertEqual(r.status, "timeout")


def dsatur_k(inst):
    return max(dsatur(inst)) + 1


class TestAgreementOnRandomInstances(unittest.TestCase):
    def test_all_exact_methods_agree(self):
        for gen in (random_instance, uz_instance):
            for n in (8, 11, 14):
                for seed in range(3):
                    inst = gen(n, seed=seed)
                    b = solve_bnb(inst)
                    self.assertTrue(b.optimal)
                    k = b.num_slots
                    for mode in MODES:
                        self.assertEqual(solve_k(inst, k, mode).status, "sat")
                        if k > 1:
                            self.assertEqual(solve_k(inst, k - 1, mode).status, "unsat")
                    self.assertEqual(solve_hybrid(inst, time_limit=3).num_slots, k)
                    self.assertGreaterEqual(k, lower_bound(inst)[0])
                    self.assertLessEqual(k, dsatur_k(inst))

    def test_brute_force_optimum_tiny(self):
        rng = random.Random(5)
        for _ in range(15):
            n = rng.randint(3, 7)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5]
            inst = Instance.from_graph(n, edges)
            best = n
            for assign in itertools.product(range(n), repeat=n):
                if count_conflicts(inst, assign) == 0:
                    best = min(best, len(set(assign)))
            self.assertEqual(solve_bnb(inst).num_slots, best)

    def test_capacity_changes_optimum(self):
        # 4 independent exams of 6 seats, capacity 10 -> one per slot (4 slots), even with no conflicts
        inst = Instance([Exam(i, f"E{i}", 6) for i in range(4)], {}, [10])
        self.assertEqual(solve_bnb(inst).num_slots, 4)
        self.assertEqual(seat_bound(inst), 3)       # fractional bound
        self.assertEqual(big_exam_bound(inst), 4)   # big-exam bound is tight here
        self.assertEqual(solve_k(inst, 3, "mac").status, "unsat")


class TestApproximation(unittest.TestCase):
    def test_greedy_bounds(self):
        for seed in range(5):
            inst = random_instance(60, density=0.2, seed=seed)
            for kind in ("input", "degree", "size", "random"):
                sol = first_fit(inst, make_order(inst, kind, seed))
                self.assertTrue(validate_assignment(inst, sol)[0])
            # uncapacitated first-fit uses at most Delta+1 slots
            free = Instance(inst.exams, inst.enrolments, [], [])
            self.assertLessEqual(max(first_fit(free, make_order(free, "input"))) + 1, max(free.degree) + 1)

    def test_crown_graph_worst_case(self):
        for h in (3, 6, 10):
            inst = crown_graph(h)
            self.assertEqual(max(first_fit(inst, crown_bad_order(h))) + 1, h)   # ratio h/2
            self.assertEqual(max(dsatur(inst)) + 1, 2)                          # DSATUR exact on bipartite

    def test_clique_bound(self):
        cl, exact = max_clique(complete(6))
        self.assertEqual((len(cl), exact), (6, True))
        cl, _ = max_clique(petersen())
        self.assertEqual(len(cl), 2)


class TestReductions(unittest.TestCase):
    def check(self, nv, clauses):
        n, edges, _ = sat_to_3col(nv, clauses)
        g = Instance.from_graph(n, edges)
        colourable = solve_k(g, 3, "mac", time_limit=20).status == "sat"
        self.assertEqual(colourable, brute_force_sat(nv, clauses), (nv, clauses))
        # and the timetabling instance obtained from the graph agrees (k = 3 slots)
        tt = three_col_to_timetabling(n, edges)
        self.assertEqual(solve_k(tt, 3, "mac", time_limit=20).status == "sat", colourable)

    def test_satisfiable(self):
        self.check(3, [[1, 2, 3], [-1, -2, 3], [1, -2, -3]])

    def test_unsatisfiable(self):
        cl = [[a * 1, b * 2, c * 3] for a in (1, -1) for b in (1, -1) for c in (1, -1)]   # all 8 sign patterns
        self.check(3, cl)

    def test_random_formulas(self):
        rng = random.Random(11)
        for _ in range(12):
            nv = 3
            cls = [[rng.choice([-1, 1]) * rng.randint(1, nv) for _ in range(3)] for _ in range(rng.randint(2, 9))]
            self.check(nv, cls)


if __name__ == "__main__":
    unittest.main()
