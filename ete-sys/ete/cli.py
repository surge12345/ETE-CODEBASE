
from __future__ import annotations
import argparse
import csv
import json
import sys

from .bnb import solve_bnb
from .bounds import lower_bound
from .generators import random_instance, uz_instance
from .greedy import solve_greedy
from .hybrid import solve_hybrid
from .instance import Exam, Instance
from .validate import validate_assignment
from .venues import allocate_venues


def to_dict(inst: Instance) -> dict:
    return {"name": inst.name, "venues": inst.venues,
            "exams": [{"code": e.code, "size": e.size, "lecturers": list(e.lecturers)} for e in inst.exams],
            "cohorts": {str(k): sorted(v) for k, v in inst.enrolments.items()}}


def from_dict(d: dict) -> Instance:
    exams = [Exam(i, e["code"], e["size"], tuple(e["lecturers"])) for i, e in enumerate(d["exams"])]
    return Instance(exams, {int(k): set(v) for k, v in d["cohorts"].items()}, d["venues"], [], name=d["name"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ete")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve", help="generate/load an instance and solve it")
    s.add_argument("--instance"); s.add_argument("--kind", choices=["uz", "random"], default="uz")
    s.add_argument("--n", type=int, default=50); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--density", type=float, default=0.12)
    s.add_argument("--algo", choices=["firstfit", "welsh", "dsatur", "bnb", "hybrid"], default="hybrid")
    s.add_argument("--time", type=float, default=10.0); s.add_argument("--out")
    a = ap.parse_args(argv)

    if a.instance:
        inst = from_dict(json.load(open(a.instance)))
    elif a.kind == "uz":
        inst = uz_instance(a.n, a.seed)
    else:
        inst = random_instance(a.n, a.density, a.seed)
    print(inst.summary())
    lb, parts = lower_bound(inst)
    if a.algo in ("firstfit", "welsh", "dsatur"):
        r = solve_greedy(inst, {"firstfit": "input", "welsh": "degree", "dsatur": "dsatur"}[a.algo])
    elif a.algo == "bnb":
        r = solve_bnb(inst, time_limit=a.time)
    else:
        r = solve_hybrid(inst, time_limit=a.time, seed=a.seed)
    ok, problems = validate_assignment(inst, r.slot_of)
    print(f"{r.algorithm}: {r.num_slots} slots (lower bound {lb}), optimal={r.optimal}, "
          f"nodes={r.nodes}, time={r.time:.2f}s, valid={ok}")
    if not ok:
        print("PROBLEMS:", problems[:5]); return 1
    if a.out:
        json.dump(to_dict(inst), open(a.out + ".json", "w"), indent=1)
        tt = allocate_venues(inst, r.slot_of)
        with open(a.out + "_timetable.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["exam", "students", "slot", "venues(seats)"])
            alloc = {}
            for s_, lst in tt.venue_alloc.items():
                for e, v, seats in lst:
                    alloc.setdefault(e, []).append(f"venue{v}({seats})")
            for e in sorted(range(inst.n), key=lambda x: (r.slot_of[x], x)):
                w.writerow([inst.exams[e].code, inst.size[e], r.slot_of[e] + 1, ";".join(alloc.get(e, []))])
        print(f"wrote {a.out}.json and {a.out}_timetable.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
