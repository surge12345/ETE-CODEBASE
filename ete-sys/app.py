#!/usr/bin/env python3
"""ETE - Exam Timetabling Engine: interactive demo application.

Start it with:      python app.py

Menu-driven front end for the engine (ete/) and the experiment scripts (experiments/).
Every prompt shows a default in [brackets]; press Enter to accept it. Ctrl+C goes back one level.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from ete.bnb import solve_bnb                                   # noqa: E402
from ete.bounds import lower_bound                              # noqa: E402
from ete.cli import from_dict, to_dict                          # noqa: E402
from ete.common import num_slots_used                           # noqa: E402
from ete.generators import crown_bad_order, crown_graph, random_instance, uz_instance   # noqa: E402
from ete.greedy import first_fit, solve_greedy                  # noqa: E402
from ete.hybrid import solve_hybrid                             # noqa: E402
from ete.instance import Exam, Instance                         # noqa: E402
from ete.validate import validate_assignment                    # noqa: E402
from ete.venues import allocate_venues                          # noqa: E402

OUTPUT_DIR = "output"            # solved timetables are written here
DEMO_RESULTS = "results_demo"    # safe default place for experiment runs (does not touch results/)
OFFICIAL_RESULTS = "results"     # the results shipped with the project / used by the report

def _color_ok() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if platform.system() == "Windows":
        os.system("")            # switches on ANSI colours in recent Windows terminals
    return True


_COLOR = _color_ok()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(t): return _c("1", t)
def dim(t): return _c("2", t)
def green(t): return _c("32", t)
def red(t): return _c("31", t)
def yellow(t): return _c("33", t)
def cyan(t): return _c("36;1", t)


WIDTH = 78


def banner(title: str, sub: str = "") -> None:
    print()
    print(cyan("=" * WIDTH))
    print(cyan(f"  {title}"))
    if sub:
        print(dim(f"  {sub}"))
    print(cyan("=" * WIDTH))


def section(title: str) -> None:
    print()
    print(bold(f"--- {title} " + "-" * max(3, WIDTH - len(title) - 6)))


def ok(msg): print(green("  [ok] ") + msg)
def warn(msg): print(yellow("  [!]  ") + msg)
def err(msg): print(red("  [x]  ") + msg)
def info(msg): print(dim("  " + msg))


class Back(Exception):
    """Raised when the user presses Ctrl+C / Ctrl+D at a prompt: return to the previous menu."""


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        raise Back()


def ask_choice(title: str, options: list, default: int | None = None, back_label: str | None = "Back") -> int:
    """Show a numbered menu. Returns the chosen number (1..len) or 0 for the back/quit entry."""
    print()
    if title:
        print(bold(title))
    for i, opt in enumerate(options, 1):
        print(f"  {cyan(str(i))}) {opt}")
    if back_label:
        print(f"  {cyan('0')}) {back_label}")
    while True:
        raw = _ask(f"Choose{f' [{default}]' if default else ''}: ")
        if raw == "" and default is not None:
            return default
        if raw.isdigit():
            v = int(raw)
            if (v == 0 and back_label) or 1 <= v <= len(options):
                return v
        err(f"Please type a number from {0 if back_label else 1} to {len(options)}.")


def ask_int(prompt: str, default: int, lo: int = 1, hi: int | None = None) -> int:
    while True:
        raw = _ask(f"{prompt} [{default}]: ")
        if raw == "":
            return default
        try:
            v = int(raw)
        except ValueError:
            err("Please type a whole number.")
            continue
        if v < lo or (hi is not None and v > hi):
            err(f"Please choose a value between {lo} and {hi if hi is not None else 'a large number'}.")
            continue
        return v


def ask_float(prompt: str, default: float, lo: float, hi: float) -> float:
    while True:
        raw = _ask(f"{prompt} [{default}]: ")
        if raw == "":
            return default
        try:
            v = float(raw)
        except ValueError:
            err("Please type a number, for example 0.15.")
            continue
        if not (lo <= v <= hi):
            err(f"Please choose a value between {lo} and {hi}.")
            continue
        return v


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _ask(f"{prompt} [{hint}]: ").lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        err("Please answer y or n.")


def ask_text(prompt: str, default: str = "") -> str:
    raw = _ask(f"{prompt}{f' [{default}]' if default else ''}: ")
    return raw or default


def clear_screen() -> None:
    if not sys.stdout.isatty():
        return
    if platform.system() == "Windows":
        os.system("cls")
    else:
        print("\033[2J\033[H", end="", flush=True)


def pause() -> None:
    try:
        input(dim("\n  Press Enter to continue..."))
    except (KeyboardInterrupt, EOFError):
        print()


def rel(path: str) -> str:
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def have_modules(*names: str) -> list:
    return [n for n in names if importlib.util.find_spec(n) is None]


def open_folder(path: str) -> None:
    path = os.path.abspath(path)
    try:
        if platform.system() == "Windows":
            os.startfile(path)                                    # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        warn(f"Could not open a file window automatically. Open this folder yourself: {path}")


def offer_open(path: str) -> None:
    if ask_yes_no("Open this folder in your file explorer?", False):
        open_folder(path)


def run_script(args: list) -> int:
    """Run a project script in a child process, letting its progress output show live."""
    print(dim("  $ " + " ".join([os.path.basename(sys.executable)] + args)))
    t0 = time.time()
    try:
        rc = subprocess.run([sys.executable] + args, cwd=ROOT).returncode
    except KeyboardInterrupt:
        print()
        warn("Stopped by you (Ctrl+C).")
        return 130
    print(dim(f"  (finished in {time.time() - t0:.1f}s)"))
    return rc


ALGOS = [
    ("firstfit", "First-Fit"),
    ("welsh", "Welsh-Powell"),
    ("dsatur", "DSATUR"),
    ("bnb", "Branch and Bound"),
    ("hybrid", "Hybrid B&B + local search"),
]


def solve_with(inst: Instance, algo: str, time_limit: float, seed: int = 0):
    if algo in ("firstfit", "welsh", "dsatur"):
        return solve_greedy(inst, {"firstfit": "input", "welsh": "degree", "dsatur": "dsatur"}[algo])
    if algo == "bnb":
        return solve_bnb(inst, time_limit=time_limit)
    return solve_hybrid(inst, time_limit=time_limit, seed=seed)


def describe_problem(inst: Instance) -> None:
    section("The problem")
    print(f"  Name            : {bold(inst.name)}")
    print(f"  Exams           : {inst.n}")
    print(f"  Student groups  : {len(inst.enrolments)}")
    print(f"  Clashing pairs  : {inst.m}   (density {inst.density:.3f})")
    if inst.venues:
        print(f"  Venues          : {len(inst.venues)}   (seats per slot: {inst.capacity})")
    else:
        print("  Venues          : none given (unlimited seats)")
    lb, parts = lower_bound(inst)
    print(f"  Lower bound     : {bold(str(lb))} slots  "
          + dim(f"(clique {parts['clique']}, seats {parts['seat']}, big exams {parts['big']})"))
    info("The lower bound is the least number of slots that is definitely needed.")




def print_result(r, inst: Instance, lb: int) -> bool:
    ok_, problems = validate_assignment(inst, r.slot_of)
    section("Result")
    print(f"  Algorithm       : {bold(r.algorithm)}")
    print(f"  Slots used      : {bold(str(r.num_slots))}    (lower bound {lb}, gap {r.num_slots - lb})")
    if r.optimal or r.num_slots == lb:
        print("  Best possible?  : " + green("YES - proven optimal (nothing can use fewer slots)"))
    else:
        print("  Best possible?  : " + yellow("not proven (the true best is between "
                                              f"{lb} and {r.num_slots})"))
    print(f"  Search steps    : {r.nodes}      Time: {r.time:.2f}s")
    if ok_:
        print("  Checker         : " + green("VALID - no clashes, no overfull slots"))
    else:
        print("  Checker         : " + red("PROBLEMS FOUND"))
        for p in problems[:5]:
            err(p)
    return ok_


def timetable_rows(inst: Instance, slot_of: list) -> dict:
    by_slot: dict = {}
    for e in range(inst.n):
        by_slot.setdefault(slot_of[e], []).append(e)
    return dict(sorted(by_slot.items()))


def show_timetable(inst: Instance, slot_of: list, limit_exams_per_slot: int = 12) -> None:
    import textwrap
    section("Timetable (one block per slot)")
    cap = inst.capacity if inst.venues else None
    by_slot = timetable_rows(inst, slot_of)
    for s, exams in by_slot.items():
        load = sum(inst.size[e] for e in exams)
        loadtxt = f"{load}/{cap} seats" if cap else f"{load} students"
        print(bold(f"  Slot {s + 1:>2}") + f"  ({len(exams)} exam{'' if len(exams) == 1 else 's'}, {loadtxt})")
        names = [f"{inst.exams[e].code}({inst.size[e]})" for e in sorted(exams)]
        extra = ""
        if len(names) > limit_exams_per_slot:
            extra = f" ... +{len(names) - limit_exams_per_slot} more"
            names = names[:limit_exams_per_slot]
        for line in textwrap.wrap(", ".join(names) + extra, width=WIDTH - 10):
            print("          " + line)
    print(dim("  Format: CODE(number of students). Seats = students in the slot / seats in all venues."))


def show_rooms(inst: Instance, slot_of: list) -> None:
    section("Room allocation")
    if not inst.venues:
        info("No venues were given for this problem, so there is nothing to allocate.")
        return
    tt = allocate_venues(inst, slot_of)
    for s in sorted(tt.venue_alloc):
        print(bold(f"  Slot {s + 1}"))
        for e, v, seats in tt.venue_alloc[s]:
            print(f"     {inst.exams[e].code:<10} venue {v} ({inst.venues[v]} seats): {seats} students")


def instance_to_dict(inst: Instance) -> dict:
    """Like ete.cli.to_dict, but also keeps 'extra' clash edges (used by the crown puzzle) by saving
    each one as a two-exam group, so the saved JSON file reloads into the same problem."""
    d = to_dict(inst)
    if inst.extra_edges:
        nxt = max([int(k) for k in d["cohorts"]] + [-1]) + 1
        for i, (a, b) in enumerate(inst.extra_edges):
            d["cohorts"][str(nxt + i)] = sorted((a, b))
    return d


def save_outputs(inst: Instance, r, prefix: str) -> list:
    """Same file formats as `python -m ete.cli solve --out ...`."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    json_path, csv_path = prefix + ".json", prefix + "_timetable.csv"
    with open(json_path, "w") as f:
        json.dump(instance_to_dict(inst), f, indent=1)
    alloc: dict = {}
    if inst.venues:
        tt = allocate_venues(inst, r.slot_of)
        for s_, lst in tt.venue_alloc.items():
            for e, v, seats in lst:
                alloc.setdefault(e, []).append(f"venue{v}({seats})")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exam", "students", "slot", "venues(seats)"])
        for e in sorted(range(inst.n), key=lambda x: (r.slot_of[x], x)):
            w.writerow([inst.exams[e].code, inst.size[e], r.slot_of[e] + 1, ";".join(alloc.get(e, []))])
    return [json_path, csv_path]


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s) or "problem"


def compare_all(inst: Instance, time_limit: float, seed: int = 0) -> None:
    section("Comparing all algorithms on this problem")
    lb, _ = lower_bound(inst)
    rows = []
    for key, name in ALGOS:
        r = solve_with(inst, key, time_limit, seed)
        ok_, _ = validate_assignment(inst, r.slot_of)
        rows.append((name, r.num_slots, "yes" if (r.optimal or r.num_slots == lb) else "no", r.nodes, r.time, ok_))
    if inst.name.startswith("crown"):
        h = inst.n // 2
        so = first_fit(inst, crown_bad_order(h))
        ok_, _ = validate_assignment(inst, so)
        rows.insert(1, ("First-Fit (unlucky order)", num_slots_used(so), "no", inst.n, 0.0, ok_))
    print(f"  {'Algorithm':<28}{'Slots':>6}{'Proven':>8}{'Steps':>9}{'Time(s)':>9}  Valid")
    print("  " + "-" * 70)
    best = min(r[1] for r in rows)
    for name, k, proven, nodes, t, ok_ in rows:
        line = f"  {name:<28}{k:>6}{proven:>8}{nodes:>9}{t:>9.2f}  {'yes' if ok_ else 'NO'}"
        print(green(line) if k == best else line)
    print(dim(f"  Lower bound: {lb}. Green = fewest slots found."))
    if inst.name.startswith("crown"):
        info("Crown puzzle: the best answer is 2 slots. First-Fit with an unlucky order uses h slots.")


def solve_and_report(inst: Instance, algo: str, time_limit: float, seed: int, save_prefix: str) -> None:
    """Solve, print everything, auto-save, then offer follow-up actions."""
    describe_problem(inst)
    lb, _ = lower_bound(inst)
    print(f"\n  Solving with {bold(dict(ALGOS)[algo])} ...", flush=True)
    r = solve_with(inst, algo, time_limit, seed)
    valid = print_result(r, inst, lb)
    if not valid:
        err("The checker rejected this timetable, so nothing was saved.")
        return
    if inst.n <= 80:
        show_timetable(inst, r.slot_of)
    paths = save_outputs(inst, r, save_prefix)
    section("Saved files")
    for p in paths:
        ok(rel(p))
    info("The *_timetable.csv file lists every exam with its slot and rooms (open it in Excel).")

    while True:
        c = ask_choice("What next?", [
            "Show the timetable, slot by slot",
            "Show the room allocation",
            "Compare ALL algorithms on this problem",
            "Solve again with a different algorithm / time limit",
            "Open the output folder",
        ], back_label="Back to the previous menu")
        try:
            if c == 0:
                return
            if c != 5:
                clear_screen()
            if c == 1:
                show_timetable(inst, r.slot_of, limit_exams_per_slot=10 ** 6 if inst.n <= 80 else 12)
            elif c == 2:
                show_rooms(inst, r.slot_of)
            elif c == 3:
                compare_all(inst, time_limit, seed)
            elif c == 4:
                algo = pick_algorithm(default=5)
                time_limit = ask_time_limit(algo)
                r = solve_with(inst, algo, time_limit, seed)
                lb, _ = lower_bound(inst)
                if print_result(r, inst, lb):
                    save_outputs(inst, r, save_prefix + "_" + algo)
                    ok(f"Saved as {rel(save_prefix)}_{algo}.json / _timetable.csv")
            elif c == 5:
                open_folder(os.path.dirname(save_prefix) or ".")
        except Back:
            continue


def pick_algorithm(default: int = 5) -> str:
    c = ask_choice("Which algorithm should solve it?",
                   [n for _k, n in ALGOS], default=default, back_label=None)
    return ALGOS[c - 1][0]


def ask_time_limit(algo: str) -> float:
    if algo in ("bnb", "hybrid"):
        return float(ask_int("Time limit in seconds (the search stops and reports its best result)", 10, 1, 3600))
    return 10.0


def menu_generate() -> None:
    while True:
        clear_screen()
        banner("Generate a problem", "Create a test problem, solve it, and see the timetable")
        c = ask_choice("What kind of problem?", [
            "UZ-inspired university",
            "Random",
            "Crown puzzle",
        ], back_label="Back to the main menu")
        try:
            if c == 0:
                return
            clear_screen()
            if c == 1:
                n = ask_int("Number of exams", 60, 2, 2000)
                seed = ask_int("Seed (same seed = same problem every time)", 1, 0, 10 ** 9)
                adv = ask_yes_no("Change advanced settings (venues, seat tightness)?", False)
                venues, slot_factor = 8, 0.8
                if adv:
                    venues = ask_int("Number of venues", 8, 1, 30)
                    slot_factor = ask_float("Seat tightness (smaller = fewer seats per slot = harder)", 0.8, 0.2, 3.0)
                inst = uz_instance(n, seed=seed, slot_factor=slot_factor, n_venues=venues)
            elif c == 2:
                n = ask_int("Number of exams", 50, 2, 2000)
                seed = ask_int("Seed (same seed = same problem every time)", 1, 0, 10 ** 9)
                dens = ask_float("Crowdedness / density of clashes, 0.01 to 0.9 (0.12 is typical, 0.5 is very hard)",
                                 0.12, 0.01, 0.9)
                adv = ask_yes_no("Change advanced settings (venues, seat tightness)?", False)
                venues, slot_factor = 6, 0.8
                if adv:
                    venues = ask_int("Number of venues", 6, 1, 30)
                    slot_factor = ask_float("Seat tightness (smaller = fewer seats per slot = harder)", 0.8, 0.2, 3.0)
                inst = random_instance(n, density=dens, seed=seed, slot_factor=slot_factor, n_venues=venues)
            else:
                h = ask_int("Size h (the puzzle has 2h exams; the best answer is always 2 slots)", 20, 2, 500)
                inst = crown_graph(h)
                seed = 0
            algo = pick_algorithm()
            tl = ask_time_limit(algo)
            prefix = os.path.join(OUTPUT_DIR, safe_name(inst.name))
            solve_and_report(inst, algo, tl, seed, prefix)
        except Back:
            continue
        except Exception as e:
            err(f"Something went wrong: {e}")
            pause()


TEMPLATE = {
    "name": "my_problem",
    "venues": [100, 70],
    "exams": [
        {"code": "MAT101", "size": 60, "lecturers": [0]},
        {"code": "PHY101", "size": 50, "lecturers": [1]},
        {"code": "CHM101", "size": 40, "lecturers": [1]},
        {"code": "CSC101", "size": 30, "lecturers": [2]},
    ],
    "cohorts": {"0": [0, 1], "1": [1, 2, 3]},
}

TEMPLATE_HELP = """\
  How to fill in the file:
    name     any label
    venues   seats in each room, for example [100, 70]. All rooms are free in every slot.
    exams    one entry per exam, in order:
               code       the exam code
               size       number of students taking it
               lecturers  numbers identifying the lecturers (same number = same person,
                          so their exams cannot be at the same time)
    cohorts  each entry is a group of students (a class or programme) and lists the POSITIONS
             (0 = first exam above, 1 = second, ...) of the exams that group takes.
             Exams in the same group can never share a slot."""


def validate_problem_dict(d: dict) -> list:
    """Friendly checks before handing the data to the engine. Returns a list of error messages."""
    errs = []
    for key in ("name", "venues", "exams", "cohorts"):
        if key not in d:
            errs.append(f"The file has no '{key}' entry.")
    if errs:
        return errs
    if not isinstance(d["exams"], list) or not d["exams"]:
        return ["'exams' must be a non-empty list."]
    if not isinstance(d["venues"], list) or not all(isinstance(v, int) and v > 0 for v in d["venues"]):
        errs.append("'venues' must be a list of positive whole numbers, for example [100, 70].")
    for i, e in enumerate(d["exams"]):
        if not isinstance(e, dict) or "code" not in e or "size" not in e:
            errs.append(f"Exam number {i} needs at least a 'code' and a 'size'.")
            continue
        if not isinstance(e["size"], int) or e["size"] <= 0:
            errs.append(f"Exam {e['code']}: 'size' must be a positive whole number.")
        e.setdefault("lecturers", [])
    codes = [e.get("code") for e in d["exams"] if isinstance(e, dict)]
    if len(set(codes)) != len(codes):
        errs.append("Two exams have the same code. Each code must be unique.")
    if isinstance(d["cohorts"], dict):
        for k, members in d["cohorts"].items():
            if not isinstance(members, list):
                errs.append(f"Group '{k}' must be a list of exam positions.")
                continue
            for m in members:
                if not isinstance(m, int) or not (0 <= m < len(d["exams"])):
                    errs.append(f"Group '{k}' mentions exam position {m}, but exams are numbered 0 to {len(d['exams']) - 1}.")
    else:
        errs.append("'cohorts' must be an object such as {\"0\": [0, 1]}.")
    if d["venues"] and all(isinstance(v, int) and v > 0 for v in d["venues"]):
        cap = sum(d["venues"])
        for e in d["exams"]:
            if isinstance(e, dict) and isinstance(e.get("size"), int) and e["size"] > cap:
                errs.append(f"Exam {e['code']} has {e['size']} students but all venues together hold only {cap}.")
    return errs


def load_problem_file(path: str):
    try:
        with open(path) as f:
            d = json.load(f)
    except FileNotFoundError:
        err(f"File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        err(f"That file is not valid JSON ({e}). Check for a missing comma or bracket.")
        return None
    problems = validate_problem_dict(d) if isinstance(d, dict) else ["The file must contain a JSON object."]
    if problems:
        err("The problem file has mistakes:")
        for p in problems:
            print(red("      - ") + p)
        return None
    try:
        return from_dict(d)
    except Exception as e:
        err(f"Could not read the problem: {e}")
        return None


def find_json_files() -> list:
    found = []
    for folder in ("data", OUTPUT_DIR, "."):
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.endswith(".json"):
                    found.append(os.path.normpath(os.path.join(folder, fn)))
    return found


def build_by_typing():
    """Guided entry: the user types venues, exams and student groups."""
    section("Build your problem by answering questions")
    info("Press Ctrl+C at any time to cancel.")
    name = safe_name(ask_text("Name for this problem", "my_problem"))
    raw = ask_text("Venue capacities, separated by commas (e.g. 100,70,50)", "100,70")
    try:
        venues = [int(x) for x in raw.replace(" ", "").split(",") if x]
        if not venues or any(v <= 0 for v in venues):
            raise ValueError
    except ValueError:
        err("Venues must be positive whole numbers separated by commas.")
        return None
    n = ask_int("How many exams?", 4, 1, 500)
    exams, codes = [], []
    print(dim("  For each exam type the code, the number of students and the lecturer number."))
    for i in range(n):
        while True:
            code = ask_text(f"  Exam {i + 1} code", f"EX{i + 1:02d}")
            if code in codes:
                err("That code is already used.")
                continue
            break
        size = ask_int(f"    students taking {code}", 40, 1, sum(venues))
        lect = ask_int(f"    lecturer number for {code} (0, 1, 2, ...)", 0, 0, 10 ** 6)
        codes.append(code)
        exams.append({"code": code, "size": size, "lecturers": [lect]})
    print()
    info("Now the student groups. A group is a class or programme; its exams can never share a slot.")
    info("Type the exam codes for each group separated by commas. Leave empty to finish.")
    cohorts, gi = {}, 0
    while True:
        raw = ask_text(f"  Group {gi + 1} exams (e.g. {codes[0]},{codes[-1]})", "")
        if not raw:
            break
        members = []
        for tok in raw.replace(" ", "").split(","):
            if tok in codes:
                members.append(codes.index(tok))
            elif tok:
                err(f"Unknown exam code '{tok}'. Known codes: {', '.join(codes)}")
                members = None
                break
        if members is None:
            continue
        if len(members) < 2:
            warn("A group needs at least 2 exams to create a clash. Ignored.")
            continue
        cohorts[str(gi)] = sorted(set(members))
        gi += 1
    d = {"name": name, "venues": venues, "exams": exams, "cohorts": cohorts}
    problems = validate_problem_dict(d)
    if problems:
        for p in problems:
            err(p)
        return None
    if ask_yes_no("Save this problem to a file so you can reload it later?", True):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        p = os.path.join(OUTPUT_DIR, name + "_problem.json")
        with open(p, "w") as f:
            json.dump(d, f, indent=1)
        ok(f"Saved {rel(p)}")
    return from_dict(d)


def menu_own_problem() -> None:
    while True:
        clear_screen()
        banner("Use my own problem", "Give the engine your exams and get a clash-free timetable")
        c = ask_choice("How do you want to provide your problem?", [
            "Load a problem file (JSON)",
            "Type it in step by step",
            "Create a template file to edit",
        ], back_label="Back to the main menu")
        try:
            inst = None
            if c == 0:
                return
            clear_screen()
            if c == 1:
                files = find_json_files()
                path = None
                if files:
                    print()
                    print(bold("Problem files found:"))
                    for i, fp in enumerate(files, 1):
                        print(f"  {cyan(str(i))}) {fp}")
                    print(f"  {cyan('p')}) type a different path")
                    pick = ask_text("Choose a number, or p to type a path", "1")
                    if pick.isdigit() and 1 <= int(pick) <= len(files):
                        path = files[int(pick) - 1]
                if path is None:
                    path = ask_text("Path to the JSON problem file", "data/uz60.json")
                inst = load_problem_file(path)
            elif c == 2:
                inst = build_by_typing()
            elif c == 3:
                p = "my_problem_template.json"
                if os.path.exists(p) and not ask_yes_no(f"{p} already exists. Overwrite it?", False):
                    info("Left the existing file as it is.")
                else:
                    with open(p, "w") as f:
                        json.dump(TEMPLATE, f, indent=1)
                    ok(f"Created {p}")
                print(TEMPLATE_HELP)
                info("Edit the file in any text editor, save it, then choose option 1 here and load it.")
                pause()
                continue
            if inst is None:
                pause()
                continue
            algo = pick_algorithm()
            tl = ask_time_limit(algo)
            prefix = os.path.join(OUTPUT_DIR, safe_name(inst.name))
            solve_and_report(inst, algo, tl, 0, prefix)
        except Back:
            continue
        except Exception as e:
            err(f"Something went wrong: {e}")
            pause()


EXPERIMENTS = [
    # key, title, output csv
    ("pruning", "Experiment 1 - Pruning", "exp1_pruning.csv"),
    ("scaling", "Experiment 2 - Scaling", "exp2_scaling.csv"),
    ("density", "Experiment 3 - Density", "exp3_density.csv"),
    ("bounds", "Experiment 4 - Lower bounds", "exp4_bounds.csv"),
    ("crown", "Experiment 5 - Greedy worst case", "exp5_crown.csv"),
    ("trace", "Experiment 6 - Improvement over time", "exp6_trace.csv"),
]
EXP_KEYS = {k: (t, f) for k, t, f in EXPERIMENTS}


def count_rows(path: str) -> int:
    try:
        with open(path) as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def show_result_locations(out_dir: str, keys: list) -> None:
    section("Where the results are")
    print(f"  Folder: {bold(rel(out_dir))}{os.sep}")
    for k in keys:
        f = EXP_KEYS[k][1]
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            ok(f"{rel(p):<40} {count_rows(p)} rows   <- {EXP_KEYS[k][0]}")
        else:
            warn(f"{rel(p):<40} was not created")
    info("Each CSV holds the raw numbers (one row per run). Open them in Excel.")


def analyse_results(out_dir: str) -> None:
    missing = have_modules("numpy", "scipy", "pandas", "matplotlib")
    if missing:
        err("Tables and charts need these Python packages: " + ", ".join(missing))
        info("Install them with:   pip install -r requirements.txt")
        return
    cmd = ["experiments/analyze.py", "--results", rel(out_dir)]
    print(dim("  $ " + " ".join([os.path.basename(sys.executable)] + cmd)))
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True)
    except KeyboardInterrupt:
        print()
        warn("Stopped by you (Ctrl+C).")
        return
    skipped = [ln for ln in p.stdout.splitlines() if ln.startswith("skipping")]
    for ln in p.stdout.splitlines():
        if not ln.startswith("skipping"):
            print("  " + ln)
    if skipped:
        info(f"({len(skipped)} experiment(s) have no results in this folder yet, so they were left out.)")
    print(dim(f"  (finished in {time.time() - t0:.1f}s)"))
    if p.returncode != 0:
        err("The analysis step stopped with an error. The raw CSV files are still available.")
        print(dim("  " + (p.stderr.strip().splitlines() or [""])[-1]))
        return
    summary = os.path.join(out_dir, "summary.md")
    figs = os.path.join(out_dir, "figures")
    section("Tables and charts")
    ok(f"Tables and statistics : {rel(summary)}")
    if os.path.isdir(figs):
        pngs = sorted(f for f in os.listdir(figs) if f.endswith(".png"))
        ok(f"Charts ({len(pngs)} pictures): {rel(figs)}{os.sep}")
        for f in pngs:
            print(dim(f"        {f}"))
    if os.path.exists(summary) and ask_yes_no("Show the first part of the summary here?", False):
        with open(summary) as f:
            lines = f.read().splitlines()
        print()
        for ln in lines[:45]:
            print("  " + ln[:WIDTH + 20])
        if len(lines) > 45:
            print(dim(f"  ... ({len(lines) - 45} more lines in {rel(summary)})"))


def choose_run_settings():
    """Ask for run size and results folder. Returns (quick, out_dir), or None if the user backs out."""
    mode = ask_choice("How big a run?", ["Quick", "Full"], default=1, back_label=None)
    loc = ask_choice("Where should the results be written?",
                     [f"{DEMO_RESULTS}/", f"{OFFICIAL_RESULTS}/"], default=1, back_label=None)
    out_dir = os.path.join(ROOT, DEMO_RESULTS if loc == 1 else OFFICIAL_RESULTS)
    if loc == 2 and not ask_yes_no("This replaces the report's results files. Are you sure?", False):
        return None
    return mode == 1, out_dir


def menu_experiments() -> None:
    while True:
        clear_screen()
        banner("Experiments", "Choose an experiment, run it, and see where the results are")
        opts = [t for _k, t, _f in EXPERIMENTS]
        opts.append("Run ALL six experiments")
        c = ask_choice("Which experiment?", opts, back_label="Back to the main menu")
        try:
            if c == 0:
                return
            clear_screen()
            keys = [EXPERIMENTS[c - 1][0]] if c <= len(EXPERIMENTS) else [e[0] for e in EXPERIMENTS]

            settings = choose_run_settings()
            if settings is None:
                continue
            quick, out_dir = settings
            if not quick and len(keys) > 1:
                warn("A full run of everything takes a long time. Press Ctrl+C to stop it at any moment.")
                if not ask_yes_no("Start the full run?", True):
                    continue

            section("Running")
            args = ["experiments/run_experiments.py", "--out", rel(out_dir), "--only"] + keys
            if quick:
                args.insert(1, "--quick")
            rc = run_script(args)
            if rc != 0:
                err("The experiment run did not finish properly (see the messages above).")
                pause()
                continue
            show_result_locations(out_dir, keys)

            if ask_yes_no("Turn the numbers into tables and charts now?", True):
                analyse_results(out_dir)
            offer_open(out_dir)
            pause()
        except Back:
            continue
        except Exception as e:
            err(f"Something went wrong: {e}")
            pause()


def menu_pipeline() -> None:
    banner("Full pipeline", "Tests, all experiments, tables and charts, illustrative figures")
    try:
        settings = choose_run_settings()
        if settings is None:
            return
        quick, out_dir = settings
        run_tests = ask_yes_no("Run the unit tests first?", True)
        if not quick:
            warn("A full run takes a long time. Press Ctrl+C to stop it at any moment.")
            if not ask_yes_no("Start the full run?", True):
                return

        section("Running")
        if run_tests and run_script(["-m", "unittest", "discover", "-s", "tests"]) != 0:
            err("The tests failed, so the pipeline was stopped.")
            pause()
            return
        args = ["experiments/run_experiments.py", "--out", rel(out_dir)]
        if quick:
            args.append("--quick")
        if run_script(args) != 0:
            err("The experiment run did not finish properly (see the messages above).")
            pause()
            return
        show_result_locations(out_dir, [e[0] for e in EXPERIMENTS])

        analyse_results(out_dir)
        if run_script(["experiments/make_figures.py", "--results", rel(out_dir)]) != 0:
            err("The illustrative figures could not be built (see the messages above).")
        offer_open(out_dir)
        pause()
    except Back:
        return
    except Exception as e:
        err(f"Something went wrong: {e}")
        pause()


def menu_selfcheck() -> None:
    banner("Self-check", "Runs the automatic tests of the engine")
    run_script(["-m", "unittest", "discover", "-s", "tests"])
    pause()


def main() -> int:
    if sys.version_info < (3, 9):
        print("This project needs Python 3.9 or newer.")
        return 1
    while True:
        clear_screen()
        banner("ETE - Exam Timetabling Engine",
               "Interactive demo.  Press Enter to accept [defaults].  Ctrl+C goes back.")
        try:
            c = ask_choice("MAIN MENU", [
                "Experiments",
                "Generate a problem",
                "Use my own problem",
                "Full pipeline",
                "Self-check",
            ], back_label="Quit")
        except Back:
            print()
            break
        if c == 0:
            break
        clear_screen()
        try:
            {1: menu_experiments, 2: menu_generate, 3: menu_own_problem,
             4: menu_pipeline, 5: menu_selfcheck}[c]()
        except Back:
            continue
    print(dim("\n  Goodbye.\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
