# Exam Timetabling Engine (ETE)

A hybrid CSP / graph-colouring solver for university examination timetabling
(project: *NP-Hard Scheduling Engine for University Exam Timetabling*).

Given exams, student cohorts, lecturers and venue capacities, ETE produces a **conflict-free
timetable using as few exam slots as possible**, and lets you compare six algorithmic ideas on the same problem:

| Algorithm | Module | Role |
|---|---|---|
| Naive chronological backtracking | `ete/csp.py` (`mode="naive"`) | baseline, $O(k^n)$ |
| Backtracking + forward checking | `mode="fc"` | pruning |
| BT + FC + MRV + degree heuristic | `mode="fc_mrv"` | efficient search |
| BT + MAC (singleton propagation) + MRV | `mode="mac"` | stronger propagation |
| Branch-and-bound with clique / seat / big-exam lower bounds | `ete/bnb.py`, `ete/bounds.py` | exact optimum |
| Greedy colouring (First-Fit, Welsh-Powell, DSATUR) | `ete/greedy.py` | approximation |
| TabuCol local search (with capacity penalty) | `ete/local_search.py` | upper bounds |
| **Hybrid** B&B + local search | `ete/hybrid.py` | practical solver |

## Model in one paragraph

Exams are variables whose domain is the set of `k` slots. Two exams that share a student cohort or a
lecturer must differ (conflict graph = graph colouring); the seats used in one slot may not exceed the total venue
capacity. The objective is to minimise `k`. See `../ete-paperwork/CSP_Model.docx`; NP-completeness
(3-SAT -> 3-COLOURING -> ETT) is proved in `../ete-paperwork/NP_Completeness_Proof.docx` and the reductions are
executable in `ete/reductions.py`.

## Quick start

```bash
pip install -r requirements.txt            # numpy scipy pandas matplotlib networkx
python app.py                              # interactive menu: the single entry point for everything
```

`app.py` is a menu-driven interface for the whole project:

| Menu entry | What it does |
|---|---|
| Experiments | run one experiment (or all six), quick or full, then build tables and charts |
| Generate a problem | create a UZ-inspired, random or crown-puzzle instance and solve it |
| Use my own problem | load a JSON file, type a problem in, or write a template to edit |
| Full pipeline | unit tests, all experiments, tables and charts, illustrative figures |
| Self-check | run the unit tests |

Experiment output goes to `results_demo/` by default; the project's own `results/` (used by the report) is only
overwritten if you choose it explicitly and confirm. Solved timetables are written to `output/`.

Python API:

```python
from ete.generators import uz_instance
from ete.hybrid import solve_hybrid
from ete.validate import validate_assignment
inst = uz_instance(120, seed=3)
res  = solve_hybrid(inst, time_limit=10)
print(res.num_slots, res.lower_bound, res.optimal, validate_assignment(inst, res.slot_of)[0])
```

## Repository layout

```
app.py          interactive entry point (menus for experiments, problem solving, full pipeline, tests)
ete/            the engine (model, generators, solvers, bounds, reductions, validator, CLI helpers)
tests/          unit + property tests (python -m unittest)
experiments/    run_experiments.py -> results/*.csv ; analyze.py -> summary.md + figures ; make_figures.py
results/        CSV data, summary.md (tables, CIs, hypothesis tests), figures/*.png
data/           a sample instance + timetable
```

## Deliverables map (project brief -> files)

| Deliverable | File |
|---|---|
| Literature review (5 pages) | `../ete-paperwork/Literature_Review.docx` |
| CSP modelling - formal model | `../ete-paperwork/CSP_Model.docx` |
| NP-completeness proof | `../ete-paperwork/NP_Completeness_Proof.docx` |
| Algorithm implementations | `ete/` |
| Experimental evaluation (benchmarks, plots) | `experiments/`, `results/` |
| Final report | `../ete-paperwork/Final_Report.docx` |
| Presentation (20 min) | `../ete-presentantion/ETE_Presentation_Revamped.pptx` |
| Code repository (Git, documented) | this repository |
| Peer evaluations | `../ete-paperwork/Peer_Evaluation.docx` |

## Honest notes and limitations

* The "UZ-inspired" generator is **synthetic**: it imitates faculty/programme/level structure, service courses,
  carry-over cohorts and shared lecturers; venue names and capacities are illustrative. No official UZ data was used.
  Plug in real enrolment data through *Use my own problem* in `app.py`, or by building an `Instance` from your own cohorts (see `ete/cli.py: from_dict`).
* Exams may be split across venues inside a slot (Lemma 3.1). Unsplittable exams and soft constraints (spreading,
  back-to-back exams) are not modelled.
* Time-limited results (B&B/hybrid at n >= 100) depend slightly on machine speed.
* Please verify the bibliographic details in the report against the originals before submission.
* Group member names go in the report title page / Appendix B (left blank on purpose).

## Git

```bash
git init && git add . && git commit -m "ETE: exam timetabling engine"
```
