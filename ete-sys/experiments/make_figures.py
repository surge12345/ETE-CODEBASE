"""Conceptual / illustrative figures: architecture, OR-gadget, conflict graph, timetable."""
from __future__ import annotations
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ete.bnb import solve_bnb
from ete.generators import uz_instance
from ete.hybrid import solve_hybrid

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
F = os.path.join(ap.parse_known_args()[0].results, "figures")
os.makedirs(F, exist_ok=True)
PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
       "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000"]


def box(ax, x, y, w, h, text, fc="#ecf0f1", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#2c3e50", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def architecture():
    fig, ax = plt.subplots(figsize=(7.5, 5.2)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    box(ax, 1, 7, 8, 0.8, "INPUT: exams - venues - lecturers - student cohorts - constraints", "#d6eaf8", 8.5, True)
    box(ax, 1, 5.7, 8, 0.8, "MODELLING: conflict graph G=(E,C) + capacity  ->  CSP / graph colouring (ete.instance)", "#d5f5e3", 7.5, True)
    ax.add_patch(FancyBboxPatch((0.6, 1.9), 8.8, 3.3, boxstyle="round,pad=0.02", fc="#fdfefe", ec="#7f8c8d", lw=1, ls="--"))
    ax.text(5, 4.95, "SOLVER LAYER", ha="center", fontsize=9, weight="bold")
    box(ax, 0.9, 3.9, 2.7, 0.8, "Naive backtracking\n(baseline)", "#fadbd8")
    box(ax, 3.8, 3.9, 2.7, 0.8, "BT + FC + MRV + degree\n(+ MAC)", "#fdebd0")
    box(ax, 6.7, 3.9, 2.5, 0.8, "Greedy: FF / WP / DSATUR\n(approximation)", "#e8daef")
    box(ax, 0.9, 2.2, 4.0, 1.2, "Branch & Bound (DSATUR branching)\nlower bounds: clique, seats, big exams", "#d6eaf8")
    box(ax, 5.2, 2.2, 4.0, 1.2, "HYBRID: DSATUR -> B&B probe -> TabuCol\n-> B&B with tightened UB", "#d1f2eb", 8, True)
    box(ax, 1, 0.8, 8, 0.7, "EVALUATION: nodes explored - time - quality (slots, conflicts) - CIs & tests", "#fcf3cf", 7.5, True)
    box(ax, 1, -0.1, 8, 0.7, "OUTPUT: conflict-free timetable + venue allocation (validated)", "#d6eaf8", 8.5, True)
    for y0, y1 in ((7, 6.5), (5.7, 5.2), (1.9, 1.5), (0.8, 0.6)):
        ax.annotate("", xy=(5, y1), xytext=(5, y0), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_ylim(-0.3, 8)
    fig.savefig(os.path.join(F, "fig_architecture.png"), bbox_inches="tight"); plt.close(fig)


def or_gadget():
    G = nx.Graph()
    pos = {"a": (0, 2), "b": (0, 0), "p": (2, 2), "q": (2, 0), "o": (4, 1)}
    G.add_edges_from([("p", "a"), ("q", "b"), ("p", "q"), ("p", "o"), ("q", "o")])
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.6)
    cols = {"a": "#fdebd0", "b": "#fdebd0", "p": "#d6eaf8", "q": "#d6eaf8", "o": "#d5f5e3"}
    nx.draw_networkx_nodes(G, pos, node_color=[cols[n] for n in G.nodes], edgecolors="k", node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11)
    ax.text(0, 2.55, "input literal", ha="center", fontsize=8); ax.text(4, 1.55, "output", ha="center", fontsize=8)
    ax.set_title("OR-gadget: o can be coloured T iff a or b is coloured T", fontsize=9)
    ax.axis("off"); ax.set_ylim(-0.6, 3)
    fig.savefig(os.path.join(F, "fig_or_gadget.png"), bbox_inches="tight"); plt.close(fig)


def sample_timetable():
    inst = uz_instance(36, seed=4)
    r = solve_hybrid(inst, time_limit=5)
    G = nx.Graph(); G.add_nodes_from(range(inst.n))
    G.add_edges_from((u, v) for u in range(inst.n) for v in inst.adj_list[u] if v > u)
    pos = nx.spring_layout(G, seed=3, k=0.9)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    nx.draw_networkx_nodes(G, pos, node_color=[PAL[r.slot_of[v] % len(PAL)] for v in G.nodes], node_size=260,
                           edgecolors="k", linewidths=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={v: str(v) for v in G.nodes}, font_size=6, ax=ax)
    ax.set_title(f"Conflict graph of a UZ-inspired instance (n={inst.n}, m={inst.m}); colour = exam slot; "
                 f"{r.num_slots} slots", fontsize=9)
    ax.axis("off")
    fig.savefig(os.path.join(F, "fig_conflict_graph.png"), bbox_inches="tight"); plt.close(fig)

    slots = {}
    for e, s in enumerate(r.slot_of):
        slots.setdefault(s, []).append(e)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for s in sorted(slots):
        bottom = 0
        for i, e in enumerate(sorted(slots[s], key=lambda x: -inst.size[x])):
            ax.bar(s + 1, inst.size[e], bottom=bottom, color=PAL[(e) % len(PAL)], edgecolor="white", lw=0.5)
            if inst.size[e] > 0.07 * inst.capacity:
                ax.text(s + 1, bottom + inst.size[e] / 2, inst.exams[e].code, ha="center", va="center", fontsize=5.5)
            bottom += inst.size[e]
    ax.axhline(inst.capacity, color="r", ls="--", label=f"total venue capacity = {inst.capacity}")
    ax.set_xlabel("exam slot"); ax.set_ylabel("seats used"); ax.legend(fontsize=7)
    ax.set_title("Seat load per slot (capacity constraint C2)", fontsize=9)
    fig.savefig(os.path.join(F, "fig_slot_load.png"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    architecture(); or_gadget(); sample_timetable()
    print("figures written to", F)
