"""Phase 1 evidence for gate G1: defect histograms and linear-time scaling.

Run after the week 3 exercises pass:

    python experiments/phase1_sanity.py            # quick (about a minute on CPU)
    python experiments/phase1_sanity.py --full     # up to 1e5 edges, for the report

Writes plots and a JSON summary to experiments/out/. Thresholds come from
notes/protocol.md; keep the two in sync.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtvm.defects import chord_defects, holonomy_scores  # noqa: E402
from rtvm.synthetic import consistent_graph, corrupt_edges, random_connected_graph  # noqa: E402
from rtvm.utils import seed_everything  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CONSISTENT_MAX = 1e-4   # G1: defects on consistent graphs stay below this
CORRUPT_MIN = 1e-2      # G1: every cycle through a corrupted edge exceeds this
MAX_SLOPE = 1.3         # G1: log-log slope of time vs |V|+|E| (cache effects push it a little above 1)


def histograms(n_graphs: int, n_blocks: int, b: int) -> dict:
    clean, dirty, missed = [], [], 0
    for s in range(n_graphs):
        edges = random_connected_graph(40, 100, seed=s)
        graph, _ = consistent_graph(40, edges, n_blocks, b, seed=s)
        tree = graph.spanning_tree(0)
        clean += holonomy_scores(graph, tree).tolist()
        bad, idx = corrupt_edges(graph, 3, seed=s)
        tree = bad.spanning_tree(0)
        for c, score in zip(tree.chords, holonomy_scores(bad, tree).tolist()):
            if bad.cycle_edges(tree, c) & set(idx):
                dirty.append(score)
                missed += score <= CORRUPT_MIN
    return {"clean": clean, "dirty": dirty, "missed": int(missed)}


def scaling(sizes: list[int], n_blocks: int, b: int, repeats: int = 3) -> list[tuple[int, float]]:
    rows = []
    for n_edges in sizes:
        n_vertices = max(10, n_edges // 3)
        edges = random_connected_graph(n_vertices, n_edges, seed=n_edges)
        graph, _ = consistent_graph(n_vertices, edges, n_blocks, b, seed=0)
        tree = graph.spanning_tree(0)
        psi = torch.randn(graph.dim)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            chord_defects(graph, tree, psi)
            times.append(time.perf_counter() - t0)
        rows.append((n_vertices + n_edges, min(times)))
        print(f"  |V|+|E| = {n_vertices + n_edges:>7}: {min(times) * 1e3:8.1f} ms")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="scale up to 1e5 edges")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed_everything(args.seed)
    OUT.mkdir(exist_ok=True)

    print("Defect histograms (2 blocks of 64)...")
    h = histograms(20, 2, 64)
    sizes = [100, 300, 1000, 3000, 10000] + ([30000, 100000] if args.full else [])
    print("Scaling of chord_defects (2 blocks of 64):")
    rows = scaling(sizes, 2, 64)
    x, y = np.log([r[0] for r in rows]), np.log([r[1] for r in rows])
    slope = float(np.polyfit(x[1:], y[1:], 1)[0])  # skip the smallest size (overhead-dominated)

    g1 = {
        "max_clean_defect": max(h["clean"]),
        "min_corrupted_defect": min(h["dirty"]),
        "corrupted_cycles_missed": h["missed"],
        "loglog_slope": slope,
        "pass_consistent": max(h["clean"]) < CONSISTENT_MAX,
        "pass_corrupted": h["missed"] == 0,
        "pass_linear": slope < MAX_SLOPE,
    }
    (OUT / "phase1_summary.json").write_text(json.dumps(g1, indent=2))
    print(json.dumps(g1, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.logspace(-9, 2, 60)
    ax.hist(np.clip(h["clean"], 1e-9, None), bins=bins, alpha=0.7, label="consistent graphs")
    ax.hist(h["dirty"], bins=bins, alpha=0.7, label="cycles through a corrupted edge")
    ax.axvline(CORRUPT_MIN, color="k", ls="--", lw=1, label="G1 threshold")
    ax.set_xscale("log")
    ax.set_xlabel("holonomy score |H - I|_F")
    ax.set_ylabel("chords")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "defect_histogram.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.loglog([r[0] for r in rows], [r[1] * 1e3 for r in rows], "o-")
    ax.set_xlabel("|V| + |E|")
    ax.set_ylabel("chord_defects time (ms)")
    ax.set_title(f"log-log slope {slope:.2f} (1.0 = linear)")
    fig.tight_layout()
    fig.savefig(OUT / "scaling.png", dpi=150)
    print(f"Plots written to {OUT}")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as exc:
        sys.exit(f"Not ready yet: {exc}. Finish the week 3 exercises first (pytest -m week3).")
