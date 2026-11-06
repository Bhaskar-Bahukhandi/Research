"""Phase 1, week 4 experiments (plan: Nov 24, Nov 30, Dec 2, Dec 3).

    python experiments/phase1_week4.py localise   # accuracy vs number of corruptions and trees
    python experiments/phase1_week4.py noise      # does the defect survive noisy rotations?
    python experiments/phase1_week4.py timing     # Woodbury vs dense Cayley; engine vs dense SVD
    python experiments/phase1_week4.py all

Each part writes experiments/out/week4_<part>.json and a plot. These are the
figures for the Phase 1 report; G1 itself is decided by phase1_sanity.py.
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
from rtvm.localise import bridges, cut_pair_classes, localise_corrupted_edges  # noqa: E402
from rtvm.rotations import cayley, cayley_lowrank_apply, lowrank_skew  # noqa: E402
from rtvm.synthetic import consistent_graph, corrupt_edges, random_connected_graph, small_rotations  # noqa: E402
from rtvm.utils import seed_everything  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


# ---------------------------------------------------------------------------
# Nov 30: localisation accuracy
# ---------------------------------------------------------------------------

def localise(n_graphs: int = 20) -> dict:
    """Recall of corrupted edges among the top-scored edges.

    For each graph with m corrupted non-bridge edges, take the m highest-scoring
    edges (ties broken at random, so a tie never looks like a success) and
    count how many corrupted edges they contain ("exact"). "group" also counts
    a hit when a top edge is in the same cut-pair group as a corrupted edge:
    the best any cycle-based method can do (needs your cut_pair_classes).

    Blocks are 2 x 8: localisation depends on the graph, not on the width.
    """
    try:
        cut_pair_classes(consistent_graph(3, [(0, 1), (1, 2), (2, 0)], 1, 2)[0])
        have_groups = True
    except NotImplementedError:
        have_groups = False
        print("  cut_pair_classes not implemented yet: reporting exact-edge recall only")

    settings = {"dense (40 V, 100 E)": (40, 100), "sparse (40 V, 52 E)": (40, 52)}
    ks, tree_counts = [1, 2, 3, 5, 8], [1, 5, 20]
    results: dict = {"have_groups": have_groups, "settings": {}}
    for name, (n_v, n_e) in settings.items():
        rows, n_bridge, n_corrupt, n_shared = [], 0, 0, 0
        for k in ks:
            for n_trees in tree_counts:
                exact, group = [], []
                for s in range(n_graphs):
                    edges = random_connected_graph(n_v, n_e, seed=s)
                    graph, _ = consistent_graph(n_v, edges, 2, 8, seed=s)
                    bad, idx = corrupt_edges(graph, k, seed=1000 + s)
                    unseen = bridges(bad)
                    found = [i for i in idx if i not in unseen]
                    if n_trees == tree_counts[0]:
                        n_bridge += len(idx) - len(found)
                        n_corrupt += len(idx)
                    if not found:
                        continue
                    scores = localise_corrupted_edges(bad, n_trees=n_trees, seed=s)
                    perm = torch.randperm(bad.n_edges, generator=torch.Generator().manual_seed(s))
                    order = perm[torch.argsort(scores[perm], descending=True, stable=True)]
                    top = set(order[:len(found)].tolist())
                    exact.append(len(top & set(found)) / len(found))
                    if have_groups:
                        labels = cut_pair_classes(bad)
                        top_groups = {labels[t] for t in top}
                        group.append(sum(labels[i] in top_groups for i in found) / len(found))
                        if n_trees == tree_counts[0]:
                            n_shared += sum(labels.count(labels[i]) > 1 for i in found)
                rows.append({"k": k, "n_trees": n_trees, "exact": float(np.mean(exact)),
                             "group": float(np.mean(group)) if group else None})
                print(f"  {name:20s} k={k} trees={n_trees:2d} exact={rows[-1]['exact']:.2f}"
                      + (f" group={rows[-1]['group']:.2f}" if group else ""))
        results["settings"][name] = {
            "rows": rows,
            "corrupted_edges_that_were_bridges": n_bridge / n_corrupt,
            "corrupted_edges_in_a_shared_group": (n_shared / (n_corrupt - n_bridge)) if have_groups else None,
        }
        print(f"  {name}: {n_bridge}/{n_corrupt} corrupted edges were bridges (undetectable)")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, (name, res) in zip(axes, results["settings"].items()):
        for n_trees in tree_counts:
            sel = [r for r in res["rows"] if r["n_trees"] == n_trees]
            line, = ax.plot(ks, [r["exact"] for r in sel], "o-", label=f"{n_trees} trees, exact edge")
            if have_groups:
                ax.plot(ks, [r["group"] for r in sel], "s--", color=line.get_color(), alpha=0.7,
                        label=f"{n_trees} trees, cut-pair group")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("corrupted edges k")
    axes[0].set_ylabel("recall of detectable corrupted edges")
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "week4_localise.png", dpi=150)
    return results


# ---------------------------------------------------------------------------
# Dec 3: noise test
# ---------------------------------------------------------------------------

def noise(n_graphs: int = 10) -> dict:
    """AUC of the holonomy score at separating corrupted from clean cycles as noise grows.

    Every edge gets noise of size sigma (a stand-in for imperfectly learned
    rotations); k = 3 edges also get a corruption of size tau. A cycle is
    "positive" if it passes through a corrupted edge. tau = "random" replaces
    the rotation outright, as corrupt_edges does (|R - I|_F about sqrt(2d) = 16).
    Also records how the clean score grows with cycle length.
    """
    n_v, n_e, n_blocks, b, k = 40, 100, 2, 64, 3
    sigmas = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    taus: list = [0.3, 1.0, 3.0, "random"]
    results: dict = {"sigmas": sigmas, "taus": [str(t) for t in taus], "auc": {}, "length": {}}
    by_length: dict = {s: {} for s in sigmas}
    for tau in taus:
        aucs = []
        for sigma in sigmas:
            pos, neg = [], []
            for g in range(n_graphs):
                gen = torch.Generator().manual_seed(10_000 * g + 7)
                edges = random_connected_graph(n_v, n_e, seed=g)
                graph, _ = consistent_graph(n_v, edges, n_blocks, b, seed=g)
                if tau == "random":
                    graph, idx = corrupt_edges(graph, k, seed=g)
                    rots = graph.rotations.clone()
                else:
                    idx = sorted(torch.randperm(n_e, generator=gen)[:k].tolist())
                    rots = graph.rotations.clone()
                    rots[idx] = small_rotations(k, n_blocks, b, tau, generator=gen) @ rots[idx]
                if sigma > 0:
                    rots = small_rotations(n_e, n_blocks, b, sigma, generator=gen) @ rots
                noisy = graph.with_rotations(rots)
                tree = noisy.spanning_tree(0)
                for c, score in zip(tree.chords, holonomy_scores(noisy, tree).tolist()):
                    cycle = noisy.cycle_edges(tree, c)
                    if cycle & set(idx):
                        pos.append(score)
                    else:
                        neg.append(score)
                        if tau == taus[0]:
                            by_length[sigma].setdefault(len(cycle), []).append(score)
            aucs.append(auc(np.array(pos), np.array(neg)))
            print(f"  tau={str(tau):6s} sigma={sigma:<5} AUC={aucs[-1]:.3f}")
        results["auc"][str(tau)] = aucs
    for sigma in sigmas[1:]:
        lengths = sorted(L for L, v in by_length[sigma].items() if len(v) >= 5)
        means = [float(np.mean(by_length[sigma][L])) for L in lengths]
        slope = float(np.polyfit(np.log(lengths), np.log(means), 1)[0]) if len(lengths) > 2 else None
        results["length"][str(sigma)] = {"lengths": lengths, "mean_clean_score": means, "loglog_slope": slope}
        print(f"  sigma={sigma}: clean score ~ length^{slope:.2f}" if slope else "")

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    xs = [max(s, 3e-3) for s in sigmas]
    for tau in results["taus"]:
        axes[0].semilogx(xs, results["auc"][tau], "o-", label=f"corruption size {tau}")
    axes[0].set_xlabel("noise per edge |R - I|_F  (leftmost point: no noise)")
    axes[0].set_ylabel("AUC, corrupted vs clean cycles")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].axhline(0.5, color="k", lw=0.5, ls=":")
    axes[0].legend(frameon=False, fontsize=7)
    for sigma, res in results["length"].items():
        axes[1].loglog(res["lengths"], res["mean_clean_score"], "o-", label=f"noise {sigma}")
    axes[1].set_xlabel("cycle length (edges)")
    axes[1].set_ylabel("mean clean holonomy score")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "week4_noise.png", dpi=150)
    return results


# ---------------------------------------------------------------------------
# Nov 24 and Dec 2: timing contrasts
# ---------------------------------------------------------------------------

def _best_of(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def timing(full: bool = False) -> dict:
    """(a) Low-rank Cayley through Woodbury vs forming the dense b x b rotation,
    one different rotation per edge, b = 64. (b) Chord defects vs the dense
    route (singular values of the full delta^0 matrix), small graphs, 2 x 8 blocks."""
    results: dict = {"woodbury": [], "svd": []}
    n, b = 4096, 64
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(n, b, generator=gen)
    for r in (1, 2, 4, 8):
        U, V = torch.randn(n, b, r, generator=gen), torch.randn(n, b, r, generator=gen)
        t_fast = _best_of(lambda: cayley_lowrank_apply(U, V, x))
        t_dense = _best_of(lambda: (cayley(lowrank_skew(U, V)) @ x.unsqueeze(-1)).squeeze(-1))
        results["woodbury"].append({"r": r, "woodbury_ms": t_fast * 1e3, "dense_ms": t_dense * 1e3})
        print(f"  r={r}: Woodbury {t_fast * 1e3:7.1f} ms, dense {t_dense * 1e3:7.1f} ms "
              f"({t_dense / t_fast:.0f}x) for {n} edges at b={b}")

    for n_e in [30, 100, 300] + ([1000] if full else []):
        n_v = max(10, n_e // 3)
        graph, _ = consistent_graph(n_v, random_connected_graph(n_v, n_e, seed=n_e), 2, 8, seed=0)
        tree = graph.spanning_tree(0)
        psi = torch.randn(graph.dim)
        d = graph.dim
        D = torch.zeros(n_e * d, n_v * d)
        for e, (u, v) in enumerate(graph.edges):
            D[e * d:(e + 1) * d, u * d:(u + 1) * d] = torch.block_diag(*graph.rotations[e])
            D[e * d:(e + 1) * d, v * d:(v + 1) * d] -= torch.eye(d)
        t_engine = _best_of(lambda: chord_defects(graph, tree, psi))
        t_svd = _best_of(lambda: torch.linalg.svdvals(D), repeats=1)
        results["svd"].append({"n_edges": n_e, "engine_ms": t_engine * 1e3, "dense_svd_ms": t_svd * 1e3,
                               "matrix_shape": list(D.shape)})
        print(f"  |E|={n_e:5d}: engine {t_engine * 1e3:7.2f} ms, dense SVD of "
              f"{tuple(D.shape)} {t_svd * 1e3:9.1f} ms")
    e = np.log([row["n_edges"] for row in results["svd"]])
    for key in ("engine_ms", "dense_svd_ms"):
        results[f"{key}_loglog_slope"] = float(np.polyfit(e, np.log([row[key] for row in results["svd"]]), 1)[0])
    print(f"  slopes: engine {results['engine_ms_loglog_slope']:.2f}, "
          f"dense SVD {results['dense_svd_ms_loglog_slope']:.2f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["localise", "noise", "timing", "all"])
    ap.add_argument("--full", action="store_true", help="timing: include |E| = 1000 for the SVD")
    args = ap.parse_args()
    seed_everything(0)
    OUT.mkdir(exist_ok=True)
    parts = ["localise", "noise", "timing"] if args.part == "all" else [args.part]
    for part in parts:
        print(f"[{part}]")
        result = timing(args.full) if part == "timing" else globals()[part]()
        (OUT / f"week4_{part}.json").write_text(json.dumps(result, indent=2))
    print(f"Results written to {OUT}")


if __name__ == "__main__":
    main()
