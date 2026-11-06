"""Locating corrupted edges from cycle defects.

Exercise (week 4, Mon Nov 30). Spec: tests/test_localise.py.
"""
from __future__ import annotations

import torch  # noqa: F401
from torch import Tensor

from .defects import holonomy_scores  # noqa: F401
from .graph import RotationGraph


def localise_corrupted_edges(graph: RotationGraph, n_trees: int = 5, threshold: float = 1e-3,
                             seed: int = 0) -> Tensor:
    """Score each edge by how often the cycles through it are defective, shape (n_edges,), in [0, 1].

    Suggested method (you may find a better one, as long as the tests pass):
      * Repeat ``n_trees`` times: pick a random root and random edge weights
        (``torch.Generator().manual_seed(seed)`` for reproducibility), build
        ``graph.spanning_tree(root, weights)``, and compute ``holonomy_scores``.
      * For every chord, walk its cycle (``graph.cycle_edges``): add 1 to each
        edge's total count, and 1 to its bad count if the chord's score exceeds
        ``threshold``.
      * Return bad / max(total, 1).

    Why several trees? With one tree, edges that always share cycles with the
    corrupted edge tie with it. Different trees break those ties.
    Note: an edge on no cycle (a bridge) can never be located. Say so in your report.
    """
    generator = torch.Generator().manual_seed(seed)
    total = [0] * graph.n_edges
    bad = [0] * graph.n_edges

    for _ in range(n_trees):
        root = torch.randint(graph.n_vertices, (1,), generator=generator).item()
        weights = torch.rand(graph.n_edges, generator=generator)
        tree = graph.spanning_tree(root=root, weights=weights)
        scores = holonomy_scores(graph, tree)

        for chord, score in zip(tree.chords, scores.tolist()):
            cycle = graph.cycle_edges(tree, chord)
            is_bad = score > threshold
            for edge in cycle:
                total[edge] += 1
                bad[edge] += int(is_bad)

    values = [bad_count / max(total_count, 1) for bad_count, total_count in zip(bad, total)]
    return torch.tensor(values, dtype=graph.rotations.dtype, device=graph.rotations.device)
