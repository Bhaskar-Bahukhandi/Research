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


# ---------------------------------------------------------------------------
# What CAN be located (provided helper + week 4 exercise)
# ---------------------------------------------------------------------------

def bridges(graph: RotationGraph) -> set[int]:
    """Indices of edges that lie on no cycle. No cycle test can ever check them.

    Uses one spanning tree: an edge lies on SOME cycle if and only if it lies
    on at least one fundamental cycle, because every cycle is a sum (mod 2)
    of fundamental cycles.
    """
    tree = graph.spanning_tree(0)
    on_a_cycle: set[int] = set()
    for chord in tree.chords:
        on_a_cycle |= graph.cycle_edges(tree, chord)
    return set(range(graph.n_edges)) - on_a_cycle


def cut_pair_classes(graph: RotationGraph) -> list[int]:
    """Group the edges that no cycle test can tell apart. Exercise (week 4).

    Two edges are indistinguishable when they lie on exactly the same cycles,
    which happens exactly when removing both disconnects the graph (a "cut
    pair"). Every cycle test then sees them together, so localisation can only
    ever name the group. (At a degree-2 vertex this is your gauge argument:
    re-choosing w's frame moves a corruption from one edge to the other.)

    Returns one label per edge:
      * -1 for a bridge (on no cycle at all; see ``bridges``)
      * 0, 1, 2, ... for the other edges, numbered in order of first
        appearance by edge index, so edge groups get labels 0, 1, ... as you
        scan edges 0, 1, 2, ...
    Edges e != f get the same label exactly when {e, f} is a cut pair.

    Two ways to do it (try the second; it is the one worth understanding):
      * Brute force: for every pair, delete both and test connectivity.
        O(|E|^2 (|V| + |E|)). Fine for tests, hopeless for a knowledge graph.
      * Signatures: with ONE spanning tree, give each edge the set of
        fundamental cycles (chords) that contain it, then group equal sets.
        Prove first why one tree is enough: if e and f lie on the same
        fundamental cycles, why must they lie on the same cycles of every
        kind? (Hint: an edge lies in a mod-2 sum of cycles exactly when it
        lies in an odd number of them.)
    Spec: tests/test_cut_pairs.py.
    """
    raise NotImplementedError("cut_pair_classes")
