"""Week 4 (extra): which edges can be told apart at all. Spec for localise.cut_pair_classes."""
from itertools import combinations

import pytest
import torch

from rtvm.graph import RotationGraph
from rtvm.localise import bridges, cut_pair_classes
from rtvm.synthetic import random_connected_graph

pytestmark = pytest.mark.week4


def make(n_vertices, edges):
    """Rotations do not matter for this topic; identities keep it simple."""
    return RotationGraph(n_vertices, edges, torch.eye(2).expand(len(edges), 1, 2, 2).clone())


def groups(labels):
    """The partition as a set of frozensets, ignoring the label values."""
    out = {}
    for e, label in enumerate(labels):
        if label >= 0:
            out.setdefault(label, set()).add(e)
    return {frozenset(g) for g in out.values()}


def connected(n_vertices, edges, removed):
    adj = [[] for _ in range(n_vertices)]
    for i, (u, v) in enumerate(edges):
        if i not in removed:
            adj[u].append(v)
            adj[v].append(u)
    seen, stack = {0}, [0]
    while stack:
        for y in adj[stack.pop()]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == n_vertices


def test_bridges_helper():
    # two triangles joined by the bridge (2, 3)
    g = make(6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    assert bridges(g) == {3}


def test_labels_shape_and_numbering():
    g = make(6, [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    labels = cut_pair_classes(g)
    assert labels == [0, 0, 0, -1, 1, 1, 1]


def test_tree_is_all_bridges():
    g = make(5, [(0, 1), (1, 2), (1, 3), (3, 4)])
    assert cut_pair_classes(g) == [-1, -1, -1, -1]


def test_single_cycle_is_one_group():
    g = make(6, [(i, (i + 1) % 6) for i in range(6)])
    assert cut_pair_classes(g) == [0] * 6


def test_complete_graph_has_only_singletons():
    g = make(5, list(combinations(range(5), 2)))
    assert len(groups(cut_pair_classes(g))) == 10


def test_degree_two_chain_is_one_group():
    """K4 on {0,1,2,3} plus the chain 0 - 4 - 5 - 1: the three chain edges tie."""
    k4 = list(combinations(range(4), 2))
    chain = [(0, 4), (5, 4), (5, 1)]  # mixed orientations on purpose
    labels = cut_pair_classes(make(6, k4 + chain))
    assert groups(labels) == {frozenset({i}) for i in range(6)} | {frozenset({6, 7, 8})}


def test_parallel_edges_can_be_a_cut_pair():
    """Triangle, then two parallel edges 2 = 3, then triangle: the parallel pair is the only link."""
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 2), (3, 4), (4, 5), (5, 3)]
    assert groups(cut_pair_classes(make(6, edges))) == {
        frozenset({0, 1, 2}), frozenset({3, 4}), frozenset({5, 6, 7})}


def test_self_loop_is_its_own_group():
    edges = [(0, 1), (1, 2), (2, 0), (1, 1)]
    labels = cut_pair_classes(make(3, edges))
    assert labels[3] >= 0 and labels.count(labels[3]) == 1


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_on_sparse_random_graphs(seed):
    n, m = 18, 22  # sparse, so cut pairs are common
    edges = random_connected_graph(n, m, seed=seed)
    labels = cut_pair_classes(make(n, edges))
    for e in range(m):
        assert (labels[e] == -1) == (not connected(n, edges, {e}))
    for e, f in combinations(range(m), 2):
        if labels[e] >= 0 and labels[f] >= 0:
            assert (labels[e] == labels[f]) == (not connected(n, edges, {e, f})), (e, f)


def test_orientation_does_not_matter():
    edges = random_connected_graph(15, 19, seed=7)
    flipped = [(v, u) if i % 2 else (u, v) for i, (u, v) in enumerate(edges)]
    assert cut_pair_classes(make(15, edges)) == cut_pair_classes(make(15, flipped))
