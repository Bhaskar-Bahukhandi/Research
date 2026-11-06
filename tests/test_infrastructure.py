"""Tests for the provided infrastructure. These should pass from day one."""
import torch

from rtvm.graph import RotationGraph
from rtvm.synthetic import (consistent_graph, corrupt_edges, partially_consistent_graph,
                            random_connected_graph, random_rotation)
from rtvm.utils import count_parameters, seed_everything


def test_random_rotation_is_special_orthogonal():
    R = random_rotation(4, 16, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    eye = torch.eye(16, dtype=torch.float64)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand_as(R), atol=1e-10)
    assert torch.allclose(torch.linalg.det(R), torch.ones(4, dtype=torch.float64), atol=1e-10)


def test_random_connected_graph_shape():
    edges = random_connected_graph(30, 50, seed=0)
    assert len(edges) == 50
    assert len({tuple(sorted(e)) for e in edges}) == 50
    g = RotationGraph(30, edges, torch.zeros(50, 1, 2, 2))
    tree = g.spanning_tree(0)  # raises if not connected
    assert len(tree.chords) == g.betti_1 == 21


def test_consistent_graph_relation(consistent):
    graph, frames = consistent
    for i, (u, v) in enumerate(graph.edges):
        assert torch.allclose(graph.rotations[i] @ frames[u], frames[v], atol=1e-5)


def test_spanning_tree_structure(consistent):
    graph, _ = consistent
    tree = graph.spanning_tree(root=5)
    assert tree.order[0] == 5 and tree.parent[5] == -1
    assert sorted(tree.order) == list(range(graph.n_vertices))
    assert len(tree.chords) == graph.betti_1
    for v in tree.order[1:]:
        assert set(graph.edges[tree.parent_edge[v]]) == {v, tree.parent[v]}


def test_weighted_spanning_trees_differ(consistent):
    graph, _ = consistent
    gen = torch.Generator().manual_seed(0)
    chord_sets = {tuple(graph.spanning_tree(0, torch.rand(graph.n_edges, generator=gen)).chords)
                  for _ in range(5)}
    assert len(chord_sets) > 1


def test_cycle_edges_form_a_cycle(consistent):
    graph, _ = consistent
    tree = graph.spanning_tree(0)
    for c in tree.chords:
        degree: dict[int, int] = {}
        for e in graph.cycle_edges(tree, c):
            for x in graph.edges[e]:
                degree[x] = degree.get(x, 0) + 1
        assert all(d == 2 for d in degree.values())


def test_corrupt_edges_changes_only_chosen(consistent):
    graph, _ = consistent
    bad, idx = corrupt_edges(graph, 3, seed=0)
    assert len(idx) == 3
    changed = [i for i in range(graph.n_edges) if not torch.equal(bad.rotations[i], graph.rotations[i])]
    assert changed == idx


def test_partially_consistent_fixes_leading_coordinates(small_edges):
    g = partially_consistent_graph(20, small_edges, n_blocks=2, b=8, k_fixed=3, seed=0)
    assert torch.allclose(g.rotations[:, :, :3, :3], torch.eye(3).expand(g.n_edges, 2, 3, 3))


def test_seed_everything_is_reproducible():
    seed_everything(7)
    a = torch.randn(5)
    seed_everything(7)
    assert torch.equal(a, torch.randn(5))


def test_count_parameters_counts_tied_once():
    lin = torch.nn.Linear(3, 4)
    assert count_parameters(lin) == 16
    model = torch.nn.Sequential(lin, lin)
    assert count_parameters(model) == 16
