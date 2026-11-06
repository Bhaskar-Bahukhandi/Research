"""Regression tests for the Nov 6 2026 engine revision.

Covers: level-batched vector propagation, loud failure on malformed trees,
self-loops, and gradients through the new code path.
"""
import pytest
import torch

from rtvm.defects import chord_defects, holonomy_scores
from rtvm.graph import RotationGraph, SpanningTree
from rtvm.rotations import block_apply, block_transpose
from rtvm.synthetic import corrupt_edges, random_rotation

pytestmark = pytest.mark.week3


def naive_transports(graph: RotationGraph, tree: SpanningTree) -> torch.Tensor:
    """One vertex at a time, straight from the definition. Slow but obviously right."""
    T = [None] * graph.n_vertices
    T[tree.root] = torch.eye(graph.block_size, dtype=graph.rotations.dtype).expand(
        graph.n_blocks, -1, -1)
    for v in tree.order[1:]:
        p, e = tree.parent[v], tree.parent_edge[v]
        U = graph.rotations[e]
        T[v] = (U if graph.edges[e] == (p, v) else block_transpose(U)) @ T[p]
    return torch.stack(T)


@pytest.mark.parametrize("root", [0, 4, 19])
def test_batched_transports_match_naive(consistent, root):
    bad, _ = corrupt_edges(consistent[0], 6, seed=1)
    tree = bad.spanning_tree(root)
    assert torch.allclose(bad.transports(tree), naive_transports(bad, tree), atol=1e-5)


@pytest.mark.parametrize("root", [0, 7])
def test_vector_propagation_matches_matrix_route(consistent, psi_root, root):
    bad, _ = corrupt_edges(consistent[0], 6, seed=4)
    tree = bad.spanning_tree(root)
    expected = block_apply(bad.transports(tree), psi_root)
    assert torch.allclose(bad.propagate(tree, psi_root), expected, atol=1e-5)


def test_tree_from_random_weights_also_works(consistent, psi_root):
    graph, _ = consistent
    tree = graph.spanning_tree(3, weights=torch.rand(graph.n_edges, generator=torch.Generator().manual_seed(0)))
    expected = block_apply(naive_transports(graph, tree), psi_root)
    assert torch.allclose(graph.propagate(tree, psi_root), expected, atol=1e-5)


def test_incomplete_tree_order_raises(consistent):
    graph, _ = consistent
    tree = graph.spanning_tree(0)
    broken = SpanningTree(tree.root, tree.order[:-1], tree.parent, tree.parent_edge, tree.chords)
    with pytest.raises(ValueError, match="exactly once"):
        graph.transports(broken)


def test_wrong_parent_edge_raises(consistent):
    graph, _ = consistent
    tree = graph.spanning_tree(0)
    v = tree.order[-1]
    parent_edge = list(tree.parent_edge)
    parent_edge[v] = next(e for e in range(graph.n_edges) if v not in graph.edges[e])
    broken = SpanningTree(tree.root, tree.order, tree.parent, parent_edge, tree.chords)
    with pytest.raises(ValueError, match="does not join"):
        graph.transports(broken)


def test_wrong_psi_shape_raises(consistent, psi_root):
    graph, _ = consistent
    with pytest.raises(ValueError, match="psi_root"):
        graph.propagate(graph.spanning_tree(0), psi_root[:-1])


def test_self_loop_is_a_chord_and_is_checked():
    gen = torch.Generator().manual_seed(5)
    R = random_rotation(1, 4, generator=gen, dtype=torch.float64)
    eye = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    edges = [(0, 1), (1, 1)]
    psi_root = torch.randn(4, generator=gen, dtype=torch.float64)

    clean = RotationGraph(2, edges, torch.stack([R, eye]))
    tree = clean.spanning_tree(0)
    assert tree.chords == [1]
    assert clean.cycle_edges(tree, 1) == {1}
    assert clean.betti_1 == 1
    assert chord_defects(clean, tree, psi_root).abs().max() < 1e-12

    dirty = clean.with_rotations(torch.stack([R, R]))
    assert holonomy_scores(dirty, dirty.spanning_tree(0))[0] > 1e-2


def test_gradients_flow_through_vector_propagation(consistent, psi_root):
    graph, _ = consistent
    graph = graph.with_rotations(graph.rotations.double().requires_grad_())
    tree = graph.spanning_tree(2)
    psi = psi_root.double().requires_grad_()

    def f(rotations, x):
        return graph.with_rotations(rotations).propagate(tree, x)

    assert torch.autograd.gradcheck(f, (graph.rotations, psi))
