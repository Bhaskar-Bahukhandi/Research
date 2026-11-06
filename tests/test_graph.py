"""Week 3, Wed: tree transport. Spec for RotationGraph.transports and .propagate."""
import pytest
import torch

from rtvm.graph import RotationGraph
from rtvm.synthetic import corrupt_edges, random_rotation

pytestmark = pytest.mark.week3


def test_transport_by_hand_with_a_reversed_edge():
    """Path 0 -> 1 <- 2. Crossing edge (2, 1) from 1 to 2 must use the transpose."""
    gen = torch.Generator().manual_seed(0)
    R01 = random_rotation(1, 4, generator=gen, dtype=torch.float64)
    R21 = random_rotation(1, 4, generator=gen, dtype=torch.float64)
    g = RotationGraph(3, [(0, 1), (2, 1)], torch.stack([R01, R21]))
    tree = g.spanning_tree(root=0)
    T = g.transports(tree)
    assert T.shape == (3, 1, 4, 4)
    assert torch.allclose(T[0], torch.eye(4, dtype=torch.float64).unsqueeze(0))
    assert torch.allclose(T[1], R01)
    assert torch.allclose(T[2], R21.transpose(-1, -2) @ R01)


def test_transports_are_orthogonal(consistent):
    graph, _ = consistent
    T = graph.transports(graph.spanning_tree(0))
    eye = torch.eye(graph.block_size)
    assert torch.allclose(T @ T.transpose(-1, -2), eye.expand_as(T), atol=1e-5)


def test_propagate_matches_frames_on_consistent_graph(consistent, psi_root):
    graph, frames = consistent
    tree = graph.spanning_tree(root=4)
    psi = graph.propagate(tree, psi_root)
    assert psi.shape == (graph.n_vertices, graph.dim)
    # psi_v = F_v F_r^T psi_r, block by block
    c = (frames[4].transpose(-1, -2) @ psi_root.reshape(2, 8, 1))
    expected = (frames @ c).reshape(graph.n_vertices, graph.dim)
    assert torch.allclose(psi, expected, atol=1e-5)


def test_tree_edges_always_agree_even_when_corrupted(consistent, psi_root):
    graph, _ = consistent
    bad, _ = corrupt_edges(graph, 5, seed=0)
    tree = bad.spanning_tree(0)
    psi = bad.propagate(tree, psi_root)
    for v in tree.order[1:]:
        e = tree.parent_edge[v]
        u, w = bad.edges[e]
        r = (bad.rotations[e] @ psi[u].reshape(2, 8, 1)).reshape(-1) - psi[w]
        assert r.norm() < 1e-4
