"""Week 3, Thu-Fri: defects. Spec for rtvm/defects.py. This is gate G1's core."""
import pytest
import torch

from rtvm.defects import chord_defects, edge_residuals, holonomies, holonomy_scores
from rtvm.synthetic import consistent_graph, corrupt_edges, random_connected_graph

pytestmark = pytest.mark.week3


def test_edge_residuals_zero_for_consistent_state(consistent, psi_root):
    graph, frames = consistent
    c = frames[0].transpose(-1, -2) @ psi_root.reshape(2, 8, 1)
    psi = (frames @ c).reshape(graph.n_vertices, graph.dim)
    r = edge_residuals(graph, psi)
    assert r.shape == (graph.n_edges,)
    assert r.max() < 1e-5


def test_one_defect_per_chord(consistent, psi_root):
    graph, _ = consistent
    tree = graph.spanning_tree(0)
    assert chord_defects(graph, tree, psi_root).shape == (graph.betti_1,)
    assert holonomies(graph, tree).shape == (graph.betti_1, 2, 8, 8)
    assert holonomy_scores(graph, tree).shape == (graph.betti_1,)


@pytest.mark.parametrize("seed", range(10))
def test_consistent_graphs_have_zero_defect(seed):
    edges = random_connected_graph(30, 70, seed=seed)
    graph, _ = consistent_graph(30, edges, n_blocks=2, b=64, seed=seed)
    tree = graph.spanning_tree(0)
    psi_root = torch.randn(128, generator=torch.Generator().manual_seed(seed))
    assert chord_defects(graph, tree, psi_root).max() < 1e-5 * psi_root.norm()
    assert holonomy_scores(graph, tree).max() < 1e-4


def test_vector_defect_equals_holonomy_form(consistent, psi_root):
    """|U_e psi_u - psi_v| = |(H_c - I) psi_r|, since transports are orthogonal."""
    graph, _ = consistent
    bad, _ = corrupt_edges(graph, 4, seed=1)
    tree = bad.spanning_tree(0)
    H = holonomies(bad, tree)
    x = psi_root.reshape(2, 8, 1)
    via_h = ((H @ x).reshape(len(tree.chords), -1) - psi_root).norm(dim=-1)
    assert torch.allclose(chord_defects(bad, tree, psi_root), via_h, atol=1e-4)


@pytest.mark.parametrize("seed", range(10))
def test_corruption_is_seen_exactly_on_cycles_through_it(seed):
    edges = random_connected_graph(25, 60, seed=100 + seed)
    graph, _ = consistent_graph(25, edges, n_blocks=2, b=8, seed=seed)
    bad, idx = corrupt_edges(graph, 2, seed=seed)
    tree = bad.spanning_tree(0)
    scores = holonomy_scores(bad, tree)
    for c, s in zip(tree.chords, scores.tolist()):
        if bad.cycle_edges(tree, c) & set(idx):
            assert s > 1e-2
        else:
            assert s < 1e-4
