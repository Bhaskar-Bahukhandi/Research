"""Week 4, Tue: projection onto ker(delta^0). Spec for rtvm/projection.py."""
import pytest
import torch

from rtvm.defects import edge_residuals
from rtvm.projection import common_fixed_space, project_to_kernel
from rtvm.synthetic import consistent_graph, partially_consistent_graph, random_connected_graph

pytestmark = pytest.mark.week4


def _dense_coboundary(graph):
    """delta^0 as a dense (E*d, V*d) matrix: block row e has +U_e at u and -I at v."""
    d, V, E = graph.dim, graph.n_vertices, graph.n_edges
    D = torch.zeros(E * d, V * d, dtype=graph.rotations.dtype)
    for e, (u, v) in enumerate(graph.edges):
        D[e * d:(e + 1) * d, u * d:(u + 1) * d] = torch.block_diag(*graph.rotations[e])
        D[e * d:(e + 1) * d, v * d:(v + 1) * d] = -torch.eye(d, dtype=D.dtype)
    return D


def test_consistent_graph_fixes_everything(consistent):
    graph, _ = consistent
    bases = common_fixed_space(graph, graph.spanning_tree(0))
    assert [B.shape[1] for B in bases] == [8, 8]


@pytest.mark.parametrize("k", [0, 2, 5])
def test_partially_consistent_fixed_dimension(small_edges, k):
    graph = partially_consistent_graph(20, small_edges, n_blocks=2, b=8, k_fixed=k, seed=k,
                                       dtype=torch.float64)
    bases = common_fixed_space(graph, graph.spanning_tree(0))
    assert [B.shape[1] for B in bases] == [k, k]


def test_projection_matches_dense_least_squares():
    edges = random_connected_graph(8, 14, seed=0)
    graph = partially_consistent_graph(8, edges, n_blocks=2, b=4, k_fixed=2, seed=0, dtype=torch.float64)
    psi = torch.randn(8, 8, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    D = _dense_coboundary(graph)
    _, S, Vh = torch.linalg.svd(D)
    null = Vh[(S > 1e-9).sum():].T
    expected = (null @ (null.T @ psi.reshape(-1))).reshape(8, 8)
    got = project_to_kernel(graph, graph.spanning_tree(3), psi)
    assert torch.allclose(got, expected, atol=1e-8)


def test_projection_is_idempotent_and_consistent(small_edges):
    graph = partially_consistent_graph(20, small_edges, n_blocks=2, b=8, k_fixed=3, seed=1,
                                       dtype=torch.float64)
    tree = graph.spanning_tree(0)
    psi = torch.randn(20, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    p1 = project_to_kernel(graph, tree, psi)
    assert edge_residuals(graph, p1).max() < 1e-8
    assert torch.allclose(project_to_kernel(graph, tree, p1), p1, atol=1e-8)


def test_consistent_state_is_unchanged():
    edges = random_connected_graph(15, 30, seed=2)
    graph, frames = consistent_graph(15, edges, n_blocks=2, b=8, seed=2, dtype=torch.float64)
    c = torch.randn(2, 8, 1, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    psi = (frames @ c).reshape(15, 16)
    assert torch.allclose(project_to_kernel(graph, graph.spanning_tree(0), psi), psi, atol=1e-8)
