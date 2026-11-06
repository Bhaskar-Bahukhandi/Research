"""Localisation with noisy rotations (what Phase 2's learned rotations will look like).

Spec for ``localise_corrupted_edges(..., threshold=None)``, the cached
``RotationGraph.edge_index`` and ``synthetic.small_rotations``.
"""
import pytest
import torch

from rtvm.localise import localise_corrupted_edges
from rtvm.synthetic import consistent_graph, corrupt_edges, random_connected_graph, small_rotations

pytestmark = pytest.mark.week4


def noisy_single_corruption(seed: int, noise: float):
    edges = random_connected_graph(25, 70, seed=200 + seed)
    graph, _ = consistent_graph(25, edges, n_blocks=1, b=8, seed=seed)
    bad, (idx,) = corrupt_edges(graph, 1, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    rots = small_rotations(bad.n_edges, 1, 8, noise, generator=gen) @ bad.rotations
    return bad.with_rotations(rots), idx


def hits(threshold, noise):
    found = 0
    for seed in range(20):
        graph, idx = noisy_single_corruption(seed, noise)
        scores = localise_corrupted_edges(graph, n_trees=8, threshold=threshold, seed=seed)
        found += int(scores.argmax().item() == idx and (scores == scores.max()).sum() == 1)
    return found


def test_mean_mode_survives_noise():
    assert hits(None, 0.1) >= 16


def test_counting_mode_breaks_under_noise():
    """Why the mean mode exists: every noisy cycle clears 1e-3, so all edges tie."""
    assert hits(1e-3, 0.1) <= 2


def test_mean_mode_is_near_zero_on_consistent_graphs(consistent):
    graph, _ = consistent
    assert localise_corrupted_edges(graph, n_trees=3, threshold=None).max() < 1e-4


def test_small_rotations_are_rotations_of_the_requested_size():
    gen = torch.Generator().manual_seed(0)
    R = small_rotations(50, 2, 16, 0.01, generator=gen, dtype=torch.float64)
    eye = torch.eye(16, dtype=torch.float64)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand_as(R), atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(50, 2, dtype=torch.float64))
    size = (R - eye).flatten(1).norm(dim=1)
    assert torch.allclose(size, torch.full_like(size, 0.01), rtol=1e-2)


def test_edge_index_is_cached_and_correct(consistent):
    graph, _ = consistent
    first = graph.edge_index
    assert first is graph.edge_index
    assert first.dtype == torch.long and first.shape == (graph.n_edges, 2)
    assert [tuple(e) for e in first.tolist()] == graph.edges
