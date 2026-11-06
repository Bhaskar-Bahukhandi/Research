"""Week 4, Mon: localisation. Spec for rtvm/localise.py."""
import pytest
import torch

from rtvm.localise import localise_corrupted_edges
from rtvm.synthetic import consistent_graph, corrupt_edges, random_connected_graph

pytestmark = pytest.mark.week4


def test_scores_shape_and_range(consistent):
    graph, _ = consistent
    bad, _ = corrupt_edges(graph, 1, seed=0)
    s = localise_corrupted_edges(bad, n_trees=3, seed=0)
    assert s.shape == (graph.n_edges,)
    assert s.min() >= 0 and s.max() <= 1


def test_consistent_graph_scores_zero(consistent):
    graph, _ = consistent
    assert localise_corrupted_edges(graph, n_trees=3).max() == 0


def test_single_corruption_is_found_most_of_the_time():
    """Pass threshold for gate G1: the corrupted edge ranks first in >= 80% of graphs."""
    hits = 0
    for seed in range(20):
        edges = random_connected_graph(25, 70, seed=200 + seed)
        graph, _ = consistent_graph(25, edges, n_blocks=1, b=8, seed=seed)
        bad, (idx,) = corrupt_edges(graph, 1, seed=seed)
        scores = localise_corrupted_edges(bad, n_trees=8, seed=seed)
        hits += int(scores.argmax().item() == idx and (scores == scores.max()).sum() == 1)
    assert hits >= 16, f"found {hits}/20"
