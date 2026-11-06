"""Coboundary residuals and cycle holonomy defects (proposal Section 4.3).

Exercises (week 3, Thu Nov 26 - Fri Nov 27). Spec: tests/test_defects.py.
These four functions are what gate G1 tests.
"""
from __future__ import annotations

import torch  # noqa: F401
from torch import Tensor

from .graph import RotationGraph, SpanningTree
from .rotations import block_apply, block_transpose  # noqa: F401  (you will need these)


def edge_residuals(graph: RotationGraph, psi: Tensor) -> Tensor:
    """The coboundary delta^0 applied to psi, as one norm per edge: |U_e psi_u - psi_v|.

    Args:
        psi: states of shape (n_vertices, d), arbitrary (not necessarily propagated).
    Returns:
        shape (n_edges,)

    Hint: gather psi at all edge sources and targets with index tensors, then a
    single ``block_apply`` over all edges. No Python loop over edges needed.
    """
    edge_index = torch.tensor(graph.edges, dtype=torch.long, device=psi.device).reshape(-1, 2)
    sources = edge_index[:, 0]
    targets = edge_index[:, 1]
    predicted = block_apply(graph.rotations, psi[sources])
    return (predicted - psi[targets]).norm(dim=-1)


def chord_defects(graph: RotationGraph, tree: SpanningTree, psi_root: Tensor) -> Tensor:
    """Vector-form defect on every chord, shape (n_chords,), in ``tree.chords`` order.

    Propagate psi_root along the tree, then for each chord e_c = (u, v) return
    |U_e psi_u - psi_v|. Tree edges are zero by construction, so only chords
    can carry a defect. Cost: linear in |V| + |E|.
    """
    states = graph.propagate(tree, psi_root)
    chord_index = torch.tensor(tree.chords, dtype=torch.long, device=graph.rotations.device)
    edge_index = torch.tensor(graph.edges, dtype=torch.long, device=graph.rotations.device).reshape(-1, 2)
    chord_edges = edge_index[chord_index]
    sources, targets = chord_edges.unbind(dim=-1)
    predicted = block_apply(graph.rotations[chord_index], states[sources])
    return (predicted - states[targets]).norm(dim=-1)


def holonomies(graph: RotationGraph, tree: SpanningTree) -> Tensor:
    """Holonomy H_c = T_v^T U_e T_u around each chord's cycle, shape (n_chords, n_blocks, b, b).

    H_c is the identity exactly when the cycle is consistent. Note that
    |(H_c - I) psi_r| equals the chord's vector defect (why? transports are orthogonal).
    """
    if not tree.chords:
        return graph.rotations.new_empty((0, graph.n_blocks, graph.block_size, graph.block_size))

    transports = graph.transports(tree)
    chord_index = torch.tensor(tree.chords, dtype=torch.long, device=graph.rotations.device)
    edge_index = torch.tensor(graph.edges, dtype=torch.long, device=graph.rotations.device)
    chord_edges = edge_index[chord_index]
    sources, targets = chord_edges.unbind(dim=-1)
    return (
        block_transpose(transports[targets])
        @ graph.rotations[chord_index]
        @ transports[sources]
    )


def holonomy_scores(graph: RotationGraph, tree: SpanningTree) -> Tensor:
    """psi-free score |H_c - I|_F per chord, shape (n_chords,).

    The Frobenius norm over the whole block-diagonal matrix: square the
    per-block norms, sum over blocks, take the square root.
    """
    H = holonomies(graph, tree)
    eye = torch.eye(graph.block_size, dtype=H.dtype, device=H.device)
    difference = H - eye
    return difference.flatten(start_dim=-3).norm(dim=-1)
