"""Projection onto ker(delta^0): the nearest fully consistent state (proposal Section 4.3).

Exercises (week 4, Tue Dec 1). Spec: tests/test_projection.py.
"""
from __future__ import annotations

import torch  # noqa: F401
from torch import Tensor

from .defects import holonomies  # noqa: F401
from .graph import RotationGraph, SpanningTree
from .rotations import block_apply, block_transpose  # noqa: F401


def common_fixed_space(graph: RotationGraph, tree: SpanningTree, tol: float = 1e-6) -> list[Tensor]:
    """Orthonormal basis of the vectors fixed by EVERY holonomy, one basis per block.

    Returns a list of n_blocks tensors; block j's basis has shape (b, k_j), where
    k_j may be 0 (the usual case for random rotations: the kernel is trivial).

    Hints:
      * Holonomies are block-diagonal, so the fixed space splits block by block.
      * x is fixed by all H_c  <=>  sum_c |(H_c - I) x|^2 = 0  <=>  x^T M x = 0
        with M = sum_c (H_c - I)^T (H_c - I), which is symmetric PSD.
      * So: build M per block, ``torch.linalg.eigh``, keep eigenvectors whose
        eigenvalue is at most tol * max(1, n_chords).
      * A tree with no chords fixes everything.
    """
    H = holonomies(graph, tree)
    identity = torch.eye(graph.block_size, dtype=H.dtype, device=H.device)
    bases = []

    for block in range(graph.n_blocks):
        if H.shape[0] == 0:
            bases.append(identity.clone())
            continue

        difference = H[:, block] - identity
        gram = difference.transpose(-1, -2) @ difference
        matrix = gram.sum(dim=0)
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        cutoff = tol * max(1, len(tree.chords))
        bases.append(eigenvectors[:, eigenvalues <= cutoff])

    return bases


def project_to_kernel(graph: RotationGraph, tree: SpanningTree, psi: Tensor, tol: float = 1e-6) -> Tensor:
    """Orthogonal projection of arbitrary states psi (n_vertices, d) onto ker(delta^0).

    Every consistent state has the form psi_v = T_v x with x in the common
    fixed space F. The map x -> (T_v x)_v scaled by 1/sqrt(|V|) is an isometry
    (each T_v is orthogonal), so the nearest consistent state has root value

        x* = Pi_F( (1/|V|) * sum_v T_v^T psi_v )

    and then psi*_v = T_v x*. Derive this yourself before coding it: expand
    |psi - T x|^2 and complete the square.
    """
    transports = graph.transports(tree)
    states = psi.reshape(graph.n_vertices, graph.n_blocks, graph.block_size)
    root_coordinates = (
        block_transpose(transports) @ states.unsqueeze(-1)
    ).squeeze(-1).mean(dim=0)

    bases = common_fixed_space(graph, tree, tol=tol)
    projected = []
    for block, basis in enumerate(bases):
        coordinates = root_coordinates[block]
        if basis.shape[1] == 0:
            projected.append(torch.zeros_like(coordinates))
        else:
            projected.append(basis @ (basis.transpose(-1, -2) @ coordinates))

    root_state = torch.stack(projected)
    return graph.propagate(tree, root_state.reshape(-1))
