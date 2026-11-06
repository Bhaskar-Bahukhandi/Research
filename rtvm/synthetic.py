"""Synthetic graphs for testing the consistency engine.

Everything in this module is provided complete: it is test infrastructure, not
part of the Phase 1 exercises. It deliberately avoids ``rotations.cayley`` so
the tests can run before you have implemented it.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .graph import RotationGraph


def random_rotation(n_blocks: int, b: int, *, generator: torch.Generator | None = None,
                    dtype: torch.dtype = torch.float32) -> Tensor:
    """Random block rotations of shape (n_blocks, b, b), Haar-distributed on SO(b).

    Uses QR of a Gaussian matrix with the standard sign fix, then flips one
    column where needed so every block has determinant +1.
    """
    G = torch.randn(n_blocks, b, b, generator=generator, dtype=torch.float64)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2)
    det_sign = torch.linalg.det(Q).sign()
    Q[..., 0] = Q[..., 0] * det_sign.unsqueeze(-1)
    return Q.to(dtype)


def random_connected_graph(n_vertices: int, n_edges: int, seed: int = 0) -> list[tuple[int, int]]:
    """A random connected simple graph with randomly oriented edges.

    Builds a random spanning tree first, then adds distinct extra edges, so the
    graph has exactly ``n_edges`` edges and first Betti number
    ``n_edges - n_vertices + 1``.
    """
    max_edges = n_vertices * (n_vertices - 1) // 2
    if not (n_vertices - 1 <= n_edges <= max_edges):
        raise ValueError(f"n_edges must lie in [{n_vertices - 1}, {max_edges}]")
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_vertices, generator=gen).tolist()
    pairs: set[tuple[int, int]] = set()
    for i in range(1, n_vertices):
        j = perm[int(torch.randint(i, (1,), generator=gen))]
        pairs.add(tuple(sorted((perm[i], j))))
    while len(pairs) < n_edges:
        u, v = torch.randint(n_vertices, (2,), generator=gen).tolist()
        if u != v:
            pairs.add(tuple(sorted((u, v))))
    edges = []
    for u, v in sorted(pairs):
        flip = bool(torch.randint(2, (1,), generator=gen))
        edges.append((v, u) if flip else (u, v))
    return edges


def consistent_graph(n_vertices: int, edges: list[tuple[int, int]], n_blocks: int, b: int,
                     seed: int = 0, dtype: torch.dtype = torch.float32) -> tuple[RotationGraph, Tensor]:
    """A graph whose rotations agree around every cycle.

    Each vertex gets a random frame F_v; each edge (u, v) gets U_e = F_v F_u^T.
    Then psi_v = F_v c is consistent for any c, and every holonomy is the identity.
    Returns the graph and the frames, shape (n_vertices, n_blocks, b, b).

    Memory: frames and rotations are stored once, directly in ``dtype``, so the
    peak is about (n_vertices + n_edges) * n_blocks * b^2 * itemsize. (The first
    version kept float64 copies of both plus a Python list of edge rotations,
    roughly 4x more, and ran out of memory at d = 1024.) Each product is still
    computed in float64 and rounded once, so clean defects stay near 1e-6.
    Dense storage is the real limit at full width: 16 blocks of 64 x 64 in
    float32 are 256 KiB per edge, so 40k edges already need 10 GiB.
    """
    gen = torch.Generator().manual_seed(seed)
    frames = torch.empty(n_vertices, n_blocks, b, b, dtype=dtype)
    for v in range(n_vertices):
        frames[v] = random_rotation(n_blocks, b, generator=gen, dtype=dtype)
    rots = torch.empty(len(edges), n_blocks, b, b, dtype=dtype)
    for i, (u, v) in enumerate(edges):
        rots[i] = (frames[v].double() @ frames[u].double().transpose(-1, -2)).to(dtype)
    return RotationGraph(n_vertices, edges, rots), frames


def corrupt_edges(graph: RotationGraph, k: int, seed: int = 0) -> tuple[RotationGraph, list[int]]:
    """Replace the rotations on k randomly chosen edges with fresh random ones.

    Returns the corrupted graph and the sorted indices of the corrupted edges.
    A corrupted edge that lies on no cycle (a bridge) cannot be detected by any
    cycle test; the tests account for this.
    """
    gen = torch.Generator().manual_seed(seed)
    idx = sorted(torch.randperm(graph.n_edges, generator=gen)[:k].tolist())
    rots = graph.rotations.clone()
    for i in idx:
        rots[i] = random_rotation(graph.n_blocks, graph.block_size, generator=gen, dtype=rots.dtype)
    return graph.with_rotations(rots), idx


def partially_consistent_graph(n_vertices: int, edges: list[tuple[int, int]], n_blocks: int, b: int,
                               k_fixed: int, seed: int = 0,
                               dtype: torch.dtype = torch.float32) -> RotationGraph:
    """Rotations that fix the first ``k_fixed`` coordinates of every block.

    Each edge gets U_e = diag(I_k, Q_e) per block with Q_e random. For generic
    Q_e the holonomies share exactly a k_fixed-dimensional fixed space per
    block, which makes ker(delta^0) non-trivial but smaller than the whole space.
    """
    gen = torch.Generator().manual_seed(seed)
    rots = torch.zeros(len(edges), n_blocks, b, b, dtype=dtype)
    for i in range(len(edges)):
        rots[i, :, :k_fixed, :k_fixed] = torch.eye(k_fixed, dtype=dtype)
        rots[i, :, k_fixed:, k_fixed:] = random_rotation(n_blocks, b - k_fixed, generator=gen, dtype=dtype)
    return RotationGraph(n_vertices, edges, rots)
