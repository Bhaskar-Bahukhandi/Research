"""Graphs whose edges carry block-diagonal rotations.

Convention: edge e = (u, v) carries U_e, which maps u's frame into v's frame.
A state psi is consistent on e when U_e psi_u = psi_v.

Provided (complete): the RotationGraph container, spanning trees (BFS, or
minimum spanning trees under random weights for localisation), tree paths and
cycle edges.

Exercises (week 3, Wed Nov 25): ``RotationGraph.transports`` and
``RotationGraph.propagate``. Spec: tests/test_graph.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from .rotations import block_apply, block_transpose, identity_blocks  # noqa: F401  (you will need these)


@dataclass
class SpanningTree:
    """A rooted spanning tree and its cotree.

    root:        root vertex r
    order:       vertices in BFS order, root first (parents always come before children)
    parent:      parent vertex of each vertex, -1 for the root
    parent_edge: index of the edge joining each vertex to its parent, -1 for the root
    chords:      indices of edges NOT in the tree; there are |E| - |V| + 1 of them
    """

    root: int
    order: list[int]
    parent: list[int]
    parent_edge: list[int]
    chords: list[int]


class RotationGraph:
    """Directed multigraph with one block rotation per edge.

    Args:
        n_vertices: number of vertices, labelled 0 .. n_vertices - 1
        edges: list of (u, v) pairs; orientation matters
        rotations: tensor of shape (n_edges, n_blocks, b, b)
    """

    def __init__(self, n_vertices: int, edges: list[tuple[int, int]], rotations: Tensor):
        if rotations.dim() != 4 or rotations.shape[0] != len(edges):
            raise ValueError("rotations must have shape (n_edges, n_blocks, b, b)")
        if rotations.shape[-1] != rotations.shape[-2]:
            raise ValueError("rotation blocks must be square")
        for u, v in edges:
            if not (0 <= u < n_vertices and 0 <= v < n_vertices) or u == v:
                raise ValueError(f"invalid edge {(u, v)}")
        self.n_vertices = n_vertices
        self.edges = [tuple(e) for e in edges]
        self.rotations = rotations
        self._adj: list[list[tuple[int, int]]] = [[] for _ in range(n_vertices)]
        for i, (u, v) in enumerate(self.edges):
            self._adj[u].append((v, i))
            self._adj[v].append((u, i))

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_blocks(self) -> int:
        return self.rotations.shape[1]

    @property
    def block_size(self) -> int:
        return self.rotations.shape[-1]

    @property
    def dim(self) -> int:
        return self.n_blocks * self.block_size

    @property
    def betti_1(self) -> int:
        """First Betti number |E| - |V| + 1 (for a connected graph): the number of chords."""
        return self.n_edges - self.n_vertices + 1

    def with_rotations(self, rotations: Tensor) -> "RotationGraph":
        """Same graph, different rotations."""
        return RotationGraph(self.n_vertices, list(self.edges), rotations)

    def spanning_tree(self, root: int = 0, weights: Tensor | None = None) -> SpanningTree:
        """BFS spanning tree from ``root``.

        With ``weights`` (one per edge), the tree is the minimum spanning tree
        under those weights; random weights give different trees, which the
        week 4 localisation exercise uses.
        """
        allowed = None if weights is None else self._kruskal(weights)
        parent = [-1] * self.n_vertices
        parent_edge = [-1] * self.n_vertices
        seen = [False] * self.n_vertices
        seen[root] = True
        order = [root]
        queue = deque([root])
        while queue:
            x = queue.popleft()
            for y, ei in sorted(self._adj[x]):
                if seen[y] or (allowed is not None and ei not in allowed):
                    continue
                seen[y] = True
                parent[y] = x
                parent_edge[y] = ei
                order.append(y)
                queue.append(y)
        if not all(seen):
            raise ValueError("graph is not connected")
        tree_edges = set(parent_edge) - {-1}
        chords = [i for i in range(self.n_edges) if i not in tree_edges]
        return SpanningTree(root, order, parent, parent_edge, chords)

    def _kruskal(self, weights: Tensor) -> set[int]:
        parent = list(range(self.n_vertices))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        chosen: set[int] = set()
        for ei in torch.argsort(weights).tolist():
            u, v = self.edges[ei]
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                chosen.add(ei)
        return chosen

    def path_to_root(self, tree: SpanningTree, v: int) -> list[int]:
        """Edge indices on the tree path from v up to the root."""
        path = []
        while v != tree.root:
            path.append(tree.parent_edge[v])
            v = tree.parent[v]
        return path

    def cycle_edges(self, tree: SpanningTree, chord: int) -> set[int]:
        """Edge indices of the fundamental cycle closed by ``chord``."""
        u, v = self.edges[chord]
        return {chord} | (set(self.path_to_root(tree, u)) ^ set(self.path_to_root(tree, v)))

    # -----------------------------------------------------------------------
    # Exercises
    # -----------------------------------------------------------------------

    def transports(self, tree: SpanningTree) -> Tensor:
        """Transport T_v from the root to every vertex, shape (n_vertices, n_blocks, b, b).

        Defined so that psi_v = T_v psi_r makes every TREE edge consistent.

        Hints:
          * T_root is the identity (``identity_blocks``).
          * Visit vertices in ``tree.order`` so each parent is done before its child.
          * For child v with parent p and tree edge e = parent_edge[v]:
              - if edges[e] == (p, v): psi_v = U_e psi_p,   so T_v = U_e T_p
              - if edges[e] == (v, p): psi_p = U_e psi_v,   so T_v = U_e^T T_p
            Iteration 7 of the PDF forgot the second case.
          * Use the rotations' dtype and device.
        """
        identity = identity_blocks(
            self.n_blocks,
            self.block_size,
            dtype=self.rotations.dtype,
            device=self.rotations.device,
        )
        transports: list[Tensor | None] = [None] * self.n_vertices
        transports[tree.root] = identity

        for vertex in tree.order[1:]:
            parent = tree.parent[vertex]
            edge = tree.parent_edge[vertex]
            parent_transport = transports[parent]
            if parent_transport is None:
                raise ValueError("tree order must visit each parent before its children")

            source, target = self.edges[edge]
            rotation = self.rotations[edge]
            if (source, target) == (parent, vertex):
                transports[vertex] = rotation @ parent_transport
            else:
                transports[vertex] = block_transpose(rotation) @ parent_transport

        return torch.stack([transport for transport in transports if transport is not None])

    def propagate(self, tree: SpanningTree, psi_root: Tensor) -> Tensor:
        """States psi_v = T_v psi_root for all vertices, shape (n_vertices, d).

        Hint: ``block_apply`` broadcasts over the vertex dimension.
        """
        return block_apply(self.transports(tree), psi_root)
