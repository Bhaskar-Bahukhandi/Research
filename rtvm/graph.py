"""Graphs whose edges carry block-diagonal rotations.

Convention: edge e = (u, v) carries U_e, which maps u's frame into v's frame.
A state psi is consistent on e when U_e psi_u = psi_v.

Provided (complete): the RotationGraph container, spanning trees (BFS, or
minimum spanning trees under random weights for localisation), tree paths and
cycle edges.

Exercises (week 3, Wed Nov 25): ``RotationGraph.transports`` and
``RotationGraph.propagate``. Spec: tests/test_graph.py. Revised Nov 6 2026 to
push vectors level by level (see the "Tree transport" section).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from .rotations import gather_apply, identity_blocks


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


@dataclass
class TreeLevel:
    """All vertices at one depth of a spanning tree, as index tensors.

    vertices:   (n,) vertex ids at this depth
    parent_pos: (n,) position of each vertex's parent inside the PREVIOUS level
    edges:      (n,) the tree edge joining each vertex to its parent
    n_forward:  the first n_forward edges are stored as (parent, child) and use
                U_e; the remaining ones are stored as (child, parent) and use U_e^T
    """

    vertices: Tensor
    parent_pos: Tensor
    edges: Tensor
    n_forward: int


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
            if not (0 <= u < n_vertices and 0 <= v < n_vertices):
                raise ValueError(f"invalid edge {(u, v)}")
        self.n_vertices = n_vertices
        self.edges = [tuple(e) for e in edges]
        self.rotations = rotations
        self._adj: list[list[tuple[int, int]]] = [[] for _ in range(n_vertices)]
        for i, (u, v) in enumerate(self.edges):
            # A self-loop (u, u) is allowed: it asks U_e psi_u = psi_u. It can
            # never be a tree edge, so it is always a chord whose cycle is
            # itself, with holonomy T_u^T U_e T_u. Knowledge graphs contain a few.
            self._adj[u].append((v, i))
            if u != v:
                self._adj[v].append((u, i))

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def edge_index(self) -> Tensor:
        """Edges as a (n_edges, 2) long tensor [source, target] on the rotations' device.

        Built once and cached: converting the Python edge list costs O(|E|)
        interpreter work, which adds up inside a training loop.
        """
        cached = getattr(self, "_edge_index", None)
        if cached is None or cached.device != self.rotations.device:
            cached = torch.tensor(self.edges, dtype=torch.long,
                                  device=self.rotations.device).reshape(-1, 2)
            self._edge_index = cached
        return cached

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
    # Tree transport
    # -----------------------------------------------------------------------
    #
    # Both methods below push something from the root down the tree:
    #   * transports() pushes the identity MATRIX, giving every T_v. Needed when
    #     you want psi-free quantities (holonomies, the projection).
    #   * propagate() pushes one VECTOR psi_root, giving every psi_v = T_v psi_r.
    #     This is all a defect evaluation needs, and it costs O(|V| d b) instead
    #     of O(|V| d b^2), a factor of b = 64 less work.
    #
    # Vertices at the same depth never depend on each other (each depends only
    # on its parent, one level up), so a whole level is one batched matmul.
    # The Python loop runs once per LEVEL, not once per vertex.

    def tree_levels(self, tree: SpanningTree) -> list[TreeLevel]:
        """Group the tree's vertices by depth, ready for batched transport.

        Raises ValueError if ``tree`` does not match this graph: an incomplete
        or repeated vertex order, a parent that comes after its child, or a
        parent edge that does not join the vertex to its parent. Failing loudly
        here is what stops a bad tree from silently mislabelling states.
        """
        n = self.n_vertices
        if len(tree.order) != n or sorted(tree.order) != list(range(n)):
            raise ValueError("tree.order must list every vertex exactly once")
        if tree.order[0] != tree.root:
            raise ValueError("tree.order must start at the root")

        depth = [-1] * n
        depth[tree.root] = 0
        buckets: list[list[int]] = [[tree.root]]
        for v in tree.order[1:]:
            p, e = tree.parent[v], tree.parent_edge[v]
            if p < 0 or depth[p] < 0:
                raise ValueError(f"parent of vertex {v} is missing or comes after it in tree.order")
            if e < 0 or set(self.edges[e]) != {p, v}:
                raise ValueError(f"parent_edge of vertex {v} does not join it to its parent {p}")
            depth[v] = depth[p] + 1
            if depth[v] == len(buckets):
                buckets.append([])
            buckets[depth[v]].append(v)

        device = self.rotations.device
        as_index = lambda xs: torch.tensor(xs, dtype=torch.long, device=device)  # noqa: E731
        position = [0] * n  # index of each vertex within its own level
        position[tree.root] = 0
        levels = [TreeLevel(as_index([tree.root]), as_index([]), as_index([]), 0)]
        for bucket in buckets[1:]:
            # Put edges stored as (parent, child) first: they use U_e. The rest
            # are stored as (child, parent) and use U_e^T. Sorting this way lets
            # one level be two contiguous slices instead of a gather + where.
            forward = [v for v in bucket if self.edges[tree.parent_edge[v]][0] == tree.parent[v]]
            backward = [v for v in bucket if self.edges[tree.parent_edge[v]][0] != tree.parent[v]]
            vertices = forward + backward
            parent_pos = [position[tree.parent[v]] for v in vertices]
            for i, v in enumerate(vertices):
                position[v] = i
            levels.append(TreeLevel(
                vertices=as_index(vertices),
                parent_pos=as_index(parent_pos),
                edges=as_index([tree.parent_edge[v] for v in vertices]),
                n_forward=len(forward),
            ))
        return levels

    def _push_down(self, tree: SpanningTree, root_payload: Tensor) -> Tensor:
        """Carry ``root_payload`` (n_blocks, b, k) down the tree; returns (n_vertices, n_blocks, b, k).

        k = b pushes matrices (transports); k = 1 pushes one column vector.
        """
        levels = self.tree_levels(tree)
        current = root_payload.unsqueeze(0)
        per_level = [current]
        for level in levels[1:]:
            x = current[level.parent_pos]
            f = level.n_forward
            current = torch.cat((
                gather_apply(self.rotations, level.edges[:f], x[:f]),
                gather_apply(self.rotations, level.edges[f:], x[f:], transpose=True),
            ))
            per_level.append(current)
        # Rows are now in level order; put them back in vertex order. Building
        # the result out of fresh tensors (no in-place writes) keeps autograd simple.
        level_order = torch.cat([level.vertices for level in levels])
        inverse = torch.empty_like(level_order)
        inverse[level_order] = torch.arange(self.n_vertices, device=level_order.device)
        return torch.cat(per_level)[inverse]

    def transports(self, tree: SpanningTree) -> Tensor:
        """Transport T_v from the root to every vertex, shape (n_vertices, n_blocks, b, b).

        Defined so that psi_v = T_v psi_r makes every TREE edge consistent.
        For child v with parent p and tree edge e = parent_edge[v]:
          - if edges[e] == (p, v): psi_v = U_e psi_p,   so T_v = U_e T_p
          - if edges[e] == (v, p): psi_p = U_e psi_v,   so T_v = U_e^T T_p
        Only call this when you need the matrices; for states use ``propagate``.
        """
        identity = identity_blocks(self.n_blocks, self.block_size,
                                   dtype=self.rotations.dtype, device=self.rotations.device)
        return self._push_down(tree, identity)

    def propagate(self, tree: SpanningTree, psi_root: Tensor) -> Tensor:
        """States psi_v = T_v psi_root for all vertices, shape (n_vertices, d).

        Pushes the vector itself down the tree; never forms a b x b transport.
        """
        if psi_root.shape != (self.dim,):
            raise ValueError(f"psi_root must have shape ({self.dim},), got {tuple(psi_root.shape)}")
        column = psi_root.reshape(self.n_blocks, self.block_size, 1)
        return self._push_down(tree, column).reshape(self.n_vertices, self.dim)
