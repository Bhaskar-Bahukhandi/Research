"""Numerical checks of notes/proofs.md (Lemmas 2 and 3), on random graphs.

Lemma 3: if one side S of a cut pair {e, f} is internally consistent, a corruption C on e
is gauge-equivalent to the corruption C~ = T C T^T on f (T = transport from e's end to
f's start inside S), so every cycle score is identical. If S is not consistent, the
signatures still agree but the scores generally differ.
"""
from __future__ import annotations

import numpy as np
import pytest

D = 4


def rand_rot(rng, d=D):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_connected(rng, vertices, extra):
    vs = list(vertices)
    edges = [(vs[i], vs[rng.integers(i)]) for i in range(1, len(vs))]       # random tree
    while extra:
        a, b = rng.choice(vs, 2, replace=False)
        edges.append((int(a), int(b)))
        extra -= 1
    return edges


def closed_walks(edges, n_vertices, rng, count=300, length=12):
    """Random closed walks, as lists of (edge index, forward)."""
    inc = [[] for _ in range(n_vertices)]
    for i, (a, b) in enumerate(edges):
        inc[a].append((i, True, b))
        inc[b].append((i, False, a))
    walks = []
    while len(walks) < count:
        start = int(rng.integers(n_vertices))
        v, walk = start, []
        for _ in range(length):
            i, fwd, w = inc[v][rng.integers(len(inc[v]))]
            walk.append((i, fwd))
            v = w
            if v == start and len(walk) >= 3:
                walks.append(walk)
                break
    return walks


def scores(U, walks):
    out = []
    for walk in walks:
        H = np.eye(D)
        for i, fwd in walk:
            H = (U[i] if fwd else U[i].T) @ H
        out.append(np.linalg.norm(H - np.eye(D)))
    return np.array(out)


def cut_pair_graph(seed):
    """S = {0..5}, outside = {6..11}; e = 6 -> 0 enters S at v = 0, f = 3 -> 7 leaves S at v' = 3."""
    rng = np.random.default_rng(seed)
    edges = random_connected(rng, range(6), 4) + random_connected(rng, range(6, 12), 4)
    e, f = len(edges), len(edges) + 1
    edges += [(6, 0), (3, 7)]
    frames = [rand_rot(rng) for _ in range(12)]
    U = [frames[b] @ frames[a].T for a, b in edges]            # consistent everywhere
    return rng, edges, e, f, frames, U


@pytest.mark.parametrize("seed", range(5))
def test_lemma3_exact_when_side_consistent(seed):
    rng, edges, e, f, frames, U = cut_pair_graph(seed)
    C = rand_rot(rng)
    T = frames[3] @ frames[0].T                                 # transport v=0 -> v'=3 inside S
    on_e = list(U)
    on_e[e] = C @ U[e]
    on_f = list(U)
    on_f[f] = U[f] @ (T @ C @ T.T)
    walks = closed_walks(edges, 12, rng)
    a, b = scores(on_e, walks), scores(on_f, walks)
    assert a.max() > 1e-3                                       # some walks do see the corruption
    assert np.allclose(a, b, atol=1e-9)


def test_lemma3_breaks_when_both_sides_inconsistent():
    rng, edges, e, f, frames, U = cut_pair_graph(0)
    C = rand_rot(rng)
    T = frames[3] @ frames[0].T
    inner = next(i for i, (a, b) in enumerate(edges) if a < 6 and b < 6 and i != e)
    outer = next(i for i, (a, b) in enumerate(edges) if a >= 6 and b >= 6)
    U = list(U)
    U[inner] = rand_rot(rng) @ U[inner]                         # S no longer consistent
    U[outer] = rand_rot(rng) @ U[outer]                         # nor the other side
    on_e, on_f = list(U), list(U)
    on_e[e] = C @ U[e]
    on_f[f] = U[f] @ (T @ C @ T.T)
    walks = closed_walks(edges, 12, rng)
    assert not np.allclose(scores(on_e, walks), scores(on_f, walks), atol=1e-6)


def test_lemma2_degree_two_vertex():
    rng = np.random.default_rng(7)
    edges = random_connected(rng, range(8), 6)
    edges += [(0, 8), (8, 5)]                                   # vertex 8 has degree 2
    frames = [rand_rot(rng) for _ in range(9)]
    U = [frames[b] @ frames[a].T for a, b in edges]
    A, B = U[-2], U[-1]
    C = rand_rot(rng)
    first, second = list(U), list(U)
    first[-2] = C @ A                                           # (CA, B)
    second[-1] = B @ C                                          # (A, BC)
    walks = closed_walks(edges, 9, rng)
    assert np.allclose(scores(first, walks), scores(second, walks), atol=1e-9)
