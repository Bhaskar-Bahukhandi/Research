"""G2 data pipeline: validation split, role-matched negatives, path counts and path samples.

Every rule here is fixed in notes/protocol.md, section "G2 (H2) in full"; the
docstrings quote the rule they implement. The tests in tests/test_g2data.py are the
spec. Functions that raise NotImplementedError are your exercises; the test run shows
them as SKIPPED until you implement them.

Leakage rule: paths and counts come from a TrainGraph, and a TrainGraph is built from
training triples only. No function here takes validation or test triples as graph
input, so they can never become part of a path.

Path encoding: a path from h to t is a sequence of training edges. Step i is stored
as a signed code: +(e + 1) if training triple e is traversed head -> tail, and
-(e + 1) if it is traversed tail -> head. 0 is padding. Saved path arrays have shape
(n_triples, K, 3).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

K_PATHS = 32          # paths kept per triple
MAX_LEN = 3           # longest path, in edges
NEG_PER_SIDE = 5      # head replacements, then tail replacements, per positive


def encode(edge: int, forward: bool) -> int:
    return edge + 1 if forward else -(edge + 1)


def decode(code: int) -> tuple[int, bool]:
    """(edge id, forward) from a signed code; code must not be 0."""
    return abs(code) - 1, code > 0


# ---------------------------------------------------------------- training graph

@dataclass
class TrainGraph:
    """Undirected multigraph of the training triples; each fact is its own edge.

    Self-loop facts (h == t) are kept in `heads`/`tails` (edge ids stay aligned with
    the training array) but are left out of the adjacency, because a simple path
    never uses them.
    """
    num_entities: int
    heads: np.ndarray          # (E,) head of training edge e
    relations: np.ndarray      # (E,)
    tails: np.ndarray          # (E,)
    indptr: np.ndarray         # CSR over vertices: incidences of vertex v are indptr[v]:indptr[v+1]
    inc_edge: np.ndarray       # edge id of each incidence
    inc_other: np.ndarray      # the other endpoint
    inc_forward: np.ndarray    # True if leaving v along the edge goes head -> tail

    @classmethod
    def from_training(cls, train: np.ndarray, num_entities: int) -> "TrainGraph":
        train = np.asarray(train, dtype=np.int64)
        h, r, t = train[:, 0], train[:, 1], train[:, 2]
        e = np.arange(len(train))
        keep = h != t
        # each non-loop edge gives two incidences: at h (forward) and at t (backward)
        v = np.concatenate([h[keep], t[keep]])
        other = np.concatenate([t[keep], h[keep]])
        edge = np.concatenate([e[keep], e[keep]])
        fwd = np.concatenate([np.ones(keep.sum(), bool), np.zeros(keep.sum(), bool)])
        order = np.lexsort((edge, v))
        v, other, edge, fwd = v[order], other[order], edge[order], fwd[order]
        indptr = np.zeros(num_entities + 1, dtype=np.int64)
        np.add.at(indptr, v + 1, 1)
        indptr = np.cumsum(indptr)
        return cls(num_entities, h, r, t, indptr, edge, other, fwd)

    @property
    def num_edges(self) -> int:
        return len(self.heads)

    def incidences(self, v: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(edge ids, other endpoints, forward flags) of the non-loop edges at v."""
        s, e = self.indptr[v], self.indptr[v + 1]
        return self.inc_edge[s:e], self.inc_other[s:e], self.inc_forward[s:e]

    def adjacency(self) -> sp.csr_matrix:
        """Symmetric multigraph adjacency: A[u, v] = number of non-loop facts between u and v."""
        keep = self.heads != self.tails
        h, t = self.heads[keep], self.tails[keep]
        n = self.num_entities
        A = sp.coo_matrix((np.ones(2 * len(h)), (np.r_[h, t], np.r_[t, h])), shape=(n, n))
        return A.tocsr()

    def degree(self) -> np.ndarray:
        """Training facts touching each entity (a self-loop counts once), for the degree feature."""
        deg = np.bincount(self.heads, minlength=self.num_entities)
        deg += np.bincount(self.tails, minlength=self.num_entities)
        deg -= np.bincount(self.heads[self.heads == self.tails], minlength=self.num_entities)
        return deg


def path_vertices(graph: TrainGraph, h: int, codes) -> list[int]:
    """Vertices visited by a path that starts at h (padding codes are ignored)."""
    out = [h]
    for c in codes:
        if c == 0:
            continue
        e, fwd = decode(int(c))
        a, b = (graph.heads[e], graph.tails[e]) if fwd else (graph.tails[e], graph.heads[e])
        if a != out[-1]:
            raise ValueError(f"path breaks at edge {e}: expected to leave {out[-1]}, edge leaves {a}")
        out.append(int(b))
    return out


def check_no_leakage(paths: np.ndarray, train: np.ndarray, eval_triples: np.ndarray,
                     own_edges: np.ndarray | None = None) -> None:
    """Raise if any saved path step is not a training fact, or is a validation/test fact.

    Run on every saved path array before it is committed. Checks that every nonzero
    code names a valid training edge id, and that no edge used by any path equals a
    triple in `eval_triples` (validation and test positives). With own_edges (training
    triples), also checks that no triple's paths use its own edge.
    """
    if own_edges is not None:
        own = np.asarray(own_edges)
        steps = np.abs(np.asarray(paths)).reshape(len(own), -1) - 1
        if ((steps == own[:, None]) & (own[:, None] >= 0)).any():
            raise AssertionError("a training triple's paths use its own edge")
    train = np.asarray(train)
    codes = np.asarray(paths).ravel()
    codes = codes[codes != 0]
    edges = np.unique(np.abs(codes) - 1)
    if len(edges) and (edges.min() < 0 or edges.max() >= len(train)):
        raise AssertionError("a path step names an edge id outside the training array")
    used = set(map(tuple, train[edges].tolist()))
    leaked = used & set(map(tuple, np.asarray(eval_triples).tolist()))
    if leaked:
        raise AssertionError(f"{len(leaked)} validation/test facts appear in paths, e.g. {next(iter(leaked))}")


# ---------------------------------------------------------------- validation split

def split_validation(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Protocol: "Validation is split once, 50/50 by positive, seed 0".

    Returns sorted index arrays (V1, V2). V1 gets the extra positive when n is odd.
    """
    perm = np.random.default_rng(seed).permutation(n)
    half = (n + 1) // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


# ---------------------------------------------------------------- negatives

def role_pools(train: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Sorted arrays of the entities seen as head, and as tail, of each relation in training."""
    heads, tails = defaultdict(set), defaultdict(set)
    for h, r, t in np.asarray(train).tolist():
        heads[r].add(h)
        tails[r].add(t)
    return ({r: np.array(sorted(s)) for r, s in heads.items()},
            {r: np.array(sorted(s)) for r, s in tails.items()})


def sample_negatives(positives: np.ndarray, train: np.ndarray, known: set[tuple[int, int, int]],
                     num_entities: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """EXERCISE. Ten role-matched negatives per positive (protocol, "Negatives").

    Returns (negatives, fallback):
      negatives: int64 array (n, 10, 3). Columns 0-4 replace the head, 5-9 the tail.
      fallback:  bool array (n, 10), True where a replacement came from the fallback.

    Rules:
      * Head replacements h' are drawn uniformly without replacement from the heads of r
        in training (role_pools); tail replacements t' likewise from the tails of r.
      * A candidate that is in `known` (true in train, valid or test) is discarded and
        the next one drawn. The true entity itself is always in `known`, so it is
        never used.
      * Fallback: if a pool runs out before 5 valid replacements, take the rest uniformly
        without replacement from all entities not already used for this positive and
        side, with the same filtering, and flag them. Within a side, pool draws come
        first, then fallback draws.
      * Use one generator per positive, np.random.default_rng([seed, i]) for positive i,
        so the result does not depend on how positives are batched. Use it for both
        sides, head first.
      * Raise ValueError if even the fallback cannot supply 5 distinct valid replacements.

    Hint: shuffling the pool once (rng.permutation) and walking through it is "uniform
    without replacement". structure_check.py does this without the fallback.
    """
    positives = np.asarray(positives, dtype=np.int64)
    head_pools, tail_pools = role_pools(train)
    negatives = np.empty((len(positives), 2 * NEG_PER_SIDE, 3), dtype=np.int64)
    fallback = np.zeros((len(positives), 2 * NEG_PER_SIDE), dtype=bool)
    all_entities = np.arange(num_entities, dtype=np.int64)

    for i, (h, r, t) in enumerate(positives.tolist()):
        rng = np.random.default_rng([seed, i])

        # Keep the head side first so each positive always consumes random numbers
        # in the same order, even if the positives are processed in batches.
        for side, pool, start in (
            ("head", head_pools.get(r, np.empty(0, dtype=np.int64)), 0),
            ("tail", tail_pools.get(r, np.empty(0, dtype=np.int64)), NEG_PER_SIDE),
        ):
            replacements = []
            used = set()

            def candidate_triple(entity: int) -> tuple[int, int, int]:
                if side == "head":
                    return entity, r, t
                return h, r, entity

            for entity in rng.permutation(pool):
                entity = int(entity)
                triple = candidate_triple(entity)
                if triple in known:
                    continue
                replacements.append(entity)
                used.add(entity)
                if len(replacements) == NEG_PER_SIDE:
                    break

            role_pool_count = len(replacements)
            if role_pool_count < NEG_PER_SIDE:
                # Pool candidates have all been checked, so these draws fill only
                # the shortfall. This shuffle uses the same per-positive generator, so a
                # head-side fallback changes the tail side's draws; that is still
                # deterministic, because the order (head first) is fixed.
                for entity in rng.permutation(all_entities):
                    entity = int(entity)
                    if entity in used or candidate_triple(entity) in known:
                        continue
                    replacements.append(entity)
                    used.add(entity)
                    if len(replacements) == NEG_PER_SIDE:
                        break

            if len(replacements) < NEG_PER_SIDE:
                raise ValueError(
                    f"could not find {NEG_PER_SIDE} valid {side} replacements "
                    f"for positive {i}"
                )

            for offset, entity in enumerate(replacements):
                if side == "head":
                    negatives[i, start + offset] = (entity, r, t)
                else:
                    negatives[i, start + offset] = (h, r, entity)
                fallback[i, start + offset] = offset >= role_pool_count

    return negatives, fallback


# ---------------------------------------------------------------- paths

def count_paths(graph: TrainGraph, triples: np.ndarray, own_edges: np.ndarray | None = None,
                chunk: int = 512) -> np.ndarray:
    """EXERCISE. Exact numbers (n1, n2, n3) of simple paths of length 1, 2, 3 from h to t.

    Returns an int64 array (n, 3). A path is a sequence of distinct-vertex steps along
    training edges, in either direction; parallel facts give different paths. Triples
    with h == t get (0, 0, 0) (protocol note of 2026-11-08).

    own_edges[i] >= 0 is the edge id of triple i itself (training triples). That edge
    joins h and t, and it is removed before counting, so a triple never supports
    itself. -1 (or own_edges=None) means nothing is removed.

    Do not enumerate paths: with hub entities (degree up to 7,614 on FB15k-237) that is
    far too slow. Use the adjacency A (graph.adjacency()), with no diagonal:
      * n1 = A[h, t]
      * n2 = (A^2)[h, t], since h != t and there are no loops
      * n3 = (A^3)[h, t] minus the walks h-t-b-t and h-a-h-t that revisit a vertex.
        Work out that correction by hand first (inclusion-exclusion; mind the walk
        h-t-h-t, which is counted in both), then the change when one h-t edge is removed.
    Compute rows in chunks of `chunk` triples, as structure_check.walks does.

    Derivation (your hand proof, Week 1): for h != t and no loops, a length-3 walk
    h-a-b-t repeats a vertex only if b = h or a = t, and both only for h-t-h-t, so
        n3 = (A^3)_ht - A_ht ((A^2)_hh + (A^2)_tt) + A_ht^3,
    where (A^2)_vv = sum_a A_va^2 (not the degree, in a multigraph). Removing the own
    h-t edge changes only n1: a simple path of length 2 or 3 cannot use an h-t edge.
    """
    tr = np.asarray(triples, dtype=np.int64)
    out = np.zeros((len(tr), 3), dtype=np.int64)
    if len(tr) == 0:
        return out
    A = graph.adjacency().astype(np.int64)
    sq = np.asarray(A.multiply(A).sum(axis=1)).ravel()          # (A^2)_vv
    h, t = tr[:, 0], tr[:, 2]
    for s in range(0, len(tr), chunk):
        hh, tt = h[s:s + chunk], t[s:s + chunk]
        H, T = A[hh], A[tt]
        n1 = np.asarray(A[hh, tt]).ravel()
        n2 = np.asarray(H.multiply(T).sum(axis=1)).ravel()
        walks3 = np.asarray((H @ A).multiply(T).sum(axis=1)).ravel()
        n3 = walks3 - n1 * (sq[hh] + sq[tt]) + n1 ** 3
        out[s:s + chunk] = np.stack([n1, n2, n3], axis=1)
    if own_edges is not None:
        own = (np.asarray(own_edges) >= 0) & (h != t)          # self-loop facts are not in A
        if (out[own, 0] < 1).any():
            raise ValueError("an own edge was given for a triple with no h-t training edge")
        out[own, 0] -= 1
    out[h == t] = 0
    return out


def sample_paths(graph: TrainGraph, triples: np.ndarray, own_edges: np.ndarray | None = None,
                 k: int = K_PATHS, seed: int = 0) -> np.ndarray:
    """EXERCISE. Up to k simple paths per triple, chosen by length tier (protocol, "Paths and cycles").

    Returns int32 codes of shape (n, k, MAX_LEN), padded with 0: path j of triple i is
    out[i, j, :L] for its length L, followed by zeros. Unused path slots are all zero.

    Rules:
      * Tiers in order: length 1, then 2, then 3. A tier that fits in the remaining
        slots is taken whole. The first tier that overflows is sampled uniformly without
        replacement, and later tiers are not used.
      * own_edges[i] >= 0 is the edge id of triple i itself (training triples); that edge
        is never used. -1 means no own edge (validation, test and negatives).
      * Paths are simple: no vertex repeats. Triples with h == t get no paths.
      * One generator per triple: np.random.default_rng([seed, i]).
      * Within a tier taken whole, order paths canonically (sorted by their code tuples)
        so the output is deterministic.

    Hint for the overflowing length-3 tier at hubs: a length-3 path h-a-b-t is fixed by
    its middle edge (a, b) and one fact for each of the outer steps. Weight each middle
    edge by A[h, a] * A[b, t] (with a != t, b != h), draw path indices uniformly without
    replacement from range(n3), and map each index to a path through the cumulative
    weights. That avoids listing all n3 paths.
    """
    tr = np.asarray(triples, dtype=np.int64)
    out = np.zeros((len(tr), k, MAX_LEN), dtype=np.int32)
    for i, (h, _, t) in enumerate(tr.tolist()):
        rng = np.random.default_rng([seed, i])
        own = -1 if own_edges is None else int(own_edges[i])
        for j, path in enumerate(_paths_for(graph, h, t, own, k, rng)):
            out[i, j, :len(path)] = path
    return out


def _signed(edges: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Codes for leaving a vertex along its incidences."""
    return np.where(forward, edges + 1, -(edges + 1))


def _side(graph: TrainGraph, v: int, banned: int, own: int):
    """Neighbours of v other than `banned`, without the own edge, grouped by neighbour.

    Returns (nbrs, first, count, codes): nbrs sorted; the edges from v to nbrs[j] are
    codes[first[j]: first[j] + count[j]], as codes for leaving v (sorted by edge id).
    """
    e, o, f = graph.incidences(v)
    keep = (o != banned) & (e != own)
    e, o, f = e[keep], o[keep], f[keep]
    order = np.lexsort((e, o))
    e, o, f = e[order], o[order], f[order]
    nbrs, first, count = np.unique(o, return_index=True, return_counts=True)
    return nbrs, first, count, _signed(e, f)


def _pick(weights: np.ndarray, radix: np.ndarray, room: int, rng: np.random.Generator):
    """All path indices if they fit in `room`, else `room` drawn uniformly without
    replacement; each index is mapped to (unit u, outer choices p, q) by mixed radix."""
    total = int(weights.sum())
    idx = np.arange(total) if total <= room else np.sort(rng.choice(total, room, replace=False))
    cum = np.cumsum(weights)
    u = np.searchsorted(cum, idx, side="right")
    off = idx - (cum[u] - weights[u])
    return total, u, off // radix[u], off % radix[u]


def _paths_for(graph: TrainGraph, h: int, t: int, own: int, room: int, rng: np.random.Generator):
    """The protocol's tiered path choice for one triple, as a list of code tuples."""
    if h == t:
        return []
    out: list[tuple[int, ...]] = []

    # length 1: parallel facts between h and t
    e, o, f = graph.incidences(h)
    m = (o == t) & (e != own)
    tier = sorted((int(c),) for c in _signed(e[m], f[m]))
    if len(tier) > room:
        return sorted(tier[j] for j in rng.choice(len(tier), room, replace=False))
    out += tier
    room -= len(tier)
    if room == 0:
        return out

    hn, hf, hc, hcode = _side(graph, h, t, own)
    tn, tf, tc, tcode = _side(graph, t, h, own)

    # length 2: h - a - t through each common neighbour a; arriving at t reverses t's leaving code
    _, ih, it = np.intersect1d(hn, tn, assume_unique=True, return_indices=True)
    total, u, p, q = _pick(hc[ih] * tc[it], tc[it], room, rng)
    tier = sorted((int(hcode[hf[ih[x]] + a]), int(-tcode[tf[it[x]] + b])) for x, a, b in zip(u, p, q))
    out += tier
    if total > room:
        return out
    room -= total
    if room == 0:
        return out

    # length 3: h - a - b - t, one unit per middle step a -> b; iterate from the cheaper end
    cost = lambda nbrs: int((graph.indptr[nbrs + 1] - graph.indptr[nbrs]).sum())   # noqa: E731
    flip = cost(tn) < cost(hn)
    X, Y = ((tn, tf, tc, tcode), (hn, hf, hc, hcode)) if flip else ((hn, hf, hc, hcode), (tn, tf, tc, tcode))
    xn, xf, xc, xcode = X
    yn, yf, yc, ycode = Y
    lens = graph.indptr[xn + 1] - graph.indptr[xn]
    unit_x = np.repeat(np.arange(len(xn)), lens)
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(graph.indptr[xn], lens)
    b = graph.inc_other[pos]
    jy = np.minimum(np.searchsorted(yn, b), max(len(yn) - 1, 0))
    ok = (yn[jy] == b) if len(yn) else np.zeros(len(b), bool)
    ux, uy = unit_x[ok], jy[ok]
    mid = _signed(graph.inc_edge[pos[ok]], graph.inc_forward[pos[ok]])
    total, u, p, q = _pick(xc[ux] * yc[uy], yc[uy], room, rng)
    paths = [(int(xcode[xf[ux[x]] + a]), int(mid[x]), int(-ycode[yf[uy[x]] + c])) for x, a, c in zip(u, p, q)]
    if flip:                                   # built from t to h: reverse order and direction
        paths = [tuple(-c for c in reversed(path)) for path in paths]
    return out + sorted(paths)
