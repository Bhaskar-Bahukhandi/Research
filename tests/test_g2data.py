"""Spec for rtvm/g2data.py: the G2 data pipeline (protocol, "G2 (H2) in full").

Brute-force references below enumerate every simple path on small random
multigraphs; your fast implementations must agree with them exactly.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from rtvm.g2data import (MAX_LEN, NEG_PER_SIDE, TrainGraph, check_no_leakage, count_paths,
                         decode, encode, path_vertices, role_pools, sample_negatives,
                         sample_paths, split_validation)

pytestmark = pytest.mark.g2data


# ---------------------------------------------------------------- helpers

def random_kg(n_entities=14, n_relations=4, n_triples=70, seed=0, loops=3):
    """Unique (h, r, t) triples with parallel facts and a few self-loops."""
    rng = np.random.default_rng(seed)
    triples = set()
    while len(triples) < n_triples - loops:
        h, t = rng.integers(n_entities, size=2)
        if h != t:
            triples.add((int(h), int(rng.integers(n_relations)), int(t)))
    while len(triples) < n_triples:
        v = int(rng.integers(n_entities))
        triples.add((v, int(rng.integers(n_relations)), v))
    return np.array(sorted(triples), dtype=np.int64)


def brute_paths(graph: TrainGraph, h: int, t: int, own: int = -1) -> list[tuple[int, ...]]:
    """Every simple path from h to t of length <= MAX_LEN, as tuples of codes."""
    if h == t:
        return []
    out = []

    def walk(v, visited, codes):
        if v == t:
            out.append(tuple(codes))
            return
        if len(codes) == MAX_LEN:
            return
        for e, w, fwd in zip(*graph.incidences(v)):
            if e == own or w in visited:
                continue
            walk(int(w), visited | {int(w)}, codes + [encode(int(e), bool(fwd))])

    walk(h, {h}, [])
    return out


def own_edge_ids(train, triples):
    index = {tuple(x): i for i, x in enumerate(train.tolist())}
    return np.array([index.get(tuple(x), -1) for x in triples.tolist()])


def path_set(row):
    return {tuple(int(c) for c in p if c != 0) for p in row if p.any()}


@pytest.fixture
def kg():
    train = random_kg()
    return train, TrainGraph.from_training(train, 14)


@pytest.fixture
def queries(kg):
    """Training triples (with own edges), random non-training pairs, and h == t."""
    train, _ = kg
    rng = np.random.default_rng(1)
    other = np.c_[rng.integers(14, size=40), rng.integers(4, size=40), rng.integers(14, size=40)]
    return np.concatenate([train, other])


# ---------------------------------------------------------------- plumbing (provided)

def test_encode_decode_roundtrip():
    for e in (0, 5, 271_000):
        for fwd in (True, False):
            assert decode(encode(e, fwd)) == (e, fwd)


def test_train_graph_incidences(kg):
    train, g = kg
    loops = train[:, 0] == train[:, 2]
    assert len(g.inc_edge) == 2 * (~loops).sum()
    assert not set(np.flatnonzero(loops)) & set(g.inc_edge.tolist())
    A = g.adjacency().toarray()
    assert (A == A.T).all() and (np.diag(A) == 0).all()
    assert A.sum() == 2 * (~loops).sum()
    deg = Counter(train[:, 0].tolist()) + Counter(train[~loops, 2].tolist())
    assert g.degree().tolist() == [deg[v] for v in range(14)]


def test_split_validation():
    v1, v2 = split_validation(101)
    assert len(v1) == 51 and len(v2) == 50
    assert not set(v1) & set(v2) and set(v1) | set(v2) == set(range(101))
    assert (split_validation(101)[0] == v1).all()
    assert not (split_validation(101, seed=1)[0] == v1).all()


def test_check_no_leakage_catches_eval_fact(kg):
    train, g = kg
    paths = np.zeros((1, 2, MAX_LEN), dtype=np.int32)
    paths[0, 0, 0] = encode(3, True)
    check_no_leakage(paths, train, train[[10]])          # edge 3 is not the eval fact
    with pytest.raises(AssertionError):
        check_no_leakage(paths, train, train[[3]])        # edge 3 is
    paths[0, 1, 0] = encode(len(train), True)
    with pytest.raises(AssertionError):
        check_no_leakage(paths, train, train[[10]])       # out of range


# ---------------------------------------------------------------- negatives (exercise)

@pytest.fixture
def neg_setup():
    train = random_kg(n_entities=40, n_relations=3, n_triples=300, seed=2, loops=0)
    rng = np.random.default_rng(3)
    positives = train[rng.choice(len(train), 60, replace=False)]
    known = set(map(tuple, train.tolist()))
    return train, positives, known


def test_negatives_shape_and_roles(neg_setup):
    train, pos, known = neg_setup
    neg, fb = sample_negatives(pos, train, known, 40)
    assert neg.shape == (60, 10, 3) and fb.shape == (60, 10) and fb.dtype == bool
    assert (neg[:, :, 1] == pos[:, None, 1]).all()
    assert (neg[:, :NEG_PER_SIDE, 2] == pos[:, None, 2]).all()     # head replaced, tail kept
    assert (neg[:, NEG_PER_SIDE:, 0] == pos[:, None, 0]).all()     # tail replaced, head kept


def test_negatives_filtered_and_distinct(neg_setup):
    train, pos, known = neg_setup
    neg, _ = sample_negatives(pos, train, known, 40)
    for row in neg.tolist():
        assert not any(tuple(x) in known for x in row)
        assert len({x[0] for x in row[:NEG_PER_SIDE]}) == NEG_PER_SIDE
        assert len({x[2] for x in row[NEG_PER_SIDE:]}) == NEG_PER_SIDE


def test_negatives_come_from_role_pools(neg_setup):
    train, pos, known = neg_setup
    heads, tails = role_pools(train)
    neg, fb = sample_negatives(pos, train, known, 40)
    for p, row, flags in zip(pos.tolist(), neg.tolist(), fb.tolist()):
        for j, (x, flag) in enumerate(zip(row, flags)):
            pool = heads[p[1]] if j < NEG_PER_SIDE else tails[p[1]]
            ent = x[0] if j < NEG_PER_SIDE else x[2]
            assert (ent in pool) != flag      # pool draws are unflagged, fallback draws are not in the pool


def test_negatives_fallback():
    # relation 0 has only 3 heads in training, and one of them is a known fact for the positive
    train = np.array([[0, 0, 9], [1, 0, 9], [2, 0, 8], [3, 1, 4], [5, 1, 6], [7, 1, 8]])
    pos = np.array([[0, 0, 9]])
    known = set(map(tuple, train.tolist()))
    neg, fb = sample_negatives(pos, train, known, 12)
    heads = neg[0, :NEG_PER_SIDE, 0].tolist()
    assert fb[0, :NEG_PER_SIDE].sum() == 4                      # pool {0,1,2}: 0 and 1 are known, only 2 is valid
    assert heads[0] == 2 and not fb[0, 0]                       # pool draws come first
    assert not set(heads[1:]) & {0, 1, 2}


def test_negatives_deterministic_and_prefix_stable(neg_setup):
    train, pos, known = neg_setup
    a, fa = sample_negatives(pos, train, known, 40)
    b, fb = sample_negatives(pos, train, known, 40)
    c, _ = sample_negatives(pos[:7], train, known, 40)
    assert (a == b).all() and (fa == fb).all() and (a[:7] == c).all()


def test_negatives_roughly_uniform():
    # one relation whose 20 heads all appear with the same tail: every head swap is valid except the truth
    train = np.array([[h, 0, 30] for h in range(20)] + [[h, 0, 31] for h in range(20)])
    pos = np.repeat(np.array([[0, 0, 32]]), 2000, axis=0)
    known = set(map(tuple, train.tolist())) | {(0, 0, 32)}
    neg, _ = sample_negatives(pos, train, known, 40)
    counts = Counter(neg[:, :NEG_PER_SIDE, 0].ravel().tolist())
    assert set(counts) == set(range(1, 20))
    expected = 2000 * NEG_PER_SIDE / 19
    assert all(abs(c - expected) < 0.2 * expected for c in counts.values())


def test_negatives_impossible_raises():
    train = np.array([[0, 0, 1], [1, 0, 2]])
    known = set(map(tuple, train.tolist()))
    with pytest.raises(ValueError):
        sample_negatives(np.array([[0, 0, 1]]), train, known, 4)


# ---------------------------------------------------------------- path counts (exercise)

def test_counts_match_brute_force(kg, queries):
    train, g = kg
    own = own_edge_ids(train, queries)
    n = count_paths(g, queries, own_edges=own)
    assert n.shape == (len(queries), 3) and n.dtype == np.int64
    for (h, _, t), o, row in zip(queries.tolist(), own.tolist(), n.tolist()):
        lengths = Counter(len(p) for p in brute_paths(g, h, t, own=o))
        assert row == [lengths[1], lengths[2], lengths[3]], (h, t, o)


def test_counts_without_exclusion_include_own_edge(kg):
    train, g = kg
    tr = train[train[:, 0] != train[:, 2]][:10]
    with_own = count_paths(g, tr)
    without = count_paths(g, tr, own_edges=own_edge_ids(train, tr))
    assert (with_own[:, 0] == without[:, 0] + 1).all()


def test_counts_chunking_does_not_matter(kg, queries):
    _, g = kg
    assert (count_paths(g, queries, chunk=7) == count_paths(g, queries, chunk=1000)).all()


# ---------------------------------------------------------------- path sampler (exercise)

@pytest.mark.parametrize("k", [2, 6, 32])
def test_sampled_paths_are_valid_simple_paths(kg, queries, k):
    train, g = kg
    own = own_edge_ids(train, queries)
    out = sample_paths(g, queries, own_edges=own, k=k)
    assert out.shape == (len(queries), k, MAX_LEN) and out.dtype == np.int32
    for (h, _, t), o, row in zip(queries.tolist(), own.tolist(), out):
        paths = [p for p in row if p.any()]
        assert len(paths) == len(path_set(row))                     # no duplicates
        for p in paths:
            L = int((p != 0).sum())
            assert (p[L:] == 0).all() and (p[:L] != 0).all()        # zeros only as trailing padding
            vs = path_vertices(g, h, p)
            assert vs[-1] == t and len(set(vs)) == len(vs)
            assert o < 0 or o not in {decode(int(c))[0] for c in p[:L]}
        check_no_leakage(row[None], train, np.empty((0, 3), dtype=np.int64))


@pytest.mark.parametrize("k", [2, 6, 32])
def test_sampler_follows_length_tiers(kg, queries, k):
    train, g = kg
    own = own_edge_ids(train, queries)
    out = sample_paths(g, queries, own_edges=own, k=k)
    for (h, _, t), o, row in zip(queries.tolist(), own.tolist(), out):
        every = brute_paths(g, h, t, own=o)
        got = path_set(row)
        assert got <= set(every)
        lens = [len(p) for p in (tuple(int(c) for c in q if c) for q in row if q.any())]
        assert lens == sorted(lens)                                  # shorter tiers first
        room = k
        for L in (1, 2, 3):
            tier = {p for p in every if len(p) == L}
            if len(tier) <= room:
                assert tier <= got                                   # whole tier taken
                room -= len(tier)
            else:
                assert len(got & tier) == room                       # overflow tier fills the rest
                assert max(lens, default=0) <= L                     # later tiers unused
                break
        else:
            assert got == set(every)                                 # everything fits: all paths kept


def test_sampler_h_equals_t_gets_nothing(kg):
    _, g = kg
    out = sample_paths(g, np.array([[3, 0, 3], [5, 1, 5]]))
    assert not out.any()


def test_sampler_deterministic(kg, queries):
    _, g = kg
    a = sample_paths(g, queries, k=4)
    assert (a == sample_paths(g, queries, k=4)).all()
    assert (a[:9] == sample_paths(g, queries[:9], k=4)).all()


def test_sampler_overflow_tier_is_uniform():
    # h=0 and t=1 joined by 1 x 4 x 1 length-3 paths through two middle layers: 0-a-b-1
    train = [[0, 0, a] for a in (2, 3)] + [[a, 0, b] for a in (2, 3) for b in (4, 5)] + [[b, 0, 1] for b in (4, 5)]
    train = np.array(train)
    g = TrainGraph.from_training(train, 6)
    q = np.repeat(np.array([[0, 1, 1]]), 3000, axis=0)
    out = sample_paths(g, q, k=2)
    counts = Counter(p for row in out for p in path_set(row))
    assert len(counts) == 4                                          # n3 = 2 * 2 = 4 paths
    expected = 3000 * 2 / 4
    assert all(abs(c - expected) < 0.1 * expected for c in counts.values())


def test_validation_fact_is_never_a_path():
    # h=0 and t=1 are linked only by a validation fact; training has no path between them
    train = np.array([[0, 0, 2], [2, 0, 3], [1, 0, 4], [4, 0, 5]])
    valid = np.array([[0, 0, 1]])
    g = TrainGraph.from_training(train, 6)
    assert count_paths(g, valid).tolist() == [[0, 0, 0]]
    assert not sample_paths(g, valid).any()
