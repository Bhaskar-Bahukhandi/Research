"""Tests for experiments/g2_secondary.py (S2, S7, S9; protocol notes of 2026-11-22)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rtvm.g2eval import grouped_auc, grouped_auc_rows
from rtvm.g2gate import primary_test
from test_g2gate import synthetic_split        # not tests.*: an installed "tests" package can shadow ours

ROOT = Path(__file__).resolve().parents[1]


def _secondary():
    spec = importlib.util.spec_from_file_location("g2_secondary", ROOT / "experiments" / "g2_secondary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_masked_rows_match_the_gate_and_the_v1_rule():
    sec = _secondary()
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(size=200).round(1), rng.normal(size=(200, 10)).round(1)     # rounding makes ties
    assert np.array_equal(sec.masked_rows(pos, neg, np.ones((200, 10), bool)), grouped_auc_rows(pos, neg))
    valid = rng.random((200, 10)) < 0.4
    valid[:5] = False                                                                  # positives with no negative
    rows = sec.masked_rows(pos, neg, valid)
    assert np.isnan(rows[:5]).all() and not np.isnan(rows[(valid.sum(1) > 0)]).any()
    assert np.nanmean(rows) == pytest.approx(grouped_auc(pos, neg, neg_valid=valid), abs=1e-12)


def test_pooled_auc_by_resample_matches_explicit_repetition():
    sec = _secondary()
    rng = np.random.default_rng(1)
    n, k, G = 120, 4, 7
    rel = rng.integers(0, G, n) * 3 + 5                       # arbitrary relation ids
    pos, neg = rng.normal(0.5, 1, n).round(1), rng.normal(size=(n, k)).round(1)
    draws = rng.integers(0, G, size=(5, G))
    times = np.vstack([np.ones((1, G)), sec._times(draws, G)])
    got = sec.pooled_auc_by_resample(pos, neg, rel, times)
    y = np.r_[np.ones(n), np.zeros(n * k)]
    assert got[0] == pytest.approx(roc_auc_score(y, np.r_[pos, neg.ravel()]), abs=1e-12)
    _, cl = np.unique(rel, return_inverse=True)
    for b in range(1, len(times)):
        w = times[b][cl]                                      # each positive's weight, shared by its negatives
        rp, rn = np.repeat(pos, w.astype(int)), np.repeat(neg, w.astype(int), axis=0).ravel()
        want = roc_auc_score(np.r_[np.ones(len(rp)), np.zeros(len(rn))], np.r_[rp, rn])
        assert got[b] == pytest.approx(want, abs=1e-12)


def test_decile_indicators():
    sec = _secondary()
    rng = np.random.default_rng(2)
    vp, vn = rng.random(1000), rng.random((1000, 10))
    cp, cn = (rng.random(1000) < 0.8).astype(float), (rng.random((1000, 10)) < 0.8).astype(float)
    edges, columns = sec.decile_indicators(vp, vn, cp, cn)
    assert len(edges) == 11
    cols = columns(vn, cn)
    assert set(cols) == set(sec.BIN_NAMES) and len(cols) == 9
    total = sum(cols.values())
    assert (total[cn == 0] == 0).all() and set(np.unique(total[cn > 0])) <= {0.0, 1.0}
    share = [cols[b][cn > 0].mean() for b in sec.BIN_NAMES]   # V2 covered rows, about a tenth in each decile
    assert all(abs(x - 0.1) < 0.02 for x in share)


def _with_coverage(split, rng, share=0.8):
    n, k = split["neg"]["covered"].shape
    split["pos"]["covered"] = (rng.random(n) < share).astype(float)
    split["neg"]["covered"] = (rng.random((n, k)) < share).astype(float)
    return split


def test_basic_analyses_reproduce_the_primary_test():
    sec = _secondary()
    rng = np.random.default_rng(3)
    v2 = _with_coverage(synthetic_split(rng, 800, 10, True), rng)
    test = _with_coverage(synthetic_split(rng, 800, 10, True), rng)
    primary = primary_test(v2, test, n_boot=300)
    none = np.zeros((800, 10), bool)
    res = sec.basic_analyses(v2, test, none, n_boot=300, pooled_boot=50)
    for m in ("B0", "B1"):
        for s in (0, 1, 2):
            assert res["check"]["grouped_auc"][f"{m}:{s}"] == pytest.approx(primary["seeds"][s][m]["grouped_auc"],
                                                                            abs=1e-12)
    # with no fallback negative, S9(a) is the primary test itself
    assert res["s9_fallback_excluded"]["delta_bar"] == pytest.approx(primary["delta_bar"], abs=1e-12)
    assert res["s9_fallback_excluded"]["ci95"] == pytest.approx(primary["ci95"], abs=1e-12)
    # S2(a) uses only covered positives with a covered negative
    assert res["s2_covered"]["n_positives"] < 800
    assert res["s2_covered"]["n_negatives"] <= int((test["neg"]["covered"] > 0).sum())
    # S7 covers every relation once, and its counts add up
    s7 = res["s7"]
    assert s7["positive"] + s7["negative"] + s7["zero"] == s7["n_relations"] == 25
    assert sum(c for c, _ in s7["all_relations"].values()) == 800
    # an informative holonomy feature shows up in every view
    assert res["s2_pooled"]["delta_bar"] > 0 and res["s9_binned"]["delta_bar"] > 0
    assert len(res["s9_binned"]["per_seed"][0]["log_odds_vs_decile1"]) == 9


def test_rotate_rotations_act_as_the_rotate_score():
    import torch
    sec = _secondary()
    gamma, dim = 9.0, 5
    g = torch.Generator().manual_seed(0)
    w = (torch.rand(3, dim, generator=g) - 0.5) * 2 * (gamma + 2) / dim       # RotatE's init range
    R = sec.rotate_rotations(w, gamma, dim)                                   # (3, dim, 2, 2)
    assert R.shape == (3, dim, 2, 2)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(2).expand(3, dim, 2, 2), atol=1e-6)
    h = torch.complex(torch.randn(dim, generator=g), torch.randn(dim, generator=g))
    theta = w[1] * (np.pi / ((gamma + 2) / dim))
    want = h * torch.polar(torch.ones(dim), theta)                            # PhaseRotatEInteraction's h * exp(i theta)
    got = (R[1] @ torch.stack([h.real, h.imag], -1)[..., None])[..., 0]
    assert torch.allclose(got, torch.stack([want.real, want.imag], -1), atol=1e-6)


def test_rotate_holonomy_is_zero_exactly_when_phases_compose():
    import torch
    from rtvm.holonomy_model import cycle_defect, path_transport
    sec = _secondary()
    gamma, dim = 9.0, 4
    a, b = torch.tensor([0.3, -1.1, 0.7, 0.2]), torch.tensor([0.5, 0.4, -0.9, 1.0])
    w = torch.stack([a, b, a + b, a - b])               # r2 = a then b; r3 = a then b backwards
    table = sec.rotation_table(sec.rotate_rotations(w, gamma, dim))           # [U_0..U_3, U_0^T..U_3^T, I]
    pad = 8
    steps = torch.tensor([[0, 1, pad], [0, 4 + 1, pad], [0, 1, pad]])
    closing = torch.tensor([2, 3, 3])
    d = cycle_defect(table, path_transport(table, steps), closing)
    assert d[0] < 1e-5 and d[1] < 1e-5                                         # cycles that close
    phi = ((a + b) - (a - b)) * (np.pi / ((gamma + 2) / dim))                  # the third does not
    assert d[2].item() == pytest.approx(torch.sqrt((4 - 4 * torch.cos(phi)).sum()).item(), rel=1e-5)


def test_quate_rotations_act_as_the_official_score():
    import torch
    from pykeen.nn import quaternion
    from rtvm.baselines import NormalizedQuatEInteraction
    sec = _secondary()
    g = torch.Generator().manual_seed(1)
    R, dim = 3, 5
    w = torch.randn(R, dim * 4, generator=g) * 0.3                           # stored unnormalised, flattened
    table = quaternion.multiplication_table()
    M = sec.quate_rotations(w, table)                                          # (R, dim, 4, 4)
    assert M.shape == (R, dim, 4, 4)
    assert torch.allclose(M @ M.transpose(-1, -2), torch.eye(4).expand(R, dim, 4, 4), atol=1e-5)
    assert torch.allclose(torch.linalg.det(M), torch.ones(R, dim), atol=1e-5)  # rotations, not reflections
    h, t = torch.randn(dim, 4, generator=g), torch.randn(dim, 4, generator=g)
    want = NormalizedQuatEInteraction()(h, w[1].reshape(dim, 4), t)          # -sum_d <h_d (x) r^_d, t_d>
    got = -(t * (M[1] @ h[..., None])[..., 0]).sum()
    assert got.item() == pytest.approx(want.item(), rel=1e-5, abs=1e-5)


def test_quate_holonomy_closes_for_the_product_and_not_the_reverse_order():
    import torch
    from pykeen.nn import quaternion
    from rtvm.holonomy_model import cycle_defect, path_transport
    sec = _secondary()
    table = quaternion.multiplication_table()
    g = torch.Generator().manual_seed(2)
    a, b = torch.randn(2, 4, generator=g), torch.randn(2, 4, generator=g)      # 2 coordinates, unit after scaling
    a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    ab = torch.einsum("di,dj,ijk->dk", a, b, table)                            # a (x) b
    ba = torch.einsum("di,dj,ijk->dk", b, a, table)                            # b (x) a
    w = torch.stack([a, b, ab, ba]).reshape(4, -1)
    T = sec.rotation_table(sec.quate_rotations(w, table))
    steps = torch.tensor([[0, 1, 8], [0, 1, 8]])                               # step a, then step b; pad = 2R = 8
    d = cycle_defect(T, path_transport(T, steps), torch.tensor([2, 3]))
    assert d[0] < 1e-5                                                         # (h (x) a) (x) b = h (x) (a (x) b)
    assert d[1] > 1e-2                                                         # noncommutative: b (x) a differs


def test_s6_with_the_gate_features_is_the_primary_test():
    sec = _secondary()
    rng = np.random.default_rng(5)
    v2, test = synthetic_split(rng, 500, 10, True), synthetic_split(rng, 500, 10, True)
    same = {split: {s: (d["pos"][f"holonomy:{s}"], d["neg"][f"holonomy:{s}"]) for s in (0, 1, 2)}
            for split, d in (("v2", v2), ("test", test))}
    res = sec.s6_analysis(v2, test, same, n_boot=200)
    primary = primary_test(v2, test, n_boot=200)
    assert "decision" not in res
    assert res["delta_bar"] == primary["delta_bar"] and res["ci95"] == primary["ci95"]
    noise = {split: {s: (rng.normal(size=500), rng.normal(size=(500, 10))) for s in (0, 1, 2)}
             for split in ("v2", "test")}
    assert sec.s6_analysis(v2, test, noise, n_boot=200)["delta_bar"] < primary["delta_bar"]
    assert v2["pos"]["holonomy:0"] is same["v2"][0][0]                        # the inputs are not modified


def test_estimation_set_draws_up_to_the_cap_per_relation():
    sec = _secondary()
    rng = np.random.default_rng(6)
    train = np.stack([rng.integers(0, 50, 400), rng.choice(3, 400, p=[0.7, 0.25, 0.05]), rng.integers(0, 50, 400)], 1)
    edges = sec.estimation_set(train, 4, per_relation=60)
    counts = np.bincount(train[edges, 1], minlength=4)
    assert list(counts) == [60, 60, int((train[:, 1] == 2).sum()), 0]   # all of a small relation; none of an absent one
    assert len(np.unique(edges)) == len(edges)
    assert np.array_equal(edges, sec.estimation_set(train, 4, per_relation=60))      # deterministic


def test_pattern_keys_by_hand():
    sec = _secondary()
    train_rel = np.array([0, 1, 2])                    # R = 3, base 7; step = 2 * rel + (backward) + 1
    codes = np.array([[[1, -2, 0], [0, 0, 0], [3, 0, 0]]])
    keys = sec.pattern_keys(codes, train_rel, 3)
    assert keys.tolist() == [[(1 * 7 + 4) * 7, 0, 5 * 49]]
    # the same pattern through different edges of the same relations gets the same key
    one = sec.pattern_keys(np.array([[1, -2, 0]]), np.array([0, 1]), 3)        # edge 0 (rel 0) fwd, edge 1 (rel 1) bwd
    other = sec.pattern_keys(np.array([[2, -1, 0]]), np.array([1, 0]), 3)      # edge 1 (rel 0) fwd, edge 0 (rel 1) bwd
    assert one.tolist() == other.tolist() == [(1 * 7 + 4) * 7]
    assert sec.pattern_keys(np.array([[-1, 2, 0]]), np.array([0, 1]), 3).tolist() != one.tolist()   # direction counts


def test_rule_table_and_features_by_hand():
    sec = _secondary()
    A, B, C = 11, 22, 33                               # pattern keys
    keys = np.array([[A, A], [A, B], [A, 0], [A, B], [A, B], [B, 0], [C, 0]])   # duplicate A in row 0 counts once
    relations = np.array([0, 0, 0, 0, 0, 0, 1])
    labels = np.array([1, 1, 1, 1, 0, 0, 1])
    tk, tc, seen = sec.rule_table(keys, relations, labels, num_relations=2, min_support=5)
    base3 = 5 ** 3
    assert seen == 3 and tk.tolist() == [0 * base3 + A] and tc.tolist() == [0.8]    # B: support 4, C: 1 -> unrated
    f = sec.rule_features(np.array([[A, B], [B, 0], [0, 0], [A, 0]]), np.array([0, 0, 0, 1]), tk, tc, 2)
    assert f["rule_max"].tolist() == [0.8, 0, 0, 0] and f["rule_mean"].tolist() == [0.8, 0, 0, 0]
    assert f["rule_rated"].tolist() == [1, 0, 0, 0]    # a pattern is rated per relation, not globally


def test_compare_is_the_primary_test_for_b0_against_b1():
    sec = _secondary()
    rng = np.random.default_rng(7)
    v2, test = synthetic_split(rng, 500, 10, True), synthetic_split(rng, 500, 10, True)
    primary = primary_test(v2, test, n_boot=200)
    res = sec.compare(v2, test, sec.B0, sec.B1, n_boot=200)
    assert res["delta_bar"] == pytest.approx(primary["delta_bar"], abs=1e-15)
    assert res["ci95"] == pytest.approx(primary["ci95"], abs=1e-15)


def test_s3_analysis_sees_an_informative_rule():
    sec = _secondary()
    rng = np.random.default_rng(8)
    v2, test = synthetic_split(rng, 600, 10, False), synthetic_split(rng, 600, 10, False)

    def rules(n, k):
        good_p, good_n = rng.random(n) < 0.8, rng.random((n, k)) < 0.2
        return dict(pos=dict(rule_max=good_p * 0.9, rule_mean=good_p * 0.7, rule_rated=np.ones(n)),
                    neg=dict(rule_max=good_n * 0.9, rule_mean=good_n * 0.7, rule_rated=np.ones((n, k))))
    res = sec.s3_analysis(v2, test, {"v2": rules(600, 10), "test": rules(600, 10)}, n_boot=200)
    assert res["rules_beyond_b0"]["ci95"][0] > 0                       # rules carry signal beyond B0
    assert res["holonomy_beyond_rules"]["ci95"][0] < 0.005             # uninformative holonomy adds nothing


def test_build_estimation_on_a_small_graph():
    sec = _secondary()
    from rtvm.g2data import decode
    rng = np.random.default_rng(9)
    train = np.unique(np.stack([rng.integers(0, 40, 300), rng.integers(0, 3, 300), rng.integers(0, 40, 300)], 1), axis=0)
    train = train[train[:, 0] != train[:, 2]]
    d = dict(train=train, num_relations=3, num_entities=40)
    est = sec.build_estimation(d)
    assert est["neg"].shape == (len(est["pos"]), 10, 3) and est["pos_paths"].shape == (len(est["pos"]), 32, 3)
    known = set(map(tuple, train.tolist()))
    assert not any(tuple(t) in known for t in est["neg"].reshape(-1, 3).tolist())     # filtered against training
    for i, e in enumerate(est["edges"]):                                            # a triple never supports itself
        used = {decode(int(c))[0] for c in est["pos_paths"][i].ravel() if c != 0}
        assert int(e) not in used
    again = sec.build_estimation(d)
    assert all(np.array_equal(est[key], again[key]) for key in est)
    tk, tc, seen = sec.rules_from_estimation(est, train, 3)
    assert len(tk) <= seen and ((tc >= 0) & (tc <= 1)).all()


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_rotation_norms_are_zero_at_identity_and_bounded_by_the_rank(rank):
    import torch
    from rtvm.holonomy_model import HolonomyModel
    sec = _secondary()
    still = HolonomyModel(3, 5, n_blocks=2, b=16, rank=rank, rotation_init=0.0)
    assert np.allclose(sec.rotation_norms(still), 0)
    wild = HolonomyModel(3, 50, n_blocks=2, b=16, rank=rank, rotation_init=30.0,
                         generator=torch.Generator().manual_seed(0))
    norms = sec.rotation_norms(wild)
    bound = np.sqrt(8 * 2 * rank)                       # 2 blocks x rank planes, each at most 8 in ||R - I||^2
    assert (norms <= bound * (1 + 1e-5)).all() and norms.max() > 0.5 * bound     # float32 rounding at the bound


def test_reverse_in_training_by_hand():
    sec = _secondary()
    train = np.array([[1, 0, 2], [3, 1, 4], [5, 0, 5]])
    triples = np.array([[2, 0, 1], [1, 0, 2], [4, 1, 3], [4, 0, 3], [5, 0, 5]])
    # reverse of (2,0,1) is (1,0,2): yes; reverse of (1,0,2) is (2,0,1): no; (4,1,3) -> (3,1,4): yes;
    # (4,0,3) -> (3,0,4): no (relation differs); a self-loop is its own reverse
    assert sec.reverse_in_training(triples, train).tolist() == [True, False, True, False, True]


def test_grouped_deltas_with_one_group_is_the_primary_test():
    sec = _secondary()
    rng = np.random.default_rng(10)
    v2, test = synthetic_split(rng, 500, 10, True), synthetic_split(rng, 500, 10, True)
    primary = primary_test(v2, test, n_boot=200)
    mask = rng.random(500) < 0.3
    res = sec.grouped_deltas(v2, test, {"all": np.ones(500, bool), "some": mask, "none": np.zeros(500, bool)},
                             n_boot=200)
    assert res["all"]["delta_bar"] == pytest.approx(primary["delta_bar"], abs=1e-15)
    assert res["all"]["ci95"] == pytest.approx(primary["ci95"], abs=1e-15)
    assert res["some"]["n_positives"] == int(mask.sum()) and "none" not in res


def test_s5_with_rotate_features_is_the_primary_test():
    sec = _secondary()
    rng = np.random.default_rng(11)
    v2, test = synthetic_split(rng, 500, 10, True), synthetic_split(rng, 500, 10, True)
    same = {split: {s: (d["pos"][f"rotate:{s}"], d["neg"][f"rotate:{s}"]) for s in (0, 1, 2)}
            for split, d in (("v2", v2), ("test", test))}
    res = sec.s5_analysis(v2, test, same, n_boot=200)
    primary = primary_test(v2, test, n_boot=200)
    assert "decision" not in res and res["delta_bar"] == primary["delta_bar"] and res["ci95"] == primary["ci95"]


def test_keep_paths_by_hand():
    sec = _secondary()
    codes = np.array([[[3, 0, 0], [1, -2, 0], [4, 5, -6], [0, 0, 0]]])
    assert sec.keep_paths(codes, lambda L: L <= 2).tolist() == [[[3, 0, 0], [1, -2, 0], [0, 0, 0], [0, 0, 0]]]
    assert sec.keep_paths(codes, lambda L: L == 3).tolist() == [[[0, 0, 0], [0, 0, 0], [4, 5, -6], [0, 0, 0]]]


def test_restricting_saved_paths_to_length_two_equals_sampling_with_that_cap():
    # Under the tier rule a sample keeps every path of lengths 1 and 2 unless their tiers overflow,
    # and then holds no length-3 path: so the length <= 2 part always has min(n1 + n2, K) paths.
    from rtvm.g2data import TrainGraph, count_paths, sample_paths
    from rtvm.g2gate import structure_features
    sec = _secondary()
    rng = np.random.default_rng(12)
    hubs = rng.integers(0, 3, 600)                                         # a few hubs make tiers overflow
    train = np.stack([np.where(rng.random(600) < 0.5, hubs, rng.integers(0, 60, 600)), rng.integers(0, 4, 600),
                      rng.integers(0, 60, 600)], 1)
    train = train[train[:, 0] != train[:, 2]]
    graph = TrainGraph.from_training(train, 60)
    triples = np.stack([rng.integers(0, 60, 400), rng.integers(0, 4, 400), rng.integers(0, 60, 400)], 1)
    triples = triples[triples[:, 0] != triples[:, 2]]
    counts, paths = count_paths(graph, triples), sample_paths(graph, triples)
    short = sec.keep_paths(paths, lambda L: L <= 2)
    kept = ((short != 0).sum(-1) > 0).sum(1)
    assert np.array_equal(kept, np.minimum(counts[:, 0] + counts[:, 1], paths.shape[1]))
    assert (kept == paths.shape[1]).any() and (kept < paths.shape[1]).any()          # both regimes occur
    c2 = counts.copy()
    c2[:, 2] = 0
    structure_features(triples, c2, short, graph.degree())                # coverage stays consistent


def test_s4_with_the_gate_structure_is_the_primary_test():
    sec = _secondary()
    rng = np.random.default_rng(13)
    v2, test = synthetic_split(rng, 400, 10, True), synthetic_split(rng, 400, 10, True)
    struct = {f"{part}/{name}": test[part][name] for part in ("pos", "neg") for name in sec.STRUCTURE}
    same = sec.replace_structure(test, struct)
    assert primary_test(v2, same, n_boot=100)["delta_bar"] == primary_test(v2, test, n_boot=100)["delta_bar"]
    other = {key: np.zeros_like(v) for key, v in struct.items()}
    assert sec.replace_structure(test, other)["pos"]["covered"].sum() == 0 and test["pos"]["covered"].any()


def test_known_answers_by_hand():
    sec = _secondary()
    known = np.array([[0, 0, 1], [0, 0, 2], [3, 0, 1], [0, 1, 4]])
    tail = sec.answer_index(known, "tail", 5, 2)
    rows, cols = sec.known_answers(tail, np.array([0 * 2 + 0, 3 * 2 + 0, 4 * 2 + 1]))  # (0,0,?), (3,0,?), (4,1,?)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (0, 2), (1, 1)]
    head = sec.answer_index(known, "head", 5, 2)
    rows, cols = sec.known_answers(head, np.array([1 * 2 + 0, 4 * 2 + 1]))                # (?,0,1), (?,1,4)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (0, 3), (1, 0)]


def test_realistic_ranks_by_hand():
    sec = _secondary()
    scores = np.array([[0.5, 0.9, 0.5, 0.1, 0.7], [0.2, 0.2, 0.2, 0.2, 0.2]])
    # row 0: target 0; entity 1 is a known answer (filtered); 4 scores higher; 2 ties -> 1 + 1 + 0.5
    # row 1: target 3; every other entity ties -> 1 + 0 + 4 / 2
    ranks = sec.realistic_ranks(scores, np.array([0, 3]), np.array([0]), np.array([1]))
    assert ranks.tolist() == [2.5, 3.0]


def test_link_prediction_matches_pykeen_realistic_metrics():
    import torch
    from pykeen.datasets import Nations
    from pykeen.evaluation import RankBasedEvaluator
    from rtvm.baselines import BaselineConfig, build_model
    sec = _secondary()
    ds = Nations()
    net = build_model(BaselineConfig(model="rotate_official", dim=6, seed=3), ds.training)   # untrained: many ties
    net.eval()
    train, valid, test = (x.mapped_triples for x in (ds.training, ds.validation, ds.testing))
    known = torch.cat([train, valid, test]).numpy()
    tails, heads = sec.baseline_scorers(net)
    ours = sec.link_prediction(tails, heads, test.numpy(), known, ds.num_entities, ds.num_relations, batch=37)
    res = RankBasedEvaluator().evaluate(net, test, additional_filter_triples=[train, valid], use_tqdm=False,
                                        batch_size=64)
    assert ours["mrr"] == pytest.approx(res.get_metric("both.realistic.inverse_harmonic_mean_rank"), abs=1e-6)
    assert ours["hits_at_1"] == pytest.approx(res.get_metric("both.realistic.hits_at_1"), abs=1e-6)
    assert ours["hits_at_10"] == pytest.approx(res.get_metric("both.realistic.hits_at_10"), abs=1e-6)
    assert ours["n_ranks"] == 2 * len(test)


def test_quate_scorers_match_pykeen():
    import torch
    from pykeen.datasets import Nations
    from rtvm.baselines import BaselineConfig, build_model
    sec = _secondary()
    ds = Nations()
    net = build_model(BaselineConfig(model="quate_official", dim=5, regularizer_weight=0.1, seed=2), ds.training)
    net.eval()
    tails, heads = sec.quate_scorers(net)
    h, r, t = np.array([0, 4, 9]), np.array([1, 7, 30]), np.array([2, 11, 13])
    with torch.no_grad():
        want_t = net.score_t(torch.as_tensor(np.stack([h, r], 1))).double()
        want_h = net.score_h(torch.as_tensor(np.stack([r, t], 1))).double()
    assert torch.allclose(tails(h, r), want_t, atol=1e-5)
    assert torch.allclose(heads(r, t), want_h, atol=1e-5)


def test_holonomy_scorers_are_the_single_triple_score():
    import torch
    from rtvm.holonomy_model import HolonomyModel
    sec = _secondary()
    model = HolonomyModel(9, 3, n_blocks=2, b=4, rank=1, rotation_init=0.5, generator=torch.Generator().manual_seed(4))
    tails, heads = sec.holonomy_scorers(model)
    h, r, t = np.array([1, 5]), np.array([2, 0]), np.array([7, 3])
    every = torch.arange(9)
    with torch.no_grad():
        want_t = torch.stack([-model.single_distance(torch.full((9,), int(a)), torch.full((9,), int(b)), every)
                              for a, b in zip(h, r)])
        want_h = torch.stack([-model.single_distance(every, torch.full((9,), int(b)), torch.full((9,), int(c)))
                              for b, c in zip(r, t)])
    assert torch.allclose(tails(h, r), want_t.double(), atol=1e-5)
    assert torch.allclose(heads(r, t), want_h.double(), atol=1e-5)


def test_fallback_mask_changes_only_what_it_should():
    sec = _secondary()
    rng = np.random.default_rng(4)
    v2 = _with_coverage(synthetic_split(rng, 600, 10, False), rng)
    test = _with_coverage(synthetic_split(rng, 600, 10, False), rng)
    fallback = rng.random((600, 10)) < 0.1
    fallback[:3] = True                                       # positives left with no negative are skipped
    res = sec.basic_analyses(v2, test, fallback, n_boot=200, pooled_boot=20)
    assert res["s9_fallback_excluded"]["n_positives"] == 597
    assert res["s9_fallback_excluded"]["n_fallback_removed"] == int(fallback.sum())
