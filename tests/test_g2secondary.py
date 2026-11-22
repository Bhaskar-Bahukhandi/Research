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
