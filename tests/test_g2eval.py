"""Spec for rtvm/g2eval.py: grouped AUC (protocol, "Primary test")."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rtvm.g2eval import grouped_auc, stratified_cycle_auc

pytestmark = pytest.mark.g2model


def test_perfect_reversed_and_tied():
    pos = np.array([1.0, 2.0])
    assert grouped_auc(pos, np.zeros((2, 10))) == 1.0
    assert grouped_auc(-pos, np.zeros((2, 10))) == 0.0
    assert grouped_auc(np.zeros(2), np.zeros((2, 10))) == 0.5


def test_mean_over_positives_not_pooled():
    pos = np.array([0.5, 10.0])
    neg = np.array([[0.0, 1.0], [11.0, 12.0]])    # positive 0 beats 1 of 2; positive 1 beats 0 of 2
    assert grouped_auc(pos, neg) == pytest.approx((0.5 + 0.0) / 2)


def test_matches_per_group_sklearn_with_ties():
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 5, size=200).astype(float)
    neg = rng.integers(0, 5, size=(200, 10)).astype(float)
    expected = np.mean([roc_auc_score([1] + [0] * 10, np.r_[p, n]) for p, n in zip(pos, neg)])
    assert grouped_auc(pos, neg) == pytest.approx(expected)


def test_nan_scores_give_nan_not_zero():
    assert np.isnan(grouped_auc(np.array([np.nan, 1.0]), np.zeros((2, 3))))
    neg = np.zeros((2, 3))
    neg[1, 2] = np.nan
    assert np.isnan(grouped_auc(np.ones(2), neg))


def test_masks():
    pos = np.array([1.0, 1.0, 1.0])
    neg = np.array([[0.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    neg_valid = np.array([[True, False], [False, False], [True, True]])
    pos_valid = np.array([True, True, False])
    # positive 0: only its first negative counts (1.0); positive 1 has none; positive 2 is invalid
    assert grouped_auc(pos, neg, pos_valid, neg_valid) == 1.0
    assert np.isnan(grouped_auc(pos, neg, np.zeros(3, bool), neg_valid))


# ---------------------------------------------------------------- stratified cycle AUC (note 2026-11-10)

def strata_data(seed=0, n=400):
    rng = np.random.default_rng(seed)
    pl, pr = rng.integers(1, 4, n), rng.integers(0, 3, n)
    nl, nr = rng.integers(1, 4, 4 * n), rng.integers(0, 3, 4 * n)
    pd, nd = rng.integers(0, 6, n).astype(float), rng.integers(0, 6, 4 * n).astype(float)
    return pd, pl, pr, nd, nl, nr


def test_stratified_matches_weighted_per_stratum_sklearn():
    pd, pl, pr, nd, nl, nr = strata_data()
    total, weight = 0.0, 0
    for L in (1, 2, 3):
        for r in range(3):
            p, n = pd[(pl == L) & (pr == r)], nd[(nl == L) & (nr == r)]
            auc = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], -np.r_[p, n])
            total, weight = total + auc * len(p), weight + len(p)
    assert stratified_cycle_auc(pd, pl, pr, nd, nl, nr) == pytest.approx(total / weight)


def test_stratified_ignores_length_and_relation_offsets():
    """The reason for the note: a defect level that depends only on length or relation is not information."""
    pd, pl, pr, nd, nl, nr = strata_data(1)
    base = stratified_cycle_auc(pd, pl, pr, nd, nl, nr)
    shift = lambda d, L, r: d + 10.0 * L + 3.0 * r          # noqa: E731
    assert stratified_cycle_auc(shift(pd, pl, pr), pl, pr, shift(nd, nl, nr), nl, nr) == pytest.approx(base)


def test_stratified_skips_strata_without_negatives_and_handles_nan():
    pd, nd = np.array([0.0, 5.0]), np.array([1.0])
    # stratum (1, 0) has a negative; stratum (2, 0) has none and is skipped
    assert stratified_cycle_auc(pd, np.array([1, 2]), np.array([0, 0]), nd, np.array([1]), np.array([0])) == 1.0
    assert np.isnan(stratified_cycle_auc(pd, np.array([2, 2]), np.array([0, 0]), nd, np.array([1]), np.array([0])))
    assert np.isnan(stratified_cycle_auc(np.array([np.nan]), np.array([1]), np.array([0]), nd, np.array([1]), np.array([0])))
