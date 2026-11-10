"""Spec for rtvm/g2eval.py: grouped AUC (protocol, "Primary test")."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rtvm.g2eval import grouped_auc

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


def test_masks():
    pos = np.array([1.0, 1.0, 1.0])
    neg = np.array([[0.0, 2.0], [2.0, 2.0], [0.0, 0.0]])
    neg_valid = np.array([[True, False], [False, False], [True, True]])
    pos_valid = np.array([True, True, False])
    # positive 0: only its first negative counts (1.0); positive 1 has none; positive 2 is invalid
    assert grouped_auc(pos, neg, pos_valid, neg_valid) == 1.0
    assert np.isnan(grouped_auc(pos, neg, np.zeros(3, bool), neg_valid))
