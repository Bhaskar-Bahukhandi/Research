"""G2 evaluation: grouped AUC (protocol, "Primary test") and the holonomy model's
tuning metric (protocol note of 2026-11-10). Spec: tests/test_g2eval.py.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def stratified_cycle_auc(pos_defect: np.ndarray, pos_length: np.ndarray, pos_relation: np.ndarray,
                         neg_defect: np.ndarray, neg_length: np.ndarray, neg_relation: np.ndarray) -> float:
    """Protocol note of 2026-11-10: the holonomy model's V1 tuning and early-stopping metric.

    Inputs are one entry per saved cycle: its defect ||H_c - I||_F, its path length and the
    relation of the triple it closes, for covered V1 positives (pos_*) and for their
    covered negatives (neg_*). Within each stratum (length, relation): AUC of positive
    cycles against negative cycles, where a LOWER defect is more plausible and ties count
    1/2. Strata are averaged with weights equal to their number of positive cycles; a
    stratum with no positive or no negative cycle is skipped. NaN if none is left, or if
    any defect is NaN.
    """
    pos_defect, neg_defect = np.asarray(pos_defect, float), np.asarray(neg_defect, float)
    if np.isnan(pos_defect).any() or np.isnan(neg_defect).any():
        return float("nan")
    total, weight = 0.0, 0
    pos_key = np.stack([np.asarray(pos_length), np.asarray(pos_relation)], axis=1)
    neg_key = np.stack([np.asarray(neg_length), np.asarray(neg_relation)], axis=1)
    for key in np.unique(pos_key, axis=0):
        p = pos_defect[(pos_key == key).all(axis=1)]
        n = neg_defect[(neg_key == key).all(axis=1)]
        if not len(n):
            continue
        ranks = rankdata(-np.concatenate([p, n]))           # higher rank = more plausible
        auc = (ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))
        total, weight = total + auc * len(p), weight + len(p)
    return total / weight if weight else float("nan")


def grouped_auc(pos: np.ndarray, neg: np.ndarray, pos_valid: np.ndarray | None = None,
                neg_valid: np.ndarray | None = None) -> float:
    """EXERCISE. Protocol: "For each positive: the fraction of its 10 negatives scored
    below it, with ties counting 1/2. Then the mean over positives."

    pos: (n,) plausibility scores (higher = more plausible; pass -holonomy for the
    holonomy score). neg: (n, k) the scores of each positive's own negatives.
    pos_valid (n,) and neg_valid (n, k) are optional masks (default: all True). Only
    valid negatives count; a positive that is not valid, or has no valid negative, is
    skipped (the V1 tuning rule for the holonomy model). Returns NaN if every positive
    is skipped.

    Vectorise it: no Python loop over positives.
    """
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pv = np.ones(pos.shape, bool) if pos_valid is None else np.asarray(pos_valid, bool)
    nv = np.ones(neg.shape, bool) if neg_valid is None else np.asarray(neg_valid, bool)
    wins = ((neg < pos[:, None]) + 0.5 * (neg == pos[:, None])) * nv
    n_neg = nv.sum(axis=1)
    keep = pv & (n_neg > 0)
    if not keep.any():
        return float("nan")
    if np.isnan(pos[keep]).any() or np.isnan(neg[keep][nv[keep]]).any():
        return float("nan")                                  # a diverged model must not score 0 silently
    return float(np.mean(wins.sum(axis=1)[keep] / n_neg[keep]))
