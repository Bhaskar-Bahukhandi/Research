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
    # one sort by stratum, then one rank computation per stratum slice (no full pass per stratum)
    length = np.concatenate([np.asarray(pos_length), np.asarray(neg_length)]).astype(np.int64)
    relation = np.concatenate([np.asarray(pos_relation), np.asarray(neg_relation)]).astype(np.int64)
    score = -np.concatenate([pos_defect, neg_defect])          # higher = more plausible
    is_pos = np.r_[np.ones(len(pos_defect), bool), np.zeros(len(neg_defect), bool)]
    order = np.lexsort((relation, length))
    length, relation, score, is_pos = length[order], relation[order], score[order], is_pos[order]
    change = np.flatnonzero((np.diff(length) != 0) | (np.diff(relation) != 0)) + 1
    total, weight = 0.0, 0
    for lo, hi in zip(np.r_[0, change], np.r_[change, len(score)]):
        p_mask = is_pos[lo:hi]
        n_pos = int(p_mask.sum())
        n_neg = (hi - lo) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(score[lo:hi])
        auc = (ranks[p_mask].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        total, weight = total + auc * n_pos, weight + n_pos
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
