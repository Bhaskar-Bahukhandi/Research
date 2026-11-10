"""G2 evaluation: grouped AUC (protocol, "Primary test").

Exercise: ``grouped_auc``. Spec: tests/test_g2eval.py (`pytest -m g2model -rs`).
"""
from __future__ import annotations

import numpy as np


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
    raise NotImplementedError("grouped_auc")
