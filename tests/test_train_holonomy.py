"""Smoke tests for rtvm/train_holonomy.py on a tiny random dataset (CPU, seconds).

They need your exercises (path_transport, cycle_loss, grouped_auc), so they skip until
those are implemented.
"""
from __future__ import annotations

import numpy as np
import pytest

from rtvm.g2data import MAX_LEN
from rtvm.train_holonomy import HolonomyConfig, HolonomyData, train

pytestmark = pytest.mark.g2model

E, R, K = 30, 4, 6


def random_steps(rng, shape, fill=0.6):
    """Random table steps; a slot is empty with probability 1 - fill, filled slots come first."""
    pad = 2 * R
    steps = np.full((*shape, K, MAX_LEN), pad)
    n_filled = (rng.random(shape) * (K + 1) * fill).astype(int)
    for idx in np.ndindex(*shape):
        for j in range(n_filled[idx]):
            L = rng.integers(1, MAX_LEN + 1)
            steps[idx + (j,)][:L] = rng.integers(0, 2 * R, size=L)
    return steps


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    train_ = np.unique(np.c_[rng.integers(E, size=120), rng.integers(R, size=120), rng.integers(E, size=120)], axis=0)
    pos = np.c_[rng.integers(E, size=12), rng.integers(R, size=12), rng.integers(E, size=12)]
    neg = np.repeat(pos[:, None], 10, axis=1)
    neg[:, :, 0] = rng.integers(E, size=(12, 10))
    return HolonomyData(E, R, train_, random_steps(rng, (len(train_),)), pos, random_steps(rng, (12,)),
                        neg, random_steps(rng, (12, 10)))


def small_config(**kw):
    base = dict(num_negatives=8, batch_size=32, eval_every=2, max_epochs=4, tf32=False, seed=0)
    return HolonomyConfig(**{**base, **kw})


def test_training_runs_and_evaluates(data):
    out = train(small_config(), data, log=lambda *_: None)
    assert out["epochs_run"] == 4 and len(out["history"]) == 2
    assert all(np.isfinite(h["loss"]) and 0 <= h["v1_auc"] <= 1 for h in out["history"])
    assert out["evaluation"] == "strict FP32"


def test_training_is_deterministic(data):
    a = train(small_config(), data, log=lambda *_: None)
    b = train(small_config(), data, log=lambda *_: None)
    # same draws and same AUC; the loss may differ in the last bits (multithreaded float sums)
    assert [h["v1_auc"] for h in a["history"]] == [h["v1_auc"] for h in b["history"]]
    assert [h["loss"] for h in a["history"]] == pytest.approx([h["loss"] for h in b["history"]], rel=1e-5)


def test_early_stopping_counts_evaluations(data):
    out = train(small_config(max_epochs=40, eval_every=1, patience=2, lr=0.0), data, log=lambda *_: None)
    assert out["epochs_run"] == 3                   # lr 0: AUC never improves after the first evaluation


def test_checkpoint_resume(data, tmp_path):
    ckpt = tmp_path / "run.pt"
    first = train(small_config(), data, checkpoint=ckpt, log=lambda *_: None)
    msgs = []
    again = train(small_config(), data, checkpoint=ckpt, log=msgs.append)
    assert any("resumed at epoch 4" in m for m in msgs)
    assert again["best_v1_auc"] == first["best_v1_auc"] and again["epochs_run"] == 4
    with pytest.raises(ValueError):
        train(small_config(lr=0.5), data, checkpoint=ckpt, log=lambda *_: None)
