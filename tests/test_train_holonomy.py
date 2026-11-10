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
    assert a["history"] == b["history"]                # exact: deterministic algorithms are on
    assert a["deterministic_algorithms"]


class Interrupt(Exception):
    pass


def test_resume_after_interruption_matches_uninterrupted_run(data, tmp_path):
    full = train(small_config(max_epochs=6), data, log=lambda *_: None)
    ckpt = tmp_path / "run.pt"

    def stop_after_epoch_2(msg):
        if msg.startswith("epoch 2:"):
            raise Interrupt                             # simulated disconnect, after the checkpoint is saved

    with pytest.raises(Interrupt):
        train(small_config(max_epochs=6), data, checkpoint=ckpt, log=stop_after_epoch_2)
    msgs = []
    resumed = train(small_config(max_epochs=6), data, checkpoint=ckpt, log=msgs.append)
    assert msgs[0] == "resumed at epoch 2"
    assert resumed["history"] == full["history"] and resumed["best_epoch"] == full["best_epoch"]


def test_evaluate_v1_ignores_negatives_of_uncovered_positives(data):
    """Protocol note 2026-11-10: cycles of covered positives and of THEIR covered negatives."""
    import copy

    import torch

    from rtvm.holonomy_model import HolonomyModel
    from rtvm.train_holonomy import evaluate_v1

    model = HolonomyModel(E, R, rank=2, generator=torch.Generator().manual_seed(0))
    d = copy.deepcopy(data)
    d.v1_pos_steps[0] = 2 * R                           # positive 0 now has no path
    d.v1_neg_steps[0, :, 0, 0] = 1                      # ...but its negatives all have one
    blanked = copy.deepcopy(d)
    blanked.v1_neg_steps[0] = 2 * R                     # same, with those negatives' cycles removed
    assert evaluate_v1(model, d, "cpu")["stratified"] == evaluate_v1(model, blanked, "cpu")["stratified"]


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
