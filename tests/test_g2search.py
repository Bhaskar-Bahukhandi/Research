"""Tests for the G2 random search (rtvm/g2search.py), the three configs and the baseline
trainer (rtvm/baselines.py). CPU, seconds; no dataset download."""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import pytest

from rtvm.g2search import BUDGET, CONFIGS, N_TRIALS, check_budget, load_spec, sample_trials, select_best

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"fb15k237": (14_505, 237), "wn18rr": (40_559, 11)}


# ---------------------------------------------------------------- sampling and selection

def test_sampling_is_deterministic_and_in_range():
    search = {"a": {"choice": [1, 2, 4]}, "b": {"log_uniform": [1e-4, 3e-3]}, "c": {"uniform": [0.5, 4.0]}}
    t1, t2 = sample_trials(search, 20, 0), sample_trials(search, 20, 0)
    assert t1 == t2 and len(t1) == 20
    assert all(t["a"] in (1, 2, 4) and 1e-4 <= t["b"] <= 3e-3 and 0.5 <= t["c"] <= 4.0 for t in t1)
    assert sample_trials(search, 5, 0) == t1[:5]          # trial i does not depend on later trials
    assert sample_trials(search, 20, 1) != t1
    assert len({t["a"] for t in t1}) == 3                 # every choice appears in 20 draws (seed 0)


def test_sampling_rejects_bad_specs():
    with pytest.raises(ValueError):
        sample_trials({"a": {"normal": [0, 1]}})
    with pytest.raises(ValueError):
        sample_trials({"a": {"log_uniform": [0.0, 1.0]}})


def test_select_best():
    res = {i: dict(complete=True, best_v1_auc=0.5 + 0.01 * (i % 7)) for i in range(N_TRIALS)}
    assert select_best(res) == 6                          # 0.56 at 6 and 13: ties go to the lower index
    res[3]["best_v1_auc"] = float("nan")
    assert select_best(res) == 6
    res[19]["complete"] = False
    with pytest.raises(ValueError):
        select_best(res)


# ---------------------------------------------------------------- the committed configs

@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_config_values_match_the_trainer(model):
    from rtvm.baselines import BaselineConfig
    from rtvm.train_holonomy import HolonomyConfig

    spec = load_spec(ROOT / CONFIGS[model])
    cls = HolonomyConfig if model == "holonomy" else BaselineConfig
    fields = {f.name for f in dataclasses.fields(cls)}
    trial = sample_trials(spec["search"], 1, 0)[0]
    assert set(spec["fixed"]) <= fields and set(trial) <= fields
    assert not set(spec["fixed"]) & set(trial)            # a value is fixed or searched, never both
    if model != "holonomy":
        cls(model=spec["model"], dim=spec["dim"]["fb15k237"], **spec["fixed"], **trial)
    else:
        cls(**spec["fixed"], **trial)
    assert spec["fixed"]["eval_every"] == 10 and spec["fixed"]["patience"] == 5
    assert spec["fixed"]["max_epochs"] == 1000            # shared protocol rules
    assert spec["fixed"]["batch_size"] == 1024


def test_margin_range_is_attainable_for_every_rank():
    """A rank-r Cayley rotation pair gives ||H - I||_F <= sqrt(16 r (L + 1)) (2 blocks)."""
    spec = load_spec(ROOT / CONFIGS["holonomy"])
    assert spec["search"]["margin"]["uniform"][1] < math.sqrt(16 * 1 * 2)


@pytest.mark.parametrize("dataset", sorted(BUDGET))
def test_every_model_fits_the_budget(dataset):
    import torch
    from pykeen.triples import CoreTriplesFactory

    from rtvm.baselines import BaselineConfig, build_model
    from rtvm.holonomy_model import HolonomyModel
    from rtvm.utils import count_parameters

    E, R = SIZES[dataset]
    for rank in (1, 2, 4):
        check_budget(count_parameters(HolonomyModel(E, R, rank=rank)), dataset)
    tf = CoreTriplesFactory.create(torch.tensor([[0, 0, 1], [1, R - 1, 2]]), num_entities=E, num_relations=R)
    for model in ("rotate", "quate"):
        spec = load_spec(ROOT / CONFIGS[model])
        cfg = BaselineConfig(model=spec["model"], dim=spec["dim"][dataset], **spec["fixed"],
                             **sample_trials(spec["search"], 1, 0)[0])
        check_budget(count_parameters(build_model(cfg, tf)), dataset)


# ---------------------------------------------------------------- baseline trainer

E, R = 30, 4


@pytest.fixture
def data():
    import torch
    from pykeen.triples import CoreTriplesFactory

    from rtvm.baselines import BaselineData

    rng = np.random.default_rng(0)
    tr = np.unique(np.c_[rng.integers(E, size=150), rng.integers(R, size=150), rng.integers(E, size=150)], axis=0)
    tf = CoreTriplesFactory.create(torch.as_tensor(tr), num_entities=E, num_relations=R)
    pos = np.c_[rng.integers(E, size=12), rng.integers(R, size=12), rng.integers(E, size=12)]
    neg = np.repeat(pos[:, None], 10, axis=1)
    neg[:, :5, 0] = rng.integers(E, size=(12, 5))
    neg[:, 5:, 2] = rng.integers(E, size=(12, 5))
    return BaselineData(training=tf, v1_pos=pos, v1_neg=neg)


def small(model="rotate_official", **kw):
    from rtvm.baselines import BaselineConfig
    base = dict(model=model, dim=8, num_negatives=4, batch_size=32, eval_every=2, max_epochs=6, tf32=False,
                filtered_negatives=model == "rotate_official", loss="nssa" if model == "rotate_official" else "bcewithlogits")
    return BaselineConfig(**{**base, **kw})


@pytest.mark.parametrize("model", ["rotate_official", "quate_pykeen"])
def test_baseline_trains_evaluates_and_is_deterministic(data, model):
    from rtvm.baselines import train
    a = train(small(model), data, log=lambda *_: None)
    b = train(small(model), data, log=lambda *_: None)
    assert a["epochs_run"] == 6 and len(a["history"]) == 3 and a["complete"]
    assert all(0 <= h["v1_auc"] <= 1 and np.isfinite(h["loss"]) for h in a["history"])
    assert a["history"] == b["history"]


def test_baseline_resume_and_deadline(data, tmp_path):
    from rtvm.baselines import train
    full = train(small(), data, log=lambda *_: None)
    ckpt = tmp_path / "run.ckpt"
    part = train(small(), data, checkpoint=ckpt, log=lambda *_: None, deadline=time.time() - 1)
    assert not part["complete"] and part["epochs_run"] == 2   # stops after the first evaluation
    msgs = []
    rest = train(small(), data, checkpoint=ckpt, log=msgs.append)
    assert msgs[0] == "resumed at epoch 2" and rest["complete"]
    assert rest["history"] == full["history"]                 # resumed run matches the uninterrupted one


def test_holonomy_deadline_stops_after_a_checkpoint(tmp_path):
    from tests.test_train_holonomy import random_steps
    from rtvm.train_holonomy import HolonomyConfig, HolonomyData, train

    rng = np.random.default_rng(0)
    E2, R2 = 30, 4
    tr = np.unique(np.c_[rng.integers(E2, size=120), rng.integers(R2, size=120), rng.integers(E2, size=120)], axis=0)
    pos = np.c_[rng.integers(E2, size=12), rng.integers(R2, size=12), rng.integers(E2, size=12)]
    neg = np.repeat(pos[:, None], 10, axis=1)
    neg[:, :, 0] = rng.integers(E2, size=(12, 10))
    d = HolonomyData(E2, R2, tr, random_steps(rng, (len(tr),)), pos, random_steps(rng, (12,)),
                     neg, random_steps(rng, (12, 10)))
    cfg = HolonomyConfig(num_negatives=8, batch_size=32, eval_every=2, max_epochs=6, tf32=False)
    full = train(cfg, d, log=lambda *_: None)
    ckpt = tmp_path / "h.ckpt"
    part = train(cfg, d, checkpoint=ckpt, log=lambda *_: None, deadline=time.time() - 1)
    assert not part["complete"] and part["epochs_run"] == 2
    rest = train(cfg, d, checkpoint=ckpt, log=lambda *_: None)
    assert rest["complete"] and rest["history"] == full["history"]
