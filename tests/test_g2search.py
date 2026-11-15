"""Tests for the G2 random search (rtvm/g2search.py), the three configs and the baseline
trainer (rtvm/baselines.py). CPU, seconds; no dataset download."""
from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import pytest

from rtvm.g2search import (BUDGET, CONFIGS, N_TRIALS, check_budget, fixed_values, load_spec, sample_trials,
                           select_best)

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

@pytest.mark.parametrize("dataset", sorted(BUDGET))
@pytest.mark.parametrize("model", sorted(CONFIGS))
def test_config_values_match_the_trainer(model, dataset):
    from rtvm.baselines import BaselineConfig
    from rtvm.train_holonomy import HolonomyConfig

    spec = load_spec(ROOT / CONFIGS[model])
    cls = HolonomyConfig if model == "holonomy" else BaselineConfig
    fields = {f.name for f in dataclasses.fields(cls)}
    trial = sample_trials(spec["search"], 1, 0)[0]
    fixed = fixed_values(spec, dataset)
    assert set(fixed) <= fields and set(trial) <= fields
    assert not set(fixed) & set(trial)                    # a value is fixed or searched, never both
    if model != "holonomy":
        cls(model=spec["model"], dim=spec["dim"][dataset], **fixed, **trial)
    else:
        cls(**fixed, **trial)
    assert fixed["eval_every"] == 10 and fixed["patience"] == 5
    assert fixed["max_epochs"] == 1000                    # shared protocol rules
    assert fixed["batch_size"] == 1024 and fixed["tf32"] is True


def test_quate_config_follows_the_2026_11_16_note():
    spec = load_spec(ROOT / CONFIGS["quate"])
    assert spec["model"] == "quate_official"
    assert fixed_values(spec, "fb15k237")["num_negatives"] == 10 and fixed_values(spec, "wn18rr")["num_negatives"] == 1
    fixed = fixed_values(spec, "fb15k237")
    assert fixed["optimizer"] == "adagrad" and fixed["filtered_negatives"] is True and fixed["loss"] == "bcewithlogits"
    assert spec["search"]["lr"] == {"log_uniform": [0.02, 0.5]}
    assert spec["search"]["regularizer_weight"] == {"log_uniform": [1.0e-3, 1.0]}


def test_fixed_values_resolves_per_dataset_entries():
    spec = {"fixed": {"a": 1, "b": {"fb15k237": 10, "wn18rr": 1}}}
    assert fixed_values(spec, "fb15k237") == {"a": 1, "b": 10}
    assert fixed_values(spec, "wn18rr") == {"a": 1, "b": 1}
    with pytest.raises(ValueError):
        fixed_values(spec, "nations")


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
        cfg = BaselineConfig(model=spec["model"], dim=spec["dim"][dataset], **fixed_values(spec, dataset),
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


@pytest.mark.parametrize("model", ["rotate_official", "quate_pykeen", "quate_official"])
def test_baseline_trains_evaluates_and_is_deterministic(data, model):
    from rtvm.baselines import train
    a = train(small(model), data, log=lambda *_: None)
    b = train(small(model), data, log=lambda *_: None)
    assert a["epochs_run"] == 6 and len(a["history"]) == 3 and a["complete"]
    assert all(0 <= h["v1_auc"] <= 1 and np.isfinite(h["loss"]) for h in a["history"])
    assert a["history"] == b["history"]


def test_baseline_optimizer_is_the_configured_one(data, tmp_path):
    import torch

    from rtvm.baselines import train
    for name, state_key in (("adam", "exp_avg"), ("adagrad", "sum")):
        ckpt = tmp_path / f"{name}.ckpt"
        out = train(small("quate_official", optimizer=name, filtered_negatives=True, lr=0.1), data,
                    checkpoint=ckpt, log=lambda *_: None)
        assert out["complete"] and all(np.isfinite(h["loss"]) for h in out["history"])
        state = torch.load(ckpt, weights_only=False)["optimizer"]["state"]
        assert state and all(state_key in s for s in state.values())
    with pytest.raises(ValueError):
        train(small(optimizer="sgd"), data, log=lambda *_: None)


def test_quate_official_adagrad_resumes_exactly(data, tmp_path):
    """The G2 QuatE setting (Adagrad, exact-filtered Bernoulli negatives) resumes like RotatE."""
    from rtvm.baselines import train
    cfg = small("quate_official", optimizer="adagrad", filtered_negatives=True, lr=0.1)
    full = train(cfg, data, log=lambda *_: None)
    ckpt = tmp_path / "q.ckpt"
    part = train(cfg, data, checkpoint=ckpt, log=lambda *_: None, deadline=time.time() - 1)
    assert not part["complete"] and part["epochs_run"] == 2
    msgs = []
    rest = train(cfg, data, checkpoint=ckpt, log=msgs.append)
    assert msgs[0] == "resumed at epoch 2" and rest["complete"] and rest["history"] == full["history"]


def _cuda_available():
    import torch
    return torch.cuda.is_available()


# The cuda case skips on a CPU machine; the Kaggle notebook runs it as a preflight before the search
# (2026-11-11: resuming on cuda crashed in session 2 because the RNG state was loaded onto the GPU).
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not _cuda_available(), reason="needs a GPU"))])
def test_baseline_resume_and_deadline(data, tmp_path, device):
    from rtvm.baselines import train
    full = train(small(), data, device=device, log=lambda *_: None)
    ckpt = tmp_path / "run.ckpt"
    part = train(small(), data, device=device, checkpoint=ckpt, log=lambda *_: None, deadline=time.time() - 1)
    assert not part["complete"] and part["epochs_run"] == 2   # stops after the first evaluation
    msgs = []
    rest = train(small(), data, device=device, checkpoint=ckpt, log=msgs.append)
    assert msgs[0] == "resumed at epoch 2" and rest["complete"]
    if device == "cpu":
        assert rest["history"] == full["history"]             # resumed run matches the uninterrupted one
    else:                                                      # CUDA scatter-add backward is not bitwise deterministic
        assert len(rest["history"]) == len(full["history"])


# The cuda case is the holonomy search notebook's preflight (it also runs rtvm's cayley on the GPU,
# which crashed in MAGMA on Kaggle T4s before d91ab20).
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not _cuda_available(), reason="needs a GPU"))])
def test_holonomy_deadline_stops_after_a_checkpoint(tmp_path, device):
    from test_train_holonomy import random_steps        # not tests.*: an installed "tests" package can shadow ours (Kaggle, 2026-11-15)
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
    full = train(cfg, d, device=device, log=lambda *_: None)
    ckpt = tmp_path / "h.ckpt"
    part = train(cfg, d, device=device, checkpoint=ckpt, log=lambda *_: None, deadline=time.time() - 1)
    assert not part["complete"] and part["epochs_run"] == 2
    msgs = []
    rest = train(cfg, d, device=device, checkpoint=ckpt, log=msgs.append)
    assert msgs[0] == "resumed at epoch 2" and rest["complete"]
    if device == "cpu":
        assert rest["history"] == full["history"]             # resumed run matches the uninterrupted one
    else:                                                      # bitwise equality is not promised on CUDA
        assert [h["epoch"] for h in rest["history"]] == [h["epoch"] for h in full["history"]]


def test_config_hash_ignores_line_endings(tmp_path):
    from rtvm.g2search import file_sha256
    (tmp_path / "lf.yaml").write_bytes(b"a: 1\nb: 2\n")
    (tmp_path / "crlf.yaml").write_bytes(b"a: 1\r\nb: 2\r\n")
    assert file_sha256(tmp_path / "lf.yaml") == file_sha256(tmp_path / "crlf.yaml")


def test_exact_filterer_matches_python_set_filterer():
    """Protocol note 2026-11-10 (exact filtering): same answer as PyKEEN's exact set filter."""
    import torch
    from pykeen.sampling.filtering import PythonSetFilterer

    from rtvm.baselines import ExactFilterer

    gen = torch.Generator().manual_seed(0)
    triples = torch.unique(torch.stack([torch.randint(0, 40, (500,), generator=gen),
                                        torch.randint(0, 5, (500,), generator=gen),
                                        torch.randint(0, 40, (500,), generator=gen)], dim=1), dim=0)
    exact, ref = ExactFilterer(triples), PythonSetFilterer(triples)
    cand = torch.stack([torch.randint(0, 42, (64, 30), generator=gen),     # ids up to 41: beyond the training max
                        torch.randint(0, 6, (64, 30), generator=gen),
                        torch.randint(0, 42, (64, 30), generator=gen)], dim=-1)
    cand[:, :5] = triples[torch.randint(0, len(triples), (64, 5), generator=gen)]   # some true triples
    assert torch.equal(exact.contains(cand), ref.contains(cand))
    assert exact.contains(cand).sum() >= 64 * 5
    assert torch.equal(exact(cand), ~ref.contains(cand))                   # forward: True = valid negative


def test_rotate_uses_the_exact_filterer(data, monkeypatch):
    """Filtered training negatives go through ExactFilterer (not PyKEEN's default Bloom filter)."""
    from rtvm import baselines

    calls = []
    original = baselines.ExactFilterer.contains
    monkeypatch.setattr(baselines.ExactFilterer, "contains",
                        lambda self, batch: calls.append(1) or original(self, batch))
    assert baselines.train(small(max_epochs=2), data, log=lambda *_: None)["complete"]
    assert calls                                       # rotate_official: filtered, through ExactFilterer
    calls.clear()
    baselines.train(small("quate_pykeen", max_epochs=2), data, log=lambda *_: None)
    assert not calls                                   # QuatE: unfiltered
