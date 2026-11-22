"""Tests for experiments/g2_s10.py and the s10 command of experiments/g2_secondary.py (S10, dense rotations)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    sys.path.insert(0, str(ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dense_values_are_trial_4_at_rank_32_with_3_80m_parameters():
    from rtvm.holonomy_model import HolonomyModel
    from rtvm.utils import count_parameters
    s10 = _module("g2_s10")
    best, values = s10.dense_values()
    best_again, original = s10.best_trial_values()
    assert best == best_again == 4                       # the committed search's best trial (protocol, first note of 2026-11-21)
    assert values == {**original, "rank": 32} and original["rank"] == 2
    n = count_parameters(HolonomyModel(14505, 237, rank=values["rank"]))
    assert n == 14505 * 128 + 2 * 237 * 2 * 64 * 32 == 3_798_144                     # "3.80M" (note of 2026-11-06)


def test_the_trainer_accepts_the_s10_config():
    from rtvm import train_holonomy as trainer
    from rtvm.g2search import CONFIGS, fixed_values, load_spec
    s10 = _module("g2_s10")
    _, values = s10.dense_values()
    cfg = trainer.HolonomyConfig(**{**fixed_values(load_spec(ROOT / CONFIGS["holonomy"]), "fb15k237"), **values,
                                    "seed": 1})
    assert cfg.rank == 32 and cfg.seed == 1 and cfg.max_epochs == 1000 and cfg.patience == 5


def test_dense_runs_checks_each_record(tmp_path):
    import torch
    from rtvm.g2search import CONFIGS, file_sha256
    from rtvm.holonomy_model import HolonomyModel
    sec = _module("g2_secondary")
    s10 = _module("g2_s10")
    best, values = s10.dense_values()
    run_dir = tmp_path / "fb15k237" / "holonomy_dense"
    run_dir.mkdir(parents=True)

    def write(s, **changes):
        net = HolonomyModel(5, 3, rank=32, generator=torch.Generator().manual_seed(s))
        torch.save(net.state_dict(), run_dir / f"seed_{s}_best.pt")
        rec = dict(complete=True, seed=s, values=values, best_trial=best, parameters=1,
                   config_sha256=file_sha256(ROOT / CONFIGS["holonomy"]), best_v1_auc=0.5, best_epoch=10, epochs_run=60)
        (run_dir / f"seed_{s}.json").write_text(json.dumps({**rec, **changes}))
    for s in (0, 1, 2):
        write(s)
    runs = sec.dense_runs(tmp_path)
    assert sorted(runs) == [0, 1, 2] and runs[1][0].shape == (2 * 3 + 1, 2, 64, 64)
    write(2, values={**values, "rank": 2})
    with pytest.raises(SystemExit):
        sec.dense_runs(tmp_path)
    write(2, complete=False)
    with pytest.raises(SystemExit):
        sec.dense_runs(tmp_path)
