"""G2 random search: trial sampling, parameter budget, best-trial selection.

Protocol (notes/protocol.md, Models): 20 random-search trials per model over its config's
ranges; a trial's score is its best V1 metric under the shared early-stopping rules; the
best trial's configuration is then run with seeds 1 and 2 (seed 0 is the trial itself).
Spec: tests/test_g2search.py. Driver: experiments/g2_search.py.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

CONFIGS = {"holonomy": "configs/g2_holonomy.yaml", "rotate": "configs/g2_rotate.yaml",
           "quate": "configs/g2_quate.yaml"}
N_TRIALS, TRIAL_SEED = 20, 0

# Trainable-parameter band per dataset, checked for every model before it trains.
BUDGET = {
    "fb15k237": (1_900_000, 2_100_000),  # protocol note 2026-11-06: 2.0M +- 5%
    "wn18rr": (4_940_000, 5_460_000),    # protocol note 2026-11-10 (WN18RR sizes): 5.2M +- 5%
}


def load_spec(path: str | Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_trials(search: dict, n: int = N_TRIALS, seed: int = TRIAL_SEED) -> list[dict]:
    """n trials, each one value per `search` entry, drawn in file order from default_rng(seed).

    Entry forms: {choice: [...]}, {uniform: [lo, hi]}, {log_uniform: [lo, hi]}.
    Draws are trial-major (all of trial 0's values, then trial 1's), so trial i never
    depends on how many trials are run after it.
    """
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n):
        trial = {}
        for name, dist in search.items():
            if len(dist) != 1:
                raise ValueError(f"{name}: expected one distribution, got {dist}")
            (kind, args), = dist.items()
            if kind == "choice":
                trial[name] = args[int(rng.integers(len(args)))]
            elif kind == "uniform":
                lo, hi = map(float, args)
                trial[name] = float(rng.uniform(lo, hi))
            elif kind == "log_uniform":
                lo, hi = map(float, args)
                if not 0 < lo < hi:
                    raise ValueError(f"{name}: log_uniform needs 0 < lo < hi, got {args}")
                trial[name] = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
            else:
                raise ValueError(f"{name}: unknown distribution {kind!r}")
        trials.append(trial)
    return trials


def check_budget(parameters: int, dataset: str) -> None:
    lo, hi = BUDGET[dataset]
    if not lo <= parameters <= hi:
        raise ValueError(f"{parameters:,} trainable parameters is outside the {dataset} budget "
                         f"[{lo:,}, {hi:,}]")


def select_best(results: dict[int, dict], n: int = N_TRIALS) -> int:
    """Index of the best trial: highest best_v1_auc; ties go to the lower index.

    Requires all n trials to be complete. A trial whose best V1 metric is NaN (diverged)
    can never be selected.
    """
    missing = [i for i in range(n) if i not in results or not results[i].get("complete")]
    if missing:
        raise ValueError(f"trials not complete: {missing}")
    scores = [results[i]["best_v1_auc"] for i in range(n)]
    finite = [i for i in range(n) if scores[i] is not None and math.isfinite(scores[i])]
    if not finite:
        raise ValueError("no trial has a finite V1 score")
    return max(finite, key=lambda i: (scores[i], -i))
