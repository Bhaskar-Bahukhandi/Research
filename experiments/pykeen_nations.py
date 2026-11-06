"""Learn PyKEEN's API on a tiny dataset (plan: Wed Nov 18).

    .venv/Scripts/python experiments/pykeen_nations.py

Nations has 14 entities, 55 relations and about 2k triples, so this runs in
about a minute on CPU. The point is NOT the score: it is that you can explain
every argument below. Write one line per argument in the weekly log.
"""
from __future__ import annotations

import json
from pathlib import Path

from pykeen.pipeline import pipeline

OUT = Path(__file__).resolve().parent / "out"


def main() -> None:
    result = pipeline(
        dataset="nations",                  # bundled with PyKEEN; FB15k-237 / WN18RR download on first use
        model="RotatE",
        model_kwargs=dict(embedding_dim=64),            # complex dimension: 64 complex = 128 real numbers per entity
        loss="nssa",                                    # self-adversarial negative sampling loss (RotatE paper, Sec. 3.3)
        loss_kwargs=dict(margin=6.0, adversarial_temperature=1.0),  # gamma and alpha in the paper
        negative_sampler="basic",                       # corrupt head or tail uniformly at random
        negative_sampler_kwargs=dict(num_negs_per_pos=32),
        optimizer="Adam",
        optimizer_kwargs=dict(lr=1e-3),
        training_loop="sLCWA",                          # stochastic local closed-world assumption: positives + sampled negatives
        training_kwargs=dict(num_epochs=200, batch_size=256),
        evaluator_kwargs=dict(filtered=True),           # rank against all entities, skipping other TRUE triples
        random_seed=0,
        device="cpu",
    )
    metrics = {
        "mrr": result.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        "hits_at_1": result.get_metric("both.realistic.hits_at_1"),
        "hits_at_10": result.get_metric("both.realistic.hits_at_10"),
    }
    print(json.dumps(metrics, indent=2))
    OUT.mkdir(exist_ok=True)
    (OUT / "pykeen_nations.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
