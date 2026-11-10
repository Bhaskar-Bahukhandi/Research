"""Sanity check of the G2 training loop on a synthetic graph with planted composition rules.

    .venv/Scripts/python experiments/g2_synthetic_check.py

Entities are the cells of an n x n torus; relations are moves (east, north, north-east,
north-west). Every cycle closes exactly in the ground truth (east then north = north-east),
so a model that learns relation rotations which compose like the moves should give true
triples low holonomy and entity-swapped negatives high holonomy.

Uses no real dataset, so it cannot leak anything into G2 tuning. It checks that:
  1. with the cycle term, V1 grouped AUC of the holonomy score rises well above 0.5;
  2. without it (alpha1 = alpha2 = 0), the holonomy score learns much less.
Hyperparameters here are chosen for this toy only and say nothing about G2's ranges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtvm.g2data import TrainGraph, sample_negatives, sample_paths  # noqa: E402
from rtvm.holonomy_model import HolonomyModel, cycle_defect, path_transport, steps_from_codes  # noqa: E402
from rtvm.train_holonomy import HolonomyConfig, HolonomyData, evaluate_v1, train  # noqa: E402

MOVES = [(1, 0), (0, 1), (1, 1), (-1, 1)]


def torus(n: int = 10, seed: int = 0) -> tuple[HolonomyData, int]:
    cell = lambda x, y: (x % n) * n + (y % n)       # noqa: E731
    facts = np.array([(cell(x, y), r, cell(x + dx, y + dy))
                      for x in range(n) for y in range(n) for r, (dx, dy) in enumerate(MOVES)])
    rng = np.random.default_rng(seed)
    facts = facts[rng.permutation(len(facts))]
    n_valid = len(facts) // 10
    valid, train = facts[:n_valid], facts[n_valid:]
    n_ent, n_rel = n * n, len(MOVES)
    known = set(map(tuple, facts.tolist()))
    neg, _ = sample_negatives(valid, train, known, n_ent)
    graph = TrainGraph.from_training(train, n_ent)

    def steps(triples, own=None):
        return steps_from_codes(sample_paths(graph, triples, own_edges=own), train[:, 1], n_rel)

    m = len(valid)
    data = HolonomyData(n_ent, n_rel, train, steps(train, np.arange(len(train))),
                        valid, steps(valid), neg, steps(neg.reshape(-1, 3)).reshape(m, 10, -1, 3))
    return data, len(train)


@torch.no_grad()
def length_matched_auc(model: HolonomyModel, data: HolonomyData) -> dict[int, float]:
    """AUC of single-cycle defects, positives' cycles vs negatives' cycles OF THE SAME LENGTH.

    The triple score averages cycles of mixed lengths, and with near-identity rotations a
    longer cycle has a larger defect, so path-length mix alone separates positives from
    negatives. Matching lengths removes that shortcut: only learned composition is left.
    """
    pad = 2 * data.num_relations
    table = model.table()
    out = {}

    def cycles(rel, steps):
        steps = steps.reshape(-1, *steps.shape[-2:])
        rel = np.repeat(np.asarray(rel).reshape(-1), steps.shape[1])
        steps = steps.reshape(-1, steps.shape[-1])
        L = (steps != pad).sum(-1)
        keep = L > 0
        d = cycle_defect(table, path_transport(table, torch.as_tensor(steps[keep])),
                         torch.as_tensor(rel[keep])).numpy()
        return d, L[keep]

    d_pos, l_pos = cycles(data.v1_pos[:, 1], data.v1_pos_steps)
    d_neg, l_neg = cycles(data.v1_neg[..., 1], data.v1_neg_steps)
    for L in (1, 2, 3):
        p, n = d_pos[l_pos == L], d_neg[l_neg == L]
        if len(p) and len(n):
            out[L] = float(((n[None] > p[:, None]) + 0.5 * (n[None] == p[:, None])).mean())
    return out


def main() -> None:
    torch.manual_seed(0)
    data, n_train = torus()
    print(f"torus: {data.num_entities} entities, {data.num_relations} relations, {n_train} training facts, "
          f"{len(data.v1_pos)} validation positives")
    base = dict(rank=2, lr=3e-3, rotation_init=0.1, gamma=6.0, temperature=1.0, margin=2.0,
                num_negatives=32, batch_size=128, eval_every=10, patience=5, max_epochs=200, tf32=False)
    untrained = HolonomyModel(data.num_entities, data.num_relations, rank=2, rotation_init=0.1,
                              generator=torch.Generator().manual_seed(0))
    fmt = lambda d: ", ".join(f"length {L}: {a:.3f}" for L, a in d.items())     # noqa: E731
    print(f"{'untrained':24s}: V1 grouped AUC {evaluate_v1(untrained, data, 'cpu'):.3f} | "
          f"length-matched cycle AUC {fmt(length_matched_auc(untrained, data))}")
    runs = (("cycle term, protocol stop", dict(alpha1=1.0, alpha2=1.0)),
            ("cycle term, 200 epochs", dict(alpha1=1.0, alpha2=1.0, eval_every=200, patience=10 ** 6)),
            ("single term, 200 epochs", dict(alpha1=0.0, alpha2=0.0, eval_every=200, patience=10 ** 6)))
    for name, changes in runs:
        out = train(HolonomyConfig(**{**base, **changes}), data, log=lambda *_: None)
        print(f"{name:24s}: V1 grouped AUC {out['best_v1_auc']:.3f} | length-matched cycle AUC "
              f"{fmt(length_matched_auc(out['model'], data))} (weights from epoch {out['best_epoch']})")
    print("Reading: the triple-level V1 metric saturates from path-length mix alone, so early stopping on it\n"
          "keeps weights from before the model has learned composition (see the length-matched column).")


if __name__ == "__main__":
    main()
