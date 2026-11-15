"""Time the G2 holonomy model's training step and V1 evaluation on the real G2 data.

    python experiments/time_holonomy.py --dataset fb15k237 --data data/g2 --device cuda --batches 30

Uses trial 0 of the frozen config (configs/g2_holonomy.yaml) and the committed training code
(rtvm.train_holonomy._batch_loss and evaluate_v1). Three warm-up batches, then `--batches`
timed batches and one V1 evaluation; projects hours for 100, 300 and 1,000 epochs. Saves
nothing and selects nothing: a measurement of cost only, like experiments/time_budget.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtvm import train_holonomy as th  # noqa: E402
from rtvm.g2search import CONFIGS, N_TRIALS, TRIAL_SEED, load_spec, sample_trials  # noqa: E402
from rtvm.holonomy_model import HolonomyModel, fact_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--saved", default=str(ROOT / "saved/g2"))
    ap.add_argument("--data", default=str(ROOT / "data/g2"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batches", type=int, default=20)
    args = ap.parse_args()

    spec = load_spec(ROOT / CONFIGS["holonomy"])
    values = sample_trials(spec["search"], N_TRIALS, TRIAL_SEED)[0]
    cfg = th.HolonomyConfig(**{**spec["fixed"], **values, "seed": 0})
    t = time.perf_counter()
    data = th.load_data(args.dataset, saved=args.saved, data=args.data)
    print(f"{args.dataset}: trial 0 {values}; data loaded and hash-checked in {time.perf_counter() - t:.0f} s; "
          f"{len(data.train):,} training triples", flush=True)

    torch.use_deterministic_algorithms(True)                 # as in train_holonomy.train
    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    model = HolonomyModel(data.num_entities, data.num_relations, rank=cfg.rank, rotation_init=cfg.rotation_init,
                          entity_init=cfg.entity_init, generator=torch.Generator().manual_seed(0)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pad = 2 * data.num_relations
    n_paths = (data.train_steps != pad).any(axis=-1).sum(axis=-1)
    train_keys = np.sort(fact_key(data.train[:, 0], data.train[:, 1], data.train[:, 2],
                                  data.num_entities, data.num_relations))
    n_total = max(1, int(np.ceil(len(data.train) / cfg.batch_size)))
    rng = np.random.default_rng([0, 1])
    batches = np.array_split(rng.permutation(len(data.train)), n_total)
    th._set_tf32(cfg.tf32)
    model.train()
    warm = 3
    for i, idx in enumerate(batches[: warm + args.batches]):
        if i == warm:
            sync()
            t = time.perf_counter()
        loss = th._batch_loss(model, data, idx, n_paths, train_keys, cfg, rng, dev)
        opt.zero_grad()
        loss.backward()
        opt.step()
    sync()
    per_batch = (time.perf_counter() - t) / args.batches
    t = time.perf_counter()
    metrics = th.evaluate_v1(model, data, dev)
    sync()
    ev = time.perf_counter() - t
    epoch = per_batch * n_total
    mem = f"{torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GiB" if dev.type == "cuda" else "n/a"
    print(f"{args.dataset}: {per_batch:.3f} s/batch x {n_total} batches = {epoch:.0f} s/epoch; V1 evaluation "
          f"{ev:.0f} s (stratified {metrics['stratified']:.4f}); peak GPU memory {mem}; loss {loss.item():.4f}",
          flush=True)
    for epochs in (100, 300, 1000):
        print(f"  {epochs} epochs incl. {epochs // 10} evaluations: {(epoch * epochs + ev * epochs / 10) / 3600:.1f} h",
              flush=True)


if __name__ == "__main__":
    main()
