"""How long will G2 take on this machine?

Measured on a Colab A100-SXM4-40GB, 2026-11-06 (torch 2.11, PyKEEN 1.11.1), seconds per epoch:
  RotatE dim 68: 4.1 | QuatE dim 34: 11.1-11.5 | cycle term, 32 cycles/triple: 43.6 FP32,
  24.9 TF32 | 8 cycles/triple: 11.2 FP32, 6.4 TF32.

Times one epoch per model and projects the protocol's runs:

    .venv/Scripts/python experiments/time_budget.py              # CPU (slow; checks the script works)
    python experiments/time_budget.py --device cuda             # on Colab: the real estimate

Uses the FB15k-237 TRAINING split only (the protocol bars the test split).
Model sizes match the G2 parameter budget (notes/protocol.md):
  * holonomy model: 14,505 entities x 128 + 237 relations x 2 blocks of 64,
    with rank-r Cayley rotations; its training cost is dominated by the cycle
    term, timed here with --train-cycles cycles per triple (default 32), one
    corrupted cycle per true one, and the worst-case cycle length 4 (paths of length 3).
  * RotatE (complex dim 68) and QuatE (dim 34), sized to the 2.0M budget fixed in
    the protocol note of 2026-11-06.
Projection: 22 runs per model (20 tuning trials + 2 extra seeds) at 300
epochs (typical with early stopping; the RotatE paper's 100k steps at batch
1024 is about 376 epochs on FB15k-237) and at the 1,000-epoch cap.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rtvm.rotations import cayley, skew  # noqa: E402

N_ENT, N_REL, D, B = 14_505, 237, 128, 64
RUNS, EPOCHS = 22, (300, 1000)


def budgets() -> None:
    entities = N_ENT * D
    print("Parameter budget (trainable numbers stored):")
    for name, rel in [("rank 1", N_REL * 2 * 2 * B * 1), ("rank 2", N_REL * 2 * 2 * B * 2),
                      ("rank 4", N_REL * 2 * 2 * B * 4), ("dense", N_REL * 2 * B * B)]:
        total = entities + rel
        k = total / (2 * N_ENT + N_REL)      # RotatE: 2k reals per entity, k phases per relation
        q = total / (4 * N_ENT + 4 * N_REL)  # QuatE: 4q per entity and per relation
        print(f"  holonomy {name:6s}: {total / 1e6:5.2f}M -> RotatE dim {k:5.1f}, QuatE dim {q:5.1f}")


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_pykeen(name: str, dim: int, device: str, negatives: int) -> float:
    from pykeen.datasets import get_dataset
    from pykeen.models import QuatE, RotatE
    from pykeen.training import SLCWATrainingLoop

    tf = get_dataset(dataset="fb15k237").training
    cls = {"RotatE": RotatE, "QuatE": QuatE}[name]
    model = cls(triples_factory=tf, embedding_dim=dim, random_seed=0).to(device)
    loop = SLCWATrainingLoop(model=model, triples_factory=tf,
                             optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
                             negative_sampler="basic",
                             negative_sampler_kwargs=dict(num_negs_per_pos=negatives))
    sync(device)
    t0 = time.perf_counter()
    loop.train(triples_factory=tf, num_epochs=1, batch_size=1024, use_tqdm=False)
    sync(device)
    return time.perf_counter() - t0


def time_cycle_term(device: str, rank: int, steps: int, chunk: int, train_cycles: int) -> float:
    """Seconds per epoch for the cycle term: 272,115 triples / 1024 per batch."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    U = torch.randn(N_REL, 2, B, rank, generator=gen).to(device).requires_grad_()
    V = torch.randn(N_REL, 2, B, rank, generator=gen).to(device).requires_grad_()
    n_cycles = 1024 * train_cycles * 2   # true cycles plus one corrupted cycle each
    rel = torch.randint(N_REL, (n_cycles, 4), generator=gen)
    flip = torch.randint(2, (n_cycles, 4), generator=gen)
    idx = (rel + N_REL * flip).to(device)  # index into the 474-entry table [R; R^T]
    eye = torch.eye(B, device=device)
    opt = torch.optim.Adam([U, V], lr=1e-3)

    def step() -> None:
        opt.zero_grad()
        R = cayley(skew(U @ V.transpose(-1, -2)))          # (237, 2, 64, 64), exactly orthogonal
        table = torch.cat([R, R.transpose(-1, -2)])        # one gather per step, not two plus a where
        for s in range(0, n_cycles, chunk):
            i = idx[s:s + chunk]
            H = table[i[:, 0]]
            for j in range(1, 4):
                H = table[i[:, j]] @ H
            ((H - eye) ** 2).sum().backward(retain_graph=True)
        opt.step()

    step()  # warm-up
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    per_step = (time.perf_counter() - t0) / steps
    return per_step * (272_115 / 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--negatives", type=int, default=256, help="negatives per positive (RotatE paper: 256)")
    ap.add_argument("--cycle-steps", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=8192, help="cycles per chunk (lower if out of memory)")
    ap.add_argument("--train-cycles", type=int, default=32, help="training cycles per triple per epoch")
    ap.add_argument("--skip-pykeen", action="store_true", help="time only the cycle term")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmuls on Ampere GPUs (faster, ~1e-3 precision)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    if args.device.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(0), "| TF32:", args.tf32)
    budgets()

    epoch = {} if args.skip_pykeen else {
        "RotatE (dim 68)": time_pykeen("RotatE", 68, args.device, args.negatives),
        "QuatE (dim 34)": time_pykeen("QuatE", 34, args.device, args.negatives),
    }
    epoch[f"holonomy cycle term (rank 2, {args.train_cycles}/triple)"] = time_cycle_term(
        args.device, 2, args.cycle_steps, args.chunk, args.train_cycles)
    print(f"\nSeconds per epoch on {args.device}, and projected hours for {RUNS} runs per model:")
    total = [0.0, 0.0]
    for name, sec in epoch.items():
        hours = [sec * e * RUNS / 3600 for e in EPOCHS]
        total = [a + b for a, b in zip(total, hours)]
        print(f"  {name:30s} {sec:8.1f} s/epoch | {hours[0]:7.1f} h at {EPOCHS[0]} epochs | "
              f"{hours[1]:7.1f} h at {EPOCHS[1]} epochs")
    print(f"  {'TOTAL':30s} {'':8s}           | {total[0]:7.1f} h            | {total[1]:7.1f} h")
    print("Not included: path sampling and feature building (CPU, one-off), evaluation passes "
          "(small: grouped AUC on ~8.8k V1 positives x 11), and the holonomy model's single-triple term.")


if __name__ == "__main__":
    main()
