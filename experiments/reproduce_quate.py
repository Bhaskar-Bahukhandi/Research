"""Baseline reproduction check: does an official-code QuatE in PyKEEN reach the published QuatE results?

Second half of the QuatE official-code check (notes/quate_check.md; protocol note 2026-11-08). Allowed
by the protocol note of 2026-11-06: test-split numbers here check the implementation only. They are not
G2 features and must not influence any G2 configuration, search range or threshold. Published sizes
(dimension 100) are far above the G2 budget.

Settings are the official ones (github.com/cheungdaven/QuatE, 77cdd49f2bc2, train_QuatE_*.py):
    FB15k-237: dim 100, 5,000 epochs, 10 batches/epoch, 10 negatives, lambda 0.3, Adagrad lr 0.1
    WN18RR:    dim 100, 40,000 epochs, 10 batches/epoch, 1 negative,  lambda 0.1, Adagrad lr 0.1
    (lmbda_two is set in the scripts but unused by the model; both terms use lambda.)
    Validation Hits@10 every 1,000 (FB15k-237) / 10,000 (WN18RR) epochs; the best weights are kept.
Published QuatE^1 filtered test MRR (as recorded in PyKEEN's bundled zhang2019 configs): FB15k-237
0.311, WN18RR 0.481. Check these against Zhang et al. (2019), arXiv:1904.10281, before citing.

Variants:
  official  rtvm.baselines.build_quate_official (raw relations normalised in the score, squared-modulus
            regularisers on raw parameters, entity-count relation init), exactly filtered Bernoulli
            negatives, batch = train / 10.
  stock     PyKEEN 1.11.1's QuatE with its bundled zhang2019 settings: Lp regulariser (p 2, not
            normalised) applied after relation normalisation, uniform unfiltered negatives, batch 9,300.

Known remaining differences from the official code (official variant): PyKEEN shuffles each epoch
(OpenKE draws batch rows at random with replacement); a filtered negative is dropped, not redrawn;
ranks use PyKEEN's realistic rank, and PyKEEN drops evaluation triples with entities unseen in
training (notes/baseline_reproduction.md).

    .venv/Scripts/python experiments/reproduce_quate.py --dataset nations --smoke        # CPU check
    python experiments/reproduce_quate.py --dataset fb15k237 --variant official --device cuda --time 3
    python experiments/reproduce_quate.py --dataset fb15k237 --variant official --device cuda --max-hours 11
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

import pykeen
from pykeen.datasets import get_dataset
from pykeen.models import QuatE
from pykeen.training import SLCWATrainingLoop
from pykeen.utils import set_random_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtvm.baselines import ExactFilterer, _rng_state, _set_rng_state, build_quate_official  # noqa: E402
from reproduce_rotate import evaluate  # noqa: E402

OFFICIAL = {
    "fb15k237": dict(dim=100, epochs=5000, nbatches=10, negatives=10, lmbda=0.3, lr=0.1, valid_every=1000),
    "wn18rr": dict(dim=100, epochs=40000, nbatches=10, negatives=1, lmbda=0.1, lr=0.1, valid_every=10000),
    # not official; only for the CPU smoke test
    "nations": dict(dim=16, epochs=40, nbatches=10, negatives=4, lmbda=0.1, lr=0.1, valid_every=20),
}
STOCK_BATCH = 9300                     # PyKEEN's bundled zhang2019 configs
PUBLISHED_MRR = {"fb15k237": 0.311, "wn18rr": 0.481}


def build(variant: str, tf, hp: dict, seed: int, device: str):
    loss = dict(loss="bcewithlogits", loss_kwargs=dict(reduction="mean"))
    if variant == "official":
        model = build_quate_official(tf, hp["dim"], hp["lmbda"], seed, **loss)
        sampler = "bernoulli"
        sampler_kwargs = dict(num_negs_per_pos=hp["negatives"], filtered=True, filterer=ExactFilterer,
                              filterer_kwargs=dict(device=device))
        batch = tf.num_triples // hp["nbatches"]
    else:
        reg = dict(weight=hp["lmbda"], p=2.0, normalize=False)
        model = QuatE(triples_factory=tf, embedding_dim=hp["dim"], entity_regularizer_kwargs=reg,
                      relation_regularizer_kwargs=reg, random_seed=seed, **loss)
        sampler, sampler_kwargs, batch = "basic", dict(num_negs_per_pos=hp["negatives"]), STOCK_BATCH
    model = model.to(device)
    loop = SLCWATrainingLoop(model=model, triples_factory=tf,
                             optimizer=torch.optim.Adagrad(model.parameters(), lr=hp["lr"]),
                             negative_sampler=sampler, negative_sampler_kwargs=sampler_kwargs,
                             automatic_memory_optimization=False)
    return model, loop, batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(OFFICIAL), required=True)
    ap.add_argument("--variant", choices=["official", "stock"], default="official")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=int, default=0, metavar="N", help="time N epochs, print projected hours, exit")
    ap.add_argument("--smoke", action="store_true", help="tiny run on CPU to check the script")
    ap.add_argument("--chunk", type=int, default=50, help="epochs between checkpoints")
    ap.add_argument("--max-hours", type=float, default=None, help="stop after a checkpoint past this time")
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--out", default="experiments/out")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False  # strict FP32 throughout
    set_random_seed(args.seed)
    hp = dict(OFFICIAL[args.dataset])
    dataset = get_dataset(dataset=args.dataset)
    tf = dataset.training
    model, loop, batch = build(args.variant, tf, hp, args.seed, args.device)
    name = f"quate_{args.dataset}_{args.variant}_seed{args.seed}"
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else platform.processor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {tf.num_triples} training triples, batch {batch}, {hp['epochs']} epochs, "
          f"{n_params:,} parameters, device {gpu}", flush=True)

    if args.time:
        loop.train(triples_factory=tf, num_epochs=1, batch_size=batch, use_tqdm=False)   # warm-up
        sync = torch.cuda.synchronize if args.device.startswith("cuda") else (lambda: None)
        sync()
        t0 = time.perf_counter()
        loop.train(triples_factory=tf, num_epochs=1 + args.time, batch_size=batch, use_tqdm=False,
                   continue_training=True)
        sync()
        sec = (time.perf_counter() - t0) / args.time
        t0 = time.perf_counter()
        evaluate(model, dataset, "validation", args.eval_batch)
        ev = time.perf_counter() - t0
        n_valid = hp["epochs"] // hp["valid_every"]
        print(f"{sec:.2f} s/epoch, validation eval {ev:.0f} s -> {(sec * hp['epochs'] + ev * n_valid) / 3600:.2f} h "
              f"for {hp['epochs']} epochs with {n_valid} validations", flush=True)
        return

    deadline = time.time() + args.max_hours * 3600 if args.max_hours is not None else None
    ckpt = Path(args.checkpoint_dir) / f"{name}.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    state = dict(epoch=0, best_hits10=-1.0, best_epoch=0, history=[], train_seconds=0.0)
    best_weights = None
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=args.device, weights_only=False)
        model.load_state_dict(saved["model"])
        loop.optimizer.load_state_dict(saved["optimizer"])
        loop._epoch = saved["state"]["epoch"]
        state, best_weights = saved["state"], saved["best_weights"]
        _set_rng_state(saved["rng"], torch.device(args.device))
        print(f"resumed at epoch {state['epoch']}", flush=True)

    chunk = hp["epochs"] if args.smoke else args.chunk
    while state["epoch"] < hp["epochs"]:
        t0 = time.perf_counter()
        target = min(state["epoch"] + chunk, hp["epochs"])
        target = min(target, (state["epoch"] // hp["valid_every"] + 1) * hp["valid_every"])   # stop at validations
        losses = loop.train(triples_factory=tf, num_epochs=target, batch_size=batch, use_tqdm=False,
                            continue_training=state["epoch"] > 0)
        state["epoch"] = target
        state["train_seconds"] += time.perf_counter() - t0
        line = f"epoch {target}: loss {losses[-1]:.5f}"
        if target % hp["valid_every"] == 0:
            model.eval()
            with torch.no_grad():
                hits10 = evaluate(model, dataset, "validation", args.eval_batch)["realistic"]["hits@10"]
            state["history"].append(dict(epoch=target, valid_hits10=hits10, loss=losses[-1]))
            line += f", valid H@10 {hits10:.4f}"
            if hits10 > state["best_hits10"]:
                state["best_hits10"], state["best_epoch"] = hits10, target
                best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(line, flush=True)
        tmp = ckpt.with_suffix(".tmp")
        torch.save(dict(model=model.state_dict(), optimizer=loop.optimizer.state_dict(), state=state,
                        best_weights=best_weights, rng=_rng_state(torch.device(args.device))), tmp)
        tmp.replace(ckpt)
        if deadline is not None and time.time() >= deadline and state["epoch"] < hp["epochs"]:
            print(f"stopped at the time limit after epoch {state['epoch']}; rerun to continue", flush=True)
            return

    model.load_state_dict(best_weights)          # official: keep the best validation Hits@10 weights
    model.eval()
    with torch.no_grad():
        valid = evaluate(model, dataset, "validation", args.eval_batch)
        test = evaluate(model, dataset, "testing", args.eval_batch)
    record = {
        "name": name, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "hyperparameters": hp, "batch_size": batch, "parameters": n_params,
        "best_epoch": state["best_epoch"], "history": state["history"],
        "train_hours": state["train_seconds"] / 3600,
        "validation": valid, "test": test, "published_test_mrr": PUBLISHED_MRR.get(args.dataset),
        "device": gpu, "torch": torch.__version__, "pykeen": pykeen.get_version(),
        "purpose": "implementation check only (protocol note 2026-11-06); not a G2 result",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2))
    r = test["realistic"]
    print(f"done ({state['train_seconds'] / 3600:.2f} h training, best epoch {state['best_epoch']}) | test MRR "
          f"{r['mrr']:.4f} (published {PUBLISHED_MRR.get(args.dataset)}) | H@1 {r['hits@1']:.4f} "
          f"H@10 {r['hits@10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
