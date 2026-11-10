"""Baseline reproduction check: does our RotatE pipeline reach the published RotatE results?

Allowed by the protocol note of 2026-11-06 ("baseline reproduction may use the test split"):
its numbers are not G2 features and must not influence any G2 configuration, search range or
threshold. These models use the published sizes, far above the 2.0M G2 budget.

Hyperparameters are the official ones from
https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/blob/master/best_config.sh
(positional order from run.sh: batch, negatives, hidden dim, gamma, alpha, lr, max steps):
    RotatE FB15k-237: 1024 256 1000 9.0 1.0 0.00005 100000   (-de, -adv)
    RotatE wn18rr:     512 1024  500 6.0 0.5 0.00005  80000   (-de, -adv)
Published filtered test MRR (as recorded in PyKEEN's bundled sun2019 configs): FB15k-237 0.338,
WN18RR 0.476. Check these against Table 5 of Sun et al. (2019), arXiv:1902.10197, before citing.

Two variants:
  official  PyKEEN training loop and NSSA loss, with the official model details:
            distance = sum of complex moduli (L1), relations stored as phases
            theta = param * pi / range, uniform init in [-range, range] with
            range = (gamma + 2) / dim, filtered negatives, lr / 10 at half the steps.
  stock     PyKEEN 1.11.1's RotatE with its bundled sun2019 config: L2 distance, relations as
            unit complex numbers (renormalised each step), U(0, 1) entity init, no lr decay.
            The bundled config says 1,000 epochs ("could not find number of epochs"); here
            both variants train for the official number of steps instead.

Known remaining differences from the official code (official variant):
  * no word2vec-style subsampling weights (official default; PyKEEN has no equivalent);
  * each negative corrupts head or tail at random, instead of whole head- or tail-batches;
  * PyKEEN drops filtered negatives instead of redrawing them;
  * ranks use PyKEEN's realistic (tie-averaged) rank; optimistic and pessimistic also saved.

    .venv/Scripts/python experiments/reproduce_rotate.py --dataset nations --smoke   # CPU check
    python experiments/reproduce_rotate.py --dataset fb15k237 --variant official --device cuda
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

import pykeen
from pykeen.datasets import get_dataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import ERModel, RotatE
from pykeen.training import SLCWATrainingLoop
from pykeen.utils import set_random_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtvm.baselines import PhaseRotatEInteraction  # noqa: E402  (the class used by G2)

OFFICIAL = {  # best_config.sh, RotatE
    "fb15k237": dict(batch=1024, negatives=256, dim=1000, gamma=9.0, alpha=1.0, lr=5e-5, steps=100_000),
    "wn18rr": dict(batch=512, negatives=1024, dim=500, gamma=6.0, alpha=0.5, lr=5e-5, steps=80_000),
    # not in best_config.sh; only for the CPU smoke test
    "nations": dict(batch=128, negatives=16, dim=32, gamma=6.0, alpha=0.5, lr=1e-3, steps=600),
}
PUBLISHED_MRR = {"fb15k237": 0.338, "wn18rr": 0.476}


def build_model(variant: str, tf, hp: dict, seed: int):
    loss = dict(loss="nssa", loss_kwargs=dict(margin=hp["gamma"], adversarial_temperature=hp["alpha"],
                                              reduction="mean"))
    if variant == "official":
        rng = (hp["gamma"] + 2.0) / hp["dim"]  # official: (gamma + epsilon) / hidden_dim, epsilon = 2
        init = functools.partial(torch.nn.init.uniform_, a=-rng, b=rng)
        return ERModel(
            triples_factory=tf,
            interaction=PhaseRotatEInteraction(rng),
            entity_representations_kwargs=dict(shape=hp["dim"], dtype=torch.cfloat, initializer=init),
            relation_representations_kwargs=dict(shape=hp["dim"], initializer=init),
            random_seed=seed, **loss,
        )
    return RotatE(triples_factory=tf, embedding_dim=hp["dim"], entity_initializer="uniform",
                  relation_initializer="init_phases", relation_constrainer="complex_normalize",
                  random_seed=seed, **loss)


def evaluate(model, dataset, split: str, batch_size: int) -> dict:
    triples = getattr(dataset, split).mapped_triples
    others = [s.mapped_triples for name, s in
              [("training", dataset.training), ("validation", dataset.validation), ("testing", dataset.testing)]
              if name != split]
    result = RankBasedEvaluator(filtered=True).evaluate(
        model, triples, additional_filter_triples=others, batch_size=batch_size, use_tqdm=False)
    out = {}
    for rank_type in ("realistic", "optimistic", "pessimistic"):
        out[rank_type] = {
            "mrr": result.get_metric(f"both.{rank_type}.inverse_harmonic_mean_rank"),
            "mr": result.get_metric(f"both.{rank_type}.arithmetic_mean_rank"),
            **{f"hits@{k}": result.get_metric(f"both.{rank_type}.hits_at_{k}") for k in (1, 3, 10)},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(OFFICIAL), required=True)
    ap.add_argument("--variant", choices=["official", "stock"], default="official")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", action="store_true", help="time one epoch, print the projected hours, exit")
    ap.add_argument("--smoke", action="store_true", help="tiny run on CPU to check the script")
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--checkpoint-minutes", type=int, default=15)
    ap.add_argument("--progress", action="store_true", help="print an epoch progress bar every 2 minutes")
    ap.add_argument("--out", default="experiments/out")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False  # strict FP32 throughout
    set_random_seed(args.seed)
    hp = dict(OFFICIAL[args.dataset])
    dataset = get_dataset(dataset=args.dataset)
    tf = dataset.training
    steps_per_epoch = math.ceil(tf.num_triples / hp["batch"])
    epochs = max(1, round(hp["steps"] / steps_per_epoch))
    if args.time:
        epochs = 1
    model = build_model(args.variant, tf, hp, args.seed).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    scheduler = None
    if args.variant == "official":  # official: lr / 10 once warm_up_steps = max_steps // 2 is reached
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)
    loop = SLCWATrainingLoop(
        model=model, triples_factory=tf, optimizer=optimizer, lr_scheduler=scheduler,
        negative_sampler="basic",
        negative_sampler_kwargs=dict(num_negs_per_pos=hp["negatives"], filtered=args.variant == "official"),
    )
    name = f"rotate_{args.dataset}_{args.variant}_seed{args.seed}"
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else platform.processor()
    print(f"{name}: {tf.num_triples} training triples, {steps_per_epoch} steps/epoch, {epochs} epochs "
          f"({epochs * steps_per_epoch} steps; official {hp['steps']}), device {gpu}", flush=True)

    t0 = time.perf_counter()
    if args.time:
        # one epoch is the unit PyKEEN trains in; time it and project
        loop.train(triples_factory=tf, num_epochs=1, batch_size=hp["batch"], use_tqdm=False)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        full = round(hp["steps"] / steps_per_epoch)
        print(f"one epoch: {sec:.1f} s -> {sec * full / 3600:.2f} h for {full} epochs", flush=True)
        return

    ckpt = Path(args.checkpoint_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    # an existing checkpoint with this name is resumed automatically (e.g. after a Colab disconnect)
    losses = loop.train(triples_factory=tf, num_epochs=epochs, batch_size=hp["batch"],
                        use_tqdm=args.progress, tqdm_kwargs=dict(mininterval=120, ascii=True),
                        checkpoint_name=f"{name}.pt", checkpoint_directory=ckpt,
                        checkpoint_frequency=0 if args.smoke else args.checkpoint_minutes)
    train_sec = time.perf_counter() - t0
    model.eval()
    with torch.no_grad():
        valid = evaluate(model, dataset, "validation", args.eval_batch)
        test = evaluate(model, dataset, "testing", args.eval_batch)
    record = {
        "name": name, "dataset": args.dataset, "variant": args.variant, "seed": args.seed,
        "hyperparameters": hp, "epochs": epochs, "steps": epochs * steps_per_epoch,
        "train_hours": train_sec / 3600, "loss_first_last": [losses[0], losses[-1]],
        "validation": valid, "test": test, "published_test_mrr": PUBLISHED_MRR.get(args.dataset),
        "device": gpu, "torch": torch.__version__, "pykeen": pykeen.get_version(),
        "purpose": "implementation check only (protocol note 2026-11-06); not a G2 result",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2))
    r = test["realistic"]
    print(f"done in {train_sec / 3600:.2f} h | test MRR {r['mrr']:.4f} (published "
          f"{PUBLISHED_MRR.get(args.dataset)}) | H@1 {r['hits@1']:.4f} H@10 {r['hits@10']:.4f}", flush=True)


if __name__ == "__main__":
    main()
