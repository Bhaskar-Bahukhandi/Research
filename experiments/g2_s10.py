"""G2 secondary analysis S10: the holonomy model with dense rotations (protocol note of 2026-11-22, fourth).

    python experiments/g2_s10.py --seeds 0 1 2 --device cuda --out runs/g2s10 --max-hours 9

Dense rotations are the holonomy model at rank 32. Then each block's generator U V^T - V U^T can
have full rank 64, and FB15k-237's parameter count is exactly the 3.80M of the note of 2026-11-06.
That is outside the 2.0M budget, so the budget check is not applied and the size is stated.

Every other value is the best trial's (rtvm.g2search.select_best over the committed search records
in results/g2/holonomy/fb15k237, trial 4), including rotation_init. Fixed values come from the
committed configs/g2_holonomy.yaml. There is no new search. Training follows the shared rules
through rtvm.train_holonomy.train, exactly as experiments/g2_search.py runs the search: V1 early
stopping, the 1,000-epoch cap, TF32 training and strict FP32 evaluation.

FB15k-237 only. This code lives outside the frozen paths, so the gate's test-split guard keeps
holding.

Output, under --out/fb15k237/holonomy_dense/, resumable as in g2_search.py:
  seed_S.json, seed_S_best.pt, seed_S.ckpt, seed_S.log
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtvm.g2search import CONFIGS, N_TRIALS, TRIAL_SEED, file_sha256, fixed_values, load_spec, sample_trials, \
    select_best  # noqa: E402

DATASET = "fb15k237"
DENSE_RANK = 32                     # 2 * 32 = 64: a full-rank skew generator in each 64-dimensional block
RECORDS = ROOT / "results/g2/holonomy" / DATASET


def best_trial_values(records: Path = RECORDS) -> tuple[int, dict]:
    """The best trial of the committed holonomy search, and its values (checked against the config)."""
    spec = load_spec(ROOT / CONFIGS["holonomy"])
    trials = sample_trials(spec["search"], N_TRIALS, TRIAL_SEED)
    search = json.loads((records / "search.json").read_text())
    if search["config_sha256"] != file_sha256(ROOT / CONFIGS["holonomy"]) or search["trials"] != trials:
        raise SystemExit(f"{records}: the search records do not match the committed config")
    results = {i: json.loads((records / f"trial_{i:02d}.json").read_text()) for i in range(N_TRIALS)}
    best = select_best(results)
    return best, trials[best]


def dense_values(records: Path = RECORDS) -> tuple[int, dict]:
    best, values = best_trial_values(records)
    return best, {**values, "rank": DENSE_RANK}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "runs/g2s10"))
    ap.add_argument("--saved", default=str(ROOT / "saved/g2"))
    ap.add_argument("--data", default=str(ROOT / "data/g2"))
    ap.add_argument("--max-hours", type=float, help="stop at the first evaluation after this many hours")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only: allow uncommitted changes")
    args = ap.parse_args()
    deadline = time.time() + 3600 * args.max_hours if args.max_hours else None

    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    state = dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
    if state["dirty"] and not args.allow_dirty:
        raise SystemExit("uncommitted changes: commit first (the run must be traceable), or --allow-dirty for a smoke test")

    import torch
    from rtvm import train_holonomy as trainer
    from rtvm.holonomy_model import HolonomyModel
    from rtvm.utils import count_parameters

    spec = load_spec(ROOT / CONFIGS["holonomy"])
    best, values = dense_values()
    data = trainer.load_data(DATASET, saved=args.saved, data=args.data)
    run_dir = Path(args.out) / DATASET / "holonomy_dense"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"S10: trial {best}'s values with rank {DENSE_RANK}: {values}", flush=True)

    for seed in args.seeds:
        name = f"seed_{seed}"
        out_json = run_dir / f"{name}.json"
        if out_json.exists() and json.loads(out_json.read_text()).get("complete"):
            print(f"{name}: already complete, skipped")
            continue
        if deadline is not None and time.time() >= deadline:
            print("time limit reached before", name)
            break
        cfg = trainer.HolonomyConfig(**{**fixed_values(spec, DATASET), **values, "seed": seed})
        n_params = count_parameters(HolonomyModel(data.num_entities, data.num_relations, rank=cfg.rank))
        log_file = (run_dir / f"{name}.log").open("a", encoding="utf-8")

        def log(msg, name=name, log_file=log_file):
            line = f"[{name}] {msg}"
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        log(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} on {args.device}; {n_params:,} parameters "
            f"(outside the 2.0M budget: S10); {values}")
        result = trainer.train(cfg, data, device=args.device, checkpoint=run_dir / f"{name}.ckpt", log=log,
                               deadline=deadline)
        log_file.close()
        model = result.pop("model")
        if not result["complete"]:
            print(f"{name}: stopped at the time limit after epoch {result['epochs_run']}; rerun to continue")
            break
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / f"{name}_best.pt")
        record = {**result, "name": name, "analysis": "S10 (protocol note of 2026-11-22, fourth)",
                  "best_trial": best, "values": values, "seed": seed, "parameters": n_params,
                  "config_sha256": file_sha256(ROOT / CONFIGS["holonomy"]), "git": state,
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(out_json)
        print(f"{name}: done, best V1 {result['best_v1_auc']:.4f} at epoch {result['best_epoch']}")


if __name__ == "__main__":
    main()
