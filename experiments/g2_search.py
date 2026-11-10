"""G2 random search and seed runs for one model on one dataset (resumable).

    python experiments/g2_search.py --model rotate --dataset fb15k237 --trials 0-19 --device cuda
    python experiments/g2_search.py --model rotate --dataset fb15k237 --seeds 1 2 --device cuda
    python experiments/g2_search.py --model rotate --dataset fb15k237 --status

Implements notes/protocol.md (Models, Tuning budget) with the dated notes:
  * 20 trials per model, drawn once from the model's config (rtvm.g2search.sample_trials,
    trial seed 0); every trial trains with model seed 0;
  * a trial's score is its best V1 metric (holonomy: stratified cycle AUC, note
    2026-11-10; RotatE/QuatE: grouped AUC of score_hrt), with the shared early stopping;
  * after all 20 trials, the best trial's configuration is trained with seeds 1 and 2.

Output, under --out/<dataset>/<model>/:
  search.json        the 20 sampled trials, the config file's SHA-256 and the git commit.
                     A later invocation refuses to continue if the config file changed.
  trial_NN.json      the finished run (config, best V1 metric, history, GPU, versions).
  trial_NN_best.pt   the best-V1 weights (state_dict).
  trial_NN.ckpt      the resumable checkpoint, written at every evaluation.
  trial_NN.log       the training log.
  seed_S.*           the same for seed runs.

Resuming: rerun the same command. Finished runs are skipped, and an unfinished run
continues from its checkpoint. --max-hours stops cleanly at the first evaluation after the
time limit, so a Kaggle or Colab session can end without losing work.

Refuses to run from a git tree with uncommitted changes (the run must be traceable to a
commit) unless --allow-dirty is given, which is for smoke tests only.
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

from rtvm.g2search import (BUDGET, CONFIGS, N_TRIALS, TRIAL_SEED, check_budget, file_sha256,  # noqa: E402
                           load_spec, sample_trials, select_best)


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))


def parse_trials(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    if not all(0 <= i < N_TRIALS for i in out):
        raise SystemExit(f"--trials must lie in 0-{N_TRIALS - 1}")
    return out


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)                                    # atomic: parallel processes never see half a file


def read_results(run_dir: Path) -> dict[int, dict]:
    out = {}
    for i in range(N_TRIALS):
        p = run_dir / f"trial_{i:02d}.json"
        if p.exists():
            out[i] = json.loads(p.read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(CONFIGS), required=True)
    ap.add_argument("--dataset", choices=sorted(BUDGET), required=True)
    job = ap.add_mutually_exclusive_group(required=True)
    job.add_argument("--trials", help="trial indices, e.g. 0-19 or 0-9 or 3,7")
    job.add_argument("--seeds", type=int, nargs="+", help="seed runs of the best trial, e.g. 1 2")
    job.add_argument("--status", action="store_true", help="print the trials and results so far")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/g2")
    ap.add_argument("--saved", default=str(ROOT / "saved/g2"))
    ap.add_argument("--data", default=str(ROOT / "data/g2"), help="path files (holonomy model only)")
    ap.add_argument("--max-hours", type=float, help="stop at the first evaluation after this many hours")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke tests only: allow uncommitted changes")
    args = ap.parse_args()
    deadline = time.time() + 3600 * args.max_hours if args.max_hours else None

    config_path = ROOT / CONFIGS[args.model]
    spec = load_spec(config_path)
    trials = sample_trials(spec["search"], N_TRIALS, TRIAL_SEED)
    run_dir = Path(args.out) / args.dataset / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    search_file = run_dir / "search.json"
    git = git_state()

    record = dict(model=args.model, dataset=args.dataset, config=CONFIGS[args.model],
                  config_sha256=file_sha256(config_path), trial_seed=TRIAL_SEED, trials=trials)
    if search_file.exists():
        old = json.loads(search_file.read_text())
        if {k: old.get(k) for k in record} != record:
            raise SystemExit(f"{search_file} was made from a different config ({old['config_sha256'][:12]}, "
                             f"now {record['config_sha256'][:12]}); a frozen config needs a dated protocol note, "
                             "and a new search needs a new --out")
    elif not args.status:
        write_json(search_file, {**record, "git": git, "created": time.strftime("%Y-%m-%d %H:%M:%S")})

    results = read_results(run_dir)
    if args.status:
        print(f"{args.model} on {args.dataset}: config {record['config_sha256'][:12]}")
        for i, t in enumerate(trials):
            r = results.get(i)
            res = (f"V1 {r['best_v1_auc']:.4f} at epoch {r['best_epoch']}/{r['epochs_run']}"
                   if r else ("running/partial" if (run_dir / f"trial_{i:02d}.ckpt").exists() else "-"))
            params = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.items())
            print(f"  trial {i:2d}: {res:32s} {params}")
        if all(i in results for i in range(N_TRIALS)):
            print(f"best trial: {select_best(results)}")
        return

    if git["dirty"] and not args.allow_dirty:
        raise SystemExit("uncommitted changes: commit first (the run must be traceable), or --allow-dirty for a smoke test")

    if args.seeds:
        best = select_best(results)
        jobs = [(f"seed_{s}", trials[best], s) for s in args.seeds]
        print(f"best trial {best} (V1 {results[best]['best_v1_auc']:.4f}): {trials[best]}")
    else:
        jobs = [(f"trial_{i:02d}", trials[i], 0) for i in parse_trials(args.trials)]

    import torch
    if args.model == "holonomy":
        from rtvm import train_holonomy as trainer
        from rtvm.holonomy_model import HolonomyModel
        from rtvm.utils import count_parameters
        data = trainer.load_data(args.dataset, saved=args.saved, data=args.data)

        def make(values, seed):
            return trainer.HolonomyConfig(**{**spec["fixed"], **values, "seed": seed})

        def parameters(cfg):
            return count_parameters(HolonomyModel(data.num_entities, data.num_relations, rank=cfg.rank))
    else:
        from rtvm import baselines as trainer
        from rtvm.utils import count_parameters
        data = trainer.load_data(args.dataset, saved=args.saved)

        def make(values, seed):
            return trainer.BaselineConfig(model=spec["model"], dim=spec["dim"][args.dataset],
                                          **{**spec["fixed"], **values, "seed": seed})

        def parameters(cfg):
            return count_parameters(trainer.build_model(cfg, data.training))

    for name, values, seed in jobs:
        out_json = run_dir / f"{name}.json"
        if out_json.exists() and json.loads(out_json.read_text()).get("complete"):
            print(f"{name}: already complete, skipped")
            continue
        if deadline is not None and time.time() >= deadline:
            print("time limit reached before", name)
            break
        cfg = make(values, seed)
        n_params = parameters(cfg)
        check_budget(n_params, args.dataset)
        log_file = (run_dir / f"{name}.log").open("a", encoding="utf-8")

        def log(msg, name=name, log_file=log_file):
            line = f"[{name}] {msg}"
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        log(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} on {args.device}; {n_params:,} parameters; {values}")
        result = trainer.train(cfg, data, device=args.device, checkpoint=run_dir / f"{name}.ckpt",
                               log=log, deadline=deadline)
        log_file.close()
        model = result.pop("model")
        if not result["complete"]:
            print(f"{name}: stopped at the time limit after epoch {result['epochs_run']}; rerun to continue")
            break
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / f"{name}_best.pt")
        write_json(out_json, {**result, "name": name, "values": values, "seed": seed, "parameters": n_params,
                              "config_sha256": record["config_sha256"], "git": git,
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"{name}: done, best V1 {result['best_v1_auc']:.4f} at epoch {result['best_epoch']}")


if __name__ == "__main__":
    main()
