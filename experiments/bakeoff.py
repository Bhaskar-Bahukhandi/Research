"""Phase 3 page-rotation bake-off driver (notes/protocol.md, seventh note of 2026-11-22).

    python experiments/bakeoff.py run --candidate rotate --trials 0-19 [--device cuda] [--max-hours H]
    python experiments/bakeoff.py run --candidate rotate --seeds 1,2       # after the 20 trials
    python experiments/bakeoff.py select                                   # best trial per candidate
    python experiments/bakeoff.py decide                                   # V_select and the rule

Runs go to runs/bakeoff/<candidate>/{trial_XX,seed_S}.{json,_best.pt,ckpt,log}. Seed 0 of a
candidate is its best trial; seeds 1 and 2 rerun the best trial's values. Trials and seeds are
scored on V_tune only; `decide` is the only command that reads V_select, and only after every run
of every candidate is complete. No MetaQA question file is read.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase3 import bakeoff  # noqa: E402
from phase3.metaqa import load_data  # noqa: E402

MARGIN = 0.01          # the rule's MRR margin over RotatE (seventh note)
DEFAULT = "rotate"
SEEDS = (0, 1, 2)
CODE_PATHS = ["phase3", "experiments/bakeoff.py", "configs", "rtvm"]


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def code_state() -> dict:
    return dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", *CODE_PATHS)))


def parse_range(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def records(out: Path, candidate: str) -> dict[str, dict]:
    found = {}
    for p in sorted((out / candidate).glob("*.json")):
        found[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return found


def best_trial(recs: dict[str, dict]) -> int:
    """Highest best V_tune MRR among the 20 complete trials; ties go to the lower index."""
    missing = [i for i in range(bakeoff.N_TRIALS) if not recs.get(f"trial_{i:02d}", {}).get("complete")]
    if missing:
        raise SystemExit(f"trials not complete: {missing}")
    scores = {i: recs[f"trial_{i:02d}"]["best_v_tune_mrr"] for i in range(bakeoff.N_TRIALS)}
    finite = [i for i, s in scores.items() if s is not None and math.isfinite(s)]
    if not finite:
        raise SystemExit("no trial has a finite V_tune MRR")
    return max(finite, key=lambda i: (scores[i], -i))


def run_jobs(args) -> None:
    import torch
    state = code_state()
    if state["dirty"] and not args.allow_dirty:
        sys.exit("code has uncommitted changes: commit first (or --allow-dirty for a dry run)")
    data = load_data(args.kb, args.saved)
    trials = bakeoff.trials(args.candidate, ROOT)
    out = Path(args.out) / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    if args.trials:
        jobs = [(f"trial_{i:02d}", trials[i], 0, i) for i in parse_range(args.trials)]
    else:
        best = best_trial(records(Path(args.out), args.candidate))
        jobs = [(f"seed_{s}", trials[best], s, best) for s in parse_range(args.seeds) if s != 0]
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    for name, values, seed, trial in jobs:
        rec_path = out / f"{name}.json"
        if rec_path.exists() and json.loads(rec_path.read_text(encoding="utf-8")).get("complete"):
            print(f"{args.candidate}/{name}: complete, skipped")
            continue
        logf = open(out / f"{name}.log", "a", encoding="utf-8")

        def log(msg, logf=logf, name=name):
            line = f"[{args.candidate}/{name}] {msg}"
            print(line, flush=True)
            logf.write(line + "\n")
            logf.flush()
        log(f"start {time.strftime('%Y-%m-%d %H:%M:%S')} on {args.device}; trial {trial}, seed {seed}; {values}")
        res = bakeoff.train(args.candidate, values, seed, data, args.device, out / f"{name}.ckpt", log, deadline)
        model = res.pop("model")
        torch.save(model.state_dict(), out / f"{name}_best.pt")
        res.update(name=name, trial=trial, git=state, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   g2_config=bakeoff.CONFIGS[bakeoff.G2_SOURCE[args.candidate]],
                   g2_config_sha256=bakeoff.file_sha256(ROOT / bakeoff.CONFIGS[bakeoff.G2_SOURCE[args.candidate]]))
        rec_path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        log(f"{'done' if res['complete'] else 'stopped at the deadline'}: best V_tune MRR "
            f"{res['best_v_tune_mrr']:.4f} at epoch {res['best_epoch']}")
        logf.close()
        if deadline is not None and time.time() >= deadline:
            break


def select(args) -> None:
    for c in bakeoff.CANDIDATES:
        recs = records(Path(args.out), c)
        try:
            b = best_trial(recs)
        except SystemExit as e:
            print(f"{c}: {e}")
            continue
        print(f"{c}: best trial {b}, V_tune MRR {recs[f'trial_{b:02d}']['best_v_tune_mrr']:.4f}, "
              f"values {recs[f'trial_{b:02d}']['values']}")


def apply_rule(mrrs: dict[str, list[float]]) -> tuple[dict, dict, str]:
    """The seventh note's rule. mrrs: candidate -> its three V_select MRRs (seeds 0, 1, 2).

    RotatE unless another candidate's mean is at least MARGIN above RotatE's mean and each of its
    seeds is above RotatE's mean; if both others qualify, the higher mean wins.
    """
    mean = {c: float(np.mean(v)) for c, v in mrrs.items()}
    qualifies = {c: bool(mean[c] >= mean[DEFAULT] + MARGIN and all(x > mean[DEFAULT] for x in v))
                 for c, v in mrrs.items() if c != DEFAULT}
    winners = [c for c, q in qualifies.items() if q]
    return mean, qualifies, (max(winners, key=lambda c: mean[c]) if winners else DEFAULT)


def decide(args) -> None:
    import torch
    out = Path(args.out)
    target = out / "decision.json"
    if target.exists():
        sys.exit(f"{target} exists: the decision is made once")
    data = load_data(args.kb, args.saved)
    trials = {c: bakeoff.trials(c, ROOT) for c in bakeoff.CANDIDATES}
    runs = {}
    for c in bakeoff.CANDIDATES:                      # every run of every candidate, checked before V_select is read
        recs = records(out, c)
        best = best_trial(recs)
        for s in SEEDS:
            name = f"trial_{best:02d}" if s == 0 else f"seed_{s}"
            rec = recs.get(name)
            if not rec or not rec.get("complete") or rec["seed"] != s or rec["values"] != trials[c][best]:
                sys.exit(f"{c}/{name}: missing, incomplete or not the best trial's values")
            runs[(c, s)] = (name, rec)
    device = torch.device(args.device)
    rel = np.concatenate([data.v_select[:, 1], data.v_select[:, 1]])     # ranks: tail queries, then head queries
    result = {}
    for (c, s), (name, rec) in runs.items():
        model = bakeoff.build(c, rec["values"], data.kb.num_entities, data.kb.num_relations,
                              data.kb.entities, data.kb.relations)
        model.load_state_dict(torch.load(out / c / f"{name}_best.pt", map_location="cpu"))
        res = bakeoff.evaluate(c, model.to(device), data.v_select, data)
        ranks = res.pop("ranks")
        res["mrr_per_relation"] = {data.kb.relations[r]: float(np.mean(1 / ranks[rel == r]))
                                   for r in range(data.kb.num_relations)}
        res.update(run=f"{c}/{name}", seed=s, rank=rec["values"].get("rank"), train_seconds=rec["train_seconds"],
                   best_v_tune_mrr=rec["best_v_tune_mrr"])
        result.setdefault(c, {})[s] = res
        print(f"{c} seed {s}: V_select MRR {res['mrr']:.4f}", flush=True)
    mean, qualifies, chosen = apply_rule({c: [result[c][s]["mrr"] for s in SEEDS] for c in bakeoff.CANDIDATES})
    decision = dict(kind="Phase 3 page-rotation bake-off, notes/protocol.md seventh note of 2026-11-22",
                    rule=f"{DEFAULT} unless another candidate's mean V_select MRR is >= {MARGIN} above {DEFAULT}'s "
                         f"mean and each of its seeds is above {DEFAULT}'s mean; if both qualify, the higher mean",
                    mean_v_select_mrr=mean, qualifies=qualifies, chosen=chosen, runs=result,
                    data_provenance=data.provenance, git=code_state(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    target.write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    print(f"means {mean}; qualifies {qualifies}; chosen: {chosen}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "select", "decide"])
    ap.add_argument("--candidate", choices=bakeoff.CANDIDATES)
    ap.add_argument("--trials", help="e.g. 0-19 or 3,5")
    ap.add_argument("--seeds", help="1,2 (after the trials)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-hours", type=float)
    ap.add_argument("--kb", default=str(ROOT / "data/metaqa/kb.txt"))
    ap.add_argument("--saved", default=str(ROOT / "saved/metaqa"))
    ap.add_argument("--out", default=str(ROOT / "runs/bakeoff"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    if args.command == "run":
        if not args.candidate or bool(args.trials) == bool(args.seeds):
            sys.exit("run needs --candidate and exactly one of --trials, --seeds")
        run_jobs(args)
    elif args.command == "select":
        select(args)
    else:
        decide(args)


if __name__ == "__main__":
    main()
