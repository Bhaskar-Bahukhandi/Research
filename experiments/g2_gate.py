"""G2 primary test: compute features, fit the combiners, bootstrap, decide (notes/protocol.md).

    # model-free and baseline features on V2 (allowed any time; never summarised by label here)
    python experiments/g2_gate.py features --dataset fb15k237 --split v2 --models rotate holonomy --device cuda
    # negative control on V2 only: noise in place of the holonomy score (no test split, no holonomy model)
    python experiments/g2_gate.py smoke --dataset fb15k237
    # after the freeze note (see below): V2 and test features from the frozen code, then the decision
    python experiments/g2_gate.py features --dataset fb15k237 --split v2 --models rotate holonomy --device cuda
    python experiments/g2_gate.py features --dataset fb15k237 --split test --models rotate holonomy --device cuda
    python experiments/g2_gate.py decide --dataset fb15k237

The rules and the implementation choices are in rtvm/g2gate.py. Models are read from the G2
search output (--runs, as written by experiments/g2_search.py): seed 0 is the best trial
(rtvm.g2search.select_best), seeds 1 and 2 are seed_1 and seed_2, each from its *_best.pt.
Every run must carry the SHA-256 of the committed config, and the seed runs must use the
best trial's values.

Test split guard (protocol, Timing). `features --split test` and `decide` refuse to run unless:
  * the git tree is clean;
  * notes/protocol.md at HEAD has a line that is exactly "G2 test freeze: commit <40 hex>"
    (optionally as a list item), and all such lines name the same commit;
  * that commit is an ancestor of HEAD, and nothing under FROZEN_PATHS changed since it;
  * the manifest read is byte-identical to the one committed at HEAD.
`decide` also refuses any feature file that was not written by clean code identical to the
current FROZEN_PATHS (every .npz records its provenance), and refuses to overwrite gate.json.
So V2 features made before the freeze must be recomputed after it.

Before the first test evaluation, also run `experiments/g2_build_data.py --dataset D --verify`
(the 2026-11-08 note); this script checks every file's hash against the manifest and runs
check_no_leakage on every path array it loads.

Outputs, under --out/<dataset>/: <split>_structure.npz, <split>_<model>_seed<s>.npz,
smoke.json and gate.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtvm.g2gate import B1, STRUCTURE, null_rotations, primary_test, rotation_table, structure_features, triple_holonomy  # noqa: E402,E501

FREEZE_RE = re.compile(r"^[ \t]*(?:[-*][ \t]+)?G2 test freeze: commit ([0-9a-f]{40})[ \t]*$", re.MULTILINE)
FROZEN_PATHS = ["rtvm", "experiments/g2_build_data.py", "experiments/g2_gate.py", "experiments/g2_search.py",
                "configs", "saved/g2"]
SPLITS = ("v2", "test")
SEEDS = (0, 1, 2)
S11_DATASETS = ("fb15k237",)            # S11 is defined on the primary dataset only (note 2026-11-17)
MODEL_CONFIG = {"rotate": "rotate", "holonomy": "holonomy"}     # gate model name -> rtvm.g2search.CONFIGS key


# ---------------------------------------------------------------- guard and provenance

def git(*args: str) -> str:
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def freeze_commit(protocol_text: str) -> str:
    """The commit named by the "G2 test freeze: commit <hash>" line(s); error if none or two different."""
    found = set(FREEZE_RE.findall(protocol_text))
    if len(found) != 1:
        raise ValueError(f"notes/protocol.md must name exactly one G2 test freeze commit, found {len(found)}")
    return found.pop()


def check_test_allowed() -> str:
    """Exit unless the Timing rule is met (module docstring). Returns the freeze commit."""
    if git("status", "--porcelain"):
        sys.exit("the test split needs a clean git tree")
    try:
        commit = freeze_commit(git("show", "HEAD:notes/protocol.md"))
    except ValueError as e:
        sys.exit(f"test split refused: {e}")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"]).returncode != 0:
        sys.exit(f"freeze commit {commit[:7]} is not an ancestor of HEAD")
    changed = git("diff", "--name-only", commit, "HEAD", "--", *FROZEN_PATHS)
    if changed:
        sys.exit(f"frozen files changed since {commit[:7]}: {changed.split()}")
    return commit


def code_state() -> dict:
    """Identity of the code and saved data that make features: HEAD, whether FROZEN_PATHS are
    dirty, and a hash of their committed tree (equal across commits that leave them unchanged)."""
    tree = git("ls-tree", "-r", "HEAD", "--", *FROZEN_PATHS)
    return dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", *FROZEN_PATHS)),
                frozen_tree=hashlib.sha256(tree.encode()).hexdigest())


def versions() -> dict:
    import pykeen
    import scipy
    import sklearn
    import torch
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, torch=torch.__version__, pykeen=pykeen.get_version())


def check_provenance(records: dict[str, dict], current: dict) -> None:
    """Every feature file must come from clean code with the current frozen tree."""
    for name, rec in records.items():
        if rec.get("dirty", True) or rec.get("frozen_tree") != current["frozen_tree"]:
            sys.exit(f"{name} was not made by the current frozen code ({rec.get('head', '?')[:7]}, "
                     f"dirty={rec.get('dirty')}): recompute it with `features`")


# ---------------------------------------------------------------- data

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_split(dataset: str, split: str, saved: Path, data: Path, allow_test: bool = False,
               committed_manifest: str | None = None) -> dict:
    """Positives, negatives, fallback flags, path counts and codes of V2 or test, hash-checked.

    committed_manifest: the manifest text committed at HEAD; when given, the manifest on disk
    must equal it byte for byte (required by the script whenever the test split is touched).
    """
    from pykeen.datasets import get_dataset

    from rtvm.g2data import TrainGraph, check_no_leakage

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("the test split is loaded only after check_test_allowed()")
    small, large = saved / dataset, data / dataset
    manifest_path = small / "manifest.json"
    if committed_manifest is not None and manifest_path.read_bytes().replace(b"\r\n", b"\n") != \
            committed_manifest.encode().replace(b"\r\n", b"\n"):
        raise ValueError(f"{manifest_path} differs from the manifest committed at HEAD")
    manifest = json.loads(manifest_path.read_text())
    ds = get_dataset(dataset=dataset)
    train, valid, test = (x.mapped_triples.numpy().astype(np.int64) for x in (ds.training, ds.validation, ds.testing))
    for name, arr in (("train", train), ("valid", valid), ("test", test)):
        if hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest() != manifest["inputs"][f"{name}_sha256"]:
            raise ValueError(f"{dataset} {name} triples differ from the manifest")

    def load(name):
        entry = manifest["files"][name]
        path = (small if entry["area"] == "saved" else large) / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match {manifest_path}")
        return np.load(path)

    prefix = "valid" if split == "v2" else "test"
    neg = load(f"{prefix}_negatives").astype(np.int64)
    m, k = neg.shape[:2]
    out = dict(pos=valid if split == "v2" else test, neg=neg, fallback=load(f"{prefix}_fallback"),
               pos_counts=load(f"{prefix}_pos_counts"), pos_paths=load(f"{prefix}_pos_paths"),
               neg_counts=load(f"{prefix}_neg_counts").reshape(m, k, 3))
    out["neg_paths"] = load(f"{prefix}_neg_paths").reshape(m, k, *out["pos_paths"].shape[1:])
    if split == "v2":
        idx = load("valid_v2_index")
        out = {key: value[idx] for key, value in out.items()}
    eval_triples = np.concatenate([valid, test])
    check_no_leakage(out["pos_paths"], train, eval_triples)
    check_no_leakage(out["neg_paths"], train, eval_triples)
    out.update(train=train, num_entities=ds.num_entities, num_relations=ds.num_relations, training=ds.training,
               degree=TrainGraph.from_training(train, ds.num_entities).degree(),
               manifest_sha256=sha256(manifest_path))
    return out


def steps(d: dict, codes: np.ndarray) -> np.ndarray:
    from rtvm.holonomy_model import steps_from_codes
    return steps_from_codes(codes, d["train"][:, 1], d["num_relations"])


def flat(x: np.ndarray, trailing: int) -> np.ndarray:
    """(m, k, ...) -> (m * k, ...), keeping `trailing` trailing axes."""
    return x.reshape(-1, *x.shape[x.ndim - trailing:]) if trailing else x.reshape(-1)


# ---------------------------------------------------------------- features

def structure_and_null(d: dict, device: str) -> dict:
    m, k = d["neg"].shape[:2]
    table = rotation_table(null_rotations(d["num_relations"]))
    out = {}
    for part, triples, counts, codes in (("pos", d["pos"], d["pos_counts"], d["pos_paths"]),
                                         ("neg", flat(d["neg"], 1), flat(d["neg_counts"], 1), flat(d["neg_paths"], 2))):
        f = structure_features(triples, counts, codes, d["degree"])
        null, covered = triple_holonomy(table, triples[:, 1], steps(d, codes), device)
        if not np.array_equal(covered, f["covered"] > 0):
            raise ValueError("null holonomy coverage differs from the path counts")
        f["null_holonomy"] = null
        for key, value in f.items():
            out[f"{part}/{key}"] = value if part == "pos" else value.reshape(m, k)
    out["relations"] = d["pos"][:, 1]
    out["triples"] = d["pos"]                      # (h, r, t) of the positives, for S11 (note 2026-11-17)
    out["fallback"] = d["fallback"]
    return out


def model_runs(runs: Path, dataset: str, model: str) -> dict[int, Path]:
    """Seed -> run-file stem: seed 0 is the best of the 20 trials, seeds 1 and 2 its seed runs.

    Refuses runs whose config hash differs from the committed config, and seed runs whose
    values or seed differ from the best trial's.
    """
    from rtvm.g2search import CONFIGS, N_TRIALS, file_sha256, select_best
    run_dir = runs / dataset / model
    config_hash = file_sha256(ROOT / CONFIGS[MODEL_CONFIG[model]])
    search = json.loads((run_dir / "search.json").read_text())
    if search["config_sha256"] != config_hash:
        sys.exit(f"{run_dir}: search made with config {search['config_sha256'][:12]}, committed is {config_hash[:12]}")
    results = {i: json.loads((run_dir / f"trial_{i:02d}.json").read_text()) for i in range(N_TRIALS)
               if (run_dir / f"trial_{i:02d}.json").exists()}
    if len(results) < N_TRIALS:
        sys.exit(f"{run_dir}: only {len(results)} of {N_TRIALS} trials finished")
    best = select_best(results)
    stems = {0: run_dir / f"trial_{best:02d}", 1: run_dir / "seed_1", 2: run_dir / "seed_2"}
    for s, stem in stems.items():
        rec = json.loads(stem.with_suffix(".json").read_text())
        if not rec.get("complete") or not Path(f"{stem}_best.pt").exists():
            sys.exit(f"{stem}: run incomplete or {stem.name}_best.pt missing")
        if rec.get("config_sha256") != config_hash or rec.get("seed") != s or rec.get("values") != search["trials"][best]:
            sys.exit(f"{stem}: config hash, seed or values do not match trial {best} of the committed search")
    return stems


def model_scores(d: dict, model: str, stem: Path, device: str) -> dict:
    """Feature values of one trained model: RotatE's score_hrt, or the triple holonomy score."""
    import torch
    rec = json.loads(stem.with_suffix(".json").read_text())
    weights = torch.load(f"{stem}_best.pt", map_location=device, weights_only=True)
    m, k = d["neg"].shape[:2]
    if model == "rotate":
        from rtvm.baselines import BaselineConfig, build_model, score_triples
        net = build_model(BaselineConfig(**rec["config"]), d["training"]).to(device)
        net.load_state_dict(weights)
        pos, neg = score_triples(net, d["pos"], device), score_triples(net, flat(d["neg"], 1), device)
    else:
        from rtvm.holonomy_model import HolonomyModel
        net = HolonomyModel(d["num_entities"], d["num_relations"], rank=rec["config"]["rank"]).to(device)
        net.load_state_dict(weights)
        with torch.no_grad():
            table = net.table()
        pos, _ = triple_holonomy(table, d["pos"][:, 1], steps(d, d["pos_paths"]), device)
        neg, _ = triple_holonomy(table, flat(d["neg"], 1)[:, 1], steps(d, flat(d["neg_paths"], 2)), device)
    return dict(pos=pos, neg=neg.reshape(m, k), source=str(stem), weights_sha256=sha256(Path(f"{stem}_best.pt")),
                config_sha256=rec["config_sha256"])


def save_npz(path: Path, arrays: dict, provenance: dict) -> None:
    """Atomic: a killed process never leaves a truncated file that looks finished."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays, provenance=json.dumps(provenance))
    tmp.replace(path)


# ---------------------------------------------------------------- assemble

def assemble(out_dir: Path, split: str, models=("rotate", "holonomy"), stand_in: dict | None = None) -> dict:
    """The dict primary_test expects, from the saved feature files, with each file's provenance.
    stand_in replaces a model's files with given arrays (smoke test only)."""
    S = np.load(out_dir / f"{split}_structure.npz")
    feats = {"pos": {k: S[f"pos/{k}"] for k in STRUCTURE}, "neg": {k: S[f"neg/{k}"] for k in STRUCTURE},
             "relations": S["relations"], "triples": S["triples"], "sources": {},
             "provenance": {f"{split}_structure.npz": json.loads(str(S["provenance"]))}}
    for model in models:
        for s in SEEDS:
            if stand_in and model in stand_in:
                pos, neg = stand_in[model][s]
            else:
                name = f"{split}_{model}_seed{s}.npz"
                M = np.load(out_dir / name)
                pos, neg = M["pos"], M["neg"]
                feats["sources"][f"{model}:{s}"] = dict(source=str(M["source"]), weights_sha256=str(M["weights_sha256"]),
                                                        config_sha256=str(M["config_sha256"]))
                feats["provenance"][name] = json.loads(str(M["provenance"]))
            feats["pos"][f"{model}:{s}"], feats["neg"][f"{model}:{s}"] = pos, neg
    return feats


def subset(split: dict, rows: np.ndarray) -> dict:
    return dict(pos={k: v[rows] for k, v in split["pos"].items()}, neg={k: v[rows] for k, v in split["neg"].items()},
                relations=split["relations"][rows], triples=split["triples"][rows], sources=split["sources"])


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def main() -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # strict FP32 (note 2026-11-07)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features", "smoke", "decide"])
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--split", choices=SPLITS, default="v2")
    ap.add_argument("--models", nargs="*", default=[], choices=sorted(MODEL_CONFIG))
    ap.add_argument("--runs", default=str(ROOT / "runs/g2"))
    ap.add_argument("--out", default=str(ROOT / "runs/g2gate"))
    ap.add_argument("--saved", default=str(ROOT / "saved/g2"))
    ap.add_argument("--data", default=str(ROOT / "data/g2"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    touches_test = args.command == "decide" or (args.command == "features" and args.split == "test")
    freeze = check_test_allowed() if touches_test else None
    state = code_state()
    t0 = time.time()

    if args.command == "features":
        committed = git("show", f"HEAD:saved/g2/{args.dataset}/manifest.json") if freeze else None
        d = load_split(args.dataset, args.split, Path(args.saved), Path(args.data), allow_test=freeze is not None,
                       committed_manifest=committed)
        prov = dict(**state, split=args.split, manifest_sha256=d["manifest_sha256"], versions=versions(),
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
        target = out_dir / f"{args.split}_structure.npz"
        save_npz(target, structure_and_null(d, args.device), prov)
        print(f"wrote {target} ({time.time() - t0:.0f} s)", flush=True)
        for model in args.models:
            for s, stem in model_runs(Path(args.runs), args.dataset, model).items():
                save_npz(out_dir / f"{args.split}_{model}_seed{s}.npz", model_scores(d, model, stem, args.device), prov)
                print(f"wrote {args.split}_{model}_seed{s}.npz from {stem.name} ({time.time() - t0:.0f} s)", flush=True)
        return

    if args.command == "smoke":
        # Negative control on V2 only: fit on one half of the V2 positives, evaluate on the other,
        # with Gaussian noise in place of the holonomy score. No holonomy model, no test split.
        base = assemble(out_dir, "v2", models=("rotate",))
        m, k = base["neg"]["covered"].shape
        noise = {"holonomy": {s: (np.random.default_rng([1, s]).standard_normal(m),
                                  np.random.default_rng([2, s]).standard_normal((m, k))) for s in SEEDS}}
        split = assemble(out_dir, "v2", models=("rotate", "holonomy"), stand_in=noise)
        perm = np.random.default_rng(0).permutation(m)
        fit_rows, eval_rows = np.sort(perm[: m // 2]), np.sort(perm[m // 2:])
        res = primary_test(subset(split, fit_rows), subset(split, eval_rows), two_way=args.dataset in S11_DATASETS)
        res.update(kind="smoke: V2 halves, Gaussian noise as the holonomy feature; not a G2 result",
                   expected="the CI contains 0", sources=split["sources"], provenance=split["provenance"])
        write_json(out_dir / "smoke.json", res)
        print(f"smoke: B0 grouped AUC per seed {[round(res['seeds'][s]['B0']['grouped_auc'], 4) for s in SEEDS]}; "
              f"delta-bar {res['delta_bar']:+.5f}, CI {res['ci95'][0]:+.5f} .. {res['ci95'][1]:+.5f} -> {res['decision']}")
        return

    # decide: the primary test, once
    gate_file = out_dir / "gate.json"
    if gate_file.exists():
        sys.exit(f"{gate_file} exists: the test split is evaluated once (protocol, General rules)")
    v2, test = assemble(out_dir, "v2"), assemble(out_dir, "test")
    check_provenance({**v2["provenance"], **test["provenance"]}, state)
    manifests = {p["manifest_sha256"] for p in {**v2["provenance"], **test["provenance"]}.values()}
    if len(manifests) != 1:
        sys.exit("feature files were made from different manifests")
    for key in v2["sources"]:
        if v2["sources"][key] != test["sources"][key]:
            sys.exit(f"{key}: V2 and test features came from different model runs")
    res = primary_test(v2, test, two_way=args.dataset in S11_DATASETS)
    res.update(dataset=args.dataset, freeze_commit=freeze, head=state["head"], frozen_tree=state["frozen_tree"],
               manifest_sha256=manifests.pop(), features=list(B1), sources=test["sources"],
               provenance={**v2["provenance"], **test["provenance"]}, versions=versions(),
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(gate_file, res)
    print(f"{args.dataset}: delta per seed {[round(res['seeds'][s]['delta'], 5) for s in SEEDS]}, delta-bar "
          f"{res['delta_bar']:+.5f}, 95% CI {res['ci95'][0]:+.5f} .. {res['ci95'][1]:+.5f} -> {res['decision']}")
    s11 = res.get("s11")
    if s11 and "error" in s11:
        print(f"S11 failed: {s11['error']} (the primary result above stands)")
    elif s11:
        print(f"S11 two-way CI {s11['ci95'][0]:+.5f} .. {s11['ci95'][1]:+.5f} ({s11['n_entities']} entities"
              f"{', narrower than the primary CI' if s11['narrower_than_primary'] else ''}) "
              f"-> reported as: {s11['report']}")


if __name__ == "__main__":
    main()
