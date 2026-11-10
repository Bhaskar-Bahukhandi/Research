"""Build, save and verify G2's data: validation split, negatives, path counts and path samples.

    .venv/Scripts/python experiments/g2_build_data.py --dataset fb15k237           # build (clean tree)
    .venv/Scripts/python experiments/g2_build_data.py --dataset fb15k237 --check   # hash existing files
    .venv/Scripts/python experiments/g2_build_data.py --dataset fb15k237 --verify  # regenerate and compare

Implements notes/protocol.md ("Data", "Negatives", "Paths and cycles") and the dated
notes of 2026-11-07 (generating test negatives and paths is allowed; summarising them
by label is not) and 2026-11-08 (h = t triples; per-triple generators; manifest).

Outputs:
  saved/g2/<dataset>/      small, COMMITTED: V1/V2 split, negatives, fallback flags, and
                           manifest.json (SHA-256 of every file, input-data fingerprints,
                           library versions, code commit).
  data/g2/<dataset>/       large, gitignored, kept on Drive: path codes and exact path
                           counts for training triples (own edge excluded), validation and
                           test positives, and their negatives.

--verify is the protocol's check before the first test evaluation: it regenerates every
file into a temporary folder from the current code and compares the hashes with the
manifest COMMITTED in git (not the file on disk, which a rebuild could have replaced).

Random streams: negatives for positive i of a split use default_rng([0, i]); paths for
row i of an array use default_rng([0, i]), where negative triples are flattened row-major
(positive i, slot j -> row i * 10 + j). Files are .npy (byte-for-byte reproducible).
Printed statistics never compare positives with negatives on the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtvm.g2data import (TrainGraph, check_no_leakage, count_paths, sample_negatives,  # noqa: E402
                         sample_paths, split_validation)

# Everything the build's output depends on. Later G2 code (features, combiners, bootstrap)
# lives elsewhere in rtvm/ and may change without invalidating these files.
CODE_PATHS = ["rtvm/__init__.py", "rtvm/g2data.py", "experiments/g2_build_data.py"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_fingerprint(a: np.ndarray) -> str:
    """SHA-256 of an int64 C-order copy: identifies PyKEEN's id mapping and row order."""
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def git(*args: str) -> str:
    """Run git and fail closed: any git error stops the script instead of reading as 'no output'."""
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def versions() -> dict:
    import pykeen
    import scipy
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                pykeen=pykeen.get_version())


def load_splits(dataset: str):
    from pykeen.datasets import get_dataset
    ds = get_dataset(dataset=dataset)
    splits = [x.mapped_triples.numpy().astype(np.int64) for x in (ds.training, ds.validation, ds.testing)]
    return ds.num_entities, ds.num_relations, splits


def build(dataset: str, small: Path, large: Path) -> dict:
    """Write every file and return the manifest (not yet saved)."""
    n_ent, n_rel, (train, valid, test) = load_splits(dataset)
    known = set(map(tuple, np.concatenate([train, valid, test]).tolist()))
    graph = TrainGraph.from_training(train, n_ent)
    eval_triples = np.concatenate([valid, test])
    small.mkdir(parents=True, exist_ok=True)
    large.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[str, Path]] = {}

    def save(area: str, name: str, array: np.ndarray) -> None:
        path = (small if area == "saved" else large) / f"{name}.npy"
        np.save(path, array)
        files[name] = (area, path)

    t0 = time.time()
    v1, v2 = split_validation(len(valid), seed=0)
    save("saved", "valid_v1_index", v1)
    save("saved", "valid_v2_index", v2)

    stats = {}
    for split, pos in (("valid", valid), ("test", test)):
        neg, fb = sample_negatives(pos, train, known, n_ent, seed=0)
        save("saved", f"{split}_negatives", neg.astype(np.int32))
        save("saved", f"{split}_fallback", fb)
        stats[f"{split}_fallback_rate"] = float(fb.mean())              # protocol: report for valid and test
        print(f"{split}: {len(pos):,} positives, fallback share of negatives {fb.mean():.4f}", flush=True)
        for part, triples in ((f"{split}_pos", pos), (f"{split}_neg", neg.reshape(-1, 3))):
            save("data", f"{part}_counts", count_paths(graph, triples))
            paths = sample_paths(graph, triples)
            check_no_leakage(paths, train, eval_triples)
            save("data", f"{part}_paths", paths)
            print(f"  {part}: {len(triples):,} triples ({time.time() - t0:.0f} s)", flush=True)

    own = np.arange(len(train))
    save("data", "train_counts", count_paths(graph, train, own_edges=own))
    train_paths = sample_paths(graph, train, own_edges=own)
    check_no_leakage(train_paths, train, eval_triples, own_edges=own)
    save("data", "train_paths", train_paths)
    stats["train_covered_share"] = float((train_paths != 0).any(axis=(1, 2)).mean())
    print(f"train: {len(train):,} triples, {stats['train_covered_share']:.4f} have a path ({time.time() - t0:.0f} s)")

    return dict(
        dataset=dataset, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        code=dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", *CODE_PATHS))),
        versions=versions(),
        inputs=dict(entities=n_ent, relations=n_rel,
                    **{f"{name}_triples": len(a) for name, a in (("train", train), ("valid", valid), ("test", test))},
                    **{f"{name}_sha256": array_fingerprint(a) for name, a in (("train", train), ("valid", valid), ("test", test))}),
        rules="notes/protocol.md; dated notes 2026-11-07 and 2026-11-08", statistics=stats,
        files={name: dict(area=area, file=path.name, sha256=sha256(path),
                          shape=list(np.load(path, mmap_mode="r").shape), dtype=str(np.load(path, mmap_mode="r").dtype))
               for name, (area, path) in sorted(files.items())})


def compare(expected: dict, small: Path, large: Path) -> list[str]:
    return [name for name, f in expected["files"].items()
            if sha256((small if f["area"] == "saved" else large) / f["file"]) != f["sha256"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--saved", default=str(ROOT / "saved" / "g2"))
    ap.add_argument("--data", default=str(ROOT / "data" / "g2"))
    ap.add_argument("--allow-dirty", action="store_true", help="build from uncommitted code (dry runs only)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing manifest (never after it is committed)")
    ap.add_argument("--check", action="store_true", help="hash the existing files against the manifest")
    ap.add_argument("--verify", action="store_true", help="regenerate into a temporary folder and compare with "
                                                          "the manifest committed in git")
    args = ap.parse_args()
    small, large = Path(args.saved) / args.dataset, Path(args.data) / args.dataset
    manifest_path = small / "manifest.json"

    if args.check:
        manifest = json.loads(manifest_path.read_text())
        bad = compare(manifest, small, large)
        if bad:
            sys.exit(f"MISMATCH in {len(bad)} files: {bad}")
        print(f"all {len(manifest['files'])} files match {manifest_path}")
        return

    if args.verify:
        rel = manifest_path.resolve().relative_to(ROOT).as_posix()
        expected = json.loads(git("show", f"HEAD:{rel}"))          # fails if the manifest is not committed
        commit = expected["code"]["commit"]
        if expected["code"]["dirty"]:
            sys.exit("the committed manifest was built from uncommitted code: rebuild from a clean tree")
        git("cat-file", "-e", f"{commit}^{{commit}}")              # fails if that commit is missing (shallow clone)
        if git("status", "--porcelain", "--", *CODE_PATHS):
            sys.exit(f"uncommitted changes in {CODE_PATHS}: verification must run committed code")
        changed = git("diff", "--name-only", commit, "HEAD", "--", *CODE_PATHS)
        if changed:
            sys.exit(f"build code changed since the manifest's commit {commit[:7]}: {changed.split()}")
        if versions() != expected["versions"]:
            print(f"WARNING: library versions differ from the manifest: {versions()} vs {expected['versions']}")
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = build(args.dataset, Path(tmp) / "saved", Path(tmp) / "data")
            if regenerated["inputs"] != expected["inputs"]:
                sys.exit("input data differ from the manifest (PyKEEN id mapping or split order changed)")
            bad = compare(expected, Path(tmp) / "saved", Path(tmp) / "data")
        if bad:
            sys.exit(f"MISMATCH: regenerated files differ from the committed manifest: {bad}")
        print(f"verified: all {len(expected['files'])} regenerated files match the committed manifest")
        return

    if git("status", "--porcelain", "--", *CODE_PATHS) and not args.allow_dirty:
        sys.exit("code under rtvm/ or experiments/g2_build_data.py is uncommitted: commit first, so the manifest "
                 "names the code that made the files (or pass --allow-dirty for a dry run).")
    if manifest_path.exists() and not args.overwrite:
        sys.exit(f"{manifest_path} exists; use --check or --verify, or --overwrite before it is committed")
    manifest = build(args.dataset, small, large)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
