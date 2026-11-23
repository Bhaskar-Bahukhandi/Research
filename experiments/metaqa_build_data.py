"""Build the Phase 3 bake-off's data record: saved/metaqa/{manifest.json, split.npz}.

    python experiments/metaqa_build_data.py            # writes, refuses to overwrite
    python experiments/metaqa_build_data.py --check    # rebuilds in memory, compares with the committed files

Protocol: notes/protocol.md, seventh note of 2026-11-22. Reads only data/metaqa/kb.txt (never a
MetaQA question file). The manifest records kb.txt's sha256 and size, the counts of entities,
relation types and triples, and the split; it is committed before any bake-off training.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase3.metaqa import (HELD_OUT_FRACTION, SPLIT_SEED, SPLITS, array_sha256, parse_kb,  # noqa: E402
                           sha256_bytes, split_kb)

CODE_PATHS = ["phase3/metaqa.py", "experiments/metaqa_build_data.py"]
SOURCE = ("MetaQA (Zhang et al., AAAI 2018): kb.txt from the Google Drive folder linked by "
          "github.com/yuyuz/MetaQA (Drive file id 0B-36Uca2AvwhMUFHeUpTRWp5eUU); Creative Commons licence per its README")


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def split_bytes(split: dict) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{name: split[name].astype(np.int64) for name in SPLITS})
    return buf.getvalue()


def same_split(saved_npz: Path, split: dict) -> bool:
    """Whether the committed split.npz holds the same index arrays as a fresh split. The arrays are
    what must reproduce; the zip container's bytes differ between NumPy builds (seen on Kaggle,
    2026-11-23), and its sha256 only guards the committed file (phase3.metaqa.load_data)."""
    z = np.load(saved_npz)
    return all(np.array_equal(z[name], split[name]) for name in SPLITS)


def matches_record(out: Path, manifest: dict, split: dict) -> bool:
    """A fresh build agrees with the committed record: same kb entry, same split entry apart from the
    container hash, and the same arrays."""
    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    strip = lambda d: {k: v for k, v in d.items() if k != "sha256"}          # noqa: E731
    return (saved["kb"] == manifest["kb"] and strip(saved["split"]) == strip(manifest["split"])
            and same_split(out / saved["split"]["file"], split))


def build(kb_path: Path) -> tuple[dict, bytes, dict]:
    raw = kb_path.read_bytes()
    kb = parse_kb(raw.decode("utf-8"))
    split = split_kb(kb.triples)
    blob = split_bytes(split)
    per_relation = {kb.relations[r]: int(n) for r, n in zip(*np.unique(kb.triples[:, 1], return_counts=True))}
    counts = {name: dict(triples=int(len(split[name])),
                         entities=int(len(np.unique(kb.triples[split[name]][:, [0, 2]]))))
              for name in SPLITS}
    manifest = dict(
        source=SOURCE,
        rules="notes/protocol.md, seventh note of 2026-11-22",
        kb=dict(file="data/metaqa/kb.txt", bytes=len(raw), sha256=sha256_bytes(raw),
                lines=kb.n_lines, triples=int(len(kb.triples)), duplicates_removed=kb.n_lines - int(len(kb.triples)),
                entities=kb.num_entities, relations=kb.num_relations, triples_per_relation=per_relation,
                triples_sha256=array_sha256(kb.triples),
                entities_sha256=sha256_bytes("\n".join(kb.entities).encode("utf-8")),
                relations_sha256=sha256_bytes("\n".join(kb.relations).encode("utf-8"))),
        split=dict(file="split.npz", sha256=sha256_bytes(blob), seed=SPLIT_SEED, held_out_fraction=HELD_OUT_FRACTION,
                   counts=counts, moved_back_to_train=split["moved_back"]),
        code=dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain", "--", *CODE_PATHS))),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__),
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    return manifest, blob, split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", default=str(ROOT / "data/metaqa/kb.txt"))
    ap.add_argument("--out", default=str(ROOT / "saved/metaqa"))
    ap.add_argument("--check", action="store_true", help="compare a fresh build with the committed record")
    ap.add_argument("--allow-dirty", action="store_true", help="build from uncommitted code (dry runs only)")
    args = ap.parse_args()
    out = Path(args.out)
    manifest, blob, split = build(Path(args.kb))
    if args.check:
        same = matches_record(out, manifest, split)
        print("check:", "PASS" if same else "FAIL")
        sys.exit(0 if same else 1)
    if manifest["code"]["dirty"] and not args.allow_dirty:
        sys.exit("phase3/metaqa.py or this script has uncommitted changes: commit first")
    if (out / "manifest.json").exists():
        sys.exit(f"{out / 'manifest.json'} exists: the record is written once")
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.npz").write_bytes(blob)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    k = manifest["kb"]
    print(f"{k['triples']:,} triples ({k['duplicates_removed']:,} duplicates removed), {k['entities']:,} entities, "
          f"{k['relations']} relation types")
    for name, c in manifest["split"]["counts"].items():
        print(f"  {name}: {c['triples']:,} triples")
    print("  moved back to train:", manifest["split"]["moved_back_to_train"])


if __name__ == "__main__":
    main()
