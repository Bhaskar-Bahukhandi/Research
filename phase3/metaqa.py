"""MetaQA's knowledge base (kb.txt) and the bake-off's link-prediction split.

Protocol: notes/protocol.md, seventh note of 2026-11-22.
  * kb.txt from the MetaQA dataset (github.com/yuyuz/MetaQA links its Drive folder); one
    `subject|relation|object` triple per line.
  * Duplicates are removed. Entities and relations get ids in sorted (code-point) order.
  * Split: a seed-0 permutation puts 5% of the triples in V_tune, the next 5% in V_select, the
    rest in training. A held-out triple whose head or tail is absent from those training triples
    goes back to training (one pass; moving triples into training only adds entities to it, so
    every triple still held out afterwards has both entities in training).
  * Ranks are filtered against every triple of the knowledge base.
No MetaQA question file is read here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPLIT_SEED = 0
HELD_OUT_FRACTION = 0.05
SPLITS = ("train", "v_tune", "v_select")


@dataclass
class KB:
    entities: list[str]
    relations: list[str]
    triples: np.ndarray            # (n, 3) int64 ids, unique, sorted by (h, r, t)
    n_lines: int                   # triple lines in the file, duplicates included

    @property
    def num_entities(self) -> int:
        return len(self.entities)

    @property
    def num_relations(self) -> int:
        return len(self.relations)


def parse_kb(text: str) -> KB:
    """Parse kb.txt's text. Every non-empty line must hold exactly two '|'."""
    rows = []
    for i, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError(f"line {i}: expected subject|relation|object, got {line!r}")
        rows.append(tuple(parts))
    unique = sorted(set(rows))
    entities = sorted({h for h, _, _ in unique} | {t for _, _, t in unique})
    relations = sorted({r for _, r, _ in unique})
    e_id = {e: i for i, e in enumerate(entities)}
    r_id = {r: i for i, r in enumerate(relations)}
    triples = np.array([(e_id[h], r_id[r], e_id[t]) for h, r, t in unique], dtype=np.int64).reshape(-1, 3)
    order = np.lexsort((triples[:, 2], triples[:, 1], triples[:, 0]))
    return KB(entities, relations, triples[order], len(rows))


def split_kb(triples: np.ndarray, seed: int = SPLIT_SEED, fraction: float = HELD_OUT_FRACTION) -> dict:
    """Index arrays into `triples` for train, v_tune and v_select, plus how many went back."""
    n = len(triples)
    k = int(round(fraction * n))
    perm = np.random.default_rng(seed).permutation(n)
    held = {"v_tune": perm[:k], "v_select": perm[k:2 * k]}
    train = perm[2 * k:]
    known = np.zeros(int(triples[:, [0, 2]].max()) + 1, dtype=bool)
    known[triples[train, 0]] = True
    known[triples[train, 2]] = True
    moved = {}
    back = []
    for name, idx in held.items():
        ok = known[triples[idx, 0]] & known[triples[idx, 2]]
        moved[name] = int((~ok).sum())
        back.append(idx[~ok])
        held[name] = np.sort(idx[ok])
    train = np.sort(np.concatenate([train, *back]))
    if len(np.unique(triples[train, 1])) != len(np.unique(triples[:, 1])):
        raise ValueError("a relation type has no training triple")
    return dict(train=train, **held, moved_back=moved)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a, dtype=np.int64).tobytes())


@dataclass
class BakeoffData:
    kb: KB
    train: np.ndarray              # (n, 3) id triples
    v_tune: np.ndarray
    v_select: np.ndarray
    provenance: dict


def load_data(kb_path: str | Path = "data/metaqa/kb.txt", saved: str | Path = "saved/metaqa") -> BakeoffData:
    """kb.txt and the committed split, each checked against saved/metaqa/manifest.json."""
    saved = Path(saved)
    manifest_path = saved / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = Path(kb_path).read_bytes()
    if sha256_bytes(raw) != manifest["kb"]["sha256"]:
        raise ValueError(f"{kb_path} is not the kb.txt in {manifest_path}")
    kb = parse_kb(raw.decode("utf-8"))
    if array_sha256(kb.triples) != manifest["kb"]["triples_sha256"]:
        raise ValueError("parsed triples differ from the manifest: the parser changed")
    entry = manifest["split"]
    path = saved / entry["file"]
    if sha256_bytes(path.read_bytes()) != entry["sha256"]:
        raise ValueError(f"{path} does not match {manifest_path}")
    z = np.load(path)
    parts = {name: kb.triples[z[name]] for name in SPLITS}
    return BakeoffData(kb=kb, **parts, provenance=dict(
        manifest_sha256=sha256_bytes(manifest_path.read_bytes()), data_code_commit=manifest["code"]["commit"]))
