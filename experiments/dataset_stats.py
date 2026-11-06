"""Dataset statistics for designing Phase 2 (plan: Thu Nov 19).

    .venv/Scripts/python experiments/dataset_stats.py                 # WN18RR and FB15k-237
    .venv/Scripts/python experiments/dataset_stats.py --datasets wn18rr

Measures what the G2 protocol blanks depend on:
  1. Size: entities, relations, triples per split; evaluation triples that
     mention an entity never seen in training (no method can embed those).
  2. Relation patterns in training: symmetric relations, inverse pairs and
     length-2 composition rules r <- r1 . r2. With one rotation per relation
     type the holonomy contains no entity states, so it is a path-rule check:
     it can only help where such rules exist.
  3. Cycle coverage: can h reach t within L = 1, 2, 3 training edges? Only
     then does the evaluation triple (h, r, t) close any cycle for the
     holonomy score. Walk counts bound how many cycles each triple has.
  4. Checkability of training edges: bridges (on no cycle) and edges in
     shared cut-pair groups (only locatable as a group).

The training graph is undirected (a cycle may cross an edge backwards, using
U^T) and a multigraph (two facts between the same entities are two edges).
Writes experiments/out/dataset_stats.json and notes/dataset_stats.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtvm.graph import RotationGraph  # noqa: E402
from rtvm.localise import cut_pair_classes  # noqa: E402

OUT = ROOT / "experiments" / "out"
NAMES = {"wn18rr": "WN18RR", "fb15k237": "FB15k-237", "nations": "Nations"}
STRONG = 0.8        # a pattern "holds" for a relation when it covers >= 80% of its triples
RULE_SAMPLE = 300   # triples sampled per relation when mining composition rules


def raw_split_sizes(key: str) -> dict[str, int] | None:
    """Triple counts in the files as published, before PyKEEN maps them to ids.

    PyKEEN DROPS evaluation triples whose head or tail never occurs in
    training (it only logs a warning), so its split sizes are smaller than the
    published ones. Papers differ on whether they evaluate on the full or the
    reduced split, so record both and say which one every number uses.
    """
    import zipfile

    base = Path.home() / ".data" / "pykeen" / "datasets" / key
    files = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
    try:
        if key == "fb15k237":
            with zipfile.ZipFile(base / "FB15K-237.2.zip") as z:
                return {s: sum(1 for line in z.read(f"Release/{f}").splitlines() if line.strip())
                        for s, f in files.items()}
        return {s: sum(1 for line in (base / f).read_text(encoding="utf-8").splitlines() if line.strip())
                for s, f in files.items()}
    except (FileNotFoundError, KeyError):
        return None


def size_stats(key: str, ds, splits: dict[str, np.ndarray]) -> dict:
    out = {"entities": ds.num_entities, "relations": ds.num_relations}
    raw = raw_split_sizes(key)
    for name, tr in splits.items():
        out[f"{name}_triples"] = len(tr)
        if raw is not None:
            out[f"{name}_triples_published"] = raw[name]
            out[f"{name}_dropped_by_pykeen"] = raw[name] - len(tr)
    return out


def relation_patterns(train: np.ndarray, n_rel: int, rel_names: list[str], seed: int = 0) -> dict:
    """Per relation: symmetry, best inverse partner, best length-2 composition rule.

    Coverages are AMIE-style head coverage: the share of the relation's own
    triples the pattern explains. Composition is estimated on up to
    RULE_SAMPLE sampled triples per relation (exact counts are too slow on hubs).
    """
    facts = set(map(tuple, train.tolist()))
    between = defaultdict(set)                       # (h, t) -> relations from h to t
    adj = defaultdict(lambda: defaultdict(list))     # x -> y -> signed labels for steps x -> y
    for h, r, t in train.tolist():
        between[(h, t)].add(r)
        adj[h][t].append(r)                          # forward step along (h, r, t)
        adj[t][h].append(-r - 1)                     # backward step, uses U_r^T
    rng = np.random.default_rng(seed)

    def label(s: int) -> str:
        return rel_names[s] if s >= 0 else f"{rel_names[-s - 1]}^-1"

    rows = []
    for r in range(n_rel):
        mine = train[train[:, 1] == r]
        n = len(mine)
        if n == 0:
            continue
        sym = sum((t, r, h) in facts for h, _, t in mine.tolist()) / n
        inverse = Counter()
        for h, _, t in mine.tolist():
            inverse.update(r2 for r2 in between.get((t, h), ()) if r2 != r)
        inv_rel, inv_count = inverse.most_common(1)[0] if inverse else (None, 0)

        sample = mine[rng.choice(n, size=min(n, RULE_SAMPLE), replace=False)]
        rules = Counter()
        for h, _, t in sample.tolist():
            patterns = set()
            for m in adj[h].keys() & adj[t].keys():
                if m in (h, t):
                    continue
                for l1 in adj[h][m]:
                    for l2 in adj[m][t]:
                        patterns.add((l1, l2))
            rules.update(patterns)
        rule, rule_count = rules.most_common(1)[0] if rules else (None, 0)
        rows.append({
            "relation": rel_names[r], "train_triples": n,
            "symmetric": round(sym, 3),
            "inverse_of": rel_names[inv_rel] if inv_rel is not None else None,
            "inverse_coverage": round(inv_count / n, 3),
            "best_rule": f"{label(rule[0])} . {label(rule[1])}" if rule else None,
            "rule_coverage": round(rule_count / len(sample), 3),
        })
    return {
        "per_relation": rows,
        "n_symmetric": sum(row["symmetric"] >= STRONG for row in rows),
        "n_with_inverse": sum(row["inverse_coverage"] >= STRONG for row in rows),
        "n_compositional": sum(row["rule_coverage"] >= STRONG for row in rows),
        "train_share_in_compositional": sum(row["train_triples"] for row in rows
                                            if row["rule_coverage"] >= STRONG) / len(train),
    }


def adjacency(train: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric multigraph adjacency: entry (x, y) counts the facts joining x and y."""
    rows = np.concatenate([train[:, 0], train[:, 2]])
    cols = np.concatenate([train[:, 2], train[:, 0]])
    return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))


def cycle_coverage(M: sp.csr_matrix, split: np.ndarray, chunk: int = 256) -> dict:
    """Share of triples whose h and t are joined by a training walk of length 1, <= 2, <= 3.

    Walk counts include walks that revisit a vertex, so they are upper bounds
    on the number of cycles (simple paths) a triple closes.
    """
    h, t = split[:, 0], split[:, 2]
    direct = np.asarray(M[h, t]).ravel()
    walks2 = np.zeros(len(h))
    walks3 = np.zeros(len(h))
    for s in range(0, len(h), chunk):
        H = M[h[s:s + chunk]]
        T = M[t[s:s + chunk]].toarray()        # M is symmetric: row t = column t
        walks2[s:s + chunk] = np.asarray(H.multiply(T).sum(axis=1)).ravel()
        walks3[s:s + chunk] = ((H @ M).toarray() * T).sum(axis=1)
    le1, le2 = direct > 0, (direct > 0) | (walks2 > 0)
    le3 = le2 | (walks3 > 0)
    total = direct + walks2 + walks3
    covered = total[le3]
    return {
        "reach_le1": float(le1.mean()), "reach_le2": float(le2.mean()), "reach_le3": float(le3.mean()),
        "walks_le3_median_if_covered": float(np.median(covered)) if len(covered) else 0.0,
        "walks_le3_p90_if_covered": float(np.percentile(covered, 90)) if len(covered) else 0.0,
        "share_with_ge10_walks": float((total >= 10).mean()),
    }


def checkability(train: np.ndarray, M: sp.csr_matrix) -> dict:
    """Bridges and shared cut-pair groups in the training graph, per connected component."""
    n_comp, comp = connected_components(M, directed=False)
    by_comp = defaultdict(list)
    for i, (h, _, t) in enumerate(train.tolist()):
        by_comp[comp[h]].append(i)
    n_bridges = n_shared = 0
    largest = max(len(v) for v in by_comp.values())
    for edge_ids in by_comp.values():
        local = {}
        edges = []
        for i in edge_ids:
            h, _, t = train[i].tolist()
            edges.append((local.setdefault(h, len(local)), local.setdefault(t, len(local))))
        graph = RotationGraph(len(local), edges, torch.ones(len(edges), 1, 1, 1))
        labels = cut_pair_classes(graph)
        sizes = Counter(labels)
        n_bridges += sizes.get(-1, 0)
        n_shared += sum(c for lab, c in sizes.items() if lab >= 0 and c > 1)
    non_isolated = len({x for x in train[:, [0, 2]].ravel().tolist()})
    return {
        "components": int(n_comp - (M.shape[0] - non_isolated)),
        "largest_component_edges": largest,
        "bridge_share": n_bridges / len(train),
        "shared_group_share": n_shared / len(train),
    }


def analyse(key: str) -> dict:
    from pykeen.datasets import get_dataset

    t0 = time.perf_counter()
    ds = get_dataset(dataset=key)
    splits = {"train": ds.training.mapped_triples.numpy(),
              "valid": ds.validation.mapped_triples.numpy(),
              "test": ds.testing.mapped_triples.numpy()}
    rel_names = [None] * ds.num_relations
    for name, i in ds.relation_to_id.items():
        rel_names[i] = name
    M = adjacency(splits["train"], ds.num_entities)
    result = {"dataset": NAMES.get(key, key), "size": size_stats(key, ds, splits)}
    print(f"  size: {result['size']}")
    result["coverage"] = {s: cycle_coverage(M, splits[s]) for s in ("valid", "test")}
    print(f"  test coverage: {result['coverage']['test']}")
    result["checkability"] = checkability(splits["train"], M)
    print(f"  checkability: {result['checkability']}")
    result["patterns"] = relation_patterns(splits["train"], ds.num_relations, rel_names)
    p = result["patterns"]
    print(f"  patterns: {p['n_symmetric']} symmetric, {p['n_with_inverse']} with an inverse, "
          f"{p['n_compositional']} compositional (of {ds.num_relations})")
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


def markdown(results: list[dict], commit: str) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Dataset statistics (generated)",
        "",
        f"Generated by `experiments/dataset_stats.py` at commit `{commit}`. Do not edit by hand; "
        "rerun the script. Interpretation belongs in the weekly log and the protocol.",
        "",
        "Counts below use PyKEEN's splits, which drop evaluation triples with an entity unseen in training; "
        "the published sizes are listed for comparison.",
        "",
        "\\* Head coverage, not rule confidence: the share of a relation's triples that the 2-hop pattern "
        "connects. A generic pattern through a hub (for example \"same profession\") can cover a relation "
        "without predicting it, so these rows are an upper bound on genuine composition.",
        "",
        f"\"Holds\" means the pattern covers at least {STRONG:.0%} of the relation's training triples. "
        f"Composition rules are estimated on up to {RULE_SAMPLE} sampled triples per relation.",
        "",
        "| | " + " | ".join(r["dataset"] for r in results) + " |",
        "| --- | " + " | ".join("---" for _ in results) + " |",
    ]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(r) for r in results) + " |")

    row("Entities / relations", lambda r: f"{r['size']['entities']:,} / {r['size']['relations']}")
    row("Train / valid / test triples", lambda r: f"{r['size']['train_triples']:,} / "
        f"{r['size']['valid_triples']:,} / {r['size']['test_triples']:,}")
    row("Valid / test triples as published", lambda r: f"{r['size'].get('valid_triples_published', 0):,} / "
        f"{r['size'].get('test_triples_published', 0):,}")
    row("Valid / test triples PyKEEN drops (entity unseen in training)",
        lambda r: f"{r['size'].get('valid_dropped_by_pykeen', 0):,} / {r['size'].get('test_dropped_by_pykeen', 0):,}")
    row("Relations: symmetric / with an inverse partner", lambda r: f"{r['patterns']['n_symmetric']} / "
        f"{r['patterns']['n_with_inverse']}")
    row("Relations with a 2-hop path pattern on >= 80% of their triples*",
        lambda r: f"{r['patterns']['n_compositional']}")
    row("Training triples in those relations*", lambda r: pct(r["patterns"]["train_share_in_compositional"]))
    for L in (1, 2, 3):
        row(f"Test triples closing a cycle with paths of length <= {L}",
            lambda r, L=L: pct(r["coverage"]["test"][f"reach_le{L}"]))
    row("Walks of length <= 3 per covered test triple: median / 90th pct",
        lambda r: f"{r['coverage']['test']['walks_le3_median_if_covered']:,.0f} / "
                  f"{r['coverage']['test']['walks_le3_p90_if_covered']:,.0f}")
    row("Training edges that are bridges (uncheckable)", lambda r: pct(r["checkability"]["bridge_share"]))
    row("Training edges in a shared cut-pair group", lambda r: pct(r["checkability"]["shared_group_share"]))
    row("Connected components of the training graph", lambda r: f"{r['checkability']['components']:,}")

    for r in results:
        rels = sorted(r["patterns"]["per_relation"], key=lambda x: -x["rule_coverage"])[:10]
        lines += ["", f"## {r['dataset']}: relations with the strongest composition rules", "",
                  "| Relation | Train triples | Best 2-hop pattern (r1 . r2) | Head coverage | Symmetric | Inverse coverage |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for x in rels:
            lines.append(f"| `{x['relation']}` | {x['train_triples']:,} | `{x['best_rule']}` | "
                         f"{x['rule_coverage']:.2f} | {x['symmetric']:.2f} | {x['inverse_coverage']:.2f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["wn18rr", "fb15k237"], choices=list(NAMES))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    results = []
    for key in args.datasets:
        print(f"[{NAMES[key]}]")
        results.append(analyse(key))
    (OUT / "dataset_stats.json").write_text(json.dumps(results, indent=2))
    import subprocess
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                            cwd=ROOT).stdout.strip() or "unknown"
    (ROOT / "notes" / "dataset_stats.md").write_text(markdown(results, commit), encoding="utf-8")
    print(f"Wrote {OUT / 'dataset_stats.json'} and notes/dataset_stats.md")


if __name__ == "__main__":
    main()
