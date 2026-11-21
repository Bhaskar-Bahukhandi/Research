"""G2 secondary analyses, as fixed in the protocol notes of 2026-11-22 (second note).

    python experiments/g2_secondary.py basic --dataset fb15k237      # S2, S7 (FB15k-237 part), S9

S3 and S6 follow in this file.

This code lives outside the frozen paths on purpose, so that the test-split guard of
experiments/g2_gate.py keeps holding. It imports the frozen gate code unchanged, reads the
gate's V2 and test feature files and refits the combiners on V2 exactly as the primary test
does. As a check, it reproduces the gate's per-seed B0 and B1 grouped AUCs before it reports
anything else.

The test feature files are read only when:
  * the gate's guard passes (clean tree, one freeze line, frozen code unchanged);
  * the gate has been decided (gate.json exists);
  * every feature file was made by the current frozen code.

None of these analyses has a pass rule, and none can change the gate.
Outputs go to runs/g2secondary/<dataset>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import g2_gate as gate  # noqa: E402
from rtvm.g2gate import (B0, B1, BOOT_SEED, N_BOOT, STRUCTURE, feature_matrix, fit_combiner,  # noqa: E402
                         relation_bootstrap)

SEEDS = (0, 1, 2)
POOLED_BOOT = 1_000         # S2(b): the first 1,000 rows of the primary resample matrix
N_BINS = 10                 # S9(b): holonomy deciles
TABLE_MIN = 100             # S7: relations with at least this many test positives are tabulated
BIN_NAMES = tuple(f"holonomy_bin{j}" for j in range(1, N_BINS))   # decile 1 (index 0) is the reference


# ---------------------------------------------------------------- building blocks

def model_columns(part: dict, s: int) -> dict:
    """The primary test's named feature columns for seed s (structure, RotatE run s, holonomy run s)."""
    return {**{k: part[k] for k in STRUCTURE}, "rotate": part[f"rotate:{s}"], "holonomy": part[f"holonomy:{s}"]}


def fit_and_score(v2_pos: dict, v2_neg: dict, test_pos: dict, test_neg: dict, names: tuple[str, ...]):
    """Fit a combiner on V2 as the primary test does; return it with its test scores (n,), (n, k)."""
    Vp, Vn = feature_matrix(v2_pos, names), feature_matrix(v2_neg, names)
    X = np.concatenate([Vp, Vn.reshape(-1, len(names))])
    y = np.r_[np.ones(len(Vp)), np.zeros(Vn.shape[0] * Vn.shape[1])]
    comb = fit_combiner(X, y, names)
    Tp, Tn = feature_matrix(test_pos, names), feature_matrix(test_neg, names)
    n, k, f = Tn.shape
    return comb, comb.decision(Tp), comb.decision(Tn.reshape(n * k, f)).reshape(n, k)


def masked_rows(pos: np.ndarray, neg: np.ndarray, neg_valid: np.ndarray) -> np.ndarray:
    """Per-positive grouped AUC over the valid negatives only (ties 1/2); NaN where none is valid."""
    pos, neg, nv = np.asarray(pos, float), np.asarray(neg, float), np.asarray(neg_valid, bool)
    if np.isnan(pos).any() or np.isnan(neg).any():
        raise ValueError("NaN score")
    wins = (((neg < pos[:, None]) + 0.5 * (neg == pos[:, None])) * nv).sum(axis=1)
    count = nv.sum(axis=1)
    out = np.full(len(pos), np.nan)
    np.divide(wins, count, out=out, where=count > 0)
    return out


def delta_summary(diffs: np.ndarray, relations: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Delta per seed, Delta-bar and the primary relation bootstrap's percentile 95% CI.
    diffs (S, n) over the positives used; relations (n,) theirs."""
    diffs = np.atleast_2d(diffs)
    boot = relation_bootstrap(diffs, relations, n_boot=n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(delta_per_seed=[float(d) for d in diffs.mean(axis=1)], delta_bar=float(diffs.mean()),
                ci95=[float(lo), float(hi)], n_positives=int(diffs.shape[1]),
                n_clusters=int(len(np.unique(relations))), n_boot=n_boot)


def _times(draws: np.ndarray, G: int) -> np.ndarray:
    """times[b, g]: how often resample b drew cluster g (as in rtvm.g2gate.relation_bootstrap)."""
    times = np.zeros((len(draws), G))
    np.add.at(times, (np.repeat(np.arange(len(draws)), G), draws.ravel()), 1.0)
    return times


def pooled_auc_by_resample(pos: np.ndarray, neg: np.ndarray, relations: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Pooled ROC AUC (ties 1/2) over all positives and negatives, with each positive and its
    negatives weighted by how often its relation cluster was drawn. times (B, G) as _times;
    a row of ones gives the plain pooled AUC.

    AUC(w) = w'Mw / ((w'n_pos)(w'n_neg)), where M[g', g] sums, over the positives of cluster
    g', the negatives of cluster g scored below them (ties 1/2).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    _, cl = np.unique(np.asarray(relations), return_inverse=True)
    cl = cl.reshape(-1)
    G, k = times.shape[1], neg.shape[1]
    M = np.zeros((G, G))
    for g in range(G):
        ns = np.sort(neg[cl == g].ravel())
        below, upto = np.searchsorted(ns, pos, "left"), np.searchsorted(ns, pos, "right")
        M[:, g] = np.bincount(cl, weights=below + 0.5 * (upto - below), minlength=G)
    n_pos = np.bincount(cl, minlength=G).astype(float)
    return np.einsum("bi,ij,bj->b", times, M, times) / ((times @ n_pos) * (times @ (n_pos * k)))


def decile_indicators(v2_pos: np.ndarray, v2_neg: np.ndarray, v2_cov_pos: np.ndarray, v2_cov_neg: np.ndarray):
    """S9(b): bin edges from the holonomy values of covered V2 rows (positives and negatives
    together, no labels), and a function giving a split's 9 indicator columns (decile 1 is the
    reference; uncovered rows get none)."""
    values = np.concatenate([v2_pos[v2_cov_pos > 0], v2_neg[v2_cov_neg > 0]])
    edges = np.quantile(values, np.linspace(0, 1, N_BINS + 1))

    def columns(values: np.ndarray, covered: np.ndarray) -> dict:
        b = np.searchsorted(edges[1:-1], values, side="right")          # 0..9
        return {name: ((covered > 0) & (b == j + 1)).astype(float) for j, name in enumerate(BIN_NAMES)}
    return edges, columns


# ---------------------------------------------------------------- the analyses

def basic_analyses(v2: dict, test: dict, fallback: np.ndarray, relation_names: dict | None = None,
                   n_boot: int = N_BOOT, pooled_boot: int = POOLED_BOOT) -> dict:
    """S2, S7 (FB15k-237 part) and S9 from assembled V2 and test features (g2_gate.assemble)."""
    rel = np.asarray(test["relations"])
    n, k = test["neg"]["covered"].shape
    cov_pos, cov_neg = test["pos"]["covered"] > 0, test["neg"]["covered"] > 0
    G = len(np.unique(rel))
    times = _times(np.random.default_rng(BOOT_SEED).integers(0, G, size=(N_BOOT, G))[:pooled_boot], G)
    times = np.vstack([np.ones((1, G)), times])                        # row 0: the plain pooled AUC
    scores, rows, pooled, binned = {}, {}, {}, {}
    for s in SEEDS:
        vp, vn = model_columns(v2["pos"], s), model_columns(v2["neg"], s)
        tp, tn = model_columns(test["pos"], s), model_columns(test["neg"], s)
        for name, names in (("B0", B0), ("B1", B1)):
            _, sp, sn = fit_and_score(vp, vn, tp, tn, names)
            scores[name, s] = sp, sn
            rows[name, s] = masked_rows(sp, sn, np.ones((n, k), bool))
            pooled[name, s] = pooled_auc_by_resample(sp, sn, rel, times)
        edges, cols = decile_indicators(vp["holonomy"], vn["holonomy"], vp["covered"], vn["covered"])
        names = B0 + BIN_NAMES
        comb, sp, sn = fit_and_score({**vp, **cols(vp["holonomy"], vp["covered"])},
                                     {**vn, **cols(vn["holonomy"], vn["covered"])},
                                     {**tp, **cols(tp["holonomy"], tp["covered"])},
                                     {**tn, **cols(tn["holonomy"], tn["covered"])}, names)
        idx = [names.index(b) for b in BIN_NAMES]
        binned[s] = dict(rows=masked_rows(sp, sn, np.ones((n, k), bool)), edges=edges.tolist(),
                         log_odds_vs_decile1=(comb.coef[idx] / comb.scale[idx]).tolist())

    out = dict(check=dict(grouped_auc={f"{m}:{s}": float(rows[m, s].mean()) for m in ("B0", "B1") for s in SEEDS}))

    # S2(a): covered positives against their covered negatives
    cov_rows = {(m, s): masked_rows(*scores[m, s], cov_neg) for m in ("B0", "B1") for s in SEEDS}
    keep = cov_pos & ~np.isnan(cov_rows["B0", 0])
    d = np.stack([cov_rows["B1", s][keep] - cov_rows["B0", s][keep] for s in SEEDS])
    out["s2_covered"] = dict(**delta_summary(d, rel[keep], n_boot), n_negatives=int(cov_neg[keep].sum()),
                             grouped_auc={f"{m}:{s}": float(cov_rows[m, s][keep].mean())
                                          for m in ("B0", "B1") for s in SEEDS})

    # S2(b): pooled AUC on the full set
    dp = np.stack([pooled["B1", s] - pooled["B0", s] for s in SEEDS])   # (S, 1 + B)
    lo, hi = np.percentile(dp[:, 1:].mean(axis=0), [2.5, 97.5])
    out["s2_pooled"] = dict(auc={f"{m}:{s}": float(pooled[m, s][0]) for m in ("B0", "B1") for s in SEEDS},
                            delta_per_seed=dp[:, 0].tolist(), delta_bar=float(dp[:, 0].mean()),
                            ci95=[float(lo), float(hi)], n_boot=int(times.shape[0] - 1))

    # S7, FB15k-237 part: per-relation Delta-bar
    d_full = np.stack([rows["B1", s] - rows["B0", s] for s in SEEDS])
    ids, inv, counts = np.unique(rel, return_inverse=True, return_counts=True)
    per_rel = np.stack([np.bincount(inv.reshape(-1), weights=x, minlength=len(ids)) for x in d_full]) / counts
    dbar = per_rel.mean(axis=0)
    table = [dict(relation=int(r), name=(relation_names or {}).get(int(r)), n_positives=int(c), delta_bar=float(v))
             for r, c, v in zip(ids, counts, dbar) if c >= TABLE_MIN]
    out["s7"] = dict(n_relations=int(len(ids)), positive=int((dbar > 0).sum()), negative=int((dbar < 0).sum()),
                     zero=int((dbar == 0).sum()),
                     table=sorted(table, key=lambda r: -r["n_positives"]), min_positives_in_table=TABLE_MIN,
                     all_relations={int(r): [int(c), float(v)] for r, c, v in zip(ids, counts, dbar)})

    # S9(a): flagged fallback negatives removed
    valid = ~np.asarray(fallback, bool)
    fb_rows = {(m, s): masked_rows(*scores[m, s], valid) for m in ("B0", "B1") for s in SEEDS}
    keep = ~np.isnan(fb_rows["B0", 0])
    d = np.stack([fb_rows["B1", s][keep] - fb_rows["B0", s][keep] for s in SEEDS])
    out["s9_fallback_excluded"] = dict(**delta_summary(d, rel[keep], n_boot), n_negatives=int(valid[keep].sum()),
                                       n_fallback_removed=int((~valid).sum()))

    # S9(b): holonomy in decile bins, against B0
    d = np.stack([binned[s]["rows"] - rows["B0", s] for s in SEEDS])
    out["s9_binned"] = dict(**delta_summary(d, rel, n_boot),
                            per_seed={s: dict(edges=binned[s]["edges"],
                                              log_odds_vs_decile1=binned[s]["log_odds_vs_decile1"]) for s in SEEDS})
    return out


# ---------------------------------------------------------------- main

def relation_labels(dataset: str) -> dict[int, str]:
    from pykeen.datasets import get_dataset
    return {int(i): str(lbl) for lbl, i in get_dataset(dataset=dataset).training.relation_to_id.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["basic"])
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--gate", default=str(ROOT / "runs/g2gate"))
    ap.add_argument("--out", default=str(ROOT / "runs/g2secondary"))
    args = ap.parse_args()
    gate_dir, out_dir = Path(args.gate) / args.dataset, Path(args.out) / args.dataset
    freeze = gate.check_test_allowed()
    gate_json = gate_dir / "gate.json"
    if not gate_json.exists():
        sys.exit(f"{gate_json} missing: secondary analyses follow the decided gate")
    state = gate.code_state()
    t0 = time.time()
    v2, test = gate.assemble(gate_dir, "v2"), gate.assemble(gate_dir, "test")
    gate.check_provenance({**v2["provenance"], **test["provenance"]}, state)
    fallback = np.load(gate_dir / "test_structure.npz")["fallback"]
    res = basic_analyses(v2, test, fallback, relation_labels(args.dataset))
    decided = json.loads(gate_json.read_text(encoding="utf-8"))
    for key, value in res["check"]["grouped_auc"].items():               # must reproduce the gate exactly
        m, s = key.split(":")
        if abs(value - decided["seeds"][s][m]["grouped_auc"]) > 1e-12:
            sys.exit(f"{key}: {value} does not reproduce the gate's {decided['seeds'][s][m]['grouped_auc']}")
    res["check"]["reproduces_gate"] = True
    res.update(dataset=args.dataset, kind="S2, S7 (FB15k-237 part), S9; protocol notes of 2026-11-22; no pass rule",
               freeze_commit=freeze, head=state["head"], frozen_tree=state["frozen_tree"],
               sources=test["sources"], versions=gate.versions(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    gate.write_json(out_dir / "basic.json", res)
    print(f"wrote {out_dir / 'basic.json'} ({time.time() - t0:.0f} s); the gate's grouped AUCs are reproduced")


if __name__ == "__main__":
    main()
