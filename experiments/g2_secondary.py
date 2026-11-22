"""G2 secondary analyses, as fixed in the protocol notes of 2026-11-22 (second and fourth notes).

    python experiments/g2_secondary.py basic --dataset fb15k237      # S2, S7 (FB15k-237 part), S9
    python experiments/g2_secondary.py s6 --dataset fb15k237 --baseline rotate   # S6, RotatE's rotations
    python experiments/g2_secondary.py s6 --dataset fb15k237 --baseline quate    # S6, QuatE's rotations
    python experiments/g2_secondary.py s3 --dataset fb15k237         # S3, typed path rules
    python experiments/g2_secondary.py s4 --dataset fb15k237         # S4, path length
    python experiments/g2_secondary.py s5 --dataset fb15k237         # S5, QuatE in place of RotatE
    python experiments/g2_secondary.py s7wn --dataset wn18rr         # S7, WN18RR part
    python experiments/g2_secondary.py s8 --dataset fb15k237         # S8, filtered link prediction (also wn18rr)
    python experiments/g2_secondary.py s10 --dataset fb15k237        # S10, after experiments/g2_s10.py
    python experiments/g2_secondary.py collapse --dataset wn18rr     # rotation collapse check (exploratory)

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
                         primary_test, relation_bootstrap, rotation_table, triple_holonomy)

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


# ---------------------------------------------------------------- S6: holonomy from the baselines' rotations

def rotate_rotations(relation_weights, gamma: float, dim: int):
    """RotatE's relations as block rotations, (R, dim, 2, 2): phase theta = r * pi / range with
    range = (gamma + 2) / dim (rtvm.baselines.build_model), each coordinate the 2 x 2 rotation
    [[cos, -sin], [sin, cos]] acting on (Re, Im), i.e. h * exp(i theta) as in the score."""
    import torch
    theta = torch.as_tensor(relation_weights, dtype=torch.float32) * (np.pi / ((gamma + 2.0) / dim))
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def quate_rotations(relation_weights, table):
    """QuatE's relations as block rotations, (R, dim, 4, 4): each coordinate's normalised
    quaternion r^ as the matrix of right multiplication h -> h (x) r^, the action in the official
    score (rtvm.baselines.NormalizedQuatEInteraction). table (4, 4, 4) is PyKEEN's quaternion
    multiplication table, so (h (x) r)_k = sum_ij h_i r_j table[i, j, k]."""
    import torch
    w = torch.as_tensor(relation_weights, dtype=torch.float32)
    r = w.reshape(w.shape[0], -1, 4)
    r = r / r.norm(p=2, dim=-1, keepdim=True)
    return torch.einsum("rdj,ijk->rdki", r, torch.as_tensor(table, dtype=torch.float32))


def model_runs_any(runs: Path, dataset: str, model: str) -> dict:
    """g2_gate.model_runs, which checks config hash, best trial and seed runs, for any searched
    model (the gate script itself only names its two gate models)."""
    saved = gate.MODEL_CONFIG
    gate.MODEL_CONFIG = {**saved, model: model}
    try:
        return gate.model_runs(runs, dataset, model)
    finally:
        gate.MODEL_CONFIG = saved


def baseline_tables(runs: Path, dataset: str, baseline: str) -> dict:
    """Seed -> (step table, run stem) of the baseline's own relation rotations (S6)."""
    import torch
    out = {}
    for s, stem in model_runs_any(runs, dataset, baseline).items():
        rec = json.loads(stem.with_suffix(".json").read_text())
        weights = torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True)
        if baseline == "rotate":
            R = rotate_rotations(weights["relation_representations.0._embeddings.weight"],
                                 rec["config"]["gamma"], rec["config"]["dim"])
        elif baseline == "quate":
            R = quate_rotations(weights["relation_representations.0._embeddings.weight"],
                                weights["interaction.table"])
        else:
            raise ValueError(f"S6 has no rotations for {baseline!r}")
        out[s] = (rotation_table(R), stem)
    return out


def holonomy_features(d: dict, table) -> tuple[np.ndarray, np.ndarray]:
    """The triple holonomy score of a split's positives (n,) and negatives (n, k), as the gate computes it."""
    m, k = d["neg"].shape[:2]
    pos, _ = triple_holonomy(table, d["pos"][:, 1], gate.steps(d, d["pos_paths"]))
    neg, _ = triple_holonomy(table, gate.flat(d["neg"], 1)[:, 1], gate.steps(d, gate.flat(d["neg_paths"], 2)))
    return pos, neg.reshape(m, k)


def replace_columns(split: dict, model: str, features: dict) -> dict:
    """A copy of an assembled split whose <model>:<s> columns are replaced by features[s] = (pos, neg)."""
    out = dict(split, pos=dict(split["pos"]), neg=dict(split["neg"]))
    for s, (p, n) in features.items():
        out["pos"][f"{model}:{s}"], out["neg"][f"{model}:{s}"] = p, n
    return out


def replace_holonomy(split: dict, features: dict) -> dict:
    return replace_columns(split, "holonomy", features)


def s6_analysis(v2: dict, test: dict, features: dict, n_boot: int = N_BOOT) -> dict:
    """B0 against B0 + the baseline-rotation holonomy, by the primary test's procedure; no pass rule."""
    res = primary_test(replace_holonomy(v2, features["v2"]), replace_holonomy(test, features["test"]), n_boot=n_boot)
    res.pop("decision")                                   # S6 has no pass rule (note of 2026-11-22, second)
    return res


# ---------------------------------------------------------------- S5 and S7 (WN18RR part)

def baseline_scores(d: dict, stem: Path) -> tuple[np.ndarray, np.ndarray]:
    """A baseline's score_hrt on a split's positives (n,) and negatives (n, k), computed as the gate
    computes RotatE's (experiments/g2_gate.py model_scores), for any G2 baseline config."""
    import torch
    from rtvm.baselines import BaselineConfig, build_model, score_triples
    rec = json.loads(stem.with_suffix(".json").read_text())
    net = build_model(BaselineConfig(**rec["config"]), d["training"])
    net.load_state_dict(torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True))
    m, k = d["neg"].shape[:2]
    return score_triples(net, d["pos"], "cpu"), score_triples(net, gate.flat(d["neg"], 1), "cpu").reshape(m, k)


def s5_analysis(v2: dict, test: dict, features: dict, n_boot: int = N_BOOT) -> dict:
    """S5: QuatE's score in RotatE's place in both combiners, by the primary test's procedure."""
    res = primary_test(replace_columns(v2, "rotate", features["v2"]), replace_columns(test, "rotate", features["test"]),
                       n_boot=n_boot)
    res.pop("decision")
    return res


def reverse_in_training(triples: np.ndarray, train: np.ndarray) -> np.ndarray:
    """True where the exact reverse (t, r, h) of a triple is a training triple."""
    known = set(map(tuple, np.asarray(train).tolist()))
    return np.array([(t, r, h) in known for h, r, t in np.asarray(triples).tolist()], dtype=bool)


def grouped_deltas(v2: dict, test: dict, groups: dict, n_boot: int = N_BOOT) -> dict:
    """Delta of B1 over B0 (primary combiners, refitted on V2 as in the primary test) within each
    group of test positives, each with the relation bootstrap restricted to its positives."""
    rows = {}
    n, k = test["neg"]["covered"].shape
    for s in SEEDS:
        vp, vn = model_columns(v2["pos"], s), model_columns(v2["neg"], s)
        tp, tn = model_columns(test["pos"], s), model_columns(test["neg"], s)
        for name, names in (("B0", B0), ("B1", B1)):
            _, sp, sn = fit_and_score(vp, vn, tp, tn, names)
            rows[name, s] = masked_rows(sp, sn, np.ones((n, k), bool))
    d = np.stack([rows["B1", s] - rows["B0", s] for s in SEEDS])
    rel = np.asarray(test["relations"])
    return {g: delta_summary(d[:, mask], rel[mask], n_boot) for g, mask in groups.items() if mask.any()}


# ---------------------------------------------------------------- S4: path length

def keep_paths(codes: np.ndarray, keep) -> np.ndarray:
    """Saved path codes with every path slot whose length fails keep(length) emptied."""
    codes = np.asarray(codes)
    lengths = (codes != 0).sum(axis=-1)
    return np.where((keep(lengths) & (lengths > 0))[..., None], codes, 0).astype(codes.dtype)


def holonomy_model_tables(runs: Path, dataset: str) -> dict:
    """Seed -> (step table, run stem) of the gate's holonomy runs (as experiments/g2_gate.py model_scores)."""
    import torch
    from rtvm.holonomy_model import HolonomyModel
    out = {}
    for s, stem in gate.model_runs(runs, dataset, "holonomy").items():
        rec = json.loads(stem.with_suffix(".json").read_text())
        weights = torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True)
        net = HolonomyModel(weights["psi"].shape[0], weights["U"].shape[0], rank=rec["config"]["rank"])
        net.load_state_dict(weights)
        with torch.no_grad():
            out[s] = (net.table(), stem)
    return out


def restricted(d: dict, keep, drop_n3: bool = False) -> dict:
    """A loaded split whose saved paths keep only lengths passing keep(length); with drop_n3, n3 = 0."""
    out = dict(d, pos_paths=keep_paths(d["pos_paths"], keep), neg_paths=keep_paths(d["neg_paths"], keep))
    if drop_n3:
        for key in ("pos_counts", "neg_counts"):
            c = np.array(d[key], copy=True)
            c[..., 2] = 0
            out[key] = c
    return out


def replace_structure(split: dict, struct: dict) -> dict:
    """A copy of an assembled split whose structure columns come from g2_gate.structure_and_null output."""
    out = dict(split, pos=dict(split["pos"]), neg=dict(split["neg"]))
    for name in STRUCTURE:
        out["pos"][name], out["neg"][name] = struct[f"pos/{name}"], struct[f"neg/{name}"]
    return out


# ---------------------------------------------------------------- S8: filtered link prediction

def answer_index(known: np.ndarray, query: str, num_entities: int, num_relations: int):
    """For tail queries (h, r, ?) or head queries (?, r, t): sorted query keys, CSR starts and the
    known answers, from every known true triple."""
    known = np.asarray(known, dtype=np.int64)
    if query == "tail":
        keys, answers = known[:, 0] * num_relations + known[:, 1], known[:, 2]
    else:
        keys, answers = known[:, 2] * num_relations + known[:, 1], known[:, 0]
    order = np.lexsort((answers, keys))
    keys, answers = keys[order], answers[order]
    uniq, starts = np.unique(keys, return_index=True)
    return uniq, np.r_[starts, len(keys)], answers


def known_answers(index, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row, entity) pairs of the known answers of each query key (row i = keys[i])."""
    uniq, starts, answers = index
    pos = np.searchsorted(uniq, keys)
    found = (pos < len(uniq)) & (uniq[np.minimum(pos, len(uniq) - 1)] == keys)
    lo = np.where(found, starts[np.minimum(pos, len(uniq) - 1)], 0)
    hi = np.where(found, starts[np.minimum(pos, len(uniq) - 1) + 1], 0)
    count = hi - lo
    rows = np.repeat(np.arange(len(keys)), count)
    offsets = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    return rows, answers[np.repeat(lo, count) + offsets]


def realistic_ranks(scores, targets: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Filtered realistic ranks: 1 + (# scored higher) + (# tied) / 2, over the candidates that are
    not known true answers (rows, cols); the target itself is never a competitor."""
    import torch
    scores = torch.as_tensor(scores, dtype=torch.float64).clone()
    idx = torch.arange(scores.shape[0])
    t = torch.as_tensor(targets, dtype=torch.long)
    true = scores[idx, t].clone()
    if np.isnan(true.numpy()).any():
        raise ValueError("NaN score")
    scores[torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)] = float("-inf")
    scores[idx, t] = float("-inf")
    higher = (scores > true[:, None]).sum(dim=1)
    tied = (scores == true[:, None]).sum(dim=1)
    return (1 + higher + 0.5 * tied).double().numpy()


def link_prediction(score_tails, score_heads, test: np.ndarray, known: np.ndarray, num_entities: int,
                    num_relations: int, batch: int = 256) -> dict:
    """Filtered MRR and Hits@1/10 with realistic ranks over both directions. score_tails(h, r) and
    score_heads(r, t) return (B, num_entities) scores, higher = more plausible."""
    test = np.asarray(test, dtype=np.int64)
    ranks = []
    for query, fn in (("tail", score_tails), ("head", score_heads)):
        index = answer_index(known, query, num_entities, num_relations)
        for s in range(0, len(test), batch):
            h, r, t = (test[s:s + batch, j] for j in range(3))
            if query == "tail":
                scores, keys, targets = fn(h, r), h * num_relations + r, t
            else:
                scores, keys, targets = fn(r, t), t * num_relations + r, h
            rows, cols = known_answers(index, keys)
            ranks.append(realistic_ranks(scores, targets, rows, cols))
    ranks = np.concatenate(ranks)
    return dict(mrr=float(np.mean(1 / ranks)), hits_at_1=float(np.mean(ranks <= 1)),
                hits_at_10=float(np.mean(ranks <= 10)), n_ranks=int(len(ranks)))


def baseline_scorers(net, slice_size: int = 2048):
    """PyKEEN's score_t / score_h against every entity, in slices of `slice_size` candidates: RotatE
    builds a (batch x candidates x dim) complex tensor, and unsliced on WN18RR (40,559 entities) that
    exhausted this PC's memory on 2026-11-22. Slicing changes no score."""
    import torch

    def tails(h, r):
        with torch.no_grad():
            return net.score_t(torch.as_tensor(np.stack([h, r], 1), dtype=torch.long), slice_size=slice_size).double()

    def heads(r, t):
        with torch.no_grad():
            return net.score_h(torch.as_tensor(np.stack([r, t], 1), dtype=torch.long), slice_size=slice_size).double()
    return tails, heads


def quate_scorers(net):
    """QuatE's score against every entity without PyKEEN's broadcast einsum (which needs about 8 GB
    per batch here). The score is -<h (x) r^, t> summed over coordinates (NormalizedQuatEInteraction),
    i.e. -<M h, t> with M the right-multiplication matrix of r^, and <M h, t> = <h, M^T t>."""
    import torch
    with torch.no_grad():
        ent = net.entity_representations[0]._embeddings.weight.double()          # (E, d * 4)
        rel = net.relation_representations[0]._embeddings.weight
        M = quate_rotations(rel, net.interaction.table).double()                # (R, d, 4, 4)
    E = ent.reshape(ent.shape[0], -1, 4)

    def tails(h, r):
        with torch.no_grad():
            q = (M[torch.as_tensor(r)] @ E[torch.as_tensor(h)][..., None])[..., 0]       # (B, d, 4)
            return -(q.reshape(q.shape[0], -1) @ ent.T)

    def heads(r, t):
        with torch.no_grad():
            u = (M[torch.as_tensor(r)].transpose(-1, -2) @ E[torch.as_tensor(t)][..., None])[..., 0]
            return -(u.reshape(u.shape[0], -1) @ ent.T)
    return tails, heads


def holonomy_scorers(model):
    """The holonomy model's single-triple score -||U_r psi_h - psi_t|| against every entity; head
    queries use ||U_r psi_h - psi_t|| = ||psi_h - U_r^T psi_t|| (U_r is orthogonal)."""
    import torch
    from rtvm.rotations import block_apply
    with torch.no_grad():
        R, psi = model.rotations().double(), model.psi.double()

    def dist(x):
        return torch.cdist(x, psi, compute_mode="donot_use_mm_for_euclid_dist")

    def tails(h, r):
        with torch.no_grad():
            return -dist(block_apply(R[torch.as_tensor(r)], psi[torch.as_tensor(h)]))

    def heads(r, t):
        with torch.no_grad():
            return -dist(block_apply(R[torch.as_tensor(r)].transpose(-1, -2), psi[torch.as_tensor(t)]))
    return tails, heads


def s8_models(runs: Path, dataset: str, training) -> dict:
    """Model name -> seed -> (scorers, run stem), for RotatE, QuatE and the holonomy model."""
    import torch
    from rtvm.baselines import BaselineConfig, build_model
    from rtvm.holonomy_model import HolonomyModel
    out = {}
    for name in ("rotate", "quate"):
        out[name] = {}
        for s, stem in model_runs_any(runs, dataset, name).items():
            rec = json.loads(stem.with_suffix(".json").read_text())
            net = build_model(BaselineConfig(**rec["config"]), training)
            net.load_state_dict(torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True))
            net.eval()
            out[name][s] = (quate_scorers(net) if name == "quate" else baseline_scorers(net), stem)
    out["holonomy_single_triple"] = {}
    for s, stem in gate.model_runs(runs, dataset, "holonomy").items():
        rec = json.loads(stem.with_suffix(".json").read_text())
        weights = torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True)
        net = HolonomyModel(weights["psi"].shape[0], weights["U"].shape[0], rank=rec["config"]["rank"])
        net.load_state_dict(weights)
        out["holonomy_single_triple"][s] = (holonomy_scorers(net), stem)
    return out


# ---------------------------------------------------------------- S10: dense rotations

def dense_runs(s10_runs: Path) -> dict:
    """Seed -> (step table, run stem, record, model) of the S10 runs (experiments/g2_s10.py), each
    checked: complete, seed s, the committed config, and the best trial's values at rank 32."""
    import torch
    import g2_s10
    from rtvm.g2search import CONFIGS, file_sha256
    from rtvm.holonomy_model import HolonomyModel
    best, values = g2_s10.dense_values()
    out = {}
    for s in SEEDS:
        stem = Path(s10_runs) / g2_s10.DATASET / "holonomy_dense" / f"seed_{s}"
        rec = json.loads(stem.with_suffix(".json").read_text())
        if not rec.get("complete") or rec.get("seed") != s or rec.get("values") != values or \
                rec.get("best_trial") != best or rec.get("config_sha256") != file_sha256(ROOT / CONFIGS["holonomy"]):
            sys.exit(f"{stem}: not a complete S10 run with seed {s} and trial {best}'s values at rank 32")
        weights = torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True)
        net = HolonomyModel(weights["psi"].shape[0], weights["U"].shape[0], rank=values["rank"])
        net.load_state_dict(weights)
        with torch.no_grad():
            out[s] = (net.table(), stem, rec, net)
    return out


# ---------------------------------------------------------------- S3: typed path rules

EST_PER_RELATION = 300      # estimation triples per relation (note of 2026-11-22, second)
MIN_SUPPORT = 5             # a pattern is rated for r only with at least this many estimation triples
RULES = ("rule_max", "rule_mean", "rule_rated")


def estimation_set(train: np.ndarray, num_relations: int, per_relation: int = EST_PER_RELATION,
                   seed: int = 0) -> np.ndarray:
    """Training-edge ids of the estimation triples: for each relation in id order, up to
    `per_relation` of its training triples, uniform without replacement, one generator."""
    rng = np.random.default_rng(seed)
    out = []
    for r in range(num_relations):
        idx = np.flatnonzero(train[:, 1] == r)
        out.append(np.sort(rng.choice(idx, per_relation, replace=False)) if len(idx) > per_relation else idx)
    return np.concatenate(out).astype(np.int64)


def pattern_keys(codes: np.ndarray, train_relations: np.ndarray, num_relations: int) -> np.ndarray:
    """One integer per sampled path naming its typed pattern, the sequence of (relation,
    direction) steps from h to t; 0 for an empty path slot. codes (..., MAX_LEN) as saved."""
    codes = np.asarray(codes, dtype=np.int64)
    base = 2 * num_relations + 1
    edge = np.abs(codes) - 1
    step = np.where(codes != 0, 2 * np.asarray(train_relations)[np.maximum(edge, 0)] + (codes < 0) + 1, 0)
    key = np.zeros(codes.shape[:-1], dtype=np.int64)
    for j in range(codes.shape[-1]):
        key = key * base + step[..., j]
    return key


def _distinct(keys: np.ndarray):
    """Distinct (row, key) pairs of a (n, K) key array, empty slots dropped."""
    rows = np.repeat(np.arange(keys.shape[0]), keys.shape[1])
    flat = keys.reshape(-1)
    pairs = np.unique(np.stack([rows[flat != 0], flat[flat != 0]], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def rule_table(keys: np.ndarray, relations: np.ndarray, labels: np.ndarray, num_relations: int,
               min_support: int = MIN_SUPPORT) -> tuple[np.ndarray, np.ndarray, int]:
    """Confidence of each rated (relation, pattern): among the estimation triples of r whose
    sampled paths contain p, the fraction that are positives; rated only with support >= min_support.
    Returns sorted combined keys, their confidences, and the number of (r, p) pairs seen."""
    rows, pat = _distinct(keys)
    base3 = (2 * num_relations + 1) ** 3
    combined = np.asarray(relations)[rows] * base3 + pat
    uniq, inv, support = np.unique(combined, return_inverse=True, return_counts=True)
    positives = np.bincount(inv.reshape(-1), weights=np.asarray(labels, float)[rows], minlength=len(uniq))
    rated = support >= min_support
    return uniq[rated], (positives / support)[rated], int(len(uniq))


def rule_features(keys: np.ndarray, relations: np.ndarray, table_keys: np.ndarray, table_conf: np.ndarray,
                  num_relations: int) -> dict:
    """rule_max, rule_mean over the distinct rated patterns on each triple's paths, and the flag
    rule_rated; all three 0 when no pattern is rated. keys (n, K); relations (n,)."""
    n = keys.shape[0]
    rows, pat = _distinct(keys)
    combined = np.asarray(relations)[rows] * (2 * num_relations + 1) ** 3 + pat
    pos = np.searchsorted(table_keys, combined)
    hit = (pos < len(table_keys)) & (table_keys[np.minimum(pos, len(table_keys) - 1)] == combined)
    rows, conf = rows[hit], table_conf[pos[hit]]
    count = np.bincount(rows, minlength=n)
    mean = np.divide(np.bincount(rows, weights=conf, minlength=n), count, out=np.zeros(n), where=count > 0)
    mx = np.zeros(n)
    np.maximum.at(mx, rows, conf)
    return dict(rule_max=mx, rule_mean=mean, rule_rated=(count > 0).astype(float))


def compare(v2: dict, test: dict, base: tuple[str, ...], candidate: tuple[str, ...], extra: dict | None = None,
            n_boot: int = N_BOOT) -> dict:
    """Candidate against base combiner, by the primary test's procedure (V2 fit, per-seed pairing,
    relation bootstrap). extra[split] = dict(pos={name: (n,)}, neg={name: (n, k)}) adds seed-free columns."""
    extra = extra or {"v2": dict(pos={}, neg={}), "test": dict(pos={}, neg={})}
    rows, out = {}, dict(seeds={})
    for s in SEEDS:
        cols = {split: {part: {**model_columns(d[part], s), **extra[split][part]} for part in ("pos", "neg")}
                for split, d in (("v2", v2), ("test", test))}
        out["seeds"][s] = {}
        for name, names in (("base", base), ("candidate", candidate)):
            comb, sp, sn = fit_and_score(cols["v2"]["pos"], cols["v2"]["neg"], cols["test"]["pos"],
                                         cols["test"]["neg"], names)
            rows[name, s] = masked_rows(sp, sn, np.ones(sn.shape, bool))
            out["seeds"][s][name] = dict(grouped_auc=float(rows[name, s].mean()), combiner=comb.to_dict())
    d = np.stack([rows["candidate", s] - rows["base", s] for s in SEEDS])
    out.update(delta_summary(d, np.asarray(test["relations"]), n_boot), base=list(base), candidate=list(candidate))
    return out


def s3_analysis(v2: dict, test: dict, rules: dict, n_boot: int = N_BOOT) -> dict:
    """(i) the pre-registered S3: B0' = B0 + rules against B1' = B0' + holonomy;
    (ii) added on 2026-11-22: B0 against B0'. rules[split] as compare's extra."""
    b0r = B0 + RULES
    return dict(holonomy_beyond_rules=compare(v2, test, b0r, b0r + ("holonomy",), rules, n_boot),
                rules_beyond_b0=compare(v2, test, B0, b0r, rules, n_boot))


def build_estimation(d: dict) -> dict:
    """S3's estimation set with its negatives and sampled paths (note of 2026-11-22, second)."""
    from rtvm.g2data import TrainGraph, sample_negatives, sample_paths
    train = d["train"]
    edges = estimation_set(train, d["num_relations"])
    pos = train[edges]
    known = set(map(tuple, train.tolist()))                       # training triples only (not valid or test)
    neg, fallback = sample_negatives(pos, train, known, d["num_entities"])
    graph = TrainGraph.from_training(train, d["num_entities"])
    pos_paths = sample_paths(graph, pos, own_edges=edges)
    neg_paths = sample_paths(graph, neg.reshape(-1, 3)).reshape(len(pos), neg.shape[1], *pos_paths.shape[1:])
    return dict(edges=edges, pos=pos, neg=neg, fallback=fallback, pos_paths=pos_paths, neg_paths=neg_paths)


def rules_from_estimation(est: dict, train: np.ndarray, num_relations: int) -> tuple[np.ndarray, np.ndarray, int]:
    m, k = est["neg"].shape[:2]
    keys = np.concatenate([pattern_keys(est["pos_paths"], train[:, 1], num_relations),
                           pattern_keys(est["neg_paths"].reshape(m * k, *est["pos_paths"].shape[1:]),
                                        train[:, 1], num_relations)])
    relations = np.r_[est["pos"][:, 1], est["neg"].reshape(-1, 3)[:, 1]]
    labels = np.r_[np.ones(m), np.zeros(m * k)]
    return rule_table(keys, relations, labels, num_relations)


def split_rules(d: dict, table_keys: np.ndarray, table_conf: np.ndarray) -> dict:
    """compare's extra for one split: rule features of its positives (n,) and negatives (n, k)."""
    m, k = d["neg"].shape[:2]
    R = d["num_relations"]
    pos = rule_features(pattern_keys(d["pos_paths"], d["train"][:, 1], R), d["pos"][:, 1], table_keys, table_conf, R)
    neg = rule_features(pattern_keys(gate.flat(d["neg_paths"], 2), d["train"][:, 1], R),
                        gate.flat(d["neg"], 1)[:, 1], table_keys, table_conf, R)
    return dict(pos=pos, neg={key: v.reshape(m, k) for key, v in neg.items()})


# ---------------------------------------------------------------- collapse check (exploratory)

def rotation_norms(model) -> np.ndarray:
    """||U_r - I||_F per relation over the whole block-diagonal rotation."""
    import torch
    with torch.no_grad():
        R = model.rotations()
        return (R - torch.eye(R.shape[-1])).flatten(1).norm(dim=1).double().numpy()


def collapse_check(runs: Path, dataset: str) -> dict:
    """For the holonomy model's best trial (seed 0) and its seed runs: the distribution over
    relations of ||U_r - I||_F, against a fresh draw at the run's initial scale (generator
    seed 0) and the largest value its rank allows (each block rotates `rank` planes, and one
    plane contributes at most 8 to the squared norm). No validation or test data is used."""
    import torch
    from rtvm.holonomy_model import HolonomyModel
    q = lambda x: dict(p10=float(np.quantile(x, 0.1)), median=float(np.median(x)),  # noqa: E731
                       p90=float(np.quantile(x, 0.9)), mean=float(np.mean(x)))
    out = {}
    for s, stem in gate.model_runs(runs, dataset, "holonomy").items():
        cfg = json.loads(stem.with_suffix(".json").read_text())["config"]
        weights = torch.load(f"{stem}_best.pt", map_location="cpu", weights_only=True)
        n_ent, n_rel = weights["psi"].shape[0], weights["U"].shape[0]
        n_blocks = weights["U"].shape[1]
        trained = HolonomyModel(n_ent, n_rel, rank=cfg["rank"], rotation_init=cfg["rotation_init"])
        trained.load_state_dict(weights)
        fresh = HolonomyModel(n_ent, n_rel, rank=cfg["rank"], rotation_init=cfg["rotation_init"],
                              generator=torch.Generator().manual_seed(0))
        t, f = rotation_norms(trained), rotation_norms(fresh)
        out[s] = dict(run=stem.name, rank=cfg["rank"], rotation_init=cfg["rotation_init"],
                      trained=q(t), fresh_at_initial_scale=q(f), largest_possible=float(np.sqrt(8 * n_blocks * cfg["rank"])),
                      median_ratio_trained_to_fresh=float(np.median(t) / np.median(f)))
    return out


# ---------------------------------------------------------------- main

def relation_labels(dataset: str) -> dict[int, str]:
    from pykeen.datasets import get_dataset
    return {int(i): str(lbl) for lbl, i in get_dataset(dataset=dataset).training.relation_to_id.items()}


def reproduce_gate(v2: dict, test: dict, decided: dict) -> dict:
    """The primary test rerun from the assembled files must give the gate's numbers exactly."""
    res = primary_test(v2, test)
    for s in SEEDS:
        for m in ("B0", "B1"):
            if abs(res["seeds"][s][m]["grouped_auc"] - decided["seeds"][str(s)][m]["grouped_auc"]) > 1e-12:
                sys.exit(f"{m}:{s} does not reproduce the gate")
    if abs(res["delta_bar"] - decided["delta_bar"]) > 1e-15 or not np.allclose(res["ci95"], decided["ci95"], 0, 1e-15):
        sys.exit("Delta-bar or its CI does not reproduce the gate")
    return dict(reproduces_gate=True, delta_bar=res["delta_bar"], ci95=res["ci95"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["basic", "s6", "s3", "s4", "s5", "s7wn", "s8", "s10", "collapse"])
    ap.add_argument("--s10-runs", default=str(ROOT / "runs/g2s10"), help="s10: output of experiments/g2_s10.py")
    ap.add_argument("--dataset", required=True, choices=["fb15k237", "wn18rr"])
    ap.add_argument("--baseline", choices=["rotate", "quate"], help="s6: whose relation rotations")
    ap.add_argument("--gate", default=str(ROOT / "runs/g2gate"))
    ap.add_argument("--runs", default=str(ROOT / "runs/g2"))
    ap.add_argument("--saved", default=str(ROOT / "saved/g2"))
    ap.add_argument("--data", default=str(ROOT / "data/g2"))
    ap.add_argument("--out", default=str(ROOT / "runs/g2secondary"))
    args = ap.parse_args()
    gate_dir, out_dir = Path(args.gate) / args.dataset, Path(args.out) / args.dataset
    if args.command == "collapse":                                        # checkpoints only: no split is read
        out_dir.mkdir(parents=True, exist_ok=True)
        state = gate.code_state()
        res = dict(kind="collapse check, exploratory (protocol note of 2026-11-22, second); no split is read",
                   dataset=args.dataset, seeds=collapse_check(Path(args.runs), args.dataset), head=state["head"],
                   dirty=bool(gate.git("status", "--porcelain")), created=time.strftime("%Y-%m-%d %H:%M:%S"))
        gate.write_json(out_dir / "collapse.json", res)
        print(f"wrote {out_dir / 'collapse.json'}")
        return
    freeze = gate.check_test_allowed()
    gate_json = gate_dir / "gate.json"
    if not gate_json.exists():
        sys.exit(f"{gate_json} missing: secondary analyses follow the decided gate")
    decided = json.loads(gate_json.read_text(encoding="utf-8"))
    state = gate.code_state()
    t0 = time.time()
    v2, test = gate.assemble(gate_dir, "v2"), gate.assemble(gate_dir, "test")
    gate.check_provenance({**v2["provenance"], **test["provenance"]}, state)
    meta = dict(dataset=args.dataset, freeze_commit=freeze, head=state["head"], frozen_tree=state["frozen_tree"],
                gate_sources=test["sources"], versions=gate.versions(), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "basic":
        fallback = np.load(gate_dir / "test_structure.npz")["fallback"]
        res = basic_analyses(v2, test, fallback, relation_labels(args.dataset))
        for key, value in res["check"]["grouped_auc"].items():           # must reproduce the gate exactly
            m, s = key.split(":")
            if abs(value - decided["seeds"][s][m]["grouped_auc"]) > 1e-12:
                sys.exit(f"{key}: {value} does not reproduce the gate's {decided['seeds'][s][m]['grouped_auc']}")
        res["check"]["reproduces_gate"] = True
        res.update(kind="S2, S7 (FB15k-237 part), S9; protocol notes of 2026-11-22; no pass rule", **meta)
        target = out_dir / "basic.json"

    elif args.command == "s10":
        if args.dataset != "fb15k237":
            sys.exit("S10 is on FB15k-237 only")
        check = reproduce_gate(v2, test, decided)
        runs = dense_runs(Path(args.s10_runs))
        committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
        feats, sources, collapse = {"v2": {}, "test": {}}, {}, {}
        for split, assembled in (("v2", v2), ("test", test)):
            d = gate.load_split(args.dataset, split, Path(args.saved), Path(args.data), allow_test=True,
                                committed_manifest=committed)
            if not np.array_equal(d["pos"][:, 1], assembled["relations"]):
                sys.exit(f"{split}: the loaded split and the gate's feature files disagree")
            for s, (table, stem, rec, net) in runs.items():
                feats[split][s] = holonomy_features(d, table)
        for s, (table, stem, rec, net) in runs.items():
            norms = rotation_norms(net)
            sources[f"holonomy_dense:{s}"] = dict(source=stem.relative_to(ROOT).as_posix() if stem.is_relative_to(ROOT)
                                                  else stem.name, weights_sha256=gate.sha256(Path(f"{stem}_best.pt")),
                                                  parameters=rec["parameters"], best_v1_auc=rec["best_v1_auc"],
                                                  best_epoch=rec["best_epoch"], epochs_run=rec["epochs_run"])
            collapse[s] = dict(median=float(np.median(norms)), p10=float(np.quantile(norms, 0.1)),
                               p90=float(np.quantile(norms, 0.9)), largest_possible=float(np.sqrt(8 * 2 * 32)))
        res = dict(kind="S10, dense rotations (rank 32, 3.80M parameters, outside the 2.0M budget): the primary "
                        "test with the dense model's holonomy; protocol note of 2026-11-22 (fourth); no pass rule",
                   check=check, result=s6_analysis(v2, test, feats), dense_runs=sources, collapse=collapse,
                   rank2_gate=dict(delta_bar=decided["delta_bar"], ci95=decided["ci95"]), **meta)
        target = out_dir / "s10.json"

    elif args.command == "s8":
        from pykeen.datasets import get_dataset
        committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
        d = gate.load_split(args.dataset, "test", Path(args.saved), Path(args.data), allow_test=True,
                            committed_manifest=committed)
        valid = get_dataset(dataset=args.dataset).validation.mapped_triples.numpy().astype(np.int64)
        known = np.concatenate([d["train"], valid, d["pos"]])
        per_model = {}
        for name, seeds in s8_models(Path(args.runs), args.dataset, d["training"]).items():
            per_seed = {}
            for s, ((tails, heads), stem) in seeds.items():
                per_seed[s] = dict(run=stem.relative_to(ROOT).as_posix(),
                                   **link_prediction(tails, heads, d["pos"], known, d["num_entities"], d["num_relations"]))
                print(f"{name} seed {s}: done ({time.time() - t0:.0f} s)", flush=True)
            summary = {m: dict(mean=float(np.mean([v[m] for v in per_seed.values()])),
                               std=float(np.std([v[m] for v in per_seed.values()], ddof=1)))
                       for m in ("mrr", "hits_at_1", "hits_at_10")}
            per_model[name] = dict(seeds=per_seed, mean_std_over_seeds=summary)
        res = dict(kind="S8, filtered link prediction on the test split, realistic ranks, both directions; std "
                        "over the 3 seeds with ddof = 1; protocol note of 2026-11-22 (fourth); no pass rule",
                   models=per_model, n_test=int(len(d["pos"])), n_known=int(len(known)), **meta)
        target = out_dir / "s8.json"

    elif args.command == "s4":
        check = reproduce_gate(v2, test, decided)
        committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
        splits = {split: gate.load_split(args.dataset, split, Path(args.saved), Path(args.data), allow_test=True,
                                         committed_manifest=committed) for split in ("v2", "test")}
        assembled = {"v2": v2, "test": test}
        for split in ("v2", "test"):
            if not np.array_equal(splits[split]["pos"][:, 1], assembled[split]["relations"]):
                sys.exit(f"{split}: the loaded split and the gate's feature files disagree")
        tables = holonomy_model_tables(Path(args.runs), args.dataset)
        by_length = {}
        for k in (1, 2, 3):                                   # (a) cycles of length k + 1
            feats = {split: {s: holonomy_features(restricted(splits[split], lambda L, k=k: L == k), table)
                             for s, (table, _) in tables.items()} for split in ("v2", "test")}
            res_k = s6_analysis(v2, test, feats)
            res_k["rows_with_a_cycle_pooled"] = {split: float(np.r_[(feats[split][0][0] > 0),
                                                                    (feats[split][0][1] > 0).ravel()].mean())
                                                 for split in ("v2", "test")}
            by_length[f"path_length_{k}"] = res_k
            print(f"path length {k} done ({time.time() - t0:.0f} s)", flush=True)
        short = {split: restricted(splits[split], lambda L: L <= 2, drop_n3=True) for split in ("v2", "test")}
        struct = {split: gate.structure_and_null(short[split], "cpu") for split in ("v2", "test")}   # (b) L <= 2
        feats = {split: {s: holonomy_features(short[split], table) for s, (table, _) in tables.items()}
                 for split in ("v2", "test")}
        l2 = primary_test(replace_holonomy(replace_structure(v2, struct["v2"]), feats["v2"]),
                          replace_holonomy(replace_structure(test, struct["test"]), feats["test"]))
        l2.pop("decision")
        l2["covered_share_pooled"] = {split: float(np.r_[struct[split]["pos/covered"],
                                                         struct[split]["neg/covered"].ravel()].mean())
                                      for split in ("v2", "test")}
        res = dict(kind="S4: (a) holonomy from cycles of one path length, B0 against B0 + it; (b) the whole "
                        "test with paths of length <= 2 and n3 = 0; protocol note of 2026-11-22 (fourth); no pass rule",
                   check=check, by_length=by_length, length_at_most_2=l2, **meta)
        target = out_dir / "s4.json"

    elif args.command in ("s5", "s7wn"):
        check = reproduce_gate(v2, test, decided)
        committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
        splits = {split: gate.load_split(args.dataset, split, Path(args.saved), Path(args.data), allow_test=True,
                                         committed_manifest=committed) for split in ("v2", "test")}
        for split, assembled in (("v2", v2), ("test", test)):
            if not np.array_equal(splits[split]["pos"][:, 1], assembled["relations"]):
                sys.exit(f"{split}: the loaded split and the gate's feature files disagree")
        if args.command == "s5":
            features, sources = {"v2": {}, "test": {}}, {}
            for s, stem in model_runs_any(Path(args.runs), args.dataset, "quate").items():
                for split in ("v2", "test"):
                    features[split][s] = baseline_scores(splits[split], stem)
                sources[f"quate:{s}"] = dict(source=stem.relative_to(ROOT).as_posix(),
                                             weights_sha256=gate.sha256(Path(f"{stem}_best.pt")))
                print(f"QuatE run {stem.name} scored ({time.time() - t0:.0f} s)", flush=True)
            res = dict(kind="S5, QuatE's score_hrt in RotatE's place in both combiners; protocol note of "
                            "2026-11-22 (fourth); no pass rule",
                       check=check, result=s5_analysis(v2, test, features), quate_sources=sources,
                       rotate_gate=dict(delta_bar=decided["delta_bar"], ci95=decided["ci95"]), **meta)
            target = out_dir / "s5.json"
        else:
            if args.dataset != "wn18rr":
                sys.exit("s7wn is the WN18RR part of S7")
            rev = reverse_in_training(splits["test"]["pos"], splits["test"]["train"])
            res = dict(kind="S7, WN18RR part: Delta-bar for test positives whose exact reverse is a training "
                            "triple, and for the rest; protocol note of 2026-11-22 (fourth); no pass rule",
                       check=check, groups=grouped_deltas(v2, test, {"reverse_in_training": rev, "rest": ~rev}),
                       n_reverse_in_training=int(rev.sum()), n_positives=int(len(rev)), **meta)
            target = out_dir / "s7_wn18rr.json"

    elif args.command == "s3":
        check = reproduce_gate(v2, test, decided)
        committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
        splits = {split: gate.load_split(args.dataset, split, Path(args.saved), Path(args.data), allow_test=True,
                                         committed_manifest=committed) for split in ("v2", "test")}
        for split, assembled in (("v2", v2), ("test", test)):
            if not np.array_equal(splits[split]["pos"][:, 1], assembled["relations"]):
                sys.exit(f"{split}: the loaded split and the gate's feature files disagree")
        d = splits["v2"]
        cache = out_dir / "s3_estimation.npz"
        code = dict(secondary_sha256=gate.sha256(Path(__file__)), frozen_tree=state["frozen_tree"],
                    manifest_sha256=d["manifest_sha256"])
        est = None
        if cache.exists():
            z = np.load(cache)
            if json.loads(str(z["provenance"])).get("code") == code:
                est = {key: z[key] for key in ("edges", "pos", "neg", "fallback", "pos_paths", "neg_paths")}
                print(f"estimation set from {cache.name}", flush=True)
        if est is None:
            est = build_estimation(d)
            gate.save_npz(cache, est, dict(code=code, created=time.strftime("%Y-%m-%d %H:%M:%S")))
            print(f"estimation set built ({time.time() - t0:.0f} s)", flush=True)
        keys, conf, n_seen = rules_from_estimation(est, d["train"], d["num_relations"])
        rules = {split: split_rules(splits[split], keys, conf) for split in ("v2", "test")}
        pooled = lambda r: float(np.r_[r["pos"]["rule_rated"], r["neg"]["rule_rated"].ravel()].mean())  # noqa: E731
        res = dict(kind="S3, typed path rules: (i) B0' = B0 + rules against B0' + holonomy (pre-registered); "
                        "(ii) B0 against B0' (added 2026-11-22); no pass rule",
                   check=check, **s3_analysis(v2, test, rules),
                   estimation=dict(n_positives=int(len(est["pos"])), n_negatives=int(est["neg"].shape[0] * est["neg"].shape[1]),
                                   n_fallback=int(est["fallback"].sum()), relations=int(len(np.unique(est["pos"][:, 1]))),
                                   pairs_seen=n_seen, pairs_rated=int(len(keys)), min_support=MIN_SUPPORT,
                                   per_relation=EST_PER_RELATION),
                   rated_share_pooled={split: pooled(rules[split]) for split in ("v2", "test")}, **meta)
        target = out_dir / "s3.json"

    else:                                                                 # s6
        if args.baseline is None:
            sys.exit("s6 needs --baseline")
        check = reproduce_gate(v2, test, decided)
        tables = baseline_tables(Path(args.runs), args.dataset, args.baseline)
        features, sources = {}, {}
        for split in ("v2", "test"):
            committed = gate.committed_file(f"saved/g2/{args.dataset}/manifest.json")
            d = gate.load_split(args.dataset, split, Path(args.saved), Path(args.data), allow_test=True,
                                committed_manifest=committed)
            if not np.array_equal(d["pos"][:, 1], (v2 if split == "v2" else test)["relations"]):
                sys.exit(f"{split}: the loaded split and the gate's feature files disagree")
            features[split] = {}
            for s, (table, stem) in tables.items():
                features[split][s] = holonomy_features(d, table)
                sources[f"{args.baseline}:{s}"] = dict(source=Path(stem).relative_to(ROOT).as_posix(),
                                                       weights_sha256=gate.sha256(Path(f"{stem}_best.pt")))
                gate.save_npz(out_dir / f"{split}_{args.baseline}_holonomy_seed{s}.npz",
                              dict(pos=features[split][s][0], neg=features[split][s][1]),
                              dict(**{k: meta[k] for k in ("head", "frozen_tree", "created")}, split=split,
                                   source=sources[f"{args.baseline}:{s}"]))
            print(f"{split}: {args.baseline}-rotation holonomy features ({time.time() - t0:.0f} s)", flush=True)
        res = dict(kind=f"S6, holonomy from {args.baseline}'s own relation rotations, B0 against B0 + it; "
                        "protocol note of 2026-11-22 (second); no pass rule",
                   check=check, result=s6_analysis(v2, test, features), rotation_sources=sources,
                   holonomy_model_gate=dict(delta_bar=decided["delta_bar"], ci95=decided["ci95"]), **meta)
        target = out_dir / f"s6_{args.baseline}.json"

    gate.write_json(target, res)
    reproduced = res.get("check", {}).get("reproduces_gate", False)  # s8 scores single models and has no check
    print(f"wrote {target} ({time.time() - t0:.0f} s)" + ("; the gate is reproduced" if reproduced else ""))


if __name__ == "__main__":
    main()
