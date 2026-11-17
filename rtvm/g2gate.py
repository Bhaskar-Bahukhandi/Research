"""G2 primary test (notes/protocol.md, "Primary test (the gate)"): features, combiners, bootstrap, decision.

Spec: tests/test_g2gate.py. The script that runs it is experiments/g2_gate.py.

Fixed here before the test split is touched (the Timing rule), as implementation choices the
protocol leaves open:

  * Structure features, in this column order: covered; log(1 + n1), log(1 + n2), log(1 + n3);
    the shares of lengths 1, 2, 3 among the triple's SAMPLED paths (0 when uncovered);
    log(1 + degree) of h and of t; null holonomy (0 when uncovered).
  * Null rotations: one generator torch.Generator().manual_seed(0); relation r = 0, 1, ...,
    R - 1 in order draws rtvm.synthetic.random_rotation(2, 64) from it. FP32, like the model.
  * Holonomy scores (null and learned) are computed once per distinct (path steps, closing
    relation) pattern and spread back to the cycles. A cycle's defect depends only on that
    pattern, so this is the protocol's mean over each triple's cycles, computed faster.
  * Combiner: features standardised with the mean and population standard deviation of the
    V2 rows it is fitted on (a constant column keeps scale 1); scikit-learn LogisticRegression
    with C = 1.0 and l1_ratio = 0 (pure L2), lbfgs, tol 1e-6, max_iter 10,000, unpenalised
    intercept and no class weights; a fit that does not converge is an error. Test rows are
    ranked by the decision function (log-odds), which ranks like the probability without
    rounding ties at 0 or 1.
  * Bootstrap: the clusters are the relations that occur among the test positives (sorted).
    numpy.random.default_rng(0).integers(0, G, size=(10_000, G)) draws all resamples at once;
    row b lists the G clusters of resample b. Each drawn cluster brings all its positives
    with their per-positive grouped-AUC differences. Within a resample, Delta_s is the mean
    difference over the drawn positives (a cluster drawn twice counts twice), and Delta-bar
    is the mean over seeds. The 95% CI is numpy.percentile(..., [2.5, 97.5]) (linear).
  * Decision: the protocol's four outcomes. One case is not named there: CI lower bound
    >= 0.005 while Delta_s <= 0 in some seed. It is not a pass, and is reported as
    "not pass: not positive in every seed".
  * S11 (protocol note 2026-11-17), reported next to the primary CI and never used by the
    decision: a two-way (relation x entity, "pigeonhole") bootstrap of the same per-positive
    differences. Relation multiplicities are the primary bootstrap's draws; entity multiplicities
    come from numpy.random.default_rng(1).integers(0, E, size=(10_000, E)), over the entities
    that occur as h or t among the test positives (sorted). A positive (h, r, t) has weight
    w_r * v_h * v_t (v_h ** 2 when h = t); Delta_s is the weighted mean of seed s's differences
    and Delta-bar the mean over seeds. The entity draws are made in row chunks to bound memory;
    tests/test_g2gate.py checks that this gives the same draws as the single call.
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .g2eval import grouped_auc_rows
from .holonomy_model import cycle_defect, path_transport, triple_scores_from_cycles
from .synthetic import random_rotation

STRUCTURE = ("covered", "log1p_n1", "log1p_n2", "log1p_n3", "share_len1", "share_len2", "share_len3",
             "log1p_deg_h", "log1p_deg_t", "null_holonomy")
B0 = STRUCTURE + ("rotate",)
B1 = B0 + ("holonomy",)
MIN_EFFECT = 0.005
N_BOOT = 10_000
BOOT_SEED = 0
ENTITY_SEED = 1                 # S11 only (protocol note 2026-11-17)
NULL_SEED = 0
N_BLOCKS, BLOCK = 2, 64


# ---------------------------------------------------------------- features

def structure_features(triples: np.ndarray, counts: np.ndarray, codes: np.ndarray,
                       degree: np.ndarray) -> dict[str, np.ndarray]:
    """Every structure feature except null holonomy, one (n,) array per name.

    triples (n, 3); counts (n, 3) exact n1, n2, n3; codes (n, K, MAX_LEN) saved path codes
    (rtvm.g2data); degree (num_entities,) training facts per entity (TrainGraph.degree).
    """
    triples, counts, codes = np.asarray(triples), np.asarray(counts), np.asarray(codes)
    covered = counts.sum(axis=1) > 0
    lengths = (codes != 0).sum(axis=-1)                         # (n, K): 0 marks an empty slot
    n_paths = (lengths > 0).sum(axis=1)
    if not np.array_equal(n_paths > 0, covered):
        raise ValueError("saved paths and exact counts disagree on which triples are covered")
    shares = np.stack([(lengths == L).sum(axis=1) for L in (1, 2, 3)], axis=1) / np.maximum(n_paths, 1)[:, None]
    deg = np.asarray(degree)
    out = dict(covered=covered.astype(float))
    for L in (1, 2, 3):
        out[f"log1p_n{L}"] = np.log1p(counts[:, L - 1].astype(float))
        out[f"share_len{L}"] = shares[:, L - 1]
    out["log1p_deg_h"] = np.log1p(deg[triples[:, 0]].astype(float))
    out["log1p_deg_t"] = np.log1p(deg[triples[:, 2]].astype(float))
    return out


def null_rotations(num_relations: int, seed: int = NULL_SEED) -> torch.Tensor:
    """Haar-random block rotations, (R, 2, 64, 64) FP32: relation r is the r-th draw from one generator."""
    gen = torch.Generator().manual_seed(seed)
    return torch.stack([random_rotation(N_BLOCKS, BLOCK, generator=gen) for _ in range(num_relations)])


def rotation_table(R: torch.Tensor) -> torch.Tensor:
    """The (2R + 1, n_blocks, b, b) step table [U_0..U_{R-1}, U_0^T..U_{R-1}^T, I] (HolonomyModel.table)."""
    eye = torch.eye(R.shape[-1], dtype=R.dtype, device=R.device).expand(1, *R.shape[1:])
    return torch.cat([R, R.transpose(-1, -2), eye])


@torch.no_grad()
def triple_holonomy(table: torch.Tensor, relations: np.ndarray, steps: np.ndarray,
                    device: str | torch.device = "cpu", chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Protocol "triple holonomy score" (mean ||H_c - I||_F over the triple's saved cycles, 0 if
    uncovered) and the covered flag, from a step table.

    relations (n,) the closing relations; steps (n, K, MAX_LEN) table indices
    (rtvm.holonomy_model.steps_from_codes). Strict FP32: TF32 is off inside this call and
    the previous flags are restored afterwards. Build `table` in FP32 too (the gate script
    switches TF32 off before anything else).
    """
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        return _triple_holonomy(table, relations, steps, device, chunk)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _triple_holonomy(table, relations, steps, device, chunk):
    steps, relations = np.asarray(steps, dtype=np.int64), np.asarray(relations, dtype=np.int64)
    n, base = len(steps), table.shape[0]
    pad = base - 1
    tri, slot = np.nonzero((steps != pad).any(axis=-1))
    if len(tri) == 0:
        return np.zeros(n), np.zeros(n, bool)
    cyc = steps[tri, slot]                                     # (m, MAX_LEN)
    key = relations[tri].copy()                                # one int64 per (steps, closing relation)
    for j in range(cyc.shape[1]):
        key = key * base + cyc[:, j]
    uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    u_steps, u_rel = cyc[first], relations[tri[first]]
    table = table.to(device=device, dtype=torch.float32)
    u_def = np.empty(len(uniq))
    for s in range(0, len(uniq), chunk):
        st = torch.as_tensor(u_steps[s:s + chunk], device=device)
        rel = torch.as_tensor(u_rel[s:s + chunk], device=device)
        u_def[s:s + chunk] = cycle_defect(table, path_transport(table, st), rel).double().cpu().numpy()
    return triple_scores_from_cycles(n, tri, u_def[inverse.reshape(-1)])


def feature_matrix(features: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    """Stack named feature arrays on a new last axis: (n,) arrays give (n, f), (n, k) give (n, k, f)."""
    return np.stack([np.asarray(features[k], dtype=float) for k in names], axis=-1)


# ---------------------------------------------------------------- combiners

@dataclass
class Combiner:
    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    n_iter: int

    def decision(self, X: np.ndarray) -> np.ndarray:
        """Log-odds of being a positive; higher = more plausible."""
        return ((np.asarray(X, float) - self.mean) / self.scale) @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}


def fit_combiner(X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> Combiner:
    """Protocol "Two combiners": logistic regression (L2, C = 1.0) on V2-standardised features."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    X, y = np.asarray(X, float), np.asarray(y)
    if X.shape[1] != len(names) or not np.isfinite(X).all():
        raise ValueError("feature matrix has the wrong width or a non-finite value")
    mean, std = X.mean(axis=0), X.std(axis=0)
    scale = np.where(std > 0, std, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        lr = LogisticRegression(C=1.0, l1_ratio=0.0, solver="lbfgs", tol=1e-6, max_iter=10_000)
        lr.fit((X - mean) / scale, y)
    return Combiner(tuple(names), mean, scale, lr.coef_.ravel().copy(), float(lr.intercept_[0]),
                    int(np.max(lr.n_iter_)))


def combiner_rows(comb: Combiner, X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
    """Per-positive grouped AUC of a fitted combiner. X_pos (n, f); X_neg (n, k, f)."""
    n, k, f = X_neg.shape
    return grouped_auc_rows(comb.decision(X_pos), comb.decision(X_neg.reshape(n * k, f)).reshape(n, k))


# ---------------------------------------------------------------- bootstrap and decision

def relation_bootstrap(diffs: np.ndarray, relations: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> np.ndarray:
    """Delta-bar in each of n_boot relation-cluster resamples (see the module docstring).

    diffs (S, n): per-seed, per-positive grouped-AUC differences B1 - B0. relations (n,).
    """
    diffs = np.atleast_2d(np.asarray(diffs, float))
    _, cluster = np.unique(np.asarray(relations), return_inverse=True)
    cluster = cluster.reshape(-1)
    G = int(cluster.max()) + 1
    sums = np.stack([np.bincount(cluster, weights=d, minlength=G) for d in diffs])     # (S, G)
    sizes = np.bincount(cluster, minlength=G).astype(float)                           # (G,)
    draws = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    times = np.zeros((n_boot, G))                       # times[b, g]: how often resample b drew cluster g
    np.add.at(times, (np.repeat(np.arange(n_boot), G), draws.ravel()), 1.0)
    per_seed = (times @ sums.T) / (times @ sizes)[:, None]                            # (n_boot, S)
    return per_seed.mean(axis=1)


def _row_counts(draws: np.ndarray, size: int) -> np.ndarray:
    """counts[b, g]: how often row b of draws (b, k) contains g, for g < size."""
    b = len(draws)
    flat = (draws + np.arange(b)[:, None] * size).ravel()
    return np.bincount(flat, minlength=b * size).reshape(b, size).astype(float)


def two_way_bootstrap(diffs: np.ndarray, relations: np.ndarray, heads: np.ndarray, tails: np.ndarray,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED, entity_seed: int = ENTITY_SEED,
                      chunk: int = 250) -> np.ndarray:
    """S11: Delta-bar in each of n_boot relation x entity resamples (see the module docstring).

    diffs (S, n) as for relation_bootstrap; relations, heads, tails (n,) of the test positives.
    """
    diffs = np.atleast_2d(np.asarray(diffs, float))
    n = diffs.shape[1]
    if not len(relations) == len(heads) == len(tails) == n:
        raise ValueError("diffs, relations, heads and tails must describe the same positives")
    _, rel = np.unique(np.asarray(relations), return_inverse=True)
    rel = rel.reshape(-1)
    G = int(rel.max()) + 1
    _, ent = np.unique(np.concatenate([np.asarray(heads), np.asarray(tails)]), return_inverse=True)
    ent = ent.reshape(-1)
    h, t = ent[:n], ent[n:]
    E = int(ent.max()) + 1
    rel_draws = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))     # the primary bootstrap's draws
    ent_rng = np.random.default_rng(entity_seed)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        w = _row_counts(rel_draws[start:stop], G)                                  # (b, G)
        v = _row_counts(ent_rng.integers(0, E, size=(stop - start, E)), E)         # (b, E)
        weight = w[:, rel] * v[:, h] * v[:, t]                                     # (b, n)
        total = weight.sum(axis=1)
        if np.any(total == 0):
            raise ValueError("a two-way resample has total weight 0 (protocol note 2026-11-17: an error)")
        out[start:stop] = ((weight @ diffs.T) / total[:, None]).mean(axis=1)
    return out


def s11_report(decision: str, ci_low: float, min_effect: float = MIN_EFFECT) -> str:
    """How the gate is reported next to S11 (protocol note 2026-11-17). Never changes the decision."""
    if decision == "pass" and ci_low < min_effect:
        return "pass, not robust to entity dependence"
    return decision


def decide(delta_per_seed, ci_low: float, ci_high: float, min_effect: float = MIN_EFFECT) -> str:
    """Protocol "Decision", in its order; plus the one case it does not name (module docstring)."""
    every_seed = all(d > 0 for d in delta_per_seed)
    if ci_low >= min_effect and every_seed:
        return "pass"
    if ci_high < 0:
        return "fail: holonomy hurts"
    if ci_low <= 0:
        return "fail: no added information"
    if ci_low < min_effect:
        return "negligible"
    return "not pass: not positive in every seed"


def primary_test(v2: dict, test: dict, seeds=(0, 1, 2), n_boot: int = N_BOOT, two_way: bool = False) -> dict:
    """Fit B0 and B1 on V2 per seed, score the test split, bootstrap, decide.

    v2 and test: dict(pos=features of the positives, neg=features of their negatives,
    relations=(n,) the positives' relations), where a features entry maps each name in B1
    to an array: (n,) for positives and (n, k) for negatives. Model features are keyed per
    seed as "rotate:<s>" and "holonomy:<s>". two_way=True adds S11 under "s11"; it needs
    test["triples"] (n, 3), the positives as (h, r, t), and does not affect "decision".
    """
    def matrices(split, part, s, names):
        cols = {**{k: split[part][k] for k in STRUCTURE},
                "rotate": split[part][f"rotate:{s}"], "holonomy": split[part][f"holonomy:{s}"]}
        return feature_matrix(cols, names)

    if two_way:                                         # checked first, so a bad input stops before any result
        triples = np.asarray(test["triples"])
        if not np.array_equal(triples[:, 1], test["relations"]):
            raise ValueError("test triples and relations disagree")
    out, diffs = dict(seeds={}), []
    for s in seeds:
        res, auc_rows = {}, {}
        for name, names in (("B0", B0), ("B1", B1)):
            Vp, Vn = matrices(v2, "pos", s, names), matrices(v2, "neg", s, names)       # (m, f), (m, k, f)
            X = np.concatenate([Vp, Vn.reshape(-1, len(names))])
            y = np.r_[np.ones(len(Vp)), np.zeros(Vn.shape[0] * Vn.shape[1])]
            comb = fit_combiner(X, y, names)
            auc_rows[name] = combiner_rows(comb, matrices(test, "pos", s, names), matrices(test, "neg", s, names))
            res[name] = dict(grouped_auc=float(auc_rows[name].mean()), combiner=comb.to_dict())
        d = auc_rows["B1"] - auc_rows["B0"]
        diffs.append(d)
        out["seeds"][s] = dict(**res, delta=float(d.mean()))
    deltas = [out["seeds"][s]["delta"] for s in seeds]
    boot = relation_bootstrap(np.stack(diffs), test["relations"], n_boot=n_boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out.update(delta_bar=float(np.mean(deltas)), ci95=[float(lo), float(hi)], n_boot=n_boot,
               n_clusters=int(len(np.unique(test["relations"]))), n_positives=int(len(test["relations"])),
               decision=decide(deltas, lo, hi))
    if two_way:
        s11 = dict(kind="S11, two-way (relation x entity) bootstrap; reported, never decides (note 2026-11-17)",
                   n_entities=int(len(np.unique(triples[:, [0, 2]]))))
        try:
            boot2 = two_way_bootstrap(np.stack(diffs), triples[:, 1], triples[:, 0], triples[:, 2], n_boot=n_boot)
        except ValueError as e:                         # an S11 error is reported; the primary result still stands
            s11.update(error=str(e), report=out["decision"])
        else:
            lo2, hi2 = (float(x) for x in np.percentile(boot2, [2.5, 97.5]))
            s11.update(ci95=[lo2, hi2], narrower_than_primary=bool(hi2 - lo2 < hi - lo),
                       report=s11_report(out["decision"], lo2))
        out["s11"] = s11
    return out
