"""G2 holonomy model: one block rotation per relation, one state per entity.

Protocol (notes/protocol.md, "Models"): 2 blocks of 64 (d = 128), rank-r Cayley
rotations, trained on training triples and training cycles only, with the loss of
proposal Section 5.2 restricted to the terms that exist for a knowledge graph:

    L = L_single + alpha1 * mean_true ||H_c - I||_F^2
                 + alpha2 * mean_corrupt max(0, m - ||H_c' - I||_F)^2

L_single is the single-triple term on ||U_r psi_h - psi_t|| (self-adversarial NSSA,
as RotatE uses). The test's feature is the triple holonomy score; the single-triple
score is reported only in S8.

Conventions:
  * A forward step along a training fact (a, q, b) maps a's frame to b's: U_q.
    A backward step uses U_q^T. A path p = (s_1, ..., s_L) from h to t has transport
    U_p = U_{s_L} ... U_{s_1}.
  * The cycle that closes p with the evaluated edge (h, r, t) returns from t to h
    along r backwards, so H_c = U_r^T U_p, and its defect is ||H_c - I||_F over the
    whole d x d block-diagonal matrix.
  * Rotations live in a "table" of 2R + 1 entries: [U_0..U_{R-1}, U_0^T..U_{R-1}^T, I].
    Step index q means U_q, q + R means U_q^T, and 2R (the identity) is padding.

Exercises (spec: tests/test_holonomy_model.py, run `pytest -m g2model -rs`):
``path_transport`` and ``cycle_loss``.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .rotations import block_apply, cayley, lowrank_skew


class HolonomyModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, *, n_blocks: int = 2, b: int = 64,
                 rank: int = 2, rotation_init: float = 0.1, entity_init: float = 1.0,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.num_relations = num_relations
        self.n_blocks, self.b = n_blocks, b
        d = n_blocks * b
        # Entity states: N(0, entity_init^2 / d) per coordinate, so ||psi|| is about entity_init.
        self.psi = nn.Parameter(torch.randn(num_entities, d, generator=generator) * entity_init / d ** 0.5)
        # Rank-r generators A = U V^T - V U^T per relation and block (rotations.lowrank_skew).
        self.U = nn.Parameter(torch.randn(num_relations, n_blocks, b, rank, generator=generator) * rotation_init)
        self.V = nn.Parameter(torch.randn(num_relations, n_blocks, b, rank, generator=generator) * rotation_init)

    @property
    def d(self) -> int:
        return self.n_blocks * self.b

    def rotations(self) -> Tensor:
        """Exactly orthogonal block rotations, shape (R, n_blocks, b, b)."""
        return cayley(lowrank_skew(self.U, self.V))

    def table(self, R: Tensor | None = None) -> Tensor:
        """The (2R + 1, n_blocks, b, b) step table: rotations, their transposes, then I."""
        R = self.rotations() if R is None else R
        eye = torch.eye(self.b, dtype=R.dtype, device=R.device).expand(1, self.n_blocks, self.b, self.b)
        return torch.cat([R, R.transpose(-1, -2), eye])

    def single_distance(self, h: Tensor, r: Tensor, t: Tensor, R: Tensor | None = None) -> Tensor:
        """||U_r psi_h - psi_t|| for index tensors of one shape; returns that shape."""
        R = self.rotations() if R is None else R
        moved = block_apply(R[r], self.psi[h])
        return (moved - self.psi[t]).norm(dim=-1)


# ---------------------------------------------------------------- paths as table indices

def steps_from_codes(codes: np.ndarray, edge_relations: np.ndarray, num_relations: int) -> np.ndarray:
    """Signed path codes (rtvm.g2data) -> table step indices, same shape, padding -> 2R.

    edge_relations[e] is the relation of training edge e.
    """
    codes = np.asarray(codes)
    out = np.full(codes.shape, 2 * num_relations, dtype=np.int64)
    used = codes != 0
    edge = np.abs(codes[used]) - 1
    rel = np.asarray(edge_relations)[edge]
    out[used] = np.where(codes[used] > 0, rel, rel + num_relations)
    return out


def path_transport(table: Tensor, steps: Tensor) -> Tensor:
    """EXERCISE. Transport U_p of each path, shape (n, n_blocks, b, b).

    steps: int64 (n, MAX_LEN), table indices of the path's steps in order, padded
    with the identity index 2R after the last step.

    U_p = T[steps[:, L-1]] @ ... @ T[steps[:, 1]] @ T[steps[:, 0]]: the FIRST step is
    applied first, so it sits on the right. Padding needs no special case, because
    the padding entry is the identity.

    Gather the table rows for one step column at a time (table[steps[:, j]]) and
    multiply; never build an (n, MAX_LEN, ...) gather of all steps at once.
    """
    raise NotImplementedError("path_transport")


def cycle_defect(table: Tensor, transport: Tensor, closing: Tensor) -> Tensor:
    """||U_r^T U_p - I||_F over the whole block-diagonal matrix, one value per cycle.

    closing: relation ids r (n,). Shapes: transport (n, n_blocks, b, b) -> (n,).
    """
    num_relations = (table.shape[0] - 1) // 2
    H = table[closing + num_relations] @ transport
    eye = torch.eye(H.shape[-1], dtype=H.dtype, device=H.device)
    return (H - eye).pow(2).sum(dim=(-3, -2, -1)).clamp_min(0).sqrt()


@torch.no_grad()
def triple_holonomy_scores(model: HolonomyModel, relations: np.ndarray, steps: np.ndarray,
                           device: str | torch.device = "cpu", chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """Protocol "triple holonomy score": mean defect over the triple's saved cycles.

    relations: (n,) the evaluated triples' relations. steps: (n, K, MAX_LEN) table
    indices from steps_from_codes (a path slot is empty when all its steps are padding).
    Returns (scores, covered): scores is 0 where the triple has no path.
    Evaluation precision is strict FP32: the caller turns TF32 off.
    """
    steps = np.asarray(steps)
    n, k, _ = steps.shape
    pad = 2 * model.num_relations
    valid = (steps != pad).any(axis=-1)                       # (n, K)
    tri, slot = np.nonzero(valid)
    sums = np.zeros(n)
    table = model.table().to(device)
    for s in range(0, len(tri), chunk):
        i, j = tri[s:s + chunk], slot[s:s + chunk]
        st = torch.as_tensor(steps[i, j], device=device)
        rel = torch.as_tensor(np.asarray(relations)[i], device=device)
        d = cycle_defect(table, path_transport(table, st), rel).double().cpu().numpy()
        np.add.at(sums, i, d)
    count = valid.sum(axis=1)
    covered = count > 0
    scores = np.where(covered, sums / np.maximum(count, 1), 0.0)
    return scores, covered


# ---------------------------------------------------------------- losses

def cycle_loss(defect_true: Tensor, defect_corrupt: Tensor, *, alpha1: float, alpha2: float,
               margin: float) -> Tensor:
    """EXERCISE. Proposal Section 5.2, cycle terms, as means rather than sums:

        alpha1 * mean(defect_true ** 2) + alpha2 * mean(max(0, margin - defect_corrupt) ** 2)

    Means keep the weights comparable across batch sizes. An empty input contributes 0
    (a batch can have no covered triple), never NaN.
    """
    raise NotImplementedError("cycle_loss")


def nssa_loss(pos_dist: Tensor, neg_dist: Tensor, *, gamma: float, temperature: float) -> Tensor:
    """Self-adversarial negative sampling loss (RotatE), on distances.

    pos_dist (n,), neg_dist (n, k). Scores are gamma - distance. Negative weights are a
    softmax of temperature * score, detached (no gradient through the weights).
    """
    pos = -nn.functional.logsigmoid(gamma - pos_dist)
    w = torch.softmax(temperature * (gamma - neg_dist), dim=-1).detach()
    neg = -(w * nn.functional.logsigmoid(neg_dist - gamma)).sum(dim=-1)
    return 0.5 * (pos.mean() + neg.mean())


def corrupt_closing(heads: np.ndarray, rels: np.ndarray, tails: np.ndarray, train_keys: np.ndarray,
                    num_entities: int, num_relations: int, rng: np.random.Generator,
                    tries: int = 20) -> np.ndarray:
    """Corrupted cycles: the same path, closed with a different relation r'.

    Draws r' uniformly from all relations except r, rejecting r' when (h, r', t) is a
    training fact (train_keys = sorted fact_key values of the training triples). After
    `tries` rounds, leftover draws keep their last r' (rare; it only weakens one negative).
    """
    rels = np.asarray(rels)
    out = (rels + rng.integers(1, num_relations, size=len(rels))) % num_relations
    for _ in range(tries):
        bad = np.isin(fact_key(heads, out, tails, num_entities, num_relations), train_keys)
        if not bad.any():
            break
        out[bad] = (rels[bad] + rng.integers(1, num_relations, size=bad.sum())) % num_relations
    return out


def fact_key(h, r, t, num_entities: int, num_relations: int) -> np.ndarray:
    """One int64 per triple, for fast set membership with np.isin."""
    h, r, t = (np.asarray(x, dtype=np.int64) for x in (h, r, t))
    return (h * num_relations + r) * num_entities + t
