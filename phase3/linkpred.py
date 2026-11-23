"""Filtered link prediction with realistic ranks, for the Phase 3 bake-off.

The four ranking functions are copies of experiments/g2_secondary.py's S8 code (which a test there
checks against PyKEEN's RankBasedEvaluator); tests/test_phase3.py checks that these copies give
the same ranks as the originals. The scorers here run on the training device in strict FP32
(protocol: FP32 evaluation), because the bake-off evaluates every 10 epochs during training.
"""
from __future__ import annotations

import numpy as np
import torch


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
            ranks.append(realistic_ranks(scores.cpu() if torch.is_tensor(scores) else scores, targets, rows, cols))
    ranks = np.concatenate(ranks)
    return dict(mrr=float(np.mean(1 / ranks)), hits_at_1=float(np.mean(ranks <= 1)),
                hits_at_10=float(np.mean(ranks <= 10)), n_ranks=int(len(ranks)), ranks=ranks)


# ---------------------------------------------------------------- scorers (strict FP32, on the model's device)

def _strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def pykeen_scorers(net, slice_size: int = 2048):
    """RotatE: PyKEEN's score_t / score_h against every entity, in slices of candidates."""
    device = next(net.parameters()).device

    def tails(h, r):
        _strict_fp32()
        with torch.no_grad():
            return net.score_t(torch.as_tensor(np.stack([h, r], 1), dtype=torch.long, device=device),
                               slice_size=slice_size).float()

    def heads(r, t):
        _strict_fp32()
        with torch.no_grad():
            return net.score_h(torch.as_tensor(np.stack([r, t], 1), dtype=torch.long, device=device),
                               slice_size=slice_size).float()
    return tails, heads


def quate_scorers(net):
    """QuatE (rtvm.baselines.NormalizedQuatEInteraction) against every entity as a matrix product:
    the score -sum <h (x) r^, t> is -<M_r h, t>, with M_r the per-coordinate 4x4 matrix of right
    multiplication by the normalised r^ (as experiments/g2_secondary.quate_rotations builds it)."""
    device = next(net.parameters()).device
    with torch.no_grad():
        ent = net.entity_representations[0]._embeddings.weight.float()           # (E, d * 4)
        rel = net.relation_representations[0]._embeddings.weight.float()
        table = net.interaction.table.float()                                    # (4, 4, 4): e_i e_j = sum_k T_ijk e_k
        r = rel.reshape(rel.shape[0], -1, 4)
        r = r / r.norm(p=2, dim=-1, keepdim=True)
        M = torch.einsum("rdj,ijk->rdki", r, table)                              # (R, d, 4, 4): (h (x) r)_k = sum_i M_ki h_i
    E = ent.reshape(ent.shape[0], -1, 4)

    def tails(h, r_):
        _strict_fp32()
        with torch.no_grad():
            q = (M[torch.as_tensor(r_, device=device)] @ E[torch.as_tensor(h, device=device)][..., None])[..., 0]
            return -(q.reshape(q.shape[0], -1) @ ent.T)

    def heads(r_, t):
        _strict_fp32()
        with torch.no_grad():
            u = (M[torch.as_tensor(r_, device=device)].transpose(-1, -2)
                 @ E[torch.as_tensor(t, device=device)][..., None])[..., 0]
            return -(u.reshape(u.shape[0], -1) @ ent.T)
    return tails, heads


def cayley_scorers(model):
    """Rank-r Cayley blocks: -||U_r psi_h - psi_t|| against every entity; head queries use
    ||U_r psi_h - psi_t|| = ||psi_h - U_r^T psi_t|| (U_r is orthogonal)."""
    from rtvm.rotations import block_apply
    device = model.psi.device
    with torch.no_grad():
        R, psi = model.rotations().float(), model.psi.float()

    def dist(x):
        return torch.cdist(x, psi, compute_mode="donot_use_mm_for_euclid_dist")

    def tails(h, r):
        _strict_fp32()
        with torch.no_grad():
            return -dist(block_apply(R[torch.as_tensor(r, device=device)], psi[torch.as_tensor(h, device=device)]))

    def heads(r, t):
        _strict_fp32()
        with torch.no_grad():
            return -dist(block_apply(R[torch.as_tensor(r, device=device)].transpose(-1, -2),
                                     psi[torch.as_tensor(t, device=device)]))
    return tails, heads
