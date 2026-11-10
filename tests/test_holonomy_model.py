"""Spec for rtvm/holonomy_model.py (G2 holonomy model). Run `pytest -m g2model -rs`."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from rtvm.g2data import encode
from rtvm.holonomy_model import (HolonomyModel, corrupt_closing, cycle_defect, cycle_loss, fact_key,
                                 nssa_loss, path_transport, steps_from_codes, triple_holonomy_scores)
from rtvm.utils import count_parameters

pytestmark = pytest.mark.g2model

R_, E_, B_, NB_ = 5, 7, 4, 2       # relations, entities, block size, blocks (d = 8) for small tests


def small_model(rank=2, seed=0):
    return HolonomyModel(E_, R_, n_blocks=NB_, b=B_, rank=rank, rotation_init=0.5,
                         generator=torch.Generator().manual_seed(seed))


def loop_transport(table, steps):
    """Reference: compose step by step, first step rightmost."""
    out = []
    for row in steps.tolist():
        M = table[row[0]]
        for s in row[1:]:
            M = table[s] @ M
        out.append(M)
    return torch.stack(out)


# ---------------------------------------------------------------- provided

def test_rotations_are_orthogonal_with_det_one():
    R = small_model().rotations()
    eye = torch.eye(B_)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand_as(R), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(R_, NB_), atol=1e-5)


@pytest.mark.parametrize("rank, expected", [(1, 1_917_312), (2, 1_977_984), (4, 2_099_328)])
def test_parameter_budget_on_fb15k237(rank, expected):
    """The counts behind the 2026-11-06 budget note: 14,505 entities, 237 relations, 2 x 64."""
    model = HolonomyModel(14_505, 237, rank=rank)
    assert count_parameters(model) == expected


def test_table_layout():
    m = small_model()
    R = m.rotations()
    T = m.table(R)
    assert T.shape == (2 * R_ + 1, NB_, B_, B_)
    assert torch.equal(T[:R_], R) and torch.equal(T[R_:2 * R_], R.transpose(-1, -2))
    assert torch.equal(T[2 * R_], torch.eye(B_).expand(NB_, B_, B_))


def test_steps_from_codes():
    edge_relations = np.array([3, 0, 4])
    codes = np.array([[encode(0, True), encode(2, False), 0], [0, 0, 0]])
    steps = steps_from_codes(codes, edge_relations, R_)
    assert steps.tolist() == [[3, 4 + R_, 2 * R_], [2 * R_] * 3]


def test_single_distance():
    m = small_model()
    R = m.rotations()
    h, r, t = torch.tensor([1, 2]), torch.tensor([0, 4]), torch.tensor([3, 3])
    for i in range(2):
        moved = torch.cat([R[r[i], j] @ m.psi[h[i]].view(NB_, B_)[j] for j in range(NB_)])
        assert torch.isclose(m.single_distance(h, r, t)[i], (moved - m.psi[t[i]]).norm(), atol=1e-5)


def test_cycle_defect_is_zero_when_path_equals_closing_rotation():
    m = small_model()
    T = m.table()
    closing = torch.tensor([1, 3])
    assert torch.allclose(cycle_defect(T, T[closing], closing), torch.zeros(2), atol=1e-4)


def test_corrupt_closing_avoids_true_relation_and_training_facts():
    rng = np.random.default_rng(0)
    train = np.array([[0, r, 1] for r in range(4)] + [[2, 0, 3]])      # (0, r, 1) true for r = 0..3
    keys = np.sort(fact_key(train[:, 0], train[:, 1], train[:, 2], 10, 6))
    h, r, t = np.zeros(500, int), np.zeros(500, int), np.ones(500, int)
    out = corrupt_closing(h, r, t, keys, 10, 6, rng)
    assert set(out.tolist()) == {4, 5}                                    # the only relations not true for (0, 1)
    again = corrupt_closing(h, r, t, keys, 10, 6, np.random.default_rng(0))
    assert (out == again).all()


def test_nssa_loss_matches_formula():
    pos = torch.tensor([1.0, 4.0])
    neg = torch.tensor([[2.0, 8.0, 5.0], [3.0, 3.0, 9.0]])
    gamma, temp = 6.0, 0.5
    w = torch.softmax(temp * (gamma - neg), dim=-1)
    expected = 0.5 * (-torch.nn.functional.logsigmoid(gamma - pos).mean()
                      - (w * torch.nn.functional.logsigmoid(neg - gamma)).sum(-1).mean())
    assert torch.isclose(nssa_loss(pos, neg, gamma=gamma, temperature=temp), expected)


# ---------------------------------------------------------------- path_transport (exercise)

def test_path_transport_matches_step_by_step_product():
    m = small_model()
    T = m.table()
    pad = 2 * R_
    steps = torch.tensor([[0, pad, pad], [3, R_ + 1, pad], [2, 4, R_ + 2], [R_ + 4, 1, 0]])
    assert torch.allclose(path_transport(T, steps), loop_transport(T, steps), atol=1e-5)


def test_path_transport_order_matters():
    T = small_model().table()
    ab, ba = torch.tensor([[0, 1, 2 * R_]]), torch.tensor([[1, 0, 2 * R_]])
    assert torch.allclose(path_transport(T, ab)[0], T[1] @ T[0], atol=1e-5)
    assert not torch.allclose(path_transport(T, ab), path_transport(T, ba), atol=1e-3)


def test_there_and_back_is_identity():
    T = small_model().table()
    steps = torch.tensor([[2, 2 + R_, 2 * R_], [R_ + 3, 3, 2 * R_]])     # along q then back along q
    assert torch.allclose(path_transport(T, steps), torch.eye(B_).expand(2, NB_, B_, B_), atol=1e-5)


def test_path_transport_has_gradients():
    m = small_model()
    T = m.table()
    out = path_transport(T, torch.tensor([[0, 1, 2]]))
    cycle_defect(T, out, torch.tensor([3])).sum().backward()
    assert m.U.grad is not None and m.U.grad.abs().sum() > 0


def test_triple_scores_average_paths_and_flag_coverage():
    m = small_model()
    T = m.table()
    pad = 2 * R_
    steps = np.full((3, 4, 3), pad)
    steps[0, 0] = [0, 1, pad]
    steps[0, 1] = [2, pad, pad]
    steps[2, 0] = [R_ + 1, pad, pad]
    rels = np.array([3, 0, 4])
    scores, covered = triple_holonomy_scores(m, rels, steps)
    assert covered.tolist() == [True, False, True] and scores[1] == 0
    d0 = cycle_defect(T, loop_transport(T, torch.tensor(steps[0, :2])), torch.tensor([3, 3]))
    d2 = cycle_defect(T, loop_transport(T, torch.tensor(steps[2, :1])), torch.tensor([4]))
    assert np.isclose(scores[0], d0.mean().item(), atol=1e-5) and np.isclose(scores[2], d2.item(), atol=1e-5)


# ---------------------------------------------------------------- cycle_loss (exercise)

def test_cycle_loss_value():
    loss = cycle_loss(torch.tensor([0.0, 1.0]), torch.tensor([0.5, 3.0]), alpha1=1.0, alpha2=2.0, margin=2.0)
    assert torch.isclose(loss, torch.tensor(0.5 + 2.0 * (1.5 ** 2) / 2))


def test_cycle_loss_empty_inputs_give_zero_not_nan():
    empty = torch.empty(0)
    assert cycle_loss(empty, empty, alpha1=1.0, alpha2=1.0, margin=1.0) == 0
    one = cycle_loss(torch.tensor([2.0]), empty, alpha1=1.0, alpha2=1.0, margin=1.0)
    assert torch.isclose(one, torch.tensor(4.0))


def test_cycle_loss_gradient_directions():
    d_true = torch.tensor([0.5], requires_grad=True)
    d_bad = torch.tensor([0.5, 5.0], requires_grad=True)
    cycle_loss(d_true, d_bad, alpha1=1.0, alpha2=1.0, margin=2.0).backward()
    assert d_true.grad[0] > 0                     # true cycles: pushed toward 0
    assert d_bad.grad[0] < 0                      # corrupted below the margin: pushed up
    assert d_bad.grad[1] == 0                     # corrupted beyond the margin: left alone
