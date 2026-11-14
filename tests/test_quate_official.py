"""The official-code QuatE (rtvm.baselines.build_quate_official) against formulas written out from the
official code (github.com/cheungdaven/QuatE, models/QuatE.py; notes/quate_check.md)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

E, R, D = 40, 5, 6


@pytest.fixture
def tf():
    from pykeen.triples import CoreTriplesFactory
    rng = np.random.default_rng(0)
    tr = np.unique(np.c_[rng.integers(E, size=200), rng.integers(R, size=200), rng.integers(E, size=200)], axis=0)
    return CoreTriplesFactory.create(torch.as_tensor(tr), num_entities=E, num_relations=R)


def official_score(h, r, t):
    """models/QuatE.py _calc, component by component."""
    s_a, x_a, y_a, z_a = h.unbind(-1)
    s_b, x_b, y_b, z_b = r.unbind(-1)
    s_c, x_c, y_c, z_c = t.unbind(-1)
    den = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
    s_b, x_b, y_b, z_b = s_b / den, x_b / den, y_b / den, z_b / den
    A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
    B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
    C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
    Dq = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
    return -torch.sum(A * s_c + B * x_c + C * y_c + Dq * z_c, -1)


def raw(rep, idx):
    return rep._embeddings.weight[idx].view(len(idx), -1, 4)


def test_score_matches_official_formula():
    from rtvm.baselines import NormalizedQuatEInteraction
    g = torch.Generator().manual_seed(0)
    h, r, t = (torch.randn(7, D, 4, generator=g, dtype=torch.float64) for _ in range(3))
    r = r * 3.0                                                   # raw, not unit length
    ours = NormalizedQuatEInteraction().double()(h, r, t)
    assert torch.allclose(ours, official_score(h, r, t))


def test_regulariser_matches_official_loss_term(tf):
    from rtvm.baselines import build_quate_official
    lam = 0.3
    m = build_quate_official(tf, D, lam, seed=0)
    m.train()
    pos = tf.mapped_triples[:16]
    neg = pos.repeat(3, 1)
    neg[:, 2] = torch.randint(E, (len(neg),), generator=torch.Generator().manual_seed(1))
    m.score_hrt(pos)                                              # as PyKEEN's sLCWA does: two calls
    m.score_hrt(neg)
    ours = m.collect_regularization_term()
    rows = torch.cat([pos, neg])                                  # official: one batch of all rows
    ent, rel = m.entity_representations[0], m.relation_representations[0]
    sq = lambda x: x.pow(2).sum(-1).mean()                         # = sum over s,x,y,z of mean(c^2)
    official = lam * (sq(raw(ent, rows[:, 0])) + sq(raw(ent, rows[:, 2]))) + lam * sq(raw(rel, rows[:, 1]))
    assert torch.allclose(ours.reshape(()), official, rtol=1e-6)
    ours.backward()
    assert rel._embeddings.weight.grad.abs().sum() > 0            # unlike PyKEEN's QuatE, relations are regularised


def test_relation_init_uses_entity_fan_in(tf):
    from rtvm.baselines import build_quate_official
    m = build_quate_official(tf, D, 0.1, seed=0)
    s = 1 / math.sqrt(2 * E)                                      # official quaternion_init(entTotal, ...)
    for rep in (m.entity_representations[0], m.relation_representations[0]):
        modulus = raw(rep, torch.arange(rep.max_id)).norm(dim=-1)
        assert modulus.max() <= s + 1e-6 and modulus.max() > s / 2


def test_no_regularisation_in_eval(tf):
    from rtvm.baselines import build_quate_official
    m = build_quate_official(tf, D, 0.3, seed=0)
    m.eval()
    with torch.no_grad():
        m.score_hrt(tf.mapped_triples[:8])
    assert float(m.collect_regularization_term()) == 0.0
