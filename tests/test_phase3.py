"""Tests for the Phase 3 bake-off (phase3/, experiments/bakeoff.py, experiments/metaqa_build_data.py).

Protocol: notes/protocol.md, seventh note of 2026-11-22.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    sys.path.insert(0, str(ROOT / "experiments"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_kb(n_ent=40, n_rel=3, n=300, seed=1):
    from phase3.metaqa import parse_kb
    rng = np.random.default_rng(seed)
    lines = [f"e{h}|r{r}|e{t}" for h, r, t in zip(rng.integers(n_ent, size=n), rng.integers(n_rel, size=n),
                                                   rng.integers(n_ent, size=n))]
    return parse_kb("\n".join(lines + lines[:5]) + "\n")       # 5 duplicates


# ---------------------------------------------------------------- knowledge base and split

def test_parse_kb_dedupes_and_sorts():
    from phase3.metaqa import parse_kb
    kb = parse_kb("b|x|a\na|y|b\nb|x|a\n\n")
    assert kb.entities == ["a", "b"] and kb.relations == ["x", "y"] and kb.n_lines == 3
    assert kb.triples.tolist() == [[0, 1, 1], [1, 0, 0]]
    with pytest.raises(ValueError):
        parse_kb("a|b\n")
    with pytest.raises(ValueError):
        parse_kb("a|b|c|d\n")


def test_split_is_a_partition_with_held_out_entities_in_training():
    from phase3.metaqa import split_kb
    kb = _toy_kb()
    s = split_kb(kb.triples, fraction=0.1)
    parts = [s["train"], s["v_tune"], s["v_select"]]
    allidx = np.sort(np.concatenate(parts))
    assert np.array_equal(allidx, np.arange(len(kb.triples)))                  # disjoint and complete
    seen = set(kb.triples[s["train"]][:, [0, 2]].ravel())
    for name in ("v_tune", "v_select"):
        assert set(kb.triples[s[name]][:, [0, 2]].ravel()) <= seen
        assert len(s[name]) + s["moved_back"][name] == int(round(0.1 * len(kb.triples)))
    again = split_kb(kb.triples, fraction=0.1)
    assert all(np.array_equal(s[k], again[k]) for k in ("train", "v_tune", "v_select"))


def test_split_moves_back_a_triple_with_an_unseen_entity():
    from phase3.metaqa import parse_kb, split_kb
    lines = [f"a{i}|r|b{i % 3}" for i in range(40)] + ["lonely|r|b0"]
    kb = parse_kb("\n".join(lines))
    s = split_kb(kb.triples, fraction=0.25)
    lonely = kb.entities.index("lonely")
    assert lonely in kb.triples[s["train"]][:, 0]                                # always kept in training


def test_build_check_passes_on_the_committed_record():
    saved = ROOT / "saved/metaqa/manifest.json"
    kb = ROOT / "data/metaqa/kb.txt"
    if not saved.exists() or not kb.exists():
        pytest.skip("needs data/metaqa/kb.txt and the committed saved/metaqa record")
    build = _module("metaqa_build_data")
    manifest, _, split = build.build(kb)
    assert build.matches_record(ROOT / "saved/metaqa", manifest, split)      # arrays, not zip bytes


def test_record_check_catches_a_changed_split(tmp_path):
    import json
    import shutil
    if not (ROOT / "saved/metaqa/manifest.json").exists() or not (ROOT / "data/metaqa/kb.txt").exists():
        pytest.skip("needs data/metaqa/kb.txt and the committed saved/metaqa record")
    build = _module("metaqa_build_data")
    manifest, _, split = build.build(ROOT / "data/metaqa/kb.txt")
    shutil.copy(ROOT / "saved/metaqa/manifest.json", tmp_path / "manifest.json")
    bad = {k: v.copy() for k, v in split.items() if k in ("train", "v_tune", "v_select")}
    bad["v_tune"][0], bad["train"][0] = bad["train"][0], bad["v_tune"][0]          # one triple swapped
    np.savez(tmp_path / "split.npz", **bad)
    assert not build.matches_record(tmp_path, manifest, split)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["split"]["file"] == "split.npz"


# ---------------------------------------------------------------- link prediction

def test_ranking_copies_match_the_s8_code():
    from phase3 import linkpred
    sec = _module("g2_secondary")
    rng = np.random.default_rng(0)
    E, R = 30, 4
    known = np.unique(np.stack([rng.integers(E, size=200), rng.integers(R, size=200), rng.integers(E, size=200)], 1), axis=0)
    test = known[rng.choice(len(known), 40, replace=False)]
    table = rng.normal(size=(E, R, E)).round(1)                                 # rounding makes ties

    def tails(h, r):
        return torch.as_tensor(table[h, r, :])

    def heads(r, t):
        return torch.as_tensor(table[:, r, t].T)
    a = linkpred.link_prediction(tails, heads, test, known, E, R, batch=7)
    b = sec.link_prediction(tails, heads, test, known, E, R, batch=7)
    assert {k: a[k] for k in b} == b


def _pykeen_model(candidate, E=12, R=3):
    from phase3 import bakeoff
    values = bakeoff.trials(candidate, ROOT)[0]
    return bakeoff.build(candidate, values, E, R), values


@pytest.mark.parametrize("candidate", ["rotate", "quate"])
def test_pykeen_scorers_match_score_hrt(candidate):
    from phase3 import linkpred
    torch.manual_seed(0)
    net, _ = _pykeen_model(candidate)
    tails, heads = (linkpred.pykeen_scorers if candidate == "rotate" else linkpred.quate_scorers)(net)
    h, r, t = np.array([0, 5, 11]), np.array([0, 2, 1]), np.array([3, 3, 7])
    E = 12
    with torch.no_grad():
        full_t = net.score_hrt(torch.as_tensor(np.stack([np.repeat(h, E), np.repeat(r, E), np.tile(np.arange(E), 3)], 1))).view(3, E)
        full_h = net.score_hrt(torch.as_tensor(np.stack([np.tile(np.arange(E), 3), np.repeat(r, E), np.repeat(t, E)], 1))).view(3, E)
    assert torch.allclose(tails(h, r).cpu(), full_t.float(), atol=1e-4)
    assert torch.allclose(heads(r, t).cpu(), full_h.float(), atol=1e-4)


def test_cayley_scorers_match_single_distance():
    from phase3 import linkpred
    from rtvm.holonomy_model import HolonomyModel
    m = HolonomyModel(12, 3, rank=4, rotation_init=0.3, generator=torch.Generator().manual_seed(0))
    tails, heads = linkpred.cayley_scorers(m)
    h, r, t = np.array([0, 5, 11]), np.array([0, 2, 1]), np.array([3, 3, 7])
    E = torch.arange(12)
    with torch.no_grad():
        dt = torch.stack([m.single_distance(torch.full((12,), int(a)), torch.full((12,), int(b)), E) for a, b in zip(h, r)])
        dh = torch.stack([m.single_distance(E, torch.full((12,), int(b)), torch.full((12,), int(c))) for b, c in zip(r, t)])
    assert torch.allclose(tails(h, r), -dt, atol=1e-5)
    assert torch.allclose(heads(r, t), -dh, atol=1e-5)


# ---------------------------------------------------------------- candidates, trials, rule

def test_specs_follow_the_seventh_note():
    from phase3 import bakeoff
    from rtvm.g2search import CONFIGS, load_spec
    rf, rs = bakeoff.spec("rotate", ROOT)
    assert rf["dim"] == 64 and rf["model"] == "rotate_official" and rs == load_spec(ROOT / CONFIGS["rotate"])["search"]
    qf, qs = bakeoff.spec("quate", ROOT)
    assert qf["dim"] == 32 and qf["num_negatives"] == 10 and qf["optimizer"] == "adagrad"
    cf, cs = bakeoff.spec("cayley", ROOT)
    assert cs["rank"] == {"choice": [1, 2, 4, 8]}
    assert not set(bakeoff.B_DROPPED) & (set(cf) | set(cs))
    for c in bakeoff.CANDIDATES:
        f, _ = bakeoff.spec(c, ROOT)
        assert (f["eval_every"], f["patience"], f["max_epochs"], f["batch_size"]) == (10, 5, 1000, 1024)
        tr = bakeoff.trials(c, ROOT)
        assert len(tr) == 20 and tr == bakeoff.trials(c, ROOT)


def test_entities_have_128_reals_in_every_candidate():
    from phase3 import bakeoff
    for c in bakeoff.CANDIDATES:
        net, values = _pykeen_model(c, E=7, R=2)
        if c == "cayley":
            assert net.psi.shape == (7, 128)
        else:
            w = net.entity_representations[0]._embeddings.weight
            reals = w.shape[1] * (2 if w.is_complex() else 1)
            assert reals == 128, (c, w.shape, w.dtype)


def test_rule():
    rule = _module("bakeoff").apply_rule
    base = [0.30, 0.31, 0.32]                                                   # mean 0.31
    assert rule({"rotate": base, "cayley": [0.33, 0.32, 0.31], "quate": [0.30] * 3})[2] == "rotate"   # +0.01, a seed not above
    assert rule({"rotate": base, "cayley": [0.33, 0.32, 0.315], "quate": [0.30] * 3})[2] == "cayley"
    assert rule({"rotate": base, "cayley": [0.33] * 3, "quate": [0.34] * 3})[2] == "quate"
    assert rule({"rotate": base, "cayley": [0.319] * 3, "quate": [0.30] * 3})[2] == "rotate"         # under the margin


# ---------------------------------------------------------------- training

def _toy_data(tmp_path):
    from phase3.metaqa import BakeoffData, split_kb
    kb = _toy_kb()
    s = split_kb(kb.triples, fraction=0.1)
    return BakeoffData(kb=kb, train=kb.triples[s["train"]], v_tune=kb.triples[s["v_tune"]],
                       v_select=kb.triples[s["v_select"]], provenance={})


def _small(values, **over):
    return {**values, "eval_every": 1, "max_epochs": 2, "batch_size": 64, "num_negatives": 8, **over}


@pytest.mark.parametrize("candidate", ["rotate", "cayley", "quate"])
def test_each_candidate_trains_and_records(candidate, tmp_path):
    from phase3 import bakeoff
    data = _toy_data(tmp_path)
    values = _small(bakeoff.trials(candidate, ROOT)[0])
    res = bakeoff.train(candidate, values, 0, data, "cpu", tmp_path / "x.ckpt", log=lambda m: None)
    assert res["complete"] and res["epochs_run"] == 2 and len(res["history"]) == 2
    assert np.isfinite(res["best_v_tune_mrr"]) and 0 < res["best_v_tune_mrr"] <= 1


def test_cayley_resume_equals_uninterrupted(tmp_path):
    from phase3 import bakeoff
    data = _toy_data(tmp_path)
    values = _small(bakeoff.trials("cayley", ROOT)[0], max_epochs=4)
    full = bakeoff.train("cayley", values, 3, data, "cpu", tmp_path / "a.ckpt", log=lambda m: None)
    part = bakeoff.train("cayley", values, 3, data, "cpu", tmp_path / "b.ckpt", log=lambda m: None, deadline=0)
    assert not part["complete"] and part["epochs_run"] == 1
    rest = bakeoff.train("cayley", values, 3, data, "cpu", tmp_path / "b.ckpt", log=lambda m: None)
    assert rest["history"] == full["history"]
    for k, v in full["model"].state_dict().items():
        assert torch.equal(v, rest["model"].state_dict()[k])


def test_checkpoint_of_another_run_is_refused(tmp_path):
    from phase3 import bakeoff
    data = _toy_data(tmp_path)
    values = _small(bakeoff.trials("cayley", ROOT)[0], max_epochs=1)
    bakeoff.train("cayley", values, 0, data, "cpu", tmp_path / "a.ckpt", log=lambda m: None)
    with pytest.raises(ValueError):
        bakeoff.train("cayley", values, 1, data, "cpu", tmp_path / "a.ckpt", log=lambda m: None)
