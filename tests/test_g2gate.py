"""Spec for rtvm/g2gate.py and the guard in experiments/g2_gate.py (protocol, "Primary test")."""
from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from rtvm.g2eval import grouped_auc, grouped_auc_rows
from rtvm.g2gate import (B0, B1, STRUCTURE, decide, fit_combiner, null_rotations, primary_test,
                         relation_bootstrap, rotation_table, s11_report, structure_features, triple_holonomy,
                         two_way_bootstrap)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- grouped AUC rows

def test_grouped_auc_rows_mean_equals_grouped_auc():
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 4, 50).astype(float), rng.integers(0, 4, (50, 10)).astype(float)
    rows = grouped_auc_rows(pos, neg)
    assert rows.shape == (50,) and rows.mean() == pytest.approx(grouped_auc(pos, neg))
    assert grouped_auc_rows(np.array([1.0]), np.array([[0.0, 1.0, 2.0, 1.0]]))[0] == pytest.approx((1 + 0.5 + 0 + 0.5) / 4)
    with pytest.raises(ValueError):
        grouped_auc_rows(np.array([np.nan]), np.zeros((1, 3)))


# ---------------------------------------------------------------- structure features

def test_structure_features_by_hand():
    triples = np.array([[0, 0, 1], [2, 1, 3], [4, 0, 4]])
    counts = np.array([[1, 2, 40], [0, 0, 0], [0, 0, 0]])
    codes = np.zeros((3, 4, 3), np.int32)
    codes[0, 0, :1] = 5                        # one length-1 path
    codes[0, 1, :2] = [1, -2]                  # two length-2 paths
    codes[0, 2, :2] = [3, 4]
    codes[0, 3, :3] = [1, 2, 3]                # one sampled length-3 path (of 40)
    degree = np.array([3, 0, 7, 1, 2])
    f = structure_features(triples, counts, codes, degree)
    assert list(f["covered"]) == [1, 0, 0]
    assert f["log1p_n3"][0] == pytest.approx(np.log(41))
    assert (f["share_len1"][0], f["share_len2"][0], f["share_len3"][0]) == (0.25, 0.5, 0.25)
    assert f["share_len1"][1] == f["share_len2"][1] == f["share_len3"][1] == 0     # uncovered
    assert f["log1p_deg_h"][1] == pytest.approx(np.log(8)) and f["log1p_deg_t"][0] == 0
    assert set(f) | {"null_holonomy"} == set(STRUCTURE)


def test_structure_features_refuse_count_path_mismatch():
    with pytest.raises(ValueError):
        structure_features(np.array([[0, 0, 1]]), np.array([[0, 0, 0]]), np.ones((1, 2, 3), np.int32), np.ones(2))


# ---------------------------------------------------------------- holonomy scores

def test_null_rotations_are_fixed_haar_rotations():
    R = null_rotations(3)
    assert R.shape == (3, 2, 64, 64) and R.dtype == torch.float32
    eye = torch.eye(64)
    assert torch.allclose(R @ R.transpose(-1, -2), eye.expand_as(R), atol=1e-5)
    assert torch.allclose(torch.linalg.det(R.double()), torch.ones(3, 2, dtype=torch.float64), atol=1e-6)
    assert torch.equal(R, null_rotations(3))                     # seed 0, the same for every run
    assert torch.equal(R[:2], null_rotations(2))                 # relation r is the r-th draw


def test_rotation_table_matches_the_model():
    from rtvm.holonomy_model import HolonomyModel
    m = HolonomyModel(5, 3, n_blocks=2, b=4, rank=1, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.equal(rotation_table(m.rotations()), m.table())


def test_triple_holonomy_equals_the_training_code():
    """Deduplicated patterns give the model code's mean over every cycle (holonomy_model)."""
    from rtvm.holonomy_model import HolonomyModel, triple_holonomy_scores
    E, R, K = 6, 3, 5
    m = HolonomyModel(E, R, n_blocks=2, b=4, rank=2, rotation_init=0.3, generator=torch.Generator().manual_seed(1))
    rng = np.random.default_rng(0)
    pad = 2 * R
    lengths = rng.integers(0, 4, size=(40, K))
    lengths[:5] = 0                                              # uncovered triples
    steps = np.full((40, K, 3), pad)
    for i in range(40):
        for j in range(K):
            steps[i, j, :lengths[i, j]] = rng.integers(0, 2 * R, lengths[i, j])
    rel = rng.integers(0, R, 40)
    steps[6:12], rel[6:12] = steps[5], rel[5]                    # repeated patterns across triples
    with torch.no_grad():
        ours, cov = triple_holonomy(m.table(), rel, steps)
    ref, ref_cov = triple_holonomy_scores(m, rel, steps)
    assert np.array_equal(cov, ref_cov) and not cov[:5].any()
    assert np.allclose(ours, ref, rtol=1e-6, atol=1e-6) and (ours[:5] == 0).all()


# ---------------------------------------------------------------- combiner

def test_combiner_standardises_on_its_data_and_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(5, 3, 400), rng.normal(size=400), np.ones(400)]       # last column constant
    y = (X[:, 0] + rng.normal(size=400) > 5).astype(int)
    c = fit_combiner(X, y, ("a", "b", "c"))
    assert np.allclose(c.mean, X.mean(0)) and np.allclose(c.scale[:2], X.std(0)[:2]) and c.scale[2] == 1
    ref = LogisticRegression(C=1.0, tol=1e-6, max_iter=10_000).fit((X - c.mean) / c.scale, y)
    assert np.allclose(c.coef, ref.coef_.ravel(), atol=1e-5) and c.coef[0] > 0
    assert np.allclose(c.decision(X), ref.decision_function((X - c.mean) / c.scale), atol=1e-5)
    with pytest.raises(ValueError):
        fit_combiner(np.c_[X[:, :2], np.full(400, np.nan)], y, ("a", "b", "c"))


# ---------------------------------------------------------------- bootstrap and decision

def slow_bootstrap(diffs, relations, n_boot, seed):
    """Reference: resample relations explicitly and average the drawn positives."""
    clusters = np.unique(relations)
    G = len(clusters)
    draws = np.random.default_rng(seed).integers(0, G, size=(n_boot, G))
    out = []
    for row in draws:
        rows = np.concatenate([np.flatnonzero(relations == clusters[g]) for g in row])
        out.append(np.mean([d[rows].mean() for d in diffs]))
    return np.array(out)


def test_bootstrap_matches_explicit_resampling():
    rng = np.random.default_rng(0)
    relations = rng.choice([3, 7, 8, 20, 21], size=60)
    diffs = rng.normal(0.01, 0.05, size=(3, 60))
    fast = relation_bootstrap(diffs, relations, n_boot=300, seed=0)
    assert np.allclose(fast, slow_bootstrap(diffs, relations, 300, 0))
    assert np.array_equal(fast, relation_bootstrap(diffs, relations, n_boot=300, seed=0))


def test_bootstrap_of_a_constant_gain_is_that_gain():
    boot = relation_bootstrap(np.full((3, 30), 0.02), np.arange(30) % 4, n_boot=200)
    assert np.allclose(boot, 0.02)


# ---------------------------------------------------------------- S11: two-way bootstrap (note 2026-11-17)

def slow_two_way(diffs, relations, heads, tails, n_boot, seed=0, entity_seed=1):
    """Reference: build each resample as an explicit list, a positive repeated w_r * v_h * v_t times."""
    rels, ents = np.unique(relations), np.unique(np.r_[heads, tails])
    rel_draws = np.random.default_rng(seed).integers(0, len(rels), size=(n_boot, len(rels)))
    ent_draws = np.random.default_rng(entity_seed).integers(0, len(ents), size=(n_boot, len(ents)))
    out = []
    for rd, ed in zip(rel_draws, ent_draws):
        w = Counter(rels[rd].tolist())
        v = Counter(ents[ed].tolist())
        rows = [i for i in range(len(relations))
                for _ in range(w[relations[i]] * v[heads[i]] * v[tails[i]])]
        out.append(np.mean([d[rows].mean() for d in diffs]))
    return np.array(out)


def two_way_case(rng, n=50):
    relations = rng.choice([2, 5, 9, 11], size=n)
    heads, tails = rng.integers(0, 6, n), rng.integers(0, 6, n)
    tails[:5] = heads[:5]                                       # h = t rows: weight w_r * v_h ** 2
    return rng.normal(0.01, 0.05, size=(3, n)), relations, heads, tails


def test_two_way_bootstrap_matches_explicit_resampling():
    diffs, relations, heads, tails = two_way_case(np.random.default_rng(0))
    fast = two_way_bootstrap(diffs, relations, heads, tails, n_boot=200, chunk=64)
    assert np.allclose(fast, slow_two_way(diffs, relations, heads, tails, 200))


def test_two_way_bootstrap_is_chunk_invariant():
    diffs, relations, heads, tails = two_way_case(np.random.default_rng(1))
    a = two_way_bootstrap(diffs, relations, heads, tails, n_boot=300, chunk=7)
    b = two_way_bootstrap(diffs, relations, heads, tails, n_boot=300, chunk=300)
    assert np.allclose(a, b, rtol=1e-12, atol=0)


def test_chunked_entity_draws_equal_the_single_call_in_the_note():
    """The note fixes default_rng(1).integers(0, E, size=(10_000, E)); the code draws it in row chunks."""
    for E in (3, 7, 1001):
        whole = np.random.default_rng(1).integers(0, E, size=(40, E))
        g = np.random.default_rng(1)
        parts = np.concatenate([g.integers(0, E, size=(rows, E)) for rows in (1, 6, 7, 26)])
        assert np.array_equal(whole, parts)


def test_two_way_with_one_entity_is_the_relation_bootstrap():
    """With a single entity every v is 1, so S11 reduces to the primary bootstrap: same relation draws."""
    rng = np.random.default_rng(2)
    relations = rng.choice([3, 7, 8, 20, 21], size=60)
    diffs = rng.normal(0.01, 0.05, size=(3, 60))
    one = np.zeros(60, dtype=int)
    assert np.allclose(two_way_bootstrap(diffs, relations, one, one, n_boot=300),
                       relation_bootstrap(diffs, relations, n_boot=300))


def test_two_way_of_a_constant_gain_is_that_gain():
    rng = np.random.default_rng(3)
    boot = two_way_bootstrap(np.full((3, 40), 0.02), np.arange(40) % 4, rng.integers(0, 20, 40),
                             rng.integers(0, 20, 40), n_boot=200)
    assert np.allclose(boot, 0.02)


def test_two_way_zero_weight_resample_is_an_error():
    # one positive with h != t among two entities: about half the resamples miss h or t
    with pytest.raises(ValueError, match="total weight 0"):
        two_way_bootstrap(np.full((3, 1), 0.02), np.array([0]), np.array([0]), np.array([1]), n_boot=50)


def test_two_way_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        two_way_bootstrap(np.zeros((3, 4)), np.zeros(4), np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("decision, lo, expected", [
    ("pass", 0.006, "pass"),
    ("pass", 0.005, "pass"),
    ("pass", 0.004, "pass; not robust under the conservative two-way bootstrap (S11)"),   # note 2026-11-18
    ("negligible", -0.01, "negligible"),
    ("fail: no added information", -0.02, "fail: no added information"),
])
def test_s11_report(decision, lo, expected):
    assert s11_report(decision, lo) == expected


@pytest.mark.parametrize("deltas, lo, hi, expected", [
    ([0.01, 0.02, 0.01], 0.006, 0.03, "pass"),
    ([0.01, 0.02, 0.01], 0.005, 0.03, "pass"),                  # lower bound >= 0.005
    ([0.01, 0.02, 0.01], 0.002, 0.03, "negligible"),
    ([0.01, -0.01, 0.01], -0.001, 0.02, "fail: no added information"),
    ([0.0, 0.0, 0.0], 0.0, 0.0, "fail: no added information"),
    ([-0.01, -0.02, -0.01], -0.03, -0.001, "fail: holonomy hurts"),
    ([0.02, 0.03, -0.001], 0.006, 0.04, "not pass: not positive in every seed"),
])
def test_decision_rule(deltas, lo, hi, expected):
    assert decide(deltas, lo, hi) == expected


# ---------------------------------------------------------------- end to end on synthetic features

def synthetic_split(rng, n, k, informative_holonomy: bool):
    """Positives and negatives where RotatE carries signal and holonomy carries extra signal (or none)."""
    rel = rng.integers(0, 25, n)
    truth_pos, truth_neg = rng.normal(1.0, 1, n), rng.normal(0.0, 1, (n, k))
    split = dict(pos={}, neg={}, relations=rel)
    for name in STRUCTURE:
        split["pos"][name], split["neg"][name] = rng.normal(size=n), rng.normal(size=(n, k))
    for s in (0, 1, 2):
        split["pos"][f"rotate:{s}"] = truth_pos + rng.normal(size=n)
        split["neg"][f"rotate:{s}"] = truth_neg + rng.normal(size=(n, k))
        extra_p, extra_n = (rng.normal(1.0, 1, n), rng.normal(0.0, 1, (n, k))) if informative_holonomy else \
            (rng.normal(size=n), rng.normal(size=(n, k)))
        split["pos"][f"holonomy:{s}"], split["neg"][f"holonomy:{s}"] = extra_p, extra_n
    return split


@pytest.mark.parametrize("informative, expected", [(True, "pass"), (False, "fail: no added information")])
def test_primary_test_end_to_end(informative, expected):
    rng = np.random.default_rng(0)
    v2, test = synthetic_split(rng, 1500, 10, informative), synthetic_split(rng, 1500, 10, informative)
    res = primary_test(v2, test, n_boot=2000)
    assert res["decision"] == expected
    assert len(res["seeds"][0]["B0"]["combiner"]["coef"]) == len(B0)
    assert len(res["seeds"][0]["B1"]["combiner"]["coef"]) == len(B1)
    assert res["n_clusters"] == 25 and res["n_positives"] == 1500
    assert "s11" not in res


def test_primary_test_with_s11_keeps_the_decision():
    rng = np.random.default_rng(0)
    v2, test = synthetic_split(rng, 1500, 10, True), synthetic_split(rng, 1500, 10, True)
    test["triples"] = np.c_[rng.integers(0, 300, 1500), test["relations"], rng.integers(0, 300, 1500)]
    plain = primary_test(v2, test, n_boot=500)
    both = primary_test(v2, test, n_boot=500, two_way=True)
    assert {k: v for k, v in both.items() if k != "s11"} == plain
    lo, hi = both["s11"]["ci95"]
    assert lo <= both["delta_bar"] <= hi
    assert both["s11"]["report"] == s11_report(both["decision"], lo)
    assert both["s11"]["n_entities"] == len(np.unique(test["triples"][:, [0, 2]]))
    assert both["s11"]["narrower_than_primary"] == (hi - lo < both["ci95"][1] - both["ci95"][0])
    test["triples"] = test["triples"][:, [1, 0, 2]]             # relations in the wrong column
    with pytest.raises(ValueError, match="disagree"):
        primary_test(v2, test, n_boot=50, two_way=True)


def test_an_s11_error_leaves_the_primary_result_intact():
    """A zero-weight resample is an S11 error (note 2026-11-17); it must not stop gate.json being written."""
    rng = np.random.default_rng(0)
    v2, test = synthetic_split(rng, 1500, 10, True), synthetic_split(rng, 1500, 10, True)
    test["triples"] = np.c_[np.zeros(1500, int), test["relations"], np.ones(1500, int)]   # every h = 0, t = 1
    plain = primary_test(v2, test, n_boot=200)
    res = primary_test(v2, test, n_boot=200, two_way=True)
    assert {k: v for k, v in res.items() if k != "s11"} == plain
    assert "total weight 0" in res["s11"]["error"] and res["s11"]["report"] == res["decision"]
    assert "ci95" not in res["s11"]


def test_s11_runs_on_the_primary_dataset_only():
    assert _gate_script().S11_DATASETS == ("fb15k237",)


# ---------------------------------------------------------------- test-split guard

def _gate_script():
    spec = importlib.util.spec_from_file_location("g2_gate", ROOT / "experiments" / "g2_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_freeze_line_must_name_exactly_one_commit():
    gate = _gate_script()
    c = "a" * 40
    assert gate.freeze_commit(f"- note\n  G2 test freeze: commit {c}\n") == c
    assert gate.freeze_commit(f"- G2 test freeze: commit {c}\nlater: \n  - G2 test freeze: commit {c}") == c
    with pytest.raises(ValueError):
        gate.freeze_commit("no freeze yet")
    with pytest.raises(ValueError):
        gate.freeze_commit(f"G2 test freeze: commit {c}\nG2 test freeze: commit {'b' * 40}")
    with pytest.raises(ValueError):
        gate.freeze_commit("G2 test freeze: commit abc1234")              # short hashes are not accepted
    for quoted in (f"> G2 test freeze: commit {c}", f"`G2 test freeze: commit {c}`",
                   f"write a line like G2 test freeze: commit {c}"):
        with pytest.raises(ValueError):
            gate.freeze_commit(quoted)                                    # a quoted template does not count


@pytest.mark.parametrize("split", ["test", "Test", "v1", "valid"])
def test_test_split_is_refused_without_the_guard(tmp_path, split):
    gate = _gate_script()
    with pytest.raises((PermissionError, ValueError)):
        gate.load_split("fb15k237", split, tmp_path, tmp_path)


def _git(repo, *args):
    import subprocess
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def test_check_test_allowed_in_a_scratch_repo(tmp_path, monkeypatch):
    gate = _gate_script()
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    (tmp_path / "rtvm").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "rtvm" / "a.py").write_text("x = 1\n")
    (tmp_path / "notes" / "protocol.md").write_text("# protocol\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "code")
    with pytest.raises(SystemExit):
        gate.check_test_allowed()                                         # no freeze line yet
    frozen = gate.git("rev-parse", "HEAD")
    (tmp_path / "notes" / "protocol.md").write_text(f"# protocol\n- G2 test freeze: commit {frozen}\n")
    _git(tmp_path, "commit", "-qam", "freeze note")
    assert gate.check_test_allowed() == frozen
    state = gate.code_state()
    assert not state["dirty"] and state["head"] != frozen
    (tmp_path / "rtvm" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit):
        gate.check_test_allowed()                                         # dirty tree
    assert gate.code_state()["dirty"]
    _git(tmp_path, "commit", "-qam", "change frozen code")
    with pytest.raises(SystemExit):
        gate.check_test_allowed()                                         # frozen code changed since the freeze
    assert gate.code_state()["frozen_tree"] != state["frozen_tree"]


def test_provenance_check():
    gate = _gate_script()
    current = dict(head="h", dirty=False, frozen_tree="T")
    gate.check_provenance({"a.npz": dict(head="old", dirty=False, frozen_tree="T")}, current)   # same code, other commit
    for bad in (dict(dirty=True, frozen_tree="T"), dict(dirty=False, frozen_tree="U"), {}):
        with pytest.raises(SystemExit):
            gate.check_provenance({"a.npz": bad}, current)


def _fake_runs(tmp_path, config_hash, best=3, seed_values=None):
    run_dir = tmp_path / "fb15k237" / "rotate"
    run_dir.mkdir(parents=True)
    trials = [dict(lr=float(i)) for i in range(20)]
    (run_dir / "search.json").write_text(__import__("json").dumps(dict(config_sha256=config_hash, trials=trials)))

    def write(stem, seed, values, auc):
        rec = dict(complete=True, best_v1_auc=auc, seed=seed, values=values, config_sha256=config_hash)
        (run_dir / f"{stem}.json").write_text(__import__("json").dumps(rec))
        (run_dir / f"{stem}_best.pt").write_bytes(b"")
    for i in range(20):
        write(f"trial_{i:02d}", 0, trials[i], 0.9 if i == best else 0.5)
    for s in (1, 2):
        write(f"seed_{s}", s, seed_values or trials[best], 0.9)
    return run_dir


def test_model_runs_checks_config_and_seed_runs(tmp_path):
    from rtvm.g2search import CONFIGS, file_sha256
    gate = _gate_script()
    good = file_sha256(ROOT / CONFIGS["rotate"])
    stems = gate.model_runs(_fake_runs(tmp_path / "ok", good).parents[1], "fb15k237", "rotate")
    assert [p.name for p in stems.values()] == ["trial_03", "seed_1", "seed_2"]
    with pytest.raises(SystemExit):                                       # search made from another config
        gate.model_runs(_fake_runs(tmp_path / "cfg", "0" * 64).parents[1], "fb15k237", "rotate")
    with pytest.raises(SystemExit):                                       # seed runs of a different trial
        gate.model_runs(_fake_runs(tmp_path / "seed", good, seed_values=dict(lr=7.0)).parents[1], "fb15k237", "rotate")
