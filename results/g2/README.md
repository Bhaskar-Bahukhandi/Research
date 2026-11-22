# G2 results

Run records from the G2 searches, committed because the runs cost GPU time and can't simply be regenerated. Checkpoints are not stored here.

## RotatE (complete 2026-11-15)

- **Config:** `configs/g2_rotate.yaml`, sha256 `d1e0f09b9869…`. The search started at commit `c5c44e2` (see `search.json`). Each run's own commit is recorded in its JSON.
- **Compute:** Kaggle, 2× T4, across five sessions. The last session, notebook `rtvm-g2-rotate-s5`, continued from the earlier ones' checkpoints and holds the final state.
- **Files:** for each dataset there are `search.json` (the 20 sampled trial settings), `trial_00.json`–`trial_19.json` and `seed_1.json`, `seed_2.json`. Seed 0 of the best trial is the trial itself.

| Dataset | Best trial | lr | γ | T | V₁ AUC by seed (0 / 1 / 2) | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| FB15k-237 | 17 | 3.37e-4 | 9 | 0.5 | 0.8801 / 0.8764 / 0.8815 | 0.8793 |
| WN18RR | 8 | 1.60e-3 | 3 | 1.0 | 0.9669 / 0.9681 / 0.9649 | 0.9666 |

The best trial is picked by V₁ AUC; ties go to the lower index (`rtvm.g2search.select_best`). On FB15k-237, trial 9 (0.8791) is within the seed spread of trial 17. Four WN18RR trials (1, 9, 10, 16, all with low learning rates) stopped at or near the 1,000-epoch cap.

## Holonomy model (complete 2026-11-21)

- **Config:** `configs/g2_holonomy.yaml`, sha256 `a4b7bb683a26…`. The search started at commit `32ecffc`; later runs used `9cac9a2`, which changes no holonomy training code (protocol, first note of 2026-11-21).
- **Compute:** Kaggle, 2× T4, across three sessions. The last, `rtvm-g2-holonomy-s3`, holds the final state.
- **Files:** as for RotatE.
- **V₁ is the stratified cycle AUC** of the triple holonomy score (protocol note of 2026-11-10). It is a different metric from the baselines' V₁ and is not compared with them.

| Dataset | Best trial | rank | lr | γ | T | α₁ | α₂ | margin | V₁ AUC by seed (0 / 1 / 2) | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FB15k-237 | 4 | 2 | 2.81e-3 | 12 | 1.0 | 1.16 | 0.147 | 0.97 | 0.5676 / 0.5672 / 0.5669 | 0.5672 |
| WN18RR | 14 | 4 | 1.05e-4 | 12 | 0.5 | 7.44 | 0.028 | 3.90 | 0.7372 / 0.7273 / 0.7267 | 0.7304 |

On FB15k-237, trial 14 (0.5674) is within the seed spread of trial 4. No holonomy trial reached the 1,000-epoch cap.

## QuatE (complete 2026-11-21)

- **Config:** `configs/g2_quate.yaml`, sha256 `cc97fddb1f1b…`. This is the official-code QuatE trained with Adagrad (protocol note of 2026-11-16). The search started at commit `d59679e`; session 2 ran `7210e1e`, which changes no training code.
- **Compute:** Kaggle, 2× T4, across two sessions. The last, `rtvm-g2-quate-s2`, holds the final state.
- **Use:** secondary only (S5, S6). QuatE is not part of the gate.

| Dataset | Best trial | lr | λ | V₁ AUC by seed (0 / 1 / 2) | Mean |
| --- | --- | --- | --- | --- | --- |
| FB15k-237 | 8 | 0.322 | 0.042 | 0.8737 / 0.8746 / 0.8728 | 0.8737 |
| WN18RR | 13 | 0.496 | 0.876 | 0.9395 / 0.9351 / 0.9414 | 0.9386 |

On WN18RR, the best learning rate lies at the top of the searched range [0.02, 0.5], and six trials (3, 8, 11, 14, 17, 18) reached the 1,000-epoch cap. S5 and S6 on WN18RR may therefore understate QuatE (protocol, first note of 2026-11-21).

## Gate (decided 2026-11-21)

`gate/fb15k237/gate.json` is the record written by `experiments/g2_gate.py decide`, run once at commit `725feba` with the test freeze at `e7ee642`. It holds the per-seed combiners, grouped AUCs, the bootstrap CI, S11, the code and data hashes, and the checkpoint hashes of the six models used.

The model paths in `sources` are made relative to the repository; the file is otherwise identical to the one written. The written file (sha256 `d4087fb8ae3b…`) stays in the git-ignored `runs/g2gate/`.

**Result: fail, no added information.**

| | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| B₀ grouped AUC (RotatE + structure) | 0.8851 | 0.8818 | 0.8842 |
| B₁ grouped AUC (+ holonomy) | 0.8851 | 0.8817 | 0.8842 |
| Δ | −0.00002 | −0.00008 | +0.00001 |

- **Δ̄** = −0.00003, with 95% relation-cluster CI −0.00013 to +0.00005.
- **Sample:** 224 relations and 20,438 test positives.
- **S11, two-way bootstrap over 10,317 entities:** −0.00023 to +0.00013.

## Secondary analyses (FB15k-237, run 2026-11-22)

`secondary/fb15k237/` holds the records written by `experiments/g2_secondary.py`, as fixed in the protocol note of 2026-11-22 (second). Each record names the commit that made it. Before computing anything new, each command reproduced the gate's per-seed grouped AUCs (and, for S3 and S6, Δ̄ and its CI) exactly. Model paths are repository-relative; the records are otherwise as written. None of these analyses has a pass rule, and none can change the gate.

| Analysis | Δ per seed | Δ̄ | 95% CI |
| --- | --- | --- | --- |
| Gate: holonomy-model holonomy added to B₀ | −0.00002 / −0.00008 / +0.00001 | −0.00003 | −0.00013 to +0.00005 |
| S2(a) covered positives and negatives only (20,310 positives) | −0.00002 / −0.00010 / +0.00000 | −0.00004 | −0.00016 to +0.00005 |
| S2(b) pooled AUC (1,000 resamples) | +0.00004 / +0.00003 / +0.00001 | +0.00003 | −0.00009 to +0.00016 |
| S9(a) fallback negatives removed (11,830) | +0.00003 / −0.00003 / +0.00002 | +0.00001 | −0.00007 to +0.00008 |
| S9(b) holonomy as decile indicators | −0.00112 / −0.00047 / −0.00148 | −0.00102 | −0.00168 to −0.00045 |
| S6 holonomy from RotatE's rotations | −0.00037 / −0.00041 / −0.00020 | −0.00032 | −0.00055 to −0.00011 |
| S6 holonomy from QuatE's rotations | −0.00052 / −0.00027 / −0.00046 | −0.00042 | −0.00086 to −0.00005 |
| **S3(ii) typed path rules added to B₀** | **+0.00989 / +0.01175 / +0.01043** | **+0.01069** | **+0.00672 to +0.01565** |
| S3(i) holonomy added to B₀ + rules | +0.00009 / −0.00009 / +0.00001 | +0.00000 | −0.00014 to +0.00012 |

- **S7:** of 224 test relations, Δ̄ is positive in 44, negative in 35 and exactly 0 in 145. It ranges from −0.033 to +0.0048, so no relation alone reaches 0.005. `basic.json` lists every relation, and names those with at least 100 test positives.
- **S9(b) decile coefficients:** against the lowest-defect decile, deciles 2–9 lower the log-odds by about 0.2 to 0.6 in every seed. The highest-defect decile is back near 0, and in seed 1 at +0.38. The fitted effect of holonomy is therefore not monotone.
- **Direction of the holonomy coefficient:** it is positive for the holonomy model (0.04–0.07) and for RotatE's rotations (0.15–0.34), so there a higher defect predicts "true". It is negative for QuatE's (−0.28 to −0.34). In none of these cases does the feature help on test.
- **S3's estimation set:**
  - 57,605 training triples with 576,050 negatives, filtered against training only; 42,674 of the negatives come from the fallback;
  - 158,225 rated (relation, pattern) pairs, out of 513,471 seen;
  - rated patterns cover 97.4% of test rows (positives and negatives pooled).

## Collapse check (exploratory, 2026-11-22)

`secondary/<dataset>/collapse.json` gives ‖U_r − I‖_F over relations, for the holonomy model's best trial and its two seed runs. No split is read.

| Dataset | Rank | Trained median (seeds 0 / 1 / 2) | Fresh draw at the initial scale | Largest possible | Trained ÷ fresh |
| --- | --- | --- | --- | --- | --- |
| FB15k-237 | 2 | 1.35 / 1.14 / 1.21 | 0.54 | 5.66 | 2.1–2.5 |
| WN18RR | 4 | 0.070 / 0.059 / 0.058 | 2.18 | 8.00 | 0.03 |

On WN18RR, training moved the rotations to about 3% of their initial distance from the identity in all three seeds. This is the "rotations collapse to the identity" risk listed in the proposal. On FB15k-237, the rotations moved away from the identity.

## Secondary analyses, second batch (run 2026-11-22)

The records follow the protocol note of 2026-11-22 (fourth). Each names the commit that made it:

| Record | Analysis | Commit |
| --- | --- | --- |
| `gate/wn18rr/gate.json` | S1 | `16a783b` |
| `secondary/wn18rr/s7_wn18rr.json` | S7 (WN18RR part) | `4f7e405` |
| `secondary/fb15k237/s5.json` | S5 | `4f7e405` |
| `secondary/fb15k237/s4.json` | S4 | `8605233` |
| `secondary/{fb15k237,wn18rr}/s8.json` | S8 | `2981339` |
| `secondary/fb15k237/s10.json` | S10 | `2981339` |

As in the first batch:
- S1 is the gate's own record for WN18RR, written once by `g2_gate.py decide --dataset wn18rr`.
- S4, S5, S7 and S10 each reproduced their dataset's gate (per-seed grouped AUCs, Δ̄ and CI) exactly before computing anything new. S8 scores single models and uses no combiner.
- Model paths are repository-relative, and the records are otherwise as written.
- None of these analyses has a pass rule, and none can change the gate.

### S1: WN18RR, the gate's procedure

**Decision label: fail, no added information.** It is not a gate.

| | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| B₀ grouped AUC (RotatE + structure) | 0.9610 | 0.9632 | 0.9600 |
| B₁ grouped AUC (+ holonomy) | 0.9611 | 0.9633 | 0.9602 |
| Δ | +0.00007 | +0.00014 | +0.00024 |

- **Δ̄** = +0.00015, with 95% relation-cluster CI −0.00035 to +0.00122.
- **Sample:** only 11 relations and 2,924 test positives, so the CI is about nine times wider than on FB15k-237.
- **Holonomy coefficient:** −0.31 to −0.38 in B₁. WN18RR's rotations collapsed toward the identity (collapse check below).

### S7, WN18RR part: test positives whose reverse is in training

| Group | Positives | Relations | Δ per seed | Δ̄ | 95% CI |
| --- | --- | --- | --- | --- | --- |
| Exact reverse (t, r, h) is a training triple | 1,086 | 4 | 0 / 0 / 0 | 0 | 0 to 0 |
| The rest | 1,838 | 10 | +0.00011 / +0.00022 / +0.00038 | +0.00024 | −0.00055 to +0.00200 |

Holonomy changes no ranking among the positives whose reverse is in training.

### S5: QuatE in RotatE's place

In both combiners, QuatE's `score_hrt` takes the column RotatE held. The column keeps the name `rotate` in the record.

| | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| B₀ᵠ grouped AUC (QuatE + structure) | 0.8734 | 0.8747 | 0.8734 |
| B₁ᵠ grouped AUC (+ holonomy) | 0.8724 | 0.8738 | 0.8722 |
| Δ | −0.00101 | −0.00086 | −0.00119 |

- **Δ̄** = −0.00102, with 95% CI −0.00201 to +0.00018.
- **Holonomy coefficient:** −0.88 to −1.03, against +0.04 to +0.07 next to RotatE.

### S4: path length

| Analysis | Test rows with such a cycle | Δ per seed | Δ̄ | 95% CI |
| --- | --- | --- | --- | --- |
| (a) cycles with a path of length 1 only | 1.2% | +0.00000 / −0.00000 / +0.00000 | −0.00000 | −0.00001 to +0.00000 |
| (a) length 2 only | 48.7% | −0.00000 / −0.00002 / −0.00007 | −0.00003 | −0.00012 to +0.00004 |
| (a) length 3 only | 95.8% | −0.00000 / −0.00003 / +0.00001 | −0.00001 | −0.00008 to +0.00005 |
| (b) whole test, paths of length ≤ 2 | 48.9% covered | +0.00001 / −0.00004 / +0.00001 | −0.00001 | −0.00004 to +0.00002 |

- "Test rows" pools positives and negatives.
- In (b), B₀ with the restricted structure has grouped AUC 0.8861 / 0.8828 / 0.8851. That is 0.001 above the gate's B₀ with L ≤ 3.

### S8: filtered link prediction (test split)

These are realistic ranks over both directions, filtered against train, validation and test. Values are the mean ± std over the 3 seeds (ddof 1).

| Model | Dimension | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- | --- |
| **FB15k-237** (20,438 test triples) | | | | |
| RotatE | 68 | 0.311 ± 0.001 | 0.216 ± 0.001 | 0.503 ± 0.002 |
| QuatE | 34 | 0.305 ± 0.001 | 0.215 ± 0.001 | 0.487 ± 0.002 |
| Holonomy model, single-triple score | 128 | 0.096 ± 0.042 | 0.027 ± 0.027 | 0.241 ± 0.082 |
| **WN18RR** (2,924 test triples) | | | | |
| RotatE | 64 | 0.472 ± 0.003 | 0.402 ± 0.006 | 0.603 ± 0.005 |
| QuatE | 32 | 0.363 ± 0.013 | 0.276 ± 0.018 | 0.532 ± 0.003 |
| Holonomy model, single-triple score | 128 | 0.0002 ± 0.0001 | 0 | 0.0003 ± 0.0005 |

**Published values**, from PyKEEN's bundled configs. Check them against the papers before citing, as `notes/baseline_reproduction.md` and `notes/quate_check.md` say.

| Dataset | RotatE (dimension 1000 / 500) | QuatE¹ |
| --- | --- | --- |
| FB15k-237 | 0.338 / 0.241 / 0.533 | 0.311 / 0.221 / 0.495 |
| WN18RR | 0.476 / 0.428 / 0.571 | 0.481 / 0.436 / 0.564 |

Two things limit the comparison:
- **Test sets:** the published figures include the test triples PyKEEN drops (28 on FB15k-237 and 210 on WN18RR). Counting those at reciprocal rank ≈ 0 multiplies our WN18RR figures by about 0.933.
- **Sizes:** our models are budget-matched (2.0M parameters), far smaller than the papers'.

The WN18RR QuatE search's best learning rate lies at the edge of its range (see QuatE above), so its numbers may understate QuatE.

**The holonomy model's single-triple score is not a link predictor.** On WN18RR, a check on V₂ put the true tail's median rank at about 24,000 of 40,559 entities. Early stopping selects the model by V₁ cycle AUC, not by link prediction.

### S10: dense rotations (rank 32, 3,798,144 parameters, outside the 2.0M budget)

The model has every other value of the best trial (trial 4); there was no new search. It was trained on Kaggle T4s (session `rtvm-g2-s10-s1`), and all three seeds stopped early:

| Seed | Best V₁ AUC | Best epoch | Epochs run |
| --- | --- | --- | --- |
| 0 | 0.5657 | 40 | 90 |
| 1 | 0.5662 | 60 | 110 |
| 2 | 0.5659 | 20 | 70 |

The gate's rank-2 runs reached about 0.567.

| | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| B₀ grouped AUC | 0.8851 | 0.8817 | 0.8842 |
| B₁ grouped AUC (+ dense holonomy) | 0.8851 | 0.8817 | 0.8842 |
| Δ | +0.00003 | −0.00001 | 0.00000 |

- **Δ̄** = +0.00001, with 95% CI −0.00004 to +0.00006, against the gate's −0.00003 (−0.00013 to +0.00005).
- **Holonomy coefficient:** −0.03 to −0.11.
- **Rotation distance from the identity:** median ‖U_r − I‖_F of 1.34 / 1.35 / 1.03, against a largest possible 22.6. That is the same range as the rank-2 runs (1.35 / 1.14 / 1.21). No fresh-draw reference was computed at rank 32.
