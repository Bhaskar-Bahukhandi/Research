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
