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
