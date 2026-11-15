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
