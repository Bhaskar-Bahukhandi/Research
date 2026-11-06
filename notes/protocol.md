# Experiment protocol (pre-registration)

**Commit this file on Fri Nov 20, 2026, before any result exists. After that, thresholds change only with a dated note explaining why, and the original value stays visible.**

Committed on: ____ (git hash: ____)

## General rules

- Every reported number is the mean ± std over **3 seeds** (0, 1, 2) unless the row says otherwise.
- Hyperparameters are chosen on **validation** splits only. Test splits are run once per final configuration.
- Every comparison holds **parameter count and hardware** fixed (proposal Section 6).
- Indexing time and query time are reported separately.
- A failed gate is reported, not re-run with a moved threshold.

## Gates

| Gate | Date | Measured by | Pass if | Status |
| --- | --- | --- | --- | --- |
| G1 | 2026-12-05 | `pytest -m "week3 or week4"` and `experiments/phase1_sanity.py --full` | All week 3/4 tests pass; max holonomy score on consistent graphs < 1e-4; every cycle through a corrupted edge > 1e-2; log-log time slope < 1.3; single corruption localised in ≥ 16/20 graphs | |
| G2 (H2) | 2027-01-02 | Phase 2 scripts (to write) | Holonomy AUC ≥ RotatE AUC on FB15k-237 **or** combined score beats RotatE, 95% bootstrap CI excluding 0. *Fill in: which split, how many cycles per triple, path-length cap.* | |
| G3 (H1) | 2027-01-30 | Phase 3 scripts | Page recall@k ≥ FAISS recall@k (k = ____); MetaQA Hits@1 within ____ points of baseline B | |
| G4 (H3–H4) | 2027-02-27 | Phase 4 scripts | Unsupported-claim rate below max-softmax abstention at coverage ____%; parser edge F1 ≥ ____ | |
| G5 (H5) | 2027-03-20 | Phase 5 scripts | Verified pass rate above generate-then-verify at equal compute, by ≥ ____ points | |

The blanks are yours to decide **before** the relevant experiment. Choose them from what would convince a skeptical reviewer, not from what you expect to get.

## Datasets and splits

| Dataset | Source | Split used | Notes |
| --- | --- | --- | --- |
| FB15k-237 | PyKEEN loader | standard train/valid/test | |
| WN18RR | PyKEEN loader | standard train/valid/test | |
| MetaQA | ____ | 1/2/3-hop, standard | |
| WebNLG | ____ | ____ | |
