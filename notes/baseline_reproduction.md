# RotatE baseline reproduction (2026-11-07)

**Purpose:** implementation check only, allowed by the protocol note of 2026-11-06. These numbers are not G2 features and must not set any G2 configuration, search range or threshold. The models use the published sizes (dimension 1000 / 500), far above the 2.0M G2 budget.

- **Script:** `experiments/reproduce_rotate.py`, seed 0, one run per cell.
- **Hardware:** Colab A100-SXM4-40GB, strict FP32, torch 2.14.1, PyKEEN 1.11.1.
- **Raw results:** `experiments/out/rotate_*_seed0.json` (gitignored; copies on Google Drive in `rtvm_rotate/out/`).
- **Settings:** official `best_config.sh` of DeepGraphLearning/KnowledgeGraphEmbedding, trained for the official number of steps (100k on FB15k-237, 80k on WN18RR).
- **Variants:**
  - *official:* PyKEEN training loop with the official model details (L1 distance over complex moduli, relations as phases, init ±(γ+2)/dim, filtered negatives, learning rate ÷10 at half the steps);
  - *stock:* PyKEEN 1.11.1's RotatE with its bundled `sun2019` config (L2 distance, relations as unit complex numbers, U(0, 1) entity init, no learning-rate decay).

## Filtered test results (realistic rank, both sides)

| Dataset | Variant | MRR | Hits@1 | Hits@10 | Train time |
| --- | --- | --- | --- | --- | --- |
| FB15k-237 | official | 0.329 | 0.229 | 0.531 | 3.31 h |
| FB15k-237 | stock | 0.311 | 0.223 | 0.487 | 1.88 h |
| FB15k-237 | published | 0.338 | 0.241 | 0.533 | |
| WN18RR | official | 0.507 | 0.458 | 0.607 | 3.15 h |
| WN18RR | stock | 0.217 | 0.184 | 0.281 | 1.63 h |
| WN18RR | published | 0.476 | 0.428 | 0.571 | |

"Published" values are those recorded in PyKEEN's bundled `sun2019` configs. **Check them against Table 5 of Sun et al. (2019), arXiv:1902.10197, before citing.** Validation MRR (official variant): FB15k-237 0.335, WN18RR 0.511.

## Reading

1. **The test sets differ.** PyKEEN drops evaluation triples whose entities never appear in training: WN18RR keeps 2,924 of 3,134 test triples (210 dropped), FB15k-237 keeps 20,438 of 20,466 (28 dropped; see `notes/dataset_stats.md`). The published figures include the dropped triples. Counting them back with reciprocal rank ≈ 0 (an assumption: their entities have untrained embeddings) multiplies our scores by 2,924 / 3,134 = 0.933 on WN18RR and 0.9986 on FB15k-237:

   | Official variant, on the full published test set (estimate) | MRR | Hits@1 | Hits@10 |
   | --- | --- | --- | --- |
   | FB15k-237 | 0.328 (published 0.338) | 0.229 (0.241) | 0.530 (0.533) |
   | WN18RR | 0.473 (0.476) | 0.427 (0.428) | 0.567 (0.571) |

2. **The official variant reproduces the paper closely.** On WN18RR it is within 0.005 on every metric after the adjustment. On FB15k-237 it is 0.010 below on MRR and 0.012 below on Hits@1, but within 0.003 on Hits@10. The remaining gap is plausibly the differences still listed in the script (no subsampling weights, per-negative head/tail corruption, dropped rather than redrawn filtered negatives); this was not tested. One seed only, so a gap of this size may partly be seed variation.

3. **Stock PyKEEN RotatE is a much weaker baseline.** It loses 0.018 MRR and 0.044 Hits@10 on FB15k-237, and on WN18RR it falls to less than half the official variant's MRR (0.217 against 0.507). Which of the differences causes this (L2 distance, relation parameterisation, initialisation, or the missing learning-rate decay) was not isolated.

## Consequence for G2

The protocol said RotatE runs "via PyKEEN" and did not say which RotatE. With stock PyKEEN RotatE, the G2 baseline would be handicapped and any gain credited to holonomy could be partly a weak-baseline artefact.

**Decided** in the protocol note of 2026-11-08, written before any G2 config:
- G2 uses the official-variant RotatE (`PhaseRotatEInteraction` in `experiments/reproduce_rotate.py`) at complex dimension 68. That is 1,988,796 parameters on FB15k-237, within budget.
- Training negatives are filtered.
- No learning-rate step decay, so every model follows the same early-stopping rules.
- QuatE gets the same check against its official code before it is used.
