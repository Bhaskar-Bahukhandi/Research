# Weekly log

Copy the template each Saturday. Newest week at the top.

---

## Week N · dates

**Planned (from the day-by-day plan):**
-

**Done (link commits or plots):**
-

**Results this week (numbers, with seeds):**
-

**Blocked / slipped, and why:**
-

**Decisions made (and what evidence they rest on):**
-

**Next week's first task:**
-

---

## Week 0 · 6–8 November 2026

*Drafted with Claude from my notes and the repository; I checked the numbers against the files linked.*

**Planned (from the day-by-day plan):** nothing yet; the plan starts on 9 November. I worked ahead on Phase 1, the G2 pre-registration, and the RotatE baseline check planned for 7–9 December.

**Done (link commits or plots):**
- Phase 1 consistency engine and the week 3/4 experiments (`f4c4c09` to `bac7635`). This includes `cut_pair_classes` (`add3315`).
- Dataset statistics for FB15k-237 and WN18RR, and a PyKEEN practice run on Nations (`6f42ba0`, `d98498c`).
- G2 protocol pre-registered at tag `prereg-g2` (`59b4ab4`, 6 Nov). Dated notes since then:
  - 6 Nov: baseline test MRR allowed as an implementation check only; parameter budget fixed at 2.0M (`462ab33`).
  - 7 Nov: 8 training cycles per triple; TF32 for training only; each run logs its GPU instead of all models needing the same hardware; what "touched" means.
  - 8 Nov: which RotatE G2 uses (`d70ff65`).
- A100 compute benchmark (`11ae3d7`).
- RotatE reproduction on a Colab A100 (`d70ff65`, `notes/baseline_reproduction.md`).

**Results this week (numbers, with seeds):**
- **G1** (`experiments/out/phase1_summary.json`):
  - largest clean defect 5.0e-6 (threshold < 1e-4);
  - smallest corrupted defect 15.8 (threshold > 1e-2), with 0 corrupted cycles missed;
  - log-log time slope 0.95 (threshold < 1.3).
  - single corruption localised in **20/20** graphs (threshold ≥ 16), from `tests/test_localise.py::test_single_corruption_is_found_most_of_the_time`. Recounted on 10 Nov at commit `06cb177`; the engine files are unchanged since 6 Nov (`bac7635`).
  - All 80 week 3/4 tests pass. Recorded as a pass.
- **Parameter budget**, measured with `rtvm.utils.count_parameters` on FB15k-237:

  | Model | Parameters |
  | --- | --- |
  | Holonomy model, rank 1 | 1.92M |
  | Holonomy model, rank 2 | 1.98M |
  | Holonomy model, rank 4 | 2.10M (at the edge of 2.0M ± 5%) |
  | Holonomy model, dense rotations | 3.80M (outside the budget, so ablation S10 only) |
  | RotatE, complex dimension 68, official variant | 1,988,796 |
  | RotatE, complex dimension 68, stock PyKEEN | 2,004,912 |
  | QuatE, dimension 34 | 2,004,912 |

- **A100 timings** (A100-SXM4-40GB, seconds per epoch):
  - RotatE: 4.1;
  - QuatE: 11.1–11.5;
  - holonomy cycle term at 8 cycles per triple: 6.4 with TF32, 11.2 in FP32;
  - holonomy cycle term at 32 cycles per triple (the benchmark default, not a protocol value): 24.9 with TF32, 43.6 in FP32.

  The projection uses 22 runs per model (20 tuning trials plus 2 extra seeds) and an *assumed* 300 epochs each. The protocol allows up to 1,000 epochs with early stopping, so the real cost depends on when runs stop.
  - With the adopted setting (8 cycles, TF32): about 40 A100-hours.
  - Adding an assumed 4 s/epoch for the single-triple term, which wasn't benchmarked: about 47 h.
  - At 32 cycles it would have been about 74 h, or about 246 h at 1,000 epochs.
  - These figures exclude path sampling, feature construction and evaluation.
- **RotatE reproduction**: published hyperparameters and sizes, seed 0, filtered test set; an implementation check only.

  | Dataset | Variant | MRR | Hits@1 | Hits@10 |
  | --- | --- | --- | --- | --- |
  | FB15k-237 | official | 0.329 | 0.229 | 0.531 |
  | FB15k-237 | stock PyKEEN | 0.311 | 0.223 | 0.487 |
  | FB15k-237 | published | 0.338 | 0.241 | 0.533 |
  | WN18RR | official | 0.507 | 0.458 | 0.607 |
  | WN18RR | stock PyKEEN | 0.217 | 0.184 | 0.281 |
  | WN18RR | published | 0.476 | 0.428 | 0.571 |

  - **WN18RR adjustment:** PyKEEN drops 210 of WN18RR's 3,134 test triples. Counting them back with a reciprocal rank of about 0 (an assumption) gives about 0.473 / 0.427 / 0.567 for the official variant.
  - **Published figures** are taken from PyKEEN's `sun2019` configs; check them against Table 5 of the paper before citing.

**Blocked / slipped, and why:**
- **First Colab reproduction run lost.** It idle-disconnected after about 1.5 h, costing about 10 units. The periodic checks only read the notebook, which doesn't count as activity. The rerun saved checkpoints to Drive every 15 minutes, ran in a background process, and executed a status cell at each check. It ran for 10 hours without a disconnect.
- **No mentor.** I'm asking a faculty member to read the G2 protocol (Monday).
- **Compute.** Colab units reset to 300 and don't carry over. G2 needs most of that. I let the leftover pre-reset units expire, so RotatE seeds 1 and 2 (the plan's 9 December step) are still to do and will cost units after the reset.

**Decisions made (and what evidence they rest on):**
- **G1 slope threshold 1.2 → 1.3.** Made after the reference implementation (written with Claude) measured 1.13, and before my own implementation ran. Reason: cache effects inflate slopes at small sizes. Because the thresholds were first committed together with working code, G1 is recorded as not strictly pre-registered. Strict pre-registration starts with G2.
- **Budget fixed at 2.0M ± 5%, with G2 tuning over rank r ∈ {1, 2, 4} only.** Evidence: the counts above. Dense rotations would change the baseline sizes from trial to trial, so they become ablation S10.
- **20 random-search trials per model, the same for every model.** Equal search effort and a fixed cost. Trials are scored on V₁. V₂ is used **only** to fit the combiners and their feature standardisation, never for tuning or early stopping.
- **8 training cycles per triple, TF32 for training only, strict FP32 for evaluation.** Evidence: the A100 timings. 32 cycles with all three models would not fit the Colab budget.
- **G2 uses the official-code RotatE.** Evidence: the reproduction above. Stock PyKEEN RotatE would be a handicapped baseline. Training negatives are filtered. There is no learning-rate step decay, so every model follows the same early-stopping rules; the decay's effect is untested. QuatE gets the same check against its official code before S5 and S6.
- **Leftover units allowed to expire** rather than used for an undeclared "pretraining" run. No holonomy training code exists yet, and a run before the G2 configs are committed would count as tuning outside the budget.

**Next week's first task:**
- Monday 9 Nov: email about GPU access, and ask a faculty member to read `notes/protocol.md` at `prereg-g2` plus the dated notes.
- Then the G2 data pipeline: the V₁/V₂ split, saved negatives, the path sampler and structure features. Write the test that no validation or test triple ever appears in a path *before* writing the sampler.
