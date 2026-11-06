# Experiment protocol (pre-registration)

**Commit this file by Fri Nov 20, 2026, before any G2 result exists. After that, thresholds change only with a dated note explaining why, and the original value stays visible.**

Committed on: 2026-11-06. The frozen version is the commit tagged `prereg-g2` (a file cannot contain its own commit hash, so the tag names it).

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
| G2 (H2) | 2027-01-02 | Phase 2 scripts, committed before the test split is first used (see Timing below) | One primary test, defined in full in the G2 section below: on the FB15k-237 test split, adding the holonomy feature to RotatE + structure raises grouped AUC; the 95% relation-cluster bootstrap CI of the seed-averaged gain has lower bound ≥ 0.005, and the gain is positive in each of the 3 seeds. | |
| G3 (H1) | 2027-01-30 | Phase 3 scripts | Page recall@k ≥ FAISS recall@k (k = ____); MetaQA Hits@1 within ____ points of baseline B | |
| G4 (H3–H4) | 2027-02-27 | Phase 4 scripts | Unsupported-claim rate below max-softmax abstention at coverage ____%; parser edge F1 ≥ ____ | |
| G5 (H5) | 2027-03-20 | Phase 5 scripts | Verified pass rate above generate-then-verify at equal compute, by ≥ ____ points | |

The blanks are yours to decide **before** the relevant experiment. Choose them from what would convince a skeptical reviewer, not from what you expect to get.

## Dated changes

Each change keeps the original value visible.

- **2026-11-06, G1 slope threshold 1.2 → 1.3.** Changed during scaffolding, after the reference implementation (written with Claude) measured a slope of 1.13, and before my own implementation was run. Reason: cache effects at small sizes push measured slopes above 1 even for a linear-time method.
- **2026-11-06, G1 is not pre-registered in the strict sense.** The G1 thresholds were written in the scaffold, but they were first committed (commit `f4c4c09`) together with my already-working implementation, and G1 was measured the same day. G1 is recorded here for completeness. This file pre-registers G2 onwards.
- **2026-11-06, H2 sharpened.** Proposal Section 2 states H2 as "the cycle holonomy defect separates true from false facts better than existing rotation-based scores", falsified by "AUC no better than RotatE, and no gain when the two scores are combined". That allows two routes to success. Here it becomes one primary test of added information, and graph structure joins the baseline, because the structure checks below showed shortcuts that would otherwise be credited to holonomy. The proposal text will be updated to match.
- **2026-11-06 (after tag `prereg-g2`, before any use of a test split), baseline reproduction may use the test split.** For the plan's baseline reproduction (Dec 7–9), RotatE and QuatE may be trained at their published hyperparameters, and their filtered test MRR and Hits@1/10 computed, solely to check the implementation against published figures. These runs are separate from G2:
  - their numbers are not G2 features;
  - they do not influence any G2 configuration, search range or threshold;
  - no holonomy-model score, negative set or path feature is evaluated on a test split before the G2 timing condition below is met.

  The published-hyperparameter models are larger than the G2 parameter budget, so their MRR is not comparable with the budget-matched G2 models.
- **2026-11-06 (after tag `prereg-g2`, before any tuning), parameter budget fixed at 2.0M; dense rotations become an ablation.** On FB15k-237 the holonomy model's trainable parameters are:
  - 1.92M with rank-1 Cayley rotations;
  - 1.98M with rank 2;
  - 2.10M with rank 4;
  - 3.80M with dense rotations (`experiments/time_budget.py`).

  The plan's rank sweep included "dense", so the budget-matched baseline sizes would change from trial to trial. Changes:
  - **Budget:** fixed at **2.0M ± 5%** (1.90M–2.10M) for every model;
  - **G2 tuning:** sweeps rank r ∈ {1, 2, 4} only;
  - **Baselines:** RotatE (complex dimension 68) and QuatE (dimension 34) are sized once to this budget;
  - **Dense rotations:** run only as secondary ablation S10. It is outside the budget and reported as such, and cannot affect the gate.

## G2 (H2) in full

**Hypothesis H2.** The holonomy defect of learned relation rotations carries information about whether a triple is true that is not already in RotatE's score or in the untyped graph structure around the triple.

**Scope of a pass.** In this model every relation type has one rotation, so a cycle's holonomy depends only on r and the relation types along the path: it is a learned path-rule feature. A pass therefore shows that learned holonomy adds information beyond RotatE and untyped structure. It does **not** show that holonomy beats typed path rules (PRA/AMIE-style). That stronger claim is tested in secondary analysis S3, and the paper makes it only if S3 also meets the primary criterion.

**What the data already says.** Sources: `notes/dataset_stats.md` and `experiments/structure_check.py`, structure-only checks run 2026-11-06 before any model was trained. The check uses 3,000 sampled test positives per dataset, with negatives drawn as below but without the fallback.

- **FB15k-237 has a construction artefact.** By construction, 0% of test positives have a direct training edge between h and t, but 1.2% of the negatives do. Positives' tails also have higher degree: median 87 against 48.
- **WN18RR has the opposite one.** 37.5% of test positives have a direct training edge, nearly all exact reverses: 37.1% have (t, r, h) in training. 0% of negatives do.
- **Whether a triple closes a cycle at all is a strong signal.** On WN18RR, 70.5% of positives but 2.7% of negatives close a cycle of length ≤ 3. On FB15k-237, the median number of length-≤3 walks among covered triples is 990 for positives and 216 for negatives. These are walks, not simple paths.

The baseline below therefore includes all of these signals, so that holonomy is credited only for what goes beyond them.

### Data

- **Splits:** PyKEEN's versions of the standard splits. These drop evaluation triples with an entity unseen in training (WN18RR: 210 of 3,134 test triples; FB15k-237: 28 of 20,466). Every paper-facing table also states the published sizes and the dropped counts.
- **Validation is split once, 50/50 by positive, seed 0:**
  - **V₁** is used for hyperparameter search and early stopping;
  - **V₂** is used only to fit the combiners and their feature standardisation.
- **Primary dataset:** FB15k-237. **Secondary:** WN18RR, analysed with the same procedure but not part of the gate.
- **Expectation, stated in advance:** holonomy should add little on WN18RR. No relation has an inverse partner or a 2-hop path pattern covering ≥ 80% of its triples. Only 70.5% of its test triples close a cycle of length ≤ 3, and its cycle and reverse-edge signals are already captured by the structure features. A negative WN18RR result is expected and will be reported as such.
- **No leakage through paths:** paths, cycles and every structure feature use **training triples only**. No validation or test triple is ever part of a path, a training cycle or a count.

### Negatives (false triples)

- **Count:** for each evaluation positive (h, r, t), 5 head replacements (h′, r, t) and 5 tail replacements (h, r, t′), giving 10 negatives per positive.
- **Sampling:** replacements are drawn uniformly without replacement from the entities seen in that role for r in training: heads of r for h′, tails of r for t′.
- **Filtering:** any candidate that is a known true triple in train, validation or test is discarded and redrawn.
- **Fallback:** a role pool can be too small to supply 5 distinct valid replacements. In the 2026-11-06 check this happened for 459 of the 6,000 sides (3,000 positives × 2 sides) on FB15k-237. The remainder is then drawn uniformly from all entities not already used for that positive and side, with the same filtering, and flagged. The fallback rate is reported for validation and test.
- **Reuse:** negatives are generated once with seed 0, saved to disk and used unchanged by every model. Validation negatives are built the same way from the validation split.

### Paths and cycles

- **Path definition:** a path is a simple path from h to t in the training graph. The graph is treated as undirected, since a cycle may cross an edge backwards using U^T, and as a multigraph, where each fact is its own edge. Length cap: **L ≤ 3** edges.
- **Own edge excluded:** when the triple being scored is itself a training triple (training cycles in the loss, and S3), its own edge is removed before its paths are searched, so a triple never supports itself.
- **Up to K = 32 paths per triple, chosen by length tier:** length 1 first, then length 2, then length 3, until 32 are chosen.
  - A tier that fits entirely in the remaining slots is taken whole.
  - The first tier that overflows is sampled uniformly without replacement (seed 0), and later tiers are not used.
  - Reason: uniform sampling over all paths would pick length 3 almost every time. In `structure_check.py`, the median share of length-3 walks among a covered triple's length-≤3 walks is 99.7% on FB15k-237 and 95.8% on WN18RR (walks, not simple paths). In the Phase 1 noise test, clean scores grow as √L.
- **Exact counts** n₁, n₂, n₃ of simple paths of each length are recorded for every triple. A triple is **covered** when n₁ + n₂ + n₃ > 0.
- **Reuse:** paths are sampled once with seed 0, saved to disk, and shared by every model that computes a holonomy.
- **Triple holonomy score:** the mean of ‖H_c − I‖_F over the triple's sampled cycles, where H_c closes the path with the evaluated edge. Sensitivity analysis: the median.
- **No paths:** a triple with no path of length ≤ 3 has no holonomy score and is never treated as clean.

### Models

- **Holonomy model:** one block-diagonal rotation per relation type (2 blocks of 64, d = 128) and one state per entity. It is trained on training triples and training cycles only, with the loss of proposal Section 5.2. The loss terms, their search ranges and all fixed values are written in `configs/g2_holonomy.yaml`.
- **Baselines:** RotatE (primary, commutative) and QuatE (secondary, noncommutative), both via PyKEEN. Their search ranges are in `configs/g2_rotate.yaml` and `configs/g2_quate.yaml`.
- **Parameter budget:** every model's trainable parameter count, counting stored tensor elements as `rtvm.utils.count_parameters` does, is within ±5% of the holonomy model's. Embedding dimensions are set by this budget and are not tuned.
- **Tuning budget:** 20 random-search trials per model over its config ranges. A model's score in a trial is the grouped AUC (defined below) of its own triple score on V₁: RotatE's and QuatE's PyKEEN `score_hrt`, and the holonomy model's triple holonomy score. For the holonomy model only covered V₁ positives count, each compared against its covered negatives; a positive with no covered negative is skipped. Training evaluates on V₁ every 10 epochs, stops after 5 evaluations without improvement, and runs at most 1,000 epochs. These rules are the same for every model.
- **Seeds and hardware:** seeds 0, 1, 2; the same training data and hardware for every model. Seed s of every model is used together: B₀ and B₁ for seed s use RotatE run s and holonomy-model run s.

### Primary test (the gate)

- **Structure features**, computed from training triples for every positive and negative:
  - covered (0/1);
  - log(1 + n₁), log(1 + n₂), log(1 + n₃);
  - the shares of lengths 1, 2 and 3 among the triple's sampled paths;
  - log(1 + degree) of h and of t, where degree is the number of training facts touching the entity;
  - **null holonomy:** the same triple holonomy score on the same saved paths, with every relation's rotation replaced by a Haar-random rotation (`rtvm.synthetic.random_rotation`, 2 blocks of 64, seed 0, the same for all seeds), and 0 for uncovered triples. It absorbs anything the holonomy score encodes through path count and length mix alone.
- **RotatE feature:** RotatE's `score_hrt`.
- **Holonomy feature:** the holonomy model's triple holonomy score, set to 0 for uncovered triples. The "covered" flag tells the combiner it is missing. The holonomy model's own single-triple score ‖U_r ψ_h − ψ_t‖ is not a feature: it is a RotatE-like score, and including it would test a second embedding rather than holonomy. It is reported in S8.
- **Two combiners.** Each is a logistic regression on features standardised with V₂ statistics (L2, C = 1.0), fitted on V₂ positives and negatives only:
  - baseline B₀ = RotatE + structure;
  - candidate B₁ = B₀ + holonomy.
- **Grouped AUC.** For each positive: the fraction of its 10 negatives scored below it, with ties counting ½. Then the mean over positives. Pooled AUC is secondary, because it rewards per-relation recalibration that carries no information about individual triples.
- **Statistic:** Δ_s = grouped AUC(B₁) − grouped AUC(B₀) on the full FB15k-237 test set, covering all positives and all their negatives, covered or not, for seed s. Δ̄ is the mean over the 3 seeds.
- **Confidence interval:** a cluster bootstrap with 10,000 resamples (RNG seed 0) over **relations**. Each resample draws relations with replacement and takes all their test positives with their negatives. The same resample applies to all 3 seeds and both combiners, and Δ̄ is recomputed within each resample. The 95% CI is the percentile interval.
- **Smallest effect of interest:** 0.005 grouped AUC.
- **Decision:**
  - **pass:** the CI's lower bound is ≥ 0.005 **and** Δ_s > 0 in each of the 3 seeds;
  - **negligible:** the CI is entirely above 0 but its lower bound is below 0.005;
  - **fail, no added information:** the CI contains 0;
  - **fail, holonomy hurts:** the CI is entirely below 0.

  There is one primary test. The secondary analyses cannot turn a fail into a pass.
- **Timing:** the test split is touched for the first time only after all of the following are committed:
  - the G2 scripts (negative and path generation, feature computation, combiners, bootstrap);
  - the saved negatives and paths;
  - the three configs.

  The commit hash is recorded in a dated note above before the first test evaluation. Gate date: 2027-01-02.

### Secondary analyses (reported whatever the gate says)

1. **S1:** the same test on WN18RR. It has only 11 relations, so its relation-cluster CI is wide and is reported as such.
2. **S2:** grouped AUC on the covered subset only, with the counts of covered positives and negatives, and pooled AUC on the full set.
3. **S3, typed path rules:** B₀ is extended with a rule feature, and the primary test is repeated against it.
   - **Estimation set:** up to 300 training triples per relation (uniform, seed 0), each with 10 role-matched negatives built from training data with the same rules as above. Paths come from the same sampler, with the own edge excluded.
   - **Confidence of a pattern p** (a sequence of relation types, with directions) for relation r: among the estimation triples of r whose sampled paths contain p, the fraction that are positives.
   - **Rule feature:** the maximum and the mean confidence over the patterns on the triple's sampled paths, and 0 with the covered flag when uncovered.
4. **S4:** results by path length (cycles of length 2, 3, 4), and the whole test repeated with L ≤ 2.
5. **S5:** QuatE in place of RotatE in both combiners.
6. **S6:** holonomy computed from RotatE's and from QuatE's learned relation rotations on the same paths. This separates the scoring rule, noncommutativity and the cycle-term training.
7. **S7:** per-relation grouped AUC. On WN18RR, this compares triples whose exact reverse (t, r, h) is in training against the rest.
8. **S8:** filtered MRR and Hits@1/10 for every model, for comparison with published results, including the holonomy model's single-triple score.
9. **S9:** results with flagged fallback negatives excluded, and B₁ with the holonomy feature in quantile bins, to catch non-monotone effects.
10. **S10:** the holonomy model with dense rotations (3.80M parameters, outside the 2.0M budget), with the primary test repeated and the size difference stated.

## Method choices fixed in advance

- **Localisation with learned or noisy rotations** uses `localise_corrupted_edges(..., threshold=None)` (mean cycle score per edge). Edge scores are averaged within each cut-pair group (`cut_pair_classes`) and groups are ranked. With m the number of distinct non-bridge groups that contain a corrupted edge, recall is the fraction of those groups among the top m ranked groups. Bridges are excluded from recall and their share is reported separately. (The G1 test above uses the counting mode on exact rotations, as specified before this change.)

## Datasets and splits

| Dataset | Source | Split used | Notes |
| --- | --- | --- | --- |
| FB15k-237 | PyKEEN loader (Microsoft release `FB15K-237.2.zip`) | standard train/valid/test, PyKEEN-reduced | drops 9 valid, 28 test triples (entity unseen in training) |
| WN18RR | PyKEEN loader (ConvE repository `WN18RR.tar.gz`) | standard train/valid/test, PyKEEN-reduced | drops 210 valid, 210 test triples (entity unseen in training) |
| MetaQA | ____ | 1/2/3-hop, standard | |
| WebNLG | ____ | ____ | |
