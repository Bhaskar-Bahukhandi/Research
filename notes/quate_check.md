# QuatE: PyKEEN 1.11.1 against the official code (2026-11-14)

**Purpose:** the official-code check required by the protocol note of 2026-11-08 ("before QuatE is used in S5 or S6, it gets the same check against its official code. The check could change which QuatE is used."). This note covers the **code comparison**. No QuatE model has been trained. No G2 config, range or threshold has changed. `configs/g2_quate.yaml` stays provisional, and no QuatE trial has run.

- **Official code:** github.com/cheungdaven/QuatE, commit `77cdd49f2bc2` (2020-04-22). PyKEEN's QuatE docstring cites it. It is built on OpenKE and has no licence file, so it was read for comparison only and no code was copied. Files read: `models/QuatE.py`, `models/Model.py`, `config/Config.py`, `base/Corrupt.h`, `train_QuatE_FB15K237.py`, `train_QuatE_WN18RR.py`.
- **PyKEEN:** `pykeen/models/unimodal/quate.py`, `QuatEInteraction` in `pykeen/nn/modules.py`, `pykeen/nn/quaternion.py`, `init_quaternions` in `pykeen/nn/init.py`, `LpRegularizer`/`lp_norm`, `Representation.forward`, and the bundled `pykeen/experiments/quate/zhang2019_*.json`.
- **Variant:** the official code is QuatE¹ (no type constraints, no N3, no reciprocal relations). Dropout, batch norm, `fc` and `rel_w` are defined but not used in `forward`.

## Published numbers (QuatE¹)

These come from PyKEEN's bundled `zhang2019` configs. **Check them against Zhang et al. (2019), arXiv:1904.10281, before citing.**

| Dataset | MRR | Hits@1 | Hits@10 |
| --- | --- | --- | --- |
| FB15k-237 | 0.311 | 0.221 | 0.495 |
| WN18RR | 0.481 | 0.436 | 0.564 |

## What matches

1. **Score.** Both use −Σ_d ⟨h_d ⊗ r̂_d, t_d⟩: the Hamilton product (same multiplication table), the real inner product with t, and the same sign. The sign convention is harmless: replacing every relation r by −r flips the score, so training under either convention reaches the same model up to that flip.
2. **Relation normalisation.** Each relation quaternion is scaled to unit length before scoring, per dimension (`denominator_b` in the official code, `quaternion.normalize` in PyKEEN).
3. **Entity initialisation.** Both draw modulus ~ U(−s, s) with s = 1/√(2·n_entities), a uniform phase and a random unit imaginary axis. PyKEEN draws the phase from [0, 2π] and the official code from [−π, π], which gives the same distribution.
4. **Loss form.** Softplus of the signed score, averaged over positives and negatives together. In the official code this is `mean(softplus(score · y))`, with y = ±1. In PyKEEN it is `BCEWithLogitsLoss(reduction="mean")`, whose `process_slcwa_scores` concatenates positive and negative scores.

## What differs in the model definition

5. **The relation regulariser has no effect in PyKEEN.** The official code adds λ·Σ_c mean(r_c²) on the *raw* (unnormalised) relation parameters. PyKEEN applies the regulariser *after* the normaliser (`Representation.forward`: normalize, then `regularizer.update`), so it only ever sees unit quaternions. Checked numerically on 2026-11-14: the term is a constant 0.5 and its gradient is ~1e-7, i.e. zero. In the provisional G2 config, `regularizer_weight` therefore acts on entities only.
6. **The entity regulariser has a different form.** The official code uses λ·(mean |h|² + mean |t|²), the *squared* modulus per quaternion, averaged over batch × dimension, with h and t taken over positives and negatives. PyKEEN's `LpRegularizer(p=2, normalize=True)` takes `x.norm(p=2, dim=-1)` over the 4 components, which is the *unsquared* modulus |q|, divided by its expected value and averaged. The unsquared version pulls small embeddings towards zero more strongly and large ones less.
7. **Relations start with a different scale.** The official code initialises the relation tables with `quaternion_init(entTotal, …)`, so s = 1/√(2·n_entities), and the tables actually have n_entities rows, of which only the first n_relations are used. PyKEEN uses s = 1/√(2·n_relations). The official raw relations therefore start √(n_entities/n_relations) times smaller: about 7.8× on FB15k-237 and about 61× on WN18RR. Scores only see r̂, but Adagrad and Adam take steps of roughly fixed size per coordinate, so a smaller raw r turns faster. Together with item 5, where the official code keeps shrinking raw r, the official relations keep turning faster throughout training.
8. **`lmbda_two` is set but never used.** The scripts set `lmbda_two` (0.3 on FB15k-237, 0.01 on WN18RR), but `QuatE.loss` multiplies both terms by `lmbda`. The effective weights are therefore 0.3 / 0.3 on FB15k-237 and 0.1 / 0.1 on WN18RR.

## What differs in training (official vs provisional G2 config)

| | Official | Provisional G2 (`configs/g2_quate.yaml`) |
| --- | --- | --- |
| Negatives | exactly filtered against train (OpenKE `corrupt_head`/`corrupt_tail` draw from non-training entities), head/tail chosen by Bernoulli statistics (`bern=1`) | not filtered, head/tail uniform |
| Negatives per positive | 10 (FB15k-237), 1 (WN18RR) | 256 (shared with RotatE) |
| Optimiser | Adagrad, lr 0.1 | Adam, lr searched |
| Batch | train/10 positives (≈27.2k FB15k-237, ≈8.7k WN18RR) | 1,024 |
| Schedule | 5,000 / 40,000 epochs; validation Hits@10 every 1,000 / 10,000 epochs | G2 rules: V₁ every 10 epochs, patience 5, max 1,000 |
| Dimension | 100 | 34 / 32 (parameter budget) |

**Negatives.** The 2026-11-08 note called filtering "a RotatE-only choice". The official QuatE code also filters, exactly. With 256 negatives under a plain mean, positives make up 1/257 of the loss, against 1/11 (FB15k-237) and 1/2 (WN18RR) in the official setting.

## Reading

- PyKEEN's QuatE uses the official **score**. Its **regularisation differs** (items 5–7), and the provisional G2 config also differs in sampling and optimiser.
- Whether these differences change accuracy is **not known from code**. The RotatE check found a large gap only by training. For QuatE the most likely effects are item 5 together with item 7 (how quickly relations turn) and the 256:1 negative ratio under a plain mean.
- **Not decided here:** which QuatE G2 uses, and its sampler, optimiser and negative ratio. Changing `configs/g2_quate.yaml` needs a dated protocol note before any QuatE trial.

## Reproduction (in progress)

Following the RotatE check, the official definition is now implemented next to the stock one:

- **Model:** `rtvm.baselines.build_quate_official` (model name `quate_official`). Relations are stored raw and normalised inside the score. Both regularisers use squared moduli on raw parameters, pooled over every scored row. Relations are initialised with the entity-count fan-in. `tests/test_quate_official.py` checks the score against the official `_calc` written out component by component, and checks the regulariser against the official term over positive and negative rows. It also checks the relation-init bound and that the relation regulariser has a gradient. In the G2 trainer, `quate_official` samples negatives with PyKEEN's Bernoulli sampler.
- **Run script:** `experiments/reproduce_quate.py` trains `official` or `stock` (PyKEEN's bundled zhang2019 settings) at the official settings: dimension 100, Adagrad 0.1, official epochs and negatives. It keeps the best validation Hits@10 weights, as the official code does, and reports filtered test MRR. It saves a checkpoint every 50 epochs and resumes exactly, including RNG state, which was checked on Nations. This is an implementation check under the 2026-11-06 note, with test metrics used only to check the implementation.
- **Timing (2026-11-14, Kaggle T4, code at `aea1be7`, notebook `rtvm-quate-timing`):** one warm-up epoch, then 3 (FB15k-237) or 20 (WN18RR) timed epochs and one validation evaluation (eval batch 256, no out-of-memory error). Projected full runs at the official settings:

  | Run | Batch | s/epoch | Validation eval | Projected |
  | --- | --- | --- | --- | --- |
  | FB15k-237 official | 27,211 | 6.08 | 57 s (×5) | 8.5 h |
  | FB15k-237 stock | 9,300 | 5.52 | 56 s (×5) | 7.7 h |
  | WN18RR official | 8,683 | 0.66 | 27 s (×4) | 7.4 h |
  | WN18RR stock | 9,300 | 0.62 | 31 s (×4) | 7.0 h |

  The projections exclude start-up and the final validation and test evaluation, which take a few minutes. A two-GPU session can run one FB15k-237 and one WN18RR run side by side, and Kaggle counts that session's time once.
- **Full runs (2026-11-15).**
  - **Setup:** the official variant only, seed 0, official settings, one Kaggle T4 per dataset. Notebook `rtvm-quate-repro`, code at `a908e0c`.
  - **Training time:** 7.8 h on FB15k-237 and 6.5 h on WN18RR.
  - **Best weights:** chosen by validation Hits@10, as in the official code. For both datasets they came from the **last** validation (epochs 5,000 and 40,000). Validation Hits@10 was still rising:
    - FB15k-237: 0.455, 0.459, 0.469, 0.469, 0.473;
    - WN18RR: 0.572, 0.580, 0.585, 0.589.
  - **Test metrics:** filtered, realistic rank. Optimistic and pessimistic ranks give the same MRR, so ties play no part.

  | Dataset | Variant | MRR | Hits@1 | Hits@10 |
  | --- | --- | --- | --- | --- |
  | FB15k-237 | official, PyKEEN test set | 0.288 | 0.199 | 0.468 |
  | FB15k-237 | official, adjusted ×0.9986 | 0.288 | 0.198 | 0.467 |
  | FB15k-237 | published QuatE¹ | 0.311 | 0.221 | 0.495 |
  | WN18RR | official, PyKEEN test set | 0.514 | 0.469 | 0.594 |
  | WN18RR | official, adjusted ×0.933 | 0.480 | 0.438 | 0.554 |
  | WN18RR | published QuatE¹ | 0.481 | 0.436 | 0.564 |

  The adjustment counts back the test triples that PyKEEN drops, with reciprocal rank ≈ 0, as in `notes/baseline_reproduction.md`; it is an assumption, not a measurement.
- **Reading:**
  - **WN18RR reproduces** after the adjustment: MRR and Hits@1 are within 0.002 of the published values, and Hits@10 is 0.010 below.
  - **FB15k-237 is 0.023 below on MRR and Hits@1, and 0.028 below on Hits@10, about 7% relative.** That is a larger gap than the RotatE reproduction's 0.010.
  - **Plausible causes** (none was tested):
    - training was still improving at the official 5,000 epochs;
    - PyKEEN shuffles each epoch, where OpenKE draws batch rows with replacement;
    - filtered negatives are dropped rather than redrawn;
    - one seed only.
  - **The stock PyKEEN variant was not run**, so this check does not show how much the official definition gains over it. The code differences (items 5–7) are the evidence for preferring the official definition.
- **Next:** decide which QuatE G2 uses (S5, S6), and its sampler, optimiser and negatives, in a dated protocol note before any QuatE trial.
