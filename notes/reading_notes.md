# Reading notes and contribution statement

*Revised 8 Nov 2026 after review, drafted with Claude. Verify every citation and equation against the original papers before reusing them in the report or paper. The references are listed at the end.*

## RotatE (Sun et al., ICLR 2019)

### Scoring function
- Entities are vectors h, t ∈ ℂᵏ. A relation is a vector r ∈ ℂᵏ with every |rᵢ| = 1, so rᵢ = e^{iθᵢ} rotates coordinate i.
- A true triple should satisfy t ≈ h ∘ r (coordinate-wise product). The distance is d_r(h, t) = ‖h ∘ r − t‖.
  - The official code sums the complex moduli, an L1 norm over moduli.
  - It stores relations as phases θ = p·π/ρ with ρ = (γ + 2)/k, and initialises parameters uniformly in ±ρ.
  - PyKEEN 1.11.1's stock RotatE instead uses an L2 distance and unit complex numbers renormalised each step. In `notes/baseline_reproduction.md` it scores well below the published figures.
- **Relation patterns RotatE can express:**
  - symmetry: every θᵢ ∈ {0, π};
  - antisymmetry: r ∘ r ≠ 1;
  - inversion: r₂ = r̄₁;
  - composition: r₃ = r₁ ∘ r₂.

  Because the rotations are diagonal, composition is **commutative**: r₁ ∘ r₂ = r₂ ∘ r₁.

### Loss and self-adversarial negative sampling
- **Loss**, with margin γ, n negatives (h′ᵢ, r, t′ᵢ) and σ the logistic function:

  L = −log σ(γ − d_r(h, t)) − Σᵢ p(h′ᵢ, r, t′ᵢ) · log σ(d_r(h′ᵢ, t′ᵢ) − γ)

- **Self-adversarial weights:** negatives are weighted by a softmax over the current model's scores, at temperature α:

  p(h′ⱼ, r, t′ⱼ) = exp(α·f_r(h′ⱼ, t′ⱼ)) / Σᵢ exp(α·f_r(h′ᵢ, t′ᵢ)), where f_r = −d_r.

  - Hard negatives, the ones the model currently scores as plausible, get more weight.
  - The weights are treated as constants, so no gradient flows through them. This replaces sampling from a separately trained adversary, as in KBGAN.
  - PyKEEN implements this as the `nssa` loss, with a factor of ½ on the positive and negative parts.

### Metrics
- **Link prediction:** for each test triple, rank the true tail among all entities as candidate tails, and likewise for the head.
- **Filtered setting:** other candidates that form known true triples (in train, validation or test) are removed before ranking.
- **Reported metrics:** mean rank (MR), mean reciprocal rank (MRR), and Hits@1/3/10, the share of ranks ≤ k. All are averaged over head and tail prediction.
- **Our implementation:** PyKEEN uses the *realistic* rank, which breaks ties by their expected position.
- **G2 doesn't gate on these metrics.** It uses grouped AUC against 10 role-matched negatives.

### Five open questions
1. **Why is stock PyKEEN RotatE so much weaker?** It scores 0.217 MRR against 0.507 on WN18RR. Which difference matters: the L2 distance, the unit-modulus parameterisation (relation updates are about 285× smaller at the same learning rate), the U(0, 1) initialisation, or the missing learning-rate decay? An ablation changing one at a time would answer it.
2. **How much does self-adversarial sampling contribute at our scale?** The paper compares it with other negative-sampling schemes at full size. Does the gain hold at complex dimension 68 and the 2.0M budget?
3. **Does an MRR-tuned RotatE do as well on role-matched negatives?** G2 ranks each positive against 10 negatives drawn from the entities seen in that role for r. These are harder than uniform corruptions. A model trained with uniform negatives may separate them less well than its MRR suggests.
4. **How do rotations handle 1-to-N and N-to-1 relations?** A rotation is a bijection, so h ∘ r cannot sit close to many different tails at once. How much of RotatE's FB15k-237 error comes from such relations? Does the holonomy model, which also uses orthogonal maps, share the limitation?
5. **How much of RotatE's WN18RR score comes from reverse edges?** 37.1% of WN18RR test positives have their exact reverse (t, r, h) in training (`notes/dataset_stats.md`), and inversion r₂ = r̄₁ captures that directly. What is RotatE's MRR on test triples without such a reverse edge?

## How the holonomy score differs from Knowledge Sheaves (Gebhart et al.)

**The sheaf view.**
- A knowledge sheaf assigns each entity a vector space (its stalk) and each relation a pair of restriction maps.
- An embedding ψ is consistent if the restriction maps agree on every edge. The Laplacian energy ψᵀLψ sums these disagreements.
- Using it as a score means choosing or optimising ψ, with a constraint such as ‖ψ‖ = 1 so that ψ = 0 doesn't give zero energy trivially. The result describes the graph as a whole, or the edges around a given ψ.

**The holonomy score.**
- Once the relation rotations are learned, each cycle is scored by ‖H_c − I‖_F. No vertex signal is involved, so the score is ψ-free.
- **The link to the Laplacian:** for orthogonal transports on a connected graph, ψ ∈ ker L means ψ is preserved by every transport.
  - So dim ker L equals the dimension of the subspace that every holonomy based at a root vertex leaves fixed.
  - The Laplacian answers "is there any globally consistent signal?". ‖H_c − I‖ says which cycle is inconsistent, and by how much.
- **What it cannot do:** locate a corrupted edge more precisely than its cut-pair group (`notes/proofs.md`, Lemmas 1–3).

## Contribution statement

**Closest prior work.** For orthogonal U_r, the holonomy of the cycle that closes a path p with the edge (h, r, t) satisfies

‖H_c − I‖_F = ‖U_rᵀ U_p − I‖_F = ‖U_p − U_r‖_F.

So the score measures how far the composed path operator is from the operator of the relation being scored. This is the relation-path composition idea of PTransE (Lin et al., 2015) and Guu et al. (2015), with orthogonal, possibly noncommutative, operators in place of their composition functions. The work must be positioned as building on these papers, not as a new kind of score.

**What is proposed:**
1. **A ψ-free cycle score from learned orthogonal relation operators.** Block rotations in SO(64) can be noncommutative, so the order of relations along a path can change the score.
2. **Training with a cycle-consistency term** alongside the single-triple term.
3. **A pre-registered test of added information.** The baseline is a strong RotatE (the official-code variant at the 2.0M budget), plus untyped graph-structure features, plus holonomy computed from Haar-random rotations on the same paths. The random-rotation feature measures how much signal path lengths, relation sequences and path counts carry without learned operators.
4. **A localisation limit:** cycle tests identify a corrupted edge only up to its cut-pair group, and the ambiguity is exact when one side of the cut is consistent (`notes/proofs.md`). The graph facts behind it, such as the cycle space being orthogonal to the cut space, are standard. The contribution is applying them to localisation.

**What a G2 pass would show.** On FB15k-237, adding learned holonomy to that baseline (B₁ against B₀) raises grouped AUC on the test set. The lower bound of the 95% relation-cluster bootstrap CI must be ≥ 0.005, with a positive gain in each of seeds 0, 1 and 2. That shows learned holonomy adds predictive information beyond this baseline in this setup.

**What it would not show:**
- **That noncommutativity caused the gain.** S6 compares holonomy from RotatE's and QuatE's rotations on the same paths, which separates the operators. Holonomy from RotatE's commutative rotations is itself a path-composition score, so S6 is also the closest test against PTransE-style composition.
- **That it beats typed path rules** (PRA/AMIE-style). That claim is made only if S3 also meets the primary criterion.
- **That it works on WN18RR.** The protocol expects little gain there, stated in advance.
- **That it generalises** to other datasets, budgets or negative schemes.
- **Edge-level localisation.** Localisation is reported by cut-pair group.

## References (verify before citing)

- Sun, Deng, Nie, Tang. *RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space.* ICLR 2019. arXiv:1902.10197.
- Zhang, Tay, Yao, Liu. *Quaternion Knowledge Graph Embeddings.* NeurIPS 2019.
- Gebhart, Hansen, Schrater. *Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding.* AISTATS 2023.
- Hansen, Ghrist. *Toward a Spectral Theory of Cellular Sheaves.* Journal of Applied and Computational Topology, 2019.
- Lin, Liu, Luan, Sun, Rao, Liu. *Modeling Relation Paths for Representation Learning of Knowledge Bases.* EMNLP 2015. (PTransE)
- Guu, Miller, Liang. *Traversing Knowledge Graphs in Vector Space.* EMNLP 2015.
