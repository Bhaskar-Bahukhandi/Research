# Proofs: what cycle tests can and cannot localise

*Lemmas 1 and 2 are my hand proofs (7–8 Nov 2026), with the gaps found in review closed. Lemma 3 was written by Claude to answer the follow-up question from that review. Check every step yourself before using any of this in the Phase 1 report.*

*Numerical check (10 Nov 2026, `tests/test_proofs.py`):*
- *Lemmas 2 and 3 hold to 1e-9 on random graphs, checked through the scores of random closed walks.*
- *When both sides of the cut are inconsistent, the scores differ, as Lemma 3's "when it fails" paragraph says.*
- *A numerical check is evidence, not a proof.*

## Setting

**The graph.**
- G = (V, E) is a connected multigraph.
- Each edge u→v carries a transport U_{u→v} ∈ O(d), and traversing it backwards uses U_{v→u} = U_{u→v}ᵀ.
- The holonomy H of a closed walk based at v₀ is the product of the transports along the walk.

**Gauge transformations.** A gauge transformation is a choice of G_v ∈ O(d) for every vertex. It acts by

U′_{u→v} = G_v U_{u→v} G_uᵀ,  ψ′_v = G_v ψ_v.

A holonomy based at v₀ becomes H′ = G_{v₀} H G_{v₀}ᵀ. The ψ-free score is unchanged because conjugation preserves the Frobenius norm:

‖H′ − I‖_F = ‖G_{v₀}(H − I)G_{v₀}ᵀ‖_F = ‖H − I‖_F.

The ψ-dependent defect ‖(H − I)ψ_{v₀}‖ is unchanged too. **So no cycle-based observation can tell two gauge-equivalent sets of transports apart.**

**Corruptions.** Any corrupted transport can be written as U′_{x→v} = C U_{x→v} with C = U′_{x→v} U_{x→v}ᵀ ∈ O(d). This is a corruption "on edge x→v, in v's frame".

## Lemma 1: a cut pair has one fundamental-cycle signature

**Definition.** Edges e and f form a *cut pair* if neither is a bridge and G − e − f is disconnected.

**Step 1: the two sides.** There is a vertex set S whose crossing edges δ(S) are exactly {e, f}.
- Since e is not a bridge, G − e is connected.
- Removing f from G − e disconnects it, so f is a bridge of G − e. Then G − e − f has exactly two components, S and V ∖ S, and f joins them.
- If e did not also join S and V ∖ S, then G − f would be disconnected, contradicting that f is not a bridge. So δ(S) = {e, f}.

**Step 2: every cycle uses both edges or neither.**
- Every element z of the mod-2 cycle space meets every cut in an even number of edges, because the cycle space is orthogonal to the cut space. A closed walk leaves S as often as it enters it.
- With δ(S) = {e, f}, this gives 𝟙[e ∈ z] + 𝟙[f ∈ z] ≡ 0 (mod 2), so 𝟙[e ∈ z] = 𝟙[f ∈ z].
- Fundamental cycles belong to the cycle space. So for **every** spanning tree, e and f appear in exactly the same fundamental cycles: they have the same signature.

**Step 3: consequence.** No set of cycles, from any number of trees, separates e from f by which cycles they lie on.

**Remark.** At least one of e and f is a tree edge of every spanning tree, because a spanning tree must cross δ(S).

## Lemma 2: a degree-2 vertex makes the corrupted edge exactly unidentifiable

**Setting.** Let w have degree 2, with edges x→w (transport A) and w→y (transport B), where x ≠ w ≠ y.

**Why only cycles matter.** Every *cycle* through w uses both edges, and its transport through w is BA. This is not true of paths: a path can end at w and use only one of the two edges. So the claim is about cycle observations only.

**The two corruptions are gauge-equivalent.** Put a corruption C on the first edge, giving (CA, B). Take the gauge G_w = Cᵀ, with G_v = I everywhere else. Then:

- U′_{x→w} = G_w (CA) G_xᵀ = CᵀCA = A;
- U′_{w→y} = G_y B G_wᵀ = B(Cᵀ)ᵀ = BC;
- every other transport is unchanged;
- ψ_w becomes Cᵀψ_w.

So (CA, B) is gauge-equivalent to (A, BC). By the Setting:
- holonomies based away from w are unchanged;
- holonomies based at w are conjugated;
- every ψ-free score ‖H − I‖_F and every ψ-dependent defect is unchanged.

**Conclusion.** No number of trees or cycles can tell which of the two edges carries the corruption. Localisation is possible only up to the pair.

## Lemma 3: when the ambiguity is exact for a general cut pair

**The question.** Lemma 2 needs nothing about the rest of the graph. For a general cut pair with δ(S) = {e, f}, does a gauge transformation still move a corruption from e to f?

**Setting.**
- e = x→v and f = v′→y, with v, v′ ∈ S and x, y ∉ S.
- A corruption C sits on e: U′_e = C U_e.
- Assume the transports **inside S are consistent**: every closed walk using only edges inside S has holonomy I.

Consistency gives path-independent transports T_{v→u} within S, with T_{v→v} = I. For each edge u→u′ inside S, U_{u→u′} T_{v→u} = T_{v→u′}.

**The gauge.** Define G_u = T_{v→u} Cᵀ T_{v→u}ᵀ for u ∈ S, and G_u = I outside S.

- **Edge e:** U′_{x→v} = G_v (C U_e) G_xᵀ = CᵀC U_e = U_e. The corruption is removed.
- **Edges inside S:** G_{u′} U_{u→u′} G_uᵀ = T_{v→u′}Cᵀ T_{v→u′}ᵀ · T_{v→u′} T_{v→u}ᵀ · T_{v→u} C T_{v→u}ᵀ = T_{v→u′} T_{v→u}ᵀ = U_{u→u′}. They are unchanged.
- **Edge f:** U′_{v′→y} = G_y U_f G_{v′}ᵀ = U_f · C̃, where C̃ = T_{v→v′} C T_{v→v′}ᵀ is the same corruption carried across S to v′.
- **Edges outside S:** unchanged.

**Result.** A corruption on e is gauge-equivalent to a corruption C̃ on f. By the Setting, the two are exactly indistinguishable. The same construction works if instead V ∖ S is consistent, using the gauge on that side. Lemma 2 is the special case S = {w}, which contains no cycle and so is trivially consistent.

**When it fails.** If both sides contain inconsistency (another corruption, or noise), the construction needs C to commute with the holonomies of S based at v. That generally fails. Then e and f still have identical signatures (Lemma 1), but their defect magnitudes can differ, so a method that uses magnitudes could in principle separate them.

**What this means for the Phase 1 localisation plateau.** In the week 4 experiments the clean graph is consistent. A corrupted cut-pair edge is therefore *exactly* indistinguishable from its partner whenever one side of the cut holds no other corruption. With few corruptions that is the usual case.
- More trees cannot help, so recall plateaus.
- With noisy rotations the symmetry is only approximate.
- For a sparse knowledge graph, with many low-degree entities and so many cut pairs and bridges, edge-level localisation is capped by its cut-pair structure. That is why the protocol reports localisation by cut-pair group.
