# Phase 1 report: the consistency engine

*About 2 pages. Write it in your own words; the prompts under each heading say what a reader needs, not what to conclude. Regenerate every figure from the commands given and quote the git hash you ran them at, so anyone can reproduce them.*

Author: · Date: · Code at commit:

## 1. What the engine computes

Prompts: the coboundary δ⁰ and what "consistent" means; tree–cotree split; transport T_v; holonomy H_c and why ‖(H_c − I)ψ_r‖ equals the chord defect (your derivation); the ψ-free score ‖H_c − I‖_F; projection onto ker δ⁰. Keep the math to what the results below need.

## 2. Correctness

Prompts: what the test suite checks (`pytest -rs`, count of tests); the G1 numbers from `experiments/phase1_sanity.py --full` (largest clean score, smallest corrupted score, missed cycles); the projection test against dense least squares; gradcheck through the engine.

Figure: `experiments/out/defect_histogram.png`

## 3. Cost

Prompts: measured log–log slope of chord defects vs |V| + |E| and the hardware it ran on; the dense-SVD contrast and its slope (what does theory predict for SVD of an (|E|d × |V|d) matrix?); Woodbury vs dense Cayley at b = 64 for each rank r, and why the advantage shrinks as r grows. Say how noisy the timings were and what you did about it (best of 3, free memory).

Figures: `experiments/out/scaling.png`; numbers in `week4_timing.json` (`python experiments/phase1_week4.py timing --full`)

## 4. What can be located

Prompts: localisation recall vs number of corruptions and number of trees, dense vs sparse graphs; the share of corrupted edges that were bridges; exact-edge vs cut-pair-group recall (needs your `cut_pair_classes`). Explain the plateau as the number of trees grows, using your degree-2 / cut-pair argument. What does this predict for a sparse knowledge graph?

Figure: `experiments/out/week4_localise.png` (`python experiments/phase1_week4.py localise`)

## 5. Robustness to noise

Prompts: the noise model (every edge perturbed by size σ in ‖R − I‖_F, k = 3 edges corrupted by size τ); where the AUC starts to fall relative to τ; how the clean score grows with cycle length and why that exponent is expected. What does this imply for the path-length cap in Phase 2 (G2 asks you to fix one)?

Figure: `experiments/out/week4_noise.png` (`python experiments/phase1_week4.py noise`)

## 6. Limitations

Prompts: bridges; cut-pair groups; synthetic graphs vs real knowledge graphs; dense storage of rotations at d = 1024; anything that surprised you.

## 7. Gate G1 decision

Pass / fail against the thresholds in `notes/protocol.md`, with the commit hash the protocol was frozen at. If anything was changed after freezing, say what and why.
