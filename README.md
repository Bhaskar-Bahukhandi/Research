# RTVM: sheaf holonomy consistency engine

Phase 1 of the RTVM-100M research project. The proposal and the day-by-day plan live in the project's Claude doc; this repository holds the code.

**Status:** scaffold. The infrastructure works; the core functions are exercises you implement in weeks 3–4, each specified by tests.

## Setup

```bash
pip install -r requirements.txt
pytest -rs
```

On day one you should see the infrastructure tests pass and every exercise reported as `SKIPPED (not implemented yet: ...)`. That skip list is your to-do list.

## Layout

| Path | What it is | Status |
| --- | --- | --- |
| `rtvm/rotations.py` | Cayley map, low-rank Woodbury rotations, block helpers | Exercises: week 3 Mon–Tue |
| `rtvm/graph.py` | `RotationGraph`, spanning trees, cycles; tree transport | Transport exercise: week 3 Wed |
| `rtvm/defects.py` | Coboundary residuals, chord defects, holonomies | Exercises: week 3 Thu–Fri |
| `rtvm/localise.py` | Finding the corrupted edge from defects; bridges; cut-pair groups | Exercises: week 4 Mon, plus `cut_pair_classes` |
| `rtvm/projection.py` | Projection onto ker(δ⁰) | Exercise: week 4 Tue |
| `rtvm/synthetic.py` | Random rotations and graphs, consistent / corrupted / partially consistent graphs | Provided |
| `rtvm/utils.py` | Seeding, parameter counting | Provided |
| `tests/` | The specification for every exercise | Provided |
| `experiments/phase1_sanity.py` | Histograms and scaling plot for gate G1 | Provided |
| `experiments/phase1_week4.py` | Localisation curves, noise test, timing contrasts | Provided |
| `notes/phase1_report.md` | Phase 1 report skeleton | Fill in (Dec 4) |
| `notes/protocol.md` | Pre-registered pass thresholds; commit by Nov 20 | Fill in |
| `notes/weekly_log.md` | Saturday review template | Fill in |

## Working through the exercises

```bash
pytest -m week3 -rs          # Nov 23-28
pytest -m week4 -rs          # Nov 30-Dec 5
python experiments/phase1_sanity.py --full   # evidence for gate G1
```

Each stub's docstring gives the shapes, the math and hints, not the code. Suggested order: `skew` → `cayley` → `lowrank_skew` → `cayley_lowrank_apply` → `transports` → `propagate` → `edge_residuals` → `chord_defects` → `holonomies` → `holonomy_scores` → `localise_corrupted_edges` → `common_fixed_space` → `project_to_kernel`.

Rules for yourself:

- **Do not edit a test to make it pass.** If you believe a test is wrong, write down why in the weekly log and check the math with your mentor first.
- Prove the key facts on paper before coding them: Cayley gives an orthogonal matrix; the Woodbury determinant is 1 + |u|²|v|² − (u·v)²; the chord defect equals |(H − I)ψ_r|; the projection formula in `project_to_kernel`.
- Commit at least daily, with messages that say what changed and why.

## Conventions

- Edge `e = (u, v)` carries `U_e`, mapping u's frame to v's: consistent means `U_e ψ_u = ψ_v`.
- States have width `d = n_blocks × b`; rotations are stored as `(..., n_blocks, b, b)`.
- Prototype size: `d = 128`, `b = 64`. The final model uses `d = 1024`, `b = 64`.
