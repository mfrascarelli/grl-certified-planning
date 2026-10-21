# Certified GRL planning

Companion material for the brief paper
**"Certified Coordinated Path Generation via Generalized Root Locus and Harmonic Potentials"**
(M. Frascarelli, L. Pallottino, A. Landi; submitted to *Automatica*).

**Interactive notebook, runs in the browser with no installation:**
https://mfrascarelli.github.io/grl-certified-planning/

## Contents

| Folder / file | What it is |
|---|---|
| `grl_lesson.py` | Guided, interactive walkthrough of the method ([marimo](https://marimo.io) notebook). Published on GitHub Pages. |
| `matlab/` | Implementation used for the numerical results of the paper: images, critical points, certificate, branch tracing, Monte Carlo and baselines. See [`matlab/README.md`](matlab/README.md). |
| `verification/theorem_numcheck.py` | Short independent numerical check of Theorem 1 (departure direction, permutation, synchronous arrival, one reflection). |

## The interactive notebook

Short explanations alternate with interactive figures that recompute the objects of the paper on small examples.

| Section | Paper | What it shows |
|---|---|---|
| 1. GRL as a level set | Sec. 2, Lemma 1 | one robot, the family of GRL arcs, the gain along a branch |
| 2. Gradient motion | Sec. 2, Lemma 2 | what streamline invariance gives and what it does not |
| 3. Critical phases Θb | Sec. 3 | break points, collisions, change of the robot–goal assignment |
| 4. Main theorem | Sec. 3, Thm. 1 | statement and proof outline |
| 5. Synchronous arrival | Thm. 1, item 4 | robots = closed-loop roots, numerical check of κ(t) = κ₀/(1 − κ₀t) |
| 6. Boundary phases Θ∂ | Sec. 3–4 | obstacles, one excluded phase interval per circle |
| 7. Reflections and certificate | Sec. 4, Prop. 1 | residual, bound π r ε̄, finite safety test |
| 8. Reciprocal root locus | Sec. 4 | the classical RRL as a special case |
| 9. Computational procedure | Sec. 4, Alg. 1 | critical points checked by the argument principle |

Run it locally:

```bash
pip install -r requirements.txt
marimo run grl_lesson.py        # read-only app
marimo edit grl_lesson.py       # to see and edit the code
```

The notebook uses only NumPy and Matplotlib; its numerical routines are in the first (hidden) cells.

## MATLAB code: quick start

```matlab
cd matlab
run_certificate_flagship     % certificate on the Flagship example (Table 1 of the paper)
run_certificate_warehouse    % certificate on the Warehouse example
mc_summary({'results/mc_flagship.mat','results/mc_warehouse.mat'})   % Monte Carlo table (Table 2)
```

No toolbox is needed: starts, goals and circles of the canonical domain are stored in `matlab/data/`.

## Examples

- `examples/ten_robots_ten_obstacles/` – the *Forest* example of the extended version
  (10 robots, 10 circular obstacles, reflection depth 3). Python/NumPy; see the README in that folder.

## Extended version

M. Frascarelli, L. Pallottino, A. Landi, *Extending Harmonic Function Methods via Root Locus
for Coordinated Path Generation*, arXiv preprint (2026).
