# column_sketch

Global **rMPS-sketched column QR** for the block Moses move: replace the
sequential *local* Moses sweep (split each row, carry a residual) with **one**
randomized range finder over the whole active column, using random matrix product
state (rMPS) probes (Camaño–Epperly–Meyer–Tropp, `docs/rmps.pdf`).

The whole active column is one linear map `C_j : X_j → Y_j` (absorbed legs →
retained legs). The exact column QR is `C_j = Q_j R_j` with `Q_j` the new
isometric column and `R_j` the residual absorbed into the neighbour. The global
move approximates this in one shot:

```
Ω  : rMPS / Gaussian test matrix over the absorbed legs   (sketch bond χ_sk)
Y  = C_j Ω            (ℓ matrix–MPS products = the access model)
Q  = orth(Y)          (the new isometric column)
Ĉ  = Q (Q* C_j)       (rank-ℓ column approximation;  R = Q* C_j)
```

This is a **mathematical-validation** suite (tiny materialized columns), not an
end-to-end speedup claim. The reusable library is built to slot straight into a
real isoTNS run later — a real active column, once its top/bottom environment is
contracted, *is* an MPO from absorbed to retained legs, so the same
`ColumnOperator`, range finder, and experiments run on real columns by swapping
the operator constructor.

## Library (in `src/rand_isopeps/`)

- `linalg/rmps_sketch.py` — rMPS probe vectors / test matrices (Def. 1.1): per-core
  Gaussian variances `1/χ_sk` (interior), `1/√χ_sk` (boundary) giving isotropy
  `E[ωω*] = I`. `χ_sk = 1` is the Gaussian-Kronecker / Khatri–Rao limit. Also wired
  into `SketchSpec(kind="rmps")`.
- `column/operator.py` — `ColumnOperator`, the access seam: a column as an MPO with
  `.materialize()` (dense, for the tiny validation) **and** `.matvec_mps()` (the
  matrix-free MPO–MPS product, for the real access model). `random_column_operator`
  (faithful MPO) and `controlled_spectrum_column_matrix` (dense, prescribed decay).
- `column/global_range.py` — `global_column_range`: the sketched column QR + its
  diagnostics (column error, excess over the best rank-`r` SVD, isometry defect, and
  the **OSI** subspace-injection diagnostic `σ_min(V_r* Ω)²`). `sampled_bond_growth`
  is the matrix-free bond-growth probe (`D·χ_sk`).
- `column/local_moses.py` — `local_column_qr`: the sequential local Moses column QR
  baseline (det / randomized), for the global-vs-local comparison.

## Running

```
pip install -e .
python experiments/column_sketch/scripts/<name>.py --quick
```

Outputs (CSVs and figures) land in `outputs/column_sketch/` (gitignored).

## Scripts

- `exp01_materialized_column_rmps.py` — **Phase 1+2.** Materialize tiny columns and
  run the global range finder with Kronecker (χ=1), rMPS χ=2/4/8, and dense Gaussian
  probes, faceted by column height `Lx`. Headline figure (`exp1-injection-*`): the
  paper's `χ_sk ≳ Lx` thesis — the **subspace-injection** `σ_min(V_r* Ω)²` collapses
  for Kronecker as `Lx` grows (overwhelming orthogonality) while Gaussian stays flat
  and rMPS interpolates upward with χ. Detail figure (`exp1-materialized-detail-*`):
  per-`Lx` column error, excess over rank-`r`, OSI vs χ, and the isometry defect
  (≈1e-14 everywhere — every probe yields a genuinely isometric column).
- `exp02_global_vs_local_moses.py` — **Phase 4.** Global sketch vs the sequential
  local Moses on the *same* column at matched absorbed rank `k`. Result: the global
  flat-rank sketch hugs the Eckart–Young floor and is ~2–4× closer to optimal than
  the greedy local sweep; rMPS (χ≥2) matches the dense Gaussian; Kron (χ=1) begins to
  lag as `Lx` grows; local-det ≈ local-rand (randomizing local SVDs is
  accuracy-neutral). Global uses **1** range-finder primitive vs the local sweep's
  **`Lx`** sequential SVDs. Wall-clock is a labeled *secondary* panel (no speedup
  claim — that needs the full matrix-free algorithm including absorption).

## The fork it answers (briefing sec 5)

The local sweep's error is a sum of *local* truncation errors `Σ_i ε_i` (it caps the
local tensor-network rank); the global sketch targets the *flat* (whole-matrix)
column rank with error `~ τ_r(C_j)` (the best rank-`r` tail). Global wins when the
column has a low flat rank captured better in one shot than by greedy local splits —
which is what these synthetic columns exhibit. The decisive open question (briefing
sec 8), reachable only matrix-free at large `Lx`, is whether the sampled range
`range(C_j Ω)` rounds into a *low-bond* isometric column without losing the gain;
`sampled_bond_growth` measures the `D·χ_sk` bond the rounding must control.
