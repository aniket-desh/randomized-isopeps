# Real Moses Move — randomized SVD: accuracy, work, and speed

This note explains the `real_moses_move` results: **E1** (per-stage accuracy ablation) and
**exp02** (matched-accuracy *time-to-accuracy*). It defines every term — the methods (`det`,
`gaussian`, `sparsestack`, `sketch-Q`), the rank fraction $\rho$, the matched-accuracy
speedup — and, crucially, **how we compare fairly**: with implementation-independent work
metrics (FLOPs, passes over data, peak memory), not just wall-clock, because we are testing
*algorithms*, not BLAS implementations or hardware. Figures live in
[`figures/real_moses_move/`](figures/real_moses_move/).

---

## 1. What the Moses move computes (the kernels we randomize)

An **isoTNS** represents a 2D state as a PEPS with an *orthogonality hypersurface*: tensors
on one side are isometries. The **Moses move** shifts that surface by one column — it
re-canonicalizes the PEPS. It sweeps a column, and at each site performs a **local isometric
tensor-ring split** built from **two truncated SVDs** plus an optional **disentangler**. These
local SVDs are the kernels we replace with randomized ones.

Grouped local dimensions (block-Moses model), for horizontal bond $\chi$, vertical bond
$\eta$, physical dim $d$, block index $p$:
$$
n_1=\chi\,\eta\,d,\qquad n_2=\eta\,p,\qquad n_3=\chi\,\eta,
\qquad k_1=\chi\,\eta,\qquad k_2=\eta .
$$
- **First SVD (`svd1`)**: reshape to $B_{(1),(23)}$ of shape $n_1\times(n_2 n_3)$, truncate to
  $k_1=\chi\eta$ → isometric column $U_1$ + residual $\tilde V=\Sigma_1 V_1^{\dagger}$.
- **Disentangler (`sketch-Q`)**: a unitary gauge $Q$ on the $\chi\eta$ bond minimizing the
  rank-$\eta$ tail $c_k(Q)=\sum_{i>k}\sigma_i^2(A(Q\tilde V))$. It is a pure gauge
  ($U_1\tilde V=(U_1 Q^\dagger)(Q\tilde V)$), so it reshapes the second-cut spectrum without
  changing the state before truncation. Solved by alternating minimization.
- **Second SVD (`svd2`)**: reshuffle to $(\eta n_2)\times(\chi n_3)$, truncate to $k_2=\eta$.

The residual ("R") column is then absorbed sideways by a deterministic zip-up compression
(not randomized here — a known future insertion point).

---

## 2. The methods

For $A\in\mathbb{C}^{m\times n}$ and target rank $k$, every method returns a rank-$k$
factorization $A\approx U_k\Sigma_k V_k^\dagger$; they differ in *how* they find the leading
$k$-dimensional subspace.

**`det` — deterministic truncated SVD (baseline).** Full SVD, keep top $k$. By **Eckart–Young**
this is the optimal rank-$k$ approximation: $\|A-U_k\Sigma_k V_k^\dagger\|_F=(\sum_{i>k}\sigma_i^2)^{1/2}$.
Cost $O(mn\min(m,n))$ — touches the whole matrix. Reference $\varepsilon_{\det}$, $T_{\det}$.

**Randomized SVD (HMT range finder) — template for `gaussian`/`sparsestack`.** Find an
approximate range with a random **sketch** $\Omega\in\mathbb{C}^{n\times\ell}$, $\ell=k+s$
($s$ = oversampling): $Y=A\Omega$, $Q=\operatorname{orth}(Y)$, $B=Q^\dagger A$, SVD of the small
$B$, map back; optional $q$ **power iterations** $(AA^\dagger)^q A\Omega$ for slow-decaying
spectra. Dominant cost is the sketch $A\Omega$ at $O(mn\ell)$ instead of $O(mn\min(m,n))$ —
cheaper by $\sim\ell/\min(m,n)=\rho$. Accuracy is a constant factor above Eckart–Young,
controlled by $s$. The two randomized methods differ only in the **distribution of $\Omega$**:

- **`gaussian`** — dense $\Omega_{ij}\sim\mathcal N(0,1)$. Most robust; applied as a dense
  BLAS-3 GEMM, $O(mn\ell)$.
- **`sparsestack`** — $\zeta$ stacked **CountSketch** blocks (each: one signed nonzero per
  input coordinate, hashed to a bucket), concatenated and scaled $1/\sqrt\zeta$ (recommended
  $\zeta=4$). A reliable oblivious subspace embedding where a single CountSketch is fragile.
  Returned width $\zeta\lceil\ell/\zeta\rceil\ge\ell$ (hence we record the **actual** width).
  Applied as a **compiled sparse matmul** (`scipy.sparse`); see §4 for why its wall-clock still
  trails `gaussian` despite doing *fewer* FLOPs.

**`sketch-Q`** — the disentangler's inner rank-$\eta$ search done with a randomized SVD (fresh
sketch each iteration); the Procrustes update keeps $Q$ exactly unitary. `RSVD1`/`RSVD2`/
`all-rand` name which stage(s) are randomized.

---

## 3. The rank fraction $\rho$ (the explanatory variable)

$$
\boxed{\;\rho=\frac{\ell}{\min(m,n)}=\frac{k+s}{\min(m,n)}\;}
$$
the fraction of the matrix's smaller dimension the sketch spans. $\rho\to1$: no room,
randomization only loses; $\rho\ll1$: the sketch is far smaller than $A$, randomization can win
big. Per stage $\rho_1\approx 1/d$ (high — little room; $\approx1$ on the small real lattice)
and $\rho_2\approx 1/n_2$. On the real move the carrier bond is $n_2\approx\text{bond}/4$, so
$\rho_2\approx 4/\text{bond}$ — **the source bond is the $\rho_2$ lever.**

---

## 4. Measuring fairly — algorithm work vs wall-clock

**Wall-clock alone is a biased way to compare algorithms**, because it conflates the algorithm
with (a) the maturity of the linear-algebra library and (b) hardware. Concretely: a dense
Gaussian sketch is a **BLAS-3 GEMM** running near peak FLOP/s, while a CountSketch is a sparse
scatter. We measured the gap on the exp02 svd2 matrix ($m{=}128,n{=}1024,\ell{=}8$): the dense
GEMM is **~100× faster** than a `np.add.at` CountSketch and still **~16× faster** than a tuned
`scipy.sparse` CountSketch — *despite the sparse sketch doing fewer FLOPs* ($\zeta/\ell$ of the
dense work). That gap does not shrink with $n$ (it is dense-vs-sparse-BLAS, not an $n$-regime
effect). So wall-clock differences below the algorithm level reflect libraries/hardware, not
the method.

To compare *algorithms*, we therefore report **implementation-independent work**, computed
analytically from matrix shapes ([`experiment_utils/cost_model.py`](../../src/rand_isopeps/experiment_utils/cost_model.py)):

1. **FLOPs** — multiply-adds for the sketch + QR + $B=Q^\dagger A$ + small SVD ($+$ power iters);
   SVD modeled by the Golub–Reinsch count $2ab^2+11b^3$. Implementation-free.
2. **Passes over $A$** — how many times the input is streamed (det: 1; rsvd: $1+2q$ — the
   classic architecture-free metric, and randomization's selling point for huge/streaming $A$).
3. **Peak intermediate memory** — the working set beyond $A$ (det $\sim\min(m,n)(m{+}n)$; rsvd
   $\sim\ell(m{+}n)$, ratio $\rho$).

The clean, library-free statement of the matched-accuracy speedup:
$$
\text{FLOP-speedup}\;\approx\;\frac{O(mn\min(m,n))}{O(mn\ell)}\;=\;\frac{\min(m,n)}{\ell}\;=\;\frac1\rho .
$$
**We do not discard wall-clock** — part of randomization's real value *is* that it maps onto
BLAS-3 / fewer passes / less communication (a genuine hardware benefit). We report it as a
*labeled, machine-dependent* secondary metric, with the slow `np.add.at` scatter replaced by
`scipy.sparse` so no method is artificially hobbled.

---

## 5. Results

### E1 — per-stage accuracy ablation (`exp01`, 3×3, bond 8, χ=16, η∈{2..7}, 20×4 seeds)

Metric: represented-state error $\varepsilon_{\text{state}}=1-|\langle\psi_0|\hat\psi\rangle|$
(3×3 contracts exactly). The $\eta$-truncation is lossy, so **all methods sit at the same floor**
$\approx0.34$. $\rho_1=1.000$ flat; $\rho_2=0.25\to0.875$.
- **Low-rank stages accuracy-neutral**: `RSVD2` median excess $\sim+1.4\times10^{-3}$, `sketch-Q`
  $\sim-2\times10^{-5}$. **Full-rank `svd1` is pure damage**: `RSVD1`/`all-rand` near-zero median
  but a $\pm14\%$ tail. No speedup at 3×3 (overhead-bound) — which is *why* exp02 asks the speed
  question on the isolated kernel.

![E1 — state error, paired excess over det, Moses-move runtime, and per-stage rank fraction (rho1 vs rho2) vs eta](figures/real_moses_move/exp01-stage-ablation.png)

### exp02 — matched-accuracy time-to-accuracy (the speed result)

"Time to same error" = **matched-accuracy speedup**: dial the randomized method down to just
match deterministic accuracy, then compare cost:
$$
S=\frac{\text{cost}_{\det}}{\displaystyle\min_{s,q\,:\,\varepsilon_{\text{rand}}\le(1+\text{tol})\varepsilon_{\det}}\text{cost}_{\text{rand}}(s,q)},\qquad \text{tol}=0.05 .
$$
We capture the *actual* svd2 matrix from a real deterministic move (real spectrum) and sweep the
source bond to drive $\rho_2\ll1$. Reported both ways — **algorithm work** (primary) and
**wall-clock** (secondary, this machine):

| bond | $\rho_2$ | FLOP-speedup (g / ss) | mem ratio (g / ss) | passes | wall-clock (g / ss) | excess |
|---|---|---|---|---|---|---|
| 12 | 0.333 | 0.7 / 0.8 | 1.6 / 2.4 | 6 | 1.4 / 0.6 | +4.2% |
| 16 | 0.250 | 1.1 / 1.1 | 2.2 / 2.3 | 4 | 2.0 / 0.8 | +4.1% |
| 24 | 0.167 | 2.9 / 3.3 | 2.7 / 3.5 | 2 | 4.3 / 1.7 | +4.4% |
| 32 | 0.125 | 6.2 / 8.1 | 4.9 / 6.3 | 2 | 9.2 / 2.9 | +4.5% |
| 48 | 0.083 | 9.6 / 12.6 | 7.3 / 9.4 | 2 | 21.0 / 7.3 | +2.9% |
| 64 | 0.062 | 13.0 / 14.7 | 9.8 / 10.9 | 2 | 32.0 / 9.2 | +2.2% |

As $\rho_2\to0.06$: the **FLOP-speedup** climbs to ~13–15× ($\approx1/\rho$, modulo the small SVD
of $B$), randomized reads $A$ in **2 passes**, and **peak memory** drops ~10×. (At high $\rho_2$,
bond 12–16, the matched config needs power iterations → FLOP-speedup $<1$ and 4–6 passes; the
win only appears once $\rho_2$ is small.) Excess stays within tolerance and shrinks.

![exp02 — FLOP-speedup, wall-clock speedup, peak-memory ratio, passes over A, svd2 rank fraction, and excess vs source bond](figures/real_moses_move/exp02-svd2-time-to-accuracy.png)

### Why `gaussian` and `sparsestack` differ — and why it isn't the algorithm

This is the headline of the fair-comparison exercise. **At the matched config the two methods
differ only in applying $\Omega$** (the QR, $B=Q^\dagger A$, and small SVD are identical):

- **FLOP-speedup (algorithm): they match** — `sparsestack` is in fact slightly *better*
  (14.7 vs 13.0 at bond 64), because a sparse sketch does $\sim\zeta mn$ MACs vs the dense
  $2mn\ell$. The cost model sees no meaningful difference.
- **Wall-clock (machine): gaussian wins ~3–4×** (32× vs 9× at bond 64) — purely because a dense
  BLAS-3 GEMM runs far above sparse-matmul throughput on this hardware. Replacing `np.add.at`
  with `scipy.sparse` already recovered ~6× of this; the residual is irreducible
  dense-vs-sparse-BLAS, not a property of the distribution.
- A minor secondary effect: a single CountSketch realization is a weaker embedding than Gaussian,
  so `sparsestack` occasionally needs a touch more oversampling to match accuracy.

**Conclusion:** the `gaussian` vs `sparsestack` wall-clock gap is an implementation/hardware
artifact, not an algorithmic one — at the work level they are equivalent (sparse marginally
better). The fair, citable headline is **randomized vs deterministic** (`gaussian`, dense BLAS,
vs `det`, dense BLAS): up to **31× wall-clock** and **~13× FLOPs / ~10× memory / 2 passes** at
$\rho_2=0.06$.

---

## 6. Honesty / scope

- exp02 is the **kernel** (svd2-stage) speedup, **not end-to-end**. The full move stays
  Amdahl-bounded by the deterministic zip-up absorption + quimb bookkeeping + the full-rank
  `svd1` ($\rho_1\approx1$). exp02 shows the stage E1 flagged as *safe* is also *cheap where it
  matters*; an end-to-end win needs randomized absorption (roadmap 0E) + larger lattices.
- Matched-accuracy is enforced on the **local** second-cut error, the faithful per-stage proxy
  for the state-error contribution.
- FLOP counts are a consistent *model* (Golub–Reinsch), used for ratios; they do not predict
  wall-clock (and are not meant to — that is the point of reporting both tiers).
- `khatri_rao` is excluded from timing (it materializes $\Omega$ — no matvec speedup yet).
- Deep-$\rho_2$ svd2 in a *single* move needs a large source bond; the block index $p>1$
  (excited states) would make $n_2=\eta p$ the lever instead.

---

## 7. Implementation notes (correctness + fairness fixes)

1. **Actual sketch width**: `SVDResult.ell` records the realized range-basis width; for
   `sparsestack` that is $\zeta\lceil(k+s)/\zeta\rceil\ge k+s$, so $\rho_2$ is now exact.
2. **`split_dim` → $(\chi,\eta)$**: the second cut is exactly the block-Moses
   $(\eta n_2)\times(\chi n_3)$. Validated: lossless overlap $=1.0000000000$; E1 numbers
   unchanged (the disentangler optimizes the full $\chi\eta$ bond, so the state is
   split-invariant — the fix only reshapes the internal matrix, which is what $\rho_2$ depends on).
3. **`scipy.sparse` sketch apply**: CountSketch/SparseStack now apply via a compiled sparse
   matmul instead of `np.add.at` (bit-identical result, ~10× faster) so the wall-clock tier is
   not hobbled by a slow primitive.
4. **Cost model**: `experiment_utils/cost_model.py` provides the implementation-free FLOPs /
   passes / peak-memory counts used as the primary cross-method metric.
