# briefing for the next agent

This repository is a synthetic randomized isoPEPS / block Moses Move research testbed. It was created from scratch in an initially empty repo that only contained two papers under `docs/`:

- `docs/isotns.pdf`: original isoTNS / Moses Move paper.
- `docs/computing-isotn.pdf`: block-isoPEPS excited-state paper by Alec Dektor, Runze Chi, Roel Van Beeumen, and Chao Yang.

The user is studying randomized methods to reduce computational costs in the Moses Move, especially by attacking the isometric tensor-ring decomposition subproblem and the residual-column absorption/compression subproblem. The implementation should remain small enough to run locally on an M2 MacBook Air, but rich enough to produce meaningful synthetic plots and advisor-facing evidence.

The repo has no git history. Treat `log.md` as the shared local research and implementation log. Update it when you plan, implement, debug, or run experiments.

## current repository state

Important files:

- `README.md`: commands and high-level usage.
- `log.md`: detailed implementation/research log. Keep this updated.
- `briefing.md`: this file.
- `rand_isopeps/tn_shapes.py`: grouped dimension bookkeeping for block Moses Move kernels.
- `rand_isopeps/randomized_svd.py`: deterministic SVD and randomized SVD backends. Currently supports `gaussian`, `rademacher`, and `countsketch`.
- `rand_isopeps/synthetic_tensors.py`: synthetic local tensor ensembles.
- `rand_isopeps/local_ring_decomp.py`: local two-SVD isometric tensor-ring decomposition skeleton.
- `rand_isopeps/column_moses.py`: product-column surrogate for local Moses Move error accumulation.
- `rand_isopeps/mpo_mps_absorb.py`: synthetic MPO-MPS absorption/compression baseline.
- `rand_isopeps/parallel.py`: coarse-grained parallel helper with process-pool to thread-pool fallback.
- `rand_isopeps/plotting.py`: matplotlib plotting helper (non-interactive `Agg` backend) that emits PDF figures. An earlier agent used a hand-rolled SVG writer because matplotlib's font cache hung in an older sandbox; that hang no longer reproduces here (cache builds in <0.3s), so the plotting path is now matplotlib with a restrained paper style. The `Panel`/`Series`/`PALETTE`/`MARKERS` API is unchanged; `write_line_panels` is the entry point (`write_line_panels_svg` remains as a thin alias).
- `experiments/exp1_local_first_vs_second_svd.py`: local first/second SVD randomization experiment.
- `experiments/exp2_column_error_accumulation.py`: column-height local error accumulation experiment.
- `experiments/exp3_r_column_absorption.py`: synthetic R-column absorption experiment.
- `experiments/exp4_tiny_full_isopeps_validation.py`: tiny explicit isometry validation.
- `outputs/data/`: CSV outputs from prior smoke and modest runs.
- `outputs/figures/`: PDF figure outputs from smoke and modest runs.

## verified commands

Smoke tests that passed:

```bash
python3 -m compileall rand_isopeps experiments
python3 experiments/exp1_local_first_vs_second_svd.py --quick
python3 experiments/exp2_column_error_accumulation.py --quick
python3 experiments/exp3_r_column_absorption.py --quick
python3 experiments/exp4_tiny_full_isopeps_validation.py --quick
```

Modest runs that passed:

```bash
python3 experiments/exp1_local_first_vs_second_svd.py --chi 4 --etas 4 6 8 --p 2 --trials 2 --workers 2 --sketch gaussian
python3 experiments/exp2_column_error_accumulation.py --chi 4 --eta 6 --lx-values 2 4 6 --trials 2 --workers 2 --sketch gaussian
python3 experiments/exp3_r_column_absorption.py --l-sites 6 --chi 4 --etas 4 6 8 --trials 2 --workers 2 --sketch gaussian
python3 experiments/exp1_local_first_vs_second_svd.py --chi 4 --etas 4 6 8 --p 2 --trials 2 --workers 2 --sketch countsketch
python3 experiments/exp3_r_column_absorption.py --l-sites 6 --chi 4 --etas 4 6 8 --trials 2 --workers 2 --sketch countsketch
```

Useful flags:

- `--sketch gaussian`: dense Gaussian randomized SVD correctness baseline.
- `--sketch rademacher`: dense sign/random phase baseline.
- `--sketch countsketch`: sparse sketch baseline.
- `--workers 0`: conservative local automatic worker count.
- `--workers 1 --blas-threads 0`: run one process/thread and let BLAS use its native thread defaults.
- `--blas-threads 1`: default, avoids oversubscription when multiple workers are used.

Output convention:

- CSV data goes to `outputs/data/`.
- PDF figures go to `outputs/figures/`.

## caveats from implementation

Plotting now uses matplotlib with the non-interactive `Agg` backend and writes PDF. The old warning that "matplotlib import hung during font-cache construction" no longer applies in this environment: importing matplotlib + pyplot and saving a PDF takes well under a second and the font cache at `~/.matplotlib` is already built. If a future sandbox does hang on the font cache again, the dependency-light SVG writer is recoverable from git history, but do not pre-emptively revert.

Process pools can fail in the managed sandbox because `concurrent.futures.process` calls a system semaphore query that raises `PermissionError`. `rand_isopeps.parallel.run_parallel` catches this and falls back to a thread pool. Keep that fallback.

The current absorption experiment is not SRC. It forms a synthetic MPO-MPS product and compresses the resulting product MPS by local SVD or local randomized SVD. This is useful as a baseline, but the next serious step is an SRC/Khatri-Rao-style method that avoids forming the inflated product bond.

The current column Moses Move experiment is a product-column surrogate. It measures accumulation of local approximation errors without fully modeling the recursive carrier tensor that gets absorbed upward. It is useful for first-pass stability plots, but a later agent may want to implement a more faithful sequential column carrier.

The local tensor-ring implementation intentionally skips the optional nonlinear disentangler. This is deliberate: the first phase isolates the two SVD insertion points.

## paper context: original isoTNS / Moses Move

The original isoTNS paper introduces a 2D canonical form for tensor networks. The key idea is to impose isometry constraints so that the exterior of an orthogonality hypersurface contracts to identity. This makes local contraction and local optimization cheaper, similarly to how MPS canonical forms make 1D algorithms efficient.

In 1D, an MPS can move its orthogonality center exactly by QR or SVD:

$$
\Lambda_l B_{l+1} = A_l \Lambda_{l+1}.
$$

In 2D, the analogous object is a whole column or row. A direct QR/SVD of an entire column is not useful because it destroys the locality needed to express the new orthogonality hypersurface as an MPS-like object. The Moses Move is the approximate, local, unzipping procedure that replaces exact 1D QR/SVD.

The original paper frames the column split as:

$$
\Lambda_l \approx A_l \Lambda,
$$

where $A_l$ is an isometric column and $\Lambda$ is a residual zero-column wavefunction with only ancilla degrees of freedom. This move can be tacked onto the neighboring column to shift the orthogonality hypersurface.

The local Moses Move subproblem groups the center tensor into a tripartite state:

$$
|ABC\rangle,
$$

then seeks a splitting isometry:

$$
a^\dagger : B \rightarrow B_L \otimes B_R,
$$

with

$$
a^\dagger a = I,
$$

so that the four-partite state

$$
|A B_L B_R C\rangle = a^\dagger |ABC\rangle
$$

has small entanglement across the desired vertical cut. The original paper uses an initial isometry and an optional unitary disentangler to reduce this entanglement, followed by SVD.

The key research insight from the original isoTNS paper is:

$$
\text{direct column QR/SVD is too global, so the Moses Move uses local unzipping.}
$$

For this repo, do not try to implement the full physical isoTNS algorithm first. We are isolating the synthetic kernels where randomized linear algebra can enter.

## paper context: block-isoPEPS excited states

The block-isoPEPS paper generalizes block MPS ideas to 2D. A block index $\alpha = 1,\dots,p$ is attached to the orthogonality center so one tensor network can represent multiple states. The block Moses Move shifts both:

- the orthogonality center,
- the block index.

The block Moses Move can be viewed as a column factorization:

$$
C_j \approx Q_j R_j,
$$

where:

- $Q_j$ is an isometric column,
- $R_j$ is a non-isometric residual column without physical indices,
- $R_j$ is absorbed into the neighboring column $C_{j+1}$.

This is the 2D analogue of MPS QR, except it is approximate because exact center motion in PEPS would require either increasing bond dimension or introducing error.

The paper identifies two main expensive subproblems:

1. Local isometric tensor-ring decomposition at each row of the active column.
2. Absorbing/compressing the residual column $R_j$ into the neighboring column.

## local tensor-ring decomposition: dimensions

At each row of the active column, the block Moses Move decomposes an active tensor

$$
A_{ij}
$$

into a small tensor ring with isometric constraints.

For an interior tensor, after grouping indices:

$$
B \in \mathbb{C}^{n_1 \times n_2 \times n_3},
$$

with block-Moses dimensions:

$$
n_1 = \chi \eta d,
$$

$$
n_2 = \eta p,
$$

$$
n_3 = \chi \eta.
$$

Here:

- $\chi$ is the horizontal bond dimension,
- $\eta$ is the vertical / column bond dimension,
- $d$ is the physical dimension,
- $p$ is the block size.

The two truncation ranks are:

$$
k_1 = \eta \chi,
$$

$$
k_2 = \eta.
$$

The local routine is:

$$
\text{first truncated SVD}
\rightarrow
\text{optional disentangler}
\rightarrow
\text{second truncated SVD}.
$$

The current implementation skips the optional disentangler.

## first local SVD

The first SVD separates the left grouped index from the up/right grouped indices:

$$
B_{n_1,(n_2 n_3)} \approx Q \widetilde{V}.
$$

The matrix shape is:

$$
B_{n_1,(n_2 n_3)}
\in
\mathbb{C}^{(\chi \eta d) \times (\chi \eta^2 p)}.
$$

The kept rank is:

$$
k_1 = \eta \chi.
$$

The deterministic dense SVD cost in the paper is:

$$
O(n_1^2 n_2 n_3).
$$

Substituting dimensions gives:

$$
O\left((\chi \eta d)^2(\eta p)(\chi \eta)\right)
=
O(\chi^3 \eta^4 d^2 p).
$$

This first SVD is a possible randomized target, but it may have limited asymptotic room because for spin systems $d=2$, so:

$$
n_1 = 2 \eta \chi \approx 2 k_1.
$$

That means the target rank is not dramatically smaller than the left dimension.

Research expectation:

$$
\text{first SVD randomization may give constant-factor or memory-movement benefits,}
$$

but it is probably less promising than the second SVD.

## second local SVD

After the first SVD, reshape:

$$
\widetilde{V}
\in
\mathbb{C}^{(\eta \chi) \times n_2 \times n_3}
$$

into:

$$
\widetilde{V}
\in
\mathbb{C}^{\eta \times \chi \times n_2 \times n_3}.
$$

The second SVD views it as:

$$
M \in \mathbb{C}^{(\eta n_2) \times (\chi n_3)}.
$$

Substituting block dimensions:

$$
\eta n_2 = \eta(\eta p) = \eta^2 p,
$$

$$
\chi n_3 = \chi(\chi \eta) = \chi^2 \eta.
$$

So:

$$
M
\in
\mathbb{C}^{(\eta^2 p) \times (\chi^2 \eta)}.
$$

It keeps rank:

$$
k_2 = \eta.
$$

The deterministic dense SVD dominant cost is:

$$
O\left((\eta n_2)^2 \chi n_3\right)
=
O\left((\eta^2 p)^2(\chi^2 \eta)\right)
=
O(\chi^2 \eta^5 p^2).
$$

This is the strongest local randomized-SVD target because $k_2=\eta$ can be much smaller than both dimensions of $M$.

An ideal dense randomized SVD would form:

$$
Y = M \Omega,
$$

where:

$$
\Omega \in \mathbb{C}^{(\chi^2 \eta) \times (k_2+s)}.
$$

If $k_2+s \sim \eta$, the leading dense multiply is:

$$
O((\eta^2 p)(\chi^2 \eta)(\eta))
=
O(\chi^2 \eta^4 p).
$$

Compared with:

$$
O(\chi^2 \eta^5 p^2),
$$

this suggests an idealized improvement by roughly:

$$
\eta p.
$$

The catch is that the output must preserve more than a generic low-rank matrix approximation. It must preserve approximate isometry and downstream tensor-network behavior.

Relevant diagnostics:

$$
\frac{\|B-\widehat{B}\|_F}{\|B\|_F},
$$

$$
\|Q^\dagger Q - I\|_F,
$$

$$
\frac{\|C_j - Q_j R_j\|_F}{\|C_j\|_F},
$$

and eventually:

$$
\left|
\frac{\langle T,H T\rangle}{\langle T,T\rangle}
-
\frac{\langle \widehat{T},H \widehat{T}\rangle}{\langle \widehat{T},\widehat{T}\rangle}
\right|.
$$

## local cost model

One local interior Moses Move step costs:

$$
O(\chi^3 \eta^4 d^2 p + \chi^2 \eta^5 p^2).
$$

Since the local tensor-ring decomposition is repeated $L_x$ times down/up a column:

$$
O\left(
L_x(\chi^3 \eta^4 d^2 p + \chi^2 \eta^5 p^2)
\right).
$$

The paper states that this local tensor-ring decomposition dominates one block sequential Moses Move step.

## residual column absorption

After the column split:

$$
C_j \approx Q_j R_j,
$$

the residual column must be contracted into the neighboring column:

$$
R_j C_{j+1} \rightarrow \widetilde{C}_{j+1}.
$$

This is essentially an MPO-MPS / MPO-MPO compression problem. If done naively, vertical bond dimensions multiply. A following SVD truncation costs:

$$
O(\chi^5 \eta^3 p).
$$

The paper says this can be reduced to:

$$
O(\chi^4 \eta^3 p)
$$

using MPO-MPS multiplication techniques such as:

- variational compression,
- zip-up,
- randomized SVD.

The paper's numerical experiments use zip-up, not randomized SVD.

Research expectation:

$$
\text{R-column absorption is the strongest structured-sketch target.}
$$

Reason: it is already naturally an MPO-MPS compressed product, and Camanho-Epperly-Tropp-style successive randomized compression (SRC) is designed for this setting.

Current implementation status: `rand_isopeps/mpo_mps_absorb.py` implements a synthetic MPO-MPS product and compresses the explicit product MPS using local deterministic SVD or randomized local SVD. This is a baseline only, not a true SRC implementation.

## full algorithm cost from the paper

The block-isoPEPS excited-state algorithm also has two-site gate updates. With reduced updates, each gate truncation costs:

$$
O(\chi^4 \eta^3 d p).
$$

The full split-exponential iteration scales as:

$$
O\left(
L_x^2(
\chi^4 \eta^3 d p
+
\chi^3 \eta^4 d^2 p
+
\chi^2 \eta^5 p^2
)
\right).
$$

For this repo, focus on Moses Move kernels first:

$$
\boxed{\text{local tensor-ring SVDs}}
$$

and:

$$
\boxed{R_j C_{j+1}\text{ absorption/compression}}.
$$

Do not implement the full excited-state algorithm unless explicitly asked.

## randomized SVD template

The baseline randomized low-rank algorithm is the Halko-Martinsson-Tropp randomized range finder:

$$
Y = A \Omega,
$$

$$
Y = \widehat{Q} R,
$$

$$
B_{\mathrm{small}} = \widehat{Q}^\dagger A,
$$

then compute a small SVD:

$$
B_{\mathrm{small}} = \widetilde{U} \Sigma V^\dagger,
$$

and map back:

$$
U = \widehat{Q}\widetilde{U}.
$$

The approximation is:

$$
A \approx U \Sigma V^\dagger.
$$

The hope is that if $A$ has effective numerical rank $k$, the cost behaves more like:

$$
\text{cost of applying } A \text{ and } A^\dagger
\text{ to } k+s \text{ test vectors}
$$

rather than a full dense SVD. Here $s$ is oversampling.

Power iterations are:

$$
Y = (A A^\dagger)^q A \Omega,
$$

or implemented stably by alternating orthonormalization. They help when singular values decay slowly, but add passes over $A$ and $A^\dagger$.

## sketch selection principles

The user provided the following NLA decision rule:

$$
\boxed{
\text{choose the sketch by matching the guarantee you need to the cost model of your operator.}
}
$$

Gaussian sketches are the theoretical and scientific control. They are robust and universal, but dense Gaussian sketch application may cost as much as the original problem or destroy tensor structure.

The project strategy should be:

$$
\boxed{
\text{Start Gaussian to establish correctness, then move to structured sketches that respect tensor-network contraction.}
}
$$

The core question is:

$$
\boxed{
\text{What is the cheapest } \Omega
\text{ such that } A\Omega
\text{ captures the rank-}k\text{ subspace I need?}
}
$$

There are three NLA axes:

1. Required guarantee.
2. Cost of applying the sketch.
3. Structure of the input/operator.

For local SVDs, the desired guarantee is a low-rank approximation guarantee:

$$
\|A - Q Q^\dagger A\|
\approx
\|A - A_k\|.
$$

For least-squares, Rayleigh quotient, or observable work, one might need subspace embeddings:

$$
\|Sx\|_2^2 \approx \|x\|_2^2
\quad
\text{for all } x \in \mathcal{U}.
$$

For this project:

$$
\text{local SVDs: low-rank approximation error},
$$

$$
R\text{-column absorption: compressed MPO-MPS product error},
$$

$$
\text{Rayleigh quotient: inner-product / quadratic-form preservation}.
$$

## sketch families and relevance

Gaussian:

$$
\Omega_{ij} \sim \mathcal{N}(0,1)
$$

or complex Gaussian. Best correctness baseline. Dense, likely not final.

Rademacher:

$$
\Omega_{ij} \in \{-1,+1\}
$$

or complex sign. Dense but cheaper to sample and sometimes a useful baseline.

CountSketch:

Each input coordinate maps to one random bucket with one random sign. It can support input-sparsity-time sketching. It is noisier than Gaussian for fixed sketch size, so it may require larger $s$.

TensorSketch:

Designed for tensor-product features. Useful when an index is a product of tensor legs and one wants to avoid materializing the full tensor-product sketch.

Kronecker sketch:

$$
\Omega = \Omega_1 \otimes \Omega_2.
$$

Khatri-Rao sketch:

$$
\Omega = \Omega_1 \odot \Omega_2,
$$

where columns are columnwise Kronecker products.

TT/TN sketch:

Potentially most natural for tensor-network operators, but more specialized and not the first implementation step.

Recommended ordering for local kernels:

$$
\boxed{
\text{Gaussian rSVD}
\rightarrow
\text{CountSketch}
\rightarrow
\text{Khatri-Rao/TensorSketch}
\rightarrow
\text{TT/TN sketches}
}
$$

For R-column absorption, jump sooner to SRC/Khatri-Rao thinking:

$$
\boxed{
\text{MPO-MPS product compression should be structured from the beginning.}
}
$$

## target-specific sketch choices

### target 1: first local SVD

The first SVD sees:

$$
B_{n_1,(n_2 n_3)}
$$

with:

$$
k_1 = \eta \chi.
$$

For spin systems:

$$
n_1 = \chi \eta d = 2\chi\eta = 2k_1.
$$

Decision:

$$
\boxed{
\text{Use Gaussian rSVD first; only try structured sketches if spectra decay and timing suggests a gain.}
}
$$

This is not the highest-priority target.

### target 2: second local SVD

The second SVD sees:

$$
M \in \mathbb{C}^{(\eta^2 p) \times (\chi^2 \eta)}
$$

and keeps:

$$
k_2 = \eta.
$$

Decision:

$$
\boxed{
\text{Start Gaussian rSVD, then try CountSketch and product/Khatri-Rao/TensorSketch.}
}
$$

This is the strongest local SVD target.

The right index has product structure:

$$
\chi n_3 = \chi(\chi \eta),
$$

and the left index also has product structure:

$$
\eta n_2 = \eta(\eta p).
$$

A dense sketch over the flattened index ignores this structure. Product/Khatri-Rao/TensorSketch should exploit the grouped tensor-leg factorization.

### target 3: R-column absorption

The operation is morally:

$$
\text{MPO} \times \text{MPS} \rightarrow \text{compressed MPS}.
$$

Decision:

$$
\boxed{
\text{Use SRC/Khatri-Rao-style sketches here first.}
}
$$

The current code does not yet implement SRC. That is a high-value next task.

## synthetic tensor ensembles

Do not use only Gaussian random tensors. Gaussian random matrices often have flat-ish spectra and can make randomized SVD look artificially bad.

The intended ensembles are:

### ensemble A: exactly low-rank tensor-ring data

Generate factors so the tensor exactly matches the two-SVD skeleton. This checks implementation correctness.

Expected deterministic error:

$$
\frac{\|B-\widehat{B}\|_F}{\|B\|_F}
\approx
10^{-12}
$$

when no truncation is needed.

Implementation: `synthetic_tensors.make_local_ring_exact`.

### ensemble B: low-rank plus noise

Use:

$$
B = B_{\mathrm{ring}} + \sigma G,
$$

where $G$ is Gaussian noise and:

$$
\sigma \in \{10^{-6},10^{-4},10^{-2},10^{-1}\}.
$$

This mimics a tensor approximately representable by an isometric tensor ring.

Implementation: `synthetic_tensors.make_local_tensor(..., ensemble="noisy_ring")`.

### ensemble C: controlled singular decay

Construct a matrix with singular values:

$$
s_j \sim \exp(-j/\xi),
$$

or:

$$
s_j \sim j^{-\alpha}.
$$

This tests when randomized SVD succeeds or fails. Fast-decaying spectra should be easy; slow power-law spectra are harder.

Implementation: `synthetic_tensors.make_controlled_spectrum_matrix`.

This is not yet wired into a full experiment script. A good next experiment is a standalone matrix/sketch diagnostic using this function.

## desired metrics

Local decomposition:

$$
\epsilon_B =
\frac{\|B-\widehat{B}\|_F}{\|B\|_F}.
$$

First isometry defect:

$$
\epsilon_{\mathrm{iso},1}
=
\|Q_1^\dagger Q_1 - I\|_F.
$$

Second isometry defect, currently measured as row-isometry defect of the second right factor:

$$
\epsilon_{\mathrm{iso},2}
=
\|V_2 V_2^\dagger - I\|_F.
$$

Column surrogate:

$$
\epsilon_{\mathrm{MM}}
=
\frac{\|C-\widehat{C}\|_F}{\|C\|_F}.
$$

Absorption:

$$
\epsilon_{\mathrm{absorb}}
=
\frac{\|\widetilde{C}_{\mathrm{exact}}-\widetilde{C}_{\mathrm{approx}}\|_F}
{\|\widetilde{C}_{\mathrm{exact}}\|_F}.
$$

Future energy diagnostic:

$$
\Delta E
=
\left|
\frac{\langle T,H T\rangle}{\langle T,T\rangle}
-
\frac{\langle \widehat{T},H \widehat{T}\rangle}{\langle \widehat{T},\widehat{T}\rangle}
\right|.
$$

Posterior diagnostics should eventually include independent sketched residual estimates:

$$
\|S(A-\widehat{A})\|_F
\approx
\|A-\widehat{A}\|_F,
$$

for an independently drawn test sketch $S$.

## implementation details: local two-SVD skeleton

`local_ring_decomp.local_ring_decomp` does this:

1. Reshape:

$$
B \mapsto B_{n_1,(n_2 n_3)}.
$$

2. Compute first truncated SVD:

$$
B_{n_1,(n_2 n_3)} \approx U_1 \Sigma_1 V_1^\dagger.
$$

3. Absorb singular values:

$$
\widetilde{V}_1 = \Sigma_1 V_1^\dagger.
$$

4. Reshape:

$$
\widetilde{V}_1
\rightarrow
\widetilde{V}_{\eta,\chi,n_2,n_3}.
$$

5. Permute and flatten:

$$
M =
\widetilde{V}_{(\eta n_2),(\chi n_3)}.
$$

6. Compute second truncated SVD:

$$
M \approx U_2 \Sigma_2 V_2^\dagger.
$$

7. Reconstruct the approximated tensor by inverting reshapes and multiplying back by $U_1$.

Mode flags:

- `det`: deterministic SVD for both.
- `rand_first`: randomized first SVD, deterministic second SVD.
- `rand_second`: deterministic first SVD, randomized second SVD.
- `rand_both`: randomized SVD for both.

## implementation details: randomized SVD

`randomized_svd.rsvd_truncate` uses:

$$
Y = A\Omega,
$$

then orthonormalizes $Y$, applies optional power iterations, forms:

$$
B = Q^\dagger A,
$$

and SVDs the small matrix.

For `countsketch`, the current implementation does not explicitly form $\Omega$. It draws bucket indices and signs, then accumulates:

$$
Y[:,h(j)] \mathrel{+}= s(j) A[:,j].
$$

This is correct for explicit dense matrices, but it is not yet a matrix-free tensor-network CountSketch. A future agent can adapt this idea to avoid explicit unfolding/materialization.

## implementation details: column surrogate

`column_moses.run_column_moses_surrogate` generates $L_x$ independent local tensors and decomposes each one. It then estimates a product-column relative error without materializing a huge tensor product.

If:

$$
C = \bigotimes_{i=1}^{L_x} B_i,
$$

and:

$$
\widehat{C} = \bigotimes_{i=1}^{L_x} \widehat{B}_i,
$$

then:

$$
\|C\|^2 = \prod_i \|B_i\|^2,
$$

$$
\|\widehat{C}\|^2 = \prod_i \|\widehat{B}_i\|^2,
$$

$$
\langle C,\widehat{C}\rangle = \prod_i \langle B_i,\widehat{B}_i\rangle.
$$

Therefore:

$$
\frac{\|C-\widehat{C}\|}{\|C\|}
=
\sqrt{
\frac{
\|C\|^2 + \|\widehat{C}\|^2 - 2\operatorname{Re}\langle C,\widehat{C}\rangle
}
{\|C\|^2}
}.
$$

This is only a surrogate, not the actual recursive block sequential Moses Move with upward carrier absorption.

## implementation details: absorption baseline

`mpo_mps_absorb.py` creates:

- a random MPS $C$,
- a random mostly-identity MPO $R$,
- their explicit product MPS $R C$ with inflated product bonds.

Then it compresses the product MPS by sweeping left to right and truncating local matrices by either:

- deterministic SVD: `zipup_svd`,
- randomized SVD: `randomized`.

The vector-level exact product is obtained by explicitly contracting the MPS into a vector for small synthetic systems. The error is:

$$
\frac{\| (R C)_{\mathrm{exact}} - (R C)_{\mathrm{compressed}}\|_2}
{\| (R C)_{\mathrm{exact}}\|_2}.
$$

This is deliberately small-scale.

## plot style

The user asked for inspiration from Alec Dektor's paper. The current matplotlib/PDF style is:

- white background, top/right spines removed,
- compact small-multiple panels in one row,
- log axes for error quantities, faint major/minor grid,
- thin lines with white-edged markers,
- restrained Okabe-Ito colorblind-safe palette,
- a single shared legend below the panels.

Do not make decorative dashboards. The figures should look like compact research plots.

## sample results already logged

See `log.md` for exact output filenames. Summary:

For `chi=4`, `eta in {4,6,8}`, `p=2`, `trials=2`:

- Gaussian local experiment: all reconstruction errors near $10^{-4}$, matching injected noise.
- CountSketch local experiment: all reconstruction errors near $10^{-4}$; on these tiny cases CountSketch is competitive.
- Absorption Gaussian randomized local compression has similar error to zip-up-style SVD but is slower on tiny cases.
- Absorption CountSketch randomized local compression has similar error and closer runtime, but still is not SRC.

Do not overinterpret these timings. The cases are tiny and mainly validate the pipeline.

## recommended next tasks

### task 1: structured product sketch for second local SVD

Implement a product-structured sketch for the second SVD matrix:

$$
M \in \mathbb{C}^{(\eta n_2) \times (\chi n_3)}.
$$

Its right dimension is product-structured:

$$
\chi n_3 = \chi(\chi\eta).
$$

A simple first version:

$$
\Omega = \Omega_1 \otimes \Omega_2,
$$

where:

$$
\Omega_1 \in \mathbb{C}^{\chi \times r_1},
$$

$$
\Omega_2 \in \mathbb{C}^{n_3 \times r_2},
$$

and $r_1 r_2 \approx k+s$.

But be careful: a full Kronecker product may produce awkward sketch sizes. A Khatri-Rao sketch may be more natural:

$$
\Omega = \Omega_1 \odot \Omega_2,
$$

with:

$$
\Omega[:,\ell] = \Omega_1[:,\ell] \otimes \Omega_2[:,\ell].
$$

Then:

$$
Y = M\Omega
$$

can be computed through tensor contractions without flattening the whole right index in future versions.

For the current explicit matrix implementation, it is acceptable to materialize the product sketch first to compare accuracy and timing. Then optimize.

### task 2: add experiment comparing sketch families

Add a new experiment or extend exp1 so it can sweep:

$$
\text{sketch} \in \{\text{gaussian}, \text{rademacher}, \text{countsketch}, \text{khatri-rao}\}.
$$

Recommended fixed mode for first plot:

$$
\text{mode} = \text{rand_second}.
$$

Reason: second SVD is the main local target.

Plot:

- runtime vs $\eta$,
- relative reconstruction error vs $\eta$,
- error vs oversampling $s$,
- error vs power iterations $q$.

### task 3: controlled singular spectrum experiment

Use:

$$
s_j \sim \exp(-j/\xi)
$$

and:

$$
s_j \sim j^{-\alpha}.
$$

Compare Gaussian vs CountSketch vs product sketch at fixed rank:

$$
k=\eta.
$$

This will separate "randomization fails because the spectrum is hard" from "randomization fails because the tensor-ring structure is sensitive."

### task 4: closer SRC-style absorption

The current absorption baseline forms the product MPS with inflated bond dimension:

$$
D_{\mathrm{product}} = D_{\mathrm{MPO}} D_{\mathrm{MPS}}.
$$

SRC should avoid forming this full intermediate and discover compressed spaces directly through sketches.

A rough target:

$$
Y = (R C)\Omega
$$

applied site-by-site/successively using MPO-MPS contractions, not by flattening the global product.

The user's research narrative is strongest here because the paper explicitly lists randomized SVD as an option for this step but uses zip-up in experiments.

### task 5: more faithful sequential column Moses Move

Current column surrogate treats locals independently. A more faithful model would propagate a carrier/residual upward:

$$
A_{L_x,j} \rightarrow (Q_{L_x,j}, U_{L_x,j}, R_{L_x,j}),
$$

then absorb:

$$
U_{L_x,j}
$$

into the tensor above, and continue:

$$
A_{L_x-1,j} \rightarrow (Q_{L_x-1,j}, U_{L_x-1,j}, R_{L_x-1,j}).
$$

This would better test accumulated error in the actual block sequential Moses Move.

## research framing for advisor

Do not frame this as "which sketch is fastest?"

Frame it as:

$$
\boxed{
\text{Which sketch preserves the relevant low-rank/isometric subspace at the lowest contraction cost?}
}
$$

For Moses Move, "relevant" means:

$$
\|B-\widehat{B}\|_F \text{ small},
$$

$$
\|Q^\dagger Q-I\|_F \text{ small},
$$

$$
\|C - Q R\|_F \text{ small},
$$

and eventually:

$$
|\widehat{E}-E| \text{ small}.
$$

The clean thesis is:

$$
\boxed{
\text{Structured randomized low-rank approximation can replace selected deterministic SVD/compression steps inside the Moses Move without destroying approximate isometry or energy-relevant accuracy.}
}
$$

The staged roadmap:

1. Replace zip-up-style R-column absorption with SRC/randomized compression.
2. Replace the second local tensor-ring SVD with structured randomized SVD.
3. Try the first local tensor-ring SVD only if timing/spectrum diagnostics make it worthwhile.
4. Consider sketched disentangler optimization later.

## important implementation norms for the next agent

Use `rg` and inspect existing modules before editing.

Use `apply_patch` for source edits.

Keep `log.md` updated with:

- plan,
- implementation notes,
- bug notes,
- commands run,
- output filenames,
- result summaries.

Prefer small, testable changes. Run `compileall` and at least one quick experiment after changes.

Keep experiments laptop-scale:

$$
\chi \leq 6,
$$

$$
\eta \leq 10,
$$

$$
p \leq 3,
$$

$$
L_x \leq 10.
$$

Larger sweeps should be opt-in.

## changelog: 2026-06-08 plotting overhaul

Switched figures from hand-rolled SVG to matplotlib-generated PDF.

- `rand_isopeps/plotting.py` now uses matplotlib (forced `Agg` backend) and writes vector PDF. The old font-cache hang no longer reproduces (verified <0.3s).
- Restyled to a compact paper look: white background, no top/right spines, faint grid, thin lines with white-edged markers, Okabe-Ito palette, `constrained` layout, single shared legend below the panels.
- API unchanged (`Panel`/`Series`/`PALETTE`/`MARKERS`); entry point is `write_line_panels`, with `write_line_panels_svg` kept as an alias. exp1/exp2/exp3 now emit `.pdf`; exp4 emits no figure. Stale `.svg` outputs removed.

