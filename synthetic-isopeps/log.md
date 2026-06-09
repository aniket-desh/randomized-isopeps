# research implementation log

## 2026-06-08 15:46:59 pdt

### paper notes

- read `docs/isotns.pdf`, the original isometric tensor network paper. the moses move shifts a 2d orthogonality hypersurface by splitting a column into an isometric column and a residual zero-column. direct qr/svd of a full column is not appropriate because it destroys locality, so the practical algorithm uses local unzipping and an optional disentangler.
- read `docs/computing-isotn.pdf`, the block-isopeps excited-state paper by alec dektor, runze chi, roel van beeumen, and chao yang. the block sequential moses move recursively decomposes local active tensors into isometric tensor rings, absorbs the local residual upward, and then absorbs the resulting r-column into the neighboring column.
- confirmed the grouped local dimensions for the synthetic kernel:
  - `n1 = chi * eta * d`
  - `n2 = eta * p`
  - `n3 = chi * eta`
  - first truncation rank `k1 = chi * eta`
  - second truncation rank `k2 = eta`
- confirmed the paper cost model:
  - first local svd: `o(n1^2 n2 n3) = o(chi^3 eta^4 d^2 p)`
  - second/final stage dominant term: `o((eta n2)^2 chi n3) = o(chi^2 eta^5 p^2)`
  - per-column local split: `o(lx (chi^3 eta^4 d^2 p + chi^2 eta^5 p^2))`
  - r-column absorption naive truncation: `o(chi^5 eta^3 p)`, reducible to `o(chi^4 eta^3 p)` by mpo-mps techniques such as variational compression, zip-up, or randomized svd.
- plotting style note from the dektor paper: compact white-background panels, log error axes, small multiples by parameter group, restrained line and marker styling.

### implementation plan

- create a small package named `rand_isopeps`.
- implement deterministic svd and randomized svd with oversampling and power iterations.
- implement local two-svd tensor-ring skeleton with modes:
  - `det`
  - `rand_first`
  - `rand_second`
  - `rand_both`
- implement exact ring-compatible and noisy ring-compatible synthetic tensors.
- implement a column accumulation surrogate to measure how local reconstruction errors compound with `lx`.
- implement r-column absorption as a synthetic mpo-mps product followed by deterministic or randomized local compression.
- write experiment scripts that emit csv files and figures under `outputs/`.

### implementation notes

- initialized package and experiment scaffold.
- the first version intentionally skips the nonlinear disentangler and full isopeps evolution. this keeps the randomized low-rank insertion points isolated.
- plotting now uses direct svg generation instead of matplotlib because matplotlib font-cache construction hung in the sandbox.

### open follow-ups

- add a true structured sketch, for example tensor-product rademacher or countsketch, after the dense randomized svd baselines are working.
- replace the absorption experiment's randomized local compression with a closer successive randomized compression implementation.
- add a real sequential carrier model for column moses move if the product-column surrogate is too weak for advisor discussion.

## 2026-06-08 15:52:00 pdt

### optimization notes

- added coarse-grained parallelism to the experiment scripts using `processpoolexecutor`.
- each independent parameter/trial case is a separate task:
  - exp1: one task per `(eta, trial)`, running all four local modes on the same synthetic tensor.
  - exp2: one task per `(lx, trial, mode)`.
  - exp3: one task per `(eta, trial)`, running deterministic and randomized absorption on the same synthetic product.
- added `--workers` and `--blas-threads` flags. default `--workers 0` selects a conservative local worker count and default `--blas-threads 1` avoids blas oversubscription inside worker processes. use `--workers 1 --blas-threads 0` for one large case where native blas threading should dominate.
- switched dense svd calls to `scipy.linalg.svd(..., check_finite=false, lapack_driver="gesdd")`.
- vectorized identity insertion in synthetic mpo generation.
- fixed the mpo-mps einsum subscript to remove whitespace and keep the contraction explicit.

### bug notes

- first quick run found that launching scripts as `python3 experiments/foo.py` did not put the repository root on `sys.path`. patched all experiment scripts to insert the repo root based on `__file__`.

## 2026-06-08 16:02:00 pdt

### sketch selection notes

- logged sketch-method guidance from user. core decision rule: choose the sketch by matching the needed guarantee to the cost model of the operator.
- gaussian sketching is the scientific control, not the final algorithm. use it first to establish that randomized low-rank approximation works for the kernel at all.
- after gaussian baselines, the order should be:
  - countsketch / sparse embeddings for cheap sparse sketch application.
  - product-structured sketches such as kronecker, khatri-rao, and tensorsketch for grouped tensor-leg indices.
  - src / tt-style methods for tensor-network objects once the baseline behavior is understood.
- first local svd is lower priority for structured sketching because `k1 = eta * chi` is close to `n1 = chi * eta * d` when `d = 2`.
- second local svd is the best local randomized-svd target because `k2 = eta` can be much smaller than the matrix dimensions `(eta * n2) x (chi * n3)`.
- r-column absorption is the strongest structured-sketch target. it should be treated as mpo-mps compression, where src/khatri-rao style sketches are more principled than flattening and applying a dense gaussian sketch.
- posterior diagnostics should be included in future experiments: exact residuals for small synthetic matrices, independent sketched residual estimates for larger matrix-free/tensor-network cases, and downstream quantities such as local reconstruction error, isometry defect, column surrogate error, and eventually rayleigh quotient drift.

### implementation implications

- keep `gaussian` as the default randomized svd sketch.
- add `countsketch` and product-structured sketches as explicit options instead of replacing gaussian.
- tag experiment outputs by sketch family so plots compare deterministic, gaussian rsvd, countsketch rsvd, and product-structured rsvd cleanly.
- for absorption, move from the current local randomized svd baseline toward an src-style implementation that avoids forming the full mpo-mps product bond.

### plotting bug notes

- matplotlib import hung during font-cache construction in this sandbox. replaced the plotting helper with direct svg generation to avoid font-cache and gui/backend issues.
- experiment figures now write `.svg` files.

### parallel bug notes

- process pools can fail in the managed sandbox because `concurrent.futures.process` calls `os.sysconf("sc_sem_nsems_max")`, which raises `permissionerror`. patched `run_parallel` to fall back to a thread pool when process pools are unavailable.

### smoke test results

- `python3 -m compileall rand_isopeps experiments` passed.
- quick experiment scripts passed:
  - `experiments/exp1_local_first_vs_second_svd.py --quick`
  - `experiments/exp2_column_error_accumulation.py --quick`
  - `experiments/exp3_r_column_absorption.py --quick`
  - `experiments/exp4_tiny_full_isopeps_validation.py --quick`
- countsketch quick checks passed for exp1 and exp3.
- modest gaussian runs completed with `--workers 2` after the thread-pool fallback:
  - exp1 output: `outputs/data/exp1-local-20260608-160707.csv`, `outputs/figures/exp1-local-20260608-160707.svg`
  - exp2 output: `outputs/data/exp2-column-20260608-160707.csv`, `outputs/figures/exp2-column-20260608-160707.svg`
  - exp3 output: `outputs/data/exp3-absorption-20260608-160706.csv`, `outputs/figures/exp3-absorption-20260608-160706.svg`
- modest countsketch runs completed:
  - exp1 output: `outputs/data/exp1-local-20260608-160718.csv`, `outputs/figures/exp1-local-20260608-160718.svg`
  - exp3 output: `outputs/data/exp3-absorption-20260608-160718.csv`, `outputs/figures/exp3-absorption-20260608-160718.svg`
- sample exp1 means over `chi=4`, `eta in {4,6,8}`, `p=2`, `trials=2`:
  - gaussian: deterministic mean time `0.015609s`, randomized-first `0.046302s`, randomized-second `0.029192s`, randomized-both `0.017710s`; all mean reconstruction errors were about `1e-4`, matching the injected noise scale.
  - countsketch: deterministic mean time `0.006398s`, randomized-first `0.008460s`, randomized-second `0.005811s`, randomized-both `0.005715s`; all mean reconstruction errors were about `1e-4`.
- sample exp3 means over `l_sites=6`, `chi=4`, `eta in {4,6,8}`, `trials=2`:
  - gaussian randomized local compression had similar error to zip-up-style svd (`2.54e-2`) but was slower on these tiny cases.
  - countsketch randomized local compression had similar error (`2.62e-2`) and was closer in runtime, but this is still a local rsvd baseline, not src.

## 2026-06-08 16:15:00 pdt

### handoff notes

- added `briefing.md` as a detailed handoff document for another coding agent.
- it includes the paper context, block moses move mathematics, local svd cost models, sketch-selection guidance, implementation state, verified commands, caveats, and recommended next tasks.

## 2026-06-08 16:27:00 pdt

### plotting: svg -> pdf + matplotlib restyle

- re-checked the old "matplotlib font-cache hang" caveat. it no longer reproduces: `import matplotlib` + pyplot + save a pdf runs in <0.3s and the font cache at `~/.matplotlib` is already built. matplotlib 3.7.2, numpy 1.26.4, scipy 1.11.1.
- rewrote `rand_isopeps/plotting.py` to use matplotlib (forced `Agg` backend, no gui) and emit vector pdf instead of the hand-rolled svg writer.
- kept the public api stable: `Panel`, `Series`, `PALETTE`, `MARKERS` unchanged. new entry point `write_line_panels`; `write_line_panels_svg` retained as a thin alias so nothing breaks. `MARKERS` now holds matplotlib marker codes, with aliases for the old `circle/square/triangle/diamond` names.
- restyled for a compact research-paper look (dektor-inspired): white background, top/right spines removed, faint major/minor grid, thin lines with white-edged markers, okabe-ito colorblind-safe palette, `constrained` layout, and a single shared legend placed `outside lower center` so it never overlaps the panels. fixed the earlier ylabel/title collision and the legend gap by moving from `tight_layout(rect=...)` to constrained layout.
- updated the three plotting experiments (exp1/exp2/exp3) to import `write_line_panels` and write `.pdf` paths. exp4 emits no figure, unchanged.
- removed the 10 stale `.svg` files under `outputs/figures/`; that directory is now pdf-only.
- updated `README.md` and `briefing.md` (file description, output convention, plotting caveat, plot-style section) to reflect matplotlib + pdf.

### commands run

- `python3 -m compileall -q rand_isopeps experiments` passed.
- quick runs passed and wrote pdf:
  - `experiments/exp1_local_first_vs_second_svd.py --quick`
  - `experiments/exp2_column_error_accumulation.py --quick`
  - `experiments/exp3_r_column_absorption.py --quick`
- modest runs passed (`--workers 2`):
  - exp1: `--chi 4 --etas 4 6 8 10 --p 2 --trials 3`
  - exp2: `--chi 4 --eta 8 --lx-values 2 4 6 8 10 --trials 3`
  - exp3: `--chi 4 --etas 4 6 8 10 --l-sites 8 --trials 3`
- visually inspected rasterized previews of all three figures; layout is clean, no clipping or overlap.

## 2026-06-09 09:40:00 pdt

### plotting: fix clipped labels / cramped layout

- the matplotlib/pdf figures were rendering too tightly: long rotated y-axis labels ("relative reconstruction error", "second isometry defect") were clipped at the top, left-aligned titles ran wide, and a large dead band of whitespace sat between the panels and the shared legend. root cause was short axes (panels were 300px tall) so the rotated y-labels overflowed the axis box.
- `rand_isopeps/plotting.py` changes:
  - default panel height `300 -> 380`; gives the rotated y-labels enough vertical room to fit inside the axes instead of overflowing.
  - constrained-layout padding loosened: `set(w_pad=0.16, h_pad=0.10, wspace=0.10)` for clear gaps between panels and breathing room at the edges.
  - `savefig.pad_inches 0.03 -> 0.12` for a little outer margin.
  - added `labelpad` to x/y labels (4/5) and bumped `ax.margins` to `x=0.08, y=0.10` so data does not touch the spines.
- experiment call-site dimensions bumped for "fuller" figures: exp1 (3 panels) `980x300 -> 1140x420`; exp2/exp3 (2 panels) `720x300 -> 820x420`. public api unchanged.
- regenerated all three figures from fresh modest runs and visually verified rasterized previews: y-labels fully render, titles fit, even inter-panel whitespace, legend sits directly under the panels.
  - exp1: `outputs/figures/exp1-local-20260609-094036.pdf`
  - exp2: `outputs/figures/exp2-column-20260609-094041.pdf`
  - exp3: `outputs/figures/exp3-absorption-20260609-094044.pdf`
- the older `20260608-*` figure pdfs are the superseded (messy) renderings; left in place since they are regenerable outputs.

## 2026-06-09 09:44:00 pdt

### plotting: clean up y-tick formatting + remove stale figures

- removed the superseded `20260608-*` figure pdfs from `outputs/figures/`; only the clean regenerated figures remain.
- fixed awkward y-axis tick labels on "log" error panels. the error data here spans less than one decade (e.g. exp1 reconstruction error ~9.6e-5..1.05e-4, isometry defect ~1.2e-15..3.6e-15, exp3 error ~2.88e-2..3.2e-2), so a log axis just labelled minor ticks with wide, mixed-exponent strings like `1.04 x 10^-4` next to `9.6 x 10^-5`.
- `rand_isopeps/plotting.py`:
  - added `_log_span_decades(panel)` helper measuring the base-10 span of a panel's positive y-values.
  - a panel declared `yscale="log"` now only renders on a true log axis when it spans >= 1 decade; otherwise it falls back to a linear axis with `MaxNLocator(nbins=5)` and a `ScalarFormatter(useMathText=True, powerlimits=(-2,3))` so the panel shows clean evenly-spaced ticks (e.g. `0.96 0.98 1.00 1.02 1.04`) under a single shared `x10^n` multiplier.
  - linear panels (timing/runtime) also pick up the shared-multiplier formatter, so e.g. `0.00 0.01 0.02 0.03` becomes `0 1 2 3 x10^-2`.
- regenerated all three figures and visually verified: each panel carries one shared multiplier, consistent tick values, no clipping, no mixed exponents.
  - exp1: `outputs/figures/exp1-local-20260609-094405.pdf`
  - exp2: `outputs/figures/exp2-column-20260609-094410.pdf`
  - exp3: `outputs/figures/exp3-absorption-20260609-094412.pdf`
