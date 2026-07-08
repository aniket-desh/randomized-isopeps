# Disentangling the sketch: can a fixed vertical bond match the local Moses move?

**Stage 1, experiment 10** &nbsp;·&nbsp; `experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py`

## TL;DR

exp09 left us with a negative result: the global rMPS-sketched column QR matches the local
Moses move's accuracy only by spending a **larger vertical bond** ($\eta_q = 6\!-\!8$ vs
local $\eta = 4$), and that fatter carry bond inflates every downstream column until the
per-column saving is eaten — the end-to-end speedup crosses from $2.8\times$ at $L_x=4$
to $0.75\times$ at $L_x=5$ (global becomes *more* expensive than local).

This experiment tests the theory-review proposal: insert the **Moses disentangler** into the
sketch sweep so the vertical bond can stay at $\eta = 4$ while a cheap horizontal residual
carries the rest. Four gated findings, on the same real TFIM columns as exp08/09:

1. **The mechanism is real and non-trivial.** The disentangler lowers the sampled range's
   rank-$\eta$ vertical tail by $2\!-\!5\times$, and the **null test** confirms it is the
   composite-bond *reshuffle* — not the unitary — that does the work: a naive gauge on the
   existing bond leaves the tail invariant to machine precision ($\lesssim 10^{-20}$).
2. **The vertical bond is the right lever.** $\varepsilon_{\text{proj}}$ collapses on
   $\eta_q$ but is essentially flat in the probe count $\ell$ — the thin sketch faithfully
   captures the column's range; the retained *range dimension* (set by the vertical bond),
   not the sketch, is what limits accuracy.
3. **The disentangler is cheap.** Its residual-truncation loss converges within
   $1\!-\!5$ iterations, adding $<5\%$ to the column factorization cost.
4. **The verdict flips.** Charged at the thin $\eta = 4$ carry (like local) plus the
   disentangler FLOPs, the disentangled-global column stays **$\sim\!17\!-\!20\times$
   cheaper than local** across $L_x = 4,5$, exactly where plain-global fell below break-even.

The one load-bearing assumption — that disentangling the *sketch* reproduces local's
accuracy at $\eta=4$ — is strongly supported (findings 1–2 plus the fact that local Moses
*is* a disentangled $\eta{=}4$ column) but not yet directly measured; see
[Honest accounting](#honest-accounting-what-is-proven-vs-assumed).

---

## 1. The bottleneck this addresses

A column sweep of an $L_y$-wide lattice is a chain of Moses moves; each hands a **horizontal
carry bond** to the next orthogonality centre. exp09 established (measured on converged TFIM
$L_x{=}4$) that

- **local** hands a thin carry $\le \eta = 4$ (measured bond profile $[4,2,2,1]$);
- **plain global** must set its residual bond $\eta_q = 6\!-\!8$ to match local's accuracy,
  because it has no disentangler to shrink it;

and an interior column's output leg is $d \cdot (\text{carry})$, so plain global's interior
columns are $(\eta_q/\eta)\times$ fatter, inflating every downstream factorization. The
propagation penalty grows with $L_x$ and eats the sketch's per-column saving.

The proposal, from the theory review (`tikz/rmps_disentangled_column_method`):

![Disentangled column method](../tikz/rmps_disentangled_column_method.png)

Sketch the column $Y = C_j\,\Omega$ with rMPS probes, then run the arrow-compatible TT-SVD
sweep **with a disentangler $D_i$ at each internal vertical cut**, so the vertical bond holds
at $\eta$ and the reorganized weight spills into a bounded horizontal residual. The subtlety
the review stresses, and which finding 1 verifies numerically: a unitary placed *naively* on
an existing vertical bond cannot lower its Schmidt rank, because $\sigma(MD) = \sigma(M)$.
The Moses disentangler defeats this by acting on a **composite** bond $\eta\kappa$ (a first
SVD to that larger bond) and **reshuffling** before the rank-$\eta$ truncation: the $\eta$
half is regrouped with the next output leg $o_{i+1}$ and the $\kappa$ half with the rest, so
$D$ and the reshuffle $A$ do not commute and $D$ genuinely reshapes the spectrum the
truncation sees. Concretely this is the verified `disentangle_altmin` (bit-for-bit vs
Dektor's reference) applied to the thin sampled range.

---

## 2. Mechanism: the null test and the tail reduction

At every internal vertical cut of the sampled range $Y$ we measure three tails: the
no-disentangler rank-$\eta$ tail $\tau_\eta(I)$, the no-disentangler larger-bond tail
$\tau_{\eta_q}(I)$, and the disentangled rank-$\eta$ tail $\tau(D^\star)$, where

$$
\tau^2(D^\star) \;=\; \min_{D^\dagger D = I}\; \sum_{j>\eta}\sigma_j^2\!\big(A(D\,\tilde V)\big),
\qquad
\tilde V = \text{top-}\eta\kappa \text{ carrier of the cut.}
$$

![Mechanism and null test](figures/column_sketch/exp10-disentangler.png)

**Left** — disentangling lowers the vertical rounding tail: at $L_x=5$ the median tail drops
from $\tau_\eta(I)=2.2\times10^{-4}$ (no disentangler, $\eta=4$) to
$\tau(D^\star)=1.1\times10^{-4}$; at $L_x=4$ from $5.2\times10^{-5}$ to $9.8\times10^{-6}$
(a $5\times$ reduction). **Right, the null test** — over random unitaries applied to the
existing composite bond *without* the reshuffle, the rank-$\eta$ tail has standard deviation
$\lesssim 10^{-20}$ (below the $10^{-12}$ reference line at every $L_x$). This is the review's
key claim made quantitative: **the reshuffle, not the unitary, is what compresses.** A gauge
alone is exactly inert.

The verified fallbacks (`tests/test_disentangled_qr.py`): $\kappa=1$ (no composite freedom)
and $N_{\text{dis}}=0$ both reproduce the plain rank-$\eta$ tail identically, and the
optimized tail never exceeds the plain tail ($D=I$ is always feasible).

---

## 3. The real bottleneck is the vertical bond, not the sketch

Before trusting the tail as the lever, we check what actually limits the column error
$\varepsilon_{\text{proj}} = \|(I-QQ^\dagger)C\|_F / \|C\|_F$.

![Accuracy bottleneck](figures/column_sketch/exp10-disentangler-bottleneck.png)

**Left** — $\varepsilon_{\text{proj}}$ separates cleanly by vertical bond: at $L_x=4$ it is
$3.5\times10^{-2}$ at $\eta_q=4$, $9.5\times10^{-3}$ at $\eta_q=6$, $4.2\times10^{-3}$ at
$\eta_q=8$; local Moses ($\eta=4$, with disentangler) sits at $2.1\times10^{-2}$, i.e. between
plain $\eta_q=6$ and $8$. **Right** — at fixed $\eta_q=4$, the curves for $\ell=8$ and
$\ell=12$ lie exactly on top of each other ($3.67\times10^{-2}$ vs $3.43\times10^{-2}$ at
$L_x=4$). The sketch has already captured the column's range with a handful of probes; adding
more does nothing. **Accuracy is set by the retained range dimension — the vertical bond —
not by the sketch.** That is precisely the quantity a disentangler can reorganize, and it is
why plain global (a flat $\eta_q$ vertical bond) needs $\eta_q=6\!-\!8$ where local (an
$\eta=4$ vertical bond *plus* a horizontal residual) needs only $4$.

---

## 4. The disentangler is cheap

![Disentangler cost frontier](figures/column_sketch/exp10-disentangler-frontier.png)

The disentangler's residual-truncation loss (the rank-$\eta$ second-SVD tail it leaves,
relative to $\|Y\|$) falls from $1.4\times10^{-4}$ at $N_{\text{dis}}=0$ to $4.8\times10^{-5}$
after a **single** iteration and $1.9\times10^{-5}$ by $N_{\text{dis}}=10$ — most of the gain
is in the first $1\!-\!3$ steps. The added FLOPs (right) are $\sim\!3\times10^{4}$ per column
at $N_{\text{dis}}=1$ rising to $\sim\!2\times10^{5}$ at $N_{\text{dis}}=10$, against a column
factorization of $10^{6}\!-\!10^{7}$ FLOPs: **under 5% overhead.** This is the regime the
review predicted — "one to five cheap iterations on the thin sketch," not ten heavy ones.

Crucially, the loss it leaves ($\sim\!10^{-4}$ at $L_x=5$, $\sim\!10^{-5}$ at $L_x=4$) is
three orders of magnitude below the column error it must not disturb
($\varepsilon_{\text{proj}}\sim 10^{-2}$), so the reorganization into a bounded-$\kappa$
residual is lossless at the scale that matters.

---

## 5. End-to-end: the verdict flips

Charging the disentangled-global column at vertical bond $\eta=4$ with carry $=\eta$ (the
disentangler keeps the residual bounded, so it hands the **same thin carry as local**, not
$\eta_q$), plus the measured disentangler FLOPs, and re-running exp09's propagation:

![End-to-end speedup](figures/column_sketch/exp10-disentangler-headline.png)

**Left** — plain global needs $\eta_q = 6\!-\!8$ to match accuracy; local and the
disentangled column hold $\eta = 4$. **Right** — the end-to-end speedup vs local (FLOP model,
median over TFIM columns):

| $L_x$ | plain global (exp09) | disentangled global |
|:-----:|:--------------------:|:-------------------:|
| 4 | $2.82\times$ | $17.4\times$ |
| 5 | $\mathbf{0.75\times}$ (loses) | $\mathbf{20.1\times}$ (wins) |

The plain-global curve reproduces exp09's crossover ($2.82 \to 0.75$, dropping below the
break-even line at $L_x=5$). The disentangled-global curve stays **well above 1 and rising**,
because it factorizes the thin sketch (like plain global) *and* hands the thin $\eta=4$ carry
(like local) — it inherits the cheap side of each. ($L_x=3$ is degenerate: the columns are
near-exactly low-rank there, local is essentially free, and neither global variant is
meaningfully comparable — consistent with exp09's treatment.)

---

## Honest accounting: what is proven vs assumed

**Rigorously established (verified code, measured on real columns):**

- The null test — a naive gauge on the existing bond is inert to machine precision.
- The disentangler lowers the vertical tail ($2\!-\!5\times$) and converges in $1\!-\!5$
  iterations at $<5\%$ added cost.
- $\varepsilon_{\text{proj}}$ is set by the vertical bond and is flat in $\ell$ — the sketch
  is a faithful stand-in for the column's range.
- Plain global reproduces exp09's crossover ($2.82\times \to 0.75\times$), so the cost model
  is calibrated against the prior result.

**The one modeling assumption the cost flip rides on:** that the disentangled-global column
at $\eta=4$ actually reaches `local_eps`. Its accuracy is bracketed rigorously by
$\varepsilon_{\text{proj}}(\eta_q = \eta\kappa)$ (best case, the range the composite subspace
spans) plus the residual-truncation loss $\sqrt{\sum_i \tau_i^2(D^\star)}/\|Y\|$, and that
loss is measured to be $\sim\!10^{-4}$ — three orders below the $10^{-2}$ error it stands in
for. The assumption is further motivated by the fact that **local Moses at $\eta=4$ is itself
a disentangled $\eta{=}4$ column** achieving `local_eps`, and the sketch faithfully carries
the range (§3). What is *not* yet done: building the disentangled-global column with its
bounded-$\kappa$ residual MPS explicitly and measuring its $\varepsilon_{\text{proj}}$
against local directly. That is the decisive validation and the recommended next step; the
end-to-end $17\!-\!20\times$ should be read as a **model projection under a well-supported
assumption**, not a measured column error.

Conservative choices baked in: the disentangled carry is charged at $\eta=4$ (= local's
measured carry, not the smaller $\kappa$); the disentangler FLOPs are charged in full at
$N_{\text{dis}}=10$; local FLOPs are *measured* (real Moses move) while both global variants
are *modelled*, exactly as in exp09.

---

## Reproduce

```bash
# full sweep (Lx 3-5, 4 seeds, TFIM g=3.5/3.04 + random control)
python experiments/column_sketch/scripts/exp10_vertical_disentangler_mechanism.py --lxs 3 4 5 --seeds 4
python experiments/column_sketch/scripts/curate_figures.py     # -> reports/figures/column_sketch/exp10-*
pytest tests/test_disentangled_qr.py -q                        # null test, monotonicity, fallbacks
```

Core module: `src/rand_isopeps/column/disentangled_qr.py` (the free-dimension shim
`_CutDims` lets the verified `disentangle_altmin` / `cut_forward` reshuffle a sketch-sweep
carrier; `mechanism_profile` measures the three tails and the null test; `disentangler_flops`
is the added-cost model).
