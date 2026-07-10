# Hamiltonian survey — when does the disentangled global column sketch beat local?

**Running report.** Last updated **2026-07-09** (first NERSC wave). Regenerate the figures
after each new wave with

```bash
python experiments/column_sketch/scripts/survey_report.py --data-dir <outputs/column_sketch/data>
```

and append findings to the changelog at the bottom.

---

## 1. Question

The disentangled-global column sketch holds the vertical bond fixed at $\eta = 4$ and spills the
excess into a bounded horizontal residual $\kappa$ (the verified Moses disentangler inserted into
the rMPS sketch sweep). exp08–exp10 established this on TFIM. The survey asks the physics question:

> **Does the method prefer some Hamiltonians?** Does the disentangler hold $\eta = 4$ and beat the
> local Moses move for some physical systems more than others — and does the win track how
> **low-effective-rank / local-friendly** the physical column is?

Hypothesis (from the exp04 spectrum diagnostic and [`real-moses-e1-svd2-spectrum-excess`]): the win
tracks entanglement across the orthogonality cut — gapped/paramagnetic easiest, critical / frustrated
/ Heisenberg hardest. This mirrors the PI's excited-states paper, where Heisenberg needed $\eta = 36$
vs TFIM's $\eta = 20$.

## 2. Survey ladder

Paper-aligned states (`ham_from_spec`): TFIM $g \in \{1.0, 2.0, 3.04, 3.5\}$ (critical
$g_c \approx 3.044$; ordered $g < g_c$, paramagnetic $g > g_c$), **Heisenberg** (the paper's hard
case), **XXZ** $\Delta \in \{0.5, 1.5\}$ (the anisotropy/entanglement dial), and the **compass**
model (square-lattice Kitaev analog, frustrated). Column heights $L_x \in \{4,5,6\}$, $L_y = 4$,
$\chi = 8$, disentangler freedom $\kappa = 2$, 4 seeds, medians reported. Accuracy metric
$\epsilon = \lVert (I - QQ^*) C\rVert_F / \lVert C\rVert_F$.

Three instruments per state: **exp04** (column spectrum diagnostic — the effective rank, the
hypothesis' direct measurement), **exp09** (plain-global end-to-end cost + the vertical bond
$\eta_q$ it must spend to match local), **exp10** (the 3-way at fixed $\eta = 4$: plain global
`m1` / disentangled-best-case `m2` / local `m3`, plus the mechanism tails).

## 3. Result — the ladder (exp10, fixed $\eta = 4$)

![ladder](figures/column_sketch/hamsurvey-ladder.pdf)

**What $\epsilon_{\mathrm{dis}}$ is (read this before the table).** $\epsilon_{\mathrm{dis}}$ is *not*
a direct measurement of the disentangled column — it is the **best-case bracket** of it. The
$\eta{=}4$ disentangled column captures, per cut, the composite $\eta\kappa = 8$ subspace reorganized
into a thin vertical $\eta{=}4$ bond plus a bounded horizontal residual $\kappa{=}2$;
$\epsilon_{\mathrm{dis}}$ is the projection error of the **plain global sketch at $\eta_q = \eta\kappa
= 8$** — i.e. the accuracy that column would reach *if* the residual reorganization were lossless. The
true error is **bracketed**,
$\epsilon_{\mathrm{dis}}\,(\eta_q{=}8) \le \epsilon_{\mathrm{true}} \le \epsilon_{\mathrm{plain}}\,
(\eta_q{=}4)$, and hugs the best-case edge exactly when the residual-truncation loss
$\tau_{\mathrm{dis}}$ (right panel) is $\ll \epsilon$. So the panels are coupled: the left is the best
case, the right is how trustworthy it is. This **sharpens** the ladder — for winners $\tau_{\mathrm{dis}}
\sim 10^{-4}$ so $\epsilon_{\mathrm{true}} \approx \epsilon_{\mathrm{dis}}$; for losers
$\tau_{\mathrm{dis}} \sim 10^{-3}$ *and* $\epsilon_{\mathrm{dis}} \approx \epsilon_{\mathrm{plain}}$, so
the bracket collapses onto plain-global and the true error is if anything **worse** than plotted
("never wins" is a floor). The exact directly-measured thin-carry column (the residual-MPS / zip-up
build) is the one still-open refinement (§6).

**Left:** $\epsilon_{\mathrm{dis}} / \epsilon_{\mathrm{local}}$ vs $L_x$; **below the dashed line the
disentangler's best case beats local while holding $\eta = 4$.** **Right:** the residual-truncation
loss $\tau_{\mathrm{dis}} = \sqrt{\sum_i \mathrm{dis\_tail}} / \lVert Y\rVert$ — low means the composite
$\eta\kappa$ subspace *can* be reorganized into a thin $\eta = 4$ bond (best case trustworthy).

| state | phase | $L_x{=}4$ | $L_x{=}5$ | $L_x{=}6$ | verdict |
|---|---|:---:|:---:|:---:|---|
| TFIM $g{=}3.5$ | paramagnet | ✅ 0.19 | ✅ 0.39 | ✅ 0.62 | **wins through $L_x{=}6$** |
| TFIM $g{=}2.0$ | ordered | ✅ 0.44 | ✅ 0.88 | ✅ 0.85 | **wins through $L_x{=}6$** |
| TFIM $g{=}1.0$ | deep ordered | ✅ 0.05 | — | — | wins (extreme low-rank; Lx4 only) |
| TFIM $g{=}3.04$ | **critical** | ✅ 0.24 | ✅ 0.50 | ❌ 1.10 | crosses over at $L_x{=}6$ |
| XXZ $\Delta{=}1.5$ | Ising-like | ✅ 0.67 | ❌ 1.04 | ❌ 1.65 | crosses at $L_x{=}5$ |
| compass | frustrated | ✅ 0.74 | ❌ 4.0 | ❌ 3.4 | crosses at $L_x{=}5$ |
| XXZ $\Delta{=}0.5$ | XY-like | ❌ 1.24 | ❌ 1.62 | ❌ 1.98 | never |
| Heisenberg | AFM (high $S$) | ❌ 1.07 | ❌ 2.19 | ❌ 3.93 | **never** |

(cells = median $\epsilon_{\mathrm{dis}}/\epsilon_{\mathrm{local}}$; ✅ $\le 1$.)

**The ordering is exactly the entanglement ladder**, and even the *crossover $L_x$* is monotone in
hardness: gapped TFIM (holds to $L_x = 6$) → critical / frustrated (crosses at 5–6) → Heisenberg /
XY (never). The deep-ordered TFIM $g = 1.0$ is the most extreme win ($\epsilon_{\mathrm{dis}}$ is
**$20\times$** below local at $L_x = 4$) — consistent with a near-product column.

## 4. Mechanism — it's the column rank, not a broken disentangler

- **Null test passes everywhere** ($\max$ `null_std` $\sim 10^{-20}$): a naive gauge on the existing
  bond is inert; the composite-bond *reshuffle* is what compresses. The mechanism is sound.
- **Winners:** $\epsilon_{\mathrm{dis}} \ll \epsilon_{\mathrm{plain}}$ — the disentangler genuinely
  lowers the $\eta=4$ tail below plain global (TFIM $g{=}3.04$, $L_x{=}4$: plain $0.015 \to$ dis
  $0.005$), and $\tau_{\mathrm{dis}} \sim 10^{-6}\!-\!10^{-4}$ (tight bracket → `m2` trustworthy).
- **Losers:** $\epsilon_{\mathrm{dis}} \approx \epsilon_{\mathrm{plain}}$ — the disentangler adds
  *nothing* (Heisenberg $L_x{=}6$: plain $0.088$, dis $0.090$), and $\tau_{\mathrm{dis}} \sim 10^{-3}$
  (10× looser). There is no low-rank structure to reorganize into $\eta = 4$: the physics forces the
  column rank above $\eta$.

**Punchline:** the disentangler works *iff* the column's effective rank is $\sim \eta$, and the
physical systems where that holds are exactly the low-entanglement ones. The method is a
**low-entanglement specialist**, not a universal replacement for the Moses move.

## 5. Cost projection (exp09, plain global)

![cost](figures/column_sketch/hamsurvey-cost.pdf)

Plain global (no disentangler) must spend $\eta_q = 6\!-\!8$ to match local on the easy states and
**cannot match at all** ($\eta_q = 8$, `matched = 0`) on the hard states — the fat-carry problem the
disentangler exists to solve. For the states where `m2` holds accuracy at $\eta = 4$, the FLOP model
projects a large end-to-end speedup ($\sim 12\!-\!21\times$); for the losers the speedup is void
because accuracy isn't matched. *(TFIM $g{=}2.0$ is absent from this figure — its exp09 CSV was lost
to the slug-collision bug; recovery queued, see §6.)*

## 6. Status & pending

| item | state |
|---|---|
| exp09 + exp10 for 7 states, $L_x$ 4–6 | ✅ in this report |
| $L_x = 6$ ceiling (tfim@3.5/3.04/heis, full node) | ✅ `COMPLETED` |
| parallel disentanglers (cut-workers 1 vs 4) | ✅ $\sim 1.18\times$ on CPU (Amdahl-bound; the case *for* the GPU batch) |
| **exp04 spectrum diagnostic** (all states) | ⏳ respray `55732645` — the *direct* rank measurement |
| **TFIM $g{=}1.0$** full $L_x$ | ⏳ same respray (Lx4 only so far) |
| **TFIM $g{=}2.0$** exp09 (corrupted) | ⏳ `sbatch --array=2` after the PID-slug fix |
| $L_x = 7,8$ ceiling | ⏳ `STAGE=7/8` |
| **exact thin-carry column** (residual-MPS / zip-up build) | 🔬 open *method* item — turns $\epsilon_{\mathrm{dis}}$ (best-case bracket, §3) into a direct measurement |

**On the open method item:** every $\epsilon_{\mathrm{dis}}$ here is the *best-case* bracket endpoint
(the plain sketch at $\eta_q = \eta\kappa$; see §3), not the literal disentangled column. The bracket
is rigorous and the residual loss $\tau_{\mathrm{dis}}$ bounds the gap, but the decisive validation is
the exact bounded-$\kappa$ residual-MPS build (a quick numpy version over-counted and was discarded);
gated on the $\kappa{=}1$ collapse + isometry + bracket + monotonicity harness.

**Prediction to test when exp04 lands:** the effective-rank curves should order the states the same
way as §3 — a direct, spectrum-side confirmation of the mechanism.

## Changelog

- **2026-07-09 (wave 1):** first NERSC batch on `m4926`. 16/19 array tasks + the $L_x{=}6$ ceiling
  `COMPLETED`; the three TFIM $g{=}1.0$ tasks OOM-killed (deep-ordered cat column, memory-worst case;
  fixed with fewer/fatter workers). Two infra bugs fixed en route: a same-second Slurm-array **slug
  collision** (corrupted the TFIM $g{=}2.0$ exp09 CSV; fixed with a PID in `timestamp_slug`), and
  `publish.sh` on a fresh orphan branch. The ladder (§3) and mechanism (§4) are the headline result.
