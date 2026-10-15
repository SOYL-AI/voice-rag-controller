# Theory: when does joint control of retrieval and timing beat decomposed control?

Status: draft for the INDICON 2026 major revision (methods/theory). Every
number in this document comes from one of two scripts, which regenerate it
from committed artifacts with one command each:

| script | output | what it computes |
|---|---|---|
| `analysis/v1_theory_check.py` | `docs/research_v2/v1_theory_check.json`, `.png` | Props. 1–2 and the corrected objective, verified on the real v1 E3 sweep, using the controllers' own code for every cut |
| `analysis/v2_power.py` | `docs/research_v2/v2_power.json` | Props. 3–4 effect sizes in simulation (v1 per-item outcomes plus synthetic beliefs and timing), the power analysis for `PREREG_V2.md` |

```
PYTHONUTF8=1 python -m analysis.v1_theory_check
PYTHONUTF8=1 python -m analysis.v2_power
```

Section 9 lists where the data disagree with the framing we started from. The
data win in each case.

---

## 1. Model and notation

One user turn (item) $i$. Indices are dropped where unambiguous.

| symbol | meaning | v1 value |
|---|---|---|
| $N\in\{0,1\}$ | retrieval is needed (closed-book answer wrong) | label in `scenarios.json` |
| $b=\Pr(N=1\mid\mathcal I)$ | calibrated belief given the controller's information $\mathcal I$ | `estimate_voi()`, floored at 0.05 |
| $Q^{r},Q^{s}$ | answer correctness if the turn retrieves or skips | judged, per seed |
| $g^{+}=\mathbb E[Q^{r}-Q^{s}\mid N=1]$ | retrieval gain on needed turns | v1 controller assumes 1 |
| $g^{-}=-\mathbb E[Q^{r}-Q^{s}\mid N=0]$ | retrieval harm on non-needed turns | v1 controller assumes 0 |
| $\mathcal T$ | finite set of timing actions (dispatch rules) | {sentence_end, immediate, fixed_early_proxy} |
| $L$ | retrieval latency if fully exposed | 0.672 s |
| $o_t$ | latency hidden by timing $t$ (overlap) | 0, 0.050, 0.188 s |
| $e_t$ | expected exposed latency of timing $t$ | $L-o_t$ |
| $w_t$ | waste rate (expected extra retrievals) | 0, 0.02, 0.08 (placeholder) |
| $\rho_t$ | partial-query risk (fraction of $g^{+}$ lost) | 0 for all $t$ |
| $\lambda_l,\lambda_c$ | latency price (per s), compute price (per retrieval) | grid, 0.05 |
| $R_t$ | remaining user speech after dispatch under $t$ (v2) | not modelled |
| $\pi$ | non-hideable latency of retrieval (longer prompt etc., v2) | not modelled |
| $Z$ | turn-level timing state observed before dispatch (speech progress, predicted remaining time) | not modelled |

**Objective.** Skipping is the reference, $U(\text{skip})=0$. Retrieving with timing $t$:

$$U_t(b)=b\,g^{+}(1-\rho_t)-(1-b)\,g^{-}-\kappa_t,\qquad \kappa_t=\lambda_l e_t+\lambda_c(1+w_t).$$

It is convenient to write $U_t(b)=s_t b-g^{-}-\kappa_t$ with slope
$s_t=g^{+}(1-\rho_t)+g^{-}>0$. The **v1 objective** is the special case
$g^{+}=1,\ g^{-}=0,\ \rho_t\equiv0$, so $U_t(b)=b-\kappa_t$
(`_objective.scalarized_utility` with quality $=$ VoI). The **corrected v2
objective** (Section 7) uses measured $g^{+},g^{-}$ and, in v2, the turn-level
exposed latency $e_t(Z)=\mathbb E[\pi+\max(0,L-R_t)\mid Z]$.

**Controllers.** All share the belief $b$, the prices and the cost model.

* **Ours (joint).** $\arg\max$ over $\{\text{skip}\}\cup\{(\text{retrieve},t):t\in\mathcal T\}$; ties go to skip, then to the earlier element of `TIMING_ORDER` (`ours.py`).
* **B9 (decomposed with prices).** Retrieval sub-optimizer: retrieve iff $s b-g^{-}-\lambda_l L-\lambda_c(1+w_{\text{se}})>0$, pricing the *full* latency $L$ (`_optimize_retrieval`). Timing sub-optimizer, run independently: $\tau_B=\arg\max_t[-b\,g^{+}\rho_t+\lambda_l o_t-\lambda_c w_t]$ (`_optimize_timing`). Note that B9's timing optimizer already reads the belief.
* **B9+ (decomposed, expected-exposure pricing).** Same timing sub-optimizer $\hat\tau$; the retrieval sub-optimizer prices the expected exposed latency of the timing that sub-optimizer will use: retrieve iff $U_{\hat\tau}(b)>0$. It never sees a turn-level timing state, so it stays decomposed. (Implemented as `b9plus_decide` in `analysis/v1_theory_check.py`.)

**Policy classes (used from Section 5 on).** A policy is a pair $(r,\tau)$ of a
retrieve indicator and a timing. The *joint class* $\Pi_J$ lets both be
measurable in everything the controller observes, $(b,Z)$. The *decomposed
class* $\Pi_D$ requires the retrieve decision $r$ to be measurable in the belief
$b$ alone, using fixed prices and any fixed scalar summary of the timing model,
while $\tau$ may be anything. B9 and B9+ at any price lie in $\Pi_D$. This
information-structure definition is the one Props. 3–4 are about. It is also
the right reading of "decomposed": in v1 the two sub-optimizers already share
the belief, and what they do not share is turn-specific timing information.

**Assumptions of the v1 regime.**

* **A1** (timing-invariant quality): $\rho_t=\bar\rho$ for all $t$. Measured $\bar\rho=0$ in v1.
* **A2** (constant overlap): $e_t$ is a constant per timing policy, the same on every turn.
* **A3** (dispatch-invariant belief): the belief used for the decision does not depend on when retrieval is dispatched; there is one $b$ per turn.

Under A1 every slope equals $s=g^{+}(1-\bar\rho)+g^{-}$. B9 is evaluated on the
same slope (in v1, $s=1$).

**Two identities used throughout.** For an integrable random variable $X$, write $x^{\pm}=\max(\pm x,0)$:

* **(J)** $\ \mathbb E[X^{+}]-(\mathbb E X)^{+}=\min(\mathbb E X^{+},\mathbb E X^{-})=\tfrac12(\mathbb E|X|-|\mathbb EX|)\le\tfrac12\mathbb E|X-\mathbb EX|.$
  *Proof.* If $\mathbb EX\ge0$: $\mathbb EX^{+}-\mathbb EX=\mathbb EX^{-}$, and $\mathbb EX^{-}\le\mathbb EX^{+}$. The case $\mathbb EX<0$ is symmetric. The middle form follows from $x^{+}=(x+|x|)/2$. The bound uses $|X|\le|X-\mathbb EX|+|\mathbb EX|$. ∎
  The gap is therefore zero **iff** $X\ge0$ a.s. or $X\le0$ a.s.
* **(I)** For square-integrable $C$: $\int_{-\infty}^{\infty}\big(\mathbb E(u-C)^{+}-(u-\mathbb EC)^{+}\big)\,du=\tfrac12\mathrm{Var}(C).$
  *Proof.* Let $m=\mathbb EC$. For $M$ large, $\int_{-M}^{M}[(u-c)^{+}-(u-m)^{+}]du=\tfrac12[(M-c)^2-(M-m)^2]=(m-c)(M-m)+\tfrac12(m-c)^2$. Take expectations; the first term has mean 0. Integrability lets $M\to\infty$. ∎

---

## 2. Proposition 1 (equivalence under A1–A3)

Let $t^{*}(\lambda_l,\lambda_c)=\arg\min_t\kappa_t$, with ties broken by
`TIMING_ORDER`, and $c_O=\kappa_{t^{*}}$. Define the price map

$$\lambda_l'(\lambda_l,\lambda_c)=\frac{c_O-\lambda_c(1+w_{\text{se}})}{L}=\frac{\lambda_l e_{t^{*}}+\lambda_c\,(w_{t^{*}}-w_{\text{se}})}{L}.$$

**Proposition 1.** Assume A1–A3, $\lambda_l,\lambda_c\ge0$, and $w_{\text{se}}=\min_t w_t$ (true in v1). Then:

1. **Retrieve decisions are identical.** For every $b\in[0,1]$, Ours at $(\lambda_l,\lambda_c)$ retrieves iff B9 at $(\lambda_l',\lambda_c)$ retrieves. This is exact, ties included.
2. **The map is a bijection of $[0,\infty)$.** $\lambda_l\mapsto\lambda_l'$ is continuous, strictly increasing, $\lambda_l'(0)=0$ and $\lambda_l'\to\infty$.
3. **Timing.** When both retrieve, Ours uses $t^{*}(\lambda_l)$ and B9 uses $t^{*}(\lambda_l')$, with the same function $t^{*}$. They coincide whenever $\lambda_l$ and $\lambda_l'$ lie in the same linear piece of $\lambda\mapsto\min_t\kappa_t$. This holds for every price if all $w_t$ are equal. Otherwise it fails only on a bounded price interval just above each timing breakpoint.
4. **Exact versus approximate map.** $\lambda_l'=\lambda_l\frac{L-o^{*}}{L}+\lambda_c\frac{w^{*}-w_{\text{se}}}{L}$ when $t^{*}$ is the maximal-overlap timing ($o^{*},w^{*}$ its overlap and waste). The commonly quoted $\lambda_l(L-o^{*})/L$ is exact **iff** $\lambda_c(w^{*}-w_{\text{se}})=0$.
5. **Robustness.** The same map holds for the corrected objective (any $g^{+},g^{-}$) and for any constant $\bar\rho<1$, because $g^{-}$ and $s$ enter both thresholds identically.

*Proof.* (1) Under A1, $U_t(b)=sb-g^{-}-\kappa_t$. Since $s>0$, $\max_tU_t(b)>0$ iff $sb-g^{-}>\min_t\kappa_t=c_O$. The tie rule puts skip first, so Ours retrieves iff $b>\theta_O=(c_O+g^{-})/s$ strictly. B9 retrieves iff $sb-g^{-}-\lambda_l'L-\lambda_c(1+w_{\text{se}})>0$ strictly (`utility > 0.0`), i.e. iff $b>\theta_B(\lambda_l')=(\lambda_l'L+\lambda_c(1+w_{\text{se}})+g^{-})/s$. Substituting the definition of $\lambda_l'$ gives $\theta_B(\lambda_l')=\theta_O$. Both inequalities are strict, so the retrieve sets coincide for every $b$. Non-negativity holds because $c_O\ge\lambda_c(1+w_{\text{se}})$ when $w_{\text{se}}$ is minimal and $e_t\ge0$.

(2) $\lambda\mapsto\min_t\kappa_t(\lambda)$ is a minimum of affine functions with slopes $e_t>0$. It is therefore continuous, concave and strictly increasing, with value $\lambda_c(1+w_{\text{se}})$ at $\lambda=0$ (where $t^{*}=$ sentence_end, the minimal-waste policy first in tie order).

(3) Under A1 the belief term $-b\,g^{+}\bar\rho$ in B9's timing objective is constant in $t$. Then $\arg\max_t[\lambda o_t-\lambda_c w_t]=\arg\min_t[\lambda(L-o_t)+\lambda_c(1+w_t)]=t^{*}(\lambda)$, with the same tie order. So B9 at $\lambda'$ uses $t^{*}(\lambda')$, and Ours at $\lambda$ uses $t^{*}(\lambda)$, which is $b$-independent because $U_t-U_{t'}=\kappa_{t'}-\kappa_t$. Inside one linear piece the map is affine with a fixed point at that piece's degeneracy price. Above the fixed point, $\lambda'$ stays above it but can drop below the piece's lower breakpoint.

(4) Substitute $e_{t^{*}}=L-o^{*}$. (5) Neither $g^{-}$ nor $s$ appears in $\lambda_l'$. ∎

**Corollary 1 (identical frontiers).** Suppose a turn's measured outcome is a
function of (item, seed, retrieve, timing) only, as in E3's cell design
(`simulator/e3_full_sweep.py`, Phase B). Then off the timing-mismatch interval
of (3), Ours$(\lambda)$ and B9$(\lambda'(\lambda))$ produce identical per-item
outcomes, and hence identical operating points (accuracy, p95 TTFA, compute).
By (2) the two price sweeps over $[0,\infty)$ trace the same set of operating
points, so neither frontier can dominate the other.

In v1 the grid is generated by the tiling recurrence
$\theta_O(\lambda_{i})=\theta_B(\lambda_{i-1})$ (`simulator/e3_grid.py`), so
$\lambda'(\lambda_i)=\lambda_{i-1}$ **by construction**: Ours at grid price $i$
must equal B9 at grid price $i-1$. The empirical content of Section 8.1 is that
the implementation matches the theory, not that the shift exists.

---

## 3. Proposition 2 (gap at a fixed price)

Assume A1–A3 and **model correctness (M)**: $\mathbb E[N\mid b]=b$ (calibration),
$\mathbb E[Q^{r}-Q^{s}\mid b,t]=b\,g^{+}(1-\bar\rho)-(1-b)g^{-}$, and the realized
expected latency and compute of a retrieval at $t$ are $e_t$ and $1+w_t$.

**Proposition 2.** At a fixed price $(\lambda_l,\lambda_c)$, with $c_B=\lambda_lL+\lambda_c(1+w_{\text{se}})$ and the band $\mathcal B=(\theta_O,\theta_B]$, $\theta_B=(c_B+g^{-})/s$:

$$\mathbb E[U_{\text{Ours}}-U_{\text{B9}}]=\mathbb E\big[(s\,b-g^{-}-c_O)\,\mathbf 1\{b\in\mathcal B\}\big],$$

$$0\le\mathbb E[U_{\text{Ours}}-U_{\text{B9}}]\le(c_B-c_O)\Pr(b\in\mathcal B)=\big(\lambda_l o_{t^{*}}-\lambda_c(w_{t^{*}}-w_{\text{se}})\big)\Pr(b\in\mathcal B)\le\lambda_l o^{*}\Pr(b\in\mathcal B).$$

For the v1 objective this reads $\mathbb E[(b-c_O)\mathbf 1\{c_O<b\le c_B\}]$. For B9+ at the same price the gap is $0$ pointwise.

*Proof.* At the same price, B9's timing sub-optimizer returns $t^{*}(\lambda_l)$ (proof of Prop. 1(3)). So whenever B9 retrieves it incurs the same realized cost $c_O$ as Ours, even though its *decision* priced $c_B$. Three cases:

* $b\le\theta_O$: both skip, so the difference is 0.
* $b>\theta_B$: both retrieve with the same timing, so the difference is 0.
* $b\in\mathcal B$: Ours retrieves and earns, by (M), $sb-g^{-}-c_O\in(0,\,s\theta_B-g^{-}-c_O]=(0,\,c_B-c_O]$, while B9 earns 0.

Taking expectations gives the identity. The bounds follow from $0<sb-g^{-}-c_O\le c_B-c_O$ on $\mathcal B$. For B9+, $\hat\tau=t^{*}$ and its rule is $U_{t^{*}}(b)>0$, which is Ours' rule. ∎

**Remark 2.1 (the bound is a model statement).** Without (M), let $m(b)=\mathbb E[Q^{r}-Q^{s}\mid b]$ be the true quality effect and $\tilde c$ the realized cost. The realized gap is $\mathbb E[(m(b)-\tilde c)\mathbf 1\{b\in\mathcal B\}]$. It can be negative, and its magnitude is bounded only by $\sup_{\mathcal B}|m-\tilde c|\cdot\Pr(\mathcal B)$, which is not $O(\lambda_lo^{*})$. Section 8.4 shows this is what happened in v1. The v1 controller's model ($g^{-}=0$, exposed latency 0.484 s) predicts a positive gap at every band price, while the realized gap is negative at 10 of 11.

**Remark 2.2 (reading the band).** $\Pr(b\in\mathcal B)$ is the fraction of turns
on which the two controllers can disagree. Under A1–A3 the decisions differ on
exactly the in-band turns, so the observed disagreement count equals the band
count at every price (Section 8.3).

---

## 4. The objective corrected for retrieval harm

With $\rho\equiv0$, retrieval at cost $\kappa$ is worthwhile iff

$$b\,g^{+}-(1-b)g^{-}-\kappa>0\iff b>b^{*}(\kappa)=\frac{\kappa+g^{-}}{g^{+}+g^{-}}.$$

With partial-query risk, $g^{+}$ becomes $g^{+}(1-\rho_t)$.

* **v1 objective** ($g^{+}=1,g^{-}=0$): $b^{*}=\kappa$. At $\lambda_l=0$, $b^{*}=\lambda_c=0.05$, so the v1 controller retrieves on every turn with $b>0.05$.
* **Measured on v1** (25 needed and 25 not-needed scenarios, five seeds; item-bootstrap 95% CIs): $g^{+}=0.720\ [0.504,0.888]$ and $g^{-}=0.592\ [0.408,0.768]$. Accuracy on needed items rises from 0.112 (skip) to 0.832 (retrieve). On non-needed items it falls from 0.704 to 0.112. Hence $b^{*}(0)=g^{-}/(g^{+}+g^{-})=0.451$.

Why the correction matters:

1. **It moves the zero-price boundary from 0.05 to 0.45.** Under the v1 objective, latency prices are the *only* thing that lifts the cut off the floor. The v1 grid had to reach $\lambda_l\ge1.24$ (more than one accuracy unit per second) before the band met the populated part of the belief distribution (Section 8.3). Under the corrected objective, the cut sits mid-range at zero latency price, and prices move it within the region where a non-saturated belief distribution has mass.
2. **It is necessary for external validity, not sufficient to break the null.** The decision is still a threshold in $b$, so Prop. 1 holds unchanged (5).
3. **It is conditional on the data.** On v1's saturated belief distribution only 8% of scenarios lie within ±0.15 of $b^{*}=0.451$ (38% lie below it). The claim that the boundary sits where the data have mass must be checked on the v2 datasets. `PREREG_V2.md` makes that an explicit dev-set check.
4. **It explains the sign of v1's mid-grid gap.** Section 8.4 shows the corrected objective predicts the negative accuracy gap the v1 objective cannot produce.

---

## 5. Proposition 3 (turn-dependent overlap, channel C1)

Drop A2. Exposed latency now depends on a turn-level timing state $Z$ observed
before dispatch, such as predicted remaining speech:
$e_t(Z)=\mathbb E[\pi+\max(0,L-R_t)\mid Z]$. Keep A1 (take $\rho\equiv\bar\rho$)
and A3. Write

$$A(b)=sb-g^{-},\qquad C(Z)=\min_t\big[\lambda_l e_t(Z)+\lambda_c(1+w_t)\big],\qquad \Delta(b,Z)=A(b)-C(Z).$$

$\Delta$ is the net value of retrieving with the best timing for this turn.

**Proposition 3.** Under (M) with $e_t(Z)$ correct:

1. $\sup_{\Pi_J}\mathbb EU=\mathbb E[\Delta(b,Z)^{+}]$, attained by **Ours-v2**: retrieve iff $\Delta(b,Z)>0$, with $\tau=\arg\min_t\kappa_t(Z)$.
2. $\sup_{\Pi_D}\mathbb EU=\mathbb E[(\mathbb E[\Delta\mid b])^{+}]$, attained by $r_D(b)=\mathbf 1\{\mathbb E[\Delta\mid b]>0\}$ with the same $\tau$. If $b\perp Z$, then $\mathbb E[\Delta\mid b]=A(b)-\mathbb EC(Z)$. That is B9+ pricing the expected exposed cost $\mathbb EC$, so **B9+ is the best decomposed controller**, and B9 (pricing $L$) is dominated by it.
3. **Exact gain.**
$$G:=\sup_{\Pi_J}\mathbb EU-\sup_{\Pi_D}\mathbb EU=\mathbb E_b\Big[\min\big(\mathbb E[\Delta^{+}\mid b],\ \mathbb E[\Delta^{-}\mid b]\big)\Big]\ \ge0.$$
4. **Strictness.** $G>0$ iff, with positive probability over $b$, $\Delta(b,Z)$ takes both signs with positive conditional probability. When $b\perp Z$, this means $\Pr\big(A(b)\in(\operatorname{ess\,inf}C,\operatorname{ess\,sup}C)\big)>0$ with $C(Z)$ non-degenerate: positive belief mass in the *C1 band* $\mathcal B_{C1}=A^{-1}\big((\inf C,\sup C)\big)$, plus genuine turn-to-turn variation in exposed latency.
5. **Bound.** If $b\perp Z$: $G\le\tfrac12\,\mathbb E|C(Z)-\mathbb EC(Z)|\cdot\Pr(b\in\mathcal B_{C1})$. When the best timing does not change with $Z$, this is $G\le\tfrac{\lambda_l}{2}\mathbb E|e^{*}(Z)-\mathbb Ee^{*}(Z)|\Pr(b\in\mathcal B_{C1})$.
6. **Second-order form.** If $b\perp Z$ and $b$ has a density $f$ that is continuous near $\theta^{*}=(\mathbb EC+g^{-})/s$, then as the spread of $C$ shrinks,
$$G=\frac{f(\theta^{*})}{2s}\,\mathrm{Var}(C(Z))\,(1+o(1))=\frac{f(\theta^{*})\,\lambda_l^{2}\,\mathrm{Var}(e^{*}(Z))}{2(g^{+}+g^{-})}(1+o(1))\quad(\bar\rho=0).$$

*Proof.* (1) For fixed $(b,Z)$ the best action is skip (0) or retrieve with the cheapest timing ($\Delta$), so pointwise optimization gives $\Delta^{+}$, and the pointwise maximizer is measurable.

(2) If $r$ depends on $b$ only, $\mathbb E[r(b)\Delta(b,Z)]=\mathbb E[r(b)\,\mathbb E[\Delta\mid b]]$, which is maximized by $r_D$. With $b\perp Z$, $\mathbb E[C(Z)\mid b]=\mathbb EC$.

(3) Subtract (2) from (1) and apply identity (J) conditionally on $b$.

(4) By (J), the conditional gap is zero iff $\Delta(b,\cdot)$ does not straddle 0.

(5) Apply the bound in (J) conditionally; the gap is zero outside $\mathcal B_{C1}$.

(6) $G=\int f(b)\,[\mathbb E(A(b)-C)^{+}-(A(b)-\mathbb EC)^{+}]\,db$. Substitute $u=A(b)$, $db=du/s$. The integrand vanishes outside a neighbourhood of $u=\mathbb EC$ of width $O(\mathrm{spread}(C))$, so $f(b)=f(\theta^{*})+o(1)$ there. Apply (I). ∎

**Interpretation.** Conditioning a threshold decision on turn-level information
pays only on turns near the threshold, and on those turns the decision is worth
nearly nothing either way. This is the envelope-theorem logic. The expected
gain is therefore **second order**: quadratic in $\lambda_l$ times the spread of
exposed latency. This drives the power analysis.

**Verification in simulation** (`v2_power.json`, `cells_C1`). The closed form
(6) is checked against the exact model-expected gain (both evaluated with the
controller's own exposure model). For d1_open × remote_api the two agree to
three significant figures: 0.00323 vs 0.00333 at the 0.8 belief-quantile price
and 0.00517 vs 0.00516 at 0.9. Where the density varies across the band, or
the predictor is noisy, (6) overestimates; for example d2_support ×
faiss_rerank gives 0.00355 vs 0.00147 at 0.9.

**Estimation error (the proposition is an expected-utility statement given correct models).** Ours-v2 acts on estimates $\hat b,\hat C(Z)$. If $|\hat C-C|\le\varepsilon$ a.s., the plug-in rule loses at most $\mathbb E[|\Delta|\mathbf 1\{|\Delta|\le\varepsilon\}]\le\varepsilon\Pr(|\Delta|\le\varepsilon)$ against the oracle joint rule, since it can only err when $|\Delta|\le\varepsilon$. B9+ needs only the scalar $\mathbb EC$, which is estimated far more accurately on dev. So the realized advantage is positive only if

$$G>\varepsilon\Pr(|\Delta|\le\varepsilon).$$

The left side is second order in the latency spread; the right side is first order in the predictor error. **A remaining-speech predictor that is merely unbiased is not enough; it must be accurate relative to $\lambda_l\,\mathrm{sd}(e^{*})$.**

Simulation (d2_support × remote_api, four-price average, `sensitivity_C1_d2_remote_api`):

| predictor log-sd | gain |
|---|---|
| 0 | +0.0035 |
| 0.15 | +0.0029 |
| 0.30 | +0.0022 |
| 0.60 | −0.0003 (net loss) |

Belief miscalibration enters the same way as in Remark 2.1.

---

## 6. Proposition 4 (information vs. overlap, channel C2) and channel C3

### 6.1 The stopping problem (commit architecture)

Drop A3. Decision epochs $k=1,\dots,K$ are the successive ASR partials; $K$ is
end of turn. $\mathcal F_k$ is the information at $k$ (prefix signals, timing
state). Assume the belief is Bayes-consistent, $b_k=\Pr(N=1\mid\mathcal F_k)$,
so $(b_k)$ is an $\mathcal F$-martingale. **Commit architecture:** a retrieval
dispatched at $k$ is used in the answer. Dispatching at $k$ is worth

$$D_k=s_kb_k-g^{-}-\lambda_lX_k-\lambda_c(1+w_k),\qquad s_k=g^{+}(1-\rho_k)+g^{-},\quad X_k=\mathbb E[\pi+\max(0,L-R_k)\mid\mathcal F_k].$$

Skipping is worth 0 and can be deferred to $K$ at no cost. The value is the
Snell envelope

$$V_K=D_K^{+},\qquad V_k=\max\big(D_k,\ \mathbb E[V_{k+1}\mid\mathcal F_k]\big),$$

and $\tau^{*}=\min\{k:D_k\ge\mathbb E[V_{k+1}\mid\mathcal F_k]\}$ is optimal.

**One-step value of waiting.** Let $W_k=\mathbb E[D_{k+1}^{+}\mid\mathcal F_k]-D_k^{+}$
(wait one step and act myopically, minus act now). Using the martingale
property, with $\rho_k,w_k$ deterministic in $k$:

$$W_k=\underbrace{\min\big(\mathbb E[D_{k+1}^{+}\mid\mathcal F_k],\mathbb E[D_{k+1}^{-}\mid\mathcal F_k]\big)}_{I_k\ \text{(value of information, identity J)}}+\underbrace{(D_k+\delta_k)^{+}-D_k^{+}}_{\text{drift}},$$

$$\delta_k=\mathbb E[D_{k+1}-D_k\mid\mathcal F_k]=b_kg^{+}(\rho_k-\rho_{k+1})-\lambda_l\big(\mathbb E[X_{k+1}\mid\mathcal F_k]-X_k\big)-\lambda_c(w_{k+1}-w_k).$$

The drift collects the query improvement from a longer prefix (C3, $+$) and the
overlap lost by waiting (C1, $-$). $I_k$ is Howard's (1966) value of
information: the expected gain from deciding after the next belief update
rather than before.

**Proposition 4.**

1. **(Wait.)** If $W_k>0$, waiting at least one step is strictly better than both dispatching and skipping now. This holds in particular when $D_k\le0$, $D_k+\delta_k\le0$ and $I_k>0$ (a borderline turn), or when $I_k>D_k^{+}-(D_k+\delta_k)^{+}$ (information value exceeds the overlap lost).
2. **(Dispatch now.)** Suppose $\delta_j\le0$ for all $j\ge k$ (no query improvement left; overlap only shrinks). Then $V_k-D_k\le\mathbb E[\max_{j\ge k}D_j^{-}\mid\mathcal F_k]$. Hence if $D_k>0$ and $\Pr(\min_{j\ge k}D_j<0\mid\mathcal F_k)=0$, dispatching now is optimal. It is strictly optimal if moreover $\delta_k<0$.
3. **(Not representable by decomposition.)** Let $\mathcal G_k\subseteq\mathcal F_k$ be the timing filtration (speech progress, partial availability, predicted remaining time; no belief). A decomposed policy chooses its dispatch epoch as a $\mathcal G$-stopping time and decides retrieve/skip with $b$ at that epoch. Suppose there are $k$ and a $\mathcal G_k$-event $E$ with $\Pr(E)>0$ on which both the strict-dispatch set of (2) and the strict-wait set of (1) have positive conditional probability. Then every decomposed policy is strictly suboptimal.

*Proof.* (1) "Wait one step, then act myopically" is feasible, so $\mathbb E[V_{k+1}\mid\mathcal F_k]\ge\mathbb E[D_{k+1}^{+}\mid\mathcal F_k]>D_k^{+}\ge\max(D_k,0)$. The decomposition of $W_k$ is identity (J) applied to $D_{k+1}$ given $\mathcal F_k$, plus adding and subtracting $(\mathbb E[D_{k+1}\mid\mathcal F_k])^{+}=(D_k+\delta_k)^{+}$.

(2) Under the drift condition, $(D_j)_{j\ge k}$ is a supermartingale. For any stopping time $\sigma\ge k$ that ends in dispatch or skip, the payoff is at most $D_\sigma+D_\sigma^{-}\le D_\sigma+\max_{j\ge k}D_j^{-}$. Optional stopping gives $\mathbb E[D_\sigma\mid\mathcal F_k]\le D_k$. Strictness: the continuation value is at most $\mathbb E[D_{k+1}\mid\mathcal F_k]=D_k+\delta_k<D_k$ when $D$ cannot go negative.

(3) On $E$ the decomposed policy's epoch-$k$ action (commit now, with dispatch or skip decided by $b_k$, versus defer) is the same for every belief value. If it commits, it loses $W_k>0$ on the wait subset. If it defers, it loses $D_k-\mathbb E[V_{k+1}\mid\mathcal F_k]>0$ on the dispatch subset. Both subsets have positive probability. ∎

The stopping region $\{(k,b,z):D_k\ge\mathbb E[V_{k+1}\mid\mathcal F_k]\}$ is not a product of a belief set and a timing set: dispatch early when confident, wait when borderline. That is the trade-off a decomposed controller cannot represent.

### 6.2 Caveat: speculative dispatch removes the C2 advantage when compute is cheap

Prop. 4 assumes **commit**. Suppose the pipeline instead allows a retrieval to be dispatched speculatively and the decision to *use* it to be taken at $K$ with $b_K$; an unused result costs compute only. Then:

**Proposition 4′.** With $\rho\equiv0$ and $R_k$ non-increasing pathwise, the decomposed policy SPEC is optimal over all policies when $\lambda_c=0$. SPEC dispatches at the earliest epoch and uses the result iff $A(b_K)-\lambda_lX_1>0$. For $\lambda_c>0$ the joint optimum exceeds SPEC by at most $\lambda_c(1+w_1)$ per turn.

*Proof.* Any policy that uses a retrieval dispatched at $\tau\le K$ pays exposure $X_\tau\ge X_1$ pathwise, and its use decision can do no better than the one based on $b_K$. So its value is at most $\mathbb E[(A(b_K)-\lambda_lX_1)^{+}]$, which SPEC attains up to the compute it spends, at most $\lambda_c(1+w_1)$ per turn. ∎

So **C2 is a joint-vs-decomposed channel only under commit**, or when compute is expensive. The simulation agrees. In `cells_C1C2_commit`, SPEC beats the joint-commit controller at every price above the median cut in d2_support (joint minus SPEC between −0.0018 and −0.0121), and joint-commit wins only at the lowest price, where SPEC's compute dominates (+0.020 to +0.022). The v2 pipeline must state its architecture. If discarding a retrieval is possible, SPEC is a mandatory pre-registered rival (`PREREG_V2.md`).

### 6.3 Proposition 5 (partial-query risk, channel C3)

Keep A2 and A3, and let $\rho_t$ vary with $t$. Then $U_t(b)=s_tb-g^{-}-\kappa_t$.

**Proposition 5.**

1. Given retrieval, the optimal timing is $t^{*}(b)=\arg\max_t[-g^{+}\rho_tb-\kappa_t]$, the upper envelope of lines in $b$. Order the timings so that later ones have lower risk and higher cost ($\rho_t\downarrow$, $\kappa_t\uparrow$). Then $U$ has increasing differences in $(b,t)$, because $\partial_bU_t=s_t$ is increasing in $t$. By Topkis' theorem $t^{*}(b)$ is non-decreasing: confident-needed turns wait for a better query, while borderline turns that are still worth retrieving dispatch earlier. It is non-constant on the retrieve region iff some switch point $b_{ts}=(\kappa_s-\kappa_t)/\big(g^{+}(\rho_t-\rho_s)\big)$ lies in $(\theta,1)$.
2. The retrieve set $\{b:\max_tU_t(b)>0\}=(\theta,1]$ is still an upper set, since every $U_t$ is increasing in $b$.
3. Consequently **C3 alone does not separate $\Pi_J$ from $\Pi_D$**. The pair ($r(b)=\mathbf 1\{b>\theta\}$, $\tau(b)=t^{*}(b)$) is $b$-measurable and attains the optimum. What C3 breaks:
   * every decomposition whose timing rule does not see $b$;
   * B9+, whose scalar price $\mathbb E_b[e_{\hat\tau(b)}]$ averages over all beliefs rather than being evaluated at the boundary $\theta$. This gives a Prop. 2-type gap with band between $\theta$ and $\theta_{B9+}$;
   * the v1 price-shift equivalence. If B9's price $\lambda'$ is chosen to match $\theta$, its timing optimizer then runs at $\lambda'\ne\lambda$ and picks a different $t^{*}(b)$.

*Proof.* (1) Topkis' monotonicity theorem; the switch points solve $U_t=U_s$. (2) Positive slopes. (3) Exhibit the $b$-measurable optimal pair; the gaps follow from Prop. 2's argument with $\theta_{B9+}\ne\theta$. ∎

---

## 7. What the theory predicts for v2 (inputs to `PREREG_V2.md`)

| channel set active | Ours-v2 vs B9+ (matched price) | Ours-v2 vs B9 (matched price) | frontiers |
|---|---|---|---|
| none (A1–A3 enforced) | **identical decisions; gap exactly 0** (Props. 1–2) | ≥ 0, bounded by Prop. 2 | identical (Cor. 1) |
| C3 only | ≥ 0, Prop. 2-type band gap; 0 against a $b$-aware decomposed rule | ≥ 0 | may differ |
| C1 | $G\ge0$, **second order** (Prop. 3(6)); grows with $\lambda_l^{2}\,\mathrm{Var}(e^{*})$ and $f(\theta^{*})$; about 0 for low-latency retrieval that is almost always hidden | Prop. 2 gap plus $G$ | Ours ⊇ |
| C1 + C2, commit | first-order gain against the best decomposed rule (Prop. 4) | larger | Ours ⊇ |
| C1 + C2, speculation allowed | ≤ $\lambda_c(1+w_1)$ per turn against SPEC; can be negative for joint-commit (Prop. 4′) | — | — |

Simulated effect sizes (`v2_power.json`; assumptions in Section 4 of `PREREG_V2.md`), in utility units per item, at the dev-selected price:

* **C1 only vs B9+.** At most +0.0021 (d2_support × remote_api, four-price average). About 0 for faiss (the call is fully hidden on 89% of d1_open and 98% of d2_support turns).
* **C1+C2 commit vs the best tuned decomposed rival.** +0.0066 to +0.0148.
* **vs B9 (full-latency pricing).** +0.0095 to +0.0213 (C1 cells, four-price average).

---

## 8. Empirical verification on the v1 E3 sweep

All numbers below are from `docs/research_v2/v1_theory_check.json`; the figure is `docs/research_v2/v1_theory_check.png`.

**Data.** 50 scenarios (25 need retrieval), 5 seeds, 26 grid prices, $\lambda_c=0.05$. Constants as in Section 1.

**Sanity.** The live controllers reproduce all 2,600 logged (controller × price × scenario) decisions (0 mismatches). No decision varies across seeds.

### 8.1 Proposition 1 on the grid

* **Ours$(\lambda_i)$ vs B9$(\lambda_{i-1})$, $i=1..25$.** Retrieve decisions agree on 1,250/1,250. Timing agrees on 941/941 turns where both retrieve.
* **Grid identity.** $\theta_O(\lambda_i)=\theta_B(\lambda_{i-1})$ with maximum absolute error 0.0.
* **Operating points** (Ours$(i)$ vs B9$(i-1)$, 125 per-seed comparisons). Accuracy and compute are identical in 125/125. p95 TTFA is identical in 123/125 as logged (maximum difference 84.15 ms), and in 125/125 once skipped turns are read from one skip cell.
* **Why the two p95 values differ (sweep artifact).** On skipped turns, B9 logs its timing optimizer's choice while Ours logs sentence_end. The cache measured a separate no-retrieval cell per timing label. The fixed_early_proxy minus sentence_end skip-cell TTFA difference is −64.2 ms on average (SD 172.9 ms), pure run-to-run noise. This is not a controller difference. v2 should log `timing = null` on skips, or score skips from one canonical cell.

### 8.2 Proposition 1 off the grid, and the timing interval

* **Dense prices.** 200 prices log-spaced on [0.005, 3.0], 10,000 decisions:
  * exact map: retrieve agreement 10,000/10,000, timing agreement 6,973/6,973;
  * approximate map $\lambda(L-o^{*})/L$: 9,938/10,000; it fails at 46 of the 200 prices, between $\lambda_l=0.0127$ and 1.913.
* **The waste term matters.** The additive term $\lambda_c(w^{*}-w_{\text{se}})/L$ is not negligible: with the placeholder $w^{*}=0.08$ the exact map differs from the approximate one, and the difference matters on 46 of 200 prices.
* **Timing regions.** $t^{*}$ is sentence_end below 0.02, immediate on (0.02, 0.021739), and fixed_early_proxy above 0.021739.
* **Timing-mismatch interval.** Prop. 1(3)'s interval is (0.021739, 0.021919).
* **Where the grid starts.** The grid's first price sits $4.0\times10^{-10}$ above the lower edge of the fixed_early_proxy region, so all shifted grid comparisons fall outside the mismatch interval.
* **Grid recurrence.** Raw ratios $\lambda_i/\lambda_{i-1}$ range from 1.008 to 1.382. The ratio of distances to the fixed point $\lambda_{\deg}=\lambda_cw^{*}/o^{*}=0.021277$ is exactly 1.388430 $=L/(L-o^{*})$ at every step.

### 8.3 The band at every grid price

* **Band mass.** The band $(c_O,c_B]$, computed with `divergence_band()`'s exact cuts, contains at least one scenario at 11 of the 26 prices:
  * 1 scenario at 8 prices, 3 at $\lambda_l=0.476$, 14 at 1.239, 16 at 1.713;
  * 41 scenario-prices in total (3.15% of the 1,300 decisions);
  * each of 41 distinct scenarios falls in exactly one band, as the edge-to-edge tiling implies.
* **Disagreement equals band membership.** At every one of the 26 prices, the set of scenarios on which Ours and B9 (same price) disagree **equals** the in-band set.
* **Where the gap comes from.** At the top price B9 retrieves 0 of 50, because its cut (1.201) exceeds every belief. This is the "disabled, not beaten" point.

### 8.4 Proposition 2: predicted vs. observed gap (Ours − B9, same price)

| quantity | value |
|---|---|
| controller-model gap $\mathbb E[(b-c_O)\mathbf 1_{\mathcal B}]$, maximum over prices | 0.0345 (at $\lambda_l=1.239$) |
| bound $\lambda_lo^{*}\Pr(\mathcal B)$, maximum | 0.103 |
| model gap ≤ $(c_B-c_O)\Pr(\mathcal B)$ ≤ $\lambda_lo^{*}\Pr(\mathcal B)$ at every price | yes |
| mean observed accuracy gap over the 11 band prices | +0.0189 |
| mean predicted accuracy gap, v1 objective ($\sum_{\mathcal B}b$) | +0.0521 |
| mean predicted accuracy gap, corrected objective | +0.0242 |
| Pearson(corrected prediction, observed accuracy gap), 11 prices | 0.975 (true-label prediction: 0.984) |
| prices where the observed accuracy CI excludes 0 | only the top price ($+0.212$, $[0.092,0.340]$) |

The v1 objective predicts a non-negative accuracy gap at every price by construction. Observed, it is −0.020 at five single-scenario band prices: a not-needed scenario with a high belief enters the band, and Ours retrieves on it. The corrected objective predicts a negative gap at four of these five prices (−0.009 to −0.005; +0.003 at the fifth). The true-label prediction is −0.012 at all five.

**Realized utility.** This is the v1 objective's utility, computed with the *measured* TTFA instead of the model's exposed latency. The latency model is badly off:

| | measured (mean over 250 scenario-seed pairs) | controller model |
|---|---|---|
| TTFA increase of retrieving at sentence_end vs skipping | 1.566 s | 0.672 s |
| TTFA increase of retrieving at fixed_early_proxy vs skipping | 1.202 s | 0.484 s |
| overlap saving of fixed_early_proxy vs sentence_end | 0.364 s | 0.188 s |

With the corrected quality model and the measured cost, the predicted realized gap matches the observed one:

* Pearson 0.952 across the 11 band prices, mean absolute error 0.034;
* sign agreement 10/11;
* the observed 95% CI contains the prediction at 8/11 prices.

The v1 controller's own model gets the sign right at 1/11 prices. At the two top prices the observed realized gaps are −0.530 and −0.479, against bounds of 0.065 and 0.103: Remark 2.1 made concrete. Both correlations are dominated by those two top prices, which carry 30 of the 41 in-band scenario-prices; with $n=50$ the mid-grid points are single scenarios.

### 8.5 B9+ reproduces Ours

At every grid price, B9+ (expected-exposure pricing, B9's own timing optimizer) agrees with Ours:

* grid: 1,300/1,300 retrieve decisions, and 982/982 timings where both retrieve;
* dense set: 10,000/10,000 retrieve decisions.

Under A1–A3, the decomposed controller the reviewer describes *is* Ours.

### 8.6 Calibration on v1

* Brier score 0.113.
* By belief bin:

| belief bin | scenarios | need rate | mean retrieval effect |
|---|---|---|---|
| [0, 0.2) | 14 | 0.00 | −0.73 |
| [0.6, 0.8) | 8 | 0.625 | |
| [0.8, 1.0] | 22 | 0.864 | +0.61 |

* 7 not-needed scenarios have $b>0.5$; the highest belief on a not-needed scenario is 0.973.

These high-belief negatives are what enter the band at high prices and drive the miscalibration term of Remark 2.1.

---

## 9. Where the starting framing disagrees with the data

1. **"Ours(λ) ≡ B9(λ·(L−o\*)/L)."** This is exact only when $\lambda_c(w^{*}-w_{\text{se}})=0$. The exact map adds $\lambda_c(w^{*}-w_{\text{se}})/L$ (Prop. 1(4)). The approximate map misclassifies 62 of 10,000 dense-price decisions; the exact map misclassifies none (Section 8.2).
2. **"The geometric grid ratio ≈1.38 equals L/(L−o\*)."** The grid is not geometric. Raw ratios run from 1.008 to 1.382. It is affine-geometric about $\lambda_{\deg}=0.021277$ with ratio exactly 1.388430. More importantly, Ours$(\lambda_i)$ = B9$(\lambda_{i-1})$ is **forced by the grid construction** (`e3_grid.py` tiles bands edge to edge) together with Prop. 1. It is not an independent empirical observation.
3. **"+0.72 / −0.59."** Confirmed (0.720 and 0.592, with wide CIs: [0.504, 0.888] and [0.408, 0.768]), but on 25+25 hotel-domain scenarios. These numbers are dataset-specific and must be re-estimated on v2 dev.
4. **"B9 is a near-strawman."** It is a decomposition with a *pricing* error. Its timing optimizer already reads the belief. The reviewer's B9+ equals Ours on all 11,300 checked decisions (Section 8.5). v2 must compare against B9+, and against SPEC where speculation is allowed.
5. **"C3 makes optimal timing belief-dependent, so decomposition fails."** The first half is true (Prop. 5(1)). The second half is not, in the information sense: the optimum remains $b$-measurable (Prop. 5(3)). C3 only defeats decompositions whose timing ignores $b$, and scalar-priced ones such as B9+.
6. **"C2 (information vs overlap) breaks the equivalence."** Only under the commit architecture, or with expensive compute (Prop. 4′). In simulation, the decomposed SPEC beats joint-commit at most prices in d2_support.
7. **"C1 gives a positive gain where active."** True in expectation, but the gain is second order (Prop. 3(6)). Against B9+ it requires about $2\times10^{4}$ test items in the best simulated cell (`v2_power.json`). C1 cannot be the confirmatory primary at the planned dataset sizes.
8. **"Show the band fraction matches the convergence."** Matches exactly at the decision level (Section 8.3). At the outcome level, the predicted gap matches the observed one only after correcting the objective and **the latency model**. The v1 constants understate the TTFA cost of retrieval by more than 2× (1.20 s vs 0.484 s at fixed_early_proxy). v2 must calibrate $L$ as an end-to-end TTFA increment and separate its non-hideable part $\pi$, which caps C1.
9. **Prop. 2 as "|E[U_Ours] − E[U_B9]| ≤ λ_l o\* P(band)."** Under the model, the gap is signed (≥ 0) and the tighter bound is $(\lambda_lo^{*}-\lambda_c\Delta w)P$. Under misspecification no $O(\lambda_lo^{*})$ bound exists (Remark 2.1), as v1's top prices show.

## 10. References

Howard, R. A. (1966). Information value theory. *IEEE Trans. Systems Science and Cybernetics*, 2(1), 22–26.

Chow, Y. S., Robbins, H., & Siegmund, D. (1971). *Great Expectations: The Theory of Optimal Stopping.* Houghton Mifflin.

Topkis, D. M. (1978). Minimizing a submodular function on a lattice. *Operations Research*, 26(2), 305–321.
