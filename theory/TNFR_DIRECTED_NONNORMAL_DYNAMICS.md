# TNFR Directed and Non-Normal Dynamics (R9)

**Status:** exact finite-dimensional results are available for the fixed linear
pure-EPI channel; sampled gains and benchmark correlations remain finite-family
evidence. The canonical directed U2 metric, nonlinear operator extension,
changing-topology theory, and universal U6 trajectory bound remain open.

**Modules:**
[directed_diffusion.py](../src/tnfr/physics/directed_diffusion.py),
[transient_u2.py](../src/tnfr/physics/transient_u2.py),
[nonnormal_prediction.py](../src/tnfr/physics/nonnormal_prediction.py),
[heterogeneous_vf.py](../src/tnfr/physics/heterogeneous_vf.py), and
[structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py).

## 1. Declared transport model

For finite nonnegative outgoing conductance, the EPI-only pressure is

$$p=\Delta\mathrm{NFR}_{\mathrm{epi}}=-L_{\mathrm{rw}}x,
\qquad L_{\mathrm{rw}}=I-P,$$

where positive rows of $P$ are normalized by outgoing strength and a
zero-strength row is absorbing. The fixed scalar-capacity evolution is

$$\dot x=-\nu_f L_{\mathrm{rw}}x.$$

This is one channel of the canonical pressure, on a fixed graph. It does not
cover phase, frequency, topology, nonlinear operator gains, or graph mutation.
Even for symmetric conductance, the raw random-walk matrix is generally only
*similar* to a symmetric normalized Laplacian; Euclidean normality is a
separate property.

## 2. Normality, spectrum, and metric

A non-normal generator can have a nonpositive spectral abscissa while its
Euclidean semigroup norm grows transiently. Consequently:

- `eigh` is used only for symmetric/Hermitian representations;
- right eigenvectors are not converted into projectors as $QQ^*$;
- Schur/Riesz methods or direct semigroup and resolvent diagnostics are used;
- the Kreiss resolvent quantity is a lower bound on peak semigroup gain, not an
  equality or an infinite-window measurement.

Directed circulants are normal and form a benign special case. General
directed graphs need not be normal.

For a strongly connected row-stochastic $P$ with stationary distribution
$\pi$, Jensen's inequality gives

$$\|Pf\|_{2,\pi}\leq\|f\|_{2,\pi},\qquad
\|f\|_{2,\pi}^2=\sum_i\pi_i f_i^2.$$

Because $e^{-tL}=e^{-t}\sum_{k\geq0}t^kP^k/k!$, the diffusion semigroup is a
contraction in $L^2(\pi)$. This exact weighted result does not make the raw
Euclidean norm canonical for U2. The APIs expose both metrics.

## 3. Exact pressure-transient criterion

For fixed $L=L_{\mathrm{rw}}$,

$$p=-Lx\quad\Longrightarrow\quad \dot p=-Lp.$$

When the graph has one consensus mode, $\operatorname{range}(L)$ is the
non-consensus pressure space $\{p:\pi^Tp=0\}$. Let $V$ be an orthonormal basis
of that space and $L_{\mathrm{sub}}=V^TLV$. Every vector in this restricted
space is a reachable pressure. In the declared Euclidean metric, the
logarithmic norm

$$\mu_2(-L_{\mathrm{sub}})=
\lambda_{\max}\!\left(\frac{-L_{\mathrm{sub}}-L_{\mathrm{sub}}^T}{2}\right)$$

has an exact qualitative meaning:

- $\mu_2\leq0$ implies contraction for every time;
- $\mu_2>0$ implies that some reachable pressure direction grows immediately.

A negative spectral abscissa gives asymptotic decay and cannot exclude the
second case. A fixed weighted counterexample in the test suite disproves the
older conjecture that every random-walk digraph contracts in unweighted
Euclidean pressure energy.

`nonnormal_prediction.py` applies this theorem to a predeclared deterministic
family. Its peak gains, rank correlations, and calibration/holdout accuracies
are measurements of that finite family. The sign theorem itself is an exact
matrix result and uses no fitted threshold.

## 4. U2 readings

The implementation distinguishes two finite-window quantities:

$$\left\|\int_0^T\dot x\,dt\right\|
=\|x(T)-x(0)\|,
\qquad
\int_0^T\|\dot x\|\,dt.$$

The first allows cancellation; the second is total variation, and the triangle
inequality gives `net <= total`. Neither a sampled finite window nor a stable
full-state spectrum alone proves an infinite-horizon U2 theorem. On the
non-consensus subspace, a declared exponential envelope

$$\|e^{-sL}Q\|\leq M e^{-\omega s}$$

yields the sufficient bound

$$J\leq \frac{M\|LQ\|\|x_0\|}{\omega}.$$

The bound is restricted to the fixed linear EPI channel and its stated norm.
Choosing the canonical directed U2 metric and extending the result to the full
four-channel dynamics remain open.

## 5. U6 scope

Canonical U6 compares structural-potential fields from two declared graph
states using the centralized $\pi/2$ drift policy. The sampled quantity

$$\max_{s,i}|\Phi_s(s)_i|$$

is an absolute magnitude, not a U6 drift. `TransientU2Certificate` therefore
reports it as `peak_structural_potential_magnitude`, exposes its operator-norm
bound, and sets `u6_drift_assessed=False`. The old
`peak_structural_potential`, `structural_potential_bound`, and `u6_confined`
names remain compatibility aliases with explicit legacy semantics.

`finite_schedule_readout` can evaluate mean potential drift from the initial
state at declared segment endpoints. It still does not cover values between
endpoints or an unobserved tail.

## 6. Structural time and heterogeneous capacity

For a common scalar schedule $\nu_f(t)\geq0$ on a fixed graph,

$$x(t)=e^{-s(t)L}x_0,\qquad s(t)=\int_0^t\nu_f(\tau)\,d\tau.$$

This is an exact clock change because all generators are scalar multiples of
one fixed $L$. Total variation expressed in structural time is invariant under
that reparameterization, subject to the stated integral and decay hypotheses.

For heterogeneous capacity $D_{\nu_f}(t)$, generators
$D_{\nu_f}(t)L$ generally do not commute, so one scalar clock does not describe
the trajectory. Several useful results survive:

1. A frozen nonnegative generator $-D_{\nu_f}L$ is Metzler with zero row sums,
   so its semigroup preserves the scalar convex hull.
2. For fixed positive capacity on a strongly connected graph, the invariant
   measure is proportional to $\pi_i/\nu_{f,i}$.
3. On fixed symmetric conductance $B=D-W$, the Dirichlet energy
   $E_D=x^TBx/2$ obeys
   $$\dot E_D=-(Bx)^TM(t)(Bx)\leq0,
   \qquad M_{ii}(t)=\nu_{f,i}(t)/d_i,$$
   for finite nonnegative time-varying capacity. Uniformly positive mobility
   plus a divergent mobility clock gives a sufficient consensus bound.
4. For switching among symmetric fixed-node regimes, the exact common-metric
   theorem applies only when their normalized diagonal metrics agree up to a
   positive scalar. Without that condition the provided certificate reports
   that the theorem is unavailable; it does not infer instability.

These statements do not create a single conserved mean for arbitrary changing
capacity ratios, and they do not cover changing node sets.

## 7. Claim ledger

| Claim | Status |
|---|---|
| Directed circulants are normal | Exact algebraic property; numerically checked |
| General non-normal stable generators may amplify in Euclidean norm | Exact possibility; finite witnesses measured |
| Diffusion contracts in stationary $L^2(\pi)$ | Derived by Jensen |
| $p=-Lx$ implies $\dot p=-Lp$ | Exact for fixed linear pure-EPI flow |
| Sign of $\mu_2(-L_{\mathrm{sub}})$ classifies possible Euclidean pressure gain | Exact finite-dimensional theorem |
| Spectral abscissa alone classifies transient pressure gain | False; counterexamples retained |
| Every random-walk digraph contracts in restricted Euclidean pressure energy | False; weighted counterexample retained |
| Scalar $\nu_f(t)$ is a clock change | Exact on a fixed graph |
| General heterogeneous $\nu_f(t)$ is one scalar clock | False when generators do not commute |
| Fixed-symmetric Dirichlet energy decreases under nonnegative heterogeneous capacity | Exact instantaneous identity |
| Absolute $|\Phi_s|$ magnitude is U6 drift | False; compatibility aliases are marked |
| Canonical directed U2 metric and full nonlinear bound | Open |
| Universal U6 interval/tail safety from finite samples | Open |

No result here changes U1-U6 or resolves an external open mathematical problem.
