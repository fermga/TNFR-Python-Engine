# Native local recovery and regional interaction

Full-law local recovery, regular-equilibrium stiffness, regional transmission and work accounting; interacting environments remain retained.

Part of [Native relational exchange admission](RELATIONAL_EXCHANGE_ADMISSION.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 15. Local recovery under the complete relational law

<a id="relational-local-recovery"></a>

### Equilibria reuse the existing circulation classification

Fix the same connected simple unit graph, now with **every held capacity
strictly positive** and `e,w,beta>0`. Let `theta_*` have strictly acute edge
gaps and `grad(V_phi)(theta_*)=0`. The equilibrium family is

\[
\mathcal M_*=
\{(x,\theta)=(c\mathbf1,\theta_*+\alpha\mathbf1):c\in\mathbb R,
\ \alpha\in S^1\}.
\]

These are all acute equilibria: the phase row and invertibility of `N,H`
force `Bx=0`, then the form row forces `g=0`. Conversely those two conditions
make both rows vanish. In the acute chart, zero `g` is equivalent to zero
neighbor sine sums. Therefore the integral-cycle circulation classification
in [the existing owner](FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)
applies to the *geometry*, without importing that owner's supplied sine
evolution law or its common-capacity premise. Each admitted winding sector
contains at most one such phase shape up to common rotation. Trees and
graphs whose cycle space is spanned by triangles/quadrilaterals admit only
consensus; a unit cycle admits twists `2*pi*ell/n` with `abs(ell)<n/4`.

The resulting form is uniform, but the phase geometry need not be. In
particular, winding-one C5 is an admissible nonconsensus geometry. Its
existence is inherited from the supplied support and initial sector; it is
not spontaneous formation or a differentiated stationary scalar form.

This also separates geometric identity from scalar summaries: at the exact
winding-one C5 lock, the mean phasor is zero and every local mean absolute
edge gap is `2*pi/5`, while `g=0` and both nodal rows vanish. A zero global
phase-order parameter or a configured gradient warning cannot rule out this
locally recoverable geometry. These observations do not select its dynamics.

### Quotient linearization and the restoring mechanism

Choose an orthonormal matrix `R` spanning `1^perp` and local lifted deviations
`x=c*1+R*u`, `theta=theta_*+alpha*1+R*v`. These are local differences from
the prepared circle-valued phase, not a common semicircle assumption on
`theta_*`. Let `K_*` be the Hessian of `V_phi` there: its edge weights are
`cos(delta_*)>0`. Write

\[
\begin{gathered}
B_q=R^TBR>0,\quad K_q=R^TK_*R>0,\\
A_q=R^TND^{-1}R>0,\qquad M_q=R^TNH_*^{-1}R>0.
\end{gathered}
\]

At equilibrium, differentiating `Hg=-grad(V_phi)` gives
`Dg=-H_*^-1 K_*`. The derivative of `H^-1` contributes nothing to the phase
row there because `Bx=0`. The exact quotient Jacobian is therefore

\[
J_q=\begin{pmatrix}
-eA_qB_q&-wM_qK_q\\
(w/\beta)M_qB_q&0
\end{pmatrix}.
\]

This includes arbitrary positive heterogeneous capacities and irregular
degrees. Passing to storage coordinates
`(B_q^(1/2)*u, sqrt(beta)*K_q^(1/2)*v)` gives the similar matrix

\[
\widetilde J_q=\begin{pmatrix}-\mathsf D&-\mathsf C\\
\mathsf C^T&0\end{pmatrix},\quad
\mathsf D=eB_q^{1/2}A_qB_q^{1/2}>0,\quad
\mathsf C=\frac w{\sqrt\beta}B_q^{1/2}M_qK_q^{1/2}.
\]

The coupling matrix `C` is invertible. For a complex eigenvector `(a,b)`,
the real part of its work identity is
`Re(lambda)*(||a||^2+||b||^2)=-a^* D a`. Thus no eigenvalue has positive
real part. Equality would force `a=0`; the first block would then give
`C*b=0`, hence `b=0`, contradicting an eigenvector. All quotient eigenvalues
have strictly negative real part.

**Conditional local theorem.** Smoothness of the admitted field and this
Hurwitz Jacobian imply local exponential recovery of `M_*`: sufficiently
small joint form/phase perturbations remain acute and their deviations
modulo the two common offsets decay exponentially. A pure phase deformation
can initially have zero storage loss, but it induces form contrast through
native pressure; exchange makes that deformation accessible to form
dissipation. No added phase damping, telemetry selector or restorative
operator is needed under this specified law.

<a id="regular-equilibrium-stiffness"></a>
### Full-network stiffness at a regular equilibrium

The same quotient Jacobian applies at any exact regular equilibrium, with
`e,w,beta>0` and positive held capacities, even when some edge cosines are
negative. Here `K_q` is symmetric but is not presumed positive. Criticality
`g=0` is essential to the identity `Dg=-H^-1*K`; uniform form alone does not
establish it. Eliminating the form component of an eigenvector gives the
symmetric quadratic pencil

\[
\begin{aligned}
P(\lambda)&=\lambda^2\mathsf M+e\lambda\mathsf C+
                (w^2/\beta)K_q,\\
\mathsf M&=M_q^{-1}B_q^{-1}M_q^{-1}>0,\qquad
\mathsf C=M_q^{-1}A_qM_q^{-1}>0.
\end{aligned}
\]

For a nonreal eigenvalue and its phase vector `v`, the imaginary part of
`v^*P(lambda)v=0` gives
`2*Re(lambda)*v^*M*v + e*v^*C*v=0`; hence its real part is strictly negative.
For real `lambda>=0`, `P` increases strictly in the positive-definite order
and becomes positive definite for large `lambda`. Every negative direction
of `K_q` therefore supplies a positive real eigenvalue, counted with
multiplicity. If `K_q>0`, all quotient modes have negative real part, extending
the local recovery theorem beyond the sufficient all-acute edge condition.
If `K_q` has a kernel, those additional zero modes require a separate analysis.

These statements concern the full nodal quotient, including perturbations
that break an imposed reflection. A negative cosine cut sum is a useful
sufficient instability witness: the cut's indicator vector has that Hessian
quadratic value and can be made orthogonal to the common phase by subtracting
its mean. For two joined rings, opposite uniform rotations have stiffness
`8*cos(a-A)` when the matching bridge gaps agree in cosine. A negative value
proves instability at an exact equilibrium; at a transient state it is only
instantaneous phase curvature.

The [reflected equilibrium classification](RELATIONAL_PATTERN_COMPOSITION.md#reflected-regular-equilibria)
uses this full-network result, with independent exact Hessian controls and
the existing materialized uniform-tangent adapter. It does not infer ideal
criticality or an irrational equilibrium from a small floating residual.

### An explicit local domain and the offset limits

An energy barrier supplies a sufficient continuous admission neighborhood.
Let `m=min_edges(pi/2-abs(delta_*))`, choose `0<r<m/sqrt(2)`, and define

\[
c_r=\min_{\{i,j\}\in E}\cos(|\delta_{*,ij}|+\sqrt2r)>0,
\quad k_r=\frac{\lambda_2(B)}2\min(1,\beta c_r).
\]

Within `||z||<=r`, `z=(u,v)`, every interpolated edge gap stays acute because
`abs((Rv)_j-(Rv)_i)<=sqrt(2)*||v||`. Taylor's formula about the critical
phase and the graph spectral gap give

\[
\mathcal E_{rel}=E_D(x)+\beta[V_\phi(\theta)-V_\phi(\theta_*)]
\ \ge k_r\|z\|^2,\qquad
\dot{\mathcal E}_{rel}=-e(Bx)^TND^{-1}Bx\le0.
\]

If `||z(0)||<r` and `E_rel(0)<k_r*r^2`, the trajectory cannot first exit
this ball. The quotient stays in a compact interior sublevel, its metric
remains nonsingular, and the smooth law continues for all future time.
The largest invariant set with zero loss has `Bx=0`. To retain that equality,
`N*g` must be a common vector; `1^T H*g=0` and positive `H,N` force it to be
zero. Sector uniqueness then gives precisely the reference phase shape.
LaSalle's argument consequently gives convergence in this sufficient basin;
the local Hurwitz result supplies exponential convergence near its limit.
These statements do not give a graph-independent decay rate or an error
bound for a numerical integrator.

The common form and phase velocities are smooth functions of the quotient
state and vanish at its equilibrium. Exponential quotient convergence makes
both velocities integrable, so the actual offsets tend to finite limits
`c_infinity, alpha_infinity`. They need not equal their initial values: the
native Arg pressure does not generally conserve a form mean, and the phase
row does not generally conserve a phase mean. The theorem recovers the
prepared geometry modulo those symmetries, not a prescribed absolute offset.
The limiting raw storage is `beta*V_phi(theta_*)`, which is positive for a
nonzero twist; only the local excess storage tends to zero.
Unlike section 3's low-total-storage condition, this local excess-storage
barrier can surround a winding equilibrium whose total storage exceeds beta.

### Cycle rates, competing laws and excluded boundaries

For a unit `C_n` with common capacity `nu>0` and uniform twist
`kappa=2*pi*ell/n`, `K_*=cos(kappa)*B` and `H_*=pi*D*cos(kappa)`. Each
nonzero random-walk Laplacian mode `lambda` satisfies

\[
\boxed{\quad s^2+e\nu\lambda s+
\frac{\nu^2w^2\lambda^2}{\beta\pi^2\cos\kappa}=0.\quad}
\]

At phase consensus this reproduces section 5. With `e=w=1/2`,
`beta=nu=1`, winding-one C5 has damped oscillatory linear modes, whereas
winding-one C6 has two negative real roots per mode. Those are predictions
of this joint law, not the relaxation rates of the earlier supplied sine
law and not a claim about particles or a physical clock.

For positive heterogeneous capacities on the same cycle, replace `nu*lambda`
in this polynomial by an eigenvalue `mu` of
`T=(R^T N R)*B_q/2`. This matrix is similar to a symmetric positive definite
matrix, and `mu_min>=nu_min*lambda_2(B)/2`. All modes still have the same
damped/critical/oscillatory class. For winding-one C5 with `e=w=1/2`,
`beta=1` and `nu_min=3/4`, their real parts satisfy
`Re(s)<=-3*lambda_2(B)/32`. This linear spectral bound does not remove the
nonlinear neighborhood restriction or supply an Euler error estimate.

The section 12 countermodel has the same equilibria and linearization here.
Indeed its skew correction is `Z=S*grad(V_phi)`, with `S=0` at uniform form
and `grad(V_phi)=0` at the phase equilibrium; locally
`Z=O(||u||*||v||)`. Its ideal storage balance is unchanged. Thus it too has
the stated local recovery: stability alone cannot select the independent
capacity premise. The finite capacity-intervention discriminator remains
necessary to separate their nonlinear responses.

The explicit acute-basin theorem has concrete limits. At `e=0` its local
positive excess storage is conserved, so a nonzero quotient perturbation
cannot converge to the reference equilibrium family; with positive quotient
stiffness the linearized nonzero modes are purely imaginary. If
`w=0`, phase is frozen and arbitrary phase damage cannot recover (that
boundary is outside the admitted production model). Zero capacities require
new observability conditions: for example, two inactive nodes with distinct
frozen forms prevent convergence to common form, however small their
initial difference. That explicit basin does not cover nonacute equilibria;
the separate regular-stiffness theorem above supplies a local criterion for
them. Neither result covers changing capacity/support, forcing, operator
events, global attraction across winding sectors or indefinite stability of
finite Euler execution.

### Bounded C5 recovery control and first retained evaluation

The [shared-engine instrument](../../benchmarks/relational_local_recovery.py)
uses one supplied winding-one C5 with `e=w=1/2`, `beta=1` and held capacities
`(1,5/4,3/4,3/2,1)`. At `epsilon=2^-8`, prepare

\[
x(0)=\epsilon(1,-1,0,0,0),\qquad
\theta_i(0)=2\pi i/5+\epsilon(0,1,-1,0,0)_i.
\]

The ideal preparation lies in the sufficient continuous basin *before any
trajectory is computed*. Take `r=pi/(20*sqrt(2))` in the preceding barrier.
Then `c_r=sin(pi/20)>1/10`, `lambda_2(B)=(5-sqrt(5))/2>1`, and
`k_r*r^2>9/16000`. The initial quotient norm is `2*epsilon=1/128<r`.
Using the unweighted graph Hessian as an upper bound on the cosine Hessian
along the initial perturbation gives
`E_rel(0)<=6*epsilon^2=3/32768<9/16000`. This is an ideal continuous
admission argument, not a floating-point trajectory enclosure.

The [prediction](../../docs/assets/relational_local_recovery/result.prediction.json)
was frozen before the two positive-capacity executions. Its finite hypotheses
are: at structural time `T=32`, joint quotient distance is at most half its
initial value on both grids; endpoint grid separation is at most `1/64` of
the initial distance; the acute margin stays at least `pi/20`; actual work
residual stays at most `1e-12`; and support, capacities, pressure path and
winding one are retained. Grids use `dt=1/64` and `1/128`. These tolerances
are numerical acceptance policies, not new dynamics or rigorous ODE error
bounds. The common form and phase offsets are retained separately.

The [response](../../docs/assets/relational_local_recovery/result.json), first
evaluated on 2026-09-27, passed every frozen criterion without retuning:

| Euler steps | Initial joint distance | Final joint distance | Final / initial |
| --- | --- | --- | --- |
| 2048 | `0.0078125` | `3.8753314601e-5` | `0.0049604243` |
| 4096 | `0.0078125` | `3.9089447971e-5` | `0.0050034493` |

The two endpoints differ by `3.9732190974e-7` in the quotient, approximately
`5.086e-5` of the initial perturbation. Both traces retained winding one and
minimum acute margin `0.3102530154` radians. Maximum observed actual work
residual was about `1.008e-18`. No positive storage increment was observed;
nonzero Euler storage defects remain in the record and are distinct from
instantaneous work residuals.

On the fine grid the final form mean is `0.00016167095`, and the common
phase-offset change is approximately `-0.00042482317`. The relative geometry
recovers while these common offsets move, as the theorem permits. Neither
the final distance nor these finite offsets are asserted to be exact limits.
The same perturbed state with all capacities zero stays exactly frozen in
one shared engine step despite nonzero pressure; the declared clock advances.
That control distinguishes inactive retention from restorative dynamics.

The [controls](../../tests/physics/test_relational_local_recovery.py) compare
the production tangent with independently derived cycle blocks, check common
offsets and the zero-capacity boundary, and reuse the saved checkpoints and
decisions. They do not regenerate the full two-grid campaign. The artifact
contains four checkpoints per trace and summaries over every executed step;
saved snapshots alone do not independently authenticate omitted steps.

**Disposition.** The named F3 local-recovery obligation is complete for this
conditional family: a proof covers positive held capacities on the stated
acute equilibria, and the finite control exercises the existing integrated
law on a nonuniform phase geometry. It adds no solver, stabilizing controller
or default policy. Supplied support and winding, formation of that sector,
interaction with other regions and physical identification remain distinct.

## 16. Interaction of two recoverable regions under the same law

<a id="relational-region-interaction"></a>

### Joined support changes the shared geometry, not the evolution rule

Prepare two oriented unit cycles `A=(0,1,2,3,4)` and `R=(5,6,7,8,9)` with
the fixed bridge `(0,5)`, all capacities one and the same `e,w,beta` as
above. For `k=0,...,4`, let `theta_*[k]=theta_*[k+5]=k*kappa`, where
`kappa=2*pi/5`. The bridge endpoints agree. Existing acute circulation
classification requires zero phase gap on a bridge at equilibrium; the
two ring circulations balance independently. Uniform form and these phases
are therefore an equilibrium of the complete joined law.

Only the geometry and balance accounting of the historical
[contact study](../COHERENT_PATTERN_CONTACT.md) are reused. Its supplied
sine-law trajectories and maximum principle do not apply to this joint law.
There is no support event during the experiment, imposed interaction force,
phase driver, selector or retuned pressure. Joining is a supplied preparation.

The joined graph has degrees three at ports 0 and 5, and two elsewhere.
Writing `c=cos(kappa)` and `h=1+2*c=(1+sqrt(5))/2`, its equilibrium phase
metric is `H_ports=pi*h`, `H_other=2*pi*c`. Both native diffusion normalization
and the phase metric must be evaluated on the joined support. An isolated
ring's row cannot be copied to a port after joining.

The disconnected reference admits two independent form offsets and two
independent phase rotations. The connected graph admits only one of each:
relative regional offsets are now part of the dynamical state. Section 15
supplies local recovery of this joined equilibrium modulo the remaining
global symmetries. Prepared regional winding is retained in that local
domain; neither the bridge nor the winding sectors are generated by the law.

### A transmitted response that changes internal geometry

Perturb only an internal left node: `x(0)=epsilon*e_1`, with both phases
unchanged. The receiver's complete form and phase velocities initially vanish.
At the donor port,

\[
\dot x_0(0)=e\epsilon/3,\qquad
\dot\theta_0(0)=-w\epsilon/(\beta\pi h).
\]

At the receiving port `q_5=(Bx)_5=0`, so differentiating `H_5^-1` supplies
no initial term in its phase acceleration. The native pressure derivative
uses `partial_0 g_5=1/(pi*h)` at the prepared lock. Consequently

\[
\boxed{\quad
\ddot x_5(0)=\epsilon\left(\frac{e^2}{9}
                      -\frac{w^2}{\beta\pi^2h^2}\right),\qquad
\ddot\theta_5(0)=-\frac{ew\epsilon}{3\beta\pi h}.
\quad}
\]

All other receiver accelerations vanish initially. The form response contains
competing diffusion and phase-mediated contributions, both from the existing
law. Its sign is not universal over storage scales: for `e=w=1/2`, `beta=1`
the displayed form coefficient is positive and the phase coefficient negative.

On consistent local lifts define the receiver's port-versus-rest contrasts

\[
\chi_x=x_5-\tfrac14\sum_{i=6}^9x_i,\qquad
\chi_\theta=(\theta_5-\theta_{*,5})
             -\tfrac14\sum_{i=6}^9(\theta_i-\theta_{*,i}).
\]

These remove a common regional shift and have the same initial accelerations.
They therefore detect internal deformation, not merely a translated regional
mean. In the ideal disconnected comparison the recipient remains at its
equilibrium for all time. Let `D_chi` be joined recipient minus separately
evolved disconnected recipient; its leading coefficients are
`D_chi(T)=T^2*chi''(0)/2+O(T^3)`. The order is a smooth local response order,
not a finite signal delay or physical propagation speed. Floating-point
drift of the disconnected preparation is measured separately rather than
assigned the ideal zero value.

The ideal preparation with `epsilon=1/256` lies within the sufficient
continuous basin before any simulation. The joined graph has ten nodes and
diameter five. For zero-mean `v`, summing pathwise Cauchy inequalities gives
`10*||v||^2=sum_(i<j)(v_i-v_j)^2<=45*5*v^T Bv`, hence
`lambda_2(B)>=2/45`. With `r=pi/(20*sqrt(2))` and `c_r>1/10`, section 15's
barrier exceeds `pi^2/360000>1/40000`. The preparation has
`E_rel(0)=epsilon^2=1/65536` and quotient norm
`epsilon*sqrt(9/10)<r`. This proves continuous admission and eventual local
recovery, not the finite Taylor band or an Euler error enclosure.

### Regional accounting and the information that cannot be discarded

For the fixed receiver, the form total weighted by full-graph degrees has
the exact balance

\[
M_R=3x_5+2\sum_{i=6}^9x_i,\qquad
\dot M_R=e(x_0-x_5)+w\sum_{i\in R}d_i g_i.
\]

The phase row likewise implies

\[
\sum_{i\in R}H_i(\theta)\dot\theta_i
       =\frac w\beta(x_5-x_0).
\]

The latter is an instantaneous weighted rate, **not** the derivative of a
weighted phase total: `H` changes with phase. The former is an EPI accounting
identity, not conservation of physical mass. It does not discard the internal
phase source or imply that the ordinary arithmetic mean obeys a pure cut-flux
law. The existing
[regional support observer](../../src/tnfr/physics/support_transport.py)
owns the represented form-total and variance decomposition. Its forcing
argument is supplied as `F=w*g` directly from the phase state, independently
of the form rate; native pressure split defects remain separate.

#### Shared signed-work and regional-rate integration

<a id="relational-work-integration"></a>

The engine now retains the exact represented graph gradient `q=Bx` in
`field.work`, alongside the older floating gradient. Using represented
capacity, phase source, phase gradient and materialized model rates, define

\[
D_i=e\nu_iq_i^2/d_i,\qquad J_i=w\nu_iq_i g_i,\qquad
F_i=q_i\dot x_i,\qquad P_i=\beta(\nabla V)_i\dot\theta_i.
\]

In exact real arithmetic the admitted law has `F_i=-D_i+J_i` and `P_i=-J_i`.
In the represented engine these are independently retained defects:

\[
\delta F_i=F_i+D_i-J_i
=q_i(\nu_i\,\epsilon_i+\eta_i),\qquad
\delta P_i=P_i+J_i,
\]

where `epsilon=pressure_split_residual` and
`eta=nodal_rate_rounding_defect`. Thus
`sum_i(delta F_i+delta P_i)=field.balance_residual` exactly. Positive `J_i`
transfers phase storage toward form storage; `D_i>=0` is dissipation.
`P_i` uses the retained trigonometric gradient and represented rate, so its
defect includes their failure to satisfy the ideal metric/source identity.
Exact rational arithmetic here does not certify the trigonometric evaluation
error. The existing global loss and work now sum these same nodal arrays;
neither rates, pressure nor constitutive premises change.

These are nodal contributions to the derivative of **total** storage, not
derivatives of independently assigned node energies. Regional sums retain
that distinction. They sum globally for a partition; overlapping regions
count shared nodes repeatedly. Uniform form has `q=0`, hence zero immediate
work and exchange, even if `g!=0` generates nonzero form rates. Zero work
therefore does not imply equilibrium or absence of a subsequent response.

For arbitrary positive capacities on a supplied region, the two ideal
weighted balances generalize the unit-capacity equations above:

\[
Q_R=\sum_{\substack{i\in R\\j\notin R}}A_{ij}(x_i-x_j)
=\sum_{i\in R}q_i,
\qquad
\sum_{i\in R}\frac{d_i}{\nu_i}\dot x_i
=-eQ_R+w\sum_{i\in R}d_i g_i,
\]
\[
\sum_{i\in R}\frac{H_i}{\nu_i}\dot\theta_i=\frac w\beta Q_R.
\]

The shared support owner computes the outward cut once for both balances.
The actual represented form rate includes `sum_R d_i*epsilon_i` and
`sum_R (d_i/nu_i)*eta_i`; subtracting them gives an exact zero accounting
residual. The actual weighted phase rate retains its difference from
`(w/beta)*Q_R`. Its weights use the captured `H`, so it remains an
instantaneous rate, not the derivative of a weighted phase total.
For the full support the cut is empty. Zero diffusion weight removes the
form boundary term without removing phase exchange. A zero capacity inside
the region makes divided rates unavailable; all nodal work and undivided
cut/model terms remain defined. A zero capacity outside the selected region
does not invalidate its divided rates. These local requirements are separate
from the older regional variance observer's positive-full-capacity contract.

[`observe_relational_pattern`](../../src/tnfr/physics/relational_observations.py)
captures one fresh field, reuses this cut owner and exposes regional `work`
and `boundary` through `Network.relational_pattern` and exact JSON projection.
No diagnostic chooses an operator, evolves a graph, or closes regional
dynamics. The [API contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-pattern-observation)
owns field names and availability; static analytic, sign-reversal, exact
gradient and retained interaction checks exercise the shared implementation
without rerunning a trajectory. This integration adds explanatory accounting,
not new evidence for physical identity or another constitutive law.

A concrete obstruction prevents replacing each ring by only its mean and
storage. Move the same impulse from node 1 to node 2. The donor mean,
form norm, Dirichlet storage, phase geometry, capacities and port form are
unchanged. Both receiver accelerations are now zero, because the donor port
has zero initial velocity in both rows. Thus identical aggregate observations
can have different transmitted responses. A reduced interaction description
must retain sufficient internal/port information or justify an appropriate
memory term; this example does not prove that every reduction is impossible.
The two donor mean derivatives already differ (`-e*epsilon/30` versus zero),
so a description that retains those rates has additional information. The
obstruction concerns the stated instantaneous means/storage, not every
augmented observation or a sufficiently resolved boundary history.

<a id="relational-local-composition"></a>

The [local composition theorem](RELATIONAL_PATTERN_COMPOSITION.md) now fixes
six regional mean/port observations on this same single-bridge equilibrium.
They require ten coordinates for exact first-variation closure; an explicit
mixed form/phase counterexample excludes nonlinear closure of that minimal
linear observation. The theorem owns its symmetry, rank and coefficient
scope; it does not replace the full nonlinear engine state.
Its [nonlinear regional budget](RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance)
identifies the missing mobility/form covariance: even the same zero outward
cut can coexist with different mean phase response. The shared observer
retains that contribution, its squared bound and actual rate-rounding defect.

### Prospective finite control

The [instrument](../../benchmarks/relational_region_interaction.py) keeps
`epsilon=1/256`, `e=w=1/2`, `beta=1`, and structural horizon `T=1/16`.
It compares the joined graph with both disconnected cycles executed separately
through the same connected-graph engine owner, at 64, 128 and 256 steps.
The prior criteria require each signed contrast divided by its quadratic
prediction to lie in `[3/4,5/4]`; successive signal differences must contract
by at least the declared factor `3/4` plus `1e-12`, and each finest signal
must exceed `32*(last_grid_difference+1e-12)`. Winding one, an acute margin
of at least `pi/20`, held support/capacities and work residual at most `1e-12`
are separate checks. Disconnected receiver drift is bounded by the separately
declared `1e-12` numerical gate.

These are prospective numerical hypotheses, not a proved Taylor remainder,
an autonomous regional closure or empirical physical evidence. The complete
prediction must be frozen before finite-horizon campaign execution; a failed criterion
remains a failed result. No prior recovery or capacity-response artifact is
changed by this control.

### First retained transmission evaluation: 2026-09-27

The [frozen prediction](../../docs/assets/relational_region_interaction/result.prediction.json)
preceded every finite-horizon campaign trace. Detached field/Jacobian checks
and a one-step Euler implementation control preceded that freeze; none
evaluated the reserved finite-horizon response. The
[retained response](../../docs/assets/relational_region_interaction/result.json)
contains nine traces (three scenarios on three grids), with initial, midpoint
and final checkpoints, runtime/source provenance and numerical decisions.
All five gates passed on the first evaluation, without retuning:

| Finest-grid contrast, joined recipient minus disconnected recipient | Observed response | Ratio to quadratic prediction | Last grid difference |
| --- | --- | --- | --- |
| Form | `1.3432193095e-7` | `0.9725660824` | `5.0174490386e-10` |
| Relative phase error | `-1.2133275087e-7` | `0.9700781649` | `4.5079212794e-10` |

The successive differences approximately halved. Every trace retained its
declared support, unit capacities and winding; minimum acute margin was
`0.3139754258` radians and maximum actual work residual was `6.661e-19`.
The largest disconnected-recipient full-state drift was `2.769e-18`; it
was subtracted as observed rather than assigned zero. These finite checks
are neither a full continuous error enclosure nor independent validation
of every omitted step from the retained checkpoints alone.

At the fine-grid endpoint, the receiver's degree-weighted form-total rate
decomposes into positive cut contribution `1.9770232650e-5`, negative phase
source contribution `-6.8497774475e-6` and native pressure-split defect about
`4.203e-17`. The existing regional observer checks the exact bookkeeping on
represented values. This is a concrete balance of the two transmission
channels, not a physical-mass measurement or a new regional evolution law.
The ordinary form mean (`2.7112836143e-8`) and full-graph transport-weighted
mean (`3.6881703848e-8`) differ and retain their distinct meanings.

The [controls](../../tests/physics/test_relational_region_interaction.py)
independently check initial accelerations, the equal-summary obstruction,
regional accounting, first-step Euler ordering and frozen-protocol admission.
They bind saved preparations and capacities, replay 27 detached checkpoint
fields and reconstruct the finite contrasts without repeating the trajectories.

**Interaction and formation have different scope.** Recovery and transmission
use one unchanged conditional law. Acute continuous fixed-support execution
preserves cycle winding, so this interaction experiment does not create its
supplied winding sectors. The [regular-domain analysis](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-domain-admission)
separates that execution restriction from genuine constitutive singularities;
the [validated transit](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit) supplies the separate
prepared formation result. Neither result selects the initial support or law.
