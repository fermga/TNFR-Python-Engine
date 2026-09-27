# Predictive local state of two interacting coherent regions

**Status:** exact conditional first-variation closure, nonlinear state and
state-plus-rate counterexamples, and a regional response budget identifying
their missing geometric contribution. These results use the admitted
relational law; they add no pressure, phase, capacity or support rule. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next task. Formation, local recovery and the supplied interaction
preparation remain in [relational exchange](RELATIONAL_EXCHANGE_ADMISSION.md).

## 1. Fixed model and observation

Use the existing single-bridge preparation: unit cycles `(0,1,2,3,4)` and
`(5,6,7,8,9)`, bridge `(0,5)`, held unit capacities, uniform form and
`theta_*[i]=theta_*[i+5]=2*pi*i/5` for `i=0,...,4`. Both rings have winding
one. This is the earlier interaction graph, not the two-bridge formation
graph. The reference coefficients are `e=w=1/2`, `beta=1`; the calculation
below allows `e>=0`, `w,beta>0`, without extending the recovery theorem to
its lossless boundary.

Put `kappa=2*pi/5`, `c=cos(kappa)=(sqrt(5)-1)/4>0` and `h=1+2c`.
For form deviations `u` and local lifted phase deviations `v`, reuse the
[proved equilibrium derivative](RELATIONAL_EXCHANGE_ADMISSION.md#quotient-linearization-and-the-restoring-mechanism):

\[
J=\begin{pmatrix}-eD^{-1}B&-wH^{-1}K\\
(w/\beta)H^{-1}B&0\end{pmatrix},\qquad
H=\pi\operatorname{diag}(h,2c,2c,2c,2c,h,2c,2c,2c,2c).
\]

Here `B` is the unit-support Laplacian, `D` its full degrees (three at ports,
two elsewhere), and `K` the phase Hessian with ring weights `c` and bridge
weight one. This matrix is the exact mathematical Jacobian at the ideal
equilibrium; a rounded phase preparation need not be an exact equilibrium
of materialized arithmetic.

For either species `z`, let `m_L,m_R` be its arithmetic ring means. Fix the
three observations

\[
O_3z=(\mu,p_L,p_R)
=(m_L-m_R,\ z_0-m_L,\ z_5-m_R).
\]

The six-output observation is `O=diag(O_3,O_3)` on `(u,v)`. It retains
relative regional means and both port contrasts while discarding the two
common offsets. The means are declared observations, not transport-weighted
conserved totals or a redefinition of primitive EPI/phase. The full tangent
state has 20 coordinates, or 18 after those common symmetries.

## 2. Six observed coordinates do not close

Define hidden regional form shapes

\[
\eta_L=(0,1,-1,-1,1,0,0,0,0,0)^T,\qquad
\eta_R=(0,0,0,0,0,0,1,-1,-1,1)^T.
\]

They have zero mean and zero port value, so `O_3*eta_L=O_3*eta_R=0`.
Nevertheless, for `(u,v)=(eta_L,0)`,

\[
O_3\dot u=(-e/15,\ 11e/15,\ 0),
\]
\[
O_3\dot v=
\left(\frac{w}{5\beta\pi ch},\
-\frac{w(10c+1)}{5\beta\pi ch},\ 0\right).
\]

The zero state and this hidden state therefore have identical outputs but
different output derivatives. Scaling the hidden state makes the witness
arbitrarily small. No matrix `G` can satisfy `O*J=G*O`. The phase response
already excludes closure even if `e=0`; this is not solely a diffusion effect.

## 3. Ten coordinates are necessary and sufficient at first variation

Independently reflect each ring through its port: swap `1<->4,2<->3`, or
`6<->9,7<->8`. The matrices `B,D,H,K`, hence `J`, commute with these
reflections. For each species the even subspace consists of ring values
`(a,b,d,d,b)`, dimension six over both rings. Removing its common offset
leaves five coordinates. The odd subspace has four coordinates per species.
Its eight joint directions are unobserved and remain unobserved under `J`.
Together with the two global offsets they bound the observable rank by ten.

On the even quotient, `ker O` consists precisely of `eta_L,eta_R` independently
in form and phase: four hidden coordinates. Observe just the derivatives of
the two form-port and two phase-port contrasts. Their matrix on those four
directions is

\[
\begin{pmatrix}
a&0&b&0\\0&a&0&b\\-d&0&0&0\\0&-d&0&0
\end{pmatrix},\qquad
a=\frac{11e}{15},\quad
b=\frac{w(10c+1)}{5\pi h},\quad
d=\frac{w(10c+1)}{5\beta\pi ch}.
\]

Its determinant is

\[
b^2d^2=
\frac{w^4(10c+1)^4}{625\beta^2\pi^4c^2h^4}>0.
\]

Thus one derivative recovers all four missing even directions:

\[
\operatorname{rank}O=6,\qquad
\operatorname{rank}\begin{pmatrix}O\\OJ\end{pmatrix}=10,\qquad
\operatorname{rank}\begin{pmatrix}O\\OJ\\OJ^2\end{pmatrix}=10.
\]

The last equality follows from the invariant ten-dimensional annihilator
of odd modes and offsets, not from a floating rank threshold. Any autonomous
linear state retaining `O` must retain `OJ`, hence needs at least ten
coordinates. Their invariant span supplies a sufficient state. Minimality
concerns linear observations retaining these fixed outputs for every tangent
state, not arbitrary nonlinear encodings or selected trajectories.

A natural choice adds two pair contrasts per species:

\[
s_L=(z_1+z_4-z_2-z_3)/2,\qquad
s_R=(z_6+z_9-z_7-z_8)/2,
\]

giving `C_5z=(mu,p_L,p_R,s_L,s_R)` and `C=diag(C_5,C_5)`.
A right inverse `T` chooses means `m_L=mu/2,m_R=-mu/2` and lifts each ring as

\[
z_{\rm port}=m+p,\quad
z_{\rm near}=m-p/4+s/2,\quad
z_{\rm far}=m-p/4-s/2.
\]

Then `C*T=I`, `C*J=(C*J*T)*C` and `O=(O*T)*C`.
The effective state consists of internal shape and relative-offset
coordinates. It is not one scalar EPI/phase pair per ring and is not already
an emergent canonical node.

## 4. The same ten coordinates fail as a nonlinear closed state

The distinction matters even arbitrarily close to equilibrium. Choose
`u=epsilon*eta_L`, `epsilon>0`, with right-ring form zero. Compare

\[
\theta^A=\theta_*,\qquad
\theta^B=\theta_*+t(e_1-e_4),\qquad 0<t<\pi/10.
\]

Here `t` labels a supplied phase displacement, not elapsed model time.
All ten natural observations agree: the phase perturbation changes neither
ring mean, port value nor pair average. Both states remain strictly acute.
The form gradient at port zero is `q_0=-2*epsilon`; all right-ring form
gradients vanish, so all right-ring phase rates vanish. The relative
resultant at port zero is positive real in both states, but its phase metric
changes from `pi*h` to `pi*[1+2*cos(kappa+t)]`.

The observable combination `mu_v+p_L_v=v_0-mean_R(v)` has derivative
`theta_dot_0-mean_R(theta_dot)`: changing from deviations to the displayed
absolute phase lifts adds only a fixed reference constant. Its exact rate
difference is

\[
\Delta\dot y=-\frac{2w\epsilon}{\beta\pi}
\left[\frac1{1+2\cos(\kappa+t)}-\frac1{1+2\cos\kappa}\right]<0.
\]

This is nonzero for arbitrarily small positive `epsilon,t`, with expansion

\[
\Delta\dot y=-\frac{4w\epsilon\sin\kappa}{\beta\pi h^2}\,t
+O(\epsilon t^2).
\]

The omitted odd phase geometry modifies the already existing resultant
metric; its interaction with form changes the observed model rate. No new
force, parameter or pressure channel is needed. Consequently no autonomous
nonlinear function of this `C` alone reproduces all nearby output rates.
Every minimal ten-dimensional linear realization has this same row space
and therefore the same failed fibers. A different nonlinear observation,
additional state or an explicitly derived memory remains a separate question.

This is not a claim that every pure odd perturbation becomes visible. A
special pure-odd family can remain invariant through combined sign/reflection
symmetry. The counterexample uses mixed even form and odd phase. Ordinary
graph reflection also reverses the prepared winding, so graph relabeling
alone cannot justify discarding perturbations around this fixed lock.

## 5. Nonlinear regional response from the existing phase metric

<a id="regional-phase-mobility-balance"></a>

### The unweighted rate retains geometry that the cut alone loses

Return to any admitted fixed simple unit support, with held nonnegative
capacities and available positive phase metrics `H_i`. The connected model's
phase row is already

\[
\dot\theta_i=k a_i q_i,\qquad
q=Bx,\quad a_i=\nu_i/H_i,\quad k=w/\beta>0.
\]

The mobility `a` combines supplied capacity with the phase-dependent metric;
it is a derived coefficient, not another primitive variable or an independently
selected feedback. These are consequences of the specified relational closure,
not a proof that the nodal identity uniquely selects that closure.

For a fixed nonempty region `R`, let `n=|R|` and let the outward cut retain
the full graph's neighbors. Symmetry cancels its internal form differences:

\[
Q_R=\sum_{i\in R}q_i
=\sum_{\substack{i\in R,\ j\notin R\\\{i,j\}\in E}}(x_i-x_j).
\]

Define population averages and covariance, including the divisor `n`:

\[
\bar a_R=\frac1n\sum_Ra_i,\qquad
\bar q_R=Q_R/n,\qquad
\operatorname{Cov}_R(a,q)
=\frac1n\sum_R(a_i-\bar a_R)(q_i-\bar q_R).
\]

Expanding the centered product gives the exact identity

\[
\boxed{\quad
S_R:=\sum_R\dot\theta_i
=k\left[\bar a_R Q_R+n\operatorname{Cov}_R(a,q)\right].
\quad}
\]

The arithmetic regional mean rate is `S_R/n`. Replacing each `a_i` by its
regional mean discards the displayed covariance; it is exact only when that
term vanishes at the state in question. This does not make its future value
predictable from the regional mean or cut. Unlike the already implemented
[weighted phase-cut balance](RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration),
the arithmetic rate retains correlation between local form response and
local phase mobility.

### An edge identity separates internal and boundary contributions

Pair the two orientations of each internal edge, counting that edge once.
The same quantity satisfies the discrete Green identity

\[
\sum_Ra_iq_i
=\sum_{\{i,j\}\in E(R)}(a_i-a_j)(x_i-x_j)
+\sum_{\substack{i\in R,\ j\notin R\\\{i,j\}\in E}}
a_i(x_i-x_j).
\]

Consequently

\[
n\operatorname{Cov}_R(a,q)
=\sum_{\{i,j\}\in E(R)}(a_i-a_j)(x_i-x_j)
+\sum_{\substack{i\in R,\ j\notin R\\\{i,j\}\in E}}
(a_i-\bar a_R)(x_i-x_j).
\]

The correction contains both internal mobility/form alignment and variation
of mobility along the boundary. It is therefore not generally a purely
internal source. Both contributions are signed; neither is a new pressure
channel, an assigned regional energy derivative or a dissipation theorem.
The shared cut owner supplies the existing outward orientation; no induced
subgraph normalization or second definition of the cut is needed.

### Exact represented accounting and a bound without a chosen tolerance

For a captured engine field, hats below denote exact rational readings of
materialized scalars. Retain `hat q=B*hat x` from the exact work owner before
its separate float materialization; compute `hat a_i=hat nu_i/hat H_i` and
`hat k=hat w/hat beta` in rational arithmetic. Let `hat r_i` be the captured
phase rate. The represented-rate residual is

\[
\delta_R=\sum_R(\widehat r_i-\widehat k\widehat a_i\widehat q_i),
\qquad
\widehat S_R=\widehat k
\left[\overline{\widehat a}_R\widehat Q_R
+n\operatorname{Cov}_R(\widehat a,\widehat q)\right]+\delta_R.
\]

This equality retains actual rate rounding rather than declaring it zero.
It does not bound trigonometric materialization errors in `H`, errors relative
to the ideal irrational equilibrium, or errors of an evolved trajectory.
The read-out is a model-rate observation at one captured state, not a measured
temporal derivative or evidence that a step occurred.

With population variance `Var_R(a)=sum_R(a_i-abar_R)^2/n`, Cauchy--Schwarz
gives

\[
\left[n\operatorname{Cov}_R(a,q)\right]^2
\le n^2\operatorname{Var}_R(a)\operatorname{Var}_R(q).
\]

Thus the exact represented report can check the squared bound

\[
\left[\widehat S_R-\delta_R
-\widehat k\overline{\widehat a}_R\widehat Q_R\right]^2
\le\widehat k^2n^2
\operatorname{Var}_R(\widehat a)\operatorname{Var}_R(\widehat q).
\]

No tolerance, fitted coefficient or rounded square root is needed. Uniform
mobility or uniform `q` makes the correction vanish instantaneously; zero
covariance can also occur without either uniformity. The bound still consumes
the internal state, so it does not supply an autonomous reduced law.

Zero capacity gives `a_i=0` without division by capacity; it does not remove
the phase-domain requirement `H_i>0`. For a singleton region the covariance
is zero. For full support the outward cut is empty and `Q_R=0`, but
`S_R=k*n*Cov_R(a,q)` may be nonzero. Common phase rotation is a symmetry,
not a proof of conservation of the arithmetic phase sum. State-dependent
weighted phase rates likewise do not integrate to a conserved weighted phase
total without additional terms. These signed instantaneous rates establish
neither persistent drift nor a monotone structural clock. Phase sums along a
trajectory require continuous lifts; branch jumps in stored wrapped angles
are not the derivatives in these identities.

### The same ten-coordinate witness changes the regional mean response

In section 4's paired-ring witness the complete left form gradient is
`q_L=epsilon*(-2,3,-2,-2,3)`. Both bridge endpoints have form zero, hence
`Q_L=Q_R=0` in both states, and all right phase rates vanish. With
`sinc(t)=sin(t)/t`, continuously extended by `sinc(0)=1`, the left metrics are

\[
H_0=\pi[1+2\cos(\kappa+t)],\qquad
H_1=H_4=2\pi c\operatorname{sinc}(t),
\]
\[
H_2=H_3=2\pi\cos(\kappa-t/2)\operatorname{sinc}(t/2).
\]

These follow directly from each neighbor-phasor resultant. They give

\[
\overline{\dot\theta}_L(t)=\frac{k\epsilon}{5\pi}F(t),\qquad
F(t)=-\frac2{1+2\cos(\kappa+t)}
+\frac3{c\operatorname{sinc}(t)}
-\frac2{\cos(\kappa-t/2)\operatorname{sinc}(t/2)}.
\]

The formula holds throughout the stated acute interval. The exact relation
`c*(1+2c)=1/2` yields

\[
F(0)=2,\qquad
F'(0)=\sin\kappa\left(\frac1{c^2}-\frac4{h^2}\right)
=4\sqrt5\sin\kappa>0.
\]

Here the derivative of `F` is with respect to the supplied phase displacement,
not evolution time. Consequently

\[
\Delta\overline{\dot\theta}_L
=\frac{4k\epsilon\sqrt5\sin\kappa}{5\pi}\,t+O(\epsilon t^2)>0
\]

for sufficiently small positive `t` and `epsilon>0`. This local sign conclusion
does not assert a sign theorem over the whole acute interval. The port-rate
change from section 4 is negative, while this regional mean-rate change is
positive. Identical ten-coordinate observations and identical zero cuts thus
coexist with different aggregate responses. Here the boundary current itself
vanishes, so the entire correction is internal. The existing geometric
mobility explains the missing response without completing a nonlinear closure
or introducing a new evolution mechanism.

## 6. The ten coordinates and their current rates still do not close

<a id="state-rate-predictivity"></a>

Let `F(z)` denote the complete admitted vector field on form and local phase
deviations, and keep the same linear observation `C`. The proposed augmented
observation is

\[
Z(z)=(Cz,CF(z)),\qquad
\dot Z(z)=(CF(z),C\,DF(z)F(z)).
\]

Its second block is a model-derived rate at a supplied full state, not an
independently measured history. A closed autonomous law for `Z` on an open
neighborhood would require the displayed derivative to agree whenever `Z`
agrees. The following exact pair violates that condition arbitrarily close
to the prepared equilibrium.

### Hidden form orientation changes the metric's next response

Put `s=sin(kappa)>0` and introduce the left reflection-odd form direction

\[
\zeta_L=(0,1,0,0,-1,0,0,0,0,0)^T.
\]

Keep phases exactly `theta_*` in both states and choose

\[
x^+=\epsilon\eta_L+\delta\zeta_L,\qquad
x^-=\epsilon\eta_L-\delta\zeta_L,\qquad \epsilon,\delta>0.
\]

Their right forms vanish; in deviation coordinates `z^+=(x^+,0)` and
`z^-=(x^-,0)`. At this phase lock `g=0`, so both vector-field
rows depend exactly linearly on form:

\[
\dot x=-eD^{-1}Bx,\qquad \dot\theta=kH^{-1}Bx.
\]

The two forms differ by a reflection-odd direction, and these fixed linear
maps preserve that parity. Consequently `Cz^+=Cz^-` and `CF(z^+)=CF(z^-)`
exactly: their complete augmented observations `Z` coincide.

Their left gradients are

\[
q_L^\pm=\epsilon(-2,3,-2,-2,3)
\ \pm\ \delta(0,2,-1,1,-2),
\]

while `q_R=0`. In particular `q_0=-2*epsilon` and `H_0=pi*h` coincide.
At the lock the relative resultant is positive real. Since the derivative
of `sinc` vanishes at zero, differentiation of the existing phase metric
at any node gives

\[
\dot H_i=-\pi\sum_{j\sim i}
\sin(\theta_{*,j}-\theta_{*,i})(\dot\theta_j-\dot\theta_i).
\]

The bridge contributes zero sine at the port. Thus

\[
\dot H_0=-\pi s(\dot\theta_1-\dot\theta_4),\qquad
\dot H_0^\pm=\mp\frac{2k\delta s}{c},
\]

because `H_1=H_4=2*pi*c`. The form row gives
`q_dot=-e*B*D^-1*q` at this state; its port value is the same
`q_dot_0=5*e*epsilon` for both preparations. Differentiating the phase row
therefore yields the exact difference

\[
\Delta\ddot\theta_0
=-\frac{kq_0}{H_0^2}\left(\dot H_0^+-\dot H_0^-\right)
=-\frac{8k^2\epsilon\delta s}{\pi^2ch^2}<0.
\]

Every right-ring phase acceleration agrees: `q_R=0`, the change of form
velocity at the donor port is zero, and hence the change of `q_dot_R` is
zero. In particular the receiving port acceleration is
`-2*e*k*epsilon/(3*pi*h)` in both states, and the other right accelerations
vanish. The retained observable `mu_v+p_L_v` consequently has exactly the
nonzero acceleration difference displayed above. Both supplied states lie in the
strictly acute domain, and choosing `epsilon,delta` arbitrarily small places
the pair in every open neighborhood of the equilibrium. No trajectory,
Taylor remainder estimate or rank threshold is needed for this obstruction.

### Equal scalar budgets do not remove the ambiguity

The two forms are reflections of one another on the left ring, while the
primitive phase winding is held fixed. Their Dirichlet form energies and
continuous losses `e*sum_i(q_i^2/d_i)` agree by that graph symmetry; their
phase storage is identical. For each complete ring, and for full support,
the scalar phase-response summaries in section 5 also agree: mean mobility,
cut, mean `q`, covariance, both variances, the squared bound and the summed
current phase rate. Mobility is reflection-even at the lock and `q` is
reflected, so the relevant sums are unchanged. This statement does not apply
to arbitrary subsets or erase the different retained nodal gradients.
In particular, direct evaluation gives
`E_D=5*epsilon^2+2*delta^2` and
`loss=e*((43/3)*epsilon^2+5*delta^2)` for both states.

There is no contradiction with reflection equivariance of the full model:
a simultaneous reflection of form and phase would reverse the prepared
left winding. This pair reverses only the hidden form orientation relative
to that fixed winding. Its different acceleration follows from the already
present phase metric, not from a new primitive variable, force or physical
spin interpretation.

### Scope of the negative result

At equilibrium the derivative of the augmented observation is the vertical
stack `DZ=[C; C*J]` and has rank ten, since `CJ=G*C`. This is an equilibrium rank
statement, not a claim about generic nearby rank. The augmented vector has
twenty entries, already exceeding the eighteen-dimensional full-state
offset quotient; its entry count alone establishes neither compression nor
predictive sufficiency. The equal-fiber counterexample, independently of
those counts, rules out an autonomous law for this `Z` on a full open
neighborhood of the equilibrium.

The proof concerns the ideal smooth law and exact trigonometric lock.
Rounded runtime preparations need not be exact locks, and finite differences
do not convert their near-equalities into these exact fibers. The model-rate
and acceleration conclusions also do not authenticate measured temporal
evidence. Pure odd invariant families and other specially restricted domains
remain separate from the open-domain obstruction.

This closes the selected state-plus-rate test without extending a derivative
ladder. Retain the full engine state for nonlinear prediction unless a
different state reduction or a memory construction is justified separately.
The counterexample does not prove that every reduction is impossible or that
the full quotient is mathematically minimal; diffusion-only memory results
cannot supply the missing joint-law justification.

The subsequent [joint memory derivation](RELATIONAL_PATTERN_MEMORY.md) retains
that initial hidden state and its nonlinear forcing. Its prepared-even
finite-time approximation theorem is separate from an exact autonomous
closure of this observation.

## 7. Shared computation and evidence boundaries

[`derive_linear_observation`](../../src/tnfr/mathematics/linear_observation.py)
now owns exact invariant-row selection for a supplied rational generator
`z'=Jz`. It returns minimal rows, lift, output map and reduced generator with
checked `CT=I`, `CJ=GC`, `O=DC`, complete rank progression and an operational
rank-call budget. The existing affine diffusion realization delegates to it,
retaining its `x'=-Ax+b` sign, original basis, API and resource diagnostics.
No diffusion reference is fabricated for this joint law. Matrix product and
inverse arithmetic share a mathematical owner, with historical import paths
preserved.

The [static instrument](../../benchmarks/relational_local_composition.py)
checks the displayed full matrices, natural coordinates and exact rank on
explicit rational coefficient probes. Its console example declares
`c=inverse_pi=1/3`, `e=w=1/2`, `beta=1`, using Python `Fraction` arithmetic,
fixed node/output order and no randomness or trajectory. Those rational
values are **not** the ideal trigonometric constants. The proof above uses
their actual irrational values through positivity and symmetry; rational
probes do not supply that proof or a certified runtime approximation.

[Controls](../../tests/physics/test_relational_local_composition.py) also
compare the independently assembled ideal Jacobian with static native-field
finite differences and check the nonlinear witness. Floating tolerances in
those implementation checks are distinct from exact rank, nonlinear identities
or certified error bounds. No existing research artifact is rewritten.

The same instrument's `analyze_state_rate_obstruction` reuses the graph and
Jacobian assembly to check section 6 on declared rational `c,s,inverse_pi`.
It retains complete deviation states, rates, accelerations, their projections
and equal storage/loss. At the lock, `F=J*(x,0)` but acceleration also includes
the derived metric-variation term; applying `J` twice alone would miss the
obstruction. Zero even or odd amplitude gives an inconclusive pair, not a
closure theorem. The native controls use `epsilon=1/32`, `delta=1/64`,
`e=w=1/2`, `beta=1`, fixed node order `0,...,9`, and binary64 arithmetic.
Central probes of the field at `z +/- h*F(z)`, `h=2^-10`, check the acceleration
with relative tolerance `2e-7` and absolute tolerance `2e-12`. They do not call
an integrator or claim a derivative of rounded machine arithmetic. These
tolerances check consistency with the ideal derivative, not certified ODE
error. The existing SDK's full centered form vectors distinguish the pair,
so no state was added to the engine to recover the missing information.

The engine's exact `phase_mobility` and `phase_rate_rounding_defect` retain the
arithmetic used by its phase row. The shared detached
[observer](../../src/tnfr/physics/relational_observations.py) consumes them and
the existing cut to expose `region.phase_response`, including its covariance
and squared bound. The SDK delegates to that owner and the exact exporter
retains the new fields; no report changes the law. The
[API contract](../../docs/API_CONTRACTS.md#relational-pattern-observation)
owns field names and availability. Independent
[static controls](../../tests/test_relational_phase_response.py) check the
analytic path case, heterogeneous capacity, exact gradients before floating
materialization, edge accounting, zero-capacity and lossless limits, sign
reversal, partitions and the mixed-state witness. They add no trajectory.

The result supplies a useful reduced local description and identifies its
precise failure beyond first variation. It establishes neither autonomous
fractal nesting, nonlinear regional closure nor physical material constituents.
