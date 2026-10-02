# Predictive local state of two interacting coherent regions

**Status:** exact conditional first-variation closure, nonlinear state and
state-plus-rate counterexamples, a regional response budget, a sufficient
instantaneous attachment interface, conditional support-event budget and
nonselection results, and passive bridge relocation with an explicit shared
recovery domain. The support-law closure audit derives symmetry and clock
restrictions but proves that they do not select occurrence. The complete-action
audit additionally admits nodal resets: actual UM and RA-then-UM examples
can offset positive attachment cost, and the existing continuous law can
produce phase compatibility. These results retain the distinction between
operator events and relational flow; no autonomous activation rule is added.
The [effective-link admission](#effective-link-admission) combines causal
paths, restoring geometry and an explicit local formation family through a
mediator, retaining the supplied fine support and complete nodal law. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next task. The general formation and local-recovery theorems remain
in [relational exchange](RELATIONAL_EXCHANGE_ADMISSION.md).

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
[API contract](../../docs/contracts/RELATIONAL_DYNAMICS.md#relational-pattern-observation)
owns field names and availability. Independent
[static controls](../../tests/test_relational_phase_response.py) check the
analytic path case, heterogeneous capacity, exact gradients before floating
materialization, edge accounting, zero-capacity and lossless limits, sign
reversal, partitions and the mixed-state witness. They add no trajectory.

The result supplies a useful reduced local description and identifies its
precise failure beyond first variation. It establishes neither autonomous
fractal nesting, nonlinear regional closure nor physical material constituents.

<a id="one-bridge-interface-admission"></a>
## 8. One supplied bridge: sufficient interface and endpoint-only obstruction

### Full-state interface card

Take two disjoint, separately admitted connected simple unit graphs, with
held nonnegative capacities and the same acute relational model
\(e\geq0,\ w,\beta>0\). Add the supplied unit edge \((a,b)\), without
changing form, phase or capacity. This is a support intervention, not an
autonomous edge-creation law or a continuous integration step.

At each port \(i\), the following internal message suffices for its new
instantaneous row:

\[
 (x_i,\theta_i,\nu_i,d_i,q_i,z_i),\qquad
 q_i=\sum_{j\sim_{\rm internal}i}(x_i-x_j),\quad
 z_i=\sum_{j\sim_{\rm internal}i}e^{\,{\rm i}(\theta_j-\theta_i)}.
\]

The degree \(d_i\) counts internal neighbors, and the complex resultant
retains both components, not just a mean phase. Keep the internal node state
and the relative regional form/phase frames behind these messages. They
reconstruct the port field at this snapshot, not the future of an autonomous
coarse node. No history is required by this fully retained Markov law;
discarding internal state reintroduces the existing memory obligation.

Let \(r=x_a-x_b\) and \(\delta=\operatorname{wrap}(\theta_b-\theta_a)\).
Directly adding one term to each neighbor sum gives

\[
\begin{array}{lll}
d'_a=d_a+1,&q'_a=q_a+r,&z'_a=z_a+e^{{\rm i}\delta},\\
d'_b=d_b+1,&q'_b=q_b-r,&z'_b=z_b+e^{-{\rm i}\delta}.
\end{array}
\]

Every nonport neighbor set is unchanged. With
\(\alpha_i=\operatorname{Arg}z_i\) and
\(\operatorname{sinc}(0)=1\), the existing law reads

\[
g_i=\alpha_i/\pi,\quad H_i=\pi|z_i|\operatorname{sinc}\alpha_i,\quad
p_i=-e q_i/d_i+w g_i,\quad
\dot x_i=\nu_i p_i,\quad
\dot\theta_i=(w/\beta)\nu_i q_i/H_i.
\]

Substitution of the primed data is the complete attachment identity; no
extra force or pressure term is introduced. Old acute edges together with
\(|\delta|<\pi/2\) suffice for the new acute graph: every real resultant
part is positive and every \(H'_i>0\). The public observer keeps that domain;
a rejected sufficient admission does not exclude all wider regular states.
Zero capacity is allowed and still suppresses both continuous rows.

Only the port rows can change in the ideal local law. In particular,

\[
\dot x'_a-\dot x_a
=\nu_a\left[
\frac{e(q_a-d_a r)}{d_a(d_a+1)}
+\frac{w}{\pi}(\operatorname{Arg}z'_a-\operatorname{Arg}z_a)
\right].
\]

Thus an unchanged cross-edge difference does not preserve the old internal
row: both its degree normalization and phase metric depend on the new
neighbor. Port resultant addition can also rotate the phase source even
when the supplied phase gap is zero, unless the old resultant is real.

The event adds exactly

\[
\Delta S_{\rm event}=\frac{r^2}{2}+\beta(1-\cos\delta)
\]

to \(S=E_D+\beta V\). Its outward form cut from the left component is \(r\).
Neither quantity is the change in a local velocity. A support event need
not inherit fixed-support storage dissipation; admit its budget separately.

### Fixed C5 discriminator, derived before evaluation

Reuse the input owner
[prepared interaction](../../benchmarks/relational_region_interaction.py):
nodes \(0,\ldots,9\), two C5 components, bridge \((0,5)\),
\(x=\epsilon e_1,\ \epsilon=1/256\),
\(\theta_k=\theta_{k+5}=2\pi k/5\), unit capacities and
\(e=w=1/2,\ \beta=1\). Regional phase offset is zero; the left and right
form means remain \(\epsilon/5\) and zero. Set
\(c=\cos(2\pi/5)\), \(h=1+2c\); the exact identity \(2ch=1\) is specific
to this cycle and is not a universal geometric selection principle.

At donor port 0, \(q_0=-\epsilon\) is unchanged, while
\(d_0:2\to3,\ z_0:2c\to h,\ H_0:2\pi c\to\pi h\).
At recipient port 5 the same metric change occurs but \(q_5=0\).
All ideal phase sources initially vanish. Therefore

| Quantity at donor port 0 | Separate component | Joined support | Joined minus separate |
| --- | --- | --- | --- |
| Form rate | \(\epsilon/4\) | \(\epsilon/6\) | \(-\epsilon/12=-1/3072\) |
| Phase rate | \(-\epsilon/(4\pi c)\) | \(-\epsilon/(2\pi h)\) | \(+\epsilon/(2\pi)=1/(512\pi)\) |

Every other ideal form/phase rate is unchanged, including the initially
stationary recipient. The bridge differences, outward cut and event storage
jump are all zero. Nevertheless the total continuous loss changes from
\(3\epsilon^2/2\) to \(17\epsilon^2/12\): it decreases by
\(\epsilon^2/12\), while the storage derivative increases by the same amount.

This is an exact counterexample to keeping the old component rows and adding
only a term that vanishes when the endpoint form and phase agree. It rejects
that explicit control, not every endpoint model with richer retained data.
The sufficient interface above retains precisely the internal quantities
that the control discards; no minimality theorem is claimed.

The existing
[interaction basin](RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
already admits this ideal preparation and eventual local recovery on this
single-bridge support. Its proof is reused, not rederived. The separate
two-adjacent-bridge capture API does not admit this graph.

### Represented admission and shared implementation

The read-only
[attachment observer](../../src/tnfr/physics/relational_observations.py)
evaluates each connected component separately, rebuilds only their captured
state on a detached graph, adds the supplied bridge, and evaluates the joined
field through the same native owner. It never sends the disconnected union
to the relational evaluator. The existing support-transport owner may read
that union for cut and exact form-energy reset accounting.

The field retains its already computed real/imaginary relative resultant.
Port cards, full component/joined fields and rational differences retain
degree, gradient, metric, source and arithmetic evidence without reimplementing
pressure or introducing evolution. The thin SDK delegate and detached report
export share that observation; neither applies the edge to a live network.

Binary64 phase lifts are not exact multiples of ideal pi. Native pressure
uses its own floating trigonometric path; a separately materialized relative
resultant need not match it exactly. The retained pressure-split and rate
rounding defects remain distinct from unbounded transcendental evaluation
error. Static comparisons use explicit absolute tolerances for implementation
agreement, not a certified ideal-input or ODE error bound.
The fixed C5 controls use binary64, no randomness or integration step, and
absolute rate tolerance \(10^{-15}\) with zero relative tolerance. The
current native execution path is retained in the field's pressure-path
metadata; the checked preparation uses the fused canonical pressure owner.

The [static controls](../../tests/test_relational_attachment.py) exercise
the rational zero-phase case, this fixed C5 witness, storage addition and
domain/read-only boundaries. They add no trajectory, parameter scan or
replacement historical response. Exact-real unchanged interior rows and the
actual represented differences remain distinct observations.

**Disposition:** instantaneous interface admission is established in this
scope and the named endpoint-only control is excluded. The existing
transmission study already supplies the temporal response, so this result
does not justify another F4 transmission campaign. It exposes a missing
premise instead: which admissible support intervention occurs, if any.
The [support-event analysis](#support-event-premise-admission) separates
inherited accounting from an additional passivity premise and an occurrence
law. The [single queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent work; this report supplies no selector.

<a id="support-event-premise-admission"></a>
## 9. Support-event budgets and the missing occurrence law

### State card and reusable event owners

Keep a fixed finite node set, signed form \(x\), circular phase \(\theta\),
held capacities \(\nu_i\geq0\), and the same coefficients
\(e\geq0,\ w,\beta>0\). Each support is simple, undirected and unit-weighted.
Every consumed edge is strictly acute. Connected components have at least
two nodes and are admitted separately when the support is disconnected;
this does not broaden the connected-graph engine interface.

Retain relative component form and phase offsets. Common form shifts and
common phase rotations preserve the accounting; independently recentering
the components generally changes a candidate edge's cost.

Between events, use exactly the admitted unforced relational flow. At an
event, only the edge set changes: \(x,\theta,\nu\) have identical left and
right limits. Pressure, degree, resultant and phase metric are recomputed
from the resulting support; they are not independent stored supplies of
work. Both event endpoints must satisfy the stated model domain. This card
does not admit node birth, continuous conductance evolution, a state reset,
forcing or a capacity event.

The relevant owners already separate action from occurrence:

| Owner | Reusable content and boundary |
| --- | --- |
| [Relational flow](../../src/tnfr/dynamics/relational.py) | Joint storage and fixed-support loss, with represented arithmetic defects kept separate from the exact-real identity |
| [Attachment observation](../../src/tnfr/physics/relational_observations.py) | Fresh component/joined fields for a supplied bridge; no event is executed or selected |
| [Support transport](../../src/tnfr/physics/support_transport.py) | Exact same-node, same-form added/removed-edge Dirichlet reset; this accounts for form storage, not the full phase contribution |
| [Pattern contact](../COHERENT_PATTERN_CONTACT.md#model-and-prospective-control) | A supplied attachment/removal schedule under a different sine phase law; neither that schedule nor that law follows from relational exchange |
| [THOL birth and transport](../THOL_BIRTH_AND_TRANSPORT.md) | Configured birth, coupling and dispatch contracts; a child construction or weighted coupling action is not this state-preserving unit-edge event |

The existing [selector symmetry owner](../../src/tnfr/physics/selector_symmetry.py)
can test an independently declared finite candidate action. Symmetry may
exclude a unique deterministic choice; it does not create an event clock.

### Exact jump and finite hybrid balance

For an unordered pair \(a,b\), define its nonnegative edge storage

\[
c_{ab}(x,\theta)=\frac{(x_a-x_b)^2}{2}
                 +\beta[1-\cos(\theta_b-\theta_a)].
\]

The declared storage on support \(E\) is \(S_E=\sum_{\{a,b\}\in E}c_{ab}\).
For a state-preserving event \(E^-\to E^+\), let
\(A=E^+\setminus E^-\) and \(R=E^-\setminus E^+\). Unchanged edges cancel,
so the exact ideal jump is

\[
\boxed{\Delta S=\sum_{\{a,b\}\in A}c_{ab}
                  -\sum_{\{a,b\}\in R}c_{ab}.}
\]

No occurrence assumption is needed for this identity. In particular, pure
addition has \(\Delta S\geq0\), even though the continuous flow satisfies

\[
\dot S_E=-D_E,\qquad
D_E=e\sum_i\frac{\nu_i q_i^2}{d_i}\geq0,\qquad q=B_E x.
\]

Suppose a declared trajectory is admitted on every continuous segment and
has finitely many such events in \([0,T]\). Integrating each fixed-support
identity and telescoping its endpoints gives

\[
S_{E(T^+)}(X(T^+))-S_{E(0^-)}(X(0^-))
=-\int_0^T D_{E(t)}(X(t))\,dt+\sum_k\Delta S_k,
\qquad X=(x,\theta,\nu).
\]

The endpoint values are before any included event at zero and after any
included event at \(T\); the sum uses those same endpoint conventions.
This is conditional accounting for an admitted hybrid trajectory, not a
construction of one. It supplies no event schedule, non-Zeno theorem,
global continuation, capture certificate on changed support or binary64
Euler-error bound. Zero capacities remain allowed; they suppress their
continuous rows without removing an event's edge-storage cost.

An event budget such as \(\Delta S_k\leq W_k\) requires independently
declared available work \(W_k\). Neither the fixed-support loss identity
nor pressure refresh supplies that work. Past dissipation cannot be spent
retrospectively as an undeclared reservoir: retaining a budget/history and
its replenishment or imposing a different cumulative criterion would add
premises to this state card.

### Passive pure addition requires exact coincidence

Impose the **additional event premise** that no work is supplied and storage
cannot increase at each event: \(\Delta S\leq0\). For pure addition, every
summand is nonnegative. Consequently

\[
\Delta S\leq0
\quad\Longleftrightarrow\quad
x_a=x_b\ \text{and}\ \theta_a=\theta_b\pmod{2\pi}
\quad\text{for every added edge }\{a,b\}.
\]

The right side gives zero jump, not strict decrease. It is sufficient for
the new edge's acute admission; the existing support/state checks remain
required. Capacities at its two endpoints need not coincide. This restriction
is derived from the selected storage **plus event passivity**, not from the
nodal equation alone. It does not preclude
work-funded attachment, simultaneous removal or an admitted nodal reset.

A zero-cost bridge can nevertheless change both port rates, as the exact
[C5 witness](#fixed-c5-discriminator-derived-before-evaluation) shows. More
generally, at a zero-cost bridge the two port gradients \(q_i\) are unchanged
and their degrees grow by one. Thus

\[
D_{\rm joined}-D_{\rm separate}
=-e\sum_{i\in\{a,b\}}\frac{\nu_i q_i^2}{d_i(d_i+1)}\leq0.
\]

Equality holds when each affected \(e\nu_i q_i^2\) vanishes. Otherwise the
new connection decreases instantaneous dissipation despite having no storage
jump. This is not a prediction that its entire future dissipates less.

### Continuous-intensity obstruction on an open full-state domain

Fix one candidate missing edge \(a,b\) and an open admitted full-state
domain \(U\) for the pre-event components. Forms and phases are independent
coordinates there; in particular, a sufficiently small change of \(x_a\)
is allowed. Capacities may be fixed, including zero. Consider a continuous
nonnegative occurrence intensity \(\lambda_{ab}:U\to[0,\infty)\) with
the requirement that whenever \(\lambda_{ab}(X)>0\), its state-preserving
pure addition satisfies the passive-event premise above.
Equivalently, the event sector requires
\(\lambda_{ab}(X)c_{ab}(X)=0\) at each state.

**Then \(\lambda_{ab}\equiv0\) on \(U\).** Indeed, the admissible
coincidence set has empty interior: perturbing \(x_a\) at a coincident
state makes \(c_{ab}>0\). If the intensity were positive at that state,
continuity would keep it positive in a neighborhood containing such an
inadmissible perturbation, a contradiction. Away from coincidence passivity
already forces the intensity to vanish. The same argument applies to each
member of a finite family of candidate additions.

This statement concerns a continuous intensity for **discrete unit-edge
jumps**. It is not a theorem about continuously changing conductance, a
state-reset process, stochastic expected-budget cancellation, a discontinuous
guard, or a domain restricted in advance to coincident endpoints. It does
not forbid an event on a discrete coincidence guard; it shows that a
nontrivial such guard is additional structure, not a continuous extension
selected by the existing flow.
In particular, the weaker expected-generator condition
\(-D+\sum_{ab}\lambda_{ab}c_{ab}\leq0\) could offset positive-cost jumps
with continuous loss. It is a different stochastic premise, not the
pathwise event passivity assumed here.

### Exact nonselection at an admitted zero-cost event

Keep the supplied ordered candidate \((0,5)\), clock origin and the entire
C5 preparation of section 8, with \(\epsilon=1/256\) and unit capacities.
Two declarations satisfy the same nodal laws and passive-event requirement:

1. Execute no attachment and continue the separately admitted components.
2. Execute the supplied bridge once at \(t=0\), without a nodal reset, and
   continue the admitted joined model.

Both have event jump zero. Both admit a local continuous continuation by
smoothness in their respective strictly acute domains. Yet their donor
form and phase rates immediately differ by \(-\epsilon/12\) and
\(+\epsilon/(2\pi)\), respectively. This is the existing static witness,
not a repeated transmission experiment. Specifying the same candidate in
both cases prevents a port-selection ambiguity from hiding the distinct
**occurrence** decision.

The two declarations are compatible conditional interventions, not two
derived autonomous laws. They prove that the flow, storage and event
passivity do not require the admissible attachment. Neither the event time
nor an occurrence rate is selected by that accounting.

### Deletion and atomic exchange are different accounting cases

Pure deletion has \(\Delta S=-\sum_R c_{ab}\leq0\), provided the resulting
support is still admitted. Energy decrease alone does not ensure connected
support, retain a named cycle or preserve a pattern's identity. A deleted
cycle edge can remove the very cycle on which winding was defined without
any continuous phase crossing.

A simultaneous addition/removal event is passive exactly when
\(\sum_A c_{ab}\leq\sum_R c_{ab}\). For an exact scalar example, take
the path \(0-1-2-3\), phase zero, unit capacities and
\(x=(0,2,1,1)\). Remove \((0,1)\) and add \((0,2)\). Both supports
are connected simple unit graphs with acute phases. The removed edge costs
\(2\), the added edge costs \(1/2\), and hence
\(\Delta S=-3/2\), despite unequal form at the new endpoints. The strict
inequality persists on a sufficiently small open neighborhood of this
state; the coincidence obstruction for pure addition does not apply.

This is a single **atomic exchange** budget. If deletion and addition were
separate events, each required to be passive, the positive-cost addition
would still fail. Edge-count preservation, candidate selection, identity
preservation and event timing are independent premises. No exchange law
or active campaign follows from this accounting example.

### Shared represented accounting and static controls

The existing attachment report now exposes `continuous_loss_change` and
`represented_zero_supply_passive` as derived properties, without storing a
second energy formula. `assess_supply(supplied_work)` compares signed
caller-declared work with its captured `storage_change`. The returned exact
rational margin and balance flag concern the represented snapshot, not an
ideal trigonometric certificate or authentication of an available resource.
Continuous loss is a rate and is never credited as event work. The common
SDK exporter retains this assessment; no operation modifies a live graph.

The [static controls](../../tests/test_relational_attachment.py) check an
independently known equal-phase bridge cost, exact budget boundaries including
sub-binary64 rational margins, the retained zero-cost C5 discriminator, and
the P4 atomic-exchange witness through the native field and shared transport
reset. They do not implement an exchange selector or rerun a trajectory.
The [API contract](../../docs/contracts/RELATIONAL_DYNAMICS.md#relational-attachment-observation)
owns represented-input and export details. The exact-real jump, passivity and
continuous-intensity conclusions above are analytic results under their stated
premises; a test of finite arithmetic is not their proof.

**Disposition:** the inherited storage identity, the additional passive-event
restriction and the unselected occurrence decision are now distinct.
Nontrivial support dynamics requires an explicitly admitted continuation
of this state/event card; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns that choice. These results introduce no selector, reservoir or runtime
support mutation.

<a id="identity-preserving-bridge-relocation"></a>
## 10. Passive bridge relocation with a shared recovery domain

### Atomic event and signed local interface

Keep the two supplied ordered cycles \((0,1,2,3,4)\) and
\((5,6,7,8,9)\), their internal unit edges, full nodal state and relative
frames. Replace the single bridge \((0,5)\) by \((1,6)\) in one event.
The pre-event and post-event graphs are connected simple unit graphs; no
intermediate disconnected flow is part of this event. Holding edge count,
preserving these cycles and supplying this candidate are additional event
premises, not consequences of the nodal identity.

The interface extends section 8 without a second pressure law. For any
supplied atomic replacement, let \(\sigma_{ij}=1\) on added edges,
\(-1\) on removed edges, and zero elsewhere. From the pre-event full
graph's port quantities, the exact ideal updates are

\[
\begin{aligned}
d_i^+&=d_i^-+\sum_j\sigma_{ij},\\
q_i^+&=q_i^-+\sum_j\sigma_{ij}(x_i-x_j),\\
z_i^+&=z_i^-+\sum_j\sigma_{ij}
                          \exp[\mathrm{i}(\theta_j-\theta_i)].
\end{aligned}
\]

They also cover shared endpoints, where degree changes can cancel while
gradient and resultant changes do not. Substituting these quantities into
the same \(g,H,p,\dot x,\dot\theta\) rows gives the post-event field.
Only nodes incident to a changed edge can change their ideal local rows.
Both graph/state domains still require independent admission. In particular,
an edge-cost inequality alone cannot admit a nonacute proposed edge.

The exact storage jump is

\[
\Delta S=c_{16}(X)-c_{05}(X).
\]

All internal cycle edges and nodal phases are retained, so both snapshot
windings are unchanged. Future identity requires the continuous recovery
argument below; it does not follow from that snapshot observation alone.

### One equilibrium and one explicit bound for both supports

Set \(\kappa=2\pi/5\), \(c=\cos\kappa\), and
\(\theta_{*,k}=\theta_{*,k+5}=k\kappa\) for \(k=0,\ldots,4\).
Both possible bridge gaps vanish at this phase state. The cycle sine sums
cancel, so uniform form and \(\theta_*\) are an equilibrium for either
complete relational law. Their common equilibrium storage is

\[
S_*=10\beta(1-c).
\]

Assume \(e,w,\beta>0\) and **strictly positive held capacities** at every
node. Equal capacities are unnecessary. Apply the existing
[local recovery theorem and explicit barrier](RELATIONAL_EXCHANGE_ADMISSION.md#an-explicit-local-domain-and-the-offset-limits)
separately to the two graphs, with no change to its flow or storage premises.
Choose consistent local phase lifts and put

\[
\Pi=I-\mathbf1\mathbf1^T/10,\qquad
\|z\|^2=\|\Pi x\|^2+\|\Pi(\theta-\theta_*)\|^2,\qquad
r=\frac{\pi}{20\sqrt2},\qquad
\mu=\min(1,\beta/10).
\]

Both graphs have ten nodes and diameter five. The same pathwise Cauchy
bound used in the [interaction proof](RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
gives \(\lambda_2(B)\geq2/45\). In the radius-\(r\) ball, every
reference edge gap changes by less than \(\sqrt2r=\pi/20\); hence all
edges in either graph remain acute and the common Hessian lower bound is
\(c_r=\sin(\pi/20)>1/10\). The strict sine inequality follows from
concavity above the chord on \([0,\pi/2]\).

For either graph the theorem's barrier consequently satisfies

\[
k_r r^2
=\frac{\lambda_2(B)}2\min(1,\beta c_r)\frac{\pi^2}{800}
\ \geq\frac{\pi^2}{36000}\mu
\ >\frac{\mu}{4000}.
\]

Write \(E_{\rm rel}^\pm=S_{E^\pm}(X)-S_*\). The following strict
conditions therefore define a sufficient open domain of full nodal states:

\[
\boxed{\quad \|z\|<r,\qquad
E_{\rm rel}^-<\mu/4000,\qquad \Delta S<0.\quad}
\]

Because the equilibrium storage is the same on both supports,
\(E_{\rm rel}^+=E_{\rm rel}^-+\Delta S<E_{\rm rel}^-\).
The event preserves \(z\), so both pre-event and post-event preparations
meet the existing basin conditions. Holding the respective support fixed,
each continuous solution remains acute and converges to the same phase
shape and uniform form modulo its limiting common offsets. Those offsets
need not agree between the two evolutions. Internal winding one persists.
No repeated-event stability or universal decay time is established.

### Exact strict preparation and independent controls

Take \(x=\epsilon e_0\) with the phase state \(\theta_*\) above and
\(\epsilon\ne0\). The two original internal edges incident to node 0
contribute \(\epsilon^2\) in total. The old bridge contributes
\(\epsilon^2/2\); the new bridge contributes zero. Therefore

\[
E_{\rm rel}^-=\frac32\epsilon^2,\qquad
E_{\rm rel}^+=\epsilon^2,\qquad
\Delta S=-\frac12\epsilon^2,\qquad
\|z\|^2=\frac9{10}\epsilon^2.
\]

All ideal phase sources vanish at this preparation. Put \(h=1+2c\).
The changed port data, inserted into the unchanged rate formulas of
section 8, are

| Node | Before \((d,q,H)\) | After \((d,q,H)\) |
| --- | --- | --- |
| 0 | \((3,3\epsilon,\pi h)\) | \((2,2\epsilon,2\pi c)\) |
| 5 | \((3,-\epsilon,\pi h)\) | \((2,0,2\pi c)\) |
| 1 | \((2,-\epsilon,2\pi c)\) | \((3,-\epsilon,\pi h)\) |
| 6 | \((2,0,2\pi c)\) | \((3,0,\pi h)\) |

Every other ideal row is unchanged. In particular, node 0's form rate is
unchanged even though its gradient and phase rate change; event cost alone
does not describe the local dynamical response.

The explicit family
\(0<\epsilon^2<\mu/6000\) satisfies the common basin conditions:
the energy bound is immediate and
\(\|z\|^2<3/20000<9/800<r^2\). This proves nonemptiness of the
strict open domain without executing a trajectory. It permits arbitrary
positive held capacities and positive \(e,w,\beta\), without a uniform
convergence-rate claim as those parameters approach excluded boundaries.

For \(\beta=1\), the fixed choice \(\epsilon=1/256\) gives

\[
E_{\rm rel}^-=\frac3{131072}<\frac1{40000},\qquad
E_{\rm rel}^+=\frac1{65536},\qquad
\Delta S=-\frac1{131072}.
\]

Two controls separate the obligations:

- **Reverse exchange at the same state:** replacing \((1,6)\) by
  \((0,5)\) costs \(+\epsilon^2/2\). Both endpoints still satisfy the
  same recovery bounds, but this reverse event violates the no-work passive
  premise. Recovery is not a substitute for an event budget.
- **Zero contrast:** at \(\epsilon=0\), both graphs are equilibria and
  either exchange has zero cost. No field response or energy preference
  selects an occurrence. This is the boundary of the strict example, not
  evidence of a strictly dissipative event.

Zero capacities remain admissible to the storage accounting but are outside
this recovery theorem. At zero form dissipation the existing local theorem
also excludes generic recovery; the event itself does not repair that limit.

### Static implementation protocol and claim boundary

The static comparison is fixed before native-field evaluation: node order
\(0,\ldots,9\), the two ordered C5 cycles, old bridge \((0,5)\), new
bridge \((1,6)\), \(\epsilon=1/256\), \(e=w=1/2\), \(\beta=1\)
and unit capacities. Form is zero except at node 0. Ideal phases are
\(2\pi k/5\); the implementation preparation uses Python
`2 * math.pi * k / 5` for both nodes \(k\) and \(k+5\).

Reuse fresh native fields and the shared support-reset owner on detached
states. Compare actual storage changes using exact rational arithmetic on
represented values. Use absolute tolerance \(10^{-15}\), with zero
relative tolerance, for field agreement with the stated ideal formulas.
No time integration, random input, fit or support event on a live graph
belongs to this comparison.

The shared `observe_relational_relocation` and SDK `relational_relocation`
compare the old and new connected fields on detached state. The removed edge
must be a bridge between two nontrivial components, and the replacement joins
those same components while preserving every internal edge. Port updates,
field differences, support-reset accounting and signed supply assessment reuse
the attachment owners. The observer admits either sign of event cost; it does
not turn passivity into a graph mutation or a theorem verdict. See the
[execution contract](../../docs/contracts/RELATIONAL_DYNAMICS.md#relational-relocation-observation)
and [static controls](../../tests/test_relational_attachment.py).

The ideal phase preparation is not its binary64 materialization. Exact
rational accounting of represented storage does not enclose trigonometric
error or certify the ideal recovery inequalities for arbitrary floating
inputs. Preserve rate/pressure defects and report this static comparison
separately from the exact-real theorem. The two-bridge capture API does not
certify either single-bridge graph.

**Disposition:** a supplied passive relocation can retain both prepared
identities and enter an explicit continuous recovery domain on an open set.
It reorganizes existing support; it does not create the substrate or select
whether, when or which relocation occurs. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the constitutive decision that remains; no autonomous event law is
introduced here.

<a id="support-law-choice-and-clock"></a>
## 11. Support-law closure: restrictions do not select occurrence

### State, candidates and the missing row

Retain `X=(x,theta,nu)`, the actual unit support `E`, held positive capacities,
and the complete relational field `F_E`. Fix a finite supplied family of
atomic bridge exchanges `a=(removed edge, added edge)`. Each reset is
`R_a(E,X)=(E_a,X)`: it changes support and retains every nodal coordinate.
Admit the old and new fields separately. If recovery is claimed, require a
valid common-basin argument such as section 10, not merely unchanged winding
or a favorable represented storage jump.

The continuous law and the reset map leave distinct data unspecified:

| Item | Restriction inherited from the present premises | Remaining choice |
| --- | --- | --- |
| Candidate family | Must transform with the full state and preserve the declared internal support | Which exchanges are eligible is itself a premise |
| Event budget | Additional zero-supply passivity requires `Delta S_a<=0` | A nonpositive cost does not require execution or rank two eligible events |
| Identity | The stated pre/post recovery conditions must both hold | Recovery does not choose an event or prove repeated-switching stability |
| Selection | An equivariant deterministic choice must be fixed by the state stabilizer | Several fixed choices can remain; ties need not be symmetries |
| Timing | Event rates have inverse-clock units and transform with the complete flow | Waiting law, guard, threshold, randomness or memory remains unspecified |
| History | Include every consumed mark, age, budget or previous event | Such memory is not present merely because a diagnostic report is retained |

Abstention, denoted `bottom`, is a valid outcome. A candidate set, a selected
candidate, an occurrence and a waiting time are not interchangeable. The
current read-only observers provide the candidate's field/budget comparison;
they implement none of these missing causal choices.

### Exact symmetry acts on events, not their list positions

Let a supplied finite group act on nodes, the support and every consumed
state/history coordinate. It acts on each event by sending **both** its
removed and added edges to their images. Candidate eligibility must be
invariant under the stabilizer `H` of the complete declared state. For a
deterministic equivariant selector `s`,

\[
s(E,X)=s(h(E,X))=h\,s(E,X)\qquad(h\in H).
\]

Thus only stabilizer-fixed events or abstention are possible. If no eligible
event is fixed, a unique deterministic non-abstaining choice is obstructed.
A singleton candidate orbit removes this particular obstruction; it does
not derive a selector, its continuity, or an event time. Sorting node names,
taking the first candidate or hiding a mark in insertion order does not
respect this premise.

The existing [finite-action owner](../../src/tnfr/physics/selector_symmetry.py)
already proves this restriction. To apply it to edges without a parallel
selector, lift the action to typed node and event slots. Retain the complete
exact nodal labels, actual support relations and each event's added/removed
endpoint incidences. Permute the event slots by the induced edge action, then
pass those slots as candidates. This is an exact declaration, not a live
graph certificate or an assertion that rounded phases have exact symmetry.

A strict passive witness exists without breaking node-label symmetry. Use
two C5 rings and bridge `(0,5)`, synchronized phase zero, common positive
capacity, and `x=epsilon*e_0`, `epsilon!=0`. Restrict candidates structurally
to replacement bridges between neighbors of the two old ports:
`{1,4} x {6,9}`. Every candidate has

\[
\Delta S_a=-\epsilon^2/2<0.
\]

The independent reflections of the two rings fix the complete old state
and act transitively on these four events. None is fixed. Hence strict
storage preference for relocation does not supply a unique equivariant
choice. This synchronized witness is distinct from the winding-one recovery
preparation in section 10. With that latter primitive phase held fixed,
equal costs need not be generated by any full-state symmetry.

If randomness is independently postulated, invariance forces equal
probabilities or intensities **within** each stabilizer orbit. It does not
fix the total event rate, the mass assigned to different orbits, or the
probability of abstention. Symmetry therefore does not derive stochastic
dynamics either.

### The event clock must transform with both continuous rows

For a constant clock conversion `tau=b*t`, `b>0`, with form units held fixed,
the same mathematical paths are represented by

\[
\nu_i'=\nu_i/b,\qquad F'_E=F_E/b,\qquad
\lambda'_a=\lambda_a/b.
\]

Here `lambda_a` is an independently supplied occurrence intensity, when such
a law is chosen. The integrated hazard `integral(lambda_a dt)` and event
choice probabilities are invariant. Storage and its event jump are unchanged;
continuous storage work and loss are divided by `b`. Increasing solver `dt`
without transforming capacity changes the evaluated evolution, not its units.
The engine normalizes its two pressure weights: dividing both by `b` leaves
the effective weights unchanged and cannot substitute for this conversion.

For `d tau/dt=alpha(t)>0`, absorbing the clock into capacity instead gives

\[
\nu'_i=\nu_i/\alpha,\qquad
\frac{d\nu'_i}{d\tau}=-\frac{\nu_i\dot\alpha}{\alpha^3}.
\]

A nonconstant `alpha` generally leaves the held-capacity family. A
state-dependent clock has `dot(alpha)=L_F alpha`; if it depends on support,
an event can also reset the transformed capacity. One cannot suppress those
terms to manufacture a native event clock. The
[full-state clock owner](../NODAL_PARAMETER_FOUNDATIONS.md#pressure-clock-full-state-closure)
retains the regularity requirements for treating this as a state chart.
For example, dividing all capacities by their mean discards their common
scale unless it is retained separately; it is not an invertible full-state
chart. Held transformed capacity requires `alpha` to be constant along each
flow segment and preserved at its events, not merely positive.

At a complete equilibrium a time-independent deterministic guard sees the
same state forever. It cannot both remain inactive initially and first become
active after a positive finite delay without additional time/history input.
This excludes that particular state-only waiting mechanism, not every event
law. An event-rate postulate can introduce randomness, but the equilibrium
does not derive it.

### Explicit compatible rival laws

Nonuniqueness remains even when a supplied candidate has strict passive slack
and both supports have proved recovery. Work on a common open domain `U`
where the finitely many supplied candidates satisfy these hypotheses. Follow
the admitted deterministic old-support flow until its first event or first
exit from `U`. Stop this comparison at exit; after an event retain its new
support and the existing recovery law. This construction needs at most one
event and asserts no arbitrary repeated-switching result.

Define the dimensionless slack and a rate from quantities already present:

\[
s_a=-\Delta S_a/\beta>0,\qquad
r=e\,\overline\nu>0.
\]

These do not select a law. For example, the following are **independent
logical countermodels**, not inferred TNFR rules or proposed engine defaults:

\[
\lambda_a^{(0)}=0,\qquad
\lambda_a^{(1)}=r s_a,\qquad
\lambda_a^{(2)}=r s_a^2.
\]

The last two explicitly postulate competing stochastic first-event clocks.
On every compact subdomain they have finite continuous rates and positive
total rate. Their first-event survival probability along the unchanged
pre-event path is

\[
\Pr(T>t)=\exp\!\left[-\int_0^t\sum_a\lambda_a(E,X(u))\,du\right],
\]

up to the stopping time. The conditional instantaneous event-type weights
are `lambda_a/sum_b lambda_b`; they are not generally the probabilities
integrated over the entire future path. All three models retain the same
continuous field, passive budget, recovery premise, node relabeling and
common form/phase origins. Two strictly positive models remain different
even if abstention is excluded. Multiplying every rate by a positive
dimensionless constant changes waiting while preserving instantaneous type
weights; holding `F_E` fixed makes this a different event law, not a common
clock conversion.

The rates also satisfy the selected model's
[form-unit covariance](RELATIONAL_EXCHANGE_ADMISSION.md#4-origin-units-and-exact-replication).
Under `x'=a*x`, `beta'=a^2*beta`, `tau=b*t`, `a,b>0`, and normalized coefficients, put
`k=e+a*w`, `e'=e/k`, `w'=a*w/k`, `nu'=k*nu/b`. Then `s'_a=s_a` and
`r'=r/b`. Transform `U` and its recovery inequalities as well; keeping a
numerical radius in mixed form/phase coordinates unchanged is not a unit
conversion. This covariance restricts admissible formulas without choosing
the exponent or probability law.

A concrete discriminator uses the aligned winding-one phases and
`x=epsilon*e_0+(epsilon/2)*e_1`, with supplied candidates `(1,6)` and `(2,7)`
replacing `(0,5)`. Their exact jumps are `-3*epsilon^2/8` and
`-epsilon^2/2`. Their ideal common equilibrium has zero bridge gaps; the
section 10 basin proof applies to each support for sufficiently small
nonzero `epsilon`. The two positive rival models predict instantaneous
type ratios `3/4` and `9/16`, hence first-type weights `3/7` and `9/25`.
These are analytic conditional predictions of different **assumed** laws,
not measured event frequencies or a reason to install either law.

### Integration and disposition

The [selection controls](../../tests/physics/test_relational_event_selection.py)
exercise the existing exact-action owner and real relocation observer; the
[clock controls](../../tests/physics/test_relational_event_clock.py) compare
the complete shared fields and a bounded Euler unit-conversion control.
Exact arithmetic on represented storage is kept separate from the ideal
phase preparation and its recovery proof. No event selector, random generator,
timer, automatic graph mutation or duplicate SDK report is introduced.

**Result:** full-state availability, event passivity, recovery, symmetry and
clock covariance do not uniquely close support evolution. No independently
justified additional occurrence premise was obtained in this audit. Retain
support changes as supplied interventions. This closes the bounded
nonselection question; it does not prove that a future independently justified
principle could never determine an event law. Fixed-support pattern formation
and interaction remain valid and need no primitive rewiring to exist.

<a id="nodal-reorganization-and-contact"></a>
## 12. Nodal reorganization and connection in one action

### Revising the event premise, not discarding the nonselection result

An NFR's proposed ability to act through operators belongs to the same
research question as connection formation. An operator specifies an action
on state. Deriving its internal activation additionally requires a map from
the acting pattern's state to the target, action and time. Neither an external
controller nor a conscious agent is required by that question; the current
implementation's invocation rules do not already supply its physical answer.

The earlier pure-addition restriction held every nodal coordinate fixed.
Actual UM changes phase and can change capacity while adding edges. RA, EN
and AL can change form. It is therefore necessary to examine a complete reset
`(E,X)->(E_plus,R X)`, rather than transfer the frozen-triad restriction to
every operator-mediated contact.

The implementation audit reuses existing owners:

| Action | Existing mechanism | Premises still supplied |
| --- | --- | --- |
| UM, Coupling | Snapshot phase proposals and optional capacity alignment; functional links to phase-compatible nonneighbors | Invocation, candidate inventory, phase limit, affinity mixture/threshold, sampling and merge policy |
| RA, Resonance | Form mixing and configured phase/capacity changes on existing compatible neighbors | Invocation, factors, target set and ordering; RA itself creates no edges |
| EN, Reception | Incoming-form blending on its declared execution path | Which incoming data are available and when the operator runs |
| AL, Emission | Form change on an existing node | Source/amplitude and invocation; it does not create the substrate |

EN's actual form input is the unweighted mean of existing incoming neighbors
(predecessors on a directed graph), not a U3-filtered or phase-ranked mix.
Its source-ranking telemetry does not select or weight that mean. AL and EN
write no phase or support on these basic paths. Operator-class callbacks and
complete words retain their separate execution contracts.

The [UM kernel](../../src/tnfr/operators/_coupling_stage_kernel.py) searches
all graph nodes or the supplied `_node_sample` for nonneighbors. That inventory
is a potential-contact relation, not a relation created by the search itself.
UM first needs an existing compatible neighbor at its target; it can join
nontrivial components or attach an isolate, but does not bootstrap two
isolates. RA uses the [shared neighbor stage](../../src/tnfr/operators/network_stage.py).
The existing [child/coupling feedback](../CHILD_COUPLING_FEEDBACK.md) and
[THOL transport](../THOL_BIRTH_AND_TRANSPORT.md) already retain supplied
targets and schedules; they are useful mechanisms, not forgotten autonomous
selection theorems.

In particular, UM's functional-link score mixes phase affinity, normalized
absolute-form similarity and Si similarity. Its form term changes under a
common EPI offset, whereas the selected continuous relational law is offset
invariant. Limited proximity sampling also uses snapshot rank for ties.
These configured policies cannot be imported as a derived relational law
without revising and testing their premises. They remain separate from the
read-only storage accounting below.

### Full reset accounting

For symmetric nonnegative conductance `W` and simple undirected support `U`,
define the same storage functional, with their distinct roles retained:

\[
S(W,U,x,\theta)=\frac12\sum_{\{i,j\}\in U}W_{ij}(x_i-x_j)^2
 +\beta\sum_{\{i,j\}\in U}[1-\cos(\theta_j-\theta_i)].
\]

At unit weights this is the selected relational storage. Reading it on other
weights is valid endpoint accounting, not admission of a weighted relational
evolution law. In particular a zero-weight support edge still contributes
phase storage, consistently with the native unweighted phase neighborhood.

Writing `X_minus` and `X_plus` for the actual endpoint states gives the exact
decomposition

\[
\begin{aligned}
\Delta S={}&S(W_-,U_-,X_+)-S(W_-,U_-,X_-)\\
 &+S(W_+,U_+,X_+)-S(W_-,U_-,X_+).
\end{aligned}
\]

The first term is nodal reorganization on the old support; the second is
support work at the new nodal state. This intermediate evaluation is an
algebraic counterfactual, not an asserted execution order or an extra event.
Capacity does not enter this storage explicitly, but any capacity change
must be retained because it changes subsequent rates. A negative first term
can offset a positive second term in **this same action**. No reservoir of
past dissipation is inferred. Event passivity remains an additional premise,
tested against the full `Delta S`, and no timing law follows.

### A strict UM attachment funded by its own phase reset

Take two existing unit edges `(0,1)` and `(2,3)`, constant form `x_i=m>0`,
equal positive capacities and equal Si. Set

\[
\theta=(0,2h,2h,4h),\qquad 0<h<\pi/4.
\]

Invoke one bidirectional UM stage at node 1 with phase factor
`0<eta<=1`, and explicitly supply node 2 as its only candidate nonneighbor.
The compatible old pair has circular mean `h`, so the ideal phase proposal is

\[
\theta^+=(\eta h,(2-\eta)h,2h,4h).
\]

The proposed bridge `(1,2)` has positive phase cost
`beta*(1-cos(eta*h))`. Uniform form makes its transport cost zero for any
nonnegative functional-link weight. The full change is nevertheless

\[
\Delta S=\beta f(\eta),\qquad
f(\eta)=\cos(2h)-\cos(2(1-\eta)h)+1-\cos(\eta h)<0.
\]

Indeed `f(0)=0`, `f(1)=cos(2h)-cos(h)<0`, and

\[
f''(\eta)=4h^2\cos(2(1-\eta)h)+h^2\cos(\eta h)>0.
\]

Convexity gives `f(eta)<=eta*f(1)<0`. This is a conditional finite-action
theorem for the declared one-candidate UM reset. It proves neither universal
UM passivity nor its autonomous invocation, and these two-node components
are a minimal mechanism witness, not a demonstrated maintained NFR identity.

The concrete production control uses `h=pi/8`, `eta=1/4`, `m=1/2`, capacities
one and Si `0.8`. All phase gates and the compatibility threshold have strict
slack. The ideal new weight is `63/64`, and the new gap is `pi/32`.
The strictly negative budget therefore also persists under sufficiently small
admitted perturbations of the inputs with this candidate policy fixed.
The theorem concerns exact-real phases; the test separately reads the actual
binary64 stage endpoints and their represented storage.

The generated weight is **not one**. The current unit-support relational
executor consequently rejects that output. Do not silently replace its weight
or reuse the unit-bridge recovery theorem as if this operator event were the
same model. The result establishes feasible contemporaneous reorganization
and attachment, not subsequent pattern maintenance.

### RA form redistribution can also offset an attachment cost

On the same two edges, take forms `(m+d,m-d,m-d,m+d)`, `m>d>0`, uniform
phase and equal positive capacities. One all-target RA stage with unclipped
form-mix factor `rho` changes each pair's contrast from `d` to
`c*d`, where `c=1-2*rho`. A subsequent admitted UM stage at node 0, with only
candidate 2, can add their positive-conductance bridge. Calling its actual
conductance `omega`, uniform phase gives

\[
\Delta S=\big[(4+2\omega)(1-2\rho)^2-4\big]d^2.
\]

For `rho=1/4` and `0<omega<=1`, this is strictly negative, while the new
edge cost `2*omega*c^2*d^2` is positive. The production control uses `m=1/2`,
`d=1/8`; RA produces forms `(9/16,7/16,7/16,9/16)`. It retains the actual
capacity amplification and UM conductance. The combined two-event budget is
telescoping endpoint accounting: RA's reduction and UM's positive increment
must also be reported separately. This is not zero-supply passivity of each
individual event or a claim that prior dissipation is a stored work reserve.
An internally justified composite action would still need its own definition.

### The continuous law can create phase admission

At held unit support, phase and capacity, an AL/EN form jump `d` changes the
next freshly evaluated relational phase row by an exact linear identity:

\[
q^+=q+B d,\qquad
\dot\theta^+-\dot\theta^-=\frac w\beta
\operatorname{diag}(\nu_i/H_i)B d,\qquad
\Delta S=q^T d+\tfrac12 d^T B d.
\]

For one unclipped EN target `i` with mix `0<=rho<=1`, its existing neighbor
mean gives `d_i=-rho*q_i/degree_i` and all other entries zero. Consequently

\[
\Delta S=-\rho(1-\rho/2)q_i^2/\operatorname{degree}_i\le0.
\]

This redistributes existing contrast; uniform form remains uniform. A
targeted AL increment `a` from uniform form instead costs
`degree_i*a^2/2` on nonisolated support. Such an increment can provide a phase
response, but its supplied action and budget must remain explicit. The
identities do not transfer unchanged to simultaneous multi-target EN,
clipping, a capacity/phase reset or arbitrary operator words.

For a supplied candidate pair `i,j` in independently admitted components,
retain a common phase reference and a declared native U3 limit
`0<gamma<=pi/2`. On a
regular lift, set `delta=theta_j-theta_i` and
`M=cos(delta)-cos(gamma)`. The existing phase row gives

\[
\dot M=-\frac w\beta\sin\delta
\left(\frac{\nu_jq_j}{H_j}-\frac{\nu_iq_i}{H_i}\right).
\]

Positive `M` is phase compatibility for this limit. It is not an edge or an
instruction to invoke UM. Consider the two separate edges `(0,1)` and `(2,3)`
with phases `(0,0,gamma,gamma)`, unit capacities and forms `(m+a,m,m,m)`.
For `a>0`, `H_i=pi`, `q=(a,-a,0,0)`, so

\[
M_{02}(0)=0,\qquad \dot M_{02}(0)=\frac{wa\sin\gamma}{\beta\pi}>0.
\]

Smoothness supplies a transverse crossing and nearby preparations with the
second component rotated to `gamma+epsilon` that start incompatible and enter
compatibility in finite positive time for sufficiently small `epsilon>0`.
The implicit-function argument also gives
`t_cross(epsilon)=beta*pi*epsilon/(w*a)+O(epsilon^2)`.
At `a=0` each component is an equilibrium and remains incompatible after that
rotation. Reversing `a` reverses the initial crossing direction. Thus this
admission timing is inherited from the complete continuous law; no new
phase-speed equation is imposed.

Both components can rotate independently without changing their internal
dynamics. Their cross-component phase difference therefore requires the
supplied common reference; it is not reconstructible from independently
phase-quotiented observations. Moreover the unit bridge at the displayed
boundary would cost `a^2/2+beta*(1-cos(gamma))>0`. Phase admission alone still
does not make a frozen-state attachment passive or cause its occurrence.

### Integration and the all-operator audit

The shared [reset observer](../../src/tnfr/physics/relational_observations.py)
and `Network.relational_reset(after, storage_scale=...)` retain both supplied
endpoints, actual conductances and phases, plus the shared transport snapshot's
capacity and stored pressure (zero defaults when those attributes are absent).
Those defaults are not evidence of measured zero capacity or pressure. They
reuse the transport reset and represented half-sine phase cost to separate
form/phase changes into state and support contributions. Exact rational
accounting of represented values is not an enclosure of ideal trigonometric
error, authentication of an operator event, or admission of future flow.
The [API contract](../../docs/contracts/RELATIONAL_DYNAMICS.md#relational-reset-observation)
owns the wider snapshot domain and supplied-work assessment.

[Actual UM/RA controls](../../tests/physics/test_coupling_attachment_budget.py),
[AL/EN and phase-admission controls](../../tests/physics/test_relational_contact_admission.py)
and [SDK/export controls](../../tests/sdk/test_relational_reset.py) reuse these
owners. Native AL/EN serialized uniform-real BEPI passes shared signed-scalar
admission directly; no test-only state conversion substitutes for integration.
Rich, complex and unrepresentable scalar inputs still reject.

The [all-thirteen mechanism map](../STRUCTURAL_OPERATORS.md#operator-mechanism-and-activation-audit)
adds IL phase relaxation, capacity-only versus edge-aware VAL/NUL resets,
stored-pressure lifetime, THOL's isolated birth and the three REMESH paths.
It distinguishes implemented writes, eligibility, configured dispatch and
autonomous occurrence. In particular UM can move phase at uniform form where
the relational phase velocity is zero; its reset is not automatically a
continuous step of the selected law.

Sustained rhythm synchronization is a candidate activation premise, distinct
from instantaneous U3 compatibility. The fields already give
`delta_dot=theta_dot_j-theta_dot_i`. Equal rates once do not prove persistence;
equal sustained rates can retain a noncompatible offset. A locking hypothesis
needs a declared time interval or an invariant dynamical condition, a candidate
relation/common reference and a separate reason why locking causes attachment.
Internal precontact evolution above avoids using the future bridge to explain
its own admission; it does not establish the remaining occurrence principle.

### What this opens, and what remains to derive

The two results are complementary: existing form/phase dynamics can create
admission, and an actual nodal action can make a positive-cost connection
compatible with total nonincrease. They are not yet one autonomous trajectory:
their preparations, candidate access and execution contracts differ.

Continuous conductance is another possible model revision, but the native
phase channel uses support independently of weight. A missing edge is not
the limit of a present edge with weight approaching zero for that channel.
Weighting phase too would change the constitutive model. The current operator
reset route can be studied before introducing that separate extension.

An open mechanism is a justified internal activation/contact rule
for a complete action: candidate access and common reference, state reset,
occurrence and clock, and the post-action evolution domain. Emission or
Reception may supply the form contrast used by the phase mechanism, but their
input and work must then be included in the same account. A named operator
does not by itself justify its source or timing. The
[sole plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) defers that
primitive-event question while prioritizing collective geometry below; no
functional-link policy is relabeled as emergent.

<a id="precontact-rhythm-and-locking"></a>
## 13. Precontact rhythm, phase agreement and sustained locking

The hypothesis that a connection is caused by synchronized rhythms first
requires an identified rhythm of the **actual** nodal law. The
[native pulse admission](RELATIONAL_EXCHANGE_ADMISSION.md#relational-pulse-scope)
reuses the existing phase/form modes and storage balance. A capacity is not
an angular velocity; a maintained phase pattern need not be a periodic orbit.
The auxiliary `Network.rhythm()` spectrum and arithmetic pulse studies have
separate laws. No additional primitive pulse variable follows from their names.

### Matching phase and speed can still hide different futures

Fix a candidate port in each independently admitted P2 component. Write the
relative port phase as `c=theta_b0-theta_a0`, retaining a common reference.
The complete fields give `c_dot=theta_dot_b0-theta_dot_a0`. Instantaneous U3
compatibility bounds `abs(wrap(c))`; equality of phase speeds gives `c_dot=0`
at that instant. Neither assertion says that this equality is invariant.

An explicit counterexample needs no forcing or new parameter. Prepare both
pairs with uniform form `m` and common positive capacity `nu`. Their phases
are `(0,a)` and `(0,-a)`, with `0<a<pi/2`. Both candidate ports therefore have
the same form, phase, capacity and instantaneous phase velocity (zero).
Each pair has `q=0`, metric `H=pi*sinc(a)` and nonzero opposite form rates.
Differentiating the existing full field gives

\[
\ddot\theta_{a0}(0)=\frac{2w^2\nu^2 a}{\beta\pi^2\operatorname{sinc}(a)},
\qquad
\ddot\theta_{b0}(0)=-\ddot\theta_{a0}(0),
\]

\[
c(0)=\dot c(0)=0,\qquad
\ddot c(0)=-\frac{4w^2\nu^2 a}{\beta\pi^2\operatorname{sinc}(a)}<0.
\]

Indeed each port has `theta_dot=w*nu*u/(beta*H)`, where `u=x_0-x_1`.
Initially `u=0`, so the differentiated metric term vanishes, while
`u_dot=2*w*nu*delta/pi`. The acceleration comes from retained internal
phase geometry through form evolution, not a hidden external frequency.
The damping term also vanishes at this instant; the result holds for `e>=0`.
The two pairs even have the same consensus tangent spectrum. Matching that
spectrum or a port's current phase/speed cannot certify sustained locking.

### Exact prepared locking does not select attachment

Conversely, let two supplied components be isomorphic with corresponding
capacities and model coefficients. Prepare their full forms/phases related
by the same node correspondence and constant offsets `b,c`:

\[
x_B(0)=P x_A(0)+b\mathbf1,\qquad
\theta_B(0)=P\theta_A(0)+c\mathbf1.
\]

Relabeling and common-offset equivariance, followed by uniqueness on a shared
regular domain, preserve these relations for as long as both solutions exist
there. Their corresponding phase velocities agree at every time. The
components have no cross-edge, however; the existing law keeps their supports
unchanged. This is inherited matching from a supplied preparation, not mutual
entrainment across a missing connection or evidence that locking causes birth.

Any offset `c` is allowed by this independent-component symmetry. Thus sustained
equal rates can coexist with a port separation outside the selected U3 limit.
Even `c=0` supplies no event law. The relative offset is a neutral preparation
freedom: replacing component B by a further constant phase rotation produces
another exact solution. Independent evolution cannot select a unique relative
phase for all those rotated preparations. Adding a contact rule which compares
the components introduces a potential-contact relation and a common reference
that must be declared and justified.

### Existing fine support aligns ports while deforming regional geometry

An NFR understood as a coherent **region** need not have the same relations
as an individual fine node. Distinguish the birth of a primitive graph edge
from an effective interaction between regions on supplied fine support. The
existing directed-triangle [derived-form phase reduction](DERIVED_FORM_PHASE.md#212-a-coupled-amplitude-and-phase-law-derived-from-fine-diffusion)
already illustrates the latter under a different law and observation. Its
contrast angle cannot be silently substituted for primitive relational phase.

There is also a direct causal control in the current joint law, without an
invoked UM event. Reuse the two unit C5 rings, common capacity `nu>0`, uniform
form, winding-one internal twist `kappa=2*pi/5`, and the supplied fine bridge
`(0,5)`. Rotate the second ring by a small positive offset `c`. At that initial
state only the two ports have an ideal nonzero phase source. Define

\[
z=2\cos\kappa+e^{ic}=R e^{i\alpha},\quad
g_0=\alpha/\pi=-g_5,\quad H_p=\pi R\operatorname{sinc}\alpha.
\]

Since `q=0`, all primitive phase velocities initially vanish. Nevertheless
`x_dot_0=w*nu*g_0=-x_dot_5`, so the full-support Laplacian gives
`q_dot_0=4*w*nu*g_0=-q_dot_5`. Differentiating the admitted phase row yields

\[
\ddot\theta_0=\frac{4w^2\nu^2g_0}{\beta H_p},\qquad
\ddot\theta_5=-\ddot\theta_0,\qquad
\ddot c=-\frac{8w^2\nu^2g_0}{\beta H_p}<0.
\]

Here `c` initially equals the port phase difference; once other nodes respond,
one must retain their full state rather than assume a closed rigid-ring angle.
For `0<c<pi/2`, `g_0>0` and all initial edges are acute. Removing the fine
bridge while keeping both prepared rings unchanged makes each a twist
equilibrium, so this acceleration vanishes. The dependency is thus
**existing relation -> phase pressure -> form contrast -> phase response**.
It supplies an interaction-induced port response, not a connection caused
by an already assumed alignment. Crucially, it is not instantaneous alignment
of the entire regions. Each internal neighbor of port 0 has initial phase
acceleration `-w^2*nu^2*g_0/(beta*H_n)`, with `H_n=2*pi*cos(kappa)`, and
the corresponding neighbors of port 5 have the opposite sign. Hence the
difference of the two regional mean phases satisfies

\[
\ddot c_{\rm mean}(0)=-\frac{2w^2\nu^2g_0}{5\beta}
\left(\frac4{H_p}-\frac2{H_n}\right)>0
\]

for sufficiently small positive `c` on C5. At zero offset,
`H_p=pi*(1+2*cos(kappa))>2*H_n` because `cos(2*pi/5)<1/2`.
Thus the ports initially approach while the regional means initially separate.
This is an explicit geometry-dependent deformation, not a contradiction or
evidence that one scalar phase per region closes the future response.

For positive dissipation and sufficiently small offset, the existing
[local recovery theorem](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
also gives convergence to the common aligned-twist equilibrium modulo common
offsets. This reuses its local theorem, not a new quantified basin or a
global synchronization result. The bridge already belongs to the fine model;
interpreting a resulting regional relation as an emergent effective connection
must state that distinction. It does not derive the underlying graph's birth.

### A candidate composite identity, with full internal geometry retained

This suggests a more precise ontological target than primitive edge creation:
two coherent regions may acquire a joint geometric identity on supplied fine
support. The existing [interaction theorem](RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
already establishes the relevant local distinction. Disconnected acute C5
twists have four independent neutral offsets (form and phase per component)
and sixteen stable tangent directions. The joined graph has only two global
neutral offsets and eighteen stable tangent directions under the positive-
capacity/dissipation hypotheses. Relative offsets are restored by the joint
dynamics rather than remaining arbitrary independent choices.

At aligned ports the bridge adds quadratic stiffness
`[(u_0-u_5)^2+beta*(v_0-v_5)^2]/2` to perturbations of form and phase.
Degrees and phase metrics change too. The two lost neutral freedoms therefore
do not define an isolated two-dimensional oscillator; they mix with internal
deformation. The [existing tangent and nonlinear closure results](#1-fixed-model-and-observation)
already prevent replacing this state by just two rigid regional clocks.

The full joined equilibrium geometry, modulo common form/phase origins,
is a **conditional candidate composite identity**: it has a restoring response
to sufficiently small relative perturbations under the same law. Calling it
a closed autonomous coarse NFR additionally requires a sufficient retained
state and justified constitutive reduction; the known hidden-state memory
cannot be discarded. A sustained composite pulse is also a separate claim,
subject to the dissipative/reversible distinction above. Supplied fine support
and prepared component patterns remain explicit premises.

The smallest useful discriminator observes internal shape. For the receiver
ring define `chi_x=x_5-mean(x_6,...,x_9)`. The phase-offset preparation gives
`chi_x_dot(0)=-w*nu*g_0<0`. Independent components or a rigid-region
approximation give zero. This is already a static analytic distinction; a
new run is justified only to test a separately frozen quantitative finite-time
prediction, not to rediscover transmission or claim physical constituents.

### Execution evidence and research consequence

The [contact controls](../../tests/physics/test_relational_contact_admission.py)
differentiate the actual native field for the acceleration counterexample and
advance the shared Euler owner for short prepared matching controls. The former
is a static directional-derivative check; the latter verifies finite execution,
not exact sustained locking or continuous error bounds. The ideal result is
the equivariance/uniqueness argument above. A separate static two-C5 control
checks both signs of the bridge-induced response and its removed-bridge null
case. It does not rerun the completed transmission or recovery campaigns.
No new oscillator, selector,
observation window or phase law is installed. Full `field.phase_rate`, internal
state and the existing port/work observations supply the needed information.

The collective-identity question retains independent-component and rigid-region
controls. Primitive support birth remains unresolved and has its own admission
below; collective organization on finer support does not settle that question.
A threshold, dwell time or phase-slip count can define a configured observation,
but is not derived merely by naming synchronization. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next quantitative admission without reopening completed modal,
transmission or memory calculations.

<a id="connection-mechanisms-and-mediators"></a>
## 14. Connection mechanisms: mediation, reinforcement and primitive birth

The [physical binding comparisons](../PHYSICAL_REGIME_CORRESPONDENCES.md#physical-binding-and-interaction)
motivate a precise distinction. A channel of influence, a dynamically bound
configuration and a primitive support event are different model claims.
Existing interactions can organize a composite without creating a new
fundamental interaction. This does not explain the origin of the fine support.

### A mediated effective connection follows from existing nodal transport

Reuse [exact hidden-state elimination](../DERIVED_EPI_MEMORY.md#3-exact-elimination-including-the-initial-hidden-state)
on unit P3, `A--M--B`, with common held capacity `nu>0` and pure-EPI pressure.
Observe `y=(x_A,x_B)` and hide `h=x_M`. The actual normalized nodal law gives

\[
\dot y=-\nu y+\nu\mathbf1h,\qquad
\dot h=\frac\nu2\mathbf1^Ty-\nu h.
\]

Eliminating the mediator gives the exact endpoint law

\[
\dot y(t)=-\nu y(t)+\nu e^{-\nu t}\mathbf1h(0)
+\int_0^t\frac{\nu^2}{2}e^{-\nu(t-s)}
\mathbf1\mathbf1^Ty(s)\,ds.
\]

Its off-diagonal memory term transmits influence without a direct A--B edge.
For an initial donor-only difference `(delta x_A,delta h,delta x_B)=(a,0,0)`,

\[
\delta x_B(t)=\frac a4(1-e^{-\nu t})^2
=\frac{a\nu^2}{4}t^2+O(t^3).
\]

The recipient's initial rate difference is zero; its acceleration difference
is `a*nu^2/2`.
Removing the mediator paths removes that influence. An instantaneous direct
edge with positive conductance instead gives a nonzero initial recipient rate
difference for `a!=0`. This is a bounded analytic discriminator, not a new
simulation campaign, autonomous edge birth or a bound-state theorem.

Replacing the hidden row by `h=(x_A+x_B)/2` gives a direct reduced coupling,
but here it is exact only on that invariant preparation manifold; the donor-
only control is outside it. The
[minimal realization](../DERIVED_EPI_MEMORY.md#11-minimal-linear-state-retaining-a-declared-observation)
requires three linear coordinates for arbitrary preparations: endpoint
observation rows have rank two, and their first generator products raise the
rank to three. The partition-based `observe_epi_memory` API cannot be called
with the mediator omitted from its partition. Reuse the elimination algebra
or the general linear-realization owner instead. The older unnormalized Kron
example has a different capacity law, as that document already specifies.

The [joint mediator extension](RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction)
now uses the existing nonlinear form/phase law on two C5 regions linked through
one nodal intermediary. It derives a two-coordinate tangent memory, a
capacity-dependent transient and a direct-versus-mediated onset discriminator;
the existing local theorem admits recovery of the supplied joint geometry.
This extends the mechanism beyond pure diffusion without explaining primitive
support origin or replacing a finite capture study.

<a id="mediated-restoring-geometry"></a>
### The same phase geometry induces a restoring relation through a mediator

The [local recovery theorem](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
uses the Hessian of the existing phase storage, with edge weights
`k_ij=cos(delta_ij)>0` at an acute equilibrium. These are derived curvatures,
not new conductances or physical spring constants. Restrict its quadratic
form to a two-edge path `A--M--B`, holding regional shapes fixed and allowing
phase offsets `a,b` at its endpoints and `h` at its mediator. Its contribution is

\[
Q(a,h,b)=\frac\beta2\big[k_1(h-a)^2+k_2(b-h)^2\big].
\]

Completing the square gives the exact quadratic identity

\[
Q=\frac{\beta(k_1+k_2)}2(h-h_*)^2
+\frac{\beta k_{\rm eff}}2(b-a)^2,\qquad
h_* = \frac{k_1a+k_2b}{k_1+k_2},\quad
k_{\rm eff}=\frac{k_1k_2}{k_1+k_2}>0.
\]

Thus eliminating a hidden coordinate in a **static constrained minimum**
leaves positive curvature in the endpoints' relative phase despite no direct
edge. A common offset costs nothing. Other routes and internal deformations
retain their own contributions; this two-edge expression is not the complete
stiffness of the return-path graph.

For an aligned unit path, `k_1=k_2=1`, so `k_eff=1/2`. More than the local
quadratic term is available here. With lifted endpoint difference
`d=b-a` satisfying `abs(d)<pi` and both path gaps acute, its exact constrained
phase-storage minimum is

\[
\min_h\beta[2-\cos(h-a)-\cos(b-h)]
=2\beta[1-\cos(d/2)]
=\frac\beta4d^2+O(d^4).
\]

Indeed the expression before minimization is
`2*beta*[1-cos(d/2)*cos(h-(a+b)/2)]`; its unique minimum in this acute lift
is at the midpoint. This calculation restricts the existing storage rather
than selecting another pressure law or introducing a synchronization threshold.

Static minimization is **not dynamic elimination**. Endpoints and ring shapes
are fixed only for this calculation; that constrained family need not be
invariant under the actual flow. Neither `h=h_*` nor a direct endpoint law
may replace the mediator's form and phase rows without another argument.
The [derived memory](RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction)
retains those rows, their initial state and their capacity. In particular a
zero-capacity mediator can stay away from this constrained minimum: geometric
curvature does not by itself make an inactive node move.
The [native mediation controls](../../tests/physics/test_relational_mediation.py)
check the exact path cost and its gradient against engine observations,
including this zero-capacity boundary, without a trajectory campaign.

Under the full local theorem's positive-capacity premises, phase deformation
drives form through pressure, and form contrast feeds back into phase; form
loss then damps the joint deviation. A restored configuration can remain at
rest with all rates zero. What follows is a conditional restoring collective
relation, not a requirement for perpetual oscillation. Holding fine support
fixed still explains why its edges persist in the model; this curvature
calculation does not explain their primitive birth or physical identification.

### Simultaneous nodal loss can admit continuous reinforcement

The shared [transport derivative](../../src/tnfr/physics/support_transport.py)
and [joint response](JOINT_PARAMETER_RESPONSE.md#10-joint-pressure-response-and-the-capacity-product-rule)
already separate changing conductance from nodal change. Fix a simple bare
support `U`, positive symmetric conductances `a_ij`, strengths `d_i>0`, held
positive `N=diag(nu_i)` and regular `H_U>0`, with `H_U*g_U=-grad(V_U)`.
Let `e>=0`, `w>0`, `beta>0` and declare the conductance/storage scales.
Consider the **explicit weighted extension**, not the unit-only executor,

\[
q=B_a x,\quad
\dot x=N(-eD_a^{-1}q+w g_U),\quad
\dot\theta=(w/\beta)H_U^{-1}Nq,\quad
S=\tfrac12x^TB_a x+\beta V_U.
\]

For held `e,w,beta,nu_i`, differentiating and cancelling the exchange terms gives

\[
\dot S=-e\sum_i\frac{\nu_iq_i^2}{d_i}
+\frac12\sum_{\{i,j\}\in U}\dot a_{ij}(x_i-x_j)^2.
\]

Thus contemporaneous form loss can accommodate positive conductance work
under an additional `S_dot<=0` premise. This is not expenditure of past loss
from an invented reservoir. It is an admissibility inequality, leaving the
allocation, speed and occurrence of reinforcement undetermined. It supplies
no evolution law for `a`, and no weighted executor or global recovery theorem
is installed by this calculation.
For common-capacity P2 with nonzero form contrast it reduces to
`a_dot<=4*e*nu*a`: positive reinforcement can be admitted without selecting
its rate. Conductance scaling leaves normalized form transport unchanged but
changes this extension's phase row at fixed `beta`; the storage budget is not
an independently calibrated physical growth law.

This calculation holds bare phase support fixed. A zero-weight edge already
belongs to that support and participates in its phase pressure; it is not an
absent relation. The existing fixed-active-edge derivative rejects birth from
zero, and a zero-strength row needs separate admission. Adding a genuinely
absent edge also changes `V_U`, `g_U`, `H_U` and support degrees, so its event
needs the [complete reset budget](#nodal-reorganization-and-contact).
At unchanged phase, its added phase cost `beta*(1-cos(theta_j-theta_i))`
does not vanish as the newborn conductance tends to zero. Small transport
weight therefore does not regularize native bare-support birth.
Likewise a regular multiplicative rule `a_dot=a*f` preserves an initially zero
weight when `f` is locally bounded: `a(t)=a(0)*exp(integral(f))`. Naming such a
rule adaptive does not make it a birth mechanism.

### What the repository supplies, and what is still missing

The [relation foundation](RELATION_FOUNDATIONS.md) defines support, conductance,
metric and causal dependence as distinct objects. Its
[zero-relation admission](RELATION_FOUNDATIONS.md#zero-relation-boundary)
shows why a small transport weight need not describe a weak or newly forming
interaction, including the separate first-neighbor normalization boundary.
That foundational contract precedes a proposed law for creating a relation;
the effective-link results below retain their supplied fine support.

UM can add a chord and create a new cycle sector; the existing
[sector-birth control](../FORCED_SUPPORT_BALANCE.md#31-one-added-chord-extends-the-cycle-lattice-not-a-phase-generation-law)
separates that event from simultaneous phase writes and its configured Si
gate. Topological REMESH constructs MST/kNN support from supplied EPI distances
and settings. The runtime's ordinary REMESH gate instead invokes delayed EPI
mixing. Topology and coupling-support observations do not install either law.
These are reusable mechanisms and controls, not a hidden autonomous selector.

With a component-local fixed-support vector field, genuinely disconnected
components have no causal cross-response. Shared initial rhythms do not alter
that fact. A prospective binding model must therefore declare whether its
precursor interaction is an existing fine path, an explicitly supplied field
or candidate relation, or a newly postulated support law. Relabeling a
zero-weight edge or a globally read candidate as no prior interaction conceals
that premise. A pre-material interpretation remains open until this state and
law are justified and a prospective response is derived. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) admits that
mechanism before extending the composite-identity experiment.

<a id="effective-link-admission"></a>
## 15. Sufficient conditions for a restoring effective link

### Operational meaning and fixed premises

Here a **local restoring effective link** means two properties under the same
declared law: interventions in a region's retained form/phase state affect the
other region, and sufficiently small joint perturbations recover the reference
geometry modulo common offsets. This is an operational criterion for the
present model, not a universal definition of an NFR, a microscopic edge-birth
law or a physical spatial binding claim. A reference equilibrium need not
oscillate or carry a nonzero signal to have this response to perturbations.

Fix the [complete relational law](RELATIONAL_EXCHANGE_ADMISSION.md#1-state-inherited-geometry-and-the-independent-premise)
on a finite connected simple unit graph, with held positive capacities,
`e,w,beta>0`, no forcing or events, and an acute equilibrium
`x_*=c*1`, `theta_*` with zero neighbor sine sums. Use local lifted phase
deviations and interleave the coordinates as `z_i=(u_i,v_i)` for each node.
This is a permutation of the native tangent's all-form, then all-phase order.
The following identities concern the ideal equilibrium and real-valued field;
a materialized tangent retains its own numerical residuals.

### A unique shortest path gives a full-pair causal response

Let `J=DF(z_*)`. Its off-diagonal block on an edge `j--i` is

\[
J_{ij}=\begin{pmatrix}
e\nu_i/d_i & w\nu_i\cos(\theta_{*,j}-\theta_{*,i})/H_i\\
-w\nu_i/(\beta H_i)&0
\end{pmatrix},\qquad
\det J_{ij}=\frac{w^2\nu_i^2\cos(\theta_{*,j}-\theta_{*,i})}
 {\beta H_i^2}>0.
\]

Nonadjacent off-diagonal blocks vanish. This follows directly from the
[equilibrium Jacobian](RELATIONAL_EXCHANGE_ADMISSION.md#quotient-linearization-and-the-restoring-mechanism),
not from a graph-wave equation. Suppose distinct nodes `a,b` have a unique
shortest support path `a=v_0,...,v_ell=b`, with length `ell>=1`. Locality gives
`(J^k)_{ba}=0` for `k<ell`. At order `ell`, a contributing walk cannot contain
a diagonal stay or a detour; uniqueness therefore gives

\[
P_{ba}=(J^\ell)_{ba}
=J_{v_\ell v_{\ell-1}}\cdots J_{v_1v_0},\qquad \det P_{ba}>0.
\]

For the flow `Phi_t`, the tangent transfer of the donor's two coordinates to
the receiver's two coordinates is consequently

\[
T_{ba}(t)=D_{z_a}(\Phi_t)_b(z_*)=(e^{tJ})_{ba}
=\frac{t^\ell}{\ell!}P_{ba}+O(t^{\ell+1}),
\]

\[
\det T_{ba}(t)
=\frac{t^{2\ell}}{(\ell!)^2}\det P_{ba}+O(t^{2\ell+1})>0
\]

for every sufficiently small positive `t`. The reversed unique path gives
the reciprocal statement. This is a rank-two response, not a guarantee that
each matrix entry, a chosen scalar sensor or a regional average is nonzero.
For each such fixed time, smoothness and the inverse function theorem also
make the actual nonlinear map from the donor pair to the receiver pair locally
invertible, with other initial coordinates held fixed. The neighborhood may
depend on time; no arbitrary-amplitude or global response follows.

With several shortest paths the leading coefficient is the **sum** of their
ordered block products. Invertibility of each product alone does not establish
invertibility of the sum. The unique-path result is sufficient, not necessary;
the following consensus specialization removes that restriction.

### At consensus, all shortest active paths reinforce the same block

At phase consensus, `H_i=pi*d_i` and every edge cosine is one. Here even held
nonnegative capacities are allowed for the transfer calculation. Put

\[
A=ND^{-1}B,\qquad
T=\begin{pmatrix}e&w/\pi\\-w/(\beta\pi)&0\end{pmatrix},
\qquad J=-A\otimes T,\qquad \det T=\frac{w^2}{\beta\pi^2}>0.
\]

Call the directed step `j -> i` active when `j--i` is a support edge and
`nu_i>0`: the receiving row determines transmission. If the shortest active
path from `a` to `b` has length `ell`, the same walk expansion gives

\[
(J^\ell)_{ba}=s_{ba}T^\ell,\qquad
s_{ba}=\sum_{\substack{a=v_0\to\cdots\to v_\ell=b\\
                         \text{shortest active paths}}}
               \prod_{r=1}^\ell\frac{\nu_{v_r}}{d_{v_r}}>0.
\]

Thus all such paths have the same matrix factor and there is no leading-block
cancellation, regardless of their number. The full-pair small-time conclusion
holds whenever an active path exists. The degrees remain those of the actual
support, including neighbors of zero capacity. In particular, `nu_a=0` does
not prevent a perturbed donor value from acting as a fixed boundary source;
capacity zero freezes its response, not its neighbors' dependence on its state.
This extension concerns transfer only: the whole-network recovery theorem
below still requires strictly positive capacities. Positive `e` is required
there for attraction, not for the displayed off-diagonal determinants.

### A frozen separator is an exact nonlinear causal null

Let a set of zero-capacity vertices separate the donor and receiver in the
support, and hold capacities and support fixed. Compare two admitted solutions
with identical initial separator and receiver-side states, changing only the
donor side. Both rows of every separator node are identically zero under the
selected joint law. Its state therefore stays the same in both solutions.
The receiver-side equations consume only their own evolving state and this
same fixed boundary. Local uniqueness makes their solutions identical for
their common existence interval. The argument is nonlinear and does not use
a tangent approximation. A remaining active route invalidates this null.

The [grounded recovery control](RELATIONAL_PATTERN_MEMORY.md#finite-mediated-response)
shows why this distinction matters: independent regions can each restore a
geometry imposed by one frozen boundary without transmitting changes to one
another. Neither similar shapes nor equal rhythms establishes causal linkage.

### Static effective curvature and dynamic recovery are complementary

Let `K` be the acute equilibrium's cosine-weighted phase Hessian. Retain two
distinct endpoints `R={a,b}` and minimize its quadratic storage over the other
vertices `I`. When `I` is nonempty, connected positive edge weights give
`K_II>0`; the unique constrained minimum has effective Hessian

\[
K_{\rm eff}=K_{RR}-K_{RI}K_{II}^{-1}K_{IR}
=k_{ab}\begin{pmatrix}1&-1\\-1&1\end{pmatrix},\qquad k_{ab}>0.
\]

Indeed the minimized quadratic is nonnegative, vanishes for common endpoint
offsets, and cannot vanish for unequal offsets: a zero full-graph quadratic
requires every phase deviation to be equal. These facts give the displayed
rank-one form and strict coefficient. For two vertices with no interior, use
`K` itself. The minimum phase contribution is
`beta*k_ab*(v_b-v_a)^2/2`. This generalizes the
[two-edge calculation](#mediated-restoring-geometry); it is static curvature,
not a new pressure, conductance or instantaneous evolution law.

Under the positive-capacity premises, the existing
[local recovery theorem and sufficient basin](RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
complete the restoring claim. In its notation, `||z(0)||<r` and
`E_rel(0)<k_r*r^2` keep the joint state in an admitted acute neighborhood and
give convergence to the reference geometry modulo common form/phase offsets.
The effective-curvature coefficient `k_ab` is distinct from that basin bound
`k_r`. The causal theorem plus this recovery result supplies a sufficient
restoring effective link for the stated pairs; no extra synchronization
threshold, pulse variable or operator schedule is required.

If hidden nodes are eliminated dynamically, the resulting effective law must
retain the [derived memory and hidden initial state](RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction).
The [pressure-state obstruction](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure)
also rules out replacing this environment in general by its instantaneous
total pressure: equal pressure can conceal different future responses under
the same law. Zero pressure is not an empty substrate.
Static Schur minimization cannot replace that law. A trajectory entering the
sufficient basin can be said to acquire the maintained joint geometry; proving
entry from another preparation is a separate capture obligation. When a fine
active path already exists, causal influence has no positive waiting interval
in this local ODE argument. A detection threshold does not create its onset,
and neither this result nor capture explains the birth of primitive support.

### An explicit local formation-to-maintenance preparation

The same two unit C5 rings with path `0--10--5` provide a capture statement
without another trajectory calculation. Fix unit capacities,
`e=w=1/2`, `beta=1`, and the aligned winding-one reference of the
[mediator owner](RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction).
Prepare uniform zero form, leave the first ring's phases at the reference,
rotate the entire second ring by `delta`, and set the mediator phase to
`delta/2`. Both internal windings are unchanged. The common phase offset is
`delta/2`, so the initial quotient norm and excess storage are exactly

\[
\|z(0)\|^2=\frac52\delta^2,\qquad
\mathcal E_{\rm rel}(0)=2[1-\cos(\delta/2)]\le\frac{\delta^2}4.
\]

This graph has eleven nodes and diameter six. For any centered vertex vector,
its squared norm is at most `n*(max-min)^2/4`, while Cauchy--Schwarz along a
shortest path between its extrema bounds the graph energy below by
`(max-min)^2/diameter`. Thus `lambda_2(B)>=4/(n*diameter)=2/33`.
The reference acute margin is `m=pi/10`. Taking

\[
r=\frac{\pi}{20\sqrt2},\qquad
c_r=\sin(\pi/20),\qquad
\underline k=\frac{\sin(\pi/20)}{33}\le k_r
\]

in the existing basin theorem proves capture whenever

\[
|\delta|<\min\{r\sqrt{2/5},\ 2r\sqrt{\underline k}\}.
\]

For example, `delta=1/128` satisfies the strict bounds without a floating
evaluation: `pi>3` and `sin(pi/20)>1/10` give `r^2>9/800` and
`underline(k)*r^2>9/264000>1/65536`, whereas the preparation has
`||z(0)||^2=5/32768` and `E_rel(0)<=1/65536`.

The full continuous law therefore keeps this preparation acute and converges
to the aligned joint geometry modulo one common phase and one common form
offset. It restores an initially nonzero relative regional offset and retains
both windings. This establishes local acquisition and maintenance of the
joint geometry on supplied support; it is not creation of the already active
causal path. No monotone regional phase difference, finite-time exact locking,
global capture, autonomous preparation or Euler-trajectory certificate is
asserted. The static midpoint preparation does not keep the mediator or the
ring shapes constrained during this subsequent evolution.

The [focused controls](../../tests/physics/test_relational_effective_link.py)
check the path, consensus and frozen-separator distinctions through the shared
native tangent and field owners. Finite represented checks support integration;
they do not replace the ideal proofs or certify a numerical recovery trajectory.

<a id="environmental-capture-domain"></a>
### A capture domain retaining the intermediary's initial state

Keep the preceding two-C5 plus mediator support, aligned winding-one reference,
unit capacities, `e=w=1/2` and `beta=1`. A wider preparation gives the mediator
form `a` while all ring forms are zero. Leave the first ring's phases at their
reference, rotate the second ring by `delta`, and set the mediator phase to
`delta/2+eta`. These are three supplied initial coordinates, not an invariant
restriction on the later evolution or a new environment law.

The common offsets of the deviations from the reference are `a/11` in form
and `delta/2+eta/11` in phase. Subtracting them gives exactly

\[
\|z(0)\|^2=\frac{10}{11}(a^2+\eta^2)+\frac52\delta^2.
\]

Only the two mediator edges change storage relative to the reference. Their
form contribution is `a^2`, and their phase gaps are `delta/2+eta` and
`delta/2-eta`. Therefore

\[
\begin{aligned}
\mathcal E_{\rm rel}(0)
 &=a^2+2[1-\cos(\delta/2)\cos\eta]\\
 &\le a^2+\eta^2+\frac{\delta^2}4
 =:\mathcal B(a,\eta,\delta).
\end{aligned}
\]

The inequality follows by applying `1-cos(u)<=u^2/2` to each gap. Reuse the
same `r` and `underline(k)` as above and define
`kappa=underline(k)*r^2`. The single sufficient condition

\[
\boxed{\quad \mathcal B(a,\eta,\delta)<\kappa\quad}
\]

implies both basin hypotheses: `E_rel(0)<kappa<=k_r*r^2`, and
`||z(0)||^2<=10*B<10*underline(k)*r^2<r^2`, since
`10*underline(k)<10/33<1`. Thus the initial state is in the acute neighborhood
and its full continuous evolution remains there and approaches the same joint
geometry modulo common offsets. The mediator's nonzero initial form and phase
are included in this theorem, rather than replaced by their equilibrium values.

A wholly nonzero rational example is
`delta=1/256`, `a=eta=1/512`. It has
`B=3/262144<9/264000<kappa`, using the preceding exact lower bound. No
trajectory or transcendental rounding assumption enters this sufficient
admission. Violating this conservative bound establishes neither escape nor
failure to recover; it leaves the stated certificate unavailable.

### Identical mediator pressure does not specify the environment

The family also connects capture to the
[pressure-state obstruction](JOINT_PARAMETER_RESPONSE.md#pressure-state-closure).
In the admitted lift the two port phases are zero and `delta`, so their
resultant points at `delta/2`. The mediator phase source and total pressure are

\[
g_m=-\eta/\pi,\qquad p_m=-ea-\frac w\pi\eta.
\]

Fix the same `delta` and compare the reference intermediary `a=eta=0` with
a compensated intermediary

\[
\eta\ne0,\qquad a=-\frac{w\eta}{\pi e}.
\]

Both preparations have identical ring form/phase coordinates, identical
capacities and the same instantaneous mediator pressure `p_m=0`. Nevertheless
the receiving port has form gradient `q_5=-a`, hence

\[
\dot\theta_5=-\frac{wa}{\beta H_5}\ne0
\]

in the compensated preparation, whereas its phase rate is zero in the
reference preparation. For example, with `c=cos(2*pi/5)` and
`alpha=Arg(2*c+exp(i*(eta-delta/2)))`, the metric is the native
`H_5=pi*abs(2*c+exp(i*(eta-delta/2)))*sinc(alpha)>0`. The distinction uses the
existing phase row and the complete hidden state, not a pressure reconstructed
from the receiver's observed derivative.

At the fixed coefficients, choose `delta=1/256`, `eta=1/512` and
`a=-eta/pi`. Then `B=(2+1/pi^2)/262144<3/262144<kappa`; the reference
intermediary also satisfies the bound. Thus both environments belong to the
same proved capture domain but produce different initial receiver responses.
The claim concerns the **mediator's** pressure, not equality of every nodal
pressure or every subsequent trajectory.

The [existing two-coordinate memory result](RELATIONAL_PATTERN_MEMORY.md#mediated-pattern-interaction)
already proves why both hidden mediator coordinates must be retained for the
specified all-ring tangent observation. This example shows concretely why
replacing them by the single mediator pressure loses information. The memory
owner retains hidden initial state and nonlinear forcing; it does not replace
the mediator by instantaneous static minimization. The initial preparation
family above need not remain rigid, pressure-compensated or otherwise closed
under the full flow.

The [effective-link controls](../../tests/physics/test_relational_effective_link.py)
check these static identities and native response distinctions without treating
finite arithmetic as the continuous capture proof. The
[sole execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent work. Supplied fine support, model premises and preparation
remain explicit; this result does not establish physical vacuum, autonomous
substrate creation or a universal zero-pressure criterion.
