# Predictive local state of two interacting coherent regions

**Status:** exact conditional first-variation closure, nonlinear state and
state-plus-rate counterexamples, a regional response budget, a sufficient
instantaneous attachment interface, conditional support-event budget and
nonselection results, and passive bridge relocation with an explicit shared
recovery domain. These use the admitted relational law; they add no
pressure, phase, capacity or support rule. The
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
