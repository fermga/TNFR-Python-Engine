# Composition, interaction and formation of coherent regions

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
The [source/receiver admission audit](#induced-formation-storage-obstruction)
separates an initial positive-resultant storage obstruction from an unresolved
formation candidate in the wider regular domain. The engine admits that
candidate through its explicit `regular` option; admission does not establish
formation, and the restricted obstruction is not a general formation no-go.
The separate [smooth-sine pattern dynamics](SINE_PATTERN_DYNAMICS.md) owner
contains the bridge and cycle geometry, whole-sector capture, prepared entry,
exact memory and reduction results, and the budget and symmetry restrictions.
Their original section links are retained below as compatibility targets.

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
[proved equilibrium derivative](RELATIONAL_RECOVERY_AND_INTERACTION.md#quotient-linearization-and-the-restoring-mechanism):

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
[weighted phase-cut balance](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-work-integration),
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
[API contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-pattern-observation)
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
[interaction basin](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)
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

Binary64 phase lifts are not exact multiples of ideal pi. Current native
execution captures each relative neighbor resultant once and uses those sums
for both the pressure source and phase metric. The field records
`pressure_path="relative_resultant_canonical"`. Sharing the captured source
does not eliminate the retained pressure-split and rate rounding defects or
bound transcendental evaluation error. Static comparisons use explicit
absolute tolerances for implementation agreement, not a certified ideal-input
or ODE error bound. The fixed C5 controls use binary64, no randomness or
integration step, and absolute rate tolerance \(10^{-15}\) with zero relative
tolerance. Frozen response artifacts retain their recorded arithmetic paths.

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
The [API contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-attachment-observation)
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
[local recovery theorem and explicit barrier](RELATIONAL_RECOVERY_AND_INTERACTION.md#an-explicit-local-domain-and-the-offset-limits)
separately to the two graphs, with no change to its flow or storage premises.
Choose consistent local phase lifts and put

\[
\Pi=I-\mathbf1\mathbf1^T/10,\qquad
\|z\|^2=\|\Pi x\|^2+\|\Pi(\theta-\theta_*)\|^2,\qquad
r=\frac{\pi}{20\sqrt2},\qquad
\mu=\min(1,\beta/10).
\]

Both graphs have ten nodes and diameter five. The same pathwise Cauchy
bound used in the [interaction proof](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)
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
[execution contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-relocation-observation)
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
The [API contract](../../docs/contracts/relational/RELATIONAL_EXECUTION.md#relational-reset-observation)
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
[native pulse admission](RELATIONAL_RESPONSE_IDENTIFICATION.md#relational-pulse-scope)
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
[local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
also gives convergence to the common aligned-twist equilibrium modulo common
offsets. This reuses its local theorem, not a new quantified basin or a
global synchronization result. The bridge already belongs to the fine model;
interpreting a resulting regional relation as an emergent effective connection
must state that distinction. It does not derive the underlying graph's birth.

### A candidate composite identity, with full internal geometry retained

This suggests a more precise ontological target than primitive edge creation:
two coherent regions may acquire a joint geometric identity on supplied fine
support. The existing [interaction theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)
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

The [joint mediator extension](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
now uses the existing nonlinear form/phase law on two C5 regions linked through
one nodal intermediary. It derives a two-coordinate tangent memory, a
capacity-dependent transient and a direct-versus-mediated onset discriminator;
the existing local theorem admits recovery of the supplied joint geometry.
This extends the mechanism beyond pure diffusion without explaining primitive
support origin or replacing a finite capture study.

<a id="mediated-restoring-geometry"></a>
### The same phase geometry induces a restoring relation through a mediator

The [local recovery theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
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
The [derived memory](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
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
[sector-birth control](FORCED_PHASE_LOCKING.md#31-one-added-chord-extends-the-cycle-lattice-not-a-phase-generation-law)
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
[equilibrium Jacobian](RELATIONAL_RECOVERY_AND_INTERACTION.md#quotient-linearization-and-the-restoring-mechanism),
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

The [grounded recovery control](RELATIONAL_MEDIATOR_DYNAMICS.md#finite-mediated-response)
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
[local recovery theorem and sufficient basin](RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
complete the restoring claim. In its notation, `||z(0)||<r` and
`E_rel(0)<k_r*r^2` keep the joint state in an admitted acute neighborhood and
give convergence to the reference geometry modulo common form/phase offsets.
The effective-curvature coefficient `k_ab` is distinct from that basin bound
`k_r`. The causal theorem plus this recovery result supplies a sufficient
restoring effective link for the stated pairs; no extra synchronization
threshold, pulse variable or operator schedule is required.

If hidden nodes are eliminated dynamically, the resulting effective law must
retain the [derived memory and hidden initial state](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction).
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
[mediator owner](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction).
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

The [existing two-coordinate memory result](RELATIONAL_MEDIATOR_DYNAMICS.md#mediated-pattern-interaction)
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

## 16. Formation-enabling support and later pattern detachment

<a id="relational-pattern-detachment"></a>

The two supplied bridges can enable formation without being necessary for
the eventual maintenance of either ring. This distinction follows from the
existing formation, storage and cycle-geometry results; it does not require
another trajectory or an autonomous edge-deletion rule.

### Event and post-event state card

Start with the two unit C5 rings and the two matching bridges at positions
0 and 1 used by the [formation proof](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit).
At a supplied time, delete both bridges and retain every nodal form, phase
and capacity coordinate. The resulting support consists of two C5 components.
Evaluate each component separately with its own newly computed degree,
gradient, resultant and phase metric. The connected relational executor is
not thereby extended to a disconnected graph.

On each component, retain positive held capacities and `e,w,beta>0`, the
native form row and the chosen Dirichlet/cosine storage. The phase law must
be admitted on that **post-event** support: autonomous, `C1`, invariant
under common form/phase shifts, at rest when `q=g=0`, and satisfying
`E_dot<=-c*L` for a fixed `c>0` on the relevant regular neighborhood. No
forcing, further event or capacity change occurs during the claimed
continuation. The same component-local rule can be used before and after
the cut; its actual support-dependent quantities must be refreshed. The
reference, rho and eta laws have this component-local property and the
previously proved loss bounds. A law that reads another component's state
does not acquire independent-component dynamics merely because the graph
was cut.

### A sufficient maintenance basin for one isolated C5

For an oriented C5, let its true wrapped edge gaps be strictly acute and
have winding `s`, with `s=+1` or `-1`. Define

\[
V_*=5[1-\cos(2\pi/5)],\qquad
V_{\rm face}=5-4\cos(3\pi/8),\qquad
E_R=\frac12x_R^TB_Rx_R+\beta V_R.
\]

**Conditional isolated-ring capture.** If

\[
\boxed{\qquad E_R<\beta V_{\rm face},\qquad}
\]

then every law admitted by the preceding state card preserves the acute
winding sector and converges to uniform component form and the winding-s
uniform twist. Convergence is eventually exponential modulo the component's
common form and phase offsets, which have finite limits. Neither reflection
nor uniform capacity is a premise.

For `s=+1`, the five acute gaps sum to `2*pi`. A first boundary gap cannot
be `-pi/2`: the other four cannot supply the required remaining `5*pi/2`.
A boundary gap of `pi/2` leaves sum `3*pi/2` for the other four. Convexity
of `1-cos(delta)` on the closed acute interval gives

\[
V_R\ge1+4[1-\cos(3\pi/8)]=V_{\rm face}
\]

there. Reversing all gaps proves the negative-winding case. Storage
nonincrease therefore prevents a first exit. The strict sublevel is compact
modulo common offsets and stays a positive distance from the acute boundary.
Every resultant and phase-metric entry remains regular and positive.

In the phase quotient the fixed-period acute component is convex, and the
phase Hessian is the cycle Laplacian with positive weights `cos(delta)`.
Its unique critical geometry is the uniform twist: zero phase gradient
makes all oriented edge sines equal, and sine is injective on the acute
interval. The common gap is consequently `s*2*pi/5`.

The strict loss bound and positive capacities force `q=B_R*x_R=0` on the
invariant zero-loss set. To preserve this condition, the unchanged form row
requires `B_R*N_R*g_R=0`, so `N_R*g_R=a*1`. The identity
`sum(H_i*g_i)=0` then gives `a*sum(H_i/nu_i)=0`, hence `a=0` and `g_R=0`.
Rest makes that target invariant. Compactness and LaSalle's principle give
convergence. The [local law-class proof](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-sector-law-class)
uses only connected support, positive capacities, a positive acute phase
Hessian and the full-neighborhood strict loss bound for its Hurwitz step;
those hypotheses hold on this C5. Its exponential-recovery and integrable
common-offset arguments therefore apply independently to each component.
The barrier is a sufficient capture test, not a classification of every
initial state that could recover.

### The joined sector certificate already admits both detached rings

Let `E_total` be the full two-ring storage immediately before deletion.
Suppose the existing [joined acute-sector conditions](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-acute-sector-capture)
hold, including common winding `s=+1` or `-1` and

\[
E_{\rm total}<\beta(V_*+V_{\rm face}).
\]

Write `E_1,E_2` for the two internal ring storages and `C_bridge>=0` for
the two bridge costs. Jensen's inequality gives `E_j>=beta*V_*` for each
acute winding-s ring. Thus

\[
E_{\rm total}=E_1+E_2+C_{\rm bridge},\qquad
E_1\le E_{\rm total}-\beta V_*<\beta V_{\rm face},
\]

and the same calculation applies to `E_2`. Deleting the bridges changes
neither internal phase gaps nor either internal storage. **Every snapshot
admitted by the joined geometric sector test therefore already satisfies
the two isolated-ring capture tests**, provided the post-cut laws and
capacities are admitted separately. No smaller Euclidean neighborhood or
extra numerical formation run is needed. Directly checking the two component
basins may also admit states outside this sufficient joined criterion.

For arbitrary full states the exact event budget is

\[
\Delta E=-\sum_{\{i,j\}\in\mathrm{bridges}}
\left\{\tfrac12(x_i-x_j)^2+\beta[1-\cos(\theta_i-\theta_j)]\right\}
\le0.
\]

This is the [existing deletion identity](#support-event-premise-admission).
The geometric basin conditions supply the additional future-identity
obligation; nonpositive event cost alone does not do so. After the cut,
component means can evolve differently, and common phase offsets need not
remain mutually aligned. The theorem maintains each ring's winding geometry,
not a restoring interaction between disconnected components or a joint
future determined by an aggregate observation.

### Zero bridge cost throughout formation does not make an early cut safe

In the exact copied preparation, corresponding endpoints of each bridge
have equal form and phase throughout the ideal pre-cut flow. Both bridge
costs are therefore identically zero, not merely zero at equilibrium.
Nevertheless, removing them changes each port's degree and resultant even
when its form gradient is unchanged. A zero storage jump does not imply
unchanged pressure, phase metric or future evolution.

The original represented seed supplies an exact early-cut control. Each
ring has phases

\[
(a_0,-a_0,-a_0,0,a_0),\qquad
a_0=\frac{875483625981347}{562949953421312},\qquad
0<a_0<\pi/2.
\]

The ordinary ring winding is zero. On the pure C5, the five resultants are
`1+exp(-2*i*a_0)`, `1+exp(2*i*a_0)`, `1+exp(i*a_0)`,
`2*cos(a_0)` and `1+exp(-i*a_0)`. Their real parts are strictly positive,
so the early cut itself has an admitted regular post-event state. It is
not rejected merely because the domain was undefined.

The [pure-cycle resultant invariant](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-cycle-resultant-obstruction)
now distinguishes it from either maintained acute twist. Around the derived
skip cycle `(0,2,4,1,3)`, its true wrapped gaps are

\[
(-2a_0,\ 2a_0,\ -2a_0,\ a_0,\ a_0),
\]

all strictly inside `(-pi,pi)` and summing to zero. Its skip winding is
therefore zero. Every acute winding-s C5 twist has skip winding `2*s`.
A continuous pure-C5 path with nonzero resultants preserves that skip
winding, independently of the pressure or phase kinetics. Thus the early
detached seed cannot reach either acute winding-one identity along a
regular continuation of any of the named laws. The conclusion does not
assert consensus, finite-time singularity, or the absence of other regular
behavior. It is a precise obstruction to the target already obtained with
the supplied bridges.

### A sufficiently late detachment window follows from formed-state recovery

The [reference formation result](RELATIONAL_FORMATION_CONTROLS.md#relational-validated-transit)
and its [bounded phase-law comparisons](RELATIONAL_FORMATION_CONTROLS.md#relational-formation-law-robustness)
converge to the same joined aligned twist. Its edge gaps are strictly acute,
each ring storage tends to `beta*V_*`, and
`V_*<V_face`. Consequently there is a finite time after which every
pre-cut state lies in the strict joined acute-sector basin and hence in
both inherited isolated-ring basins. A supplied deletion of both bridges
at any such later time retains the formed winding identities under the
admitted component laws. This establishes the existence of a detachment
window without locating its first time or producing a new trajectory.

The previously retained horizon-32 endpoint is certified in the larger
reflected `E<7*beta` basin, not in this smaller acute-sector sublevel. It
must not be silently labeled an already certified cut time. A particular
detachment snapshot needs its own post-cut geometry and law admission.
The conclusion also does not select the cut, its candidates or its clock,
and makes no claim of autonomous substrate birth, physical particles or
the origin of the supplied preparation. It distinguishes support that
enables a formation route from support required for subsequent maintenance.

### Shared observation and implementation scope

[`certify_relational_cycle_capture` and `observe_relational_detachment`](../../src/tnfr/physics/relational_capture.py)
reuse the native field, exact acute-gap/period and cosine-storage kernels.
The detachment report evaluates one connected pre-cut field and two fresh
post-cut component fields, with unchanged primitive state. It retains
component capture certificates, per-node pressure/metric/rate changes and
the shared reset budget without deleting live edges. The reset's represented
wrapped phase cost and the native fields' raw-gap phase costs retain their
separate reconciliation residual. A direct component certificate need not
require that the stronger joined-sector condition passed first.

These APIs certify the selected reference relational model. The broader
strict-loss theorem requires the chosen alternative law's independent
admission; the report does not install or validate an arbitrary phase
callback. The [engine controls](../../tests/test_relational_detachment.py)
cover independent component admission, state-preserving accounting and
early-cut limitations. The [SDK controls](../../tests/sdk/test_relational_detachment.py)
check delegation and export while retaining that same scope. No control
replays the frozen formation trajectory or chooses a cut time.

## 17. Storage admission for a formed source and an initially uniform receiver

<a id="induced-formation-storage-obstruction"></a>

The retained-receiver result in
[pattern memory](RELATIONAL_PHASE_MEMORY.md#relational-retained-receiver-record)
concerns interaction between two already formed winding-one patterns.
Here the receiver initially lacks that phase geometry. The question is
whether a state-preserving supplied connection can form the receiver while
the source also ends in its maintained winding-one sector.

Fix two simple unit C5 rings, unit held capacities, `beta=1`, `e=w=1/2`,
the native unforced relational law and one common structural clock. Initially
all ten nodes have the same uniform form. The source phases are the exact
twist `theta_L,i=i*kappa`, with `kappa=2*pi/5`; the receiver phases are all
`alpha`, where the relative common phase `alpha` is retained as a parameter.
Use either the single bridge `(L0,R0)` or the two bridges
`(L0,R0),(L1,R1)`. Both fine cycles and the candidate bridge set are supplied;
this is an internal formation question, not a construction of nodes or
primitive support.

The attachment changes no nodal coordinate. Its phase-edge storage is part
of the initial joined storage and must be accounted as event work. There
is no form impulse, phase reset, reservoir input or later event in the
continuation considered below.

### The positive-resultant chamber bounds the available initial storage

Write

\[
c=\cos\kappa=\frac{\sqrt5-1}{4},\qquad
V_*=5(1-c),\qquad \delta_j=\alpha-j\kappa.
\]

At each occupied source port the relative resultant is

\[
z_{L,j}=2c+e^{i\delta_j}.
\]

At its uniform receiver partner it is `z_R,j=2+exp(-i*delta_j)`.
Unoccupied source and receiver nodes have positive real resultants `2c`
and `2`. Consequently, membership in the ideal
[`positive_resultant` chamber](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-positive-resultant-execution)
selected by that engine option is equivalent for this family to

\[
2c+\cos\delta_j>0
\]

at every occupied source port. The receiver conditions are automatic.
Each admitted bridge therefore costs strictly less than `1+2c`.
With `k` equal to one or two bridges, the entire initial form storage is
zero and

\[
E_{\rm initial}
=V_*+\sum_{j=0}^{k-1}(1-\cos\delta_j)
<V_*+k(1+2c).
\]

This bound already includes the attachment work. Supplying an additional
unrecorded work budget without changing the declared state or equations
would not change this initial-value problem. The two-port phase gaps are
not independent, but applying this per-port bound remains valid; no claim
that its supremum is attained simultaneously is needed.

### Both maintained acute twists require more storage

The target requires both original rings to have strict acute oriented
edge gaps with winding `+1`, as in their maintained-twist basin. For each
such ring the five gaps sum to `2*pi`. Convexity of `1-cos` on the acute
interval gives phase storage at least `V_*`. Form and bridge storage are
nonnegative, so every target state obeys

\[
E_{\rm target}\ge2V_*.
\]

The fixed-support native law obeys

\[
\dot E=-\frac12\sum_i\frac{q_i^2}{d_i}\le0.
\]

Yet its initial deficit from this necessary target level is bounded below by

\[
\begin{aligned}
2V_*-E_{\rm initial}&>4-7c &&(k=1),\\
2V_*-E_{\rm initial}&>3-9c &&(k=2).
\end{aligned}
\]

Both bounds are strictly positive: `c<1/3`, and in particular
`3-9c=(21-9*sqrt(5))/4>0`. Thus **neither of the two interfaces can reach
the stated target from an initially positive-resultant member of this
equal-form family under the unforced storage-nonincreasing continuation**.
No trajectory or phase-offset search is needed to establish the obstruction.

The proof does not assume the trajectory stays acute. It also cannot be
evaded merely by allowing a later regular excursion outside the positive-real
chamber while retaining the same storage balance: the initial deficit has
already been established. Even replacing the initial strict inequalities
by their nonnegative-real limiting values leaves a positive deficit. Such
a boundary state is outside strict `positive_resultant` admission; it can
still belong to the separate `regular` domain when its imaginary resultant
components keep it away from the excluded ray.

The target qualification matters. An arbitrary nonacute state with ordinary
winding one need not have storage at least `V_*`; winding alone is not this
maintenance target. For example, the positive wrapped gaps
`(7*pi/8,9*pi/32,9*pi/32,9*pi/32,9*pi/32)` sum to `2*pi` but their
cosine storage is below `V_*`, as the shared exact interval controls verify.
The result does not forbid transient receiver structure,
a changed source identity, or formation from another preparation with a
different admitted resource budget.

### The regular option admits a wider initial family

The initial positive-real condition is a computational admission restriction,
not a universal requirement of the already defined
[full regular relational law](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-domain-admission).
This distinction is essential for the two-port case. Consider the same
equal-form preparation and two bridges, but choose

\[
\alpha=\pi+\frac\kappa2.
\]

The two source resultants are then

\[
2c-\cos(\kappa/2)\mp i\sin(\kappa/2).
\]

Their real parts are negative and their imaginary parts nonzero. They are
outside `positive_resultant`, but avoid both the zero resultant and the
negative-real branch itself. Their native metrics
`H_i=pi*Im(z_i)/Arg(z_i)` are finite and strictly positive. The receiver
ports have positive real part `2-cos(kappa/2)`; all other resultants retain
their earlier positive values. Hence this is a locally regular state of
the same mathematical law, with a smooth local flow. Its represented
preparation is admitted by the explicit
[`phase_domain="regular"` option](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-full-regular-execution),
while `positive_resultant` correctly rejects it. The wider option introduces
no new constitutive law; it certifies the domain already used in the derivation.

Its two-bridge cost exceeds the receiver twist's necessary phase storage:

\[
\begin{aligned}
E_{\rm initial}-2V_*
&=2+2\cos(\kappa/2)-V_*\\
&=\frac{7\sqrt5-15}{4}>0.
\end{aligned}
\]

This is a counterexample to extending the preceding energy deficit to all
regular initial states. It is not a formation trajectory: local regularity
and sufficient total storage do not prove global continuation, preservation
of the source, crossing into the target sector or capture. With only one
bridge, its cost is at most two, so `V_*+2<2V_*` still obstructs the
equal-form target even without the initial positive-real restriction.

The scoped conclusion therefore separates a proved obstruction for both
initially positive-resultant interfaces from an unresolved two-port formation
candidate admitted by `regular`. Treating the computational chamber
as a physical law, or treating the wider state's energy surplus as successful
formation, would conflate different obligations. The shared regular engine
retains certified point/chord admission and represented pressure/work defects;
these are not an exact trajectory enclosure or target-entry certificate.
The acute default and historical evaluated protocols remain unchanged.
The [regular reachability audit](#regular-seeded-reachability-audit) below
resolves the component and necessary-crossing questions for the fixed wider
preparation; its continuous trajectory and maintenance remain separate.

The shared
[`certify_relational_seeded_formation_obstruction`](../../src/tnfr/physics/relational_capture.py)
returns a `RelationalSeededFormationObstruction` with separate interface
cases, exact outward storage bounds and an explicit `obstructed` verdict.
It has no successful-formation or ordinary capture-admission interpretation.
The [capture API controls](../../tests/test_relational_capture.py) check the
shared report contract. The
[independent formation controls](../../tests/physics/test_relational_seeded_formation.py)
verify the radical gaps, native initial budgets and full-regular metric
witness. The [regular execution controls](../../tests/test_relational_regular_execution.py)
check its admission by `regular` and rejection by `positive_resultant`.
Neither certificate nor control advances a trajectory or changes live support.

## 18. Regular-domain geometry of the fixed source/receiver preparation

<a id="regular-seeded-reachability-audit"></a>

Freeze section 17's two-port preparation with `alpha=6*pi/5`, all forms zero,
unit held capacities, `beta=1`, `e=w=1/2`, no forcing and no subsequent event.
Write `A_*=4*pi/5`, `c=cos(2*pi/5)` and `V_*=5*(1-c)`. Its total joined
storage, including the supplied attachment, is

\[
E_0=V_*+2+2\cos(\pi/5)=\frac{35-3\sqrt5}{4},\qquad
E_0-7=\frac{7-3\sqrt5}{4}>0.
\]

The last inequality follows from `49>45`. This audit distinguishes regular
geometric connectivity, a necessary phase crossing, and the actual unforced
initial-value problem. No phase path below is imposed on that problem.

### The unchanged law preserves a reflection subspace

On each ring, reflection `j -> 1-j` modulo five exchanges ports zero and one,
exchanges nodes two and four, and fixes node three. The joined graph is
invariant under the simultaneous reflection of both rings. Compose that
permutation with `x -> -x` and `theta -> 2*alpha-theta` modulo `2*pi`.
The frozen preparation is fixed by this transformation. The full regular
law commutes with it: relative resultants are conjugated, `g` changes sign,
`H` is unchanged, and both evolution rows transform with the corresponding
signs. Smooth uniqueness therefore preserves this subspace for as long as
the solution stays regular.

Use continuous phase lifts based at the fixed node's phase `alpha`. The
source and receiver then have the respective coordinates

\[
\begin{aligned}
x_L&=(p,-p,-r,0,r),&\quad \theta_L-\alpha\mathbf1&=(a,-a,-b,0,b),\\
x_R&=(P,-P,-R,0,R),&\quad \theta_R-\alpha\mathbf1&=(A,-A,-B,0,B).
\end{aligned}
\]

Initially `(a,b,A,B)=(A_*,A_*/2,0,0)` and all four form coordinates vanish.
The degree-two middle node has resultant `2*cos(b)` on the source and
`2*cos(B)` on the receiver. Regularity excludes both zero and negative real
resultants, so the lifts stay in `|b|,|B|<pi/2`. The other degree-two
source resultants are

\[
z_{L,2}=2\cos(a/2)e^{i(b-a/2)},\qquad z_{L,4}=\overline{z_{L,2}}.
\]

A continuous lift cannot cross `a=+/-pi` without making them zero. Thus
`|a|<pi`; likewise `|A|<pi`. Within these bounds, `|b-a/2|<pi` and its
receiver counterpart hold automatically. The remaining domain conditions
are exactly the two port-resultant conditions and their conjugates:

\[
\begin{aligned}
z_{L,0}&=e^{-2ia}+e^{i(b-a)}+e^{i(A-a)},\\
z_{R,0}&=e^{-2iA}+e^{i(B-A)}+e^{i(a-A)}.
\end{aligned}
\]

These reductions follow from the supplied symmetry; they are not new
primitive coordinates or a closure for arbitrary asymmetric states.
They concern the exact ideal preparation with mathematical `pi`.
Independently materializing its phases as binary floating values can break
exact reflection. Point/chord admission by the regular executor does not
itself certify this symmetry or a continuous reduced trajectory for those
represented coordinates.

### Receiver formation must cross an internal cancellation

In these lifts a strict acute winding-one ring necessarily satisfies
`a>3*pi/4` and `b>a-pi/2>pi/4`; replace lower-case coordinates by capitals
for the receiver. Indeed, the oriented gaps are the wrapped versions of
`(-2*a,a-b,b,b,a-b)`. Acute admission forces `|a-b|<pi/2`, and winding one
then requires `wrap(-2*a)=2*pi-2*a` in `(0,pi/2)`. Consequently every such
target has `a+b>pi` and `A+B>pi`. The initially uniform receiver must cross
`A+B=pi` before reaching this target.

At that crossing its two internal neighbor phasors cancel at both ports:

\[
e^{-2iA}+e^{i(B-A)}=0,\qquad
z_{R,0}=e^{i(a-A)},\qquad z_{R,1}=e^{-i(a-A)}.
\]

An isolated C5 would have zero resultants there. The supplied bridge
neighbors instead leave unit nonzero resultants. They are regular whenever
the bridge gap is not antipodal. In particular, if the source already has
`a+b>=pi`, the lift bounds imply `a,A>pi/2`, hence `|a-A|<pi/2`; the receiver
port resultants even have positive real part at this necessary crossing.
This conclusion admits those ports, not automatically every other node.

The pure-cycle skip-winding invariant therefore does not forbid this joined
crossing. Only the two occupied ports can bypass the isolated-ring
cancellation: the other three nodes still have exactly their two ring
neighbors. This identifies the role of the supplied support without deriving
an autonomous contact or a crossing time.

The necessary boundary itself contains a regular state below the frozen
initial budget while keeping the source twist unchanged. Take

\[
(a,b)=(4\pi/5,2\pi/5),\qquad (A,B)=(3\pi/4,\pi/4),\qquad x=0.
\]

Then `A+B=pi` and the bridge gaps are `+/-pi/20`. The source ports have
positive real part `2*c+cos(pi/20)`; the receiver ports are exactly
`exp(+/-i*pi/20)`. The receiver's other resultants are
`2*cos(3*pi/8)*exp(+/-i*pi/8)` and `sqrt(2)`, all with positive real part.
Its ring storage is `5-sqrt(2)`, so the joined boundary storage `E_b` obeys

\[
E_0-E_b
=2\cos(\pi/5)+2\cos(\pi/20)-5+\sqrt2
>\frac{3}{400}>0.
\]

For an exact elementary bound, use `cos(u)>=1-u^2/2`,
`sqrt(5)>559/250`, `sqrt(2)>7071/5000` and `pi<22/7`. They give the strict
lower bound `1839/245000=3/400+3/490000`; the square-root bounds follow by
squaring the positive rationals. Thus the required cancellation boundary
is not everywhere above the available initial storage. This point is not
a reached state or a demonstrated connection within a nonincreasing-energy
path; preceding dissipation and all intermediate states still matter.

### The preparation and target share a regular storage sublevel

There is an explicit continuous geometric path with zero form throughout
from the preparation to the aligned two-twist target, entirely within
`E<=E_0`. It has two legs. This construction does not keep storage monotone
and does not preserve the source winding throughout.

First hold the receiver uniform, `A=B=0`, and decrease the source coordinates
`(a,b)=(u,u/2)` from `u=A_*` to zero. Its degree-two resultants are positive
real. The receiver ports have `z=2+exp(+/-i*u)`, with real part at least one.
The source port zero has

\[
z=e^{-2iu}+e^{-iu/2}+e^{-iu},\qquad
-\operatorname{Im}z=\sin(2u)+\sin(u/2)+\sin u
=\sin(u/2)(8h^3-2h+1),\quad h=\cos(u/2).
\]

Here `h>=c>3/10`. The polynomial `8*h^3-2*h+1` is increasing for
`h>=3/10` and at `3/10` equals `77/125>0`. Thus this imaginary part is
strictly negative for `0<u<=A_*`; the conjugate port has the opposite sign.
At zero both resultants equal three. The whole first leg is regular even
where a source port's real part is negative.

Define the phase storage of one ring on this curve as

\[
V(u)=5-\cos(2u)-4\cos(u/2).
\]

The joined storage on the first leg is

\[
F(u)=V(u)+2(1-\cos u),\qquad
F'(u)=2[\sin(2u)+\sin(u/2)+\sin u]>0
\]

for `0<u<=A_*`. Decreasing `u` therefore lowers storage from `F(A_*)=E_0`
to zero. The common endpoint is joint phase consensus.

For the second leg increase both rings together:
`a=A=u`, `b=B=u/2`, from zero to `A_*`. Both bridge gaps vanish. Port
resultants satisfy

\[
\operatorname{Re}(1+e^{-2iu}+e^{-iu/2})
=1+\cos(2u)+\cos(u/2)\ge\cos(u/2)\ge c>0.
\]

The interior resultants are again positive. This is the previously derived
[added-support regular path](RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-regular-winding-crossing),
in the reflection coordinates. Its total storage is `2*V(u)`. Since

\[
V'(u)=2\sin(u/2)(8h^3-4h+1)
=2\sin(u/2)(2h-1)(4h^2+2h-1),
\]

and `h` decreases from one to the positive root `c` of the last quadratic,
`V` increases up to `u=2*pi/3` and then decreases to its twist value.
The maximum joined storage is exactly `2*V(2*pi/3)=7<E_0`.
Every resultant stays regular on the compact two-leg path. Thus neither
disconnected regular components nor an intermediate minimum storage barrier
above `E_0` can rule out this initial/target pair without additional
constraints on the path.

This is not a trajectory of the unforced law. On the second leg storage
increases from zero, whereas an actual trajectory must obey `E_dot<=0`.
Furthermore the first leg unwinds the source before the second leg rebuilds
both ring geometries. The construction cannot certify formation while
preserving the source identity at all times. It establishes geometric
connectivity in a common sublevel, and nothing stronger about the selected
initial-value problem or the dissipation it accumulates.

### Eight coordinates retain the actual symmetric dynamics

The invariant restriction closes exactly without equating the source and
receiver. Define the full-graph form gradients at ports zero and four by

\[
q_L=4p-r-P,\quad v_L=2r-p,\qquad
q_R=4P-R-p,\quad v_R=2R-P.
\]

For the source let `g_L=Arg(z_L,0)/pi`,
`H_L=pi*Im(z_L,0)/Arg(z_L,0)` with its positive-real extension. The interior
row has

\[
g_{L,4}=\frac{a/2-b}{\pi},\qquad
H_{L,4}=2\pi\cos(a/2)\operatorname{sinc}(a/2-b).
\]

Use the same expressions with `(a,b)` replaced by `(A,B)` for the receiver.
The existing full law restricts to

\[
\begin{aligned}
\dot p&=-\frac e3q_L+w g_L,&
\dot r&=-\frac e2v_L+w g_{L,4},\\
\dot a&=\frac w\beta\frac{q_L}{H_L},&
\dot b&=\frac w\beta\frac{v_L}{H_{L,4}},
\end{aligned}
\]

and the four receiver rows obtained by exchanging upper/lower-case
coordinates. All other nodal rows follow by reflection; both fixed-node
rows are zero. This is an exact invariant restriction of the ten-node law,
not an assumed coarse law or an elimination of hidden state. Its form storage
is

\[
E_D=2p^2+(p-r)^2+r^2+2P^2+(P-R)^2+R^2+(p-P)^2.
\]

With `V(a,b)=5-cos(2*a)-2*cos(a-b)-2*cos(b)`, total storage is
`E_D+beta*[V(a,b)+V(A,B)+2*(1-cos(a-A))]`. The inherited loss is

\[
\dot E=-e\left[\frac23(q_L^2+q_R^2)+v_L^2+v_R^2\right].
\]

The shared native field already evaluates these equations after full-state
reconstruction. A future interval proof must admit both port resultants and
the degree-two rows throughout its tubes. The existing copied-ring,
positive-resultant transit proof has different hypotheses: it identifies the
two rings and retains only four coordinates. Applying that solver here would
discard the source/receiver difference that drives the question.

### The initial native response is not the constructed path

Put `s=sin(pi/5)`, `d=cos(pi/5)` and

\[
\rho=\frac1\pi\arctan\frac{s}{2-d}>0,\qquad
h_L=\frac{5s}{3},\qquad h_R=\frac{s}{\rho}.
\]

In original node order, the initial phase sources and metrics are

\[
g_L=(-3/5,3/5,0,0,0),\quad g_R=(\rho,-\rho,0,0,0),
\]
\[
H_L=(h_L,h_L,2\pi c,2\pi c,2\pi c),\quad
H_R=(h_R,h_R,2\pi,2\pi,2\pi).
\]

Equal zero form gives `x_dot=g/2` and `theta_dot=0`, but the next phase
derivative is nonzero. Write

\[
Q_L=\frac65+\frac\rho2,\quad Q_R=\frac3{10}+2\rho,
\quad \ell=\frac{Q_L}{2h_L},\quad k=\frac3{40\pi c},
\quad m=\frac{Q_R}{2h_R},\quad n=\frac\rho{8\pi}.
\]

Since `q_dot=B*x_dot` and the initial `q` vanishes, differentiation of the
native phase row gives

\[
\ddot\theta_L=(-\ell,\ell,-k,0,k),\qquad
\ddot\theta_R=(m,-m,n,0,-n).
\]

Thus `a_ddot=-ell`, `b_ddot=k`, `A_ddot=m`, `B_ddot=-n`. In particular,
the receiver starts deforming even though its initial phase velocity is zero.
Neither of the prescribed geometric paths supplies these accelerations.
These rates can also be evaluated by multiplying the existing uniform-form
joint tangent by the initial full field; no finite-step difference or
additional evolution rule is needed.

For the source port zero, let `C_0=2*c-d`, `S_0=-s` and
`psi=Arg(z_L,0)`. Initially `psi=-3*pi/5` and all first phase derivatives
vanish. Direct differentiation gives

\[
\begin{aligned}
\ddot C_0&=s(\ell+m)-\sin(2\pi/5)(\ell-k),\\
\ddot S_0&=c(3\ell+k)-d(\ell+m),\\
\ddot\psi(0)&=
\frac{C_0\ddot S_0-S_0\ddot C_0}{C_0^2+S_0^2}<0.
\end{aligned}
\]

The shared outward interval controls enclose the last derivative near
`-0.204238` and `S_0_ddot` near `0.041121`, with strict signs. The source
argument initially moves toward `-pi`; since `C_0<0`, its distance to the
excluded ray also initially decreases at second order. This is a local
direction, not a proved finite-time collision or loss of regularity.

There is initially no dissipation because `q=0`, but storage is not conserved.
Define

\[
D_2=\frac{Q_L^2+Q_R^2}{3}+\frac9{200}+\frac{\rho^2}{8}>0.
\]

Smoothness and the exact loss identity imply

\[
E(t)=E_0-\frac{D_2}{3}t^3+O(t^4).
\]

An initial zero storage derivative therefore does not certify equilibrium or
free evolution. Without an explicit remainder bound, this local expansion
cannot give a target-budget exhaustion time, branch-contact time or future
formation verdict.

The [static reachability controls](../../tests/physics/test_relational_seeded_reachability.py)
reuse the shared field, uniform tangent and rational interval kernels. They
check the exact symmetry, crossing budget and initial derivative signs against
independent expressions. They do not integrate this new initial-value problem,
modify a frozen response, install an event or certify a finite-time outcome.
The [single queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the subsequent whole-time validation obligation.

## 19. A bounded continuous response of the fixed regular preparation

<a id="regular-seeded-continuous-response"></a>

The [reflected field](../../src/tnfr/physics/_relational_reflected_flow.py)
implements section 18's exact eight-coordinate restriction for interval states
and Taylor jets. The [continuous proof owner](../../src/tnfr/physics/relational_reflected_transit.py)
consumes that field with the shared Picard/Taylor/comparison kernel. This is a
proof calculation on the ideal mathematical-pi initial-value problem, not an
Euler trajectory or a projection of separately rounded phases onto symmetry.
The same field is checked against the native ten-node owner at independently
prepared represented states. Native execution and continuous enclosure keep
their separate arithmetic provenance.

### Whole-time admission and storage accounting

For each initial box `X`, a convex tube `Y` must satisfy strict inclusion
`X+[0,h]*F(Y)` inside `Y`. Every full resultant and the declared continuous
reflection lifts are admitted on that tube. This establishes existence and
uniqueness throughout the step for every initial point in `X`. In particular,
the true exact-pi preparation is enclosed without identifying pi with a float.

The center solution is bounded by its order-six Taylor polynomial plus an
order-seven derivative remainder on `Y`. Normalized interval jets obtain those
derivatives from the same law. A Metzler matrix of Jacobian bounds on `Y`
propagates initial radii; its shared nonnegative-series comparison bound adds
outward rounding and a rigorous tail. Intersecting the endpoint enclosure with
the independently proved whole tube removes only overestimation. The older
four-coordinate transit owner now delegates this generic step arithmetic to
the same kernel, preserving its separate model and capture contract.

The six independent nodal resultants use interval principal arguments. If
`Re(z)>0` throughout a box, the inverse metric is evaluated as
`atan_ratio(S/C)/(pi*C)`, including the removable `S=0` limit. Otherwise a
separated imaginary sign permits `Arg(z)/(pi*S)`. The shared argument jet
differentiates `(C*S'-S*C')/(C^2+S^2)`; scalar function error is not substituted
for derivative uncertainty. Unresolved rectangles reject the proof step.

For each accepted tube, integrating the interval loss gives
`h*D(Y)`; summing these intervals encloses accumulated loss. The direct endpoint
storage is intersected with `E(0)-integrated_loss`, and the corresponding loss
is tightened by the same identity. A disjoint intersection reports failure.
The target budget is `E(t)-2*V_*`: a positive lower bound only retains this
necessary resource; a negative upper bound would exclude subsequent entry to
the two acute twist sectors under the same unforced loss law. Ordinary winding
flags are checked on every whole tube, separately from target capture.

### Frozen prediction and separately identified numerical correction

The [producer](../../benchmarks/relational_seeded_response.py) freezes the
existing preparation, support, unit capacities, `e=w=1/2`, `beta=1`, no inputs
or events, structural horizon `T=1/8`, step `h=1/64`, Taylor order six and
128-bit outward interval arithmetic. Its endpoint-width budget is `2^-20`.
Before evaluation it predicts an admitted full window, positive receiver port
phase `A(T)`, a source-port argument below its initial `-3*pi/5`, positive
accumulated loss, positive remaining target budget and winding `(1,0)` throughout.
None is a prediction of receiver formation within that short window.

The first record, `response-v1.json`, is retained as **inconclusive**. It
certifies two steps to `t=1/32` but its next comparison bound exceeds the
kernel's admitted norm budget. The responsible interval expression used
`Arg(z)/S` even for tiny sign-separated `S` with positive `C`. Its removable
singularity was mathematically harmless but inflated derivative intervals.
A separate static control reproduces this issue and verifies the equivalent
positive-real `atan_ratio` expression against an independent local expansion.
No constitutive law, initial state, horizon, step, Taylor order or prediction
was changed to obtain a different physical result.

The separately frozen `response-v2.json` records that correction and the v1
record's digest. It passes all declared gates, with eight accepted steps to
`T=1/8`. This is a corrected computational verification, not an independent
blind replication or a replacement of the first verdict. Outward rounded
summaries of its exact retained intervals are:

| Observation | Certified enclosure or verdict |
| --- | --- |
| Receiver port phase `A(T)` in the fixed frame | `(0.000548810055118, 0.000548810055123)` |
| Source-port argument change | `(-0.001570247840, -0.001570247837)` |
| Accumulated storage loss | `(0.000423056798, 0.000423056800)` |
| Endpoint total storage | `(7.072525960075, 7.072525960078)` |
| Remaining two-twist target budget | `(0.162695903824, 0.162695903828)` |
| Smallest retained whole-tube resultant/ray margin | greater than `0.58425` |
| Maximum endpoint coordinate width | less than `1.424e-13` |
| Whole-window ordinary winding | source `+1`, receiver `0` |

The receiver therefore begins responding while remaining in its zero-winding
sector over the entire certified window. The source approaches its branch
locally but does not encounter it during this interval. The retained positive
target budget does not select a future route or admit a maintenance basin.

The local evidence bundle is under
`artifacts/research/relational_seeded_response/`, with separate protocol and
source-archive siblings for each response. SHA-256 record identifiers are:

```text
response-v1.json  d65eb3c9fc88979d71d6abed3178a1f02307b5b37b4cb6bd0d077cdfd4c0e867
response-v2.json  183c762fa4853a1f42c566a3683eb948263e0327f60c50819684cf4bb51ece75
```

The hashes bind retained bytes, not chronology, physical authenticity or
dependency binaries. [Static flow controls](../../tests/physics/test_relational_reflected_flow.py),
[proof admission](../../tests/test_relational_reflected_transit.py) and
[producer plumbing](../../tests/test_relational_seeded_protocol.py) remain
separate from evaluating this response. The optional
[retained-response audit](../../tests/physics/test_relational_seeded_response.py)
checks both archives, strict Picard inclusion, endpoint chains, storage/loss
and saved gates without replaying the IVP; it skips absent local bundles.
The single execution plan owns any
new continuation; these artifacts must not be regenerated to follow unrelated
source changes.

## 20. Fixed-IVP continuation and the remaining target budget

<a id="regular-seeded-budget-continuation"></a>

The separately frozen horizon-one computation retained exactly section 19's
initial state, law, fixed support, clock, interval precision, step `1/64`,
Taylor order six and endpoint-width budget `2^-20`, with total horizon `T=1`.
It started from the original mathematical-pi enclosure, not a rounded retained
midpoint.
The shared producer's `target-budget` study froze this declaration and its
source archive before evaluation. The earlier records remain separate.

The discriminant is `B(T)=E(T)-2*V_*`, with
`2*V_*=10*(1-cos(2*pi/5))`. Section 17's acute-ring bound and the same unforced
loss identity imply the following conditional outcomes:

- An upper bound below zero excludes later entry into states in which both
  original rings are strictly acute and have ordinary winding `+1`, wherever
  the same fixed-support regular continuation exists.
- A lower bound above zero retains only the necessary storage resource. It
  proves neither a route, ordinary winding change nor target capture.
- An interval containing zero leaves this test unresolved, including exact
  equality. Nonacute winding configurations need not obey the acute threshold.

Numerical horizon completion and width admission are separate from that sign.
Either strict sign is an informative numerical verdict. A failed tube cannot
prove a singularity; an earlier certified negative budget remains a scoped
exclusion even if a later endpoint is unavailable. The report retains the first
such certified time, full storage/loss evidence and independent winding flags.
False winding flags indicate missing certification, not proved winding change.
The frozen stop condition allowed one retained continuation verdict,
without automatic horizon extension or an adjusted law.

### Retained horizon-one result

The separately frozen `budget-t1-v1.json` completes all 64 steps through `T=1`
with no failed tube or evaluation error. Its three numerical gates pass:
full-horizon admission, endpoint width and a strict budget sign. The sign is
positive, so the scoped verdict is **target not excluded by storage**. This
does not certify target formation. Outward decimal summaries are:

| Observation at `T=1` | Certified enclosure or verdict |
| --- | --- |
| Remaining target budget | `(0.032840217948, 0.032840218115)` |
| Accumulated loss | `(0.130278742510, 0.130278742677)` |
| Total storage | `(6.942670274198, 6.942670274366)` |
| Receiver port phase `A` | `(0.024811891363, 0.024811891367)` |
| Source port phase `a` | `(2.259008131438, 2.259008131465)` |
| Source-port argument change from preparation | `(-0.062890231290, -0.062890231102)` |
| Maximum endpoint coordinate width | less than `2.396e-11` |
| Smallest retained whole-tube domain/lift margin | greater than `0.29240` |
| Whole-window ordinary winding | source `+1`, receiver `0` |

The budget has fallen from approximately `0.16312` initially to `0.03284`.
The initial response therefore persists beyond the earlier window without
receiver winding formation. In particular, preservation of the source's
ordinary winding does not imply preservation of its acute ring geometry.
At the endpoint its wrapped port-to-port gap is `2*pi-2*a`, enclosed in
`(1.765169044251, 1.765169044302)`, strictly above `pi/2`. The saved endpoint
at `49/64` already has a strictly nonacute source gap; this statement does not
locate the first exact crossing time. The source can deform outside its
acute ring sector while all full nodal resultants remain regular.

A read-only evaluation of the retained endpoint gives
`D(1)` in `(0.321460727563, 0.321460727582)`. This positive instantaneous loss
and the small remaining resource motivated the separately frozen nine-eighths
test below. Their ratio is not a certified exhaustion time: the loss rate
evolves. These horizon-one observations alone do not establish a later
trajectory, singularity or formation result.

The evidence bundle remains in `artifacts/research/relational_seeded_response/`:

```text
budget-t1-v1.json           f142e75070b141644541e91ac549e7fa9d6c5bfbce8700735130b870a6a0c483
budget-t1-v1.protocol.json  5c748341025197b1c6219c2b553b37f29bee8199ca1088edf157e33154189bf4
budget-t1-v1.source.zip     37e2150bed71757de7399e03bf13158919c090330af1b5909784f7cfa66f059d
```

The [budget disposition controls](../../tests/test_relational_seeded_budget.py)
check strict signs, partial horizons and protocol admission without evolving
this IVP. The [optional retained audit](../../tests/physics/test_relational_seeded_response.py)
checks the archive, interval chain, loss balance and independent geometry
without replay. Record digests establish byte consistency, not independent
physical evidence or authenticated chronology. The single execution plan
owns any new continuation.

### Prospective extension to nine eighths

The separately frozen nine-eighths record retained the same state, law, clock,
step, order and width budget, declaring `T=9/8` through the shared
`target-budget` protocol. It started from the original exact initial enclosure.
The strict budget-sign test is unchanged; either resolved sign is informative.
The horizon and source archive were frozen separately before evaluation, and
no adaptive continuation or replay of the earlier evidence was used to choose
the result.

The retained `budget-t9over8-v1.json` completes all 72 steps with all numerical
gates admitted. Its final target budget lies in
`(-0.010524533169, -0.010524532905)`: the specified two-acute-twist target is
excluded under subsequent unforced fixed-support regular continuation.
The first saved endpoint certifying a negative budget is `71/64`; the previous
one, `35/32`, is still positive. These endpoints bracket a budget crossing,
not an exact exhaustion time or a phase singularity.

At `9/8`, total storage lies in `(6.899305523081, 6.899305523346)` and accumulated
loss in `(0.173643493529, 0.173643493795)`. Maximum coordinate width is less than
`3.796e-11`; all retained domain/lift margins exceed `0.28885`. Ordinary
winding remains `(1,0)` throughout. The first 64 steps and observations equal
the preceding horizon-one record exactly. This calculation closes the declared
budget test for this preparation; it does not exclude receiver formation
accompanied by source loss, nonacute patterns, other preparations or laws.

```text
budget-t9over8-v1.json           d3174948f69f600114a3dce2e4915d79cef75230c7a7c7100b6da7de0c6ef332
budget-t9over8-v1.protocol.json  a278574a78e65cd3d55376256f4e230eebb631ac645d34884907eb51b1c8cf1d
budget-t9over8-v1.source.zip     b76ce10679b49c180b21bd5491e2fd10a24922727e0ba342302df0097990f92a
```

## 21. A sharp collective energy envelope and transition barrier

<a id="reflected-collective-energy-barrier"></a>

Reusing the exact reflection, edge storage and nonincrease law gives a stronger
obstruction than the target minimum. This is a retrospective theorem applied
to retained evidence, not a changed prediction or replacement of section 20's
original verdicts. The argument requires the same fixed two-C5 support,
matching bridges, reflected state and regular unforced law with `e>=0`,
`w,beta>0`. It supplies no new pressure term or evolution rule.

### The copied geometry is a sharp envelope, not a closed mean dynamics

Introduce exact coordinates

\[
m=\frac{a+A}{2},\quad d=\frac{a-A}{2},\quad
u=\frac a2-b,\quad v=\frac A2-B.
\]

The full phase potential from section 18 satisfies the identity

\[
\begin{aligned}
V={}&W(m)+4\cos^2(m)(1-\cos(2d))\\
 &+8\cos(m/2)(1-\cos(d/2))\\
 &+4\cos(a/2)(1-\cos u)+4\cos(A/2)(1-\cos v),\\
W(m)={}&10-2\cos(2m)-8\cos(m/2).
\end{aligned}
\]

To derive it, use
`cos(a-b)+cos(b)=2*cos(a/2)*cos(a/2-b)` on each ring, then collect the two
port terms and both bridge costs using `a=m+d`, `A=m-d`. In the inherited
lift `|a|,|A|<pi`, all four remainders are nonnegative. Nonnegative form
storage therefore gives the sharp bound

\[
\boxed{\quad E\ge\beta W(m).\quad}
\]

Equality is attained by zero forms, `a=A=m` and `b=B=m/2`. Thus the copied-ring
phase profile already used in the restricted transit study also bounds the
full unequal eight-coordinate state. No equality of the evolving rings or
closed scalar equation for `m` is inferred. Form and mismatch coordinates
remain necessary to determine its evolution.

### Regularity protects the lift; storage protects a separating surface

The source's two nonport resultants include

\[
z_4=2\cos(a/2)\exp(i(a/2-b)),\qquad z_3=2\cos b.
\]

They vanish at `a=+/-pi` and `b=+/-pi/2`, respectively. The receiver obeys
the same identities. A continuous regular solution starting inside the stated
lift cannot leave it without hitting a zero resultant. Changing phase
representatives does not evade this boundary of the continuous lift.

Since `W(2*pi/3)=W(-2*pi/3)=7`, every point on `a+A=+/-4*pi/3` costs at least
`7*beta`. This bound is attained at zero forms, `a=A=+/-2*pi/3`,
`b=B=+/-pi/3`, whose full nodal resultants are regular. Hence this is a sharp
transition barrier, rather than a numerical-domain threshold.

Within this lift, strict acute winding `+1` on a ring requires `a>3*pi/4`
(and analogously `A>3*pi/4`). Indeed, strict acuteness forces the internal
gap `a-b` into `(-pi/2,pi/2)`; the positive winding then comes from wrapping
the port gap `-2*a`. Both target rings consequently have `a+A>3*pi/2`.

It follows that any admitted reflected state with

\[
E<7\beta,\qquad a+A<4\pi/3
\]

cannot subsequently reach the two acute winding-`+1` rings under the same
regular unforced continuation. More symmetrically, `E<7*beta` and
`|m|<2*pi/3` trap the collective mean between both separating surfaces,
excluding both same-sign acute pairs. The sharp target minimum `2*V_*`
remains unchanged: a target can cost less than the barrier required to reach
its component. A geometric route inside the *initial* sublevel does not imply
a route after dissipative evolution has lowered that sublevel.

### Consequence for retained evidence and limits

The saved endpoint at `51/64` already has
`E` in `(6.999362696581, 6.999362696649)` and `a+A` in
`(2.359699409647, 2.359699409659)`, strictly below `4*pi/3`.
Thus the geometric barrier excludes this preparation's two-acute-`+1` target
earlier than the `2*V_*` budget does. This is the first saved endpoint passing
this sufficient test, not the exact first loss of reachability. The original
positive-budget horizon-one verdict remains correct for its weaker criterion.

The shared
[`certify_relational_reflected_barrier`](../../src/tnfr/physics/relational_reflected_transit.py)
computes storage, exact separator margins and regular-lift admission from an
interval state. It never evolves a trajectory or accepts a fitted storage
value. The [independent static controls](../../tests/test_relational_reflected_barrier.py)
check the algebra, sharp boundary, already formed target and invalid-domain
counterexamples. The [retained audit](../../tests/physics/test_relational_seeded_response.py)
applies the theorem separately to saved endpoints without changing records.

This barrier is not global convergence or regularity. For example, zero forms
with `a=A=pi/6`, `b=pi/2`, `B=pi/12` give
`V=8-4*cos(pi/12)<7` but `z_3=0`; regular interior states approach this boundary.
Thus a central low-storage component need not be compactly contained in the
regular domain. Equilibrium classification and future boundary exclusion
remain separate obligations. Neither this theorem nor the finite response
identifies physical matter or excludes all possible pattern formation.

## 22. Complete reflected regular equilibria and full-network stability

<a id="reflected-regular-equilibria"></a>

Keep section 18's two-C5 support, matching bridges, unit capacities and
inherited regular lift. Equilibrium classification requires `e>=0,w,beta>0`;
the dissipative stability statements below additionally require `e>0`.
The classification is complete within this reflection subspace, not among
all possible states of an arbitrary graph.

### Exact stationary reduction

Vanishing phase rates and positive inverse metrics give
`q_L=v_L=q_R=v_R=0`. Thus `r=p/2`, `R=P/2`, `P=7*p/2` and `p=7*P/2`, so all
four form coordinates vanish. In the inherited lift the nonport argument is
exactly `a/2-b`, since `z_4=2*cos(a/2)*exp(i*(a/2-b))`. Vanishing form rates
therefore imply `b=a/2` and `B=A/2`. The remaining conditions are

\[
f(a)=\sin(A-a),\qquad f(A)=-f(a),\qquad
f(t)=\sin(2t)+\sin(t/2),
\]

with strictly positive port real resultants

\[
C_L=\cos(2a)+\cos(a/2)+\cos(A-a),\quad
C_R=\cos(2A)+\cos(A/2)+\cos(A-a).
\]

At a stationary point the imaginary parts vanish, so a negative real
resultant is outside the regular law, even if the sine equations alone hold.
This two-angle stationary reduction is not a transient two-coordinate law.

### Exhaustion of the stationary branches

Put `m=(a+A)/2`, `d=(a-A)/2`, `M=cos(m)`, `D=cos(d)`,
`u=cos(m/2)>0`, `v=cos(d/2)>0`. The inherited lift gives `|m|+|d|<pi`.

If `d=0`, the equation factors, with `h=cos(a/2)>0`, as

\[
\sin(a/2)(2h-1)(4h^2+2h-1)=0.
\]

It gives the five diagonal states in the table below. If `m=0` and `d!=0`,
stationarity instead gives `16*h^3-8*h+1=0`, with `h=cos(d/2)>0`.
The port resultant is `2-8*h^2`, so regularity requires `h<1/2`.
There is exactly one root there: the cubic decreases to its sole minimum in
that interval and stays negative thereafter up to `1/2`. Its signs at
`1/8` and `1/7` isolate the root `k` between those rationals.

There are no additional asymmetric regular states. For `m*d!=0`, the two
stationary relations reduce to

\[
4uM(2D^2-1)+v=0,\qquad u=-8M^2vD.
\]

They imply `D<0`, `|d|>pi/2`, `|m|<pi/2`, and hence `M>0`. Eliminating `u,v`
gives

\[
32M^3D(2D^2-1)=1,\qquad
M=\frac{2D^2-1}{1+2D},\qquad -1/\sqrt2<D<-1/2.
\]

On this interval both `M(D)` and `D*(2*D^2-1)` are positive and strictly
increasing. At `D=-5/8`, `M=7/8` and their product equation's left side is
`12005/4096>1`. Thus a stationary root has `D<-5/8` and `M<7/8`.
The exact port product under those same relations is

\[
C_L C_R=\frac{2M^2+D-1}{4M^2(M+1)}<0,
\]

because its numerator is below `49/32-5/8-1=-3/32`. One port is therefore
negative real and the candidate is inadmissible. Squared equations have not
been used to admit a state without its original signs and resultants.

### Seven equilibria, five recoverable shapes

Let `d_*=2*acos(k)` for the unique cubic root just defined. Forms are zero,
`b=a/2`, `B=A/2` in every row. Common form/phase offsets in the full graph
are understood separately from the fixed reflected representative.

| Family | `(a,A)` | Ordinary winding | Storage divided by beta | Full quotient modes: stable / unstable |
| --- | --- | --- | --- | --- |
| Consensus | `(0,0)` | `(0,0)` | `0` | `18 / 0` |
| Aligned twists | `(+/-4*pi/5, +/-4*pi/5)` | `(+1,+1)` or `(-1,-1)` | `10*(1-cos(2*pi/5))` | `18 / 0` |
| Aligned saddles | `(+/-2*pi/3, +/-2*pi/3)` | `(+1,+1)` or `(-1,-1)` | `7` | `17 / 1` |
| Opposite twists | `(d_*,-d_*)` and its negative | `(+1,-1)` or `(-1,+1)` | `8+16*k^2-6*k` | `18 / 0` |

The opposite twists are acute in their **wrapped edge differences**. Their
internal path cosine is `k>0`; port and bridge cosines are
`8*k^4-8*k^2+1>0` using `1/8<k<1/7`. A large continuous lift `a` does not mean
a nonacute edge. Their storage is nevertheless

\[
E/\beta=7+16(k-3/16)^2+7/16\ge119/16>7.
\]

For consensus and both twist families every cosine edge weight is positive
on connected support. The phase Hessian consequently has inertia `(9,0,1)`.
The [full-network stiffness theorem](RELATIONAL_RECOVERY_AND_INTERACTION.md#regular-equilibrium-stiffness)
gives local exponential recovery in all 18 quotient coordinates; this includes
perturbations outside reflection. Two common-offset modes are neutral in the
full twenty-coordinate state.

At either diagonal saddle, each ring has weight `-1/2` on its port edge and
`+1/2` on its other four edges; both bridges have weight one. Ring exchange
splits the Hessian into `K_r` and `K_r+2*diag(1,1,0,0,0)`.
The first block has exactly one negative direction: it is a positive path
Laplacian minus a rank-one edge, and `(1,-1,-1/2,0,1/2)` has quadratic value
`-3/2`. The second block is positive definite, since its quadratic form is
the positive half-weight path sum plus `(v_0+v_1)^2+(v_0-v_1)^2/2`.
Thus the full phase inertia is `(8,1,1)` and the joint quotient has exactly
one growing real mode. Numerical eigenvalues do not supply this proof.

### What the classification does and does not settle

In `E<7*beta` the only regular equilibria are consensus and the aligned
twists. In the central component `|m|<2*pi/3`, only consensus remains. Thus
the retained failed formation trajectory has no stationary reflected
mixed-winding `(1,0)` destination. This is not yet its convergence verdict.

If `e>0` and its continuation stays uniformly separated from all excluded
resultant rays, the existing storage law and reflection bounds give a
precompact invariant set. On its invariant zero-loss subset all forms must
vanish and remain zero, requiring `g=0`. The classification then leaves only
consensus, and the existing LaSalle argument gives convergence to it.
The uniform regularity premise is not supplied by the energy sublevel: zero
forms with `a=A=B=0` and `b` approaching `pi/2` from below are regular with
`E=4*beta*(1-cos(b))<4*beta`, yet approach a zero central resultant. No new
global convergence or boundary-crossing rule is inferred.

The shared
[`certify_relational_reflected_equilibrium`](../../src/tnfr/physics/relational_reflected_equilibria.py)
encloses these ideal families and reports exact-definition provenance,
stiffness intervals and full-network stability for `e>0`. The opposite branch
uses a bounded rational cubic isolation and the shared certified cosine to
enclose its angle. It accepts named analytic states, not graphs recognized
as equilibria by a tolerance. [Independent controls](../../tests/physics/test_relational_reflected_equilibria.py)
check the full Hessian and the existing native tangent without evolving a
trajectory. Formation, autonomous support and physical identification remain
separate from the existence and local recovery of these supplied geometries.

## 23. Finite-time zero-resultant access in the reflected law

<a id="reflected-boundary-exit"></a>

Section 22's missing uniform-regularity premise cannot be deduced from the
central component and `E<7*beta`. This section proves an exact counterexample
by local existence; it does not compute or re-evaluate a trajectory.
Keep the same two-C5 support, unit held capacities, `e>=0,w,beta>0` and
unforced regular law. Choose `rho>0` and define the boundary state

\[
y_*=(p,r,P,R,a,b,A,B)=(0,\rho,0,0,0,\pi/2,0,0).
\]

In the established row order `(0,4,3,5,9,8)`, its resultants are
`(2+i,-2i,0,3,2,2)`. Node 3 alone is singular. Put
`h=atan(1/2)/pi>0` and `c=w*rho/beta>0`.
The four noncentral coefficients actually consumed by the eight reduced
equations remain smooth near this point. Their algebra gives the auxiliary
limiting vector

\[
\widetilde F(y_*)=
(e\rho/3+wh,\ -e\rho-w/2,\ 0,\ 0,\ -ch,\ c/2,\ 0,\ 0).
\]

This auxiliary field is a proof construction, not an extension of the
admitted full engine field. In particular no value is assigned to the
undefined central metric. The central form and phase rows are identically
zero along every regular reflected state, which is why they are absent from
the eight consumed rows.

### Transverse arrival from an admitted open time interval

Local existence for the auxiliary smooth field gives a two-sided solution
`gamma(t)` with `gamma(0)=y_*`. Its first-order phase terms are

\[
a(t)=-ch\,t+O(t^2),\qquad
b(t)=\pi/2+ct/2+O(t^2),\qquad A(t),B(t)=O(t^2).
\]

For all sufficiently small negative `t`, both `b<pi/2` and
`b-a<pi/2`. The remaining gaps are near zero. Every edge is therefore
strictly acute, and all six full resultants are regular. On that time
interval the auxiliary equations equal the native complete law exactly.
Selecting any time in this interval as an initial time gives a valid
trajectory that reaches the zero resultant in finite future time.
No fixed initial sample or quantitative exit time is asserted.

The derived resultant kinematics give, at the endpoint,

\[
\dot z=\bigl(-c(h+1/2)+3ich,\ -c(h+1),\ -c,\ -ich,\ 0,\ 0\bigr).
\]

In particular `z_3'= -c<0`, a transverse arrival at zero along the real
axis. The limiting storage and loss are

\[
E_*=2\rho^2+4\beta,\qquad \mathcal L_*=14e\rho^2/3.
\]

For `rho^2<3*beta/2`, a sufficiently short preceding segment retains
`E<7*beta` and `|(a+A)/2|<2*pi/3`. At the reference coefficients
`e=w=1/2, beta=rho=1`, the limiting storage is exactly six and
`z_3'=-1/2`. This disproves forward regularity based on that component and
energy alone, even with acute initial edges and positive dissipation.
It leaves the frozen uniform-receiver initial-value problem undecided.

### The full-state singularity is not removable

Approach the same endpoint with `a=A=B=0`, `b<pi/2` tending to `pi/2`,
the same other forms, and central form `x_3=k*cos(b)`. Then

\[
q_3=2k\cos b,\quad z_3=2\cos b>0,\quad H_3=2\pi\cos b,\qquad
\dot\theta_3=\frac{wk}{\beta\pi}.
\]

Every fixed real `k` gives the same limiting full state, but a different
limiting phase rate. The other rows have identical limits. These approaches
already lie in the positive-resultant chamber; no branch wrapping is needed
to produce the ambiguity. Taking `x_3=sqrt(cos b)` even makes that rate
unbounded at the same limit. Hence the full vector field has no continuous
extension at this point. Cancelling the symmetric zero row would select an
extra boundary convention, not derive a globally defined law.

This is a restriction on this constitutive completion, not on the nodal
identity or every possible TNFR law. A regular chart change cannot remove
the full-state obstruction. Any revised pressure, exchange law, state or
generalized solution convention needs its own F1-F4 admission and must retain
valid interior results with their original hypotheses.

### Static engine evidence and preserved execution boundary

[`certify_relational_reflected_boundary_exit`](../../src/tnfr/physics/relational_reflected_boundary.py)
encloses the named ideal boundary, limiting consumed rates and derivative
signs. It reuses the reflected rate algebra without relaxing the full
evaluator's six-resultant admission. Its statement is local-flow existence,
not a certified response from supplied initial data or permission to step
across zero. [Controls](../../tests/physics/test_relational_reflected_boundary.py)
check those formulas and independent full-state limiting paths.
The [general domain theorem](RELATIONAL_DOMAIN_AND_CAPTURE.md#regular-domain-continuation-and-boundary-access)
separately protects sufficiently small total storage and distinguishes
zero-resultant collisions from nonzero negative-real branch collisions.

## Smooth-sine proof owner

Sections 24–32 are maintained in [Smooth-sine pattern dynamics](SINE_PATTERN_DYNAMICS.md).
The links below preserve the original section and subsection targets.

## 24. Equilibrium geometry of components joined by bridges

<a id="sine-bridge-tree-composition"></a>
<a id="complete-law-and-decomposition-premises"></a>
<a id="a-bridge-cannot-carry-a-stationary-sine-current"></a>
<a id="relative-hessian-inertia-adds-across-the-bridge-tree"></a>
<a id="full-nodal-stability-follows-under-positive-loss"></a>
<a id="trees-and-assemblies-of-c5-components"></a>
<a id="shared-geometry-reader-and-evidence-boundary"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition) are maintained in the smooth-sine owner.

## 25. Cycle periods constrain joint composition

<a id="sine-cycle-sector-compatibility"></a>
<a id="existing-circulation-criterion-on-the-complete-nodal-law"></a>
<a id="a-general-exact-obstruction-from-combined-cycle-periods"></a>
<a id="independent-components-a-compatible-and-an-excluded-composition"></a>
<a id="shared-certificate-and-retained-formation-boundary"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility) are maintained in the smooth-sine owner.

## 26. Storage below every sector face forces capture

<a id="sine-target-free-sector-capture"></a>
<a id="the-entire-acute-cell-without-an-equilibrium-target"></a>
<a id="a-computable-lower-bound-covering-every-face"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture) are maintained in the smooth-sine owner.

## 27. Prepared form can acquire a nonzero phase sector

<a id="sine-prepared-sector-entry"></a>
<a id="a-global-form-to-phase-estimate-with-a-finite-declared-horizon"></a>
<a id="convert-the-estimate-into-a-full-state-capture-certificate"></a>
<a id="one-exact-prepared-control-and-its-zero-form-comparison"></a>
<a id="quantitative-robustness-for-independent-form-and-phase-uncertainty"></a>
<a id="one-fixed-preparation-neighborhood"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry) are maintained in the smooth-sine owner.

## 28. Exact form, phase and memory representations

<a id="sine-form-phase-memory-equivalence"></a>
<a id="the-sum-coordinate-retains-the-full-joint-state"></a>
<a id="exact-elimination-gives-second-order-phase-and-retained-memory"></a>
<a id="the-nonlinear-storage-representation-contains-the-existing-resonance-pencil"></a>
<a id="a-phase-flat-preparation-distinguishes-the-full-law-from-phase-descent"></a>

[Definitions, proof and controls](SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence) are maintained in the smooth-sine owner.

## 29. A controlled fast-form and slow-phase comparison

<a id="sine-controlled-slow-phase"></a>
<a id="complete-state-reference-initialization-and-clocks"></a>
<a id="explicit-composite-phase-and-form-bounds"></a>
<a id="uniform-finite-time-meaning-and-the-retained-storage-budget"></a>
<a id="means-uncertain-sources-and-phase-potential"></a>
<a id="shared-certificate-and-bounded-numerical-controls"></a>

[Definitions, proof and controls](SINE_FORM_PHASE_REDUCTION.md#sine-controlled-slow-phase) are maintained in the smooth-sine owner.

## 30. Full-state capture from controlled phase geometry

<a id="sine-slow-phase-capture-handoff"></a>
<a id="a-proved-reference-neighborhood-from-the-preparation"></a>
<a id="the-actual-endpoint-and-its-complete-storage"></a>
<a id="handoff-to-the-whole-sector-theorem"></a>
<a id="fixed-analytic-controls-and-scope"></a>

[Definitions, proof and controls](SINE_FORM_PHASE_REDUCTION.md#sine-slow-phase-capture-handoff) are maintained in the smooth-sine owner.

## 31. A fixed original preparation budget forces consensus at small ratio

<a id="sine-budget-consensus"></a>
<a id="the-preparation-class-and-sufficient-condition"></a>
<a id="an-exact-auxiliary-lyapunov-function"></a>
<a id="first-exit-exclusion-and-full-state-convergence"></a>
<a id="budget-dependence-shared-certificate-and-fixed-controls"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-budget-consensus) are maintained in the smooth-sine owner.

## 32. Equal original budgets do not determine the acquired geometry

<a id="sine-equal-budget-preparation"></a>
<a id="a-capacity-compatible-automorphism-restricts-cycle-periods"></a>
<a id="the-fixed-c5-preparations-have-exactly-equal-storage"></a>
<a id="what-the-comparison-establishes"></a>

[Definitions, proof and controls](SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation) are maintained in the smooth-sine owner.
