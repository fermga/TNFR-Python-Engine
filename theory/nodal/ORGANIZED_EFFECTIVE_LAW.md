# Collective state and a matrix-response comparison

<a id="organized-effective-law-contract"></a>

## Status and responsibility

This owner specifies one complete collective comparison for the foundational
question raised by the [physical feasibility review](../research/ORGANIZED_ELECTRICAL_FEASIBILITY.md).
The law below is a supplied constitutive choice. Its balance and coordinate
identities are conditional algebra; a finite identity/prediction certificate,
formation under this law and physical realization remain unestablished.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone selects work and acceptance conditions.

The question is how retained state, response and storage must change together
when an organized object has an additional structural coupling. It does not
assume a new primitive capacity or a uniquely selected fundamental law. The
[NFR ontology](../EMERGENT_ONTOLOGY.md#nfr-participation-in-form) requires any
claim about constituents to retain their influence in the same configuration.

## 1. Complete comparison and units

Use the fixed, unit-weight, undirected C5 Laplacian L and five signed forms U
with five primitive circular phases Phi. These are the collective reference
coordinates in the [quarter-law dictionary](ORGANIZED_ELECTRICAL_PROTOCOL.md).
They do not include the 45-node source's internal deviations. Retain the
same normalized structural clock tau=(1023/1024)t and units,
gamma=1/(1023 pi), and held unit scalar capacities. Primes below denote
derivatives in tau. Laboratory calibration is a separate obligation.
Write m for the conserved mean form; r below always denotes the response
parameter, not the mean denoted r in the earlier electrical protocol.

Let r be a supplied finite constant with r>=0, and define

\[
 W_r=I+rL^2,\qquad M_r=W_r^{-1},\qquad
 S_i(\Phi)=\sum_{j\sim i}\sin(\Phi_j-\Phi_i).
 \tag{1}
\]

The dimensionless parameter r changes the response in this declared chart;
it is not fixed by the nodal row. The complete driven equations are

\[
 W_r U'=(-LU+\gamma S(\Phi))/4+f(\tau),\qquad
 \Phi'=\gamma LU/4.
 \tag{2}
\]

Input f is an independently specified, neutral, locally integrable
five-vector; no phase feedback, support event or input selector is implicit.
There are no jumps. Thus the solved form row has input M_r f, not f.
The unforced pressure is the independently specified map
`p=M_r*(-LU+gamma*S)/4`, with `U'=p` at unit capacity. Circular differences
enter S without choosing a principal-phase average. Means of phase lifts
are used only after declaring those lifts.

W_r is symmetric positive definite, commutes with L and satisfies W_r 1=1.
The full equations preserve the form mean under neutral input; on any
continuous phase lift they also preserve its common mean. One may fix the
neutral form leaf and quotient a common phase origin, while retaining four
relative forms and four relative phases. These are symmetry reductions,
not removal of response-relevant internal state. An admitted initial set,
its lifted chart where needed and every proposed identity must still be
specified before a finite claim.

At r=0, (2) recovers the nominal collective quarter law. For r>0 it changes
only the form row in these coordinates. This differs from the
[phase-dependent diagonal reciprocal mobility](SINE_PAIR_MOBILITY.md), which
changes both rows and keeps another storage. Its proofs do not transfer.

The [electrical audit](../research/ORGANIZED_ELECTRICAL_FEASIBILITY.md#organized-electrical-physical-admission)
motivates (1) through an additional positive ring capacitance, with
r=c_J/c. That motivation does not force this law on arbitrary NFRs or
provide a fine-node electrical realization. In particular, replacing L
by L45 would be a further constitutive choice requiring its own admission.

## 2. Storage and supplied work

Let V(Phi)=sum_edges[1-cos(Phi_j-Phi_i)], so grad V=-S. The comparison's
own storage is

\[
 H_r(U,\Phi)=\tfrac12 U^TLW_rU+V(\Phi).
 \tag{3}
\]

Since L and W_r commute, LW_r is symmetric and positive semidefinite.
Differentiating along (2) gives, almost everywhere for integrable f,

\[
 H_r'=(LU)^T W_rU'-S^T\Phi'
       =-\|LU\|^2/4+(LU)^Tf.
 \tag{4}
\]

The cross terms cancel. This is a storage/work balance, not a formation,
stability, recurrence or observation theorem. The old H0 is not the storage
in (4); the difference is `r*U^T*L^3*U/2`. Consequently equal initial states
need not have equal storage across r. A comparison must declare which
preparation quantities are matched instead of assuming both equalities.

There is an explicit obstruction to reusing the old decay claim. Set f=0,
q=LU and H0=H_r at r=0. Along the positive-r law,

\[
 H0'=-q^TM_rq/4+\gamma q^T(M_r-I)S/4.
\]

Choose Phi=(0,pi/2,0,0,0), so S=(1,-2,1,0,0), and
U=-(gamma*r/2)LS. Then q=(gamma/2)(I-W_r)S and, for every r>0,

\[
 H0'=\frac{\gamma^2}{16}S^T(M_r-I)W_r(M_r-I)S>0,
 \qquad H_r'=-\|q\|^2/4<0.
\]

The strict inequalities follow because S is a nonzero neutral vector and
L is positive definite on that subspace. This example uses the full
circular-phase domain of (2); it is not an admitted acute winding source.
It disproves a global transfer of the old storage monotonicity, without
claiming loss of an acquired identity.

Because W_r is invertible, unforced equilibria of (2) agree with those of
the r=0 complete law. Their tangent rates and finite response need not agree.
Matching equilibria does not transfer attraction, an initial-state capture
bound or an observation certificate. A finite positive-r study must declare
its parameter range and horizon before requesting uniform bounds.

## 3. Coordinate, locality and memory boundaries

The invertible charge-like chart Z=W_r U changes both equations:

\[
 Z'=(-LM_rZ+\gamma S(\Phi))/4+f,\qquad
 \Phi'=\gamma LM_rZ/4,
 \tag{5}
\]

with storage `Z^T*L*M_r*Z/2+V(Phi)` and work pairing `(LM_r Z)^T f`.
Moving W_r out of the form row while retaining `Phi'=gamma*L*Z/4` would
therefore change the model. A regular chart change is distinct from
discarding state and from choosing a different pressure law.

Although W_r is positive definite, M_r is not generally entrywise
nonnegative. For example, at r=1/100 its first row is
`(2181,81,-19,-19,81)/2305`. It is not a convex averaging map or a vector
of nonnegative nodal capacities. Its coupling need not have the one-edge
locality of the original sine-pressure row; locality must be justified
for the actual map, including the phase and input channels.

This fixed matrix is instantaneous. It introduces no temporal memory by
itself. Derived memory in the cited reductions comes from eliminating
dynamical state under a specified observation, retaining hidden
initialization and forcing, as in
the [memory owner](../DERIVED_EPI_MEMORY.md). The
[inherited-metric result](INHERITED_FORM_DYNAMICS.md#132-inherited-metric-restoring-pressure-and-coordinate-dependence)
already distinguishes coordinate response factors from primitive capacity.
Neither construction justifies replacing nu_f by M_r in the native runtime.

Holding r constant is part of (2). On the neutral form leaf, a time-dependent
charge law `q proportional to W_r U` differentiates to an expression
proportional to `W_r U'+r' L^2 U`; off that leaf the charge map uses
`W_r(U-m*1)`. Substituting W_r(tau) into (2) alone would be a different model.
Parameter changes or
support events also need their storage/work and occurrence laws.

## 4. Identity, scale and implementation boundary

The state in (2) is sufficient for this declared ten-coordinate ODE.
It is not a proof that ten coordinates predict the full acquired source.
The [same-slot source remainder](ORGANIZED_ELECTRICAL_PROTOCOL.md#fixed-support-electrical-protocol),
its hidden transport, constituent identities and acquisition use another
complete law. They remain valid there. Neither r=0 recovery nor a shared
equilibrium establishes those claims at positive r.

A claim about a parent NFR must define whether it concerns this prepared
collective reference or a finer organization with retained constituent state.
The latter needs a separately declared fine law, a state/clock/input map,
exact closure or controlled error, and own-law identity bounds. Static
re-encoding or formal associativity of an exact elimination is insufficient
as a new capability; finite nonlinear prediction must retain the consumed
boundary information. Events additionally require the existing
[operator descent criterion](../STRUCTURAL_OPERATORS.md#collective-operator-descent).

This comparison installs no execution law, SDK method or operator.
The [independent algebra controls](../../tests/physics/test_organized_effective_law.py)
check (1)-(5), the r=0 boundary and deliberate wrong-chart/input controls.
They evaluate no reserved response. Exact matrix calculations reuse
[shared linear algebra](../../src/tnfr/mathematics/_exact_linear_algebra.py).
The [coordinate-memory owner](../../src/tnfr/mathematics/linear_observation.py)
can split an admitted fixed linear/tangent generator or rational spatial
blocks. It does not close the nonlinear sine law. Such reuse must retain
nonlinear residuals, hidden initialization and forcing where consumed;
rationalizing gamma would specify a different coefficient, not certify
the exact comparison above. A nonlinear reduction needs its own proof.
Generic runtime integration is justified only by a proved consumer contract.

<a id="prepared-matrix-response-declaration"></a>
## 5. Declared finite prepared-reference question

The analytical comparison fixes the following domain before evaluating any
response. It concerns (2) alone, without a fine-node extension, source replay,
physical sensor or numerical trajectory. The two models match their complete
initial coordinates and prescribed input; their initial storages need not match.

| Primitive | Declaration |
| --- | --- |
| Response | Fixed `r in [1/100,1/10]`; separate reference `r=0` |
| Prepared identity | Either C5 winding `w=+1` or `w=-1` |
| Phase chart | `Phi_i=2*pi*w*i/5+theta_i`, with a continuous perturbation lift and `sum(theta)=0`; the closing oriented edge retains its `2*pi*w` offset |
| Full preparation | `sum(U)=0`, `||U(0)||_2+||theta(0)||_2 <= delta=2^-26` |
| Input | `f=a*v` for `0<=tau<=T`, zero afterwards; `a=1/16`, `v=(1,-1,0,0,0)` |
| Clock and horizon | The existing structural tau; `T=1/1024` |
| Observation | `y=v^T*U(T)` with separate additive error bounded by `eta=2^-26` |
| Identity criterion | Every oriented phase gap stays within `rho=1/10` of `2*pi*w/5` during the pulse and its unforced continuation |
| Finite prediction criterion | Whole-preparation observed record intervals for the positive-r family and r=0 are disjoint, with separation strictly greater than `2^-21` |
| Calculation | Exact rational inequalities and certified transcendental bounds; no integration, parameter scan, precision search or response fitting |

The prediction must retain both evolution rows, the full preparation bound,
the input channel `M_r*f`, and the law's own storage and supplied work. A
separate state-sufficiency control tests whether identical forms, winding,
conserved means and storage can still conceal different form derivatives.
Neither success condition would derive acquisition of this preparation,
constitutive selection, a laboratory clock or a physical measurement map.
