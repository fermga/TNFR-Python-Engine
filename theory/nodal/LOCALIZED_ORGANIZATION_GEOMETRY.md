# Geometry and the formation boundary of the prepared mobile pair

**Scope:** a source-specific corollary of the existing
[sector-capture theorem](SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture)
under the unchanged [complete cycle law](LOCALIZED_IDENTITY_TRANSPORT.md#localized-transport-law).
It relates the supplied geometry and preparation to future stationary
organization. No trajectory, new reserved response, parameter search, input
or support change is evaluated. The earlier
[nominal interaction result](LOCALIZED_MOBILE_RESPONSE.md#mobile-response-result)
and its unresolved original-family interaction bound remain unchanged.
The [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#mobile-organization-feasibility-admission)
retains formation of a common, finite-lived organization from the incoming pair.

<a id="geometry-and-formation"></a>
## 1. Which geometry enters this model?

Three distinct geometries must be retained:

| Geometry | Role in the present model | Boundary |
| --- | --- | --- |
| Support and coefficients | The supplied unit cycle determines incidence D, Laplacian L and the actual exchange paths; held capacities enter the complete law. | A drawn embedding or changed edge length alone does not change these rate rows. Edge length defines the localization observation, separately from conductance. |
| Relative form and primitive phase | Neighbor form differences drive phase motion; sine phase differences drive form. Phase differences determine the storage curvature and its critical configurations. | The Fourier carrier of `z=x+i*theta` is not primitive circular-phase winding. |
| Organization and boundary | A spatial window, profile, carrier band or partition specifies an observation of the retained full state. Its boundary remains coupled to the complement. | Overlap, compatible grouping and concentration do not establish actual entry into a new identity or its maintenance. |

These distinctions reuse the [relation foundation](RELATION_FOUNDATIONS.md),
[collective phase-offset conditions](SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition)
and [regional work accounting](FORCED_REGIONAL_RESPONSE.md#7-a-region-and-its-environment-on-the-same-nodal-support).
They do not introduce an extra geometric force or select a new law.

The moving centers in the prepared-transport proof are comparison references,
not imposed trajectories in (1). Actual form and phase evolve under both
complete rows and can alter the observed profiles and their paths within the
proved error bounds. A common organization could involve deflection, changed
relative motion or internal rearrangement of the incoming patterns. It need
not be a third separate object, and its definition must not require that the
original patterns first separate. Any inferred position or velocity remains
an observation of the field on fixed sites, with its own availability and
error; the elementary support vertices do not move in this model.

A proper contiguous window on a cycle induces a path. It has no internal
graph cycle on which to transfer the earlier C9 constituent-winding identity.
Closing that path by ignoring its environmental endpoints would change the
support. A localized identity can instead retain relative form/phase structure
and boundary exchange, but it needs its own prospective formation criterion.

<a id="mobile-zero-sector-basin"></a>
## 2. An all-time basin for the original preparation family

Let D be the oriented incidence matrix of a simple unit cycle `C_N`, `N>=3`,
and `L=D^T D`, `A=L/2`. Retain signed real form, real continuous phase lifts,
held unit capacities, beta one, positive e,w, and no forcing or events.
In the declared clock `tau=w*t/pi`, with `sigma=pi*e/w>0`, the complete rows are

\[
 x'=-\sigma Ax-\tfrac12D^T\sin(D\theta),\qquad \theta'=Ax. \tag{1}
\]

Both ordinary means are conserved. Each source member retains its own means;
they need not equal the nominal zero means. A constant form or phase reference
can be subtracted without changing (1). The following source premise concerns
one consistent real lift relative to such a constant reference:

\[
 \|(x(0),\theta(0))\|_2\le M,\qquad H_0=2M^2<1. \tag{2}
\]

This is a sufficient source condition, not a phase reconstruction from form,
a physical preparation certificate or an admission from a cached verdict.

### Storage and geometric trapping

The existing structural storage and its actual clock-transformed balance are

\[
 H(x,\theta)=\tfrac12x^TLx+\sum_e[1-\cos(D\theta)_e],
 \qquad H'=-\tfrac\sigma2\|Lx\|_2^2\le0. \tag{3}
\]

Here H is the original storage, not the packet localization diagnostic.
For a cycle, `||L||<=4`, and `1-cos u<=u^2/2`, so

\[
 H(0)\le2\|x(0)\|^2+2\|\theta(0)\|^2\le H_0. \tag{4}
\]

Initially every lifted edge difference has absolute value at most
`2M<sqrt(2)<pi/2`. A first exit from this zero-winding acute component would
have an edge at `+/-pi/2`, contributing one to H. All other storage terms
are nonnegative, contradicting `H<=H_0<1`. Therefore every future edge remains
in the same lifted acute component, and in fact

\[
 \cos(D\theta)_e\ge1-H_0>0,\qquad
 (1-H_0)L\preceq\nabla^2V(\theta)\preceq L,\quad
 V=\sum_e[1-\cos(D\theta)_e]. \tag{5}
\]

This bound concerns phase-storage curvature, not the tetrad's diagnostic
curvature. It is valid for all nonnegative structural time, without extending
the finite moving-template error tube. The unwrapped gaps still telescope
to zero around the cycle, so the circular winding remains zero. Changing
that winding would require an edge to cross a circular cut, which is also
excluded by the stronger acute trap.

### The only limiting state on each mean leaf

The bounded gaps and fixed phase mean bound every lifted phase coordinate.
The form term in H bounds centered form because L is positive on its
mean-zero space. The invariant sublevel is thus compact on each mean leaf.
The full vector field is globally smooth with bounded sine currents and
linear form dependence, so forward continuation is available.

In the zero-dissipation set, `Lx=0`, hence form is uniform and `theta'=0`.
An invariant trajectory there must also have `D^T sin(Dtheta)=0`: otherwise
the form row leaves that set, since its ordinary mean derivative is zero.
Taking its inner product with theta gives

\[
 0=\sum_e (D\theta)_e\sin(D\theta)_e.
\]

Every summand is nonnegative and vanishes only at zero in the retained
acute chart. Thus every gap vanishes. The sector-capture/LaSalle argument
therefore yields

\[
 x(\tau)\longrightarrow\overline x(0)\mathbf1,\qquad
 \theta(\tau)\longrightarrow\overline\theta(0)\mathbf1. \tag{6}
\]

This is convergence to one state on each actual source member's conserved
mean leaf. It supplies neither a convergence rate nor a finite settling time.
The original finite packet identities can remain valid during transport
even though they eventually relax. Equation (6) excludes a nonuniform
stationary asymptote in this source family; it does not exclude finite-lived
additional structure before that limit.

### Application without a new response calculation

The two nominal Fejer preparations have orthogonal complex carrier bands
and joint norm `sqrt(2)*a`. Each primitive full-state error is at most epsilon,
and the joint source is their exact sum. Hence the entire original family
satisfies (2) with

\[
 M=(\sqrt2+1/500)a,\qquad a=2^{-12},\qquad
 H_0=2(\sqrt2+1/500)^2a^2<\frac3{12500000}=2.4\,10^{-7}. \tag{7}
\]

For example `sqrt(2)<70711/50000` proves the last strict rational bound.
Neither the computed endpoint defect nor the unresolved uniform interaction
verdict is a premise of (7). The conclusion therefore applies to all original
source errors, despite the distinct endpoint discrimination limitation.
The same estimate gives an all-time phase-Hessian lower factor greater than
`1-2.4e-7`. It does not give an all-time `||z||<2a` bound, preserve the two
carrier projections forever, or quantify the lifetime of another pattern.

## 3. Why the source's geometry cannot be omitted

Small storage alone does not imply consensus for arbitrary circular sources
on this same support. A separately supplied twist with constant principal
edge gap `2*pi/N`, uniform form and `N>4` is a critical configuration:
the incoming sine currents cancel and its Hessian is `cos(2*pi/N)*L`.
It is a strict local minimum on the relative-phase leaf and has the existing
positive-loss local recovery property. On `C_1024` its storage satisfies

\[
 H_{\rm twist}=N[1-\cos(2\pi/N)]\le\frac{2\pi^2}{N}<1. \tag{8}
\]

It has winding one and no all-small zero-component lift around the cycle.
It therefore does not satisfy the source condition (2). This counterexample
shows why support, storage and the phase component must be matched together.
It is an extended prepared pattern, not a localized object formed by the
current encounter. It supplies no route across the inaccessible phase boundary.

## 4. An exact carrier geometry, without a localized envelope closure

For N divisible by four, the repeated period-four family

\[
 \theta=(\varphi,\psi,-\varphi,-\psi,\ldots),\qquad
 x=(r,s,-r,-s,\ldots)
\]

is invariant under both complete rows. On this subspace A acts as identity;
direct neighbor addition gives

\[
 \varphi'=r,\quad \psi'=s,\qquad
 r'=-\sigma r-\sin\varphi\cos\psi,\quad
 s'=-\sigma s-\sin\psi\cos\varphi.
\]

The invertible relative coordinates `u=varphi+psi`, `v=varphi-psi` consequently satisfy

\[
 u''+\sigma u'+\sin u=0,\qquad
 v''+\sigma v'+\sin v=0. \tag{9}
\]

Their initial velocities retain r,s; this exact change of coordinates does
not discard form or install a pendulum as a new law. It exhibits nonlinear
internal dynamics selected by a compatible carrier geometry under the
actual positive loss. The family repeats throughout the support and supplies
no localized envelope, acquisition from outside that family or self-binding.
The finite-width prepared packets are not confined to this four-dimensional
family. Their other modes and boundary exchange cannot be dropped by using (9).

At the consensus tangent, the two modal exponents are
`lambda(k)*[-sigma/2 +/- sqrt(sigma^2/4-1)]`, with `lambda(k)=1-cos k`.
Thus their second k derivatives vanish at the current `+/-pi/2` carriers.
For the frozen underdamped coefficients their third derivatives do not vanish.
Equation (9) retains exact carrier nonlinearity, but neither it nor this
dispersion fact supplies a controlled modulation or localization theorem.

## 5. Formation of a common relational pattern

The geometric circumstances already matter in (1)--(8): the support supplies
paths, the prepared phase component determines accessible critical geometry,
and the storage/loss law restricts which regions of state space can be reached.
The unchanged small pair cannot acquire another stationary trapping geometry.
Its certified nominal nonadditivity can coexist with this obstruction.

The generative target is a **common organization of the incoming patterns**:
their relative arrangement and joint evolution acquire an identity absent
initially. It may move, deform or have bounded internal relative motion; neither
unchanged outgoing trajectories nor an extra separate object is required.
Retain constituent and environmental boundary information consumed by that
identity. If the original constituent identities cease to apply, record that
reorganization rather than retaining their labels as evidence of persistence.

An admissible finite-lived claim specifies:

- **Relational identity.** A positive-width class on the actual joint state,
  with a nonvanishing signal floor and specified relative geometry, primitive
  phase information and internal response. Any collective center, trajectory
  or phase registration needs an observation rule, ambiguity handling and
  error bounds; a band observation is not an independent dynamical constituent.
- **Formation.** The original source class lacks that common identity with a
  margin. Both complete rows must carry it into the declared class. Initial
  support connectivity, the presence of two packets or transient overlap
  does not establish this acquisition.
- **Joint retention.** The relation remains in a declared nontrivial domain
  for a specified interval beyond ordinary passage, with environmental work,
  dispersion and loss accounted for. Bounded distance alone is vacuous on a
  finite cycle; a common moving center alone can also follow two separating
  packets. The incoming patterns need not separate before this test begins.
- **Continued response.** The reached organization has a discriminating later
  response without resetting its acquired state. Compare the same-source
  linear passage and separate histories so that overlap, interference and
  independently continuing packets cannot alone pass the formation test.

A third Fourier band or the difference of three histories alone does not
satisfy those conditions. The
[parent-acquisition architecture](ORGANIZED_PARENT_ACQUISITION.md) illustrates
this proof structure; its supports, windings and coefficients do not transfer.

The existing endpoint bounds still restrict the unchanged source family.
At tau=256, each nonzero band observation has less than `1/25` of its
squared norm outside its radius-96 reference window; the two reference
centers have cycle distance 256. If a common cycle ball of radius r contains
more than `1/25` of **each** band's squared norm, it must intersect both
reference windows. The cycle-distance triangle inequality then gives
`256<=2(96+r)`, hence `r>=32`. This necessary geometric restriction neither
excludes wider or two-lobed composites nor establishes one above that radius.
It concerns the retained band observations, not an additive decomposition of
local total density. Radius 32 is not a newly selected formation threshold.

No such composite identity or binding mechanism is certified here.
Comparing nonlinear coupling with dispersion and loss requires the actual
carrier geometry: at the present `pi/2` carrier the tangent's quadratic
dispersion vanishes. A quadratic-dispersion envelope law cannot be imported
as a formation theorem. Changing preparation, support, loss or events would
define a separately admitted case, preserving the present evidence.

## 6. Shared arithmetic and controls

The private `_bound_cycle_consensus_basin` reader in the
[transport owner](../../src/tnfr/physics/_localized_identity_transport.py)
re-admits the complete positive-loss model, cycle size and independent initial
norm bound before computing (2)--(5). A failed strict bound yields unavailable
sufficiency; it is not a verdict about an evaluated state or finite formation.
Existing finite transport/source reports and their schemas remain separate.

The [geometry controls](../../tests/physics/test_localized_cycle_geometry.py)
check the actual storage derivative and clock, independent source bounds,
strict admission boundaries, the low-storage winding counterexample and
the period-four reduction from direct neighbor rows.
They do not execute a trajectory or replay the frozen response producer.
The proof and source premises, rather than a passing diagnostic score,
establish the conditional all-time claim.
