# Forced-source closure and phase clocks

Phase/capacity tangency, source constraints and the limits of synchronization as a clock.

Part of [Forced support balance and model boundaries](../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 21. Closing the relaxed phase-capacity source

### Exact model and common fixed fields

Write x=EPI, let L_W be the weighted EPI random-walk Laplacian and L_U
the unweighted unique-neighbor support Laplacian. The actual canonical
pressure decomposition is

`p = -e L_W x + w_phi g_phi - w_vf L_U nu - w_topo L_U k`,

where k is the support-degree vector. These two Laplacians generally differ.
This section studies fixed physical fields of the existing phase and capacity
updates, together with zero fresh pressure. It does not assume that the EPI
nodal equation uniquely determines those configured update policies.

**Telemetry versus dynamics.** Si is a derived diagnostic of capacity, phase
and pressure, not an additional nodal force or a fundamental evolution law.
Computing/storing Si does not itself reorganize the triad. The implementation
chooses to consume Si in a threshold-based capacity-adaptation policy. That
policy has a dynamical effect because it writes capacity; its use of a
diagnostic is an extra operational assumption, not a derivation from
`dEPI/dt=nu_f*DeltaNFR`. Every result below about that gate is conditional on
this implemented policy. It cannot exclude persistent differentiated forms
under other structurally justified TNFR dynamics. The configured phase and
averaging laws likewise remain explicit premises.

Assume finite undirected support, connected positive symmetric EPI conductance,
positive capacity, e>0 and w_topo=0, as in the retained channel configuration.
The phase support is connected as well. Use exact-real means and trigonometry,
no scalar clipping, no named operator/reset interventions and no external
changes to support or fields. Required diagnostics refer to these same fields.

The [coordination owner](../../src/tnfr/dynamics/coordination.py) has the ideal map

`theta_i^+ = theta_i + k_G wrap(m_G-theta_i) + k_L wrap(m_i-theta_i)`,

with global and neighbor phasor arguments m_G and m_i. Suppose all phases
admit a common real lift in an interval of width less than pi, gains are
nonnegative and at least one is positive. Require a fixed point of this
lifted update, excluding nonzero multiples of 2*pi hidden by normalization.
Every phasor mean lies between its participating minimum and maximum, with
a strict interior value unless those phases agree. At a maximum phase both
increments are nonpositive. If k_G>0, a nonconstant state gives a strictly
negative global increment. If k_G=0 and k_L>0, a fixed maximum forces every
neighbor to share that maximum; connectedness propagates equality. Therefore
the phase fixed fields in this chart are exactly consensus phases. In
particular the ideal local phase-pressure channel g_phi vanishes.

For an active [capacity update](../../src/tnfr/dynamics/adaptation.py),

`nu_i^+ = (1-mu) nu_i + mu mean_{j in N(i)} nu_j`, with `0<mu<=1`.

If every node is eligible, a fixed capacity satisfies L_U nu=0. The same
maximum principle makes nu constant. With phase consensus and w_topo=0,
the non-EPI source vanishes. Positive capacity then makes zero nodal rate
equivalent to p=0, and e L_W x=0 forces uniform EPI on connected positive
conductance. Conversely constant positive capacity, constant EPI and phase
consensus leave these physical fields fixed. This classifies common fixed
fields of the selected substeps; it does not classify cancellations between
substeps of an arbitrary composite runtime cycle.

### Conditional consequence of the Sense Index gate policy

Within that adaptation policy, the all-eligible premise can be weakened.
At zero fresh pressure and phase
consensus, the [Sense Index](../../src/tnfr/metrics/sense_index.py) reduces to

`Si_i = clamp01(alpha*nu_i/nu_max + beta + gamma)`.

Assume its normalization uses the current positive nu_max and that its
nonnegative weights satisfy
`s_max=clamp01(alpha+beta+gamma) >= si_hi`. Normalized exact weights give
s_max=1; represented weights need the displayed inequality, not an assumed
exact sum of one. Every capacity maximum meets the Si threshold and the
zero-pressure part of the stability gate. If unchanged fields are repeatedly
evaluated without external counter resets, its finite stable_count reaches
VF_ADAPT_TAU. At a boundary of a nonconstant maximum plateau, positive-mu
averaging would strictly decrease capacity. Such a plateau cannot persist.
Connectedness therefore forces uniform capacity even if some lower-capacity
nodes are initially ineligible.

The existing held-capacity profile `x=c*1-(w_vf/e)*nu`, when L_W=L_U,
remains a valid [conditional balance](../../src/tnfr/physics/capacity_localization.py).
It is not a self-consistent heterogeneous equilibrium of this fresh-diagnostic
relaxation policy. Freezing Si or disabling its refresh supplies a different
closure. The native runtime refreshes pressure and optional Si before later
substeps; that timing agrees at a common equilibrium but cannot be ignored
along a changing trajectory. Stable counters and histories may grow even
when physical fields are fixed.

### What can and cannot be inferred

| Channel or mechanism | Exact conclusion in this branch | Remaining sustaining condition |
| --- | --- | --- |
| Attractive phase coordination | Consensus at a lifted fixed point in the common semicircle | A phase pattern outside that chart, nonstationary motion or another admitted phase feedback needs its own proof |
| Implemented capacity policy consuming fresh Si | Under this gate and averaging law, a persistent zero-pressure equilibrium cannot hide heterogeneous maxima behind inactive gates | This conditional obstruction does not constrain a different capacity law derived from nodal structure |
| Pure EPI transport after source loss | Only uniform zero-pressure form | A maintained canonical source or a different notion of identity is required for differentiation |
| Derived partial-observation memory | Reexpresses existing full dynamics | A memory kernel does not create a new sustaining source |
| Topology pressure | Absent when w_topo=0 | Nonzero topology weight changes the retained configuration and still requires weighted compatibility |

This is not a convergence theorem. On P2, local-only phase gain k_L=1
swaps two unequal phases within the semicircle; capacity gain mu=1 likewise
permits a swap. With k_L=4 and phases (0,pi/2), increments (+2*pi,-2*pi)
also show why wrapped stationarity is weaker than a lifted fixed point.
Binary64 stalled gaps, clipping and represented phasor roundoff remain outside
the exact theorem. Positive-conductance connectivity cannot be replaced by
mere support connectivity with zero-weight links.

Uniform EPI is not the same as an unstructured complete triad. For local-only
phase coordination on C_n, n>=5, the regular winding theta_j=2*pi*j/n has
neighbor resultant `2*cos(2*pi/n)*exp(i*theta_j)` and is nonconstant but fixed.
Its local phase pressure is zero: it preserves phase structure without
supplying differentiated stationary EPI. Existing winding studies retain
their scope; this observation does not reopen the archived C6 campaign.

A different conditional escape is already present in the pressure algebra.
On a unit-conductance nonregular graph with consensus phase, constant positive
capacity and symbolic w_topo>0, `x=c*1-(w_topo/e)*k` has zero pressure.
The three-node path gives a nonuniform example without a new force term.
However it holds the irregular support and changes the retained zero topology
coefficient. On the actual retained weighted support, the independently read
compatibility scalar is
`d^T g_topo = -2170365805794839/4222124650659840`, which is nonzero.
Activating topology pressure alone there would cause mean drift under
consensus phase and uniform capacity, not a stationary pattern. No coefficient
is changed or fitted in this delivery.

**Disposition.** The stationary branch of these configured relaxation laws
is classified; stationary TNFR mechanisms in general remain open. Neither
fixed-point uniqueness nor a supplied profile establishes autonomous NFR
formation. The full triad, support and any genuine endogenous history must
carry the identity being tested. A new sustaining closure must be explicit;
configured lag schedules, REMESH echoes and target-based controllers cannot
be promoted silently to a law derived from the nodal equation.

The general result is the analytic proof above, independently reviewed against
the phase, adaptation and Sense Index owners. All 108 existing targeted phase,
adaptation, capacity-balance and forced-support tests pass; this is regression
evidence, not a numerical proof of the general theorem. Finite exact controls
and the authenticated weighted-topology calculation are retained in
`artifacts/research/source_closure_analytic_checks_2026_09_18.json`, with the
script `artifacts/research/validate_source_closure_2026_09_18.py`. This analytic
delivery changes no engine code, policy parameters or historical artifacts
and executes no graph, pressure kernel or trajectory.

## 22. Source tangency without a telemetry controller

The [diagnostic scope](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#7-derived-observables-and-dynamical-closure)
separates a derived observable from a justified causal law. The following
result advances the primary closure question directly from the existing
pressure channels; it neither uses Si nor supplies a new evolution policy.

### Regular fixed-support identity

Keep fixed symmetric conductance W, finite undirected unique-neighbor support U, constant
channel coefficients e>0, w_phi>=0, v=w_vf>=0 and w_topo=0. Assume positive
capacity and differentiable exact-real fields on an interval without events,
clipping, zero neighbor resultants or phase-wrap crossings. Write

`p=-e L_W x+F`, `F=w_phi*g(theta)-v L_U nu`, `x_dot=diag(nu)*p`.

These are the same pressure conventions as section 21, not a fitted source.
The phase support is unweighted even when W is weighted. For
`S_i=sum_{j in N(i)} exp(i*theta_j)`, its mean-response matrix is

`R_ij=1[j in N(i)] Re(exp(i*theta_j)/S_i)`.

Differentiating Arg(S_i) gives R. Its rows sum to one; they need not be
nonnegative. On the declared regular branch,

`g_dot=(R-I)*theta_dot/pi`.

The exact coefficient owner already exists:
[derive_phase_response](../../src/tnfr/physics/phase_response.py) computes
`mean_response=R` from admitted unit planar cosine Gram data. Use that field,
not its separately composed operator-stage Jacobian. The support gradients
remain owned by [support_transport](../../src/tnfr/physics/support_transport.py),
and [forcing_realization](../../src/tnfr/physics/forcing_realization.py) owns the
captured channel split. Neither a Gram table nor a captured binary64
pressure is a proof that a live continuous chart is admissible.

Differentiating the declared pressure, rather than reconstructing it from
measured EPI motion, now gives the identity

`p_dot=-e L_W diag(nu)*p + (w_phi/pi)*(R-I)*theta_dot - v L_U nu_dot`.

In particular, at p=0,

`p_dot=(w_phi/pi)*(R-I)*theta_dot-v L_U nu_dot`,

`x_ddot=diag(nu)*p_dot`.

This is the **source-tangency condition**: the phase-source and capacity-source
changes must cancel if EPI is to remain at zero pressure. The direct
capacity-rate term in x_ddot is `diag(nu_dot)*p`, which vanishes here;
capacity still acts through its contribution to pressure. The identity
does not determine theta_dot or nu_dot, nor a time for any operator to act.

At phase consensus, R=I-L_U. On connected support the condition is equivalent
to spatial constancy of

`(w_phi/pi)*theta_dot + v*nu_dot`.

For fixed phase and v>0, this requires **uniform capacity increments/rates**,
not uniform initial capacity. Thus a heterogeneous capacity-supported profile
from section 21 is not excluded by the nodal equation itself. If v=0, capacity
has no pressure-source constraint of this type; its positive mobility still
sets EPI response away from p=0.

### Finite events and what an instantaneous test cannot prove

Between two supplied fields on the same support with fixed x and channel
weights, the exact finite counterpart is

`p_plus-p=w_phi*(g_plus-g)-v L_U*(nu_plus-nu)`.

For a capacity-only change with fixed phase, v>0 and connected U, zero pressure
is preserved exactly when `nu_plus-nu` is uniform. A real event that also
writes EPI must include `-e L_W*(x_plus-x)`; changing support, coefficients,
clipping or stored operator pressure needs its own terms. This detached
identity is not grammar admission, event activation or runtime provenance.

A concrete algebraic control uses unit P2, consensus phase, e=v=1/2,
nu=(1/2,1) and x=(1,1/2). Its canonical pressure is exactly zero. A supplied
capacity increment (1/4,0), with x fixed, gives pressure (-1/8,1/8) and
post-change nodal rate (-3/32,1/8). The uniform increment (1/4,1/4) preserves
zero pressure. These are detached states checked against the existing
channel owners; no capacity law or graph evolution is introduced.

Pointwise tangency is only necessary. For a counterexample, allow the
explicitly *free, adversarial completion* nu(t)=(1+t^2,1) on P2, phase zero,
x(0)=0, e>0 and v>0; let x solve the original nodal equation. At t=0,
p=p_dot=x_ddot=0, but `p_ddot=x'''=(-2v,2v)`. This proves insufficiency of
one instantaneous test, not a proposed TNFR capacity law. Conversely, if
F stays constant throughout an interval and p(0)=0, the pressure equation
is homogeneous and uniqueness gives p(t)=0 there. This all-time condition
still supplies neither a law maintaining F nor stability after perturbation.

### Consequence for the primary research question

This identifies a missing relation rather than replacing it with a controller:
the nodal EPI equation and current pressure decomposition do not uniquely
specify capacity, phase, support or event activation. Uniform phase rotation
already lies in ker(R-I), and constant capacity shifts lie in ker(L_U).
Multiple mathematical completions are therefore possible; their existence
does not make them canonical TNFR mechanisms.

Geometry can preserve a phase pattern without sourcing differentiated EPI.
A stationary lifted local-only phase update with nonzero local gain requires
`wrap(Arg(S_i)-theta_i)=0`, so g_i=0 even outside a shared semicircle.
Regular winding does not evade this condition. Prescribed winding or an
imported particle label therefore cannot fill the missing source law.

This completes the bounded tangency calculation. Any proposed law for the
missing channel rates/activation must be independently justified and checked
against this identity. That requirement does not assign a new law-search
task; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the current gate.
No controller experiment, C6 restart, topology sweep or new public physics
claim follows. The finite exact controls reuse existing algebra and are
recorded in `artifacts/research/source_tangency_checks_2026_09_18.json` with
`artifacts/research/validate_source_tangency_2026_09_18.py`. The symbolic proof
above, not a finite test count, establishes the stated conditional identity.

## 23. Capacity exposure does not determine a phase clock

### Source and implementation boundary

The original [TNFR source](https://github.com/fermga/TNFR-Python-Engine/blob/6e1ffb8ffbadb667b11230c6af9f670d5f9d48b7/theory/TNFR.pdf) distinguishes structural frequency from
periodic oscillation. Its physical PDF pages equal the printed page numbers:

- Section 1.4.10, pages 48-49, presents a classical dynamical-system
  correspondence, including a Kuramoto model with omega identified with nu_f.
- Page 211 says structural frequency "no representa una oscilación periódica
  convencional"; pages 212-213 describe coherent-reconfiguration counts per
  structural-time interval.
- Page 224 qualifies the Kuramoto analogy: nu_f "mide reorganización, no ritmo".
  Page 218 describes structural time as an internal process coordinate.

These are compatible when the oscillator identification is a particular
constitutive correspondence, not a universal deduction. The audited PDF has
378 pages and SHA256
`5ba0f4a2da2d01e550e7004c3c09c6927e6620826052df84cd28babbcbd34fc3`.

The [shared oscillator proposal](../../src/tnfr/dynamics/phase_evolution.py) uses

`theta_i_plus=wrap(theta_i+dt*nu_i+dt*K*mean_U3 sin(theta_j-theta_i))`.

The nodal optimizer and FFT engine use this additional phase model beside
their EPI diffusion. It numerically interprets the supplied frequency as an
angular rate in radians per supplied time unit. By contrast, the ordinary
runtime calls [phase coordination](../../src/tnfr/dynamics/coordination.py), a
per-invocation global/local mean relaxation with no dt or free nu_i advance.
Neither path establishes that the other is a derived limit. The optional
phase-transport formula in [canonical.py](../../src/tnfr/dynamics/canonical.py)
also declares operational sensitivities; its name is not a uniqueness proof.

Multiplying by 2*pi would convert cycles to radians only after identifying
one counted event with a complete cycle. That identification is absent from
the nodal EPI law. The [Hz bridge](../../src/tnfr/units.py) is already explicitly
configured and does not supply it. No frequency factor, phase law, pressure
weight or runtime behavior is changed by this result.

### Exact independence, including a relative-phase witness

The EPI law and current pressure channels do not determine phase speed.
On unit P2, let x=c*1, nu=nu_0*1>0 and theta=a(t)*1. Every pressure channel
vanishes for every differentiable a(t). The same initial triad therefore
admits both constant phase and advancing common phase, with unchanged x and
zero phase separation. This is common-phase freedom only; it does not by
itself show that relative phases affect a prediction.

For that stronger distinction, take the existing zero-pressure family on P2,

`nu=(nu_0,nu_1)>0`, `nu_0!=nu_1`, `x_i(0)=c-(v/e)*nu_i`, `theta(0)=0`,

with held capacities and existing coefficients e>0, v>=0, w=w_phi>0.
There are two mathematical completions of these same nodal data:

1. Keep phases fixed. The canonical source is held, p=0 and x remains fixed.
2. Let theta_i(t)=nu_i*t and solve x_dot=diag(nu)*p with the same canonical
   pressure. This is an independence witness, not a proposed physical law
   or an executed operator sequence. Restrict time so the relative phase
   stays inside the chosen regular chart and U3 bound.

Set d=nu_1-nu_0 and a=e*(nu_0+nu_1)>0. In the second completion,
g=(d*t/pi,-d*t/pi), and pressure is (q,-q), where

`q_dot=-a*q+w*d/pi`, `q(0)=0`,

`q(t)=(w*d/(pi*a))*(1-exp(-a*t))`.

Consequently `x_0(t)=x_0(0)+nu_0*integral(q)` and
`x_1(t)=x_1(0)-nu_1*integral(q)` obey the original nodal law, with

`integral_0^t(q)=(w*d/(pi*a))*(t-(1-exp(-a*t))/a)`.

The EPI futures differ whenever d and w are nonzero. No pressure was solved
backwards from an observed derivative. These are analytic alternatives within
the incompletely closed equations, not a claim that both are admitted full
runtime trajectories. Thus phase closure changes structural predictions;
the ambiguity is not only a global phase convention.

The section 22 tangent gives the same initial result on connected support:

`p_dot(0)=-(w/pi)*L_U*nu`,
`x_ddot(0)=-(w/pi)*diag(nu)*L_U*nu`.

For nu=(1/2,1), their coefficients of w/pi are respectively (1/2,-1/2)
and (1/4,-1/2). The existing sine-coupling proposal has zero interaction at
initial phase consensus, so its initial free-advance consequence does not
depend on selecting K. Its later coupled path is not identified with the
uncoupled analytic completion above. The failed implication is universal
phase speed from the EPI equation; configured oscillator models are not
thereby mathematically forbidden.

### What the nodal equation does derive: accumulated capacity

On a regular interval with nu_i(t)>0, define

`s_i(t)=integral_t0^t nu_i(u) du`.

Changing parameter along that same trajectory gives exactly

`dx_i/ds_i=p_i(t_i(s_i))`.

No conversion factor or additional driving term is introduced. This is
accumulated capacity, not circular phase or a measured physical clock.
Neighbors in p_i must still be evaluated at the shared time t_i(s_i); separate
local clocks cannot be treated as simultaneous independent network clocks.
A common clock removes every nodal prefactor only when the capacities agree
at each time; a common factor in heterogeneous capacities leaves their
relative factors. Zero capacity stalls the clock and invalidates this regular
inverse. At p=0, accumulated capacity may advance while EPI stays fixed.

The pure-EPI common-clock solution and heterogeneous boundary already belong
to [directed transport section 6](../TNFR_DIRECTED_NONNORMAL_DYNAMICS.md#6-structural-time-and-heterogeneous-capacity).
The finite-exposure retention result in
[cycle support section 4](../CYCLE_SUPPORT_DYNAMICS.md#4-default-attenuation-can-retain-epi-as-capacity-tends-to-zero)
is reused, not rerun. Capacity also enters the full pressure through
`-v L_U nu`; a time-coordinate change must not be confused with rescaling
capacity while silently holding that source unchanged.

**Disposition.** The capacity-to-phase gate is closed with an independence
result and a derived reparameterization. A fundamental law for relative
phase/capacity evolution remains missing. Geometry can constrain compatible
motions before a speed is assigned; the execution plan owns that next bounded
question. No oscillator sweep, telemetry controller or physical-particle
claim follows from this result. The finite controls and source provenance
are retained in `artifacts/research/capacity_phase_checks_2026_09_18.json`
and `artifacts/research/validate_capacity_phase_2026_09_18.py`.

## 24. Rigidity and flexibility of a held phase source

### Regular rigidity from the shared mean derivative

On fixed finite support with nonempty neighborhoods, use section 22's
`S_i=sum_j exp(i*theta_j)` and `R_ij=1[j in N(i)] Re(exp(i*theta_j)/S_i)`.
Assume nonzero resultants and a regular center-to-mean wrap chart. Then
`Dg=(R-I)/pi` and `R*1=1`. This statement refers to the canonical full-support
phase channel, not a new phase-update equation or a U3-filtered mean.

If R is nonnegative and its positive-entry directed graph is strongly
connected, its fixed-vector space is exactly the common-rotation line.
Indeed, a maximal component of a vector satisfying R*h=h is a convex mean
of its neighbors. Every neighbor with a positive coefficient must share that
maximum; connectivity propagates it to every node. Thus

`ker(R-I)=span{1}`, `rank(R-I)=n-1`.

A differentiable constant-g path staying in this domain satisfies
`(R-I)*theta_dot=0`, hence `theta_dot=a(t)*1` and
`theta(t)=theta(t0)+c(t)*1`. Common rotation conversely preserves g. Its
speed remains unspecified. This is a compatibility theorem, not relaxation,
attraction, self-maintenance or a stability theorem for the full triad.

There is also local uniqueness modulo rotation: fix one phase coordinate.
The remaining derivative has rank n-1, so n-1 independent output coordinates
give a locally invertible map. Nearby states with the same full g differ
only by rotation. This does not prove global reconstruction or connectedness
of a level set. The already recorded consensus and regular winding on a cycle
can both have g=0 and individually rigid derivatives while belonging to
different local branches; no winding campaign is repeated here.

A sufficient geometric domain on connected undirected support is that every
neighbor phasor has positive projection onto its row resultant:

`Re(exp(i*theta_j)*conj(S_i))>0` for every support neighbor j.

This makes R positive on all support edges. A common phase lift of width
strictly below pi/2 guarantees it, since every numerator is a sum of positive
pairwise cosines; it also ensures regular resultants and wraps. The rowwise
projection criterion is wider and must retain its separate wrap premise.

For **zero phase source**, a further useful corollary uses the actual U3
scale. If g_i=0 and every edge separation is strictly below pi/2, S_i points
along theta_i and

`R_ij=1[j in N(i)] cos(theta_j-theta_i)/|S_i|>0`.

Connected support then has the same rigidity. This does not automatically
extend to nonzero g: the row resultant need not point along its center.
The closed pi/2 gate permits zero projections and needs separate treatment.

### Connected support alone is insufficient: a cube family

Use the unit-conductance cube Q3=C4 x P2, with nodes (j,l), j modulo four,
l in {0,1}, horizontal neighbors (j-1,l),(j+1,l) and partner (j,1-l).
Assign the same four phases to both layers:

`theta_(j,l)=(a,b,a+pi,b+pi)_j`.

Each horizontal pair is antipodal and cancels exactly. The partner has the
center's phase, so `S_i=exp(i*theta_i)` and `g_i=0` for every a,b. Resultants
have squared magnitude one and the center-to-mean displacement is zero.
Varying b-a therefore gives a genuine finite source-preserving deformation,
not merely an extra direction of a pointwise derivative. Here a,b are phase
coordinates, not new force coefficients or a proposed activation law.

At quadrature b-a=pi/2, R is the vertical-partner permutation. It is
nonnegative but reducible into four two-node classes despite connected
graph support. Its rank(R-I)=4 and tangent dimension is four. At the exact
phasor point cos(b-a)=3/5, sin(b-a)=4/5, the horizontal derivatives are
signed and the rank is six, giving tangent dimension two. With c=cos(b-a),
the spectra of R-I split by layer parity:

`layer-symmetric: (0,0,2c,-2c)`,
`layer-antisymmetric: (-2,-2,-2+2c,-2-2c)`.

The displayed family supplies two finite phase parameters (one common and
one relative); tangent dimension four at quadrature does not prove that
all four directions integrate into a four-dimensional finite level set.

**U3 boundary.** The horizontal circular separations are |wrap(b-a)| and
pi-|wrap(b-a)|. Requiring every edge to obey the default pi/2 bound forces
quadrature. Any relative-angle departure violates one horizontal edge.
This family is therefore not an all-edge U3-compatible Coupling orbit.
The existing pressure channel reads all support neighbors; silently dropping
U3-incompatible neighbors changes its definition and destroys this argument.
No graph creation mechanism, admitted event sequence or phase-speed law is
supplied by the geometric family, and the cube is not a selected emergent
polyhedron or a particle identification.

The EPI consequence still follows exactly in its declared mathematical
scope. The unit cube is regular, so L_W=L_U=L and topology pressure is zero.
For held positive capacities, the existing family

`x=c0*1-(v/e)*nu`, `e>0`, `v=w_vf>=0`,

has `p=-L(e*x+v*nu)=0` throughout the phase deformation. It is differentiated
when v>0 and nu is nonconstant. This reuses the capacity-supported balance;
it derives neither the capacity distribution nor a law maintaining it.

### Reusable exact observation and claim boundary

[observe_phase_source_geometry](../../src/tnfr/physics/phase_response.py) rebuilds
the existing PhaseResponseReference from its primitive Gram/incidence data
and observes rank(R-I) with the shared exact-rank owner. It returns the
scaled source derivative, tangent dimension and mean-response sign. The
merged operator-stage Jacobian is deliberately separate: at phase_factor=0
that Jacobian is the identity even when the source remains locally rigid.
An exact rank n-1 also certifies conditional local rigidity for some signed
R, without needing the sufficient nonnegative proof.

For example, K4 with phasors (1,0) on three nodes and (-3/5,4/5) on the
fourth has negative mean entries -1/13 but rank(R-I)=3. Negative entries
alone do not imply geometric freedom. Conversely the quadrature cube is
nonnegative yet has more than the rotation line. Zero resultants are rejected;
a live phase chart, U3 admission, causal execution and temporal stability are
not established by an exact Gram or rank calculation.

The [portable controls](../../tests/physics/test_phase_source_geometry.py) cover
these independent matrices, exact cube cancellation, disconnected support,
permutation covariance and rejection of tampered cached references. The
general rigidity and finite-family proofs are analytic, not inferred from
the number of passing controls. This closes the planned geometric gate:
regular rigid regions and flexible boundary families both exist. The sole
execution plan owns the continuation; no controller or new evolution law
follows from this result. The subsequent
[grammar audit](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#10-u3-exact-geometric-content-and-a-strict-gate-counterexample)
settles the strict-U3 nonzero-source rank question: a connected double-star
has a two-dimensional tangent kernel strictly inside the gate, but its extra
direction is obstructed at second order. Its local finite level set has only
common rotation. The remaining question concerns finite geometry, not another
rank calculation. The proof and portable fixture have one owner in that audit.

## 25. Relational time and synchronization are separate claims

The original source describes Reception in terms of shared/relational time
and synchronization (PDF page 83), and internal process time on
pages 218-219. This motivates a relational-clock hypothesis; it is not a
derivation that global synchronization equals elapsed time. Section 23 owns
the existing accumulated-capacity identity and its source/units audit.

**Alignment need not advance, even while form evolves.** On pure-EPI unit P2,
positive equal capacities and phase consensus give `R=1`. Holding those phases
fixed is compatible with the EPI equation while a nonuniform form relaxes:
for capacity one and initial EPI `(1,0)`,
`x(t)=((1+exp(-2t))/2,(1-exp(-2t))/2)`. Synchronization remains complete while
form and its accumulated capacity change. Conversely, uniform EPI/capacity
permit any differentiable common phase rotation under that same EPI identity.
The alignment statistic does not identify its speed. Neither control selects
a complete physical phase law.

On the retained prism with phases `(-a,a,0)` in both triangles, `|a|<pi/4`,
the same statistic is `R=(1+2*cos(a))/3`. It is even in `a` and has zero
derivative at `a=0`. Along the supplied control `a=A*cos(chi)`, it repeats
after half the full phase cycle. It therefore loses orientation and cannot
serve as a globally invertible clock for that motion. This concerns phase
alignment; a broader notion of coordination involving form, capacity and
history must supply its own observation and evolution, not inherit this
statistic's name.

### A local state clock requires an already specified tangent

For a complete autonomous state law `z_dot=V(z)` and a scalar observation
`tau=T(z)`, a regular local time coordinate requires
`h=dT[V]>0`. Then `dz/dtau=V/h`. If the relevant components of `V` are missing,
the chain rule does not generate them. A single-valued real state function
cannot increase strictly around a closed orbit: its endpoint difference is
zero, whereas the integral of a strictly positive rate would be positive.
An unwrapped angular clock needs a chart/history or cycle count, as well as
a law for its advance; a circular phase alone supplies neither.

A regular positive reparameterization preserves the oriented path, and an
onto unbounded time change preserves recurrence. Finite accumulated exposure
can instead map infinite original time to a finite internal-time endpoint;
that retention mechanism is already covered by the capacity results and is
not a proof of continuing active oscillation. Relabeling time does not turn
the pure gradient trajectories of variational section 13.10 into recurrent
ones.

### Curve admission can determine a speed without selecting the curve

For a declared full-state curve `z(chi)`, let `v=dx/dchi` be its EPI tangent
and let `b=diag(nu)*p` be the nodal rate evaluated independently from that
state and the existing pressure law. A regular positive scalar clock must
satisfy

\[
b=h v,\qquad h=d\chi/dt>0.
\]

For `v!=0`, this is equivalent to collinearity and `v^T b>0`; the only
possible speed is `h=(v^T b)/(v^T v)`. The inner product merely computes the
unique proportionality coefficient and introduces no physical metric or
force. Every component must agree. If exactly one of `b,v` vanishes there
is no regular positive clock; if both vanish the EPI equation leaves the
clock unconstrained at that point. At an EPI turning point, phase or other
coordinates may still move, so failure to identify the clock there must
not be confused with complete-state stationarity.

This is an EPI admission condition for a supplied curve, not its generation
mechanism or a complete phase/capacity law. The pressure must not be obtained
retrospectively as `h*v/nu`. It provides a useful rejection test: changing a
clock cannot repair a source tangent pointing in the wrong direction.

### Reuse and implementation scope

The existing `structural_time` reader numerically accumulates supplied
capacity over the supplied grid, starting at its first point. Its trapezoidal
value is an estimate, not an exact integral for an arbitrary capacity
function. `certify_structural_time` now uses that accumulated exposure as its
finite structural observation window, including zero exposure; it previously
used the final input timestamp instead. It remains a finite numerical
diagnostic with an unassessed tail, not a derived physical clock or a general
infinite-time certificate.

Controls: [clock scope](../../tests/physics/test_structural_clock_scope.py) and
[existing structural-time implementation](../../tests/physics/test_structural_time.py).
The complementary [oriented source/form work identity](../TNFR_VARIATIONAL_PRINCIPLE.md#1312-oriented-sourceform-work-without-a-selected-clock)
allows a proposed loop to be rejected before choosing its speed. The
[single execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
uses these conditions within G3; no synchrony statistic is promoted to a
controller or a fundamental time law.
