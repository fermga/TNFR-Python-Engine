# THOL compatibility: temporal evidence, hierarchy and the complete-law domain

THOL has a pressure action on the existing nodes and, on its public or atomic
route, a conditional construction of a child. These have different continuation
contracts. A pressure-only event leaves the physical sine state unchanged and
can return to its constitutive pressure by refresh. Public child creation adds
an isolated coordinate; it is a valid runtime hierarchy event but leaves the
connected-support domain of the selected complete law.

This owner closes that distinction for general admissible states and for the
marked collective contact as a conditional control. It does not require the
EN/IL/SHA/NAV sequence, supply missing research histories or identify a child
with an autonomously maintained organization. The
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns status. The [registered action](../STRUCTURAL_OPERATORS.md#62-self-organization-thol),
[THOL pressure owner](../THOL_PRESSURE_FEEDBACK.md) and
[native birth and transport owner](../THOL_BIRTH_AND_TRANSPORT.md) retain their
respective definitions and earlier conditional results.

## 1. State and selected continuous law

Let U be a finite simple loop-free connected undirected support with at least
two nodes. Retain signed scalar form x, primitive circular phase theta, finite
nonnegative conductances W, held nonnegative capacities nu and a declared
structural clock. Uniform-real BEPI uses the same signed scalar admission;
complex/nonuniform BEPI and magnitude projection do not replace it. The
[shared weighted-form, bare-phase law](SINE_COLLECTIVE_RECEPTION.md#1-complete-model-and-event-boundary)
is

\[
 q=L_Wx,\quad S_i=\sum_{j\sim i}\sin(\theta_j-\theta_i),\quad
 d_i=|N_i|>0,\quad M=\operatorname{diag}(m_i),\quad
 m_i=\frac{\nu_i}{\pi d_i},
 \qquad p_i=\frac{S_i}{\pi d_i},\qquad
 \dot x=MS,\quad\dot\theta=Mq.
 \tag{1}
\]

Bare neighbors include supported zero-conductance edges. Pressure is specified
before evaluating the form response. There is no forcing or continuous loss.
The [shared reader](../../src/tnfr/physics/relational_sine_support.py) admits
nonnegative capacities and conductances; positivity is an additional premise
where an inverse-capacity mean or a nontrivial weighted coupling is needed.
At zero capacity both rows of this particular completion vanish at that node;
the general nodal identity alone would freeze only its form row.

With declared phase lifts, the selected storage is

\[
 H=\frac12x^TL_Wx+
 \sum_{\{i,j\}\in U}\bigl(1-\cos(\theta_j-\theta_i)\bigr).
 \tag{2}
\]

It is conserved by (1), since the form and phase work cancel. When all
capacities are positive, both means with weights `rho_i=d_i/nu_i` are also
conserved. These are mathematical storage and lifted-coordinate identities,
not laboratory energy or a physical identification.

In addition to this physical state, THOL consumes target-associated temporal
history, stored pressure, grammar context, hierarchy and child identifiers,
configuration and execution order. Metabolic construction consumes actual
bare neighbor forms and phases. Its acceleration and child diagnostics do
not replace any of these primitive inputs.

## 2. Acceleration predicted by this law and acceleration actually observed

Let `C(theta)` be the bare cosine Laplacian: an edge has coefficient
`cos(theta_j-theta_i)`, irrespective of W. Direct differentiation of (1),
with held capacity and support, gives

\[
 \ddot x=-M C(\theta)M L_Wx,
 \qquad
 \ddot x_i=m_i\sum_{j\sim i}\cos(\theta_j-\theta_i)
                      (m_jq_j-m_iq_i).
 \tag{3}
\]

Thus the selected completion does have structural acceleration. It is neither
an independently selectable extra state variable nor the cached acceleration
telemetry. In any specified state,

\[
 |\ddot x_i|\leq m_i\sum_{j\sim i}|\cos(\theta_j-\theta_i)|
                              (m_j|q_j|+m_i|q_i|).
 \tag{4}
\]

There is no universal finite acceleration ceiling obtained from capacity
alone on unrestricted signed form space. For a constructive example, take
`theta_i=0`, every other phase `phi` with `0<phi<pi/2`, and forms
`x_i=b+u`, `x_j=b` elsewhere. If `nu_i>0` and
`w_i=sum_{j~i}W_ij>0`, then

\[
 p_i=\frac{\sin\phi}{\pi}>0,\qquad
 \ddot x_i=-u m_i\cos\phi
       \left(d_i m_i w_i+\sum_{j\sim i}m_jW_{ij}\right).
 \tag{5}
\]

The parent acceleration has either sign and exceeds any supplied finite
magnitude threshold for a sufficiently large form contrast. A common shift
b can independently place its form above a supplied minimum. All edges are
initially acute; this is a zero-winding preparation, not the marked winding-one
family. The strict inequality persists on a neighborhood of admissible
positive-capacity/positive-conductance preparations. Finite form rails,
specified mean or storage leaves and particular preparations require their
own bounds; (5) does not show that they contain this example.

Smooth local flow through a state with `abs(ddot x_i)>tau` implies the
existence of sufficiently short genuine three-sample windows with the same
strict magnitude inequality. This follows from convergence of the three-point
divided difference to the derivative. It supplies a compatibility theorem,
not an executed window, recorded history, target selector or grammatical
context. For one unit edge, capacities one, phase difference `pi/3` and
`x=(3,1)`, equation (3) gives `ddot x_0=-2/pi^2`, whose magnitude exceeds 0.1.
No trajectory is computed to establish this algebraic example.

Actual THOL uses the shared
[acceleration reader](../../src/tnfr/operators/nodal_equation.py). For its
last three active timestamped samples, it computes

\[
 h_1=t_1-t_0>0,\quad h_2=t_2-t_1>0,\quad
 s_1=\frac{x_1-x_0}{h_1},\quad s_2=\frac{x_2-x_1}{h_2},\qquad
 \widehat A=\frac{2(s_2-s_1)}{h_1+h_2}.
 \tag{6}
\]

The timestamped source is authoritative when selected; its endpoint EPI must
match current signed form. Legacy histories instead use unit operator steps
and do not authenticate the endpoint or a continuous-flow interval. Fewer
than three active samples give explicit unavailability in the observation;
the compatibility numerical function returns zero in that case. Neither this
fallback nor a cached `D2_EPI` establishes observed zero acceleration.

The reader checks supplied sample values and ordering, not an independent
live physical clock or producer provenance. A record can have a matching
endpoint value while referring to a different interval. The execution
protocol must supply temporal continuity, units, pressure law, event record
and clock authority before (6) can be interpreted as evidence from (1).
Operator-step records and a structural-time record are different inputs.

For endpoint errors bounded by `epsilon_0,epsilon_1,epsilon_2` at fixed exact
times, the observation error obeys

\[
 |\delta\widehat A|\leq\frac{2}{h_1+h_2}
 \left(\frac{\epsilon_0+\epsilon_1}{h_1}
       +\frac{\epsilon_1+\epsilon_2}{h_2}\right).
 \tag{7}
\]

Arithmetic and time/solver errors require additional budgets. Primitive
nonzero values lost to materialization and nonfinite arithmetic are not
evidence of a zero response. A computed threshold decision is not an exact
derivative theorem; a strict true margin must dominate the observation error
before its decision transfers.

## 3. Admission, grammar and the actual event branches

The [public proposal](../../src/tnfr/operators/self_organization.py) reads
the active history and applies the checked stored-pressure jump

\[
 p_i^+=p_i+a\widehat A_i,\qquad x_i^+=x_i,\quad
 \nu_i^+=\nu_i,\quad\theta_i^+=\theta_i,
 \qquad a=\mathrm{THOL\_accel}>0.
 \tag{8}
\]

The configured default is `1/(4*pi)`. A supplied finite nonnegative threshold
tau selects the child proposal only when `abs(A_hat)>tau`; equality produces
no child. Reaching the configured hierarchy depth also suppresses birth while
retaining (8). Missing acceleration gives the compatibility zero update if
optional strict preconditions are disabled, without turning missing evidence
into a measured zero. A graph-backed primitive dispatch performs this pressure
proposal without hierarchy construction; a graphless primitive instead has an
explicitly supplied acceleration contract. These routes are not interchangeable.

When enabled, the shared optional strict gate additionally requires admitted
form and capacity minima, positive stored pressure, configured connectivity
and at least three active samples. It does not require the birth threshold
to be crossed. Full proposal validation separately checks hierarchy references,
depth, child-domain bounds, metabolic configuration and representable outputs.
Signed scalar admission is general; the configured nonnegative THOL minimum
restricts the stricter execution domain without redefining signed EPI.
The shared `resolve_thol_min_epi` policy resolver supplies both strict runtime
and detached admission, with unchanged core default `0.2`; a present invalid
override rejects. The [actual child's temporal result](SINE_CHILD_TEMPORAL_READINESS.md)
shows why above-threshold acceleration and positive fresh pressure do not
replace this independent form requirement or acquired history.

Grammar is independent: THOL is a stabilizer, a bifurcation handler and a
transformer that requires a recent destabilizer under U4b. It is not a closure.
`OZ -> THOL -> SHA` is a valid complete word with existing nonzero form when
its live conditions pass. This is an example, not a compulsory evolution law.
The synthetic tests execute it on an independently prepared no-child case.
Their manufactured temporal records are reader/event controls, not observations
of a retained organization. Neither passing grammar nor (5) supplies those
records to an actual research object.

The atomic stage proposes every target from one immutable snapshot, allocates
collision-free child identities in snapshot-node rank and validates the merged
hierarchy before committing. Histories and diagnostics retain requested target
order. Public single-target execution has its own transaction; sequential
public calls do not become one simultaneous stage. No callback external-effect
rollback or future stability follows from graph-state atomicity.

## 4. Pressure-only continuation and its complete accounting

When no child is committed, all physical coordinates consumed by (1), bare
support, conductances and capacities remain fixed. Consequently

\[
 \Delta H=0,\quad\Delta\mu_x=\Delta\mu_\theta=0,\qquad
 p_{\rm fresh}^+=p_{\rm fresh},\quad
 \dot x_{\rm fresh}^+=\dot x_{\rm fresh},\quad
 \dot\theta_{\rm fresh}^+=\dot\theta_{\rm fresh}.
 \tag{9}
\]

The mean statements require positive capacities. These identities hold for
one or simultaneous targets, with no pressure cross term in H because H does
not consume stored pressure. History, debt and diagnostics can still change;
the contextual event is not the identity map on sufficient execution state.

The raw stored pressure in (8) generally lies off the constitutive relation
`p=S/(pi*d)`. Immediate selected-law refresh erases its direct contribution.
Holding the changed pressure for a later interval instead changes the complete
model: its form row is no longer the refreshed row of (1). Pressure-only event
compatibility therefore includes an explicit handoff; it does not silently
promote stored pressure to an independent evolving coordinate. A finite-duration
held-pressure response would require its own law, clock, phase row and budget.

The separate scalar diagnostic satisfies

\[
 \frac12\bigl[(p_i+a\widehat A_i)^2-p_i^2\bigr]
 =a p_i\widehat A_i+\frac12a^2\widehat A_i^2,
 \tag{10}
\]

which can have either sign. THOL's stabilizer role is not universal pressure
contraction. Subsequent OZ, SHA or other word stages have their own resets;
(9) is the THOL stage budget, not a whole-word identity.

## 5. Child creation changes the state dimension and law domain

If its threshold, depth and proposal domains pass, public THOL adds a child c
with the actual represented endpoints

\[
 \nu_c^+=0.95\,\nu_i,\qquad
 \theta_c^+=\theta_i\pmod{2\pi},\qquad p_c^+=0,
 \qquad x_c^+=\mathcal B_i(x_i,\{x_j,\theta_j:j\sim i\}).
 \tag{11}
\]

The products and wrapping denote the admitted runtime results, not exact
decimal identities after binary64 evaluation. The amplitude map is owned by
[metabolism](../../src/tnfr/operators/metabolism.py): before final configured
child bounds, it clips `0.3*x_i + g*(mean_neighbor_x-x_i) + k*phase_variance`
to [0,1] when metabolism is enabled, and `0.3*x_i` otherwise. Its inputs use
bare neighbor multiplicities and circular phase variance; conductance-weighted
averages and a simple cell graph do not substitute for them. The configured
EPI interval must intersect [0,1]. Acceleration selects eligibility and is
recorded; it is not a second amplitude multiplier.

The event writes the parent/child association, hierarchy path and level,
`sub_nodes`, `sub_epis` and graph hierarchy. It leaves every old form, capacity,
phase and edge unchanged. The child has a single legacy EPI sample, so it does
not inherit the parent's observed acceleration. No parent amplitude is removed
and no neighbor form is metabolically depleted; this construction is not an
additive mass split. A common form shift need not commute with the bounded
child construction, even though the old continuous law has that symmetry.

**The whole-graph continuation obstruction is exact.** THOL creates no edge.
The child has `d_c=0`; the resulting graph is disconnected and the denominators
in (1) are undefined there, even if its capacity happens to be zero. The
existing shared sine reader correctly refuses this successor. The pressure
value zero and nesting metadata do not resolve `0/0`, provide a child phase
law or create coupling. Optional strict THOL connectivity applies to the target
before the event; it does not certify the newborn's continuous-law domain.

Restricting the actual successor to the old component gives exactly the same
fresh (1), H and inverse-capacity means as before the event. This is a
restriction result, not a whole-successor certificate. In particular a stage
whose final callback demands the existing full-graph sine refresh rejects and
rolls back the birth and contextual writes; it cannot return an admitted
selected-law successor by silently ignoring the child.

### An explicit comparison extension, not an installed law

One mathematically consistent extension would declare (1) on every connected
component of size at least two and, on each isolate, explicitly set

\[
 p_c=0,\qquad\dot x_c=0,\qquad\dot\theta_c=0.
 \tag{12}
\]

This is a supplied frozen-isolate completion, not a consequence of dividing
by zero or an implementation change. It has a smooth vector field on each
fixed-support stratum. Choosing the same edge-only storage (2) makes its old
component conservation proof apply verbatim; isolates have no storage term.
With that explicit convention, isolated birth has zero selected edge-storage
jump and leaves the old component means fixed. Assigning an intrinsic child
storage would change the birth budget and require its own premise.

For positive capacities, the old component weights remain `rho_i=d_i/nu_i`.
Setting an isolate's weight to zero only defines an observation that ignores
it; it is not a complete mean of the enlarged organization. Any additional
positive fixed child weight kappa would also give a conserved weighted mean
under (12), but birth shifts that mean by
`kappa*(x_c-mu_x)/(sum(rho)+kappa)` for one child. No particular kappa is
selected by (1). The frozen child's coordinates are independent constants,
not a coupled higher-scale dynamical mode.

Other isolate laws or actual connections are possible supplied completions.
They are not selected by the hierarchy record. Connecting a child changes
degrees, both rows, phase pressure, edge storage and mean weights; it needs
its own admitted support event and work accounting. The native birth/transport
results cannot transfer their diffusion law or coefficients to this sine
completion. This owner installs no extension or child-coupling policy.

The later [hierarchy-to-contact composition](SINE_HIERARCHY_CONTACT.md)
explicitly selects this comparison for its intermediate prefix and supplies
a separate shared detached reader. It retains the connected reader's
rejection, the additional isolate-law premise and the absence of native flow
dispatch. Its small connected source family does not establish birth or
maintenance of the marked winding contact studied here.

## 6. Collective descent, U5 and clock compatibility

For the marked twenty-node, forty-five-edge contact, the contextual eleven-cell
chart retains singleton ports 0 and 1 and nine physical twin pairs. It must
also retain per-member temporal samples/units, grammar histories and debt,
hierarchy and identifiers, support multiplicities, construction parameters,
stored pressure, semantic data and ordered targets. A mean history or unordered
list of histories does not determine which parent crosses tau or which child
is constructed. The full form origin matters to the child amplitude.

Pressure-only events at these singleton ports leave physical twin equalities
and the continuous reduction intact after refresh. Birth at both ports keeps
the old twins but adds two associated singleton children: the physical chart
has thirteen cells and twenty-six form/phase coordinates, plus its hierarchy
and temporal context. Its support is disconnected; describing that chart is
not admission to (1). More general twin-target events require compatible
target sets, histories, configuration and child association before a descended
event is defined. Physical equality alone is insufficient.

U5 has separate structural and numerical obligations. The proposal validates
a coherent hierarchy record. The detached
[U5 assessment](../../src/tnfr/physics/multiscale_coherence.py) additionally
compares `C_parent >= alpha*sum(C_child)` for an explicitly supplied alpha
and tolerance. It requires explicit finite stored-pressure and change-rate
inputs for every assessed parent and child; missing channels reject rather
than become zero observations. A newly constructed child has no supplied
change-rate observation, so birth alone does not establish a numerical U5
certificate. With actual or explicitly supplied diagnostic inputs, valid
nesting can pass one alpha and fail another. Neither a single pass nor the
variance-based sub-EPI amplitude alignment proves subsequent maintenance.

Under an affine clock change `s=c*t+b`, `c>0`, one way to retain the form of
both rows in (1) is `nu_s=nu_t/c`, with pressure unchanged. Then

\[
 \widehat A_s=\widehat A_t/c^2,\quad
 \tau_s=\tau_t/c^2,\quad a_s=c^2a_t.
 \tag{13}
\]

Transforming capacity minima and all history timestamps consistently preserves
the same event decision and stored-pressure increment. Keeping numerical tau
and gain fixed instead defines a different policy. If capacity units are held
fixed and pressure is instead rescaled, the phase row and gain need their
corresponding transformations as well. A nonlinear time change also adds the
usual acceleration term `-xdot*s''/(s')^3`; finite differences must be rebuilt
from the transformed sample intervals. Synchronization or operator count does
not supply such a clock transformation.

## 7. Verification and remaining scope

[Independent synthetic controls](../../tests/operators/test_collective_hierarchy.py)
exercise the complete no-child word, the two public branches and primitive
distinction, actual metabolic multiplicities, unchanged old-component fields,
isolated-child rejection and graph rollback, independent U4/U5 conditions,
signed temporal evidence and affine clock scaling. The law-acceleration control
compares independently differentiated edge equations with shared instantaneous
fields. The supplied quadratic records are explicitly manufactured reader
inputs; no continuous trajectory or retained research response is executed.

The shared history, proposal and transaction owners retain their broader tests;
[Testing](../../TESTING.md) owns selection. These contracts establish a scoped
partial action and a precise complete-law obstruction. They establish neither
an observed THOL birth under (1), autonomous occurrence, a recursively coupled
organization, future maintenance nor physical identification.
