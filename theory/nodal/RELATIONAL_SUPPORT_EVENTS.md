# Native relational support events and synchronization

Sections 8–13 specify supplied attachments, reset budgets, occurrence-law
nonselection and precontact locking. They retain native relational flow,
operator maps and event premises as separate models; an admitted event or
synchronized preparation does not select its occurrence.

Part of [Native regional composition](RELATIONAL_PATTERN_COMPOSITION.md).
Section numbers remain stable across this collection. Each result keeps
its full hypotheses, implementation and checks; the
[execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone assigns research work.

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
deformation. The [existing tangent and nonlinear closure results](RELATIONAL_PATTERN_COMPOSITION.md#1-fixed-model-and-observation)
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
controls. Primitive support birth remains unresolved and has its own
[connection-mechanism admission](RELATIONAL_EFFECTIVE_CONNECTIONS.md#connection-mechanisms-and-mediators);
collective organization on finer support does not settle that question.
A threshold, dwell time or phase-slip count can define a configured observation,
but is not derived merely by naming synchronization. The
[sole queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next quantitative admission without reopening completed modal,
transmission or memory calculations.
