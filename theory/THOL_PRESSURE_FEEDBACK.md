# Self-organization pressure, refresh and structural feedback

**Status:** Scoped operator and nodal-integration identities; finite runtime
observations distinguish pressure retention from refresh and hierarchy birth.
**Research links:** B2.d.5/O3.a, S3, S8, S9, S10 and S16.

## 1. What the canonical THOL operation actually changes

The public
[SelfOrganization implementation](../src/tnfr/operators/self_organization.py)
prepares the complete action before committing it atomically. On the selected
parent, let `x` be scalar EPI, `nu` capacity, `phi` phase and `p` stored
DeltaNFR. For an actually admitted THOL action, its primary channel proposal is

$$
p_T=p+a\widehat A,\qquad x_T=x,\qquad
\nu_T=\nu,\qquad\phi_T=\phi,
$$

where `a=THOL_accel>0` is the existing operational coefficient and
`A_hat` is the signed structural acceleration reconstructed from active
EPI history. The cached acceleration attribute is telemetry; public THOL
does not treat an independently assigned cached value as its history.
The displayed algebra treats supplied coefficients and states as exact
real values; executed binary64 arithmetic is identified with its measured
residuals rather than assumed equal to that reference.
The pressure write occurs whether or not the child-creation threshold is
crossed. A threshold crossing `abs(A_hat)>tau` permits a child proposal
subject to the configured hierarchy-depth and other domain checks.

The instantaneous local pressure magnitude can increase or decrease:

$$
\tfrac12(p_T^2-p^2)
=a p\widehat A+\tfrac12(a\widehat A)^2,\qquad
|p_T|\leq|p|\iff
a\widehat A(2p+a\widehat A)\leq0.
$$

Thus the U2 stabilizer role does not establish a universal local pressure
contraction, a Lyapunov decrease or an attractive EPI profile. Those are
separate dynamical claims. The glyph itself leaves the parent's EPI fixed;
an EPI response to its pressure requires an actual subsequent integration
interval or another named operator action.

## 2. Structural acceleration is a measured history diagnostic

The shared
[`compute_d2epi_dt2`](../src/tnfr/operators/nodal_equation.py)
owns the acceleration calculation. With an active timestamped history,
the last three samples `(t0,x0),(t1,x1),(t2,x2)` give

$$
h_1=t_1-t_0>0,\quad h_2=t_2-t_1>0,\qquad
s_1=\frac{x_1-x_0}{h_1},\quad
s_2=\frac{x_2-x_1}{h_2},\qquad
\widehat A=\frac{2(s_2-s_1)}{h_1+h_2}.
$$

The active samples must be finite, their times strictly increase, and the
final EPI must match the current parent state. A timestamped source takes
precedence over legacy histories. Fewer than three active samples return
zero as an unavailable-acceleration result; that return alone does not
prove that a measured trajectory has zero curvature. The legacy
`epi_history` and `_epi_history` paths retain unit operator-step spacing,
so they use `x2-2*x1+x0`. They are not silently assigned physical times.

The nodal equation explains what this diagnostic measures. On an
unclipped interval with no additional EPI jump,

$$
s_j=\frac1{h_j}\int_{t_{j-1}}^{t_j}\nu(t)p(t)\,dt.
$$

For two exact held-input intervals this reduces to

$$
\widehat A=
\frac{2(\nu_2p_2-\nu_1p_1)}{h_1+h_2}.
$$

The diagnostic therefore compares previously accumulated nodal rates.
On a smooth interval the ordinary product rule also gives
`x''=nu'*p+nu*p'`; it does not introduce a new acceleration law or inertial
degree of freedom. Finite samples can straddle changes of held pressure or
capacity, and no derivative-limit theorem follows merely from their
three-point estimate. The standard
[physical boundary recorder](../src/tnfr/dynamics/runtime.py) restarts
history at a detected same-time EPI jump, excluding that jump from the
following physical secants. The reader alone does not certify this causal
provenance for caller-prepared histories. Clipping, unrecorded jumps and
binary64 arithmetic require their own measured residuals.

## 3. Hierarchy birth does not itself connect a child to the parent walk

When birth is admitted, THOL creates a graph node with a parent reference,
hierarchy path and level, and synchronized `sub_nodes`, `sub_epis` and
graph-level `hierarchy` records. It initializes child pressure to zero,
child capacity to `0.95*nu_parent`, and child phase to the parent's wrapped
phase. Its bounded EPI amplitude uses the existing `0.3` parent scaling
and, when enabled, measured neighboring EPI and circular-phase inputs via
[`compose_subepi_amplitude`](../src/tnfr/operators/metabolism.py).
These are declared implementation coefficients, not newly derived
universal physical constants.

The commit adds a node and hierarchy metadata, **without adding a graph
edge**. The child's identity is an operational nested coordinate, not an
additive conserved EPI mass. THOL does not subtract its amplitude from the
parent or propagate it into neighboring EPI. The legacy propagation flag
is rejected by the canonical birth path; form propagation requires its
own explicit admissible operator composition.

This birth path belongs to public SelfOrganization and its shared staged
implementation. The primitive THOL node-protocol dispatch implements the
pressure action without invoking that hierarchy commit. An ordinary
runtime selection of the primitive glyph is therefore not, by itself,
evidence that new child nodes were generated.

The default pressure law in
[`dnfr.py`](../src/tnfr/dynamics/dnfr.py) reads actual graph neighbors,
EPI, capacity, phase and local degree. It does not use `parent_node`,
`sub_nodes`, `sub_epis` or `hierarchy` as implicit transport edges or a
child-pressure source. In exact arithmetic, adding these isolated children
therefore leaves every original component's canonical structural pressure
unchanged, provided its triad, edges and pressure configuration are fixed.
The isolated children themselves have zero default neighbor pressure.

On an unchanged unit undirected cycle with `j` isolated children, write
`L_Cn=I-W/2` and `e=w_epi` for the effective channel coefficient. The
EPI-channel nodal flow has block form

$$
\dot x=-\left[
e\operatorname{diag}(\nu_{\rm parent})L_{C_n}\ \oplus\ 0_j
\right]x.
$$

Each child contributes a disconnected zero mode. Its nonzero EPI can
persist at positive capacity because its default pressure is zero, while
the parent-cycle diffusion modes are unchanged. The resulting nonuniform
field across disconnected components is not a localized profile maintained
inside a coupled component. This is the existing graph-diffusion nullspace
identity applied to the actual child support, not a new confinement law.

This is a pressure-kernel statement, not universal invisibility. Node
counts, hierarchy queries, aggregate diagnostics and later selectors may
observe the new nodes; explicitly declared later operations can change
their graph connections or channels. Added nodes may also change a
runtime vectorization path, so exact component independence does not by
itself establish identical binary64 reductions. Any claimed parent
feedback must identify such an actual channel or policy dependency.

## 4. Refresh order determines whether the pressure increment reaches EPI

Let `R(G)` denote the default structural pressure recomputed from the graph
state, and let `p0` be the stored pressure just before THOL. The stored
value can include earlier operator writes and need not equal `R(G)`.
Immediately after THOL the parent pressure is `p0+a*A_hat`, while its
triad and original neighbor support are unchanged.

If the default refresh runs **before** physical integration, it overwrites
the stored value with `R(G_T)`. By the component independence just proved,
this equals `R(G)` on the original component in the exact fixed-scope
model. The direct additive THOL term is absent from that following
refreshed nodal rate. An uncoupled hierarchy birth does not retain it
through the default neighbor-pressure kernel.

If the pressure is instead **held** over a positive interval `h`, the
classical unforced, unclipped branch of the shared nodal integrator gives
the exact reference increment

$$
x_H-x=h\nu(p_0+a\widehat A).
$$

A matched control that holds `p0` over the same interval satisfies

$$
x_H-x_{\rm control}=h\nu a\widehat A.
$$

This difference derives directly from the existing pressure write and
nodal law. It adds neither a force nor an autonomous feedback rule. Relative
to an immediately refreshed reference, the same held increment contains
both earlier pressure mismatch and THOL's contribution:

$$
h\nu\bigl[p_0-R(G)+a\widehat A\bigr].
$$

Those contributions cannot be attributed to THOL as a whole when the
preparation includes other pressure-changing operators.

Once a held interval has changed EPI, a later pressure refresh does not
reverse that accumulated change. On the fixed unit cycle with common
capacity and unchanged phase, two states differing only by `delta_x`
have refreshed pressure difference `-e*L*delta_x`. Thus the transient
operator pressure can leave an ordinary structural EPI perturbation if
it is actually integrated. This surviving perturbation is not evidence
that an isolated child feeds pressure back to its parent or maintains a
localized restoring field.

For several compatible held intervals the finite accounting is

$$
x_N-x_0=\sum_j h_j\nu_jp_j+\sum_j r_j,
$$

where each `p_j` is the pressure actually held by the declared integrator
call and `r_j` records its EPI arithmetic/clipping defect. Named EPI jumps,
if any, must be included separately. Only intervals whose retained pressure
contains the THOL increment include its direct contribution. Refresh
boundaries are consequently part of the causal specification, not a
presentation detail.

## 5. Falsification criteria for a stronger feedback claim

A claim that THOL's hierarchy sustains parent-component structure through
the default pressure law must expose a reproducible dependency beyond the
uncoupled birth just described. Useful discriminating checks are:

- Record the executed glyph and grammar context. A request for THOL that
  falls back to a different admissible operator is not a THOL experiment.
- Keep the parent triad and its actual neighbor support fixed while varying
  only isolated child values. A changed exact default parent pressure would
  contradict the present component-independence claim and identify a
  previously unaccounted pressure input.
- Capture pressure immediately before THOL, after its commit, after every
  refresh and at each integration boundary. An EPI response attributed to
  an erased pressure increment would contradict the held-input accounting.
- Separate histories generated by earlier executor-owned flow from a
  caller-prepared threshold-crossing fixture. The latter tests the admitted
  birth path and its effects, not spontaneous generation of its trigger.
- Compare child creation with suppression by the existing depth boundary
  while retaining the same parent pressure update. This can distinguish
  hierarchy effects from the signed pressure increment itself.
- Identify and exercise any later actual transport edge, parent/child
  aggregation or adaptive-policy dependence before claiming persistent
  feedback. Merely observing a nonzero child count, a global diagnostic
  change or a parent EPI transient does not establish that mechanism.

These checks narrow the next structural question: whether an explicit
canonical interaction between a created hierarchy and the parent network
can maintain and restore a localized pattern while reorganization remains
active. They do not assume that such an interaction already belongs to the
default THOL pressure path.

## 6. Shared pressure preparation and the runtime evidence boundary

The shared
[`_thol_pressure.py`](../src/tnfr/operators/_thol_pressure.py) prepares the
signed pressure proposal and checks it before committing pressure or
acceleration telemetry. Public SelfOrganization and primitive dispatch
use this same arithmetic. Graph-backed primitive THOL reconstructs
acceleration from the shared history reader; graph dispatch prepares it
before creating a cached node adapter and reuses that same-call proposal.
Invalid history or a nonfinite proposal cannot be replaced by stale cached
curvature. A graphless node protocol intentionally retains its explicitly
supplied acceleration input; it supplies no graph-history provenance.

[Pressure-history regressions](../tests/operators/test_thol_pressure_history.py)
exercise graph routes, public/primitive pressure agreement, signed
acceleration, unavailable physical history, stale telemetry, invalid
history, finite-domain failure and graphless compatibility. The existing
[structural-acceleration tests](../tests/operators/test_structural_acceleration.py)
cover unequal physical spacing, legacy precedence, endpoint matching and
read-only diagnostic evaluation. This centralization aligns pressure
semantics without promoting primitive dispatch to public hierarchy birth
or replacing the declared runtime refresh policy.

The finite
[runtime tests](../tests/physics/test_thol_pressure_feedback_runtime.py)
and [benchmark](../benchmarks/thol_pressure_feedback.py) supply the
following discriminating observations with default THOL factors:

- On C8, actual Coherence/Dissonance preparation followed by two refreshed
  physical segments of lengths `0.125` and `0.375` produces three recorded
  EPI samples. The shared physical acceleration is
  `0.012579445174099646`, while the naturally produced cached value is
  `0.008386296782733171`. No poisoned cache is needed to expose the
  difference. Using the shared physical history removes a real mismatch
  between graph-backed primitive and public THOL pressure inputs.
- The stored parent pressure changes from `-0.07935727615962375` to
  `-0.07835623571921803`. Holding it for `h=0.25` yields EPI
  `0.9358019009519275`, versus `0.9355516408418261` in the matched held
  baseline. The exact observed extra EPI is
  `2254142677197/9007199254740992` (approximately `0.0002502601101014301`),
  accounted for by the
  represented-acceleration prediction, pressure arithmetic and Euler
  arithmetic residuals separately. Public and staged refresh controls
  instead recover the baseline pressure and endpoint EPI exactly in this
  finite observation.
- Ordinary runtime execution with the declared target selector consumes
  primitive THOL's pressure before integration and matches the held public
  EPI endpoint. Later ordinary phase evolution is recorded separately; this
  observation does not assert that the complete ordinary runtime map is
  identical to the public operator-plus-integrator control.
- A distinct P2 fixture with explicitly prepared physical history has
  acceleration `0.4` and creates one child at the default birth threshold.
  The child is isolated and no edge is added. Canonical refresh restores
  parent pressures `(-0.03180163034256735, 0.03180163034256735)`, gives the
  child zero pressure, and its EPI remains unchanged under the next shared
  Euler interval. This tests the birth contract and lack of immediate
  parent feedback; the prepared trigger is not a causal emergence result.

These are reproducible engine observations, not laboratory validation or
an autonomous localization theorem. The [operator-grounded birth composition](#thol-acquired-birth-participation)
below supplies actual history and an admitted parent/child interaction. Its
finite response has a distinct source and does not retroactively turn the
earlier prepared-history fixture into acquired evidence. The isolated hierarchy
result supplies no interaction implicitly.

<a id="thol-emission-history-continuation"></a>
## 7. Actual Emission history and a subsequent operator action

The [finite-duration Emission proposal](EMISSION_EPISODES.md#emission-episode-status)
supplies an operator-grounded realization of the two held intervals in section 2.
It remains a declared constitutive lift with supplied durations. Coherence (IL)
and Dissonance (OZ) change stored pressure at the boundary without a form jump;
the shared episode executor records actual physical EPI samples. No acceleration,
physical history or grammar context is inserted as a substitute for execution.

For fixed capacity nu, let the two episode pressures be p1,p2 and durations
h1,h2. In exact held-row arithmetic the history estimate and THOL map give

```text
A = 2*nu*(p2-p1)/(h1+h2),
p_T = p2 + g*A,                       g = THOL_accel > 0,
x_next - x_no_THOL = h3*nu*g*A.
```

Arbitrary binary64 endpoint rounding can separate observed secants from the
declared held rates; the dyadic controls below represent these predictions exactly.
The last equality requires a third admitted, unclipped, held AL episode of
duration h3 in both branches. It is the section 4 identity with actual AL as
the continued action. Here `[g]=[time]/[capacity]`; a numerical gain cannot be
transferred unchanged across arbitrary clock conventions. A is a finite-window
comparison of observed secants. When rates differ, the classical second
derivative at the instantaneous rate change is undefined; each open held
interior instead has zero acceleration. Neither observation replaces the other.

The finite controls use a unit-weight P2 with equal x=1/4, nu=1, phase zero and
p=1/2. Actual IL with retention 1/2 followed by OZ with factor 2 supplies context
and restores p. An AL episode of duration 1/8 ends at 5/16. Boundary IL with
retention 1/2 and OZ with factor m gives p2=m/4, followed by another duration-1/8
AL. With g=1/8 and a third duration-1/8 AL, independent exact predictions are:

| Boundary OZ factor m | Observed A | EPI before THOL | Pressure after THOL | Next EPI with THOL | Next EPI without THOL |
| --- | --- | --- | --- | --- | --- |
| 3/2 | -1 | 23/64 | 1/4 | 25/64 | 26/64 |
| 2 | 0 | 24/64 | 1/2 | 28/64 | 28/64 |
| 5/2 | 1 | 25/64 | 3/4 | 31/64 | 30/64 |

All these endpoints remain inside AL's strict latent chart. The public/staged
controls use tau=1: the strict child threshold is not crossed, while pressure
action remains available. Native primitive execution has no child commit in
any case. [Participation controls](../tests/operators/test_emission_thol_continuation.py)
retain actual grammar, lifecycle, phase, capacity, support and physical history;
the matched control omits THOL rather than resetting its consequence.

**Successor admission is independent.** In the negative row, choosing g=3/8 or
g=1/2 gives p_T=0 or p_T=-1/8. The signed THOL map itself permits this result,
but the positive-rate AL episode cannot continue. Its rejection preserves the
completed THOL effect and does not advance EPI, time or AL count. This is a
domain boundary of the composition, not evidence that THOL is a universally
positive-pressure stabilizer. [Admission controls](../tests/operators/test_emission_thol_admission.py)
also keep insufficient history, stale endpoints and native invocation distinct.

**Episode boundaries carry observational meaning.** In the negative row, split
the second duration-1/8 episode into two independently invoked duration-1/16
episodes. The same piecewise-linear form path ends at 23/64, but its last three
samples now lie entirely in the second held rate. The active estimate becomes
zero; THOL retains pressure 3/8 and the next EPI is 26/64. There is also one more
actual AL occurrence. This differs from internal numerical subdivision, which
retains the whole episode's physical boundary samples in these controls. Equal
form paths do not imply equal observation windows, memory or complete states.

For a fixed rate jump, shrinking h1+h2 changes A and hence the pressure action
unless a separate gain/observation law compensates. This section derives no
such law. The shared detached reader validates its selected samples and final
EPI, but does not authenticate their origin or equality with the live graph
clock; the actual composition supplies that provenance. These results establish
a bounded causal operator composition, not autonomous scheduling, hierarchical
birth, persistent identity or a uniquely selected feedback law. This is one
supplied THOL invocation; its history read creates no at-most-once receipt. The
[research program](RESEARCH_PROGRAM.md#research-current-question) owns the next
missing function across all thirteen operators.

<a id="thol-acquired-birth-participation"></a>
## 8. Acquired history, admitted birth and actual child participation

The [birth/contact controls](../tests/operators/test_emission_birth_contact.py)
extend the previous episode source asymmetrically. After the common first AL
episode, only the parent receives IL with retention 1/2 and OZ with factor 5/2.
The second duration-1/8 AL episode acts on both original nodes. At time 1/4,
parent and old peer have forms x=25/64 and y=3/8, pressures 5/8 and 1/2, and
actual history accelerations 1 and 0. Their samples and grammar are produced
by those operators; no threshold-crossing history is supplied in their place.

The existing [eligible THOL dispatcher](../src/tnfr/operators/self_organization_selection.py)
is invoked with tau=1/2 and gain 1/8, enabling strict preconditions and explicitly
disabling the metabolic amplitude extension. Its fresh all-node relation selects
only the parent and commits one actual birth. Parent pressure becomes 3/4.
The returned parent/child association identifies the new isolated node, whose
pressure is zero, phase zero and capacity the represented value 0.95. In this
particular control, binary64 multiplication of 0.3 by 25/64 rounds exactly to
child form c=15/128. This is not an exact decimal-inheritance theorem for
arbitrary forms or capacities.

Birth supplies one legacy child form sample and an empty own operator context;
it supplies neither an edge, inherited physical samples nor the parent's AL
activation count. The birth record's timestamp is parent operator step 7,
distinct from structural time 1/4. Eligibility chooses from the current state
at a supplied occasion; its invocation and threshold remain declared policies.

The parent already has an old peer. Actual UM, requested with this isolated
child as potential partner, creates the new contact through its admitted U3
and affinity path while preserving the old edge. Phase zero is retained. Its
configured capacity blend and pressure reduction remain real effects: with
reduction 1/4, parent pressure becomes 9/16. Three actual IL actions of retention
1/2 give 9/128, compared with 3/32 in the no-UM branch. Both then satisfy the
unchanged EN pressure ceiling 0.15. No pressure reset or weakened admission
substitutes for those stabilizing actions.

Compare three independently produced branches with the same source and birth:

| Branch | Parent EN inputs | Parent after EN | Old peer after RA |
| --- | --- | --- | --- |
| No contact | Old peer | 99/256 | 195/512 |
| UM contact, ordinary EN | Old peer and child | 363/1024 | 747/2048 |
| Same UM, selected EN | Explicit old-peer tuple | 99/256 | 195/512 |

With EN mixing a=1/4 and RA form mixing b=1/2, the conditional scalar identities
are `x_EN=(1-a)*x+a*mean(inputs)` and `y_RA=(1-b)*y+b*x_EN`. The child's causal
contribution to the old-peer response is therefore `a*b*(c-y)/2=-33/2048`.
The third branch retains UM's support, capacity, phase and pressure effects
but removes the child from EN's admitted read set. This isolates actual child
input from birth count and the other UM effects. The negative sign denotes a
response difference, not loss of validity or an optimization score. Ordinary
EN's incoming source map remains distinct from emitter-ranking telemetry.

Actual RA also acts on capacity: amplification 1/8 raises the peer from 1 to
9/8 in every branch. This is not pure-form RA or equality of complete states
between contact and no-contact. No post-contact continuous law is imported.
The [admission controls](../tests/operators/test_emission_birth_contact_admission.py)
retain short-history and equality abstention, stale-source rejection, missing
child input before contact and UM rollback that preserves completed birth.

The child can also act in its own right after contact. In a separate branch,
keep parent form 25/64 and apply EN on the child from the parent, then RA on the
child. Its forms become 95/512 and 295/1024, its own grammar becomes EN,RA,
and its actual capacity becomes `float(0.95*(1+1/8))`. This records new child
actions, not inherited parent history or a collective-identity theorem.

All these post-source actions occur at the same structural time. Public/staged
EN and RA can leave old physical samples inconsistent with the new form;
those samples do not authorize another current acceleration. The child never
acquires elapsed-time evidence merely by receiving instantaneous operators.
Timing, partner access, inheritance coefficients and later maintenance remain
open. These finite compositions do not prove that thirteen operators generate
every structure or dynamics: such a claim needs a specified state domain,
equivalence, target transformation class and reachability/completeness proof
under the actual grammar and live admission.

<a id="thol-state-derived-access"></a>
## 9. State-derived access after acquired birth

The [birth-association policy](STRUCTURAL_OPERATORS.md#coupling-birth-association-access)
uses the parent's live direct associations to obtain a unique isolated child.
The caller supplies the parent and occasion, but no child identifier. The
policy changes access to UM; it changes neither the preceding AL/THOL source
nor UM's numerical map or grammar. It is an explicit constitutive choice,
not a deduction that ancestry necessarily causes contact.

The [integration controls](../tests/operators/test_birth_access_coupling.py)
reuse section 8's actual IL/OZ/AL history and eligible THOL birth. On its unique
accessible-child domain, deriving the UM request must agree with explicitly
requesting that same child. With the same subsequent IL/EN/RA actions, the
independent prediction remains parent form 363/1024 and old-peer form 747/2048.
No contact and same-contact/peer-only EN instead give 195/512 at the old peer,
retaining the child-input discriminator -33/2048 and RA capacity 9/8.
Equivalence concerns primary state, support and actual grammar effects, not
incidental diagnostic timestamps.

An explicitly initialized unrelated compatible isolate tests the association
boundary; it is not reported as an emerged child. A uniquely associated but
phase-incompatible child must reject even when that other isolate is compatible.
Two actual THOL births test ambiguity before eligibility: repeated supplied
THOL occasions create no autonomous timing or consumed-window rule. An actual
explicit contact with one child removes its isolation and leaves the other
accessible. Re-reading after contact and consistently relabeling the initial
identities distinguish live structural access from cached or name-based choice.

The [independent boundary controls](../tests/operators/test_birth_access_coupling_admission.py)
retain malformed-association errors, legacy missing-index behavior, incoming
support, mandatory grammar and rollback. Supplied association controls establish
the reader's domain; only the actual-source controls establish their executed
birth history. No report authenticates arbitrary supplied genealogy. The new
path removes a caller-supplied partner from this bounded composition while
leaving parent selection, occasions, ambiguous competition, inheritance and
maintenance explicit or open. It does not establish autonomous organization.
