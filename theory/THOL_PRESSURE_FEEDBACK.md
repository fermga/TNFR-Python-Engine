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
an autonomous localization theorem. The next constructive test must first
generate a birth-triggering history through actual canonical dynamics,
then identify an admitted parent/child interaction and measure whether
the resulting connected structure feeds back through the existing nodal
channels. The isolated hierarchy result does not supply that interaction
implicitly.
