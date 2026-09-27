# Primitive phase closure and source selection

Symmetry obstructions, nonlinear orientation, source work and actual event occurrence.

Section numbers are stable locators across this document family.
The [parameter reference](../NODAL_PARAMETER_FOUNDATIONS.md) owns the
reading map; the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
alone owns active tasks. Each result retains its stated model and scope.

## 17. Primitive phase origin: symmetry, retained state and the missing row

**Status.** The supplied response in [section 16](PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean) is not promoted to an
autonomous law. The following admission tests identify what a proposed origin
must retain and reject several tempting replacements. They concern the same
unit prism, repeated triples, fixed unit capacity, positive coefficients
`e=w_epi`, `w=w_phase`, and the regular strict phase chart. Write `k=w/pi`.
The complete repeated EPI rows remain

\[
\dot z=-ez-k\zeta,\qquad \dot\mu=w c(\zeta).
\]

Here `z` observes EPI contrast and `zeta` observes the contrast of primitive
nodal phases. Keeping both does not add a new nodal attribute. The missing
object is a justified evolution of the existing phase state, not another
diagnostic name for its current value.

### 17.1 Existing phase writers do not select the supplied rotating contrast

Every local neighbor set on the repeated prism sees the same phase triple;
the global phasor mean has the same argument. For an ideal simultaneous
ordinary coordination step with uniform gains `k_G,k_L`, put `K=k_G+k_L`.
Inside the common lift, before wrapping,

\[
\theta_i^+=\beta+\eta_i+K\{\pi c(\eta)-\eta_i\},\qquad
\beta^+=\beta+K\pi c(\eta),\quad
\eta^+=(1-K)\eta,\quad \zeta^+=(1-K)\zeta.
\]

Thus this policy can shrink, erase or reverse the contrast; it cannot
continuously turn its orientation. A negative multiplier is a discrete
reversal, not a resolved nonzero-radius rotation. Adaptive scalar gains
change the multiplier without supplying an additional direction, provided
the repeated state and uniform-gain hypotheses remain valid. The nonzero
common shift must be kept. This is an exact ideal per-invocation identity,
not a continuous-time law, future schedule certificate or bit-exact claim
about trigonometric/wrap arithmetic.

| Existing owner | Reusable mechanism | Boundary for the phase-origin question |
| --- | --- | --- |
| [Ordinary phase coordination](../../src/tnfr/dynamics/coordination.py) | Global/local phasor relaxation and scalar gain history | The repeated contrast has the scalar map above; gain adaptation does not select a circulating source |
| [Public/staged Coherence phase proposal](../../src/tnfr/operators/_coherence_stage_kernel.py) | Snapshot phasor relaxation | A common simultaneous coefficient gives the same contrast scaling; ordinary native `apply_glyph(IL)` instead calls the pressure-only primitive |
| [Configured engine phase model](../../src/tnfr/dynamics/phase_evolution.py) | Supplied capacity-as-angular-rate plus gated sine coupling | Uniform free advance changes the common phase only; its sine term is an alignment model, not a derived persistent contrast clock |
| [Optional extended system](../../src/tnfr/dynamics/canonical.py) | Pressure-dependent phase response | Its independent pressure row fails the fresh-pressure comparison already audited; section 17.4 also classifies the fresh-pressure linear comparison |
| [Exact memory elimination](../../src/tnfr/physics/epi_memory.py) | Retain the effect of discarded state and its initial condition | Elimination requires a complete fine law; it cannot select the missing phase row |
| [Joint nodal response](../../src/tnfr/physics/phase_response.py) | Pressure derivative and EPI acceleration from supplied phase/capacity rates | Supplies the correct accounting interface, not those rates' constitutive origin |

In the configured sine comparison with common free advance and positive
coupling, the repeated lifted phase variance dissipates on the strict chart:
pair contributions are proportional to
`-(eta_i-eta_j)*sin(eta_i-eta_j)`, strictly negative for a nonzero gap.
This continuous comparison is distinct from finite Euler steps. Common
phase rotation can coexist with decay of the relative source; an absolute
phase rhythm alone does not maintain EPI contrast. Heterogeneous capacity,
additional state, nonuniform stages and changed support require their own
analysis. These restrictions do not classify the complete operator catalog.

<a id="native-phase-contrast-budget"></a>

**Full-phase extension: the configured coordinator consumes contrast.**
The repeated-triple restriction can be removed for a diameter bound.
Assume one common open-semicircle lift, with width `D<pi`. Every nonempty
local phasor mean and the global mean have representatives inside that
interval; an isolate uses its own phase. With shared finite gains
`k_G,k_L>=0`, `k_G+k_L<=1`, the ideal simultaneous coordinator is

\[
\theta_i^+=(1-k_G-k_L)\theta_i+k_G b+k_L\ell_i.
\]

The global target `b` cancels from pair differences. Hence
`D^+<= (1-k_G)D`, even for all six independent phases. This bounds phase
contrast; it neither excludes transient angular motion nor defines a
continuous clock.
The default enabled adaptive policy clamps its gains between configured
finite endpoints with a positive global floor and sum of maxima below one.
For executable accounting use the exact represented floor `m` read from
`DEFAULTS['PHASE_ADAPT']['kG_min']` and `q=1-m<1`.
Its nominal expression `1/(8*pi^2)` is not an exact identity for the stored
binary64 coefficient. Custom overrides, disabled adaptation and altered
configuration do not automatically inherit that bound.

For a finite sequence let `D_j` be the post-coordinator diameter and
`r_j=max(0,D_before_next-D_j)` the observed increase from **all** intervening
phase writers in compatible lifts. If `epsilon_j>=0` bounds the next map's
represented realization defect, then

\[
D_{j+1}\le qD_j+qr_j+\epsilon_j,\qquad
D_N\le q^N D_0+
 \sum_{j=0}^{N-1}q^{N-1-j}(qr_j+\epsilon_j).
\]

These are comparison observations, not added forcing or an event policy.
No unobserved phase write may be assigned `r_j=0`. Uniform common phase
advance alone contributes no diameter increase. For the ideal no-increase
case, repeated coordination exhausts phase contrast; the finite defect
ledger does not by itself supply a uniform asymptotic binary64 bound.
The interval must remain admissible at every invocation, and its coherent
lift must be verified rather than inferred from wrapped maxima/minima.

A detached coordinator control uses the existing evidence-returning version
to retain the actual gains, local targets, proposals and phase writes.
Its exact rational endpoint-minus-convex-proposal error is measured relative
to captured represented targets, whose interval membership is checked. This
does not certify their transcendental accuracy.
This public evidence is not a sealed full-runtime certificate. The native
runtime calls the legacy coordinator; its invocation is identified from
the existing implementation, without replaying the frozen native experiment.
The result proves neither that every other writer contracts nor that all
capacities remain equal. Positive phase contrast is also insufficient for
positive EPI work. That separate test is the
[joint transfer budget](../TNFR_VARIATIONAL_PRINCIPLE.md#1318-source-sensitivity-joint-work-and-the-passive-transfer-limit).
Controls: [configured diameter and finite ledger](../../tests/physics/test_native_phase_diameter_budget.py).

<a id="native-phase-writer-closure"></a>

**Native writer audit: no ideal contrast renewal in the retained class.**
Operator names alone do not identify an evolution map. Ordinary
`runtime._update_nodes` calls `selectors._apply_glyphs`, then
`operators.apply_glyph`, which dispatches through `GLYPH_OPERATIONS`.
It does not invoke the corresponding public operator class or all-target
stage. In particular, native `_op_IL` contracts stored pressure and leaves
phase unchanged; public `Coherence()` can additionally lock phase.
The following atlas identifies the actual default path between coordinators.

| Owner in chronological order | Primitive phase effect and required boundary |
| --- | --- |
| [Capacity adaptation](../../src/tnfr/dynamics/adaptation.py) | No phase write. Equal represented capacities inside the common rails remain fixed regardless of eligibility: the shared proposal retains its input when the represented neighbor mean equals it |
| [Auxiliary math step, history and automatic REMESH](../../src/tnfr/dynamics/runtime.py) | Auxiliary state has no inverse node-phase projection. Automatic REMESH calls the protected delayed-EPI owner, not the phase/capacity structural-memory interpolation |
| [Validators](../../src/tnfr/validation/graph.py) and [callbacks](../../src/tnfr/utils/callbacks.py) | Built-in validators do not write phase. Arbitrary callbacks receive the mutable graph and are excluded unless their writes are accounted for; suppressed callback errors do not undo partial writes |
| [Fresh pressure and default selection](../../src/tnfr/dynamics/selectors.py) | Fresh default Si with equal positive capacity chooses IL, except forced AL/EN lag branches. These native primitives write no phase, capacity or support |
| [Default integrator](../../src/tnfr/dynamics/integrators.py) | Scalar and NumPy paths advance EPI and time, holding primitive phase and capacity fixed. The extended-wrapper flag alone does not replace this runtime integrator |
| [Canonical clamps](../../src/tnfr/validation/runtime.py) | Already centered phases are retained exactly. Outside `[-math.pi,math.pi)`, represented normalization still needs a coherent lift and a signed error; ideal wrapping preserves circular phase |
| [Next coordinator](../../src/tnfr/dynamics/coordination.py) | The preceding default-gain theorem gives ideal diameter factor at most `q<1` |

The selector statement uses initialized nodes: nonzero EPI or retained
nonempty glyph history. IL is then grammar-admissible and is the first
fallback for a rejected forced candidate. The default selector does not
apply the parametric selector's soft repetition filter. Consequently the
reachable native glyph set is `{IL, AL, EN}` in this class, including when
lag counters force AL before EN. Later entries in the fallback list are
not evidence of their reachability. Fresh Si and the unchanged default
configuration are hypotheses, not facts about arbitrary selectors.

On the retained fixed unit prism, valid built-in steps preserve unit
capacity under these maps, even with different adaptation eligibility.
The former two-product blend could increase an eligible `nu=0.3` by
`2^-54` while leaving an ineligible node fixed. The shared fixed-point guard
now removes that numerical defect, extending represented equal-capacity
preservation beyond the unit input. Nonuniform arithmetic and out-of-rail
clamps keep their separate effects. The
[whole-step integration](../FORCED_SUPPORT_BALANCE.md#34-native-runtime-admission-uses-relaxation-not-the-supplied-sine-clock)
records this extension and finite ordinary-runtime controls. Phase-capable public
operators, custom integrators/selectors, generic callbacks, structural-memory
REMESH and the optimizer/FFT free-advance model are separate execution paths.

In exact arithmetic, fixed support, initialized nodes, unit capacity,
fresh default selection, the default integrator, no additional graph writers,
and the common open-semicircle domain therefore give `r_j=0` between
coordinators. The phase hull is invariant and

\[
D_N\le q^N D_0.
\]

This is a conditional ideal composition theorem, indexed by successful
coordinator events. Infinite event count, physical-time scheduling and
non-Zeno behavior are separate premises. It does not prove repeated
binary64 convergence or disappearance of all EPI patterns: native AL/EN
can still change EPI directly, and held pressure need not be fresh pressure.
The theorem excludes phase-contrast renewal in the declared route, not
every possible TNFR maintenance mechanism.

For binary64, sum the oscillations of all intervening signed phase errors
in compatible lifts into `E_j>=0`. If their ideal maps preserve the hull,
`r_j<=E_j` and the finite bound becomes
`D_(j+1)<=q*D_j+q*E_j+epsilon_j`. A common error contributes zero
oscillation. Neither omitted writers nor unchecked lifts have a zero error
by default. A uniform nonzero error bound gives a possible residual floor,
not convergence to zero. Public/staged IL is also ideally hull-preserving
for coefficients in `[0,1]`, including sequential proposals, but it must
retain its own proposal error: its certified two-neighbor displacement and
displayed mean are rounded independently.

The same interval gives `|g_i|<=D/pi` for the ideal fresh phase pressure.
On six nodes with fixed phase coefficient `w`,
`||w*g||_2<=w*sqrt(6)*D/pi` and
`|w*y^T*g|<=w*sqrt(6)*||y||_2*D/pi`.
Thus bounded form contrast cannot receive persistent nonzero phase-source
work from this ideal shrinking interval. Other source channels and finite
glyph jumps keep their own balances; this is not an all-channel work theorem.

The frozen native record supplies only one coordinator invocation, whose
before/after phases are all zero. It contains neither adjacent coordinator
endpoints nor effective gains/targets/proposals. It therefore supports one
consensus-preservation observation, not a consecutive-cycle error ledger.
An authenticated adjacent pair would suffice for aggregate `r_j`; attribution
to individual writers additionally requires intermediate chronology.
No old history is reconstructed and no frozen Mutation decision is rerun.

Controls: [native selection and dispatch](../../tests/physics/test_native_phase_writer_reachability.py),
[integrator, capacity and clamp boundaries](../../tests/physics/test_native_phase_writer_boundaries.py),
and the preceding diameter/finite-ledger controls. These component checks and
the source audit are not a sealed full-runtime execution certificate.

### 17.2 A form-only equivariant phase map cannot wind around the origin

The actual graph automorphisms simultaneously permute the three indices in
both triangles. In the normalized internal basis, a cyclic permutation and
the exchange of indices 0 and 1 act as

\[
Cz=e^{2\pi i/3}z,\qquad Sz=-\bar z,\qquad SCS=C^{-1}.
\]

The same actions apply to `zeta`. The three reflection axes have angles
`pi/6`, `pi/2`, `5pi/6` modulo `pi`. They are distinct from the three
`c(zeta)=0` source lines in [section 16.4](PHASE_FORM_EXCHANGE.md#164-the-common-source-reveals-nonlinear-threefold-geometry), which have angles `0`, `pi/3`,
`2pi/3`. In particular the common source `c` is invariant under permutations;
it is not an orientation-odd quantity.

Suppose a proposed memoryless phase observation is `zeta=F(mu,z)`, with no
other symmetry-breaking state, and respects these graph actions:
`F(mu,Oz)=O F(mu,z)` for every generated permutation matrix `O`. Assume an
invariant open regular domain and a locally Lipschitz complete reduced
vector field, so local solutions are unique. For any reflection `R`,
`Rz=z` implies `R F(mu,z)=F(mu,z)`. Hence `z_dot` also lies on that
reflection axis. The plane consisting of that axis and arbitrary `mu` is
invariant. Local uniqueness forbids a trajectory from crossing it at a
finite time while it remains in the regular domain.

Consequently any trajectory with `|z|>0` remains in one open sector of
width `pi/3`, or on one mirror ray. Its continuously lifted angle cannot
complete a winding. This excludes the circle of [section 16](PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean) as an autonomous
response of this particular class, even with the moving mean retained.
It does not exclude local angular motion, loops contained inside a sector,
nonunique/nonsmooth models, oriented boundary inputs, independent phase state,
or another support symmetry. Passage through an undefined zero-form phase
leaves the nonzero-form domain. If the equivariant locally Lipschitz law also
extends to `z=0`, then `F(mu,0)=0` makes that set invariant and excludes
finite-time passage through it as well. A discontinuous
label-based choice would leave these hypotheses, rather than derive a
canonical orientation.

At uniform form, equivariance additionally gives `F(mu,0)=0`. If `F` is
differentiable there, its real derivative must commute with both `C` and
`S`, so `D_z F=a(mu) I`. Commutation with rotations alone would allow
`aI+bJ`, where `Jz=iz`; reflection forces `b=0`. Inserting a fixed quarter
turn therefore supplies a handedness absent from this graph/form state.
The polynomial control `F(z)=Re(z^3) Jz` does respect both actions and
allows turning inside sectors, but its angular contribution vanishes on
every mirror axis. This algebraic control is not an installed phase law.

The proof uses the existing [permutation owner](../../src/tnfr/physics/symmetry_sectors.py)
and [equivariance framework](../../src/tnfr/physics/equivariance.py), with the
shared exact prism lift; it does not assume the full nonlinear phase model
has the continuous O(2) symmetry of the isolated linear EPI sector.

### 17.3 Relative oriented area is already available in the joint state

With independent primitive phase contrast, define the observation

\[
\ell=\operatorname{Im}(\bar z\zeta).
\]

Under a common orthogonal basis change, `ell` is multiplied by its
determinant: its magnitude is frame-independent and its sign reverses under
reflection. It is distinct from `c`, graph winding, and the auxiliary
field named chirality. The nodal EPI row gives, for `r=|z|>0`,

\[
r^2\dot\psi=-k\ell,\qquad
\dot\ell=-e\ell+\operatorname{Im}(\bar z\dot\zeta).
\]

The Cartesian angular numerator `Im(conj(z)*z_dot)` and the area balance
remain defined at zero form, where `psi` itself is undefined. The second
identity identifies the missing
contribution: sustaining oriented area requires a phase response to offset
its EPI damping. It is not an independent evolution law or a conserved charge.

The complete reflection-fixed state now requires **both** `z` and `zeta`
to lie on the same axis. For example, at `z=ir`, `zeta=s` with positive
`r,s`, `Re(z_dot)=-k*s` is nonzero although `z` lies on a mirror axis.
The joint state can therefore carry a direction unavailable to an
instantaneous form-only closure. This instantaneous crossing witness
supplies neither a repeated orbit nor the missing phase derivative. In the
prescribed circular control, `ell=-omega*r^2/k`; its sign is inherited
from the supplied phase motion.

An independent mode in the other triangle, a capacity contrast or an actual
retained history vector can also change the state stabilizer. Their evolution
and sustaining work still need justification; a scalar common mean does not
remove the mirror constraint. The existing
[`observe_exact_map_symmetries`](../../src/tnfr/physics/equivariance.py)
already distinguishes support/operator symmetries from the stabilizer of
several supplied fields. Reuse it when admitting such a richer state rather
than inferring symmetry breaking from an angle or adding a second audit API.

### 17.4 Admission conditions for a joint linear response

At a regular uniform equilibrium, a differentiable, permutation-equivariant
joint phase law has a contrast linearization of the form

\[
\dot z=-ez-k\zeta,\qquad \dot\zeta=a z+b\zeta,
\]

where `a,b` are real derivatives of the **still unspecified** law. Symmetry
fixes the matrix form, not either coefficient. These symbols classify a
possible response; they are not new engine settings or inferred physical
constants. For a declared constant linear law the following identities are
exact; for a nonlinear law they concern its linearization only:

\[
\dot\ell=(b-e)\ell,\qquad
\lambda^2+(e-b)\lambda+(ka-eb)=0,\qquad
\ddot z+(e-b)\dot z+(ka-eb)z=0.
\]

Each temporal eigenvalue occurs in both spatial components. The linear
contrast block is asymptotically stable precisely when `b<e` and `ka>eb`.
Its eigenvalues are non-real precisely when `(e+b)^2<4ka`. Nonzero purely
imaginary eigenvalues require both `b=e` and `ka>e^2`; the conditional
angular rate would then be `sqrt(ka-e^2)`. This does not determine a TNFR
clock. At that boundary,

\[
Q=a|z|^2+2e\operatorname{Re}(\bar z\zeta)+k|\zeta|^2
 =k|\zeta+(e/k)z|^2+(a-e^2/k)|z|^2
\]

is positive definite and conserved for the declared linear model. It is a
derived comparison quadratic, not the tetrad energy or a new canonical
Hamiltonian. Neutral linear oscillation is not amplitude selection,
attraction, nonlinear stability or source generation; the complete mean
must still solve `mu_dot=w*c(zeta)` and stay in its allowed band.

There is a direct test against an existing candidate. The optional phase
formula at unit capacity is
`theta_dot=A*sin(pi*p)+B*p+G*kappa_graph*J_phi`, with its configured positive
coefficients. In the hypothetical **fresh-pressure** comparison near uniform
phase, the repeated internal pressure is `-e*z-k*zeta` and the mean-sine
current linearizes to `-zeta`. Put `h=A*pi+B>0` and
`g=G*kappa_graph>0`. Then

\[
a=-he,\quad b=-hk-g,\quad
\operatorname{tr}M=-(e+hk+g),\quad \det M=eg,\quad
\operatorname{disc}M=(e-hk-g)^2+4khe>0.
\]

Both eigenvalues are real and negative. Thus this restricted linear
comparison cannot supply a local oscillatory contrast mode. It complements
the nonlinear one-direction work result in
[the variational audit](../TNFR_VARIATIONAL_PRINCIPLE.md#1314-existing-optional-feedback-pressure-consistency-before-recurrence).
It is not the implemented independent-pressure system: its different
pressure row and previously recorded inconsistency remain unchanged.
No sign reversal or replacement coefficient is installed to manufacture
the missing oscillation.

### 17.5 Memory preserves omitted phase information; it does not derive its law

For the declared constant linear comparison, eliminating `zeta` exactly gives

\[
\zeta(t)=e^{bt}\zeta(0)+a\int_0^t e^{b(t-s)}z(s)\,ds,
\]
\[
\dot z(t)=-ez(t)-k e^{bt}\zeta(0)
          -ka\int_0^t e^{b(t-s)}z(s)\,ds.
\]

This is the same variation-of-constants mechanism as
[derived EPI memory](../DERIVED_EPI_MEMORY.md), applied to a different,
conditionally specified complete law. The initial phase term is required.
Eliminating it or choosing a memory kernel to force a desired rhythm would
replace the model. The existing passive repeated EPI mode instead has
`z(t)=exp(-e*t)z(0)`: its retained lag vectors are parallel, with
`Im(conj(z(t))*z(t-tau))=0` for `0<=tau<=t` within that same passive
evolution, or with consistently continued prehistory. Arbitrary stored
initial history need not satisfy this identity. That passive history alone cannot create the missing
oriented area on this family. This is not a theorem against memory in
other multichannel, nonlinear or changing-support dynamics.

Even full instantaneous nodal agreement does not select the phase row.
At the same `z!=0`, `zeta=0`, both logical completions `zeta_dot=0` and
`zeta_dot=a*z` give `z_dot=-e*z`, but their EPI accelerations differ by
`-k*a*z`. Both respect the graph permutations. The distinction is an exact
underdetermination witness, not a proposed pair of engine laws.

### 17.6 Consequence for the single research queue

A mechanism intended to evade this family's obstruction must retain actual
directional state or a justified
boundary interaction and derive its evolution from the declared fine TNFR
mechanism. An internal angle alone, common phase advance, a scalar
synchronization gain, or passive single-mode history is insufficient on this
family. First establish the actual phase row and its projection, then audit
the oriented-area budget, source/form work, full mean and phase chart. If
the law is differentiable at uniform state, calculate its `a,b` from that
law; do not choose them from a desired period. Nonlinear, finite-amplitude
and hybrid mechanisms need their own tests rather than being rejected by
this local linear classification.

The tetrad remains a required read-out of primitive state. Neither `ell`
nor the auxiliary quadratic is promoted to a selector or added evolution
term. No new production dynamics, trajectory, parameter sweep or second
research queue is introduced. Exact controls and finite configured-map
comparisons have separate owners:

- [Permutation and form-only obstruction controls](../../tests/physics/test_phase_origin_symmetry_scope.py).
- [Existing coordination/phase-model controls](../../tests/physics/test_phase_contrast_coordination_scope.py).
- [Joint-state, spectrum and memory controls](../../tests/physics/test_phase_source_joint_state_scope.py).
- [Current G3 execution gate](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).

<a id="phase-premise-review"></a>

### 17.7 Contact obstructions identify closure premises, not a universal limit

The [two-C5 contact results](../COHERENT_PATTERN_CONTACT.md) use held unit
capacity, fixed reciprocal support, fresh form/phase pressure and the supplied
U3-gated sine phase row. Their continuous causal structure is triangular:
`theta_dot=G(theta)` and `x_dot=F(x,theta)`. At the same initial phases,
changing EPI can change the form response but not the phase equation. Wherever
the phase initial-value problem is unique, its trajectory is consequently
unchanged. Without uniqueness, the set of admitted phase solutions still
does not depend on EPI; an extra solution-selection rule would be another
premise. The phase cutoffs make this distinction relevant. Neither the
tetrad nor refreshed EPI pressure supplies a return term to this phase row.

The negative formation results therefore classify this completion. They do
not show that all TNFR closures prohibit formation. Conversely, conserving
or recovering an already prepared winding does not demonstrate its generation.
Full-support pressure continues to read edges excluded from phase coupling;
an interaction cutoff does not delete structural support.

**Operator admission and continuous interaction are separate choices.** U3
compares circular gaps with a configured ceiling before UM/RA execution.
The continuous comparison additionally uses that ceiling to discard sine
interactions and renormalize the remaining neighbor count. The nodal EPI
identity requires neither that reuse nor the value `pi/2`, the sine profile,
or capacity as free angular rate. Revising a continuous closure need not
silently redefine the operator contract. The
[phase-model owner](../../src/tnfr/dynamics/phase_evolution.py) and
[grammar scope](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) retain those boundaries.

**Positive alignment mobility does not supply potential work.** Consider a
conditional comparison on fixed simple undirected support. Let `a` be a fixed,
bounded, nonnegative, even, `2*pi`-periodic function, continuous or piecewise
continuous with finitely many breakpoints per period. For circular gaps set

\[
S_i=\sum_{j\sim i}a(\theta_j-\theta_i)\sin(\theta_j-\theta_i),\qquad
\dot\theta_i=c(t)+m_i S_i,\quad m_i>0,
\]
\[
\psi_a(\delta)=\int_0^\delta a(s)\sin s\,ds,\qquad
V_a=\sum_{\{i,j\}\in E}\psi_a(\theta_j-\theta_i).
\]

The edge primitive is even, periodic, nonnegative and Lipschitz, so the
potential is independent of chosen phase representatives. It is a potential
of this comparison law, not physical or tetrad energy. For absolutely
continuous solutions satisfying the row almost everywhere, with integrable
rates and common drift and finite mobilities along the retained interval,
edge pairing gives

\[
\frac{dV_a}{dt}=-\sum_i S_i\dot\theta_i
               =-\sum_i m_i S_i^2\le0
\quad\text{almost everywhere}.
\]

At regular points the ambient gradient is `-S`. At a breakpoint level set
the corresponding gap derivative vanishes almost everywhere on that set;
its assigned finite interaction value therefore does
not change the chain-rule balance. Common drift cancels because `sum_i S_i=0`.
The claim concerns admitted continuous solutions, not global existence or
uniqueness, finite Euler nonincrease or binary64 execution. Positive mobility
may depend on EPI, phase or capacity without adding a derivative-of-mobility
term: `V_a` itself has no such dependence. Feedback only through this mobility
cannot replenish its potential, although it can change trajectories and
redistribute supplied contrast. The hard gate uses `a=1` where
`abs(wrap(delta))<=pi/2` and zero outside, with `m_i=1/n_i` when the admitted
count is positive; choose any
positive mobility at an isolate of the admitted support, where `S_i=0`.

Replacing the common free rate by node rates `omega_i` instead gives
`V_a_dot=-sum_i S_i*omega_i-sum_i m_i*S_i^2`. Calling `omega_i=nu_i` does
not derive that additional work or its sign; a capacity law and its joint
budget remain necessary. Making `a` depend on EPI or time also leaves the
fixed-potential class and requires the resulting additional work terms.

**Removing the gate changes the preparation bound.** The preceding
dissipation structure does not preserve every quantitative exclusion.
Keep rings `(0,1,2,3,4)` and `(5,6,7,8,9)` with contacts `(0,5),(1,6)`.
Prepare the recipient uniformly at phase zero and the donor at
`theta_(5+k)=gamma+k*q`, modulo `2*pi`, where `q=2*pi/5` and
`gamma=pi-q/2`. The donor has winding one; the bridge gaps have representatives
`+(pi-q/2)` and `-(pi-q/2)`. Under the explicitly different ungated law
`a=1`, its cosine potential is

\[
V_1(0)=5(1-\cos q)+2+2\cos(q/2)
      =\frac{35-3\sqrt5}{4}
      >Q=10(1-\cos q)=\frac{25-5\sqrt5}{2}.
\]

Here `Q` is the same necessary ring contribution for two strictly acute wound
rings; final bridge contributions are nonnegative. The gated preparation
instead has `V_g(0)=5(1-cos q)+2<Q`. Thus removing the cap removes this
particular insufficient-budget exclusion. It neither proves a formation
trajectory nor selects the ungated law. Both comparisons dissipate their own
potential, and their respective basins and domain boundaries remain separate.
No revised law or trajectory is installed by this exact static comparison.

**Reuse the existing closure tests before adding feedback.** The
[mixed-derivative result](../TNFR_VARIATIONAL_PRINCIPLE.md#131-mixed-derivatives-require-reciprocal-coupling)
already fixes a reciprocal EPI response under its declared block-gradient
premise. The independent potential, auxiliary mobilities and time scale remain
free; [stronger structural constraints](../TNFR_VARIATIONAL_PRINCIPLE.md#1319-structural-closure-tests-exchange-jacobi-and-the-remaining-potential)
do not remove that freedom. A feedback term alone is not a generation result:
the [audited pressure feedback](#174-admission-conditions-for-a-joint-linear-response)
is locally dissipative, while the
[capacity response-slope classification](../CAPACITY_LOCALIZATION_BALANCE.md#the-response-slope-distinguishes-decay-growth-and-mere-freezing)
separates decay, instability and mere cancellation under an independently
supplied form-capacity relation. The
[derived form phase](DERIVED_FORM_PHASE.md#212-a-coupled-amplitude-and-phase-law-derived-from-fine-diffusion)
does inherit an amplitude-dependent phase law from fine transport, but its
contrasts decay; it cannot silently replace primitive phase or establish
maintained identity.

The unresolved object is an independently justified complete constitutive
relation, not an extra diagnostic or a gain chosen to obtain formation.
Before a candidate trajectory, derive its actual mixed response, complete
joint work and mean balance, and local spectrum from that relation. A
candidate intended to generate structure outside the linear regime needs
the corresponding finite-amplitude argument instead. A chosen relation
remains a declared model unless independently selected; the same-state,
different-acceleration controls above expose that distinction. The
[single execution queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next obligation; these admission conditions create no parallel
contact, capacity or phase-law campaign.

## 18. Nonlinear phase response, oriented area and tetrad reuse

Section 17's negative linear eigenvalues do not exclude a nonlinear
instantaneous angular response. This section examines the existing optional
phase formula on the same repeated prism, retaining its common pressure.
The formula and its coefficients remain configured premises. Detached
phase-row evaluations are distinguished from the hypothetical continuous
model that substitutes fresh canonical pressure at every time. The latter
is not the implemented independent-pressure extension.

### 18.1 The complete nonlinear projection retains the common pressure

Let `U=[P/sqrt(2),Q/sqrt(6)]` be the orthonormal three-node lift and
`Pi=I-11^T/3`. Use real two-vectors for the complex coordinates `z,zeta`
when multiplying by `U`. Work in the common regular lift
`max(eta)-min(eta)<pi/2`; node-dependent wrap branches outside it do not
inherit the displayed linear centered phase source. Put

\[
y=U z,\quad \eta=U\zeta,\quad k=w/\pi,\quad
h=-ey-k\eta,\quad q=w c(\eta),\quad p=h+q\mathbf1.
\]

Thus `h` is centered pressure and `q` is its common component, not the
absolute EPI mean. The EPI rows are `y_dot=h`, `mu_dot=q`.
Write the inspected optional phase formula as
`theta_dot=f(p)+g J(eta)`, with
`f(s)=A*sin(pi*s)+B*s` and a held common positive coupling
`g=G*kappa_graph`. Its exact repeated-state projection is

\[
\dot\eta=\Pi f(h+q\mathbf1)+gJ(\eta),\qquad
\dot\beta=\frac13\sum_i f(h_i+q),\qquad
\dot\zeta=U^T\dot\eta.
\]

The sum of the mean-sine current `J` is zero. The pressure mean cannot be
removed before applying the sine. Explicitly,

\[
\Pi f(h+q\mathbf1)=B h+
A\{\sin(\pi q)\Pi\cos(\pi h)+\cos(\pi q)\Pi\sin(\pi h)\}.
\]

Absolute translations of `mu` and `beta` leave these contrast equations
unchanged; their rates still belong to the complete state. With
`det(u,v)=u_1 v_2-u_2 v_1`, the oriented-area row becomes

\[
\dot\ell=-(e+Bk)\ell
 +A\det\{z,U^T\sin(\pi p)\}
 +g\det\{z,U^TJ(\eta)\}.
\]

This is a projection of a supplied formula, not a new law derived from the
nodal product. It identifies nonlinear angular contributions missed by the
linear comparison while retaining their provenance.

### 18.2 Nonzero oriented-area production from initially uniform phase

Take `e=1/2`, `w=1/4`, unit capacity, the inactive capacity-channel weight
`1/4`, and repeat the following triple in the two triangles:

\[
\mu=3/5,\quad y=(1/6,1/3,-1/2),\quad \theta=\pi\mathbf1.
\]

The EPI entries are `(23/30,14/15,1/10)`, all inside `(0,1)`. The phase
contrast, current, common pressure and oriented area initially vanish:
`zeta=0`, `J=0`, `q=0`, `ell=0`. Fresh pressure is
`p=(-1/12,-1/6,1/4)`. Evaluating the existing phase row with its declared
real coefficients `A=1/2`, `B=3/20` gives

\[
\dot\ell=
\frac{5\sqrt6-3\sqrt2-8}{48\sqrt3}>0,\qquad
\dot\beta=\frac{-\sqrt6+3\sqrt2-2}{24}\ne0.
\]

The first numerator is positive because `17>12*sqrt(2)`, or `289>288`
after squaring positive sides. Numerically `ell_dot` is about
`5.78316e-5`; the linear `B*p` term contributes no oriented area here.
The common phase rate is retained even though the initial EPI mean rate is
zero. The nonuniform EPI preparation already breaks some graph symmetries;
this is not spontaneous handedness from a fully symmetric initial state.

Production pressure and phase-row readers reproduce this detached
instantaneous comparison within binary64 rounding. No trajectory, persistent
orbit or consistency of the optional independent-pressure evolution follows.
In particular the future identity `z_dot=-e*z-k*zeta` requires fresh pressure
along the comparison, not only at its initial point.

The small-amplitude origin of the effect is explicit. At `zeta=0`,
the projected linear and cubic sine terms have zero determinant with `z`.
The first possible nonzero term is

\[
\dot\ell=-\frac{A(\pi e)^5}{4320}
 \operatorname{Im}(z^6)+O(|z|^8).
\]

This follows from
`det(z,U^T(Uz)^3)=0` and
`det(z,U^T(Uz)^5)=Im(z^6)/36`.
Thus a linear or cubic truncation misses this particular transverse
production mechanism. It remains part of the configured nonlinear response;
the expansion does not select that response or an angular clock.

### 18.3 The same oriented area is readable through the tetrad

On this fixed unit-distance prism, let `T_G` be the canonical inverse-square
source-to-potential kernel. Directly from the graph distances,

\[
T_G\mathbf1=\tfrac72\mathbf1,\qquad
T_G U_{\rm repeated}=-\tfrac14 U_{\rm repeated}.
\]

For fresh repeated pressure, the one-triangle field projections satisfy

\[
U^T K_\phi=\zeta,\qquad
U^T\Phi_s=\tfrac14(ez+k\zeta),\qquad
\boxed{\ \ell=\frac4e\det(U^T\Phi_s,U^T K_\phi)\ }.
\]

This connects the joint orientation to two existing tetrad read-outs; it
requires no additional field API or telemetry-driven action. It is restricted
to the known support, explicit unit lengths, repeated lift, positive known
`e` and fresh canonical pressure. The remaining tetrad fields retain their
usual diagnostic roles. Neither absolute EPI mean nor absolute common phase
is reconstructed, and no complete-state or generic-graph theorem follows.

Stored pressure must not silently replace fresh pressure. For a repeated
stored-minus-ideal pressure defect `delta`, its internal projection
`delta_z=U^T delta` changes the inferred area to

\[
\ell_{\rm fields}=\ell-\frac1e\det(\delta_z,\zeta).
\]

The shared forcing capture already separates stored-pressure and numerical
kernel defects. Finite field-reader comparisons retain those residuals;
the displayed exact identity is not an equality of separately rounded
binary64 operations.

### 18.4 Production of orientation is not sustained identity

The full nonlinear work balance, including the mean-pressure term, is
centralized in [the variational owner, section 13.15](../TNFR_VARIATIONAL_PRINCIPLE.md#1315-full-repeated-triple-feedback-and-the-mean-work-term).
It proves a sufficient finite-pressure-domain dissipation criterion for the
hypothetical fresh-pressure model. The state in section 18.2 can generate
oriented area while that comparison energy decreases. This separates a
nonlinear angular response from a self-maintained pattern.

The auxiliary harmonic substrate supplies no shortcut: at the current
prepared nonuniform prism phase state its proposed geometric velocity has
no primitive-phase lift, even allowing all six phase velocities. The exact
rank obstruction is in [section 3.8](../TNFR_VARIATIONAL_PRINCIPLE.md#38-prepared-prism-harmonic-geometric-velocity-has-no-phase-lift).
Passive reciprocal exchange likewise transfers energy without supplying
the loss; its scoped balance is retained there.

Controls reuse the existing normalized lift, graph pressure, configured
phase row and tetrad readers:
[nonlinear orientation](../../tests/physics/test_nonlinear_phase_orientation_scope.py),
[tetrad projection and pressure defect](../../tests/physics/test_phase_form_tetrad_area_scope.py),
and [full mean-work balance](../../tests/physics/test_nonlinear_phase_mean_balance_scope.py).
Any proposed maintenance mechanism on this domain must account for sustained
source work as well as angular response; none of these calculations installs
a new phase law or controller. This is an admissibility condition for that
mechanism, not an independent research queue.

## 19. Nonrepeated neighbors and local oriented transfer

The repeated-triple family suppressed one actual geometric input. Retain
the fixed unit prism and unit capacities, but now give its two triangles
independent form and primitive phase. Use the same normalized three-by-two
lift `U`: `x_a=mu_a*1+U z_a`, `theta_a=beta_a*1+U zeta_a`. For `b=1-a`, let

\[
\gamma_{ai}=\arg\left(\sum_{j\ne i}e^{i\theta_{aj}}+e^{i\theta_{bi}}\right),
\qquad r_a=U^T\gamma_a,
\]

in one regular common lift of the phases and neighbor directions. The
canonical source is `(gamma_a-theta_a)/pi`, not an arithmetic mean of
phase differences. With EPI and phase weights `e,w>0`, `k=w/pi`, the nodal
projection is exactly

\[
\dot z_a=-\frac{4e}{3}z_a+\frac e3z_b-k\zeta_a+kr_a,
\qquad
\dot\mu_a=\frac e3(\mu_b-\mu_a)+k(\overline\gamma_a-\beta_a).
\]

Consequently the existing relative area `ell_a=det(z_a,zeta_a)` satisfies

\[
\dot\ell_a=-\frac{4e}{3}\ell_a
 +\frac e3\det(z_b,\zeta_a)+k\det(r_a,\zeta_a)
 +\det(z_a,\dot\zeta_a).
\]

The first boundary term comes from the neighboring form; the second comes
from the actual neighbor-phasor directions. Repeated phases make `r_a=0`
because every node sees the same three phases. Nonrepeated phases need not.
Neither term specifies the missing primitive-phase evolution. If the
existing optional phase row is used as a fresh-pressure comparison, keep
`zeta_a_dot=U^T f(p_a)+g U^T J_a` and its pressure mean inside the sine.
The regional phase mean also has boundary exchange:

\[
\dot\beta_a=\overline{f(p_a)}+
\frac g9\sum_i\sin(\theta_{bi}-\theta_{ai}).
\]

Only the full-graph sine-current sum vanishes. Freezing either regional mean
would change the model. These projections reuse the full-graph normalization
and [regional balance owner](../../src/tnfr/physics/support_transport.py).

**An actual boundary-source witness.** Start with equal uniform positive
EPI in both triangles, and choose `a=pi/6`,
`theta_0=beta+(-a,a,0)`, `theta_1=beta+(-a,a,a)`. The global phase spread is
`pi/3`, strictly within the regular chart. Put
`gamma=atan(1/(3*sqrt(3)))>0`. Triangle zero's neighbor directions are
`beta+(0,0,gamma)`, so its source is `(1/6,-1/6,gamma/pi)`. At this instant,

\[
z_0=z_1=0,\quad \zeta_0=(-\sqrt2a,0),\qquad
\dot\ell_0=-\frac{w\gamma}{3\sqrt3}\ne0,\qquad
\dot\mu_0=\frac{w\gamma}{3\pi}>0.
\]

The area derivative is independent of the phase velocity at this instant,
since `det(z_0,zeta_0_dot)=0`. Replacing the second triangle by the repeated
phase triple makes the area derivative zero. Detached production captures
reproduce the difference and keep stored-pressure defects separate. Because
the initial internal form is zero, its angle is undefined: this proves an
instantaneous nonparallel form response, not a finite rotation. The prepared
phase pattern already contains structure; no creation from a uniform full
state is claimed.

The interaction is real within the declared nodal source, but its ongoing
supply is a separate question. The [full-prism storage theorem](../TNFR_VARIATIONAL_PRINCIPLE.md#1316-full-prism-feedback-boundary-exchange-does-not-remove-all-dissipation)
now bounds all six independent phase/form directions in a declared domain;
the [local nonlinear result](../TNFR_VARIATIONAL_PRINCIPLE.md#1317-all-spatial-modes-and-local-nonlinear-attraction)
also excludes a missing small-amplitude oscillatory mode. Thus a regional
change must be accounted for together with its environment. An instantaneous
boundary contribution is not evidence of indefinite replenishment.
Controls: [regional area, means and actual source](../../tests/physics/test_phase_form_boundary_exchange_scope.py).

## 20. Phase-reset source work and actual occurrence

The existing Mutation proposal provides a concrete phase map to audit.
Its default branch proposes `theta_k^+=theta_k+s` modulo `2*pi`, where
`s=factor*copysign(1,p_k_stored)*pi/4`; the default factor yields the represented
shift `|s|=1/4`. For nonzero pressure this uses its sign; the implementation's
signed-zero behavior is an additional map convention, not `sign(0)=0`.
The stored sign need not match fresh pressure. The detached control below
stores the fresh pressure before obtaining its proposal.
This is a configured operator map, not a derived occurrence law.
An admitted Mutation retains EPI, capacity, support and stored pressure.
Consequently it has **zero instantaneous EPI-energy jump** and leaves the
stored nodal rate unchanged. Its potentially useful effect is a different
phase source at a later pressure refresh. The proposal alone certifies
neither temporal eligibility, grammar, selection nor that refresh.

### 20.1 Exact source-work criterion for a finite phase proposal

Retain fixed unit capacity and the prism's pressure coefficients. For a
fixed symmetric quadratic observable `S=x^T Q x`, let `b=Qx`. On a regular
phase path `theta(t)=theta+t*s*e_k`, `0<=t<=1`, the shared source derivative
`Dg_phi=(R-I)/pi` gives

\[
\Delta\dot S_{\rm phase}
=\frac{2ws}{\pi}\left[
 \sum_{i\sim k}b_i\int_0^1R_{ik}(\theta+tse_k)\,dt-b_k\right].
\]

The path parameter is not physical time. The formula requires nonzero
neighbor resultants and a fixed wrap branch along that path. Equivalently,
each affected resultant changes exactly by
`Z_i^+=Z_i+(exp(i*s)-1)*exp(i*theta_k)`; the finite phase source must use
those actual directions, not an arithmetic midpoint or linearized mean.
If the path changes branch, its jump must be accounted for separately.
For Dirichlet energy `E_D=x^T Bx/2`, use `b=Bx` and omit the factor two.
Dirichlet rate, internal-amplitude rate and energy jump are distinct quantities.

At initial phase consensus, each of the target's three neighbors changes
its mean direction by

\[
a(s)=\arg(2+e^{is}),\qquad
a'(s)=\frac{1+2\cos s}{5+4\cos s}.
\]

For `0<s<=1/4`, `0<a(s)<s/3`. If the target is a positive fresh-pressure
maximum under pure EPI pressure `p=-e Bx/3`, then `(Bx)_k` is a negative
minimum, and

\[
\Delta\dot E_{D,\rm phase}
=\frac w\pi\left[-s(Bx)_k+a(s)\sum_{i\sim k}(Bx)_i\right]>0.
\]

This is a positive increment of a prospective rate, not proof that its total
rate becomes nonnegative. For the unchanged shared preparation
`x=(3/4,1/4,1/2)` on each triangle, phase zero, `e=1/2`, `w=1/4`, and one
default proposal at the first positive-pressure maximum, exact-real accounting
gives

\[
\Delta\dot E_D=\frac{3}{64\pi},\qquad
\Delta\dot S=\frac{1}{32\pi},\qquad
\dot S^+=-\frac14+\frac{1}{32\pi}<0,
\]

where `S=sum_a ||x_a-mean(x_a)||^2`. The proposal improves the amplitude rate
but still does not hold this shape. Tests evaluate the pure native proposal
and its copied phase-only endpoint through existing forcing and regional
readers. They keep kernel arithmetic and stale-pressure defects distinct;
they do not apply a Mutation operator or manufacture event history.
Controls: [finite proposal work](../../tests/physics/test_mutation_source_work_scope.py).

### 20.2 Eligibility, selection and the missing sustaining law

Mutation's signed sampled growth gate uses actual history, not the current
product `nu*p`. U4b context and final selector decisions are additional
requirements. The [default-policy reachability result](../DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#uniform-capacity-and-default-selector-reachability)
shows that uniform positive capacities and fresh default Si force the default
base choice to Coherence, even when Mutation's growth and grammar gates pass.
Increasing source work in a copied proposal cannot bypass this obstruction.

This yields three separate questions: can the state admit Mutation, does the
actual policy select it, and would its subsequent fresh source offset loss?
Neither a favorable proposal nor an admissible operator label answers all
three. The finite native control is recorded in the single execution plan;
it does not promote the configured Si consumer to an emergent physical law.
An absent Mutation also does not imply absent phase evolution in general:
Coherence and the later phase-coordination step have their own phase maps.

### 20.3 Geometric admissibility does not require a reset

On this fixed support, consider only regular phasor geometry and the U3
edge condition. In a regular chart with signed difference
`delta_ij=theta_i-theta_j`, the active boundary `|delta_ij|=pi/2` gives the
first-order necessary viability condition
`delta_ij*(omega_i-omega_j)<=0`. It restricts a supplied phase velocity;
when equality holds, that condition alone is not a finite-time viability
proof. At a strict interior state, every fixed finite velocity is admissible
for a sufficiently short interval by continuity.

There is a stronger exact non-necessity control: held phases and arbitrary
common rotations preserve every phase difference and every neighbor-resultant
magnitude for finite time. Neither regularity nor U3 alone therefore requires
a nonzero phase reset, selects its magnitude, or fixes an event time.
A zero neighbor resultant instead makes its direction undefined; evaluating
that undefined direction cannot supply a unique continuation. A separate
continuation rule and evidence would be required. No crossing is simulated
or asserted here, and additional coupled constraints could change the problem.

Thus the failure of the tested configured mechanism cannot be repaired by
calling its reset geometrically inevitable. The current constitutive gap is
a justified joint phase/event law; another diagnostic threshold or supplied
operator sequence would not close it.
