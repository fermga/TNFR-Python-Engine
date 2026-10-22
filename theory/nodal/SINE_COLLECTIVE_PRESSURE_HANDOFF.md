# Collective IL/OZ actions and constitutive-pressure handoff

This owner gives exact-real conditional identities, a simultaneous-IL
maintenance theorem and a [frozen represented handoff](#collective-pressure-handoff-result)
on the actual maintained count-three contact. The executed word retains the
joint identity and exhibits the predicted OZ refresh equivalence. Its frozen
per-target IL phase-error certificate was unresolved in the original report;
a [separate retained-input review](#collective-pressure-phase-review) now
certifies the original error allowance. Direct endpoint and work checks have
independent support. The
[research program](../RESEARCH_PROGRAM.md#research-current-question)
alone selects further work; the result and its accuracy distinction remain here.
No additional pressure law, event selector or research queue is introduced.

<a id="collective-pressure-handoff-model"></a>
## 1. Consumed state and the selected complete law

Retain the actual twenty-node, 45-edge contact of
[SINE_PROXIMITY_CONTACT](SINE_PROXIMITY_CONTACT.md#sine-proximity-contact-law):
forty old unit edges, the internal A0 edge and four A0--B0 edges. All actual
weights remain attached to their endpoints and held fixed. The capacities
are `nu=1/2` on the two A0 targets and one elsewhere; bare degrees are
seven on A0, six on B0 and four elsewhere. Use signed real form, primitive
circular phase with its declared lifts, no forcing or continuous loss,
and the original structural clock.

Write the consumed continuous state as `y=(x,theta,nu,U,W)`, and put

\[
S_i(\theta)=\sum_{j\sim i}\sin(\theta_j-\theta_i),\qquad
\Phi_i(y)=\frac{S_i(\theta)}{\pi d_i},\qquad
M=\operatorname{diag}\frac{\nu_i}{\pi d_i}.
\]

The law is independently specified by

\[
\dot x=MS(\theta),\qquad \dot\theta=ML_Wx.
\]

The stored pressure `p_store` consumed by operators is a distinct channel.
It need not equal the freshly evaluated constitutive pressure Phi. Keep
execution state k as well: associated histories, grammar and lifecycle
state, timestamps, policy inputs, random progress and required telemetry.
Thus an operator acts on more than y, even when the subsequent continuous
law consumes only y. The [parameter foundation](../NODAL_PARAMETER_FOUNDATIONS.md)
and [form foundation](../FUNDAMENTAL_THEORY.md#24-physical-concepts-mathematical-types-and-implementation)
own this distinction; no pressure is reconstructed from a measured form rate.

The continuous storage and mean weights are

\[
H(y)=\frac12x^TL_Wx+\sum_{\{i,j\}\in U}(1-\cos(\theta_j-\theta_i)),
\qquad \rho_i=\frac{d_i}{\nu_i}.
\]

Here `rho=14` on A0, six on B0 and four elsewhere, with totals
`rho_A=60,rho_B=44,rho_all=104`. H and both global rho-weighted
form and lifted-phase means are conserved between events. Stored pressure
and execution state do not occur in this particular H or these flow rows.

Use the same common-aligned twist reference theta_* and joint observation

\[
Z^2=\|P_{20}x\|^2+\|P_{20}(\theta-\theta_*)\|^2,\qquad r=1/16.
\]

The proved [storage barrier](SINE_PROXIMITY_CONTACT.md#sine-proximity-contact-barrier)
is `H-H_*>=Z^2/25` on `Z<=r`, with boundary value `1/6400`.
The maintained source and its
[independent-registration extension](SINE_PROXIMITY_CONTACT.md#sine-registration-maintenance)
have `Z<r` and strict storage excess below that value. Those hypotheses,
not a diagnostic coherence label, are used below.

<a id="collective-pressure-refresh-equivalence"></a>
## 2. Pressure refresh and the exact OZ projection equivalence

Let `pi_c(y,p_store,k)=y` denote projection onto the consumed continuous
state. An admitted canonical OZ action, including its ordinary propagation
or noise branch when enabled and finite, changes stored pressure and
execution context while preserving x, phase, capacity, support and weights.
Consequently

\[
\pi_c\circ J_{\rm OZ}=\pi_c,\qquad
\Phi(\pi_c J_{\rm OZ}z)=\Phi(\pi_c z).
\]

An immediate refresh that evaluates this specified Phi therefore assigns
the same constitutive pressure with or without that OZ action. The complete
fixed-law continuation from the projected state is identical in both cases
for all time, provided no further events, controllers or inputs consume the
changed execution state. OZ also has `Delta H=0` in this model because
H does not contain stored pressure.

These are projection and continuation identities, not equality of the full
operator states. OZ can leave different grammar debt, lineage, history,
propagation records, random progress or diagnostic flags. Those differences
can affect later eligibility or execution. Removing OZ from a word requires
a separate argument about those later consumers; it does not follow from
its continuous-state neutrality.

The [OZ mechanism](../STRUCTURAL_OPERATORS.md#71-dissonance-oz) includes
multiplicative pressure amplification, a nonzero near-zero seed, optional
jitter and signed accumulation of neighbor contributions. The default gain
`(pi+1)/pi` is a supplied factor. The local amplification and final merged
pressure are different observations; incoming positive contributions can
cancel negative pressure. None of these branches supplies a pressure
lifetime or a retained forcing term in the selected continuous model.

The mathematical projection above is not a new runtime operator. A detached
weighted-field report evaluates Phi without mutating stored pressure. An
executed refresh must use its declared callback boundary and selected-law
writer; the default mixed-pressure callback is not this law. A held-pressure
interval or an additional pressure-memory/input model would have different
premises and does not inherit this equivalence automatically.

<a id="collective-il-full-action"></a>
## 3. The full simultaneous IL action

The [shared IL kernel](../../src/tnfr/operators/_coherence_stage_kernel.py)
contracts each selected stored pressure by its admitted retention factor.
The ordinary default is `pi/(pi+1)`. Unlike the pressure-only glyph,
the public class and immutable snapshot stage also propose a phase change.
For target i, define its unweighted old-neighbor resultant

\[
\mathcal R_i=\sum_{j\sim i}e^{i\theta_j}=R_i e^{i\mu_i},\qquad
 a_i=\operatorname{wrap}(\mu_i-\theta_i),\qquad d_i^{\rm IL}=\lambda a_i.
\]

The event writes `theta_i'=theta_i+d_i^IL` modulo `2*pi` and preserves
x, nu, U and W. The admitted phase coefficient lies in `[0,1]`; its
ordinary implementation default is `0.3`. The theorem below covers this
entire interval and selects no new coefficient. Both A0 targets read the
same immutable snapshot. Neither proposal observes the other's new phase.
The kernel also has declared isolated/zero-resultant fallbacks, although
the maintained contact domain below excludes those cases locally.

Storage, identity and mean calculations retain the continuous endpoint
lift `theta_i+d_i^IL`. A wrapped storage representative near `2*pi` must
not be mistaken for a large physical displacement from a nearby zero lift.

The default IL retention and OZ amplification are reciprocals only in
their ideal scalar multiplicative branches. They are not inverse hybrid
actions: IL can change phase, OZ has near-zero/propagation/noise branches,
both change execution context, and a declared refresh can intervene.
Even cancellation of two stored-pressure factors would not reverse IL's
phase reset or erase its history.

The actual staged word still needs live input and grammar admission.
This section defines the primary map conditional on admission; it does
not claim that every point in a mathematical identity neighborhood has
the history, stored pressure or active runtime settings required to execute
that word. Represented trigonometric rounding is also distinct from the
exact-real event considered in the proof.

<a id="collective-il-local-resultants"></a>
## 4. Uniform local resultant and phase-displacement bounds

Let u and v be the adjacent A0 targets. Each has its partner, four old A1/A4
neighbors and two B0 neighbors. At the common-aligned reference, its
neighbor resultant is

\[
R_*=3+4\cos(2\pi/5)=2+\sqrt5.
\]

This is a local resultant. The whole doubled-twist reference has zero
first circular resultant, so it does not define a global resultant angle.
The proof instead uses the already declared continuous lifts.

For a state in `Z<r`, write `theta=theta_*+c*1+eta`, where
`eta=P_20(theta-theta_*)` and `||eta||<r`. On either seven-neighbor
set, the elementary bound `|exp(i*t)-1|<=|t|` gives

\[
|e^{-ic}\mathcal R_i-R_*|
\le\sum_{j\sim i}|\eta_j|
\le\sqrt7\,\|\eta\|<\frac{\sqrt7}{16}<\frac16.
\]

Since `sqrt(5)>11/5`, the real part of `exp(-i*c)*Rcal_i` exceeds
`2+11/5-1/6>4`, and its imaginary part has magnitude below `1/6`.
Hence

\[
R_i>4,\qquad
|\mu_i-c|<\arctan(1/24)<1/24
\]

on the unique local direction near c. Each target's reference phase is
zero, so `|theta_i-c|<1/16`. The shortest target displacement is unique
and satisfies

\[
\boxed{|a_i|<\frac1{16}+\frac1{24}=\frac5{48}<\frac18.}
\]

These bounds require neither equal target members nor equal boundary
conductances. The phase mechanism uses bare neighbors; its local
resultant is not weighted by the actual conductances.

<a id="collective-il-adjacent-work"></a>
## 5. Exact shared-edge correction for two simultaneous targets

For clarity put `d_u=d_u^IL,d_v=d_v^IL` in this section, and write
`delta=theta_v-theta_u`. If only target i moved with its old neighbors
fixed, its incident phase-storage change would be

\[
G_i=R_i\left[\cos a_i-\cos((1-\lambda)a_i)\right].
\]

Adding the two such expressions counts the shared edge as two independent
single-target changes. The true simultaneous storage jump is instead

\[
\Delta H=G_u+G_v+C_{uv},
\]

\[
\begin{aligned}
C_{uv}={}&\cos(\delta-d_u)+\cos(\delta+d_v)
          -\cos\delta-\cos(\delta+d_v-d_u)\\
={}&-4\cos\left(\delta+\frac{d_v-d_u}{2}\right)
       \sin\frac{d_u}{2}\sin\frac{d_v}{2}.
\end{aligned}
\]

All other edges are counted correctly in `G_u+G_v`; edges not touching
u or v do not change. The form part of H is unchanged. The correction
has coefficient one because the phase potential uses bare support, even
though the internal edge's form conductance is the actual omega.

The correction has no general sign, but always obeys

\[
|C_{uv}|\le|d_ud_v|\le\frac12(d_u^2+d_v^2).
\]

For example, on a graph consisting of one edge, full snapshot alignment
with `lambda=1` exchanges its two distinct phases. The true phase storage
is unchanged although each separately evaluated single-target cost is
negative. The correction supplies exactly the missing positive amount.
Thus the existing single-target IL inequality cannot simply be summed
for adjacent targets.

For the maintained contact, the local bounds in Section 4 give a uniform
negative allowance that controls this correction. On `0<=s<=1/8`,
`sin(s)>=3*s/4`, since `cos(s)>=1-s^2/2>3/4`. For `0<lambda<=1`,

\[
\begin{aligned}
G_i
&=-R_i\int_{(1-\lambda)|a_i|}^{|a_i|}\sin s\,ds\\
&\le-\frac{3R_i}{8}(2\lambda-\lambda^2)a_i^2\\
&\le-\frac32\left(\frac2\lambda-1\right)d_i^2.
\end{aligned}
\]

For nonzero d_i the last bound is strict because `R_i>4`.
Combining both targets with the shared-edge correction yields

\[
\boxed{\Delta H\le-(d_u^2+d_v^2)\le0.}
\]

The first inequality is strict whenever at least one displacement is
nonzero. If both vanish, the phase-storage change is zero. At `lambda=0`
the primary phase map is the identity and the storage jump is exactly
zero, while the stored-pressure and execution-context changes may remain.
This proves conditional simultaneous passivity on this contact domain;
it is not a universal theorem for arbitrary support or simultaneous IL
stages, nor a strict-response lower bound for every admitted source.

<a id="collective-il-identity-retention"></a>
## 6. The reset path preserves the existing joint identity

A nonpositive endpoint storage jump alone would not exclude jumping to
a different geometric neighborhood. The actual finite phase displacement
also has a continuous comparison path

\[
\theta(s)=\theta+s d^{\rm IL},\qquad 0\le s\le1,
\]

with x, nu, U and W fixed. Every prefix uses the same snapshot resultants
and equals the ideal simultaneous formula with coefficient `s*lambda`.
The preceding work identity and estimate therefore give

\[
H(x,\theta(s))-H(x,\theta)\le
 -s^2(d_u^2+d_v^2)\le0.
\]

This inequality is computed from the initial snapshot and does not assume
that the whole path already lies inside the identity radius. Suppose the
prestate has `Z<r` and `H-H_*<1/6400`, as every state on the proved
maintained continuation does. If the reset path first reached `Z=r`,
the existing boundary barrier would require `H-H_*>=1/6400` there,
contradicting its nonincreasing storage. Thus the entire path and the
actual endpoint retain `Z<r` and the strict storage margin.

Restarting the same held weighted law from that endpoint preserves the
identity for all subsequent structural time by the existing conservative
barrier proof. An admitted OZ pressure-only action and the selected-law
refresh do not alter this conclusion because they leave y fixed. This
argument supplies no occurrence schedule or phase-flow law for IL: the
path is a geometric certificate for a supplied jump, not an asserted
finite-duration dynamics or an interpolation executed by the engine.

<a id="collective-il-mean-and-rate-jumps"></a>
## 7. Mean leaves, fresh rates and sufficient collective information

IL leaves the global weighted form mean unchanged. Its global lifted-phase
mean changes by

\[
\Delta\mu_\theta=\frac{14(d_u+d_v)}{104}
=\frac7{52}(d_u+d_v).
\]

Using the joined regional phase observations, the jumps are
`Delta Theta_A=7*(d_u+d_v)/30` and `Delta Theta_B=0`.
The post-event continuous flow conserves these new means; it need not
retain the pre-event phase-mean leaf. Translating the common reference
phase to match the new global mean changes neither H nor Z.

The fresh continuous rows have distinct instantaneous behavior:

\[
\Delta(\dot\theta)=0,\qquad
\Delta(\dot x)=M\left[S(\theta+d^{\rm IL})-S(\theta)\right].
\]

The first identity follows from unchanged x, W and nu. The second is the
specified constitutive recomputation, not multiplication of the old form
rate by IL's stored-pressure retention. Fresh pressure can change at the
targets and their neighbors; OZ has zero changes in both fresh rows when
y is unchanged. A nodal product built from stale stored pressure is a
different observation and cannot replace these evaluated rows.

For the two IL phase proposals, the common external phasor from the six
neighbors other than u and v, together with both target phases, determines
both local resultants. That small description suffices for these primary
phase proposals and their incident phase work. It does not replace the
full fine state needed for subsequent motion, the separate A/B boundary
associations needed for regional exchange, or the retained stored pressure
and execution context needed for admission.

Any relabeling symmetry must transport target selection, node state,
conductance endpoints, histories and rank-dependent policies together.
A snapshot-equivariant primary formula does not create a pair-symmetric
continuous quotient for generic actual weights. Neither a common phase
mean nor an unordered multiset of independent histories proves descent
of the complete word.

<a id="collective-pressure-memory-boundary"></a>
## 8. Reuse and memory boundaries

The [complete reset accounting](RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
and [identity-retention method](SINE_PAIR_INTERACTION.md#sine-pair-reset-retention)
provide the useful pattern: retain actual event work, sufficient endpoint
state and an applicable barrier. The present proof re-admits their method
on this support and corrects its overlapping IL edge explicitly. It does
not transfer coefficients from an unweighted or dissipative comparison.

The [memory owner](../DERIVED_EPI_MEMORY.md),
[nonlinear constituent memory](ORGANIZED_CONSTITUENT_MEMORY.md) and
[ontology distinction](../EMERGENT_ONTOLOGY.md#collective-memory-and-operators)
separate eliminated dynamical state, supplied delayed history and operator
execution context. Hidden fine initialization can affect a reduced future,
and execution history can affect a later operator. Neither fact supplies
an independent lifetime for an overwritten stored-pressure intervention.
A history buffer or reciprocal scalar factor is not a realization of a
derived memory kernel.

Even differentiating the selected constitutive pressure produces only a
consequence on its own state constraint. With `D_b=diag(d_i)` and the
bare phase Hessian `K(theta)=L_(cos(theta_j-theta_i))`,

\[
\dot\Phi=-\frac1\pi D_b^{-1}K(\theta)ML_Wx.
\]

This follows by differentiating `Phi=D_b^{-1}S/pi` under the already
specified complete law. It does not prescribe an independent pressure
state away from `p=Phi(y)`, authorize an arbitrary pressure reset there,
or add a memory/source law. Any such extension needs its own hypotheses,
initial-state admission and discrimination.

If a later study explicitly declares an interval that consumes stored
pressure instead, put `r_p=p_store-Phi(y)` and retain the same phase row.
The hypothetical form row and its exact storage balance would be

\[
\dot x=\operatorname{diag}(\nu)(\Phi+r_p),\qquad
\dot H=(L_Wx)^T\operatorname{diag}(\nu)r_p.
\]

Indeed, the `q^TMS` and `-S^TMq` terms still cancel, leaving precisely
the displayed additional work. This identity specifies no law, duration
or reservoir for r_p; even held stored pressure need not give a constant
deviation from the evolving Phi. The
[THOL refresh analysis](../THOL_PRESSURE_FEEDBACK.md#4-refresh-order-determines-whether-the-pressure-increment-reaches-epi)
and [held-pressure IL accounting](../DERIVED_EPI_MEMORY.md#116-conditional-paired-evolution-under-common-coherence)
explain why retention order matters. Their models and coefficients do not
transfer here. The present gate introduces no such pressure-retention interval.

<a id="collective-pressure-preflight-scope"></a>
## 9. Scope of the preparatory theorem

The exact-real preparatory result establishes a model-specific OZ
projection equivalence and a uniform simultaneous-IL work/identity theorem
for the two existing A0 targets. It retains the complete IL phase action,
the adjacent-edge correction, actual conductance associations and changed
mean leaf. The whole allowed phase-coefficient interval is covered without
choosing a coefficient to fit a response.

No IL/OZ word or trajectory was executed for this derivation.
A concrete study separately needs its frozen preparation, actual live operator
and grammar admission, declared selected-law refresh writer and callback
schedule, retained context and represented output checks. These are separate
obligations of the [event contract](../../docs/contracts/OPERATOR_EVENTS.md),
not consequences of the storage inequality. The
[retained execution](#collective-pressure-handoff-result) supplies one
bounded case with an explicitly unresolved phase-accuracy certificate.
A rejected operation remains
a rejection, and a detached field evaluation does not perform a refresh.

The maintained organization may realize these conditional actions while
other operators, autonomous event selection, physical calibration and a
uniquely selected fundamental law remain open. No additional pressure
mechanism is inserted merely to make OZ change this particular continuation.

<a id="collective-pressure-execution-preflight"></a>
## 10. Execution and evidence preflight

The following design assembles the owners used by the bounded study. The
protocol froze its configuration, observations, arithmetic allowance and
acceptance criteria before response evaluation; the resulting evidence is
reported separately in section 13.

### Source, word and actual stage boundary

Reuse the positive pure-phase preparation from the
[existing registration protocol](SINE_PROXIMITY_CONTACT.md#sine-registration-protocol):
`a=0,g=3/4096`, nominal split source, unchanged count-three contact word,
seed, timestamps and histories. Its existing
[`_registered_preparation((0, 1))`](../../tests/operators/test_collective_organization_contact.py)
returns both detached observations and the full live runtime graph. Preserve
the cached original and use a working copy of the latter. The detached graph
used for field reports intentionally omits execution configuration; it is
not an equivalent starting point for an operator word.
The implemented copy uses the shared SDK
[`copy_graph_state`](../../src/tnfr/sdk/_state.py), with a bounded adapter
retaining graph-instance event state and original neighbor order. It explicitly
rebuilds registered runtime caches; a generic deepcopy of their locks is not
this copy contract. Unknown instance state rejects instead of disappearing.

Use `coherence,dissonance` on the existing ordered targets `(0,1)`,
`ValidatedSequence`, truthful initialized-EPI context and the canonical
`core` grammar profile with strict live rejection. Read executable tokens
from the registry-backed `TNFR.operators()` interface. U1 admits this start
only with its initialized context; terminal OZ requires the prior IL.
The U2 debt and U4 handler checks still consume the actual histories.
No AL/SHA padding, diagnostic waiver or replacement glyph is needed merely
to validate the proposed word. Static word validity does not waive live
source, branch or threshold admission.

The shared [`execute_network_operator_stage`](../../src/tnfr/operators/word_execution.py)
routes IL to immutable pointwise proposals and OZ to its all-target
local/propagated pressure reduction. It does not forward arbitrary operator
keyword arguments. Freeze its existing IL phase coefficient `0.3` explicitly;
the full interval theorem above already covers it. Use `OZ_NOISE_MODE=False`,
`OZ_ENABLE_PROPAGATION=True`, existing pressure gains and the default
`phase_weighted` propagation mode. Resolve effective factors and
Boolean settings before retaining their values. No dispatcher extension or
parallel operator implementation is required for this design.

The `uniform` propagation name currently follows the same formula as
`phase_weighted`; it is not an independent control law. The actual local
proposal and the final overlapping neighbor reduction must remain distinct.
Retain any unavailable acceleration observation as unavailable, with its
history/clock convention; neither an OZ label nor a pressure change establishes
a bifurcation or an observed second time derivative.

Use the typed `observe_structural_acceleration` observation from
[`nodal_equation.py`](../../src/tnfr/operators/nodal_equation.py): glyph history
and emission timestamps are not three EPI samples. The compatibility
`compute_d2epi_dt2` wrapper can return zero when that observation is unavailable,
and OZ metrics still consume that scalar. Such a telemetry zero must not be
reported as a measured zero acceleration or evidence against a bifurcation.

Shared admission rejects invalid primitives and nonzero inputs lost during
materialization. It does not certify every subsequent floating-point product:
propagation still evaluates products in float arithmetic, where nonzero
increments can underflow. The bounded study must check its actual propagation
arithmetic and residuals; primitive admission alone supplies no general
no-underflow theorem.

### Three declared refreshes, with no intervening flow

Use an initial selected-law refresh before IL, then one after IL and one
after OZ. The contact word's stored pressure is not presumed constitutive.
Capture the full initial state, each raw post-operator/pre-refresh state,
each refreshed state and each completed stage. The final stage capture is
needed because schedule records can be appended after the callback returns.
These are same-time event/refresh boundaries, not three elapsed-time steps.

There is no existing point-pressure writer for this weighted conservative
comparison. Reuse
[`bound_relational_sine_support_exchange`](../../src/tnfr/physics/relational_sine_support.py)
to admit current primitives and enclose the independently specified Phi.
For the bounded static study a narrow callback can materialize each pressure
interval's midpoint through shared represented-real admission, then publish
all admitted values with the shared pressure-alias writer. Retain the exact
represented value and its residual enclosure

\[
\varepsilon_{p,i}\in\operatorname{point}(p_{i,\mathrm{float}})
                         -[\Phi_i(y)].
\]

This materialization is a declared numerical realization, not exact equality
between a float and an ideal sine pressure. Admit every value before any live
write; a nonzero midpoint lost to zero during conversion must reject. When
claiming which ideal OZ branch applies, its pressure enclosure must resolve
the fixed `1e-9` cutoff, with equality in the near-zero branch. The executed
represented branch and an unresolved
ideal branch are separate observations. Do not tune the preparation after
seeing a boundary case.

A direct stage callback is opaque. Its stage transaction does not install
the event-schedule executor's pressure-only guard or sealed refresh evidence.
Check the callback's before/after snapshots explicitly: form, phase, capacity,
support, every edge attribute, histories, clock and other execution state
must be unchanged, apart from admitted pressure aliases/cache updates. Bind
the callback implementation, source configuration, node order and arithmetic
to the retained result. The [event contract](../../docs/contracts/OPERATOR_EVENTS.md#pressure-refreshed-physical-partitions)
provides the preservation boundary, not a claim that a raw hook enforces it.

The whole event-schedule executor selects all nodes and has its own ordinary
flow segments; it is not this two-target, two-row study. Native neighbor-argument
pressure, default mixed pressure and the positive-loss, unit-capacity,
rescaled sine-flow helper also have different premises. None may silently
replace Phi or advance this comparison. No continuous trajectory is needed
to classify these static handoffs.

### Evidence needed to connect the proof to execution

The comparison must retain, separately:

- Actual word/live admission and all consumed context, including changed debt,
  lineage, history and propagation records. Equal continuous state is not equal
  execution state, and independent history multisets lose member associations.
- IL's raw pressure contraction, complete simultaneous phase proposal, exact
  shared-edge work formula, changed mean leaf and represented endpoint defect.
  Re-admit the storage margin and identity on the captured output rather than
  inheriting an exact-real verdict from an operator name.
- Fresh pressure and both complete rate rows before and after each refresh.
  Compare the OZ continuation with the post-IL state under the same law while
  preserving the different execution context. Stored products and instantaneous
  fields are not measured temporal response.
- Source and code revision, complete settings, actual support/conductance,
  order, timestamps, seeds, precision, callbacks and arithmetic residuals.
  Unsupported affine EPI-jump certificates should abstain; their absence is
  not failure of the separate IL phase-work theorem.

<a id="collective-pressure-reuse-map"></a>
## 11. Reused calculations and their transfer boundaries

This is a scoped dependency map for the handoff, not a new catalog or queue.

| Existing owner | Reusable contribution | Boundary retained here |
| --- | --- | --- |
| [Maintained contact](SINE_PROXIMITY_CONTACT.md) | Actual weighted support, complete source, local geometry, storage barrier and regional current identities | Its initial response protocol does not evaluate IL/OZ. Re-admit event work and the new phase mean before continuation |
| [IL kernel](../../src/tnfr/operators/_coherence_stage_kernel.py) and [OZ stage kernel](../../src/tnfr/operators/_dissonance_stage_kernel.py) | Existing snapshot proposals, local pressure branches and overlapping-target reduction | The glyph, public class and full stage have different secondary effects; preserve the chosen route |
| [Weighted field reader](../../src/tnfr/physics/relational_sine_support.py) | Primitive admission, both ideal rate rows and storage with actual conductance | Detached evaluation performs neither a refresh nor numerical evolution |
| [Collective context controls](../../tests/operators/test_collective_grammar_context.py) | Same physical state with different grammar debt or earlier stabilization | Current pressure and regional means cannot replace the context consumed by the next token |
| [THOL refresh accounting](../THOL_PRESSURE_FEEDBACK.md#4-refresh-order-determines-whether-the-pressure-increment-reaches-epi) | Why a held pressure increment can reach form before a later refresh | Its source, default pressure and finite responses do not transfer to this conservative law |
| [Paired common-IL accounting](../DERIVED_EPI_MEMORY.md#116-conditional-paired-evolution-under-common-coherence) | Separation of generated pressure, retention, integration and arithmetic residuals | It requires its own common source and held-pressure interval; it is not the present immediate-refresh map |
| [Constituent memory](ORGANIZED_CONSTITUENT_MEMORY.md) and [linear memory owner](../../src/tnfr/mathematics/linear_observation.py) | Hidden initialization and causal kernels under declared elimination | Neither installs a persistent OZ channel; support, capacity, loss and clock must match before coefficients transfer |

The [independent foundation controls](../../tests/physics/test_collective_pressure_handoff_foundations.py)
check edgewise work, the full-swap counterexample, local resultant bounds and
the all-coefficient inequality with exact interval/polynomial arithmetic.
Their synthetic trigonometric cases are proof controls, not new contact-word
responses. Existing IL/OZ tests separately cover snapshot order, propagation,
invalid values and rollback. The [testing owner](../../TESTING.md) gives the
combined selection; an engine/API gate alone is not this research coverage.

<a id="collective-pressure-handoff-protocol"></a>
## 12. Frozen bounded-word comparison

The [machine-readable protocol](../../docs/assets/collective_pressure_handoff/correction_v2/protocol.json)
fixes one positive phase-axis preparation, the existing count-three contact,
ordered targets `(0,1)`, and `coherence,dissonance` under the selected complete
law. It admits no elapsed flow, parameter search or preparation adjustment.
The initial refresh and both post-stage refreshes use the same constitutive
pressure. The operator word's raw outputs, refreshed outputs and completed
stage records remain distinct observations.

The prospective discriminators are actual admission without replacement,
IL's phase/form-rate change and negative storage work with retained identity,
OZ's raw pressure/context change followed by continuous-state equivalence,
and explicit acceleration unavailability when no temporal EPI evidence exists.
No neighbor propagation is required: the existing `0.05` policy may discard
every candidate. Retain every candidate's arithmetic before classifying that
outcome. A rejected word or unresolved ideal pressure branch remains a result.

The represented phase coefficient is the exact rational represented by
binary64 `0.3`, not `3/10`. For each target, use the report's relative
resultant to enclose the ideal angle `a_i`, and choose the actual endpoint's
continuous lift nearest the ideal displacement `d_i=lambda*a_i`. Keep the
stored modulo angle and lift association separately. With represented lifted
displacement `dhat_i`, the frozen numerical allowance is

\[
 |\widehat d_i-d_i|\le 2^{-46}.
\]

Each target has seven incident phase edges. The phase potential is separately
7-Lipschitz in each target angle, so

\[
 |H_{\rm represented}-H_{\rm ideal}|
 \le7(|\widehat d_u-d_u|+|\widehat d_v-d_v|)
 \le14\,2^{-46}.
\]

This bounds the bridge to the exact IL theorem; it does not itself prove
strictly negative represented work near a zero displacement. Recompute the
whole-edge storage difference and the shared-edge work identity with the
actual lifted displacements. Separately require the represented endpoint's
`Z^2<1/256` and `H-H_*<1/6400`, where
`H_*=40*(1-cos(2*pi/5))`. These are endpoint admissions for the existing
barrier, not observed long-time motion.

Shared outward rational 128-bit intervals enclose pressures, rates and
trigonometric work. Any independent high-precision cross-check uses 100
decimal digits. Pressure materialization and consumed propagation arithmetic
use the fixed absolute allowance `2^-46`; unchanged primitives require exact
represented equality. The callback guard permits only pressure aliases and
the pressure-maximum cache/witness, whose consistency is checked separately.
No whole-cache exemption substitutes for that guard.

<a id="collective-pressure-handoff-result"></a>
<a id="retained-represented-result-and-unresolved-phase-certificate"></a>
## 13. Retained represented result and original certificate limitation

The [response](../../docs/assets/collective_pressure_handoff/correction_v2/response.json)
contains nine full-state boundaries from one admitted IL/OZ word, the three
refreshes, typed observations and both completed stages. The
[frozen report](../../docs/assets/collective_pressure_handoff/correction_v2/report.json)
records **eight passing checks out of nine**, not an all-passing evaluation.
The [retained-evidence controls](../../tests/research/test_collective_pressure_handoff_evidence.py)
re-admit captured primitives and rebuild the relevant fields and work without
invoking the producer or changing the frozen response.

Two earlier attempts stopped before any IL/OZ action: a
[generic-copy failure](../../docs/assets/collective_pressure_handoff/failure.json)
on a runtime-cache lock, then a
[serialization failure](../../docs/assets/collective_pressure_handoff/correction_v1/failure.json)
on the existing uniform-real BEPI representation. Its complex containers have
zero imaginary part and pass the shared signed-form owner. The corrections
change copying and lossless recording only. Both failed attempts, source
archives and protocols remain intact; law, preparation, word, numerical budget
and prospective predictions are unchanged. Their associations are recorded by
the [corrective freeze](../../docs/assets/collective_pressure_handoff/correction_v2/freeze.json).

### Actual action and complete-law handoff

Both stages were admitted under strict live rejection. The inherited
SHA-to-IL sequence emitted the existing advisory anti-pattern warning; it
neither rejected nor replaced the action, and no waiver was installed.
The captured IL phase changes modify the fresh form-rate row, leave the
fresh phase-rate row unchanged, and have strictly negative whole-edge work:

\[
 \Delta H_{\rm IL}\approx-2.71737569684\,10^{-7}.
\]

These displayed decimals summarize retained rational enclosures. The
shared-edge correction agrees with the independent full-edge difference,
and the changed lifted phase mean is approximately `2.78994519689e-5`.
Direct endpoint admission gives

\[
 Z^2\approx1.03208170158\,10^{-5}<1/256,\qquad
 H-H_*\approx1.03692602064\,10^{-5}<1/6400.
\]

Thus this actual represented endpoint meets the existing continuous
maintenance barrier independently of the unresolved ideal-phase accuracy
comparison below. This supplies conditional continuation under the same
held law, not an evaluated trajectory or autonomous occurrence.

OZ selected the amplification branch on both targets and changed raw stored
pressure. All fourteen neighbor candidates were checked; none crossed the
unchanged `0.05` propagation threshold. No consumed nonzero product vanished
to zero. After the declared refresh, stored pressure, all consumed continuous
coordinates and both complete rate rows equal the post-IL values exactly.
Histories and stage records remain different. Acceleration is unavailable
without the required temporal EPI evidence; compatibility zeros are not
measurements. This confirms the model-specific projection equivalence.

### Preserve the unsuccessful certificate

The frozen compound IL check stopped at its first assertion: the per-target
ideal-phase residual enclosures were not both contained in `[-2^-46,2^-46]`.
The approximate bounds are:

| Target | Enclosed phase defect | Frozen classification |
| --- | --- | --- |
| 0 | `[-1.91769e-16, 1.51906e-14]` | Unresolved at `2^-46 ≈ 1.42109e-14` |
| 1 | `[-1.48910e-14, 4.91012e-16]` | Unresolved at the same budget |

Both boxes contain zero: neither establishes that the actual defect exceeds
the budget. The saved, predeclared 100-digit cross-check estimates defects of
about `-4.25e-18` and `3.04e-16`; those estimates do not replace the rigorous
certificate. The width comes from the existing relative-resultant primitive's
12-term, 64-bit sine/cosine bounds. Subsequent 128-bit interval arithmetic
does not sharpen inherited input enclosures.

The failed compound assertion did not run its later checks. Independent
retained-state verification supplies the direct work, identity and mean
claims above. The separately computed energy-work defect interval,
approximately `[-1.076742e-13,1.076765e-13]`, is inside the declared
`±14*2^-46` allowance. No tolerance was relaxed, no precision increased and
no IL/OZ response regenerated to change the frozen verdict.

The original bounded handoff therefore left a numerical subclaim unresolved;
the separate review below resolves it without changing that evaluation.
The separate [pressure-completion analysis](SINE_PRESSURE_COMPLETION.md)
classifies the additional mean, storage and constitutive-invariance obligations
of an independent pressure intervention. It consumes this saved endpoint only
for a detached balance check; no pressure-retention law or reservoir is supplied
by the word, and its original verdict is unchanged.

<a id="collective-pressure-phase-review"></a>
## 14. Separate correction of the retained phase certificate

The [review protocol](../../docs/assets/collective_pressure_handoff/phase_review_v1/protocol.json)
and [freeze](../../docs/assets/collective_pressure_handoff/phase_review_v1/freeze.json)
precede one read-only corrective evaluation of the same `initial` and `IL_raw`
primitive snapshots. They retain the original represented coefficient, target
order, phase-lift association, absolute allowance `2^-46` and independent
100-decimal-digit check. No operator, preparation, continuation or response
producer is replayed. This is a retrospective numerical correction, not a new
prospective physical prediction or a replacement for the frozen 8/9 report.

The [shared detached calculator](../../src/tnfr/research/phase_reset_certificate.py)
reconstructs each bare-neighbor relative resultant directly from the exact
represented primitive phases. The existing outward 128-bit sine/cosine owner
now encloses those inputs before the angle operation; the former 12-term,
64-bit resultant boxes are not promoted into narrower boxes. The principal
actual displacement is accepted as the nearest lift to the ideal displacement
only after proving that their entire difference lies strictly inside
`(-pi,pi)`. The actual ideal-displacement coefficient remains the rational
represented by binary64 `0.3`.

The [corrective response](../../docs/assets/collective_pressure_handoff/phase_review_v1/response.json)
and [report](../../docs/assets/collective_pressure_handoff/phase_review_v1/report.json)
certify both defects inside the unchanged budget:

| Target | Approximate defect, summarizing its rigorous interval | Interval width |
| --- | --- | --- |
| 0 | `-4.25252514856e-18` | about `2.351e-38` |
| 1 | `3.03932787662e-16` | about `2.351e-38` |

The independent high-precision values lie inside these enclosures. Independent
retained-state work, identity and OZ projection checks also pass, and the later
boxes of the original compound check meet their respective assertions. The
scope is separate evidence for the formerly unresolved numerical subclaim;
the historical report remains **8/9**, with all its source artifacts intact.

[Synthetic controls](../../tests/research/test_phase_reset_certificate.py)
check angle/lift branches, tolerance classification and invalid domains.
[Read-only evidence controls](../../tests/research/test_collective_phase_review_evidence.py)
bind the review to the retained bytes, rebuild primitive inputs and compare
the certified intervals with the independent oracle. Neither a small phase
error nor corrected arithmetic selects a pressure-memory law or an autonomous
OZ occurrence mechanism. The constitutive-refresh equivalence in Section 2
and the conditional pressure-completion obstruction remain unchanged.
