# Forced support balance and model boundaries

Fixed-support form balance is the common starting point. Regional response, phase clocks, locking and native writer results retain separate complete laws. Source-bound executed comparisons are archived; mathematical restrictions remain maintained.

## Chapters and scope

| Chapter | Responsibility |
| --- | --- |
| [Retained forced-support regional observations](research/archive/support/REGIONAL_OBSERVATIONS.md) | Frozen finite regional comparisons and their original preparation and numerical limitations. |
| [Forced regional response and environmental readouts](nodal/FORCED_REGIONAL_RESPONSE.md) | Regional balances, represented phase reduction and environmental-input geometry. |
| [Forced-source closure and phase clocks](nodal/FORCED_SOURCE_AND_CLOCK.md) | Phase/capacity tangency, source constraints and the limits of synchronization as a clock. |
| [Conditional phase locking and circulation geometry](nodal/FORCED_PHASE_LOCKING.md) | P2 and network locking, capacity adaptation, circulation and cycle periods. |
| [Winding generation and native writer boundaries](nodal/FORCED_WINDING_AND_WRITERS.md) | Acute winding retention, chart escape and native Mutation/capacity writer obstructions. |
| [Geometric identity and capacity-feedback bounds](nodal/FORCED_GEOMETRIC_IDENTITY.md) | Local restoring response, geometric identity and prospective capacity tubes. |

Section numbers remain stable across this collection. The [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) alone assigns research work.

## 1. The reference comes from the existing nodal channels

The connection studied in
[THOL birth and transport](THOL_BIRTH_AND_TRANSPORT.md) changes both the
weighted EPI walk and the unweighted capacity/phase neighborhoods.
Its next evolution can be analyzed by retaining that connected support,
positive capacities and phases, then refreshing the canonical pressure
before each declared physical step.

Let `W` be fixed symmetric nonnegative conductance, connected through its
positive off-diagonal entries. Let `d_i=sum_j W_ij>0`, `D=diag(d_i)`,
`B=D-W`, and `nu_i>0`. Loops follow the shared conductance convention and
zero-weight support edges contribute no EPI transport. The existing
normalized EPI channel coefficient is `e>0`. Write the held remaining
canonical channels as

$$
F=w_{\nu}g_{\nu}+w_{\phi}g_{\phi}
  +w_{\mathrm{topo}}g_{\mathrm{topo}}.
$$

Here `g_nu` and `g_topo` use the actual unique-neighbor support; `g_phi`
uses the canonical neighbor-phasor argument. Holding these source fields
and their channel coefficients makes `F` constant. It is a decomposition
of the existing pressure, with no additional feedback coefficient.
Its phase component need not be a linear phase Laplacian.

The nodal equation on this retained state is therefore

$$
\dot x=\operatorname{diag}(\nu)(-eD^{-1}Bx+F).
$$

An exact rational companion can treat finite materialized coefficients
as exact real inputs. A live pressure refresh generally differs from
this reference by a recorded numerical residual. Neither a stored
pressure value nor an operator label establishes that the other
channels have been independently identified or held fixed.

## 2. Compatibility, mean drift and the unique relative profile

Define the positive metric vector `h_i=d_i/nu_i`, its diagonal matrix
`H=diag(h_i)`, total weight `Z=sum_i h_i`, and weighted mean

$$
m(x)=\frac{h^\top x}{Z}.
$$

Symmetry gives `1^T B=0`, hence

$$
\frac{d}{dt}(h^\top x)=\mathbf1^\top DF=:b,
\qquad \dot m=\bar v:=b/Z.
$$

The initial mean is unrestricted. Its exact reference trajectory is
`m(t)=m(x0)+bar_v*t`; the mean is conserved only when `b=0`.

A zero-pressure stationary EPI exists precisely when

$$
b=\sum_i d_iF_i=0.
$$

Indeed its equation is `e*B*x=D*F`, and the range of a connected
symmetric Laplacian is the subspace orthogonal to `1`. Positive capacity
makes zero rate equivalent to zero pressure in this fixed model.
Zero capacity, disconnected positive conductance and changing channels
require different conclusions.

For every `F`, including `b!=0`, there is a unique centered relative
profile `z` satisfying

$$
eBz=DF-\bar v h,\qquad h^\top z=0.
$$

The right-hand side sums to zero by the definition of `bar_v`, so a
solution exists and is unique after fixing its weighted mean. A useful
exact computation is the nonsingular rational system

$$
(eB+hh^\top)z=DF-\bar v h.
$$

Multiplication by `1^T` enforces `h^T z=0`; the remaining equation is the
original profile equation. The rank-one term enforces the algebraic
choice of mean and is not part of the nodal pressure or evolution.
The existing exact matrix inverse suffices for this bounded solve.

Let

$$
A=e\operatorname{diag}(\nu_i/d_i)B,
\qquad u=x-m(x)\mathbf1-z.
$$

Then `h^T u=0` and direct substitution gives

$$
\dot u=-Au,
\qquad
x(t)=[m(x_0)+\bar v t]\mathbf1+z+e^{-At}u(0).
$$

This separates a retained spatial form, uniform mean drift and relaxing
deviations without adding an independent dynamical law. A nonzero `z`
alone does not establish localization or a zero-pressure equilibrium.
On the exact drifting profile the pressure is `bar_v/nu_i`, while
the raw Dirichlet energy stays at `z^T*B*z/2` because `B*1=0`.
A constant spatial energy therefore does not establish stationarity.

## 3. Exact relaxation and the finite-chart boundary

The identity `H*A=e*B` makes `A` self-adjoint in the positive `H` metric.
Its only zero mode on connected conductance is the uniform vector.
The centered error variance and Dirichlet error satisfy

$$
V(u)=\tfrac12u^\top Hu,
\qquad \dot V=-e\,u^\top Bu\leq0,
$$

$$
E(u)=\tfrac12u^\top Bu,
\qquad
\dot E=-e(Bu)^\top\operatorname{diag}(\nu_i/d_i)(Bu)\leq0.
$$

The positive nonuniform spectrum implies `u(t)->0` in the exact held
model. Thus the relative profile attracts every initial field after
matching its initial weighted mean. When `b!=0`, the complete EPI field
continues to drift and does not converge to a stationary state.

This asymptotic statement assumes an unrestricted scalar chart with the
same fixed coefficients for all times. If every EPI coordinate must stay
within finite bounds `[l,r]`, its positively weighted mean must also stay
in that interval. The ideal affine mean therefore cannot remain inside
the chart beyond

$$
t>\frac{r-m(x_0)}{\bar v}\quad(\bar v>0),
\qquad
t>\frac{m(x_0)-l}{|\bar v|}\quad(\bar v<0).
$$

Individual nodes may reach a bound earlier; the mean does not control
every coordinate. Even `b=0` does not ensure that the limiting profile
lies inside the configured chart. Clipping changes the endpoint map
and contributes a measurable defect. A clipped stationary endpoint
need not have zero canonical pressure.

## 4. Finite refreshed Euler observations and their exact defects

For one declared duration `dt>=0`, let `p` be the captured pressure
before integration and `x_plus` the observed endpoint. Define

$$
p_*(x)=-eD^{-1}Bx+F,
\qquad \varepsilon_p=p-p_*(x),
$$

$$
\delta_x=x_+-x-dt\operatorname{diag}(\nu)p,
\qquad
\xi=dt\operatorname{diag}(\nu)\varepsilon_p+\delta_x.
$$

The first defect measures pressure realization relative to the declared
held model. The second measures the implemented endpoint relative to
its actual held pressure, including arithmetic, clipping or a different
solver. These definitions give the exact finite identity

$$
x_+=(I-dtA)x+dt\operatorname{diag}(\nu)F+\xi.
$$

Consequently the mean budget is

$$
m(x_+)-m(x)-dt\bar v
=\frac{dt\,d^\top\varepsilon_p+h^\top\delta_x}{Z}.
$$

Neither defect is inferred from the other. Their signed mean
contributions can cancel, so a small mean error alone is insufficient
evidence of an accurate pressure realization or endpoint.

Let `P_H=I-1*h^T/Z`, `Q=I-dt*A`, and `chi=P_H*xi`. Centering the observed
endpoint by its own weighted mean gives

$$
u_+=Qu+\chi,
\qquad
\chi=dtP_H\operatorname{diag}(\nu)\varepsilon_p+P_H\delta_x.
$$

The exact Dirichlet error budget is

$$
\begin{aligned}
E(u_+)-E(u)
={}&-dt\,e(Bu)^\top\operatorname{diag}(\nu_i/d_i)(Bu)\\
 &+\tfrac12dt^2(Au)^\top B(Au)\\
 &+(BQu)^\top\chi+\tfrac12\chi^\top B\chi.
\end{aligned}
$$

The first term is the dissipative nodal contribution. The next is the
finite Euler correction. The remaining terms retain the signed
interaction with the observed defect and its quadratic contribution.
They must not be discarded merely because the underlying continuous
reference is dissipative.

Likewise the weighted variance has the exact identity

$$
V(u_+)-V(u)
=-dt\,e\,u^\top Bu+\tfrac12dt^2(Au)^\top H(Au)
 +(Qu)^\top H\chi+\tfrac12\chi^\top H\chi.
$$

Over a finite sequence with the same held model, each mean identity
and each energy identity telescopes. Centered errors also propagate
through the complete matrices `Q_k`, so defects need not stay in one
spatial eigenmode. These are finite observations; a future error bound
or a runtime asymptotic theorem needs additional uniform hypotheses.

For the ideal Euler map, the sufficient step condition
`dt*e*max(nu)<=1` places the spectrum of `dt*A` in `[0,2]`. Hence neither
`V` nor `E` increases when `chi=0`. Equality can preserve an alternating
mode on an even cycle; nonincrease is not automatically strict decay.
The observed defect terms remain necessary under this step condition.

## 5. Reuse, realization and the evidence boundary

[`forced_support.py`](../src/tnfr/physics/forced_support.py) supplies
`derive_forced_support_balance`, `observe_forced_support_state` and
`observe_forced_support_step`. They rebuild public cached fields before
using them and check the held node order, conductance, unique support
and capacities. Their detached records contain no execution seal.
The profile computation reuses
[`_exact_linear_algebra.py`](../src/tnfr/physics/_exact_linear_algebra.py).
It introduces no alternative EPI integrator. The finite error budget
reuses
[`support_transport.py`](../src/tnfr/physics/support_transport.py)
on detached error coordinates with the same conductance and capacity.
Its reference error pressure is `-e*D^-1*B*u`, and its endpoint defect is
exactly `chi`. This is an algebraic observation of transformed data;
the error coordinate is not written into live nodes or interpreted as
a newly created physical node.

The held non-EPI source must be obtained from the actual canonical
channels independently of a fitted pressure residual. In particular,
the nonlinear phase contribution must use the shared phasor kernel,
retain its represented coefficients and distinguish its realization
from the subsequent full multichannel pressure arithmetic. A direct
operator pressure write or a stale stored value belongs in the
pressure defect and must not silently redefine `F`.

[`capture_non_epi_forcing`](../src/tnfr/physics/forcing_realization.py)
materializes the unit phase channel with the existing fused NumPy
kernel, then combines its represented values with the exact support
capacity/topology gradients and the effective channel coefficients.
It reads the engine's actual cached weight mix without normalizing it
again. The fresh full-kernel pressure defect and the stored-minus-fresh
pressure residual are reported separately. Their sum is
`epsilon_p` in the finite budget above.

This bridge is restricted to the default NumPy, non-JIT pressure branch
with at most 100 directed unique-support entries. It rejects a disabled
NumPy branch or a custom pressure callback. Neighbor insertion order
and zero-weight phase-support edges are preserved. The materialized
phase values define exact reference coefficients; the observer does
not prove exact transcendental evaluation, identify another backend
or validate future graph states.

The bounded experiment begins with the causally born and canonically
connected child from the preceding block. It retains the resulting
support, capacities and phases during explicitly refreshed Euler
segments. A separately prepared compatible control uses uniform
capacity and aligned phase on the same support. A separate prepared
boundary control starts near a scalar EPI bound. These comparisons
distinguish nonzero forcing, homogeneous relaxation and clipping
without changing the birth history of the causal case.

The tests must reject a claimed zero-pressure profile when `b!=0`,
verify the profile equation and its centered gauge, and account for
both pressure and endpoint defects at every captured step. They must
also keep a clipped mean change distinct from `dt*bar_v` and preserve
all fixed-support assumptions. Agreement over the measured horizon
establishes neither spontaneous selection of the held fields nor
restoration under later structural events. Those require an admitted
policy that actually changes the support, capacities or phases.

## Section link directory

These aliases route existing citations to their substantive owner.

- <a id="relative-form-and-mean-drift-on-a-held-canonical-support"></a>[Relative form and mean drift on a held canonical support](#relative-form-and-mean-drift-on-a-held-canonical-support)

- <a id="6-bounded-executed-comparison"></a>[6. Bounded executed comparison](research/archive/support/REGIONAL_OBSERVATIONS.md#6-bounded-executed-comparison)

- <a id="7-a-region-and-its-environment-on-the-same-nodal-support"></a>[7. A region and its environment on the same nodal support](nodal/FORCED_REGIONAL_RESPONSE.md#7-a-region-and-its-environment-on-the-same-nodal-support)

- <a id="conditional-child-response-is-not-an-autonomous-child-region"></a>[Conditional child response is not an autonomous child region](nodal/FORCED_REGIONAL_RESPONSE.md#conditional-child-response-is-not-an-autonomous-child-region)

- <a id="8-retained-thol-regional-audit"></a>[8. Retained THOL regional audit](research/archive/support/REGIONAL_OBSERVATIONS.md#8-retained-thol-regional-audit)

- <a id="9-finite-regional-observation-with-a-held-nodal-rate"></a>[9. Finite regional observation with a held nodal rate](nodal/FORCED_REGIONAL_RESPONSE.md#9-finite-regional-observation-with-a-held-nodal-rate)

- <a id="regional-contrast-is-not-a-universal-coherence-score"></a>[Regional contrast is not a universal coherence score](nodal/FORCED_REGIONAL_RESPONSE.md#regional-contrast-is-not-a-universal-coherence-score)

- <a id="10-retained-temporal-regional-identity-audit"></a>[10. Retained temporal regional identity audit](research/archive/support/REGIONAL_OBSERVATIONS.md#10-retained-temporal-regional-identity-audit)

- <a id="11-phase-source-relevance-at-a-fixed-regional-state"></a>[11. Phase-source relevance at a fixed regional state](nodal/FORCED_REGIONAL_RESPONSE.md#11-phase-source-relevance-at-a-fixed-regional-state)

- <a id="retained-two-phase-comparison"></a>[Retained two-phase comparison](nodal/FORCED_REGIONAL_RESPONSE.md#retained-two-phase-comparison)

- <a id="12-exact-reduction-of-represented-phase-components"></a>[12. Exact reduction of represented phase components](nodal/FORCED_REGIONAL_RESPONSE.md#12-exact-reduction-of-represented-phase-components)

- <a id="caller-policies-remain-separate-from-arithmetic"></a>[Caller policies remain separate from arithmetic](nodal/FORCED_REGIONAL_RESPONSE.md#caller-policies-remain-separate-from-arithmetic)

- <a id="versioned-global-coordination-integration"></a>[Versioned global-coordination integration](nodal/FORCED_REGIONAL_RESPONSE.md#versioned-global-coordination-integration)

- <a id="13-versioned-phase-correction-at-the-retained-regional-state"></a>[13. Versioned phase correction at the retained regional state](research/archive/support/REGIONAL_OBSERVATIONS.md#13-versioned-phase-correction-at-the-retained-regional-state)

- <a id="14-regional-recovery-versus-loss-of-form-in-the-retained-paired-window"></a>[14. Regional recovery versus loss of form in the retained paired window](research/archive/support/REGIONAL_OBSERVATIONS.md#14-regional-recovery-versus-loss-of-form-in-the-retained-paired-window)

- <a id="15-child-cohort-distortion-and-regional-mean-to-shape-transfer"></a>[15. Child-cohort distortion and regional mean-to-shape transfer](nodal/FORCED_REGIONAL_RESPONSE.md#15-child-cohort-distortion-and-regional-mean-to-shape-transfer)

- <a id="staged-nodal-accounting"></a>[Staged nodal accounting](nodal/FORCED_REGIONAL_RESPONSE.md#staged-nodal-accounting)

- <a id="a-connection-to-regional-identity"></a>[A connection to regional identity](nodal/FORCED_REGIONAL_RESPONSE.md#a-connection-to-regional-identity)

- <a id="retained-result-and-maintenance-boundary"></a>[Retained result and maintenance boundary](nodal/FORCED_REGIONAL_RESPONSE.md#retained-result-and-maintenance-boundary)

- <a id="16-localized-regional-form-damage-and-finite-configured-restoration"></a>[16. Localized regional form damage and finite configured restoration](nodal/FORCED_REGIONAL_RESPONSE.md#16-localized-regional-form-damage-and-finite-configured-restoration)

- <a id="shared-accounting-and-decision-scope"></a>[Shared accounting and decision scope](nodal/FORCED_REGIONAL_RESPONSE.md#shared-accounting-and-decision-scope)

- <a id="measured-result"></a>[Measured result](nodal/FORCED_REGIONAL_RESPONSE.md#measured-result)

- <a id="17-conditional-regional-response-and-environmental-input"></a>[17. Conditional regional response and environmental input](nodal/FORCED_REGIONAL_RESPONSE.md#17-conditional-regional-response-and-environmental-input)

- <a id="a-condition-derived-from-the-admitted-nodal-map"></a>[A condition derived from the admitted nodal map](nodal/FORCED_REGIONAL_RESPONSE.md#a-condition-derived-from-the-admitted-nodal-map)

- <a id="existing-shape-versus-incoming-mean-and-parent-differences"></a>[Existing shape versus incoming mean and parent differences](nodal/FORCED_REGIONAL_RESPONSE.md#existing-shape-versus-incoming-mean-and-parent-differences)

- <a id="why-a-region-only-gain-can-fail"></a>[Why a region-only gain can fail](nodal/FORCED_REGIONAL_RESPONSE.md#why-a-region-only-gain-can-fail)

- <a id="matched-retained-witnesses"></a>[Matched retained witnesses](nodal/FORCED_REGIONAL_RESPONSE.md#matched-retained-witnesses)

- <a id="18-environmental-input-geometry-and-protected-read-outs"></a>[18. Environmental-input geometry and protected read-outs](nodal/FORCED_REGIONAL_RESPONSE.md#18-environmental-input-geometry-and-protected-read-outs)

- <a id="image-and-annihilator-in-the-original-metric"></a>[Image and annihilator in the original metric](nodal/FORCED_REGIONAL_RESPONSE.md#image-and-annihilator-in-the-original-metric)

- <a id="exact-classification-of-the-retained-child-cohort"></a>[Exact classification of the retained child cohort](nodal/FORCED_REGIONAL_RESPONSE.md#exact-classification-of-the-retained-child-cohort)

- <a id="19-support-symmetry-versus-the-admitted-nodal-and-reset-maps"></a>[19. Support symmetry versus the admitted nodal and reset maps](nodal/FORCED_REGIONAL_RESPONSE.md#19-support-symmetry-versus-the-admitted-nodal-and-reset-maps)

- <a id="separate-mathematical-objects"></a>[Separate mathematical objects](nodal/FORCED_REGIONAL_RESPONSE.md#separate-mathematical-objects)

- <a id="complete-retained-classification"></a>[Complete retained classification](nodal/FORCED_REGIONAL_RESPONSE.md#complete-retained-classification)

- <a id="20-same-snapshot-reception-and-the-limit-of-geometric-protection"></a>[20. Same-snapshot Reception and the limit of geometric protection](nodal/FORCED_REGIONAL_RESPONSE.md#20-same-snapshot-reception-and-the-limit-of-geometric-protection)

- <a id="comparison-using-the-existing-coefficient-kernel"></a>[Comparison using the existing coefficient kernel](nodal/FORCED_REGIONAL_RESPONSE.md#comparison-using-the-existing-coefficient-kernel)

- <a id="exact-result-and-the-unmet-input-condition"></a>[Exact result and the unmet input condition](nodal/FORCED_REGIONAL_RESPONSE.md#exact-result-and-the-unmet-input-condition)

- <a id="runtime-boundary-and-disposition"></a>[Runtime boundary and disposition](nodal/FORCED_REGIONAL_RESPONSE.md#runtime-boundary-and-disposition)

- <a id="21-closing-the-relaxed-phase-capacity-source"></a>[21. Closing the relaxed phase-capacity source](nodal/FORCED_SOURCE_AND_CLOCK.md#21-closing-the-relaxed-phase-capacity-source)

- <a id="exact-model-and-common-fixed-fields"></a>[Exact model and common fixed fields](nodal/FORCED_SOURCE_AND_CLOCK.md#exact-model-and-common-fixed-fields)

- <a id="conditional-consequence-of-the-sense-index-gate-policy"></a>[Conditional consequence of the Sense Index gate policy](nodal/FORCED_SOURCE_AND_CLOCK.md#conditional-consequence-of-the-sense-index-gate-policy)

- <a id="what-can-and-cannot-be-inferred"></a>[What can and cannot be inferred](nodal/FORCED_SOURCE_AND_CLOCK.md#what-can-and-cannot-be-inferred)

- <a id="22-source-tangency-without-a-telemetry-controller"></a>[22. Source tangency without a telemetry controller](nodal/FORCED_SOURCE_AND_CLOCK.md#22-source-tangency-without-a-telemetry-controller)

- <a id="regular-fixed-support-identity"></a>[Regular fixed-support identity](nodal/FORCED_SOURCE_AND_CLOCK.md#regular-fixed-support-identity)

- <a id="finite-events-and-what-an-instantaneous-test-cannot-prove"></a>[Finite events and what an instantaneous test cannot prove](nodal/FORCED_SOURCE_AND_CLOCK.md#finite-events-and-what-an-instantaneous-test-cannot-prove)

- <a id="consequence-for-the-primary-research-question"></a>[Consequence for the primary research question](nodal/FORCED_SOURCE_AND_CLOCK.md#consequence-for-the-primary-research-question)

- <a id="23-capacity-exposure-does-not-determine-a-phase-clock"></a>[23. Capacity exposure does not determine a phase clock](nodal/FORCED_SOURCE_AND_CLOCK.md#23-capacity-exposure-does-not-determine-a-phase-clock)

- <a id="source-and-implementation-boundary"></a>[Source and implementation boundary](nodal/FORCED_SOURCE_AND_CLOCK.md#source-and-implementation-boundary)

- <a id="exact-independence-including-a-relative-phase-witness"></a>[Exact independence, including a relative-phase witness](nodal/FORCED_SOURCE_AND_CLOCK.md#exact-independence-including-a-relative-phase-witness)

- <a id="what-the-nodal-equation-does-derive-accumulated-capacity"></a>[What the nodal equation does derive: accumulated capacity](nodal/FORCED_SOURCE_AND_CLOCK.md#what-the-nodal-equation-does-derive-accumulated-capacity)

- <a id="24-rigidity-and-flexibility-of-a-held-phase-source"></a>[24. Rigidity and flexibility of a held phase source](nodal/FORCED_SOURCE_AND_CLOCK.md#24-rigidity-and-flexibility-of-a-held-phase-source)

- <a id="regular-rigidity-from-the-shared-mean-derivative"></a>[Regular rigidity from the shared mean derivative](nodal/FORCED_SOURCE_AND_CLOCK.md#regular-rigidity-from-the-shared-mean-derivative)

- <a id="connected-support-alone-is-insufficient-a-cube-family"></a>[Connected support alone is insufficient: a cube family](nodal/FORCED_SOURCE_AND_CLOCK.md#connected-support-alone-is-insufficient-a-cube-family)

- <a id="reusable-exact-observation-and-claim-boundary"></a>[Reusable exact observation and claim boundary](nodal/FORCED_SOURCE_AND_CLOCK.md#reusable-exact-observation-and-claim-boundary)

- <a id="25-relational-time-and-synchronization-are-separate-claims"></a>[25. Relational time and synchronization are separate claims](nodal/FORCED_SOURCE_AND_CLOCK.md#25-relational-time-and-synchronization-are-separate-claims)

- <a id="a-local-state-clock-requires-an-already-specified-tangent"></a>[A local state clock requires an already specified tangent](nodal/FORCED_SOURCE_AND_CLOCK.md#a-local-state-clock-requires-an-already-specified-tangent)

- <a id="curve-admission-can-determine-a-speed-without-selecting-the-curve"></a>[Curve admission can determine a speed without selecting the curve](nodal/FORCED_SOURCE_AND_CLOCK.md#curve-admission-can-determine-a-speed-without-selecting-the-curve)

- <a id="reuse-and-implementation-scope"></a>[Reuse and implementation scope](nodal/FORCED_SOURCE_AND_CLOCK.md#reuse-and-implementation-scope)

- <a id="26-conditional-phase-locking-and-form-restoration-on-fixed-p2"></a>[26. Conditional phase locking and form restoration on fixed P2](nodal/FORCED_PHASE_LOCKING.md#26-conditional-phase-locking-and-form-restoration-on-fixed-p2)

- <a id="declared-composition-and-complete-state"></a>[Declared composition and complete state](nodal/FORCED_PHASE_LOCKING.md#declared-composition-and-complete-state)

- <a id="locked-target-conserved-mean-and-local-response"></a>[Locked target, conserved mean and local response](nodal/FORCED_PHASE_LOCKING.md#locked-target-conserved-mean-and-local-response)

- <a id="an-invariant-chart-and-explicit-error-bounds"></a>[An invariant chart and explicit error bounds](nodal/FORCED_PHASE_LOCKING.md#an-invariant-chart-and-explicit-error-bounds)

- <a id="inactive-channel-control-and-implementation-boundary"></a>[Inactive-channel control and implementation boundary](nodal/FORCED_PHASE_LOCKING.md#inactive-channel-control-and-implementation-boundary)

- <a id="27-capacity-adaptation-moves-the-conditional-p2-target"></a>[27. Capacity adaptation moves the conditional P2 target](nodal/FORCED_PHASE_LOCKING.md#27-capacity-adaptation-moves-the-conditional-p2-target)

- <a id="exact-event-calculation-and-its-limits"></a>[Exact event calculation and its limits](nodal/FORCED_PHASE_LOCKING.md#exact-event-calculation-and-its-limits)

- <a id="immediate-pressure-and-subsequent-form-response"></a>[Immediate pressure and subsequent form response](nodal/FORCED_PHASE_LOCKING.md#immediate-pressure-and-subsequent-form-response)

- <a id="finite-default-policy-witness"></a>[Finite default-policy witness](nodal/FORCED_PHASE_LOCKING.md#finite-default-policy-witness)

- <a id="28-geometry-limited-regional-phase-locking-on-the-unit-barbell"></a>[28. Geometry-limited regional phase locking on the unit barbell](nodal/FORCED_PHASE_LOCKING.md#28-geometry-limited-regional-phase-locking-on-the-unit-barbell)

- <a id="declared-model-and-the-bridge-constraint"></a>[Declared model and the bridge constraint](nodal/FORCED_PHASE_LOCKING.md#declared-model-and-the-bridge-constraint)

- <a id="full-six-node-local-stability"></a>[Full six-node local stability](nodal/FORCED_PHASE_LOCKING.md#full-six-node-local-stability)

- <a id="the-actual-phasor-source-and-its-compatible-form"></a>[The actual phasor source and its compatible form](nodal/FORCED_PHASE_LOCKING.md#the-actual-phasor-source-and-its-compatible-form)

- <a id="mean-conservation-is-conditional-on-the-phase-source"></a>[Mean conservation is conditional on the phase source](nodal/FORCED_PHASE_LOCKING.md#mean-conservation-is-conditional-on-the-phase-source)

- <a id="prospective-finite-recovery-protocol"></a>[Prospective finite recovery protocol](nodal/FORCED_PHASE_LOCKING.md#prospective-finite-recovery-protocol)

- <a id="finite-outcome-with-the-frozen-target"></a>[Finite outcome with the frozen target](nodal/FORCED_PHASE_LOCKING.md#finite-outcome-with-the-frozen-target)

- <a id="29-graph-independent-locking-and-phase-source-constraints"></a>[29. Graph-independent locking and phase-source constraints](nodal/FORCED_PHASE_LOCKING.md#29-graph-independent-locking-and-phase-source-constraints)

- <a id="exact-model-phase-support-is-not-transport-conductance"></a>[Exact model: phase support is not transport conductance](nodal/FORCED_PHASE_LOCKING.md#exact-model-phase-support-is-not-transport-conductance)

- <a id="locked-rate-cut-load-and-the-sign-of-the-phase-source"></a>[Locked rate, cut load and the sign of the phase source](nodal/FORCED_PHASE_LOCKING.md#locked-rate-cut-load-and-the-sign-of-the-phase-source)

- <a id="what-equal-capacities-obstruct-and-what-they-do-not"></a>[What equal capacities obstruct, and what they do not](nodal/FORCED_PHASE_LOCKING.md#what-equal-capacities-obstruct-and-what-they-do-not)

- <a id="a-residual-bound-away-from-exact-locking"></a>[A residual bound away from exact locking](nodal/FORCED_PHASE_LOCKING.md#a-residual-bound-away-from-exact-locking)

- <a id="capacity-contrast-is-necessary-here-not-sufficient-for-stationary-form"></a>[Capacity contrast is necessary here, not sufficient for stationary form](nodal/FORCED_PHASE_LOCKING.md#capacity-contrast-is-necessary-here-not-sufficient-for-stationary-form)

- <a id="admission-boundaries-and-represented-observations"></a>[Admission boundaries and represented observations](nodal/FORCED_PHASE_LOCKING.md#admission-boundaries-and-represented-observations)

- <a id="30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods"></a>[30. Acute phase locks are circulation states with integral cycle periods](nodal/FORCED_PHASE_LOCKING.md#30-acute-phase-locks-are-circulation-states-with-integral-cycle-periods)

- <a id="fixed-support-common-capacity-and-the-declared-phase-law"></a>[Fixed support, common capacity and the declared phase law](nodal/FORCED_PHASE_LOCKING.md#fixed-support-common-capacity-and-the-declared-phase-law)

- <a id="integral-periods-and-reconstruction"></a>[Integral periods and reconstruction](nodal/FORCED_PHASE_LOCKING.md#integral-periods-and-reconstruction)

- <a id="necessary-and-sufficient-locking-equations-and-sector-uniqueness"></a>[Necessary and sufficient locking equations, and sector uniqueness](nodal/FORCED_PHASE_LOCKING.md#necessary-and-sufficient-locking-equations-and-sector-uniqueness)

- <a id="what-graph-structure-permits-or-excludes"></a>[What graph structure permits or excludes](nodal/FORCED_PHASE_LOCKING.md#what-graph-structure-permits-or-excludes)

- <a id="persistence-of-a-sector-is-not-its-formation"></a>[Persistence of a sector is not its formation](nodal/FORCED_PHASE_LOCKING.md#persistence-of-a-sector-is-not-its-formation)

- <a id="exact-implementation-boundary"></a>[Exact implementation boundary](nodal/FORCED_PHASE_LOCKING.md#exact-implementation-boundary)

- <a id="31-one-added-chord-extends-the-cycle-lattice-not-a-phase-generation-law"></a>[31. One added chord extends the cycle lattice, not a phase-generation law](nodal/FORCED_PHASE_LOCKING.md#31-one-added-chord-extends-the-cycle-lattice-not-a-phase-generation-law)

- <a id="retained-support-and-the-exact-lattice-identity"></a>[Retained support and the exact lattice identity](nodal/FORCED_PHASE_LOCKING.md#retained-support-and-the-exact-lattice-identity)

- <a id="the-new-period-comes-from-the-actual-endpoint-gap"></a>[The new period comes from the actual endpoint gap](nodal/FORCED_PHASE_LOCKING.md#the-new-period-comes-from-the-actual-endpoint-gap)

- <a id="separate-a-support-extension-from-phase-writes-at-the-same-event"></a>[Separate a support extension from phase writes at the same event](nodal/FORCED_PHASE_LOCKING.md#separate-a-support-extension-from-phase-writes-at-the-same-event)

- <a id="a-larger-cycle-lattice-does-not-guarantee-a-maintained-lock"></a>[A larger cycle lattice does not guarantee a maintained lock](nodal/FORCED_PHASE_LOCKING.md#a-larger-cycle-lattice-does-not-guarantee-a-maintained-lock)

- <a id="shared-implementation-and-production-boundary"></a>[Shared implementation and production boundary](nodal/FORCED_PHASE_LOCKING.md#shared-implementation-and-production-boundary)

- <a id="prospective-default-coupling-event-and-policy-control"></a>[Prospective default-Coupling event and policy control](nodal/FORCED_PHASE_LOCKING.md#prospective-default-coupling-event-and-policy-control)

- <a id="32-acute-cycle-relaxation-retains-phase-winding-while-form-relaxes"></a>[32. Acute-cycle relaxation retains phase winding while form relaxes](nodal/FORCED_WINDING_AND_WRITERS.md#32-acute-cycle-relaxation-retains-phase-winding-while-form-relaxes)

- <a id="the-supplied-joint-law-and-the-support-it-actually-uses"></a>[The supplied joint law and the support it actually uses](nodal/FORCED_WINDING_AND_WRITERS.md#the-supplied-joint-law-and-the-support-it-actually-uses)

- <a id="acute-gaps-form-an-invariant-region"></a>[Acute gaps form an invariant region](nodal/FORCED_WINDING_AND_WRITERS.md#acute-gaps-form-an-invariant-region)

- <a id="the-canonical-phase-source-vanishes-but-phase-geometry-remains"></a>[The canonical phase source vanishes, but phase geometry remains](nodal/FORCED_WINDING_AND_WRITERS.md#the-canonical-phase-source-vanishes-but-phase-geometry-remains)

- <a id="one-existing-margin-controls-availability-and-conditional-stiffness"></a>[One existing margin controls availability and conditional stiffness](nodal/FORCED_WINDING_AND_WRITERS.md#one-existing-margin-controls-availability-and-conditional-stiffness)

- <a id="the-weighted-form-response-and-its-nonconserved-mean"></a>[The weighted form response and its nonconserved mean](nodal/FORCED_WINDING_AND_WRITERS.md#the-weighted-form-response-and-its-nonconserved-mean)

- <a id="apply-the-theorem-to-the-actual-closing-edge-conductance"></a>[Apply the theorem to the actual closing-edge conductance](nodal/FORCED_WINDING_AND_WRITERS.md#apply-the-theorem-to-the-actual-closing-edge-conductance)

- <a id="reuse-and-numerical-evidence-boundaries"></a>[Reuse and numerical evidence boundaries](nodal/FORCED_WINDING_AND_WRITERS.md#reuse-and-numerical-evidence-boundaries)

- <a id="frozen-finite-production-continuation"></a>[Frozen finite production continuation](nodal/FORCED_WINDING_AND_WRITERS.md#frozen-finite-production-continuation)

- <a id="33-a-common-phase-semicircle-obstructs-winding-generation-by-the-existing-maps"></a>[33. A common phase semicircle obstructs winding generation by the existing maps](nodal/FORCED_WINDING_AND_WRITERS.md#33-a-common-phase-semicircle-obstructs-winding-generation-by-the-existing-maps)

- <a id="the-question-is-generation-not-maintenance-of-a-prepared-sector"></a>[The question is generation, not maintenance of a prepared sector](nodal/FORCED_WINDING_AND_WRITERS.md#the-question-is-generation-not-maintenance-of-a-prepared-sector)

- <a id="actual-coupling-proposal-and-merge-formulas-preserve-that-chart"></a>[Actual Coupling proposal and merge formulas preserve that chart](nodal/FORCED_WINDING_AND_WRITERS.md#actual-coupling-proposal-and-merge-formulas-preserve-that-chart)

- <a id="equal-capacity-sine-evolution-preserves-the-moving-chart"></a>[Equal-capacity sine evolution preserves the moving chart](nodal/FORCED_WINDING_AND_WRITERS.md#equal-capacity-sine-evolution-preserves-the-moving-chart)

- <a id="why-neither-new-edges-nor-finite-compositions-can-create-a-winding"></a>[Why neither new edges nor finite compositions can create a winding](nodal/FORCED_WINDING_AND_WRITERS.md#why-neither-new-edges-nor-finite-compositions-can-create-a-winding)

- <a id="common-capacity-is-an-ideal-identity-with-an-explicit-numerical-safeguard"></a>[Common capacity is an ideal identity with an explicit numerical safeguard](nodal/FORCED_WINDING_AND_WRITERS.md#common-capacity-is-an-ideal-identity-with-an-explicit-numerical-safeguard)

- <a id="a-necessary-capacity-contrast-budget-for-leaving-the-common-chart"></a>[A necessary capacity-contrast budget for leaving the common chart](nodal/FORCED_WINDING_AND_WRITERS.md#a-necessary-capacity-contrast-budget-for-leaving-the-common-chart)

- <a id="exact-chart-observation-finite-execution-and-the-remaining-obligation"></a>[Exact chart observation, finite execution and the remaining obligation](nodal/FORCED_WINDING_AND_WRITERS.md#exact-chart-observation-finite-execution-and-the-remaining-obligation)

- <a id="finite-controls-for-overlapping-writes-and-the-excluded-preparations"></a>[Finite controls for overlapping writes and the excluded preparations](nodal/FORCED_WINDING_AND_WRITERS.md#finite-controls-for-overlapping-writes-and-the-excluded-preparations)

- <a id="34-native-runtime-admission-uses-relaxation-not-the-supplied-sine-clock"></a>[34. Native runtime admission uses relaxation, not the supplied sine clock](nodal/FORCED_WINDING_AND_WRITERS.md#34-native-runtime-admission-uses-relaxation-not-the-supplied-sine-clock)

- <a id="reuse-the-native-owners-before-transferring-the-oscillator-result"></a>[Reuse the native owners before transferring the oscillator result](nodal/FORCED_WINDING_AND_WRITERS.md#reuse-the-native-owners-before-transferring-the-oscillator-result)

- <a id="the-actual-step-order-and-freshness-boundary"></a>[The actual step order and freshness boundary](nodal/FORCED_WINDING_AND_WRITERS.md#the-actual-step-order-and-freshness-boundary)

- <a id="default-coordination-stays-in-the-common-chart-and-consumes-its-diameter"></a>[Default coordination stays in the common chart and consumes its diameter](nodal/FORCED_WINDING_AND_WRITERS.md#default-coordination-stays-in-the-common-chart-and-consumes-its-diameter)

- <a id="which-native-glyphs-and-other-writers-preserve-the-premises"></a>[Which native glyphs and other writers preserve the premises](nodal/FORCED_WINDING_AND_WRITERS.md#which-native-glyphs-and-other-writers-preserve-the-premises)

- <a id="preserve-the-declared-adaptation-fixed-point-in-represented-arithmetic"></a>[Preserve the declared adaptation fixed point in represented arithmetic](nodal/FORCED_WINDING_AND_WRITERS.md#preserve-the-declared-adaptation-fixed-point-in-represented-arithmetic)

- <a id="conditional-whole-step-conclusion-and-its-limits"></a>[Conditional whole-step conclusion and its limits](nodal/FORCED_WINDING_AND_WRITERS.md#conditional-whole-step-conclusion-and-its-limits)

- <a id="finite-ordinary-step-control"></a>[Finite ordinary-step control](nodal/FORCED_WINDING_AND_WRITERS.md#finite-ordinary-step-control)

- <a id="35-heterogeneous-capacity-opens-a-conditional-mutation-admission-gate"></a>[35. Heterogeneous capacity opens a conditional Mutation admission gate](nodal/FORCED_WINDING_AND_WRITERS.md#35-heterogeneous-capacity-opens-a-conditional-mutation-admission-gate)

- <a id="the-ideal-fresh-state-compatibility-inequality"></a>[The ideal fresh-state compatibility inequality](nodal/FORCED_WINDING_AND_WRITERS.md#the-ideal-fresh-state-compatibility-inequality)

- <a id="which-additional-reachable-primitives-can-supply-a-phase-write"></a>[Which additional reachable primitives can supply a phase write](nodal/FORCED_WINDING_AND_WRITERS.md#which-additional-reachable-primitives-can-supply-a-phase-write)

- <a id="declared-history-endpoint-the-admission-intersection-is-nonempty"></a>[Declared-history endpoint: the admission intersection is nonempty](nodal/FORCED_WINDING_AND_WRITERS.md#declared-history-endpoint-the-admission-intersection-is-nonempty)

- <a id="a-finite-native-prefix-produces-its-own-mutation-evidence"></a>[A finite native prefix produces its own Mutation evidence](nodal/FORCED_WINDING_AND_WRITERS.md#a-finite-native-prefix-produces-its-own-mutation-evidence)

- <a id="from-admission-to-the-retained-source-budget"></a>[From admission to the retained source budget](nodal/FORCED_WINDING_AND_WRITERS.md#from-admission-to-the-retained-source-budget)

- <a id="36-native-mutation-and-coordination-in-the-dirichlet-source-budget"></a>[36. Native Mutation and coordination in the Dirichlet source budget](nodal/FORCED_WINDING_AND_WRITERS.md#36-native-mutation-and-coordination-in-the-dirichlet-source-budget)

- <a id="one-detached-channel-ledger-with-heterogeneous-mobility-retained"></a>[One detached channel ledger, with heterogeneous mobility retained](nodal/FORCED_WINDING_AND_WRITERS.md#one-detached-channel-ledger-with-heterogeneous-mobility-retained)

- <a id="reuse-the-finite-euler-endpoint-identity"></a>[Reuse the finite Euler endpoint identity](nodal/FORCED_WINDING_AND_WRITERS.md#reuse-the-finite-euler-endpoint-identity)

- <a id="what-the-retained-third-step-shows"></a>[What the retained third step shows](nodal/FORCED_WINDING_AND_WRITERS.md#what-the-retained-third-step-shows)

- <a id="interpretation-boundary"></a>[Interpretation boundary](nodal/FORCED_WINDING_AND_WRITERS.md#interpretation-boundary)

- <a id="37-separate-the-frozen-hard-rail-reference-from-its-represented-endpoint"></a>[37. Separate the frozen hard-rail reference from its represented endpoint](nodal/FORCED_WINDING_AND_WRITERS.md#37-separate-the-frozen-hard-rail-reference-from-its-represented-endpoint)

- <a id="frozen-constant-velocity-hard-clipping-has-an-exact-semigroup"></a>[Frozen constant-velocity hard clipping has an exact semigroup](nodal/FORCED_WINDING_AND_WRITERS.md#frozen-constant-velocity-hard-clipping-has-an-exact-semigroup)

- <a id="common-hard-rails-cannot-increase-dirichlet-energy-relative-to-raw-input"></a>[Common hard rails cannot increase Dirichlet energy relative to raw input](nodal/FORCED_WINDING_AND_WRITERS.md#common-hard-rails-cannot-increase-dirichlet-energy-relative-to-raw-input)

- <a id="an-exact-reference-split-without-an-internal-clipping-claim"></a>[An exact reference split without an internal clipping claim](nodal/FORCED_WINDING_AND_WRITERS.md#an-exact-reference-split-without-an-internal-clipping-claim)

- <a id="the-same-retained-interval-is-close-to-its-projected-exact-reference"></a>[The same retained interval is close to its projected exact reference](nodal/FORCED_WINDING_AND_WRITERS.md#the-same-retained-interval-is-close-to-its-projected-exact-reference)

- <a id="consequence-for-the-maintenance-claim"></a>[Consequence for the maintenance claim](nodal/FORCED_WINDING_AND_WRITERS.md#consequence-for-the-maintenance-claim)

- <a id="38-default-capacity-writers-cannot-restore-an-inward-contracted-profile"></a>[38. Default capacity writers cannot restore an inward-contracted profile](nodal/FORCED_WINDING_AND_WRITERS.md#38-default-capacity-writers-cannot-restore-an-inward-contracted-profile)

- <a id="writer-and-timing-premises"></a>[Writer and timing premises](nodal/FORCED_WINDING_AND_WRITERS.md#writer-and-timing-premises)

- <a id="exact-interval-invariance-under-those-writers"></a>[Exact interval invariance under those writers](nodal/FORCED_WINDING_AND_WRITERS.md#exact-interval-invariance-under-those-writers)

- <a id="an-arbitrarily-small-inward-perturbation-prevents-profile-recovery"></a>[An arbitrarily small inward perturbation prevents profile recovery](nodal/FORCED_WINDING_AND_WRITERS.md#an-arbitrarily-small-inward-perturbation-prevents-profile-recovery)

- <a id="the-adaptation-implementation-must-preserve-its-declared-convex-range"></a>[The adaptation implementation must preserve its declared convex range](nodal/FORCED_WINDING_AND_WRITERS.md#the-adaptation-implementation-must-preserve-its-declared-convex-range)

- <a id="reuse-and-scope-of-the-obstruction"></a>[Reuse and scope of the obstruction](nodal/FORCED_WINDING_AND_WRITERS.md#reuse-and-scope-of-the-obstruction)

- <a id="39-the-same-stationary-reduced-identity-constrains-capacity-differences"></a>[39. The same stationary reduced identity constrains capacity differences](nodal/FORCED_WINDING_AND_WRITERS.md#39-the-same-stationary-reduced-identity-constrains-capacity-differences)

- <a id="fix-the-reduced-identity-and-the-pressure-realization"></a>[Fix the reduced identity and the pressure realization](nodal/FORCED_WINDING_AND_WRITERS.md#fix-the-reduced-identity-and-the-pressure-realization)

- <a id="the-capacity-interval-obstruction-survives-this-reduced-identity"></a>[The capacity interval obstruction survives this reduced identity](nodal/FORCED_WINDING_AND_WRITERS.md#the-capacity-interval-obstruction-survives-this-reduced-identity)

- <a id="retain-phase-realization-and-kernel-differences-in-represented-captures"></a>[Retain phase realization and kernel differences in represented captures](nodal/FORCED_WINDING_AND_WRITERS.md#retain-phase-realization-and-kernel-differences-in-represented-captures)

- <a id="a-prospective-stationary-construction-and-its-capacity-controls"></a>[A prospective stationary construction and its capacity controls](nodal/FORCED_WINDING_AND_WRITERS.md#a-prospective-stationary-construction-and-its-capacity-controls)

- <a id="limits-of-this-stationary-gate"></a>[Limits of this stationary gate](nodal/FORCED_WINDING_AND_WRITERS.md#limits-of-this-stationary-gate)

- <a id="40-static-phase-compensation-and-the-native-direction-obstruction"></a>[40. Static phase compensation and the native direction obstruction](nodal/FORCED_WINDING_AND_WRITERS.md#40-static-phase-compensation-and-the-native-direction-obstruction)

- <a id="a-different-phase-geometry-can-compensate-the-lost-capacity-source"></a>[A different phase geometry can compensate the lost capacity source](nodal/FORCED_WINDING_AND_WRITERS.md#a-different-phase-geometry-can-compensate-the-lost-capacity-source)

- <a id="the-native-phase-update-moves-in-the-opposite-direction"></a>[The native phase update moves in the opposite direction](nodal/FORCED_WINDING_AND_WRITERS.md#the-native-phase-update-moves-in-the-opposite-direction)

- <a id="two-isolated-native-segments-confirm-the-direction"></a>[Two isolated native segments confirm the direction](nodal/FORCED_WINDING_AND_WRITERS.md#two-isolated-native-segments-confirm-the-direction)

- <a id="an-exact-invariant-source-class-closes-this-native-p3-route"></a>[An exact invariant source class closes this native P3 route](nodal/FORCED_WINDING_AND_WRITERS.md#an-exact-invariant-source-class-closes-this-native-p3-route)

- <a id="41-geometric-identity-and-the-undefined-global-phase-target"></a>[41. Geometric identity and the undefined global phase target](nodal/FORCED_GEOMETRIC_IDENTITY.md#41-geometric-identity-and-the-undefined-global-phase-target)

- <a id="local-coherent-geometry-can-have-zero-global-phase-order"></a>[Local coherent geometry can have zero global phase order](nodal/FORCED_GEOMETRIC_IDENTITY.md#local-coherent-geometry-can-have-zero-global-phase-order)

- <a id="a-symmetric-phase-multiset-cannot-select-one-covariant-direction"></a>[A symmetric phase multiset cannot select one covariant direction](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-symmetric-phase-multiset-cannot-select-one-covariant-direction)

- <a id="any-selected-global-target-changes-the-uniform-twist-orbit"></a>[Any selected global target changes the uniform-twist orbit](nodal/FORCED_GEOMETRIC_IDENTITY.md#any-selected-global-target-changes-the-uniform-twist-orbit)

- <a id="native-policy-and-represented-arithmetic-have-separate-scope"></a>[Native policy and represented arithmetic have separate scope](nodal/FORCED_GEOMETRIC_IDENTITY.md#native-policy-and-represented-arithmetic-have-separate-scope)

- <a id="42-a-geometric-identity-contract-for-the-retained-weighted-c5"></a>[42. A geometric-identity contract for the retained weighted C5](nodal/FORCED_GEOMETRIC_IDENTITY.md#42-a-geometric-identity-contract-for-the-retained-weighted-c5)

- <a id="declared-state-law-and-identity-family"></a>[Declared state, law and identity family](nodal/FORCED_GEOMETRIC_IDENTITY.md#declared-state-law-and-identity-family)

- <a id="a-lifted-shape-coordinate-separates-rotation-from-deformation"></a>[A lifted shape coordinate separates rotation from deformation](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-lifted-shape-coordinate-separates-rotation-from-deformation)

- <a id="form-mean-and-the-strength-of-the-persistence-claim"></a>[Form mean and the strength of the persistence claim](nodal/FORCED_GEOMETRIC_IDENTITY.md#form-mean-and-the-strength-of-the-persistence-claim)

- <a id="evidence-and-remaining-premises"></a>[Evidence and remaining premises](nodal/FORCED_GEOMETRIC_IDENTITY.md#evidence-and-remaining-premises)

- <a id="43-a-positive-local-response-supplies-the-cycle-restoring-mechanism"></a>[43. A positive local response supplies the cycle restoring mechanism](nodal/FORCED_GEOMETRIC_IDENTITY.md#43-a-positive-local-response-supplies-the-cycle-restoring-mechanism)

- <a id="admitted-class-and-the-role-of-each-assumption"></a>[Admitted class and the role of each assumption](nodal/FORCED_GEOMETRIC_IDENTITY.md#admitted-class-and-the-role-of-each-assumption)

- <a id="acute-domain-preservation-and-an-exact-dissipation-identity"></a>[Acute-domain preservation and an exact dissipation identity](nodal/FORCED_GEOMETRIC_IDENTITY.md#acute-domain-preservation-and-an-exact-dissipation-identity)

- <a id="a-restored-shape-need-not-keep-the-same-phase-offset"></a>[A restored shape need not keep the same phase offset](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-restored-shape-need-not-keep-the-same-phase-offset)

- <a id="existing-current-and-argument-pressure-belong-to-the-class"></a>[Existing current and argument pressure belong to the class](nodal/FORCED_GEOMETRIC_IDENTITY.md#existing-current-and-argument-pressure-belong-to-the-class)

- <a id="executable-comparison-controls-and-failure-boundaries"></a>[Executable comparison, controls and failure boundaries](nodal/FORCED_GEOMETRIC_IDENTITY.md#executable-comparison-controls-and-failure-boundaries)

- <a id="44-reusing-phase-geometry-and-capacity-energy-without-adding-a-law"></a>[44. Reusing phase geometry and capacity energy without adding a law](nodal/FORCED_GEOMETRIC_IDENTITY.md#44-reusing-phase-geometry-and-capacity-energy-without-adding-a-law)

- <a id="curvature-reconstructs-the-centered-phase-shape-on-the-admitted-cycle"></a>[Curvature reconstructs the centered phase shape on the admitted cycle](nodal/FORCED_GEOMETRIC_IDENTITY.md#curvature-reconstructs-the-centered-phase-shape-on-the-admitted-cycle)

- <a id="an-augmented-conserved-mean-distinguishes-the-pressure-response-candidate"></a>[An augmented conserved mean distinguishes the pressure-response candidate](nodal/FORCED_GEOMETRIC_IDENTITY.md#an-augmented-conserved-mean-distinguishes-the-pressure-response-candidate)

- <a id="one-capacity-roughness-controls-two-forcing-channels"></a>[One capacity roughness controls two forcing channels](nodal/FORCED_GEOMETRIC_IDENTITY.md#one-capacity-roughness-controls-two-forcing-channels)

- <a id="the-runtime-and-clock-boundaries-are-observable"></a>[The runtime and clock boundaries are observable](nodal/FORCED_GEOMETRIC_IDENTITY.md#the-runtime-and-clock-boundaries-are-observable)

- <a id="45-a-capacity-interval-gives-a-prospective-winding-retention-tube"></a>[45. A capacity interval gives a prospective winding-retention tube](nodal/FORCED_GEOMETRIC_IDENTITY.md#45-a-capacity-interval-gives-a-prospective-winding-retention-tube)

- <a id="scope-and-the-bootstrap-from-the-actual-event-endpoint"></a>[Scope and the bootstrap from the actual event endpoint](nodal/FORCED_GEOMETRIC_IDENTITY.md#scope-and-the-bootstrap-from-the-actual-event-endpoint)

- <a id="phase-deformation-source-and-form-contrast-bounds"></a>[Phase deformation, source and form-contrast bounds](nodal/FORCED_GEOMETRIC_IDENTITY.md#phase-deformation-source-and-form-contrast-bounds)

- <a id="one-frozen-execution-with-the-actual-default-gate"></a>[One frozen execution with the actual default gate](nodal/FORCED_GEOMETRIC_IDENTITY.md#one-frozen-execution-with-the-actual-default-gate)

- <a id="46-absolute-form-a-finite-bound-and-an-interval-law-obstruction"></a>[46. Absolute form: a finite bound and an interval-law obstruction](nodal/FORCED_GEOMETRIC_IDENTITY.md#46-absolute-form-a-finite-bound-and-an-interval-law-obstruction)

- <a id="fixed-observation-weights-separate-mean-drift-from-coordinate-changes"></a>[Fixed observation weights separate mean drift from coordinate changes](nodal/FORCED_GEOMETRIC_IDENTITY.md#fixed-observation-weights-separate-mean-drift-from-coordinate-changes)

- <a id="prescribed-capacity-and-event-mean-balance"></a>[prescribed-capacity-and-event-mean-balance](nodal/FORCED_GEOMETRIC_IDENTITY.md#prescribed-capacity-and-event-mean-balance)

- <a id="prescribed-capacity-changes-distinguish-form-charge-from-its-mean"></a>[Prescribed capacity changes distinguish form charge from its mean](nodal/FORCED_GEOMETRIC_IDENTITY.md#prescribed-capacity-changes-distinguish-form-charge-from-its-mean)

- <a id="a-no-reset-support-event-must-retain-its-changed-weights"></a>[A no-reset support event must retain its changed weights](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-no-reset-support-event-must-retain-its-changed-weights)

- <a id="the-complete-scalar-field-has-a-finite-prefix-enclosure"></a>[The complete scalar field has a finite-prefix enclosure](nodal/FORCED_GEOMETRIC_IDENTITY.md#the-complete-scalar-field-has-a-finite-prefix-enclosure)

- <a id="a-locked-phase-can-still-drive-the-uniform-epi-mode"></a>[A locked phase can still drive the uniform EPI mode](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-locked-phase-can-still-drive-the-uniform-epi-mode)

- <a id="47-actual-capacity-feedback-reduces-the-locked-source-without-cancelling-it"></a>[47. Actual capacity feedback reduces the locked source without cancelling it](nodal/FORCED_GEOMETRIC_IDENTITY.md#47-actual-capacity-feedback-reduces-the-locked-source-without-cancelling-it)

- <a id="one-signed-balance-connects-capacity-averaging-to-phase-readjustment"></a>[One signed balance connects capacity averaging to phase readjustment](nodal/FORCED_GEOMETRIC_IDENTITY.md#one-signed-balance-connects-capacity-averaging-to-phase-readjustment)

- <a id="fresh-default-admission-gives-a-nontrivial-first-write"></a>[Fresh default admission gives a nontrivial first write](nodal/FORCED_GEOMETRIC_IDENTITY.md#fresh-default-admission-gives-a-nontrivial-first-write)

- <a id="a-smaller-capacity-contrast-is-not-a-source-compatibility-lyapunov-law"></a>[A smaller capacity contrast is not a source-compatibility Lyapunov law](nodal/FORCED_GEOMETRIC_IDENTITY.md#a-smaller-capacity-contrast-is-not-a-source-compatibility-lyapunov-law)
