# Regional phase/amplitude measurement contract

**Status: 2026-09-26; specification complete, physical admission `not_admitted`,
physical prediction `not_tested`; exact-model data search parked after the
bounded review below.** This is the observation annex for the
[derived phase/amplitude candidate](../nodal/DERIVED_FORM_PHASE.md#unequal-capacity-mode-selection-and-a-finite-phase-only-discriminator).
The [execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns scheduling;
the [existing source inventory](PASSIVE_TRANSPORT_PROTOCOL.md#terrestrial-source-review-2026-09-21)
owns general dataset descriptions. This document owns the candidate-specific
measurement map, resolution criterion and source admission decision. It does
not add a simulator, a physical law or a second task queue.

## 1. What would be measured

The candidate has six scalar states: two equally oriented directed triangles,
with reciprocal links between corresponding vertices. All twelve directed
edges have unit relative weight. In the engine's row convention, `i -> j`
means node `i` reads node `j`; this must not be confused with a physical
material-flow arrow. Each row reads two neighbors. Pressure is only
`p_i=(x_next+x_cross)/2-x_i`, capacity is fixed at `a>0` or `b>0` within
each triangle, and `dx_i/dt=nu_i*p_i`. There are no operator events, Gamma
terms, other pressure channels or independently evolving primitive phases.

An admissible observation would supply six simultaneous scalar channels,
fixed channel-to-node labels and their directed connection map. Independently
calibrated sensor offsets and gains must recover a **common scalar chart**;
they may not be selected to make the predicted phase appear. One conditional
electrical chart is

\[
\widehat V_i=(y_i-\beta_i)/g_i,\qquad
\widehat x_i=(\widehat V_i-V_{\rm ref})/V_*,\qquad
t=(s-s_0)/\tau_*.
\]

Here `y` is the recorded sensor output; `beta_i,g_i` are independently
calibrated instrument offset/gain with `g_i!=0`, `V_ref` is a common reference,
`V_*>0` is a common scale, and `s` is physical time in seconds.
`tau_*>0` is a fixed number of seconds per declared structural time unit.
A measured temperature or concentration could use an analogous chart, but
would need its own constitutive and support admission. This voltage map is
a hypothesis for a suitable apparatus, not evidence that such data exist.

For each ordered triangle, apply the already derived orthonormal contrasts:

\[
\mu=(x_0+x_1+x_2)/3,\quad
u=(x_0-x_1)/\sqrt2,\quad
v=(x_0+x_1-2x_2)/\sqrt6,\quad z=u+iv.
\]

Retain both means and both Cartesian contrasts. Where amplitudes are resolved,
report `r_0=abs(z_0)`, `r_1=abs(z_1)`, `q=r_1/r_0`, and
`delta=Arg(z_1*conj(z_0))`. This is a **spatial form angle**, not a Hilbert
phase of one temporal signal, primitive `theta`, electrical superconducting
phase, or a tetrad replacement. A common calibrated offset cancels from each
contrast; unequal calibration errors do not. Fixed node ordering determines
orientation and may not be permuted after evaluation to improve agreement.
No future-dependent filtering, phase unwrapping or numerical differentiation
is needed for this instantaneous observation.

## 2. Constitutive and clock evidence

The nodal product alone does not identify its factors. An apparatus must
independently justify the stated pressure row and absence or bounded effect
of additional state, leakage, forcing and delays over the declared interval.
One conditional linear realization would have a separately verified law

\[
S_i\frac{dV_i}{ds}
 =G(V_{\rm next}+V_{\rm cross}-2V_i),\qquad
\nu_i=2\tau_*G/S_i.
\]

`S_i` is physical storage (capacitance in an electrical realization), not
TNFR capacity itself. `G` is the equal effective directed transfer coefficient;
`G/S_i` has units of inverse seconds. Both the equality of links and the
two regional rate classes require independent evidence and error bounds.
Measured components or separate calibration preparations may determine these
rates; fitting them to the reserved phase response is prohibited. A calibrated
effective rate alone does not identify `G` and `S_i` separately. Likewise
responses in seconds determine `a/tau_*` and `b/tau_*`, not an independent
absolute structural clock: fix `tau_*` before calibration instead of fitting
redundant clock and capacity scales.

This equation is a prospective **apparatus requirement**, not a claim that
reciprocal resistors implement it. The
[reciprocity obstruction](../nodal/DERIVED_FORM_PHASE.md#reciprocity-boundary-for-a-physical-realization)
proves that a passive reciprocal first-order scalar RC/thermal network with
positive storage cannot realize this generator even through an invertible
linear state chart. Its spectrum is real, while the directed model's contrast
spectrum is complex. Nonreciprocal circuitry or directed transport would need
its own justified reduction and recorded external operating conditions.
An active apparatus may supply energy while its effective state equation is
homogeneous; that is not autonomous substrate creation or a derived TNFR
energy source. No apparatus construction is scheduled for the computer-only
research route.

The graph and its generator must be specified from wiring/transport evidence,
not inferred solely from phase correlation. Removing unobserved nodes or
selecting six channels from a larger graph requires an exact closed reduction
or an independently bounded memory/error term. The
[existing memory owner](../DERIVED_EPI_MEMORY.md) supplies that distinction;
it does not certify closure for a new dataset.

## 3. Sensor, amplitude and phase uncertainty

Use the joint-error conventions already specified by
[`P2MeasurementBounds`](../../src/tnfr/validation/p2_transport.py): calibrated
coordinate error includes gain/offset error, digitization and any bounded
channel skew. Two timestamp errors contribute to an elapsed interval;
timestamp accuracy alone does not make different channels simultaneous.
Its executable two-node enclosure is **not** a six-node predictor and is not
reused outside that domain.

For one triangle suppose `x_hat=x+e`, with simultaneous deterministic bounds
`abs(e_i)<=epsilon_i`. Writing the contrast map as `z=U^T*x` in real two-vector
form gives `U^T*U=I` and `U*U^T=I-11^T/3`. Consequently

\[
|\widehat z-z|\le\rho,\qquad
\rho=\max_{\sigma_i\in\{-1,1\}}
 \left\|U^T(\sigma_i\epsilon_i)_i\right\|_2
 \le\sqrt{\sum_i\epsilon_i^2}.
\]

The finite maximum is valid because a convex norm on a box attains a maximum
at a vertex. For a common node bound `epsilon`, the exact box radius is
`rho=sqrt(8/3)*epsilon`: a mixed-sign corner has squared contrast `8*epsilon^2/3`,
whereas a common-offset corner has zero contrast. This uses simultaneous
channel bounds, not an assumption of independent Gaussian sensor noise.

If `r_hat=abs(z_hat)>rho`, the disk around the observation excludes zero, and

\[
r\in[\widehat r-\rho,\widehat r+\rho],\qquad
d_{S^1}(\widehat\psi,\psi)\le
 \arcsin(\rho/\widehat r).
\]

The angular bound follows from the tangent from the origin to that disk.
Relative-angle error is at most the sum of the two regional bounds, capped
at `pi`. The ratio lies in
`[(r_hat_1-rho_1)/(r_hat_0+rho_0), (r_hat_1+rho_1)/(r_hat_0-rho_0)]`
when both lower amplitudes are positive. If either disk contains zero, the
corresponding phase and the relative-angle comparison are unavailable.
A ratio bound still exists when the denominator disk excludes zero: use
`max(0,r_hat_1-rho_1)` for its lower numerator. If the denominator disk contains
zero, no uniformly defined finite ratio bound follows. Retain Cartesian
observations and report the separate availability of angle and ratio.
Neither zeroing an unavailable angle nor dropping low-amplitude endpoints
after evaluation is permitted.

A design budget must use a justified lower bound on the future observed
amplitudes, including initial-state, parameter, clock and observation errors;
nominal predicted amplitudes alone do not guarantee resolution. Capacity and
clock uncertainty must be propagated jointly through the declared fine
generator. Independent solver agreement is not a rigorous enclosure.
Any statistical alternative must predeclare joint coverage over the complete
comparison; per-sample standard deviations do not imply this deterministic
bound. No sensor accuracy has yet been admitted for this candidate.

## 4. Frozen finite prediction and decision

Use the full six-node initial-state observation and the independently fixed
rate/clock information to forecast both models before reading later responses.
The comparator freezes `q(t)=q_0`, where `q_0` is the **same initial amplitude
ratio**, with its uncertainty, used by the full model. Its initial ratio
interval must have a strictly positive lower bound and a finite upper bound,
since the angular comparator uses both `q_0` and `1/q_0`:

\[
\dot\delta_{\rm fr}
 =\frac{\sqrt3}{4}(a-b)-\frac12(b/q_0+a q_0)\sin\delta_{\rm fr}.
\]

The nominal equal-amplitude, quadrature preparation gives the previously
derived acceleration difference, but a real initial record must not be forced
to `q_0=1,delta_0=pi/2`. No subsequent measured amplitude is fed into either
forecast. The main observable is the **finite circular phase separation**,
with amplitudes and complete nodal response retained as checks of the full
model. Estimating a second derivative from the reserved signal is unnecessary.

Let the two nominal phase forecasts have independently justified error radii
`B_full` and `B_fr`, including initialization, parameters, clock, admitted
model-reduction error and numerical error. Let `B_obs` bound the relative-angle
observation error. A sufficient robust resolution condition is

\[
d_{S^1}(\delta_{\rm full},\delta_{\rm fr})
 >B_{\rm full}+B_{\rm fr}+2B_{\rm obs}.
\]

This makes the two possible observation tubes disjoint. It is a sufficient
design criterion, not evidence that either model is true. Predeclare finite
times, run identities, initial-state admission and joint uncertainty before
opening reserved responses. At scoring, use circular distances to the two
tubes: full-compatible and comparator-incompatible favors the full reduction
over that approximation; both compatible is inconclusive; full-incompatible
rejects the tested realization or one of its declared premises. Also check
the full predicted scalar traces; a coincidental phase match alone does not
admit the nodal model. Pointwise compatibility with outer error tubes need
not imply that one joint parameter/preparation realization explains the whole
record; it means only that this necessary check did not reject it.
An unresolved amplitude yields an unavailable angular
test, not support for either model. Reserve independent preparations and
retain all declared failures and missing observations.

**Scale of the already retained mathematical example.** At `(a,b)=(1,2)`,
`z(0)=(1,i)/16`, and `t=0.5`, the existing finite control predicts a phase
gap of about `0.05947` radians (`3.407` degrees), with full-model amplitudes
about `0.036327` and `0.024267` in its chosen EPI chart. This gives a prospective
resolution target; it is not `0.5` physical seconds without a clock bridge.
For illustration, if future observed amplitudes are independently bounded
below by `0.036` and `0.024`, a joint node error `epsilon<=0.0001` gives
`B_obs<=0.011341` radians and leaves about `0.03678` radians for the sum of
the two forecast error radii. Those lower amplitudes and errors are hypothetical
requirements, not instrument specifications or measured performance. Decay
eventually destroys angular resolution; waiting indefinitely for the limiting
lag is not the proposed test. This distinguishes a full model from a lossy
approximation, not TNFR from mathematically equivalent external physics.

## 5. Existing-source admission decision

The bounded review on 2026-09-26 rechecked source descriptions and apparatus
equations, not reserved arrays. The earlier
[inventory](PASSIVE_TRANSPORT_PROTOCOL.md#terrestrial-source-review-2026-09-21)
retains licenses, archive sizes and repeat identities. No signal archive was
downloaded, no new response scored and no calibration changed in this review.
Published qualitative summaries are prior exposure, not blind evidence.

| Existing source | Finding for this particular candidate | Decision |
| --- | --- | --- |
| [TCLab apparatus/model](https://dowlinglab.github.io/pyomo-doe/notebooks/tclab-model/) | Two measured temperatures; heater states, supplied inputs and ambient exchange enter its model. It does not provide the six measured scalar states or declared directed generator. | Not admitted for this prediction. Preserve the separate [frozen exploratory comparison](TCLAB_EXPLORATORY_PROTOCOL.md); its scored files are not new holdouts. |
| [NED3-008 thermal record and README](https://datadryad.org/dataset/doi:10.5061/dryad.mcvdnckck) | Six thermocouple channels serve a steady-state conductivity apparatus. Channel count does not establish six closed storage nodes, the required directed support or capacities; fresh-conversion timing remains unresolved. | Not admitted. A reciprocal scalar conduction realization would additionally face the proved spectral obstruction. This is not a finding about uninspected transient arrays. |
| [Rössler record](https://zenodo.org/records/3521009), [primary circuit equations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6961064/) | The measured second coordinate has hidden-state and local circuit terms besides diffusive coupling. One observed coordinate per three-state oscillator does not close the specified scalar law. Nominal acquisition/filter specs do not supply its state or clock bridge. | Not admitted. Neither temporal phase extraction nor selecting six of the 28 channels removes those missing terms or exterior couplings. |
| [Analog learning network record and README](https://datadryad.org/dataset/doi:10.5061/dryad.8w9ghx3vx) | Published 4x4 free/clamped node snapshots accompany changing gate voltages and learning steps; the stated continuous scope record contains time, input voltage and output voltage. The documented schema does not establish simultaneous six-node free evolution under the fixed directed law. | Not admitted. Gate storage is not an independently identified nodal capacity. Admitting another driven/adaptive model would be a separate decision. |

The conclusion is restricted: **none of the existing inventory admits this
particular phase/amplitude test**. It does not show that no suitable public
dataset exists, that these sources are scientifically useless, or that TNFR
is physically refuted. The missing bridge is both constitutive (known closed
nonreciprocal scalar dynamics) and observational (joint state, rate, clock and
uncertainty evidence). Six sensor traces or oscillatory-looking signals alone
would not close it. The protocol is ready to assess a matching source; the
physical objective remains open.

The subsequent [bounded directed-source review](PASSIVE_TRANSPORT_PROTOCOL.md#directed-source-decision-2026-09-26)
examined six new studies and admitted none for this exact test. That closes
the scheduled search and parks this physical experiment; it does not discard
the conditional theorem or its observation/error machinery. MOSTR is retained
as a deferred three-tank transport lead under the user's collective-emergence
priority. Its directed open
cascade has different boundaries and spectrum and cannot inherit the present
phase-locking prediction. The source inventory owns that candidate's metadata
and missing input/measurement dependencies; the plan owns the next action.
