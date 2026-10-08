# Sine pattern geometry, formation and recovery

Relative patterns and origins, composition, equilibrium and symmetry, prepared formation, recovery/capture, controlled slow references and budget exclusions.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-two-port-readout"></a>
### Generate a complete-flow local readout under a declared protocol

Use `bound_sine_two_port_readout` from
[`relational_sine_two_port_readout`](../../../src/tnfr/physics/relational_sine_two_port_readout.py)
to enclose an independently prepared full-state response. Its
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-readout)
fixes the eighteen-node support, complete sine law, held capacities,
fast clock `tau=(1023/1024)*t` and local phase dipole. It takes these five
mandatory inputs:

| Input | Meaning |
| --- | --- |
| `initial_form_bounds` | Eighteen finite endpoint pairs for the pre-event signed form coordinates |
| `initial_phase_bounds` | Eighteen finite endpoint pairs for the pre-event continuous phase lifts, in radians |
| `phase_increment` | Supplied amplitude in `(0,1]`, applied at node four and with opposite sign at node five |
| `probe_duration` | One positive fast-time step, at most one |
| `order` | Fixed ordinary integer Taylor order from one through sixteen |

Declare source, event, observation and numerical budget before a reserved
calculation. Exact fractions preserve their mathematical values; the shared
interval backend adds outward numerical enclosures. Supply all coordinates,
including common means and residuals, rather than replacing a prepared
state with an equilibrium. This guide does not execute the
[reserved inference protocol](../../../theory/nodal/SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference-protocol).
Its source and producing code must be archived before the first response.

Check `.admitted` before using the returned `true_increment_bounds`.
An unavailable result retains `failed_tube` and its reasons; do not turn
its last tube or a midpoint into a successful endpoint. A successful
`step` retains all thirty-six endpoint coordinates, the full initial box,
strict Picard tube, Taylor coefficients and remainder. The
[direct source-box proof](../../../theory/nodal/SINE_TWO_PORT_INFERENCE.md#a-separate-full-state-response-certificate)
explains why coefficients on the initial box and derivatives on the whole
tube enclose the complete evolution.

Keep `baseline_readout_bounds` and `endpoint_readout_bounds` with the
raw response. Use `true_increment_bounds` for the same-state change:
it cancels the common initial coordinate symbolically before arithmetic.
The producer contains no sensor gain, offset or noise. Apply a separately
declared held observation model, retain both recorded readings and their
error allowances, and pass only the permitted observed increment and
public calibration to the [inverse](#sine-two-port-inference).
The source coordinates and forward diagnostics belong to the response
audit, not to that inverse request. A common offset cancels only under
its held-offset premise.

The report proves a finite conditional flow enclosure. Global smoothness
does not supply acute-chart retention, recovery or a measurement bridge.
`to_dict()` and the shared SDK export retain exact rational endpoints under
schema `tnfr.sine-two-port-readout.v1`; saved reports remain separate from
source admission and provenance authentication.

<a id="sine-two-port-inference"></a>
### Interpret a calibrated local response as a geometry constraint

Use `infer_sine_two_port_geometry` from
[`relational_sine_two_port_inference`](../../../src/tnfr/physics/relational_sine_two_port_inference.py)
only after independently admitting the full-state family, complete sine law,
supplied phase pulse, elapsed structural clock and observation calibration
described by its [contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-inference).
Supply a prior range for the donor bulk angle and receiver short angle,
full-state error radii, the recorded increment and its per-reading error,
positive gain bounds, and a fixed refinement budget. Exact fractions retain
their values; rounded observations describe their represented values.

This calculation constrains a genuinely variable, generally nonstationary
geometry. It does not infer the compatible equilibrium by repeating the
assumptions that already determine it. Its observation is one local increment
on the joined graph, not the earlier joined-minus-unjoined contrast; do not
substitute the saved dipole contrast for that input.

Read the nominal-angle enclosure and the actual pre-probe long-arc mean enclosure
separately. Initial phase uncertainty makes them different. A nonempty outer
interval is a necessary constraint, not proof of an underlying state that
produces the reading. An empty result challenges the joint premises; it
does not identify an individual failed calibration or law. No trajectory,
equilibrium search or old frozen producer runs when constructing the report.

The [theorem](../../../theory/nodal/SINE_TWO_PORT_INFERENCE.md) supplies the
finite response and remote-current error proof. The separately frozen
[three-case assessment](../../../theory/nodal/SINE_TWO_PORT_INFERENCE.md#sine-two-port-inference-result)
tests this inverse using complete-flow responses and a held sensor calibrated
from separate known references. Its first evaluation passes all forty-three
conditions. The inverse worker receives only its recorded JSON public packet;
hidden geometry, source coordinates and realized reading errors remain in
the posterior audit. This is a software information boundary, not physical
calibration or cryptographic blindness.

Inspect the [saved record](../../assets/sine_formed_classes/two-port-inference-v1.json)
without rerunning the producer. Each `cases` entry retains `response`,
`readings`, `public_packet`, `inverse_outputs`, `source_audit` and
`stopping_rule`. The three inverse outputs are `primary`, `broad_gain` and
`false_prior`; the complete phase-blind comparison has its own recorded
bound. Keep exact fractional endpoints when subtracting readings: a large
common offset and a tiny increment can lose the signal in rounded displays.
The [protocol](../../assets/sine_formed_classes/two-port-inference-v1.protocol.json),
[source archive](../../assets/sine_formed_classes/two-port-inference-v1.source.zip)
and [manifest](../../assets/sine_formed_classes/two-port-inference-v1.manifest.json)
preserve source, budgets and verdicts. The
[execution plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns subsequent admission.

<a id="sine-two-port-dipole"></a>
### Inspect a common interior phase probe and its finite warmup

The [interior-dipole protocol](../../../theory/nodal/SINE_TWO_PORT_DIPOLE.md#sine-two-port-dipole-protocol)
uses a new continuation of the original captured family. It retains all
thirty-six initial form and phase errors and does not replay the earlier
uniform form pulse. After a proved finite warmup, both the joined composite
and an independently evolved unjoined control receive the same phase jump
`theta_plus=theta_minus+a*(e_4-e_5)`, with form unchanged.

The observation is the increment in local form difference:
`(x_4-x_5)_after-(x_4-x_5)_before`. The coefficient vector, affected
local edges and degrees agree in both supports. This does not make their
whole-network responses identical; the proof retains a separate finite
remainder for each full flow. Four scalar readings form the recorded
joined-minus-unjoined contrast, with no assumed cancellation of their errors.

The required inputs to `assess_sine_two_port_dipole` are:

| Primitive | Frozen value | Meaning |
| --- | --- | --- |
| `warmup_duration` | `285934809600000` | Extra fast structural time after the existing finite capture handoff |
| `form_radius` | `2^-40` | Desired original-form norm bound after warmup |
| `phase_radius` | `2^-40` | Desired target-phase norm bound after warmup, in radians |
| `phase_increment` | `2^-12` | Exact phase dipole amplitude |
| `probe_duration` | `2^-10` | Elapsed fast structural time between the jump and readout |
| `readout_error_bound` | `2^-50` | Error bound for one scalar local form-difference reading |
| `contrast_threshold` | `2^-38` | Strict lower threshold for the recorded contrast |
| `work_allowance` | `2^-21` | Storage-work allowance for each sine-law phase jump |

These powers denote exact rationals; Python callers can use
`Fraction(1, 2**40)` and the corresponding denominators. The returned
`SineTwoPortDipole` uses schema `tnfr.sine-two-port-dipole.v1`.
Its warmup is conditional on already trapped source families. The joined
norm uses its global conserved means; the unjoined norm combines both
rings after removing each ring's own means. The full-family source handoff
is a separate obligation, and the small endpoint radii do not narrow the
original preparation or replace it by an equilibrium.

Read the [API contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-dipole)
before interpreting response, heat-control, work and recovery flags.
The [retained result](../../../theory/nodal/SINE_TWO_PORT_DIPOLE.md#sine-two-port-dipole-result)
passes all thirteen conditions in its first frozen assessment. Its recorded
contrast is enclosed between approximately `8.3951e-12` and `9.6232e-12`,
above threshold `2^-38`; the phase-blind recorded bound is about
`7.1054e-15`. These are certified structural-model bounds, not measured
laboratory responses. The
[execution plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#sine-two-port-dipole-admission)
owns its closed status. Inspect the saved record from the repository root:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/two-port-dipole-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-two-port-dipole.v1"
dipole = saved["report"]

def exact(value):
    return Q(value["numerator"], value["denominator"])

assert dipole["status"] == "certified_dipole"
assert exact(dipole["warmup_duration"]) == 285934809600000
assert dipole["warmup_certified"]
assert exact(dipole["warmup_form_margin"]) > 0
assert exact(dipole["warmup_phase_margin"]) > 0
contrast_lower = exact(dipole["recorded_contrast_bounds"]["lo"])
assert contrast_lower > Q(1, 2**38)
assert contrast_lower > exact(dipole["phase_blind_recorded_contrast_upper_bound"])
assert dipole["heat_warmup_certified"] and dipole["heat_control_excluded"]
assert all(exact(value) >= 0 for value in dipole["work_allowance_margins"])
assert all(exact(value) > 0 for value in dipole["capture_storage_margins"])
assert dipole["identity_certified"] and dipole["recovery_certified"]
assert saved["original_control_handoff"]["admitted"]
assert saved["source_handoff"]["report"]["numerical_execution_replayed"] is False
assert len(saved["frozen_stopping_rule"]) == 13
assert all(saved["frozen_stopping_rule"].values())
assert saved["frozen_stopping_rule_passed"]
print(dipole["status"], len(saved["frozen_stopping_rule"]))
```

These checks read the
[retained response](../../assets/sine_formed_classes/two-port-dipole-v1.json)
without invoking any assessor or trajectory. The
[protocol](../../assets/sine_formed_classes/two-port-dipole-v1.protocol.json),
[source archive](../../assets/sine_formed_classes/two-port-dipole-v1.source.zip)
and [manifest](../../assets/sine_formed_classes/two-port-dipole-v1.manifest.json)
preserve its first execution. The earlier capture's numerical execution
remains an explicit premise of its rebuilt source handoff; the new warmup
does not independently authenticate that execution. Earlier uniform-probe
export recovery is a separate record.

The separate phase-blind comparison has `x_tau=-A*x` and
`theta_tau=gamma*A*x`, with its own actual support and the same original
preparation. Its phase pulse has zero causal effect on form, while its
remaining form background receives an explicit heat bound. The certified
contrast clears both the declared response threshold and this bounded
alternative. It does not establish uniqueness among all phase-sensitive
laws or a physical measurement bridge.

All durations use `tau=e*t`. The additional warmup starts at
`tau=1025*1023**2*pi**2`; the phase probe and observation continue that
clock without a reset. Both form and phase means remain preserved through
this phase-only intervention. Its work and post-event identity must still
be admitted separately from the earlier no-event capture theorem.

<a id="sine-two-port-probe"></a>
### Inspect a supplied pulse and receiver transmission

The [supplied-probe protocol](../../../theory/nodal/SINE_TWO_PORT_PROBE.md#sine-two-port-probe-protocol)
continues the acquired two-port family at slow time `sigma=1025`.
Its [finite capture handoff](#sine-two-port-capture) retains the original
independent form and phase errors; the reached state is not reset to its
target. A simultaneous form pulse of `1/2048` at all nine donor nodes is
followed by `1/4` of fast structural time under the same complete law.

The readout is the receiver's degree-weighted mean form increment. Compare
it with the same observation rule on two unjoined C9 rings that evolved
from the same original preparation for the same original elapsed time and
receive the same donor pulse. Receiver degree mass is `20` when joined
and `18` when unjoined, so the rule has different coefficient vectors.
The unjoined receiver mean is exactly conserved. This control does not
delete contacts or reset a captured state.
The two before-and-after increments consume four scalar readings, with
independently bounded errors and no assumed cancellation.

The required inputs to `assess_sine_two_port_probe` are:

| Primitive | Reserved value | Meaning |
| --- | --- | --- |
| `form_radius` | `1/8192` | Relative form norm at the actual pre-pulse endpoint, in the full degree metric |
| `phase_radius` | `1/1024` | Relative phase distance from the joint target, in radians and the same metric |
| `pulse_amplitude` | `1/2048` | Exact supplied donor form increment |
| `probe_duration` | `1/4` | Elapsed fast clock `tau=e*t` after the pulse |
| `readout_error_bound` | `1/67108864` | Independent additive error per declared scalar readout |
| `contrast_threshold` | `1/262144` | Strict lower threshold for the recorded joined-minus-unjoined increment |
| `work_allowance` | `1/2000000` | Upper allowance for the supplied storage jump |

The standalone assessment is conditional on its endpoint ball. It returns
`SineTwoPortProbe`, schema `tnfr.sine-two-port-probe.v1`, and separate
response, work and recovery flags; it does not establish source acquisition.
The frozen experiment needs the separately checked finite source handoff as
well. Read the [API contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-probe)
before interpreting the flags. The original frozen attempt executed its
primary assessment but failed during control-report export and saved no
complete response. The
[retained result](../../../theory/nodal/SINE_TWO_PORT_PROBE.md#sine-two-port-probe-result)
passes all ten fixed conditions after a separately archived deterministic
export recovery. Scientific inputs and runtime were unchanged; this is not
a successful first attempt. The
[execution plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#sine-two-port-probe-admission)
owns that status. Inspect the saved response from the repository root:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/two-port-probe-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-two-port-probe.v1"
probe = saved["report"]
handoff = saved["source_handoff"]["report"]

def exact(value):
    return Q(value["numerator"], value["denominator"])

assert probe["status"] == "certified_probe"
assert exact(probe["recorded_contrast_bounds"][0]) > Q(1, 262144)
assert exact(probe["joined_work_bounds"][1]) <= Q(1, 2000000)
assert exact(probe["capture_storage_margin"]) > 0
assert probe["joined_identity_certified"] and probe["recovery_certified"]
assert exact(handoff["endpoint_form_radius"]) < Q(1, 8192)
assert exact(handoff["endpoint_phase_radius"]) < Q(1, 1024)
assert handoff["numerical_execution_replayed"] is False
assert handoff["provenance_authenticated"] is False
assert len(saved["frozen_stopping_rule"]) == 10
assert all(saved["frozen_stopping_rule"].values())
assert saved["frozen_stopping_rule_passed"]
history = saved["evaluation_history"]
assert history["retained_assessment_kind"] == (
    "separately_frozen_export_recovery_recomputation"
)
assert history["scientific_inputs_or_runtime_changed"] is False
assert history["prior_capture_producer_replayed"] is False
print(probe["status"], history["retained_assessment_kind"])
```

These are read-only checks of the
[saved response](../../assets/sine_formed_classes/two-port-probe-v1.json).
The original [protocol](../../assets/sine_formed_classes/two-port-probe-v1.protocol.json),
[source archive](../../assets/sine_formed_classes/two-port-probe-v1.source.zip)
and [failure record](../../assets/sine_formed_classes/two-port-probe-v1.first-attempt.json)
remain separate from the
[export-recovery wrapper](../../assets/sine_formed_classes/two-port-probe-v1.export-recovery.py.txt).
The [manifest](../../assets/sine_formed_classes/two-port-probe-v1.manifest.json)
retains their associations. No producer is invoked by this example.

To admit the retained finite source, use the separate research reader
`audit_sine_two_port_capture_handoff` on
`docs/assets/sine_formed_classes`. It rebuilds the consumed source, target
and endpoint bounds and checks archive associations. Its direct
`SineTwoPortHandoffAudit.to_dict()` uses schema
`tnfr.sine-two-port-handoff-audit.v1`; it is not a generic SDK relational
report. The audit does not replay the archived numerical execution or
authenticate its chronology. In particular, its checked metric chain and
root signs retain the declared Taylor-execution premise. A conditional
endpoint probe report and this finite source admission have different roles.

The pulse shifts the global form mean by `1/4096`, while phase mean and
support remain fixed. The proof therefore needs post-event work and a new
trapping check; the earlier no-event convergence result cannot cross this
jump on its own. Any certified recovery preserves the joint winding identity
and approaches the same shape on the new form-mean leaf.

Transmission on supplied support is the scope of this test. A pure heat
countermodel retains the ideal signal. This result therefore does not select
the sine law or isolate the effect of the acquired internal geometry.
The intervention, structural clock and observation rule remain supplied;
they are not a physical preparation or measurement bridge.

<a id="sine-two-port-capture"></a>
### Inspect the complete same-family capture chain

The [retained capture result](../../../theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture-result)
certifies that the same midpoint-aligned preparation family enters a local
trapping region and converges to the joint equilibrium. All original form
and phase errors remain admitted. The first preserved assessment passed
every numerical and full-state handoff premise of the
[theorem](../../../theory/nodal/SINE_TWO_PORT_CAPTURE.md#sine-two-port-capture).
The [declared protocol](../../assets/sine_formed_classes/two-port-capture-v1.protocol.json)
fixes the six primitives consumed by `assess_sine_two_port_capture` and
schema `tnfr.sine-two-port-capture.v1` for its report. Neither a protocol
alone nor a successful static target report would establish this result.

Read selected fields from the saved response at the repository root:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/two-port-capture-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-two-port-capture.v1"
capture = saved["report"]

def exact(name):
    value = capture[name]
    return Q(value["numerator"], value["denominator"])

assert capture["status"] == "certified_capture"
assert len(capture["reference_steps"]) == 4096
assert exact("validated_reference_duration") == exact("reference_duration") == 1024
assert exact("full_slow_horizon") == 1025
assert exact("reference_minimum_acute_margin") > Q(1, 2048)
assert exact("reference_target_distance_upper_bound") <= Q(1, 2048)
assert exact("endpoint_excess_storage_upper_bound") < Q(1, 648000)
assert exact("capture_storage_margin") > 0
assert capture["capture_certified"]
assert saved["frozen_stopping_rule_passed"]
print(capture["status"], capture["unavailable_reasons"])
```

These are read-only checks of the retained record, not a new assessment.
The [saved response](../../assets/sine_formed_classes/two-port-capture-v1.json)
includes every compact reference-step certificate. The
[source archive](../../assets/sine_formed_classes/two-port-capture-v1.source.zip)
and [manifest](../../assets/sine_formed_classes/two-port-capture-v1.manifest.json)
retain its producing implementation and content hashes; the
[evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
checks those associations without rerunning the producer.

The nominal gradient reference is validated through slow time `1024` using
an eight-coordinate reconstruction, the shared retained-metric Taylor
kernel, step `1/4`, order eight and at most 4,096 steps. This smaller
reference does not impose symmetry on the actual eighteen-node form-phase
family. The strict reference margin `1/2048` radians and endpoint-distance
allowance `1/2048` radians in the full degree metric are checked separately
from the actual full-flow error.

One additional slow unit is handled analytically, continuing the same
reference and full trajectories to `1025`. Capture then requires the
original form coordinate, the actual phase error and the complete excess
storage to pass their common local barrier. Inspect the entire chain:
a small reference endpoint alone is insufficient, and a partial validated
prefix does not represent the requested horizon.

The protocol, producing source and proof were preserved before the first
evaluation. Inspect its saved result without rerunning the frozen producer.
The successful chain admits the full original preparation family; it does
not impose the reference's reflection symmetry on actual errors. A separate
current-source assessment with an unavailable result must preserve its
failed premise and budget rather than replace the retained first response.
The
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-capture)
separates this conditional capture from contact occurrence, event work and
physical identification.

<a id="sine-two-port-transit"></a>
### Certify a finite deformation from the undeformed pair

The [directional transit theorem](../../../theory/nodal/SINE_TWO_PORT_TRANSIT.md#sine-two-port-directional-transit)
starts from two undeformed uniform twists with nominal form zero and
midpoint-aligned contact gaps `+pi/9`, `-pi/9`. It proves motion under the
complete law while retaining independent errors in all eighteen form and
phase coordinates. The source does not use the solved equilibrium.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_two_port_transit import (
    assess_sine_two_port_transit,
)
from tnfr.sdk import relational_report_to_dict

transit = assess_sine_two_port_transit(
    form_error_radius=Q(1, 65536),
    phase_error_radius=Q(1, 65536),
)
assert transit.energy_budget_admitted
assert transit.error_certified
assert transit.whole_window_acute_certified
assert transit.short_arc_change_lower_bound > Q(1, 32)
assert transit.direction_certified
assert transit.status == "certified_directional_transit"
payload = relational_report_to_dict(transit)
```

The fixed endpoint is slow time `sigma=1/4`, with
`sigma=gamma**2*tau`, `tau=e*t`, and `gamma=1/(1023*pi)`.
This is original structural time `t=261888*pi**2`, not a laboratory clock.
At that endpoint, every admitted donor short gap has contracted by more
than `1/32` radian and every receiver short gap has expanded by more than
`1/32` radian, each compared with its own initial gap. All edges remain
acute throughout this finite window, preserving periods `(2,1,0)`.
The phase-error radius is in radians; the form radius uses the declared
structural form coordinate.

The assessor evaluates analytic bounds, not a numerical trajectory.
Its gradient reference supplies a comparison while the retained mixed
coordinate reconstructs the actual full-state law. Inspect the energy,
bootstrap and direction flags separately: candidate estimates cannot
substitute for certified bounds, and a failed sufficient margin is not
an observed failure of motion. The
[API contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-transit)
specifies those distinctions.

Midpoint alignment is a supplied preparation. In the earlier centered-ring
convention it corresponds to receiver origin `-7*pi/9`, so this is not the
old `1/1000`-origin contact experiment. The finite motion is compatible with
the earlier scalar-storage obstruction: neither that obstruction nor the
new directional certificate decides eventual capture. No contact event,
passive work budget or physical binding is established.

<a id="sine-two-port-handoff-obstruction"></a>
### Check the limit of a scalar storage handoff

The two-port `(2,1)` equilibrium is locally attracting, but that fact does
not place an undeformed pair in its basin. The
[handoff obstruction](../../../theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-handoff-obstruction)
identifies a specific limit of the target-free sector-capture theorem:
even an exact lower bound over all acute boundary faces cannot certify
these sources directly from total storage. This applies to every relative
component origin and includes a stated phase-error neighborhood.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_two_port_compatibility import (
    assess_sine_two_port_handoff_obstruction,
)
from tnfr.sdk import relational_report_to_dict

handoff = assess_sine_two_port_handoff_obstruction(
    phase_error_radius=Q(1, 65536),
)
assert handoff.storage_gap_lower_bound == Q(17, 13824) - 40 * Q(1, 65536)
assert handoff.storage_gap_lower_bound > 0
assert handoff.handoff_obstruction_certified
assert handoff.status == "certified_handoff_obstruction"
payload = relational_report_to_dict(handoff)
```

The radius is a per-node phase-lift error in radians around the isolated
uniform twists; the complete law and both unit contacts remain fixed. Form
coordinates are arbitrary finite signed values, and their nonnegative
storage cannot restore the failed scalar inequality. The assessor constructs
an exact lower-storage boundary witness. It does not integrate a trajectory,
solve for the equilibrium again or change the earlier frozen evidence.

Read `storage_gap_lower_bound` as a margin excluding this proof method.
It is not a prediction that a trajectory reaches the boundary or loses its
identity. A zero or negative conservative margin gives `unavailable`, not
proof of capture. Actual acquisition needs additional control of the
direction and evolution of the full state. Earlier central-port contact
certificates do not apply unchanged to this different support. The
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-handoff-obstruction)
specifies admission, the strict rational threshold and the scope of each flag.

<a id="sine-two-port-compatibility"></a>
### Inspect compatibility at two distinct ports

`assess_sine_two_port_compatibility` admits an implicit acute equilibrium on
two C9 rings joined at local nodes zero and one. Its three required arguments
are `classes`, `outer_refinements` and `inner_refinements`; the
[API contract](../../contracts/relational/SINE_PATTERNS.md#sine-two-port-compatibility)
owns their domains and the fixed complete law. This is a different interface
from the central-port reduction, whose trajectory bounds cannot be reused
unchanged. Read the retained primary and matched control from the repository
root without rerunning either assessment:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/two-port-compatibility-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-two-port-compatibility.v1"
primary = saved["report"]
matched = saved["matched_control"]["report"]

for label, record in (("unequal classes", primary), ("matched classes", matched)):
    print(label, record["classes"], record["status"])
    print("acute", record["acute_geometry_certified"])
    print("local attraction", record["local_attraction_certified"])
    print("undeformed pair compatible", record["uniform_pair_compatible"])
    print(record["unavailable_reasons"])

def first_contact_current(record):
    index = record["geometry"]["edges"].index([0, 9])
    interval = record["edge_current_bounds"][index]
    return tuple(Q(interval[key]["numerator"], interval[key]["denominator"])
                 for key in ("lo", "hi"))

primary_current = first_contact_current(primary)
matched_current = first_contact_current(matched)
assert primary_current[0] > 0
assert matched_current == (Q(0), Q(0))
assert saved["frozen_stopping_rule_passed"]
```

Both [saved certificates](../../assets/sine_formed_classes/two-port-compatibility-v1.json)
have status `certified_compatible`. The unequal-class first contact current
is approximately `0.11461`; the matched current is exactly zero. The unequal
twists deform to satisfy the joint balance; independently rotating their
undeformed copies cannot make both contacts compatible. These are static
geometric distinctions, not an evaluated formation trajectory.

Inspect `bridge_turn_bounds`, `edge_current_bounds` and the correlated affine
turn coefficients together. The intervals enclose one implicit geometry;
their midpoint and arbitrary independent endpoint choices are not exact
equilibria. `full_nodal_residuals_consistent` checks interval consistency,
while exact stationarity rests on the proved root equations and complete
nodal factorization. Matched classes use exact uniform twists, so absent
root brackets are expected rather than missing evidence.

The [theory owner](../../../theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md#sine-two-port-compatibility)
separates this compatibility question from earlier source formation. Local
attraction does not show that the previous isolated preparations reach the
new basin, or that a contact occurs or pays its storage cost. Stationary sine
circulation has zero nodal velocities; it is not sustained nodal motion or
a physical current. The [retained-evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
checks the frozen bundle separately from this read-only inspection.

<a id="sine-port-form-tracking"></a>
### Inspect the sharper form bound and its preserved baseline

`assess_sine_port_form_tracking` uses the same fourteen primitive arguments as
the [all-time relaxation assessment](#sine-port-relaxation). It rebuilds those
premises and applies a separately justified ordered heat comparison, retaining
the same surrogate, nonlinear bridge and channel allowances. The
[API contract](../../contracts/relational/SINE_PATTERNS.md#sine-port-form-tracking)
owns admission and field availability. To inspect the saved experiment from
the repository root without repeating its assessment:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/port-form-tracking-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-port-form-tracking.v1"
record = saved["report"]
baseline = record["baseline_certificate"]

def rational(value):
    return Q(value["numerator"], value["denominator"])

print("baseline", baseline["status"], "new method", record["status"])
new_form_bound = rational(record["all_time_form_error_upper_bound"])
old_form_bound = rational(baseline["all_time_form_error_upper_bound"])
assert new_form_bound < old_form_bound
assert record["all_time_phase_error_upper_bound"] == baseline[
    "all_time_phase_error_upper_bound"
]
print("phase", record["phase_resolution_certified"])
print("form", record["form_resolution_certified"])
print("joint", record["joint_resolution_certified"])
```

The [retained new assessment](../../assets/sine_formed_classes/port-form-tracking-v1.json)
has status `full`. Its form upper bound is approximately `5.66e-8`, below the
unchanged allowance of approximately `1.56e-7`; its original phase bound remains
approximately `2.65e-4`, below `5e-4`. These are uniform error guarantees, not
measured errors. The nested baseline remains `phase_only` and equals the
earlier frozen report body, whose joint stopping criterion remains false.

Inspect `odd_heat_bounds`, `even_heat_bounds` and the explicit bridge-variation
fields for the new method's contributions and strict feedback margins.
Unavailable bounds remain `None`; a valid envelope and each channel's
resolution flag are separate questions. Overall admission also retains the
supplied-work policy. The `*_mean_error_floor` fields remain upper budgets for
possible constant offsets, not unavoidable positive errors.

The [result owner](../../../theory/nodal/SINE_PORT_FORM_TRACKING.md#sine-port-form-tracking)
retains the proof and complete frozen evidence. The
[read-only evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
checks provenance and the unchanged stopping rules separately from this JSON
inspection. This result does not establish practical acquisition time, a
sensor specification or physical identification of the formed components.

<a id="sine-port-relaxation"></a>
### Inspect uniform tracking and separate channel resolution

`assess_sine_port_relaxation` compares the unchanged reduced composition with
the actual fine trajectories for all subsequent uninterrupted times. It takes
the original source/support budgets and a checked normalized spectral gap;
it has no contact-duration argument and accepts no cached certificate. The
[API contract](../../contracts/relational/SINE_PATTERNS.md#sine-port-relaxation)
owns its fourteen primitive arguments. Use the retained record to inspect the
existing experiment without rerunning its assessment, from the repository root:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/port-relaxation-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-port-relaxation.v1"
record = saved["report"]

def rational_or_none(value):
    return None if value is None else Q(value["numerator"], value["denominator"])

print(record["status"], record["unavailable_reasons"])
exact_error_bounds = {}
for channel in ("phase", "form"):
    exact_error_bounds[channel] = rational_or_none(
        record[f"all_time_{channel}_error_upper_bound"]
    )
    print(channel, record[f"{channel}_resolution_certified"])
print(record["resolution_limitations"])
```

Read `all_time_envelopes_certified` separately from the phase, form and joint
resolution flags. The envelopes include initial and generated odd modes and
budgets for possibly nonzero conserved-mean differences. The `*_mean_error_floor`
fields are upper budgets for constant offsets, not measured lower errors.
Phase uses a fraction of the
declared origin span; form uses its own fraction of `gamma` times that span.
Each strict outward margin must pass independently. `phase_only` and
`form_only` retain precisely that qualification; `envelopes_only` still gives
uniform bounds, while `unavailable` names unmet prerequisites. A failed
sufficient resolution test is not a measurement of a large actual error.
`exact_error_bounds` retains the rational upper bounds, or `None` when absent.

The [retained assessment](../../assets/sine_formed_classes/port-relaxation-v1.json)
is `phase_only`: it supplies both all-time envelopes and resolves the frozen
phase allowance, while its form bound does not certify the frozen form
allowance. The joint stopping criterion remains false. This qualified result
does not show that the actual form error exceeds the allowance.

These all-time allowances differ from the short-window composition budget
and the earlier donor-response resolution. Full and surrogate trapping are
both proved before the comparison; recovery alone would not establish their
closeness. The [result owner](../../../theory/nodal/SINE_PORT_RELAXATION.md#sine-port-relaxation)
retains the protocol, proof and channel verdicts. The
[evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
checks source/record consistency separately from reading a JSON. No support
selection, autonomous preparation or physical observation is inferred.

<a id="sine-reduced-port-composition"></a>
### Assemble reduced components with their actual contact degrees

Use `evaluate_sine_port_composition` for a network of unit central contacts.
Each component contributes five forms and five real phase deviations, ordered
as in the [two-component workflow](#sine-reduced-class-ports). A second contact
changes the central degree in both rows; it cannot reuse the old denominator
three unchanged. This instantaneous control needs no formation assessment:

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_port_composition import (
    evaluate_sine_port_composition,
)

forms = [Q(0)] * 15
forms[5] = Q(1)  # Central layer of the middle component.
rows = evaluate_sine_port_composition(
    classes=(1, 2, 1),
    contacts=((0, 1), (1, 2)),
    phase_origins=(Q(0),) * 3,
    forms=forms,
    phase_deviations=(Q(0),) * 15,
)
assert rows.geometry.contact_degrees == (1, 2, 1)
assert rows.form_rate_bounds[5].lo <= -1 <= rows.form_rate_bounds[5].hi
assert rows.form_rate_bounds[5].lo > -Q(4, 3)
assert rows.form_storage == 2
assert rows.storage_rate == -Q(17, 3)
assert rows.network_form_charge_rate == rows.network_phase_charge_rate == 0
```

The bridge sine remains nonlinear. Internal phase exchange is the inherited
class tangent, and the exact reduced storage includes both internal and bridge
terms. Origins are already supplied separately; phase deviations must exclude
them. These rows admit disconnected contacts but do not establish a full-law
trajectory, formation or global recovery.

The separate `assess_sine_port_composition` takes original source, support,
time and budget primitives. It rebuilds the actual unprobed formation family,
then bounds all fine coordinates against the lifted surrogate over the whole
contact window. Inspect `unprobed_handoff`,
`total_approximation_error_upper_bound`, `approximation_margin_bounds`,
`identity_certified`, `work_within_allowance` and `unavailable_reasons` together.
An instantaneous storage identity cannot replace these actual-family checks.
The absolute approximation allowance is not a readout error or an observed
class contrast; connected joined identity and all-time recovery use the full
sine law separately. Consult the
[primitive API contract](../../contracts/relational/SINE_PATTERNS.md#sine-reduced-port-composition)
and [protocol/result owner](../../../theory/nodal/SINE_REDUCED_PORT_COMPOSITION.md#sine-reduced-port-composition)
before a new assessment. Neither reader executes contact in a live graph or
establishes autonomous hierarchy selection or a physical measurement bridge.

The [saved three-component certificate](../../assets/sine_formed_classes/port-composition-v1.json)
can be inspected without running another assessment. Its report has
`status="certified_sine_port_composition"`; the outer `algebraic_control`
records the exact normalization/storage control, and
`frozen_stopping_rule_passed` records the joint frozen rule. The full-state
allowance is a different stopping rule from the fractional receiver-gap
criterion in the two-component result below. The
[retained-evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
checks their byte/source consistency and respective exact stopping criteria;
reading a passing JSON alone does not verify provenance.

<a id="sine-reduced-class-ports"></a>
### Evaluate reduced port rows and inspect their retained certificate

Use `evaluate_sine_reduced_class_ports` to inspect the instantaneous field of
the twenty-coordinate surrogate. Each state row contains five donor values
followed by five receiver values, ordered by the reflection layers
`(4), (3,5), (2,6), (1,7), (0,8)`. The form row and phase-deviation row together
contain twenty coordinates; they are not two scalar component states.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_reduced_class_ports import (
    evaluate_sine_reduced_class_ports,
)
from tnfr.sdk import relational_report_to_dict

rows = evaluate_sine_reduced_class_ports(
    donor_class=1,
    receiver_class=2,
    forms=(Q(0),) * 10,
    phase_deviations=(Q(0),) * 10,
    phase_origin_difference=Q(1, 1000),
)
assert rows.bridge_phase_difference == Q(1, 1000)
assert rows.form_rate_bounds[0].lo > 0
assert rows.form_rate_bounds[5].hi < 0
row_payload = relational_report_to_dict(rows)
```

Zero phase deviations mean each component is at its own reference twist and
declared common origin. The receiver origin is already supplied by
`phase_origin_difference`; do not add it to the receiver deviation entries
again. The [coordinate contract](../../contracts/relational/SINE_PATTERNS.md#sine-reduced-class-ports)
gives the explicit lift. These supplied coordinates evaluate reduced rows;
they do not prove formation, approximate an arbitrary fine state or install
a bridge in a live graph.

The separate `assess_sine_reduced_class_ports` reader rebuilds the original
source families and their error bounds from twelve primitive inputs. Its
retained result can be inspected without calling that reader again. Run the
following from the repository root:

```python
from fractions import Fraction as Q
from pathlib import Path
from tnfr.utils.io import json_loads

saved = json_loads(
    Path("docs/assets/sine_formed_classes/reduced-ports-v1.json").read_text(
        encoding="utf-8"
    )
)
assert saved["schema"] == "tnfr.sine-reduced-class-ports.v1"
record = saved["report"]

def rational(value):
    return Q(value["numerator"], value["denominator"])

assert record["status"] == "certified_reduced_class_ports"
assert all(record["unprobed_handoff"]["handoff_certified_by_class"])
assert record["identity_certified"] and record["work_within_allowance"]
gap_lower = rational(record["recorded_contrast_bounds"]["lo"])
error_ratio = rational(record["error_ratio_upper_bound"])
assert gap_lower > 0
assert error_ratio < rational(record["error_fraction"])
assert rational(record["error_fraction_margin_bounds"]["lo"]) > 0
```

`recorded_contrast_bounds` concerns actual receiver-port form for donor class
two minus donor class one. The error ratio uses that interval's final lower
endpoint and includes reduced evaluation, model discrepancy, preparation and
readout errors. The reduced model's accuracy is finite-window evidence;
whole-family identity and recovery are separate full-law obligations.

Reading the JSON inspects retained assertions; it does not verify its archive
or authenticate chronology. Use the
[retained-evidence audit](../../../TESTING.md#current-checks-and-retained-evidence)
for byte/source consistency, and the
[result owner](../../../theory/nodal/SINE_REDUCED_CLASS_PORTS.md#sine-reduced-class-ports)
for the frozen protocol and scope. A new assessment takes original primitives,
not `rows`, `record` or a cached passing flag. An unavailable assessment keeps
missing actual-family fields as `None`; a successful instantaneous evaluation
does not supply those missing premises. No solver, measured speedup, practical
clock or physical identification is established by this workflow.

<a id="sine-formed-class-contact"></a>
### Compare actual receiver responses after a supplied contact

This reader rebuilds the original C9 formation families and their subsequent
uninterrupted recovery. It compares the same winding-one receiver joined to
donor winding one or two by one central unit bridge. The receiver's common
phase origin is declared from initial preparation; no phase jump or reset
occurs before or at contact.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formed_class_contact import (
    assess_sine_formed_class_contact,
)
from tnfr.sdk import relational_report_to_dict

contact = assess_sine_formed_class_contact(
    formation_time=100,
    relaxation_duration=10**13,
    phase_origin_difference=Q(1, 1000),
    contact_duration=Q(1, 100),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    endpoint_radius=Q(1, 10**32),
    readout_error_bound=Q(1, 10**30),
    radius=Q(1, 12),
    work_allowance=Q(1, 10**6),
    decay_power=512,
)
assert contact.status == "certified_formed_class_contact"
assert all(contact.handoff_certified_by_class)
assert contact.identity_certified and contact.work_within_allowance
assert contact.recorded_contrast_bounds.lo > Q(1, 10**25)
evidence = relational_report_to_dict(contact)
```

All times use `tau=e*t`, with `e=1023/1024`. Contact occurs at
`tau=100+10**13`; the readout is actual form at receiver node `13`, one
hundredth of a scaled unit later. The recorded contrast is donor winding two
minus donor winding one. `decay_power=512` selects an exact rational decay
bound admitted by the derived exponent; it does not count events.
The initial form/phase errors retain their separate exact zero-sum constraints
on each component. The small endpoint allowance is proved from those actual
source families, not installed as a replacement state.

Inspect `bridge_work_bounds` and `joined_form_mean_bounds` /
`joined_phase_mean_bounds` separately. The support event changes degrees and
conserved weighted means and adds supplied storage work; continuous loss
does not pay for that event. `identity_certified` checks the complete joined
state, retaining both windings under all later uninterrupted flow and recovery
on the new mean leaf. No later forcing or event is included.

The generic status needs a strictly positive recorded contrast; the example
also checks the larger frozen threshold. `unavailable` means a sufficient
obligation did not certify. Without an admitted source handoff, actual
response, work and retention bounds remain `None`. The shared forecast is
not used for the eighteen-node graph, and no incoming report is trusted in
place of original primitives.

The long structural relaxation, small readout error, initial organization,
constitutive law and bridge occurrence are supplied premises. This is a
conditional interaction certificate, not a practical timing claim, sensor
model, autonomous support-selection law or physical identification.
See the [contract](../../contracts/relational/SINE_PATTERNS.md#sine-formed-class-contact)
and [frozen protocol, proof and evidence](../../../theory/nodal/SINE_FORMED_CLASS_CONTACT.md#sine-formed-class-contact).

<a id="sine-formed-class-maintenance"></a>
### Certify a uniform return between repeated supplied probes

This reader rebuilds the original formation and first-probe evidence, then
checks a nonlinear return bound for every member of both pre-probe sets.
It certifies all repetitions by set inclusion rather than running a finite
sequence of trajectories.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formed_class_maintenance import (
    assess_sine_formed_class_maintenance,
)
from tnfr.sdk import relational_report_to_dict

maintenance = assess_sine_formed_class_maintenance(
    formation_time=100,
    probe_time=200,
    probe_duration=1,
    phase_increment=Q(1, 100),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    readout_error_bound=Q(1, 10**10),
    radius=Q(1, 12),
    common_dwell=10**12,
)
assert maintenance.status == "certified_repeated_probe_maintenance"
assert all(maintenance.return_certified_by_class)
reference = maintenance.reference_certificate
assert reference.recorded_contrast_bounds.lo > Q(1, 10**7)
assert all(bound.lo > 0 for bound in reference.probe_work_bounds_by_class)
evidence = relational_report_to_dict(maintenance)
```

The jumps occur at `tau=200+n*10**12`, with each node-zero form readout one
scaled unit later. The large dwell is a conservative mathematical bound in
`tau=e*t`; it is not a physical operating time or a minimum-dwell estimate.
Inspect both `form_return_margin_bounds` and `phase_return_margin_bounds`:
their strictly positive lower endpoints certify return inside half the
original pre-probe radii. The original full-coordinate preparation uncertainty
remains represented in the admitted sets, and both conserved means remain zero.
There is no reset, adaptive waiting rule or additional state disturbance.

The nested reference's correlated contrast and signed work intervals apply
to every cycle. Readout error is bounded separately for each observation
and does not feed back into state. After `N` events, multiply each per-event
work interval by `N` to enclose cumulative supplied work. Its positive lower
bound in this example makes indefinite operation require unbounded total
external work; the initial preparation budget is a separate resource.

This proves invariant neighborhoods and repeatable discrimination, not an
exact periodic orbit or a unique driven attractor. Return to a neighborhood
occurs within the admitted dwell; asymptotic target convergence applies when
the interventions stop. `unavailable` means a sufficient prerequisite or
return bound failed to certify, not that repeated operation is impossible.
See the [contract](../../contracts/relational/SINE_PATTERNS.md#sine-formed-class-maintenance)
and [frozen protocol, proof and evidence](../../../theory/nodal/SINE_FORMED_CLASS_MAINTENANCE.md#sine-formed-class-maintenance).

<a id="sine-formed-class-response"></a>
### Compare a common probe and recovery of the formed classes

This assessment continues both original C9 source families after acquisition,
applies one supplied mean-preserving phase jump, and bounds actual node-zero
form after the same elapsed time. It also checks that each actual post-jump
family remains within its own recovery domain.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formed_classes import (
    assess_sine_formed_class_response,
)
from tnfr.sdk import relational_report_to_dict

response = assess_sine_formed_class_response(
    formation_time=100,
    probe_time=200,
    probe_duration=1,
    phase_increment=Q(1, 100),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    readout_error_bound=Q(1, 10**10),
    radius=Q(1, 12),
)
assert response.formation_certificate.status == "certified_two_formed_classes"
assert response.recorded_contrast_bounds.lo > Q(1, 10**7)
assert all(response.recovery_certified_by_class)
assert response.status == "certified_formed_class_response"
evidence = relational_report_to_dict(response)
```

Times are in `tau=e*t`, not laboratory seconds. The original source errors
and their separate zero-sum constraints remain present through the warmup,
probe and readout. No trajectory is reset to a target. The jump changes phase
by `(1/100)*(e_0-(1/9)*1)` while form is unchanged; the same law then resumes.
The source, support, law, probe amplitude and timing are supplied premises.

Use `recorded_contrast_bounds` for the joint discriminator: it retains the
common heat-response factor of both classes. Subtracting the marginal readout
intervals loses that information and, for this frozen example, does not
certify the declared `1e-7` threshold. Inspect `probe_work_bounds_by_class`
separately from continuous loss. Recovery is admitted again for the actual
post-jump families; an earlier unforced certificate alone cannot cover a
new intervention.

The generic status requires strict positive contrast, while this example
also checks the larger frozen research threshold. `unavailable` denotes an
uncertified sufficient obligation. This one-probe result establishes neither
indefinite repeated-probe operation, autonomous event selection nor physical
constituent identity. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-formed-class-response)
and [protocol and proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-formed-class-response).

<a id="sine-formed-class-pair"></a>
### Admit two formed attracting classes before comparing their responses

This reader assesses fixed phase-flat preparations for the winding-one and
winding-two targets on the same simple C9. Both families use the same
positive-loss law and conserved means. Their source storage differs, although
both must satisfy the same declared ceiling.

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formed_classes import (
    assess_sine_formed_class_pair,
)
from tnfr.sdk import relational_report_to_dict

classes = assess_sine_formed_class_pair(
    scaled_time=100,
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    radius=Q(1, 12),
)
assert classes.status == "certified_two_formed_classes"
assert all(classes.formation_certified_by_class)
assert all(classes.source_budget_certified_by_class)
assert classes.symmetry_inequivalent
evidence = relational_report_to_dict(classes)
```

The time is `tau=e*t`, with `e=1023/1024`. Initial errors cover every fine
coordinate but have zero sum separately in form and phase, so the admitted
neighborhood is full dimensional on the sixteen-dimensional relative leaf.
No additional common-origin uncertainty is hidden in the input widths.

Inspect the separate source, geometry, storage and formation evidence before
using the pair in another claim. `unavailable` means the sufficient analytic
bounds did not certify admission. The assessment runs no trajectory and
evaluates no response to a probe. Distinct attracting classes and a common
measurement that distinguishes them are separate obligations. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-formed-class-pair)
and [protocol and proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-formed-class-pair).

<a id="sine-formation-response"></a>
### Check formation and its inherited receiver signature together

The fixed doubled-C5 comparison starts with zero nominal phases and a declared
integer form profile. Two preparations allocate the same nominal internal
storage to different pairs. This example checks acquisition and subsequent
geometric retention, together with a finite absolute receiver reading under
the same supplied positive-loss law:

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formation_response import (
    assess_sine_formation_response,
)

report = assess_sine_formation_response(
    scaled_time=100,
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    readout_error_bound=Q(1, 10**10),
    radius=Q(1, 8),
)
assert report.initial_zero_winding_certified
assert all(report.formation_certified_by_preparation)
assert report.recorded_difference_bounds.lo > Q(1, 10**8)
assert report.status == "certified_formation_response"
evidence = report.to_dict()
```

The time is `tau=e*t` with `e=1023/1024`; it is neither operator cycles nor
laboratory seconds. The initial error boxes include every fine coordinate
and the form origin. The response is the actual mean form of receiver pair 1
at that time, including readout uncertainty. It is not an integral of just one
pressure channel. No trajectory is numerically executed or reset after entry.

Inspect the separate radius, acute-geometry, storage and response margins when
changing a budget. `unavailable` means these sufficient estimates do not
certify the requested joint claim. Geometric identity is retained after entry;
the finite receiver contrast need not persist indefinitely. The preparation,
support and loss remain supplied. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-formation-response)
and [frozen protocol and proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-formation-response).

### Describe a whole pattern relative to a moving node

For a complete supplied sine-law state, remove unknown common origins while
retaining independent errors in its geometry:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.path_graph(3)
for node, form, phase in ((0, 0, 0), (1, 1, 0.5), (2, 2, 0)):
    graph.nodes[node].update(EPI=form, theta=phase, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 1024)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error, error, error),
    phase_error_bounds=(error, error, error),
)
assert pattern.relative_form_bounds[0].width == 0
assert pattern.relative_form_rate_bounds[0].contains(0)
assert pattern.reference_form_rate_bounds.lo > 0
evidence = pattern.to_dict()
```

The zero reference-relative coordinate is an identity, although the actual
reference is moving. The supplied radii concern synchronous residual errors
after a shared offset, in consistently lifted phases; this example is not
measured data. Every intermediary and capacity remains present.

For a prospective full-state enclosure, call `pattern.forecast` with explicit
`observation_time`, `end_time`, `time_step` and `order`.
Inspect `admitted` and `full_forecast.validated_end_time`. Use the returned
forecast bounds for the propagated family; the source's static storage bound
does not cover every point of the solver's larger outer box.
See the [contract](../../contracts/relational/SINE_PATTERNS.md#relative-sine-patterns-and-moving-references).
This does not infer a missing node or certify recovery. In particular, the
absolute-rate inverse below cannot consume moving-reference rates unchanged.

### Certify recovery of an uncertain cycle pattern

This example supplies a deformed five-node cycle and nonzero residual errors.
The target remains the exact mathematical twist; the nominal phases need not
be an equilibrium:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error,)*5,
    phase_error_bounds=(error,)*5,
)
recovery = pattern.certify_cycle_recovery(
    cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
)
assert recovery.admitted
assert recovery.norm_margin > 0 and recovery.energy_margin > 0
recovery_evidence = recovery.to_dict()
```

The proof covers every state consistent with those residual bounds, including
arbitrary common origins. It uses no trajectory and claims no laboratory
measurement. An explicit analytic check of this same dyadic observation class
is given by the [proof owner](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-cycle-recovery).

The same method on a `SineRelativeForecast` assesses its full endpoint box
at the actual validated time, preserving the solver's original status.
Inspect `hypothesis_failures` and `unresolved_conditions` if admission is
unavailable. This sufficient test does not diagnose instability.
The [contract](../../contracts/relational/SINE_PATTERNS.md#whole-set-sine-cycle-recovery)
requires the **complete** support to be the cycle: do not remove environmental
connections to make a subregion pass.

### Protect cycle identity during conservative motion

The same barrier used for dissipative recovery can instead certify that a
conservative cycle stays near its prepared phase geometry. The family-level
recurrence theorem then applies without asserting a period or an individual
return. Here the support is the **whole** isolated five-node ring:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import assess_sine_cycle_identity
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(error,)*5, phase_error_bounds=(error,)*5,
)
identity = assess_sine_cycle_identity(
    pattern, cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
    excess_ceiling=Fraction(1, 2560), form_mean_bounds=(-1, 1),
)
assert identity.family_admitted
assert identity.source_set_trapping_certified
assert identity.relative_source_family_membership == "certified_inside"
assert identity.family_almost_everywhere_recurrence_certified
assert identity.individual_recurrence_status == "unavailable_for_chosen_state"
identity_evidence = relational_report_to_dict(identity)
```

All states consistent with these relative error bounds retain winding one.
The declared mean interval is a separate restriction defining a finite-volume
family: the relative observation cannot measure an absolute common form
origin. Almost every member of that family returns near its starting full
state, but the displayed flags do not certify recurrence of this particular
preparation. They also do not imply that the ring returns to its exact target
or that it formed autonomously. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-conservative-identity)
for independent family, trapping and membership verdicts.

### Check a formation preparation before evolving it

This declared family starts with a twisted donor and a uniform receiver.
A form pulse at the intermediary and unequal receiver capacities produce
an initial directed deformation, but that does not guarantee a new twist:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

model = RelationalExchangeModel(1, phase_domain="regular")
preparation = dict(
    model=model, amplitude=2, capacity_contrast=Fraction(1, 2),
)
initial_checks = assess_sine_mediated_formation(**preparation)
assert initial_checks.initial_form_storage == 4
assert initial_checks.target_budget_status == "passed"
assert initial_checks.entry_budget_status == "passed"
assert initial_checks.phase_action_bound.necessary_entry_time_lower_bound > 240
barrier = initial_checks.maintained_target_obstruction
assert barrier.maintained_target_excluded
assert barrier.target_margin_bounds.lo > Fraction(1, 4)
assert initial_checks.status == "excluded"
assert not initial_checks.early_loss_exclusion_certified
timed = assess_sine_mediated_formation(
    **preparation, exclusion_time=Fraction(1, 5),
)
assert timed.status == "excluded"
assert timed.phase_action_bound.entry_through_horizon_excluded
assert timed.early_loss_exclusion_certified
formation_evidence = timed.to_dict()

# A different exchange/loss balance inside the proved domain.
changed_balance = assess_sine_mediated_formation(
    model=RelationalExchangeModel(
        1, epi_weight=4, phase_weight=5, phase_domain="regular",
    ),
    amplitude=2, capacity_contrast=Fraction(1, 2),
)
changed_barrier = changed_balance.maintained_target_obstruction
assert 1 < changed_barrier.exchange_to_loss_ratio < Fraction(3, 2)
assert changed_barrier.maintained_target_excluded
assert changed_barrier.target_margin_bounds == barrier.target_margin_bounds

# A separate endpoint: the donor unwinds and only the receiver stays twisted.
transfer = timed.receiver_transfer()
assert transfer.source is timed
assert transfer.status == "passes_necessary_conditions"
assert transfer.target_storage_margin == 4
assert transfer.target_functional_margin == Fraction(16, 5)
assert transfer.combined_phase_action_cost == Fraction(29, 25)
assert transfer.compatible_common_phase_turn_offset == Fraction(-11, 190)
assert not transfer.early_loss_exclusion_certified
assert transfer.receiver_first_barrier_phase_storage == Fraction(7, 2)
assert transfer.necessary_receiver_running_work_strict_lower_bound == Fraction(7, 2)
assert transfer.donor_barrier_first_required
assert transfer.simultaneous_barrier_passage_excluded
transfer_evidence = transfer.to_dict()
```

The initial storage checks pass, but the independent mixed-functional
certificate excludes convergence to the maintained two-twist target without
a supplied horizon. A necessary entry time above 240 is conditional on joint
acute entry; it predicts neither entry nor capture. The timed report separately
proves that dissipation exhausts the joint-entry budget before that entry can
occur. Both reports use analytic bounds without evolving a trajectory.

The [whole-class proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-maintained-target-obstruction)
excludes the maintained target for all five donor forms and intermediary form
with initial form storage at most four, under the fixed capacity and storage
premises and the sufficient ratio domain `0<w/e<3/2`. The example calls are
applications of that theorem. This interval is not a sharp physical threshold.
The global
certificate alone does not exclude every transient acute-region visit or
receiver-only winding after losing the donor's identity. Leaving the admitted
ratio domain or changing storage scale or contrast makes this certificate
unavailable; that is not positive formation. Common raw-weight scaling is
removed by ideal normalization; rounded inputs still use their represented
effective ratio. Neither operation implements a new execution clock.

The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-formation-eligibility-and-timed-exclusion)
documents profile admission, the separate timed bounds and export fields.
Use `profile="explicit"` with five ordered `donor_epi` values to retain internal
donor differences; `amplitude` still supplies intermediary form. No support
event, new loss law or native runtime change is involved.

`receiver_transfer()` keeps this same preparation but asks about a different
final identity. Its passed necessary checks do not prove that the receiver
will acquire the pattern. The original two-pattern target remains excluded.
The [transfer contract](../../contracts/relational/SINE_PATTERNS.md#receiver-identity-transfer-with-donor-unwinding)
keeps both winding changes, their combined phase-action cost and the new loss
allowance explicit. The reported common phase offset is one compatible lift,
not a predicted final absolute phase. To certify recovery of an actual later
observation, pass `transfer.target_phase_turns` to the existing full-pattern
recovery reader; a positive initial budget does not replace that observation.

The passage flags describe necessary geometry and work **if transfer occurs**.
At form budget at most `4`, the donor must cross its own potential barrier
before the receiver reaches its first one. This is not a statement about
when either winding changes. The receiver must receive accumulated signed
port work exceeding `7/2`, with an additional loss cost for a crossing within
a given time. These are prospective requirements: the initial report does
not evaluate that accumulated work, and a positive instantaneous work rate
does not satisfy them. A false ordering flag leaves the order unknown.

### Distinguish source geometry from a formation prediction

The existing preparation report can retain its communicating and silent
components while bounding the tangent receiver response:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
    profile="explicit", donor_epi=(0, 1, 0, 0, -1),
)
excitation = source.receiver_excitation()
assert excitation.available
assert excitation.even_form_storage == 1
assert excitation.odd_form_storage == 2
assert excitation.tangent_receiver_phase_cost_upper_bound == Fraction(3, 4)
assert excitation.nonlinear_correction_status == "positive_bound"
assert excitation.full_phase_budget_status == "available"
assert excitation.actual_full_phase_storage_upper_bound == Fraction(139, 20)
assert excitation.donor_potential_barrier_first_required
assert excitation.necessary_donor_phase_release_lower_bound > 0
assert tuple(a + b for a, b in zip(
    excitation.even_initial_epi, excitation.odd_initial_epi,
)) == source.initial_epi
excitation_evidence = excitation.to_dict()
```

This example evaluates no trajectory. The tangent ceiling and correction
threshold describe the response needed for nonlinear barrier passage. The
separate full-phase ceiling applies to the actual law at total form budget
at most `4`: receiver passage requires an earlier donor barrier crossing.
Neither bound proves that passage occurs or identifies the donor's potential
decrease as useful receiver work.
Reversing only the odd source retains its energy and tangent receiver response
but changes nonlinear receiver work. Keep both components when studying the
actual law. The [contract](../../contracts/relational/SINE_PATTERNS.md#source-geometry-and-the-necessary-nonlinear-receiver-correction)
defines unavailable bounds and the [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-source-receiver-excitation)
states the exact distinction; neither selects this example for a new response.

### Certify receiver consensus without selecting the donor endpoint

A regional nonlinear certificate applies to every six-coordinate preparation
whose total initial form storage is at most `(5-sqrt(5))/2`. It can establish
receiver consensus when the separate donor-recovery checks are inconclusive:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == 1
assert not source.donor_well_retention().relative_donor_pattern_convergence_certified
assert not source.donor_dissipative_capture().relative_donor_pattern_convergence_certified
localization = source.receiver_localization()
assert localization.status == "certified"
assert localization.relative_receiver_consensus_certified
assert localization.receiver_nonflat_equilibria_excluded
transfer = source.receiver_transfer()
assert transfer.receiver_localization == localization
assert "weighted_receiver_localization" in transfer.exclusion_reasons
assert transfer.status == "excluded"
localization_evidence = localization.to_dict()
```

This evaluates no trajectory. The proof weights regional potentials and
retains their full nonlinear coupling. It selects the receiver's relative
limit, not a donor limit or a time to recovery. Transient receiver excursions
remain possible; no all-time barrier verdict follows. A `not_certified` result
above the threshold leaves acquisition unresolved. The
[contract](../../contracts/relational/SINE_PATTERNS.md#nonlinear-receiver-localization-from-regional-storage)
and [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-weighted-receiver-exclusion)
own the exact admission and scope.

### Certify recovery of the prepared donor

The same preparation reader can now select the donor-only relative limit
for a proved interval of form storage. This example is nonsilent and has
enough total storage to pass the bare phase barrier, but the full coupled
dynamics still returns to the original donor pattern:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(7, 30), capacity_contrast=Fraction(1, 2),
)
retention = source.donor_well_retention()
assert retention.initial_form_storage == Fraction(49, 900)
assert retention.relative_donor_pattern_convergence_certified
assert retention.receiver_only_targets_excluded
assert retention.exact_retention_polynomial_margin > 0
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_well_retention.relative_donor_pattern_convergence_certified
retention_evidence = retention.to_dict()
```

The certificate applies to all five donor forms and the intermediary form,
not only this localized example. It combines the exact equilibrium geometry
with the existing decreasing auxiliary function. Its threshold is
`F_c=(25*sqrt(5)-55)/16`, approximately `0.056356`, and admission compares
`(16*F+55)**2 <= 3125` exactly. Neither this proof coefficient nor its threshold
is a new physical constant.

This proves the final relative donor identity and excludes either maintained
receiver-only twist. It gives no arrival time or guarantee that winding never
changes during the transient. Above the threshold the certificate does not
decide the outcome; a connected static path is insufficient. The
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-donor-well-retention)
retains the law premises, exact admission and distinct transfer bounds.

### Prove dissipative recovery above the initial barrier

A preparation can exceed the static donor-retention bound and still recover
the donor. This certificate uses the full law's short-time dissipation and
phase displacement to prove entry into the donor component:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(1, 4), capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == Fraction(1, 16)
assert source.donor_well_retention().status == "not_certified"
capture = source.donor_dissipative_capture()
assert capture.horizon == Fraction(1, 4)
assert capture.initial_dissipative_norm_squared == Fraction(1, 6)
assert capture.donor_component_entry_certified
assert capture.relative_donor_pattern_convergence_certified
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_dissipative_capture.receiver_only_targets_excluded
capture_evidence = capture.to_dict()
```

The horizon certifies membership in a region that guarantees eventual donor
recovery; it is not the time at which recovery finishes. No trajectory is
numerically evolved. The bound uses both form storage and its full-support
gradient, so equal storage alone does not determine its verdict. A failed
sufficient check leaves the outcome open. This certificate has the original
half-weight law scope; the static donor-well result retains its wider
coefficient-ratio domain. See the
[capture contract](../../contracts/relational/SINE_PATTERNS.md#sine-donor-dissipative-capture)
for exact admission and retained evidence.

### Compose established component Hessians through a bridge tree

Use this algebra only after deriving each component's relative Hessian inertia
and the bridge signs. Here two C5 components have independently established
positive four-dimensional relative Hessians; a singleton mediates their union:

```python
from tnfr.physics.phase_cycle_geometry import compose_bridge_tree_hessian_inertia

components = ((4, 0, 0), (4, 0, 0), (0, 0, 0))
aligned = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, 1), (2, 1, 1)),
)
assert aligned.total_nodes == 11
assert aligned.relative_inertia == (10, 0, 0)
opposed = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, -1), (2, 1, 1)),
)
assert opposed.relative_inertia == (9, 1, 0)
composition_evidence = aligned.to_dict()
```

This does not reconstruct or check a live phase configuration. The specialized
C5 catalog below supplies those premises from its exact critical states and
delegates to the same algebra. Under the full positive-loss sine theorem,
an aligned bridge preserves positive restoring geometry and an antipodal
bridge adds an unstable direction. Extra connections that close a cycle
require another compatibility argument; neither stability nor the input
inertia triples establish formation. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#exact-geometric-inertia-and-full-law-stability)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition).

### Reject incompatible cycle periods before seeking an equilibrium

Cycles sharing paths cannot choose their phase periods independently. The
shared exact reader tests one combined-cycle constraint, including cancellation
on shared edges. This example has two fundamental five-edge cycles; each
period individually meets its acute length bound, but their joint assignment
fails on the four-edge cycle formed by their difference.

```python
import networkx as nx
from tnfr.physics.phase_cycle_geometry import (
    assess_acute_cycle_periods,
    derive_phase_cycle_geometry,
)

graph = nx.Graph()
graph.add_nodes_from(range(6))
graph.add_edges_from(((0, 1), (0, 2), (2, 3), (1, 4), (3, 4), (1, 5), (3, 5)))
geometry = derive_phase_cycle_geometry(graph)
assert tuple(map(len, geometry.fundamental_cycles)) == (5, 5)

single = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, 0)
)
assert single.status == "necessary_bound_passed"
joint = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, -1)
)
assert joint.combined_period == 2
assert joint.strict_period_bound == 1
assert joint.obstruction_certified
payload = joint.to_dict()
```

An obstruction rules out every strictly acute state with those periods,
not only one proposed phase assignment. Passing a witness leaves existence
unresolved. Even a reconstructed acute phase state can fail the separate
nodal sine-balance condition. The
[proof and controls](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
distinguish these obligations. This readout creates no connections and does
not establish that a compatible pattern forms.

### Certify sector capture without supplying its final pattern

A whole-sector storage barrier can guarantee convergence without giving the
reader equilibrium coordinates. This source has nonuniform form and a phase
profile with nonzero cycle period. Its future is assessed under the explicitly
supplied complete sine law; the initial state is not declared stationary.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(1, 1024) if node == 0 else 0,
        theta=Fraction(5 * node, 4),
        nu_f=2 if node == 1 else 1,
    )
source = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(Fraction(1, 65536),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
# Canonical edges: (0,1), (0,4), (1,2), (2,3), (3,4).
# Subtract a full turn from the 0->4 phase difference.
capture = source.certify_sector_capture(edge_turn_offsets=(0, -1, 0, 0, 0))
assert capture.cycle_periods == (1,)
assert capture.admitted
assert capture.energy_margin > 0
assert len(capture.boundary_face_lower_bounds) == 10
assert capture.weighted_form_mean is None  # the common origin is unobserved
payload = capture.to_dict()
```

The certificate covers all signed faces of the acute sector and the entire
declared uncertainty set. A positive verdict proves existence, uniqueness and
convergence there without a target solve. An unavailable verdict may reflect
a conservative bound even when an equilibrium exists. This does not generate
the initial winding or support, and a supplied event needs separate state and
storage-jump admission. See the [contract](../../contracts/relational/SINE_PATTERNS.md#target-free-acute-sector-capture)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture).

### Join two relative patterns with explicit origins

Internal relative coordinates do not specify how two components are aligned.
Declare the right-minus-left common form and phase origins, a common structural
observation time and one bridge. This static assessment rebuilds the full
support and hands the joint uncertainty family to the existing capture owner.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(1, phase_domain="regular")

def component(labels, capacities):
    graph = nx.path_graph(labels)
    for node, capacity in zip(labels, capacities):
        graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_pattern(
        graph, reference_node=labels[0], reference_model=model,
        form_error_bounds=(0, 0), phase_error_bounds=(0, 0),
    )

left = component((0, 1), (1, 2))
right = component((2, 3), (3, 4))
contact = left.compose_with(
    right, bridge=(1, 2), observation_time=2,
    form_origin_difference=Fraction(1, 4),
    phase_origin_difference=Fraction(1, 4),
    edge_turn_offsets=(0, 0, 0),  # canonical edges: (0,1), (1,2), (2,3)
    work_allowance=Fraction(1, 8),
)
assert contact.status == "available"
assert contact.budget_status == "within_allowance"
assert contact.capture.admitted
assert contact.joined.degrees == (1, 2, 2, 1)
assert contact.capture.weighted_form_mean is None
payload = relational_report_to_dict(contact)
```

These offsets describe the common additive origins in each source family;
they are not measurements of the reference-node gap when residuals are present.
Omitting either offset leaves composition unavailable. A supplied time declares
synchrony; the source reports do not authenticate it. Capture, bridge-work
allowance and event occurrence are separate questions. This call neither adds
a live edge nor runs a trajectory. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-relative-frame-composition)
and [derivation](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-relative-frame-composition).

### Certify acquisition from initially equal phases

The same sine law can transfer a supplied form profile into nonzero phase
winding. This example declares the profile, coefficients and horizon before
evaluating the analytic certificate. It uses no trajectory fit or supplied
equilibrium coordinates.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(EPI=4092 * (node - 2), theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
entry = certify_sine_prepared_entry(
    source, scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert entry.admitted
assert entry.initial_cycle_periods == (0,)
assert entry.capture.cycle_periods == (1,)
assert entry.horizon == Fraction(102400, 1023)
assert entry.initial_form_storage == 167444640
assert entry.capture.energy_margin > Fraction(1, 100)
payload = entry.to_dict()
```

`endpoint_form_bounds` and `endpoint_phase_bounds` enclose all nodes at the
declared horizon. Tighter correlated edge bounds feed the shared capture
kernel; the full Cartesian node box is only an outer projection. The initial
phase winding is absent, but the nonuniform form information and large initial
storage are supplied. The result proves formation and subsequent retention
under this law; it does not select that preparation or identify matter.
Uniform initial form with equal phases instead stays stationary. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#analytic-prepared-entry-into-a-captured-sector)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry).

The same graph, model and horizon also admit a full preparation family. Keep
the residual radii fixed before evaluating the following extension:

```python
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

preparations = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(Fraction(1, 16),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
family = preparations.certify_prepared_entry(
    scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert family.admitted and family.initial_zero_winding_certified
assert family.initial_cycle_periods == (0,)
assert family.capture.cycle_periods == (1,)
assert family.capture.energy_margin > Fraction(12, 1000)
assert family.endpoint_form_bounds is None
assert family.weighted_form_mean is None
assert len(family.centered_endpoint_phase_bounds) == 5
```

Every member starts with zero winding and acquires the captured unit winding.
The relative source leaves common origins unknown: use the centered endpoint
bounds, not invented absolute values. Each member keeps its own conserved
means. Initial form and phase storage are interval bounds, so the nominal
budget does not describe the entire family. With uniform nominal form and
the same small errors, the initial zero-sector capture test instead proves
convergence to consensus; that uncertain control is not stationary.

### Join two analytically acquired patterns

Keep the original preparations and analytic endpoint sets. This fixed example
joins two copies at their first nodes after the same declared duration:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024), phase_weight=Fraction(1, 1024),
    phase_domain="regular",
)

def prepared_component(start):
    graph = nx.cycle_graph(range(start, start + 5))
    for j, node in enumerate(graph):
        graph.nodes[node].update(EPI=4092 * (j - 2), theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_pattern(
        graph, reference_node=start, reference_model=model,
        form_error_bounds=(0,) * 5, phase_error_bounds=(0,) * 5,
    )
    return source.certify_prepared_entry(
        scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
    )

joined = prepared_component(0).compose_with(
    prepared_component(5), bridge=(0, 5),
    left_initial_time=0, right_initial_time=0,
    form_origin_difference=0, phase_origin_difference=0,
    edge_turn_offsets=(0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0),
)
assert joined.observation_time == Fraction(102400, 1023)
assert joined.acquisition_and_capture_certified
assert joined.capture.cycle_periods == (1, 1)
assert joined.capture.energy_margin > Fraction(9, 1000)
assert joined.budget_status == "not_supplied"
payload = relational_report_to_dict(joined)
```

The call rebuilds each analytic prefix and derives the common endpoint time.
Its source retains both preparations; the endpoint is not converted to an
independent observation box. `status` describes frame availability, while
`acquisition_and_capture_certified` requires both acquisitions and joint
maintenance. No event budget or occurrence follows. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-prepared-composition) and
[proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-composition).

### Certify an entire phase-flat budget family

Keep the preceding `1023:1` coefficient ratio, but bound the original form
storage by `B=160`. This declares every signed form within that ceiling and
exactly equal initial phases, with arbitrary common origins. The graph
supplies only support; no synthetic nodal state is needed.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_budget import certify_sine_budget_consensus
from tnfr.sdk import relational_report_to_dict

geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
model = RelationalExchangeModel(
    1, epi_weight=1023, phase_weight=1, phase_domain="regular",
)
budget = certify_sine_budget_consensus(
    geometry, reference_model=model, capacity=(1,) * 5,
    form_storage_budget=160,
)
assert budget.status == "certified"
assert budget.consensus_certified and budget.zero_winding_certified
assert budget.acute_trapping_certified
assert max(budget.phase_edge_upper_bounds) < Fraction(1, 50)
payload = relational_report_to_dict(budget)
```

This certifies consensus and zero winding for the entire family at all future
times. The earlier acquisition source costs `160*1023^2` and lies outside this
family. A zero budget instead makes the whole family stationary. In general,
the theorem protects the natural `(-pi,pi)` edge strip; strict acuteness is a
separate, stronger flag. Its `eta<=1` condition is a sufficient proof premise,
not an instability threshold. When a comparison is unavailable, inspect
`unresolved`: candidate bounds remain visible, but certified trajectory
bounds are `None`.

Under the feedback premise, `nonzero_winding_budget_lower_bound` records a
necessary cost growing as `(w/e)^-2`; exceeding it does not prove formation.
The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-budget-consensus) and
[proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus)
give the family and admission boundaries. The exact `payload` can be saved
with the SDK's `export_to_json`.

### Distinguish equal budgets by preparation symmetry

The acquisition example above uses initial storage `160*1023^2`. The
following reflection-fixed preparation has exactly the same budget under
the same law, but its symmetry excludes nonzero winding on the declared
cycle whenever principal increments are defined.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
for node, value in enumerate((8, 4, -8, -8, 4)):
    graph.nodes[node].update(EPI=1023 * value, theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
symmetry = source.assess_cycle_symmetry(
    permutation_indices=(0, 4, 3, 2, 1), cycle=(0, 1, 2, 3, 4),
)
assert symmetry.initial_form_storage == 160 * 1023**2
assert symmetry.status == "certified"
assert symmetry.trajectory_symmetry_certified
assert symmetry.zero_winding_when_nonantipodal
assert symmetry.nonzero_winding_limit_excluded
payload = relational_report_to_dict(symmetry)
```

Supply an exact comparison and a permutation in its complete node order.
Inspect the symmetry and winding flags separately: the moving source is
neither certified stationary nor guaranteed to avoid antipodal edges.
Relative uncertainty reports are unsupported by this reader.

The [positive acquisition example](#certify-acquisition-from-initially-equal-phases)
already supplies the opposite outcome at this budget. Scalar resources alone
therefore do not determine the acquired geometry. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-cycle-symmetry) and
[equal-budget proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation)
for the full-state symmetry and orientation conditions.

### Classify long-time endpoints without selecting a basin

For two C5 rings connected through one intermediary, the complete sine law
has a finite exact set of relative equilibria. The following source has
arbitrary finite form and phase; the reader does not require a prepared twist:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

graph = nx.Graph()
graph.add_nodes_from(range(11))
cycles = (tuple(range(5)), tuple(range(5, 10)))
for cycle in cycles:
    nx.add_cycle(graph, cycle)
graph.add_edges_from(((0, 10), (5, 10)))
for node in graph:
    graph.nodes[node].update(EPI=node / 10, theta=node / 7, nu_f=1)
graph.nodes[6]["nu_f"] = 1.5
graph.nodes[9]["nu_f"] = 0.5

source = bound_relational_sine_exchange(
    graph, reference_model=RelationalExchangeModel(1, phase_domain="regular"),
)
result = source.asymptotic_equilibria(cycles=cycles)
assert result.admitted
assert result.single_relative_equilibrium_convergence_certified
assert result.full_lifted_state_convergence_certified
assert result.selected_equilibrium_status == "unavailable_no_basin_selection"

catalog = result.critical_set
assert len(catalog.cycle_edge_turn_options) == 30
assert catalog.relative_state_count == 3600
# Pick one symbolic catalog member, not the predicted endpoint of this source.
member = catalog.reconstruct(
    cycle_choices=(0, 0), bridge_turns=(0, Fraction(1, 2)),
)
assert not any(member.symbolic_sine_coefficients)
endpoint_evidence = result.to_dict()

# Classify a declared target under the source's law, not the source's basin.
twist = catalog.cycle_edge_turn_options.index((Fraction(1, 5),) * 5)
stable = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, 0),
)
assert stable.local_exponential_attraction_certified
assert stable.relative_stable_modes == 20
assert stable.relative_unstable_modes == 0
opposed = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, Fraction(1, 2)),
)
assert opposed.nonlinear_instability_certified
assert opposed.relative_unstable_modes == 1
assert catalog.phase_hessian_index_counts[0] == 9
assert sum(catalog.phase_hessian_index_counts) == 3600
stability_evidence = stable.to_dict()
```

Each cycle choice indexes its declared oriented ring. The two bridge choices
follow `catalog.bridge_edge_indices` in the stored full geometry; inspect
those indices rather than assuming label sorting or cycle order. Exact turns
in the reconstructed member represent multiples of mathematical `2*pi`.
They are distinct from the captured source's represented radian values.

The convergence theorem does not select the member, its terminal phase turns
or a numerical rate. The separate stability reader classifies each declared
target: nine are locally exponentially attracting modulo common origins;
the other 3,591 are unstable. This does not place the observed source in a
target's basin or supply a certified neighborhood radius. Adding a half-turn
on one bridge in this example creates one unstable direction in the complete
form-phase dynamics, even though both ring geometries remain individually
acute. The catalog keeps all nonacute branches. It stores 30 local
options and independent bridge choices, avoiding a list of 3,600 full states.
Zero loss or any zero capacity makes the stronger convergence verdict
unavailable. The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-asymptotic-equilibria)
separates that admission from geometric classification and from native Arg
dynamics. Both evidence dictionaries can be saved using the SDK's
`export_to_json`.

### Declare the full-state receiver barrier experiment

The source-specific [receiver barrier protocol](../../contracts/relational/SINE_PATTERNS.md#frozen-receiver-barrier-exclusion)
combines a whole-time initial enclosure with a total-energy tail. Preparation
does not run the trajectory:

```python
from tnfr.research.relational_receiver_barrier import prepare_receiver_barrier

protocol = prepare_receiver_barrier()
assert len(protocol["nodes"]) == 11
assert protocol["maximum_steps"] == 256
```

The [benchmark](../../../benchmarks/relational_receiver_barrier.py) archives the
protocol with `--prepare`; the same command without that flag performs its
one fixed-budget evaluation. Use a fresh output path for a separately declared
regression, never overwrite the original response. A partial enclosure is
unavailable; a failed upper-bound comparison is unresolved. Neither establishes
receiver formation. The execution plan owns the retained scientific verdict.
