# Physical admission of the delayed digital-PLL realization

<a id="dpll-physical-admission"></a>

## Selected comparison and decision scope

**Decision:** `physical_status=not_admitted`, `evaluation=not_tested` for
the selected published realization as a bridge to the fixed sine inference.
An exact current-state mapping obstruction and missing physical uncertainty
evidence are distinct reasons; neither excludes all possible realizations.

This auxiliary P2 audit selects one terrestrial realization: the coupled digital
phase-locked loops of [Wetzel et al. (2017)][paper]. It asks whether that
realization supplies the state, law and independently justified observation
premises needed by the existing TNFR sine inference. Selection is a
methods review, not admission of measured responses or a prerequisite for
deriving collective interaction properties from TNFR patterns. The
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns its status.

The [ontology](../EMERGENT_ONTOLOGY.md#research-target-physical-properties-of-collective-patterns)
allows collective physical observations; it does not identify electronic
phase, control voltage or an oscillation frequency with a TNFR coordinate
by name. The earlier
[Rössler source obstruction](PHASE_AMPLITUDE_MEASUREMENT_PROTOCOL.md#rossler-phase-information-admission)
remains unchanged and is not evidence about this different circuit law.

<a id="dpll-source-evidence"></a>
## Primary-source evidence

The following factual ledger summarizes the selected paper's Sections 1,
5, 6 and Appendices A/C. Its equations are analyzed separately below.

| Published premise | Relevance to admission |
| --- | --- |
| CD4046B circuits combine XOR detection, low-pass filtering and voltage-controlled oscillation. | Retain detector, filter and oscillator state and constitutive domains. |
| A microcontroller supplies transmission delays; tested networks contain two, three or nine oscillators. | Delays and the actual support are part of the source model. |
| The phase approximation uses an even triangular coupling function and suppresses high-frequency detector terms. | Neither the target sine kernel nor a quantified reduction error follows. |
| Oscillators initially run independently for a random duration before coupling. | This is not the target's prescribed complete-state preparation. |
| Oscilloscope output traces supply phase time series and frequency/relaxation estimates. | Admit the extraction and instrument observation law separately. |
| The paper treats idealized dynamics without phase noise and identifies oscillator clipping. | Nominal formulas do not certify full-domain model or measurement errors. |

These statements are supported by the [primary article][paper]. Its
[experimental supplement][supplement] is identified but was not opened or
downloaded for this audit. Published figures consulted during selection are
prior information and cannot later become reserved TNFR observations.

Manufacturer documentation supplies further component-level checks.
The [TI application report, Section 3.1][ti-application] describes comparator
I's XOR/triangular characteristic and its duty-cycle dependence; Section
3.2 discusses buffered observation of the control voltage. The
[PicoScope 2205 MSO specification, page 4][pico-specification] lists two
analog and sixteen digital channels, 8-bit analog resolution, DC accuracy
of ±3% full scale and timebase accuracy of ±100 ppm. These are published
specifications, not a calibration certificate for the experimental setup.
They provide neither a transformed TNFR noise bound nor a validated source
or clock map. No instrument acquisition is proposed here.

The [source receipt](../../docs/assets/dpll_physical_admission/method-sources-v1.json)
records the selected primary URLs and retrieved-byte hashes. It is a
methods provenance record, not a reserved response, an authentication
certificate or independent validation of the published specifications.

<a id="dpll-state-law-admission"></a>
## State, law and clock admission

The target [complete sine law](../nodal/SINE_APERTURE_INFERENCE.md#sine-aperture-source-and-law)
evolves both signed form and primitive phase on a fixed eighteen-node
support. The source must supply either a complete state/clock map into
that model or a separately proved approximation with bounded defects in
both evolution rows. A scalar phase fit or a common frequency label does
not meet this requirement. A source-specific reduction must retain its
history, initialization, input and discarded-state assumptions.

For a first-order loop filter and identical nominal parameters, a local
realization of the paper's phase model (Eq. 8 and Appendix C) is

\[
\begin{aligned}
\dot\phi_i(t)&=\omega+K z_i(t),\\
b\dot z_i(t)&=-z_i(t)+\sum_j\frac{c_{ij}}{d_i}
 \Delta\!\left(\phi_j(t-\tau_d)-\phi_i(t)\right),\\
\Delta(\psi)&=\frac{2}{\pi}|\operatorname{wrap}_{\pi}\psi|-1,
\qquad d_i=\sum_jc_{ij}.
\end{aligned}
\tag{1}
\]

Here `t`, the filter time `b>0` and delay `tau_d>0` are in seconds;
`phi` is in radians, `omega` and `K>0` are in inverse seconds, and `z`
is dimensionless. Any clock change must transform both evolution rows.
Equation (1) has an independent-initial-value realization with phase
history on `[-tau_d,0]` and current filter state. Matching a convolution
initialized from an infinite past additionally constrains that filter
state, or requires its homogeneous initialization term. Arbitrary pairs
of initial history and filter state used below are supplied admissible
preparations of this delay realization; this audit does not establish their
experimental reachability or compatibility with every convolution history.
Nor does it bound the phase approximation's discrepancy from all circuit
states.

<a id="dpll-delayed-history-obstruction"></a>
### An exact obstruction to a regular current-state identification

The following deduction uses (1), without measured response data. Consider
a domain admitting two phase histories with the same current
`y=(phi(0),z(0))` and a nonzero current phase-velocity vector
`u=omega*1+K*z(0)`. Choose one incident delayed neighbor value strictly inside
a linear branch of `Delta`, and perturb that value while preserving the
current phase and filter state. Smooth history perturbations away from
zero can also preserve the current history derivative. The affected
`z_dot` changes because the branch slope is nonzero, while `phi_dot=u`
does not. The two current-state derivative vectors at that state have the form

\[
F_1=(u,v),\qquad F_2=(u,v+d),\qquad u\ne0,\quad d\ne0.
\tag{2}
\]

They are not collinear: `F1=a*F2` would give `a=1` from the phase block
and then `d=0`, a contradiction. A differentiable current-state map
`J(y)` with injective differential preserves this noncollinearity.
If it produced one autonomous target vector at `J(y)` under positive
clock factors `r1,r2`, then

\[
DJ(y)F_1=r_1F_{\rm target}(J(y)),\qquad
DJ(y)F_2=r_2F_{\rm target}(J(y)).
\tag{3}
\]

Injectivity would imply `F1=(r1/r2)*F2`, contradicting (2). Thus even
history-dependent positive clock rescaling cannot close this regular
current-state identification on a family containing the two histories.
The conclusion also covers a regular embedding into a larger current-state
target; merely appending coordinates determined by `y` does not retain the
missing independent history. The map is regular in the current coordinates,
not injective on the complete state that includes history.

This is a mapping obstruction on the stated domain. It does not exclude
restricted initialized histories, a history-dependent state map, a lossy
observable with its own closure proof, or an independently bounded
approximation. The triangular detector, filtering, support and input
differences remain separate admission obligations even after such a
history restriction. An exact phase equation on a selected synchronized
family would not establish a transient full-state realization.

An independent rational control makes the two-history premise explicit.
In phase turns `psi=phi/(2*pi)`, take `b=tau_d=1` second,
`omega=K=pi/4` radians per second and `z(0)=(0,0)`. In the following
formulas, `t` is the numerical time in seconds. On `-1<=t<=0`, set

\[
\psi_0(t)=t/8,\qquad
\psi_1^{A}(t)=1/4+t/8,\qquad
\psi_1^{B}(t)=1/4+t/8+t^2/32.
\]

Both histories have current phase `(0,1/4)` and derivative `(1/8,1/8)`;
their history derivatives remain positive. The current delayed arguments
are `(1/8,-3/8)` and `(5/32,-3/8)`, strictly inside regular detector
branches. Using `Delta_turn(a)=4*abs(wrap_turn(a))-1`, the two current
vectors in `(psi,z)` coordinates are
`(1/8,1/8,-1/2,1/2)` and `(1/8,1/8,-3/8,1/2)`.
The minor from their first and third components is `1/64`, proving
noncollinearity. The
[independent controls](../../tests/research/test_dpll_physical_admission.py)
derive the detector from exact square-wave XOR overlap before checking
these vectors. This supplied initial-value example proves neither
experimental reachability nor an extension obeying the unforced source
law throughout the negative-time history.

<a id="dpll-phase-and-clock-boundary"></a>
### Phase readout and rotating-frame boundaries

Changing to a fixed rotating frame `theta=phi-Omega*t` retains the delayed
detector argument

\[
\theta_j(t-\tau_d)-\theta_i(t)-\Omega\tau_d.
\tag{4}
\]

Dropping its last term changes the source law. A biased triangular detector
can have different parity from the unshifted one; evenness alone is not
an obstruction to every coordinate or phase-origin change.

There is a further obstruction to simply using these phases as the target
phases on the same undirected support. The target has constant degree-weighted
mean phase because `d.T*A=0`. The source instead gives
`d.T*theta_dot=d.T*(omega*1+K*z-Omega*1)`, which varies with independently
initialized filter state. One fixed carrier subtraction cannot make that
expression vanish on an open filter-state domain. This statement addresses
the direct phase readout; a constrained mean leaf or a different collective
observable needs its own proof. It does not turn the paper's smaller
supports into the target's eighteen-node geometry.

<a id="dpll-measurement-admission"></a>
## Preparation, observation and uncertainty obligations

The following are requirements of this comparison, not additional facts
about what the instrument can achieve. Missing evidence remains unavailable.

| Obligation | Required evidence before using the target certificate |
| --- | --- |
| Complete preparation | Joint bounds for every consumed source state and retained history, a declared initialization law and its map to the target source family. A phase-only specification does not establish a form/phase norm ball. |
| Support and capacities | Actual node ordering, contacts, weights, degree normalization and capacity map. Adding nodes or replacing the support changes the model rather than completing missing metadata. |
| Inputs and events | An independently specified action realizing the two target phase interventions, or a bounded finite-duration input model. A coupling switch is not automatically a phase-only jump with unchanged form. |
| Observation | A causal map from recorded voltages/timestamps to the proposed collective quantity, including extraction, bandwidth, aperture alignment and instrument memory. It must justify the fixed four normalized averages and held gain/offset if those are consumed. |
| Clock and units | A calibrated conversion from physical time to the structural clock, transformed in every row, with units for form, pressure, sensor gain and additive errors. Clock synchronization alone supplies no such conversion. |
| Uncertainty | Independent source, constitutive, timing, kernel and measurement bounds over the declared horizon. Nominal resolution, repeat dispersion and deterministic numerical enclosure width have different meanings. |
| Alternatives and evaluation | A prospective discriminator under equally informed alternatives, separated calibration and reserved response, and a frozen source/event/observation/numerical protocol. |

The [finite-noise theorem](../nodal/SINE_APERTURE_RESOLUTION.md#sine-aperture-resolution)
propagates declared budgets under its supplied model. It cannot certify
their physical attainability. A continuous mapped-law defect is not an
initial-state error or an additive sensor error; transporting one into the
other requires its own finite-flow argument. Repeated recordings do not
repair missing dynamics or an unsupported preparation map.

<a id="dpll-admission-boundary"></a>
## Evidence and remaining boundary

The direct bridge is not admitted. Equations (2)-(3) exclude its regular
current-state identification on the stated independent-history family.
They do not establish that the paper's actual initialization explores that
whole family. No justified alternative history restriction or bounded
complete-row reduction to the target is supplied here. Independently,
the preparation and observation obligations above lack the joint error
evidence needed to apply the fixed inference certificate. This is an
admission decision, not an empirical rejection of either physical model.

This review acquires no experimental response arrays, fits no trajectory
and evaluates no TNFR prediction against the published measurements.
Existing frozen software protocols, sources, responses and prospective
theorem prefixes remain unchanged. The source's own physical experiment
and a TNFR physical identification are separate evidential claims.

Any admission conclusion must name the proposed mapping and preparation
family it addresses. Failure of that bridge would leave other physical
realizations, independently controlled approximations and enlarged state
models open. It would not refute the source experiment, the conditional
TNFR theorems or the possibility of a later physical comparison. The sole
execution plan records any subsequent bounded dependency.

[paper]: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0171590
[supplement]: https://doi.org/10.1371/journal.pone.0171590.s001
[ti-application]: https://www.ti.com/lit/an/scha002a/scha002a.pdf
[pico-specification]: https://www.picotech.com/download/datasheets/picoscope-2205mso-data-sheet.pdf
