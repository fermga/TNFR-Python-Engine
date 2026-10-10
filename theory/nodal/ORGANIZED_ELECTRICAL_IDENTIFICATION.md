# Operational identification of the collective electrical model

<a id="organized-electrical-operational-identification"></a>

## Question and decision scope

The [electrical construction](ORGANIZED_ELECTRICAL_REALIZATION.md) and
[actual-source comparison](ORGANIZED_ELECTRICAL_SOURCE.md) establish a
conditional post-contact description. This owner asks which independently
measurable relation remains after calibrating that description, and whether
the existing preparation is already an admissible electrical experiment.

There are two results. Independently calibrated capacitance and conductance
leave a prospective, unfitted sine-current amplitude for the fixed law.
However, its dimensionless coefficient is supplied, and the literal
instantaneous phase events in the existing source history have no finite-work
realization in the unchanged electrical reference. This is a scoped
preparation obstruction, not an obstruction to all physical preparation or
to the valid finite post-contact theorem.

The mathematical results below are symbolic. No physical data, new source
response, parameter search, numerical campaign or laboratory apparatus is
admitted. Numerical controls use synthetic records and independent algebra;
the earlier frozen source evidence remains unchanged.

## 1. Complete reference, observation and calibration boundary

Retain the six marked electrical islands, floating neutral charge leaf,
fixed same-slot wheel support and collective circular observation of the
electrical owner. Use its notation L for the wheel Laplacian, B for an
oriented wheel incidence, `P=I-11^T/6`, and the supplied Josephson constant
`a_0=hbar/(2e)>0`. Physical quantities use coulombs, volts, seconds, farads,
siemens and amperes; structural quantities retain their original units.

The reference rows are

\[
 q=CV,\qquad
 \dot q=-GV-B\operatorname{diag}(j_e)\sin(B^T\Phi),
 \qquad \dot\Phi=V/a_0.                              \tag{1}
\]

The declared dictionary remains `q=9*kappa*(X-r*1)`, `tau=tau_0+b*t_lab`,
and unchanged circular Phi, with positive common scales `kappa,b` and
retained neutral reference r. Arbitrary sensor mixing, changed phase
coordinates, hidden dynamic circuitry and a different background-charge
leaf are outside this particular comparison. Voltage differences require
one consistent floating gauge; a common unmeasured phase is not set to a
new physical value for every record.

An operational comparison separates the following information:

| Information | Independent premise and use |
| --- | --- |
| Node labels, component connections and reference | Fix before measuring or comparing records; preserve all fifteen capacitor/resistor pairs and ten wheel junctions |
| C and G | Calibrate the full effective linear component matrices from separate records, with the junction contribution, loading, frequency/temperature regime and uncertainty accounted for. A single terminal trace does not determine these matrices |
| Charge and voltage readings | Supply calibrated transfer, offsets, aperture and noise. Computing q from independently calibrated `C V` is a model-based charge inference; it cannot independently validate that same capacitor law |
| Relative phase readings | Supply an independent phase-sensitive transfer model, or retain known initial phase and integrate `V/a_0`. The latter consumes the Josephson law and cannot test it; it also does not recover unknown initial relative phases |
| Reserved junction observation | Measure the current-phase profile with the independent phase readout. Its sine amplitude is reserved when testing (4); a maximum current alone does not establish a sinusoidal profile |
| Preparation, input and laboratory time | Record actual initial collective state, physical actuation, support changes, laboratory clock and all work; preserve their uncertainty when predicting later records |

Reconstructing phase from `arcsin(I/j)` while testing the sine-current law
would be circular. Integrating measured voltage requires the initial phase
and does not resolve that missing information by itself.

The constitutive reference is supported by equations (1)-(2) of
[Babenko et al. (2020)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929317).
That paper describes separate impedance and instrument calibration, and
reports model discrepancies attributed to nonsinusoidal behavior and
junction variation. Its apparatus and measured parameters are not evidence
for this TNFR realization or its tolerances. Direct phase-sensitive
measurements of nonsinusoidal profiles, including their loop/sensor model,
are reported by [English et al. (2016), arXiv v2](https://arxiv.org/pdf/1305.0327v2).
These primary sources justify keeping profile and sensor assumptions explicit;
neither supplies a TNFR calibration or reserved dataset. Sources reviewed
2026-10-10.

## 2. An unfitted relation after independent linear calibration

**Conditional characterization.** On an open neutral charge and relative-phase
domain, the fixed dictionary realizes the complete quarter sine law if and
only if, for positive scalars c, g, j,

\[
 C=cL^+,\qquad G=gP,\qquad j_e=j\text{ on all ten edges},\qquad
 \frac{cj}{a_0g^2}=\gamma^2,
 \quad \gamma=\frac1{1023\pi}.                        \tag{2}
\]

The statement is within the declared floating six-island reference and
observation class, not a classification of every possible electrical map.
Its necessity follows from the phase and form rows in the electrical owner:

\[
 c=\frac{36\kappa}{a_0b\gamma},\qquad
 g=\frac{9\kappa}{a_0\gamma},\qquad
 j=\frac{9b\kappa\gamma}{4}.                         \tag{3}
\]

Equality of the nodal sine fields on an open relative-phase domain forces
each edge coefficient. Differentiate a node's current with respect to one
neighbor's phase: that single edge coefficient is isolated. Cycle-current
ambiguity in a single record does not create a freedom in the complete
nonlinear field. Eliminating the two scales in (3) yields

\[
 \boxed{b=4g/c,\qquad \kappa=a_0\gamma g/9,
 \qquad j_{\rm pred}=a_0\gamma^2g^2/c.}               \tag{4}
\]

Conversely, inserting the positive scales (4) and relation (2) into (3)
recovers both full rows and their storage/work relation. This proves
sufficiency. After exact independent C/G calibration, both scales are
unique; changing either to fit a reserved j would contradict that calibration.
The scalar c has units farads, g siemens, j amperes; `cj/(a_0*g^2)` is
dimensionless. It is independent of the arbitrary charge and clock scales.

The complete component shape must be checked before using (4). From the
proved wheel inverse, the mutual-capacitance coefficients are
`5/396`, `41/396`, `11/396` times c for adjacent rim, nonadjacent rim and
rim-hub pairs, respectively. Every resistor branch has conductance `g/6`.
Equivalently, symmetric floating matrices satisfy `C L=cP` and `G=gP`.
Although `c=tr(CL)/5`, `g=tr(G)/5`, traces alone cannot establish those
matrix identities. Changing two branch values can preserve a trace and
still violate the required shape.

The same restriction must hold across independently prepared states of the
same fixed assembly. Preparation cannot retune c, g, j, gamma or sensor
gains. A conventional circuit with calibrated C and G but all amplitudes
`2*j_pred` is a positive passive reference outside this fixed-law dictionary;
it gives a concrete distinguishable alternative on an open phase domain.
A conventional circuit satisfying (2) is exactly equivalent on the ideal
carrier. No response in that common model can distinguish the two descriptions.

## 3. Why the number is not a derived universal constant

The fixed coefficient comes from the earlier specified weights
`e=1023/1024`, `w=1/1024`, giving `gamma=w/(e*pi)` in the
[complete-law owner](ORGANIZED_PATTERN_CONSTITUTIVE_FREEDOM.md).
It is not selected by the nodal identity, neutrality, positive dissipation
or the passive circuit construction. The
[constitutive comparison class](COMPOSITION_COMPATIBLE_CONSTITUTIVE_LAWS.md)
explicitly retains positive coefficient and current-profile freedom.

If gamma is allowed to be fitted after measuring a positive triple c,g,j,
then

\[
 \gamma_{\rm fit}=\sqrt{\frac{cj}{a_0g^2}},\qquad
 b=4g/c,\qquad \kappa=a_0\gamma_{\rm fit}g/9           \tag{5}
\]

matches every such triple with the required component shapes. The scalar
restriction has disappeared. Thus (2) tests the previously fixed complete
law, not TNFR with unrestricted constitutive coefficients. Replacing the
sine profile by another admitted current law also needs its own comparison;
its formation, finite source bounds and sensor inference do not transfer
merely because equilibria or tangents match.

There is a limited composition statement. If a different organization of m
nodes independently admits the same exact collective law, phase map and
passive coarse inverse, replacing 9 by m in the aggregate charge map gives

\[
 c=\frac{4m\kappa}{a_0b\gamma},\quad
 g=\frac{m\kappa}{a_0\gamma},\quad
 j=\frac{mb\kappa\gamma}{4}.
\]

The ratio (2) is unchanged. This is conditional cancellation of an admitted
replication factor; it does not prove arbitrary-scale formation, physical
scale invariance or the same law at every scale.

## 4. Uncertainty, actual source and a possible reserved balance

Let `c in [c_-,c_+]`, `g in [g_-,g_+]`, and
`a_0 in [a_-,a_+]` be independently admitted positive calibration boxes.
The exact monotone envelope for the fixed mathematical gamma is

\[
 j_{\rm pred}\in
 \left[\frac{a_-\gamma^2g_-^2}{c_+},
       \frac{a_+\gamma^2g_+^2}{c_-}\right].            \tag{6}
\]

The implementation uses the shared outward enclosure of gamma. Disjoint
reserved amplitude bounds exclude the fixed-map relation under the supplied
premises. Overlap is necessary compatibility only: it proves neither the
physical premises nor a joint noisy trajectory. Correlations discarded by
these boxes can weaken exclusion. If capacitance or conductance shape
intervals have empty intersection, calibration already excludes this
dictionary; averaging them into a best fit would erase that finding.

The [actual-source theorem](ORGANIZED_ELECTRICAL_SOURCE.md) starts its ideal
comparator at the actual collective coordinates. Its very small intrinsic
error is not an experimental initialization error budget. With initial
Euclidean coordinate uncertainty `e_X,e_Phi`, its same coupled argument gives

\[
 E_X\le\frac{e_X+c_We_\Phi+h\rho}{1-c_W^2},\qquad
 E_\Phi\le\frac{e_\Phi+c_We_X+c_Wh\rho}{1-c_W^2},
 \quad c_W=3\gamma_+h/2.                              \tag{7}
\]

Physical sensor, coefficient and clock errors require their own additional
terms; (7) holds for the otherwise unchanged model. On that same source,
`||R||_1<=rho` and `sum R=0`. For any union A of complete children, one
possible reserved observable is the integrated balance

\[
 \Delta q_A+\int (GV)_A\,dt+
 j_{\rm pred}\int[B\sin(B^T\Phi)]_A\,dt
 =9b\kappa\int R_A\,dt.                             \tag{8}
\]

Across the original structural horizon h its intrinsic right-hand bound is
`9*kappa*h*rho/2`. Independent measurements, integration aperture, inputs,
initial phases and uncertainty must be declared before (8) becomes a physical
test. It avoids differentiating noisy charge but does not eliminate the
phase observation requirement. The bound is an allowed model discrepancy,
not an observable lower bound or a TNFR-only signal. Higher precision on
that intrinsic remainder cannot replace the missing physical information.

## 5. A concrete preparation obstruction: instantaneous phase events

The original fifteen observation events occur on the five-child host before
donor attachment. The six-island contact capacitance cannot simply be used
for those earlier events. Its ideal host carrier does, however, have a
separate exact passive realization: repeat the electrical derivation for C5.
The centered inverse of the C5 Laplacian has diagonal `2/5`, adjacent
entries zero and nonadjacent entries `-1/5`, so

\[
 C_5=cL_{C5}^+,\qquad G_5=gP_5.                       \tag{9}
\]

Five positive mutual capacitors `c/5` join the nonadjacent pairs, all ten
resistor pairs have conductance `g/5`, and five ring junctions have amplitude
j. These matrices have the same scalar formulas (3) if the same dictionary
is stipulated. This admits the ideal fixed-support host circuit, not the
uncertain acquisition, observation actuator or later support switch.

The source's event changes phase by `Delta Phi=a*k_c`, where
`k_c=e_c-1/5`, and leaves form unchanged. Its three amplitudes in each block
are `a=epsilon,-2*epsilon,epsilon`, `epsilon=1/10`. Each is a nonzero
relative circular change, not a common gauge or an integer full turn.

**Finite-work obstruction.** On a fixed floating circuit with positive
capacitance and conductance on the neutral subspace, the Josephson row
`dot Phi=V/a_0`, imposed in its integral solution sense, makes relative phase absolutely continuous when voltage
has finite squared integral. Finite dissipated work guarantees that property.
It therefore excludes a nonzero relative phase jump in zero time.

More quantitatively, any pulse of laboratory duration d>0 realizing a
declared continuous lifted increment Delta Phi satisfies

\[
 \begin{split}
 \int_0^d V^TGV\,dt&\ge
       \frac{a_0^2}{d}\Delta\Phi^TG\Delta\Phi,\\
 \int_0^d E_C(t)\,dt&\ge
       \frac{a_0^2}{2d}\Delta\Phi^TC\Delta\Phi,\\
 \mathop{\rm ess\,sup}_{[0,d]} E_C(t)&\ge
       \frac{a_0^2}{2d^2}\Delta\Phi^TC\Delta\Phi,
 \qquad E_C=\tfrac12V^TCV.
 \end{split}                                         \tag{10}
\]

Each follows from the positive quadratic-form Cauchy--Schwarz inequality
and `integral V=a_0*Delta Phi`. The integrated capacitor energy is an
energy-time quantity, not work; Joule heat in the first row is dissipated
energy. A finite initial energy reservoir cannot make the first row bounded
as d tends to zero. The result concerns the fixed reference and declared lift;
other lifts or supports need their own endpoint and work accounting.

For the host event, `k_c^T C_5 k_c=2c/5` and
`k_c^T G_5 k_c=4g/5`, so these lower bounds become respectively

\[
 \frac{4a_0^2a^2g}{5d},\qquad
 \frac{a_0^2a^2c}{5d},\qquad
 \frac{a_0^2a^2c}{5d^2}.                              \tag{11}
\]

This proves that the literal instantaneous phase-only word has no finite-work
implementation in this reference. It does not rule out a finite-duration
current pulse or another preparation reaching an admissible endpoint.
No endpoint form restoration, fine-source inclusion, winding retention or
finite-pulse budget follows from the lower bounds alone. The original
structural event-work budget cannot be converted into a payment for (11).

The separate [finite-input construction](ORGANIZED_ELECTRICAL_ACTUATION.md#finite-actuation-result)
now supplies those source, identity and resource obligations for one
approximate intervention, retaining a nonzero endpoint form error. It does
not admit the original complete observation word or a physical actuator.

There is also a conditional clock restriction. If a future model extends
one constant calibrated b over the entire old preparation and contact,
its isolated recovery/contact duration ratio remains `8e10/1e-3=8e13`.
Changing b cannot alter this ratio. For an independently admitted minimum
resolvable contact time d_min and maximum recovery time D_max, that direct
history requires `D_max/d_min>=8e13`. These physical limits have not been
supplied. The current theorem only maps post-contact evolution, so neither
this clock extension nor physical preparation energy is already certified.

## 6. Detached instrument and gate closure

The [component inference owner](../../src/tnfr/physics/_organized_electrical_identification.py)
accepts only primitive positive component values or shared outward intervals.
Capacitances and conductances use all fifteen lexicographically ordered
island pairs. It intersects the individual scalar-shape constraints before
predicting j; the comparison consumes the ten independently supplied wheel
sine amplitudes in the same pair ordering restricted to wheel edges.
Reserved amplitudes never enter calibration. A comparison rebuilds calibration
from primitives rather than consuming a cached report or fitted scale.

Malformed scalars, unsupported positive interval domains and wrong coordinate
counts reject explicitly. Empty shape or amplitude intersections give scoped
exclusions. A surviving interval relation is only necessary compatibility;
the instrument admits no apparatus, phase sensor, source history or current
profile. In particular a supplied sine amplitude is not proof that a physical
current-phase curve is sinusoidal. The
[synthetic inference controls](../../tests/physics/test_organized_electrical_identification.py)
and [exact pulse-bound controls](../../tests/physics/test_organized_electrical_pulse_obstruction.py)
exercise independent algebra, invalid domains and counterexamples without
executing an acquired-source or physical response.

The operational gate closes with an unfitted conditional relation (2)-(4)
and a specific obstruction (10)-(11) to directly implementing the existing
instantaneous preparation word. No physical evaluation is admitted. The
remaining task belongs solely to the [execution plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate):
one admitted pulse must become a justified finite observation protocol before
the source-inclusive circuit comparison can support an experimental protocol.
The supplied gamma and Josephson constant remain
independent constitutive premises throughout.
