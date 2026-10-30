# Physical feasibility of the organized current discriminator

<a id="organized-electrical-physical-admission"></a>

## Decision and scope

**Physical status: not admitted. Experimental evaluation: not tested.**
The reviewed primary sources do not supply a laboratory realization of the
[frozen current discriminator](../nodal/ORGANIZED_ROBUST_CURRENT.md).
The failure has three distinct parts: its voltage observation has an extreme
clock-independent precision/aperture requirement; ordinary junction
capacitance changes an exact matrix premise; and independent preparation of
the full organized state has not been established. This is a scoped
admission result, not a physical rejection of either candidate law or TNFR.

The two conditional current envelopes and their frozen evidence remain
unchanged. This audit derives laboratory-unit consequences and reviews
methods/specifications; it acquires no experimental response, fits no
coefficient and reruns no frozen producer. A published method is not a
calibration certificate for a proposed assembly. The
[research program](../RESEARCH_PROGRAM.md#research-current-question) alone owns
the next task.

## 1. Complete object and physical reference

The subject remains the 45-node C9-by-C5 organization, its two complete
evolution rows, fixed support and held capacity. The comparison supplies
mu equal to 1 or 2, a common observation-start state class and one finite
input word. It does not prove alternative-law acquisition. Its exact
five-coordinate electrical carrier and bounded fine departures are
different objects: a prepared ideal five-island circuit cannot stand in
for the formation of five constituent C9 organizations.

Use the [fixed-support dictionary](../nodal/ORGANIZED_ELECTRICAL_PROTOCOL.md)
and the robust owner's fixed nominal coordinates. Write L for the C5
Laplacian, P for its neutral projection and L+ for its centered inverse.
The supplied physical reference has capacitance matrix c L+, conductance
matrix g P, five sine-current branches and voltage-phase constant a0.
Nominal calibration fixes

\[
 b_0=4g_0/c_0,\quad \kappa_0=a_{00}\gamma g_0/9,\quad
 v_0=a_{00}\gamma b_0/4,\quad
 j_0=a_{00}\gamma^2g_0^2/c_0,\quad \gamma=1/(1023\pi).
 \tag{1}
\]

The ratios C=c/c0, G=g/g0, A=a0/a00 and D are constant and independently
bounded by 1 +/- 1e-8. Laboratory time maps to s=b0 D t. Readout always
uses U=L+ V/v0 and nominal charge q_nom=c0 L+ V; actual charge is C q_nom.
Changing nominal conversion after viewing the current would change the test.

For numerical illustrations only, set a00=h_P/(4 pi e), where h_P denotes
Planck's constant, not the plateau length. The
[SI definitions, NIST SP 330 Section 2](https://www.nist.gov/pml/special-publication-330/sp-330-section-2)
give h_P=6.62607015e-34 J s and e=1.602176634e-19 C exactly.
Thus a00 is approximately 3.291059785e-16 V s. Those supplied physical
constants and the Josephson reference law are not derived from TNFR.
An exact SI constant does not eliminate uncertainty in its realization or
prove that a device obeys the reference law.

## 2. Voltage precision and time cannot be chosen separately

The frozen word has h=1e-5, fifteen unit-duration ramps, duration
T=15.00015, raw nominal-form error eta_raw=1e-14 and inward endpoint
aperture delta=1e-13. For every constant D,

\[
 t_{\rm plateau}=h/(b_0D),\quad t_{\rm ramp}=1/(b_0D),\quad
 t_{\rm word}=T/(b_0D),\quad
 t_{\rm aperture}\le\delta/(b_0D).
 \tag{2}
\]

All 150 scalars require their declared marking and windows. Clock accuracy
does not substitute for sensor bandwidth or an inward averaging kernel.

Because the rows of L+ contain 2/5, zero, -1/5, -1/5, zero, its infinity
norm is 4/5. Arbitrarily correlated voltage errors of size e_V therefore
give form error at most 4 e_V/(5 v0). The bound is sharp for the admitted
error class: signs aligned with one row attain it, whereas a common offset
cancels. Allocating the entire raw allowance to voltage requires

\[
 e_V\le {5\over16}a_{00}b_0\gamma\eta_{raw}.
 \tag{3}
\]

Any other raw conversion error consumes part of this allowance. Known
sensor correlations may permit a different bound, but cannot be assumed
from independent channel labels.

Suppose an independently specified sensor mode needs an averaging support
of duration at least t_min and offers a uniform absolute error guarantee
e_V for that mode. Fitting its support inside each frozen aperture and
using (3) requires

\[
 e_V t_{min}\le
 {5a_{00}\gamma\eta_{raw}\delta\over16D}
 \simeq {3.200075958\,10^{-47}\over D}\ {\rm V\,s}.
 \tag{4}
\]

For one apparatus guarantee over the entire D box, use D_max. This is a
necessary compatibility condition for the specified error-box admission,
not a universal uncertainty principle or a lower bound on every possible
sensor. It does not apply to an unmodeled charge, flux or current readout.
Positive averages must stay inside the window; a manufacturer's response
time alone does not establish their support or positivity.

The following unit conversions have D=1. They are illustrative scales,
not independently realizable component choices or new assessed protocols.

| b0 (s^-1) | Maximum aperture (s) | Maximum voltage error (V) | Word duration (s) |
| --- | --- | --- | --- |
| 1e-12 | 1e-1 | 3.2001e-46 | 1.500015e13 |
| 1 | 1e-13 | 3.2001e-34 | 15.00015 |
| 1e12 | 1e-25 | 3.2001e-22 | 1.500015e-11 |

Merely slowing the experiment lengthens the aperture while lowering the
allowed voltage error by the same factor. Increasing precision in the
existing interval calculation does not repair (4).

The nominal waveform has peak voltage (6/25) a00 b0. In the actual robust
decomposition its commanded part is multiplied by rho=1/(C D), with
background bounded by a00 b0 gamma M_mu. A single range covering the
nominal ramps and resolving the plateaus therefore encounters the
nominal commanded-component/error ratio

\[
 {96\over125\gamma\eta_{raw}}\simeq2.468236251\,10^{17}.
 \tag{5}
\]

This is not an ADC bit-depth theorem. Separate ranges or gated protection
change the range problem and need their own settling/loading evidence;
they do not alone remove (4).

## 3. Delivery, leakage and work in the same units

The robust law consumes u=rho f+d. Its physical current discrepancy obeys
d=delta_I/(9 b0 kappa0 C D). Consequently, for any part J of the word,

\[
 \int_{J_{lab}}\|\delta_I\|_\infty dt
 =9\kappa_0 C\int_J\|d\|_\infty ds.
 \tag{6}
\]

The total F=1e-4 and each plateau P_leak=1e-15 thus require respective
integrated absolute current errors at most a00 gamma g0 C F and
a00 gamma g0 C P_leak, in coulombs. A bound uniform over C uses C_min.
These are integrals of absolute error, not the absolute value of net
delivered charge; cancellation cannot certify them.

On a plateau the permitted mean absolute leakage is

\[
 {\overline{|\delta I|}_{max}\over j_0}
 ={4CD P_{leak}\over\gamma h}
 \simeq 1.28554\,10^{-6}CD.
 \tag{7}
\]

This is not a pointwise leakage bound. For the prescribed cubic
r(ramp)=3 ramp^2-2 ramp^3, k_c=e_c-1/5 and v=L+ k_c,

\[
 I_{cmd}=a_{00}g_0b_0 a(4v r''+k_c r'),\qquad
 \max\|I_{cmd}\|_\infty={48\over25}a_{00}g_0b_0.
 \tag{8}
\]

The largest increment has absolute amplitude 1/5. Substitution of the
five coordinates gives the peak at a ramp endpoint. Dividing (8) by the
allowed plateau average gives 48 h/(25 gamma C D P_leak), approximately
6.17059e13/(C D). This is an on/off integrated-delivery demand; it does
not require that many DAC levels. An L1 discrepancy budget supplies
neither a peak bound nor a bandwidth. Every switch, transient, sensor
loading current and nonneutral leakage needs its own admission.

For resources let K0=kappa0 v0 and use the robust coefficients
lambda=G/(C D), beta=A G^2/(C^2 D), eta=1/(A D). Direct power pairing gives

\[
 \int V^TI_{actual}\,dt
 =C K_0\int (L_{45}x)^TRu\,ds,\qquad
 E_{co}=9C K_0\left[{U^TLU\over2}
       +{\mu\beta\over\eta}\sum_{ring}(1-\cos\Delta\Phi)\right].
 \tag{9}
\]

Indeed I_actual=9 b0 kappa0 C D u and
(L45 x)^T R u=9 (L U)^T u. The positive part of total supplied power
scales by the same positive factor. Resistive heat equals C K0 times
fine loss minus hidden loss, each with coefficient lambda. The existing
finite resource majorants translate only on this same model and time
interval. K0 times full fine storage is not the energy of a constructed
fine electrical substrate. The dictionary starts at observation and
does not convert earlier constituent recovery/formation into lab time
or joules without an extended map.

## 4. Component shape is an independent obstruction

The exact capacitance matrix c L+ has zero adjacent off-diagonal entries.
Suppose the five ring junctions also have a positive mutual capacitance
c_J each, as permitted by the supplied RCSJ reference. Then

\[
 C_{phys}=cL^++c_J L\ne\widetilde cL^+
 \quad\hbox{for every }c_J>0\hbox{ and every scalar }\widetilde c.
 \tag{10}
\]

At each adjacent off-diagonal the left side is -c_J and the right side
is zero. This remains a disagreement on the neutral leaf: both matrices
annihilate the common mode, so equality there would imply full equality.
It cannot be covered by widening only the scalar capacitance interval.
Unequal branches or ground capacitance introduce additional changes;
sensor loading can also change the dynamical law.

Equation (10) excludes exact shape matching for this explicit passive
realization with nonzero ring capacitance. It neither sets a universal
parasitic lower bound nor excludes a controlled approximate reduction.
A small shape departure needs a matrix perturbation theorem carrying
both rows, preparation, memory, observation and discrimination error. It
cannot be assigned the intrinsic fine-state remainder by terminology.

On the neutral collective leaf, V=v0 L U gives
`q=c*v0*(I+(c_J/c)*L^2)*U`. The
[matrix-response comparison](../nodal/ORGANIZED_EFFECTIVE_LAW.md)
owns the resulting changed rows, storage and charge-coordinate contract.
This collective reference supplies no automatic transfer of the fine-source
acquisition or frozen discrimination certificates.

## 5. Independent source review and preparation gap

The [methods receipt](../../docs/assets/organized_electrical_feasibility/method-sources-v1.json)
records the reviewed sources and retrieval availability. Retrieved-byte
hashes, where available, are provenance associations, not authentication.
The following are source facts; their admission consequences are our
inferences. No reviewed figure or response becomes reserved evidence.

| Primary source | Supported facts | Remaining admission |
| --- | --- | --- |
| [Babenko et al. (2020)](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=929317), pp. 937-939 | RCSJ and voltage-phase equations; a 1500-junction array at 4 K with 5 GHz excitation and calibrated microwave measurements. The authors discuss junction variation and nonsinusoidal response. | No C5 matrix, 45-state preparation or error guarantee for this word. Junction capacitance is explicit in the reference equation. |
| [English et al., arXiv v2](https://arxiv.org/pdf/1305.0327v2), pp. 1-2 | Phase-sensitive observation through a superconducting loop, flux transformer and DC SQUID; 10 mK base temperature and a nonhysteretic operating condition. | An independent phase method is possible, but its loop state, loading and calibration must be retained. It does not establish our initial phase collar. |
| [AH2700A brochure](https://www.andeen-hagerling.com/ah2700a.pdf), pp. 1-2 | Published 1 kHz capacitance accuracy of +/-5 ppm. | This specification alone is 500 times wider than the frozen 0.01 ppm scalar allowance, and is not a matrix calibration at the proposed operating conditions. |
| [NIST impedance metrology](https://www.nist.gov/programs-projects/farad-and-impedance-metrology), Major Accomplishments | A specialized resistor/capacitor comparison reports combined standard uncertainty of 5 ppb near 1592 Hz. | The scalar budget is not universally unattainable; this is not a complete cryogenic matrix certificate or a deterministic error bound. |
| [Keithley 2182A manufacturer page](https://www.tek.com/en/products/keithley/low-level-sensitive-and-specialty-instruments/nanovoltmeter-model-2182a), Features | Two channels, 1 nV sensitivity, typical 15 nV peak-to-peak noise at 1 s response and 40-50 nV at 60 ms. | No five-coordinate synchronized transfer, positive inward kernel or simultaneous absolute guarantee meeting (3)-(4). Response time, sensitivity and typical noise must not be substituted for those distinct premises. |
| [JCGM 100:2008](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf), Sections 2.3.5, 6.2, 7.2.5 | Expanded uncertainty carries a stated coverage interpretation; multiple outputs require their uncertainty/covariance information. | RMS, typical noise and marginal coverage intervals do not supply the frozen simultaneous deterministic error box. |

For magnitude context only, the hypothetical pair e_V=1 nV and
t_min=1 ms has product 1e-12 V s, exceeding (4) by about 3.1e34 at D=1.
This pair is not asserted to be a manufacturer guarantee or a fundamental
noise floor. The reviewed voltmeter specifications cannot certify the
required pair, and do not supply the observation kernel in the first place.
Neither a finite typical peak-to-peak sample nor calibration uncertainty
may be promoted to a hard bound on every future reading. A statistical
protocol would instead need a joint confidence event over all readings,
preparation and calibration; the deterministic theorem could apply
conditionally on that event if all its other premises were met.

Preparation remains separate from precision. The model needs independently
justified bounds on the complete initial signed forms, hidden constituent
deviations and admitted relative phase set. Zero voltage does not establish
relative phases or internal
deviations. Even the ideal carrier allows identical U=0 with Phi=0 or
Phi=(d,-d,0,0,0), with small positive d: both have winding zero and zero
voltage, but different sine-current vectors. Winding alone is insufficient.
The prior need not identify an exact phase, yet must be independently
justified as a set under both candidate laws. None of the reviewed sources
supplies that preparation or extends the physical map through the TNFR
acquisition.

## 6. Consequence for research

There is no admitted physical candidate in this review. Equation (4)
identifies a limitation of the frozen endpoint-voltage method, (10) an
exact shape mismatch for a specified junction-capacitance realization,
and the preparation audit missing independent evidence. These are
different conclusions and must not be combined into a universal no-go.

A successor must explicitly change the observation/protocol or prove an
appropriate approximation; repeating the same arithmetic at higher
precision supplies neither. Finite-window observables, direct current or
flux sensing, a prepared collective reference and a generatively acquired
organization have different obligations. Any successor must keep the
source distinction, declare loading and matrix deviations, retain its own
law and work, and freeze its unfitted prediction before response selection.
The conditional current result is useful as an ideal comparison, not as
evidence that the required physical apparatus already exists.

The [independent algebra controls](../../tests/physics/test_organized_electrical_feasibility.py)
check inverse-matrix error propagation, scale cancellation, current/work
conversion and a positive-capacitance counterexample. They establish no
instrument performance. No runtime capability or public API is added by
this documentary and mathematical admission audit.
