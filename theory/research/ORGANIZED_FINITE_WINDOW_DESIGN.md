# Loaded finite-window observation of a prepared collective reference

<a id="loaded-window-design-declaration"></a>
## Declared question and candidate

This design admission follows the [prepared matrix-response theorem](../nodal/ORGANIZED_EFFECTIVE_LAW.md).
It asks whether a finite observation with an explicit passive sensor can
preserve a predictive consequence and support an independent physical test.
The analytical declaration and result below remain reusable under their stated
premises; the [research program](../RESEARCH_PROGRAM.md#research-current-question)
alone selects further work. No apparatus, measured response, source acquisition
or fine-node extension is supplied by this mathematical design.

Keep the complete C5 law, signed forms U, circular phases Phi, held unit
capacities, fixed support, normalized structural clock tau and exact
`gamma=1/(1023*pi)` of the owner. Write `v=(1,-1,0,0,0)`,
`W_r=I+r*L^2`, `y=v^T*L*U`. Add one sensor state z with

\[
 W_r U'=(-LU+\gamma S(\Phi))/4+f-kv(y-z),\qquad
 \Phi'=\gamma LU/4,\qquad \tau_s z'=y-z.
 \tag{1}
\]

The sensor loads the form row through its actual current; it is not an
additive observation error. The selected observable y differs from the
earlier `v^T*U`: it is proportional to a physical differential voltage in
the declared electrical dictionary.

| Primitive | Fixed analytical declaration |
| --- | --- |
| Response family and control | Fixed `r in [1/100,1/10]`, compared separately with `r=0` |
| Full preparation | The earlier centered winding `w=+1` or `w=-1` reference, `||U(0)||_2+||theta(0)||_2<=delta=2^-26`; the same continuous perturbation lift and closing-edge offset |
| Sensor preparation | Independently bounded `|z(0)|<=delta_s=2^-26` |
| Input | `f=a*v` on `[0,T]`, zero afterwards; `a=1/16`, `T=1/1024` |
| Sensor law | `k=1/64`, `tau_s=T`; no resets, phase forcing or support events |
| Observation | `J=(1/T)*integral_0^T z(tau) d tau`, with separate additive error `|e|<=eta=2^-26` |
| Identity criterion | Every oriented phase gap remains within `rho=1/10` of its prepared `2*pi*w/5` during the pulse and later unforced evolution |
| Matrix-response criterion | Whole-preparation observed intervals for positive r and r=0 are disjoint with separation `>2^-21` |
| Phase-sensitive control | Compare with a passive linear reference having the same r, form/sensor initialization, input, clock and sensor but omitting the sine-current branch; determine whether permitted records must be disjoint or can overlap |
| Physical admission | Independent complete preparation, loaded transfer and matrices, delivery, clock and observation error/coverage under the same laboratory protocol |
| Calculation | Exact rational inequalities and certified exponential/trigonometric bounds; no trajectory, parameter scan, fitted response or physical dataset |

The phase-blind reference tests what this observable resolves after linear
calibration. It does not override independently established junction data.
An independently justified sine-current law would be additional evidence,
not something inferred from an unresolved pulse observation.

Source review concerns terrestrial methods and specifications only. Earlier
frozen protocols, source archives, responses and the separate endpoint-voltage
feasibility review remain unchanged. A conditional sensor equation is not a
manufacturer performance claim or a hardware implementation.

<a id="loaded-window-design-result"></a>
## Analytical result and admission boundary

Revision `34baddca` freezes the declaration above. The following calculations
use its complete loaded law, without a trajectory, measured response or
parameter scan. The candidate retains the prepared identity and separates
the positive-r matrices from r=0. However, its permitted records overlap
those of the declared phase-blind linear reference. It therefore does not
admit an organization-sensitive physical test. Independent full preparation,
component/input/clock calibration and observation coverage also remain open.

### Own storage, identity and resources

Write `H_r=U^T*L*W_r*U/2+V(Phi)`,
`V=sum_edges[1-cos(Phi_j-Phi_i)]` and `V_*=5[1-cos(2*pi/5)]`, as in the
[matrix-response owner](../nodal/ORGANIZED_EFFECTIVE_LAW.md#prepared-matrix-response-result).
For the loaded law, the correct storage and its derivative are

\[
 H_{r,s}=H_r+\frac{k\tau_s}{2}z^2,\qquad
 H_{r,s}'=-\frac14\|LU\|^2-k(y-z)^2+(LU)^Tf.
 \tag{2}
\]

Indeed, the form row adds `-k*y*(y-z)` to the original balance, while
the sensor contributes `k*z*(y-z)`. Their sum is the displayed loss.
Treating the sensor as a detached reading would omit both terms. The form
mean and common mean of the declared phase lift remain fixed: v and f are
neutral and the phase row is unchanged. The solved input and loading channels
are `M_r*f` and `-k*M_r*v*(y-z)`.

Let theta be the declared centered perturbation lift. In the relative cell
`max|D theta|<=rho=1/10`, the earlier cosine bound gives
`V-V_*>=||D theta||^2/10`. Initial admission follows from `2*delta<rho`.
A first boundary contact would require excess storage
`E_s=H_{r,s}-V_*>=1/1000`. Using `||L||<=4` and completing the square in
(2), including the sensor's nonnegative energy, gives

\[
 E_s(0)\le(4+32r_{\max})\delta^2+\frac{kT}{2}\delta_s^2,
 \qquad E_s'\le\|f\|^2,\qquad
 E_s(\tau)\le\overline E_s=E_s(0)_{\rm upper}+2a^2T
 <2^{-16}<\frac1{1000}.
 \tag{3}
\]

This excludes boundary contact throughout the pulse. After T, (2) is
nonincreasing, so the identity remains inside the collar for all subsequent
unforced time. The globally Lipschitz sine source and constant invertible
matrices ensure continuation. This is retention of the supplied identity,
not acquisition. The full preparation cost includes V_*; the excess in (3)
is neither total storage nor a source for the external input.

For finite response bounds put `X=||U||+||theta||`. The sine source obeys
`||S(Phi_w^*+theta)||<=4||theta||` globally; `||M_r||<=1`,
`||v||<2`, `||Lv||=sqrt(20)<5`, and `gamma<1`.
The stable sensor convolution bounds `sup|z|<=delta_s+5*sup X`.
Integrating both live evolution rows gives, when `(2+20k)T<1`,

\[
 \sup_{[0,T]}X\le Z=
 \frac{\delta+2aT+2kT\delta_s}{1-(2+20k)T},\qquad
 B=(1+20k)Z+2k\delta_s,
 \tag{4}
\]

and `||U(tau)-a*tau*M_r*v||<=delta+tau*B`. These inequalities hold without
freezing phase or relying on the identity flag. In particular, they retain
unknown sensor initialization and back-action. The separate resource bounds
are

\[
 \int_0^T|(LU)^Tf|\,d\tau\le8aTZ,\quad
 \int_0^T\frac14\|LU\|^2\,d\tau\le4TZ^2,\quad
 \int_0^Tk(y-z)^2\,d\tau\le kT(\delta_s+10Z)^2.
 \tag{5}
\]

Equation (2) retains the signed supplied work and both continuous losses.
The growth allowance `2a^2T` in (3) is not an equality for delivered work.

### Sensor memory and finite matrix separation

The exact sensor solution is

\[
 z(\tau)=e^{-\tau/\tau_s}z(0)+\frac1{\tau_s}
 \int_0^\tau e^{-(\tau-u)/\tau_s}y(u)\,du.
 \tag{6}
\]

Eliminating z from the form row leaves the term
`-k*v*y(tau)+k*v*exp(-tau/tau_s)*z(0)` plus the convolution in (6)
multiplied by k v. Thus a reduced form law retains both sensor memory and
initialization. Replacing z by y instantaneously would change the model.
Integrating (6) over the declared window yields

\[
 J=\frac{\tau_s}{T}(1-e^{-T/\tau_s})z(0)
 +\frac1T\int_0^T[1-e^{-(T-u)/\tau_s}]y(u)\,du.
 \tag{7}
\]

For `tau_s=T`, let `alpha=1/2-exp(-1)>1/8`. Equation (4) then implies

\[
 J=aT\alpha\,g(r)+R_J^{\rm actual},\qquad
 |R_J^{\rm actual}|\le R_J=\delta_s+5\delta+\frac{5T}{2}B,
 \qquad g(r)=v^TLM_rv=\frac{6+20r}{1+15r+25r^2}.
 \tag{8}
\]

The positive kernel in (7), with total weight at most one, bounds the
initialization and constant port remainder; its first moment is at most T/2.
The neutral C5 identity `L^2-5L+5I=0` supplies the exact gain. Its derivative
has numerator `-70-300r-500r^2`, so it decreases strictly. For the fixed
family, `g in [32/11,2480/461]`, whereas `g(0)=6`.
Writing `b_J=R_J+eta`, the entire permitted record sets lie within

\[
 \mathcal J_+=
 [aT\alpha(32/11)-b_J,\ aT\alpha(2480/461)+b_J],\qquad
 \mathcal J_0=[6aT\alpha-b_J,\ 6aT\alpha+b_J].
 \tag{9}
\]

Both use the same alpha. Exact rational substitution already gives

\[
 \inf\mathcal J_0-\sup\mathcal J_+
 \ge\frac{aT}{8}\frac{286}{461}-2b_J>2^{-21}.
 \tag{10}
\]

The implementation also encloses alpha tightly through the shared rational
exponential bounds. Independently widening alpha to unrelated coarse values
in the two records would discard their correlation and needlessly weaken
the separation. This is a finite loaded-response distinction between
specified matrices, conditional on the complete preparation and error bounds.

### What the same observation cannot distinguish

The declared reference keeps the same r, U(0), z(0), input, clock and sensor:

\[
 W_r\widehat U'=-L\widehat U/4+f-kv(\widehat y-\widehat z),
 \qquad \widehat y=v^TL\widehat U,\qquad
 T\widehat z'=\widehat y-\widehat z.
 \tag{11}
\]

It consumes no phase coordinate. Its own passive storage is
`Uhat^T*L*W_r*Uhat/2+k*T*zhat^2/2`; its derivative is (2) with hats.
The comparison matches calibrated linear components and omits the sine-current
branch. Independently established junction data could exclude this reference;
the present calculation tests what the finite pulse adds by itself.

For the actual complete law, `||theta(tau)||<=delta+gamma*tau*Z`.
Write `E=sup_[0,T]||U-Uhat||`. Equal sensor initialization and (6) give
`sup|z-zhat|<=5E`. The form difference therefore satisfies

\[
 E\le(1+20k)TE+\gamma\delta T+\frac{\gamma^2ZT^2}{2},\qquad
 |J-\widehat J|\le5E\le
 \frac{5\bar\gamma(\delta T+\bar\gamma ZT^2/2)}
 {1-(1+20k)T},\quad \bar\gamma=\frac1{3000}>\gamma.
 \tag{12}
\]

The first term bounds both diffusion and actual sensor back-action; the
remaining terms bound the full sine source about the twist. These are
global Lipschitz estimates, not a tangent reduction. The fixed declaration
gives `Z<1/8000`. Even the looser substitution `Z<=1/4000` proves

\[
 |J-\widehat J|\le
 \frac{47003}{1930258022400000000}<2^{-44}<\frac\eta{100}.
 \tag{13}
\]

This establishes overlap of actual allowed noisy records, not just overlap
of two outer enclosures: the common record `(J+Jhat)/2` requires opposite
errors of magnitude `|J-Jhat|/2<=eta` in the two models. Hence no uniformly
error-robust discriminator exists for this pulse, scalar observation and
declared error class. It does not assert equality of the noiseless responses,
equality for every error realization, or impossibility for another experiment.
The matrix separation in (10) remains valid. It cannot be attributed to a
resolved phase-organization effect by this observation.

### Independent preparation cannot be inferred from energy and winding

Within an independently admitted centered collar, `L>=I` on neutral forms
and phases gives

\[
 E_r\ge\tfrac12\|U\|^2+\tfrac1{10}\|\theta\|^2,
 \qquad (\|U\|+\|\theta\|)^2\le12E_r.
 \tag{14}
\]

Thus an independent `E_r<=delta^2/12` certificate is sufficient there,
with separate sensor admission. It does not establish the collar itself
or make phase-dependent energy an independently available reading.

There is a global counterexample even with exact target energy. Set winding
w=1, kappa=2*pi/5, U=z=0 and choose oriented gaps

\[
 d(q)=\left(q,\frac{2\pi-q}{4},\frac{2\pi-q}{4},
                    \frac{2\pi-q}{4},\frac{2\pi-q}{4}\right),
 \quad V(q)=5-\cos q-4\cos((2\pi-q)/4).
 \tag{15}
\]

At `q=kappa+rho`, the local bound used in (3) gives
`V(q)-V_*>=rho^2/8=1/800>0`. At the limiting endpoint `q=pi`,
`V(pi)=6-2*sqrt(2)<16/5`, whereas
`V_*=(25-5*sqrt(5))/4>55/16`; hence `V(pi)-V_*<-19/80`.
Continuity supplies a `q0 in (kappa+rho,pi)` with exactly `V(q0)=V_*`.
All five gaps are in `(0,pi)` and sum to `2*pi`, so winding remains one.
Reconstruct a continuous lift by cumulative gaps and subtract its perturbation
mean. This preserves the gaps and places both states on the same mean leaf.

The constructed state and the prepared twist have identical zero nodal and
sensor voltages, winding, means and total own storage, yet the constructed
state lies outside the required collar. Its existence follows without
locating a root or evaluating a trajectory. Consequently these aggregate
preparation quantities cannot globally certify the local phase prior.

### Physical dictionary and method evidence

Under the separately supplied electrical reference, choose `c,g,a0>0` and
`r=c_J/c`. The laboratory clock is `tau=b*t_lab`, with both clocks zero
at observation start. With P the neutral projector, retain

\[
 C_{\rm phys}=cL^++c_JL,\quad G=gP,\quad b=4g/c,\quad
 v_0=a_0b\gamma/4,\quad V=v_0LU,\quad
 j=a_0\gamma^2g^2/c,\quad I_{\rm ext}=cv_0bf.
 \tag{16}
\]

A series resistor-capacitor branch across the v port has capacitor voltage
`V_s=v0*z`, conductance g_s and capacitance C_s. Its exact dictionary is

\[
 k=\frac{g_s}{cb}=\frac{g_s}{4g},\qquad
 \tau_s=\frac{bC_s}{g_s},\qquad k\tau_s=\frac{C_s}{c}.
 \tag{17}
\]

Thus the fixed candidate requires `g_s=g/16` and `C_s=c/65536`.
Kirchhoff's current equation and the capacitor equation give (1), including
loading. The physical storage scale is `c*v0^2=a0*j`, so physical sensor
energy is `C_s*V_s^2/2` and (2) transforms with the same clock as every row.
These equalities specify a conditional component model; they do not establish
its fabrication or accuracy. Any subsequent voltmeter loading must also be
admitted rather than silently removed.

The physical mean is `J_lab=v0*J` over `t_window=T/b`. If the stipulated
structural error eta is to cover a uniform mean-voltage error e_V, a necessary
compatibility condition is

\[
 e_V\,t_{\rm window}\le\frac{a_0\gamma T\eta}{4}.
 \tag{18}
\]

Slowing b alone leaves this product unchanged. For the conventional supplied
Josephson conversion `a0=h/(4*pi*e)`, it is approximately
`3.7254e-31 V s` at the declared T and eta. The exact SI constants h and e
come from [NIST SP 330, Section 2](https://www.nist.gov/pml/special-publication-330/sp-330-section-2).
They are not derived from TNFR. Equation (18) is a requirement of this protocol,
not a noise floor, uncertainty principle or universal impossibility theorem.
Independent timing, input, transfer and parameter errors would also need
their own propagated allowances before using (9) as a physical prediction.

The [Keithley measurement handbook](https://download.tek.com/document/LowLevelHandbook_7Ed.pdf)
describes resistance, bias and capacitive loading (Section 2.2.1),
bandwidth and settling (2.6.4), finite integration (3.2.8), and distinct
accuracy/uncertainty/noise considerations (1.4.2-1.4.4). These support modeling
sensor state and averaging. They do not certify this exact one-pole law,
the full C5 preparation or the joint deterministic error bound.
The [source receipt](../../docs/assets/organized_finite_window_design/method-sources-v1.json)
records reviewed sections and limits. It claims web-text review, not archived
source bytes, measured arrays, admitted apparatus or an evaluated response.

<a id="organization-sensitive-next-question"></a>
### Reusable consequence and remaining question

The [detached exact bound](../../src/tnfr/physics/_organized_window_observation.py)
uses [shared loaded-pulse admission](../../src/tnfr/physics/_organized_loaded_pulse.py)
for the pulse and preparation primitives, sensor energy, forced growth and
identity bound.
It separately admits the observation error and short-window norm denominator,
then rebuilds the response bounds in (3)-(13). Its
[independent controls](../../tests/physics/test_organized_window_observation.py)
check matrix gain, sensor storage, convolution coefficients, arithmetic
domains, preparation obstruction and the physical scaling. This private
proof consumer installs no runtime law or SDK pathway. Earlier frozen
protocols and responses retain their original models and budgets.

The short-window mean supplies a scoped obstruction and reusable sensor
accounting. The subsequent [recovery-area result](ORGANIZED_RECOVERY_OBSERVATION.md#recovery-area-result)
conditionally distinguishes restoring phase feedback from the same calibrated
passive linear response under a separately declared observation and error
contract. It does not identify a unique restoring law or supply independent
physical preparation and measurement. Repeating the short-window mean with
tighter arithmetic does not supply those missing premises. The sole active
task and its acceptance conditions remain in the
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate).
