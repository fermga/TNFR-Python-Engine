# Integrated sensor response during prepared-pattern recovery

<a id="recovery-area-declaration"></a>
## Declared finite recovery question

The [loaded-window result](ORGANIZED_FINITE_WINDOW_DESIGN.md) resolves matrix
response but proves overlap with a matched phase-blind reference during its
short pulse. This candidate keeps that complete loaded law, pulse and sensor,
and changes the observation to a signed integral covering recovery. The
[execution plan](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns admission.
The [form/phase memory identities](../nodal/SINE_FORM_PHASE_REDUCTION.md#sine-form-phase-memory-equivalence)
motivate retaining the endpoint information exactly; their decay coefficients
are not transferred to the loaded model.

Let L be the unit C5 Laplacian, `v=(1,-1,0,0,0)`, `W_r=I+r*L^2`,
`gamma=1/(1023*pi)`, and keep the declared normalized clock tau, unit held
capacities, circular phases and continuous centered perturbation lift theta
about winding `w=+1` or `w=-1`. The complete law is

\[
 W_r U'=(-LU+\gamma S(\Phi))/4+f-kv(y-z),\qquad
 \Phi'=\gamma LU/4,\qquad \tau_s z'=y-z,\qquad y=v^TLU.
\]

The alternative omits the sine-current branch, consumes only U and z, and
retains the same r, U(0), z(0), input, clock and sensor. It is the passive
linear reference of the preceding owner, not a different response matrix.

| Primitive | Fixed analytical declaration |
| --- | --- |
| Response family | Fixed `r in [1/100,1/10]` in both models |
| Preparation | `sum(U(0))=sum(theta(0))=0`, `||U(0)||_2+||theta(0)||_2<=delta=2^-26`, `|z(0)|<=delta_s=2^-26`; same lifted winding chart as the preceding owner |
| Input | `f=a*v` for `0<=tau<=T`, then zero; `a=1/16`, `T=2^-10` |
| Sensor | `k=1/64`, fixed `tau_s=T` throughout pulse and recovery |
| Observation | Raw signed area `A_H=integral_0^H z(tau) d tau`; no endpoint correction or phase reading is added to the observable |
| Horizon | `H=T+D`, with fixed recovery duration `D=2^40` in the declared structural clock |
| Error | One independently bounded additive area error `|e_A|<=eta_A=2^-18`, separately for each candidate; this is an integral error, not an instantaneous reading allowance |
| Identity | Full-law phase gaps remain within `rho=1/10` of their prepared twist during pulse and recovery |
| Finite criterion | Whole-preparation permitted area intervals for the full and phase-blind models are disjoint with gap `>2^-14`, uniformly over the declared r family |
| Calculation | Exact inequalities and rational decay bounds from the complete nonlinear loaded law; no trajectory, response fit, parameter scan or physical dataset |

This analytical declaration supplies no laboratory realization of the very
long structural horizon or the area-error bound. Physical preparation,
input/clock/matrix calibration, sensor drift/loading and joint coverage
remain independent obligations. The exact electrical dictionary must be
used to translate the integral and horizon together.

A successful comparison would distinguish these two declared complete laws.
It would not uniquely identify winding, a sine constitutive law, material
constituents or TNFR among all models with restoring state. In particular,
the endpoint identity must be checked for alternative restoring potentials,
not treated as a unique signature of the prepared winding pattern.

Earlier declarations, source receipts, responses and producers remain frozen.
No law is installed in the runtime and no fine-node acquisition is inherited.

<a id="recovery-area-result"></a>
## Finite nonlinear result

Declaration revision `46df23f1` fixes the candidate above. The area intervals
are disjoint by more than `2^-14` under those premises. The proof combines
exact integrated evolution rows with a new decay estimate for the actual
loaded nonlinear law. It evaluates no trajectory or physical response.
The result distinguishes restoring phase feedback from the declared passive
phase-blind reference. A separate countermodel below prevents interpreting
that distinction as unique identification of sine dynamics or winding.

### Pulse admission and shared preparation

Retain the [loaded storage and pulse bounds](ORGANIZED_FINITE_WINDOW_DESIGN.md#loaded-window-design-result):

\[
 E_s=\tfrac12U^TLW_rU+V(\Phi)-V_*+\tfrac{k\tau_s}{2}z^2,
 \qquad E_s'=-\tfrac14\|LU\|^2-k(y-z)^2+(LU)^Tf.
 \tag{1}
\]

The unchanged pulse gives `E_s(T)<=Ebar<2^-16`, and its first-exit
argument preserves `max|Dtheta|<1/10` throughout the pulse and all later
unforced time. Here Ebar includes initial sensor energy and the allowance
`2*a^2*T`; it is not delivered work. The full preparation cost still includes
V_*. There is no event or state reset at pulse release.
After release the supplied work is zero and the sum of the two continuous
losses over any recovery interval is at most Ebar, by (1) and nonnegative
excess storage within the retained collar.

The common preparation calculation
`(4+32*r_max)*delta^2`, collar radius and barrier now have one
[shared owner](../../src/tnfr/physics/_organized_c5_preparation.py).
The unloaded and loaded consumers retain distinct observations, gains,
sensor terms and balances. A separate
[loaded-pulse owner](../../src/tnfr/physics/_organized_loaded_pulse.py)
re-admits the seven pulse primitives and rebuilds sensor energy, forced
growth and identity. Both observation consumers use this energy calculation;
recovery requires no short-window gain or mean-response interval. The energy
bound itself does not require the short-window norm denominator to be positive.
Passing H as the pulse horizon would change the supplied pulse and sensor law:
sensor time remains `tau_s=T=2^-10` throughout recovery.

### A decay estimate that retains the sensor

Put `q=LU`, `d=y-z`, `sigma=k*tau_s=2^-16`, and
`N=||U||^2+||theta||^2+sigma*z^2`. On the admitted centered collar,
`I<=L<=4I` on the neutral space, `I<=W_r<=(13/5)I`, and

\[
 \tfrac1{10}\|\theta\|^2\le V-V_*\le2\|\theta\|^2,
 \qquad \theta^T\nabla V\ge\tfrac15\|\theta\|^2.
 \tag{2}
\]

These use the full phase potential's Hessian throughout the collar, not just
the tangent at the twist. N and the modified energy below use the fixed
normalized chart; they are not additional physical observables or invariant
metrics. Define

\[
 \mathcal F=E_s+
 \gamma\theta^T(W_rU+\sigma vz)+\tfrac12\|\theta\|^2.
 \tag{3}
\]

The sensor-corrected quantity obeys
`(W_rU+sigma*v*z)'=-q/4-gamma*grad(V)/4` after release.
The derivative of the last term in (3) cancels the mixed `theta^T*q` term.
Thus, exactly along the full nonlinear recovery,

\[
 \mathcal F'=-\tfrac14\|q\|^2-kd^2+
 \tfrac{\gamma^2}{4}U^TLW_rU+
 \tfrac{\gamma^2\sigma}{4}yz-
 \tfrac{\gamma^2}{4}\theta^T\nabla V.
 \tag{4}
\]

This cancellation uses the actual matrix input and sensor state. A cross term
inserting `L^+` before the nonlinear potential gradient would need a further
sign proof; its matrices need not commute.

For completeness, `E_s+||theta||^2/2` lies between `N/2` and `26*N/5`.
Since `||v||=sqrt(2)<3/2`, the absolute mixed term in (3) is at most
`(23/10)*gamma*N<N/4`. Consequently

\[
 N/4\le\mathcal F\le8N.
 \tag{5}
\]

Also `U^TLW_rU<=(13/5)||q||^2` and
`yz=y^2-yd<=3||q||^2+d^2/2`. Applying `gamma<1/3000` and the fixed
sensor coefficients to (4) gives

\[
 \mathcal F'\le-\|q\|^2/8-kd^2/2-\gamma^2\|\theta\|^2/20.
 \tag{6}
\]

Because `z=y-d`, one has
`N<=2||q||^2+||theta||^2+2*sigma*d^2`. Equations (5)-(6) therefore imply

\[
 \mathcal F'\le-\gamma^2N/20
 \le-\gamma^2\mathcal F/160
 \le-2^{-32}\mathcal F.
 \tag{7}
\]

The last inequality uses `gamma>1/4096`, following from `pi<4`.
These coefficients belong to this loaded C5 model and its fixed collar and
sensor. Earlier reciprocal-potential energy methods motivate (3); their
different-law coefficients are not reused as certificates.

At pulse release, `N(T)<=10*Ebar`, so `F(T)<=80*Ebar`. At any later
duration D, let `n=floor(D/2^32)`. Since `exp(-n)<=2^-n`,

\[
 N(T+D)\le320\overline E\,2^{-n}<2^{-7-n}.
 \tag{8}
\]

The phase-blind reference has its own storage
`E_L=Uhat^T*L*W_r*Uhat/2+sigma*zhat^2/2` and loss
`E_L'=-||qhat||^2/4-k*dhat^2`. With
`N_L=||Uhat||^2+sigma*zhat^2`, the bounds
`N_L/2<=E_L<=8*N_L` and `N_L<=2||qhat||^2+2*sigma*dhat^2` give
`E_L'<=-E_L/64`. The same forced growth allowance bounds `E_L(T)<=Ebar`.
It therefore satisfies the weaker common endpoint envelope (8) as well.
No phase coordinate is invented for this reference.

An entirely rational norm bound follows without an exponential evaluator or
a trajectory grid:

\[
 \chi=2^{-\lfloor(n+7)/2\rfloor},\qquad
 \|U_H\|,\ \|\theta_H\|,\ \|\widehat U_H\|\le\chi,
 \qquad |z_H|,|\widehat z_H|\le256\chi.
 \tag{9}
\]

For the declared D, `n=256` and `chi=2^-131`. This is a conservative finite
deadline, not an optimized settling time or a proposed laboratory duration.
The rational tail avoids losing small bounds to a fixed absolute interval
grid. The detached consumer limits n to 4096 as an arithmetic-work policy;
that limit is not a physical restriction or a failed convergence claim.

### Exact finite area and separation

Integrating the live phase and sensor rows gives, at every finite H,

\[
 A_H=\frac4\gamma v^T(\theta_H-\theta_0)
       -\tau_s(z_H-z_0).
 \tag{10}
\]

This is an identity for the raw sensor integral. It neither adds a phase
measurement nor deletes sensor endpoint memory from the proof. Integrating
the actual phase-blind form row, with the same pulse, gives separately

\[
 \widehat A_H=8aT+4v^TW_r(U_0-\widehat U_H)
             -(1+8k)\tau_s(\widehat z_H-z_0).
 \tag{11}
\]

In particular, the factor `1+8k` retains the sensor's back-action as well as
its observation memory. Omitting either changes this identity.
Use the common endpoint envelope (9) and define

\[
 \begin{aligned}
 b_F&=6(4096)\delta+\tau_s\delta_s
                  +[6(4096)+256\tau_s]\chi,\\
 b_L&=6(13/5)\delta+(1+8k)\tau_s\delta_s
                  +[6(13/5)+(1+8k)256\tau_s]\chi.
 \end{aligned}
 \tag{12}
\]

The permitted observed record sets are enclosed by

\[
 \mathcal A_F=[-b_F-\eta_A,b_F+\eta_A],\qquad
 \mathcal A_L=[8aT-b_L-\eta_A,8aT+b_L+\eta_A].
 \tag{13}
\]

For the frozen candidate, exact rational arithmetic yields

\[
 \inf\mathcal A_L-\sup\mathcal A_F
 =8aT-b_F-b_L-2\eta_A>2^{-14}.
 \tag{14}
\]

One coarse check is that the input area minus the full-law initial-phase
allowance is `2^-13`, the remaining initial allowances sum to less than
`2^-22`, the two observation errors sum to `2^-17`, and the two terminal
allowances sum to less than `2^-115`. These bounds imply (14) without
sampling a trajectory or fitting a gain. The result holds over the entire
declared preparation and r interval, not only at exact twist initialization.

At exact zero perturbation and sensor preparation, (10) tends to zero as
recovery completes, while (11) tends to `8aT=1/2048`. The positive early
sensor response of the earlier theorem is therefore followed by a negative
signed contribution in the full model. This compensation is a consequence
of return to the same phase lift and the complete sensor law. A small
late value of z by itself would not certify that accumulated area.
Signed voltage area and dissipated energy have different units and balances;
the late sign reversal is consistent with both nonnegative losses in (1).

### What the area identifies, and what it leaves unresolved

A distinct restoring law shares the decisive endpoint property. In the
declared local lift, replace `V-V_*` by
`Q(theta)=cos(2*pi/5)*theta^T*L*theta/2` and its sine source by
`-cos(2*pi/5)*L*theta`. Keep both evolution rows, W_r and the sensor.
This local quadratic law matches the twist and its full tangent but differs
from the nonlinear sine field away from that state. It consumes phase
information and is not the phase-blind reference of (11). The law is declared
on the retained local lift; it is not a globally periodic potential.

Its own potential satisfies the same collar Hessian bounds, so its own
storage, trapping and calculation (3)-(9) apply. For matched initial states,
the initial terms of (10) agree exactly between the sine and quadratic laws.
Their finite areas differ by at most twice the full terminal allowance in
(12), less than `2*eta_A` at the declared horizon. Their actual permitted
records overlap by the midpoint-error construction. Thus this area detects
restoring feedback relative to the calibrated phase-blind alternative, but
does not uniquely select the sine law or its nonlinearity. The two winding
signs also retain the same bounds. Further physical identification must
state which property and which alternative it intends to distinguish.

### Physical scaling and remaining admission

Under the existing [electrical dictionary](ORGANIZED_FINITE_WINDOW_DESIGN.md#loaded-window-design-result),
`tau=b*t_lab`, `V_s=v0*z` and `v0=a0*b*gamma/4`. The laboratory voltage area
and its corresponding error requirement are

\[
 \mathcal A_{\rm lab}=\int_0^{H/b}V_s(t)\,dt
       =\frac{a_0\gamma}{4}A_H,\qquad
 \varepsilon_{\rm area}\le\frac{a_0\gamma}{4}\eta_A.
 \tag{15}
\]

Changing b rescales duration and voltage together; it does not change this
area conversion. The integral allowance is a new observation premise with
different units from the short-window mean-error allowance. Neither implies
the other.

<a id="recovery-operational-admission"></a>
### Operational admission of the frozen candidate

**Physical status: not admitted. Experimental evaluation: not tested.**
The review below concerns direct voltage-area measurement with the supplied
Josephson conversion and voltage-only initialization checks. Its published
instrument error allowance does not certify the frozen integral error, while
those initialization checks cannot establish the full phase preparation.
These are separate scoped obstructions. They preserve the conditional
theorem and do not establish a universal physical impossibility or an
experimental rejection of either law.

#### Clock, duration and joint residual error

Let p denote the laboratory pulse duration and h the total observation
duration, avoiding confusion with Planck's constant h_P. The unchanged
candidate and sensor imply

\[
 p=T/b,\qquad h=H/b=Rp,\qquad
 R=H/T=1+2^{50},\qquad (\tau_s)_{\rm lab}=p.
 \tag{16}
\]

If an independently admitted input-delivery mode requires `p>=p_min>0`,
and preparation, calibration and measurement coverage hold only through
`h_max`, any compatible b must satisfy

\[
 H/h_{\max}\le b\le T/p_{\min},\qquad Rp_{\min}\le h_{\max}.
 \tag{17}
\]

This constrains delivery of the actual input, not sampling of its short
pulse. A continuous integrator need not resolve that pulse in separate
samples. Conversely, a finite rise time requires a justified input-error
bound; a nominal pulse-width setting does not realize the ideal rectangular
current exactly.

For the supplied conversion `a0=h_P/(4*pi*e)`, the
[SI defining constants](https://www.nist.gov/pml/special-publication-330/sp-330-section-2)
give `h_P=6.62607015e-34 J s` and `e=1.602176634e-19 C` exactly. Then

\[
 C_A=\frac{a_0\gamma}{4}
     =\frac{h_P}{16(1023)\pi^2e},\qquad
 B_A=C_A\eta_A=\frac{h_P}{1023\,2^{22}\pi^2e}
     \simeq9.7658568\,10^{-26}\ {\rm V\,s}.
 \tag{18}
\]

The nominal phase-blind area is `C_A/2048`, approximately
`1.2500297e-23 V s`; the proved structural gap corresponds to more than
`1.5625371e-24 V s`. These are unit translations, not measured responses.
The physical constants and Josephson reference remain supplied premises.

Suppose an independent residual-error model after calibration has
`|offset(0)|<=e0` and `|offset'(t)|<=d0`. Over [0,h], its sharp worst-case
absolute integrated error is `e0*h+d0*h^2/2`, attained by aligned affine
offset and drift. Allocating the entire integral allowance to these errors
would require

\[
 e_0h+\tfrac12d_0h^2\le B_A,
 \qquad e_0Rp_{\min}+\tfrac12d_0(Rp_{\min})^2\le B_A.
 \tag{19}
\]

For constant offset alone, `e0*p_min<=B_A/R`, approximately
`8.6738233e-41 V s`. As a scale illustration only, a 1 ns minimum delivered
pulse entails an observation longer than `1.1258999e6 s` and requires a
residual constant-offset bound below `8.6738233e-32 V` even if it consumes
the whole allowance. Neither value is a hardware specification or a
fundamental noise floor. This calculation assumes the stated error class;
it does not assert that all noise accumulates coherently.

Independent calibration can subtract a known offset, but its residual and
drift still need joint coverage. A final small voltage cannot certify a
small accumulated error. A gain error, gate-timing error, sensor loading,
input tracking or uncertain matrix changes further terms and cannot be
assigned the same full allowance simultaneously. Finite-time calibration
uncertainty is not an admitted perturbation of the exact dynamical law.
Changing the horizon, using paired pulses or resetting/subtracting part of
the response requires a new observation or input declaration and proof.

#### Independent preparation information

The voltage map `V=v0*L*U` and sensor voltage `V_s=v0*z` contain no initial
relative-phase information. This fails even locally: at U=z=0, both
`theta=0` and `theta=2*delta*(e_0-e_1)` have identical voltage readings and
winding and lie in the retained collar, while only the first satisfies
`||U||+||theta||<=delta`. The
[equal-energy counterexample](ORGANIZED_FINITE_WINDOW_DESIGN.md#loaded-window-design-result)
separately prevents zero voltage, winding and target storage from globally
certifying the collar. Allowing the reference law to omit phase does not
remove the full law's preparation obligation.

A phase-sensitive measurement or independently validated preparation process
must therefore bound the missing relative coordinates without using the
reserved recovery area. Any reading's loading, disturbance, uncertainty and
transport to pulse start belong to that certificate. A winding label does
not replace the four relative phases, and an acquired 45-node source does
not transfer to this supplied C5 matrix law without its own reduction and
state-error proof. Common phase origin can be quotiented; response-relevant
relative phase cannot.

For example, admit the neutral form leaf, a calibrated `v0>=v_min>0`,
projected nodal voltage estimate Vhat, sensor estimate Vshat and an
independently admitted centered perturbation lift thetahat. Let their joint
errors at pulse start be bounded in the respective Euclidean/absolute norms
by eps_V, eps_s and eps_theta. Since `||L^+||_2<1` on C5, sufficient bounds are

\[
 B_U=\frac{\|P\widehat V\|+\varepsilon_V}{v_{\min}},\qquad
 B_\theta=\|\widehat\theta\|+\varepsilon_\theta,\qquad
 B_s=\frac{|\widehat V_s|+\varepsilon_s}{v_{\min}},
 \quad B_U+B_\theta\le\delta,\quad B_s\le\delta_s.
 \tag{20}
\]

This is a sufficient information contract, not a constructed phase sensor.
Alternatively, independently admitted oriented phase gaps around the chosen
twist give `d=Dtheta`; after checking their circular branches and cycle
consistency, `L>=I` on the neutral space yields `||theta||<=||d||`.
A bound on their estimate and joint error can therefore supply B_theta.
The phase origin need not be measured, but neither a winding integer nor a
single gap replaces the full relative-state bound. Separate marginal
confidence statements are not the joint coverage required by (20).

#### Reviewed direct-area method and scope of the decision

The [Lake Shore 480 manual](https://www.lakeshore.com/docs/default-source/product-downloads/manuals/480_manual.pdf?sfvrsn=d25b3964_1),
Sections 2.1.1 and 4.6.1, describes continuous voltage integration in DC
non-peak mode. Section 8.9.6 calibrates it using an independently measured
terminal voltage and the specified burst duration. This is a relevant direct-area method.
Sections 5.9.1 and 5.9.3 distinguish pre-run drift estimation from DriftTrak,
which treats sufficiently small changes as drift. Unmodeled tracking can
therefore remove a slow contribution of the intended recovery signal.
Resetting, taking peaks or using AC RMS also changes the frozen statistic;
these modes do not automatically preserve its signed integral.

The [published DC specifications](https://www.lakeshore.com/products/categories/specification/magnetic-products/fluxmeter/model-480-fluxmeter)
give an offset allowance of `+/-10 microvolt seconds` plus integrator drift
and a gain allowance of `+/-0.25%` of reading below `10 V s/s`.
The quoted drift, `+/-1 microvolt second/minute`, applies on the 300 mV s
range with 100 kilohms input resistance at constant temperature. The offset
allowance alone is more than `10^20` times B_A. This published error envelope
cannot establish the frozen requirement; it is not a lower bound on actual
error or all possible instruments. No long-term drift extrapolation is used.
The ground-referenced, finite-resistance input also requires a loading and
grounding model when attached to the retained sensor.

The [Keithley measurement handbook](https://download.tek.com/document/LowLevelHandbook_7Ed.pdf),
Sections 3.2.3-3.2.4, 3.2.8 and 3.3.2, separately explains application zero,
drift, periodic-noise rejection and reversal/offset-compensated sequences.
These methods do not supply joint coverage of this uninterrupted signed area.
In particular, treating the predicted late signal as a measured zero would
use the reserved response to calibrate away a candidate difference.

The [source receipt](../../docs/assets/organized_recovery_observation/method-sources-v1.json)
records reviewed primary sections and their limits. This is web-text review,
not archived source bytes, measured arrays or an apparatus calibration.
The reviewed direct-area route supplies neither the required error coverage
nor independent complete preparation. The local preparation counterexample
also survives arbitrary improvement of voltage precision. Thus the scoped
operational review closes without admitting a physical evaluation.
Other independently justified observation/preparation methods remain possible;
they cannot inherit this candidate's admission or alter its frozen premises.

### Implementation and verification boundary

The [detached recovery consumer](../../src/tnfr/physics/_organized_recovery_observation.py)
rebuilds the shared pulse energy bounds, admits recovery duration and integral
error, and calculates (8)-(14) with normalized exact arithmetic. Its fixed law and pulse
are explicit; changing them requires a new proof. A shorter duration or larger
error can leave separation unavailable without proving a physical collision.
The [independent controls](../../tests/physics/test_organized_recovery_observation.py)
check the complete modified-energy derivative, sensor endpoint factors,
nonlinear collar constants, alternative restoring law, arithmetic boundaries
and laboratory-unit cancellation. Frozen producers and responses are not run.

The same controls check the clock-independent requirements, sharp residual
offset/drift bounds and preparation-information obstruction. They do not
evaluate an instrument or turn a published specification into a measured
error. Frozen producers and earlier source receipts remain unchanged.

The [sole queue](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) records this
scoped closure and selects the next foundational question. Physical
preparation and observation remain dependencies of a future physical test;
more trajectory precision would not supply those missing premises.
