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
sensor terms and balances. The recovery calculation rebuilds the loaded
pulse from its primitive declaration. It does not pass H as the short-pulse
horizon: that would change the sensor time constant and exceed that bound's
domain. Sensor time remains `tau_s=T=2^-10` throughout recovery.

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
area conversion. A bounded constant readout offset e_V contributes up to
`e_V*H/b` to area error. Hence its allowance alone would require
`e_V*H/b<=a0*gamma*eta_A/4`. Unknown offset, drift, clock, delivery and matrix
errors cannot be discarded because the mathematical trajectory decays.
The integral allowance is a new observation premise with different units
from the short-window mean-error allowance. Neither implies the other.

Independent phase preparation is still required; the earlier equal-energy
counterexample remains valid. No apparatus, full calibration, drift coverage
or physical data has been admitted. The theorem supplies a finite conditional
mechanism test and a reusable loaded recovery bound, not an empirical result.

### Implementation and verification boundary

The [detached recovery consumer](../../src/tnfr/physics/_organized_recovery_observation.py)
rebuilds the pulse bounds, admits recovery duration and integral error, and
calculates (8)-(14) with normalized exact arithmetic. Its fixed law and pulse
are explicit; changing them requires a new proof. A shorter duration or larger
error can leave separation unavailable without proving a physical collision.
The [independent controls](../../tests/physics/test_organized_recovery_observation.py)
check the complete modified-energy derivative, sensor endpoint factors,
nonlinear collar constants, alternative restoring law, arithmetic boundaries
and laboratory-unit cancellation. Frozen producers and responses are not run.

The [sole queue](FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) retains operational
preparation and observation admission as the outstanding part of the research
question. The conditional comparison is complete; more trajectory precision
would not supply those missing physical premises.
