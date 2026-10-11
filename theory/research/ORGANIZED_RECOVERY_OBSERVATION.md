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
