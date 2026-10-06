# Conditional phase locking and circulation geometry

P2 and network locking, capacity adaptation, circulation and cycle periods.

Part of [Forced support balance and model boundaries](../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 26. Conditional phase locking and form restoration on fixed P2

### Declared composition and complete state

Compose the existing [U3-gated phase proposal](../../src/tnfr/dynamics/phase_evolution.py)
with fresh [canonical multichannel pressure](../../src/tnfr/dynamics/dnfr.py)
on the fixed undirected unit edge `0--1`. This explicitly named composition
is separate from the optimizer/FFT routes, whose phase updates do not feed
back into their pure-EPI pressure. No new oscillator formula is introduced.

The continuous comparison below supplies constant positive capacities
`nu0,nu1`, fixed support, coupling `K>0`, and effective pressure coefficients
`e>0,w>=0,v>=0` for EPI, phase and capacity. The topology gradient is zero
on P2. The equations `nu_dot=0` and `support_dot=0` are premises. Capacity
as angular speed and the sine interaction are the existing configured model,
not deductions from `x_dot=nu*p`. With a continuous phase lift satisfying
`|theta1-theta0|<gamma`, where `0<gamma<=pi/2` is the effective U3 limit,
the complete varying state obeys

```text
theta0_dot=nu0+K*sin(theta1-theta0),
theta1_dot=nu1+K*sin(theta0-theta1),
p0=-e*(x0-x1)+w*(theta1-theta0)/pi+v*(nu1-nu0),  p1=-p0,
x0_dot=nu0*p0,                                      x1_dot=nu1*p1.
```

Pressure is evaluated from the current state, never evolved independently
or reconstructed to fit a desired form. These fixed equations define an
autonomous conditional model, without a time-dependent external signal or
operator dispatch. They do not explain the origin of their supplied
capacities, support or constitutive identification. Coupling is **phase to
EPI**, not bidirectional: EPI does not enter either phase equation.

### Locked target, conserved mean and local response

Define `sigma=nu0+nu1`, `d=nu1-nu0`, `delta=theta1-theta0`, `q=x0-x1` and
`b=(theta0+theta1)/2`. Direct addition and subtraction give

```text
delta_dot=d-2*K*sin(delta),
q_dot=sigma*(-e*q+w*delta/pi+v*d),
b_dot=sigma/2,
m=(nu1*x0+nu0*x1)/sigma,             m_dot=0.
```

The weighted mean is generally not the arithmetic EPI mean. Reconstruction
is `x0=m+nu0*q/sigma`, `x1=m-nu1*q/sigma`. There is exactly one strictly
admitted phase lock if and only if

```text
|d|<2*K*sin(gamma),
delta_star=asin(d/(2*K)),
q_star=(w*delta_star/pi+v*d)/e.
```

At the lock both phases advance at `sigma/2`, while EPI is stationary at
the reconstructed target with its initial m. In particular, maintained
contrast does not imply nonzero instantaneous EPI rate or growing pressure.
The phase and form contrast linearization is triangular, with eigenvalues
`-2*K*cos(delta_star)<0` and `-e*sigma<0`. This proves local attraction of
the relative state. The conserved m and common rotation leave neutral
directions; no single absolute full-state point attracts all preparations.

Equality in the gate condition places the lock on its excluded boundary.
At `|delta|=pi/2` the phase linearization also loses strict contraction.
For `K=0`, a nonzero d produces relative drift, while `d=0` leaves arbitrary
relative phase with no restoring response. Those are not admitted strict
locking cases. When capacities agree, the admitted target has
`delta_star=q_star=0`: common phase rotation alone supplies no contrast.

The fresh-pressure chain rule is a further independent consistency check:

```text
p0_dot=-e*sigma*p0+(w/pi)*(d-2*K*sin(delta)),  p1_dot=-p0_dot.
```

Thus zero pressure away from the phase lock need not remain zero. This
identity follows from the same constitutive map; it does not authorize an
independently prescribed pressure evolution.

### An invariant chart and explicit error bounds

Choose a proof interval `[-rho,rho]` with `0<rho<gamma`,
`|delta(0)|<=rho` and `|d|<=2*K*sin(rho)`. The phase vector field points
inward at its endpoints. The interval is therefore forward invariant, with
nonzero phasor resultants and strict U3 throughout the ideal evolution.
It is an admission region, not a new dynamical coefficient. Put

```text
mu=2*K*cos(rho)>0,       A=e*sigma>0,       C=sigma*w/pi,
D0=|delta(0)-delta_star|, E0=|q(0)-q_star|.
```

The sine secant on the interval is at least `cos(rho)`. Consequently, for
every `t>=0`,

```text
|delta(t)-delta_star| <= D0*exp(-mu*t),
|q(t)-q_star| <= E0*exp(-A*t)+C*D0*I(A,mu,t),
I(A,mu,t)=(exp(-mu*t)-exp(-A*t))/(A-mu)       if A!=mu,
I(A,A,t)=t*exp(-A*t).
```

The nonnegative I is the convolution of the two decaying exponentials;
the second bound follows by variation of constants. Pressure satisfies
`|p0|=|p1|<=e*|q-q_star|+(w/pi)*|delta-delta_star|`. Errors of the individual
EPI coordinates relative to the same-m target are respectively `nu0/sigma`
and `nu1/sigma` times the contrast error. A perturbation changing m changes
the final mean rather than being restored to the old absolute target.

There is also a bounded form interval. Write `Q(delta)=(w*delta/pi+v*d)/e`
and set `q_lo=min(q(0),Q(-rho))`, `q_hi=max(q(0),Q(rho))`. The q vector
field points inward at these endpoints. Its reconstruction gives bounds
for both node forms, which can verify that configured clipping remains
inactive. A clipped trajectory is not automatically this smooth model.

For the **ideal simultaneous Euler map** with a fixed step h, the sufficient
conditions `h>0`, `2*h*K<=1` and `h*A<=1` preserve both intervals. Indeed,
the phase map is nondecreasing there and maps its endpoints inward, while
the form map is a convex combination of q and Q(delta). With
`alpha=1-h*mu` and `beta=1-h*A`, both in `[0,1)`, the discrete errors obey

```text
D_n <= alpha^n*D0,
E_n <= beta^n*E0+h*C*D0*sum(beta^(n-1-j)*alpha^j, j=0,...,n-1).
```

The sum is empty at n=0. These are conservative sufficient conditions,
not a claim that every larger step is unstable. They refer to exact Euler
arithmetic, separately from the continuous theorem and represented execution.
Signed numerical defects must be accounted for and chart membership checked
before any corresponding finite binary64 claim; no infinite runtime
convergence follows merely from these formulas.

### Inactive-channel control and implementation boundary

The prospective `w=0` control must hold **effective** e and v fixed. Then
phase evolution is unchanged, but `q_star=v*d/e` and the phase-to-form term
vanishes. With v also zero the target contrast vanishes; with v nonzero the
old capacity-supported contrast remains. Conversely, v=0 with w>0 and d!=0
isolates the contribution supplied by the phase lock.

The pressure owner normalizes configured weights. Simply deleting the raw
phase weight would generally change e and v and invalidate that comparison.
On this P2 only, moving the removed phase weight to the inactive topology
channel preserves the other effective coefficients; the captured normalized
values must still be checked. This uses a zero existing channel, not an
additional force or an altered nodal equation.

The explicit [P2 composition owner](../../src/tnfr/physics/p2_phase_form.py)
exports `derive_p2_phase_form_model` and `propose_p2_phase_form_step` through
`tnfr.physics`. The first returns a frozen `P2PhaseFormModel`: represented
capacities and normalized coefficients, an exact rational locking ratio and
classification for the ideal real half-pi gate, binary64 target estimates,
and the sufficient monotone step ceiling. The displayed phase decay rate is
the least binary64 upper enclosure of the exact rational radicand's square
root `sqrt((2K-d)*(2K+d))`; it is not a lower contraction certificate. This
calculation avoids intermediate overflow and underflow near the lock boundary.
Its default K reuses the existing
operational coupling setting, not a newly derived structural constant.
The theorem above allows general `gamma<=pi/2`; this implementation uses
the half-pi case and separately checks the represented strict gate.

`propose_p2_phase_form_step` creates its own detached fixed P2 and returns
a frozen `P2PhaseFormStep`. It admits canonical phases in `[0,2*pi)`, checks
the strict represented half-pi gap before and after, and requires initial
and proposed EPI in `[-1,1]` with clipping inactive. The shared phase kernel
and `DefaultIntegrator` use one initial state and one unsplit Euler segment;
pressure is freshly evaluated before consumption and again at the endpoint.
`capture_non_epi_forcing` and `SupportTransportEuler` retain the source and
Euler residuals, alongside measured weighted-mean drift. No caller graph or
complete runtime invocation is certified. A single proposal is reproducible
with the declared recipe:

```python
from tnfr.physics import derive_p2_phase_form_model, propose_p2_phase_form_step
weights = {"phase": 1, "epi": 1, "vf": 0, "topo": 0}
model = derive_p2_phase_form_model((0.95, 1.05), pressure_weights=weights)
step = propose_p2_phase_form_step(model, (0.0, 0.0), (0.25, 0.75), dt=0.25)
print(model.lock_status, model.locked_contrast_estimate)
print(tuple(map(float, step.after_epi)), step.after_phase_gap)
```

Target estimates are not transcendental interval certificates, and a finite
accepted proposal does not prove repeated binary64 stability. The
[portable controls](../../tests/physics/test_p2_phase_form.py) compare prospective
targets, perturbations and the inactive phase channel within this scope.

The result establishes conditional restoration of a differentiated form
under a specified source-state law. It does not derive the supporting
capacity distribution, full substrate evolution, autonomous NFR creation,
a physical clock or a particle interpretation. The
[single G3 plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
any subsequent admission or experiment; existing passive and independent-
pressure results are not rerun or promoted by this composition.

## 27. Capacity adaptation moves the conditional P2 target

The [existing capacity adaptation owner](../../src/tnfr/dynamics/adaptation.py)
moves an eligible node toward the immutable-snapshot mean of its neighbors.
Its eligibility gate consumes stored pressure, stored Si and a consecutive
evaluation counter. It is a configured controller, not a capacity law derived
from the nodal equation. The following one-event calculation tests whether
this controller preserves the supporting capacities of Section 26.

### Exact event calculation and its limits

Retain the positive P2 capacities and definitions of Section 26. Let
`a0,a1` be the eligibility indicators at one event, and let `0<=mu<=1` be
the configured adaptation fraction. In exact arithmetic, with clipping
inactive, the simultaneous capacity proposals are

```text
nu0_plus=nu0+mu*a0*d,       nu1_plus=nu1-mu*a1*d,
d_plus=(1-mu*(a0+a1))*d,
sigma_plus=sigma+mu*(a0-a1)*d.
```

Positive capacities remain positive because the active proposals are convex
combinations of the two positive inputs. Moreover, `|d_plus|<=|d|`, so a
strictly admitted fixed-capacity phase lock remains available after this
event, at the same K and phase gate. Availability of a new lock does not mean
that the previous locked state remains a lock.

When both nodes are eligible, sigma is unchanged and d is multiplied by
`1-2*mu`. Its magnitude strictly decreases for `0<mu<1` and `d!=0`.
The endpoint `mu=1` swaps capacities and reverses d; it is not a strict
contraction. If just one node is eligible, d is multiplied by `1-mu`,
while sigma increases or decreases by `mu*d`. With neither node eligible,
or with `mu=0`, the capacities do not change. These alternatives matter:
partial eligibility also changes the form decay rate `e*sigma_plus` and
the sufficient Euler step ceiling. They must be checked for the next step.

The post-event comparison target follows from the same constitutive law:

```text
delta_star_plus=asin(d_plus/(2*K)),
q_star_plus=(w*delta_star_plus/pi+v*d_plus)/e.
```

The old target remains the reference for the preservation test. Computing
the new target explains its displacement; it does not replace the old
reference or fit a trajectory. For `w>0`, `v>=0` and `d!=0`, reducing
`|d|` reduces the target's magnitude. In the phase-only source witness
`v=0`, eliminating the capacity contrast eliminates the predicted
phase-supported form contrast. Indefinitely repeated eligibility would
need additional evidence; the one-event identity does not establish it.

### Immediate pressure and subsequent form response

The event changes capacity, but leaves EPI and phase unchanged. At the same
form and phase, freshly evaluated pressure therefore changes by

```text
p0_plus-p0=v*(d_plus-d),       p1_plus-p1=-(p0_plus-p0).
```

In particular, with `v=0` an immediate pressure check misses the displaced
target: pressure is unchanged even though the relative phase derivative
changes by `d_plus-d`. Starting at the ideal old lock and subsequently
holding the new capacities fixed gives

```text
delta_dot(0_plus)=d_plus-d,
p0(0_plus)=0,
p0_dot(0_plus)=(w/pi)*(d_plus-d),
q_dot(0_plus)=0,
q_ddot(0_plus)=sigma_plus*(w/pi)*(d_plus-d).
```

This is the existing phase-to-pressure-to-form path. No independent pressure
evolution is introduced. An ideal simultaneous Euler step first changes the
phase gap by `h*(d_plus-d)` while leaving q unchanged; the second changes q
by `h^2*sigma_plus*(w/pi)*(d_plus-d)`. Actual binary64 residuals and chart
admission remain separate obligations.

The conserved mean of Section 26 is also specific to fixed capacities.
Writing its post-event value using unchanged x gives the exact jump

```text
m_plus-m=-mu*d*q*(a1*nu0+a0*nu1)/(sigma*sigma_plus).
```

Subsequent fixed-capacity segments conserve the new mean. They do not
restore the old absolute target simply because each segment has its own
weighted-mean conservation law. There is no EPI jump in this capacity event;
the mean changes because its weights change.

### Finite default-policy witness

The [P2 regression controls](../../tests/physics/test_p2_phase_form.py) prepare
the prospective Section 26 lock for capacities `(0.95,1.05)`, raw pressure
weights `phase=epi=1`, `vf=topo=0`, and the existing coupling `K=0.1`.
They compare an active capacity writer with an inactive-writer control,
preserving the old target in both branches. Fresh pressure and fresh Si
are computed through their shared owners before the gate is evaluated.

For the specified binary64 preparation and current shared phase-pressure
kernel, pressure is `(0,0)` and Si is approximately
`(0.8972318538363662,0.9694744245977734)`. Both nodes qualify under the existing defaults:
`EPS_DNFR_STABLE=0.001`, `si_hi=0.5`, `VF_ADAPT_TAU=5` and
`VF_ADAPT_MU=0.1`. Beginning with zero counters, the fifth qualifying
evaluation changes capacities to `(0.96,1.04)`. These are five gate
evaluations on a prepared state, not five elapsed units of physical time.

The old target estimates are `delta_star=0.5235987755982994` and
`q_star=0.16666666666666682`. Reapplying the unchanged model to the observed
new capacities gives `delta_star_plus=0.4115168460674884` and
`q_star_plus=0.13098988043445475`. The new target is displaced, although
strict locking remains available. With only node 0 or only node 1 eligible,
the respective ideal proposals are `(0.96,1.05)` or `(0.95,1.04)`; both
give a phase target near `0.46676533904729683` and a form contrast near
`0.1485760219466835`, with different total capacities.
The unchanged high-threshold regression declares `si_hi=0.875` before
execution. Both nodes now qualify. The earlier pressure arithmetic produced
`(-2.7755575615628914e-17,2.7755575615628914e-17)` and Si approximately
`(0.8389323,0.91117487)`, admitting only the higher-capacity node. The current
shared principal-phase difference and division by represented pi instead
cancel the prepared form gradient exactly in this witness. We retain the
threshold and record the changed outcome rather than retuning it to restore
partial eligibility. The single-node proposals above remain conditional
blend identities, not the outcome of this current runtime control. This is a
separate configured policy control, not another canonical threshold.

After the writer window is closed, two existing P2 proposals with `h=0.25`
expose the delayed form response. In the active branch the first phase-gap
change is approximately `-0.005`; the second contrast change is
approximately `-0.00039788735772973427`. The inactive-writer branch retains
the old contrast within the existing `1e-14` regression criterion in this
finite comparison. These decimal
values are represented observations or target estimates, not exact
transcendental certificates or infinite-time runtime bounds.

The adaptation owner itself refreshes neither pressure nor Si. Si normalizes
pressure by the current maximum magnitude, so an arbitrarily small nonzero
antisymmetric pressure can lose the full pressure contribution to Si compared
with exact zero. The audit retains the actual fresh values instead of
silently substituting their ideal equilibrium limits. Counters count
qualifying calls, reset on a failed gate and are not reset by an adaptation
event; continued eligibility can therefore cause a write on the next call.

This detached, finite policy audit rejects preservation of this nonzero
fixed-capacity target by the existing default writer. It does not establish
instability of the new lock, a complete default runtime trajectory, permanent
eligibility, or an emergent law for Si, capacity or substrate formation.

## 28. Geometry-limited regional phase locking on the unit barbell

### Declared model and the bridge constraint

Use triangles `(0,1,2)` and `(3,4,5)`, joined only by edge `(2,3)`, with
unit symmetric conductance and degrees `d=(2,2,3,3,2,2)`. Capacities are
constant within each region: `nu_L=kappa-epsilon`, `nu_R=kappa+epsilon`,
with `kappa>|epsilon|`. These positive supplied capacities are also the
free angular rates in the existing averaged-sine phase law. Set the
effective pressure coefficients to `e>0,w>0` for form and phase, with
capacity and topology coefficients zero. The topology channel is disabled,
not identically zero on this irregular graph. The ideal equations are

```text
theta_dot_i = nu_i + (K/d_i) sum_(j~i) sin(theta_j-theta_i),  K>0,
x_dot = diag(nu) [-e L_rw x + w g(theta)],
g_i(theta) = Arg(sum_(j~i) exp(i(theta_j-theta_i))) / pi.
```

All edges must remain U3-admitted. This composition excludes Gamma,
events, clipping, capacity adaptation and other controllers. The scalar
form chart is signed real; zero capacity is outside this result. These
assumptions specify an additional phase/capacity model, not a consequence
of the nodal product alone.

Multiplication of the phase row by `d_i` cancels the sine interaction on
each undirected edge. Any common locked angular rate must therefore be
`omega=sum(d_i nu_i)/14=kappa`. Summing only the left region gives the
necessary cut balance `K sin(theta_3-theta_2)=7 epsilon`: its volume is
seven, and its sole outward edge carries all of its phase imbalance.
Consequently no lock under this model exists when `|7 epsilon|>K`.

For `|7 epsilon|<K`, let the principal angles and lock offsets be

```text
alpha = asin(2 epsilon/K),       beta = asin(7 epsilon/K),
psi = (-alpha-beta/2, -alpha-beta/2, -beta/2,
        beta/2,        alpha+beta/2, alpha+beta/2).
```

Substitution in all six rows gives `theta(t)=psi+kappa*t*1`. The interior
and bridge loads are different: `sin(alpha)=2 epsilon/K` and
`sin(beta)=7 epsilon/K`. A two-region average erases that distinction.
With the half-pi U3 gate the strict threshold is `|epsilon|<K/7`;
a narrower configured gate additionally requires `|beta|` to lie strictly
inside that gate. At equality `|7 epsilon|=K`, the bridge cosine vanishes
and strict relative restoring stability is lost. The overloaded result
uses the cut equation and does not depend on imposing equal interior phases.

### Full six-node local stability

The phase Jacobian at the lock is `-K D^(-1) B_cos`, where `B_cos` is the
weighted graph Laplacian with edge weights `cos(psi_j-psi_i)`. On the
strict branch every weight is positive. Similarity by `D^(1/2)` gives
one neutral common-rotation mode and five strictly negative modes. This
includes perturbations that split either interior pair; it is not merely
stability within the four-fiber reduction.

A finite neighborhood follows without linearizing. For a fixed common chart
offset `c0`, write `y=theta-psi-(c0+kappa*t)*1` in a continuous lift.
For any edge its nonlinear
error coupling equals `a_ij(y)(y_j-y_i)`, where

```text
a_ij(y) = integral_0^1 cos(psi_j-psi_i+s*(y_j-y_i)) ds.
```

If `|beta|+osc(y(0))<pi/2` and is strictly inside the configured gate,
the coefficients are symmetric and positive. At a largest error the
derivative is nonpositive; at a smallest error it is nonnegative. Thus
the error range is invariant and the degree-weighted error mean is
conserved. With that mean set to zero,

```text
||y(t)||_D <= exp(-K*cos(|beta|+osc(y(0)))*lambda_2*t) ||y(0)||_D,
lambda_2 = (11-sqrt(73))/12.
```

The same range argument holds for ideal simultaneous Euler when `h*K<=1`.
For the norm bound it suffices to require `h*K<=1/2`; the symmetric error
map then has nonnegative eigenvalues and contracts by at most
`1-h*K*cos(|beta|+osc(y(0)))*lambda_2` on the centered subspace.

### The actual phasor source and its compatible form

The phase source is not the sine interaction or a global Laplacian
substitute. At the lock define

```text
gamma = atan2(3 epsilon/K, 2*cos(alpha)+cos(beta)),
g_star = (alpha/2, alpha/2, gamma, -gamma, -alpha/2, -alpha/2) / pi.
```

The two-neighbor rows are phasor midpoints; a bridge row instead has the
resultant `2 exp(-i alpha)+exp(i beta)`. Its imaginary part is
`3 epsilon/K` and its real part is positive on the strict branch.
Reflection makes `sum(d_i*g_star_i)=0` exactly, so the source is
compatible with a stationary form. With `s=w/(e*pi)`, an independent
solution of `L_rw*z=(w/e)*g_star` is

```text
F = 2*alpha+3*gamma,
z = s*(F/2+alpha, F/2+alpha, F/2, -F/2, -F/2-alpha, -F/2-alpha).
```

The complete stationary family is `x_star=z+c*1`. Its physically relevant
mean for this declared nodal model uses `H=diag(d_i/nu_i)`, not uniform
weights and not `D` alone. To specify initial weighted mean `m`, set

```text
m_H(z) = s*(epsilon/kappa)*((11/7)*alpha+(3/2)*gamma),
x_star = z + (m-m_H(z))*1.
```

The [target evaluator](../../src/tnfr/physics/regional_phase_lock.py)
`derive_regional_phase_lock` reuses `derive_forced_support_balance` for
the represented source and mean constraint. The analytic formula above
provides an independent reference. Exact rational cut loads, binary64
inverse-trigonometric estimates, the exactly odd represented source and
the actual production phasor source are distinct evidence. In particular,
enforcing reflection in a comparison source is not permission to replace
or silently correct the production pressure.

### Mean conservation is conditional on the phase source

In general the full form mean obeys

```text
d(m_H(x))/dt = w*sum(d_i*g_i(theta)) / sum(d_i/nu_i).
```

Canonical phasor directions do not generally cancel in that sum. Thus an
arbitrary phase perturbation can move the absolute form target even with
fixed capacities. The phase degree mean, whose sine terms do cancel,
does not supply a form conservation law.

There is a useful invariant restricted preparation. Let reversal exchange
`0<->5`, `1<->4`, `2<->3`, and require the rotating phase offsets to be
odd under that reversal. The supplied detuning and graph share this
symmetry, so both the ideal continuous phase law and its Euler map
preserve it. The fresh phase source remains odd, its degree sum vanishes,
and `m_H(x)` is conserved for any initial form. This symmetry permits a
fixed full form reference during recovery without repeatedly recentering
the observed trajectory. General preparations retain the measured drift.

### Prospective finite recovery protocol

The following preparation and criteria precede runtime observations:
`kappa=1`, `K=1/2`, `epsilon=1/32`, `e=w=1/2`, `h=1/8`, and 256
simultaneous phase/form Euler steps. Fix `m=0`, use `x_star` above, and
initialize

```text
theta(0) = psi + a common canonical chart offset
           + (2,-1,1,-1,1,-2)/128,
x(0) = x_star + (31,31,31,-33,-33,-33)/256.
```

The phase perturbation splits both interior pairs and preserves reflection.
It has degree mean zero and `Y0^2=||y(0)||_D^2=13/8192`. The form
perturbation is H-centered with `X0^2=||x(0)-x_star||_H^2=7/32`.
Neither input nor target is fitted to an observed endpoint.

Elementary bounds give `alpha<1/7`, `beta<1/2`, total phase width below
one radian, edge separation at most `17/32`, and `lambda_2>1/5`.
The following rational lower rates suffice:

```text
lambda_p = 1759/20480,       lambda_x = 31/320,
a = 1-h*lambda_p,            b = 1-h*lambda_x.
```

For the form rate, `e*x^T B*x >= e*nu_min*lambda_2*||x||_H^2`
on the H-centered subspace follows by comparing the minimizing constant
in the D and H variances. The ideal Euler form map has nonnegative
spectrum under `h*e*nu_max<=1/2`, satisfied here.

To bound the changing source, on a common lift of width below one the
phasor-direction derivative is `(R-I)/pi`, with nonnegative entries
`R_ij<=2/d_i` on support. Hence `||R||_D<=2` and the phase-source
Lipschitz bound is strictly below one, since `pi>3`. This yields

```text
Y_n <= a^n*Y0,
X_n <= b^n*X0
       + w*sqrt(nu_max)*Y0*(a^n-b^n)/(lambda_x-lambda_p).
```

Here `w*sqrt(nu_max)*Y0/X0=sqrt(429/229376)<1/23`. Exact rational
evaluation at `n=256` therefore verifies, before execution,

```text
a^256 < 1/10,
b^256 + (a^256-b^256)/(23*(lambda_x-lambda_p)) < 1/5.
```

The selected acceptance is a phase error below one tenth and a full fixed
mean form error below one fifth of their initial norms. These are scoped
ideal-Euler predictions, not a posteriori choices of a favorable time.
The [finite execution controls](../../tests/physics/test_regional_phase_lock.py)
use the existing phase proposal, fresh canonical pressure and shared nodal
integrator. They separately retain source and Euler defects, non-symmetric
mean drift, overload and configured-gate controls. Common rotation crosses
the canonical phase wrap during this experiment; relative circular errors
must be reconstructed against the fixed rotating reference, rather than
mistaking a wrap jump for physical phase recovery or instability.

No ideal estimate above certifies infinite binary64 stability or a complete
runtime containing other writers. This mechanism conditionally maintains
regional differentiation while its supplied capacity contrast and phase law
persist. It neither selects those inputs autonomously nor establishes a
physical NFR, particle or endogenous substrate.

### Finite outcome with the frozen target

The production control passes both reserved recovery criteria after 256 steps
(`t=32` in the supplied clock). The target estimates are
`alpha=0.1253278311680654`, `beta=0.4528165947449256` and

```text
x_star = (0.10786151777372711, 0.10786151777372711, 0.06796843009895889,
         -0.07382420301501220, -0.11371729068978044, -0.11371729068978044).
```

| Error norm | Initial | Final | Observed ratio | Precomputed ideal upper ratio |
| --- | --- | --- | --- | --- |
| Phase, D norm | 0.0398361 | 0.000883519 | 0.0221789 | 0.0630834 |
| Full form, H norm | 0.467707 | 0.0160276 | 0.0342685 | 0.118917 |

No target is recentered or fitted after execution. The reflected preparation's
largest observed weighted-mean drift is below `2.77e-16`; its signed Euler
defects are below `1.42e-17`. A separate unperturbed target control keeps form
unchanged at its represented coordinates and has final phase error below
`5.48e-15`. Numerical comparisons permit a separately declared `1e-11`
allowance; neither that allowance nor these measured residuals certifies
asymptotic binary64 accuracy.

The controls also retain a genuinely nonconserving mean response to an
asymmetric phase perturbation, failure to preserve the old form when phase
pressure is disabled, exact overload classification and tighter-gate rejection
of an otherwise strict ideal lock. The target is independently checked against
80-digit phasor and closed Poisson formulas. These results support conditional
regional maintenance and recovery, while the source of the supplied capacity
contrast and phase law remains open.

## 29. Graph-independent locking and phase-source constraints

### Exact model: phase support is not transport conductance

Let the phase support be a fixed connected, simple undirected graph with at
least two nodes, unique neighbor sets `N_i` and counts `d_i=|N_i|`. Capacities
`nu_i>0` and the shared coupling `K>0` are fixed. Every support edge is admitted
by the U3 gate. The supplied phase law is the same averaged-sine law as in
Sections 26 and 28:

```text
v_i = theta_dot_i = nu_i + (K/d_i) sum_(j in N_i) sin(theta_j-theta_i),
Z_i = sum_(j in N_i) exp(i*(theta_j-theta_i)),
g_i = Arg(Z_i)/pi, when Z_i != 0.
```

Here `g` is the ideal canonical neighbor-phasor pressure channel, not the sine
coupling itself. The formulas use circular edge displacements; they do not
require all phases to lie in one global semicircle. The identification of
capacity with free angular rate and the value of K remain constitutive inputs.

Transport may separately use symmetric nonnegative conductances `W_ij` with
strengths `s_i=sum_j W_ij` and `L_W=I-diag(s)^(-1)W`. Phase counts `d_i`
include every support neighbor, even one with zero transport conductance.
Replacing `d_i` by `s_i` in the phase law changes the model. Statements about
stationary form below require the positive-conductance transport graph to be
connected, so every `s_i>0`.

### Locked rate, cut load and the sign of the phase source

At a common-rate lock `theta_i(t)=psi_i+omega*t`, multiply each phase row by
`d_i`. Contributions from the two orientations of an edge cancel exactly.
Therefore

```text
omega = sum_i d_i*nu_i / sum_i d_i,
Im(Z_i) = d_i*(omega-nu_i)/K.
```

More generally, summing only over a vertex set S gives the necessary cut balance

```text
sum_(i in S) d_i*(omega-nu_i)
    = K*sum_(i in S, j outside S, i~j) sin(psi_j-psi_i).
```

For strictly acute edge gaps, its absolute value is less than
`K*|boundary(S)|` for every nonempty proper S. If the gap magnitudes are at
most `rho<pi/2`, the sharper upper bound is `K*|boundary(S)|*sin(rho)`.
The one-bridge load in Section 28 is one instance. Cut inequalities are
necessary, not sufficient: compatible phase differences on cycles and the
actual gate must still be satisfied. No enumeration of graph cuts is required
to use the identity for a specified region.

Now assume every circular edge gap has magnitude strictly below `pi/2`.
Every cosine is positive, so define

```text
r_i = Re(Z_i)/d_i > 0,
g_i = atan((omega-nu_i)/(K*r_i))/pi.
```

The resultant lies in the right half-plane, removing both the zero-resultant
and argument-branch ambiguities. Consequently

```text
sign(g_i) = sign(omega-nu_i),
g_i = 0  if and only if  nu_i = omega.
```

For an edge bound `rho<pi/2`, `cos(rho)<=r_i<=1`, hence

```text
atan(|omega-nu_i|/K)/pi
    <= |g_i|
    <= atan(|omega-nu_i|/(K*cos(rho)))/pi
    <= |omega-nu_i|/(pi*K*cos(rho)).
```

The nodal capacity imbalance fixes the source's sign and constrains its
magnitude. It does not reconstruct the source without the retained resultant
geometry `r_i`, nor prove that a lock exists for the proposed capacities.

### What equal capacities obstruct, and what they do not

With common capacity `nu_i=kappa>0`, the locked rate is `omega=kappa` and
the strictly acute, fully admitted lock has `g=0`. For the unforced form row

```text
x_dot = diag(nu) [-e*L_W*x + w*g],   e>0, w>=0,
```

all stationary forms therefore satisfy `L_W*x=0`. Connected nonnegative
reciprocal transport makes x constant. This conclusion requires the capacity
and topology pressure contributions to be absent or zero, Gamma to be zero,
and no other event, history, clipping or controller term to alter the row.
Equal capacity itself makes the capacity-gradient channel zero; it does not
make topology pressure vanish on an irregular graph.

This is an obstruction to differentiated stationary **scalar EPI through this
phase-source mechanism**. It is not a prohibition of nonuniform phase, a
nontrivial triad geometry, moving patterns, or every possible NFR identity.
In particular, on a unit cycle C_n with `n>=5`, set

```text
psi_j = 2*pi*j/n  modulo 2*pi.
```

Each node has two relative neighbor angles `+2*pi/n` and `-2*pi/n`.
Their sine sum is zero and `Z_i=2*cos(2*pi/n)>0`. Common capacity therefore
gives an exact rotating phase lock with `g=0`, despite winding once around the
circle and lacking a narrow global phase chart. The local absolute phase
gradient is `2*pi/n`, not zero; for C6 it is `pi/3`. The pattern retains
nonuniform phase while its phase channel supplies no differentiated stationary
form. This identity alone asserts neither perturbation stability nor
autonomous formation of that winding state.
The existing [winding owner](../COUPLING_WINDING_PERSISTENCE.md#5-nonzero-winding-can-coexist-with-zero-canonical-pressure)
already records this pressure/phase distinction; its additional UM/IL
preservation and recovery theorems retain their own execution hypotheses.

Zero capacity, disconnected positive-conductance transport and other source
channels are different mechanisms. Zero capacity can freeze nonzero pressure;
disconnected transport permits componentwise constants. They are not
counterexamples to the positive-capacity connected-transport statement.

### A residual bound away from exact locking

Use the same degree-weighted capacity mean omega at an arbitrary admitted
strictly acute state, and define the exact phase-rate residual
`epsilon_i=v_i-omega`. Edge cancellation still gives `sum_i d_i*epsilon_i=0`
and

```text
Im(Z_i) = d_i*(omega-nu_i+epsilon_i)/K,
g_i = atan((omega-nu_i+epsilon_i)/(K*r_i))/pi.
```

For common capacity and the edge bound rho this yields

```text
|g_i| <= atan(|epsilon_i|/(K*cos(rho)))/pi
       <= |epsilon_i|/(pi*K*cos(rho)).
```

Thus a nonzero observed phase-rate residual can support a nonzero transient
source even when all capacities agree. A small residual is not an exact lock,
and this algebra does not prove the residual will decay. Computed sine sums,
resultants, production pressure and binary64 phase-step residuals retain their
separate numerical errors; a numerical value near zero is not a symbolic proof.

### Capacity contrast is necessary here, not sufficient for stationary form

For any fixed positive capacities, multiplication of the full form row by
`H_i=s_i/nu_i` gives the mean balance

```text
d(m_H(x))/dt = w*sum_i s_i*g_i / sum_i s_i/nu_i.
```

A stationary form with `w>0` therefore requires `sum_i s_i*g_i=0`.
For this fixed connected transport, this is also the Poisson compatibility
condition for the declared source. It is not implied by the phase identity
`sum_i d_i*(omega-nu_i)=0`: taking phasor arguments is nonlinear, and transport
strengths need not equal phase-support counts. The existing
[`derive_forced_support_balance`](../../src/tnfr/physics/forced_support.py) owns
the compatible-profile and drift calculations.

An exact countercontrol uses the unit star K1,3 with center first. Declare
`K=1` and rational capacities

```text
nu = (5/6, 3/2, 3/2, 1/2).
```

The common rate is one. The principal leaf equations give the strictly acute
lock `psi=(0,pi/6,pi/6,-pi/6)`. Set
`gamma=atan(1/(3*sqrt(3)))`. Its canonical source and compatibility sum are

```text
g = (gamma/pi, -1/6, -1/6, 1/6),
sum_i d_i*g_i = 3*gamma/pi - 1/6 > 0.
```

For the strict inequality, `0<gamma<pi/6` and
`tan(3*gamma)=10/(9*sqrt(3))>tan(pi/6)`; both compared angles lie in
`(0,pi/2)`. Thus the lock's nonzero source has nonzero mean drive. No stationary
EPI solves this phase/form model, although the forced-support owner can still
describe its relative profile and common drift. The capacities precede the
phase and form calculation; no desired EPI target defines them retrospectively.
Section 28's reflection symmetry supplies compatibility for its barbell;
heterogeneity by itself does not.

### Admission boundaries and represented observations

- **Missing gated edges:** phase motion averages only admitted neighbors, while
  canonical pressure reads all support neighbors. On P2, a gap `1/2` with an
  effective gate `1/4` removes both phase interactions. Equal capacities then
  give common free advance while full-support pressure has
  `g=(1/(2*pi),-1/(2*pi))`. This is outside full-support admission even though
  the gap is acute. The full-support identities cannot silently use the
  smaller admitted-neighbor denominator.
- **Half-pi boundary and zero resultants:** ideal C4 quarter-turn winding has
  opposite neighbor phasors and `Z_i=0`; its direction is undefined. Binary64
  trigonometric evaluation can leave a tiny nonzero resultant, which must not
  be mistaken for an exact nonzero ideal direction or exact degeneracy.
- **Nonacute branches:** an ungated antipodal P2 has zero sine sum and a negative
  real resultant, so equal capacity does not imply zero phasor pressure. This
  is a mathematical boundary control, not an admitted production U3 branch:
  the engine's hard gate cannot exceed `pi/2`, and drops those interactions.
- **Nonreciprocal support:** a directed C6 with one outgoing neighbor per node
  and winding one has common rate `kappa+K*sin(pi/3)` and `g=1/3`. The reciprocal
  cancellation and degree-weighted rate formula do not apply.
- **Other dynamics:** capacity evolution, changes of support, different phase
  laws, additional pressure channels and Gamma require their actual balances.
  The theorem does not authenticate an executor or admit all configured graph
  writers merely because one snapshot satisfies the geometric assumptions.

The detached [`observe_phase_lock_source`](../../src/tnfr/physics/phase_response.py)
read-out keeps the support, capacity, actual phase-pressure capture and
numerically evaluated lock/source relations together. It reports numerical
residuals, not an `is_lock` decision or an exact trigonometric certificate.
The [independent controls](../../tests/physics/test_phase_lock_source.py) exercise
the production owners and the symbolic identities and boundaries above.
Exact theorem hypotheses, represented coefficients, finite numerical evidence
and autonomous-law selection remain distinct.

## 30. Acute phase locks are circulation states with integral cycle periods

### Fixed support, common capacity and the declared phase law

Retain Section 29's connected simple undirected phase support, full U3
admission, common fixed capacity `nu_i=kappa>0` and supplied coupling `K>0`.
The phase law is

```text
theta_dot_i = kappa + (K/d_i)*sum_(j in N_i) sin(theta_j-theta_i).
```

Every oriented shortest-arc edge gap is strictly inside `(-pi/2,pi/2)`.
This is a local condition; no common real-valued phase chart is assumed.
The theorem concerns exact common-rate solutions of this supplied phase law.
It does not identify continuous sine evolution with UM events, establish
stability, or derive a phase or support law from the nodal EPI equation.

Choose an arbitrary orientation of each of the `M` support edges. Let B be
the `N` by `M` incidence matrix with `-1` at the tail and `+1` at the head.
Write

```text
delta_e = wrap(psi_head(e)-psi_tail(e)),
f_e = sin(delta_e),
-pi/2 < delta_e < pi/2.
```

The neighbor sine sum at node i is `-(B*f)_i`. Section 29 fixes the locked
rate to kappa, so the equal-capacity locking equations are exactly

```text
B*f = 0.
```

Thus f is an edge circulation with no nodal sources or sinks. This is an
algebraic observation derived from the existing phase coordinates, not a new
primitive or an independently prescribed physical current. Transport
conductances do not enter this unweighted phase balance.

### Integral periods and reconstruction

Choose a spanning tree T. For each edge outside T, orient the corresponding
fundamental cycle so that this chord has coefficient `+1`; complete the cycle
with its unique return path in T. The resulting matrix C has one signed
cycle column per chord. It has `b=M-N+1` columns, `B*C=0`, and its chord rows
form the identity. These columns span all real circulations: subtracting the
combination given by a circulation's chord entries leaves a circulation
supported on a tree, which must vanish successively at leaves. The same
argument for integer entries shows that C is also an integer lattice basis.

For a circle-valued phase assignment, each closed cycle returns to its
starting phase, hence

```text
C^T*delta = 2*pi*w,       w in Z^b.
```

Conversely, these fundamental-cycle period conditions suffice. Fix a root
phase and integrate delta along the tree to assign real lifts to every
vertex. The period condition for each chord says that its supplied gap
differs from the difference of those lifts by an integer multiple of `2*pi`.
Reducing all lifts modulo `2*pi` therefore reconstructs the required edge
gaps. Strict acuity makes each supplied gap the unique shortest-arc value.
Connectedness makes the reconstruction unique modulo one common rotation.

The integer-lattice qualification matters. An arbitrary real cycle-space
basis cannot be used with an unmodified integer-period test. Even an integer
basis need not generate the whole integer lattice: on C5, replacing its
oriented cycle c by `2*c` would incorrectly admit constant gaps `pi/5`, since
`(2*c)^T*delta=2*pi` while `c^T*delta=pi`. Those gaps cannot come from closed
circle-valued node phases. Fundamental cycles prevent this false admission.
Another integer lattice basis changes winding coordinates by an invertible
integer matrix; the underlying phase configuration does not change.

### Necessary and sufficient locking equations, and sector uniqueness

Since sine is one-to-one in the acute interval, every lock has a unique
circulation coordinate z and satisfies

```text
f = C*z,
abs((C*z)_e) < 1                         for every edge,
C^T*asin(C*z) = 2*pi*w,                   w in Z^b.
```

Conversely, a solution of these equations supplies
`delta=asin(C*z)`. The period conditions reconstruct phases, the circulation
condition supplies zero neighbor sine sums, and common rotation at rate
kappa gives the required exact lock. Actual configured U3 limits must also
admit every reconstructed gap; the nonlinear equations do not override a
tighter gate. The criterion is an existence equivalence, not an assertion
that every integer vector w has a solution.

There is at most one strictly acute lock per winding sector w, up to common
rotation. To prove this, suppose `f=C*z` and `f'=C*z'` solve the same sector.
Then

```text
sum_e (f_e-f'_e)*(asin(f_e)-asin(f'_e))
    = (z-z')^T*C^T*(asin(f)-asin(f'))
    = 0.
```

Each summand is nonnegative and is strictly positive when `f_e!=f'_e`.
Consequently `f=f'`, then `delta=delta'`, and the reconstruction differs only
by common rotation. In particular, the zero-winding sector contains only
phase consensus, since zero gaps supply one solution in that sector.
The result gives uniqueness of a locked geometry, not convergence toward it
or invariance of the acute domain under future evolution.

An equivalent variational formulation makes the same distinction. On the
open convex domain `abs(C*z)<1`, define

```text
F_w(z) = sum_e [f_e*asin(f_e) + sqrt(1-f_e^2) - 1]
         - 2*pi*w^T*z,       f=C*z.
```

Its gradient is `C^T*asin(C*z)-2*pi*w`, and its Hessian is
`C^T*diag(1/sqrt(1-f_e^2))*C`, positive definite when `b>0`.
This provides the same at-most-one interior critical point. It does not
guarantee an interior critical point or equate this auxiliary circulation
functional with the engine's EPI pressure or dynamical Lyapunov function.

### What graph structure permits or excludes

- **Trees:** `b=0`, so every circulation is zero. Every acute equal-capacity
  lock is phase consensus.
- **Bridges:** sum `B*f=0` over either side of a bridge. Its single cut term
  forces `f_e=0`, hence `delta_e=0`. Distinct cycle-bearing regions can carry
  their own allowed circulation, but the bridge endpoints have equal phase
  in a lock. A bridge does not generate a capacity-free phase source.
- **Short cycles:** a simple cycle of length l has
  `abs(sum_cycle delta)<l*pi/2`. For `l<=4`, its integral winding must be zero.
  If cycles of length at most four span the real cycle space, all acute
  locks are consensus. Indeed, delta is then orthogonal to every circulation,
  so `delta^T*f=0`; every term `delta_e*sin(delta_e)` is positive unless
  `delta_e=0`. Here real spanning is sufficient because the periods already
  vanish, unlike the integer reconstruction problem above.
- **Long cycles are necessary somewhere, not sufficient by themselves:** a
  nonzero acute lock must have nonzero winding on some simple cycle, which
  must have at least five edges. The mere presence of a long cycle does not
  imply a nonzero lock. K5 contains a five-cycle, but its cycle space is
  generated by triangles, excluding any nonconsensus acute lock.
- **A single cycle with attached trees:** bridges have zero gap and circulation
  is constant around the unique n-cycle. The complete acute lock family is
  `delta_cycle=2*pi*w/n`, with integer `abs(w)<n/4`, additionally restricted
  by the actual U3 gate. Tree vertices share the phase of their attachment
  vertex. Thus the unit windings of C5 and C6 are the first nonzero members,
  while C4's unit winding lies on the excluded half-pi boundary.

For the third and fourth statements, if every simple-cycle period were zero,
the fundamental-cycle periods would be zero and sector uniqueness would give
consensus. No enumeration of every graph or a new numerical recovery sweep
is needed to obtain these obstructions.

At each admitted acute lock, `B*f=0` and positive neighbor cosines imply zero
canonical phase pressure, as proved in Section 29. Nonzero winding still
gives nonzero local absolute phase gradients on its nonzero-gap edges. A
unique phase geometry in a sector can therefore retain information that a
zero-pressure label alone does not describe. This classifies candidate phase
identity; it does not prove differentiated stationary EPI, autonomous NFR
formation, physical identity or a complete state description by winding alone.

### Persistence of a sector is not its formation

For a continuous circle-valued phase trajectory on fixed support, retain the
same oriented fundamental cycles. If no edge reaches antipodal separation,
its wrapped gap varies continuously inside `(-pi,pi)`. Every cycle period is
both continuous and an integer multiple of `2*pi`; its winding is therefore
constant on that time interval. This argument does not require the sine law,
equal capacities, locking, or even acute gaps.

The U3 boundary and the winding boundary are different. Crossing a configured
gate at or below `pi/2` can change which neighbors the phase law uses, without
changing a support-cycle winding. A winding change on fixed support requires
an antipodal branch encounter at `abs(delta)=pi` in any continuous phase
history. Loss of U3 admission is therefore not evidence of a winding slip.
Conversely, gate admission at finitely sampled endpoints does not certify
the unobserved path or exclude cancelling slips. Discrete operator events
need their own path witness or endpoint interpretation.

Changing support can create or remove the cycle whose winding is being
classified. For example, prepare a five-vertex path with consecutive phase
gaps `2*pi/5`, then add the edge joining its endpoints. All five resulting
cycle gaps are acute and the new cycle has winding one, with no phase change
or branch encounter. The prepared phase arrangement and edge addition are
inputs in this example; it is not a derivation of their endogenous selection.
Comparing sectors across a support change therefore requires identifying
which old cycles survive and which cycle classes are new.

This separates two research obligations: maintaining a phase configuration
within an existing sector, and explaining how the retained configuration and
its support sector are produced. The protected UM results in the
[winding owner](../COUPLING_WINDING_PERSISTENCE.md) address specified preservation
and recovery maps. They neither replace the present continuous sine law nor
supply an autonomous mechanism that selects nonzero winding from a
zero-winding, fixed-support, branch-safe preparation.

### Exact implementation boundary

[`phase_cycle_geometry.py`](../../src/tnfr/physics/phase_cycle_geometry.py)
centralizes oriented support and the fundamental integer cycle basis. Its
exact rational-turn reconstruction uses `a_e=delta_e/(2*pi)`, requires
`abs(a_e)<1/4`, and tests that `C^T*a` has integer entries. This certifies
declared edge-phase compatibility without converting rounded trigonometric
residuals into an exact lock claim. The represented live phases and actual
U3 execution still require their separate observations and admission.

The implemented symbolic odd-sine cancellation test groups equal absolute
turns with opposite incidence signs at each node. A successful exact
cancellation supplies a sufficient locking witness; failure to cancel this
way is inconclusive, since more general trigonometric identities may also
balance a node. The implementation does not solve the full nonlinear
`C^T*asin(C*z)=2*pi*w` existence problem. The
[independent controls](../../tests/physics/test_phase_cycle_geometry.py) retain
these topology, reconstruction and sufficient-witness boundaries separately
from finite checks against the production phase and pressure owners.

One control makes this incompleteness explicit: three internally disjoint
paths between the same endpoints have lengths `(10,6,10)` and constant
oriented turns `(1/20,1/12,-3/20)`. Their endpoint differences agree modulo
one turn. Internal nodes cancel pairwise, while the endpoints use
`sin(pi/10)+sin(pi/6)=sin(3*pi/10)`, verified by the exact coefficients of
`1` and `sqrt(5)`. This is an acute lock under the supplied equal-capacity
law, but the oddness-only checker correctly reports `unresolved`. A failed
sufficient symbolic check must therefore not be presented as nonexistence.

## 31. One added chord extends the cycle lattice, not a phase-generation law

### Retained support and the exact lattice identity

Let `G=(V,E)` be a connected simple undirected phase support. Add exactly one
missing edge `e_*=(u,v)` between its existing vertices, retaining every old
edge and vertex. Orient the new edge from u to v and retain the orientations
of the old edges. No conductance, capacity, phase law or locking assumption
is needed for the following topology identity. Connecting distinct components,
deleting edges, changing node identity, loops and parallel edges are different
operations outside this one-chord result.

Retain an old spanning tree T. Write R for the old fundamental cycle basis
in **rows**, so `B*R^T=0` for Section 30's oriented incidence B. Each row has
coefficient `+1` on its own non-tree edge. Embed R in the new edge space by
inserting zero in the added-edge column. Let r be the signed old-edge vector
of the unique tree path from v back to u. Appending the row

```text
c_* = (r, 1)
```

gives the retained-tree basis `R_ext` of the new graph. Here the displayed
last coordinate denotes the added edge; implementations may retain their
lexicographic edge order instead. The edge e_* followed by this tree path
is a closed oriented cycle, so `B_+*c_*^T=0`.

The new cycle rank is `b_+=b+1`. More strongly, these rows generate the full
**integer** cycle lattice. For any integer circulation h on the new support,
subtract its added-edge coefficient times c_*. The remainder has zero on
the added edge and is an integer circulation on G, hence an integer
combination of R's rows. Independence follows first from the added-edge
coordinate and then from independence of R. Thus

```text
Z_1(G_+; Z) = embedded Z_1(G; Z) + Z*c_*
```

is a direct sum for this chosen tree, and the quotient by the embedded old
lattice has rank one. This identifies the surviving old cycles before any
comparison of winding coordinates.

A fresh deterministic spanning-tree calculation can produce a different
post-event basis `R_+`. Both are integer lattice bases, so

```text
R_ext = U*R_+,            U and U^(-1) have integer entries,
det(U) = +1 or -1.
```

One can compute U without a real-valued inverse: its columns are the
entries of `R_ext` on the non-tree edge columns of `R_+`, whose corresponding
submatrix is the identity. The reciprocal construction supplies the integer
inverse. Basis changes therefore cannot create or erase a winding; they
change its coordinates. A comparison that simply subtracts two independently
chosen basis-coordinate lists need not compare the same cycles.

### The new period comes from the actual endpoint gap

Express a circle-valued phase field in turns, with oriented edge gaps
`a_e=wrap_turn(phi_head-phi_tail)` in `(-1/2,1/2)`. This statement excludes
antipodal edge pairs so that the shortest-arc gap is unambiguous. It does not
require strict acuity or U3 admission. For the final field on the new support,
let `a_old^+` be its gaps restricted to old edges and `a_*^+` its new-edge gap.
Then

```text
w_old^+ = R*a_old^+,
w_*^+ = r*a_old^+ + a_*^+,
w_+ = U^(-1) * (w_old^+, w_*^+).
```

Every displayed cycle period is an integer. The return-path sum differs
from `phi_u-phi_v` by an integer, while the new-edge gap differs from
`phi_v-phi_u` by an integer. Their sum is therefore integral. No desired
period supplies the gap: the existing path and actual endpoint phases
determine it. Conversely, appending an arbitrary prescribed edge gap without
checking this relation can fail circular reconstruction.

The tree is a coordinate choice. Replacing its return path by another old
return path changes c_* by an old integer cycle, so

```text
w_*' = w_* + z^T*w_old              for some integer vector z.
```

The quotient lattice has a single generator up to orientation, but a winding
functional descends to that quotient only when it vanishes on all old cycles.
In general, a particular numeric "new winding" is therefore a retained-path
coordinate, not an intrinsic new scalar charge. The complete embedded old
periods together with the chosen new period are unambiguous data.

### Separate a support extension from phase writes at the same event

For branch-safe initial and final phase fields, the old cycles have endpoint
change

```text
Delta_w_old = R*(a_old^+ - a_old^-).
```

If the support-only operation leaves the phase gaps unchanged, this is zero.
The new cycle can nevertheless have nonzero winding. For example, prepare
a five-vertex path with ascending edge turns `1/5` and add the endpoint edge.
With the implementation's orientation `0 -> 4`, its gap is `-1/5` and the
tree return path has sum `-4/5`, giving `w_*=-1`. Traversing the same cycle
as `0 -> 1 -> 2 -> 3 -> 4 -> 0` gives `+1`. All gaps are strictly acute;
there is no phase change or pre-existing cycle whose winding slipped.
The initial nonuniform path field and the added edge remain supplied inputs.

When an actual event also writes phase, compute the final new period from
the **final** path gaps and endpoint gap. A useful counterfactual is to add
the same edge to the old phase field. With `S^-=r*a_old^-`, it has

```text
a_*^0 = wrap_turn(-S^-),
w_*^0 = S^- + a_*^0,
```

provided the old endpoints are not antipodal. Its gap can be nonacute even
when both actual endpoint states satisfy their strict-acute domains. If the
counterfactual endpoints are antipodal, retain its unavailability instead of
choosing a winding by a rounding convention. When defined, `w_*^+-w_*^0`
records an endpoint difference relative to that old-phase extension.

This decomposition does not claim that the counterfactual stage actually
executed, or impose an order on simultaneous phase and support writes.
An executor's event provenance and before/after fields are separate evidence.
Neither equal endpoint periods nor a changed endpoint period describes an
unobserved continuous phase path. On fixed support, an actual continuous
winding change requires an antipodal branch encounter; a discrete operator
jump instead supplies an endpoint change unless a path is also specified.
Crossing the U3 gate alone is still not a winding slip.

### A larger cycle lattice does not guarantee a maintained lock

The rank increase is topological. It neither supplies sine balance nor
preserves the set of acute locks. Adding a diagonal to C5, for example,
produces independent triangle and square cycles spanning its two-dimensional
cycle space. Section 30 then excludes every nonconsensus acute equal-capacity
lock under the fully admitted supplied sine law. The original C5 unit winding
has a nonacute gap across either such diagonal, consistently failing the
new full-support acute premise. Adding a connection can thus remove a
previously available lock class while increasing the number of cycle
coordinates.

A topology-only path closure, an operator event that closes the path while
writing phase, and subsequent relaxation to a lock are three distinct
claims. This support-reset identity establishes the first comparison and
accounts for the second's observed endpoint phases. It does not establish
the third, derive candidate selection, or show autonomous generation of a
coherence pattern or a physical entity.

### Shared implementation and production boundary

[`phase_cycle_geometry.py`](../../src/tnfr/physics/phase_cycle_geometry.py)
owns the one-chord support extension and exact endpoint reset alongside its
existing fundamental-cycle basis. The implementation requires the same
ordered nodes and exactly one added edge, rederives supplied geometry/state
data, and retains both integer basis-coordinate maps. Its exact phase reset
uses the existing rational-turn `PhaseCycleState` domain: both actual
endpoint fields are strictly acute and reconstructible. The topology and
branch-safe mathematical identities above have broader scope; this finite
implementation does not silently certify that larger domain.

The counterfactual old-phase extension is reported separately, including
antipodal unavailability. Exact turns are declared angles divided by
mathematical `2*pi`; dividing binary64 runtime radians by represented tau
does not turn an observed event into an exact-angle certificate. Actual
operator endpoints instead retain the shared branch-aware winding observer
and their numerical residuals. The
[exact reset controls](../../tests/physics/test_phase_chord_reset.py) and
[Coupling event control](../../tests/physics/test_coupling_sector_birth.py)
exercise those different evidence paths. Neither the graph-theoretic result
nor a finite observed sector birth derives the Coupling writer's configured
candidate, threshold or Sense Index policy.

### Prospective default-Coupling event and policy control

The finite control initializes the five-node path with `theta_j=2*pi*j/5`,
`EPI=1/8`, `nu=1`, Si `=0.8`, unit old conductances and refreshed pressure.
Its four path gaps span `4/5` of a turn; there is no cycle or initial winding.
It uses the existing atomic two-phase Coupling executor on target 0 with
declared history `AL`, seed 17, all eligible candidates, and the default UM
factors, bidirectional phase writes, capacity synchronization, pressure
stabilization and functional-link threshold. No factor or threshold is tuned.
The pressure observation uses effective phase/EPI weights `(1/2,1/2)` and
zero capacity/topology weights, without Gamma.

Write `f=1/(pi+1)` for the default phase push. Before execution, the kernel
predicts node 0's displacement `+f*pi/5` and node 1's `-f*pi/5`; other phases
are unchanged. Only nonneighbor 4 is initially phase-compatible. The final
closing gap and maximum old-edge gap are `(2+f)*pi/5<pi/2`. The resulting
compatibility score for equal form and Si is

```text
a = 1-(2+f)/10 = 0.7758546992994776,
threshold = pi/(pi+1) = 0.7585469929947761.
```

The executed stage adds exactly edge `(0,4)` with represented weight a.
Its observed winding is `+1` around `(0,1,2,3,4)`, or `-1` in the exact
reset owner's added-chord orientation. The finite U3 margin is approximately
`0.16244986676002404` radians. Actual phase writes agree with the prediction;
EPI and capacity stay unchanged. The retained stage result and histories
identify this finite invocation, without asserting an unobserved future.

The final state is not a sine lock. On its acute cycle, the ideal phasor
source is

```text
g = (f/10)*(-3,3,-1,0,1).
```

The runtime source matches this within the test's separate binary64 allowance.
Coupling's target-local pressure reduction leaves a positive stored pressure
at node 0, while a canonical refresh gives negative pressure there. A direct
operator write is therefore not identified with freshly computed pressure.
The read-out retains both fields and their difference.

The closing edge is not unit conductance: transport strengths are
`s=(1+a,2,2,2,1+a)`, whereas phase-neighbor counts are all two. With
`F=g/2`, the initial source compatibility sum is

```text
sum_i s_i*F_i = f*(1-a)/10 > 0
               approximately 0.005412055686023124.
```

The shared forced-support observer confirms its positive represented value.
Thus the subsequent weighted EPI mean cannot simply be assumed conserved,
even though the unweighted phase-source sum is zero. Replacing the new edge
by a unit weight would change this actual event's form dynamics.

Two controls delimit the result. Changing only candidate 4's Si from `0.8`
to `0.0` lowers compatibility by `0.2` and suppresses edge creation. Disabling
functional links also prevents closure. Both preserve the same primary
phase, EPI, capacity, stored-pressure and history writes; the differing
support changes the subsequent pressure refresh. This is evidence of the
existing writer's policy dependency, not a proposal to use Si as a fundamental
support law. The event is admitted as finite, policy-selected sector birth;
autonomous selection remains open. The event alone does not establish
subsequent maintenance; Section 32 supplies its conditional phase/form result.
