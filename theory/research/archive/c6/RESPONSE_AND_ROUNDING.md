# C6 response, contraction and represented rounding

Parked C6 response domains, numerical reserves and rounding-cell restrictions.

Part of [Cycle winding under Coupling: scope and retained evidence](../../../COUPLING_WINDING_PERSISTENCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

## 8. One shared local response for circular means

B2.d.15 compares the previous
[antipodal obstruction](../../../CHILD_COUPLING_FEEDBACK.md#18-antipodal-phase-response-under-simultaneous-um-and-il)
with winding under the **default bidirectional all-target** UM stage. It does
not change operator coefficients or disable functional links. Sections 1-7
retain their distinct target-only hypotheses.

For a nonzero mean phasor `S_s=sum_{j in N_s} exp(i*theta_j)`, differentiation
in a regular local chart gives

```text
R_sj = 1[j in N_s]*Re(exp(i*theta_j)/S_s)
     = 1[j in N_s]*sum_{k in N_s} G_jk / sum_{l,k in N_s} G_lk,
G_jk = cos(theta_j-theta_k).
```

Rows sum to one by rotation covariance, but may have negative entries. The
antipodal bridge's `(1,1,-1)` row therefore cannot be treated as convex
diffusion. For one common phase factor `t`, receiver `i` averages the proposals
from its declared source targets `T_i`, giving the derivative
`J_ij=(1-t)*delta_ij+t*mean_{s in T_i} R_sj`. Pointwise IL uses only source
`i`; bidirectional UM includes every source that proposes a write to `i`.

[`phase_response.py`](../../../../src/tnfr/physics/phase_response.py) is the shared exact
owner for these two operations. It validates an exact unit planar cosine Gram:
symmetry, diagonal one, positive semidefiniteness and rank at most two. A Schur
complement about one unit vector must be a positive rank-at-most-one matrix;
this avoids numerical angle reconstruction. Zero mean resultants are rejected.
A rounded cosine table that fails these identities is not repaired silently.
Gram data do not identify oriented winding, live graph support, phase branches
or binary64 derivatives. Both the antipodal reference and the C6 reference
now derive their matrices through this common implementation.

## 9. Default all-target C6 response and a phase neighborhood

Consider the mathematical winding base `theta_i=i*pi/3`, `i=0,...,5`, on the
unit simple cycle, with the canonical gate `pi/2`. Adjacent phases differ by
`g=pi/3`; non-neighbors have circular separation at least `2*pi/3`. Thus the
base has strict gate margin `pi/6` both for existing edges and against new
compatible nonedges. This base is declared preparation, not a derived birth.

Let `A` denote cycle adjacency. Each closed UM neighborhood has phasor
resultant two, whereas each open IL neighborhood has resultant one. Hence

```text
R_closed = I/2 + A/4,                  R_IL = A/2,
W = (I+A)*R_closed/3 = (2*I+3*A+A^2)/12,
U = (1-t)*I+t*W,                      M = (1-alpha)*I+alpha*A/2,
P = M*U.
```

Every UM receiver averages three sources: itself and its two neighbors.
The individual base displacements `+t*g,0,-t*g` cancel in the merge; they do
not vanish individually. This distinguishes the default simultaneous stage
from both one bidirectional target and the earlier target-only stage.

The Fourier multipliers of `W`, in mode order `k=0,...,5`, are
`(1,1/2,0,0,0,1/2)`. Those of `P` are

```text
(1,lambda1,lambda2,lambda3,lambda2,lambda1),
lambda1 = (1-t/2)*(1-alpha/2),
lambda2 = (1-t)*(1-3*alpha/2),
lambda3 = (1-t)*(1-2*alpha).
```

For centered tangent energy `Q(e)=sum_i(e_i-mean(e))^2/2`, symmetry gives the
optimal linear bound `Q(Pe)<=q*Q(e)`,
`q=max(lambda1^2,lambda2^2,lambda3^2)`. It is strictly below one for
`0<t<=1, 0<=alpha<=1`. The global rotation multiplier remains one, so full
uncentered contraction is not claimed. The algebraic boundary `t=0` remains
a detached control, not canonical UM admission. This exact reference uses the
rational cosine Gram of C6; mathematical `pi` is not replaced by `math.pi`.

### A sufficient nonlinear phase-map invariant box

An exact phase neighborhood can also be retained without linearizing. Write
`theta_i=i*g+e_i` in consistent periodic lifts, with all `e_i` in
`[m-r,m+r]` and `0<=r<=pi/24`. The center `m` is a common rotation, not a
new interaction parameter. The radius is a sufficient analytical bound.

Translate `m` to zero and examine one closed UM sum relative to its base
center. Its unperturbed value is two, and `|exp(i*e)-1|<=|e|` gives
`|S-2|<=3*r`. Since `r<=pi/24<1/6`, `Re(S)>=3/2` and
`|Arg(S)|<=2*r`. Every included phase is then at angular distance at most
`g+3*r<=11*pi/24<pi/2` from the mean. All its mean derivatives are positive.
Monotonicity between the all-lower and all-upper preparations consequently
sharpens the mean error `c_s` to `[m-r,m+r]`.

The source-to-receiver shortest arcs stay on their fixed branches. After
averaging the three baseline displacements, the exact UM error is

```text
e_i' = (1-t)*e_i + t*(c_(i-1)+c_i+c_(i+1))/3.
```

This is a convex combination in the same interval. For IL, the two neighbor
phases have lifted separation at most `2*g+2*r<=3*pi/4<pi`. Their circular
mean is the midpoint, so

```text
e_i'' = (1-alpha)*e_i' + alpha*(e_(i-1)'+e_(i+1)')/2.
```

It preserves the interval as well. Throughout these exact phase maps,
adjacent separation is at most `g+2*r<=5*pi/12`, and nonedge separation is
at least `2*g-2*r>=7*pi/12`. Both retain margin `pi/12` from the gate;
functional-link creation therefore has no compatible candidates, and winding
stays one. This proves invariance for finite repetitions of these restricted
exact phase maps. It does not assert the linear factor `q` globally throughout
the box, nonlinear asymptotic convergence, positive-EPI admission, a repeated
complete engine policy or a binary64 invariant class.
The fixed interval center `m` is not asserted to be the conserved arithmetic
error mean. Nonlinear UM generally has `mean(e')=(1-t)*mean(e)+t*mean(c)`,
which can differ from `mean(e)`. IL preserves that mean inside the midpoint
chamber; both local linear Jacobians preserve it at the winding base.

### Connection to nodal pressure

In this same midpoint neighborhood, the phase-pressure channel satisfies the
exact identity `pi*g_phase=(A/2-I)*e`. IL's phase jump and the pressure readout
therefore use the same structural local discrepancy, with their distinct
configured coefficients. Nodal EPI evolves through
`dEPI/dt=nu_f*(w_epi*g_epi+w_phase*g_phase+...)`; a phase update is not an
independent continuous EPI law. The implementation retains the pi-scaled
tangent as a rational vector. Finite captures separately measure represented
phase pressure and its difference from the midpoint and tangent expressions.

## 10. Finite default C6 controls and remaining scope

[`c6_winding_phase_response.py`](../../../../benchmarks/c6_winding_phase_response.py)
predeclares five preparations: the nominal winding, plus positive and negative
`epsilon=2^-12` in directions

```text
k1 = (1,1/2,-1/2,-1,-1/2,1/2),
k3 = (1,-1,1,-1,1,-1).
```

EPI is initially `0.5`, capacity is one, all six conductances/lengths are one,
and the seed is 17. The single stored base tuple `i*math.pi/3` is used both
for preparation and wrapped phase comparisons. The default bidirectional UM,
capacity mixing and functional links remain enabled. Each case executes one
all-target UM, refresh, IL, refresh, one shared `h=0.25` Euler interval with
four held-pressure substeps, then SHA after measurement.

The shared stage observer retains actual admission, an independent strict IL
readiness check, pure-kernel predictions versus actual writes, histories,
metrics and signed reference/flow budgets. Its phase readout is now supplied
explicitly, so the same mechanics serve the earlier antipodal controls without
imposing their phase chart on this cycle. The existing
[`certify_phase_winding`](../../../../src/tnfr/physics/winding_certificates.py) observes
the actual oriented cycle with the resolved UM gate. Rotation drift,
centered phase deviation, new-link exclusion margins and numerical residues
are recorded separately. A rational tangent sum never supplies the winding
integer.

The finite measured centered-energy ratios after UM/IL are:

| Preparation | Observed Q_after/Q_before | Linear prediction |
|-------------|---------------------------|-------------------|
| `+2^-12*k1` | `0.558580560057` | `0.558580559487` |
| `-2^-12*k1` | `0.558580560058` | `0.558580559487` |
| `+2^-12*k3` | `0.092062966493` | `0.092062966493` |
| `-2^-12*k3` | `0.092062966493` | `0.092062966493` |

The small `k1` excess above the tangent bound is retained, not rounded into a
finite nonlinear certificate. The `k3` scalar response has the exact local
finite multiplier `(1-t)*(1-2*alpha)` in its alternating chart, with recorded
binary64 residuals. Every observed stage and flow endpoint retains winding
one and excludes all nine nonedges; the minimum observed nonedge margin is
about `0.523325` radians. Phase stays fixed during the nodal interval while
EPI responds to its pressure; for positive `k1`, EPI changes from uniform
`0.5` to approximately
`(0.499994493,0.499997246,0.500002754,0.500005507,0.500002754,0.499997246)`.
This is a finite forced response, not EPI localization or self-restoration.
The null preparation retains a phase residual around `1.30e-17`; its initial
centered phase energy is zero, so the ratio is undefined. Its stored EPI
remains uniform after the interval despite small represented pressure
residuals. Neither coordinate stasis nor a tiny residual proves a full fixed
runtime state.

The exact matrix, mode and cache-revalidation controls extend
[`coupling_winding.py`](../../../../src/tnfr/physics/coupling_winding.py) and live in
[`test_c6_winding_phase_response.py`](../../../../tests/physics/test_c6_winding_phase_response.py).
[`test_phase_response.py`](../../../../tests/physics/test_phase_response.py) supplies
independent signed-response, Gram and receiver-average controls. The finite
[`runtime tests`](../../../../tests/physics/test_c6_winding_phase_response_runtime.py)
also compare independent nonlinear phasor formulas, preserve the null case's
roundoff evidence, and verify winding, actual source refresh and nodal flow.

Generate the independently manifested local artifact with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_phase_response.py
```

This block supplies a stronger prepared **phase** candidate than the antipodal
UM/IL configuration. A retained winding and phase neighborhood do not by
themselves preserve a localized EPI shape. The joint exact-model result below
addresses pressure refresh and nodal integration. Formation of the winding
still needs a declared canonical mechanism. Autonomous physical structures
and laboratory correspondence remain open.

## 11. Nonlinear contraction and a joint phase/capacity/EPI domain

Keep the prepared unit C6, all-target bidirectional UM and midpoint IL from
section 9, with fixed factors `0<t<=1` and `0<=alpha<=1`. Write phase error
in units of mathematical pi, `z_i=e_i/pi`, and let
`v=osc(z)=max(z)-min(z)<=1/12`. This normalization is an analysis coordinate,
not a new nodal field or an approximation of pi. The interval midpoint
supplies the common rotation `m`; the phase radius is at most `pi/24`.

### A uniform nonlinear oscillation bound

The sharpened closed-mean error lies between the smallest and largest input
errors. Every participating phase is therefore within `g+2*r<=5*pi/12` of
that mean. For its nonzero phasor sum `S`,

```text
d Arg(S)/d theta_j = cos(theta_j-Arg(S))/|S|,
cos(5*pi/12)>1/4,       |S|<=2+3*r<=2+pi/8<5/2.
```

Each of the three active derivatives exceeds `1/10`. This rational lower
bound follows from the geometry; it does not replace an operator coefficient.
Receiver averaging in UM consequently supplies derivative at least `t/30`
for each of the five columns at cycle distance at most two. Every two such
rows share at least four columns. Their overlap is at least `2*t/15`, so the
row-stochastic Jacobian contracts oscillation by at most

```text
rho = 1-2*t/15 < 1.
```

Integrate the Jacobian along the segment from a common rotation to the input
inside the same invariant box. The same row overlap survives integration.
The midpoint IL map is row stochastic and cannot increase oscillation.
Thus this bound applies to the nonlinear phase map `T`, not just its tangent:
`osc(T(z))<=rho*osc(z)`. It is a conservative range bound, distinct from the
optimal local squared-energy factor in section 9. No assertion that the
local quadratic factor holds throughout the nonlinear box is needed.

Both phase maps keep each error inside its previous min/max interval.
Nested intervals and `v_n<=rho^n*v_0` imply convergence to a common error
`z_infinity`: a rotated uniform winding. That rotation need not equal the
initial arithmetic mean. This is an exact phase-map convergence theorem on
the declared domain, not a binary64 or complete-runtime limit theorem.

### From phase pressure to a preserved EPI reserve

Assume uniform fixed positive capacity `nu`, fixed normalized channel weights
`w_epi>0,w_phase>=0`, and a held Euler interval of duration `h>=0` after each
UM/IL phase update. Uniform capacity is unchanged by these UM and IL maps.
The unit regular support gives zero capacity and topology gradients. The
midpoint pressure identity and the nodal equation yield

```text
z_plus = T(z),
DeltaNFR = -w_epi*L_rw*x - w_phase*L_rw*z_plus,
x_plus = (I-s*L_rw)*x - b*L_rw*z_plus,
s = h*nu*w_epi,                 b = h*nu*w_phase.
```

These are the refreshed inputs at the interval boundary. The default solver's
four held-pressure substeps do not refresh diffusion inside the interval.
In exact arithmetic they trace the same straight segment and have the
displayed endpoint. Require `0<=s<=1`, so `I-s*L_rw` is row stochastic.
Since `||L_rw*z_plus||_infinity<=v_plus<=rho*v`,

```text
min(x_plus) >= min(x)-b*v_plus,
max(x_plus) <= max(x)+b*v_plus.
```

Define the derived future-forcing reserve coefficient

```text
B = b*rho/(1-rho).
```

The identity `(b+B)*rho=B` proves that both inequalities

```text
min(x)-B*v >= ell,             max(x)+B*v <= U
```

are preserved at cycle endpoints. For declared `0<ell<=U<=1`, this is a
joint phase/EPI domain with constant positive capacity. Every held internal
substep is between the old and new EPI endpoints, so it also stays in
`[ell,U]`; hard clipping in the canonical interval `[-1,1]` is unnecessary.
At an internal fraction `u` of the interval, the same estimate uses
`(u*b+B)*rho<=B`, so the reserve inequalities hold there as well. This does
not make every internal substep a new pressure-refreshed cycle. The phase
box, winding and exclusion of new links persist throughout the held interval.

The same coefficient gives the explicit accumulated forcing bound
`sum_(n>=0) b*||L_rw*z_(n+1)||_infinity<=B*v_0`. The configured timestep
and channel factors remain declared inputs. The bound derives their allowed
effect; it neither chooses an action nor adds a restoring pressure law.

The positive lower bound is not, by itself, every application condition.
The runtime must also match the actual UM minimum EPI, capacity and phase
gate, independent strict IL readiness, hard clipping, pressure realization
and word/history context. SHA is a terminal closure and is excluded from
this constant-positive-capacity domain.

### What tends to persist in this restricted model

The exact EPI arithmetic mean is conserved because both Laplacian terms
sum to zero. For fixed `0<s<1`, its nonuniform-mode factor is

```text
gamma = max(|1-s/2|, |1-3*s/2|, |1-2*s|) < 1.
```

Writing `y=x-mean(x)*1`, the norm estimate
`||y_(n+1)||_2<=gamma*||y_n||_2+sqrt(6)*b*v_0*rho^(n+1)` tends to zero by
the geometric convolution. Therefore EPI tends to its initial mean while
phase tends to a rotated winding. This class supports prepared phase
organization; it does not maintain an asymptotically localized EPI shape.
At `s=1`, zero phase error and an alternating EPI perturbation give multiplier
`-1`, so positivity and boundedness do not imply convergence. At `h=0`, EPI
does not evolve. Neither boundary is silently promoted to the strict theorem.

### Executable algebra and its boundary

The joint-domain reference and observer extend
[`coupling_winding.py`](../../../../src/tnfr/physics/coupling_winding.py), reusing the
cycle Laplacian and exact input readers. The reference computes `rho`, `B`,
the Euler matrix and the separate EPI mode bound. The observer receives six
declared pre/post phase-error coordinates in pi units and the EPI state. It
checks interval nesting and contraction, evaluates the exact nodal endpoint
and reports the input/output reserves. Public reference caches are rebuilt.
Supplying two vectors that pass these checks does not authenticate a
production phase update, its phase chart or its operator admission.

## 12. Two-cycle admission and the represented null boundary

[`c6_winding_joint_domain.py`](../../../../benchmarks/c6_winding_joint_domain.py)
predeclares exactly two UM/IL/flow cycles for three inherited preparations:
the nominal winding and the positive `k1` and `k3` probes at `2^-12`.
Each cycle uses one `h=0.25` interval; a single terminal SHA follows both
measurements. The graph, default operator factors, seed and source preparation
are unchanged from section 10. No horizon or coefficient search is performed.

The EPI band is read from actual application and clipping settings. On the
positive chart, default UM requires `EPI>=0.05` and `nu>=0.01`, whereas
independent strict IL requires `EPI>0` and `nu>0`. The chosen band therefore
has lower endpoint equal to the represented UM floor and upper endpoint one.
The benchmark calls the independent UM/IL/SHA precondition validators in
addition to both whole-word validators and actual per-stage admission.
It does not enable a different precondition policy to obtain acceptance.

The full word `UM IL UM IL SHA` passes both word validators. A separate
read-only `UM UM SHA` control records string-validator rejection alongside
`ValidatedSequence` acceptance and live-candidate acceptance after the first
UM. It is not executed. These interfaces answer different checks and their
results must remain visible; a string-word refusal is not a universal live
kernel refusal. This preserves the earlier P2 distinction rather than
changing the transition policy during a research comparison.

For the executed defaults, the conservative exact-model coefficients are

```text
rho  approximately 0.967806265733,
B    approximately 5.700849477232,
gamma approximately 0.977105818448.
```

The initial positive `k1` and `k3` probes have represented pi-scaled phase
diameter about `0.000155425`, well inside the prepared box and the joint
reserve band. The entire maximum-width phase box at uniform EPI `0.5`
would only give a conservative lower reserve about `0.0249292`, below UM's
floor. Positivity or phase-box membership alone therefore does not satisfy
this sufficient joint admission condition. Failure of a sufficient reserve
does not prove that an actual trajectory must cross the floor.

Both nonzero preparations pass the conditional phase-transition checks in
each cycle. The benchmark separates the exact nodal endpoint on the supplied
coordinates from pressure realization and actual integrator endpoint defects;
their signed sum closes exactly. All three preparations retain winding one,
unit capacity before closure and admissible EPI in the observed words.
For `k1`, the minimum stored EPI changes from `0.5` to approximately
`0.499994493` and `0.499990503`; the corresponding lower future-forcing
reserves are approximately `0.499332272` and `0.499495571`. These are finite
measured reserves, not a runtime invariant-class theorem.

The null preparation exposes the next boundary. Its initial represented
phase error has zero diameter, but the first UM/IL stage produces a small
nonzero diameter. The homogeneous bound `v_plus<=rho*v` consequently fails
on that represented chart. The second cycle fails the conditional transition
check as well. Positive contraction residuals are approximately
`4.12e-18` and `2.90e-18` in represented pi units. The actual runtime words
still succeed. The report retains both outcomes rather than replacing the
observed phase by exact zero or weakening the test tolerance.

Captured wrapped errors are divided by represented binary64 `math.pi` for
these readouts; they are not silently identified with division by mathematical
pi. The null result refutes a defect-free transfer of the homogeneous bound
to this represented chart. It does not refute the exact phase theorem or
every possible numerical invariant class. Any such extension needs explicit
phase, pressure and EPI execution-defect bounds and the actual admission band.

The [exact-domain tests](../../../../tests/physics/test_c6_winding_joint_domain.py),
[independent boundary tests](../../../../tests/physics/test_c6_winding_joint_domain_adversarial.py)
and [runtime tests](../../../../tests/physics/test_c6_winding_joint_domain_runtime.py)
separate the geometric budget, Euler boundary, application floors, public
cache revalidation, finite word evidence and represented null obstruction.
Generate only this block's artifact with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_joint_domain.py
```

The additive extension below accounts for these retained defects. The exact
uniform-EPI limit must still guide action selection and formation: retaining
winding alone does not explain sustained EPI localization or physical
particle formation.

## 13. Additive defects, finite reserves and the neutral EPI mean

Reuse the same joint reference: `T=I-s*L_rw`, `b=h*nu*w_phase`,
`rho=1-2*t/15`, `B=b*rho/(1-rho)`. For supplied ordered endpoints define

```text
v = osc(z),        v_plus = osc(z_plus),
epsilon = max(0, v_plus-rho*v),
delta = x_plus - (T*x-b*L_rw*z_plus),
mu = mean(delta),                 delta_c = delta-mu*1.
```

These are arithmetic residuals of the supplied coordinates, not new driving
terms added to the nodal law. The additive observer retains them even
when the defect-free transition inequalities fail. Interval expansion and
membership in the original phase box remain separate readouts. A bound on
oscillation alone does not establish that a phase tuple was generated by
UM/IL or that a particular wrapped chart remains valid in the future.

The uniform EPI mode is neutral, so the mean identity is exactly
`mean(x_plus)-mean(x)=mu`. For the earlier reserves
`ell_x=min(x)-B*v` and `U_x=max(x)+B*v`, let `C=B+b=b/(1-rho)`.
The stochastic Euler matrix and `||L_rw*z_plus||_infinity<=v_plus` give

```text
ell_x_plus >= ell_x - [C*epsilon-min(delta)],
U_x_plus   <= U_x   + [C*epsilon+max(delta)].
```

The signed lower/upper loss bounds can be negative under favorable common
drift. Equivalently, separate `mu` and use nonnegative centered costs
`C*epsilon-min(delta_c)` and `C*epsilon+max(delta_c)`. Both signed losses
are bounded above by the conservative absolute cost
`C*epsilon+||delta||_infinity`. This retains cancellation in the actual
endpoint defect instead of summing the magnitudes of its constituent errors.

For a nonempty adjacent finite endpoint chain, summing these signed losses
gives lower/upper reserve bounds at every recorded prefix. The phase envelope
obeys `v_bound_next=rho*v_bound+epsilon`, and the EPI mean follows the exact
signed prefix `mean(x_n)=mean(x_0)+sum_(j<n)mu_j`. The implementation rebuilds
all public observation caches and verifies endpoint adjacency; this supplies
a detached finite arithmetic telescope, not a new causal execution seal.

## 14. A conditional invariant bound with persistent numerical defects

The finite reserve cost need not be summable over an unbounded sequence.
Use the existing EPI diffusion to control persistent spatial error separately
from the neutral mean. For `0<s<1`, the first row of `T^2` is

```text
(a0,a1,a2,0,a2,a1),
a0=(1-s)^2+s^2/2,        a1=s*(1-s),        a2=s^2/4.
```

Its optimal oscillation overlap is
`k=min(s^2,4*s*(1-s))>0`, hence `osc(T^2*x)<=(1-k)*osc(x)`.
Opposite rows attain the minimum overlap `4*min(a1,a2)`; the other distinct
row shifts have overlaps `2*a1+2*min(a1,a2)` and `a1+3*a2`, both no smaller.
The coefficient changes branch at `s=4/5`. This is a two-step matrix identity;
it does not require executing two additional runtime intervals.

Define the translation-invariant range functional

```text
V(x)=osc(x)+osc(T*x),           r=1-k/2 < 1.
```

Since `osc(T*x)<=osc(x)`,
`V(T*x)<=V(x)-k*osc(x)<=r*V(x)`. This gives a rational one-step bound
without treating the phase tangent factor as a nonlinear or EPI gain.

Now **assume independently for every step and prefix being claimed**:

```text
epsilon_n <= epsilon_bar,
osc(delta_n)=osc(delta_c_n) <= E_bar,
abs(sum_(j<n)mu_j) <= M.
```

The last condition concerns signed cumulative mean error. It is not implied
by a uniform per-step error bound. Choose the derived envelopes

```text
D = max(v_0, epsilon_bar/(1-rho)),
F = 2*b*D+E_bar,
R = max(V(x_0), 2*F/(1-r)).
```

The phase recursion preserves `v<=D`. If `D<=1/12`, it stays in the earlier
oscillation class, subject to a consistent winding chart. The complete
additive EPI input `-b*L_rw*z_plus+delta` has oscillation at most `F`.
Subadditivity gives `V(x_plus)<=r*V(x)+2*F`, so `V<=R` is preserved.
The sharp six-coordinate mean/range inequality yields

```text
mean(x_0)-M-5*R/6 <= x_i <= mean(x_0)+M+5*R/6.
```

Requiring this interval inside the declared application band supplies a
conditional joint phase/range/mean guarantee. Persistent nonzero errors may
maintain a nonzero spatial discrepancy; strict contraction of homogeneous
diffusion is not a claim of convergence under those errors. At `s=0` or
`s=1`, `k=0`; this bound rejects nonzero forcing. With zero forcing it
retains the initial range functional and makes no strict contraction claim.

The mean hypothesis is essential. In an abstract additive-error sequence,
take zero phase error, `x_0=(1/2)*1` and `delta_n=-(1/1024)*1`.
Spatial range remains zero, but EPI decreases uniformly: at step 460 it is
`52/1024`, above the default UM floor, and at step 461 it is `51/1024`, below
that floor while still positive and far from hard clipping. This refutes
the sufficiency of uniformly small per-step errors alone. It does not assert
that production arithmetic generates this particular error sequence at a
zero-pressure state.

The executable finite and conditional-uniform bounds extend
[`coupling_winding.py`](../../../../src/tnfr/physics/coupling_winding.py), with shared
joint-reference validation, nodal pressure evaluation and exact matrix powers.
The [finite-budget tests](../../../../tests/physics/test_c6_winding_defects.py) and
[independent uniform-bound controls](../../../../tests/physics/test_c6_winding_uniform_defects.py)
separate exact telescope identities, optimal overlap, persistent error,
neutral drift, excluded Euler boundaries and public-cache revalidation.

## 15. Offline audit of the retained numerical boundary

[`c6_winding_defect_budget.py`](../../../../benchmarks/c6_winding_defect_budget.py)
reads the existing three two-cycle records from section 12. It does not
execute another operator or integrate a longer trajectory. The input file
is bound by SHA-256; its producer manifest and source scope remain historical,
while this analysis receives its own source manifest. The input and output
paths must differ, and both input bytes and analysis source are checked across
the audit.

The adapter rebuilds ordered unit-C6 snapshots and their exact gradient
caches; checks fixed capacity, channel weights, stage EPI preservation,
recorded phase charts, event/flow endpoint continuity and held Euler evidence;
and recomputes the signed pressure/endpoint identities. Stored admission
records remain declarations from that producer, not newly executed checks.
The retained represented-pi chart is explicit. Both null cycles now enter
the additive budget while keeping their failed defect-free status.

The EPI defect is separated exactly into:

```text
phase contribution = h*nu*w_phase*(g_phase_production+L_rw*z_plus),
assembly contribution = h*nu*(kernel_pressure_defect+stored_pressure_residual),
integration contribution = x_after-(x_before+h*nu*stored_pressure).
```

Their signed sum equals `delta`. The assembly term includes rounding in
the combined pressure channels; it is not mislabeled as a pure phase error.
In the null case these components substantially cancel. The observed
two-cycle conservative costs and exact signed mean changes are:

| Preparation | Sum of absolute defect costs | Total EPI mean change |
|-------------|------------------------------|-----------------------|
| Null | `4.347315188882e-17` | `0` |
| Positive k1 | `3.167528703228e-16` | `7.401486830834e-17` |
| Positive k3 | `4.477724788020e-16` | `7.401486830834e-17` |

All finite prefix reserve bounds remain within the declared band. The
retained extrema also produce algebraically admissible examples of the
conditional uniform envelope: approximately `[0.406278,0.593722]` for the
nonzero probes and a very small interval around `0.5` for the null. These
are deliberately labeled illustrations with **future hypotheses unverified**.
Two measured cycles cannot establish uniform error or cumulative-mean bounds.

The [offline report tests](../../../../tests/physics/test_c6_winding_defect_report.py)
check the retained arithmetic and reject inconsistent captures and bindings.
To reproduce the detached report from the retained input:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_defect_budget.py
```

The local nodal arithmetic result below resolves one part of the numerical
task. Pressure realization, phase/backend accuracy and a horizon-independent
signed-mean-prefix bound still require their own declared chart/domain
arguments. Additional samples alone cannot supply them. The exact and finite
results remain available for the separate canonical action-selection and
formation question; no numerical defect is promoted to a new physical source.

## 16. Held nodal rounding cells and the remaining phase/mean boundary

### The actual quarter-flow arithmetic

Keep the retained interval's uniform capacity `nu=1`, duration `h=1/4`,
four equal substeps and zero Gamma contribution. Write `RN` for binary64
rounding to nearest with ties to even. For one represented stored pressure
`p`, the unprojected scalar arithmetic is

```text
q = RN(p/16),
x_(j+1) = RN(x_j+q),       j=0,1,2,3.
```

Capacity multiplication and addition of zero do not change the finite
pressure. The production helper
[`dynamics/_euler_kernel.py`](../../../../src/tnfr/dynamics/_euler_kernel.py) retains
the separate multiplication and addition; replacing them by a fused
multiply-add or one full-interval update would change the numerical map.
The reference in
[`physics/binary64_nodal_flow.py`](../../../../src/tnfr/physics/binary64_nodal_flow.py)
describes these arithmetic operations, not an alternative pressure law.
It does not certify how a live graph produced `p`.

At `x=1/2`, the neighboring represented values are `1/2-2^-54` and
`1/2+2^-53`. The significand of `1/2` is even, so both midpoint ties
belong to its rounding cell:

```text
RN(1/2+q)=1/2  iff  -2^-55 <= q <= 2^-54.
```

For represented finite pressure in this quarter-flow map, this is
equivalent to

```text
-2^-51 <= p <= 2^-50.
```

The endpoints are included. The interval is asymmetric because `1/2`
lies at a binary exponent boundary. A pressure in this interval can be
nonzero while all four stored EPI updates remain exactly `1/2`. The
instantaneous nodal product `nu*p` then remains nonzero: arithmetic stasis
of the represented EPI is not the zero-pressure equilibrium condition.
This statement holds for the fixed held input; it does not make the complete
UM/IL/refresh word stationary.

### A local error bound and a zero-sum counterexample

Assume finite inputs, nearest-even binary64 arithmetic and no clipping.
The initial EPI and every unprojected rounded substep endpoint must lie in
the declared positive band `[ell,U]`, with `0<ell<=U<=1`. Let
`u_min=2^-1074` be the least positive subnormal. Scaling by `1/16` is exact
when representable; otherwise its absolute rounding error is at most
`u_min/2`. Each addition returning a positive value at most one has absolute
rounding error at most `2^-53`, including the wider upper cell at `1`.
Therefore

```text
eta = x_4-(x_0+p/4),
|eta| <= 4*(u_min/2) + 4*2^-53 = 2^-51 + 2^-1073.
```

This is an a priori local bound under the stated arithmetic and endpoint
conditions. It includes subnormal scaling; an execution that clips or
leaves the declared positive band is outside this claim. A platform probe checks
format and selected rounding behavior, but is not a proof of every native
kernel's conformance. Individually captured arithmetic can be compared with
the exact map without promoting those checks to a future execution theorem.

The bound cannot be replaced by zero, even for small normal pressure. For
`x_0=1/2` and `p=2^-50`, each increment is the upper midpoint tie `2^-54`.
All four additions return `1/2`, giving `eta=-2^-52`. One direct full-step
addition instead returns `1/2+2^-52`; the two numerical schedules differ.

For six initial values `x_i=1/2`, take the held pressure tuple
`(+2^-50,-2^-50,+2^-50,-2^-50,+2^-50,-2^-50)`. Its sum is zero. The
positive-pressure coordinates remain at `1/2`, whereas each negative one
ends exactly at `1/2-2^-52`. Consequently the represented EPI mean changes
by `-2^-53`, despite the zero-sum input. This is a numerical-kernel
counterexample to automatic mean preservation, not a claim that canonical
C6 phase refresh generates that pressure tuple. No additional graph
trajectory is required to establish the rounding-cell calculation.

Across `N` otherwise admissible intervals, the absolute local integration
bound supplies only a generic cumulative bound proportional to
`N*(2^-51+2^-1073)`. It does not give a horizon-independent bound on signed
mean error. Pressure and phase errors also remain separate contributions;
the local integration estimate alone cannot discharge section 14's
all-prefix mean hypothesis.

### Retained observations and their provenance

[`c6_winding_rounding_cells.py`](../../../../benchmarks/c6_winding_rounding_cells.py)
audits the same six retained flows as section 15, through its existing
capture, pressure and chronology validation. It compares scalar and NumPy
evaluations of the shared arithmetic helper with those endpoints and
records per-coordinate cells and local error bounds. This is detached
arithmetic replay of stored inputs, not another operator invocation or a
longer research trajectory. The input bytes, producer manifest and SHA-256
remain distinct from the new analysis manifest and output
`artifacts/research/c6_winding_rounding_cells.json`.

The exact stored-EPI mean changes in these retained flows are:

| Preparation | First flow | Second flow |
|-------------|------------|-------------|
| Null | `0` | `0` |
| Positive k1 | `1/(3*2^52)` | `0` |
| Positive k3 | `2^-53` | `-1/(3*2^53)` |

The nonzero stored pressure in the null control is approximately
`-2.14453350193743e-16`, inside the `1/2` pressure cell above. Its stored
EPI remains uniform. Node zero's phase nevertheless changes in both
cycles, and the earlier defect-free phase-transition checks still fail.
Neither the EPI stasis nor the retained mean increments establish a fixed
full state, an eventual numerical attractor or a uniform future error class.

The [rounding-report tests](../../../../tests/physics/test_c6_winding_rounding_report.py)
compare the retained endpoint and defect identities with each of the four
scalar and NumPy arithmetic states. Reproduce the detached audit with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_rounding_cells.py
```

### Phase evaluation has additional, distinct numerical contracts

The phase computations do not all use the same numerical path:

| Readout or update | Current operations |
|-------------------|--------------------|
| UM closed phasor mean | `math.sin/cos`, `math.fsum`, `math.atan2` in the [coupling proposal kernel](../../../../src/tnfr/operators/_coupling_stage_kernel.py) |
| IL open neighborhood mean | `math.sin/cos`, then NumPy mean/arctangent when available; compensated scalar fallback in [trigonometric helpers](../../../../src/tnfr/metrics/trig.py) |
| Refreshed pressure for the retained C6 | NumPy sine/cosine, `np.add.at` neighborhood sums, NumPy arctangent, represented modulo and division by represented pi in the [fused pressure kernel](../../../../src/tnfr/dynamics/fused_dnfr.py) |
| Alternate pressure mean path | Cached trigonometric values, averaged sums and a resultant check in [pressure evaluation](../../../../src/tnfr/dynamics/dnfr.py); separate scalar fallback |
| Scalar phase difference | Another sine/cosine/arctangent evaluation through [unified numerical operations](../../../../src/tnfr/mathematics/unified_numerical.py), including UM proposal and receiver merging |
| Array phase difference | Subtraction, addition of represented pi, remainder modulo represented tau and subtraction of represented pi in [numeric utilities](../../../../src/tnfr/utils/numeric.py) |

The default C6 pressure dispatch uses `compute_fused_gradients_symmetric`.
Its twelve outgoing edge entries stay below the `n_edges>100` JIT threshold,
so the NumPy branch applies. That branch selects nodes with neighbors but
does not replace a zero resultant by the node's original phase. The
resultant fallback in the alternate neighbor-mean path is not part of
these retained C6 evaluations.

The [shared binary64 probe](../../../../src/tnfr/_binary64.py) explicitly does not
certify arbitrary native numerical kernels. Python documents platform C
library dependencies and possible double rounding in `fsum`; it provides
no blanket correctly-rounded sine/cosine/arctangent contract here.
See the [Python math documentation](https://docs.python.org/3.13/library/math.html#math.fsum).
NumPy describes its arctangent through the underlying C implementation,
while reduction precision can depend on the reduction path; see
[NumPy arctangent](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html#notes)
and [NumPy summation](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#notes).
These dependencies do not show that the installed functions are inaccurate.
They prevent inferring a universal ULP allowance from the binary64 format
probe alone.

A conditional propagation bound identifies what a future numerical proof
must supply. If an exact complex phasor sum obeys `|S|>=m>0`, its evaluated
sum has complex error at most `e<m`, and arctangent evaluation has angular
error at most `a`, then its wrapped angle error is at most

```text
e/(m-e)+a.
```

For example, rotate `S` to the positive real axis: the perturbed real part
is at least `m-e`, and the imaginary displacement is at most `e`; use
`atan(u)<=u`. In the prepared `pi/24` box, UM's closed sums permit
`m=3/2`. The open two-neighbor sums instead permit `m=3/4`, since their
resultant is at least `2*cos(3*pi/8)>3/4`; an averaged sum uses `m=3/8`.
The full stage must also account for scalar phase-difference evaluation,
arithmetic, modulo operations and a consistent lifted chart. These error
allowances are numerical proof premises, not new TNFR parameters or forces.

The next proof obligation is therefore narrower: establish useful bounds
for the actual phase/pressure implementation and a separate cumulative-mean
mechanism on a declared represented domain. The held arithmetic theorem,
finite reports and conditional envelopes do not establish nonlinear
formation, autonomous action selection or a global TNFR stability theorem.

## 17. Opposite-pair closure and a restricted mean-preserving arithmetic class

### An exact partial observable of the nonlinear phase map

Use the same ordered C6, winding chamber and simultaneous all-target UM/IL
maps. For normalized phase errors `z=e/pi`, define three centered opposite
pair sums:

```text
(P*z)_i = z_i+z_(i+3)-2*mean(z),       i=0,1,2.
```

Their sum is zero. This observable retains two independent coordinates;
it neither reconstructs the six phases nor fixes their common rotation.
Let `A` be cycle adjacency, `L=I-A/2`, `K=(I+A)/3`, and let `c` contain
the six closed-neighborhood phasor-mean errors in the same normalized chart.
The exact finite phase maps in the established chamber are

```text
z_U=(1-t)*z+t*K*c,
z_plus=M*z_U,           M=(1-alpha)*I+alpha*A/2.
```

Opposite receiver triples partition all six sources. Consequently

```text
P*K=0,
P*M=(1-3*alpha/2)*P,
P*L=(3/2)*P.
```

These are exact matrix identities, independent of the nonlinear function
that produced `c`. They give the finite, nonlinear observable law

```text
h_phi_plus=lambda_pair*h_phi,
lambda_pair=(1-t)*(1-3*alpha/2),      h_phi=P*z.
```

The multiplier equals the earlier `k=2` tangent coefficient, but its use
here has a separate finite-map proof. The full tangent energy bound is
not promoted to a nonlinear bound. For `0<t<=1` and `0<=alpha<=1`,
`abs(lambda_pair)<1`; opposite-pair error decays geometrically for exact
repetition in this chamber. The subspace `h_phi=0` is preserved.

This is pairing **about the current mean**. Indeed
`mean(z_U)=(1-t)*mean(z)+t*mean(c)`, so a nonlinear common rotation can
change while centered pairing remains exact. Literal `z_(i+3)=-z_i`
would additionally require the mean to remain zero. Geometric reflection
with phase conjugation is a different symmetry and is not inferred here.

The executable
[`observe_c6_winding_pairing`](../../../../src/tnfr/physics/coupling_winding.py)
rebuilds the existing joint reference and applies these identities to
supplied closed-mean coordinates. Their membership in the initial phase
interval is a necessary chamber property, not authentication as actual
phasor means. It keeps the previously derived oscillation-contraction and
reserve checks separate. The observer does not call trigonometric kernels,
admit an operator word or identify an executed graph.

### The nodal equation closes the corresponding EPI observable

For `h_x=P*x`, `s=h*nu*w_epi` and `b=h*nu*w_phase`, the same exact held
nodal map gives

```text
phase-pressure opposite sums = -(3/2)*h_phi_plus,
total-pressure opposite sums = -(3/2)*(w_epi*h_x+w_phase*h_phi_plus),
h_x_plus=(1-3*s/2)*h_x-(3*b/2)*h_phi_plus.
```

Thus each of the three dependent pair coordinates follows one common
two-component recurrence:

```text
                  [ lambda_pair                 0        ]
Q_pair =          [ -(3*b/2)*lambda_pair        1-3*s/2    ].
```

This closes a partial observable of the phase-first nonlinear model.
Both sectors retain their zero-sum relation over the three pair indices.
For `0<s<=1`, both diagonal eigenvalues have modulus below one, so these
pair defects tend to zero, including the coincident-eigenvalue case of a
triangular matrix. At `s=0`, EPI pairing can remain unchanged. This result
does not control the opposite-odd spatial modes or the common phase mean;
in particular it does not remove the alternating EPI boundary in section 11.

When both initial pair defects vanish, total pressures are opposite in
each pair. The real-valued nodal evolution then preserves `x_i+x_(i+3)`.
The arithmetic counterexamples in section 16 explain why this real-model
identity alone does not preserve represented pair sums.

### A closed binary64 class for the isolated EPI channel

There is a useful restricted arithmetic class, without choosing a new
physical coefficient. Keep unit capacity, zero Gamma, fixed unit C6,
four held substeps of `1/16`, and a fixed represented `0<=w_epi<=1`.
Pressure is refreshed from the EPI channel before each quarter interval.
The class is specified by the represented state:

- All six EPI values lie in the positive application band `[float(0.05),1]`
  and strictly inside one common normal binade `(B,2*B)`.
- Its spacing is `Delta=B*2^-52`; the exact pair sums are all `2*c`.
- The center lies on that lattice: `c/Delta` is an integer.

`B`, `Delta` and `c` are derived from the input representation. They are
arithmetic premises, not TNFR forces or adjustable model parameters.

On this domain, subtracting two EPI values and averaging the two neighbor
differences are exact. The production scalar and vector
[difference reducers](../../../../src/tnfr/mathematics/_neighbor_differences.py)
therefore both materialize

```text
g_i=(x_(i-1)+x_(i+1))/2-x_i,
p_i=RN(w_epi*g_i).
```

The statement covers their mixed-sign exact fallback too. The observer
checks both reducers against the rational pressure and its final rounding.
Opposite-pair reflection commutes with cycle adjacency, so
`g_(i+3)=-g_i`. Nearest-even rounding is sign-odd numerically; hence the
represented pressures and their scaled increments remain opposite.

First, the initial EPI interval `[m,M]` is preserved. For nonnegative
pressure, write `M-x_i=n*Delta`. Monotonic rounding gives
`0<=q_i=RN(p_i/16)<=(M-x_i)/16`; the upper bound is representable on this
positive band. A rounded lattice update advances at most
`floor(n/16+1/2)` spacings. For every integer `n>=0`,

```text
4*floor(n/16+1/2)<=n.
```

For `n<=7` the left side is zero; for `n>=8` it is at most `n/4+2<=n`.
An induction bounds all four partial updates inside the original interval.
Negative pressure has the corresponding lower-bound argument. Thus every
result cell stays strictly inside the same binade.

Second, reflection `x -> 2*c-x` preserves this uniform lattice and its
even-significand tie decisions because `c/Delta` is integral. The reflected
addition cells from section 16 coincide, so each opposite pair keeps sum
`2*c` at every substep. The common mean stays exactly `c`. The interval,
lattice-center and pressure-pair premises therefore hold again after
refreshing this **pure-channel** map. This proves forward invariance and
zero signed mean drift under its arbitrary finite repetition. It permits
nonuniform rounding plateaus; it is not a binary64 consensus theorem.

The executable
[`observe_binary64_paired_c6_diffusion`](../../../../src/tnfr/physics/binary64_nodal_flow.py)
reuses the B18 flow and its exact cells. It checks the declared class and
one supplied scalar/vector kernel application, keeping repeated pure-channel
scope separate from a live executor or a complete canonical UM/IL word.
The [exact pairing tests](../../../../tests/physics/test_c6_winding_pairing.py) and
[binary64 class tests](../../../../tests/physics/test_binary64_paired_diffusion.py)
exercise the identities, class boundaries and arithmetic obstructions.

Two exclusions are substantive. At a half-lattice center, even ties can
break pair sums: `x=(0.75,0.75+2^-53)` and opposite pressures `+/-2^-50`
make both first updates equal `0.75`. At the inherited center `0.5`, the
two sides have different spacings; there is no nontrivial symmetric
common-binade neighborhood. The earlier quarter-flow counterexample
continues to apply. Moreover, phase forcing can move uniform EPI outside
its initial min/max interval. The pure-channel interval proof does not
replace the joint phase-source budget.

### What the retained default execution actually preserves

The detached
[`c6_winding_pairing.py`](../../../../benchmarks/c6_winding_pairing.py)
audits the same B16 records through the B18 continuity and arithmetic
validation. It computes centered phase pairs, EPI pairs and pressure pairs
at the initial, raw-operator, refreshed and flow boundaries. It does not
execute another graph, change factors or extend the horizon.

Let `sigma=g_phase_production+L*z` be the phase-realization defect. At any
retained capture the pressure-pair accounting is

```text
p_i+p_(i+3) = -(3/2)*w_epi*(P*x)_i
              -(3/2)*w_phase*(P*z)_i
              +w_phase*(sigma_i+sigma_(i+3))
              +kernel_assembly_defect_i+kernel_assembly_defect_(i+3)
              +stored_pressure_residual_i+stored_pressure_residual_(i+3).
```

The last term matters at raw UM/IL boundaries, where stored pressure need
not equal a refreshed kernel. This identity separates loss of geometric
pairing from pressure arithmetic and from the operator's stored-pressure
writes. All represented-pi chart assumptions remain explicit.

All three preparations have opposite centered errors relative to the
stored winding base, yet their initial represented pressures do not pair.
For example the base has `theta_1+theta_2-Fraction(math.pi)=-2^-52`.
This is a defect of that represented reflection identity, not a claim
about an exact physical conjugation or a universal impossibility result.
The first UM produces nonzero centered pair residuals in all three cases,
so the discrepancy cannot be removed by subtracting a common rotation.

One existing boundary is especially discriminating: after the first UM
refresh in the k3 record, all opposite pressure sums are exactly zero.
After the following IL and pressure refresh,

```text
g_phase[0]+g_phase[3] = -5215/2^65,
p[0]+p[3] = -15823/2^67.
```

Thus the recorded B16 IL/refresh does not preserve that observed paired
pressure class. The nonzero preparations' EPI endpoints also straddle
the `0.5` binade boundary, outside the restricted arithmetic theorem.
Historical producer declarations and input hashes remain distinct from
this analysis's source manifest and derived output.

Reproduce this detached stage audit from the retained B16 input with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pairing.py
```

The next joint-runtime obligation is now specific: a preserved relational
contract for total pressure pairs and compatible EPI rounding cells, or
an explicit summable budget for their leakage. A bound on each phase error
separately still does not give a horizon-independent signed-mean bound.
The exact partial quotient and restricted pure-channel class are useful
positive results; complete winding-runtime preservation, canonical
formation and laboratory correspondence remain open.
