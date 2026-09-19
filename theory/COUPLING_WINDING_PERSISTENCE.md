# Canonical Coupling preserves winding in a restricted cycle regime

**Status: retained conditional derivations and finite evidence; C6 campaign
parked.** The B75 boundary remains 41 of 56 first-exit labels excluded and 15
open; global C6 stability is not proved. Dated priorities below record earlier
research steps. Only the [execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md)
owns current work; the [mechanism audit](C6_RESEARCH_MECHANISM_AUDIT.md) records
verification fixes and supersession. Mathematical claims retain their local
hypotheses and source-bound evidence.

**Status:** Exact target-only UM gap transport; default all-target C6 local
phase response, nonlinear contraction, joint nodal reserves and additive
defect bounds; local binary64 nodal rounding cells, finite production checks
and a canonical Transition loss witness.
**Research links:** B2.d/O3.a, S3, S8, S9 and S16.
**Scope:** Fixed simple undirected cycles with explicit phase coordinates.

## 1. The structural state and the actual operator realization

Let `phi_i` be the phase at node `i` of an oriented cycle `C_n`, `n>=3`, and
let its oriented shortest-arc gaps be

$$
d_i=\operatorname{wrap}(\phi_{i+1}-\phi_i),\qquad
-\pi<d_i<\pi,\qquad
\sum_i d_i=2\pi W.
$$

Indices are cyclic and `W` is the integer winding of this declared cycle.
The final identity follows from a consistent closed phase loop, not from
arbitrarily supplied real gap coordinates. Reversing orientation changes its
sign. Retaining a positive wrap-branch margin makes the observation defined;
it does not by itself prove that a future operation preserves that margin.

The following theorem uses the existing canonical Coupling implementation
in [_coupling_stage_kernel.py](../src/tnfr/operators/_coupling_stage_kernel.py),
with `UM_BIDIRECTIONAL=False` and `UM_FUNCTIONAL_LINKS=False`. The first
configuration makes each target average only its neighbors and write only
its own phase. The second keeps the simple cycle support fixed. Other
preconditions and grammar admission remain necessary at execution time.

Let `g` be the effective UM gate, including any tightening of the hard U3
limit, and suppose every gap lies in one interval

$$
-g<m\leq d_i\leq M<g,\qquad 0<g\leq\pi/2.
$$

The interval may be positive, negative or contain zero. Both cycle neighbors
are therefore compatible with every target. Their separation in the local
lift is `d_(i-1)+d_i`, whose absolute value is strictly less than `pi`.
The neighbor phasor sum is nonzero, so the circular midpoint is unambiguous.
The UM phase factor `eta=UM_theta_push` is the already configured operator
coefficient, with `0<eta<=1`. Its repository default is `1/(pi+1)` as
materialized by `UM_THETA_PUSH`. The theorem covers the declared factor
interval; it does not derive that particular default uniquely from the nodal
equation or add an independent phase-evolution law.

The default bidirectional UM configuration has a different effect: it includes
the target in the phasor mean and writes neighboring phases too. It is outside
this theorem. Even on a uniform twisted cycle, one bidirectional target shrinks
the two inner gaps by `1-eta` and expands the adjacent outer gaps by `1+eta`.
The interval invariance proved below cannot simply be transferred to it.

## 2. The midpoint identity and its canonical pressure connection

In the target's local lift, its neighbor phases are
`phi_i-d_(i-1)` and `phi_i+d_i`. Their phasor sum is

$$
2\cos\frac{d_{i-1}+d_i}{2}\;
\exp\left(i\left[\phi_i+\frac{d_i-d_{i-1}}2\right]\right).
$$

The cosine is positive under the stated strict gate. Consequently the actual
target-only UM shortest-arc update has displacement

$$
u_i=\frac\eta2(d_i-d_{i-1}).
$$

This identity connects directly to the existing canonical phase-pressure
channel in [dnfr.py](../src/tnfr/dynamics/dnfr.py). On this same simple cycle,
its unweighted neighbor-phasor mean is the midpoint above, hence

$$
\partial\phi_i
=-\frac1\pi\operatorname{wrap}(\phi_i-\bar\phi_i)
=\frac{d_i-d_{i-1}}{2\pi},\qquad
u_i=\eta\pi\,\partial\phi_i.
$$

The configured phase-channel weight belongs to the aggregation of
`DeltaNFR`; it is separate from the unweighted channel identity here. The
nodal integrator subsequently uses `nu_f*DeltaNFR` to evolve EPI. Reading the
same structural gradient in UM does not identify a phase jump with that EPI
evolution or introduce a second continuous-time law. The identity excludes
vanishing resultants, arbitrary neighborhood profiles and wrap-crossing
configurations. Binary64 phasor evaluation retains its separate residual.

## 3. One target: conserved circulation and interval protection

Updating target `i` changes just its incident gaps. Put
`a=d_(i-1)`, `b=d_i`. Then

$$
\begin{pmatrix}a'\\b'\end{pmatrix}
=\begin{pmatrix}1-\eta/2&\eta/2\\\eta/2&1-\eta/2\end{pmatrix}
\begin{pmatrix}a\\b\end{pmatrix}.
$$

Each new gap is a convex combination of the old pair, and their sum is
unchanged. Every gap remains in `[m,M]`, strictly inside U3 and away from the
wrap branch. The same is true along the affine shortest-arc completion of
the declared phase jump. Thus the closed cycle retains `W`. This completion
is a mathematical path witness; the engine event itself is a discrete update.

For the gap mean `d_bar=sum(d)/n`, define the diagnostic squared spread
`V(d)=sum((d_i-d_bar)^2)/2`. Direct expansion gives

$$
V(d)-V(d')=\frac{\eta(2-\eta)}4(a-b)^2\geq0.
$$

Every finite sequence of these admitted target-only updates preserves the
same interval and winding, even if its target order varies. Arbitrary target
selection does not imply convergence to uniform gaps: a sequence can keep
updating an already equal pair while leaving other differences untouched.

## 4. A simultaneous stage is the canonical cycle diffusion map

An immutable all-target UM stage proposes the displacement in section 2 for
every node. Since target-only UM has no overlapping neighbor writes, the
new edge gaps are

$$
d_i'=d_i+u_{i+1}-u_i
=(1-\eta)d_i+\frac\eta2(d_{i-1}+d_{i+1}).
$$

Equivalently,

$$
d'=(I-\eta L_{\rm rw,C_n})d.
$$

This is precisely the normalized cycle Laplacian already used by the canonical
pure-EPI diffusion channel. Here it acts on an oriented gap observation under
a UM stage; the index counts operator stages, not elapsed physical time.
Its nonnegative, doubly stochastic matrix preserves the gap sum, the initial
gap interval, U3 compatibility and winding in the same exact model.

The exact nonnegative spread drop is

$$
V(d)-V(d')=
\frac{\eta(1-\eta)}2\sum_i(d_{i+1}-d_i)^2
+\frac{\eta^2}{8}\sum_i(d_{i+1}-d_{i-1})^2.
$$

The Fourier eigenvalues are
`1-eta+eta*cos(2*pi*k/n)`. For fixed `0<eta<1`, their absolute values are
strictly below one for all `k!=0`, so repeated exact all-target stages converge
to `d_bar=2*pi*W/n`. At `eta=1`, an even cycle has an alternating gap mode with
eigenvalue `-1`; its spread is preserved, although winding and interval
protection still hold. An odd cycle has no such nonconstant mode. These are
exact-model statements, not asymptotic binary64 execution certificates.

For same-sign gaps and nonzero total `S=sum(d)=2*pi*W`, circulation concentration
can be read without another dynamical parameter:

$$
I(d)=\sum_i\left(\frac{d_i}{S}\right)^2
=\frac1n+\frac{2V(d)}{S^2}.
$$

It cannot increase under the protected maps. Under the strictly contracting
all-target regime it tends to `1/n`. The preserved winding therefore becomes
an evenly distributed twist; this class supplies no mechanism for sustained
localization of circulation. It neither constructs a localized physical
entity nor supplies a restoring law for localized EPI or capacity support.

## 5. Nonzero winding can coexist with zero canonical pressure

A uniform twist `phi_i=2*pi*W*i/n` with `|2*pi*W/n|<g` is fixed by target-only
UM. Its phase-pressure channel vanishes because its two neighbor phasors
have their mean at the target phase. With uniform EPI, uniform positive
capacity and the uniform degree of the simple cycle, the other three
canonical gradient channels also vanish. Therefore the complete canonical
`DeltaNFR` is zero in this exact restricted state, despite nonzero `W`.

Pressure equilibrium thus need not mean phase consensus. A positive-winding
example with the strict canonical gate requires `n>4*W`. The spectral
convergence above concerns uniformity of gaps, not uniformity of phases.
It supplies neither attraction for arbitrary multichannel trajectories nor
a dynamic preparation of nonzero winding from a zero-winding state. The
protected UM class cannot perform that preparation because it conserves `W`.

## 6. Canonical loss, branch crossings and observation limits

Outside the protected class, canonical events can change winding. For the
regular unit-winding eight-cycle, change only the phase at node zero from its
initial value zero to `theta`. The edge from node seven to node zero reaches
the wrap branch when `theta=3*pi/4`. Immediately below this value the winding
is one; immediately above it, before the next incident branch crossing, it
is zero. The cycle support can remain unchanged throughout.

The production Transition (`NAV`) word supplies a finite instance. On the
prepared regular eight-cycle with repository default regime policy and seed
17, repeated admitted single-Transition words give `theta_0=2.35` after 13
steps and `theta_0=2.5500000000000003` after 14. Their winding observations are
respectively one and zero. The reported minimum branch margins are
approximately `0.006194490192344748` and `0.19380550980765499`. The actual
EPI, capacity and pressure effects of Transition remain part of those events;
the example does not impose an auxiliary phase trajectory. Its endpoint
change requires a branch crossing in any continuous fixed-cycle completion,
but does not certify an unobserved continuous path inside a discrete event.

U3 admission applies to the operator's required compatible relations; it is
not a universal promise that every cycle edge stays admissible under every
canonical operator. Equal endpoint winding likewise cannot establish a
protected intervening path: opposite edge slips can cancel. Neither a
phase-independent EPI operation nor a static snapshot is a proof about all
future phase writes. Cycle deletion, changed support, absent phase values,
branch hits and nonzero signed branch slips must remain distinguishable.

## 7. Exact companion, runtime checks and remaining work

[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py) implements the
single-target and all-target exact gap maps with conserved sum, interval and
spread identities. It reuses the shared exact-or-represented real reader:
integer and rational inputs remain exact, including NumPy integers promoted
to Python integers; other real values retain their binary64 rational value.
The observer accepts signed gap coordinates and does not round their sum
into a purported winding integer. A separate declared-cycle phase observation
must establish closure and winding. Its factor, state and gate hypotheses
also do not admit a live operator word by themselves.

[Exact tests](../tests/physics/test_coupling_winding.py) cover convex transport,
Jensen dissipation, finite compositions, the even-cycle boundary and the
shared pure-EPI pressure map.
[Pressure tests](../tests/physics/test_coupling_pressure_bridge.py) compare the
actual scalar and vectorized canonical phase channel with the midpoint
identity, then use the shared nodal integrator for EPI evolution. Their
regular twisted controls bound observed binary64 pressure residuals; a
selected finite tolerance is not a rigorous bound on every libm evaluation.

[Runtime tests](../tests/physics/test_coupling_winding_runtime.py) and the
[benchmark](../benchmarks/canonical_winding_persistence.py) compare actual
canonical UM histories with the exact gap companion and record endpoint
residuals, branch/U3 margins, winding and actual operator histories. They
use eight UM/SHA pairs on perturbed C8 and C16 with W=0 and W=1. SHA leaves
phase unchanged and attenuates capacity by its existing default factor;
the word uses no phase-factor override. Both sequence validators and live
operator requirements remain active. Separate DERIVED and MEASURED manifests
record the working-source digest and the corresponding evidence boundary.
The word observer now uses live primary-EPI admission, shared adjacency
validation, runtime U3 limits and node/edge identity comparisons; its
[integrity tests](../tests/physics/test_winding_word_integrity.py) cover these
boundaries without changing committed-prefix behavior after later failure.
The production controls
also exercise canonical winding loss through Transition. These finite
checks are separate from the exact repeated-map theorem and do not prove
future binary64 protection, arbitrary grammar stability or a physical
memory experiment.

The [capacity-localization study](CAPACITY_LOCALIZATION_BALANCE.md) now tests
physical nodal flow in addition to the discrete phase words above. It finds
EPI diffusion under uniform capacity and a conditional nonuniform equilibrium
when a retained capacity profile balances that diffusion. The UM/SHA word
above has no physical EPI-flow interval, so retention of an EPI shape under
that word alone could not establish resistance to diffusion. The remaining
question is joint maintenance when canonical operators evolve the supporting
phase and capacity fields. That requires their complete effects and cannot
be inferred from integer winding alone.

## 8. One shared local response for circular means

B2.d.15 compares the previous
[antipodal obstruction](CHILD_COUPLING_FEEDBACK.md#18-antipodal-phase-response-under-simultaneous-um-and-il)
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

[`phase_response.py`](../src/tnfr/physics/phase_response.py) is the shared exact
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

[`c6_winding_phase_response.py`](../benchmarks/c6_winding_phase_response.py)
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
[`certify_phase_winding`](../src/tnfr/physics/winding_certificates.py) observes
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
[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py) and live in
[`test_c6_winding_phase_response.py`](../tests/physics/test_c6_winding_phase_response.py).
[`test_phase_response.py`](../tests/physics/test_phase_response.py) supplies
independent signed-response, Gram and receiver-average controls. The finite
[`runtime tests`](../tests/physics/test_c6_winding_phase_response_runtime.py)
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
[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py), reusing the
cycle Laplacian and exact input readers. The reference computes `rho`, `B`,
the Euler matrix and the separate EPI mode bound. The observer receives six
declared pre/post phase-error coordinates in pi units and the EPI state. It
checks interval nesting and contraction, evaluates the exact nodal endpoint
and reports the input/output reserves. Public reference caches are rebuilt.
Supplying two vectors that pass these checks does not authenticate a
production phase update, its phase chart or its operator admission.

## 12. Two-cycle admission and the represented null boundary

[`c6_winding_joint_domain.py`](../benchmarks/c6_winding_joint_domain.py)
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

The [exact-domain tests](../tests/physics/test_c6_winding_joint_domain.py),
[independent boundary tests](../tests/physics/test_c6_winding_joint_domain_adversarial.py)
and [runtime tests](../tests/physics/test_c6_winding_joint_domain_runtime.py)
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
[`coupling_winding.py`](../src/tnfr/physics/coupling_winding.py), with shared
joint-reference validation, nodal pressure evaluation and exact matrix powers.
The [finite-budget tests](../tests/physics/test_c6_winding_defects.py) and
[independent uniform-bound controls](../tests/physics/test_c6_winding_uniform_defects.py)
separate exact telescope identities, optimal overlap, persistent error,
neutral drift, excluded Euler boundaries and public-cache revalidation.

## 15. Offline audit of the retained numerical boundary

[`c6_winding_defect_budget.py`](../benchmarks/c6_winding_defect_budget.py)
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

The [offline report tests](../tests/physics/test_c6_winding_defect_report.py)
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
[`dynamics/_euler_kernel.py`](../src/tnfr/dynamics/_euler_kernel.py) retains
the separate multiplication and addition; replacing them by a fused
multiply-add or one full-interval update would change the numerical map.
The reference in
[`physics/binary64_nodal_flow.py`](../src/tnfr/physics/binary64_nodal_flow.py)
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

[`c6_winding_rounding_cells.py`](../benchmarks/c6_winding_rounding_cells.py)
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

The [rounding-report tests](../tests/physics/test_c6_winding_rounding_report.py)
compare the retained endpoint and defect identities with each of the four
scalar and NumPy arithmetic states. Reproduce the detached audit with:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_rounding_cells.py
```

### Phase evaluation has additional, distinct numerical contracts

The phase computations do not all use the same numerical path:

| Readout or update | Current operations |
|-------------------|--------------------|
| UM closed phasor mean | `math.sin/cos`, `math.fsum`, `math.atan2` in the [coupling proposal kernel](../src/tnfr/operators/_coupling_stage_kernel.py) |
| IL open neighborhood mean | `math.sin/cos`, then NumPy mean/arctangent when available; compensated scalar fallback in [trigonometric helpers](../src/tnfr/metrics/trig.py) |
| Refreshed pressure for the retained C6 | NumPy sine/cosine, `np.add.at` neighborhood sums, NumPy arctangent, represented modulo and division by represented pi in the [fused pressure kernel](../src/tnfr/dynamics/fused_dnfr.py) |
| Alternate pressure mean path | Cached trigonometric values, averaged sums and a resultant check in [pressure evaluation](../src/tnfr/dynamics/dnfr.py); separate scalar fallback |
| Scalar phase difference | Another sine/cosine/arctangent evaluation through [unified numerical operations](../src/tnfr/mathematics/unified_numerical.py), including UM proposal and receiver merging |
| Array phase difference | Subtraction, addition of represented pi, remainder modulo represented tau and subtraction of represented pi in [numeric utilities](../src/tnfr/utils/numeric.py) |

The default C6 pressure dispatch uses `compute_fused_gradients_symmetric`.
Its twelve outgoing edge entries stay below the `n_edges>100` JIT threshold,
so the NumPy branch applies. That branch selects nodes with neighbors but
does not replace a zero resultant by the node's original phase. The
resultant fallback in the alternate neighbor-mean path is not part of
these retained C6 evaluations.

The [shared binary64 probe](../src/tnfr/_binary64.py) explicitly does not
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
[`observe_c6_winding_pairing`](../src/tnfr/physics/coupling_winding.py)
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
[difference reducers](../src/tnfr/mathematics/_neighbor_differences.py)
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
[`observe_binary64_paired_c6_diffusion`](../src/tnfr/physics/binary64_nodal_flow.py)
reuses the B18 flow and its exact cells. It checks the declared class and
one supplied scalar/vector kernel application, keeping repeated pure-channel
scope separate from a live executor or a complete canonical UM/IL word.
The [exact pairing tests](../tests/physics/test_c6_winding_pairing.py) and
[binary64 class tests](../tests/physics/test_binary64_paired_diffusion.py)
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
[`c6_winding_pairing.py`](../benchmarks/c6_winding_pairing.py)
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

## 18. Certified two-neighbor phase realization

The retained IL/refresh failure includes a concrete error in evaluating the
canonical phase geometry. At node 3 of the first k3 IL, the two neighbor
phases have exact rational midpoint `m=14149309535295891/2^52`. Their
separation is less than mathematical pi, so this is the circular mean on
the selected sheet. The old independent proposal, matched to the recorded
endpoint, reports `m-2^-52`: the odd lower float at a tie. Correct
nearest-even rounding selects its successor. Rounding the mean correctly
and then subtracting the center would still introduce an avoidable
intermediate rounding. The displacement needs its own evaluation.

### One geometric kernel for IL and phase pressure

[`certified_two_neighbor_phase`](../src/tnfr/mathematics/_phase_midpoint.py)
treats each input binary64 angle as an exact real number. For center `theta`
and neighbors `a,b`, it certifies separate lifts

```text
d_a=a-theta+2*k_a*pi,  d_b=b-theta+2*k_b*pi,
-pi/2 < d_a,d_b < pi/2.
```

These conditions imply `|d_a-d_b|<pi` and a nonzero phasor resultant, since

```text
exp(i*d_a)+exp(i*d_b)
  = 2*cos((d_a-d_b)/2)*exp(i*(d_a+d_b)/2).
```

Consequently the exact direction relative to the center is
`d=(a+b)/2-theta+(k_a+k_b)*pi`. This is a restricted evaluation identity
of the canonical unweighted phase channel, not a replacement by linear
diffusion outside its chamber. The strict half-pi condition selects a
sufficient numerical branch; it does not change IL admission or U3 policy.

The kernel encloses mathematical pi using
`pi=16*atan(1/5)-4*atan(1/239)`, alternating rational series and their first
omitted terms. It outward-rounds those bounds to a dyadic grid. The 64-term
work limit and 256-bit grid are numerical certification choices, not new
TNFR parameters. Branch decisions must follow from the enclosure. Equal
nearest-even rounded endpoints certify a unique final float; rational
branches, including ties, use one exact rational conversion. Unsupported
format/rounding premises or undecided branches/rounding return to the
existing phasor path. The IEEE probes do not certify external libm routines.

Inputs must be finite Python floats in `[0,math.tau)`. The returned
displacement and canonical mean are rounded independently. The real mean
is normalized with mathematical `2*pi`; its rounded display may equal
`math.tau`. Consumers must not reconstruct the displacement from that
display. IL retains its existing center normalization and final phase-write
modulo. Pressure reads the stored center. Thus a stored `math.tau` can
remain outside the pressure specialization, and phase-write error is a
separate obligation.

Direct and simultaneous IL consume the same certified displacement and
share one phase-event formatter, which records the selected method. Fused,
scalar, vector fallback and eligible JIT pressure rows use `RN(d)/math.pi`
with the existing final division. The phase channel still counts support
neighbors, including zero-conductance edges; fused array calls respect
their actual contribution multiplicities and orientation. EPI conductance
and the other nodal channels are unchanged. Eligible JIT rows are assembled
from the shared channel values rather than corrected by subtracting an
already rounded phasor pressure. Other rows retain their prior semantics.

### A local cancellation bound, with explicit remaining errors

If every vertex of unit undirected C6 is eligible, its exact midpoint
displacement is `d_i=(1/2)*sum_j wrap_true(theta_j-theta_i)`. Opposite
oriented edge terms cancel, so `sum_i d_i=0`. This follows from the phase
geometry without a numerical pressure projection. Write

```text
r_i=RN(d_i),  P=Fraction(math.pi),  g_i=RN(r_i/P),
A(y)=Fraction(math.ulp(y))/2.
```

The nearest-even cells give

```text
|g_i-d_i/P| <= A(r_i)/P+A(g_i),
|mean(g)| <= sum_i[A(r_i)/P+A(g_i)]/6.
```

This includes binade boundaries, signed zero and subnormals; half an ulp
must be formed rationally to retain `2^-1075`. Comparing individual values
with `d_i/pi` adds a scale term, but its mean is zero because `sum_i d_i=0`.
The bound concerns this phase-gradient division only. Channel weighting,
assembly, UM, IL phase writes and Euler updates have separate errors. A
state-dependent local bound is not a summable or horizon-independent
signed mean budget.

Indeed, small pair leakage alone cannot supply that budget. Set
`Delta=2^-53`, start the held B18 primitive at `(0.75,0.75)` and take
`p_plus=2^-50+2^-102`, `p_minus=-2^-50+2^-103`. Their sum is only
`3*2^-103`, yet each `1/16` substep advances the first EPI by `Delta`
and freezes the second, without ties. Four substeps raise their mean by
`2^-52`; the ideal mean-pressure contribution is `3*2^-106`. This is a
numeric held-input obstruction, not evidence that the refreshed TNFR word
generates those pressures or drifts without bound.

### Fixed finite comparison

[`c6_winding_phase_kernel.py`](../benchmarks/c6_winding_phase_kernel.py)
keeps the B16 artifact unchanged, validates it through the existing defect
and Euler-cell owners, and evaluates the new kernel on its old IL inputs.
Of 36 eligible historical tuples, 33 selected displacements and 11 displayed
means differ from the correctly rounded midpoint results. These are retained
independent proposals bound to captured endpoints, not retroactively sealed
executor proposals.

A separate current execution repeats exactly the three null/k1/k3
preparations, seed 17, unchanged controls and coefficients, two `h=.25`
UM/IL/flow cycles and terminal SHA. All 36 current IL tuples and eligible
source pressure gradients agree with the common kernel. The measured
post-IL pressure means change from approximately `1e-17` in the old records
to `1e-33` for the null and at most `5.65e-22` for these nonzero controls.
The final EPI mean changes are:

| Control | Historical two-cycle change | Current two-cycle change |
|---------|-----------------------------|--------------------------|
| null | 0 | 0 |
| k1 | `1/(3*2^52)` | `1/(3*2^52)` |
| k3 | `1/(3*2^52)` | 0 |

The new arithmetic corrects an identified phase-evaluation error; it does
not eliminate all signed EPI drift. Every interval retains separate
pressure and integrator mean contributions. The original captures and
producer manifest remain historical; the comparison records a distinct
source digest and current execution. Reproduce it without extending the
declared horizon:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_phase_kernel.py
```

The next discriminator is whether the corrected phase, write and pressure
maps preserve a joint domain compatible with the EPI addition cells, or
admit a summable signed leakage estimate. A general libm guarantee,
complete-runtime invariant class, autonomous formation and empirical
correspondence are still not established.

## 19. Pressure boxes and the two-binade mean-bias law

Correct phase evaluation does not imply exact preservation of the represented
EPI mean. The remaining k1 discrepancy can now be isolated to the held nodal
arithmetic without another graph experiment. The result is an interval
theorem: a complete set of represented pressures preserves each recorded
four-substep EPI trace, including its mean bias.

### Inverting the shared nodal arithmetic

Keep the B18 premises: positive unclipped EPI in a declared band inside
`(0,1]`, unit capacity, zero Gamma, and four held Euler steps of `1/16`.
For one coordinate write `q=RN(p/16)` and `x_(j+1)=RN(x_j+q)`.
Let `C(y)` denote the exact nearest-even cell of a represented number `y`,
including its midpoint boundaries precisely when its significand is even.
For a fixed recorded trace `x_0,...,x_4`, the admissible increments satisfy

```text
q in J = intersection_(j=0,...,3) [C(x_(j+1))-x_j].
```

This is necessary and sufficient by induction over the four additions.
The open/closed flags of every binding interval boundary must be retained.
Let `q_min,q_max` be the first and last binary64 numbers in `J`. The union
of rounding cells between them is a contiguous real interval. Therefore
the complete real preimage under the scaling operation has boundaries

```text
16*lower(C(q_min)), 16*upper(C(q_max)),
```

with inclusion decided by the parity of `q_min,q_max`, independently of
the EPI result-cell parity. Selecting the first and last represented
pressures in this interval gives the exact maximal binary64 pressure
interval for the trace. Every selected increment is reachable: these
positive unit-band traces bound its magnitude below two, so `16*q` is a
finite exactly represented pressure. No pressure samples or tolerance
search establish this maximality; it follows from monotonic rounding and
the exact cells.

The constructor
[`derive_binary64_quarter_pressure_box`](../src/tnfr/physics/binary64_nodal_flow.py)
combines these independent coordinate intervals into a Cartesian box. It
reuses the existing flow replay and the shared signed fraction-rounding
helper. B18's positive result-cell implementation is centralized into one
signed cell reader used for both additions and inverse scaling. Numeric
positive and negative zero share their cell; unit-rate zero-sign
canonicalization does not change positive EPI additions.

Subnormal endpoints illustrate why scaling cannot simply be assumed exact.
Put `u=2^-1074` and start at `x_0=u`. A held increment `q=u` gives trace
`2u,3u,4u,5u`; its represented pressures range from `9u` to `23u`, with
open real boundaries `8u,24u`. For `q=2u`, the trace is `3u,5u,7u,9u`
and the pressure interval is `[24u,40u]`, including both endpoints.
The difference is increment parity. The ordinary half-EPI stasis case
recovers exactly `[-2^-51,2^-50]`.

The box is maximal for **all four EPI states**, not asserted maximal for
the final endpoint or mean alone. Membership does not preserve pressure,
derivative metadata, operator admission or later refreshes. It is not a
forward-invariant complete runtime class.

### Why exactly opposite pressures can still bias EPI

An exact family law explains the inherited `EPI=0.5` boundary. Start a pair
at `(1/2,1/2)` and supply opposite pressures `(p,-p)`. Write
`Delta=2^-53`, `q=RN(|p|/16)>0`, `a=q/Delta`, and let `n,m` be the nearest
integers to `a,2a`. Exclude fractional parts `1/4,1/2,3/4` so both
lattices have unambiguous non-tie rounding. Assume

```text
4*Delta*n < 1/2,   2*Delta*m < 1/4.
```

These conditions keep the two traces in `[1/2,1)` and `(1/4,1/2]`.
The first side has spacing `Delta`; the second has spacing `Delta/2`.
The relevant halves of the initial `1/2` cell have those same respective
widths. Induction then gives, including a side that remains stationary,

```text
x_plus_j  = 1/2+j*Delta*n,
x_minus_j = 1/2-j*Delta*m/2,       j=0,...,4.
```

The pair-sum change **per substep** is `Delta*(n-m/2)`. For fractional
part `r` of `a`, it is `-Delta/2` when `1/4<r<1/2`, `+Delta/2` when
`1/2<r<3/4`, and zero on the remaining non-tie intervals. Thus positive,
negative and zero bias all occur under exactly opposite pressures. This
is a representation effect of the solver; it is not a new structural
source in the nodal equation. Ties remain covered by the general cell
constructor even though this simple family law excludes them.

### Binding the obstruction to the corrected C6 records

The detached
[`c6_winding_pressure_cells.py`](../benchmarks/c6_winding_pressure_cells.py)
validates the six B20 intervals through the existing record and Euler-cell
owners. For each interval it derives the pressure box and constructs two
distinct arithmetic controls, when feasible:

- A single-coordinate zero-sum control tries `p_i-sum(p)` in fixed node
  order, requiring exact binary64 representation and box membership.
  Failure would exclude only this construction, not all zero-sum tuples.
- An opposite-pair control tests the represented intersections
  `I_i intersect -I_(i+3)`. Their inclusive endpoints are represented, so
  nonempty intersections give an exact feasibility decision. The selected
  first pressure is its recorded value clamped to that intersection; the
  partner is its exact negative. This is a diagnostic construction, never
  a pressure write to the graph.

Both controls exist inside every one of these six boxes and reproduce
all four EPI rows exactly. For k1, the first interval has `sum(p)=2^-68`.
Decreasing node 0's pressure by one represented ulp makes the sum zero
without changing any EPI state. In the second interval, the corresponding
change is one ulp `2^-69` at node 1. Consequently the tiny nonzero pressure
sums cannot explain the retained EPI drift: it persists even when their
contribution to the mean is exactly zero.

The opposite-pair control in the first k1 interval keeps its first three
recorded pressures and completes their negatives. Its fractional parts
`r=frac(abs(p_i)/(16*Delta))` are

```text
8875/262144, 153711/262144, 314293/524288.
```

The first lies below `1/4`; the other two lie between `1/2` and `3/4`.
The family law therefore predicts pair increments `0,2^-54,2^-54` on
each of the four substeps. The mean rise is exactly `1/(3*2^52)`, matching
the retained k1 record. Its second interval has no mean rise. The smallest
inverse-pressure margins for these paired controls are respectively
`181111/2^70` and `305705/2^70`, both strictly positive. The unchanged
EPI trace therefore persists on more than the isolated balanced tuple.

The k3 comparison also remains scoped: its zero **net** mean change consists
of `+1/(3*2^53)` and `-1/(3*2^53)` in the two intervals. It does not preserve
the mean at each interval. The null retains zero mean change in both.
Every pressure, scaling and addition term remains separately recorded.

Reproduce the detached analysis without executing another graph word:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pressure_cells.py
```

The balanced pressures are represented, box-feasible arithmetic witnesses;
their production by a canonical phase state is not asserted. Replacing
each held interval independently is not a proof that one full alternative
operator execution would produce them. Historical bytes and their producer
identity remain separate from this derived artifact.

Further pressure-only accuracy or conservation changes **inside these
boxes cannot remove this EPI bias**. Crossing a trace boundary is necessary
but does not by itself ensure a different mean. The next joint-domain
argument must connect actual pressure refresh to evolving addition cells
and accumulate their signed effects. If a different solver representation
or carried rounding remainder is studied, it needs an explicit nodal
balance, state, clipping and executor-provenance contract; no such change
is silently applied here. Complete-runtime preservation, autonomous
formation and laboratory correspondence remain open.

## 20. Exact nodal area with a carried numerical remainder

B2.d.22 addresses the executor component of the signed prefix budget with
an explicit alternative numerical representation. The shared
[`_euler_kernel.py`](../src/tnfr/dynamics/_euler_kernel.py) now additionally
provides `initialize_nodal_remainder` and `advance_nodal_remainder`.
`euler_update`, its production callers and `DefaultIntegrator` retain their
existing behavior. This is a supplied-input reference, not a graph solver.

### State and exact nodal balance

Let `F(x)` denote the exact rational value of a binary64 number. The
immutable numerical state is `(x,r)`, representing `X=F(x)+r`, with
`x=RN(X)`. Initialization uses `r=0`. For a declared represented timestep,
nonnegative capacity and supplied pressure, the kernel computes

```text
a_i = F(h)*F(nu_i)*F(p_i),
Z_i = F(x_i)+r_i+a_i,
y_i = RN(Z_i),
r_i_next = Z_i-F(y_i).
```

The exact remainder is fed into the next update. It is numerical metadata
determined by the represented nodal area, not an additional physical field
or a tunable pressure correction. Gamma is absent. Both reconstructed and
displayed EPI must remain in the declared `[F(lo),F(hi)]`, with
`0<lo<=hi<=1`. A violation is rejected without mutation; there is no clipping.
Checking displayed EPI alone would miss exact excursions hidden by rounding.
Continuation also validates nearest-even encoding and the dyadic remainder.

Every binary64 factor has denominator dividing `2^1074`. The product of
the three factors therefore has denominator dividing `2^3222`; exact sums
and carried remainders preserve that denominator bound for every finite
sequence. The bounded EPI range bounds their numerator size as well. These
limits follow from the representation, not from an imported physical scale.

The coordinate identities are exact:

```text
F(y_k)-F(x_k) = a_k+r_k-r_(k+1),
X_N-X_0 = sum_k(a_k),
F(x_N)-F(x_0) = sum_k(a_k)+r_0-r_N.
```

Consequently equal exact accumulated nodal areas produce identical final
encodings, provided all intermediate states remain admissible. Equality of
rounded total durations is insufficient; the supplied rational areas must
agree. This partition identity is not a theorem about refreshed pressure
or convergence to a continuous solution.

### The signed mean budget and its remaining source

The shared observer
[`observe_nodal_remainder_sequence`](../src/tnfr/physics/nodal_remainder.py)
executes a supplied finite schedule and checks every prefix against the
existing exact nearest-even cell owner. Its arithmetic mean satisfies

```text
mean(x_N)-mean(x_0)-sum_k mean(a_k) = mean(r_0)-mean(r_N).
```

For a final cell `[ell_i,u_i]` centered at `F(x_N,i)`, the signed right side
lies between `mean(r_0)-mean(u-F(x_N))` and
`mean(r_0)-mean(ell-F(x_N))`. Closed bounds remain valid when an odd
significand excludes the tie itself. The positive unit band also gives
the simpler bound `abs(mean(r_0))+2^-53`, independent of the finite step
count. With zero initial carry and balanced accumulated nodal area,
the reconstructed mean is exactly conserved and displayed-mean drift is
at most `2^-53`. Displayed-mean conservation is not generally exact.
For heterogeneous capacity, the required source balance concerns
`sum_i nu_i*p_i`, or its explicitly weighted analogue, not `sum_i p_i`.
The observer reports the arithmetic mean only.

In the section 13 defect notation, against a specified model pressure
`p*_k`, the defect is

```text
delta_k = h_k*diag(nu_k)*(p_k-p*_k)+r_k-r_(k+1).
```

The executor term telescopes. The pressure-realization term still
accumulates and requires its own structural bound. No zero-sum projection
is applied to it. Recording discarded rounding errors without feeding
them back would not establish this endpoint-only executor bound.

### Retained-pressure comparison

[`c6_winding_nodal_remainder.py`](../benchmarks/c6_winding_nodal_remainder.py)
validates the B20 records through the B21 owner once. Each null/k1/k3 case
supplies two pressure tuples, each held for four steps at `h=1/16`, with
unit capacity and initial displayed EPI `0.5`. The remainder continues
across all eight steps. The original, single-coordinate zero-sum and
opposite-pair witness schedules are compared separately. No graph event,
pressure refresh or new preparation is executed.

For the original prescribed pressures the terminal changes are:

| Mode | Ordinary Euler mean | Carried displayed mean | Carried reconstructed mean |
|------|---------------------|------------------------|----------------------------|
| null | `0` | `-1/(6*2^54)` | `19/2^115` |
| k1 | `1/(3*2^52)` | `-1/(6*2^54)` | `2^-72` |
| k3 | `0` | `0` | `2^-73` |

Both balanced witness schedules preserve the reconstructed mean exactly
in all three cases and at every prefix. Their displayed means can still
change: the single-coordinate witnesses give the same displayed terminal
column above; opposite-pair witnesses give `0`, `-1/(6*2^54)` and
`-1/(3*2^54)`. Exact pressure balance therefore does not imply exact
displayed mean balance even for the new encoding. It does remove the
represented nodal mean source, leaving only the bounded terminal remainder.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_nodal_remainder.py
```

Once a carried state differs from the retained source, the later supplied
pressure remains a counterfactual input, not freshly generated pressure on
that state. The artifact records both sources explicitly. The B21 inverse
boxes describe the old four-RN map and are not invariant boxes for this
alternative. Historical artifacts retain their original bytes and identity.

### Gate before graph integration

The next dependency is a declared solver-state contract: whether pressure
reads displayed or reconstructed EPI; how operator and REMESH EPI jumps
transport or reset the remainder; how topology changes transfer it; and
which coordinate histories, derivatives and Mutation gates observe.
Resetting `r` to zero changes reconstructed EPI by `-r` and must enter the
nodal jump budget. Clipping likewise requires an explicit signed defect.
Existing graph transactions, snapshots and sealed certificates must retain
the same state and verify this balance. Their present guards must not be
relaxed to hide an unaccounted carry write.

This closes a conditional numerical prefix mechanism for the declared
encoding. Generated-pressure summability, admission and invariant bands,
full operator composition, autonomous structure formation and laboratory
correspondence remain open.

## 21. Live pressure and an explicit carried flow/event contract

B2.d.23 introduces a separate finite executor,
[`execute_nodal_remainder_event_schedule`](../src/tnfr/operators/nodal_remainder_runtime.py),
for the section 20 representation. It reuses the existing whole-word
validators, canonical network-stage dispatcher, physical partition builder,
pressure-write guards, history recorder and graph transaction. The ordinary
event executor and `DefaultIntegrator` retain their previous contracts.

### Numerical state, pressure and event ownership

The restricted path uses a simple undirected graph with fixed ordered nodes,
support and scalar EPI chart. Its first invocation attaches zero carry.
Subsequent invocations require the retained graph-owned binding to match
the graph, visible EPI, clock, support and declared band. A mismatch rejects
continuation; it does not discard the old remainder and restart. The binding
is numerical solver state. It does not introduce another TNFR field.

Pressure is generated by the canonical default callback on the **current
visible EPI**, with the callback's other inputs retained from the live graph.
Each declared positive physical segment gets a refresh before its shared
carried update, and the terminal boundary is refreshed as well. An
unpartitioned positive interval is one explicit segment. These are physical
refresh boundaries, not hidden held-pressure `DT_MIN` subdivisions.
Gamma, extended dynamics, custom pressure/integration paths and soft
clipping are outside the new contract. Both visible and reconstructed EPI
must remain in the declared band; the executor never clips a failed update.

Allowed events are Coupling, Coherence and Silence. They must pass the
existing grammar and stage admission and preserve actual EPI, its kind and
the supported graph structure. The carry is retained unchanged across
each accepted event. Unsupported glyphs or an actual EPI/support change
cause failure and whole-invocation rollback. The restriction is not a claim
that every execution of those three operators preserves this domain.

For an arbitrary prospective event `x -> y`, `r -> s`, the accounting rule
would be

```text
J_X = F(y)-F(x)+s-r.
```

The admitted events have `y=x` and `s=r`, hence `J_X=0`. Resetting carry to
zero would instead introduce `-r` into the reconstructed jump. This executor
does not implement that reset. EPI-writing operators, explicit delayed
REMESH, topology transfer and clipping need separate signed jump contracts.

Displayed derivatives retain the existing `nu_f*DeltaNFR` and derivative-
difference conventions. Timestamped histories record visible EPI at the
declared physical boundaries. Mutation continues to use those visible
secants, not an invisible reconstructed increment; Mutation execution
itself remains outside this first carried-event scope. Graph-owned carry,
histories, caches, event log and structural aliases participate in the same
outer rollback. External effects retain the existing transaction exclusions.

The new result seals the executed flow inputs, carried states and event
boundaries under its own provenance. It does not repurpose an ordinary
`ExecutedNodalFlowInterval`, whose visible nodal residual generally differs
by `r_before-r_after`. Its shared prefix observer checks the complete finite
reconstructed balance. A retained result does not authenticate future calls
or establish current live graph state after other code changes it.

### Exact readout difference in the EPI pressure channel

[`observe_nodal_remainder_pressure_readout`](../src/tnfr/physics/nodal_remainder_pressure.py)
reuses the exact conductance and Laplacian owners. For fixed symmetric
nonnegative conductance with positive row strengths, set `X=x+r` and
`p_epi(x)=-w_epi*L_rw*x`. Then

```text
p_epi(x)-p_epi(X) = w_epi*L_rw*r,
p_stored-p_epi(X) = [p_stored-p_epi(x)]+w_epi*L_rw*r.
```

The bracketed residual includes the other pressure channels and their
realization. Without identifying them separately, it must not be called
numerical error. This observer makes no ideal nonlinear-phase assumption.
The capacity-weighted readout shift is `diag(nu_f)*w_epi*L_rw*r`.
For a regular graph with common capacity its arithmetic mean is exactly
zero. More generally symmetry gives zero degree-weighted pressure shift,
and positive capacities give zero nodal shift in weights `d_i/nu_f_i`.
Irregular degree or unequal capacities need not conserve the arithmetic
mean. Changing those weights between events does not give a conserved
weighted trajectory mean. Zero capacities leave this particular reversible
metric undefined, although the carried update itself still admits zero
capacity and preserves its exact encoding.

Thus the carried prefix mechanism and the pressure-readout identity fit the
same nodal equation. The generated source still needs a cumulative bound.
Removing executor rounding accumulation cannot remove a sustained nonzero
represented pressure source.

### Matched finite runtime comparison

[`c6_winding_remainder_runtime.py`](../benchmarks/c6_winding_remainder_runtime.py)
reuses the three inherited null/k1/k3 preparations, seed 17 and canonical
coefficients. Each branch executes `UM IL UM IL SHA`, with a flow of `0.25`
after each IL and four explicit `1/16` segments per flow. The ordinary
branch uses the existing physical-partition event executor. The carried
branch uses the new restricted executor. Both recompute pressure on their
own live visible EPI; neither receives the other's recorded pressure.

The mesh is a new declared pressure-refresh protocol. The B20/B22 held
pressure traces are not silently relabeled as this protocol. The report
records each branch's actual area `A=sum h*nu_f*p`, visible change `V`, and
executor contribution `R=V-A`, with the exact comparison

```text
V_carried-V_ordinary = (A_carried-A_ordinary)+(R_carried-R_ordinary).
```

Different pressure histories are allowed and explicitly retained. The
carried executor term is the endpoint remainder telescope; the source term
remains measured rather than projected to zero. The instantaneous C6
readout shift is checked alongside every carried segment. This finite
comparison tests integration and causal input ownership. It does not
establish generated-pressure summability, future admission, an invariant
joint domain, autonomous preparation or laboratory correspondence.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_remainder_runtime.py
```

The retained eight-segment observations give the following changes from
the initial mean `0.5`:

| Mode | Ordinary visible mean change | Carried visible mean change | Carried exact nodal mean area |
|------|------------------------------|-----------------------------|-------------------------------|
| null | `0` | `-1/(6*2^54)` | `53/(3*2^115)` |
| k1 | `1/(3*2^52)` | `-1/(6*2^54)` | `11/(3*2^74)` |
| k3 | `7/(6*2^54)` | `0` | `-5/(3*2^75)` |

The branches have different pressure tuples on respectively 2, 6 and 6 of
the eight segments. All nine carried events following a nonzero remainder
preserve that remainder, and every segment's exact regular/common-capacity
readout mean shift is zero. All finite prefix and branch-difference
identities hold. The null's visible finite drift is larger in the carried
branch: this result does not promise smaller error on every finite input.
Its benefit is the explicit terminal-only executor bound and exact area
accounting. Nonzero generated area is retained even when an ordinary trace
appears stationary. The k3 ordinary result also differs from the earlier
held-pressure trace because this comparison refreshes pressure each segment.

The next structural dependency is a signed cumulative bound for pressure
actually generated in this carried domain, together with preservation of
the domain and word admission. Extending the event set requires a defined
jump/carry map before runtime use; a larger horizon alone proves neither.

## 22. Generated pressure can exclude a fixed-phase numerical equilibrium

B2.d.24 supplies an obstruction before any longer runtime campaign. It
concerns the actual binary64 CPU pressure map on ordered unit C6, unit
capacity, fixed represented phases and fixed positive channel weights.
The EPI band is `[0.05,1]` or a closed subinterval. Frequency and topology
gradients vanish in this domain. It is a numerical realization result,
not a theorem that the exact-real TNFR pressure lacks an equilibrium.

### A necessary cancellation condition over the complete EPI band

[`derive_binary64_c6_pressure_equilibrium_obstruction`](../src/tnfr/physics/binary64_pressure_equilibrium.py)
reuses the shared certified phase midpoint, fused pressure kernel and exact
nearest-even rounding cells. Every binary64 EPI in this band is on the
`2^-57` lattice. Each unit C6 row has two neighbors. In the ordinary linear
reducer, rounded differences remain on that lattice; halving and reducing
the two contributions gives a value on `2^-58 Z`. The mixed-sign rational
fallback computes an exact neighbor mean difference on the same lattice
before rounding its weighted product. Thus both paths satisfy

```text
g_epi_i = RN(w_epi*q_i),   q_i in 2^-58 Z.
```

This is an image inclusion. Not every such lattice value is attainable by
an EPI tuple in the band, and the six gradients are coupled by the graph.
With fixed phases the actual weighted phase contribution `A_i` is fixed;
the CPU assembly gives `p_i=RN(A_i+g_epi_i)`. Both addition operands are
binary64 numbers, so their exact sum is a multiple of `2^-1074`. It can
round to zero only if it is exactly zero. Therefore a zero-pressure tuple
requires, at every node,

```text
RN(w_epi*q_i) = -A_i,
q_i in C(-A_i)/w_epi intersect (2^-58 Z),
```

where `C` is the exact nearest-even cell, including a midpoint tie only
for an even significand. The observer computes exact inverse endpoints
and the first/last integer lattice indices. One empty row suffices to
exclude a zero-pressure tuple throughout the EPI band. Nonempty rows are
inconclusive; they prove neither attainable local cancellation nor a joint
equilibrium. No pressure projection or coefficient adjustment is used.

The static report reuses the first post-UM/IL null capture retained by B20.
The shared CPU kernel reproduces its stored pressure exactly, with the
inherited default coefficients. The six inverse integer ranges are

```text
(8,7), (-1,-2), (-42,-43), (85,84), (-168,-169), (121,120).
```

Every range is empty. Consequently no displayed EPI tuple in `[0.05,1]^6`
has zero generated pressure on this fixed phase slice. The ideal midpoint
sum still cancels exactly, with its rational and true-pi coefficients
separately zero. That cancellation does not make each represented weighted
source cancellable by the represented EPI channel.

For the carried representation with a fixed positive step `h` and unit
capacity, an excluded row has `|X_i(k+1)-X_i(k)| >= h*2^-1074` at every
admissible update. Its increments cannot tend to zero. Reconstructed EPI
therefore cannot converge while these conditions persist. The bound is
not a practical rate estimate. Bounded oscillation, signed cancellation
over several steps and band exit remain separate possibilities. This
argument neither determines the signed mean nor excludes convergence with
vanishing steps, changing phases or another domain.

### Exact duration of an unchanged visible state

[`derive_nodal_remainder_cell_horizon`](../src/tnfr/physics/nodal_remainder.py)
uses the existing carried encoding and rounding-cell owner. For supplied
constant represented inputs, put `a_i=h*nu_i*p_i`. It derives the largest
integer `N` such that every `X_i+n*a_i`, `0<=n<=N`, stays in both the source
rounding cell and the declared EPI band. Directional distance divided by
`|a_i|`, with open/closed ties retained, gives the bound without iteration.
Zero increments have an unbounded unchanged prefix. A nonzero source has
a finite cell or band boundary even if ordinary Euler repeatedly discards
its increments.

If the pressure map reads visible EPI and all its other inputs stay fixed,
the recomputed source remains identical throughout that prefix by
induction. The null capture, initialized at zero carry with `h=1/16`, has
five unchanged steps. The sixth update leaves node 5's source rounding
cell while all reconstructed coordinates remain in the positive band.
This is a cell exit, not a positive-band exit. The pressure may change
after it; the earlier mean source must not be extrapolated indefinitely.

### Evidence and the next gate

[`c6_winding_pressure_equilibrium.py`](../benchmarks/c6_winding_pressure_equilibrium.py)
validates and analyzes one historical phase-bearing capture. It executes
no graph trajectory or event. Its freshly recomputed static pressure and
exact arithmetic have new source provenance; the phase retains its B20
provenance. B23's serialized flows do not contain post-IL phase coordinates,
so this report does not authenticate their equality with this source.
In particular the six-step conditional flow is not a continuation of
B23's schedule, whose next UM/IL occurs after four steps.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pressure_equilibrium.py
```

The next gate is a joint generated-pressure/domain argument that permits
the finite-resolution behavior actually present. Test a signed block
source bound and a preserved trapping region under the actual UM/IL phase
updates, rather than require a fixed-phase exact zero that this numerical
map cannot attain. The existing centered contraction controls disagreement
but still removes the common mean; retain both quantities. A trapping
region would not by itself prove point convergence, autonomous formation
or laboratory correspondence. These are open questions, not consequences
of this obstruction or reasons to extend a trajectory without a new bound.

## 23. Exact closure of the generated UM/IL phase component

B2.d.25 replaces the arbitrary fixed-phase premise for one prepared
numerical phase path with exact shared-kernel replay and cycle closure.
It also determines whether that cycle's phase-source mean cancels. It
does not close the EPI trapping or full-runtime admission problem.

### Projection, support and exact state identity

[`observe_c6_coupling_coherence_phase_step`](../src/tnfr/physics/c6_phase_orbit.py)
evaluates the existing pure all-target UM proposal/merge and simultaneous
IL phase proposal on fresh ordered unit-C6 read fixtures. It retains
default factors, bidirectionality, functional links and the production
neighbor/receiver order. EPI, pressure and sense-index fixture values are
not advanced or presented as an engine trajectory.

UM phase proposals depend only on phase, ordered support and the resolved
phase factor. IL's phase proposal depends only on phase, neighbors and
its default coefficient. EPI/SI and candidate selection can affect UM's
functional links, so the projection requires all six edges to pass U3 and
every nonedge to remain strictly excluded before UM, after its merge and
after IL. No link rule is disabled. Those checks remove candidate-link
dependence on the other nodal fields on the declared path. Unit capacity
is preserved by the actual UM capacity proposal; stored-pressure changes
do not feed either phase proposal.

The separate orbit observer accepts a finite path, rederives every
transition, and requires the terminal six-float tuple to equal a declared
earlier tuple bit for bit. It distinguishes signed zeros and does not
identify phases by centering, a tolerance or a common rotation. Exact
closure and determinism establish periodicity of this phase projection
in the same fixed numerical environment. No uniform accuracy theorem for
the host's transcendental functions is assumed. This is conditional
numerical recurrence, not evidence that every future engine stage passes
grammar, history, pressure or EPI-band requirements. Public declared
preperiod/period indices need not be minimal.

The benchmark uses a predeclared ceiling of 256 phase-only transitions to
search for this exact closure, starting from the unchanged inherited null
preparation. It stops on the first exact repeat. The ceiling is a compute
limit, not a physical parameter or a fitted horizon. Failure to find a
repeat would retain only a finite path. The supplied-path observer does
not perform a search or infer closure from near-repetition.

### The inherited null reaches a phase fixed point

The retained path first repeats at transition 90: preperiod 89, period 1
for the post-IL six-coordinate map. The first two UM and IL outputs match
the phase-bearing B20 records exactly. Only coordinate 0 changes along
this finite path; the other five retain their original stored `i*pi/3`.
This finite observation does not establish an invariant scalar interval.
The terminal coordinate is

```text
post-IL theta_0 = 0x1.0b8fb3e3956cbp-55,
post-UM theta_0 = 0x1.ab75f42bdf52ep-55.
```

UM still changes phase inside the block; IL returns the complete tuple to
its exact starting value. The term "fixed point" refers to their composed
phase map. All retained stage boundaries preserve edge admission and
strict nonedge exclusion. There is no execution of a 90-block grammar
word, nodal flow, carry trajectory or SHA closure in this experiment.

The shared CPU kernel gives the terminal post-IL weighted phase source
`A`, whose exact sum is `-2^-109`. Its arithmetic mean is
`b=-1/(6*2^109)`. The ideal true-circle midpoint sum remains exactly zero;
the represented source mean does not. For an inherited flow duration
`H=1/4` after each IL, its mean area per block is `-1/(24*2^109)`.
This is a source-component identity under the closed phase projection,
not an observed total EPI drift. Refreshed full pressure still depends on
visible EPI and its arithmetic.

Reusing section 22's equilibrium observer on this terminal phase gives
the six empty cancellation ranges

```text
(16,15), (-5,-6), (-42,-43), (85,84), (-168,-169), (117,116).
```

Hence the actual EPI/phase pressure map still has no zero-pressure EPI
tuple anywhere in `[0.05,1]^6` on this reachable phase-projection slice.
If an admitted complete trajectory preserves this phase cycle, support,
unit capacity and the band, with fixed positive refreshed flow steps,
its reconstructed EPI cannot converge. This excludes an exact point
limit in that restricted numeric regime; it does not exclude trapping,
compensating signed increments or changing-capacity regimes.

### A periodic source is not automatically a bounded source

The existing [pressure-readout owner](../src/tnfr/physics/nodal_remainder_pressure.py)
now derives the exact periodic source budget. Let `a_j=mean(A_j)` be the
post-IL source of block `j` in a declared period of length `m`, and define

```text
b = sum_j a_j / m,
C_r = sum_(j<r) (a_j-b),   C_0=C_m=0.
```

For `N` blocks of common duration `H`, the phase contribution is exactly
`H*(N*b+C_(N mod m))`. Only the centered offset is uniformly bounded,
by `H*max_r |C_r|`. Its full prefixes are bounded if `b=0`; otherwise
this component has a linear drift. For the period-one null result, all
centered offsets vanish and the nonzero drift remains.

For unit capacities, define the actual accumulated nonphase mean area
`E_N=sum_blocks sum_segments h*mean(p-A)`. The exact C6 EPI diffusion
channel has zero mean, so this term contains EPI-reduction and final
channel-assembly effects, separately from the already materialized phase
source. Under EPI-preserving events and retained carry, the nodal equation
then gives

```text
mean(X_N)-mean(X_0) = H*(N*b+C_(N mod m)) + E_N.
```

Visible mean adds the existing terminal carry difference. Necessary
membership of reconstructed mean in `[ell,U]` requires

```text
ell-mean(X_0)-phase_area(N) <= E_N
                          <= U-mean(X_0)-phase_area(N).
```

Thus bounded mean would require `E_N/N -> -H*b`. The new compensation
observer computes this finite necessary interval from supplied exact
area; it does not authenticate that input. A mean in the band is still
insufficient to put every node there. The phase-only benchmark supplies
no invented nonphase area and claims no actual compensation.

### Evidence and next dependency

[`c6_winding_phase_orbit.py`](../benchmarks/c6_winding_phase_orbit.py) preserves
the B20 source identity, discovered path, independently replayed cycle,
all phase margins, source budget and terminal lattice obstruction. The
cycle-source rows follow the post-IL outputs of the cycle's steps. Its
detached phase sequence has fresh source provenance; it is not a new
complete-runtime seal or a reconstruction of missing B23 phase records.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_phase_orbit.py
```

The next dependency is now narrower: determine whether the actual
refreshed EPI map, including its carry, produces the compensating signed
area while preserving a trapping region and stage admission. In the
closed period-one tail, phase itself is no longer an unspecified forcing
schedule. Do not assume that periodicity makes its nonzero source mean
harmless, subtract that mean by a pressure projection, or infer a band
exit from it. A verified complete reduced-state transition region or an
analytic signed balance is required before extending the live campaign.

## 24. Local pressure lattice, product-trap obstruction and finite compensation

B2.d.26 studies freshly generated EPI/phase pressure on section 23's
closed phase tail. It identifies a finite compensation mechanism and a
limit on the shape of any proposed trapping region. The tested EPI states
are explicitly supplied local controls; the phase-only path did not prove
that the full preparation reaches them with their supplied carry.

### The local discrete Laplacian is exact

[`derive_c6_pressure_lattice`](../src/tnfr/physics/c6_pressure_lattice.py)
uses a represented slab `[a,b]` inside `[0.05,1]`, with
`delta=ulp(a)` and exact width `b-a <= 2^52*delta`. The default analysis
slab is `[3/8,5/8]`: it crosses the `0.5` binade boundary, contains the
inherited preparation, and has `delta=2^-54`. This is a domain for a
numerical proof, not a new physical coefficient or a modified preparation.

Every displayed coordinate in the slab has the exact form
`x_i=a+delta*n_i`, with integer `n_i`. On ordered unit C6,

```text
m_i = n_(i-1)+n_(i+1)-2*n_i,
q_i = (delta/2)*m_i,       sum_i m_i = 0.
```

The width bound makes both neighbor differences, their halves and their
two-term sum exactly representable. The mixed-sign rational fallback
gives the same `q_i`. Thus both production EPI reducer paths give
`g_i=RN(w_epi*q_i)` with no prior reduction error in `q`. The remaining
roundings are the coefficient multiplication and final channel addition:

```text
p_i = RN(A_i+g_i).
```

The observation API reconstructs this integer Laplacian, checks the
shared reducer and fused pressure outputs, and separates the signed
means of the coefficient error and channel-addition error. The exact
unrounded EPI contribution has zero mean. The actual pressure mean need
not equal either zero or the fixed phase-source mean.

### A Cartesian product cannot supply the trapping certificate here

The source owner from section 22 already gives the nearest-even cell of
`-A_i`. Dividing it by `w_epi*(delta/2)` yields exact integer sign
thresholds, retaining open and closed ties. Nonnegative pressure requires
`m_i >= l_i`; nonpositive pressure requires `m_i <= u_i`. For the closed
null phase tail in the default slab these thresholds are

```text
l = (2,0,-5,11,-21,15),       sum(l) = 2,
u = (1,-1,-6,10,-22,14),      sum(u) = -4.
```

Since `sum(m)=0`, no displayed EPI tuple in the slab can have every
pressure nonnegative or every pressure nonpositive. Every tuple has at
least one strictly increasing and one strictly decreasing nodal direction.
This is an analytic result over the whole slab, not an inference from
the finite controls below.

For unit capacity and any fixed `h>0`, the carried map is
`X_next=X+h*p(RN(X))`. Consider a nonempty Cartesian product of admissible
reconstructed-coordinate sets in the slab. The bounded dyadic encoding
makes each such coordinate set finite. The product therefore contains
its coordinatewise maximum and minimum. At the maximum corner some
pressure is positive, so its exact update leaves that coordinate set;
at the minimum corner some pressure is negative, with the analogous
result. Such a product cannot be forward invariant under accepted steps.

This argument concerns reconstructed `X=x+r`, including the carry. A
positive exact increment can leave the proposed set while the displayed
float remains unchanged. It is not a proof about a box of displayed EPI
alone. Zero timestep or inactive capacity would invalidate the stated
directional argument. A bounded trajectory may lie in a correlated subset
whose enclosing Cartesian box is not invariant; neither such a trajectory
nor a correlated trapping region is excluded. In particular, this is not
a theorem that every trajectory leaves the analysis slab or positive band.
It concerns invariance under each individual carried update. A set that
permits intermediate departures and is invariant only at quarter-block
sampling times is not excluded by this corner argument.

### Actual compensation occurs, then fails at the first cell boundary

The report fixes a 13-state local control: uniform EPI `0.5` and each of
its twelve one-coordinate binary64 predecessor/successor neighbors. It
retains the closed phase tuple, unit capacity, support and default weights.
These are static pressure evaluations, not a search over operator words,
an altered forcing law or a claim of full-state reachability.

At the state with only node 0 equal to `nextafter(0.5,-infinity)`, the
shared canonical pressure satisfies `sum(p)=0` exactly. The phase source
still has `sum(A)=-2^-109`. The EPI coefficient-error sum is zero here;
final channel addition supplies `+2^-109`, exactly compensating that
phase-source bias. Every pressure vector in the stencil has both signs.
The other twelve states have strictly negative pressure sums. Thus actual
compensation is possible in this local class, without a pressure projection,
but is not an identity across adjacent states.

Initialize this supplied balanced state with zero carry and use the
inherited `h=1/16`. The existing exact cell-horizon owner gives five
unchanged visible steps. Step six changes node 5 to the predecessor of
`0.5`, while node 0 stays at its predecessor and the other nodes stay at
`0.5`. The unchanged pre-update visible state identifies the same canonical
pressure at each of those six updates. Replaying those inputs through the
shared carried-prefix owner gives exactly zero reconstructed mean change
at every prefix. This is a conditional numerical replay, not six live
graph refreshes. The final update leaves a rounding cell while remaining
in the slab.

Refreshing canonical pressure at that next visible tuple gives
`sum(p)=-2^-109` again. A further interval therefore does not inherit the
balanced source. This first-boundary discriminator explains both the
existence of genuine finite compensation and why it cannot be extended
by holding the old pressure after the state changes. No long trajectory
or claim of repeated full-runtime stability is needed for the result.

### The complete 13-state class cannot retain a bounded carried orbit

The finite control also permits a stronger exact conclusion about this
particular class. Let `b` denote the pressure vector at its unique
zero-sum state. It is nonzero. Every other pressure vector `p_j` has
`d_j=-sum(p_j)>0`. Define an exact observation functional by

```text
M = max_j |dot(b,p_j)| / d_j,
epsilon = 1/(1+2*M),
ell_i = -1+epsilon*b_i.
```

Then `dot(ell,b)=epsilon*dot(b,b)>0`; for every other vector,
`dot(ell,p_j)>=d_j*(1-epsilon*M)>d_j/2`. Thus the finite set of actually
generated pressures has a strictly positive minimum projection `c`.
These coefficients define a certificate, not a new field, pressure law
or tuned model parameter. No pressure value is modified.

[`observe_finite_nodal_pressure_drift`](../src/tnfr/physics/nodal_remainder_pressure.py)
uses each visible state's existing exact rounding cells to enclose all
its admissible carried coordinates, including the band endpoints. Their
union has finite functional width `W`. With unit capacity and fixed
positive step `h`, a trajectory staying in this finite visible class at
every update would satisfy both

```text
dot(ell,X_N-X_0) >= N*h*c,
dot(ell,X_N-X_0) <= W.
```

It must therefore leave the class, or fail the declared update contract,
by step `floor(W/(h*c))+1`. This bound covers every admissible initial
carry and requires no long integration. It is conservative and does not
give a practical timescale or a physical-time prediction. The observer
accepts supplied pressure tuples; the benchmark separately binds each to
the shared canonical producer. Nonpositive separation would be inconclusive.

This rules out a permanently class-confined oscillation as well as a
fixed point, even though one state has zero mean pressure. It does not
exclude bounded motion in a larger correlated class, departure followed
by return, or trapping in the full slab. Restricting only sampled block
endpoints to these 13 states would not satisfy the per-update hypothesis.
The balanced six-step control already leaves this class at its first
cell exit, when two coordinates are predecessors of `0.5`.

### Ownership and next gate

[`c6_winding_pressure_lattice.py`](../benchmarks/c6_winding_pressure_lattice.py)
revalidates the retained B25 phase path, uses the shared local lattice
observer for all static pressures, and reuses the existing carried update
and cell horizon for the single boundary transition. The original phase
artifact remains unchanged. No independent pressure kernel, modified
coefficient, EPI clipping or carry reset is introduced.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pressure_lattice.py
```

The next candidate must couple coordinates and carry explicitly. A union
of correlated cells or a signed structural functional must control actual
generated pressure across their boundaries, including the return path
after the first loss of compensation. The 13-state class now has a strict
separator and cannot serve as that invariant class. A repeated visible tuple alone is
insufficient: exact nodal area can accumulate in its carry. A necessary
control is whether the pressure vectors in a proposed finite class can
cancel coordinatewise over time; reuse the exact separator observer to
reject classes with uniformly signed structural drift. Passing that necessary condition would
still not prove transition closure or a reachable invariant set. These
checks should precede a longer live word and its admission/carry bridge.

## 25. Exact carry-compatible itineraries across pressure-cell boundaries

B2.d.27 replaces an existential graph of visible transitions with an exact
test of a complete supplied itinerary. The pressure at a displayed state
does not determine its next displayed state without its incoming carry.
Two separately feasible edges can therefore fail to compose. This block
adds no pressure law, coefficient, EPI projection or integration kernel.

### Translated cells retain the complete accumulated nodal area

For a declared word `x_0,...,x_N`, let `a_k=h_k*nu_k*p_k` coordinatewise,
with each represented coefficient interpreted as an exact rational, and
let `A_0=0`, `A_k=sum_(j<k) a_j`. The shared carried equation gives
`X_k=X_0+A_k`. For each coordinate define

```text
I_i = intersection_(k=0,...,N) ((C(x_ki) intersect [lower,upper]) - A_ki),
G   = 2^-3222 * Z.
```

Here `C(x)` is the existing nearest-even rounding cell, with both midpoint
ties included for an even significand and excluded for an odd significand.
The band constrains the reconstructed coordinate as well as its displayed
value. Each displayed input must itself belong to the declared band.
The exact feasible initial encodings are `product_i (I_i intersect G)`.

[`derive_nodal_remainder_itinerary`](../src/tnfr/physics/nodal_remainder.py)
intersects these intervals with their endpoint flags and computes the
first and last admissible grid integers. A nonempty real interval alone
is insufficient: an open interval between adjacent encoding-grid points
contains no admissible carry. The grid is the existing numerical encoding
bound, derived from the product of three binary64 factors; it introduces
no physical discretization parameter.

This criterion is necessary by the carried update and sufficient because
each prefix then has the declared nearest-even representation and remains
inside the band. Every accumulated area belongs to `G`, so the encoding
bound is preserved at every step. The observer constructs one admissible
initial encoding and replays it through the existing shared sequence
owner, checking every displayed output. It separately reports whether
the displayed initial state with zero carry belongs to the feasible set.
An existential carry witness is not evidence that the live preparation
produces that carry. Likewise, supplied pressure is not authenticated by
this general observer: a domain adapter must bind it to its actual producer.

### Visible recurrence is weaker than recurrence of the carried state

For a feasible closed visible word, `x_N=x_0`, the final carry satisfies
`r_N=r_0+A_N`. Thus the complete encoding returns to itself exactly when
`A_N=0` in every coordinate. A zero arithmetic mean of `A_N` does not
suffice. With zero vector area, repeating the same supplied numerical
schedule gives a conditional periodic class, indexed by the schedule
ordinal, from the feasible initial set translated by each prefix area.
For an autonomous invariant union without this ordinal, every supplied
pressure must additionally be that map's value at its displayed state,
with the other inputs fixed.
This class need not be one Cartesian product invariant at each step, so
the conditional statement does not contradict section 24's obstruction.

If a closed visible word has nonzero vector area, repeating it shifts the
initial reconstructed state by that area on each traversal. Since its
initial cell is bounded, indefinite repetition of that same word is
impossible. This conclusion does not exclude other subsequent itineraries,
nor establish a band exit. No periodic carried C6 class is established here.

The existing balanced node-0-predecessor control makes the distinction
concrete. Its pressure mean is zero but its pressure vector is nonzero.
At `h=1/16` a constant visible word with twelve transitions has admissible
initial carry; thirteen transitions have none. Both statements use the
exact whole-word intersection, over all admissible initial carries.
The twelve-transition witness requires nonzero initial carry. Section 24's
zero-carry witness instead retains its visible tuple for five transitions
and leaves on the sixth. A visible self-loop is therefore neither an
equilibrium nor a repeatable carried-state cycle.

### Two derived boundaries reveal a source-sign reversal

[`c6_winding_carry_itinerary.py`](../benchmarks/c6_winding_carry_itinerary.py)
continues section 24's retained first-exit encoding with its carry intact.
It fixes a budget of two new cell boundaries. Each boundary horizon is
derived analytically before the shared numerical replay; canonical pressure
is regenerated whenever the visible tuple changes. The post-IL phase,
unit capacity, support, default coefficients and `h=1/16` are unchanged.

Write visible EPI as `0.5+2^-54*n`. The continuation is:

| Boundary | Additional steps | Visible offsets `n` | Refreshed pressure sum |
|----------|------------------|---------------------|------------------------|
| Retained B26 exit | 0 | `(-1,0,0,0,0,-1)` | `-2^-109` |
| First new boundary | 3 | `(-1,0,0,-1,2,-1)` | `3*2^-109` |
| Second new boundary | 12 | `(-1,0,0,-1,2,-2)` | `-2^-108` |

The first interval's reconstructed mean change is `-2^-114`; the second's
is `3*2^-112`. Their total is `11*2^-114` over fifteen accepted numerical
steps. Every prefix retains the nodal balance and stays inside the local
slab. Neither endpoint returns to the thirteen-state stencil. The complete
visible word is feasible with the inherited carry, but it is not closed.
Its alternating pressure-mean signs demonstrate why the first loss of
compensation cannot be extrapolated into permanent one-sign drift. They
do not prove long-term cancellation or a bounded orbit.

Resetting carry to zero at the retained B26 endpoint does not realize this
same fifteen-step visible word. Its inherited carry does belong to the
exact feasible set; these are separately checked facts.

The report rederives the B26 source and first-exit encoding from the shared
owners before continuing it. Its path is a conditional numerical replay,
not a new live graph word or a reachable full-state phase-tail certificate.
The original preparation's EPI and carry at that tail remain unknown.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_carry_itinerary.py
```

### One nodal direction still excludes the four observed states as a cycle

The balanced source, its first exit and the two new boundary endpoints
give four distinct displayed tuples. In all four, node 1 has exactly

```text
p_1 = -128295757220873 / 2^106.
```

Thus the signed coordinate functional `ell=-e_1` has the same strictly
positive pressure projection throughout this class, despite its pressure
mean changing sign. The existing finite-class drift owner uses node 1's
cell width `W=3*2^-55` and gives at most 842 class-confined transitions
at `h=1/16`; step 843 must leave the class or fail the update contract.
This conservative bound covers every admissible initial carry and requires
no additional trajectory. It excludes a carried cycle contained in these
four states, not a larger return class, a slab exit or a physical timescale.

### Next gate: close a relational class or exclude it structurally

Use these whole-word intersections when proposing return paths or a finite
union of carried cells. A candidate must pass coordinatewise area balance
and incoming-carry compatibility, then prove forward inclusion under the
actual pressure map. Pairwise edge feasibility, pressure-mean sign changes
and convex-hull pressure balance are insufficient on their own. A strict
pressure separator remains a way to reject a whole proposed finite class.
The four observed states already fail this test: a cycle containing them
would need a further state with strictly positive pressure at node 1 to
cancel its negative accumulated area. Section 24's exact node-1 thresholds
are `l_1=0`, `u_1=-1`, and its cancellation cell contains no lattice point.
Thus positive node-1 pressure is equivalent to
`m_1=n_0+n_2-2*n_1>=0` in this slab; all four controls have `m_1=-1`.
This is the next concrete boundary condition, to be tested together with
its incoming carry and the other five coordinate budgets.
Do not extend a trajectory merely until it appears recurrent. A new class
description or an exact return/escape criterion must first justify that
extension. Reachability from the original preparation, operator admission
and the finite live bridge remain separate later obligations.

## 26. Pressure-sign sectors and the carried curvature budget

B2.d.28 tests the necessary node-1 sign crossing identified in section 25.
It also extends the finite-state drift argument to an entire pressure-sign
sector in the local slab. These are statements about the same represented
nodal map, with fixed phase, unit capacity, conductance, coefficients and
positive step. No restoring term or adjusted pressure is introduced.

### A sign sector has a uniform nodal drift bound

The local lattice owner already identifies the monotone scalar pressure
function for each node:

```text
p_i(m) = RN(A_i + RN(w_epi * (delta/2) * m)).
```

Let `l_i` be the first nonnegative index and `u_i` the last nonpositive
index from section 24. Strictly negative pressure has `m<=l_i-1`;
strictly positive pressure has `m>=u_i+1`. If
`D=(upper-lower)/delta`, actual local gradients satisfy `-2D<=m<=2D`.
Intersect each sign sector with this range before evaluating its boundary.
This avoids treating an unreachable index as a possible local state when
the EPI weight is extremely small. An empty sector supplies no drift bound.

[`derive_c6_pressure_sign_sector`](../src/tnfr/physics/c6_pressure_lattice.py)
uses monotonicity of both nearest-even roundings. In a nonempty negative
sector, its largest index gives the least negative pressure bound; in a
positive sector, its smallest index gives the least positive bound. Thus
`s*p_i>=c>0` throughout the selected sector, for `s=-1` or `s=+1`.
The index range is an enclosure: no claim that every enclosed integer
gradient has a realizable global EPI tuple is needed for this bound.

[`observe_c6_pressure_sector_exit`](../src/tnfr/physics/c6_pressure_lattice.py)
combines that bound with a validated incoming encoding `X=x+r`. Let `d`
be its distance to the band endpoint in the direction of the pressure.
While all pre-update states remain in the sector and all accepted exact
updates remain inside that band, the nodal equation gives

```text
N*h*c <= d,
N <= floor(d/(h*c)).
```

The next step therefore cannot preserve all these hypotheses. The path
may leave the sign sector, leave the declared band, or cease to satisfy
the update contract. This is not a guaranteed sign-crossing theorem or
a band-exit theorem. At other phase inputs, leaving a strict sign sector
can enter a zero-pressure state. The observer reports no initial sector
budget if the supplied state is outside that sector. It does not replace
the pressure producer or predict live operator admission.

### The visible sign index combines exact evolution and carry transfer

The integer Laplacian index is `m=-2*L_rw*x/delta`. For a retained carried
sequence with exact accumulated nodal area `A` and remainder change
`dr=r_after-r_before`, the existing shared prefix identity gives

```text
x_after-x_before = A-dr,
m_after-m_before = -2*L_rw*A/delta + 2*L_rw*dr/delta.
```

Both terms are exact rationals. They need not individually be integer;
their sum is the observed integer index change. The benchmark uses the
existing cycle Laplacian for this readout and checks the identity at every
prefix. Neither reconstructed curvature alone nor discarded carry can
decide the visible pressure sign. This is the same carry/readout issue as
section 21, now expressed at the pressure-sign boundary rather than as an
additional field or evolution law.

For node 1 on the inherited closed phase slice, the sign cut remains
`m_1>=0` for strictly positive pressure and `m_1<=-1` for strictly negative
pressure. From initial `m_1=-1`, the two budget terms must sum to at least
one before that sign crossing is realized.

### One shared bounded cell-exit replay

[`observe_nodal_remainder_cell_exit`](../src/tnfr/physics/nodal_remainder.py)
now owns the analytic-horizon, shared-prefix replay and unchanged-cell
checks used by the section 24, 25 and 26 benchmarks. Its explicit step
budget is a resource limit, not a TNFR coefficient. A stationary horizon,
an exit beyond that budget or an exact-band departure is rejected before
allocating the step schedule. Accepted replays retain their original carry
and check every pre-update visible state, each intermediate output and the
exact first-exit endpoint. Pressure constancy still requires the caller's
fixed visible-state pressure map; the numeric helper does not prove it.

### The inherited continuation reaches positive node-1 pressure

[`c6_winding_pressure_sign.py`](../benchmarks/c6_winding_pressure_sign.py)
first replays the B27 report from its retained B26 input and compares the
derived payload, including its endpoint and carry. It then continues that
endpoint, with a declared ceiling of eight new cell boundaries and 256
new numerical steps. These are work limits only. The benchmark stops at
the first positive node-1 pressure, without evolving further under that
new pressure. A next boundary exceeding the remaining budget is reported
as unresolved before its replay, rather than replaced by a longer run.

The first crossing occurs within these limits, at the third boundary
and nineteenth new step. With EPI offsets measured from `0.5` in units
`delta=2^-54`, the finite path is:

| Boundary | Additional steps | Visible offsets | `m_1` | Refreshed pressure sum |
|----------|------------------|-----------------|-------|------------------------|
| B27 endpoint | 0 | `(-1,0,0,-1,2,-2)` | -1 | `-2^-108` |
| First | 12 | `(-1,0,0,-1,4,-2)` | -1 | `3*2^-108` |
| Second | 3 | `(-1,0,0,-2,4,-2)` | -1 | `-2^-107` |
| Third | 4 | `(-1,0,2,-2,4,-3)` | 1 | `-5*2^-108` |

The first two exits alter nodes 4 and 3, which do not enter
`m_1=n_0+n_2-2*n_1`. At the third, node 2 rises by two lattice units;
nodes 0 and 1 remain visibly unchanged. This gives `Delta m_1=2` and
strictly positive pressure. Node 5 changes simultaneously but does not
enter this node-1 sign condition. Thus the observed sign reversal is a
neighbor-driven change in the existing EPI pressure channel, with fixed
phase, capacity and coefficients. It is not a direct write to node 1's
pressure or an added feedback law.

The actual node-1 pressure changes from
`-128295757220873/2^106` to `2786216251278205/2^108`.
Every pre-update state in these nineteen steps still has `m_1=-1`.
The unchanged-cell checks and fresh boundary observations therefore
establish that this is the first positive pressure on this finite
continuation, not merely the first positive sampled endpoint.

For the complete nineteen-step prefix, the node-1 index change separates as
`86621202941648393/2^58 + 489839549361775095/2^58 = 2`.
The first term is the reconstructed nodal-curvature change; the second is
the visible encoding's carry contribution. The two-unit displayed jump
must not be identified with the first term alone. Resetting carry to zero
at the B27 endpoint cannot realize this same visible itinerary.

### Sign recovery has not yet repaid accumulated evolution

At the stopping point the new positive pressure has not supplied any
positive-duration evolution. The exact node-1 accumulated area relative
to the B27 endpoint is still

```text
A_1 = -2437619387196587 / 2^110.
```

Its displayed EPI remains `0.5`, while its incoming remainder has changed
by this negative amount. The reconstructed mean changes by
`-11/(3*2^113)` over the nineteen steps. The newly generated total pressure
sum is negative even though node-1 pressure is positive. Current pressure
sign, accumulated nodal area and mean-source sign are three distinct
readouts; none may replace the coordinatewise return condition.

Every accepted prefix stays inside the local slab and satisfies the
carried nodal and curvature budgets. The complete visible itinerary and
its inherited carry are checked by the section 25 inverse-cell owner.
No complete carried-state cycle, invariant return class, original full
preparation's tail reachability or repeated live-runtime stability follows.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pressure_sign.py
```

The next gate is the positive sector's admissible duration and signed
area, coupled to the remaining five node budgets. Determine whether
canonical subsequent transitions can compensate prior negative area
without violating their carry constraints. A proposed return must still
pass the complete vector-area and whole-itinerary tests. The completed
sign crossing removes the old four-state separator's hypothesis after
that boundary; it does not close the broader trapping problem. Do not
restart the sign search or enlarge a run merely to obtain a visual cycle.

## 27. Finite positive-pressure repayment and exact return obstructions

B2.d.29 continues the first positive-pressure endpoint from section 26.
It keeps the B27 endpoint as the reference for accumulated nodal area.
Changing the sign of current pressure did not erase the nineteen-step
negative area already accumulated relative to that reference.

### An exact affine budget separates repayment, crossing and overshoot

For an incoming area vector `D` and constant represented inputs, write
`a_i=F(h)*F(nu_i)*F(p_i)`. After an integer number `n` of those steps,

```text
area_i(n) = D_i + n*a_i.
```

[`derive_nodal_area_crossings`](../src/tnfr/physics/nodal_remainder_pressure.py)
solves these scalar equations rationally before any replay. A nonzero
increment has a continuous zero parameter `-D_i/a_i`; an exact discrete
zero requires this parameter to be a nonnegative integer. For nonzero
initial area driven toward the other sign, the first zero-or-opposite
integer prefix is its ceiling. An integral root gives exact equality;
a nonintegral root gives an overshoot. A zero initial area is recorded
separately and is not described as repayment of an earlier deficit.

The observer retains the unrestricted candidates and separately marks
those inside the supplied finite step limit. Coordinates that stay
identically zero impose no further constraint on a joint return. Every
other coordinate must have the same admissible integer root for the
complete vector to vanish. Mean-area cancellation is a separate condition
and cannot replace this intersection of coordinate conditions.

The inputs are an arithmetic budget, not authenticated pressure, band
admission or a future trajectory. The first cell-exit update still uses
the old pressure; therefore a candidate through `first_exit_step` can be
bound to the shared cell replay when it remains inside the band. A
candidate beyond that step requires fresh pressure and another budget.
Holding the source after its actual cell boundary is not a valid way to
complete a failed repayment.

### The positive episode ends before the proposed repayment prefix

The inherited node-1 area and new per-step input are exactly

```text
D_1 = -9750477548786348 / 2^112,
a_1 =  2786216251278205 / 2^112.
```

The continuous zero parameter lies strictly between steps three and four.
The first nonnegative integer candidate is step four, with
positive overshoot `1394387456326472/2^112`; no integer prefix of that
unchanged source attains exact zero. Step three would still have
`-1391828794951733/2^112`. These are exact conditional calculations, not
additional executed states.

The actual next-cell horizon is only two steps. Both use positive node-1
pressure. At the second endpoint node 2 returns from offset `+2` to `0`
in units `2^-54` relative to EPI `0.5`; the other displayed nodes stay
fixed. Hence `m_1` returns from `+1` to `-1`, and freshly generated
node-1 pressure is negative again. This completes the first positive
episode on the retained conditional path. No negative-pressure step is
taken after that refresh, and the hypothetical four-step budget is not
executed.

The new endpoint has offsets `(-1,0,0,-2,4,-3)`. Its two positive steps
supply node-1 area `2786216251278205/2^111`, leaving

```text
net area_1 = -2089022523114969 / 2^111
```

relative to the B27 reference. The new local mean area is
`-5/(3*2^112)`; including the inherited area gives mean `-7/2^113`.
Every coordinate is checked through
`X_after-X_B27 = inherited_area + new_area`. The node-1 displayed EPI
matches its B27 value, but its reconstructed coordinate and remainder
do not. Neither a node returning to a visible cell nor the recovery of
an earlier pressure sign establishes a complete carried-state return.

### Two pressure levels impose a separate integer period constraint

[`derive_two_level_nodal_return`](../src/tnfr/physics/nodal_remainder_pressure.py)
expresses one negative and one positive represented pressure using a
common exact denominator: `p_minus=-M/Q`, `p_plus=N/Q`, with positive
integers `M,N`. Under a common fixed positive step and capacity, zero
accumulated area in that coordinate requires

```text
n_minus*M = n_plus*N.
```

Writing `g=gcd(M,N)`, every nonempty such return has counts proportional
to `(N/g,M/g)` and length a multiple of `(M+N)/g`. This necessary
condition does not require executing those steps. It supplies neither a
realizable itinerary nor a return of any other coordinate.

For the two actually observed node-1 pressure levels,

```text
Q = 2^108,
M = 513183028883492,
N = 2786216251278205,
g = 1,
minimum nonempty zero-area length = 3299399280161697.
```

Thus a short exact return cannot be obtained merely by rearranging
these two levels at the existing fixed step. The bound comes from the
represented numerical coefficients; it is not a physical period or a
new TNFR constant. Another pressure level, variable step or capacity
falls outside this two-level counting argument. It excludes neither
bounded motion nor a larger invariant class, and no trajectory of this
length is attempted.

### Evidence owner and next gate

[`c6_winding_pressure_repayment.py`](../benchmarks/c6_winding_pressure_repayment.py)
revalidates the retained B28 report through its B27/B26 inputs, preserves
the incoming carry, and uses the shared first-cell-exit owner for the
two actual steps. Its signed-area reference is explicit. It checks the
complete area vector, displayed and reconstructed return separately,
and inverse-itinerary compatibility. The single boundary suffices here
because its fresh pressure ends the positive episode; otherwise it would
only supply a finite duration lower bound. Resource rejection before
the complete boundary remains an unresolved observation.

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_pressure_repayment.py
```

The next structural question is how additional canonical pressure levels
or a correlated trapping class can change the signed vector-area budget.
Use the pressure-index cuts and incoming-carry constraints to define that
test before extending a run. Do not search for a short exact cycle while
silently retaining only the two excluded levels. Exact recurrence is one
possible certificate, not a requirement to impose on every useful bounded
pattern. Original-preparation reachability, future live admission and
laboratory correspondence remain distinct open obligations.

## 28. A third pressure level and a transverse finite-class drift

B2.d.30 starts from the retained section-27 endpoint, with its exact carry,
the same fixed phase source, unit capacity and `h=1/16`. The declared search
has at most eight cell boundaries and 256 steps. It stops at the first
node-1 pressure outside the two levels from section 27, before applying
that new pressure. These are computational limits, not physical parameters.

### Finite-level arithmetic reuses the accumulated nodal equation

For finitely many supplied represented pressures, choose a common exact
denominator `Q` and write `p_j=z_j/Q`. With a common fixed positive step and
capacity, a zero-area word has nonnegative integer counts `n_j` satisfying

```text
sum_j n_j*z_j = 0,    T = sum_j n_j > 0.
```

Let `d=gcd(z_j-z_0)`. If `d>0`, the first equation implies
`d | T*z_0`, so every possible length is divisible by

```text
L = d / gcd(d,z_0).
```

This is a necessary length constraint. For three or more levels it need
not be an attainable minimum; `L=1` does not establish a one-step return.
For example, the levels `(-2,3,4)` give `L=1`, but no level is zero.
When `d=0`, identical nonzero levels exclude every nonempty zero-area
word, while identical zero levels permit every length. Strictly one-sign
inputs also exclude zero area. A selectable zero level or opposite signs
permit some scalar zero-area multiset, without proving that its ordering
can be realized by canonical pressure refresh and carried transitions.

The shared owner
[`derive_finite_level_nodal_return`](../src/tnfr/physics/nodal_remainder_pressure.py)
retains these cases separately and shares exact integerization with the
two-level result. Neither arithmetic criterion authenticates its source,
changes pressure, executes a long word or proves vector return.

### A new level is reached, after actual node-1 overshoot

Four derived boundaries take `1,5,1,4` steps. Their displayed offsets
around `0.5`, in units `2^-54`, are:

```text
start:  (-1,0,0,-2,4,-3)    m_1=-1
step 1: (-1,0,2,-2,4,-3)    m_1=+1
step 6: (-1,0,0,-2,4,-3)    m_1=-1
step 7: (-1,0,2,-2,4,-3)    m_1=+1
step11: (-2,0,0,-2,4,-3)    m_1=-2
```

The final boundary changes nodes 0 and 2 simultaneously. Its newly
generated node-1 pressure is `-4325765337928681/2^109`. The eleven
executed steps use the old negative level twice and the old positive
level nine times; none uses the newly captured third level.

The node-1 area relative to the B27 endpoint first becomes positive at
new step three, with value `220301106860745/2^110`. No observed prefix
attains exact zero. At step eleven the local node-1 area is
`24049580203736861/2^112`, and its B27-reference net area is
`19871535157506923/2^112`. The local mean is `-17/2^113` and the net
mean is `-3/2^110`. Thus this node has overshot its earlier loss while
the six-coordinate area and its mean still do not vanish. Every prefix
retains `net area = inherited area + local area = displayed change +
carry change` relative to the same reference.

The three captured node-1 levels, on denominator `2^109`, have integers

```text
(-1026366057766984, 5572432502556410, -4325765337928681).
```

Their difference gcd and length divisor are both `3299399280161697`.
The third level therefore leaves the earlier length restriction unchanged.
This conclusion concerns only words using those scalar levels; it decides
neither a larger pressure family nor bounded behavior.

### A different node excludes confinement to the three visible states

Only three distinct displayed states occur in this continuation. In all
three, node 4 has the same displayed value `0.5+4*2^-54` and the same
strictly positive generated pressure

```text
c = 1668868774373469 / 2^105.
```

Apply the existing
[`observe_finite_nodal_pressure_drift`](../src/tnfr/physics/nodal_remainder_pressure.py)
to those three states with functional `ell=e_4`. Its reconstructed node-4
cell has width `W=2^-53`. The exact accumulated nodal equation then gives

```text
N*h*c <= W,
N <= floor(2^56 / 1668868774373469) = 43.
```

Consequently the next, forty-fourth step must leave that finite class or
fail a declared update premise. This covers every admissible incoming
carry in the class, including carries not visited in the finite replay.
It is stronger than a short-period obstruction for this particular class:
its permanent confinement is excluded. It does not establish positive-band
exit, the next actual transition, or absence of a larger correlated trap.
No forty-four-step trajectory is executed to obtain the bound.

The local pressure formula also gives a conditional extension beyond the
three-state list. Holding the displayed stencil `(x_3,x_4,x_5)` at offsets
`(-2,4,-3)` fixes `m_4=-13`. The row-local EPI reducer therefore fixes its
contribution; the fixed phase supplies the same phase contribution, while
unit capacities and regular C6 support give zero frequency and topology
contributions. Arbitrary admissible changes of remote EPI coordinates
`x_0,x_1,x_2` cannot change this pressure. The same node-4 cell width and
positive increment thus give the same 43-step bound while that entire
stencil remains fixed, with the other declared conditions unchanged.
This locality deduction is separate from the executable three-state
certificate. Its next-step disjunction is a stencil change or premise
failure, not a prescribed node transition or full-runtime stability claim.

### Evidence and the next structural gate

[`c6_winding_pressure_levels.py`](../benchmarks/c6_winding_pressure_levels.py)
revalidates the complete B29 report through its B28/B27/B26 inputs and
binds the four historical input byte hashes. Shared cell-exit and inverse
itinerary owners preserve the incoming carry; the signed prefix observer
retains the B27 area reference. Resource exhaustion is reported as censoring,
not as absence of another pressure level. This remains a conditional
numerical path, without live graph or original-preparation reachability.

The next gate is a compatible change of the neighborhood sustaining the
positive node-4 drift, together with the full signed vector-area budget.
The node-1 sign search, first positive episode and first third-level hit
are complete on this retained path. Do not repeat them or attempt the
large arithmetic period. A larger trapping claim must account for the
newly identified drift direction and prove forward inclusion; successful
compensation of one coordinate does not supply that proof.

## 29. Frozen-neighborhood bounds and a censored node-4 sign test

B2.d.31 makes the local-stencil deduction from section 28 executable and
tests its first observed failure from the retained B30 endpoint. The
conditional numerical map keeps the same phase source, unit capacities,
support, coefficients, positive EPI band and `h=1/16`. The incoming carry
and B27 accumulated-area reference are retained.

### Local pressure constancy does not require a frozen whole graph

[`observe_c6_frozen_pressure_stencil`](../src/tnfr/physics/c6_pressure_lattice.py)
binds one row of the actual CPU pressure map to its displayed stencil
`(x_(i-1),x_i,x_(i+1))`. Under the declared fixed conditions, the row-local
EPI and phase readers give the same pressure while that stencil stays
fixed. Remote displayed EPI and carried remainders may change.

For nonzero pressure `p_i`, the accumulated nodal law is
`X_i(n)=X_i(0)+n*h*p_i` over such a prefix. The existing held-cell owner
supplies the selected coordinate's exact directional limit inside its
nearest-even cell intersected with the positive band. Its whole-vector
first-exit time is not the stencil bound: a remote coordinate can leave
its cell while this row stays fixed. The existing finite-drift owner
independently supplies the uniform bound `floor(W/(h*abs(p_i)))` over
all admissible incoming center carries. That closed enclosure can be
conservative at odd ties; the actual-carry bound retains endpoint parity.

Both bounds are conditional deadlines, not predictions of the first
neighborhood change. A neighbor can change earlier, or an update/band
premise can fail. Zero pressure supplies no finite center deadline and
does not establish invariance of the stencil. The observer changes no
state and certifies no graph execution or pressure-sign hit.

At the B30 endpoint, node 4 has

```text
p_4 = 1668868774373469 / 2^105,
r_4 = 4148688737535375 / 2^110,
upper-cell distance = 67908905300392561 / 2^110.
```

Its exact incoming-carry limit is 20 unchanged steps, with deadline 21.
The uniform limit over all carries remains 43, with deadline 44. The
difference is entirely determined by the retained numerical state.

### The neighbor changes first, without reversing the pressure

The declared test has at most eight new boundaries and 256 steps and
stops early if freshly generated node-4 pressure becomes nonpositive.
It reaches the eight-boundary ceiling after `1,5,1,4,1,3,2,1` steps,
eighteen in total. No nonpositive pressure is observed, so the sign test
is censored. No ninth boundary or post-limit step is executed.

The first stencil change occurs at new step 15: node 5 moves from offset
`-3` to `-4`, in units `2^-54` around `0.5`. Nodes 3 and 4 retain offsets
`-2` and `+4`. The node-4 gradient index falls from `-13` to `-14` and
its pressure falls to `1462656319363363/2^105`, still positive. Thus the
neighbor invalidates the original frozen-stencil premise before the
conditional center deadline 21. A stencil exit and a sign exit are
different observations.

The endpoint has offsets `(-2,0,2,-2,4,-4)`. The local mean area is
`-131/(3*2^114)` and its net mean relative to B27 is `-275/(3*2^114)`;
the complete vector also remains nonzero. Node 4 accumulates strictly
positive local area `7355250143423031/2^107`. It has not begun to
compensate its earlier positive area. Its final carry is
`62990689884919623/2^110`. Under the newly frozen stencil, the remaining
center deadline is four steps, with at most three unchanged center steps.
This is an analytic conditional bound, not a continuation after censoring.

### The sign threshold constrains the coupled neighborhood budget

The existing exact row thresholds give nonpositive node-4 pressure only
when `m_4<=-22`. Starting at `m_4=-13` requires

```text
Delta n_3 + Delta n_5 - 2*Delta n_4 <= -9.
```

After the observed neighbor change, eight further gradient-index units
are still needed. This is a necessary displayed-state condition, not a
feasible itinerary or an instruction to modify EPI. In this binade one
upward node-4 float move has `Delta n_4=2`, reducing `m_4` by four if
the neighbors stay fixed. From the original index, even two such moves
would give `m_4=-21`, still positive. The actual coupled transitions and
their carried areas must determine whether the cut can be reached.

The shared
[`observe_nodal_remainder_cycle_gradient`](../src/tnfr/physics/nodal_remainder_pressure.py)
now owns the relation
`Delta m=-2*L_rw*A/delta+2*L_rw*Delta r/delta` used by the earlier
benchmarks. It first checks the complete identity `A=X_after-X_before`.
Taking only the Laplacian would miss an arbitrary uniform error in the
area vector. The previous callers already checked their full prefix
areas elsewhere; centralization makes that obligation explicit in the
reusable owner and preserves their existing report payloads.

### Evidence and next gate

[`c6_winding_pressure_stencil.py`](../benchmarks/c6_winding_pressure_stencil.py)
revalidates B30 through the retained B29/B28/B27/B26 chain, uses only the
shared nodal integrator, and retains whole-itinerary compatibility and
all six signed area budgets. Its ceiling is computational censoring, not
an impossibility theorem for reaching negative pressure. Neither a live
graph nor original-preparation reachability is promoted by this audit.

The next gate is a coupled neighborhood reachability or trapping argument
using the remaining gradient cut, exact carry and signed vector budget.
Simply adding another fixed number of boundaries would not supply that
argument. Pressure reduction alone is insufficient; any candidate bounded
class must account for accumulated drift in every coordinate and verify
its transitions and forward inclusion. Full-runtime admission and
physical correspondence remain separate open obligations.

## 30. A coupled profile, disagreement tube and finite numerical-band horizon

B2.d.32 through B2.d.36 replace another short continuation with five
dependent mathematical checks. They retain the fixed canonical C6 phase
source, unit conductance and capacity, zero Gamma, existing channel weights
and `h=1/16`. Their finite evidence is exactly the eighteen transitions
already retained by B31. No new trajectory or additional boundary is used.
The model concerns the shared carried numerical map with pressure refreshed
from displayed EPI; future live operator admission remains a separate task.

### B32: the actual source determines a centered relative profile

Write `L=L_rw` for the unit-cycle Laplacian, `w=w_epi`, and `A` for the
represented phase contribution returned by the existing CPU pressure
producer. These are inherited coefficients and actual source values, not
an externally imposed compensating pressure. Define arithmetic centering
by `P=I-11^T/6`. The existing forced-support owner solves

```text
w*L*z = P*A,    mean(z) = 0.
```

On unit C6, the reversible metric is `H=2I`, so this is also its existing
metric-centered Poisson profile. The new
[`c6_carried_profile.py`](../src/tnfr/physics/c6_carried_profile.py)
adapts that owner instead of introducing another profile solver. It
rebuilds the source from primitive phase and weight inputs before using
public reference caches.

The source mean is `mean(A)=-1/(6*2^109)`. Thus `z` is a relative spatial
profile with a separate uniform drift; it is not a zero-pressure equilibrium.
At the B31 endpoint, node 4 still has displayed index `m_4=-14`, while
nonpositive pressure requires `m_4<=-22`. With `delta=2^-54` and
`m_i=-2*(L*x)_i/delta`, this requires an increase of `4*delta` in
`[L*(P*x-z)]_4`. The profile expresses the same necessary neighborhood
condition in centered coordinates. It neither constructs a path to the
cut nor changes the actual EPI state.

### B33: exact carried evolution separates shape, rounding and mean

Let `X=x+r`, where `x` is displayed EPI and `r` is the retained numerical
remainder. The actual pressure has the exact decomposition

```text
p = A - w*L*x + eta,
eta = EPI-product rounding error + pressure-assembly rounding error.
```

Here `eta` is obtained from the shared pressure observation. It does not
include an assumed error of future phase evolution. Replaying the shared
nodal step checks `X_next=X+h*p`; substituting `x=X-r` gives

```text
T = I-h*w*L,
X_next = T*X + h*A + h*(w*L*r + eta),
y = P*X-z,
y_next = T*y + h*P*(w*L*r + eta).
```

The independent mean identity is

```text
mean(X_next-X) = h*mean(A) + h*mean(eta),
mean(w*L*r) = 0.
```

The carried readout feedback can therefore affect shape without supplying
a mean correction. The adapter reuses the existing pressure-readout and
forced-profile observers, validates raw carried coordinates, and requires
the complete supplied step to match the numerical kernel. All eighteen
retained steps have zero recurrence and mean residuals. Their signed mean
budget, relative to the B30 endpoint, is

```text
phase source:       -3/2^113,
pressure rounding:  -113/(3*2^114),
carry feedback:      0,
total change:       -131/(3*2^114).
```

This exact finite budget does not establish a bounded infinite signed
mean prefix. In particular, a nonuniform admissible carry can coexist
with uniform displayed EPI and zero pressure in a synchronized control;
the reconstructed centered error then persists. Homogeneous diffusion
contraction alone cannot remove this readout feedback.

### B34: exact C6 contraction distinguishes norm and energy gains

[`c6_carried_tube.py`](../src/tnfr/physics/c6_carried_tube.py)
uses the shared exact cycle Laplacian. A complete rational orthogonal
basis consists of the uniform vector and the following mean-zero vectors:

```text
lambda=1/2: (2,1,-1,-2,-1,1), (0,1,1,0,-1,-1),
lambda=3/2: (2,-1,-1,2,-1,-1), (0,1,-1,0,1,-1),
lambda=2:   (1,-1,1,-1,1,-1).
```

Their exact Laplacian identities prove, without a numerical eigensolver,
that for `s=h*w` with `0<s<1`,

```text
q = max(abs(1-s/2), abs(1-3*s/2), abs(1-2*s)) < 1,
||T*y||_2 <= q*||y||_2          when mean(y)=0.
```

The sharp squared-energy gain is `q^2`, whereas the uniform vector has
gain one. The retained configuration gives
`q=573161353023261791/2^59`. This is a property of the exact represented
coefficients and fixed support, not a measured decay fit. The statement
concerns the homogeneous map; the actual carried recurrence still has
the additive term from B33.

### B35: uniform numerical bounds give a conditional disagreement tube

All bounds are derived from the declared numerical band and coefficients,
without extrapolating observed error maxima. Let its width be `D=upper-lower`,
let `u=2^-53` and `t=2^-1075`, and write `Amax=max_i abs(A_i)`. The local
slab theorem gives the exact gradient `g=-L*x` with `abs(g_i)<=D`.
Under nearest-even binary64 arithmetic, the product and assembly errors
therefore satisfy

```text
e1 = u*w*D + t,
e2 = u*(Amax+w*D+e1) + t,
abs(eta_i) <= epsilon = e1+e2.
```

The implementation checks finite-operation envelopes before applying
these bounds. The half-subnormal term retains underflow cases. With
`R=ulp(upper)/2`, every admissible carry obeys `abs(r_i)<=R`; on the
retained band `[3/8,5/8]`, `R=2^-54`. Consequently

```text
B = 2*w*R + epsilon,
abs((w*L*r+eta)_i) <= B,
||P*(w*L*r+eta)||_2^2 <= 6*B^2.
```

For `E=||P*X-z||_2^2`, Young's inequality supplies

```text
E_next <= q*E + 6*h^2*B^2/(1-q),
E_floor = 6*h^2*B^2/(1-q)^2,
E_bar = max(E_initial,E_floor).
```

The affine bound uses `q`, not the homogeneous squared gain `q^2`.
It gives a forward-invariant disagreement envelope while the numerical
band and fixed-source premises hold. A persistent nonzero floor is not
an asymptotic-convergence claim. Separately, the exact zero carry mean gives
the per-step bound

```text
abs(mean(X_next-X)) <= b = h*(abs(mean(A))+epsilon).
```

For the retained prefix, `E_initial` is approximately
`8.781440801771191e-32`, `E_final` is approximately
`6.230400442940049e-32`, and `E_floor` is approximately
`6.656013887802292e-31`. The mean-increment bound is approximately
`6.354411872153111e-19`. The code retains exact fractions; these decimals
are summaries. The observed energy decrease neither removes the uniform
floor nor proves future mean cancellation.

### B36: close a finite band premise and assess the remaining cut

The disagreement bound also supplies `abs(y_i)^2<=5*E_bar/6`. Let
`mu_0=mean(X_0)` and define the minimum initial margin using the carried
state's own band, including when it is narrower than the pressure slab:

```text
a = min_i(mu_0+z_i-lower, upper-mu_0-z_i).
```

An initially admitted tube remains inside that band through every integer
prefix `n` satisfying

```text
a-n*b >= 0,    (a-n*b)^2 >= 5*E_bar/6.
```

This closes the premise by induction rather than assuming the conclusion.
An admitted current state supplies the pressure and carry bounds; these
bounds first place the exact next candidate inside the band. Monotonic
nearest-even rounding keeps its displayed coordinate inside the binary64
endpoints, and the shared dyadic remainder representation remains valid.
For `b>0`, an exact monotone binary search finds the largest integer
satisfying the sufficient inequalities. An admitted tube with `b=0`
would instead have an unbounded conditional prefix; a rejected initial
tube supplies neither conclusion.

The retained inputs give the maximal sufficient integer horizon
`N=196713720348826219`. That horizon is derived, not executed. The next
integer fails the sufficient inequality; this is not a prediction of
actual band exit. The result certifies the fixed carried numerical map
through a finite horizon, not live grammar, future operators, changing
phases or support, original-preparation reachability, or infinite stability.

For the node-4 sign question, put `m_i_star=-2*(L*z)_i/delta` and let `k`
be its nonpositive-pressure cut. Since a C6 Laplacian row has squared norm
`3/2`, reaching `m_i<=k` requires

```text
delta*(m_i_star-k)/2 <= sqrt(3*E_bar/2) + 2*R.
```

The implementation uses rational comparisons: a positive remaining
distance `d=delta*(m_i_star-k)/2-2*R` with `d^2>3*E_bar/2` excludes
the cut under the tube premises. The actual node-4 assessment is
inconclusive. Failure to exclude the cut does not prove a feasible
transition, compensation or trapping class; the current index remains
`-14` against the threshold `-22`.

[`c6_winding_coupled_budget.py`](../benchmarks/c6_winding_coupled_budget.py)
revalidates the six retained B31-through-B26 reports, checks the complete
eighteen-step pressure and carry history, and binds every observed defect,
energy and signed mean contribution to these shared owners. It retains
the B30 endpoint as the envelope origin and the B31 endpoint as the
current state. Reproduction uses

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_coupled_budget.py
```

The next work is to tighten the signed coupled mean/shape budget and its
correlations with the reachable integer-gradient states. The absolute mean
bound grows linearly and the present tube leaves the sign cut undecided.
Another arbitrary short continuation would not close either obligation.
A stronger result must derive cancellation, a tighter admissible set or a
controlled exit from the canonical pressure law and retained carry, without
fitting a new coefficient or inferring an infinite claim from finite data.

## 31. B37: closing the spatial and rounding bounds on each other

The band-wide B35 estimate controls an entire interval of EPI values, while
the carried trajectory is restricted by its centered profile. B2.d.37 uses
that relation to close a sharper bound analytically. The existing source,
support, capacity, timestep and pressure producer remain unchanged. Write
`E=||y||_2^2`, `y=P*(x+r)-z`, `G=max_i abs((L*z)_i)`, and retain
`R=ulp(upper)/2`, `u=2^-53`, `t=2^-1075`.

A row of the unit C6 Laplacian has squared Euclidean norm `3/2`.
Consequently, the actual displayed gradient satisfies

```text
abs((L*x)_i) <= G + sqrt(3*E/2) + 2*R.
```

Applying the same product and assembly rounding bounds as B35 gives

```text
a = u*(2+u)*w*(G+2*R) + u*max_i abs(A_i) + (2+u)*t,
b = u*(2+u)*w,
abs(eta_i) <= epsilon(E) = a + b*sqrt(3*E/2).
```

The centered projector is nonexpansive and `||L*r||_2<=2*sqrt(6)*R`.
The exact recurrence from B33 and the contraction from B34 therefore imply

```text
sqrt(E_next) <= Q*sqrt(E) + h*sqrt(6)*(2*w*R+a),
Q = q + 3*h*b.
```

The factor `3` is the exact identity `sqrt(6)*sqrt(3/2)=3`. For `Q<1`,
the closed invariant energy envelope is wholly rational:

```text
E_floor_closed = 6*h^2*(2*w*R+a)^2/(1-Q)^2,
E_bar_closed = max(E_initial,E_floor_closed).
```

This is an algebraic closure, without repeated numerical fitting of an
error estimate. The condition `Q<1` is checked separately from `q<1`:
a homogeneous map arbitrarily close to its noncontracting boundary need
not pass this sufficient rounding-feedback test. The new owner
[`c6_carried_closure.py`](../src/tnfr/physics/c6_carried_closure.py)
rebuilds the profile and initial carried state through existing owners,
checks finite-operation envelopes, and retains large initial disagreement
instead of replacing it by the smaller floor.

At the retained B31 endpoint, the closed energy envelope is approximately
`2.958228394578814e-31`, compared with the earlier
`6.656013887802292e-31`. Its combined per-coordinate rounding bound is
approximately `6.731922543446726e-32`. These are summaries of exact
rational results. In particular, the sharper error bound is derived from
the structural profile, carry and binary64 format rather than from the
largest error observed in a finite trajectory.

For each node, put `m_i_star=-(L*z)_i/(delta/2)`. The necessary gradient
condition is

```text
abs((delta/2)*(m_i-m_i_star)) <= sqrt(3*E_bar_closed/2)+2*R.
```

Exact rational square comparisons and monotone integer searches produce
the following integer hulls at the B31 origin:

| Node | Necessary integer gradient interval |
|------|-------------------------------------|
| 0 | `[-26,29]` |
| 1 | `[-28,27]` |
| 2 | `[-33,22]` |
| 3 | `[-17,38]` |
| 4 | `[-49,6]` |
| 5 | `[-13,42]` |

The tests verify both admitted endpoints and rejected adjacent integers.
The intervals are coordinate conditions; their Cartesian product is not
a set of jointly reachable states. Node 4's cut `m_4<=-22` remains inside
its interval, so the closure alone neither excludes nor establishes it.

The signed mean interval remains separate:

```text
h*(mean(A)-epsilon_bound) <= mean(X_next-X)
                         <= h*(mean(A)+epsilon_bound).
```

It still straddles zero. The carry term contributes exactly zero mean;
sharper spatial control cannot be relabeled as mean cancellation. This
envelope remains conditional on the fixed numerical source and admitted
band. The finite band bootstrap in section 30 remains available, but
neither result alone proves infinite containment or live operator admission.

## 32. B38: static compensation refutes a class-wide linear drift argument

B2.d.38 asks whether a fixed linear functional could separate all
canonical pressure vectors in the closed envelope from zero. The answer
is negative for the admitted class, including one slice with exactly the
same reconstructed mean as the B31 origin.

The new observer
[`c6_carried_balance.py`](../src/tnfr/physics/c6_carried_balance.py)
admits detached numerical states only after revalidating their carried
encoding, common declared band, centered energy and freshly recomputed
canonical pressure. The seven displayed EPI vectors in the witness are
`.5+delta*n`, with `delta=2^-54` and the following exact offsets:

```text
(-6,-1,2, 2,8,-7),
(-5, 2,4,-2,6,-7),
(-4,-2,0, 2,6,-4),
(-4,-1,2,-2,8,-5),
(-3, 0,2,-2,8,-7),
(-2,-1,4,-2,6,-7),
(-2, 2,0,-2,6,-6).
```

Every row has offset sum `-2`. Assigning each coordinate the same
admissible carry `96076792050570541/2^113` gives all seven states the
same exact reconstructed mean as the retained B31 origin. This constructs
static comparison states; it does not reset or change the carry on the
actual trajectory. All seven centered energies lie below `21*delta^2`
and satisfy the rebuilt B37 envelope.

Let their freshly computed pressure vectors be `p_1,...,p_7`. The shared
exact linear-algebra owner inverts the rational matrix whose columns are
`(p_j,1)` and derives unique weights satisfying

```text
lambda_j > 0 for every j,
sum_j lambda_j = 1,
sum_j lambda_j*p_j = (0,0,0,0,0,0).
```

The coefficients are an algebraic certificate, not a new TNFR parameter,
probability distribution or dynamical mixing rule. If one fixed linear
functional `c` had a strict common sign on every admitted pressure vector,
the same sign would hold for their positive weighted sum, contradicting
the exact zero vector above. Hence a strict class-wide linear drift
separator cannot prove escape on this envelope or its demonstrated
constant-mean slice.

This result supplies no temporal ordering of the points, no transition
compatibility of their carries, and no reachability from the B31 state.
A smaller reachable subset can still have a different drift constraint.
In particular, a static convex cancellation is not an accumulated zero
vector area or a periodic orbit. The obstruction redirects the finite
sign question toward the actual nodal recurrence rather than treating
the source's negative mean as a universal pressure-mean sign law.

## 33. B39: a finite first-passage theorem for the pressure cut

The coupled spatial envelope does more than constrain static gradients.
Together with the nodal equation, it excludes indefinitely positive
pressure at node 4 while its mean drift remains much smaller than the
least possible positive nodal pressure. B2.d.39 proves this before
executing a new continuation.

The owner
[`c6_carried_passage.py`](../src/tnfr/physics/c6_carried_passage.py)
rebuilds the earlier B35 tube and its independently proven B36 band
horizon. It uses the sharper profile identity

```text
abs((L*x)_i) <= abs((L*z)_i)+sqrt(3*E_bar/2)+2*R
```

to derive row-specific product and assembly rounding envelopes. Their
average gives an upper bound `M` on `mean(p)`; zero-mean carry feedback
does not contribute to `M`. The use of the already certified tube and
band horizon makes the finite premise independent of the desired sign
change. The new B37 closure is useful additional information but is not
required to promote this particular first-passage certificate.

The shared integer sign-sector owner gives `p_i>=p_min>0` whenever the
selected pressure is positive. Its integer sector is relaxed with respect
to joint coordinate realizability, which makes the pressure lower bound
valid without fabricating a graph state. At node 4, the smallest positive
sector index is `-21`, with

```text
p_min = 38338268585241/2^106,
M approximately 6.242107045064891e-32,
v = h*(p_min-M) > 0.
```

The carried nodal equation gives exactly

```text
y_i(next)-y_i = h*(p_i-mean(p)).
```

If every pressure readout in the first `N` transitions were positive,
then `y_i(N)>=y_i(0)+N*v`. But the tube imposes
`abs(y_i(N))^2<=5*E_bar/6`. The least integer `N` for which

```text
y_i(0)+N*v > 0,
(y_i(0)+N*v)^2 > 5*E_bar/6
```

therefore yields a contradiction, provided the independent band horizon
covers all `N` transitions. Exact monotone search gives `N=30255`, well
inside the previously proved finite band horizon. Thus some pressure
readout with index at most `30254` must be nonpositive; index zero is the
B31 endpoint. This resolves the finite conditional reachability question
for the numerical map. It does not assert the first actual hit, infinite
trapping, original-preparation reachability or future live graph admission.

## 34. B40: the realized first passage and a finite mean-budget reversal

The finite theorem supplies a justified stopping rule for a new
continuation. B2.d.40 resumes the exact B31 endpoint, including all six
retained carries. It keeps the closed phase source, unit C6 support and
capacity, existing channel weights and `h=1/16`. Every step uses the
shared carried nodal kernel; pressure is refreshed whenever the displayed
state changes. There is no pressure projection, carry reset or fitted
coefficient.

[`c6_winding_coupled_passage.py`](../benchmarks/c6_winding_coupled_passage.py)
revalidates the existing source chain and binds the B37 closure, B38
static witness, B39 passage theorem and B40 realized continuation to their
shared owners. The continuation stops at the first nonpositive node-4
pressure, with the derived B39 bound as its maximum stopping budget.
This replaces the earlier arbitrary boundary ceiling by a decisive test.

The first hit occurs at pressure-readout index `118`, after `118` shared
transitions and `59` displayed-state boundaries. Readout zero is the
inherited B31 endpoint; the source readouts at indices `0,...,117` all
have positive node-4 pressure. The fresh endpoint readout gives

```text
displayed offsets from .5 in delta=2^-54:
    (-3,0,2,-1,8,-5),
integer gradients:
    (1,-1,-5,12,-22,15),
p_4 = -187043320717485/2^105 < 0.
```

Thus the pending `m_4<=-22` cut is reached by the actual finite carried
map, rather than merely admitted by a coordinate bound. The hit at
index `118` lies within the proven maximum index `30254`. This resolves
the previously censored finite sign question on its declared numerical
domain. The certificate still concerns the supplied endpoint and source;
it does not reconstruct the endpoint from the original full preparation
or certify future live operator admission.

The continuation also shows that its signed mean pressure need not retain
the earlier negative sign. Relative to the explicit B27 endpoint reference,
the accumulated mean area immediately before transition `59` is
`-1/(3*2^114)`. Immediately afterward it is `1/2^113`. This is the first
crossing in the new continuation: the prior negative mean budget has been
overcompensated. It is not an exact zero-vector return or an assertion
that some sampled mean prefix equals zero.

At the pressure-cut endpoint, the exact local mean budget relative to the
B31 origin is

```text
phase-source contribution:       -59/(3*2^113),
pressure-rounding contribution:  305/(3*2^113),
carry-feedback contribution:       0,
total local mean area:             41/2^112.
```

Including the inherited B27-referenced budget gives the final accumulated
mean area `217/(3*2^114)`. The full coordinate budget is retained and all
six B27-referenced net areas remain nonzero. The finite mean reversal
therefore establishes actual temporal compensation of the scalar mean
budget while leaving complete vector recurrence unresolved. In particular,
the negative fixed source mean is overcome here by signed pressure-rounding
contributions; a zero-mean carry feedback is not credited with that change.

Reproduction uses the consolidated benchmark:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_coupled_passage.py
```

The next unresolved question is whether the reachable carried states admit
a bounded signed vector budget and forward inclusion for arbitrarily long
evolution. B38 rules out one class-wide linear separation strategy, while
B39-B40 resolve one finite pressure cut and a finite scalar compensation.
These results do not prove infinite trapping, a periodic orbit, convergence,
complete-runtime stability or physical correspondence. Any broader claim
must retain the actual transition order, pressure refresh, all coordinates
and carry compatibility rather than replacing them by static convex weights
or another unmotivated extension of the execution horizon.

## 35. B41: complete carry cells cannot provide invariant trapping

B2.d.41 tests a candidate class rather than extending the saved B40
trajectory. A displayed six-tuple fixes the canonical pressure when phase,
support, capacity and channel weights are held as declared. Its complete
carry fiber consists of every legal exact encoding `X=x+r` that rounds to
that tuple and remains in the positive band. A finite family of displayed
tuples can have arbitrary spatial correlations while still assigning the
complete independent carry fiber to each tuple.

The new owner
[`c6_carried_cell_escape.py`](../src/tnfr/physics/c6_carried_cell_escape.py)
proves that no nonempty finite family of this kind can be forward invariant
when its displayed integer gradients satisfy B37's bounds. This strengthens
the earlier Cartesian obstruction: the visible tuples need not form a
Cartesian product. It does not exclude a family that restricts carry jointly
with visible shape or reconstructed energy.

The proof uses the existing nodal update, pressure lattice and inverse
rounding-cell itinerary. Let `g=2^-3222` be the shared exact encoding grid
and `delta=2^-54` the local EPI quantum. Band endpoints are represented;
every band-clipped nearest-even cell has width at least `delta/2`. Removing
open tie endpoints reduces the distance between its first and last legal
grid points by at most `2*g`. The explicit remainder magnitude bound is
inactive on these cells. Monotonicity of the source-plus-rounded-product
pressure law supplies exact pressure bounds from the endpoints of B37's
integer-gradient intervals. The largest possible negative increment obeys

```text
max_i max(-h*p_i,0)/delta = 5757725321284133/2^55
                         < 0.16 < 1/2,
max_i max(-h*p_i,0) < delta/2-2*g.
```

These are bounds over the declared gradient class, not maxima fitted to a
trajectory. The pressure lattice also proves that every displayed tuple in
the slab has at least one positive pressure component: an all-nonpositive
vector would require integer-gradient sum at most `-4`, whereas the cycle
identity makes that sum exactly zero.

For an arbitrary nonempty finite candidate family, choose a displayed
tuple maximizing `sum_i x_i`. In each coordinate, select the greatest
admissible grid point of its band-clipped rounding cell. This simultaneous
choice belongs to the complete carry fiber. Every negative increment is
smaller than the cell's available grid width, so its displayed coordinate
stays fixed. Every positive increment lies on the same exact grid and
therefore moves beyond the greatest admissible grid point. Its displayed
coordinate must increase, or its exact candidate must leave the declared
band. Consequently the witness either has strictly larger displayed sum
than every tuple in the family or fails band admission. Both outcomes
contradict forward invariance of that full-cell union.

The executable witness uses B38's seven displayed rows together with the
B40 displayed endpoint. The maximum-sum row is the eighth entry, index `7`.
The hypothetical greatest-carry state produces

```text
source offsets:   (-3,0,2,-1,8,-5),
endpoint offsets: (-3,0,4, 0,8,-4),
increased nodes:  (2,3,5),
increase in displayed sum: 4*delta = 2^-52.
```

The source pressure is freshly recomputed and the admissible endpoint is
verified by the shared carried nodal kernel. The selected carry is an
existence witness from a complete fiber; it is not B40's retained carry,
is not written into a live graph, and is not claimed reachable from B40.
Its reconstructed energy need not remain in the smaller B37 energy set.
Thus B41 proves that **some legal carry escapes each candidate full-cell
union**. It does not prove that every carry escapes, that the saved
trajectory leaves the band, or that all correlated invariant regions fail.

## 36. B42: every carry leaves the seven static compensation cells

B2.d.42 addresses the actual temporal compatibility of B38's seven
displayed compensation rows. The owner
[`c6_carried_cell_graph.py`](../src/tnfr/physics/c6_carried_cell_graph.py)
constructs their complete finite transition graph using the fixed canonical
pressure source, unit capacity, `h=1/16` and the declared band
`[.375,.625]`. Each of the `7*7=49` directed edge tests asks whether any
legal incoming carry can execute that one transition. The shared inverse
itinerary intersects translated nearest-even cells exactly, including tie
parity, band clipping and the `2^-3222` encoding grid.

All seven rows have a self-loop. The only nonself edge is from row `6`
to row `5`, using the one-based order in section 32; in zero-based indices
it is `5 -> 4`. Removing self-loops therefore leaves a directed acyclic
graph. An edge is existential: consecutive edge witnesses need not carry
the same incoming residual. The graph is an overapproximation of actual
carried trajectories, which is sufficient for an upper escape deadline.
It must not be interpreted as a realizable schedule of arbitrary edges.

For a fixed displayed row, pressure remains constant until that row changes.
To maximize residence, choose each positive-pressure coordinate's first
legal grid point and each negative-pressure coordinate's last. These
independent choices simultaneously maximize every directional residence
limit. If a coordinate's legal grid interval is `[a_i,b_i]` and its exact
increment is `g*k_i`, its maximal unchanged prefix is
`floor((b_i-a_i)/abs(k_i))` when `k_i` is nonzero; zero increments impose
no finite limit. The shared cell-horizon owner verifies these bounds and
their nearest-even endpoint convention. Taking the minimum across
coordinates gives the sharp maximal unchanged residence for that row.

Reverse topological induction then bounds residence in the entire family:

```text
first_exit_i = maximal_unchanged_steps_i + 1,
D_i = first_exit_i + max(D_j : i -> j, j != i),
```

with the maximum over an empty successor set equal to zero. The first
exit step includes the transition into a successor, if one occurs; the
successor's bound starts from its resulting carried state. Enlarging the
set of possible paths by ignoring carry compatibility between edges can
only make this upper bound more conservative.

| B38 row, one-based | Maximum unchanged steps | First exit from its cell | Upper deadline to leave the family or fail band admission |
|---|---:|---:|---:|
| 1 | 76 | 77 | 77 |
| 2 | 50 | 51 | 51 |
| 3 | 38 | 39 | 39 |
| 4 | 50 | 51 | 51 |
| 5 | 39 | 40 | 40 |
| 6 | 29 | 30 | 70 |
| 7 | 49 | 50 | 50 |

Every legal carried trajectory starting in these seven complete cells must
therefore leave this family or fail band admission within `77` steps,
conditional on the fixed numerical map. This excludes a periodic carried
orbit confined to those seven displayed rows. A larger band may retain
the trajectory after it leaves the family; no whole-band exit is proved.
Nor does this result prove that B40's saved state ever enters the seven
cells. Unlike B41's existence of an outward carry, B42's deadline covers
**every legal incoming carry in this particular seven-cell family**.

The positive static convex pressure balance of section 32 remains correct.
It cannot supply a closed temporal class using its seven displayed points:
exact carry compatibility and residence expose the missing dynamical
condition. A cyclic transition graph or a stationary-pressure row would
make this sufficient deadline test abstain, not certify recurrence.

The consolidated producer
[`c6_winding_temporal_compatibility.py`](../benchmarks/c6_winding_temporal_compatibility.py)
replays the retained source chain, rebuilds the B41 bounds and witness, and
checks the B42 graph without advancing the saved B40 endpoint. Reproduction:

```powershell
.venv313\Scripts\python.exe -X utf8 benchmarks/c6_winding_temporal_compatibility.py
```

B41 already rules out every finite full-cell union in its bounded-gradient
class, so searching a larger family of complete carry cells is not the next
invariance strategy. The remaining candidate must restrict translated carry
subcells jointly with visible shape and signed accumulated vector area.
It needs forward inclusion over its entire declared domain, or a signed
recurrent-budget obstruction. Neither a static convex cancellation nor
another longer sampled trajectory supplies that proof.

## 37. B43: a finite live bridge from the original winding preparation

The phase projection in section 23 and the carried pressure studies address
different provenance obligations. The former reaches an exact numerical
phase tail using detached proposal inputs; the latter evolves explicitly
supplied EPI and carry under that fixed source. Neither fact alone supplies
the EPI/carry reached when the original graph preparation executes the
complete operator word. B2.d.43 addresses that finite bridge through the
existing carried event executor.

The declared preparation is the original null winding on ordered unit C6:
`EPI_i=.5`, `nu_i=1`, stored phases `i*pi/3`, seed `17`, and canonical
default channel weights, with Gamma disabled as in the existing study.
The declared numerical band is `[.375,.625]`, inside the configured graph
bounds. The finite word and its flow partition are

```text
(UM IL)^89 SHA,
after each IL: one interval of duration 1/4,
each positive interval: four refreshed carried steps of duration 1/16,
no flow between UM and IL or after terminal SHA.
```

The budget therefore specifies `179` operator events and `356` numerical
steps, ending at represented time `22.25`. It targets entry into the phase
tail already identified by the detached map, rather than adding an
arbitrary trajectory length. Complete-word grammar admission is checked
before execution. The whole invocation uses one graph-owned transaction;
failure rolls back the graph and its carried encoding.

Two implementation boundaries matter. A fresh graph receives a zero carry
encoding from its actual initial EPI. An existing nonzero encoding can be
continued only through its intact graph-owned binding. Consequently B43
does not attach a detached B40 carry to a new graph or bypass the executor
seal. Moreover, canonical Silence attenuates capacity. The final SHA is
placed after all unit-capacity measurements; unit capacity is not claimed
for a subsequent invocation on the closed graph.

### Actual phase evidence accompanies each finite nodal record

The shared
[`nodal_remainder_runtime.py`](../src/tnfr/operators/nodal_remainder_runtime.py)
already retains actual pressure, EPI, capacity, carry, support and clock
evidence. Its nodal-flow snapshot intentionally has no phase channel. Since
pressure is not an injective phase readout, reconstructing a phase path
from its pressure values would leave a provenance gap.

Flow and event records now additionally capture the ordered stored
`phase_before` and `phase_after` tuples. These raw binary64 values, including
signed zero, enter the existing complete execution seal. No phase formula,
operator behavior, pressure law or integration step changes. Tests verify
actual two-cycle UM/IL proposals against the existing phase owner, unchanged
phases during flow and SHA, and loss of certification if any retained phase
field is tampered with.

The finite adapter
[`c6_winding_phase_tail_runtime.py`](../benchmarks/c6_winding_phase_tail_runtime.py)
compares every actual UM/IL capture with the shared phase projection and
checks every refreshed pressure against its captured phase and displayed
EPI. It separately verifies unit capacity for all measured flows and exact
preservation of EPI/carry across each named event. It retains both the state
immediately after the last IL, before its four flow segments, and the state
immediately before SHA. The former is the appropriate input for a new
fixed-source carried profile and self-consistent closure.

### Completed finite capture and remaining scope

The complete 89-cycle invocation succeeds with `179` admitted operator
events and `356` refreshed carried steps. Its executor seal is valid at
capture. Every observed UM/IL transition matches the shared phase owner,
and every measured flow has unit capacity and the recorded phase-bound
canonical pressure. The final phase tuple repeats exactly under the
detached composite UM/IL projection. This establishes entry into that
phase tail on an actual finite graph-owned word.

The two relevant captured states are distinct. Their displayed offsets
from `.5`, in units `delta=2^-54`, are

| Live boundary | Time | Displayed offsets |
|---------------|------|-------------------|
| Immediately after the last IL, before its flow | `22` | `(-3,0,2,-1,6,-5)` |
| After its four carried segments, before SHA | `22.25` | `(-3,0,2,0,8,-5)` |

Both states retain their actual nonzero carry vectors in the result.
The first supplies the fixed-phase profile and B37 closure directly,
without rewriting its numerical band or substituting a detached carry.
Its closed energy envelope is approximately `2.958228394578814e-31`.
The complete invocation's mean nodal area from the original preparation
is `3293/(3*2^114)`. All six exact nodal balance residuals vanish.
An independent replay of all `356` recorded steps, their signed areas and
the extracted closure agrees with the retained result.

Terminal SHA preserves phase, EPI and carry but changes every capacity
from `1` to the represented value `0.9204225284540524`. The final live
graph therefore does not retain the unit-capacity premise of the preceding
measurements. A later invocation must use its actual capacity; the historic
pre-SHA snapshot is not a mutable live continuation checkpoint.

The completed artifact retains its scientific source digest. A subsequent
input-hardening correction makes scalar captures resolve exact-string
aliases through the runtime's nonvirtual mapping reader. It rejects
string-subclass core aliases rather than silently substituting a default,
and prevents mapping read hooks from making uncertified auxiliary writes
during phase observation. The campaign uses ordinary canonical dictionaries;
its preserved source archive and independent verification retain the
precise provenance of the completed run. Serialized records do not recreate
the original executor seal or acquire a later source identity.

B43 closes original-preparation reachability of these finite causal tail
states under the declared carried, refreshed solver. This solver is distinct
from older held-pressure visible-EPI experiments. The displayed states and
their carries also differ from B40's conditional trajectory, so they cannot
retroactively authenticate its endpoint or replace that historical branch.
Subsequent use of the extracted fixed-phase closure remains conditional on
its declared source and capacity. Indefinite trapping, future complete-word
admission and empirical correspondence remain open.

## 38. B44: a bounded mean interval does not close the centered-energy tube

B2.d.44 tests a specific candidate for joint trapping from B43's actual
pre-SHA state. It adds a bounded reconstructed-mean interval to the existing
centered-energy envelope. This trims the complete carry fibers excluded by
B41 and retains the exact numerical encoding. The new result nevertheless
refutes forward invariance for every such interval inside one derived local
window containing the actual initial mean. It supplies counterexamples to
that proposed region, not an escape claim about the actual starting carry.

The owner
[`c6_carried_mean_cylinder.py`](../src/tnfr/physics/c6_carried_mean_cylinder.py)
reuses the canonical pressure, forced profile, self-consistent closure,
inverse nearest-even cells and carried nodal kernel. No pressure source,
coefficient, capacity or numerical step is adjusted. Its model remains the
conditional fixed-phase, unit-capacity map with `h=1/16`; the terminal SHA
in B43 does not supply this capacity for a future live invocation.

### Candidate region and actual starting-state membership

Let `X=x+r`, `P*X=X-mean(X)*1`, and let `z` solve the existing centered
Poisson identity. Rebuild the B37 bound `E_bar` at B43's actual pre-SHA
endpoint and write

```text
E(X) = ||P*X-z||^2,
mu0 = 1/2 + 3293/(3*2^114),
delta = 2^-54,
g = 2^-3222.
```

The candidate `K[a,b]` consists of every legal carried encoding in the
declared band `[.375,.625]` satisfying both `E(X)<=E_bar` and
`a<=mean(X)<=b`. Its mean endpoints are bounds on a proposed certificate
domain, not new dynamical parameters. The actual B43 endpoint satisfies
the energy bound and belongs whenever `a<=mu0<=b`. Its energy is about
`2.82386869748*delta^2`, while `E_bar` is about
`96.00000000000064*delta^2`.

Two existing displayed rows from B38 provide independent static templates:
the first row, with negative mean pressure, and the fourth, with positive
mean pressure. Their offsets from `.5` are

```text
negative template: (-6,-1,2, 2,8,-7),
positive template: (-4,-1,2,-2,8,-5).
```

Both have displayed mean `m_x=1/2-delta/3`. Giving each coordinate the same
carry

```text
c0 = mu0-m_x = 384307168202283423/2^114
```

places both templates at `mu0`, with a legal dyadic encoding. Their exact
centered energies satisfy the rederived envelope; the ratios to `delta^2`
are approximately `12.1297581819` and `2.88720884922`, respectively. These
are hypothetical comparison states. This construction neither changes
B43's retained carry nor claims that its trajectory reaches a template.

### Exact common translation window

The shared inverse-cell owner derives every legal grid endpoint, including
nearest-even parity and the physical band. For these two displayed tuples,
the largest common interval of equal coordinate carries is exactly

```text
-delta/2 + g <= c <= delta/2 - g,     c on the g-grid.
```

The subtraction of one grid quantum at each end follows from open odd-tie
boundaries in the common intersection; it is not a fitted tolerance. Thus
the common mean window for these uniform translations is

```text
W_lower = 1/2 - 5*delta/6 + g,
W_upper = 1/2 + delta/6 - g.
```

The actual `mu0` lies strictly inside it. Relative to `mu0`, the window is

```text
W_lower-mu0 = -5*delta/6 - 3293/(3*2^114) + g,
W_upper-mu0 =    delta/6 - 3293/(3*2^114) - g.
```

This is the maximal *shared uniform-translation window for the chosen
templates*, not a maximal domain for all possible pressure states.
Translation by any admissible `k*g*1` keeps each displayed row, and hence
its refreshed canonical pressure, unchanged. It also leaves centered
energy unchanged exactly, because `P*(X+k*g*1)=P*X`.

### Outward witnesses for arbitrary rational mean endpoints

For the selected negative and positive templates, fresh canonical pressure
gives the exact mean increments

```text
mean(p_negative) = -1/2^110,    h*mean(p_negative) = -1/2^114,
mean(p_positive) =  3/2^110,    h*mean(p_positive) =  3/2^114.
```

Both signed magnitudes exceed `g`. Consider any rational interval satisfying
`W_lower<=a<=mu0<=b<=W_upper`. Its endpoints need not lie on an encoding
grid. Choose the uniform template translations

```text
k_upper = floor((b-mu0)/g),
k_lower = ceil((a-mu0)/g).
```

Since the interval contains `mu0`, the translated positive template has
mean between `mu0` and `b`, and the translated negative template has mean
between `a` and `mu0`. The common translation window makes both carried
states legal. Their centered energies still satisfy `E_bar`, so both
belong to `K[a,b]`. Their distances to the respective mean boundaries obey

```text
0 <= b-(mu0+k_upper*g) < g,
0 <= (mu0+k_lower*g)-a < g.
```

Applying the actual nodal increments therefore sends the positive witness
strictly above `b` and the negative witness strictly below `a`. Each valid
endpoint is verified through the shared carried kernel. If an exact
candidate leaves the physical band, it already violates the candidate
domain and is recorded as band-admission failure without fabricating a
valid endpoint. For the two declared templates and their common window,
the coordinate bounds keep these candidate endpoints in the band; their
mean inequalities provide the escape.

This proof uses a uniform-translation sublattice of admissible means. It
does not assume that every legal reconstructed mean is attainable by
uniform dyadic shifts: the full mean grid is finer. A witness within less
than `g` of each boundary suffices because its signed increment is larger
than that gap. No grid enumeration, statistical mean argument or arbitrary
long trajectory is needed.

### Scope of the obstruction and the next dependency

`derive_c6_carried_mean_cylinder_obstruction` rebuilds the closure and both
templates, checks their common initial mean and energies, recomputes their
pressure and derives the common legal translation window. The paired
`observe_c6_carried_mean_cylinder_escape` constructs and verifies the two
outward witnesses for a supplied admissible rational interval. Derived
caches, static convex weights and endpoint equality are not substituted
for these checks.

B44 excludes the natural centered-energy tube closed by any independent
mean interval in this local window containing B43's actual mean. Its
counterexamples remain hypothetical states inside that proposed region;
the theorem does not establish their reachability from B43 or mean escape
of the B43 trajectory. The candidate does not impose additional
coordinatewise congruences inherited from the initial state and its
allowable nodal increments. Legal encoding alone does not establish
membership in that finer reachable arithmetic class. Wider mean windows, shape-dependent mean limits,
and joint restrictions on individual carries remain unresolved. It proves
neither whole-band exit nor an infinite stable regime.

Within the tested local window, a successful invariant certificate must
therefore distinguish the pressure signs through additional correlations
between shape, carry and mean, or demonstrate a smaller reachable domain
that excludes the outward templates. Translation-invariant centered
energy plus an independent mean bound is insufficient. Any subsequent
piecewise domain still needs exact initial membership and universal
forward inclusion of its translated pieces, with the full signed vector
budget retained.

## 39. B45: the coordinate arithmetic class does not rescue local mean confinement

B2.d.45 strengthens section 38 by imposing a necessary arithmetic
restriction inherited from B43's actual starting state. The uniform-carry
templates used by B44 need not satisfy this finer restriction. The new
construction places both outward witnesses in the correct coordinate
arithmetic class while preserving their displayed canonical pressures.
It still addresses a proposed region, not reachability of its witnesses
along the saved trajectory.

The owner
[`c6_carried_affine_mean.py`](../src/tnfr/physics/c6_carried_affine_mean.py)
implements `derive_c6_carried_affine_mean_obstruction` and
`observe_c6_carried_affine_mean_escape`. The domain is the same conditional
fixed-phase, unit-capacity C6 map, with the same represented channel weights,
`h=1/16`, declared band and B37 centered-energy bound. The reference state
`X0` is B43's actual pre-SHA encoding, not a zero-carry replacement.

### Coordinate congruences derived from the allowed nodal areas

The B37 envelope supplies necessary integer-gradient intervals. For each
node `i`, evaluate the existing rounded pressure law at every integer in
its declared interval and take the exact rational greatest common divisor
of the resulting represented nodal increments `h*p_i(m)`. Denote this
coordinate spacing by `gamma_i`. These are finite arithmetic operations
on the fixed numerical pressure law, not an additional physical parameter
or a claim of physical quantization. The row intervals can include gradients
that are not jointly realizable; including them only weakens the necessary
congruence restriction.

Every admitted increment is an integer multiple of its `gamma_i`. Thus,
while the fixed source, timestep and gradient bounds apply, every successive
reconstructed coordinate satisfies

```text
X_i in X0_i + gamma_i*Z.
```

For the declared B43 source the exact spacings are

```text
m = 2^-113,
(gamma_0,...,gamma_5) = (m,m,2*m,4*m,8*m,4*m),
q = m/6 = 1/(3*2^114).
```

All six actual `X0_i` have zero residue modulo their respective spacings.
The mean of any state in this product of affine lattices belongs to
`mu0+q*Z`. This is a necessary condition, not a converse reachability
criterion. It is also coarser than the general encoding grid `g=2^-3222`:
B44's common carry `c0` is a half-integer multiple of `m` and cannot by
itself supply the required coordinate congruences.

### A mean-preserving lift onto all six coordinate classes

Use the two displayed templates of section 38, with common displayed mean
`m_x=1/2-delta/3`. For a requested arithmetic mean `mu=mu0+k*q`, set
`c=mu-m_x`. In a general affine class, let `a_i` be the residue of
`X0_i-x_i` modulo `gamma_i`, chosen in `[0,gamma_i)` for the selected
displayed row. The present two templates have `a_i=0`, but the construction
retains the affine residues rather than assuming they vanish.

Select coordinate `0`, whose spacing is the minimum `m`, to balance the
sum. Define

```text
r_i = a_i + gamma_i*floor((c-a_i)/gamma_i),    i=1,...,5,
r_0 = 6*c - sum_{i=1}^5 r_i.
```

The last identity makes the mean exactly `mu`. Moreover,
`6*c-sum_i a_i` is an integer multiple of `m`: the requested mean differs
from `mu0` by `k*q`, and every coordinate spacing is an integer multiple
of `m`. Each `r_i-a_i` for `i>=1` is already an integer multiple of its
own spacing. Consequently `r_0-a_0` is a multiple of `m` as well. All six
reconstructed coordinates therefore belong to their initial arithmetic
classes. This is a direct exact construction, without searching periods
or treating a convex pressure combination as an executable schedule.

The auxiliary common carry `c` need not itself be dyadic: a requested
mean can contain a factor of three in its denominator. It is not asserted
to be an intermediate encoded state. Instead,
`6*c=sum_i X0_i+k*m-sum_i x_i` is dyadic on the shared encoding grid.
The affine residues, spacings and five floored carries lie on that grid,
so their balanced sixth carry does as well. The final six-coordinate
state, rather than the auxiliary uniform vector, undergoes the shared
encoding and cell checks.

Write `r=c*1+e`. The floor operations give

```text
-gamma_i < e_i <= 0,    i=1,...,5,
0 <= e_0 < G,          G=sum_{i=1}^5 gamma_i=19*m,
sum_i e_i = 0.
```

The largest nonpivot spacing is `8*m`. Therefore every constructed carry
lies in the shared legal interval `[c_min,c_max]` of section 38 whenever

```text
c_min + 8*m <= c <= c_max - 19*m,
c_min = -delta/2+g,    c_max = delta/2-g.
```

The resulting sufficient mean window is

```text
W'_lower = W_lower + 8*m,
W'_upper = W_upper - 19*m.
```

It still contains the actual `mu0` strictly. Its shrinkage comes directly
from the coordinate spacings and the balancing construction. It is a
sufficient window for this lift, not a maximal domain of possible
arithmetic-class states or an optimized physical threshold.

### Uniform energy control without enumerating lifted states

Since the lift error has zero mean, the centered state differs from the
uniform-carry template by exactly `e`. Its squared norm obeys

```text
||e||^2 <= G^2 + sum_{i=1}^5 gamma_i^2
        = (19^2+1^2+2^2+4^2+8^2+4^2)*m^2
        = 462*m^2.
```

The elementary inequality `||v+e||^2<=2*||v||^2+2*||e||^2`, applied to
the template's centered profile error `v`, yields

```text
E_lift <= 2*E_template + 924*m^2.
```

Both right-hand sides are strictly below the rederived B43 envelope by
exact rational comparison. In units of `delta^2`, they are approximately
`24.2595163638` for the negative template and `5.77441769843` for the
positive template, against `E_bar` approximately `96.00000000000064`.
Thus every lift in the sufficient mean window satisfies the centered
energy constraint. This analytic estimate covers the complete integer
family; no finite residue enumeration is needed to establish it.

### The independent mean boundary still has an outward witness

Consider a rational interval satisfying
`W'_lower<=a<=mu0<=b<=W'_upper`. Choose the requested means

```text
mu_upper = mu0 + q*floor((b-mu0)/q),
mu_lower = mu0 + q*ceil((a-mu0)/q).
```

They remain inside `[a,b]` and within less than `q` of its respective
boundaries. Apply the exact lift to the positive template at `mu_upper`
and the negative template at `mu_lower`. Each input now satisfies all four
candidate restrictions: legal encoding in the band, the energy envelope,
the mean interval and every coordinate congruence inherited from `X0`.
Displayed EPI has not changed, so the freshly recomputed mean increments
remain

```text
h*mean(p_negative) = -3*q,
h*mean(p_positive) =  9*q.
```

They strictly exceed the possible boundary gaps in their outward
directions. A valid shared-kernel endpoint therefore leaves the proposed
mean interval; a formal candidate outside the physical band already
violates admission and is recorded separately. The declared witnesses
remain in the band. Their coordinate congruences are also retained after
the step, because each canonical nodal increment is a multiple of its
derived coordinate spacing.

B45 consequently excludes the local centered-energy/independent-mean
candidate even after intersection with this necessary arithmetic class.
It strengthens B44's domain statement without turning congruence
membership into temporal reachability. Both witness states are hypothetical;
neither is claimed to lie on B43's actual future trajectory. Wider mean
windows, tighter jointly reachable subsets, and restrictions that couple
mean to shape and individual carries remain open. These results supply
neither an infinite stability theorem nor whole-band escape.

## 40. B46: opposite-node relay strips and a signed local budget

B2.d.46 addresses joint carry/shape compatibility at the actual B43
pre-SHA endpoint. The source remains the fixed phase, unit C6 support,
unit capacity, canonical pressure weights, band `[.375,.625]` and
`h=1/16`. All six incoming remainders are retained. This is a conditional
numerical branch: B43's historical terminal SHA changed capacity, and
replaying its serialized evidence does not recreate a live execution seal.

The owner is [c6_carried_relay.py](../src/tnfr/physics/c6_carried_relay.py),
with a reproducible producer in
[c6_winding_relay_budget.py](../benchmarks/c6_winding_relay_budget.py).
The producer reuses the complete B43 arithmetic replay, shared C6 pressure,
closure and carried nodal integrator. No pressure projection, extra
oscillator, coefficient fit or carry reset is introduced.

### Two coupled cell coordinates with independent switches

Write `delta=2^-54`, `m=2^-113`, `X=x+r`. From the actual initial visible
row `(.5 + delta*(-3,0,2,0,8,-5))`, select nodes 0 and 3 and their immediate
predecessor floats. This gives four displayed rows, in delta offsets:

```text
C00 = (-3,0,2, 0,8,-5),    C10 = (-4,0,2, 0,8,-5),
C01 = (-3,0,2,-1,8,-5),    C11 = (-4,0,2,-1,8,-5).
```

The two nodes are opposite on C6. Their pressure-response supports are
disjoint, but the certificate still recomputes all four canonical pressure
rows and verifies the exact binary64 increment identity

```text
A00 = h*p(C00),
d0  = h*p(C10)-A00,    d3 = h*p(C01)-A00,
h*p(C11) = A00+d0+d3.
```

No idealized `-1/2` neighbor coefficient is substituted for these represented
differences. Each selected node has strictly inward increments: `-a_s`
in its upper cell and `+b_s` in its lower cell. They are

| Node | `a_s/m` | `b_s/m` | `(a_s+b_s)/m` |
|------|---------|---------|---------------|
| 0 | 3039824576180305 | 3558973984143090 | 6598798560323395 |
| 3 | 1803052714421816 | 4795745845901584 | 6598798560323400 |

Let `t_s` be the exact midpoint of its adjacent floats, and
`u_s=X_s-t_s`. Nearest-even rounding assigns the node-0 facet to the lower
cell, giving the invariant strip `-a0<u0<=b0`. The node-3 facet belongs to
the upper cell, giving `-a3<=u3<b3`. The complete strips fit strictly
inside their two-cell unions and stay in the declared band. The actual
incoming offsets are `2638626784309416*m` and `2678732075009744*m`, so both
membership checks pass without modifying the state.

For the first orientation the exact rotation is

```text
u0(n) = b0 - ((b0-u0(0)+n*a0) mod (a0+b0)),
```

and for the second it is

```text
u3(n) = -a3 + ((u3(0)+a3-n*a3) mod (a3+b3)).
```

These identities prove inclusion for every legal point of each strip,
conditional on the other four displayed values staying fixed. They do
not establish full six-node invariance. They require no enumeration of
the very long rational rotation periods.

### Removing the oscillatory contribution exposes a signed drift

Define `k_s=d_s/(a_s+b_s)` and `v=A00+k0*a0+k3*a3`. Exact pressure
independence gives `k_s[s]=1`, zero cross-relay entries and `v0=v3=0`.
For every step before another displayed coordinate exits, including the
step that first exits, the complete vector identity is

```text
X(n) = X(0) + n*v + k0*(u0(n)-u0(0)) + k3*(u3(n)-u3(0)).
```

The correction columns are exactly

```text
k0 = (1, -3299399280161697/6598798560323395, 0, 0, 0,
         -3299399280161696/6598798560323395),
k3 = (0, 0, -412424910020212/824849820040425, 1,
         -137474970006737/274949940013475, 0).
```

Thus `X_j-k0[j]*X0-k3[j]*X3` has constant increment `v_j` within this
four-cell region. In particular,

```text
v4 = -163545780653844840044770794592 * m / 274949940013475 < 0.
```

This is a signed recurrent budget derived from the actual nodal increments.
It is valid across switches of either relay; a frozen single-cell pressure
is not assumed. It is also local: after a non-relay displayed value changes,
these four pressure rows no longer describe the next transition.

For each non-relay coordinate bound the oscillatory correction over the
two strips by exact rational endpoints `c_j^-` and `c_j^+`. If `v_j<0`,
choose the first positive integer `N` satisfying
`X_j(0)+N*v_j+c_j^+<RN_lower(x_j)`. For positive drift use
`X_j(0)+N*v_j+c_j^->RN_upper(x_j)`. Strict inequalities keep the bound
valid for either rounding-tie orientation. The four individual deadlines
for nodes `(1,2,4,5)` are `(205,299,10,2899)`. Therefore a non-relay cell
exit or earlier band failure must occur by step 10 from this input.

The observer first derives that deadline, then evaluates the exact formula
and shared nodal transitions only until the first exit. It finds node 4
exiting at step **7**, with visible endpoint offsets
`(-3,0,2,0,6,-5)`. All seven steps remain inside the physical band. Their
complete nodal balance residual is zero in all six coordinates, the local
mean area is exactly `1/2^114`, and the vector area is nonzero. Adding this
conditional area to B43's original budget gives `3296/(3*2^114)` relative
to `.5`; it is neither a complete return nor a new live graph observation.

### Result and next boundary

This block gives a constructive joint subcell mechanism and its precise
limit: the two relay coordinates are conditionally trapped, while a
different corrected coordinate forces departure from the local region.
The retained report is `artifacts/research/c6_winding_relay_budget.json`.
Its endpoint includes the exact remainder and is the next conditional
numerical input; the original B43 record remains unchanged.

Any extension must include node 4's actual new displayed value and recompute
the affected pressure rows. Local corrected-coordinate budgets can be
combined only after checking exact transition compatibility and the change
of correction across their common boundary. Merely concatenating local
deadlines, convex pressure balances or projected relay periods does not
establish a global signed budget or an invariant region. General trapping,
indefinite mean control, future operator admission and physical
correspondence remain open.

## 41. B47: a local relay budget survives the interacting boundary

B2.d.47 resumes the exact conditional B46 endpoint, including all six
remainders. Its visible offsets from `.5`, in `delta=2^-54`, are
`(-3,0,2,0,6,-5)`. The fixed phase, unit capacities, unit C6 support,
canonical pressure coefficients, `h=1/16` and `[.375,.625]` band are retained.
The complete B43 arithmetic evidence and seven B46 conditional steps are
replayed before accepting this input. This binds serialized numerical
lineage, without recreating a live graph seal or undoing historical SHA.

The new APIs extend [the existing relay owner](../src/tnfr/physics/c6_carried_relay.py):
`derive_c6_carried_local_relay_budget` and
`observe_c6_carried_local_relay_exit`. The
[handoff producer](../benchmarks/c6_winding_relay_handoff.py) reuses
`replay_c6_relay_budget_evidence`; the original B46 producer shares that
reconstruction and record logic. There is no second solver or pressure law.

### The third switch cannot be treated as another independent relay

Add node 4's displayed pair `6/8` to B46's node-0 pair `-4/-3` and node-3
pair `-1/0`. Recomputing all eight canonical pressure rows reveals the
nonzero mixed increment

```text
h*p(node3 lower, node4 lower) - h*p(node3 lower, node4 upper)
  - h*p(node3 upper, node4 lower) + h*p(node3 upper, node4 upper)
    = -8 * 2^-113 * (e3+e4).
```

Node 0 is held at its upper value in this expression; the same result
holds in its lower branch. Its two pairwise mixed differences and the
three-way mixed difference vanish. The nonzero term is an exact consequence
of the represented pressure product/assembly. Treating nodes 3 and 4 as
independent would omit it. These eight evaluations are a finite control,
not the proof of the broader locality statement below.

### Local support supplies a common corrected coordinate

On fixed C6 the two required EPI pressure gradients are exactly

```text
g0 = (x5+x1)/2-x0,
g1 = (x0+x2)/2-x1.
```

The shared indexed pressure reader applies its represented product and
fixed phase-source assembly to these gradients. Consequently, holding
displayed nodes `(1,2,5)` fixes both pressure functions of `x0`, regardless
of the admitted displayed values of nodes `(3,4)`. The source/target support
union derives this held set; it is not a selected cancellation assumption.
The proof allows arbitrary node-3/node-4 values in the declared band, not
only the eight boundary cells. Their own pressure rows continue to refresh.

Node 0 therefore keeps its B46 invariant strip `-a<u<=b`, where
`u=X0-facet`, `a=3039824576180305*2^-113` and
`b=3558973984143090*2^-113`. The incoming B46 remainder belongs to it.
Recompute the two local pressure rows and set

```text
k = h*(p1_lower-p1_upper)/(a+b)
  = -3299399280161697/6598798560323395,
v = h*p1_upper+k*a
  = -3360475576564941293533924113653 * 2^-113 / 1319759712064679 < 0.
```

For every transition whose source retains the held neighborhood,

```text
W = X1-k*X0,          W(n)-W(0) = n*v,
X1(n) = X1(0)+n*v+k*(u(n)-u(0)).
```

The node-0 modular formula remains the same exact rotation as in B46.
Only these two coordinates have a closed formula; no independent formula
for the other four coordinates is asserted. This common budget crosses
the node-4 boundary without an omitted change of correction potential:
its coefficient and drift are identical on both sides. It remains valid
through the first outgoing step whose endpoint changes a held value.

### Analytic deadline and first held-neighborhood exit

Let `c_plus=max(k*(-a-u(0)), k*(b-u(0)))`. Since `v<0`, the least integer

```text
N = floor((X1(0)+c_plus-RN_lower(x1))/(-v)) + 1
```

puts even the largest possible corrected coordinate strictly below its
original rounding cell. For the actual incoming state this gives **198**.
Until that deadline, either a held displayed value changes, or numerical
band admission fails. The existing independent band-horizon certificate
covers **196713720348826205** steps from this input, which exceeds 198.
That sufficient bound is not a simulated duration or a claim of indefinite
stability. Here it eliminates band failure as the earlier alternative.

The observer derives the deadline before taking any new step, refreshes
all six canonical pressure rows at every step and stops at the first held
change. This occurs at **197**, node **1**, with endpoint offsets
`(-3,-1,2,0,8,-5)`. All 197 shared-kernel steps and all 198 formula points
have zero nodal and corrected-budget residuals. The complete vector area
is retained and nonzero. The local mean area is `349/(3*2^114)`; together
with B43 and B46 it is `3645/(3*2^114)=1215/2^114` relative to the original
uniform `.5` preparation. There are 204 conditional steps after B43, and
no additional live graph-owned word.

This observed branch stays within the eight boundary cells until the same
step 197. The broader theorem's benefit is its independence from any
restriction on the free coordinates, not a claim that this particular
trajectory visited additional cells. Membership in the local theorem
alone does not prove that an arbitrary supplied state descends from B46;
the report establishes that numerical lineage separately by replay.

The retained result is `artifacts/research/c6_winding_relay_handoff.json`.
Its `B47_conditional_first_exit.endpoint` is the next conditional input,
with the full remainder. Node 1's changed value now modifies node 0's
pressure as well as the corrected-coordinate budget. The next gate is
therefore a new dependency boundary: rederive the affected relay and
account for any change in correction potential before composing another
budget. Repeating B47's formula beyond this endpoint is invalid. The result
does not establish a global signed budget, invariant trapping, future
operator admission or empirical correspondence.

## 42. B48: exact correlated-set viability and the remaining global gap

The target is indefinite boundedness of the **current fixed C6 numerical
branch**, with the unchanged B47 endpoint and complete carry. No theorem
about arbitrary TNFR networks or future operator words is being inferred.
The canonical pressure, phase tuple, unit capacity, `h=1/16` and positive
band remain fixed. No post-B47 trajectory steps are taken in this audit.

### Universal preimages, rather than an observed return

The shared carried nodal equation is a translation in each displayed cell:

```text
X = x+r,   F(X) = X+h*p(x),   x_next = RN(F(X)).
```

For a finite declared visible family, derive each coordinate spacing as
`gamma_i=gcd({h*p_i(x)})`. These are arithmetic consequences of the source
table, not fitted physical parameters. If a coordinate has identically zero
increment, the implementation uses the shared `2^-3222` encoding grid.
The affine cosets through the unchanged origin are preserved while pressure
comes from that family. On them, a subcell is a closed integer box inside
the exact band-clipped nearest-even cell. Different pieces must be disjoint
on these cosets; their geometric hulls may overlap only off that grid.

Starting from the candidate `K_0`, compute

```text
K_(n+1) = K_n intersect F^-1(K_n).
```

Canonical pressure is freshly read for every source cell. Exact translation
and intersection with target boxes retain the incoming-carry relation.
Adjacent boxes are merged only when their other coordinate bounds agree.
Because the sets are finite and nested, equality of **exact integer point
counts** proves equality of sets. It is not a floating volume comparison.

The reusable owner
[`derive_c6_carried_viability`](../src/tnfr/physics/c6_carried_viability.py)
has three distinct outcomes:

- `fixed_point`: a complete nonempty fixed set contains the supplied origin.
  Induction proves arbitrary repetition remains in the retained boxes and
  band. This is conditional numerical boundedness, without convergence or
  live-graph provenance.
- `origin_excluded`: the origin leaves the initial candidate or its band
  within the recorded descent depth. This does not prove whole-band escape.
- `resource_limit`: only the last fully completed descent is returned.
  No infinite-time or origin-exit conclusion follows from unfinished work.

The positive proof contract is tested on a stationary C6 case and a
nonstationary alternating C6 two-cycle, including retained carry. Neither
control replaces the pressure source of the research branch.

### The actual profile candidate

Use the existing exact forced profile `z` and the actual B47 mean `mu`.
The adjacent represented values bracketing `mu+z_i` give the lower and upper
offsets, in units `delta=2^-54` from `.5`:

```text
lower = (-4,-1,2,-1,6,-6)
upper = (-3, 0,4, 0,8,-5).
```

Their 64 combinations include the unchanged B47 endpoint at mask 57, where
bit `i` selects the upper value of coordinate `i`. The candidate-pressure
gcds give `gamma=2^-113*(1,1,4,8,8,8)`, which are specific to this family.
Three complete descents retain 64, 416, and 4,285 boxes respectively, with strictly
decreasing exact point counts and the origin still present. The fourth
descent exhausts the 500,000-work-item guard. The result is therefore
**undecided**. The guard controls computational work, not physical time.
No new trajectory, invariant set or escape is asserted by these counts.

### Exact scope of the quadratic controls

Write the cube's nodal increments in units `m=2^-113` as
`a_sigma=b-M*sigma+e_sigma`, using the six one-bit differences to define
`M`. With `K=3299399280161697`, the positive diagonal symmetrizer is

```text
D = diag(1,1,2K/(K+3),K/(K+3),2K/(K-1),K/(K-1)).
v = (1,1,1/2,1,1/2,1).
M*v = (1,0,-8,-8,0,0).
v^T*D*M*v = (3-15K)/(K+3) < 0.
```

Thus the proposed inverse-affine positive quadratic cannot supply the
desired energy. This is a failure of that surrogate, not a dynamical
instability proof. The exact mixed remainders range over `(0..1,0,-8..0,
-8..0,-8..0,0)` in these units; they cannot be dropped in a mean argument.

A second control uses the **actual** nonlinear table. Let `Q_i=e_i-e_5`.
Every symmetric matrix can be written
`H=Q*S*Q^T+Q*a*1^T+1*a^T*Q^T+K*beta*1*1^T`. A retained 21-term positive
rational dual, recomputed and checked against all its exact coefficients,
gives

```text
sum_j y_j * sign(sigma_j[i_j]-1/2) * (H*a_sigma_j)[i_j]/K
    = lambda * trace(S),
y_j > 0, sum_j y_j=1, lambda > 0.
```

Positive definite `H` implies positive definite `S` by congruence, so at
least one selected oriented component must be outward. This excludes the
all-orthant inward common-quadratic **sufficient criterion**. It does not
exclude every quadratic invariant of a finite region, shifted or piecewise
potentials, or a reachability-trimmed subset. No numerical LP tolerance is
used to admit either exact obstruction.

### Reproduction and unresolved target

Run `benchmarks/c6_winding_invariant_region.py`. It reuses the centralized
B47 evidence replay, verifies the complete B43/B46/B47 numerical lineage,
and retains the source digest, input hashes, canonical pressure table,
exact quadratic witnesses and all last-completed subcell bounds in
`artifacts/research/c6_winding_invariant_region.json`. The original B47
endpoint remains the next dynamical input; the proof search does not create
a later state or reverse historical SHA.

Global boundedness of this actual C6 branch remains **open**. The required
next mathematical object is a closed carry-correlated region containing that
state, or an exact temporally compatible signed budget proving escape.
Independent pressure balances, pairwise existential edges and longer finite
prefixes cannot substitute for this universal inclusion or budget argument.

## 43. B49: protected pair contrasts and relational past envelopes

This block keeps the B47 input, canonical pressure source, unit capacities,
`h=1/16`, positive band and complete carry. It changes the representation of
a proof candidate, not the nodal dynamics. Both search directions now share
source, state, pressure and exact RN-cell preparation in
[`c6_carried_viability.py`](../src/tnfr/physics/c6_carried_viability.py).

### Conditional contrast strips from the nodal map

Let `g` be the rational gcd of all declared nodal increments, with the shared
encoding grid used if all increments vanish. Write
`X_i=X_initial_i+g*k_i` on the initial state's preserved arithmetic class.
All `k_i` are integers; this common grid may contain more states than the
separate coordinate gcds. It is a sound relaxation,
not a change of carry. Each exact RN cell gives integer coordinate bounds,
including nearest-even endpoint ownership, and one fixed translation `a_c`.

For a pair `(i,j)`, project that cell onto `d=k_i-k_j`, obtaining `[L_c,U_c]`
and signed increment `b_c=a_c[i]-a_c[j]`. Start a candidate strip `[l,u]`
at the actual value zero. Whenever it intersects a projected cell with
`b_c<0`, extend `l` to include `L_c+b_c`; for `b_c>0`, extend `u` to include
`U_c+b_c`. Only the finite set of cell endpoints can change either bound.
At termination, verify directly, for every nonempty intersection,

```text
max(l,L_c)+b_c >= l,   min(u,U_c)+b_c <= u.
```

Thus every pair strip is preserved by a step whose source stays in the
declared cube. Their simultaneous intersection contains the unchanged
initial state. This is a **conditional** result: it does not itself keep
the next visible state in the cube. All 15 pair contrasts tighten the
current candidate. For example, the contrast between coordinates 4 and 2,
after subtracting their central-facet difference, is confined to roughly
`[-0.0125*delta, 2.022*delta]`, compared with the original `[-4*delta,4*delta]`,
where `delta=2^-54`. Exact bounds and their source premises are retained.

### A compact envelope of states with a compatible past

One relational zone per visible cell stores the integer inequalities
`k_i-k_j<=B_ij`, including a seventh fixed coordinate `k_6=0` for absolute
bounds. Shortest-path closure gives exact implied bounds. Translating a
zone adds `a_i-a_j` to each entry. Intersections are closed exactly; the
entrywise maximum of closed nonempty matrices is their least zone hull.
The hull can include extra states, so membership alone never proves a
compatible past.

Let `D` be the RN cube intersected with the independently protected pair
strips, and let `alpha_D` form this hull separately inside each cell of `D`.
The successive outer envelopes are

```text
R_0=D,   R_(n+1)=alpha_D(F(R_n) intersect D).
```

Monotonicity implies `R_(n+1) subset R_n`. If a complete image `F(R_n)`
stays inside the original cube, the conditional pair-strip theorem also
keeps it in `D`. Consequently

```text
F(R_n) subset alpha_D(F(R_n)) = R_(n+1) subset R_n.
```

This proves an invariant core without requiring equality between successive
envelopes. The complete Cartesian RN cover is checked before an outer-box
test is used; holes in an arbitrary family cannot be silently filled.
An empty core, a stationary abstraction with outgoing states and an
unfinished layer do not establish boundedness of the supplied origin.

Actual initial membership is a separate gate. If a certified core contains
the original carried state, induction applies immediately. Otherwise only
a replay of the proof-derived finite entry prefix can bind that state to
the core. Its entire prefix must stay admitted and end inside the retained
core. Union with those prefix points then gives an invariant set containing
the original state. No trajectory is advanced merely to look for recurrence.

### Current result and reproduction

Run `benchmarks/c6_winding_invariant_region.py --method forward`. The same
campaign and B47 lineage replay serve both B48 and B49, with distinct output
files. The current result is retained in
`artifacts/research/c6_winding_forward_envelope.json`.

The relational representation keeps only 64 zones. With the pair strips,
the retained bounded search completes 259 image layers and reduces possible
outgoing coordinate facets from 72 to 56, while retaining the original B47
point. Its 250000-intersection guard is computational. Since outgoing
facets remain, the result is **inconclusive**, not a trapping theorem or a
trajectory escape. The report makes no new live-graph or asymptotic claim.

An independent exact replay recomputes all 259 layers and verifies 15 pair
barriers, 128 initial/final closed matrices and all 56 outward facets. Ten
of those outward slabs contain explicit hypothetical states with **exactly
the original B47 mean** and its separate coordinate cosets
`g*(1,1,4,8,8,8)`. Six leave through node 0's lower boundary (masks
`8,12,16,20,24,28`); four through node 5's upper boundary (masks
`49,51,57,59`). Every witness satisfies all retained zone inequalities and
the exact RN cell. Consequently, intersecting this particular retained
envelope with any independent mean interval containing the original mean
cannot make it invariant. This does not exclude a mean/shape/carry-correlated
subset, and none of these hypothetical points is claimed reachable. The
independent replay and witnesses are retained in
`artifacts/research/c6_winding_forward_envelope.validation.json` and
`artifacts/research/validate_b49_envelope.py`.

A separate static refinement restores those coordinate cosets by snapping
difference bounds to their exact gcds and reclosing the matrices. It tightens
1,588 entries, but all 56 outward facets still have exact coset-admissible
witnesses. One complete refined image also retains 56 such facets. These
witnesses are checked with the shared nodal kernel; they are hypothetical
one-step states, not a continuation of B47. Thus restoring coordinate
arithmetic alone does not close this retained envelope. Evidence:
`artifacts/research/c6_b49_coordinate_coset_probe.json`.

Validation includes 636 targeted passing tests. A separate canonical
zero-phase control has 729 exactly enumerated states and forward layers
`729 -> 45 -> 9 -> 3`; the implementation verifies entry into that invariant
core from a nonstationary initial state. Other controls prevent an escaping
origin, a partial image or a stationary outer envelope with outgoing states
from being certified. These are verifier checks, not parameter changes to
the current fixed-phase `h=1/16` research branch. The retained B49 source
delta and validation record bind the report to its scientific source digest.

A separate bounded falsification probe took 10000 shared-kernel steps from
the same B47 state and found no exit from the 64-cell cube. Its explicit
computational budget and full finite trace are retained in
`artifacts/research/c6_cube_counterexample_probe.json`. This does not prove
stability and does not replace B47 as the primary proof input. Global
boundedness remains open: the surviving outward states may require stronger
correlations than pair differences, a smaller reachable region, or a
different candidate. Their presence in an outer envelope does not establish
their reachability from B47.

## 44. B50: exact point predecessors and the temporal-correlation gap

B49's outward states are hypothetical members of an outer envelope. B50 asks
whether selected points can have an exact carried past, using the same C6
pressure, phase, unit capacity, timestep, RN rule and full remainder. It adds
no physical law or parameter. The public predecessor observer shares source
and RN-cell preparation with both existing viability methods in
[`c6_carried_viability.py`](../src/tnfr/physics/c6_carried_viability.py).

### Exact predecessor sets, not a graph of visible labels

Fix a complete target state `X`, a declared source domain `D` and the original
affine coordinate cosets. In visible cell `c`, the only possible predecessor is

```text
Y_c = X - h*p(c).
```

It is admitted only if it has that exact nearest-even visible row, stays in
the band and domain, and belongs to the preserved coordinate cosets. Every
accepted edge is checked with the shared nodal integrator, including its full
carry. Each complete predecessor layer retains exact states and successor
indices, so adjacent edges cannot silently change their intermediate state.
Pressure is reconstructed from the canonical source rather than trusted from
a cached or serialized table.

Starting from `P_0={X}`, compute `P_(n+1)=D intersect F^-1(P_n)`. These sets
are not generally nested, and repeated cardinalities do not imply equality.
If a complete `P_n` is empty, no admitted depth-n history ends at `X`.
This excludes occurrence after n transitions during a path wholly inside D;
it does **not** exclude earlier transient visits. A nonempty layer certifies
finite compatible pasts, without proving connection to the actual B47 origin.
Resource interruption discards the partial next layer.

Origin matching is a separate exact test across all retained layers. A match
plus its linked steps proves a finite origin-to-target path under the declared
map. If the complete predecessor tree exhausts without any origin match,
every origin-to-target path wholly inside D is excluded, including transient
ones. Neither result applies to paths that leave D and later return. A point
certificate says nothing universal about its containing facet or other points
with the same displayed EPI.

### Current branch and pointwise first-exit exclusions

The campaign `benchmarks/c6_winding_invariant_region.py --method predecessors`
first reconstructs all B49 payload fields from B47's unchanged lineage. It
checks the selected witness bytes, exact original mean, RN cell and actual
outgoing nodal step. Each target is then audited in two distinct domains:
the full 64-cell RN cube and the retained relational envelope `R259`.

The campaign also verifies the exact clipped identity
`F(R259) intersect D subset R259`, where D is the cube intersected with B49's
protected pair strips. Since the original state belongs to R259, every
prefix before its first cube exit remains in R259. Complete point-predecessor
exclusion from that origin therefore rules out that point as the cause of the
first cube exit. It does not exclude the entire facet, prove confinement or
control a later trajectory that has already left the cube.

The retained point audit is
`artifacts/research/c6_winding_temporal_predecessors.json`. It introduces no
new actual trajectory steps or live execution; shared-kernel checks on
hypothetical edges are explicitly distinct from the B47 research trajectory.

Of the ten selected original-mean outward points, nine have exhausted exact
predecessor trees in R259, with no B47 origin in any complete layer. Their
first empty depths, indexed by visible mask, are
`8:12, 12:9, 16:3, 20:1, 24:3, 49:1, 51:1, 57:1, 59:4`.
Those nine particular points cannot cause the first cube exit of the retained
branch under the fixed map. Mask 28 remains undecided after 21 complete
predecessor layers and the 32,768-row-check guard; a partial deeper tree is
not promoted. The broader complete RN cube gives five exhausted trees:
`20:42, 49:91, 51:120, 57:91, 59:40`; its other five searches are resource
limited. These two domains and depth bounds must not be conflated.

The implementation passes 690 targeted tests, including 54 new point/past
and report cases. A separate validator independently reconstructs all 20
serialized certificates and replays 4,014 exact predecessor edges plus ten
hypothetical outgoing steps. It verifies 893 nonempty clipped image pieces
for the R259 inclusion. Artifacts:
`artifacts/research/c6_winding_temporal_predecessors.validation.json`,
`artifacts/research/validate_b50_predecessors.py` and
`artifacts/research/b50_final_validation.json`. These tests and mathematical
checks do not supply empirical physical evidence or a whole-region theorem.

### Why retaining temporal labels helps, and what remains open

A separate bounded prototype keeps the preceding visible cell as part of
each abstract mode before taking any matrix hull. One preceding cell gives
870 modes after seven complete refinements, reducing distinct outward facets
from 56 to 48 and retaining six of the ten selected point witnesses. Two
preceding cells give 7,387 modes after one complete refinement, 46 outward
facets and four surviving selected witnesses. Each lifted clipped-image
inclusion is verified; no actual entry or invariant core is claimed.
The mode counts are computational state representations, not extra TNFR
variables. Evidence: `artifacts/research/c6_temporal_mode_envelope_probe.json`.

For comparison, exact full unsafe-set preimages without any hull retain
56, 280, 861, 2,553 and 7,963 zones through depths zero to four before a
20,000-zone guard interrupts the next complete layer. This is again a
representation limit, not an escape or stability theorem; see
`artifacts/research/c6_b50_unsafe_past_probe.json`. These probes show precisely
where temporal information is lost and where retaining every disjunction
becomes expensive. The remaining task is a compact, universally checked
temporal refinement or a source-derived invariant region. More finite
trajectory steps cannot replace that inclusion proof.

## 45. B51: whole outgoing regions excluded by their complete pasts

B51 replaces selected point targets with the entire outward slabs of the
retained R259 domain. The same nodal source, unit capacity, `h=1/16`, band
and complete carried B47 state are unchanged. Both point and region campaigns
now use one B49 reconstruction and clipped-inclusion helper, and the physics
owner reuses the existing source, RN-cell and relational-domain validation.

### Region targets and the complete-past argument

Write `X_i=X_initial_i+g*k_i`, with B49's common increment grid `g=2^-113`.
The full 64-cell RN cube has exact integer endpoints `l_i,u_i`, including
nearest-even ownership. These endpoints are reconstructed from complete RN
cells: the protected pair strips can tighten coordinate projections and
must not redefine the original cube's exit boundary. In source cell c with
integer increment `a_c`, the lower and upper outward slabs are respectively

```text
T_(c,i,lower) = R259_c intersect {k_i <= l_i-a_c[i]-1},
T_(c,i,upper) = R259_c intersect {k_i >= u_i-a_c[i]+1}.
```

The nonempty slabs cover every possible first-cube-exit source in R259.
They can overlap; their number is not a probability or a progress percentage.
The common grid relaxes the finer coordinate cosets, so exclusion is sound
for the actual state even though an admitted hypothetical point need not
have its full arithmetic provenance.

For each complete slab T separately, begin with `P_0=T`. In each source cell
i, translate every target zone backward by the actual increment `a_i`,
intersect with R259_i and take the least closed difference-bound hull. Thus

```text
P_(n+1) = alpha_R259(R259 intersect F^-1(P_n)).
```

Every exact n-step predecessor is included. The layers need not be nested;
cardinality equality is insufficient. If a complete layer is empty, there
are no predecessors at that or any later depth. If consecutive complete
abstract layers are exactly equal, the deterministic abstract recurrence
stays equal thereafter. Either case proves no origin-to-T path within R259
when the origin is absent from every earlier complete layer as well.
An origin admitted by a hull is only inconclusive, not an actual path.

The established identity `F(R259) intersect D subset R259`, with D the cube
and its protected pair strips, combines with the conditional pair-strip
preservation proved in section 43 and initial membership to keep every
actual pre-cube-exit prefix in R259.
Therefore an entire excluded target slab cannot cause the first cube exit.
This includes early transients because every complete preceding layer is
checked for the origin. It does not assert that the slab is empty, that
arbitrary hypothetical starts are safe, or that later exit-and-return paths
are excluded.

### Current C6 result

The production batch owner `derive_c6_carried_region_exclusions` advances
all 56 labeled slab queries in round-robin order under one computational
intersection guard. It records each completed layer and discards an
interrupted partial layer. The campaign is
`benchmarks/c6_winding_invariant_region.py --method regions`; its report is
`artifacts/research/c6_winding_region_exclusions.json`.

The retained 250,000-intersection search excludes 24 **whole slabs**:
all 16 node-1 upper slabs and all eight node-5 upper slabs. Their complete
predecessor layers become empty within depths two through eight for node 1,
and at depth six for node 5, with the actual origin absent throughout.
These two node/direction classes cannot produce the first cube exit under
the fixed map. The remaining 32 slabs are eight each for node-0 lower,
node-2 lower, node-3 upper and node-4 upper. Their bounded searches remain
inconclusive. The engine does not certify indefinite trapping or actual
escape, and no new actual trajectory or live graph execution is run.

### Complementary probes and the next boundary

An independent increasing backward-reachability hull gives nine complete
closed-set exclusions, all within the node-1 upper class. The hull of the
union of all unsafe targets instead admits the origin immediately; that is
loss of information, not an observed exit. This supports keeping target
labels separate. Evidence: `artifacts/research/c6_unsafe_backward_hull_probe.json`.

A separate point-tree lifting calculation translates the same perturbation
through every ancestor of an exhausted B50 tree. Preserving one violated
domain inequality for each rejected row prevents new branches; preserving
one signed coordinate of every admitted ancestor excludes the origin.
Intersecting these exact difference constraints gives nontrivial excluded
regions around the nine former point witnesses. Five lie in node-0 lower
slabs (masks 8, 12, 16, 20 and 24), outside the two fully excluded exit
classes. These partial-region exclusions remain complementary evidence;
they do not exclude their whole slabs. See
`artifacts/research/c6_b51_point_region_probe.json`.

Coarsening temporal labels to only the prior bits of node 1 and its two
neighbors reduces memory cost, but leaves 51 outward facets, versus 48/46
with one/two complete preceding cells. The local pressure dependency alone
does not preserve all useful cross-node temporal constraints.

The regional exclusions also support a sound domain refinement: remove each
proven unreachable slab by its exact complementary bound in that source
cell. Every actual prefix before the first cube exit remains in the refined
domain, although universal forward invariance of that domain is not claimed.
The retained probe rechecks all 24 exclusions, tightens 104 matrix entries
across 22 cells and preserves the actual origin. With the same 250,000
intersection guard, all 32 remaining queries are still inconclusive after
33 or 34 complete layers; no further slab is excluded. Evidence:
`artifacts/research/c6_b51_iterated_certified_cuts.json`.

The next task is a compact proof over the four remaining exit classes,
preserving more of their coupled pressure, carry and temporal relations.
Merely repeating the certified cuts, increasing a resource guard or testing
additional isolated points cannot supply the missing universal proof.

### Independent validation and retained source

`artifacts/research/validate_b51_regions.py` independently reconstructs the
64 canonical pressure rows, complete RN cells, all 56 unsafe slabs, 893
clipped forward-image pieces and all 1,176 completed backward layers. Its
exact 250,000-intersection replay confirms the 24 whole-region exclusions
and 32 resource-limited queries. The validation is bound to the four input
hashes, report bytes and implementation source in
`artifacts/research/c6_winding_region_exclusions.validation.json`.
The source archive chain ends at `b51_scientific_source_delta.json`; the
combined software and documentation checks are retained in
`artifacts/research/b51_final_validation.json`.
The targeted suite passes 729 tests in 163.37 seconds, including 39 new
regional-owner and report cases. Tests cover independently enumerated
finite models, exact stationary frontiers, nonnested layers, false origin
membership caused by hulls, interrupted budgets and evidence integrity.

## 46. B52: a nodal excursion budget excludes the node-2 lower boundary

The remaining regional queries need information that survives their hulls.
A useful exact relation is now available for the half of the original
64-cell cube in which node 2 has its lower represented value. Keep B47 as
the primary origin and use the same integer carried coordinates
`X=X_B47+g*k`, `g=2^-113`. Define the proof coordinate

```text
W(k) = -2*k_0 + 2*k_2 + k_3 - k_5.
```

This is a linear read-out of the existing nodal state. It adds no dynamics,
pressure coefficient, capacity, timestep or state reset.

Its coefficients are derived from the unit-C6 nodal diffusion geometry:
they are the primitive integer zero-mean Poisson contrast satisfying
`L_rw*w=(3/2)*(e_2-e_0)`. The campaign solves that rational system rather
than selecting new physical weights. This geometric identity motivates
the read-out; the complete binary64 pressure table separately proves its
drift, including all nonlinear rounding terms:

```text
W(k_next)-W(k) >= d = 1418638309884336 > 0.
```

The complete node-2 lower unsafe slabs inside R259 have
`W <= M = 237475340900157211`. Every admitted transition from the other half
of R259 into this half has
`W >= L = 449419196811482809 > M`. Each extremum is certified by an integer
transport dual and a matching admissible primal point; every crossing
intersection is enumerated exactly. These bounds already hold in R259,
without the optional B51 cuts.

### The infinite claim reduces to one derived finite gate

After any reentry, W starts above M and increases throughout the active
visit, so that visit cannot reach a node-2 lower unsafe slab. The original
B47 point starts in the active half at W=0. During its initial visit, a
target could therefore be reached only at source ordinals
`n <= floor(M/d) = 167`. This is a mathematical bound, not a selected
simulation horizon.

The shared carried integrator checks the initial visit with freshly rebuilt
canonical pressure and complete incoming remainders. It can stop as soon as
the visit ends or W exceeds M. On the actual B47 input this occurs at step
56, with `W = 247713108641769776 > M`. Every checked source avoids the
targets, all 56 steps remain in R259 and the full RN cube, and every nodal
balance residual is exactly zero. The earlier scratch check of all 168
steps is redundant for this certificate; its endpoint never replaces B47.

Consequently, no path from the unchanged B47 origin that remains in R259
can reach any of the eight node-2 lower unsafe slabs. Section 43's
conditional pair-strip preservation and the rebuilt clipped forward
inclusion bind this to every actual prefix before a first cube exit. Thus
the eight complete slabs cannot cause that first exit at any later time.
This is neither invariance for arbitrary starts nor a claim about paths
that have already left the proof domain.

### Shared implementation and present coverage

`src/tnfr/physics/c6_carried_excursion.py` owns the excursion theorem,
integer extremum certificates, complete ingress checks and derived-prefix
observer. It shares canonical source, RN-cell and relational-domain
preparation with `c6_carried_viability.py`, and uses the shared nodal
integrator for the finite gate. A nonpositive drift, nonstrict ingress gap
or unfinished prefix remains inconclusive. Domain departure only settles
domain-confined path exclusion; the owner never certifies global trapping.

The existing campaign's `--method excursion` reconstructs B49, rechecks all
24 positive B51 queries, and applies the new certificate to the full
node-2 lower target family. Full RN geometry and unsafe-slab construction
are now shared by the regional and excursion campaigns. The combined
result excludes **32 of 56** complete first-exit slabs. The remaining 24
are eight each for node-0 lower, node-3 upper and node-4 upper. Indefinite
boundedness of this C6 case remains open; no live graph invocation occurs.
Evidence: `artifacts/research/c6_winding_excursion_exclusion.json` and its
independent validation; source/check retention uses the B52 archive and
`artifacts/research/b52_final_validation.json`.
The combined targeted suite passes 772 tests in 215.06 seconds, including
43 new owner/report cases. Independent validation checks all 199 ingress
intersections, 271 attaining primal/dual extrema, the 56 shared prefix
steps, and the 24 prior exclusions through 112 complete predecessor layers.

### Complementary controls and next proof target

A second exact coordinate, with coefficients `(1,-1,-3,-5,5,3)`, has positive
drift in the node-3 upper half, but its ingress lower bound is below the
unsafe ceiling. That coefficient vector supplies no corresponding reentry
exclusion. A bounded separation search further produces an exact obstruction
to this entire single-linear-coordinate template on the full common-grid
R259 domain: seven strictly positive rational coefficients balance five
active pressure vectors and two admissible ingress-minus-target point
differences to zero. No linear functional can be strictly positive on all
those vectors. Thus changing the weights alone cannot simultaneously give
positive drift on every node-3 upper row and strict separation of every
ingress from every unsafe target in this domain. The obstruction does not
cover smaller reachable domains, mode-dependent coordinates or nonlinear
barriers. Evidence: `artifacts/research/c6_b52_node3_excursion_separator_probe.json`.
The next attack is a coupled ingress/exit budget or finer certified domain
for the three remaining classes, retaining the distinctions that made node
2 decidable.

One concrete next candidate is a cell-dependent proof coordinate
`V_c(k)=w*k+b_c`. Every admitted active transition `c -> d` would need a
strictly positive verified increment `w*a_c+b_d-b_c`, with target and
ingress extrema using their own cell offsets. The initial budget must
start at `V_initial`, not silently at zero. The current seven-vector
obstruction does not include such cell offsets. Their feasibility and
finite-prefix gate remain unproved; the offsets would belong only to the
proof, never to the nodal update.

Three bounded refinements explain why more geometric detail alone has not
closed the argument. Successor-cell modes preserve immediate temporal
correlation but leave all former 32 queries undecided at depths 13/14 under
250,000 intersections. Removing the five lifted node-0 excluded regions
creates 102 disjoint domain pieces and reaches depths 23/24 without another
exclusion; joining those pieces per original cell fills all five holes.
Two affine read-outs in a lifted DBM reach depths 34/35 but likewise yield
no additional exclusion at the same guard. These are conditional outer
approximations, not actual exits. Retained probes are
`c6_b52_successor_modes_probe.json`, `c6_b52_lifted_domain_probe.json` and
`c6_b52_affine_lift_probe.json` under `artifacts/research`.

## 47. B53: separate cell-offset budgets exclude four further regions

The original B47 state and canonical carried map remain unchanged. B53
first removes the 32 whole first-exit slabs already excluded by B51/B52,
using their exact complementary inequalities in the original RN cube.
Call this restricted domain D32. Every actual prefix before the first cube
exit remains in D32; this is not a forward-invariance assertion for every
point of D32.

### A shared origin-containing forward envelope

The new `derive_c6_carried_reachable_envelope` in
`src/tnfr/physics/c6_carried_viability.py` computes

```text
R_0 = D32
R_(n+1) = hull_per_RN_cell({B47 origin} union (F(R_n) intersect D32)).
```

The complete layers descend and contain the unchanged origin. Induction
therefore puts every domain-confined origin path inside every retained
layer, including a complete layer retained after a resource limit. Each
layer satisfies `F(R_n) intersect D32 subset R_n`. In this instance, 264
strict descents are followed by exact matrix equality, after 232,802
intersection checks. The resulting R* still has 24 outward labels. Its
clipped fixed point is not a trapping certificate.

### Preserve each target's own proof coordinate

For a visit to the 32 cells with node 3 at its upper displayed value, use
`V_c(k)=w dot k+b_c`. Its exact change on an admitted active transition
`c -> d` is `w dot a_c+b_d-b_c`. The added constants are proof read-outs,
not new nodal parameters, pressure terms or state writes.

Four separate candidates succeed for the whole node-3 upper unsafe slabs
with masks **29, 31, 61 and 63**. Each has strict positive change on every
one of the 242 admitted active transitions. Each of the 189 ingress
pieces has potential strictly above that candidate's complete target
ceiling. The original B47 point is already above the ceiling as well, so
the derived initial gate is zero: these four results require no additional
trajectory steps. The independently checked 760 attaining primal/dual
extrema cover the four target maxima and all their ingress minima.

The candidates are stored as exact rational proof proposals in
`benchmarks/c6_winding_mode_excursion_candidates.json`. A shared positive
normalization converts each complete weight/offset family to integers.
`derive_c6_carried_mode_excursion_exclusion` shares source, RN, grid,
extremum and prefix machinery with the B52 owner, and checks every active
edge explicitly. The initial value and all ingress/target bounds include
their own cell offsets. A family with no internal edge has visits of at
most one point; it receives no fictitious positive drift certificate.

The existing campaign's `--method modes` reconstructs the prior evidence
and validates these proposals against one common pre-candidate domain.
No candidate assumes its own exclusion. It then removes the four newly
proved slabs and closes the remaining domain, D36. This second closure is
already fixed at its first complete image, with 830 intersection checks.
The report binds the B47, B46, B43, B49, B51 and B52 input bytes and the
candidate bytes. Replaying B52's 56-step gate is prior-proof verification,
separate from the zero new steps required by B53.

### Remaining gap and useful negative controls

Coverage is now **36 of 56** complete first-exit slabs:

| Pending class | Masks | Count |
|---------------|-------|-------|
| Node 0 lower | 0, 4, 8, 12, 16, 20, 24, 28 | 8 |
| Node 3 upper | 28, 30, 60, 62 | 4 |
| Node 4 upper | 56, 57, 58, 59, 60, 61, 62, 63 | 8 |

Twenty slabs remain unresolved. Neither these counts nor a successful
restricted proof establish global C6 stability, convergence or a live
post-SHA runtime theorem. All statements still concern the fixed-phase,
unit-capacity carried continuation from B47, with `h=1/16`.

The successful single-target functions do not combine into a proof for
all eight original node-3 targets. Bounded common-gradient/offset searches
retain exact template obstructions or explicit inconclusive outcomes for
the residual targets. An apparently nonincreasing mean proposal also
fails exactly: an admissible coordinate-coset transition `50 -> 0`
increases `sum(k)/2^59` by `13/2^59`. This hypothetical edge is checked
with the shared nodal kernel, but is not claimed reachable from B47.
Ignoring the small residual would create a false theorem. Reparameterized
mean-corrector searches keep this integer defect visible and have not
produced a validated barrier. These failures restrict those proof
templates; they do not prove instability of the actual path.

The exact next structural reduction is to preserve complete short return
relations, including every intermediate exit gate, instead of joining
away the coupled ingress and pressure information. Retained evidence and
validation for the integrated four-region result are in
`artifacts/research/c6_winding_mode_excursions.json` and its independent
validation. The B53 source archive records the exact scientific source.

### Validated return reduction and stopped refinements

Inside D36, no admitted transition connects two cells whose node-2 bit is
high. The original B47 point has that bit low. Retain each direct low-to-low
transition and each low-to-high-to-low path as a separate translated guard;
there are 1,265 such pieces. Origin-containing closure on the low modes
reaches equality after three iterations and 3,795 intersections. Rebuilding
every high intermediate from every retained low source tightens 26 of the
64 cells. An intermediate is retained even if it has no subsequent low
return, so an intermediate outward transition cannot disappear from the
audit. All 20 remaining labels still occur.

The coverage argument is induction over successive low visits, followed by
one-step reconstruction of high visits. It is not a claim of one-step
invariance for the projected per-cell hulls. Independent validation checks
all 830 ordinary edges, every compressed guard and all three complete
layers. The retained prototype is
`artifacts/research/c6_b53_compressed_return_probe.json`, with its domain
loader and validation. Sixty strict and sixty nondecreasing coupled-return
separator queries produce no positive certificate; 728 independently
checked constraints and overlap witnesses certify their stated negative
results, with other queries explicitly inconclusive.

The exact coordinate cosets also admit a finite temporal check. Write
`k=8*z+r`, with 128 allowed residues: `r_0,r_1` range from 0 through 7,
`r_2` is 0 or 4, and `r_3=r_4=r_5=0`. Integer closure of
`z_i-z_j <= floor((B_ij-r_i+r_j)/8)` decides every guard/coset intersection.
All 106,240 intersections are nonempty; abstract reachability from B47
reaches all 8,192 cell/residue vertices. Each of the 20 unsafe slabs remains
nonempty in all 128 residues. This rules out exclusion by this particular
necessary graph test, not by exact trajectory analysis. Evidence and full
independent enumeration are retained in
`artifacts/research/c6_b53_mod8_temporal_graph_probe.json` and its validation.

A lifted total-coordinate envelope also retains all 20 labels after 301
complete layers and its 250,000-intersection guard. Increasing these guards
or retaining the same residue labels supplies no new established mechanism.
The next proof needs a stronger coupled temporal relation or a barrier
verified on complete return guards, preserving their common source state
and all intermediate exits. An abstract path assembled from separate edge
witnesses does not provide such a state.

The final production campaign passes independent validation of 64 pressure
rows, 1,016 primal/dual extrema, 968 active transitions and 756 ingress
pieces. The targeted suite passes **845 tests in 267.02 seconds**, including
73 new reachable-owner, mode-owner and campaign cases. The unchanged B47
origin, seven input hashes and scientific source archive are recorded in
`artifacts/research/b53_final_validation.json`. This validation closes the
four additional regional claims, while the requested remaining twenty-region
and indefinite-boundedness proof remains open.

## 48. B54: complete return guards exclude two node-0 lower regions

The two remaining layers of information in section 47 have different
roles. The return envelope is a forward outer bound on possible origin
paths. A backward query asks whether a complete unsafe slab can have an
origin-compatible past. Combining them now excludes **node-0 lower masks
4 and 12**, without advancing the actual trajectory.

### Preserve the intermediate state in one shared relation

`src/tnfr/physics/c6_carried_return.py` owns both the exact short-return
envelope and its whole-region predecessor queries. It rebuilds canonical
pressures, RN cells, the full carried origin and the common integer grid.
The transient partition selects node 2's upper displayed value; the
complete one-step relation must have no transient-to-transient edge, and
the original state must be in the complementary base set. These are
verified proof premises, not changes to the dynamics.

For a direct return `a -> c`, retain its exact source guard and increment.
For a two-step return `a -> b -> c`, intersect the incoming guard in cell b
with the outgoing guard **at that same intermediate state**. Translate
this intersection back to a and forward to c, retaining the complete
six-coordinate increment. Separate first-step records retain every
base-to-transient visit, including visits with no subsequent return.
The current D36 instance has 830 ordinary edges and 1,265 return records.

The origin-injected base closure and reconstruction of all transient
visits reproduce the independently validated envelope from section 47.
Resource accounting includes ordinary candidate pairs, return construction,
complete closure layers and transient reconstruction. Incomplete relation
construction retains the supplied domain and exposes no usable partial
relation. An interrupted closure layer is discarded. A return fixed point
still does not assert one-step invariance of joined intermediate hulls.

### Backward exclusion covers every possible visit length

Let Q0 contain unsafe base states together with every base predecessor of
an unsafe transient state. The latter is computed from the complete
first-step records, even where no later return exists. For each exact
return guard G with displacement a, the next backward piece is

```text
G intersect (Q_n - a).
```

Only after applying each complete guard are these pieces joined per base
RN cell. Each resulting complete layer is checked against the unchanged
B47 origin. For both newly excluded slabs, **layer 7 is empty**, and the
origin is absent from every preceding layer. Earlier lengths are therefore
excluded directly; every later layer remains empty. Seven counts returns,
not a selected physical-time simulation horizon. These layers need not be
nested, and an abstract origin collision would be inconclusive.

The original full supplied targets remain in the certificate. Their
intersection with the origin-containing return envelope is justified by
its path-coverage theorem, not by an assumption that every point in that
envelope is reachable. Independent reconstruction of the two positive
queries checks 1,375 intersections, six transient target preimages, the
complete 830-edge ordinary relation and all 1,265 return records. The
production envelope accounts for 10,303 intersections: 5,578 during
construction and 4,725 during closure and intermediate reconstruction.

The existing campaign adds `--method returns`. It reconstructs B53 and
checks its complete retained payload against eight frozen input byte
streams before using D36. The shared unsafe-slab helper uses original RN
cube facets throughout. The result is **38 of 56** complete first-exit
slabs excluded, with **18 unresolved**:

| Pending class | Masks | Count |
|---------------|-------|-------|
| Node 0 lower | 0, 8, 16, 20, 24, 28 | 6 |
| Node 3 upper | 28, 30, 60, 62 | 4 |
| Node 4 upper | 56, 57, 58, 59, 60, 61, 62, 63 | 8 |

The original fixed-phase, unit-capacity `h=1/16` map and B47 carry are
unchanged. B52's finite gate is replayed only as a prior proof premise;
B54 requires zero new trajectory steps and no live graph invocation.
Global C6 boundedness, convergence and post-SHA runtime promotion remain
open. Report: `artifacts/research/c6_winding_return_regions.json`.

### A tested stronger temporal refinement

A separate exact prototype composes pairs of returns before joining their
middle state. Of 85,272 compatible-label pairs, 9,590 have a nonempty
common carried-source guard. Twenty-six complete even-return closure
layers use 249,340 intersections before the computational guard. Coverage
then includes the one-return image for odd visits and every transient
intermediate, avoiding an even-endpoint-only claim. This tightens 53 cells
relative to the single-return envelope, but excludes no additional label
by itself. Its independent validator reconstructs every pair, every
complete layer and the odd/intermediate coverage.

Evidence: `artifacts/research/c6_b54_double_return_probe.json` and its
validation. A backward search on this stronger outer domain finds the
same two positive slabs, with the other 18 still resource-limited. The
useful gain in B54 comes from complete guarded backward exclusion; tighter
forward hulls alone have not supplied the missing full confinement proof.

Removing the two newly excluded slabs and repeating the return audit does
not add an exclusion at the same 500,000-query-intersection guard. A further
backward prototype keeps direct and transient first-return components
separate and intersects each exact guard before joining. All 18 queries
still reach their guard at complete depths 19 or 20, without an empty or
stationary layer or an origin collision. These retained controls are
`c6_b54_post_exclusion_probe.json` and
`c6_b54_partitioned_return_predecessors.json` under `artifacts/research`.
Their unfinished searches do not prove that their targets are reachable.

The final targeted suite passes **925 tests in 346.90 seconds**, including
80 new return-owner, regional-query and campaign cases. Independent
validation binds all eight input hashes, the archived scientific source,
all complete return guards, every original unsafe target and both positive
proofs. Software, documentation and provenance checks are retained in
`artifacts/research/b54_final_validation.json`; source reconstruction uses
`b54_scientific_source_delta.json`. The remaining 18-region proof is open.

## 49. B55: exact unions exclude another complete first-exit region

This block preserves the B47 origin, full carry, source pressure, phases,
unit capacity and `h=1/16`. It changes only the representation of possible
predecessors. A convex hull can fill a gap between two regions and thereby
introduce a spurious path. The new query retains a finite union of closed
integer difference-bound regions for each base RN cell.

For a return edge `e` with exact source guard `G_e`, displacement `a_e`
and target cell `d`, each individual target piece `Z` contributes

```text
Pre_e(Z) = G_e intersect (Z - a_e).
```

The union is formed without a hull. A piece may be removed only when a
retained piece in the same cell contains it. This pairwise subsumption
preserves the exact union. Initial targets remain the full original unsafe
slabs, clipped only by the previously justified path envelope; transient
targets contribute every unjoined first-step preimage. Thus possible last
visits to transient cells are not lost.

Every complete layer is checked against the unchanged origin. A complete
empty layer, or identical consecutive normalized unions with all earlier
layers excluding the origin, excludes all domain-confined visit lengths.
Equal counts alone are insufficient. These layers need not be nested.
The common integer grid and the inherited forward envelope remain outer
approximations, so origin membership would not prove actual reachability.

For **mask 0, node 0 lower**, the complete zone counts are
`1 -> 5 -> 9 -> 14 -> 0`. The origin is absent at every depth. Production
accounting verifies **1,237 intersections** (one initial clipping plus
1,236 predecessor pairs) and **90 subsumption comparisons**. Layer 4 is
empty; all later layers are therefore empty as well. This excludes the
entire original target slab as a first-cube-exit source, not merely a
chosen outward point. It requires no new trajectory step.

The implementation lives in the existing
[`c6_carried_return.py`](../src/tnfr/physics/c6_carried_return.py) owner as
`derive_c6_carried_return_union_exclusions`. Shared preparation for both
query representations rebuilds pressure, RN cells, carried origin and all
intermediate guards. The previous B54 envelope and all twenty B54 query
payloads remain identical. Query limits bound intersections, pairwise
comparisons and the number of pieces; interrupted initialization publishes
no absence claim, and later interruptions retain the last complete layer.

The existing benchmark adds `--method unions`. It reconstructs the whole
B54 result before comparing its retained bytes and binds nine input files.
The final campaign spends 500,000 query intersections and 894,744
subsumption comparisons. Its positive proof raises coverage to **39/56**:

| Pending first-exit class | Masks | Count |
|--------------------------|-------|-------|
| Node 0 lower | 8, 16, 20, 24, 28 | 5 |
| Node 3 upper | 28, 30, 60, 62 | 4 |
| Node 4 upper | 56, 57, 58, 59, 60, 61, 62, 63 | 8 |

The other seventeen queries remain resource-limited. Report:
`artifacts/research/c6_winding_return_unions.json`. Its independent
validator binds the source and nine-input chain, reconstructs original
targets and replays the positive finite-union proof using separate matrix
arithmetic. Global C6 boundedness, convergence and post-SHA runtime
promotion remain open.

### Bounded complementary controls and reuse

Independent exact-union probes with 250,000 intersections per query find
no additional positive among the other seventeen targets. Coordinate-
coset tightening adds none; 61 verified adjacent integer-cut union merges
reduce fragmentation but add none. Rebuilding every return guard after
removing node-0 masks 0, 4 and 12 leaves all completed residual backward
layer hashes unchanged.

A signed embedding `(k0,-k0,...,k5,-k5,0)` additionally preserves pair sums.
Its coherence, integer unary tightening and difference closure preserve
all embedded integer points, as checked by independent finite-set oracles.
The 100,000-intersection forward probe and eighteen backward queries with
50,000 intersections each produce no further exclusion. A separate fixed
partition at zero nodal carry preserves each sign combination before
joining. Four representative targets reach their 100,000-intersection
limits at depths 52, 51, 30 and 21 without a proof. These are inconclusive
bounded searches, not impossibility results for richer domains.

The earlier complete spatial increment lattice is reverified against all
64 B54 pressure rows: its exact basis is `diag(1,1,4,8,8,8)`, of index 2048.
It supplies no additional time-free spatial congruence. The existing
necessary full-return period divisor remains
`2721508902482880257845796439528`; it proves neither a periodic orbit nor
confinement. Reuse that arithmetic result rather than searching the period.

A cumulative backward worklist additionally drops a newly encountered
region if an earlier visited region in the same cell contains it. Across
all seventeen residual targets, its 134 complete frontiers (depths 5--12)
show no cross-depth subsumption and match the ordinary exact-union layers.
Sixteen queries stop at comparison limits and one at an intersection
limit. This bookkeeping alone supplies no further proof; the retained
control is `c6_b55_cumulative_union_return_probe.summary.json`.

Source reconstruction and final checks are retained in
`artifacts/research/b55_scientific_source_delta.json` and
`artifacts/research/b55_final_validation.json`. The remaining proof needs
inductive nonconvex coverage or a stronger coupled temporal barrier, with
the original full-target and pre-first-exit quantifiers preserved.
The focused shared-return and campaign suite passes **146 tests in 168.17
seconds**, including 66 new cases; documentation integrity and lint checks
also pass. These checks concern the implementation and its scoped proof,
not empirical confirmation of the paradigm.

## 50. B56: global counts lose chronology while word budgets recover it

The B47 carried origin, nodal pressure realization, fixed phase, unit
capacity and `h=1/16` remain unchanged. This block distinguishes two kinds
of path information: total displacement with transition counts, and the
joint source constraints of an ordered return word.

### An exact obstruction to the global count relaxation

For return edge `e:c->d`, let `a_e` be its integer carried displacement.
Any finite path from origin label `o` to label `t` has nonnegative integer
counts `n_e` satisfying

```text
sum_e n_e (1_source(e)=c - 1_target(e)=c) = 1_o=c - 1_t=c,
k_target = sum_e n_e a_e.
```

These are necessary conditions from graph incidence and the accumulated
nodal equation. They do not check the RN guards at the successive states.
For the complete 1,265-edge, 32-base-label relation, six signed balanced
integer count vectors `z_i` satisfy `sum_e z_i,e a_e = g_i e_i`, with
`g=(1,1,4,8,8,8)`. Every edge displacement lies in this product lattice,
so the vectors prove equality of the cycle-displacement lattice with it.
An independently checked integer circulation `c_e>0` on every edge has
zero incidence and zero nodal displacement.

Choose any graph path from `o` to a base label `t`, with displacement `p`.
For any `k` in the product lattice, add
`sum_i ((k_i-p_i)/g_i) z_i` to its path counts. This gives a signed integer
solution with displacement `k`. Adding a sufficiently large integer
multiple of `c` makes every count strictly positive. The graph is strongly
connected, and the resulting directed multigraph has the required Euler
imbalance, so an abstract label walk exists. Its carried coordinates need
not satisfy even one complete chronological RN itinerary.

Consequently this relaxation cannot exclude any of the seventeen remaining
slabs: explicit endpoint proposals in the complete original targets pass
both the exact coordinate-coset test and the count construction. Transient
terminal targets additionally retain a valid final source/target guard and
the base endpoint's envelope membership. The earlier chronology is still
unverified. This is a limitation of a proof representation, not a reached
exit or evidence of dynamical instability.

The existing owner provides `derive_c6_carried_return_count_relaxation`
and its `construct_counts` method. Canonical source reconstruction, strong
connectivity, all six generators and the positive circulation are checked
before the theorem is emitted. Zero coordinate gcds are outside its stated
domain. Integer candidates are mathematical witnesses, not new physical
coefficients. The method constructs counts without enumerating the
potentially enormous abstract walk.

### Joint word guards give exact repetition limits

Consider a nonempty closed sequence of return edges with displacements
`a_0,...,a_(m-1)`. Each edge includes its exact intermediate RN guard and
is clipped to the retained source and target envelope. Translate those
source guards by the preceding accumulated shifts and intersect them:

```text
B = intersection_j (G_j - sum_(r<j) a_r),    A = sum_j a_j.
```

No extra first-edge guard is imposed after the final edge of one word.
For `n>=1` consecutive repetitions, the exact common-grid source is
`intersection_(r=0..n-1) (B-r*A)`. If `B_ij` are its original closed
difference bounds and `A_6=0`, the repeated source is obtained by closing

```text
B_ij - (n-1) max(0,A_i-A_j).
```

This formula follows by minimizing the translated bound over the integer
repeat index. For nonzero `A`, the pairwise widths give the finite bound
`n <= 1 + min_(A_i!=A_j) floor((B_ij+B_ji)/abs(A_i-A_j))`.
Exact monotone search then finds the sharp last nonempty source and first
empty source. A closed integer DBM supplies an attaining grid point.
Empty `B` excludes one complete word; nonempty `B` with `A=0` is an
identity on that conditional word class. No case by itself asserts origin
reachability or complete-runtime stability. Interrupted composition/search
cannot emit a maximum.

`derive_c6_carried_return_word_budget` implements this derivation in the
same owner. Two examples use indices in the fingerprinted complete return
relation:

| Ordered edge indices | Base RN labels | Intermediate label | Sharp common-grid maximum |
|----------------------|----------------|--------------------|---------------------------|
| 884, 154 | 42 -> 26 -> 42 | 21 in edge 884 | 1 repetition |
| 182, 72, 116 | 32 -> 18 -> 24 -> 32 | None | 2 repetitions |

The first word can occur once in its relaxed source class. What is
impossible is repeating it twice; requiring its first guard immediately
after one traversal must not be mislabeled as failure of the first word.
For the second word, the node-3 shift per cycle is
`1,189,640,417,057,952`. Three repetitions require their starting points to
span `2,379,280,834,115,904` in that coordinate, while its source interval
has width `1,803,052,714,421,815`: an exact deficit of
`576,228,119,694,089`. A source for two repetitions is retained separately.
These sharpness witnesses are on the relaxed common grid, not asserted
reachable from B47 or on its finer coordinate cosets.

### Integration, tested boundaries and the remaining proof

The existing campaign adds `--method histories`, reconstructs the complete
B55 payload, and binds eleven inputs including the untrusted integer and
word proposals. It verifies all seventeen endpoint/count identities and
the two word budgets. No actual trajectory is extended. Whole first-exit
coverage remains **39/56**, with **17 unresolved**.

Retaining the first complete return-word label in backward hulls remains
inconclusive within its intersection guard. Separately, exact backward
frontiers at depths 3 or 6 were combined with distinct affine gradients
per active cell. Six proposed families are ruled out by independent
positive-dual checks of 284 exact witness inequalities. This refutes those
specific monotonicity/entry-separation criteria, not every barrier or the
stability of TNFR.

An additional finite prefix-memory product forbids the first word twice
and the second word three times wherever they occur. From the unchanged
origin it has 43 reachable vertices and 2,113 arcs; exactly two pattern-
completion arcs are removed. A direct substring oracle checks 9,619 finite
words. Exact balanced generators and a strictly positive zero-displacement
circulation show that the product still admits every original coordinate
coset through incidence/displacement counts. Thus these two exclusions
alone do not repair the count relaxation. This does not rule out combining
the product with joint carried-state guards.

The next proof representation must preserve more of those joint guards
across arbitrary interleavings, including transient terminal visits.
Refine obstructing cycles only after verifying their source constraints
and repetition limits; do not repeat the unrestricted count search or the
same two-pattern count product. A budget for one word does not bound
arbitrary interleavings. Report:
`artifacts/research/c6_winding_return_histories.json`; exact independent
replay and source reconstruction accompany the B56 validation artifacts.
The complementary product is retained in
`artifacts/research/c6_b56_forbidden_word_automaton_probe.json`.

The focused shared-return and campaign suite passes **197 tests in 133.01
seconds**, including 121 new cases. Independent arithmetic verifies all
eleven inputs, six count generators, 1,265 positive circulation counts,
seventeen endpoint witnesses and the two repetition maxima. Nine physical
steps from conditional sharpness witnesses are also checked; these are
not a continuation of the B47 origin. Source reconstruction and final
checks are retained in `artifacts/research/b56_scientific_source_delta.json`
and `artifacts/research/b56_final_validation.json`. These implementation
checks add no whole-region exclusion or physical confirmation.

## 51. B57: joint last-return memory excludes a further whole exit slab

The B47 origin, canonical pressure realization, carried residual, fixed
phase, unit capacity and `h=1/16` remain unchanged. The new partition records
the last complete return edge, including any transient middle cell. It
therefore retains a joint carried-state constraint that a bare RN label or
the two forbidden-word prefix states can lose.

### A descending cover with an explicit origin sentinel

For each complete return `e`, intersect its source guard with the retained
source, target and intermediate envelopes at their proper time coordinates.
Call the result `G_e` and the nodal displacement `a_e`. Initially its
endpoint cover is `Z_e = G_e+a_e`. A compatibility arc `i->j` exists only
when the RN labels compose and `Z_i` intersects `G_j`. The source of this
arc is the endpoint of return `i`; its next displacement is **`a_j`**.

Keep the original zero-coordinate state as a separate sentinel. Its first
images `R_j` are injected into every complete update:

```text
Z_j(new) = hull(R_j, union_(i->j) ((Z_i intersect G_j)+a_j)).
```

Every accepted update recomputes all predecessors. It remains a subset of
the previous cover and continues to contain every compatible origin
history confined to the declared domain. A worklist revisits dependents
when a cover shrinks. Interrupted initialization emits no usable geometry;
interrupted graph construction emits no partial graph; an interrupted
update retains its complete predecessor cover. A resource stop can still
leave a valid outer cover, but does not establish a fixed point or trapping.

The default production memory budget uses 499,995 work items: 4,676 for
guard and root initialization, 85,272 for pair construction and 410,047 for
descent. The complete graph has 9,590 compatibility arcs. Its 41,704 complete
visits make 21,999 strict updates and reduce 1,265 nonempty endpoint classes
to 856. The worklist is still pending; the boundedness claim remains open.

### Whole-target exclusion preserves terminal transient visits

Backward queries initialize the original complete unsafe slab in every
compatible memory class, including every transient first step without a
subsequent return. The sentinel is distinct from later visits to the same
RN cell. Each complete backward layer intersects translated target covers
with the current source guards, retaining separate last-return indices.
Empty or exactly stationary layers prove exclusion only when every prior
complete layer also excludes the sentinel. An abstract origin collision
does not prove an actual trajectory.

For **node 3 upper, mask 30**, the production query starts with four
nonempty terminal-history pieces. It reaches an empty complete layer at
depth **136**, using **76,178** query work items, with no origin membership
at any completed depth. This excludes the full original first-exit slab;
it does not merely eliminate selected points. An independent control using
a weaker 26-layer memory cover also empties, at depth 137 and 74,714
intersections. Its four terminal pairs are `(160,274)`, `(162,292)`,
`(372,274)` and `(374,292)`, indexed in the unchanged canonical return and
intermediate relations. This control does not require the additional
three previously proved node-0 cuts or the later worklist refinement.

### Reuse and boundaries of the method

The existing owner `src/tnfr/physics/c6_carried_return.py` now provides
`derive_c6_carried_return_memory_envelope` and
`derive_c6_carried_return_memory_region_exclusions`. Both rebuild the
canonical nodal source. The existing campaign adds `--method memory`,
reconstructs B56 and binds twelve input files. The query graph has its own
budget; each target then has a separate backward-work budget. This avoids
losing all later targets when an earlier query uses its allowance.

The weaker two-pattern prefix product was also checked with exact state
guards: its 42 nonempty history states and 1,313 guarded arcs reach a fixed
point after one image, with the same projected RN cover as the baseline.
Increasing that forward budget cannot improve this fixed point. Bounded
backward hull and exact-union controls on four representative targets add
no exclusion with that partition. The stronger last-return representation
supplies the new positive result; its resource-limited controls do not
establish a general impossibility for richer histories.

The main report is `artifacts/research/c6_winding_return_memory.json`.
The complete campaign excludes **40/56** original first-exit slabs. The
remaining sixteen are node 0 lower masks 8, 16, 20, 24 and 28; node 3 upper
masks 28, 60 and 62; and node 4 upper masks 56 through 63. Those queries
retain complete layers at their 250,000-work guards; none establishes an
actual exit or an impossibility of further refinement.

Independent arithmetic replays the twelve-input chain, all 1,265 source
guards, all 9,590 memory arcs, the complete accepted worklist updates and
all seventeen target initializations. It verifies every layer of the new
positive exclusion. The focused suite passes **275 tests in 166.60 seconds**,
including 78 new cases. Source reconstruction and final checks are retained
in `artifacts/research/b57_scientific_source_delta.json` and
`artifacts/research/b57_final_validation.json`.

The proposed safe-half-space refinement is resolved by the exact B58
comparison in section 52. Its operator is unchanged, so additional precision
requires preserving distinctions beyond that scalar cut.
Global boundedness, convergence, continuation of the actual post-SHA
runtime and physical correspondence remain separate open claims.


## 52. B58: exact redundancy and selective safe-memory partitions

### Propagated scalar exclusions leave the residual operator unchanged

Keep the B47 origin, original candidate cube, canonical nodal pressure,
full incoming carry, fixed phase, unit capacity and `h=1/16`. The safe
complements of the already excluded node-0 lower slabs (masks 4, 12 and 0)
and node-3 upper slab (mask 30) are necessary conditions before the first
exit from the original cube. They introduce no physical coefficient.

At mask 30, the original node-3 upper grid bound and the canonical
one-step nodal shift give exactly

```text
k3 <= 860302430744065456 - 4795745845901584
   = 855506684898163872,       grid quantum = 2^-113.
```

The next integer is unsafe; equality is safe. Every one of the 50 complete
returns through mask 30 already implies this inequality through its next
in-cube endpoint. It tightens only three of the seventeen terminal guards.
After propagating all four cuts, five complete source guards tighten, but
all 1,265 retained endpoint arrays and 9,590 compatibility pairs equal B57.
The complete clipped backward graph still has the **same 6,292 arcs with
identical integer guards and nodal shifts**. Every initial array for the
sixteen remaining original full targets is also exactly equal. Induction
therefore gives the same abstract backward operator at every depth.
This is an exact redundancy result, not a conclusion from equal counts or
from the four representative finite-query controls alone.

The shared return owner now centralizes query-graph construction and
whole-target initialization. Existing exclusion queries and the B58
comparison call those same helpers, with preserved complete-layer and
resource accounting. `--method memory-cuts` in the existing campaign
reconstructs B57, binds thirteen input files, derives the cuts from verified
positives and compares the complete geometry. Resource-limited comparisons
abstain. The report is `artifacts/research/c6_winding_memory_cuts.json`.
The original cube and targets are recorded separately from the cut domain;
a partially clipped target cannot inherit a whole-target claim.

### Preserve the exact excluded pieces instead of their hulls

The four terminal pairs from section 51 define exact unjoined bad pieces
`P_i` inside memory zones `Z_i`. Any origin path reaching such a piece would
follow its canonical guarded terminal step into the already excluded slab.
Removing these pieces is therefore justified before original-cube exit.
This reasoning does **not** justify removing a joined backward hull: such
an outer approximation can include points that never follow the bad path.

For an integer DBM inequality `k_a-k_b <= c`, its exact violation is
`k_b-k_a <= -c-1`. Splitting `Z_i` by the first violated facet of `P_i`
partitions `Z_i \ P_i` disjointly. Removing constraints redundant relative
to `Z_i` leaves respectively **5, 5, 6 and 6** pieces for memories
160, 162, 372 and 374. Their hulls are exactly the original `Z_i` in all
four cases. None of those exact bad pieces is a single relative half-space.
Thus a single DBM per memory erases this new information immediately.

A bounded prototype splits only those four memories, increasing 856 occupied
classes to 874. For each fixed safe seed `S_j`, its update is

```text
Z_j(new) = S_j intersect hull(R_j, union_i post_j(Z_i intersect G_j)).
```

The persistent seed intersection is essential: abstract predecessor images
can otherwise refill a certified hole. Roots remain covered, every guarded
pair is constructed before refinement, and an interrupted update retains
its last complete outer cover. The proof partition changes neither the
nodal dynamics nor the physical state.

Against an unsplit continuation from the identical B57 cover, complete
synchronous layers show equal projected hulls at depths 0 and 1. The split
cover is strictly tighter in **25, 38, 66 and 75** original memories at
depths 2, 3, 4 and 5, and every projection remains inside its same-depth
baseline. This isolates a real precision benefit from the effects of simply
continuing B57's unfinished worklist. The later bounded worklist tightens
459 projections relative to the sealed B57 cover; that larger count alone
cannot be attributed to the partition.

### Bounded results and the next gate

The partition prototype constructs 6,924 guarded arcs and uses 499,996
forward work items. No original target disappears from its full direct and
terminal observations. Backward tests of node-3 upper masks 28 and 62 stop
at complete depths 195 and 85 with 250,000 work items each; both remain
unresolved. Separate unjoined-union tests on the B57 memory cover also stop
without a new exclusion: node-0 lower mask 8 at depth 14, node-3 upper mask
28 at depth 13 and mask 62 at depth 6. These resource stops prove neither
actual reachability nor impossibility of better certificates.

Exact unjoined predecessors of the excluded target have 4, 9, 12 and 18
pieces at depths 0 through 3. Their cumulative union has no overlap with
current mask-28/62 target initializations. They may still refine intermediate
histories. The retained artifacts are `c6_b58_review_terminal_complements`,
`c6_b58_review_exact_predecessors`, `c6_b58_memory_union_probe`,
`c6_b58_safe_piece_partition_probe` and `c6_b58_safe_piece_partition_control`
(`.py` and `.json`, under `artifacts/research`).

Coverage remains **40/56**, with the same sixteen original slabs open.
The next delivery should integrate selective fixed-safe-piece partitions
into the canonical owner, independently verify their exact coverage and
persistent clipping, and reuse one cover for all remaining whole targets.
Only then should target-relevant exact predecessors or a joint temporal
barrier extend the representation. Repeating the unchanged scalar-cut
operator is unnecessary. No new conditional or live trajectory step,
global boundedness proof, runtime continuation or physical confirmation is
claimed by B58.

Focused regression and report checks pass **142 tests**, including 64 new
cases. The partition validator independently rebuilds all 6,924 arcs and
54,353 complete forward updates, verifies the sixteen target observations
and matched-depth inclusion, and rejects sixteen adversarial partition
corruptions. Its record is
`artifacts/research/c6_b58_safe_piece_partition.validation.json`.
The full cut-campaign replay, scientific source delta and final checks are
retained in `artifacts/research/c6_winding_memory_cuts.validation.json`,
`artifacts/research/b58_scientific_source_delta.json` and
`artifacts/research/b58_final_validation.json`.


## 53. B59: canonical fixed-safe-piece memory certificates

### Proof-derived partitions of the unchanged nodal state

A region here is a set of carried numerical configurations of the six-node
ring. It is not a physical spatial area. The original 56 first-exit slabs
identify configurations whose next canonical nodal step crosses a declared
candidate boundary; they can overlap and their counts are not probabilities
or a percentage of a stability proof. A remaining slab is an unresolved
proof obligation. A candidate exit alone would not prove global divergence.

The selective partition from section 52 is now implemented in the existing
`c6_carried_return.py` owner. Its public entry points are
`derive_c6_carried_return_safe_partition` and
`derive_c6_carried_return_safe_partition_region_exclusions`.
They rebuild the canonical pressure/return/memory source before deriving any
cut. Exclusion proposals are full target regions, without trusted success
flags. The existing complete-layer query must first verify every proposed
exclusion from the unchanged origin. Failure leaves the partition explicitly
uninitialized and cannot supply a new exclusion.

For a verified target, the owner enumerates each direct intersection and
individual terminal preimage separately. It records the original memory,
exclusion group, target ordinal and terminal transition for every bad piece.
It never uses the hull of several predecessor pieces as a removable set.
The same target-piece reader initializes ordinary memory queries and
partition queries, so terminal visits without a later return retain one
semantics throughout the proof.

For source DBM `Z` and exact bad piece `P`, the owner derives a relative
facet basis for `Z intersect P`. Removing a redundant inequality is allowed
only when exact DBM closure remains equal to that intersection. The safe
complement splits by the first violated integer inequality, using
`k_j-k_i <= -b-1` to complement `k_i-k_j <= b`. Each subtraction record keeps
its source zone, bad-piece index, relative facets and disjoint safe pieces.
The reduction does not claim a unique or globally minimum facet basis.
Several bad pieces are subtracted sequentially without joining safe pieces.

### Complete graph construction and persistent source coverage

A partition vertex is `(canonical return index, safe-piece ordinal)`. It is
a proof label, not a newly invented engine edge. Its initial region is its
fixed safe seed; the unchanged B47 point has a separate sentinel. Every
canonical first-return image must remain covered exactly once by the seeds.
All compatible guarded partition pairs are built before forward refinement.
An interrupted construction publishes no partial pair relation.

Every accepted forward update recomputes every predecessor image, injects
the original first image and intersects the result with the fixed seed.
The update is reserved as a complete work unit before it begins. This keeps
all original domain-confined histories covered, preserves the certified
holes and prevents a partial predecessor union from excluding valid paths.
Separate limits bound construction work, pieces, arcs, forward work,
exclusion queries and residual target queries. These are computational
limits; they do not alter phase, capacity, pressure, carry or the nodal map.

The shared backward kernel processes all original full targets over the new
indices, including direct visits, the sentinel and terminal transient steps.
Initialization is charged to each target's budget. Only complete origin-free
empty or stationary layers prove exclusion. An exhausted budget retains the
last complete layer and reports an unresolved result; an abstract origin
collision is not an actual trajectory witness.

### Canonical campaign and next refinement

The existing benchmark adds `--method safe-partition` and binds fourteen
inputs. It reconstructs B58 completely, then uses the original B57 domain to
reverify the excluded mask-30 target. The scalar-cut domain cannot replace
that domain for this step: it has already removed the slab that must be
proved unreachable. The original cube, B47 origin and all full residual
targets remain separately identified. No conditional or live trajectory is
extended by this campaign.

The canonical run reproduces the four complement sizes 5/5/6/6, 874
partition vertices and 6,924 arcs. Construction uses 34,820 work items;
forward descent uses 499,996, with 54,353 complete visits and 29,885 strict
updates. The clipped target graph requires 14,722 work items. Every full
target has a separate budget of 250,000, including initialization:

| Original target family | Masks | Last complete depths | Result |
|---|---|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 | 97, 94, 93, 94, 95 | Resource limit |
| Node 3 upper | 28, 60, 62 | 194, 85, 85 | Resource limit |
| Node 4 upper | 56, 57 | 71, 71 | Resource limit |
| Node 4 upper | 58 through 63 | 70 for each | Resource limit |

No new complete first-exit slab is excluded: the total stays **40/56**, with
sixteen open. Differences from the prototype's depths reflect the declared
accounting and representation; depth alone is not a precision comparison.
The campaign report is `artifacts/research/c6_winding_safe_partition.json`.

A bounded exact-predecessor study reuses the independently certified mask-30
result through twelve complete return layers. Every retained piece carries
its full guarded path, which is recomposed exactly. Its 1,499 retained pieces
require 159,759 intersections and 128,802 subsumption comparisons, with a
separate 17,483-intersection path check. They overlap retained B57 backward
hulls for thirteen of the sixteen remaining target families: node-0 lower
24/28, all three node-3 upper targets and all eight node-4 upper targets.
No single piece covers a complete target-history hull. Node-0 lower
8/16/20 have no such overlap within this bounded search.

There is no direct overlap with the original target initializations in
these twelve layers. Exact paths to a different exit cell must first
complete their prescribed in-cube returns, whereas a first-exit target
requires leaving before the next complete return. This chronology explains
why the approximate backward hulls are the useful refinement target here;
joined target initializations must not silently be treated as exact unions.
The prototype is `artifacts/research/c6_b59_exact_bad_preimages_probe.json`.

The next least expensive extension considers the six additional memory
classes (730, 735, 1199, 1204, 1232, 1237) already present one exact
predecessor step before the four terminal pieces: nine exact additional
regions. It should verify each full guarded path, keep exact disjoint safe
pieces and measure improvement against a matched baseline before increasing
the depth or budgets. Subtracting joined danger hulls remains unjustified.
Global boundedness, convergence, post-SHA runtime continuation and physical
correspondence remain separate open claims.

The focused implementation suite passes **225 tests**, including 83 new
cases. Independent replay checks the original exclusion proof, exact bad
pieces and disjoint subtraction, the complete graph and persistent-seed
worklist, and every original target's complete backward history and resource
stop. Verification and source capture are retained in
`artifacts/research/c6_winding_safe_partition.validation.json`,
`artifacts/research/b59_scientific_source_delta.json` and
`artifacts/research/b59_final_validation.json`.

## 54. B60: exact predecessor cuts and their computational cost

### Guarded paths justify each additional removed set

A verified whole-target exclusion permits removing an exact predecessor of
that target as well. In the last-return memory graph, let `Z_i` be the
retained endpoint cover, `G_ij` the complete clipped transition guard and
`s_j` the carried displacement of return record `j`. For an exact bad piece
`P_j`, the one-return predecessor is

`P_i = G_ij intersect (P_j - s_j)`.

Each nonempty piece retains the original bad-piece index and its ordered
successor return records. Composing that complete guarded word must recover
the same region and terminate in the original exact target preimage. The
deterministic canonical nodal map then makes an origin path to that piece
incompatible with the verified exclusion. A hull containing several exact
predecessors does not have this implication and cannot authorize a cut.
These are refinements of proof sets; the origin, pressure, capacity, phase,
carry, timestep and physical support are unchanged.

The bounded independent control reconstructs nine such pieces one return
before the four B59 terminal pieces, in additional memory classes 730, 735,
1199, 1204, 1232 and 1237. It verifies each complete path before forming
disjoint safe complements and retaining the original first-return points.

### Representation strength and work budget are distinct controls

The depth-one prototype has 916 partition vertices and 7,165 arcs, compared
with B59's 874 and 6,924. Its partition construction costs 37,950 work
items rather than 34,820, excluding the separately recorded predecessor
derivation. At equal complete forward sweeps zero through five, its
projected memory cover is always contained in the depth-zero cover. Strict
improvements affect 0, 0, 0, 2, 3 and 2 memories. Five sweeps cost 40,405
versus 38,990 forward work items. This establishes a bounded precision
gain at matched depth; it does not establish an equal-cost advantage.

At a shared nominal forward budget of 40,000, the depth-zero worklist uses
39,994 work items and completes 4,266 visits; depth one uses 40,000 and
completes 4,205 visits. Across the 1,265 original memory projections,
1,116 are equal, 146 are strictly tighter at depth zero and three at depth
one. No individual memory pair is incomparable, but neither complete
projected cover contains the other. The additional partition is therefore
not a uniform optimization under that fixed budget. The retained control is
`artifacts/research/c6_b60_depth_one_control.json`; its script records its
input hashes and independently recomposes all nine exact paths.

### Canonical integration and verified campaign

The existing safe-partition APIs now accept `excluded_predecessor_depth`,
defaulting to zero with the unchanged B59 result schema and work path.
Positive depth uses `C6CarriedReturnPredecessorPartition`; every generated
piece carries its original bad-piece index and future return-record word.
Complete layer records expose intersections and subset comparisons. Only
individually contained pieces may be pruned; unrelated paths are never joined.
Predecessor graph construction, intersections, comparisons and layer visits
share the existing construction budget. The existing piece limit also
bounds the cumulative bad pieces. An interrupted generation publishes no
usable partition, even if earlier complete layers remain as information.

The shared benchmark accepts
`--method safe-partition --excluded-predecessor-depth 1`, writes the distinct
`c6_winding_safe_predecessors.json` report and binds the unchanged fourteen
inputs. Depth zero remains the default; predecessor depth is a numerical
proof setting. The full sixteen-target campaign and independent replay are
complete. The focused suite passes 141 tests, including 58 new cases;
the 83 B59 regressions retain the default behavior and resource semantics.

The canonical predecessor graph costs 17,146 work items. Its complete
depth-one layer adds 392 intersections, six subset checks and one layer
charge: 399 items. The total construction is therefore 55,495, including
the 37,950 partition cost. The 916-vertex/7,165-arc forward worklist uses
499,998 work items, completes 55,106 visits and performs 31,225 strict
updates. The clipped target-query graph costs 15,246 items. Every original
target receives a separate 250,000-work budget, including initialization:

| Original target family | Masks | Last complete depths | Result |
|---|---|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 | 93, 90, 89, 91, 92 | Resource limit |
| Node 3 upper | 28, 60, 62 | 176, 79, 79 | Resource limit |
| Node 4 upper | 56, 57, 58, 59 | 68, 67, 67, 67 | Resource limit |
| Node 4 upper | 60, 61, 62, 63 | 66, 66, 66, 66 | Resource limit |

There is no additional complete first-exit exclusion: the total remains
**40/56**, with sixteen open. Different partition indices and work per layer
mean that raw depth and occupied-index counts are not precision comparisons.
The report preserves the complete B47 origin, original candidate cube and
every full target; it adds no conditional or live trajectory step. Neither
these resource outcomes nor the bounded controls certify global stability.
A deeper partition requires evidence of utility under the declared resource
limit before becoming the next research step.

The independent comparison at the shared nominal 500,000 forward-work cap
finds B59's original-memory DBM hull strictly tighter in 454 histories and
equal in 811; B60 improves none. This is containment of the hull projections,
not containment of the complete nonconvex partition unions. Construction
work is additionally 34,820 for B59 versus 55,495 for B60. Together with no
new whole-target exclusion, this finite cost result supports retaining the
depth-zero default. It does not make exact predecessor cuts intrinsically
invalid or establish that they can never help another proof computation.
Independent evidence and complete source reconstruction are retained in
`artifacts/research/c6_winding_safe_predecessors.validation.json`,
`artifacts/research/b60_scientific_source_delta.json` and
`artifacts/research/b60_final_validation.json`.

### Next bounded discriminator

An independently recorded 144-intersection read-out compares the nine new
pieces with all retained B59 query hulls. Every piece overlaps a query hull
for each of the eleven node-3/node-4 upper targets; none overlaps a hull of
the five node-0 lower targets. No individual bad piece contains a whole
query hull. Simple overlap selection therefore cannot reduce the nine cuts
for an upper target. The smallest affected B59 frontier is node-3 upper
mask 28: 76 occupied partition histories at complete depth 194. These are
finite proof-set diagnostics, not physical likelihoods or trajectory visits.
The evidence is `artifacts/research/c6_b60_target_relevance.json`.

Since B60 closes no additional target, the single A/B test completed in section 55 retains
its depth-one seeds, root images and arcs exactly and change only the order
of complete forward updates for that original mask-28 target. A seed's
intersection with a retained B59 query hull can supply advisory priority;
it cannot authorize a cut. Charge priority classification and all complete
visits to one shared 500,000-work cap, then use the unchanged 250,000-query
cap. Two FIFO queues keep the priority rule reproducible. Success requires
a new complete exclusion or componentwise strict containment at a common
complete query depth without extra work. Otherwise reject that efficiency
hypothesis for the tested target. Do not increase predecessor depth or
change the nodal map as part of this comparison.
The experiment needs both the sealed B60 FIFO control at identical geometry
and the best verified B59 baseline. Improvement over B60 alone is diagnostic;
adoption requires a useful gain over B59 as well. Compare original-memory
projections at common complete query depth without promoting that comparison
to an unproved containment of the full nonconvex unions.

## 55. B61: priority refinement excludes another original exit label

### Count and coverage premises

The 56 labels are an enumerated property of the retained C6 proof domain,
not a structural constant. Two adjacent binary64 values per EPI coordinate
give 64 rounding cells. Six coordinates and two exit directions give 768
candidate conditions. Exactly 72 are nonempty on the initial cube. The
verified B49 source envelope removes eight node-3 lower and eight node-5
lower conditions, leaving 56. The count uses the original cube boundaries
and canonical carried nodal increments. Nonempty local conditions do not
establish origin reachability, and labels need not define disjoint sets.
The independent reconstruction is
`artifacts/research/c6_b61_region_count.json`.

Later origin covers narrow these geometric slabs without changing their
labels. In particular, let `U` be the original B51 node-3 upper mask-28
slab and `D` the independently certified B53 origin cover. The B57-B60
target carried into B61 is exactly `U intersect D`. The separate coverage
bridge verifies that equality, recomputes all 4,096 clipped B53 image pairs
(830 nonempty), and checks the origin-injected fixed point and four cuts.
It reuses the hash-bound earlier 32-exclusion coverage certificate. The
new label is not among the assumed exclusions. Both `U` and `U intersect D`
remain geometrically nonempty; exclusion below concerns their reachable
history observations. This distinction prevents promoting emptiness on an
arbitrary smaller target into a whole-label claim.

### Advisory priority preserves the complete history cover

The scratch experiment keeps the B60 geometry unchanged: 916 vertices,
7,165 arcs, every persistent seed and every original root image. A seed's
intersection with a retained B59 query piece in the same original memory
assigns priority; it authorizes no cut. Two FIFO queues retain all pending
vertices and their fixed priority. Every accepted visit combines all
guarded predecessor images with the original root, intersects the fixed
seed and commits only a complete update. Consequently, changing the visit
order preserves coverage of every original-domain origin prefix. Finite
resource limits need not reach a fixed point or give a fair infinite schedule.

Classification selects 109 priority vertices and costs 992 comparisons.
Forward updates use another 498,989 work items, for 499,981 under the
shared 500,000 cap. The run completes 32,112 visits and 30,111 strict
updates, leaving 625 vertices pending. These queues are retained explicitly;
their presence prevents a claim of global fixed-point completion.

### Complete target observation and independent verification

The whole mask-28/node-3/upper target has no direct, terminal-transient or
origin-sentinel observation in the refined cover. Full initialization
produces zero pieces and an empty complete layer at depth zero, using
1,697 work items. The clipped query graph costs 15,231 items. The complete
cost, including the B60 partition construction, is
`55,495 + 499,981 + 15,231 + 1,697 = 572,404`.

B59 and B60 FIFO each retain one nonempty target piece in original memory
374 at the same complete depth. The B60 same-index comparison has one
strictly tighter query vertex (283) and 916 equal entries, including the
origin sentinel. Original-memory hull comparison likewise has one strict
improvement. This does not identify the full nonconvex unions. B59's
construction-inclusive depth-zero cost is 551,157; the experiment therefore
succeeds by a new complete exclusion, not by a lower-cost initialization.
No blanket performance claim or default-policy replacement follows.

Since every origin prefix before first cube exit lies in `D`, absence of
all complete observations of `U intersect D` excludes the original label
`U`. Coverage rises to **41/56**, leaving **15** labels:

| Remaining original target family | Masks |
|---|---|
| Node 0 lower | 8, 16, 20, 24, 28 |
| Node 3 upper | 60, 62 |
| Node 4 upper | 56, 57, 58, 59, 60, 61, 62, 63 |

Independent arithmetic replay checks every forward visit, pending queue,
root, seed and full target observation, plus B59/B60 common-depth controls.
Thirty-one separate corruption checks use hash-bound cached arithmetic
expectations to exercise the real evidence admission/comparison assertions;
they are not fresh full arithmetic replays. Twenty-four fresh finite-oracle
controls check reachable-state coverage under two priorities and twelve
budgets, with partial-visit, terminal-only and sentinel checks. Evidence:

- `artifacts/research/c6_b61_target_priority.json`;
- `artifacts/research/c6_b61_target_priority.validation.json`;
- `artifacts/research/c6_b61_original_label_coverage.json`;
- `artifacts/research/c6_b61_priority_adversarial.json`;
- `artifacts/research/b61_final_validation.json`.

The production scientific snapshot remains B60. Physical parameters,
pressure, timestep, phase, origin and carry are unchanged; no conditional
or live trajectory is advanced. Global C6 boundedness, convergence and
post-SHA runtime behavior remain open.

The next implementation gate centralizes the optional priority policy in
the existing return-proof owner while preserving FIFO and depth-zero
defaults. It must reproduce the sealed result. Target-specific scheduling
does not make the retained cover target-specific in its coverage: it still
contains every original-domain prefix. Reuse that cover for the other
fifteen full targets, retaining their coverage premises, terminal visits,
origin sentinel and declared work accounting. No deeper cuts or additional
physical assumptions are needed for this next comparison.

## 56. B62: canonical synergies and proof ownership

### What belongs to the nodal model and what belongs to its representation

The physical variables remain EPI, structural capacity and phase on the
declared graph, with pressure supplied by the existing canonical producer.
The current continuation holds support, phase and unit capacity fixed and
uses the declared timestep `h=1/16`. Those assumptions identify the case;
the nodal factorization alone does not derive their values or choose an
operator schedule. The finite B43 preparation and B47 carried origin remain
the provenance boundary. Future live admission and post-SHA continuation
are separate obligations.

The exact remainder in `X=x+r` is numerical bookkeeping, not an additional
physical TNFR field. The shared kernel advances `X_next=X+h*p(x)` and reads
pressure from the represented `x=RN(X)`. Its validation of this numerical
map is not a solver-convergence theorem for a continuous trajectory. The
integer grid, regions, memory indices, proof coordinates and priority queues
describe or bound that same map. They must never be inserted into its
pressure as new physical feedback.

### A reusable certificate for retained history covers

The descending worklist has a local verification rule that does not require
replaying its discovery order. For canonically certified fixed seeds `S_v`,
root images `r_v`, complete guarded arcs `u -> v` with displacement `a_uv`,
define the monotone map

```text
F_S(Z)_v = S_v intersect hull(r_v,
                           { (Z_u intersect G_uv) + a_uv : u -> v }).
```

If `r_v subset Z_v subset S_v` and `F_S(Z)_v subset Z_v` for every vertex,
induction over complete returns covers every domain-confined history. The
seed coverage theorem supplies the clipping premise. Complete intermediate
and terminal visits and the origin sentinel must still be reconstructed
when observing a target. This is closure of a clipped proof relation, not
physical confinement in the original cube.

Starting from `S`, the inclusion `F_S(S) subset S` holds by construction.
An atomic update replaces one component by its full `F_S` image. Monotonicity
and the shrinking components preserve `F_S(Z) subset Z`, even at a finite
resource stop. Consequently a one-pass closure check can verify the retained
cover independently of the priority history. It must also authenticate the
source map, seeds, roots, arcs and prior exclusions; arbitrary user-supplied
matrices and truth flags are not certificates.

For two covers closed under this same `F_S`, their componentwise intersection
is also closed: monotonicity bounds its image by the image of each cover.
When partitions differ, a hull projection into an original memory may add
spurious states. Intersection with that projection remains a reachable-state
cover if both original coverage premises are valid, but closure under the
other partition's `F_S` requires a separate check. Neither hull comparisons
nor root inclusion alone justify that stronger claim.

### Reuse priorities and mathematical discriminators

The shared return owner already builds target observations for many groups
after one partition construction. The remaining integration should expose
validated retained-cover reuse and the optional priority policy through that
owner, keeping FIFO and depth-zero defaults. It should not create a second
production DBM, pressure, return-graph or target-query implementation.

New original-label exclusions can feed the exact unjoined predecessor and
safe-subtraction owners. They must be reverified independently of the new
cuts, preserve all roots and terminal visits, and demonstrate a nonredundant
refinement before a broader campaign. Merely repeating old scalar cuts,
deepening every predecessor or increasing work limits has already failed to
supply an equal-cost gain on the retained controls.

For the new mask-28 exclusion this distinction matters operationally: its
B61 initialization is empty, so extracting bad pieces from that same cover
adds nothing. B59/B60 still expose an exact piece in original memory 374.
A reusable exclusion adapter would bind the independently verified B61
premise to that weaker reference before exact predecessor subtraction.
Requiring the weaker memory-only query to rediscover the stronger proof
would discard the very synergy being tested.

The next dynamical synergy is local pressure residence combined with guarded
return memory. A node's pressure on the fixed phase slice depends on its
local EPI stencil; several pending labels share that stencil. Its signed
nodal area bounds a consecutive residence, but a changed neighbor ends that
premise. Any certificate must retain ingress, carry and pressure changes
across successive local episodes. A local deadline alone excludes neither
reentry nor the first exit. High-node-2 terminal visits cannot be discarded
just because the low-return graph omits them as vertices.

Mean and shape also remain coupled obligations. The existing exact profile
identity separates spatial contraction from the signed mean contribution of
the represented phase source and pressure rounding. A memory-dependent
budget on a newly refined history cover is a legitimate new discriminator;
the refuted common mean separator, global displacement counts and unguarded
label cycles are not. Proof coefficients may be proposed and then checked
exactly, but they do not become fitted pressure parameters.

No symmetry quotient follows from the cycle graph alone: phase source,
binary64 chart, full carried origin, guards and targets must respect the
same permutation. Likewise, tetrad diagnostics cannot replace the carried
state and chronological constraints required by these exclusions.

### Bounded cover controls and decisions

The retained evidence supports a finite discriminator before another full
campaign. `artifacts/research/c6_b62_cover_synergies.json` observes all fifteen
pending full targets on five covers: B59, B60, B61, B61 intersected with
B59's original-memory hull, and the same-index intersection of B61 with B60.
Every initialization retains direct, terminal and origin-sentinel checks.
All 75 initializations are nonempty. The study performs zero forward
refinement visits, backward layers or trajectory steps: it establishes
neither another exclusion nor failure of a future backward query.

Intersection construction uses 1,832 operations. Three one-pass image checks
cost 8,081 each; two complete query-graph constructions cost another 30,462,
for 56,537 geometry operations in total. Target observations cost 103,620.
These are declared proof-operation counters, not wall-time speedup estimates
or the historical cost of discovering the parent covers.

The B61 image inclusion passes, as does the same-index B61/B60 intersection.
The B59-hull intersection retains roots and stays inside seeds but fails
image inclusion at thirteen vertices. Its reachable-state coverage still
follows from the two original premises; a standalone one-pass certificate
on the B60 graph cannot replace those premises. This control demonstrates
why matching memory indices and distinguishing hulls from unions matters.

The original-memory projections of B61 are stricter in 53 memories, weaker
in 377, incomparable in 50 and equal in 785 relative to either baseline.
Both intersections improve 103 projections versus their respective baseline
and 427 versus B61, yet their closure outcomes differ. Equal counts do not
identify the sets or their proof properties. These are projection results,
not full nonconvex-union containment claims.

Both inspected clipped query graphs have 911 active vertices including the
origin sentinel, four strongly connected components and one component of
908 vertices. Every one of the 910 history vertices is graph-reachable from
the sentinel; graph reachability does not establish a compatible carried
trajectory. A coarse component decomposition supplies little separation
here. The two node-3 upper targets remain terminal observations: B61 reduces
their piece counts from 11 to 10 and 14 to 13 relative to B59/B60 without
removing them. They must remain explicit in every subsequent certificate.

The independent static nodal study rederives all 64 represented pressure
rows and calls `observe_c6_frozen_pressure_stencil` for each pending family.
The sufficient uniform bounds are 189 confined updates for node 0, 180 for
node 3 and 3,759 for node 4. Each is the floor of the closed center-cell
width divided by the absolute exact nodal increment. It assumes unchanged
local displayed triples at every endpoint and is not a sharp deadline from
B47. Node 3's upper targets are already high-node-2 transients, making that
residence bound unhelpful alone. Node 0 is the first proposed history/age
relevance test; a literal 3,759-state node-4 counter needs a demonstrated
benefit before its construction. Every two-step return must update this
observer twice and preserve any intervening stencil change.

Among all twelve literal dihedral coordinate permutations, only identity
preserves the original candidate, and only identity preserves the represented
fixed phase-source vector. This is a finite obstruction to a bare coordinate
symmetry shortcut for this case, not to all other mathematical transformations.
Exact inputs and rational bounds are retained in
`artifacts/research/c6_b62_nodal_synergy_controls.json`; implementation owners
and the proposed bounded tests are mapped in `c6_b62_nodal_synergies.md` and
`c6_b62_proof_synergies.md` in the same directory.

Accordingly, B62 identifies reusable cover certification and priority
integration, implemented in B63 below, followed by a measured nonredundancy
check for the new exact mask-28 predecessor cuts and local-stencil/history
budgets. Do not repeat
the five-cover initialization test or a bare component split as if either
were an untested route to closure. Coverage remains **41/56**, with **15
pending**. No production source or physical parameter changes in this study.

## 57. B63: shared priority discovery and retained-cover verification

The shared return owner now implements the B62 integration gate. The
existing FIFO safe-partition APIs and their depth-zero defaults retain their
result classes, fields and work semantics. Only their descending update
loop is extracted into one private kernel. Two new public entry points,
`derive_c6_carried_return_safe_cover` and
`derive_c6_carried_return_safe_cover_region_exclusions`, reuse the same
construction, subtraction and whole-target query owners.

### Primitive reconstruction is the admission boundary

The public functions accept canonical source/state/domain primitives,
exclusion proposals, explicit resource limits and optional exact candidate
matrices. They reconstruct the canonical return-memory relation, reverify
the excluded target and rebuild every safe seed, root and guarded arc.
An externally constructed partition, certificate object, arc list or success
flag cannot replace those premises. Mutable caches on a supplied pressure
reference are rederived from its primitive inputs.

Supplying a retained candidate skips forward discovery. Admission requires
one closed seven-by-seven integer DBM or `None` per canonical vertex, exact
root/seed inclusion and the complete image inequality from section 56.
Malformed integers, including booleans, are rejected. An incomplete or
failed check exposes no accepted retained cover and cannot authorize a
target query. Strict image inclusion is sufficient; equality is not required.
The query entry point reconstructs and checks once, builds its clipped graph
once and shares it across independently quantified complete target groups.

### Advisory scheduling and complete accounting

Without hints, the extracted kernel follows the old FIFO order. Optional
ordered `(original_memory_index, DBM_or_None)` hints select fixed priority
through seed intersection. They authorize no cut and need not be accepted
as a physical trajectory or an exclusion proof. Shape/index/closure validation
is separately bounded and reported; seed classification and complete updates
share the forward-work cap. A seed without a matching advisory class still
costs one classification visit. Supplying both hints and a retained candidate
is rejected because discovery and candidate checking are distinct modes.

Every forward visit reserves its entire predecessor-plus-seed work before
committing. Both pending queues, priority flags, complete visits and strict
updates are retained. The companion schedule/check records avoid changing
the legacy partition schema. Exhaustion during classification, cover checking,
query-graph construction, initialization or a backward layer yields an
explicit incomplete result rather than an exclusion.

### Canonical campaign and regression boundary

The existing benchmark enables the new path with

```text
python benchmarks/c6_winding_invariant_region.py --method safe-partition \
  --excluded-predecessor-depth 1 \
  --priority-input artifacts/research/c6_winding_safe_partition.json \
  --priority-target 28 3 upper
```

It retains all fourteen original ancestry inputs and separately hashes the
B59 advisory snapshot as input fifteen. The original source/domain, cube
and complete target labels come from canonical ancestry reconstruction;
the advisory file supplies scheduling data only. Output is
`artifacts/research/c6_winding_safe_cover.json`, with claim
`O3.a-C6-carried-safe-cover` and `B63` fields. A default invocation without
priority input continues to produce the historical B59/B60 report path.

The integration reproduces the B61 priority geometry and counters exactly:
109 priority vertices, 992 classification operations, 499,981 forward work,
32,112 complete visits, 30,111 strict updates and 625 pending vertices.
The new explicit admission costs are 950 hint-validation operations and
12,655 retained-cover check operations: 4,574 validation/inclusion checks
plus the 8,081 image operations already identified in B62. The full mask-28
target still has empty initialization, including terminal and origin cases.
These are computational certificate costs, not new physical parameters.

The full fifteen-input production campaign reconstructs the ancestry and
queries all sixteen original pending labels. Mask28/node3/upper is excluded
with 1,697 initialization operations; the other fifteen queries each stop
at their 250,000-intersection limit. Thus the total remains **41 of 56**
whole first-exit slabs excluded. No new slab beyond B61 is claimed, and
an incomplete backward query does not establish a reachable exit.

The focused suite passes **232 distinct tests**, including 91 new cases:
65 shared-owner controls and 26 benchmark/report controls. The initial
mixed regression run contains 23 report cases repeated within the final
26-case run; these are counted once. Controls cover finite reachable-state
oracles, malformed and forged certificates, complete image inclusion,
advisory scheduling, budget exhaustion and legacy serialization.
The scientific snapshot is retained by
`artifacts/research/b63_scientific_source_delta.json` and its companion ZIP,
with full source digest
`sha256:85cc69c97b874814270be42b0592a337134e49b623515cc072f7f3ab51cd1063`.

Independent validation in
`artifacts/research/c6_winding_safe_cover.validation.json` replays the
priority arithmetic and all sixteen complete target queries. It verifies
three cover checks against B62: the B61 candidate and same-index B60
intersection pass; the projected B59-hull intersection fails at the same
thirteen vertices. The shared FIFO kernel reproduces the exact B59 and B60
retained arrays, queues and counters; these controls are captured once in
`c6_b63_fifo_controls.json` and bound by the final validator. All fifteen
input snapshots and scientific source hashes remain unchanged. The official
report digest is
`74dd580f738c64b61bdcd28d48cef8d7070cd1f559f1fec05ac0e208e47d60e7`.

Scope remains domain-confined origin-history coverage. In particular,
accepting `F_S(Z) subset Z` on the clipped graph does not prove confinement
in the original cube, future live admission or post-SHA runtime stability.
Every canonical pressure, physical setting and initial carried coordinate
is unchanged by this implementation.

## 58. B64: exact exclusion transfer and local chronology

B63's mask28/node3/upper exclusion can supply constraints to a weaker
history cover. The premise is the independently verified whole original
target under the same B47 origin and domain, including its original-label
coverage bridge. A success flag or a retained hull alone is insufficient.
The detached study binds the original cube, return relation, signed shifts,
memory relation and full target matrices before reusing that premise.

### New exact pieces and their limits

In each B59/B60 retained cover, the target has one terminal preimage at
partition vertex 283, original return memory 374, terminal transition 290.
The two covers give different exact matrices, so their pieces are derived
separately. The shared target owner retains the individual terminal guard;
it does not subtract a joined backward hull. For a guarded return
`k_next=k+a`, the exact predecessor of a forbidden piece `P` is
`G intersect (P-a)`. The complete return guard already includes its
intermediate visit where present.

One complete predecessor layer gives nine further pieces, in original
memories 631, 730, 735, 1142, 1173, 1199, 1204, 1232 and 1237. Sequential
exact integer-DBM subtraction shows that all ten pieces have nonempty
remainders outside the old mask30 bad-piece unions, including their B60
depth-one extension. Each piece is disjoint from those old unions. Feasible
integer potentials witness geometric nonredundancy; they are not claimed
to satisfy the finer runtime arithmetic class or to be reachable from B47.

All ten pieces intersect both weaker retained covers, but none intersects
the B63 retained cover or any first-return root. This is consistent with
B63's complete image inclusion and empty target initialization: guarded
predecessors on that same complete operator cannot supply a new forbidden
piece inside its already accepted cover. Transfer into a weaker partition
is therefore a distinct question from improving B63 directly.

The bounded extraction, exact differences, cover intersections and complete
initialization controls cost 39,155 operations for the B59-derived pieces
and 40,251 for the B60-derived pieces. Of these, target extraction costs
1,619/1,703 and predecessor construction costs 17,302 in either case,
including the 17,146-operation complete memory graph and 155 guarded
intersections. Budgets are 200,000 operations and 5,000 pieces per variant;
they are proof-computation limits, not physical coefficients.

Subtracting these pieces leaves all fifteen pending whole-target
initializations unchanged in each weaker cover. Thus this test supplies
new nonconvex geometric information but no additional regional exclusion.
Exact evidence is retained in
`artifacts/research/c6_b64_mask28_cut_probe.json`; all computation reuses the
shared return/DBM owners with unchanged scientific source.

### Matched transfer control

The new pieces are tested in a B59 continuation from its retained cover,
with identical fixed starting state and a 40,000-operation forward budget.
The baseline, direct terminal cut and full ten-piece cut are kept separate.
Each child seed is an exact safe difference; roots and every old guarded
arc are clipped against all appropriate children. None of the initial
parent DBM hulls changes: joining the new safe pieces immediately would
discard their nonconvex information.

| Variant | Classes | Arcs | Construction work | Used forward work | Tighter / weaker / equal final parent hulls |
|---|---:|---:|---:|---:|---|
| Baseline | 874 | 6,924 | 23,394 | 39,995 | 0 / 0 / 875 |
| Direct cut | 875 | 6,959 | 23,589 | 39,998 | 0 / 12 / 863 |
| Ten cuts | 920 | 7,256 | 26,187 | 40,000 | 5 / 188 / 682 |

The final comparison includes the original zero sentinel and concerns
parent DBM hulls only; it does not compare complete nonconvex unions.
Atomic visits account for the small unused forward budgets. All cover
gates pass and all 45 complete pending-target initializations remain
nonempty. Declared incremental costs, including cover checks, query-graph
construction and observations, are 110,229/110,563/115,449; inherited cut
derivation additionally costs 0/1,619/18,921. This bounded control supplies
no componentwise improvement at the selected budget and does not justify
promoting either split policy. It does not exclude improvement at a
different, separately justified budget or for another representation.
Evidence: `artifacts/research/c6_b64_cut_transfer_control.json`.

### Exhausting the local node-0 hull calculation

The five pending node-0 targets share the same displayed `(5,0,1)` stencil.
In the B63 cover, this predicate selects 35 history vertices. Its complete
clipped graph has 228 incoming arcs from outside the class and only six
internal arcs, with no composable internal pair. Outside histories remain
held at their authenticated B63 cover; they are not deleted or considered
unreachable. Every two-step return is expanded when checking the stencil,
and standalone terminal visits are audited separately.

Two complete synchronous local image passes, 271 operations each, reach
the local fixed point. Three zones tighten, at vertices 188, 215 and 243.
The full global image remains included in the resulting cover, while all
outside zones and original roots are preserved. Complete target arrays
tighten for masks 16 and 24, but the five nonempty history counts remain
1/9/9/7/6 for masks 8/16/20/24/28. The independent replay agrees exactly.

The local DAG exhausts this particular hull update with held exterior.
It is not a global bound on repeated departure and reentry. Introducing a
189-state residence counter without first testing its relevance would add
complexity to a local chain that already has no two successive internal
return arcs. The appropriate finer control is the exact union of separate
ingress paths, retaining intermediate and terminal observations.
Evidence: `artifacts/research/c6_b64_local_stencil_probe.json` and its
independent `.validation.json`.

### Separate ingress paths retain additional correlations

The next bounded control uses exact unions at the same 35 local vertices,
holding every exterior history at its B63 cover. For each exterior ingress,
it retains the guarded image as one piece with its source preimage and
complete nodal word. The local DAG then propagates each individual piece,
without a convex join. There are 228 ingress pieces, no local root pieces
and 97 possible internal extensions; 66 extensions are empty, leaving 31.
The complete local representation therefore contains 259 pieces, below
the 325-piece a priori bound. This changes the proof representation, not
the nodal pressure or any runtime variable.

Every original root and guarded arc is checked against the combined
representation: the local image has a retained individual-piece witness,
while the exterior remains inside the inherited cover. Admission uses
these exact pieces; their hulls are comparison diagnostics only. The local
union's hull is strictly tighter than the local DBM fixed point at vertices
20, 63 and 65. This demonstrates additional information lost by joining
ingresses before propagating them, even after the hull update has converged.

Complete target observations include all exterior histories, every local
piece, terminal first steps and the original zero sentinel. For masks
8/16/20/24/28, the nonempty piece-history counts are 2/20/13/17/7; they
project to the same 1/9/9/7/6 original histories. All five targets remain
nonempty. These are overapproximated possibilities, not proven trajectories.

Charged federation work is 30,442 operations: 689 for construction, 21,916
for complete root/image coverage and 7,837 for target initialization.
The inherited B63 cover gate additionally costs 12,655 operations, and two
shared-owner initialization cross-checks cost 15,674. These validation
costs are separate from the 30,442 counter. The retained evidence is
`artifacts/research/c6_b64_local_ingress_federation.json`.

The regional total remains **41/56 excluded, 15 pending**. The next useful
constraint must affect the surviving exterior ingress histories or a
correlation those histories still lose. Repeating the local hull fixed
point, joining the ingress pieces again, or adding the unused 189-state
counter does not address that remaining boundary. Production dynamics,
the B47 origin, physical parameters and historical artifacts are unchanged.

The smallest surviving family is mask8/node0/lower: pieces 19 and 55 at
vertex 180 (memory 262, `48 -> 8`) enter from exterior vertices 45 and 97
(memories 84 and 156, `18 -> 48` and `26 -> 48`). Those two exterior
histories have eight and 26 incoming original arcs, respectively, and no
crossedge. The next bounded test is their complete 34-candidate exact
prehistory layer. The other targets have 18/13/15/7 distinct exterior
feeders; broad history growth is deferred until the smallest family has
been evaluated. Provenance and implementation notes are centralized in
`artifacts/research/c6_b64_local_history_design.md`.

All four B64 studies have independent arithmetic replays in their matching
`.validation.json` artifacts. The exact predecessor-owner regression suite
passes 22 tests; this block changes no scientific production source. The
combined evidence, unchanged source and documentation checks are captured
by `artifacts/research/b64_final_validation.json`.

## 59. B65: a complete mask-8 prehistory layer

B65 evaluates the smallest B64 family without enlarging the domain,
changing the pressure, advancing the origin or joining predecessor pieces.
The complete federation initialization of mask8/node0/lower has exactly
two direct observations, pieces 19 and 55 at local vertex 180; its terminal
and zero-sentinel observations are empty. The original B51 unsafe slab
intersected with the certified B53 pre-exit domain is rechecked to equal
the complete refined target. That domain equality and the inherited
origin-history cover remain distinct premises of the calculation.

For a target observation `Q_b` reached on an ingress `a -> b` with guard
`G_ab` and shift `s_ab`, retain its exact feeder preimage

```text
B_a = Z_a intersect G_ab intersect (Q_b - s_ab).
```

For every original incoming return `c -> a`, independently compute

```text
P_c = Z_c intersect G_ca intersect (B_a - s_ca).
```

Each complete return guard includes any intermediate nearest-rounding
visit. Local roots, feeder roots and the surviving source-root intersections
are checked separately. The resulting pieces are necessary prehistories
within the fixed B63 cover; nonemptiness proves neither coordinate-coset
feasibility nor actual reachability from B47. In particular, a surviving
piece cannot be subtracted as though the target had been excluded.

The two feeders have eight and 26 incoming arcs. Of these **34 complete
candidates, 27 are empty and seven survive**:

| Local piece / feeder | Incoming arc | Source vertex | Original return memory |
|---|---:|---:|---:|
| 19 / 45 | 2 | 1 | 13 |
| 19 / 45 | 36 | 6 | 22 |
| 19 / 45 | 2082 | 138 | 208 |
| 55 / 97 | 217 | 21 | 43 |
| 55 / 97 | 474 | 39 | 74 |
| 55 / 97 | 1108 | 66 | 115 |
| 55 / 97 | 1396 | 87 | 146 |

All checked root intersections are empty. The calculation uses 86 charged
DBM intersections: one original-target bridge, two observation checks,
four ingress intersections, four local/feeder-root checks, 68 predecessor
intersections and seven surviving-source-root checks. This counter does
not include inherited evidence verification, arc enumeration, exact
subset/word audits or independent replay. No cover is modified and no
second prehistory layer is run.

### Common nodal content, distinct full histories

The surviving constrained suffixes are `18 -> 48 -> 8` and
`26 -> 48 -> 8`. In this data each return is one nodal step; there are no
hidden intermediate rows. With the unchanged `q=2^-113`, `h=1/16` and
unit capacity, both suffixes have exact node-0 increments

```text
(h * DeltaNFR_0(first), h * DeltaNFR_0(48))
    = (259574703981392*q, 259574703981392*q).
h * DeltaNFR_0(8) = -3039824576180305*q.
```

The two increments and the hypothetical following mask8 update sum to
`-2520675168217521*q`. This is the signed accumulated nodal budget for
those specified rows, not an observed trajectory or a repeated-cycle
theorem. The same local node-0 pressure can occur with its left/right
displayed neighbor values exchanged. Equality of that scalar read-out
does not identify the six-coordinate pressures, carry guards or source
histories; the two families therefore remain separate in the proof.

The next complete layer would have 55 candidates. Its smaller complete
family is feeder 45: sources 1, 6 and 138 have two, five and nine incoming
arcs, respectively, giving **16 candidates plus three root checks**.
Feeder 97 requires a separate 39-candidate family. These counts are not
full computation budgets: each candidate needs up to two DBM intersections,
with relation enumeration, provenance and validation separately counted.
The authenticated previous-memory labels provide context; they have not
been promoted to an additional replayed predecessor layer.

The regional count remains **41/56 excluded and 15 pending**. The 27
eliminations are prehistory candidates, not 27 additional regions. Full
records are retained in `artifacts/research/c6_b65_mask8_prehistory.json`;
its independent validation and the nodal interpretation are sealed by
`c6_b65_mask8_prehistory.validation.json` and `b65_final_validation.json`.

## 60. B66: exact geometric gain and reuse of local histories

This block tests whether another prehistory layer actually removes geometric
possibilities, rather than merely renaming them. It retains the B63 source,
B47 origin, nodal pressure, timestep and rounding model. The complete B65
target/domain, root, terminal and sentinel checks remain sealed premises.
The selected family is piece19/feeder45; feeder97 remains unchanged.

For a retained parent piece `P` at vertex `v`, include its root intersection
and **every** original incoming arc `u -> v`. Its necessary child piece is

```text
Q_uv = Z_u intersect G_uv intersect (P - s_uv).
H_v = (root_v intersect P) union union_uv (Q_uv + s_uv).
```

`H_v` is a subset of `P`. Exact integer-DBM subtraction computes
`P minus H_v` without joining the images. A nonempty difference proves a
loss of possibilities at the same parent coordinates. Counting empty
children alone does not establish this gain. Translations through the
already constrained suffix permit the same comparison at the target.
These are target-conditioned restrictions; they are not globally forbidden
cover zones and cannot be promoted to cuts without a new admission argument.

### A complete selected family, with measured gain

All two, five and nine incoming arcs of parent vertices1,6,138 are checked,
including the three root cases. Of **16 candidates, 12 are empty and four
survive**. All surviving source-root intersections are empty.

| Parent | Surviving incoming arcs | Source vertices | Complete nodal word |
|---|---|---|---|
| 1 | 2953, 3235 | 180, 207 | `8 -> 18 -> 48 -> 8` |
| 6 | none | none | no surviving piece |
| 138 | 232, 569 | 22, 41 | `40 -> 18 -> 48 -> 8` |

Each return in these words is one nodal step. Their shifts are rederived
from `h * DeltaNFR`, with every canonical guard and any intermediate visit
audited. A repeated displayed row in the first word does not identify the
carried state or prove a periodic orbit.

The exact differences at the three parents have4,1,9 pieces, respectively.
The one difference at vertex6 is its entire former parent piece. The union
of all four surviving target endpoint pieces is also strictly smaller than
the previous selected-family endpoint union, but remains nonempty.

### Reusing an admitted cover reveals information that a hull loses

Two surviving sources, vertices180 and207, already have complete exact B64
local covers. Their three and eleven existing pieces can therefore be
intersected with the new preimages without generating another history layer.
All14 intersections are evaluated. At180, pieces19,55,133 survive; at207,
pieces7,21,35,57,125 survive. Sources22 and41 lie outside this local cover
and retain their B63-derived preimages unchanged.

There are consequently eight local intersections plus two exterior pieces.
Although this representation has more pieces, its exact endpoint union is
strictly smaller. Subtracting it from the four pre-reuse endpoint pieces
leaves2,8,0,0 residual pieces. The DBM hull before and after this reuse is
**identical**: joining the pieces would erase all of this additional gain.
Piece counts and hull comparisons therefore cannot replace exact-union
comparison. This is a demonstrated synergy between the B64 admitted cover
and B66 prehistory, not a new physical memory variable.

Charged work is726 DBM operations: three parent-root checks,32 candidate
intersections, four child-root checks,130 parent-union difference work,
272 selected-family endpoint difference work,14 reused-piece intersections
and271 further endpoint difference work. Hash binding, enumeration, nodal
word/subset audits and independent validation are separate. No production
source, runtime trajectory or physical coefficient changes.

An additional complete-observation relevance check restores all four
unchanged feeder97 endpoint pieces. Comparing the seven B65 endpoint
pieces with the fourteen B66 pieces gives exact residual counts
`14,2,28,0,0,0,0`, at a separate cost of2,243 subtraction operations. Thus
the gain is still strict for the combined necessary mask8 endpoint union,
not merely within one feeder. Its hull also shrinks across the complete
B65-to-B66 change; this is distinct from the unchanged hull in the second,
reuse-only phase. The combined union remains nonempty.

Evidence is retained in `artifacts/research/c6_b66_feeder45_gain.py/.json`,
with independent replay in `validate_b66_feeder45_gain.py` and the matching
`.validation.json`. `b66_final_validation.json` seals the complete checkpoint.
The regional result remains **41/56 excluded, 15 pending**. Neither feeder45
nor the whole mask8 target is excluded; feeder97's four B65 candidates are
unchanged. Nonempty DBMs do not prove finer coordinate-coset feasibility or
origin reachability. Indefinite C6 boundedness and future runtime remain open.

### Research value and the next decision

These calculations address this numerical C6 proof; their necessity for
understanding TNFR in general is not established. The reusable content is
exact accumulated nodal accounting, complete guarded induction and measured
control of information lost by joining histories. Binary64 carry and proof
history labels do not introduce new ontological primitives. This block
establishes neither a general multichannel stability theorem nor physical
correspondence with laboratory observations.

The next bounded gate is feeder97's complete39-candidate layer at B65 sources
21,39,66,87 (5,15,2,17 incoming arcs), with all root cases. Preserve the ten
new feeder45 endpoint pieces and reuse existing admitted local pieces where
applicable. Measure exact endpoint-union loss both within the selected
family and after restoring the other family. Do not increase depth merely
because many child candidates are empty; require demonstrated geometric
gain, a smaller complete certificate or a new useful nodal discriminator.
The broader mechanism programme should distinguish contraction of spatial
differences from control of signed uniform drift, using the existing
mean/shape, diffusion and event-budget owners. Its scope and the C6 proof
engineering boundary are reviewed in
`artifacts/research/c6_b66_research_value_review.md`.

## 61. B67: the second mask8 family and a shared refinement kernel

B67 completes the feeder97 gate while retaining all ten B66 feeder45
endpoint pieces. It uses the same B65 target-conditioned parent preimages,
sealed whole-target/domain and origin-cover premises, and unchanged B63
scientific source. The detached orchestration is centralized in
`artifacts/research/c6_exact_history_refinement.py`: complete original-arc
pullback, nodal word/guard checks, admitted local-cover reuse and exact-union
comparison. Production remains the owner of DBM geometry. Historical B65/B66
scripts and certificates remain immutable; the new helper does not admit a
new cover or supply a new physical rule.

### Complete incoming histories and reuse of known restrictions

All 5,15,2,17 incoming arcs at parent vertices21,39,66,87 are evaluated.
Parent and surviving-source root intersections are empty. Of **39
candidates, 32 are empty and seven survive**:

| Parent | Incoming arcs | Source vertices | Complete nodal word |
|---|---|---|---|
| 21 | 2011, 2023, 2071, 2329 | 125, 130, 136, 151 | `16 -> 26 -> 48 -> 8` |
| 39 | none | none | no surviving piece |
| 66 | 2441 | 155 | `24 -> 26 -> 48 -> 8` |
| 87 | 4994, 5570 | 331, 458 | `26 -> 26 -> 48 -> 8` |

Every displayed return is one nodal step. In particular, the repeated26
row is a guarded translation of carried coordinates, not a stationary
carried state. Exact parent-minus-image differences have11,1,2,11 pieces;
the second is the entire former parent39 piece. The selected-family
endpoint union strictly shrinks, with residual counts11,4,1,1 against its
four B65 endpoint pieces.

Five surviving sources already belong to B64's admitted local federation.
Every existing piece at each such source is tested; exterior sources331
and458 keep their preimages unchanged:

| Source | Existing local pieces tested | Nonempty local piece IDs |
|---|---:|---|
| 125 | 1 | 46 |
| 130 | 4 | 32, 47 |
| 136 | 3 | 16, 49, 115 |
| 151 | 19 | 6, 17, 33, 51, 104 |
| 155 | 22 | none |

Thus **49 intersections leave11 local pieces and two exterior pieces**.
Source155's entire conditioned preimage disappears; this also eliminates
the last alternative under B65 parent66, using existing evidence without
another predecessor layer. The remaining family has13 endpoint pieces.
The reuse-only exact differences against the seven preceding endpoint
pieces have0,1,15,27,4,0,0 components.

### Complete-observation gain, without a hull gain

Restoring the ten unchanged feeder45 pieces gives23 endpoint pieces in
place of the14 retained after B66. The complete necessary mask8 endpoint
union is strictly smaller. Subtracting the new union from the old pieces,
ordered as ten feeder45 pieces followed by four feeder97 pieces, gives
ten zero counts followed by40,38,7,7. The whole-observation DBM hull is
**unchanged**. More pieces encode fewer possibilities here; a hull-only
metric would miss the gain for the complete observation.

Charged producer work is5,495 operations: four parent-root checks,78
candidate intersections, seven surviving-source-root checks,326 parent
difference operations,509 selected-family difference operations,49
existing-piece intersections,1,151 reuse difference operations and3,371
whole-observation difference operations. Enumeration, hash/source binding,
nodal word/subset audits, independent replay and compatibility controls are
separate. No origin trajectory or additional prehistory layer is executed.

The retained report is `artifacts/research/c6_b67_feeder97_gain.json`,
produced by `c6_b67_feeder97_gain.py`, with independent arithmetic in
`validate_b67_feeder97_gain.py` and its matching `.validation.json`.
The final provenance, documentation and next-gate checks are sealed by
`b67_final_validation.json`. Coverage remains **41/56 excluded, 15 pending**.
The complete target and both feeder families remain nonempty. Eliminating
parents39/66 concerns these target-conditioned histories, not two further
whole exit slabs. Neither finer RN-coordinate feasibility, actual origin
reachability, indefinite C6 boundedness nor future runtime is established.

### Smaller observations, complete histories, one shared next gate

Exact pairwise endpoint inclusion reduces the23-piece observation union to19
DBMs. With zero-based indices in the retained complete-observation order,
the containment witnesses are `2 -> 1`, `7 -> 5`, `21 -> 13`, `22 -> 14`.
No equality duplicates occur. All23 history records remain: containing the
endpoint of a different path does not authorize substituting that path's
preimage. In particular, indices21/22 are the exterior331/458 alternatives;
they cannot be discarded when their containing local-history alternatives
are subsequently refined.

Translate each surviving local intersection backward along its already
certified B64 prefix, preserving the full path, exact endpoint and source
cover. This produces23 pieces at22 exterior histories with no new
predecessor layer. Only source41 is shared, at frontier indices9 and13:

```text
40 -> 18 -> 48 -> 8       (feeder45 alternative)
40 -> 16 -> 26 -> 48 -> 8 (feeder97 alternative)
```

Their source-coordinate DBMs are disjoint. They must remain alternatives,
not be intersected. Both use the same12 original incoming arc identities,
so a bounded joint gate can cache12 source-cover/guard intersections and
perform24 distinct target-conditioned pullbacks. Including two parent-root
checks gives38 base intersections rather than50; child-root checks, union
differences, provenance and validation remain separate. The other21 paths
must remain in the final whole-observation comparison.

This gate has not run. Expanding the entire frontier would require308
branch/arc tests across296 distinct source/arc guards; that broader search
is deferred. The exact normalization, containment witnesses and complete
incoming lists are retained in `b67_final_validation.json`; rationale and
the full source table are in `artifacts/research/c6_b67_next_gate_review.md`.
This is a shared computation over already derived nodal histories, not a
new physical memory assumption or a symmetry quotient.

## 62. B68: a shared-source control and the boundary of local progress

The B67 source41 gate is evaluated completely: both disjoint conditioned
preimages, all twelve incoming arc identities, both parent-root cases and
all twenty-one other path records. The old observation-containment links
are not used to delete histories. A small adapter,
`artifacts/research/c6_shared_source_refinement.py`, caches twelve source
cover/guard intersections around the sealed B67 two-intersection protocol.
Every call position and argument is checked; all twenty-four branch-specific
pullbacks remain separate. The underlying layer, nodal word audit and exact
DBM operations retain their existing owners.

### One positive branch and one negative control

Of **24 candidates, 18 are empty and six survive**:

| B67 frontier index | Surviving incoming arcs | Source vertices | Complete nodal word |
|---|---|---|---|
| 9 | 209, 3895 | 19, 245 | `18 -> 40 -> 18 -> 48 -> 8` |
| 13 | 209, 381, 1095, 1337 | 19, 35, 64, 83 | `18 -> 40 -> 16 -> 26 -> 48 -> 8` |

All root intersections are empty. Each return is one nodal step. Arc209
is shared but its two preimages remain different alternatives, as required
by the two disjoint target-conditioned parents.

The exact parent differences have four and zero pieces. The first branch
strictly shrinks; the second branch's incoming union reconstructs its
entire old preimage. Reducing twelve candidates to four in that second
branch supplies **no geometric gain**. It is an explicit negative control
for candidate-count-based progress claims.

None of the five distinct surviving sources belongs to the B64 local
federation. The existing-cover reuse stage therefore retains all six
exterior pieces and yields no additional restriction; no new local cover
is generated. After restoring all twenty-one unselected paths, the old
23-piece endpoint union is replaced by 27 pieces. It strictly shrinks:
the exact difference has six components against old frontier index9 and
is empty against each of the other22 old endpoint pieces. The complete
observation hull is unchanged. The original target remains nonempty.

Charged work is1,976 DBM operations: two parent-root checks, twelve shared
source-guard intersections, twenty-four distinct target pullbacks, six
child-root checks,46 parent-difference operations,64 selected-family
difference operations,30 no-change reuse-difference operations and1,792
whole-observation difference operations. Sharing saves twelve source-guard
meets; it does not reduce the number of predecessor obligations.
Hash/source binding, arc enumeration, path/subset audits, independent
replay and cache controls are accounted separately.

The report is `artifacts/research/c6_b68_shared_source41.json`; the producer
and independent validator are `c6_b68_shared_source41.py` and
`validate_b68_shared_source41.py`. No scientific source, physical parameter,
B47 origin or runtime trajectory changes. The regional result remains
**41/56 excluded, 15 pending**. These local reductions do not supply a
convergence rate for the proof search or guarantee eventual global closure.
The clipped origin-history cover describes behavior before an original
cube exit; it is not by itself an invariant of the un-clipped nodal map.
Leaving the chosen cube would not, by itself, prove an unbounded trajectory.

### Coordinate-coset relevance: real geometric gain, no modular closure

A separate bounded control reuses the B53 exact residue adapter instead of
adding another history layer. Write carried coordinates as `X = a + q*k`,
where `a` is the exact B47 EPI-plus-remainder origin and `q = 2^-113`.
Rebuilding all64 canonical pressure rows and their exact nodal increments
gives the coordinate gcds

```text
g_i = gcd_m |h * DeltaNFR_i(m) / q| = (1,1,4,8,8,8).
```

The origin has `k=0`; every declared nodal shift preserves `k_i=0 mod g_i`
before exit from the original cube. This is a necessary arithmetic
condition, not an origin-reachability characterization. With common
modulus8, enumerate the128 allowed residues. For each DBM `B` and residue
`r`, the quotient constraints are

```text
z_i - z_j <= floor((B_ij - r_i + r_j)/8),  k = 8*z + r,  k_6 = 0.
```

An integer quotient potential supplies an explicit membership witness;
emptiness would require excluding all128 residues. Every one of the27
retained endpoint pieces has a witness in the first residue class, as does
every one of the six removed difference pieces. All33 points round to
the original mask8 EPI row, and each removed witness lies outside the
entire new27-piece union. The test uses33 quotient closures against a
declared worst-case cap of4,224. Canonical-pressure rebuilding and
independent witness verification are separate costs.

Consequently, the geometric improvement removes coordinate-coset-admissible
possibilities, while imposing these cosets alone excludes **none** of the
remaining pieces. The witnesses are hypothetical states, not trajectories
from B47. Records and independent checks are retained in
`c6_b68_coset_relevance.py/.json` and its matching `.validation.json` under
`artifacts/research`; `b68_final_validation.json` seals both B68 studies.

### Change the next gate, preserve the unresolved objective

Whole-label coverage has remained41/56 since B61. B66–B68 have demonstrated
strict geometric improvements, but no bound on the number of further
refinements needed for closure. B68 also exhibits a branch where fewer
candidates buy no geometric improvement, and its modular control cannot
close a survivor. Further predecessor depth is therefore deferred as the
automatic next action.

The next task audits whether existing exact separator obstructions still
apply to the retained history cover and the current origin-to-target
obligation. The selected bounded audit is result0 of
`c6_b56_affine_frontier_probe.json`:64 supported inequalities at backward
depth6 for the mask8 lower target, over11 RN modes and77 affine coefficients.
Completion requires classifying every witness and target implication, and
rechecking the exact weighted dual identity. It must bind the original root, complete guarded relation,
current target-conditioned pieces and exact certificate template. An old
dual witness cannot be transferred merely because the displayed RN modes
match: history labels and guard geometry may change its admissibility.
Failure to transfer an old obstruction does not prove that a new separator
exists. This applicability audit precedes any new floating proposal or
large search; the detailed gate is recorded in
`artifacts/research/c6_b68_progress_and_next_gate.md`.

## 63. B69: direct dual transfer fails, and a short entry-policy obstruction

B69 completes the selected B56 result0 audit without another LP or
predecessor expansion. B55 and B63 have identical B47 state, affine origin,
quantum, timestep, all64 RN rows and pressure values. The old64 supported
inequalities are rebuilt from their labeled points and returns, including
their sparse coefficient encoding, in the original77-dimensional affine
space with normalization `2^59`. All weights are strictly positive; the
weighted vector sum is exactly zero and the strict mass is positive. The
old certificate therefore remains valid in its original declared domain.

### Direct applicability to the current histories

Each fixed witness is tested against every matching retained history and
complete guarded arc. For an ingress, its source point is recovered by
subtracting the exact return shift; checking only its endpoint is unsound.
Target points are checked against the current27 endpoint pieces and all136
recorded physical positions on their existing paths. Intermediate atomic
guards are retained. Telescoping a return-level potential uses the active
membership of return endpoints; a transient middle alone does not require
an additional potential coordinate.

| Old constraint kind | Total | Directly admitted | Failing direct admission |
|---|---:|---:|---:|
| Internal drift | 51 | 14 | 37 |
| Strict ingress | 7 | 2 | 5 |
| Target nonpositivity | 6 | 0 | 6 |
| Total | 64 | 16 | 48 |

All37 failed drift witnesses and all five failed ingress witnesses lack an
admitted current source point. Ingress index40 has an admitted endpoint
but no admitted source; it is rejected. None of the six old target points
belongs to any current retained history, endpoint piece or recorded suffix
position. Their original depth-six target obligations cannot be identified
with the newer endpoint union merely by the common mask8 label.

All64 old fixed points also violate the necessary coordinate cosets; in
particular every point violates coordinate5's modulus. This does not
invalidate the old common-grid affine-domain result. Nor does it exclude
possible convex or algebraic derivations of related inequalities on a
narrower domain: those implications have not been audited. The precise
negative outcome is **failure to reuse the published dual by direct
admission of its fixed witnesses**, not existence of a positive separator
or absence of every possible obstruction. A history-indexed coefficient
space would likewise require a newly balanced dual.

The audit uses5,607 point-membership and existing-path subset checks,
separately records one916-vertex/7,165-arc index construction and4,928
rational weighted component products, and performs no optimizer call.
Evidence: `artifacts/research/c6_b69_obstruction_applicability.py/.json` and
the matching independent validation.

### Five inequalities expose the cost of blanket entry positivity

A distinct certificate uses an already retained, coordinate-coset-admissible
path. B68 endpoint9 is B67 normalized frontier10. Its fixed witness follows

```text
history vertices: 89 -> 125 -> 21 -> 97 -> 180
original arcs:    1461, 2011, 217, 1542
RN states:       32 -> 16 -> 26 -> 48 -> 8
```

All four returns are direct nodal steps. The exact B47-relative carried
points, RN rows, coordinate cosets, original/partition guards and retained
cover membership are checked. Four shared-kernel steps reproduce the
points with zero nodal-balance residual, after rebuilding all64 canonical
pressure rows. This is a hypothetical admitted path, not an execution from
the B47 initial point.

Use the old eleven-mode active set solely to define a **current-endpoint
certificate policy**: positivity on every geometrically admitted entrance
from outside, nondecrease on every internal return, and nonpositivity on
the complete current27-piece endpoint target. Mode32 is outside, while
16,26,48,8 are inside. Writing `b_j` for the potential at the four inside
states, the policy demands

```text
b_1 > 0,
b_2 - b_1 >= 0,
b_3 - b_2 >= 0,
b_4 - b_3 >= 0,
-b_4 >= 0.
```

Their sum is `0 > 0`, a contradiction. The retained affine certificate has
five77-component vectors, all weights one, exact zero sum and strict mass
one. The same telescope applies to arbitrary consistent real state-function
values under these identical entry, internal and endpoint conditions.
Increasing polynomial degree or adding history coefficients cannot repair
this policy while keeping all five requirements at the displayed states.

This is **not** a transfer of the old depth-six target dual. Its endpoint
target is explicitly different. It does not refute an origin-conditioned
global barrier, a different active domain or an entry restriction separately
proved from B47. The geometric ingress premise includes a hypothetical
state whose reachability is unknown; that is the overly strong requirement.
Evidence: `c6_b69_entry_obstruction.py/.json` and its independent validation
under `artifacts/research`.

### Root-conditioned continuation

Retire blanket positive-entry requirements for this active-region endpoint
template. The next formulation must carry the actual origin information
through the retained history graph. In B63,910 of916 history zones are
nonempty; root vertex260 is the singleton reached by the first direct
`57 -> 58` return. Its B47-relative coordinate is that nonzero return shift,
not the zero vector. A global certificate must admit this exact root and
the already checked initial sentinel/terminal bridge, constrain all relevant
guarded transitions, and separate the complete current target. Proof
coefficients would be verification functionals, not new nodal forces or
physical parameters. The measured formulation has6,370 coefficients,
7,150 fully clipped ordinary arcs and27 target pieces at vertex180:
7,178 root/drift/target verification obligations. A naive robust DBM
dualization would add301,434 nonnegative multipliers. This is a size count,
not a runtime estimate. The reproducible read-only inventory is
`c6_b69_next_gate_inventory.py/.json`. Specification, exact-verifier cost
and sparsity checks precede any candidate search.

Both studies and the documentation checkpoint are sealed by
`artifacts/research/b69_final_validation.json`; the design boundary is in
`c6_b69_scope_and_next_gate.md`. The result remains **41/56 excluded,
15 pending**. No actual origin trajectory is extended; no new cover,
positive global barrier, C6 instability or indefinite boundedness is proved.

## 64. B70: executable history-affine obligations and exact verification

The B69 proposal now has one complete executable specification. The 910
nonempty history vertices retain independent six-coordinate gradients and
offsets. Its sole root is vertex 260 at the exact nonzero first-return image
of B47; all 7,150 ordinary clipped arcs and all 27 endpoint pieces remain
separate obligations. The original sentinel, terminal and complete-target
bridge is retained as a previously verified premise. Storage and local
verification do not independently re-prove that bridge.

For `beta_v(k)=a_v dot (k/2^59)+b_v`, the drift on an arc `(v,w,H,s)` is

```
c = (a_w-a_v)/2^59
d = b_w-b_v + a_w dot s/2^59
beta_w(k+s)-beta_v(k) = c dot k+d.
```

The target check applies the same lower-bound kernel to `-beta_180` on each
target DBM. Root evaluation requires `beta_260(r)>=1`; a zero function is
insufficient. The normalization and all certificate coefficients are proof
representations, with no change to nodal pressure, capacity or evolution.

### Exact sparse certificate, complete history checks

For a finite closed integer DBM `k_i-k_j<=B_ij`, `k_6=0`, a supplied sparse
family `lambda_ij>=0` proves the affine lower bound if

```
sum lambda_ij (e_i-e_j) = -c       [coordinates 0..5]
c dot k+d >= d-sum lambda_ij B_ij >= 0.
```

The coefficient on the fixed-zero coordinate is immaterial. The local
checker uses exact rational arithmetic, rejects floating inputs and
duplicate facets, verifies closure/consistency, and checks the whole
coefficient identity. A sufficient dual need not be optimal. The full
checker binds the complete decoded specification by canonical JSON hash,
requires exactly one coefficient block and proof record for every declared
identity, and includes the translated shift term above. Its rational bit
limit is a verification resource limit, not a physical assumption or a
mathematical impossibility criterion.

Integer and rational algebraic fixtures exercise positive acceptance;
shift-sign, nonzero-root, missing-history, missing-target and stale-binding
controls exercise rejection. Two controls on the full C6 specification
evaluate all 7,178 obligations: the zero function fails only the root and
the constant-one function fails all 27 targets. Neither is a candidate
search, and no positive C6 certificate is supplied.

### Equivalent geometry and an existing exact extremum owner

Exact equality permits storage of 7,108 DBMs for 7,177 uses. The 69 repeated
geometries share computation, while their history coefficients and proof
obligations remain distinct. There are no duplicate complete history/shift/
geometry obligations to discard. Sequential deletion of an inequality only
when the remaining system implies it reduces the weighted facet count from
301,434 to 97,228. All final closures equal their original DBMs, with 11 to
22 retained inequalities per geometry. This procedure uses 312,752 closures
within a fixed 320,000 cap; the largest bound has 61 magnitude bits.
The chosen irredundant description is order-dependent and is not asserted
to have globally minimum cardinality.

The existing `_linear_extremum` in `c6_carried_excursion.py` already solves
an exact integer transport dual and checks an attaining primal point. A
rational adapter factors `c` into a positive rational scalar times primitive
integer weights, reuses that owner and rescales its dual. It independently
checks `c dot point+d` and the lower-bound certificate. A negative minimum
gives a violating point for that fixed affine candidate; it does not show
that the point is reachable from B47. For seven anchored coordinates, the
positive-to-negative transport graph has at most 12 possible edges. That
bound is for a fixed signed coefficient vector; an LP with unknown signs
cannot silently replace all robust constraints by one fixed such graph.

After B71's verification-boundary preparation below, the bounded step is
candidate discovery followed by complete exact verification, with the scope
and resource caps in
`artifacts/research/c6_b70_verifier_and_next_gate.md`. Geometry reduction and
local extremum calculations neither establish a global separator nor close
another first-exit label. The checkpoint is `b70_final_validation.json`:
**41/56 exclusions and 15 pending**, on the unchanged B63 scientific source.

## 65. B71: mechanism reuse and explicit verification boundaries

The [central B71 audit](C6_RESEARCH_MECHANISM_AUDIT.md) owns the 308-file
inventory, scoped verification-boundary defects and repair plan. None
demonstrates a false certificate in the recorded normal C6 validation.
The actual root has `sum(k)=-4`; rank-six controls exclude a nonzero shared-
gradient exact invariant on all arcs, not the history-affine barrier family.
The known 908-history corridor offers only modest optional savings; rational
sink completion need not preserve discovery norm or bit limits.

B71's repair plan requires explicit execution/input boundaries and one
validated affine-obligation transform, preserving independent validators.
B72 implements it below before the first bounded candidate attempt.
No LP ran, origin trajectory was extended or scientific source changed.
The checkpoint is `b71_final_validation.json`: **41/56 excluded, 15 pending**,
with indefinite C6 boundedness open.

## 66. B72: shared exact verification and the first bounded LP

The [central mechanism and verification report](C6_RESEARCH_MECHANISM_AUDIT.md)
owns B72's source repairs, shared history-affine owner and search outcome.
The owner unifies admission, translated forms, point rows, exact extrema
and final checks. Validation records 451 tests, 11 independent mutation
controls, 7,177 matching historical forms/rows and 62 matching local extrema.
Nodal pressure, capacity and phase evolution remain unchanged; historical
artifacts retain their original source bindings.

The single LP reaches its time limit with no primal candidate (20.051 solver
seconds, status 1), so no exact scan, oracle call or second LP follows.
B73 below completes the gate removing only norm-objective auxiliaries in a
separately versioned feasibility-only point LP: one 20-second attempt, 6,370 free
coefficients, zero objective and the same 7,178 point constraints. Finite
feasibility is equivalent, with no guarantee of faster solving. Full exact
acceptance and coefficient limits remain unchanged. The checkpoint is
b72_final_validation.json (`artifacts/research/b72_final_validation.json`, local evidence):
**41/56 excluded, 15 pending**, with indefinite C6 boundedness open.

## 67. B73: feasibility-only control and equivalent proof coordinates

The B73 result and next-gate note (`artifacts/research/c6_b73_result_and_next_gate.md`, local evidence)
owns the complete outcome. The zero-objective LP retains all 7,178 point
rows with 6,370 free coefficients and 100,288 nonzeros. It reaches the
20-second solver limit without a primal candidate; no exact scan, oracle
call or new counterexample cut follows. This is inconclusive. The unchanged
B72 source supports reuse of 451 tests and 62 extrema; neither was rerun.

Independent preflight checks all 7,177 drift/target rows plus the root,
seven malformed proposals and optimized-execution refusal. The 27 target
identities remain, despite only 12 distinct anchored rows. Full guarded
obligations cannot be removed on that basis. Two equivalent proof charts
are checked: first-evaluation centering reduces nonzeros to 59,682, while
the signed-tree chart gives 100,282. The selected chart worsens the largest
within-column magnitude ratio; sparsity does not prove better conditioning.
Its 2,454 zero point columns may be omitted only with a lift to all 6,370
slots; new cuts require a new active-column audit. B74 below completes
binary64 materialization/scaling and inverse original-coordinate preflight
before one separately versioned 20-second translated proposal. No further LP ran
in B73. The checkpoint is
b73_final_validation.json (`artifacts/research/b73_final_validation.json`, local evidence):
**41/56 excluded, 15 pending**, with indefinite C6 boundedness open.

## 68. B74: scaled proposal, full guarded rejection and retained witnesses

The B74 result (`artifacts/research/c6_b74_result_and_next_gate.md`, local evidence) retains
all 7,178 sampled conditions in a 3,916-column proposal. Exact positive
scaling of 22 rows prevents loading losses in the installed solver; all
59,682 entries, bounds and the objective are checked byte for byte.
Independent full-row preflight and nine reconstruction controls pass.
The single LP returns a candidate in 7.629 seconds, with no change in nodal
dynamics. Reconstruction returns to all 6,370 original slots before checking
root normalization, denominator/bit limits and every guarded condition.

The complete scan and independent primal/dual verification reject 5,949
conditions: 5,922 drifts and all 27 targets. Total experiment time is 18.501
seconds. The raw exact lift has 1,688 negative drift anchors; the limited
candidate has 1,546, despite floating solver success. Separately, 4,403 failed
guarded conditions have nonnegative limited anchors. The raw lift also fails
at 5,933 of the retained negative witnesses, so denominator limiting alone
does not explain the rejection. None of these points is certified reachable.

The next bounded refinement reuses at most 256 verified witness constraints,
starting with all 27 target identities, while preserving all initial rows.
New points require recomputing active columns and scaling before another
single, separately versioned 20-second proposal and complete exact scan.
No further LP or new cut was executed in B74. The current candidate does not
refute the full affine family. Source and coverage remain unchanged; the
checkpoint is b74_final_validation.json (`artifacts/research/b74_final_validation.json`, local evidence):
**41/56 excluded, 15 pending**, with indefinite C6 boundedness open.

## 69. B75: verified witness refinement and inconclusive bounded search

The B75 result (`artifacts/research/c6_b75_result_and_next_gate.md`, local evidence) implements
the preceding refinement: all 7,178 initial rows plus 256 verified B74
witnesses, including every target identity. New points reactivate 134 columns;
independent preflight checks the resulting 7,434-by-4,050 matrix and all
62,379 entries. Exact positive scaling applies to 81 rows. The full proof
problem still has 6,370 original coefficient slots and 7,177 guarded checks.

Its single LP reaches the 20-second limit after 20.023 solver seconds,
without coefficients. No exact scan or new extremum follows. Original-point
and witness evaluations are unavailable, rather than passed or failed.
The intermediate primal-status message is not a mathematical infeasibility
certificate. Independent post-run validation passes; nodal dynamics and
source remain unchanged.

A no-optimization audit confirms that selecting `highs-ipm` preserves every
outgoing B75 model array and changes only the solver selector. The next gate
is one separately versioned 20-second comparison on the frozen B75 matrix,
after independent preflight. No further cuts or history are added; any
candidate retains the same reconstruction, root/cap and exact full-guard
requirements. Algorithm choice guarantees no speed or feasibility gain.
The checkpoint is b75_final_validation.json (`artifacts/research/b75_final_validation.json`, local evidence):
**41/56 excluded, 15 pending**, with indefinite C6 boundedness open.
