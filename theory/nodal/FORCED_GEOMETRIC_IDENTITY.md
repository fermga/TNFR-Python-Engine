# Geometric identity and capacity-feedback bounds

Local restoring response, geometric identity and prospective capacity tubes.

Part of [Forced support balance and model boundaries](../FORCED_SUPPORT_BALANCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

## 41. Geometric identity and the undefined global phase target

Recovering every coordinate of a stationary target is not the definition of
every coherent entity. Section 32 already supplies a different conditional
identity: acute cycle phases approach a uniform twist while scalar EPI
approaches consensus. The limiting phase geometry remains nonuniform. This
section checks that identity against the native global coordinator, without
changing the earlier P3 target or assuming that its supplied sine law is the
default runtime law.

### Local coherent geometry can have zero global phase order

On a simple C5, fix the ideal phases on the circle as

```text
theta_i = theta_0 + i*delta modulo 2*pi,  delta = 2*pi/5.
```

The global unweighted phasor sum is zero. Each node's two neighbors instead
have the nonzero resultant

```text
exp(i*theta_(i-1)) + exp(i*theta_(i+1))
    = 2*cos(delta)*exp(i*theta_i),  cos(delta)>0.
```

Thus its local phase target is its own phase, canonical phase pressure and
curvature are zero, and absolute local phase gradient is `delta`. All edges
are strictly inside the half-pi U3 limit; winding is one. With constant EPI,
common positive capacity and no additional forcing, the other pressure
channels also vanish on this cycle. With fresh pressure and its nodal rate,
the ideal coherence read-out is one although global Kuramoto order is zero.
This distinguishes two existing
diagnostic meanings; it does not redefine either metric.

Positive unequal transport conductances, including section 32's retained
closing-edge weight, do not change these phase facts. The local and global
phase reducers use unique neighbors and unit global weights, respectively.
Under section 32's supplied phase law this is a relative equilibrium, with
common phase advance and unchanged relative geometry. Its conditional
relaxation bounds do not transfer automatically to the native coordinator.

### A symmetric phase multiset cannot select one covariant direction

Suppose a phase-only global target `T` takes values on the circle, is
invariant under permutations of its inputs and covariant under a common
phase rotation. Let P cyclically permute the five regular phases. Then

```text
P*theta = theta + delta*1 modulo 2*pi,
T(P*theta) = T(theta),
T(theta+delta*1) = T(theta)+delta modulo 2*pi.
```

The last two conditions contradict `delta!=0 modulo 2*pi`. No deterministic
single-valued target can satisfy both symmetries at this multiset. This is
the same stabilizer logic used for parent selection in
[the THOL audit](../THOL_BIRTH_AND_TRANSPORT.md#8-selection-actual-dispatch-and-the-unique-parent-obstruction),
applied to a circular output rather than a chosen vertex.

The argument concerns the phase multiset consumed by the global reducer.
It does not assume that a cycle with a distinguished conductance has a
cyclic automorphism preserving its complete weighted state. A marked node,
external direction or other additional input changes the target-selection
problem. None is supplied by the zero phasor resultant itself.

### Any selected global target changes the uniform-twist orbit

The actual [coordinator](../../src/tnfr/dynamics/coordination.py) applies a
shared global gain gamma and local gain to shortest-arc displacements. At
the ideal uniform twist the local displacement is zero. If a single global
target T is nevertheless supplied, its ideal phase update is

```text
theta_i^+ = theta_i + gamma*wrap(T-theta_i) modulo 2*pi.
```

For `0<gamma<1`, these increments cannot all agree modulo `2*pi`, so the
result is not merely a common rotation. Around the oriented cycle, four
gaps and the one gap crossing the target's shortest-arc cut become

```text
four gaps:       (1-gamma)*delta,
exceptional gap: delta+gamma*(2*pi-delta).
```

An antipodal tie can move which edge is exceptional; it does not restore
equal gaps. Their unwrapped sum remains `2*pi`. Every gap remains inside
the open shortest-arc branch exactly when `gamma<3/8`; winding then remains
one despite the changed geometry. Strict acuteness additionally requires
`gamma<1/16`. At equality the exceptional gap is `pi/2`: the closed hard U3
gate may admit it, but the strictly acute theorem no longer applies. At
`gamma=3/8` that gap reaches the antipodal branch boundary.

Consequently preservation of a winding integer is weaker than invariance
of the uniform-twist orbit. This obstruction needs no EPI integration or
long trajectory. It also does not say that the resulting phase state has
lost every possible coherent identity.

### Native policy and represented arithmetic have separate scope

The default adaptive policy classifies sufficiently low global Kuramoto
order as dissonant and retains a strictly positive global gain. That policy
therefore reacts to the ideal twist's zero global order despite its regular
local geometry. It is a configured preference for global alignment, not a
consequence that local TNFR coherence or winding is absent.

The ideal regular polygon and its materialized binary64 angles are distinct
inputs. Their transcendental sums need not agree exactly; cached trigonometry
and reduction add further numerical effects. The existing
[retained phase audit](../THOL_BIRTH_AND_TRANSPORT.md#global-phase-direction-near-the-symmetric-input)
already exposes that distinction and the conditioning of a small resultant.
The shared represented-component reducer sums its supplied component pairs
exactly; it does not certify the transcendental values or a rotation symmetry.
The coordinator's opt-in `exact_components_v1` rejects an exactly zero
represented resultant when the effective global term is active. A nonzero
represented sum remains a numerical direction, even if very small. The
legacy path instead retains its existing numerical `atan2` behavior.

The [static writer controls](../../tests/physics/test_native_phase_writer_boundaries.py)
use unit C5, represented `theta_i=0.125+2*pi*i/5`, EPI `0.5`, capacity one,
stored pressure zero, empty glyph histories, injected defaults and seed 17.
One legacy invocation and a matched `exact_components_v1` invocation both
select the dissonant branch with `kG=0.05929429797231117`. Four observed gaps
are approximately `1.1821256490720864` and the fifth `1.5546827108912409`,
matching the derived deformation. Winding remains one and the strict U3
margin is approximately `0.0161136159036559`. The exact sum of cached
component pairs is `(-7,3)/2^55`, nonzero; this does not contradict the ideal
polygon's zero sum. Both calls preserve EPI, capacity, stored pressure,
support and clock. They execute no glyph, nodal integration or full runtime
step and establish neither future invariance nor formation of this pattern.

These results identify a model-compatibility boundary between an existing
geometric identity and an existing native policy. They select no replacement
target, zero-resultant threshold, phase law or autonomous event schedule.

## 42. A geometric-identity contract for the retained weighted C5

### Declared state, law and identity family

Keep Section 32's scalar form chart, supplied clock and continuous phase/form
law. The modeled state is `(x,theta,nu,W)`, not every engine history, cache or
controller variable. Retain the actual Section 31 post-UM support and its
materialized nonunit closing conductance. The ordered phase support is C5;
all transport conductances are fixed, positive and symmetric. Capacity is
held at its captured common positive value `kappa`, and effective pressure
weights and `K>0` are fixed. No Gamma, subsequent event, active clipping or
native controller is included. The phase law identifies kappa with radians
per supplied time unit; neither laboratory seconds nor a clock emergent from
synchronization is inferred.

The identity family is specified before evaluating a response:

```text
I(W,kappa) = { x=c*1, nu=kappa*1,
              theta_i=alpha+2*pi*i/5 modulo 2*pi, W fixed }.
```

Its free labels are common rotation alpha and uniform form c in the declared
chart. It excludes phase consensus and fixes winding `+1` in the chosen
orientation. Reversing the cycle enumeration changes the displayed winding
sign, not the physical state; conjugating the physical phases is a different
operation and is not quotiented out. The family preserves nontrivial phase
geometry even though form is uniform. It does not require nonzero scalar-form
amplitude; that would be a further identity obligation. It is not a definition
of every NFR or an identification with a physical particle.

On this family the local phasor source and EPI pressure vanish, the sine
corrections cancel, and `theta_dot=kappa*1`. Thus it is exactly invariant
under the declared continuous law: c is constant and alpha rotates. The
global Kuramoto order is zero while each normalized local resultant is
`cos(2*pi/5)>0`. The global-target obstruction in Section 41 therefore remains
relevant; invariance here is not invariance under that native coordinator.

### A lifted shape coordinate separates rotation from deformation

Admit initial oriented gaps inside one strictly acute interval `[m,M]`,
contained in the fixed U3 gate, with sum `2*pi`. Section 32 preserves this
domain. Write `delta_bar=2*pi/5`, `u_i=delta_i-delta_bar`, and define

```text
p_0=0,  p_i=sum_(j<i) u_j,
h_i=p_i-mean(p),
alpha=theta_lift_0+mean(p).
```

Then `sum h_i=0`, the cyclic difference of h equals u, and
`theta_lift_i=alpha+i*delta_bar+h_i`. The last cyclic difference is valid
because `sum u_i=0`; the phase lift itself closes after one full turn.
Under the supplied equal-capacity sine law the corrections telescope in
`sum theta_dot_i=5*kappa`. Consequently

```text
alpha(t)=alpha(0)+kappa*t,
sum_i h_i(t)^2 <= ||u(t)||_2^2 / lambda_2(L_C5)
              <= Q^2*exp(-2*gamma*t) / lambda_2(L_C5).
```

The second line is the cycle Poincare inequality on centered h. This is a
lifted shape norm. The circular orbit distance is
`inf_alpha sum_i wrap(theta_i-alpha-i*delta_bar)^2`. It is no greater than
the displayed lifted bound, because wrapping each residual and then
minimizing over a common rotation cannot increase this chosen value. No
claim is made that the lifted norm is the global circular minimum.

The offset alpha uses the ordered support and its admitted lift. It is a
coordinate, not a permutation-invariant target obtained from the phase
multiset, and it supplies no exception to Section 41's symmetry obstruction.

The existing `CycleRelaxationEnvelope` exposes this decomposition as exact
affine-pi pairs in `initial_phase_offset_affine` and
`initial_phase_shape_affine`, with shared rational enclosures in
`initial_phase_shape_enclosures`. Its
`phase_orbit_distance_squared_upper` reuses the existing sample times, gap
envelopes and certified spectral lower bound. These are derived read-outs of
one captured state/model, not new dynamic parameters, a second solver or
another evidence record. The properties also apply to the evaluator's other
admitted cycle sizes and winding sectors; the identity selected here is C5,
winding one.

### Form mean and the strength of the persistence claim

Use the actual transport strengths s, their sum S and
`m_s=s^T*x/S`. The form residual remains
`D=sum_i s_i*(x_i-m_s)^2`; phase and form residuals are reported separately
without inventing a cross-channel normalization. Section 32 supplies D's
Duhamel bound and

```text
m_s_dot = kappa*w*s^T*g/S,
|m_s(t)-m_s(0)| <= (M_source/gamma)*(1-exp(-gamma*t)),
|m_infinity-m_s(t)| <= (M_source/gamma)*exp(-gamma*t).
```

Here `M_source` is that section's mean-rate prefactor (its symbol M there),
not the upper phase-gap endpoint used above. Thus the limiting uniform form
is unknown but constrained by the captured initial mean and
`mean_limit_offset_upper`. Permitting this governed mean drift is not
permission to fit an arbitrary c(t). On the invariant family itself g=0,
so there is no mean drift. With the retained nonunit edge, a perturbed state
can have nonzero weighted drift even though `sum g_i=0`.

These estimates establish conditional attraction to the identity family.
They also give local orbital stability relative to that family: keep W,
kappa, coefficients and a common strictly acute neighborhood fixed. As
initial gap and form residuals tend to zero, their all-time upper bounds
and the mean-shift bound tend to zero. The neighborhood keeps the rates
bounded away from zero. Form disagreement need not decrease monotonically;
the retained post-event state initially develops form contrast before it
relaxes. Capacity, support, operator-policy and phase-law perturbations are
outside this stability claim.

### Evidence and remaining premises

The [existing control owner](../../tests/physics/test_cycle_postevent_relaxation.py)
reuses its retained initial capture to reconstruct the phase lift independently
at high precision, check centering, affine-pi enclosures, reconstruction and
the Poincare bound. Its winding is already one and its initial EPI disagreement
is zero, but its phase deformation is strictly positive. Thus a winding label
and uniform form do not alone identify a regular geometric shape. Reversal
with relabeling retains the same nodewise shape; a consensus control has zero
shape in the separate winding-zero family. No new trajectory is generated for
these assertions. The earlier finite binary64 continuation remains evidence
only for its stated execution horizon and numerical defects.

The derived parts are phase-domain preservation, this rotation/shape split,
conditional invariance, attraction and mean bounds. Fixed support/capacity,
the supplied sine phase law, its clock interpretation, pressure realization,
coefficients and prepared sector remain explicit model premises. The native
global coordinator is incompatible with the regular orbit; support creation
still depends on the recorded UM policy. Autonomous maintenance, formation
from a different sector and empirical identification remain unresolved. The
[single G3 queue](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
the next mechanism-admission task.

## 43. A positive local response supplies the cycle restoring mechanism

### Admitted class and the role of each assumption

Retain Section 32's fixed ordered C_n, common free phase rate `kappa`, actual
positive symmetric transport conductances, fixed capacity and scalar form law.
The phase support has two unique neighbors per node. In an admitted acute
sector let

```text
J_i = (sin(delta_i)-sin(delta_(i-1)))/2,
theta_dot_i = kappa + a_i(t,state)*J_i,
0 < a_minus <= a_i(t,state) <= a_plus < infinity.
```

J is the existing local sine current. The coefficient a is a response mobility
in this separately specified phase row, not nodal form capacity or a new TNFR
primitive. Bounds must hold throughout the declared domain and time interval;
reading one positive coefficient at the initial state does not establish them.
Each correction multiplies its local current; locality of the entire law also
requires a_i to read only the declared local state.
No derivative of a is needed for the inequalities below.

The estimates concern absolutely continuous solutions of that declared row.
A sufficient existence/uniqueness contract is measurable time dependence and
local Lipschitz dependence on the declared state, with locally integrable
Lipschitz bounds and the displayed uniform coefficient bounds. If a also
depends on form, this contract applies to the joint state; the linear form
row with bounded phase source prevents finite-time escape. Positivity alone
does not make an arbitrary discontinuous constitutive law well posed. Common
rotation or relabeling covariance of the whole law additionally requires the
corresponding covariance of a; the inequalities do not assert that symmetry
for every possible coefficient function.

### Acute-domain preservation and an exact dissipation identity

With cyclic indices, subtracting neighboring phase rates gives

```text
delta_dot_i = [a_(i+1)*(sin(delta_(i+1))-sin(delta_i))
               -a_i*(sin(delta_i)-sin(delta_(i-1)))]/2.
```

On a fixed acute interval `[m,M]`, sine is increasing. At a maximal gap
both contributions are nonpositive, and at a minimal gap both are
nonnegative. Therefore the interval is forward invariant. Its gap sum
remains `2*pi*ell`, retaining winding and the same U3 admission. Local phasor
resultants remain bounded below by `cos(rho)>0`, where
`rho=max(abs(m),abs(M))<pi/2`. Unequal free phase rates would add their
neighbor difference to this equation and require a separate forcing analysis.

Put `delta_bar=2*pi*ell/n`, `u=delta-delta_bar*1`, `Q=||u(0)||_2` and
`lambda_C=lambda_2(L_cycle)`. Summation by parts gives the exact identity

```text
(1/2)*d||u||_2^2/dt
 = -(1/2)*sum_i a_i*(u_i-u_(i-1))
                       *(sin(delta_i)-sin(delta_(i-1)))
 <= -(a_minus*cos(rho)/2)*u^T*L_cycle*u
 <= -gamma*||u||_2^2,

gamma = a_minus*cos(rho)*lambda_C/2 > 0.
```

Thus `||u(t)||_2<=Q*exp(-gamma*t)`. This is the restoring mechanism: the
local response has the dissipative sign and sufficient persistent strength
on an available phase chart. Neither a winding integer nor matching fixed
points supplies those properties. The same class decreases the unweighted
alignment potential `V=sum_edges(1-cos(delta))`, since
`grad_theta V=-2*J` and `V_dot=-2*sum_i a_i*J_i^2`. This potential identity
does not imply monotonicity of the forced EPI disagreement or tetrad energy.

The existing phase-to-form bridge is unchanged: on this cycle
`g_i=(delta_i-delta_(i-1))/(2*pi)`, so
`||g(t)||_2<=(Q/pi)*exp(-gamma*t)`. Section 32's weighted EPI, Duhamel,
mean-displacement, mean-tail and sufficient clipping bounds apply with this
gamma and their original actual transport coefficients. Section 42's
lifted shape bound also applies. The regular-twist/uniform-form family is
therefore conditionally attracting and locally orbitally stable with the
same fixed-state and common-neighborhood qualifications. No new form source
or capacity law is needed for this conditional result.

### A restored shape need not keep the same phase offset

The centered lift still obeys

```text
alpha_dot = kappa + (1/n)*sum_i a_i*J_i.
```

Since `sum J_i=0`, subtracting `(a_minus+a_plus)/2` inside the sum and using
`||J||_2<=||u||_2` gives

```text
|alpha_dot-kappa| <= B_alpha*exp(-gamma*t),
B_alpha = (a_plus-a_minus)*Q/(2*sqrt(n)).
```

Consequently `alpha-kappa*t` has a finite limit. Its displacement from the
initial offset is at most `B_alpha*(1-exp(-gamma*t))/gamma`, and its remaining
tail is at most `B_alpha*exp(-gamma*t)/gamma`. This is distinct from the
transport-weighted form-mean bound. A node-independent response makes the
phase correction sum zero, including a common time-dependent response.
The pressure specialization below also cancels its mean correction, even
though its current multiplier varies from node to node.

### Existing current and argument pressure belong to the class

For two adjacent oriented gaps a and b in the acute interval, define the
positive sine divided difference by its continuous integral expression

```text
s(a,b) = integral_0^1 cos(b+v*(a-b)) dv,
cos(rho) <= s(a,b) <= 1.
```

It equals `(sin(a)-sin(b))/(a-b)` when the gaps differ and `cos(a)` when
they coincide. Therefore the ideal cycle pressure and current satisfy

```text
g_i = b_i*J_i,
b_i = 1/(pi*s(delta_i,delta_(i-1))),
1/pi <= b_i <= 1/(pi*cos(rho)).
```

At equal gaps both readings vanish, but the continuous multiplier is
`1/(pi*cos(delta_i))`. No division of measured zero or near-zero current
is licensed. This relation also connects to the existing curvature read-out
`K_phi=-pi*g` on the same regular chart. It uses the common alignment
potential with distinct response metrics; it does not make pressure and
current identical or replace one by the other in the nodal form row.

Compare the following two **separately supplied** phase laws using the
already declared positive coupling K:

| Candidate phase correction | Current multiplier | Gap decay rate justified here |
| --- | --- | --- |
| `K*J` | constant K | `K*cos(rho)*lambda_C/2` |
| `K*g`, through the response-class bound | `K*b_i` | `K*cos(rho)*lambda_C/(2*pi)` |
| `K*g`, using its exact cycle reduction | same `K*b_i` | `K*lambda_C/(2*pi)` |

The last line is sharper because its exact gap equation is
`delta_dot=-K*L_cycle*delta/(2*pi)`. Its Jacobian does not contain the
cosine stiffness of the sine law. Admission still requires the regular acute
sector. Both candidates have `alpha_dot=kappa` because their respective
corrections telescope. Their clock identification and positive gain remain
supplied; the nodal EPI equation does not choose either phase law. In
particular, reading g as pressure is implemented, while treating it as a
phase velocity would be an additional constitutive choice.

### Executable comparison, controls and failure boundaries

`compare_cycle_restoring_responses` in the existing
[cycle owner](../../src/tnfr/physics/cycle_relaxation.py) consumes its retained
envelope and returns a detached `CycleRestoringResponseComparison`. With
the shared exact pi interval `[pi_lower,pi_upper]`, certified cosine lower
bound c and spectral lower bound lambda, it encloses `b_i` by
`[1/pi_upper,1/(pi_lower*c)]` and materializes the three conservative rates
above. No graph is recaptured, no sample bounds are relabeled as another
law, and no response, solver or controller is installed. Scalar/rate
consistency checks do not authenticate caller-created reports or live state.

The [existing cycle controls](../../tests/physics/test_cycle_postevent_relaxation.py)
reuse the retained post-UM endpoint and actual pressure/current owners.
Independent high-precision divided differences check the exact comparison,
including the regular-twist zero-response limit. Represented current and
pressure errors have separate absolute tolerances; their ratio is never
treated as an exact-real certificate. Static response budgets verify the
dissipation identity and maximum/minimum signs. A positive local comparison
`a_i=K*(1+J_i/2)` has mean correction `K*sum J_i^2/(2*n)>0` off the lock;
this delimits the generic phase-offset claim without proposing that law.

Three negative controls delimit the sufficient mechanism:

- Zero response preserves every nonregular gap configuration while the
  common phase advances. One actual zero-coupling production proposal retains
  those gaps within its stated binary64 tolerance; it supplies no restoring
  action. It is outside the positive-lower-bound class.
- A common negative response reverses the gap-energy sign at every
  nonconstant acute state, despite the same regular fixed shapes and phase
  symmetries. The shared phase proposal rejects negative coupling; the
  reversed instantaneous budget is a mathematical control, not execution
  of an admitted engine law.
- Pointwise positivity alone is insufficient. A common `a(t)=exp(-t)`
  traverses only finite effective time in the unit sine flow and need not
  reach its regular limit. A uniform positive lower bound is sufficient,
  not necessary: a nonnegative time-dependent lower envelope gives decay in
  accumulated action `integral_0^t a_minus(v) dv`. An infinite integral
  guarantees shape restoration under the same regularity/domain assumptions.
  An integrable decay bound is separately sufficient to extend the absolute
  source, form-mean and general phase-offset tail estimates; shape convergence
  alone does not provide those estimates. Exact mean cancellations can give
  stronger conclusions than the absolute bounds.

This completes the response-class admission at fixed support and common
capacity. It identifies a shared sufficient mechanism and its failure modes;
it does not derive the positive response, its clock or autonomous occurrence
from the nodal product. The native global coordinator is outside this local
class. Support selection, capacity evolution, formation and physical identity
retain their explicit unresolved obligations in the single G3 plan.

## 44. Reusing phase geometry and capacity energy without adding a law

The preceding results connect existing state, field and transport owners.
They yield the following deductions under their stated domains; they do not
select the supplied phase law or establish autonomous pattern formation.
The [single G3 plan](../research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
retains the capacity-forcing task. No additional trajectory campaign is needed
to establish the identities in this section.

### Curvature reconstructs the centered phase shape on the admitted cycle

Use Section 42's centered lift h, winding ell and oriented gap deviation u.
Let L_C be the **unweighted, unnormalized support** cycle Laplacian. The
two-neighbor midpoint identity gives, in exact arithmetic,

```text
u_i = h_(i+1)-h_i,
g = -L_C*h/(2*pi),
K_phi = -pi*g = L_C*h/2,
h = 2*L_C^+*K_phi,
||u||_2^2 = 4*K_phi^T*L_C^+*K_phi.
```

Here L_C^+ is its inverse on centered vectors. For two admitted shapes on
the same support and winding sector,
`||h_1-h_2||_2 <= 2*||K_phi_1-K_phi_2||_2/lambda_2(L_C)`.
Thus the existing spectral gap measures both restoring stiffness and the
sensitivity of this reconstruction. This is the centered version of the
phase coordinate already owned by
[cycle-support dynamics](../CYCLE_SUPPORT_DYNAMICS.md#1-all-three-active-pressure-channels-share-a-cycle-coordinate),
not another nodal primitive.

This identity needs the full node-resolved curvature, support, winding and
acute chart. It does not reconstruct common phase, capacity, EPI or transport
weights. Exact regular C5 twists of winding +1 and -1 both have zero curvature
and current. At uniform EPI and common capacity their ideal complete tetrads
also coincide: zero pressure/potential, equal gradient magnitudes and the same
correlation input. Winding therefore cannot be discarded. Nor can a lossy
tetrad summary replace the full curvature vector. The common-semicircle
premise of `phase_quotient.py` excludes a winding-one cycle; the block-constant
phase premise of `joint_quotient.py` excludes its nontrivial regular grouping.
Neither quotient automatically supplies the missing winding coordinate.

The existing cycle envelope exposes `initial_phase_curvature_affine` in cycle
order, with mathematical pi. Controls invert `L_C+11^T/n` using the shared
rational matrix inverse and reconstruct the retained shape independently.
Production phasor/curvature rounding remains separate; exact centeredness or
exact reconstruction is not asserted for uncorrected binary64 field readings.

### An augmented conserved mean distinguishes the pressure-response candidate

Retain common fixed capacity kappa, positive EPI diffusion coefficient e,
phase weight w, positive K and actual fixed symmetric conductances W. For the
**separately supplied** candidate `theta_dot=kappa*1+K*g`, the centered shape
satisfies `h_dot=K*g`. If s is the actual transport-strength vector,
`S=sum s_i` and `m_s=s^T*x/S`, the form row gives

```text
m_s_dot = kappa*w*s^T*g/S,
d/dt [m_s - kappa*w*s^T*h/(K*S)] = 0,
m_infinity = m_s(0) - kappa*w*s^T*h(0)/(K*S).
```

The last equality follows from exponential shape/source decay and stable EPI
diffusion on the connected cycle. It predicts the uniform limiting EPI from
the initial state, without fitting a late endpoint. For unit conductances,
`s^T*h=0` and the ordinary mean is conserved. With only the closing edge
changed to beta, `s^T*h=(beta-1)*(h_left+h_right)`; the mean shift is determined
by the deformation at that edge's endpoints. This uses `s^T*L_W=0`, and
does **not** replace the actual transport L_W by L_C/2.

`compare_cycle_restoring_responses` reports this affine-pi prediction and its
outward enclosure alongside the existing candidate rates. It assumes the
unclipped scalar model, with no added source, events or controllers. The
original sine-law samples and clipping bounds are not relabeled as a
pressure-law trajectory. For the sine law the same expression has derivative
`kappa*w*s^T*(g-J)/S`, generally nonzero; shared fixed points do not establish
a shared invariant. These are conditional model predictions, not evidence
that the engine or physical observations select the pressure-rate candidate.

### One capacity roughness controls two forcing channels

On the same cycle let f be the nonnegative configured capacity-pressure
weight and define the existing Dirichlet quantity on the capacity field,

```text
E_nu = nu^T*L_C*nu/2 = (1/2)*sum_i (nu_(i+1)-nu_i)^2,
b_i = nu_(i+1)-nu_i,
p_nu = -(f/2)*L_C*nu.
```

Then `||b||_2=sqrt(2*E_nu)` and
`||p_nu||_2 <= f*sqrt(2*E_nu)`, since `lambda_max(L_C)<=4`.
The same measured capacity roughness bounds detuning in the phase-gap row
and capacity forcing in the form row. The phase/capacity source also combines
exactly as `w*g+p_nu = -(1/2)*L_C*((w/pi)*h+f*nu)` while the acute chart holds.
This reuses the earlier structural-coordinate reduction and preserves signed
channel cancellation. It neither adds a fundamental parameter nor identifies
L_C with the weighted EPI transport operator. Topology pressure remains zero
on this fixed degree-two support.

For ideal snapshot averaging with eligibility mask M, write
`nu_next=nu-(mu/2)*M*L_C*nu`, where M is diagonal with entries zero or one.
With `z=M*L_C*nu` and `0<=mu<=1`,

```text
E_nu_next-E_nu = -(mu/2)*||z||_2^2 + (mu^2/8)*z^T*L_C*z
               <= -mu*(1-mu)*||z||_2^2/2 <= 0.
```

This extends the existing
[compatible-capacity Dirichlet balance](../../src/tnfr/physics/coupling_support.py)
to partial eligibility on the cycle. It is stronger than interval invariance,
but does not prove strict decay, preservation of a capacity mean, consensus
or an integrable forcing budget for arbitrary masks. An actual represented
update has an additional arithmetic defect; its energy change is not silently
identified with the ideal expression.

For the separately supplied extension `theta_dot_i=nu_i+a_i*J_i`, choose a
prospective fixed radius `rho_star<min(pi/2,effective_U3_gate)` with
`abs(delta_bar)<rho_star`. The coefficients satisfy Section 43's positive
bounds. While all gaps stay inside that chart, set
`gamma=a_minus*cos(rho_star)*lambda_2(L_C)/2`. The differential inequality
holds almost everywhere for the admitted absolutely continuous solution,
with the usual norm comparison at zero; its integrated form is

```text
d||u||_2/dt <= -gamma*||u||_2 + sqrt(2*E_nu),
||u(t)||_2 <= exp(-gamma*t)*||u(0)||_2
             + integral_0^t exp(-gamma*(t-r))*sqrt(2*E_nu(r)) dr.
```

A bound strictly below `rho_star-abs(delta_bar)` for the displayed norm
envelope closes the chart bootstrap. This is a conservative sufficient
condition. The radius from the unforced initial extrema cannot be reused
without this check: detuning can push gaps outward. A persistent forcing
bound gives a residual deformation
tube, not exact restoration of the regular twist. This distinction matters
for the actual averaging writer below.

For form, reuse the common energy `E_x=x^T*B_W*x/2`, where `B_W=D_s-W`,
and mobility `M_x=diag(nu_i/s_i)`. If F is the combined non-EPI source,

```text
E_x_dot = -e*(B_W*x)^T*M_x*(B_W*x) + (B_W*x)^T*diag(nu)*F
         <= -e*m_M*lambda_2(B_W)*E_x
            + sum_i s_i*nu_i*F_i^2/(2*e),
```

provided `nu_i/s_i>=m_M>0`. The inequality is Young's inequality followed by
`||B_W*x||^2>=2*lambda_2(B_W)*E_x`. It reuses the
[time-varying diffusion owner](../../src/tnfr/physics/structural_diffusion.py)
and [signed forcing ledger](../../src/tnfr/physics/forcing_realization.py).
Capacity-only events leave E_x unchanged, but other form/support events need
their actual jump budget. No moving metric or old fixed-capacity conserved
mean is assumed. Pressure realization and stale-input defects remain separate.

### The runtime and clock boundaries are observable

The [capacity synergy controls](../../tests/physics/test_cycle_capacity_synergies.py)
include an actual default-writer boundary: initialize unit C5 with EPI=0.5,
`theta_i=0.125+2*pi*i/5`, capacity `nextafter(1,+infinity)` at node zero and
capacity one elsewhere. From zero counters, five consecutive fresh-pressure,
fresh-Si, adaptation calls reach the default eligibility duration at every
node while leaving the capacities bitwise unchanged. The Si minimum exceeds
0.94 and the maximum pressure magnitude is below 5e-17 in this retained
binary64 control. It exercises the actual writer, not invented eligibility.
It is a finite writer control with phase/form held between calls, not a
complete-runtime stability result. The configured positive mixing factor
therefore cannot be used as a uniform realized contraction rate. The
separate P2 UM lattice certificate concerns a different arithmetic map.

Physical event timing also remains essential. Even ideal full averaging can
have a positive capacity profile `kappa*1+epsilon*r^k*v`, `0<r<1`, for a
nonuniform eigenvector v and sufficiently small epsilon. With held
interval durations h_k, its accumulated detuning is proportional to
`sum_k h_k*r^k`. Bounded durations make this sum finite; `h_k=r^(-k)` makes it
diverge despite convergence by call index. This reuses Section 23's clock
distinction and the response-action condition of Section 43. A combined bound
must carry eligibility, physical timing and represented defects together,
and distinguish exact-model recovery from a numerically resolved tube.

## 45. A capacity interval gives a prospective winding-retention tube

### Scope and the bootstrap from the actual event endpoint

Retain the fixed weighted C5 and supplied sine phase law, but now let every
capacity vary within one fixed positive interval `[a,b]`. The chosen execution
subclass uses only existing guarded snapshot adaptation events; initialize
inside the interval. Its represented interval-preservation property applies
to every eligible subset, including updates rounded to zero. Interpreting
these held represented values as exact coefficients gives the conditional
continuous model below. The mathematical bound also covers arbitrary bounded
measurable capacity schedules inside this same interval:

```text
theta_dot = nu(t) + K*J(theta),
x_dot = diag(nu(t))*(-e*L_W*x + w*g(theta) - (f/2)*L_C*nu(t)),
0 < a <= nu_i(t) <= b,  K>0, e>0, w>=0, f>=0.
```

The support, actual positive reciprocal transport weights and pressure
coefficients remain fixed. L_C is the unweighted support Laplacian; L_W is
the actual weighted random-walk Laplacian. Topology pressure vanishes on
this degree-two support. No Gamma source, clipping, form/phase jump or
additional operator is admitted. A binary64 phase/EPI trajectory is separate
evidence, even when its capacity values satisfy this interval premise.

The original endpoint has `||u(0)||_2` approximately 0.37161, exceeding
`pi/2-2*pi/5`, approximately 0.31416. Thus the sufficient ball test of
Section 44 cannot admit it even at zero capacity spread. This is a limitation
of that norm bound, not evidence that the initial acute geometry is invalid.
Its actual largest absolute gap is approximately 1.40835, below pi/2.

The centered-shape and alignment-potential owners supply a sharper bootstrap.
Compare h with the constant-capacity sine reference h_0 having exactly the
same initial phase. The reference needs no new simulation: Section 32 already
proves that its gaps stay within their initial extrema. Let `rho_0` bound
those extrema and choose a prospective `rho_star` below both pi/2 and the
effective U3 gate. On this convex fixed-winding acute lift chart,

```text
V(h) = sum_i [1-cos(delta_bar+h_(i+1)-h_i)],
Hess V >= cos(rho_star)*L_C,
z = h-h_0,
z_dot = Pi*nu - (K/2)*(grad V(h)-grad V(h_0)),
Pi = I-11^T/n.
```

The interval variance bound gives `||Pi*nu||_2<=sqrt(n)*(b-a)/2`.
Strong monotonicity on the centered subspace therefore yields, almost
everywhere and then by the usual norm comparison,

```text
gamma = K*cos(rho_star)*lambda_2(L_C)/2,
||z(t)||_2 <= sqrt(n)*(b-a)*(1-exp(-gamma*t))/(2*gamma),
|(delta_i-delta_0_i)(t)| <= sqrt(n/2)*(b-a)/gamma.
```

The notation delta_0 here denotes the evolving reference gap, not a fixed
initial gap. Consequently the strict prospective condition

```text
rho_0 + sqrt(n/2)*(b-a)/gamma < rho_star
```

closes the chart bootstrap from the original event endpoint. Winding and
the fixed U3 gate remain admitted for all times in this exact continuous
model. The comparison uses the selected constant-K sine law; it does not
extend automatically to every state-dependent mobility in Section 43.
The radius is a proof domain, not a dynamical parameter or a controller.

### Phase deformation, source and form-contrast bounds

Write `R=b-a`, `B=sqrt(n)*R` and `Q=||u(0)||_2`. Once the reference comparison
admits the chart, Section 44 gives

```text
q(t) = exp(-gamma*t)*Q + (1-exp(-gamma*t))*B/gamma,
||u(t)||_2 <= q(t),
||g(t)||_2 <= q(t)/pi,
||w*g(t)-(f/2)*L_C*nu(t)||_2 <= w*q(t)/pi + f*B.
```

The nonzero bound `B/gamma` is a residual tube, not an assertion that an
actual trajectory attains it. It does not establish exact return to the
regular twist. The accumulated detuning bound over a finite physical horizon
T is `B*T`; for R>0 this estimate supplies no finite infinite-time integral.
No mixing rate is inferred from call counters or the configured factor mu.

For form let `E=x^T*B_W*x/2`, `s_max=max s_i`, and use the existing exact
positive lower bound lambda_W for `lambda_2(B_W)`. With
`Q_star=max(Q,B/gamma)`, the common Dirichlet estimate gives

```text
F_star = w*Q_star/pi + f*B,
r = e*(a/s_max)*lambda_W,
d = s_max*b*F_star^2/(2*e),
E(t) <= exp(-r*t)*E(0) + (1-exp(-r*t))*d/r.
```

Thus EPI contrast is bounded; for example squared distance to the ordinary
constant subspace is at most `2*E(t)/lambda_W`. This does not control the
uniform form mode or preserve the former fixed-capacity weighted mean.
The constant source bound is conservative and need not vanish even when
the actual source decays. Inactivity of finite EPI clipping for all times,
limiting mean, exact recovery and complete-runtime maintenance remain open.

The existing [cycle owner](../../src/tnfr/physics/cycle_relaxation.py) exposes
`bound_cycle_capacity_forcing`. It reuses the retained envelope as initial
phase/EPI/support/weight data and explicitly replaces its held-capacity law
by the declared interval premise. The report uses shared exact pi, square
root, exponential and spectral owners. A failed margin returns nonadmission,
not a negative stability conclusion. No graph write, extra trajectory or
numerical solver is performed. The [arithmetic controls](../../tests/physics/test_cycle_capacity_forcing.py)
check both increasing and decreasing bound branches against independent
high-precision expressions, zero spread, failed admission and invalid inputs.

### One frozen execution with the actual default gate

The [finite writer study](../../tests/physics/test_cycle_capacity_synergies.py)
retains the actual post-UM weighted cycle. Before perturbation, it declares
the full default pressure mixture, replacing the earlier phase/EPI-only
experimental recipe so the capacity channel remains active. This change
preserves the support and phase/EPI endpoint; the new mixture has its own
fresh capture and prospective budget. The preparation adds `2^-12` to node
zero's capacity; all other capacities remain one. This is a declared initial
perturbation, not spontaneous capacity creation.

The frozen execution uses `K=1/2`, `dt=1/8`, 32 intervals and horizon T=4.
At every declared endpoint, after the shared phase proposal and nodal
integration, the caller refreshes pressure, computes fresh Si, then invokes
the actual adaptation writer. Thresholds, duration and mixing factor remain
at their defaults. The prospective interval is `[1,1+2^-12]` and proof radius
`rho_star=3/2`; the reference-comparison condition admits this geometry.
There is no alternate solver, gate tuning or fitted sustaining force.

The observed writer admits only node three during this horizon. Its neighbors
already have capacity one, so every admitted update is an identity and the
initial capacity difference persists. High Si alone does not admit the other
nodes: their fresh pressure fails the stability gate. Nevertheless winding
one remains observed and phase deformation decreases. This is useful partial
eligibility evidence, not numerical or physical capacity consensus.

The source-bound control uses seed 17. The rational tube has displayed upper
radius 1.433137, strictly below 1.5; its reference-gap allowance is approximately
0.0247903. All 33 observed endpoints satisfy their prospective comparisons
with the declared `2e-13` observation tolerance. At T=4:

| Quantity | Observed endpoint | Prospective continuous-model upper bound |
| --- | --- | --- |
| Gap-deformation norm | 0.12223481 | 0.35128774 |
| Actual-conductance EPI Dirichlet energy | 0.01487817 | 0.14148314 |

The actual-strength weighted form mean changes from 0.125 to approximately
0.12726777. This is a finite measured drift, not an infinite-time growth
theorem. Maximum observed Euler and pressure-assembly defects are below
`1.4e-17` and `7.1e-18`; the independent 80-digit estimate of local phase-step
arithmetic error is below `5.9e-16`. These local observations do not bound
global numerical error or prove a future chart margin.

The finite ledger retains the actual mask, counters and physical time, plus
the represented averaging defect. With exact rationals of binary64 inputs,

```text
nu_hat = nu - (mu/2)*M*L_C*nu,
epsilon = nu_after-nu_hat,
E_nu_after-E_nu_before
 = (E_nu_hat-E_nu_before)
   + epsilon^T*L_C*nu_hat + epsilon^T*L_C*epsilon/2.
```

This separates ideal masked dissipation from arithmetic effects without
assuming a uniform numerical contraction. The retained one-ulp stalling
control supplies the complementary rounding boundary. Fresh pressure work,
capacity-channel work, Euler state defects and finite field values use the
existing observation owners. Observed endpoints inside the prospective
envelopes do not prove binary64 convergence or validate unobserved times.

This closes the bounded capacity-forcing admission: a conditional robust
phase/winding and form-contrast tube, with actual gate limitations retained.
Exact recovery does not follow. Section 46 addresses the separate
absolute-form/clipping obligation.

## 46. Absolute form: a finite bound and an interval-law obstruction

### Fixed observation weights separate mean drift from coordinate changes

Retain Section 45's fixed conductance W, strengths s and `S=sum s_i`.
For the actual-strength mean `m_D=s^T*x/S`, the nodal row gives

```text
m_D_dot = [-e*nu^T*B_W*x + (s*nu)^T*F]/S,
F = w*g - (f/2)*L_C*nu.
```

The first term need not vanish with heterogeneous capacity. This mean uses
fixed observation weights and is well defined for measurable capacities;
it needs no capacity derivative. Capacity-only events leave m_D unchanged.
The read-only `observe_forcing_mean_balance` in the
[forcing owner](../../src/tnfr/physics/forcing_realization.py) contracts the
same validated channel data as the Dirichlet observer with covector
`s_i*nu_i/S`. It retains signed diffusion, separate channel work, fresh
assembly defects and stale stored-pressure work. It does not subtract the
observed mean or install a balancing force.

For one retained Euler interval of duration dt and shared state defect r_x,

```text
m_D_after-m_D_before
 = dt*(modeled_mean_rate + kernel_defect_rate + stored_residual_rate)
   + s^T*r_x/S.
```

The existing 32-interval control verifies this exact rational identity and
its telescope, using the same execution rather than a new producer. Its
observed weighted-mean shift remains approximately 0.00226777. This is a
finite source/transport budget, not an asymptotic drift prediction.

The alternative moving metric `H(t)=diag(s_i/nu_i(t))` has different duties.
If capacities are differentiable, `h=s/nu`, `Z=sum h_i` and `m_H=h^T*x/Z`
give `m_H_dot=s^T*F/Z+h_dot^T*(x-m_H*1)/Z`. At a capacity jump its coordinate
reweights even at unchanged EPI. That jump is already owned by
`observe_forced_support_reset`, which reports `mean_reweighting` separately
from form evolution. No derivative of a merely measurable capacity schedule
is assumed by the new fixed-weight observer or interval theorem.

<a id="prescribed-capacity-and-event-mean-balance"></a>

### Prescribed capacity changes distinguish form charge from its mean

The moving-weight identity has a useful consequence for the complete
[normalized-sine law](SINE_CONSTITUTIVE_INFORMATION.md#global-closure-pressure-comparison).
Keep fixed finite connected simple unit support with positive degrees, fixed `e>=0`,
`w,beta>0`, no Gamma or other forcing, and both prescribed sine rows.
Let capacities be strictly positive `C^1` time schedules, supplied
independently of form and phase. Set `rho_i=d_i/nu_i(t)`, `W=sum_i rho_i`,
`Q=rho^T*x` and `M=Q/W`. Reciprocity cancels the instantaneous source and
diffusion contributions even while these weights vary, giving

\[
\dot Q=\dot\rho^{\mathsf T}x,\qquad
\dot M=\frac{\dot\rho^{\mathsf T}(x-M\mathbf1)}W.
\]

For this prescribed schedule, `dot M=0` at every form state is equivalent
to `dot rho=(dot W/W)*rho`. Integrating on a connected time interval gives
`rho(t)=rho(0)/a(t)`, or

\[
\nu_i(t)=a(t)\nu_i(0),\qquad a(t)>0,\quad a(0)=1.
\]

Thus a common proportional capacity schedule is necessary and sufficient
for universal conservation of this normalized mean under the stated law.
For raw `Q`, universal conservation instead requires `dot rho=0`.
These necessities concern prescribed state-independent schedules. A
state-dependent capacity law can make `dot rho` orthogonal to the particular
centered form without being proportional; that different admission problem
is not resolved here.

The common factor multiplies **both** consumed sine rows, so
`tau(t)=integral_0^t a(s) ds` gives the existing fixed-capacity law in the
activity clock. Along the corresponding solution,
`Q(t)=Q(0)/a(t)` and `W(t)=W(0)/a(t)`, while `M` stays constant.
This is consistent with the [whole-law clock contract](../NODAL_PARAMETER_FOUNDATIONS.md#31-capacity-clock-and-positivity-require-compatible-laws),
not a clock inferred from phase or a derived capacity mechanism. An
unbounded activity horizon is additionally needed to transfer an entire
fixed-capacity asymptotic limit. The held-capacity form-charge theorem and
this variable-capacity accounting therefore have distinct hypotheses.

### A no-reset support event must retain its changed weights

For two supplied snapshots on the same ordered node set, assume all old and
new degrees and capacities are positive and `x^+=x^-=x`. Define prescribed
state-independent weights `rho_i^\pm=d_i^\pm/nu_i^\pm`, with
`W^\pm=sum_i rho_i^\pm`, `Q^\pm=(rho^\pm)^T*x` and
`M^\pm=Q^\pm/W^\pm`. Direct subtraction gives

\[
M^+-M^-=
\frac{(\rho^+-\rho^-)^{\mathsf T}(x-M^-\mathbf1)}{W^+}.
\]

Mean preservation for every form state is therefore equivalent to
`rho^+/W^+=rho^-/W^-`, or `rho^+=c*rho^-` for one positive scalar `c`.
For example, a supplied update
`nu_i^+=(d_i^+/d_i^-)*nu_i^-/c` has that property; it is one compatible
event prescription, not a cause of support change or a selected capacity law.
Raw charge is preserved for all forms only when `rho^+=rho^-`.

Common-origin covariance makes the distinction unavoidable. Replacing
`x` by `x+h*1` changes `Q^+-Q^-` by `h*(W^+-W^-)`, whereas it leaves
`M^+-M^-` unchanged. Thus requiring raw-charge conservation across an event
with changing `W` requires a fixed form-origin convention unless additional state
or event terms transform with it. This is separate from the support's
Dirichlet-storage jump, which depends on form differences.

The existing `observe_forced_support_reset` already reports the exact
same-EPI `mean_reweighting` and independent support-storage change; its
detached snapshots do not assert that an event occurred. The
[static composition controls](../../tests/physics/test_relational_pressure_composition.py)
reuse that reader for a supplied P3-to-K3 comparison and check the complete
sine rows under common capacity scaling. No new runtime, support event,
capacity law or frozen trajectory is introduced.

### The complete scalar field has a finite-prefix enclosure

The positive-transport maximum principle works directly with Section 45's
source envelope. At a maximal form coordinate diffusion contributes a
nonpositive rate; at a minimum it contributes a nonnegative rate. Consequently,
with `nu_max=b` and `A(T)=integral_0^T ||F(t)||_2 dt`,

```text
min x(0) - b*A(T) <= x_i(t) <= max x(0) + b*A(T),  0<=t<=T.
```

This bound controls the whole physical prefix, including times between the
displayed samples. Using `q_tail=B/gamma`, Section 45 supplies

```text
I_q(T) = q_tail*T + (Q-q_tail)*(1-exp(-gamma*T))/gamma,
A(T) <= (w/pi)*I_q(T) + f*B*T.
```

The cycle sample now exposes `gap_deviation_integral_upper`,
`non_epi_source_integral_upper` and `epi_interval`. Shared exponential
enclosures handle either sign of `Q-q_tail`; no fitted rate or new solver
is introduced. If this entire interval lies strictly inside declared scalar
rails, the exact continuous model does not require clipping on that prefix.
Numerical execution still retains its own endpoint and pressure defects.
For the retained full-mixture study, the computed T=4 interval lies inside
the outward rational band `[-0.225,0.475]`, strictly inside its `[-1,1]`
rails. This certifies finite-prefix inactivity in the declared continuous
model; the separately observed binary64 endpoints also lie within the band.
The zero-source control preserves the initial extrema. With a persistent
nonzero source bound this enclosure generally expands with T, so it alone
does not imply an all-time scalar bound.

### A locked phase can still drive the uniform EPI mode

An explicit exact-model member of the interval family rules out that stronger
implication. This is a counterexample to the sufficiency of interval premises,
not a proposed mechanism or a prediction of the default adaptation schedule.
It retains the actual C5 transport: four unit edges and closing weight beta.
Put `v=(1,0,0,0,1)`, so `s=2*1+(beta-1)*v`, and choose

```text
d = 2*pi/5,
h_* = alpha*(v-(2/5)*1),
delta_* = (d-alpha, d, d, d+alpha, d),
nu_* = kappa*1-K*J(h_*).
```

Here alpha is a declared small geometric displacement, not a response gain.
For `abs(alpha)<rho_star-d`, the phase lock stays inside the same acute/U3
chart, since rho_star is below both limits. Since sine is 1-Lipschitz,
`abs(nu_*_i-kappa)<=K*abs(alpha)/2`. Choose kappa at the capacity interval's
midpoint and `K*abs(alpha)<=b-a`; all capacities stay inside the prescribed
positive interval. The supplied phase law then gives `theta_dot=kappa*1`
exactly. No pressure coefficient or supporting force is fitted.

For the complete configured phase/capacity source, direct contraction gives

```text
F_* = -(w/(2*pi))*L_C*h_* - (f/2)*L_C*nu_*,
s^T*F_* = -(beta-1)*[w*alpha/pi + f*K*cos(d)*sin(alpha)].
```

Choose alpha with the opposite sign to `beta-1`. For beta unequal to one
and either positive w or positive f, this expression is strictly positive.
The retained closing weight is below one, and its full default mixture has
both channels positive. Thus a positive arbitrarily small alpha suffices.
For unit closing weight this particular obstruction vanishes, retaining
the support/transport distinction rather than discarding actual weights.

The existing compatibility and forced-support theorem in Sections 2-3 now
applies with these fixed capacities and locked source:

```text
Z = sum_i s_i/nu_*_i,
c = (s^T*F_*)/Z > 0,
m_H(t) = m_H(0)+c*t.
```

EPI contrast relaxes toward its bounded relative profile while every form
coordinate acquires the same secular drift. For positive phase weight an
explicit lower bound avoids any approximate sine evaluation:

```text
c >= a*abs(beta-1)*w*abs(alpha)/(pi_upper*(8+2*beta)) > 0.
```

The lower capacity endpoint here is a, distinct from the displacement alpha.
If initial EPI is uniformly x_0 and the phase starts at this lock, any finite
upper rail U must be reached by the ideal unclipped solution no later than
`(U-x_0)/c_lower` (strict crossing for larger times). Clipping changes the
equation; an endpoint pinned to a rail is not a zero-pressure equilibrium.

The obstruction is not confined to initialization exactly at the lock.
Choose the displacement also inside `abs(alpha)<rho_star-d`, freeze the
same nu_* and start at the original retained phase endpoint.
Section 45 admits the same interval tube. The centered phase equation is
the gradient flow of the sine alignment potential plus its fixed linear
tilt. Strong convexity on that admitted chart makes `h(t)-h_*` decay
exponentially. Hence `F(t)-F_*` is integrable and
`m_H(t)=c*t+O(1)`. This extends the failure of a uniform all-time bound to
the retained phase preparation in the exact interval class. It does not
assert that the invoked default capacity writer keeps nu_* fixed forever.

The static obstruction control reuses the retained support with
`alpha=2^-12`, `kappa=1+alpha/2`, `K=1/2`, and the full default mixture.
Independent high-precision contractions verify the locked response, interval
inclusion and positive source expression; separate comparisons retain errors
of represented pressure and phase proposals. The sign proof above is exact,
not inferred from those finite numerical tolerances. No long trajectory,
new lock classification or automatic change of the constitutive law is needed.

This closes the absolute-form obligation with its proper scope: finite-prefix
scalar control is available, but interval-bounded capacity and a stable phase
geometry alone do not guarantee indefinite absolute EPI retention. The actual
adaptation policy's first effect on this source compatibility is examined
below; its repeated effect remains subject to the single execution plan.

## 47. Actual capacity feedback reduces the locked source without cancelling it

### One signed balance connects capacity averaging to phase readjustment

Retain Section 46's ordered weighted C5, full pressure mixture and acute
two-neighbor phase chart. Write `L=L_C`, `L_U=L/2` and let M be the diagonal
mask of eligible nodes. The ideal snapshot average and its represented
writer defect r satisfy

```text
delta_nu = -mu*M*L_U*nu + r,
b = s^T*F,
delta_b = -f*s^T*L_U*delta_nu
        = mu*f*(L_U*s)^T*M*L_U*nu - f*s^T*L_U*r.
```

This is an event identity at unchanged EPI, phase, support and weights.
The phase source does not jump. Symmetry of the unweighted support gives
the second equality; actual EPI strengths s still carry the unequal closing
conductance. The signed cross term is different from the nonnegative
capacity-energy dissipation in Section 44. Capacity smoothing alone therefore
does not establish decreasing `abs(b)`.

For the exact locked family of Section 46, a full mask and a positive blend, put
`t=beta-1<0`, `d=2*pi/5`, `C=cos(d)*sin(alpha)>0`. Then

```text
delta_nu = (mu*K/2)*L*J,
s^T*L^2*J = -5*t*C,
delta_b = 5*mu*f*K*t*C/4 < 0,
b_after = -t*[w*alpha/pi + f*K*C*(1-5*mu/4)].
```

These are exact-model expressions, before the represented writer defect.
For the mathematical blend `mu=1/10`, the capacity contribution is multiplied
by `7/8`; with the exact rational represented by binary64 `0.1`, this factor
is `7/8-2^-57`. In both cases the full source remains positive. The phase
contribution is unchanged by the capacity-only event.

The same event changes the supplied phase velocity from `kappa*1` to
`kappa*1+delta_nu`. Subtract its uniform component to obtain the new lock
mismatch. It generally leaves the one-parameter locked family; reducing
alpha is not a justified description of the new state. If the new capacity
is then held while phase evolves, the acute midpoint identity gives

```text
g_dot = -(1/pi)*L_U*theta_dot,
b_dot_phase(0+) = -(w/pi)*s^T*L_U*delta_nu
               = [w/(pi*f)]*delta_b,                 f>0,
               = 5*mu*w*K*t*C/(4*pi) < 0             (ideal full mask).
```

The first equality reuses the existing local phasor derivative, not a new
phase law. For an imperfect represented initial lock, the displayed event
formula is the **change** of its instantaneous phase-source derivative;
the retained pre-event derivative must be added. The compatibility jump and
the subsequent derivative have different units and are not two immediate
source reductions. The negative tangent proves neither future monotonicity
nor a finite accumulated source. It does identify a shared signed geometric
contraction for the two feedback paths.

### Fresh default admission gives a nontrivial first write

The [capacity synergy controls](../../tests/physics/test_cycle_capacity_synergies.py)
centralize the Section 46 lock preparation in one fixture. It retains the
actual post-UM weighted support (seed 17), declares the full default pressure
mixture, initializes `alpha=2^-12`, `kappa=1+alpha/2`, `K=1/2`, and starts
stability counters at zero. The bounded control executes exactly five shared
phase/EPI intervals of `dt=1/8`. At every endpoint it refreshes full pressure
and then Si before calling the unchanged capacity writer. This is the same
declared composition as Section 45, on a different, explicitly prepared
initial state; it is not the complete native scheduler.

All five nodes pass the fresh gates on each call: the largest absolute
pressure decreases from approximately `3.005e-5` to `2.683e-5`, below the
default `0.001`, and the smallest observed Si exceeds `0.9416`, above `0.5`.
No capacity changes during the first four calls. All nodes become eligible
at the default `tau=5`, physical time `5/8`, with the represented default
blend `mu=0.1`. No gate, pressure weight, state amplitude or horizon is tuned
after inspecting the response.

At the resulting capacity-only event, the retained observations give:

| Readout | Before | After / change |
| --- | --- | --- |
| Source compatibility `s^T*F` | `1.37059588934e-5` | `1.36443423539e-5`, still positive |
| Frozen reversible-mean coefficient `c=b/sum(s/nu)` | `1.43509598918e-6` | `1.42864449498e-6` |
| Maximum absolute centered phase-rate mismatch | approximately `1.11e-16` | `2.83020e-6` |
| Actual fixed-strength EPI mean `m_D` | retained | exactly unchanged by the event |
| Reversible-metric EPI mean `m_H` | retained | reweighting of approximately `-3.40878e-11` |
| Maximum absolute capacity writer defect | — | approximately `1.78e-16` |

The comparison uses the existing `observe_forcing_capacity_difference` and
`observe_forcing_mean_balance`; it adds no production observer, integrator or
controller. Exact rational contractions separate the source change from
fresh pressure-assembly and stale stored-pressure differences. The capacity
write does not refresh stored pressure; a detached fresh capture supplies
the post-event comparison. Shared Euler residuals separately account for
the preceding form evolution. EPI and phase are exactly unchanged across
the capacity event, so the jump of the moving metric is not physical form
motion. The frozen coefficient `c=b/sum(s/nu)` is a separate held-model
quantity, not the instantaneous derivative of the fixed-strength mean.
The signed contraction of the post-event pressure-assembly defect is about
`-8.87e-21`; the stale stored-pressure residual is instead approximately
`+6.16165e-8`. Ignoring freshness would hide almost the entire source change.

Applying the ideal midpoint Jacobian to the retained represented phase-rate
coefficients gives a numerator `pi*b_dot_phase` changing from approximately
`-2.12e-17` to `-8.01705e-7`. This differentiates the declared local model,
not the binary64 pressure program. Exact signed contractions and
independent 110-digit evaluation check the analytic expressions above. One
unapplied proposal by the shared phase owner also verifies the newly induced
gap displacement; it does not extend the executed horizon or certify a new
lock. These local comparisons retain finite rounding errors rather than
promoting the binary64 state to an exact analytic lock.

### A smaller capacity contrast is not a source-compatibility Lyapunov law

The [existing difference controls](../../tests/physics/test_forcing_capacity_difference.py)
include a complementary exact detached map. Take the same C5 support with
closing weight `beta=3/4`, uniform phase, uniform EPI and
`nu=(1,1,1+R,1,1)`, `R=2^-12`. Declare a full dyadic blend `mu=1/2` and
pressure weights `(w,e,f,topo)=(1/2,1/4,1/4,0)`. Its capacity range halves
and its unweighted Dirichlet energy falls from `R^2` to `R^2/8`, but

```text
b_before = 0,
b_after = mu*f*(beta-1)*R/2 = -mu*f*R/8 < 0.
```

Thus `abs(b)` increases while both smoothing readouts improve. Unit closing
weight gives zero and `beta=5/4` reverses the sign with the identical capacity
map. These controls are detached algebra, not observed default eligibility,
locked-phase states or alternative settings for the five-step experiment.
A separate masked contraction checks the same signed balance with the
captured transport strengths.

This closes the first-feedback gate: the implemented default writer is
admitted and weakens this specific positive source; its phase readjustment
initially acts in the same direction. It does not cancel the source in one
event. Recurrent admission, the accumulated source under repeated updates,
binary64 residuals and indefinite absolute-form retention remain unproved.
These are mechanism conditions, not evidence of physical particle emergence.
