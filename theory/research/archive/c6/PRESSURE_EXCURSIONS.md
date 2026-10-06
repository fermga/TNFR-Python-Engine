# C6 pressure excursions and compensation

Finite pressure, mean-budget and compensation obstructions; no global closure.

Part of [Cycle winding under Coupling: scope and retained evidence](../../../COUPLING_WINDING_PERSISTENCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

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

[`derive_c6_pressure_sign_sector`](../../../../src/tnfr/physics/c6_pressure_lattice.py)
uses monotonicity of both nearest-even roundings. In a nonempty negative
sector, its largest index gives the least negative pressure bound; in a
positive sector, its smallest index gives the least positive bound. Thus
`s*p_i>=c>0` throughout the selected sector, for `s=-1` or `s=+1`.
The index range is an enclosure: no claim that every enclosed integer
gradient has a realizable global EPI tuple is needed for this bound.

[`observe_c6_pressure_sector_exit`](../../../../src/tnfr/physics/c6_pressure_lattice.py)
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

[`observe_nodal_remainder_cell_exit`](../../../../src/tnfr/physics/nodal_remainder.py)
now owns the analytic-horizon, shared-prefix replay and unchanged-cell
checks used by the section 24, 25 and 26 benchmarks. Its explicit step
budget is a resource limit, not a TNFR coefficient. A stationary horizon,
an exit beyond that budget or an exact-band departure is rejected before
allocating the step schedule. Accepted replays retain their original carry
and check every pre-update visible state, each intermediate output and the
exact first-exit endpoint. Pressure constancy still requires the caller's
fixed visible-state pressure map; the numeric helper does not prove it.

### The inherited continuation reaches positive node-1 pressure

[`c6_winding_pressure_sign.py`](../../../../benchmarks/c6_winding_pressure_sign.py)
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

The remaining condition at this boundary is the positive sector's admissible
duration and signed area, coupled to the other five node budgets. Subsequent
transitions would have to compensate prior negative area
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

[`derive_nodal_area_crossings`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

[`derive_two_level_nodal_return`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

<a id="evidence-owner-and-next-gate"></a>
### Evidence owner and extension conditions

[`c6_winding_pressure_repayment.py`](../../../../benchmarks/c6_winding_pressure_repayment.py)
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
[`derive_finite_level_nodal_return`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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
[`observe_finite_nodal_pressure_drift`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

[`c6_winding_pressure_levels.py`](../../../../benchmarks/c6_winding_pressure_levels.py)
revalidates the complete B29 report through its B28/B27/B26 inputs and
binds the four historical input byte hashes. Shared cell-exit and inverse
itinerary owners preserve the incoming carry; the signed prefix observer
retains the B27 area reference. Resource exhaustion is reported as censoring,
not as absence of another pressure level. This remains a conditional
numerical path, without live graph or original-preparation reachability.

Extending this retained path requires a compatible change of the neighborhood sustaining the
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

[`observe_c6_frozen_pressure_stencil`](../../../../src/tnfr/physics/c6_pressure_lattice.py)
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
[`observe_nodal_remainder_cycle_gradient`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
now owns the relation
`Delta m=-2*L_rw*A/delta+2*L_rw*Delta r/delta` used by the earlier
benchmarks. It first checks the complete identity `A=X_after-X_before`.
Taking only the Laplacian would miss an arbitrary uniform error in the
area vector. The previous callers already checked their full prefix
areas elsewhere; centralization makes that obligation explicit in the
reusable owner and preserves their existing report payloads.

<a id="evidence-and-next-gate"></a>
### Evidence and remaining requirements

[`c6_winding_pressure_stencil.py`](../../../../benchmarks/c6_winding_pressure_stencil.py)
revalidates B30 through the retained B29/B28/B27/B26 chain, uses only the
shared nodal integrator, and retains whole-itinerary compatibility and
all six signed area budgets. Its ceiling is computational censoring, not
an impossibility theorem for reaching negative pressure. Neither a live
graph nor original-preparation reachability is promoted by this audit.

Extending this boundary result requires a coupled neighborhood reachability or trapping argument
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
[`c6_carried_profile.py`](../../../../src/tnfr/physics/c6_carried_profile.py)
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

[`c6_carried_tube.py`](../../../../src/tnfr/physics/c6_carried_tube.py)
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

[`c6_winding_coupled_budget.py`](../../../../benchmarks/c6_winding_coupled_budget.py)
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
[`c6_carried_closure.py`](../../../../src/tnfr/physics/c6_carried_closure.py)
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
[`c6_carried_balance.py`](../../../../src/tnfr/physics/c6_carried_balance.py)
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
[`c6_carried_passage.py`](../../../../src/tnfr/physics/c6_carried_passage.py)
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

[`c6_winding_coupled_passage.py`](../../../../benchmarks/c6_winding_coupled_passage.py)
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
[`c6_carried_cell_escape.py`](../../../../src/tnfr/physics/c6_carried_cell_escape.py)
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
[`c6_carried_cell_graph.py`](../../../../src/tnfr/physics/c6_carried_cell_graph.py)
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
[`c6_winding_temporal_compatibility.py`](../../../../benchmarks/c6_winding_temporal_compatibility.py)
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
[`nodal_remainder_runtime.py`](../../../../src/tnfr/operators/nodal_remainder_runtime.py)
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
[`c6_winding_phase_tail_runtime.py`](../../../../benchmarks/c6_winding_phase_tail_runtime.py)
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
[`c6_carried_mean_cylinder.py`](../../../../src/tnfr/physics/c6_carried_mean_cylinder.py)
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
