# C6 relational regions and exit exclusions

Correlated regions, histories and regional exclusion proofs.

Part of [Cycle winding under Coupling: scope and retained evidence](../../../COUPLING_WINDING_PERSISTENCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

## 39. B45: the coordinate arithmetic class does not rescue local mean confinement

B2.d.45 strengthens section 38 by imposing a necessary arithmetic
restriction inherited from B43's actual starting state. The uniform-carry
templates used by B44 need not satisfy this finer restriction. The new
construction places both outward witnesses in the correct coordinate
arithmetic class while preserving their displayed canonical pressures.
It still addresses a proposed region, not reachability of its witnesses
along the saved trajectory.

The owner
[`c6_carried_affine_mean.py`](../../../../src/tnfr/physics/c6_carried_affine_mean.py)
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

The owner is [c6_carried_relay.py](../../../../src/tnfr/physics/c6_carried_relay.py),
with a reproducible producer in
[c6_winding_relay_budget.py](../../../../benchmarks/c6_winding_relay_budget.py).
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

The new APIs extend [the existing relay owner](../../../../src/tnfr/physics/c6_carried_relay.py):
`derive_c6_carried_local_relay_budget` and
`observe_c6_carried_local_relay_exit`. The
[handoff producer](../../../../benchmarks/c6_winding_relay_handoff.py) reuses
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
pressure as well as the corrected-coordinate budget. Continuing from this
dependency boundary requires rederiving the affected relay to
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
[`derive_c6_carried_viability`](../../../../src/tnfr/physics/c6_carried_viability.py)
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
[`c6_carried_viability.py`](../../../../src/tnfr/physics/c6_carried_viability.py).

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
[`c6_carried_viability.py`](../../../../src/tnfr/physics/c6_carried_viability.py).

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

The unresolved obligation at this stage is a compact proof over the four remaining exit classes,
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
[`c6_carried_return.py`](../../../../src/tnfr/physics/c6_carried_return.py) owner as
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
