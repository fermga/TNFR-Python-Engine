# C6 carried flow and pressure cells

Exact carry, live-pressure and finite itinerary results under their numerical model.

Part of [Cycle winding under Coupling: scope and retained evidence](../../../COUPLING_WINDING_PERSISTENCE.md). Section numbers are stable across the collection; hypotheses and model changes remain local to each result.

**Archived research record.** This preserves conditional derivations and source-bound evidence. Its local next-step language is historical and does not schedule work.

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

[`certified_two_neighbor_phase`](../../../../src/tnfr/mathematics/_phase_midpoint.py)
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

[`c6_winding_phase_kernel.py`](../../../../benchmarks/c6_winding_phase_kernel.py)
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
[`derive_binary64_quarter_pressure_box`](../../../../src/tnfr/physics/binary64_nodal_flow.py)
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
[`c6_winding_pressure_cells.py`](../../../../benchmarks/c6_winding_pressure_cells.py)
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
[`_euler_kernel.py`](../../../../src/tnfr/dynamics/_euler_kernel.py) now additionally
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
[`observe_nodal_remainder_sequence`](../../../../src/tnfr/physics/nodal_remainder.py)
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

[`c6_winding_nodal_remainder.py`](../../../../benchmarks/c6_winding_nodal_remainder.py)
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
[`execute_nodal_remainder_event_schedule`](../../../../src/tnfr/operators/nodal_remainder_runtime.py),
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

[`observe_nodal_remainder_pressure_readout`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

[`c6_winding_remainder_runtime.py`](../../../../benchmarks/c6_winding_remainder_runtime.py)
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

[`derive_binary64_c6_pressure_equilibrium_obstruction`](../../../../src/tnfr/physics/binary64_pressure_equilibrium.py)
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

[`derive_nodal_remainder_cell_horizon`](../../../../src/tnfr/physics/nodal_remainder.py)
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

<a id="evidence-and-the-next-gate"></a>
### Evidence and extension conditions

[`c6_winding_pressure_equilibrium.py`](../../../../benchmarks/c6_winding_pressure_equilibrium.py)
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

Extending this result requires a joint generated-pressure/domain argument
that permits the finite-resolution behavior actually present: a signed block
source bound and a preserved trapping region under the actual UM/IL phase
updates, rather than a fixed-phase exact zero that this numerical
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

[`observe_c6_coupling_coherence_phase_step`](../../../../src/tnfr/physics/c6_phase_orbit.py)
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

The existing [pressure-readout owner](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

[`c6_winding_phase_orbit.py`](../../../../benchmarks/c6_winding_phase_orbit.py) preserves
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

[`derive_c6_pressure_lattice`](../../../../src/tnfr/physics/c6_pressure_lattice.py)
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

[`observe_finite_nodal_pressure_drift`](../../../../src/tnfr/physics/nodal_remainder_pressure.py)
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

<a id="ownership-and-next-gate"></a>
### Ownership and scope

[`c6_winding_pressure_lattice.py`](../../../../benchmarks/c6_winding_pressure_lattice.py)
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

[`derive_nodal_remainder_itinerary`](../../../../src/tnfr/physics/nodal_remainder.py)
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

[`c6_winding_carry_itinerary.py`](../../../../benchmarks/c6_winding_carry_itinerary.py)
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

<a id="next-gate-close-a-relational-class-or-exclude-it-structurally"></a>
### Requirements for closure or structural exclusion of a relational class

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
