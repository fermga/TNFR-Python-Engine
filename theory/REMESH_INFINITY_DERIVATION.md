# REMESH Fixed-Delay Models and Runtime-Limit Boundary

**Status**: CORRECTED N15 HISTORICAL RECORD — restricted finite cyclic,
companion, conditional policy, robust-defect and represented-number boundary
results, including transactional per-invocation P2 EN/REMESH execution and
finite certification; runtime limit and catalog completeness open
**Date**: May 26, 2026 — corrected September 2026
**Owner**: `theory/REMESH_INFINITY_DERIVATION.md`
**Source implementations**: `src/tnfr/operators/remesh.py::apply_network_remesh`,
`src/tnfr/physics/remesh_history_stability.py`,
`src/tnfr/physics/remesh_schedule_policy_stability.py`,
`src/tnfr/physics/remesh_schedule_policy_stability.pyi`,
`src/tnfr/physics/remesh_schedule_relative_defect_stability.py`,
`src/tnfr/physics/remesh_schedule_relative_defect_stability.pyi`,
`src/tnfr/physics/binary64_remesh_relative_defect.py`,
`src/tnfr/physics/binary64_remesh_relative_defect.pyi`,
`src/tnfr/physics/binary64_p2_reception_stability.py`,
`src/tnfr/physics/binary64_p2_reception_stability.pyi`,
`src/tnfr/physics/runtime_p2_reception_stage.py`,
`src/tnfr/physics/runtime_p2_reception_stage.pyi`,
`src/tnfr/physics/runtime_p2_reception_remesh_sequence.py`,
`src/tnfr/physics/runtime_p2_reception_remesh_sequence.pyi`,
`src/tnfr/physics/runtime_p2_reception_remesh_policy.py`,
`src/tnfr/physics/runtime_p2_reception_remesh_policy.pyi`,
`src/tnfr/physics/runtime_remesh_history_stability.py`,
`src/tnfr/physics/remesh_schedule_stability.py`,
`src/tnfr/physics/runtime_remesh_schedule_stability.py`,
`src/tnfr/operators/event_remesh_causal_runtime.py`,
`src/tnfr/physics/runtime_remesh_schedule_block_margin.py`, and
`src/tnfr/physics/runtime_remesh_schedule_relative_defect.py`, with their typed
interfaces where present, plus the finite
reference-family certificate and typed interface in
`src/tnfr/physics/event_remesh_reference.py` and
`src/tnfr/physics/event_remesh_reference.pyi`.

---

## Abstract

The historical N15 programme used the name $\mathcal R_\infty$ for several
different objects: a delay parameter limit, iterates of a history-advance map,
and a Fourier projection. Those objects are not interchangeable.

Two exact finite results survive after separating them. On a **finite cyclic
history window**, with fixed delays, fixed $0<\alpha<1$, and no clipping, the filter

$$F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}$$

is a normal contraction because $S$ is a unitary cyclic shift and
$\beta,\gamma,\delta>0$ sum to one. Its Cesàro averages converge to the
orthogonal projection onto $\ker(I-F)$. The fixed modes satisfy both delay
conditions, so the fixed subspace consists of sequences whose periods divide
$\gcd(\tau_l,\tau_g)$.

For the distinct **finite companion history recurrence**, one fixed uniform
$\alpha\in[0,1]$, fixed delays, fixed ordered spatial support, one fixed
positive diagonal metric and no clipping yield an exact stationary-weighted
Jensen Lyapunov functional. It is nonincreasing for every history transition.
When $0<\alpha<1$, the companion is primitive and every spatial coordinate
converges to its stationary history barycenter. At $\alpha=1$, the companion is
a permutation: the functional is conserved and periodic histories remain
possible.

For a further conditional **exact REMESH/schedule policy class**, assume every
possibly varying schedule preserves spatial consensus and has one common
fixed-metric disagreement gain bound $q$. The post-schedule history energy has
prefix gain at most one and block gain at most $q$ over the sufficient horizon
$L=\texttt{active\_max\_delay}+1$. Thus $q<1$ gives uniform normalized margin
$1-q$ and repeated geometric decay of spatial disagreement. This theorem does
not verify the schedule family or identify it with the binary64 runtime.

A robust conditional extension admits a bounded signed pre-schedule head
defect. If $J_k$ is the Jensen energy envelope of the active REMESH history,
$y_k$ is the ideal head, $z_k$ is the bounded head and
$E_H(z_k)-E_H(y_k)\leq\eta J_k$, then the combined head gain is
$q_{\mathrm{eff}}=q(1+\eta)$. The same companion proof gives nonincrease for
$q_{\mathrm{eff}}\leq1$ and geometric spatial-disagreement decay for
$q_{\mathrm{eff}}<1$. A finite causal observer checks these inequalities on a
recorded execution block. It does not prove that one uniform $\eta$ is forward
invariant for a runtime class.

The represented-number boundary is now exact at two complementary points.
A pairwise reduction computes the optimal nonnegative relative defect for one
admissible pair of binary64 temporal tuples. A normal-valued
$\alpha=1/2$ witness requires $\eta=2^{210}-1/4$, so a bounded hard-clipped
box alone is far too broad for the existing $q=9/16$ policy. At $\alpha=1$,
every sufficient finite represented history inside a fixed hard-clipping
interval instead has $\eta=0$ and is forward invariant under REMESH alone.
This boundary class does not by itself certify a schedule or its execution.

A first restricted global represented numeric EPI kernel closes the composition
on $P_2$.
With mutual singleton neighbor sets, an immutable all-target Reception snapshot,
the exact binary64 mix `0.5` and one common hard clamp, both target proposals
are the same reversed sum of two half-scaled operands. They are numerically
equal for every finite represented input pair in the source interval, so the
global kernel gain is
$q=0$. Composed with the $\alpha=1$, $\eta=0$ REMESH class, the active
post-schedule history has zero spatial disagreement after
$L=\tau_g+1$ cycles. This proves arbitrary finite repetition of the restricted
numeric EPI kernels. One sealed adapter binds a single executed,
grammar-admitted two-phase EN EPI stage and its captured endpoint to the first
kernel. A second adapter binds each such stage and its same-cycle $\alpha=1$
REMESH transition inside one completed graph-owned causal sequence. For an
observed sequence of $N\geq L$ cycles, it certifies extinction only on the active
history suffix of length $L$. It does not prove future or unobserved complete
runtime repetition.

Neither base finite model is the complete clipped, history-gated runtime
operation.
They do not establish a literal $\tau_g\to\infty$ limit, conserve the TNFR
structural charge, make the full tetrad energy monotone, imply spatial
consensus or $\Delta\mathrm{NFR}=0$, or prove that the 13 registered operators
exhaust all admissible TNFR transformations.

A separate effective-$P_2$ reference family now connects three executed
pressure-refreshed Euler meshes to the exact continuous nonuniform mode and to
one unit-delay REMESH step. It supplies a rational finite-mesh error enclosure,
strict improvement under the two declared proper subdivisions, exact ideal
REMESH error scaling and a runtime residual bound. This finite family is not a
general or binary64 asymptotic convergence theorem.

The sections below retain the N15 programme structure and commit anchors while
recording the corrected statements.

---

## §1. Runtime REMESH Semantics

### §1.1 Exact raw update

For every node with sufficient history, `apply_network_remesh` computes

$$
x_{\rm raw}
=(1-\alpha)^2x_0+\alpha(1-\alpha)x_l+\alpha x_g,
$$

where $x_0$ is the current EPI and $x_l,x_g$ are snapshots at the configured
local and global delays. With

$$
\beta=(1-\alpha)^2,
\qquad
\gamma=\alpha(1-\alpha),
\qquad
\delta=\alpha,
$$

the coefficient identity is exact:

$$\beta+\gamma+\delta=1.$$

The runtime factor contract requires 0 < alpha <= 1. It rejects values outside
that interval instead of clamping them. The exact-real coefficients represented
by a validated alpha are therefore nonnegative and sum to one. The implementation
evaluates the equivalent nested binary64 expression; the optional evidence
reports its rounding residual from the exact-real affine value.

### §1.2 Guards, shape and clipping

Both delays are strict positive integers. The outer EPI history must be replayable
and support indexed access. If it contains fewer than max(tau_l, tau_g) + 1
snapshots, the planner returns an immutable insufficient-history no-op and the
executor changes no graph state. Empty live support returns the distinct
`empty_support` no-op and emits no metadata, callback, history event or
cooldown update.

Once the length guard passes, both selected lag entries must be mappings with
support exactly equal to the live node support, and every selected EPI must lie
in the real-scalar chart. There is no fallback from a missing historical node
to its present EPI. The local and global labels denote two temporal lags for
each node; the global-lag snapshot is not a network mean.

The public committed effect is

$$x_{\rm new}=\operatorname{structural\_clip}(x_{\rm raw}).$$

Every immutable node proposal retains both raw and bounded EPI and records
whether clipping intervened. Hard and soft clipping can therefore be inspected
without treating either as part of the raw affine recurrence. The operation
reads the history but does not append or shift it; advancement belongs to the
surrounding runtime.

The executor validates the complete proposal before writing and wraps EPI,
topology, graph metadata, history, caches and capturable graph-owned callback
state in one rollback boundary. A propagated commit, telemetry or strict
callback failure restores that graph state. An external effect already emitted
by a callback cannot be undone by a graph snapshot.

### §1.3 Mean and disagreement statements

If all three scalar inputs are the same constant and clipping leaves that
constant in range, the value is fixed:

$$x_0=x_l=x_g=c \quad\Longrightarrow\quad x_{\rm new}=c.$$

For network vectors and any declared positive diagonal weights h, the exact-real
raw weighted mean obeys

$$
\mu_h(x_{\rm raw})
=\beta\mu_h(x_0)+\gamma\mu_h(x_l)+\delta\mu_h(x_g).
$$

This is a mean-combination identity, not automatic preservation of the current
mean. Let V_h be weighted disagreement from that mean. Convexity gives

$$
V_h(x_{\rm raw})
\leq
\beta V_h(x_0)+\gamma V_h(x_l)+\delta V_h(x_g).
$$

The right-hand side contains all three snapshots. If delayed histories remain
free, it supplies no finite multiplicative gain relative to V_h(x_0) alone.

For a one-step map with the delayed vectors fixed, write
b = gamma x_l + delta x_g. The raw map of x_0 preserves the consensus subspace
exactly when b is uniform. Under that condition its disagreement gain is at
most beta squared. If b is nonuniform, a consensus x_0 can be sent to positive
disagreement and no finite global multiplicative gain exists. Hard clipping
after a uniform-offset raw map is nonexpansive and retains the beta-squared
bound in the exact-real scalar model. The soft knee is globally 4/3-Lipschitz,
giving the sufficient bound (4 beta / 3) squared. A degenerate clipping interval
is a constant map with zero gain.

The returned opt-in evidence keeps these universal fixed-history bounds apart
from observed pre/post mean and disagreement values. It also reports binary64
rounding and clipping interventions separately. Exact rational diagnostics
outside the finite binary64 reporting range produce an explicit domain error
before the graph commit rather than an infinite or partial result. None of
these one-step facts alone proves stability of the history-advance recurrence,
pressure closure, structural-charge conservation, or U2 convergence. Section
2.4 derives a separate exact result only after the finite history shift, fixed
coefficients and fixed metric are declared explicitly.

---

## §2. Three Operators That Must Be Distinguished

### §2.1 Runtime map $M_{G,h}$

The runtime map acts on a graph and a stored history, applies the guard and
clipping, and uses configuration-dependent parameters. It is the canonical
engine operation.

### §2.2 Companion history advance $T$

A mathematical recurrence can insert the new value at the head of a history
vector and shift all prior entries. This companion-style map is useful for
studying an isolated recurrence, but `apply_network_remesh` does not perform
that shift itself. Section 2.4 analyzes its finite, uniform, unclipped form
without replacing it by a convolution.

### §2.3 Finite cyclic filter $F$

On $\mathbb C^n$, let $S$ be the unitary cyclic shift and define

$$F=\beta I+\gamma S^{\tau_l}+\delta S^{\tau_g}.$$

This is a convolution filter on a periodic sample window. It is the model used
by the corrected Fourier projector in
`src/tnfr/riemann/remesh_infinity_residue_split.py`.

The spectrum of $F$ does not describe the companion map $T$ merely because
both contain the same coefficients. The historical N15 derivation conflated
these two constructions.

### §2.4 Exact finite companion-history stability

Let $m$ be the largest delay with a positive coefficient and define the
newest-first augmented history

$$X_k=(x_k,x_{k-1},\ldots,x_{k-m}).$$

After combining coincident delays, write $c_0=\beta$,
$c_{\tau_l}\mathrel{+}=\gamma$ and
$c_{\tau_g}\mathrel{+}=\delta$. The recurrence and shift are

$$
x_{k+1}=\sum_{r=0}^{m}c_r x_{k-r},
\qquad
X_{k+1}=P X_k,
$$

where the first row of $P$ is $(c_0,\ldots,c_m)$ and its remaining rows shift
the history. The coefficients are nonnegative and sum to one, so $P$ is row
stochastic. Its exact stationary distribution is

$$
D=1+\gamma\tau_l+\delta\tau_g,
\qquad
\pi_0=D^{-1},
\qquad
\pi_j=\frac{\gamma\mathbf 1_{j\leq\tau_l}
                 +\delta\mathbf 1_{j\leq\tau_g}}{D}
\quad(1\leq j\leq m).
$$

Fix a positive diagonal spatial metric $H$ and let $Q_H$ remove the
$H$-weighted spatial mean. With
$E_H(x)=\tfrac12\lVert Q_Hx\rVert_H^2$, define

$$V(X_k)=\sum_{j=0}^{m}\pi_j E_H(x_{k-j}).$$

Stationarity of $\pi$ and the quadratic Jensen identity give the exact balance

$$
V(X_k)-V(X_{k+1})
=\frac{\pi_0}{2}
  \sum_{a<b}c_a c_b
  \lVert Q_Hx_{k-a}-Q_Hx_{k-b}\rVert_H^2
\geq0.
$$

Equality holds exactly when all active centered input fields agree pairwise.
This is an augmented temporal disagreement functional. It does not control
differences between spatially uniform history rows and therefore is not a
strict Lyapunov function for the complete augmented state.

For $0<\alpha<1$, $c_0=\beta>0$ and the maximum-delay coefficient is positive.
The finite companion is irreducible and aperiodic, hence primitive. It follows
independently of strict decrease of $V$ that every coordinate converges to the
preserved stationary history barycenter $\sum_j\pi_jx_{k-j}$. This is temporal
pointwise convergence; it does not force different nodes to share one value.

At $\alpha=0$, $m=0$ and the map is the identity. At $\alpha=1$, only the
global-delay coefficient remains: $P$ is a cyclic permutation of order
$\tau_g+1$, $V$ is conserved and an individual orbit can have any period
dividing that order. The alternating-history witness is therefore retained.

[`certify_uniform_remesh_history_stability`](../src/tnfr/physics/remesh_history_stability.py)
materializes the exact companion and stationary measure.
[`observe_uniform_remesh_history_transition`](../src/tnfr/physics/remesh_history_stability.py)
checks one rational transition, its barycenter and the dissipation identity.
Both APIs exclude binary64 runtime identification, clipping, changing
coefficients, metric or support, and schedule/REMESH gain composition.

### §2.5 One executed binary64 transition

[`observe_runtime_remesh_history_bridge`](../src/tnfr/physics/runtime_remesh_history_stability.py)
closes the first runtime boundary for one applied, executor-sealed
`EventRemeshCycleResult`. It reverses exactly the retained oldest-first runtime
window required by the newest-first companion, rebuilds the exact theorem from
the represented `alpha` and delays, and replays both the nested binary64 affine
formula and the canonical clipping call bit for bit.

Let `y` be the exact companion head, `b` the rational value represented by the
binary64 raw result, and `z` the rational value represented by the committed
bounded result. The bridge retains the signed decomposition

$$
b=y+r_{\rm round},\qquad
z=b+r_{\rm clip},\qquad
z=y+r_{\rm round}+r_{\rm clip}.
$$

Only the new history head changes relative to the ideal companion step. If
$\pi_0$ is the first stationary temporal weight, the exact lifted balances are

$$
\begin{aligned}
V(X)-V(b,x_0,\ldots,x_{m-1})
 &=D_J-\pi_0\bigl(E_H(b)-E_H(y)\bigr),\\
V(X)-V(z,x_0,\ldots,x_{m-1})
 &=D_J-\pi_0\bigl(E_H(z)-E_H(y)\bigr).
\end{aligned}
$$

For a signed error $e$, the bridge also evaluates the exact a posteriori bound

$$
\left|E_H(y+e)-E_H(y)\right|
\leq
\sum_i h_i\left|(Q_Hy)_i(Q_He)_i\right|+E_H(e).
$$

Multiplying this bound by $\pi_0$ gives a sufficient one-step Lyapunov margin.
The signed defect remains separate because it can establish decrease when the
absolute bound is conservative. Rounding and clipping are staged: the clipping
defect is based at `b`, so the cross term between the two residuals is not lost.

Hard clipping on one common scalar interval is nonexpansive for every positive
diagonal spatial metric. This follows from the pairwise identity

$$
E_H(x)=\frac{1}{2\sum_i h_i}
       \sum_{i<j}h_i h_j(x_i-x_j)^2
$$

and the 1-Lipschitz property of the scalar clamp. Soft clipping has no such
nonincrease theorem: points in its Hermite knee can expand pairwise separation
by almost `4/3`, and disagreement by almost `16/9`. The runtime bridge therefore
records its exact observed defect rather than promoting the analytic ideal-map
Lipschitz bound to the binary64 evaluator.

The result is deliberately a **lifted** companion observation. The current
runtime appends the pre-REMESH state and does not immediately insert `z` into
`_epi_hist`. Consequently, this bridge alone proves neither live history
advance nor repeated runtime stability. It also records the exact stationary
history-barycenter drift $\pi_0r_{\rm round}$ or
$\pi_0(r_{\rm round}+r_{\rm clip})$; disagreement nonincrease does not imply
preservation of that barycenter.

Exact diagnostics used by delayed-REMESH evidence now reject a nonzero rational
that would underflow to displayed binary64 zero. This prevents a real rounding
residual from being published as `0.0`.

### §2.6 Exact REMESH-head/schedule-head balance

The next algebraic step inserts a scheduled head `s` after the ideal, raw and
bounded REMESH heads `y`, `b` and `z`. In one fixed positive diagonal metric,
define the stationary-weighted defects

$$
\varepsilon_r=\pi_0(E_H(b)-E_H(y)),\qquad
\varepsilon_c=\pi_0(E_H(z)-E_H(b)),\qquad
\varepsilon_s=\pi_0(E_H(s)-E_H(z)).
$$

Replacing only the newest history row gives the exact telescoping identity

$$
V(X)-V(s,x_0,\ldots,x_{m-1})
=D_J-\varepsilon_r-\varepsilon_c-\varepsilon_s.
$$

Suppose a separately established represented schedule bound satisfies
`E_H(s) <= q E_H(z)` with `q >= 0`. Its nonnegative slack is
`sigma = q E_H(z) - E_H(s)`, and therefore

$$
\begin{aligned}
L
 &=D_J-\varepsilon_r-\varepsilon_c
   +\pi_0(1-q)E_H(z),\\
V(X)-V(s,x_0,\ldots,x_{m-1})
 &=L+\pi_0\sigma \geq L.
\end{aligned}
$$

Thus `L >= 0` is a sufficient finite-step nonincrease condition. A schedule
gain `q <= 1` is not sufficient by itself when binary64 rounding or a
disagreement-expanding clipping map consumes the Jensen margin. Conversely,
`q > 1` need not force increase when the ideal Jensen dissipation pays for the
scheduled expansion. The stationary-history barycenter changes exactly by
`pi[0] * (s - y)`.

[`observe_remesh_schedule_history_transition`](../src/tnfr/physics/remesh_schedule_stability.py)
implements this identity as a sealed exact-rational observation. Its inputs
are caller-supplied algebraic heads and a declared nonnegative gain bound that
is verified against those heads; the result
does not identify them with one executor trace, prove repetition or constrain
future schedules. The stricter recorded-artifact binding is handled in §2.7.

### §2.7 Adjacent recorded-cycle binding

[`observe_runtime_remesh_schedule_sequence`](../src/tnfr/physics/runtime_remesh_schedule_stability.py)
binds the preceding pure identity to an
`ObservedEventRemeshCycleSequence`. For every adjacent supplied pair it starts
from the applied REMESH bridge of cycle `i`, then requires the represented EPI
composition inside cycle `i+1` to start at the bounded REMESH head `z_i`, end
at its exact pre-REMESH head `s_i`, and expose a gain in the sequence's common
normalized metric.

The decisive runtime-history check is

$$
X_{i+1}=(s_i,x_i,\ldots,x_{i-m+1}),
$$

where the right side must equal the newest-first window reconstructed from the
next cycle's sealed outgoing `_epi_hist`. Equal REMESH configuration fixes the
same coefficients, delay width and clipping policy across every paired row.
This certifies the recorded append identity between the supplied artifacts,
which the isolated runtime bridge could only lift hypothetically.

For `r` adjacent boundaries in the same metric, exact history equality makes
the intermediate augmented energies cancel:

$$
\sum_{i=0}^{r-1}\bigl(V(X_i)-V(X_{i+1})\bigr)
=V(X_0)-V(X_r).
$$

Each summand retains its own Jensen, raw, clipping and schedule defects. The
finite total drop is also the sum of the gain-based lower bounds and the
nonnegative augmented schedule-gain slacks $\pi_0\sigma_i$. This is an additive
balance; no fixed-history REMESH gain is multiplied across cycles.

The adapter identifies exact represented values in the supplied sealed cycle
records. The underlying sequence remains a caller-ordered observation of
individually atomic executions: it does not prove that the records came from
one graph in causal succession, make the calls jointly atomic, identify a
global binary64 map, or establish repeated or future stability.

### §2.8 Effective-P2 three-mesh reference family

The generic three-mesh observer becomes runtime-linked on one deliberately
small domain. Its diffusion component is exactly the two-node,
homogeneous-capacity specialization of the
[reversible single-eigenmode Euler theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md#exact-reversible-single-eigenmode-euler-reference-theorem),
where the complete exponential, Euler, norm-bound, subdivision and conditional
exact-real convergence proofs are centralized.

For the effective path $P_2$, fixed positive conductance and homogeneous
capacity $\nu>0$ give

$$
L_{\rm rw}=\begin{pmatrix}1&-1\\-1&1\end{pmatrix},
\qquad u=(1,-1)^\mathsf T,
\qquad A u=\lambda u,\quad \lambda=2\nu.
$$

Thus a nonuniform initial field has the form

$$
x_0=m\mathbf 1+d u,\qquad d\ne0,
$$

and the general theorem supplies

$$
x(T)=m\mathbf 1+d e^{-\lambda T}u,
\qquad
x_h(T)=m\mathbf 1+d p_h u,
\qquad
p_h=\prod_j(1-\lambda h_j).
$$

The adapter requires each exact represented partition to have positive
durations summing to the same $T$, every segment to expose an intact
pressure-refreshed exact-affine Euler identification, and
$0<\lambda h_j<1$. Its inherited finite bounds are

$$
0\le e^{-\lambda T}-p_h
\le \frac{\lambda^2}{2}\sum_j h_j^2
\le \frac{\lambda^2}{2}T h_{\max},
\qquad
\lVert x(T)-x_h(T)\rVert_\infty
=|d|\bigl(e^{-\lambda T}-p_h\bigr).
$$

The implementation stores rational lower and upper enclosures of both errors
and requires two successive proper positive subdivisions. It submits the
three executor-owned physical partitions to the general runtime eigenmode
observer once, then strengthens that result by requiring exact-affine
identification, zero pressure-realization and held-input residuals, and zero
local and endpoint defects in every row. The strict finite
improvements follow from the cited general theorem. Its rational exponential
routine restricts $\lambda T\le4096$ solely to cap the integer-power exponent
used by the enclosure. That restriction is not a dynamical threshold and does
not bound the total bit size of arbitrary rational inputs. Although the pure
theorem proves convergence for fixed-data admissible exact-real partition
families with $h_{\max}\to0$, this runtime adapter observes only three finite
meshes and does not promote binary64 asymptotic convergence or solver order.

The REMESH part fixes $\tau_l=\tau_g=1$ and requires the delayed row to be the
exact initial field $x_0$. With

$$
\beta=(1-\alpha)^2,
$$

the ideal Euler-driven and continuous-reference REMESH heads are

$$
y_h=\beta x_h(T)+(1-\beta)x_0,
\qquad
y_*=\beta x(T)+(1-\beta)x_0.
$$

Their error therefore scales exactly as

$$
\lVert y_*-y_h\rVert_\infty
=\beta |d|\bigl(e^{-\lambda T}-p_h\bigr).
$$

The strict subdivision improvement survives this ideal REMESH step when
$\beta>0$; at $\alpha=1$, $\beta=0$ and the ideal error is already zero. For
the committed binary64 head, the executor-linked bridge supplies

$$
z_h=y_h+r_{\rm round}+r_{\rm clip},
$$

and hence the certified finite runtime bound is

$$
\lVert z_h-y_*\rVert_\infty
\le
\beta |d|\bigl(U_{\exp}-p_h\bigr)
+\lVert r_{\rm round}+r_{\rm clip}\rVert_\infty,
$$

where $U_{\exp}$ is the rational upper enclosure of $e^{-\lambda T}$. The
runtime family requires hard clipping on one common nonempty scalar interval;
soft clipping is rejected.

[`observe_p2_event_remesh_reference_family`](../src/tnfr/physics/event_remesh_reference.py)
constructs the sealed family from three already executed cycles. Its mesh and
family records preserve every exact premise, factor, enclosure, ideal scaling
and runtime residual. It certifies neither solver order nor generic mesh or
binary64 asymptotic convergence, arbitrary glyphs or mixed channels, changing
support or metric, repetition, or future behavior. The executable
[`166_event_remesh_reference_family.py`](../examples/02_physics_regimes/166_event_remesh_reference_family.py)
instantiates the scope with two, four and eight equal segments.

The separate pure
[`167_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py)
exercises both exact nonuniform modes of nonregular `P3` without claiming a
runtime or REMESH binding.

### §2.9 Causal finite cycle-sequence execution

The algebraic telescope in §2.7 becomes causally identified for one finite
invocation through
[`execute_event_remesh_cycle_sequence`](../src/tnfr/operators/event_remesh_causal_runtime.py).
It accepts ordered `EventRemeshCycleExecutionSpec` values and executes every
cycle on one graph inside one outer graph transaction. Each sealed
`CausalEventRemeshCycleReceipt` retains its zero-based ordinal, exact submitted
spec and `EventRemeshCycleResult`; the executed schedule is the schedule object
carried by that spec. The sealed `ExecutedEventRemeshCycleSequence` always
retains the receipts and the ordinary offline
`ObservedEventRemeshCycleSequence`. By default,
`require_runtime_telescope=True` also requires and retains the compatible
`RuntimeRemeshScheduleSequenceObservation`. The explicit `False` branch always
stores `runtime_telescope=None`, thereby admitting a valid word without one
common affine schedule metric; telescope-specific properties then remain false.

Both branches of the outer wrapper prove same-invocation causal order, one graph identity and
finite graph-owned atomicity. It does not alter the contracts of the nested
offline observers when they are used separately. The finite additive energy
identity is still not a global schedule-times-REMESH gain and has no uniform
positive normalized block margin over a forward-invariant runtime class or
runtime intrablock prefix-amplification bound. Solver accuracy/order, mesh convergence, repeated
or future stability and rollback of emitted I/O, warnings, external resources
or external-only aliases remain outside the result. In particular, the lag-one
`alpha=1` alternating history remains a valid causal counterexample to
inferring convergence from provenance alone.

### §2.10 Exact normalized margin on one causal finite block

Let $B=(j_0,\ldots,j_0+r-1)$ be a nonempty contiguous block of boundaries in
one intact `ExecutedEventRemeshCycleSequence`. Every boundary already satisfies

$$
D_j=V_j-V_{j+1}=K_j+S_j,
\qquad S_j\geq0,
$$

where $K_j$ is its gain-based lower bound and $S_j$ is its exact augmented
schedule-gain slack. Exact intermediate-energy continuity gives

$$
D_B=V_{j_0}-V_{j_0+r}
=K_B+S_B,
\quad
K_B=\sum_{j\in B}K_j,
\quad
S_B=\sum_{j\in B}S_j.
$$

When $V_{j_0}>0$, define the block-specific normalized lower margin

$$
\kappa_B=\frac{K_B}{V_{j_0}}.
$$

Then the observed endpoint obeys the exact bound

$$
\frac{V_{j_0+r}}{V_{j_0}}\leq1-\kappa_B.
$$

[`observe_executed_event_remesh_block_margin`](../src/tnfr/physics/runtime_remesh_schedule_block_margin.py)
revalidates the causal source, runtime telescope and selected boundary objects
by identity, and seals every term above. Its normalized fields are undefined
when $V_{j_0}=0$. The executable
[`170_runtime_remesh_block_margin.py`](../examples/02_physics_regimes/170_runtime_remesh_block_margin.py)
has one finite block with $\kappa_B=139/256$ and endpoint gain upper bound
$117/256$. Its separate lag-one `alpha=1` orbit has $\kappa_B=0$, so causal
provenance alone supplies no positive margin.

There cannot be a positive uniform *absolute* drop over a class containing
equilibrium or amplitude-scaled copies approaching it: this disagreement
energy and every term in the balance scale quadratically, while equilibrium
has zero drop. A repeated result would instead need a declared forward-invariant
class with one uniform $\kappa_*>0$ for every positive-energy block, together
with a finite constant bounding energy amplification at each intrablock prefix.
The finite observer proves neither condition.

### §2.11 Uniform exact REMESH/schedule policy stability

The missing uniform margin and prefix bound can be proved for a restricted
exact-model class. Fix the REMESH companion $P$, its positive stationary row
vector $\pi$, the ordered spatial support and one positive spatial metric $H$.
Let every schedule map $S_k$ preserve the spatial-consensus subspace and obey
the same disagreement-energy estimate

$$
E_H(S_kx)\leq qE_H(x),
\qquad 0\leq q\leq1.
$$

The maps may vary with $k$. The hypotheses, fixed metric and common $q$ must
hold for every member of the declared family. Index the augmented history
immediately after each schedule. The next operation mixes those stored heads by
REMESH, after which the next schedule acts on the new mixture. If

$$
e_k=(E_H(x_k),E_H(x_{k-1}),\ldots,E_H(x_{k-m}))^\mathsf T,
$$

Jensen's inequality and the schedule bound give the componentwise envelope

$$
e_{k+1}\leq B_qe_k,
\qquad
B_q=D_qP,
\qquad
D_q=\operatorname{diag}(q,1,\ldots,1).
$$

The order $D_qP$ is essential: the schedule contracts the newly mixed head.
Since $0\leq B_q\leq P$ entrywise and both matrices are nonnegative,
$B_q^s\leq P^s$ for every $s\geq0$. With

$$
V_k=\pi^\mathsf Te_k,
\qquad
\pi^\mathsf TP=\pi^\mathsf T,
$$

every finite prefix is nonexpansive:

$$V_{k+s}\leq V_k.$$

Set $L=m+1$. A length-$L$ path through a companion matrix cannot remain in the
shift rows: every such path visits the head at least once. Equivalently, the
head-avoiding matrix $B_0=D_0P$ satisfies $B_0^L=0$. Every monomial contributing
to $B_q^L$ therefore contains at least one factor $q$, so

$$
B_q^L\leq qP^L,
\qquad
\pi^\mathsf TB_q^L\leq q\pi^\mathsf T,
\qquad
V_{k+L}\leq qV_k.
$$

Repeated blocks and their remaining prefix yield

$$
V_{k+n}\leq q^{\lfloor n/L\rfloor}V_k.
$$

Thus this exact class has intrablock amplification bound one and uniform
normalized block-margin lower bound $1-q$. If $q<1$, the augmented spatial
disagreement converges geometrically to zero. Positivity of every $\pi_j$ then
forces the spatial disagreement of every retained history row to vanish. This
does not control differences among their spatially uniform temporal means.

The horizon $L$ is a sufficient universal path bound. It is sharp for the
worst-case pure-delay companion, but it need not be the smallest contractive
horizon for a particular interior coefficient choice. The endpoint cases are
also exact:

- at $\alpha=0$, $L=1$ and the bound is the ordinary schedule gain $q$;
- at $\alpha=1$ and $q<1$, each spatial-disagreement component is damped once
  per pure-delay circuit, so the former periodic disagreement obstruction is
  removed;
- at $q=1$, the margin supplied by this theorem is zero. Identity schedules
  and a nonzero constant centered history retain the earlier counterexample,
  but the loose bound $q=1$ alone does not prove that another concrete family
  fails to converge.

[`certify_uniform_remesh_schedule_policy_stability`](../src/tnfr/physics/remesh_schedule_policy_stability.py)
materializes $P$, $B_q$, the head-avoiding matrix and every rational power
needed at the $L$-cycle block boundary, then rechecks the one-step induction
hypotheses and all block and stationary-weight inequalities in its sealed
result. Intermediate prefix powers are not retained. The API receives no
schedule map, metric or runtime record;
the common-gain and consensus-preservation statements are explicit hypotheses.
It is therefore a repeated theorem for the declared exact family, not a
promotion of the causal binary64 observer in §2.10. Such a promotion still
needs a forward-invariant runtime class, fixed support and metric, and a
derived uniform centered relative defect bound. Section §2.12 shows how such a
bound enters the exact margin; absolute additive defect bounds can at most
imply convergence to a neighborhood.

### §2.12 Relative signed-defect envelope and finite causal verification

The common-$q$ theorem can absorb a bounded discrepancy between the ideal
REMESH head and the bounded head actually presented to the next schedule.
For one transition, write

$$
J_k=\sum_d c_d E_H(x_{k-d}),
\qquad
y_k=\sum_d c_d x_{k-d},
$$

where the nonnegative active coefficients sum to one. Let $z_k$ be the
bounded binary64 head represented as an exact rational vector. The relevant
quantity is the signed centered-energy defect

$$
\delta_k=E_H(z_k)-E_H(y_k).
$$

It is not $E_H(z_k-y_k)$ and it is not a norm of the state residual: the
energy cross term prevents either substitution. Assume one finite
$\eta\geq0$ satisfies

$$
\delta_k\leq\eta J_k
$$

for every transition in the declared fixed-support, fixed-metric family.
Jensen gives $E_H(y_k)\leq J_k$, hence

$$
E_H(z_k)\leq(1+\eta)J_k.
$$

If the following consensus-preserving schedule has disagreement-energy gain
at most $q$, including on $z_k$, then

$$
E_H(S_k z_k)\leq q(1+\eta)J_k
  =q_{\mathrm{eff}}J_k,
\qquad
q_{\mathrm{eff}}=q(1+\eta).
$$

Therefore the proof in §2.11 applies verbatim with $q_{\mathrm{eff}}$ in the
head row:

$$
B_{q_{\mathrm{eff}}}
  =\operatorname{diag}(q_{\mathrm{eff}},1,\ldots,1)P,
\qquad
V_{k+n}\leq
q_{\mathrm{eff}}^{\lfloor n/L\rfloor}V_k.
$$

The exact certificate rejects $q_{\mathrm{eff}}>1$. At
$q_{\mathrm{eff}}=1$ it proves prefix and block nonexpansion with zero
margin. At $q_{\mathrm{eff}}<1$ it proves normalized block-margin lower bound
$1-q_{\mathrm{eff}}$ and geometric decay of spatial disagreement in every
retained history row. If $q=0$, every finite $\eta$ gives
$q_{\mathrm{eff}}=0$. These conclusions still leave spatially uniform
temporal means uncontrolled.

The normalization by $J_k$ is essential. Opposing active centered fields can
cancel in $y_k$, making $E_H(y_k)=0$ while $J_k>0$; a ratio against the ideal
energy would then be singular even though a finite robust budget exists. If
$J_k=0$, Jensen forces $E_H(y_k)=0$, and the declared inequality directly
requires $E_H(z_k)=0$. Implementations must check that algebraic condition and
must not divide by zero.

[`certify_uniform_remesh_schedule_relative_defect_stability`](../src/tnfr/physics/remesh_schedule_relative_defect_stability.py)
materializes this conditional theorem by reusing the sealed policy envelope,
rather than maintaining a second matrix-power implementation. It keeps the
schedule-only $q$ separate from $q_{\mathrm{eff}}$ and exact-rationalizes the
caller-supplied $\eta$.

[`observe_executed_event_remesh_relative_defect_block`](../src/tnfr/physics/runtime_remesh_schedule_relative_defect.py)
provides the finite causal bridge. For every selected adjacent boundary of one
intact `ExecutedEventRemeshCycleSequence` that retained the compatible runtime
telescope, it reconstructs $J_k$, $E_H(y_k)$,
$E_H(z_k)$ and $\delta_k$; checks $\delta_k\leq\eta J_k$ and the represented
schedule gain $q_k\leq q$; and verifies the complete post-schedule
history-energy vector against $B_{q_{\mathrm{eff}}}$ times its input vector.
For $N$ selected boundaries it checks the finite endpoint factor
$q_{\mathrm{eff}}^{\lfloor N/L\rfloor}$. A prefix shorter than $L$ therefore
retains factor one.

This observer establishes the inequalities only on the recorded causal block.
It does not show that future heads remain in a class satisfying the same
$\eta$, or that support, metric, REMESH coefficients and schedule policy stay
fixed. A repeated binary64 theorem still requires that forward-invariance
argument. Solver accuracy and order, adaptive grammar, and full multichannel
TNFR stability also remain outside scope.

### §2.13 Pairwise binary64 boundary and the $\alpha=1$ class

An EPI interval and hard clipping do not by themselves provide a useful
uniform $\eta$. This follows from an exact scalar reduction. For the temporal
coefficients $c=(\beta,\gamma,\delta)$, let $a$ and $b$ contain the current,
local-delay and global-delay values at two spatial coordinates, and define

$$
D_c(a,b)=\sum_d c_d(a_d-b_d)^2.
$$

Fix the represented $\alpha$, the production evaluation order and one common
hard clamp. Let $g$ be the exact-rational convex combination and let
$\widehat g$ be that actual nested binary64 evaluation followed by the clamp.
For a finite binary64 alphabet
$K=[\mathrm{EPI}_{\min},\mathrm{EPI}_{\max}]\cap\mathbb F_{64}$ and any finite
set $A\subseteq K^3$ of admissible nodewise temporal tuples, including the
equality of local and global entries when the two delays coincide, define

$$
\eta_A^\star=
\max_{a,b\in A,\,D_c(a,b)>0}
\frac{\left[(\widehat g(a)-\widehat g(b))^2
 -(g(a)-g(b))^2\right]_+}{D_c(a,b)}.
$$

The weighted centered energy has the pairwise identity

$$
E_H(v)=\frac{1}{2\sum_i h_i}
       \sum_{i<j}h_i h_j(v_i-v_j)^2.
$$

Applying the scalar inequality to every pair proves
$E_H(z)-E_H(y)\leq\eta_A^\star J$ for every finite support and positive
diagonal metric whose nodewise tuples belong to $A$. The constant is optimal
for that tuple class because a two-node support realizes each admissible pair.
If no pair has positive $D_c$, set the maximum to zero. A common hard clamp
cannot increase the pairwise difference. If $D_c=0$, the two input tuples agree
on every active coefficient; the deterministic evaluator gives equal outputs
as numeric values, although an IEEE signed-zero bit may differ, and the branch
closes without division. This characterizes a finite tuple class exactly, but
it does not imply that its maximum is small enough for
$q(1+\eta_A^\star)<1$.

The obstruction already occurs with normal values and inactive clipping. Set
$\alpha=1/2$, use two equally weighted nodes, and take

$$
\begin{aligned}
x_{0,1}=x_{0,2}&=2^{-52},\\
x_{g,1}=x_{g,2}&=1,\\
x_{l,1}&=2^{-105},\\
x_{l,2}&=2^{-105}+2^{-157}.
\end{aligned}
$$

All inputs lie in $[0,1]$. The exact head separation is $2^{-159}$, whereas
the nested runtime evaluation crosses two rounding-cell boundaries and returns
a separation $2^{-53}$. With the repository energy convention,

$$
J=2^{-318},\qquad E_H(y)=2^{-320},\qquad E_H(z)=2^{-108},
$$

so this one witness requires

$$
\eta\geq 2^{210}-\frac14.
$$

Consequently, strict contraction would require
$q<4/(2^{212}+3)\approx2^{-210}$; the finite witness policy $q=9/16$ cannot
be promoted over the whole bounded binary64 box. The cause is structural:
$J$ removes spatially uniform modes, while floating-point evaluation is not
translation equivariant. Large common temporal components can therefore move
nearby coordinates across different rounding cells. Replacing $J$ by the
centered augmented energy $V$ does not see those common components either.

There is one immediate positive class. Fix a nonempty ordered support, one
positive spatial metric, two positive delays, and a sufficient represented
history of at least $\max(\tau_l,\tau_g)+1$ rows. At $\alpha=1$, the exact
recurrence is the pure global delay and the nested runtime expression returns
the global delayed value numerically. If every retained value is finite and
lies inside one fixed hard-clipping interval, the clamp is the identity, hence

$$
E_H(z)=E_H(y),\qquad \eta=0,\qquad q_{\mathrm{eff}}=q.
$$

This REMESH-only step preserves that interval and the fixed support. IEEE signed
zero may be canonicalized by the zero multiplications and addition, so the
claim is equality as a real/binary64 numeric value and in exact-rational EPI,
not preservation of the zero sign bit. This REMESH-only class supplies no
schedule. Section §2.14 supplies one restricted global P2 numeric EPI-kernel
family that preserves the interval and has common gain $q=0$. Promoting that
composition to repeated complete runtime execution still requires real-graph,
stage, grammar and event provenance plus preservation of support, metric and
configuration. A gain measured on a finite trace is insufficient.

[`observe_binary64_remesh_pair_relative_defect`](../src/tnfr/physics/binary64_remesh_relative_defect.py)
replays the shared production scalar kernel and records $D_c$, both exact
squared separations, the signed rounding and clipping defects, and the local
$\eta$ witness without promoting it to a uniform class bound.
[`certify_alpha_one_hard_clip_remesh_class`](../src/tnfr/physics/binary64_remesh_relative_defect.py)
seals the fixed-support, fixed-metric, bounded-history REMESH-only class above.
It checks represented-history membership and explicitly leaves schedule-family,
repeated-execution and future-runtime claims false.

For $0<\alpha<1$, viable later branches include a quantization-aware invariant
alphabet, a centered or antisymmetric representation with explicit
normal-range guards, or an augmented defect budget. If only the lifted positive
defect obeys $U_k\leq\rho V_k$, the correct envelope is a rank-one update of
the companion matrix and must be analyzed with its induced block factors; it
cannot be replaced by the scalar $q(1+\eta)$ formula.

### §2.14 A global binary64 $P_2$ half-Reception kernel family

The first restricted global numeric EPI-kernel family closes on the smallest
connected support.
Fix two ordered nodes with mutual singleton neighbor sets, the exact binary64
factor `EN_mix=0.5`, one immutable all-target Jacobi snapshot and one common
hard-clipping interval. For an input pair $(x_1,x_2)$, the shared Reception
kernel evaluates

$$
\begin{aligned}
\widehat S_1(x)&=
 \operatorname{clip}\!\left(\operatorname{fl}
 (\operatorname{fl}(x_1/2)+\operatorname{fl}(x_2/2))\right),\\
\widehat S_2(x)&=
 \operatorname{clip}\!\left(\operatorname{fl}
 (\operatorname{fl}(x_2/2)+\operatorname{fl}(x_1/2))\right).
\end{aligned}
$$

A singleton `fmean` reproduces its nonzero operand; for zero it may normalize
the sign without changing the numeric value. Multiplication by one half leaves
both terms finite, IEEE addition is commutative, and the two raw outputs are
therefore numerically equal. A common hard clamp preserves that equality and
returns a finite value in the interval. Thus, for every positive diagonal
spatial metric,

$$
E_H(\widehat S(x))=0,
\qquad q=0.
$$

This is a global statement over every finite represented pair in the declared
interval; it is not a gain fitted to a finite trace. It remains true when
underflow makes the numeric consensus value differ from the exact-real average.
Consequently the binary64 kernel is not promoted to a global affine map.
Signed-zero bits are also outside the theorem.

Compose this numeric EPI kernel with the $\alpha=1$ class of §2.13. REMESH has
$\eta=0$, so $q_{\mathrm{eff}}=0$. Each cycle inserts one consensus row into
the post-schedule history. Since the only active delay is $\tau_g$, the common
policy theorem has

$$
L=\tau_g+1,
\qquad
V_{k+n}=0\quad\text{for every }n\geq L.
$$

This proves arbitrary finite repetition and exact spatial-disagreement
extinction for the restricted pair of numeric EPI kernels. It does not certify
the complete Reception stage: operator preconditions, grammar replacement,
semantic EPI-kind and history writes, callbacks, graph transactions and event
ownership still require a causal execution binding. It also says nothing about
solver accuracy, temporal consensus values or full TNFR stability.

The selected half mix is essential to this interval-independent commutative
argument. Narrow or degenerate intervals may admit other factors. The canonical
default represented factor is
`0x1.ee7eea04ddca0p-3`. For the adjacent normal values

```text
x1 = 0x1.d2775ff0a1eccp-1021
x2 = 0x1.d2775ff0a1ecdp-1021,
```

the two default-factor outputs are respectively
`0x1.d2775ff0a1eccp-1021` and `0x1.d2775ff0a1ecep-1021`. Their separation is
twice the input separation, so the centered-energy gain is exactly four.
Any broad interval containing this inactive-clipping witness therefore admits
no global default-factor bound $q<1$.

[`certify_p2_half_reception_remesh_stability`](../src/tnfr/physics/binary64_p2_reception_stability.py)
seals the fixed-support, fixed-metric composition by reusing the $\alpha=1$
REMESH class, the common-$q$ theorem and its relative-defect extension. Its
pair evaluator calls the shared production mean, blend and clipping kernels.
The result exposes the restricted global and finite-repetition claims while
keeping complete-stage, grammar, live-execution and solver properties false.

### §2.15 Finite causal binding of one executed $P_2$ Reception stage

The abstract kernel can now be identified with one completed executor event.
[`certify_executed_p2_half_reception_stage`](../src/tnfr/physics/runtime_p2_reception_stage.py)
accepts one intact kernel certificate and one intact
`OperatorEventExecutionResult`. It selects an executor-owned EN stage and
requires same-invocation event identity, zero duration, exact two-phase Jacobi
scheduling, two captured EPI endpoints and a deeply revalidated one-step
neighbor-stage certificate. The observed targets must equal the ordered P2
support; runtime neighbor sets must be the two mutual singletons; the binary64
mix must be exactly one half; and the hard interval, fixed effective
conductance, capacity and normalized diffusion-metric ray must match the source
class.

The adapter then replays both coordinates through the shared production mean,
blend and clipping kernel and requires bit identity with the accepted stage
state and captured right endpoint. This proves for that completed EPI jump

$$
E_H(x^+)=0,
\qquad q=0,
$$

including underflow cases where the generic represented-affine stage bridge
correctly abstains. Opposite signed-zero outputs remain valid numeric
consensus. The result carries the finite grammar admission and graph-owned
schedule atomicity of its source execution.

This binding stops at one observed EPI stage. By itself it does not identify the source
$\alpha=1$ REMESH history or configuration with that graph, certify all
Reception auxiliary state, preserve raw topology or current live-graph
identity, or prove repeated/future runtime stability, solver accuracy or full
TNFR stability.

### §2.16 Finite causal binding of executed $P_2$ Reception/REMESH cycles

[`certify_executed_p2_half_reception_remesh_sequence`](../src/tnfr/physics/runtime_p2_reception_remesh_sequence.py)
closes the finite same-graph boundary left open by §2.15. It accepts the global
kernel certificate and one intact `ExecutedEventRemeshCycleSequence`. For each
cycle it selects one executor-owned EN event, constructs the stage certificate
of §2.15 and requires its two EPI endpoints to equal the complete schedule EPI
endpoints. Thus the observed schedule EPI transition is the represented
half-Reception kernel with $q=0$, even when the schedule has no common affine
metric and the causal executor was invoked with
`require_runtime_telescope=False`.

The adapter also constructs a `RuntimeRemeshHistoryBridgeObservation` for every
cycle, including the last. Each bridge must be identity-bound to that cycle and
must identify an applied hard-clipped binary64 $\alpha=1$ REMESH as the exact
numeric copy of its selected global-delay row. The signed rounding and clipping
residuals vanish, realizing the source class's $\eta=0$ boundary. Ordered P2
support, the positive metric ray, interval, delays, history capacity and the
runtime alpha-source provenance are fixed across the finite invocation.

Let $L=\tau_g+1$ and write the outgoing runtime history chronologically. Only
its last $L$ rows are active in this pure-global-delay recurrence. Each cycle
appends its consensus post-schedule row before REMESH, so after $N\geq L$
observed cycles the final active suffix has zero spatial-disagreement energy in
every row. The same-cycle bridges identify zero-energy post-REMESH fields from
the horizon onward and at the final committed endpoint. Retained rows before
the suffix may be stale or outside the source interval; no claim uses them.

This is a sealed certificate of one completed finite trace. It does not bind
the current live graph after observation, audit auxiliary Reception writes,
prove future or unobserved complete-runtime repetition, establish solver
accuracy or extend the result to full TNFR stability.

### §2.17 Transactional policy for each finite $P_2$ invocation

[`execute_p2_half_reception_remesh_policy_invocation`](../src/tnfr/physics/runtime_p2_reception_remesh_policy.py)
closes the operational gap between preparing a live graph and constructing the
certificate of §2.16. Every call takes an outer graph snapshot before
materializing caller-owned specifications or metric weights. Its
read-only preflight requires the intact abstract kernel; the same ordered
undirected mutual-singleton two-node support and positive diffusion-metric ray;
positive capacities and
symmetric conductance; exact binary64 `EN_mix=0.5`; matching hard-clipped
$\alpha=1$ controls, delays, interval and history capacity; an in-interval live
EPI pair that is not identically zero; active global-delay history; and at least
$L=\tau_g+1$ distinct,
clock-chained, one-cycle zero-flow `EN -> IL -> REMESH` specifications. This
initial policy excludes physical flow partitions.

The policy then invokes the graph-owned causal executor with a live grammar
context that rederives the U1a premise at every cycle start. A cycle-start pair
that is identically zero rejects the next Reception. The policy requests no
affine telescope and constructs, revalidates and identity-binds the existing
finite certificate before the outer transaction can commit. Post-certification
must leave the graph unchanged. Any preflight, execution, callback, observation,
certificate or post-certification-read failure restores graph-owned state to
the call boundary.
Emitted I/O, warnings, external resources and aliases that exist only outside
the graph remain subject to the executor's existing rollback boundary.

This mechanism permits successive calls, but each call is validated from its
own live entry state and produces a distinct finite trace. It does not prove
that the hypotheses will hold before a later unobserved call. Reception also
writes auxiliary kind, source, operator-history and pattern telemetry; those
fields are transactionally restored on failure, but their stability is not part
of the centered-EPI theorem. The forward-invariant complete-runtime class,
solver properties and full TNFR stability therefore remain open.

---

## §3. The Literal $\tau_g\to\infty$ Question

For a fixed finite `_epi_hist`, increasing $\tau_g$ eventually triggers the
insufficient-history guard, so the runtime call becomes a no-op. If the stored
history grows with $\tau_g$, then the domain, selected snapshot, deque length,
and possibly the graph state change together. A nontrivial limit requires a
declared common state space, an embedding of histories, parameter bounds, and a
mode of convergence.

N15 supplied none of those data. Consequently:

- the fixed-history pointwise behavior is eventually the guard-induced no-op;
- a growing-history runtime limit remains undefined until an embedding and
  trajectory are specified;
- neither behavior is the finite cyclic Cesàro projection of §7.

---

## §4. Corrected Finite Cyclic Surrogate

Fix integers $n,\tau_l,\tau_g>0$ with $n$ divisible by both delays, and fix
$0<\alpha<1$. Let $S$ be the cyclic shift on $\mathbb C^n$. The DFT basis
$v_k(j)=n^{-1/2}e^{2\pi i kj/n}$ diagonalizes $S$ and therefore $F$.

For $\omega_k=2\pi k/n$, the eigenvalue is

$$
\mu_k
=\beta+\gamma e^{-i\omega_k\tau_l}
       +\delta e^{-i\omega_k\tau_g}.
$$

This statement is exact for the finite cyclic filter. It is not a transfer
function for the companion history-advance map.

---

## §5. Contractivity and Power Boundedness

Because all three coefficients are positive and sum to one,

$$|\mu_k|\leq\beta+\gamma+\delta=1.$$

The matrix $F$ is a polynomial in the unitary matrix $S$, hence is normal. It
follows that

$$\|F\|_2=\max_k|\mu_k|\leq1,
\qquad
\|F^m\|_2\leq1.
$$

Power boundedness is therefore proved for this finite cyclic model. It was not
proved for the historical infinite companion operator. In particular, for the
old weight $w(k)=\rho^{-k}$, the stated right-shift norm $\sqrt\rho$ had the
direction reversed; the norm is $\rho^{-1/2}$ with that convention.

---

## §6. Fixed Modes: GCD, Not LCM

For $0<\alpha<1$, equality $\mu_k=1$ in the convex combination requires

$$
e^{-i\omega_k\tau_l}=1,
\qquad
e^{-i\omega_k\tau_g}=1.
$$

Let $d=\gcd(\tau_l,\tau_g)$. On a compatible window, the common solutions are

$$
\omega_m=\frac{2\pi m}{d},
\qquad m=0,\ldots,d-1,
$$

or DFT-bin indices $k=mn/d$. The fixed subspace has dimension $d$.

The historical use of $\operatorname{lcm}(\tau_l,\tau_g)$ as the fixed-mode
period was incorrect. The LCM can still be used as a convenient sample-window
alignment, but it does not determine the common fixed modes. For the defaults
$(\tau_l,\tau_g)=(4,8)$, $d=4$, so the fixed frequencies are
$0,\pi/2,\pi,3\pi/2$ modulo $2\pi$.

---

## §7. Cesàro Projection Theorem for the Surrogate

Define

$$
A_M=\frac1M\sum_{j=0}^{M-1}F^j.
$$

In the DFT basis, the multiplier of $A_M$ is $1$ when $\mu_k=1$ and

$$
\frac{1-\mu_k^M}{M(1-\mu_k)}
$$

otherwise. Since the model is finite-dimensional, these nonfixed multipliers
converge to zero. Therefore

$$
A_M\xrightarrow[M\to\infty]{\|\cdot\|_2}P_d,
$$

where $P_d$ is the orthogonal projector onto $\ker(I-F)$.

For a fixed window and fixed delays, an $O(1/M)$ operator-norm bound follows
with a constant depending on
$\min_{\mu_k\ne1}|1-\mu_k|$. No uniform constant follows as the window,
delays, or coefficients vary. No rate for the nonlinear structural candidate
energy follows from this state-space estimate.

---

## §8. Projection Algebra

The finite surrogate projection satisfies

$$P_d^*=P_d,
\qquad
P_d^2=P_d,
\qquad
\|P_d\|_2=1$$

when its range is nonzero. Its spectrum is contained in $\{0,1\}$; both values
occur when the selected subspace is proper and nontrivial.

These are properties of $P_d$ in the Euclidean norm on the declared periodic
history window. They do not transfer automatically to graph observables or to
the runtime map.

---

## §9. Why the Historical $H^2$ Proof Does Not Apply

The historical note represented the companion head-insertion plus shift as if
it were multiplication by
$\beta+\gamma z^{\tau_l}+\delta z^{\tau_g}$. That symbol belongs to a
translation-invariant convolution or cyclic filter, not to the companion map
whose first row is updated while the remaining rows shift.

Additional problems were:

1. constant infinite histories do not belong to unweighted $\ell^2$ or
   $H^2$ coefficient space;
2. the claimed shift norm used the inverse geometric factor;
3. a spectral-radius bound was asserted without a valid spectrum formula;
4. power boundedness of the companion map was assumed rather than proved;
5. symmetrizing an operator does not preserve the spectra of all its powers.

The mean ergodic theorem is valid when its hypotheses hold. Those hypotheses
were not established for the operator that the historical proof defined. The
finite cyclic theorem in §7 supplies a valid restricted replacement.

---

## §10. Relation to the Nodal Equation

REMESH changes EPI through a canonical operator path. The finite filter can be
applied to a sampled EPI history as an auxiliary diagnostic. An identity such
as

$$\partial_t(P_d\mathrm{EPI})=P_d(\partial_t\mathrm{EPI})$$

requires a common linear function space and sufficient regularity. Even when
that commutation is valid, substituting the nodal equation only gives

$$\partial_t(P_d\mathrm{EPI})=P_d(\nu_f\Delta\mathrm{NFR}).$$

It does not show that the projected trajectory is produced by the runtime
REMESH operator or that pressure and capacity close on the projected state.

---

## §11. Structural Charge

The structural charge diagnostic is

$$Q=\sum_i(\Phi_s(i)+K_\phi(i)).$$

It depends on pressure, phase, topology, and the corresponding field
extractions. An EPI-history projector does not act on all of these arguments.
Exact preservation would require a specified lifted state map and an
invariance relation such as $Q\circ P_d=Q$ along a separately conserved
evolution. Neither follows from $P_d^2=P_d$ or $P_d^*=P_d$.

Thus the historical projected-Noether conservation claim is superseded.
Charge before and after REMESH remains trajectory telemetry unless a
model-specific proof supplies the missing commutation and conservation laws.

---

## §12. Energy and Isometry Boundary

The structural functional

$$
E=\frac12\sum_i
(\Phi_s^2+|\nabla\phi|^2+K_\phi^2+J_\phi^2+J_{\Delta\mathrm{NFR}}^2)_i
$$

is a nonnegative candidate energy. EPI is not an explicit term. Therefore an
EPI-only write leaves a same-snapshot evaluation unchanged when every derived
field is held fixed. This dependency fact is not an isometry and not a
conservation theorem after pressure, currents, phase, or fields are updated.

For the finite surrogate, orthogonality gives

$$\|P_dx\|_2\leq\|x\|_2.$$

This contracts the declared history norm, not the structural functional $E$.
For $E[P_dx]\leq E[x]$ one would need an $E$-compatible state space and
contractivity in that energy metric. Monotonic decay along a trajectory needs
an evolution law with a verified nonpositive derivative.

---

## §13. Registry Reuse

`P_d` is a read-only numerical projection derived from a selected surrogate.
It can be computed by a utility function without adding a class to the engine's
operator registry. This establishes only implementation reuse:

$$\text{auxiliary computation} \not\Rightarrow
  \text{new registered state transformation}.$$

The legacy phrase “no fourteenth operator is required” is valid only in this
narrow engineering sense.

---

## §14. Catalog Completeness Remains Open

The current registry has 13 declared operators and coherent metadata. To prove
that these operators exhaust all admissible TNFR transformations would require:

1. a transformation space defined independently of the existing names;
2. admissibility axioms derived from the nodal equation and invariants;
3. a representation or generation theorem for every admissible map;
4. an irreducibility or equivalence criterion for proposed new maps.

Registry size, reload idempotence, metadata alignment, and reuse of `P_d` do
not supply those ingredients. This is the open S10 boundary in
[CORE_RESEARCH_PROGRAM.md](CORE_RESEARCH_PROGRAM.md).

---

## §15. Spectral Comparisons

For fixed $n$ the projector has finitely many selected DFT modes. Calling their
spacing a continuum spectral density adds an unsupported limit. Comparisons
with Riemann-zero density, Kolmogorov spatial spectra, or random-matrix spacing
laws can be posed only after specifying a scaling family and an intertwining
map between observables.

The corrected fixed-mode calculation is useful as a finite periodicity test.
It proves no spectral universality and no mismatch theorem about the full
runtime operation.

---

## §16. P50 Finite Fourier Diagnostic

`build_resonant_bin_mask` retains the historical requirement that the sample
count be divisible by $\operatorname{lcm}(\tau_l,\tau_g)$. Within that window,
it now selects the bins fixed by both delays, using
$d=\gcd(\tau_l,\tau_g)$.

`split_residue_by_remesh_infinity` is a legacy API name. It returns
$P_dx$ and $(I-P_d)x$ using an orthogonal DFT mask. Parseval gives, up to
roundoff,

$$
\frac{\|P_dx\|_2^2}{\|x\|_2^2}
+\frac{\|(I-P_d)x\|_2^2}{\|x\|_2^2}=1.
$$

This identity validates the numerical split. It does not identify the
analytic support of an off-grid signal: a finite rectangular window creates
spectral leakage.

---

## §17. TNFR-Riemann Boundary

The P31 prime-ladder signal can be projected onto the finite periodic subspace
and its complement. A small selected fraction means only that the declared
finite sample has little DFT energy in those bins. No equality relates this
split to the smooth and oscillatory parts of the Riemann counting formula.

Consequently N15 and P50 do not advance T-HP or RH. Historical B1/B2 language
is retained only as programme history, not as a theorem implied by the Fourier
certificate.

---

## §18. Navier–Stokes Boundary

The finite surrogate acts on a one-dimensional sample index. A temporal
projection does not by itself produce, preserve, or rule out a spatial
$k^{-5/3}$ spectrum. Any statement about a Navier–Stokes field requires a
declared space-time tensor product and proof that the temporal map acts as the
identity on the spatial factor.

No conclusion about vortex stretching, regularity, or a cascade follows from
the REMESH surrogate.

---

## §19. Historical Claim Ledger

| Historical N15 claim | Current status |
|---|---|
| $\tau_g\to\infty$ equals a Cesàro projector | Superseded: these are different limits |
| Companion history map has the polynomial Fourier symbol | Superseded: the symbol belongs to the cyclic/convolution filter |
| Fixed modes use $\operatorname{lcm}(\tau_l,\tau_g)$ | Corrected to $\gcd(\tau_l,\tau_g)$ for $0<\alpha<1$ |
| The history operator is power bounded on the stated $H^2$ model | Unproved for that companion map |
| No finite companion-history Lyapunov functional is available | Corrected in the fixed uniform, unclipped, finite-dimensional scope of §2.4 |
| Projected structural charge is conserved | Conditional and unproved |
| Projected structural energy is monotone with universal $O(1/n)$ decay | Superseded |
| The surrogate closes the 13-operator catalog | Superseded; registry reuse only |
| Direct Riemann/K41/RMT verdicts apply to runtime REMESH | Superseded; finite-surrogate comparisons only |

The original commits remain useful provenance:

- `a1f298fd`: historical W1 operator-existence claim;
- `badac156`: historical W2 conservation and Lyapunov claim;
- `48b0574a`: historical W3 spectral and branch verdict.

---

## §20. Corrected Branch Status

- **Branch A**: established only for the finite cyclic fixed-delay surrogate:
  its Cesàro projector exists.
- **Finite companion branch**: the stationary-weighted disagreement functional
  of §2.4 is nonincreasing; strict temporal mixing holds only for
  $0<\alpha<1$, while $\alpha=1$ retains periodic orbits.
- **Exact policy branch**: under the common fixed-metric schedule-gain
  hypothesis of §2.11, prefix gain is at most one and `q<1` gives uniform
  geometric spatial-disagreement decay, including at $\alpha=1$.
- **Relative-defect branch**: under the uniform signed bound of §2.12, the
  effective factor is `q_eff=q*(1+eta)`; a causal adapter verifies that bound
  and its vector envelope on finite executed blocks, but not future invariance.
- **Represented-number boundary**: §2.13 gives the exact pairwise reduction,
  a bounded normal-valued obstruction at $\alpha=1/2$, and the
  forward-invariant hard-clipped REMESH-only class at $\alpha=1$ with
  $\eta=0$.
- **Binary64 $P_2$ kernel branch**: the configured half-Reception Jacobi kernel
  has global gain $q=0$ and composes with that $\alpha=1$ class to extinguish
  active-history disagreement after $\tau_g+1$ restricted cycles. One sealed
  executor event binds this kernel to a real two-phase EN EPI stage and its
  captured endpoint. A finite sequence adapter additionally binds every
  selected EN and same-cycle REMESH, verifies the active suffix of length
  $\tau_g+1$, and records post-horizon extinction on one executor-owned graph.
  A transactional policy now revalidates and executes this restricted protocol
  per call and post-certifies before commit. Future or unobserved repeated
  complete-runtime composition remains open.
- **Branch B1**: no universality conclusion follows without a scaling family
  and an intertwining map.
- **Branch B2**: no extra registry entry is needed to compute this projection;
  admissible-transformation completeness remains open.
- **Branch B3**: nonexistence is ruled out for the finite cyclic surrogate, not
  for the literal runtime $\tau_g\to\infty$ problem or the historical infinite
  companion map.

---

## §21. Scope Across TNFR Programs

The corrected results are internal and limited:

- it supplies a finite periodic-history diagnostic;
- it supplies an exact augmented-history disagreement balance and temporal
  limit for one fixed uniform unclipped companion recurrence;
- it supplies a conditional exact common-$q$ schedule-family theorem with a
  uniform normalized margin and prefix bound;
- it supplies a conditional signed relative-defect extension and a finite
  causal verifier for its complete energy-vector envelope;
- it characterizes pairwise represented REMESH defects and certifies the
  sufficient-history $\alpha=1$ hard-clip class under REMESH alone;
- it composes that class with the global $P_2$ half-Reception numeric kernel and
  obtains exact finite-time spatial-disagreement extinction for the restricted
  repeated binary64 recurrence;
- it binds one completed graph-owned EN event and its captured EPI endpoints to
  that global $P_2$ kernel by exact binary64 replay;
- it binds every selected EN and $\alpha=1$ REMESH in one finite graph-owned P2
  sequence and verifies extinction on the active suffix, without promoting the
  observation to future repetition;
- it supplies a reusable outer transaction that preflights, executes and
  post-certifies every finite P2 invocation independently;
- it separates an auxiliary linear model from the canonical clipped runtime;
- it corrects the fixed-mode arithmetic from LCM to GCD;
- it leaves all classical open problems unchanged;
- it leaves the S10 catalog-completeness problem open.

No “structural/operational universality” follows across arbitrary graphs,
initial states, clipping policies, or time-varying parameters.

---

## §22. Reproducibility and Direct Checks

The corrected numerical surface is
`src/tnfr/riemann/remesh_infinity_residue_split.py`.
For a compatible sample count $n$:

1. the selected bin count is $d=\gcd(\tau_l,\tau_g)$;
2. the selected indices are multiples of $n/d$;
3. applying the split twice leaves each part unchanged;
4. the two parts reconstruct the input up to FFT roundoff;
5. their inner product vanishes up to FFT roundoff;
6. their squared-norm fractions sum to one for nonzero input.

The operator-registry diagnostic in
`src/tnfr/riemann/operator_catalog_discipline_signature.py` checks declared
schema consistency only.

The finite companion theorem and its one-transition identity are exercised by
`tests/physics/test_remesh_history_stability.py`. The executed residual bridge,
pure schedule balance and adjacent runtime telescope are checked by
`tests/physics/test_runtime_remesh_history_stability.py`,
`tests/physics/test_remesh_schedule_stability.py`, and
`tests/physics/test_runtime_remesh_schedule_stability.py`. Together they verify
the exact stationary distribution, Jensen dissipation, equality case,
preserved ideal-history barycenter, signed runtime defects, recorded history
advance, finite additive telescoping and the distinct $\alpha=0$, $\alpha=1$
and $0<\alpha<1$ regimes.

The exact common-$q$ policy theorem and its public example are checked by
[`test_remesh_schedule_policy_stability.py`](../tests/physics/test_remesh_schedule_policy_stability.py)
and
[`test_remesh_schedule_policy_stability_example.py`](../tests/physics/test_remesh_schedule_policy_stability_example.py).
They verify the `D_q P` order, exact block powers, varying rational schedule
witnesses, prefix and repeated bounds, endpoint regimes, public facade and
fail-closed seals. A separate consensus-amplification witness confirms that the
result controls spatial disagreement rather than uniform temporal means.

The relative-defect theorem, causal adapter and public example are checked by
[`test_remesh_schedule_relative_defect_stability.py`](../tests/physics/test_remesh_schedule_relative_defect_stability.py),
[`test_runtime_remesh_schedule_relative_defect.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect.py)
and
[`test_runtime_remesh_schedule_relative_defect_example.py`](../tests/physics/test_runtime_remesh_schedule_relative_defect_example.py).
They verify exact `q_eff`, cancellation-safe normalization by `J`, the `J=0`
case without division, schedule-gain and componentwise envelope inequalities,
complete-block endpoint factors, a positive binary64 defect at its exact
minimum `eta`, causal identity and fail-closed seals.

The represented-number boundary and its public example are checked by
[`test_binary64_remesh_relative_defect.py`](../tests/physics/test_binary64_remesh_relative_defect.py),
[`test_delayed_remesh_contract.py`](../tests/operators/test_delayed_remesh_contract.py)
and
[`test_binary64_remesh_relative_defect_example.py`](../tests/physics/test_binary64_remesh_relative_defect_example.py).
They verify shared-kernel replay, the exact $\alpha=1/2$ obstruction, the
zero-denominator branch, hard-clamp nonexpansiveness, signed-zero scope, the
$\alpha=1$ class at endpoints and subnormals, public type boundaries and
fail-closed proof seals.

The global P2 half-Reception composition and its public example are checked by
[`test_binary64_p2_reception_stability.py`](../tests/physics/test_binary64_p2_reception_stability.py)
and
[`test_binary64_p2_reception_stability_example.py`](../tests/physics/test_binary64_p2_reception_stability_example.py).
They cover the exact `q=eta=q_eff=0` composition, finite extinction horizon,
maximum finite values, subnormals, noncentered and degenerate intervals,
numeric-but-not-bitwise signed-zero consensus, hostile reseals and the canonical
default-factor energy-gain-four counterexample.

The finite executor binding and its public example are checked by
[`test_runtime_p2_reception_stage.py`](../tests/physics/test_runtime_p2_reception_stage.py)
and
[`test_runtime_p2_reception_stage_example.py`](../tests/physics/test_runtime_p2_reception_stage_example.py).
They verify same-invocation event/stage identity, grammar-aware two-phase EN,
runtime P2 neighbors, exact half mix, hard interval, capacity/conductance and
metric continuity, bit-exact endpoint replay, underflow and signed-zero scope,
nested proof revalidation and explicit negative future-runtime claims.

The finite causal P2 EN/REMESH sequence is checked by
[`test_runtime_p2_reception_remesh_sequence.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence.py).
It covers same-cycle stage/REMESH identity, fixed support, metric and
configuration provenance, the active suffix of length $\tau_g+1$, observed
post-horizon extinction and fail-closed broader claims. Its public facade, stub
and example are checked by
[`test_runtime_p2_reception_remesh_sequence_example.py`](../tests/physics/test_runtime_p2_reception_remesh_sequence_example.py).
Example
[`176_runtime_p2_reception_remesh_sequence.py`](../examples/02_physics_regimes/176_runtime_p2_reception_remesh_sequence.py)
records the finite same-invocation witness.

The transactional per-invocation policy is checked by
[`test_runtime_p2_reception_remesh_policy.py`](../tests/physics/test_runtime_p2_reception_remesh_policy.py).
It covers successful and successive calls, exact static preconditions, active
history scope, live U1a derivation, directed preflight rejection and
complete rollback when execution or post-certification fails or mutates the
graph. Its public facade, stub and example are checked by
[`test_runtime_p2_reception_remesh_policy_example.py`](../tests/physics/test_runtime_p2_reception_remesh_policy_example.py).
Example
[`177_runtime_p2_reception_remesh_policy.py`](../examples/02_physics_regimes/177_runtime_p2_reception_remesh_policy.py)
records two independently validated finite invocations.

The graph-owned finite causal wrapper and public example are checked by
[`test_event_remesh_causal_runtime.py`](../tests/operators/test_event_remesh_causal_runtime.py)
and
[`test_event_remesh_causal_runtime_example.py`](../tests/operators/test_event_remesh_causal_runtime_example.py).
They verify receipt/spec/schedule identity, same-invocation graph provenance,
outer rollback and the explicit negative repeated-stability scope.

The finite causal block-margin observer and its public example are checked by
[`test_runtime_remesh_schedule_block_margin.py`](../tests/physics/test_runtime_remesh_schedule_block_margin.py)
and
[`test_runtime_remesh_schedule_block_margin_example.py`](../tests/physics/test_runtime_remesh_schedule_block_margin_example.py).
They verify exact contiguous telescoping, normalized fields, identity binding,
zero-energy abstention, `kappa=139/256`, the `alpha=1` zero-margin boundary and
the explicit false uniform-class scope.

The effective-P2 family and its public example are checked by
[`test_event_remesh_reference.py`](../tests/physics/test_event_remesh_reference.py)
and
[`test_event_remesh_reference_example.py`](../tests/physics/test_event_remesh_reference_example.py).
They verify the exact
Euler recurrence, rational enclosures, strict proper-subdivision inequalities,
ideal REMESH scaling, runtime residual bound, public stub and explicit negative
scope properties.
The general pure theorem and nonregular-`P3` example are checked separately by
[`test_reversible_eigenmode_reference.py`](../tests/physics/test_reversible_eigenmode_reference.py)
and
[`test_reversible_eigenmode_reference_example.py`](../tests/physics/test_reversible_eigenmode_reference_example.py).
They verify the reversible metric, exact eigenmode identity, exponential and
Euler enclosures, both norm bounds, proper subdivision, public stub and the
conditional exact-real convergence scope without promoting a REMESH bridge.

---

## §23. Open Research Questions

The following problems remain open:

1. Define a common state space and convergence mode for a nontrivial runtime
   $\tau_g\to\infty$ limit.
2. Derive a useful uniform `eta` and a forward-invariant binary64
   relative-defect class for $0<\alpha<1$ with `q_eff<1`; retain the common-mode, canonical-factor,
   zero-margin, scaling and soft-knee obstructions.
3. Promote the finite same-graph $P_2$ EN/REMESH observation to a
   forward-invariant class under future complete grammar-aware event execution,
   including the auxiliary Reception state; or extend the numeric result to a
   broader verified kernel family without replacing class-wide proof by finite
   traces.
4. Determine the extra hypotheses needed to compose the implemented general
   finite executor/eigenmode binding with REMESH beyond effective $P_2$; its
   pressure, held-input and full-matrix endpoint defects currently stop at the
   pre-REMESH boundary.
5. Extend the companion result to changing $\alpha$, metric, delays or node
   support, or produce counterexamples.
6. Determine when a lifted REMESH map preserves a declared structural charge
   or dissipates the full tetrad energy.
7. Establish any bridge from the finite periodic projection to the Riemann,
   Navier–Stokes, or other programme observables.
8. Define the admissible TNFR transformation space independently and resolve
   catalog generation and irreducibility within it.

The exact current conclusion is therefore narrow: **a fixed finite cyclic
REMESH filter has a computable Cesàro fixed-mode projection, and a distinct
fixed finite companion recurrence has a nonincreasing augmented disagreement
functional with sharply separated mixing and pure-delay regimes. One applied
binary64 transition is decomposed into exact ideal, rounding and clipping
terms, and compatible adjacent recorded cycles have an exact finite additive
schedule/history telescope. One graph-owned outer executor supplies a compatible
requested telescope with same-invocation causal provenance and finite atomicity;
its explicit no-telescope branch retains the finite causal trace when the
schedule has no common affine metric. A sealed
observer extracts exact normalized lower margins from its contiguous finite
blocks; the implemented witnesses give `139/256` and zero without establishing
uniform runtime-class coercivity. A separate conditional exact common-$q$
policy theorem gives prefix gain upper bound one, uniform normalized margin $1-q$ and
repeated geometric spatial-disagreement decay on the fixed companion. Its
conditional relative-defect extension replaces $q$ by
$q_{\mathrm{eff}}=q(1+\eta)$, and a finite causal observer verifies every
signed defect, represented gain, history-energy envelope and complete-block
endpoint bound, including a positive binary64-defect witness. The exact
pairwise boundary then shows that a bounded hard-clipped box can require
$\eta=2^{210}-1/4$, while the $\alpha=1$ class has uniform $\eta=0$ and is
forward invariant under REMESH alone. A global binary64 $P_2$ half-Reception
EPI kernel has $q=0$ and composes with that class to extinguish active-history
spatial disagreement after $\tau_g+1$ restricted cycles. This repeated result
belongs to the numeric EPI kernels. One completed graph-owned event now binds
the P2 kernel to a grammar-admitted two-phase Reception EPI stage and its
captured endpoints. A second finite adapter binds every selected stage and
same-cycle $\alpha=1$ REMESH on one graph, verifies only the active suffix of
length $\tau_g+1$, and records post-horizon extinction. It does not establish
future or unobserved repetition. A reusable transactional policy now revalidates
the current restricted conditions on every call, runs the causal sequence and
constructs that finite certificate before commit; successive accepted calls
remain separate observations, and auxiliary Reception-state stability is not
certified. One
event-free effective-$P_2$ family additionally
has a rational continuous/Euler error enclosure, strict improvement across its
two declared proper subdivisions, exact ideal REMESH scaling and an explicit
runtime residual bound. A separate
[`general reversible exact-mode runtime adapter`](../src/tnfr/physics/runtime_eigenmode_reference.py)
binds finite executor-owned pressure-refreshed partitions and propagates
represented defects through complete Euler matrices, but it contains no REMESH
claim.
Derivation of a useful forward-invariant binary64 relative-defect class with
uniform `eta` for $0<\alpha<1$ and `q_eff<1`, a priori preservation of the P2
EN/REMESH hypotheses for future unobserved complete event execution, generic or
binary64 asymptotic convergence, repeated complete-runtime stability, the
clipped binary64 runtime limit, full structural invariants and global operator
completeness remain unresolved.**
