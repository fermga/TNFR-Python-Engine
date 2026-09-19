# Passive cycle relaxation with canonical delayed memory

**Status:** Exact conditional cycle relaxation derived from the existing
REMESH history theorem; finite runtime correspondence is checked separately.
**Research links:** B2.d.4/O3.a, S3, S8, S9 and S16.

## 1. Retained positive capacity removes the finite-clock obstruction

Use a fixed simple unit-conductance cycle with `n>=3` nodes, scalar EPI `x`,
uniform capacity `nu>0` and a regular phase twist. Its adjacent circular gap
is strictly below both the effective U3 gate and `pi/2`. Target-only
all-node Coupling without functional links preserves this regular twist
and uniform capacity. The topology, capacity and phase pressure gradients
vanish; with the effective fully normalized EPI coefficient `e>0`, the
refreshed canonical pressure and nodal flow are

$$
\Delta\mathrm{NFR}=-eLx,\qquad \dot x=-e\nu Lx,
\qquad L=I-W/2.
$$

Thus the flow uses the same canonical channels as
[joint support dynamics](CYCLE_SUPPORT_DYNAMICS.md). No Silence attenuation
or additional restoring law is inserted. Capacity persists, so repeating
one fixed partition satisfying the exact Euler restriction
`0 < e*nu*h_j <= 1/2` from section 2 supplies a common strict contraction
on every delay circuit. Positive duration alone is insufficient for Euler
stability; the continuous-flow result has its separate spectral bound. This removes the finite-clock explanation for EPI
contrast in the preceding block; it does not itself prove a localized
restoring mechanism.

The graph, coefficients, phase chart and capacities must remain fixed over
the declared family. Grammar admission of a complete event word is a
separate requirement. In particular, a registered REMESH word token is
advisory and must not be identified with the separately executed delayed
EPI map. The admitted bridge uses Coupling, Coherence and this advisory
Recursivity token. Coherence has its own phase-lock action and is not
generally inert in the phase sector: the regular strict twist is a fixed
point of its exact neighbor-midpoint update. In this restricted preparation
the bridge leaves EPI, capacity and phase unchanged before the flow;
its direct pressure changes are replaced by the canonical refresh needed
to identify the flow above. Runtime phase drift and pressure-realization
residuals remain separate measured quantities.

## 2. A rational full-state flow gain comes from the cycle spectrum

Let `bar(x)` be the spatial arithmetic mean and
`E(x)=sum_i(x_i-bar(x))^2/2`. A constant positive multiple of this metric
gives the same gain. The cycle eigenvalues are
`lambda_j=1-cos(2*pi*j/n)`. In particular,

$$
\lambda_2=2\sin^2(\pi/n)\geq\ell_n:=\frac8{n^2}>0,
\qquad \lambda_{\max}\leq2.
$$

Here the subscript `2` denotes the first nonconstant eigenvalue. The
inequality uses `sin(t)>=2t/pi` on `[0,pi/2]`; `ell_n` is a derived rational
bound, not a fitted parameter or an asserted exact cycle eigenvalue.

For an exact refreshed Euler partition with positive segment durations
`h_j`, put `kappa_j=e*nu*h_j` and require `kappa_j<=1/2`. Every modal factor
lies in `[0,1]`, and for each nonconstant mode it is at most
`1-kappa_j*ell_n`. Therefore the complete partition map `S_E` preserves
the spatial mean and obeys

$$
E(S_E x)\leq q_E E(x),\qquad
q_E=\prod_j(1-\kappa_j\ell_n)^2<1.
$$

This bounds all spatial modes, including a local EPI bump; it is not a
single-eigenmode fit. For the exact continuous flow over
`h=sum_j h_j`, a separate rational upper bound is

$$
E(S_Cx)\leq e^{-2e\nu h\lambda_2}E(x)
\leq q_CE(x),\qquad
q_C=\frac1{1+2e\nu h\ell_n}<1.
$$

The last step uses `exp(-u)<=1/(1+u)` for `u>=0`. The continuous and Euler
maps have separate bounds; this comparison is not an Euler accuracy or
binary64 convergence theorem. The restricted half-step condition avoids
the nondecaying checkerboard boundary at `kappa=1` on an even cycle.

## 3. The memory state contains pre-REMESH samples

The actual [event/REMESH executor](../src/tnfr/operators/event_remesh_runtime.py)
first completes its schedule and physical flow, appends the resulting EPI
field to `_epi_hist`, and then applies the delayed map. A delay `tau` reads
`history[-(tau+1)]` after that append. The post-REMESH output is not appended
again.

Write `y_k` for this stored pre-REMESH EPI and `x_k` for its same-cycle
post-REMESH result. Here `y` is a history label, not the shifted support
coordinate of the preceding note. For one uniform `alpha` in `[0,1]`,

$$
\begin{aligned}
\beta&=(1-\alpha)^2,&
\gamma&=\alpha(1-\alpha),&\delta&=\alpha,\\
x_k&=\beta y_k+\gamma y_{k-\tau_l}+\delta y_{k-\tau_g},\\
y_{k+1}&=Sx_k.
\end{aligned}
$$

The endpoint `alpha=0` is a formal identity-memory control of the exact
certificate. The actual delayed REMESH configuration requires positive
alpha. A runtime memory-free control instead executes the same admitted
word and flow while omitting the separate delayed-map call.

Coincident delays combine. Let `m` be the greatest delay with a positive
coefficient and set `H=m+1`. Then `Y_k=(y_k,...,y_{k-m})` is the active
history. At `alpha=0`, `m=0`; at `alpha=1`, only `tau_global` is active.
Older retained rows lie outside this active state. A startup buffer is
declared initial data and is not automatically a past solution of the
nodal equation. Insufficient-history no-op behavior lies outside the
applied recurrence until the required samples exist.

This indexing matters. The schedule acts on the mixed head to produce the
next stored pre-REMESH row. Storing post-REMESH states instead changes the
history recurrence and cannot be substituted without a separate proof.

Nor can unchanged delayed rows be repeatedly reused as rolling memory.
If their combined contribution is held at `b`, the pre-map recurrence is
`y_(k+1)=S*(beta*y_k+b)`. For a spatial mode with schedule factor `0<r<1`,
its stationary amplitude is `r*b_mode/(1-beta*r)`, which can be nonzero.
This is sustained by held input, not the advancing companion proved below.
History advancement and its causal provenance distinguish these experiments.

## 4. Reuse of the sealed common-gain history theorem

Let `P` and its positive stationary row `pi` be the temporal companion and
stationary distribution already constructed by
[`remesh_history_stability.py`](../src/tnfr/physics/remesh_history_stability.py).
For the chronological energies in newest-first order,

$$
v_k=(E(y_k),\ldots,E(y_{k-m}))^\top,\qquad V_k=\pi^\top v_k.
$$

Convexity of the centered quadratic energy and either gain `q=q_E` or
`q=q_C` give the componentwise envelope

$$
v_{k+1}\leq D_qPv_k,\qquad D_q=\operatorname{diag}(q,1,\ldots,1).
$$

This is exactly the order and convention of the existing
[policy certificate](../src/tnfr/physics/remesh_schedule_policy_stability.py),
whose proof is in [the REMESH derivation](REMESH_INFINITY_DERIVATION.md).
That certificate proves prefix gain at most one and

$$
V_{k+N}\leq q^{\lfloor N/H\rfloor}V_k.
$$

No new temporal stability theorem is required: the cycle calculation
supplies the previously conditional spatial schedule gain. Since `q<1`
and every `pi_j>0`, all active rows lose their nonuniform spatial component.
The post-REMESH field also loses it, by the same convexity inequality.
The physical pressure therefore tends to zero in this exact fixed family.

With a prepared row `y_{-1}=x_initial` and unit delays, the first executed
flow creates `y_0`. The initial augmented energy is then formed from
`(y_0,y_{-1})`; two successive executed cycles provide one transition
from `V_0` to `V_1`. Counting the initial preparation as an additional
verified schedule transition would change this finite bound's provenance.

At `alpha=1`, the recurrence becomes
`y_{k+1}=S*y_{k-tau_global}`: diffusion contracts each delay circuit once
per `tau_global+1` stored transitions. Pure-delay temporal permutation
therefore does not protect nonuniform spatial structure when every circuit
includes the positive-capacity flow. At `alpha=0`, `H=1` and the certificate
reduces to the ordinary memory-free schedule estimate.

The common strict gain cannot be replaced by total elapsed time alone.
For `alpha=1`, `tau_global=1`, each mode obeys
`a_k=exp(-e*nu*lambda*h_k)*a_{k-2}`. Taking `h_{2j}=1` and
`h_{2j-1}=2^{-j}` gives infinite total time, but the odd delay lineage
receives only finite exposure and can retain nonzero contrast. The fixed
partition assumed here excludes that control; a broader varying-duration
theorem would need sufficient exposure along every active delay lineage.

## 5. Mean history, transients and the localization boundary

The result concerns spatial disagreement. The schedule preserves means,
so historical means obey the scalar REMESH companion. If every initial
active row has the same mean, every subsequent pre- and post-REMESH field
has it, and spatial relaxation gives full consensus to that constant.
Otherwise `0<alpha<1` mixes these means to their stationary history
barycenter, while `alpha=1` can retain a period dividing
`tau_global+1` among spatially uniform fields. That temporal effect is not
localized structure. Zero pressure during a physical flow also does not
forbid a subsequent named REMESH event from changing a uniform EPI level.

Even in the decaying regime, the current EPI energy need not decrease at
each delayed map: older high-contrast rows can raise the current contrast.
The nonincrease theorem applies to the augmented energy at stored-row
boundaries. It does not justify multiplying a gain assigned only to the
current REMESH input across an arbitrary history.

A simple comparison shows why memory can prolong a passive transient.
Let a nonconstant mode have a positive schedule factor `0<r<1`, and prepare
every active startup amplitude equal to `a_0>0`. The exact scalar recurrence
has nonnegative coefficients summing to one:

$$
a_{k+1}=r\left(\beta a_k+
\gamma a_{k-\tau_l}+\delta a_{k-\tau_g}\right).
$$

Induction gives `a_{k+1}<=a_k` and `a_{k+1}>=r*a_k`; hence
`a_k>=r^k*a_0`, the memory-free amplitude. The common-gain theorem still
forces `a_k->0`. This comparison requires the specified positive mode and
repeated startup preparation; arbitrary signed histories can interfere
differently. It is not a claim that memory always slows every trajectory.

Thus uniform delayed memory plus retained positive-capacity diffusion does
not sustain nonuniform EPI in this canonical cycle class. A persistent
twisted phase pattern can coexist with spatially uniform EPI, but delayed
echoes alone do not turn the passive EPI transient into a self-restoring
localized entity. Moving structural support, nonlinear branch changes,
clipping and more general graph/operator families require their own
analysis rather than a transfer of this result.

## 6. Exact adapter and finite executor evidence

[`cycle_memory_relaxation.py`](../src/tnfr/physics/cycle_memory_relaxation.py)
derives `ell_n`, the complete Euler-partition gain and the continuous gain.
It passes each gain to the existing sealed policy certificate while
retaining their common sealed REMESH history model. Node count, uniform
capacity, fully normalized EPI coefficient, positive physical partition,
alpha and active delays remain explicit inputs. Exact rationals are
preserved, and other supported real values use the shared materialized-real
reader. The adapter introduces no second history recurrence or new
transport kernel. Its declared physical partition repeats unchanged in
the exact family; the history delays count stored cycle samples, not units
of physical time.

[Independent exact tests](../tests/physics/test_cycle_memory_relaxation.py)
check local-bump gains across cycle sizes, the true pre-REMESH recurrence,
finite augmented envelopes, coincident and inactive delays, common means,
pure-delay damping, uniform temporal cycles and current-energy transients.
They also verify that altered nested policy payloads lose their seal.

[Runtime tests](../tests/physics/test_cycle_memory_relaxation_runtime.py)
and the [benchmark](../benchmarks/cycle_memory_relaxation.py) compare the
same admitted Coupling/Coherence/advisory-Recursivity word and physical
partition with and without the separately invoked delayed map. The memory
case uses explicit unit delays and a prepared initial history row. Four
executed cycles expose three adjacent pre-REMESH history transitions.
The records separately retain actual channel states, applied delayed
proposals, the ideal exact mixture, canonical pressure refresh and Euler
arithmetic residuals. The regular phase twist, uniform positive capacity
and fixed support must be checked in these records; they are not inferred
from the exact companion's input vectors.

These finite observations connect the implementation to the restricted
reference. They do not prove that future binary64 histories remain inside
a uniform relative-defect class, that rounding preserves a positive
contraction margin indefinitely, or that a generic adaptive operator
schedule satisfies the same assumptions. Those remain distinct from the
exact passive-relaxation theorem established here.

In the bounded default C8 run at physical `T=2`, all capacities stay at one
and every accumulated nodal clock is exactly two. The initial centered
energy `0.109375` becomes `0.07105253529907486` with delayed memory and
`0.05179949153244585` in the advisory-only control. Winding remains one.
The four observed augmented energies decrease, and both complete two-step
block slacks against the conservative `q_E≈0.9773016233624776` are positive.
Clipping is inactive; the maximum measured REMESH rounding defect is
`8.326672684688674e-17`. These recorded finite values demonstrate delayed
relaxation in this preparation; the asymptotic statement belongs to the
separate exact model and its hypotheses.
