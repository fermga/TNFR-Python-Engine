# Heterogeneous EPI diffusion stability theorem

**Status**: Derived for fixed, connected, symmetric conductance with positive
fixed structural frequencies or time-varying frequencies bounded above and
away from zero, and for
fixed-node-set symmetric switching families with a common metric. A directed
fixed-capacity pressure-transient criterion is also exact. Affine EPI resets
now have an exact finite-gain criterion in the same common metric and a
conservative hybrid-word bound. Reception and Resonance are the first two
catalog realizations connected to that theorem at local and all-target stage
boundaries, with fixed-map repetition under explicit domains. Emission, Silence,
Expansion, Contraction, Mutation and Transition additionally have a conditional
internal frozen-proposal realization/gain certificate; Coherence has an
immutable execution contract without such a gain. Coupling has an explicit
simultaneous circular/topological merge, Dissonance has a snapshot-bound
local/propagated pressure reduction, and Self-organization has a collision-safe
snapshot-bound child-support and hierarchy merge. Recursivity has an immutable
advisory-only glyph stage. The separate delayed REMESH operation now has an
exact three-input recurrence, graph-state atomic execution and scoped one-step
convex and fixed-history gain evidence. A distinct exact, uniform, unclipped
fixed-delay companion recurrence has a stationary-history Lyapunov theorem;
see
[`REMESH_INFINITY_DERIVATION.md` section 2.4](REMESH_INFINITY_DERIVATION.md#24-exact-finite-companion-history-stability).
For any fixed connected symmetric rational conductance and positive rational
capacity, a separate sealed reference theorem now gives the exact exponential
solution, pressure-refreshed Euler product and finite error bounds for one
exact nonuniform eigenmode. It also proves conditional exact-real convergence
under admissible partitions whose maximum step tends to zero. This pure theorem
has no runtime provenance; the effective-P2 event/REMESH family remains its
first executor-linked specialization.
Event execution can bind supported flow
and glyph certificates into an exact represented-map gain product for one fully
eligible observed finite trace in one common metric; it does not identify a
global executable binary64 map. Stability of clipped binary64 runtime
history-updated repetition, unrestricted mixed words, nonlinear regimes and
the full catalog remains open.

## Question and hypotheses

For the isolated EPI channel of the nodal equation, does heterogeneous
structural frequency preserve convergence?

Let `W` be a fixed symmetric nonnegative conductance matrix on a finite connected
graph with at least two nodes. Write `d_i = sum_j W_ij`, `D = diag(d_i)`,
`B = D-W`, and assume `d_i > 0` and `nu_i > 0` for every node. With
`x = EPI`, the canonical EPI pressure gives

```text
x' = -diag(nu) L_rw x = -diag(nu_i/d_i) Bx.
```

Here `W` is read from the repository edge attribute `weight`, with unit
conductance when `weight` is absent. The independent `length` attribute belongs
to shortest-path geometry and does not enter this diffusion theorem. Structural
potential uses explicit `length` when present, otherwise falls back to `weight`
for compatibility and then to unit length. A model whose coupling strength and
path distance differ should declare both explicitly; see
[`_edge_semantics.py`](../src/tnfr/physics/_edge_semantics.py).

The null hypothesis was that heterogeneous `nu_i` could destroy a common
Lyapunov function even in this restricted setting. The theorem below rejects
that hypothesis. It does not address the other pressure channels or arbitrary
operator sequences.

## Ideal real-arithmetic theorem

Define

```text
H = diag(d_i/nu_i),
c = (1^T H x)/(1^T H 1),
y = x-c*1,
V(x) = 1/2 y^T H y.
```

Then `1^T H x` and therefore `c` are conserved. Moreover,

```text
V' = -y^T B y.
```

Because the graph is connected, `B` is positive semidefinite with kernel
`span{1}`. Let `lambda_* > 0` be the first positive generalized eigenvalue of

```text
Bv = lambda Hv.
```

The generalized Poincare inequality on the `H`-orthogonal complement of `1`
gives

```text
V' <= -2 lambda_* V,
V(t) <= exp(-2 lambda_* t) V(0).
```

Thus the continuous-time EPI field converges exponentially to the uniform
state `c*1`. Heterogeneous structural frequency changes both the conserved mean
and the relaxation rate; replacing it by its arithmetic mean generally changes
the dynamics.

## Proof

Since `H diag(nu) D^-1 = I`, the flow is `x'=-H^-1 Bx`. Symmetry gives
`1^T B=0`, hence

```text
d/dt (1^T Hx) = -1^T Bx = 0.
```

The centered field therefore satisfies `y'=x'`. Differentiating its weighted
energy yields

```text
V' = y^T H y' = -y^T Bx = -y^T By,
```

where `B1=0` was used in the last equality. Finally, `1^T Hy=0`, so the
Rayleigh characterization of the first positive eigenvalue of
`H^-1/2 B H^-1/2` gives `y^T By >= lambda_* y^T Hy = 2 lambda_* V`.

## Executable represented-coefficient theorem

The analytic cancellation `H diag(nu) D^-1=I` uses exact real arithmetic.
The engine performs several binary64 operations: it materializes a Laplacian
`B_r`, mobility `M_r` and displayed metric

```text
H_r = diag(fl(d_i/nu_i)).
```

Even when every input is finite and positive, `H_r M_r` need not equal the
identity exactly. Consequently, the displayed weighted mean need not be an
exact invariant and the numerical generalized eigenvalue of the ideal pair
`(B,H)` is not by itself a safe lower bound for the executed coefficients.

The executable certificate instead treats every already materialized
binary64 coefficient of `B_r`, `M_r` and `H_r` as an exact rational number and
sets `A_r=M_r B_r`. Binary64 row accumulation can make `B_r 1` nonzero as an
exact rational identity, even when the underlying real-weight Laplacian kills
constants. A global homogeneous disagreement bound therefore first requires

```text
Q_r A_r 1 = 0,
Q_r = I - 1 h_r^T/(h_r^T 1).
```

This is the exact invariance condition for the consensus subspace. For the
instantaneous `H_r`-orthogonal projection

```text
c_r(x) = (1^T H_r x)/(1^T H_r 1),
y = x-c_r(x)1,
V_r(x) = (1/2)y^T H_r y,
```

the projection derivative and the uniform image of `A_r 1` drop out because
`1^T H_r y=0`. Therefore

```text
V_r' = -y^T S_r y,
S_r = (H_r M_r B_r + B_r^T M_r H_r)/2.
```

The implementation verifies the consensus-subspace identity, restricts `S_r`
and `H_r` to the exact rational subspace `1^T H_r y=0`, proves positive
definiteness by rational LDL elimination and computes a rigorous
generalized-quotient lower bound `lambda_cert`. When both checks pass, it
follows that

```text
V_r(t) <= exp(-2 lambda_cert t) V_r(0).
```

Canonical pure-EPI diffusion has the stronger fixed-point contract
`A_r 1=0`: every uniform EPI field must remain stationary. The executable
certificate reports the weaker quotient condition `Q_r A_r 1=0`, but promotes
`is_certified` only when the rational quotient bound is positive **and** the
uniform fixed-point identity holds exactly. A represented matrix can therefore
have a valid quotient contraction calculation while still being rejected as a
canonical diffusion flow.

The separately reported eigensolver gap and rate remain estimates. The public
`certified_exponential_rate_lower_bound` is a downward-rounded representation
of `2 lambda_cert` and is the only rate consumed by hybrid composition.
Exact conservation of the displayed weighted mean is checked independently as
`1^T H_r A_r=0`. This is separate from consensus-subspace invariance. If the
mean identity fails while `Q_r A_r 1=0` passes, contraction to the consensus
subspace remains certified, while the initial displayed weighted mean is not
promoted as the limiting consensus value. If consensus-subspace invariance
itself fails, a uniform zero-energy input supplies an exact obstruction to any
global multiplicative disagreement bound and the executable certificate
abstains.

## Engine certificate

[`verify_heterogeneous_diffusion_stability`](../src/tnfr/physics/structural_diffusion.py)
computes the displayed metric, snapshot projection center and total,
instantaneous Lyapunov balance,
generalized-gap estimate, exact quotient lower bound and certified exponential
rate. Its compatibility fields `conserved_total` and `equilibrium_value` are
snapshot projection values unless `exact_weighted_mean_preservation` is true.
`is_consensus` tests whether the snapshot lies in the uniform subspace within
the declared numerical tolerance; `is_equilibrium` instead checks the exact
rational identity `M_r B_r x=0` for the represented coefficients. These need
not agree: on the weighted triangle with conductances `0.1`, `0.2` and `0.3`,
binary64 degree accumulation makes a uniform field consensus while the
materialized generator does not leave it fixed, so the executable theorem
abstains.
The displayed `lyapunov_value`, `lyapunov_derivative` and
`derivative_upper_bound` are binary64 snapshot diagnostics. Hybrid composition
uses only the downward-rounded certified rate.
The certificate arrays are detached and read-only, and the function rejects
graphs outside the theorem's assumptions. Tests include a closed-form
two-node equality case, weighted heterogeneous paths, the uniform equilibrium,
shift invariance, zero capacity, zero-weight disconnection and asymmetric
transport.

This fixed-capacity certificate is the stability component used at both frozen
endpoints by the restricted
[`S16 integration certificate`](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#5-executable-s16-endpoint-certificate).
Running it twice does not establish a trajectory between those endpoints.

## Boundaries and falsification

The ideal theorem would be falsified by an in-scope exact-real graph for which
the weighted quantity changes, `V' > -2 lambda_* V`, or `lambda_* <= 0`.
The executable theorem is auditable through coefficient-level conditions:
exact uniform fixed-point preservation, positive rational quotient
dissipation, and the separately reported consensus-subspace and weighted-mean
identities. Failure of a required certificate condition is an abstention about
that represented matrix, not a refutation of the ideal theorem.

The fixed-graph theorem above does not cover:

- phase, frequency-gradient or topology pressure channels;
- time-dependent `W` outside the separately stated finite common-metric
  switching family, or time-dependent `nu_i` without the positive finite bounds
  treated below; nesting or REMESH;
- zero-capacity nodes or disconnected effective conductance;
- directed non-normal transport;
- explicit finite-step stability;
- convergence of every grammar-compliant operator word.

These exclusions define the next stability research steps rather than implicit
extensions of this theorem.

## Time-varying heterogeneous capacities

The fixed-capacity weighted energy above depends on `nu_i` and is therefore not
a common time-independent functional when capacity ratios change. The
Dirichlet energy itself supplies the required common functional.

Assume a measurable schedule with declared bounds

```text
0 < lower_i <= nu_i(t) <= upper_i < infinity
```

for every node and time. Set `M(t)=diag(nu_i(t)/d_i)` and
`E_D=x^T Bx/2`. The same pure EPI channel obeys

```text
E_D' = -(Bx)^T M(t)(Bx).
```

Let `mu=min_i(lower_i/d_i)` and let `lambda_2(B)>0` be the first
positive combinatorial eigenvalue. Since `B^2 >= lambda_2(B) B` on the
nonuniform subspace,

```text
E_D' <= -mu ||Bx||^2
     <= -2 mu lambda_2(B) E_D.
```

Therefore

```text
E_D(t) <= exp(-2 mu lambda_2(B)t) E_D(0).
```

The upper capacity bounds make `M(t)` bounded. Together with exponential
decay this makes `x'(t)` integrable, so `x(t)` converges to a uniform state.
Unlike the fixed-capacity theorem, its consensus value is generally dependent
on the capacity schedule. Exact piecewise-constant two-node evolutions in the
test suite give different final consensus values when the same two capacity
profiles are applied in opposite order.

[`derive_time_varying_diffusion_stability_bound`](../src/tnfr/physics/structural_diffusion.py)
computes this conditional common-Lyapunov rate from per-node lower and upper
capacity bounds. The bounds are declared assumptions; the read-only function
cannot certify that an unknown future schedule respects them.

The executable certificate interprets each effective binary64 conductance and
capacity bound as an exact real coefficient. It forms `W`, `D`, `B=D-W`, the
quotient gap, mobilities, current Dirichlet energy and decay rate with rational
arithmetic. The ordinary binary64 eigengap, products and energy are retained as
estimates only. In particular, floating accumulation in the separately
materialized `diag(strength)-adjacency` may fail to annihilate constants even
though the induced exact-real `B` does; both facts are reported without mixing
their scopes. If the exact positive rate underflows when rounded downward for
publication, `exact_real_continuous_time_model_certified` remains true while
`operational_binary64_rate_available` and its compatibility alias
`is_certified` are false. `runtime_integration_certified` is always false: this
function proves the conditional continuous-time model and does not inspect a
solver trajectory.

### Proportional-schedule corollary

If `nu_i(t)=a(t) nu_bar_i`, all generators are scalar multiples of one fixed
generator. Structural time `s(t)=integral_0^t a(tau) dtau` gives

```text
x(t) = exp(-s(t) diag(nu_bar) L_rw) x(0).
```

The weights `d_i/nu_bar_i` then define a conserved mean. If `s(t)` diverges,
the fixed-capacity theorem applies in structural time and determines the final
uniform value. This is the exact boundary between a clock change and genuinely
time-ordered heterogeneous transport.

## Explicit-Euler modal window

For a frozen symmetric network, the continuous decay rates are the eigenvalues
`lambda_k` of `diag(nu_f)L_rw`. An explicit-Euler step acts on each
nonstationary mode by

```text
q_k = 1-dt lambda_k.
```

All nonstationary modal amplitudes contract exactly when
`max_k |q_k| < 1`, equivalently `0 < dt < 2/lambda_max`. For target fraction
`epsilon`, the graph-specific step
count is the first `n` satisfying

```text
(max_k |q_k|)^n < epsilon.
```

[`diagnose_euler_relaxation_window`](../src/tnfr/physics/structural_diffusion.py)
reports the actual heterogeneous spectrum, stability limit and modal step
count. At `dt=0.5`, `nu_f=1` and `epsilon=1/(pi+1)`, the 21-node path requires
231 solver steps while `K4` requires 2. The canonical U4 window remains 3
**operator positions** in both cases. The operator-event contract now supplies
the missing physical timeline—zero-duration jumps separated by declared flow
intervals—but it does not convert the positional U4 policy into a spectral one.
Stationary-mode resolution uses a reported
dimensionless tolerance times the fastest decay rate. It is not an absolute
frequency threshold and is kept separate from any scaled EPI-update tolerance.

### Exact reversible single-eigenmode Euler reference theorem

This section centralizes the complete proof used by the pure reference kernel.
Let `W` be an exact rational symmetric nonnegative conductance matrix with zero
diagonal and connected positive support. Set

```text
d_i = sum_j W_ij,        D = diag(d_i),        B = D-W,
A = diag(nu) L_rw = diag(nu) D^-1 B,
H = diag(d_i/nu_i),
```

where every `d_i` and `nu_i` is positive. Then

```text
H A = B = B^T,
```

so `A` is self-adjoint in the weighted inner product
`<p,q>_H = p^T H q`. Define

```text
m_H = (1^T H x_0)/(1^T H 1),        v = x_0-m_H*1.
```

Assume that `v != 0` and that the exact rational identity

```text
A v = mu v,        mu > 0,
```

holds. Since `A 1=0`, the pure-EPI equation `x'=-Ax` has the exact solution

```text
x(t) = m_H*1 + exp(-mu*t)*v.                         (1)
```

The weighted mean is conserved, and this mode has centered energy

```text
E_H(v) = (1/2) v^T H v > 0.
```

Now let `P=(h_1,...,h_s)` be a positive partition of one fixed duration `T`,
and refresh the pressure before every exact-real explicit-Euler segment:

```text
x_(j+1) = (I-h_j A)x_j,        sum_j h_j=T.
```

If `0 < mu*h_j < 1` for every segment, the modal recurrence closes exactly and
its endpoint is

```text
x_P(T) = m_H*1 + g_P*v,
g_P = product_j (1-mu*h_j).                         (2)
```

Write `a_j=mu*h_j`. For `0 <= a <= 1`, the exact integral remainder identity

```text
exp(-a)-(1-a) = integral_0^a (a-s) exp(-s) ds
```

gives

```text
0 <= exp(-a)-(1-a) <= a^2/2.                       (3)
```

Set `b_j=exp(-a_j)` and `c_j=1-a_j`. Because
`0 < c_j <= b_j <= 1`, the product-difference identity

```text
product_j b_j - product_j c_j
  = sum_j (b_j-c_j)
      product_(k<j) b_k product_(k>j) c_k
```

and (3) prove

```text
0 <= exp(-mu*T)-g_P
   <= (mu^2/2) sum_j h_j^2
   <= (mu^2/2) T h_max,                             (4)
```

where `h_max=max_j h_j`. This is an enclosure of the exact continuous factor,
not a floating-point estimate. The endpoint consequences are the exact
identities

```text
||x(T)-x_P(T)||_inf
  = ||v||_inf [exp(-mu*T)-g_P],                     (5)

E_H(x(T)-x_P(T))
  = E_H(v) [exp(-mu*T)-g_P]^2,                     (6)
```

where `E_H(z)=(1/2)z^T H z`. Combining (4) with (5) and (6) gives the
reported `L_inf` and weighted error-energy upper bounds. The implementation
also stores rational lower and upper enclosures obtained from its rational
enclosure of `exp(-mu*T)`.

A proper positive subdivision replaces at least one step `h=a+b` by two
positive steps `a,b`, without moving an existing boundary. Its local factor
changes by

```text
(1-mu*a)(1-mu*b) - (1-mu*(a+b)) = mu^2*a*b > 0.    (7)
```

All untouched factors are positive, so (7) strictly raises `g_P` and strictly
lowers the exact factor error. At the same time,

```text
(mu^2/2)[(a+b)^2-a^2-b^2] = mu^2*a*b > 0,          (8)
```

so the quadratic factor bound and its nonzero `L_inf` and `H`-error-energy
consequences strictly improve. Repeating this argument proves the result for a
proper subdivision that splits several coarse intervals.

Finally, fix `W`, `nu`, `x_0`, the exact mode and `T`. For any family of
positive partitions satisfying `0 < mu*h < 1` and `h_max -> 0`, (4) tends to
zero. Equations (5) and (6) therefore prove convergence of the exact-real Euler
endpoints to (1). This is a mathematical partition-family theorem. It does not
certify a binary64 trajectory, numerical solver order, mixed-mode initial data,
directed or changing generators, glyph or REMESH dynamics, or full TNFR
stability.

[`certify_reversible_single_eigenmode_euler_reference`](../src/tnfr/physics/reversible_eigenmode_reference.py)
implements the theorem entirely over exact `Fraction` inputs and returns a
sealed `ReversibleSingleEigenmodeEulerReferenceCertificate`. One supplied
partition is sufficient for the theorem; when several are supplied they must
form a strict proper-subdivision chain, and only then is an observed strict
subdivision improvement reported. The public surface is declared exactly in
[`reversible_eigenmode_reference.pyi`](../src/tnfr/physics/reversible_eigenmode_reference.pyi)
and re-exported from `tnfr.physics`. The rational exponential routine requires
`mu*T <= 4096` solely to cap the integer-power exponent used by that enclosure.
This implementation limit is not a dynamical threshold and does not bound the
total bit size of arbitrary `Fraction` inputs or derived rational values.

[`test_reversible_eigenmode_reference.py`](../tests/physics/test_reversible_eigenmode_reference.py)
checks the hypotheses, both norm bounds, subdivision identities, scope flags
and proof sealing. The public
[`167_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/167_reversible_eigenmode_reference.py)
certifies both exact nonuniform modes of the nonregular path `P3`; its facade,
stub and report are checked by
[`test_reversible_eigenmode_reference_example.py`](../tests/physics/test_reversible_eigenmode_reference_example.py).

### Finite executor binding of the exact mode

The pure theorem now has a finite runtime adapter:
[`observe_executed_reversible_single_eigenmode_euler_reference`](../src/tnfr/physics/runtime_eigenmode_reference.py).
Its inputs are one or more already executed, individually sealed
`ExecutedPressureRefreshedFlowPartition` records. The adapter derives `W`,
`nu`, `x_0`, the ordered node support and every exact represented duration from
those records, requires them to define one fixed reference problem, and invokes
the exact theorem once for the resulting partition family. Thus it rejects a
changed support, conductance, capacity, initial field or total duration, as
well as a centered initial field that is not one exact positive eigenmode.
The inherited reference conditions also require `0 < mu*h < 1` for every
segment and a strict proper-subdivision chain when several partitions are
supplied.

Let `z_j` denote the exact rationalization of the represented boundary EPI,
`p64_j` the similarly rationalized stored pressure and

```text
p*_j = -L_rw z_j.
```

For each segment the observer separates two measured defects:

```text
rho_j = p64_j - p*_j,
eta_j = z_(j+1) - z_j - h_j diag(nu) p64_j.
```

The first is the binary64 pressure-realization residual. The second is the
held-input execution residual. Substitution into the nodal Euler step gives
the exact represented-value identity

```text
z_(j+1) = (I-h_j A) z_j + epsilon_j,
epsilon_j = h_j diag(nu) rho_j + eta_j.             (9)
```

These residual vectors are not assumed to lie in the initial eigenspace. If
`y_j` is the exact-real pressure-refreshed Euler reference from equation (2)
and `r_j=z_j-y_j`, then the observer uses the full generator matrix:

```text
r_0 = 0,
r_(j+1) = (I-h_j A) r_j + epsilon_j.                (10)
```

Equation (10), rather than multiplication by `1-mu*h_j`, is essential for
off-mode binary64 defects. The sealed partition row checks it against the
actual represented endpoint. It also subtracts the rational lower and upper
continuous endpoint enclosures coordinate by coordinate, producing signed
runtime-minus-continuous intervals and conservative exact lower and upper
bounds for the `L_inf` norm and `H`-error energy.

Every boundary must identify canonical binary64 pure-EPI pressure and every
segment must expose a trusted held-pressure Euler replay. An exact-affine-map
flag is retained separately and may be false: nonzero `rho_j` and `eta_j` are
the data measured by (9), rather than grounds for erasing the finite binding.
Each input partition has executor provenance, but the supplied family is
caller ordered and offline. Consequently the result proves neither common
causal execution provenance nor binary64 asymptotic/runtime mesh convergence,
solver accuracy or order, glyph/REMESH dynamics, repeated behavior, future
stability or full TNFR stability.

The exact public interface is declared in
[`runtime_eigenmode_reference.pyi`](../src/tnfr/physics/runtime_eigenmode_reference.pyi).
[`test_runtime_eigenmode_reference.py`](../tests/physics/test_runtime_eigenmode_reference.py)
checks the source binding, residual identities, off-mode full-matrix
propagation, norm enclosures, family compatibility and proof seals. The public
[`168_runtime_reversible_eigenmode_reference.py`](../examples/02_physics_regimes/168_runtime_reversible_eigenmode_reference.py)
reports three independently executed nonregular-`P3` partitions and keeps all
stronger runtime claims false.

## Switching-topology common-metric theorem

Let a finite family of connected symmetric graph regimes share one fixed node
set. Regime `r` has Laplacian `B_r`, capacity vector `nu_r` and metric

```text
h_r,i = d_r,i / nu_r,i.
```

In exact real arithmetic, if every `h_r` is a positive scalar multiple of one
vector `p`, the normalized metric and its conserved consensus value are common
to every regime. For

```text
V_p = (1/2) sum_i p_i (x_i-c)^2,
```

each active regime satisfies `V_p' <= -2 lambda_r V_p`, where `lambda_r` is
the first positive generalized eigenvalue of `(B_r, diag(h_r))`. Consequently,
arbitrary switching obeys

```text
V_p(t) <= exp(-2 min_r(lambda_r) t) V_p(0).
```

[`verify_switching_diffusion_stability`](../src/tnfr/physics/structural_diffusion.py)
applies the represented-coefficient theorem rather than assuming the ideal
cancellation survives rounding. It tests exact projective equality of the
displayed binary64 metric vectors by rational cross-products, constructs the
rational quotient dissipation for every materialized regime and takes the
minimum certified rate. It separately reports eigensolver estimates, normalized
metric mismatch, caller-tolerance proximity and exact weighted-mean
preservation by regime. Exact common-metric equality, positive certified
quotient bounds and exact `A_r 1=0` in every regime promote
arbitrary-switching disagreement contraction;
tolerance alone is a numerical diagnostic. Convergence to the *initial*
displayed weighted mean additionally requires the separate common
mean-preservation flag. A path, cycle and
complete graph with capacities adjusted so `d_i/nu_i` remains proportional
pass the same exponential envelope under a switched exact flow. Keeping
capacity fixed while degrees change generally fails the common-metric test.
That failure delimits this theorem; it does not prove instability.

The switching object's `equilibrium_value`, `lyapunov_value` and
`derivative_upper_bound` describe the first supplied snapshot in ordinary
binary64 arithmetic with the displayed normalized metric. They are diagnostics,
not inputs to the proof. Hybrid composition and the sampled S16 certificate use
the rationalized `reference_metric_weights`; S16 recomputes the corresponding
weighted center at every snapshot. Only the certified rational rate is consumed
as flow decay.

The theorem covers changing edge sets and `weight` conductances on one fixed node
set. Node creation, deletion, directed transport and the
radial/annular/multinodal classification require separate models. The next
section treats only explicitly declared affine EPI resets.

## Affine-reset gain and hybrid-word theorem

Let `h` be the positive metric shared by the continuous regimes, set
`H=diag(h)`, and define the `H`-orthogonal consensus projector

```text
Q = I - 1 h^T/(h^T 1),
V(x) = (1/2) ||Qx||_H^2.
```

Consider one declared affine EPI reset

```text
J(x) = A x + b.
```

A finite global multiplicative bound `V(J(x)) <= gamma V(x)` exists for every
`x` **if and only if**

```text
Q A 1 = 0,       Q b = 0.
```

These identities say that `A1` and `b` are uniform vectors, so the affine map
preserves the consensus subspace. Necessity follows immediately from a
zero-energy input: if `Qb != 0`, use `x=0`; otherwise, if `QA1 != 0`, use
`x=1`. In either case `V(x)=0` and `V(J(x))>0`, excluding every finite
multiplicative gain. For sufficiency, write `x=Qx+c1`. The two identities give

```text
QJ(x) = Q A Qx,
gamma_sharp = ||H^(1/2) Q A Q H^(-1/2)||_2^2.
```

The sharp formula is an exact mathematical characterization. Its binary64 SVD
evaluation is only a diagnostic. The executable theorem instead works with
rational arithmetic on the represented coefficients. It always forms the
weighted-Frobenius fallback

```text
Gamma_F = sum_ij (h_i/h_j) (QAQ)_ij^2 >= gamma_sharp.
```

`Gamma_F` is the squared weighted Frobenius norm. The certificate then tightens
this value on the disagreement quotient without using a floating eigensolver.
If `QAQ=cQ`, the gain is exactly `c^2`. One- and two-dimensional quotient
problems use exact rational reduction, with a rational upper enclosure for the
quadratic root in the two-dimensional case. Other small quotient problems use
exact positive-semidefinite tests and rational bisection of the generalized
Rayleigh quotient. Larger non-scalar problems retain `Gamma_F`. The resulting
`Gamma_Q <= Gamma_F` is therefore always a proved rational upper bound; its
published binary64 value is rounded upward. In particular, the identity map has
`Gamma_Q=1` instead of the dimension-dependent Frobenius value `N-1`.

A failure of either bound to prove contraction does not show that the sharp
gain is at least one. A caller-declared bound is certified when it is no smaller
than `Gamma_Q`; the stronger legacy diagnostic records independently whether it
also dominates `Gamma_F`. A declared value never replaces the internally
derived compositional bound.

Now interleave affine resets `J_k` with continuous common-metric diffusion for
durations whose sum is `T_flow`. If the represented diffusion certificate gives
the rigorous lower-rate bound `V' <= -r_cert V` and every reset has a certified
finite bound `Gamma_k`, then

```text
V(T) <= exp(-r_cert T_flow) product_k(Gamma_k) V(0).
```

The implementation evaluates the budget in log space. A declared finite word
has `len(jumps)+1` nonnegative flow durations: before the first reset, between
resets, and after the last reset. Mathematically it contracts disagreement
whenever

```text
sum_k log(Gamma_k) - r_cert T_flow < 0.
```

The executable decision does not use rounded logarithms or a caller tolerance.
It forms an exact rational lower bound for the represented flow decay. Each
precise rational quotient-gain bound remains certificate data; for log-space
composition alone, the implementation rounds that gain upward to an exact
32-bit-significand dyadic rational. This conservatively bounds the gain while
limiting every logarithm-series input. The actual factors remain visible in
`exact_log_composition_gain_factors`, separately from the precise product in
`exact_cumulative_jump_energy_gain_bound`. A rational upper enclosure for each
resulting logarithm is summed, and strict negativity of that rational upper
budget is the decision. The published multiplier uses an upward-rounded
rational exponential enclosure. The compatibility field
`log_contraction_decision_margin` is zero; `tolerance` affects only numerical
diagnostics.

Duration inputs, including integers and `Fraction` values, are first
materialized as binary64; the exact budget then rationalizes those represented
duration values. It therefore certifies the declared executable timeline,
rather than the pre-conversion source rational. Nonzero durations that
underflow to zero are rejected. `build_operator_event_schedule` centralizes the
same `m` jumps/`m + 1` intervals convention, exact prefix offsets and explicit
ordering for coincident events. Absolute float timestamps are display fields and
are never subtracted to obtain duration. The separate fixed-flow diagnostic uses
the exact quotient-rate lower bound plus rational log/exp enclosures to decide a
sufficient duration for a requested energy fraction. Neither interface executes
the operator word or the continuous solver. The finite-horizon multiplicative bound only
needs valid flow and reset proofs. Asymptotic fields remain `None` unless
`repeat_schedule=True`; a repeated schedule must have positive total flow time
to exclude Zeno accumulation. Under those conditions a negative certified
budget certifies exponential disagreement decay at the word boundaries.
Preservation of the initial weighted consensus is a separate property:

```text
h^T A_k = h^T,       h^T b_k = 0
```

for every reset. The represented continuous flow must also preserve that same
weighted mean exactly. When all of those identities hold, the repeated word
converges to the initial `h`-weighted uniform state. They are a sufficient exact
route to that particular limit, not a necessary condition for every affine
consensus dynamics to possess some other fixed point.

[`certify_affine_epi_jump_gain`](../src/tnfr/physics/hybrid_operator_stability.py)
returns the exact subspace and mean-preservation decisions, the sharp numerical
estimate, the rational quotient bound, the Frobenius fallback and a
zero-to-positive witness when finite gain is impossible. Its proof stamp covers
the decisive affine inputs and conclusions, so ordinary replacement or mutation
cannot promote a forged certificate property.
[`compose_hybrid_epi_stability`](../src/tnfr/physics/hybrid_operator_stability.py)
combines those certificates with either the fixed heterogeneous-flow theorem or
the exact-common-metric switching theorem. It requires the same persistent node
order and exact projective metric; tolerance proximity cannot join different
Lyapunov functions. It reconstructs each jump proof from the supplied affine
inputs instead of trusting copied result fields. Flow proof stamps detect
ordinary stale, replaced or mutated payloads; they are consistency checks, not
an authentication or security boundary.

The reproducible
[`162_hybrid_epi_stability.py`](../examples/02_physics_regimes/162_hybrid_epi_stability.py)
uses a two-node heterogeneous flow with the exact represented metric `h=(1,3)`.
It exhibits three distinct boundaries: diffusion overcomes a certified
weighted-mean-preserving disagreement amplification, although the represented
flow need not conserve that same displayed mean exactly; a uniform translation
contracts disagreement while shifting the consensus coordinate; and a local
offset produces the exact `V=0 -> V>0` obstruction. These are declared affine
maps, not identifications of the named operators' runtime implementations.

### Reception and Resonance runtime realizations

Reception (EN) and Resonance (RA) now supply the first two catalog bridges at
both the local and all-target stage boundaries. The Reception graph-backed
handler reads one target, forms the unweighted binary64 `fmean` of its runtime
neighbours, blends with the configured factor `m`, reads the target in the
canonical real-scalar EPI domain, and applies structural clipping. Raw finite
real values and uniform-real BEPI embeddings retain their signed scalar value.
Genuinely nonuniform or complex BEPI values have a maximum-component magnitude
for generic read-only diagnostics, but EN rejects them before proposing a blend.
The runtime is not globally the matrix multiplication used above: the mean and
blend round in separate stages, and soft or active hard clipping is nonlinear.

On a declared fixed connected undirected support, the EN bridge therefore
requires a uniform real scalar embedding, `0<=m<=1`, values inside the
configured bounds, and inactive runtime hard clipping at the audited snapshot.
No sign restriction is needed. In exact real arithmetic the neighbour mean and
the second blend are convex combinations, so hard clipping is automatically
inactive throughout the declared bounded domain. The ideal map for target `i`
is

```text
A_i = I + m e_i (p_i^T-e_i^T),
p_i,j = 1/|N_i| for j in N_i, and 0 otherwise.
```

It preserves constants because `A_i 1=1`. It does **not** preserve a positive
weighted mean as a global functional for any nonisolated target and `m>0`:

```text
h^T A_i-h^T = m h_i (p_i^T-e_i^T) != 0.
```

The EN bridge also constructs the coefficient matrix produced by the
individual binary64 coefficient operations. That represented map is nested
into the affine jump theorem only if its row sum equals one as an exact rational
identity. Agreement of the actual two-stage runtime result with either matrix
is reported only for the supplied snapshot; it does not assert global binary64
affinity.

RA reuses the same unweighted blend kernel but first filters graph neighbours
individually by the circular U3 condition
`|wrap(theta_i-theta_j)| <= Delta_phi_max`. Incompatible neighbours contribute
to neither its EPI mean, circular phase mean, nor capacity-amplification
trigger. The runtime requires a non-isolated target with at least one compatible
neighbour; its graph phase limit must be finite and in `[0,pi/2]`, so a local
configuration may tighten but cannot relax the canonical gate. The EPI mix,
capacity amplification, and phase coupling factors must respectively satisfy

```text
0 <= RA_epi_diff <= 1,
RA_vf_amplification >= 0,
0 <= RA_phase_coupling <= 1.
```

These checks, scalarization, the proposed finite post-frequency, and the
identity gate all precede mutation. Resonance may change the scalar EPI through
the convex blend. “Preserves identity” instead means that a proposed strict
negative-to-positive or positive-to-negative crossing is rejected and an
established nonempty `epi_kind` is retained exactly. Exact zero is a neutral
sign boundary, and an absent kind may be initialized by the existing runtime
convention. Sign and kind are independent conditions; satisfying one cannot
repair failure of the other.

The RA certificate separates four layers:

1. the ideal-real convex EPI blend;
2. the affine matrix assembled from represented binary64 coefficients;
3. the actual two-stage binary64 target proposal at the audited snapshot; and
4. the accepted identity-gated runtime snapshot.

Only the second layer is eligible for the affine-jump theorem, and only when
its exact rational consensus-row identity and the declared scalar,
fixed-support, clipping-free domain pass. The accepted layer can equal the
pre-state after an identity rejection even though the proposal is nontrivial.
The two-stage mean/blend rounding, clipping, identity and U3 gates, circular
phase map, and conditional capacity change rule out a global binary64 affinity
claim for the runtime operator.

A successful RA capacity boost changes `nu_i` at one target and therefore
generally changes the diffusion metric `h_i=d_i/nu_i`. The certificate still
constructs the fixed post-RA heterogeneous diffusion theorem and audits the
represented jump in that post-RA metric. It withholds an arbitrary pre/post
switching theorem unless the represented metric vectors are exactly
proportional. Optional recovery composes the represented jump with this fixed
post-RA flow; the displayed break-even duration is only an estimate, while the
Boolean recovery decision uses the exact hybrid composer.

The node-local EN and RA writes do not recompute the stored pressure field.
If the pre-event pure-EPI pressure is `p=-L_rw x` and an accepted target update
changes EPI by `delta`, retaining `p` gives the exact post-event manifold defect

```text
p + L_rw(x+delta e_i) = delta L_rw e_i.
```

On connected positive conductance this defect vanishes exactly iff the update
is trivial. Pressure refresh is therefore required before the next segment is
interpreted as pure-EPI flow. The shared all-target executor invokes the graph's
configured `compute_delta_nfr` callback after structural commit and keeps that
refresh inside the same rollback boundary. The exact support theorem and the
observed binary64 defect norm are separate certificate fields.

[`certify_reception_epi_realization`](../src/tnfr/physics/reception_realization.py)
and
[`certify_resonance_epi_realization`](../src/tnfr/physics/resonance_realization.py)
perform these audits without mutating the graph.

[`certify_all_target_neighbor_stage`](../src/tnfr/physics/network_stage_stability.py)
assembles every local EN or RA row from one stage snapshot and records a finite
repeated trace. For a convex factor and an initial field inside finite hard
bounds, every ideal-real row is a convex combination; the interval is forward
invariant, so hard clipping remains inactive for arbitrarily many ideal-real
stages. A fixed-map EN theorem additionally requires declared fixed support. RA
also requires declared fixed U3 neighbour sets and a forward-invariant common
sign/kind domain. The certificate checks the observed neighbour sets, maps,
identity admissions and pre/post diffusion metrics at every requested stage.

The represented fixed map is promoted only when its rational row sums preserve
the consensus subspace exactly. Its repeated disagreement bound is
`Gamma_Q^k`; weighted-mean drift and RA metric drift remain separate outputs.
Regression tests compare every finite structural-trace step with the public
all-target runtime, but no global binary64 runtime-affinity theorem is asserted:
the two-stage
`fmean`/blend evaluation can differ from matrix multiplication even when every
snapshot agrees within tolerance. Soft or active clipping, changing RA gate
sets, identity rejection and metric changes therefore produce explicit
observations or abstentions rather than being hidden inside the affine claim.

[`compose_neighbor_stage_diffusion_stability`](../src/tnfr/physics/network_stage_stability.py)
closes the finite one-stage boundary. It accepts exactly one certified
all-target EN or RA stage and a binary64-representable strictly positive flow
duration. It rebuilds every reconstructible local domain condition, validates
node order, represented coefficients, exact row algebra, nested proof stamps and
the post-stage metric, then composes the stage map with the fixed post-stage flow
using `repeat_schedule=False`. The raw scalar EPI embedding cannot be revisited
after the graph has been discarded and remains an explicit carried observation.
Honest support, clipping or gate failures return an abstention; inconsistent
records are rejected.

The resulting finite disagreement bound and strict-contraction decision do not
assert weighted-mean preservation. The exact represented-map identity, the
observed runtime mean shift and any RA pre/post metric change are separate
fields. Linearity of pure-EPI pressure allows the single-target defects from the
common snapshot to be summed into an all-target binary64 diagnostic. This is not
the local exact refresh-iff-nontrivial theorem, and RA phase or capacity may
require a full runtime pressure refresh even when EPI does not change. The bridge
therefore leaves stored-pressure refresh, global binary64 runtime affinity and
repetition of the stage-flow schedule explicitly uncertified.

The scheduler result is broader than this theorem. AL, IL, OZ, SHA, VAL, NUL,
THOL, ZHIR, NAV, UM and REMESH join EN/RA in the implemented thirteen-stage
two-phase set: every
target proposal is derived from one stage-start snapshot and the complete stage
is failure-atomic. Their committed primary channels are target-order invariant
before the opaque pressure-refresh callback. IL contracts signed pressure
magnitude and locks phase through one direct/stage kernel; its canonical
structural `C(t)` fields remain distinct from auxiliary pressure-dispersion
telemetry. OZ reduces snapshot-bound local and propagated pressure and advances
per-node RNG progress independently of target order. ZHIR binds phase and
structural-acceleration telemetry to
snapshot-bound temporal evidence and U4 context; AL/SHA share one timestamp
across the stage. NAV binds `nu_f`, phase, `DeltaNFR` and per-node RNG progress
to the snapshot, resolves a missing graph seed within the transaction and uses
one shared latency-observation instant. The seed is restored on rollback and
persists on success. Ordered lifecycle, audit/telemetry and monitor streams
retain requested target order. IL warnings are the final transactional effect;
cache state and the opaque pressure refresh are excluded. Identity-bearing
caches remain tied to their live graph and node objects. UM uses a deterministic
snapshot-rank circular-displacement reducer, final U3 validation and
deterministic functional-link coalescing; this is an execution policy rather
than a Lyapunov result. THOL allocates cross-parent child-ID collisions and
commits `d2EPI`, `DeltaNFR`, child nodes, `sub_nodes`, `sub_epis` and
graph `hierarchy` in snapshot-node rank after detached validation. REMESH leaves structural channels unchanged and deduplicates one advisory event per telemetry step.

For AL/SHA/VAL/NUL/ZHIR/NAV, the shared pointwise stage executor can request a
certificate computed from the same detached snapshot and frozen proposals used
by its commit. A successful `NetworkStageResult` conditionally proves exact
runtime EPI realization, a consensus-preserving affine gain in the pre-flow
metric, and an aligned pre/post diffusion metric. NUL's pressure channel
remains separate. Unsupported, empty and grammar-replaced stages reject the
request before live mutation. This seam does not certify future repetition;
IL, OZ and THOL receive no affine gain from their execution contracts.

OZ nevertheless has an immutable all-target execution contract: local actions
and outgoing propagation read the stage snapshot, and overlapping incoming
increments are summed with `math.fsum` in snapshot-node rank. Its local
pressure-magnitude postcondition is distinct from the final signed field, where
positive propagation can partially cancel a negative pressure. REMESH has an
immutable advisory proposal and deterministic graph-event merge; it does not
invoke the separate delayed EPI operation. Multi-target IL historically used
the sequential schedule; the canonical word path now uses its immutable
proposal.

Canonical names remain metadata outside such a realization proof. Neither a
U2 role nor a legacy policy multiplier supplies a runtime gain. Branching,
richer BEPI, arbitrary multichannel pressure laws, topology or history mutation,
unsupported glyph gains, unrestricted mixed operator words and global binary64
runtime affinity remain outside this theorem. The event runtime's finite
composition applies only to exact represented maps bound to one observed trace
with a common metric. Repetition is covered only for the explicitly certified
fixed ideal-real or represented EN/RA map. The RA bridge also
requires finite positive capacities on a fixed connected undirected
positive-conductance support; it does not certify finite-step integration or a
general phase/nonlinear trajectory.

## Directed pressure-transient criterion

For a fixed strongly connected directed graph with outgoing random-walk
Laplacian `L` and scalar structural frequency absorbed into structural time,
the pure-EPI pressure is

```text
p = DeltaNFR_epi = -Lx,        p' = -Lp.
```

If the graph has one consensus mode, `range(L)=ker(pi^T)`. Let `V` be an
orthonormal basis of this invariant subspace and `L_sub=V^T L V`. Then

```text
max_t ||exp(-t L_sub)||_2
```

is precisely the worst-case Euclidean amplification of an attainable pressure
state. The logarithmic norm

```text
mu_2(-L_sub) = lambda_max((-L_sub-L_sub^T)/2)
```

gives the exact qualitative criterion: `mu_2<=0` implies contraction for all
times, while `mu_2>0` gives a direction with immediate norm growth. By
contrast, `alpha(-L_sub)<0` proves asymptotic decay only.

[`measure_nonnormal_pressure_prediction`](../src/tnfr/physics/nonnormal_prediction.py)
implements the matrix certificate and a finite-window peak scan. Its companion
benchmark fixes a seeded family of 16 ten-node directed graphs and an
even/odd calibration split. Four graphs show sampled pressure amplification;
all 16 have stable spectra. The logarithmic-norm sign agrees with every case,
whereas the stable-spectrum rule misses the four bursts. The sign theorem is
exact under the stated fixed linear hypotheses. Peak magnitudes, correlations
and benchmark accuracy are measurements of this finite family only.

This result does not select the canonical U2 metric for directed graphs. It
also excludes changing support, nonlinear operators, time-dependent
heterogeneous capacity and claims about total coherence `C(t)`.
