# Exact scale, coherence-geometry and bridge results

This note records restricted results for core research lines S8, S9, S11, S12,
S14, S15 and S16. Each exact claim states its domain and executable certificate.
None establishes closure of all 13 operators or a coupled global TNFR geometry.

Repository graphs keep two edge channels distinct. `weight` is transport
conductance: it defines `W`, diffusion, Dirichlet energy, quotient generators
and the conductance coordinate of the fixed-topology state metric; it defaults
to one.
`length` is the independent shortest-path channel used by structural potential
geometry and is the second edge coordinate of that metric. An explicit `length`
wins there; if it is absent, path geometry falls back to `weight` for
compatibility, and then to one if neither is present.
Consequently `length` does not change EPI diffusion, while `weight` also changes
legacy path geometry unless `length` is declared. Models in which coupling and
distance differ should declare both attributes explicitly. The executable
policy is centralized in
[`_edge_semantics.py`](../src/tnfr/physics/_edge_semantics.py).

## 1. Pure-EPI coarse-graining

On a fixed symmetric graph with positive capacity, write

```text
x' = -A x,                 A = H^-1 B,
H = diag(h_i),             h_i = d_i/nu_i.
```

For a node partition, let `P` copy one macro value to every node in a block.
The reversible block average is

```text
R = (P^T H P)^-1 P^T H,   R P = I.
```

The quotient conductance between two blocks is the sum of all micro
conductances crossing between them. If `h_bar=P^T h`, its generator is

```text
A_bar = diag(h_bar)^-1 B_bar.
```

This is again a TNFR pure-EPI nodal equation. Its macro capacity is
`nu_bar_a=d_bar_a/h_bar_a` and macro EPI is `R x`.

### Exact closure condition

The macro observable is autonomous for every micro state exactly when

```text
R A = A_bar R.
```

Equivalently in this reversible construction, the block-constant subspace is
invariant:

```text
A P = P A_bar.
```

Thus exact coarse-graining is an intertwining structural morphism, not a
fourteenth operator. The information removed by the quotient has dimension
`N-m`, where `m` is the number of blocks. A nonzero intertwining defect measures
the influence of those unresolved within-block modes; the effective dynamics
then needs memory or additional macro variables and the nodal equation is not
closed in `R x` alone.

[`certify_epi_coarse_graining`](../src/tnfr/physics/structural_morphism.py)
constructs the quotient, reports both closure defects and a clearly named
within-tolerance decision, then delegates morphism classification to the
existing certificate. The exact theorem is the zero-residual identity; a caller
tolerance cannot make a nonzero defect algebraically exact. An equitable
reflection partition of a path has zero residual to machine precision. A path partition
whose middle block mixes boundary and interior nodes is an executable
counterexample.

### Consequence for S9

The pure-EPI diffusion channel is a fixed family under every exact quotient:
its generator remains `diag(nu_bar)L_rw,bar`. This does not show that Emission,
Coherence, REMESH or the other nonlinear operators close under the same map.
Operator RG flow therefore remains open beyond this one channel.

### Generic fixed-vector operator test

For any declared micro map `F`, macro map `F_bar`, projection `R` and right
inverse lift `P`, two logically independent identities are relevant:

```text
R F = F_bar R,        F P = P F_bar.
```

The first makes the projected dynamics autonomous for every micro state; the
second keeps lifted macro states inside the lifted subspace. Exact strong
closure requires both. A deterministic counterexample shows that projected
closure need not imply lift invariance.

[`certify_operator_quotient`](../src/tnfr/physics/operator_quotient.py) checks
both identities globally for matrices and names every Boolean decision as
within the declared numerical tolerance. For nonlinear callables it reports only
sampled residuals and repeated-evaluation consistency on declared macro and
micro probes, including the defect
between two micro states with the same projection. A cubic componentwise map
closes on block-constant probes yet depends on unresolved within-block fibers,
providing an explicit nonlinear obstruction. Graph mutation, operator history
and nested-EPI changes are rejected as outside a fixed-dimensional vector-map
test. Thus the certificate advances S9 without claiming closure or completeness
of the 13-operator catalog.

## 2. Geometry forced by canonical coherence

For the signed local chart `p=DeltaNFR`, `v=dEPI`, the constitutive kernel is

```text
C(p,v) = 1 / (1+|p|+|v|).
```

For `0<c<1`, its exact level set is

```text
|p|+|v| = 1/c-1.
```

It is an L1 diamond. It is smooth on each open edge and nondifferentiable at
the four axis vertices. Its Euclidean distance from equilibrium is not fixed:
it ranges from `r/sqrt(2)` to `r`, whereas its L1 distance is exactly
`r=1/c-1`. At `c=1` the level collapses to `(0,0)`.

[`coherence_level_set_geometry`](../src/tnfr/physics/coherence_geometry.py)
exposes this exact geometry. The result rejects the assumption that canonical
coherence alone supplies a smooth Riemannian manifold. It naturally supplies
an L1 gauge in the local constitutive chart. A smooth information metric on the
full graph state would require an additional modeling choice.

### Fixed-topology structural-state metric

One such explicit choice is now available within a declared topology and label
class. For each node, form the five-nodal-channel state

```text
q_i = (EPI_i, nu_f_i, phase_i, DeltaNFR_i, dEPI_i),
```

and assign each edge both its `weight` conductance coordinate (unit when
`weight` is absent) and its effective structural-length coordinate (`length`,
then the compatibility fallback `weight`, then one). Declare one finite positive
reference scale per channel, use circular geodesic distance for phase, and
minimize the scaled Euclidean product over node and edge coordinates across all
topology- and declared label-preserving graph isomorphisms. The compatibility
API lets an omitted `edge_length` scale reuse `edge_conductance`; dimensional
studies should declare both. Because a finite graph has finitely many
isomorphisms and they act by isometries, the minimum is an exact metric on the
resulting state-isomorphism classes. The certificate also reports the two
`L_infinity` residuals of
`dEPI=nu_f*DeltaNFR`; it does not project inconsistent inputs onto the nodal
equation.

[`fixed_topology_structural_state_distance`](../src/tnfr/physics/structural_state_distance.py)
implements this quotient metric for finite simple graphs. It preserves phase
wrapping and node relabeling, accepts explicit node and edge labels, validates
finite nonnegative conductance and structural length, and rejects nonisomorphic
supports, mixed
direction and multigraphs. It reports every distance minimizer, uses an explicit
lexicographic component refinement, and withholds a node mapping when that
refinement remains ambiguous. The exhaustive
isomorphism search can be factorial. Cross-topology edit costs, nested EPI
identity and operator-history geometry remain open, so this is a structural
state metric rather than a completed global information geometry.

The structural-length coordinate is a numeric cost. A caller that requires
identical path geometry may additionally include `"length"` in
`edge_label_attributes`, which makes exact attribute equality part of the
admissible-isomorphism test rather than merely charging a nonzero distance.

## 3. Dissipative-symplectic direct product

Let `z` be the `4N`-dimensional harmonic substrate coordinate and `x` the EPI
field. Define

```text
X = (z,x),
H = (1/2)||z||^2,
V = (1/2)x^T Bx,
J = diag(J_sub,0),
G = diag(0,diag(nu_i/d_i)).
```

Then

```text
X' = J grad(H) - G grad(V)
```

simultaneously reproduces the specified auxiliary harmonic-substrate flow and
fixed symmetric EPI diffusion. The degeneracy identities

```text
J grad(V) = 0,             G grad(H) = 0
```

give `H'=0` and `V'<=0`. This is an exact metriplectic-style **direct product**.
[`verify_metriplectic_product`](../src/tnfr/physics/metriplectic.py) checks the
antisymmetry, positive semidefiniteness, degeneracies and both vector-field
residuals.

The graph's stored `DeltaNFR` initializes part of the auxiliary coordinate `z`
through `Phi_s` and `J_DeltaNFR`. The dissipative EPI block nevertheless
constructs its pressure independently as `p_epi=-D^-1 Bx` from the `weight`
conductance and supplied EPI. It does not replace the stored value. The
certificate returns
`stored_pressure_consistency_residual=||DeltaNFR_stored-p_epi||_2` and the
separate relative-tolerance decision `stored_pressure_matches_epi_channel`.
Neither is included in `is_decoupled_metriplectic_bridge`: the block-product
identity can pass while stored pressure contains other channels or is otherwise
inconsistent with pure-EPI diffusion. A caller asserting a pure-EPI graph state
must therefore require both the bridge Boolean and the pressure-consistency
Boolean. This separation is exercised by
[`test_metriplectic_product.py`](../tests/physics/test_metriplectic_product.py).

The cross blocks are zero, so the result does not derive how a pulse changes
EPI relaxation or how dissipation feeds back into the substrate. A coupled
bridge must specify nonzero cross tensors while retaining the degeneracy laws;
that is the remaining S12 problem.

## 4. Inverse identifiability

### Contract snapshot

[`contract_identifiability_certificate`](../src/tnfr/operators/operator_contracts.py)
partitions the catalog using only declared executable contract features. With
the richest non-tautological tuple `(channel, direction, scale, context)`, eleven
operators are singletons and one class contains both Silence and Contraction.
Both are node-scale decreases of `nu_f` observed at network context.

This is an exact negative result for instantaneous contract identification. It
does not say their trajectories are always identical: Silence targets latency,
whereas Contraction also densifies pressure. Resolving them requires temporal
state changes or a quantitative postcondition observer. Catalog coverage is
therefore distinct from catalog identifiability and from universal catalog
completeness.

### Quantitative one-step signatures

[`probe_canonical_operator_identifiability`](../src/tnfr/physics/temporal_identifiability.py)
enumerates the 13 operators from the current contract catalog and applies each
one to a fresh deterministic heterogeneous path graph at the known target node
`n_nodes//2`. Its features
contain the target-node change, the across-node mean and RMS change in the four
raw channels EPI, `nu_f`, `DeltaNFR` and wrapped phase; the corresponding target,
mean and RMS change in nodal velocity `nu_f*DeltaNFR`; and node/edge-count
changes. Operator names, glyphs and contract categories are evaluation labels
and never enter the feature matrix. Executed history is checked separately to
reject a fallback mislabeled as the requested operator.

With the declared default three probes, all 13 rows are distinct after the
declared decimal quantization. Matrix rank and affine rank are 12. Row
distinctness exactly partitions this finite generated hypothesis matrix; rank is
only a numerical diagnostic. Silence and Contraction separate because both
reduce `nu_f` while only Contraction changes EPI on these probes. The result is
measured on this finite family.

This is closed-set classification against the fixed current catalog and a known
target, not target localization or operator discovery. Although mean and RMS
summaries of the raw changes are included, the protocol does not invert from
aggregate TNFR telemetry alone: `C(t)`, Si, phase synchronization and the tetrad
are absent, and target-indexed raw coordinates are present. Identification from
those aggregate read-outs, localization of an unknown target, unseen states,
compositions and arbitrary grammar words remain open.

For any declared positive feature scales, let `delta` be the minimum pairwise
distance between these finite prototypes in the scaled L-infinity or L2 norm.
The triangle inequality gives an exact robustness statement: an additive error
strictly smaller than `delta/2` cannot cross a nearest-prototype boundary. The
noise-margin certificate compares the supplied binary64 values through exact
rational arithmetic, rounds the minimum distance downward and halves that
lower bound. It therefore reports a conservative radius and marks midpoint
ties as uncertified. Stable L2 accumulation avoids false underflow, large
declared feature scales avoid false subtraction overflow, and unrepresentable
scaled separations are rejected. The default operator probe attaches a
unit-scaled L-infinity margin after applying
the same decimal quantization as its row partition. Those unit scales define a
coordinate convention, not a calibrated sensor or process-noise law. This
handles bounded perturbations around the fixed prototypes only. It does not
provide a stochastic noise law or extend the prototypes beyond the declared
states.

## 5. Executable S16 endpoint certificate

[`certify_core_research_integration`](../src/tnfr/physics/core_research_integration.py)
turns the restricted S16 intersection into one inspectable endpoint
certificate. It accepts two independently frozen states, one partition and
explicit `StructuralChannelScales`. The inputs must be finite undirected simple
graphs with exactly the same node identifiers and bare edge support. At each
endpoint the effective positive-conductance graph must also be connected, every
capacity must be positive and frozen, and all raw state channels required by the
constituent certificates must exist. Identical bare support alone is therefore
insufficient when zero `weight` disconnects effective conductance. Conductance,
structural length and state values may differ between endpoints.

At each endpoint it runs the fixed heterogeneous pure-EPI stability certificate,
the graph-specific full-potential EPI reconstruction certificate and the
reversible pure-EPI partition certificate. It also computes the declared
fixed-topology structural-state distance between the endpoints. Finally, it
checks two state-consistency conditions independently at each endpoint: stored
`DeltaNFR` must match `-L_rw EPI`, and stored `dEPI` must match
`nu_f*DeltaNFR`, after scaling by the declared pressure and EPI-rate scales.

The result exposes all constituent certificates, raw and scaled consistency
residuals, `numerical_conditions` with fifteen named entries, and
`failed_conditions`. `joint_numerical_conditions_pass` is true only when every
entry passes. Numerical rank, closure, balance and consistency decisions use the
single declared relative tolerance; the exact metric-on-isomorphism-classes
entry remains a structural Boolean. A nonclosing partition, a pure-EPI pressure
mismatch or an independent nodal-equation mismatch blocks joint promotion while
leaving the other evidence inspectable; changed support is rejected before
composition. These positive and negative paths are executable in
[`test_core_research_integration.py`](../tests/physics/test_core_research_integration.py).

A passing Boolean demonstrates simultaneous endpoint membership in this shared
restricted numerical hypothesis class under the declared tolerance. It does not
certify a trajectory or persistence between the endpoints. Phase dynamics,
nonlinear or multichannel pressure,
operator histories and words, S15 inverse identification, REMESH/nesting,
changing support/topology and nonzero dissipative-symplectic coupling remain
open. The exact L1 coherence level-set result also remains a separate theorem;
the current S16 Boolean does not consume it. Explicit `length` values contribute
to the numeric state distance; they may still differ unless the caller includes
`"length"` among the exact edge-label constraints.

## 6. Time-resolved S16 boundary

[`certify_core_research_trajectory`](../src/tnfr/physics/core_research_trajectory.py)
extends the endpoint intersection to a finite ordered sample path without
claiming unobserved interpolation. It retains persistent node identifiers and
fixed bare edge support, applies the endpoint certificate to every adjacent
pair, and checks the left-explicit update

`EPI^(k+1) - EPI^k = dt_k diag(nu_f^k) DeltaNFR^k`

in scaled L-infinity norm. Stored `dEPI` remains an independent endpoint channel;
it is not substituted for the vector field in this step test.

Two extra stability conditions prevent a merely self-consistent data sequence
from being called stable. First, each timestep must satisfy the modal
explicit-Euler condition for its left-hand transport regime. Its stationary-mode
resolution uses a separate dimensionless tolerance relative to the fastest
decay rate; the EPI residual tolerance is not reused as an absolute frequency
cutoff. Second, all sampled regimes must share the exact projective `d_i/nu_i`
metric required by the common switching theorem. The implementation evaluates
the resulting common quadratic at every supplied state. At each snapshot it
recomputes that metric's instantaneous weighted projection onto the consensus
subspace before evaluating disagreement energy; it does not freeze the first
snapshot's consensus coordinate. This is necessary because the represented
binary64 generator can contract disagreement in the displayed metric without
preserving that metric's weighted mean as an exact rational identity. The
certificate limits both each
positive increment and the cumulative positive variation over the whole path
by one declared EPI-squared scale budget. This prevents individually small
increases from accumulating without bound as samples are added.

The Euler residual and both local and cumulative Lyapunov-budget decisions
compare exact rational values of the represented binary64 inputs with the exact
rationalization of the caller tolerance. Their exposed float magnitudes are
diagnostics only and do not decide a boundary case.

The analytic switching theorem covers arbitrary
piecewise-constant continuous solutions among this finite regime family. The
snapshot certificate only checks its supplied Euler steps: it does not prove
that the samples lie on such a continuous solution, identify a switching law
between them, or cover an unseen regime.

The companion refinement comparison certifies both paths before comparing
them. It separates the strict path tolerance from the coarser agreement
tolerance, requires every coarse time to match exactly one fine time, and
requires a smaller fine-grid maximum step. Direct EPI differences on persistent
ids determine agreement. Quotient structural distance is exposed only as a
diagnostic because a different minimizing isomorphism at each time is not a
node trajectory. The joint result also requires the caller to set
`same_dynamics_declared=True`. That recorded assertion closes the previous
semantic gap in which identical equilibrium samples from different generators
could be promoted as a refinement pair; snapshots cannot independently verify
the declaration.

The deterministic
[`161_core_research_trajectory.py`](../examples/02_physics_regimes/161_core_research_trajectory.py)
uses two nested stable Euler meshes for one non-equilibrium fixed generator and
also evaluates the analytic fixed-generator semigroup numerically through the symmetric similarity
`H^(-1/2) B H^(-1/2)`. The fine solution is closer at every noninitial common
time in this experiment. This finite result does not prove numerical
convergence or its order. Phase evolution, nonlinear/multichannel pressure,
operator histories, REMESH/nesting, adaptive/event timesteps, changing support
and nonzero metriplectic cross-coupling remain open.
