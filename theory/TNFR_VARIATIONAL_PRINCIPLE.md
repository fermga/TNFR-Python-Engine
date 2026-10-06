<a id="tnfr-variational-principle-lagrangian-action-formulation"></a>

# Variational diagnostics and conditional constitutive models

**Status**: Exact restricted EPI balances, conditional forced-potential family and reciprocal closure constraints; decoupled product and fixed-P2 harmonic realizability obstruction; broader coupled tetrad/nodal bridge unresolved
**Implementation**: `physics/variational.py` owns field-action readouts; the
pressure, diffusion, phase-response and constitutive owners linked below supply
the separate dynamical calculations.
**Tests**: `tests/physics/test_diffusion_energy_balance.py`, `tests/physics/test_variational.py`, `tests/physics/test_symplectic_substrate.py`, `tests/physics/test_metriplectic_product.py`, `tests/physics/test_symplectic_graph_realizability.py`, `tests/physics/test_constitutive_variational_scope.py`
**Updated**: 2026-10-07

---

## 1. Main Result

The TNFR nodal equation

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

is the unforced nodal row; a complete model must also supply pressure and the
other consumed evolution laws. This note distinguishes three responsibilities:

| Question | Section and mathematical scope |
| --- | --- |
| What do the field-action diagnostics compute? | [Sections 2–12](#2-the-tnfr-lagrangian) define the readouts and the separate harmonic-coordinate model. Its brackets and flow are supplied premises. |
| Which diffusion and harmonic bridges actually hold? | [Sections 3.4–3.8](#34-overdamped-graph-diffusion-and-the-missing-bridge) retain exact restricted balances, a decoupled product and graph-realizability obstructions. |
| What does a proposed joint variational law have to satisfy? | [Section 13](#13-forced-potential-family-and-reciprocal-closure-constraints) derives conditional source, metric, work and reciprocity constraints; it does not uniquely select a constitutive law. |

A derivation of the full nodal law from the diagnostic field potential is
**not established**. The conditional pressure-compatible potentials in section
13 are different objects. Neither a field identity nor an auxiliary harmonic
motion proves that an engine trajectory obeys the corresponding action
principle. The current native relational law and its competing storage
premises are owned by [relational exchange admission](nodal/RELATIONAL_EXCHANGE_ADMISSION.md).

---

## 2. The TNFR Lagrangian

### 2.1 Lagrangian Density

On a graph $G=(V,E)$ the implemented **Lagrangian diagnostic** at node $i$ is

$$\mathcal{L}(i) \;=\; T(i) \;-\; V(i)$$

where the historical kinetic and potential labels denote the following
quadratic field groups:

$$T(i) = \tfrac{1}{2}\bigl[J_\varphi(i)^{\,2} + J_{\Delta\mathrm{NFR}}(i)^{\,2}\bigr]$$

$$V(i) = \tfrac{1}{2}\bigl[\Phi_s(i)^{\,2} + |\nabla\varphi|(i)^{\,2} + K_\varphi(i)^{\,2}\bigr]$$

These graph read-outs are not independently supplied positions and velocities.
The harmonic model in section 3 adds that coordinate interpretation explicitly.
Physical energy units require field scales or a metric; raw pressure, angle
and path-length quantities cannot be added dimensionally without them. The
[normalization boundary](STRUCTURAL_CONSERVATION_THEOREM.md#main-result)
owns that distinction for all field-energy and charge diagnostics.

### 2.2 Action Functional

$$S_{\mathrm{TNFR}} = \int \mathrm{d}t \;\sum_{i\in V} \mathcal{L}(i,t)$$

In discrete form (time steps of width $\Delta t$):

$$S_{\mathrm{TNFR}} = \Delta t \;\sum_n \sum_i \mathcal{L}(i,\,t_n)$$

### 2.3 Hamiltonian

Treating the two conjugate sectors as independent harmonic-model coordinates gives the **substrate Hamiltonian plus held-fixed background**:

$$\mathcal{H}(i) = T(i) + V(i) = \tfrac{1}{2}\bigl[\Phi_s^2 + |\nabla\varphi|^2 + K_\varphi^2 + J_\varphi^2 + J_{\Delta\mathrm{NFR}}^2\bigr]$$

This is exactly $\tfrac{1}{2}\,\mathcal{E}(i)$ where $\mathcal{E}$ is the energy density from `unified.compute_energy_density()`, and the total Hamiltonian $H = \sum_i \mathcal{H}(i)$ equals `conservation.compute_energy_functional()`.

---

## 3. Harmonic Model and the Unresolved Nodal Correspondence

### 3.1 Canonical Conjugate Pairs

The substrate model assigns two canonical coordinate pairs $(q,\,p)$ to the extracted field sectors:

| Sector | Coordinate $q$ | Momentum $p$ | Graph-field interpretation |
|--------|----------------|---------------|------------|
| **Geometric** | $K_\varphi$ (curvature) | $J_\varphi$ (phase current) | Two instantaneous phase statistics; no continuity law follows from pairing them |
| **Potential** | $\Phi_s$ (potential) | $J_{\Delta\mathrm{NFR}}$ (ΔNFR flux) | Two pressure statistics; their graph balance has a measured residual |

The ambient canonical brackets are a model premise. Conservation diagnostics
do not establish those brackets or a small continuity residual. Sections 3.7–3.8
test whether the resulting auxiliary velocity is realizable by the graph fields.

### 3.2 Hamilton's Equations

$$\frac{\partial q_i}{\partial t} = \frac{\partial \mathcal{H}}{\partial p_i} = p_i \qquad\text{(dynamics)}$$

$$\frac{\partial p_i}{\partial t} = -\frac{\partial \mathcal{H}}{\partial q_i} = -q_i \qquad\text{(force law)}$$

### 3.3 Scope of Hamilton's Equations

The equations above define isotropic oscillators, `q''=-q`, on the specified
ambient coordinate space. Extracting coordinates from a graph does not prove
that all ambient points are realizable by graph fields or that an evolved
ambient point is the image of an engine trajectory.

### 3.4 Overdamped Graph Diffusion and the Missing Bridge

For a separate graph-wave model,

```text
q'' + gamma*q' + L_rw*q = 0,
```

the overdamped slow rates approach `lambda_k/gamma`, producing the EPI
channel's diffusion with mobility `nu_f=1/gamma`. This requires an explicit
stiffness operator and damping law. It does not identify the graph wave with
the isotropic substrate oscillator, whose stiffness is the identity.

The previously stated identity `DeltaNFR=-dV/dEPI` fails for the implemented
potential even on one edge. With pure EPI-channel pressure, equal phases,
and `EPI=[1,0]`, one obtains `DeltaNFR=[-1,1]`, `Phi_s=[1,-1]`, and
`V=(EPI_0-EPI_1)^2`. Hence `-gradient(V)=[-2,2]`, not `DeltaNFR`.

The EPI-only diffusion does have a degree-metric gradient formulation using
its Dirichlet energy. A bridge to the different tetrad potential and to the
full pressure still requires a specified coordinate map, metric and damping
mechanism. The nodal equation makes `nu_f` a mobility; it does not derive an
inverse inertial mass.

### 3.5 Exact Restricted Dirichlet Balance

Hold a finite symmetric nonnegative adjacency `W` fixed. Let `d_i=Σ_j W_ij`,
`B=D−W`, and `x=EPI`. For the isolated EPI channel, define

```text
E_D(x) = ½ xᵀ B x = ¼ Σ_ij W_ij (x_i−x_j)²,
∇E_D = Bx,
M_ii = νf_i/d_i if d_i>0, otherwise 0.
```

Symmetry gives `∇E_D=Bx`. Multiplying by the nonnegative mobility yields
`−M∇E_D=−diag(νf)L_rw x`, exactly the canonical pure EPI pressure times
capacity. The chain rule therefore proves

```text
dE_D/dt = −(∇E_D)ᵀ M (∇E_D) = −Σ_i M_ii (∇E_D_i)² ≤ 0.
```

For positive strengths and capacities, the metric is `diag(d_i/νf_i)`.
Zero capacity instead gives degenerate mobility: nonzero pressure may coexist
with zero velocity and constant positive energy. Isolates have zero flow;
self-loops add row strength but no Dirichlet energy. Parallel-edge conductance
is summed. Disconnected components need not share a uniform equilibrium value.

The read-only implementation
[`compute_diffusion_energy`](../src/tnfr/physics/structural_diffusion.py) reuses
the diffusion module's validated adjacency, capacity and EPI readers. Its edge
difference evaluation avoids cancellation under large common EPI offsets and
requires only the core NumPy dependency. The generic spectral smoothness helper
evaluates `xᵀLx` for a supplied matrix and imports optional SciPy; it neither
supplies this mobility nor validates this transport scope.

Tests compare the gradient and directional energy derivative against independent
edge-energy finite differences, then compare velocity with the canonical pressure
callback for weighted graphs, parallel edges, loops and heterogeneous capacities.
This establishes the restricted identity, not a theorem about the different
tetrad potential, changing graph weights, all four pressure channels, arbitrary
discrete time steps, or a lift to the isotropic substrate.

### 3.6 Exact Decoupled Metriplectic Product

The harmonic substrate and the restricted EPI gradient flow can coexist in one
block system `X'=J grad(H)-G grad(E_D)`. With zero cross blocks,
`J grad(E_D)=0` and `G grad(H)=0`; therefore the substrate Hamiltonian is
conserved while Dirichlet energy decreases. The implementation
[`verify_metriplectic_product`](../src/tnfr/physics/metriplectic.py) verifies
both degeneracies and both component vector fields.

Its numerical Dirichlet energy, gradient, mobility and rate now come from
the shared [diffusion balance](../src/tnfr/physics/structural_diffusion.py),
aligned explicitly to the substrate's node order. Recomputing these
quantities locally previously admitted false zero motion after underflow:
on P2 with conductance `1e200`, capacity `1e-200` and EPI `(1,0)`, a computed
zero `nu/d` erased representable nodal rates of magnitude `1e-200`. The exact
Dirichlet derivative is `-2*W*nu`, approximately `-2` for those binary64
inputs. The shared owner rejects this
unrepresentable positive mobility, as well as lost nonzero energy/rate
terms, instead of certifying their zero substitutes. Normal-range and
reordered weighted-loop controls independently compare the edge energy and
dissipation. This numerical admission does not extend the mathematical model.

This is an exact direct product, not the missing coupled derivation. Since the
cross blocks vanish, it supplies no pulse-relaxation feedback. A full bridge
would require a TNFR-derived interaction and preservation of the realizable
graph-field image, not just nonzero cross tensors with those degeneracy laws.
See [TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).

### 3.7 P2 Read-Out Realizability: Obstruction and Derived Flow

An ambient harmonic solution need not be a trajectory of the graph fields
used to initialize it. For a graph state `X`, actual vector field `F` and
read-out `h`, a proposed smooth autonomous field flow `f` requires
`Dh(X) F(X) = f(h(X))`. In particular, `f` must be tangent to the realizable
image of `h`. An instantaneous coordinate pairing does not establish this.

**Exact geometric obstruction.** On fixed mutual-singleton P2 support,
choose an open wrap branch and set `k=wrap(theta_i-theta_j)` in `(-pi,pi)`.
The production definitions of curvature and phase current give

```text
K_phi(i) = k,                 J_phi(i) = -sin(k),
c(k,j) = j + sin(k) = 0.
```

The singleton phasor resultant has unit magnitude, so no ambiguous circular
mean is used. The auxiliary harmonic velocity `(k_dot,j_dot)=(j,-k)` has
constraint derivative

```text
c_dot = -k + cos(k)*j = -k - sin(k)*cos(k).
k + sin(k)*cos(k) = integral_0^k 2*cos(u)^2 du.
```

The integral has the strict sign of `k`, hence tangency holds only at `k=0`.
No nonzero continuous harmonic geometric trajectory can therefore be produced
by **any differentiable phase evolution** with this fixed support and these
same read-outs. Changing capacity or EPI does not change that constraint.
A nonvanishing scalar rescaling of time cannot remove the obstruction.
Synchronized phases may rotate together while the geometric read-out stays
zero; the theorem does not freeze all possible phase evolution.

This is an exact-real, fixed-support, open-branch result. It is not a claim
about larger graphs, different derived observables, or branch-crossing events.
It also does not exclude isolated endpoint coincidences: a harmonic half-turn
maps `(k,-sin(k))` to `(-k,sin(k))`, again on the geometric image, although the
intermediate harmonic path leaves that image. No graph event is certified by
that isolated coincidence.

**A field flow that does derive from the nodal equation.** Now restrict to
unit conductance and unit metric length on P2, positive fixed capacities
`nu_0,nu_1`, scalar EPI `x`, fixed phase, pure-EPI pressure refreshed from
the current state, and no additional forcing. Write `d=x_1-x_0`. The nodal
law and the same production field definitions give

```text
DeltaNFR = (d,-d),            d_dot = -(nu_0+nu_1)*d,
Phi_s = (-d,d),               J_DeltaNFR = (-2*d,2*d) = 2*Phi_s,
K_phi_dot = J_phi_dot = 0,
Phi_s_dot = -(nu_0+nu_1)*Phi_s,
J_DeltaNFR_dot = -(nu_0+nu_1)*J_DeltaNFR.
```

Thus the extracted potential/flux sector has a closed **dissipative** flow
with rate determined by the nodal capacities, without a fitted coefficient
or an imported interaction. The lost common EPI offset does not prevent this
read-out closure. This does not establish complete state reconstruction.
On the realized potential line `p=2q`, the auxiliary harmonic velocity instead
has `(p-2q)_dot=-q-2p=-5q`, nonzero for `q!=0`. Consequently even synchronized
phases do not rescue that harmonic comparison when EPI disagreement remains.
The synchronized uniform-EPI state is a common zero-field fixed point.

**Executable boundary.** The fixture `x=(0,1)`, `nu=(1,1)` and
`theta=(0,pi/3)` gives `K_phi(0)=-pi/3`, `J_phi(0)=sqrt(3)/2` and geometric
tangency defect `pi/3+sqrt(3)/4`. Tests use the production pressure callback,
field extractors and shared nodal integrator for three explicitly refreshed
Euler segments of length `1/8`, with no active clipping or added forcing.
The exact ideal segment disagreement factor is `1-(nu_0+nu_1)*dt`; this finite
check is not an asymptotic binary64 solver theorem. Additional fixtures check
heterogeneous capacities, both signs of the geometric defect, the zero control
and the half-turn boundary. Numerical tolerances verify represented readings;
the nonzero-for-all-`k` claim follows from the analytic integral above.

Regression owner:
[test_symplectic_graph_realizability.py](../tests/physics/test_symplectic_graph_realizability.py).
Field owners: [canonical.py](../src/tnfr/physics/canonical.py),
[extended.py](../src/tnfr/physics/extended.py),
[symplectic_substrate.py](../src/tnfr/physics/symplectic_substrate.py).
No new physical primitive or general TNFR-to-Hamiltonian equivalence is assumed.

---

### 3.8 Prepared prism harmonic geometric velocity has no phase lift

The current unit-prism preparation supplies a separate exact realizability
test. At repeated phases `(0,pi/3,pi/6)`, the actual geometric fields are
`K=(-pi/6,pi/6,0)` and `J_phi=((1+sqrt(3))/6,-(1+sqrt(3))/6,0)`, repeated
in both triangles. Let `R` be the shared phasor mean-response matrix.
Then `DK=I-R` has rank five and kernel equal to common phase rotation.

The auxiliary harmonic direction requires `K_dot=J_phi` and `J_phi_dot=-K`.
Since `DK*J_phi=J_phi`, every possible primitive phase velocity satisfying
the first row is `theta_dot=J_phi+q*1`. The actual mean-sine derivative
annihilates `1`, and necessarily gives

\[
DJ_\phi\,\dot\theta=
\left(-\frac{5+3\sqrt3}{36},\frac{5+3\sqrt3}{36},0\right)_{\rm repeated},
\]

which differs from `-K`. The stacked derivative has rank five, while
adjoining the proposed harmonic velocity raises the rank to six. Therefore
**no primitive phase velocity**, even a nonrepeated one, lifts this auxiliary
geometric direction at this state. This is a prepared-state obstruction,
not a classification of every prism state or every coupled Hamiltonian.
It reuses the existing field extraction and harmonic-vector-field owners;
no new flow is executed.

Reciprocal exchange also does not automatically replenish loss. In the
conditional linear family `z_dot=-e*z-k*zeta`, `zeta_dot=a*z+b*zeta`, with
`a,e,k>0`, `b<=0`, the positive storage
`S=a*|z|^2+k*|zeta|^2` satisfies
`S_dot=-2*a*e*|z|^2+2*k*b*|zeta|^2<=0`. After scaling coordinates by
`sqrt(a),sqrt(k)`, the generator is an antisymmetric exchange plus diagonal
loss. At `b=0`, zero instantaneous loss need not imply zero velocity: a
state with `z=0,zeta!=0` immediately leaves that set. The largest invariant
zero-loss set is only the origin; the characteristic polynomial in the
[joint-response classification](nodal/PRIMITIVE_PHASE_CLOSURE.md#174-admission-conditions-for-a-joint-linear-response)
gives asymptotic decay throughout this passive class, including damped
oscillations where its discriminant is negative. This storage is a
comparison quadratic, not a derived canonical joint energy.

Controls: [prepared prism lift and passive exchange](../tests/physics/test_prism_harmonic_phase_lift.py).

## 4. Grammar-Labelled Heuristics

`analyze_grammar_stationarity()` reports energy and interaction statistics
labelled by related grammar rules. Each result now explicitly carries
`verification_scope="heuristic"`. Its legacy `is_satisfied` flag evaluates that
statistic only. In particular:

- Nonzero Lagrangian density does not establish a generator history.
- Potential dominance does not prove an attractor or valid closure.
- A finite sampled action does not prove infinite-horizon convergence.
- Bounded bilinear interactions do not verify the U3 phase gate.
- `T/V` does not determine a Hessian, bifurcation, or handler history.
- Energy dispersion does not establish nested identities or per-level stabilizers.
- Absolute potential magnitude is not U6 drift from a reference state.

Canonical sequence validation and confinement telemetry remain responsible for
the actual grammar requirements.

---

## 5. Operator Energy Heuristics and Local Symplectic Checks

Older code attached sign labels to operators. Those labels are priors for a
snapshot statistic, not consequences of U2 or of the glyph factors. The
five-field Hamiltonian has no explicit EPI or capacity term, while phase and
pressure changes propagate through graph-dependent fields. Every realized sign
therefore remains state-dependent and must be measured.

| Operators | U2 bookkeeping role | Historical sign prior | Valid conclusion |
|-----------|----------------------|-----------------------|------------------|
| **IL, THOL** | Stabilizer | decrease | No universal $\Delta H$ sign |
| **OZ, ZHIR, VAL** | Destabilizer | increase | No universal $\Delta H$ sign |
| **AL, EN, UM, RA, SHA, NUL, NAV, REMESH** | Neutral | unchanged | U2 neutrality does not imply $\Delta H=0$ |

The policy-multiplier layer in `physics.lyapunov` now exposes this scope through
`is_energy_bound=False` and `is_lyapunov_certificate=False`. A canonical
transformation can also change a particular Hamiltonian; energy conservation
and symplecticity are separate properties.

`check_symplectic_preservation(before, after, ...)` preserves its positional
arguments and legacy ratio fields, but snapshots alone now return
`is_canonical=None`, `classification="inconclusive"`. The old ratio category
is available as `heuristic_classification`; `sum(abs(q*p))` is not a volume.

For a supplied local derivative, use:

```python
result = check_symplectic_preservation(
    before, after, "map name", jacobian=M, jacobian_tolerance=1e-9
)
```

The matrix must be finite, real, and `(4N,4N)`, with coordinates interleaved
`(K_phi,J_phi,Phi_s,J_DNFR)` per node. The shared substrate helper evaluates
`max(abs(M.T @ omega @ M - omega))`. A passing result concerns this supplied
Jacobian only; establishing that it is an engine operator's derivative and
proving preservation over its domain requires additional evidence.

The functions historically named `compute_phase_space_volume` and
`compute_poisson_bracket_estimate` retain their numeric statistics for
compatibility. They no longer claim those statistics are geometric certificates.

---

## 6. Telemetry Thresholds Are Regular Points of the Quadratic Potential

For each field coordinate, `V(x)=x^2/2`, `V'(x)=x`, and `V''(x)=1`. The only
critical point is zero. Nonzero configured thresholds do not create extrema
or saddles of this potential, and the restoring force does not reverse sign
when a threshold is crossed.

`analyze_potential_critical_points()` retains its API and configured comparison
values. It reports the correct derivatives and `critical_type="regular"` at
nonzero thresholds. The added `near_threshold_count` preserves the observation
of nodes near a threshold without mislabelling them as stationary points. A
nonlinear constrained effective potential would require an explicit formula
and a separate derivative calculation.

---

## 7. Interaction Lagrangian

The bilinear cross-sector coupling is:

$$\mathcal{A}(i) = \Phi_s \cdot |\nabla\varphi| + K_\varphi \cdot J_\varphi + |\nabla\varphi| \cdot J_{\Delta\mathrm{NFR}}$$

This equals the snapshot contraction `unified.compute_action_density()`.
One can formally write a candidate interaction functional:

$$\mathcal{L}_{\mathrm{full}} = T - V - \mathcal{A}$$

This algebra alone does not derive a coupled equation of motion for the graph
or its extracted fields. Such a claim requires a declared state space,
admissible variations, metric/mobilities and a realizability check against the
nodal dynamics. The specified harmonic substrate supplies its own independent
sector flow; the snapshot bilinear term does not establish the missing
phase/capacity feedback or the restoring contribution in section 13.5.

---

## 8. Energy Partition

For the harmonic conjugate core, a complete-period average satisfies
`<T>=<V_core>`. This does not imply instantaneous equipartition, and the full
potential also contains the held-fixed `|grad_phi|` background. The instantaneous
`T/V` ratio is an energy-partition statistic; it does not by itself certify an
attractor or bifurcation.

---

## 9. Consistency Relations

The following algebraic readout identities use shared field definitions:

| Relation | Verification |
|----------|-------------|
| $\mathcal{H}(i) = \tfrac{1}{2}\,\mathcal{E}(i)$ | `compute_hamiltonian_density()` vs `unified.compute_energy_density()` |
| $\sum_i \mathcal{H}(i) = E$ | Total Hamiltonian vs `conservation.compute_energy_functional()` |
| $\rho = \Phi_s + K_\varphi$ | Charge density from conjugate pair coordinates |
| $\mathcal{A} = $ action density | `compute_interaction_density()` vs `unified.compute_action_density()` |
| Recorded $dH/dt$ | Finite-difference diagnostic; no universal sign is certified here |

Tests cover these readout identities across deterministic graph fixtures. Passing identity tests does not prove the unresolved model correspondence in section 3.

---

## 10. Single Source of Truth Architecture

The five fields in the energy expression admit three readout groupings.
This algebraic regrouping does not assert a six-dimensional independent phase space.
The implementation ensures a **single computational path** with no duplicated
formulae:

```
unified.py                    (Layer 3 — RAW quadratic forms)
  ├── compute_energy_density   ℰ(i) = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
  └── compute_action_density   𝒜(i) = bilinear coupling term
       │
       ├──► variational.py      H(i) = ½·ℰ(i),  T, V, ℒ = T−V
       │    (delegates via _raw_energy_density / _action_density)
       │
       └──► conservation.py     E = ½·Σℰ(i),  ρ, div(J), Q
            (delegates via _raw_energy_density)
```

### 10.1 Sector Decompositions

| Decomposition | Split | Grouping criterion |
|---|---|---|
| **Variational** (T / V) | $T = \tfrac{1}{2}[J_\varphi^2 + J_{\Delta NFR}^2]$, $V = \tfrac{1}{2}[\Phi_s^2 + \vert \nabla\varphi\vert ^2 + K_\varphi^2]$ | Temporal role (kinetic / configuration) |
| **Conservation** ($\rho$ / J) | $\rho = \Phi_s + K_\varphi$, $\mathbf{J} = (J_\varphi, J_{\Delta NFR})$ | Diagnostic charge/current grouping; conservation requires a separate law |
| **Unified** ($\Psi$) | $\Psi = K_\varphi + i\,J_\varphi$ | Dual structure (geometry-transport) |

All three satisfy the **consistency identity** $T(i) + V(i) = \tfrac{1}{2}\,\mathcal{E}(i)$ at every node, verified by `translate_sectors()` with residual < $10^{-12}$.

### 10.2 translate_sectors() API

```python
from tnfr.physics.variational import translate_sectors

result = translate_sectors(G)
# result['variational']        -> {'T': {...}, 'V': {...}}
# result['conservation']       -> {'rho': {...}, 'J_phi': {...}, 'J_dnfr': {...}}
# result['unified_psi']        -> {node: K_φ + i·J_φ}
# result['energy_density']     -> raw ℰ from unified.py
# result['consistency_check']  -> max |T+V − ½ℰ|  (should be ~0)
```

---

## 11. API Summary

```python
from tnfr.physics.variational import (
    # Core densities
    compute_kinetic_density,       # T(i) = ½[J_φ² + J_ΔNFR²]
    compute_potential_density,     # V(i) = ½[Φ_s² + |∇φ|² + K_φ²]
    compute_lagrangian_density,    # ℒ(i) = T(i) − V(i)
    compute_hamiltonian_density,   # H(i) = ½·ℰ(i) — delegates to unified.py
    compute_interaction_density,   # 𝒜(i) — delegates to unified.compute_action_density

    # Sector translation
    translate_sectors,             # All three decompositions + consistency check

    # Phase space
    identify_conjugate_pairs,      # (K_φ,J_φ), (Φ_s,J_ΔNFR)
    compute_phase_space_volume,    # Legacy sum |q*p| statistic
    compute_poisson_bracket_estimate,

    # Snapshot & tracking
    capture_lagrangian_snapshot,    # Complete instant analysis
    compute_euler_lagrange_residual,
    compute_action_functional,     # S = ∫ℒ dt
    VariationalTracker,            # Time-series accumulation

    # Canonical checks
    check_symplectic_preservation,
    classify_operator_canonical,

    # Grammar stationarity
    analyze_grammar_stationarity,  # Explicitly heuristic grammar-labelled readouts
    analyze_potential_critical_points,

    # Comprehensive
    compute_variational_suite,     # Full analysis in one call
)
```

---

## 12. Symplectic Reduction Scope

For `J=H_sub=|z|^2/2>0`, the diagonal U(1) action is free and the quotient has
dimension `4N-2`. Its global space is complex projective space `CP^(2N-1)`,
not a flat linear space. The certificate restricts the symplectic form to the
actual horizontal tangent space orthogonal to `z` and `omega*z`. Relative
phase checks use only nonzero complex coordinates and a defined reference.

At `J=0`, the level set and quotient are a point. The certificate reports
`reduced_dimension=0`, `reduction_status="singular_zero_level"`, and
`is_valid_reduction=False`, meaning the regular-level test is inapplicable.
The legacy constant `reduced_symplectic_form_matrix(n_nodes)` remains a local
non-orthonormal action-angle basis matrix; it does not certify a supplied state.

## 13. Forced-Potential Family and Reciprocal Closure Constraints

The full canonical pressure constrains a possible variational completion more
strongly than the EPI-only energy does. These are exact conditional identities,
not an added engine law. Fix a connected symmetric conductance `W`, positive
strengths `d`, unweighted support `U`, and constant channel coefficients. Write
`D=diag(d)`, `B=D-W`, `L_W=D^-1 B`, and `L_U` for the support random-walk
Laplacian. On a regular phase chart with nonzero neighbor resultants,

```text
p = -e L_W x + F(theta,nu,U),
F = w_phi g(theta) - v L_U nu - t L_U k,
Dg = (R-I)/pi,
```

where `k` is support degree and `R` is the exact neighbor-mean derivative from
[phase_response](../src/tnfr/physics/phase_response.py). Positive capacity gives
the already established EPI mobility `M_x=diag(nu_i/d_i)`. Suppose a `C2`
scalar potential supplies the EPI component specifically as
`x'=-M_x gradient_x E`, with no cross-mobility contribution to that component.
The nodal equation then requires

```text
gradient_x E = e Bx - DF,
E(x,theta,nu,W) = (e/2) x^T Bx - x^T DF + Psi(theta,nu,W).
```

Integration in `x` proves both necessity and sufficiency on a connected EPI
coordinate domain. The arbitrary `x`-independent function `Psi` is genuine
remaining freedom; setting it to zero is an additional choice. The expression
is not identified with the different field energy in section 2.

### 13.1 Mixed derivatives require reciprocal coupling

For this entire potential family, equality of mixed partial derivatives fixes
the following dependence on EPI:

```text
gradient_theta E = -(w_phi/pi)(R-I)^T D x + gradient_theta Psi,
gradient_nu E    = v L_U^T D x + gradient_nu Psi.
```

Thus a proposed joint gradient completion with positive definite block
mobilities `M_theta(theta,nu,W)` and `M_nu(theta,nu,W)`, independent of `x`,
would necessarily have these terms in `theta'=-M_theta gradient_theta E` and
`nu'=-M_nu gradient_nu E`. An `x`-independent phase relaxation cannot satisfy
that completion on an open full-state domain when `w_phi>0` and `R-I!=0`.
An `x`-independent capacity relaxation likewise cannot satisfy it when `v>0`
and `L_U!=0`. An `x`-independent `Psi` cannot cancel an `x`-linear term for all
`x`. The obstruction does not cover degenerate mobilities, cross blocks,
other metrics, constrained submanifolds or non-gradient dynamics. In
particular it does not turn the engine's discrete phase/capacity policies
into smooth laws or assume that their complete state dependence is absent.

Even if a gradient completion is selected, the EPI equation determines neither
`Psi` nor the two auxiliary mobilities and their time scale. For smoothly
varying positive off-diagonal edge conductances on fixed support, it also forces

```text
partial E / partial W_ij = (e/2)(x_i-x_j)^2 - x_i F_i - x_j F_j
                          + partial Psi / partial W_ij,
```

where each undirected edge `i<j` is one coordinate and `F` is independent of its
weight on this fixed-support chart. This supplies a necessary energy
derivative, not an edge evolution law. Edge birth/deletion changes `U` and
the pressure realization; it is a separate hybrid closure obligation.

### 13.2 Lower bounds require the actual source compatibility

At held `theta,nu,W`, shifting `x` by `c*1` changes every compatible potential
by `-c*d^T F`. Therefore no member of this family is bounded below over all
real EPI fields if `d^T F!=0`. An arbitrary `Psi` cannot repair this tilt.
Conversely, for `e>0` and connected conductance, `d^T F=0` yields a solution
`e Bx_*=DF`, and completing the square gives

```text
E = (e/2)(x-x_*)^T B(x-x_*) - (e/2)x_*^T Bx_* + Psi.
```

This is bounded below in `x`, with the common-offset freedom retained for
the held source. Section 13.9 tests whether that freedom extends jointly
to phase variations. This held-source result reuses the compatibility and profile in
[forced_support](../src/tnfr/physics/forced_support.py); it supplies no new
stationary phase or capacity maintenance law. A weighted three-node path
with edge weights `(1,2)`, capacities `(1/2,1,3/2)`, phase consensus and
channel weights `(e,v,t)=(1/2,1/4,1/4)` has
`F=(3/8,-1/4,1/8)` and `d^T F=-1/8`. This is an exact canonical-source
example of the unbounded tilt, despite symmetric conductance.

### 13.3 Canonical phase pressure is not generally a fixed-metric gradient

It is also necessary to check integrability before treating `g` itself as a
phase gradient flow. On the unit star `K1,3`, index the center by zero and
choose phases `(0,0,0,alpha)` with `0<alpha<pi/2`. Every edge satisfies strict
U3, all resultants are nonzero and the wrap chart is regular. Put
`c=cos(alpha)`. The shared circular-mean derivative gives

```text
R_01=R_02=(2+c)/(5+4c),  R_03=(1+2c)/(5+4c),
R_j0=1 for j=1,2,3; all other entries are zero.
```

A constant positive diagonal metric `H` could represent `g` as
`-H^-1 gradient V` only if `H(R-I)` were symmetric throughout the domain.
At consensus this forces `H` proportional to `diag(3,1,1,1)`. At every
nonzero arbitrarily small `alpha` its center/first-leaf antisymmetry is

```text
3 R_01 - R_10 = (1-c)/(5+4c) > 0.
```

Consequently no such constant diagonal metric works on any open neighborhood
of consensus. For the exact planar cosine Gram with `c=3/5`, the center row
is `(0,13/37,13/37,11/37)`, its squared resultant is `37/5`, and the metric
antisymmetries are `2/37,2/37,-4/37`. This does not exclude state-dependent
metrics, non-diagonal metrics, different phase laws or a variational
completion of the joint pressure family above. Strict U3 and symmetric graph
support alone do not establish the missing gradient identity.
Section 13.6 derives an exact **state-dependent** diagonal representation
on the regular phase domain. It preserves this constant-metric obstruction
and does not by itself select a joint evolution law.

### 13.4 Relation to the full tetrad

Neither this conditional potential nor its mixed derivatives close the four
read-outs `Phi_s`, phase-gradient magnitude, `K_phi` and `xi_C`. Their
evolution still requires actual phase/capacity/support laws and the derivative
of their common graph observation map, with branch and estimator boundaries
handled separately. The five-field energy in section 2 has no explicit
`xi_C` term; its algebraic identity cannot establish a correlation-length
evolution. The complete closure obligations remain in
[DIAGNOSTIC_AND_GRAMMAR_SCOPE section 13](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#13-structural-grammar-refactor-and-the-full-nodal-system).

Exact finite regression controls reuse the pressure decomposition,
support-transport and phase-response owners in
[test_constitutive_variational_scope.py](../tests/physics/test_constitutive_variational_scope.py).
They check source signs, the required mixed coupling and the fixed-metric
obstruction without trajectories, new feedback coefficients or fitted laws.

### 13.5 Necessary restoring contribution on the compatible P2 preparation

The [finite joint compatibility result](nodal/JOINT_PARAMETER_RESPONSE.md#11-finite-joint-phase-capacity-source-compatibility)
does not select a maintenance mechanism. The same two-node preparation gives
an exact test of what the conditional potential family in section 13 would
require. Keep its EPI mobility premise and fixed unit P2 support. Write
`xi=x_0-x_1`, `eta=nu_0-nu_1`, and hold phase and both common coordinates
fixed **only as a perturbation subspace**. This does not assert that the
subspace is invariant under an unspecified joint flow.

For regular singleton phase means, set `a(theta)=w_phi*g_0(theta)`, so
`g_1=-g_0`. The topology channel vanishes on this support. Restricting the
already-required potential to the plane gives

\[
E(\xi,\eta)=\frac e2\xi^2+v\xi\eta-a\xi+\Psi(\eta),
\qquad e>0,\quad v>0.
\]

Here `Psi` is the restriction of the still-undetermined, EPI-independent
potential contribution; it is not the complex geometric field with the same
letter elsewhere. The source owners determine the mixed coefficient `v`.
They do not determine `Psi`. Assume it is twice continuously differentiable
near the preparation, with all held coordinates understood as parameters.

**Stationarity and restoring curvature are distinct requirements.** A local
minimum in the interior requires

\[
e\xi_*+v\eta_*-a=0,\qquad
\Psi'(\eta_*)=-v\xi_*.
\]

The first equation is the zero-pressure condition; it does not imply the
second. At the retained consensus-phase preparation

```text
x=(1/4,0), nu=(1,3/2), theta=(0,0),
e=1/2, v=w_phi=1/4, xi*=1/4, eta*=-1/2, a=0,
```

the additional requirement is `Psi'(eta*)=-1/16`. Setting `Psi=0` therefore
does not even make this point stationary on the plane. Calling that
noncritical point a saddle would be incorrect.

Let `k=Psi''(eta*)`. The restricted Hessian and its quadratic form are

\[
H=\begin{pmatrix}e&v\\v&k\end{pmatrix},\qquad
(u,s)H(u,s)^\top
=e\left(u+\frac ve s\right)^2
 +\left(k-\frac{v^2}{e}\right)s^2.
\]

Consequently a local energy minimum **necessarily** requires

\[
k\ge\frac{v^2}{e};\qquad
\text{at this preparation: }k\ge\frac18.
\]

Together with stationarity, strict inequality suffices for a strict local
minimum on this two-coordinate plane. It does not establish a minimum in
the full phase/capacity/common-coordinate space or dynamical stability. With
`[xi]=X`, `[eta]=T^-1` and unit conductance, `[E]=X^2`, `[v]=XT` and
`[k]=X^2*T^2`; `1/8` belongs to this normalized fixture, not a universal
dimensionless structural threshold.

**Exact insufficient-curvature obstruction.** If stationarity is supplied
but `k<v^2/e`, take `delta_eta=s` and `delta_xi=-(v/e)*s`. Taylor's theorem
gives

\[
E(\xi_*-(v/e)s,\eta_*+s)-E(\xi_*,\eta_*)
=\frac12\left(k-\frac{v^2}{e}\right)s^2+o(s^2)<0
\]

for sufficiently small nonzero `s`. The pure `xi` direction has positive
quadratic curvature, so a stationary point in this case is a saddle on
the plane. At the retained fixture, the perturbation is

\[
x(s)=(1/4-s/4,\ s/4),\qquad
\nu(s)=(1+s/2,\ 3/2-s/2),\qquad\theta=0.
\]

For `0<s<1/2`, both capacities stay strictly inside the earlier closed bands
`[3/4,7/4]`, and EPI remains nonnegative. Thus the descending direction is
not an artifact of negative capacity or a forbidden phase separation. Each
point has exactly zero canonical pressure: the capacity source compensates
the new EPI contrast. The curve is **not a nodal trajectory**, since EPI
changes along it while the nodal equation would give `xdot=0` at each point.

**Equality is genuinely undecided by the Hessian.** Put
`s=eta-eta*` and `r=xi-xi*+s/2`. At this fixture the exact identity is

\[
\Delta E=\frac{r^2}{4}
 +\Psi(\eta_*+s)-\Psi(\eta_*)+\frac{s}{16}-\frac{s^2}{16}.
\]

The three logical completions

\[
\Psi(\eta_*+s)-\Psi(\eta_*)
=-\frac{s}{16}+\frac{s^2}{16}+\{0,\ +s^4,\ -s^4\}
\]

share the required slope and `k=1/8`. They yield a flat valley, a strict
restricted minimum, and a descending null direction, respectively. These
polynomials show why second-order evidence is insufficient at equality;
none is installed or proposed as a selected TNFR law.

**Tetrad and physical scope.** Along the displayed curve, zero pressure
gives `Phi_s=0`; consensus phase gives zero phase gradient and `K_phi=0`.
The graph-distance metric and static pressure-coherence inputs to `xi_C`
are unchanged; its actual fit/fallback provenance is retained. All four
current diagnostics can therefore agree while the conditional joint
potential changes. This is a failure of observation sufficiency, not an
executed relaxation or a new diagnostic feedback law. Auxiliary field-energy
positivity cannot substitute for the missing capacity derivatives.

The obstruction concerns a local energy minimum in the stated passive
gradient class. It is not a no-go theorem for TNFR patterns, non-gradient
motion, oscillatory identity, hybrid events or constrained/degenerate
mobilities. Those possibilities need their own independently justified laws.
Even a strict minimum on the plane supplies neither an activation mechanism
nor a route by which a preparation forms spontaneously.

| Required ingredient | Provenance and remaining obligation |
| --- | --- |
| EPI force and mixed coefficient `v` | Fixed by the declared pressure realization and the section 13 EPI mobility premise; checked through the shared pressure and support owners. |
| `Psi'(eta*)` and lower bound on `Psi''(eta*)` | Necessary consequences for a restricted stationary minimum, not a derivation or selection of `Psi`. |
| Phase, capacity and support evolution | Not supplied by the nodal product or the local Hessian; auxiliary mobilities and any event occurrence remain unjustified unless separately derived. |
| Existing capacity adaptation | [adaptation.py](../src/tnfr/dynamics/adaptation.py) implements configured averaging and pressure/Si gates; its gains, timing and activation policy do not establish this missing potential. |
| Reduction and memory | [epi_memory.py](../src/tnfr/physics/epi_memory.py) derives its closure/kernel from supplied dynamics with held capacity/forcing; [operator quotients](../src/tnfr/physics/operator_quotient.py) check supplied maps. Neither derives a law for those held inputs. |
| Physical emergence | Requires a justified complete mechanism, spontaneous formation and independently testable consequences; the present conditional obstruction establishes none of these. |

The exact regression extension in
[test_constitutive_variational_scope.py](../tests/physics/test_constitutive_variational_scope.py)
reuses the existing forced-potential helper and exact semidefinite primitive.
No new public certificate, runtime potential or phase/capacity update is
needed for this bounded result. The
[single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) records the
constitutive decision required before another origin experiment.

### 13.6 Exact state-dependent metric for canonical phase pressure

The nonlinear pressure admits a geometric representation that the
constant-metric test in section 13.3 could not decide. Fix reciprocal,
unweighted **unique neighbor support**, including any zero-conductance
support edges. This is the phase channel's support, not the weighted EPI
conductance. At a node with a nonzero neighbor resultant, put

\[
S_i=\sum_{j\in N(i)}e^{i\theta_j},\quad r_i=|S_i|>0,\quad
\delta_i=\operatorname{wrap}(\arg S_i-\theta_i),\quad
g_i=\delta_i/\pi.
\]

Work on the open branch `|delta_i|<pi` at every node. Let

\[
V_\phi(\theta)=\sum_{\{i,j\}\in U}\bigl(1-\cos(\theta_j-\theta_i)\bigr)
=\frac12\sum_{\{i,j\}\in U}|e^{i\theta_j}-e^{i\theta_i}|^2.
\]

Each undirected distinct-neighbor edge is counted once. A self-loop has
zero cost and zero derivative, although its own phasor still contributes to
`S_i` and changes the metric below. Parallel multiplicity and conductance
weights do not multiply this cost. It is an alignment cost derived from
the specified phasor geometry, not a new definition of tetrad energy or
physical energy. Choosing a scale for it as part of a joint potential
would require the corresponding units and a constitutive premise.

Reciprocity gives the exact derivative

\[
\partial_i V_\phi=-\sum_{j\in N(i)}\sin(\theta_j-\theta_i)
                 =-r_i\sin\delta_i.
\]

Writing `sinc(delta)=sin(delta)/delta`, continuously extended by one at
zero, therefore gives

\[
H_\phi(\theta)=\operatorname{diag}
  \left(\pi r_i\operatorname{sinc}\delta_i\right)>0,\qquad
g=-H_\phi^{-1}\nabla V_\phi.
\]

The diagonal mobility is
`M_i=delta_i/(pi*r_i*sin(delta_i))`, with the regular limit
`M_i=1/(pi*r_i)` at zero displacement. These factors are fixed by the
displayed pressure/cost representation; they are not primitive capacity
or an angular clock. A positive constant rescaling of the cost rescales this metric by the
same factor without changing `g`, so no absolute energy scale is inferred.

Equivalently, set `z_i=exp(-i*theta_i)*S_i=C_i+i*A_i`, using `A_i` here
only for the imaginary resultant component. The full regular domain is
`z_i` outside the nonpositive real axis. It is wider than either strict
acute edges or positive real relative resultants. On that domain

\[
H_i=\pi A_i/\operatorname{Arg}z_i
\quad(\operatorname{Arg}z_i\ne0),\qquad
H_i=\pi C_i\quad(z_i=C_i>0).
\]

The imaginary component and principal argument have the same sign,
including when `C_i<0`; the metric therefore stays positive. The second
formula is its removable limit, not a value assigned at a negative-real
branch or zero resultant. No new state variable or phase law is introduced
by this wider domain. The [relational execution owner](nodal/RELATIONAL_DOMAIN_AND_CAPTURE.md#relational-full-regular-execution)
separately certifies sufficient point and straight-proposal margins using
rational enclosures; it retains the existing acute and positive-resultant
options. This representation does not establish phase dynamics from the
nodal form equation alone or turn every regular state into a stable pattern.

The previously established constant-metric obstruction remains correct.
Here differentiating `H_phi*g=-gradient V_phi` includes
`(dH_phi)*g` as well as `H_phi*Dg`. Testing only the latter for symmetry
would incorrectly reject a variable metric. The star counterexample in
section 13.3 is a useful control for this distinction.

**Complete response identity.** Write `J=Dg`, `H=H_phi`, and let `DH`
denote the matrix with entries `partial_j H_i` for the diagonal coefficients.
On this regular domain, differentiating every component gives

\[
H J+\operatorname{diag}(g)DH=-\operatorname{Hess}V_\phi.
\]

Thus the antisymmetric part of `HJ` is exactly the negative of the
antisymmetric part of `diag(g)DH`. This accounts for the entire matrix,
not just one off-diagonal entry. It does not make `J` reversible or remove
the retained response's cycle ratio `4/5`. At `g=0`, the correction vanishes
and `HJ=-Hess(V_phi)` is symmetric. For direct evaluation, with
`T_ij=1[j in N_i]*sin(Arg S_i-theta_j)`,

\[
\partial_j H_i=\pi\operatorname{sinc}(\delta_i)T_{ij}
 +\pi r_i\operatorname{sinc}'(\delta_i)(R_{ij}-\delta_{ij}),
\]

where `sinc(0)=1`, `sinc'(0)=0`. All 36 entries are checked at the retained
nonrepeated preparation and its repeated comparator, including removable
zero displacements. The underlying covector identity is stronger:

\[
g^TH\,d\theta=-dV_\phi,
\qquad \oint g^TH\,d\theta=0
\]

for a closed regular phase loop. There is no independent circulating work
in this **metric-weighted phase covector**. The unweighted source response
and the joint EPI source work are different objects; neither vanishes by
this argument. In particular it specifies no phase velocity. The maintenance
implication is the joint budget in section 13.18, not an invented phase
propagator `R`. Controls:
[complete metric and phase-work identities](../tests/physics/test_phase_metric_transfer_identity.py).

**Current and curvature share this geometry.** With
`n_i=|N(i)|`, the existing
[phase current](../src/tnfr/physics/extended.py) reads
`J_phi(i)=r_i*sin(delta_i)/n_i`. The regular canonical curvature has
`K_phi(i)=-delta_i`; hence

\[
\nabla_i V_\phi=-n_iJ_\phi(i),\qquad
J_\phi(i)=-\frac{r_i}{n_i}\sin K_\phi(i).
\]

Nonzero current and curvature have **opposite** signs on this domain.
Current and phase pressure have the same sign. The former documentation's
same-sign/current-curvature claim does not match these definitions and has
been removed. None of these instantaneous directional statistics measures
an actual phase velocity without a phase evolution law.

The domain exclusions matter:

- At `delta=0` with `r>0`, the apparent `0/0` is removable.
- At an exact half-turn, the sine torque can vanish while `|g|=1`;
  no finite positive diagonal mobility represents that component. The metric
  degenerates on approach to that branch boundary.
- At a zero neighbor resultant the circular mean has no direction. Engine
  fallback conventions or represented residuals do not supply this theorem's
  missing ideal direction. Isolates likewise require their separate zero
  convention rather than division by zero.
- Strict edge U3 (`|wrap(theta_j-theta_i)|<pi/2`) implies a positive
  real component of every nonempty rotated neighbor sum and therefore
  `r>0`, `|delta|<pi/2`. It is a sufficient current-state domain condition,
  not a proof that a future trajectory remains there.
- Directed neighbor support does not generally yield this gradient from
  the undirected cost; reciprocal support is part of the claim.

Exact trigonometric identities and represented observations remain distinct.
For example binary64 `sin(pi)` is not ideal zero. The shared forcing capture
and current/curvature owners verify numerical readings with their explicit
residual or tolerance; exact symbolic controls establish the limiting and
branch statements. No trigonometric reimplementation or energy API is needed.

#### Classification of edge-additive storage aligned with native pressure

<a id="native-phase-storage-classification"></a>

The cosine cost is also selected within a specific **additional storage
class**, not merely an available representation. Restrict here to all finite
connected simple unit graphs with at least two vertices, and the full regular
phase domain above. Let the same even, `2*pi`-periodic, continuously
differentiable edge cost `u` define

\[
U_\phi=\sum_{\{i,j\}\in E}u(\theta_i-\theta_j),\qquad u(0)=0.
\]

Assume that, on every admitted state and graph, a finite positive diagonal
metric satisfies `grad(U_phi)=-H_U*g`, with the native resultant pressure
`g` unchanged. Edge additivity and this alignment are explicit premises;
neither follows from the nodal product. Then, for one positive constant
`kappa`,

\[
\boxed{u(\delta)=\kappa(1-\cos\delta),\qquad
U_\phi=\kappa V_\phi.}
\]

`C1` regularity is sufficient; no assertion that it is minimal is needed.
Metric continuity is not required to classify the cost, but has a separate
role at zero pressure below.

**Proof.** Write `f=u'`, which is odd with `f(0)=0`. Reuse the
[phase-balanced star construction](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#primitive-locality-phase-clock).
For any finite nonantipodal gaps `delta_1,...,delta_m` with zero sine sum,
append `m+1` aligned leaves. The center has positive real resultant at least
one and `g_center=0`, and every leaf is regular. Finite-metric alignment
therefore requires

\[
0=\partial_{\theta_{center}}U_\phi
 =-\sum_{r=1}^m f(\delta_r).
\]

Aligned leaves contribute `f(0)=0`. The balanced pair
`delta,-arcsin(sin(delta))` shows that `f` factors through the sine, including
obtuse gaps. Define `b(s)=f(arcsin(s))` on `[-1,1]`. Balanced triples give
`b(s+t)=b(s)+b(t)` whenever all three arguments are in that interval. The
same continuous additive-function argument as in the linked proof gives
`b(s)=kappa*s`. Thus `u'(delta)=kappa*sin(delta)` on nonantipodal gaps;
continuity and periodicity include the endpoints. Integrating and using
`u(0)=0` yields the displayed cost. On P2 at a nonzero regular gap, alignment
requires `H_U=kappa*pi*sinc(delta)>0`, so `kappa>0`. Conversely, the existing
representation supplies `H_U=kappa*H_phi`, proving sufficiency.

**Metric at zero pressure.** Where `g_i!=0`, the product fixes
`(H_U)_i=kappa*(H_phi)_i`. Where `g_i=0`, both sides vanish, and any finite
positive value at that point satisfies the product. Continuity on the full
regular phase domain selects the same removable limit: nonzero `g_i` states
are locally dense, since varying the node phase with neighbors held gives
`partial_i g_i=-1/pi`. Without that continuity condition, changing only the
zero-pressure metric entries preserves alignment but not the regular metric
field. A numerical tolerance is not an exact zero set or a continuation rule.

**An acute obstruction to the alternative edge cost.** Section 13.8's
`V_phi+Q_phi`, with `Q_phi=sum_edges(1-cos(delta))^2`, retains its stated
symmetries and consensus Hessian, but fails this stronger alignment condition
even if a different finite diagonal metric is allowed. Take a three-leaf
star with center phase zero and leaf phases
`(alpha,alpha,-pi/6)`, where `alpha=arcsin(1/4)`. Every edge is strictly
acute. The center resultant is `(sqrt(15)+sqrt(3))/2>0`, its sine sum is
zero, and therefore

\[
g_{center}=\partial_{center}V_\phi=0,\qquad
\partial_{center}Q_\phi=\frac{\sqrt{15}-2\sqrt3}{4}>0.
\]

No finite diagonal metric can multiply zero pressure into that nonzero
gradient. This is a static obstruction, not a failed trajectory. It does not
invalidate section 13.8's auxiliary-potential nonselection, which did not
impose the present native-pressure alignment.

**Limits of selection.** If alignment is required only on strictly acute
edge states, the argument fixes the cost only for `|wrap(delta)|<pi/2`.
A smooth even periodic addition `epsilon*b(cos(delta))`, with
`b(t)=exp(-1/t^2)` for `t<0` and zero otherwise, vanishes throughout that
acute interval but changes obtuse costs. Global classification therefore
uses the stated full regular graph family; a restricted execution chamber
cannot establish that global classification on its own. Nor does alignment derive edge
additivity: `U=V_phi+V_phi^2` has the positive metric
`(1+2*V_phi)*H_phi` on the regular domain. That different construction
generally consumes global storage and is outside the shared edge-cost class;
its induced global metric factor is not an admitted primitive-local phase
completion under the separate locality theorem.

**Scale and joint-law integration.** With fixed Dirichlet form storage,
`E_D+beta_U*U_phi=E_D+beta_eff*V_phi`, where `beta_eff=kappa*beta_U`.
The existing relational engine fixes the edge-cost normalization to one and
already consumes this effective value as `RelationalExchangeModel.storage_scale`.
No second cost-scale parameter is needed. Holding `beta_U` fixed while changing
`kappa` changes the phase rates; it is not a harmless relabeling. The relative
invariant remains `chi=beta_eff*(e/w)^2`, whose value is not selected here.
The condition `u(0)=0` also fixes an otherwise free constant per edge; such a
constant drops out of fixed-support flow but can change support-event storage
jumps. A convention for the former is not a derived event budget for the latter.

The [alignment controls](../tests/physics/test_phase_alignment_metric.py)
retain the exact obstruction, metric continuation, nonadditive alternative
and effective-scale accounting, alongside the existing
[balanced-star controls](../tests/physics/test_relational_locality_selection.py).
This result classifies a cost under explicit premises. It does not derive
Dirichlet form storage, joint separability, an absolute energy scale, physical
energy or primitive support formation.
The separate [joint-cost theorem](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#joint-storage-locality-classification)
does derive the quadratic/cosine split from a prescribed exact loss and uniform
primitive-local completion on the full regular graph family. It admits mixed
terms initially and needs neither the alignment premise used here nor a
preselected phase row. Its stronger premises retain their own proof and scope.

### 13.7 Phase alignment representation does not close the joint law

Representing `g` as a gradient does not establish `theta_dot=g`. To examine
that tempting completion, retain the section 13 premises: the EPI block
uses `M_x=diag(nu/d)`, there are no cross-mobility blocks, the phase block
uses the newly derived `H_phi^-1`, and `Psi` is independent of EPI. With
`J_g=(R-I)/pi`, the required phase velocity is

\[
\dot\theta=w_\phi H_\phi^{-1}J_g^\top D x
                  -H_\phi^{-1}\nabla_\theta\Psi.
\]

Making this equal `g=-H_phi^-1*gradient V_phi` would require
`gradient_theta Psi=gradient V_phi+w_phi*J_g^T*D*x`. For `w_phi>0` and
`J_g!=0`, an EPI-independent `Psi` cannot satisfy that identity throughout
an open EPI domain. The metric representation thus exposes the missing
reciprocal interaction rather than removing it. Other block metrics,
cross mobilities, constraints and non-gradient laws remain separate cases;
the nodal identity still selects none of them automatically.

The retained unit prism gives a concrete same-phase comparison. Write
`q=(1,-1,0,1,-1,0)`, `r_0=1+sqrt(3)` and keep the prepared phases
`(0,pi/3,pi/6)` in each triangle. Then `Rq=R^Tq=0`, `D=3I` and
`H_phi*q=3*r_0*q`. Two EPI states differing by `epsilon*q` therefore
require phase velocities differing by

\[
-\frac{w_\phi\epsilon}{\pi(1+\sqrt3)}q.
\]

Every `Psi` term cancels in this comparison. For `w_phi=1/4` and
`epsilon=1/16`, the difference is `-q/(64*pi*(1+sqrt(3)))`, whereas the
isolated pressure vector `g` is identical in the two preparations. This is
an exact conditional model obstruction, not a phase update or a runtime
derivative certificate.

There is also a useful **conditional budget**, without installing that
completion. Suppose a supplied exact-real model does use `theta_dot=g`
and stays on the regular unit-prism domain for all time, with
`H_phi>=h_min*I`, `h_min>0`. Assume fixed unit capacity and conductance,
fixed EPI coefficient `e>0`, fixed phase coefficient `w>=0`, no other
nonzero forcing, events or pressure residuals. Then

\[
\dot V_\phi=-\sum_i H_{\phi,i}g_i^2,
\qquad \int_0^\infty\|g\|^2dt\le V_\phi(0)/h_{\min}.
\]

Combining this with the existing internal-form budget and Young's
inequality gives, for `S=|z_0|^2+|z_1|^2`,

\[
\dot S\le-eS+\frac{w^2}{e}\|g\|^2,\qquad
\frac{d}{dt}\left(S+\frac{w^2}{e h_{\min}}V_\phi\right)\le-eS.
\]

Consequently `integral S dt<=S(0)/e+w^2*V_phi(0)/(e^2*h_min)`.
The scalar comparison bound is an exponentially decaying initial term
plus the convolution of `exp(-e*t)` with an integrable nonnegative input;
both tend to zero, so `S(t)->0`. Thus this separately supplied phase-only
relaxation cannot maintain nonzero internal amplitude under the stated
uniform regularity hypotheses, even if it initially supports growth.
It may retain phase structure with zero source; the result is not a theorem
of phase consensus or a rejection of other TNFR completions. If the metric
approaches zero or a branch/resultant boundary, this uniform bound does
not apply. No future-domain or engine-trajectory claim is made here.

Portable controls reuse the existing current, curvature, forcing, phase
response and regional-balance owners:
[phase alignment metric](../tests/physics/test_phase_alignment_metric.py) and
[joint reciprocity and cost budget](../tests/physics/test_joint_phase_alignment_scope.py).
No second field-energy implementation or phase solver is introduced.

The new result is the intrinsic metric/cost representation and its
conditional implications. A law generating a joint source, its reciprocal
response, and an appropriate phase/capacity time scale remain open. The
[single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md) retains those
dependencies before any maintenance experiment.

### 13.8 Structural restrictions do not select the auxiliary potential

Retain the section 13.7 potential family and phase metric. Even common-phase
rotation invariance, individual `2*pi` periodicity, graph relabeling covariance,
undirected edge locality, nonnegativity and the same consensus Hessian do not
select `Psi`. In the already declared nondimensional comparison, put

\[
Q_\phi=\sum_{\{i,j\}\in U}(1-\cos(\theta_j-\theta_i))^2.
\]

The two logical alternatives `Psi=V_phi` and `Psi=V_phi+Q_phi` have all those
properties. Neither is installed as a constitutive law. Since each edge cost
`a=1-cos(delta)` lies in `[0,2]`, `0<=Q_phi<=2*V_phi`; the alternatives have
identical zeros. The added term has zero gradient and Hessian at consensus,
so their quadratic normalization and linearization agree. On strict U3 edges,
the respective edge curvatures are

\[
\cos\delta>0,\qquad
\cos\delta+2(1-\cos\delta)(1+2\cos\delta)>0.
\]

Thus even local convexity on that phase domain does not remove the freedom.
Dimensional analysis can require a cost scale; it cannot select between these
two dimensionless edge functions after that scale is fixed.

At the prepared prism phases, with `q=(1,-1,0,1,-1,0)` and
`r_0=1+sqrt(3)`, exact evaluation gives

\[
Q_\phi=\frac{15}{2}-4\sqrt3,\qquad \nabla Q_\phi=-q.
\]

For the same state, coefficients and phase block `H_phi^-1`, replacing the
first alternative by the second leaves the instantaneous EPI rate unchanged
but changes the conditional rates by

\[
\Delta\dot\theta=\frac{q}{3r_0},\qquad
\Delta\dot p=-\frac{q}{12\pi r_0},\quad w_\phi=\frac14.
\]

These are ideal derivatives of two supplied joint models, not derivatives
of binary64 trigonometric execution. The captured current-pressure residual
is retained separately. Nor does `V_phi+Q_phi` represent the original isolated
`g` under the same metric: this comparison concerns the still-free auxiliary
part of the joint potential. No phase clock or full-pattern stability follows.
The [edge-storage classification](#native-phase-storage-classification) gives
the separate acute zero-pressure obstruction to representing `V_phi+Q_phi`
with any finite diagonal metric while retaining native `g`. Its added alignment
premise does not change the present auxiliary-potential result.

### 13.9 Joint source compatibility and the common-mean saddle

Fix support, conductance and capacity, and write `C(theta)=d^T F(theta)`.
The letter `C` here denotes source compatibility, not the coherence metric.
The conditional potential family forced by the EPI block is

\[
\mathcal E(x,\theta)=\frac e2x^TBx-x^TDF(\theta)+\Psi(\theta),
\qquad B\mathbf1=0.
\]

Other held arguments of `Psi` and `F` are suppressed. Under a common EPI shift,

\[
\mathcal E(x+c\mathbf1,\theta)-\mathcal E(x,\theta)=-cC(\theta),
\qquad
\Delta\dot\theta=cH_\phi^{-1}\nabla C.
\]

Pointwise source compatibility `C=0` does not establish phase-rate covariance,
whose condition here is `gradient C=0`. These two pointwise conditions are
logically independent. Indeed phase-rotation invariance
gives `1^T*gradient C=0`. A positive definite phase mobility cannot map a
nonzero `gradient C` to a pure common rotation, since that would contradict
its positive quadratic form. This tests inheritance of the frozen nodal
flow's shift symmetry; it does not declare absolute EPI unphysical or remove
the Emission/vacuum contracts.

**Centering changes the law unless justified separately.** For fixed weights
`rho^T*1=1`, replacing the cross term by `-(Px)^T D F`,
`P=I-1*rho^T`, changes the pressure to

\[
p_{\rm centered}=p-C D^{-1}\rho.
\]

Its phase derivative differs by `-D^-1*rho*(gradient C)^T`, even at a point
where `C=0`. Degree-weighted centering subtracts uniform pressure `C/sum(d)`;
centering with weights proportional to `d/nu` subtracts a uniform nodal drift.
An invertible chart retaining the mean retains the omitted coupling term.
A fixed-mean constraint or an invariant `C=0` manifold therefore needs its
own derivation; it is not supplied by recentering a diagnostic.

**Stationary-pattern obstruction, independent of `Psi`.** Assume a `C^2`
potential on an open EPI/phase chart. At a point with `C=0` and
`gradient C!=0`, choose a phase direction `v` with
`b=gradient C dot v!=0`. Restrict its second variation to
`x=x_*+a*1`, `theta=theta_*+t*v`. Its Hessian on this plane is

\[
\begin{pmatrix}0&-b\\-b&k\end{pmatrix},\qquad \det=-b^2<0,
\]

where `k` includes every possible `C^2`, EPI-independent `Psi`. The zero
entry follows from `B*1=0`, and the mixed entry from `-a*C(theta)`.
Consequently any full critical point there is a saddle; if it is not
critical, it already cannot be an interior local minimum. Smooth symmetric
positive definite gradient mobility makes such a critical point linearly unstable:
`-M*Hessian(E)` is similar to the symmetric matrix
`-sqrt(M)*Hessian(E)*sqrt(M)`, which has a positive eigenvalue. This last
statement also applies to cross mobility for this same potential family;
allowing cross mobility when deriving the EPI law in the first place can
change the family and is not excluded by this argument.

The retained source has exactly this obstruction. For the prepared phases,
unit capacity/conductance and `e=1/2`, `w_phi=1/4`, put

\[
q=(1,-1,0,1,-1,0),\quad s=(1,1,-2,1,1,-2),\qquad
x_*=\tfrac12\mathbf1+\tfrac1{12}q.
\]

This is an **ideal exact** zero-pressure reference with source `F=q/24`;
it is distinct from the earlier growing preparation with amplitude `1/16`.
Although `C=0`,

\[
\nabla C=-\frac{w_\phi\gamma}{\pi}s\ne0,\qquad
\gamma=\frac{3(2-\sqrt3)}{2(1+\sqrt3)}>0.
\]

The reference EPI lies strictly between zero and one and the largest edge
phase gap is `pi/3<pi/2`, so sufficiently small two-sided variations are
admissible. No binary64 zero-pressure equilibrium is asserted. As an
additional control, the hypothetical completion `Psi=V_phi` at this reference
has `C_dot=(1/2)*gradient C^T*H_phi^-1*gradient C>0`: initial compatibility
does not even persist under that particular trial law.

**Repeated-triple family corollary.** This is not just an isolated phase
choice. On the unit prism, let each triangle carry the same arbitrary three
phases, with common nonzero neighbor resultant `r*exp(i*beta)` and a regular
phase-source chart. Every row contains one copy of each of those phases.
Column sums of `R` are `3*cos(theta_j-beta)/r`. Hence `J_g^T*d=0` forces
each of the three cosines to equal `r/3>0`. Their sines have equal magnitude
and sum to zero by definition of `beta`. Three signed copies of a nonzero
magnitude cannot sum to zero, so every sine vanishes. The positive cosine
then forces phase consensus modulo `2*pi`. No semicircle premise needs to
be added; derivative compatibility itself implies it.

For `w_phi>0` and uniform capacity on this unit support, every nonconsensus
regular repeated triple therefore has either `C!=0`, excluding a critical
point in the common-EPI direction, or `C=0` and `gradient C!=0`, giving the
saddle above. This excludes an interior stationary minimum supported by
such phases throughout this family, for every admissible `Psi`. It does
not classify arbitrary six-phase configurations, resultant singularities,
other supports or moving patterns.

The obstruction does not require the two-dimensional test plane to be
dynamically invariant. Allowing capacity to vary cannot restore an interior
minimum while this negative direction remains admissible. Conversely a
derived fixed-mean constraint, a boundary point, degenerate mobility,
non-gradient motion, moving identity or a different justified joint law lies
outside the conclusion. This rejects stabilization of this preparation by
choosing `Psi` alone in the stated gradient class, not autonomous TNFR
patterns generally.

Controls reuse the existing forced-support solver and the shared prism
geometry/phase-metric fixture:
[auxiliary freedom](../tests/physics/test_phase_auxiliary_freedom.py) and
[compatibility and stability](../tests/physics/test_joint_source_compatibility_stability.py).
The [single execution plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns the next admission gate; no new potential, dynamics API or trajectory
is introduced here.

### 13.10 Persistent motion is a separate dynamical target

A maintained NFR need not be a stationary point. The interior-minimum
obstructions above concern one conditional gradient class; they do not
define persistence for the whole TNFR programme. A periodic state, a relative
equilibrium under a stated symmetry, a response sustained by its environment,
and a transient oscillatory observation are different claims. Specify the
full state, the identity observation and any allowed symmetry before testing
one. Constant amplitude alone does not establish identity or robustness.

There is nevertheless a useful stronger exclusion **within pure gradient
dynamics**. Let the complete continuous state obey

\[
\dot z=-M(z)\nabla\mathcal E(z),\qquad M=M^T\succeq0,
\]

with a single-valued, time-independent differentiable energy and sufficiently
regular autonomous flow. Then

\[
\dot{\mathcal E}=-\nabla\mathcal E^T M\nabla\mathcal E\le0.
\]

A periodic return, recurrence to the initial state, or motion entirely along
an energy-invariant symmetry orbit forces the energy to be constant. For a
positive semidefinite symmetric matrix, zero quadratic form implies
`M*gradient E=0`, hence `z_dot=0`. Thus even degeneracy cannot generate a
nonconstant recurrent orbit in this pure gradient model. Positive definiteness
additionally forces `gradient E=0`. This does not exclude a transient
oscillatory projection, a nonrecurrent drifting state or recurrence of a lossy
observation while hidden energy is consumed.

A common phase rotation preserves canonical phase differences and regular
`g`; it changes neither the phase source nor its work by itself. If the joint
energy is rotation invariant, that rotation is also excluded as nonzero
motion of a pure gradient law. A supplied angular clock can generate it in a
different law, but the phase symmetry does not derive that clock.

Possible recurrent completions must therefore justify something absent from
this restricted class: for example a circulation term, an explicit sustaining
boundary, a time-dependent input, or hybrid energy transfers with their actual
occurrence law. The algebra `z_dot=(J-M)*gradient E`, `J^T=-J`, shows why an
antisymmetric term does no energy work. It does **not** derive `J`, a Poisson
structure or a TNFR phase velocity. The existing auxiliary substrate and
decoupled product in section 3 retain their realizability/closure obligations.

### 13.11 Cycle budgets reuse the nodal transport owners

Pointwise pressure equilibrium is too strong a test for a moving pattern.
For fixed symmetric connected conductance, positive fixed capacity and a
declared source `F(t)`, the actual EPI model is

\[
\dot x=\operatorname{diag}(\nu)(-eL_{\rm rw}x+F(t)),\quad
H=\operatorname{diag}(d/\nu),\quad
m_H=\frac{\mathbf1^THx}{\mathbf1^TH\mathbf1}.
\]

Writing `C(t)=d^T F(t)` gives

\[
\dot m_H=\frac{C(t)}{\sum_i d_i/\nu_i}.
\]

A periodic EPI field requires `integral_0^T C(t) dt=0`, not `C(t)=0` at
every instant. A relative pattern translating in common EPI may have a
nonzero mean drift; it is not a bounded recurrent full EPI state. The held
case is already implemented by `derive_forced_support_balance`; no second
Poisson solver or mean-centering law is needed.

For the retained unit prism with unit capacity, let `y` be the within-triangle centered EPI,
`z_a=sqrt(2)*u_a+i*sqrt(6)*v_a`, `S=|z_0|^2+|z_1|^2=||y||^2` and
`D_z=|z_0-z_1|^2`. Summing the existing exact regional balances gives

\[
\dot S=-2eS-\frac{2e}{3}D_z+2\langle y,F\rangle.
\]

Whenever `S(T)=S(0)`,

\[
\int_0^T\langle y,F\rangle\,dt
=e\int_0^T S\,dt+\frac e3\int_0^T D_z\,dt.
\]

For positive `e` and continuous nonzero internal form this work is positive.
A zero-average source can still supply positive work through its correlation
with the moving form. Returning `S` is only an accounting condition, not a
return of the full pattern. Stored-pressure defects add their own work;
changing capacity/geometry, boundaries, Euler steps and events require the
existing derivative/finite/reset owners. None may be silently omitted.

**An exact driven control with canonical phase pressure.** Set
`q=(1,-1,0,1,-1,0)` and prescribe identical phase triples `(-a(t),a(t),0)`
in the two triangles. For `a(t)=A*cos(omega*t)`, `0<A<pi/4`, `omega>0`,
every neighbor resultant is the positive real number `1+2*cos(a)` and the
largest edge gap is below `pi/2`. The canonical ideal pressure is therefore
`g=(a/pi)*q`. Use nondimensional structural time with common capacity
normalized to one. With `F=w*g`, `L_rw*q=q` and `e,w>0`,

\[
x=\mu\mathbf1+u(t)q,\qquad
u(t)=\frac{wA}{\pi(e^2+\omega^2)}
       (e\cos\omega t+\omega\sin\omega t)
\]

solves `u_dot=-e*u+w*a/pi` exactly. In an unscaled time with common capacity
`nu`, replace the decay coefficient by `nu*e` and forcing coefficient by
`nu*w`; the denominator becomes `(nu*e)^2+omega^2`. Here `S=4u^2` and the cycle source work
equals dissipation. If a scalar EPI band is required, choose a declared mean
and amplitude whose entire analytic range lies in that band. The general
scalar solution differs by `constant*exp(-e*t)` under this **supplied** source.
No phase evolution is derived, no graph is integrated and no autonomous
periodic orbit of the full state is certified.

The response amplitude is `w*A/(pi*sqrt(e^2+omega^2))`, strictly decreasing
for `omega>0`. It has no nonzero-frequency resonance peak in this particular
first-order channel. This does not rule out resonance in another justified
joint law; it prevents mistaking a periodically driven relaxation for such
a derivation. Neither this frequency nor the auxiliary graph-wave
`sqrt(lambda)` is selected by the nodal product alone.

The exact controls are in
[test_resonant_pattern_scope.py](../tests/physics/test_resonant_pattern_scope.py).
They reuse the shared prism and regional budgets, and separate the analytic
source from represented runtime pressure. The
[resonant-persistence audit](NODAL_RESEARCH_STRATEGY.md#resonant-persistence-and-pulse-audit)
maps the existing implementations; the single execution plan retains the
constitutive gate for a real autonomous mechanism.

### 13.12 Oriented source/form work without a selected clock

The preceding cycle balance concerns the squared internal amplitude. A
different identity tests the **oriented joint loop** without first choosing
its traversal speed. Fix symmetric conductance with positive row strengths,
positive fixed capacities and coefficient `e>0`. Set `B=D-W`,
`H=diag(d/nu)` and retain the actual EPI law

\[
H\dot x=-eBx+DF.
\]

Taking its scalar product with `dx=dot(x)*dt` and integrating an actual
trajectory segment yields

\[
\int F^TD\,dx
=\frac e2[x^TBx]_{t_0}^{t_1}
 +\int_{t_0}^{t_1}\dot x^TH\dot x\,dt.
\]

When `x(t_1)=x(t_0)`, this becomes

\[
\boxed{\ \int_{t_0}^{t_1} F^TD\,dx=\int_{t_0}^{t_1}\|\dot x\|_H^2dt>0\ }
\]

whenever its continuously differentiable EPI path is nonconstant. This is
work against form displacement, distinct from the amplitude-budget integral
`integral <y,F> dt` in section 13.11. The left side is an oriented line
integral along the traced joint `(x,F)` path. If the source also returns,
this is a closed joint loop and may be written with `oint`. Closing EPI
alone does not guarantee that stronger return. The integral is unchanged by an
orientation-preserving reparameterization, negated by reversing the path,
and zero for a constant held source on a closed EPI path. A nonconstant
periodic EPI path under a held source is therefore excluded in this fixed
model without choosing a global time unit.

The right side uses the actual time/capacity of an admitted solution. Arbitrarily
changing its speed while keeping the same fixed capacities and pressure law
does not generally produce another solution. Under a consistent change
`d tau/dt=h>0`, `dx/dtau=dot(x)/h` and the effective EPI metric is `H_tau=hH`,
so its kinetic integral is preserved. This is a change of coordinate for
the complete supplied vector field, not a constitutive invariance theorem
with unchanged coefficients. The multichannel pressure's capacity dependence
must also be transformed consistently. The parameter foundations derive the
separate constant-time-unit special case; they do not establish arbitrary
state-dependent clock covariance of the configured engine. The line integral
is not a license to rescale one channel in isolation.

**Reciprocal phase geometry.** If all nonphase source components are held and
`F=w_phi*g(theta)+F_0`, a closed joint source/form loop satisfies integration
by parts:

\[
\oint F^TD\,dx
=-w_\phi\oint x^TD J_g(\theta)\,d\theta.
\]

The regular source Jacobian `J_g=(R-I)/pi` is the existing response owner.
Common rotation has `J_g*dtheta=0` and cannot supply this loop work. More
generally a finite path preserving the entire source gives zero work when
EPI closes. Relative source/form variation with the proper orientation is
necessary here; a high synchronization score alone says nothing about it.
Positive varying capacity alone leaves the first work identity valid with
the instantaneous `H(t)=diag(d/nu(t))`: no metric derivative is taken in
this velocity-action identity. If capacity variation also changes the source,
integration by parts must include that part of `dF`; the phase-only expression
then does not suffice. Changing support, conductance or `e` requires the
corresponding Dirichlet derivative and event terms. These belong to the existing
derivative and event budgets, not an unchanged fixed-coefficient formula.

For the supplied prism control of section 13.11, in normalized structural
time, exact evaluation gives

\[
\oint F^TD\,dx
=\frac{12w^2A^2\omega}{\pi(e^2+\omega^2)}.
\]

This equals the positive nodal-rate integral and the reciprocal phase line
integral. Reversing the same geometric loop makes the left side negative,
so that orientation cannot solve the same positive-capacity nodal law.
Changing `omega` in the displayed control changes the EPI amplitude and lag,
hence changes the loop itself; it is not just a relabeling of one loop.

The positive value is a necessary admission result, not a derivation of the
loop, its clock, attraction or autonomous phase feedback. The existing
common-mean/phase saddle and gradient recurrence exclusions are not bypassed:
any proposed reciprocal completion still needs its full energy/work balance.
The [relational-clock tangent test](nodal/FORCED_SOURCE_AND_CLOCK.md#25-relational-time-and-synchronization-are-separate-claims)
supplies another necessary condition before a curve may be called a trajectory.
Exact controls reuse the shared conductance and phase geometry in
[test_phase_form_cycle_work_scope.py](../tests/physics/test_phase_form_cycle_work_scope.py).
No new force, integration path or physical-time calibration is introduced.

### 13.13 Work pulled back to a proposed geometric phase relation

A phase observation proposed as an endogenous source must pass a geometric
test as well as the earlier full-vector-field test. Let `z` be coordinates
on a regular open chart, and declare `x=chi(z)`, `theta=Theta(z)`. Hold
conductance, the EPI coefficient, positive capacities and all nonphase source
components fixed. Put `F=w*g(Theta)+F_0`, with fixed `w>0`. The source work
one-form on this chart is

\[
\alpha=F^TD\,D\chi\,dz.
\]

For twice differentiable `chi` and a differentiable regular phase map, define

\[
K=w(D\chi)^TDJ_gD\Theta.
\]

The mixed derivatives of `chi` cancel in the exterior derivative of `alpha`;
the constant source contributes the exact differential `d(F_0^TD chi)`.
Consequently `d alpha=0` exactly when `K=K^T`. This criterion must hold
throughout the chart, not merely at one sampled state. On a simply connected
chart it makes `alpha` exact, hence every closed embedded loop has zero work.
Section 13.12 then excludes a closed embedded-state cycle with nonconstant
EPI under the stated law. EPI return alone suffices only if `chi` is
injective on the chart or otherwise guarantees return of the latent state;
a lossy EPI map can hide different endpoints of the work primitive.
On a domain with holes, closedness alone
does not establish global exactness. A nonzero antisymmetric part only permits
local circulation; it proves neither nodal tangency nor an autonomous orbit.

This reuses the canonical source Jacobian `J_g`, the actual conductance metric
`D`, and the proposed observation map. Testing symmetry of the raw phase
Jacobian, or only counting available coordinates, would answer a different
question. No additional field, energy API or phase solver is needed.

**Exact two-coordinate prism control.** Use the retained repeated-triple
chart, with `P=(1,-1,0)^2`, `Q=(1,1,-2)^2`,
`x=mu*1+u*P+v*Q`, fixed `mu`, and the declared phase relation
`Theta=beta*1-pi*kappa*(x-mu*1)`, `kappa>0`. Choose a simply connected
neighborhood of `(u,v)=(0,0)` with a fixed nonzero-resultant branch and
strict U3. Each node has one neighbor of each triple type, so its neighbor
resultant is common and

\[
g=\kappa(uP+vQ)+c(u,v)\mathbf1.
\]

The common component drops out of the fixed-mean work because
`1^T P=1^T Q=0`, `D=3I`, `P^T P=4`, `Q^T Q=12` and `P^T Q=0`:

\[
\alpha=w\kappa(12u\,du+36v\,dv)
       =d\{w\kappa(6u^2+18v^2)\}.
\]

This phase-from-form relation may support instantaneous amplitude growth but
cannot supply positive net work around a closed embedded loop. It is a
declared geometric candidate, not a derived phase law. Its fixed-mean chart
also need not be invariant: at `u=0`, `v=1/12`, `kappa=1`, the common resultant
has imaginary part `1/2-2*sin(pi/12)!=0`, so the common phase source generally
moves the EPI mean. Dropping that source would alter the model.

Allowing independent source and form coordinates can instead give nonzero
circulation. For `x=mu*1+uP`, `theta=-aP`, the work is
`alpha=(12w/pi)*a du`; its exterior derivative is nonzero. This is the
geometry already used by the supplied loop in section 13.12, not proof that
`a` has its own justified equation. A freely drawn loop can fail nodal
tangency at an EPI turning point despite positive total work.
Controls: [pullback scope](../tests/physics/test_phase_form_pullback_scope.py).

### 13.14 Existing optional feedback: pressure consistency before recurrence

The optional extended integrator contains a phase response depending on
pressure, so it is a concrete candidate to audit rather than inventing a
feedback term. Its full pressure block, however, evolves stored pressure
independently and fails the fresh-pressure chain rule on the retained prism.
The [constitutive audit](DIAGNOSTIC_AND_GRAMMAR_SCOPE.md#optional-pressure-composition-and-source-consistency)
owns that exact positive-squared-Laplacian obstruction. It is not a canonical
phase/form closure or a source of admitted resonant motion.

A separate conditional comparison isolates what the existing phase formula
would do **if** paired with freshly realized pressure. This substitution is
not the implemented optional model and is not installed as a new one. Retain
unit capacity, the repeated antisymmetric prism family `x=mu*1+uP`,
`theta=-aP`, no other source, and let `c=w_phi/pi>0`, `e=w_epi>0`. Then

\[
p=-\eta P,\quad \eta=eu-ca,\qquad
J_\phi=j(a)P,\quad j(a)=\frac{\sin(2a)+\sin a}{3}.
\]

Write the existing configured phase coefficients as `A=0.5`, `B=0.15`,
`G=0.135`, and hold the prism's common configured coupling `kappa>0` fixed.
Set `f(eta)=A*sin(pi*eta)+B*eta`, `k=G*kappa>0`. Its phase formula and the
fresh EPI product give

\[
\dot u=-\eta,\qquad \dot a=f(\eta)-k j(a),\qquad
\dot\eta=-e\eta-c f(\eta)+ck j(a).
\]

No coefficient or clock has been inferred by writing these supplied laws.
On `|eta|<1`, `|a|<pi/4`, consider the derived comparison function

\[
V=\int_0^\eta f(s)ds+ck\int_0^a j(s)ds
 =\frac A\pi(1-\cos\pi\eta)+\frac B2\eta^2
  +ck\left(\frac{1-\cos2a}{6}+\frac{1-\cos a}{3}\right).
\]

Direct substitution yields

\[
\dot V=-e\eta f(\eta)-c\{f(\eta)-k j(a)\}^2.
\]

Here `eta*f(eta)>0` for nonzero `eta` and `j(a)=0` only at `a=0`.
Thus `V_dot<0` away from `(eta,a)=(0,0)` on this chart. A nonconstant
periodic orbit contained there is excluded. This neither proves forward
invariance of the chart nor classifies its exterior or the actual optional
runtime. It shows why pressure-dependent feedback alone is insufficient
evidence for sustained resonance: the fresh-pressure comparison is dissipative
here, while the implemented independent-pressure completion has a different
and incompatible pressure slope. Exact and production-reader controls share
[one test module](../tests/physics/test_extended_pressure_feedback_scope.py).

The [phase/form exchange owner](nodal/PHASE_FORM_EXCHANGE.md#16-phase-and-form-directed-exchange-frames-and-the-moving-mean)
extends these work tests to radial/tangential source projections, transported
internal frames and a supplied circular phase-contrast response with a moving
mean. Its primitive source and clock remain prescribed. The full work includes
the mean even when the time integral of the mean source is zero.

The [phase-origin admission study](nodal/PRIMITIVE_PHASE_CLOSURE.md#17-primitive-phase-origin-symmetry-retained-state-and-the-missing-row)
complements this work test with actual prism reflection constraints and the
relative oriented-area budget of independent phase/form state. Its joint
linear classification places the same hypothetical fresh-pressure feedback
in a nonoscillatory stable contrast class near uniform state. The nonlinear
one-direction proof above and that local two-direction result have distinct
scopes; neither certifies the implemented independent-pressure extension.

### 13.15 Full repeated-triple feedback and the mean-work term

The one-direction result in section 13.14 extends to the two internal
directions only when the common phase pressure is retained. This remains a
**hypothetical fresh-pressure comparison** of the configured optional phase
formula, not the implemented independent-pressure model or a selected TNFR
completion. Fix the repeated unit prism, unit capacities, positive weights
`e,w`, `k=w/pi`, and held common `g>0`. Write

\[
p=h+q\mathbf1,\quad h=-ey-k\eta,\quad q=w c(\eta),\quad
v=\dot\eta=\Pi f(p)+gJ(\eta),\quad \dot h=-eh-kv,
\]

where `sum h=sum eta=0`, `Pi=I-11^T/3`, and
`f(s)=A*sin(pi*s)+B*s`, `A,B>0`. The common EPI rate is `mu_dot=q`, and
the common phase rate is `beta_dot=mean f(p)`. Let

\[
F(s)=\frac A\pi(1-\cos\pi s)+\frac B2s^2,\qquad
V(\eta)=\sum_{i<j}[1-\cos(\eta_i-\eta_j)].
\]

The actual three-neighbor mean-sine current is `J=-grad V/3`. Define the
comparison storage, counted once per repeated triple,

\[
\mathcal E=\sum_i[F(h_i+q)-F(q)]+\frac{kg}{3}V(\eta).
\]

If all pressures and their mean lie in a convex interval where `f'>=m>0`,
Jensen's inequality gives `E>=m*||h||^2/2+(kg/3)V>=0`. Exact differentiation
using both nodal rows yields

\[
\boxed{\quad
\dot{\mathcal E}=-e\sum_i h_i f(p_i)-k\|v\|^2
       +\underbrace{\left[\sum_i f(p_i)-3f(q)\right]
                    w\,Dc(\eta)[v]}_{\mathcal W_{\rm mean}}.
\quad}
\]

The last term is a nonlinear mean-pressure contribution. It vanishes for
affine `f`, but can have either sign for the sine response, even at a state
where `c=0`: zero current value does not imply zero derivative. Omitting it
would turn the one-direction dissipation proof into a false unrestricted
two-direction proof. It is a balance term, not evidence of an independently
generated energy supply.

**A sufficient complete-mean dissipation criterion.** Suppose on a declared
domain that `f'>=m>0`, `|f''|<=M`, `||h||<=H` and the derivative of `c`
restricted to centered phase directions has norm at most `C`. Taylor's
theorem and `sum h=0` give

\[
\left|\sum_i f(p_i)-3f(q)\right|\le\tfrac M2\|h\|^2,
\qquad
\sum_i h_i f(p_i)\ge m\|h\|^2.
\]

Thus, with `gamma=w*M*C*H/2`,

\[
\dot{\mathcal E}\le-em\|h\|^2-k\|v\|^2+
                    \gamma\|h\|\|v\|.
\]

The right side is strictly negative away from `h=v=0` whenever
`gamma^2<4*e*m*k`. This is a declared-domain test; its bounds are not
new evolution coefficients. Within the strict phase chart, `h=v=0`
also implies `eta=0`: then `v=gJ`, and
`eta^T J=-(1/3)sum_pairs (eta_i-eta_j)*sin(eta_i-eta_j)<0`
for nonzero centered `eta`. The complete stationary contrast has `q=0`.

**An explicit finite-pressure class.** On any centered triple whose phase
spread is strictly less than `pi/2`, write `a=Arg(sum exp(i eta_i))` and
`s_i=cos(eta_i-a)/|sum exp(i eta_i)|`. Each `s_i` is positive and
`sum s_i=1`, so

\[
\|Dc|_{\mathbf1^\perp}\|^2
 =\frac{\sum_i s_i^2-1/3}{\pi^2}\le\frac{2}{3\pi^2}.
\]

For the declared exact-real coefficients
`e=1/2`, `w=1/4`, `A=1/2`, `B=3/20`, take the natural monotonic-sine
pressure interval `|p_i|<=1/2`. Then one may use
`m=3/20`, `M=pi^2/2`, `H=sqrt(3)/2`, `C=sqrt(2/3)/pi`.
Consequently

\[
\gamma=\frac{\pi\sqrt2}{32},\qquad
\frac{\gamma^2}{4emk}=\frac{5\pi^3}{192}<\frac{64}{75}<1,
\]

where `pi<16/5` suffices for the strict rational bound. This proves strict
contrast dissipation on the stated class despite the potentially positive
mean-work term. It excludes a nonconstant periodic orbit wholly contained
in that class. It does **not** prove forward invariance, convergence of all
initial states, behavior beyond this pressure interval, binary64 runtime
stability or the independent-pressure extension's consistency.
The existing [local coupling estimator](../src/tnfr/dynamics/integrators.py)
depends only on degree, so it is common and constant on this fixed prism.
A changing coefficient would add `(k*g_dot/3)*V` to the displayed balance
and would require a new bound; it cannot silently inherit the result.

The [finite-amplitude orientation witness](nodal/PRIMITIVE_PHASE_CLOSURE.md#182-nonzero-oriented-area-production-from-initially-uniform-phase)
lies in this pressure class and creates oriented area while this storage
decreases. Angular generation and sustained identity are therefore distinct
even in one explicitly specified nonlinear comparison. The single G3 queue
must retain both tests. No phase formula, coefficient or pressure-refresh
policy is changed by this derivation.

Exact controls: [complete mean-work balance](../tests/physics/test_nonlinear_phase_mean_balance_scope.py).

### 13.16 Full-prism feedback: boundary exchange does not remove all dissipation

The repeated-triple restriction can be removed on the same unit prism.
Keep fixed unit capacities, fixed `e,w,g>0`, `k=w/pi`, and the **conditional
fresh-pressure comparison** of sections 13.14-13.15. In a regular phase chart,
write the actual six-node equations as

\[
p=-eLx+w g_\phi(\theta),\qquad \dot x=p,\qquad
v=\dot\theta=f(p)+gJ(\theta),\qquad
\dot p=-eLp+k(R-I)v.
\]

Here `L` is the unit prism's symmetric random-walk Laplacian, `R` is the
actual neighbor-phasor mean derivative, `R1=1`, and `J` is the mean-sine
current. This is not the optional runtime's independent pressure derivative.
No repeated phase, form, or triangle mean is assumed. Set
`q=mean p`, `h=p-q1`, `u=Pi v`, `Pi=I-11^T/6`. With `F'=f` as above and
`V=sum_edges(1-cos(theta_i-theta_j))`, each undirected edge counted once,
`grad V=-3J` and `sum J=0`. The full-graph comparison storage is

\[
\mathcal E_6=\sum_{i=1}^6 F(p_i)-6F(q)+\frac{kg}{3}V,
\]

with the exact balance

\[
\boxed{\quad \dot{\mathcal E}_6
=-e f(p)^TLp-k\|u\|^2
 +k\,[f(p)-f(q)\mathbf1]^T R u.\quad}
\]

The last term contains the full nonlinear neighbor response, including the
common pressure rate. To verify cancellation of the common phase velocity,
use `v=u+mean(f(p))*1`, `R1=1`, and `sum J=0`; do not set that velocity to zero.
On repeated triples the formula reduces to twice section 13.15's balance.
Its last term is not generally zero and cannot be discarded as internal
exchange. Thus the previous proof is reused through its storage and exact
chain rule, not extended by assuming the same reduced equations.

Suppose all six lifted phases have spread less than `pi/2`, and all
`|p_i|<=1/4`. The regular phasor geometry makes `R` nonnegative with row
sums one, giving `||R||_2<=sqrt(6)`. For `A=1/2`, `B=3/20`,

\[
m=B+A\pi/\sqrt2\le f'(p_i)\le M=B+A\pi.
\]

These bounds hold on the whole pressure interval, including its mean.
The graph's gap is `lambda_2=2/3`. Pairwise monotonicity and the gap give
`f(p)^T Lp>=m lambda_2 ||h||^2`; componentwise Lipschitz continuity gives
`||f(p)-f(q)1||<=M||h||`. Therefore

\[
\dot{\mathcal E}_6\le
-em\lambda_2\|h\|^2-k\|u\|^2+kM\sqrt6\|h\|\|u\|.
\]

For `e=1/2`, `w=1/4`, strict negativity away from `h=u=0` follows from
`6kM^2<4em lambda_2`. The constants need no numerical parameter scan:
`3<pi<16/5` and `sqrt2<3/2` imply

\[
6kM^2<\frac{49}{32}<\frac{23}{15}<4em\lambda_2,
\qquad 735<736.
\]

Storage is nonnegative, with `E_6>=m||h||^2/2+(kg/3)V`. In this strict phase
chart, `h=u=0` forces `J=0`, hence uniform phase by connected attractive
sine coupling. Then `g_phi=0`, `q=0`, and `Lp=0` with `p=-eLx` forces
uniform form. Consequently no nonconstant periodic full-state orbit, or
recurrent nonuniform quotient orbit, can lie wholly within this domain.
Inter-triangle exchange does not evade this sufficient dissipation test.

This six-node theorem uses a narrower pressure band than the repeated
theorem (`1/4` versus `1/2`) and controls additional directions. Neither
declared band is proved forward invariant. This is not global convergence,
a theorem for larger graphs or variable capacities, a binary64 solver result,
or admission of the comparison as a physically derived law.
Controls: [full nonlinear balance](../tests/physics/test_full_prism_phase_feedback_scope.py).

### 13.17 All spatial modes and local nonlinear attraction

Near uniform form and phase, the same conditional model has an exact local
classification on the **whole prism**. At uniform phase, both implemented
support derivatives share `L`: `Dg_phi=-L/pi`, `DJ=-L`. Write
`sigma=f'(0)=A*pi+B>0`. Every Laplacian mode `lambda` carries the block

\[
\frac{d}{dt}\begin{pmatrix}\delta x_\lambda\\\delta\theta_\lambda\end{pmatrix}
=\lambda\begin{pmatrix}-e&-k\\-\sigma e&-\sigma k-g\end{pmatrix}
\begin{pmatrix}\delta x_\lambda\\\delta\theta_\lambda\end{pmatrix}.
\]

The full spatial spectrum is `0,2/3,1,1,5/3,5/3`. For every `lambda>0`,
the temporal characteristic polynomial is

\[
r^2+\lambda(e+\sigma k+g)r+\lambda^2eg.
\]

Its discriminant divided by `lambda^2` is
`(e-g)^2+2*sigma*k*(e+g)+(sigma*k)^2>0`. Thus both roots are real and
strictly negative for **any positive** `e,k,sigma,g`. The inter-triangle mean
mode and the opposite internal modes add no growing or oscillatory linear
direction. The two neutral directions are the independent uniform shifts
of EPI and primitive phase; their neutrality is not a pulsation.

There is also a local nonlinear consequence, distinct from an instantaneous
linear diagnostic. The regular continuous comparison is smooth and invariant
under both uniform shifts. Its ten-dimensional centered quotient is therefore
autonomous, with a Hurwitz Jacobian at the origin. Local linearization stability
gives a sufficiently small exponentially attracting quotient neighborhood.
The absolute means can drift during that decay; their smooth rate functions
vanish with the contrast and are integrable, so both approach finite constants.
For example, along the nonrepeated small phase preparation
`theta=t*(0,0,0,1,1,-2)`, the mean phase pressure is
`t^3/(18*pi)+O(t^5)`. Mean stationarity cannot be imposed even this close
to consensus. Here `t` is a preparation amplitude, not trajectory time.

This proves existence of a local basin for the exact conditional ODE, not
that every state in section 13.16's displayed pressure/phase band belongs
to it. No basin radius, global attraction, recurrence outside these domains,
event law, or implemented-runtime stability is established. It closes the
small-amplitude, missing-spatial-mode loophole without a trajectory campaign.
Controls: [full local mode and mean tests](../tests/physics/test_full_prism_local_phase_stability.py).

### 13.18 Source sensitivity, joint work and the passive-transfer limit

This closes the source-asymmetry check before returning to the maintenance
mechanism. Fix the reciprocal unit prism, unit capacities, fixed `e,w>0`,
regular phase geometry and freshly constituted pressure
`p=-eLx+w*g(theta)`. Let `Pi` center the **complete** six-node field,
`y=Pi*x`, `mu=mean(x)`, `omega=theta_dot` and `J=Dg`. Then

\[
\dot\mu=w\,\overline g,\qquad
\frac{d}{dt}\frac{\|y\|^2}{2}
 =-e\,y^TLy+w\,y^Tg,
\qquad \dot p=-eLp+wJ\omega.
\]

The instantaneous contribution to form contrast is `w*y^T*g`. The matrix
`J` enters its change, through the still-unspecified `omega`; matrix cycle
imbalance is not an additional term in the nodal equation. In particular,

\[
\ddot\mu=w\,\overline{J\omega},\qquad
\frac{d^2}{dt^2}\|y\|^2
 =2\|\Pi p\|^2-2e\,y^TLp+2w\,y^TJ\omega.
\]

**Unchanged-preparation controls.** Keep the shared EPI
`x=(3/4,1/4,1/2)` repeated, `e=1/2`, `w=1/4`. At the retained phase
`theta=(-a,a,0,-a,a,a)`, `a=pi/6`, put
`gamma=atan(1/(3*sqrt(3)))`. The phase reflection has exactly the same `R`
and cycle ratio, but opposite `g`. Its contribution to `d||y||^2/dt` is
respectively `+1/12` and `-1/12`; total rates are `-1/6` and `-1/3`.
Both lose contrast. The mean rates are opposite as well:
`mu_dot=+/- (3*gamma-pi/6)/(24*pi)`. Detached production pressure readings
verify those exact-model expressions with their floating-point scope.

At the same unreflected state, the three *tangent controls*
`omega=0,+u,-u`, `u=(-1,1,0,0,0,0)`, give phase contributions to
`d^2||y||^2/dt^2` of zero and
`+/- (40-7*sqrt(3))/(112*pi)`, and mean accelerations zero and
`+/-1/(168*pi)`. These are local derivatives, not chosen trajectories.
Common phase rotation contributes zero. Thus neither the sign of work nor
its subsequent change is determined by the cycle defect alone.
Controls: [source work, sensitivity and mean](../tests/physics/test_phase_source_work_classification.py).

**A conditional storage criterion for maintenance.** For a fixed declared
comparison coefficient `beta>0`, with the units needed to compare the two
terms, define

\[
\mathcal E=\frac12\|y\|^2+\beta V_\phi\ge0,\qquad
\mathcal R=w\,y^Tg-\beta g^TH\omega.
\]

This is a mathematical storage and its signed transfer residual, not a new
TNFR parameter, physical energy law, external power meter or selector. No
`beta` is fitted or installed in the dynamics. The exact identity is

\[
\dot{\mathcal E}=-e\,y^TLy+\mathcal R.
\]

It retains both work exchanged with form and the change of phase storage.
In the class `R<=0`, phase/form transfer cannot maintain nonzero contrast
indefinitely. For example the *supplied comparison family*
`omega=(w/beta)*H^-1*y+lambda*g+c*1`, `lambda>=0`, gives
`R=-beta*lambda*g^THg<=0`. State-dependent skew exchange preserves power;
this does not prove a Poisson structure or derive that comparison law.
Common rotation cancels because `g^TH*1=-1^T gradient V_phi=0`.

More generally, let `B(T)` be an independently justified upper bound on
`integral_0^T R(t) dt` for the specified complete trajectory. Since this
prism has gap `lambda_2=2/3`,

\[
e\lambda_2\int_0^T\|y(t)\|^2dt
 \le \mathcal E(0)+B(T).
\]

If `||y||>=rho>0` throughout that interval, its duration obeys
`e*lambda_2*rho^2*T<=E(0)+B(T)`. With a uniform finite cumulative bound,
indefinite uniformly nonzero form and nonzero periodic centered form are
excluded while the hypotheses persist. Gauge motion at zero contrast is
not excluded. Integral control alone proves neither pointwise convergence
nor domain invariance; no binary64 or changing-support theorem is implied.

Conversely, sustained contrast with bounded storage requires the cumulative
residual to pay its integrated loss. Sufficient positive cumulative or average
residual is a necessary accounting condition, not pointwise positivity or
a generation law: held phase support can satisfy
it without explaining why the phases are held. Nor does this diagnostic
residual establish a literal physical external supply. A proposed complete
law must determine its value independently and retain any changing capacity,
support, boundary, memory or event contributions. Those terms cannot be
inserted retrospectively to make the balance close.
Controls: [passive joint storage and duration bounds](../tests/physics/test_phase_form_passive_transfer_budget.py).

The [relational passive-loss comparison](nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-passive-loss-completion)
reuses this work method with Dirichlet form storage and heterogeneous held
capacities. Its equilibrium, response and local-recovery statements belong to
that complete law; the centered-storage prism budget here is not transferred
by replacing symbols or setting a damping coefficient in the production engine.

The geometric asymmetry branch is therefore resolved in scope. Its metric
correction and actual work have separate owners, and no further matrix or
amplitude sweep is needed. The remaining task is the law supporting the
joint transfer, not another reinterpretation of the response matrix.

Validation: 418 tests across 32 relevant modules, including 14 new metric,
work, storage and configured-coordination controls, pass. The shared symbolic
fixture replaces duplicate retained preparation/response algebra in tests.
No production dynamics changes. Source-bound record:
`artifacts/research/phase_transfer_maintenance_validation_2026_09_19.json`.

### 13.19 Structural closure tests: exchange, Jacobi and the remaining potential

This tests whether stronger structural conditions determine the missing
phase row. Keep the same fixed unit prism, canonical regular phase source,
positive `e,w`, complete EPI field and declared comparison `beta` from
section 13.18. No new dynamical coefficient or trajectory is installed.
Write `a=H*g=-gradient V_phi`, `M=H^-1` and `y=Pi*x`.

**Power balance leaves four nongauge phase directions.** Prescribing a
storage residual `R_*` imposes only

\[
a^T\omega=(w y^Tg-R_*)/\beta.
\]

At `a!=0` the complete solution is

\[
\omega=\frac w\beta My-
 \frac{R_*}{\beta}\frac a{a^Ta}+v,\qquad a^Tv=0.
\]

There are five free instantaneous directions on six nodes. Because
`a^T 1=0`, common phase rotation is one of them; four remain after taking
that quotient. These are directions of the existing phase state, not four
new coordinates or attributes. At `a=0`, positive `H` implies `g=0`;
only `R_*=0` is compatible and the balance constrains no phase velocity.
Prescribing the residual is itself a premise, not a measured law inferred
from the desired outcome.

Locality and graph symmetries do not eliminate all this freedom. With
`W_U` the unit reciprocal support adjacency, form the derived matrix

\[
 B(a)=\operatorname{diag}(W_U(a\odot a))
       -\operatorname{diag}(a)W_U\operatorname{diag}(a),\qquad
 r=B(a)Lx.
\]

It has `B(a)a=0` and
`z^T B(a)z=sum_{edges ij}(a_j z_i-a_i z_j)^2>=0`.
Thus `a^T r=0`. This construction uses only the existing nodal fields and
support, respects relabeling, common phase rotation and common EPI shifts,
and has radius-two dependence in the primitive fields. It is a witness
against uniqueness under finite locality, not a new operator or an assertion
of strict one-hop dependence.

At the unchanged preparation `x=(3/4,1/4,1/2)` repeated and
`theta=(-pi/6,pi/6,0,-pi/6,pi/6,pi/6)`,

\[
\|\Pi r\|^2=\frac{131}{96}+\frac{283\sqrt3}{768}>0.
\]

The two comparison rows `omega_0=(w/beta)My` and
`omega_1=omega_0+(w/beta)r` have the same zero residual and the same
instantaneous full EPI rate. They have different subsequent responses:

\[
\Delta\ddot\mu=\frac{w^2}{\beta}
 \frac{155\sqrt3-267}{2688\pi},\qquad
\Delta\frac{d^2}{dt^2}\|y\|^2=\frac{w^2}{\beta}
 \frac{13\sqrt3-333}{448\pi}.
\]

The displayed zero-residual baseline uses the global mean through `y`;
only its null addition `r` has the asserted radius-two dependence.
There is also a fully local pair, `omega_bar_0=(w/beta)M Lx` and
`omega_bar_1=omega_bar_0+(w/beta)r`. They share the independently specified
residual function `R_*=w*g^T*(y-Lx)` and preserve the stated symmetries.
This residual is not generally zero. At the retained form `Lx=y`, they
coincide with the preceding pair, including the instantaneous response
differences. The residual is part of these mathematical countercontrols,
not a rule inferred retrospectively or installed in the engine.

Neither pair is selected as a TNFR law. The zero-residual family retains
the previous dissipation budget, so its freedom does not bypass that
maintenance obstruction; the local pair has its own nonzero-residual
budget away from the retained point. Controls:
[intrinsic exchange freedom](../tests/physics/test_phase_exchange_constitutive_freedom.py).

**A power-canceling skew tensor need not be Poisson.** The lossless comparison
above can be written, before adding `(-eLx,0)`, as

\[
\binom{\dot x}{\dot\theta}=
P(\theta)\nabla_{x,\theta}\mathcal E,\qquad
P=c\begin{pmatrix}0&-M\\M&0\end{pmatrix},\quad c=w/\beta.
\]

Since `gradient_x E=y`, this gives the **full** source row `xdot=w*g`
and `theta_dot=c*M*y`, including `mean(xdot)=w*mean(g)`. No centering
of the source row is implicit. Antisymmetry cancels exchange power, but
the Jacobi identity for this particular full tensor additionally requires

\[
\{x_i,\{x_j,\theta_k\}\}+\text{cyclic}
=c^2(\delta_{jk}m_i\partial_i m_j
       -\delta_{ik}m_j\partial_j m_i)=0,
\qquad m_i=1/H_i.
\]

Because every `m_i>0`, this is equivalent to
`partial_i m_j=0` for every `i!=j`. Together with the canonical metric's
common-phase-shift invariance, it would require a locally constant metric.
The retained nonrepeated preparation violates it: `H_0=H_1=3(1+sqrt(3))`,
`partial_1 H_0=3-9sqrt(3)/pi`, and the `(x_1,x_0,theta_0)` Jacobiator is

\[
\frac{c^2(3\sqrt3-\pi)}{9\pi(1+\sqrt3)^3}>0.
\]

This rejects promotion of that skew tensor to a Hamiltonian structure.
It does not reject other cross blocks, reduced brackets or the existing
closed wave pullback. Adding a dissipative term does not repair this
tensor's Jacobi identity. The substrate's constant bracket and supplied
tangent-map tests are different owners and cannot certify it.
Controls: [Jacobi condition and exact obstruction](../tests/physics/test_phase_exchange_jacobi_scope.py).

**Even a selected genuine symplectic structure leaves a potential to derive.**
Reuse the [existing local pressure coordinates](nodal/DERIVED_FORM_PHASE.md#214-local-wave-realizability-does-not-select-a-wave-law),
`(y,theta modulo rotation)->(y,v=Pi*p)`, on global phase spread below `pi/2`.
In orthonormal centered coordinates the selected canonical form is
`3 sum(dy_alpha wedge dv_alpha)`. For an autonomous Hamiltonian on this
ten-dimensional chart, requiring the nodal first row `ydot=v` forces,
on connected velocity fibers,

\[
\mathcal H(y,v)=\frac32\|v\|^2+U(y),\qquad
\dot v=-\frac13\nabla U(y).
\]

The function `U` remains undetermined. The existing wave chooses
`U_L=3*y^T L y/2`; the distinct `U_L2=3*y^T L^2 y/2` shares positivity,
graph relabeling covariance and common-offset invariance. Both have the
same genuine coordinate pullback and local nodal first-row lift. A separate
one-hop acceleration condition could exclude `L^2`, but is not implied by
symplecticity or a one-hop assumption about a different state variable.
At the original EPI contrast, an eigenvector of `L` with eigenvalue one,
these two accelerations coincide. Their derivative in the existing
inter-triangle mean direction differs because its eigenvalue is `2/3`.
This distinguishes full local laws without changing the retained preparation
or constructing a favorable new trajectory. The executable phase-lift
controls reuse the existing repeated phase comparator from section 21.4;
they do not claim to have executed a lift at the nonrepeated phase point
of the preceding two tests. The acceleration equality at `L y=y` is
independent of that choice of regular phase.

For every such comparison the complete mean still obeys
`mu_dot=w*mean(g)` and `mu_ddot=w*mean(Dg*omega)`; the centered Hamiltonian
does not justify setting either to zero. Common phase rotation is also
outside its ten-dimensional form. No hypothesis here establishes global
chart invariance, autonomous law selection, runtime realization or physical
validation. Controls:
[Hamiltonian family and retained-state derivatives](../tests/physics/test_phase_hamiltonian_closure_freedom.py).

The constitutive gap is now sharper: a specified storage balance, finite
locality and symmetries do not determine phase evolution; imposing a genuine
Hamiltonian structure in the known pressure chart still leaves `U` and the
choice of that structure as premises. Further calculation inside those
same assumptions cannot establish uniqueness. The single G3 plan must
identify an independent structural relation that constrains this remaining
freedom, or report it as an unresolved premise. It must not rename a chosen
potential, tensor or diagnostic as a derivation.

The integration also corrects the decoupled product observer's validation
order: all nodal scalar channels pass its existing finite-real contract
before substrate fields are extracted. Invalid phase/pressure booleans no
longer reach a different downstream exception after field/cache work.
Validated values retain the substrate's canonical node order. This repairs
the observer boundary, without changing its valid-input product dynamics.

Validation: 193 tests in 19 modules pass, including 12 new exact controls.
The common metric differential and centered pressure-coordinate fixture
replace duplicate test construction. Source-bound checkpoint:
`artifacts/research/joint_constitutive_conditions_validation_2026_09_19.json`.

### 13.20 Locality and exact diffusion compatibility

The next structural restrictions can be tested without a new trajectory.
Retain the same unit-conductance triangular prism, degree three, unit
capacity, `e,w>0` and complete nodal equation
`xdot=-e*L*x+w*g(theta)`. Here `L=L_rw`, `Pi=I-11^T/6`, `y=Pi*x` and
`v=Pi*xdot`. The selected centered symplectic chart from section 13.19
has `H=3||v||^2/2+U(y)`. These are conditional restrictions on that
comparison family; they do not establish its physical origin.

**One-hop acceleration and symmetry leave two edge coefficients.**
For a quadratic `U=y^T Q y/2`, require a constant symmetric Hessian,
`Q*1=0`, zero off-support entries and invariance under all twelve prism
automorphisms. The graph is vertex-transitive but has two edge orbits:
six triangle edges and three edges joining the triangles. Exactly

\[
Q=aB_{\triangle}+bB_{\mathrm{matching}},\qquad
\operatorname{spec}(Q)=\{0,2b,3a,3a,3a+2b,3a+2b\},
\]

where the `B` matrices are the combinatorial edge-orbit Laplacians and
`B_triangle+B_matching=3L`. The nine edge coefficients obey seven
independent symmetry constraints. Positivity on centered forms requires
and is implied by `a,b>0`; it does not fix their ratio. The original form
only probes the triangle coefficient. Its existing inter-triangle tangent
probes the other coefficient, without changing the preparation.

Requiring acceleration `-Q*y/3=-sigma*L*y` on every centered form forces
`a=b=sigma`. This *transport-matching premise* is stronger than graph
symmetry and leaves the scalar unspecified. Matching only the consensus
Hessian is weaker still: the unit edge quadratic and that quadratic plus
`sum_edges (x_i-x_j)^4/4` have the same Hessian there, all graph symmetries
and one-hop gradients, but different responses at the retained form.
The quartic is a dimensionless mathematical countercontrol; a physical
use would additionally need its scale and justification. Controls:
[local potential classification](../tests/physics/test_local_potential_stiffness_scope.py).

**Exact embedded diffusion differs from a leading overdamped balance.**
For the supplied auxiliary model

\[
\dot y=v,\qquad \dot v=-Ky-\gamma v,
\]

the complete diffusion graph `v=-eL*y` is invariant if and only if
`K*Pi=gamma*e*L-e^2*L^2`. With `K*1=0` this gives

\[
K=\gamma eL-e^2L^2,\qquad
s^2+\gamma s+\gamma e\lambda-e^2\lambda^2
=(s+e\lambda)(s+\gamma-e\lambda).
\]

Damping remains free and the initial velocity must lie on the specified
graph. Positive centered stiffness requires `gamma>e*lambda_max`, with
`lambda_max=5/3`; the diffusion root is the strictly slower root in every
nonuniform mode only when `gamma>2e*lambda_max`. The leading choice
`K=gamma*e*L` instead has exact tangency residual `-e^2*L^2*y`.
Discarding inertia in `tau*vdot+v=-eL*y` gives a leading balance, not an
exact finite-`tau` embedding or a proof of convergence of that limit.

On ordered nodes `((0,0),(0,1),(0,2),(1,0),(1,1),(1,2))`, nodes 0 and 4
are nonadjacent, yet `(L^2)[0,4]=2/9`. Thus the exact embedding requires
`K[0,4]=-2e^2/9`, independently of damping: strict one-hop acceleration
and this exact embedding cannot both hold on the retained support.
At `gamma=0` the embedding requires negative centered stiffness
`K=-e^2*L^2`, not a positive conservative wave. When `gamma>e*lambda_max`,
the positive centered energy satisfies

\[
\mathcal H=\frac32(\|v\|^2+y^TKy),\qquad
\dot{\mathcal H}=-3\gamma\|v\|^2.
\]

A periodic solution then has `v=0` throughout and `y=0`; this family
does not supply nonzero periodic maintenance. These centered auxiliary
identities are not primitive phase or full runtime certificates. Controls:
[exact diffusion embedding](../tests/physics/test_exact_diffusion_embedding_scope.py).

**Primitive phase locality is a separate restriction.**
At the existing repeated phase comparator
`theta=(0,pi/3,pi/6,0,pi/3,pi/6)`, let `J=Dg`, `B=pi*J` and
`T=(Pi*B+11^T/6)^(-1)*Pi`, the shared exact pressure-coordinate inverse.
Consider the selected Hamiltonian acceleration `vdot=-kappa*L*y`,
`kappa>=0`. Differentiating the nodal lift at fixed phase gives

\[
w\Pi J D_x\omega=-(\kappa L+e^2L^2),\qquad
D_x\omega=-\frac\pi w T(\kappa L+e^2L^2)+\mathbf1 c^T.
\]

The row difference eliminates every common-phase gauge derivative:

\[
\partial_{x_0}\omega_4-\partial_{x_0}\omega_5
=\frac\pi w\left[
 \frac{193-111\sqrt3}{1716}\kappa
+\frac{37-19\sqrt3}{396}e^2\right]>0.
\]

Both 4 and 5 are nonneighbors of 0. A continuously differentiable phase
law depending only on each node's and its neighbors' **primitive** EPI,
phase and fixed capacity would require both derivatives to vanish. Such
a law cannot realize this selected acceleration on an open EPI neighborhood
at this phase. Relabeling and choosing another common phase velocity do
not remove the obstruction. The complete form mean is retained: its
acceleration derivative in the existing repeated `Q_MODE=(1,1,-2)`
direction is `(kappa+e^2)*(5-3sqrt(3))/2`, not zero.

At phase consensus a sparse lift exists, so this is a state-dependent
obstruction rather than an artifact of centering. It excludes the specified
`kappa*L` family under this strict locality requirement, not all local
stiffnesses or every TNFR law. Access to freshly computed neighboring
pressures can already have radius-two primitive dependence; it is a
different restriction. Neither the nodal equation nor the current global
coordinator establishes strict primitive one-hop phase locality. Controls:
[primitive phase locality and mean](../tests/physics/test_primitive_phase_locality_scope.py).

The same study repairs a separate numerical diagnostic error: subtracting
near-equal terms lost the slow damped-wave root at large damping. All
graph-wave observers now share a root kernel that recovers the small root
from the root product. On P2 with `gamma=1e10`, the nonuniform rate is
approximately `-2e-10`, rather than zero. Decimal comparison, zero/critical/
complex-root controls and existing coordinate regressions cover this fix.
It does not change native evolution, prove a wave limit, remove all
floating-range limitations or repair the distinct critical-trajectory
fallback. Controls:
[root precision](../tests/physics/test_damped_wave_root_precision.py).

These restrictions do not identify a unique potential or phase law.
Locality must name its primitive variables, and diffusion compatibility
must distinguish a limiting approximation from an exact invariant graph.
The single G3 plan retains the remaining independent closure premise;
no coefficient, clock or maintaining trajectory is selected here.

Validation: 144 tests in 11 modules pass, including 12 new exact mathematical
controls and five numerical regression cases. Shared owners supply the
support, symmetries and pressure-coordinate inverse. Earlier research
artifacts are preserved byte-for-byte. Source-bound checkpoint:
`artifacts/research/locality_transport_closure_validation_2026_09_19.json`.

### 13.21 Cotangent phase exchange: a conditional constitutive candidate

<a id="cotangent-phase-exchange"></a>

The failed Jacobi tensor in section 13.19 can be repaired by an actual
coordinate construction. This supplies a **new conditional model**, not a
derivation of the engine's unchanged pressure. Work in declared nondimensional
coordinates, on finite simple loopless fixed regular reciprocal support with
unit conductances and unique-neighbor degree `d>0`,
with held unit capacity, positive phase weight `w`, and EPI weight `e>=0`.
Let `H=H_phi(theta)>0`, `M=H^-1` and `g=-M*gradient V_phi` have exactly their
section 13.6 meanings on the regular phase chart.

The independent premise is that

\[
\pi_i=H_i(\theta)x_i
\]

is the canonical momentum conjugate to `theta_i`, with Hamiltonian
`E=||x||^2/2+w*V_phi`. This chooses a symplectic structure, an uncentered
form storage and a time normalization. The existing pressure metric alone
does not force that identification or its units. Momentum adds no primitive
state on this chart: `x_i=pi_i/H_i` is invertible. It is distinct from the
centered `(y,v)` wave construction in sections 13.19-13.20.

**The correction is forced by this premise.** Pushing the canonical bracket
through `x=H^-1*pi`, in coordinate order `(x,theta)`, gives

\[
\mathcal J=\begin{pmatrix}K&-M\\M&0\end{pmatrix},\qquad
K_{ij}=\frac{x_j\partial_i H_j-x_i\partial_j H_i}{H_iH_j}.
\]

Here `partial_i` differentiates `theta_i`. This tensor satisfies Jacobi
because it is a genuine coordinate pullback; its additional `K=-K^T` block
is fixed by the chosen momentum map. Dropping it generally recovers the
non-Poisson tensor already rejected in section 13.19. The Hamiltonian rows,
with optional existing form diffusion added separately, are

\[
\boxed{\dot x=-eLx+wg+Kx,\qquad \dot\theta=Mx,\qquad\dot\nu=0.}
\]

Thus the proposed pressure is `p_new=-eLx+wg+Kx`; `p_old=-eLx+wg` remains
the configured two-channel engine pressure. A different phase row alone
cannot turn the latter into this model. The correction is quadratic in form
and cannot generally be absorbed into a phase-only potential `Psi`. On the
retained nonrepeated prism, taking `x=(1,0,0,0,0,0)` gives

\[
(Kx)_1=\frac{3-9\sqrt3/\pi}{9(1+\sqrt3)^2}<0.
\]

This is an exact, regular-state discriminator, not a numerical trajectory.
The phase row now has real form feedback, `D_x dot(theta)=M`; two states
with identical phase and different form have different phase rates. At P2
consensus, changing form from zero to `(1,-1)/4` changes phase velocity by
`(1,-1)/(4*pi)` and phase acceleration by `-e*(1,-1)/(2*pi)`.

`H_i` reads the node and its support neighbors; `Kx` generally reaches their
neighbors and therefore has radius-two primitive dependence. Absolute form
enters the phase row. A common EPI translation is not automatically a
redundancy of this joint model, and centering its storage would change it.
No operator selector, tetrad threshold, gain fitted to persistence, capacity
adaptation or external phase clock has been added.

**Complete storage and momentum balances.** Regularity of the support makes
`L` symmetric positive semidefinite. Antisymmetry of `K` and `Hg=-grad V_phi`
give the exact continuous balance

\[
\dot E=-e\,x^TLx\le0.
\]

The correction transfers form between components with `x^T Kx=0`; it does
not inject storage. The complete form mean nevertheless obeys
`mean(x)'=w*mean(g)+mean(Kx)`, which need not vanish. Common phase-rotation
invariance supplies a different conserved quantity for the reversible row,
the canonical momentum `P=sum_i H_i*x_i`. With diffusion included,

\[
\dot P=-e\sum_i H_i(Lx)_i.
\]

It need not remain conserved on a nonuniform phase state. Neither quantity
is the engine's coherence telemetry or an experimentally identified energy.

There is a useful forward-domain guarantee on any finite connected regular
support. If initially `E_0<w`, every edge obeys
`1-cos(gap)<=E_0/w<1` for all future times. Put
`a=acos(1-E_0/w)<pi/2`. All edge gaps and local mean displacements stay
within `[-a,a]`, all resultants have magnitude at least `d*cos(a)`, and
`H_i>=pi*d*cos(a)*sinc(a)>0`. Bounded form and the compact phase torus then
keep the smooth vector field away from its branch/resultant singularities,
giving global existence within this conditional model. The bound establishes
admission, not nontrivial persistence or full-state convergence.

**A geometry-dependent linear prediction.** At zero form and phase consensus,
`H=pi*d*I`, `DH=0` and `Dg=-L/pi`. A Laplacian mode with eigenvalue `ell>0`
has characteristic polynomial

\[
s^2+e\ell s+\frac{w\ell}{\pi^2d}=0.
\]

This is a damped oscillatory mode exactly when
`e^2*ell<4*w/(pi^2*d)`. With the same retained coefficients `e=w=1/2`,
unit cycles have `d=2`: the lowest C6 mode `ell=1/2` fails this test,
whereas the lowest C8 mode `ell=1-sqrt(2)/2` satisfies it. The elementary
bounds `9<pi^2<10` separate both cases. Geometry supplies this difference
without coefficient adjustment. This is a linear prediction of the new
conditional model, unrelated to the earlier C6 full-runtime stability
campaign. It does not certify a nonlinear waveform or maintained NFR.

**A nonlinear sign-reversal discriminator without fitting a trajectory.**
On that C8 comparison let `v_i=cos(2*pi*i/8)`, so `||v||^2=4`, and prepare
`x(0)=epsilon*v`, `theta(0)=0`. Observe
`A(t;epsilon)=v^T*x(t)/4`, not a refitted pattern amplitude. Write
`alpha=e*ell/2` and `omega=sqrt(w*ell/(pi^2*d)-alpha^2)>0`.
The linear initial-value problem predicts

\[
A_1(t)=e^{-\alpha t}\left(\cos(\omega t)
             -\frac{\alpha}{\omega}\sin(\omega t)\right),\qquad
A_1(T)=-\frac{\alpha}{\omega}e^{-\alpha T}<0,
\quad T=\frac{\pi}{2\omega}.
\]

This implies a genuine nonlinear existence result. The metric is even under
phase reflection, its derivative and `g` are odd, and
`K(-x,-theta)=K(x,theta)`. The full vector field is therefore odd under the
joint reflection and smooth near consensus. Smooth dependence on the signed
preparation amplitude gives, at fixed finite `T`,
`A(T;epsilon)=epsilon*A_1(T)+O(epsilon^3)`.
Consequently there exists `epsilon_0>0` such that every sufficiently small
positive preparation reverses this form mode's sign by `T`. Its initial
storage is `2*epsilon^2`; `abs(epsilon)<1/2` guarantees the regular domain
but is **not** a computed error bound or value of `epsilon_0`.

The old held-capacity, two-channel continuous comparison with its supplied
gated-sine phase row instead keeps phase consensus up to common rotation and
has `A_old(t)=epsilon*exp(-e*ell*t)>0`. This distinguishes the two complete
laws beyond merely renaming a phase variable. It does not compare the native
operator runtime or prove formation. This existence argument alone does not
select a finite amplitude. Section 13.22 supplies the quantitative nonlinear
remainder, separate ideal-integration budget and frozen executable comparison.

**A complete nonlinear P2 sector.** On P2 write `delta=theta_1-theta_0`
in `(-pi,pi)`. Then `H=h(delta)*I`, `h=pi*sin(delta)/delta`, with `h(0)=pi`.
The zero-momentum sector `x=(a,-a)` is invariant, `Kx=0` there, and

\[
\dot a=-2ea+w\delta/\pi,\qquad
\dot\delta=-2a/h(\delta),\qquad
E=a^2+w(1-\cos\delta),\qquad \dot E=-4ea^2.
\]

For `E_0<2w` its compact sublevel stays strictly inside the regular chart;
`E_0<w` additionally preserves strict U3. At `e=0`, every nonzero energy
level below `2w` is a regular closed curve with a nonvanishing vector field,
so it gives periodic reversible form/phase exchange. At `e>0`, the largest
invariant subset of `dot(E)=0` requires both `a=0` and `delta=0`; LaSalle's
argument proves convergence to that point in this sector. The common phase
is constant there. Positive diffusion therefore does not give indefinite
nontrivial maintenance. In particular `e=w=1/2` is overdamped near P2
consensus: the model is not made oscillatory by describing it as resonant.

The constructive result is a closed, genuinely Poisson exchange law with a
forced geometric pressure correction, an admitted global small-energy domain,
and a discriminating graph-dependent response. Its cotangent premise remains
open to independent justification or rejection. It is not installed in the
motor, physically validated, uniquely selected by NFR identity, or shown to
generate support/capacity. Exact pullback, Jacobi, production-source boundary,
balance and mode controls are in
[cotangent exchange scope](../tests/physics/test_cotangent_phase_exchange_scope.py).
The [single plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) owns
subsequent work; no trajectory was used to select this law.

### 13.22 A finite C8 prediction fixed before numerical evaluation

<a id="cotangent-c8-finite-response"></a>

Keep every constitutive choice of section 13.21: the unit C8, unit held
capacity, `e=w=1/2`, no forcing or operator events, and the cotangent pressure
including `Kx`. Fix `epsilon=1/1024`, `x_i(0)=epsilon*cos(2*pi*i/8)` and
`theta(0)=0`. The observable remains `A=v^T*x/4`. The amplitude, observable
and response time are chosen from the preceding analytic prediction, before
inspecting a trajectory. This comparison does not modify engine defaults.

**A quantitative continuous bound.** On this degree-two support, with
continuous phase lifts from the initial consensus, put

\[
a_i=(\theta_{i+1}+\theta_{i-1})/2-\theta_i=-(L\theta)_i,
\qquad b_i=(\theta_{i+1}-\theta_{i-1})/2.
\]

While edges are acute, the exact metric and pressure source simplify to
`H_i=2*pi*cos(b_i)*sinc(a_i)` and `g_i=a_i/pi`; the latter has no nonlinear
remainder. For any `0<epsilon<=1/16`, the previously proved energy balance
gives `||x||<=2*epsilon` and `V_phi<=4*epsilon^2`. Each edge separation is
less than `1/2`, since its potential cost is below `1-cos(1/2)`. The bound
`1-cos(delta)>=(47/96)*delta^2` on this interval then gives
`||D theta||^2<9*epsilon^2`, where `D` is the oriented cycle incidence map.
Consequently `abs(a_i),abs(b_i)<=3*epsilon` and

\[
H_i\ge2\pi(1-6\epsilon^2),\qquad \|M\|\le1/6.
\]

For `B_ij=partial_j log(H_i)`, the cycle difference operators have norms
`||D_theta a||<=2` and `||D_theta b||<=1`. Using
`abs(tan b)<=abs(b)/(1-b^2/2)` and
`abs(sinc'(a)/sinc(a))<=abs(a)/(3*(1-a^2/6))` therefore yields

\[
\|B\|\le\left(\frac3{1-9\epsilon^2/2}
                 +\frac2{1-3\epsilon^2/2}\right)\epsilon
             \le\frac{21}{4}\epsilon.
\]

In particular, the exact expression
`Kx=M*B^T*(x*x)-diag(x)*B*M*x`, with componentwise `x*x`, satisfies
`||Kx||<=7*epsilon^3`. Also
`||(M-I/(2*pi))*x||<=2*epsilon^3`.

Use the phase-rotation quotient and its energy norm

\[
N(x,\theta)^2=\|x\|^2+\theta^TL\theta.
\]

The linearized generator `mathcal A` is dissipative in this norm; its
semigroup is contractive. The complete nonlinear remainder `R` has
`N(R)<=sqrt(57)*epsilon^3<8*epsilon^3` along the exact trajectory, because
`||L||=2`. Duhamel's formula and `||v||/4=1/2` now give the finite bound

\[
\boxed{|A(t;\epsilon)-\epsilon A_1(t)|\le4t\epsilon^3.}
\]

For the fixed response time of section 13.21,
`T=pi/(2*omega)=34.630474785...`, the linear response is
`epsilon*A_1(T)=-1.2485970787...e-4`, whereas the continuous nonlinear
allowance is less than `1.291e-7`. Thus this specific finite preparation,
not merely an unspecified sufficiently small one, has `A(T)<0` in the
exact continuous model. The displayed decimals summarize the explicit
formulas; the sign has a wide margin under their rounding.

**A separate exact-arithmetic Euler bound.** Freeze `h=1/2048`,
`n=ceil(T/h)=70924` and `tau=n*h=34.630859375`, and evaluate the same
observable at `tau`. A coarse, independently checkable error allowance
suffices; no step-size search against an observed response is needed.

For this purpose put `q^2=theta^T*L*theta`. At any state with `N<=0.1`,
`abs(a_i),abs(b_i)<=q`, and `m=1-2*q^2/3>=149/150`. The identities above
give `H_i` in `[2*pi*m,2*pi]`,
`abs(H_a)<=2*pi*q/3`, `abs(H_b)<=2*pi*q`, and both the maximum row-sum
and column-sum norms of `D H` at most `10*pi*q/3`. Hence

\[
\begin{split}
\|Kx\|&\le\frac5{3\pi m^2}q\|x\|^2,\\
\|L^{1/2}(M-I/(2\pi))x\|
 &\le\frac{\sqrt2}{3\pi m}q^2\|x\|.
\end{split}
\]

The Euclidean norm of these two coefficients is below `0.6`, so the
convenient bound `N(R)<=N^3` holds throughout this domain. In energy
coordinates each linear modal block is
`[[-ell/2,-sqrt(ell)/(2*pi)],[sqrt(ell)/(2*pi),0]]`.
For `0<=ell<=2`, its squared operator norm is at most `c^2=10/9`.
Linear dissipativity therefore bounds one Euler step by
`||I+h*mathcal A||<=exp(5*h^2/9)`.

Let `B0=4*epsilon`. The bootstrap

\[
N(u_k)\le2\epsilon\exp[t_k(5h/9+B_0^2)]<2.021\epsilon<B_0
\qquad(0\le k\le n)
\]

holds inductively, because `tau<=35` and the exponential is below `2`.
Thus the numerical iterates also remain inside the domain used for the
remainder estimate. Separating full Euler from linear Euler, and linear
Euler from the exact linear flow, gives the observable bounds

\[
\begin{split}
|A_{\rm Euler}(\tau)-A_{{\rm linear\ Euler}}(\tau)|
 &\le\frac\tau2 e^{5h\tau/9}B_0^3,\\
|A_{{\rm linear\ Euler}}(\tau)-\epsilon A_1(\tau)|
 &\le\frac{\tau h}{2}\frac{10}{9}\epsilon e^{5h\tau/9}.
\end{split}
\]

Their combined allowance is less than `1.044e-5`, using only
`tau<=35` and `exp(5*h*tau/9)<1.01`. The declared linear target is
`epsilon*A_1(tau)=-1.2485754095...e-4`; even this coarse allowance leaves
the exact-arithmetic Euler response strictly negative. The old supplied
phase law instead predicts the positive
`epsilon*exp(-ell*tau/2)=6.1255393889...e-6`.

These are **two separate bounds**: the continuous nonlinear deviation and
the full Euler deviation from its linear target. Adding them is not needed
to establish either statement. The Euler bound concerns exact arithmetic;
it does not certify binary64 rounding or assign an unproved floating-point
error allowance. A subsequent executable comparison must retain the frozen
law, initial state, step, horizon and observable, and report floating-point
execution as finite numerical evidence. Neither sign reversal nor its
prediction establishes maintained identity, substrate generation or physical
identification. The cotangent constitutive premise remains conditional.

**Frozen finite execution, 2026-09-26.** The
[bounded instrument](../benchmarks/cotangent_phase_exchange.py) executed one
preparation with the amplitude and clock above. Every step reuses the shared
fused production pressure and `dynamics._euler_kernel.euler_update` for both
rows from the same preceding state; `Kx` is added explicitly. At steps
`0,17731,35462,53193,70924`, the ordinary graph pressure owner independently
checks the old-pressure component. Detached SDK diagnostics then observe the
candidate pressure and its declared model rate. This is a research execution
path, not the default phase coordinator or an operator study.

| Quantity | Frozen prediction or admission | Observed binary64 result |
| --- | --- | --- |
| Final mode amplitude | Linear target `-1.2485754095182383e-4`; negative | `-1.2485839574377552e-4` |
| Absolute deviation from linear target | At most `1.044e-5` | `8.5479195169e-10` |
| Quotient energy norm, all endpoints | At most `4*epsilon=0.00390625` | Maximum `0.001953125` |
| Old pressure residual | Each checkpoint and all-step formula residual at most `1e-12` | `0` and `5.4211e-20`, respectively |
| Instantaneous storage balance residual | At most `1e-15` | Maximum `3.1764e-22` |
| Geometric form-work residual | At most `1e-15` | Maximum `3.9444e-30` |

The old continuous triangular comparison predicts `+6.1255393889e-6` at the
same endpoint. No amplitude, coefficient, step, observation or acceptance cut
was changed after the reserved execution. Maximum adjacent separation was
`0.000521212` radians (rounded upward), with minimum metric `6.2831844537`.
The tetrad was available at all five checkpoints; `xi_C=1.8477590650` came
from the spectral fallback at every checkpoint, not an observed growing
correlation length. No field selected or corrected the dynamics.

The discrete energy fell overall from `1.9073486328e-6` to `7.4318907298e-8`,
but explicit Euler had a maximum positive single-step increment of
`5.6298e-16`. Its cumulative energy-balance defect reached `1.1397e-10`.
These observations must not be called exact discrete dissipation or a
validated floating-point enclosure. The analytic continuous and ideal Euler
claims above remain separate from this finite check.

Reproduce the fixed run with:

```sh
python benchmarks/cotangent_phase_exchange.py --amplitude 0.0009765625 --timestep 0.00048828125 --steps 70924 --checkpoint 17731 --checkpoint 35462 --checkpoint 53193 --output artifacts/research/cotangent_c8_execution_2026_09_26.json
```

The retained local records are
`artifacts/research/cotangent_c8_protocol_2026_09_26.json`,
`artifacts/research/cotangent_c8_execution_2026_09_26.json` and
`artifacts/research/cotangent_c8_verdict_2026_09_26.json`. The protocol SHA-256
is `a1ee858cece6371f177827658626adcea349aaeab5649ec44957ff0cbae14775`;
the execution SHA-256 is
`20d2aa43311dec4b13c49a41d3b7ef8ac9d56f3be0b0820d83470cf74c3e65f5`.
Execution used Python 3.13.6, NumPy 2.5.3, Windows 11, binary64, no randomness,
and base revision `f27413a0a29dddc132e5aafea081b0caee0b0135` with scoped
working-source digest
`sha256:af0c6fe2dd2428df0da278be4960aa90e271f375d0322538d40f341fd2b97940`.
The source scope is this instrument and `src/tnfr`; the report retains full
state at checkpoints and all-step extrema. The local artifacts are generated
evidence, not a substitute for the reproducible recipe and scope above.
[Small execution controls](../tests/physics/test_cotangent_phase_exchange_execution.py)
check independent derivatives, pressure, balances, simultaneous integration
and diagnostic provenance without regenerating the full trajectory.

**What this comparison cannot select.** Suppressing `Kx` while retaining
`theta_dot=Mx` has the same linearization at consensus; its storage balance
also holds because `x^T Kx=0`. The sign reversal therefore discriminates
reciprocal feedback from the old triangular law, not the cotangent premise
from every alternative reciprocal law. Moreover, this preparation is
antiperiodic under a four-node shift: `x_{i+4}=-x_i` and
`theta_{i+4}=-theta_i`. Since the paired metrics agree, its exact total
momentum is zero by symmetry. Near-zero observed momentum is not an
independent test of the general momentum balance. The existing nonrepeated
state and Jacobi witnesses in section 13.21 already distinguish the geometric
correction algebraically; no second C8 sweep is needed to repeat that result.

### 13.23 Nonzero momentum and relative identity on P2

<a id="cotangent-p2-nonzero-momentum"></a>

The same cotangent law admits an exact reduction beyond the zero-momentum
sector in section 13.21. This changes the prepared state, not the closure.
On P2, take the regular chart `delta=theta_1-theta_0` in `(-pi,pi)`, and set

\[
h(\delta)=\pi\operatorname{sinc}\delta,\quad
\psi=(\theta_0+\theta_1)/2,\quad a=(x_0-x_1)/2,\quad
P=h(\delta)(x_0+x_1).
\]

Here `H=h*I`, so the general momentum budget reduces to `P_dot=0`, even
when `e>0`: `h*sum(Lx)=0`. This P2 simplification is not momentum conservation
on every graph. Substitution into the complete law gives

\[
\begin{gathered}
x_{0,1}=\frac{P}{2h}\mathbin{\pm}a,\qquad
U_P(\delta)=\frac{P^2}{4h(\delta)^2}+w(1-\cos\delta),\\
\boxed{\dot\delta=-\frac{2a}{h},\qquad
\dot a=-2ea+\frac{U_P'(\delta)}h,\qquad
\dot\psi=\frac{P}{2h^2}.}
\end{gathered}
\]

In particular `U_P'/h=w*delta/pi-P^2*h'/(2*h^4)`. The momentum-dependent
term comes from `Kx`; it is not a new potential inserted to maintain a
pattern. The reduced storage and balance are exactly
`E=a^2+U_P(delta)` and `E_dot=-4e*a^2`.

**Regularity and the possible identity.** On `0<delta<pi`, `h>0` and
`h'<0`: `sin(delta)-delta*cos(delta)` has derivative `delta*sin(delta)>0`
and vanishes at zero. Thus `U_P` increases strictly with `abs(delta)`.
For `P!=0` it diverges at the antipodal boundary. Every finite-energy
sublevel consequently stays compactly inside the regular chart, without
requiring the earlier sufficient bound `E_0<w`. Strict acute admission
additionally follows from `E_0<U_P(pi/2)=P^2/16+w`; regularity alone must
not be reported as U3 admission.

For `e>0`, the only invariant part of `E_dot=0` in a fixed-`P` sublevel
requires `a=0` and `U_P'=0`, hence `delta=0`. The reduced LaSalle argument
therefore gives

\[
\delta,a\longrightarrow0,\qquad
x_0,x_1\longrightarrow\frac{P}{2\pi},\qquad
\dot\psi\longrightarrow\frac{P}{2\pi^2}.
\]

The resulting family is a uniform rotating **relative equilibrium**: its
relative state is fixed while its common phase advances. It is labelled by
prepared momentum. It does not retain a nonzero internal form or phase
contrast, produce momentum from nothing, or identify a physical particle.
At `e=0`, nonminimal fixed-energy curves give periodic relative exchange;
the accumulated common phase need not make the full state periodic.

**Geometric contribution and a local discriminator.** Since
`h=pi*(1-delta^2/6+O(delta^4))`, the characteristic polynomial of relative
perturbations about this rotating equilibrium is

\[
\lambda^2+2e\lambda+\frac{2w}{\pi^2}
                         +\frac{P^2}{3\pi^4}=0.
\]

Thus prepared momentum changes relative restoring stiffness with no change
of coefficients. Omitting `Kx` while keeping `theta_dot=Mx` preserves the
same storage balance but generally gives
`P_dot=h'*(x_1^2-x_0^2)/h`, rather than zero. The zero-momentum preparation
hides that defect. Storage dissipation, a plausible oscillation and a
genuine cotangent structure are therefore distinct obligations. The
[exact scope controls](../tests/physics/test_cotangent_phase_exchange_scope.py)
derive the reduced rows and momentum boundary from the full pullback law.
On larger supports, use the actual momentum balance and test relative
identity and its transverse stability; this P2 result does not supply them.

### 13.24 Integration audit: inherited structure and missing selection

<a id="cotangent-integration-audit"></a>

Sections 13.21-13.23 establish consequences of a conditional completion.
Before another pattern study, distinguish its integration with the nodal
framework from selection of a fundamental law. No new production default,
controller, fitted parameter or physical identification follows here.

| Ingredient | Status in this comparison |
| --- | --- |
| Signed scalar form, circular phase, support and nodal product | Existing state/flow contracts; the comparison restricts capacity to one |
| `g=-H^-1 grad(V_phi)` | Derived from the configured phasor pressure and declared cosine cost; neither its absolute cost scale nor a phase clock follows |
| Canonical momentum `H*x`, uncentered quadratic form storage | Additional constitutive premises, not consequences of the NFR definition |
| `Kx`, reciprocal phase row, Jacobi and balance | Forced after those premises; independently checked and conditionally predictive |
| Support, initialization and held capacity | Supplied; no birth, adaptation or autonomous partition-selection law is derived |
| Tetrad and grammar | Existing observers and event-admission contracts; neither selects this continuous law |

**The existing principles admit more than one law.** The cost normalization
freedom already identified in section 13.6 becomes dynamically consequential
after the momentum premise. For any constant `eta>0`, take

\[
\pi_\eta=\eta Hx,\qquad
E_\eta=\tfrac12\|x\|^2+\eta wV_\phi,\qquad
\mathcal J_\eta=\mathcal J/\eta.
\]

The genuine coordinate pullback gives

\[
\dot x=-eLx+wg+\eta^{-1}Kx,\qquad
\dot\theta=\eta^{-1}Mx,\qquad
\dot E_\eta=-e\,x^TLx.
\]

The configured form/phase pressure part, circular symmetry, finite primitive
locality, genuine Poisson structure and storage dissipation all survive.
Nevertheless the consensus-mode polynomial becomes
`s^2+e*ell*s+w*ell/(eta*pi^2*d)`. At fixed nonzero `e,w`, state chart and
clock, changing `eta` preserves the trace while changing the determinant;
it is not a common rescaling of time. The earlier C6/C8 prediction remains
valid for its declared `eta=1`, not a parameter-free consequence of the
phase-pressure identity.

There is an exact finite-state discriminator without another trajectory.
On P2 with `x=(a,-a)` and phase consensus, every member gives
`a_dot(0)=-2e*a`, but

\[
\ddot a(0)=4e^2a-\frac{2wa}{\eta\pi^2}.
\]

This is exact because the zero-momentum P2 sector has `Kx=0`, not merely
a linearized formula. The family is an underdetermination witness, not a
new gain to tune until a pattern persists. It establishes that the listed
conditions do not select `eta=1`; it does not establish that no stronger
structural principle can do so.

The difference is not removable by a **state-dependent** common clock either.
At the same consensus preparation with `a,e>0`, the joint contrast tangent is
`F_eta=(-2*e*a,-2*a/(eta*pi))` in coordinates `(a,delta)`. Consequently

\[
\frac{d\delta}{da}=\frac1{\eta\pi e},\qquad
\det[F_1,F_\eta]=\frac{4ea^2}{\pi}(\eta^{-1}-1).
\]

A positive clock change divides both components by the same local value;
it cannot change this slope. Only `eta=1` aligns the two tangents in the
fixed state chart. This is an exact orbit discriminator, separate from a
comparison of periods or a fitted time unit. The
[clock control](../tests/physics/test_structural_clock_scope.py) derives it
from the full nonlinear P2 field without another trajectory.

Genuine unit/chart conversion remains distinct: for `x_new=A*x` and
`t_new=B*t`, with positive constants, the same family has
`e_new=e/B`, `w_new=A*w/B`, `eta_new=A*B*eta`. Thus `eta*e^2/w` is invariant
under these conversions. Changing only `eta` changes that invariant.
Likewise a fixed-formula EPI shift adds `beta/(eta*H_i)` to the phase rate;
when `H` is nonuniform, relative phase rates change. The same physical law
can be pushed into a shifted chart, but its storage and momentum origin
must transform too. Pure-EPI offset covariance cannot silently be imposed
on this joint law while retaining its unshifted formulas.

**An exact hierarchy result supplies part of the scale information.** Reuse
the [counted-support reduction](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md#joint-constitutive-reduction-with-inherited-support-counts).
Start with a regular simple base graph. Replace every node by `m` replicas,
place no edges within a block, and replace every base edge by all `m^2`
cross-block edges with unit conductance. Keep fine capacity one, `eta=1`,
and identical form/phase within each block. Within-block permutation symmetry
makes this manifold invariant under the complete cotangent vector field.
This is an imposed exact preparation, not formation of coherent blocks.

On that manifold, with base variables `X,Theta`,

\[
H_{\rm fine}=mH,\qquad g_{\rm fine}=g,\qquad
L_{{\rm rw,fine}}X=L_{{\rm rw,base}}X,\qquad
(K_{\rm fine}x_{\rm fine})_{i\alpha}=(KX)_i/m.
\]

For distinct blocks, each fine `K` entry is the base entry divided by `m^2`;
summing its `m` equal neighbors gives the last identity. Off-diagonal entries
inside a block vanish. Hence the **exact inherited** law is

\[
\dot X=-eLX+wg+KX/m,\qquad \dot\Theta=MX/m.
\]

Its effective exchange scale is `eta_eff=m`, although every fine capacity
is still one and the old pressure is unchanged. There is no fitted rate or
new primitive state. Fine storage divided by `m` equals
`||X||^2/2+m*w*V_phi`; the pulled-back canonical one-form divided by `m` is
`m*sum_i(H_i*X_i*dTheta_i)`, reproducing the same effective momentum.
For a common fine scale `eta`, the inherited scale is `m*eta`; repeated
equal-replica reductions multiply the counts consistently.

This is a constructive link to nested effective NFR descriptions and a
counterexample to discarding multiplicity and resetting `eta=1` at every
scale. It does not select the microscopic exchange premise, make the chosen
partition emerge, or establish transverse stability of its synchronized
blocks. Outside this invariant manifold, existing closure and memory tools
must retain hidden state or bound its effect; they cannot derive an unknown
fine law from a chosen projection.

If one additionally requires a degree-only scale `eta(d)` to retain the
same bare-graph recipe under every such reduction, with unchanged form,
clock and capacity, consistency imposes `eta(d)=m*eta(m*d)`. For all
positive integer degrees this is equivalent to `eta(d)=C/d`: use `d=1`
to obtain the necessary formula, which also satisfies every replication
identity. Global `C>0` remains free. The extra requirement treats this
particular replication as constitutive equivalence; physical replication
is not automatically an observational redundancy. It would change the
previous cross-degree `eta=1` predictions, and is not adopted here as a
default or a chosen sustaining mechanism.

**Capacity and support cannot be restored by substitution.** Common
prescribed capacity `kappa(t)>=0` can multiply both rows, giving the same
orbits under `s(t)=integral(kappa dt)` and storage rate
`-kappa*e*x^T*L*x`. Infinite-structural-time results require `s(t)` to diverge;
finite accumulated activity can stop earlier. Pausing both rows at zero is
a chosen joint law: the nodal identity alone pauses only form. Multiplying
by a positive state-dependent clock still reparameterizes orbits, but in
general does not preserve the Poisson identity in dimension at least four.
Time reparameterization and a Hamiltonian structure are separate claims.

For held heterogeneous `N=diag(nu_i)>0`, naively multiplying both rows by
`N` gives a form bracket `NK`, which is not generally skew. A genuine
pullback through `pi_i=H_i*x_i/nu_i` instead requires

\[
(K_\nu)_{ij}
=\frac{\nu_i x_j\partial_iH_j-\nu_jx_i\partial_jH_i}{H_iH_j}.
\]

Even this corrected pullback, with the same storage and diffusion `-eNLx`,
has `E_dot=-e*x^T*N*L*x`, not an established nonpositive expression.
At phase consensus on P2, `N=diag(1,4)` and `x=(2,1)` give `x^T*N*L*x=-2`.
For `e=1/2` the storage grows at rate `1`, despite valid stable diffusion
in its own capacity-weighted metric. This refutes a proposed extension of
the storage theorem, not the existing diffusion theorem.

The map `H/nu` also excludes zero capacity. On P2 at `delta=pi/3`,
`x=(1,1)` and `nu_1=1`, its form rate as `nu_0` tends to zero is
`2*(3*sqrt(3)-pi)/(9*pi)>0` at node zero. It therefore cannot extend
continuously as a finite-pressure unforced nodal row there. Changing to
inverse-capacity storage or another pressure can repair selected balances,
but defines another model and does not resolve that boundary automatically.
Evolving capacity additionally introduces moving-coordinate/storage terms.

There is an independent graph-domain obstruction. On unit P3 with phase
consensus and `x=(1,3/2,1)`, `x^T*L_rw*x=-1/4`; the unchanged Euclidean
storage would grow at rate `e/4`. A degree-weighted Hamiltonian
`E_D=x^T*D*x/2+w*V_phi` repairs diffusion on this reciprocal graph, but
with momentum `Hx` its Hamiltonian rows are `K*D*x+wg` and `M*D*x`.
Both change. Naming a weighted energy while retaining the old joint rows
would not repair the proof. The full pressure also has configured capacity
and topology source channels; none is included by these two-channel results.

**Engine, events and formation.** The
[default runtime](../src/tnfr/dynamics/runtime.py) interleaves pressure,
glyphs, held-pressure form integration, clamps, phase coordination and
capacity adaptation. Disabling glyphs or supplying a pressure callback does
not implement the joint cotangent flow. The bounded producer correctly reuses
pressure, Euler arithmetic and detached SDK diagnostics through a separate
declared path. A future opt-in joint model must preserve that ownership and
admit its scalar, support, capacity, clock and phase-chart domains explicitly.

Operator/grammar admission does not imply preservation of this storage,
momentum or regular chart. Support changes also change `H` and `V_phi`;
THOL birth changes the state dimension. Their event balances and occurrence
laws remain independent obligations. On fixed cycles in the strict acute
domain, continuous phase motion preserves integer winding. Thus the
small-energy theorem cannot also establish creation of nonzero winding
from zero within that domain. This does not exclude non-topological patterns
or finite lifetimes. Neither indefinite persistence nor a Hamiltonian form
is required by the definition of every NFR.

The audit verdict is **conditional model with a derived hierarchy law and
explicit integration gaps**, not a complete emergent theory and not a failed
nodal identity. Known principles permit several completions; no reviewed
law yet derives formation, identity, interaction, capacity and support
together from those principles. The
[single plan](research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate) places
constitutive/scale admission before further C8 identity experiments. The
[scope tests](../tests/physics/test_cotangent_phase_exchange_scope.py) retain
the exact alternative-family, inherited-scale and production-pressure
counterexamples without another trajectory or a public tuning parameter.

<a id="cotangent-unequal-block-closure"></a>
### 13.25 Unequal coherent blocks inherit a joint law, not a bare graph

Keep the section 13.21 fine law with `eta=1`, unit capacity, fixed unit C4
support and no inputs or events. Prepare blocks `A={0}`, `B={1,3}`, `C={2}`
with equal form and phase inside each block. The automorphism exchanging
nodes 1 and 3 leaves the full vector field equivariant on its regular chart.
Its fixed-point manifold is therefore invariant for as long as the solution
stays in that chart. This is an exact preparation, not attraction or formation.

The existing [counted-support owner](../src/tnfr/physics/quotient_structure.py)
retains the neighbor counts and block mass

\[
N=\begin{pmatrix}0&2&0\\1&0&1\\0&2&0\end{pmatrix},\qquad
S=\operatorname{diag}(1,2,1),\qquad \bar L=I-N/2.
\]

Choose a consistent real lift with acute edge differences, and put
`u=Theta_B-Theta_A`, `v=Theta_B-Theta_C`, `a=-(u+v)/2`, `b=(u-v)/2`.
In particular, `abs(u),abs(v)<pi/2`; the midpoint formulas below are not
asserted across a resultant branch change.
The inherited per-node phase metric, pressure source and phase cost are

\[
H=\operatorname{diag}(2\pi\operatorname{sinc}u,
  2\pi\cos b\operatorname{sinc}a,2\pi\operatorname{sinc}v),\qquad
g=\pi^{-1}(u,a,v)^T,
\]
\[
\bar V=2(1-\cos u)+2(1-\cos v),\qquad SHg=-\nabla_\Theta\bar V.
\]

Here `sinc(0)=1`. The factors of two count actual fine edges and neighbors;
they are not fitted coefficients. Restricting the canonical one-form and
storage gives

\[
\bar\pi=SHX,\qquad
\bar E=\tfrac12X^TSX+w\bar V.
\]

For block sizes `n=(1,2,1)` and diagonal entries `h_i` of `H`, the pullback
connection is

\[
\bar K_{ij}=
\frac{n_jX_j\partial_i h_j-n_iX_i\partial_jh_i}
     {n_in_jh_ih_j}.
\]

It is skew, but acts on `grad_X E=S*X`. The exact restricted rows are

\[
\boxed{\dot X=-e\bar L X+wg+\bar K SX,\qquad
\dot\Theta=H^{-1}X.}
\]

They follow either by restricting the fine field or by using the pulled-back
canonical coordinates; no new primitive capacity is introduced. The balances
are

\[
\dot{\bar E}=-eX^TS\bar LX
  =-e[(X_A-X_B)^2+(X_C-X_B)^2],\qquad
\dot P=-e\,\mathbf1^TSH\bar LX,
\quad P=\mathbf1^TSHX.
\]

The first uses `S*Kbar*S` skew and the phase-gradient identity. The second
uses common phase-rotation symmetry for the Hamiltonian part and retains
the diffusion contribution. It does not assert conserved `P` in general.

Recomputing a bare unit P3 keeps `Lbar` and `g`, but gives metric
`diag(1/2,1,1/2)*H`. With the bare unweighted storage, the endpoint phase
rates double and the geometric correction changes. No common exchange
factor repairs this: nonzero endpoint forms require `eta=2`, whereas a
nonzero center form requires `eta=1`. Thus preserving old pressure is
insufficient to preserve the joint law. Inherited mass and multiplicity
carry information needed for composition of effective regions.

The [exact controls](../tests/physics/test_cotangent_quotient_scope.py) reuse
production joint-quotient metadata, independently differentiate the macro
trigonometric expressions, and compare fine and restricted fields at
`Theta=(0,pi/6,pi/2)` for symbolic arbitrary `X`. There,
`H=diag(6,6*(sqrt(3)-1),3*sqrt(3))`. These finite controls accompany the
invariant-manifold argument; they do not test every phase preparation or
establish an all-state quotient. Hidden differences inside block B,
changing support, evolving capacity and the microscopic momentum premise
remain outside this result.

<a id="cotangent-tetrad-future-witness"></a>
### 13.26 Identical snapshot fields can have different futures under one law

Keep the same P2 cotangent law, unit capacity and clock, with `e=w=1/2`.
Compare `x=(a,-a)` and `x=(m+a,m-a)` at phase consensus. Both have
`Kx=0`, initial pressure `(-a,a)` and the same phase fields. Their stored
nodal rates and coherence agree. This comparison fixes the law; it does
not exploit uncertainty about which phase equation to choose.

The full P2 reduction in section 13.23 gives, at that preparation,

\[
\dot a=-2ea,\qquad
\ddot a=4e^2a-\frac{2wa}{\pi^2}-\frac{4m^2a}{3\pi^2}.
\]

Indeed `P=2*pi*m` there, and differentiating `U_P` gives
`U_P''(0)=w+P^2/(6*pi^2)`. The resulting acceleration difference is in
relative form, so it cannot be removed by common phase rotation. The
uncentered form/storage premise makes the hidden mean dynamically relevant.

For `a=1/8`, `m=0` versus `m=1/4`, the actual shared SDK diagnostic reports
identical tetrads: `Phi_s=(1/8,-1/8)`, zero phase gradient and curvature,
and `xi_C=1/sqrt(2)` with the same spectral-fallback provenance. Yet

\[
\Delta\ddot a=-\frac1{96\pi^2},\qquad
\Delta\dot\Phi_s=
\frac1{\pi^2}\left(\frac1{64},-\frac1{192}\right).
\]

For the last identity, unit-length P2 has `Phi_s=(p_1,p_0)` and the
declared unit-capacity field has `p=xdot`. Pressure is independently given
by the law, not reconstructed from an observed derivative. Its derivatives
are computed from that law's full field before imposing consensus.

Thus the snapshot tetrad does not close even its own evolution in this
conditional model. This does not exclude observability from a time series
or require a new fundamental variable: the omitted EPI mean was already
part of the full state. The
[single witness](../tests/physics/test_cotangent_observability_scope.py)
checks both the exact field derivative and the actual SDK readout. At this
initial state the default two-channel pressure agrees because `Kx=0`;
the subsequent law is the declared cotangent comparison, not default runtime.
No trajectory, synthetic measurement history or physical identification is
claimed.

**A passive chart translation does preserve the law.** This state comparison
does not privilege a physical absolute EPI origin. For `y=x+beta*1`, with
fixed coordinate offset beta, transform momentum and storage to

\[
\widetilde\pi=H(y-\beta\mathbf1),\qquad
\widetilde E=\tfrac12\|y-\beta\mathbf1\|^2+wV_\phi.
\]

Writing `K_theta(x)` to expose the connection's form dependence, the same law
in the new chart is

\[
\dot y=-eLy+wg+K_\theta(y-\beta\mathbf1)(y-\beta\mathbf1),\qquad
\dot\theta=M(y-\beta\mathbf1).
\]

This uses `L*1=0`, not a new force. Resetting momentum to `Hy` and uncentered
storage to `||y||^2/2` is a different premise. A positive control in the same
test owner independently pushes the canonical bracket through `y=pi/h+beta`
on regular P2, retaining both absolute phase rows, momentum and the storage
balance. It complements the state-shift counterexample; it does not select
the origin or microscopic law through mathematical consistency alone.

### 13.27 Storage admission beyond the cotangent choice

The [relational exchange owner](nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
compares the existing Dirichlet form storage with the uncentered cotangent
storage. A declared nodewise work-cancellation premise supplies a different
joint phase/form law while keeping the native two-channel pressure intact.
It retains origin covariance, heterogeneous held capacity and inherited
coherent replication; its modal restoring coefficient scales as the square
of the diffusion eigenvalue, rather than the cotangent first power.
The same owner gives a P2 Noether obstruction independent of the chosen
smooth Hamiltonian storage: conserving both the native form mean and the
cotangent momentum precludes relative phase motion on a full open state
domain. These are conditional admission results, not a unique law selected
by the nodal equation. Proofs and controls remain in that single owner.

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [27_variational_principle_demo.py](../examples/02_physics_regimes/27_variational_principle_demo.py) | Finite field-action readouts, auxiliary coordinate pairs, harmonic momentum residual and grammar-labelled heuristics; snapshot-only symplecticity is inconclusive |

### Key Source Modules

- [variational.py](../src/tnfr/physics/variational.py) — field-action diagnostics and supplied harmonic-coordinate checks
- [conservation.py](../src/tnfr/physics/conservation.py) — field-energy diagnostic (a Lyapunov candidate, not a general certificate)
- [forced_support.py](../src/tnfr/physics/forced_support.py) and [phase_response.py](../src/tnfr/physics/phase_response.py) — source compatibility and phase derivatives consumed by section 13
- [classical_mechanics.py](../src/tnfr/physics/classical_mechanics.py) — auxiliary mechanical readouts with their own scope

---

## References

- Nodal equation and model premises: [Fundamental theory](FUNDAMENTAL_THEORY.md), [parameter foundations](NODAL_PARAMETER_FOUNDATIONS.md)
- Conservation theorem: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Classical mechanics analogy: `src/tnfr/physics/classical_mechanics.py`
- Unified fields (canonical source): `src/tnfr/physics/unified.py`
- Energy functional: `src/tnfr/physics/conservation.py`
