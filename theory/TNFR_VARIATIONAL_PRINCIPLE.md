# TNFR Variational Principle — Lagrangian Action Formulation

**Status**: Exact restricted EPI Dirichlet balance and decoupled metriplectic product; full coupled tetrad/nodal bridge unresolved
**Module**: `src/tnfr/physics/variational.py`
**Tests**: `tests/physics/test_diffusion_energy_balance.py`, `tests/physics/test_variational.py`, `tests/physics/test_symplectic_substrate.py`, `tests/physics/test_metriplectic_product.py`
**Date**: 2026-09-05

---

## 1. Main Result

The TNFR nodal equation

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

is the engine evolution law. This note specifies the implemented field-based energy and action readouts and the associated harmonic substrate model. A derivation of the full nodal law from the particular potential below is **not established**. The exact algebraic identities of the readouts must be distinguished from that unresolved dynamical correspondence.

---

## 2. The TNFR Lagrangian

### 2.1 Lagrangian Density

On a graph $G=(V,E)$ the **TNFR Lagrangian density** at node $i$ is

$$\mathcal{L}(i) \;=\; T(i) \;-\; V(i)$$

where the **transport (kinetic) energy** and **configuration (potential) energy** are

$$T(i) = \tfrac{1}{2}\bigl[J_\varphi(i)^{\,2} + J_{\Delta\mathrm{NFR}}(i)^{\,2}\bigr]$$

$$V(i) = \tfrac{1}{2}\bigl[\Phi_s(i)^{\,2} + |\nabla\varphi|(i)^{\,2} + K_\varphi(i)^{\,2}\bigr]$$

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

| Sector | Coordinate $q$ | Momentum $p$ | Continuity |
|--------|----------------|---------------|------------|
| **Geometric** | $K_\varphi$ (curvature) | $J_\varphi$ (phase current) | $\partial K_\varphi/\partial t + \mathrm{div}(J_\varphi) \approx 0$ |
| **Potential** | $\Phi_s$ (potential) | $J_{\Delta\mathrm{NFR}}$ (ΔNFR flux) | $\partial\Phi_s/\partial t + \mathrm{div}(J_{\Delta\mathrm{NFR}}) \approx 0$ |

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

This is an exact direct product, not the missing coupled derivation. Since the
cross blocks vanish, it supplies no pulse-relaxation feedback. A full bridge
still requires nonzero cross tensors that preserve the same degeneracy laws.
See [TNFR_SCALE_GEOMETRY_AND_BRIDGE.md](TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).

---

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

This equals `unified.compute_action_density()`. In the complete Lagrangian with interactions:

$$\mathcal{L}_{\mathrm{full}} = T - V - \mathcal{A}$$

The free Lagrangian $\mathcal{L} = T - V$ governs the independent sector dynamics, while $\mathcal{A}$ mediates the coupling between geometric and potential sectors.

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
| **Variational** (T / V) | $T = \tfrac{1}{2}[J_\varphi^2 + J_{\Delta NFR}^2]$, $V = \tfrac{1}{2}[\Phi_s^2 + |\nabla\varphi|^2 + K_\varphi^2]$ | Temporal role (kinetic / configuration) |
| **Conservation** ($\rho$ / J) | $\rho = \Phi_s + K_\varphi$, $\mathbf{J} = (J_\varphi, J_{\Delta NFR})$ | Physical role (conserved charge / flow) |
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

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [27_variational_principle_demo.py](../examples/02_physics_regimes/27_variational_principle_demo.py) | Lagrangian snapshots, conjugate pairs, Euler-Lagrange residual, action functional, symplectic preservation, grammar stationarity, critical points |

### Key Source Modules

- `src/tnfr/physics/variational.py` — Lagrangian, Hamiltonian, Euler-Lagrange, symplectic checks
- `src/tnfr/physics/conservation.py` — Energy functional (Lyapunov candidate)
- `src/tnfr/physics/classical_mechanics.py` — Classical limit correspondence

---

## References

- Nodal equation: `TNFR.pdf` §2.1, `AGENTS.md` §Foundational Physics
- Conservation theorem: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Classical mechanics analogy: `src/tnfr/physics/classical_mechanics.py`
- Unified fields (canonical source): `src/tnfr/physics/unified.py`
- Energy functional: `src/tnfr/physics/conservation.py`
