# TNFR Structural Field Tetrad (Canonical)

Status: CANONICAL (Updated 2025-11-12)

This guide centralizes the physics, math, implementation, telemetry, and usage of the four structural fields that characterize TNFR networks across scales. It is the **single canonical source** for formal definitions of Φ_s, \|∇φ\|, K_φ, and ξ_C; other documents SHOULD reference this file instead of restating the equations.

- Structural Potential (Φ_s): Global potential from ΔNFR distribution (inverse-square law analog)
- Phase Gradient (|∇φ|): Local phase desynchronization (stress proxy)
- Phase Curvature (K_φ): Geometric confinement/torsion
- Coherence Length (ξ_C): Spatial correlation scale of local coherence

References (single sources of truth):
- Theory: TNFR.pdf (§1–2), UNIFIED_GRAMMAR_RULES.md (§U1–U6)
- Canonical status and thresholds: AGENTS.md (Structural Field Tetrad)
- Implementation: src/tnfr/physics/fields.py
- Research background: docs/TNFR_FORCES_EMERGENCE.md (§14–15)
 - Physics module overview: src/tnfr/physics/README.md (unified, expandable)

---

## 1. Physics Basis

Nodal equation (core TNFR dynamics):

\[\frac{\partial EPI}{\partial t} = \nu_f\, \Delta NFR(t)\]

Integrated form and boundedness (U2):

\[EPI(t_f) = EPI(t_0) + \int_{t_0}^{t_f}\! \nu_f(\tau)\,\Delta NFR(\tau)\, d\tau,\quad \text{require } \int \nu_f\, \Delta NFR\, dt < \infty\]

Operator grammar enforces convergence via stabilizers (IL, THOL) around destabilizers (OZ, ZHIR, VAL) and requires phase-verification for coupling (U3). Telemetry fields below are read-only; they do not mutate EPI.

---

## 2. Canonical Field Definitions

All functions live in `tnfr.physics.fields` and accept a NetworkX graph `G` with the following node attributes when applicable:
- `theta` or `phase` (float in [0, 2π))
- `delta_nfr` or `dnfr` (float; structural pressure proxy)
- Optional `coherence` (float ∈ (0,1]) for ξ_C estimation

### 2.1 Structural Potential Φ_s (Global)

Definition (α = 2 by default):

\[\Phi_s(i) = \sum_{j\neq i} \frac{\Delta NFR_j}{d(i,j)^\alpha}\]

- Long-range, global potential derived from ΔNFR distribution
- Safety criterion (telemetry-based): ΔΦ_s < 2.0 typical; escape threshold at ~2.0
- Implementation: `compute_structural_potential(G, alpha=2.0)`

### 2.2 Phase Gradient |∇φ| (Local stress)

Wrapped neighbor differences (circular topology):

\[|\nabla\varphi|(i) = \operatorname{mean}_{j\in N(i)} \big|\operatorname{wrap}(\varphi_j-\varphi_i)\big|\]

- Early warning for fragmentation via local desynchronization
- Safety criterion: |∇φ| < 0.38 for stable operation (empirical)
- Implementation: `compute_phase_gradient(G)`

### 2.3 Phase Curvature K_φ (Geometric confinement)

Deviation from circular neighbor mean:

\[K_\varphi(i) = \varphi_i - \frac{1}{\deg(i)} \sum_{j\in N(i)} \varphi_j\]

- Use circular mean (unit vectors) and wrap deltas to (−π, π]
- Local threshold: |K_φ| ≥ 3.0 flags confinement/fault zones
- Multiscale behavior: var(K_φ) ~ 1/r^α with α≈2.76 (asymptotic freedom)
- Implementation:
  - `compute_phase_curvature(G)`
  - `compute_k_phi_multiscale_variance(G, scales)`
  - `fit_k_phi_asymptotic_alpha(var_by_scale)`
  - `k_phi_multiscale_safety(G, alpha_hint=2.76)`

### 2.4 Coherence Length ξ_C (Spatial correlations)

Local coherence: \(c_i = 1 / (1 + |\Delta NFR_i|)\) and spatial autocorrelation \(C(r) = \langle c_i c_j\rangle\) for pairs at distance r. Fit exponential decay:

\[C(r) \sim \exp(-r/\xi_C)\]

- Critical point behavior: ξ_C diverges near I_c (phase transitions)
- Safety cues: ξ_C > system diameter → system-wide reorganization imminent
- Implementation:
  - `estimate_coherence_length(G, coherence_key='coherence')`
  - `fit_correlation_length_exponent(Is, xi_vals, I_c, min_distance)`

---

## 3. Contracts, Units, and Invariants

- Read-only telemetry: No EPI mutation; fields compute from current node attributes
- Units: ν_f in Hz_str; do not mix with physical Hz (Invariant #2)
- Phase coupling requires explicit verification |Δφ| ≤ Δφ_max (U3)
- Valid sequences must satisfy U1 initiation/closure and U2 boundedness
- Nested EPIs require stabilizers at each level (U5)

Edge cases:
- Isolated nodes: return 0.0 for gradients/curvature; ignore in Φ_s sums
- Missing attributes: functions attempt sensible defaults; callers should initialize at least `theta`/`phase` and `delta_nfr`/`dnfr`

---

## 4. API Summary (tnfr.physics.fields)

- `compute_structural_potential(G, alpha: float = 2.0) -> dict[int,float]`
- `compute_phase_gradient(G) -> dict[int,float]`
- `compute_phase_curvature(G) -> dict[int,float]`
- `compute_k_phi_multiscale_variance(G, scales: tuple[int,...]) -> dict[int,float]`
- `fit_k_phi_asymptotic_alpha(var_by_scale: dict[int,float]) -> dict`
- `k_phi_multiscale_safety(G, scales=(1,2,3,5), alpha_hint=2.76, tolerance_factor=2.0, fit_min_r2=0.5) -> dict`
- `estimate_coherence_length(G, coherence_key='coherence') -> float`
- `fit_correlation_length_exponent(Is: array, xi_vals: array, I_c: float, min_distance=0.01) -> dict`
- `measure_phase_symmetry(G) -> dict`
- `path_integrated_gradient(G, path: list[int]) -> float`

Each function documents parameters and return types inline in `fields.py`.

---

## 5. Validation and Safety Thresholds

Canonical telemetry thresholds (empirical, cross-topology):
- Φ_s: maintain ΔΦ_s < 2.0 (escape threshold)  — see AGENTS.md (U6)
- |∇φ|: keep < 0.38 for stable operation; track spikes as early warning
- K_φ: flag |K_φ| ≥ 3.0 as hotspots; assess multiscale decay var(K_φ) ~ 1/r^α
- ξ_C: monitor divergence around I_c; large ξ_C indicates global reorganization

Minimum tests (see tests/ and AGENTS.md):
- Coherence monotonicity under IL
- Dissonance-triggered bifurcation with handlers present
- Resonance propagation increases phase synchrony
- Silence preserves EPI
- Mutation threshold crossing changes phase label
- Multiscale nested EPIs maintain coherence
- Seed reproducibility

---

## 6. Workflows and Tooling

- Integrated study: `benchmarks/integrated_force_regime_study.py` (six-task harness)
- Methods comparison: `benchmarks/grammar_2_0_benchmarks.py` and summaries
- Plotting: `benchmarks/plot_force_study_summaries.py` → saves to `results/plots/*.png`
- Notebook: `notebooks/Force_Fields_Tetrad_Exploration.ipynb` (end-to-end)
- Static report: `results/reports/Force_Fields_Tetrad_Exploration.html`
- VS Code tasks: `.vscode/tasks.json`
  - “Export TNFR tetrad HTML report”
  - “Generate force study plots”

Artifacts:
- `results/integrated_force_study_summary.json`
- `results/field_methods_battery_summary.json`
- `results/plots/*.png`

---

## 7. Minimal Example

```python
import networkx as nx
from tnfr.physics.fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
)

G = nx.watts_strogatz_graph(60, k=4, p=0.2, seed=42)
# Initialize minimal telemetry
for n in G.nodes():
    G.nodes[n]['theta'] = 0.1 * (n/59.0)
    G.nodes[n]['delta_nfr'] = 0.1

phi = compute_structural_potential(G, alpha=2.0)
grad = compute_phase_gradient(G)
kphi = compute_phase_curvature(G)
xi = estimate_coherence_length(G, coherence_key='coherence')  # if provided
```

---

## 8. Governance and Traceability

- Physics-first: all field definitions derive from nodal equation semantics
- No ad-hoc mutations: fields are telemetry-only; EPI changes go through operators
- Units and invariants preserved (see AGENTS.md invariants 1–10)
- Canonical docs: this page + UNIFIED_GRAMMAR_RULES.md are the reference; research-phase content (e.g., TNFR_FORCES_EMERGENCE.md) is linked but not normative

---

## 9. Further Reading

- AGENTS.md — Canonical invariants and field promotions (Φ_s, |∇φ|, K_φ, ξ_C)
- UNIFIED_GRAMMAR_RULES.md — U1–U6 derivations and constraints
- TNFR_FORCES_EMERGENCE.md — Historical research path for field validation
- SHA_ALGEBRA_PHYSICS.md — Supporting mathematical apparatus

---

## 10. FAQ

Q1. What node attributes are required to compute each field?
- Φ_s: requires `delta_nfr` or `dnfr` on nodes; uses graph distances.
- |∇φ| and K_φ: require `theta` or `phase` on nodes (float in [0, 2π)).
- ξ_C: optionally uses `coherence` on nodes; if absent, it estimates from `delta_nfr` via c_i = 1/(1+|ΔNFR_i|).

Q2. Why are phase differences wrapped? Can I just subtract angles?
- Phases live on the circle. Direct subtraction misinterprets, e.g., 0 and 2π as far apart. We compute circular means (via unit vectors) and wrap differences to (−π, π] to preserve correct geometry.

Q3. How should I choose α in Φ_s?
- α = 2.0 is canonical (inverse-square analog) and validated across topologies. Deviations are research-only; if you change α, document and justify the physics in your application.

Q4. Are thresholds (ΔΦ_s < 2.0, |∇φ| < 0.38, |K_φ| ≥ 3.0) universal?
- They are telemetry-based and robust across the tested families (WS, scale-free, grid, trees) but still empirical. Treat them as safety guidance, not as hard correctness proofs. Monitor trends over time, not just single snapshots.

Q5. What graphs are supported? Weighted? Directed?
- Implementations are designed for undirected, unweighted graphs. Φ_s currently uses unweighted shortest-path distances. If your graph is weighted or directed, pre-process to an appropriate undirected/unweighted view or extend the distance routine consistently with TNFR physics.

Q6. What happens if attributes are missing?
- Functions fall back conservatively (e.g., 0.0 for empty neighborhoods) but you should initialize at least `theta`/`phase` and `delta_nfr`/`dnfr`. For ξ_C, if `coherence` is missing, it infers local coherence from ΔNFR magnitudes.

Q7. ξ_C returned NaN/inf. What does that mean?
- Near criticality, an exponential fit may be ill-posed (flat or noisy C(r)). Re-run with more samples, verify `coherence` distribution, or widen the r-range. If the system is truly at/near I_c, very large ξ_C is expected; treat it as a warning for imminent system-wide reorganization.

Q8. How does the tetrad relate to C(t) and Si?
- C(t) is a global coherence scalar; Si measures stable reorganization capacity. The tetrad provides complementary structure: Φ_s (global field), |∇φ| (local stress), K_φ (geometric confinement), ξ_C (spatial correlation scale). Use them together for a complete picture. Note: C(t) is invariant to proportional ΔNFR scaling; |∇φ| often captures early local stress better.

Q9. Performance tips for large graphs?
- Φ_s requires many distance evaluations; on large graphs, consider limiting to a radius, sampling source nodes, or caching all-pairs shortest paths if topology is static. |∇φ| and K_φ are O(E) and scale well. ξ_C can subsample pairs at each distance bin.

Q10. What should I do when safety flags trigger?
- Apply stabilizers (IL, THOL), verify phase-compatibility before coupling (U3), reduce destabilizer intensity (OZ, ZHIR, VAL), and monitor ΔΦ_s, |∇φ|, and K_φ decay across scales. Ensure ν_f units remain in Hz_str and do not mutate EPI outside operators.

Q11. Reproducibility and randomness?
- Set seeds (Python, NumPy) before generating telemetry or randomized structures. The same seed must yield identical trajectories and telemetry (Invariant #8).

Q12. Can I extend these fields or add new ones?
- Yes, but only with physics-first justification. Derive from the nodal equation, preserve invariants, map to operators where applicable, and add tests and documentation. Experimental fields must be clearly labeled non-canonical until validated.

---

## Appendix: Topological winding (Q) — complementary telemetry

While not part of the field tetrad, the topological winding number around a
closed loop provides a complementary invariant for identifying phase defects
and vortex-like structures:

Definition:
  Q = round( (1 / 2π) · Σ wrap(φ_{i+1} − φ_i) ) over a closed cycle.

- Implementation: `tnfr.physics.fields.compute_phase_winding(G, cycle_nodes)`
- Usage: helpful to distinguish plane-wave-like (Q≈0) from vortex-like (Q=±1)
  configurations in pattern studies.
- Related initializations: `tnfr.physics.patterns.apply_vortex`,
  `apply_plane_wave`, `apply_quark_triplet_cluster`.

This metric is telemetry-only and preserves all canonical invariants.
