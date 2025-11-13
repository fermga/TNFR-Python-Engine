"""TNFR Physics Module — Canonical Structural Telemetry (Expandable)

This package exposes physics-based structural telemetry derived from the
TNFR nodal equation and validated empirically. All functions are centralized
in a single module to avoid duplication and ensure a clear single source of
truth. Documentation is English-only and organized for incremental growth.

Canonical Structural Field Tetrad (Telemetry)
---------------------------------------------
All four fields below are CANONICAL (November 2025) and read-only:

1) Structural Potential (Φ_s)
   - Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α (α=2)
   - Validation: 2,400+ experiments; corr(ΔΦ_s, ΔC) = -0.822; CV = 0.1%
   - Safety criterion: ΔΦ_s < 2.0 (escape threshold; U6 telemetry)

2) Phase Gradient (|∇φ|)
   - |∇φ|(i) = mean_{j∈N(i)} |wrap(φ_j − φ_i)| (circular differences)
   - Predicts peak stress; |corr| > 0.5 across topologies; threshold 0.38

3) Phase Curvature (K_φ)
   - K_φ(i) = φ_i − mean_circular_{j∈N(i)} φ_j (Laplacian-like curvature)
   - Threshold |K_φ| ≥ 3.0 (hotspots); multiscale var(K_φ) ~ 1/r^α, α≈2.76

4) Coherence Length (ξ_C)
   - From spatial decay of local coherence correlations C(r) ~ exp(−r/ξ_C)
   - Diverges near I_c (phase transitions); large ξ_C warns system-wide reorg

Physics Foundation
------------------
Nodal equation (per node):  ∂EPI/∂t = ν_f · ΔNFR(t)
EPI changes only via operators; telemetry functions compute read-only fields
from current graph attributes. ν_f uses structural units Hz_str.

Modules
-------
fields : Centralized structural field computations and research utilities
    - compute_structural_potential, compute_phase_gradient,
      compute_phase_curvature, estimate_coherence_length
    - k_φ multiscale helpers and topological winding (Q)
interactions : Canonical operator sequences with telemetry guards
      - em_like, weak_like, strong_like, gravity_like
         (returning InteractionResult)
life : Life emergence detection from autopoietic TNFR dynamics
    - detect_life_emergence, LifeTelemetry, autopoietic coefficients
    - Threshold: A > 1.0 for autopoietic behavior
cell : Cell formation from compartmentalized TNFR life patterns
    - detect_cell_formation, CellTelemetry, membrane selectivity
    - Requires life foundation (A > 1.0) plus spatial organization

See Also
--------
tnfr.operators.grammar : Unified Grammar (U1–U6) and validations
tnfr.dynamics           : Nodal equation integration utilities
docs/STRUCTURAL_FIELDS_TETRAD.md : Canonical tetrad documentation
AGENTS.md               : Canonical invariants and field promotions
src/tnfr/physics/README.md        : Module hub (Patterns,
                                     Interactions, Workflows)

References
----------
- UNIFIED_GRAMMAR_RULES.md (U6: Structural potential confinement)
- docs/TNFR_FORCES_EMERGENCE.md (§14–15: Φ_s, |∇φ|, K_φ, ξ_C validation)
- AGENTS.md (Structural Fields Tetrad: canonical status, thresholds)
- TNFR.pdf (§2.1: Nodal equation foundation)

Examples
--------
>>> from tnfr.physics.fields import compute_structural_potential
>>> import networkx as nx
>>> G = nx.karate_club_graph()
>>> for node in G.nodes():
...     G.nodes[node]['delta_nfr'] = 0.5
>>> phi_s = compute_structural_potential(G, alpha=2.0)
>>> print(f"Potential at node 0: {phi_s[0]:.3f}")

>>> # Telemetry-based U6 safety (ΔΦ_s drift)
>>> phi_before = compute_structural_potential(G)
>>> # ... apply sequence ...
>>> phi_after = compute_structural_potential(G)
>>> drift = sum(
...     abs(phi_after[n] - phi_before[n]) for n in G.nodes()
... ) / G.number_of_nodes()
>>> assert drift < 2.0, "Escape threshold exceeded"

"""

from .fields import (
    compute_structural_potential,
    compute_phase_gradient,
    compute_phase_curvature,
    estimate_coherence_length,
    compute_k_phi_multiscale_variance,
    fit_k_phi_asymptotic_alpha,
    k_phi_multiscale_safety,
    compute_phase_winding,
)
from .interactions import InteractionResult
from .interactions import em_like
from .interactions import weak_like
from .interactions import strong_like
from .interactions import gravity_like
from .life import LifeTelemetry
from .life import compute_self_generation
from .life import compute_autopoietic_coefficient
from .life import compute_self_org_index
from .life import compute_stability_margin
from .life import detect_life_emergence
from .cell import CellTelemetry
from .cell import compute_boundary_coherence
from .cell import compute_selectivity_index
from .cell import compute_homeostatic_index
from .cell import compute_membrane_integrity
from .cell import detect_cell_formation
from .cell import apply_membrane_flux

__all__ = []
__all__ += ["compute_structural_potential"]
__all__ += ["compute_phase_gradient"]
__all__ += ["compute_phase_curvature"]
__all__ += ["estimate_coherence_length"]
__all__ += ["compute_k_phi_multiscale_variance"]
__all__ += ["fit_k_phi_asymptotic_alpha"]
__all__ += ["k_phi_multiscale_safety"]
__all__ += ["compute_phase_winding"]
__all__ += ["InteractionResult"]
__all__ += ["em_like"]
__all__ += ["weak_like"]
__all__ += ["strong_like"]
__all__ += ["gravity_like"]
__all__ += ["LifeTelemetry"]
__all__ += ["compute_self_generation"]
__all__ += ["compute_autopoietic_coefficient"]
__all__ += ["compute_self_org_index"]
__all__ += ["compute_stability_margin"]
__all__ += ["detect_life_emergence"]
__all__ += ["CellTelemetry"]
__all__ += ["compute_boundary_coherence"]
__all__ += ["compute_selectivity_index"]
__all__ += ["compute_homeostatic_index"]
__all__ += ["compute_membrane_integrity"]
__all__ += ["detect_cell_formation"]
__all__ += ["apply_membrane_flux"]
