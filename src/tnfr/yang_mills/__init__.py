"""TNFR–Yang–Mills structural gap diagnostics.

This package contains TNFR-native diagnostics for the Yang–Mills / mass-gap
research programme documented in ``theory/TNFR_YANG_MILLS_RESEARCH_NOTES.md``.
The implementation is strictly read-only with respect to EPI: it assembles
finite spectral diagnostics from canonical fields (Ψ, Φ_s) and the auxiliary
pure-gauge quantities (A, F) provided by ``tnfr.physics.gauge``.

Honest scope
------------
These routines do not solve the Clay Yang–Mills and Mass Gap problem.  They
create the Y1 finite-graph attack surface: a reproducible structural gauge
operator and its first spectral gap under declared finite diagnostics. The
bundled connection is a pure vertex-phase difference, so its cycle curvature
vanishes analytically and the reported curvature is numerical residue.
Y2 adds a compatibility-named single-snapshot structural-potential magnitude
sweep normalized by the U6 drift scale; it reports the separate π/4 magnitude
warning and does not assess the two-snapshot U6 drift policy.
Y3 audits whether a non-Abelian gauge sector is derivable from TNFR-internal
data only; the current conservative verdict is an open derivability gap.
Y4 adds conditional finite graph-size scaling diagnostics.
Y5 classifies the programme closure / obstruction state.
"""

from .closure import YangMillsClosureReport, classify_yang_mills_closure
from .derivability import (
    NonAbelianCandidateAudit,
    NonAbelianDerivabilityReport,
    audit_nonabelian_derivability,
)
from .scaling import FiniteScalingPoint, FiniteScalingReport, run_finite_scaling_study
from .structural_gap import (
    StructuralGaugeGapOperator,
    StructuralGaugeGapResult,
    build_structural_gauge_gap_operator,
    build_structural_gauge_graph,
    compute_structural_gauge_gap,
)
from .u6_sweep import (
    U6ConfinementSweepPoint,
    U6ConfinementSweepReport,
    run_u6_confinement_sweep,
)

__all__ = [
    "FiniteScalingPoint",
    "FiniteScalingReport",
    "NonAbelianCandidateAudit",
    "NonAbelianDerivabilityReport",
    "StructuralGaugeGapOperator",
    "StructuralGaugeGapResult",
    "U6ConfinementSweepPoint",
    "U6ConfinementSweepReport",
    "YangMillsClosureReport",
    "audit_nonabelian_derivability",
    "build_structural_gauge_gap_operator",
    "build_structural_gauge_graph",
    "compute_structural_gauge_gap",
    "run_finite_scaling_study",
    "run_u6_confinement_sweep",
    "classify_yang_mills_closure",
]
