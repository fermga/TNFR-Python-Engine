"""Operator metrics facade for backward compatibility."""

from .metrics_basic import (
    emission_metrics,
    reception_metrics,
    coherence_metrics,
    dissonance_metrics,
)
from .metrics_network import (
    coupling_metrics,
    resonance_metrics,
    silence_metrics,
    # Private helpers exposed for testing
    _compute_epi_variance,
    _compute_preservation_integrity,
    _compute_reactivation_readiness,
    _estimate_time_to_collapse,
)
from .metrics_structural import (
    expansion_metrics,
    contraction_metrics,
    self_organization_metrics,
    mutation_metrics,
    transition_metrics,
    recursivity_metrics,
    # Private helper exposed for testing
    _detect_regime_from_state,
)

# U6 experimental telemetry
try:
    from .metrics_u6 import (
        measure_tau_relax_observed,
        measure_nonlinear_accumulation,
        compute_bifurcation_index,
    )
except Exception:
    from typing import Any
    def measure_tau_relax_observed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "metrics_u6 missing", "metric_type": "u6_relaxation_time"}
    def measure_nonlinear_accumulation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "metrics_u6 missing", "metric_type": "u6_nonlinear_accumulation"}
    def compute_bifurcation_index(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "metrics_u6 missing", "metric_type": "u6_bifurcation_index"}

__all__ = [
    "emission_metrics",
    "reception_metrics",
    "coherence_metrics",
    "dissonance_metrics",
    "coupling_metrics",
    "resonance_metrics",
    "silence_metrics",
    "expansion_metrics",
    "contraction_metrics",
    "self_organization_metrics",
    "mutation_metrics",
    "transition_metrics",
    "recursivity_metrics",
    "measure_tau_relax_observed",
    "measure_nonlinear_accumulation",
    "compute_bifurcation_index",
]
