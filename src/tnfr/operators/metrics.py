"""Operator metrics facade for backward compatibility."""

from .metrics_basic import (
    coherence_metrics,
    dissonance_metrics,
    emission_metrics,
    reception_metrics,
)
from .metrics_network import coupling_metrics, resonance_metrics, silence_metrics
from .metrics_structural import (
    contraction_metrics,
    expansion_metrics,
    mutation_metrics,
    recursivity_metrics,
    self_organization_metrics,
    transition_metrics,
)

# U6 experimental telemetry
try:
    from .metrics_u6 import (
        compute_bifurcation_index,
        measure_nonlinear_accumulation,
        measure_tau_relax_observed,
    )
except Exception:
    from typing import Any

    def measure_tau_relax_observed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"error": "metrics_u6 missing", "metric_type": "u6_relaxation_time"}

    def measure_nonlinear_accumulation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "error": "metrics_u6 missing",
            "metric_type": "u6_nonlinear_accumulation",
        }

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
