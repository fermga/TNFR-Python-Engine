"""SDK interfaces for declared TNFR graph construction, execution and observation.

``TNFR`` and ``Network`` provide the compact chainable interface;
``TNFRNetwork`` retains the configurable fluent workflows. ``StudySpec`` and
``run_study`` share finite operator recipes and JSON reports with the CLI.
``diagnose_network`` reads detached state with explicit field availability.
These interfaces reuse implemented model and grammar contracts. Their results
are not complete state reconstruction, autonomous-law or physical certificates.

Usage, report scope and reproducibility: ``docs/CLI_AND_SDK.md``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "StudySpec",
    "StudyResult",
    "run_study",
    "diagnose_network",
    "list_sequences",
    "STUDY_TOPOLOGIES",
    # Simplified API (recommended entry point)
    "TNFR",
    "Network",
    "Results",
    "TetradSnapshot",
    "ConservationReport",
    "SymplecticReport",
    "FactorizationReport",
    "PrimalityReport",
    "NodalStateReport",
    "NodalDynamicsReport",
    # Fluent API
    "TNFRNetwork",
    "NetworkConfig",
    "NetworkResults",
    "TNFRTemplates",
    "TNFRExperimentBuilder",
    "TNFRAdaptiveSystem",
    # Utilities
    "compare_networks",
    "compute_network_statistics",
    "export_to_json",
    "import_from_json",
    "format_comparison_table",
    "suggest_sequence_for_goal",
    "run_partition_self_optimization",
    "run_pattern_discovery_optimization",
    "run_fractal_partition_optimization",
    "run_batch_certificate_optimization",
]

# Lazy imports to avoid circular dependencies and optional dependency issues


def __getattr__(name: str) -> Any:
    """Lazy load SDK components."""
    if name in (
        "StudySpec",
        "StudyResult",
        "run_study",
        "diagnose_network",
        "list_sequences",
        "STUDY_TOPOLOGIES",
    ):
        from . import study

        return getattr(study, name)
    if name in (
        "TNFR",
        "Network",
        "Results",
        "TetradSnapshot",
        "ConservationReport",
        "SymplecticReport",
        "FactorizationReport",
        "PrimalityReport",
        "NodalStateReport",
        "NodalDynamicsReport",
    ):
        from .simple import (
            TNFR,
            ConservationReport,
            FactorizationReport,
            Network,
            NodalDynamicsReport,
            NodalStateReport,
            PrimalityReport,
            Results,
            SymplecticReport,
            TetradSnapshot,
        )

        mapping = {
            "TNFR": TNFR,
            "Network": Network,
            "Results": Results,
            "TetradSnapshot": TetradSnapshot,
            "ConservationReport": ConservationReport,
            "SymplecticReport": SymplecticReport,
            "FactorizationReport": FactorizationReport,
            "PrimalityReport": PrimalityReport,
            "NodalStateReport": NodalStateReport,
            "NodalDynamicsReport": NodalDynamicsReport,
        }
        return mapping[name]
    elif name == "TNFRNetwork" or name == "NetworkConfig" or name == "NetworkResults":
        from .fluent import NetworkConfig, NetworkResults, TNFRNetwork

        if name == "TNFRNetwork":
            return TNFRNetwork
        elif name == "NetworkConfig":
            return NetworkConfig
        else:
            return NetworkResults
    elif name == "TNFRTemplates":
        from .templates import TNFRTemplates

        return TNFRTemplates
    elif name == "TNFRExperimentBuilder":
        from .builders import TNFRExperimentBuilder

        return TNFRExperimentBuilder
    elif name == "TNFRAdaptiveSystem":
        from .adaptive_system import TNFRAdaptiveSystem

        return TNFRAdaptiveSystem
    elif name in [
        "compare_networks",
        "compute_network_statistics",
        "export_to_json",
        "import_from_json",
        "format_comparison_table",
        "suggest_sequence_for_goal",
    ]:
        from .utils import (
            compare_networks,
            compute_network_statistics,
            export_to_json,
            format_comparison_table,
            import_from_json,
            suggest_sequence_for_goal,
        )

        mapping = {
            "compare_networks": compare_networks,
            "compute_network_statistics": compute_network_statistics,
            "export_to_json": export_to_json,
            "import_from_json": import_from_json,
            "format_comparison_table": format_comparison_table,
            "suggest_sequence_for_goal": suggest_sequence_for_goal,
        }
        return mapping[name]
    elif name == "run_partition_self_optimization":
        from .self_opt import run_partition_self_optimization

        return run_partition_self_optimization
    elif name == "run_pattern_discovery_optimization":
        from .self_opt import run_pattern_discovery_optimization

        return run_pattern_discovery_optimization
    elif name == "run_fractal_partition_optimization":
        from .self_opt import run_fractal_partition_optimization

        return run_fractal_partition_optimization
    elif name == "run_batch_certificate_optimization":
        from .self_opt import run_batch_certificate_optimization

        return run_batch_certificate_optimization
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
