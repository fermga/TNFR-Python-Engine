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

from importlib import import_module
from typing import Any

# One owner map supplies both discovery and lazy public dispatch.
_EXPORT_GROUPS = {
    ".study": (
        "StudySpec",
        "StudyResult",
        "run_study",
        "diagnose_network",
        "list_sequences",
        "STUDY_TOPOLOGIES",
    ),
    "..dynamics.relational": ("RelationalExchangeModel",),
    ".relational_reports": ("relational_report_to_dict",),
    ".simple": (
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
    ),
    ".fluent": (
        "TNFRNetwork",
        "NetworkConfig",
        "NetworkResults",
    ),
    ".templates": ("TNFRTemplates",),
    ".builders": ("TNFRExperimentBuilder",),
    ".adaptive_system": ("TNFRAdaptiveSystem",),
    ".utils": (
        "compare_networks",
        "compute_network_statistics",
        "export_to_json",
        "import_from_json",
        "format_comparison_table",
        "suggest_sequence_for_goal",
    ),
    ".self_opt": (
        "run_partition_self_optimization",
        "run_pattern_discovery_optimization",
        "run_fractal_partition_optimization",
        "run_batch_certificate_optimization",
    ),
}
_EXPORT_MODULES = {
    name: module for module, names in _EXPORT_GROUPS.items() for name in names
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Resolve a supported SDK export without eagerly loading its owner."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(import_module(module_name, __name__), name)
