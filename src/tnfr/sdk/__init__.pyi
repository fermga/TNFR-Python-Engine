"""Type stubs for tnfr.sdk module."""

from typing import Any

from .simple import (
    ConservationReport as ConservationReport,
    FactorizationReport as FactorizationReport,
    Network as Network,
    NodalDynamicsReport as NodalDynamicsReport,
    NodalStateReport as NodalStateReport,
    PrimalityReport as PrimalityReport,
    Results as Results,
    SymplecticReport as SymplecticReport,
    TetradSnapshot as TetradSnapshot,
    TNFR as TNFR,
)

__all__: tuple[str, ...]

TNFRNetwork: Any
NetworkConfig: Any
NetworkResults: Any
TNFRTemplates: Any
TNFRExperimentBuilder: Any
TNFRAdaptiveSystem: Any

compare_networks: Any
compute_network_statistics: Any
export_to_json: Any
import_from_json: Any
format_comparison_table: Any
suggest_sequence_for_goal: Any
run_partition_self_optimization: Any
run_pattern_discovery_optimization: Any
run_fractal_partition_optimization: Any
run_batch_certificate_optimization: Any
