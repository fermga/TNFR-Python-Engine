"""Type stubs for tnfr.sdk module."""

from typing import Any

from .simple import TNFR as TNFR
from .simple import ConservationReport as ConservationReport
from .simple import FactorizationReport as FactorizationReport
from .simple import Network as Network
from .simple import NodalDynamicsReport as NodalDynamicsReport
from .simple import NodalStateReport as NodalStateReport
from .simple import PrimalityReport as PrimalityReport
from .simple import Results as Results
from .simple import SymplecticReport as SymplecticReport
from .simple import TetradSnapshot as TetradSnapshot

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
