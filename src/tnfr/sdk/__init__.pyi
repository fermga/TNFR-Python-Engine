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
from .study import STUDY_TOPOLOGIES as STUDY_TOPOLOGIES
from .study import StudyResult as StudyResult
from .study import StudySpec as StudySpec
from .study import diagnose_network as diagnose_network
from .study import list_sequences as list_sequences
from .study import run_study as run_study

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
