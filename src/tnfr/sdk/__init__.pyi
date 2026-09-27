"""Type stubs for tnfr.sdk module."""

from ..dynamics.relational import RelationalExchangeModel as RelationalExchangeModel
from .adaptive_system import TNFRAdaptiveSystem as TNFRAdaptiveSystem
from .builders import TNFRExperimentBuilder as TNFRExperimentBuilder
from .fluent import NetworkConfig as NetworkConfig
from .fluent import NetworkResults as NetworkResults
from .fluent import TNFRNetwork as TNFRNetwork
from .relational_reports import relational_report_to_dict as relational_report_to_dict
from .self_opt import (
    run_batch_certificate_optimization as run_batch_certificate_optimization,
)
from .self_opt import (
    run_fractal_partition_optimization as run_fractal_partition_optimization,
)
from .self_opt import run_partition_self_optimization as run_partition_self_optimization
from .self_opt import (
    run_pattern_discovery_optimization as run_pattern_discovery_optimization,
)
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
from .templates import TNFRTemplates as TNFRTemplates
from .utils import compare_networks as compare_networks
from .utils import compute_network_statistics as compute_network_statistics
from .utils import export_to_json as export_to_json
from .utils import format_comparison_table as format_comparison_table
from .utils import import_from_json as import_from_json
from .utils import suggest_sequence_for_goal as suggest_sequence_for_goal

__all__: list[str]
