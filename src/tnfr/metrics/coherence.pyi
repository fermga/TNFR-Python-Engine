from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from _typeshed import Incomplete

from .._compat import TypeAlias as TypeAlias
from ..alias import collect_attr as collect_attr
from ..alias import collect_theta_attr as collect_theta_attr
from ..alias import get_attr as get_attr
from ..alias import set_attr as set_attr
from ..constants import get_param as get_param
from ..constants.aliases import ALIAS_D2VF as ALIAS_D2VF
from ..constants.aliases import ALIAS_DEPI as ALIAS_DEPI
from ..constants.aliases import ALIAS_DNFR as ALIAS_DNFR
from ..constants.aliases import ALIAS_DSI as ALIAS_DSI
from ..constants.aliases import ALIAS_DVF as ALIAS_DVF
from ..constants.aliases import ALIAS_EPI as ALIAS_EPI
from ..constants.aliases import ALIAS_SI as ALIAS_SI
from ..constants.aliases import ALIAS_VF as ALIAS_VF
from ..glyph_history import append_metric as append_metric
from ..glyph_history import ensure_history as ensure_history
from ..observers import DEFAULT_GLYPH_LOAD_SPAN as DEFAULT_GLYPH_LOAD_SPAN
from ..observers import DEFAULT_WBAR_SPAN as DEFAULT_WBAR_SPAN
from ..observers import glyph_load as glyph_load
from ..observers import kuramoto_order as kuramoto_order
from ..observers import phase_sync as phase_sync
from ..sense import sigma_vector as sigma_vector
from ..types import CoherenceMetric as CoherenceMetric
from ..types import FloatArray as FloatArray
from ..types import FloatMatrix as FloatMatrix
from ..types import GlyphLoadDistribution as GlyphLoadDistribution
from ..types import HistoryState as HistoryState
from ..types import NodeId as NodeId
from ..types import ParallelWijPayload as ParallelWijPayload
from ..types import SigmaVector as SigmaVector
from ..types import TNFRGraph as TNFRGraph
from ..utils import CallbackEvent as CallbackEvent
from ..utils import callback_manager as callback_manager
from ..utils import clamp01 as clamp01
from ..utils import ensure_node_index_map as ensure_node_index_map
from ..utils import get_logger as get_logger
from ..utils import get_numpy as get_numpy
from ..utils import normalize_weights as normalize_weights
from ..utils import resolve_chunk_size as resolve_chunk_size
from .common import compute_coherence as compute_coherence
from .common import min_max_range as min_max_range
from .trig_cache import compute_theta_trig as compute_theta_trig
from .trig_cache import get_trig_cache as get_trig_cache

logger: Incomplete
GLYPH_LOAD_STABILIZERS_KEY: str

@dataclass
class SimilarityInputs:
    th_vals: Sequence[float]
    epi_vals: Sequence[float]
    vf_vals: Sequence[float]
    si_vals: Sequence[float]
    cos_vals: Sequence[float] | None = ...
    sin_vals: Sequence[float] | None = ...

CoherenceMatrixDense = list[list[float]]
CoherenceMatrixSparse = list[tuple[int, int, float]]
CoherenceMatrixPayload = CoherenceMatrixDense | CoherenceMatrixSparse
PhaseSyncWeights: TypeAlias
SimilarityComponents = tuple[float, float, float, float]
VectorizedComponents: TypeAlias
ScalarOrArray: TypeAlias
StabilityChunkArgs = tuple[
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float | None],
    Sequence[float],
    Sequence[float | None],
    Sequence[float | None],
    float,
    float,
    float,
]
StabilityChunkResult = tuple[
    int, int, float, float, list[float], list[float], list[float]
]
MetricValue: TypeAlias
MetricProvider = Callable[[], MetricValue]
MetricRecord: TypeAlias

def compute_wij_phase_epi_vf_si(
    inputs: SimilarityInputs,
    i: int | None = None,
    j: int | None = None,
    *,
    trig: Any | None = None,
    G: TNFRGraph | None = None,
    nodes: Sequence[NodeId] | None = None,
    epi_range: float = 1.0,
    vf_range: float = 1.0,
    np: ModuleType | None = None,
) -> SimilarityComponents | VectorizedComponents: ...
def coherence_matrix(
    G: TNFRGraph, use_numpy: bool | None = None, *, n_jobs: int | None = None
) -> tuple[list[NodeId] | None, CoherenceMatrixPayload | None]: ...
def local_phase_sync_weighted(
    G: TNFRGraph,
    n: NodeId,
    nodes_order: Sequence[NodeId] | None = None,
    W_row: PhaseSyncWeights | None = None,
    node_to_index: Mapping[NodeId, int] | None = None,
) -> float: ...
def local_phase_sync(G: TNFRGraph, n: NodeId) -> float: ...
def register_coherence_callbacks(G: TNFRGraph) -> None: ...
