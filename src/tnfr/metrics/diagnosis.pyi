from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__: Any

def __getattr__(name: str) -> Any: ...

ALIAS_DNFR: Any
ALIAS_EPI: Any
ALIAS_SI: Any
ALIAS_VF: Any
CallbackEvent: Any
CoherenceMatrixPayload: Any
Iterable: Any
ProcessPoolExecutor: Any
StatisticsError: Any
TNFRGraph: Any
TRANSITION: Any
VF_KEY: Any
append_metric: Any
callback_manager: Any
clamp01: Any
coherence_matrix: Any
compute_dnfr_accel_max: Any
compute_theta_trig: Any
dissonance_events: Any
ensure_history: Any
fmean: Any
ge: Any
get_aliases: Any
get_attr: Any
get_numpy: Any
get_param: Any
get_trig_cache: Any
le: Any
local_phase_sync: Any
math: Any
min_max_range: Any
normalize_dnfr: Any
partial: Any
register_diagnosis_callbacks: Any
similarity_abs: Any

class RLocalWorkerArgs:
    chunk: Sequence[Any]
    coherence_nodes: Sequence[Any]
    weight_matrix: Any
    weight_index: Mapping[Any, int]
    neighbors_map: Mapping[Any, tuple[Any, ...]]
    cos_map: Mapping[Any, float]
    sin_map: Mapping[Any, float]

    def __init__(
        self,
        chunk: Sequence[Any],
        coherence_nodes: Sequence[Any],
        weight_matrix: Any,
        weight_index: Mapping[Any, int],
        neighbors_map: Mapping[Any, tuple[Any, ...]],
        cos_map: Mapping[Any, float],
        sin_map: Mapping[Any, float],
    ) -> None: ...

class NeighborMeanWorkerArgs:
    chunk: Sequence[Any]
    neighbors_map: Mapping[Any, tuple[Any, ...]]
    epi_map: Mapping[Any, float]

    def __init__(
        self,
        chunk: Sequence[Any],
        neighbors_map: Mapping[Any, tuple[Any, ...]],
        epi_map: Mapping[Any, float],
    ) -> None: ...

def _rlocal_worker(args: RLocalWorkerArgs) -> list[float]: ...
def _neighbor_mean_worker(args: NeighborMeanWorkerArgs) -> list[float | None]: ...
def _state_from_thresholds(Rloc: float, dnfr_n: float, cfg: Mapping[str, Any]) -> str: ...
def _recommendation(state: str, cfg: Mapping[str, Any]) -> list[Any]: ...
def _get_last_weights(
    G: TNFRGraph,
    hist: Mapping[str, Sequence[CoherenceMatrixPayload | None]],
) -> tuple[CoherenceMatrixPayload | None, CoherenceMatrixPayload | None]: ...
