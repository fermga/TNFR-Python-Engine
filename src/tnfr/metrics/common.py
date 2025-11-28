"""Shared helpers for TNFR metrics."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ..alias import collect_attr, get_attr, multi_recompute_abs_max
from ..constants import DEFAULTS
from ..constants.aliases import ALIAS_D2EPI, ALIAS_DEPI, ALIAS_DNFR, ALIAS_VF
from ..utils import clamp01, kahan_sum_nd, normalize_optional_int
from ..types import GraphLike, NodeAttrMap
from ..utils import edge_version_cache, get_numpy, normalize_weights

__all__ = (
    "GraphLike",
    "compute_coherence",
    "compute_dnfr_accel_max",
    "normalize_dnfr",
    "ensure_neighbors_map",
    "merge_graph_weights",
    "merge_and_normalize_weights",
    "min_max_range",
    "_coerce_jobs",
    "_get_vf_dnfr_max",
)


def compute_coherence(
    G: GraphLike, *, return_means: bool = False
) -> float | tuple[float, float, float]:
    """Compute global coherence ``C`` from ``ΔNFR`` and ``dEPI``."""

    count = G.number_of_nodes()
    if count == 0:
        return (0.0, 0.0, 0.0) if return_means else 0.0

    nodes = G.nodes
    np = get_numpy()
    dnfr_values = collect_attr(G, nodes, ALIAS_DNFR, 0.0, np=np)
    depi_values = collect_attr(G, nodes, ALIAS_DEPI, 0.0, np=np)

    if np is not None:
        dnfr_mean = float(np.mean(np.abs(dnfr_values)))
        depi_mean = float(np.mean(np.abs(depi_values)))
    else:
        dnfr_sum, depi_sum = kahan_sum_nd(
            ((abs(d), abs(e)) for d, e in zip(dnfr_values, depi_values)),
            dims=2,
        )
        dnfr_mean = dnfr_sum / count
        depi_mean = depi_sum / count

    coherence = 1.0 / (1.0 + dnfr_mean + depi_mean)
    return (coherence, dnfr_mean, depi_mean) if return_means else coherence


def ensure_neighbors_map(G: GraphLike) -> Mapping[Any, Sequence[Any]]:
    """Return cached neighbors list keyed by node as a read-only mapping."""

    def builder() -> Mapping[Any, Sequence[Any]]:
        return MappingProxyType({n: tuple(G.neighbors(n)) for n in G})

    return edge_version_cache(G, "_neighbors", builder)


def merge_graph_weights(G: GraphLike, key: str) -> dict[str, float]:
    """Merge default weights for ``key`` with any graph overrides."""

    overrides = G.graph.get(key, {})
    if overrides is None or not isinstance(overrides, Mapping):
        overrides = {}
    return {**DEFAULTS[key], **overrides}


def merge_and_normalize_weights(
    G: GraphLike,
    key: str,
    fields: Sequence[str],
    *,
    default: float = 0.0,
) -> dict[str, float]:
    """Merge defaults for ``key`` and normalise ``fields``."""

    w = merge_graph_weights(G, key)
    return normalize_weights(
        w,
        fields,
        default=default,
        error_on_conversion=False,
        error_on_negative=False,
        warn_once=True,
    )


def compute_dnfr_accel_max(G: GraphLike) -> dict[str, float]:
    """Compute absolute maxima of |ΔNFR| and |d²EPI/dt²|."""

    return multi_recompute_abs_max(G, {"dnfr_max": ALIAS_DNFR, "accel_max": ALIAS_D2EPI})


def normalize_dnfr(nd: NodeAttrMap, max_val: float) -> float:
    """Normalise ``|ΔNFR|`` using ``max_val``."""

    if max_val <= 0:
        return 0.0
    val = abs(get_attr(nd, ALIAS_DNFR, 0.0))
    return clamp01(val / max_val)


def min_max_range(
    values: Iterable[float], *, default: tuple[float, float] = (0.0, 0.0)
) -> tuple[float, float]:
    """Return the minimum and maximum values observed in ``values``."""

    it = iter(values)
    try:
        first = next(it)
    except StopIteration:
        return default
    min_val = max_val = first
    for val in it:
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val
    return min_val, max_val


def _get_vf_dnfr_max(G: GraphLike) -> tuple[float, float]:
    """Ensure and return absolute maxima for ``νf`` and ``ΔNFR``."""

    vfmax = G.graph.get("_vfmax")
    dnfrmax = G.graph.get("_dnfrmax")
    if vfmax is None or dnfrmax is None:
        maxes = multi_recompute_abs_max(G, {"_vfmax": ALIAS_VF, "_dnfrmax": ALIAS_DNFR})
        if vfmax is None:
            vfmax = maxes["_vfmax"]
        if dnfrmax is None:
            dnfrmax = maxes["_dnfrmax"]
        G.graph["_vfmax"] = vfmax
        G.graph["_dnfrmax"] = dnfrmax
    vfmax = 1.0 if vfmax == 0 else vfmax
    dnfrmax = 1.0 if dnfrmax == 0 else dnfrmax
    return float(vfmax), float(dnfrmax)


def _coerce_jobs(raw_jobs: Any | None) -> int | None:
    """Normalise parallel job hints shared by metrics modules."""

    return normalize_optional_int(
        raw_jobs,
        allow_non_positive=False,
        strict=False,
        sentinels=None,
    )
