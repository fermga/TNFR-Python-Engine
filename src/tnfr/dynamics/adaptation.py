"""Structural-stability-gated frequency adaptation.

The gate combines a small absolute DeltaNFR value with a high Sense Index.
It does not evaluate the canonical total-coherence kernel C(t), because no
EPI-rate channel is read.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral, Real
from typing import Any

from ..alias import set_vf
from ..constants import get_param
from ..constants.canonical import DYNAMICS_SI_HI_THRESHOLD_CANONICAL
from ..metrics.common import ensure_neighbors_map
from ..types import TNFRGraph
from ..utils import clamp, resolve_chunk_size
from .aliases import ALIAS_DNFR, ALIAS_SI, ALIAS_VF

__all__ = (
    "adapt_vf_after_structural_stability",
    "adapt_vf_by_coherence",
)


def _finite_real(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return one finite real scalar within optional closed bounds."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(f"{label} must be a finite real number") from None
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite real number")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return normalized


def _integer_at_least(value: Any, label: str, minimum: int) -> int:
    """Return one integral value without truncating floats or booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer >= {minimum}")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return normalized


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return one configuration mapping without coercing other containers."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _first_alias_value(
    attributes: Mapping[str, Any],
    aliases: Sequence[str],
    default: Any,
) -> Any:
    """Read the first present alias without permissive scalar coercion."""

    for key in aliases:
        if key in attributes:
            return attributes[key]
    return default


def _stable_mean(
    values: tuple[float, ...],
    neighbor_indices: tuple[int, ...],
    fallback: float,
) -> float:
    """Return a range-safe arithmetic mean for nonnegative frequencies."""

    if not neighbor_indices:
        return fallback
    scale = max(values[index] for index in neighbor_indices)
    if scale == 0.0:
        return 0.0
    normalized_total = math.fsum(
        values[index] / scale for index in neighbor_indices
    )
    mean = scale * (normalized_total / len(neighbor_indices))
    if not math.isfinite(mean):
        raise ValueError("neighbor frequency mean must remain finite")
    return mean


def _vf_adapt_chunk(
    args: tuple[
        list[tuple[Any, int, tuple[int, ...]]],
        tuple[float, ...],
        float,
    ],
) -> list[tuple[Any, float]]:
    """Return immutable-snapshot frequency proposals for one work chunk."""

    chunk, vf_values, mu = args
    updates: list[tuple[Any, float]] = []
    for node, index, neighbor_indices in chunk:
        vf = vf_values[index]
        mean = _stable_mean(vf_values, neighbor_indices, vf)
        proposed = (1.0 - mu) * vf + mu * mean
        if not math.isfinite(proposed):
            raise ValueError("adapted structural frequency must remain finite")
        updates.append((node, proposed))
    return updates


def _validated_parameters(
    G: TNFRGraph,
    n_jobs: int | None,
) -> tuple[int, float, float, float, float, float, int | None]:
    """Validate every graph-owned and call-owned adaptation parameter."""

    required_keys = ("VF_ADAPT_TAU", "VF_ADAPT_MU")
    missing_keys = [key for key in required_keys if key not in G.graph]
    if missing_keys:
        missing_list = ", ".join(sorted(missing_keys))
        raise KeyError(
            "adapt_vf_after_structural_stability requires graph parameters "
            f"{missing_list}; call tnfr.constants.inject_defaults(G) "
            "before adaptation."
        )

    tau = _integer_at_least(get_param(G, "VF_ADAPT_TAU"), "VF_ADAPT_TAU", 1)
    mu = _finite_real(
        get_param(G, "VF_ADAPT_MU"),
        "VF_ADAPT_MU",
        minimum=0.0,
        maximum=1.0,
    )
    eps_dnfr = _finite_real(
        get_param(G, "EPS_DNFR_STABLE"),
        "EPS_DNFR_STABLE",
        minimum=0.0,
    )

    selector_thresholds = _mapping(
        get_param(G, "SELECTOR_THRESHOLDS"),
        "SELECTOR_THRESHOLDS",
    )
    fallback_thresholds = _mapping(
        get_param(G, "GLYPH_THRESHOLDS"),
        "GLYPH_THRESHOLDS",
    )
    si_hi = _finite_real(
        selector_thresholds.get(
            "si_hi",
            fallback_thresholds.get(
                "hi",
                DYNAMICS_SI_HI_THRESHOLD_CANONICAL,
            ),
        ),
        "SELECTOR_THRESHOLDS['si_hi']",
        minimum=0.0,
        maximum=1.0,
    )

    vf_min = _finite_real(get_param(G, "VF_MIN"), "VF_MIN", minimum=0.0)
    vf_max = _finite_real(get_param(G, "VF_MAX"), "VF_MAX", minimum=0.0)
    if vf_max < vf_min:
        raise ValueError("VF_MAX must be >= VF_MIN")

    if n_jobs is None:
        jobs = None
    else:
        requested_jobs = _integer_at_least(n_jobs, "n_jobs", 1)
        jobs = None if requested_jobs == 1 else requested_jobs

    return tau, mu, eps_dnfr, si_hi, vf_min, vf_max, jobs


def _validated_node_state(
    G: TNFRGraph,
    nodes: Sequence[Any],
    *,
    vf_min: float,
    vf_max: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[int, ...],
]:
    """Read and validate all node inputs before any state mutation."""

    si_values: list[float] = []
    dnfr_values: list[float] = []
    vf_values: list[float] = []
    stable_counts: list[int] = []

    for node in nodes:
        attributes = G.nodes[node]
        prefix = f"node {node!r}"
        si_values.append(
            _finite_real(
                _first_alias_value(attributes, ALIAS_SI, 0.0),
                f"{prefix} Si",
                minimum=0.0,
            )
        )
        dnfr_values.append(
            _finite_real(
                _first_alias_value(attributes, ALIAS_DNFR, 0.0),
                f"{prefix} DeltaNFR",
            )
        )
        vf_values.append(
            _finite_real(
                _first_alias_value(attributes, ALIAS_VF, 0.0),
                f"{prefix} nu_f",
                minimum=vf_min,
                maximum=vf_max,
            )
        )
        stable_counts.append(
            _integer_at_least(
                attributes.get("stable_count", 0),
                f"{prefix} stable_count",
                0,
            )
        )

    return (
        tuple(si_values),
        tuple(dnfr_values),
        tuple(vf_values),
        tuple(stable_counts),
    )


def _restore_transaction(
    G: TNFRGraph,
    graph_snapshot: dict[str, Any],
    node_snapshots: Mapping[Any, dict[str, Any]],
) -> None:
    """Restore graph and node attribute mappings after a failed commit."""

    G.graph.clear()
    G.graph.update(graph_snapshot)
    for node, snapshot in node_snapshots.items():
        attributes = G.nodes[node]
        attributes.clear()
        attributes.update(snapshot)


def adapt_vf_after_structural_stability(
    G: TNFRGraph,
    n_jobs: int | None = None,
) -> None:
    """Synchronize nu_f after the DeltaNFR-plus-Si stability gate persists.

    A node is stable for this operational gate when abs(DeltaNFR) is no larger
    than EPS_DNFR_STABLE and Si is at least the configured si_hi threshold.
    After VF_ADAPT_TAU consecutive qualifying evaluations, its frequency moves
    by VF_ADAPT_MU toward the immutable-snapshot mean of its neighbors.

    This routine does not read dEPI/dt and therefore does not compute or gate on
    canonical total coherence C(t). All parameters and node scalars are
    validated before mutation. Stable counters and frequency updates commit as
    one transaction; any proposal, worker, or setter failure restores both.

    Parameters
    ----------
    G
        Graph with injected TNFR defaults and scalar Si, DeltaNFR, nu_f, and
        optional stable_count node attributes.
    n_jobs
        None or 1 selects serial proposal calculation. A positive integer
        greater than one enables process-based proposal calculation.

    Examples
    --------
    >>> from tnfr.constants import inject_defaults
    >>> from tnfr.dynamics import adapt_vf_after_structural_stability
    >>> from tnfr.structural import create_nfr
    >>> G, seed = create_nfr("seed", vf=0.2)
    >>> _, anchor = create_nfr("anchor", graph=G, vf=1.0)
    >>> G.add_edge(seed, anchor)
    >>> inject_defaults(G)
    >>> G.graph["VF_ADAPT_TAU"] = 2
    >>> G.graph["VF_ADAPT_MU"] = 0.5
    >>> G.graph["SELECTOR_THRESHOLDS"] = {"si_hi": 0.8}
    >>> for node in G.nodes:
    ...     G.nodes[node]["Si"] = 0.9
    ...     G.nodes[node]["ΔNFR"] = 0.0
    ...     G.nodes[node]["stable_count"] = 1
    >>> adapt_vf_after_structural_stability(G)
    >>> round(G.nodes[seed]["νf"], 2), round(G.nodes[anchor]["νf"], 2)
    (0.6, 0.6)
    """

    (
        tau,
        mu,
        eps_dnfr,
        si_hi,
        vf_min,
        vf_max,
        jobs,
    ) = _validated_parameters(G, n_jobs)

    nodes = list(G.nodes)
    if not nodes:
        return

    (
        si_values,
        dnfr_values,
        vf_values,
        previous_counts,
    ) = _validated_node_state(
        G,
        nodes,
        vf_min=vf_min,
        vf_max=vf_max,
    )

    stable_flags = tuple(
        si >= si_hi and abs(dnfr) <= eps_dnfr
        for si, dnfr in zip(si_values, dnfr_values)
    )
    new_counts = tuple(
        previous + 1 if stable else 0
        for previous, stable in zip(previous_counts, stable_flags)
    )
    eligible_indices = tuple(
        index for index, count in enumerate(new_counts) if count >= tau
    )

    graph_snapshot = dict(G.graph)
    node_snapshots = {node: dict(G.nodes[node]) for node in nodes}

    try:
        neighbors_map = ensure_neighbors_map(G)
        node_index = {node: index for index, node in enumerate(nodes)}
        work_items = [
            (
                nodes[index],
                index,
                tuple(
                    node_index[neighbor]
                    for neighbor in neighbors_map.get(nodes[index], ())
                    if neighbor in node_index
                ),
            )
            for index in eligible_indices
        ]

        if jobs is None or len(work_items) <= 1:
            raw_updates = _vf_adapt_chunk((work_items, vf_values, mu))
        else:
            worker_count = min(jobs, len(work_items))
            approximate_chunk = math.ceil(len(work_items) / worker_count)
            chunk_size = resolve_chunk_size(
                approximate_chunk,
                len(work_items),
                minimum=1,
            )
            chunks = [
                work_items[index : index + chunk_size]
                for index in range(0, len(work_items), chunk_size)
            ]
            raw_updates = []
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                arguments = ((chunk, vf_values, mu) for chunk in chunks)
                for chunk_updates in executor.map(_vf_adapt_chunk, arguments):
                    raw_updates.extend(chunk_updates)

        proposals = {
            node: clamp(
                _finite_real(value, f"node {node!r} adapted nu_f"),
                vf_min,
                vf_max,
            )
            for node, value in raw_updates
        }
        eligible_nodes = [nodes[index] for index in eligible_indices]
        if any(node not in proposals for node in eligible_nodes):
            raise RuntimeError("frequency adaptation proposal set is incomplete")

        for node, count in zip(nodes, new_counts):
            G.nodes[node]["stable_count"] = count
        for node in eligible_nodes:
            set_vf(G, node, proposals[node])
    except BaseException:
        _restore_transaction(G, graph_snapshot, node_snapshots)
        raise


def adapt_vf_by_coherence(
    G: TNFRGraph,
    n_jobs: int | None = None,
) -> None:
    """Compatibility wrapper for the structural-stability adaptation gate.

    The historical name does not mean that this function computes C(t).
    """

    adapt_vf_after_structural_stability(G, n_jobs=n_jobs)
