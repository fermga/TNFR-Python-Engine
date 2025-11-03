"""Profile the full ΔNFR pipeline alongside Sense Index updates.

This benchmark runs a realistic TNFR graph through the structural telemetry
pipeline that feeds coherence monitoring (`compute_Si`) and ΔNFR updates
(`default_compute_delta_nfr`).  It contrasts two execution modes:

* **Vectorised** – NumPy is available and both Sense Index and ΔNFR operators
  rely on dense array kernels.
* **Fallback** – NumPy is disabled and the pure-Python paths are exercised.

Each mode executes the following operators in sequence for a configurable
number of loops:

``compute_Si`` → ``_prepare_dnfr_data`` → ``_compute_dnfr_common`` →
``default_compute_delta_nfr``

The workload mimics the setup used in the performance regression tests.  For
every run the script collects:

* ``.pstats`` dumps that can be inspected with :mod:`pstats` or visualised in
  tools such as Snakeviz.
* Structured JSON summaries including cumulative timings for the core
  operators, manual wall-clock measurements per loop, ΔNFR stage profiles
  (``dnfr_cache_rebuild``, ``dnfr_neighbor_accumulation``,
  ``dnfr_neighbor_means``, ``dnfr_gradient_assembly`` and
  ``dnfr_inplace_write``) for both the manual and default hooks, and the full
  list of profiling rows sorted by the requested key (``cumtime`` or
  ``tottime``).
"""

from __future__ import annotations

import argparse
import cProfile
import math
import pstats
from collections.abc import Iterable
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import _prepare_dnfr_data, default_compute_delta_nfr
from tnfr.dynamics.dnfr import (
    _DNFR_APPROX_BYTES_PER_EDGE,
    _build_neighbor_sums_common,
    _compute_dnfr_common,
    _resolve_parallel_jobs,
)
from tnfr.metrics.sense_index import (
    _SI_APPROX_BYTES_PER_NODE,
    _coerce_jobs as _coerce_si_jobs,
    compute_Si,
)
import tnfr.utils as tnfr_utils
from tnfr.cli.utils import _parse_cli_variants
from tnfr.utils.chunks import resolve_chunk_size
from tnfr.utils import json_dumps
from tnfr.utils.graph import get_graph

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

_TRIG_CACHE_KEYS = ("_cos_th", "_sin_th", "_thetas", "_trig_cache")

_TARGET_FUNCTIONS: Mapping[str, tuple[str, str]] = {
    "tnfr.metrics.sense_index.compute_Si": ("tnfr/metrics/sense_index.py", "compute_Si"),
    "tnfr.dynamics.dnfr._prepare_dnfr_data": ("tnfr/dynamics/dnfr.py", "_prepare_dnfr_data"),
    "tnfr.dynamics.dnfr._compute_dnfr_common": ("tnfr/dynamics/dnfr.py", "_compute_dnfr_common"),
    "tnfr.dynamics.dnfr.default_compute_delta_nfr": (
        "tnfr/dynamics/dnfr.py",
        "default_compute_delta_nfr",
    ),
}
def _format_config_value(value: int | None) -> str:
    """Human-friendly rendering for configuration values."""

    return "auto" if value is None else str(value)


def _format_suffix_value(value: int | None) -> str:
    """Make configuration values safe for file-name suffixes."""

    if value is None:
        return "auto"
    if value < 0:
        return f"m{abs(value)}"
    return str(value)


def _build_config_suffix(
    label: str,
    *,
    si_chunk_size: int | None,
    dnfr_chunk_size: int | None,
    si_workers: int | None,
    dnfr_workers: int | None,
) -> str:
    """Return a deterministic suffix that encodes the configuration knobs."""

    return "_".join(
        (
            label,
            f"si{_format_suffix_value(si_chunk_size)}",
            f"dn{_format_suffix_value(dnfr_chunk_size)}",
            f"siw{_format_suffix_value(si_workers)}",
            f"dnw{_format_suffix_value(dnfr_workers)}",
        )
    )


def _describe_configuration(
    *,
    si_chunk_size: int | None,
    dnfr_chunk_size: int | None,
    si_workers: int | None,
    dnfr_workers: int | None,
) -> str:
    """Return a concise textual summary for console logs."""

    parts = (
        f"SI_CHUNK_SIZE={_format_config_value(si_chunk_size)}",
        f"DNFR_CHUNK_SIZE={_format_config_value(dnfr_chunk_size)}",
        f"SI_N_JOBS={_format_config_value(si_workers)}",
        f"DNFR_N_JOBS={_format_config_value(dnfr_workers)}",
    )
    return ", ".join(parts)


def _apply_configuration(
    graph: nx.Graph,
    *,
    si_chunk_size: int | None,
    dnfr_chunk_size: int | None,
    si_workers: int | None,
    dnfr_workers: int | None,
) -> None:
    """Seed graph-level knobs controlling batching and parallel workers."""

    def _set(key: str, value: int | None) -> None:
        if value is None:
            graph.graph.pop(key, None)
        else:
            graph.graph[key] = value

    _set("SI_CHUNK_SIZE", si_chunk_size)
    _set("DNFR_CHUNK_SIZE", dnfr_chunk_size)
    _set("SI_N_JOBS", si_workers)
    _set("DNFR_N_JOBS", dnfr_workers)


def _seed_graph(
    *,
    node_count: int,
    edge_probability: float,
    seed: int,
    si_weights: Iterable[tuple[str, float]],
    dnfr_weights: Mapping[str, float],
) -> nx.Graph:
    """Build a deterministic TNFR graph mirroring the performance tests."""

    graph = nx.gnp_random_graph(node_count, edge_probability, seed=seed)
    graph.graph["SI_WEIGHTS"] = dict(si_weights)
    graph.graph["DNFR_WEIGHTS"] = dict(dnfr_weights)

    for node in graph.nodes:
        set_attr(graph.nodes[node], ALIAS_THETA, (node % 48) * (math.pi / 24))
        set_attr(graph.nodes[node], ALIAS_EPI, 0.08 + 0.015 * ((node * 5) % 37))
        set_attr(graph.nodes[node], ALIAS_VF, 0.12 + 0.01 * (node % 29))
        set_attr(graph.nodes[node], ALIAS_DNFR, 0.05 + 0.02 * ((node * 7) % 31))

    return graph


def _invalidate_trig_cache(graph: nx.Graph) -> None:
    """Ensure Sense Index recomputes cached θ trigonometric data."""

    graph.graph["_trig_version"] = graph.graph.get("_trig_version", 0) + 1
    for key in _TRIG_CACHE_KEYS:
        graph.graph.pop(key, None)


@contextmanager
def _numpy_override(enabled: bool):
    """Temporarily toggle NumPy availability across Sense Index and ΔNFR paths."""

    if enabled:
        yield
        return

    from tnfr.metrics import sense_index, trig, trig_cache
    from tnfr.dynamics import dnfr

    modules = (
        (sense_index, "get_numpy"),
        (trig, "get_numpy"),
        (trig_cache, "get_numpy"),
        (dnfr, "get_numpy"),
        (tnfr_utils, "get_numpy"),
    )
    originals = {(module, attr): getattr(module, attr) for module, attr in modules}

    try:
        for module, attr in modules:
            setattr(module, attr, lambda: None)
        yield
    finally:
        for module, attr in modules:
            setattr(module, attr, originals[(module, attr)])


def _extract_target_stats(stats: pstats.Stats) -> dict[str, dict[str, float | int]]:
    """Collect cumulative data for the primary operators."""

    summary: dict[str, dict[str, float | int]] = {}
    for (filename, _lineno, func), (cc, nc, tt, ct, _callers) in stats.stats.items():
        filename_norm = filename.replace("\\", "/")
        for label, (path_fragment, expected_name) in _TARGET_FUNCTIONS.items():
            if func != expected_name:
                continue
            if path_fragment not in filename_norm:
                continue
            summary[label] = {
                "callcount": cc,
                "reccallcount": nc,
                "totaltime": tt,
                "cumtime": ct,
            }
    return summary


def _format_operator_timings(
    timings: Mapping[str, float],
    *,
    loops: int,
) -> dict[str, dict[str, float]]:
    """Expose total and per-loop wall-clock timings for each operator."""

    if loops <= 0:
        loops = 1
    return {
        name: {"total": total, "per_loop": total / loops}
        for name, total in timings.items()
    }


def _extract_cache_metrics(graph: nx.Graph) -> dict[str, Any]:
    """Extract cache hit/miss/eviction statistics from graph cache managers.

    Returns
    -------
    dict[str, Any]
        Cache metrics including aggregate stats and per-cache breakdown.
    """
    graph_dict = get_graph(graph)
    metrics_result: dict[str, Any] = {
        "aggregate": {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "hit_rate": 0.0,
        },
        "by_cache": {},
    }

    # Check for CacheManager on graph
    cache_manager = graph_dict.get("_tnfr_cache_manager")
    if cache_manager is not None and hasattr(cache_manager, "aggregate_metrics"):
        try:
            aggregate = cache_manager.aggregate_metrics()
            metrics_result["aggregate"] = {
                "hits": aggregate.hits,
                "misses": aggregate.misses,
                "evictions": aggregate.evictions,
                "hit_rate": (
                    aggregate.hits / (aggregate.hits + aggregate.misses)
                    if (aggregate.hits + aggregate.misses) > 0
                    else 0.0
                ),
            }

            # Collect per-cache metrics
            if hasattr(cache_manager, "iter_metrics"):
                for name, stats in cache_manager.iter_metrics():
                    total = stats.hits + stats.misses
                    metrics_result["by_cache"][name] = {
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "evictions": stats.evictions,
                        "hit_rate": stats.hits / total if total > 0 else 0.0,
                    }
        except Exception:
            pass  # Silently ignore metrics collection errors

    # Check for EdgeCacheManager
    edge_cache_manager = graph_dict.get("_edge_cache_manager")
    if edge_cache_manager is not None and hasattr(edge_cache_manager, "_manager"):
        try:
            edge_manager = edge_cache_manager._manager
            if hasattr(edge_manager, "aggregate_metrics"):
                edge_aggregate = edge_manager.aggregate_metrics()
                # Merge with aggregate if we have data
                if metrics_result["aggregate"]["hits"] == 0:
                    metrics_result["aggregate"] = {
                        "hits": edge_aggregate.hits,
                        "misses": edge_aggregate.misses,
                        "evictions": edge_aggregate.evictions,
                        "hit_rate": (
                            edge_aggregate.hits / (edge_aggregate.hits + edge_aggregate.misses)
                            if (edge_aggregate.hits + edge_aggregate.misses) > 0
                            else 0.0
                        ),
                    }
        except Exception:
            pass  # Silently ignore metrics collection errors

    return metrics_result


def _dump_profile_outputs(
    profile: cProfile.Profile,
    *,
    base_path: Path,
    mode: str,
    loops: int,
    timings: Mapping[str, float],
    operator_timings: Mapping[str, Mapping[str, float]],
    si_breakdown: Mapping[str, Any],
    dnfr_breakdown: Mapping[str, Any],
    metadata: Mapping[str, Any],
    configuration: Mapping[str, Any],
    cache_metrics: Mapping[str, Any],
    sort: str,
) -> None:
    """Persist profiling artefacts in ``.pstats`` and JSON formats."""

    stats = pstats.Stats(profile)
    stats.sort_stats(sort)
    stats.dump_stats(str(base_path.with_suffix(".pstats")))

    rows = []
    sort_key = {"tottime": "totaltime", "time": "totaltime"}.get(sort, "cumtime")
    for (filename, lineno, func), (cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append(
            {
                "function": func,
                "file": filename,
                "line": lineno,
                "callcount": cc,
                "reccallcount": nc,
                "totaltime": tt,
                "cumtime": ct,
            }
        )
    rows.sort(key=lambda entry: entry[sort_key], reverse=True)

    operator_totals = {name: float(total) for name, total in timings.items()}
    operator_timings_dict = {
        name: {"total": float(values["total"]), "per_loop": float(values["per_loop"])}
        for name, values in operator_timings.items()
    }
    si_totals = {
        name: float(total)
        for name, total in si_breakdown.get("totals", {}).items()
    }
    si_per_loop = {
        name: float(value)
        for name, value in si_breakdown.get("per_loop", {}).items()
    }
    si_path_counts = {
        name: int(count)
        for name, count in si_breakdown.get("path_counts", {}).items()
    }

    dnfr_sections = {}
    for label, section in dnfr_breakdown.items():
        totals = section.get("totals", {})
        per_loop = section.get("per_loop", {})
        paths = section.get("path_counts", {})
        dnfr_sections[label] = {
            "totals": {name: float(value) for name, value in totals.items()},
            "per_loop": {name: float(value) for name, value in per_loop.items()},
            "path_counts": {name: int(count) for name, count in paths.items()},
        }

    report = {
        "mode": mode,
        "loops": loops,
        "configuration": dict(configuration),
        "metadata": dict(metadata),
        "operator_totals": operator_totals,
        "operator_timings": operator_timings_dict,
        "manual_timings": operator_timings_dict,
        "compute_Si_breakdown": {
            "totals": si_totals,
            "per_loop": si_per_loop,
            "path_counts": si_path_counts,
        },
        "dnfr_breakdown": dnfr_sections,
        "cache_metrics": dict(cache_metrics),
        "target_functions": _extract_target_stats(stats),
        "rows": rows,
    }
    base_path.with_suffix(".json").write_text(
        json_dumps(report, indent=2, ensure_ascii=False)
    )


def _run_pipeline(
    *,
    graph: nx.Graph,
    vectorized: bool,
    loops: int,
    dnfr_workers: int | None,
    configuration: Mapping[str, Any],
) -> tuple[
    cProfile.Profile,
    dict[str, float],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Execute the Sense Index + ΔNFR pipeline under ``vectorized`` conditions."""

    timings = {
        "compute_Si": 0.0,
        "_prepare_dnfr_data": 0.0,
        "_compute_dnfr_common": 0.0,
        "default_compute_delta_nfr": 0.0,
    }

    si_sub_totals = {
        "cache_rebuild": 0.0,
        "neighbor_phase_mean_bulk": 0.0,
        "normalize_clamp": 0.0,
        "inplace_write": 0.0,
    }
    si_path_counts: dict[str, int] = {}

    dnfr_manual_totals = {
        "dnfr_cache_rebuild": 0.0,
        "dnfr_neighbor_accumulation": 0.0,
        "dnfr_neighbor_means": 0.0,
        "dnfr_gradient_assembly": 0.0,
        "dnfr_inplace_write": 0.0,
    }
    dnfr_manual_paths: dict[str, int] = {}

    dnfr_default_totals = {
        "dnfr_cache_rebuild": 0.0,
        "dnfr_neighbor_accumulation": 0.0,
        "dnfr_neighbor_means": 0.0,
        "dnfr_gradient_assembly": 0.0,
        "dnfr_inplace_write": 0.0,
    }
    dnfr_default_paths: dict[str, int] = {}

    metadata: dict[str, Any] = {
        "vectorized": vectorized,
        "configuration_label": configuration.get("label"),
        "configuration_index": configuration.get("index"),
        "si_chunk_size_requested": configuration.get("si_chunk_size"),
        "dnfr_chunk_size_requested": configuration.get("dnfr_chunk_size"),
        "si_workers_requested": configuration.get("si_workers"),
        "dnfr_workers_requested": configuration.get("dnfr_workers"),
    }

    node_total = graph.number_of_nodes()
    edge_total = graph.number_of_edges()

    metadata["resolved_si_chunk_size"] = resolve_chunk_size(
        configuration.get("si_chunk_size"),
        node_total,
        approx_bytes_per_item=_SI_APPROX_BYTES_PER_NODE,
    )
    metadata["resolved_dnfr_chunk_size"] = (
        resolve_chunk_size(
            configuration.get("dnfr_chunk_size"),
            edge_total,
            minimum=1,
            approx_bytes_per_item=_DNFR_APPROX_BYTES_PER_EDGE,
            clamp_to=None,
        )
        if edge_total
        else 0
    )
    metadata["si_workers_effective"] = _coerce_si_jobs(configuration.get("si_workers"))
    metadata["dnfr_workers_effective"] = _resolve_parallel_jobs(dnfr_workers, node_total)
    metadata["node_count"] = node_total
    metadata["edge_count"] = edge_total

    if not vectorized:
        graph.graph["vectorized_dnfr"] = False
    else:
        graph.graph.pop("vectorized_dnfr", None)

    profile = cProfile.Profile()
    resolved_neighbor_chunk_size: int | None = None

    with _numpy_override(vectorized and tnfr_utils.get_numpy() is not None):
        np_module = tnfr_utils.get_numpy()
        metadata["numpy_available"] = bool(np_module)
        metadata["dnfr_vectorized"] = bool(vectorized and np_module is not None)

        # Warm caches for steady-state measurements.
        _invalidate_trig_cache(graph)
        compute_Si(graph, inplace=True)
        default_compute_delta_nfr(graph, n_jobs=dnfr_workers)

        profile.enable()
        try:
            for _ in range(loops):
                _invalidate_trig_cache(graph)

                si_profile: dict[str, Any] = {}
                start = perf_counter()
                compute_Si(graph, inplace=True, profile=si_profile)
                elapsed = perf_counter() - start
                timings["compute_Si"] += elapsed
                for key in si_sub_totals:
                    value = si_profile.get(key)
                    if isinstance(value, (int, float)):
                        si_sub_totals[key] += float(value)
                path = si_profile.get("path")
                if isinstance(path, str):
                    si_path_counts[path] = si_path_counts.get(path, 0) + 1

                dnfr_profile_stage: dict[str, float | str] = {
                    key: 0.0 for key in dnfr_manual_totals
                }
                start = perf_counter()
                data = _prepare_dnfr_data(graph, profile=dnfr_profile_stage)
                timings["_prepare_dnfr_data"] += perf_counter() - start

                use_numpy = vectorized and tnfr_utils.get_numpy() is not None
                neighbor_start = perf_counter()
                neighbor_stats = _build_neighbor_sums_common(
                    graph,
                    data,
                    use_numpy=use_numpy,
                    n_jobs=dnfr_workers,
                )
                neighbor_elapsed = perf_counter() - neighbor_start
                dnfr_profile_stage.setdefault("dnfr_neighbor_accumulation", 0.0)
                dnfr_profile_stage["dnfr_neighbor_accumulation"] = float(
                    dnfr_profile_stage.get("dnfr_neighbor_accumulation", 0.0)
                ) + neighbor_elapsed

                if resolved_neighbor_chunk_size is None:
                    resolved_neighbor_chunk_size = data.get("neighbor_chunk_size")
                    metadata["dnfr_neighbor_chunk_hint"] = data.get("neighbor_chunk_hint")

                start = perf_counter()
                if neighbor_stats is not None:
                    x, y, epi_sum, vf_sum, count, deg_sum, degs = neighbor_stats
                    _compute_dnfr_common(
                        graph,
                        data,
                        x=x,
                        y=y,
                        epi_sum=epi_sum,
                        vf_sum=vf_sum,
                        count=count,
                        deg_sum=deg_sum,
                        degs=degs,
                        n_jobs=dnfr_workers,
                        profile=dnfr_profile_stage,
                    )
                timings["_compute_dnfr_common"] += perf_counter() - start

                dnfr_profile_stage.setdefault(
                    "dnfr_path", "vectorized" if use_numpy else "fallback"
                )
                for key in dnfr_manual_totals:
                    value = dnfr_profile_stage.get(key)
                    if isinstance(value, (int, float)):
                        dnfr_manual_totals[key] += float(value)
                path = dnfr_profile_stage.get("dnfr_path")
                if isinstance(path, str):
                    dnfr_manual_paths[path] = dnfr_manual_paths.get(path, 0) + 1

                start = perf_counter()
                dnfr_default_profile: dict[str, float | str] = {}
                default_compute_delta_nfr(
                    graph, n_jobs=dnfr_workers, profile=dnfr_default_profile
                )
                timings["default_compute_delta_nfr"] += perf_counter() - start

                for key in dnfr_default_totals:
                    value = dnfr_default_profile.get(key)
                    if isinstance(value, (int, float)):
                        dnfr_default_totals[key] += float(value)
                default_path = dnfr_default_profile.get("dnfr_path")
                if isinstance(default_path, str):
                    dnfr_default_paths[default_path] = (
                        dnfr_default_paths.get(default_path, 0) + 1
                    )
        finally:
            profile.disable()

    metadata.setdefault("numpy_available", False)
    if resolved_neighbor_chunk_size is not None:
        metadata["dnfr_neighbor_chunk_size"] = resolved_neighbor_chunk_size
    loops_for_avg = loops if loops else 1
    metadata["si_vectorized_calls"] = si_path_counts.get("vectorized", 0)
    metadata["si_fallback_calls"] = si_path_counts.get("fallback", 0)
    si_details = {
        "totals": {name: float(total) for name, total in si_sub_totals.items()},
        "per_loop": {
            name: float(total / loops_for_avg) if loops else 0.0
            for name, total in si_sub_totals.items()
        },
        "path_counts": dict(sorted(si_path_counts.items())),
    }

    dnfr_manual_details = {
        "totals": {name: float(total) for name, total in dnfr_manual_totals.items()},
        "per_loop": {
            name: float(total / loops_for_avg) if loops else 0.0
            for name, total in dnfr_manual_totals.items()
        },
        "path_counts": dict(sorted(dnfr_manual_paths.items())),
    }

    dnfr_default_details = {
        "totals": {name: float(total) for name, total in dnfr_default_totals.items()},
        "per_loop": {
            name: float(total / loops_for_avg) if loops else 0.0
            for name, total in dnfr_default_totals.items()
        },
        "path_counts": dict(sorted(dnfr_default_paths.items())),
    }

    dnfr_details = {"manual": dnfr_manual_details, "default": dnfr_default_details}

    # Collect cache metrics from graph cache manager
    cache_metrics = _extract_cache_metrics(graph)

    return profile, timings, metadata, si_details, dnfr_details, cache_metrics


def profile_full_pipeline(
    *,
    node_count: int,
    edge_probability: float,
    loops: int,
    seed: int,
    output_dir: Path,
    sort: str,
    si_chunk_sizes: Iterable[int | None] = (None,),
    dnfr_chunk_sizes: Iterable[int | None] = (None,),
    si_workers: Iterable[int | None] = (None,),
    dnfr_workers: Iterable[int | None] = (None,),
) -> None:
    """Profile the Sense Index + ΔNFR pipeline under vectorised and fallback runs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    si_weights = (("alpha", 0.35), ("beta", 0.40), ("gamma", 0.25))
    dnfr_weights = {
        "phase": 0.35,
        "epi": 0.25,
        "vf": 0.25,
        "topo": 0.15,
    }

    modes: list[tuple[str, bool]] = [("vectorized", True)]
    if tnfr_utils.get_numpy() is None:
        print("NumPy is unavailable; skipping the vectorized run.")
        modes = []
    modes.append(("fallback", False))

    si_chunk_options = tuple(si_chunk_sizes) if si_chunk_sizes is not None else (None,)
    dnfr_chunk_options = tuple(dnfr_chunk_sizes) if dnfr_chunk_sizes is not None else (None,)
    si_worker_options = tuple(si_workers) if si_workers is not None else (None,)
    dnfr_worker_options = tuple(dnfr_workers) if dnfr_workers is not None else (None,)

    configurations = list(
        product(si_chunk_options, dnfr_chunk_options, si_worker_options, dnfr_worker_options)
    )
    if not configurations:
        configurations = [(None, None, None, None)]

    for config_index, (si_chunk, dnfr_chunk, si_jobs, dnfr_jobs) in enumerate(
        configurations, start=1
    ):
        config_label = f"cfg{config_index:02d}"
        config_description = _describe_configuration(
            si_chunk_size=si_chunk,
            dnfr_chunk_size=dnfr_chunk,
            si_workers=si_jobs,
            dnfr_workers=dnfr_jobs,
        )
        print(f"\n== Configuration {config_label}: {config_description} ==")

        config_suffix = _build_config_suffix(
            config_label,
            si_chunk_size=si_chunk,
            dnfr_chunk_size=dnfr_chunk,
            si_workers=si_jobs,
            dnfr_workers=dnfr_jobs,
        )
        configuration = {
            "index": config_index,
            "label": config_label,
            "si_chunk_size": si_chunk,
            "dnfr_chunk_size": dnfr_chunk,
            "si_workers": si_jobs,
            "dnfr_workers": dnfr_jobs,
            "description": config_description,
        }

        for label, vectorized in modes:
            graph = _seed_graph(
                node_count=node_count,
                edge_probability=edge_probability,
                seed=seed,
                si_weights=si_weights,
                dnfr_weights=dnfr_weights,
            )

            _apply_configuration(
                graph,
                si_chunk_size=si_chunk,
                dnfr_chunk_size=dnfr_chunk,
                si_workers=si_jobs,
                dnfr_workers=dnfr_jobs,
            )

            profile, timings, metadata, si_details, dnfr_details, cache_metrics = _run_pipeline(
                graph=graph,
                vectorized=vectorized,
                loops=loops,
                dnfr_workers=dnfr_jobs,
                configuration=configuration,
            )

            metadata = {
                **metadata,
                "edge_probability": edge_probability,
            }

            operator_timings = _format_operator_timings(timings, loops=loops)
            base = output_dir / f"full_pipeline_{label}_{config_suffix}"
            _dump_profile_outputs(
                profile,
                base_path=base,
                mode=label,
                loops=loops,
                timings=timings,
                operator_timings=operator_timings,
                si_breakdown=si_details,
                dnfr_breakdown=dnfr_details,
                metadata=metadata,
                configuration=configuration,
                cache_metrics=cache_metrics,
                sort=sort,
            )

            timing_lines = [
                f"  {name}: total={values['total']:.6f}s per_loop={values['per_loop']:.6f}s"
                for name, values in operator_timings.items()
            ]
            si_lines = [
                f"  {name}: total={si_details['totals'].get(name, 0.0):.6f}s "
                f"per_loop={si_details['per_loop'].get(name, 0.0):.6f}s"
                for name in sorted(si_details.get("totals", {}))
            ]
            path_summary = ", ".join(
                f"{key}={value}" for key, value in si_details.get("path_counts", {}).items()
            ) or "none"
            print(
                "Stored {label} profiles for {config_label} at {pstats_path} and {json_path}".format(
                    label=label,
                    config_label=config_label,
                    pstats_path=base.with_suffix(".pstats"),
                    json_path=base.with_suffix(".json"),
                )
            )
            print("Per-operator wall-clock timings:")
            for line in timing_lines:
                print(line)
            print("compute_Si breakdown:")
            for line in si_lines:
                print(line)
            print(f"  paths: {path_summary}")

            manual_dnfr = dnfr_details.get("manual", {})
            default_dnfr = dnfr_details.get("default", {})

            def _format_dnfr_lines(section: Mapping[str, Any]) -> list[str]:
                totals = section.get("totals", {})
                per_loop = section.get("per_loop", {})
                return [
                    f"  {name}: total={totals.get(name, 0.0):.6f}s "
                    f"per_loop={per_loop.get(name, 0.0):.6f}s"
                    for name in sorted(totals)
                ]

            manual_lines = _format_dnfr_lines(manual_dnfr)
            default_lines = _format_dnfr_lines(default_dnfr)
            manual_paths = ", ".join(
                f"{key}={value}" for key, value in manual_dnfr.get("path_counts", {}).items()
            ) or "none"
            default_paths = ", ".join(
                f"{key}={value}" for key, value in default_dnfr.get("path_counts", {}).items()
            ) or "none"

            print("ΔNFR manual stages breakdown:")
            for line in manual_lines:
                print(line)
            print(f"  paths: {manual_paths}")

            print("ΔNFR default hook breakdown:")
            for line in default_lines:
                print(line)
            print(f"  paths: {default_paths}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the full pipeline profiler."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile compute_Si, ΔNFR preparation, and default_compute_delta_nfr "
            "under vectorized and fallback execution modes."
        )
    )
    parser.add_argument("--nodes", type=int, default=240, help="Number of nodes")
    parser.add_argument(
        "--edge-probability",
        type=float,
        default=0.32,
        help="Probability used by the Erdos-Renyi generator",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=5,
        help="How many times to execute the pipeline inside the profiler",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for graph generation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles"),
        help="Directory where profiling artefacts will be written",
    )
    parser.add_argument(
        "--sort",
        choices=("cumtime", "tottime"),
        default="cumtime",
        help="Sort order applied when exporting profiling rows",
    )
    parser.add_argument(
        "--si-chunk-sizes",
        nargs="+",
        metavar="SIZE",
        help=(
            "One or more chunk sizes applied to G.graph['SI_CHUNK_SIZE']; "
            "use 'auto' to rely on engine heuristics."
        ),
    )
    parser.add_argument(
        "--dnfr-chunk-sizes",
        nargs="+",
        metavar="SIZE",
        help=(
            "One or more chunk sizes applied to G.graph['DNFR_CHUNK_SIZE']; "
            "use 'auto' to rely on engine heuristics."
        ),
    )
    parser.add_argument(
        "--si-workers",
        nargs="+",
        metavar="COUNT",
        help=(
            "Worker counts propagated to G.graph['SI_N_JOBS']; "
            "use 'auto' to keep deterministic single-process execution."
        ),
    )
    parser.add_argument(
        "--dnfr-workers",
        nargs="+",
        metavar="COUNT",
        help=(
            "Worker counts propagated to G.graph['DNFR_N_JOBS'] and DNFR helpers; "
            "use 'auto' to preserve the default behaviour."
        ),
    )

    args = parser.parse_args(argv)
    try:
        si_chunk_sizes = _parse_cli_variants(args.si_chunk_sizes)
        dnfr_chunk_sizes = _parse_cli_variants(args.dnfr_chunk_sizes)
        si_workers = _parse_cli_variants(args.si_workers)
        dnfr_workers = _parse_cli_variants(args.dnfr_workers)
    except ValueError as exc:
        parser.error(str(exc))

    profile_full_pipeline(
        node_count=args.nodes,
        edge_probability=args.edge_probability,
        loops=args.loops,
        seed=args.seed,
        output_dir=args.output_dir,
        sort=args.sort,
        si_chunk_sizes=si_chunk_sizes,
        dnfr_chunk_sizes=dnfr_chunk_sizes,
        si_workers=si_workers,
        dnfr_workers=dnfr_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
