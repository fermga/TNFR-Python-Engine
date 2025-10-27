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
  operators, manual wall-clock measurements per loop, and the full list of
  profiling rows sorted by the requested key (``cumtime`` or ``tottime``).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import pstats
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.dynamics import _prepare_dnfr_data, default_compute_delta_nfr
from tnfr.dynamics.dnfr import _build_neighbor_sums_common, _compute_dnfr_common
from tnfr.metrics.sense_index import compute_Si
import tnfr.utils as tnfr_utils

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


def _format_manual_timings(
    timings: Mapping[str, float],
    *,
    loops: int,
) -> dict[str, dict[str, float]]:
    """Expose total and per-loop wall-clock timings for manual measurements."""

    if loops <= 0:
        loops = 1
    return {
        name: {"total": total, "per_loop": total / loops}
        for name, total in timings.items()
    }


def _dump_profile_outputs(
    profile: cProfile.Profile,
    *,
    base_path: Path,
    mode: str,
    loops: int,
    timings: Mapping[str, float],
    metadata: Mapping[str, Any],
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

    report = {
        "mode": mode,
        "loops": loops,
        "metadata": dict(metadata),
        "manual_timings": _format_manual_timings(timings, loops=loops),
        "target_functions": _extract_target_stats(stats),
        "rows": rows,
    }
    base_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )


def _run_pipeline(
    *,
    graph: nx.Graph,
    vectorized: bool,
    loops: int,
) -> tuple[cProfile.Profile, dict[str, float], dict[str, Any]]:
    """Execute the Sense Index + ΔNFR pipeline under ``vectorized`` conditions."""

    timings = {
        "compute_Si": 0.0,
        "_prepare_dnfr_data": 0.0,
        "_compute_dnfr_common": 0.0,
        "default_compute_delta_nfr": 0.0,
    }

    metadata: dict[str, Any] = {"vectorized": vectorized}

    if not vectorized:
        graph.graph["vectorized_dnfr"] = False
    else:
        graph.graph.pop("vectorized_dnfr", None)

    profile = cProfile.Profile()

    with _numpy_override(vectorized and tnfr_utils.get_numpy() is not None):
        np_module = tnfr_utils.get_numpy()
        metadata["numpy_available"] = bool(np_module)
        metadata["dnfr_vectorized"] = bool(vectorized and np_module is not None)

        # Warm caches for steady-state measurements.
        _invalidate_trig_cache(graph)
        compute_Si(graph, inplace=True)
        default_compute_delta_nfr(graph)

        profile.enable()
        try:
            for _ in range(loops):
                _invalidate_trig_cache(graph)

                start = perf_counter()
                compute_Si(graph, inplace=True)
                timings["compute_Si"] += perf_counter() - start

                start = perf_counter()
                data = _prepare_dnfr_data(graph)
                timings["_prepare_dnfr_data"] += perf_counter() - start

                use_numpy = vectorized and tnfr_utils.get_numpy() is not None
                neighbor_stats = _build_neighbor_sums_common(
                    graph,
                    data,
                    use_numpy=use_numpy,
                    n_jobs=None,
                )

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
                        n_jobs=None,
                    )
                timings["_compute_dnfr_common"] += perf_counter() - start

                start = perf_counter()
                default_compute_delta_nfr(graph)
                timings["default_compute_delta_nfr"] += perf_counter() - start
        finally:
            profile.disable()

    metadata.setdefault("numpy_available", False)
    return profile, timings, metadata


def profile_full_pipeline(
    *,
    node_count: int,
    edge_probability: float,
    loops: int,
    seed: int,
    output_dir: Path,
    sort: str,
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

    for label, vectorized in modes:
        graph = _seed_graph(
            node_count=node_count,
            edge_probability=edge_probability,
            seed=seed,
            si_weights=si_weights,
            dnfr_weights=dnfr_weights,
        )

        profile, timings, metadata = _run_pipeline(
            graph=graph,
            vectorized=vectorized,
            loops=loops,
        )

        base = output_dir / f"full_pipeline_{label}"
        _dump_profile_outputs(
            profile,
            base_path=base,
            mode=label,
            loops=loops,
            timings=timings,
            metadata={
                **metadata,
                "node_count": node_count,
                "edge_probability": edge_probability,
            },
            sort=sort,
        )

        formatted = _format_manual_timings(timings, loops=loops)
        timing_lines = [
            f"  {name}: total={values['total']:.6f}s per_loop={values['per_loop']:.6f}s"
            for name, values in formatted.items()
        ]
        print(
            "Stored {label} profiles at {pstats_path} and {json_path}".format(
                label=label,
                pstats_path=base.with_suffix(".pstats"),
                json_path=base.with_suffix(".json"),
            )
        )
        print("Manual wall-clock timings:")
        for line in timing_lines:
            print(line)


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

    args = parser.parse_args(argv)
    profile_full_pipeline(
        node_count=args.nodes,
        edge_probability=args.edge_probability,
        loops=args.loops,
        seed=args.seed,
        output_dir=args.output_dir,
        sort=args.sort,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
