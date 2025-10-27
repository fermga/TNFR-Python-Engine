"""Profile Sense Index computations with and without NumPy.

This script builds a deterministic resonance graph and measures how long
``tnfr.metrics.sense_index.compute_Si`` takes under two execution modes:

* **Vectorised** – NumPy is available and the metric uses dense arrays.
* **Fallback** – NumPy is disabled to exercise the pure-Python path.

Both runs are captured with :mod:`cProfile` so that we can inspect which
functions dominate the runtime.  The resulting statistics can be exported as
either binary ``.pstats`` dumps (loadable with :mod:`pstats`) or JSON summaries
sorted by the cumulative time spent per function.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import pstats
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics.sense_index import compute_Si

ALIAS_THETA = get_aliases("THETA")
ALIAS_VF = get_aliases("VF")
ALIAS_DNFR = get_aliases("DNFR")

_TRIG_CACHE_KEYS = ("_cos_th", "_sin_th", "_thetas", "_trig_cache")


def _seed_graph(
    *, node_count: int, chord_step: int, si_weights: Iterable[tuple[str, float]]
) -> nx.Graph:
    """Populate a graph with deterministic θ, νf, and ΔNFR assignments."""

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    # Couple every node with its successor and with a fixed chord to mimic the
    # workload used in the performance regression tests.
    graph.add_edges_from(((idx, (idx + 1) % node_count) for idx in range(node_count)))
    graph.add_edges_from(
        ((idx, (idx + chord_step) % node_count) for idx in range(node_count))
    )

    weights = dict(si_weights)
    if weights:
        graph.graph["SI_WEIGHTS"] = weights

    for node in graph.nodes:
        theta = (node % 36) * (math.pi / 18)
        vf = 0.15 + 0.02 * (node % 25)
        dnfr = 0.05 + 0.015 * ((node * 3) % 30)
        set_attr(graph.nodes[node], ALIAS_THETA, theta)
        set_attr(graph.nodes[node], ALIAS_VF, vf)
        set_attr(graph.nodes[node], ALIAS_DNFR, dnfr)

    return graph


def _invalidate_trig_cache(graph: nx.Graph) -> None:
    """Reset cached trigonometric data stored on ``graph``."""

    graph.graph["_trig_version"] = graph.graph.get("_trig_version", 0) + 1
    for key in _TRIG_CACHE_KEYS:
        graph.graph.pop(key, None)


@contextmanager
def _numpy_override(enabled: bool):
    """Temporarily toggle the NumPy backend used by ``compute_Si``."""

    from tnfr.metrics import sense_index

    original = sense_index.get_numpy
    if enabled:
        yield
        return

    sense_index.get_numpy = lambda: None  # type: ignore[assignment]
    try:
        yield
    finally:
        sense_index.get_numpy = original  # type: ignore[assignment]


def _dump_stats(profile: cProfile.Profile, path: Path, *, fmt: str, sort: str) -> None:
    """Persist profiling results using the requested ``fmt``."""

    stats = pstats.Stats(profile)
    stats.sort_stats(sort)

    if fmt == "pstats":
        stats.dump_stats(str(path))
        return

    # ``pstats`` exposes columns matching cProfile documentation.  We keep the
    # relevant aggregates so they can be inspected without custom tooling.
    sort_key = {
        "tottime": "totaltime",
        "time": "totaltime",
        "cumtime": "cumtime",
    }.get(sort, "cumtime")
    rows = []
    for (filename, lineno, func), (cc, nc, tt, ct, callers) in stats.stats.items():
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
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def profile_compute_si(
    *,
    node_count: int,
    chord_step: int,
    loops: int,
    output_dir: Path,
    fmt: str,
    sort: str,
) -> None:
    """Profile vectorised and fallback Sense Index computations."""

    output_dir.mkdir(parents=True, exist_ok=True)
    si_weights = (("alpha", 0.35), ("beta", 0.45), ("gamma", 0.20))

    def run_once(vectorised: bool) -> cProfile.Profile:
        graph = _seed_graph(
            node_count=node_count, chord_step=chord_step, si_weights=si_weights
        )
        # Warm caches so both modes measure steady-state behaviour.
        with _numpy_override(vectorised):
            compute_Si(graph, inplace=False)

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            for _ in range(loops):
                _invalidate_trig_cache(graph)
                with _numpy_override(vectorised):
                    compute_Si(graph, inplace=False)
        finally:
            profiler.disable()
        return profiler

    numpy_profile = run_once(vectorised=True)
    fallback_profile = run_once(vectorised=False)

    suffix = "json" if fmt == "json" else "pstats"
    numpy_path = output_dir / f"compute_Si_numpy.{suffix}"
    fallback_path = output_dir / f"compute_Si_python.{suffix}"

    _dump_stats(numpy_profile, numpy_path, fmt=fmt, sort=sort)
    _dump_stats(fallback_profile, fallback_path, fmt=fmt, sort=sort)

    print(f"Stored NumPy profile at {numpy_path}")
    print(f"Stored Python fallback profile at {fallback_path}")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run the profiler."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile compute_Si with and without NumPy to compare cumulative times."
        )
    )
    parser.add_argument("--nodes", type=int, default=240, help="Number of nodes")
    parser.add_argument(
        "--chord-step",
        type=int,
        default=7,
        help="Chord distance when wiring the deterministic resonance graph",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=5,
        help="How many times to recompute Si during profiling",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiles"),
        help="Directory where profile outputs will be written",
    )
    parser.add_argument(
        "--format",
        choices=("pstats", "json"),
        default="pstats",
        help="Persist results as binary .pstats files or JSON",
    )
    parser.add_argument(
        "--sort",
        choices=("cumtime", "tottime"),
        default="cumtime",
        help="Sort order used when aggregating profiling rows",
    )

    args = parser.parse_args(argv)
    profile_compute_si(
        node_count=args.nodes,
        chord_step=args.chord_step,
        loops=args.loops,
        output_dir=args.output_dir,
        fmt=args.format,
        sort=args.sort,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
