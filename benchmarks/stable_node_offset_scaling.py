"""Compare explicit offset batching with unchanged canonical execution.

Run from the repository root. The optional wheel is inserted before src,
without installing it. Timings include scope entry/exit and validation counters;
graph initialization and telemetry serialization are excluded.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--scoped", action="store_true")
    parser.add_argument("--workload", choices=["jitter", "sdk"], default="sdk")
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 2000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 3 for size in args.sizes):
        parser.error("repeats must be positive and sizes at least 3")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    if args.wheel:
        sys.path.insert(0, str(args.wheel.resolve()))

    import networkx as nx
    from tnfr.alias import get_attr
    from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR
    from tnfr.initialization import init_node_attrs
    from tnfr.metrics.common import compute_coherence
    from tnfr.metrics.sense_index import compute_Si
    from tnfr.node import NodeNX
    from tnfr.operators.jitter import random_jitter
    from tnfr.sdk.simple import _run_network_sequence
    import tnfr.utils.cache as cache

    print(json.dumps({"source": cache.__file__, "python": sys.version.split()[0],
                      "networkx": nx.__version__, "seed": 7, "scoped": args.scoped,
                      "workload": args.workload, "repeats": args.repeats,
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest()
                      if args.wheel else None}), flush=True)
    for size in args.sizes:
        elapsed, hashes, compared = [], [], []
        for repeat in range(args.repeats):
            graph = nx.cycle_graph(size)
            graph.graph.update(RANDOM_SEED=7, INIT_RANDOM_PHASE=False, INIT_EPI_VALUE=0.0,
                               INIT_VF_MODE="uniform", INIT_VF_MIN=0.4, INIT_VF_MAX=0.7,
                               OZ_NOISE_MODE=True, OZ_SIGMA=0.1, GLYPH_HYSTERESIS_WINDOW=20)
            init_node_attrs(graph)
            adapters = [NodeNX.from_graph(graph, node) for node in graph]
            cache.ensure_node_offset_map(graph)
            comparisons = 0
            original = cache._same_node_snapshot

            def counted(previous, current):
                nonlocal comparisons
                comparisons += len(current)
                return original(previous, current)

            records = []

            def record(operator):
                sense = compute_Si(graph, inplace=False)
                records.append((operator, compute_coherence(graph), [
                    (node, *(float(get_attr(data, alias, 0.0))
                             for alias in (ALIAS_EPI, ALIAS_VF, ALIAS_THETA, ALIAS_DNFR)),
                     float(sense[node]), tuple(data.get("glyph_history", [])))
                    for node, data in graph.nodes(data=True)
                ]))

            cache._same_node_snapshot = counted
            try:
                start = time.perf_counter()
                with cache.stable_node_offsets(graph) if args.scoped else nullcontext():
                    if args.workload == "sdk":
                        _run_network_sequence(
                            graph, ["emission", "coherence", "dissonance", "coherence", "silence"],
                            validate=True, on_step=record,
                        )
                    else:
                        records = [random_jitter(node, 0.1) for node in adapters]
                seconds = time.perf_counter() - start
            finally:
                cache._same_node_snapshot = original
            payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
            digest = hashlib.sha256(payload.encode()).hexdigest()
            elapsed.append(seconds)
            hashes.append(digest)
            compared.append(comparisons)
            print(json.dumps({"nodes": size, "repeat": repeat + 1, "seconds": seconds,
                              "snapshot_elements_compared": comparisons,
                              "trajectory_sha256": digest}), flush=True)
        assert len(set(hashes)) == 1, "identical seeds did not reproduce the trajectory"
        print(json.dumps({"nodes": size, "median_seconds": statistics.median(elapsed),
                          "snapshot_elements_compared": compared,
                          "trajectory_sha256": hashes[0]}), flush=True)


if __name__ == "__main__":
    main()
