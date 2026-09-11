"""Measure node lookup cost and exact seeded jitter trajectories.

Run from the repository root. ``--wheel`` selects an immutable baseline ahead
of the working source, without installing or changing either implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 2000])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("sizes and repeats must be positive integers")

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "src"))
    if args.wheel is not None:
        sys.path.insert(0, str(args.wheel.resolve()))

    import networkx as nx
    import tnfr.utils.cache as cache
    from tnfr.node import NodeNX
    from tnfr.operators.jitter import random_jitter

    print(json.dumps({"source": cache.__file__, "python": sys.version.split()[0],
                      "networkx": nx.__version__, "seed": 7, "amplitude": 0.1,
                      "repeats": args.repeats,
                      "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest()
                      if args.wheel is not None else None}), flush=True)
    for size in args.sizes:
        graph = nx.empty_graph(size)
        graph.graph["RANDOM_SEED"] = 7
        nodes = [NodeNX(graph, node) for node in graph]
        for node in nodes:
            random_jitter(node, 0.1)

        elapsed = []
        digest_misses = []
        digest_hits = []
        trajectory = hashlib.sha256()
        for repeat in range(args.repeats):
            before = cache._node_repr_digest.cache_info()
            start = time.perf_counter()
            draws = [random_jitter(node, 0.1) for node in nodes]
            elapsed.append(time.perf_counter() - start)
            after = cache._node_repr_digest.cache_info()
            digest_misses.append(after.misses - before.misses)
            digest_hits.append(after.hits - before.hits)
            for draw in draws:
                trajectory.update(struct.pack(">d", draw))
            print(json.dumps({"nodes": size, "repeat": repeat + 1,
                              "seconds": elapsed[-1], "digest_misses": digest_misses[-1],
                              "digest_hits": digest_hits[-1]}), flush=True)
        print(json.dumps({"nodes": size, "median_seconds": statistics.median(elapsed),
                          "trajectory_sha256": trajectory.hexdigest(),
                          "draws_per_node": graph.nodes[0]["_rng_jitter_progress"]["draws"]}),
              flush=True)


if __name__ == "__main__":
    main()
