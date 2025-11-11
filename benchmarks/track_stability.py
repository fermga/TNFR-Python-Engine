"""Benchmark vectorized and fallback stability tracking paths."""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any

import networkx as nx

from tnfr.alias import set_attr
from tnfr.constants import get_aliases
from tnfr.metrics import coherence

ALIAS_DNFR = get_aliases("DNFR")
ALIAS_DEPI = get_aliases("DEPI")
ALIAS_SI = get_aliases("SI")
ALIAS_VF = get_aliases("VF")


def _build_graph(node_count: int) -> Any:
    rng = random.Random(42)
    G = nx.gnp_random_graph(node_count, 0.05, seed=123)
    for node in G.nodes:
        dnfr = rng.uniform(-1.0, 1.0)
        depi = rng.uniform(-1.0, 1.0)
        si = rng.uniform(0.0, 3.0)
        vf = rng.uniform(0.0, 3.0)
        set_attr(G.nodes[node], ALIAS_DNFR, dnfr)
        set_attr(G.nodes[node], ALIAS_DEPI, depi)
        set_attr(G.nodes[node], ALIAS_SI, si)
        set_attr(G.nodes[node], ALIAS_VF, vf)
        G.nodes[node]["_prev_Si"] = si - rng.uniform(-0.2, 0.2)
        G.nodes[node]["_prev_vf"] = vf - rng.uniform(-0.2, 0.2)
        G.nodes[node]["_prev_dvf"] = rng.uniform(-0.5, 0.5)
    return G


@contextlib.contextmanager
def _override_numpy(module):
    original = coherence.get_numpy

    def _patched():
        return module

    coherence.get_numpy = _patched
    try:
        yield
    finally:
        coherence.get_numpy = original


def _run_once(node_count: int, *, np_module, n_jobs: int | None) -> float:
    hist = {"stable_frac": [], "delta_Si": [], "B": []}
    G = _build_graph(node_count)
    with _override_numpy(np_module):
        start = time.perf_counter()
        coherence._track_stability(
            G,
            hist,
            dt=1.0,
            eps_dnfr=0.5,
            eps_depi=0.5,
            n_jobs=n_jobs,
        )
        duration = time.perf_counter() - start
    return duration


def run(node_count: int = 2000, repeats: int = 5, n_jobs: int = 4) -> None:
    np_module = coherence.get_numpy()
    print(f"Benchmarking with {node_count} nodes, repeats={repeats}")

    if np_module is not None:
        vectorized = (
            sum(
                _run_once(node_count, np_module=np_module, n_jobs=None)
                for _ in range(repeats)
            )
            / repeats
        )
        print(f"vectorized (NumPy): {vectorized:.6f}s")
    else:
        print("NumPy not available; skipping vectorized measurement.")

    fallback_seq = (
        sum(_run_once(node_count, np_module=None, n_jobs=None) for _ in range(repeats))
        / repeats
    )
    print(f"fallback (sequential): {fallback_seq:.6f}s")

    fallback_parallel = (
        sum(
            _run_once(node_count, np_module=None, n_jobs=n_jobs) for _ in range(repeats)
        )
        / repeats
    )
    print(f"fallback (parallel, n_jobs={n_jobs}): {fallback_parallel:.6f}s")


if __name__ == "__main__":
    run()
