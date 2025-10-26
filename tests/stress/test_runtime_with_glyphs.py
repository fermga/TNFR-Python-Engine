"""Stress test covering glyph-enabled runtime telemetry under load."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

import networkx as nx
import pytest

pytest.importorskip("numpy")

try:  # pragma: no cover - optional plugin detection
    import pytest_timeout  # noqa: F401
except ImportError:  # pragma: no cover - fallback when plugin missing
    def timeout_mark(_: float):
        def decorator(func):
            return func

        return decorator
else:  # pragma: no cover - executed when plugin available
    timeout_mark = pytest.mark.timeout

from tnfr.alias import get_attr, set_attr
from tnfr.constants import DEFAULTS, get_aliases, inject_defaults
from tnfr.dynamics import selectors
import tnfr.dynamics.runtime as runtime
from tnfr.glyph_history import ensure_history
from tnfr.metrics import register_metrics_callbacks

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")
ALIAS_SI = get_aliases("SI")
ALIAS_DNFR = get_aliases("DNFR")

pytestmark = [pytest.mark.slow, pytest.mark.stress]


def _seed_glyph_runtime_graph(
    *,
    num_nodes: int,
    edge_probability: float,
    seed: int,
) -> nx.Graph:
    """Create a reproducible graph configured for glyph selection stress runs."""

    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed)
    inject_defaults(graph)
    graph.graph["DNFR_WEIGHTS"] = dict(DEFAULTS["DNFR_WEIGHTS"])
    graph.graph.setdefault("RANDOM_SEED", seed)
    graph.graph["HISTORY_MAXLEN"] = 0
    graph.graph["glyph_selector"] = selectors.ParametricGlyphSelector()

    metrics_cfg = dict(graph.graph.get("METRICS", {}))
    metrics_cfg["verbosity"] = "detailed"
    metrics_cfg["enabled"] = True
    metrics_cfg.setdefault("normalize_series", True)
    graph.graph["METRICS"] = metrics_cfg
    graph.graph["STOP_EARLY"] = {"enabled": False, "window": 48, "fraction": 0.92}

    twopi = 2.0 * math.pi
    for node, data in graph.nodes(data=True):
        base = seed + int(node)
        theta = ((base * 0.017) % twopi) - math.pi
        epi = math.sin(base * 0.029) * 0.42
        vf = 0.35 + 0.55 * ((base % 23) / 22.0)
        si = 0.5 + 0.5 * math.cos(base * 0.021)
        dnfr = 0.06 * math.sin(base * 0.033)

        set_attr(data, ALIAS_THETA, theta)
        set_attr(data, ALIAS_EPI, epi)
        set_attr(data, ALIAS_VF, vf)
        set_attr(data, ALIAS_SI, max(0.0, min(1.0, si)))
        set_attr(data, ALIAS_DNFR, dnfr)

    return graph


@timeout_mark(30)
def test_runtime_run_glyph_pipeline_history_is_finite() -> None:
    """Runtime with glyph application must keep telemetry bounded and finite."""

    node_count = 200
    edge_probability = 0.06
    steps = 12
    dt = 0.05
    seed = 7341

    graph = _seed_glyph_runtime_graph(
        num_nodes=node_count,
        edge_probability=edge_probability,
        seed=seed,
    )

    register_metrics_callbacks(graph)
    ensure_history(graph)

    start = time.perf_counter()
    runtime.run(graph, steps=steps, dt=dt, use_Si=True, apply_glyphs=True)
    elapsed = time.perf_counter() - start

    assert elapsed < 30.0

    history = ensure_history(graph)

    numeric_series_keys = (
        "C_steps",
        "stable_frac",
        "glyph_load_stabilizers",
        "glyph_load_disr",
        "sense_sigma_mag",
        "Si_mean",
    )

    for key in numeric_series_keys:
        series = history.get(key, [])
        assert isinstance(series, list)
        assert len(series) >= steps
        tail = series[-min(5, len(series)) :]
        for value in tail:
            assert math.isfinite(float(value))

    since_al = history.get("since_AL")
    since_en = history.get("since_EN")
    assert isinstance(since_al, Mapping)
    assert isinstance(since_en, Mapping)
    assert since_al and since_en

    glyph_histories = sum(
        1 for _, data in graph.nodes(data=True) if data.get("glyph_history")
    )
    assert glyph_histories >= node_count // 2

    dnfr_total = sum(
        float(get_attr(data, ALIAS_DNFR, 0.0)) for _, data in graph.nodes(data=True)
    )
    assert math.isfinite(dnfr_total)
