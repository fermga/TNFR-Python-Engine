from __future__ import annotations

import copy

import numpy as np
import pytest

import tnfr.glyph_history as glyph_history
from tnfr.constants import inject_defaults
from tnfr.dynamics.runtime import run, step
from tnfr.glyph_history import ensure_history
from tnfr.mathematics import (
    BasicStateProjector,
    CoherenceOperator,
    HilbertSpace,
    MathematicalDynamicsEngine,
    make_frequency_operator,
)
from tnfr.structural import create_nfr
from tnfr.validation import validate_window as _validate_window

if not hasattr(glyph_history, "validate_window"):
    glyph_history.validate_window = _validate_window


def _attach_math_engine_config(graph):  # noqa: ANN001 - test helper
    hilbert = HilbertSpace(3)
    generator = np.diag([0.1, -0.05, 0.02])
    coherence_operator = CoherenceOperator(np.eye(3))
    frequency_operator = make_frequency_operator(np.eye(3))
    graph.graph["MATH_ENGINE"] = {
        "enabled": True,
        "hilbert_space": hilbert,
        "coherence_operator": coherence_operator,
        "coherence_threshold": coherence_operator.c_min,
        "frequency_operator": frequency_operator,
        "dynamics_engine": MathematicalDynamicsEngine(generator, hilbert),
        "state_projector": BasicStateProjector(),
    }


def _build_seed_graph():  # noqa: ANN001 - test helper
    base_graph, _ = create_nfr("seed", epi=0.9, vf=1.1, theta=0.2)
    create_nfr("partner", epi=0.6, vf=0.8, theta=-0.1, graph=base_graph)
    inject_defaults(base_graph)
    return base_graph


def _snapshot_nodes(graph):  # noqa: ANN001 - test helper
    return {
        node: {
            key: float(value) for key, value in data.items() if isinstance(value, float)
        }
        for node, data in graph.nodes(data=True)
    }


def test_step_with_math_engine_preserves_classical_updates():
    base_graph = _build_seed_graph()
    math_graph = copy.deepcopy(base_graph)
    classic_graph = copy.deepcopy(base_graph)

    _attach_math_engine_config(math_graph)

    step(math_graph, dt=0.05)
    step(classic_graph, dt=0.05)

    math_snapshot = _snapshot_nodes(math_graph)
    classic_snapshot = _snapshot_nodes(classic_graph)
    assert math_snapshot.keys() == classic_snapshot.keys()
    for node, values in math_snapshot.items():
        base_values = classic_snapshot[node]
        assert values.keys() == base_values.keys()
        for key, value in values.items():
            assert value == pytest.approx(base_values[key])

    history = ensure_history(math_graph)
    summaries = history.get("math_engine_summary", [])
    assert len(summaries) == 1
    summary = summaries[-1]
    assert summary["normalized"] is True
    assert summary["coherence"]["passed"] is True
    freq_summary = summary["frequency"]
    assert isinstance(freq_summary, dict)
    assert freq_summary["passed"] is True
    assert freq_summary["projection_passed"] is True

    telemetry = math_graph.graph.get("telemetry", {}).get("math_engine")
    assert telemetry
    assert telemetry["step"] == summary["step"]
    assert telemetry["norm"] == pytest.approx(summary["norm"])


def test_math_engine_evolution_is_deterministic():
    graph_one = _build_seed_graph()
    graph_two = copy.deepcopy(graph_one)
    _attach_math_engine_config(graph_one)
    _attach_math_engine_config(graph_two)

    run(graph_one, 3, dt=0.05)
    run(graph_two, 3, dt=0.05)

    history_one = ensure_history(graph_one)["math_engine_summary"]
    history_two = ensure_history(graph_two)["math_engine_summary"]

    assert history_one == history_two

    state_one = graph_one.graph["MATH_ENGINE"]["_state_vector"]
    state_two = graph_two.graph["MATH_ENGINE"]["_state_vector"]
    np.testing.assert_allclose(state_one, state_two)
