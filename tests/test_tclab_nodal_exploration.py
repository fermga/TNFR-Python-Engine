"""Synthetic-only equation and ingestion checks for the TCLab exploration.

These controls do not load observations, fit the real runs, or establish a
physical identification of the supplied thermal graph model.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.alias import get_attr, set_attr
from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_EPI
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry

_PATH = Path(__file__).resolve().parents[1] / "benchmarks/tclab_nodal_exploration.py"
_SPEC = importlib.util.spec_from_file_location("tclab_exploration_test", _PATH)
BENCH = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = BENCH
_SPEC.loader.exec_module(BENCH)


def _csv(tmp_path, text, name="synthetic.csv"):
    """Write synthetic bytes and independently compute their Git blob hash."""
    path = tmp_path / name
    raw = text.encode("ascii")
    path.write_bytes(raw)
    blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    return path, blob_sha


def test_markov_equal_channels_follow_grounded_scalar_exponential():
    # Equal forcing removes interchannel exchange. The bath remains at the
    # initial temperature, giving x(t)=x0+gain*q*(1-exp(-leak*t))/leak.
    leak, coupling, gain = 0.08, 0.17, 0.04
    times = np.array([12.0, 12.1, 13.7, 18.0, 31.0])
    commands = np.full((len(times), 2), 40.0)
    initial = np.array([23.0, 23.0])

    actual = BENCH.predict("markov", [leak, coupling, gain], times, commands, initial)
    expected = 23.0 + gain * 40.0 / leak * (-np.expm1(-leak * (times - times[0])))

    np.testing.assert_allclose(
        actual, np.column_stack((expected, expected)), rtol=1e-11
    )
    np.testing.assert_array_equal(actual[0], initial)


def test_memory_matrix_has_grounded_heater_and_sensor_exchange_rows():
    leak, coupling, backreaction, sensor_rate, gain = 0.07, 0.11, 0.13, 0.19, 0.03
    graph, matrix, forcing, observation = BENCH.model_matrices(
        "memory", [leak, coupling, backreaction, sensor_rate, gain]
    )
    expected = np.array(
        [
            [-(leak + coupling + backreaction), coupling, backreaction, 0.0],
            [coupling, -(leak + coupling + backreaction), 0.0, backreaction],
            [sensor_rate, 0.0, -sensor_rate, 0.0],
            [0.0, sensor_rate, 0.0, -sensor_rate],
        ]
    )
    np.testing.assert_allclose(matrix, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(forcing, [[gain, 0], [0, gain], [0, 0], [0, 0]])
    np.testing.assert_array_equal(observation, [[0, 0, 1, 0], [0, 0, 0, 1]])

    # Compute the graph's normalized pressure independently of the benchmark
    # and shared matrix builder, including the finite clamped-bath coordinate.
    nodes = list(graph)
    assert len(nodes) == 5
    capacities = np.array([graph.nodes[node]["nu_f"] for node in nodes])
    assert capacities[-1] == 0.0
    weights = nx.to_numpy_array(graph, nodelist=nodes, weight="weight")
    strengths = weights.sum(axis=1)
    state = np.array([2.1, 3.5, -1.0, 4.0, 0.7])
    pressure = (weights @ state) / strengths - state
    nodal_rate = capacities * pressure
    grounded_rate = expected @ state[:4] + np.array([leak, leak, 0, 0]) * state[-1]
    np.testing.assert_allclose(nodal_rate[:4], grounded_rate, rtol=1e-13, atol=1e-15)
    assert nodal_rate[-1] == 0.0


def test_memory_graph_executes_grounded_transport_and_declared_heater_forcing(
    monkeypatch,
):
    graph, generator, input_map, _ = BENCH.model_matrices(
        "memory", [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1 / 32]
    )
    inject_defaults(graph)
    graph.graph.update(
        DNFR_WEIGHTS={"epi": 1.0, "phase": 0.0, "vf": 0.0, "topo": 0.0},
        use_extended_dynamics=False,
        DT_MIN=0.0,
        CLIP_MODE="hard",
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        GAMMA={"type": "synthetic-tclab-heater"},
    )
    before = np.array([0.625, 0.375, 0.5625, 0.4375, 0.5])
    for node, value in zip(graph, before):
        set_attr(graph.nodes[node], ALIAS_EPI, value)
    commands = np.array([2.0, 3.0])
    supplied_rates = np.r_[input_map @ commands, 0.0]

    def heater_source(current, node, time, spec):
        assert current is graph
        return float(supplied_rates[node])

    monkeypatch.setitem(
        GAMMA_REGISTRY, "synthetic-tclab-heater", GammaEntry(heater_source, False)
    )
    dt = 1 / 64
    expected_rate = np.r_[
        generator @ (before[:4] - before[-1]) + input_map @ commands, 0.0
    ]
    expected = before + dt * expected_rate
    assert np.all((-10.0 < expected) & (expected < 10.0))

    # Run the actual pressure owner, then one held-pressure Euler step through
    # the live registry. This is a finite execution check, not an exact flow.
    default_compute_delta_nfr(graph)
    update_epi_via_nodal_equation(graph, dt=dt, t=0.0, method="euler")

    actual = np.array([get_attr(graph.nodes[node], ALIAS_EPI) for node in graph])
    retained_rates = [get_attr(graph.nodes[node], ALIAS_DEPI) for node in graph]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-16)
    np.testing.assert_allclose(retained_rates, expected_rate, rtol=0.0, atol=2e-16)
    assert actual[-1] == before[-1]
    assert graph.graph["_t"] == dt


@pytest.mark.parametrize("kind", ["markov", "memory"])
def test_piecewise_prediction_matches_independent_scalar_ode(kind):
    leak, coupling, backreaction, sensor_rate, gain = 0.06, 0.09, 0.14, 0.21, 0.025
    parameters = (
        [leak, coupling, gain]
        if kind == "markov"
        else [leak, coupling, backreaction, sensor_rate, gain]
    )
    times = np.array([11.0, 11.2, 12.4, 14.1, 17.0])
    commands = np.array([[0, 50], [80, 0], [25, 75], [0, 0], [99, 99]], dtype=float)
    initial = np.array([22.0, 25.0])
    bath = initial.mean()
    state = initial.copy() if kind == "markov" else np.tile(initial, 2)
    expected = [initial.copy()]

    for index, duration in enumerate(np.diff(times)):
        q1, q2 = commands[index]

        def derivative(_time, x):
            h1, h2 = x[:2]
            h1_rate = leak * (bath - h1) + coupling * (h2 - h1) + gain * q1
            h2_rate = leak * (bath - h2) + coupling * (h1 - h2) + gain * q2
            if kind == "markov":
                return [h1_rate, h2_rate]
            s1, s2 = x[2:]
            return [
                h1_rate + backreaction * (s1 - h1),
                h2_rate + backreaction * (s2 - h2),
                sensor_rate * (h1 - s1),
                sensor_rate * (h2 - s2),
            ]

        solution = solve_ivp(
            derivative, (0.0, duration), state, method="DOP853", rtol=1e-12, atol=1e-13
        )
        assert solution.success
        state = solution.y[:, -1]
        expected.append(state[:2].copy() if kind == "markov" else state[2:].copy())

    actual = BENCH.predict(kind, parameters, times, commands, initial)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    # The final command has no following interval and must not act retroactively.
    changed = commands.copy()
    changed[-1] = [0.0, 0.0]
    np.testing.assert_array_equal(
        BENCH.predict(kind, parameters, times, changed, initial), actual
    )


def test_csv_preserves_schedule_and_initial_state_without_exposing_suffix(tmp_path):
    header = "Time,Q1,Q2,T1,T2\n"
    original, first_blob = _csv(
        tmp_path,
        header + "10,0,20,21,23\n11.5,50,10,22,24\n14,0,0,24,25\n",
        "original.csv",
    )
    changed, second_blob = _csv(
        tmp_path,
        header + "10,0,20,21,23\n11.5,50,10,80,-15\n14,0,0,99,100\n",
        "changed.csv",
    )
    first = BENCH.load_csv(original, first_blob, responses=False)
    second = BENCH.load_csv(changed, second_blob, responses=False)

    assert "observations" not in first
    assert "observations" not in second
    for key in ("times", "commands", "initial"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(first["times"], [10, 11.5, 14])
    np.testing.assert_array_equal(first["commands"], [[0, 20], [50, 10], [0, 0]])
    np.testing.assert_array_equal(first["initial"], [21, 23])
    assert first["blob_sha"] == first_blob
    assert second["blob_sha"] == second_blob
    assert first_blob != second_blob
    assert first["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert first["sha256"] != second["sha256"]

    observed = BENCH.load_csv(original, first_blob, responses=True)
    np.testing.assert_array_equal(
        observed["observations"], [[21, 23], [22, 24], [24, 25]]
    )


@pytest.mark.parametrize(
    "text",
    [
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n0,5,0,22,23\n",
        "Time,Q1,Q2,T1,T2\n1,0,0,20,21\n0,5,0,22,23\n",
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\nnan,5,0,22,23\n",
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n1,inf,0,22,23\n",
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n1,5,0,nan,23\n",
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n1,5,0,22,inf\n",
        "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n",
        "Time,Q1,Q2,T1,Unknown\n0,0,0,20,21\n1,5,0,22,23\n",
        "Time,Time,Q1,Q2,T1,T2\n9,0,0,0,20,21\n8,1,5,0,22,23\n",
    ],
    ids=[
        "duplicate-time",
        "decreasing-time",
        "nonfinite-time",
        "nonfinite-command",
        "nonfinite-response-one",
        "nonfinite-response-two",
        "one-row",
        "wrong-columns",
        "duplicate-column",
    ],
)
def test_csv_rejects_invalid_schedule_schema_and_scored_values(tmp_path, text):
    path, blob_sha = _csv(tmp_path, text)
    with pytest.raises(ValueError):
        BENCH.load_csv(path, blob_sha, responses=True)


def test_csv_rejects_unpinned_bytes_even_with_valid_rows(tmp_path):
    path, _ = _csv(tmp_path, "Time,Q1,Q2,T1,T2\n0,0,0,20,21\n1,5,0,22,23\n")
    with pytest.raises(ValueError):
        BENCH.load_csv(path, "0" * 40, responses=False)


def test_csv_size_bound_applies_before_consuming_an_otherwise_valid_schedule(tmp_path):
    long_zero = "0." + "0" * 100_000
    path, blob_sha = _csv(
        tmp_path,
        "Time,Q1,Q2,T1,T2\n" + f"0,{long_zero},0,20,21\n1,5,0,22,23\n",
    )
    with pytest.raises(ValueError, match="bounded intake size"):
        BENCH.load_csv(path, blob_sha, responses=False)


@pytest.mark.parametrize(
    "first_row,second_row",
    [
        ("0,0,0,nan,21", "1,5,0,22,23"),
        ("0,0,0,20,inf", "1,5,0,22,23"),
        ("0,0,0,20,21", "1,-1,0,22,23"),
        ("0,0,0,20,21", "1,5,101,22,23"),
    ],
    ids=[
        "nonfinite-initial-one",
        "nonfinite-initial-two",
        "negative-command",
        "command-above-100",
    ],
)
def test_forecast_intake_rejects_invalid_consumed_state_and_commands(
    tmp_path, first_row, second_row
):
    path, blob_sha = _csv(tmp_path, f"Time,Q1,Q2,T1,T2\n{first_row}\n{second_row}\n")
    with pytest.raises(ValueError):
        BENCH.load_csv(path, blob_sha, responses=False)


@pytest.mark.parametrize(
    "calibration,evaluation",
    [
        (
            [{"name": "cal.csv", "sha": "a" * 40}],
            [{"name": "alias.csv", "sha": "a" * 40}],
        ),
        (
            [{"name": "same.csv", "sha": "a" * 40}],
            [{"name": "same.csv", "sha": "b" * 40}],
        ),
        (
            [
                {"name": "cal.csv", "sha": "a" * 40},
                {"name": "alias.csv", "sha": "a" * 40},
            ],
            [{"name": "eval.csv", "sha": "b" * 40}],
        ),
        (
            [{"name": "cal.csv", "sha": "a" * 40}],
            [
                {"name": "eval.csv", "sha": "b" * 40},
                {"name": "alias.csv", "sha": "b" * 40},
            ],
        ),
    ],
    ids=[
        "cross-split-alias",
        "cross-split-name",
        "calibration-alias",
        "evaluation-alias",
    ],
)
def test_split_rejects_duplicate_names_or_byte_identical_aliases(
    calibration, evaluation
):
    with pytest.raises(ValueError):
        BENCH.validate_split(calibration, evaluation)


def test_split_accepts_distinct_named_blobs():
    BENCH.validate_split(
        [{"name": "cal.csv", "sha": "a" * 40}],
        [{"name": "eval.csv", "sha": "b" * 40}],
    )
