"""Synthetic full-law generator controls, independent of reserved preparations."""

import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_two_port_readout as owner


def _arguments():
    # Deliberately unrelated to the unknown affine winding preparation.
    return dict(
        initial_form_bounds=tuple((Q((3 * i) % 7 - 3, 4096),) * 2 for i in range(18)),
        initial_phase_bounds=tuple((Q((5 * i) % 11 - 5, 64),) * 2 for i in range(18)),
        phase_increment=Q(1, 32),
        probe_duration=Q(1, 16),
        order=4,
    )


@pytest.fixture(scope="module")
def readout():
    return owner.bound_sine_two_port_readout(**_arguments())


def _independent_geometry():
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    degrees = np.array([graph.degree[i] for i in range(18)], dtype=float)
    laplacian = nx.laplacian_matrix(graph, nodelist=range(18)).toarray()
    return graph, degrees, laplacian


def _numeric_flow(graph, degrees, laplacian):
    gamma = 1 / (1023 * np.pi)

    def flow(_, state):
        x, phase = state[:18], state[18:]
        gradient = laplacian @ x
        currents = np.zeros(18)
        for i, j in graph.edges:
            current = np.sin(phase[j] - phase[i])
            currents[i] += current
            currents[j] -= current
        return np.concatenate(
            ((-gradient + gamma * currents) / degrees, gamma * gradient / degrees)
        )

    return flow


def test_five_required_primitive_inputs_have_no_calibration_or_inverse_report():
    parameters = signature(owner.bound_sine_two_port_readout).parameters
    assert set(parameters) == set(_arguments())
    assert len(parameters) == 5
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )


def test_complete_graph_and_phase_only_event_preserve_all_source_coordinates(readout):
    arguments = _arguments()
    graph, degrees, _ = _independent_geometry()
    assert readout.admitted and readout.status == "admitted"
    assert readout.geometry.edges == tuple(
        sorted(tuple(sorted(edge)) for edge in graph.edges)
    )
    assert readout.degrees == tuple(int(v) for v in degrees)
    assert len(readout.step.initial_box) == 36
    assert len(readout.step.series) == 36
    assert all(len(row) == 5 for row in readout.step.series)
    assert readout.post_event_initial_box[:18] == readout.initial_form_bounds
    for i in range(18):
        before = arguments["initial_phase_bounds"][i][0]
        kick = arguments["phase_increment"] * (int(i == 4) - int(i == 5))
        assert readout.post_event_initial_box[18 + i].contains(before + kick)
    assert readout.baseline_readout_bounds.contains(
        arguments["initial_form_bounds"][4][0] - arguments["initial_form_bounds"][5][0]
    )
    assert readout.clock == "tau=e*t"
    assert readout.capacity == (Q(1),) * 18


def test_both_complete_rows_and_jet_derivative_match_independent_fast_law(readout):
    graph, degrees, laplacian = _independent_geometry()
    numeric = _numeric_flow(graph, degrees, laplacian)
    state = np.array(
        [float(value.midpoint) for value in readout.post_event_initial_box]
    )
    expected = numeric(0, state)
    actual = np.array([float(row[1].midpoint) for row in readout.step.series])
    assert np.allclose(actual, expected, atol=2e-18, rtol=2e-14)
    assert np.linalg.norm(actual[18:]) > 1e-7  # Hidden phase row truly moves.
    flow, _ = owner._full_sine_field(
        readout.reference_model, readout.geometry, readout.degrees
    )
    direction = tuple(Q((i % 5) - 2, 32) for i in range(36))
    jets = tuple(
        Jet((value, I(v)))
        for value, v in zip(readout.post_event_initial_box, direction)
    )
    rates = flow(jets)
    epsilon = 1e-6
    d = np.array(direction, dtype=float)
    finite_difference = (
        numeric(0, state + epsilon * d) - numeric(0, state - epsilon * d)
    ) / (2 * epsilon)
    jet_derivative = np.array([float(row.coeffs[1].midpoint) for row in rates])
    assert np.allclose(jet_derivative, finite_difference, atol=2e-13, rtol=1e-7)
    assert abs(degrees @ actual[:18]) < 1e-17
    assert abs(degrees @ actual[18:]) < 1e-20


def test_synthetic_numerical_crosscheck_lies_inside_validated_full_state(readout):
    # An independent nonvalidated check of this synthetic control only. The
    # shared retained Picard/Taylor evidence, not solve_ivp, supplies the bound.
    graph, degrees, laplacian = _independent_geometry()
    initial = np.array(
        [float(value.midpoint) for value in readout.post_event_initial_box]
    )
    response = solve_ivp(
        _numeric_flow(graph, degrees, laplacian),
        (0, float(readout.probe_duration)),
        initial,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert response.success
    endpoint = response.y[:, -1]
    for value, expected in zip(readout.step.endpoint, endpoint):
        assert float(value.lo) <= expected <= float(value.hi)
    increment = endpoint[4] - endpoint[5] - initial[4] + initial[5]
    assert (
        float(readout.true_increment_bounds.lo)
        <= increment
        <= float(readout.true_increment_bounds.hi)
    )
    assert readout.step.picard_interior_margin > 0
    assert readout.step.domain_lower_bounds == (Q(1),)
    assert any(value.width > 0 for value in readout.step.local_remainder_bounds)


def test_zero_event_continuation_carries_the_complete_evolving_endpoint(readout):
    duration = Q(1, 32)
    carried = tuple((value.lo, value.hi) for value in readout.step.endpoint)
    continuation = owner.bound_sine_two_port_readout(
        initial_form_bounds=carried[:18],
        initial_phase_bounds=carried[18:],
        phase_increment=Q(0),
        probe_duration=duration,
        order=4,
    )
    assert continuation.admitted
    assert continuation.phase_increment == 0
    assert continuation.post_event_initial_box == readout.step.endpoint
    assert continuation.step.initial_box == readout.step.endpoint
    assert continuation.initial_form_bounds == readout.step.endpoint[:18]
    assert continuation.initial_phase_bounds == readout.step.endpoint[18:]
    assert continuation.baseline_readout_bounds == readout.endpoint_readout_bounds

    # Integrate the independent complete law continuously across the observation
    # time. The second segment has no kick and carries the hidden phase row.
    graph, degrees, laplacian = _independent_geometry()
    initial = np.array(
        [float(value.midpoint) for value in readout.post_event_initial_box]
    )
    t_middle = float(readout.probe_duration)
    t_end = float(readout.probe_duration + duration)
    response = solve_ivp(
        _numeric_flow(graph, degrees, laplacian),
        (0, t_end),
        initial,
        t_eval=(t_middle, t_end),
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )
    assert response.success
    middle, endpoint = response.y.T
    for bounds, expected in zip(continuation.step.endpoint, endpoint):
        assert float(bounds.lo) <= expected <= float(bounds.hi)
    for bounds, expected in zip(continuation.step.initial_box, middle):
        assert float(bounds.lo) <= expected <= float(bounds.hi)
    increment = endpoint[4] - endpoint[5] - middle[4] + middle[5]
    assert (
        float(continuation.true_increment_bounds.lo)
        <= increment
        <= float(continuation.true_increment_bounds.hi)
    )
    assert np.linalg.norm(endpoint[:18] - middle[:18]) > 1e-5
    assert np.linalg.norm(endpoint[18:] - middle[18:]) > 1e-8
    assert continuation.step.picard_interior_margin > 0


@pytest.mark.parametrize("zero", (0, Q(0), 0.0, -0.0))
def test_zero_amplitude_is_admitted_without_coercing_boolean_scalars(monkeypatch, zero):
    arguments = _arguments()
    arguments["phase_increment"] = zero

    def stopped(box, duration, flow, domain, *, order):
        expected = tuple(
            I(lo, hi)
            for row in (
                arguments["initial_form_bounds"],
                arguments["initial_phase_bounds"],
            )
            for lo, hi in row
        )
        assert box == expected
        return None, box, "controlled_stop_after_source_admission"

    monkeypatch.setattr(owner, "validated_box_taylor_step", stopped)
    result = owner.bound_sine_two_port_readout(**arguments)
    assert result.phase_increment == Q(0)
    assert result.unavailable_reasons == ("controlled_stop_after_source_admission",)


def test_direct_increment_is_not_independent_endpoint_minus_source_boxes():
    arguments = _arguments()
    radius = Q(1, 2**16)
    arguments["initial_form_bounds"] = tuple(
        (lo - radius, hi + radius) for lo, hi in arguments["initial_form_bounds"]
    )
    arguments["probe_duration"] = Q(1, 128)
    result = owner.bound_sine_two_port_readout(**arguments)
    assert result.admitted
    subtracted = result.endpoint_readout_bounds - result.baseline_readout_bounds
    assert result.true_increment_bounds.width < subtracted.width / 10
    assert (
        result.true_increment_bounds
        == result.step.increment[4] - result.step.increment[5]
    )


def test_global_smooth_domain_does_not_assert_acuity():
    arguments = _arguments()
    arguments["initial_phase_bounds"] = ((Q(0), Q(0)), (Q(3), Q(3))) + (
        (Q(0), Q(0)),
    ) * 16
    arguments["probe_duration"] = Q(1, 128)
    result = owner.bound_sine_two_port_readout(**arguments)
    assert result.admitted
    assert (
        "global_smooth_domain_is_not_acute_chart_or_identity_retention" in result.scope
    )


def test_failed_shared_step_keeps_source_and_failure_without_readout(monkeypatch):
    calls = []

    def unavailable(box, duration, flow, domain, *, order):
        calls.append((box, duration, order))
        return None, box, "controlled_Picard_failure"

    monkeypatch.setattr(owner, "validated_box_taylor_step", unavailable)
    result = owner.bound_sine_two_port_readout(**_arguments())
    assert len(calls) == 1
    assert not result.admitted and result.status == "unavailable"
    assert result.failed_tube == result.post_event_initial_box
    assert (
        result.true_increment_bounds
        is result.endpoint_readout_bounds
        is result.step
        is None
    )
    assert result.unavailable_reasons == ("controlled_Picard_failure",)
    assert result.baseline_readout_bounds is not None


def test_serialization_retains_all_derivatives_and_exact_baseline(readout):
    data = json.loads(json.dumps(readout.to_dict(), allow_nan=False))
    assert data["schema"] == "tnfr.sine-two-port-readout.v1"
    result = data["report"]
    assert result["probe_duration"] == {"numerator": 1, "denominator": 16}
    assert len(result["step"]["series"]) == 36
    assert all(len(row) == 5 for row in result["step"]["series"])
    assert result["step"]["method"] == "direct_source_box_Picard_Taylor_dyadic128_v1"
    value = result["baseline_readout_bounds"]["lo"]
    assert (
        Q(value["numerator"], value["denominator"])
        == readout.baseline_readout_bounds.lo
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("phase_increment", -1),
        ("phase_increment", Q(-1, 10**400)),
        ("phase_increment", Q(1001, 1000)),
        ("probe_duration", 0),
        ("probe_duration", Q(1001, 1000)),
        ("phase_increment", True),
        ("phase_increment", False),
        ("phase_increment", np.bool_(False)),
        ("probe_duration", np.bool_(True)),
        ("phase_increment", float("nan")),
        ("probe_duration", float("inf")),
        ("order", True),
        ("order", Q(4)),
        ("order", 0),
        ("order", 17),
        ("initial_form_bounds", ((0, 0),) * 17),
        ("initial_phase_bounds", ((0, 0),) * 19),
        ("initial_form_bounds", ((True, 1),) * 18),
        ("initial_phase_bounds", ((float("nan"), 1),) * 18),
        ("initial_form_bounds", (I(0),) * 18),
        ("initial_phase_bounds", ((1, 0),) * 18),
    ),
)
def test_invalid_primitive_input_is_rejected_before_flow(monkeypatch, field, value):
    arguments = _arguments()
    arguments[field] = value
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *args: pytest.fail("invalid input reached flow"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_two_port_readout(**arguments)


def test_inverse_and_old_producers_cannot_generate_the_readout(monkeypatch):
    from tnfr.physics import relational_sine_two_port_capture as capture
    from tnfr.physics import relational_sine_two_port_compatibility as compatibility
    from tnfr.physics import relational_sine_two_port_dipole as dipole
    from tnfr.physics import relational_sine_two_port_inference as inference

    def forbidden(*args, **kwargs):
        pytest.fail("independent source-box readout called inverse or frozen producer")

    monkeypatch.setattr(inference, "infer_sine_two_port_geometry", forbidden)
    monkeypatch.setattr(compatibility, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(capture, "assess_sine_two_port_capture", forbidden)
    monkeypatch.setattr(dipole, "assess_sine_two_port_dipole", forbidden)
    assert owner.bound_sine_two_port_readout(**_arguments()).admitted
