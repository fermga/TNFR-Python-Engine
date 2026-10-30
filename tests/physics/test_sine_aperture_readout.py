"""Independent affine-clock average controls; no reserved response is replayed."""

import json
from fractions import Fraction as Q
from inspect import Parameter, signature

import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_aperture_readout as owner
from tnfr.utils.io import json_loads


def _arguments():
    form = [Q(0)] * 18
    form[4], form[5] = Q(1, 6), Q(-1, 6)
    return dict(
        initial_form_bounds=tuple((v, v) for v in form),
        initial_phase_bounds=((Q(0), Q(0)),) * 18,
        phase_increments=(Q(1, 6), Q(2, 3)),
        probe_duration=Q(1, 5),
        initial_clock_rate=Q(1, 2),
        clock_slope=Q(1, 7),
        order=4,
    )


def _polynomial_structural_field(*_):
    # Independent structural law, chosen to make every observed-time
    # coordinate and integral an exactly known polynomial. Both nodal row
    # types must receive the clock factor; the passive integral must not.
    def flow(state):
        zero = state[0] * 0
        rates = [zero] * 36
        rates[4], rates[5], rates[18] = zero + Q(1, 2), zero - Q(1, 2), zero + 3
        return tuple(rates)

    return flow, lambda _: (Q(1),)


@pytest.fixture(scope="module")
def polynomial_readout():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_full_sine_field", _polynomial_structural_field)
        return owner.bound_sine_aperture_readout(**_arguments())


def _exposure(time, rate, slope):
    return rate * time + slope * time**2 / 2


def _integral(time, rate, slope):
    return time / 3 + rate * time**2 / 2 + slope * time**3 / 6


def test_seven_required_primitives_exclude_inverse_and_sensor_parameters():
    parameters = signature(owner.bound_sine_aperture_readout).parameters
    assert set(parameters) == set(_arguments())
    assert len(parameters) == 7
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )


def test_observed_time_integral_excludes_structural_weighting_and_endpoint_rules(
    polynomial_readout,
):
    report = polynomial_readout
    rate, slope = report.initial_clock_rate, report.clock_slope
    assert report.admitted and report.completed_window_count == 4
    for index, ((start, end), step, average) in enumerate(
        zip(report.aperture_windows, report.steps, report.averaged_readout_bounds)
    ):
        exposure = _exposure(end, rate, slope)
        expected = Q(1, 3) + rate * (start + end) / 2
        expected += slope * (start**2 + start * end + end**2) / 6
        assert average.contains(expected)
        assert step.endpoint[4].contains(Q(1, 6) + exposure / 2)
        assert step.endpoint[5].contains(-Q(1, 6) - exposure / 2)
        assert step.endpoint[18].contains(3 * exposure)
        assert step.endpoint[36].contains(rate + slope * end)
        assert step.endpoint[37].contains(_integral(end, rate, slope))
        assert step.time == start and step.duration == end - start
        assert step.picard_interior_margin > 0
        assert step.domain_lower_bounds == (Q(1),)
        assert average == I(
            step.increment[37].lo / (end - start),
            step.increment[37].hi / (end - start),
        )

        tau0, tau1 = _exposure(start, rate, slope), exposure
        wrong_structural_average = ((tau1 - tau0) / 3 + (tau1**2 - tau0**2) / 2) / (
            end - start
        )
        wrong_trapezoid = Q(1, 3) + (tau0 + tau1) / 2
        assert not average.contains(wrong_structural_average)
        assert not average.contains(wrong_trapezoid)
        if index:
            assert not average.contains(_integral(end, rate, slope) / (end - start))
            assert step.initial_box[37].lo > 0


def test_every_event_and_window_carries_all_thirty_eight_coordinates(
    polynomial_readout,
):
    report = polynomial_readout
    initial = (
        report.initial_form_bounds
        + report.initial_phase_bounds
        + (
            I(report.initial_clock_rate),
            I(0),
        )
    )
    assert report.initial_augmented_box == initial
    for index, jump in enumerate(report.phase_event_increments):
        previous = initial if index == 0 else report.steps[index - 1].endpoint
        expected = (
            previous[:18]
            + tuple(
                phase + jump * Q(int(i == 4) - int(i == 5))
                for i, phase in enumerate(previous[18:36])
            )
            + previous[36:]
        )
        assert report.window_initial_boxes[index] == expected
        assert report.steps[index].initial_box == expected
        assert report.window_initial_boxes[index][36:] == previous[36:]
        assert len(report.steps[index].series) == 38
        assert all(len(row) == report.order + 1 for row in report.steps[index].series)
    assert report.phase_event_increments == (Q(1, 6), 0, 0, Q(1, 2))
    assert len(report.state_order) == 38
    assert (
        report.completed_endpoint_box
        == report.final_state_bounds
        == report.steps[-1].endpoint
    )
    assert report.completed_observed_time == 2 * report.probe_duration
    assert report.failed_window_index is report.failed_tube is None
    assert report.unavailable_reasons == ()
    assert report.clock_rate_bounds == (Q(1, 2), Q(39, 70))
    boundaries = (Q(0),) + tuple(end for _, end in report.aperture_windows)
    assert report.cumulative_structural_exposures == tuple(
        _exposure(t, report.initial_clock_rate, report.clock_slope) for t in boundaries
    )


def test_equal_endpoint_exposure_does_not_determine_observed_average(monkeypatch):
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_structural_field)
    arguments = _arguments()
    width = arguments["probe_duration"] / 3
    results = []
    for slope in (Q(-1, 7), Q(1, 7)):
        rate = Q(1, 2) - slope * width / 2
        result = owner.bound_sine_aperture_readout(
            **dict(arguments, initial_clock_rate=rate, clock_slope=slope)
        )
        assert result.admitted
        results.append(result)
        assert result.cumulative_structural_exposures[1] == width / 2
        assert result.steps[0].endpoint[4].contains(Q(1, 6) + width / 4)
        assert result.averaged_readout_bounds[0].contains(
            Q(1, 3) + width / 4 - slope * width**2 / 12
        )
    assert (
        results[0].averaged_readout_bounds[0].lo
        > results[1].averaged_readout_bounds[0].hi
    )


def test_source_box_corner_averages_keep_uncertainty_without_subtracting_baselines(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_structural_field)
    arguments = _arguments()
    radius = Q(1, 256)
    form = list(arguments["initial_form_bounds"])
    for index in (4, 5):
        lower, upper = form[index]
        form[index] = (lower - radius, upper + radius)
    arguments["initial_form_bounds"] = tuple(form)
    report = owner.bound_sine_aperture_readout(**arguments)
    assert report.admitted
    for (start, end), average in zip(
        report.aperture_windows, report.averaged_readout_bounds
    ):
        nominal = Q(1, 3) + report.initial_clock_rate * (start + end) / 2
        nominal += report.clock_slope * (start**2 + start * end + end**2) / 6
        for form4_error in (-radius, radius):
            for form5_error in (-radius, radius):
                assert average.contains(nominal + form4_error - form5_error)
        assert average.width >= 4 * radius
    step = report.steps[2]
    independent_difference = step.endpoint[37] - step.initial_box[37]
    # The passive cumulative baseline is uncertain by this third window.
    # It cancels algebraically in the Taylor increment, while the true
    # source form uncertainty still determines every average's width.
    assert step.initial_box[37].width > 0
    assert step.increment[37].width < independent_difference.width / 2


@pytest.mark.parametrize("failed_index", (0, 1, 3))
def test_failed_window_retains_only_completed_prefix(monkeypatch, failed_index):
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_structural_field)
    kernel = owner.validated_box_taylor_step
    calls = []

    def controlled(box, duration, flow, domain, *, order, time):
        calls.append((box, duration, time))
        if len(calls) == failed_index + 1:
            return None, box, "controlled_Picard_failure"
        return kernel(box, duration, flow, domain, order=order, time=time)

    monkeypatch.setattr(owner, "validated_box_taylor_step", controlled)
    report = owner.bound_sine_aperture_readout(**_arguments())
    assert len(calls) == failed_index + 1
    assert not report.admitted and report.status == "unavailable"
    assert report.completed_window_count == failed_index
    assert len(report.steps) == len(report.averaged_readout_bounds) == failed_index
    assert len(report.window_initial_boxes) == failed_index + 1
    assert report.failed_window_index == failed_index
    assert report.failed_tube == report.window_initial_boxes[-1] == calls[-1][0]
    assert report.final_state_bounds is None
    assert report.unavailable_reasons == ("controlled_Picard_failure",)
    expected_time = report.aperture_windows[failed_index][0]
    assert report.completed_observed_time == expected_time
    expected_endpoint = report.steps[-1].endpoint if failed_index else None
    assert report.completed_endpoint_box == expected_endpoint
    if failed_index == 3:
        assert report.window_initial_boxes[-1][22] == expected_endpoint[22] + Q(1, 2)
        assert report.window_initial_boxes[-1][36:] == expected_endpoint[36:]
    data = report.to_dict()["report"]
    assert data["final_state_bounds"] is None
    assert len(data["averaged_readout_bounds"]) == failed_index


def _sine_arguments(slope):
    # The complete nonzero-width box includes the exact nominal source and
    # its floating representation; numerical comparison is not a proof of
    # enclosure at precision finer than binary64.
    radius = Q(1, 2**44)
    return dict(
        initial_form_bounds=tuple(
            (
                Q(3 * i - 10, 512) + Q(1, 17) - radius,
                Q(3 * i - 10, 512) + Q(1, 17) + radius,
            )
            for i in range(18)
        ),
        initial_phase_bounds=tuple(
            (
                Q((i * i + 3 * i) % 19, 16) - Q(2, 9) - radius,
                Q((i * i + 3 * i) % 19, 16) - Q(2, 9) + radius,
            )
            for i in range(18)
        ),
        phase_increments=(Q(1, 7), Q(3, 7)),
        probe_duration=Q(1, 64),
        initial_clock_rate=Q(7, 6),
        clock_slope=slope,
        order=4,
    )


@pytest.fixture(scope="module", params=(Q(-1, 9), Q(1, 9)))
def sine_readout(request):
    arguments = _sine_arguments(request.param)
    return arguments, owner.bound_sine_aperture_readout(**arguments)


def test_independent_complete_sine_flow_with_observed_time_accumulator(sine_readout):
    # Fresh numerical controls, not a validated solver or reserved prediction.
    # The independent graph, currents and quadrature check the retained
    # interval certificate's implementation, not its mathematical premises.
    arguments, report = sine_readout
    graph = nx.disjoint_union(nx.cycle_graph(9), nx.cycle_graph(9))
    graph.add_edges_from(((0, 9), (1, 10)))
    degrees = np.array([graph.degree[i] for i in range(18)], dtype=float)
    laplacian = nx.laplacian_matrix(graph, nodelist=range(18)).toarray()
    gamma = 1 / (1023 * np.pi)
    slope = float(arguments["clock_slope"])

    def flow(_, state):
        x, phase, rate = state[:18], state[18:36], state[36]
        gradient = laplacian @ x
        currents = np.zeros(18)
        for i, j in graph.edges:
            current = np.sin(phase[j] - phase[i])
            currents[i] += current
            currents[j] -= current
        return np.concatenate(
            (
                rate * (-gradient + gamma * currents) / degrees,
                rate * gamma * gradient / degrees,
                (slope, x[4] - x[5]),
            )
        )

    assert report.admitted
    assert report.degrees == tuple(int(v) for v in degrees)
    assert report.geometry.edges == tuple(
        sorted(tuple(sorted(edge)) for edge in graph.edges)
    )
    state = np.array(
        [
            float((v[0] + v[1]) / 2)
            for v in arguments["initial_form_bounds"]
            + arguments["initial_phase_bounds"]
        ]
        + [float(arguments["initial_clock_rate"]), 0.0]
    )
    initial_phase = state[18:36].copy()
    for (start, end), jump, step, average in zip(
        report.aperture_windows,
        report.phase_event_increments,
        report.steps,
        report.averaged_readout_bounds,
    ):
        state[22] += float(jump)
        state[23] -= float(jump)
        initial_integral = state[37]
        solution = solve_ivp(
            flow,
            (float(start), float(end)),
            state,
            method="DOP853",
            rtol=1e-12,
            atol=1e-15,
        )
        assert solution.success
        state = solution.y[:, -1]
        for bounds, actual in zip(step.endpoint[:36], state[:36]):
            assert float(bounds.lo) <= actual <= float(bounds.hi)
        expected_average = (state[37] - initial_integral) / float(end - start)
        assert float(average.lo) <= expected_average <= float(average.hi)
        # Constant rate derivative has an exact primitive solution; test it
        # separately rather than requiring a floating ODE to hit an ulp-wide box.
        assert step.endpoint[36].contains(
            arguments["initial_clock_rate"] + arguments["clock_slope"] * end
        )
        assert float(step.endpoint[37].lo) <= state[37] <= float(step.endpoint[37].hi)
        assert step.picard_interior_margin > 0 and step.domain_lower_bounds == (Q(1),)
    assert np.linalg.norm(state[18:22] - initial_phase[:4]) > 1e-8


@pytest.mark.parametrize(
    "field,value",
    (
        ("initial_form_bounds", ((0, 0),) * 17),
        ("initial_phase_bounds", ((0, 0),) * 19),
        ("initial_form_bounds", ((True, 1),) * 18),
        ("initial_phase_bounds", ((float("nan"), 1),) * 18),
        ("initial_form_bounds", (I(0),) * 18),
        ("initial_phase_bounds", ((1, 0),) * 18),
        ("phase_increments", (Q(1, 2), Q(1, 4))),
        ("phase_increments", (Q(-1, 10**400), Q(1, 2))),
        ("phase_increments", (0, Q(1001, 1000))),
        ("phase_increments", (True, 1)),
        ("phase_increments", (0, np.bool_(True))),
        ("phase_increments", (0, 0, 0)),
        ("phase_increments", (float("inf"), 1)),
        ("probe_duration", 0),
        ("probe_duration", -1),
        ("probe_duration", True),
        ("probe_duration", float("nan")),
        ("initial_clock_rate", 0),
        ("initial_clock_rate", False),
        ("initial_clock_rate", np.bool_(True)),
        ("initial_clock_rate", float("inf")),
        ("initial_clock_rate", 3),
        ("clock_slope", Q(-5, 4)),
        ("clock_slope", -2),
        ("clock_slope", True),
        ("clock_slope", float("nan")),
        ("order", True),
        ("order", Q(4)),
        ("order", np.int64(4)),
        ("order", 0),
        ("order", 17),
    ),
)
def test_invalid_primitives_rejected_before_flow(monkeypatch, field, value):
    arguments = _arguments()
    arguments[field] = value
    monkeypatch.setattr(
        owner,
        "_full_sine_field",
        lambda *args: pytest.fail("invalid primitive reached flow"),
    )
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_aperture_readout(**arguments)


@pytest.mark.parametrize(
    "horizon,rate", ((Q(2), Q(1, 8)), (Q(1, 16), Q(1, 2**400)), (Q(1, 2**400), Q(1)))
)
def test_exact_clock_and_duration_admission_survives_subgrid_materialization(
    horizon, rate
):
    result = owner.bound_sine_aperture_readout(
        initial_form_bounds=((0, 0),) * 18,
        initial_phase_bounds=((0, 0),) * 18,
        phase_increments=(0, 0),
        probe_duration=horizon,
        initial_clock_rate=rate,
        clock_slope=0,
        order=1,
    )
    assert result.admitted
    assert result.probe_duration == horizon and result.initial_clock_rate == rate
    assert result.clock_positivity_certified and result.clock_rate_bounds == (
        rate,
        rate,
    )
    assert result.cumulative_structural_exposures[-1] == 2 * horizon * rate
    assert all(average.contains(0) for average in result.averaged_readout_bounds)
    assert all(step.domain_lower_bounds == (Q(1),) for step in result.steps)
    if rate < Q(1, 2**128):
        assert result.initial_augmented_box[36].contains(0)
    if horizon < Q(1, 2**128):
        assert I(horizon / 3).contains(0)


def test_serialization_preserves_exact_clock_full_prefix_and_increment(
    polynomial_readout,
):
    data = json_loads(json.dumps(polynomial_readout.to_dict(), allow_nan=False))
    assert data["schema"] == "tnfr.sine-aperture-readout.v1"
    report = data["report"]
    assert report["clock_slope"] == {"numerator": 1, "denominator": 7}
    assert report["probe_duration"] == {"numerator": 1, "denominator": 5}
    assert len(report["steps"]) == 4
    assert len(report["final_state_bounds"]) == 38
    assert all(
        len(step["increment"]) == len(step["series"]) == 38 for step in report["steps"]
    )
    assert report["failed_window_index"] is report["failed_tube"] is None


def test_forward_acquisition_cannot_consume_inverse_or_previous_readout(monkeypatch):
    from tnfr.physics import relational_sine_aperture_inference as inverse
    from tnfr.physics import relational_sine_two_port_readout as previous

    def forbidden(*args, **kwargs):
        pytest.fail("forward average called an inverse or previous producer")

    monkeypatch.setattr(inverse, "infer_sine_geometry_gain_clock_aperture", forbidden)
    monkeypatch.setattr(previous, "bound_sine_two_port_readout", forbidden)
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_structural_field)
    assert owner.bound_sine_aperture_readout(**_arguments()).admitted
