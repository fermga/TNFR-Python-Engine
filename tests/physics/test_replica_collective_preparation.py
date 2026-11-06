"""Static controls before the one fixed complete-field preparation attempt.

Constructed states and one scalar shared-kernel fixture are evaluated here.
These controls do not run the ten-node acquisition trajectory. The mocked
worker failure is bookkeeping evidence only, never a trajectory certificate.
"""

from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics._validated_taylor import (
    reconstruct_box_taylor_arithmetic,
    validated_box_taylor_step,
)
from tnfr.research import replica_collective_preparation as protocol
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads

# Independent static oracle allowances, fixed before the first evaluation.
ORACLE_DIGITS = 140
ORACLE_TOLERANCE = Q(1, 10**110)
POINT_WIDTH_CAP = Q(1, 10**30)


@pytest.fixture(scope="module")
def source():
    return protocol.declared_source()


def _context():
    context = mpmath.mp.clone()
    context.dps = ORACLE_DIGITS
    return context


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _edges():
    return tuple(
        (left, right)
        for left in range(10)
        for right in range(left + 1, 10)
        if (left // 2 - right // 2) % 5 in (1, 4)
    )


def _encloses(context, interval, expected):
    tolerance = _mp(context, ORACLE_TOLERANCE)
    assert interval.width < POINT_WIDTH_CAP
    assert _mp(context, interval.lo) - tolerance <= expected
    assert expected <= _mp(context, interval.hi) + tolerance


def _state_with_means(means, *, delta=Q(1, 8)):
    forms = tuple(Q(sign, 8) for _ in range(5) for sign in (1, -1))
    phases = tuple(mean + sign * delta for mean in means for sign in (1, -1))
    return forms + phases


def _read(path):
    return decode_exact_tree(json_loads(path.read_text(encoding="utf-8")))


def test_exact_source_redistribution_and_uncertainty_have_distinct_storage_scopes(
    source,
):
    edges = _edges()
    assert len(edges) == 20 and edges == protocol.EDGES
    assert all(sum(node in edge for edge in edges) == 4 for node in range(10))
    u = (Q(1), Q(5, 4), Q(3, 2), Q(7, 4), Q(2))
    delta = (Q(1, 100), Q(3, 250), Q(7, 500), Q(2, 125), Q(1, 50))
    profile = (-1, -1, 0, 1, 1)
    c, scale = Q(570, 287), Q(17, 287)
    expected = tuple(
        c * profile[i] + sign * scale * u[i] for i in range(5) for sign in (1, -1)
    )
    expected += tuple(sign * value for value in delta for sign in (1, -1))
    assert source.nominal == expected
    assert source.original_nominal[:10] == tuple(
        sign * value for value in u for sign in (1, -1)
    )
    assert source.original_nominal[10:] == expected[10:]
    assert 12 * c**2 + Q(95, 2) * scale**2 == Q(95, 2)
    assert sum(expected[:10]) == sum(expected[10:]) == 0
    for center in (source.nominal, source.original_nominal):
        form_energy = sum(((center[a] - center[b]) ** 2 / 2 for a, b in edges), Q(0))
        assert (
            form_energy
            == source.nominal_form_storage
            == source.original_form_storage
            == Q(95, 2)
        )
    assert source.intended_coordinate_error == Q(1, 2**20)
    assert source.numerical_enclosure_allowance == Q(1, 2**128)
    assert (
        source.proof_coordinate_error
        == source.intended_coordinate_error + source.numerical_enclosure_allowance
    )
    for center, box in (
        (source.nominal, source.initial_box),
        (source.original_nominal, source.original_box),
    ):
        for value, bound in zip(center, box):
            assert bound.lo <= value - Q(1, 2**20)
            assert bound.hi >= value + Q(1, 2**20)
            assert value - source.proof_coordinate_error <= bound.lo
            assert bound.hi <= value + source.proof_coordinate_error
    # An allowed independent error can change storage: equality belongs only
    # to the two prescribed centers, not to all pairs of error-box members.
    changed = list(source.nominal)
    changed[0] += source.intended_coordinate_error
    changed_form = sum(((changed[a] - changed[b]) ** 2 / 2 for a, b in edges), Q(0))
    assert changed_form != Q(95, 2)
    assert not source.memberwise_equal_storage_claimed
    context = _context()
    for center, upper in (
        (source.nominal, source.storage_upper_bound),
        (source.original_nominal, source.original_storage_upper_bound),
    ):
        values = tuple(_mp(context, value) for value in center)
        energy = sum(
            (values[a] - values[b]) ** 2 / 2
            + 1
            - context.cos(values[10 + a] - values[10 + b])
            for a, b in edges
        )
        assert _mp(context, Q(95, 2)) < energy < _mp(context, upper) < 48
    assert source.nominal_storage_equality_certified and source.enclosure_admitted
    assert source.global_chart_and_nontip_certified and source.certified
    assert source.internal_radius_squared_future_upper < 1
    assert source.internal_radius_future_lower > Q(1, 160)
    assert source.raw_gap_upper_bound < 13
    assert (
        source.early_exclusion_certified and source.original_control_exclusion_certified
    )
    assert source.original_control_mean_norm_squared_upper < 1
    pi = pi_interval()
    assert source.early_raw_gap_upper_bound < 3 * pi.lo / 2
    expected_activity = 5 * context.sin(_mp(context, Q(1, 100))) / (1920 * context.pi)
    assert 0 < _mp(context, source.target_activity_lower_bound) <= expected_activity
    assert source.clock == "original structural t; both full rows retain 1/pi"
    assert not source.empirical_validation
    with pytest.raises(FrozenInstanceError):
        source.certified = False


def test_complete_twenty_rows_match_independent_original_time_oracle(source):
    context = _context()
    field = protocol.complete_field()
    epsilon = Q(1, 2**20)
    companion = tuple(
        value + (epsilon / 2 if i % 2 else -epsilon / 2)
        for i, value in enumerate(source.nominal)
    )
    for state in (source.nominal, companion):
        values = tuple(_mp(context, value) for value in state)
        actual = field(tuple(I(value) for value in state))
        assert len(actual) == 20
        expected = []
        for node in range(10):
            neighbors = tuple(
                other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
            )
            assert len(neighbors) == 4
            expected.append(
                sum(
                    context.sin(values[10 + other] - values[10 + node])
                    for other in neighbors
                )
                / (4 * context.pi)
            )
        for node in range(10):
            neighbors = tuple(
                other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
            )
            expected.append(
                sum(values[node] - values[other] for other in neighbors)
                / (4 * context.pi)
            )
        for bound, value in zip(actual, expected):
            _encloses(context, bound, value)
        assert abs(sum(expected[:10])) < _mp(context, ORACLE_TOLERANCE)
        assert abs(sum(expected[10:])) < _mp(context, ORACLE_TOLERANCE)
        for pair in range(5):
            # The exact internal phase row is u/pi even with nonuniform X.
            u = (values[2 * pair] - values[2 * pair + 1]) / 2
            expected_internal = (expected[10 + 2 * pair] - expected[11 + 2 * pair]) / 2
            assert abs(expected_internal - u / context.pi) < _mp(
                context, ORACLE_TOLERANCE
            )


def test_whole_tube_winding_uses_continuous_means_and_a_separate_pair_chart():
    pi = pi_interval()
    means = tuple(2 * (i - 2) * pi / 5 for i in range(5))
    state = _state_with_means(means)
    report = protocol.observe_tube(state)
    assert report.selected_shifts == (0, 0, 0, 0, 1)
    assert report.mean_geometry_certified and report.box_chart_certified
    assert report.box_physical_geometry_certified and report.raw_gap_range_certified
    assert not report.source_membership_inferred
    assert not report.activity_inferred_for_arbitrary_box
    assert all(
        value.lo >= Q(1, 100) and value.hi <= pi.lo / 2 - Q(1, 100)
        for value in report.shifted_gaps
    )
    assert protocol.observe_tube(
        _state_with_means(tuple(mean + Q(7, 3) for mean in means))
    ).mean_geometry_certified
    assert not protocol.observe_tube(
        _state_with_means(tuple(-mean for mean in means))
    ).mean_geometry_certified
    assert not protocol.observe_tube(
        _state_with_means((Q(0),) * 5)
    ).mean_geometry_certified
    # Adding a complete turn to a single pair preserves its physical phase;
    # the explicit carried integer branches change, not the evolution state.
    shifted = tuple(mean + (2 * pi if i == 0 else 0) for i, mean in enumerate(means))
    translated = protocol.observe_tube(_state_with_means(shifted))
    assert translated.mean_geometry_certified
    assert translated.selected_shifts == (1, 0, 0, 0, 0)
    outside = protocol.observe_tube(_state_with_means(means, delta=Q(2)))
    assert outside.mean_geometry_certified
    assert (
        not outside.box_chart_certified and not outside.box_physical_geometry_certified
    )
    assert (
        not outside.source_membership_inferred
        and not outside.activity_inferred_for_arbitrary_box
    )
    # Broadening one supplied mean across a target boundary must reject a
    # whole-tube certificate even if its original center was a target.
    broad = list(state)
    broad[10] = broad[10] + I(-2, 2)
    broad[11] = broad[11] + I(-2, 2)
    assert not protocol.observe_tube(broad).mean_geometry_certified
    with pytest.raises(FrozenInstanceError):
        report.mean_geometry_certified = False


def test_observer_admits_all_twenty_exact_inputs_before_phase_work(monkeypatch):
    def forbidden():
        raise AssertionError("phase work preceded complete raw admission")

    monkeypatch.setattr(protocol, "pi_interval", forbidden)
    for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        with pytest.raises((TypeError, ValueError)):
            protocol.observe_tube((Q(0),) * 19 + (bad,))
    for size in (0, 19, 21):
        with pytest.raises((TypeError, ValueError)):
            protocol.observe_tube((Q(0),) * size)
    forged = I(0)
    object.__setattr__(forged, "lo", False)
    with pytest.raises((TypeError, ValueError)):
        protocol.observe_tube((Q(0),) * 19 + (forged,))


def test_residence_requires_four_contiguous_cells_and_survives_later_failure():
    assert protocol.residence_windows(()) == ()
    assert protocol.residence_windows((True,) * 3) == ()
    assert protocol.residence_windows((True, True, False, True, True, True)) == ()
    assert protocol.residence_windows((False,) * 2 + (True,) * 5) == (
        (Q(1, 4), Q(3, 4)),
        (Q(3, 8), Q(7, 8)),
    )
    incomplete = protocol._outcome((True,) * 4, "synthetic_later_inclusion_failure")
    assert incomplete["residence_status"] == "certified_residence"
    assert incomplete["first_certified_window"] == (Q(0), Q(1, 2))
    assert not incomplete["integration_complete"]
    assert incomplete["completion_status"] == "integration_incomplete"
    assert not incomplete["absence_claimed"] and not incomplete["empirical_validation"]
    negative = protocol._outcome((False,) * 48, None)
    assert (
        negative["integration_complete"]
        and negative["residence_status"] == "residence_not_certified"
    )
    assert not negative["absence_claimed"]
    for bad in ((True,) * 49, (1,), (None,)):
        with pytest.raises(ValueError):
            protocol.residence_windows(bad)


def test_runner_retains_first_failure_and_refuses_an_unrecorded_retry(
    source, tmp_path, monkeypatch
):
    calls = []
    sentinel = object()

    def fail_step(box, duration, field, domain, *, order, time):
        calls.append((box, duration, field, order, time))
        assert box == source.initial_box
        assert duration == Q(1, 8) and order == 8 and time == 0
        assert field is sentinel
        assert domain((I(-1000, 1000),) * 20) == (Q(1),)
        return None, box, "constructed_inclusion_failure"

    monkeypatch.setattr(protocol, "declared_source", lambda: source)
    monkeypatch.setattr(protocol, "complete_field", lambda: sentinel)
    monkeypatch.setattr(protocol, "validated_box_taylor_step", fail_step)
    monkeypatch.setattr(protocol, "monotonic", lambda: 0.0)
    result = protocol.run_trajectory(tmp_path)
    assert len(calls) == 1
    assert result["cells_completed"] == 0 and result["completed_through"] == 0
    assert result["failure_reason"] == "constructed_inclusion_failure"
    assert result["residence_status"] == "residence_not_certified"
    assert not result["integration_complete"] and not result["absence_claimed"]
    assert not tuple(tmp_path.glob("cell-*.json"))
    assert _read(tmp_path / "source.json") == decode_exact_tree(asdict(source))
    failure = _read(tmp_path / "failure.json")
    assert failure["cell_index"] == 0 and failure["time"] == 0
    assert failure["initial_box"] == tuple(
        asdict(value) for value in source.initial_box
    )
    assert failure["failed_tube"] == failure["initial_box"]
    assert _read(tmp_path / "trajectory-result.json") == decode_exact_tree(result)
    attempt = _read(tmp_path / "attempt.json")
    assert (
        attempt["horizon"],
        attempt["step"],
        attempt["order"],
        attempt["cell_count"],
    ) == (Q(6), Q(1, 8), 8, 48)
    assert attempt["wall_seconds"] == 900 and attempt["residence_cells"] == 4
    assert attempt["margin"] == Q(1, 100)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def forbidden_source():
        raise AssertionError("an existing attempt must refuse before re-admission")

    monkeypatch.setattr(protocol, "declared_source", forbidden_source)
    with pytest.raises(FileExistsError):
        protocol.run_trajectory(tmp_path)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_shared_box_step_retains_an_independent_constant_field_source_interval():
    # A separate scalar identity checks the shared step/retained arithmetic;
    # it is not an acquisition pilot or a shortened ten-node trajectory.
    initial = (I(-1, 1),)

    def flow(state):
        return (0 * state[0] + 3,)

    step, failed, reason = validated_box_taylor_step(
        initial, Q(1, 8), flow, lambda tube: (Q(1),), order=3
    )
    assert failed is None and reason is None and step is not None
    assert step.initial_box == initial and step.order == 3
    assert step.duration == Q(1, 8) and step.time == 0
    assert step.picard_interior_margin > 0 and step.domain_lower_bounds == (Q(1),)
    assert step.increment == (I(Q(3, 8)),)
    assert step.endpoint == (I(Q(-5, 8), Q(11, 8)),)
    assert step.local_remainder_bounds == (I(0),)
    assert step.tube[0].lo < -1 and step.tube[0].hi > Q(11, 8)
    rebuilt = reconstruct_box_taylor_arithmetic(
        step.initial_box,
        step.tube,
        step.series,
        step.local_remainder_bounds,
        step.duration,
        order=step.order,
    )
    assert rebuilt == (step.increment, step.endpoint)
    assert step.method == "direct_source_box_Picard_Taylor_dyadic128_v1"
