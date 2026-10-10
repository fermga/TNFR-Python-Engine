"""Independent algebra and admission controls for a carried observation word.

Endpoint records solve a static linear identity; they are not trajectories or
reserved responses. Source builders and the reserved noise assessment remain
blocked, including while the source wrapper's wiring is exercised.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from types import SimpleNamespace

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_finite_observation as owner
from tnfr.physics import _organized_finite_observation_source as source
from tnfr.physics._organized_outer_winding import _derive_organized_outer_winding


_H, _EPSILON, _NOISE = Q(1, 10**5), Q(1, 10), Q(1, 10**15)
_TOTAL = 15 * (1 + _H)
_SHAPE = (5, 3, 2, 5)


@pytest.fixture(scope="module", autouse=True)
def no_reserved_assessment_or_trajectory():
    pure_bound = owner._bound_organized_finite_observation

    def synthetic_bound(*, raw_readout_error):
        if raw_readout_error == Q(1, 10**14):
            pytest.fail("the reserved finite-observation corner must not run in tests")
        return pure_bound(raw_readout_error=raw_readout_error)

    def forbidden(*args, **kwargs):
        pytest.fail("an actual reserved source builder was executed")

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_bound_organized_finite_observation", synthetic_bound)
        for name in (
            "_assess_sine_recovery_resources",
            "_derive_organized_outer_resource_bounds",
            "_bound_organized_finite_observation",
        ):
            patch.setattr(source, name, forbidden)
        yield


@pytest.fixture(scope="module")
def arithmetic():
    mp = mpmath.mp.clone()
    mp.dps = 110
    laplacian = tuple(
        tuple(Q(2 if i == j else -1 if (i - j) % 5 in (1, 4) else 0) for j in range(5))
        for i in range(5)
    )
    endpoint_inverse = exact_matrix_inverse(
        tuple(
            tuple(Q(i == j) + _H * value / 8 for j, value in enumerate(row))
            for i, row in enumerate(laplacian)
        )
    )
    pseudoinverse = exact_matrix_inverse(
        tuple(tuple(value + Q(1, 5) for value in row) for row in laplacian)
    )
    pseudoinverse = tuple(
        tuple(value - Q(1, 5) for value in row) for row in pseudoinverse
    )
    return mp, laplacian, endpoint_inverse, pseudoinverse


@pytest.fixture(scope="module")
def protocol():
    return owner._bound_organized_finite_observation(raw_readout_error=_NOISE)


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _q(value):
    return Q(str(value))


def _action(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _infer(readings, *, noise=_NOISE):
    return owner._infer_finite_observation_winding(
        readings=readings, raw_readout_error=noise
    )


def _empty_records():
    return [
        [[[Q(0) for _ in range(5)] for _ in range(2)] for _ in range(3)]
        for _ in range(5)
    ]


def _state(mp, winding):
    if winding:
        gaps = tuple(
            winding * (2 * mp.pi / 5 + _mp(mp, perturbation))
            for perturbation in (Q(1, 100), -Q(3, 200), Q(1, 200), Q(1, 50), -Q(1, 50))
        )
    else:
        gaps = tuple(
            _mp(mp, value) for value in (Q(2, 5), -Q(1, 5), Q(1, 10), -Q(1, 2), Q(1, 5))
        )
    origins = [mp.mpf(0)]
    for gap in gaps[:-1]:
        origins.append(origins[-1] + gap)
    residuals = tuple(
        mp.mpf(c + 1) * (3 * (j - 4) ** 2 - 20) / (500 * 10**6)
        for c in range(5)
        for j in range(9)
    )
    phase = tuple(
        2 * mp.pi * (j - 4) / 9 + origins[c] + residuals[9 * c + j]
        for c in range(5)
        for j in range(9)
    )
    return phase, gaps, residuals


def _algebraic_records(arithmetic, winding, *, noisy=False):
    """Solve endpoint equations independently, without asserting attainability."""
    mp, laplacian, inverse, _ = arithmetic
    phase, gaps, residuals = _state(mp, winding)
    gamma, h, epsilon = 1 / (1023 * mp.pi), _mp(mp, _H), _mp(mp, _EPSILON)
    records, expected, cosine, sine = [], [], [], []
    for c in range(5):
        receiver = (c + 1) % 5
        edge_angles = tuple(
            phase[9 * receiver + j] - phase[9 * c + j] for j in range(9)
        )
        cosine.append(sum(mp.cos(angle) for angle in edge_angles) / 9)
        sine.append(sum(mp.sin(angle) for angle in edge_angles) / 9)
        block, targets = [], []
        for plateau, level in enumerate((0, 1, -1)):
            shifted = tuple(
                value + level * epsilon * (int(i // 9 == c) - mp.mpf(1) / 5)
                for i, value in enumerate(phase)
            )
            currents = tuple(
                sum(
                    mp.sin(shifted[9 * ((child - 1) % 5) + j] - shifted[9 * child + j])
                    + mp.sin(
                        shifted[9 * ((child + 1) % 5) + j] - shifted[9 * child + j]
                    )
                    for j in range(9)
                )
                / 9
                for child in range(5)
            )
            start = tuple(
                Q(child - 2, 5000) + Q(c - plateau, 100000) for child in range(5)
            )
            gradient = _action(laplacian, start)
            rhs = tuple(
                value - _H * diffusion / 8 + _q(gamma * h * current / 4)
                for value, diffusion, current in zip(start, gradient, currents)
            )
            end = _action(inverse, rhs)
            targets.append(gamma * h * currents[receiver] / 4)
            endpoints = [list(start), list(end)]
            if noisy:
                # Saturate all six nonzero reading coefficients of D together.
                direction = (-1) ** (c + plateau)
                for endpoint in range(2):
                    endpoints[endpoint][receiver] += (
                        direction * (2 * endpoint - 1) * _NOISE
                    )
                    for neighbor in ((receiver - 1) % 5, (receiver + 1) % 5):
                        endpoints[endpoint][neighbor] -= direction * _NOISE
            block.append(endpoints)
        records.append(block)
        expected.append(tuple(targets))
    return records, gaps, residuals, tuple(cosine), tuple(sine), tuple(expected)


def _moment_records(arithmetic, cosine, sine):
    """Receiver-only algebra for deliberately nonrealizable cycle controls."""
    mp, _, inverse, _ = arithmetic
    gamma, h, epsilon = 1 / (1023 * mp.pi), _mp(mp, _H), _mp(mp, _EPSILON)
    records = _empty_records()
    for c in range(5):
        for plateau, sign in ((1, 1), (2, -1)):
            rhs = [Q(0)] * 5
            rhs[(c + 1) % 5] = _q(
                gamma
                * h
                * (sign * mp.sin(epsilon) * cosine[c] + (1 - mp.cos(epsilon)) * sine[c])
                / 4
            )
            records[c][plateau][1] = list(_action(inverse, rhs))
    return records


@pytest.mark.parametrize("winding", (-1, 0, 1))
@pytest.mark.parametrize("noisy", (False, True))
def test_independent_endpoint_algebra_retains_all_three_initial_sectors(
    arithmetic, winding, noisy
):
    mp, laplacian, _, _ = arithmetic
    records, gaps, residuals, cosine, sine, expected = _algebraic_records(
        arithmetic, winding, noisy=noisy
    )
    assert max(abs(value) for value in residuals) < mp.mpf("1e-6")
    assert max(abs(value) for value in gaps) < mp.mpf("1.4")
    for c in range(5):
        assert abs(sum(residuals[9 * c : 9 * c + 9])) < mp.mpf("1e-100")
    result = _infer(records)
    assert result.candidate_windings == (winding,)
    assert result.identified_winding == winding
    assert result.status == "conditionally_identified"
    assert result.candidate_meaning == "necessary_compatibility_not_joint_realizability"
    for c, block in enumerate(records):
        receiver = (c + 1) % 5
        for plateau, (start, end) in enumerate(block):
            corrected = (
                end[receiver]
                - start[receiver]
                + _H
                * _action(laplacian, tuple(a + b for a, b in zip(start, end)))[receiver]
                / 8
            )
            assert result.corrected_plateau_increments[c][plateau] == corrected
            reading_error = _mp(mp, corrected) - expected[c][plateau]
            target = (
                (-1) ** (c + plateau) * _mp(mp, (2 + _H / 2) * _NOISE) if noisy else 0
            )
            assert abs(reading_error - target) < mp.mpf("1e-100")
        _contains(mp, result.cosine_bounds[c], cosine[c])
        _contains(mp, result.sine_bounds[c], sine[c])
        _contains(mp, result.child_origin_angle_bounds[c], gaps[c])
    _contains(mp, result.cycle_sum_bounds, 2 * mp.pi * winding)
    assert result.odd_reading_differences == tuple(
        row[1] - row[2] for row in result.corrected_plateau_increments
    )
    assert result.even_reading_combinations == tuple(
        row[1] + row[2] - 2 * row[0] for row in result.corrected_plateau_increments
    )


def test_diffusion_compensation_is_necessary_for_nonuniform_endpoint_forms(arithmetic):
    records, _, _, _, _, expected = _algebraic_records(arithmetic, 1)
    mp = arithmetic[0]
    start, end = records[0][0]
    naive = _mp(mp, end[1] - start[1])
    assert abs(naive - expected[0][0]) > mp.mpf("1e-15")
    result = _infer(records)
    assert abs(
        _mp(mp, result.corrected_plateau_increments[0][0]) - expected[0][0]
    ) < mp.mpf("1e-100")


def test_local_angles_must_close_the_complete_five_child_cycle(arithmetic):
    mp = arithmetic[0]
    angle = mp.mpf(1) / 5
    result = _infer(
        _moment_records(arithmetic, (mp.cos(angle),) * 5, (mp.sin(angle),) * 5)
    )
    assert all(result.resultant_compatible)
    _contains(mp, result.cycle_sum_bounds, mp.mpf(1))
    assert not result.candidate_windings and result.identified_winding is None
    assert result.status == "incompatible_records_or_premises"


def test_incompatible_annulus_and_missing_quadratures_do_not_mean_zero(arithmetic):
    mp = arithmetic[0]
    half = _infer(_moment_records(arithmetic, (mp.mpf(1) / 2,) * 5, (mp.mpf(0),) * 5))
    assert not any(half.resultant_compatible)
    for result in (half, _infer(_empty_records())):
        assert not result.candidate_windings and result.identified_winding is None
        assert result.status == "incompatible_records_or_premises"


def test_common_raw_offset_cancels_exactly_but_cannot_authenticate_impossible_forms(
    arithmetic,
):
    records = _algebraic_records(arithmetic, 1)[0]
    original = _infer(records)
    huge = Q(10**500) + Q(1, 10**500)
    translated = [
        [
            [[value + huge for value in endpoint] for endpoint in plateau]
            for plateau in block
        ]
        for block in records
    ]
    result = _infer(translated)
    assert result.corrected_plateau_increments == original.corrected_plateau_increments
    assert not any(result.raw_readings_compatible)
    assert result.identified_winding is None and not result.candidate_windings


def test_unavailable_uniform_protocol_never_assigns_identity(
    arithmetic, protocol, monkeypatch
):
    failed = replace(
        protocol,
        admitted=False,
        uniform_identification_certified=False,
        unresolved_conditions=("uniform_resolution_unavailable",),
    )
    monkeypatch.setattr(
        owner, "_bound_organized_finite_observation", lambda **kwargs: failed
    )
    result = _infer(_algebraic_records(arithmetic, -1)[0])
    assert result.candidate_windings == (-1,)
    assert result.identified_winding is None
    assert result.status == "unavailable_protocol_certificate"
    assert "uniform_resolution_unavailable" in result.unresolved_conditions


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("change", (-1, 1))
def test_every_nested_reading_dimension_has_exact_ordered_size(depth, change):
    records = _empty_records()
    parent = records
    for _ in range(depth):
        parent = parent[0]
    if change < 0:
        parent.pop()
    else:
        parent.append(deepcopy(parent[-1]))
    with pytest.raises(ValueError):
        _infer(records)


@pytest.mark.parametrize("depth", range(4))
@pytest.mark.parametrize("invalid", (None, "0", {0: 0}, {0, 1}))
def test_every_nested_reading_container_rejects_unordered_or_scalar_values(
    depth, invalid
):
    records = _empty_records()
    if depth == 0:
        records = invalid
    else:
        parent = records
        for _ in range(depth - 1):
            parent = parent[0]
        parent[0] = invalid
    with pytest.raises((TypeError, ValueError)):
        _infer(records)


@pytest.mark.parametrize("depth", range(4))
def test_oversized_generators_consume_only_one_excess_item(depth):
    records = _empty_records()
    container = records
    for _ in range(depth):
        container = container[0]
    item, consumed = container[0], []

    def endless():
        for index in range(_SHAPE[depth] + 2):
            consumed.append(index)
            if index > _SHAPE[depth]:
                pytest.fail("reader consumed beyond the first excess item")
            yield item

    if depth == 0:
        records = endless()
    else:
        parent = records
        for _ in range(depth - 1):
            parent = parent[0]
        parent[0] = endless()
    with pytest.raises(ValueError):
        _infer(records)
    assert len(consumed) == _SHAPE[depth] + 1


@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "0.0", 1j))
def test_raw_reading_scalars_reject_before_protocol_arithmetic(invalid, monkeypatch):
    records = _empty_records()
    records[4][2][1][4] = invalid

    def forbidden(**kwargs):
        pytest.fail("invalid primitive reached protocol arithmetic")

    monkeypatch.setattr(owner, "_bound_organized_finite_observation", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _infer(records)


@pytest.mark.parametrize("invalid", (True, -Q(1), float("nan"), float("inf"), "0", 1j))
def test_raw_error_allowance_requires_a_nonnegative_real(invalid):
    with pytest.raises((TypeError, ValueError)):
        owner._bound_organized_finite_observation(raw_readout_error=invalid)


def test_finite_word_error_ledger_and_cost_bounds_follow_independent_formulas(
    arithmetic, protocol
):
    mp = arithmetic[0]
    gamma, h, duration = 1 / (1023 * mp.pi), _mp(mp, _H), _mp(mp, _TOTAL)
    form = mp.mpf("0.01") + gamma * duration
    drift = 2 * gamma * mp.mpf("0.01") + 2 * gamma**2 * duration
    second = 4 * form + 2 * gamma + 4 * gamma**2 * form
    assert form <= _mp(mp, protocol.observation_form_upper_bound)
    assert drift <= _mp(mp, protocol.phase_residual_upper_bound)
    assert second <= _mp(mp, protocol.plateau_second_derivative_upper_bound)
    assert gamma * h * drift <= _mp(mp, protocol.plateau_transport_error_upper_bound)
    assert second * h**3 / 12 <= _mp(mp, protocol.plateau_trapezoid_error_upper_bound)
    zero = owner._bound_organized_finite_observation(raw_readout_error=Q(0))
    assert protocol.plateau_readout_error_upper_bound == (2 + _H / 2) * _NOISE
    assert (
        protocol.plateau_total_error_upper_bound - zero.plateau_total_error_upper_bound
        == (2 + _H / 2) * _NOISE
    )
    assert (
        protocol.odd_total_error_upper_bound
        == 2 * protocol.plateau_total_error_upper_bound
    )
    assert (
        protocol.even_total_error_upper_bound
        == 4 * protocol.plateau_total_error_upper_bound
    )
    assert protocol.total_elapsed_scaled_time == _TOTAL
    assert protocol.total_elapsed_original_time == Q(1024, 1023) * _TOTAL
    assert protocol.total_input_scaled_time == 15
    assert protocol.reading_shape == _SHAPE
    assert protocol.admitted and not protocol.unresolved_conditions
    assert 0 < _mp(mp, protocol.cycle_sum_width_upper_bound) < 2 * mp.pi
    assert all(value > 0 for _, value in protocol.sufficient_margins)
    assert (
        0
        < protocol.positive_supplied_work_lower
        == protocol.continuous_loss_lower - protocol.initial_storage_upper
    )
    assert protocol.continuous_loss_lower < protocol.continuous_loss_upper
    assert protocol.peak_form_storage_lower < protocol.peak_storage_upper
    assert protocol.positive_supplied_work_lower < protocol.absolute_work_upper
    offset = 24 * mp.mpf("0.1") / (5 * gamma)
    gradient = 2 * offset + 8 * form
    forcing = 108 * mp.mpf("0.1") / (5 * gamma)
    assert 180 * (offset + form) ** 2 + 180 <= _mp(mp, protocol.peak_storage_upper)
    assert 45 * duration * gradient**2 / 4 <= _mp(mp, protocol.continuous_loss_upper)
    assert 45 * 15 * gradient * forcing <= _mp(mp, protocol.absolute_work_upper)
    small, large = mp.mpf("0.1") - 4 * drift, mp.mpf("0.2") - 4 * drift
    assert (
        _mp(mp, protocol.continuous_loss_lower)
        <= 90 * (2 * small**2 + large**2) / gamma**2
    )
    assert _mp(mp, protocol.peak_form_storage_lower) <= 12 * large**2 / gamma**2
    assert (
        _mp(mp, protocol.outer_acute_gap_margin)
        <= mp.pi / 2 - mp.mpf("1.5") - mp.mpf("2e-6") - 2 * drift
    )


@pytest.mark.parametrize("child", (0, 2, 4))
@pytest.mark.parametrize("plateau", range(3))
def test_word_carries_cumulative_command_and_satisfies_linear_phase_row(
    arithmetic, child, plateau
):
    mp, laplacian, _, pseudoinverse = arithmetic
    start = (3 * child + plateau) * (1 + _H)
    direction = tuple(Q(int(i == child)) - Q(1, 5) for i in range(5))
    inverse_direction = _action(pseudoinverse, direction)
    assert _action(laplacian, inverse_direction) == direction
    gamma, level, amplitude = (
        1 / (1023 * mp.pi),
        (0, 1, -1)[plateau],
        (1, -2, 1)[plateau] * _EPSILON,
    )
    for position in (Q(0), Q(1, 4), Q(1, 2), Q(3, 4)):
        report = owner._evaluate_organized_finite_observation_word(
            elapsed=start + _H + position
        )
        assert report.stage == "pulse" and report.local_elapsed == position
        assert report.marked_child == child and report.receiver_child == (child + 1) % 5
        u, a = _mp(mp, position), _mp(mp, amplitude)
        shape, slope, curvature = 3 * u**2 - 2 * u**3, 6 * u - 6 * u**2, 6 - 12 * u
        form = tuple(
            4 * a * _mp(mp, value) * slope / gamma for value in inverse_direction
        )
        gradient = _action(laplacian, form)
        for i in range(5):
            phase = (level * _mp(mp, _EPSILON) + a * shape) * _mp(mp, direction[i])
            form_rate = 4 * a * _mp(mp, inverse_direction[i]) * curvature / gamma
            _contains(mp, report.phase_offset_bounds[i], phase)
            _contains(mp, report.form_offset_bounds[i], form[i])
            _contains(mp, report.form_tau_rate_bounds[i], form_rate)
            _contains(mp, report.form_input_bounds[i], form_rate + gradient[i] / 4)
            _contains(
                mp, report.phase_tau_rate_bounds[i], a * slope * _mp(mp, direction[i])
            )
            assert mp.almosteq(
                gamma * gradient[i] / 4, a * slope * _mp(mp, direction[i])
            )
        assert abs(level * _mp(mp, _EPSILON) + a * shape) <= _mp(mp, _EPSILON)


def test_all_switches_are_right_continuous_and_complete_command_returns_to_zero(
    protocol,
):
    for slot in range(15):
        start = slot * (1 + _H)
        plateau = owner._evaluate_organized_finite_observation_word(elapsed=start)
        before = owner._evaluate_organized_finite_observation_word(
            elapsed=start + _H / 2
        )
        pulse = owner._evaluate_organized_finite_observation_word(elapsed=start + _H)
        assert plateau.stage == before.stage == "plateau"
        assert pulse.stage == "pulse" and pulse.local_elapsed == 0
        assert plateau.phase_offset_bounds == before.phase_offset_bounds
        for name in (
            "form_offset_bounds",
            "form_input_bounds",
            "phase_tau_rate_bounds",
            "form_tau_rate_bounds",
        ):
            assert all(box.lo == box.hi == 0 for box in getattr(plateau, name))
        assert all(box.lo == box.hi == 0 for box in pulse.form_offset_bounds)
        assert any(box.abs_max > 0 for box in pulse.form_input_bounds)
        for left, right in zip(plateau.phase_offset_bounds, pulse.phase_offset_bounds):
            assert left.lo <= right.hi and right.lo <= left.hi
    terminal = owner._evaluate_organized_finite_observation_word(elapsed=_TOTAL)
    assert terminal.stage == "terminal"
    assert (
        terminal.block_index
        is terminal.plateau_index
        is terminal.marked_child
        is terminal.receiver_child
        is None
    )
    for name in (
        "phase_offset_bounds",
        "form_offset_bounds",
        "form_input_bounds",
        "phase_tau_rate_bounds",
        "form_tau_rate_bounds",
    ):
        assert all(box.lo == box.hi == 0 for box in getattr(terminal, name))
    assert protocol.positive_supplied_work_lower > 0


@pytest.mark.parametrize(
    "elapsed", (True, -Q(1), _TOTAL + 1, float("nan"), float("inf"), "1", 1j)
)
def test_word_rejects_invalid_clock_or_time_outside_declared_horizon(elapsed):
    with pytest.raises((TypeError, ValueError)):
        owner._evaluate_organized_finite_observation_word(elapsed=elapsed)


def test_detached_protocol_word_and_inference_are_immutable(arithmetic, protocol):
    with pytest.raises(FrozenInstanceError):
        protocol.admitted = False
    with pytest.raises(FrozenInstanceError):
        owner._evaluate_organized_finite_observation_word(elapsed=Q(0)).stage = "pulse"
    with pytest.raises(FrozenInstanceError):
        _infer(_algebraic_records(arithmetic, 0)[0]).identified_winding = 1


@pytest.fixture
def synthetic_sources(protocol):
    recovery = SimpleNamespace(
        recovery_certified=True,
        endpoint_radius=Q(1, 10**8),
        endpoint_form_margin=Q(1, 10**9),
        endpoint_phase_margin=Q(1, 10**9),
    )
    outer = SimpleNamespace(
        geometry=_derive_organized_outer_winding(),
        primitive_entry=SimpleNamespace(
            endpoint_form_bounds=(I(-Q(1, 1000), Q(1, 1000)),) * 5,
            capture=SimpleNamespace(phase_edge_gap_bounds=(I(Q(1), Q(6, 5)),) * 5),
        ),
        full_form_error_upper_bound=Q(1, 10**8),
        full_phase_error_upper_bound=Q(2, 10**8),
        recovered_child_norm_bound=Q(1, 10**8),
        primitive_law_and_clock_certified=True,
        fine_error_certified=True,
        child_winding_retained_during_transit=True,
        outer_cell_certified=True,
        conditional_capture_certified=True,
        control_conditional_capture_certified=True,
        attachment_budget_certified=True,
        capture_margin=Q(1, 100),
        control_capture_margin=Q(1, 20),
        attachment_work_margin=Q(100),
        initial_storage_margin=Q(200),
    )
    # Metadata-only stub: no calculation at the reserved noise is performed.
    fake_protocol = SimpleNamespace(**vars(protocol))
    fake_protocol.raw_readout_error_upper_bound = Q(1, 10**14)
    return recovery, outer, fake_protocol


@pytest.fixture
def assess(synthetic_sources, monkeypatch):
    def run(change=None):
        recovery, outer, protocol = deepcopy(synthetic_sources)
        if change is not None:
            change(recovery, outer, protocol)
        calls = []

        def builder(name, result):
            def build(**arguments):
                calls.append((name, arguments))
                return result

            return build

        monkeypatch.setattr(
            source, "_assess_sine_recovery_resources", builder("recovery", recovery)
        )
        monkeypatch.setattr(
            source, "_derive_organized_outer_resource_bounds", builder("outer", outer)
        )
        monkeypatch.setattr(
            source, "_bound_organized_finite_observation", builder("protocol", protocol)
        )
        return source._assess_organized_finite_observation(), calls

    return run


def test_source_wrapper_uses_only_mocked_acquisition_and_new_finite_protocol(assess):
    result, calls = assess()
    assert result.admitted and not result.unresolved_conditions
    assert [name for name, _ in calls] == ["recovery", "outer", "protocol"]
    assert calls[0][1] == dict(
        form_error_bound=Q(1, 10**6),
        phase_error_bound=Q(1, 10**6),
        quarter_dwell=Q(8 * 10**10),
        endpoint_radius=Q(1, 10**8),
        decay_power=80,
    )
    assert calls[1][1] == dict(recovered_radius=Q(1, 10**8))
    assert calls[2][1] == dict(raw_readout_error=Q(1, 10**14))
    assert (
        result.initial_form_upper
        == result.nominal_form_upper + result.outer.full_form_error_upper_bound
    )
    assert (
        result.initial_hidden_phase_upper == result.outer.full_phase_error_upper_bound
    )
    assert (
        result.initial_mean_gap_upper
        == max(
            box.abs_max
            for box in result.outer.primitive_entry.capture.phase_edge_gap_bounds
        )
        + 2 * result.initial_hidden_phase_upper
    )
    assert result.source_families == ("outer_zero", "outer_one")
    assert result.source_stage == "actual_end_of_outer_formation_before_first_plateau"


@pytest.mark.parametrize(
    "branch,field",
    (
        ("recovery", "recovery_certified"),
        ("outer", "primitive_law_and_clock_certified"),
        ("outer", "fine_error_certified"),
        ("outer", "child_winding_retained_during_transit"),
        ("outer", "outer_cell_certified"),
        ("outer", "conditional_capture_certified"),
        ("outer", "control_conditional_capture_certified"),
        ("outer", "attachment_budget_certified"),
        ("protocol", "admitted"),
    ),
)
@pytest.mark.parametrize("value", (False, 1))
def test_each_source_certificate_requires_true_not_truthiness(
    assess, branch, field, value
):
    def change(recovery, outer, protocol):
        setattr(
            dict(recovery=recovery, outer=outer, protocol=protocol)[branch],
            field,
            value,
        )

    result, _ = assess(change)
    assert not result.admitted and result.unresolved_conditions


@pytest.mark.parametrize(
    "field",
    (
        "endpoint_form_margin",
        "endpoint_phase_margin",
        "capture_margin",
        "control_capture_margin",
        "attachment_work_margin",
        "initial_storage_margin",
    ),
)
@pytest.mark.parametrize("value", (Q(0), Q(-1), None))
def test_every_numeric_history_margin_is_required_and_positive(assess, field, value):
    def change(recovery, outer, protocol):
        setattr(recovery if field.startswith("endpoint_") else outer, field, value)

    result, _ = assess(change)
    assert not result.admitted and result.unresolved_conditions


@pytest.mark.parametrize(
    "field,value,condition",
    (
        ("full_form_error_upper_bound", Q(1, 100), "outer_one_form_in_prior"),
        (
            "full_phase_error_upper_bound",
            Q(1, 10**6),
            "orthogonal_hidden_phase_in_prior",
        ),
        ("recovered_child_norm_bound", Q(1, 10**7), "recovered_radius_matches"),
    ),
)
def test_source_bounds_are_read_before_class_inclusion(assess, field, value, condition):
    result, _ = assess(lambda recovery, outer, protocol: setattr(outer, field, value))
    assert not result.admitted and condition in result.unresolved_conditions


@pytest.mark.parametrize(
    "field,value",
    (
        ("clock_factor", Q(1, 2)),
        ("phase_domain", "acute"),
        ("held_capacities", (Q(2),) * 45),
    ),
)
def test_source_law_must_match_the_complete_quarter_model(assess, field, value):
    def change(recovery, outer, protocol):
        outer.geometry = replace(outer.geometry, **{field: value})

    result, _ = assess(change)
    assert (
        not result.admitted and "complete_quarter_law" in result.unresolved_conditions
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("pulse_duration", Q(2)),
        ("raw_readout_error_upper_bound", _NOISE),
        ("pulse_multipliers", (1, -1, 1)),
        ("marked_child_order", (1, 0, 2, 3, 4)),
        ("plateau_command_multipliers", (0, -1, 1)),
    ),
)
def test_cached_protocol_verdict_cannot_replace_the_fixed_recipe(assess, field, value):
    result, _ = assess(
        lambda recovery, outer, protocol: setattr(protocol, field, value)
    )
    assert (
        not result.admitted
        and "fixed_finite_observation_protocol" in result.unresolved_conditions
    )


def test_missing_source_edge_cannot_be_replaced_by_cached_success(assess):
    def change(recovery, outer, protocol):
        outer.geometry = replace(
            outer.geometry, weighted_edges=outer.geometry.weighted_edges[:-1]
        )

    result, _ = assess(change)
    assert (
        not result.admitted
        and "precontact_product_support" in result.unresolved_conditions
    )


@pytest.mark.parametrize("branch", ("forms", "gaps"))
@pytest.mark.parametrize(
    "boxes", ((I(Q(1, 100)),) * 4, (SimpleNamespace(abs_max=Q(1, 100)),) * 5)
)
def test_source_interval_primitives_cannot_be_replaced_by_cached_bounds(
    assess, branch, boxes
):
    def change(recovery, outer, protocol):
        if branch == "forms":
            outer.primitive_entry.endpoint_form_bounds = boxes
        else:
            outer.primitive_entry.capture.phase_edge_gap_bounds = boxes

    with pytest.raises((TypeError, ValueError)):
        assess(change)


@pytest.mark.parametrize(
    "field",
    ("full_form_error_upper_bound", "full_phase_error_upper_bound", "capture_margin"),
)
@pytest.mark.parametrize("value", (True, -Q(1), float("inf")))
def test_source_error_and_margin_scalars_reject_invalid_values(assess, field, value):
    def change(recovery, outer, protocol):
        setattr(outer, field, value)

    if field == "capture_margin" and value == -Q(1):
        result, _ = assess(change)
        assert not result.admitted
    else:
        with pytest.raises((TypeError, ValueError)):
            assess(change)
