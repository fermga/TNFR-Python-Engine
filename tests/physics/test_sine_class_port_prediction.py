"""Independent causal-kernel algebra and unreserved predictor admission controls."""

import subprocess
from dataclasses import asdict, replace
from decimal import Decimal
from fractions import Fraction as Q
from itertools import product
from math import factorial

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import _sine_class_port_prediction as owner


@pytest.fixture(scope="module", autouse=True)
def no_old_coefficient_or_response_execution():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_flow,
        _sine_formed_contact,
        relational_sine_class_comparison_readout,
        relational_sine_class_cubic_response,
        relational_sine_class_readout,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("port controls must not call old coefficients or full responses")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (_sine_flow, ("_full_sine_field",)),
            (_sine_formed_contact, ("_unprobed_handoff",)),
            (
                relational_sine_class_cubic_response,
                (
                    "_class_cubic_coefficients",
                    "_time_coefficients",
                    "_coefficient_segment",
                    "_linear_variation",
                ),
            ),
            (relational_sine_class_readout, ("bound_sine_class_four_history_readout",)),
            (
                relational_sine_class_comparison_readout,
                ("bound_sine_class_comparison_readout",),
            ),
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (subprocess, ("run", "Popen", "check_output")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


def _graph():
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    return edges, degrees


@pytest.fixture(scope="module")
def synthetic_descriptor():
    descriptor = owner._derive_collective_interface(2)
    gamma = I(Q(1, 128))
    sines = (I(Q(1, 4)),) * 9 + (I(Q(1, 2)),) * 9 + (I(Q(1, 4)),) * 9 + (I(0),) * 2
    cosines = (I(Q(3, 4)),) * 9 + (I(Q(1, 4)),) * 9 + (I(Q(3, 4)),) * 9 + (I(1),) * 2
    parameters = replace(
        descriptor.parameter_bounds,
        gamma=gamma,
        eta=gamma**2,
        edge_sines=sines,
        edge_cosines=cosines,
    )
    return replace(
        descriptor,
        parameter_bounds=parameters,
        quadratic_edge_scalar_bounds=tuple(-gamma * sine / 2 for sine in sines),
        cubic_edge_scalar_bounds=tuple(-gamma * cosine / 6 for cosine in cosines),
    )


def _direct_linear(state, descriptor):
    edges, degrees = _graph()
    gamma = descriptor.parameter_bounds.gamma.lo
    result = [Q(0)] * 54
    for edge, cosine in zip(edges, descriptor.parameter_bounds.edge_cosines):
        i, j = edge
        dx, dy = state[j] - state[i], state[27 + j] - state[27 + i]
        flow = dx + gamma * cosine.lo * dy
        result[i] += flow / degrees[i]
        result[j] -= flow / degrees[j]
        result[27 + i] -= gamma * dx / degrees[i]
        result[27 + j] += gamma * dx / degrees[j]
    return tuple(result)


def _direct_coefficients(descriptor, impulse, order):
    """Low-order exact full-edge Taylor substitution, independent of G/K/beta."""
    edges, degrees = _graph()
    gamma = descriptor.parameter_bounds.gamma.lo
    initial = [Q(0)] * 54
    for i, value in zip((4, 13, 22), impulse):
        initial[i] = value
    levels = [[tuple(initial)], [(Q(0),) * 54], [(Q(0),) * 54]]
    for n in range(order):
        rates = [list(_direct_linear(level[n], descriptor)) for level in levels]
        for edge_index, (i, j) in enumerate(edges):
            p = tuple(row[27 + j] - row[27 + i] for row in levels[0])
            q = tuple(row[27 + j] - row[27 + i] for row in levels[1])
            square = sum(p[k] * p[n - k] for k in range(n + 1))
            mixed = sum(p[k] * q[n - k] for k in range(n + 1))
            cube = sum(
                p[k] * p[joint] * p[n - k - joint]
                for k in range(n + 1)
                for joint in range(n + 1 - k)
            )
            sine = descriptor.parameter_bounds.edge_sines[edge_index].lo
            cosine = descriptor.parameter_bounds.edge_cosines[edge_index].lo
            nonlinear = (
                -gamma * sine * square / 2,
                -gamma * sine * mixed - gamma * cosine * cube / 6,
            )
            for level, current in enumerate(nonlinear, 1):
                rates[level][i] += current / degrees[i]
                rates[level][j] -= current / degrees[j]
        for level in range(3):
            levels[level].append(tuple(value / (n + 1) for value in rates[level]))
    return tuple(tuple(level) for level in levels)


@pytest.fixture(scope="module")
def low_order(synthetic_descriptor):
    impulse = (Q(1, 512), Q(-1, 1024), Q(1, 2048))
    source = tuple(I(Q((i * 7) % 17 - 8, 4096)) for i in range(54))
    comparator = tuple(I(Q(i - 3, 8192)) for i in range(6))
    coefficients = owner._causal_coefficients(
        synthetic_descriptor, impulse, source, comparator, order=6
    )
    return impulse, source, comparator, coefficients


def test_causal_levels_match_independent_original_edge_substitution(
    synthetic_descriptor, low_order
):
    impulse, _, _, result = low_order
    expected = _direct_coefficients(synthetic_descriptor, impulse, 6)
    for actual_level, expected_level in zip(
        result.nominal_level_coefficients, expected
    ):
        for actual_row, expected_row in zip(actual_level, expected_level):
            assert len(actual_row) == 54
            assert all(
                value.contains(target)
                for value, target in zip(actual_row, expected_row)
            )
    # A second level which vanishes at ports still has nonzero hidden content.
    assert any(value for row in expected[1] for value in row)
    assert all(
        row[i] == I(0)
        for row in result.nominal_level_coefficients[1]
        for i in result.kernels.port_indices
    )
    # Omitting the quadratic-to-cubic term changes the order-six coefficient.
    cubic = owner._nonlinear_forcing(
        result.nominal_level_coefficients[0],
        result.nominal_level_coefficients[1],
        synthetic_descriptor,
        order=6,
    )
    without_second = owner._nonlinear_forcing(
        result.nominal_level_coefficients[0],
        ((I(0),) * 54,) * 7,
        synthetic_descriptor,
        order=6,
    )
    assert any(
        not (a.lo <= b.hi and b.lo <= a.hi) for a, b in zip(cubic[5], without_second[5])
    )


def test_all_level_charges_and_reflection_are_retained(synthetic_descriptor, low_order):
    impulse, _, _, result = low_order
    expected = _direct_coefficients(synthetic_descriptor, impulse, 6)
    _, degrees = _graph()
    reflection = tuple(9 * (i // 9) + 8 - i % 9 for i in range(27))
    for level, rows in enumerate(expected):
        parity = 1 if level % 2 == 0 else -1
        for n, row in enumerate(rows):
            for channel in (0, 27):
                assert all(
                    row[channel + reflection[i]] == parity * row[channel + i]
                    for i in range(27)
                )
                expected_charge = (
                    sum(degrees[i] * value for i, value in zip((4, 13, 22), impulse))
                    if level == n == channel == 0
                    else 0
                )
                assert (
                    sum(degrees[i] * row[channel + i] for i in range(27))
                    == expected_charge
                )
    assert sum(degrees[i] * expected[0][1][i] for i in (4, 13, 22)) != 0


def test_kernel_factors_reconstruct_hidden_powers_and_memory(
    synthetic_descriptor, low_order
):
    kernels = low_order[-1].kernels
    for part, indices in enumerate(kernels.hidden_component_indices):
        for column in (0, 7, 8, 15):
            vector = tuple(Q(int(i == indices[column])) for i in range(54))
            for n in range(4):
                for row, index in enumerate(indices):
                    assert kernels.grounded_kernel_coefficients[part][n][row][
                        column
                    ].contains(vector[index])
                applied = _direct_linear(vector, synthetic_descriptor)
                vector = tuple(
                    applied[i] / (n + 1) if i in indices else Q(0) for i in range(54)
                )
    # The central return walk carries the full phase feedback, not heat alone.
    gamma = synthetic_descriptor.parameter_bounds.gamma.lo
    cosine = synthetic_descriptor.parameter_bounds.edge_cosines[9].lo
    assert kernels.memory_kernel_coefficients[0][1][1].contains(
        (1 - gamma**2 * cosine) / 4
    )


def test_independent_initialization_is_not_dropped_or_replaced(
    synthetic_descriptor, low_order
):
    _, source, comparator, result = low_order
    exact = tuple(value.lo for value in source)
    for n, row in enumerate(result.linear_source_coefficients):
        assert all(value.contains(target) for value, target in zip(row, exact))
        exact = tuple(
            value / (n + 1) for value in _direct_linear(exact, synthetic_descriptor)
        )
    assert result.comparator_source_coefficients[0] == comparator
    assert tuple(source[i] for i in result.kernels.port_indices) != comparator
    assert any(
        value.abs_max
        for value in result.hidden_initialization_port_forcing_coefficients[0]
    )
    eps = Q(1, 4096)
    hidden = [I(0)] * 54
    hidden[12], hidden[11] = I(eps / 2), I(-eps / 2)
    series, initial_force = owner._causal_linear_series(result.kernels, tuple(hidden))
    assert initial_force[0][1].contains(eps / 8)
    assert series[1][13].contains(eps / 8)
    assert (
        sum(
            synthetic_descriptor.parameter_bounds.degrees[i] * hidden[i].lo
            for i in range(27)
        )
        == 0
    )
    assert all(hidden[i] == I(0) for i in result.kernels.port_indices)


@pytest.mark.parametrize("i,j", product(range(5), repeat=2))
def test_beta_weight_matches_exact_monomial_integral(i, j):
    # Expand (1-s)^i and integrate against s^j independently.
    from math import comb

    exact = sum(Q((-1) ** k * comb(i, k), j + k + 1) for k in range(i + 1))
    assert owner._beta(i, j) == exact


def test_time_tail_bounds_cover_independent_positive_majorant_series():
    amplitude, h, g, rate = Q(1, 1300), Q(3, 4), Q(1, 3000), Q(201, 100)
    tails = owner._port_time_tails(amplitude, h)
    with mpmath.workdps(100):

        def mp(q):
            return mpmath.mpf(q.numerator) / q.denominator

        def tail(z, m):
            return mpmath.exp(mp(z)) - sum(mp(z) ** n / factorial(n) for n in range(m))

        exact = (
            mp(amplitude) * tail(rate * h, 33),
            mp(2 * g * amplitude**2 * h) * tail(2 * rate * h, 32),
            mp(Q(4, 3) * g * amplitude**3 * h) * tail(3 * rate * h, 32)
            + mp(4 * g**2 * amplitude**3 * h**2) * tail(3 * rate * h, 31),
        )
        assert all(0 < value <= mp(bound) for value, bound in zip(exact, tails))
        assert all(mp(bound) < 2 * value for value, bound in zip(exact, tails))
    assert owner._port_time_tails(Q(0), h) == (0, 0, 0)
    assert owner._port_time_tails(amplitude, Q(0)) == (0, 0, 0)


def _arguments():
    return dict(
        mediator_class=1,
        initial_form_bounds=((Q(0), Q(0)),) * 27,
        initial_phase_bounds=((Q(0), Q(0)),) * 27,
        comparator_initial_bounds=((Q(0), Q(0)),) * 6,
        port_impulse=(Q(1, 10000), Q(0), Q(-1, 12000)),
        horizon=Q(1, 16),
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("mediator_class", True),
        ("mediator_class", 0),
        ("mediator_class", 1.0),
        ("horizon", True),
        ("horizon", float("nan")),
        ("horizon", float("inf")),
        ("horizon", Q(-1)),
        ("horizon", Q(2001, 1000)),
        ("horizon", Decimal("1e-400")),
        ("port_impulse", (0, 0)),
        ("port_impulse", (0, 0, 0, 0)),
        ("port_impulse", (True, 0, 0)),
        ("port_impulse", (Q(7, 5000), Q(1, 10**100), 0)),
        ("initial_form_bounds", ((0, 0),) * 26),
        ("initial_form_bounds", ((0, 0),) * 28),
        ("initial_phase_bounds", ((0, 0),) * 28),
        ("comparator_initial_bounds", ((0, 0),) * 7),
        ("initial_form_bounds", ((0, 0, 0),) + ((0, 0),) * 26),
        ("initial_phase_bounds", ((1, 0),) + ((0, 0),) * 26),
        ("comparator_initial_bounds", ((False, 0),) + ((0, 0),) * 5),
    ],
)
def test_invalid_consumed_primitives_reject_before_kernel(monkeypatch, key, value):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid inputs reached scientific coefficient construction")

    monkeypatch.setattr(owner, "_derive_collective_interface", forbidden)
    arguments = _arguments()
    arguments[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner._predict_collective_port_response(**arguments)


def test_infinite_collections_are_bounded_and_rejected_before_computation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized input reached a kernel")

    monkeypatch.setattr(owner, "_derive_collective_interface", forbidden)
    seen = []

    def endless():
        while True:
            seen.append(1)
            yield (0, 0)

    args = _arguments()
    args["initial_form_bounds"] = endless()
    with pytest.raises(ValueError):
        owner._predict_collective_port_response(**args)
    assert len(seen) == 28


def test_public_projection_preserves_source_and_tail_accounting_without_selected_call(
    monkeypatch, synthetic_descriptor, low_order
):
    # This controls admission/projection, not numerical certification: substitute
    # unrelated low-order coefficient evidence while checking the public call's
    # fixed requested order and the exact source/error arithmetic independently.
    calls = []

    def coefficient_control(descriptor, impulse, source, comparator, *, order):
        calls.append((impulse, source, comparator, order))
        return low_order[-1]

    monkeypatch.setattr(
        owner, "_derive_collective_interface", lambda k: synthetic_descriptor
    )
    monkeypatch.setattr(owner, "_causal_coefficients", coefficient_control)
    args = _arguments()
    eps = Q(1, 10**44)
    args["initial_form_bounds"] = ((-eps, eps),) * 27
    args["initial_phase_bounds"] = ((-eps / 2, eps / 2),) * 27
    args["comparator_initial_bounds"] = ((-2 * eps, eps),) * 6
    args["horizon"] = 0.125
    report = owner._predict_collective_port_response(**args)
    assert len(calls) == 1 and calls[0][-1] == report.order == 32
    assert report.horizon == Q(1, 8)
    assert report.initial_form_bounds == args["initial_form_bounds"]
    assert report.source_radius == eps and report.comparator_source_radius == 2 * eps
    assert report.linear_source_uniform_bound == eps / (
        1 - 2 * Q(1, 3000) * report.horizon
    )
    assert (
        report.comparator_source_uniform_bound == 2 * report.linear_source_uniform_bound
    )
    assert (
        report.nominal_time_tail_upper_bound
        == report.nominal_level_time_tail_upper_bounds[0]
        + report.nominal_level_time_tail_upper_bounds[2]
    )
    for polynomial, tail, bounds, radius in zip(
        report.nominal_endpoint_polynomials,
        (report.nominal_time_tail_upper_bound,) * 6,
        report.nominal_endpoint_bounds,
        report.nominal_numerical_radius_upper_bounds,
    ):
        assert bounds == polynomial + I(-tail, tail)
        assert radius == bounds.radius
    assert report.linear_source_time_tail_upper_bound == eps * owner._exponential_tail(
        Q(201, 100) * report.horizon, 33
    )
    assert (
        report.comparator_source_time_tail_upper_bound
        == 2 * report.linear_source_time_tail_upper_bound
    )
    assert report.fidelity.nonlinear_initialization_error_upper_bound > 0
    payload = asdict(report)
    assert payload["initial_form_bounds"][0] == (-eps, eps)
    assert payload["nominal_endpoint_bounds"][0] == dict(
        lo=report.nominal_endpoint_bounds[0].lo, hi=report.nominal_endpoint_bounds[0].hi
    )
    assert "theta-Theta_k" in report.source_coordinates


def test_exact_tiny_impulses_reach_arithmetic_without_being_coerced_to_zero(
    monkeypatch, synthetic_descriptor, low_order
):
    tiny = Q(1, 10**400)
    received = []

    def control(descriptor, impulse, source, comparator, *, order):
        received.append(impulse)
        return low_order[-1]

    monkeypatch.setattr(
        owner, "_derive_collective_interface", lambda k: synthetic_descriptor
    )
    monkeypatch.setattr(owner, "_causal_coefficients", control)
    args = _arguments()
    args["port_impulse"] = (tiny, -tiny, 0)
    args["horizon"] = 0
    report = owner._predict_collective_port_response(**args)
    assert received == [(tiny, -tiny, Q(0))]
    assert report.port_impulse == (tiny, -tiny, Q(0))
    assert report.nominal_time_tail_upper_bound == 0
    assert report.fidelity.form_error_upper_bound == 0
