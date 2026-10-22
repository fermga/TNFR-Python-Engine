"""Independent small-support controls; the reserved packet is not evaluated."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import product
from math import factorial

import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics import _localized_mobile_spectral as owner


@pytest.fixture(scope="module", autouse=True)
def no_organized_response_or_source_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _model(**changes):
    return RelationalExchangeModel(
        **(
            dict(
                epi_weight=Q(1, 1000),
                phase_weight=Q(999, 1000),
                storage_scale=1,
                phase_domain="regular",
            )
            | changes
        )
    )


def _kernel(**changes):
    return owner._prepare_mobile_spectral_kernel(
        **(
            dict(
                reference_model=_model(),
                node_count=16,
                fejer_order=2,
                amplitude=Q(1, 10000),
                horizon=4,
            )
            | changes
        )
    )


def _multiply_matrix(left, right):
    return tuple(
        sum((left[2 * i + k] * right[2 * k + j] for k in range(2)), I(0))
        for i in range(2)
        for j in range(2)
    )


def _series_propagator(eigenvalue, sigma, time, order=96):
    """Independent matrix exponential with a geometric majorant of its tail."""
    generator = (
        -sigma * eigenvalue * time,
        -eigenvalue * time,
        eigenvalue * time,
        I(0),
    )
    term = result = (I(1), I(0), I(0), I(1))
    for degree in range(1, order + 1):
        term = tuple(entry / degree for entry in _multiply_matrix(term, generator))
        result = tuple(value + addition for value, addition in zip(result, term))
    norm_bound = max(generator[0].abs_max + generator[1].abs_max, generator[2].abs_max)
    assert norm_bound < order + 2
    tail = (
        norm_bound ** (order + 1)
        / factorial(order + 1)
        / (1 - norm_bound / (order + 2))
    )
    return tuple(entry + I(-tail, tail) for entry in result)


def _cmul(left, right):
    a, b = left
    c, d = right
    return a * c - b * d, a * d + b * c


def _dft(values):
    size = len(values)
    root = sqrt(I(size))
    return {
        mode: (
            sum(
                (
                    value * cos(-2 * pi_interval() * Q(mode * j, size))
                    for j, value in enumerate(values)
                ),
                I(0),
            )
            / root,
            sum(
                (
                    value * sin(-2 * pi_interval() * Q(mode * j, size))
                    for j, value in enumerate(values)
                ),
                I(0),
            )
            / root,
        )
        for mode in range(-size // 2, size // 2)
    }


def _small_physical_integrand(metadata, time):
    """Build real site preparations, evolve modes by series, then cube sites."""
    size, order = metadata.node_count, metadata.fejer_order
    c = metadata.horizon // 2
    coefficient_sum = sum((1 - Q(abs(m), order)) ** 2 for m in range(1 - order, order))
    normalization = I(metadata.amplitude) / sqrt(I(size * coefficient_sum))
    form_a, phase_a = [], []
    for j in range(size):
        envelope = sum(
            (
                (1 - Q(abs(m), order)) * cos(2 * pi_interval() * Q(m * (j - c), size))
                for m in range(1 - order, order)
            ),
            I(0),
        )
        carrier = pi_interval() * Q(j - c, 2)
        form_a.append(normalization * envelope * cos(carrier))
        phase_a.append(normalization * envelope * sin(carrier))
    form_modes, phase_modes = _dft(form_a), _dft(phase_a)
    sigma = (
        pi_interval() * metadata.form_loss_coefficient / metadata.exchange_coefficient
    )
    evolved_phase = {}
    for mode in metadata.phase_modes:
        eigenvalue = 1 - cos(2 * pi_interval() * Q(mode, size))
        _, _, u21, u22 = _series_propagator(eigenvalue, sigma, I(time))
        evolved_phase[mode] = tuple(
            u21 * a + u22 * b for a, b in zip(form_modes[mode], phase_modes[mode])
        )
    phase_values = []
    for j in range(size):
        value = I(0)
        for mode, coefficient in evolved_phase.items():
            angle = 2 * pi_interval() * Q(mode * j, size)
            value += _cmul(coefficient, (cos(angle), sin(angle)))[0]
        phase_values.append(value / sqrt(I(size)))
    reflected = [phase_values[-j % size] for j in range(size)]
    edges_a = [phase_values[j] - phase_values[(j + 1) % size] for j in range(size)]
    edges_b = [reflected[j] - reflected[(j + 1) % size] for j in range(size)]
    mixed_edges = [a * b * (a + b) / 4 for a, b in zip(edges_a, edges_b)]
    mixed_sites = [mixed_edges[j] - mixed_edges[(j - 1) % size] for j in range(size)]
    source = _dft(mixed_sites)
    result = []
    for mode in metadata.modes:
        eigenvalue = 1 - cos(2 * pi_interval() * Q(mode, size))
        u11, _, u21, _ = _series_propagator(
            eigenvalue, sigma, I(metadata.horizon - time)
        )
        result.append(
            (
                u11 * source[mode][0],
                u11 * source[mode][1],
                u21 * source[mode][0],
                u21 * source[mode][1],
            )
        )
    return tuple(result), mixed_sites


def test_dyadic_convolution_contains_independent_corner_values_and_modular_aliases():
    left = {
        1: (I(Q(-3, 8), Q(1, 4)), I(Q(1, 8), Q(3, 8))),
        -2: (I(Q(1, 16), Q(3, 16)), I(Q(-1, 4), 0)),
    }
    right = {
        2: (I(Q(1, 8), Q(1, 2)), I(Q(-1, 8), Q(1, 8))),
        3: (I(Q(-1, 8), 0), I(Q(1, 16), Q(3, 16))),
    }
    observed = owner._dyadic_cyclic_convolution(left, right, 8)
    intervals = [
        component
        for rows in (left, right)
        for coefficient in rows.values()
        for component in coefficient
    ]
    for vertices in product(*[(interval.lo, interval.hi) for interval in intervals]):
        point_left = dict(
            zip(left, ((vertices[0], vertices[1]), (vertices[2], vertices[3])))
        )
        point_right = dict(
            zip(right, ((vertices[4], vertices[5]), (vertices[6], vertices[7])))
        )
        exact = {}
        for lmode, (a, b) in point_left.items():
            for rmode, (c, d) in point_right.items():
                mode = (lmode + rmode + 4) % 8 - 4
                real, imag = exact.get(mode, (Q(0), Q(0)))
                exact[mode] = real + a * c - b * d, imag + a * d + b * c
        for mode, coefficient in exact.items():
            assert observed[mode][0].contains(coefficient[0])
            assert observed[mode][1].contains(coefficient[1])
    assert -4 in observed  # 1+3 aliases to the retained Nyquist coordinate.


def test_dyadic_convolution_keeps_cancellation_and_subgrid_products():
    tiny = Q(1, 1 << 128)
    left = {1: (I(tiny), I(0)), -1: (I(-tiny), I(0))}
    right = {1: (I(tiny), I(0)), -1: (I(tiny), I(0))}
    result = owner._dyadic_cyclic_convolution(left, right, 16)
    assert result[0] == (I(0), I(0))
    assert result[2][0].contains(tiny**2)
    assert result[-2][0].contains(-(tiny**2))
    assert result[2][0].hi > 0 > result[-2][0].lo


@pytest.mark.parametrize("turns", [-1031, -3, -2, -1, 0, 1, 2, 3, 1040])
def test_quarter_turn_reduction_retains_the_shared_small_angle_enclosure(turns):
    offset = I(Q(-1, 13), Q(-1, 13) + Q(1, 10**30))
    observed = owner._reduced_sine_cosine(turns * pi_interval() / 2 + offset)
    s, c = sin(offset), cos(offset)
    expected = ((s, c), (c, -s), (-s, -c), (-c, s))[turns % 4]
    for value, reference in zip(observed, expected):
        assert (value - reference).contains(0)
        assert (value - reference).abs_max < Q(1, 10**28)


@pytest.mark.parametrize("quarter_pi", [Q(-3, 4), Q(-1, 4), Q(1, 4), Q(3, 4)])
def test_quarter_turn_choice_boundary_is_safe_for_nonpoint_arguments(quarter_pi):
    angle = quarter_pi * pi_interval() + I(Q(-1, 10**12), Q(1, 10**12))
    observed = owner._reduced_sine_cosine(angle)
    for value, independent in zip(observed, (sin(angle), cos(angle))):
        assert (value - independent).contains(0)
        assert (value - independent).abs_max < Q(1, 10**10)


@pytest.mark.parametrize("eigenvalue", [Q(0), Q(1, 3), Q(2)])
def test_actual_modal_propagator_matches_independent_matrix_exponential(eigenvalue):
    sigma, time = I(Q(1, 100)), I(Q(3, 7))
    frequency = sqrt(1 - sigma**2 / 4)
    observed = owner._mode_propagator(I(eigenvalue), sigma, frequency, time)
    independent = _series_propagator(I(eigenvalue), sigma, time, order=48)
    for actual, expected in zip(observed, independent):
        difference = actual - expected
        assert difference.contains(0)
        assert difference.abs_max < Q(1, 10**30)
    if eigenvalue:
        assert observed[0].hi < observed[3].lo
        assert observed[1].hi < 0 < observed[2].lo
    else:
        assert all(
            value.contains(expected) for value, expected in zip(observed, (1, 0, 0, 1))
        )


@pytest.mark.parametrize("time", [Q(0), Q(1, 3), Q(4)])
def test_small_spectral_integrand_matches_real_space_cubic_and_independent_tangent(
    time,
):
    metadata, evaluate = _kernel()
    spectral = evaluate(time)
    direct, mixed_sites = _small_physical_integrand(metadata, time)
    assert any(not value.contains(0) for value in mixed_sites)
    for actual_row, independent_row in zip(spectral, direct):
        for actual, independent in zip(actual_row, independent_row):
            difference = actual - independent
            assert difference.contains(0)
            assert difference.abs_max < Q(1, 10**26)
    # Losing either 1/sqrt(N) in the two Fourier products would multiply N.
    nonzero = next((row[0] for row in spectral if not row[0].contains(0)), None)
    assert nonzero is not None
    assert (nonzero * metadata.node_count - nonzero).lo * (
        nonzero * metadata.node_count - nonzero
    ).hi > 0


def test_reflection_is_exact_in_both_channels_and_support_is_complete():
    metadata, evaluate = _kernel(node_count=32, fejer_order=3, horizon=8)
    rows = dict(zip(metadata.modes, evaluate(Q(2, 7))))
    expected_modes = {
        (first + second + third + metadata.node_count // 2) % metadata.node_count
        - metadata.node_count // 2
        for first in metadata.phase_modes
        for second in metadata.phase_modes
        for third in metadata.phase_modes
    }
    assert set(metadata.modes) == expected_modes
    assert len(rows) == 26
    for mode, row in rows.items():
        assert rows[-mode] == row
        assert row[1] == row[3] == I(0)
    assert all(
        (j, (j + 1) % metadata.node_count) in metadata.edges
        for j in metadata.node_order
    )
    with pytest.raises(FrozenInstanceError):
        metadata.amplitude = Q(1)


def test_time_boxes_enclose_pointwise_values_without_reconstructing_time_from_response():
    _, evaluate = _kernel()
    whole = evaluate(I(Q(1, 7), Q(1, 7) + Q(1, 10**10)))
    for point in (Q(1, 7), Q(1, 7) + Q(1, 10**10)):
        value = evaluate(point)
        for outer, inner in zip(whole, value):
            for a, b in zip(outer, inner):
                # Dependency can make separate enclosures incomparable by a
                # few grid units; independent true values must overlap.
                assert (a - b).contains(0)


@pytest.mark.parametrize(
    "bad",
    [
        True,
        np.bool_(False),
        float("nan"),
        float("inf"),
        {},
        [],
        complex(1, 0),
        -1,
        5,
        I(-1, 0),
        I(3, 5),
    ],
)
def test_time_admission_rejects_invalid_primitives_and_outside_intervals(bad):
    _, evaluate = _kernel()
    with pytest.raises((TypeError, ValueError)):
        evaluate(bad)


@pytest.mark.parametrize(
    "changes",
    [
        {"amplitude": True},
        {"amplitude": float("nan")},
        {"amplitude": 0},
        {"amplitude": -1},
        {"node_count": np.int64(16)},
        {"horizon": True},
        {"horizon": 3},
        {"node_count": 8192},
        {"fejer_order": 33},
        {"node_count": 8},
        {"reference_model": _model(epi_weight=0)},
        {"reference_model": _model(storage_scale=2)},
        {"reference_model": _model(epi_weight=1, phase_weight=1)},
    ],
)
def test_factory_rebuilds_only_supported_primitive_models(changes):
    with pytest.raises((TypeError, ValueError)):
        _kernel(**changes)


def test_actual_tiny_clock_coefficients_are_retained_without_float_admission():
    model = _model()
    tiny = Q(1, 10**500)
    object.__setattr__(model, "phase_weight", tiny)
    object.__setattr__(model, "epi_weight", tiny / 999)
    metadata, _ = _kernel(reference_model=model, amplitude=tiny)
    assert metadata.exchange_coefficient == metadata.amplitude == tiny
    assert metadata.form_loss_coefficient == tiny / 999
    object.__setattr__(model, "epi_weight", True)
    with pytest.raises((TypeError, ValueError)):
        _kernel(reference_model=model)
