"""Static full-law controls for the small real-lift cycle consensus basin."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import numpy as np
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics import _localized_identity_transport as owner
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _model(**changes):
    return RelationalExchangeModel(
        **(
            dict(
                storage_scale=1,
                epi_weight=Q(1, 2**20),
                phase_weight=1 - Q(1, 2**20),
                phase_domain="regular",
            )
            | changes
        )
    )


def _bound(**changes):
    return owner._bound_cycle_consensus_basin(
        **(
            dict(
                reference_model=_model(),
                node_count=1024,
                initial_state_norm_upper_bound=Q(1, 1000),
            )
            | changes
        )
    )


def _near_zero(value):
    assert value.contains(0)
    assert value.abs_max < Q(1, 10**32)


def test_actual_nonlinear_rows_and_both_clock_rows_give_exact_storage_loss():
    size = 7
    forms = tuple(Q(j * j - 11, 100) for j in range(size))
    phases = tuple(Q((-1) ** j * (j + 1), 100) for j in range(size))
    laplacian_forms = tuple(
        2 * forms[j] - forms[(j - 1) % size] - forms[(j + 1) % size]
        for j in range(size)
    )
    currents = tuple(
        sin(phases[(j - 1) % size] - phases[j])
        + sin(phases[(j + 1) % size] - phases[j])
        for j in range(size)
    )
    model = _model()
    rates = _sine_rates(model, (2,) * size, laplacian_forms, (Q(1),) * size, currents)
    loss_coefficient = Q(1, 2**20)
    exchange_coefficient = 1 - loss_coefficient
    clock_conversion = pi_interval() / exchange_coefficient
    form_rates = tuple(rate * clock_conversion for rate in rates["form_rates"])
    phase_rates = tuple(rate * clock_conversion for rate in rates["phase_rates"])
    derivative = sum(
        (
            laplacian_forms[j] * form_rates[j] - currents[j] * phase_rates[j]
            for j in range(size)
        ),
        I(0),
    )
    sigma = pi_interval() * loss_coefficient / exchange_coefficient
    loss = sigma * sum(value**2 for value in laplacian_forms) / 2
    _near_zero(derivative + loss)
    assert derivative.hi < 0
    _near_zero(sum(form_rates, I(0)))
    _near_zero(sum(phase_rates, I(0)))
    assert not (
        sum(
            (
                laplacian_forms[j] * form_rates[j]
                - currents[j] * rates["phase_rates"][j]
                for j in range(size)
            ),
            I(0),
        )
        + loss
    ).contains(0)


def test_period_four_invariant_carrier_has_two_coupled_rows_and_pendulum_coordinates():
    first, second, rate_first, rate_second = sp.symbols("u v r s", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    phases = (first, second, -first, -second) * 2
    forms = (rate_first, rate_second, -rate_first, -rate_second) * 2
    size = len(forms)
    expected_phase_rates = forms
    acceleration_first = -sigma * rate_first - sp.sin(first) * sp.cos(second)
    acceleration_second = -sigma * rate_second - sp.sin(second) * sp.cos(first)
    expected_form_rates = (
        acceleration_first,
        acceleration_second,
        -acceleration_first,
        -acceleration_second,
    ) * 2
    for j in range(size):
        ax = forms[j] - (forms[(j - 1) % size] + forms[(j + 1) % size]) / 2
        phase_current = (
            sp.sin(phases[(j - 1) % size] - phases[j])
            + sp.sin(phases[(j + 1) % size] - phases[j])
        ) / 2
        assert sp.simplify(ax - expected_phase_rates[j]) == 0
        assert sp.trigsimp(-sigma * ax + phase_current - expected_form_rates[j]) == 0
    for sign in (1, -1):
        assert (
            sp.trigsimp(
                acceleration_first
                + sign * acceleration_second
                + sigma * (rate_first + sign * rate_second)
                + sp.sin(first + sign * second)
            )
            == 0
        )
    # The spatial cell repeats: this exact carrier alone is not a localized packet.
    assert phases[:4] == phases[4:]
    assert sum(phases) == sum(forms) == 0


@pytest.mark.parametrize("size", [3, 4, 7])
def test_cycle_energy_majorant_includes_both_channels_and_constant_backgrounds(size):
    coordinates = sp.symbols(f"u:{size}", real=True)
    differences_squared = sum(
        (coordinates[(j + 1) % size] - coordinates[j]) ** 2 for j in range(size)
    )
    complement = sum(
        (coordinates[(j + 1) % size] + coordinates[j]) ** 2 for j in range(size)
    )
    assert (
        sp.expand(
            4 * sum(value**2 for value in coordinates)
            - differences_squared
            - complement
        )
        == 0
    )
    forms = tuple(Q((-1) ** j * (j + 1), 100) for j in range(size))
    phases = tuple(Q(j - 2, 200) for j in range(size))

    def energy(form, phase):
        return sum(
            (
                (form[(j + 1) % size] - form[j]) ** 2 / 2
                + 1
                - cos(phase[(j + 1) % size] - phase[j])
                for j in range(size)
            ),
            I(0),
        )

    storage = energy(forms, phases)
    norm_squared = sum(x**2 + theta**2 for x, theta in zip(forms, phases))
    assert storage.hi <= 2 * norm_squared
    shifted = energy(
        tuple(value - 13 for value in forms), tuple(value + 7 for value in phases)
    )
    assert shifted == storage
    assert sum(forms) != 0  # The certificate does not discard actual mean residuals.


def test_all_acute_cycle_faces_cost_at_least_one_with_a_zero_period_witness():
    size = 8
    pi = pi_interval()
    distinguished = pi / 2
    remaining = -pi / (2 * (size - 1))
    _near_zero(distinguished + (size - 1) * remaining)
    face_energy = 1 + (size - 1) * (1 - cos(remaining))
    assert face_energy.lo > 1
    # A first principal winding crossing costs two on its antipodal edge alone.
    _near_zero(1 - cos(pi) - 2)
    report = _bound(initial_state_norm_upper_bound=Q(1, 2))
    assert report.consensus_basin_certified
    assert report.initial_storage_upper_bound == Q(1, 2)
    assert report.edge_cosine_lower_bound == Q(1, 2)


def test_low_energy_winding_one_is_not_a_small_real_lift_consensus_source():
    size = 1024
    alpha = 2 * pi_interval() / size
    phase_energy = size * (1 - cos(alpha))
    assert 0 < phase_energy.lo < phase_energy.hi < 1
    assert cos(alpha).lo > 0
    _near_zero(size * alpha - 2 * pi_interval())
    # Every edge has the same principal increment; opposite node currents cancel.
    _near_zero(sin(alpha) + sin(-alpha))
    # Centering the continuous ramp minimizes its real-lift norm over common origins.
    minimum_norm_squared = alpha**2 * Q(size * (size**2 - 1), 12)
    assert minimum_norm_squared.lo > 1
    report = _bound(initial_state_norm_upper_bound=sqrt(minimum_norm_squared).hi)
    assert report.status == "not_certified"
    assert report.edge_cosine_lower_bound is None
    assert not report.consensus_basin_certified
    assert report.unavailable_reasons == ("acute_storage_barrier_not_certified",)


def test_frozen_pair_source_norm_gives_an_all_time_basin_without_loading_a_response():
    # Exact outward geometric bound, independent of every retained endpoint response.
    amplitude = Q(1, 2**12)
    radius = (Q(99, 70) + Q(1, 500)) * amplitude
    assert Q(99, 70) ** 2 > 2
    report = _bound(initial_state_norm_upper_bound=radius)
    assert report.initial_storage_upper_bound < Q(3, 12500000)
    assert report.edge_cosine_lower_bound > 1 - Q(3, 12500000)
    assert report.status == "certified"
    assert report.unavailable_reasons == ()
    assert report.conditional_on_supplied_preparation
    with pytest.raises(FrozenInstanceError):
        report.status = "not_certified"


def test_boundary_is_strict_and_failed_sufficiency_leaves_geometry_unavailable():
    # Adjacent rational numbers on opposite sides of the irrational M=1/sqrt(2).
    inside = Q(707106781186547524400, 10**21)
    outside = inside + Q(1, 10**21)
    assert 2 * inside**2 < 1 < 2 * outside**2
    assert _bound(initial_state_norm_upper_bound=inside).consensus_basin_certified
    unavailable = _bound(initial_state_norm_upper_bound=outside)
    assert unavailable.acute_barrier_margin < 0
    assert unavailable.edge_cosine_lower_bound is None
    assert unavailable.status == "not_certified"
    assert not unavailable.consensus_basin_certified
    zero = _bound(node_count=3, initial_state_norm_upper_bound=0)
    assert zero.status == "certified"
    assert zero.edge_cosine_lower_bound == 1


def test_fresh_exact_primitives_are_normalized_before_arithmetic_and_retention():
    represented = Q(np.int64(1), np.int64(1000))
    report = _bound(initial_state_norm_upper_bound=represented)
    assert report == _bound(initial_state_norm_upper_bound=Q(1, 1000))
    for value in (
        report.initial_state_norm_upper_bound,
        report.initial_storage_upper_bound,
        report.acute_barrier_margin,
    ):
        assert type(value.numerator) is type(value.denominator) is int
    tiny = Q(1, 10**500)
    report = _bound(initial_state_norm_upper_bound=tiny)
    assert report.initial_state_norm_upper_bound == tiny
    assert report.initial_storage_upper_bound == 2 * tiny**2 > 0
    assert report.edge_cosine_lower_bound < 1


@pytest.mark.parametrize("bad", [True, np.int64(8), 8.0, Q(8), "8", None])
def test_cycle_size_requires_an_ordinary_integer(bad):
    with pytest.raises(TypeError, match="ordinary Python integer"):
        _bound(node_count=bad)


@pytest.mark.parametrize("bad", [-1, 0, 1, 2])
def test_simple_cycle_requires_at_least_three_nodes(bad):
    with pytest.raises(ValueError, match="at least three"):
        _bound(node_count=bad)


@pytest.mark.parametrize(
    "bad", [True, np.bool_(False), float("nan"), float("inf"), "0.1", None, 1j]
)
def test_norm_admission_precedes_arithmetic(bad):
    with pytest.raises((TypeError, ValueError)):
        _bound(initial_state_norm_upper_bound=bad)


def test_negative_subfloat_norm_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        _bound(initial_state_norm_upper_bound=-Q(1, 10**500))


@pytest.mark.parametrize(
    "changes",
    [{"storage_scale": 2}, {"epi_weight": 0}, {"phase_domain": "acute"}],
)
def test_different_complete_laws_do_not_inherit_consensus(changes):
    with pytest.raises(ValueError):
        _bound(reference_model=_model(**changes))


def test_authoritative_model_fields_are_readmitted_and_tiny_loss_is_not_zero():
    forged = _model()
    object.__setattr__(forged, "epi_weight", True)
    with pytest.raises(TypeError):
        _bound(reference_model=forged)
    tiny = Q(1, 10**500)
    object.__setattr__(forged, "epi_weight", tiny)
    report = _bound(reference_model=forged)
    assert report.form_loss_coefficient == tiny
    assert report.exchange_coefficient == 1 - Q(1, 2**20)
    assert report.status == "certified"
    with pytest.raises((TypeError, ValueError)):
        _bound(reference_model=report)
