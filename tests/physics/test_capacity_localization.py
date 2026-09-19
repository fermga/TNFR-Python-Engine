"""Exact controls for capacity-conditioned balance of canonical pressures."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from tnfr.constants import DEFAULTS
from tnfr.physics.capacity_localization import observe_cycle_capacity_balance
from tnfr.utils import normalize_weights

F = Fraction


def _cycle_gradient(values):
    return tuple(
        (values[(i - 1) % len(values)] + values[(i + 1) % len(values)]) / 2 - value
        for i, value in enumerate(values)
    )


def _dirichlet(values):
    return (
        sum(
            (values[(i + 1) % len(values)] - value) ** 2
            for i, value in enumerate(values)
        )
        / 2
    )


def test_prepared_capacity_dip_produces_a_quantified_equilibrium_peak():
    nu = (F(1, 2),) + (F(1),) * 7
    result = observe_cycle_capacity_balance(
        (F(1, 2),) * 8, nu, epi_weight=F(1, 2), vf_weight=F(1, 2)
    )
    assert result.harmonic_capacity == F(8, 9)
    assert result.equilibrium_epi == (F(8, 9),) + (F(7, 18),) * 7
    assert result.equilibrium_pressure == (0,) * 8
    assert result.epi_dirichlet_energy == 0
    assert result.epi_rate[0] > 0
    assert result.epi_rate[1] < 0 and result.epi_rate[-1] < 0
    assert result.mean_conservation_residual == 0


@pytest.mark.parametrize("nu", [(F(1, 2), 1, 2, 1), (3, 2, 1, 4, 2)])
def test_nonuniform_pressure_balance_is_stationary_with_positive_capacity(nu):
    nu = tuple(F(value) for value in nu)
    epi = tuple(F(3) - value / 4 for value in nu)
    result = observe_cycle_capacity_balance(
        epi, nu, epi_weight=F(4, 5), vf_weight=F(1, 5)
    )
    assert result.epi == result.equilibrium_epi
    assert result.shifted_epi == (3,) * len(nu)
    assert result.pressure == result.epi_rate == (0,) * len(nu)
    assert result.lyapunov_value == result.lyapunov_derivative == 0
    assert result.shifted_dirichlet_energy == 0
    assert result.epi_dirichlet_energy > 0


def test_uniform_capacity_has_only_the_uniform_equilibrium_profile():
    epi = (F(1), F(0), F(1, 2), F(-1, 2))
    result = observe_cycle_capacity_balance(epi, (2,) * 4)
    assert result.equilibrium_epi == (F(1, 4),) * 4
    assert result.lyapunov_derivative < 0
    assert result.epi_dirichlet_derivative < 0


def test_zero_capacity_weight_recovers_heterogeneous_pure_epi_consensus():
    result = observe_cycle_capacity_balance((1, 0, 0, 0), (1, 2, 2, 2), vf_weight=0)
    assert result.shift_ratio == 0
    assert result.shifted_epi == result.epi
    assert result.equilibrium_epi == (F(2, 5),) * 4
    assert result.pressure_identity_residual == (0,) * 4


def test_epi_gradient_energy_can_rise_while_derived_balances_decrease():
    result = observe_cycle_capacity_balance(
        (F(3, 4), F(1, 2), F(1, 2), F(1, 2)),
        (F(1, 2), 1, 1, 1),
        epi_weight=F(1, 2),
        vf_weight=F(1, 2),
    )
    assert result.epi_dirichlet_derivative == F(1, 16)
    assert result.shifted_dirichlet_derivative == F(-1, 16)
    assert result.lyapunov_derivative == F(-1, 16)
    assert result.lyapunov_balance_residual == 0
    assert result.shifted_dirichlet_balance_residual == 0


def test_channel_superposition_and_two_energy_derivatives_are_independent():
    epi = tuple(F(value, 4) for value in (3, 0, -1, 2, 4))
    nu = tuple(F(value, 2) for value in (1, 2, 3, 2, 4))
    e, f = F(3, 5), F(1, 5)
    result = observe_cycle_capacity_balance(epi, nu, epi_weight=e, vf_weight=f)
    gx, gn = _cycle_gradient(epi), _cycle_gradient(nu)
    expected_pressure = tuple(e * a + f * b for a, b in zip(gx, gn))
    assert result.pressure == expected_pressure
    assert result.epi_rate == tuple(a * b for a, b in zip(nu, expected_pressure))
    assert result.lyapunov_derivative == -2 * e * _dirichlet(result.deviation)
    assert result.shifted_dirichlet_energy == _dirichlet(result.shifted_epi)
    gradient = tuple(-2 * value for value in _cycle_gradient(result.shifted_epi))
    expected_dirichlet_rate = -e * sum(
        capacity * value**2 / 2 for capacity, value in zip(nu, gradient)
    )
    assert result.shifted_dirichlet_derivative == expected_dirichlet_rate


def test_capacity_scaling_changes_forcing_as_well_as_the_nodal_clock():
    nu = (F(1, 2), F(1), F(1), F(1))
    first = observe_cycle_capacity_balance((0,) * 4, nu)
    second = observe_cycle_capacity_balance((0,) * 4, tuple(2 * v for v in nu))
    assert second.epi_rate == tuple(4 * value for value in first.epi_rate)
    assert second.equilibrium_epi == tuple(2 * value for value in first.equilibrium_epi)


def test_conserved_total_selects_the_equilibrium_level():
    result = observe_cycle_capacity_balance((1, 2, 0, -1), (1, 2, 3, 4))
    total = sum(h * x for h, x in zip(result.metric_weights, result.equilibrium_epi))
    assert total == result.conserved_epi_total
    assert sum(h * z for h, z in zip(result.metric_weights, result.deviation)) == 0
    assert result.mean_conservation_residual == 0


def test_defaults_use_the_normalized_full_four_channel_mix():
    result = observe_cycle_capacity_balance((0,) * 3, (1,) * 3)
    defaults = normalize_weights(
        DEFAULTS["DNFR_WEIGHTS"], ("phase", "epi", "vf", "topo")
    )
    assert result.epi_weight == F.from_float(defaults["epi"])
    assert result.vf_weight == F.from_float(defaults["vf"])
    assert result.epi_weight + result.vf_weight < 1
    changed = observe_cycle_capacity_balance((0,) * 3, (1,) * 3, vf_weight=0)
    assert changed.epi_weight == result.epi_weight


def test_operator_capacity_writes_release_the_old_equilibrium():
    nu = tuple(F(value, 2) for value in (1, 2, 3, 2, 1))
    e, f, s, q = F(1, 2), F(1, 4), F(1, 8), F(7, 8)
    epi = tuple(F(2) - f * value / e for value in nu)
    gradient = _cycle_gradient(nu)
    synced = tuple(v + s * dv for v, dv in zip(nu, gradient))
    um = observe_cycle_capacity_balance(epi, synced, epi_weight=e, vf_weight=f)
    sha = observe_cycle_capacity_balance(
        epi, tuple(q * v for v in nu), epi_weight=e, vf_weight=f
    )
    twice = _cycle_gradient(gradient)
    assert um.pressure == tuple(f * s * value for value in twice)
    assert sha.pressure == tuple(-f * (1 - q) * value for value in gradient)
    assert any(um.pressure) and any(sha.pressure)


def test_shared_reader_preserves_numpy_and_arbitrarily_small_rationals():
    result = observe_cycle_capacity_balance(
        np.array([1, 0, -1], dtype=np.int64),
        tuple(F(np.int64(value)) for value in (1, 2, 3)),
    )
    assert all(type(v.numerator) is int for v in result.capacity)
    tiny = F(1, 10**500)
    exact = observe_cycle_capacity_balance((tiny, 0, -tiny), (tiny,) * 3)
    assert exact.capacity == (tiny,) * 3
    assert exact.lyapunov_value > 0


def test_results_are_detached_and_immutable():
    epi, nu = [1, 0, 0], [1, 2, 1]
    result = observe_cycle_capacity_balance(epi, nu)
    epi[0], nu[0] = 99, 99
    assert result.epi == (1, 0, 0) and result.capacity == (1, 2, 1)
    with pytest.raises(FrozenInstanceError):
        result.shifted_consensus = F(0)
    with pytest.raises(TypeError):
        result.equilibrium_epi[0] = F(0)


@pytest.mark.parametrize(
    "epi,nu",
    [
        ((), ()),
        ((0, 0), (1, 1)),
        ((0, 0, 0), (1, 1)),
        ((0, 0, 0), (0, 1, 1)),
        ((0, 0, 0), (-1, 1, 1)),
        ((0, 0, 0), (True, 1, 1)),
        ((True, 0, 0), (1, 1, 1)),
        ((float("inf"), 0, 0), (1, 1, 1)),
        ("000", (1, 1, 1)),
        ((0, 0, 0), {1, 2, 3}),
    ],
)
def test_invalid_cycle_coordinates_are_rejected(epi, nu):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_capacity_balance(epi, nu)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epi_weight": 0},
        {"epi_weight": -1},
        {"vf_weight": -1},
        {"epi_weight": True},
        {"vf_weight": "0.5"},
        {"vf_weight": float("nan")},
    ],
)
def test_invalid_channel_coefficients_are_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        observe_cycle_capacity_balance((0,) * 3, (1,) * 3, **kwargs)
