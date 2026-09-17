"""Independent lattice, inverse-cell and scope checks for fixed C6 pressure."""

from dataclasses import FrozenInstanceError
from fractions import Fraction
import math

import numpy as np
import pytest

from tnfr.constants.canonical import CHANNEL_WEIGHT_PRIMARY, CHANNEL_WEIGHT_SECONDARY
from tnfr.dynamics.fused_dnfr import compute_fused_gradients_symmetric
from tnfr.mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference
from tnfr.mathematics._phase_midpoint import certified_two_neighbor_phase
from tnfr.physics import binary64_pressure_equilibrium as owner


PHASE = tuple(i * math.pi / 3 for i in range(6))
QUANTUM = Fraction(1, 2**58)


def _derive(**kwargs):
    inputs = dict(
        phase=PHASE, epi_weight=CHANNEL_WEIGHT_SECONDARY,
        phase_weight=CHANNEL_WEIGHT_PRIMARY,
    )
    inputs.update(kwargs)
    return owner.derive_binary64_c6_pressure_equilibrium_obstruction(**inputs)


@pytest.fixture(scope="module")
def reference():
    return _derive()


def _cell(target):
    lower = (Fraction(math.nextafter(target, -math.inf)) + Fraction(target)) / 2
    upper = (Fraction(math.nextafter(target, math.inf)) + Fraction(target)) / 2
    index = Fraction(target) / Fraction(math.ulp(target))
    assert index.denominator == 1
    return lower, upper, index.numerator % 2 == 0


def test_prepared_winding_is_a_sufficient_obstruction_not_a_full_runtime_claim(reference):
    assert reference.phase == PHASE
    assert reference.gradient_quantum == QUANTUM
    assert reference.no_zero_pressure
    assert any(row.grid_excluded for row in reference.rows)
    assert not all(row.grid_excluded for row in reference.rows)
    assert reference.fixed_positive_step_convergence_excluded
    assert reference.positive_band_exit_certified is False
    with pytest.raises(FrozenInstanceError):
        reference.no_zero_pressure = False


def test_phase_source_uses_the_shared_midpoint_then_two_separate_roundings(reference):
    for i, row in enumerate(reference.rows):
        midpoint = certified_two_neighbor_phase(PHASE[i], PHASE[i - 1], PHASE[(i + 1) % 6])
        assert midpoint is not None
        gradient = float(Fraction(midpoint.delta) / Fraction(math.pi))
        contribution = float(Fraction(CHANNEL_WEIGHT_PRIMARY) * Fraction(gradient))
        assert row.index == i
        assert row.phase_response == midpoint
        assert row.phase_gradient.hex() == gradient.hex()
        assert row.phase_contribution == contribution
        assert row.target_epi_pressure == -contribution


def test_inverse_cell_endpoints_and_grid_indices_follow_an_independent_exact_oracle(reference):
    seen_parities = set()
    for row in reference.rows:
        lower, upper, closed = _cell(row.target_epi_pressure)
        seen_parities.add(closed)
        assert row.cancellation_cell.lower == lower
        assert row.cancellation_cell.upper == upper
        assert row.inverse_gradient_lower == lower / reference.epi_weight
        assert row.inverse_gradient_upper == upper / reference.epi_weight
        assert row.inverse_gradient_lower_closed is closed
        assert row.inverse_gradient_upper_closed is closed
        left = lower / reference.epi_weight / QUANTUM
        right = upper / reference.epi_weight / QUANTUM
        first = math.ceil(left) + int(not closed and left.denominator == 1)
        last = math.floor(right) - int(not closed and right.denominator == 1)
        assert (row.first_grid_index, row.last_grid_index) == (first, last)
        assert row.grid_excluded is (first > last)
        if first <= last:
            for candidate in {first, last}:
                product = reference.epi_weight * candidate * QUANTUM
                assert float(product) == row.target_epi_pressure
    assert seen_parities == {False, True}


@pytest.mark.parametrize("phase", ((0.0,) * 6, (0.25,) * 6, (-0.0,) * 6))
def test_synchronized_phase_does_not_certify_an_obstruction(phase):
    result = _derive(phase=phase)
    assert result.no_zero_pressure is False
    assert result.fixed_positive_step_convergence_excluded is False
    assert result.positive_band_exit_certified is False
    for row in result.rows:
        assert row.phase_contribution == 0.0
        assert row.first_grid_index == row.last_grid_index == 0
        assert not row.grid_excluded


def test_nonempty_unbounded_row_cells_do_not_imply_any_bounded_joint_root():
    # Each row's unbounded lattice intersects its inverse cell, although the
    # bounded EPI channel cannot cancel this much larger phase contribution.
    result = _derive(phase=(0.0, 0.25, 0.25, 0.25, 0.25, 0.25), epi_weight=2.0**-60, phase_weight=1.0)
    assert all(row.first_grid_index <= row.last_grid_index for row in result.rows)
    assert not result.no_zero_pressure
    assert not result.fixed_positive_step_convergence_excluded
    row = result.rows[0]
    assert abs(Fraction(row.phase_contribution)) > result.epi_weight
    assert abs(row.first_grid_index * QUANTUM) > 1


@pytest.mark.parametrize("center,neighbors", (
    (0.05, (0.0625, 1.0)),
    (0.0625, (0.05, 0.125)),
    (0.5, (math.nextafter(0.5, 0.0), math.nextafter(0.5, math.inf))),
    (1.0, (0.05, math.nextafter(1.0, 0.0))),
    (0.05, (0.05, math.nextafter(0.05, math.inf))),
    (0.5, (1.0, 1.0)),
    (0.05, (0.05, 0.05)),
))
@pytest.mark.parametrize("coefficient", (1.0, CHANNEL_WEIGHT_SECONDARY))
def test_actual_scalar_and_vector_reducers_fit_the_grid_for_both_branches(center, neighbors, coefficient):
    mixed = min(neighbors) < center < max(neighbors)
    if mixed:
        gradient = sum(map(Fraction, neighbors)) / 2 - Fraction(center)
    else:
        # Exact 1/2 probabilities; each represented subtraction precedes the
        # two-term sum on the ordinary source branch.
        gradient = Fraction(sum(0.5 * (value - center) for value in neighbors))
    assert (gradient / QUANTUM).denominator == 1
    expected = float(Fraction(coefficient) * gradient)
    scalar = mean_neighbor_difference(center, neighbors, coefficient=coefficient)
    vector = edge_mean_differences(
        np.asarray((center, *neighbors)), np.asarray((0, 0)), np.asarray((1, 2)),
        coefficient=coefficient,
    )
    assert scalar == expected
    assert float(vector[0]) == expected
    assert tuple(vector[1:]) == (0.0, 0.0)


def test_half_neighbor_difference_attains_the_declared_smallest_grid_step():
    result = mean_neighbor_difference(
        0.05, (0.05, math.nextafter(0.05, math.inf)), coefficient=1.0,
    )
    assert Fraction(result) == QUANTUM


@pytest.mark.parametrize("epi", (
    (0.5,) * 6,
    (0.05, 0.0625, 0.5, 1.0, 0.125, math.nextafter(0.5, math.inf)),
    (1.0, 0.05, 1.0, 0.05, 1.0, 0.05),
))
def test_actual_full_cpu_pressure_matches_phase_plus_linear_assembly(reference, epi):
    source = np.repeat(np.arange(6), 2)
    target = np.asarray(tuple(j for i in range(6) for j in ((i - 1) % 6, (i + 1) % 6)))
    linear = edge_mean_differences(np.asarray(epi), source, target, coefficient=float(reference.epi_weight))
    actual = compute_fused_gradients_symmetric(
        edge_src=source, edge_dst=target, phase=np.asarray(PHASE), epi=np.asarray(epi),
        vf=np.ones(6), weights={"w_epi": float(reference.epi_weight), "w_phase": float(reference.phase_weight)},
        edge_weight=np.ones(12), accumulate_both_directions=False, use_jit=False,
    )
    for row, epi_pressure, pressure in zip(reference.rows, linear, actual, strict=True):
        assert float(pressure) == float(Fraction(row.phase_contribution) + Fraction(float(epi_pressure)))
        if row.grid_excluded:
            assert pressure != 0.0


@pytest.mark.parametrize("key,value,error", (
    ("phase", [0.0] * 6, TypeError), ("phase", (0.0,) * 5, ValueError),
    ("phase", (0.0,) * 7, ValueError), ("phase", (0.0,) * 5 + (True,), TypeError),
    ("phase", (0.0,) * 5 + (math.nan,), ValueError),
    ("phase", (0.0,) * 5 + (math.inf,), ValueError),
    ("phase", (0.0,) * 5 + (math.tau,), ValueError),
    ("phase", (0.0,) * 5 + (-0.1,), ValueError),
    ("phase", (0.0, math.pi, 0.0, 0.0, 0.0, 0.0), ValueError),
    ("epi_weight", 0.0, ValueError), ("epi_weight", -0.1, ValueError),
    ("epi_weight", 1.01, ValueError), ("epi_weight", math.inf, ValueError),
    ("epi_weight", Fraction(1, 2), TypeError), ("epi_weight", True, TypeError),
    ("phase_weight", 0.0, ValueError), ("phase_weight", 1.01, ValueError),
    ("phase_weight", math.nan, ValueError), ("phase_weight", 1, TypeError),
    ("epi_lower", math.nextafter(0.05, 0.0), ValueError),
    ("epi_lower", 0, TypeError), ("epi_upper", math.nextafter(1.0, math.inf), ValueError),
    ("epi_upper", 0.04, ValueError),
))
def test_invalid_declared_coefficients_phase_or_band_are_rejected(key, value, error):
    with pytest.raises(error):
        _derive(**{key: value})


def test_narrower_band_retains_only_the_sufficient_parent_lattice_obstruction(reference):
    narrowed = _derive(epi_lower=0.5, epi_upper=0.5)
    assert narrowed.rows == reference.rows
    assert narrowed.no_zero_pressure == reference.no_zero_pressure


def test_unsupported_binary64_rounding_cannot_produce_the_theorem(monkeypatch):
    monkeypatch.setattr(owner, "uses_ieee_binary64_rounding", lambda: False)
    with pytest.raises(RuntimeError, match="IEEE"):
        _derive()


def test_missing_cpu_backend_cannot_produce_a_canonical_phase_source(monkeypatch):
    monkeypatch.setattr(owner.fused_dnfr, "np", None)
    with pytest.raises(RuntimeError, match="NumPy"):
        _derive()
