"""Independent rational geometry and modal checks for the declared C6 chart."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
import math

import pytest

from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics.coupling_winding import (
    derive_c6_winding_phase_response, observe_c6_winding_phase_response,
)


F = Fraction
MODES = (
    (0, (1, 1, 1, 1, 1, 1)),
    (1, (2, 1, -1, -2, -1, 1)),
    (1, (0, 1, 1, 0, -1, -1)),
    (2, (2, -1, -1, 2, -1, -1)),
    (2, (0, 1, -1, 0, 1, -1)),
    (3, (1, -1, 1, -1, 1, -1)),
)


def _reference(**overrides):
    factors = {"coupling_phase_factor": F(1, 2), "coherence_phase_factor": F(1, 3)}
    factors.update(overrides)
    return derive_c6_winding_phase_response(**factors)


def test_c6_exact_phasor_geometry_and_hand_first_rows():
    reference = _reference()
    um, il = reference.coupling_response, reference.coherence_response
    assert um.cosine_gram[0] == (1, F(1, 2), F(-1, 2), -1, F(-1, 2), F(1, 2))
    assert um.mean_neighbors[0] == um.receiver_sources[0] == (0, 5, 1)
    assert il.mean_neighbors[0] == (5, 1)
    assert il.receiver_sources[0] == (0,)
    assert um.mean_resultant_squared == (4,) * 6
    assert il.mean_resultant_squared == (1,) * 6
    assert um.mean_response[0] == (F(1, 2), F(1, 4), 0, 0, 0, F(1, 4))
    assert il.mean_response[0] == (0, F(1, 2), 0, 0, 0, F(1, 2))
    assert um.receiver_response[0] == (F(1, 3), F(1, 4), F(1, 12), 0, F(1, 12), F(1, 4))
    assert reference.coupling_matrix[0] == (F(2, 3), F(1, 8), F(1, 24), 0, F(1, 24), F(1, 8))
    assert reference.coherence_matrix[0] == (F(2, 3), F(1, 6), 0, 0, 0, F(1, 6))
    assert reference.product_matrix[0] == (F(35, 72), F(29, 144), F(7, 144), F(1, 72), F(7, 144), F(29, 144))
    assert um.is_nonnegative and il.is_nonnegative
    assert reference.rotation_identity_residuals == (0,) * 6
    assert reference.modal_multipliers == (1, F(5, 8), F(1, 4), F(1, 6), F(1, 4), F(5, 8))
    assert reference.quotient_energy_bound == F(25, 64)
    assert reference.strict_quotient_contraction


@pytest.mark.parametrize("mode,direction", MODES)
def test_all_six_independent_real_fourier_directions(mode, direction):
    reference = _reference()
    response = observe_c6_winding_phase_response(reference, direction=direction)
    factor = (1, F(5, 8), F(1, 4), F(1, 6))[mode]
    assert response.after_coherence == tuple(factor * value for value in direction)
    assert response.energy_after == factor**2 * response.energy_before
    assert response.mean_identity_residual == 0
    assert response.energy_bound_residual <= 0
    laplacian_eigenvalue = (0, F(1, 2), F(3, 2), 2)[mode]
    assert response.phase_pressure_pi_scaled == tuple(-laplacian_eigenvalue * factor * v for v in direction)


def test_mean_rotation_is_preserved_and_excluded_from_the_optimal_energy_bound():
    response = observe_c6_winding_phase_response(_reference(), direction=(5, 4, 2, 1, 2, 4))
    assert response.mean == 3
    assert response.centered_direction == (2, 1, -1, -2, -1, 1)
    assert response.energy_before == 6
    assert response.energy_after == F(75, 32)
    assert response.energy_change == F(-117, 32)
    assert response.energy_bound_residual == 0
    assert response.after_coupling == (F(9, 2), F(15, 4), F(9, 4), F(3, 2), F(9, 4), F(15, 4))
    assert response.after_coherence == (F(17, 4), F(29, 8), F(19, 8), F(7, 4), F(19, 8), F(29, 8))
    assert response.phase_pressure_pi_scaled == (F(-5, 8), F(-5, 16), F(5, 16), F(5, 8), F(5, 16), F(-5, 16))


@pytest.mark.parametrize("t", (F(0), F(1, 4), F(1)))
@pytest.mark.parametrize("alpha", (F(0), F(1, 3), F(1)))
def test_boundary_factors_preserve_modal_contract_including_alternating_neutral_case(t, alpha):
    reference = _reference(coupling_phase_factor=t, coherence_phase_factor=alpha)
    factors = (1, (1 - t / 2) * (1 - alpha / 2),
               (1 - t) * (1 - 3 * alpha / 2), (1 - t) * (1 - 2 * alpha))
    assert reference.quotient_energy_bound == max(value**2 for value in factors[1:])
    assert reference.strict_quotient_contraction == (t > 0 or 0 < alpha < 1)
    for mode, direction in MODES:
        response = observe_c6_winding_phase_response(reference, direction=direction)
        assert response.after_coherence == tuple(factors[mode] * v for v in direction)
        assert response.energy_after <= reference.quotient_energy_bound * response.energy_before


def test_full_coupling_annihilates_higher_modes_without_removing_the_slowest_pair():
    reference = _reference(coupling_phase_factor=1)
    assert reference.modal_multipliers == (1, F(5, 12), 0, 0, 0, F(5, 12))
    for mode, direction in MODES:
        response = observe_c6_winding_phase_response(reference, direction=direction)
        if mode in (2, 3):
            assert response.after_coherence == (0,) * 6
            assert response.phase_pressure_pi_scaled == (0,) * 6


def test_shared_default_factors_are_rationalized_and_have_strict_quotient_contraction():
    t = canonical_glyph_factor_defaults()["UM_theta_push"]
    alpha = DEFAULT_PHASE_LOCKING_COEFFICIENT
    reference = _reference(coupling_phase_factor=t, coherence_phase_factor=alpha)
    assert reference.coupling_phase_factor == F.from_float(t)
    assert reference.coherence_phase_factor == F.from_float(alpha)
    assert reference.quotient_energy_bound == ((1 - F(t) / 2) * (1 - F(alpha) / 2))**2
    assert 0 < reference.quotient_energy_bound < 1


def test_receiver_average_has_three_sources_without_tripling_a_direction():
    reference = _reference()
    direction = tuple(map(F, (3, -2, 4, 0, -1, 2)))
    mean_derivatives = tuple(direction[i] / 2 + (direction[(i - 1) % 6] + direction[(i + 1) % 6]) / 4 for i in range(6))
    incoming = [[] for _ in range(6)]
    for source in range(6):
        for receiver in (source, (source - 1) % 6, (source + 1) % 6):
            incoming[receiver].append(direction[receiver] / 2 + mean_derivatives[source] / 2)
    assert tuple(map(len, incoming)) == (3,) * 6
    expected_um = tuple(sum(row, F(0)) / 3 for row in incoming)
    expected_il = tuple(2 * expected_um[i] / 3 + (expected_um[(i - 1) % 6] + expected_um[(i + 1) % 6]) / 6 for i in range(6))
    response = observe_c6_winding_phase_response(reference, direction=direction)
    assert response.after_coupling == expected_um
    assert response.after_coherence == expected_il
    assert response.energy_bound_residual < 0
    assert sum(response.phase_pressure_pi_scaled) == 0


def test_observer_rebuilds_all_reference_caches_and_revalidates_factors():
    reference = _reference()
    forged = replace(reference, coupling_response=None, coherence_response=None,
                     coupling_matrix=(), coherence_matrix=(), product_matrix=(),
                     modal_multipliers=(), quotient_energy_bound=-1,
                     rotation_identity_residuals=(99,), strict_quotient_contraction=False)
    response = observe_c6_winding_phase_response(forged, direction=(1,) * 6)
    assert response.reference == reference
    assert response.after_coherence == (1,) * 6
    assert response.energy_after == 0
    with pytest.raises(ValueError, match="phase factors"):
        observe_c6_winding_phase_response(replace(reference, coherence_phase_factor=2), direction=(1,) * 6)


@pytest.mark.parametrize("field", ("coupling_phase_factor", "coherence_phase_factor"))
@pytest.mark.parametrize("value", (True, "0.3", complex(1, 0), math.nan, math.inf, F(-1, 9), F(10, 9)))
def test_invalid_phase_factor_is_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        _reference(**{field: value})


@pytest.mark.parametrize("direction", ((), (1,) * 5, (1,) * 7, set(range(6)),
                                       {i: i for i in range(6)}, "123456",
                                       (True, 0, 0, 0, 0, 0), (math.nan, 0, 0, 0, 0, 0)))
def test_invalid_order_or_tangent_coordinates_are_rejected(direction):
    with pytest.raises((TypeError, ValueError)):
        observe_c6_winding_phase_response(_reference(), direction=direction)


def test_reference_type_and_immutable_results():
    with pytest.raises(TypeError, match="C6WindingPhaseReference"):
        observe_c6_winding_phase_response(None, direction=(0,) * 6)
    reference = _reference()
    response = observe_c6_winding_phase_response(reference, direction=(0,) * 6)
    with pytest.raises(FrozenInstanceError):
        reference.quotient_energy_bound = 0
    with pytest.raises(FrozenInstanceError):
        response.energy_after = 1


def test_exact_tangent_coordinates_do_not_require_binary64_representability():
    scale = F(10**400)
    response = observe_c6_winding_phase_response(_reference(), direction=(scale, -scale) * 3)
    assert response.after_coherence == (scale / 6, -scale / 6) * 3
    assert response.energy_before == 3 * scale**2
    assert response.energy_after == scale**2 / 12
