"""Exact finite pairing identities and the limits of their supplied inputs."""

from dataclasses import replace
from fractions import Fraction

import pytest

from tnfr.physics._cycle_algebra import c6_pair_sums
from tnfr.physics.coupling_winding import (
    c6_centered_opposite_pairs,
    derive_c6_winding_joint_domain,
    derive_c6_winding_pairing,
    observe_c6_winding_pairing,
)

F = Fraction
ZERO = (F(0),) * 6
ZERO_PAIRS = (F(0),) * 3
PHASE = tuple(F(value, 1000) for value in (1, 2, -1, 2, -1, 0))
CLOSED = tuple(F(value, 1000) for value in (2, -1, 0, 1, -1, 2))
EPI = tuple(F(value, 100) for value in (49, 52, 50, 48, 51, 50))


def _reference(**overrides):
    parameters = dict(
        coupling_phase_factor=F(1, 4),
        coherence_phase_factor=F(3, 10),
        capacity=1,
        epi_weight=1,
        phase_weight=F(1, 8),
        timestep=F(1, 4),
        epi_lower=F(1, 8),
        epi_upper=F(7, 8),
    )
    parameters.update(overrides)
    return derive_c6_winding_joint_domain(**parameters)


def _step(reference=None, *, phase=PHASE, closed=CLOSED, epi=EPI):
    return observe_c6_winding_pairing(
        _reference() if reference is None else reference,
        phase_before_pi=phase,
        closed_mean_errors_pi=closed,
        epi=epi,
    )


def _multiply(left, right):
    return tuple(
        tuple(
            sum(a * b for a, b in zip(row, column, strict=True))
            for column in zip(*right, strict=True)
        )
        for row in left
    )


def test_projection_receiver_coherence_and_laplacian_identities_independently():
    source = _reference()
    result = derive_c6_winding_pairing(source)
    adjacency = tuple(
        tuple(F((i - j) % 6 in (1, 5)) for j in range(6)) for i in range(6)
    )
    identity = tuple(tuple(F(i == j) for j in range(6)) for i in range(6))
    receiver = tuple(
        tuple((identity[i][j] + adjacency[i][j]) / 3 for j in range(6))
        for i in range(6)
    )
    laplacian = tuple(
        tuple(identity[i][j] - adjacency[i][j] / 2 for j in range(6)) for i in range(6)
    )
    alpha = source.coherence_phase_factor
    coherence = tuple(
        tuple(
            (1 - alpha) * identity[i][j] + alpha * adjacency[i][j] / 2 for j in range(6)
        )
        for i in range(6)
    )
    projection = tuple(
        tuple(F(j in (i, i + 3)) - F(1, 3) for j in range(6)) for i in range(3)
    )
    assert result.receiver_average_matrix == receiver
    assert result.coherence_matrix == coherence
    assert result.projection_matrix == projection
    assert _multiply(projection, receiver) == (ZERO,) * 3
    assert _multiply(projection, coherence) == tuple(
        tuple((1 - 3 * alpha / 2) * x for x in row) for row in projection
    )
    assert _multiply(projection, laplacian) == tuple(
        tuple(3 * x / 2 for x in row) for row in projection
    )
    assert (
        result.receiver_projection_residual
        == result.coherence_projection_residual
        == (ZERO,) * 3
    )
    assert result.laplacian_projection_residual == (ZERO,) * 3


def test_full_finite_map_pressure_and_two_coordinate_quotient_are_exact():
    result = _step()
    source = result.reference.reference
    t, alpha = source.coupling_phase_factor, source.coherence_phase_factor
    coupled = tuple(
        (1 - t) * PHASE[i] + t * (CLOSED[i - 1] + CLOSED[i] + CLOSED[(i + 1) % 6]) / 3
        for i in range(6)
    )
    after = tuple(
        (1 - alpha) * coupled[i] + alpha * (coupled[i - 1] + coupled[(i + 1) % 6]) / 2
        for i in range(6)
    )
    phase_pressure = tuple(
        (after[i - 1] + after[(i + 1) % 6]) / 2 - after[i] for i in range(6)
    )
    pressure = tuple(
        source.epi_weight * ((EPI[i - 1] + EPI[(i + 1) % 6]) / 2 - EPI[i])
        + source.phase_weight * phase_pressure[i]
        for i in range(6)
    )
    endpoint = tuple(
        EPI[i] + source.timestep * source.capacity * pressure[i] for i in range(6)
    )
    assert result.phase_after_coupling_pi == coupled
    assert result.phase_after_coherence_pi == after
    assert result.phase_pressure == phase_pressure
    assert result.modeled_pressure == pressure and result.epi_after == endpoint
    assert any(result.phase_pair_before) and any(result.epi_pair_before)
    for i in range(3):
        original = (result.phase_pair_before[i], result.epi_pair_before[i])
        predicted = tuple(
            sum(a * b for a, b in zip(row, original, strict=True))
            for row in result.reference.quotient_matrix
        )
        assert predicted == (result.phase_pair_after[i], result.epi_pair_after[i])
    assert (
        result.phase_pair_identity_residual
        == result.epi_pair_identity_residual
        == ZERO_PAIRS
    )
    assert (
        result.phase_pressure_pair_identity_residual
        == result.total_pressure_pair_identity_residual
        == ZERO_PAIRS
    )
    assert sum(result.modeled_pressure) == 0
    assert result.epi_mean_after == result.epi_mean_before


def test_centered_pairing_survives_supplied_common_mean_drift():
    a, b = F(1, 100), F(1, 200)
    initial = (a, 0, b, -a, 0, -b)
    result = _step(phase=initial, closed=(a,) * 6, epi=(F(1, 2),) * 6)
    # Such supplied means are not authenticated as the actual phasor means.
    expected_mean = result.reference.reference.coupling_phase_factor * a
    assert result.phase_mean_before_pi == 0
    assert (
        result.phase_mean_after_coupling_pi
        == result.phase_mean_after_coherence_pi
        == expected_mean
    )
    assert c6_pair_sums(result.phase_after_coherence_pi) == (2 * expected_mean,) * 3
    assert result.phase_pair_before == result.phase_pair_after == ZERO_PAIRS
    assert result.phase_pressure_pairs == result.total_pressure_pairs == ZERO_PAIRS
    assert result.epi_pair_after == ZERO_PAIRS
    assert result.phase_mean_identity_residual == result.epi_mean_identity_residual == 0


def test_common_rotation_changes_only_phase_means_in_the_readout():
    shift = F(123, 7)
    initial = _step()
    shifted = _step(
        phase=tuple(value + shift for value in PHASE),
        closed=tuple(value + shift for value in CLOSED),
    )
    assert (
        shifted.phase_mean_after_coherence_pi
        == initial.phase_mean_after_coherence_pi + shift
    )
    assert shifted.phase_after_coherence_pi == tuple(
        value + shift for value in initial.phase_after_coherence_pi
    )
    for field in (
        "phase_pair_before",
        "phase_pair_after",
        "phase_pressure",
        "modeled_pressure",
        "epi_after",
    ):
        assert getattr(shifted, field) == getattr(initial, field)


def test_interval_only_closed_means_need_not_contract_rho_or_preserve_the_next_reserve():
    source = _reference(
        coupling_phase_factor=1,
        coherence_phase_factor=0,
        phase_weight=1,
        epi_lower=F(1, 20),
        epi_upper=1,
    )
    a = F(1, 120)
    phase = (a, a, 0, 0, 0, a)
    result = _step(source, phase=phase, closed=phase, epi=(F(61, 960),) * 6)
    assert result.phase_oscillation_before == result.phase_oscillation_after == a
    assert (
        result.phase_contraction_slack < 0
        and not result.meets_nonlinear_phase_contraction
    )
    assert result.lower_reserve_before == source.epi_lower
    assert result.lower_reserve_after == source.epi_lower - a / 12
    assert not result.epi_reserve_preserved
    assert min(result.epi_after) >= source.epi_lower
    assert (
        result.phase_pair_identity_residual
        == result.epi_pair_identity_residual
        == ZERO_PAIRS
    )
    with pytest.raises(ValueError, match="reserve"):
        _step(
            source,
            phase=result.phase_after_coherence_pi,
            closed=result.phase_after_coherence_pi,
            epi=result.epi_after,
        )


def test_pair_quotient_decay_does_not_imply_full_state_decay_at_s_one():
    source = _reference(timestep=1)
    initial = (F(1, 4), F(3, 4)) * 3
    first = _step(source, phase=ZERO, closed=ZERO, epi=initial)
    second = _step(source, phase=ZERO, closed=ZERO, epi=first.epi_after)
    assert first.reference.strict_quotient_decay
    assert first.reference.epi_pair_factor == F(-1, 2)
    assert not source.strict_epi_convergence
    assert first.epi_pair_before == first.epi_pair_after == ZERO_PAIRS
    assert first.epi_after != initial and second.epi_after == initial


def test_zero_duration_keeps_epi_and_prevents_strict_quotient_decay():
    result = _step(_reference(timestep=0))
    assert result.epi_after == EPI and any(result.modeled_rate)
    assert result.reference.quotient_matrix[1] == (0, 1)
    assert result.reference.quotient_spectral_radius == 1
    assert not result.reference.strict_quotient_decay
    assert result.phase_after_coherence_pi != PHASE


@pytest.mark.parametrize(
    "t,alpha,expected",
    (
        (F(1, 4), 0, F(3, 4)),
        (F(1, 4), F(2, 3), 0),
        (F(1, 4), 1, F(-3, 8)),
        (1, F(3, 10), 0),
    ),
)
def test_signed_phase_pair_factor_including_annihilation_and_reversal(
    t, alpha, expected
):
    result = _step(_reference(coupling_phase_factor=t, coherence_phase_factor=alpha))
    assert result.reference.phase_pair_factor == expected
    assert result.phase_pair_after == tuple(
        expected * value for value in result.phase_pair_before
    )
    assert result.reference.strict_quotient_decay


def test_zero_phase_weight_removes_only_the_off_diagonal_epi_forcing():
    result = _step(_reference(phase_weight=0))
    assert any(result.phase_pressure_pairs)
    assert result.reference.postphase_to_epi_pair_factor == 0
    assert result.reference.quotient_matrix[1][0] == 0
    assert result.epi_pair_after == tuple(
        result.reference.epi_pair_factor * value for value in result.epi_pair_before
    )


def test_public_outer_and_nested_reference_caches_are_rebuilt():
    source = _reference()
    correct = _step(source)
    phase = replace(source.phase_reference, coherence_matrix=(), modal_multipliers=())
    forged_source = replace(
        source, phase_reference=phase, nodal_euler_matrix=(), epi_phase_budget_weight=0
    )
    assert derive_c6_winding_pairing(forged_source) == correct.reference
    forged = replace(
        correct.reference,
        reference=forged_source,
        projection_matrix=(),
        receiver_average_matrix=(),
        coherence_matrix=(),
        quotient_matrix=(),
        phase_pair_factor=99,
        epi_pair_factor=99,
        quotient_spectral_radius=99,
        strict_quotient_decay=False,
    )
    assert _step(forged) == correct
    with pytest.raises(ValueError):
        _step(replace(forged, reference=replace(source, epi_weight=0)))


def test_a_forged_zero_budget_cannot_admit_an_unfunded_initial_field():
    source = _reference()
    forged = replace(
        derive_c6_winding_pairing(source),
        reference=replace(source, epi_phase_budget_weight=0),
    )
    with pytest.raises(ValueError, match="reserve"):
        _step(forged, epi=(source.epi_lower,) * 6)


def test_projection_preserves_represented_real_values_and_removes_the_mean():
    values = (0.1, F(1, 7), 2, 0.2, F(-1, 7), -2)
    exact = tuple(F(value) for value in values)
    expected = tuple(exact[i] + exact[i + 3] - sum(exact) / 3 for i in range(3))
    assert c6_centered_opposite_pairs(iter(values)) == expected
    assert sum(expected) == 0


@pytest.mark.parametrize("values", ((0,) * 5, (0,) * 7))
def test_private_pair_kernel_refuses_truncation(values):
    with pytest.raises(ValueError, match="six"):
        c6_pair_sums(values)


@pytest.mark.parametrize(
    "values",
    (
        "000000",
        {i: 0 for i in range(6)},
        set(range(6)),
        (0,) * 5,
        (0,) * 7,
        (0, 0, 0, 0, 0, True),
        (0, 0, 0, 0, 0, float("nan")),
        (0, 0, 0, 0, 0, complex(1, 1)),
    ),
)
def test_public_projection_rejects_ambiguous_or_invalid_coordinates(values):
    with pytest.raises((ValueError, TypeError)):
        c6_centered_opposite_pairs(values)


@pytest.mark.parametrize("field", ("phase", "closed", "epi"))
def test_observation_vectors_cannot_silently_zip_truncate(field):
    with pytest.raises(ValueError, match="six"):
        _step(**{field: ZERO[:-1]})


def test_phase_chamber_and_closed_mean_interval_guards_are_independent():
    with pytest.raises(ValueError, match="winding box"):
        _step(phase=(F(1, 6), 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="inside the initial phase interval"):
        _step(closed=(F(1, 10),) * 6)
    with pytest.raises(TypeError, match="C6WindingJointDomain"):
        _step(reference={})
