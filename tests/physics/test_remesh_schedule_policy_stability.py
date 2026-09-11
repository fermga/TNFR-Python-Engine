"""Exact uniform policy theorem for REMESH followed by schedule maps."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics.remesh_schedule_policy_stability as policy_module
from tnfr.errors import TNFRValueError
from tnfr.physics._exact_linear_algebra import (
    ExactSquareMatrix,
    exact_square_matrix_power,
    exact_square_matrix_product,
)
from tnfr.physics.remesh_history_stability import (
    UniformRemeshHistoryStabilityCertificate,
    certify_uniform_remesh_history_stability,
)
from tnfr.physics.remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)


def _source(
    *,
    alpha: Fraction = Fraction(1, 2),
    tau_local: int = 2,
    tau_global: int = 4,
) -> UniformRemeshHistoryStabilityCertificate:
    return certify_uniform_remesh_history_stability(
        alpha=alpha,
        tau_local=tau_local,
        tau_global=tau_global,
    )


def _diagonal_head_gain(q: Fraction, dimension: int) -> ExactSquareMatrix:
    return tuple(
        tuple(
            (q if row == 0 else Fraction(1))
            if row == column
            else Fraction(0)
            for column in range(dimension)
        )
        for row in range(dimension)
    )


def _scaled(q: Fraction, matrix: ExactSquareMatrix) -> ExactSquareMatrix:
    return tuple(tuple(q * entry for entry in row) for row in matrix)


def _entrywise_at_most(
    left: ExactSquareMatrix,
    right: ExactSquareMatrix,
) -> bool:
    return all(
        left_entry <= right_entry
        for left_row, right_row in zip(left, right, strict=True)
        for left_entry, right_entry in zip(left_row, right_row, strict=True)
    )


def _left_action(
    vector: tuple[Fraction, ...],
    matrix: ExactSquareMatrix,
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (
                vector[row] * matrix[row][column]
                for row in range(len(matrix))
            ),
            Fraction(0),
        )
        for column in range(len(matrix))
    )


def _zero_matrix(dimension: int) -> ExactSquareMatrix:
    return tuple(
        tuple(Fraction(0) for _column in range(dimension))
        for _row in range(dimension)
    )


def _matrix_vector(
    matrix: ExactSquareMatrix,
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    return tuple(
        sum(
            (entry * value for entry, value in zip(row, vector, strict=True)),
            Fraction(0),
        )
        for row in matrix
    )


def _spatial_disagreement_energy(vector: tuple[Fraction, ...]) -> Fraction:
    mean = sum(vector, Fraction(0)) / len(vector)
    return sum(((value - mean) ** 2 for value in vector), Fraction(0)) / 2


def _augmented_energy(
    history: tuple[tuple[Fraction, ...], ...],
    temporal_weights: tuple[Fraction, ...],
) -> Fraction:
    return sum(
        (
            weight * _spatial_disagreement_energy(row)
            for weight, row in zip(temporal_weights, history, strict=True)
        ),
        Fraction(0),
    )


def test_direct_module_and_stub_expose_policy_api() -> None:
    assert policy_module.__all__ == (
        "UniformRemeshSchedulePolicyStabilityCertificate",
        "certify_uniform_remesh_schedule_policy_stability",
    )
    package = Path(policy_module.__file__).parent
    stub = (package / "remesh_schedule_policy_stability.pyi").read_text(
        encoding="utf-8"
    )
    assert "class UniformRemeshSchedulePolicyStabilityCertificate" in stub
    assert "def certify_uniform_remesh_schedule_policy_stability" in stub
    assert "def exact_cycle_energy_gain_upper_bound" in stub


def test_exact_policy_envelope_has_DqP_order_and_expected_interior_rows() -> None:
    source = _source()
    q = Fraction(2, 3)
    certificate = certify_uniform_remesh_schedule_policy_stability(source, q)

    assert type(certificate) is UniformRemeshSchedulePolicyStabilityCertificate
    assert certificate.remesh_certificate is source
    assert certificate.schedule_energy_gain_upper_bound == q
    assert certificate.history_length == 5
    assert certificate.universal_block_horizon == 5
    assert certificate.remesh_companion_matrix == (
        (
            Fraction(1, 4),
            Fraction(0),
            Fraction(1, 4),
            Fraction(0),
            Fraction(1, 2),
        ),
        (Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
    )

    diagonal = _diagonal_head_gain(q, certificate.history_length)
    expected = exact_square_matrix_product(
        diagonal,
        certificate.remesh_companion_matrix,
    )
    assert certificate.schedule_energy_domination_matrix == expected
    assert certificate.schedule_energy_domination_matrix[0] == (
        Fraction(1, 6),
        Fraction(0),
        Fraction(1, 6),
        Fraction(0),
        Fraction(1, 3),
    )
    assert certificate.schedule_energy_domination_matrix[1:] == (
        certificate.remesh_companion_matrix[1:]
    )
    assert certificate.policy_stability_certificate_certified
    assert certificate.failed_conditions == ()


def test_exact_prefix_powers_and_stationary_inequalities_are_reproducible() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    p = certificate.remesh_companion_matrix
    b = certificate.schedule_energy_domination_matrix
    pi = certificate.remesh_certificate.stationary_distribution
    q = certificate.schedule_energy_gain_upper_bound
    horizon = certificate.universal_block_horizon

    for exponent in range(horizon + 1):
        p_power = exact_square_matrix_power(p, exponent)
        b_power = exact_square_matrix_power(b, exponent)
        assert _entrywise_at_most(b_power, p_power)
        assert all(
            observed <= reference
            for observed, reference in zip(
                _left_action(pi, b_power),
                pi,
                strict=True,
            )
        )

    assert _entrywise_at_most(
        certificate.schedule_block_domination_power,
        _scaled(q, certificate.remesh_block_power),
    )
    assert all(
        observed <= q * reference
        for observed, reference in zip(
            _left_action(
                pi,
                certificate.schedule_block_domination_power,
            ),
            pi,
            strict=True,
        )
    )
    assert (
        certificate
        .conditional_policy_family_spatial_disagreement_nonincrease_certified
    )
    assert certificate.uniform_intrablock_prefix_bound_certified
    assert certificate.repeated_exact_model_spatial_disagreement_stability_certified


def test_universal_horizon_closes_every_head_avoiding_companion_path() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(3, 4),
    )
    horizon = certificate.universal_block_horizon
    avoidance = certificate.head_avoidance_matrix

    assert certificate.head_avoidance_block_power == _zero_matrix(horizon)
    assert exact_square_matrix_power(avoidance, horizon) == _zero_matrix(horizon)
    # The shift chain still has a head-avoiding path one step earlier.  This
    # makes L a sharp universal path bound; it does not claim that every
    # concrete interior-alpha orbit needs L cycles to contract.
    assert exact_square_matrix_power(avoidance, horizon - 1) != _zero_matrix(
        horizon
    )


def test_q_zero_annihilates_the_energy_envelope_within_one_block() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(_source(), 0)
    horizon = certificate.universal_block_horizon

    assert certificate.schedule_energy_gain_upper_bound == 0
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == 1
    assert certificate.exact_uniform_block_energy_gain_upper_bound == 0
    assert certificate.exact_intrablock_prefix_energy_gain_upper_bound == 1
    assert certificate.schedule_block_domination_power == _zero_matrix(horizon)
    assert certificate.uniform_positive_normalized_block_margin_certified
    assert certificate.geometric_spatial_disagreement_convergence_certified
    assert certificate.exact_cycle_energy_gain_upper_bound(horizon - 1) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(horizon) == 0
    assert certificate.exact_cycle_energy_gain_upper_bound(9 * horizon) == 0


def test_q_one_is_a_valid_zero_margin_boundary_without_convergence_claim() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(_source(), 1)

    assert certificate.schedule_energy_gain_upper_bound == 1
    assert certificate.schedule_energy_domination_matrix == (
        certificate.remesh_companion_matrix
    )
    assert certificate.schedule_block_domination_power == (
        certificate.remesh_block_power
    )
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == 0
    assert certificate.exact_uniform_block_energy_gain_upper_bound == 1
    assert (
        certificate
        .conditional_policy_family_spatial_disagreement_nonincrease_certified
    )
    assert certificate.repeated_exact_model_spatial_disagreement_stability_certified
    assert not certificate.uniform_positive_normalized_block_margin_certified
    assert not certificate.geometric_spatial_disagreement_convergence_certified
    assert certificate.q_one_zero_margin_boundary_certified
    assert certificate.exact_cycle_energy_gain_upper_bound(10_000) == 1


def test_alpha_zero_reduces_to_one_step_schedule_gain() -> None:
    q = Fraction(3, 7)
    source = _source(alpha=Fraction(0), tau_local=7, tau_global=9)
    certificate = certify_uniform_remesh_schedule_policy_stability(source, q)

    assert source.companion_matrix == ((Fraction(1),),)
    assert certificate.history_length == 1
    assert certificate.universal_block_horizon == 1
    assert certificate.schedule_energy_domination_matrix == ((q,),)
    assert certificate.head_avoidance_matrix == ((Fraction(0),),)
    assert certificate.exact_cycle_energy_gain_upper_bound(0) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(1) == q
    assert certificate.exact_cycle_energy_gain_upper_bound(4) == q**4


def test_alpha_one_pure_delay_is_damped_once_per_delay_cycle() -> None:
    q = Fraction(3, 5)
    source = _source(alpha=Fraction(1), tau_local=4, tau_global=2)
    certificate = certify_uniform_remesh_schedule_policy_stability(source, q)
    identity = exact_square_matrix_power(source.companion_matrix, 0)

    assert source.companion_matrix == (
        (Fraction(0), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    assert certificate.universal_block_horizon == 3
    assert certificate.remesh_block_power == identity
    assert certificate.schedule_block_domination_power == _scaled(q, identity)
    assert certificate.alpha_one_spatial_disagreement_decay_certified
    assert certificate.exact_cycle_energy_gain_upper_bound(2) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(3) == q
    assert certificate.exact_cycle_energy_gain_upper_bound(8) == q**2


def test_alpha_one_q_one_retains_only_the_zero_margin_boundary() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(alpha=Fraction(1), tau_local=9, tau_global=2),
        1,
    )

    assert certificate.q_one_zero_margin_boundary_certified
    assert not certificate.alpha_one_spatial_disagreement_decay_certified
    assert not certificate.geometric_spatial_disagreement_convergence_certified


def test_coincident_delays_are_supported_after_source_combination() -> None:
    source = _source(tau_local=2, tau_global=2)
    certificate = certify_uniform_remesh_schedule_policy_stability(
        source,
        Fraction(4, 5),
    )

    assert source.combined_delay_coefficients == (
        (0, Fraction(1, 4)),
        (2, Fraction(3, 4)),
    )
    assert certificate.history_length == 3
    assert certificate.universal_block_horizon == 3
    assert certificate.policy_stability_certificate_certified


def test_varying_exact_schedule_maps_obey_every_prefix_and_block_bound() -> None:
    source = _source(tau_local=1, tau_global=2)
    q = Fraction(1, 4)
    certificate = certify_uniform_remesh_schedule_policy_stability(source, q)
    j = (
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    identity = (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    quotient = tuple(
        tuple(identity[row][column] - j[row][column] for column in range(2))
        for row in range(2)
    )
    schedules = (
        tuple(
            tuple(j[row][column] + quotient[row][column] / 2 for column in range(2))
            for row in range(2)
        ),
        tuple(
            tuple(j[row][column] - quotient[row][column] / 3 for column in range(2))
            for row in range(2)
        ),
    )
    history = (
        (Fraction(3), Fraction(-1)),
        (Fraction(-2), Fraction(4)),
        (Fraction(5), Fraction(0)),
    )
    coefficients = dict(source.combined_delay_coefficients)
    temporal_weights = source.stationary_distribution
    initial_energy = _augmented_energy(history, temporal_weights)
    assert initial_energy > 0

    for cycle in range(1, 13):
        remesh_head = tuple(
            sum(
                (
                    coefficient * history[delay][column]
                    for delay, coefficient in coefficients.items()
                ),
                Fraction(0),
            )
            for column in range(2)
        )
        scheduled_head = _matrix_vector(
            schedules[(cycle - 1) % len(schedules)],
            remesh_head,
        )
        previous_energy = _augmented_energy(history, temporal_weights)
        history = (scheduled_head, *history[:-1])
        observed_energy = _augmented_energy(history, temporal_weights)
        assert observed_energy <= previous_energy
        assert observed_energy <= (
            certificate.exact_cycle_energy_gain_upper_bound(cycle)
            * initial_energy
        )


def test_spatial_disagreement_theorem_does_not_control_consensus_amplitude() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(tau_local=1, tau_global=2),
        Fraction(1, 4),
    )
    expansive_consensus_schedule = (
        (Fraction(5, 4), Fraction(3, 4)),
        (Fraction(3, 4), Fraction(5, 4)),
    )
    uniform = (Fraction(1), Fraction(1))
    mapped = _matrix_vector(expansive_consensus_schedule, uniform)

    assert mapped == (Fraction(2), Fraction(2))
    assert _spatial_disagreement_energy(uniform) == 0
    assert _spatial_disagreement_energy(mapped) == 0
    assert certificate.geometric_spatial_disagreement_convergence_certified
    assert not certificate.full_tnfr_stability_certified


@pytest.mark.parametrize(
    ("cycle_count", "expected"),
    (
        (0, Fraction(1)),
        (1, Fraction(1)),
        (4, Fraction(1)),
        (5, Fraction(2, 3)),
        (9, Fraction(2, 3)),
        (10, Fraction(4, 9)),
        (27, Fraction(32, 243)),
    ),
)
def test_cycle_gain_uses_complete_universal_blocks_only(
    cycle_count: int,
    expected: Fraction,
) -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    assert certificate.exact_cycle_energy_gain_upper_bound(cycle_count) == expected


@pytest.mark.parametrize("cycle_count", (True, -1, 1.0, "5", None))
def test_cycle_gain_rejects_invalid_counts(cycle_count: object) -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    with pytest.raises(TNFRValueError, match="nonnegative integer"):
        certificate.exact_cycle_energy_gain_upper_bound(
            cycle_count  # type: ignore[arg-type]
        )


def test_cycle_gain_rejects_a_tampered_policy_certificate() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    object.__setattr__(certificate, "universal_block_horizon", 4)

    assert not certificate.policy_stability_certificate_certified
    with pytest.raises(TNFRValueError, match="unsealed or inconsistent"):
        certificate.exact_cycle_energy_gain_upper_bound(8)


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("history_length", 4),
        ("universal_block_horizon", 4),
        ("schedule_energy_gain_upper_bound", Fraction(1, 3)),
        ("exact_uniform_normalized_block_margin_lower_bound", Fraction(99)),
        ("exact_uniform_block_energy_gain_upper_bound", Fraction(99)),
        ("exact_intrablock_prefix_energy_gain_upper_bound", Fraction(99)),
        ("conditions", (("forged", True),)),
    ),
)
def test_direct_field_tampering_fails_closed(
    field_name: str,
    forged_value: object,
) -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    object.__setattr__(certificate, field_name, forged_value)

    assert not certificate.policy_stability_certificate_certified
    assert certificate.failed_conditions == (
        "remesh_schedule_policy_proof_fields_intact",
    )


def test_nested_source_tampering_invalidates_the_policy_certificate() -> None:
    source = _source()
    certificate = certify_uniform_remesh_schedule_policy_stability(
        source,
        Fraction(2, 3),
    )
    object.__setattr__(source, "beta", Fraction(99))

    assert not source.stability_certificate_certified
    assert not certificate.policy_stability_certificate_certified
    assert certificate.failed_conditions == (
        "remesh_schedule_policy_proof_fields_intact",
    )


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        ("history_length", 4),
        ("schedule_energy_domination_matrix", ((Fraction(1),),)),
        ("remesh_block_power", ((Fraction(1),),)),
        ("schedule_block_domination_power", ((Fraction(1),),)),
        ("head_avoidance_block_power", ((Fraction(1),),)),
        ("conditions", (("forged", True),)),
    ),
)
def test_private_reseal_cannot_promote_inconsistent_derivatives(
    field_name: str,
    forged_value: object,
) -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    forged = replace(
        certificate,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    resealed = policy_module._seal(forged)

    assert not resealed.policy_stability_certificate_certified
    assert resealed.failed_conditions == (
        "remesh_schedule_policy_proof_fields_intact",
    )


def test_tampered_hostile_numeric_field_is_rejected_without_dispatch() -> None:
    marker: list[str] = []

    class HostileGain:
        def __le__(self, other: object) -> bool:
            del other
            marker.append("numeric comparison dispatched")
            raise SystemExit("must not escape proof validation")

        def __ge__(self, other: object) -> bool:
            del other
            marker.append("numeric comparison dispatched")
            raise SystemExit("must not escape proof validation")

    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    object.__setattr__(
        certificate,
        "schedule_energy_gain_upper_bound",
        HostileGain(),
    )

    assert not certificate.policy_stability_certificate_certified
    assert marker == []


def test_hostile_proof_stamp_is_rejected_without_equality_dispatch() -> None:
    marker: list[str] = []

    class HostileStamp:
        def __eq__(self, other: object) -> bool:
            del other
            marker.append("caller equality dispatched")
            raise SystemExit("must not escape proof validation")

    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )
    object.__setattr__(
        certificate,
        "_proof_stamp",
        ("uniform_remesh_schedule_policy_stability_v1", HostileStamp()),
    )

    assert not certificate.policy_stability_certificate_certified
    assert marker == []


@pytest.mark.parametrize(
    "gain",
    (
        True,
        -1,
        Fraction(-1, 100),
        Fraction(101, 100),
        1.0000000000000002,
        float("nan"),
        float("inf"),
        "0.5",
        None,
    ),
)
def test_invalid_schedule_energy_gain_is_rejected(gain: object) -> None:
    with pytest.raises(TNFRValueError, match="schedule_energy_gain_upper_bound"):
        certify_uniform_remesh_schedule_policy_stability(
            _source(),
            gain,  # type: ignore[arg-type]
        )


def test_binary64_gain_is_rationalized_exactly() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(_source(), 0.1)
    assert certificate.schedule_energy_gain_upper_bound == Fraction.from_float(0.1)
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == (
        Fraction(1) - Fraction.from_float(0.1)
    )


def test_nonreal_hostile_gain_is_rejected_without_conversion_dispatch() -> None:
    marker: list[str] = []

    class HostileGain:
        def __float__(self) -> float:
            marker.append("float conversion dispatched")
            raise SystemExit("must not escape")

        def __eq__(self, other: object) -> bool:
            del other
            marker.append("equality dispatched")
            raise SystemExit("must not escape")

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        certify_uniform_remesh_schedule_policy_stability(
            _source(),
            HostileGain(),  # type: ignore[arg-type]
        )
    assert marker == []


def test_wrong_or_tampered_source_certificate_is_rejected() -> None:
    with pytest.raises(TNFRValueError, match="exact uniform REMESH certificate"):
        certify_uniform_remesh_schedule_policy_stability(
            object(),  # type: ignore[arg-type]
            Fraction(1, 2),
        )

    source = _source()
    object.__setattr__(source, "stationary_denominator", Fraction(99))
    with pytest.raises(TNFRValueError, match="unsealed, tampered, or inconsistent"):
        certify_uniform_remesh_schedule_policy_stability(
            source,
            Fraction(1, 2),
        )


def test_scope_keeps_conditional_exact_theorem_boundaries_explicit() -> None:
    certificate = certify_uniform_remesh_schedule_policy_stability(
        _source(),
        Fraction(2, 3),
    )

    assert "immediately after each schedule" in policy_module.__doc__
    assert "possibly varying" in policy_module.__doc__
    assert "does not verify any schedule map" in certificate.scope
    assert "binary64 runtime" in certificate.scope
    assert "adaptive U2/U4" in certificate.scope
    assert not certificate.runtime_schedule_maps_verified
    assert not certificate.binary64_runtime_stability_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.solver_order_certified
    assert not certificate.adaptive_grammar_certified
    assert not certificate.full_tnfr_stability_certified
