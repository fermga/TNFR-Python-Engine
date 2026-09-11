"""Robust exact REMESH/schedule stability under a relative head defect."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics.remesh_schedule_relative_defect_stability as robust_module
from tnfr.errors import TNFRValueError
from tnfr.physics._exact_linear_algebra import (
    ExactSquareMatrix,
    exact_square_matrix_power,
    exact_square_matrix_product,
)
from tnfr.physics.remesh_history_stability import (
    certify_uniform_remesh_history_stability,
)
from tnfr.physics.remesh_schedule_policy_stability import (
    UniformRemeshSchedulePolicyStabilityCertificate,
    certify_uniform_remesh_schedule_policy_stability,
)
from tnfr.physics.remesh_schedule_relative_defect_stability import (
    UniformRemeshScheduleRelativeDefectStabilityCertificate,
    certify_uniform_remesh_schedule_relative_defect_stability,
)


def _policy(
    *,
    alpha: Fraction = Fraction(1, 2),
    tau_local: int = 2,
    tau_global: int = 4,
    q: Fraction = Fraction(1, 2),
) -> UniformRemeshSchedulePolicyStabilityCertificate:
    remesh = certify_uniform_remesh_history_stability(
        alpha=alpha,
        tau_local=tau_local,
        tau_global=tau_global,
    )
    return certify_uniform_remesh_schedule_policy_stability(remesh, q)


def _certificate(
    *,
    alpha: Fraction = Fraction(1, 2),
    tau_local: int = 2,
    tau_global: int = 4,
    q: Fraction = Fraction(1, 2),
    eta: Fraction = Fraction(1, 2),
) -> UniformRemeshScheduleRelativeDefectStabilityCertificate:
    return certify_uniform_remesh_schedule_relative_defect_stability(
        _policy(
            alpha=alpha,
            tau_local=tau_local,
            tau_global=tau_global,
            q=q,
        ),
        eta,
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


def test_direct_module_and_stub_expose_relative_defect_api() -> None:
    assert robust_module.__all__ == (
        "UniformRemeshScheduleRelativeDefectStabilityCertificate",
        "certify_uniform_remesh_schedule_relative_defect_stability",
    )
    package = Path(robust_module.__file__).parent
    stub = (
        package / "remesh_schedule_relative_defect_stability.pyi"
    ).read_text(encoding="utf-8")
    assert "class UniformRemeshScheduleRelativeDefectStabilityCertificate" in stub
    assert "def certify_uniform_remesh_schedule_relative_defect_stability" in stub
    assert "def exact_cycle_energy_gain_upper_bound" in stub


def test_effective_gain_is_exact_q_times_one_plus_eta() -> None:
    policy = _policy(q=Fraction(2, 5))
    eta = Fraction(1, 2)
    certificate = certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        eta,
    )

    assert type(certificate) is (
        UniformRemeshScheduleRelativeDefectStabilityCertificate
    )
    assert certificate.policy_certificate is policy
    assert certificate.remesh_certificate is policy.remesh_certificate
    assert certificate.pre_schedule_relative_energy_defect_upper_bound == eta
    assert certificate.schedule_energy_gain_upper_bound == Fraction(2, 5)
    assert certificate.exact_effective_head_energy_gain_upper_bound == Fraction(3, 5)
    assert certificate.effective_head_energy_gain_upper_bound == Fraction(3, 5)
    assert (
        certificate.effective_head_gain_envelope_certificate
        .schedule_energy_gain_upper_bound
        == Fraction(3, 5)
    )
    assert certificate.relative_defect_stability_certificate_certified
    assert certificate.failed_conditions == ()


def test_effective_envelope_reuses_DqeffP_in_the_exact_order() -> None:
    certificate = _certificate(q=Fraction(1, 2), eta=Fraction(1, 2))
    q_effective = Fraction(3, 4)
    companion = certificate.remesh_companion_matrix
    diagonal = _diagonal_head_gain(q_effective, certificate.history_length)

    assert certificate.effective_head_energy_domination_matrix == (
        exact_square_matrix_product(diagonal, companion)
    )
    assert certificate.effective_head_energy_domination_matrix == (
        certificate.effective_head_gain_envelope_certificate
        .schedule_energy_domination_matrix
    )
    assert certificate.effective_head_energy_domination_matrix[0] == tuple(
        q_effective * entry for entry in companion[0]
    )
    assert certificate.effective_head_energy_domination_matrix[1:] == companion[1:]


def test_prefix_and_universal_block_inequalities_use_effective_gain() -> None:
    certificate = _certificate(q=Fraction(2, 5), eta=Fraction(1, 2))
    p = certificate.remesh_companion_matrix
    b = certificate.effective_head_energy_domination_matrix
    q_effective = Fraction(3, 5)
    horizon = certificate.universal_block_horizon

    for exponent in range(horizon + 1):
        assert _entrywise_at_most(
            exact_square_matrix_power(b, exponent),
            exact_square_matrix_power(p, exponent),
        )
    assert _entrywise_at_most(
        certificate.effective_head_block_domination_power,
        _scaled(q_effective, certificate.remesh_block_power),
    )
    assert certificate.exact_uniform_block_energy_gain_upper_bound == q_effective
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == Fraction(
        2,
        5,
    )
    assert certificate.exact_intrablock_prefix_energy_gain_upper_bound == 1
    assert (
        certificate
        .conditional_exact_model_spatial_disagreement_nonincrease_certified
    )
    assert certificate.uniform_intrablock_prefix_bound_certified
    assert certificate.repeated_exact_model_spatial_disagreement_stability_certified
    assert certificate.uniform_positive_normalized_block_margin_certified
    assert certificate.geometric_spatial_disagreement_convergence_certified


def test_eta_zero_reduces_exactly_to_the_supplied_policy() -> None:
    policy = _policy(q=Fraction(5, 7))
    certificate = certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        0,
    )

    assert certificate.pre_schedule_relative_energy_defect_upper_bound == 0
    assert certificate.exact_effective_head_energy_gain_upper_bound == Fraction(5, 7)
    assert certificate.effective_head_energy_domination_matrix == (
        policy.schedule_energy_domination_matrix
    )
    assert certificate.remesh_block_power == policy.remesh_block_power
    assert certificate.effective_head_block_domination_power == (
        policy.schedule_block_domination_power
    )
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == Fraction(
        2,
        7,
    )


def test_q_zero_absorbs_any_finite_nonnegative_relative_defect() -> None:
    certificate = _certificate(q=Fraction(0), eta=Fraction(10**40))
    horizon = certificate.universal_block_horizon

    assert certificate.exact_effective_head_energy_gain_upper_bound == 0
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == 1
    assert certificate.exact_uniform_block_energy_gain_upper_bound == 0
    assert certificate.q_zero_preschedule_defect_absorption_certified
    assert certificate.uniform_positive_normalized_block_margin_certified
    assert certificate.geometric_spatial_disagreement_convergence_certified
    assert certificate.exact_cycle_energy_gain_upper_bound(horizon - 1) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(horizon) == 0


def test_effective_gain_one_is_a_valid_zero_margin_boundary_only() -> None:
    certificate = _certificate(q=Fraction(2, 3), eta=Fraction(1, 2))

    assert certificate.exact_effective_head_energy_gain_upper_bound == 1
    assert certificate.effective_head_energy_domination_matrix == (
        certificate.remesh_companion_matrix
    )
    assert certificate.exact_uniform_normalized_block_margin_lower_bound == 0
    assert certificate.exact_uniform_block_energy_gain_upper_bound == 1
    assert certificate.effective_gain_one_zero_margin_boundary_certified
    assert (
        certificate
        .conditional_exact_model_spatial_disagreement_nonincrease_certified
    )
    assert certificate.repeated_exact_model_spatial_disagreement_stability_certified
    assert not certificate.uniform_positive_normalized_block_margin_certified
    assert not certificate.geometric_spatial_disagreement_convergence_certified
    assert certificate.exact_cycle_energy_gain_upper_bound(10_000) == 1


def test_effective_gain_above_one_is_rejected() -> None:
    with pytest.raises(TNFRValueError, match="effective|q_eff|nonexpansive"):
        _certificate(q=Fraction(2, 3), eta=Fraction(3, 4))


@pytest.mark.parametrize(
    ("alpha", "tau_local", "tau_global", "expected_horizon"),
    (
        (Fraction(0), 7, 9, 1),
        (Fraction(1), 4, 2, 3),
        (Fraction(1, 2), 2, 2, 3),
        (Fraction(1, 2), 2, 4, 5),
    ),
)
def test_remesh_endpoints_and_delay_configurations_are_inherited(
    alpha: Fraction,
    tau_local: int,
    tau_global: int,
    expected_horizon: int,
) -> None:
    certificate = _certificate(
        alpha=alpha,
        tau_local=tau_local,
        tau_global=tau_global,
        q=Fraction(1, 3),
        eta=Fraction(1),
    )

    assert certificate.remesh_certificate.alpha == alpha
    assert certificate.universal_block_horizon == expected_horizon
    assert certificate.history_length == expected_horizon
    assert certificate.exact_effective_head_energy_gain_upper_bound == Fraction(2, 3)


def test_zero_jensen_input_envelope_has_no_relative_defect_budget() -> None:
    eta = Fraction(10**12)
    jensen_input_envelope = Fraction(0)
    ideal_energy = Fraction(0)
    bounded_energy = Fraction(0)
    defect = bounded_energy - ideal_energy

    assert defect <= eta * jensen_input_envelope
    assert eta * jensen_input_envelope - defect == 0
    assert not (
        Fraction(1) - ideal_energy <= eta * jensen_input_envelope
    )


def test_jensen_envelope_avoids_an_ideal_energy_cancellation_singularity() -> None:
    # Opposite active centered fields can make the ideal mixed head vanish while
    # the Jensen input envelope remains positive.  A finite relative defect
    # budget must therefore be normalized by J, not by E(ideal).
    active_energies = (Fraction(1, 2), Fraction(1, 2))
    coefficients = (Fraction(1, 2), Fraction(1, 2))
    jensen_input_envelope = sum(
        (coefficient * energy for coefficient, energy in zip(
            coefficients,
            active_energies,
            strict=True,
        )),
        Fraction(0),
    )
    ideal_energy = Fraction(0)
    bounded_energy = Fraction(1, 8)
    eta = Fraction(1, 4)

    assert jensen_input_envelope == Fraction(1, 2)
    assert bounded_energy - ideal_energy == eta * jensen_input_envelope
    assert bounded_energy - ideal_energy > eta * ideal_energy


def test_absolute_defect_bound_cannot_supply_a_scale_free_effective_gain() -> None:
    # With normalized H=(1/2, 1/2), E_H((a, -a))=a**2/2.  One fixed
    # absolute defect budget therefore dominates arbitrarily small inputs.
    absolute_budget = Fraction(1, 2)
    schedule_gain = Fraction(1, 4)
    bounded_energy = Fraction(1, 2)
    scheduled_energy = Fraction(1, 8)

    assert scheduled_energy == schedule_gain * bounded_energy
    for denominator in (4, 8, 16):
        ideal_energy = Fraction(1, 2 * denominator**2)
        assert bounded_energy - ideal_energy <= absolute_budget
        assert scheduled_energy / ideal_energy == Fraction(denominator**2, 4)
        assert scheduled_energy > ideal_energy

    # At exact consensus an absolute budget even permits positive disagreement,
    # whereas delta<=eta*J with J=0 forces delta<=0.
    assert bounded_energy - Fraction(0) <= absolute_budget
    assert scheduled_energy > 0


@pytest.mark.parametrize(
    "eta",
    (
        True,
        -1,
        Fraction(-1, 100),
        float("nan"),
        float("inf"),
        "0.5",
        None,
    ),
)
def test_invalid_relative_defect_bound_is_rejected(eta: object) -> None:
    with pytest.raises(TNFRValueError, match="relative|defect|finite"):
        certify_uniform_remesh_schedule_relative_defect_stability(
            _policy(),
            eta,  # type: ignore[arg-type]
        )


def test_binary64_eta_is_rationalized_before_effective_gain_validation() -> None:
    policy = _policy(q=Fraction(1, 2))
    certificate = certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        0.1,
    )

    exact_eta = Fraction.from_float(0.1)
    assert certificate.pre_schedule_relative_energy_defect_upper_bound == exact_eta
    assert certificate.exact_effective_head_energy_gain_upper_bound == (
        Fraction(1, 2) * (Fraction(1) + exact_eta)
    )


def test_binary64_decimal_boundary_is_not_silently_rounded_down() -> None:
    policy = _policy(q=Fraction.from_float(0.8))
    with pytest.raises(TNFRValueError, match="effective|q_eff|nonexpansive"):
        certify_uniform_remesh_schedule_relative_defect_stability(policy, 0.25)


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        (
            "pre_schedule_relative_energy_defect_upper_bound",
            Fraction(1, 3),
        ),
        ("exact_effective_head_energy_gain_upper_bound", Fraction(1, 3)),
        ("conditions", (("forged", True),)),
    ),
)
def test_direct_field_tampering_fails_closed(
    field_name: str,
    forged_value: object,
) -> None:
    certificate = _certificate()
    object.__setattr__(certificate, field_name, forged_value)

    assert not certificate.relative_defect_stability_certificate_certified
    assert certificate.failed_conditions == (
        "remesh_schedule_relative_defect_proof_fields_intact",
    )


def test_nested_policy_tampering_invalidates_relative_defect_certificate() -> None:
    policy = _policy()
    certificate = certify_uniform_remesh_schedule_relative_defect_stability(
        policy,
        Fraction(1, 2),
    )
    object.__setattr__(policy, "history_length", 99)

    assert not policy.policy_stability_certificate_certified
    assert not certificate.relative_defect_stability_certificate_certified


def test_nested_effective_certificate_tampering_invalidates_outer_certificate() -> None:
    certificate = _certificate()
    object.__setattr__(
        certificate.effective_head_gain_envelope_certificate,
        "universal_block_horizon",
        99,
    )

    assert not (
        certificate.effective_head_gain_envelope_certificate
        .policy_stability_certificate_certified
    )
    assert not certificate.relative_defect_stability_certificate_certified


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    (
        (
            "pre_schedule_relative_energy_defect_upper_bound",
            Fraction(1, 3),
        ),
        ("exact_effective_head_energy_gain_upper_bound", Fraction(1, 3)),
        ("conditions", (("forged", True),)),
    ),
)
def test_private_reseal_cannot_promote_inconsistent_derivatives(
    field_name: str,
    forged_value: object,
) -> None:
    certificate = _certificate()
    forged = replace(
        certificate,
        **{field_name: forged_value},
        _proof_stamp=(),
    )
    resealed = robust_module._seal(forged)

    assert not resealed.relative_defect_stability_certificate_certified


def test_tampered_hostile_eta_field_is_rejected_without_dispatch() -> None:
    marker: list[str] = []

    class HostileEta:
        def __ge__(self, other: object) -> bool:
            del other
            marker.append("comparison dispatched")
            raise SystemExit("must not escape")

        def __eq__(self, other: object) -> bool:
            del other
            marker.append("equality dispatched")
            raise SystemExit("must not escape")

    certificate = _certificate()
    object.__setattr__(
        certificate,
        "pre_schedule_relative_energy_defect_upper_bound",
        HostileEta(),
    )

    assert not certificate.relative_defect_stability_certificate_certified
    assert marker == []


def test_hostile_public_eta_is_rejected_without_conversion_dispatch() -> None:
    marker: list[str] = []

    class HostileEta:
        def __float__(self) -> float:
            marker.append("conversion dispatched")
            raise SystemExit("must not escape")

        def __eq__(self, other: object) -> bool:
            del other
            marker.append("equality dispatched")
            raise SystemExit("must not escape")

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        certify_uniform_remesh_schedule_relative_defect_stability(
            _policy(),
            HostileEta(),  # type: ignore[arg-type]
        )
    assert marker == []


def test_wrong_or_tampered_policy_is_rejected() -> None:
    with pytest.raises(TNFRValueError, match="policy"):
        certify_uniform_remesh_schedule_relative_defect_stability(
            object(),  # type: ignore[arg-type]
            Fraction(0),
        )

    policy = _policy()
    object.__setattr__(policy, "history_length", 99)
    with pytest.raises(TNFRValueError, match="unsealed|tampered|inconsistent"):
        certify_uniform_remesh_schedule_relative_defect_stability(
            policy,
            Fraction(0),
        )


def test_scope_keeps_conditional_and_runtime_boundaries_explicit() -> None:
    certificate = _certificate()

    assert "Jensen" in robust_module.__doc__
    assert "conditional" in certificate.scope.lower()
    assert "binary64 runtime" in certificate.scope
    assert not certificate.runtime_relative_defect_bound_verified
    assert not certificate.runtime_schedule_maps_verified
    assert not certificate.runtime_forward_invariance_certified
    assert not certificate.future_runtime_stability_certified
    assert not certificate.binary64_runtime_stability_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.solver_order_certified
    assert not certificate.adaptive_grammar_certified
    assert not certificate.full_tnfr_stability_certified
