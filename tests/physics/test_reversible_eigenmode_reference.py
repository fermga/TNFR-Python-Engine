"""Exact reversible single-eigenmode Euler reference contracts."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
from pathlib import Path

import mpmath as mp
import pytest

import tnfr.physics.reversible_eigenmode_reference as reference_module
from tnfr.errors import TNFRValueError
from tnfr.physics.reversible_eigenmode_reference import (
    ReversibleSingleEigenmodeEulerReferenceCertificate,
    certify_reversible_single_eigenmode_euler_reference,
)


F = Fraction

P2_CONDUCTANCE = (
    (F(0), F(1)),
    (F(1), F(0)),
)
P3_CONDUCTANCE = (
    (F(0), F(1), F(0)),
    (F(1), F(0), F(1)),
    (F(0), F(1), F(0)),
)
P2_PARTITIONS = (
    (F(1, 4), F(1, 4)),
    (F(1, 8),) * 4,
    (F(1, 16),) * 8,
)
P3_MODE_ONE_PARTITIONS = (
    (F(1, 4), F(1, 4)),
    (F(1, 8),) * 4,
    (F(1, 16),) * 8,
)


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / value.denominator


def _p2_certificate() -> ReversibleSingleEigenmodeEulerReferenceCertificate:
    return certify_reversible_single_eigenmode_euler_reference(
        P2_CONDUCTANCE,
        nu_f=(F(1), F(1)),
        initial_epi=(F(1), F(-1)),
        partitions=P2_PARTITIONS,
    )


def test_p2_reference_recovers_the_exact_modal_problem() -> None:
    certificate = _p2_certificate()

    assert type(certificate) is ReversibleSingleEigenmodeEulerReferenceCertificate
    assert certificate.reference_certificate_certified
    assert certificate.failed_conditions == ()
    assert certificate.exact_degrees == (F(1), F(1))
    assert certificate.exact_reversible_metric == (F(1), F(1))
    assert certificate.exact_normalized_reversible_metric == (F(1, 2), F(1, 2))
    assert certificate.exact_generator == (
        (F(1), F(-1)),
        (F(-1), F(1)),
    )
    assert certificate.exact_weighted_mean == 0
    assert certificate.exact_centered_mode == (F(1), F(-1))
    assert certificate.exact_mode_eigenvalue == 2
    assert certificate.exact_mode_residual == (F(0), F(0))
    assert certificate.exact_mode_linf_norm == 1
    assert certificate.exact_mode_h_energy == 1
    assert certificate.exact_total_duration == F(1, 2)


def test_p2_euler_products_and_exact_error_bounds() -> None:
    certificate = _p2_certificate()

    assert certificate.exact_euler_factors == (
        F(1, 4),
        F(81, 256),
        F(5_764_801, 16_777_216),
    )
    assert certificate.exact_euler_endpoints[0] == (F(1, 4), F(-1, 4))
    assert certificate.exact_quadratic_factor_error_upper_bounds == (
        F(1, 4),
        F(1, 8),
        F(1, 16),
    )
    assert certificate.exact_hmax_factor_error_upper_bounds == (
        F(1, 4),
        F(1, 8),
        F(1, 16),
    )
    for factor, lower, upper, quadratic, hmax in zip(
        certificate.exact_euler_factors,
        certificate.exact_factor_error_lower_bounds,
        certificate.exact_factor_error_upper_bounds,
        certificate.exact_quadratic_factor_error_upper_bounds,
        certificate.exact_hmax_factor_error_upper_bounds,
        strict=True,
    ):
        assert 0 <= lower <= upper <= quadratic <= hmax
        with mp.workdps(80):
            true_factor_error = mp.exp(-1) - _mp_fraction(factor)
            assert _mp_fraction(lower) <= true_factor_error
            assert true_factor_error <= _mp_fraction(upper)
    assert certificate.exact_real_euler_error_bound_certified
    assert certificate.exact_linf_error_bound_certified
    assert certificate.exact_h_energy_error_bound_certified
    assert certificate.exact_linf_error_lower_bounds == (
        certificate.exact_factor_error_lower_bounds
    )
    assert certificate.exact_linf_error_upper_bounds == (
        certificate.exact_factor_error_upper_bounds
    )
    assert certificate.exact_h_energy_error_lower_bounds == tuple(
        value * value for value in certificate.exact_factor_error_lower_bounds
    )
    assert certificate.exact_h_energy_error_upper_bounds == tuple(
        value * value for value in certificate.exact_factor_error_upper_bounds
    )


def test_proper_subdivision_strictly_improves_factor_and_bounds() -> None:
    certificate = _p2_certificate()

    assert certificate.exact_euler_factor_improvements == (
        F(17, 256),
        F(456_385, 16_777_216),
    )
    assert certificate.exact_quadratic_bound_improvements == (
        F(1, 8),
        F(1, 16),
    )
    assert all(value > 0 for value in certificate.exact_euler_factor_improvements)
    assert all(
        value > 0
        for value in certificate.exact_linf_quadratic_bound_improvements
    )
    assert all(
        value > 0
        for value in certificate.exact_h_energy_quadratic_bound_improvements
    )
    assert certificate.strict_proper_subdivision_improvement_certified
    assert certificate.conditional_exact_real_partition_convergence_certified


@pytest.mark.parametrize(
    ("initial_epi", "expected_eigenvalue", "expected_energy"),
    (
        ((F(1), F(0), F(-1)), F(1), F(1)),
        ((F(1), F(-1), F(1)), F(2), F(2)),
    ),
)
def test_p3_nonregular_path_certifies_both_exact_modes(
    initial_epi,
    expected_eigenvalue,
    expected_energy,
) -> None:
    certificate = certify_reversible_single_eigenmode_euler_reference(
        P3_CONDUCTANCE,
        nu_f=(F(1), F(1), F(1)),
        initial_epi=initial_epi,
        partitions=P3_MODE_ONE_PARTITIONS,
    )

    assert certificate.exact_degrees == (F(1), F(2), F(1))
    assert certificate.exact_reversible_metric == (F(1), F(2), F(1))
    assert certificate.exact_normalized_reversible_metric == (
        F(1, 4),
        F(1, 2),
        F(1, 4),
    )
    assert certificate.exact_weighted_mean == 0
    assert certificate.exact_centered_mode == initial_epi
    assert certificate.exact_mode_eigenvalue == expected_eigenvalue
    assert certificate.exact_mode_h_energy == expected_energy
    assert certificate.reference_certificate_certified


def test_weighted_mean_is_removed_before_the_mode_test() -> None:
    certificate = certify_reversible_single_eigenmode_euler_reference(
        P3_CONDUCTANCE,
        nu_f=(F(1), F(1), F(1)),
        initial_epi=(F(4), F(3), F(2)),
        partitions=P3_MODE_ONE_PARTITIONS,
    )

    assert certificate.exact_weighted_mean == 3
    assert certificate.exact_centered_mode == (F(1), F(0), F(-1))
    assert certificate.exact_mode_eigenvalue == 1
    exact_factor = math.exp(-0.5)
    for index, value in enumerate((4.0, 3.0, 2.0)):
        exact_endpoint = 3.0 + exact_factor * (value - 3.0)
        assert float(certificate.exact_continuous_endpoint_lower_bound[index]) <= (
            exact_endpoint
        )
        assert exact_endpoint <= float(
            certificate.exact_continuous_endpoint_upper_bound[index]
        )
    assert certificate.exact_modal_solution_enclosure_certified


def test_heterogeneous_capacity_changes_h_but_preserves_the_exact_mode() -> None:
    certificate = certify_reversible_single_eigenmode_euler_reference(
        P3_CONDUCTANCE,
        nu_f=(F(1), F(1, 2), F(1)),
        initial_epi=(F(1), F(0), F(-1)),
        partitions=P3_MODE_ONE_PARTITIONS,
    )

    assert certificate.exact_reversible_metric == (F(1), F(4), F(1))
    assert certificate.exact_normalized_reversible_metric == (
        F(1, 6),
        F(2, 3),
        F(1, 6),
    )
    assert certificate.exact_mode_eigenvalue == 1
    assert certificate.exact_mode_residual == (F(0), F(0), F(0))
    assert certificate.reference_certificate_certified


def test_mixed_p3_modes_are_rejected_instead_of_projected() -> None:
    with pytest.raises(TNFRValueError, match="not one exact positive eigenmode"):
        certify_reversible_single_eigenmode_euler_reference(
            P3_CONDUCTANCE,
            nu_f=(F(1), F(1), F(1)),
            initial_epi=(F(2), F(-1), F(0)),
            partitions=P3_MODE_ONE_PARTITIONS,
        )


def test_partition_contract_rejects_nonrefinement_and_unstable_steps() -> None:
    with pytest.raises(TNFRValueError, match="proper-subdivision"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=(
                (F(1, 4), F(1, 4)),
                (F(1, 5), F(3, 10)),
            ),
        )

    with pytest.raises(TNFRValueError, match=r"mu\*h must lie in \(0, 1\)"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=((F(1, 2),),),
        )


@pytest.mark.parametrize(
    ("conductance", "message"),
    (
        (
            ((F(1), F(1)), (F(1), F(0))),
            "diagonal",
        ),
        (
            ((F(0), F(1)), (F(0), F(0))),
            "symmetric",
        ),
        (
            (
                (F(0), F(1), F(0), F(0)),
                (F(1), F(0), F(0), F(0)),
                (F(0), F(0), F(0), F(1)),
                (F(0), F(0), F(1), F(0)),
            ),
            "connected",
        ),
    ),
)
def test_nonreversible_or_disconnected_conductance_is_rejected(
    conductance,
    message,
) -> None:
    size = len(conductance)
    with pytest.raises(TNFRValueError, match=message):
        certify_reversible_single_eigenmode_euler_reference(
            conductance,
            nu_f=(F(1),) * size,
            initial_epi=tuple(F(index) for index in range(size)),
            partitions=((F(1, 8),),),
        )


def test_negative_weight_zero_capacity_and_invalid_states_are_rejected() -> None:
    negative = ((F(0), F(-1)), (F(-1), F(0)))
    with pytest.raises(TNFRValueError, match="nonnegative"):
        certify_reversible_single_eigenmode_euler_reference(
            negative,
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=((F(1, 4),),),
        )

    with pytest.raises(TNFRValueError, match="capacity must be positive"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(0)),
            initial_epi=(F(1), F(-1)),
            partitions=((F(1, 4),),),
        )

    with pytest.raises(TNFRValueError, match="nonuniform mode"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(1)),
            initial_epi=(F(3), F(3)),
            partitions=((F(1, 4),),),
        )

    with pytest.raises(TNFRValueError, match="one exact total duration"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=((F(1, 4),), (F(1, 8), F(1, 4))),
        )


def test_one_partition_keeps_theorem_without_observed_improvement() -> None:
    certificate = certify_reversible_single_eigenmode_euler_reference(
        P3_CONDUCTANCE,
        nu_f=(F(1), F(1), F(1)),
        initial_epi=(F(1), F(0), F(-1)),
        partitions=((F(1, 4), F(1, 4)),),
    )

    assert certificate.reference_certificate_certified
    assert certificate.exact_euler_factor_improvements == ()
    assert not certificate.strict_proper_subdivision_improvement_certified
    assert certificate.conditional_exact_real_partition_convergence_certified


def test_scope_does_not_promote_runtime_or_full_dynamics() -> None:
    certificate = _p2_certificate()

    assert "exact-real convergence" in certificate.scope
    assert "positive partitions with equal total duration" in certificate.scope
    assert "representational cutoff mu*T <= 4096" in certificate.scope
    assert "binary64 execution or asymptotics" in certificate.scope
    assert certificate.conditional_exact_real_partition_convergence_certified
    assert not certificate.binary64_asymptotic_convergence_certified
    assert not certificate.arbitrary_or_mixed_mode_initial_data_certified
    assert not certificate.directed_or_nonreversible_generator_certified
    assert not certificate.changing_generator_or_metric_certified
    assert not certificate.glyph_or_remesh_dynamics_certified
    assert not certificate.solver_order_certified
    assert not certificate.full_tnfr_stability_certified


def test_direct_and_private_reseal_tampering_fail_closed() -> None:
    certificate = _p2_certificate()
    directly_changed = replace(certificate, exact_mode_eigenvalue=F(99))
    assert not directly_changed.reference_certificate_certified

    forged = replace(
        certificate,
        exact_mode_eigenvalue=F(99),
        exact_mode_residual=(F(0), F(0)),
        _proof_stamp=(),
    )
    resealed = reference_module._seal(forged)
    assert not resealed.reference_certificate_certified
    assert resealed.failed_conditions == (
        "reversible_single_eigenmode_proof_fields_intact",
    )


def test_hostile_privately_resealed_derivative_does_not_dispatch_equality() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    certificate = _p2_certificate()
    forged = replace(
        certificate,
        exact_mode_eigenvalue=AlwaysEqual(),
        _proof_stamp=(),
    )
    resealed = reference_module._seal(forged)

    assert not resealed.reference_certificate_certified
    assert equality_calls == []


def test_type_stub_exposes_the_exact_public_contract() -> None:
    stub = Path(reference_module.__file__).with_suffix(".pyi").read_text(
        encoding="utf-8"
    )

    assert "class ReversibleSingleEigenmodeEulerReferenceCertificate" in stub
    assert "def certify_reversible_single_eigenmode_euler_reference" in stub
    assert "conditional_exact_real_partition_convergence_certified" in stub


def test_exact_input_and_exponential_size_bound_fail_closed() -> None:
    with pytest.raises(TNFRValueError, match="exact Fraction"):
        certify_reversible_single_eigenmode_euler_reference(
            ((0, 1), (1, 0)),
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=P2_PARTITIONS,
        )

    with pytest.raises(TNFRValueError, match="exponent <= 4096"):
        reference_module._negative_exp_bounds(F(4097))


def test_oversized_exponent_rejects_before_exact_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_product(*args, **kwargs):
        del args, kwargs
        raise AssertionError("exact partition products must not be materialized")

    monkeypatch.setattr(reference_module.math, "prod", unexpected_product)

    with pytest.raises(TNFRValueError, match="exponent <= 4096"):
        certify_reversible_single_eigenmode_euler_reference(
            P2_CONDUCTANCE,
            nu_f=(F(1), F(1)),
            initial_epi=(F(1), F(-1)),
            partitions=((F(1, 4),) * 8193,),
        )


@pytest.mark.parametrize(
    "surface",
    (
        "conductance",
        "conductance_row",
        "nu_f",
        "initial_epi",
        "partitions",
        "partition_row",
    ),
)
def test_unordered_input_surfaces_are_rejected(surface: str) -> None:
    conductance = P2_CONDUCTANCE
    nu_f = (F(1), F(1))
    initial_epi = (F(1), F(-1))
    partitions = P2_PARTITIONS

    if surface == "conductance":
        conductance = frozenset(P2_CONDUCTANCE)  # type: ignore[assignment]
    elif surface == "conductance_row":
        conductance = (  # type: ignore[assignment]
            frozenset((F(0), F(1))),
            (F(1), F(0)),
        )
    elif surface == "nu_f":
        nu_f = {F(1), F(2)}  # type: ignore[assignment]
    elif surface == "initial_epi":
        initial_epi = {F(1), F(-1)}  # type: ignore[assignment]
    elif surface == "partitions":
        partitions = set(P2_PARTITIONS)  # type: ignore[assignment]
    else:
        partitions = (  # type: ignore[assignment]
            frozenset((F(1, 4), F(1, 8))),
        )

    with pytest.raises(TypeError, match="iterable sequence"):
        certify_reversible_single_eigenmode_euler_reference(
            conductance,
            nu_f=nu_f,
            initial_epi=initial_epi,
            partitions=partitions,
        )
