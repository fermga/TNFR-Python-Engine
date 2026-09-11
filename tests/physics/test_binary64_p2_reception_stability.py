"""Global P2 half-Reception binary64 kernel stability tests."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from fractions import Fraction

import pytest

import tnfr.physics.binary64_p2_reception_stability as p2_module
from tnfr.constants.canonical import EN_MIX_FACTOR
from tnfr.errors import TNFRValueError
from tnfr.operators._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
)
from tnfr.physics.binary64_p2_reception_stability import (
    P2HalfReceptionRemeshStabilityCertificate,
    certify_p2_half_reception_remesh_stability,
)
from tnfr.physics.binary64_remesh_relative_defect import (
    UniformAlphaOneHardClipRemeshClassCertificate,
    certify_alpha_one_hard_clip_remesh_class,
)


def _source(
    *,
    nodes: tuple[str, ...] = ("left", "right"),
    metric_weights: object = (1.0, 3.0),
    tau_local: int = 5,
    tau_global: int = 2,
    epi_min: float = -1.0,
    epi_max: float = 1.0,
) -> UniformAlphaOneHardClipRemeshClassCertificate:
    return certify_alpha_one_hard_clip_remesh_class(
        nodes,
        metric_weights,
        tau_local=tau_local,
        tau_global=tau_global,
        epi_min=epi_min,
        epi_max=epi_max,
    )


def _certificate(**kwargs: object) -> P2HalfReceptionRemeshStabilityCertificate:
    return certify_p2_half_reception_remesh_stability(_source(**kwargs))


def test_direct_module_exposes_only_the_narrow_public_api() -> None:
    assert p2_module.__all__ == (
        "P2HalfReceptionRemeshStabilityCertificate",
        "certify_p2_half_reception_remesh_stability",
    )


def test_certificate_composes_the_global_q_zero_and_eta_zero_results() -> None:
    certificate = _certificate()

    assert type(certificate) is P2HalfReceptionRemeshStabilityCertificate
    assert certificate.node_order == ("left", "right")
    assert certificate.exact_normalized_metric == (
        Fraction(1, 4),
        Fraction(3, 4),
    )
    assert certificate.mutual_singleton_neighbor_indices == ((1,), (0,))
    assert certificate.operator_name == "Reception"
    assert certificate.operator_glyph == "EN"
    assert certificate.stage_schedule == "two_phase_jacobi"
    assert certificate.binary64_mix_factor.hex() == "0x1.0000000000000p-1"
    assert certificate.exact_mix_factor == Fraction(1, 2)
    assert certificate.exact_ideal_consensus_projector == (
        (Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), Fraction(1, 2)),
    )
    assert certificate.exact_schedule_energy_gain_upper_bound == 0
    assert certificate.exact_pre_schedule_relative_energy_defect_upper_bound == 0
    assert certificate.exact_effective_head_energy_gain_upper_bound == 0
    assert certificate.policy_certificate.policy_stability_certificate_certified
    assert (
        certificate.relative_defect_certificate
        .relative_defect_stability_certificate_certified
    )
    assert (
        certificate.relative_defect_certificate
        .q_zero_preschedule_defect_absorption_certified
    )
    assert certificate.p2_half_reception_remesh_stability_certificate_certified
    assert certificate.global_binary64_epi_kernel_family_certified
    assert certificate.numeric_consensus_projection_certified
    assert certificate.restricted_epi_kernel_interval_forward_invariant_certified
    assert (
        certificate
        .restricted_kernel_support_metric_configuration_preservation_certified
    )
    assert certificate.arbitrary_finite_binary64_kernel_repetition_certified
    assert certificate.active_history_exact_extinction_certified
    assert certificate.failed_conditions == ()


def test_q_zero_extinguishes_the_active_history_at_tau_global_plus_one() -> None:
    certificate = _certificate(tau_local=9, tau_global=2)

    assert certificate.remesh_class_certificate.required_history_length == 10
    assert (
        certificate.remesh_class_certificate.remesh_certificate.active_max_delay
        == 2
    )
    assert certificate.active_history_extinction_horizon == 3
    assert certificate.exact_cycle_energy_gain_upper_bound(0) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(1) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(2) == 1
    assert certificate.exact_cycle_energy_gain_upper_bound(3) == 0
    assert certificate.exact_cycle_energy_gain_upper_bound(4) == 0
    assert certificate.exact_cycle_energy_gain_upper_bound(100) == 0


def test_global_binary64_pair_kernel_covers_extremes_subnormals_and_zeros() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    minimum_normal = float.fromhex("0x1.0000000000000p-1022")
    minimum_subnormal = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certificate(epi_min=-maximum, epi_max=maximum)
    alphabet = (
        -maximum,
        -1.0,
        -minimum_normal,
        -minimum_subnormal,
        -0.0,
        0.0,
        minimum_subnormal,
        minimum_normal,
        1.0,
        maximum,
    )

    pairs = tuple(zip(alphabet, reversed(alphabet), strict=True)) + (
        (maximum, maximum),
        (-maximum, -maximum),
        (minimum_subnormal, minimum_subnormal),
        (-0.0, 0.0),
    )
    for pair in pairs:
        output = certificate.evaluate_binary64_schedule_pair(pair)
        assert output[0] == output[1]
        assert math.isfinite(output[0])
        assert -maximum <= output[0] <= maximum


def test_underflow_refutes_global_affinity_but_not_numeric_consensus() -> None:
    tiny = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certificate(epi_min=-tiny, epi_max=tiny)

    output = certificate.evaluate_binary64_schedule_pair((tiny, tiny))

    assert output == (0.0, 0.0)
    assert Fraction.from_float(tiny) > 0
    assert not certificate.global_binary64_runtime_affinity_certified
    assert certificate.numeric_consensus_projection_certified


def test_hard_clamp_restores_a_degenerate_subnormal_interval() -> None:
    tiny = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certificate(epi_min=tiny, epi_max=tiny)

    output = certificate.evaluate_binary64_schedule_pair((tiny, tiny))

    assert output == (tiny, tiny)
    assert certificate.restricted_epi_kernel_interval_forward_invariant_certified


@pytest.mark.parametrize(
    ("lower", "upper", "pair"),
    (
        (
            1.0,
            float.fromhex("0x1.fffffffffffffp+1023"),
            (1.0, float.fromhex("0x1.fffffffffffffp+1023")),
        ),
        (
            -float.fromhex("0x1.fffffffffffffp+1023"),
            -1.0,
            (-float.fromhex("0x1.fffffffffffffp+1023"), -1.0),
        ),
    ),
)
def test_noncentered_extreme_intervals_remain_forward_invariant(
    lower: float,
    upper: float,
    pair: tuple[float, float],
) -> None:
    certificate = _certificate(epi_min=lower, epi_max=upper)

    output = certificate.evaluate_binary64_schedule_pair(pair)

    assert output[0] == output[1]
    assert lower <= output[0] <= upper


def test_signed_zero_adversary_is_equal_numerically_but_not_bitwise() -> None:
    tiny = float.fromhex("0x0.0000000000001p-1022")
    certificate = _certificate()

    output = certificate.evaluate_binary64_schedule_pair((-tiny, -0.0))

    assert output[0] == output[1] == 0.0
    assert output[0].hex() == "0x0.0p+0"
    assert output[1].hex() == "-0x0.0p+0"
    assert not certificate.signed_zero_bit_preservation_certified


def test_scope_does_not_promote_the_kernel_to_stage_grammar_or_solver() -> None:
    certificate = _certificate()

    assert not certificate.complete_reception_stage_certified
    assert not certificate.grammar_execution_certified
    assert not certificate.live_graph_execution_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.full_tnfr_stability_certified
    assert "restricted binary64 EPI kernels" in certificate.scope
    assert "not validated against a graph" in certificate.scope
    assert "does not certify the complete EN stage" in certificate.scope


def test_canonical_default_en_mix_has_an_exact_energy_gain_four_witness() -> None:
    left = float.fromhex("0x1.d2775ff0a1eccp-1021")
    right = float.fromhex("0x1.d2775ff0a1ecdp-1021")

    left_output = neighbor_epi_blend_value(
        left,
        neighbor_epi_unweighted_mean((right,)),
        EN_MIX_FACTOR,
    )
    right_output = neighbor_epi_blend_value(
        right,
        neighbor_epi_unweighted_mean((left,)),
        EN_MIX_FACTOR,
    )
    input_separation = Fraction.from_float(right) - Fraction.from_float(left)
    output_separation = (
        Fraction.from_float(right_output) - Fraction.from_float(left_output)
    )

    assert EN_MIX_FACTOR.hex() == "0x1.ee7eea04ddca0p-3"
    assert left_output.hex() == "0x1.d2775ff0a1eccp-1021"
    assert right_output.hex() == "0x1.d2775ff0a1ecep-1021"
    assert 0.0 < left < right < 1.0
    assert 0.0 < left_output < right_output < 1.0
    assert output_separation == 2 * input_separation
    assert output_separation**2 == 4 * input_separation**2
    assert EN_MIX_FACTOR != 0.5


@pytest.mark.parametrize(
    "pair",
    (
        (),
        (0.0,),
        (0.0, 0.0, 0.0),
        (0, 0.0),
        (False, 0.0),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (-math.inf, 0.0),
        (-1.0000000000000002, 0.0),
        (0.0, 1.0000000000000002),
        "00",
        None,
    ),
)
def test_pair_evaluator_rejects_values_outside_the_strict_class(
    pair: object,
) -> None:
    certificate = _certificate()

    with pytest.raises(TNFRValueError):
        certificate.evaluate_binary64_schedule_pair(pair)  # type: ignore[arg-type]


def test_factory_requires_exactly_two_nodes() -> None:
    for nodes in (("only",), ("a", "b", "c")):
        source = _source(nodes=nodes, metric_weights=None)
        with pytest.raises(TNFRValueError, match="exactly two ordered nodes"):
            certify_p2_half_reception_remesh_stability(source)


def test_factory_rejects_wrong_or_tampered_source_types() -> None:
    with pytest.raises(TNFRValueError, match="must be an exact"):
        certify_p2_half_reception_remesh_stability(object())  # type: ignore[arg-type]

    source = _source()
    object.__setattr__(
        source,
        "exact_uniform_relative_defect_upper_bound",
        Fraction(1),
    )
    with pytest.raises(TNFRValueError, match="unsealed, tampered, or inconsistent"):
        certify_p2_half_reception_remesh_stability(source)


def test_certificate_is_frozen_sealed_and_fails_closed_after_tamper() -> None:
    certificate = _certificate()

    with pytest.raises(FrozenInstanceError):
        certificate.exact_mix_factor = Fraction(1, 3)  # type: ignore[misc]

    payload = {
        item.name: object.__getattribute__(certificate, item.name)
        for item in fields(P2HalfReceptionRemeshStabilityCertificate)
        if item.name != "_proof_stamp"
    }
    unsealed = P2HalfReceptionRemeshStabilityCertificate(**payload)
    assert not unsealed.p2_half_reception_remesh_stability_certificate_certified
    assert unsealed.failed_conditions == (
        "p2_half_reception_remesh_proof_fields_intact",
    )

    object.__setattr__(
        certificate,
        "exact_schedule_energy_gain_upper_bound",
        Fraction(1),
    )
    assert not certificate.p2_half_reception_remesh_stability_certificate_certified
    assert not certificate.global_binary64_epi_kernel_family_certified
    assert certificate.failed_conditions == (
        "p2_half_reception_remesh_proof_fields_intact",
    )
    with pytest.raises(TNFRValueError, match="unsealed or inconsistent"):
        certificate.exact_cycle_energy_gain_upper_bound(3)
    with pytest.raises(TNFRValueError, match="unsealed or inconsistent"):
        certificate.evaluate_binary64_schedule_pair((0.0, 0.0))


def test_nested_source_tamper_invalidates_the_outer_certificate() -> None:
    certificate = _certificate()
    object.__setattr__(
        certificate.remesh_class_certificate,
        "exact_uniform_relative_defect_upper_bound",
        Fraction(1),
    )

    assert not certificate.p2_half_reception_remesh_stability_certificate_certified
    assert not certificate.arbitrary_finite_binary64_kernel_repetition_certified


def test_private_reseal_does_not_promote_inconsistent_derived_fields() -> None:
    certificate = _certificate()
    object.__setattr__(
        certificate,
        "exact_schedule_energy_gain_upper_bound",
        Fraction(1),
    )
    resealed = p2_module._seal(certificate)

    assert not resealed.p2_half_reception_remesh_stability_certificate_certified
    assert resealed.failed_conditions == (
        "p2_half_reception_remesh_proof_fields_intact",
    )


def test_hostile_private_reseal_fails_without_numeric_or_equality_dispatch() -> None:
    calls: list[str] = []

    class HostileScalar:
        def __float__(self) -> float:
            calls.append("float")
            raise AssertionError("hostile float dispatch")

        def __eq__(self, other: object) -> bool:
            del other
            calls.append("eq")
            raise AssertionError("hostile equality dispatch")

        def __le__(self, other: object) -> bool:
            del other
            calls.append("le")
            raise AssertionError("hostile ordering dispatch")

        def __bool__(self) -> bool:
            calls.append("bool")
            raise AssertionError("hostile truth dispatch")

    certificate = _certificate()
    object.__setattr__(certificate, "exact_mix_factor", HostileScalar())
    resealed = p2_module._seal(certificate)

    assert not resealed.p2_half_reception_remesh_stability_certificate_certified
    assert resealed.failed_conditions == (
        "p2_half_reception_remesh_proof_fields_intact",
    )
    assert calls == []
