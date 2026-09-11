"""Uniform binary64 REMESH tests for the half-alpha antisymmetric P2 class."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction
from pathlib import Path

import pytest

import tnfr.physics.binary64_remesh_relative_defect as boundary_module
from tnfr.errors import TNFRValueError
from tnfr.physics.binary64_remesh_relative_defect import (
    UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate,
    certify_half_alpha_antisymmetric_hard_clip_remesh_class,
    observe_binary64_remesh_pair_relative_defect,
)


def _certificate(
    *,
    nodes: tuple[str, ...] = ("left", "right"),
    metric_weights: object = None,
    tau_local: int = 2,
    tau_global: int = 3,
    epi_bound: object = 1.0,
) -> UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate:
    return certify_half_alpha_antisymmetric_hard_clip_remesh_class(
        nodes,
        metric_weights,
        tau_local=tau_local,
        tau_global=tau_global,
        epi_bound=epi_bound,
    )


def _runtime_step(
    certificate: UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate,
    history: tuple[tuple[float, float], ...],
):
    current = history[-1]
    local = history[-1 - certificate.tau_local]
    global_ = history[-1 - certificate.tau_global]
    return observe_binary64_remesh_pair_relative_defect(
        current,
        local,
        global_,
        alpha=0.5,
        epi_min=-float(certificate.epi_bound),
        epi_max=float(certificate.epi_bound),
        clip_mode="hard",
    )


def _weighted_center(pair: tuple[float, float]) -> Fraction:
    left, right = (Fraction.from_float(value) for value in pair)
    return (left + 3 * right) / 4


def test_direct_module_and_stub_expose_the_half_alpha_class_api() -> None:
    expected = {
        "UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate",
        "certify_half_alpha_antisymmetric_hard_clip_remesh_class",
    }

    assert expected <= set(boundary_module.__all__)
    package = Path(boundary_module.__file__).parent
    stub = (package / "binary64_remesh_relative_defect.pyi").read_text(
        encoding="utf-8"
    )
    assert (
        "class UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate"
        in stub
    )
    assert (
        "def certify_half_alpha_antisymmetric_hard_clip_remesh_class" in stub
    )
    assert "def certify_schedule_relative_defect_stability" in stub


def test_half_alpha_class_materializes_the_exact_uniform_contract() -> None:
    certificate = _certificate(metric_weights=(1.0, 3.0))

    assert type(certificate) is (
        UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate
    )
    assert certificate.node_order == ("left", "right")
    assert certificate.exact_normalized_metric == (
        Fraction(1, 4),
        Fraction(3, 4),
    )
    assert certificate.tau_local == 2
    assert certificate.tau_global == 3
    assert certificate.history_maxlen == 64
    assert certificate.required_history_length == 4
    assert certificate.epi_bound == 1
    assert certificate.epi_min == -1
    assert certificate.epi_max == 1
    assert certificate.alpha == Fraction(1, 2)
    assert certificate.clip_mode == "hard"
    assert certificate.remesh_certificate.alpha == Fraction(1, 2)
    assert certificate.remesh_certificate.beta == Fraction(1, 4)
    assert certificate.remesh_certificate.gamma == Fraction(1, 4)
    assert certificate.remesh_certificate.delta == Fraction(1, 2)
    assert certificate.exact_uniform_relative_defect_upper_bound == Fraction(
        135, 124
    )
    assert certificate.exact_strict_schedule_gain_threshold == Fraction(
        124, 259
    )
    assert certificate.sharpness_witness.pair_relative_defect_observation_certified
    assert (
        certificate.sharpness_witness
        .exact_minimum_nonnegative_relative_defect_bound
        == Fraction(135, 124)
    )
    assert certificate.example_schedule_energy_gain_upper_bound == Fraction(4, 9)
    assert (
        certificate.exact_example_effective_head_energy_gain_upper_bound
        == Fraction(259, 279)
    )
    assert (
        certificate.exact_example_normalized_block_margin_lower_bound
        == Fraction(20, 279)
    )
    assert (
        certificate.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert certificate.binary64_antisymmetry_preserved_certified
    assert certificate.hard_clip_preserves_antisymmetric_interval_certified
    assert certificate.uniform_binary64_relative_defect_bound_certified
    assert certificate.uniform_relative_defect_bound_is_sharp_certified
    assert certificate.remesh_class_forward_invariant_certified
    assert certificate.strict_schedule_composition_threshold_certified
    assert certificate.example_schedule_composition_certified
    assert certificate.failed_conditions == ()
    assert not certificate.schedule_family_certificate_certified
    assert not certificate.repeated_binary64_stability_certified
    assert not certificate.runtime_forward_invariance_certified
    assert not certificate.binary64_runtime_stability_certified
    assert not certificate.future_binary64_execution_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.full_tnfr_stability_certified


def test_global_bound_uses_an_exact_tail_and_exhaustive_finite_core() -> None:
    certificate = _certificate()
    unit_roundoff = Fraction(1, 2**53)
    linear = Fraction(3, 2) * unit_roundoff + unit_roundoff**2 / 2
    absolute = (
        Fraction(9, 4)
        + Fraction(9, 4) * unit_roundoff
        + unit_roundoff**2 / 2
    )
    ratio_at_eleven = linear + absolute / 11
    tail_bound = 4 * (ratio_at_eleven + ratio_at_eleven**2)

    assert certificate.exact_tail_error_linear_coefficient == linear
    assert certificate.exact_tail_error_absolute_coefficient == absolute
    assert certificate.exact_tail_error_ratio_at_eleven_subnormals == (
        ratio_at_eleven
    )
    assert certificate.exact_tail_relative_defect_upper_bound == tail_bound
    assert tail_bound < Fraction(135, 124)
    assert certificate.finite_core_candidate_count == 6615
    assert certificate.finite_core_admissible_count == 3890
    assert certificate.finite_core_maximizer_amplitudes == (-3, -2, -3)

    enumerated = boundary_module._enumerate_half_class_subnormal_core()
    assert enumerated == (
        Fraction(135, 124),
        (-3, -2, -3),
        6615,
        3890,
    )


def test_exact_gain_threshold_distinguishes_strict_boundary_and_failure() -> None:
    certificate = _certificate()
    strict = certificate.certify_schedule_relative_defect_stability(
        Fraction(4, 9)
    )
    boundary = certificate.certify_schedule_relative_defect_stability(
        Fraction(124, 259)
    )

    assert strict.pre_schedule_relative_energy_defect_upper_bound == Fraction(
        135, 124
    )
    assert strict.schedule_energy_gain_upper_bound == Fraction(4, 9)
    assert strict.exact_effective_head_energy_gain_upper_bound == Fraction(259, 279)
    assert strict.exact_uniform_normalized_block_margin_lower_bound == Fraction(
        20, 279
    )
    assert strict.uniform_positive_normalized_block_margin_certified
    assert strict.geometric_spatial_disagreement_convergence_certified

    assert boundary.exact_effective_head_energy_gain_upper_bound == 1
    assert boundary.exact_uniform_normalized_block_margin_lower_bound == 0
    assert boundary.effective_gain_one_zero_margin_boundary_certified
    assert not boundary.uniform_positive_normalized_block_margin_certified
    assert not boundary.geometric_spatial_disagreement_convergence_certified

    with pytest.raises(TNFRValueError, match="effective|q_eff|gain"):
        certificate.certify_schedule_relative_defect_stability(Fraction(9, 16))
    assert Fraction(9, 16) * (1 + Fraction(135, 124)) == Fraction(
        2331, 1984
    )


def test_subnormal_witness_attains_the_uniform_bound_sharply() -> None:
    smallest = math.ulp(0.0)
    bound = 4 * smallest
    certificate = _certificate(epi_bound=bound)
    observation = observe_binary64_remesh_pair_relative_defect(
        (-3 * smallest, 3 * smallest),
        (-2 * smallest, 2 * smallest),
        (-3 * smallest, 3 * smallest),
        alpha=0.5,
        epi_min=-bound,
        epi_max=bound,
        clip_mode="hard",
    )
    unit = Fraction.from_float(smallest)

    assert observation.exact_ideal_pair == (
        -Fraction(11, 4) * unit,
        Fraction(11, 4) * unit,
    )
    assert observation.runtime_raw_pair == (
        -4 * smallest,
        4 * smallest,
    )
    assert observation.runtime_bounded_pair == observation.runtime_raw_pair
    centered_jensen_energy = (
        observation.exact_input_pairwise_jensen_denominator / 4
    )
    centered_energy_defect = (
        observation.exact_total_signed_squared_separation_defect / 4
    )
    assert centered_jensen_energy == (
        Fraction(31, 4) * unit * unit
    )
    assert centered_energy_defect == (
        Fraction(135, 16) * unit * unit
    )
    assert observation.exact_minimum_nonnegative_relative_defect_bound == (
        Fraction(135, 124)
    )
    assert (
        observation.exact_minimum_nonnegative_relative_defect_bound
        == certificate.exact_uniform_relative_defect_upper_bound
    )


@pytest.mark.parametrize(
    ("tau_local", "tau_global", "history"),
    (
        (
            2,
            3,
            (
                (0.75, -0.75),
                (-0.25, 0.25),
                (1.0, -1.0),
                (-1.0, 1.0),
            ),
        ),
        (
            3,
            3,
            (
                (-0.0, 0.0),
                (0.0, -0.0),
                (-math.ulp(0.0), math.ulp(0.0)),
                (math.ulp(0.0), -math.ulp(0.0)),
            ),
        ),
        (
            5,
            2,
            (
                (-0.5, 0.5),
                (0.25, -0.25),
                (-0.0, 0.0),
                (math.ulp(0.0), -math.ulp(0.0)),
                (1.0, -1.0),
                (-0.75, 0.75),
            ),
        ),
    ),
)
def test_ordinary_subnormal_signed_zero_and_delay_histories_are_invariant(
    tau_local: int,
    tau_global: int,
    history: tuple[tuple[float, float], ...],
) -> None:
    certificate = _certificate(
        tau_local=tau_local,
        tau_global=tau_global,
    )

    assert certificate.required_history_length == max(tau_local, tau_global) + 1
    assert certificate.represented_history_belongs_to_class(history)
    observation = _runtime_step(certificate, history)
    left, right = observation.runtime_bounded_pair
    assert right == -left
    assert -1.0 <= left <= 1.0
    shifted = history[1:] + (observation.runtime_bounded_pair,)
    assert certificate.represented_history_belongs_to_class(shifted)


@pytest.mark.parametrize(
    "history",
    (
        (),
        ((0.0, -0.0),) * 3,
        ((0.0, -0.0),) * 65,
        ((0.5, -0.25),) * 4,
        ((1.0000000000000002, -1.0000000000000002),) * 4,
        ((0.0,),) * 4,
        ((0.0, -0.0, 0.0),) * 4,
        ((0, 0.0),) * 4,
        ((False, 0.0),) * 4,
        ((math.nan, math.nan),) * 4,
        ((math.inf, -math.inf),) * 4,
        ("not-a-row",) * 4,
    ),
)
def test_history_membership_rejects_values_outside_the_exact_class(
    history: object,
) -> None:
    assert not _certificate().represented_history_belongs_to_class(history)


@pytest.mark.parametrize(
    ("nodes", "metric", "controls"),
    (
        ((), None, {}),
        (("one",), None, {}),
        (("a", "b", "c"), None, {}),
        (("same", "same"), None, {}),
        (([], "right"), None, {}),
        (("left", "right"), (1.0,), {}),
        (("left", "right"), (1.0, 0.0), {}),
        (("left", "right"), (1.0, math.nan), {}),
        (("left", "right"), None, {"tau_local": False}),
        (("left", "right"), None, {"tau_global": 0}),
        (("left", "right"), None, {"epi_bound": False}),
        (("left", "right"), None, {"epi_bound": 0.0}),
        (("left", "right"), None, {"epi_bound": -1.0}),
        (("left", "right"), None, {"epi_bound": math.nan}),
        (("left", "right"), None, {"epi_bound": math.inf}),
        (
            ("left", "right"),
            None,
            {"epi_bound": 3 * math.ulp(0.0)},
        ),
        (
            ("left", "right"),
            None,
            {"epi_bound": Fraction(1, 2**2000)},
        ),
    ),
)
def test_constructor_rejects_invalid_support_metric_delays_or_bound(
    nodes: object,
    metric: object,
    controls: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "tau_local": 2,
        "tau_global": 3,
        "epi_bound": 1.0,
    }
    arguments.update(controls)

    with pytest.raises(TNFRValueError):
        certify_half_alpha_antisymmetric_hard_clip_remesh_class(
            nodes,
            metric,
            **arguments,
        )


def test_asymmetric_interval_tamper_invalidates_the_sealed_class() -> None:
    certificate = _certificate()
    asymmetric = replace(certificate.configuration, epi_min=-0.5)
    object.__setattr__(certificate, "configuration", asymmetric)

    assert not (
        certificate.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert not certificate.remesh_class_forward_invariant_certified
    assert not certificate.represented_history_belongs_to_class(
        ((0.0, -0.0),) * certificate.required_history_length
    )


def test_class_is_frozen_unsealed_construction_fails_and_tampering_closes() -> None:
    certificate = _certificate()

    with pytest.raises(FrozenInstanceError):
        certificate.alpha = Fraction(1)  # type: ignore[misc]

    payload = {
        item.name: object.__getattribute__(certificate, item.name)
        for item in fields(type(certificate))
        if item.name != "_proof_stamp"
    }
    unsealed = UniformHalfAlphaAntisymmetricHardClipRemeshClassCertificate(
        **payload
    )
    object.__setattr__(
        certificate,
        "exact_uniform_relative_defect_upper_bound",
        Fraction(1),
    )

    assert not (
        unsealed.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert not (
        certificate.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert certificate.failed_conditions == (
        "half_alpha_antisymmetric_hard_clip_remesh_class_proof_fields_intact",
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("exact_tail_error_linear_coefficient", Fraction(0)),
        ("exact_tail_error_absolute_coefficient", Fraction(0)),
        ("exact_tail_error_ratio_at_eleven_subnormals", Fraction(0)),
        ("exact_tail_relative_defect_upper_bound", Fraction(0)),
        ("finite_core_candidate_count", 6614),
        ("finite_core_admissible_count", 3889),
        ("finite_core_maximizer_amplitudes", (-3, -2, 3)),
    ),
)
def test_tampering_any_global_bound_proof_field_fails_closed(
    field_name: str,
    replacement: object,
) -> None:
    certificate = _certificate()
    object.__setattr__(certificate, field_name, replacement)

    assert not (
        certificate.half_alpha_antisymmetric_hard_clip_class_certificate_certified
    )
    assert not certificate.uniform_binary64_relative_defect_bound_certified
    assert not certificate.uniform_relative_defect_bound_is_sharp_certified


def test_general_metric_centering_is_not_a_forward_invariant_claim() -> None:
    amplitudes = (
        float.fromhex("0x1.37b40d1c51f86p-29"),
        float.fromhex("0x1.2be8875fa6dd8p-29"),
        float.fromhex("-0x1.0994982458cc8p-28"),
    )
    current, local, global_ = tuple((3 * value, -value) for value in amplitudes)
    observation = observe_binary64_remesh_pair_relative_defect(
        current,
        local,
        global_,
        alpha=0.5,
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="hard",
    )

    assert tuple(_weighted_center(pair) for pair in (current, local, global_)) == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
    assert _weighted_center(observation.runtime_bounded_pair) == -Fraction(
        1, 2**84
    )
    assert not _certificate(
        metric_weights=(1.0, 3.0)
    ).represented_history_belongs_to_class((current, local, global_, current))


def test_a_fixed_integer_lattice_is_not_closed_without_quantization() -> None:
    observation = observe_binary64_remesh_pair_relative_defect(
        (1.0, -1.0),
        (0.0, -0.0),
        (0.0, -0.0),
        alpha=0.5,
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="hard",
    )

    assert observation.runtime_bounded_pair == (0.25, -0.25)
    assert any(not value.is_integer() for value in observation.runtime_bounded_pair)
    assert _certificate().represented_history_belongs_to_class(
        (
            (0.0, -0.0),
            (0.0, -0.0),
            (0.0, -0.0),
            (1.0, -1.0),
        )
    )
