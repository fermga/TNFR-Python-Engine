"""Exact pairwise binary64 REMESH-defect boundary tests."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from fractions import Fraction

import pytest

import tnfr.physics.binary64_remesh_relative_defect as boundary_module
from tnfr.errors import TNFRValueError
from tnfr.physics.binary64_remesh_relative_defect import (
    Binary64RemeshPairRelativeDefectObservation,
    UniformAlphaOneHardClipRemeshClassCertificate,
    certify_alpha_one_hard_clip_remesh_class,
    observe_binary64_remesh_pair_relative_defect,
)


def _observe(
    current_pair: tuple[float, float],
    local_pair: tuple[float, float],
    global_pair: tuple[float, float],
    *,
    alpha: float = 0.5,
    epi_min: float = 0.0,
    epi_max: float = 1.0,
    clip_mode: str = "hard",
) -> Binary64RemeshPairRelativeDefectObservation:
    return observe_binary64_remesh_pair_relative_defect(
        current_pair,
        local_pair,
        global_pair,
        alpha=alpha,
        epi_min=epi_min,
        epi_max=epi_max,
        clip_mode=clip_mode,
    )


def _alpha_one_class(
    *,
    nodes: tuple[str, ...] = ("left", "right"),
    metric_weights: object = None,
    tau_local: int = 2,
    tau_global: int = 3,
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


def test_direct_module_exposes_the_fixed_binary64_boundary_api() -> None:
    expected = {
        "Binary64RemeshPairRelativeDefectObservation",
        "UniformAlphaOneHardClipRemeshClassCertificate",
        "observe_binary64_remesh_pair_relative_defect",
        "certify_alpha_one_hard_clip_remesh_class",
    }

    assert expected <= set(boundary_module.__all__)


def test_normal_bounded_witness_requires_eta_two_to_210_minus_one_quarter() -> None:
    current = math.ldexp(1.0, -52)
    local_left = math.ldexp(1.0, -105)
    local_right = math.nextafter(local_left, math.inf)

    observation = _observe(
        (current, current),
        (local_left, local_right),
        (1.0, 1.0),
    )

    denominator = Fraction(1, 2**316)
    ideal_squared_separation = Fraction(1, 2**318)
    runtime_squared_separation = Fraction(1, 2**106)
    signed_defect = runtime_squared_separation - ideal_squared_separation
    exact_eta = Fraction(2**210) - Fraction(1, 4)

    assert type(observation) is Binary64RemeshPairRelativeDefectObservation
    assert observation.alpha == Fraction(1, 2)
    assert observation.beta == Fraction(1, 4)
    assert observation.gamma == Fraction(1, 4)
    assert observation.delta == Fraction(1, 2)
    assert observation.exact_current_pair == (
        Fraction(1, 2**52),
        Fraction(1, 2**52),
    )
    assert observation.exact_local_pair == (
        Fraction(1, 2**105),
        Fraction(1, 2**105) + Fraction(1, 2**157),
    )
    assert observation.exact_global_pair == (Fraction(1), Fraction(1))
    assert (
        observation.exact_input_pairwise_jensen_denominator == denominator
    )
    assert observation.exact_ideal_squared_separation == ideal_squared_separation
    assert observation.exact_raw_squared_separation == runtime_squared_separation
    assert observation.exact_bounded_squared_separation == (
        runtime_squared_separation
    )
    assert observation.exact_rounding_signed_squared_separation_defect == (
        signed_defect
    )
    assert observation.exact_clipping_signed_squared_separation_defect == 0
    assert observation.exact_total_signed_squared_separation_defect == (
        signed_defect
    )
    assert observation.exact_relative_signed_defect_ratio == exact_eta
    assert (
        observation.exact_minimum_nonnegative_relative_defect_bound
        == exact_eta
    )
    assert Fraction(1, 1) / (1 + exact_eta) == Fraction(
        4,
        2**212 + 3,
    )
    assert Fraction(9, 16) * (1 + exact_eta) > 1
    assert observation.runtime_raw_pair[0].hex() == "0x1.0000000000000p-1"
    assert observation.runtime_raw_pair[1].hex() == "0x1.0000000000001p-1"
    assert observation.exact_runtime_raw_pair == tuple(
        Fraction.from_float(value) for value in observation.runtime_raw_pair
    )
    assert observation.runtime_bounded_pair == observation.runtime_raw_pair
    assert observation.exact_runtime_bounded_pair == (
        observation.exact_runtime_raw_pair
    )
    assert observation.pair_relative_defect_observation_certified
    assert observation.hard_clipping_pairwise_nonexpansive_certified
    assert observation.failed_conditions == ()


def test_two_node_energy_convention_scales_pair_quantities_by_one_quarter() -> None:
    current = math.ldexp(1.0, -52)
    local_left = math.ldexp(1.0, -105)
    observation = _observe(
        (current, current),
        (local_left, math.nextafter(local_left, math.inf)),
        (1.0, 1.0),
    )

    jensen_energy = (
        observation.exact_input_pairwise_jensen_denominator / 4
    )
    centered_energy_defect = (
        observation.exact_total_signed_squared_separation_defect / 4
    )

    assert jensen_energy == Fraction(1, 2**318)
    assert centered_energy_defect == (
        Fraction(1, 2**108) - Fraction(1, 2**320)
    )
    assert centered_energy_defect / jensen_energy == (
        observation.exact_minimum_nonnegative_relative_defect_bound
    )


def test_ideal_head_cancellation_keeps_a_positive_pairwise_denominator() -> None:
    current_left = math.ldexp(1.0, -105)
    current_right = math.nextafter(current_left, math.inf)
    local = math.ldexp(1.0, -52)
    global_left = math.ldexp(1.5, -106)
    global_right = math.nextafter(global_left, -math.inf)

    observation = _observe(
        (current_left, current_right),
        (local, local),
        (global_left, global_right),
        epi_min=-1.0,
    )

    assert observation.exact_ideal_pair[0] == observation.exact_ideal_pair[1]
    assert observation.exact_ideal_squared_separation == 0
    assert observation.exact_input_pairwise_jensen_denominator == (
        Fraction(3, 2**317)
    )
    assert observation.runtime_raw_pair[0].hex() == (
        "0x1.0000000000001p-54"
    )
    assert observation.runtime_raw_pair[1].hex() == (
        "0x1.0000000000002p-54"
    )
    assert observation.exact_raw_squared_separation == Fraction(1, 2**212)
    assert observation.exact_relative_signed_defect_ratio == Fraction(2**105, 3)
    assert (
        observation.exact_minimum_nonnegative_relative_defect_bound
        == Fraction(2**105, 3)
    )


def test_zero_denominator_closes_without_division() -> None:
    observation = _observe(
        (1.0, -1.0),
        (-1.0, 1.0),
        (-0.0, 0.0),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )

    assert observation.exact_input_pairwise_jensen_denominator == 0
    assert observation.exact_ideal_squared_separation == 0
    assert observation.exact_raw_squared_separation == 0
    assert observation.exact_bounded_squared_separation == 0
    assert observation.exact_total_signed_squared_separation_defect == 0
    assert observation.exact_relative_signed_defect_ratio is None
    assert observation.exact_minimum_nonnegative_relative_defect_bound == 0
    assert observation.runtime_raw_pair == (0.0, 0.0)
    assert observation.runtime_raw_pair[0].hex() == "-0x0.0p+0"
    assert observation.runtime_raw_pair[1].hex() == "0x0.0p+0"
    assert observation.pair_relative_defect_observation_certified


def test_common_hard_clamp_can_only_reduce_pairwise_separation() -> None:
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (-2.0, 3.0),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )

    assert observation.exact_input_pairwise_jensen_denominator == 25
    assert observation.exact_ideal_squared_separation == 25
    assert observation.exact_raw_squared_separation == 25
    assert observation.runtime_bounded_pair == (-1.0, 1.0)
    assert observation.exact_bounded_squared_separation == 4
    assert observation.exact_rounding_signed_squared_separation_defect == 0
    assert observation.exact_clipping_signed_squared_separation_defect == -21
    assert observation.exact_total_signed_squared_separation_defect == -21
    assert observation.exact_relative_signed_defect_ratio == Fraction(-21, 25)
    assert observation.exact_minimum_nonnegative_relative_defect_bound == 0
    assert any(observation.clipping_intervened)
    assert observation.hard_clipping_pairwise_nonexpansive_certified
    assert observation.pair_relative_defect_observation_certified


def test_soft_clip_is_recorded_as_a_finite_pair_observation_only() -> None:
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (-0.9999, 0.9999),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
        clip_mode="soft",
    )

    assert observation.clip_mode == "soft"
    assert any(observation.clipping_intervened)
    assert observation.pair_relative_defect_observation_certified
    assert not observation.hard_clipping_pairwise_nonexpansive_certified
    assert not observation.uniform_binary64_relative_defect_bound_certified
    assert not observation.future_binary64_relative_defect_bound_certified
    assert not observation.solver_accuracy_certified
    assert not observation.full_tnfr_stability_certified


@pytest.mark.parametrize(
    "global_pair",
    (
        (-1.0, 1.0),
        (-math.ulp(0.0), math.ulp(0.0)),
        (-0.0, 0.0),
    ),
)
def test_alpha_one_copies_endpoint_subnormal_and_zero_numeric_values(
    global_pair: tuple[float, float],
) -> None:
    observation = _observe(
        (0.75, -0.75),
        (-0.5, 0.5),
        global_pair,
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )

    assert observation.alpha == 1
    assert observation.beta == 0
    assert observation.gamma == 0
    assert observation.delta == 1
    assert observation.exact_ideal_pair == tuple(
        Fraction.from_float(value) for value in global_pair
    )
    assert observation.exact_runtime_raw_pair == observation.exact_ideal_pair
    assert observation.exact_runtime_bounded_pair == observation.exact_ideal_pair
    assert observation.exact_rounding_signed_squared_separation_defect == 0
    assert observation.exact_clipping_signed_squared_separation_defect == 0
    assert observation.exact_total_signed_squared_separation_defect == 0
    assert observation.exact_minimum_nonnegative_relative_defect_bound == 0
    assert observation.pair_relative_defect_observation_certified


def test_alpha_one_signed_zero_claim_is_numeric_not_bitwise() -> None:
    observation = _observe(
        (-0.0, 0.0),
        (0.0, -0.0),
        (-0.0, -0.0),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )

    assert observation.exact_runtime_raw_pair == observation.exact_ideal_pair
    assert observation.runtime_raw_pair == (-0.0, -0.0)
    assert all(value.hex() == "0x0.0p+0" for value in observation.runtime_raw_pair)


def test_alpha_one_handles_maximum_finite_endpoints_without_float_squaring() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (-maximum, maximum),
        alpha=1.0,
        epi_min=-maximum,
        epi_max=maximum,
    )

    expected = tuple(
        Fraction.from_float(value) for value in (-maximum, maximum)
    )
    assert observation.exact_ideal_pair == expected
    assert observation.exact_runtime_raw_pair == expected
    assert observation.exact_runtime_bounded_pair == expected
    assert observation.exact_total_signed_squared_separation_defect == 0
    assert observation.exact_minimum_nonnegative_relative_defect_bound == 0
    assert observation.pair_relative_defect_observation_certified


def test_alpha_one_hard_clip_class_materializes_exact_runtime_contract() -> None:
    certificate = _alpha_one_class()

    assert type(certificate) is UniformAlphaOneHardClipRemeshClassCertificate
    assert certificate.node_order == ("left", "right")
    assert certificate.exact_normalized_metric == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert certificate.tau_local == 2
    assert certificate.tau_global == 3
    assert certificate.history_maxlen == 64
    assert certificate.required_history_length == 4
    assert certificate.epi_min == -1
    assert certificate.epi_max == 1
    assert certificate.alpha == 1
    assert certificate.clip_mode == "hard"
    assert certificate.remesh_certificate.alpha == 1
    assert certificate.exact_uniform_relative_defect_upper_bound == 0
    assert certificate.alpha_one_hard_clip_class_certificate_certified
    assert certificate.binary64_global_delay_numeric_copy_certified
    assert certificate.hard_clip_identity_on_class_certified
    assert certificate.remesh_class_forward_invariant_certified
    assert certificate.failed_conditions == ()
    assert not certificate.schedule_family_certificate_certified
    assert not certificate.repeated_binary64_stability_certified
    assert not certificate.future_binary64_execution_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.full_tnfr_stability_certified


def test_alpha_one_class_normalizes_a_nonuniform_metric_exactly() -> None:
    certificate = _alpha_one_class(metric_weights=(1.0, 3.0))

    assert certificate.exact_normalized_metric == (
        Fraction(1, 4),
        Fraction(3, 4),
    )
    assert certificate.remesh_certificate.alpha_one_pure_delay_map_certified
    assert certificate.alpha_one_hard_clip_class_certificate_certified


def test_alpha_one_class_combines_coincident_delays() -> None:
    certificate = _alpha_one_class(tau_local=3, tau_global=3)

    assert certificate.required_history_length == 4
    assert certificate.remesh_certificate.combined_delay_coefficients == (
        (3, Fraction(1)),
    )
    assert certificate.remesh_certificate.active_max_delay == 3
    assert certificate.alpha_one_hard_clip_class_certificate_certified


def test_alpha_one_class_retains_the_inactive_local_runtime_delay() -> None:
    certificate = _alpha_one_class(tau_local=5, tau_global=2)

    assert certificate.required_history_length == 6
    assert certificate.history_maxlen == 64
    assert certificate.remesh_certificate.active_max_delay == 2
    assert certificate.remesh_certificate.combined_delay_coefficients == (
        (2, Fraction(1)),
    )
    assert certificate.alpha_one_hard_clip_class_certificate_certified


def test_degenerate_interval_is_an_invariant_alpha_one_class() -> None:
    certificate = _alpha_one_class(epi_min=0.0, epi_max=0.0)
    history = ((-0.0, 0.0),) * certificate.required_history_length

    assert certificate.epi_min == 0
    assert certificate.epi_max == 0
    assert certificate.remesh_class_forward_invariant_certified
    assert certificate.represented_history_belongs_to_class(history)


def test_maximum_finite_interval_is_materialized_exactly() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    certificate = _alpha_one_class(epi_min=-maximum, epi_max=maximum)
    row = (-maximum, maximum)
    history = (row,) * certificate.required_history_length

    assert certificate.epi_min == Fraction.from_float(-maximum)
    assert certificate.epi_max == Fraction.from_float(maximum)
    assert certificate.represented_history_belongs_to_class(history)
    assert certificate.alpha_one_hard_clip_class_certificate_certified


def test_runtime_history_class_accepts_endpoints_subnormals_and_signed_zero() -> None:
    certificate = _alpha_one_class()
    tiny = math.ulp(0.0)
    chronological_history = (
        (-1.0, 1.0),
        (-tiny, tiny),
        (-0.0, 0.0),
        (1.0, -1.0),
    )

    assert certificate.represented_history_belongs_to_class(
        chronological_history
    )
    assert certificate.represented_history_belongs_to_class(
        chronological_history * 16
    )


@pytest.mark.parametrize(
    "history",
    (
        (),
        ((0.0, 0.0),) * 3,
        ((0.0, 0.0),) * 65,
        ((0.0,),) * 4,
        ((0.0, 0.0, 0.0),) * 4,
        ((0, 0.0),) * 4,
        ((False, 0.0),) * 4,
        ((math.nan, 0.0),) * 4,
        ((math.inf, 0.0),) * 4,
        ((-math.inf, 0.0),) * 4,
        ((-1.0000000000000002, 0.0),) * 4,
        ((0.0, 1.0000000000000002),) * 4,
        ("not-a-field",) * 4,
    ),
)
def test_runtime_history_class_rejects_out_of_contract_histories(
    history: object,
) -> None:
    assert not _alpha_one_class().represented_history_belongs_to_class(history)


@pytest.mark.parametrize(
    ("current_pair", "local_pair", "global_pair"),
    (
        ((0.0,), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0,), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), (0.0,)),
        ((0, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((False, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((math.nan, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((math.inf, 0.0), (0.0, 0.0), (0.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), ("0", 0.0)),
    ),
)
def test_pair_observer_rejects_nonbinary64_or_nonpair_inputs(
    current_pair: object,
    local_pair: object,
    global_pair: object,
) -> None:
    with pytest.raises(TNFRValueError):
        observe_binary64_remesh_pair_relative_defect(
            current_pair,
            local_pair,
            global_pair,
            alpha=0.5,
            epi_min=-1.0,
            epi_max=1.0,
        )


def test_pair_observer_rejects_hostile_containers_without_iteration() -> None:
    protocol_calls: list[str] = []

    class HostileList(list[float]):
        def __iter__(self):
            protocol_calls.append("iteration dispatched")
            raise SystemExit("must not escape strict input validation")

    with pytest.raises(TNFRValueError):
        observe_binary64_remesh_pair_relative_defect(
            HostileList((0.0, 0.0)),
            (0.0, 0.0),
            (0.0, 0.0),
            alpha=0.5,
            epi_min=-1.0,
            epi_max=1.0,
        )

    assert protocol_calls == []


def test_pair_observer_rejects_hostile_scalar_without_float_dispatch() -> None:
    protocol_calls: list[str] = []

    class HostileFloat:
        def __float__(self) -> float:
            protocol_calls.append("float dispatched")
            raise SystemExit("must not escape strict input validation")

    with pytest.raises(TNFRValueError):
        observe_binary64_remesh_pair_relative_defect(
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            alpha=HostileFloat(),
            epi_min=-1.0,
            epi_max=1.0,
        )

    assert protocol_calls == []


def test_pair_observer_translates_hostile_real_conversion() -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise SystemExit("must be contained at the public boundary")

    with pytest.raises(TNFRValueError):
        observe_binary64_remesh_pair_relative_defect(
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            alpha=HostileFloat(0.5),
            epi_min=-1.0,
            epi_max=1.0,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"alpha": False},
        {"alpha": 0.0},
        {"alpha": math.nan},
        {"alpha": math.inf},
        {"alpha": -math.ulp(0.0)},
        {"alpha": 1.0000000000000002},
        {"epi_min": math.nan},
        {"epi_max": math.inf},
        {"epi_min": 2.0, "epi_max": 1.0},
        {"clip_mode": "inactive"},
        {"clip_mode": 1},
    ),
)
def test_pair_observer_rejects_invalid_controls(overrides: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "alpha": 0.5,
        "epi_min": -1.0,
        "epi_max": 1.0,
        "clip_mode": "hard",
    }
    arguments.update(overrides)

    with pytest.raises(TNFRValueError):
        observe_binary64_remesh_pair_relative_defect(
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
            **arguments,
        )


@pytest.mark.parametrize(
    ("nodes", "metric", "controls"),
    (
        ((), None, {}),
        (("same", "same"), None, {}),
        (([], "right"), None, {}),
        (("left", "right"), (1.0,), {}),
        (("left", "right"), (1.0, 0.0), {}),
        (("left", "right"), (1.0, math.nan), {}),
        (("left", "right"), None, {"tau_local": False}),
        (("left", "right"), None, {"tau_global": 0}),
        (("left", "right"), None, {"epi_min": math.nan}),
        (("left", "right"), None, {"epi_max": math.inf}),
        (
            ("left", "right"),
            None,
            {"epi_min": 2.0, "epi_max": 1.0},
        ),
    ),
)
def test_alpha_one_class_rejects_invalid_support_metric_or_controls(
    nodes: object,
    metric: object,
    controls: dict[str, object],
) -> None:
    arguments: dict[str, object] = {
        "tau_local": 2,
        "tau_global": 3,
        "epi_min": -1.0,
        "epi_max": 1.0,
    }
    arguments.update(controls)

    with pytest.raises(TNFRValueError):
        certify_alpha_one_hard_clip_remesh_class(
            nodes,
            metric,
            **arguments,
        )


@pytest.mark.parametrize("field", ("nodes", "metric", "epi_min"))
def test_alpha_one_class_contains_hostile_input_protocols(field: str) -> None:
    class HostileList(list[object]):
        def __iter__(self):
            raise SystemExit("must be contained at the public boundary")

        def __len__(self) -> int:
            raise SystemExit("must be contained at the public boundary")

    class HostileFloat(float):
        def __float__(self) -> float:
            raise SystemExit("must be contained at the public boundary")

    nodes: object = ("left", "right")
    metric: object = None
    epi_min: object = -1.0
    if field == "nodes":
        nodes = HostileList(("left", "right"))
    elif field == "metric":
        metric = HostileList((1.0, 1.0))
    else:
        epi_min = HostileFloat(-1.0)

    with pytest.raises(TNFRValueError):
        certify_alpha_one_hard_clip_remesh_class(
            nodes,
            metric,
            tau_local=2,
            tau_global=3,
            epi_min=epi_min,
            epi_max=1.0,
        )


def test_pair_observation_is_frozen_and_fails_closed_after_tampering() -> None:
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )

    with pytest.raises(FrozenInstanceError):
        observation.alpha = Fraction(0)  # type: ignore[misc]

    object.__setattr__(
        observation,
        "exact_total_signed_squared_separation_defect",
        Fraction(1),
    )

    assert not observation.pair_relative_defect_observation_certified
    assert not observation.hard_clipping_pairwise_nonexpansive_certified
    assert observation.failed_conditions == (
        "binary64_remesh_pair_relative_defect_proof_fields_intact",
    )


def test_directly_constructed_pair_observation_is_unsealed() -> None:
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    payload = {
        item.name: object.__getattribute__(observation, item.name)
        for item in fields(Binary64RemeshPairRelativeDefectObservation)
        if item.name != "_proof_stamp"
    }

    unsealed = Binary64RemeshPairRelativeDefectObservation(**payload)

    assert not unsealed.pair_relative_defect_observation_certified
    assert unsealed.failed_conditions == (
        "binary64_remesh_pair_relative_defect_proof_fields_intact",
    )


def test_signed_zero_runtime_tamper_invalidates_the_bit_faithful_seal() -> None:
    observation = _observe(
        (1.0, -1.0),
        (-1.0, 1.0),
        (-0.0, 0.0),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    assert observation.runtime_raw_pair[0].hex() == "-0x0.0p+0"

    object.__setattr__(observation, "runtime_raw_pair", (0.0, 0.0))

    assert not observation.pair_relative_defect_observation_certified


def test_hostile_pair_field_and_proof_stamp_fail_without_equality_dispatch() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    field_tamper = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    object.__setattr__(field_tamper, "alpha", AlwaysEqual())
    stamp_tamper = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    object.__setattr__(stamp_tamper, "_proof_stamp", (AlwaysEqual(),))

    assert not field_tamper.pair_relative_defect_observation_certified
    assert not stamp_tamper.pair_relative_defect_observation_certified
    assert equality_calls == []


def test_privately_resealed_hostile_pair_payload_fails_without_dispatch() -> None:
    conversion_calls: list[str] = []

    class HostileFloat(float):
        def __float__(self) -> float:
            conversion_calls.append("called")
            raise SystemExit("must not dispatch during proof validation")

    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    object.__setattr__(observation, "binary64_alpha", HostileFloat(1.0))
    resealed = boundary_module._seal(
        observation,
        Binary64RemeshPairRelativeDefectObservation,
        boundary_module._PAIR_PROOF_VERSION,
    )

    assert not resealed.pair_relative_defect_observation_certified
    assert resealed.failed_conditions == (
        "binary64_remesh_pair_relative_defect_proof_fields_intact",
    )
    assert conversion_calls == []


def test_privately_resealed_inconsistent_pair_field_fails_closed() -> None:
    observation = _observe(
        (0.0, 0.0),
        (0.0, 0.0),
        (0.25, 0.75),
        alpha=1.0,
        epi_min=-1.0,
        epi_max=1.0,
    )
    object.__setattr__(
        observation,
        "exact_total_signed_squared_separation_defect",
        Fraction(1),
    )
    resealed = boundary_module._seal(
        observation,
        Binary64RemeshPairRelativeDefectObservation,
        boundary_module._PAIR_PROOF_VERSION,
    )

    assert not resealed.pair_relative_defect_observation_certified
    assert not resealed.hard_clipping_pairwise_nonexpansive_certified


def test_alpha_one_class_is_frozen_and_fails_closed_after_tampering() -> None:
    certificate = _alpha_one_class()

    with pytest.raises(FrozenInstanceError):
        certificate.alpha = Fraction(0)  # type: ignore[misc]

    object.__setattr__(
        certificate,
        "exact_uniform_relative_defect_upper_bound",
        Fraction(1),
    )

    assert not certificate.alpha_one_hard_clip_class_certificate_certified
    assert not certificate.binary64_global_delay_numeric_copy_certified
    assert not certificate.hard_clip_identity_on_class_certified
    assert not certificate.remesh_class_forward_invariant_certified
    assert certificate.failed_conditions == (
        "alpha_one_hard_clip_remesh_class_proof_fields_intact",
    )
    assert not certificate.represented_history_belongs_to_class(
        ((0.0, 0.0),) * certificate.required_history_length
    )


def test_directly_constructed_alpha_one_class_is_unsealed() -> None:
    certificate = _alpha_one_class()
    payload = {
        item.name: object.__getattribute__(certificate, item.name)
        for item in fields(UniformAlphaOneHardClipRemeshClassCertificate)
        if item.name != "_proof_stamp"
    }

    unsealed = UniformAlphaOneHardClipRemeshClassCertificate(**payload)

    assert not unsealed.alpha_one_hard_clip_class_certificate_certified
    assert unsealed.failed_conditions == (
        "alpha_one_hard_clip_remesh_class_proof_fields_intact",
    )


def test_nested_remesh_certificate_tamper_invalidates_the_outer_class() -> None:
    certificate = _alpha_one_class()
    object.__setattr__(certificate.remesh_certificate, "delta", Fraction(0))

    assert not certificate.remesh_certificate.stability_certificate_certified
    assert not certificate.alpha_one_hard_clip_class_certificate_certified
    assert not certificate.remesh_class_forward_invariant_certified


def test_hostile_class_field_and_proof_stamp_fail_without_equality_dispatch() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    field_tamper = _alpha_one_class()
    object.__setattr__(field_tamper, "alpha", AlwaysEqual())
    stamp_tamper = _alpha_one_class()
    object.__setattr__(stamp_tamper, "_proof_stamp", (AlwaysEqual(),))

    assert not field_tamper.alpha_one_hard_clip_class_certificate_certified
    assert not stamp_tamper.alpha_one_hard_clip_class_certificate_certified
    assert equality_calls == []


def test_privately_resealed_hostile_class_payload_fails_without_dispatch() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    certificate = _alpha_one_class()
    hostile_conditions = list(certificate.conditions)
    hostile_conditions[0] = (hostile_conditions[0][0], AlwaysEqual())
    object.__setattr__(certificate, "conditions", tuple(hostile_conditions))
    resealed = boundary_module._seal(
        certificate,
        UniformAlphaOneHardClipRemeshClassCertificate,
        boundary_module._CLASS_PROOF_VERSION,
    )

    assert not resealed.alpha_one_hard_clip_class_certificate_certified
    assert resealed.failed_conditions == (
        "alpha_one_hard_clip_remesh_class_proof_fields_intact",
    )
    assert equality_calls == []


def test_privately_resealed_inconsistent_class_field_fails_closed() -> None:
    certificate = _alpha_one_class()
    object.__setattr__(
        certificate,
        "exact_uniform_relative_defect_upper_bound",
        Fraction(1),
    )
    resealed = boundary_module._seal(
        certificate,
        UniformAlphaOneHardClipRemeshClassCertificate,
        boundary_module._CLASS_PROOF_VERSION,
    )

    assert not resealed.alpha_one_hard_clip_class_certificate_certified
    assert not resealed.remesh_class_forward_invariant_certified
