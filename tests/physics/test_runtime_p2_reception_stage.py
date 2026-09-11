"""Finite executor binding for the global P2 half-Reception kernel."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from fractions import Fraction

import networkx as nx
import pytest

import tnfr.operators.event_runtime as event_runtime_module
import tnfr.physics.runtime_p2_reception_stage as runtime_p2_module
from tnfr.constants.canonical import EN_MIX_FACTOR
from tnfr.errors import TNFRValueError
from tnfr.operators.event_runtime import (
    OperatorEventExecutionResult,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.binary64_p2_reception_stability import (
    certify_p2_half_reception_remesh_stability,
)
from tnfr.physics.binary64_remesh_relative_defect import (
    certify_alpha_one_hard_clip_remesh_class,
)
from tnfr.physics.runtime_p2_reception_stage import (
    ExecutedP2HalfReceptionStageCertificate,
    certify_executed_p2_half_reception_stage,
)

_WORD = ("reception", "coherence", "recursivity")
_TINY = float.fromhex("0x0.0000000000001p-1022")


def _execute(
    pair: tuple[float, float],
    *,
    lower: float,
    upper: float,
    en_mix: float,
):
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=lower,
        EPI_MAX=upper,
        RANDOM_SEED=7,
        GLYPH_FACTORS={
            "EN_mix": en_mix,
            "IL_lambda": 0.1,
            "REMESH_alpha": 1.0,
        },
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=1,
    )
    for node, epi in zip(graph, pair, strict=True):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            latent=False,
            glyph_history=["AL"],
            epi_history=[epi, epi],
        )
    schedule = build_operator_event_schedule(
        _WORD,
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0),
    )
    return execute_operator_event_schedule(
        graph,
        schedule,
        context={"initial_epi_nonzero": True},
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )


def _kernel(
    *,
    lower: float,
    upper: float,
    weights=(1.0, 1.0),
    nodes=(0, 1),
):
    source = certify_alpha_one_hard_clip_remesh_class(
        nodes,
        weights,
        tau_local=1,
        tau_global=1,
        epi_min=lower,
        epi_max=upper,
    )
    return certify_p2_half_reception_remesh_stability(source)


@pytest.fixture(scope="module")
def underflow_case():
    execution = _execute(
        (_TINY, 0.0),
        lower=-_TINY,
        upper=_TINY,
        en_mix=0.5,
    )
    kernel = _kernel(lower=-_TINY, upper=_TINY)
    certificate = certify_executed_p2_half_reception_stage(kernel, execution)
    return _TINY, execution, kernel, certificate


@pytest.fixture(scope="module")
def signed_zero_case():
    execution = _execute(
        (-_TINY, -0.0),
        lower=-1.0,
        upper=1.0,
        en_mix=0.5,
    )
    kernel = _kernel(lower=-1.0, upper=1.0)
    certificate = certify_executed_p2_half_reception_stage(kernel, execution)
    return execution, kernel, certificate


@pytest.fixture(scope="module")
def default_factor_execution():
    return _execute(
        (0.0, 0.0),
        lower=-1.0,
        upper=1.0,
        en_mix=EN_MIX_FACTOR,
    )


def test_direct_module_exposes_only_the_narrow_public_api() -> None:
    assert runtime_p2_module.__all__ == (
        "ExecutedP2HalfReceptionStageCertificate",
        "certify_executed_p2_half_reception_stage",
    )


def test_execution_binds_the_real_en_stage_to_the_global_q_zero_kernel(
    underflow_case,
) -> None:
    tiny, execution, kernel, certificate = underflow_case

    assert type(certificate) is ExecutedP2HalfReceptionStageCertificate
    assert certificate.kernel_certificate is kernel
    assert certificate.execution_result is execution
    assert certificate.executed_stage is execution.glyph_stage_evidence[0]
    assert certificate.event_index == 0
    assert certificate.node_order == (0, 1)
    assert certificate.binary64_epi_before == (tiny, 0.0)
    assert certificate.binary64_epi_after == (0.0, 0.0)
    assert certificate.exact_normalized_metric == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert certificate.runtime_neighbor_indices == ((1,), (0,))
    assert certificate.exact_mix_factor == Fraction(1, 2)
    assert certificate.clip_mode == "hard"
    assert certificate.exact_epi_lower_bound == Fraction.from_float(-tiny)
    assert certificate.exact_epi_upper_bound == Fraction.from_float(tiny)
    assert certificate.exact_centered_energy_before > 0
    assert certificate.exact_centered_energy_after == 0
    assert certificate.exact_global_kernel_energy_gain_upper_bound == 0
    assert certificate.runtime_kernel_replay_matches_by_bits
    assert certificate.observed_output_binary64_bits_identical
    assert certificate.executed_p2_half_reception_stage_certificate_certified
    assert certificate.finite_executor_bound_epi_stage_certified
    assert certificate.finite_grammar_admission_observed
    assert certificate.canonical_two_phase_reception_stage_observed
    assert certificate.runtime_p2_support_observed
    assert certificate.observed_capacity_and_effective_conductance_preserved
    assert certificate.observed_source_metric_preserved
    assert certificate.source_global_q_zero_applies_to_observed_epi_transition
    assert certificate.observed_numeric_consensus_certified
    assert certificate.finite_schedule_graph_state_atomicity_certified
    assert certificate.failed_conditions == ()


def test_underflow_bridge_does_not_depend_on_global_binary64_affinity(
    underflow_case,
) -> None:
    _tiny, execution, _kernel_certificate, certificate = underflow_case
    stage = execution.glyph_stage_evidence[0]

    assert not stage.represented_affine_gain_bound_at_observed_endpoint_certified
    assert not certificate.represented_affine_bridge_available_at_observed_endpoint
    assert certificate.executed_p2_half_reception_stage_certificate_certified
    assert certificate.exact_centered_energy_before > 0
    assert certificate.exact_centered_energy_after == 0


def test_private_reseal_cannot_forge_the_executor_abstention_reason(
    underflow_case,
) -> None:
    _tiny, execution, kernel, _certificate = underflow_case
    stage = execution.glyph_stage_evidence[0]
    altered_stage = replace(
        stage,
        certificate_abstention_reason="neighbor_certificate_type_mismatch",
    )
    altered_stage = replace(
        altered_stage,
        _proof_stamp=event_runtime_module._executed_glyph_stage_stamp(
            altered_stage
        ),
    )
    altered_stages = (altered_stage,) + execution.glyph_stage_evidence[1:]
    altered_composition = (
        event_runtime_module._compose_observed_represented_epi_schedule(
            execution.schedule,
            execution.target_nodes,
            execution.flow_interval_evidence,
            altered_stages,
            execution.physical_flow_partition_evidence,
        )
    )
    execution_payload = {
        item.name: object.__getattribute__(execution, item.name)
        for item in fields(OperatorEventExecutionResult)
        if item.init and item.name != "_proof_stamp"
    }
    execution_payload["glyph_stage_evidence"] = altered_stages
    execution_payload["represented_epi_schedule_composition"] = (
        altered_composition
    )
    unsealed_execution = OperatorEventExecutionResult(**execution_payload)
    altered_execution = replace(
        unsealed_execution,
        _proof_stamp=event_runtime_module._sealed_dataclass_stamp(
            unsealed_execution,
            event_runtime_module._OPERATOR_EVENT_EXECUTION_RESULT_PROOF_VERSION,
        ),
    )
    assert altered_execution._proof_fields_are_intact()

    with pytest.raises(TNFRValueError, match="affine-bridge observation"):
        certify_executed_p2_half_reception_stage(kernel, altered_execution)


def test_signed_zero_bits_are_observed_without_becoming_a_global_claim(
    signed_zero_case,
) -> None:
    _execution, _kernel_certificate, certificate = signed_zero_case

    assert tuple(value.hex() for value in certificate.binary64_epi_after) == (
        "0x0.0p+0",
        "-0x0.0p+0",
    )
    assert certificate.binary64_epi_after[0] == certificate.binary64_epi_after[1]
    assert not certificate.observed_output_binary64_bits_identical
    assert certificate.runtime_kernel_replay_matches_by_bits
    assert certificate.exact_centered_energy_after == 0
    assert certificate.observed_numeric_consensus_certified


def test_canonical_default_en_factor_cannot_be_promoted_to_the_half_mix_class(
    default_factor_execution,
) -> None:
    assert EN_MIX_FACTOR.hex() == "0x1.ee7eea04ddca0p-3"
    kernel = _kernel(lower=-1.0, upper=1.0)

    with pytest.raises(TNFRValueError, match="exact binary64 one half"):
        certify_executed_p2_half_reception_stage(
            kernel,
            default_factor_execution,
        )


def test_private_outer_reseal_cannot_hide_inconsistent_local_factor_evidence(
    default_factor_execution,
) -> None:
    original_stage = default_factor_execution.glyph_stage_evidence[0]
    original_certificate = original_stage.certificate
    assert original_certificate is not None
    original_step = original_certificate.steps[0]
    altered_step = replace(
        original_step,
        mix_factor=0.5,
        exact_mix_factor=Fraction(1, 2),
    )
    altered_certificate = replace(original_certificate, steps=(altered_step,))
    altered_stage = replace(original_stage, certificate=altered_certificate)
    altered_stage = replace(
        altered_stage,
        _proof_stamp=event_runtime_module._executed_glyph_stage_stamp(
            altered_stage
        ),
    )
    assert not altered_stage._proof_fields_are_intact()

    execution_payload = {
        item.name: object.__getattribute__(default_factor_execution, item.name)
        for item in fields(OperatorEventExecutionResult)
        if item.init and item.name != "_proof_stamp"
    }
    stages = list(default_factor_execution.glyph_stage_evidence)
    stages[0] = altered_stage
    execution_payload["glyph_stage_evidence"] = tuple(stages)
    with pytest.raises(
        ValueError,
        match="glyph-stage evidence proof fields are not intact",
    ):
        OperatorEventExecutionResult(**execution_payload)


@pytest.mark.parametrize(
    ("kernel", "message"),
    (
        (_kernel(lower=-2.0, upper=2.0), "hard-clamp interval"),
        (
            _kernel(lower=-_TINY, upper=_TINY, weights=(1.0, 3.0)),
            "diffusion metric",
        ),
        (
            _kernel(lower=-_TINY, upper=_TINY, nodes=(1, 0)),
            "node order",
        ),
    ),
)
def test_runtime_interval_and_metric_must_match_the_global_source_class(
    underflow_case,
    kernel,
    message: str,
) -> None:
    _tiny, execution, _source_kernel, _certificate = underflow_case

    with pytest.raises(TNFRValueError, match=message):
        certify_executed_p2_half_reception_stage(kernel, execution)


def test_explicit_index_must_select_the_reception_stage(underflow_case) -> None:
    _tiny, execution, kernel, certificate = underflow_case

    assert (
        certify_executed_p2_half_reception_stage(
            kernel,
            execution,
            event_index=0,
        ).binary64_epi_after
        == certificate.binary64_epi_after
    )
    with pytest.raises(TNFRValueError, match="Reception/EN"):
        certify_executed_p2_half_reception_stage(
            kernel,
            execution,
            event_index=1,
        )
    for invalid in (False, 0.0, "0", -1, 3):
        with pytest.raises(TNFRValueError, match="event_index"):
            certify_executed_p2_half_reception_stage(
                kernel,
                execution,
                event_index=invalid,  # type: ignore[arg-type]
            )


def test_scope_refuses_unobserved_stage_graph_and_future_claims(
    underflow_case,
) -> None:
    _tiny, _execution, _kernel_certificate, certificate = underflow_case

    assert not certificate.complete_reception_stage_stability_class_certified
    assert not certificate.executed_remesh_configuration_bound
    assert not certificate.reception_auxiliary_state_preservation_certified
    assert not certificate.raw_graph_topology_preservation_certified
    assert not certificate.current_live_graph_state_bound
    assert not certificate.future_or_repeated_live_graph_stability_certified
    assert not certificate.solver_accuracy_certified
    assert not certificate.full_tnfr_stability_certified
    assert "this finite epi stage" in certificate.scope.lower()
    assert "atomic schedule provenance" in certificate.scope.lower()
    assert "does not bind the source REMESH history" in certificate.scope


def test_certificate_is_frozen_sealed_and_fails_closed_after_tamper(
    underflow_case,
) -> None:
    _tiny, _execution, _kernel_certificate, certificate = underflow_case

    with pytest.raises(FrozenInstanceError):
        certificate.event_index = 1  # type: ignore[misc]

    altered = replace(certificate, exact_mix_factor=Fraction(1, 3))
    assert not altered.executed_p2_half_reception_stage_certificate_certified
    assert altered.failed_conditions == (
        "executed_p2_reception_stage_proof_fields_intact",
    )

    payload = {
        item.name: object.__getattribute__(certificate, item.name)
        for item in fields(ExecutedP2HalfReceptionStageCertificate)
        if item.name != "_proof_stamp"
    }
    unsealed = ExecutedP2HalfReceptionStageCertificate(**payload)
    assert not unsealed.executed_p2_half_reception_stage_certificate_certified


def test_hostile_private_reseal_fails_closed_without_protocol_calls(
    underflow_case,
) -> None:
    _tiny, _execution, _kernel_certificate, certificate = underflow_case
    calls: list[str] = []

    class Hostile:
        def __float__(self):
            calls.append("float")
            raise AssertionError

        def __eq__(self, _other):
            calls.append("eq")
            raise AssertionError

        def __le__(self, _other):
            calls.append("le")
            raise AssertionError

        def __bool__(self):
            calls.append("bool")
            raise AssertionError

    payload = {
        item.name: object.__getattribute__(certificate, item.name)
        for item in fields(ExecutedP2HalfReceptionStageCertificate)
        if item.name != "_proof_stamp"
    }
    payload["kernel_certificate"] = Hostile()
    hostile = ExecutedP2HalfReceptionStageCertificate(**payload)
    privately_resealed = runtime_p2_module._seal(hostile)

    assert not privately_resealed.executed_p2_half_reception_stage_certificate_certified
    assert privately_resealed.failed_conditions == (
        "executed_p2_reception_stage_proof_fields_intact",
    )
    assert calls == []
