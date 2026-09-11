"""Finite causal P2 Reception/REMESH extinction certificates."""

from __future__ import annotations

from collections import deque
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

import tnfr.physics.runtime_p2_reception_remesh_sequence as sequence_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
    execute_event_remesh_cycle_sequence,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.binary64_p2_reception_stability import (
    certify_p2_half_reception_remesh_stability,
)
from tnfr.physics.binary64_remesh_relative_defect import (
    certify_alpha_one_hard_clip_remesh_class,
)
from tnfr.physics.runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate,
    certify_executed_p2_half_reception_remesh_sequence,
)

_WORD = ("reception", "coherence", "recursivity")


def _preserve_pressure(_graph: nx.Graph) -> None:
    """Supply the executor-required post-REMESH refresh boundary."""


def _graph(
    *,
    tau_global: int = 1,
    stale_history_outside_interval: bool = False,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        RANDOM_SEED=7,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={
            "EN_mix": 0.5,
            "IL_lambda": 0.1,
            "REMESH_alpha": 1.0,
        },
        REMESH_TAU_LOCAL=1,
        REMESH_TAU_GLOBAL=tau_global,
        REMESH_ALPHA=1.0,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        compute_delta_nfr=_preserve_pressure,
    )
    for node, epi in enumerate((-1.0, 0.5)):
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
    initial_rows = [
        {0: 0.5, 1: -0.5},
        {0: 1.0, 1: -1.0},
    ][-tau_global:]
    if stale_history_outside_interval:
        initial_rows.insert(0, {0: 2.0, 1: -2.0})
    graph.graph["_epi_hist"] = deque(initial_rows, maxlen=64)
    return graph


def _specs(count: int) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    return tuple(
        EventRemeshCycleExecutionSpec(
            build_operator_event_schedule(
                _WORD,
                start_time=0.0,
                flow_durations=(0.0, 0.0, 0.0, 0.0),
            )
        )
        for _index in range(count)
    )


def _specs_with_post_reception_flow(
    count: int,
) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    specs: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(count):
        schedule = build_operator_event_schedule(
            _WORD,
            start_time=start,
            flow_durations=(0.0, 0.125, 0.0, 0.0),
        )
        specs.append(EventRemeshCycleExecutionSpec(schedule))
        start = schedule.end_time
    return tuple(specs)


def _execution(
    *,
    count: int = 2,
    tau_global: int = 1,
    stale_history_outside_interval: bool = False,
    weights=(1.0, 1.0),
):
    return execute_event_remesh_cycle_sequence(
        _graph(
            tau_global=tau_global,
            stale_history_outside_interval=stale_history_outside_interval,
        ),
        _specs(count),
        metric_weights=weights,
        context={"initial_epi_nonzero": True},
        suppress_birth_warnings=True,
        require_runtime_telescope=False,
    )


def _kernel(*, tau_global: int = 1, weights=(1.0, 1.0)):
    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        weights,
        tau_local=1,
        tau_global=tau_global,
        epi_min=-1.0,
        epi_max=1.0,
    )
    return certify_p2_half_reception_remesh_stability(source)


@pytest.fixture(scope="module")
def certified_sequence():
    execution = _execution(count=3)
    kernel = _kernel()
    certificate = certify_executed_p2_half_reception_remesh_sequence(
        kernel,
        execution,
    )
    return execution, kernel, certificate


def test_direct_module_exposes_only_the_narrow_public_api() -> None:
    assert sequence_module.__all__ == (
        "ExecutedP2HalfReceptionRemeshSequenceCertificate",
        "certify_executed_p2_half_reception_remesh_sequence",
    )


def test_more_than_one_horizon_certifies_finite_causal_extinction(
    certified_sequence,
) -> None:
    execution, kernel, certificate = certified_sequence

    assert type(certificate) is ExecutedP2HalfReceptionRemeshSequenceCertificate
    assert certificate.kernel_certificate is kernel
    assert certificate.execution is execution
    assert certificate.cycle_indices == (0, 1, 2)
    assert certificate.reception_event_indices == (0, 0, 0)
    assert certificate.node_order == (0, 1)
    assert certificate.exact_normalized_metric == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert certificate.cycle_count == 3
    assert certificate.active_history_extinction_horizon == 2
    assert certificate.guaranteed_extinction_cycle_index == 1
    assert certificate.exact_post_reception_energies == (0, 0, 0)
    assert certificate.exact_post_remesh_energies[0] > 0
    assert certificate.exact_post_remesh_energies[1:] == (0, 0)
    assert all(len(row) == 2 for row in certificate.exact_active_history_suffixes)
    assert certificate.exact_final_post_remesh_energy == 0
    assert all(
        stage.execution_result is cycle.event_execution
        for stage, cycle in zip(
            certificate.reception_stage_certificates,
            execution.cycles,
            strict=True,
        )
    )
    assert all(
        bridge.cycle_result is cycle
        for bridge, cycle in zip(
            certificate.remesh_history_bridges,
            execution.cycles,
            strict=True,
        )
    )
    assert (
        certificate
        .executed_p2_half_reception_remesh_sequence_certificate_certified
    )


def test_cycle_count_must_reach_the_global_delay_horizon() -> None:
    execution = _execution(count=2, tau_global=2)
    kernel = _kernel(tau_global=2)

    with pytest.raises(TNFRValueError, match="extinction horizon"):
        certify_executed_p2_half_reception_remesh_sequence(kernel, execution)


def test_stale_rows_before_the_active_history_suffix_are_out_of_scope() -> None:
    execution = _execution(stale_history_outside_interval=True)
    certificate = certify_executed_p2_half_reception_remesh_sequence(
        _kernel(),
        execution,
    )

    assert execution.cycles[0].history_transition.outgoing_exact_history[0] == (
        Fraction(2),
        Fraction(-2),
    )
    assert (Fraction(2), Fraction(-2)) not in (
        certificate.exact_active_history_suffixes[0]
    )
    assert all(
        -1 <= value <= 1
        for suffix in certificate.exact_active_history_suffixes
        for row in suffix
        for value in row
    )
    assert (
        certificate
        .executed_p2_half_reception_remesh_sequence_certificate_certified
    )


def test_active_suffix_need_not_fill_an_inactive_longer_local_delay() -> None:
    graph = _graph(tau_global=1)
    graph.graph["REMESH_TAU_LOCAL"] = 5
    graph.graph["_epi_hist"] = deque(
        ({0: 0.5, 1: -0.5} for _index in range(5)),
        maxlen=64,
    )
    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        (1.0, 1.0),
        tau_local=5,
        tau_global=1,
        epi_min=-1.0,
        epi_max=1.0,
    )
    kernel = certify_p2_half_reception_remesh_stability(source)
    execution = execute_event_remesh_cycle_sequence(
        graph,
        _specs(2),
        metric_weights=(1.0, 1.0),
        context={"initial_epi_nonzero": True},
        suppress_birth_warnings=True,
        require_runtime_telescope=False,
    )
    certificate = certify_executed_p2_half_reception_remesh_sequence(
        kernel,
        execution,
    )

    assert source.required_history_length == 6
    assert certificate.active_history_extinction_horizon == 2
    assert all(
        len(suffix) == certificate.active_history_extinction_horizon
        for suffix in certificate.exact_active_history_suffixes
    )
    assert all(
        source.epi_min <= value <= source.epi_max
        for suffix in certificate.exact_active_history_suffixes
        for row in suffix
        for value in row
    )
    assert certificate.finite_causal_extinction_certified


def test_explicit_indices_are_strict_and_select_reception(
    certified_sequence,
) -> None:
    execution, kernel, certificate = certified_sequence

    explicit = certify_executed_p2_half_reception_remesh_sequence(
        kernel,
        execution,
        reception_event_indices=(0, 0, 0),
    )
    assert explicit.exact_post_remesh_energies == (
        certificate.exact_post_remesh_energies
    )
    with pytest.raises(TNFRValueError, match="Reception/EN"):
        certify_executed_p2_half_reception_remesh_sequence(
            kernel,
            execution,
            reception_event_indices=(1, 1, 1),
        )
    for invalid in (
        (0,),
        [0, 0, 0],
        (False, 0, 0),
        (0.0, 0, 0),
        (-1, 0, 0),
    ):
        with pytest.raises(TNFRValueError, match="reception_event_indices"):
            certify_executed_p2_half_reception_remesh_sequence(
                kernel,
                execution,
                reception_event_indices=invalid,  # type: ignore[arg-type]
            )


def test_source_metric_must_match_every_executed_cycle(
    certified_sequence,
) -> None:
    execution, _kernel_certificate, _certificate = certified_sequence
    mismatched = _kernel(weights=(1.0, 3.0))

    with pytest.raises(TNFRValueError, match="source metric"):
        certify_executed_p2_half_reception_remesh_sequence(
            mismatched,
            execution,
        )


def test_public_energy_series_use_the_normalized_metric_ray() -> None:
    baseline = certify_executed_p2_half_reception_remesh_sequence(
        _kernel(weights=(1.0, 1.0)),
        _execution(count=3, weights=(1.0, 1.0)),
    )
    scaled = certify_executed_p2_half_reception_remesh_sequence(
        _kernel(weights=(2.0, 2.0)),
        _execution(count=3, weights=(2.0, 2.0)),
    )

    assert baseline.exact_schedule_input_energies[0] == Fraction(9, 32)
    assert baseline.exact_post_remesh_energies[0] == Fraction(1, 2)
    assert scaled.exact_normalized_metric == baseline.exact_normalized_metric
    assert scaled.exact_schedule_input_energies == (
        baseline.exact_schedule_input_energies
    )
    assert scaled.exact_post_reception_energies == (
        baseline.exact_post_reception_energies
    )
    assert scaled.exact_post_remesh_energies == baseline.exact_post_remesh_energies


def test_common_alpha_source_check_rejects_per_cycle_drift(
    certified_sequence,
) -> None:
    execution, kernel, certificate = certified_sequence
    cycle = execution.cycles[1]
    bridge = certificate.remesh_history_bridges[1]
    source = kernel.remesh_class_certificate

    assert not sequence_module._remesh_configuration_matches_source(
        cycle,
        bridge,
        source=source,
        nodes=kernel.node_order,
        metric=kernel.exact_normalized_metric,
        alpha_source="different_runtime_source",
    )


def test_epi_flow_after_reception_cannot_be_hidden_in_the_cycle() -> None:
    graph = _graph()
    graph.nodes[0]["delta_nfr"] = 1.0
    graph.nodes[1]["delta_nfr"] = -1.0
    execution = execute_event_remesh_cycle_sequence(
        graph,
        _specs_with_post_reception_flow(2),
        metric_weights=(1.0, 1.0),
        context={"initial_epi_nonzero": True},
        suppress_birth_warnings=True,
        require_runtime_telescope=False,
    )

    assert (
        execution.cycles[0].event_execution.glyph_stage_evidence[0].right.epi
        != execution.cycles[0].pre_remesh_epi.epi_values
    )
    with pytest.raises(TNFRValueError, match="schedule output"):
        certify_executed_p2_half_reception_remesh_sequence(
            _kernel(),
            execution,
        )


def test_nested_private_reseal_cannot_substitute_a_stage_or_bridge(
    certified_sequence,
) -> None:
    _execution, _kernel_certificate, certificate = certified_sequence
    altered_stages = tuple(reversed(certificate.reception_stage_certificates))
    altered = replace(certificate, reception_stage_certificates=altered_stages)
    resealed = sequence_module._seal(altered)
    assert not (
        resealed.executed_p2_half_reception_remesh_sequence_certificate_certified
    )
    assert resealed.failed_conditions == (
        "executed_p2_reception_remesh_sequence_proof_fields_intact",
    )

    altered_bridges = tuple(reversed(certificate.remesh_history_bridges))
    altered = replace(certificate, remesh_history_bridges=altered_bridges)
    resealed = sequence_module._seal(altered)
    assert not (
        resealed.executed_p2_half_reception_remesh_sequence_certificate_certified
    )


def test_certificate_is_frozen_sealed_and_refuses_broader_claims(
    certified_sequence,
) -> None:
    _execution, _kernel_certificate, certificate = certified_sequence

    with pytest.raises(FrozenInstanceError):
        certificate.cycle_count = 3  # type: ignore[misc]
    altered = replace(certificate, exact_final_post_remesh_energy=Fraction(1))
    assert not (
        altered.executed_p2_half_reception_remesh_sequence_certificate_certified
    )
    assert not certificate.future_runtime_stability_certified
    assert not certificate.unobserved_repetition_stability_certified
    assert not certificate.auxiliary_state_stability_certified
    assert not certificate.current_live_graph_state_bound
    assert not certificate.solver_accuracy_certified
    assert not certificate.full_tnfr_stability_certified
    assert "one executor-sealed finite causal p2 sequence" in (
        certificate.scope.lower()
    )
