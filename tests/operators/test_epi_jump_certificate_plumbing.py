"""Executor-owned plumbing for optional all-target EPI jump evidence."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import networkx as nx
import numpy as np
import pytest

from tnfr.operators.definitions import (
    Coherence,
    Emission,
    Expansion,
    Reception,
    Resonance,
    Silence,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import real_scalar_epi


def _graph(node_count: int = 3) -> nx.Graph:
    graph = nx.path_graph(node_count)
    graph.graph.update(
        GLYPH_FACTORS={
            "EN_mix": 0.25,
            "IL_dnfr_factor": 0.8,
            "RA_epi_diff": 0.25,
            "RA_vf_amplification": 0.25,
            "RA_phase_coupling": 0.5,
            "VAL_scale": 1.1,
        },
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.2 + 0.1 * node,
            nu_f=1.0,
            theta=0.1 * node,
            delta_nfr=0.2,
            EPI_kind="wave",
            glyph_history=["AL", "IL"],
        )
    return graph


@pytest.mark.parametrize("operator_type", [Reception, Resonance])
def test_neighbor_certificate_is_bound_to_the_committed_stage(
    operator_type: type[Any],
) -> None:
    graph = _graph()

    with pytest.warns(UserWarning) if operator_type is Reception else _no_warning():
        result = execute_network_operator_stage(
            graph,
            operator_type(),
            tuple(graph),
            include_epi_jump_certificate=True,
        )

    certificate = result.neighbor_epi_jump_certificate
    assert certificate is not None
    assert result.epi_jump_certificate is certificate
    assert result.epi_jump_certificate_kind == "neighbor"
    assert result.epi_jump_certificate_abstention_reason is None
    assert certificate.repetitions_requested == 1
    assert certificate.repetitions_completed == 1
    assert tuple(certificate.steps[0].runtime_accepted_state_after) == tuple(
        float(real_scalar_epi(graph.nodes[node]["EPI"])) for node in graph
    )


class _no_warning:
    """Minimal context manager paired with ``pytest.warns`` above."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


def test_pointwise_certificate_uses_generic_result_accessors() -> None:
    graph = _graph()
    result = execute_network_operator_stage(
        graph,
        Expansion(),
        tuple(graph),
        include_epi_jump_certificate=True,
    )

    assert result.pointwise_epi_jump_certificate is not None
    assert result.epi_jump_certificate is result.pointwise_epi_jump_certificate
    assert result.epi_jump_certificate_kind == "pointwise"
    assert result.epi_jump_certificate_abstention_reason is None


def test_unsupported_glyph_abstains_without_changing_execution() -> None:
    graph = _graph()
    with pytest.warns(UserWarning):
        result = execute_network_operator_stage(
            graph,
            Coherence(),
            tuple(graph),
            include_epi_jump_certificate=True,
        )

    assert result.schedule == "two_phase_jacobi"
    assert result.epi_jump_certificate is None
    assert result.epi_jump_certificate_kind is None
    assert (
        result.epi_jump_certificate_abstention_reason
        == "epi_jump_certificate_unavailable_for_glyph:IL"
    )


def test_certificate_domain_rejection_does_not_abort_valid_singleton_stage() -> None:
    graph = _graph(1)

    with pytest.warns(UserWarning):
        result = execute_network_operator_stage(
            graph,
            Reception(),
            tuple(graph),
            include_epi_jump_certificate=True,
        )

    assert result.nodes_processed == 1
    assert result.epi_jump_certificate is None
    assert "certificate_domain_rejected" in (
        result.epi_jump_certificate_abstention_reason or ""
    )


def test_neighbor_certificate_mismatch_rolls_back_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.physics import network_stage_stability

    graph = _graph()
    baseline = network_stage_stability.certify_all_target_neighbor_stage(
        graph,
        "reception",
        fixed_support_declared=True,
        repetitions=1,
        mix_factor=0.25,
    )
    wrong_state = np.array(
        baseline.steps[0].runtime_accepted_state_after,
        dtype=float,
        copy=True,
    )
    wrong_state[0] += 0.125
    wrong_state.setflags(write=False)
    wrong_step = replace(
        baseline.steps[0],
        runtime_accepted_state_after=wrong_state,
    )
    tampered = replace(baseline, steps=(wrong_step,))
    monkeypatch.setattr(
        network_stage_stability,
        "certify_all_target_neighbor_stage",
        lambda *args, **kwargs: tampered,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(RuntimeError, match="frozen stage proposals"):
        execute_network_operator_stage(
            graph,
            Reception(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


def test_neighbor_certificate_rejects_a_different_resolved_mix_on_uniform_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coincident endpoints cannot authenticate a different affine stage map."""

    from tnfr.physics import network_stage_stability

    graph = _graph()
    for node in graph:
        graph.nodes[node]["EPI"] = 0.2
    original = network_stage_stability.certify_all_target_neighbor_stage

    def certify_wrong_mix(*args: Any, **kwargs: Any):
        kwargs["mix_factor"] = 0.75
        return original(*args, **kwargs)

    monkeypatch.setattr(
        network_stage_stability,
        "certify_all_target_neighbor_stage",
        certify_wrong_mix,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(RuntimeError, match="resolved runtime factors"):
        execute_network_operator_stage(
            graph,
            Reception(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


@pytest.mark.parametrize(
    ("factor", "wrong_value"),
    [
        ("mix_factor", 0.75),
        ("vf_amplification_factor", 0.75),
        ("phase_coupling_factor", 0.75),
    ],
)
def test_resonance_certificate_binds_every_resolved_runtime_factor(
    monkeypatch: pytest.MonkeyPatch,
    factor: str,
    wrong_value: float,
) -> None:
    """Dormant RA channels cannot hide a certificate-factor mismatch."""

    from tnfr.physics import network_stage_stability

    graph = _graph()
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0)
    original = network_stage_stability.certify_all_target_neighbor_stage

    def certify_wrong_factor(*args: Any, **kwargs: Any):
        kwargs[factor] = wrong_value
        return original(*args, **kwargs)

    monkeypatch.setattr(
        network_stage_stability,
        "certify_all_target_neighbor_stage",
        certify_wrong_factor,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(
        RuntimeError,
        match="resolved runtime factors|local certificate",
    ):
        execute_network_operator_stage(
            graph,
            Resonance(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("internal certificate type failure"),
        ValueError("internal certificate value failure"),
        ZeroDivisionError("internal certificate arithmetic failure"),
        nx.NetworkXError("internal certificate graph failure"),
    ],
    ids=("type", "value", "arithmetic", "networkx"),
)
def test_internal_neighbor_certificate_failures_propagate_and_roll_back(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    from tnfr.physics import network_stage_stability

    graph = _graph()

    def fail_certificate(*_args: Any, **_kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(
        network_stage_stability,
        "certify_all_target_neighbor_stage",
        fail_certificate,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(type(failure), match="internal certificate"):
        execute_network_operator_stage(
            graph,
            Reception(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


def test_wrong_glyph_pointwise_certificate_rolls_back_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.physics import pointwise_stage_stability

    certificate_graph = _graph()
    wrong_certificate = execute_network_operator_stage(
        certificate_graph,
        Silence(),
        tuple(certificate_graph),
        include_epi_jump_certificate=True,
    ).pointwise_epi_jump_certificate
    assert wrong_certificate is not None

    graph = _graph()
    monkeypatch.setattr(
        pointwise_stage_stability,
        "certify_pointwise_epi_jump_realization",
        lambda *_args, **_kwargs: wrong_certificate,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(RuntimeError, match="identity or schedule"):
        execute_network_operator_stage(
            graph,
            Expansion(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


def test_pointwise_certificate_binds_affine_offset_when_clipping_hides_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tnfr.physics import pointwise_stage_stability

    certificate_graph = _graph()
    graph = _graph()
    for target in (certificate_graph, graph):
        for node in target:
            target.nodes[node]["EPI"] = 1.0
    certificate_graph.graph["GLYPH_FACTORS"]["AL_boost"] = 0.75
    graph.graph["GLYPH_FACTORS"]["AL_boost"] = 0.25
    wrong_certificate = execute_network_operator_stage(
        certificate_graph,
        Emission(),
        tuple(certificate_graph),
        include_epi_jump_certificate=True,
    ).pointwise_epi_jump_certificate
    assert wrong_certificate is not None

    monkeypatch.setattr(
        pointwise_stage_stability,
        "certify_pointwise_epi_jump_realization",
        lambda *_args, **_kwargs: wrong_certificate,
    )
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    callback_calls = 0

    def callback(_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1

    with pytest.raises(RuntimeError, match="affine map"):
        execute_network_operator_stage(
            graph,
            Emission(),
            tuple(graph),
            compute_delta_nfr=callback,
            include_epi_jump_certificate=True,
        )

    assert callback_calls == 0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


def test_dispatcher_certificate_flag_is_strict_before_mutation() -> None:
    graph = _graph()
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TypeError, match="must be a bool"):
        execute_network_operator_stage(
            graph,
            Expansion(),
            tuple(graph),
            include_epi_jump_certificate=1,
        )

    assert dict(graph.nodes(data=True)) == before


def test_certificate_plumbing_is_disabled_by_default() -> None:
    graph = _graph()
    result = execute_network_operator_stage(graph, Expansion(), tuple(graph))

    assert result.pointwise_epi_jump_certificate is None
    assert result.neighbor_epi_jump_certificate is None
    assert result.epi_jump_certificate is None
    assert result.epi_jump_certificate_kind is None
    assert result.epi_jump_certificate_abstention_reason is None
