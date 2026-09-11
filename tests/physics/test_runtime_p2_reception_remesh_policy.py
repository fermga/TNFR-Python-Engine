"""Transactional runtime policy for finite P2 Reception/REMESH sequences."""

from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
import pytest

import tnfr.physics.runtime_p2_reception_remesh_policy as policy_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.binary64_p2_reception_stability import (
    certify_p2_half_reception_remesh_stability,
)
from tnfr.physics.binary64_remesh_relative_defect import (
    certify_alpha_one_hard_clip_remesh_class,
)
from tnfr.physics.runtime_p2_reception_remesh_policy import (
    execute_p2_half_reception_remesh_policy_invocation,
)
from tnfr.physics.runtime_p2_reception_remesh_sequence import (
    ExecutedP2HalfReceptionRemeshSequenceCertificate,
)
from tnfr.utils._structural_signature import structural_proof_signature

_WORD = ("reception", "coherence", "recursivity")


def _preserve_pressure(_graph: nx.Graph) -> None:
    """Supply the required post-REMESH pressure-refresh boundary."""


def _graph(
    *,
    tau_local: int = 1,
    tau_global: int = 1,
    directed: bool = False,
) -> nx.Graph:
    if directed:
        graph = nx.DiGraph()
        graph.add_edges_from(((0, 1), (1, 0)))
    else:
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
        REMESH_TAU_LOCAL=tau_local,
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
    graph.graph["_epi_hist"] = deque(
        ({0: 0.5, 1: 0.25} for _index in range(max(tau_local, tau_global))),
        maxlen=64,
    )
    return graph


def _kernel(*, tau_local: int = 1, tau_global: int = 1):
    source = certify_alpha_one_hard_clip_remesh_class(
        (0, 1),
        (1.0, 1.0),
        tau_local=tau_local,
        tau_global=tau_global,
        epi_min=-1.0,
        epi_max=1.0,
    )
    return certify_p2_half_reception_remesh_stability(source)


def _specs(count: int, *, start_time: float = 0.0):
    return tuple(
        EventRemeshCycleExecutionSpec(
            build_operator_event_schedule(
                _WORD,
                start_time=start_time,
                flow_durations=(0.0, 0.0, 0.0, 0.0),
            )
        )
        for _index in range(count)
    )


def _graph_signature(graph: nx.Graph) -> tuple[Any, ...]:
    return (
        structural_proof_signature(graph),
        id(graph.graph),
        id(graph._node),
        id(graph._adj),
        tuple(id(graph.nodes[node]) for node in graph),
        id(graph.graph["_epi_hist"]),
    )


def _assert_graph_unchanged(
    graph: nx.Graph,
    signature: tuple[Any, ...],
    history: deque[Any],
) -> None:
    assert graph.graph["_epi_hist"] is history
    assert _graph_signature(graph) == signature


def _invoke(graph: nx.Graph, kernel, specs):
    return execute_p2_half_reception_remesh_policy_invocation(
        graph,
        kernel,
        specs,
        metric_weights=(1.0, 1.0),
        suppress_birth_warnings=True,
    )


def test_policy_executes_and_certifies_one_finite_invocation() -> None:
    graph = _graph()
    certificate = _invoke(graph, _kernel(), _specs(2))

    assert type(certificate) is ExecutedP2HalfReceptionRemeshSequenceCertificate
    assert certificate.cycle_count == 2
    assert certificate.execution.runtime_telescope is None
    assert certificate.execution.runtime_telescope_required is False
    assert certificate.exact_post_remesh_energies[-1] == 0
    assert certificate.finite_causal_extinction_certified
    assert certificate.future_runtime_stability_certified is False
    assert certificate.unobserved_repetition_stability_certified is False
    assert certificate.auxiliary_state_stability_certified is False
    assert certificate.current_live_graph_state_bound is False


def test_policy_revalidates_and_accepts_two_successive_invocations() -> None:
    graph = _graph()
    kernel = _kernel()
    specs = _specs(2)

    first = _invoke(graph, kernel, specs)
    first_history = tuple(graph.graph["_epi_hist"])
    second = _invoke(graph, kernel, specs)

    assert first is not second
    assert first.execution is not second.execution
    assert second.finite_causal_extinction_certified
    assert len(graph.graph["_epi_hist"]) == len(first_history) + 2
    assert second.exact_post_remesh_energies == (0, 0)


def test_reciprocal_directed_preflight_rejection_rolls_back() -> None:
    graph = _graph(directed=True)
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    with pytest.raises(TNFRValueError, match="undirected graph support"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)


def test_supported_indexed_history_is_rebuilt_by_the_canonical_append() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = list(graph.graph["_epi_hist"])

    certificate = _invoke(graph, _kernel(), _specs(2))

    assert certificate.finite_causal_extinction_certified
    assert type(graph.graph["_epi_hist"]) is deque
    assert graph.graph["_epi_hist"].maxlen == 64


def test_noncanonical_schedule_is_rejected_without_writes() -> None:
    graph = _graph()
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)
    specs = tuple(
        EventRemeshCycleExecutionSpec(
            build_operator_event_schedule(
                _WORD,
                start_time=0.0,
                flow_durations=(0.0, 0.125, 0.0, 0.0),
            )
        )
        for _index in range(2)
    )

    with pytest.raises(TNFRValueError):
        _invoke(graph, _kernel(), specs)

    _assert_graph_unchanged(graph, before, history)
    assert "hybrid_event_log" not in graph.graph


def test_policy_rejects_a_noncanonical_partition_source_without_consuming_it(
) -> None:
    graph = _graph()
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)
    base = _specs(2)
    specs = (
        EventRemeshCycleExecutionSpec(
            base[0].schedule,
            physical_flow_partitions=[],
        ),
        base[1],
    )

    with pytest.raises(TNFRValueError, match="partition sources"):
        _invoke(graph, _kernel(), specs)

    _assert_graph_unchanged(graph, before, history)


def test_postcertification_failure_rolls_back_the_complete_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    def fail_after_execution(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("postcertification failure")

    monkeypatch.setattr(
        policy_module,
        "certify_executed_p2_half_reception_remesh_sequence",
        fail_after_execution,
    )
    with pytest.raises(RuntimeError, match="postcertification failure"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)
    assert "hybrid_event_log" not in graph.graph


def test_invalid_postcertification_result_rolls_back_the_complete_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    monkeypatch.setattr(
        policy_module,
        "certify_executed_p2_half_reception_remesh_sequence",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(TNFRValueError, match="invalid certificate"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)
    assert "hybrid_event_log" not in graph.graph


def test_postcertification_graph_mutation_is_detected_and_rolled_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _graph()
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)
    original = policy_module.certify_executed_p2_half_reception_remesh_sequence

    def mutate_after_certification(*args: Any, **kwargs: Any) -> Any:
        certificate = original(*args, **kwargs)
        graph.graph["postcertification_side_effect"] = True
        return certificate

    monkeypatch.setattr(
        policy_module,
        "certify_executed_p2_half_reception_remesh_sequence",
        mutate_after_certification,
    )
    with pytest.raises(TNFRValueError, match="materialization changed"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)
    assert "postcertification_side_effect" not in graph.graph


@pytest.mark.parametrize(
    "boundary",
    ("factor", "remesh_alpha", "metric", "support"),
)
def test_static_runtime_preconditions_fail_without_writes(boundary: str) -> None:
    graph = _graph()
    metric = (1.0, 1.0)
    if boundary == "factor":
        graph.graph["GLYPH_FACTORS"]["EN_mix"] = 0.75
    elif boundary == "remesh_alpha":
        graph.graph["REMESH_ALPHA"] = 0.5
    elif boundary == "metric":
        metric = (2.0, 1.0)
    else:
        graph.remove_edge(0, 1)

    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)
    with pytest.raises((TNFRValueError, ValueError)):
        execute_p2_half_reception_remesh_policy_invocation(
            graph,
            _kernel(),
            _specs(2),
            metric_weights=metric,
            suppress_birth_warnings=True,
        )

    _assert_graph_unchanged(graph, before, history)
    assert "hybrid_event_log" not in graph.graph


def test_zero_live_epi_cannot_claim_preexisting_form() -> None:
    graph = _graph()
    for node in graph:
        graph.nodes[node]["EPI"] = 0.0
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    with pytest.raises(TNFRValueError, match="pre-existing nonzero EPI"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)


def test_each_cycle_rederives_u1a_from_its_live_start_and_rolls_back() -> None:
    graph = _graph()
    graph.graph["_epi_hist"] = deque([{0: 0.0, 1: 0.0}], maxlen=64)
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    with pytest.raises(TNFRValueError):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)
    assert "hybrid_event_log" not in graph.graph


def test_only_the_active_global_history_suffix_must_be_in_the_interval() -> None:
    graph = _graph(tau_local=5, tau_global=1)
    rows = tuple(graph.graph["_epi_hist"])
    rows[0][0] = 2.0
    rows[0][1] = -2.0

    certificate = _invoke(
        graph,
        _kernel(tau_local=5, tau_global=1),
        _specs(2),
    )

    assert certificate.active_history_extinction_horizon == 2
    assert certificate.finite_causal_extinction_certified


def test_active_incoming_history_outside_the_interval_is_rejected() -> None:
    graph = _graph()
    graph.graph["_epi_hist"][-1][0] = 2.0
    history = graph.graph["_epi_hist"]
    before = _graph_signature(graph)

    with pytest.raises(TNFRValueError, match="active incoming"):
        _invoke(graph, _kernel(), _specs(2))

    _assert_graph_unchanged(graph, before, history)
