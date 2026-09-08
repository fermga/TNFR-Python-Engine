"""Immutable pointwise proposals for AL and SHA network stages."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators.al_sha_stage_proposals import (
    EmissionStageProposal,
    SilenceStageProposal,
    commit_emission_lifecycle,
    commit_emission_structure,
    commit_silence_lifecycle,
    commit_silence_structure,
    propose_emission_stage,
    propose_silence_stage,
)


def _graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(
            EPI=0.1 * (node + 1),
            nu_f=1.0 + node,
            phase=0.0,
            DeltaNFR=0.0,
            glyph_history=["AL"],
        )
    return graph


def _logical_state(graph: nx.Graph) -> tuple[object, object]:
    return (
        deepcopy({node: dict(data) for node, data in graph.nodes(data=True)}),
        deepcopy(dict(graph.graph)),
    )


def test_emission_proposal_is_read_only_and_deeply_value_shaped() -> None:
    graph = _graph()
    graph.nodes[0].update(
        latent=True,
        latency_start_time="old",
        preserved_epi=0.1,
        silence_duration=0.0,
        was_initial_on_silence=False,
    )
    before = _logical_state(graph)

    proposal = propose_emission_stage(
        graph,
        0,
        {"AL_boost": 0.2},
        timestamp="2030-01-02T03:04:05+00:00",
    )

    assert _logical_state(graph) == before
    assert proposal.epi_before == pytest.approx(0.1)
    assert proposal.epi_after == pytest.approx(0.3)
    assert proposal.latency_keys_to_clear == (
        "latent",
        "latency_start_time",
        "preserved_epi",
        "silence_duration",
        "was_initial_on_silence",
    )
    assert proposal.warning_messages == ()
    assert proposal.initialize_emission is True
    assert proposal.emission_timestamp == "2030-01-02T03:04:05+00:00"
    assert proposal.increment_lineage is False
    with pytest.raises(FrozenInstanceError):
        proposal.epi_after = 1.0  # type: ignore[misc]


def test_first_emission_commit_replays_exact_lifecycle_patch_then_epi() -> None:
    graph = _graph()
    graph.nodes[0].update(
        latent=True,
        latency_start_time="old",
        preserved_epi=0.1,
        silence_duration=0.0,
        was_initial_on_silence=False,
    )
    timestamp = "2030-01-02T03:04:05+00:00"
    proposal = propose_emission_stage(
        graph, 0, {"AL_boost": 0.2}, timestamp=timestamp
    )

    commit_emission_lifecycle(graph, proposal)

    for key in (
        "latent",
        "latency_start_time",
        "preserved_epi",
        "silence_duration",
        "was_initial_on_silence",
    ):
        assert key not in graph.nodes[0]
    assert graph.nodes[0]["emission_timestamp"] == timestamp
    assert graph.nodes[0]["_emission_activated"] is True
    assert graph.nodes[0]["_emission_origin"] == timestamp
    assert graph.nodes[0]["_structural_lineage"] == {
        "origin": timestamp,
        "activation_count": 1,
        "derived_nodes": [],
        "parent_emission": None,
    }
    assert get_attr(graph.nodes[0], ALIAS_EPI) == pytest.approx(0.1)

    commit_emission_structure(graph, proposal)

    assert get_attr(graph.nodes[0], ALIAS_EPI) == pytest.approx(0.3)
    assert 0 in graph.graph["_node_cache"]


def test_repeated_emission_preserves_origin_and_increments_existing_lineage() -> None:
    graph = _graph()
    lineage = {
        "origin": "2029-01-01T00:00:00+00:00",
        "activation_count": 4,
        "derived_nodes": ["child"],
        "parent_emission": "parent",
    }
    graph.nodes[0].update(
        _emission_activated=True,
        _emission_origin=lineage["origin"],
        emission_timestamp=lineage["origin"],
        _structural_lineage=lineage,
    )
    proposal = propose_emission_stage(
        graph,
        0,
        {"AL_boost": 0.1},
        timestamp="2030-01-02T03:04:05+00:00",
    )

    assert proposal.initialize_emission is False
    assert proposal.emission_timestamp is None
    assert proposal.increment_lineage is True

    commit_emission_lifecycle(graph, proposal)

    assert graph.nodes[0]["emission_timestamp"] == lineage["origin"]
    assert graph.nodes[0]["_emission_origin"] == lineage["origin"]
    assert graph.nodes[0]["_structural_lineage"] is lineage
    assert lineage == {
        "origin": "2029-01-01T00:00:00+00:00",
        "activation_count": 5,
        "derived_nodes": ["child"],
        "parent_emission": "parent",
    }


def test_emission_warning_plan_is_immutable_and_emitted_before_metadata() -> None:
    graph = _graph()
    graph.graph["MAX_SILENCE_DURATION"] = 2.0
    graph.nodes[0].update(
        latent=True,
        latency_start_time="old",
        preserved_epi=0.2,
        silence_duration=3.0,
        was_initial_on_silence=False,
    )
    proposal = propose_emission_stage(
        graph,
        0,
        {"AL_boost": 0.1},
        timestamp="2030-01-02T03:04:05+00:00",
    )

    assert len(proposal.warning_messages) == 2
    with pytest.warns(UserWarning) as captured:
        commit_emission_lifecycle(graph, proposal)
    assert [str(item.message) for item in captured] == list(
        proposal.warning_messages
    )
    assert "latent" not in graph.nodes[0]


def test_emission_proposal_applies_canonical_boundary_before_commit() -> None:
    graph = _graph()
    graph.nodes[0]["EPI"] = 0.95
    before = _logical_state(graph)

    proposal = propose_emission_stage(
        graph,
        0,
        {"AL_boost": 0.2},
        timestamp="2030-01-02T03:04:05+00:00",
    )

    assert proposal.epi_after == pytest.approx(1.0)
    assert _logical_state(graph) == before


def test_silence_proposal_is_read_only_frozen_and_commits_exact_patch() -> None:
    graph = _graph()
    graph.nodes[1].update(
        latent=False,
        latency_start_time="old",
        preserved_epi=-9.0,
        silence_duration=17.0,
        was_initial_on_silence=True,
    )
    before = _logical_state(graph)
    timestamp = "2030-01-02T03:04:05+00:00"

    proposal = propose_silence_stage(
        graph,
        1,
        {"SHA_vf_factor": 0.5},
        timestamp=timestamp,
    )

    assert _logical_state(graph) == before
    assert proposal.vf_before == pytest.approx(2.0)
    assert proposal.vf_after == pytest.approx(1.0)
    assert proposal.preserved_epi == pytest.approx(0.2)
    assert proposal.was_initial_on_silence is False
    with pytest.raises(FrozenInstanceError):
        proposal.vf_after = 0.0  # type: ignore[misc]

    commit_silence_lifecycle(graph, proposal)
    assert graph.nodes[1]["latent"] is True
    assert graph.nodes[1]["latency_start_time"] == timestamp
    assert graph.nodes[1]["preserved_epi"] == pytest.approx(0.2)
    assert graph.nodes[1]["silence_duration"] == 0.0
    assert graph.nodes[1]["was_initial_on_silence"] is False
    assert get_attr(graph.nodes[1], ALIAS_VF) == pytest.approx(2.0)

    commit_silence_structure(graph, proposal)
    assert get_attr(graph.nodes[1], ALIAS_VF) == pytest.approx(1.0)
    assert 1 in graph.graph["_node_cache"]


@pytest.mark.parametrize("glyph", ["AL", "SHA"])
def test_pointwise_primary_commits_are_target_order_invariant(glyph: str) -> None:
    forward = _graph()
    reverse = deepcopy(forward)
    timestamp = "2030-01-02T03:04:05+00:00"
    if glyph == "AL":
        proposals = tuple(
            propose_emission_stage(
                forward,
                node,
                {"AL_boost": 0.2},
                timestamp=timestamp,
            )
            for node in forward
        )
        reverse_proposals = tuple(
            propose_emission_stage(
                reverse,
                node,
                {"AL_boost": 0.2},
                timestamp=timestamp,
            )
            for node in reverse
        )
        for proposal in proposals:
            commit_emission_lifecycle(forward, proposal)
            commit_emission_structure(forward, proposal)
        for proposal in reversed(reverse_proposals):
            commit_emission_lifecycle(reverse, proposal)
            commit_emission_structure(reverse, proposal)
        alias = ALIAS_EPI
    else:
        proposals = tuple(
            propose_silence_stage(
                forward,
                node,
                {"SHA_vf_factor": 0.5},
                timestamp=timestamp,
            )
            for node in forward
        )
        reverse_proposals = tuple(
            propose_silence_stage(
                reverse,
                node,
                {"SHA_vf_factor": 0.5},
                timestamp=timestamp,
            )
            for node in reverse
        )
        for proposal in proposals:
            commit_silence_lifecycle(forward, proposal)
            commit_silence_structure(forward, proposal)
        for proposal in reversed(reverse_proposals):
            commit_silence_lifecycle(reverse, proposal)
            commit_silence_structure(reverse, proposal)
        alias = ALIAS_VF

    assert tuple(get_attr(forward.nodes[node], alias) for node in forward) == (
        tuple(get_attr(reverse.nodes[node], alias) for node in reverse)
    )


@pytest.mark.parametrize(
    ("builder", "factors", "timestamp", "match"),
    [
        (propose_emission_stage, {"AL_boost": float("nan")}, "time", "finite"),
        (propose_silence_stage, {"SHA_vf_factor": 0.5}, "", "timestamp"),
    ],
)
def test_invalid_proposal_inputs_leave_graph_unchanged(
    builder, factors, timestamp, match
) -> None:
    graph = _graph()
    before = _logical_state(graph)

    with pytest.raises((TNFRValueError, ValueError), match=match):
        builder(graph, 0, factors, timestamp=timestamp)

    assert _logical_state(graph) == before


def test_proposal_records_have_operator_specific_types_and_glyphs() -> None:
    graph = _graph()
    timestamp = "2030-01-02T03:04:05+00:00"

    emission = propose_emission_stage(graph, 0, {}, timestamp=timestamp)
    silence = propose_silence_stage(graph, 0, {}, timestamp=timestamp)

    assert isinstance(emission, EmissionStageProposal)
    assert isinstance(silence, SilenceStageProposal)
    assert emission.glyph.value == "AL"
    assert silence.glyph.value == "SHA"
