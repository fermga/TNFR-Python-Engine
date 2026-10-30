"""Runtime recording and configured consumption of Mutation evidence.

The selector controls explicitly request ZHIR; the acceptance control also
supplies timestamped samples and disables grammar. They test evidence gating,
not autonomous default selection or endogenous occurrence of Mutation.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import (
    DNFR_PRIMARY,
    EPI_PRIMARY,
    THETA_PRIMARY,
    VF_PRIMARY,
    inject_defaults,
)
from tnfr.dynamics.runtime import _record_mutation_flow_boundary, _update_nodes
from tnfr.dynamics.selectors import _apply_glyphs
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.physics.mutation_trigger import certify_mutation_trigger
from tnfr.types import serialize_bepi


def _graph(*, epi: float = 0.0, nu_f: float = 1.0, dnfr: float = 0.4):
    graph = nx.Graph()
    graph.add_node(
        0,
        **{
            EPI_PRIMARY: epi,
            VF_PRIMARY: nu_f,
            DNFR_PRIMARY: dnfr,
            THETA_PRIMARY: 0.0,
        },
    )
    inject_defaults(graph)
    graph.graph["compute_delta_nfr"] = lambda _graph: None
    graph.graph["INTEGRATOR_METHOD"] = "euler"
    return graph


def test_runtime_step_produces_physical_two_point_mutation_evidence():
    graph = _graph()

    _update_nodes(
        graph,
        dt=0.5,
        use_Si=False,
        apply_glyphs=False,
        step_idx=0,
        hist={},
    )

    history = list(graph.nodes[0]["epi_time_history"])
    assert history[0] == pytest.approx((0.0, 0.0))
    assert history[-1][0] == pytest.approx(0.5)
    assert graph.graph["_t"] == pytest.approx(0.5)

    certificate = certify_mutation_trigger(
        current_epi=graph.nodes[0][EPI_PRIMARY],
        nu_f=graph.nodes[0][VF_PRIMARY],
        delta_nfr=graph.nodes[0][DNFR_PRIMARY],
        epi_time_history=history,
    )
    assert certificate.evidence_valid is True
    assert certificate.physical_time_resolved is True
    assert certificate.observed_depi_dt == pytest.approx(0.4)


def test_same_time_epi_jump_restarts_history_instead_of_fabricating_rate():
    graph = _graph()
    _record_mutation_flow_boundary(graph)

    graph.nodes[0][EPI_PRIMARY] = 0.7
    _record_mutation_flow_boundary(graph)

    assert list(graph.nodes[0]["epi_time_history"]) == [(0.0, 0.7)]
    certificate = certify_mutation_trigger(
        current_epi=0.7,
        nu_f=1.0,
        delta_nfr=0.4,
        epi_time_history=graph.nodes[0]["epi_time_history"],
    )
    assert certificate.evidence_available is False
    assert certificate.observed_crossed is None


def test_history_recording_is_atomic_across_nodes_on_invalid_input():
    graph = _graph()
    graph.add_node(
        1,
        **{
            EPI_PRIMARY: 0.0,
            VF_PRIMARY: 1.0,
            DNFR_PRIMARY: 0.1,
            THETA_PRIMARY: 0.0,
            "epi_time_history": [(1.0, 0.0), (0.0, 0.1)],
        },
    )
    before = deepcopy(dict(graph.nodes[0]))

    with pytest.raises(TNFRValueError, match="increase strictly"):
        _record_mutation_flow_boundary(graph)

    assert dict(graph.nodes[0]) == before


@pytest.mark.parametrize(
    "location", ["clock", "live_epi", "history_time", "history_epi"]
)
def test_history_rejects_lost_nonzero_samples_before_any_node_write(location):
    graph = _graph()
    graph.add_node(1, **dict(graph.nodes[0]))
    tiny = Fraction(1, 10**400)
    if location == "clock":
        graph.graph["_t"] = tiny
    elif location == "live_epi":
        graph.nodes[1][EPI_PRIMARY] = tiny
    elif location == "history_time":
        graph.nodes[1]["epi_time_history"] = [(tiny, 0.0)]
    else:
        graph.nodes[1]["epi_time_history"] = [(0.0, tiny)]
    before_nodes, before_graph = deepcopy(dict(graph.nodes(data=True))), deepcopy(
        graph.graph
    )
    with pytest.raises(TNFRValueError):
        _record_mutation_flow_boundary(graph)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


@pytest.mark.parametrize("serialized", [False, True])
def test_history_cannot_record_rich_form_magnitude_as_a_scalar_state(serialized):
    rich = BEPIElement((0.25, -0.25), (0.25, -0.25), (0.0, 1.0))
    graph = _graph()
    graph.add_node(1, **dict(graph.nodes[0]))
    graph.nodes[1][EPI_PRIMARY] = serialize_bepi(rich) if serialized else rich
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises(TNFRValueError, match="uniform-real EPI"):
        _record_mutation_flow_boundary(graph)
    assert dict(graph.nodes(data=True)) == before


def test_history_preserves_signed_uniform_form_and_representable_subnormal_time():
    tiny = Fraction.from_float(float.fromhex("0x0.0000000000001p-1022"))
    graph = _graph(epi=serialize_bepi(-0.25))
    graph.graph["_t"] = tiny
    _record_mutation_flow_boundary(graph)
    assert tuple(graph.nodes[0]["epi_time_history"]) == ((float(tiny), -0.25),)


def test_configured_selector_abstains_when_mutation_evidence_is_missing():
    graph = _graph(epi=0.3)
    history_state = {}

    _apply_glyphs(graph, lambda _graph, _node: "ZHIR", history_state)

    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]
    abstention = history_state["mutation_abstentions"][-1]
    assert abstention["requested_glyph"] == "ZHIR"
    assert abstention["applied_glyph"] == "IL"
    assert abstention["reason"] == "missing_history"
    assert abstention["observed_depi_dt"] is None


def test_configured_selector_executes_mutation_with_matching_physical_samples():
    graph = _graph(epi=0.2)
    graph.graph["GRAMMAR_CANON"] = {"enabled": False}
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.0), (1.0, 0.2)]
    graph.graph["_t"] = 1.0
    history_state = {}

    _apply_glyphs(graph, lambda _graph, _node: "ZHIR", history_state)

    assert list(graph.nodes[0]["glyph_history"]) == ["ZHIR"]
    assert "mutation_abstentions" not in history_state
    assert graph.nodes[0][THETA_PRIMARY] != pytest.approx(0.0)
