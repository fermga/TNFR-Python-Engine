"""Counterexamples separating advisory routing from public THOL admission."""

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.bifurcation import compute_bifurcation_score, get_bifurcation_paths
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.nodal_equation import compute_d2epi_dt2
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import Glyph


def _prepared_graph():
    graph = nx.path_graph(3)
    graph.graph.update(VALIDATE_OPERATOR_PRECONDITIONS=True, RANDOM_SEED=17)
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.6,
                ALIAS_VF[0]: 1.0,
                ALIAS_DNFR[0]: 0.2,
                ALIAS_THETA[0]: 0.0,
                "epi_history": [0.0, 0.1, 0.6],
                "glyph_history": ["IL", "OZ"],
            }
        )
    return graph


def test_advisory_thol_suggestion_does_not_authorize_public_pressure_gate():
    graph = _prepared_graph()
    graph.nodes[1].update(_bifurcation_ready=True)
    graph.nodes[1][ALIAS_DNFR[0]] = 0.0
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = tuple(graph.edges)

    assert Glyph.THOL in get_bifurcation_paths(graph, 1)
    with pytest.raises(OperatorPreconditionError, match="positive"):
        SelfOrganization()(graph, 1)

    assert dict(graph.nodes(data=True)) == before_nodes
    assert tuple(graph.edges) == before_edges


def test_missing_readiness_flag_is_not_recomputed_from_current_acceleration():
    graph = _prepared_graph()
    before = deepcopy(dict(graph.nodes(data=True)))

    assert compute_d2epi_dt2(graph, 1, store=False) > 0.1
    assert get_bifurcation_paths(graph, 1) == []
    assert dict(graph.nodes(data=True)) == before


def test_score_can_exceed_half_without_any_acceleration():
    score = compute_bifurcation_score(
        d2epi=0.0,
        dnfr=1.0,
        vf=2.0,
        epi=0.9,
        tau=0.25,
    )
    assert score == pytest.approx(0.54)
    assert score > 0.5


@pytest.mark.parametrize("acceleration", (0.25, 0.5, -0.5))
def test_score_can_stay_below_half_at_or_above_its_acceleration_scale(acceleration):
    score = compute_bifurcation_score(
        d2epi=acceleration,
        dnfr=0.0,
        vf=0.0,
        epi=0.0,
        tau=0.25,
    )
    assert abs(acceleration) >= 0.25
    assert score == pytest.approx(0.46)
    assert score < 0.5
