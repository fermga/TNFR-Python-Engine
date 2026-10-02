"""Pointwise state/proposal validation must precede nodal and lifecycle writes."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import Contraction, Emission, Expansion, Silence
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import Glyph

_OPERATORS = {"AL": Emission, "SHA": Silence, "VAL": Expansion, "NUL": Contraction}


def _graph() -> nx.Graph:
    graph = nx.Graph(RANDOM_SEED=17)
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 2.0,
            ALIAS_DNFR[0]: 0.3,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": ["AL", "IL"],
        },
    )
    return graph


def _execute(graph, glyph, route):
    if route == "glyph":
        apply_glyph(graph, 0, glyph)
    elif route == "public":
        _OPERATORS[glyph]()(graph, 0)
    else:
        execute_pointwise_stage(graph, _OPERATORS[glyph](), (0,))


@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
def test_emission_rejects_soft_projection_that_reverses_its_signed_effect(route):
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = -0.99
    graph.graph.update(CLIP_MODE="soft", GLYPH_FACTORS={"AL_boost": 0.001})
    before = deepcopy(dict(graph.nodes[0]))

    with pytest.raises(TNFRValueError, match="must not decrease EPI"):
        _execute(graph, "AL", route)

    assert dict(graph.nodes[0]) == before
    assert "_emission_activated" not in graph.nodes[0]
    # The same declared boost with hard projection satisfies the AL contract.
    graph.graph["CLIP_MODE"] = "hard"
    _execute(graph, "AL", route)
    assert get_attr(graph.nodes[0], ALIAS_EPI) == pytest.approx(-0.989)


def test_public_emission_rejects_bad_bounds_before_clearing_latency_or_lineage():
    graph = _graph()
    graph.graph.update(EPI_MIN=1.0, EPI_MAX=0.0)
    graph.nodes[0].update(latent=True, preserved_epi=0.4, silence_duration=0.0)
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(TNFRValueError, match="EPI_MIN"):
        Emission()(graph, 0)
    assert dict(graph.nodes[0]) == before


@pytest.mark.parametrize(
    "glyph,route,aliases,value",
    [
        ("SHA", "glyph", ALIAS_VF, -1.0),
        ("SHA", "public", ALIAS_VF, "bad"),
        ("SHA", "stage", ALIAS_VF, float("nan")),
        ("VAL", "glyph", ALIAS_VF, True),
        ("VAL", "stage", ALIAS_VF, "2"),
        ("NUL", "glyph", ALIAS_DNFR, "bad"),
        ("NUL", "stage", ALIAS_DNFR, True),
    ],
)
def test_consumed_raw_state_cannot_be_coerced_into_a_pointwise_proposal(
    glyph, route, aliases, value
):
    graph = _graph()
    graph.nodes[0][aliases[0]] = value
    if glyph == "SHA":
        # Zero attenuation must not erase an invalid source capacity.
        graph.graph["GLYPH_FACTORS"] = {"SHA_vf_factor": 0.0}
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(TNFRValueError):
        _execute(graph, glyph, route)
    assert dict(graph.nodes[0]) == before
    assert "latent" not in graph.nodes[0]


@pytest.mark.parametrize(
    "glyph,sink,epi",
    [("NUL", "nul_densification_log", 0.4), ("VAL", "edge_aware_interventions", 0.99)],
)
def test_scale_telemetry_sink_rejection_precedes_all_primary_writes(glyph, sink, epi):
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = epi
    graph.graph[sink] = {}
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(OperatorPreconditionError, match=sink):
        apply_glyph(graph, 0, Glyph(glyph))
    assert dict(graph.nodes[0]) == before
    assert graph.graph[sink] == {}


def test_inactive_scale_telemetry_and_epi_branch_are_not_new_admission_requirements():
    graph = _graph()
    graph.graph.update(
        EDGE_AWARE_ENABLED=False,
        EPI_MIN="unused",
        EPI_MAX=None,
        edge_aware_interventions={},
        nul_densification_log={},
        GLYPH_FACTORS={"VAL_scale": 1.5},
    )
    before = get_attr(graph.nodes[0], ALIAS_EPI)
    apply_glyph(graph, 0, "VAL")
    assert get_attr(graph.nodes[0], ALIAS_VF) == 3.0
    assert get_attr(graph.nodes[0], ALIAS_EPI) == before
