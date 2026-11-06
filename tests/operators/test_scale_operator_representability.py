"""Scale events reject accidental zeroing before state or telemetry commit."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.definitions import Contraction, Expansion
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.types import Glyph, real_scalar_epi


def _source(*, vf=1.0, epi=0.2, factor=0.25, edge_aware=False, glyph="NUL"):
    graph = nx.Graph(
        EDGE_AWARE_ENABLED=edge_aware,
        GLYPH_FACTORS={f"{glyph}_scale": factor},
        GRAMMAR_REJECTION_MODE="raise",
    )
    graph.add_node(
        0, EPI=epi, nu_f=vf, delta_nfr=0.2, theta=0.0, glyph_history=["AL", "IL"]
    )
    return graph


def _apply(graph, path, glyph="NUL"):
    operator = Contraction() if glyph == "NUL" else Expansion()
    if path == "glyph":
        apply_glyph(graph, 0, glyph)
    elif path == "public":
        operator(graph, 0)
    else:
        execute_pointwise_stage(graph, operator, (0,))


def _snapshot(graph):
    # Direct dispatch may materialize a NodeNX adapter before the proposal.
    return deepcopy(
        (
            {key: value for key, value in graph.graph.items() if key != "_node_cache"},
            dict(graph.nodes[0]),
            list(graph.edges),
        )
    )


@pytest.mark.parametrize("path", ("glyph", "public", "stage"))
def test_positive_capacity_lost_to_zero_rejects_before_commit(path):
    graph = _source(vf=math.ulp(0.0))
    before = _snapshot(graph)
    with pytest.raises(TNFRValueError, match="capacity product.*underflows"):
        _apply(graph, path)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("path", ("glyph", "public", "stage"))
@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_nonzero_form_lost_before_clipping_rejects_before_commit(path, sign):
    graph = _source(epi=sign * math.ulp(0.0), edge_aware=True)
    before = _snapshot(graph)
    with pytest.raises(TNFRValueError, match="EPI product.*underflows"):
        _apply(graph, path)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("path", ("glyph", "public", "stage"))
@pytest.mark.parametrize(("glyph", "factor"), (("VAL", 2.0), ("NUL", 0.25)))
def test_initial_zero_capacity_and_form_remain_admitted(path, glyph, factor):
    graph = _source(vf=0.0, epi=0.0, factor=factor, edge_aware=True, glyph=glyph)
    _apply(graph, path, glyph)
    assert graph.nodes[0]["nu_f"] == 0.0
    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) == 0.0
    assert list(graph.nodes[0]["glyph_history"])[-1] == glyph


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_representable_subnormal_products_have_no_artificial_floor(sign):
    proposal = propose_scale_operator(
        glyph=Glyph.NUL,
        factor=0.5,
        vf_before=2.0 * math.ulp(0.0),
        dnfr_before=0.0,
        edge_aware_enabled=True,
        epi_before=sign * 2.0 * math.ulp(0.0),
    )
    assert proposal.vf_after == math.ulp(0.0)
    assert proposal.epi_after == sign * math.ulp(0.0)


@pytest.mark.parametrize("glyph", (Glyph.NUL, Glyph.VAL))
def test_explicit_zero_boundary_action_is_not_underflow(glyph):
    proposal = propose_scale_operator(
        glyph=glyph,
        factor=0.5 if glyph is Glyph.NUL else 2.0,
        vf_before=1.0,
        dnfr_before=0.0,
        edge_aware_enabled=True,
        epi_before=-0.2,
        epi_min=0.0,
        epi_max=1.0,
    )
    assert proposal.epi_after == 0.0
    if glyph is Glyph.NUL:
        assert proposal.raw_epi_after == -0.1
        assert proposal.clip_delta == 0.1
    else:
        assert proposal.effective_epi_scale == 0.0
