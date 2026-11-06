"""Resonance raw boundary admission and its established assignment policy."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.errors import TNFRValueError
from tnfr.node import NodeNX
from tnfr.operators import _op_RA, apply_glyph, get_glyph_factors
from tnfr.physics.reception_realization import certify_reception_epi_realization
from tnfr.physics.resonance_realization import certify_resonance_epi_realization
from tnfr.types import real_scalar_epi

_ROUTES = ("kernel", "glyph", "resonance-certificate", "reception-certificate")
_TINY = Fraction(1, 2**1200)


def _graph(target=0.75, neighbor=1.0):
    graph = nx.path_graph(2)
    for node, value in enumerate((target, neighbor)):
        graph.nodes[node].update(
            EPI=value,
            nu_f=1.0,
            theta=node * 0.25,
            delta_nfr=0.125,
            Si=0.8,
            epi_kind="wave",
            glyph_history=["AL", "IL"],
        )
    graph.graph["GLYPH_FACTORS"] = {
        "RA_epi_diff": 0.5,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
        "EN_mix": 0.5,
    }
    return graph


def _execute(graph, route):
    if route == "kernel":
        node = NodeNX.from_graph(graph, 0)
        _op_RA(node, get_glyph_factors(node, "RA"))
    elif route == "glyph":
        apply_glyph(graph, 0, "RA")
    else:
        certify = (
            certify_resonance_epi_realization
            if route == "resonance-certificate"
            else certify_reception_epi_realization
        )
        return certify(
            graph, 0, fixed_support_declared=True, mix_factor=0.5
        ).runtime_target_value
    return real_scalar_epi(graph.nodes[0]["EPI"])


def _snapshot(graph):
    # Node adapter caches are rebuildable reads, outside the event's state commit.
    return (
        deepcopy(dict(graph.nodes(data=True))),
        deepcopy(tuple(graph.edges(data=True))),
        deepcopy(graph.graph.get("operator_metrics")),
        deepcopy(graph.graph.get("ra_metrics")),
        deepcopy(graph.graph.get("_ra_c_tracking")),
    )


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("key", ("EPI_MIN", "EPI_MAX"))
@pytest.mark.parametrize(
    "invalid",
    (True, False, _TINY, -_TINY, math.inf, math.nan, "0.0", 0j),
    ids=("true", "false", "tiny", "negative-tiny", "inf", "nan", "text", "complex"),
)
def test_invalid_raw_rails_reject_without_event_writes(route, key, invalid):
    graph = _graph()
    graph.graph.update(TRACK_NETWORK_COHERENCE=True, COLLECT_RA_METRICS=True)
    graph.graph[key] = invalid
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match=key):
        _execute(graph, route)

    assert _snapshot(graph) == before


@pytest.mark.parametrize("route", _ROUTES)
def test_reversed_rails_reject_without_event_writes(route):
    graph = _graph()
    graph.graph.update(EPI_MIN=1.0, EPI_MAX=-1.0)
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="EPI_MIN"):
        _execute(graph, route)

    assert _snapshot(graph) == before


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("mode", (None, "hard", "unrecognized", True))
def test_default_hard_projection_and_legacy_mode_fallback_are_preserved(route, mode):
    graph = _graph()
    graph.graph.update(EPI_MIN=Fraction(-1), EPI_MAX=Fraction(13, 16))
    if mode is not None:
        graph.graph["CLIP_MODE"] = mode

    # The unprojected convex blend is 7/8; the declared upper rail is 13/16.
    assert _execute(graph, route) == 13 / 16


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("sign", (-1, 1))
@pytest.mark.parametrize(
    "unused_stiffness", (math.log(2), True, math.nan, _TINY, "unused")
)
def test_soft_assignment_uses_default_stiffness_not_integrator_configuration(
    route, sign, unused_stiffness
):
    width = math.exp(-CORE_DEFAULTS["CLIP_SOFT_K"])
    graph = _graph(sign * (1.0 - width), float(sign))
    graph.graph.update(CLIP_MODE="soft", CLIP_SOFT_K=unused_stiffness)

    # The blend is halfway through the default knee. The Hermite ordinate at
    # t=1/2 is 5/8, so the distance from the signed rail is 3*width/8.
    expected = sign * (1.0 - 3.0 * width / 8.0)
    actual = _execute(graph, route)

    assert actual == pytest.approx(expected, rel=0, abs=4e-16)
    assert actual != sign * (1.0 - width / 2.0)


@pytest.mark.parametrize("route", ("kernel", "glyph"))
def test_default_hard_interior_keeps_the_existing_resonance_channels(route):
    graph = _graph()
    neighbor_before = deepcopy(dict(graph.nodes[1]))

    assert _execute(graph, route) == 7 / 8
    assert graph.nodes[0]["nu_f"] == 1.25
    assert graph.nodes[0]["theta"] == pytest.approx(0.125)
    assert graph.nodes[0]["delta_nfr"] == 0.125
    assert graph.nodes[0]["epi_kind"] == "wave"
    assert dict(graph.nodes[1]) == neighbor_before
