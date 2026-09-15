"""Primitive compatibility and shared prepared-dispatch boundaries for THOL."""

from copy import deepcopy
from decimal import Decimal

import networkx as nx
import pytest

import tnfr.operators as operators
from tnfr.errors import TNFRValueError
from tnfr.operators._reception_kernel import capture_reception_read_snapshot
from tnfr.types import Glyph, real_scalar_epi


class _FloatScalar:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value


class _ProtocolNode:
    def __init__(self, pressure, acceleration):
        self.dnfr = pressure
        self.d2EPI = acceleration
        self.graph = {"GLYPH_FACTORS": {"THOL_accel": 0.25}}
        self.storage = {"glyph_history": ["IL", "OZ"]}

    def _glyph_storage(self):
        return self.storage


@pytest.mark.parametrize("route", ["primitive", "protocol_dispatch"])
@pytest.mark.parametrize("scalar", [Decimal, str, _FloatScalar])
def test_graphless_thol_preserves_finite_float_conversion_compatibility(route, scalar):
    pressure, acceleration = scalar(0.5), scalar(-2.0)
    node = _ProtocolNode(pressure, acceleration)
    if route == "primitive":
        operators._op_THOL(node, {"THOL_accel": 0.25})
        assert node.storage == {"glyph_history": ["IL", "OZ"]}
    else:
        operators.apply_glyph_obj(node, "THOL", window=8)
        assert tuple(node.storage["glyph_history"]) == ("IL", "OZ", "THOL")
    assert node.dnfr == 0.0
    assert node.d2EPI is acceleration


@pytest.mark.parametrize("route", ["primitive", "protocol_dispatch"])
@pytest.mark.parametrize("value", [Decimal("NaN"), "nan", _FloatScalar(float("inf"))])
def test_graphless_thol_retains_numeric_error_family_before_writes(route, value):
    node = _ProtocolNode(0.5, value)
    before = deepcopy(node.storage)
    with pytest.raises(TNFRValueError, match="THOL d2EPI state"):
        if route == "primitive":
            operators._op_THOL(node, {"THOL_accel": 0.25})
        else:
            operators.apply_glyph_obj(node, "THOL", window=8)
    assert node.dnfr == 0.5
    assert node.d2EPI is value
    assert node.storage == before


def test_prepared_reception_sources_survive_subsequent_thol_dispatch():
    graph = nx.path_graph(2)
    graph.graph["GLYPH_FACTORS"] = {"EN_mix": 0.5, "THOL_accel": 0.25}
    for node, epi in enumerate((0.0, 0.5)):
        graph.nodes[node].update(
            EPI=epi, nu_f=1.0, theta=0.0, delta_nfr=0.25,
            glyph_history=["AL"], EPI_kind="seed",
        )
    snapshot = capture_reception_read_snapshot(graph, 0, track_sources=True)
    assert snapshot.reception_sources is not None

    operators._apply_prepared_reception_glyph(
        graph, 0, Glyph.EN, window=8, prepared_state=snapshot,
    )
    assert real_scalar_epi(graph.nodes[0]["EPI"]) == 0.25
    assert graph.nodes[0]["_reception_sources"] == list(snapshot.reception_sources)
    recorded_sources = deepcopy(graph.nodes[0]["_reception_sources"])

    # These are two primitive calls, not a claim of complete-word admission.
    operators.apply_glyph(graph, 0, Glyph.THOL, window=8)
    assert real_scalar_epi(graph.nodes[0]["EPI"]) == 0.25
    assert graph.nodes[0]["delta_nfr"] == 0.25
    assert graph.nodes[0]["_reception_sources"] == recorded_sources
    assert tuple(graph.nodes[0]["glyph_history"]) == ("AL", "EN", "THOL")
