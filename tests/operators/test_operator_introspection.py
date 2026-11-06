"""Public metadata follows its owners without enlarging operator effects."""

from dataclasses import fields

import networkx as nx
import pytest

from tnfr.operators import apply_glyph
from tnfr.operators.grammar_canon import u_rules_for_operator
from tnfr.operators.introspection import (
    OperatorMeta,
    get_operator_meta,
    iter_operator_meta,
)
from tnfr.operators.operator_contracts import iter_contracts
from tnfr.types import require_finite_real_scalar_epi


def test_introspection_preserves_public_shape_categories_and_order():
    assert tuple(field.name for field in fields(OperatorMeta)) == (
        "name",
        "mnemonic",
        "category",
        "grammar_roles",
        "contracts",
        "doc",
    )
    assert tuple((meta.mnemonic, meta.category) for meta in iter_operator_meta()) == (
        ("AL", "generator"),
        ("EN", "integrator"),
        ("IL", "stabilizer"),
        ("OZ", "destabilizer"),
        ("UM", "coupling"),
        ("RA", "propagation"),
        ("SHA", "closure"),
        ("VAL", "destabilizer"),
        ("NUL", "simplifier"),
        ("THOL", "stabilizer"),
        ("ZHIR", "transformer"),
        ("NAV", "generator"),
        ("REMESH", "generator"),
    )


def test_introspection_shared_fields_resolve_to_contract_and_role_owners():
    for contract in iter_contracts():
        meta = get_operator_meta(contract.glyph)
        assert get_operator_meta(contract.english_name) is meta
        assert meta.name == contract.english_name
        assert meta.contracts == (contract.postcondition,)
        assert meta.doc == contract.purpose
        assert meta.grammar_roles == u_rules_for_operator(contract.name)
    with pytest.raises(KeyError):
        get_operator_meta("unknown")


@pytest.mark.parametrize(
    ("glyph", "factor", "epi_after", "capacity_after", "pressure_after"),
    [
        ("VAL", {"VAL_scale": 2.0}, 1.0, 2.0, 0.25),
        ("NUL", {"NUL_scale": 0.5}, 0.25, 0.5, 0.5),
        ("SHA", {"SHA_vf_factor": 0.25}, 0.5, 0.25, 0.25),
    ],
)
def test_capacity_events_preserve_support_and_do_not_freeze_nonzero_nodal_product(
    glyph, factor, epi_after, capacity_after, pressure_after
):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=0.5, nu_f=1.0, theta=0.0, delta_nfr=0.25)
    graph.graph["GLYPH_FACTORS"] = factor

    apply_glyph(graph, 0, glyph)

    assert tuple(graph) == (0, 1)
    assert tuple(graph.edges) == ((0, 1),)
    assert require_finite_real_scalar_epi(graph.nodes[0]["EPI"]) == epi_after
    assert graph.nodes[0]["nu_f"] == capacity_after
    assert graph.nodes[0]["delta_nfr"] == pressure_after
    assert graph.nodes[0]["nu_f"] * graph.nodes[0]["delta_nfr"] > 0.0
    if glyph == "NUL":
        assert graph.nodes[0]["nu_f"] * graph.nodes[0]["delta_nfr"] == 0.25
