"""Runtime factor gates fail before committing nodal state or metadata."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph, get_glyph_factors
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.types import real_scalar_epi


def _graph(*, factors=None, edge_aware=True) -> nx.Graph:
    graph = nx.Graph(
        GLYPH_FACTORS={} if factors is None else factors,
        EDGE_AWARE_ENABLED=edge_aware,
    )
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 2.0,
            ALIAS_DNFR[0]: 0.3,
            ALIAS_THETA[0]: 0.0,
            "epi_kind": "wave",
            "glyph_history": ["AL"],
        },
    )
    return graph


def _graph_metadata_without_adapter_cache(graph: nx.Graph) -> dict:
    """Return observable graph metadata, excluding NodeNX's transparent cache."""

    return {key: value for key, value in graph.graph.items() if key != "_node_cache"}


@pytest.mark.parametrize(
    ("glyph", "key", "value"),
    [
        ("AL", "AL_boost", -0.1),
        ("IL", "IL_dnfr_factor", 1.1),
        ("OZ", "OZ_dnfr_factor", 0.9),
        ("SHA", "SHA_vf_factor", 1.1),
        ("VAL", "VAL_scale", 0.9),
        ("NUL", "NUL_scale", 0.0),
        ("THOL", "THOL_accel", -0.1),
        ("NAV", "NAV_eta", 1.1),
    ],
)
def test_used_invalid_factor_rejects_before_any_mutation(glyph, key, value):
    graph = _graph(factors={key: value})
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(GlyphFactorValidationError, match=key):
        apply_glyph(graph, 0, glyph)

    assert graph.nodes[0] == node_before
    assert graph.graph == graph_before


def test_invalid_unrelated_override_does_not_block_current_operator():
    graph = _graph(factors={"RA_epi_diff": 2.0})

    apply_glyph(graph, 0, "AL")

    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) > 0.4
    assert list(graph.nodes[0]["glyph_history"])[-1] == "AL"


def test_context_free_factor_view_keeps_nul_inverse_relation():
    graph = _graph(factors={"NUL_scale": 0.5})

    factors = get_glyph_factors(type("Node", (), {"graph": graph.graph})())

    assert factors["NUL_scale"] == 0.5
    assert factors["NUL_densification_factor"] == 2.0


def test_nul_derives_pressure_densification_and_records_binary64_residual():
    graph = _graph(factors={"NUL_scale": 0.5})

    apply_glyph(graph, 0, "NUL")

    assert graph.nodes[0][ALIAS_VF[0]] == 1.0
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.6)
    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) == pytest.approx(0.2)
    event = graph.graph["nul_densification_log"][-1]
    assert event["densification_factor"] == 2.0
    assert event["derived_inverse_coefficient"] is True
    assert event["binary64_inverse_product_residual"] == 0.0


def test_inconsistent_explicit_nul_pair_rejects_atomically():
    graph = _graph(
        factors={"NUL_scale": 0.5, "NUL_densification_factor": 3.0}
    )
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(GlyphFactorValidationError, match="derived, not independent"):
        apply_glyph(graph, 0, "NUL")

    assert graph.nodes[0] == node_before
    assert graph.graph == graph_before


def test_val_frequency_overflow_rejects_before_epi_or_history_changes():
    graph = _graph(factors={"VAL_scale": 2.0})
    graph.nodes[0][ALIAS_VF[0]] = 1e308
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(TNFRValueError, match="nu_f proposal"):
        apply_glyph(graph, 0, "VAL")

    assert graph.nodes[0] == node_before
    assert _graph_metadata_without_adapter_cache(graph) == graph_before


def test_nul_pressure_overflow_rejects_before_frequency_or_log_changes():
    graph = _graph(factors={"NUL_scale": 0.5})
    graph.nodes[0][ALIAS_DNFR[0]] = 1e308
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(TNFRValueError, match="DeltaNFR proposal"):
        apply_glyph(graph, 0, "NUL")

    assert graph.nodes[0] == node_before
    assert _graph_metadata_without_adapter_cache(graph) == graph_before


def test_disabled_edge_adaptation_ignores_its_unused_invalid_epsilon():
    graph = _graph(factors={"VAL_scale": 1.1}, edge_aware=False)
    graph.graph["EDGE_AWARE_EPSILON"] = "unused-invalid-value"

    apply_glyph(graph, 0, "VAL")

    assert graph.nodes[0][ALIAS_VF[0]] == pytest.approx(2.2)
    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) == pytest.approx(0.4)
