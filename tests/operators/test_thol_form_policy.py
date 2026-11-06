"""Detached form admission and the strict THOL gate share configured policy."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.operators._thol_config import resolve_thol_min_epi
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)


def _graph(epi, **settings):
    graph = nx.Graph(**settings)
    graph.add_node(
        "child",
        **{
            ALIAS_EPI[0]: epi,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_VF[0]: 1.0,
            "epi_time_history": [(0.0, epi), (1.0, epi), (2.0, epi)],
            "glyph_history": [],
        },
    )
    graph.add_node("parent")
    graph.add_edge("child", "parent")
    return graph


@pytest.mark.parametrize(
    "settings,expected",
    [({}, 0.2), ({"THOL_MIN_EPI": 0.0}, 0.0), ({"THOL_MIN_EPI": Fraction(1, 4)}, 0.25)],
)
def test_resolver_admits_default_zero_and_represented_override_without_writes(
    settings, expected
):
    before = deepcopy(settings)
    assert resolve_thol_min_epi(settings) == expected
    assert settings == before


@pytest.mark.parametrize(
    "value", [None, True, False, "0.2", -0.1, math.inf, -math.inf, math.nan]
)
def test_invalid_present_policy_rejects_in_both_consumers_without_fallback(value):
    graph = _graph(0.5, THOL_MIN_EPI=value)
    for read in (
        lambda: resolve_thol_min_epi(graph.graph),
        lambda: validate_self_organization_strict(graph, "child", emit_warnings=False),
    ):
        with pytest.raises(OperatorPreconditionError, match="THOL_MIN_EPI") as error:
            read()
        assert error.value.operator == "Self-organization"
    assert graph.graph["THOL_MIN_EPI"] is value


@pytest.mark.parametrize(
    "settings,minimum", [({}, 0.2), ({"THOL_MIN_EPI": 0.25}, 0.25)]
)
def test_strict_gate_accepts_boundary_and_rejects_adjacent_lower_form(
    settings, minimum
):
    admitted = _graph(minimum, **settings)
    before = deepcopy(dict(admitted.nodes["child"]))
    validate_self_organization_strict(admitted, "child", emit_warnings=False)
    assert dict(admitted.nodes["child"]) == before

    rejected = _graph(math.nextafter(minimum, -math.inf), **settings)
    with pytest.raises(OperatorPreconditionError, match="EPI too low"):
        validate_self_organization_strict(rejected, "child", emit_warnings=False)


def test_lower_form_policy_does_not_supply_missing_temporal_evidence():
    graph = _graph(0.15, THOL_MIN_EPI=0.1)
    graph.nodes["child"]["epi_time_history"] = [(0.0, 0.15)]
    with pytest.raises(OperatorPreconditionError, match="1 samples; 3 required"):
        validate_self_organization_strict(graph, "child", emit_warnings=False)
