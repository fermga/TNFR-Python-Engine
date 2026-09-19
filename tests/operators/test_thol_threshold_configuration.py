"""Public THOL and metabolism share one strict threshold precedence."""

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.dynamics.metabolism import _configured_tau as metabolic_tau
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.self_organization import _configured_tau as public_tau


@pytest.mark.parametrize(
    "settings,requested,expected",
    (
        ({}, None, CORE_DEFAULTS["THOL_BIFURCATION_THRESHOLD"]),
        ({"THOL_BIFURCATION_THRESHOLD": 0.4}, None, 0.4),
        (
            {"BIFURCATION_THRESHOLD_TAU": 0.3, "THOL_BIFURCATION_THRESHOLD": 0.4},
            None,
            0.3,
        ),
        (
            {"BIFURCATION_THRESHOLD_TAU": None, "THOL_BIFURCATION_THRESHOLD": 0.4},
            None,
            0.4,
        ),
        (
            {"BIFURCATION_THRESHOLD_TAU": 0.3, "THOL_BIFURCATION_THRESHOLD": 0.4},
            0.2,
            0.2,
        ),
        ({"BIFURCATION_THRESHOLD_TAU": 0.3}, 0.0, 0.0),
        ({"THOL_BIFURCATION_THRESHOLD": 0.0}, None, 0.0),
        (
            {
                "BIFURCATION_THRESHOLD_TAU": math.nan,
                "THOL_BIFURCATION_THRESHOLD": "unused",
            },
            0.2,
            0.2,
        ),
        (
            {"BIFURCATION_THRESHOLD_TAU": 0.3, "THOL_BIFURCATION_THRESHOLD": math.nan},
            None,
            0.3,
        ),
        # Irrelevant policies never substitute for the THOL-specific default.
        (
            {"ZHIR_BIFURCATION_THRESHOLD": 0.8, "OZ_BIFURCATION_THRESHOLD": 0.9},
            None,
            0.1,
        ),
    ),
)
def test_precedence_and_zero_match_all_callers_without_configuration_writes(
    settings,
    requested,
    expected,
):
    graph = nx.Graph(**deepcopy(settings))
    before = deepcopy(graph.graph)
    assert resolve_thol_bifurcation_threshold(graph.graph, requested) == expected
    assert public_tau(graph.graph, {"tau": requested}) == expected
    assert metabolic_tau(graph, requested) == expected
    assert graph.graph == before


@pytest.mark.parametrize(
    "value", (True, False, "0.1", -0.1, math.inf, -math.inf, math.nan)
)
@pytest.mark.parametrize(
    "location", ("requested", "BIFURCATION_THRESHOLD_TAU", "THOL_BIFURCATION_THRESHOLD")
)
def test_invalid_active_threshold_never_silently_falls_through(value, location):
    settings = {} if location == "requested" else {location: value}
    requested = value if location == "requested" else None
    graph = nx.Graph(**settings)
    calls = (
        lambda: resolve_thol_bifurcation_threshold(graph.graph, requested),
        lambda: public_tau(graph.graph, {"tau": requested}),
        lambda: metabolic_tau(graph, requested),
    )
    for call in calls:
        with pytest.raises(OperatorPreconditionError, match="tau"):
            call()
    assert graph.graph.keys() == settings.keys()


def test_present_invalid_thol_alias_is_not_the_same_as_missing_override():
    graph = nx.Graph(THOL_BIFURCATION_THRESHOLD=None)
    with pytest.raises(OperatorPreconditionError, match="tau"):
        metabolic_tau(graph, None)
    assert public_tau(graph.graph, {"tau": 0.25}) == 0.25
    assert resolve_thol_bifurcation_threshold(graph.graph, 0.25) == 0.25


def test_shared_validation_preserves_callers_error_context():
    with pytest.raises(OperatorPreconditionError) as error:
        metabolic_tau(nx.Graph(), math.nan)
    assert error.value.operator == "Structural metabolism"
    with pytest.raises(OperatorPreconditionError) as error:
        public_tau({}, {"tau": math.nan})
    assert error.value.operator == "Self-organization"
