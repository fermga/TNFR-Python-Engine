"""AL/SHA guard admission before public lifecycle writes or stage commits."""

import math
import warnings
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import Emission, Silence
from tnfr.operators.network_stage import execute_pointwise_stage
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_emission,
    validate_silence,
)
from tnfr.operators.preconditions.emission import validate_emission_strict

INVALID_SCALARS = (
    True,
    False,
    float("nan"),
    float("inf"),
    -float("inf"),
    "0.5",
    Fraction(1, 10**400),
    Fraction(-1, 10**400),
    Fraction(10**400),
)


def _graph():
    graph = nx.path_graph(2)
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
    )
    for node in graph:
        graph.nodes[node].update(
            EPI=0.25,
            nu_f=0.5,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=["AL", "IL"],
        )
    return graph


def _logical_state(graph):
    return deepcopy((dict(graph.nodes(data=True)), dict(graph.graph)))


def _invoke(graph, operator, path):
    if path == "guard":
        guard = validate_silence if operator is Silence else validate_emission_strict
        guard(graph, 0)
    elif path == "public":
        operator()(graph, 0)
    else:
        execute_pointwise_stage(
            graph,
            operator(),
            (0, 1),
            timestamp="2030-01-02T03:04:05+00:00",
        )


@pytest.mark.parametrize("path", ("guard", "public", "atomic"))
@pytest.mark.parametrize(
    "threshold,operator,error",
    (
        ("EPI_LATENT_MAX", Emission, TNFRValueError),
        ("VF_BASAL_THRESHOLD", Emission, TNFRValueError),
        ("SHA_MIN_VF", Silence, OperatorPreconditionError),
    ),
)
@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_invalid_policy_rejects_before_lifecycle_or_any_commit(
    path, threshold, operator, error, value
):
    graph = _graph()
    graph.graph[threshold] = value
    before = _logical_state(graph)
    with pytest.raises(error, match=threshold):
        _invoke(graph, operator, path)
    assert _logical_state(graph) == before
    assert "_node_cache" not in graph.graph
    assert "_last_operator_applied" not in graph.__dict__


@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_legacy_emission_keeps_its_error_type_and_admits_its_own_policy(value):
    graph = _graph()
    graph.graph["AL_MAX_EPI_FOR_EMISSION"] = value
    before = _logical_state(graph)
    with pytest.raises(OperatorPreconditionError, match="AL_MAX_EPI_FOR_EMISSION"):
        validate_emission(graph, 0)
    assert _logical_state(graph) == before


@pytest.mark.parametrize(
    "guard,aliases,error",
    (
        (validate_emission_strict, ALIAS_EPI, TNFRValueError),
        (validate_emission_strict, ALIAS_VF, TNFRValueError),
        (validate_emission, ALIAS_EPI, OperatorPreconditionError),
        (validate_silence, ALIAS_VF, OperatorPreconditionError),
    ),
)
@pytest.mark.parametrize("value", INVALID_SCALARS)
def test_authoritative_invalid_state_cannot_fall_through(guard, aliases, error, value):
    graph = _graph()
    graph.nodes[0][aliases[0]] = value
    graph.nodes[0][aliases[1]] = 0.25
    before = _logical_state(graph)
    with pytest.raises(error):
        guard(graph, 0)
    assert _logical_state(graph) == before


@pytest.mark.parametrize(
    "operator,error", ((Emission, TNFRValueError), (Silence, OperatorPreconditionError))
)
def test_invalid_later_target_rejects_the_complete_atomic_stage(operator, error):
    graph = _graph()
    graph.nodes[1][ALIAS_VF[0]] = True
    graph.nodes[1][ALIAS_VF[1]] = 0.5
    before = _logical_state(graph)
    with pytest.raises(error):
        _invoke(graph, operator, "atomic")
    assert _logical_state(graph) == before
    assert "_last_operator_applied" not in graph.__dict__


@pytest.mark.parametrize("path", ("guard", "public", "atomic"))
@pytest.mark.parametrize("value", (True, False, -1, 1.5, 1.0, "1", Fraction(1, 2)))
def test_degree_policy_is_a_nonnegative_integer(path, value):
    graph = _graph()
    graph.graph["MIN_NETWORK_DEGREE_COUPLING"] = value
    before = _logical_state(graph)
    with pytest.raises(TNFRValueError, match="MIN_NETWORK_DEGREE_COUPLING"):
        _invoke(graph, Emission, path)
    assert _logical_state(graph) == before


@pytest.mark.parametrize("guard", (validate_emission_strict, validate_emission))
def test_emission_keeps_uniform_real_bepi_signed(guard):
    graph = _graph()
    form = BEPIElement((-0.5, -0.5), (-0.5, -0.5), (0.0, 1.0))
    graph.nodes[0][ALIAS_EPI[0]] = form
    graph.graph.update(EPI_LATENT_MAX=-0.25, AL_MAX_EPI_FOR_EMISSION=-0.25)
    guard(graph, 0)
    assert graph.nodes[0][ALIAS_EPI[0]] is form


@pytest.mark.parametrize(
    "guard,error",
    (
        (validate_emission_strict, TNFRValueError),
        (validate_emission, OperatorPreconditionError),
    ),
)
def test_emission_cannot_project_nonuniform_form_to_a_magnitude(guard, error):
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = BEPIElement((-0.5, -0.4), (-0.5, -0.5), (0.0, 1.0))
    with pytest.raises(error):
        guard(graph, 0)


def test_strict_and_legacy_emission_keep_distinct_ceiling_policies():
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.6
    graph.graph.update(EPI_LATENT_MAX=0.5, AL_MAX_EPI_FOR_EMISSION=0.8)
    validate_emission(graph, 0)
    with pytest.raises(TNFRValueError, match="EPI=0.600"):
        validate_emission_strict(graph, 0)


@pytest.mark.parametrize(
    "guard,threshold,error",
    (
        (validate_emission_strict, "EPI_LATENT_MAX", TNFRValueError),
        (validate_emission, "AL_MAX_EPI_FOR_EMISSION", OperatorPreconditionError),
    ),
)
def test_emission_ceiling_is_still_strict(guard, threshold, error):
    graph = _graph()
    graph.graph[threshold] = 0.25
    with pytest.raises(error):
        guard(graph, 0)
    graph.nodes[0][ALIAS_EPI[0]] = math.nextafter(0.25, -math.inf)
    guard(graph, 0)


@pytest.mark.parametrize("floor", (0.0, 0.5, -0.25))
def test_capacity_minima_remain_inclusive_with_finite_signed_policies(floor):
    graph = _graph()
    graph.graph.update(VF_BASAL_THRESHOLD=floor, SHA_MIN_VF=floor)
    graph.nodes[0][ALIAS_VF[0]] = max(0.0, floor)
    validate_emission_strict(graph, 0)
    validate_silence(graph, 0)
    graph.nodes[0][ALIAS_VF[0]] = -0.125
    with pytest.raises(TNFRValueError, match="nonnegative"):
        validate_emission_strict(graph, 0)
    with pytest.raises(OperatorPreconditionError, match="must be >="):
        validate_silence(graph, 0)


@pytest.mark.parametrize(
    "size,minimum,warns", ((2, 1, True), (1, 1, False), (2, 0, False))
)
def test_missing_connectivity_remains_an_advisory(size, minimum, warns):
    graph = nx.empty_graph(size)
    graph.graph["MIN_NETWORK_DEGREE_COUPLING"] = minimum
    graph.nodes[0].update(EPI=0.0, nu_f=1.0)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        validate_emission_strict(graph, 0)
    assert len(recorded) == int(warns)
    if warns:
        assert recorded[0].category is UserWarning
        assert "Emission is possible" in str(recorded[0].message)
