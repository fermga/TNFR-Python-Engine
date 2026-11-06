"""Scalar maximum caches must account for untouched nodes on their first write."""

import math
from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr_with_max, set_dnfr, set_vf
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import TWO_PHASE_JACOBI
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.types import ensure_bepi, serialize_bepi, serialize_bepi_json


@pytest.mark.parametrize(
    "setter,aliases,key,untouched,updated,expected",
    [
        (set_vf, ALIAS_VF, "_vfmax", 7.0, 0.5, 7.0),
        (set_dnfr, ALIAS_DNFR, "_dnfrmax", -7.0, 0.5, 7.0),
        (set_vf, ALIAS_VF, "_vfmax", 0.0, 0.0, 0.0),
        (set_dnfr, ALIAS_DNFR, "_dnfrmax", -0.0, 0.0, 0.0),
    ],
)
@pytest.mark.parametrize("cache_state", ["absent", "no_witness", "dead_witness"])
def test_first_write_rebuilds_global_maximum(
    setter, aliases, key, untouched, updated, expected, cache_state
):
    graph = nx.Graph()
    graph.add_node("untouched", **{aliases[0]: untouched})
    graph.add_node("target", **{aliases[0]: 1.0})
    if cache_state != "absent":
        graph.graph[key] = 99.0
    if cache_state == "dead_witness":
        graph.graph[f"{key}_node"] = "removed"
    result = setter(graph, "target", updated)
    assert result.max_value == expected
    assert graph.graph[key] == expected
    witness = graph.graph[f"{key}_node"]
    assert abs(get_attr(graph.nodes[witness], aliases)) == expected
    assert graph.nodes["untouched"][aliases[0]] == untouched


def test_warm_cache_updates_and_rebuilds_after_maximum_decreases():
    graph = nx.Graph()
    graph.add_node("first", **{ALIAS_DNFR[0]: -4.0})
    graph.add_node("second", **{ALIAS_DNFR[0]: 1.0})
    set_dnfr(graph, "second", 2.0)
    assert (graph.graph["_dnfrmax"], graph.graph["_dnfrmax_node"]) == (4.0, "first")
    set_dnfr(graph, "second", -5.0)
    assert (graph.graph["_dnfrmax"], graph.graph["_dnfrmax_node"]) == (5.0, "second")
    set_dnfr(graph, "second", 0.0)
    assert (graph.graph["_dnfrmax"], graph.graph["_dnfrmax_node"]) == (4.0, "first")


@pytest.mark.parametrize(
    "invalid", [True, math.nan, math.inf, "bad", Fraction(1, 10**400)]
)
def test_cold_rebuild_does_not_replace_invalid_authoritative_state(invalid):
    graph = nx.Graph()
    graph.add_node("invalid", **{ALIAS_VF[0]: invalid, ALIAS_VF[1]: 7.0})
    graph.add_node("target", **{ALIAS_VF[0]: 1.0})
    with pytest.raises((TypeError, ValueError)):
        set_vf(graph, "target", 0.5)
    # The generic setter has no transaction contract; the failed reduction
    # nevertheless must not publish a fabricated maximum or alias fallback.
    assert "_vfmax" not in graph.graph
    assert "_vfmax_node" not in graph.graph


@pytest.mark.parametrize(
    "representation", [ensure_bepi, serialize_bepi, serialize_bepi_json]
)
def test_generic_epi_maximum_retains_signed_scalar_bepi(representation):
    graph = nx.Graph()
    graph.add_node("signed", **{ALIAS_EPI[0]: representation(-7.0)})
    graph.add_node("target", **{ALIAS_EPI[0]: 1.0})
    result = set_attr_with_max(graph, "target", ALIAS_EPI, 0.5, cache="epi_max")
    assert result.max_value == 7.0
    assert result.node == "signed"
    assert get_attr(graph.nodes["signed"], ALIAS_EPI) == -7.0


@pytest.mark.parametrize(
    "invalid",
    [
        BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0)),
        BEPIElement((3.0 + 4.0j,) * 2, (3.0 + 4.0j,) * 2, (0.0, 1.0)),
    ],
)
@pytest.mark.parametrize("serialize", [False, True])
def test_generic_epi_maximum_rejects_magnitude_only_bepi(invalid, serialize):
    graph = nx.Graph()
    graph.add_node(
        "invalid", **{ALIAS_EPI[0]: serialize_bepi(invalid) if serialize else invalid}
    )
    graph.add_node("target", **{ALIAS_EPI[0]: 1.0})
    with pytest.raises(ValueError, match="finite uniform-real EPI"):
        set_attr_with_max(graph, "target", ALIAS_EPI, 0.5, cache="epi_max")
    assert "epi_max" not in graph.graph


def _doubled_cycle():
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    for block in range(5):
        for member in (0, 1):
            for neighbor in (0, 1):
                graph.add_edge(2 * block + member, 2 * ((block + 1) % 5) + neighbor)
    graph.graph.update(
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EPI_LATENT_MAX=1.0,
        VF_BASAL_THRESHOLD=0.125,
        SHA_MIN_VF=0.125,
        GLYPH_FACTORS={"AL_boost": 0.0625, "SHA_vf_factor": 0.5},
    )
    for node in graph:
        graph.nodes[node].update(
            {
                ALIAS_EPI[0]: 0.25,
                ALIAS_THETA[0]: 0.0,
                ALIAS_VF[0]: 1.0 + (node // 2) / 4,
                ALIAS_DNFR[0]: 0.0,
                "glyph_history": [],
            }
        )
    return graph


@pytest.mark.parametrize("targets", [(0, 1), (1, 0)])
def test_selected_pair_sha_cache_includes_untargeted_capacities(targets):
    graph = _doubled_cycle()
    source = deepcopy(dict(graph.nodes))
    operators = (Emission(), Silence())
    sequence = ValidatedSequence(operators, context={"compatibility_profile": "core"})
    for index, operator in enumerate(operators):
        result = execute_network_operator_stage(
            graph, operator, targets, sequence_context=sequence.step(index)
        )
        assert result.schedule == TWO_PHASE_JACOBI
    assert graph.graph["_vfmax"] == 2.0
    witness = graph.graph["_vfmax_node"]
    assert get_attr(graph.nodes[witness], ALIAS_VF) == 2.0
    for node in targets:
        assert get_attr(graph.nodes[node], ALIAS_VF) == 0.5
    for node in set(graph) - set(targets):
        assert graph.nodes[node] == source[node]


def test_selected_pair_sha_rolls_back_invalid_untargeted_cache_input():
    graph = _doubled_cycle()
    graph.nodes[8][ALIAS_VF[0]] = math.inf
    source = deepcopy((dict(graph.nodes), dict(graph.graph), dict(graph.edges)))
    with pytest.raises(ValueError, match="Non-finite"):
        execute_network_operator_stage(graph, Silence(), (0, 1))
    assert (dict(graph.nodes), dict(graph.graph), dict(graph.edges)) == source
