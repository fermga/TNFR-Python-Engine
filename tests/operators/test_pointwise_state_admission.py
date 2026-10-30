"""Pointwise state/proposal validation must precede nodal and lifecycle writes."""

from __future__ import annotations

import math
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
from tnfr.types import Glyph, real_scalar_epi

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


@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
def test_sha_product_underflow_rejects_before_lifecycle_and_state_writes(route):
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"SHA_vf_factor": 0.5}
    graph.nodes[0][ALIAS_VF[0]] = float.fromhex("0x0.0000000000001p-1022")
    graph.nodes[0].update(latent=False, preserved_epi=-0.25, silence_duration=3.0)
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(TNFRValueError, match="underflows to represented zero") as error:
        _execute(graph, "SHA", route)
    assert error.value.context["reason"] == "capacity_product_underflow"
    assert dict(graph.nodes[0]) == before


def test_late_target_sha_underflow_leaves_entire_stage_uncommitted():
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"SHA_vf_factor": 0.5}
    graph.add_node(1, **deepcopy(dict(graph.nodes[0])))
    graph.nodes[1][ALIAS_VF[0]] = float.fromhex("0x0.0000000000001p-1022")
    graph.add_edge(0, 1, weight=1)
    source = deepcopy(graph)
    with pytest.raises(TNFRValueError, match="underflows to represented zero"):
        execute_pointwise_stage(graph, Silence(), (0, 1))
    assert graph.graph == source.graph
    assert dict(graph.nodes) == dict(source.nodes)
    assert dict(graph.edges) == dict(source.edges)
    assert not hasattr(graph, "_last_operator_applied")


@pytest.mark.parametrize("glyph", ["VAL", "NUL"])
@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("EDGE_AWARE_ENABLED", "false"),
        ("EDGE_AWARE_ENABLED", 1),
        ("CLIP_MODE", "typo"),
        ("CLIP_MODE", False),
        ("CLIP_SOFT_K", 0.0),
        ("CLIP_SOFT_K", True),
        ("CLIP_SOFT_K", float("nan")),
        ("CLIP_SOFT_K", float("inf")),
    ],
)
def test_scale_rejects_invalid_active_configuration_before_primary_writes(
    glyph, route, key, value
):
    graph = _graph()
    graph.graph[key] = value
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises((TNFRValueError, OperatorPreconditionError)):
        _execute(graph, glyph, route)
    assert dict(graph.nodes[0]) == before
    assert "nul_densification_log" not in graph.graph
    assert "edge_aware_interventions" not in graph.graph


@pytest.mark.parametrize("glyph,factor", [("VAL", 1.01), ("NUL", 0.99)])
@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
def test_scale_consumes_configured_soft_k_in_every_execution_path(glyph, factor, route):
    graph = _graph()
    graph.nodes[0][ALIAS_EPI[0]] = 0.95
    graph.graph.update(
        CLIP_MODE="soft", CLIP_SOFT_K=0.4, GLYPH_FACTORS={f"{glyph}_scale": factor}
    )
    raw = 0.95 * factor
    knee = math.exp(-0.4)
    identity_limit = -math.expm1(-0.4)
    t = (raw - identity_limit) / knee
    expected = identity_limit + knee * (t + t * t - t**3)
    assert expected != pytest.approx(raw)

    _execute(graph, glyph, route)

    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) == pytest.approx(
        expected, rel=0.0, abs=2e-16
    )
    assert get_attr(graph.nodes[0], ALIAS_VF) == 2.0 * factor


@pytest.mark.parametrize("glyph", ["VAL", "NUL"])
def test_scale_certificate_replays_the_configured_soft_boundary(glyph):
    graph = _graph()
    graph.add_node(1, **deepcopy(dict(graph.nodes[0])))
    graph.add_edge(0, 1, weight=1.0)
    graph.nodes[0][ALIAS_EPI[0]] = 0.85
    graph.graph.update(CLIP_MODE="soft", CLIP_SOFT_K=0.4)
    result = execute_pointwise_stage(
        graph,
        _OPERATORS[glyph](),
        (0,),
        epi_jump_fixed_support_declared=True,
    )
    certificate = result.pointwise_epi_jump_certificate
    assert certificate is not None
    assert certificate.proposal_builder_replayed_from_declared_inputs
    assert certificate.runtime_proposed_state_after.tolist() == [
        real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI)) for node in graph
    ]
    assert not certificate.runtime_matches_represented_affine_exactly


@pytest.mark.parametrize("glyph", ["VAL", "NUL"])
@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
def test_inactive_scale_projection_does_not_admit_unused_settings(glyph, route):
    graph = _graph()
    graph.graph.update(
        EDGE_AWARE_ENABLED=False,
        EPI_MIN="unused",
        EPI_MAX=None,
        EDGE_AWARE_EPSILON="unused",
        CLIP_MODE="unused",
        CLIP_SOFT_K=float("nan"),
    )
    before = get_attr(graph.nodes[0], ALIAS_EPI)
    _execute(graph, glyph, route)
    assert real_scalar_epi(get_attr(graph.nodes[0], ALIAS_EPI)) == before


@pytest.mark.parametrize("route", ["public", "stage"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "0.1"])
@pytest.mark.parametrize(
    "glyph,key",
    [
        ("VAL", "VAL_MAX_VF"),
        ("VAL", "VAL_MIN_DNFR"),
        ("VAL", "VAL_MIN_EPI"),
        ("NUL", "NUL_MIN_VF"),
        ("NUL", "NUL_MIN_EPI"),
        ("NUL", "NUL_MAX_DENSITY"),
    ],
)
def test_scale_strict_policy_thresholds_cannot_fail_open(glyph, key, invalid, route):
    graph = _graph()
    graph.graph.update(VALIDATE_OPERATOR_PRECONDITIONS=True)
    graph.graph[key] = invalid
    before = deepcopy(dict(graph.nodes[0]))
    with pytest.raises(OperatorPreconditionError, match=key):
        _execute(graph, glyph, route)
    assert dict(graph.nodes[0]) == before
    assert "nul_densification_log" not in graph.graph


@pytest.mark.parametrize(
    "configuration",
    [
        {"EDGE_AWARE_ENABLED": "false"},
        {"CLIP_MODE": "typo"},
        {"CLIP_SOFT_K": 0.0},
        {"VALIDATE_OPERATOR_PRECONDITIONS": True, "VAL_MIN_DNFR": float("inf")},
        {
            "VALIDATE_OPERATOR_PRECONDITIONS": True,
            "VAL_CHECK_NETWORK_CAPACITY": "false",
        },
        {
            "VALIDATE_OPERATOR_PRECONDITIONS": True,
            "VAL_CHECK_NETWORK_CAPACITY": True,
            "VAL_MAX_NETWORK_SIZE": 1.5,
        },
    ],
)
def test_scale_configuration_rejection_leaves_complete_stage_uncommitted(configuration):
    graph = _graph()
    graph.graph.update(configuration)
    graph.add_node(1, **deepcopy(dict(graph.nodes[0])))
    graph.add_edge(0, 1, weight=1.0)
    source = deepcopy(graph)
    with pytest.raises((TNFRValueError, OperatorPreconditionError)):
        execute_pointwise_stage(graph, Expansion(), (0, 1))
    assert graph.graph == source.graph
    assert dict(graph.nodes) == dict(source.nodes)
    assert dict(graph.edges) == dict(source.edges)
    assert not hasattr(graph, "_last_operator_applied")


def test_strict_val_ignores_inactive_network_size_policy():
    graph = _graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        VAL_CHECK_NETWORK_CAPACITY=False,
        VAL_MAX_NETWORK_SIZE="unused",
    )
    Expansion()(graph, 0)
    assert get_attr(graph.nodes[0], ALIAS_VF) > 2.0
