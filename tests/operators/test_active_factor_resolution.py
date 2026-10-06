"""Runtime factor validation follows the concrete active operator branch."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.operators import apply_glyph
from tnfr.operators.definitions import (
    Coherence,
    Dissonance,
    Emission,
    Reception,
    Silence,
    Transition,
)
from tnfr.operators.factor_contracts import (
    GlyphFactorValidationError,
    resolve_runtime_operator_factors,
    runtime_active_glyph_factor_keys,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.network_stage import (
    execute_dissonance_stage,
    execute_pointwise_stage,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import Glyph


@pytest.mark.parametrize(
    ("glyph", "graph_data", "overrides", "inactive_key"),
    [
        (
            Glyph.OZ,
            {"OZ_NOISE_MODE": True},
            {"OZ_dnfr_factor": "unused"},
            "OZ_dnfr_factor",
        ),
        (
            Glyph.ZHIR,
            {},
            {"ZHIR_theta_shift": 0.25, "ZHIR_theta_shift_factor": "unused"},
            "ZHIR_theta_shift_factor",
        ),
        (Glyph.NAV, {"NAV_STRICT": True}, {"NAV_eta": "unused"}, "NAV_eta"),
        (Glyph.REMESH, {}, {"REMESH_alpha": "unused"}, "REMESH_alpha"),
        (
            Glyph.UM,
            {"UM_SYNC_VF": False, "UM_STABILIZE_DNFR": False},
            {"UM_vf_sync": "unused", "UM_dnfr_reduction": "unused"},
            "UM_vf_sync",
        ),
    ],
)
def test_disabled_factor_channels_do_not_block_the_active_branch(
    glyph, graph_data, overrides, inactive_key
):
    active = runtime_active_glyph_factor_keys(glyph, graph_data, overrides)
    resolved = resolve_runtime_operator_factors(overrides, glyph, graph_data)

    assert inactive_key not in active
    assert resolved[inactive_key] == overrides[inactive_key]


@pytest.mark.parametrize(
    ("glyph", "graph_data", "overrides", "key"),
    [
        (Glyph.OZ, {"OZ_NOISE_MODE": False}, {"OZ_dnfr_factor": 1.0}, "OZ_dnfr_factor"),
        (Glyph.ZHIR, {}, {"ZHIR_theta_shift_factor": 0.0}, "ZHIR_theta_shift_factor"),
        (Glyph.NAV, {"NAV_STRICT": False}, {"NAV_eta": 2.0}, "NAV_eta"),
        (Glyph.UM, {"UM_SYNC_VF": True}, {"UM_vf_sync": 2.0}, "UM_vf_sync"),
    ],
)
def test_active_factor_channels_keep_their_hard_domain(
    glyph, graph_data, overrides, key
):
    with pytest.raises(GlyphFactorValidationError, match=key):
        resolve_runtime_operator_factors(overrides, glyph, graph_data)


def _public_graph(factors) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph["GLYPH_FACTORS"] = factors
    for index in graph:
        graph.nodes[index].update(
            EPI=0.4,
            nu_f=1.0,
            delta_nfr=0.2,
            theta=0.1 * index,
            epi_kind="wave",
            glyph_history=["AL", "IL"],
        )
    return graph


@pytest.mark.parametrize(
    ("operator", "factor"),
    [
        (Emission(), {"AL_boost": 0.0}),
        (Reception(), {"EN_mix": 0.0}),
        (Silence(), {"SHA_vf_factor": 1.0}),
        (Transition(), {"NAV_jitter": -0.1}),
    ],
)
def test_public_operator_factor_preflight_precedes_subclass_metadata(operator, factor):
    graph = _public_graph(factor)
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(GlyphFactorValidationError):
        operator(graph, 0)

    assert graph.nodes[0] == node_before
    assert graph.graph == graph_before


@pytest.mark.parametrize("disabled", [False, "false", "off"])
def test_coupling_string_flags_disable_their_factor_domains(disabled):
    graph_data = {"UM_SYNC_VF": disabled, "UM_STABILIZE_DNFR": disabled}
    overrides = {"UM_vf_sync": "unused", "UM_dnfr_reduction": "unused"}

    active = runtime_active_glyph_factor_keys(Glyph.UM, graph_data, overrides)
    resolved = resolve_runtime_operator_factors(overrides, Glyph.UM, graph_data)

    assert "UM_vf_sync" not in active
    assert "UM_dnfr_reduction" not in active
    assert resolved["UM_vf_sync"] == resolved["UM_dnfr_reduction"] == "unused"


@pytest.mark.parametrize(
    "key",
    ["UM_BIDIRECTIONAL", "UM_SYNC_VF", "UM_STABILIZE_DNFR", "UM_FUNCTIONAL_LINKS"],
)
def test_coupling_invalid_flags_reject_before_factor_selection(key):
    with pytest.raises(GlyphFactorValidationError, match=key):
        resolve_runtime_operator_factors(
            {"UM_theta_push": "also invalid"}, Glyph.UM, {key: "unknown"}
        )


def _apply_branch_probe(graph, glyph, route):
    if route == "glyph":
        apply_glyph(graph, 0, glyph)
    elif glyph == "OZ":
        context = ValidatedSequence(
            [Emission(), Dissonance(), Coherence(), Silence()]
        ).step(1)
        if route == "public":
            Dissonance()(graph, 0, sequence_context=context)
        else:
            execute_dissonance_stage(
                graph, Dissonance(), (0,), sequence_context=context
            )
    elif route == "public":
        Transition()(graph, 0)
    else:
        execute_pointwise_stage(graph, Transition(), (0,))


@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
@pytest.mark.parametrize("disabled", [False, "false", "off", None, 0])
def test_oz_disabled_noise_uses_the_declared_amplification(route, disabled):
    graph = _public_graph({"OZ_dnfr_factor": 2.0})
    graph.nodes[0]["delta_nfr"] = 0.25
    graph.graph.update(
        OZ_NOISE_MODE=disabled, OZ_SIGMA=0.0, OZ_ENABLE_PROPAGATION=False
    )
    noise = deepcopy(graph)
    noise.graph["OZ_NOISE_MODE"] = True

    _apply_branch_probe(graph, "OZ", route)
    _apply_branch_probe(noise, "OZ", route)

    assert graph.nodes[0]["delta_nfr"] == 0.5
    assert noise.nodes[0]["delta_nfr"] == 0.25
    assert graph.nodes[0]["nu_f"] == noise.nodes[0]["nu_f"] == 1.0


@pytest.mark.parametrize("enabled", [True, "true", "on"])
def test_oz_enabled_noise_does_not_admit_unused_amplification_factor(enabled):
    factors = resolve_runtime_operator_factors(
        {"OZ_dnfr_factor": "unused"}, Glyph.OZ, {"OZ_NOISE_MODE": enabled}
    )
    assert factors["OZ_dnfr_factor"] == "unused"


@pytest.mark.parametrize("route", ["glyph", "public", "stage"])
@pytest.mark.parametrize(
    ("glyph", "key", "value", "error"),
    [
        ("OZ", "OZ_NOISE_MODE", "unknown", GlyphFactorValidationError),
        ("OZ", "OZ_NOISE_MODE", "", GlyphFactorValidationError),
        *[
            ("NAV", key, value, OperatorPreconditionError)
            for key in ("NAV_STRICT", "NAV_RANDOM")
            for value in ("false", "unknown", 0, 1, None)
        ],
    ],
)
def test_invalid_branch_flags_reject_before_any_graph_write(
    route, glyph, key, value, error
):
    graph = _public_graph({})
    graph.graph[key] = value
    before = deepcopy(graph)

    with pytest.raises(error, match=key):
        _apply_branch_probe(graph, glyph, route)

    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert list(graph.edges(data=True)) == list(before.edges(data=True))
    assert graph.graph == before.graph


@pytest.mark.parametrize("key", ["NAV_STRICT", "NAV_RANDOM"])
def test_nav_strict_flags_reject_before_factor_selection(key):
    with pytest.raises(OperatorPreconditionError, match=key):
        resolve_runtime_operator_factors(
            {"NAV_jitter": "also invalid"}, Glyph.NAV, {key: "false"}
        )
