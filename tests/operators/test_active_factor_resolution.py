"""Runtime factor validation follows the concrete active operator branch."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.operators.definitions import Emission, Reception, Silence, Transition
from tnfr.operators.factor_contracts import (
    GlyphFactorValidationError,
    resolve_runtime_operator_factors,
    runtime_active_glyph_factor_keys,
)
from tnfr.types import Glyph


@pytest.mark.parametrize(
    ("glyph", "graph_data", "overrides", "inactive_key"),
    [
        (Glyph.OZ, {"OZ_NOISE_MODE": True}, {"OZ_dnfr_factor": "unused"}, "OZ_dnfr_factor"),
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
def test_public_operator_factor_preflight_precedes_subclass_metadata(
    operator, factor
):
    graph = _public_graph(factor)
    node_before = deepcopy(graph.nodes[0])
    graph_before = deepcopy(graph.graph)

    with pytest.raises(GlyphFactorValidationError):
        operator(graph, 0)

    assert graph.nodes[0] == node_before
    assert graph.graph == graph_before
