"""Runtime and certificate boundary for local Resonance (RA)."""

from copy import deepcopy
from fractions import Fraction
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.canonical import (
    COUPLING_FINE,
    COUPLING_GENTLE,
    COUPLING_MODERATE,
)
from tnfr.errors import TNFRValueError
from tnfr.node import NodeNX
from tnfr.operators import _op_RA, apply_glyph, get_glyph_factors
from tnfr.physics import certify_resonance_epi_realization


def _state(graph, epi, *, frequency=1.0, phase=None, kind="seed"):
    phases = phase or {node: 0.0 for node in graph}
    frequencies = (
        frequency if hasattr(frequency, "keys") else {node: frequency for node in graph}
    )
    kinds = kind if hasattr(kind, "keys") else {node: kind for node in graph}
    for node in graph:
        graph.nodes[node].update(
            EPI=epi[node],
            nu_f=frequencies[node],
            theta=phases[node],
            delta_nfr=0.0,
            Si=0.8,
            epi_kind=kinds[node],
        )
    return graph


def _channels(graph, node):
    data = graph.nodes[node]
    return (
        deepcopy(data.get("EPI")),
        data.get("nu_f"),
        data.get("theta"),
        data.get("delta_nfr"),
        data.get("epi_kind"),
        deepcopy(data.get("glyph_history")),
        deepcopy(graph.graph.get("ra_metrics")),
        deepcopy(graph.graph.get("_ra_c_tracking")),
    )


def test_runtime_matches_certificate_for_epi_kind_frequency_and_phase():
    target = ("hub", 1)
    good = "good"
    bad = 17
    graph = nx.Graph()
    graph.add_edge(target, good, weight=9.0)
    graph.add_edge(target, bad, weight=1.0)
    _state(
        graph,
        {target: 0.0, good: 1.0, bad: -1.0},
        frequency={target: 10.0, good: 9.0, bad: 1.0},
        phase={target: 0.0, good: 0.2, bad: math.pi},
    )
    certificate = certify_resonance_epi_realization(
        graph,
        target,
        fixed_support_declared=True,
        mix_factor=0.5,
        vf_amplification_factor=0.25,
        phase_coupling_factor=0.5,
    )
    runtime = deepcopy(graph)
    runtime.graph["GLYPH_FACTORS"] = {
        "RA_epi_diff": 0.5,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
    }
    _op_RA(NodeNX.from_graph(runtime, target), get_glyph_factors(NodeNX(runtime, target)))

    assert certificate.graph_neighbors == (good, bad)
    assert certificate.runtime_neighbors == (good,)
    assert certificate.phase_incompatible_neighbors == (bad,)
    assert float(NodeNX(runtime, target).EPI) == certificate.runtime_target_value
    assert NodeNX(runtime, target).epi_kind == certificate.epi_kind_after
    assert NodeNX(runtime, target).vf == certificate.frequency_after[0]
    assert NodeNX(runtime, target).theta == certificate.phase_after
    assert certificate.unweighted_runtime_neighbor_mean == 1.0
    assert certificate.transport_weighted_neighbor_mean == pytest.approx(0.8)


def test_sign_flip_is_rejected_atomically_and_kind_clause_passes():
    graph = _state(nx.path_graph(2), {0: -0.1, 1: 1.0})
    graph.graph["GLYPH_FACTORS"] = {
        "RA_epi_diff": 0.5,
        "RA_vf_amplification": 0.5,
        "RA_phase_coupling": 0.5,
    }
    node = NodeNX.from_graph(graph, 0)
    before = _channels(graph, 0)

    with pytest.raises(TNFRValueError, match="nonzero_epi_sign_would_flip"):
        _op_RA(node, get_glyph_factors(node))

    assert _channels(graph, 0) == before
    certificate = certify_resonance_epi_realization(
        graph, 0, fixed_support_declared=True, mix_factor=0.5
    )
    assert not certificate.sign_identity_compatible
    assert certificate.kind_identity_compatible
    assert not certificate.identity_gate_passed
    assert np.array_equal(certificate.state_after, certificate.state_before)
    assert certificate.post_diffusion_certificate is None


def test_kind_change_is_rejected_independently_and_atomically():
    graph = _state(
        nx.path_graph(2),
        {0: 0.2, 1: 1.0},
        kind={0: "seed", 1: "wave"},
    )
    graph.graph["GLYPH_FACTORS"] = {"RA_epi_diff": 0.5}
    node = NodeNX.from_graph(graph, 0)
    before = _channels(graph, 0)

    with pytest.raises(TNFRValueError, match="established_epi_kind_would_change"):
        _op_RA(node, get_glyph_factors(node))

    assert _channels(graph, 0) == before
    certificate = certify_resonance_epi_realization(
        graph, 0, fixed_support_declared=True, mix_factor=0.5
    )
    assert certificate.sign_identity_compatible
    assert not certificate.kind_identity_compatible


def test_zero_is_neutral_and_missing_kind_can_be_initialized():
    graph = _state(
        nx.path_graph(2),
        {0: 0.0, 1: 0.4},
        kind={0: None, 1: "wave"},
    )
    graph.graph["GLYPH_FACTORS"] = {"RA_epi_diff": 0.5}
    node = NodeNX.from_graph(graph, 0)
    _op_RA(node, get_glyph_factors(node))

    assert float(node.EPI) == pytest.approx(0.2)
    assert node.epi_kind == "wave"


@pytest.mark.parametrize(
    "factors",
    [
        {"RA_epi_diff": -0.1},
        {"RA_epi_diff": 1.1},
        {"RA_vf_amplification": -0.1},
        {"RA_phase_coupling": -0.1},
        {"RA_phase_coupling": 1.1},
    ],
)
def test_anticontractual_factor_overrides_fail_atomically(factors):
    graph = _state(nx.path_graph(2), {0: 0.2, 1: 0.4})
    node = NodeNX.from_graph(graph, 0)
    before = _channels(graph, 0)

    with pytest.raises(TNFRValueError, match="factor gate"):
        _op_RA(node, get_glyph_factors(node) | factors)

    assert _channels(graph, 0) == before


def test_isolate_is_rejected_by_direct_apply_glyph_without_history():
    graph = _state(nx.empty_graph(1), {0: 0.2})
    before = _channels(graph, 0)

    with pytest.raises(TNFRValueError, match="coupled neighbor"):
        apply_glyph(graph, 0, "RA")

    assert _channels(graph, 0) == before


def test_exact_antipodal_boundary_does_not_choose_an_arbitrary_phase():
    graph = _state(
        nx.star_graph(2),
        {0: 0.2, 1: 0.3, 2: 0.3},
        phase={0: 0.0, 1: math.pi / 2, 2: -math.pi / 2},
    )
    result = certify_resonance_epi_realization(
        graph, 0, fixed_support_declared=True, mix_factor=0.5
    )

    assert not result.neighbor_circular_mean_defined
    assert result.neighbor_circular_mean_phase is None
    assert result.proposed_phase_after == result.phase_before


def test_one_third_consensus_exposes_binary64_nonaffinity_boundary():
    graph = _state(nx.star_graph(2), {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
    result = certify_resonance_epi_realization(
        graph, 0, fixed_support_declared=True
    )

    assert result.runtime_proposed_target_value != 1 / 3
    assert result.exact_represented_target_row_sum != Fraction(1)
    assert not result.represented_map_exact_consensus_subspace_preservation
    assert result.affine_jump_certificate is None
    assert not result.global_binary64_runtime_affinity_certified


def test_successful_ra_has_post_flow_but_abstains_from_pre_post_switching():
    graph = nx.Graph()
    graph.add_edge("hub", "a", weight=9.0)
    graph.add_edge("hub", "b", weight=1.0)
    _state(
        graph,
        {"hub": 0.0, "a": 1.0, "b": 0.0},
        frequency={"hub": 10.0, "a": 9.0, "b": 1.0},
    )
    result = certify_resonance_epi_realization(
        graph,
        "hub",
        fixed_support_declared=True,
        mix_factor=0.5,
        vf_amplification_factor=0.25,
    )

    assert result.identity_gate_passed
    assert result.post_diffusion_certificate is not None
    assert result.post_diffusion_certificate.is_certified
    assert not result.pre_post_metric_exactly_proportional
    assert result.pre_post_switching_certificate is None
    assert "pre_post_metrics_not_exactly_proportional" in result.switching_abstention_reasons
    assert result.affine_jump_certificate is not None
    assert result.nested_affine_certificate_uses_post_resonance_metric
    assert result.pressure_refresh_required

    estimate = result.recovery_break_even_duration_estimate
    assert estimate is not None
    recovered = certify_resonance_epi_realization(
        graph,
        "hub",
        fixed_support_declared=True,
        mix_factor=0.5,
        vf_amplification_factor=0.25,
        recovery_flow_duration=max(1.0, 2.0 * estimate),
    )
    assert recovered.hybrid_certificate is not None
    assert recovered.represented_hybrid_recovery_certified is True


def test_default_certificate_factors_equal_runtime_factor_resolution():
    graph = _state(nx.path_graph(2), {0: 0.2, 1: 0.3})
    graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.7}
    factors = get_glyph_factors(NodeNX.from_graph(graph, 0))
    result = certify_resonance_epi_realization(
        graph, 0, fixed_support_declared=True
    )

    assert result.mix_factor == factors["RA_epi_diff"] == COUPLING_MODERATE
    assert (
        result.vf_amplification_factor
        == factors["RA_vf_amplification"]
        == COUPLING_FINE
    )
    assert (
        result.phase_coupling_factor
        == factors["RA_phase_coupling"]
        == COUPLING_GENTLE
    )


def test_certificate_rejects_directed_isolated_rich_and_relaxed_phase_gate():
    directed = _state(nx.DiGraph([(0, 1)]), {0: 0.2, 1: 0.3})
    with pytest.raises(ValueError, match="undirected"):
        certify_resonance_epi_realization(
            directed, 0, fixed_support_declared=True
        )

    isolated = _state(nx.empty_graph(1), {0: 0.2})
    with pytest.raises(ValueError, match="at least two nodes|connected"):
        certify_resonance_epi_realization(
            isolated, 0, fixed_support_declared=True
        )

    rich = _state(nx.path_graph(2), {0: 0.2, 1: 0.3})
    rich.nodes[1]["EPI"] = {
        "continuous": (0.2 + 0j, 0.3 + 0j),
        "discrete": (0.2 + 0j, 0.3 + 0j),
        "grid": (0.0, 1.0),
    }
    with pytest.raises(ValueError, match="uniform-real BEPI"):
        certify_resonance_epi_realization(rich, 0, fixed_support_declared=True)

    relaxed = _state(nx.path_graph(2), {0: 0.2, 1: 0.3})
    relaxed.graph["DELTA_PHI_MAX"] = math.pi
    with pytest.raises(ValueError, match="canonical interval"):
        certify_resonance_epi_realization(relaxed, 0, fixed_support_declared=True)


def test_certificate_arrays_are_detached_and_read_only():
    result = certify_resonance_epi_realization(
        _state(nx.path_graph(2), {0: 0.2, 1: 0.3}),
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )
    for array in (
        result.state_before,
        result.proposed_state_after,
        result.state_after,
        result.frequency_before,
        result.frequency_after,
        result.metric_weights_before,
        result.metric_weights_after,
        result.represented_linear_map,
        result.post_reset_pressure_manifold_defect,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 99.0
