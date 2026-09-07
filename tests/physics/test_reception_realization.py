"""Runtime-realization boundary for local Reception (EN)."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.canonical import EN_MIX_FACTOR
from tnfr.node import NodeNX
from tnfr.operators import _op_EN, get_glyph_factors
from tnfr.physics.reception_realization import (
    certify_reception_epi_realization,
)
from tnfr.types import serialize_bepi


def _state(graph, epi, frequency=1.0):
    values = dict(epi)
    if hasattr(frequency, "keys"):
        frequencies = frequency
    else:
        frequencies = {node: frequency for node in graph}
    for node in graph:
        graph.nodes[node].update(
            EPI=values[node], nu_f=frequencies[node], theta=0.0
        )
    return graph


def _path_state(values=(0.8, 0.2, 0.6)):
    graph = nx.path_graph(3)
    return _state(graph, dict(zip(graph, values)))


def test_arbitrary_ids_and_weighted_transport_keep_en_mean_unweighted():
    hub = ("hub", 1)
    left = "left"
    right = 17
    graph = nx.Graph()
    graph.add_edge(hub, left, weight=9.0)
    graph.add_edge(hub, right, weight=1.0)
    _state(graph, {hub: 0.0, left: 1.0, right: 0.0})
    attrs_before = {node: dict(graph.nodes[node]) for node in graph}
    graph_before = dict(graph.graph)

    result = certify_reception_epi_realization(
        graph,
        hub,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.target == hub
    assert set(result.runtime_neighbors) == {left, right}
    assert result.unweighted_runtime_neighbor_mean == 0.5
    assert result.transport_weighted_neighbor_mean == pytest.approx(0.9)
    assert result.runtime_target_value == 0.25
    assert result.ideal_real_hard_clipping_inactive_by_convexity
    assert result.runtime_hard_clipping_inactive_at_snapshot
    assert result.affine_jump_certificate is not None
    assert result.runtime_snapshot_in_affine_model_domain
    assert {node: dict(graph.nodes[node]) for node in graph} == attrs_before
    assert dict(graph.graph) == graph_before

    runtime_copy = deepcopy(graph)
    runtime_node = NodeNX.from_graph(runtime_copy, hub)
    _op_EN(runtime_node, get_glyph_factors(runtime_node) | {"EN_mix": 0.5})
    assert runtime_node.EPI == result.runtime_target_value


def test_signed_scalar_embedding_matches_runtime_and_is_affine_eligible():
    graph = _state(nx.path_graph(2), {0: 0.2, 1: -0.8})

    result = certify_reception_epi_realization(
        graph,
        1,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.runtime_target_value == pytest.approx(-0.3)
    assert result.exact_runtime_minus_ideal_target != 0
    assert result.exact_runtime_minus_represented_affine_target != 0
    assert not result.runtime_matches_ideal_target_exactly_at_snapshot
    assert not result.runtime_matches_represented_affine_exactly_at_snapshot
    assert result.runtime_matches_represented_affine_within_tolerance
    assert result.affine_jump_certificate is not None
    assert result.runtime_snapshot_in_affine_model_domain
    assert not result.affine_abstention_reasons
    assert result.nested_affine_certificate_is_represented_model_only
    assert not result.global_binary64_runtime_affinity_certified

    runtime_copy = deepcopy(graph)
    runtime_node = NodeNX.from_graph(runtime_copy, 1)
    _op_EN(runtime_node, {"EN_mix": 0.5})
    assert float(runtime_node.EPI) == result.runtime_target_value


def test_serialized_signed_scalar_embedding_uses_the_same_runtime_channel():
    graph = _state(nx.path_graph(2), {0: 0.2, 1: -0.8})
    for node in graph:
        graph.nodes[node]["EPI"] = serialize_bepi(graph.nodes[node]["EPI"])

    result = certify_reception_epi_realization(
        graph,
        1,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.state_before.tolist() == [0.2, -0.8]
    assert result.runtime_target_value == pytest.approx(-0.3)
    assert result.runtime_snapshot_in_affine_model_domain
    assert result.affine_jump_certificate is not None

    runtime_copy = deepcopy(graph)
    runtime_node = NodeNX.from_graph(runtime_copy, 1)
    _op_EN(runtime_node, {"EN_mix": 0.5})
    assert float(runtime_node.EPI) == result.runtime_target_value


def test_nonuniform_bepi_is_rejected_without_a_common_real_diffusion_state():
    graph = _path_state()
    graph.nodes[0]["EPI"] = {
        "continuous": (0.2 + 0.0j, 0.3 + 0.0j),
        "discrete": (0.2 + 0.0j, 0.3 + 0.0j),
        "grid": (0.0, 1.0),
    }

    with pytest.raises(ValueError, match="uniform real BEPI embedding"):
        certify_reception_epi_realization(
            graph,
            0,
            fixed_support_declared=True,
            mix_factor=0.5,
        )


@pytest.mark.parametrize(
    ("values", "mix"),
    [
        ((0.3, 0.3, 0.3), EN_MIX_FACTOR),
        ((0.8, 0.4, 0.6), 0.0),
    ],
)
def test_constant_field_or_zero_mix_is_a_runtime_noop(values, mix):
    result = certify_reception_epi_realization(
        _path_state(values),
        1,
        fixed_support_declared=True,
        mix_factor=mix,
    )

    assert not result.nontrivial_runtime_reception
    assert not result.pressure_refresh_required
    assert result.post_reset_pressure_manifold_defect_norm == 0.0
    assert result.current_weighted_mean_preserved_exactly
    if mix:
        assert not result.ideal_real_global_weighted_mean_preservation
    else:
        assert result.ideal_real_global_weighted_mean_preservation


def test_nontrivial_local_en_cannot_preserve_the_weighted_mean_functional():
    result = certify_reception_epi_realization(
        _path_state(),
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.nontrivial_runtime_reception
    assert not result.ideal_real_global_weighted_mean_preservation
    assert any(result.exact_ideal_weighted_mean_linear_defect)
    assert result.exact_current_weighted_mean_shift != 0
    assert not result.current_weighted_mean_preserved_exactly
    assert result.affine_jump_certificate is not None
    assert not result.affine_jump_certificate.exact_weighted_mean_preservation


@pytest.mark.parametrize(
    ("values", "expected_nontrivial"),
    [
        ((0.2, 0.2, 0.2), False),
        ((0.8, 0.2, 0.6), True),
    ],
)
def test_pressure_refresh_defect_occurs_iff_runtime_en_is_nontrivial(
    values, expected_nontrivial
):
    result = certify_reception_epi_realization(
        _path_state(values),
        1,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.nontrivial_runtime_reception is expected_nontrivial
    assert result.pressure_refresh_required is expected_nontrivial
    assert result.pressure_refresh_detected_at_snapshot is expected_nontrivial
    assert result.exact_pressure_refresh_iff_nontrivial_theorem
    assert result.numerical_pressure_refresh_iff_nontrivial_at_snapshot
    assert (result.post_reset_pressure_manifold_defect_norm > 0.0) is (
        expected_nontrivial
    )


def test_exact_pressure_refresh_theorem_is_separate_from_binary64_detection():
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=1e308)
    graph.add_edge(1, 1, weight=1e308)
    graph.add_edge(0, 1, weight=5e-324)
    _state(graph, {0: 0.0, 1: 1.0}, frequency=1e308)

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.nontrivial_runtime_reception
    assert result.pressure_refresh_required
    assert result.exact_pressure_refresh_iff_nontrivial_theorem
    assert not result.pressure_refresh_detected_at_snapshot
    assert not result.numerical_pressure_refresh_iff_nontrivial_at_snapshot
    assert not result.diffusion_certificate.is_certified
    assert result.recovery_break_even_duration_estimate is None
    np.testing.assert_array_equal(
        result.post_reset_pressure_manifold_defect, [0.0, 0.0]
    )


def test_uncertified_binary64_flow_abstains_from_optional_hybrid_composition():
    graph = nx.Graph()
    graph.add_edge(0, 0, weight=1e308)
    graph.add_edge(1, 1, weight=1e308)
    graph.add_edge(0, 1, weight=5e-324)
    _state(graph, {0: 0.0, 1: 1.0}, frequency=1e308)

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
        recovery_flow_duration=1.0,
    )

    assert result.affine_jump_certificate is not None
    assert not result.diffusion_certificate.is_certified
    assert result.hybrid_certificate is None
    assert result.represented_hybrid_recovery_certified is None


def test_display_estimate_samples_both_sides_of_exact_recovery_decision():
    graph = _path_state()
    baseline = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )
    estimate = baseline.recovery_break_even_duration_estimate

    assert baseline.represented_affine_jump_eligible
    assert estimate is not None and estimate > 0.0

    below = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
        recovery_flow_duration=0.5 * estimate,
    )
    above = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
        recovery_flow_duration=2.0 * estimate,
    )

    assert below.hybrid_certificate is not None
    assert below.represented_hybrid_recovery_certified is False
    assert above.hybrid_certificate is not None
    assert above.represented_hybrid_recovery_certified is True
    assert above.hybrid_certificate.disagreement_contracts_over_declared_horizon


@pytest.mark.parametrize(
    ("duration", "message"),
    [
        (Fraction(-1, 10**1000), "nonnegative"),
        (Fraction(1, 10**1000), "below nonzero floating-point range"),
    ],
)
def test_recovery_duration_rejects_values_lost_to_binary64(duration, message):
    with pytest.raises(ValueError, match=message):
        certify_reception_epi_realization(
            _path_state(),
            0,
            fixed_support_declared=True,
            mix_factor=0.5,
            recovery_flow_duration=duration,
        )


@pytest.mark.parametrize(
    ("graph_updates", "mix", "reason"),
    [
        ({"CLIP_MODE": "soft"}, 0.5, "hard_clip_mode"),
        ({"CLIP_MODE": "hard"}, 1.25, "convex_mix_factor"),
        ({"CLIP_MODE": "hard"}, -0.25, "convex_mix_factor"),
    ],
)
def test_nonlinear_or_extrapolating_runtime_abstains_from_affine_promotion(
    graph_updates, mix, reason
):
    graph = _path_state()
    graph.graph.update(graph_updates)

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=mix,
        recovery_flow_duration=1.0,
    )

    assert not result.ideal_real_affine_regime
    assert not result.ideal_real_hard_clipping_inactive_by_convexity
    assert not result.represented_affine_jump_eligible
    assert result.affine_jump_certificate is None
    assert result.nested_affine_certificate_is_represented_model_only
    assert not result.global_binary64_runtime_affinity_certified
    assert result.hybrid_certificate is None
    assert result.represented_hybrid_recovery_certified is None
    assert reason in result.affine_abstention_reasons


def test_state_outside_declared_bounds_abstains_even_when_snapshot_does_not_clip():
    graph = _path_state((1.2, 0.2, 0.2))

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    assert result.runtime_hard_clipping_inactive_at_snapshot
    assert not result.ideal_real_hard_clipping_inactive_by_convexity
    assert not result.ideal_real_affine_regime
    assert "state_inside_epi_bounds" in result.affine_abstention_reasons
    assert result.affine_jump_certificate is None


def test_graph_factor_override_and_canonical_fallback_match_runtime_resolution():
    fallback_graph = _path_state()
    fallback_graph.graph["GLYPH_FACTORS"] = {"AL_boost": 0.7}
    override_graph = _path_state()
    override_graph.graph["GLYPH_FACTORS"] = {"EN_mix": 0.4}

    fallback = certify_reception_epi_realization(
        fallback_graph,
        0,
        fixed_support_declared=True,
    )
    override = certify_reception_epi_realization(
        override_graph,
        0,
        fixed_support_declared=True,
    )

    assert fallback.mix_factor == EN_MIX_FACTOR
    assert override.mix_factor == 0.4


def test_binary64_degree_rounding_is_not_promoted_by_tolerance():
    graph = nx.star_graph(5)
    _state(graph, {node: 1.0 for node in graph})

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
    )

    assert result.mix_factor == EN_MIX_FACTOR
    assert result.ideal_real_consensus_subspace_preservation
    assert result.exact_represented_target_row_sum != Fraction(1)
    assert not result.represented_map_exact_consensus_subspace_preservation
    assert result.runtime_matches_ideal_target_exactly_at_snapshot
    assert not result.runtime_matches_represented_affine_exactly_at_snapshot
    assert result.runtime_matches_represented_affine_within_tolerance
    assert not result.represented_affine_jump_eligible
    assert result.affine_jump_certificate is None
    assert "represented_binary64_row_sum_is_not_exactly_one" in (
        result.affine_abstention_reasons
    )


def test_degree_with_exact_represented_row_can_nest_the_affine_theorem():
    graph = nx.star_graph(3)
    _state(graph, {0: 0.5, 1: 0.2, 2: 0.1, 3: 0.8})

    result = certify_reception_epi_realization(
        graph,
        0,
        fixed_support_declared=True,
    )

    assert result.exact_represented_target_row_sum == 1
    assert result.represented_map_exact_consensus_subspace_preservation
    assert result.represented_affine_jump_eligible
    assert result.affine_jump_certificate is not None
    assert result.affine_jump_certificate.supports_global_gain_theorem


def test_certificate_arrays_are_detached_and_read_only():
    result = certify_reception_epi_realization(
        _path_state(),
        0,
        fixed_support_declared=True,
        mix_factor=0.5,
    )

    for array in (
        result.state_before,
        result.state_after,
        result.represented_linear_map,
        result.current_pure_epi_pressure,
        result.post_reset_pure_epi_pressure,
        result.post_reset_pressure_manifold_defect,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 99.0


def test_fixed_support_must_be_declared_for_theorem_promotion():
    result = certify_reception_epi_realization(
        _path_state(),
        0,
        fixed_support_declared=False,
        mix_factor=0.5,
    )

    assert "fixed_support_declared" in result.failed_ideal_real_affine_conditions
    assert result.affine_jump_certificate is None
    assert not result.represented_affine_jump_eligible


def test_empty_neighbor_and_directed_support_are_rejected():
    isolated = nx.Graph()
    isolated.add_node("only", EPI=0.0, nu_f=1.0, theta=0.0)
    with pytest.raises(ValueError, match="at least two nodes|connected"):
        certify_reception_epi_realization(
            isolated, "only", fixed_support_declared=True
        )

    directed = nx.DiGraph()
    directed.add_edge("a", "b", weight=1.0)
    _state(directed, {"a": 0.0, "b": 1.0})
    with pytest.raises(ValueError, match="undirected"):
        certify_reception_epi_realization(
            directed, "a", fixed_support_declared=True
        )
