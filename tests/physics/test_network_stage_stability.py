"""Full-map and repetition boundary for atomic EN/RA Jacobi stages."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.errors import TNFRValueError
from tnfr.operators._neighbor_epi_kernel import neighbor_epi_proposed_kind
from tnfr.operators._resonance_identity import resonance_proposed_epi_kind
from tnfr.operators.word_execution import run_network_sequence
from tnfr.physics import (
    certify_all_target_neighbor_stage,
    certify_reception_all_target_stage,
    certify_resonance_all_target_stage,
)
from tnfr.types import real_scalar_epi


def _graph(
    *,
    topology: nx.Graph | None = None,
    epi: tuple[float, ...] = (0.0, 0.2, 0.9),
    phase: tuple[float, ...] = (0.0, 0.2, 0.4),
    frequency: tuple[float, ...] | None = None,
    kind: tuple[str, ...] | None = None,
) -> nx.Graph:
    graph = topology if topology is not None else nx.path_graph(len(epi))
    frequencies = frequency or (1.0,) * len(epi)
    kinds = kind or ("wave",) * len(epi)
    for node, value, theta, nu_f, epi_kind in zip(
        graph,
        epi,
        phase,
        frequencies,
        kinds,
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=value,
            nu_f=nu_f,
            theta=theta,
            delta_nfr=0.1,
            Si=0.8,
            EPI_kind=epi_kind,
            glyph_history=["AL", "IL"],
        )
    graph.graph["GLYPH_FACTORS"] = {
        "EN_mix": 0.25,
        "RA_epi_diff": 0.25,
        "RA_vf_amplification": 0.25,
        "RA_phase_coupling": 0.5,
    }
    return graph


def _epi_state(graph: nx.Graph) -> tuple[float, ...]:
    return tuple(
        float(real_scalar_epi(get_attr(graph.nodes[node], ALIAS_EPI)))
        for node in graph
    )


def _kind(graph: nx.Graph, node: int) -> str:
    return str(
        get_attr(
            graph.nodes[node],
            ALIAS_EPI_KIND,
            "",
            strict=True,
            conv=lambda item: item,
        )
    )


def test_reception_full_map_trace_matches_three_actual_atomic_stages() -> None:
    graph = _graph()
    node_state_before = deepcopy(dict(graph.nodes(data=True)))
    graph_state_before = deepcopy(dict(graph.graph))

    result = certify_reception_all_target_stage(
        graph,
        fixed_support_declared=True,
        repetitions=3,
    )

    assert dict(graph.nodes(data=True)) == node_state_before
    assert dict(graph.graph) == graph_state_before
    assert result.repetitions_completed == 3
    assert result.repetitions_observed == 3
    assert result.all_stages_admissible
    assert result.ideal_real_hard_clipping_inactive_for_arbitrary_repetitions
    assert result.ideal_real_fixed_map_repetition_certified
    assert result.represented_fixed_map_repetition_certified
    assert result.represented_finite_repetition_disagreement_contraction_certified
    assert result.represented_asymptotic_disagreement_convergence_certified
    assert result.observed_ideal_real_maps_constant
    assert result.observed_represented_maps_constant
    assert result.observed_diffusion_metric_exactly_common
    assert not result.global_binary64_runtime_repetition_certified

    actual = deepcopy(graph)
    for step in result.steps:
        run_network_sequence(
            actual,
            ["reception"],
            cycles=1,
            validate=False,
            suppress_birth_warnings=True,
        )
        assert _epi_state(actual) == tuple(step.runtime_accepted_state_after)
        for node, local in zip(actual, step.local_certificates):
            assert _kind(actual, node) == local.epi_kind_after

    # The observed kernel agrees with runtime at every step, while its exact
    # two-stage binary64 result need not equal the represented matrix power.
    assert not result.observed_runtime_matches_represented_repeated_state_exactly
    assert any(
        value != 0
        for value in result.exact_observed_runtime_minus_represented_repeated_state
    )


def test_resonance_full_trace_matches_epi_frequency_phase_and_kind() -> None:
    graph = _graph()
    result = certify_resonance_all_target_stage(
        graph,
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=True,
        repetitions=3,
    )

    assert result.all_stages_admissible
    assert result.ideal_real_fixed_map_repetition_certified
    assert result.represented_fixed_map_repetition_certified
    assert result.ra_global_sign_identity_forward_invariant
    assert result.ra_global_kind_identity_forward_invariant
    assert result.observed_runtime_neighbor_sets_constant
    assert result.observed_diffusion_metric_exactly_common

    actual = deepcopy(graph)
    for step in result.steps:
        run_network_sequence(actual, ["resonance"], cycles=1, validate=False)
        assert _epi_state(actual) == tuple(step.runtime_accepted_state_after)
        for node, local in zip(actual, step.local_certificates):
            target = local.target_index
            assert get_attr(actual.nodes[node], ALIAS_VF) == (
                local.frequency_after[target]
            )
            assert get_attr(actual.nodes[node], ALIAS_THETA) == local.phase_after
            assert _kind(actual, node) == local.epi_kind_after


def test_soft_clipping_is_observed_and_blocks_affine_repetition_promotion() -> None:
    graph = _graph(epi=(0.99, 0.99, 0.99))
    graph.graph["CLIP_MODE"] = "soft"

    result = certify_reception_all_target_stage(
        graph,
        fixed_support_declared=True,
        repetitions=2,
    )

    assert result.any_runtime_clip_intervention
    assert not result.ideal_real_hard_clipping_inactive_for_arbitrary_repetitions
    assert not result.ideal_real_fixed_map_repetition_certified
    assert not result.represented_fixed_map_repetition_certified
    assert "hard_clip_mode" in result.failed_ideal_real_fixed_map_conditions
    assert result.exact_represented_repeated_map is None
    assert result.represented_repeated_energy_gain_bound is None

    actual = deepcopy(graph)
    for step in result.steps:
        run_network_sequence(
            actual,
            ["reception"],
            cycles=1,
            validate=False,
            suppress_birth_warnings=True,
        )
        assert _epi_state(actual) == tuple(step.runtime_accepted_state_after)
        for node, local in zip(actual, step.local_certificates):
            assert _kind(actual, node) == local.epi_kind_after


def test_represented_row_rounding_abstains_on_runtime_consensus() -> None:
    graph = _graph(
        topology=nx.star_graph(3),
        epi=(0.4, 0.4, 0.4, 0.4),
        phase=(0.0, 0.0, 0.0, 0.0),
    )
    graph.graph["GLYPH_FACTORS"]["EN_mix"] = 0.5

    result = certify_reception_all_target_stage(
        graph,
        fixed_support_declared=True,
        repetitions=2,
    )

    center_sum = result.steps[0].exact_represented_row_sums[0]
    assert center_sum != Fraction(1)
    assert _epi_state(graph) == (0.4, 0.4, 0.4, 0.4)
    assert tuple(result.steps[-1].runtime_accepted_state_after) == (
        0.4,
        0.4,
        0.4,
        0.4,
    )
    assert not result.represented_fixed_map_repetition_certified
    assert (
        "represented_map_exact_consensus_subspace_preservation"
        in result.failed_represented_fixed_map_conditions
    )


def test_identity_stage_uses_exact_quotient_gain_instead_of_frobenius() -> None:
    graph = _graph()

    result = certify_reception_all_target_stage(
        graph,
        fixed_support_declared=True,
        repetitions=4,
        mix_factor=0.0,
    )

    assert result.represented_fixed_map_repetition_certified
    assert result.exact_represented_repeated_energy_gain_bound == 1
    assert result.represented_repeated_energy_gain_bound == 1.0
    assert not result.represented_finite_repetition_disagreement_contraction_certified
    assert not result.represented_asymptotic_disagreement_convergence_certified


def test_observed_ra_gate_change_refutes_a_false_fixed_neighbor_declaration() -> None:
    graph = _graph(
        topology=nx.complete_graph(3),
        epi=(0.2, 0.4, 0.8),
        phase=(0.0, 1.4, 2.8),
    )

    result = certify_resonance_all_target_stage(
        graph,
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=True,
        repetitions=2,
        phase_coupling_factor=0.5,
    )

    assert result.all_stages_admissible
    assert not result.observed_runtime_neighbor_sets_constant
    assert not result.observed_ideal_real_maps_constant
    assert not result.ideal_real_fixed_map_repetition_certified
    assert "observed_ideal_real_maps_constant" in (
        result.failed_ideal_real_fixed_map_conditions
    )

    actual = deepcopy(graph)
    for step in result.steps:
        run_network_sequence(actual, ["resonance"], cycles=1, validate=False)
        assert _epi_state(actual) == tuple(step.runtime_accepted_state_after)
        for node, local in zip(actual, step.local_certificates):
            target = local.target_index
            assert get_attr(actual.nodes[node], ALIAS_VF) == (
                local.frequency_after[target]
            )
            assert get_attr(actual.nodes[node], ALIAS_THETA) == local.phase_after
            assert _kind(actual, node) == local.epi_kind_after


def test_ra_selective_capacity_amplification_reports_metric_change_separately() -> None:
    graph = _graph(epi=(0.0, 0.0, 0.8))

    result = certify_resonance_all_target_stage(
        graph,
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=True,
        repetitions=1,
    )

    assert result.all_stages_admissible
    assert not result.steps[0].pre_post_metric_exactly_proportional
    assert not result.observed_diffusion_metric_exactly_common
    assert result.ideal_real_fixed_map_repetition_certified


def test_ra_identity_failure_is_an_atomic_negative_stage_certificate() -> None:
    graph = _graph(
        topology=nx.path_graph(2),
        epi=(-0.8, 0.2),
        phase=(0.0, 0.0),
    )
    graph.graph["GLYPH_FACTORS"]["RA_epi_diff"] = 1.0
    before = deepcopy(dict(graph.nodes(data=True)))

    result = certify_resonance_all_target_stage(
        graph,
        fixed_support_declared=True,
        fixed_phase_neighbor_sets_declared=True,
        repetitions=3,
    )

    assert result.repetitions_observed == 1
    assert result.repetitions_completed == 0
    assert not result.all_stages_admissible
    assert result.steps[0].atomic_rejection_nodes == (0, 1)
    np.testing.assert_array_equal(
        result.steps[0].runtime_accepted_state_after,
        result.steps[0].state_before,
    )
    assert dict(graph.nodes(data=True)) == before
    assert not result.ideal_real_fixed_map_repetition_certified

    actual = deepcopy(graph)
    with pytest.raises(TNFRValueError, match="identity gate rejected"):
        run_network_sequence(actual, ["resonance"], cycles=1, validate=False)
    assert dict(actual.nodes(data=True)) == before


def test_shared_kind_kernel_preserves_operator_specific_unlabeled_policy() -> None:
    neighbors = [(0.9, "")]

    assert neighbor_epi_proposed_kind(
        "seed", neighbors, 0.1, fallback_kind="EN"
    ) == "seed"
    assert resonance_proposed_epi_kind("seed", neighbors, 0.1) == "RA"


@pytest.mark.parametrize("repetitions", [True, 0, -1, 1.5])
def test_repetition_count_is_strictly_positive_integer(repetitions) -> None:
    graph = _graph()
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        certify_all_target_neighbor_stage(
            graph,
            "EN",
            fixed_support_declared=True,
            repetitions=repetitions,
        )
