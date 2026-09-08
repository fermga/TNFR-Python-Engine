"""Rigorous affine-jump and hybrid pure-EPI stability certificates."""

from dataclasses import replace
from fractions import Fraction
import math
import sys

import networkx as nx
import numpy as np
import pytest

import tnfr._exact_time as exact_time
import tnfr.physics.hybrid_operator_stability as hybrid_stability
from tnfr.physics.hybrid_operator_stability import (
    certify_affine_epi_jump_gain,
    compose_hybrid_epi_stability,
)
from tnfr.physics.structural_diffusion import (
    verify_heterogeneous_diffusion_stability,
    verify_switching_diffusion_stability,
)


def _set_state(graph, epi, frequency):
    for node, value, nu_f in zip(graph, epi, frequency):
        graph.nodes[node].update(EPI=value, nu_f=nu_f, theta=0.0)
    return graph


def _path_flow():
    graph = _set_state(nx.path_graph(3), [2.0, -1.0, 4.0], [1.0] * 3)
    result = verify_heterogeneous_diffusion_stability(graph)
    assert result.is_certified
    assert result.exponential_rate == pytest.approx(2.0)
    return result


def _local_reception(flow, *, declared=None):
    return certify_affine_epi_jump_gain(
        "Reception",
        [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        flow.metric_weights,
        nodes=flow.nodes,
        declared_energy_gain_bound=declared,
    )


def test_affine_jump_certificate_rejects_single_node_domain():
    with pytest.raises(ValueError, match="at least two nodes"):
        certify_affine_epi_jump_gain("Reception", [[1.0]], [1.0])


def test_local_reception_separates_sharp_estimate_from_rational_proof_bound():
    result = _local_reception(_path_flow())

    sharp = (41.0 + math.sqrt(657.0)) / 64.0
    assert result.exact_consensus_subspace_preservation
    assert result.sharp_quotient_energy_gain_estimate == pytest.approx(sharp)
    assert result.exact_weighted_frobenius_energy_bound == Fraction(41, 32)
    assert result.weighted_frobenius_energy_bound == 41.0 / 32.0
    assert result.exact_quotient_energy_gain_upper_bound < Fraction(41, 32)
    assert result.quotient_energy_gain_upper_bound == pytest.approx(sharp)
    assert result.energy_gain_bound_for_composition == (
        result.quotient_energy_gain_upper_bound
    )
    assert result.weighted_frobenius_energy_bound > (
        result.quotient_energy_gain_upper_bound
    )
    assert result.supports_global_gain_theorem
    assert not result.exact_weighted_mean_preservation


def test_nonuniform_offset_has_exact_zero_to_positive_counterexample():
    result = certify_affine_epi_jump_gain(
        "Emission",
        np.eye(3),
        [1.0, 2.0, 1.0],
        offset=[0.1, 0.0, 0.0],
    )

    assert not result.exact_consensus_subspace_preservation
    assert not result.finite_global_energy_gain
    assert math.isinf(result.global_energy_gain_bound)
    assert result.consensus_counterexample_level == 0.0
    assert result.exact_consensus_counterexample_energy_after > 0
    assert result.consensus_counterexample_energy_after > 0.0


def test_nonuniform_linear_consensus_image_uses_level_one_counterexample():
    result = certify_affine_epi_jump_gain(
        "Contraction", np.diag([0.5, 1.0, 1.0]), [1.0, 2.0, 1.0]
    )

    assert not result.exact_consensus_subspace_preservation
    assert result.consensus_counterexample_level == 1.0
    assert result.exact_consensus_counterexample_energy_after > 0
    assert not result.supports_global_gain_theorem


def test_tolerance_cannot_promote_a_near_consensus_identity():
    matrix = np.eye(3)
    matrix[0, 0] = np.nextafter(1.0, 2.0)

    result = certify_affine_epi_jump_gain(
        "Coherence", matrix, [1.0, 2.0, 1.0], tolerance=1e-10
    )

    assert result.consensus_subspace_preservation_within_tolerance
    assert not result.exact_consensus_subspace_preservation
    assert not result.supports_global_gain_theorem
    assert result.exact_consensus_counterexample_energy_after > 0


def test_declared_bounds_are_metadata_and_never_change_internal_proof_bound():
    flow = _path_flow()
    internal = _local_reception(flow)
    too_small = _local_reception(
        flow, declared=Fraction(41, 32) - Fraction(1, 10**12)
    )
    looser = _local_reception(flow, declared=2)

    assert too_small.declared_bound_within_tolerance
    assert too_small.declared_energy_gain_bound_certified
    assert not too_small.declared_bound_certified_by_frobenius
    assert looser.declared_energy_gain_bound_certified
    assert looser.declared_bound_certified_by_frobenius
    assert (
        internal.energy_gain_bound_for_composition
        == too_small.energy_gain_bound_for_composition
        == looser.energy_gain_bound_for_composition
    )
    assert internal.energy_gain_bound_for_composition < 41.0 / 32.0
    assert too_small.supports_global_gain_theorem


def test_fraction_inputs_certify_their_represented_binary64_map():
    rational = certify_affine_epi_jump_gain(
        "Reception",
        [
            [Fraction(9, 10), Fraction(1, 10)],
            [Fraction(1, 10), Fraction(9, 10)],
        ],
        [2, 3],
    )
    binary64 = certify_affine_epi_jump_gain(
        "Reception", [[0.9, 0.1], [0.1, 0.9]], [2.0, 3.0]
    )

    np.testing.assert_array_equal(rational.linear_map, binary64.linear_map)
    assert (
        rational.exact_weighted_frobenius_energy_bound
        == binary64.exact_weighted_frobenius_energy_bound
    )


def test_certificate_arrays_are_immutable():
    result = _local_reception(_path_flow())

    with pytest.raises(ValueError, match="read-only"):
        result.linear_map[0, 0] = 7.0
    with pytest.raises(ValueError, match="read-only"):
        result.metric_weights[0] = 7.0


def test_identity_and_consensus_projection_have_exact_unit_quotient_gain():
    flow = _path_flow()
    weights = tuple(Fraction.from_float(float(value)) for value in flow.metric_weights)
    total = sum(weights, Fraction(0))
    projection = np.asarray(
        [
            [
                float(Fraction(i == j) - weights[j] / total)
                for j in range(len(weights))
            ]
            for i in range(len(weights))
        ]
    )

    identity = certify_affine_epi_jump_gain(
        "Reception",
        np.eye(3),
        flow.metric_weights,
        nodes=flow.nodes,
        declared_energy_gain_bound=1,
    )
    projected = certify_affine_epi_jump_gain(
        "Coherence", projection, flow.metric_weights, nodes=flow.nodes
    )

    assert identity.exact_quotient_energy_gain_upper_bound == 1
    assert identity.energy_gain_bound_for_composition == 1.0
    assert identity.exact_weighted_frobenius_energy_bound == 2
    assert identity.declared_energy_gain_bound_certified
    assert not identity.declared_bound_certified_by_frobenius
    assert projected.exact_consensus_subspace_preservation
    assert projected.exact_quotient_energy_gain_upper_bound == 1
    assert projected.energy_gain_bound_for_composition == 1.0


def test_affine_theorem_properties_reject_replaced_decisive_fields():
    result = _local_reception(_path_flow())

    forged_gain = replace(
        result,
        exact_quotient_energy_gain_upper_bound=Fraction(0),
        energy_gain_bound_for_composition=0.0,
    )
    forged_mean = replace(result, exact_weighted_mean_preservation=True)

    assert not forged_gain.supports_global_gain_theorem
    assert not forged_mean.preserves_initial_weighted_consensus


def test_exact_gain_survives_ill_conditioned_positive_metric():
    tiny = sys.float_info.min
    result = certify_affine_epi_jump_gain(
        "Reception", np.eye(2), [tiny, 1.0]
    )

    assert result.exact_quotient_energy_gain_upper_bound == 1
    assert result.energy_gain_bound_for_composition == 1.0
    assert result.supports_global_gain_theorem


def test_non_scalar_exact_gain_handles_near_singular_metric_without_svd(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise np.linalg.LinAlgError("diagnostic eigensolver unavailable")

    monkeypatch.setattr(np.linalg, "norm", unavailable)
    linear_map = np.asarray(
        [
            [0.75, 0.25, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [0.25, 0.0, 0.0, 0.75],
        ]
    )
    result = certify_affine_epi_jump_gain(
        "Reception", linear_map, [2.0**-900, 1.0, 2.0, 4.0]
    )

    assert result.sharp_quotient_operator_norm_estimate is None
    assert result.sharp_quotient_energy_gain_estimate is None
    assert result.exact_consensus_subspace_preservation
    assert 0 < result.exact_quotient_energy_gain_upper_bound
    assert (
        result.exact_quotient_energy_gain_upper_bound
        < result.exact_weighted_frobenius_energy_bound
    )
    assert math.isfinite(result.energy_gain_bound_for_composition)
    assert result.supports_global_gain_theorem


def test_fixed_flow_composes_against_raw_metric_before_normalized_display():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    flow = verify_heterogeneous_diffusion_stability(
        _set_state(graph, [1.0, -1.0], [1.0, 2.0 / 3.0])
    )
    np.testing.assert_array_equal(flow.metric_weights, [2.0, 3.0])
    jump = certify_affine_epi_jump_gain(
        "Reception", np.eye(2), flow.metric_weights, nodes=flow.nodes
    )

    result = compose_hybrid_epi_stability(flow, [jump], [0.0, 1.0])

    assert result.finite_horizon_disagreement_bound_certified
    np.testing.assert_allclose(result.normalized_metric_weights, [0.4, 0.6])
    assert result.exact_cumulative_jump_energy_gain_bound == (
        jump.exact_quotient_energy_gain_upper_bound
    )
    assert result.exact_flow_log_energy_decay_lower_bound == (
        Fraction.from_float(flow.certified_exponential_rate_lower_bound)
    )
    assert result.exact_net_log_energy_gain_upper_bound is not None


def test_switching_certificate_retains_raw_reference_metric():
    epi = [2.0, -1.0, 4.0]
    metric = np.array([1.0, 2.0, 4.0])
    path = nx.path_graph(3)
    cycle = nx.cycle_graph(3)
    for graph, scale in ((path, 1.0), (cycle, 2.0)):
        adjacency = nx.to_numpy_array(graph, nodelist=list(graph))
        strength = adjacency.sum(axis=1)
        _set_state(graph, epi, strength / (scale * metric))

    flow = verify_switching_diffusion_stability([path, cycle])

    assert flow.supports_exact_switching_theorem
    np.testing.assert_array_equal(flow.reference_metric_weights, metric)
    jump = certify_affine_epi_jump_gain(
        "Reception", np.eye(3), flow.reference_metric_weights, nodes=flow.nodes
    )
    hybrid = compose_hybrid_epi_stability(flow, [jump], [0.0, 1.0])
    assert hybrid.finite_horizon_disagreement_bound_certified


def test_hybrid_reception_converges_in_disagreement_but_not_to_initial_mean():
    flow = _path_flow()
    jump = _local_reception(flow)

    result = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 0.2], repeat_schedule=True
    )

    assert result.finite_horizon_disagreement_bound_certified
    assert result.disagreement_contracts_over_declared_horizon
    assert result.repeated_schedule_disagreement_convergence_certified
    assert not result.initial_weighted_mean_preserved
    assert not (
        result.repeated_schedule_initial_weighted_consensus_convergence_certified
    )


def test_composition_gain_quantization_has_exact_relative_inflation_bound():
    """The 32-bit policy has a rigorous per-factor and cumulative bound."""

    precision_bits = hybrid_stability._COMPOSITION_GAIN_SIGNIFICAND_BITS
    assert precision_bits == 32
    relative_limit = Fraction(
        (1 << (precision_bits - 1)) + 1,
        1 << (precision_bits - 1),
    )
    precise_gains = (
        Fraction(1, 3),
        Fraction((1 << 80) + 1, 1 << 77),
        Fraction((1 << 80) - 1, 1 << 140),
        Fraction(1 << 200, 3),
        Fraction(999_999_937, 1_000_000_007),
    )
    composition_factors = tuple(
        hybrid_stability._bounded_composition_gain_factor(gain)
        for gain in precise_gains
    )

    # For g in [2**e, 2**(e+1)), the exact ceiling error is below one
    # quantum, 2**(e-(p-1)) <= g*2**(-(p-1)).
    for precise, factor in zip(precise_gains, composition_factors):
        assert precise <= factor
        assert factor < precise * relative_limit

    precise_product = math.prod(precise_gains, start=Fraction(1))
    composition_product = math.prod(composition_factors, start=Fraction(1))
    assert precise_product <= composition_product
    assert composition_product < precise_product * relative_limit ** len(
        precise_gains
    )


def test_hybrid_log_composition_bounds_transcendental_input_complexity(
    monkeypatch,
):
    flow = _path_flow()
    jump = _local_reception(flow)
    precise_gain = jump.exact_quotient_energy_gain_upper_bound
    binary64_bits = sys.float_info.mant_dig
    assert max(
        precise_gain.numerator.bit_length(),
        precise_gain.denominator.bit_length(),
    ) > binary64_bits
    composition_factor = hybrid_stability._bounded_composition_gain_factor(
        precise_gain
    )
    assert composition_factor >= precise_gain

    observed_log_bit_lengths = []
    observed_exp_inputs = []
    exact_log_series = exact_time.atanh_log_bounds
    exact_exp_series = exact_time.exp_unit_bounds

    def bounded_log_series(value):
        bit_length = max(
            value.numerator.bit_length(), value.denominator.bit_length()
        )
        observed_log_bit_lengths.append(bit_length)
        assert bit_length <= hybrid_stability._COMPOSITION_GAIN_SIGNIFICAND_BITS
        return exact_log_series(value)

    def bounded_exp_series(value):
        observed_exp_inputs.append(value)
        assert value == Fraction.from_float(float(value))
        assert value.numerator.bit_length() <= binary64_bits
        assert value.denominator.bit_length() <= 1075
        return exact_exp_series(value)

    monkeypatch.setattr(exact_time, "atanh_log_bounds", bounded_log_series)
    monkeypatch.setattr(exact_time, "exp_unit_bounds", bounded_exp_series)
    result = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 0.2], repeat_schedule=True
    )

    assert observed_log_bit_lengths
    assert observed_exp_inputs
    assert result.exact_cumulative_jump_energy_gain_bound == precise_gain
    assert result.exact_log_composition_gain_factors == (composition_factor,)
    assert result.disagreement_contracts_over_declared_horizon


def test_uniform_translation_can_contract_disagreement_while_mean_drifts():
    flow = _path_flow()
    jump = certify_affine_epi_jump_gain(
        "Emission",
        np.eye(3),
        flow.metric_weights,
        offset=[0.1, 0.1, 0.1],
        nodes=flow.nodes,
    )

    result = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 0.6], repeat_schedule=True
    )

    assert jump.exact_consensus_subspace_preservation
    assert not jump.exact_weighted_mean_preservation
    assert result.repeated_schedule_disagreement_convergence_certified
    assert not result.initial_weighted_mean_preserved
    assert not (
        result.repeated_schedule_initial_weighted_consensus_convergence_certified
    )


def test_zero_gain_word_and_nonrepeated_schedule_have_honest_conclusions():
    flow = _path_flow()
    jump = certify_affine_epi_jump_gain(
        "Transition", np.zeros((3, 3)), flow.metric_weights, nodes=flow.nodes
    )

    repeated = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 0.1], repeat_schedule=True
    )
    finite_only = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 0.1], repeat_schedule=False
    )

    assert repeated.energy_multiplier_bound == 0.0
    assert repeated.repeated_schedule_disagreement_convergence_certified
    assert not repeated.initial_weighted_mean_preserved
    assert finite_only.repeated_schedule_disagreement_convergence_certified is None
    assert (
        finite_only.repeated_schedule_initial_weighted_consensus_convergence_certified
        is None
    )
    assert finite_only.asymptotic_disagreement_energy_decay_rate is None


def test_hybrid_rejects_reordered_nodes_and_near_but_distinct_metric():
    flow = _path_flow()
    reordered = certify_affine_epi_jump_gain(
        "Reception", np.eye(3), flow.metric_weights, nodes=tuple(reversed(flow.nodes))
    )
    near_metric = flow.metric_weights.copy()
    near_metric[0] = np.nextafter(near_metric[0], math.inf)
    near = certify_affine_epi_jump_gain(
        "Reception", np.eye(3), near_metric, nodes=flow.nodes
    )

    with pytest.raises(ValueError, match="node order"):
        compose_hybrid_epi_stability(flow, [reordered], [0.0, 1.0])
    with pytest.raises(ValueError, match="not exactly proportional"):
        compose_hybrid_epi_stability(
            flow, [near], [0.0, 1.0], tolerance=0.5
        )


def test_infinite_gain_jump_prevents_hybrid_certificate():
    flow = _path_flow()
    jump = certify_affine_epi_jump_gain(
        "Emission",
        np.eye(3),
        flow.metric_weights,
        offset=[1.0, 0.0, 0.0],
        nodes=flow.nodes,
    )

    result = compose_hybrid_epi_stability(
        flow, [jump], [0.0, 1.0], repeat_schedule=True
    )

    assert not result.finite_horizon_disagreement_bound_certified
    assert math.isinf(result.energy_multiplier_bound)
    assert not result.repeated_schedule_disagreement_convergence_certified


def test_finite_time_exponential_underflow_never_claims_exact_extinction():
    result = compose_hybrid_epi_stability(
        _path_flow(), [], [1000.0], repeat_schedule=False
    )

    assert math.isfinite(result.net_log_energy_gain_bound)
    assert result.energy_multiplier_bound == math.nextafter(0.0, math.inf)


def test_near_zero_log_budget_is_decided_by_the_exact_log_enclosure():
    flow = _path_flow()
    expansion = certify_affine_epi_jump_gain(
        "Expansion", 2.0 * np.eye(3), flow.metric_weights, nodes=flow.nodes
    )
    duration = math.nextafter(
        math.log(expansion.energy_gain_bound_for_composition)
        / flow.certified_exponential_rate_lower_bound,
        0.0,
    )

    result = compose_hybrid_epi_stability(
        flow, [expansion], [0.0, duration], repeat_schedule=True
    )

    assert abs(result.net_log_energy_gain_bound) < 1e-12
    assert result.log_contraction_decision_margin == 0.0
    assert not result.disagreement_contracts_over_declared_horizon
    assert not result.repeated_schedule_disagreement_convergence_certified


def test_extreme_finite_coefficients_do_not_produce_an_unsafe_finite_bound():
    huge = np.finfo(float).max
    result = certify_affine_epi_jump_gain(
        "Expansion", huge * np.eye(2), [1.0, 1.0]
    )

    assert result.exact_consensus_subspace_preservation
    assert result.exact_quotient_energy_gain_upper_bound == (
        Fraction.from_float(huge) ** 2
    )
    assert math.isinf(result.weighted_frobenius_energy_bound)
    assert math.isinf(result.energy_gain_bound_for_composition)
    assert not result.supports_global_gain_theorem
    assert result.sharp_quotient_energy_gain_estimate is None


def test_composition_rebuilds_a_jump_instead_of_trusting_replaced_proof_fields():
    flow = _path_flow()
    expansion = certify_affine_epi_jump_gain(
        "Expansion", 2.0 * np.eye(3), flow.metric_weights, nodes=flow.nodes
    )
    forged = replace(
        expansion,
        finite_global_energy_gain=True,
        energy_gain_bound_for_composition=0.0,
    )

    result = compose_hybrid_epi_stability(
        flow, [forged], [0.0, 0.1], repeat_schedule=True
    )

    assert result.jumps[0] is not forged
    assert result.jumps[0].energy_gain_bound_for_composition == 4.0
    assert result.energy_multiplier_bound > 1.0
    assert not result.repeated_schedule_disagreement_convergence_certified


def test_composition_rejects_replaced_or_mutated_flow_proof_fields():
    flow = _path_flow()
    replaced = replace(
        flow,
        certified_exponential_rate_lower_bound=1e300,
        is_certified=True,
    )

    with pytest.raises(ValueError, match="proof fields were replaced or mutated"):
        compose_hybrid_epi_stability(replaced, [], [1.0])

    flow.metric_weights.setflags(write=True)
    flow.metric_weights[0] = 1000.0
    with pytest.raises(ValueError, match="proof fields were replaced or mutated"):
        compose_hybrid_epi_stability(flow, [], [1.0])


@pytest.mark.parametrize(
    "changes",
    [
        {"exact_quotient_gap_lower_bound": Fraction(0)},
        {"exact_consensus_subspace_preservation": False},
        {"exact_uniform_fixed_point_preservation": False},
        {"exact_weighted_mean_preservation": False},
    ],
)
def test_composition_stamp_covers_every_fixed_flow_proof_fact(changes):
    flow = replace(_path_flow(), **changes)

    with pytest.raises(ValueError, match="proof fields were replaced or mutated"):
        compose_hybrid_epi_stability(flow, [], [1.0])


def test_switching_stamp_covers_per_regime_proof_facts():
    first = _set_state(nx.path_graph(3), [1.0, 0.0, -1.0], [1.0] * 3)
    second = nx.path_graph(3)
    nx.set_edge_attributes(second, 2.0, "weight")
    _set_state(second, [1.0, 0.0, -1.0], [1.0] * 3)
    flow = verify_switching_diffusion_stability([first, second])
    assert flow.supports_exact_switching_theorem
    tampered = replace(
        flow,
        exact_uniform_fixed_point_preservation_by_regime=(False, True),
    )

    with pytest.raises(ValueError, match="proof fields were replaced or mutated"):
        compose_hybrid_epi_stability(tampered, [], [1.0])


def test_flow_payloads_disable_ordinary_post_certification_writes():
    flow = _path_flow()

    assert isinstance(flow.nodes, tuple)
    with pytest.raises(ValueError, match="read-only"):
        flow.metric_weights[0] = 1000.0


def test_composition_uses_certified_quotient_rate_instead_of_eigensolver_estimate():
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 1, 2.0), (0, 2, 3.0), (1, 2, 1.0)])
    flow = verify_heterogeneous_diffusion_stability(
        _set_state(graph, [1.0, 2.0, 4.0], [2.0, 3.0, 31.0])
    )

    result = compose_hybrid_epi_stability(flow, [], [1.0])

    assert flow.certified_exponential_rate_lower_bound < flow.exponential_rate
    assert (
        Fraction.from_float(flow.certified_exponential_rate_lower_bound)
        <= 2 * flow.exact_quotient_gap_lower_bound
    )
    assert result.flow_energy_decay_rate == (
        flow.certified_exponential_rate_lower_bound
    )


def test_invalid_switching_flow_cannot_claim_initial_mean_preservation():
    path = _set_state(nx.path_graph(3), [1.0, 0.0, -1.0], [1.0] * 3)
    cycle = _set_state(nx.cycle_graph(3), [1.0, 0.0, -1.0], [1.0] * 3)
    flow = verify_switching_diffusion_stability([path, cycle])

    result = compose_hybrid_epi_stability(
        flow, [], [1.0], repeat_schedule=True
    )

    assert not result.flow_hypotheses_pass
    assert not result.initial_weighted_mean_preserved
    assert not result.repeated_schedule_initial_weighted_consensus_convergence_certified


def test_certified_rate_never_exceeds_the_actual_two_node_decay_rate():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=8.737355994060252e99)
    frequencies = (450110698248.3714, 8.638210136628038)
    flow = verify_heterogeneous_diffusion_stability(
        _set_state(graph, [1.0, -1.0], frequencies)
    )
    exact_actual_rate = 2 * sum(
        (Fraction.from_float(value) for value in frequencies), Fraction(0)
    )

    assert (
        Fraction.from_float(flow.certified_exponential_rate_lower_bound)
        <= exact_actual_rate
    )
    assert not flow.exact_weighted_mean_preservation


def test_initial_mean_claim_requires_exact_flow_mean_preservation():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=0.1)
    flow = verify_heterogeneous_diffusion_stability(
        _set_state(graph, [1.0, -1.0], [0.1, 2.3])
    )
    identity = certify_affine_epi_jump_gain(
        "Reception", np.eye(2), flow.metric_weights, nodes=flow.nodes
    )

    result = compose_hybrid_epi_stability(
        flow, [identity], [0.0, 1.0], repeat_schedule=True
    )

    assert flow.is_certified
    assert not flow.exact_weighted_mean_preservation
    assert not result.initial_weighted_mean_preserved
    assert not (
        result.repeated_schedule_initial_weighted_consensus_convergence_certified
    )


def test_duration_sum_overflow_is_reported_as_validation_error():
    flow = _path_flow()
    identity = certify_affine_epi_jump_gain(
        "Reception", np.eye(3), flow.metric_weights, nodes=flow.nodes
    )

    with pytest.raises(ValueError, match="total flow duration"):
        compose_hybrid_epi_stability(
            flow, [identity], [sys.float_info.max, sys.float_info.max]
        )


@pytest.mark.parametrize(
    "duration",
    [Fraction(-1, 10**1000), Fraction(1, 10**1000)],
)
def test_unrepresentable_fraction_durations_are_not_coerced_to_zero(duration):
    with pytest.raises(ValueError, match="finite nonnegative real"):
        compose_hybrid_epi_stability(_path_flow(), [], [duration])


def test_repeated_schedule_rejects_an_all_zero_flow_duration():
    with pytest.raises(ValueError, match="requires positive flow duration"):
        compose_hybrid_epi_stability(
            _path_flow(), [], [Fraction(0)], repeat_schedule=True
        )
