"""Exact resource accounting, conditional exclusions and retained responses."""

from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.passive_contact import (
    bound_passive_contact_consensus_source,
    bound_passive_contact_growth,
    bound_passive_contact_seed_response,
    evaluate_passive_contact_budget,
)


def _budget(**changes):
    values = dict(
        node_losses=(Q(1, 2), Q(1, 2), 0, 0),
        candidates=((1, 2),),
        contact_costs=(Q(1, 2),),
        port_shares=((1, 1),),
        storage_scale=1,
    )
    values.update(changes)
    return evaluate_passive_contact_budget(**values)


@pytest.mark.parametrize(
    "law,rate,storage_rate,accounting_rate",
    [
        ("reciprocal", Q(1, 3), Q(-5, 6), Q(-1, 2)),
        ("squared_reciprocal", Q(2, 9), Q(-8, 9), Q(-2, 3)),
    ],
)
def test_first_contact_exact_independent_values(
    law, rate, storage_rate, accounting_rate
):
    result = _budget(law=law)
    assert result.contact_rates == (rate,)
    assert result.storage_rate == storage_rate
    assert result.accounting_rate == accounting_rate
    assert result.unallocated_loss == Q(1, 2)


def test_shared_endpoint_cannot_fund_two_contacts_with_the_same_loss():
    # Naive duplicate allocation: two contacts each consume 9/10 of loss 1,
    # giving positive S_dot=4/5 despite each individually passive formula.
    assert -1 + 2 * Q(9, 10) == Q(4, 5)
    data = dict(
        node_losses=(1, 0, 0),
        candidates=((0, 1), (0, 2)),
        contact_costs=(9, 9),
        port_shares=((1, 1), (1, 1)),
    )
    with pytest.raises(ValueError, match="overspend"):
        _budget(**data)
    data["port_shares"] = ((Q(1, 2), 1), (Q(1, 2), 1))
    result = _budget(**data)
    assert result.contact_rates == (Q(1, 20), Q(1, 20))
    assert result.storage_rate == Q(-1, 10)
    assert result.accounting_rate == 0


def test_allocation_admission_is_not_waived_at_zero_current_loss():
    with pytest.raises(ValueError, match="overspend"):
        _budget(
            node_losses=(0, 0, 0),
            candidates=((0, 1), (0, 2)),
            contact_costs=(0, 0),
            port_shares=((1, 0), (1, 0)),
        )
    with pytest.raises(ValueError, match="overspend"):
        _budget(port_shares=((1 + Q(1, 10**60), 1),))


@pytest.mark.parametrize("law", ["reciprocal", "squared_reciprocal"])
def test_positive_allocation_at_zero_cost_has_finite_rate_and_exact_balance(law):
    result = _budget(
        law=law, node_losses=(0, 1, 0, 0), contact_costs=(0,), storage_scale=2
    )
    assert result.contact_rates == (Q(1, 2),)
    assert result.support_work == 0
    assert result.storage_rate == -1
    assert result.accounting_rate == 0


@pytest.mark.parametrize("law", ["reciprocal", "squared_reciprocal"])
def test_zero_allocations_and_empty_contact_set_preserve_unspent_loss(law):
    result = _budget(law=law, port_shares=((0, 0),))
    assert result.contact_rates == (0,)
    assert result.unallocated_loss == 1
    assert result.storage_rate == result.accounting_rate == -1
    empty = _budget(law=law, candidates=(), contact_costs=(), port_shares=())
    assert empty.contact_rates == ()
    assert empty.storage_rate == empty.accounting_rate == -1


@pytest.mark.parametrize("law", ["reciprocal", "squared_reciprocal"])
def test_joint_weight_scaling_and_form_clock_conversion(law):
    base = _budget(law=law)
    rho, form, clock = Q(7, 3), Q(5, 2), Q(11, 7)
    weight_scaled = _budget(
        law=law, node_losses=tuple(rho * d for d in base.node_losses)
    )
    assert weight_scaled.contact_rates == tuple(
        rho * rate for rate in base.contact_rates
    )
    assert weight_scaled.storage_rate == rho * base.storage_rate
    assert weight_scaled.accounting_rate == rho * base.accounting_rate
    converted = _budget(
        law=law,
        storage_scale=form**2,
        node_losses=tuple(form**2 * d / clock for d in base.node_losses),
        contact_costs=tuple(form**2 * c for c in base.contact_costs),
    )
    assert converted.contact_rates == tuple(rate / clock for rate in base.contact_rates)
    assert converted.storage_rate == form**2 * base.storage_rate / clock


@pytest.mark.parametrize("law", ["reciprocal", "squared_reciprocal"])
def test_node_relabeling_and_candidate_orientation_retain_the_ledger(law):
    # Old nodes 0,1,2,3 become 3,2,1,0, then reverse the candidate endpoints.
    base = _budget(law=law, node_losses=(1, 2, 3, 4), port_shares=((Q(1, 3), Q(2, 5)),))
    changed = _budget(
        law=law,
        node_losses=(4, 3, 2, 1),
        candidates=((1, 2),),
        port_shares=((Q(2, 5), Q(1, 3)),),
    )
    assert base.contact_rates == changed.contact_rates
    assert base.storage_rate == changed.storage_rate
    assert base.accounting_rate == changed.accounting_rate


def test_exact_resource_bound_and_strict_exclusion_do_not_certify_acquisition():
    options = dict(initial_storage=3, storage_scale=2, initial_weights=(Q(1, 2), 2))
    unknown = bound_passive_contact_growth(**options)
    assert unknown.sum_increase_upper_bound == Q(3, 2)
    assert unknown.individual_weight_upper_bounds == (2, Q(7, 2))
    assert unknown.resource_excluded is None
    boundary = bound_passive_contact_growth(**options, required_increases=(1, Q(1, 2)))
    assert boundary.resource_excluded is False  # Only not excluded by this bound.
    beyond = bound_passive_contact_growth(
        **options, required_increases=(1, Q(1, 2) + Q(1, 10**60))
    )
    assert beyond.resource_excluded is True
    zero = bound_passive_contact_growth(
        initial_storage=0,
        storage_scale=1,
        initial_weights=(0,),
        required_increases=(1,),
    )
    assert zero.resource_excluded is True


def test_rational_growth_budget_does_not_round_or_overflow_binary64():
    large = Q(10**400)
    result = bound_passive_contact_growth(
        initial_storage=large, storage_scale=3, initial_weights=(large,)
    )
    assert result.sum_increase_upper_bound == large / 3
    assert result.individual_weight_upper_bounds == (4 * large / 3,)


def test_storage_based_phase_margin_is_strict_and_failure_is_inconclusive():
    admitted = bound_passive_contact_growth(
        initial_storage=Q(1, 2),
        storage_scale=1,
        initial_weights=(0,),
        background_strengths=(1, 1, 2),
    )
    assert admitted.resultant_real_lower_bounds == (Q(1, 2), Q(1, 2), Q(3, 2))
    boundary = bound_passive_contact_growth(
        initial_storage=1,
        storage_scale=1,
        initial_weights=(0,),
        background_strengths=(1, 2),
    )
    assert boundary.resultant_real_lower_bounds == (0, 1)
    # A zero bound supplies no positive-margin certificate. It is not a
    # prediction that an actual phase resultant is zero.
    missing = bound_passive_contact_growth(
        initial_storage=1, storage_scale=1, initial_weights=(0,)
    )
    assert missing.resultant_real_lower_bounds is None


def test_low_storage_regular_margin_does_not_require_acute_edges_or_consensus():
    # Exact phasors (1,1,-1,-1) mean phases (0,0,pi,pi). The weak bridge
    # is antipodal, yet all weighted resultants are positive real. This is
    # a stationary conditional state, not a claimed attracting organization.
    phasors = (1, 1, -1, -1)
    edges = ((0, 1, Q(1)), (2, 3, Q(1)), (1, 2, Q(1, 4)))
    real_resultants = [Q(0)] * 4
    storage = Q(0)
    for i, j, weight in edges:
        cosine = phasors[i] * phasors[j]
        real_resultants[i] += weight * cosine
        real_resultants[j] += weight * cosine
        storage += weight * (1 - cosine)
    assert storage == Q(1, 2)
    assert real_resultants == [1, Q(3, 4), Q(3, 4), 1]
    domain = bound_passive_contact_growth(
        initial_storage=storage,
        storage_scale=1,
        initial_weights=(Q(1, 4),),
        background_strengths=(1,) * 4,
    )
    assert all(
        0 < lower <= actual
        for lower, actual in zip(
            domain.resultant_real_lower_bounds, real_resultants, strict=True
        )
    )
    rest = _budget(node_losses=(0,) * 4, contact_costs=(2,))
    assert rest.contact_rates == (0,)


@pytest.mark.parametrize(
    "changes,exception",
    [
        ({"node_losses": (True, 0, 0, 0)}, TypeError),
        ({"node_losses": (float("nan"), 0, 0, 0)}, ValueError),
        ({"node_losses": (-1, 0, 0, 0)}, ValueError),
        ({"contact_costs": (False,)}, TypeError),
        ({"contact_costs": (-1,)}, ValueError),
        ({"storage_scale": 0}, ValueError),
        ({"storage_scale": float("inf")}, ValueError),
        ({"candidates": ((True, 2),)}, TypeError),
        ({"candidates": ((1.0, 2),)}, TypeError),
        ({"candidates": ((1, 1),)}, ValueError),
        ({"candidates": ((1, 4),)}, ValueError),
        ({"candidates": {(1, 2)}}, TypeError),
        (
            {
                "candidates": ((1, 2), (2, 1)),
                "contact_costs": (0, 0),
                "port_shares": ((0, 0), (0, 0)),
            },
            ValueError,
        ),
        ({"port_shares": ((1,),)}, ValueError),
        ({"port_shares": ((-1, 0),)}, ValueError),
        ({"port_shares": ((True, 0),)}, TypeError),
        ({"contact_costs": ()}, ValueError),
        ({"law": "derived_from_coherence"}, ValueError),
    ],
)
def test_invalid_primitive_budget_data_rejected(changes, exception):
    with pytest.raises(exception):
        _budget(**changes)


@pytest.mark.parametrize(
    "changes,exception",
    [
        ({"initial_storage": True}, TypeError),
        ({"initial_storage": -1}, ValueError),
        ({"storage_scale": 0}, ValueError),
        ({"initial_weights": (float("inf"),)}, ValueError),
        ({"initial_weights": (-1,)}, ValueError),
        ({"required_increases": (0, 1)}, ValueError),
        ({"required_increases": (-1,)}, ValueError),
        ({"background_strengths": ()}, ValueError),
        ({"background_strengths": (0, 1)}, ValueError),
        ({"background_strengths": (True, 1)}, TypeError),
    ],
)
def test_invalid_growth_bound_data_rejected(changes, exception):
    values = dict(initial_storage=1, storage_scale=1, initial_weights=(0,))
    values.update(changes)
    with pytest.raises(exception):
        bound_passive_contact_growth(**values)


def _consensus_with_bridge(weight, pulse=0):
    graph = nx.Graph()
    graph.add_weighted_edges_from(((0, 1, 1), (2, 3, 1), (1, 2, weight)))
    for node in graph:
        graph.nodes[node].update(
            EPI=float(pulse if node == 1 else 0), theta=0.0, nu_f=1.0, delta_nfr=0.0
        )
    graph.graph["DNFR_WEIGHTS"] = {"epi": 1, "phase": 1, "vf": 0, "topo": 0}
    return graph


def test_retained_relative_contact_changes_response_despite_equal_nodal_state():
    # Actual native EPI kernel at phase consensus; this is NOT execution of
    # the fully weighted nonlinear phase law or an acquired contact history.
    responses = []
    for a, expected in ((1, Q(1, 16)), (3, Q(3, 32))):
        resting = capture_non_epi_forcing(_consensus_with_bridge(a))
        assert resting.full_kernel_pressure == (0,) * 4
        graph = _consensus_with_bridge(a, pulse=Q(1, 4))
        before = deepcopy(graph)
        response = capture_non_epi_forcing(graph)
        assert response.full_kernel_pressure[2] == expected
        assert nx.utils.graphs_equal(graph, before)
        responses.append(response.full_kernel_pressure[2])
    assert responses[0] != responses[1]


def _source_ball(**changes):
    values = dict(
        form_deviations=(0, Q(1, 32), 0, 0),
        phase_lift_deviations=(0, 0, 0, 0),
        background_strengths=(1, 1, 1, 1),
    )
    values.update(changes)
    return bound_passive_contact_consensus_source(**values)


def test_small_source_bound_and_onset_are_distinct_obligations():
    source = _source_ball()
    assert source.metric_upper_bound == Q(1, 512)
    assert source.strict_threshold == Q(1, 256)
    assert source.within_sufficient_ball
    # q=(-epsilon,epsilon,0,0), e=1/2: independent exact port loss.
    for law, rate in (
        ("reciprocal", Q(1, 2049)),
        ("squared_reciprocal", Q(2048, 2049**2)),
    ):
        onset = _budget(
            node_losses=(Q(1, 2048), Q(1, 2048), 0, 0),
            contact_costs=(Q(1, 2048),),
            law=law,
        )
        assert onset.contact_rates == (rate,)
    # Consensus at zero contact satisfies the ball but has no available loss.
    zero = _source_ball(form_deviations=(0, 0, 0, 0))
    assert zero.within_sufficient_ball
    assert _budget(node_losses=(0,) * 4).contact_rates == (0,)
    # The older amplitude-one source is not covered, not proved impossible.
    original = _source_ball(form_deviations=(0, 1, 0, 0))
    assert original.metric_upper_bound == 2
    assert not original.within_sufficient_ball


def test_source_ball_strict_boundary_is_decided_before_rounding():
    boundary = (Q(1, 32), Q(1, 32), 0, 0)
    result = _source_ball(form_deviations=boundary)
    assert result.metric_upper_bound == result.strict_threshold
    assert not result.within_sufficient_ball
    tiny = Q(1, 10**60)
    for sign in (-1, 1):
        perturbed = (boundary[0] + sign * tiny, *boundary[1:])
        result = _source_ball(form_deviations=perturbed)
        assert result.within_sufficient_ball is (sign < 0)


def test_source_metric_upper_bound_retains_lifts_and_mixed_term():
    result = _source_ball(
        form_deviations=(Q(-1, 100), 0, 0, 0),
        phase_lift_deviations=(Q(1, 100), 0, 0, 0),
    )
    assert result.metric_upper_bound == Q(71, 35000)
    assert result.within_sufficient_ball
    # An arbitrary lift is not silently wrapped into the common small chart.
    lifted = _source_ball(phase_lift_deviations=(0, 7, 0, 0))
    assert lifted.phase_lift_deviations[1] == 7
    assert not lifted.within_sufficient_ball


def test_source_bound_respects_common_background_scale_and_relabeling():
    base = _source_ball()
    scale = Q(7, 3)
    scaled = _source_ball(background_strengths=(scale,) * 4)
    assert scaled.metric_upper_bound == scale * base.metric_upper_bound
    assert scaled.strict_threshold == scale * base.strict_threshold
    assert scaled.within_sufficient_ball == base.within_sufficient_ball
    permuted = _source_ball(form_deviations=(0, 0, 0, Q(-1, 32)))
    assert permuted.metric_upper_bound == base.metric_upper_bound


def test_common_metric_identity_and_trapping_constants_symbolically():
    s = pytest.importorskip("sympy")
    e, omega = s.symbols("e omega", positive=True)
    mode = s.Matrix([[-e, -omega], [omega, 0]])
    metric = s.Matrix(
        [[1 / e, 1 / (2 * omega)], [1 / (2 * omega), 1 / e + e / (2 * omega**2)]]
    )
    assert s.simplify(mode.T * metric + metric * mode) == -s.eye(2)
    assert s.simplify(metric.det()) == 1 / e**2 + 1 / (4 * omega**2)
    selected = metric.subs({e: s.Rational(1, 2), omega: 1 / (2 * s.pi)})
    assert s.simplify((selected - s.eye(2)).det()) == 1
    # Independent conservative constants: ||P||<14, omega<1/6, R=1/16.
    assert 1 - 14 * (20 * Q(1, 6) + 2) / 256 == Q(17, 24)


@pytest.mark.parametrize(
    "changes,exception",
    [
        ({"form_deviations": (True, 0, 0, 0)}, TypeError),
        ({"phase_lift_deviations": (float("nan"), 0, 0, 0)}, ValueError),
        ({"form_deviations": (float("inf"), 0, 0, 0)}, ValueError),
        ({"form_deviations": {0, 1}}, TypeError),
        ({"phase_lift_deviations": (0, 0)}, ValueError),
        ({"background_strengths": ()}, ValueError),
        ({"background_strengths": (1, 0, 1, 1)}, ValueError),
        ({"background_strengths": (-1, 1, 1, 1)}, ValueError),
        ({"background_strengths": (True, 1, 1, 1)}, TypeError),
    ],
)
def test_source_ball_rejects_invalid_primitive_premises(changes, exception):
    with pytest.raises(exception):
        _source_ball(**changes)


def test_seed_retained_response_has_independent_exact_bounds():
    result = bound_passive_contact_seed_response(amplitude=Q(1, 32))
    assert result.leading_contact == Q(1, 4096)
    assert result.contact_error_bound == Q(68611, 1073741824)
    assert result.contact_bounds == (
        Q(193533, 1073741824),
        Q(330755, 1073741824),
    )
    assert result.susceptibility_bounds == (
        Q(193533, 1073935357),
        Q(330755, 1074072579),
    )
    assert result.receiver_gain_bounds == tuple(
        v / 2 for v in result.susceptibility_bounds
    )
    lower, upper = result.tangent_decay_rate_bounds
    assert 0 < lower < upper
    assert lower == result.susceptibility_bounds[0] / 20
    assert upper == result.susceptibility_bounds[1] / 15


def test_seed_bound_distinguishes_zero_source_and_sign_equivalence():
    positive = bound_passive_contact_seed_response(amplitude=Q(1, 64))
    negative = bound_passive_contact_seed_response(amplitude=Q(-1, 64))
    assert positive.amplitude == -negative.amplitude
    assert positive.contact_bounds == negative.contact_bounds
    assert positive.susceptibility_bounds == negative.susceptibility_bounds
    assert positive.tangent_decay_rate_bounds == negative.tangent_decay_rate_bounds
    zero = bound_passive_contact_seed_response(amplitude=0)
    assert zero.contact_bounds == zero.receiver_gain_bounds == (0, 0)
    assert zero.tangent_decay_rate_bounds == (0, 0)


def test_seed_bound_retains_tiny_exact_source_and_controlled_scaled_error():
    for amplitude in (Q(1, 32), Q(1, 128), Q(1, 10**100)):
        result = bound_passive_contact_seed_response(amplitude=amplitude)
        lower, upper = result.contact_bounds
        assert amplitude**2 / 6 < lower <= upper < amplitude**2 / 3
        assert result.contact_error_bound / amplitude**2 < Q(1, 12)
        assert result.tangent_decay_rate_bounds[0] > 0
    outer = bound_passive_contact_seed_response(amplitude=Q(1, 32))
    inner = bound_passive_contact_seed_response(amplitude=Q(1, 64))
    # The error normalized by amplitude squared shrinks with amplitude; a
    # uniform positive floor
    # across sources approaching zero is not inferred from positive bounds.
    assert inner.contact_error_bound / inner.amplitude**2 < (
        outer.contact_error_bound / outer.amplitude**2
    )
    assert inner.susceptibility_bounds[1] < outer.susceptibility_bounds[0]


def test_p4_response_gap_is_full_normalized_spectrum():
    s = pytest.importorskip("sympy")
    a, lam = s.symbols("a lam", positive=True)
    # Direct weighted graph construction, not a copied two-mode reduction.
    weights = s.Matrix([[0, 1, 0, 0], [1, 0, a, 0], [0, a, 0, 1], [0, 0, 1, 0]])
    degree = s.diag(1, 1 + a, 1 + a, 1)
    normalized = degree.inv() * (degree - weights)
    characteristic = s.factor((lam * s.eye(4) - normalized).det())
    expected = lam * (lam - 2) * (lam - a / (1 + a)) * (lam - (2 + a) / (1 + a))
    assert s.simplify(characteristic - expected) == 0
    # Rational pi brackets used for the tangent rate: 3<pi<sqrt(10).
    assert Q(11, 15) ** 2 < Q(5, 9)
    assert Q(3, 5) < Q(4, 5) ** 2


@pytest.mark.parametrize(
    "value,exception",
    [
        (True, TypeError),
        (float("inf"), ValueError),
        (float("nan"), ValueError),
        (Q(1, 32) + Q(1, 10**60), ValueError),
        (-Q(1, 32) - Q(1, 10**60), ValueError),
        (1, ValueError),
    ],
)
def test_seed_response_rejects_invalid_or_unproved_sources(value, exception):
    with pytest.raises(exception):
        bound_passive_contact_seed_response(amplitude=value)
