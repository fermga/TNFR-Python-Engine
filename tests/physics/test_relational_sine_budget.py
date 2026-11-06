"""Independent all-time Lyapunov and whole-preparation-budget controls."""

from dataclasses import replace
from fractions import Fraction as Q
from functools import partial

import mpmath as mp
import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_budget import certify_sine_budget_consensus
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry
from tnfr.physics.relational_sine_forecast import _sine_flow
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

MODEL = RelationalExchangeModel(
    1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _zero(value):
    assert value.lo <= 0 <= value.hi
    assert value.abs_max < Q(1, 10**25)


def _lyapunov_jet(graph, model, capacity, form, phase):
    """Differentiate W using the shared complete field, in its original clock."""
    n = len(graph)
    rows = flow_jets(
        tuple(I(v) for v in (*form, *phase, capacity[-1])),
        1,
        partial(
            _sine_flow,
            neighbors=tuple(tuple(graph[i]) for i in range(n)),
            visible_capacity=capacity[:-1],
            model=model,
        ),
    )
    rows = tuple(Jet(row) for row in rows)
    weights = tuple(Q(graph.degree(i)) / capacity[i] for i in range(n))
    total = sum(weights, Q(0))
    form_mean = (
        sum((m * row for m, row in zip(weights, rows[:n])), Jet.constant(0, 1)) / total
    )
    phase_mean = (
        sum((m * row for m, row in zip(weights, rows[n : 2 * n])), Jet.constant(0, 1))
        / total
    )
    e, w = map(Q, model.effective_weights)
    beta, pi = Q(model.storage_scale), pi_interval()
    alpha = (w / (beta * e)) / pi
    eta = (w**2 / (beta * e**2)) / pi**2
    z = tuple((row - form_mean) * alpha for row in rows[:n])
    theta = tuple(row - phase_mean for row in rows[n : 2 * n])
    lyapunov = sum(
        (m * ((t + v) ** 2 + v**2) / 2 for m, t, v in zip(weights, theta, z)),
        Jet.constant(0, 1),
    )
    edge_identity = I(0)
    residual_terms = []
    for i, j in graph.edges:
        delta = I(phase[j] - phase[i])
        zeta = z[j].coeffs[0] - z[i].coeffs[0]
        current = sin(delta)
        remainder = delta * current - eta * current**2
        residual_terms.append(remainder)
        edge_identity -= (zeta + eta * current) ** 2 + eta * remainder
    # Jets use t, whereas the Lyapunov theorem differentiates in tau=e*t.
    return lyapunov.coeffs[1] / e, edge_identity, eta, tuple(residual_terms)


@pytest.fixture(scope="module")
def geometry():
    return derive_phase_cycle_geometry(nx.cycle_graph(5))


@pytest.fixture(scope="module")
def bounded_family(geometry):
    return certify_sine_budget_consensus(
        geometry, reference_model=MODEL, capacity=(1,) * 5, form_storage_budget=160
    )


def test_lyapunov_identity_uses_both_rows_and_allows_nonacute_raw_phases():
    graph = nx.Graph(((0, 1), (1, 2), (2, 3), (0, 2)))
    model = RelationalExchangeModel(
        2, epi_weight=Q(3, 4), phase_weight=Q(1, 4), phase_domain="regular"
    )
    capacity = (Q(1), Q(3, 2), Q(2), Q(5, 4))
    form = (Q(1, 2), Q(-3, 4), Q(5, 4), Q(-1))
    phase = (Q(0), Q(2), Q(1), Q(-1, 2))
    derivative, independent, eta, residuals = _lyapunov_jet(
        graph, model, capacity, form, phase
    )
    assert pi_interval().hi / 2 < phase[1] - phase[0] < pi_interval().lo
    assert all(abs(phase[j] - phase[i]) < pi_interval().lo for i, j in graph.edges)
    assert eta.hi < 1 and all(term.lo > 0 for term in residuals)
    _zero(derivative - independent)
    assert derivative.hi < 0
    # Separate common origins do not enter the centered auxiliary function.
    shifted = _lyapunov_jet(
        graph,
        model,
        capacity,
        tuple(x + Q(17, 3) for x in form),
        tuple(t - Q(11, 7) for t in phase),
    )
    _zero(derivative - shifted[0])


def test_feedback_above_one_can_increase_W_without_consensus_instability():
    graph = nx.path_graph(2)
    model = RelationalExchangeModel(
        1, epi_weight=1, phase_weight=4, phase_domain="regular"
    )
    derivative, independent, eta, _ = _lyapunov_jet(
        graph,
        model,
        (Q(1), Q(1)),
        (Q(3, 8), Q(-3, 8)),
        (Q(-1, 4), Q(1, 4)),
    )
    _zero(derivative - independent)
    assert eta.lo > 1 and derivative.lo > 0
    # The actual nonuniform P2 tangent has stable eigenvalues even though this
    # particular W is not monotone. Failed proof admission is not instability.
    e, w = model.effective_weights
    tangent = np.array([[-2 * e, -2 * w / np.pi], [2 * w / np.pi, 0]])
    assert np.linalg.eigvals(tangent).real.max() < 0
    report = certify_sine_budget_consensus(
        derive_phase_cycle_geometry(graph),
        reference_model=model,
        capacity=(1, 1),
        form_storage_budget=1,
    )
    assert report.status == "unavailable" and not report.consensus_certified
    assert "feedback_strength_at_most_one_not_certified" in report.unresolved
    assert report.nonzero_winding_budget_lower_bound is None
    assert report.phase_edge_upper_bounds is report.form_norm_upper_bound is None


def test_whole_budget_bound_matches_independent_weighted_spectral_estimate(
    bounded_family,
):
    report = bounded_family
    assert report.status == "certified" and report.consensus_certified
    assert report.zero_winding_certified and report.acute_trapping_certified
    assert not report.stationary and report.unresolved == ()
    assert report.family.form_storage_budget == 160
    assert report.mobility == (Q(1, 2),) * 5 and report.metric_weights == (Q(2),) * 5
    with mp.workdps(95):
        gap = _mp(report.weighted_gap_lower_bound)
        assert 0 < gap <= (5 - mp.sqrt(5)) / 4
        eta = 1 / (1023**2 * mp.pi**2)
        assert (
            _mp(report.feedback_strength_bounds.lo)
            <= eta
            <= _mp(report.feedback_strength_bounds.hi)
        )
        initial_W = 2 * eta * 160 / gap
        assert initial_W <= _mp(report.initial_lyapunov_upper_bound)
        for bound, branch, acute in zip(
            report.candidate_phase_edge_upper_bounds,
            report.branch_margins,
            report.acute_margins,
        ):
            assert 2 * mp.sqrt(initial_W) <= _mp(bound) < mp.pi / 2
            assert _mp(branch.lo) <= mp.pi - _mp(bound) <= _mp(branch.hi)
            assert _mp(acute.lo) <= mp.pi / 2 - _mp(bound) <= _mp(acute.hi)
        necessary_budget = gap * mp.pi**4 * 1023**2 / 8
        assert 160 < _mp(report.nonzero_winding_budget_lower_bound) <= necessary_budget
        assert mp.sqrt(4 * 160 / gap) <= _mp(report.form_norm_upper_bound)
    assert report.phase_edge_upper_bounds == report.candidate_phase_edge_upper_bounds


def test_heterogeneous_capacities_and_storage_scale_keep_weighted_budget_constants():
    graph = nx.Graph(((0, 1), (1, 2), (2, 3), (0, 2)))
    capacity = (Q(1), Q(3, 2), Q(2), Q(5, 4))
    model = RelationalExchangeModel(
        2, epi_weight=Q(3, 4), phase_weight=Q(1, 4), phase_domain="regular"
    )
    report = certify_sine_budget_consensus(
        derive_phase_cycle_geometry(graph),
        reference_model=model,
        capacity=capacity,
        form_storage_budget=Q(3, 2),
    )
    assert report.consensus_certified
    metric = tuple(Q(graph.degree(i)) / nu for i, nu in enumerate(capacity))
    assert report.metric_weights == metric
    laplacian = nx.laplacian_matrix(graph, nodelist=range(4)).toarray()
    roots = np.sqrt([float(1 / value) for value in metric])
    actual_gap = np.linalg.eigvalsh(roots[:, None] * laplacian * roots[None, :])[1]
    assert 0 < float(report.weighted_gap_lower_bound) <= actual_gap + 1e-14
    with mp.workdps(95):
        gap = _mp(report.weighted_gap_lower_bound)
        eta = 1 / (18 * mp.pi**2)
        assert 2 * eta * mp.mpf("1.5") / (2 * gap) <= _mp(
            report.initial_lyapunov_upper_bound
        )
        kappa = max(1 / metric[i] + 1 / metric[j] for i, j in graph.edges)
        necessary_budget = 4 * gap * mp.pi**4 / (8 * _mp(kappa) / 9)
        assert _mp(report.nonzero_winding_budget_lower_bound) <= necessary_budget


def test_nonacute_candidate_strip_can_certify_consensus_without_acute_claim():
    report = certify_sine_budget_consensus(
        derive_phase_cycle_geometry(nx.path_graph(2)),
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
        capacity=(1, 1),
        form_storage_budget=4,
    )
    assert report.consensus_certified and report.zero_winding_certified
    assert not report.acute_trapping_certified
    assert report.branch_margins[0].lo > 0 and report.acute_margins[0].hi < 0
    with mp.workdps(95):
        assert mp.pi / 2 < 4 * mp.sqrt(2) / mp.pi < mp.pi
        assert 4 * mp.sqrt(2) / mp.pi <= _mp(report.phase_edge_upper_bounds[0])


def test_large_rescaled_preparation_remains_outside_the_obstruction(geometry):
    original_budget = 160 * 1023**2
    report = certify_sine_budget_consensus(
        geometry,
        reference_model=MODEL,
        capacity=(1,) * 5,
        form_storage_budget=original_budget,
    )
    assert report.status == "unavailable" and not report.consensus_certified
    assert not report.zero_winding_certified and not report.acute_trapping_certified
    assert "strict_raw_pi_strip_not_certified" in report.unresolved
    assert all(
        getattr(report, name) is None
        for name in (
            "lyapunov_upper_bound",
            "phase_norm_upper_bound",
            "scaled_form_norm_upper_bound",
            "form_norm_upper_bound",
            "phase_edge_upper_bounds",
        )
    )
    assert report.candidate_phase_edge_upper_bounds
    assert 0 < report.nonzero_winding_budget_lower_bound < original_budget
    # Reuse the published analytic producer once. No trajectory is rerun and
    # no preparation, law, horizon or numerical budget is optimized here.
    graph = nx.cycle_graph(5)
    graph.graph["GAMMA"] = {"type": "none"}
    for i in graph:
        graph.nodes[i].update(EPI=4092 * (i - 2), theta=0, nu_f=1)
    source = bound_relational_sine_exchange(graph, reference_model=MODEL)
    offsets = tuple(
        int((Q(j - i, 5) + Q(1, 2)) % 1 - Q(1, 2) - Q(j - i, 5))
        for i, j in geometry.edges
    )
    entrant = certify_sine_prepared_entry(
        source, scaled_time=100, edge_turn_offsets=offsets
    )
    assert entrant.initial_form_storage == original_budget
    assert entrant.admitted and entrant.initial_cycle_periods == (0,)
    assert entrant.capture.cycle_periods == (1,)


def test_zero_budget_is_stationary_even_when_feedback_certificate_is_unavailable(
    geometry,
):
    report = certify_sine_budget_consensus(
        geometry,
        reference_model=RelationalExchangeModel(
            1, epi_weight=1, phase_weight=4, phase_domain="regular"
        ),
        capacity=(1,) * 5,
        form_storage_budget=0,
    )
    assert report.feedback_strength_bounds.lo > 1
    assert report.stationary and report.consensus_certified
    assert report.zero_winding_certified
    assert report.acute_trapping_certified and report.unresolved == ()
    assert (
        report.lyapunov_upper_bound
        == report.form_norm_upper_bound
        == report.phase_norm_upper_bound
        == 0
    )
    assert report.phase_edge_upper_bounds == (Q(0),) * 5
    assert report.nonzero_winding_budget_lower_bound is None


def test_tiny_positive_ratio_keeps_certified_form_bound_without_interval_division(
    geometry,
):
    report = certify_sine_budget_consensus(
        geometry,
        reference_model=RelationalExchangeModel(
            1, epi_weight=1, phase_weight=1e-50, phase_domain="regular"
        ),
        capacity=(1,) * 5,
        form_storage_budget=1,
    )
    assert report.feedback_strength_bounds.lo == 0 < report.feedback_strength_bounds.hi
    assert report.consensus_certified and not report.stationary
    assert 0 < report.form_norm_upper_bound < 4
    assert report.initial_lyapunov_upper_bound > 0


@pytest.mark.parametrize("budget", (True, -1, float("nan"), float("inf")))
def test_invalid_budget_is_not_a_zero_or_stationary_family(geometry, budget):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_budget_consensus(
            geometry,
            reference_model=MODEL,
            capacity=(1,) * 5,
            form_storage_budget=budget,
        )


@pytest.mark.parametrize(
    "capacity",
    (
        (1,) * 4,
        (True, 1, 1, 1, 1),
        (0, 1, 1, 1, 1),
        (-1, 1, 1, 1, 1),
        (float("nan"), 1, 1, 1, 1),
    ),
)
def test_held_capacity_must_be_complete_finite_and_strictly_positive(
    geometry, capacity
):
    with pytest.raises((TypeError, ValueError)):
        certify_sine_budget_consensus(
            geometry,
            reference_model=MODEL,
            capacity=capacity,
            form_storage_budget=0,
        )


def test_wrong_law_and_tampered_geometry_cannot_certify_even_zero_budget(geometry):
    models = (
        None,
        RelationalExchangeModel(1),
        RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    for model in models:
        with pytest.raises((TypeError, ValueError)):
            certify_sine_budget_consensus(
                geometry,
                reference_model=model,
                capacity=(1,) * 5,
                form_storage_budget=0,
            )
    with pytest.raises(ValueError, match="derived fields"):
        certify_sine_budget_consensus(
            replace(geometry, cycle_rank=7),
            reference_model=MODEL,
            capacity=(1,) * 5,
            form_storage_budget=0,
        )


def test_sdk_export_retains_whole_family_and_exact_budget(bounded_family, tmp_path):
    direct = bounded_family.to_dict()
    generic = relational_report_to_dict(bounded_family)
    assert generic["schema"] == "tnfr.relational-report.v1"
    assert generic["report_type"] == "SineBudgetConsensus"
    assert generic["report"] == direct["report"]
    path = tmp_path / "budget-consensus.json"
    export_to_json(bounded_family, path)
    payload = json_loads(path.read_text(encoding="utf-8"))
    assert payload == direct
    assert payload["schema"] == "tnfr.relational-sine-budget-consensus.v1"
    report = payload["report"]
    assert report["family"]["form_storage_budget"] == {
        "numerator": 160,
        "denominator": 1,
    }
    assert report["status"] == "certified" and report["consensus_certified"] is True
    assert "source" not in report and "slow_time" not in report
