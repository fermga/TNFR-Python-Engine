"""Independent finite-time reduction, preparation and arithmetic controls."""

import pickle
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_reduction import bound_sine_slow_phase
from tnfr.sdk import export_to_json
from tnfr.utils.io import json_loads

SIGMA = Q(1, 16)
CAPACITY = (Q(1), Q(3, 2), Q(2), Q(5, 4))
PROFILE = (Q(1, 2), Q(-3, 4), Q(5, 4), Q(-1))
PHASE = (Q(1, 8), Q(-1, 4), Q(1, 2), Q(-3, 8))


def _model(ratio=16):
    return RelationalExchangeModel(
        2, epi_weight=ratio, phase_weight=1, phase_domain="regular"
    )


def _graph(model, *, phase=PHASE, form_shift=0, phase_shift=0):
    graph = nx.Graph(((0, 1), (1, 2), (2, 3), (0, 2)))
    e, w = map(Q, model.effective_weights)
    for i in graph:
        graph.nodes[i].update(
            EPI=form_shift + Q(model.storage_scale) * e / w * PROFILE[i],
            theta=phase_shift + phase[i],
            nu_f=CAPACITY[i],
        )
    graph.graph["GAMMA"] = {"type": "none"}
    return graph


def _source(ratio=16, **kwargs):
    model = _model(ratio)
    return bound_relational_sine_exchange(
        _graph(model, **kwargs), reference_model=model
    )


def _pattern():
    model = _model()
    return bound_relational_sine_pattern(
        _graph(model),
        reference_model=model,
        reference_node=0,
        form_error_bounds=(Q(1, 64),) * 4,
        phase_error_bounds=(Q(1, 128),) * 4,
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(box, value):
    assert _mp(box.lo) <= value <= _mp(box.hi)


@pytest.fixture(scope="module")
def bound():
    return bound_sine_slow_phase(_source(), slow_time=SIGMA)


def test_bound_constants_against_independent_high_precision_math(bound):
    source = bound.source
    with mp.workdps(95):
        e, w = map(_mp, source.reference_model.effective_weights)
        beta = _mp(source.reference_model.storage_scale)
        alpha, eta = w / (beta * mp.pi * e), w**2 / (beta * mp.pi**2 * e**2)
        metric = [_mp(d) / _mp(nu) for d, nu in zip(source.degrees, source.capacity)]
        center = sum(m * _mp(x) for m, x in zip(metric, source.epi)) / sum(metric)
        norm = mp.sqrt(
            sum(m * (_mp(x) - center) ** 2 for m, x in zip(metric, source.epi))
        )
        forcing = mp.sqrt(
            sum(_mp(nu * d) for nu, d in zip(source.capacity, source.degrees))
        )
        gap, ell = _mp(bound.weighted_gap_lower_bound), _mp(2 * max(CAPACITY))
        tau = _mp(SIGMA) / eta
        decay, growth = mp.exp(-gap * tau), mp.exp(ell * _mp(SIGMA))
        composite = (
            eta * (ell * alpha * norm + forcing) / (gap + eta * ell) * (growth - decay)
        )
        _contains(bound.scaled_time_bounds, tau)
        _contains(bound.horizon_bounds, tau / e)
        _contains(bound.feedback_strength_bounds, eta)
        _contains(bound.scaled_initial_norm_bounds, alpha * norm)
        _contains(bound.exponential_decay_bounds, decay)
        _contains(bound.exponential_growth_bounds, growth)
        assert composite <= _mp(bound.composite_phase_error_upper_bound)
        assert decay * alpha * norm + composite <= _mp(bound.phase_error_upper_bound)
        form_remainder = (w / (e * mp.pi)) * forcing * (1 - decay) / gap
        assert form_remainder <= _mp(bound.form_remainder_upper_bound)
    graph = nx.Graph(source.edges)
    matrix = nx.laplacian_matrix(graph, nodelist=source.nodes).toarray()
    roots = np.sqrt([float(nu / d) for nu, d in zip(source.capacity, source.degrees)])
    actual_gap = np.linalg.eigvalsh(roots[:, None] * matrix * roots[None, :])[1]
    assert 0 < float(bound.weighted_gap_lower_bound) <= actual_gap + 1e-14
    assert 0 < bound.composite_phase_error_upper_bound < Q(1, 20)


def _integrated_errors(source, bound):
    n = len(source.nodes)
    graph = nx.Graph(source.edges)
    adjacency = nx.to_numpy_array(graph, nodelist=source.nodes)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    k = np.array([float(nu / d) for nu, d in zip(source.capacity, source.degrees)])
    generator = k[:, None] * laplacian
    metric = 1 / k
    e, w = source.reference_model.effective_weights
    beta = source.reference_model.storage_scale
    alpha, eta = w / (beta * np.pi * e), w**2 / (beta * np.pi**2 * e**2)
    x0, theta0 = np.array(source.epi, dtype=float), np.array(source.phase, dtype=float)
    centered = x0 - np.dot(metric, x0) / metric.sum()
    z0 = alpha * centered

    def f(theta):
        return k * (adjacency * np.sin(theta[None, :] - theta[:, None])).sum(axis=1)

    def field(_time, state):
        z, theta, psi = state[:n], state[n : 2 * n], state[2 * n :]
        rate = generator @ z
        return np.concatenate((-rate + eta * f(theta), rate, eta * f(psi)))

    end = float(bound.slow_time) / eta
    result = solve_ivp(
        field,
        (0, end),
        np.concatenate((z0, theta0, theta0 + z0)),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
    )
    assert result.success

    def norm(value):
        return np.sqrt(np.dot(metric, value**2))

    # Fixed samples cross-check a proved bound, not a validated solver enclosure.
    maximum_composite = 0.0
    for at in np.linspace(0, end, 33):
        z, theta, psi = np.split(result.sol(at), 3)
        transient = expm(-generator * at) @ z0
        error = norm(theta + transient - psi)
        maximum_composite = max(maximum_composite, error)
        assert error <= float(bound.uniform_composite_phase_error_upper_bound) + 2e-10
    z, theta, psi = np.split(result.y[:, -1], 3)
    transient = expm(-generator * end) @ z0
    assert (
        norm(theta + transient - psi)
        <= float(bound.composite_phase_error_upper_bound) + 2e-10
    )
    assert norm(theta - psi) <= float(bound.phase_error_upper_bound) + 2e-10
    assert norm(theta + z - psi) <= float(bound.joint_error_upper_bound) + 2e-10
    assert (
        norm((z - transient) / alpha) <= float(bound.form_remainder_upper_bound) + 2e-10
    )
    assert norm(z / alpha) <= float(bound.form_norm_upper_bound) + 2e-10

    def potential(phase):
        return sum(1 - np.cos(phase[j] - phase[i]) for i, j in graph.edges)

    assert (
        abs(potential(theta) - potential(psi))
        <= float(bound.phase_potential_error_upper_bound) + 2e-10
    )
    return maximum_composite


@pytest.mark.parametrize("ratio", (8, 16))
def test_frozen_full_and_reference_flows_obey_composite_phase_and_form_bounds(ratio):
    source = _source(ratio)
    report = bound_sine_slow_phase(source, slow_time=SIGMA)
    assert _integrated_errors(source, report) > 1e-7


def test_fixed_scaled_preparation_keeps_its_original_storage_cost(bound):
    coarse = bound_sine_slow_phase(_source(8), slow_time=SIGMA)

    def scaled_numerators(report):
        e, w = map(Q, report.reference_model.effective_weights)
        return tuple(
            w
            * (x - report.weighted_form_mean)
            / (Q(report.reference_model.storage_scale) * e)
            for x in report.source.epi
        )

    # The exact z preparations agree; different outward evaluation paths
    # need not produce byte-identical interval endpoints for their norms.
    assert scaled_numerators(coarse) == scaled_numerators(bound)
    assert (
        coarse.scaled_initial_norm_bounds - bound.scaled_initial_norm_bounds
    ).abs_max < Q(1, 10**30)
    assert bound.initial_storage_bounds.lo > coarse.initial_storage_bounds.hi
    ratio = (
        bound.composite_phase_error_upper_bound
        / coarse.composite_phase_error_upper_bound
    )
    assert Q(1, 4) < ratio < Q(26, 100)


def test_zero_horizon_preserves_initial_phase_mismatch(bound):
    initial = bound_sine_slow_phase(bound.source, slow_time=0)
    assert initial.scaled_time_bounds == initial.horizon_bounds == I(0)
    assert (
        initial.composite_phase_error_upper_bound
        == initial.scaled_form_remainder_upper_bound
        == 0
    )
    assert initial.form_remainder_upper_bound == 0
    assert initial.phase_error_upper_bound == initial.scaled_initial_norm_bounds.hi > 0
    for theta, box in zip(bound.source.phase, initial.reference_initial_phase_bounds):
        assert box.lo != theta or box.hi != theta


def test_global_bound_does_not_require_flat_or_acute_initial_phase():
    source = _source(phase=(0, 2, -1, 3))
    report = bound_sine_slow_phase(source, slow_time=SIGMA)
    assert report.reference_initial_phase_bounds is not None
    assert _integrated_errors(source, report) > 0


def test_common_origins_leave_errors_and_centered_reference_unchanged(bound):
    shifted = bound_sine_slow_phase(
        _source(form_shift=Q(17, 2), phase_shift=Q(-9, 4)), slow_time=SIGMA
    )
    assert shifted.phase_error_upper_bound == bound.phase_error_upper_bound
    assert shifted.form_norm_upper_bound == bound.form_norm_upper_bound
    assert (
        shifted.centered_reference_initial_phase_bounds
        == bound.centered_reference_initial_phase_bounds
    )
    assert shifted.weighted_form_mean == bound.weighted_form_mean + Q(17, 2)
    assert shifted.weighted_phase_mean == bound.weighted_phase_mean - Q(9, 4)


def test_relative_family_uses_memberwise_references_and_original_residuals():
    pattern = _pattern()
    report = pattern.bound_slow_phase(slow_time=SIGMA)
    assert report.reference_initial_phase_bounds is None
    assert report.weighted_form_mean is report.weighted_phase_mean is None
    assert "memberwise" in report.reference_scope
    source = _source()
    for signs in ((1, -1, 1, -1), (-1, 1, -1, 1)):
        member = replace(
            source,
            epi=tuple(x + sign * Q(1, 64) + 7 for x, sign in zip(source.epi, signs)),
            phase=tuple(
                t - sign * Q(1, 128) - 3 for t, sign in zip(source.phase, signs)
            ),
        )
        _integrated_errors(member, report)
    forged = replace(pattern, relative_form_bounds=(I(100),) * 4, storage_bounds=I(0))
    unchanged = forged.bound_slow_phase(slow_time=SIGMA)
    assert unchanged.initial_storage_bounds == report.initial_storage_bounds
    assert unchanged.phase_error_upper_bound == report.phase_error_upper_bound


@pytest.mark.parametrize("value", (True, -1, float("inf"), float("nan")))
def test_invalid_slow_clock_rejects(value):
    with pytest.raises((TypeError, ValueError)):
        bound_sine_slow_phase(_source(), slow_time=value)


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity", ()),
        ("capacity", (0, 1, 1, 1)),
        ("law", "another_law"),
        ("epi", (True, 0, 0, 0)),
    ),
)
def test_invalid_consumed_source_cannot_receive_a_bound(field, value):
    with pytest.raises((TypeError, ValueError)):
        bound_sine_slow_phase(replace(_source(), **{field: value}), slow_time=SIGMA)


def test_zero_loss_and_forecast_sources_are_outside_this_preparation_contract():
    source = _source()
    model = RelationalExchangeModel(
        2, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    with pytest.raises(ValueError, match="positive form loss"):
        bound_sine_slow_phase(replace(source, reference_model=model), slow_time=SIGMA)
    with pytest.raises(TypeError):
        bound_sine_slow_phase(object(), slow_time=SIGMA)


def test_growth_budget_and_decay_tail_preserve_the_declared_slow_horizon():
    source = _source()
    with pytest.raises(ValueError, match="4096"):
        bound_sine_slow_phase(source, slow_time=1025)
    # Positive exponent100 would have a zero lower endpoint if a negative
    # exponential were rounded to dyadic128 before taking its reciprocal.
    large = bound_sine_slow_phase(source, slow_time=25)
    with mp.workdps(95):
        _contains(large.exponential_growth_bounds, mp.exp(100))
    assert large.decay_tail_enclosure_used
    assert large.slow_time == 25 and large.horizon_bounds.lo > 25
    assert large.exponential_decay_bounds.lo == 0 < large.exponential_decay_bounds.hi


def test_tiny_positive_ratio_is_not_rejected_when_eta_rounds_down_to_zero():
    model = RelationalExchangeModel(
        2, epi_weight=1, phase_weight=2.0**-200, phase_domain="regular"
    )
    graph = _graph(model)
    source = bound_relational_sine_exchange(graph, reference_model=model)
    report = bound_sine_slow_phase(source, slow_time=SIGMA)
    assert report.feedback_strength_bounds.lo == 0 < report.feedback_strength_bounds.hi
    assert report.scaled_time_bounds.lo > 2**390
    assert report.decay_tail_enclosure_used
    assert report.phase_error_upper_bound > 0


def test_sdk_export_and_source_provenance_are_detached(tmp_path):
    model = _model()
    graph = _graph(model)

    def state():
        return pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
            protocol=5,
        )

    before = state()
    source = bound_relational_sine_exchange(graph, reference_model=model)
    report = bound_sine_slow_phase(source, slow_time=SIGMA)
    assert report.source is source
    assert state() == before
    path = tmp_path / "slow-phase.json"
    export_to_json(report, path)
    payload = json_loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tnfr.relational-sine-slow-phase.v1"
    assert payload["report"]["slow_time"] == {"numerator": 1, "denominator": 16}
