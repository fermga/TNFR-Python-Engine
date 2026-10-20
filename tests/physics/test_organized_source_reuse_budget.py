"""Independent loss, weighted-mean and finite-sum checks without events or flow."""

from dataclasses import replace
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_reuse_budget as owner
from tnfr.physics._organized_source_port import _derive_organized_source_port


@pytest.fixture(scope="module")
def factors():
    source = canonical_glyph_factor_defaults()
    return {
        "reception_mix": Q(source["EN_mix"]),
        "silence_capacity_factor": Q(source["SHA_vf_factor"]),
    }


def _budget(factors, **overrides):
    parameters = dict(
        **factors,
        initial_mean_bound=Q(1, 30),
        energy_budget=Q(1, 100),
        word_count=56,
    )
    parameters.update(overrides)
    return owner._assess_organized_source_reuse_budget(**parameters)


def _mean(form, capacity):
    weights = (1 / capacity, 1 / capacity) + (Q(1),) * 43
    return sum((weight * x for weight, x in zip(weights, form)), Q(0)) / sum(
        weights, Q(0)
    )


def test_support_diameter_from_independent_shortest_paths(factors):
    geometry = _derive_organized_source_port()
    graph = nx.Graph()
    graph.add_edges_from((a, b) for a, b, _ in geometry.weighted_edges)
    report = _budget(factors)
    assert nx.diameter(graph) == report.support_diameter == 7
    assert graph.degree(0) == graph.degree(1) == report.target_degree == 5


def test_exact_finite_sum_is_enclosed_without_a_collapsed_report_denominator(factors):
    report = _budget(factors)
    kappa = report.silence_capacity_factor
    exact_sum = sum((1 / (43 * kappa**j + 2) ** 2 for j in range(56)), Q(0))
    assert report.inverse_mean_denominator_squared_sum_bounds.contains(exact_sum)
    assert report.inverse_mean_denominator_squared_sum_bounds.width < Q(1, 10**32)
    assert report.final_pair_capacity == kappa**56
    assert report.pair_weight_increment == 2 / (43 * kappa**56 + 2) - Q(2, 45)
    assert report.reception_mean_loss_coefficient == (
        2 * report.reception_step / (1 - 2 * report.reception_step)
    )


def _mp(value):
    return mpmath.mpf(value.numerator) / value.denominator


def test_high_precision_oracle_and_coarse_56_word_domain(factors):
    report = _budget(factors)
    with mpmath.workdps(80):
        beta, kappa = map(_mp, (report.reception_mix, report.silence_capacity_factor))
        a, energy = beta / 5, _mp(report.energy_budget)
        summed = mpmath.fsum(1 / (43 * kappa**j + 2) ** 2 for j in range(56))
        radius = mpmath.sqrt(14 * energy)
        en = mpmath.sqrt(energy * 2 * a / (1 - 2 * a) * summed)
        sha = (2 / (43 * kappa**56 + 2) - mpmath.mpf(2) / 45) * radius
        total = _mp(report.initial_mean_bound) + en + sha + radius
        pressure = (mpmath.sqrt(10 * energy) + 5 / (1023 * mpmath.pi)) / 4
        assert _mp(report.form_absolute_upper_bound) >= total
        assert _mp(report.form_absolute_upper_bound) - total < mpmath.mpf("1e-33")
        assert _mp(report.reception_pressure_absolute_upper_bound) >= pressure
    assert report.form_absolute_upper_bound < Q(107, 120) < Q(9, 10)
    assert report.reception_pressure_absolute_upper_bound < Q(81, 1000)
    assert report.unit_form_rails_certified


@pytest.mark.parametrize("capacity", (Q(1), Q(1, 7), Q(3)))
def test_exact_pair_mean_jumps_and_correlated_event_loss(factors, capacity):
    geometry = _derive_organized_source_port()
    form = tuple(Q((7 * node) % 17 - 8, 128) for node in range(45))
    gradient = [Q(0)] * 45
    for a, b, _ in geometry.weighted_edges:
        gradient[a] += form[a] - form[b]
        gradient[b] += form[b] - form[a]
    a = factors["reception_mix"] / 5
    successor = list(form)
    for node in (0, 1):
        successor[node] -= a * gradient[node]
    q_sum, q_difference = gradient[0] + gradient[1], gradient[0] - gradient[1]
    before = sum(
        ((form[b] - form[a_]) ** 2 / 2 for a_, b, _ in geometry.weighted_edges),
        Q(0),
    )
    after = sum(
        (
            (successor[b] - successor[a_]) ** 2 / 2
            for a_, b, _ in geometry.weighted_edges
        ),
        Q(0),
    )
    loss = a * ((1 - 2 * a) * q_sum**2 + (1 - 3 * a) * q_difference**2) / 2
    assert before - after == loss > 0
    assert _mean(successor, capacity) - _mean(form, capacity) == (
        -a * q_sum / (43 * capacity + 2)
    )
    next_capacity = capacity * factors["silence_capacity_factor"]
    pair_mean = (successor[0] + successor[1]) / 2
    other_mean = sum(successor[2:], Q(0)) / 43
    weight_change = 2 / (43 * next_capacity + 2) - 2 / (43 * capacity + 2)
    assert _mean(successor, next_capacity) - _mean(successor, capacity) == (
        weight_change * (pair_mean - other_mean)
    )


def test_joint_cauchy_loss_budget_and_telescope_not_per_event_energy(factors):
    report = _budget(
        factors, initial_pair_capacity=Q(1, 2), word_count=4, energy_budget=Q(1, 1000)
    )
    a, kappa = report.reception_step, report.silence_capacity_factor
    # Independent signed gradient sums; the same joint event losses pay for
    # these four shifts. No extra copy of the energy budget is used per word.
    sums = (Q(1, 100), Q(-1, 200), Q(1, 300), Q(-1, 400))
    losses = tuple(a * (1 - 2 * a) * value**2 / 2 for value in sums)
    exact_drift = sum(
        (
            abs(a * value / (43 * Q(1, 2) * kappa**j + 2))
            for j, value in enumerate(sums)
        ),
        Q(0),
    )
    assert sum(losses, Q(0)) < report.energy_budget
    assert exact_drift <= report.reception_mean_drift_bounds.hi
    increments = tuple(
        2 / (43 * Q(1, 2) * kappa ** (j + 1) + 2) - 2 / (43 * Q(1, 2) * kappa**j + 2)
        for j in range(4)
    )
    assert all(increment > 0 for increment in increments)
    assert sum(increments, Q(0)) == report.pair_weight_increment


def test_zero_words_and_zero_energy_have_no_invented_event_drift(factors):
    no_words = _budget(factors, word_count=0, initial_pair_capacity=Q(1, 3))
    assert no_words.inverse_mean_denominator_squared_sum_bounds.contains(0)
    assert no_words.reception_mean_drift_bounds.hi == 0
    assert no_words.silence_mean_drift_bounds.hi == 0
    assert no_words.pair_weight_increment == 0
    assert no_words.final_pair_capacity == Q(1, 3)
    zero_energy = _budget(factors, energy_budget=0)
    assert zero_energy.form_absolute_upper_bound == zero_energy.initial_mean_bound
    assert zero_energy.reception_mean_drift_bounds.hi == 0
    assert zero_energy.silence_mean_drift_bounds.hi == 0


def test_large_finite_budget_and_tiny_capacity_remain_outward_and_available(factors):
    report = _budget(
        factors,
        silence_capacity_factor=Q(1, 2),
        initial_pair_capacity=Q(1, 2**1100),
        word_count=4096,
    )
    assert report.final_pair_capacity == Q(1, 2**5196)
    assert report.inverse_mean_denominator_squared_sum_bounds.hi <= 1024
    assert report.inverse_mean_denominator_squared_sum_bounds.lo > Q(1023)
    assert not report.unit_form_rails_certified
    assert report.reception_pressure_absolute_upper_bound < Q(81, 1000)
    assert "finite_word_count_is_a_work_budget_not_capacity_or_grammar_admission" in (
        report.scope
    )


@pytest.mark.parametrize("name", ("reception_mix", "silence_capacity_factor"))
def test_exact_factor_association_rejects_unrepresented_fraction(factors, name):
    with pytest.raises(ValueError, match="exact represented runtime factor"):
        _budget(factors, **{name: Q(1, 3)})


@pytest.mark.parametrize(
    "name,value",
    (
        ("word_count", True),
        ("word_count", Q(1)),
        ("word_count", -1),
        ("word_count", 4097),
        ("initial_mean_bound", -1),
        ("energy_budget", Q(-1, 100)),
        ("energy_budget", Q(11, 1000)),
        ("initial_pair_capacity", 0),
        ("initial_pair_capacity", -1),
        ("silence_capacity_factor", 0),
        ("silence_capacity_factor", 1),
        ("reception_mix", 0),
        ("reception_mix", 2),
    ),
)
def test_domain_rejections(factors, name, value):
    with pytest.raises((TypeError, ValueError)):
        _budget(factors, **{name: value})


@pytest.mark.parametrize(
    "name",
    (
        "initial_mean_bound",
        "energy_budget",
        "initial_pair_capacity",
        "silence_capacity_factor",
        "reception_mix",
    ),
)
@pytest.mark.parametrize("value", (True, float("nan"), float("inf")))
def test_scalar_admission_before_geometry(factors, monkeypatch, name, value):
    def forbidden():
        pytest.fail("invalid primitives reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _budget(factors, **{name: value})


def test_changed_support_cannot_reuse_diameter_or_coefficients(factors, monkeypatch):
    geometry = _derive_organized_source_port()
    changed = replace(geometry, weighted_edges=geometry.weighted_edges[:-1])
    monkeypatch.setattr(owner, "_derive_organized_source_port", lambda: changed)
    with pytest.raises(ValueError, match="support contract"):
        _budget(factors)
