"""Prospective sine-law smoothness from independent edge derivatives.

The oracle differentiates the complete fine-edge law through third order.
No source preparation, prior sampling producer or trajectory is evaluated.
"""

import json
from copy import copy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_sampling import bound_sine_sampling_smoothness
from tnfr.sdk import export_to_json

MODEL = RelationalExchangeModel(2, phase_domain="regular")


def _stored_model():
    model = copy(MODEL)
    for field, value in (
        ("epi_weight", Q(7, 19)),
        ("phase_weight", Q(5, 17)),
        ("storage_scale", Q(11, 13)),
    ):
        object.__setattr__(model, field, value)
    return model


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _edge_oracle(graph, forms, phases, capacities, model):
    """Differentiate currents on all edges without using any bound formula."""
    with mp.workdps(90):
        e, w = map(_mp, model.effective_weights)
        beta = _mp(model.storage_scale)
        x = dict(zip(graph, map(_mp, forms)))
        theta = dict(zip(graph, map(_mp, phases)))
        nu = dict(zip(graph, map(_mp, capacities)))
        form_rate, phase_rate = {}, {}
        for i in graph:
            gradient = sum(x[i] - x[j] for j in graph[i])
            current = sum(mp.sin(theta[j] - theta[i]) for j in graph[i])
            form_rate[i] = (
                nu[i] * (-e * gradient + w * current / mp.pi) / graph.degree[i]
            )
            phase_rate[i] = nu[i] * w * gradient / (beta * mp.pi * graph.degree[i])
        form_acceleration, phase_acceleration = {}, {}
        for i in graph:
            gradient_rate = sum(form_rate[i] - form_rate[j] for j in graph[i])
            current_rate = sum(
                mp.cos(theta[j] - theta[i]) * (phase_rate[j] - phase_rate[i])
                for j in graph[i]
            )
            form_acceleration[i] = (
                nu[i]
                * (-e * gradient_rate + w * current_rate / mp.pi)
                / graph.degree[i]
            )
            phase_acceleration[i] = (
                nu[i] * w * gradient_rate / (beta * mp.pi * graph.degree[i])
            )
        form_third, phase_third = {}, {}
        for i in graph:
            gradient_acceleration = sum(
                form_acceleration[i] - form_acceleration[j] for j in graph[i]
            )
            current_acceleration = sum(
                -mp.sin(theta[j] - theta[i]) * (phase_rate[j] - phase_rate[i]) ** 2
                + mp.cos(theta[j] - theta[i])
                * (phase_acceleration[j] - phase_acceleration[i])
                for j in graph[i]
            )
            form_third[i] = (
                nu[i]
                * (-e * gradient_acceleration + w * current_acceleration / mp.pi)
                / graph.degree[i]
            )
            phase_third[i] = (
                nu[i] * w * gradient_acceleration / (beta * mp.pi * graph.degree[i])
            )
        return {
            "form_speed_bound": form_rate,
            "phase_speed_bound": phase_rate,
            "form_acceleration_bound": form_acceleration,
            "phase_acceleration_bound": phase_acceleration,
            "form_third_derivative_bound": form_third,
            "phase_third_derivative_bound": phase_third,
        }


def _arguments(**changes):
    arguments = dict(
        reference_model=MODEL,
        form_diameter_bound=3,
        capacity_ceiling=2,
        window_start=0,
        window_end=Q(1, 4),
    )
    arguments.update(changes)
    return arguments


@pytest.mark.parametrize(
    "graph",
    (
        nx.path_graph(4),
        nx.Graph(((0, 1), (1, 2), (2, 0), (2, 3), (3, 4))),
        nx.complete_bipartite_graph(2, 3),
    ),
    ids=("path", "triangle_with_tail", "bipartite"),
)
@pytest.mark.parametrize(
    "model",
    (
        MODEL,
        RelationalExchangeModel(3, epi_weight=0, phase_domain="regular"),
        _stored_model(),
    ),
    ids=("dissipative", "zero_form_diffusion", "authoritative_stored_coefficients"),
)
def test_whole_class_bounds_contain_independent_full_edge_third_derivatives(
    graph, model
):
    size = len(graph)
    forms = (Q(-2), Q(1, 3), Q(7, 4), Q(-5, 7), Q(1, 11))[:size]
    phases = (Q(-7, 2), Q(9, 4), Q(15, 8), Q(-17, 6), Q(5))[:size]
    capacities = (Q(0), Q(1, 3), Q(7, 4), Q(1), Q(9, 8))[:size]
    report = bound_sine_sampling_smoothness(
        **_arguments(
            reference_model=model,
            form_diameter_bound=max(forms) - min(forms),
            capacity_ceiling=max(capacities),
            window_start=5,
            window_end=Q(21, 4),
        )
    )
    oracle = _edge_oracle(graph, forms, phases, capacities, model)
    with mp.workdps(90):
        for bound_name, rows in oracle.items():
            bound = _mp(getattr(report, bound_name))
            assert all(abs(value) <= bound for value in rows.values())
            assert rows[0] == 0  # All derivatives of a zero-capacity row freeze.
        assert any(
            abs(value) > 0 for value in oracle["form_third_derivative_bound"].values()
        )


def test_phase_forcing_can_expand_zero_form_diameter_at_the_maximum_principle_limit():
    with mp.workdps(90):
        angle = Q(mp.nstr(mp.pi / 2, 85))
        capacity = Q(7, 3)
        # At equal forms the dissipative row vanishes, even for strong diffusion.
        model = RelationalExchangeModel(
            1, epi_weight=19, phase_weight=1, phase_domain="regular"
        )
        rows = _edge_oracle(
            nx.path_graph(2), (0, 0), (0, angle), (capacity,) * 2, model
        )
        velocities = rows["form_speed_bound"]
        initial_diameter_rate = max(velocities.values()) - min(velocities.values())
        report = bound_sine_sampling_smoothness(
            **_arguments(
                reference_model=model, form_diameter_bound=0, capacity_ceiling=capacity
            )
        )
        permitted_rate = _mp(report.form_diameter_upper_bound / Q(1, 4))
        assert 0 < initial_diameter_rate <= permitted_rate
        assert permitted_rate - initial_diameter_rate < mp.mpf("1e-35")
        assert report.form_diameter_upper_bound > 0


def test_common_form_and_phase_shifts_do_not_require_absolute_state_bounds():
    graph = nx.path_graph(4)
    forms, phases, capacities = (
        (Q(-1), Q(1, 2), Q(2), Q(0)),
        (Q(0), Q(4), Q(-2), Q(1)),
        (Q(1), Q(0), Q(2), Q(1, 3)),
    )
    shifted_forms = tuple(value + 10**25 for value in forms)
    shifted_phases = tuple(value - 10**20 for value in phases)
    before = _edge_oracle(graph, forms, phases, capacities, MODEL)
    after = _edge_oracle(graph, shifted_forms, shifted_phases, capacities, MODEL)
    reports = tuple(
        bound_sine_sampling_smoothness(
            **_arguments(form_diameter_bound=max(values) - min(values))
        )
        for values in (forms, shifted_forms)
    )
    assert reports[0] == reports[1]
    with mp.workdps(90):
        for name in before:
            for node in graph:
                assert abs(before[name][node] - after[name][node]) < mp.mpf("1e-60")
                assert abs(after[name][node]) <= _mp(getattr(reports[1], name))


def test_zero_capacity_class_is_constant_for_all_forms_and_phases():
    report = bound_sine_sampling_smoothness(
        **_arguments(form_diameter_bound=6, capacity_ceiling=0, window_end=1000)
    )
    assert report.form_diameter_upper_bound == 6
    oracle = _edge_oracle(
        nx.complete_graph(3), (3, -1, 5), (1, -4, 8), (0, 0, 0), MODEL
    )
    for field, rows in oracle.items():
        assert getattr(report, field) == 0
        assert all(value == 0 for value in rows.values())
    budget = report.sample_budget(
        sample_step=2,
        form_sample_error_bounds=(0, 0, 0),
        phase_sample_error_bounds=(0, 0, 0),
        timestamp_error_bounds=(1, 1, 1),
        observation_time=10,
    )
    assert budget.form.rate_error_bound == budget.form.acceleration_error_bound == 0
    assert budget.phase.rate_error_bound == budget.phase.acceleration_error_bound == 0
    assert budget.phase_increments_resolved


def test_fixed_prospective_budget_stays_within_declared_errors_without_samples():
    step, noise, jitter = Q(1, 4096), Q(1, 2**44), Q(1, 2**50)
    report = bound_sine_sampling_smoothness(
        **_arguments(
            reference_model=RelationalExchangeModel(1, phase_domain="regular"),
            form_diameter_bound=2,
            window_end=2 * step + jitter,
        )
    )
    budget = report.sample_budget(
        sample_step=step,
        form_sample_error_bounds=(0, noise, noise),
        phase_sample_error_bounds=(0, noise, noise),
        timestamp_error_bounds=(0, jitter, jitter),
    )
    assert budget.form.evidence_window == budget.phase.evidence_window == report.window
    assert budget.form.value_error_bound == budget.phase.value_error_bound == 0
    assert budget.form.rate_error_bound < Q("2.362e-7")
    assert budget.form.acceleration_error_bound < Q("0.002897")
    assert budget.phase.rate_error_bound < Q("6.830e-8")
    assert budget.phase.acceleration_error_bound < Q("0.0008350")
    assert all(value > 3 for value in budget.phase_increment_margins)
    assert budget.phase_increments_resolved
    # A nonzero noise/clock budget is retained instead of reusing old derivative padding.
    assert budget.form.acceleration_sample_error_bound > 0
    assert budget.form.acceleration_timing_error_bound > 0
    assert budget.form.acceleration_truncation_error_bound > 0


def test_sampling_reuses_one_shot_clock_admission_for_both_channels():
    report = bound_sine_sampling_smoothness(**_arguments(window_end=1))
    budget = report.sample_budget(
        sample_step=Q(1, 10),
        form_sample_error_bounds=iter((0, Q(1, 1000), Q(1, 2000))),
        phase_sample_error_bounds=iter((0, Q(1, 500), Q(1, 1000))),
        timestamp_error_bounds=iter((0, Q(1, 10000), Q(1, 20000))),
        observation_time=Q(1, 4),
    )
    assert budget.form.sample_times == budget.phase.sample_times
    assert budget.form.timestamp_error_bounds == budget.phase.timestamp_error_bounds
    assert budget.form.evidence_window == budget.phase.evidence_window
    assert budget.form.sample_error_bounds != budget.phase.sample_error_bounds


def test_sample_budget_rebuilds_cached_derivatives_from_unchanged_class_premises():
    source = bound_sine_sampling_smoothness(**_arguments())
    changed = replace(
        source,
        form_speed_bound=Q(0),
        phase_speed_bound=Q(0),
        form_third_derivative_bound=Q(0),
        phase_third_derivative_bound=Q(0),
    )
    arguments = dict(
        sample_step=Q(1, 16),
        form_sample_error_bounds=(0, 0, 0),
        phase_sample_error_bounds=(0, 0, 0),
    )
    budget = changed.sample_budget(**arguments)
    assert budget == source.sample_budget(**arguments)
    assert budget.smoothness == source
    assert budget.smoothness is not changed
    assert budget.form.acceleration_truncation_error_bound > 0
    assert budget.phase.acceleration_truncation_error_bound > 0
    assert budget.phase_increment_margins[0] < Q(22, 7)


def test_sample_budget_rebuilds_changed_capacity_and_enlarged_window():
    source = bound_sine_sampling_smoothness(**_arguments(capacity_ceiling=0))
    changed = replace(source, capacity_ceiling=Q(2), window=(Q(0), Q(1000)))
    budget = changed.sample_budget(
        sample_step=Q(1, 16),
        form_sample_error_bounds=(0, 0, 0),
        phase_sample_error_bounds=(0, 0, 0),
        observation_time=999,
    )
    expected = bound_sine_sampling_smoothness(**_arguments(window_end=1000))
    assert budget.smoothness == expected
    assert budget.form.evidence_window == (Q(999), Q(7993, 8))
    assert (
        budget.smoothness.form_diameter_upper_bound > source.form_diameter_upper_bound
    )
    assert budget.form.acceleration_truncation_error_bound > 0
    assert budget.phase.acceleration_truncation_error_bound > 0
    assert not budget.phase_increments_resolved


@pytest.mark.parametrize(
    "changes",
    (
        {"initial_form_diameter_bound": -1},
        {"initial_form_diameter_bound": True},
        {"capacity_ceiling": -1},
        {"capacity_ceiling": True},
        {"capacity_ceiling": float("inf")},
        {"window": ()},
        {"window": (0,)},
        {"window": (0, 1, 2)},
        {"window": (True, 1)},
        {"window": (-1, 1)},
        {"window": (2, 1)},
        {"window": (0, float("inf"))},
        {"reference_model": RelationalExchangeModel(1)},
    ),
)
def test_sample_budget_readmits_authoritative_class_before_sampling(changes):
    changed = replace(bound_sine_sampling_smoothness(**_arguments()), **changes)

    def unconsumed():
        pytest.fail("invalid source premises must reject before consuming samples")
        yield

    with pytest.raises((TypeError, ValueError)):
        changed.sample_budget(
            sample_step=Q(1, 16),
            form_sample_error_bounds=unconsumed(),
            phase_sample_error_bounds=unconsumed(),
        )


def test_sample_budget_readmits_stored_model_coefficients():
    source = bound_sine_sampling_smoothness(**_arguments())
    model = copy(source.reference_model)
    object.__setattr__(model, "storage_scale", True)
    with pytest.raises((TypeError, ValueError)):
        replace(source, reference_model=model).sample_budget(
            sample_step=Q(1, 16),
            form_sample_error_bounds=(0, 0, 0),
            phase_sample_error_bounds=(0, 0, 0),
        )


@pytest.mark.parametrize("observation_time", (Q(0), Q(9, 10)))
def test_complete_evidence_window_must_fit_independent_smoothness_window(
    observation_time,
):
    report = bound_sine_sampling_smoothness(
        **_arguments(window_start=Q(1, 10), window_end=1)
    )
    with pytest.raises(ValueError, match="independently bounded window"):
        report.sample_budget(
            sample_step=Q(1, 10),
            form_sample_error_bounds=(0, 0, 0),
            phase_sample_error_bounds=(0, 0, 0),
            observation_time=observation_time,
        )


def test_jitter_outside_end_is_not_hidden_by_nominal_times():
    report = bound_sine_sampling_smoothness(**_arguments(window_end=Q(1, 2)))
    with pytest.raises(ValueError, match="independently bounded window"):
        report.sample_budget(
            sample_step=Q(1, 4),
            form_sample_error_bounds=(0, 0, 0),
            phase_sample_error_bounds=(0, 0, 0),
            timestamp_error_bounds=(0, 0, Q(1, 100)),
        )


def test_unresolved_phase_increment_remains_explicit_and_does_not_unwrap():
    report = bound_sine_sampling_smoothness(
        **_arguments(form_diameter_bound=10, capacity_ceiling=10, window_end=2)
    )
    budget = report.sample_budget(
        sample_step=1,
        form_sample_error_bounds=(0, 0, 0),
        phase_sample_error_bounds=(0, 0, 0),
    )
    assert not budget.phase_increments_resolved
    assert all(value < 0 for value in budget.phase_increment_margins)
    assert budget.phase.acceleration_error_bound > 0


@pytest.mark.parametrize(
    "field,value",
    (
        ("reference_model", None),
        ("reference_model", RelationalExchangeModel(1)),
        ("form_diameter_bound", -1),
        ("form_diameter_bound", True),
        ("form_diameter_bound", float("nan")),
        ("capacity_ceiling", -1),
        ("capacity_ceiling", True),
        ("capacity_ceiling", float("inf")),
        ("window_start", -1),
        ("window_start", True),
        ("window_start", 1),
        ("window_end", float("inf")),
    ),
)
def test_invalid_independent_class_premises_reject(field, value):
    with pytest.raises((TypeError, ValueError)):
        bound_sine_sampling_smoothness(**_arguments(**{field: value}))


def test_reports_keep_exact_separate_budgets_and_export_without_running_a_flow(
    tmp_path, monkeypatch
):
    from tnfr.dynamics import relational
    from tnfr.physics import relational_sine_forecast

    def fail(*args, **kwargs):
        pytest.fail("a prospective budget must not run a native or sine trajectory")

    monkeypatch.setattr(relational, "evaluate_relational_exchange", fail)
    monkeypatch.setattr(relational_sine_forecast, "bound_sine_flow", fail)
    report = bound_sine_sampling_smoothness(**_arguments())
    budget = report.sample_budget(
        sample_step=Q(1, 10),
        form_sample_error_bounds=(0, Q(1, 1000), Q(1, 2000)),
        phase_sample_error_bounds=(0, Q(1, 500), Q(1, 1000)),
    )
    for name, value in (("smoothness", report), ("budget", budget)):
        payload = value.to_dict()
        path = tmp_path / f"{name}.json"
        export_to_json(payload, path)
        assert json.loads(path.read_text(encoding="utf-8")) == payload
    encoded = budget.to_dict()["report"]["form"]
    assert encoded["rate_timing_error_bound"] == {"numerator": 0, "denominator": 1}
    assert encoded["rate_sample_error_bound"]["numerator"] > 0
    assert encoded["rate_truncation_error_bound"]["numerator"] > 0
    with pytest.raises(FrozenInstanceError):
        report.capacity_ceiling = 99
    payload = report.to_dict()
    payload["report"]["form_speed_bound"]["numerator"] = -99
    assert report.form_speed_bound > 0
