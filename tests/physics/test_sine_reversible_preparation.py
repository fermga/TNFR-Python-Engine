"""Reversible preparation consumer controls, not a formation experiment.

Stationary exact solutions provide analytical endpoint evidence. Explicit
stubs test successful-path selection separately; they cannot establish a
scientific acquisition or replace the reserved full-field producer.
"""

import json
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tests.physics.test_sine_cycle_barrier import _state
from tests.physics.test_sine_phase_offset_partition import CYCLE_LEAF_EDGES
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.mathematics._validated_taylor import ValidatedTaylorStep
from tnfr.physics import relational_sine_regional as owner
from tnfr.physics.relational_sine_forecast import SineForecast
from tnfr.sdk import export_to_json, relational_report_to_dict

TARGET_ERROR = Q(1, 4096)
SOURCE_ERROR = Q(1, 2**100)


def _stationary_forecast(checkpoint, *, duration=Q(1, 64), count=2):
    """Exact zero-state flow; broad tubes have a strict Picard inclusion."""
    assert not any(checkpoint.epi) and not any(checkpoint.phase)
    size = len(checkpoint.nodes)
    rows = [set() for _ in checkpoint.nodes]
    for i, j in CYCLE_LEAF_EDGES:
        rows[i].add(j)
        rows[j].add(i)
    initial = (I(0),) * (2 * size) + (I(1),)
    form_radius = 1 + 2 * duration
    phase_radius = 1 + 4 * duration * form_radius
    tube = (
        (I(-form_radius, form_radius),) * size
        + (I(-phase_radius, phase_radius),) * size
        + (I(Q(99, 100), Q(101, 100)),)
    )
    steps = tuple(
        ValidatedTaylorStep(
            time=index * duration,
            duration=duration,
            tube=tube,
            endpoint=initial,
            picard_interior_margin=Q(1, 1000),
            domain_lower_bounds=(Q(1),),
            propagated_initial_radii=(Q(0),) * len(initial),
            local_remainder_bounds=(I(0),) * len(initial),
        )
        for index in range(count)
    )
    return SineForecast(
        model=checkpoint.reference_model,
        neighbors=tuple(tuple(sorted(row)) for row in rows),
        visible_capacity=(Q(1),) * (size - 1),
        initial_box=initial,
        observation_time=Q(0),
        end_time=count * duration,
        time_step=duration,
        order=4,
        steps=steps,
        validated_end_time=count * duration,
        endpoint=initial,
        failed_tube=None,
        status="admitted",
        reasons=(),
    )


@pytest.fixture(scope="module")
def checkpoint():
    return _state()


@pytest.fixture(scope="module")
def forecast(checkpoint):
    return _stationary_forecast(checkpoint)


def _assess(forecast, checkpoint, **changes):
    args = dict(
        cycle=range(5),
        target_error_bound=TARGET_ERROR,
        source_error_bound=SOURCE_ERROR,
        scaled_retention_duration=Q(1, 1000),
    )
    return owner.assess_sine_reversible_preparation(
        forecast, checkpoint, **(args | changes)
    )


def test_stationary_zero_winding_flow_never_becomes_an_acquisition(
    checkpoint, forecast
):
    report = _assess(forecast, checkpoint)
    assert report.horizon_complete
    assert report.outcome == "no_certificate_on_declared_grid"
    assert report.selected_step_index is None and report.formation_time is None
    assert not report.target_retention.whole_window_retention_certified
    assert len(report.steps) == 2
    assert all(step.source_zero_winding_certified for step in report.steps)
    assert all(step.target_entry_certified for step in report.steps)
    assert not any(step.formation_retention_certified for step in report.steps)
    assert all(
        step.reasons == ("target_retention_not_certified",) for step in report.steps
    )
    assert all(step.endpoint_error_bound == 0 for step in report.steps)


def test_reversing_forms_is_the_full_nodal_time_reversal_including_environment():
    forms = tuple(Q(i * i - 4 * i + 1, 13) for i in range(10))
    phases = tuple(Q((-1) ** i * (i + 1), 17) for i in range(10))
    rows = [set() for _ in forms]
    for i, j in CYCLE_LEAF_EDGES:
        rows[i].add(j)
        rows[j].add(i)
    with mp.workdps(80):

        def number(value):
            return mp.mpf(value.numerator) / value.denominator

        def full_field(x, theta):
            return tuple(
                mp.fsum(mp.sin(theta[j] - theta[i]) for j in row) / len(row)
                for i, row in enumerate(rows)
            ) + tuple(
                mp.fsum(x[i] - x[j] for j in row) / len(row)
                for i, row in enumerate(rows)
            )

        x, theta = tuple(map(number, forms)), tuple(map(number, phases))
        direct = full_field(x, theta)
        reversed_field = full_field(tuple(-value for value in x), theta)
        assert reversed_field == direct[:10] + tuple(-value for value in direct[10:])
        # The derivative of R changes the form-row sign, so F(Rz)=-R F(z).
        assert any(abs(value) > 0 for value in direct[5:10])
        assert any(abs(value) > 0 for value in direct[15:])


def test_outward_endpoint_radius_cannot_be_replaced_by_its_midpoint(checkpoint):
    forecast = _stationary_forecast(checkpoint, count=1)
    tiny = Q(1, 10**80)
    endpoint = (I(0, tiny),) + forecast.endpoint[1:]
    forecast = replace(
        forecast,
        endpoint=endpoint,
        steps=(replace(forecast.steps[0], endpoint=endpoint),),
    )
    report = _assess(forecast, checkpoint, source_error_bound=Q(1, 2**200))
    step = report.steps[0]
    assert step.source_form_center[0] == -endpoint[0].midpoint
    assert step.endpoint_error_bound == endpoint[0].radius > tiny / 2
    assert (
        step.propagated_target_error_bound
        == step.lipschitz_amplification_upper_bound
        * (endpoint[0].radius + Q(1, 2**200))
    )
    with mp.workdps(80):
        exponent = 2 * mp.mpf(step.time.numerator) / step.time.denominator / mp.pi
        upper = step.lipschitz_amplification_upper_bound
        assert mp.exp(exponent) <= mp.mpf(upper.numerator) / upper.denominator


def test_strict_radius_and_time_guards_are_independent(checkpoint, forecast):
    report = _assess(forecast, checkpoint)
    amplification = report.steps[0].lipschitz_amplification_upper_bound
    equality = _assess(
        forecast, checkpoint, source_error_bound=TARGET_ERROR / amplification
    )
    assert equality.steps[0].target_radius_margin_lower_bound == 0
    assert not equality.steps[0].target_entry_certified
    short = _assess(forecast, checkpoint, scaled_retention_duration=1)
    assert all(step.target_entry_certified for step in short.steps)
    assert not any(step.time_separation_certified for step in short.steps)
    assert all(step.time <= pi_interval().hi for step in short.steps)


def test_energy_overlap_is_only_a_separate_necessary_check(checkpoint):
    forecast = _stationary_forecast(checkpoint, count=1)
    # The exact stationary endpoint remains in this deliberately loose box.
    endpoint = (I(0, 1),) + forecast.endpoint[1:]
    forecast = replace(
        forecast,
        endpoint=endpoint,
        steps=(replace(forecast.steps[0], endpoint=endpoint),),
    )
    step = _assess(forecast, checkpoint).steps[0]
    assert step.source_zero_winding_certified
    assert not step.source_target_energy_overlap
    assert "source_target_storage_intervals_disjoint" in step.reasons


def test_cached_checkpoint_fields_do_not_supply_retention(checkpoint, forecast):
    poisoned = replace(
        checkpoint,
        storage=I(4),
        form_gradient=(Q(999),),
        form_rates=(),
        phase_rates=(),
        relative_resultant=(),
    )
    ordinary, actual = _assess(forecast, checkpoint), _assess(forecast, poisoned)
    assert (
        actual.target_retention.full_storage_bounds
        == ordinary.target_retention.full_storage_bounds
    )
    assert actual.steps == ordinary.steps
    assert actual.outcome == ordinary.outcome


def test_first_success_selection_is_wiring_only_and_survives_later_numerical_stop(
    checkpoint, forecast, monkeypatch
):
    actual_reader = owner.assess_sine_cycle_retention

    def target_stub(source, **kwargs):
        result = actual_reader(source, **kwargs)
        if kwargs["source_error_bound"] == TARGET_ERROR:
            # Deliberately impossible target assumption tests predicate and
            # prefix wiring only, never a scientific positive control.
            return replace(result, whole_window_retention_certified=True)
        return result

    monkeypatch.setattr(owner, "assess_sine_cycle_retention", target_stub)
    complete = _assess(forecast, checkpoint)
    assert complete.outcome == "certified"
    assert complete.selected_step_index == 0
    assert complete.formation_time == forecast.time_step
    prefix = replace(
        forecast,
        steps=forecast.steps[:1],
        validated_end_time=forecast.time_step,
        status="unavailable",
        reasons=("synthetic_numerical_stop",),
        failed_tube=forecast.steps[1].tube,
    )
    partial = _assess(prefix, checkpoint)
    assert partial.outcome == "certified" and not partial.horizon_complete
    assert partial.selected_step_index == 0 and not partial.reasons


def test_unresolved_prefix_is_not_a_complete_grid_verdict(checkpoint, forecast):
    prefix = replace(
        forecast,
        steps=(),
        validated_end_time=0,
        status="unavailable",
        reasons=("synthetic_numerical_stop",),
        failed_tube=forecast.steps[0].tube,
    )
    result = _assess(prefix, checkpoint)
    assert result.outcome == "unavailable" and result.steps == ()
    assert not result.horizon_complete
    assert "forecast_validated_prefix_incomplete" in result.reasons


def test_exponential_overflow_is_unavailable_not_a_zero_amplification(checkpoint):
    forecast = _stationary_forecast(checkpoint, duration=Q(10000), count=1)
    step = _assess(forecast, checkpoint).steps[0]
    assert step.lipschitz_amplification_upper_bound is None
    assert step.propagated_target_error_bound is None
    assert not step.target_entry_certified
    assert "lipschitz_amplification_not_representable" in step.reasons


@pytest.mark.parametrize(
    "field,value",
    (
        ("target_error_bound", False),
        ("target_error_bound", 0),
        ("source_error_bound", True),
        ("source_error_bound", -1),
        ("source_error_bound", float("nan")),
        ("source_error_bound", "1/4096"),
        ("scaled_retention_duration", 0),
        ("scaled_retention_duration", False),
    ),
)
def test_source_target_and_time_domains_reject_before_evidence_use(
    checkpoint, forecast, field, value
):
    with pytest.raises((TypeError, ValueError)):
        _assess(forecast, checkpoint, **{field: value})


@pytest.mark.parametrize("coordinate", ("epi", "phase"))
def test_exact_checkpoint_association_cannot_be_rounded_away(
    checkpoint, forecast, coordinate
):
    tiny = Q(1, 10**200)
    modified = replace(
        checkpoint, **{coordinate: (tiny,) + getattr(checkpoint, coordinate)[1:]}
    )
    with pytest.raises(ValueError, match="R\\(checkpoint\\)"):
        _assess(forecast, modified)


def test_forecast_support_capacity_prior_and_cached_picard_claim_are_readmitted(
    checkpoint, forecast
):
    extra = list(forecast.neighbors)
    extra[0] = tuple(sorted((*extra[0], 2)))
    extra[2] = tuple(sorted((*extra[2], 0)))
    cases = (
        replace(forecast, neighbors=tuple(extra)),
        replace(forecast, visible_capacity=(Q(0),) + forecast.visible_capacity[1:]),
        replace(forecast, prior_admission=object()),
        replace(forecast, forecast_start=0),
        replace(forecast, freeze_hidden=True),
        replace(
            forecast,
            steps=(replace(forecast.steps[0], picard_interior_margin=100),)
            + forecast.steps[1:],
        ),
    )
    for invalid in cases:
        with pytest.raises((TypeError, ValueError)):
            _assess(invalid, checkpoint)

    shifted = replace(
        forecast,
        observation_time=forecast.time_step,
        end_time=forecast.end_time + forecast.time_step,
        validated_end_time=forecast.validated_end_time + forecast.time_step,
        steps=tuple(
            replace(step, time=step.time + forecast.time_step)
            for step in forecast.steps
        ),
    )
    with pytest.raises(ValueError, match="time zero"):
        _assess(shifted, checkpoint)


def test_sdk_export_retains_endpoint_uncertainty_and_checks_checkpoint_labels(
    checkpoint, forecast, tmp_path
):
    result = _assess(forecast, checkpoint)
    payload = relational_report_to_dict(result)
    assert payload["report_type"] == "SineReversiblePreparation"
    assert result.to_dict()["schema"] == "tnfr.sine-reversible-preparation.v1"
    assert payload["report"]["selected_step_index"] is None
    assert payload["report"]["source_error_bound"] == {
        "numerator": 1,
        "denominator": 2**100,
    }
    output = tmp_path / "reversible.json"
    export_to_json(payload, output)
    assert json.loads(output.read_text(encoding="utf-8")) == payload

    @dataclass(frozen=True)
    class Opaque:
        value: int

    with pytest.raises(TypeError):
        relational_report_to_dict(
            replace(result, cycle=(Opaque(0),) + result.cycle[1:])
        )
    with pytest.raises(TypeError):
        relational_report_to_dict(
            replace(
                result,
                target_retention=replace(
                    result.target_retention, cycle=(Opaque(0),) + result.cycle[1:]
                ),
            )
        )
