"""Constructed admission and finite-work controls; no retained source is read."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest

from tnfr.research import sine_capacity_metric as owner
from tnfr.research import sine_preparation_metric as metric


def _family(*, error=Q(1, 1024), cut=False):
    context = metric.prepare_metric_context(
        capacities=(1, 1), edges=((0, 1),), conductances=(4,), degrees=(1, 1)
    )
    return metric.prepare_metric_family(
        context,
        form_errors=((error, error),) * 2,
        phase_errors=((error, error),) * 2,
        parent=0,
        reference_weight=4,
        **(
            dict(
                child_form_error=(error, 2 * error),
                child_phase_error=(2 * error, error),
            )
            if cut
            else dict(child=1)
        ),
    )


def _arguments(count=1):
    return dict(
        capacity_intervals=((Q(127, 128), Q(129, 128)), (1, 1)),
        phase_cells=(((0, 0), (Q(1, 8), Q(1, 8))),) * count,
        form_cells=(((0, 0), (Q(1, 4), Q(1, 4))),) * count,
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64),) * count,
    )


@pytest.mark.parametrize("cut", (False, True))
@pytest.mark.parametrize("error", (Q(0), Q(1, 1024), Q(1, 2**1074)))
def test_zero_capacity_width_exactly_recovers_old_radius_and_portrait(cut, error):
    family = _family(error=error, cut=cut)
    arguments = _arguments(2)
    arguments["capacity_intervals"] = ((1, 1), (1, 1))
    old = metric.bound_metric_cells(
        family,
        **{
            key: arguments[key]
            for key in ("phase_cells", "edge_allowances", "durations")
        },
    )
    new = owner.bound_capacity_metric_cells(family, **arguments)
    assert old.status == new.status == "completed"
    assert new.relative_capacity_error == 0
    assert new.final_radius_upper == old.final_radius_upper
    assert new.final_normalized_radius == old.final_normalized_radius
    assert new.final_portrait_error_upper == old.final_portrait_error_upper
    assert new.final_form_mean_error == family.form_mean_error
    assert new.final_phase_mean_error == family.phase_mean_error
    assert new.final_form_offset == family.form_offset
    assert new.final_phase_offset == family.phase_offset
    for previous, current in zip(old.steps, new.steps, strict=True):
        assert current.gamma_upper == previous.coefficient.gamma_upper
        assert current.normalized_radius_after == previous.normalized_radius_after
        assert current.edge_margins == previous.edge_margins
        assert current.forcing_upper == 0
        assert current.form_mean_drift_upper == current.phase_mean_drift_upper == 0


@pytest.mark.parametrize(
    "field,value",
    (
        ("capacity_intervals", ((0, 1), (1, 1))),
        ("capacity_intervals", ((-1, 1), (1, 1))),
        ("capacity_intervals", ((2, 3), (1, 1))),
        ("capacity_intervals", ((1, Q(1, 2)), (1, 1))),
        ("capacity_intervals", ((True, 1), (1, 1))),
        ("capacity_intervals", ((1.0, 1), (1, 1))),
        ("capacity_intervals", ((1, 1),)),
        ("capacity_intervals", {0: (1, 1), 1: (1, 1)}),
        ("form_cells", ()),
        ("form_cells", (((0, 0),),)),
        ("form_cells", (((1, 0), (0, 0)),)),
        ("form_cells", (((False, 0), (0, 0)),)),
        ("phase_cells", (((0, 0), (float("inf"), float("inf"))),)),
        ("durations", ()),
        ("durations", (0,)),
        ("durations", (0.5,)),
        ("edge_allowances", (0,)),
        ("edge_allowances", (True,)),
        ("edge_allowances", (Q(1, 8), Q(1, 8))),
    ),
)
def test_primitive_batches_reject_invalid_domains_before_cell_work(
    monkeypatch, field, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("no cell arithmetic is permitted before complete input admission")

    monkeypatch.setattr(owner, "_bound_cell", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_capacity_metric_cells(_family(), **arguments)


def test_later_form_cell_is_admitted_before_any_growth_or_trigonometry(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "a later invalid primitive box cannot be hidden by an earlier result"
        )

    monkeypatch.setattr(owner, "cos", forbidden)
    monkeypatch.setattr(owner, "sin", forbidden)
    arguments = _arguments(2)
    arguments["form_cells"] = (arguments["form_cells"][0], ((2, 1), (0, 0)))
    with pytest.raises(ValueError):
        owner.bound_capacity_metric_cells(_family(), **arguments)


def test_whole_batch_limit_rejects_257_cells_before_arithmetic(monkeypatch):
    monkeypatch.setattr(
        owner, "_bound_cell", lambda *args, **kwargs: pytest.fail("work cap")
    )
    with pytest.raises(ValueError, match="work limit"):
        owner.bound_capacity_metric_cells(_family(), **_arguments(257))


def test_input_bit_limit_rejects_before_capacity_product(monkeypatch):
    monkeypatch.setattr(
        owner, "_bound_cell", lambda *args, **kwargs: pytest.fail("bit cap")
    )
    arguments = _arguments()
    arguments["capacity_intervals"] = ((1, 1 << metric.MAX_INPUT_BITS), (1, 1))
    with pytest.raises(metric.MetricBudgetError, match="input"):
        owner.bound_capacity_metric_cells(_family(), **arguments)


def test_forged_family_cannot_substitute_a_cached_preparation_verdict():
    family = _family()
    with pytest.raises(ValueError, match="live result"):
        owner.bound_capacity_metric_cells(replace(family), **_arguments())


def test_initial_first_exit_boundary_is_strict_without_an_attempt():
    family = _family(error=Q(1, 16))
    run = owner.bound_capacity_metric_cells(family, **_arguments())
    assert run.status == "unavailable" and run.reason == "initial_domain"
    assert run.steps == () and run.attempted_cells == run.completed_cells == 0


def test_forcing_can_fail_first_exit_from_identical_initial_state_and_stops_suffix():
    arguments = _arguments(3)
    arguments["capacity_intervals"] = ((1, 2), (1, 1))
    arguments["form_cells"] = (((0, 0), (100, 100)),) * 3
    run = owner.bound_capacity_metric_cells(_family(error=Q(0)), **arguments)
    assert run.status == "unavailable" and run.reason == "first_exit"
    assert run.attempted_cells == len(run.steps) == 1
    assert run.completed_cells == 0 and run.steps[0].forcing_upper > 0


def test_scalar_work_keeps_only_the_completed_prefix_and_combined_growth():
    arguments = _arguments(2)
    arguments["phase_cells"] = (((3, 3), (0, 0)),) * 2
    arguments["durations"] = (Q(1, 128), Q(1))
    run = owner.bound_capacity_metric_cells(_family(), **arguments)
    assert run.status == "unavailable" and run.reason == "scalar_work"
    assert run.completed_cells == len(run.steps) == 1
    assert run.attempted_cells == 2
    assert run.failed_gamma_upper > run.failed_coefficient.gamma_upper
    assert run.final_normalized_radius == run.steps[0].normalized_radius_after


def test_intermediate_bit_failure_keeps_prior_radius_and_both_means(monkeypatch):
    original = owner.scalar_affine_upper
    calls = []

    def oversized_second(*arguments):
        calls.append(arguments)
        return original(*arguments) if len(calls) == 1 else Q(1 << 65537)

    monkeypatch.setattr(owner, "scalar_affine_upper", oversized_second)
    run = owner.bound_capacity_metric_cells(_family(cut=True), **_arguments(3))
    assert run.status == "unavailable" and run.reason == "intermediate_budget"
    assert run.attempted_cells == len(calls) == 2
    assert run.completed_cells == len(run.steps) == 1
    assert run.final_radius_upper == run.steps[0].radius_upper
    assert run.final_form_mean_error == run.steps[0].form_mean_error
    assert run.final_phase_mean_error == run.steps[0].phase_mean_error
    assert run.maximum_bits >= 65538


def test_scalar_affine_owner_receives_forcing_scaled_by_initial_family_scale(
    monkeypatch,
):
    original = owner.scalar_affine_upper
    calls = []

    def record(*arguments):
        result = original(*arguments)
        calls.append((arguments, result))
        return result

    monkeypatch.setattr(owner, "scalar_affine_upper", record)
    family = _family(error=Q(1, 2**200))
    run = owner.bound_capacity_metric_cells(family, **_arguments(2))
    assert run.status == "completed" and len(calls) == 2
    for index, (step, (arguments, result)) in enumerate(
        zip(run.steps, calls, strict=True)
    ):
        assert arguments == (
            step.normalized_radius_before,
            step.gamma_upper,
            step.duration,
            step.forcing_upper / family.radius_scale,
        )
        assert result == step.normalized_radius_after
        assert step.radius_upper == family.radius_scale * result
        if index:
            assert arguments[0] == calls[index - 1][1]


def test_primitive_form_box_changes_forcing_even_with_identical_phase_and_radius():
    family = _family(error=Q(0))
    first = owner.bound_capacity_metric_cells(family, **_arguments())
    arguments = _arguments()
    arguments["form_cells"] = (((0, 0), (2, 2)),)
    second = owner.bound_capacity_metric_cells(family, **arguments)
    assert second.steps[0].nominal_form_gradient != first.steps[0].nominal_form_gradient
    assert second.steps[0].forcing_upper > first.steps[0].forcing_upper
    assert second.final_radius_upper > first.final_radius_upper


def test_shared_arithmetic_defect_is_not_retried_or_reported_as_unavailable(
    monkeypatch,
):
    calls = []

    def broken(*arguments):
        calls.append(arguments)
        raise ArithmeticError("constructed shared arithmetic failure")

    monkeypatch.setattr(owner, "scalar_affine_upper", broken)
    with pytest.raises(ArithmeticError, match="constructed"):
        owner.bound_capacity_metric_cells(_family(), **_arguments(3))
    assert len(calls) == 1
