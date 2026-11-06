"""Constructed association boundaries for a single NUL-to-sine consumer."""

import math
from fractions import Fraction as Q

import pytest

from tnfr.research import nul_successor as owner
from tnfr.research import sine_capacity_metric as capacity_metric
from tnfr.research import sine_preparation_metric as metric


def _arguments(*, width=2.0**-16, cut=False):
    negative_width = -width if width else 0.0
    return dict(
        factor=0.5,
        parent=0,
        form_bounds=((negative_width, 0.0, width), (0.0, 0.0, 0.0)),
        phase_bounds=((negative_width, 0.0, width), (0.0, 0.0, 0.0)),
        capacity_bounds=((2.0 - width, 2.0, 2.0 + width), (1.0, 1.0, 1.0)),
        pressure_bounds=(negative_width, 0.0, width),
        edges=((0, 1),),
        conductances=(1,),
        degrees=(1, 1),
        reference_weight=1,
        nominal_form=(0, 0),
        nominal_phase=(0, 0),
        nominal_capacities=(1, 1),
        phase_cells=(((0, 0), (0, 0)),),
        form_cells=(((0, 0), (0, 0)),),
        edge_allowances=(Q(1, 16),),
        durations=(Q(1, 64),),
        **(
            dict(
                child_form_bounds=(3.0 - width, 3.0, 3.0 + 2 * width),
                child_phase_bounds=(1.0 - 2 * width, 1.0, 1.0 + width),
            )
            if cut
            else dict(child=1)
        ),
    )


def test_joint_consumer_rebuilds_actual_successor_capacities_and_error_family():
    result = owner.bound_nul_successor_cells(**_arguments())
    epsilon = Q(1, 2**16)
    assert result.run.status == "completed"
    assert result.family.context.capacities == (1, 1)
    assert result.family.context.alpha == (Q(1, 2), Q(1, 2))
    assert result.family.form_errors == ((epsilon / 2, epsilon / 2), (0, 0))
    assert result.family.phase_errors == ((epsilon, epsilon), (0, 0))
    assert result.run.capacity_intervals == ((1 - epsilon / 2, 1 + epsilon / 2), (1, 1))
    assert result.run.relative_capacity_error == epsilon / 2
    assert result.successor_phase == result.source_phase
    assert result.successor_form[1] == result.source_form[1]
    assert result.successor_capacity[1] == result.source_capacity[1]
    assert result.reset.image_stored_pressure.lower == -2 * epsilon
    assert not hasattr(result, "admitted")


def test_zero_width_recovers_identical_nominal_state_without_forcing():
    result = owner.bound_nul_successor_cells(**_arguments(width=0.0))
    assert result.run.status == "completed"
    assert result.run.final_radius_upper == 0
    assert result.run.final_portrait_error_upper == 0
    assert result.run.relative_capacity_error == 0
    assert result.run.steps[0].forcing_upper == 0


def test_zero_capacity_width_recovers_existing_homogeneous_arithmetic():
    arguments = _arguments()
    arguments["capacity_bounds"] = ((2.0, 2.0, 2.0), (1.0, 1.0, 1.0))
    result = owner.bound_nul_successor_cells(**arguments)
    previous = metric.bound_metric_cells(
        result.family,
        **{
            key: arguments[key]
            for key in ("phase_cells", "edge_allowances", "durations")
        },
    )
    assert result.run.final_radius_upper == previous.final_radius_upper
    assert result.run.final_portrait_error_upper == previous.final_portrait_error_upper
    assert result.run.relative_capacity_error == 0


def test_cut_child_keeps_both_absolute_coordinates_and_both_asymmetric_offsets():
    result = owner.bound_nul_successor_cells(**_arguments(cut=True))
    epsilon = Q(1, 2**16)
    assert result.child_form.nominal == 3
    assert result.child_phase.nominal == 1
    assert result.family.form_mean_error == (-epsilon / 4, epsilon / 4)
    assert result.family.phase_mean_error == (-epsilon / 2, epsilon / 2)
    assert result.family.form_offset == (-5 * epsilon / 4, 9 * epsilon / 4)
    assert result.family.phase_offset == (-5 * epsilon / 2, 3 * epsilon / 2)
    for initial, final, drift in (
        (
            result.family.form_offset,
            result.run.final_form_offset,
            result.run.final_form_mean_drift_upper,
        ),
        (
            result.family.phase_offset,
            result.run.final_phase_offset,
            result.run.final_phase_mean_drift_upper,
        ),
    ):
        assert drift > 0
        assert final == (initial[0] - drift, initial[1] + drift)
    assert result.run.final_portrait_error_upper > result.run.final_radius_upper


@pytest.mark.parametrize(
    "field,value",
    (
        ("nominal_form", (1, 0)),
        ("nominal_phase", (1, 0)),
        ("nominal_capacities", (2, 1)),
        ("nominal_capacities", (True, 1)),
        ("nominal_form", (0.0, 0)),
        ("nominal_phase", (Q(0),)),
        ("form_cells", (((1, 2), (0, 0)),)),
        ("phase_cells", (((1, 2), (0, 0)),)),
    ),
)
def test_false_nominal_association_rejects_before_comparison(field, value, monkeypatch):
    def forbidden(*arguments, **keywords):
        pytest.fail("a mismatched nominal image must not enter comparison arithmetic")

    monkeypatch.setattr(capacity_metric, "bound_capacity_metric_cells", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_nul_successor_cells(**arguments)


def test_pre_reset_form_cannot_be_relabelled_as_the_successor_nominal():
    arguments = _arguments(width=0.0)
    arguments["form_bounds"] = ((2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    arguments["epi_max"] = 4.0
    arguments["nominal_form"] = (2, 0)
    arguments["form_cells"] = (((0, 2), (0, 0)),)
    with pytest.raises(ValueError, match="actual nominal NUL image"):
        owner.bound_nul_successor_cells(**arguments)
    arguments["nominal_form"] = (1, 0)
    result = owner.bound_nul_successor_cells(**arguments)
    assert result.source_form[0].nominal == 2
    assert result.successor_form[0].nominal == 1


@pytest.mark.parametrize(
    "field,value",
    (
        ("parent", True),
        ("capacity_bounds", ((2.0, 2.0, 2.0), (0.0, 1.0, 1.0))),
        ("form_bounds", ((0.0, 0.0, 0.0), (0.0, False, 0.0))),
        ("phase_bounds", ((0.0, 0.0, 0.0), (0.0, math.inf, math.inf))),
        ("phase_bounds", ((0.0, 0.0, 0.0),)),
        ("child", 0),
        ("child_form_bounds", (0.0, 0.0, 0.0)),
    ),
)
def test_invalid_complete_preparation_rejects_before_reset(field, value, monkeypatch):
    def forbidden(**keywords):
        pytest.fail("complete primitive preparation must be admitted before reset")

    monkeypatch.setattr(owner.nul_reset, "bound_nul_capacity_reset", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_nul_successor_cells(**arguments)


def test_later_invalid_cell_cannot_hide_behind_first_cell_success(monkeypatch):
    def forbidden(*arguments, **keywords):
        pytest.fail("all cells must be admitted before any comparison work")

    monkeypatch.setattr(capacity_metric, "bound_capacity_metric_cells", forbidden)
    arguments = _arguments()
    arguments["form_cells"] *= 2
    arguments["phase_cells"] += (((0, 0), (1, 0)),)
    arguments["durations"] *= 2
    with pytest.raises(ValueError, match="reversed"):
        owner.bound_nul_successor_cells(**arguments)


def test_conditional_first_exit_failure_remains_unavailable_without_retry():
    arguments = _arguments(width=0.25)
    result = owner.bound_nul_successor_cells(**arguments)
    assert result.run.status == "unavailable"
    assert result.run.reason == "initial_domain"
    assert result.run.steps == ()
    assert result.run.attempted_cells == 0
