"""Constructed metric arithmetic controls; no retained source is consumed."""

import pickle
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from itertools import product

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import sine_preparation_metric as owner


def _two_node():
    return owner.prepare_metric_context(
        capacities=(1, 1), edges=((0, 1),), conductances=(4,), degrees=(1, 1)
    )


def _triangle(*, conductances=(2, 3, 5)):
    return owner.prepare_metric_context(
        capacities=(2, 4, 6),
        edges=((0, 1), (1, 2), (0, 2)),
        conductances=conductances,
        degrees=(2, 2, 2),
    )


def _multiply(left, right):
    return tuple(
        tuple(
            sum((a * b for a, b in zip(row, column, strict=True)), Q(0))
            for column in zip(*right)
        )
        for row in left
    )


def _quadratic(vector, matrix):
    return sum(
        (
            vector[i] * value * vector[j]
            for i, row in enumerate(matrix)
            for j, value in enumerate(row)
        ),
        Q(0),
    )


def test_two_node_completed_inverse_and_edge_duals_have_independent_values():
    context = _two_node()
    assert context.beta == context.rho == (Q(1), Q(1))
    assert context.alpha == (Q(1, 2), Q(1, 2))
    assert context.laplacian_weighted == ((4, -4), (-4, 4))
    assert context.laplacian_bare == ((1, -1), (-1, 1))
    assert context.form_inverse == ((Q(9, 16), Q(7, 16)), (Q(7, 16), Q(9, 16)))
    assert context.phase_inverse == ((Q(3, 4), Q(1, 4)), (Q(1, 4), Q(3, 4)))
    assert context.edge_phase_duals == (1,)
    assert context.edge_form_duals == (Q(1, 4),)
    assert context.edge_sandwich == (16,)
    assert (context.a0, context.b0) == (2, 8)
    identity = ((Q(1), Q(0)), (Q(0), Q(1)))
    for matrix, inverse in (
        (context.form_metric, context.form_inverse),
        (context.phase_metric, context.phase_inverse),
    ):
        assert _multiply(matrix, inverse) == _multiply(inverse, matrix) == identity


def test_unequal_mobility_triangle_preserves_actual_mean_and_parallel_path_dual():
    context = _triangle()
    assert context.beta == (1, 2, 3)
    assert context.rho == (1, Q(1, 2), Q(1, 3))
    assert context.alpha == (Q(6, 11), Q(3, 11), Q(2, 11))
    ell = (Q(1), Q(-1), Q(0))
    # Independent conductance: the direct weight2 is parallel to series3,5.
    assert _quadratic(ell, context.form_inverse) == Q(1) / (2 + Q(15, 8)) == Q(8, 31)
    assert _quadratic(ell, context.phase_inverse) == Q(2, 3)
    for inverse in (context.form_inverse, context.phase_inverse):
        response = tuple(
            sum((row[i] * ell[i] for i in range(3)), Q(0)) for row in inverse
        )
        assert (
            sum((a * z for a, z in zip(context.alpha, response, strict=True)), Q(0))
            == 0
        )


def test_zero_conductance_keeps_its_bare_edge_but_requires_weighted_connectivity():
    context = _triangle(conductances=(2, 3, 0))
    assert context.degrees == (2, 2, 2)
    assert context.laplacian_bare == ((2, -1, -1), (-1, 2, -1), (-1, -1, 2))
    assert context.laplacian_weighted == ((2, -2, 0), (-2, 5, -3), (0, -3, 3))
    assert _quadratic((1, 0, -1), context.form_inverse) == Q(1, 2) + Q(1, 3)
    with pytest.raises(ValueError):
        _triangle(conductances=(2, 0, 0))


def test_context_detaches_caller_collections_without_mutating_them():
    capacities, edges, weights, degrees = [1, 1], [[0, 1]], [4], [1, 1]
    before = deepcopy((capacities, edges, weights, degrees))
    context = owner.prepare_metric_context(
        capacities=capacities, edges=edges, conductances=weights, degrees=degrees
    )
    assert (capacities, edges, weights, degrees) == before
    capacities[0] = 9
    edges[0][0] = 1
    weights[0] = 0
    degrees[0] = 7
    assert context.capacities == (1, 1)
    assert context.edges == ((0, 1),)
    assert context.conductances == (4,)
    assert context.degrees == (1, 1)


@pytest.mark.parametrize("value", (True, 1.0, "1", float("nan"), float("inf")))
@pytest.mark.parametrize("field", ("capacities", "conductances"))
def test_physical_coefficients_require_exact_nonboolean_scalars(value, field):
    arguments = dict(
        capacities=(1, 1), edges=((0, 1),), conductances=(4,), degrees=(1, 1)
    )
    arguments[field] = (value,) + arguments[field][1:]
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_metric_context(**arguments)


@pytest.mark.parametrize("value", (True, Q(1), 1.0, 0, -1, 2))
def test_degrees_are_actual_positive_bare_counts(value):
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_metric_context(
            capacities=(1, 1), edges=((0, 1),), conductances=(4,), degrees=(value, 1)
        )


@pytest.mark.parametrize(
    "edges,weights,degrees",
    (
        (((0, 0),), (4,), (1, 1)),
        (((0, 1), (1, 0)), (4, 4), (2, 2)),
        (((0, 2),), (4,), (1, 1)),
        (((False, 1),), (4,), (1, 1)),
        (((Q(0), 1),), (4,), (1, 1)),
        ((), (), (0, 0)),
    ),
)
def test_invalid_support_cannot_be_coerced_to_a_valid_component(
    edges, weights, degrees
):
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_metric_context(
            capacities=(1, 1), edges=edges, conductances=weights, degrees=degrees
        )


@pytest.mark.parametrize(
    "capacities,weights", (((0, 1), (4,)), ((-1, 1), (4,)), ((1, 1), (-4,)))
)
def test_nonpositive_capacity_or_negative_weight_is_outside_dynamic_component(
    capacities, weights
):
    with pytest.raises(ValueError):
        owner.prepare_metric_context(
            capacities=capacities, edges=((0, 1),), conductances=weights, degrees=(1, 1)
        )


def test_numpy_fraction_payloads_normalize_before_large_rational_products():
    np = pytest.importorskip("numpy")
    payload = Q(np.int64(2**62), np.int64(3))
    context = owner.prepare_metric_context(
        capacities=(payload, 1), edges=((0, 1),), conductances=(4,), degrees=(1, 1)
    )
    assert context.capacities[0] == context.beta[0] == Q(2**62, 3)
    assert context.form_sandwich[0][0] == 4 * Q(2**62, 3) ** 2
    for scalar in context.capacities + context.rho + context.alpha:
        assert (
            type(scalar) is Q
            and type(scalar.numerator) is type(scalar.denominator) is int
        )
    for field, value in (
        ("capacities", (np.int64(1), 1)),
        ("degrees", (np.int64(1), 1)),
    ):
        arguments = dict(
            capacities=(1, 1), edges=((0, 1),), conductances=(4,), degrees=(1, 1)
        )
        arguments[field] = value
        with pytest.raises((TypeError, ValueError)):
            owner.prepare_metric_context(**arguments)


def test_input_bit_budget_rejects_before_unbounded_exact_algebra():
    with pytest.raises(ValueError, match="bit|budget|work"):
        owner.prepare_metric_context(
            capacities=(Q(1 << 4097), 1),
            edges=((0, 1),),
            conductances=(4,),
            degrees=(1, 1),
        )


@pytest.mark.parametrize(
    "forge",
    (
        lambda context: replace(context),
        lambda context: replace(context, phase_inverse=((Q(0), Q(0)), (Q(0), Q(0)))),
        lambda context: replace(context, alpha=(Q(1), Q(0))),
        lambda context: pickle.loads(pickle.dumps(context)),
    ),
)
def test_external_context_copies_and_forged_cached_fields_are_not_capabilities(forge):
    original = _two_node()
    with pytest.raises((TypeError, ValueError)):
        owner.bound_secant_metric(forge(original), secant_intervals=((Q(0), Q(0)),))


@pytest.mark.parametrize(
    "interval,trace_square,norm",
    (
        ((Q(1), Q(1)), Q(0), Q(0)),
        ((Q(0), Q(0)), Q(16), Q(4)),
        ((Q(-1), Q(-1)), Q(64), Q(8)),
        ((Q(1, 2), Q(1)), Q(1), Q(2)),
    ),
)
def test_two_node_secant_norm_and_growth_normalization(interval, trace_square, norm):
    report = owner.bound_secant_metric(_two_node(), secant_intervals=(interval,))
    assert report.trace_square == trace_square
    assert report.norm_upper >= norm
    assert report.norm_upper - norm < Q(1, 2**110)
    assert report.norm_upper == min(report.trace_upper, report.degree_upper)
    if norm == 0:
        assert report.gamma_upper == 0
    else:
        pi = pi_interval()
        assert report.gamma_upper >= norm / (2 * pi.lo)
        assert report.gamma_upper - norm / (2 * pi.hi) < Q(1, 2**110)


def test_uncertain_secant_retains_residual_instead_of_only_its_midpoint():
    context = _two_node()
    point = owner.bound_secant_metric(context, secant_intervals=((0, 0),))
    wide = owner.bound_secant_metric(context, secant_intervals=((-1, 1),))
    assert wide.midpoints == point.midpoints == (0,)
    assert wide.radii == (1,)
    assert wide.trace_square == point.trace_square == 16
    assert wide.trace_residual_upper >= 4
    assert wide.norm_upper >= 8 > point.norm_upper


@pytest.mark.parametrize(
    "intervals",
    (
        (),
        ((Q(-2), Q(1)),),
        ((Q(0), Q(2)),),
        ((Q(1), Q(0)),),
        ((False, 1),),
        ((0.0, 1),),
        (I(0),),
        ((0, 1, 2),),
        {0: (0, 1)},
    ),
)
def test_secant_domains_are_exact_ordered_pairs_inside_minus_one_to_one(intervals):
    with pytest.raises((TypeError, ValueError)):
        owner.bound_secant_metric(_two_node(), secant_intervals=intervals)


def _connected_family(*, form=Q(1, 1024), phase=Q(1, 1024)):
    return owner.prepare_metric_family(
        _two_node(),
        form_errors=((form, form),) * 2,
        phase_errors=((phase, phase),) * 2,
        parent=0,
        child=1,
        reference_weight=4,
    )


def test_tiny_and_subnormal_initial_errors_are_scaled_before_outward_rounding():
    for tiny in (Q(1, 2**200), Q(1, 2**1074)):
        family = _connected_family(form=tiny, phase=tiny)
        assert family.initial_norm_squared == 20 * tiny**2
        assert family.radius_scale > 0
        assert (
            family.initial_radius_upper
            == family.radius_scale * family.initial_normalized_radius
        )
        assert 4 * tiny < family.initial_radius_upper < 5 * tiny
        run = owner.bound_metric_cells(
            family,
            phase_cells=(((0, 0), (0, 0)),),
            edge_allowances=(Q(1, 8),),
            durations=(Q(1, 64),),
        )
        assert run.status == "completed" and run.reason is None
        assert family.initial_radius_upper <= run.final_radius_upper < 6 * tiny
        assert run.final_portrait_error_upper > 0


def test_zero_error_is_exact_without_inventing_a_positive_preparation_floor():
    family = _connected_family(form=Q(0), phase=Q(0))
    assert family.initial_norm_squared == family.initial_radius_upper == 0
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((0, 0), (0, 0)),),
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64),),
    )
    assert run.status == "completed"
    assert run.final_radius_upper == run.final_portrait_error_upper == 0


def test_tiny_conductance_preserves_form_only_energy_and_large_observation_dual():
    weight, tiny = Q(1, 2**200), Q(1, 2**1074)
    context = owner.prepare_metric_context(
        capacities=(1, 1), edges=((0, 1),), conductances=(weight,), degrees=(1, 1)
    )
    family = owner.prepare_metric_family(
        context,
        form_errors=((tiny, tiny),) * 2,
        phase_errors=((0, 0),) * 2,
        parent=0,
        child=1,
        reference_weight=1,
    )
    exact_radius = 2 * tiny / 2**100
    assert family.initial_norm_squared == 4 * weight * tiny**2
    assert exact_radius <= family.initial_radius_upper < 2 * exact_radius
    assert family.observation_form_dual == family.observation_factor == 1 / weight
    assert family.observation_root_upper >= 2**100
    coefficient = owner.bound_secant_metric(context, secant_intervals=((0, 0),))
    assert 2 / Q(2**100) <= coefficient.norm_upper < 3 / Q(2**100)
    assert coefficient.gamma_upper > 0
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((0, 0), (0, 0)),),
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64),),
    )
    assert run.status == "completed"
    assert 2 * tiny <= run.final_portrait_error_upper < 3 * tiny


def test_cut_asymmetric_means_and_child_offsets_keep_the_whole_initial_box():
    errors_x = ((Q(1, 8), Q(1, 4)), (Q(1, 16), Q(1, 8)))
    errors_theta = ((Q(1, 32), Q(1, 16)), (Q(1, 64), Q(1, 32)))
    child_x, child_theta = (Q(1, 4), Q(1, 8)), (Q(1, 8), Q(1, 16))
    family = owner.prepare_metric_family(
        _two_node(),
        form_errors=errors_x,
        phase_errors=errors_theta,
        parent=0,
        reference_weight=2,
        child_form_error=child_x,
        child_phase_error=child_theta,
    )
    assert family.form_mean_error == (Q(-3, 32), Q(3, 16))
    assert family.phase_mean_error == (Q(-3, 128), Q(3, 64))
    assert family.form_offset == (Q(-7, 16), Q(7, 32))
    assert family.phase_offset == (Q(-11, 64), Q(11, 128))
    assert family.observation_form_dual == Q(1, 16)
    assert family.observation_phase_dual == Q(1, 4)
    # The strongest affine offset corners are independently attainable.
    offsets = []
    all_pairs = errors_x + errors_theta + (child_x, child_theta)
    for choices in product((0, 1), repeat=6):
        values = tuple(
            -pair[0] if choice == 0 else pair[1]
            for choice, pair in zip(choices, all_pairs, strict=True)
        )
        u0, u1, v0, v1, uc, vc = values
        bx, bt = uc - (u0 + u1) / 2, vc - (v0 + v1) / 2
        offsets.append((bx, bt))
        assert family.form_offset[0] <= bx <= family.form_offset[1]
        assert family.phase_offset[0] <= bt <= family.phase_offset[1]
        assert bt**2 + 2 * bx**2 <= family.offset_norm_upper**2
    assert (
        min(x for x, _ in offsets),
        max(x for x, _ in offsets),
    ) == family.form_offset
    assert (
        min(t for _, t in offsets),
        max(t for _, t in offsets),
    ) == family.phase_offset
    assert family.offset_norm_upper > 0


def test_child_uncertainty_remains_even_when_the_cut_host_has_zero_error():
    tiny = Q(1, 2**1074)
    family = owner.prepare_metric_family(
        _two_node(),
        form_errors=((0, 0),) * 2,
        phase_errors=((0, 0),) * 2,
        parent=0,
        reference_weight=4,
        child_form_error=(tiny, 2 * tiny),
        child_phase_error=(tiny, tiny),
    )
    assert family.initial_norm_squared == 0
    assert family.form_offset == (-tiny, 2 * tiny)
    assert family.phase_offset == (-tiny, tiny)
    assert 4 * tiny < family.offset_norm_upper < 5 * tiny
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((0, 0), (0, 0)),),
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64),),
    )
    assert run.status == "completed" and run.final_radius_upper == 0
    assert run.final_portrait_error_upper == family.offset_norm_upper > 0


@pytest.mark.parametrize(
    "field,value",
    (
        ("form_errors", ((-1, 0), (0, 0))),
        ("phase_errors", ((False, 0), (0, 0))),
        ("phase_errors", ((0.0, 0), (0, 0))),
        ("form_errors", ((0, 0),)),
        ("reference_weight", 0),
        ("reference_weight", 1.0),
        ("parent", True),
        ("parent", 2),
        ("child", 0),
    ),
)
def test_family_inputs_cannot_change_physical_or_coordinate_contract(field, value):
    arguments = dict(
        form_errors=((0, 0),) * 2,
        phase_errors=((0, 0),) * 2,
        parent=0,
        child=1,
        reference_weight=4,
    )
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_metric_family(_two_node(), **arguments)


def test_cut_requires_both_declared_child_error_pairs():
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_metric_family(
            _two_node(),
            form_errors=((0, 0),) * 2,
            phase_errors=((0, 0),) * 2,
            parent=0,
            reference_weight=4,
            child_form_error=(0, 0),
        )


@pytest.mark.parametrize(
    "forge",
    (
        lambda family: replace(family),
        lambda family: replace(family, initial_norm_squared=Q(0)),
        lambda family: replace(family, phase_offset=(Q(0), Q(0))),
        lambda family: pickle.loads(pickle.dumps(family)),
    ),
)
def test_prepared_family_report_cannot_be_forged_into_a_live_capability(forge):
    family = _connected_family()
    with pytest.raises((TypeError, ValueError)):
        owner.bound_metric_cells(
            forge(family),
            phase_cells=(((0, 0), (0, 0)),),
            edge_allowances=(Q(1, 8),),
            durations=(Q(1, 64),),
        )


def test_initial_phase_membership_is_strict_and_failure_is_an_ordinary_result():
    allowance = Q(1, 8)
    family = _connected_family(form=Q(0), phase=allowance / 2)
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((0, 0), (0, 0)),),
        edge_allowances=(allowance,),
        durations=(Q(1, 64),),
    )
    assert run.status == "unavailable" and run.reason == "initial_domain"
    assert run.initial_edge_margins == (0,)
    assert run.steps == ()


def test_first_exit_failure_retains_the_failing_cell_and_stops_the_suffix():
    family = _connected_family(form=Q(3, 128), phase=Q(0))
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((3, 3), (0, 0)),) * 3,
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 2),) * 3,
    )
    assert all(value > 0 for value in run.initial_edge_margins)
    assert run.status == "unavailable" and run.reason == "first_exit"
    assert len(run.steps) == 1
    assert run.steps[0].domain_passed is False
    assert any(value <= 0 for value in run.steps[0].edge_margins)


def test_finite_inflation_keeps_negative_cosines_and_the_original_lift():
    family = _connected_family()
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((3, 3), (0, 0)),),
        edge_allowances=(Q(1, 16),),
        durations=(Q(1, 128),),
    )
    assert run.status == "completed"
    step = run.steps[0]
    assert step.inflated_edges == ((Q(47, 16), Q(49, 16)),)
    assert step.coefficient.secant_intervals[0][1] < 0
    assert step.coefficient.gamma_upper > 0


def test_cells_carry_the_prior_upper_radius_on_the_declared_chronology():
    family = _connected_family()
    durations = (Q(1, 128), Q(1, 64), Q(1, 256))
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((0, 0), (0, 0)),) * 3,
        edge_allowances=(Q(1, 8),),
        durations=durations,
    )
    assert run.status == "completed" and len(run.steps) == 3
    carried, time = family.initial_normalized_radius, Q(0)
    for index, step in enumerate(run.steps):
        assert (
            step.index == index
            and step.time == time
            and step.duration == durations[index]
        )
        assert step.normalized_radius_before == carried
        assert step.normalized_radius_after >= carried
        assert step.radius_upper == family.radius_scale * step.normalized_radius_after
        carried = step.normalized_radius_after
        time += durations[index]
    assert run.final_normalized_radius == carried


def test_scalar_work_failure_keeps_only_the_previously_certified_prefix():
    family = _connected_family()
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((3, 3), (0, 0)),) * 2,
        edge_allowances=(Q(1, 16),),
        durations=(Q(1, 128), Q(1)),
    )
    assert run.status == "unavailable" and run.reason == "scalar_work"
    assert len(run.steps) == 1 and run.steps[0].domain_passed


@pytest.mark.parametrize(
    "cells,allowances,durations",
    (
        ((), (Q(1, 8),), ()),
        ((((0, 0), (0, 0)),), (Q(1, 8),), ()),
        ((((0, 0),),), (Q(1, 8),), (Q(1, 64),)),
        ((((1, 0), (0, 0)),), (Q(1, 8),), (Q(1, 64),)),
        ((((0, 0), (0, 0)),), (0,), (Q(1, 64),)),
        ((((0, 0), (0, 0)),), (True,), (Q(1, 64),)),
        ((((0, 0), (0, 0)),), (Q(1, 8),), (0,)),
        ((((0, 0), (0, 0)),), (Q(1, 8),), (0.1,)),
        ((((0, 0), (0, 0)),) * 257, (Q(1, 8),), (Q(1, 512),) * 257),
    ),
)
def test_cell_schedule_and_domain_admission_cannot_drop_or_coerce_inputs(
    cells, allowances, durations
):
    with pytest.raises((TypeError, ValueError)):
        owner.bound_metric_cells(
            _connected_family(),
            phase_cells=cells,
            edge_allowances=allowances,
            durations=durations,
        )


def test_batch_admission_checks_later_cells_before_the_first_scalar_update(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the complete cell batch must be admitted before growth arithmetic")

    monkeypatch.setattr(owner, "scalar_affine_upper", forbidden)
    with pytest.raises(ValueError):
        owner.bound_metric_cells(
            _connected_family(),
            phase_cells=(((0, 0), (0, 0)), ((1, 0), (0, 0))),
            edge_allowances=(Q(1, 8),),
            durations=(Q(1, 64),) * 2,
        )


def test_scalar_owner_carries_one_radius_and_receives_zero_defect_without_refinement(
    monkeypatch,
):
    original = owner.scalar_affine_upper
    calls = []

    def record(radius, rate, duration, defect):
        result = original(radius, rate, duration, defect)
        calls.append((radius, rate, duration, defect, result))
        return result

    monkeypatch.setattr(owner, "scalar_affine_upper", record)
    family = _connected_family()
    run = owner.bound_metric_cells(
        family,
        phase_cells=(((3, 3), (0, 0)),) * 2,
        edge_allowances=(Q(1, 16),),
        durations=(Q(1, 128), Q(1)),
    )
    assert run.reason == "scalar_work" and len(calls) == 1
    assert calls[0][0] == family.initial_normalized_radius
    assert calls[0][2:4] == (Q(1, 128), Q(0))
    assert calls[0][4] == run.steps[0].normalized_radius_after


def test_decreasing_scalar_carry_is_rejected_instead_of_becoming_a_better_bound(
    monkeypatch,
):
    monkeypatch.setattr(owner, "scalar_affine_upper", lambda radius, *args: radius / 2)
    with pytest.raises(ArithmeticError, match="reduced"):
        owner.bound_metric_cells(
            _connected_family(),
            phase_cells=(((0, 0), (0, 0)),),
            edge_allowances=(Q(1, 8),),
            durations=(Q(1, 64),),
        )


def test_shared_kernel_failure_is_not_retried_or_reported_as_success(monkeypatch):
    calls = []

    def fail(*args):
        calls.append(args)
        raise ArithmeticError("constructed scalar failure")

    monkeypatch.setattr(owner, "scalar_affine_upper", fail)
    with pytest.raises(ArithmeticError, match="constructed"):
        owner.bound_metric_cells(
            _connected_family(),
            phase_cells=(((0, 0), (0, 0)),) * 3,
            edge_allowances=(Q(1, 8),),
            durations=(Q(1, 64),) * 3,
        )
    assert len(calls) == 1


def test_post_operation_bit_budget_preserves_only_the_accepted_prefix(monkeypatch):
    original = owner.scalar_affine_upper
    calls = []

    def oversized_second_result(*args):
        calls.append(args)
        return original(*args) if len(calls) == 1 else Q(1 << 65537)

    monkeypatch.setattr(owner, "scalar_affine_upper", oversized_second_result)
    run = owner.bound_metric_cells(
        _connected_family(),
        phase_cells=(((0, 0), (0, 0)),) * 3,
        edge_allowances=(Q(1, 8),),
        durations=(Q(1, 64),) * 3,
    )
    assert run.status == "unavailable" and run.reason == "intermediate_budget"
    assert run.attempted_cells == len(calls) == 2
    assert run.completed_cells == len(run.steps) == 1
    assert run.final_normalized_radius == run.steps[0].normalized_radius_after
    assert run.maximum_bits >= 65538


def test_component_size_and_edge_work_limits_reject_before_inverse(monkeypatch):
    def forbidden(*args):
        pytest.fail("work limits must precede exact inversion")

    monkeypatch.setattr(owner, "exact_matrix_inverse", forbidden)
    with pytest.raises(ValueError):
        owner.prepare_metric_context(
            capacities=(1,) * 22,
            edges=tuple((i, i + 1) for i in range(21)),
            conductances=(1,) * 21,
            degrees=(1,) + (2,) * 20 + (1,),
        )
    edges = tuple((i, j) for i in range(11) for j in range(i + 1, 11))[:54]
    with pytest.raises(ValueError):
        owner.prepare_metric_context(
            capacities=(1,) * 11,
            edges=edges,
            conductances=(1,) * 54,
            degrees=tuple(sum(i in edge for edge in edges) for i in range(11)),
        )


def test_wrong_shared_inverse_is_detected_by_an_exact_residual(monkeypatch):
    monkeypatch.setattr(
        owner, "exact_matrix_inverse", lambda matrix: ((Q(1), Q(0)), (Q(0), Q(1)))
    )
    with pytest.raises(ArithmeticError, match="residual"):
        _two_node()
