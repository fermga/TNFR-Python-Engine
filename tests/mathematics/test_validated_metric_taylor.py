"""Manufactured complete fields for retained-metric Taylor enclosures.

These generic analytic systems exercise the numerical owner; no research
preparation, scientific trajectory or retained producer is evaluated.
"""

from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._validated_metric import (
    _metric_error_upper,
    certify_metric_logarithmic_bound,
    validated_metric_taylor_step,
)
from tnfr.mathematics._validated_taylor import comparison_matrix, interval_jacobian


def _everywhere(_):
    return (Q(1),)


def _zero(state):
    return tuple(0 * value for value in state)


def _quadratic(matrix, vector):
    return sum(
        value * vector[i] * vector[j]
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(interval, value):
    return _mp(interval.lo) <= value <= _mp(interval.hi)


@pytest.mark.parametrize("decay", (Q(0), Q(1, 2)))
def test_rotating_ellipsoid_retains_radius_across_steps_without_reboxing(decay):
    # WJ+J^T W=-2*decay*W, exactly. The off-diagonal metric carries
    # correlations even when the coordinate projection is much larger.
    metric = ((Q(2), Q(1)), (Q(1), Q(2)))
    generator = ((-1 - decay, Q(-2)), (Q(2), 1 - decay))
    weighted = tuple(
        tuple(sum(metric[i][k] * generator[k][j] for k in range(2)) for j in range(2))
        for i in range(2)
    )
    assert all(
        weighted[i][j] + weighted[j][i] == -2 * decay * metric[i][j]
        for i in range(2)
        for j in range(2)
    )

    def flow(state):
        return tuple(
            sum((a * x for a, x in zip(row, state)), 0 * state[0]) for row in generator
        )

    initial, initial_radius = (Q(1), Q(2)), Q(1, 100)
    center, radius = initial, initial_radius
    step_size = Q(1, 16)
    for index in range(32):
        step, failed, reason = validated_metric_taylor_step(
            center,
            radius,
            metric,
            step_size,
            flow,
            _everywhere,
            growth_rate=-decay,
            order=10,
            time=index * step_size,
        )
        assert step is not None and failed is reason is None
        assert step.initial_center == center and step.initial_radius == radius
        assert step.picard_interior_margin > 0
        assert all(bound > 0 for bound in step.domain_lower_bounds)
        assert (
            step.endpoint_radius
            >= step.propagated_initial_radius + step.local_metric_error_upper_bound
        )
        center, radius = step.endpoint_center, step.endpoint_radius

    total_time = 32 * step_size
    with mp.workdps(85):
        angle = mp.sqrt(3) * _mp(total_time)
        c, s = mp.cos(angle), mp.sin(angle) / mp.sqrt(3)
        scale = mp.exp(-_mp(decay * total_time))
        # The exact flow is exp(-d*t)(cos(sqrt3*t)I+sin(sqrt3*t)J0/sqrt3).
        # Include non-axis initial boundary errors, not just the center orbit.
        for unit in (
            (mp.mpf(1) / mp.sqrt(2), 0),
            (0, mp.mpf(1) / mp.sqrt(2)),
            (mp.mpf(1) / mp.sqrt(6), mp.mpf(1) / mp.sqrt(6)),
            (mp.mpf(1) / mp.sqrt(2), -mp.mpf(1) / mp.sqrt(2)),
        ):
            start = tuple(
                _mp(value) + _mp(initial_radius) * error
                for value, error in zip(initial, unit)
            )
            exact = (
                scale * (c * start[0] + s * (-start[0] - 2 * start[1])),
                scale * (c * start[1] + s * (2 * start[0] + start[1])),
            )
            error = tuple(value - _mp(mid) for value, mid in zip(exact, center))
            assert (
                _quadratic(tuple(tuple(map(_mp, row)) for row in metric), error)
                <= _mp(radius) ** 2
            )
            assert all(
                _contains(bound, value) for bound, value in zip(step.endpoint, exact)
            )
        analytic_radius = _mp(initial_radius) * scale
        assert analytic_radius <= _mp(radius) < analytic_radius + mp.mpf("1e-9")

    # Replacing one projected axis box by its enclosing W-ball already
    # doubles this radius; the retained calculation has avoided that factor.
    axis_radius = step.coordinate_projection_factors[0] * radius
    assert _quadratic(metric, (axis_radius, axis_radius)) >= 4 * radius**2
    assert radius < initial_radius + Q(1, 10**9)


def test_nonlinear_rational_solution_is_enclosed_from_the_entire_initial_ball():
    center, radius, metric, duration = (Q(1),), Q(1, 16), ((Q(9),),), Q(1, 64)
    # On 1/2<x<2 the scalar derivative2x<4 proves the supplied bound.
    step, failed, reason = validated_metric_taylor_step(
        center,
        radius,
        metric,
        duration,
        lambda state: (state[0] ** 2,),
        lambda tube: (tube[0].lo - Q(1, 2), 2 - tube[0].hi),
        growth_rate=Q(4),
        order=8,
    )
    assert step is not None and failed is reason is None
    for offset in (-radius / 3, Q(0), radius / 3):
        initial = center[0] + offset
        exact = initial / (1 - duration * initial)
        assert 9 * (exact - step.endpoint_center[0]) ** 2 <= step.endpoint_radius**2
        assert step.endpoint[0].contains(exact)
    assert step.local_remainder_bounds[0].lo > 0
    assert step.propagated_initial_radius > radius


def test_nondyadic_center_rounding_is_part_of_local_metric_error():
    center = (Q(1, 3), Q(1, 7))
    metric = ((Q(3), Q(1)), (Q(1), Q(2)))
    step, failed, reason = validated_metric_taylor_step(
        center,
        Q(0),
        metric,
        Q(1, 8),
        _zero,
        _everywhere,
        growth_rate=Q(0),
        order=4,
    )
    assert step is not None and failed is reason is None
    assert step.propagated_initial_radius == 0
    assert step.local_metric_error_upper_bound > 0
    exact_error = tuple(value - mid for value, mid in zip(center, step.endpoint_center))
    assert _quadratic(metric, exact_error) <= step.endpoint_radius**2
    assert all(bound.contains(value) for bound, value in zip(step.endpoint, center))
    assert all(remainder == I(0) for remainder in step.local_remainder_bounds)


def test_maximum_shared_dimension_retains_every_coordinate_of_a_translation():
    center = tuple(Q(i - 12, 8) for i in range(24))
    rates = tuple(Q(2 * i + 1, 16) for i in range(24))
    metric = tuple(
        tuple(Q(i + 1) if i == j else Q(0) for j in range(24)) for i in range(24)
    )
    radius, duration = Q(1, 32), Q(1, 8)
    step, failed, reason = validated_metric_taylor_step(
        center,
        radius,
        metric,
        duration,
        lambda state: tuple(0 * value + rate for value, rate in zip(state, rates)),
        _everywhere,
        growth_rate=0,
        order=4,
    )
    assert step is not None and failed is reason is None
    assert step.endpoint_center == tuple(
        value + duration * rate for value, rate in zip(center, rates)
    )
    assert step.endpoint_radius == radius
    assert step.local_metric_error_upper_bound == 0
    assert len(step.endpoint) == len(step.tube) == 24


def test_ball_is_not_shrunk_to_the_picard_projection():
    # A deliberately loose, valid growth bound creates a large final ball.
    # Its parts outside the Picard tube cannot be removed by intersection.
    radius = Q(1, 16)
    step, failed, reason = validated_metric_taylor_step(
        (Q(0),),
        radius,
        ((Q(1),),),
        Q(1),
        _zero,
        _everywhere,
        growth_rate=Q(1),
        order=2,
    )
    assert step is not None and failed is reason is None
    assert step.endpoint_radius > 2 * radius
    assert step.endpoint[0].hi > step.tube[0].hi
    assert step.endpoint[0].lo < step.tube[0].lo


def test_positive_radius_below_interval_resolution_remains_positive():
    radius = Q(1, 2**200)
    step, _, _ = validated_metric_taylor_step(
        (Q(0),),
        radius,
        ((Q(1),),),
        Q(1, 8),
        _zero,
        _everywhere,
        growth_rate=Q(0),
        order=2,
    )
    assert step is not None
    assert step.initial_radius == radius
    assert step.endpoint_radius >= radius > 0


@pytest.mark.parametrize("power", (100, 120))
def test_metric_local_norm_preserves_small_error_scale_before_square_root(power):
    epsilon = Q(1, 2**power)
    metric = ((Q(2), Q(1)), (Q(1), Q(2)))
    # With errors(epsilon,epsilon/2), the exact positive-corner norm is
    # sqrt(7/2)*epsilon. Squaring before dyadic rounding would lose this scale.
    result = _metric_error_upper(metric, (epsilon, epsilon / 2))
    assert result**2 >= Q(7, 2) * epsilon**2
    assert result < 2 * epsilon


def test_whole_tube_failure_remains_unavailable_without_retrying_the_step():
    step, failed, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(0),
        ((Q(1),),),
        Q(1),
        lambda state: (state[0] * 0 + 1,),
        lambda tube: (1 - tube[0].hi,),
        growth_rate=Q(0),
        order=4,
        domain_failure="growth_proof_domain_reached",
    )
    assert step is None and failed is not None
    assert failed[0].hi >= 1
    assert reason == "growth_proof_domain_reached"


def test_inflation_does_not_replace_strict_picard_inclusion():
    step, failed, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(0),
        ((Q(1),),),
        Q(1),
        lambda state: (state[0] + 1,),
        _everywhere,
        growth_rate=Q(1),
        order=4,
    )
    assert step is None and failed is not None
    assert reason == "strict_Picard_inclusion_not_resolved"


@pytest.mark.parametrize(
    "field,value",
    [
        ("radius", True),
        ("radius", -1),
        ("radius", 0.1),
        ("duration", True),
        ("duration", 0),
        ("duration", -1),
        ("time", True),
        ("time", -1),
        ("growth_rate", True),
        ("growth_rate", float("nan")),
        ("growth_rate", float("inf")),
        ("growth_rate", 1.0),
        ("order", True),
        ("order", 0),
        ("order", 17),
    ],
)
def test_invalid_scalar_evidence_is_rejected(field, value):
    values = dict(radius=Q(0), duration=Q(1, 8), time=Q(0), growth_rate=Q(0), order=4)
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        validated_metric_taylor_step(
            (Q(0),), metric=((Q(1),),), flow=_zero, domain=_everywhere, **values
        )


@pytest.mark.parametrize(
    "metric",
    [
        ((1, 0), (0, 0)),
        ((1, 0), (0, -1)),
        ((1, 2), (0, 1)),
        ((1, 2), (2, 1)),
        ((True, 0), (0, 1)),
        ((1.0, 0), (0, 1)),
        ((1,),),
        ((1, 0),),
        {0: (1, 0), 1: (0, 1)},
    ],
)
def test_metric_must_be_ordered_exact_symmetric_positive_definite(metric):
    with pytest.raises((TypeError, ValueError)):
        validated_metric_taylor_step(
            (Q(0), Q(0)),
            Q(0),
            metric,
            Q(1, 8),
            _zero,
            _everywhere,
            growth_rate=0,
            order=4,
        )


@pytest.mark.parametrize(
    "center", [(), {Q(0)}, {0: Q(0)}, (True,), (0.0,), (I(0),), (Q(0),) * 25]
)
def test_center_has_exact_ordered_scalars_within_the_shared_dimension_limit(center):
    with pytest.raises((TypeError, ValueError)):
        validated_metric_taylor_step(
            center,
            Q(0),
            ((Q(1),),),
            Q(1, 8),
            _zero,
            _everywhere,
            growth_rate=0,
            order=4,
        )


@pytest.mark.parametrize("growth", (Q(-9), Q(9)))
def test_scalar_exponential_work_limit_applies_to_both_time_signs(growth):
    with pytest.raises(ValueError, match="growth_rate"):
        validated_metric_taylor_step(
            (Q(0),),
            Q(0),
            ((Q(1),),),
            Q(1, 8),
            _zero,
            _everywhere,
            growth_rate=growth,
            order=4,
        )


@pytest.mark.parametrize("margin", (True, 1.0, float("nan")))
def test_domain_evidence_cannot_be_replaced_by_boolean_or_floating_values(margin):
    with pytest.raises(TypeError):
        validated_metric_taylor_step(
            (Q(0),),
            Q(0),
            ((Q(1),),),
            Q(1, 8),
            _zero,
            lambda _: (margin,),
            growth_rate=0,
            order=4,
        )


def test_missing_field_coordinate_is_explicitly_unavailable():
    step, failed, reason = validated_metric_taylor_step(
        (Q(1), Q(2)),
        Q(0),
        ((Q(1), Q(0)), (Q(0), Q(1))),
        Q(1, 8),
        lambda state: (state[0],),
        _everywhere,
        growth_rate=0,
        order=4,
    )
    assert step is None and failed is not None
    assert "field dimension differs" in reason


def test_signed_jacobian_owner_preserves_rotation_cancellation():
    metric = ((Q(2), Q(1)), (Q(1), Q(2)))
    tube = (I(-2, 3), I(-4, 5))

    def flow(state):
        x, y = state
        return (-x - 2 * y, 2 * x + y)

    assert interval_jacobian(tube, flow) == ((I(-1), I(-2)), (I(2), I(1)))
    assert comparison_matrix(tube, flow) == ((-1, 2), (2, 1))
    bound = certify_metric_logarithmic_bound(
        tube, metric, flow, rate_bounds=(-1, 1), bisections=8
    )
    assert bound.certified and bound.growth_rate_upper_bound == 0
    assert bound.majorant == ((0, 0), (0, 0))
    step, failed, reason = validated_metric_taylor_step(
        (Q(1), Q(2)),
        Q(1, 100),
        metric,
        Q(1, 16),
        flow,
        _everywhere,
        growth_rate_bounds=(-1, 1),
        growth_bisections=8,
        order=8,
    )
    assert step is not None and failed is reason is None
    assert step.growth_certificate.certified
    assert step.growth_certificate.tube == step.tube
    assert step.growth_rate_upper_bound == 0
    assert (
        step.initial_radius
        <= step.propagated_initial_radius
        < step.initial_radius + Q(1, 2**128)
    )


def test_nonlinear_whole_tube_majorant_bounds_actual_symmetric_derivatives():
    # J=[[0,2y],[-1,0]]. For y in[1,2], S has off-diagonal range[1,3].
    # Its midpoint2 plus diagonal inflation1 has largest eigenvalue3.
    metric = ((Q(1), Q(0)), (Q(0), Q(1)))
    tube = (I(0, 1), I(1, 2))
    bound = certify_metric_logarithmic_bound(
        tube,
        metric,
        lambda state: (state[1] ** 2, -state[0]),
        rate_bounds=(0, 2),
        bisections=2,
    )
    assert bound.certified and bound.growth_rate_upper_bound == Q(3, 2)
    assert bound.majorant == ((1, 2), (2, 1))
    assert bound.slack_matrix == ((2, -2), (-2, 2))
    for y in (Q(1), Q(5, 4), Q(2)):
        actual = ((Q(0), 2 * y - 1), (2 * y - 1, Q(0)))
        for direction in ((Q(1), Q(1)), (Q(2), Q(-3)), (Q(-7, 3), Q(-2))):
            assert _quadratic(actual, direction) <= _quadratic(
                bound.majorant, direction
            )
            assert _quadratic(
                actual, direction
            ) <= 2 * bound.growth_rate_upper_bound * _quadratic(metric, direction)
    # The upper-bound corner attains gamma3/2 in direction(1,1), so a
    # certificate based only on the midpoint y=3/2 would be unsound here.
    assert _quadratic(((0, 3), (3, 0)), (1, 1)) == 6


@pytest.mark.parametrize("refinements", (0, 8, 16))
def test_fixed_rational_search_certifies_the_analytic_rate_without_expansion(
    refinements,
):
    bound = certify_metric_logarithmic_bound(
        (I(-1, 1), I(-1, 1)),
        ((1, 0), (0, 1)),
        lambda state: (Q(2, 3) * state[0], -state[1]),
        rate_bounds=(0, 1),
        bisections=refinements,
    )
    assert bound.certified
    assert Q(2, 3) <= bound.growth_rate_upper_bound <= Q(2, 3) + Q(1, 2**refinements)
    assert bound.bisections_performed == refinements
    assert bound.rate_bounds == (0, 1)


def test_negative_computed_rate_retains_true_contraction():
    step, failed, reason = validated_metric_taylor_step(
        (Q(1),),
        Q(1, 8),
        ((Q(1),),),
        Q(1, 8),
        lambda state: (-state[0],),
        _everywhere,
        growth_rate_bounds=(-2, 0),
        growth_bisections=8,
        order=8,
    )
    assert step is not None and failed is reason is None
    assert step.growth_certificate.growth_rate_upper_bound == -1
    assert step.propagated_initial_radius < step.initial_radius
    with mp.workdps(85):
        for start in (Q(7, 8), Q(9, 8)):
            assert _contains(step.endpoint[0], _mp(start) * mp.exp(-mp.mpf(1) / 8))


def test_failed_declared_upper_rate_is_unavailable_not_an_expanded_search():
    bound = certify_metric_logarithmic_bound(
        (I(-1, 1),),
        ((Q(1),),),
        lambda state: (2 * state[0],),
        rate_bounds=(0, 1),
        bisections=8,
    )
    assert not bound.certified and bound.growth_rate_upper_bound is None
    assert bound.slack_matrix == ((-2,),)
    assert bound.rate_bounds == (0, 1) and bound.bisections_performed == 0
    step, failed, reason = validated_metric_taylor_step(
        (Q(0),),
        Q(0),
        ((Q(1),),),
        Q(1, 8),
        lambda state: (2 * state[0],),
        _everywhere,
        growth_rate_bounds=(0, 1),
        order=4,
    )
    assert step is None and failed is not None
    assert reason == "declared_growth_upper_endpoint_not_certified"


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"growth_rate": 0, "growth_rate_bounds": (0, 1)},
        {"growth_rate_bounds": (1, 0)},
        {"growth_rate_bounds": (0,)},
        {"growth_rate_bounds": (False, 1)},
        {"growth_rate_bounds": (0, 1.0)},
        {"growth_rate_bounds": {0, 1}},
        {"growth_rate_bounds": (0, 1), "growth_bisections": True},
        {"growth_rate_bounds": (0, 1), "growth_bisections": -1},
        {"growth_rate_bounds": (0, 1), "growth_bisections": 17},
    ],
)
def test_computed_growth_configuration_is_admitted_before_execution(arguments):
    with pytest.raises((TypeError, ValueError)):
        validated_metric_taylor_step(
            (Q(0),),
            Q(0),
            ((Q(1),),),
            Q(1, 8),
            _zero,
            _everywhere,
            order=4,
            **arguments,
        )


def test_one_static_corridor_domain_has_a_certified_rate_outside_the_local_saddle_box():
    # This is a static derivative/domain check, not a trajectory, parameter
    # search, observed crossing or reserved scientific response.
    import networkx as nx

    from tnfr.dynamics.relational import RelationalExchangeModel
    from tnfr.physics.relational_sine_comparison import (
        _comparison_neighbors,
        bound_relational_sine_exchange,
    )
    from tnfr.physics.relational_sine_metric_forecast import _saddle_flow
    from tnfr.physics.relational_sine_sensitivity import assess_sine_saddle_sensitivity

    graph = nx.cycle_graph(5)
    graph.add_edges_from((i, i + 5) for i in range(5))
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    source = bound_relational_sine_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_weight=1, phase_domain="regular"
        ),
    )
    sensitivity = assess_sine_saddle_sensitivity(source, cycle=range(5))
    phases = (Q(-9, 5), Q(-9, 10), Q(0), Q(9, 10), Q(9, 5)) * 2
    tube = (I(0),) * 10 + tuple(
        I(value - Q(1, 10000), value + Q(1, 10000)) for value in phases
    )
    neighbors = _comparison_neighbors(source)
    bound = certify_metric_logarithmic_bound(
        tube,
        sensitivity.full_metric,
        lambda values: _saddle_flow(
            values, source=source, neighbors=neighbors, direction=1
        ),
        rate_bounds=(0, 1),
        bisections=8,
    )
    assert bound.certified and bound.growth_rate_upper_bound < Q(1)
    assert len(bound.jacobian_bounds) == 20
    with mp.workdps(85):
        assert abs(_mp(phases[0]) + 2 * mp.pi / 3) > mp.mpf(".29")
        # Directly differentiate the original-time full nodal rows, including
        # contacts. This also detects an accidental second division by pi.
        for i in graph:
            for j in graph:
                form_derivative = (
                    -sum(mp.cos(_mp(phases[k] - phases[i])) for k in graph[i])
                    / graph.degree[i]
                    if i == j
                    else (
                        mp.cos(_mp(phases[j] - phases[i])) / graph.degree[i]
                        if graph.has_edge(i, j)
                        else mp.mpf(0)
                    )
                ) / mp.pi
                assert _contains(bound.jacobian_bounds[i][10 + j], form_derivative)
