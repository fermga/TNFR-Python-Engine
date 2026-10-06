"""Picard/Taylor steps retaining a full positive-metric uncertainty ball.

The caller either supplies a proved logarithmic growth bound or requests an
exact matrix certificate computed from the whole-tube interval Jacobian.
Coordinate boxes are used for domain/derivative enclosures and observation
only; they never replace the retained metric radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exp_unit_bounds
from ._comparison_flow import MAX_COMPARISON_DIMENSION, _exact, _ordered
from ._exact_linear_algebra import exact_matrix_inverse, exact_symmetric_semidefinite
from ._interval_taylor import MAX_ORDER
from ._rational_interval import I, sqrt
from ._validated_taylor import flow_jets, interval_jacobian, picard_tube

METRIC_TAYLOR_METHOD = "retained_spd_ball_picard_taylor_dyadic128_scalar_growth_v1"
METRIC_TAYLOR_LMI_METHOD = (
    "retained_spd_ball_picard_taylor_dyadic128_tube_lmi_growth_v1"
)


@dataclass(frozen=True)
class MetricLogarithmicBound:
    """Exact sufficient logarithmic growth bound on a supplied convex tube."""

    tube: tuple[I, ...]
    metric: tuple[tuple[Q, ...], ...]
    jacobian_bounds: tuple[tuple[I, ...], ...]
    symmetric_jacobian_bounds: tuple[tuple[I, ...], ...]
    majorant: tuple[tuple[Q, ...], ...]
    rate_bounds: tuple[Q, Q]
    bisections_requested: int
    bisections_performed: int
    growth_rate_upper_bound: Q | None
    slack_matrix: tuple[tuple[Q, ...], ...]
    certified: bool
    reason: str | None


@dataclass(frozen=True)
class ValidatedMetricTaylorStep:
    """One conditional enclosure with its radius retained in the same metric."""

    time: Q
    duration: Q
    order: int
    metric: tuple[tuple[Q, ...], ...]
    initial_center: tuple[Q, ...]
    initial_radius: Q
    initial_box: tuple[I, ...]
    coordinate_projection_factors: tuple[Q, ...]
    tube: tuple[I, ...]
    picard_interior_margin: Q
    domain_lower_bounds: tuple[Q, ...]
    growth_rate_upper_bound: Q
    growth_factor_upper_bound: Q
    propagated_initial_radius: Q
    center_polynomial_bounds: tuple[I, ...]
    local_remainder_bounds: tuple[I, ...]
    center_solution_endpoint: tuple[I, ...]
    local_coordinate_error_bounds: tuple[Q, ...]
    local_metric_error_upper_bound: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q
    endpoint: tuple[I, ...]
    growth_certificate: MetricLogarithmicBound | None = None
    method: str = METRIC_TAYLOR_METHOD
    conditional_premises: tuple[str, ...] = (
        "flow_is_the_declared_smooth_autonomous_complete_field_on_the_whole_tube",
        "domain_margins_certify_all_consumed_model_and_growth_bound_hypotheses",
        "supremum_of_the_metric_logarithmic_Jacobian_norm_on_the_convex_tube_is_at_most_growth_rate",
        "initial_uncertainty_is_the_declared_metric_ball_not_its_coordinate_projection",
    )


def _admit_metric(metric, dimension):
    rows = tuple(_ordered(row, "metric row") for row in _ordered(metric, "metric"))
    if len(rows) != dimension or any(len(row) != dimension for row in rows):
        raise ValueError("metric must be square and match the center dimension")
    rows = tuple(tuple(_exact(value) for value in row) for row in rows)
    if not exact_symmetric_semidefinite(rows, strict=True):
        raise ValueError("metric must be symmetric positive definite")
    return rows


def _admit_rate_search(rate_bounds, bisections):
    if type(bisections) is not int or not 0 <= bisections <= 16:
        raise ValueError(
            "growth bisections must be an integer between zero and sixteen"
        )
    values = tuple(
        _exact(value) for value in _ordered(rate_bounds, "growth rate bounds")
    )
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError("growth rate bounds must be two ordered exact endpoints")
    return values


def certify_metric_logarithmic_bound(tube, metric, flow, *, rate_bounds, bisections=8):
    """Certify a whole-tube rate by exact PSD tests within a fixed bracket.

    For the symmetric interval S=J.T W+W J, its midpoint C plus the diagonal
    row sums of interval radii is a Loewner upper bound M. Indeed each symmetric
    off-diagonal error obeys 2|z_i*z_j|<=z_i^2+z_j^2. Thus 2*gamma*W-M PSD
    proves the actual field's logarithmic norm is at most gamma on the tube.

    Only the declared rational bracket is used. A failing upper endpoint
    returns an unavailable certificate, without expanding the bracket. Each
    successful bisection retains an exactly verified upper bound; the lower
    endpoint is a search parameter, not a claimed lower bound on the norm.
    The smooth-field and interval derivative contracts remain caller premises.
    """
    rates = _admit_rate_search(rate_bounds, bisections)
    tube = tuple(I.coerce(value) for value in _ordered(tube, "metric growth tube"))
    dimension = len(tube)
    if not 1 <= dimension <= MAX_COMPARISON_DIMENSION:
        raise ValueError("metric growth dimension outside the shared comparison domain")
    metric = _admit_metric(metric, dimension)
    jacobian = interval_jacobian(tube, flow)
    weighted = tuple(
        tuple(
            sum(
                (
                    metric[i][k] * jacobian[k][j]
                    for k in range(dimension)
                    if metric[i][k]
                ),
                I(0),
            )
            for j in range(dimension)
        )
        for i in range(dimension)
    )
    symmetric = tuple(
        tuple(weighted[i][j] + weighted[j][i] for j in range(dimension))
        for i in range(dimension)
    )
    radii = tuple(sum((value.radius for value in row), Q(0)) for row in symmetric)
    majorant = tuple(
        tuple(
            symmetric[i][j].midpoint + (radii[i] if i == j else 0)
            for j in range(dimension)
        )
        for i in range(dimension)
    )

    def slack(rate):
        return tuple(
            tuple(2 * rate * metric[i][j] - majorant[i][j] for j in range(dimension))
            for i in range(dimension)
        )

    lower, upper = rates
    matrix = slack(upper)
    certified = exact_symmetric_semidefinite(matrix)
    performed = 0
    if certified:
        lower_matrix = slack(lower)
        if exact_symmetric_semidefinite(lower_matrix):
            upper, matrix = lower, lower_matrix
        else:
            for _ in range(bisections):
                candidate = (lower + upper) / 2
                candidate_matrix = slack(candidate)
                performed += 1
                if exact_symmetric_semidefinite(candidate_matrix):
                    upper, matrix = candidate, candidate_matrix
                else:
                    lower = candidate
    return MetricLogarithmicBound(
        tube=tube,
        metric=metric,
        jacobian_bounds=jacobian,
        symmetric_jacobian_bounds=symmetric,
        majorant=majorant,
        rate_bounds=rates,
        bisections_requested=bisections,
        bisections_performed=performed,
        growth_rate_upper_bound=upper if certified else None,
        slack_matrix=matrix,
        certified=certified,
        reason=None if certified else "declared_growth_upper_endpoint_not_certified",
    )


def _growth_upper(exponent):
    if abs(exponent) > 1:
        raise ValueError("metric growth requires abs(growth_rate*duration)<=1")
    lower, upper = exp_unit_bounds(abs(exponent))
    return I(upper if exponent >= 0 else 1 / lower).hi


def _metric_error_upper(metric, coordinate_errors):
    scale = max(coordinate_errors)
    if scale == 0:
        return Q(0)
    normalized = tuple(value / scale for value in coordinate_errors)
    square = sum(
        (
            abs(value) * normalized[i] * normalized[j]
            for i, row in enumerate(metric)
            for j, value in enumerate(row)
        ),
        Q(0),
    )
    # Round only the normalized square. Rounding error^2 to the absolute
    # dyadic grid first would introduce a spurious square-root precision floor.
    return scale * sqrt(I(square)).hi


def validated_metric_taylor_step(
    center,
    radius,
    metric,
    duration,
    flow,
    domain,
    *,
    growth_rate=None,
    growth_rate_bounds=None,
    growth_bisections=8,
    order,
    time=Q(0),
    domain_failure="whole_tube_domain_not_admitted",
):
    """Return a retained-ball step, or a failed tube and explicit reason.

    Inputs are exact rationals; booleans and floating approximations are not
    silently admitted. Dimension/order/Picard work limits match the existing
    Taylor owner. The scalar exponential additionally requires
    ``abs(growth_rate*duration)<=1``; negative rates retain contraction.

    Exactly one of growth_rate or growth_rate_bounds is required. The former
    retains the caller's proof obligation; the latter computes the interval
    Jacobian/PSD certificate on the actual admitted tube, using at most sixteen
    declared bisections and no bracket expansion.

    The whole initial coordinate projection receives strict Picard admission,
    enclosing the center solution and every actual initial point.
    The endpoint radius is ``exp(gamma*h)*radius + local_metric_error``.
    The latter contains the full Taylor remainder and outward arithmetic,
    including rounding an exact nondyadic initial center. No observation box
    is converted back to a radius, and no tube intersection shrinks the ball.
    """
    center = tuple(_exact(value) for value in _ordered(center, "center"))
    dimension = len(center)
    if not 1 <= dimension <= MAX_COMPARISON_DIMENSION:
        raise ValueError(
            f"validated state dimension must lie between 1 and {MAX_COMPARISON_DIMENSION}"
        )
    radius, duration, time = map(_exact, (radius, duration, time))
    if radius < 0 or duration <= 0 or time < 0:
        raise ValueError("require nonnegative radius/time and positive duration")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    if (growth_rate is None) == (growth_rate_bounds is None):
        raise ValueError("supply exactly one growth rate or growth rate bracket")
    if type(growth_bisections) is not int or not 0 <= growth_bisections <= 16:
        raise ValueError(
            "growth bisections must be an integer between zero and sixteen"
        )
    if growth_rate_bounds is not None:
        growth_rate_bounds = _admit_rate_search(growth_rate_bounds, growth_bisections)
    else:
        growth_rate = _exact(growth_rate)
        growth = _growth_upper(growth_rate * duration)
    metric = _admit_metric(metric, dimension)
    inverse = exact_matrix_inverse(metric)
    projection = tuple(sqrt(I(inverse[i][i])).hi for i in range(dimension))
    box = tuple(
        I(value - factor * radius, value + factor * radius)
        for value, factor in zip(center, projection)
    )
    admission, failed, reason = picard_tube(
        box, duration, flow, domain, domain_failure=domain_failure
    )
    if admission is None:
        return None, failed, reason
    tube, margin, bounds = admission
    certificate = None
    try:
        if growth_rate_bounds is not None:
            certificate = certify_metric_logarithmic_bound(
                tube,
                metric,
                flow,
                rate_bounds=growth_rate_bounds,
                bisections=growth_bisections,
            )
            if not certificate.certified:
                return None, tube, certificate.reason
            growth_rate = certificate.growth_rate_upper_bound
            growth = _growth_upper(growth_rate * duration)
        series = flow_jets(tuple(I(value) for value in center), order, flow)
        remainder = tuple(
            row[-1] * duration ** (order + 1)
            for row in flow_jets(tube, order + 1, flow)
        )
        polynomial = []
        for row in series:
            value = row[-1]
            for coefficient in reversed(row[:-1]):
                value = value * duration + coefficient
            polynomial.append(value)
        center_endpoint = tuple(
            value + error for value, error in zip(polynomial, remainder)
        )
        if any(
            max(value.lo, bound.lo) > min(value.hi, bound.hi)
            for value, bound in zip(center_endpoint, tube)
        ):
            raise ArithmeticError("disjoint center endpoint and whole-time enclosures")
        endpoint_center = tuple(value.midpoint for value in center_endpoint)
        coordinate_error = tuple(value.radius for value in center_endpoint)
        local_error = _metric_error_upper(metric, coordinate_error)
        propagated = I(growth * radius).hi
        endpoint_radius = I(propagated + local_error).hi
        endpoint = tuple(
            I(value - factor * endpoint_radius, value + factor * endpoint_radius)
            for value, factor in zip(endpoint_center, projection)
        )
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return None, tube, f"Taylor_metric_unavailable: {exc}"
    return (
        ValidatedMetricTaylorStep(
            time=time,
            duration=duration,
            order=order,
            metric=metric,
            initial_center=center,
            initial_radius=radius,
            initial_box=box,
            coordinate_projection_factors=projection,
            tube=tube,
            picard_interior_margin=margin,
            domain_lower_bounds=bounds,
            growth_rate_upper_bound=growth_rate,
            growth_factor_upper_bound=growth,
            propagated_initial_radius=propagated,
            center_polynomial_bounds=tuple(polynomial),
            local_remainder_bounds=remainder,
            center_solution_endpoint=center_endpoint,
            local_coordinate_error_bounds=coordinate_error,
            local_metric_error_upper_bound=local_error,
            endpoint_center=endpoint_center,
            endpoint_radius=endpoint_radius,
            endpoint=endpoint,
            growth_certificate=certificate,
            method=(
                METRIC_TAYLOR_METHOD
                if certificate is None
                else METRIC_TAYLOR_LMI_METHOD
            ),
        ),
        None,
        None,
    )
