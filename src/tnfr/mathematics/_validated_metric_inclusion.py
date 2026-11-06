"""Retained metric Taylor enclosures with bounded measurable disturbances.

The smooth drift owns every Taylor/Jacobian derivative. A deliberately wider
interval extension admits a tube for all disturbances; no derivative of the
disturbance is invented. The nominal certificate schema remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exp_unit_bounds
from ._comparison_flow import MAX_COMPARISON_DIMENSION, _exact, _ordered
from ._interval_taylor import Jet, MAX_ORDER
from ._rational_interval import I
from ._validated_metric import (
    ValidatedMetricTaylorStep,
    _admit_metric,
    _metric_coordinate_projection_factors,
    metric_taylor_observation_boxes,
    validated_metric_taylor_step,
)
from ._validated_taylor import picard_tube


METRIC_INCLUSION_METHOD = "retained_spd_ball_bounded_disturbance_taylor_v1"
_PREMISES = (
    "drift_is_the_declared_smooth_field_with_valid_interval_and_jet_extensions",
    "actual_path_is_absolutely_continuous_and_satisfies_drift_plus_measurable_disturbance",
    "disturbance_metric_norm_is_at_most_forcing_bound_on_the_whole_step",
    "effective_forcing_bound_is_valid_on_the_actual_path_in_the_admitted_tube",
    "growth_bound_covers_the_same_time_drift_Jacobian_on_the_whole_disturbed_convex_tube",
    "initial_uncertainty_is_the_declared_metric_ball_and_is_never_reboxed",
)


@dataclass(frozen=True)
class ValidatedMetricInclusionStep:
    """Nominal drift certificate and the additional actual-path allowance."""

    nominal_step: ValidatedMetricTaylorStep
    forcing_bound: Q
    effective_forcing_bound: Q
    forcing_gain_upper_bound: Q
    forcing_radius_upper_bound: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q
    endpoint: tuple[I, ...]
    forcing_refined: bool
    external_growth_certificate: object | None = None
    growth_proof_tube: tuple[I, ...] | None = None
    method: str = METRIC_INCLUSION_METHOD
    conditional_premises: tuple[str, ...] = _PREMISES


def metric_forcing_gain_upper(rate, duration):
    """Outward upper bound on integral_0^duration exp(rate*s) ds.

    Negative rates use a lower exponential bound before subtracting. Inputs
    obey the existing exact scalar/exponential contract, including |rate*h|<=1.
    """
    rate, duration = _exact(rate), _exact(duration)
    if duration < 0:
        raise ValueError("forcing gain requires nonnegative duration")
    exponent = rate * duration
    if abs(exponent) > 1:
        raise ValueError("forcing gain requires abs(rate*duration)<=1")
    if rate == 0 or duration == 0:
        return I(duration).hi
    _, upper = exp_unit_bounds(abs(exponent))
    gain = (upper - 1) / rate if rate > 0 else (1 - 1 / upper) / (-rate)
    return I(gain).hi


def _contained(inner, outer):
    return len(inner) == len(outer) and all(
        a.lo >= b.lo and a.hi <= b.hi for a, b in zip(inner, outer, strict=True)
    )


def validated_metric_inclusion_step(
    center,
    radius,
    metric,
    duration,
    flow,
    domain,
    *,
    forcing_bound,
    forcing_refinement=None,
    growth_rate=None,
    growth_rate_bounds=None,
    growth_bisections=8,
    growth_certifier=None,
    order,
    time=Q(0),
    time_dependent=False,
    domain_failure="whole_disturbed_tube_domain_not_admitted",
):
    """Enclose y'=flow(t,y)+d, ||d||_metric<=forcing_bound.

    ``forcing_refinement(tube)`` may supply a smaller exact nonnegative bound
    only after robust tube admission; its validity remains an explicit caller
    premise. It never shrinks the Picard tube. With prescribed time it receives
    ``(time_interval, tube)``. It need not describe a differentiable input.

    Exactly one growth mode is required: a proved scalar, a fixed LMI bracket,
    or ``growth_certifier(tube)``. An external report must bind its exact tube,
    metric, certified flag and rate. That path performs a robust Picard pass,
    certifies it once, then reuses the unchanged nominal kernel and verifies
    its resulting tube is contained in the certified tube. No bracket expands.

    Returns ``(certificate, None, None)`` or ``(None, failed_tube, reason)``.
    No returned box replaces the retained endpoint metric radius.
    """
    if type(time_dependent) is not bool:
        raise TypeError("time_dependent must be a Boolean")
    center = tuple(_exact(x) for x in _ordered(center, "center"))
    size = len(center)
    if not 1 <= size <= MAX_COMPARISON_DIMENSION:
        raise ValueError("inclusion dimension outside the shared metric domain")
    radius, duration, time, forcing_bound = map(
        _exact, (radius, duration, time, forcing_bound)
    )
    if min(radius, time, forcing_bound) < 0 or duration <= 0:
        raise ValueError("require nonnegative radius/time/forcing and positive duration")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("Taylor order outside the shared jet domain")
    if not callable(flow) or not callable(domain):
        raise TypeError("flow and domain must be callable")
    if forcing_refinement is not None and not callable(forcing_refinement):
        raise TypeError("forcing refinement must be callable")
    if growth_certifier is not None and not callable(growth_certifier):
        raise TypeError("growth certifier must be callable")
    if sum(x is not None for x in (growth_rate, growth_rate_bounds, growth_certifier)) != 1:
        raise ValueError("supply exactly one growth rate, bracket or certifier")
    if type(growth_bisections) is not int or not 0 <= growth_bisections <= 16:
        raise ValueError("growth bisections must be an integer between zero and sixteen")
    metric = _admit_metric(metric, size)
    projections = _metric_coordinate_projection_factors(metric)
    disturbance = tuple(I(-forcing_bound*p, forcing_bound*p) for p in projections)
    clock = I(time, time+duration) if time_dependent else None

    def enclosure_flow(*arguments):
        state = arguments[-1]
        jets = tuple(isinstance(x, Jet) for x in state)
        if any(jets):
            if not all(jets):
                raise TypeError("mixed interval/Jet state in inclusion adapter")
            return flow(*arguments)
        values = tuple(_ordered(flow(*arguments), "drift rates"))
        if len(values) != size:
            raise ValueError("field dimension differs from state")
        # This is a widened interval extension of the SAME smooth nominal
        # drift. Jet calls above retain only that drift and its derivatives.
        return tuple(I.coerce(x)+d for x, d in zip(values, disturbance, strict=True))

    proof_tube = None
    external = None
    if growth_certifier is not None:
        box = tuple(I(x-p*radius, x+p*radius) for x, p in zip(center, projections))
        admission, failed, reason = picard_tube(
            box, duration, enclosure_flow, domain,
            domain_failure=domain_failure, prescribed_time=clock,
        )
        if admission is None:
            return None, failed, reason
        proof_tube = admission[0]
        try:
            external = (growth_certifier(proof_tube) if clock is None
                        else growth_certifier(clock, proof_tube))
            if type(getattr(external, "certified", None)) is not bool:
                raise ValueError("growth certificate requires an explicit Boolean verdict")
            if (type(external.tube) is not tuple
                    or not all(type(x) is I for x in external.tube)):
                raise ValueError("growth certificate requires an exact interval tuple")
            admitted_external_metric = _admit_metric(external.metric, size)
            if external.tube != proof_tube or admitted_external_metric != metric:
                raise ValueError("external growth certificate tube/metric mismatch")
            if getattr(external, "prescribed_time_interval", None) != clock:
                raise ValueError("external growth certificate clock mismatch")
            if not external.certified:
                return None, proof_tube, (getattr(external, "reason", None)
                                          or "external_growth_bound_not_certified")
            growth_rate = _exact(external.growth_rate_upper_bound)
            metric_forcing_gain_upper(growth_rate, duration)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError) as exc:
            return None, proof_tube, f"external_growth_unavailable: {exc}"

    nominal, failed, reason = validated_metric_taylor_step(
        center, radius, metric, duration, enclosure_flow, domain,
        growth_rate=growth_rate, growth_rate_bounds=growth_rate_bounds,
        growth_bisections=growth_bisections, order=order, time=time,
        time_dependent=time_dependent, domain_failure=domain_failure,
    )
    if nominal is None:
        return None, failed, reason
    if proof_tube is not None and not _contained(nominal.tube, proof_tube):
        return None, nominal.tube, "nominal_tube_exceeds_external_growth_proof_tube"
    effective = forcing_bound
    try:
        if forcing_refinement is not None:
            effective = _exact(forcing_refinement(nominal.tube) if clock is None
                               else forcing_refinement(clock, nominal.tube))
            if not 0 <= effective <= forcing_bound:
                raise ValueError("refined forcing must lie between zero and the robust bound")
        gain = metric_forcing_gain_upper(nominal.growth_rate_upper_bound, duration)
        added = I(effective*gain).hi
        endpoint_radius = I(nominal.endpoint_radius+added).hi
        endpoint = tuple(
            I(x-p*endpoint_radius, x+p*endpoint_radius)
            for x, p in zip(nominal.endpoint_center, projections, strict=True)
        )
    except (TypeError, ValueError, ZeroDivisionError, ArithmeticError) as exc:
        return None, nominal.tube, f"inclusion_bound_unavailable: {exc}"
    return (
        ValidatedMetricInclusionStep(
            nominal_step=nominal,
            forcing_bound=forcing_bound,
            effective_forcing_bound=effective,
            forcing_gain_upper_bound=gain,
            forcing_radius_upper_bound=added,
            endpoint_center=nominal.endpoint_center,
            endpoint_radius=endpoint_radius,
            endpoint=endpoint,
            forcing_refined=forcing_refinement is not None,
            external_growth_certificate=external,
            growth_proof_tube=proof_tube,
        ),
        None,
        None,
    )


def metric_inclusion_observation_boxes(step, flow, *, bin_count=1):
    """Whole-bin actual-path boxes; requires an admitted same-field step.

    Like the nominal observation helper, this does not authenticate a saved
    certificate or re-prove its premises. Evidence readers must do that first.
    Forcing is integrated to each elapsed upper endpoint, never Taylor-scaled.
    """
    if type(step) is not ValidatedMetricInclusionStep:
        raise TypeError("an admitted ValidatedMetricInclusionStep is required")
    effective = _exact(step.effective_forcing_bound)
    robust = _exact(step.forcing_bound)
    if not 0 <= effective <= robust:
        raise ValueError("invalid forcing bound in admitted inclusion")
    nominal = step.nominal_step
    boxes = metric_taylor_observation_boxes(nominal, flow, bin_count=bin_count)
    projections = _metric_coordinate_projection_factors(nominal.metric)
    result = []
    for lower, upper, box in boxes:
        added = I(effective*metric_forcing_gain_upper(
            nominal.growth_rate_upper_bound, upper
        )).hi
        result.append((lower, upper, tuple(
            value+I(-p*added, p*added)
            for value, p in zip(box, projections, strict=True)
        )))
    return tuple(result)
