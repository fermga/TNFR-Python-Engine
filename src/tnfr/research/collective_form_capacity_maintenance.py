"""Fixed-contact nonlinear maintenance with exact symmetry and metric error.

The weighted sine-law and symmetry owners admit primitives and supply the
field. This research adapter supplies only the declared contact metric,
identity observation and fixed-budget use of the shared validated solver.
"""

from __future__ import annotations

from fractions import Fraction as Q

from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_metric import (
    metric_taylor_observation_boxes,
    validated_metric_taylor_step,
)
from ..physics.relational_sine_comparison import _sine_model_coefficients
from ..physics.relational_sine_symmetry import (
    _support_involution_field,
    assess_sine_support_involution,
)

RADIUS = Q(1, 16)
DOMAIN_RADIUS = Q(1, 8)
DURATION = Q(40)
STEP = Q(1, 4)
ORDER = 12
OBSERVATION_SUBDIVISIONS = 8
GROWTH = Q(1, 5)


def _laplacian(size, edges):
    result = [[Q(0) for _ in range(size)] for _ in range(size)]
    for left, right, weight in edges:
        if left == right:
            continue
        result[left][left] += weight
        result[right][right] += weight
        result[left][right] -= weight
        result[right][left] -= weight
    return tuple(tuple(row) for row in result)


def admit_contact(source):
    """Rebuild the actual law and admit this contact's metric hypotheses."""
    permutation = (0, 1) + tuple(i + 1 if i % 2 == 0 else i - 1 for i in range(2, 20))
    reduction, flow = _support_involution_field(
        assess_sine_support_involution(source, permutation_indices=permutation)
    )
    source = reduction.source
    if _sine_model_coefficients(source.reference_model) != (Q(0), Q(1), Q(1)):
        raise ValueError("the contact metric requires the declared e=0, w=beta=1 law")
    if not reduction.source_trajectory_reduction_certified:
        raise ValueError("source is outside the exact weighted fixed family")
    if (
        source.nodes != tuple(range(20))
        or reduction.multiplicities != (1, 1) + (2,) * 9
    ):
        raise ValueError("contact order or retained cell multiplicities differ")
    old = {
        (i, j)
        for i in range(20)
        for j in range(i + 1, 20)
        if i // 10 == j // 10 and ((i % 10 // 2 - j % 10 // 2) % 5) in (1, 4)
    }
    added = {(0, 1), (0, 10), (0, 11), (1, 10), (1, 11)}
    if set(source.edges) != old | added:
        raise ValueError("contact support differs from the frozen 45-edge model")
    if any(not 0 < nu <= 1 for nu in source.capacity):
        raise ValueError("the growth proof requires positive capacities at most one")
    weighted, curvature = [], []
    for edge, weight in zip(source.edges, source.conductances, strict=True):
        if (edge in old and weight != 1) or not Q(3, 4) < weight <= 1:
            raise ValueError("actual conductance violates the contact comparison")
        i, j = (reduction.node_orbits[node] for node in edge)
        weighted.append((i, j, weight))
        curvature.append((i, j, Q(309, 1000) if edge in old else Q(1)))
    size = len(reduction.orbits)
    form = _laplacian(size, weighted)
    phase = _laplacian(size, curvature)
    rho = tuple(
        Q(m * degree) / nu
        for m, degree, nu in zip(
            reduction.multiplicities,
            reduction.reduced_degrees,
            reduction.reduced_capacity,
            strict=True,
        )
    )
    total = sum(rho)
    metric = tuple(
        tuple(
            block[i][j] + rho[i] * rho[j] / total if b == bb else Q(0)
            for bb in range(2)
            for j in range(size)
        )
        for b, block in enumerate((form, phase))
        for i in range(size)
    )
    # A fixed nearest-reference chart is selected before continuous evaluation.
    # It changes only the reference representative, not the supplied phase.
    pi = pi_interval()
    references = []
    for node, value in zip(
        reduction.representative_indices,
        reduction.source_phase_coordinates,
        strict=True,
    ):
        coefficient = Q(2 * (node % 10 // 2), 5)
        turns = (I(value) - coefficient * pi) / (2 * pi) + Q(1, 2)
        left, right = turns.lo // 1, turns.hi // 1
        if left != right:
            raise ValueError("source phase does not select a unique reference lift")
        references.append(coefficient + 2 * left)
    return reduction, flow, metric, tuple(references)


def identity_squared(values, multiplicities, reference_pi_coefficients):
    """Bound the original twenty-node centered identity without losing cells."""
    size = len(multiplicities)
    if len(values) != 2 * size or len(reference_pi_coefficients) != size:
        raise ValueError("identity dimensions differ")
    forms = tuple(I.coerce(value) for value in values[:size])
    phases = tuple(
        I.coerce(value) - coefficient * pi_interval()
        for value, coefficient in zip(
            values[size:], reference_pi_coefficients, strict=True
        )
    )
    total = sum(multiplicities)
    result = I(0)
    for channel in (forms, phases):
        mean = (
            sum((m * x for m, x in zip(multiplicities, channel, strict=True)), I(0))
            / total
        )
        result += sum(
            (m * (x - mean) ** 2 for m, x in zip(multiplicities, channel, strict=True)),
            I(0),
        )
    return result


def _observation_bins(step, flow, multiplicities, references):
    """Observe fixed subintervals of one validated step, without restarting it."""
    return tuple(
        identity_squared(box, multiplicities, references)
        for _, _, box in metric_taylor_observation_boxes(
            step, flow, bin_count=OBSERVATION_SUBDIVISIONS
        )
    )


def forecast_contact(source, *, progress=None):
    """One fixed 40-unit forecast, or an explicit earlier certified exit/failure.

    The initial mathematical state is exactly the represented event output.
    Its source belongs to the exact symmetry family. Solver uncertainty is
    carried in the 22-dimensional metric; it is never projected into the
    symmetric family from an independently uncertain 40-dimensional source.
    """
    reduction, flow, metric, references = admit_contact(source)
    sizes = reduction.multiplicities
    center = reduction.source_form_coordinates + reduction.source_phase_coordinates
    radius, records = Q(0), []
    initial = identity_squared(center, sizes, references)
    if initial.hi >= RADIUS**2:
        raise ValueError(
            "the complete word did not admit its endpoint to the identity ball"
        )

    def domain(values):
        return (DOMAIN_RADIUS**2 - identity_squared(values, sizes, references).hi,)

    status, reason = "finite_horizon_maintained", None
    for index in range(int(DURATION / STEP)):
        step, failed, reason = validated_metric_taylor_step(
            center,
            radius,
            metric,
            STEP,
            flow,
            domain,
            growth_rate=GROWTH,
            order=ORDER,
            time=index * STEP,
            domain_failure="whole_time_contact_metric_domain_not_admitted",
        )
        if step is None:
            status = "numerical_certificate_unavailable"
            break
        bins = _observation_bins(step, flow, sizes, references)
        endpoint = identity_squared(step.endpoint, sizes, references)
        records.append(
            {
                "time": step.time,
                "duration": step.duration,
                "initial_center": step.initial_center,
                "initial_metric_radius": step.initial_radius,
                "tube": step.tube,
                "picard_interior_margin": step.picard_interior_margin,
                "domain_lower_bounds": step.domain_lower_bounds,
                "local_remainder_bounds": step.local_remainder_bounds,
                "local_metric_error_upper_bound": step.local_metric_error_upper_bound,
                "endpoint_center": step.endpoint_center,
                "endpoint_metric_radius": step.endpoint_radius,
                "endpoint_identity_squared": endpoint,
                "whole_step_identity_squared_bins": bins,
            }
        )
        center, radius = step.endpoint_center, step.endpoint_radius
        if progress is not None:
            progress(index + 1, records[-1])
        if endpoint.lo > RADIUS**2:
            status, reason = (
                "finite_departure_certified",
                "endpoint_identity_lower_bound_exceeds_frozen_radius",
            )
            break
        if any(value.hi >= RADIUS**2 for value in bins):
            status = "identity_verdict_unavailable"
    else:
        failed = None
    return {
        "status": status,
        "reason": reason,
        "validated_duration": len(records) * STEP,
        "initial_identity_squared": initial,
        "metric": metric,
        "reference_phase_pi_coefficients": references,
        "multiplicities": sizes,
        "orbits": reduction.orbits,
        "initial_source": reduction.source,
        "records": tuple(records),
        "failed_tube": failed,
        "all_time_maintenance_claimed": False,
    }
