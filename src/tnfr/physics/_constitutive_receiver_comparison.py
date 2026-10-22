"""Static own-law transport around an already enclosed sine reference.

CONSTITUTIVE_RECEIVER_COMPARISON.md owns the comparison theorem. This owner
admits local reference primitives, narrows their observation boxes by two
derivative cones and propagates a full mean-zero error ball. It constructs
no reference trajectory, Taylor derivatives, source or capture certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._comparison_flow import scalar_affine_upper
from ..mathematics._rational_interval import I, pi_interval, sin, sqrt
from ..mathematics._validated_metric import _metric_error_upper
from ._composition_compatible_constitutive import (
    _amplitude,
    _deformation_current_bounds,
    _derive_composition_compatible_law,
)
from ._organized_receiver_enclosure_geometry import (
    _lift_receiver_reference,
    _observed_winding,
    _receiver_local_osl,
    _receiver_weighted_edges,
    _ReceiverLocalOSL,
)
from .relational_observations import _ordered

_SUBDIVISIONS = 8
_METRIC = tuple(tuple(Q(2 * (i == j)) for j in range(13)) for i in range(13))
_INFLATION = Q(5, 4)
_ADDITIVE_RADIUS = Q(1, 2**80)


def _center(values, label):
    raw = _ordered(values, label, limit=14)
    if len(raw) != 13:
        raise ValueError(f"{label} requires 13 reference coordinates")
    return tuple(_exact(value, f"{label}[{i}]") for i, value in enumerate(raw))


def _box(values, label):
    raw = _ordered(values, label, limit=14)
    if len(raw) != 13:
        raise ValueError(f"{label} requires 13 reference intervals")
    result = []
    for i, value in enumerate(raw):
        if not isinstance(value, I):
            raise TypeError(f"{label}[{i}] must be a rational interval")
        result.append(
            I(
                _exact(value.lo, f"{label}[{i}].lo"),
                _exact(value.hi, f"{label}[{i}].hi"),
            )
        )
    return tuple(result)


def _winding_observation(phases, radius):
    edge_error = sqrt(I(2)).hi * radius
    margins = tuple(
        min(
            pi_interval().lo
            - (phases[9 * region + (j + 1) % 9] - phases[9 * region + j]).abs_max
            - edge_error
            for j in range(9)
        )
        for region in range(3)
    )
    return (
        tuple(_observed_winding(phases, region, edge_error) for region in range(3)),
        margins,
    )


@dataclass(frozen=True)
class _ConstitutiveReferenceCone:
    """A reference observation cone and its full-dimensional error comparison."""

    time_bounds: tuple[Q, Q]
    phase_box: tuple[I, ...]
    forcing_norm_upper_bound: Q
    defect_upper_bound: Q
    initial_error: Q
    endpoint_error: Q
    whole_error: Q
    windings: tuple[int | None, ...] | None
    zero_branch_margins: tuple[Q, ...] | None


@dataclass(frozen=True)
class _ConstitutiveReceiverComparison:
    """Conditional local transport, with source and reference premises separate.

    Actual source errors occupy every global mean-zero phase direction. The
    13-coordinate sign reflection belongs only to the supplied reference.
    Both donor endpoints must have known nonunit winding for the aggregate
    static classification to pass. A consumer needs independently admitted
    previous donor identities to infer their loss.
    """

    initial_center: tuple[Q, ...]
    reference_tube: tuple[I, ...]
    reference_endpoint: tuple[I, ...]
    endpoint_center: tuple[Q, ...]
    incoming_error: Q
    deformation_bound: Q
    velocity_defect_bound: Q
    duration: Q
    reference_velocity_bounds: tuple[I, ...]
    deformation_jacobian_norm_bound: Q
    global_growth_bound: Q
    cones: tuple[_ConstitutiveReferenceCone, ...]
    prospective_radius: Q
    osl: _ReceiverLocalOSL
    growth_rate_upper_bound: Q
    strict_error_margin: Q
    whole_error: Q
    continuous_endpoint_error: Q
    recenter_radius: Q
    endpoint_error: Q | None
    whole_step_windings: tuple[int | None, ...] | None
    whole_step_zero_branch_margins: tuple[Q, ...] | None
    endpoint_windings: tuple[int | None, ...] | None
    endpoint_zero_branch_margins: tuple[Q, ...] | None
    receiver_zero_throughout: bool | None
    donor_endpoint_nonunit: bool | None
    status: str
    reason: str | None
    subdivisions: int = _SUBDIVISIONS
    scope: tuple[str, ...] = (
        "supplied_sine_reference_local_enclosure_is_a_retained_numerical_premise",
        "fixed_eight_forward_reverse_derivative_cones_without_reference_evolution",
        "own_law_residual_from_factored_deformation_current_with_node_cancellations",
        "fresh_sine_OSL_on_expanded_domain_plus_own_law_Jacobian_bound",
        "actual_errors_retain_all_26_global_mean_zero_phase_directions",
        "source_release_and_uniform_velocity_defect_require_separate_admission",
        "endpoint_winding_classification_does_not_supply_initial_donor_identity",
        "no_capture_event_or_physical_identification_verdict",
    )


def _compare_constitutive_receiver_step(
    *,
    initial_center,
    reference_tube,
    reference_endpoint,
    endpoint_center,
    incoming_error,
    deformation_bound,
    velocity_defect_bound,
    duration,
) -> _ConstitutiveReceiverComparison:
    """Re-admit local primitives and compare without producing a trajectory."""
    start = _center(initial_center, "initial_center")
    tube = _box(reference_tube, "reference_tube")
    end = _box(reference_endpoint, "reference_endpoint")
    center = _center(endpoint_center, "endpoint_center")
    radius = _exact(incoming_error, "incoming_error")
    bound = _amplitude(deformation_bound)
    defect = _exact(velocity_defect_bound, "velocity_defect_bound")
    h = _exact(duration, "duration")
    if min(radius, bound, defect) < 0 or h <= 0:
        raise ValueError("comparison bounds must be nonnegative and duration positive")
    sine = _derive_composition_compatible_law(deformation=0)
    boundary = _derive_composition_compatible_law(deformation=Q(1, 324))
    # The factor-current derivative has no independent coefficient catalogue.
    derivative_norm = 2 * sum(
        (
            n * abs((b - a) * 324)
            for (n, a), (_, b) in zip(
                sine.current_coefficients, boundary.current_coefficients
            )
        ),
        Q(0),
    )
    global_rate = 2 + derivative_norm * bound
    if global_rate * h > 1:
        raise ValueError("comparison requires global_growth_bound*duration<=1")
    if any(not box.contains(value) for box, value in zip(tube, start)):
        raise ValueError("initial center lies outside its supplied reference tube")

    phases = _lift_receiver_reference(tube)
    initial = _lift_receiver_reference(start)
    terminal = _lift_receiver_reference(end)
    edges = _receiver_weighted_edges()
    velocity = [I(0)] * 27
    for a, b, weight in edges:
        current = sin(phases[b] - phases[a]) * weight / 4
        velocity[a] += current
        velocity[b] -= current

    # This cache contains only rebuildable function values during one admitted
    # call. It never stores a source, geometry decision or observation verdict.
    correction = lru_cache(maxsize=2048)(_deformation_current_bounds)
    step = h / _SUBDIVISIONS
    phase_cones, forcing_bounds = [], []
    for index in range(_SUBDIVISIONS):
        a, b = index * step, (index + 1) * step
        cone = []
        for t, c, e, v in zip(phases, initial, terminal, velocity):
            forward, backward = c + I(a, b) * v, e - I(h - b, h - a) * v
            lower, upper = max(t.lo, forward.lo, backward.lo), min(
                t.hi, forward.hi, backward.hi
            )
            if lower > upper:
                raise ValueError("supplied reference has an empty derivative cone")
            cone.append(I(lower, upper))
        cone = tuple(cone)
        currents = [I(0)] * 27
        for left, right, weight in edges:
            current = correction(cone[right] - cone[left]) * weight / 4
            currents[left] += current
            currents[right] -= current
        forcing_bounds.append(sqrt(I(sum(value.abs_max**2 for value in currents))).hi)
        phase_cones.append(cone)

    prospective = I(
        _INFLATION
        * (
            scalar_affine_upper(
                radius, global_rate, h, defect + bound * max(forcing_bounds)
            )
            + _ADDITIVE_RADIUS
        )
    ).hi
    osl = _receiver_local_osl(phases, phase_radius=prospective)
    growth = osl.growth_rate_upper_bound + derivative_norm * bound
    current_error, whole = radius, radius
    errors = []
    for forcing in forcing_bounds:
        next_error = scalar_affine_upper(
            current_error, growth, step, defect + bound * forcing
        )
        interval_error = max(current_error, next_error)
        errors.append((current_error, next_error, interval_error))
        whole = max(whole, interval_error)
        current_error = next_error
    margin = prospective - whole
    accepted = osl.certified and margin > 0
    coordinate_errors = tuple(
        max(abs(box.lo - value), abs(box.hi - value)) for box, value in zip(end, center)
    )
    recenter = I(_metric_error_upper(_METRIC, coordinate_errors)).hi
    endpoint_error = I(current_error + recenter).hi if accepted else None
    cones = []
    for index, (cone, forcing, bounds) in enumerate(
        zip(phase_cones, forcing_bounds, errors)
    ):
        windings, margins = (
            _winding_observation(cone, bounds[2]) if accepted else (None, None)
        )
        cones.append(
            _ConstitutiveReferenceCone(
                time_bounds=(index * step, (index + 1) * step),
                phase_box=cone,
                forcing_norm_upper_bound=forcing,
                defect_upper_bound=defect + bound * forcing,
                initial_error=bounds[0],
                endpoint_error=bounds[1],
                whole_error=bounds[2],
                windings=windings,
                zero_branch_margins=margins,
            )
        )
    whole_windings, whole_margins = (
        _winding_observation(phases, whole) if accepted else (None, None)
    )
    endpoint_windings, endpoint_margins = (
        _winding_observation(_lift_receiver_reference(center), endpoint_error)
        if accepted
        else (None, None)
    )
    receiver = None
    donors = None
    if accepted:
        receiver_values = tuple(cone.windings[2] for cone in cones)
        receiver = (
            False
            if any(value not in (0, None) for value in receiver_values)
            else True if all(value == 0 for value in receiver_values) else None
        )
        donors = (
            True
            if all(value not in (1, None) for value in endpoint_windings[:2])
            else False if any(value == 1 for value in endpoint_windings[:2]) else None
        )
    return _ConstitutiveReceiverComparison(
        initial_center=start,
        reference_tube=tube,
        reference_endpoint=end,
        endpoint_center=center,
        incoming_error=radius,
        deformation_bound=bound,
        velocity_defect_bound=defect,
        duration=h,
        reference_velocity_bounds=tuple(velocity),
        deformation_jacobian_norm_bound=derivative_norm,
        global_growth_bound=global_rate,
        cones=tuple(cones),
        prospective_radius=prospective,
        osl=osl,
        growth_rate_upper_bound=growth,
        strict_error_margin=margin,
        whole_error=whole,
        continuous_endpoint_error=current_error,
        recenter_radius=recenter,
        endpoint_error=endpoint_error,
        whole_step_windings=whole_windings,
        whole_step_zero_branch_margins=whole_margins,
        endpoint_windings=endpoint_windings,
        endpoint_zero_branch_margins=endpoint_margins,
        receiver_zero_throughout=receiver,
        donor_endpoint_nonunit=donors,
        status="certified_local_comparison" if accepted else "unavailable",
        reason=None if accepted else "prospective_error_domain_not_closed",
    )
