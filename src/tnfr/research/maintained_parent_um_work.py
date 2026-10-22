"""Complete conditional work and geometry of the retained parent's default UM.

This detached consumer rebuilds the ideal reset from primitive graph state.
It executes no event or flow. Rational enclosures describe the mathematical
map; a separate widened result is conditional on a future implementation
meeting the prospective per-output allowance. No observed error is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import I, cos, pi_interval, sin
from .maintained_parent_um import predict_parent_um

OLD_COUNT = 20
OLD_RADIUS_SQUARED = Q(1, 256)


@dataclass(frozen=True)
class StateBounds:
    """Fresh rows and means using this state's full support and capacities.

    The old-marked means restrict the full-support weights to the original
    twenty nodes. They are observations, not conserved quantities of the
    connected enlarged law. A zero isolate weight does not define a positive
    metric for all coordinates, even when the displayed total is positive.
    """

    degrees: tuple[int, ...]
    pressure: tuple[I, ...]
    form_rate: tuple[I, ...]
    phase_rate: tuple[I, ...]
    mean_weights: tuple[I, ...]
    total_mean_weight: I
    form_mean: I
    phase_mean: I
    old_marked_form_mean: I
    old_marked_phase_mean: I
    positive_mean_metric: bool


@dataclass(frozen=True)
class WorkBounds:
    """Edge-associated event work, without an assumed event reservoir.

    Old-edge form work is exactly zero because all forms and old weights
    are held. Capacity and stored pressure do not occur in this storage.
    Phase work on old edges is retained separately from every new-edge cost.
    """

    old_edge_phase: tuple[I, ...]
    new_edge_form: tuple[I, ...]
    new_edge_phase: tuple[I, ...]
    old_phase_total: I
    new_form_total: I
    new_phase_total: I
    total: I


@dataclass(frozen=True)
class SuccessorBounds:
    state: StateBounds
    work: WorkBounds
    old_radius_squared: I
    old_radius_excluded: bool


@dataclass(frozen=True)
class UmWorkPrediction:
    """Ideal and conditional represented-output statements remain separate."""

    reset: object
    before: StateBounds
    before_old_radius_squared: I
    ideal: SuccessorBounds
    conditional: SuccessorBounds
    numerical_allowance: Q


def _sum(values):
    return sum(values, I(0))


def _weighted_mean(values, weights):
    total = _sum(weights)
    if total.lo <= 0:
        raise ValueError("a displayed weighted mean requires positive total weight")
    return (
        _sum(value * weight for value, weight in zip(values, weights, strict=True))
        / total
    )


def _bound_state(epi, phases, capacities, edges, conductances):
    """Rebuild all constitutive rows, never reading stored-pressure relief."""
    count = len(epi)
    degrees = [0] * count
    sine_sum, form_gradient = [I(0) for _ in epi], [I(0) for _ in epi]
    if any(value.lo <= 0 for value in capacities):
        raise ValueError("the admitted successor requires strictly positive capacity")
    for (left, right), weight in zip(edges, conductances, strict=True):
        if I.coerce(weight).lo <= 0:
            raise ValueError("the admitted successor requires positive conductance")
        degrees[left] += 1
        degrees[right] += 1
        sine = sin(phases[right] - phases[left])
        gradient = weight * (epi[left] - epi[right])
        sine_sum[left] += sine
        sine_sum[right] -= sine
        form_gradient[left] += gradient
        form_gradient[right] -= gradient
    pi = pi_interval()
    pressure = tuple(
        value / (pi * degree) if degree else I(0)
        for value, degree in zip(sine_sum, degrees, strict=True)
    )
    form_rate = tuple(
        capacity * value for capacity, value in zip(capacities, pressure, strict=True)
    )
    phase_rate = tuple(
        capacity * gradient / (pi * degree) if degree else I(0)
        for capacity, gradient, degree in zip(
            capacities, form_gradient, degrees, strict=True
        )
    )
    weights = tuple(
        degree / capacity for degree, capacity in zip(degrees, capacities, strict=True)
    )
    return StateBounds(
        tuple(degrees),
        pressure,
        form_rate,
        phase_rate,
        weights,
        _sum(weights),
        _weighted_mean(epi, weights),
        _weighted_mean(phases, weights),
        _weighted_mean(epi[:OLD_COUNT], weights[:OLD_COUNT]),
        _weighted_mean(phases[:OLD_COUNT], weights[:OLD_COUNT]),
        all(degree > 0 for degree in degrees),
    )


def _storage_work(epi, phase_before, phase_after, old_edges, new_edges, new_weights):
    """Use direct jump algebra; independent storage intervals are not subtracted."""
    old_phase = tuple(
        cos(phase_before[right] - phase_before[left])
        - cos(phase_after[right] - phase_after[left])
        for left, right in old_edges
    )
    new_form = tuple(
        weight * (epi[left] - epi[right]) ** 2 / 2
        for (left, right), weight in zip(new_edges, new_weights, strict=True)
    )
    new_phase = tuple(
        1 - cos(phase_after[right] - phase_after[left]) for left, right in new_edges
    )
    old_phase_total = _sum(old_phase)
    new_form_total = _sum(new_form)
    new_phase_total = _sum(new_phase)
    return WorkBounds(
        old_phase,
        new_form,
        new_phase,
        old_phase_total,
        new_form_total,
        new_phase_total,
        old_phase_total + new_form_total + new_phase_total,
    )


def _old_radius(epi, phases):
    """Original marked geometry; no new support-dependent barrier is inferred.

    The reference uses mathematical 2*pi*k/5 on each original doubled C5.
    The supplied real lifts stay associated with their original node marks.
    Centering removes only the common form and phase origins. A new support,
    new weight, new capacity or enlarged coordinate is not silently absorbed
    into this original twenty-node radius-1/16 identity.
    """
    epi = epi[:OLD_COUNT]
    phases = phases[:OLD_COUNT]
    if len(epi) != OLD_COUNT or len(phases) != OLD_COUNT:
        raise ValueError("the old identity needs all twenty marked coordinates")
    form_mean = _sum(epi) / OLD_COUNT
    reference = tuple(
        (index % 10 // 2) * (2 * pi_interval() / 5) for index in range(OLD_COUNT)
    )
    errors = tuple(phase - base for phase, base in zip(phases, reference, strict=True))
    error_mean = _sum(errors) / OLD_COUNT
    return _sum((form - form_mean) ** 2 for form in epi) + _sum(
        (error - error_mean) ** 2 for error in errors
    )


def _successor(reset, phases, capacities, new_weights):
    work = _storage_work(
        reset.epi,
        tuple(I(value) for value in reset.phase_before),
        phases,
        reset.old_edges,
        reset.new_edges,
        new_weights,
    )
    state = _bound_state(
        reset.epi,
        phases,
        capacities,
        reset.old_edges + reset.new_edges,
        tuple(I(value) for value in reset.old_conductances) + new_weights,
    )
    radius = _old_radius(reset.epi, phases)
    return SuccessorBounds(state, work, radius, radius.lo > OLD_RADIUS_SQUARED)


def _widen(interval, allowance):
    return I(interval.lo - allowance, interval.hi + allowance)


def predict_parent_um_work(graph):
    """Re-admit graph primitives and bound the full default event analytically.

    Neither a cached reset nor a cached work/radius certificate is accepted.
    The widened result assumes the future event preserves forms, old weights
    and unmodified coordinates exactly, selects the proven support, and has
    absolute error at most ``numerical_allowance`` in each changed phase,
    capacity and new conductance. This is a conditional implementation budget,
    not an observed response, an arbitrary-libm guarantee or a new identity.
    """
    reset = predict_parent_um(graph)
    phases_before = tuple(I(value) for value in reset.phase_before)
    before = _bound_state(
        reset.epi,
        phases_before,
        tuple(I(value) for value in reset.capacity_before),
        reset.old_edges,
        tuple(I(value) for value in reset.old_conductances),
    )
    ideal = _successor(
        reset, reset.phase_after, reset.capacity_after, reset.new_conductances
    )
    allowance = reset.numerical_allowance
    phases = tuple(
        _widen(value, allowance) if index in reset.phase_changed_indices else value
        for index, value in enumerate(reset.phase_after)
    )
    capacities = tuple(
        _widen(value, allowance) if index in reset.capacity_changed_indices else value
        for index, value in enumerate(reset.capacity_after)
    )
    conditional = _successor(
        reset,
        phases,
        capacities,
        tuple(_widen(weight, allowance) for weight in reset.new_conductances),
    )
    return UmWorkPrediction(
        reset,
        before,
        _old_radius(reset.epi, phases_before),
        ideal,
        conditional,
        allowance,
    )
