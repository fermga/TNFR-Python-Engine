"""Validated reference transit and full-law capture of the two-port C9 family.

Only the nominal gradient reference is folded by an exact affine reflection.
The actual form/phase family retains all thirty-six independent source errors.
Shared retained-metric Taylor steps provide the reference evidence; analytic
comparison, an extra unit of slow time and a local storage barrier close the
complete-law argument. No solver retry or supplied endpoint is accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sin, sqrt
from ..mathematics._validated_metric import validated_metric_taylor_step
from .relational_sine_two_port_compatibility import (
    SineTwoPortCompatibility,
    assess_sine_two_port_compatibility,
)
from .relational_sine_two_port_transit import (
    SineTwoPortTransit,
    assess_sine_two_port_transit,
)

__all__ = ("SineTwoPortCapture", "assess_sine_two_port_capture")

_LOGGER = logging.getLogger(__name__)
_REFERENCE_MARGIN = Q(1, 2048)
_REPRESENTATIVES = (0, 2, 3, 4, 9, 11, 12, 13)
_FOLDED_WEIGHTS = (Q(6), Q(4), Q(4), Q(4)) * 2


@dataclass(frozen=True)
class _FoldedReference:
    """Exact affine geometry rebuilt from a freshly admitted preparation."""

    permutation: tuple[int, ...]
    representatives: tuple[int, ...]
    reconstruction_matrix: tuple[tuple[Q, ...], ...]
    metric: tuple[tuple[Q, ...], ...]
    angle_turns: tuple[Q, ...]
    angle_coefficients: tuple[tuple[Q, ...], ...]
    edge_angle_indices: tuple[int, ...]
    edge_angle_signs: tuple[int, ...]
    current_coefficients: tuple[tuple[Q, ...], ...]


def _folded_reference(preparation):
    """Preserve nominal affine phase symmetry without projecting actual errors."""
    permutation = tuple(9 * (i // 9) + (1 - i % 9) % 9 for i in range(18))
    reconstruction = tuple(
        tuple(Q(int(i == r) - int(i == permutation[r])) for r in _REPRESENTATIVES)
        for i in range(18)
    )
    keys, key_indices, edge_indices, signs = [], {}, [], []
    for (i, j), turn in zip(preparation.geometry.edges, preparation.nominal_edge_turns):
        coefficients = tuple(
            b - a for a, b in zip(reconstruction[i], reconstruction[j])
        )
        first = next(value for value in (turn, *coefficients) if value)
        sign = 1 if first > 0 else -1
        key = (sign * turn, tuple(sign * value for value in coefficients))
        if key not in key_indices:
            key_indices[key] = len(keys)
            keys.append(key)
        edge_indices.append(key_indices[key])
        signs.append(sign)
    currents = []
    for node in _REPRESENTATIVES:
        row = [Q(0)] * len(keys)
        for edge, (index, sign) in enumerate(zip(edge_indices, signs)):
            row[index] -= Q(
                preparation.geometry.incidence[node][edge] * sign,
                preparation.degrees[node],
            )
        currents.append(tuple(row))
    metric = tuple(
        tuple(weight if i == j else Q(0) for j in range(8))
        for i, weight in enumerate(_FOLDED_WEIGHTS)
    )
    return _FoldedReference(
        permutation,
        _REPRESENTATIVES,
        reconstruction,
        metric,
        tuple(key[0] for key in keys),
        tuple(key[1] for key in keys),
        tuple(edge_indices),
        tuple(signs),
        tuple(currents),
    )


def _reference_field(folded):
    """Sparse interval/jet rates and all full-support acute tube premises."""
    pi = pi_interval()
    bases = tuple(2 * pi * turn for turn in folded.angle_turns)
    sparse_angles = tuple(
        tuple((i, value) for i, value in enumerate(row) if value)
        for row in folded.angle_coefficients
    )
    sparse_currents = tuple(
        tuple((i, value) for i, value in enumerate(row) if value)
        for row in folded.current_coefficients
    )

    @lru_cache(maxsize=17)
    def constants(order):
        return tuple(Jet.constant(value, order) for value in bases)

    def angles(values):
        jet = isinstance(values[0], Jet)
        initial = constants(values[0].order) if jet else bases
        return tuple(
            base + sum((coefficient * values[i] for i, coefficient in row), 0)
            for base, row in zip(initial, sparse_angles)
        )

    @lru_cache(maxsize=32)
    def flow(values):
        jet = isinstance(values[0], Jet)
        sine = jet_sin if jet else sin
        zero = Jet.constant(0, values[0].order) if jet else I(0)
        currents = tuple(sine(value) for value in angles(values))
        return tuple(
            sum((coefficient * currents[i] for i, coefficient in row), zero)
            for row in sparse_currents
        )

    def domain(values):
        margins = tuple(
            (pi / 2 - abs(value)).lo - _REFERENCE_MARGIN for value in angles(values)
        )
        return tuple(margins[index] for index in folded.edge_angle_indices)

    return flow, domain


@dataclass(frozen=True)
class SineTwoPortReferenceStep:
    """Compact projection of a successful shared retained-metric Taylor step.

    The fixed metric, field, order and growth zero belong to the enclosing
    report. Centers, retained radii, whole tubes, strict inclusion/domain
    margins and local metric errors suffice to reconstruct the finite chain.
    This projection neither replaces a failed step nor reboxes uncertainty.
    """

    time: Q
    duration: Q
    initial_center: tuple[Q, ...]
    initial_radius: Q
    tube: tuple[I, ...]
    picard_interior_margin: Q
    domain_lower_bounds: tuple[Q, ...]
    local_metric_error_upper_bound: Q
    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q


def _compact_step(step):
    return SineTwoPortReferenceStep(
        step.time,
        step.duration,
        step.initial_center,
        step.initial_radius,
        step.tube,
        step.picard_interior_margin,
        step.domain_lower_bounds,
        step.local_metric_error_upper_bound,
        step.endpoint_center,
        step.endpoint_radius,
    )


@dataclass(frozen=True)
class SineTwoPortCapture:
    """Same-preparation conditional capture after a declared finite transit.

    The folded reference has no preparation errors. All actual errors enter
    the complete-law energy comparison and retain memberwise conserved means.
    Numerical endpoint boxes only observe the retained reference metric ball.
    Failed or partial validation never acquires the requested endpoint status.
    """

    preparation: SineTwoPortTransit
    target: SineTwoPortCompatibility | None
    reference_duration: Q
    time_step: Q
    order: int
    max_steps: int
    folded_reference: _FoldedReference
    initial_excess_storage_upper_bound: Q
    full_slow_horizon: Q
    full_horizon_pi_squared_coefficient: Q
    joint_error_candidate: Q
    scaled_form_norm_candidate: Q
    phase_error_candidate: Q
    comparison_margin: Q
    target_acute_margin_lower_bound: Q | None
    reference_steps: tuple[SineTwoPortReferenceStep, ...]
    validated_reference_duration: Q
    reference_endpoint_center: tuple[Q, ...]
    reference_endpoint_radius: Q
    reference_minimum_acute_margin: Q | None
    reference_target_distance_upper_bound: Q | None
    failed_tube: tuple[I, ...] | None
    endpoint_phase_distance_upper_bound: Q | None
    endpoint_relative_form_norm_upper_bound: Q | None
    endpoint_excess_storage_upper_bound: Q | None
    capture_storage_margin: Q | None
    preparation_admitted: bool
    target_admitted: bool
    reference_validated: bool
    reference_endpoint_certified: bool
    full_comparison_certified: bool
    capture_certified: bool
    status: str
    unavailable_reasons: tuple[str, ...]
    root_outer_refinements: int = 32
    root_inner_refinements: int = 64
    analytic_tail_slow_duration: Q = Q(1)
    reference_margin_threshold: Q = _REFERENCE_MARGIN
    reference_target_distance_threshold: Q = Q(1, 2048)
    tail_reference_forcing_upper_bound: Q = Q(1, 1024)
    fast_tail_decay_upper_bound: Q = Q(1, 2**512)
    capture_phase_radius: Q = Q(1, 12)
    capture_cosine_lower_bound: Q = Q(1, 25)
    capture_barrier_lower_bound: Q = Q(1, 648000)
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_two_port_complete_sine_law_midpoint_preparation_and_full_source_errors",
        "nominal_reference_only_uses_exact_affine_reflection_eight_coordinate_fold",
        "actual_thirty_six_coordinates_and_memberwise_conserved_means_are_retained",
        "fresh_primitive_preparation_and_target_proofs_no_incoming_evaluated_report",
        "shared_Picard_Taylor_metric_steps_with_acute_growth_zero_and_fixed_budget",
        "all_twenty_edge_domain_margins_must_exceed_one_over_2048",
        "compact_steps_project_the_shared_certificate_without_reboxing_or_retry",
        "actual_errors_are_against_the_member_mean_matched_nominal_reference",
        "one_analytic_tail_unit_controls_original_form_not_only_scaled_form",
        "local_storage_barrier_proves_later_full_law_convergence_on_each_mean_leaf",
        "partial_validation_or_failed_margins_are_unavailable_not_dynamical_failure",
        "no_support_event_formation_autonomous_contact_or_physical_binding_claim",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-two-port-capture.v1", "report": _project(self)}


def assess_sine_two_port_capture(
    *,
    form_error_radius,
    phase_error_radius,
    reference_duration,
    time_step,
    order,
    max_steps,
):
    """Evaluate one predeclared reference transit and conditional actual capture.

    Nonnegative radii and positive duration/step use shared scalar admission.
    Require duration<=1024, step<=1/4, ordinary integer order 1..16 and
    max_steps 1..4096, with the declared step count fitting that budget.
    Compatibility uses the fixed 32/64 root budget. Failed comparison premises
    abstain before a reference trajectory; numerical failures retain their
    actual validated prefix, without smaller-step or longer-horizon retries.
    """
    raw = dict(
        form_error_radius=form_error_radius,
        phase_error_radius=phase_error_radius,
        reference_duration=reference_duration,
        time_step=time_step,
    )
    values = {key: exact_or_represented_real(value, key) for key, value in raw.items()}
    rx, rt, duration, step_size = tuple(values.values())
    if rx < 0 or rt < 0:
        raise ValueError("form_error_radius and phase_error_radius must be nonnegative")
    if not 0 < duration <= 1024 or not 0 < step_size <= Q(1, 4):
        raise ValueError("require 0<reference_duration<=1024 and 0<time_step<=1/4")
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError("order must be an ordinary integer from 1 through 16")
    if type(max_steps) is not int or not 1 <= max_steps <= 4096:
        raise ValueError("max_steps must be an ordinary integer from 1 through 4096")
    if duration / step_size > max_steps:
        raise ValueError("declared reference step count must not exceed max_steps")
    preparation = assess_sine_two_port_transit(
        form_error_radius=rx, phase_error_radius=rt
    )
    folded = _folded_reference(preparation)
    z0 = 7 * rx / 3069
    joint = 7 * rt + z0 + Q(24, 100000)
    scaled_form = (z0 + Q(1, 100000) * (Q(5, 12) + 2 * joint)) / (1 - Q(1, 50000))
    phase_error = joint + scaled_form
    comparison_margin = _REFERENCE_MARGIN - phase_error
    preparation_admitted = preparation.energy_budget_admitted and comparison_margin > 0
    reasons = []
    if not preparation.energy_budget_admitted:
        reasons.append("strict_initial_excess_storage_budget_not_certified")
    if comparison_margin <= 0:
        reasons.append("strict_full_horizon_comparison_margin_not_certified")
    target = None
    target_margin = None
    target_admitted = False
    if not reasons:
        target = assess_sine_two_port_compatibility(
            classes=(2, 1), outer_refinements=32, inner_refinements=64
        )
        if target.acute_margin_turns_bounds is not None:
            target_margin = (2 * pi_interval() * target.acute_margin_turns_bounds).lo
        target_admitted = (
            target.local_attraction_certified
            and target_margin is not None
            and target_margin > Q(1, 8)
        )
        if not target_admitted:
            reasons.append("required_implicit_target_geometry_not_certified")
    center, radius, elapsed, steps = (Q(0),) * 8, Q(0), Q(0), []
    failed = None
    minimum_margin = None
    if not reasons:
        flow, domain = _reference_field(folded)
        while elapsed < duration:
            delta = min(step_size, duration - elapsed)
            result, failed, reason = validated_metric_taylor_step(
                center,
                radius,
                folded.metric,
                delta,
                flow,
                domain,
                growth_rate=Q(0),
                order=order,
                time=elapsed,
                domain_failure="full_reference_tube_margin_not_above_one_over_2048",
            )
            if result is None:
                reasons.append(reason)
                break
            steps.append(_compact_step(result))
            center, radius = result.endpoint_center, result.endpoint_radius
            elapsed += delta
            margin = min(result.domain_lower_bounds) + _REFERENCE_MARGIN
            minimum_margin = (
                margin if minimum_margin is None else min(minimum_margin, margin)
            )
            if len(steps) % 128 == 0:
                _LOGGER.info(
                    "Validated two-port reference steps=%d slow_time=%s/%s",
                    len(steps),
                    elapsed,
                    duration,
                )
    validated = elapsed == duration and not reasons
    distance = None
    endpoint_certified = False
    if validated:
        pi = pi_interval()
        target_displacement = tuple(
            target.target_phase_bounds[i] - 2 * pi * preparation.nominal_phase_turns[i]
            for i in folded.representatives
        )
        distance = (
            sqrt(
                sum(
                    (
                        weight * (I(value) - bound) ** 2
                        for weight, value, bound in zip(
                            _FOLDED_WEIGHTS, center, target_displacement
                        )
                    ),
                    I(0),
                )
            ).hi
            + radius
        )
        endpoint_certified = distance <= Q(1, 2048)
        if not endpoint_certified:
            reasons.append("reference_endpoint_target_distance_not_certified")
    comparison_certified = (
        preparation_admitted and target_admitted and validated and endpoint_certified
    )
    phase_distance = form_norm = energy = storage_margin = None
    capture = False
    if comparison_certified:
        phase_distance = distance + phase_error
        endpoint_z = scaled_form / 2**512 + Q(1, 100000) * (
            Q(1, 1024) + 2 * phase_error
        )
        form_norm = 3216 * endpoint_z
        energy = phase_distance**2 + form_norm**2
        storage_margin = Q(1, 648000) - energy
        capture = phase_distance < Q(1, 12) and storage_margin > 0
        if not capture:
            reasons.append("strict_local_capture_storage_barrier_not_certified")
    return SineTwoPortCapture(
        preparation=preparation,
        target=target,
        reference_duration=duration,
        time_step=step_size,
        order=order,
        max_steps=max_steps,
        folded_reference=folded,
        initial_excess_storage_upper_bound=preparation.initial_excess_storage_upper_bound,
        full_slow_horizon=duration + 1,
        full_horizon_pi_squared_coefficient=Q(1023 * 1024) * (duration + 1),
        joint_error_candidate=joint,
        scaled_form_norm_candidate=scaled_form,
        phase_error_candidate=phase_error,
        comparison_margin=comparison_margin,
        target_acute_margin_lower_bound=target_margin,
        reference_steps=tuple(steps),
        validated_reference_duration=elapsed,
        reference_endpoint_center=center,
        reference_endpoint_radius=radius,
        reference_minimum_acute_margin=minimum_margin,
        reference_target_distance_upper_bound=distance,
        failed_tube=failed,
        endpoint_phase_distance_upper_bound=phase_distance,
        endpoint_relative_form_norm_upper_bound=form_norm,
        endpoint_excess_storage_upper_bound=energy,
        capture_storage_margin=storage_margin,
        preparation_admitted=preparation_admitted,
        target_admitted=target_admitted,
        reference_validated=validated,
        reference_endpoint_certified=endpoint_certified,
        full_comparison_certified=comparison_certified,
        capture_certified=capture,
        status="certified_capture" if capture else "unavailable",
        unavailable_reasons=tuple(reasons),
    )
