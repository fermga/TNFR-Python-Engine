"""Detached regular-domain certificates and instantaneous resultant kinematics.

The storage theorem concerns the ideal continuous held-law model initialized
at represented coordinates. Kinematic bounds instead use the captured native
phase rates as a supplied direction; they do not enclose ideal ODE rates.
Neither observation advances a state or permits execution at a singularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from ..mathematics._phase_resultant_chamber import (
    certified_cosine_bounds,
    relative_resultant_rate_bounds,
)
from ..mathematics._rational_interval import pi_interval

__all__ = ("RelationalRegularityObservation", "observe_relational_regularity")


@dataclass(frozen=True)
class RelationalRegularityObservation:
    field: RelationalExchangeField
    storage_bounds: tuple[Q, Q]
    regularity_storage_threshold: Q
    regularity_certified: bool
    phase_metric_lower_bounds: tuple[Q, ...] | None
    resultant_cut_distance_lower_bound: Q | None
    resultant_rate_bounds: tuple[tuple[tuple[Q, Q], tuple[Q, Q]], ...]
    resultant_speed_upper_bounds: tuple[Q, ...]
    status: str
    scope: tuple[str, ...] = (
        "ideal_continuous_regular_reference_law_from_represented_initial_state",
        "fixed_simple_connected_unit_support_held_nonnegative_capacity_no_input_or_events",
        "storage_enclosed_with_exact_represented_forms_and_mathematical_trigonometry",
        "sufficient_global_regular_continuation_not_convergence_or_Euler_stability",
        "instantaneous_kinematics_use_captured_rates_not_certified_ideal_ODE_rates",
        "instantaneous_speed_is_not_a_whole_time_rate_bound",
        "unresolved_storage_test_does_not_establish_boundary_access",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.relational-regularity.v1", "report": _project(self)}


def observe_relational_regularity(graph, *, model):
    """Capture one admitted regular field and apply a sufficient storage bound.

    E < beta*max(2, minimum_degree) keeps every resultant separated from the
    excluded ray for the ideal continuous law. All positive margins use an
    outward upper storage bound; absence of a margin is inconclusive. The
    explicit regular model prevents confusing this theorem with remaining in
    the narrower acute or positive-resultant execution chambers.
    """
    if (
        not isinstance(model, RelationalExchangeModel)
        or model.phase_domain != "regular"
    ):
        raise ValueError("an explicit regular relational model is required")
    field = evaluate_relational_exchange(graph, model=model)
    positions = {node: i for i, node in enumerate(field.nodes)}
    rows = [[] for _ in field.nodes]
    phase = tuple(map(Q, field.phase))
    phase_lower = phase_upper = Q(0)
    for left, right in field.edges:
        i, j = positions[left], positions[right]
        rows[i].append(j)
        rows[j].append(i)
        lo, hi = certified_cosine_bounds(phase[j] - phase[i])
        phase_lower += 1 - hi
        phase_upper += 1 - lo
    neighbors = tuple(tuple(row) for row in rows)
    degrees = tuple(map(len, neighbors))
    beta = Q(model.storage_scale)
    storage = (
        field.form_storage + beta * phase_lower,
        field.form_storage + beta * phase_upper,
    )
    minimum_degree = min(degrees)
    threshold = beta * max(2, minimum_degree)
    admitted = storage[1] < threshold
    metric_bounds = margin = None
    if admitted:
        v = storage[1] / beta
        pi_lower = pi_interval().lo
        metric_bounds = tuple(
            pi_lower * ((2 - v) / 2 if degree == 1 else degree - v)
            for degree in degrees
        )
        margin = min(Q(1), 2 - v) if minimum_degree == 1 else minimum_degree - v
    rates = tuple(map(Q, field.phase_rate))
    drifts = relative_resultant_rate_bounds(phase, neighbors, rates)
    speeds = tuple(
        sum((abs(rates[j] - rates[i]) for j in row), Q(0))
        for i, row in enumerate(neighbors)
    )
    return RelationalRegularityObservation(
        field,
        storage,
        threshold,
        admitted,
        metric_bounds,
        margin,
        drifts,
        speeds,
        (
            "global_regular_continuation_certified"
            if admitted
            else "storage_test_unresolved"
        ),
    )
