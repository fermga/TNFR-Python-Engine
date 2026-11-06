"""Conditional fixed-metric bounds for uncertain positive held capacities.

Each member keeps its own capacities constant. The nominal component, clock,
support and full sine law are shared; supplied nominal form/phase boxes must
enclose complete solution cells. This detached arithmetic does not establish
that association, authenticate a source, execute a trajectory or select events.

The existing nominal metric and scalar comparison are reused without changing
their same-capacity implementation. Capacity mismatch supplies both an affine
forcing and drift of the nominal weighted means. All primitive batches are
admitted before iteration. Fixed limits and rational-work checks are those of
``sine_preparation_metric``; they do not bound internal kernel temporaries or
replace an external time/memory budget for a measured application.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._comparison_flow import scalar_affine_upper
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from . import sine_preparation_metric as metric

__all__ = (
    "CapacityMetricStep",
    "CapacityMetricRun",
    "bound_capacity_metric_cells",
)


def _add(left, right, budget):
    return budget.check((left[0] + right[0], left[1] + right[1]))


def _subtract(left, right, budget):
    return budget.check((left[0] - right[1], left[1] - right[0]))


def _multiply(left, right, budget):
    corners = budget.check(tuple(a * b for a in left for b in right))
    return min(corners), max(corners)


def _magnitude(pair):
    return max(map(abs, pair))


def _widen(pair, radius, budget):
    return budget.check((pair[0] - radius, pair[1] + radius))


def _mean_and_portrait(family, radius, form_drift, phase_drift, budget):
    means = (
        _widen(family.form_mean_error, form_drift, budget),
        _widen(family.phase_mean_error, phase_drift, budget),
    )
    if family.child is None:
        offsets = (
            _widen(family.form_offset, form_drift, budget),
            _widen(family.phase_offset, phase_drift, budget),
        )
    else:
        offsets = (family.form_offset, family.phase_offset)
    if not form_drift and not phase_drift:
        portrait = metric._portrait_upper(family, radius, budget)
    else:
        square = budget.check(
            _magnitude(offsets[1]) ** 2
            + family.reference_weight * _magnitude(offsets[0]) ** 2
        )
        portrait = budget.check(
            metric._sqrt_upper(square, budget) + family.observation_root_upper * radius
        )
    return means, offsets, portrait


@dataclass(frozen=True)
class CapacityMetricStep:
    """One attempted cell, retaining both law-mismatch rows and mean drift.

    Forcing intervals are the signed nodal forcing before division by pi.
    ``forcing_upper`` includes that division. Mean drift fields accumulate
    symmetric bounds from time zero; the corresponding rates bound this cell.
    """

    index: int
    time: Q
    duration: Q
    inflated_edges: tuple
    coefficient: metric.MetricCoefficientBound
    gamma_upper: Q
    nominal_sine_edges: tuple
    nominal_form_edges: tuple
    nominal_sine_nodes: tuple
    nominal_form_gradient: tuple
    forcing_form_intervals: tuple
    forcing_phase_intervals: tuple
    forcing_norm_squared: Q
    forcing_upper: Q
    normalized_radius_before: Q
    normalized_radius_after: Q
    radius_upper: Q
    form_mean_rate_upper: Q
    phase_mean_rate_upper: Q
    form_mean_drift_upper: Q
    phase_mean_drift_upper: Q
    form_mean_error: tuple
    phase_mean_error: tuple
    form_offset: tuple
    phase_offset: tuple
    portrait_error_upper: Q
    edge_margins: tuple
    domain_passed: bool
    maximum_bits: int


@dataclass(frozen=True)
class CapacityMetricRun:
    """Conditional completion or an explicit unresolved prefix.

    A completed run closes every sufficient first-exit gate conditional on
    the nominal whole-cell and complete-law premises. An unavailable endpoint
    is only comparison arithmetic, not a certified whole-window state bound.
    No response-angle ordering follows without its independent nominal gates.
    """

    status: str
    reason: str | None
    steps: tuple
    capacity_intervals: tuple
    mobility_errors: tuple
    relative_capacity_error: Q
    initial_edge_margins: tuple
    final_normalized_radius: Q
    final_radius_upper: Q
    final_form_mean_drift_upper: Q
    final_phase_mean_drift_upper: Q
    final_form_mean_error: tuple
    final_phase_mean_error: tuple
    final_form_offset: tuple
    final_phase_offset: tuple
    final_portrait_error_upper: Q
    maximum_bits: int
    attempted_cells: int
    completed_cells: int
    failed_coefficient: metric.MetricCoefficientBound | None
    failed_gamma_upper: Q | None


class _ScalarWorkError(ValueError):
    def __init__(self, coefficient, gamma, maximum_bits):
        super().__init__("the fixed scalar cell exceeds gamma*duration <= 1")
        self.coefficient = coefficient
        self.gamma = gamma
        self.maximum_bits = maximum_bits


def _nominal_rows(context, phase, form, budget):
    sine_edges, form_edges = [], []
    sine_nodes = [(Q(0), Q(0)) for _ in phase]
    gradients = [(Q(0), Q(0)) for _ in form]
    for (i, j), weight in zip(context.edges, context.conductances):
        phase_edge = _subtract(phase[i], phase[j], budget)
        sine = sin(I(*phase_edge))
        sine_pair = budget.check((sine.lo, sine.hi))
        form_pair = _subtract(form[i], form[j], budget)
        gradient = budget.check((weight * form_pair[0], weight * form_pair[1]))
        sine_edges.append(sine_pair)
        form_edges.append(form_pair)
        sine_nodes[i] = _add(sine_nodes[i], sine_pair, budget)
        sine_nodes[j] = _subtract(sine_nodes[j], sine_pair, budget)
        gradients[i] = _add(gradients[i], gradient, budget)
        gradients[j] = _subtract(gradients[j], gradient, budget)
    return tuple(sine_edges), tuple(form_edges), tuple(sine_nodes), tuple(gradients)


def _forcing(context, mobility, sine_nodes, gradients, budget):
    form, phase = [], []
    for delta, sine, gradient in zip(mobility, sine_nodes, gradients):
        pair = _multiply(delta, sine, budget)
        form.append((-pair[1], -pair[0]))
        phase.append(_multiply(delta, gradient, budget))
    square = Q(0)
    for (i, j), weight in zip(context.edges, context.conductances):
        square = budget.check(
            square
            + weight * _magnitude(_subtract(form[i], form[j], budget)) ** 2
            + _magnitude(_subtract(phase[i], phase[j], budget)) ** 2
        )
    upper = budget.check(metric._sqrt_upper(square, budget) / pi_interval().lo)
    return tuple(form), tuple(phase), square, upper


def _bound_cell(
    family,
    phase,
    form,
    allowances,
    mobility,
    eta,
    kappas,
    dual_roots,
    *,
    index,
    time,
    duration,
    normalized_radius,
    form_drift,
    phase_drift,
    maximum_bits,
):
    budget = metric._Budget(maximum_bits)
    context = family.context
    inflated = budget.check(
        tuple(
            (phase[i][0] - phase[j][1] - h, phase[i][1] - phase[j][0] + h)
            for (i, j), h in zip(context.edges, allowances)
        )
    )
    secants = []
    for pair in inflated:
        result = cos(I(*pair))
        secants.append(budget.check((result.lo, result.hi)))
    coefficient = metric._coefficient_bound(context, tuple(secants), budget)
    gamma = budget.check(
        (coefficient.norm_upper + eta * coefficient.degree_upper)
        / (2 * pi_interval().lo)
    )
    if budget.check(gamma * duration) > 1:
        raise _ScalarWorkError(coefficient, gamma, budget.maximum)
    sine_edges, form_edges, sine_nodes, gradients = _nominal_rows(
        context, phase, form, budget
    )
    forcing_form, forcing_phase, square, forcing = _forcing(
        context, mobility, sine_nodes, gradients, budget
    )
    normalized_forcing = budget.check(forcing / family.radius_scale)
    after = budget.check(
        scalar_affine_upper(normalized_radius, gamma, duration, normalized_forcing)
    )
    if after < normalized_radius:
        raise ArithmeticError("a nonnegative affine bound reduced the carried radius")
    radius = budget.check(family.radius_scale * after)
    form_rate, phase_rate = Q(0), Q(0)
    for kappa, weight, sine, edge, phase_root, form_root in zip(
        kappas, context.conductances, sine_edges, form_edges, *dual_roots
    ):
        form_rate = budget.check(
            form_rate + kappa * (_magnitude(sine) + phase_root * radius)
        )
        phase_rate = budget.check(
            phase_rate + kappa * weight * (_magnitude(edge) + form_root * radius)
        )
    form_rate = budget.check(form_rate / pi_interval().lo)
    phase_rate = budget.check(phase_rate / pi_interval().lo)
    form_drift = budget.check(form_drift + duration * form_rate)
    phase_drift = budget.check(phase_drift + duration * phase_rate)
    means, offsets, portrait = _mean_and_portrait(
        family, radius, form_drift, phase_drift, budget
    )
    margins = metric._edge_margins(context, allowances, radius, budget)
    return CapacityMetricStep(
        index,
        time,
        duration,
        inflated,
        coefficient,
        gamma,
        sine_edges,
        form_edges,
        sine_nodes,
        gradients,
        forcing_form,
        forcing_phase,
        square,
        forcing,
        normalized_radius,
        after,
        radius,
        form_rate,
        phase_rate,
        form_drift,
        phase_drift,
        means[0],
        means[1],
        offsets[0],
        offsets[1],
        portrait,
        margins,
        all(value > 0 for value in margins),
        budget.maximum,
    )


def bound_capacity_metric_cells(
    family, *, capacity_intervals, phase_cells, form_cells, edge_allowances, durations
):
    """Bound one nonadaptive sequence under a compact held-capacity box.

    Capacity pairs are strictly positive and include each nominal capacity.
    Phase and form cells are complete node-ordered whole-cell endpoint boxes,
    with exact integer/Fraction endpoints. They must refer to the same nominal
    full-sine solution and clock. Reconstructing its sine and form gradients
    here does not prove this external association. Stored pressure, sampled
    endpoints, tangents and cached reports cannot substitute for these boxes.

    The supplied live MetricFamily retains the initial state-error box and
    nominal weighted means. Its Cartesian product with the capacity box can
    over-enclose correlated preparations; no independence claim is required.
    The external child, when present, keeps its independently supplied constant
    law and both initial errors. All limits and input admission complete before
    a cell is attempted. No refinement or retry occurs after a failed gate.
    """
    family = metric._require_family(family)
    context = family.context
    budget = metric._Budget(family.maximum_bits)
    size = len(context.capacities)
    capacities = metric._pairs(capacity_intervals, size, "capacity_intervals", budget)
    if any(
        not 0 < lo <= nominal <= hi
        for (lo, hi), nominal in zip(capacities, context.capacities)
    ):
        raise ValueError("positive capacity intervals must contain nominal capacities")
    mobility = budget.check(
        tuple(
            ((lo - nominal) / degree, (hi - nominal) / degree)
            for (lo, hi), nominal, degree in zip(
                capacities, context.capacities, context.degrees
            )
        )
    )
    eta = budget.check(
        max(
            max(nominal - lo, hi - nominal) / nominal
            for (lo, hi), nominal in zip(capacities, context.capacities)
        )
    )
    raw_phases = metric._items(phase_cells, "phase_cells", metric.MAX_METRIC_CELLS)
    raw_forms = metric._items(form_cells, "form_cells", metric.MAX_METRIC_CELLS)
    if not raw_phases or len(raw_phases) != len(raw_forms):
        raise ValueError("matching nonempty phase and form cell batches are required")
    phases = tuple(
        metric._pairs(cell, size, "phase cell", budget) for cell in raw_phases
    )
    forms = tuple(metric._pairs(cell, size, "form cell", budget) for cell in raw_forms)
    raw_durations = metric._items(durations, "durations", metric.MAX_METRIC_CELLS)
    if len(raw_durations) != len(phases):
        raise ValueError("durations must match the complete cell sequence")
    durations = tuple(
        metric._input(value, "duration", budget) for value in raw_durations
    )
    if min(durations) <= 0:
        raise ValueError("cell durations must be positive")
    times, end = [], Q(0)
    for duration in durations:
        times.append(end)
        end = budget.check(end + duration)
    raw_allowances = metric._items(
        edge_allowances, "edge_allowances", len(context.edges)
    )
    if len(raw_allowances) != len(context.edges):
        raise ValueError("allowances must match complete bare support")
    allowances = tuple(
        metric._input(value, "edge allowance", budget) for value in raw_allowances
    )
    if min(allowances) <= 0:
        raise ValueError("edge allowances must be positive")
    weighted_mobility = tuple(
        budget.check((alpha * pair[0], alpha * pair[1]))
        for alpha, pair in zip(context.alpha, mobility)
    )
    kappas = tuple(
        _magnitude(_subtract(weighted_mobility[i], weighted_mobility[j], budget))
        for i, j in context.edges
    )
    dual_roots = tuple(
        tuple(metric._sqrt_upper(value, budget) for value in duals)
        for duals in (context.edge_phase_duals, context.edge_form_duals)
    )
    normalized, radius = family.initial_normalized_radius, family.initial_radius_upper
    form_drift, phase_drift = Q(0), Q(0)
    means, offsets, portrait = _mean_and_portrait(
        family, radius, form_drift, phase_drift, budget
    )
    initial = metric._edge_margins(context, allowances, radius, budget)
    reason = None if all(value > 0 for value in initial) else "initial_domain"
    steps, attempted = [], 0
    failed_coefficient, failed_gamma = None, None
    if reason is None:
        for index, (phase, form, duration, time) in enumerate(
            zip(phases, forms, durations, times)
        ):
            attempted += 1
            try:
                step = _bound_cell(
                    family,
                    phase,
                    form,
                    allowances,
                    mobility,
                    eta,
                    kappas,
                    dual_roots,
                    index=index,
                    time=time,
                    duration=duration,
                    normalized_radius=normalized,
                    form_drift=form_drift,
                    phase_drift=phase_drift,
                    maximum_bits=budget.maximum,
                )
            except _ScalarWorkError as exc:
                budget.maximum = max(budget.maximum, exc.maximum_bits)
                failed_coefficient, failed_gamma = exc.coefficient, exc.gamma
                reason = "scalar_work"
                break
            except metric.MetricBudgetError as exc:
                budget.maximum = max(budget.maximum, exc.maximum_bits)
                reason = "intermediate_budget"
                break
            steps.append(step)
            budget.maximum = max(budget.maximum, step.maximum_bits)
            normalized, radius = step.normalized_radius_after, step.radius_upper
            form_drift, phase_drift = (
                step.form_mean_drift_upper,
                step.phase_mean_drift_upper,
            )
            means = (step.form_mean_error, step.phase_mean_error)
            offsets = (step.form_offset, step.phase_offset)
            portrait = step.portrait_error_upper
            if not step.domain_passed:
                reason = "first_exit"
                break
    return CapacityMetricRun(
        "completed" if reason is None else "unavailable",
        reason,
        tuple(steps),
        capacities,
        mobility,
        eta,
        initial,
        normalized,
        radius,
        form_drift,
        phase_drift,
        means[0],
        means[1],
        offsets[0],
        offsets[1],
        portrait,
        budget.maximum,
        attempted,
        sum(step.domain_passed for step in steps),
        failed_coefficient,
        failed_gamma,
    )
