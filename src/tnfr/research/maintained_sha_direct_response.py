"""Direct marked radius and angle on the unchanged retained SHA cells.

The public boundary re-admits the complete retained evidence once. The new
observation evaluates only the marked difference rows, without a reference,
solver, derivative jets, phase wrapping or replacement trajectory. Its bounds
remain conditional on the retained producer's derivative/remainder evidence.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sin, sqrt
from ..physics._sine_admission import _sine_model_coefficients
from ..physics.relational_observations import _ordered
from ..physics.relational_sine_support import _capture_support_rows, _support_topology
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_principal_retention import HORIZON, STEP, STEP_COUNT
from .maintained_sha_retention_evidence import ShaRetentionEvidence, load_sha_retention


@dataclass(frozen=True)
class DirectResponseSource:
    """Fresh admitted primitives and exact marked rows in original node order.

    ``edges`` uses integer positions in ``nodes``. Sine coefficients multiply
    sin(theta_j-theta_i) in that edge orientation before division by pi.
    The phase coefficients multiply the complete form vector before the same
    divisor. Bare degrees normalize both rows; conductance enters phase only.
    """

    nodes: tuple
    edges: tuple[tuple[int, int], ...]
    conductances: tuple[Q, ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    beta: tuple[Q, ...]
    parent_index: int
    child_index: int
    reference_conductance: Q
    structural_time: Q
    form_rate_edge_coefficients: tuple[Q, ...]
    phase_rate_coefficients: tuple[Q, ...]
    mean_weights: tuple[Q, ...]
    form_mean: Q
    phase_lift_mean: Q


@dataclass(frozen=True)
class DirectResponseGate:
    """A declared radius or net-turn inequality with explicit availability.

    Radius requires a strictly positive lower bound. Net clockwise travel
    admits equality at 2*pi; an excluded net gate proves its enclosed travel
    is strictly below that target, not that all clockwise motion is absent.
    An unresolved enclosure or missing premise is unavailable, never a zero.
    """

    margin: I | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class DirectResponseCell:
    """Whole-cell marked observation with its original continuous phase lift."""

    index: int
    time: Q
    duration: Q
    r: I
    delta: I
    r_dot: I
    delta_dot: I
    radius_squared: I
    angular_numerator: I
    angular_rate: I | None
    angular_increment: I | None
    orientation: str
    form_mean: I
    phase_mean: I
    means_consistent: tuple[bool, bool]
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DirectResponseAggregate:
    """The fixed full-window certificate, never a shortened-prefix verdict.

    ``radius_squared_hull`` and its lower endpoint summarize observed cells
    even on a partial prefix. Only ``radius_gate`` can certify the full window.
    ``angular_increment`` is the integral enclosure of the continuous angle
    rate. Clockwise travel is its negative, not an integral of absolute speed.
    """

    full_window: bool
    accepted_cells: int
    accepted_duration: Q
    radius_squared_hull: I | None
    minimum_radius_squared: Q | None
    radius_gate: DirectResponseGate
    angular_increment: I | None
    clockwise_travel: I | None
    clockwise_turns: I | None
    net_turn_gate: DirectResponseGate
    all_clockwise: bool | None
    orientation_counts: tuple[tuple[str, int], ...]
    unavailable_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShaDirectResponseAudit:
    source: DirectResponseSource
    cells: tuple[DirectResponseCell, ...]
    aggregate: DirectResponseAggregate
    source_receipts: tuple[dict, ...]
    retained_receipts: tuple[dict, ...]
    retained_summary: dict
    persistent_source_unchanged: bool
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "one_fresh_retained_source_admission_original_held_complete_sine_law",
        "fixed_256_cell_target_original_half_unit_window_and_phase_lifts",
        "signed_marked_rows_and_direct_cross_product_before_angular_division",
        "strict_positive_radius_required_on_every_cell_for_continuous_angle",
        "net_clockwise_travel_is_negative_integrated_signed_angular_rate",
        "both_weighted_means_checked_without_projecting_or_resetting_cells",
        "old_derivative_jets_and_remainders_retain_frozen_producer_provenance",
        "no_new_flow_callback_solver_trajectory_event_or_pressure_refresh",
        "no_periodicity_formation_identity_or_matched_cut_discrimination_claim",
    )


def _capture_direct_source(graph, *, parent, child, reference_model):
    """Reuse shared graph/primitive admission without evaluating other reports."""
    if _sine_model_coefficients(reference_model) != (Q(0), Q(1), Q(1)):
        raise ValueError(
            "direct response requires the beta=1 unit-exchange zero-loss law"
        )
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if any("weight" not in data for _, _, data in graph.edges(data=True)):
        raise ValueError("direct response requires stored conductance on every edge")
    captured = _capture_support_rows(graph, reference_model)
    nodes = captured["nodes"]
    edges, neighbors = _support_topology(nodes, captured["edges"])
    weights, capacity = captured["conductances"], captured["capacity"]
    if any(value <= 0 for value in weights + capacity):
        raise ValueError(
            "direct response requires positive capacities and conductances"
        )
    positions = {node: i for i, node in enumerate(nodes)}
    if parent == child or parent not in positions or child not in positions:
        raise ValueError("distinct parent and child must belong to the source")
    a, c = positions[parent], positions[child]
    degrees = tuple(map(len, neighbors))
    if degrees[c] != 1 or a not in neighbors[c]:
        raise ValueError("the marked child must be pendant at the marked parent")
    if degrees[a] < 2:
        raise ValueError("the marked parent requires an external neighbor")
    beta = tuple(nu / degree for nu, degree in zip(capacity, degrees, strict=True))
    edge_coefficients, phase_coefficients = [], [Q(0)] * len(nodes)
    reference_conductance = None
    for (i, j), weight in zip(edges, weights, strict=True):
        # Difference of the two complete rows. Combine rational coefficients
        # before the interval sine or common pi divisor, including the bridge.
        signed = beta[c] * ((i == c) - (j == c)) - beta[a] * ((i == a) - (j == a))
        edge_coefficients.append(signed)
        phase_coefficients[i] += signed * weight
        phase_coefficients[j] -= signed * weight
        if {i, j} == {a, c}:
            reference_conductance = weight
    mean_weights = tuple(
        Q(degree) / nu for degree, nu in zip(degrees, capacity, strict=True)
    )
    weight_sum = sum(mean_weights, Q(0))

    def mean(values):
        return (
            sum(
                (
                    weight * value
                    for weight, value in zip(mean_weights, values, strict=True)
                ),
                Q(0),
            )
            / weight_sum
        )

    return DirectResponseSource(
        nodes,
        edges,
        weights,
        captured["epi"],
        captured["phase"],
        capacity,
        degrees,
        beta,
        a,
        c,
        reference_conductance,
        clock,
        tuple(edge_coefficients),
        tuple(phase_coefficients),
        mean_weights,
        mean(captured["epi"]),
        mean(captured["phase"]),
    )


def _cell_time(value, label, *, positive=False):
    value = exact_or_represented_real(value, label)
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return value


def _radius_gate(margin):
    """Admit a strict radius floor; a zero lower bound does not prove a zero."""
    if margin is None:
        return DirectResponseGate(None, "unavailable", "full_window_radius_unavailable")
    margin = I.coerce(margin)
    return DirectResponseGate(
        margin,
        "pass" if margin.lo > 0 else "unavailable",
        None if margin.lo > 0 else "strict_positive_angular_denominator_unavailable",
    )


def _net_turn_gate(margin):
    """Admit clockwise travel >=2*pi; exclude only with a strict upper gap."""
    if margin is None:
        return DirectResponseGate(
            None, "unavailable", "full_window_angular_increment_unavailable"
        )
    margin = I.coerce(margin)
    status = (
        "pass" if margin.lo >= 0 else "excluded" if margin.hi < 0 else "unavailable"
    )
    return DirectResponseGate(
        margin,
        status,
        "net_clockwise_turn_enclosure_unresolved" if status == "unavailable" else None,
    )


def _bound_cell(source, tube, *, index=0, time=Q(0), duration=STEP):
    """Observe the direct marked rows on one unchanged complete state box.

    There is one estimator: N=sqrt(w)*(delta*r_dot-r*delta_dot), divided by
    delta**2+w*r**2 only when its outward lower endpoint is strictly positive.
    A zero-containing denominator abstains without sampling or factorization.
    """
    if type(source) is not DirectResponseSource:
        raise TypeError("fresh direct-response primitives are required")
    if type(index) is not int or index < 0:
        raise ValueError("cell index must be a nonnegative ordinary integer")
    time = _cell_time(time, "cell time")
    duration = _cell_time(duration, "cell duration", positive=True)
    size = len(source.nodes)
    values = _ordered(tube, "complete retained state", limit=2 * size + 1)
    if len(values) != 2 * size:
        raise ValueError("state must retain every form and primitive phase coordinate")
    values = tuple(map(I.coerce, values))
    forms, phases = values[:size], values[size:]
    a, c = source.parent_index, source.child_index
    r, delta = forms[c] - forms[a], phases[c] - phases[a]
    divisor = pi_interval()
    r_dot = (
        sum(
            (
                coefficient * sin(phases[j] - phases[i])
                for (i, j), coefficient in zip(
                    source.edges, source.form_rate_edge_coefficients, strict=True
                )
                if coefficient
            ),
            I(0),
        )
        / divisor
    )
    delta_dot = (
        sum(
            (
                coefficient * value
                for coefficient, value in zip(
                    source.phase_rate_coefficients, forms, strict=True
                )
                if coefficient
            ),
            I(0),
        )
        / divisor
    )
    radius_squared = delta**2 + source.reference_conductance * r**2
    numerator = sqrt(I(source.reference_conductance)) * (delta * r_dot - r * delta_dot)
    weight_sum = sum(source.mean_weights, Q(0))

    def mean(coordinates):
        return (
            sum(
                (
                    weight * value
                    for weight, value in zip(
                        source.mean_weights, coordinates, strict=True
                    )
                ),
                I(0),
            )
            / weight_sum
        )

    form_mean, phase_mean = mean(forms), mean(phases)
    consistent = (
        form_mean.contains(source.form_mean),
        phase_mean.contains(source.phase_lift_mean),
    )
    reasons = []
    if not all(consistent):
        reasons.append("cell_does_not_enclose_both_conserved_means")
    rate = increment = None
    orientation = "unavailable"
    if radius_squared.lo <= 0:
        reasons.append("strict_positive_angular_denominator_unavailable")
    else:
        rate = numerator / radius_squared
        increment = duration * rate
        orientation = (
            "clockwise"
            if rate.hi < 0
            else (
                "counterclockwise"
                if rate.lo > 0
                else "stationary" if rate.lo == rate.hi == 0 else "unresolved"
            )
        )
    return DirectResponseCell(
        index,
        time,
        duration,
        r,
        delta,
        r_dot,
        delta_dot,
        radius_squared,
        numerator,
        rate,
        increment,
        orientation,
        form_mean,
        phase_mean,
        consistent,
        tuple(reasons),
    )


def _aggregate(cells, *, full_window):
    """Accumulate all declared cells; never promote a partial prefix to a turn."""
    if type(full_window) is not bool:
        raise TypeError("full_window must be Boolean")
    cells = tuple(cells)
    if len(cells) > STEP_COUNT:
        raise ValueError("direct response exceeds the fixed retained window")
    for index, cell in enumerate(cells):
        if type(cell) is not DirectResponseCell:
            raise TypeError("direct cell observations are required")
        if cell.index != index or cell.time != index * STEP or cell.duration != STEP:
            raise ValueError("direct cells must retain the original chronological grid")
    duration = sum((cell.duration for cell in cells), Q(0))
    full_window = full_window and len(cells) == STEP_COUNT and duration == HORIZON
    hull = (
        I(
            min(cell.radius_squared.lo for cell in cells),
            max(cell.radius_squared.hi for cell in cells),
        )
        if cells
        else None
    )
    minimum = hull.lo if hull is not None else None
    counts = tuple(sorted(Counter(cell.orientation for cell in cells).items()))
    reasons = []
    if not full_window:
        reasons.append("fixed_256_cell_half_unit_window_unavailable")
    if any(not all(cell.means_consistent) for cell in cells):
        reasons.append("one_or_more_cells_do_not_enclose_both_conserved_means")
    radius_gate = (
        DirectResponseGate(None, "unavailable", reasons[0])
        if reasons
        else _radius_gate(I(minimum))
    )
    if not reasons and radius_gate.status != "pass":
        reasons.append("strict_positive_angular_denominator_unavailable")
    if any(
        cell.angular_increment is None or cell.unavailable_reasons for cell in cells
    ):
        reasons.append("one_or_more_angular_cells_unavailable")
    increment = clockwise = turns = None
    all_clockwise = None
    if reasons:
        turn_gate = DirectResponseGate(None, "unavailable", reasons[0])
    else:
        increment = sum((cell.angular_increment for cell in cells), I(0))
        clockwise = -increment
        turns = clockwise / (2 * pi_interval())
        margin = clockwise - 2 * pi_interval()
        turn_gate = _net_turn_gate(margin)
        all_clockwise = all(cell.orientation == "clockwise" for cell in cells)
        if turn_gate.status == "unavailable":
            reasons.append(turn_gate.reason)
    return DirectResponseAggregate(
        full_window,
        len(cells),
        duration,
        hull,
        minimum,
        radius_gate,
        increment,
        clockwise,
        turns,
        turn_gate,
        all_clockwise,
        counts,
        tuple(reasons),
    )


def _audit_from_fresh_retention(evidence):
    """Internal only: the public boundary reloads the retained primitive files."""
    if type(evidence) is not ShaRetentionEvidence:
        raise TypeError("fresh shared retained SHA evidence required")
    graph = evidence.admitted.graph
    signature = _signature(graph)
    source = _capture_direct_source(
        graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    expected_initial = tuple(map(I, source.epi + source.phase))
    prediction = evidence.prediction
    edge_weights = {
        tuple(sorted(edge)): weight
        for edge, weight in zip(source.edges, source.conductances, strict=True)
    }
    predicted_weights = dict(
        zip(prediction.edges, prediction.conductances, strict=True)
    )
    if (
        evidence.initial != expected_initial
        or prediction.nodes != source.nodes
        or prediction.epi != source.epi
        or prediction.phases != source.phase
        or prediction.capacities != source.capacity
        or predicted_weights != edge_weights
        or source.structural_time != TIMES[-1]
        or prediction.horizon != HORIZON
    ):
        raise ValueError(
            "retained primitive association, clock or law differs from fresh source"
        )
    cells, carried = [], expected_initial
    if len(evidence.steps) > STEP_COUNT:
        raise ValueError("retained evidence exceeds the fixed half-unit window")
    for index, step in enumerate(evidence.steps):
        if (
            step.index != index
            or step.time != index * STEP
            or step.duration != STEP
            or step.initial != carried
        ):
            raise ValueError(
                "retained chronological endpoint carry changed after admission"
            )
        cells.append(
            _bound_cell(
                source, step.tube, index=index, time=step.time, duration=step.duration
            )
        )
        carried = step.endpoint
    aggregate = _aggregate(cells, full_window=len(cells) == STEP_COUNT)
    if _signature(graph) != signature:
        raise ValueError("direct retained observation changed persistent source state")
    return ShaDirectResponseAudit(
        source,
        tuple(cells),
        aggregate,
        deepcopy(evidence.admitted.receipts),
        deepcopy(evidence.receipts),
        deepcopy(evidence.summary),
        True,
    )


def audit_sha_direct_response(evidence_dir, *, repository_root):
    """Re-admit once, then observe the fixed retained radius and signed angle.

    This retrospective calculation requires its own frozen protocol and
    instrument. Cached caller reports never replace primitive admission.
    It invokes no new field callback, solver, event, trajectory or phase reset.
    """
    return _audit_from_fresh_retention(
        load_sha_retention(evidence_dir, repository_root=repository_root)
    )
