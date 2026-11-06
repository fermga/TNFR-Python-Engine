"""One fixed, validated conservative doubled-C5 preparation protocol.

This research runner retains all twenty fine coordinates and calls the
unchanged normalized-sine field. Its source, numerical policy and sufficient
residence detector are fixed prospectively. Importing this module evaluates
neither the source bounds nor a trajectory. No graph recapture is performed.
"""

from dataclasses import asdict, dataclass
from fractions import Fraction as Q
from pathlib import Path
from time import monotonic

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_BITS, I, pi_interval, sin
from ..mathematics._validated_taylor import validated_box_taylor_step
from ..physics._sine_flow import _held_sine_neighbor_field
from ..physics.relational_observations import _ordered
from .artifact_io import write_json_once

HORIZON, STEP, ORDER, CELL_COUNT = Q(6), Q(1, 8), 8, 48
MARGIN, RESIDENCE_CELLS = Q(1, 100), 4
ERROR_RADIUS, WALL_SECONDS, STATUS_SECONDS = Q(1, 2**20), 900, 300
COLLECTIVE_SCALE, INTERNAL_SCALE = Q(570, 287), Q(17, 287)
PROFILE = (-1, -1, 0, 1, 1)
INTERNAL_FORMS = (Q(1), Q(5, 4), Q(3, 2), Q(7, 4), Q(2))
INTERNAL_PHASES = (Q(1, 100), Q(3, 250), Q(7, 500), Q(2, 125), Q(1, 50))
PAIRS = tuple((2 * index, 2 * index + 1) for index in range(5))
EDGES = tuple(
    sorted(
        tuple(sorted((left, right)))
        for index in range(5)
        for left in PAIRS[index]
        for right in PAIRS[(index + 1) % 5]
    )
)


def _exact(value, label):
    if type(value) is not int and not isinstance(value, Q):
        raise TypeError(f"{label} requires an exact integer or Fraction")
    return exact_or_represented_real(value, label)


def _box(values):
    """Normalize every coordinate before interval arithmetic or pi work."""
    rows = _ordered(values, "full state box", limit=21)
    if len(rows) != 20:
        raise ValueError("full state requires ten forms followed by ten phase lifts")
    endpoints = tuple(
        (
            (
                _exact(value.lo, "interval lower"),
                _exact(value.hi, "interval upper"),
            )
            if isinstance(value, I)
            else (_exact(value, "coordinate"),) * 2
        )
        for value in rows
    )
    return tuple(I(lower, upper) for lower, upper in endpoints)


def _source_center(*, redistributed):
    forms = tuple(
        (COLLECTIVE_SCALE * PROFILE[index] if redistributed else Q(0))
        + sign * (INTERNAL_SCALE if redistributed else 1) * INTERNAL_FORMS[index]
        for index in range(5)
        for sign in (1, -1)
    )
    phases = tuple(sign * value for value in INTERNAL_PHASES for sign in (1, -1))
    return forms + phases


def _energy_upper(center, radius):
    """Fine-edge quadratic upper bound, including independent fine errors."""
    return sum(
        (
            (abs(center[left] - center[right]) + 2 * radius) ** 2 / 2
            + (abs(center[10 + left] - center[10 + right]) + 2 * radius) ** 2 / 2
            for left, right in EDGES
        ),
        Q(0),
    )


@dataclass(frozen=True)
class ReplicaPreparationSource:
    nominal: tuple[Q, ...]
    original_nominal: tuple[Q, ...]
    initial_box: tuple[I, ...]
    original_box: tuple[I, ...]
    intended_coordinate_error: Q
    numerical_enclosure_allowance: Q
    proof_coordinate_error: Q
    nominal_form_storage: Q
    original_form_storage: Q
    storage_upper_bound: Q
    original_storage_upper_bound: Q
    internal_radius_squared_initial_upper: Q
    internal_radius_initial_lower: Q
    internal_growth_upper: Q
    internal_radius_squared_future_upper: Q
    internal_radius_future_lower: Q
    raw_gap_upper_bound: Q
    early_raw_gap_upper_bound: Q
    original_control_mean_norm_squared_upper: Q
    target_activity_lower_bound: Q
    nominal_storage_equality_certified: bool
    enclosure_admitted: bool
    global_chart_and_nontip_certified: bool
    early_exclusion_certified: bool
    original_control_exclusion_certified: bool
    certified: bool
    clock: str = "original structural t; both full rows retain 1/pi"
    memberwise_equal_storage_claimed: bool = False
    empirical_validation: bool = False


def declared_source():
    """Rebuild the fixed source and its analytical bounds, without a flow call.

    Energy equality concerns the two nominal centers. The independent error
    families instead have separate energy bounds. Outward interval rounding
    is retained as numerical enclosure slack, not added physical uncertainty.
    Bounds are also proved for that slightly wider computational cover.
    """
    center, original = _source_center(redistributed=True), _source_center(
        redistributed=False
    )
    slack = Q(1, 2**INTERVAL_BITS)
    proof_error = ERROR_RADIUS + slack
    initial = tuple(I(value - ERROR_RADIUS, value + ERROR_RADIUS) for value in center)
    original_box = tuple(
        I(value - ERROR_RADIUS, value + ERROR_RADIUS) for value in original
    )
    enclosures = all(
        bound.lo <= value - ERROR_RADIUS <= value + ERROR_RADIUS <= bound.hi
        and value - proof_error <= bound.lo <= bound.hi <= value + proof_error
        for row, boxes in ((center, initial), (original, original_box))
        for value, bound in zip(row, boxes)
    )
    form_storage = sum(((center[a] - center[b]) ** 2 / 2 for a, b in EDGES), Q(0))
    original_form = sum(((original[a] - original[b]) ** 2 / 2 for a, b in EDGES), Q(0))
    same_energy = (
        form_storage == original_form == Q(95, 2)
        and center[10:] == original[10:]
        and 12 * COLLECTIVE_SCALE**2 + Q(95, 2) * INTERNAL_SCALE**2 == Q(95, 2)
    )
    energy, original_energy = _energy_upper(center, proof_error), _energy_upper(
        original, proof_error
    )
    pi = pi_interval()
    # e < sum_{j=0}^4(1/j!) + (1/5!)/(1-1/6) < 11/4.
    # pi>3 then gives exp(6/pi)<8 and exp(12/pi)<64, without
    # numerical exponential evaluation or a trajectory-dependent rate.
    exponential_majorant = Q(65, 24) + Q(1, 100)
    growth_admitted = pi.lo > 3 and exponential_majorant < Q(11, 4)
    growth_admitted = growth_admitted and Q(11, 4) ** 2 < 8 and Q(11, 4) ** 4 < 64
    initial_radius_squared = (2 * INTERNAL_SCALE + proof_error) ** 2 + (
        Q(1, 50) + proof_error
    ) ** 2
    initial_radius_lower = INTERNAL_SCALE - proof_error
    future_radius_squared, future_radius_lower = (
        64 * initial_radius_squared,
        initial_radius_lower / 8,
    )
    chart = (
        enclosures
        and growth_admitted
        and energy < 48
        and future_radius_squared < 1
        and future_radius_lower > Q(1, 160)
    )
    raw_gap = 2 * proof_error + 36 / pi.lo
    early_raw_gap = 2 * proof_error + 12 / pi.lo
    early_exclusion = chart and early_raw_gap < 3 * pi.lo / 2 + MARGIN
    original_norm_squared = 64**2 * 10 * proof_error**2
    original_exclusion = (
        enclosures
        and growth_admitted
        and original_energy < 48
        and original_norm_squared < 1
        and pi.lo**2 > 8
    )
    activity = (5 * sin(I(MARGIN)) / (1920 * pi)).lo
    certified = (
        same_energy
        and chart
        and raw_gap < 13
        and early_exclusion
        and original_exclusion
        and activity > 0
    )
    return ReplicaPreparationSource(
        center,
        original,
        initial,
        original_box,
        ERROR_RADIUS,
        slack,
        proof_error,
        form_storage,
        original_form,
        energy,
        original_energy,
        initial_radius_squared,
        initial_radius_lower,
        Q(8),
        future_radius_squared,
        future_radius_lower,
        raw_gap,
        early_raw_gap,
        original_norm_squared,
        activity,
        same_energy,
        enclosures,
        chart,
        early_exclusion,
        original_exclusion,
        certified,
    )


def complete_field():
    """Compile the unchanged twenty-row conservative law in its original clock."""
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    rows = tuple(
        tuple(
            (right if left == node else left, 1, Q(1))
            for left, right in EDGES
            if node in (left, right)
        )
        for node in range(10)
    )
    return _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, rows)


def _smooth_domain(_tube):
    # The complete sine field is smooth globally in these real coordinates.
    # Target admission must not become an artificial integration boundary.
    return (Q(1),)


@dataclass(frozen=True)
class ReplicaTubeObservation:
    phase_means: tuple[I, ...]
    phase_half_differences: tuple[I, ...]
    raw_gaps: tuple[I, ...]
    selected_shifts: tuple[int | None, ...]
    shifted_gaps: tuple[I | None, ...]
    mean_geometry_certified: bool
    box_chart_certified: bool
    box_physical_geometry_certified: bool
    raw_gap_range_certified: bool
    source_membership_inferred: bool = False
    activity_inferred_for_arbitrary_box: bool = False


def observe_tube(tube):
    """Observe all twenty supplied intervals without assuming source membership.

    Midpoint geometry alone is not physical pair-phase geometry outside an
    admitted pair chart. The runner may use its separately proved source
    chart for its own validated tube; an arbitrary caller cannot inherit it.
    The fixed branch range is a sufficient detector, not a general unwrap.
    """
    state = _box(tube)
    phases = state[10:]
    means = tuple((phases[a] + phases[b]) / 2 for a, b in PAIRS)
    deltas = tuple((phases[a] - phases[b]) / 2 for a, b in PAIRS)
    gaps = tuple(means[(index + 1) % 5] - means[index] for index in range(5))
    pi = pi_interval()
    selected, shifted = [], []
    for gap in gaps:
        candidates = tuple(
            (index, gap + 2 * index * pi)
            for index in range(-2, 3)
            if (gap + 2 * index * pi).lo >= MARGIN
            and (gap + 2 * index * pi).hi <= pi.lo / 2 - MARGIN
        )
        if len(candidates) > 1:
            raise ArithmeticError(
                "disjoint phase branches were simultaneously admitted"
            )
        index, interval = candidates[0] if candidates else (None, None)
        selected.append(index)
        shifted.append(interval)
    admitted = all(index is not None for index in selected)
    winding_one = admitted and sum(selected) == 1
    chart = all(value.abs_max < 1 for value in deltas)
    return ReplicaTubeObservation(
        means,
        deltas,
        gaps,
        tuple(selected),
        tuple(shifted),
        winding_one,
        chart,
        winding_one and chart,
        all(value.abs_max < 13 for value in gaps),
    )


def residence_windows(flags):
    """Bookkeep a contiguous validated prefix; Boolean flags are not evidence."""
    flags = _ordered(flags, "cell flags", limit=CELL_COUNT + 1)
    if len(flags) > CELL_COUNT or any(type(value) is not bool for value in flags):
        raise ValueError("require at most48 ordinary Boolean contiguous-prefix flags")
    return tuple(
        (start * STEP, (start + RESIDENCE_CELLS) * STEP)
        for start in range(len(flags) - RESIDENCE_CELLS + 1)
        if all(flags[start : start + RESIDENCE_CELLS])
    )


def _outcome(flags, reason):
    windows = residence_windows(flags)
    complete = len(flags) == CELL_COUNT and reason is None
    return dict(
        residence_status=(
            "certified_residence" if windows else "residence_not_certified"
        ),
        completion_status="complete" if complete else "integration_incomplete",
        integration_complete=complete,
        cells_completed=len(flags),
        completed_through=len(flags) * STEP,
        certified_windows=windows,
        first_certified_window=windows[0] if windows else None,
        failure_reason=reason,
        absence_claimed=False,
        empirical_validation=False,
    )


def run_trajectory(output):
    """Execute the one fixed protocol, retaining every completed step once.

    The parent worker supplies the hard900s limit and live300s status notice.
    This runner also checks that budget between cells. An externally killed
    worker may leave a partial next file; only complete parsed cell records
    count toward the retained prefix. An existing attempt refuses replay.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    write_json_once(
        output / "attempt.json",
        dict(
            schema="tnfr.replica-fixed-budget-trajectory.v1",
            horizon=HORIZON,
            step=STEP,
            order=ORDER,
            cell_count=CELL_COUNT,
            wall_seconds=WALL_SECONDS,
            margin=MARGIN,
            residence_cells=RESIDENCE_CELLS,
            phase_capture="exact_rational_interval_no_graph_recapture",
        ),
    )
    source = declared_source()
    write_json_once(output / "source.json", asdict(source))
    flags, reason, status_sent = [], None, False
    if not source.certified:
        reason = "declared_source_bounds_unavailable"
    else:
        field = complete_field()
        box = source.initial_box
        for index in range(CELL_COUNT):
            elapsed = monotonic() - started
            if elapsed >= STATUS_SECONDS and not status_sent:
                print(
                    f"status: {index}/{CELL_COUNT} cells completed at {elapsed:.1f}s",
                    flush=True,
                )
                status_sent = True
            if elapsed >= WALL_SECONDS:
                reason = "wall_budget_exhausted_before_next_cell"
                write_json_once(
                    output / "failure.json",
                    dict(
                        cell_index=index,
                        time=index * STEP,
                        reason=reason,
                        initial_box=[asdict(value) for value in box],
                        elapsed_seconds=elapsed,
                    ),
                )
                break
            print(
                f"cell {index + 1}/{CELL_COUNT}: starting t={index * STEP}", flush=True
            )
            step, failed, failure_reason = validated_box_taylor_step(
                box,
                STEP,
                field,
                _smooth_domain,
                order=ORDER,
                time=index * STEP,
            )
            if step is None:
                reason = failure_reason or "step_unavailable_without_reason"
                write_json_once(
                    output / "failure.json",
                    dict(
                        cell_index=index,
                        time=index * STEP,
                        reason=reason,
                        initial_box=[asdict(value) for value in box],
                        failed_tube=(
                            None
                            if failed is None
                            else [asdict(value) for value in failed]
                        ),
                    ),
                )
                break
            if (
                step.time != index * STEP
                or step.duration != STEP
                or step.initial_box != box
            ):
                raise ArithmeticError(
                    "shared step changed the declared clock or source box"
                )
            observation = observe_tube(step.tube)
            qualified = (
                source.global_chart_and_nontip_certified
                and observation.mean_geometry_certified
            )
            write_json_once(
                output / f"cell-{index:03d}.json",
                dict(
                    cell_index=index,
                    step=asdict(step),
                    observation=asdict(observation),
                    source_chart_used=source.global_chart_and_nontip_certified,
                    residence_cell_certified=qualified,
                    conditional_activity_lower_bound=(
                        source.target_activity_lower_bound if qualified else None
                    ),
                    elapsed_seconds=monotonic() - started,
                ),
            )
            flags.append(qualified)
            box = step.endpoint
            print(
                f"cell {index + 1}/{CELL_COUNT}: retained; geometry={qualified}",
                flush=True,
            )
    result = _outcome(flags, reason)
    result["elapsed_seconds"] = monotonic() - started
    write_json_once(output / "trajectory-result.json", result)
    return result
