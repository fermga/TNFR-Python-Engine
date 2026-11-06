"""One validated collective prediction using retained internal-error bounds.

Ten coordinates evolve on the existing synchronized invariant restriction.
The original full field is lifted and projected without introducing separate
pair uncertainty. Prior allowance arithmetic is re-admitted; no old producer
or actual-source trajectory is replayed. Geometry observations certify only
exclusion, conditional on the source chart and retained comparison theorem.
"""

from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path
from time import monotonic

from ..mathematics._comparison_flow import nonnegative_exponential_upper
from ..mathematics._interval_taylor import MAX_ORDER, Jet
from ..mathematics._rational_interval import INTERVAL_BITS, I, pi_interval, sqrt
from ..mathematics._validated_taylor import validated_box_taylor_step
from ..physics.relational_observations import _ordered
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
    write_json_once,
)
from .replica_collective_preparation import (
    CELL_COUNT,
    COLLECTIVE_SCALE,
    ERROR_RADIUS,
    HORIZON,
    MARGIN,
    ORDER,
    PAIRS,
    PROFILE,
    STEP,
    WALL_SECONDS,
    ReplicaPreparationSource,
    _exact,
    _source_center,
    complete_field,
)
from .replica_modal_criterion import _endpoints, _rows, c5_modes
from .replica_synchronized_comparison import CLOCK, _admitted_source
from .replica_target_separation import (
    ARCHIVE_LIMIT,
    JSON_LIMIT,
    _integer,
    _inventory,
    _match,
    _receipt,
    time_cover,
)

PRIOR_DIRECTORY = "artifacts/replica-internal-comparison-20261026"
FINAL_STATE_PATH = PRIOR_DIRECTORY + "/final-state.json"
CERTIFICATE_PATH = PRIOR_DIRECTORY + "/source-certificate.json"
REQUIRED_SOURCE_NAMES = (
    "src/tnfr/research/replica_synchronized_comparison.py",
    "src/tnfr/research/replica_modal_criterion.py",
    "src/tnfr/research/replica_collective_preparation.py",
    "src/tnfr/research/artifact_io.py",
    "src/tnfr/mathematics/_comparison_flow.py",
    "src/tnfr/mathematics/_exact_linear_algebra.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/_exact_time.py",
    "src/tnfr/utils/io.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ten_box(values):
    rows = _rows(values, 10, "collective state")
    endpoints = tuple(_endpoints(value, "collective coordinate") for value in rows)
    return tuple(I(*pair) for pair in endpoints)


def synchronized_field():
    """Lift exact/interval coordinates or common-order jets into identical pairs.

    The complete fine field retains both rows and the original clock. Pair
    entries share the same admitted scalar/jet; they are not independent
    evolving coordinates. The callback admits all coordinates before work.
    """
    fine_field = complete_field()

    def flow(values):
        rows = _rows(values, 10, "collective field state")
        if any(isinstance(value, Jet) for value in rows):
            _require(all(isinstance(value, Jet) for value in rows), "mixed jet state")
            order = rows[0].order
            _require(
                type(order) is int and 0 <= order <= MAX_ORDER, "invalid jet order"
            )
            _require(all(value.order == order for value in rows), "unequal jet orders")
            endpoints = tuple(
                tuple(_endpoints(value, "jet coefficient") for value in row.coeffs)
                for row in rows
            )
            state = tuple(Jet(tuple(I(*pair) for pair in row)) for row in endpoints)
        else:
            state = _ten_box(rows)
        lifted = tuple(value for value in state for _ in range(2))
        rates = fine_field(lifted)
        return tuple((rates[2 * i] + rates[2 * i + 1]) / 2 for i in range(10))

    return flow


@dataclass(frozen=True)
class PredictionSource:
    nominal: tuple[Q, ...]
    physical_coordinate_error: Q
    numerical_enclosure_allowance: Q
    initial_box: tuple[I, ...]
    endpoint_rounding: tuple[tuple[Q, Q], ...]
    memberwise_image_certified: bool
    clock: str = CLOCK
    state_dimension: int = 10
    nominal_only: bool = False
    whole_previous_fine_computational_box_claimed: bool = False
    equal_internal_storage_claimed: bool = False


def declared_prediction_source():
    """Enclose the intended memberwise means with one outward materialization.

    The physical fine errors average to at most the same epsilon. Additional
    interval rounding belongs to the numerical cover, not a new actual source
    or an initial mismatch in the internal-error comparison theorem.
    """
    center = _source_center(redistributed=True)
    means = tuple((center[a] + center[b]) / 2 for a, b in PAIRS) + tuple(
        (center[10 + a] + center[10 + b]) / 2 for a, b in PAIRS
    )
    nominal = tuple(COLLECTIVE_SCALE * value for value in PROFILE) + (Q(0),) * 5
    _require(means == nominal, "fine-to-collective nominal association differs")
    epsilon, slack = ERROR_RADIUS, Q(1, 2**INTERVAL_BITS)
    initial = tuple(I(value - epsilon, value + epsilon) for value in nominal)
    rounding = tuple(
        (value - epsilon - bound.lo, bound.hi - value - epsilon)
        for value, bound in zip(nominal, initial)
    )
    _require(
        all(0 <= value <= slack for row in rounding for value in row),
        "initial collective enclosure slack unavailable",
    )
    return PredictionSource(nominal, epsilon, slack, initial, rounding, True)


@dataclass(frozen=True)
class PredictionAllowances:
    source: ReplicaPreparationSource
    early_gap_upper: Q
    horizon_gap_upper: Q
    comparison_matrix: tuple[tuple[Q, ...], ...]
    initial_envelope: tuple[Q, ...]
    receipts: tuple[dict, ...] = ()
    final_state_receipt: dict | None = None
    file_bound: bool = False
    prior_kernel_arithmetic_rebuilt: bool = True
    conditional_on_retained_comparison_theorem: bool = True
    execution_authenticated: bool = False
    old_producer_replayed: bool = False
    actual_trajectory_read: bool = False


def admit_allowance_record(raw):
    """Re-admit source and two retained kernel records, never the old producer.

    Rebuilding the shared positive exponential is explicitly prior-certificate
    arithmetic validation. Stored target verdicts and an old actual trajectory
    do not enter the new predictor. No full-state field is evaluated here.
    """
    record = decode_exact_tree(raw)
    required = {
        "source",
        "geometry",
        "clock",
        "physical_chart_horizon",
        "alpha_upper",
        "kappa_upper",
        "forcing_coefficient_enclosure",
        "forcing_coefficient_upper",
        "internal_phase_initial_upper",
        "internal_form_initial_upper",
        "internal_scale_factors",
        "edge_quadratic_factors",
        "comparison_matrix",
        "initial_envelope",
        "bounds",
    }
    _require(
        type(record) is dict and required <= set(record), "incomplete allowance record"
    )
    source, cover, d, v = _admitted_source()
    _match(record["source"], asdict(source), "retained source")
    geometry = c5_modes()
    _match(record["geometry"], asdict(geometry), "retained C5 geometry")
    _match(record["clock"], CLOCK, "retained clock")
    _match(record["physical_chart_horizon"], HORIZON, "chart horizon")
    pi, eigenvalue = pi_interval(), geometry.eigenvalues[2]
    alpha, kappa = (1 / pi).hi, (eigenvalue / (2 * pi)).hi
    internal_u = tuple((source.nominal[a] - source.nominal[b]) / 2 for a, b in PAIRS)
    internal_delta = tuple(
        (source.nominal[10 + a] - source.nominal[10 + b]) / 2 for a, b in PAIRS
    )
    scales = tuple(
        max((u + cover) / v, (delta + cover) / d)
        for u, delta in zip(internal_u, internal_delta)
    )
    edges = tuple((scales[i] ** 2 + scales[(i + 1) % 5] ** 2) / 2 for i in range(5))
    forcing = sqrt(eigenvalue * sum((value**2 for value in edges), Q(0))) / (2 * pi)
    zero = Q(0)
    matrix = (
        (zero, kappa, forcing.hi, zero, zero),
        (kappa, zero, zero, zero, zero),
        (zero, zero, zero, 2 * alpha, zero),
        (zero, zero, alpha, zero, alpha),
        (zero, zero, zero, 2 * alpha, zero),
    )
    initial = (zero, zero, d * d, d * v, v * v)
    expected = dict(
        alpha_upper=alpha,
        kappa_upper=kappa,
        forcing_coefficient_enclosure=asdict(forcing),
        forcing_coefficient_upper=forcing.hi,
        internal_phase_initial_upper=d,
        internal_form_initial_upper=v,
        internal_scale_factors=scales,
        edge_quadratic_factors=edges,
        comparison_matrix=matrix,
        initial_envelope=initial,
    )
    for key, value in expected.items():
        _match(record[key], value, key)
    errors = []
    for raw_bound, time in zip(
        _rows(record["bounds"], 2, "fixed allowance bounds"), (Q(2), HORIZON)
    ):
        _require(type(raw_bound) is dict, "invalid allowance bound")
        kernel = nonnegative_exponential_upper(matrix, time)
        envelope = tuple(
            sum((entry * value for entry, value in zip(row, initial)), Q(0))
            for row in kernel.upper
        )
        gap = sqrt(I(2)).hi * envelope[1]
        expected_bound = dict(
            time=time,
            kernel=asdict(kernel),
            envelope_upper=envelope,
            collective_form_error_upper=envelope[0],
            collective_phase_error_upper=envelope[1],
            collective_gap_error_upper=gap,
            all_prior_times_covered=True,
            actual_error_evaluated=False,
            comparator_trajectory_evaluated=False,
        )
        _require(set(expected_bound) <= set(raw_bound), "incomplete retained bound")
        for key, value in expected_bound.items():
            _match(raw_bound[key], value, "retained bound." + key)
        errors.append(gap)
    _require(0 < errors[0] <= errors[1], "allowance ordering unavailable")
    return PredictionAllowances(source, *errors, matrix, initial)


def read_allowances(repository, *, final_state_receipt):
    """Bind the separately anchored prior certificate and its evaluated sources."""
    root, receipts = Path(repository).resolve(), {}

    def bound_bytes(raw, limit=JSON_LIMIT):
        row = _receipt(raw)
        path = (root / row["path"]).resolve()
        _require(path.is_relative_to(root), "allowance artifact escapes repository")
        _require(row["bytes"] <= limit, "allowance artifact exceeds read budget")
        data = read_bytes_bounded(path, max_bytes=limit)
        _require(
            len(data) == row["bytes"] and sha256_bytes(data) == row["sha256"],
            "allowance byte association differs",
        )
        receipts[row["path"]] = row
        return data

    def record(row):
        return decode_exact_tree(json_loads(bound_bytes(row)))

    anchor = _receipt(final_state_receipt)
    _require(anchor["path"] == FINAL_STATE_PATH, "wrong allowance manifest")
    final = record(anchor)
    _require(
        type(final) is dict
        and final.get("schema") == "tnfr.replica-internal-comparison-final.v1",
        "unsupported allowance manifest",
    )
    _require(
        {"artifacts", "evaluated_source", "evaluated_source_archive"} <= set(final),
        "incomplete allowance manifest",
    )
    artifacts = _inventory(final["artifacts"])
    names = [
        PRIOR_DIRECTORY + "/" + name
        for name in (
            "validation-plan.json",
            "evaluation-result.json",
            "source-certificate.json",
        )
    ]
    _require(
        all(name in artifacts for name in names), "missing retained allowance evidence"
    )
    plan = record(artifacts[names[0]])
    _require(
        type(plan) is dict and {"source", "source_archive"} <= set(plan),
        "incomplete allowance source plan",
    )
    sources = _inventory(plan["source"])
    _require(
        all(name in sources for name in REQUIRED_SOURCE_NAMES),
        "missing allowance production owner",
    )
    _require(
        _inventory(final["evaluated_source"]) == sources,
        "allowance source inventory differs",
    )
    archive = _receipt(plan["source_archive"])
    _require(
        _receipt(final["evaluated_source_archive"]) == archive,
        "allowance evaluated archive differs",
    )
    verify_archive_members(
        BytesIO(bound_bytes(archive, ARCHIVE_LIMIT)),
        {name: row["sha256"] for name, row in sources.items()},
        max_bytes=ARCHIVE_LIMIT,
    )
    for name, row in sources.items():
        if name.startswith("src/tnfr/") and name.endswith(".py"):
            bound_bytes(row)
    execution = record(artifacts[names[1]])
    _require(
        type(execution) is dict and "result" in execution,
        "incomplete allowance execution record",
    )
    _require(
        _receipt(execution["result"]) == artifacts[CERTIFICATE_PATH],
        "allowance result association differs",
    )
    admitted = admit_allowance_record(record(artifacts[CERTIFICATE_PATH]))
    return replace(
        admitted,
        receipts=tuple(receipts[name] for name in sorted(receipts)),
        final_state_receipt=anchor,
        file_bound=True,
    )


def allowance_for_cell(index, allowances):
    """Select an already per-gap allowance over the complete closed cell."""
    index = _integer(index, "cell index")
    _require(0 <= index < CELL_COUNT, "cell index outside the fixed horizon")
    early = _exact(allowances.early_gap_upper, "early allowance")
    late = _exact(allowances.horizon_gap_upper, "horizon allowance")
    _require(0 < early <= late, "allowances must be positive and ordered")
    return early if (index + 1) * STEP <= 2 else late


@dataclass(frozen=True)
class PredictionObservation:
    status: str
    available: bool
    raw_gaps: tuple[I, ...]
    allowance: Q
    dilated_target_lower: Q
    dilated_target_upper: Q
    possible_shifts: tuple[tuple[int, ...], ...]
    reachable_sums: tuple[int, ...]
    source_membership_inferred: bool = False
    actual_chart_inferred: bool = False
    positive_acquisition_claimed: bool = False


def observe_prediction(tube, error):
    """Exclude only when every closed dilated lift assignment is impossible.

    E is already a gap error. Initial/source and numerical uncertainty are in
    the predictor tube; neither another epsilon nor a doubled E is added.
    Missing branch coverage and endpoint contact remain unresolved.
    """
    error = _exact(error, "gap allowance")
    _require(error >= 0, "gap allowance must be nonnegative")
    state = _ten_box(tube)
    phases = state[5:]
    gaps = tuple(phases[(i + 1) % 5] - phases[i] for i in range(5))
    pi = pi_interval()
    lower, upper = MARGIN - error, pi.hi / 2 - MARGIN + error
    outside = 6 * pi.lo - 13
    available = (
        all(gap.abs_max < 13 for gap in gaps) and outside > upper and outside > -lower
    )
    if not available:
        return PredictionObservation(
            "unresolved", False, gaps, error, lower, upper, (), ()
        )
    possible = tuple(
        tuple(
            shift
            for shift in range(-2, 3)
            if (gap + 2 * shift * pi).hi >= lower and (gap + 2 * shift * pi).lo <= upper
        )
        for gap in gaps
    )
    reachable = {0}
    for choices in possible:
        reachable = {total + choice for total in reachable for choice in choices}
    status = "separated" if 1 not in reachable else "unresolved"
    return PredictionObservation(
        status, True, gaps, error, lower, upper, possible, tuple(sorted(reachable))
    )


def prediction_time_cover(statuses):
    """Fill every unevaluated suffix cell with unresolved coverage."""
    rows = _ordered(statuses, "prediction classifications", limit=CELL_COUNT + 1)
    _require(len(rows) <= CELL_COUNT, "too many prediction cells")
    _require(
        all(
            type(value) is str and value in ("separated", "unresolved")
            for value in rows
        ),
        "invalid prediction classification",
    )
    return time_cover(rows + ("unresolved",) * (CELL_COUNT - len(rows)))


def prediction_attempt(final_state_receipt):
    """Return the fixed numerical/source association written before work."""
    return dict(
        schema="tnfr.replica-synchronized-prediction.v1",
        horizon=HORIZON,
        step=STEP,
        order=ORDER,
        cell_count=CELL_COUNT,
        wall_seconds=WALL_SECONDS,
        margin=MARGIN,
        state_dimension=10,
        clock=CLOCK,
        allowance_final_state=_receipt(final_state_receipt),
    )


def prediction_outcome(observations, reason=None):
    """Summarize only admitted completed cells, retaining any missing suffix."""
    observations = _ordered(
        observations, "prediction observations", limit=CELL_COUNT + 1
    )
    cover = prediction_time_cover(tuple(item.status for item in observations))
    complete = len(observations) == CELL_COUNT and reason is None
    status = (
        "comparison_incomplete_or_evidence_unavailable"
        if not complete
        else (
            "full_horizon_target_excluded_by_reduced_prediction"
            if cover.full_horizon_separated
            else "reduced_prediction_task_resolution_incomplete"
        )
    )
    return dict(
        status=status,
        completed_cells=len(observations),
        separated_cells=sum(item.status == "separated" for item in observations),
        unavailable_cells=sum(not item.available for item in observations),
        unresolved_cells=CELL_COUNT
        - sum(item.status == "separated" for item in observations),
        time_cover=asdict(cover),
        failure_reason=reason,
        evolving_coordinates=10,
        actual_source_integrations=0,
        actual_error_evaluated=False,
        acquisition_claimed=False,
        conditional_on_validated_comparison=True,
        conditional_on_internal_error_theorem=True,
        independent_predictive_experiment=False,
        empirical_validation=False,
    )


def _smooth_domain(_tube):
    return (Q(1),)


def run_prediction(output, repository, *, final_state_receipt):
    """Run one fixed ten-coordinate predictor, preserving each complete cell.

    The parent enforces the hard900s deadline including evidence admission.
    This runner additionally checks the same elapsed budget between cells.
    Exclusive attempt/cell records block unrecorded retries.
    """
    started = monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json_once(output / "attempt.json", prediction_attempt(final_state_receipt))
    observations, reason = [], None
    try:
        allowances = read_allowances(
            repository, final_state_receipt=final_state_receipt
        )
        source = declared_prediction_source()
        write_json_once(output / "source.json", asdict(source))
        write_json_once(output / "allowances.json", asdict(allowances))
        field, box = synchronized_field(), source.initial_box
        for index in range(CELL_COUNT):
            if monotonic() - started >= WALL_SECONDS:
                reason = "wall_budget_exhausted_before_next_cell"
                write_json_once(
                    output / "failure.json", dict(cell_index=index, reason=reason)
                )
                break
            print(
                f"cell {index + 1}/{CELL_COUNT}: starting t={index * STEP}", flush=True
            )
            step, failed, failure = validated_box_taylor_step(
                box,
                STEP,
                field,
                _smooth_domain,
                order=ORDER,
                time=index * STEP,
            )
            if step is None:
                reason = failure or "step_unavailable_without_reason"
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
            _require(
                step.time == index * STEP
                and step.duration == STEP
                and step.initial_box == box,
                "shared step changed source or clock",
            )
            error = allowance_for_cell(index, allowances)
            observation = observe_prediction(step.tube, error)
            write_json_once(
                output / f"cell-{index:03d}.json",
                dict(
                    cell_index=index,
                    step=asdict(step),
                    allowance=error,
                    observation=asdict(observation),
                    elapsed_seconds=monotonic() - started,
                ),
            )
            observations.append(observation)
            box = step.endpoint
            print(
                f"cell {index + 1}/{CELL_COUNT}: retained; {observation.status}",
                flush=True,
            )
    except (ValueError, TypeError, ArithmeticError, OSError) as exc:
        reason = f"prediction_unavailable: {type(exc).__name__}: {exc}"
        if not (output / "failure.json").exists():
            write_json_once(
                output / "failure.json",
                dict(cell_index=len(observations), reason=reason),
            )
    result = prediction_outcome(observations, reason)
    result["elapsed_seconds"] = monotonic() - started
    write_json_once(output / "trajectory-result.json", result)
    return result
