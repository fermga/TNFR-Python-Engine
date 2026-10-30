"""Use the unchanged positive majorant at five retained cell endpoints.

All new exponential evaluations start from the same original majorant state
at time zero. Their capped bounds replace only the five previously unresolved
allowances. Earlier source, prediction and threshold arithmetic is re-admitted
without replaying a scientific producer or integrating either trajectory.
"""

from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path

from ..mathematics._comparison_flow import (
    NonnegativeExponentialUpper,
    nonnegative_exponential_upper,
)
from ..mathematics._rational_interval import I, sqrt
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)
from .replica_collective_preparation import CELL_COUNT, HORIZON, STEP, _exact
from .replica_prediction_resolution import (
    ResolutionObservation,
    RetainedPrediction,
    read_prediction,
    resolve_prediction,
)
from .replica_synchronized_prediction import (
    allowance_for_cell,
    observe_prediction,
    prediction_time_cover,
)
from .replica_target_separation import (
    ARCHIVE_LIMIT,
    JSON_LIMIT,
    ReplicaTimeCover,
    _inventory,
    _match,
    _receipt,
    _rows,
)

TARGET_CELLS = (36, 37, 38, 39, 40)
ENDPOINT_TIMES = tuple((index + 1) * STEP for index in TARGET_CELLS)
PRIOR_DIRECTORY = "artifacts/replica-prediction-resolution-20261027"
FINAL_STATE_PATH = PRIOR_DIRECTORY + "/final-state.json"
REQUIRED_SOURCE_NAMES = (
    "src/tnfr/research/replica_prediction_resolution.py",
    "src/tnfr/research/replica_synchronized_prediction.py",
    "src/tnfr/research/replica_synchronized_comparison.py",
    "src/tnfr/research/replica_collective_preparation.py",
    "src/tnfr/research/replica_target_separation.py",
    "src/tnfr/research/artifact_io.py",
    "src/tnfr/mathematics/_comparison_flow.py",
    "src/tnfr/mathematics/_exact_linear_algebra.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/utils/io.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative_vector(raw, size, label):
    values = tuple(_exact(value, label) for value in _rows(raw, size, label))
    _require(all(value >= 0 for value in values), label + " must be nonnegative")
    return values


@dataclass(frozen=True)
class EndpointAllowance:
    time: Q
    kernel: NonnegativeExponentialUpper
    initial_envelope: tuple[Q, ...]
    envelope_upper: tuple[Q, ...]
    uncapped_gap_upper: Q
    retained_gap_cap: Q
    capped_gap_upper: Q
    all_prior_times_covered: bool = True
    majorant_restarted: bool = False
    numerical_bounds_assumed_monotone: bool = False
    actual_error_measured: bool = False


def evaluate_endpoint(matrix, initial, time, cap):
    """Evaluate the same five-dimensional positive majorant at absolute time.

    Nonnegative coefficients and initial state justify monotonicity of the
    exact majorant. The supplied cap must separately bound the same gap error
    through this time; that source association belongs to the retained reader.
    All primitive values are admitted before the exponential or square root.
    """
    time, cap = _exact(time, "absolute endpoint time"), _exact(cap, "retained cap")
    rows = tuple(
        _nonnegative_vector(row, 5, "majorant row")
        for row in _rows(matrix, 5, "majorant matrix")
    )
    initial = _nonnegative_vector(initial, 5, "initial majorant")
    _require(0 <= time <= HORIZON, "endpoint time outside the retained horizon")
    _require(cap >= 0, "retained cap must be nonnegative")
    kernel = nonnegative_exponential_upper(rows, time)
    envelope = tuple(
        sum((entry * value for entry, value in zip(row, initial)), Q(0))
        for row in kernel.upper
    )
    gap = sqrt(I(2)).hi * envelope[1]
    return EndpointAllowance(time, kernel, initial, envelope, gap, cap, min(gap, cap))


def classify_allowance(upper, threshold):
    """Compare admitted primitive scalars; closed threshold contact stays possible."""
    upper, threshold = _exact(upper, "gap allowance"), _exact(
        threshold, "critical dilation"
    )
    _require(
        upper >= 0 and threshold >= 0, "allowance and threshold must be nonnegative"
    )
    return "separated" if upper < threshold else "unresolved"


def merge_allowances(previous, replacements):
    """Replace only the five declared cells by nonincreasing exact allowances."""
    previous = _nonnegative_vector(previous, CELL_COUNT, "original allowances")
    replacements = _nonnegative_vector(
        replacements, len(TARGET_CELLS), "new allowances"
    )
    _require(
        all(
            value <= previous[index] for index, value in zip(TARGET_CELLS, replacements)
        ),
        "a replacement exceeds its retained allowance",
    )
    merged = list(previous)
    for index, value in zip(TARGET_CELLS, replacements):
        merged[index] = value
    return tuple(merged)


def _classifications(raw, size):
    rows = _rows(raw, size, "cell classifications")
    _require(
        all(
            type(value) is str and value in ("separated", "unresolved")
            for value in rows
        ),
        "invalid cell classification",
    )
    return rows


def endpoint_outcome(previous, replacements):
    """Keep forty-three prior decisions and rebuild the complete closed-time cover."""
    previous = _classifications(previous, CELL_COUNT)
    replacements = _classifications(replacements, len(TARGET_CELLS))
    _require(
        tuple(index for index, value in enumerate(previous) if value == "unresolved")
        == TARGET_CELLS,
        "the prior unresolved cells differ from the declared consumer",
    )
    merged = list(previous)
    for index, value in zip(TARGET_CELLS, replacements):
        merged[index] = value
    merged = tuple(merged)
    cover = prediction_time_cover(merged)
    status = (
        "full_horizon_target_excluded_by_endpoint_allowances"
        if cover.full_horizon_separated
        else (
            "half_unit_residence_excluded_by_endpoint_allowances"
            if cover.residence_excluded
            else "endpoint_allowance_resolution_incomplete"
        )
    )
    return dict(
        status=status,
        updated_classifications=merged,
        time_cover=asdict(cover),
        newly_separated_cells=sum(value == "separated" for value in replacements),
        retained_separated_cells=CELL_COUNT - len(TARGET_CELLS),
        unresolved_cells=sum(value == "unresolved" for value in merged),
    )


@dataclass(frozen=True)
class EndpointPreparation:
    prediction: RetainedPrediction
    resolution_observations: tuple[ResolutionObservation, ...]
    thresholds: tuple[Q, ...]
    cell_indices: tuple[int, ...] = TARGET_CELLS
    receipts: tuple[dict, ...] = ()
    final_state_receipt: dict | None = None
    file_bound: bool = False
    prior_threshold_arithmetic_rebuilt: bool = True
    execution_authenticated: bool = False
    prior_producer_replayed: bool = False
    actual_source_trajectory_read: bool = False


def admit_resolution_record(raw, *, prediction):
    """Re-admit the five consumed thresholds against an already admitted prediction.

    This pure entry accepts constructed, non-file-bound prediction records.
    The outer reader obtains its prediction only from the existing bound
    reader. Rebuilding old threshold arithmetic is not a new minimax study.
    """
    record = decode_exact_tree(raw)
    required = {
        "evidence",
        "observations",
        "prior_unresolved_indices",
        "frozen_classifications",
    }
    _require(
        type(record) is dict and required <= set(record), "incomplete resolution record"
    )
    _match(record["evidence"], asdict(prediction), "retained prediction evidence")
    _match(record["prior_unresolved_indices"], TARGET_CELLS, "prior unresolved indices")
    frozen = tuple(item.status for item in prediction.observations)
    _match(record["frozen_classifications"], frozen, "frozen classifications")
    _require(
        tuple(index for index, value in enumerate(frozen) if value == "unresolved")
        == TARGET_CELLS,
        "retained prediction has a different unresolved consumer",
    )
    records = _rows(
        record["observations"], CELL_COUNT, "retained resolution observations"
    )
    observations, thresholds = [], []
    for index in TARGET_CELLS:
        observation = resolve_prediction(
            prediction.tubes[index], prediction.allowances.horizon_gap_upper
        )
        _match(records[index], asdict(observation), "consumed threshold observation")
        _require(
            observation.available
            and observation.threshold_status == "within_domain"
            and observation.threshold is not None
            and observation.threshold > 0,
            "consumed positive threshold unavailable",
        )
        _require(
            classify_allowance(
                allowance_for_cell(index, prediction.allowances), observation.threshold
            )
            == frozen[index],
            "threshold does not reconcile its frozen classification",
        )
        observations.append(observation)
        thresholds.append(observation.threshold)
    return EndpointPreparation(prediction, tuple(observations), tuple(thresholds))


def read_resolution(repository, *, final_state_receipt):
    """Bind the closed resolution report and its evaluated production owners."""
    root, receipts = Path(repository).resolve(), {}

    def bound_bytes(raw, limit=JSON_LIMIT):
        row = _receipt(raw)
        path = (root / row["path"]).resolve()
        _require(path.is_relative_to(root), "resolution artifact escapes repository")
        _require(row["bytes"] <= limit, "resolution artifact exceeds read budget")
        data = read_bytes_bounded(path, max_bytes=limit)
        _require(
            len(data) == row["bytes"] and sha256_bytes(data) == row["sha256"],
            "resolution byte association differs",
        )
        receipts[row["path"]] = row
        return data

    def record(row):
        return decode_exact_tree(json_loads(bound_bytes(row)))

    anchor = _receipt(final_state_receipt)
    _require(anchor["path"] == FINAL_STATE_PATH, "wrong resolution final manifest")
    final = record(anchor)
    _require(
        type(final) is dict
        and final.get("schema") == "tnfr.replica-prediction-resolution-final.v1",
        "unsupported resolution manifest",
    )
    _require(
        {"artifacts", "evaluated_source", "evaluated_source_archive"} <= set(final),
        "incomplete resolution manifest",
    )
    artifacts = _inventory(final["artifacts"])
    names = tuple(
        PRIOR_DIRECTORY + "/" + name
        for name in (
            "validation-plan.json",
            "execution-result.json",
            "resolution-result.json",
        )
    )
    _require(all(name in artifacts for name in names), "missing resolution evidence")
    plan = record(artifacts[names[0]])
    _require(
        type(plan) is dict and {"source", "source_archive"} <= set(plan),
        "incomplete resolution source plan",
    )
    sources = _inventory(plan["source"])
    _require(
        all(name in sources for name in REQUIRED_SOURCE_NAMES),
        "missing resolution production owner",
    )
    _require(
        _inventory(final["evaluated_source"]) == sources,
        "resolution evaluated inventory differs",
    )
    archive = _receipt(plan["source_archive"])
    _require(
        _receipt(final["evaluated_source_archive"]) == archive,
        "resolution evaluated archive differs",
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
        "missing resolution execution association",
    )
    _require(
        _receipt(execution["result"]) == artifacts[names[2]],
        "resolution result association differs",
    )
    response = record(artifacts[names[2]])
    _require(
        type(response) is dict
        and type(response.get("evidence")) is dict
        and "final_state_receipt" in response["evidence"],
        "missing original prediction anchor",
    )
    prediction = read_prediction(
        root, final_state_receipt=response["evidence"]["final_state_receipt"]
    )
    admitted = admit_resolution_record(response, prediction=prediction)
    for row in prediction.receipts:
        _require(
            row["path"] not in receipts or receipts[row["path"]] == row,
            "nested prediction receipt conflicts",
        )
        receipts[row["path"]] = row
    return replace(
        admitted,
        receipts=tuple(receipts[name] for name in sorted(receipts)),
        final_state_receipt=anchor,
        file_bound=True,
    )


@dataclass(frozen=True)
class EndpointAllowanceAssessment:
    preparation: EndpointPreparation
    endpoint_allowances: tuple[EndpointAllowance, ...]
    original_allowances: tuple[Q, ...]
    updated_allowances: tuple[Q, ...]
    original_classifications: tuple[str, ...]
    updated_classifications: tuple[str, ...]
    time_cover: ReplicaTimeCover
    status: str
    newly_separated_cells: int
    unresolved_cells: int
    retained_separated_cells: int = CELL_COUNT - len(TARGET_CELLS)
    new_endpoint_evaluations: int = len(TARGET_CELLS)
    new_majorants: int = 0
    new_integrations: int = 0
    prior_producer_replays: int = 0
    same_original_initial_majorant: bool = True
    original_source_and_tubes_unchanged: bool = True
    actual_error_measured: bool = False
    cause_identified: bool = False
    exact_target_existence_claimed: bool = False
    independent_predictive_experiment: bool = False
    empirical_validation: bool = False


def assess_endpoint_allowances(repository, *, final_state_receipt):
    """Perform the sole five-time calculation and retain all other allowances."""
    preparation = read_resolution(repository, final_state_receipt=final_state_receipt)
    prediction = preparation.prediction
    prior = prediction.allowances
    endpoint_allowances = tuple(
        evaluate_endpoint(
            prior.comparison_matrix,
            prior.initial_envelope,
            time,
            prior.horizon_gap_upper,
        )
        for time in ENDPOINT_TIMES
    )
    original = tuple(allowance_for_cell(index, prior) for index in range(CELL_COUNT))
    updated = merge_allowances(
        original, tuple(item.capped_gap_upper for item in endpoint_allowances)
    )
    old_statuses = tuple(item.status for item in prediction.observations)
    new_statuses = tuple(
        classify_allowance(item.capped_gap_upper, threshold)
        for item, threshold in zip(endpoint_allowances, preparation.thresholds)
    )
    for index, status in zip(TARGET_CELLS, new_statuses):
        observation = observe_prediction(prediction.tubes[index], updated[index])
        _require(
            observation.available and observation.status == status,
            "endpoint decision disagrees with the closed observer",
        )
    outcome = endpoint_outcome(old_statuses, new_statuses)
    return EndpointAllowanceAssessment(
        preparation,
        endpoint_allowances,
        original,
        updated,
        old_statuses,
        outcome["updated_classifications"],
        prediction_time_cover(outcome["updated_classifications"]),
        outcome["status"],
        outcome["newly_separated_cells"],
        outcome["unresolved_cells"],
    )
