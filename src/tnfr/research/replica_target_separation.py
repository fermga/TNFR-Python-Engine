"""Detached target/separation analysis of one retained preparation protocol.

No field, derivative, Picard or integration evaluation occurs here. File
receipts associate the supplied evidence with independently frozen bytes;
they do not authenticate execution. Conclusions remain conditional on the
previous producer's validated enclosures and the stated source contract.
"""

import re
from dataclasses import asdict, dataclass, replace
from fractions import Fraction as Q
from io import BytesIO
from pathlib import Path

from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_taylor import reconstruct_box_taylor_arithmetic
from ..physics.relational_observations import _ordered
from ..utils.io import json_loads
from .artifact_io import (
    decode_exact_tree,
    read_bytes_bounded,
    sha256_bytes,
    validate_artifact_path,
    verify_archive_members,
)
from .replica_collective_preparation import (
    CELL_COUNT,
    HORIZON,
    MARGIN,
    ORDER,
    RESIDENCE_CELLS,
    STEP,
    WALL_SECONDS,
    ReplicaPreparationSource,
    ReplicaTubeObservation,
    _exact,
    declared_source,
    observe_tube,
    residence_windows,
)

JSON_LIMIT = 16 * 1024**2
ARCHIVE_LIMIT = 128 * 1024**2
PRIOR_DIRECTORY = "artifacts/replica-fixed-budget-formation-20261026"
FINAL_STATE_PATH = PRIOR_DIRECTORY + "/final-state.json"
STATUSES = ("target_certified", "separated", "unresolved")
STEP_METHOD = "direct_source_box_Picard_Taylor_dyadic128_v1"
REQUIRED_SOURCE_NAMES = (
    "src/tnfr/research/replica_collective_preparation.py",
    "src/tnfr/research/artifact_io.py",
    "src/tnfr/_exact_time.py",
    "src/tnfr/mathematics/_rational_interval.py",
    "src/tnfr/mathematics/_interval_taylor.py",
    "src/tnfr/mathematics/_validated_taylor.py",
    "src/tnfr/physics/_sine_flow.py",
    "src/tnfr/physics/relational_observations.py",
    "src/tnfr/dynamics/relational.py",
    "src/tnfr/utils/io.py",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(values, size, label):
    result = _ordered(values, label, limit=size + 1)
    _require(len(result) == size, f"{label} dimension differs")
    return result


def _integer(value, label):
    if type(value) is not int:
        raise TypeError(f"{label} requires an ordinary integer")
    return value


def _match(raw, expected, label):
    """Admit types before comparing a fixed source or protocol tree."""
    if type(expected) is Q:
        normalized = _exact(raw, label)
    elif type(expected) is int:
        normalized = _integer(raw, label)
    elif type(expected) is bool:
        _require(type(raw) is bool, f"{label} requires a Boolean flag")
        normalized = raw
    elif expected is None:
        _require(raw is None, f"{label} must be unavailable")
        normalized = None
    elif type(expected) is str:
        _require(type(raw) is str, f"{label} requires text")
        normalized = raw
    elif isinstance(expected, (tuple, list)):
        normalized = tuple(
            _match(value, target, label)
            for value, target in zip(_rows(raw, len(expected), label), expected)
        )
        expected = tuple(expected)
    elif type(expected) is dict:
        _require(
            type(raw) is dict and set(raw) == set(expected), f"{label} keys differ"
        )
        normalized = {
            key: _match(raw[key], value, f"{label}.{key}")
            for key, value in expected.items()
        }
    else:
        raise TypeError("unsupported fixed source schema")
    _require(normalized == expected, f"{label} differs from its primitive rebuild")
    return normalized


def _interval(raw):
    if isinstance(raw, I):
        lower, upper = raw.lo, raw.hi
    else:
        _require(
            type(raw) is dict and set(raw) == {"lo", "hi"}, "invalid interval record"
        )
        lower, upper = raw["lo"], raw["hi"]
    lower, upper = _exact(lower, "interval lower"), _exact(upper, "interval upper")
    _require(lower <= upper, "reversed interval")
    result = I(lower, upper)
    # Retained kernel endpoints are dyadic. Do not silently replace their
    # recorded bounds with a widened interval during evidence admission.
    _require(
        result.lo == lower and result.hi == upper,
        "retained endpoint is not representable exactly",
    )
    return result


def _vector(raw, label):
    return tuple(_interval(value) for value in _rows(raw, 20, label))


@dataclass(frozen=True)
class ReplicaTargetObservation:
    status: str
    available: bool
    possible_shifts: tuple[tuple[int, ...], ...]
    reachable_sums: tuple[int, ...]
    positive_observation: ReplicaTubeObservation
    source_membership_inferred: bool = False


def observe_target(tube):
    """Classify a supplied whole box using sufficient, distinct certificates.

    An overlap with the outer target stays possible, including endpoint
    contact. Rectangular gap/constant relaxations can leave unresolved cases;
    they cannot justify deleting a possible physical assignment.
    """
    positive = observe_tube(tube)
    pi = pi_interval()
    branch_complete = pi.lo > 3 and 6 * pi.lo - 13 > pi.hi / 2 - MARGIN
    available = (
        positive.box_chart_certified
        and positive.raw_gap_range_certified
        and branch_complete
    )
    if not available:
        return ReplicaTargetObservation("unresolved", False, (), (), positive)
    possible = tuple(
        tuple(
            shift
            for shift in range(-2, 3)
            if (gap + 2 * shift * pi).hi >= MARGIN
            and (gap + 2 * shift * pi).lo <= pi.hi / 2 - MARGIN
        )
        for gap in positive.raw_gaps
    )
    reachable = {0}
    for choices in possible:
        reachable = {total + choice for total in reachable for choice in choices}
    separated = 1 not in reachable
    _require(
        not (separated and positive.mean_geometry_certified),
        "inconsistent positive and negative certificates",
    )
    status = (
        "target_certified"
        if positive.mean_geometry_certified
        else "separated" if separated else "unresolved"
    )
    return ReplicaTargetObservation(
        status, True, possible, tuple(sorted(reachable)), positive
    )


@dataclass(frozen=True)
class PossibleTimeComponent:
    start: Q
    end: Q
    left_included: bool
    right_included: bool
    residence_possible: bool


@dataclass(frozen=True)
class ReplicaTimeCover:
    components: tuple[PossibleTimeComponent, ...]
    residence_excluded: bool
    full_horizon_separated: bool
    positive_residence_windows: tuple[tuple[Q, Q], ...]
    existence_claimed: bool = False


def time_cover(classifications):
    """Cover the fixed full horizon; labels alone do not establish evidence.

    Separation removes closed cells and both their endpoints. A component
    of length exactly the residence duration admits a closed window only
    when both of its endpoints are available. Unresolved cells stay possible.
    """
    rows = _rows(classifications, CELL_COUNT, "cell classifications")
    _require(
        all(type(value) is str and value in STATUSES for value in rows),
        "invalid classification",
    )
    _require(
        not any(
            {left, right} == {"target_certified", "separated"}
            for left, right in zip(rows, rows[1:])
        ),
        "closed adjacent membership/separation certificates conflict",
    )
    components, start = [], None
    duration = RESIDENCE_CELLS * STEP
    for index in range(CELL_COUNT + 1):
        possible = index < CELL_COUNT and rows[index] != "separated"
        if possible and start is None:
            start = index
        if not possible and start is not None:
            left, right = start == 0, index == CELL_COUNT
            length = (index - start) * STEP
            components.append(
                PossibleTimeComponent(
                    start * STEP,
                    index * STEP,
                    left,
                    right,
                    length > duration or (length == duration and left and right),
                )
            )
            start = None
    windows = residence_windows(tuple(value == "target_certified" for value in rows))
    excluded = all(not item.residence_possible for item in components)
    # Abstract contradictory labels cannot become a temporal certificate.
    _require(
        not (excluded and windows),
        "positive residence conflicts with closed separation",
    )
    return ReplicaTimeCover(tuple(components), excluded, not components, windows)


@dataclass(frozen=True)
class RetainedReplicaTrajectory:
    source: ReplicaPreparationSource
    tubes: tuple[tuple[I, ...], ...]
    initial_box: tuple[I, ...]
    endpoint: tuple[I, ...]
    cell_count: int = CELL_COUNT
    receipts: tuple[dict, ...] = ()
    file_bound: bool = False
    derivative_or_picard_recomputed: bool = False
    execution_authenticated: bool = False


def admit_retained_records(attempt, source_record, cells):
    """Read complete primitive evidence without replaying its numerical proof.

    Artificial traces may satisfy these arithmetic checks. Their derivatives
    and Picard margins are NOT independently certified here. The file-backed
    caller separately binds the previously reviewed producer and response.
    """
    expected_attempt = dict(
        schema="tnfr.replica-fixed-budget-trajectory.v1",
        horizon=HORIZON,
        step=STEP,
        order=ORDER,
        cell_count=CELL_COUNT,
        wall_seconds=WALL_SECONDS,
        margin=MARGIN,
        residence_cells=RESIDENCE_CELLS,
        phase_capture="exact_rational_interval_no_graph_recapture",
    )
    _match(attempt, expected_attempt, "attempt")
    source = declared_source()
    _require(source.certified, "fresh fixed source bounds unavailable")
    _match(source_record, asdict(source), "source")
    records = _rows(cells, CELL_COUNT, "retained cells")
    box, tubes = source.initial_box, []
    keys = {
        "time",
        "duration",
        "order",
        "initial_box",
        "tube",
        "series",
        "local_remainder_bounds",
        "increment",
        "endpoint",
        "picard_interior_margin",
        "domain_lower_bounds",
        "method",
    }
    for index, cell in enumerate(records):
        _require(
            type(cell) is dict and {"cell_index", "step"} <= set(cell),
            "invalid cell record",
        )
        _require(
            _integer(cell["cell_index"], "cell index") == index,
            "noncontiguous cell index",
        )
        step = cell["step"]
        _require(type(step) is dict and set(step) == keys, "step schema differs")
        time, duration = _exact(step["time"], "cell time"), _exact(
            step["duration"], "cell duration"
        )
        order = _integer(step["order"], "Taylor order")
        _require(
            time == index * STEP and duration == STEP and order == ORDER,
            "cell clock or numerical policy differs",
        )
        _require(
            type(step["method"]) is str and step["method"] == STEP_METHOD,
            "step method differs",
        )
        initial, tube = _vector(step["initial_box"], "initial box"), _vector(
            step["tube"], "tube"
        )
        series = tuple(
            tuple(
                _interval(value)
                for value in _rows(row, ORDER + 1, "Taylor coefficients")
            )
            for row in _rows(step["series"], 20, "Taylor rows")
        )
        remainder = _vector(step["local_remainder_bounds"], "Taylor remainder")
        increment, endpoint = _vector(step["increment"], "increment"), _vector(
            step["endpoint"], "endpoint"
        )
        margin = _exact(step["picard_interior_margin"], "Picard margin")
        domains = tuple(
            _exact(value, "domain margin")
            for value in _rows(step["domain_lower_bounds"], 1, "domain margins")
        )
        _require(
            margin > 0 and domains == (Q(1),),
            "retained smooth-domain or Picard premise unavailable",
        )
        _require(initial == box, "initial/endpoint chain differs")
        try:
            rebuilt_increment, rebuilt_endpoint = reconstruct_box_taylor_arithmetic(
                initial, tube, series, remainder, duration, order=order
            )
        except (ValueError, ArithmeticError) as exc:
            raise ValueError("retained Taylor arithmetic unavailable") from exc
        _require(
            increment == rebuilt_increment and endpoint == rebuilt_endpoint,
            "retained endpoint arithmetic differs",
        )
        box = endpoint
        tubes.append(tube)
    return RetainedReplicaTrajectory(source, tuple(tubes), source.initial_box, box)


def _receipt(raw):
    _require(
        type(raw) is dict and set(raw) == {"path", "bytes", "sha256"},
        "invalid content receipt",
    )
    name = validate_artifact_path(raw["path"])
    size = _integer(raw["bytes"], "receipt byte count")
    digest = raw["sha256"]
    _require(
        size >= 0
        and type(digest) is str
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        "invalid receipt size/digest",
    )
    return dict(path=name, bytes=size, sha256=digest)


def _inventory(rows):
    result = {}
    aliases = set()
    for raw in _ordered(rows, "receipt inventory", limit=20001):
        row = _receipt(raw)
        name = row["path"]
        _require(name.casefold() not in aliases, "duplicate or aliased receipt path")
        aliases.add(name.casefold())
        result[name] = row
    return result


def read_retained_trajectory(repository, *, final_state_receipt):
    """Bind one separately frozen receipt and re-admit the complete response.

    The caller must obtain the expected final-state receipt independently of
    this read (the prospective scope does so). Hashes supply byte association,
    not execution authentication. Each payload is parsed from the same bytes
    whose digest and bounded size were checked.
    """
    root = Path(repository).resolve()
    receipts = {}

    def bound_bytes(raw, limit=JSON_LIMIT):
        row = _receipt(raw)
        path = (root / row["path"]).resolve()
        _require(path.is_relative_to(root), "artifact escapes repository root")
        _require(row["bytes"] <= limit, "declared artifact exceeds read budget")
        data = read_bytes_bounded(path, max_bytes=limit)
        _require(
            len(data) == row["bytes"] and sha256_bytes(data) == row["sha256"],
            "artifact byte association differs",
        )
        receipts[row["path"]] = row
        return data

    def record(row):
        return decode_exact_tree(json_loads(bound_bytes(row)))

    anchor = _receipt(final_state_receipt)
    _require(anchor["path"] == FINAL_STATE_PATH, "wrong retained protocol manifest")
    final = record(anchor)
    _require(
        type(final) is dict
        and final.get("schema") == "tnfr.replica-fixed-budget-formation-final.v1",
        "unsupported prior manifest",
    )
    _require(
        {"artifacts", "evaluated_source", "evaluated_source_archive"} <= set(final),
        "incomplete prior manifest",
    )
    artifacts = _inventory(final["artifacts"])
    required = [
        PRIOR_DIRECTORY + "/" + name
        for name in (
            "validation-plan.json",
            "execution-result.json",
            "prefix-audit.json",
            "trajectory/attempt.json",
            "trajectory/source.json",
            "trajectory/trajectory-result.json",
        )
    ]
    cell_names = tuple(
        PRIOR_DIRECTORY + f"/trajectory/cell-{index:03d}.json"
        for index in range(CELL_COUNT)
    )
    _require(
        all(name in artifacts for name in required + list(cell_names)),
        "required retained evidence is missing",
    )
    actual_cells = {
        name
        for name in artifacts
        if name.startswith(PRIOR_DIRECTORY + "/trajectory/cell-")
    }
    _require(actual_cells == set(cell_names), "retained cell inventory differs")
    plan = record(artifacts[PRIOR_DIRECTORY + "/validation-plan.json"])
    _require(
        type(plan) is dict and {"source", "source_archive"} <= set(plan),
        "incomplete evaluated-source plan",
    )
    sources = _inventory(plan["source"])
    _require(
        all(name in sources for name in REQUIRED_SOURCE_NAMES),
        "required numerical/source owner missing",
    )
    _require(
        _inventory(final["evaluated_source"]) == sources,
        "evaluated source inventory differs",
    )
    archive_receipt = _receipt(plan["source_archive"])
    _require(
        _receipt(final["evaluated_source_archive"]) == archive_receipt,
        "evaluated archive association differs",
    )
    verify_archive_members(
        BytesIO(bound_bytes(archive_receipt, ARCHIVE_LIMIT)),
        {name: row["sha256"] for name, row in sources.items()},
        max_bytes=ARCHIVE_LIMIT,
    )
    # The retained archive owns the numerical dependency graph. Rebind all
    # its production Python files, so an indirect scalar/trigonometry owner
    # cannot change while its immediate consumer keeps the old digest.
    for name, row in sources.items():
        if name.startswith("src/tnfr/") and name.endswith(".py"):
            bound_bytes(row)
    # Read and bind old accounting records without using their cached
    # scientific verdicts to decide the new observer or time-cover result.
    for name in (
        "execution-result.json",
        "prefix-audit.json",
        "trajectory/trajectory-result.json",
    ):
        record(artifacts[PRIOR_DIRECTORY + "/" + name])
    attempt = record(artifacts[PRIOR_DIRECTORY + "/trajectory/attempt.json"])
    source_record = record(artifacts[PRIOR_DIRECTORY + "/trajectory/source.json"])
    cells = tuple(record(artifacts[name]) for name in cell_names)
    admitted = admit_retained_records(attempt, source_record, cells)
    return replace(
        admitted,
        receipts=tuple(receipts[name] for name in sorted(receipts)),
        file_bound=True,
    )


@dataclass(frozen=True)
class ReplicaSeparationAssessment:
    evidence: RetainedReplicaTrajectory
    observations: tuple[ReplicaTargetObservation, ...]
    time_cover: ReplicaTimeCover
    status: str
    conditional_on_retained_enclosures: bool = True
    independent_predictive_experiment: bool = False
    new_integrations: int = 0
    empirical_validation: bool = False


def assess_retained_target(repository, *, final_state_receipt):
    """Apply the frozen detached criterion once to freshly admitted evidence."""
    evidence = read_retained_trajectory(
        repository, final_state_receipt=final_state_receipt
    )
    observations = tuple(observe_target(tube) for tube in evidence.tubes)
    cover = time_cover(tuple(item.status for item in observations))
    status = (
        "closed_half_unit_residence_excluded"
        if cover.residence_excluded
        else "residence_certified" if cover.positive_residence_windows else "unresolved"
    )
    return ReplicaSeparationAssessment(evidence, observations, cover, status)
