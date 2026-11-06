"""Static changed-law bounds around one retained validated formation transit.

The reader verifies immutable evidence and evaluates interval derivatives on
its inflated tubes. It never calls a trajectory producer, advances a graph or
installs either comparison law. The two parameter bounds are separate
single-law comparisons with the same reflected preparation, not a joint box.
"""

from __future__ import annotations

import ast
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import exp_unit_bounds
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, pi_interval
from ..physics import relational_transit as transit
from ..sdk.relational_reports import _project
from ..utils.io import json_loads
from .artifact_io import _require
from .artifact_io import decode_exact_tree as _decode
from .artifact_io import exact_record as _exact
from .artifact_io import read_bytes_bounded, sha256_bytes, sha256_file
from .artifact_io import verify_archive_members as _verify_archive

__all__ = (
    "RelationalFormationComparisonStep",
    "RelationalFormationLawBound",
    "RelationalFormationRobustnessEvidence",
    "RelationalFormationRobustnessAudit",
    "audit_relational_formation_robustness",
)

_REFERENCE_HASHES = (
    (
        "continuous-transit.audit.json",
        "ac3104c42cf9054968eac770fee5c1b1e174547a408875edac0b635fe04ca781",
    ),
    (
        "continuous-transit.audit.protocol.json",
        "d8913d6af0d40fd1c903eaf24be29fb8ebeb6d3be28d6415d4e8dafb9974db99",
    ),
    (
        "continuous-transit.audit.source.zip",
        "381212374968f32fe59751c752b4dc812844a3309e00f771704a72b31a154c93",
    ),
)
_RADIUS = Q(1, 1024)
_HORIZON = Q(32)
_STEP = Q(1, 8)
_COORDINATE_LIMITS = (Q(101, 100), Q(1, 4))
_RESULTANT_LIMITS = (Q(97, 100), Q(21, 25), Q(29, 1000))


def _read(path):
    return read_bytes_bounded(path, max_bytes=32 * 1024**2)


@dataclass(frozen=True)
class RelationalFormationComparisonStep:
    """Static derivative bound on one reference tube plus the fixed radius."""

    time: Q
    duration: Q
    comparison_matrix: tuple[tuple[Q, ...], ...]
    logarithmic_norm_upper_bound: Q


@dataclass(frozen=True)
class RelationalFormationLawBound:
    """One comparison coefficient, with the other comparison absent."""

    law: str
    parameter_upper_bound: Q
    forcing_factor_upper_bound: Q
    deviation_upper_bound: Q
    corridor_slack: Q


@dataclass(frozen=True)
class RelationalFormationRobustnessEvidence:
    """Exact margins and comparison factors; no alternate trajectory."""

    horizon: Q
    corridor_radius: Q
    coordinate_hull: tuple[I, ...]
    inflated_coordinate_hull: tuple[I, ...]
    reference_resultant_lower_bounds: tuple[Q, ...]
    inflated_resultant_lower_bounds: tuple[Q, ...]
    minimum_picard_margin: Q
    inflated_endpoint: tuple[I, ...]
    endpoint_storage: I
    endpoint_rectangle_margins: tuple[I, ...]
    endpoint_storage_margin: Q
    comparison_steps: tuple[RelationalFormationComparisonStep, ...]
    integrated_logarithmic_norm_upper_bound: Q
    exponential_upper_bound: Q
    law_bounds: tuple[RelationalFormationLawBound, ...]
    checks: tuple[tuple[str, bool], ...]
    audit_source_sha256: tuple[tuple[str, str], ...]
    method: str = "retained_Picard_corridor_static_Metzler_rowsum_Gronwall_v1"


@dataclass(frozen=True)
class RelationalFormationRobustnessAudit:
    """Conditional single-law robustness of the authenticated retained record.

    File identity is checked against a declared reference, not against the
    current producer. Public construction is not provenance authentication.
    ``admitted`` concerns the two named mathematical comparisons and supplied
    reflected support/preparation; it is not a physical coefficient estimate.
    """

    status: str
    reference_sha256: tuple[tuple[str, str], ...] = ()
    source_file_count: int | None = None
    evidence: RelationalFormationRobustnessEvidence | None = None
    unavailable_reasons: tuple[str, ...] = ()
    scope: tuple[str, ...] = (
        "same_exact_represented_reflected_initial_state_and_supplied_two_C5_two_bridge_support",
        "unit_held_capacity_e_equals_w_equals_half_beta_equals_one_no_forcing_or_events",
        "retained_reference_tubes_verified_without_trajectory_or_frozen_producer_replay",
        "static_reference_Jacobian_bounds_on_inflated_whole_time_tubes",
        "audit_source_sha256_records_selected_owners_not_a_complete_dependency_archive",
        "separate_Frel_plus_rho_Ng_OR_Frel_plus_eta_e_over_pi_N_h_g_squared_comparisons",
        "parameter_caps_do_not_authorize_simultaneous_rho_and_eta_terms",
        "reflected_Rplus_energy_below_seven_not_full_state_acute_sector_admission",
        "conditional_law_class_capture_requires_copy_reflection_invariance_and_coercive_loss",
        "no_physical_parameter_selection_optimal_radius_or_future_binary64_certificate",
        "does_not_rewrite_the_original_failed_finite_prediction",
    )

    @property
    def admitted(self) -> bool:
        return self.status == "admitted"

    def to_dict(self):
        return {
            "schema": "tnfr.relational-formation-robustness.v1",
            "report": _project(self),
        }


def _box(values):
    _require(len(values) == 4, "reference requires four reduced coordinates")
    return tuple(I(_exact(value["lo"]), _exact(value["hi"])) for value in values)


def _inflate(box):
    return tuple(I(value.lo - _RADIUS, value.hi + _RADIUS) for value in box)


def _verify_reference_flow(archive_path):
    """Compare formula syntax without importing or executing archived code."""
    name = "src/tnfr/physics/relational_transit.py"
    with zipfile.ZipFile(archive_path) as archive:
        old = ast.parse(archive.read(name).decode("utf-8"))
    current = ast.parse(Path(transit.__file__).read_text(encoding="utf-8"))
    audit = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for function in (
        "_regular_bounds",
        "_flow",
        "_sinc",
        "_comparison_matrix",
        "_storage",
    ):
        nodes = []
        for tree in (old, audit if function == "_comparison_matrix" else current):
            name = "_reference_comparison_matrix" if tree is audit else function
            nodes.append(
                next(
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name
                )
            )
        if function == "_comparison_matrix":
            # Qualifying the already-verified flow and renaming this frozen
            # oracle are the only permitted syntactic changes.
            nodes[1].name = function
            for node in ast.walk(nodes[1]):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "transit"
                    and node.func.attr == "_flow"
                ):
                    node.func = ast.Name(id="_flow", ctx=ast.Load())
        _require(
            ast.dump(nodes[0]) == ast.dump(nodes[1]),
            f"reference formula changed: {function}",
        )


# Audit-only oracle: its syntax must match the immutable four-coordinate
# reference. Production continues to use the shared comparison owner.
def _reference_comparison_matrix(tube, e, w, beta):
    columns = []
    for column in range(4):
        variables = tuple(
            Jet((value, I(int(index == column)))) for index, value in enumerate(tube)
        )
        columns.append(
            tuple(row.coeffs[1] for row in transit._flow(variables, e, w, beta))
        )
    return tuple(
        tuple(columns[j][i].hi if i == j else columns[j][i].abs_max for j in range(4))
        for i in range(4)
    )


def _source_hashes():
    from ..mathematics import _interval_taylor, _rational_interval, _validated_taylor
    from . import artifact_io

    paths = (
        Path(__file__),
        Path(transit.__file__),
        Path(_interval_taylor.__file__),
        Path(_rational_interval.__file__),
        Path(_validated_taylor.__file__),
        Path(artifact_io.__file__),
    )
    return tuple((path.name, sha256_file(path)) for path in paths)


def _bound_reference(record, protocol):
    _require(record["protocol"] == protocol, "embedded reference protocol differs")
    _require(
        record["continuous_capture_admitted"] is True, "reference capture unavailable"
    )
    _require(
        record["original_finite_prediction_passed"] is False,
        "original finite verdict differs",
    )
    report = record["certificate"]["report"]
    _require(
        report["initial_winding_zero"] is True and report["target_sector"] == 1,
        "reference winding/target differs",
    )
    _require(
        report["horizon"] == report["validated_horizon"] == _HORIZON,
        "reference horizon differs",
    )
    _require(
        report["time_step"] == _STEP and len(report["steps"]) == 256,
        "reference step chain differs",
    )
    _require(
        report["status"] == "admitted"
        and not report["unavailable_reasons"]
        and report["failed_tube"] is None,
        "reference proof is incomplete",
    )
    _require(
        protocol["model"]
        == {
            "epi_weight": 0.5,
            "phase_weight": 0.5,
            "phase_domain": "positive_resultant",
            "storage_scale": 1.0,
        },
        "unsupported reference model",
    )
    _require(protocol["capacity"] == (1.0,) * 10, "unsupported reference capacities")

    initial = _box(report["initial_box"])
    _require(
        initial
        == (
            I(1 + Q(1, 1 << 54)),
            I(0),
            I(Q(protocol["initial_phase"][0])),
            I(Q(protocol["initial_phase"][4])),
        ),
        "reference initial coordinates differ",
    )
    preceding = initial
    tubes, margins, resultants, steps = [], [], [], []
    time = Q(0)
    for retained in report["steps"]:
        _require(
            retained["time"] == time and retained["duration"] == _STEP,
            "reference time chain differs",
        )
        tube, endpoint = _box(retained["tube"]), _box(retained["endpoint"])
        _require(
            all(
                value.subset_of(bound)
                for values in (preceding, endpoint)
                for value, bound in zip(values, tube)
            ),
            "reference box leaves its retained tube",
        )
        lower = tuple(map(_exact, retained["resultant_real_lower_bounds"]))
        margin = _exact(retained["picard_interior_margin"])
        _require(
            len(lower) == 3 and min(lower) > 0 and margin > 0,
            "reference whole-time admission unresolved",
        )
        tubes.append(tube)
        resultants.append(lower)
        margins.append(margin)
        preceding = endpoint
        time += _STEP
    _require(
        time == _HORIZON and preceding == _box(report["endpoint"]),
        "reference endpoint chain differs",
    )
    hull = tuple(
        I(min(tube[i].lo for tube in tubes), max(tube[i].hi for tube in tubes))
        for i in range(4)
    )
    inflated_hull = _inflate(hull)
    reference_lower = tuple(min(row[i] for row in resultants) for i in range(3))
    # C0 has phase-gradient l1 norm <=4, C4 <=3 and C3 <=2.
    inflated_lower = tuple(
        value - multiplier * _RADIUS
        for value, multiplier in zip(reference_lower, (4, 3, 2))
    )
    endpoint = _inflate(preceding)
    storage = transit._storage(endpoint, Q(1))
    rectangle = transit._positive_margins(endpoint)
    pi_lower = pi_interval().lo
    a, b = inflated_hull[2:]
    checks = [
        (
            "inflated_form_corridor",
            all(
                value.abs_max < limit
                for value, limit in zip(inflated_hull, _COORDINATE_LIMITS)
            ),
        ),
        (
            "inflated_positive_resultants",
            all(
                value > limit for value, limit in zip(inflated_lower, _RESULTANT_LIMITS)
            ),
        ),
        (
            "inflated_phase_branch",
            a.lo > 0 and a.hi < pi_lower and (a / 2 - b).abs_max < pi_lower / 2,
        ),
        ("inflated_endpoint_in_Rplus", all(value.lo > 0 for value in rectangle)),
        ("inflated_endpoint_storage_below_seven", storage.hi < 7),
        ("mathematical_pi_above_three", pi_lower > 3),
    ]
    if not all(value for _, value in checks):
        raise ArithmeticError("inflated corridor or endpoint premises unresolved")
    for index, tube in enumerate(tubes):
        matrix = transit._comparison_matrix(_inflate(tube), Q(1, 2), Q(1, 2), Q(1))
        _require(
            matrix
            == _reference_comparison_matrix(_inflate(tube), Q(1, 2), Q(1, 2), Q(1)),
            "delegated comparison differs from reference formula",
        )
        _require(
            all(matrix[i][j] >= 0 for i in range(4) for j in range(4) if i != j),
            "comparison matrix is not Metzler",
        )
        mu = max(Q(0), max(sum(row, Q(0)) for row in matrix))
        steps.append(
            RelationalFormationComparisonStep(index * _STEP, _STEP, matrix, mu)
        )
    integrated = sum(
        (step.duration * step.logarithmic_norm_upper_bound for step in steps), Q(0)
    )
    # exp(integrated)<exp(9)<(11/4)^9<2^14, proved with the shared
    # exact exponential enclosure rather than a floating exponential.
    _, e_upper = exp_unit_bounds(Q(1))
    exponential = Q(1 << 14)
    checks.extend(
        (
            ("integrated_logarithmic_norm_below_nine", integrated < 9),
            (
                "exponential_rational_majorant",
                e_upper < Q(11, 4) and Q(11, 4) ** 9 < exponential,
            ),
        )
    )
    bounds = []

    # On the positive-real chart |g|<1/2 and |h(u)|<=|u|. With e=1/2
    # and pi>3 the eta additions are bounded by eta*|q|/72 and
    # eta*|r|/48, both below eta/64. Both form rows are unchanged.
    checks.append(
        (
            "nonlinear_forcing_majorant",
            max(_COORDINATE_LIMITS[0] / 72, _COORDINATE_LIMITS[1] / 48) < Q(1, 64),
        )
    )
    for law, cap, factor in (
        ("rho_phase_loss_only", Q(1, 1 << 29), Q(1, 2)),
        ("eta_nonlinear_form_only", Q(1, 1 << 24), Q(1, 64)),
    ):
        deviation = _HORIZON * exponential * cap * factor
        bounds.append(
            RelationalFormationLawBound(
                law, cap, factor, deviation, _RADIUS - deviation
            )
        )
    checks.append(
        (
            "separate_law_errors_below_corridor",
            all(
                bound.deviation_upper_bound <= _RADIUS / 2 and bound.corridor_slack > 0
                for bound in bounds
            ),
        )
    )
    return RelationalFormationRobustnessEvidence(
        _HORIZON,
        _RADIUS,
        hull,
        inflated_hull,
        reference_lower,
        inflated_lower,
        min(margins),
        endpoint,
        storage,
        rectangle,
        7 - storage.hi,
        tuple(steps),
        integrated,
        exponential,
        tuple(bounds),
        tuple(checks),
        _source_hashes(),
    )


def audit_relational_formation_robustness(
    reference_path,
) -> RelationalFormationRobustnessAudit:
    """Bound two separate phase-law changes from the retained reference proof.

    Reads the response and sibling ``.protocol.json`` and ``.source.zip``.
    Known file hashes and archived source membership are required before any
    derivative calculation. The immutable reference proves its own ideal ODE;
    this audit only compares perturbed fields within its inflated corridor.
    Missing data or unresolved new bounds are unavailable. Changed records,
    invalid provenance or incompatible reference formulas are inconsistent.
    """
    path = Path(reference_path)
    paths = (path, path.with_suffix(".protocol.json"), path.with_suffix(".source.zip"))
    missing = tuple(str(item) for item in paths if not item.is_file())
    if missing:
        return RelationalFormationRobustnessAudit(
            "unavailable", unavailable_reasons=("missing_artifacts", *missing)
        )
    try:
        raw = tuple(_read(item) for item in paths)
        for data, (name, digest) in zip(raw, _REFERENCE_HASHES):
            _require(
                sha256_bytes(data) == digest,
                f"immutable reference changed: {name}",
            )
        record, protocol = (_decode(json_loads(data)) for data in raw[:2])
        _verify_archive(paths[2], protocol["source_sha256"])
        _verify_reference_flow(paths[2])
        evidence = _bound_reference(record, protocol)
        reasons = tuple(name for name, admitted in evidence.checks if not admitted)
        return RelationalFormationRobustnessAudit(
            "unavailable" if reasons else "admitted",
            _REFERENCE_HASHES,
            len(protocol["source_sha256"]),
            evidence,
            reasons,
        )
    except (ArithmeticError, RuntimeError) as error:
        return RelationalFormationRobustnessAudit(
            "unavailable", _REFERENCE_HASHES, unavailable_reasons=(str(error),)
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KeyError,
        StopIteration,
        SyntaxError,
        zipfile.BadZipFile,
    ) as error:
        return RelationalFormationRobustnessAudit(
            "inconsistent", unavailable_reasons=(str(error),)
        )
