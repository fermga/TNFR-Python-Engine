"""Read-only admission of the retained two-port capture image for a new probe.

Primitive preparation, target brackets, reference tubes and endpoint arithmetic
are re-admitted. Cached target bounds and verdicts do not supply these values.
Compact steps do not retain every Taylor coefficient: the archived validated
execution remains an explicit premise, not something hashes authenticate.
No capture producer, root search, trajectory, or graph mutation is executed.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sin, sqrt
from ..physics.phase_cycle_geometry import _derive
from ..physics.relational_sine_two_port_compatibility import (
    _affine_geometry,
    _current_factorization,
)
from ..utils.io import json_loads
from .relational_acquisition import _exact, _require, _verify_archive

__all__ = (
    "SineTwoPortHandoffAudit",
    "audit_sine_two_port_capture_record",
    "audit_sine_two_port_capture_handoff",
)

_STEM = "two-port-capture-v1"
_BASE = "29e4c02339f6246d003a7fbfbdf7f37c8a2c1277"
_NODES = tuple(range(18))
_EDGES = tuple(
    sorted(
        {tuple(sorted((s + i, s + (i + 1) % 9))) for s in (0, 9) for i in range(9)}
        | {(0, 9), (1, 10)}
    )
)
_REPS = (0, 2, 3, 4, 9, 11, 12, 13)
_WEIGHTS = (Q(6), Q(4), Q(4), Q(4)) * 2
_LIMIT = 64 * 1024**2


def _rows(value, size, name):
    _require(
        isinstance(value, (tuple, list)) and len(value) == size,
        f"invalid {name} length",
    )
    return tuple(value)


def _integers(value, expected, name):
    values = _rows(value, len(expected), name)
    _require(
        all(type(v) is int for v in values) and values == tuple(expected),
        f"invalid {name}",
    )


def _bounds(value):
    _require(
        isinstance(value, dict) and set(value) == {"lo", "hi"},
        "invalid interval record",
    )
    lower, upper = _exact(value["lo"]), _exact(value["hi"])
    _require(lower <= upper, "reversed interval record")
    return lower, upper


def _source(report, protocol):
    source = report["preparation"]
    inputs = protocol["inputs"]
    _require(
        source["law"] == "normalized_sine_reciprocal_exchange", "different source law"
    )
    _require(
        source["clock"] == "sigma=gamma^2*tau; tau=e*t; gamma=1/(1023*pi)",
        "different source clock",
    )
    model = protocol["complete_model"]
    declarations = {
        "law": "normalized_sine_reciprocal_exchange",
        "rows_in_tau": "x'=-K L x+gamma K S(theta); theta'=gamma K L x",
        "gamma": "1/(1023*pi)",
        "eta": "gamma^2",
        "e": "1023/1024",
        "w": "1/1024",
        "clocks": "tau=e*t; sigma=eta*tau; every companion row uses these same clocks",
        "capacity": "held unit at all eighteen nodes",
        "forcing": "absent",
        "events": "none during the assessed fixed-support evolution",
        "pressure": "complete sine forcing refreshed at all reference interval/jet stages, with the separate proved full-law comparison",
    }
    for name, expected in declarations.items():
        _require(model[name] == expected, f"different protocol model {name}")
    _integers(
        (model["beta"], protocol["support"]["degree_mass"]),
        (1, 40),
        "protocol beta/mass",
    )
    _require(protocol["support"]["weights"] == "unit", "different support weights")
    _integers(
        (
            protocol["handoff"]["additional_slow_duration"],
            protocol["handoff"]["full_endpoint_slow_time"],
        ),
        (1, 1025),
        "protocol handoff clock",
    )
    _require(
        _exact(report["analytic_tail_slow_duration"]) == 1
        and _exact(report["full_slow_horizon"]) == 1025
        and _exact(report["full_horizon_pi_squared_coefficient"]) == 1025 * 1023 * 1024,
        "different retained handoff clock",
    )
    for name, expected in (
        ("form_error_radius", Q(1, 65536)),
        ("phase_error_radius", Q(1, 65536)),
    ):
        _require(
            _exact(source[name]) == _exact(inputs[name]) == expected,
            f"different source {name}",
        )
    for name, expected in (("reference_duration", Q(1024)), ("time_step", Q(1, 4))):
        _require(
            _exact(report[name]) == _exact(inputs[name]) == expected,
            f"different reference {name}",
        )
    for name, expected in (("order", 8), ("max_steps", 4096)):
        _integers((report[name], inputs[name]), (expected, expected), name)
    _integers(source["geometry"]["nodes"], _NODES, "source nodes")
    _integers(protocol["support"]["nodes"], _NODES, "protocol nodes")
    for rows in (source["geometry"]["edges"], protocol["support"]["edges"]):
        for row, edge in zip(_rows(rows, 20, "support"), _EDGES):
            _integers(row, edge, "support edge")
    geometry = _derive(_NODES, _EDGES)
    degrees, _, nodal, edge_affine, offsets = _affine_geometry((2, 1), geometry)
    _integers(source["degrees"], degrees, "degrees")
    phases = tuple(row[0] + row[1] * Q(2, 9) + row[2] * Q(1, 9) for row in nodal)
    edges = tuple(phases[j] - phases[i] - k for (i, j), k in zip(_EDGES, offsets))
    _require(
        tuple(map(_exact, _rows(source["nominal_phase_turns"], 18, "nominal phases")))
        == phases,
        "different nominal phase preparation",
    )
    _require(
        tuple(map(_exact, _rows(source["nominal_edge_turns"], 20, "nominal edges")))
        == edges,
        "nominal edge gaps differ from primitive phases",
    )
    _integers(source["edge_integer_offsets"], offsets, "edge branches")
    _integers(source["named_cycle_periods"], (2, 1, 0), "cycle periods")
    for name, expected in (("nominal_epi", Q(0)), ("capacity", Q(1))):
        _require(
            all(_exact(v) == expected for v in _rows(source[name], 18, name)),
            f"different {name}",
        )
    for name, expected in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        _require(
            exact_or_represented_real(source["reference_model"][name], name)
            == expected,
            f"different model {name}",
        )
    _require(
        source["reference_model"]["phase_domain"] == "regular",
        "different reference model domain",
    )
    _require(
        report["arithmetic_method"] == INTERVAL_METHOD, "different interval method"
    )
    laplacian = tuple(
        tuple(
            Q(degrees[i] if i == j else -int(tuple(sorted((i, j))) in _EDGES))
            for j in _NODES
        )
        for i in _NODES
    )
    lower = tuple(
        tuple(
            laplacian[i][j]
            - Q(1, 90) * (degrees[i] * int(i == j) - Q(degrees[i] * degrees[j], 40))
            for j in _NODES
        )
        for i in _NODES
    )
    upper = tuple(
        tuple(2 * degrees[i] * int(i == j) - laplacian[i][j] for j in _NODES)
        for i in _NODES
    )
    _require(
        exact_symmetric_semidefinite(lower) and exact_symmetric_semidefinite(upper),
        "normalized gap proof failed",
    )
    return geometry, phases, edges, nodal, edge_affine


def _target(report, geometry, nodal, edge_affine):
    """Rebuild the correlated target from admitted root endpoints, not flags."""
    target = report["target"]
    _require(isinstance(target, dict), "missing retained target evidence")
    _integers(target["classes"], (2, 1), "target classes")
    pi = pi_interval()

    def h(k, value):
        return sin(2 * pi * ((k - I.coerce(value)) / 8)) - sin(2 * pi * value)

    outer = target["canonical_root_turn_bracket"]
    a0, a1 = _exact(outer["lower"]), _exact(outer["upper"])
    _require(Q(1, 9) <= a0 < a1 <= Q(2, 9), "invalid target outer bracket")
    _integers((outer["refinements"],), (32,), "outer refinements")
    _require(a1 - a0 == Q(1, 9 * 2**32), "different target outer budget")
    inner_bounds = []
    for index, (a, inner) in enumerate(
        zip(
            (a0, a1),
            _rows(
                target["inner_root_brackets_at_outer_endpoints"], 2, "inner brackets"
            ),
        )
    ):
        c0, c1 = _exact(inner["lower"]), _exact(inner["upper"])
        _require(Q(1, 9) <= c0 < c1 <= Q(2, 9), "invalid target inner bracket")
        _integers((inner["refinements"],), (64,), "inner refinements")
        _require(c1 - c0 == Q(1, 9 * 2**64), "different target inner budget")
        _require(
            (h(1, c0) + h(2, a)).lo > 0 > (h(1, c1) + h(2, a)).hi,
            "target inner signs not certified",
        )
        residual = h(2, a) - sin(pi * (a - I(c0, c1)))
        _require(
            residual.lo > 0 if index == 0 else residual.hi < 0,
            "target outer signs not certified",
        )
        inner_bounds.append((c0, c1))
    short = I(a0, a1), I(inner_bounds[1][0], inner_bounds[0][1])

    def affine(row):
        return I(row[0]) + row[1] * short[0] + row[2] * short[1]

    phases = tuple(2 * pi * affine(row) for row in nodal)
    margin = min((pi / 2 - abs(2 * pi * affine(row))).lo for row in edge_affine)
    _require(margin > Q(1, 8), "target margin does not admit the capture theorem")
    _current_factorization(geometry)
    return phases, margin


def _reference(report, edge_turns):
    """Admit the compact chain without claiming a replay of Taylor remainders."""
    folded = report["folded_reference"]
    permutation = tuple(9 * (i // 9) + (1 - i % 9) % 9 for i in _NODES)
    _integers(folded["representatives"], _REPS, "folded representatives")
    _integers(folded["permutation"], permutation, "folded permutation")
    matrix = tuple(
        tuple(Q(int(i == r) - int(i == permutation[r])) for r in _REPS) for i in _NODES
    )
    admitted_matrix = tuple(
        tuple(map(_exact, _rows(row, 8, "reconstruction row")))
        for row in _rows(folded["reconstruction_matrix"], 18, "reconstruction")
    )
    _require(admitted_matrix == matrix, "different reference reconstruction")
    metric = tuple(
        tuple(map(_exact, _rows(row, 8, "metric row")))
        for row in _rows(folded["metric"], 8, "metric")
    )
    _require(
        metric
        == tuple(
            tuple(weight * int(i == j) for j in range(8))
            for i, weight in enumerate(_WEIGHTS)
        ),
        "different inherited metric",
    )
    steps = _rows(report["reference_steps"], 4096, "complete reference")
    center, radius, elapsed, minimum = (Q(0),) * 8, Q(0), Q(0), None
    pi = pi_interval()
    for step in steps:
        _require(
            _exact(step["time"]) == elapsed and _exact(step["duration"]) == Q(1, 4),
            "reference time chain differs",
        )
        _require(
            tuple(map(_exact, _rows(step["initial_center"], 8, "initial center")))
            == center
            and _exact(step["initial_radius"]) == radius,
            "reference state chain differs",
        )
        tube = tuple(map(_bounds, _rows(step["tube"], 8, "tube")))
        for value, weight, (lo, hi) in zip(center, _WEIGHTS, tube):
            _require(
                lo <= value <= hi
                and weight * (value - lo) ** 2 >= radius**2
                and weight * (hi - value) ** 2 >= radius**2,
                "tube does not contain the retained initial ball",
            )
        _require(
            _exact(step["picard_interior_margin"]) > 0,
            "nonpositive Picard inclusion evidence",
        )
        recorded = tuple(
            map(_exact, _rows(step["domain_lower_bounds"], 20, "edge domains"))
        )
        for bound, (i, j), turn in zip(recorded, _EDGES, edge_turns):
            lo, hi = min(2 * pi.lo * turn, 2 * pi.hi * turn), max(
                2 * pi.lo * turn, 2 * pi.hi * turn
            )
            for coefficient, (left, right) in zip(
                (b - a for a, b in zip(matrix[i], matrix[j])), tube
            ):
                lo += min(coefficient * left, coefficient * right)
                hi += max(coefficient * left, coefficient * right)
            _require(
                0 < bound <= pi.lo / 2 - max(abs(lo), abs(hi)) - Q(1, 2048),
                "reference edge margin not justified by its tube",
            )
        margin = min(recorded) + Q(1, 2048)
        minimum = margin if minimum is None else min(minimum, margin)
        local = _exact(step["local_metric_error_upper_bound"])
        _require(local >= 0, "negative retained local error")
        expected = I(radius + local).hi
        radius = _exact(step["endpoint_radius"])
        _require(radius == expected, "retained metric radius recurrence differs")
        center = tuple(
            map(_exact, _rows(step["endpoint_center"], 8, "endpoint center"))
        )
        elapsed += Q(1, 4)
    _require(
        _exact(report["validated_reference_duration"]) == elapsed == 1024,
        "incomplete reference horizon",
    )
    _require(
        tuple(map(_exact, report["reference_endpoint_center"])) == center
        and _exact(report["reference_endpoint_radius"]) == radius,
        "reference endpoint association differs",
    )
    _require(
        report["failed_tube"] is None, "failed tube cannot supply a complete handoff"
    )
    return center, radius, minimum


@dataclass(frozen=True)
class SineTwoPortHandoffAudit:
    """Reconstructed handoff conditional on retained validated execution.

    This is a content audit with rebuilt bounds, not a fresh trajectory or
    independent authentication that an archived program produced a record.
    """

    reference_target_distance_upper_bound: Q
    target_acute_margin_lower_bound: Q
    reference_minimum_acute_margin: Q
    phase_error_upper_bound: Q
    endpoint_phase_radius: Q
    endpoint_form_radius: Q
    endpoint_excess_storage_upper_bound: Q
    capture_storage_margin: Q
    admitted_probe_phase_radius: Q = Q(1, 1024)
    admitted_probe_form_radius: Q = Q(1, 8192)
    full_slow_handoff_time: Q = Q(1025)
    original_handoff_time_pi_squared_coefficient: Q = Q(1025 * 1023 * 1024)
    reference_steps_checked: int = 4096
    numerical_execution_replayed: bool = False
    provenance_authenticated: bool = False
    conditional_premises: tuple[str, ...] = (
        "retained_metric_steps_were_produced_under_the_archived_validated_field_and_remainder_contract",
        "recorded_Picard_inclusions_local_remainders_and_endpoint_centers_are_valid_shared_kernel_evidence",
        "compact_record_does_not_retain_every_Taylor_coefficient_or_independently_reprove_local_errors",
        "content_hashes_and_source_revision_support_recoverability_not_independent_execution_authentication",
        "complete_sine_law_fixed_support_capacity_clocks_and_original_36_coordinate_source_family",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {
            "schema": "tnfr.sine-two-port-handoff-audit.v1",
            "report": _project(self),
        }


def audit_sine_two_port_capture_record(record, *, protocol):
    """Re-admit a complete fixed v1 record; malformed or incomplete input rejects.

    Cached summary bounds and flags are not premises. Retained step inclusion,
    local-error and endpoint-center validity are explicit execution premises.
    The returned endpoint values
    are recomputed from target root signs, the metric chain and analytic tail.
    The declared protocol is fixed; this is not an arbitrary capture reader.
    Mapping-level admission alone checks no source/archive association.
    """
    _require(
        isinstance(record, dict) and isinstance(protocol, dict),
        "record and protocol must be mappings",
    )
    _require(
        record["schema"] == "tnfr.sine-two-port-capture.v1"
        and protocol["schema"] == "tnfr.sine-two-port-capture-protocol.v1",
        "unsupported capture schema",
    )
    report = record["report"]
    geometry, nominal, edge_turns, nodal, edge_affine = _source(report, protocol)
    target, target_margin = _target(report, geometry, nodal, edge_affine)
    center, radius, minimum = _reference(report, edge_turns)
    pi = pi_interval()
    displacements = tuple(target[i] - 2 * pi * nominal[i] for i in _REPS)
    distance = (
        sqrt(
            sum(
                (
                    weight * (I(value) - bound) ** 2
                    for weight, value, bound in zip(_WEIGHTS, center, displacements)
                ),
                I(0),
            )
        ).hi
        + radius
    )
    _require(
        distance <= Q(1, 2048),
        "reference endpoint does not reach the admitted target ball",
    )
    rx = rt = Q(1, 65536)
    _require(Q(10, 81) + 40 * rt + 40 * rx**2 < Q(1, 8), "source energy premise failed")
    z0 = 7 * rx / 3069
    joint = 7 * rt + z0 + Q(24, 100000)
    z = (z0 + Q(1, 100000) * (Q(5, 12) + 2 * joint)) / (1 - Q(1, 50000))
    error = joint + z
    _require(error < Q(1, 2048), "full-family comparison margin failed")
    phase = distance + error
    form = 3216 * (z / 2**512 + Q(1, 100000) * (Q(1, 1024) + 2 * error))
    energy = phase**2 + form**2
    margin = Q(1, 648000) - energy
    _require(
        phase < Q(1, 1024) and form < Q(1, 8192) and margin > 0,
        "complete source handoff bounds unavailable",
    )
    return SineTwoPortHandoffAudit(
        distance, target_margin, minimum, error, phase, form, energy, margin
    )


def audit_sine_two_port_capture_handoff(evidence_directory):
    """Admit the fixed retained artifacts and rebuild their source-to-probe bounds.

    Sizes, hashes, exact archive inventory and protocol bytes are checked.
    Earlier producers are never run. Hash consistency is not authentication.
    """
    directory = Path(evidence_directory)

    def read(name):
        path = directory / name
        _require(
            path.stat().st_size <= _LIMIT, "retained artifact exceeds the read budget"
        )
        return path.read_bytes()

    manifest = json_loads(read(f"{_STEM}.manifest.json"))
    expected = {
        f"{_STEM}{suffix}" for suffix in (".json", ".protocol.json", ".source.zip")
    }
    artifacts = _rows(manifest["artifacts"], 3, "artifact inventory")
    _require(
        {item["file"] for item in artifacts} == expected,
        "unexpected handoff artifact inventory",
    )
    content = {}
    for item in artifacts:
        data = read(item["file"])
        _require(
            type(item["bytes"]) is int
            and len(data) == item["bytes"]
            and hashlib.sha256(data).hexdigest() == item["sha256"],
            "retained artifact size or digest differs",
        )
        content[item["file"]] = data
    protocol = json_loads(content[f"{_STEM}.protocol.json"])
    with zipfile.ZipFile(directory / f"{_STEM}.source.zip") as archive:
        source_bytes = archive.read("source-manifest.json")
        source = json_loads(source_bytes)
        paths = [item["path"] for item in source["files"]]
        _require(
            len(paths) == len(set(paths)) and "source-manifest.json" not in paths,
            "invalid source inventory",
        )
        _require(
            set(paths)
            == {
                "src/tnfr/physics/relational_sine_two_port_capture.py",
                "src/tnfr/sdk/relational_reports.py",
                "theory/nodal/SINE_TWO_PORT_CAPTURE.md",
                f"docs/assets/sine_formed_classes/{_STEM}.protocol.json",
                "build/two-port-capture-freeze/evaluate_two_port_capture.py",
            },
            "different original capture source inventory",
        )
        digests = {item["path"]: item["sha256"] for item in source["files"]}
        digests["source-manifest.json"] = hashlib.sha256(source_bytes).hexdigest()
        _verify_archive(directory / f"{_STEM}.source.zip", digests)
        for item in source["files"]:
            _require(
                type(item["bytes"]) is int
                and len(archive.read(item["path"])) == item["bytes"],
                "source byte count differs",
            )
        _require(
            archive.read(f"docs/assets/sine_formed_classes/{_STEM}.protocol.json")
            == content[f"{_STEM}.protocol.json"],
            "archived protocol differs",
        )
    base = source["source_base_commit"]
    _require(
        isinstance(base, str)
        and re.fullmatch(r"[0-9a-f]{40}", base) is not None
        and base
        == manifest["source_base_commit"]
        == protocol["source_base_commit"]
        == _BASE,
        "source base association differs",
    )
    overlays = (
        "src/tnfr/physics/relational_sine_two_port_capture.py",
        "src/tnfr/sdk/relational_reports.py",
    )
    _require(
        tuple(source["runtime_overlays"])
        == tuple(manifest["runtime_overlays"])
        == overlays
        and set(overlays) == {path for path in paths if path.startswith("src/")},
        "runtime source overlays differ",
    )
    _require(
        protocol["source_overlay_archive"] == f"{_STEM}.source.zip",
        "source archive association differs",
    )
    return audit_sine_two_port_capture_record(
        json_loads(content[f"{_STEM}.json"]), protocol=protocol
    )
