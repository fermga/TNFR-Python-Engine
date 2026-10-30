"""Source-aware reference components for the formed two-port entry protocol.

The reference is an auxiliary eighteen-phase gradient flow. The actual family
retains all thirty-six form/phase coordinates through the comparison in
SINE_TWO_PORT_CAPTURE, equations M1--M9. This module takes no integration step,
launches no producer and selects no endpoint. Importing it evaluates no field,
trigonometric sign or target. Consumers must admit the complete acute prefix,
fixed execution policy and producing source separately.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path

from .._exact_time import exact_or_represented_real
from ..mathematics._interval_taylor import Jet
from ..mathematics._interval_taylor import sin as jet_sin
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sin
from ..mathematics._validated_metric import _metric_error_upper
from ..physics.phase_cycle_geometry import _derive
from ..physics.relational_sine_two_port_compatibility import _affine_geometry
from ..utils.io import json_loads
from . import sine_two_port_handoff
from .artifact_io import (
    exact_record,
    read_bytes_bounded,
    sha256_bytes,
    verify_archive_members,
)

__all__ = (
    "FormedTwoPortReference",
    "FormedTwoPortTarget",
    "FormedTwoPortBounds",
    "FormedTwoPortEndpoint",
    "build_formed_two_port_reference",
    "admit_formed_two_port_target",
    "formed_two_port_comparison_bounds",
    "admit_formed_two_port_endpoint",
)

_NODES = tuple(range(18))
_ORIENTED_EDGES = tuple(
    (start + j, start + (j + 1) % 9) for start in (0, 9) for j in range(9)
) + ((4, 13), (5, 14))
_DEGREES = tuple(3 if i in (4, 5, 13, 14) else 2 for i in _NODES)
_METRIC = tuple(
    tuple(Q(degree if i == j else 0) for j in _NODES)
    for i, degree in enumerate(_DEGREES)
)
_CANONICAL_EDGES = tuple(
    sorted({tuple(sorted(edge)) for edge in _ORIENTED_EDGES[:18]} | {(0, 9), (1, 10)})
)
_RHO = Q(1, 2048)
_PHI = Q(1, 1000)
_STEM = "two-port-compatibility-v1"
_BASE = "3a313196640bfd99d2164d8c2c2e5c89cb504c50"
_OVERLAYS = (
    "src/tnfr/physics/relational_sine_two_port_compatibility.py",
    "src/tnfr/sdk/relational_reports.py",
)
_ARCHIVED_PROTOCOL = f"docs/assets/sine_formed_classes/{_STEM}.protocol.json"
_SOURCE_FILES = frozenset(
    (
        *_OVERLAYS,
        "theory/nodal/SINE_TWO_PORT_COMPATIBILITY.md",
        _ARCHIVED_PROTOCOL,
        "build/two-port-compatibility-freeze/evaluate_two_port_compatibility.py",
    )
)
_SMALL_LIMIT = 128 * 1024
_REPORT_LIMIT = 2 * 1024**2
_ARCHIVE_LIMIT = 8 * 1024**2
_EXPANDED_LIMIT = 16 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _row(values, length, name):
    _require(
        isinstance(values, (tuple, list)) and len(values) == length,
        f"{name} requires exactly {length} entries",
    )
    return tuple(values)


def _integers(values, expected, name):
    values = _row(values, len(expected), name)
    _require(
        all(type(value) is int for value in values) and values == tuple(expected),
        f"different {name}",
    )


@dataclass(frozen=True)
class FormedTwoPortReference:
    """Fixed affine reference, with full degree-metric error coordinates.

    ``domain`` returns twenty *slacks above rho*, not raw acute margins.
    The kernel must require every slack strictly positive over its whole tube.
    Nominal phases use the one joined zero-mean gauge; actual memberwise means
    are restored in the analytical comparison, never by component resets.
    """

    pi: I
    nominal_pi_coefficients: tuple[Q, ...]
    nominal_offsets: tuple[Q, ...]
    nominal_phases: tuple[I, ...]
    base_angles: tuple[I, ...]
    oriented_edges: tuple[tuple[int, int], ...] = _ORIENTED_EDGES
    degrees: tuple[int, ...] = _DEGREES
    metric: tuple[tuple[Q, ...], ...] = _METRIC
    reference_margin: Q = _RHO
    initial_center: tuple[Q, ...] = (Q(0),) * 18
    initial_radius: Q = Q(0)

    def angles(self, values):
        """Retain the fixed cycle lifts for interval or equal-order Jet rows."""
        values = _row(values, 18, "reference state")
        is_jet = isinstance(values[0], Jet)
        if is_jet:
            order = values[0].order
            _require(
                all(
                    isinstance(value, Jet) and value.order == order for value in values
                ),
                "reference Jets require one common order",
            )
            bases = tuple(Jet.constant(value, order) for value in self.base_angles)
        else:
            _require(not any(isinstance(value, Jet) for value in values), "mixed Jets")
            values = tuple(I.coerce(value) for value in values)
            bases = self.base_angles
        return tuple(
            base + values[right] - values[left]
            for base, (left, right) in zip(bases, self.oriented_edges, strict=True)
        )

    def field(self, values):
        """Compute each edge current once and scatter using both live degrees."""
        angles = self.angles(values)
        is_jet = isinstance(angles[0], Jet)
        sine = jet_sin if is_jet else sin
        zero = Jet.constant(0, angles[0].order) if is_jet else I(0)
        rates = [zero for _ in _NODES]
        for (left, right), angle in zip(self.oriented_edges, angles, strict=True):
            current = sine(angle)
            rates[left] = rates[left] + current / self.degrees[left]
            rates[right] = rates[right] - current / self.degrees[right]
        return tuple(rates)

    def domain(self, values):
        """Return ``(pi/2-|gap|).lower-rho`` for every whole-tube edge."""
        values = _row(values, 18, "reference domain")
        _require(
            not any(isinstance(value, Jet) for value in values),
            "domain needs intervals",
        )
        return tuple(
            (self.pi / 2 - abs(angle)).lo - self.reference_margin
            for angle in self.angles(values)
        )


def build_formed_two_port_reference():
    """Build the fixed original-label reference; do not evaluate a field."""
    pi = pi_interval()
    # m0=phi/2+pi/60. No independent component recentering is performed.
    coefficients = tuple(
        Q((4 if i < 9 else 2) * (i % 9 - 4), 9) - Q(1, 60) for i in _NODES
    )
    offsets = tuple(-_PHI / 2 if i < 9 else _PHI / 2 for i in _NODES)
    nominal = tuple(
        coefficient * pi + offset
        for coefficient, offset in zip(coefficients, offsets, strict=True)
    )
    bases = (4 * pi / 9,) * 9 + (2 * pi / 9,) * 9 + (I(_PHI), _PHI - 2 * pi / 9)
    return FormedTwoPortReference(pi, coefficients, offsets, nominal, bases)


@dataclass(frozen=True)
class FormedTwoPortTarget:
    """One correlated critical target, enclosed in the new reference chart.

    Marginal intervals are enclosures of that target, not a box of equilibria.
    Receipts associate consumed bytes; they do not authenticate chronology.
    """

    target_phases: tuple[I, ...]
    target_deviations: tuple[I, ...]
    target_turn_affine_coefficients: tuple[tuple[Q, ...], ...]
    acute_margin_lower_bound: Q
    evidence_receipts: tuple[dict, ...]
    source_manifest_sha256: str
    source_base_commit: str


def _evidence_label(directory, name):
    path = directory / name
    if directory.parts[-3:] == ("docs", "assets", "sine_formed_classes"):
        return path.relative_to(directory.parents[2]).as_posix()
    return path.as_posix()


def _read_target_evidence(directory):
    """Verify a bounded primary report/protocol/archive from the same read bytes."""
    directory = Path(directory).resolve()
    receipts = []

    def read(name, limit):
        data = read_bytes_bounded(directory / name, max_bytes=limit)
        receipts.append(
            {
                "path": _evidence_label(directory, name),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
        return data

    manifest = json_loads(read(f"{_STEM}.manifest.json", _SMALL_LIMIT))
    _require(
        manifest["schema"] == "tnfr.sine-two-port-compatibility-evidence.v1",
        "different target evidence schema",
    )
    expected = {
        f"{_STEM}{suffix}" for suffix in (".json", ".protocol.json", ".source.zip")
    }
    rows = _row(manifest["artifacts"], 3, "target artifact inventory")
    _require({item["file"] for item in rows} == expected, "different target artifacts")
    content = {}
    for item in rows:
        name = item["file"]
        limit = _ARCHIVE_LIMIT if name.endswith(".zip") else _REPORT_LIMIT
        if name.endswith(".protocol.json"):
            limit = _SMALL_LIMIT
        data = read(name, limit)
        _require(
            type(item["bytes"]) is int
            and item["bytes"] == len(data)
            and item["sha256"] == sha256_bytes(data),
            "target artifact size or digest differs",
        )
        content[name] = data
    protocol = json_loads(content[f"{_STEM}.protocol.json"])
    archive_bytes = content[f"{_STEM}.source.zip"]
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        info = archive.getinfo("source-manifest.json")
        _require(info.file_size <= _SMALL_LIMIT, "source manifest exceeds byte budget")
        with archive.open(info) as stream:
            source_bytes = stream.read(_SMALL_LIMIT + 1)
        _require(
            len(source_bytes) <= _SMALL_LIMIT, "source manifest exceeds byte budget"
        )
        source = json_loads(source_bytes)
        files = _row(source["files"], 5, "target archived inventory")
        names = tuple(item["path"] for item in files)
        _require(
            len(set(names)) == 5 and set(names) == _SOURCE_FILES,
            "different target archive inventory",
        )
        hashes = {item["path"]: item["sha256"] for item in files}
        hashes["source-manifest.json"] = sha256_bytes(source_bytes)
        verify_archive_members(
            io.BytesIO(archive_bytes), hashes, max_bytes=_EXPANDED_LIMIT
        )
        for item in files:
            _require(
                type(item["bytes"]) is int
                and item["bytes"] == archive.getinfo(item["path"]).file_size,
                "target source byte count differs",
            )
        _require(
            archive.read(_ARCHIVED_PROTOCOL) == content[f"{_STEM}.protocol.json"],
            "archived target protocol differs",
        )
    _require(
        source["schema"] == "tnfr.sine-two-port-compatibility-source-snapshot.v1"
        and source["source_base_commit"]
        == manifest["source_base_commit"]
        == protocol["source_base_commit"]
        == _BASE,
        "target source base association differs",
    )
    _require(
        tuple(source["runtime_overlays"])
        == tuple(manifest["runtime_overlays"])
        == _OVERLAYS,
        "different target runtime overlays",
    )
    _require(
        protocol["source_overlay_archive"] == f"{_STEM}.source.zip",
        "different target archive association",
    )
    return (
        json_loads(content[f"{_STEM}.json"]),
        protocol,
        tuple(receipts),
        sha256_bytes(source_bytes),
    )


def _admit_primary_primitives(record, protocol):
    """Reject different source primitives; ignore cached target verdicts/bounds."""
    _require(
        record["schema"] == "tnfr.sine-two-port-compatibility.v1",
        "different target report schema",
    )
    _require(
        protocol["schema"] == "tnfr.sine-two-port-compatibility-protocol.v1",
        "different target protocol schema",
    )
    report = record["report"]
    for data in (report, protocol["inputs"]):
        _integers(data["classes"], (2, 1), "target classes")
        _integers(
            (data["outer_refinements"], data["inner_refinements"]),
            (32, 64),
            "target root budgets",
        )
    for data in (report["geometry"], protocol["support"]):
        _integers(data["nodes"], _NODES, "canonical target nodes")
        edges = _row(data["edges"], 20, "canonical target edges")
        for actual, expected in zip(edges, _CANONICAL_EDGES, strict=True):
            _integers(actual, expected, "canonical target edge")
    _require(
        report["law"] == "normalized_sine_reciprocal_exchange"
        and report["clock"] == "tau=e*t"
        and report["arithmetic_method"] == INTERVAL_METHOD,
        "different target law, clock or arithmetic",
    )
    model = report["reference_model"]
    for name, expected in (
        ("storage_scale", Q(1)),
        ("epi_weight", Q(1023, 1024)),
        ("phase_weight", Q(1, 1024)),
    ):
        _require(
            exact_or_represented_real(model[name], name) == expected,
            "different target model coefficient",
        )
    _require(model["phase_domain"] == "regular", "different target phase domain")
    _require(
        all(
            exact_record(value) == 1
            for value in _row(report["capacity"], 18, "target capacities")
        ),
        "different target capacities",
    )
    _require(
        protocol["support"]["weights"] == "unit", "different target support weights"
    )
    geometry = _derive(_NODES, _CANONICAL_EDGES)
    degrees, _, nodal, edge_affine, _ = _affine_geometry((2, 1), geometry)
    _integers(report["degrees"], degrees, "canonical target degrees")
    _integers(protocol["target"]["cycle_periods"], (2, 1, 0), "target cycle periods")
    return report, geometry, nodal, edge_affine


def admit_formed_two_port_target(reference, evidence_directory):
    """Re-admit retained primitive signs and transport their one critical target.

    No bisection, old capture assessor or old reference-chain reader is called.
    The existing private target-admission owner is deliberately reused. Source
    receipts use repository-relative labels for the standard evidence directory.
    """
    _require(isinstance(reference, FormedTwoPortReference), "reference type required")
    try:
        record, protocol, receipts, source_digest = _read_target_evidence(
            evidence_directory
        )
        report, geometry, nodal, edges = _admit_primary_primitives(record, protocol)
        phases, margin = sine_two_port_handoff._target(
            {"target": report}, geometry, nodal, edges
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid target evidence structure") from exc
    transported = [None] * 18
    affine = [None] * 18
    for old, (phase, coefficients) in enumerate(zip(phases, nodal, strict=True)):
        component, local = divmod(old, 9)
        new = 9 * component + (local + 4) % 9
        turns = (2 if component == 0 else 1) if local >= 5 else 0
        # Inverse of the source chart transport: remove old increasing-ramp
        # turns, then center the new full degree-weighted mean by +6*pi/5.
        transported[new] = phase + (Q(6, 5) - 2 * turns) * reference.pi
        affine[new] = (coefficients[0] + Q(3, 5) - turns, *coefficients[1:])
    target = tuple(transported)
    deviations = tuple(
        phase - nominal
        for phase, nominal in zip(target, reference.nominal_phases, strict=True)
    )
    return FormedTwoPortTarget(
        target, deviations, tuple(affine), margin, receipts, source_digest, _BASE
    )


@dataclass(frozen=True)
class FormedTwoPortBounds:
    """Exact M3--M9 design bounds, conditional on the complete reference prefix.

    The endpoint form bound is ``||P_D x||_D`` relative to the conserved
    uniform form, rather than the norm of the absolute form coordinates.
    """

    source_error_radius: Q
    source_relative_norm_upper_bound: Q
    source_excess_storage_upper_bound: Q
    source_reference_speed_upper_bound: Q
    initial_scaled_form_upper_bound: Q
    initial_mixed_phase_upper_bound: Q
    mixed_phase_error_upper_bound: Q
    scaled_form_upper_bound: Q
    phase_comparison_error_upper_bound: Q
    reference_margin: Q
    reference_horizon: Q
    full_slow_horizon: Q
    endpoint_scaled_form_upper_bound: Q
    endpoint_form_norm_upper_bound: Q
    endpoint_phase_distance_upper_bound: Q
    endpoint_excess_storage_upper_bound: Q
    capture_radius: Q
    capture_storage_boundary: Q
    absolute_tau_rational_part: Q
    absolute_tau_pi_squared_coefficient: Q
    absolute_t_rational_part: Q
    absolute_t_pi_squared_coefficient: Q
    form_norm_observation: str = "||P_D x||_D relative to the conserved uniform form"


def formed_two_port_comparison_bounds():
    """Rebuild rational comparison/tail bounds without trigonometry or roots."""
    epsilon = Q(1, 10**32)
    z0 = 3 * epsilon / 3069
    v0 = 3 * epsilon + z0
    v = v0 + Q(17, 50000)
    z = (z0 + Q(1, 100000) * (Q(3, 5) + 2 * v)) / (1 - Q(1, 50000))
    q = v + z
    _require(
        Q(49, 200) + Q(1, 500000) + 10 * epsilon**2 < Q(1, 4)
        and v < Q(7, 20000)
        and z < Q(1, 100000)
        and q < Q(9, 25000) < _RHO,
        "source-aware comparison design bounds unavailable",
    )
    zend = Q(1, 2**512 * 100000) + Q(1, 51200000)
    form = 3216 * zend
    phase = q + _RHO
    storage = form**2 + phase**2
    _require(
        form < Q(1, 8192)
        and phase < Q(1, 1024)
        and storage < Q(65, 67108864) < Q(1, 648000)
        and storage < Q(1, 12) ** 2,
        "source-aware final-unit bounds unavailable",
    )
    contact_tau = Q(100 + 10**13)
    return FormedTwoPortBounds(
        epsilon,
        3 * epsilon,
        Q(1, 4),
        Q(3, 5),
        z0,
        v0,
        v,
        z,
        q,
        _RHO,
        Q(1024),
        Q(1025),
        zend,
        form,
        phase,
        storage,
        Q(1, 12),
        Q(1, 648000),
        contact_tau,
        Q(1025 * 1023**2),
        contact_tau * Q(1024, 1023),
        Q(1025 * 1023 * 1024),
    )


@dataclass(frozen=True)
class FormedTwoPortEndpoint:
    """Endpoint/tail admission conditional on the independently admitted prefix.

    ``endpoint_form_norm_upper_bound`` bounds ``||P_D x||_D`` relative to
    the conserved uniform form; it does not bound the absolute form norm.
    """

    certified: bool
    reasons: tuple[str, ...]
    reference_distance_upper_bound: Q
    reference_distance_margin: Q
    endpoint_form_norm_upper_bound: Q
    endpoint_phase_distance_upper_bound: Q
    endpoint_excess_storage_upper_bound: Q
    form_margin: Q
    phase_margin: Q
    capture_radius_squared_margin: Q
    capture_storage_margin: Q
    conditional_on_acute_reference_prefix: bool = True
    form_norm_observation: str = "||P_D x||_D relative to the conserved uniform form"


def admit_formed_two_port_endpoint(reference, target, center, radius):
    """Bound full-metric target distance plus radius; do not evaluate a step.

    ``certified`` admits this endpoint and the conditional M8--M9 tail only.
    It cannot replace the runner's complete time/center/radius chain, every
    whole-tube margin, exact target/source association or prospective policy.
    An unavailable result retains conditional design bounds, not actual entry.
    """
    _require(isinstance(reference, FormedTwoPortReference), "reference type required")
    _require(isinstance(target, FormedTwoPortTarget), "admitted target type required")
    center = tuple(exact_record(value) for value in _row(center, 18, "endpoint center"))
    radius = exact_record(radius)
    _require(radius >= 0, "negative endpoint radius")
    errors = tuple(
        (I(value) - interval).abs_max
        for value, interval in zip(center, target.target_deviations, strict=True)
    )
    distance = radius + _metric_error_upper(reference.metric, errors)
    bounds = formed_two_port_comparison_bounds()
    reasons = (
        ()
        if distance <= bounds.reference_margin
        else ("reference_target_distance_exceeds_rho",)
    )
    return FormedTwoPortEndpoint(
        not reasons,
        reasons,
        distance,
        bounds.reference_margin - distance,
        bounds.endpoint_form_norm_upper_bound,
        bounds.endpoint_phase_distance_upper_bound,
        bounds.endpoint_excess_storage_upper_bound,
        Q(1, 8192) - bounds.endpoint_form_norm_upper_bound,
        Q(1, 1024) - bounds.endpoint_phase_distance_upper_bound,
        bounds.capture_radius**2 - bounds.endpoint_excess_storage_upper_bound,
        bounds.capture_storage_boundary - bounds.endpoint_excess_storage_upper_bound,
    )
