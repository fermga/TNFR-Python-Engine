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
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..mathematics._validated_metric import (
    METRIC_TAYLOR_METHOD,
    _metric_error_upper,
    enclose_metric_ball_differences,
    metric_coordinate_projection_factors,
    metric_difference_dual_norms,
)
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
    "FormedTwoPortAcuteExitCertificate",
    "FormedTwoPortOriginWindowCertificate",
    "FormedTwoPortAcuteExitPrefix",
    "FormedTwoPortOriginTransport",
    "FormedTwoPortPostAcuteWindow",
    "FormedTwoPortTransportBounds",
    "build_formed_two_port_reference",
    "admit_formed_two_port_target",
    "formed_two_port_comparison_bounds",
    "admit_formed_two_port_endpoint",
    "certify_formed_two_port_fifth_step_acute_exit",
    "admit_formed_two_port_origin_window",
    "admit_formed_two_port_acute_exit_prefix",
    "transport_formed_two_port_origin_source",
    "bound_formed_two_port_post_acute_window",
    "bound_formed_two_port_transport",
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
_CORRIDOR_KAPPA = 2 * _RHO
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

# Fixed published evidence, not a replacement execution or mutable manifest.
# Hashes bind the historical conditional Picard/Taylor proof and its audit;
# the new corridor and endpoint arithmetic are rederived below.
_PREFIX_RECEIPTS = (
    (
        "audit.json",
        2706,
        "73457c3783ec9fcc8620d8b2eb23cc7d8c3a5e09b932cfc16e76db8a1eeb9010",
    ),
    (
        "attempt/source.zip",
        4854391,
        "43e2a0760cb5e0e8db5822ebde72ecc7e9bda0992490d26e8b8c3abe47fec374",
    ),
    (
        "attempt/protocol.json",
        172037,
        "94b7e48015a3c7767996a1c41ad355f96fd69887a26b01819c13191e829f125a",
    ),
    (
        "attempt/steps/certificate-fields.json",
        12410,
        "31b008de3160b6e31ee1f6efc56df4b26a1f2d7572aea6298acfb4adb17754bc",
    ),
    (
        "attempt/steps/0000.json",
        30087,
        "eac5ee5f4c11be99095fcafc6dd0009fd6f081fefbf9aa29eb8e8f531b6feeea",
    ),
    (
        "attempt/steps/0001.json",
        34269,
        "f7557892a772d1dcfd64d26e7ecf29a70f0b075e6d31e3439d17077bccfe1ae9",
    ),
    (
        "attempt/steps/0002.json",
        34389,
        "9ddbc5e3bd16de4cbc303877d04213672869fc97dd8764e3e8b4bf22573644cf",
    ),
    (
        "attempt/steps/0003.json",
        34413,
        "b9fc7c6946db8dfe7c42b16dbdd17e5a1f26560cb5e8154bbcae406377d6ce60",
    ),
)


def _require(condition: bool, message: str) -> None:
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

    def edge_dual_norms(self) -> tuple[Q, ...]:
        """Return outward ``D^-1`` dual norms for the twenty oriented edges."""
        return metric_difference_dual_norms(self.metric, self.oriented_edges)

    def metric_ball_angles(self, center, radius) -> tuple[I, ...]:
        """Enclose all twenty edge gaps directly on ``||xi - center||_D <= radius``."""
        center = tuple(
            exact_record(value) for value in _row(center, 18, "metric ball center")
        )
        return enclose_metric_ball_differences(
            self.angles(center), radius, self.edge_dual_norms()
        )

    def corridor_domain(
        self,
        values,
        *,
        obtuse_edge: int = 5,
        kappa: Q = _CORRIDOR_KAPPA,
        donor_margin: Q = Q(1, 64),
        receiver_margin: Q = Q(1, 2),
    ) -> tuple[Q, ...]:
        """Return slacks above ``reference_margin`` in ``Omega(kappa, m_D, m_R)``."""
        values = _row(values, 18, "corridor domain")
        _require(
            not any(isinstance(value, Jet) for value in values),
            "corridor domain needs intervals",
        )
        _require(
            type(obtuse_edge) is int and 0 <= obtuse_edge < 9,
            "obtuse_edge must be a donor ring edge index in 0..8",
        )
        kappa = exact_record(kappa)
        donor_margin = exact_record(donor_margin)
        receiver_margin = exact_record(receiver_margin)
        _require(
            kappa >= 0 and donor_margin > 0 and receiver_margin > 0,
            "corridor margins must be positive and kappa nonnegative",
        )
        _require(
            I(kappa).hi <= (self.pi / 18).lo
            and I(donor_margin).hi < (self.pi / 18).lo
            and I(receiver_margin).hi < (self.pi / 2).lo,
            "corridor parameters exceed extended floor domain (D8)",
        )
        _require(
            (8 * sin(I(kappa))).hi <= sin(I(donor_margin)).lo,
            "corridor parameters violate path-compensation condition 8*sin(kappa)<=sin(m_D)",
        )
        slacks = []
        for idx, angle in enumerate(self.angles(values)):
            if idx == obtuse_edge:
                lower = (
                    angle - (-self.pi / 2 + donor_margin)
                ).lo - self.reference_margin
                upper = (self.pi / 2 + kappa - angle).lo - self.reference_margin
                slacks.append(min(lower, upper))
            elif idx < 9:
                slacks.append(
                    (self.pi / 2 - donor_margin - abs(angle)).lo - self.reference_margin
                )
            else:
                slacks.append(
                    (self.pi / 2 - receiver_margin - abs(angle)).lo
                    - self.reference_margin
                )
        return tuple(slacks)


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


def _require_fixed_reference(reference):
    """Admit all fixed law/chart primitives, not merely a dataclass label."""
    message = "different formed two-port reference primitives"
    _require(type(reference) is FormedTwoPortReference, message)

    def check(value, expected):
        if type(expected) is I:
            _require(type(value) is I, message)
            _require(
                type(value.lo) in (int, Q)
                and type(value.hi) in (int, Q)
                and exact_record(value.lo) == expected.lo
                and exact_record(value.hi) == expected.hi,
                message,
            )
        elif type(expected) is Q:
            _require(
                type(value) in (int, Q) and exact_record(value) == expected, message
            )
        elif type(expected) is int:
            _require(type(value) is int and value == expected, message)
        else:
            _require(
                type(expected) is tuple
                and type(value) is tuple
                and len(value) == len(expected),
                message,
            )
            for actual, canonical in zip(value, expected, strict=True):
                check(actual, canonical)

    try:
        for name, expected in vars(build_formed_two_port_reference()).items():
            check(getattr(reference, name), expected)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc


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


@dataclass(frozen=True)
class FormedTwoPortAcuteExitPrefix:
    """Documentary prefix admission; this report is not an authentication token."""

    endpoint_center: tuple[Q, ...]
    endpoint_radius: Q
    corridor_slacks: tuple[Q, ...]
    evidence_receipts: tuple[dict, ...]


def _interval_row(values, name):
    result = []
    for item in _row(values, 18, name):
        _require(isinstance(item, dict) and set(item) == {"lo", "hi"}, name)
        result.append(I(exact_record(item["lo"]), exact_record(item["hi"])))
    return tuple(result)


def admit_formed_two_port_acute_exit_prefix(reference, evidence_directory):
    """Re-admit exactly four published steps and their whole corridor tubes.

    ``evidence_directory`` is ``formed-two-port-attempt-20261029``. Fixed
    byte receipts bind the retained audit, producing protocol/archive and
    all four complete records. The shared reader reconstructs their fields;
    no producer, field, target sign, Taylor step or old auditor is executed.
    Consumed centers/radii and corridor slacks are rebuilt. Recorded Picard
    inclusions and derivative remainders retain their historical proof
    premise; hashes establish association, not chronology or authenticity.
    The public exit certifier repeats this admission rather than trusting
    a caller-constructed report or cached endpoint.
    """
    from .formed_two_port_run import read_step_certificate

    _require_fixed_reference(reference)
    directory = Path(evidence_directory).resolve(strict=True)
    content, receipts = {}, []
    for name, size, digest in _PREFIX_RECEIPTS:
        data = read_bytes_bounded(directory / name, max_bytes=size)
        _require(
            len(data) == size and sha256_bytes(data) == digest,
            f"retained prefix receipt differs: {name}",
        )
        content[name] = data
        receipts.append({"path": name, "bytes": size, "sha256": digest})
    audit = json_loads(content["audit.json"])
    _require(
        audit["verified_prefix_steps"] == 4
        and audit["source_associations_verified"] is True,
        "historical four-step audit association differs",
    )
    center, radius = reference.initial_center, reference.initial_radius
    slacks = None
    previous = None
    source = None
    projection = metric_coordinate_projection_factors(reference.metric)
    for index in range(4):
        name = f"attempt/steps/{index:04d}.json"
        record = json_loads(content[name])
        _require(
            record["previous_complete_step"] == previous,
            "retained prefix receipt chain differs",
        )
        if source is None:
            source = record["source"]
            for key, position in (("archive", 1), ("protocol", 2)):
                expected = dict(receipts[position])
                expected["path"] = expected["path"].removeprefix("attempt/")
                _require(source[key] == expected, "producing source receipt differs")
        _require(record["source"] == source, "producing source changed in prefix")
        step = read_step_certificate(directory / "attempt", index)
        # The shared reader reopens files. Require the fixed bytes to remain
        # unchanged across its read; this does not claim adversarial I/O isolation.
        for path in (name, "attempt/steps/certificate-fields.json"):
            _require(
                read_bytes_bounded(directory / path, max_bytes=len(content[path]))
                == content[path],
                "prefix changed during shared read",
            )
        _require(
            step["time"] == index * Q(1, 4)
            and step["duration"] == Q(1, 4)
            and type(step["order"]) is int
            and step["order"] == 8
            and step["metric"] == reference.metric
            and step["coordinate_projection_factors"] == projection
            and step["initial_center"] == center
            and step["initial_radius"] == radius,
            "retained prefix primitive chain differs",
        )
        _require(
            step["method"] == METRIC_TAYLOR_METHOD
            and step["growth_certificate"] is None
            and step["prescribed_time_interval"] is None
            and step["growth_rate_upper_bound"] == 0
            and step["growth_factor_upper_bound"] == 1
            and exact_record(step["picard_interior_margin"]) > 0,
            "retained prefix enclosure premises differ",
        )
        tube = _interval_row(step["tube"], "retained whole tube")
        current = reference.corridor_domain(tube)
        _require(min(current) > 0, "whole prefix not inside buffered corridor")
        slacks = current if slacks is None else tuple(map(min, zip(slacks, current)))
        polynomial = _interval_row(step["center_polynomial_bounds"], "polynomial")
        remainder = _interval_row(step["local_remainder_bounds"], "remainder")
        endpoint = tuple(p + r for p, r in zip(polynomial, remainder, strict=True))
        _require(
            endpoint == _interval_row(step["center_solution_endpoint"], "endpoint"),
            "retained center endpoint differs from polynomial plus remainder",
        )
        center = tuple(value.midpoint for value in endpoint)
        local = _metric_error_upper(
            reference.metric, tuple(value.radius for value in endpoint)
        )
        radius = I(I(radius).hi + local).hi
        _require(
            center == step["endpoint_center"] and radius == step["endpoint_radius"],
            "retained endpoint projection differs",
        )
        previous = dict(receipts[4 + index])
        previous["path"] = previous["path"].removeprefix("attempt/")
    return FormedTwoPortAcuteExitPrefix(center, radius, slacks, tuple(receipts))


@dataclass(frozen=True)
class FormedTwoPortAcuteExitCertificate:
    """Analytical certificate of reference and full-law acute exit on (1, 5/4).

    ``actual_full_law_excess_over_pi_half_lower`` holds at the existential
    first reference crossing of ``pi/2 + rho``, not at the endpoint ``5/4``.
    The ``*_contradiction_*`` fields retain their continued-domain premises.
    """

    certified: bool
    step_start_time: Q
    step_duration: Q
    initial_radius_at_one: Q
    active_edge_index: int
    active_gap_lower_at_one: Q
    active_rate_lower_at_one: Q
    reference_speed_upper_at_one: Q
    corridor_kappa: Q
    corridor_donor_margin: Q
    corridor_receiver_margin: Q
    second_derivative_coarse_upper: Q
    second_derivative_tight_upper: Q
    corridor_contradiction_excess_lower: Q
    full_law_comparison_error_upper: Q
    actual_full_law_excess_over_pi_half_lower: Q
    hessian_exit_horizon: Q
    psd_contradiction_slope_upper: Q
    psd_contradiction_gap_excess_lower: Q
    psd_contradiction_bulk_cosine_upper: Q
    psd_contradiction_q_bulk_upper: Q
    prefix_corridor_slacks: tuple[Q, ...]
    prefix_evidence_receipts: tuple[dict, ...]


def certify_formed_two_port_fifth_step_acute_exit(
    reference, evidence_directory
) -> FormedTwoPortAcuteExitCertificate:
    """Certify exit only after re-admitting the four-step source/corridor prefix.

    Given the re-admitted retained endpoint at slow time
    ``sigma=1``, this pure arithmetic verifier uses exact ``D``-metric edge dual
    norms, the path-compensated convex corridor ``Omega(1/1024, 1/64, 1/2)`` and
    the four-term second-derivative bound on ``g''_{5->6}`` to prove that both
    the reference ``psi(sigma)`` and every actual 36-variable trajectory
    ``(x(tau), theta(tau))`` of the formed-source family (A1) cross
    ``g_{5->6} = pi/2`` before ``sigma = 5/4``, and that the reference exits
    full-network Hessian positivity ``{H(g) >= 0}`` before ``sigma = 7/4``.
    The ``psd_contradiction_*`` values assume PSD for all of [1, 7/4];
    they are not bounds on the actual state after PSD has been lost.
    """
    prefix = admit_formed_two_port_acute_exit_prefix(reference, evidence_directory)
    return _certify_acute_exit_arithmetic(reference, prefix)


def _certify_acute_exit_arithmetic(reference, prefix):
    """Conditional arithmetic only; a supplied report does not prove its source."""
    _require_fixed_reference(reference)
    center_at_one, radius_at_one = prefix.endpoint_center, prefix.endpoint_radius
    center = tuple(
        exact_record(value) for value in _row(center_at_one, 18, "sigma=1 center")
    )
    radius = exact_record(radius_at_one)
    _require(radius >= 0, "negative sigma=1 radius")
    weighted_sum = sum(d * c for d, c in zip(reference.degrees, center))
    _require(
        abs(weighted_sum) <= Q(1, 2**64), "sigma=1 center violates zero joined mean"
    )

    duals = reference.edge_dual_norms()
    gaps1 = reference.metric_ball_angles(center, radius)
    projections = metric_coordinate_projection_factors(reference.metric)
    box1 = tuple(
        I(value - factor * radius, value + factor * radius)
        for value, factor in zip(center, projections, strict=True)
    )
    rates1 = reference.field(box1)
    speed1 = sqrt(
        sum((d * (r**2) for d, r in zip(reference.degrees, rates1, strict=True)), I(0))
    ).hi
    g56_1_lo = gaps1[5].lo
    g56_prime_1_lo = (rates1[6] - rates1[5]).lo

    kappa = Q(1, 1024)
    m_d = Q(1, 64)
    m_r = Q(1, 2)
    rho = reference.reference_margin
    pi = reference.pi

    ok = (
        radius < Q(1, 2**26)
        and (pi / 2 - g56_1_lo).hi < Q(44, 2048)
        and (pi / 2 - gaps1[5].hi).lo > rho
        and g56_prime_1_lo > Q(53, 500)
        and speed1 < Q(1, 4)
        and (8 * sin(I(kappa))).hi < sin(I(m_d)).lo
        and all((pi / 2 - abs(gaps1[idx])).lo > Q(1, 6) for idx in range(9) if idx != 5)
        and all((pi / 2 - abs(gaps1[idx])).lo > Q(2, 3) for idx in range(9, 20))
        and (gaps1[5].lo - Q(1, 16)) > (4 * pi / 9).hi
        and (gaps1[6].lo - Q(1, 16)) > Q(4, 3)
        and (gaps1[4].lo - Q(1, 16)) > Q(7, 6)
    )

    c56_max = cos(4 * pi / 9).hi
    c67_max = cos(I(Q(4, 3))).hi
    c45_max = cos(I(Q(7, 6))).hi
    second_coarse = (
        Q(5, 6) * c56_max * duals[5]
        + Q(1, 2) * c67_max * duals[6]
        + Q(1, 3) * c45_max * duals[4]
        + Q(1, 3) * duals[19]
    ) * speed1
    c56_tight = Q(55, 2048)
    second_tight = (
        Q(5, 6) * c56_tight * duals[5]
        + Q(1, 2) * c67_max * duals[6]
        + Q(1, 3) * c45_max * duals[4]
        + Q(1, 3) * duals[19]
    ) * speed1
    lower_54 = I(g56_1_lo + Q(1, 4) * g56_prime_1_lo - Q(1, 32) * second_tight)
    ref_excess = (lower_54 - pi / 2).lo
    bounds = formed_two_port_comparison_bounds()
    q_bound = bounds.phase_comparison_error_upper_bound
    actual_excess = rho - duals[5] * q_bound

    # (P1)--(P4): Hessian-positivity exit on [1, 7/4] (h = 3/4).
    h_hess = Q(3, 4)
    const_56_drop = Q(5, 6) * (Q(1) - sin(I(g56_1_lo)).lo)
    cosine_lowers = {
        index: gaps1[index].lo - duals[index] * speed1 * h_hess for index in (4, 6)
    }
    c67_hess = cos(I(cosine_lowers[6])).hi
    c45_hess = cos(I(cosine_lowers[4])).hi
    psd_gap_tube = tuple(
        gap + I(-dual * speed1 * h_hess, dual * speed1 * h_hess)
        for gap, dual in zip(gaps1, duals, strict=True)
    )
    # P1 does not require these two edges to remain acute. On [L, pi-L]
    # with 0 <= L <= pi/2, |cos g| <= cos L on both sides of pi/2.
    cosine_intervals_admitted = all(
        0 <= cosine_lowers[index] < (pi / 2).lo
        and psd_gap_tube[index].hi <= (pi - cosine_lowers[index]).lo
        for index in (4, 6)
    )
    w3_sq = (
        Q(1, 4) * (c67_hess**2)
        + Q(1, 27) * (c45_hess**2)
        + Q(1, 27) * Q(1)
        + Q(1, 27) * ((c45_hess + Q(1)) ** 2)
    )
    slope_3 = sqrt(I(w3_sq)).hi * speed1
    g56_74_lo = (
        g56_1_lo
        + h_hess * (g56_prime_1_lo - const_56_drop)
        - Q(1, 2) * (h_hess**2) * slope_3
    )
    hess_excess = (I(g56_74_lo) - pi / 2).lo
    xi64_hi = (center[6] - center[4]) + duals[5] * (radius + speed1 * h_hess)
    avg_gap_lo = (4 * pi / 9 - xi64_hi / 7).lo
    c_rest_hi = cos(I(avg_gap_lo)).hi / 7
    q_bulk_74_hi = -sin(I(hess_excess)).lo + c_rest_hi

    # These are lifted intervals, not principal-value wraps. Under the
    # hypothetical PSD continuation, P2 makes edge 5 obtuse; C3 then forces
    # the seven remaining bulk cosines strictly positive. The checks below
    # put their lifts in (0, pi), so they must be in (0, pi/2), where Jensen
    # is valid. No actual post-exit PSD or endpoint value is inferred.
    jensen_lifts_admitted = (
        all(0 < gap.lo <= gap.hi < pi.lo for gap in psd_gap_tube[:9])
        and 0 < avg_gap_lo < (pi / 2).lo
        and 0 < hess_excess < (pi / 2).lo
    )

    ok = (
        ok
        and c56_max < Q(1, 5)
        and c67_max < Q(1, 4)
        and c45_max < Q(2, 5)
        and second_coarse < Q(11, 64)
        and second_tight < Q(9, 64)
        and ref_excess > rho
        and actual_excess > 0
        and const_56_drop < Q(1, 4096)
        and slope_3 < Q(1, 10)
        and cosine_intervals_admitted
        and jensen_lifts_admitted
        and hess_excess > Q(1, 32)
        and c_rest_hi < Q(3, 100)
        and q_bulk_74_hi < -Q(1, 256) < 0
    )
    return FormedTwoPortAcuteExitCertificate(
        certified=bool(ok),
        step_start_time=Q(1),
        step_duration=Q(1, 4),
        initial_radius_at_one=radius,
        active_edge_index=5,
        active_gap_lower_at_one=g56_1_lo,
        active_rate_lower_at_one=g56_prime_1_lo,
        reference_speed_upper_at_one=speed1,
        corridor_kappa=kappa,
        corridor_donor_margin=m_d,
        corridor_receiver_margin=m_r,
        second_derivative_coarse_upper=second_coarse,
        second_derivative_tight_upper=second_tight,
        corridor_contradiction_excess_lower=ref_excess,
        full_law_comparison_error_upper=q_bound,
        actual_full_law_excess_over_pi_half_lower=actual_excess,
        hessian_exit_horizon=Q(7, 4),
        psd_contradiction_slope_upper=slope_3,
        psd_contradiction_gap_excess_lower=hess_excess,
        psd_contradiction_bulk_cosine_upper=c_rest_hi,
        psd_contradiction_q_bulk_upper=q_bulk_74_hi,
        prefix_corridor_slacks=prefix.corridor_slacks,
        prefix_evidence_receipts=prefix.evidence_receipts,
    )


@dataclass(frozen=True)
class FormedTwoPortOriginWindowCertificate:
    """Conditional box inclusion, not authentication of an actual flow image.

    The supplied origin is ``pi_coefficient*pi + rational_offset``; keeping
    these exact coefficients preserves correlation at the exact midpoint.
    """

    certified: bool
    receiver_origin_pi_coefficient: Q
    receiver_origin_rational_offset: Q
    origin_window_radius: Q
    origin_deviation_from_midpoint_upper: Q
    source_residual_norm_upper: Q
    nodewise_form_error_upper: Q
    nodewise_phase_error_upper: Q
    midpoint_capture_box_radius: Q
    contact_work_upper_bound: Q
    midpoint_work_budget: Q
    formed_source_work_allowance: Q


def admit_formed_two_port_origin_window(
    reference, phi_0: Q = Q(355, 1017), *, phi_pi_coefficient: Q = Q(0)
) -> FormedTwoPortOriginWindowCertificate:
    """Check O1--O3 for the exact origin ``phi_pi_coefficient*pi + phi_0``.

    This changes the supplied disjoint preparation. It does not transform
    the inherited phi=1/1000 joined trajectory into a different trajectory,
    admit the historical capture chain, or prove that a supplied point is
    an actual isolated formation image.
    """
    _require_fixed_reference(reference)
    phi_0 = exact_record(phi_0)
    coefficient = exact_record(phi_pi_coefficient)
    epsilon = Q(1, 10**32)
    box_radius = Q(1, 65536)
    window_radius = Q(1, 32768) - 2 * epsilon
    pi = reference.pi
    delta_phi = ((coefficient - Q(1, 9)) * pi + phi_0).abs_max
    form_error = epsilon
    phase_error = Q(1, 2) * delta_phi + epsilon
    contact_gap_max = pi / 9 + I(delta_phi + 2 * epsilon)
    work_upper = 2 * (epsilon**2) + 2 * (I(1) - cos(contact_gap_max)).hi
    ok = (
        delta_phi <= window_radius
        and form_error <= box_radius
        and phase_error <= box_radius
        and work_upper < Q(1, 8) < Q(1, 4)
    )
    return FormedTwoPortOriginWindowCertificate(
        certified=bool(ok),
        receiver_origin_pi_coefficient=coefficient,
        receiver_origin_rational_offset=phi_0,
        origin_window_radius=window_radius,
        origin_deviation_from_midpoint_upper=delta_phi,
        source_residual_norm_upper=epsilon,
        nodewise_form_error_upper=form_error,
        nodewise_phase_error_upper=phase_error,
        midpoint_capture_box_radius=box_radius,
        contact_work_upper_bound=work_upper,
        midpoint_work_budget=Q(1, 8),
        formed_source_work_allowance=Q(1, 4),
    )


@dataclass(frozen=True)
class FormedTwoPortOriginTransport:
    """An A1-bounded point transported without erasing memberwise means.

    Meeting A1 residual bounds does not authenticate isolated acquisition.
    The four supplied rows retain their joint values; no independent reset
    or new endpoint preparation is inferred from this chart calculation.
    """

    original_indices: tuple[int, ...]
    original_phase: tuple[I, ...]
    form: tuple[Q, ...]
    phase: tuple[I, ...]
    midpoint_nominal_phase: tuple[I, ...]
    phase_deviation: tuple[I, ...]
    common_gauge: I
    joined_form_mean: Q
    joined_phase_residual_mean: Q
    window: FormedTwoPortOriginWindowCertificate


def transport_formed_two_port_origin_source(
    reference,
    phi_0,
    donor_form,
    donor_phase,
    receiver_form,
    receiver_phase,
    *,
    phi_pi_coefficient=Q(0),
):
    """Transport all 36 coordinates through A3 and the nominal midpoint gauge.

    Inputs are four original-label residual rows in the isolated A1 lifts.
    Their exact zero sums and Euclidean norm bounds are necessary conditions,
    not proof of membership in the actual formation-image family. Fixed
    winding lift restoration is applied once, not repeatedly wrapped.
    The gauge is nominal: the actual member's nonzero joined residual means
    are retained, as required by the capture box with independent errors.
    """
    window = admit_formed_two_port_origin_window(
        reference, phi_0, phi_pi_coefficient=phi_pi_coefficient
    )
    _require(window.certified, "supplied origin is outside the admitted window")
    rows = []
    epsilon = window.source_residual_norm_upper
    for name, values in (
        ("donor form", donor_form),
        ("donor phase", donor_phase),
        ("receiver form", receiver_form),
        ("receiver phase", receiver_phase),
    ):
        row = tuple(exact_record(v) for v in _row(values, 9, name))
        _require(
            sum(row) == 0 and sum(v * v for v in row) <= epsilon**2,
            f"{name} violates A1 residual bounds",
        )
        rows.append(row)
    vd, ud, vr, ur = rows
    pi = reference.pi
    offset = window.receiver_origin_rational_offset
    coefficient = window.receiver_origin_pi_coefficient
    delta = (coefficient - Q(1, 9)) * pi + offset
    origin = coefficient * pi + offset
    gauge = (coefficient / 2 + Q(73, 60)) * pi + offset / 2
    original = tuple(
        Q((4 if i < 9 else 2) * (i % 9 - 4), 9) * pi
        + (I(0) if i < 9 else origin)
        + (ud if i < 9 else ur)[i % 9]
        for i in _NODES
    )
    indices = tuple(9 * (i // 9) + (i % 9 + 4) % 9 for i in _NODES)
    form = tuple((vd if j < 9 else vr)[j % 9] for j in indices)
    nominal = tuple(
        (
            Q((4 if i < 9 else 2) * (i % 9), 9)
            + (Q(0) if i < 9 else Q(1, 9))
            - Q(229, 180)
        )
        * pi
        for i in _NODES
    )
    deviations = tuple(
        (ud if j < 9 else ur)[j % 9] + (-delta / 2 if j < 9 else delta / 2)
        for j in indices
    )
    # Correlated affine evaluation of original[indices] + fixed lifts - gauge.
    phase = tuple(p + d for p, d in zip(nominal, deviations, strict=True))
    old_degrees = tuple(3 if i % 9 in (0, 1) else 2 for i in _NODES)
    form_mean = sum(d * v for d, v in zip(old_degrees, form, strict=True)) / 40
    phase_mean = (
        sum(
            d * (ud if j < 9 else ur)[j % 9]
            for d, j in zip(old_degrees, indices, strict=True)
        )
        / 40
    )
    _require(
        all(abs(v) <= window.nodewise_form_error_upper for v in form)
        and all(v.abs_max <= window.nodewise_phase_error_upper for v in deviations),
        "transported point exceeds the admitted midpoint box",
    )
    return FormedTwoPortOriginTransport(
        indices,
        original,
        form,
        phase,
        nominal,
        deviations,
        gauge,
        form_mean,
        phase_mean,
        window,
    )


@dataclass(frozen=True)
class FormedTwoPortPostAcuteWindow:
    """Uniform full-law/reference error on the fixed slow interval [1, 7/4].

    These are norm bounds relative to the exact mean-matched reference, not
    an enclosure of its unevaluated later state. The retained numerical
    radius at time one is a separate quantity and is not silently set to
    zero, added to the model error, or propagated without a new numerical
    certificate. Original form is centered on each member's conserved mean.
    No capture, winding change or actual post-acute Hessian sign is certified.
    """

    certified: bool
    start_slow_time: Q
    end_slow_time: Q
    source_error_radius: Q
    source_scaled_form_upper_bound: Q
    source_mixed_error_upper_bound: Q
    initial_mixed_error_upper_bound: Q
    initial_scaled_form_upper_bound: Q
    forcing_norm_upper_bound: Q
    eta_over_gap_upper_bound: Q
    phase_lipschitz_bound: Q
    scaled_form_norm_upper_bound: Q
    half_exponential_upper_bound: Q
    growth_factor_upper_bound: Q
    mixed_error_upper_bound: Q
    phase_error_upper_bound: Q
    phase_error_design_ceiling: Q
    centered_original_form_norm_upper_bound: Q
    reference_numerical_radius_at_start: Q
    prefix_corridor_slacks: tuple[Q, ...]
    prefix_evidence_receipts: tuple[dict, ...]
    scope: tuple[str, ...] = (
        "original_phi_1_over_1000_complete_law_and_memberwise_means_retained",
        "prefix_one_uses_the_historical_conditional_Picard_Taylor_evidence",
        "post_acute_comparison_uses_global_forcing_and_Lipschitz_bounds_not_PSD",
        "phase_error_is_relative_to_the_exact_mean_matched_reference",
        "original_form_bound_is_on_P_D_x_not_the_absolute_common_mode",
        "no_reference_step_after_one_or_actual_endpoint_state_is_evaluated",
        "no_capture_winding_change_or_actual_Hessian_sign_conclusion",
    )


def bound_formed_two_port_post_acute_window(reference, evidence_directory):
    """Rebuild the fixed [1, 7/4] corollary from source primitives and prefix.

    The original source, degree metric, lifts and clock are admitted by the
    existing four-step reader. Its whole corridor supports M3--M6 only up
    to slow time one. Thereafter global ||f||_D < 7 and Lipschitz constant
    two replace every acute/PSD premise: variation of constants gives
    Z < 7/100000, and D+V <= 2V + 2Z gives Q <= exp(2h)(V1+7/100000).
    No historical long-horizon comparison report or cached source bound is
    consumed. This function evaluates only the retained-prefix admission
    and exact rational inequalities; it performs no new flow or root solve.
    """
    prefix = admit_formed_two_port_acute_exit_prefix(reference, evidence_directory)
    epsilon = Q(1, 10**32)
    z0 = 3 * epsilon / 3069
    v0 = 3 * epsilon + z0
    # At Sigma=1, Lambda*sqrt(Sigma*E0) <= 1 and eta/lambda < 1/100000.
    ratio = Q(1, 100000)
    lipschitz = Q(2)
    v1 = v0 + ratio
    z1 = (z0 + ratio * (Q(3, 5) + lipschitz * v1)) / (1 - ratio * lipschitz)
    forcing = Q(7)
    zbar = forcing * ratio
    _require(
        Q(49, 200) + Q(1, 500000) + 10 * epsilon**2 < Q(1, 4)
        and sum(reference.degrees) == 40 < forcing**2
        and Q(90) / (1023 * reference.pi.lo) ** 2 < ratio
        and 3069 < 1023 * reference.pi.lo <= 1023 * reference.pi.hi < 3216
        and 0 < v1 < Q(1, 90000)
        and 0 < z1 < Q(1, 100000) < zbar
        and v1 + z1 < reference.reference_margin,
        "short-prefix post-acute source bounds unavailable",
    )
    # exp(1/2): terms through degree two are 13/8. The first omitted
    # term is 1/48; all later successive ratios are at most 1/8.
    # This exact geometric tail majorant needs no exponential evaluator.
    half_partial = Q(1) + Q(1, 2) + Q(1, 8)
    half_tail = Q(1, 48) / (1 - Q(1, 8))
    half_exponential = half_partial + half_tail
    growth = half_exponential**3
    start, end = Q(1), Q(7, 4)
    phase = growth * (v1 + zbar)
    mixed = phase - zbar
    ceiling = Q(73, 200000)
    original_form = 3216 * zbar
    _require(
        lipschitz * (end - start) == 3 * Q(1, 2)
        and half_exponential == Q(277, 168) < Q(33, 20)
        and Q(33, 20) ** 3 < Q(9, 2)
        and Q(9, 2) * (Q(1, 90000) + zbar) == ceiling
        and v1 < mixed < phase < ceiling < reference.reference_margin
        and original_form == Q(1407, 6250),
        "fixed post-acute rational corollary unavailable",
    )
    return FormedTwoPortPostAcuteWindow(
        certified=True,
        start_slow_time=start,
        end_slow_time=end,
        source_error_radius=epsilon,
        source_scaled_form_upper_bound=z0,
        source_mixed_error_upper_bound=v0,
        initial_mixed_error_upper_bound=v1,
        initial_scaled_form_upper_bound=z1,
        forcing_norm_upper_bound=forcing,
        eta_over_gap_upper_bound=ratio,
        phase_lipschitz_bound=lipschitz,
        scaled_form_norm_upper_bound=zbar,
        half_exponential_upper_bound=half_exponential,
        growth_factor_upper_bound=growth,
        mixed_error_upper_bound=mixed,
        phase_error_upper_bound=phase,
        phase_error_design_ceiling=ceiling,
        centered_original_form_norm_upper_bound=original_form,
        reference_numerical_radius_at_start=prefix.endpoint_radius,
        prefix_corridor_slacks=prefix.corridor_slacks,
        prefix_evidence_receipts=prefix.evidence_receipts,
    )


@dataclass(frozen=True)
class FormedTwoPortTransportBounds:
    """Source-specific velocity, retention and conditional cut/capture bounds.

    The global lag compares the actual phase velocity with f at the actual
    phase, not with an unevaluated reference trajectory. Principal retention
    has its separate finite horizon. The outgoing patch gives a local
    implication only, as does the target-ball handoff. Source arrival and
    the eventual basin remain unproved.
    """

    certified: bool
    post_acute_window: FormedTwoPortPostAcuteWindow
    tracking_start_slow_time: Q
    initial_phase_velocity_upper_bound: Q
    initial_velocity_lag_upper_bound: Q
    lag_decay_lower_bound: Q
    initial_transient_decay_upper_bound: Q
    velocity_lag_upper_bound: Q
    velocity_lag_design_ceiling: Q
    edge_dual_norm_squares: tuple[Q, ...]
    edge_velocity_lag_squared_bounds: tuple[Q, ...]
    retention_slow_interval: tuple[Q, Q]
    principal_margin_lower_bound: Q
    retained_periods: tuple[int, int, int]
    reference_edge_speed_upper_bound: Q
    reference_edge_displacement_upper_bound: Q
    cut_gap_pi_coefficients: tuple[Q, ...]
    cut_edge_index: int
    cut_patch_radius: Q
    cut_gradient_normal_lower_bound: Q
    conditional_cut_normal_lower_bound: Q
    conditional_cut_normal_design_floor: Q
    local_outgoing_periods: tuple[int, int, int]
    conditional_capture_phase_radius: Q
    conditional_capture_form_upper_bound: Q
    conditional_capture_excess_upper_bound: Q
    conditional_capture_storage_margin: Q
    scope: tuple[str, ...] = (
        "unchanged_original_phi_1_over_1000_A2_source_and_all_36_coordinates",
        "fixed_support_unit_capacities_no_forcing_or_subsequent_events",
        "sigma_gamma_squared_times_tau_minus_original_contact_tau",
        "velocity_lag_valid_after_tracking_start_without_acute_or_PSD_domain",
        "lag_compares_theta_sigma_to_f_at_actual_theta_not_to_reference_velocity",
        "principal_retention_only_on_the_reported_finite_interval",
        "prefix_inherits_historical_Picard_and_remainder_premises_not_authentication",
        "cut_patch_uses_original_continuous_lifts_and_one_common_mean_matching",
        "local_outgoing_periods_conditional_on_source_reaching_the_cut_patch",
        "capture_bounds_require_actual_phase_distance_to_the_exact_retained_target",
        "no_patch_arrival_target_entry_eventual_winding_or_basin_is_selected",
        "no_new_field_evaluation_reference_step_target_sign_or_producer",
    )


def bound_formed_two_port_transport(
    reference: FormedTwoPortReference, evidence_directory: str | Path
) -> FormedTwoPortTransportBounds:
    """Re-admit the prefix and rebuild ST1--ST7 by rational arithmetic.

    For v=theta_sigma and r=v-f(theta), the full two-row law gives
    r_sigma=-(A/eta)*r-Df(theta)*v. Source and global field bounds imply
    D+||r||_D <= -99998*||r||_D+14. After sigma=1/128 the lag is
    below 1/7000, without extending the finite position-error certificate.
    """
    post: FormedTwoPortPostAcuteWindow = bound_formed_two_port_post_acute_window(  # type: ignore[no-untyped-call]
        reference, evidence_directory
    )
    epsilon = post.source_error_radius
    lipschitz = post.phase_lipschitz_bound
    initial_velocity = Q(2) * 3216 * 3 * epsilon
    initial_lag = initial_velocity + Q(3, 5) + lipschitz * 3 * epsilon
    decay = 1 / post.eta_over_gap_upper_bound - lipschitz
    start = Q(1, 128)
    transient = Q(1, 2**512)
    lag = initial_lag * transient + lipschitz * post.forcing_norm_upper_bound / decay
    lag_ceiling = Q(1, 7000)
    dual_squares = tuple(
        Q(1, reference.degrees[left]) + Q(1, reference.degrees[right])
        for left, right in reference.oriented_edges
    )

    pi_lower = Q(25, 8)
    edge_speed = Q(2)
    displacement = edge_speed * (post.end_slow_time - post.start_slow_time)
    prefix_margin = pi_lower / 2 - _CORRIDOR_KAPPA
    endpoint_excess = _CORRIDOR_KAPPA - reference.reference_margin
    later_margin = (
        pi_lower / 2 - endpoint_excess - displacement - post.phase_error_upper_bound
    )
    principal_margin = min(prefix_margin, later_margin)

    cut = (
        tuple(Q(1) if i == 5 else Q(3, 8) for i in range(9))
        + (Q(2, 9),) * 9
        + (Q(11, 144), Q(-11, 144))
    )
    radius = Q(1, 256)
    # Chord bounds for sin(3*pi/8) and sin(11*pi/144), not a field evaluation.
    gradient_lower = Q(5, 6) * 2 * Q(3, 8) + Q(1, 3) * 2 * Q(11, 144)
    normal_lower = gradient_lower - lipschitz * radius - lag
    capture_phase = Q(1, 1024)
    capture_form = Q(90, 3069) * (lipschitz * capture_phase + lag)
    capture_excess = capture_form**2 + capture_phase**2
    capture_margin = Q(1, 648000) - capture_excess
    _require(
        0 < initial_lag < 1
        and decay == 99998
        and decay * start > 512
        and 0 < lag < lag_ceiling
        and len(dual_squares) == 20
        and all(0 < value <= 1 for value in dual_squares)
        and pi_lower < reference.pi.lo
        and _CORRIDOR_KAPPA == 2 * reference.reference_margin
        and displacement == Q(3, 2)
        and principal_margin > Q(3, 50)
        and sum(cut[:9]) == 4
        and sum(cut[9:18]) == 2
        and cut[18] + cut[13] - cut[19] - cut[4] == 0
        and radius
        < (1 - max(abs(value) for i, value in enumerate(cut) if i != 5)) * pi_lower
        and gradient_lower == Q(73, 108)
        and normal_lower > Q(2, 3)
        and capture_form < Q(1, 8192)
        and capture_excess < Q(1, 12) ** 2
        and capture_margin > 0,
        "source-dependent transport inequalities unavailable",
    )
    return FormedTwoPortTransportBounds(
        certified=True,
        post_acute_window=post,
        tracking_start_slow_time=start,
        initial_phase_velocity_upper_bound=initial_velocity,
        initial_velocity_lag_upper_bound=initial_lag,
        lag_decay_lower_bound=decay,
        initial_transient_decay_upper_bound=transient,
        velocity_lag_upper_bound=lag,
        velocity_lag_design_ceiling=lag_ceiling,
        edge_dual_norm_squares=dual_squares,
        edge_velocity_lag_squared_bounds=tuple(
            value * lag**2 for value in dual_squares
        ),
        retention_slow_interval=(Q(0), post.end_slow_time),
        principal_margin_lower_bound=principal_margin,
        retained_periods=(2, 1, 0),
        reference_edge_speed_upper_bound=edge_speed,
        reference_edge_displacement_upper_bound=displacement,
        cut_gap_pi_coefficients=cut,
        cut_edge_index=5,
        cut_patch_radius=radius,
        cut_gradient_normal_lower_bound=gradient_lower,
        conditional_cut_normal_lower_bound=normal_lower,
        conditional_cut_normal_design_floor=Q(2, 3),
        local_outgoing_periods=(1, 1, 0),
        conditional_capture_phase_radius=capture_phase,
        conditional_capture_form_upper_bound=capture_form,
        conditional_capture_excess_upper_bound=capture_excess,
        conditional_capture_storage_margin=capture_margin,
    )
