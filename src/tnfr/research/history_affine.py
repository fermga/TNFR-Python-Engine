"""Exact affine obligations on a separately authenticated finite history graph.

Admission binds immutable algebraic data, not its provenance from TNFR physics.
Callers separately establish canonical guards, root and whole-target coverage.
Acceptance never implies origin reachability or a global stability theorem.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from weakref import WeakKeyDictionary

__all__ = [
    "AdmittedSpecification",
    "CoefficientBlocks",
    "Obligation",
    "parse_rational",
    "specification_digest",
    "admit_specification",
    "parse_coefficients",
    "affine_form",
    "point_constraint",
    "root_constraint",
    "exact_affine_minimum",
    "exact_minimum",
    "verify_candidate",
]

_MAX_BITS = 16384
_RATIO = re.compile(r"([+-]?[0-9]+)/([0-9]+)\Z")
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _limit(value, label, ceiling=1_000_000):
    if type(value) is not int or not 1 <= value <= ceiling:
        raise ValueError(f"{label} must be a positive integer at most {ceiling}")
    return value


def _bits(value):
    return max(abs(value.numerator).bit_length(), value.denominator.bit_length())


def parse_rational(value, *, max_bits=4096):
    """Read bounded exact input; reject floats and oversized text before parsing.

    Decimal/exponent strings denote exact rationals. Conservative textual,
    digit and exponent caps also apply to values that could simplify to zero.
    A resource rejection is not evidence of mathematical infeasibility.
    """
    _limit(max_bits, "max_bits", _MAX_BITS)
    if type(value) is int:
        if abs(value).bit_length() > max_bits:
            raise ValueError("Rational integer bit budget exceeded")
        return Fraction(value)
    if type(value) is Fraction:
        if (
            type(value.numerator) is not int
            or type(value.denominator) is not int
            or value.denominator <= 0
        ):
            raise ValueError("Malformed exact Fraction internals")
        if _bits(value) > max_bits:
            raise ValueError("Rational bit budget exceeded")
        return Fraction(value.numerator, value.denominator)
    if type(value) is not str:
        raise ValueError(
            "Exact rational inputs must be integers, strings or Fractions; floats/bools are forbidden"
        )
    digits = (max_bits * 30103 + 99999) // 100000 + 1
    if len(value) > 2 * digits + 32:
        raise ValueError("Rational textual length budget exceeded before parsing")
    text = value.strip()
    ratio, decimal = _RATIO.fullmatch(text), _DECIMAL.fullmatch(text)
    if ratio:
        if any(len(part.lstrip("+-")) > digits for part in ratio.groups()):
            raise ValueError("Rational digit budget exceeded before parsing")
    elif decimal:
        exponent = decimal.group(1)
        if exponent is not None and (
            len(exponent.lstrip("+-")) > 6 or abs(int(exponent)) > digits
        ):
            raise ValueError("Rational exponent budget exceeded before parsing")
        mantissa = re.split("[eE]", text)[0]
        if sum(character.isdigit() for character in mantissa) > digits:
            raise ValueError("Rational digit budget exceeded before parsing")
    else:
        raise ValueError("Malformed bounded exact rational")
    try:
        result = Fraction(text)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("Malformed finite exact rational") from error
    if _bits(result) > max_bits:
        raise ValueError("Rational bit budget exceeded")
    return result


def _integer(value, label, max_bits=4096, nonnegative=False):
    if (
        type(value) is not int
        or abs(value).bit_length() > max_bits
        or (nonnegative and value < 0)
    ):
        raise ValueError(
            f"{label} must be an exact {'nonnegative ' if nonnegative else ''}bounded integer"
        )
    return value


def _sequence(value, length, label):
    if type(value) not in (tuple, list) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} entries")
    return value


def _mapping(value, keys, label):
    if (
        type(value) is not dict
        or any(type(key) is not str for key in value)
        or set(value) != set(keys)
    ):
        raise ValueError(f"Malformed {label} fields")
    return value


def _json_preflight(raw):
    """Bound canonicalization before allocating its complete JSON encoding."""
    pending, nodes, characters = [(raw, 0)], 0, 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 2_000_000 or depth > 40:
            raise ValueError("Specification JSON structure budget exceeded")
        if type(value) is dict:
            if len(value) > 1_000_000 or any(type(key) is not str for key in value):
                raise ValueError(
                    "Specification JSON needs bounded string-keyed objects"
                )
            pending.extend((key, depth + 1) for key in value)
            pending.extend((item, depth + 1) for item in value.values())
        elif type(value) in (list, tuple):
            if len(value) > 1_000_000:
                raise ValueError("Specification JSON sequence budget exceeded")
            pending.extend((item, depth + 1) for item in value)
        elif type(value) is str:
            characters += len(value)
            if characters > 32_000_000:
                raise ValueError("Specification JSON text budget exceeded")
        elif type(value) is int:
            _integer(value, "JSON integer", _MAX_BITS)
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("Nonfinite JSON metadata")
        elif value is not None and type(value) is not bool:
            raise ValueError("Unsupported specification JSON value")


def _canonical_bytes(raw):
    if type(raw) is not dict:
        raise ValueError("Specification must be a decoded JSON object")
    _json_preflight(raw)
    return json.dumps(
        raw, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf8")


def specification_digest(raw):
    """Hash canonical decoded JSON, including metadata outside the proof fields."""
    return hashlib.sha256(_canonical_bytes(raw)).hexdigest()


def _bounds(raw, max_bits):
    rows = _sequence(raw, 7, "DBM")
    bounds = tuple(
        tuple(
            _integer(value, "DBM bound", max_bits)
            for value in _sequence(row, 7, "DBM row")
        )
        for row in rows
    )
    return bounds


def _closed(bounds):
    if any(bounds[i][i] != 0 for i in range(7)):
        raise ValueError("A consistent closed DBM requires zero diagonal")
    if any(
        bounds[i][j] > bounds[i][k] + bounds[k][j]
        for i in range(7)
        for j in range(7)
        for k in range(7)
    ):
        raise ValueError("DBM is not closed and consistent")


class _WeakReferenceable:
    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class Obligation:
    """One retained identity in declaration order; never a merged geometry."""

    kind: str
    index: int
    ordinal: int
    guard_id: int
    source_vertex: int
    target_vertex: int
    shift: tuple[int, ...]


@dataclass(frozen=True, slots=True, eq=False)
class AdmittedSpecification(_WeakReferenceable):
    """Immutable algebraic specification; obtain it through admit_specification."""

    digest: str
    vertices: tuple[int, ...]
    normalization: int
    guards: tuple[tuple[tuple[int, ...], ...], ...]
    root_vertex: int
    root_point: tuple[int, ...]
    root_minimum: Fraction
    drift_obligations: tuple[Obligation, ...]
    target_obligations: tuple[Obligation, ...]
    obligations: tuple[Obligation, ...]
    _columns: object = field(repr=False)

    @property
    def coefficient_count(self):
        return 7 * len(self.vertices)


@dataclass(frozen=True, slots=True, eq=False)
class CoefficientBlocks(_WeakReferenceable):
    """Exact ordered coefficients belonging to one admitted specification."""

    flat: tuple[Fraction, ...]
    records: tuple[tuple[int, tuple[Fraction, ...], Fraction], ...]
    largest_bits: int
    _spec: AdmittedSpecification = field(repr=False)


_SPECIFICATIONS = WeakKeyDictionary()
_COEFFICIENTS = WeakKeyDictionary()
_SPEC_FIELDS = tuple(AdmittedSpecification.__dataclass_fields__)
_BLOCK_FIELDS = tuple(CoefficientBlocks.__dataclass_fields__)
_OBLIGATION_FIELDS = tuple(Obligation.__dataclass_fields__)


def _fraction_pair(value):
    if (
        type(value) is not Fraction
        or type(value.numerator) is not int
        or type(value.denominator) is not int
    ):
        raise ValueError("Mutated admitted rational")
    return value.numerator, value.denominator


def _specification(spec):
    if type(spec) is not AdmittedSpecification or spec not in _SPECIFICATIONS:
        raise ValueError("Use the immutable result of admit_specification")
    fields, root_pair, _obligations = _SPECIFICATIONS[spec]
    if any(
        getattr(spec, name) is not value for name, value in zip(_SPEC_FIELDS, fields)
    ):
        raise ValueError("Admitted specification fields were mutated")
    if _fraction_pair(spec.root_minimum) != root_pair:
        raise ValueError("Admitted root rational was mutated")


def _coefficient_blocks(spec, blocks):
    _specification(spec)
    if (
        type(blocks) is not CoefficientBlocks
        or blocks not in _COEFFICIENTS
        or blocks._spec is not spec
    ):
        raise ValueError("Coefficients must be parsed for this admitted specification")
    fields, _values = _COEFFICIENTS[blocks]
    if any(
        getattr(blocks, name) is not value for name, value in zip(_BLOCK_FIELDS, fields)
    ):
        raise ValueError("Parsed coefficient fields were mutated")


def _coefficient_value(blocks, index):
    """Use the private primitive snapshot, after detecting exposed mutation."""
    pair = _COEFFICIENTS[blocks][1][index]
    if _fraction_pair(blocks.flat[index]) != pair:
        raise ValueError("Parsed coefficient rational was mutated")
    return Fraction(*pair)


def _obligation(spec, obligation):
    _specification(spec)
    if (
        type(obligation) is not Obligation
        or type(obligation.ordinal) is not int
        or not 0 <= obligation.ordinal < len(spec.obligations)
        or spec.obligations[obligation.ordinal] is not obligation
    ):
        raise ValueError("Obligation does not belong to this admitted specification")
    recorded = _SPECIFICATIONS[spec][2][obligation.ordinal]
    if any(
        getattr(obligation, name) is not value
        for name, value in zip(_OBLIGATION_FIELDS, recorded)
    ):
        raise ValueError("Admitted obligation fields were mutated")


def admit_specification(
    raw,
    expected_digest,
    *,
    max_vertices=10000,
    max_obligations=100000,
    max_guards=100000,
    max_integer_bits=4096,
):
    """Validate each pooled geometry once and freeze all complete identities.

    The expected digest binds caller-authenticated data. This pure algebraic
    admission does not reconstruct a physical source or establish target scope.
    """
    for value, label in (
        (max_vertices, "max_vertices"),
        (max_obligations, "max_obligations"),
        (max_guards, "max_guards"),
    ):
        _limit(value, label)
    _limit(max_integer_bits, "max_integer_bits", _MAX_BITS)
    if type(expected_digest) is not str or not _DIGEST.fullmatch(expected_digest):
        raise ValueError("Expected a canonical specification SHA256")
    encoded = _canonical_bytes(raw)
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise ValueError("Specification content does not match its expected digest")
    # Hash and admission read the same private snapshot even if caller-owned
    # dict/list objects are modified after canonicalization.
    raw = json.loads(encoded)
    required = {
        "normalization",
        "coefficient_vertices",
        "guards",
        "root",
        "drift_obligations",
        "target_obligations",
    }
    if not required <= raw.keys():
        raise ValueError("Missing specification proof fields")
    vertices = raw["coefficient_vertices"]
    if type(vertices) not in (list, tuple) or not 0 < len(vertices) <= max_vertices:
        raise ValueError("Coefficient domain size budget exceeded or empty")
    vertices = tuple(_integer(v, "Vertex", max_integer_bits, True) for v in vertices)
    if vertices != tuple(sorted(set(vertices))):
        raise ValueError("Coefficient vertices must be distinct and sorted")
    columns = {v: 7 * i for i, v in enumerate(vertices)}
    normalization = _integer(
        raw["normalization"], "Normalization", max_integer_bits, True
    )
    if normalization == 0:
        raise ValueError("Normalization must be positive")
    guard_records = raw["guards"]
    if (
        type(guard_records) not in (list, tuple)
        or not 0 < len(guard_records) <= max_guards
    ):
        raise ValueError("Guard pool size budget exceeded or empty")
    guards, pool = [], {}
    for ordinal, record in enumerate(guard_records):
        _mapping(record, ("guard_id", "bounds"), "pooled guard")
        if (
            _integer(record["guard_id"], "Guard identity", max_integer_bits, True)
            != ordinal
        ):
            raise ValueError("Guard identities must be contiguous")
        bounds = _bounds(record["bounds"], max_integer_bits)
        if bounds not in pool:
            _closed(bounds)
            pool[bounds] = bounds
        guards.append(pool[bounds])
    root = _mapping(raw["root"], ("vertex", "point", "minimum"), "root")
    root_vertex = _integer(root["vertex"], "Root vertex", max_integer_bits, True)
    if root_vertex not in columns:
        raise ValueError("Root lies outside the coefficient domain")
    root_point = tuple(
        _integer(x, "Root coordinate", max_integer_bits)
        for x in _sequence(root["point"], 6, "Root point")
    )
    root_minimum = parse_rational(root["minimum"], max_bits=max_integer_bits)
    if root_minimum <= 0:
        raise ValueError("Root minimum must be strictly positive")
    obligations, families = [], []
    for kind, key in (("drift", "drift_obligations"), ("target", "target_obligations")):
        records = raw[key]
        if (
            type(records) not in (list, tuple)
            or len(records) + len(obligations) > max_obligations
        ):
            raise ValueError("Obligation budget exceeded or malformed family")
        if kind == "target" and not records:
            raise ValueError("Target family must be complete and nonempty")
        seen, family = set(), []
        for record in records:
            identity = "arc_index" if kind == "drift" else "piece_index"
            keys = (
                (identity, "source_vertex", "target_vertex", "guard_id", "shift")
                if kind == "drift"
                else (identity, "vertex", "guard_id")
            )
            _mapping(record, keys, kind + " obligation")
            index = _integer(
                record[identity], "Obligation identity", max_integer_bits, True
            )
            if index in seen:
                raise ValueError("Duplicate obligation identity")
            seen.add(index)
            guard_id = _integer(
                record["guard_id"], "Guard identity", max_integer_bits, True
            )
            if guard_id >= len(guards):
                raise ValueError("Unknown guard identity")
            if kind == "drift":
                source = _integer(
                    record["source_vertex"], "Source vertex", max_integer_bits, True
                )
                target = _integer(
                    record["target_vertex"], "Target vertex", max_integer_bits, True
                )
                shift = tuple(
                    _integer(x, "Shift coordinate", max_integer_bits)
                    for x in _sequence(record["shift"], 6, "Shift")
                )
            else:
                source = target = _integer(
                    record["vertex"], "Target vertex", max_integer_bits, True
                )
                shift = (0,) * 6
            if source not in columns or target not in columns:
                raise ValueError("Obligation lies outside coefficient domain")
            item = Obligation(
                kind, index, len(obligations), guard_id, source, target, shift
            )
            family.append(item)
            obligations.append(item)
        families.append(tuple(family))
    spec = AdmittedSpecification(
        expected_digest,
        vertices,
        normalization,
        tuple(guards),
        root_vertex,
        root_point,
        root_minimum,
        families[0],
        families[1],
        tuple(obligations),
        MappingProxyType(columns),
    )
    _SPECIFICATIONS[spec] = (
        tuple(getattr(spec, name) for name in _SPEC_FIELDS),
        _fraction_pair(root_minimum),
        tuple(
            tuple(getattr(item, name) for name in _OBLIGATION_FIELDS)
            for item in obligations
        ),
    )
    return spec


def parse_coefficients(spec, records, *, max_bits=256):
    """Parse every sorted seven-coefficient block without accepting a prefix."""
    _specification(spec)
    _limit(max_bits, "max_bits", _MAX_BITS)
    _sequence(records, len(spec.vertices), "Coefficient records")
    flat, parsed = [], []
    for vertex, record in zip(spec.vertices, records):
        _mapping(record, ("vertex", "gradient", "offset"), "coefficient block")
        if (
            _integer(
                record["vertex"], "Coefficient vertex", _MAX_BITS, nonnegative=True
            )
            != vertex
        ):
            raise ValueError("Missing, reordered or mismatched coefficient block")
        gradient = tuple(
            parse_rational(x, max_bits=max_bits)
            for x in _sequence(record["gradient"], 6, "Gradient")
        )
        offset = parse_rational(record["offset"], max_bits=max_bits)
        parsed.append((vertex, gradient, offset))
        flat.extend((*gradient, offset))
    blocks = CoefficientBlocks(tuple(flat), tuple(parsed), max(map(_bits, flat)), spec)
    _COEFFICIENTS[blocks] = (
        tuple(getattr(blocks, name) for name in _BLOCK_FIELDS),
        tuple(map(_fraction_pair, flat)),
    )
    return blocks


def _terms(spec, obligation):
    """Single signed, translated potential expression shared by every adapter."""
    _obligation(spec, obligation)
    left = (spec._columns[obligation.source_vertex], -1, (0,) * 6)
    if obligation.kind == "target":
        return (left,)
    return (left, (spec._columns[obligation.target_vertex], 1, obligation.shift))


def affine_form(spec, blocks, obligation):
    """Return c,d for the shared obligation c dot k+d >= 0."""
    _coefficient_blocks(spec, blocks)
    linear, constant = [Fraction(0)] * 6, Fraction(0)
    for start, sign, shift in _terms(spec, obligation):
        constant += sign * _coefficient_value(blocks, start + 6)
        for i in range(6):
            value = sign * _coefficient_value(blocks, start + i) / spec.normalization
            linear[i] += value
            constant += value * shift[i]
    return tuple(linear), constant


def _point(raw, max_bits=_MAX_BITS):
    return tuple(
        _integer(x, "Point coordinate", max_bits) for x in _sequence(raw, 6, "Point")
    )


def _point_inside(bounds, point):
    full = point + (0,)
    return all(full[i] - full[j] <= bounds[i][j] for i in range(7) for j in range(7))


def _point_row(spec, terms, point):
    row = {}
    for start, sign, shift in terms:
        for i in range(6):
            row[start + i] = row.get(start + i, Fraction(0)) + Fraction(
                sign * (point[i] + shift[i]), spec.normalization
            )
        row[start + 6] = row.get(start + 6, Fraction(0)) + sign
    return tuple((i, value) for i, value in sorted(row.items()) if value)


def point_constraint(spec, obligation, point):
    """Return sparse coefficient-column weights for an admitted point cut >=0."""
    terms = _terms(spec, obligation)
    point = _point(point)
    if not _point_inside(spec.guards[obligation.guard_id], point):
        raise ValueError("Point cut is outside the complete obligation guard")
    return _point_row(spec, terms, point)


def root_constraint(spec):
    """Return the exact sparse root row and its unchanged positive minimum."""
    _specification(spec)
    return (
        _point_row(
            spec, ((spec._columns[spec.root_vertex], 1, (0,) * 6),), spec.root_point
        ),
        spec.root_minimum,
    )


def _multipliers(raw, max_bits):
    if type(raw) not in (tuple, list) or len(raw) > 42:
        raise ValueError("Expected at most 42 sparse facet multipliers")
    parsed, seen = [], set()
    for item in raw:
        i, j, weight = _sequence(item, 3, "Facet multiplier")
        if (
            type(i) is not int
            or type(j) is not int
            or not 0 <= i < 7
            or not 0 <= j < 7
            or i == j
        ):
            raise ValueError("Facet indices must be distinct exact integers in 0..6")
        if (i, j) in seen:
            raise ValueError("Duplicate facet multiplier")
        seen.add((i, j))
        weight = parse_rational(weight, max_bits=max_bits)
        if weight < 0:
            raise ValueError("Negative facet multiplier")
        parsed.append((i, j, weight))
    return tuple(parsed)


def _dual_lower(bounds, linear, constant, multipliers):
    balance, cost = [Fraction(0)] * 6, Fraction(0)
    for i, j, amount in multipliers:
        if i < 6:
            balance[i] += amount
        if j < 6:
            balance[j] -= amount
        cost += amount * bounds[i][j]
    if tuple(balance) != tuple(-x for x in linear):
        raise ValueError("Facet dual does not balance the affine coefficients")
    return constant - cost


@dataclass(frozen=True, slots=True)
class _BoundsOnlyZone:
    bounds: tuple[tuple[int, ...], ...]


def _exact_minimum(bounds, linear, constant, max_bits):
    if not __debug__:
        raise RuntimeError(
            "The exact extrema owner requires enabled assertions; optimized Python is unsupported"
        )
    from ..physics.c6_carried_excursion import _linear_extremum

    linear = tuple(
        parse_rational(x, max_bits=max_bits)
        for x in _sequence(linear, 6, "Linear form")
    )
    constant = parse_rational(constant, max_bits=max_bits)
    denominator = 1
    for value in linear:
        denominator = math.lcm(denominator, value.denominator)
        _integer(denominator, "Scaled common denominator", max_bits)
    integers = tuple(int(value * denominator) for value in linear)
    divisor = math.gcd(*(abs(value) for value in integers))
    factor = Fraction(divisor, denominator) if divisor else Fraction(1)
    weights = tuple(value // divisor for value in integers) if divisor else (0,) * 6
    parse_rational(factor, max_bits=max_bits)
    for weight in weights:
        _integer(weight, "Scaled objective weight", max_bits)
    if factor <= 0 or any(
        value != factor * weight for value, weight in zip(linear, weights)
    ):
        raise RuntimeError("Exact objective scaling failed")
    owner = _linear_extremum(_BoundsOnlyZone(bounds), weights, "lower")
    if type(owner.value) is not int:
        raise RuntimeError("Exact owner returned a noninteger objective")
    point = _point(owner.point, _MAX_BITS)
    if not _point_inside(bounds, point):
        raise RuntimeError("Exact owner primal left the admitted DBM")
    if any(
        type(amount) is not int or amount <= 0 for _i, _j, amount in owner.dual_flow
    ):
        raise RuntimeError(
            "Exact owner returned a nonpositive or noninteger transport amount"
        )
    multipliers = _multipliers(
        [(i, j, factor * amount) for i, j, amount in owner.dual_flow], max_bits
    )
    minimum = constant + factor * owner.value
    if (
        constant + sum(value * coordinate for value, coordinate in zip(linear, point))
        != minimum
    ):
        raise RuntimeError("Exact owner primal does not attain its claimed minimum")
    if _dual_lower(bounds, linear, constant, multipliers) != minimum:
        raise RuntimeError("Exact owner primal and dual objectives differ")
    signed = tuple(-value for value in weights) + (sum(weights),)
    sources, sinks = [i for i, x in enumerate(signed) if x > 0], [
        i for i, x in enumerate(signed) if x < 0
    ]
    possible = len(sources) * len(sinks)
    if (
        possible > 12
        or len(multipliers) > possible
        or any(
            not (signed[i] > 0 > signed[j] and value > 0) for i, j, value in multipliers
        )
    ):
        raise RuntimeError("Exact owner lost the signed transport support")
    return dict(
        coefficients=list(map(str, linear)),
        constant=str(constant),
        primitive_integer_weights=weights,
        positive_rational_weight_scale=str(factor),
        owner_integer_minimum=owner.value,
        minimum=str(minimum),
        attaining_point=point,
        multipliers=[(i, j, str(value)) for i, j, value in multipliers],
        nonnegative_certified=minimum >= 0,
        negative_attaining_witness=minimum < 0,
        exact_primal_dual_equality_checked=True,
        local_lower_bound_checker_passed=True,
        transport_positive_indices=sources,
        transport_negative_indices=sinks,
        possible_transport_edges=possible,
        used_transport_edges=len(multipliers),
        existing_integer_extremum_owner_calls=1,
        existing_min_cost_flow_calls=int(any(weights)),
        max_rational_bits=max_bits,
        scope="One fixed affine objective on one closed DBM; no history-barrier or origin-reachability claim",
    )


def exact_affine_minimum(bounds, coefficients, constant, *, max_bits=4096):
    """Standalone exact local adapter; validate geometry before using the owner."""
    _limit(max_bits, "max_bits", _MAX_BITS)
    bounds = _bounds(bounds, max_bits)
    _closed(bounds)
    return _exact_minimum(bounds, coefficients, constant, max_bits)


def exact_minimum(spec, obligation, blocks, *, max_bits=4096):
    """Reuse one admitted guard and the same affine transform as point/final checks."""
    _limit(max_bits, "max_bits", _MAX_BITS)
    linear, constant = affine_form(spec, blocks, obligation)
    return _exact_minimum(spec.guards[obligation.guard_id], linear, constant, max_bits)


def verify_candidate(spec, candidate, *, max_bits=4096, collect_failures=False):
    """Explicitly verify every complete root, drift and target certificate.

    Malformed/incomplete proof data always raise, even when collecting
    mathematical failures. An admitted but insufficient dual produces a scoped
    failure. No closure is recomputed after specification admission.
    """
    _specification(spec)
    _limit(max_bits, "max_bits", _MAX_BITS)
    if type(collect_failures) is not bool:
        raise ValueError("collect_failures must be boolean")
    _mapping(
        candidate,
        (
            "schema_version",
            "specification_sha256",
            "coefficients",
            "drift_proofs",
            "target_proofs",
        ),
        "candidate",
    )
    if type(candidate["schema_version"]) is not int or candidate["schema_version"] != 1:
        raise ValueError("Unsupported candidate schema")
    candidate_digest = candidate["specification_sha256"]
    if (
        type(candidate_digest) is not str
        or not _DIGEST.fullmatch(candidate_digest)
        or candidate_digest != spec.digest
    ):
        raise ValueError("Candidate belongs to a different specification")
    parse_rational(spec.root_minimum, max_bits=max_bits)
    blocks = parse_coefficients(spec, candidate["coefficients"], max_bits=max_bits)
    parsed_proofs, terms, largest = (
        [],
        0,
        max(blocks.largest_bits, _bits(spec.root_minimum)),
    )
    for family, key in (
        (spec.drift_obligations, "drift_proofs"),
        (spec.target_obligations, "target_proofs"),
    ):
        proofs = _sequence(candidate[key], len(family), key)
        for obligation, proof in zip(family, proofs):
            _obligation(spec, obligation)
            identity = "arc_index" if obligation.kind == "drift" else "piece_index"
            _mapping(proof, (identity, "multipliers"), "obligation proof")
            if (
                _integer(proof[identity], "Proof identity", _MAX_BITS, nonnegative=True)
                != obligation.index
            ):
                raise ValueError("Missing, reordered or mismatched obligation proof")
            multipliers = _multipliers(proof["multipliers"], max_bits)
            parsed_proofs.append(multipliers)
            terms += len(multipliers)
            largest = max([largest, *(_bits(value) for _i, _j, value in multipliers)])
    failures = []

    def failure(kind, index, reason):
        if not collect_failures:
            raise ValueError(f"{kind} obligation {index}: {reason}")
        failures.append(
            dict(
                kind=kind,
                **({"vertex": index} if kind == "root" else {"index": index}),
                reason=reason,
            )
        )

    root_row, minimum = root_constraint(spec)
    root_value = sum(
        value * _coefficient_value(blocks, column) for column, value in root_row
    )
    if root_value < minimum:
        failure("root", spec.root_vertex, "Root positivity failed")
    for obligation, multipliers in zip(spec.obligations, parsed_proofs):
        linear, constant = affine_form(spec, blocks, obligation)
        try:
            lower = _dual_lower(
                spec.guards[obligation.guard_id], linear, constant, multipliers
            )
        except ValueError as error:
            failure(obligation.kind, obligation.index, str(error))
        else:
            if lower < 0:
                failure(
                    obligation.kind,
                    obligation.index,
                    "The supplied affine lower bound is negative",
                )
    return dict(
        accepted=not failures,
        specification_sha256=spec.digest,
        root_value=str(root_value),
        required_root_minimum=str(minimum),
        coefficient_blocks=len(spec.vertices),
        coefficient_count=spec.coefficient_count,
        distinct_guard_count=len(spec.guards),
        checked_root_obligations=1,
        checked_drift_obligations=len(spec.drift_obligations),
        checked_target_obligations=len(spec.target_obligations),
        checked_sparse_dual_terms=terms,
        parsed_rational_inputs=spec.coefficient_count + 1 + terms,
        largest_rational_input_bits=largest,
        max_rational_bits=max_bits,
        failures=failures,
        original_target_coverage_verified=False,
        whole_C6_first_exit_label_excluded=False,
        indefinite_boundedness_certified=False,
        scope="Exact obligations for the separately validated specification only",
    )
