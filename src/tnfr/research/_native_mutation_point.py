"""Exact readmission of one proposed two-call Mutation source.

Primitive parameters and four source pressure coordinates are rebuilt through
the existing complete-law compiler. This owner neither proposes sources nor
executes native dynamics. A shared offset associates all fifteen form values;
the first secant is computed from those associated endpoint samples. Both
actual controller predicates use one fixed rational cosine-enclosure policy.

Admission concerns ideal common-lift dynamics only. In particular, a solver
flag, cached certificate, algebraic negative zero or supplied temporal rate
cannot replace the exact source checks and associated ideal-form secant below.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import _ordered
from ..mathematics._phase_resultant_chamber import (
    COSINE_ENCLOSURE_METHOD,
    certified_cosine_bounds,
)
from ._native_mutation_feasibility import (
    _NEXT_PEAKS,
    _SOURCE_PEAKS,
    CellKey,
    ControllerObligation,
    ExecutionPremises,
    NativeMutationLaw,
    NativeMutationSlice,
    ParameterCheck,
    _cell,
    _prepare,
)

Q = Fraction
COSINE_TERMS = 12
COSINE_BITS = 128


@dataclass(frozen=True, slots=True)
class PointRowCheck:
    """One rebuilt canonical affine row at the proposed exact point."""

    label: str
    lhs: Fraction
    rhs: Fraction
    strict: bool
    unit: str

    @property
    def residual(self) -> Fraction:
        """Signed lhs-rhs; a strict row requires a negative residual."""
        return self.lhs - self.rhs

    @property
    def satisfied(self) -> bool:
        return self.lhs < self.rhs if self.strict else self.lhs <= self.rhs


@dataclass(frozen=True, slots=True)
class ControllerPointCheck:
    """A fixed-work interval decision for one associated controller."""

    obligation: ControllerObligation
    cosine_enclosures: tuple[tuple[Fraction, Fraction], ...]
    r_squared_interval: tuple[Fraction, Fraction]
    lower_status: str
    upper_status: str
    status: str
    terms: int = COSINE_TERMS
    bits: int = COSINE_BITS
    method: str = COSINE_ENCLOSURE_METHOD


@dataclass(frozen=True, slots=True)
class NativeMutationPoint:
    """A primitive-associated ideal point result, never native evidence."""

    spec: NativeMutationSlice
    pressure_coordinates: tuple[Fraction, ...]
    law: NativeMutationLaw
    premises: ExecutionPremises
    pressure: tuple[Fraction, ...]
    successor_pressure: tuple[Fraction, ...]
    pressure_peaks: tuple[Fraction, Fraction]
    source_peak_owners: tuple[tuple[int, int], ...]
    successor_peak_owners: tuple[tuple[int, int], ...]
    selected_cell_key: CellKey | None
    cell_unavailability: tuple[str, ...]
    form_offsets: tuple[tuple[Fraction, ...], ...]
    form_boundaries: tuple[tuple[Fraction, ...], ...]
    common_offset_interval: tuple[Fraction, Fraction]
    common_offset: Fraction
    temporal_histories: tuple[tuple[tuple[Fraction, Fraction], ...], ...]
    first_secants: tuple[Fraction, ...]
    first_phase: tuple[Fraction, ...]
    mutation_phase: tuple[Fraction, ...]
    terminal_phase: tuple[Fraction, ...]
    row_checks: tuple[PointRowCheck, ...]
    parameter_checks: tuple[ParameterCheck, ...]
    controller_checks: tuple[ControllerPointCheck, ControllerPointCheck]
    failed_checks: tuple[str, ...]
    unresolved_checks: tuple[str, ...]
    status: str
    obligations: tuple[str, ...]


def _controller_status(obligation, interval):
    """Decide already-owned interval predicates without choosing a point."""
    lower, upper = interval
    if lower > upper:
        raise ValueError("controller enclosure endpoints must be ordered")
    bound = obligation.lower_squared
    if bound is None:
        lower_status = "not_required"
    elif lower > bound if obligation.lower_strict else lower >= bound:
        lower_status = "certified"
    elif upper <= bound if obligation.lower_strict else upper < bound:
        lower_status = "rejected"
    else:
        lower_status = "unresolved"
    bound = obligation.upper_squared
    if bound is None:
        upper_status = "not_required"
    elif upper < bound if obligation.upper_strict else upper <= bound:
        upper_status = "certified"
    elif lower >= bound if obligation.upper_strict else lower > bound:
        upper_status = "rejected"
    else:
        upper_status = "unresolved"
    statuses = (lower_status, upper_status)
    status = (
        "rejected"
        if "rejected" in statuses
        else "unresolved" if "unresolved" in statuses else "certified"
    )
    return lower_status, upper_status, status


def _check_controller(obligation):
    bounds = tuple(
        certified_cosine_bounds(angle, terms=COSINE_TERMS, bits=COSINE_BITS)
        for angle in obligation.phase_differences
    )
    interval = tuple(
        obligation.constant
        + obligation.cosine_coefficient * sum(row[i] for row in bounds)
        for i in (0, 1)
    )
    return ControllerPointCheck(
        obligation, bounds, interval, *_controller_status(obligation, interval)
    )


def check_native_mutation_point(
    spec: NativeMutationSlice, pressure_coordinates: Iterable[Any]
) -> NativeMutationPoint:
    """Rebuild and check one exact-or-represented primitive proposal.

    The canonical source pressure is zero-sum. All source/endpoint forms
    share the midpoint of one fifteen-slot offset intersection; an empty
    intersection is retained as a rejection, never repaired. Source and
    successor peak ties are retained in canonical order. A wrongly signed
    target or missing allowed peak owner leaves the cell explicitly
    unavailable. Controller enclosures are evaluated once at fixed precision
    even for an otherwise rejected point, without numerical refinement.
    """
    values = _ordered(pressure_coordinates, "source pressure coordinates")
    if len(values) != 4:
        raise ValueError("exactly four source pressure coordinates are required")
    supplied = tuple(
        exact_or_represented_real(value, f"source pressure coordinate {i}")
        for i, value in enumerate(values)
    )
    prepared = _prepare(spec)
    (
        normalized,
        law,
        premises,
        pressure_rows,
        successor_rows,
        boundary_rows,
        first,
        mutated,
        terminal,
        controllers,
        geometry,
    ) = prepared
    if any(row.coefficients[4] for row in (*pressure_rows, *successor_rows)) or any(
        row.coefficients[4] != 1 for boundary in boundary_rows for row in boundary
    ):
        raise ValueError("the canonical shared-offset dependence changed")
    coordinates = (*supplied, Q(0))

    def evaluate(expression):
        return expression.constant + sum(
            value * weight
            for value, weight in zip(coordinates, expression.coefficients)
        )

    pressure = tuple(map(evaluate, pressure_rows))
    successor = tuple(map(evaluate, successor_rows))
    if pressure != (*supplied, -sum(supplied)):
        raise RuntimeError("the canonical source pressure lost exact zero sum")
    offsets = tuple(tuple(map(evaluate, rows)) for rows in boundary_rows)
    flat_offsets = tuple(value for boundary in offsets for value in boundary)
    lower = law.form_rails[0] - min(flat_offsets)
    upper = law.form_rails[1] - max(flat_offsets)
    offset = (lower + upper) / 2
    forms = tuple(tuple(value + offset for value in boundary) for boundary in offsets)
    histories = tuple(
        tuple((time, forms[j][i]) for j, time in enumerate(premises.times))
        for i in range(5)
    )
    secants = tuple(
        (history[1][1] - history[0][1]) / (history[1][0] - history[0][0])
        for history in histories
    )
    peaks = max(map(abs, pressure)), max(map(abs, successor))
    source_owners = tuple(
        owner
        for owner in _SOURCE_PEAKS
        if peaks[0] > 0 and owner[1] * pressure[owner[0]] == peaks[0]
    )
    next_owners = tuple(
        owner
        for owner in _NEXT_PEAKS
        if peaks[1] > 0 and owner[1] * successor[owner[0]] == peaks[1]
    )
    unavailable = tuple(
        label
        for label, condition in (
            ("cell.source_target_not_positive", pressure[0] <= 0),
            ("cell.successor_target_not_negative", successor[0] >= 0),
            ("cell.source_peak_owner_unavailable", not source_owners),
            ("cell.successor_peak_owner_unavailable", not next_owners),
        )
        if condition
    )
    key = None if unavailable else CellKey(source_owners[0], next_owners[0])
    coordinates = (*supplied, offset)
    rows = () if key is None else _cell(prepared, key).rows
    row_checks = tuple(
        PointRowCheck(
            row.label,
            sum(value * weight for value, weight in zip(coordinates, row.coefficients)),
            row.rhs,
            row.strict,
            row.unit,
        )
        for row in rows
    )
    checks = (
        *geometry,
        ParameterCheck("point.common_offset", lower, upper, "<="),
        ParameterCheck("point.source_pressure", law.source_tiny, pressure[0], "<"),
        ParameterCheck("point.source_peak", Q(0), peaks[0], "<"),
        ParameterCheck("point.successor_pressure", successor[0], Q(0), "<"),
        ParameterCheck("point.successor_peak", Q(0), peaks[1], "<"),
        ParameterCheck("point.acquired_secant", law.xi, secants[0], "<"),
    )
    controller_checks = tuple(map(_check_controller, controllers))
    failures = (
        *unavailable,
        *(check.label for check in checks if not check.satisfied),
        *(check.label for check in row_checks if not check.satisfied),
        *(
            f"controller.{check.obligation.step}"
            for check in controller_checks
            if check.status == "rejected"
        ),
    )
    unresolved = tuple(
        f"controller.{check.obligation.step}"
        for check in controller_checks
        if check.status == "unresolved"
    )
    status = (
        "rejected"
        if failures
        else "unresolved" if unresolved else "conditionally_admitted_ideal"
    )
    return NativeMutationPoint(
        normalized,
        supplied,
        law,
        premises,
        pressure,
        successor,
        peaks,
        source_owners,
        next_owners,
        key,
        unavailable,
        offsets,
        forms,
        (lower, upper),
        offset,
        histories,
        secants,
        first,
        mutated,
        terminal,
        row_checks,
        checks,
        controller_checks,
        failures,
        unresolved,
        status,
        (
            "all algebra, fields, histories and controller bounds use one freshly rebuilt primitive point",
            "an admitted point concerns ideal common-lift dynamics with the supplied initial grammar context",
            "the associated ideal secant is not measured native temporal evidence",
            "strictly negative successor target pressure excludes algebraic signed-zero substitution",
            "native wrapping, execution, persistence, formation and physical identification remain separate",
            "a rejected or unresolved point does not exclude other preparations or the complete domain",
        ),
    )
