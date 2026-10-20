"""Exact conditional elimination of all successor pressures in the C5 study.

This private instrument retains the same primitive parameter and execution
domain as the source-feasibility compiler. Let L=N-I, tau=dt*k, and
D=diag(f*v0,ell*v1,...,ell*v4). The complete first pressure transport is
q=(I+tau*L*D)*p+d*L*(C1-I)*theta. When all five q vanish, the pressure
vector is unique: p=-(I+tau*L*D)^(-1)*d*L*(C1-I)*theta.

The single owned bound 2*tau*VF_MAX*max(f,ell)<1 makes this transport matrix
nonnegative and column-stochastic, with positive strict column diagonal
dominance. If its transpose annihilated a nonzero vector, a coordinate of
maximal absolute value would contradict that strict dominance. Thus the
matrix is invertible. Its inverse is not asserted to be nonnegative.

With q=0 the second held flow is zero, so x2=x1. The first-endpoint
potential k*x1+b*v+d*C1*theta is constant because the connected C5 has
only constants in ker(L). The ten source/first-endpoint offset slots determine
one closed common form-offset interval. The returned offset uses the ORIGINAL
compiler gauge, not a separately chosen endpoint-potential constant.

All nine canonical zero-pressure source-peak cells are rebuilt and their
strict/weak rows retained after substitution. Only the thirty weak form
rails may depend on the offset. Actual controller predicates, phase
geometry, grammar/history and negative-zero realization are not supplied
by algebraic compatibility. No feasibility optimizer, parameter search, native response,
equilibrium inference or evaluation of a controller resultant occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from ..mathematics._exact_linear_algebra import exact_matrix_inverse
from ._native_mutation_feasibility import (
    _PI,
    _SOURCE_PEAKS,
    CellKey,
    ControllerObligation,
    ExecutionPremises,
    NativeMutationLaw,
    NativeMutationSlice,
    ParameterCheck,
    _cell,
    _gains,
    _owned_law,
    _prepare,
)

Q = Fraction


@dataclass(frozen=True, slots=True)
class ZeroPressureInvertibility:
    law: NativeMutationLaw
    premises: ExecutionPremises
    maximum_flow_diagonal: Fraction
    maximum_pressure_step: Fraction
    column_dominance_margin: Fraction
    dimension: int = 5
    scope: str = "owned full-capacity-rail pressure transport; not inverse positivity"


@dataclass(frozen=True, slots=True)
class OffsetConstraint:
    """offset_coefficient*eta <= rhs, or < rhs when strict is true."""

    label: str
    offset_coefficient: Fraction
    rhs: Fraction
    strict: bool
    unit: str


@dataclass(frozen=True, slots=True)
class ZeroPressureCell:
    key: CellKey
    rows: tuple[OffsetConstraint, ...]
    parameter_checks: tuple[ParameterCheck, ...]
    status: str
    zero_pressure_direction: str


@dataclass(frozen=True, slots=True)
class ZeroPressureReduction:
    spec: NativeMutationSlice
    law: NativeMutationLaw
    premises: ExecutionPremises
    invertibility: ZeroPressureInvertibility
    transport_matrix: tuple[tuple[Fraction, ...], ...]
    transport_inverse: tuple[tuple[Fraction, ...], ...]
    phase_forcing: tuple[Fraction, ...]
    source_pressure: tuple[Fraction, ...]
    successor_pressure: tuple[Fraction, ...]
    form_offsets: tuple[tuple[Fraction, ...], ...]
    offset_interval: tuple[Fraction, Fraction]
    offset_check: ParameterCheck
    first_rates: tuple[Fraction, ...]
    second_rates: tuple[Fraction, ...]
    first_phase: tuple[Fraction, ...]
    mutation_phase: tuple[Fraction, ...]
    terminal_phase: tuple[Fraction, ...]
    controller_obligations: tuple[ControllerObligation, ...]
    phase_geometry: tuple[ParameterCheck, ...]
    cells: tuple[ZeroPressureCell, ...]
    algebraic_status: str
    obligations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ZeroPressureObstructionCell:
    """One universal rate ceiling, never a jointly feasible source."""

    first_branch: str
    second_branch: str
    first_gains: tuple[Fraction, Fraction]
    second_gains: tuple[Fraction, Fraction]
    phase_matrix: tuple[tuple[Fraction, ...], ...]
    phase_inverse: tuple[tuple[Fraction, ...], ...]
    forcing_matrix: tuple[tuple[Fraction, ...], ...]
    forcing_oscillation_norm: Fraction
    gap_ceiling: Fraction
    ratio_ceiling: Fraction
    capacity_quadratic: tuple[Fraction, Fraction, Fraction]
    capacity_vertex: Fraction
    capacity_numerator_max: Fraction
    capacity_ceiling: Fraction
    phase_diameter_ceiling: Fraction
    pressure_peak_ceiling: Fraction
    target_forcing_ceiling: Fraction
    target_denominator: Fraction
    rate_ceiling: Fraction
    threshold_gap: Fraction
    status: str


@dataclass(frozen=True, slots=True)
class ZeroPressureObstruction:
    """Fresh coefficients of the frozen six-pair whole-zero-pressure bound."""

    law: NativeMutationLaw
    premises: ExecutionPremises
    capacity_floor: Fraction
    zero_ratio_intercept: Fraction
    zero_ratio_slope: Fraction
    capacity_order_margin: Fraction
    endpoint_arc_bound: Fraction
    maximum_flow_diagonal: Fraction
    pressure_dominance_margin: Fraction
    cells: tuple[ZeroPressureObstructionCell, ...]
    obligations: tuple[str, ...]


def derive_native_mutation_zero_obstruction() -> ZeroPressureObstruction:
    """Evaluate the single frozen all-zero-pressure bound, without sources.

    The complete q=0 equation gives A*p=d*(C1^-1-I)*L*y. Its forcing
    therefore depends only on the associated first-phase image, whereas
    arbitrary source pressures also contain independent form/capacity terms.
    Endpoint potential, zero-ratio masks and renewal bound capacity and
    phase. The final scalar ceilings discard correlations: an inconclusive
    cell does not admit any joint state. No refinement or optimizer is used.
    """
    law, premises = _owned_law()
    k, b, w = law.pressure
    alpha, beta, gamma = law.stability
    d = w / _PI
    f, ell = law.oz_factor, law.il_factor
    maximum, shift = law.capacity_rails[1], law.mutation_shift
    tau = premises.dt * k
    floor = (law.si_high - beta - gamma) / alpha
    intercept = (law.si_low - beta - gamma) / alpha
    slope = beta / (alpha * _PI)
    ordering = (law.si_high - law.si_low - beta) / alpha
    if not (intercept >= 0 and slope > 0 and ordering > 0):
        raise ValueError("the zero-mask capacity-ordering domain changed")
    arc = 2 * k / d + shift
    # Since |L*y|/pi < 1, the zero-q target mask also bounds u/V.
    target_ratio = (law.si_low - gamma) / alpha
    diagonal = maximum * (ell if f * target_ratio <= ell else max(f, ell))
    margin = 1 - 2 * tau * diagonal
    if margin <= 0:
        raise ValueError("the zero-mask pressure dominance domain changed")
    laplacian = tuple(
        tuple(
            -Q(1) if i == j else Q(1, 2) if (i - j) % 5 in (1, 4) else Q(0)
            for j in range(5)
        )
        for i in range(5)
    )
    cells = []
    for first, second in product(
        ("stable", "transition", "dissonant"), ("stable", "transition")
    ):
        gains1 = _gains(law, law.phase_gains, first)
        gains2 = _gains(law, gains1, second)
        g, local = gains1
        g2, local2 = gains2
        a, a2 = 1 - g - local, 1 - g2 - local2
        if not (min(g, local, g2, local2) > 0 and a >= local / 2 and a2 >= local2 / 2):
            raise ValueError("the positive phase-renewal domain changed")
        matrix = tuple(
            tuple(
                (1 - g if i == j else Q(0)) + local * laplacian[i][j] for j in range(5)
            )
            for i in range(5)
        )
        inverse = exact_matrix_inverse(matrix)
        forcing = tuple(
            tuple(
                sum((inverse[i][n] - Q(i == n)) * laplacian[n][j] for n in range(5))
                for j in range(5)
            )
            for i in range(5)
        )
        if any(sum(row) != 0 for row in forcing) or any(
            forcing[i][j] != forcing[0][(j - i) % 5] for i in range(5) for j in range(5)
        ):
            raise RuntimeError("the exact phase forcing lost its circulant identity")
        norm = sum(max(value, Q(0)) for value in forcing[0])
        gap = min(a2 * shift / (g2 + local2), 2 * k / d)
        ratio = intercept + slope * gap
        if not floor > ratio:
            raise ValueError("the joint zero-mask capacity separation is unavailable")
        quadratic = (2 * k * intercept, 2 * k * slope - d * intercept, -d * slope)
        c0, c1, c2 = quadratic
        vertex = min(gap, max(Q(0), -c1 / (2 * c2)))
        numerator = max(c0 + c1 * z + c2 * z * z for z in (Q(0), gap, vertex))
        capacity = min(maximum * ratio, numerator / (b * (floor - ratio)))
        diameter = min(a2 * shift / (g2 + local2 / 2), (1 - g2) * arc)
        peak = d * norm * diameter / margin
        determinant = a * (a - local / 2) - local * local / 4
        if determinant <= 0:
            raise ValueError("the grouped first-phase inverse is unavailable")
        ch, ct = a / determinant - 1, local / (2 * determinant)
        if min(ch, ct) < 0:
            raise ValueError("the target forcing has left its monotone domain")
        target_forcing = d * (ch * gap + ct * a2 * (shift + gap))
        denominator = 1 - tau * f * capacity
        if denominator <= 0:
            raise ValueError("the target pressure division is unavailable")
        rate = (
            f * capacity * (target_forcing + tau * ell * maximum * peak) / denominator
        )
        cells.append(
            ZeroPressureObstructionCell(
                first,
                second,
                gains1,
                gains2,
                matrix,
                inverse,
                forcing,
                norm,
                gap,
                ratio,
                quadratic,
                vertex,
                numerator,
                capacity,
                diameter,
                peak,
                target_forcing,
                denominator,
                rate,
                law.xi - rate,
                "excluded" if rate <= law.xi else "inconclusive",
            )
        )
    return ZeroPressureObstruction(
        law,
        premises,
        floor,
        intercept,
        slope,
        ordering,
        arc,
        diagonal,
        margin,
        tuple(cells),
        (
            "all five successor pressures vanish under the complete associated law",
            "both actual masks, common endpoint potential and source/endpoint rails hold",
            "negative-outward target-minimum complete renewal uses the same phase lifts",
            "the six retained controller pairs still require their actual predicates",
            "a rate ceiling above xi is inconclusive, not a source or feasibility witness",
            "independent maxima discard correlations; no adaptive refinement is performed",
            "ideal predicted rates do not supply native history or a negative zero sign bit",
        ),
    )


def _invertibility(law, premises):
    maximum = law.capacity_rails[1] * max(law.oz_factor, law.il_factor)
    step = premises.dt * law.pressure[0] * maximum
    margin = 1 - 2 * step
    if not (0 < step and margin > 0):
        raise ValueError("owned rails leave the strict pressure-invertibility domain")
    return ZeroPressureInvertibility(law, premises, maximum, step, margin)


def derive_native_mutation_zero_invertibility() -> ZeroPressureInvertibility:
    """Check the universal coefficient bound from a fresh owned default law."""
    law, premises = _owned_law()
    return _invertibility(law, premises)


def _common_offset_interval(offsets, rails):
    """Intersect common-offset rails on already normalized coefficients."""
    return rails[0] - min(offsets), rails[1] - max(offsets)


def reduce_native_mutation_zero_pressure(
    spec: NativeMutationSlice,
) -> ZeroPressureReduction:
    """Eliminate pressures and one offset in the nine original zero cells.

    Primitive parameters are re-admitted and the law is rebuilt exactly
    once. Passing scalar rows means only compatibility with these affine
    cells; retained geometry and controller obligations are separate.
    """
    prepared = _prepare(spec)
    (
        normalized,
        law,
        premises,
        _,
        successor,
        boundaries,
        first,
        mutated,
        terminal,
        controller_obligations,
        geometry,
    ) = prepared
    certificate = _invertibility(law, premises)
    diagonal = tuple(
        capacity * (law.oz_factor if i == 0 else law.il_factor)
        for i, capacity in enumerate(normalized.capacities)
    )
    tau = premises.dt * law.pressure[0]
    matrix = tuple(
        tuple(
            Q(i == j)
            + tau
            * diagonal[j]
            * (-1 if i == j else Q(1, 2) if (i - j) % 5 in (1, 4) else 0)
            for j in range(5)
        )
        for i in range(5)
    )
    inverse = exact_matrix_inverse(matrix)
    forcing = tuple(expression.constant for expression in successor)
    pressure = tuple(-sum(a * b for a, b in zip(row, forcing)) for row in inverse)
    coordinates = (*pressure[:4], Q(0))

    def substitute(expression):
        return expression.constant + sum(
            a * b for a, b in zip(expression.coefficients, coordinates)
        )

    next_pressure = tuple(substitute(expression) for expression in successor)
    offsets = tuple(tuple(map(substitute, boundary)) for boundary in boundaries)
    if sum(pressure) != 0 or any(next_pressure) or offsets[2] != offsets[1]:
        raise RuntimeError(
            "the exact pressure elimination failed its transport identity"
        )
    unique_offsets = (*offsets[0], *offsets[1])
    lower, upper = _common_offset_interval(unique_offsets, law.form_rails)
    offset_check = ParameterCheck("common_form_offset", lower, upper, "<=")
    cells = []
    for source_peak in _SOURCE_PEAKS:
        original = _cell(prepared, CellKey(source_peak, None))
        rows = tuple(
            OffsetConstraint(
                row.label,
                row.coefficients[4],
                row.rhs
                - sum(a * b for a, b in zip(row.coefficients[:4], pressure[:4])),
                row.strict,
                row.unit,
            )
            for row in original.rows
        )
        rails = tuple(row for row in rows if row.label.startswith("rail."))
        scalar_rows = tuple(row for row in rows if not row.label.startswith("rail."))
        if (
            len(rails) != 30
            or any(
                row.strict
                or row.unit != "form"
                or row.offset_coefficient not in (-1, 1)
                for row in rails
            )
            or any(row.offset_coefficient != 0 for row in scalar_rows)
        ):
            raise ValueError(
                "canonical offset rows leave the supported weak-rail structure"
            )
        rail_lower = max(-row.rhs for row in rails if row.offset_coefficient == -1)
        rail_upper = min(row.rhs for row in rails if row.offset_coefficient == 1)
        if (rail_lower, rail_upper) != (lower, upper):
            raise RuntimeError(
                "the original rails disagree with the shared offset interval"
            )
        checks = tuple(
            ParameterCheck(row.label, Q(0), row.rhs, "<" if row.strict else "<=")
            for row in scalar_rows
        )
        compatible = offset_check.satisfied and all(check.satisfied for check in checks)
        cells.append(
            ZeroPressureCell(
                original.key,
                rows,
                checks,
                "algebraically_compatible" if compatible else "algebraically_excluded",
                original.zero_pressure_direction,
            )
        )
    first_rates = tuple(value * p for value, p in zip(diagonal, pressure))
    second_rates = tuple(
        capacity * (1 if i == 0 else law.il_factor) * p
        for i, (capacity, p) in enumerate(zip(normalized.capacities, next_pressure))
    )
    return ZeroPressureReduction(
        normalized,
        law,
        premises,
        certificate,
        matrix,
        inverse,
        forcing,
        pressure,
        next_pressure,
        offsets,
        (lower, upper),
        offset_check,
        first_rates,
        second_rates,
        first,
        mutated,
        terminal,
        controller_obligations,
        geometry,
        tuple(cells),
        (
            "algebraically_compatible"
            if any(cell.status == "algebraically_compatible" for cell in cells)
            else "algebraically_excluded"
        ),
        (
            "exact elimination concerns only the canonical nine all-zero-pressure cells",
            "actual phase/controller predicates and complete geometry remain separate obligations",
            "one shared compiler offset must associate every source and endpoint form/history",
            "ideal first-flow secants are conditional predictions, not acquired native observations",
            "algebraic zero does not prove a negative binary64 sign bit or Mutation direction",
            "a zero held second flow is not equilibrium, formation or complete-source admission",
        ),
    )
