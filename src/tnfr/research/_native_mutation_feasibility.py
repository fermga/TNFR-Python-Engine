"""Conditional pressure systems for the two-call Mutation research question.

This private research instrument compiles Dissonance (OZ)/Coherence (IL),
then Mutation (ZHIR)/Coherence (IL), on the declared paired unit five-cycle.
It does not select parameters, solve a system, evaluate a source or execute
an operator. A compiled cell is conditional on its retained phase/controller
obligations and complete execution premises. No successful compilation proves
source existence. In particular, algebraic zero does not realize negative
binary64 zero or establish the direction of a native Mutation.

The arithmetic model is ideal real common-lift dynamics with the exact
normalization of represented default coefficients. It is not bit-exact native
trigonometry, modular phase arithmetic or a physical-unit identification.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from typing import Any

from .._exact_time import exact_or_represented_real
from ..config.constants import DISRUPTORS
from ..config.defaults_core import CoreDefaults
from ..config.operator_names import BIFURCATION_WINDOW, U2_DEBT_CAPACITY
from ..mathematics._exact_linear_algebra import _ordered
from ..operators.grammar_types import (
    BIFURCATION_HANDLERS,
    DESTABILIZERS,
    FUNCTION_TO_GLYPH,
    STABILIZERS,
)

Q = Fraction
VARIABLES = ("p0", "p1", "p2", "p3", "eta")
BRANCHES = ("stable", "transition", "dissonant")
_SOURCE_PEAKS = ((0, 1),) + tuple(product(range(1, 5), (-1, 1)))
_NEXT_PEAKS = tuple(product(range(1, 5), (-1, 1)))
_DT, _PI = Q(1, 16), Q(math.pi)
# The direct OZ implementation owns this literal cutoff (operators._op_OZ).
# It is a fixed premise of this compiler, not a configurable replacement.
_OZ_TINY = Q(1e-9)


@dataclass(frozen=True, slots=True)
class NativeMutationSlice:
    """Primitive parameters, re-admitted each time a consumer uses them."""

    capacities: tuple[Any, ...]
    phases: tuple[Any, ...]
    first_branch: str
    second_branch: str


@dataclass(frozen=True, slots=True)
class CellKey:
    source_peak: tuple[int, int]
    successor_peak: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class AffineExpression:
    """Constant plus coefficients in the fixed order VARIABLES."""

    coefficients: tuple[Fraction, ...]
    constant: Fraction = Q(0)


@dataclass(frozen=True, slots=True)
class LinearConstraint:
    """coefficients dot VARIABLES <= rhs, or < rhs when strict is true."""

    label: str
    coefficients: tuple[Fraction, ...]
    rhs: Fraction
    strict: bool
    unit: str


@dataclass(frozen=True, slots=True)
class ParameterCheck:
    """A recomputed algebraic parameter obligation, never caller testimony."""

    label: str
    lhs: Fraction
    rhs: Fraction
    relation: str

    @property
    def satisfied(self) -> bool:
        return self.lhs < self.rhs if self.relation == "<" else self.lhs <= self.rhs


@dataclass(frozen=True, slots=True)
class ControllerObligation:
    """Unevaluated R-squared predicate with an exact cosine expression.

    R_squared = constant + cosine_coefficient * sum(cos(angle) for angle in
    phase_differences). Bounds refer to that expression, not to an accepted
    observation. The loads here are below disr_lo, so the native stable-first
    priority gives these disjoint closed/strict R boundaries.
    """

    step: int
    branch: str
    phase_differences: tuple[Fraction, ...]
    constant: Fraction
    cosine_coefficient: Fraction
    lower_squared: Fraction | None
    upper_squared: Fraction | None
    lower_strict: bool
    upper_strict: bool
    disruptor_load: Fraction
    incoming_gains: tuple[Fraction, Fraction]
    outgoing_gains: tuple[Fraction, Fraction]
    status: str = "unevaluated"


@dataclass(frozen=True, slots=True)
class _PhasePolicy:
    R_hi: Fraction
    R_lo: Fraction
    disr_hi: Fraction
    disr_lo: Fraction
    kG_min: Fraction
    kG_max: Fraction
    kL_min: Fraction
    kL_max: Fraction
    up: Fraction
    down: Fraction


@dataclass(frozen=True, slots=True)
class NativeMutationLaw:
    pressure: tuple[Fraction, Fraction, Fraction]
    stability: tuple[Fraction, Fraction, Fraction]
    form_rails: tuple[Fraction, Fraction]
    capacity_rails: tuple[Fraction, Fraction]
    oz_factor: Fraction
    il_factor: Fraction
    mutation_shift: Fraction
    source_tiny: Fraction
    xi: Fraction
    pressure_cut: Fraction
    si_low: Fraction
    si_high: Fraction
    phase_gains: tuple[Fraction, Fraction]
    phase_policy: _PhasePolicy
    provenance: str


@dataclass(frozen=True, slots=True)
class ExecutionPremises:
    """Fixed declared execution; these are not recorded native events."""

    dt: Fraction
    times: tuple[Fraction, Fraction, Fraction]
    support: tuple[tuple[int, ...], ...]
    traces: tuple[tuple[tuple[str, ...], ...], ...]
    debts: tuple[tuple[int, ...], ...]
    node_ordinals: tuple[int, int, int]
    runtime_ordinals: tuple[int, int, int]
    lags: tuple[int, int, int]
    possible_counts: tuple[tuple[int, ...], ...]
    hypotheses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompiledCell:
    key: CellKey
    rows: tuple[LinearConstraint, ...]
    zero_pressure_direction: str


@dataclass(frozen=True, slots=True)
class CompiledSlice:
    spec: NativeMutationSlice
    law: NativeMutationLaw
    pressure: tuple[AffineExpression, ...]
    successor_pressure: tuple[AffineExpression, ...]
    form_boundaries: tuple[tuple[AffineExpression, ...], ...]
    temporal_histories: tuple[tuple[tuple[Fraction, AffineExpression], ...], ...]
    first_phase: tuple[Fraction, ...]
    mutation_phase: tuple[Fraction, ...]
    terminal_phase: tuple[Fraction, ...]
    controller_obligations: tuple[ControllerObligation, ControllerObligation]
    phase_geometry: tuple[ParameterCheck, ...]
    premises: ExecutionPremises
    cells: tuple[CompiledCell, ...]
    status: str


@dataclass(frozen=True, slots=True)
class CellContradiction:
    """A one-way cell result; false never means feasible."""

    key: CellKey
    proven: bool
    weighted_rhs: Fraction
    positive_strict_weight: bool
    coefficients: tuple[Fraction, ...]
    multipliers: tuple[tuple[str, Fraction], ...]
    spec: NativeMutationSlice
    law: NativeMutationLaw
    premises: ExecutionPremises
    controller_obligations: tuple[ControllerObligation, ControllerObligation]
    phase_geometry: tuple[ParameterCheck, ...]
    scope: str = "conditional fixed-parameter cell only"


@dataclass(frozen=True, slots=True)
class MaskPeakProjection:
    """Exact normalized-pressure mask test for one canonical peak owner."""

    peak: tuple[int, int] | None
    case: str
    conditions: tuple[ParameterCheck, ...]
    status: str


@dataclass(frozen=True, slots=True)
class MaskBoundaryProjection:
    """A single boundary, with pressure scale and transport discarded.

    ``stability_bases`` are Si before the pressure penalty; the four budgets
    correspond to blocks 1..4. A compatible owner admits only the normalized
    mask/peak/zero-sum subsystem, never an NFR source or execution.
    """

    boundary: str
    stability_bases: tuple[Fraction, ...]
    surrounding_budgets: tuple[Fraction, ...]
    target_ratio_lower: Fraction
    pressure_cut: Fraction
    owners: tuple[MaskPeakProjection, ...]
    status: str


@dataclass(frozen=True, slots=True)
class NativeMutationMaskProjection:
    """Necessary parameter restrictions with their complete source association.

    The status concerns the two separately projected mask systems. Even
    ``projection_compatible`` leaves their joint pressure transport, form
    rails, acquired history and actual controller predicates unresolved.
    Geometry and controller obligations are retained separately, not silently
    discharged by the projection.
    """

    spec: NativeMutationSlice
    law: NativeMutationLaw
    premises: ExecutionPremises
    source: MaskBoundaryProjection
    successor: MaskBoundaryProjection
    phase_geometry: tuple[ParameterCheck, ...]
    controller_obligations: tuple[ControllerObligation, ControllerObligation]
    status: str
    scope: str = "separate normalized-mask/peak/zero-sum subsystems only"


def _exact(value: Any, label: str) -> Fraction:
    return exact_or_represented_real(value, label)


def _five(values: Any, label: str) -> tuple[Fraction, ...]:
    items = _ordered(values, label)
    if len(items) != 5:
        raise ValueError(f"{label} requires exactly five coordinates")
    return tuple(_exact(value, f"{label}[{i}]") for i, value in enumerate(items))


def _owned_law() -> tuple[NativeMutationLaw, ExecutionPremises]:
    """Read a fresh default owner, never an input report or global graph."""
    cfg = CoreDefaults()
    if (
        cfg.INTEGRATOR_METHOD != "euler"
        or cfg.CLIP_MODE != "hard"
        or cfg.GAMMA["type"] != "none"
        or cfg.OZ_NOISE_MODE
        or not cfg.PHASE_ADAPT["enabled"]
        or cfg.VF_ADAPT_TAU <= 2
        or min(cfg.AL_MAX_LAG, cfg.EN_MAX_LAG) < 2
        or cfg.GLYPH_HYSTERESIS_WINDOW < 5
    ):
        raise ValueError("current defaults leave the declared two-call domain")

    def normalized(weights, channels, expected):
        if set(weights) != set(expected):
            raise ValueError("default weight channels leave the supported law")
        exact = {name: _exact(value, name) for name, value in weights.items()}
        if min(exact.values()) < 0 or sum(exact.values()) <= 0:
            raise ValueError("default diagnostic weights must be nonnegative")
        return tuple(exact[name] / sum(exact.values()) for name in channels)

    pressure = normalized(
        cfg.DNFR_WEIGHTS, ("epi", "vf", "phase"), ("epi", "vf", "phase", "topo")
    )
    stability = normalized(
        cfg.SI_WEIGHTS, ("alpha", "beta", "gamma"), ("alpha", "beta", "gamma")
    )
    if cfg.DNFR_WEIGHTS["topo"] != 0 or min(*pressure, *stability) <= 0:
        raise ValueError("the supported default law requires three positive channels")
    policy = _PhasePolicy(
        **{
            name: _exact(cfg.PHASE_ADAPT[name], f"phase policy {name}")
            for name in _PhasePolicy.__dataclass_fields__
        }
    )
    if not (
        0 < policy.R_lo < policy.R_hi < 1
        and 0 < policy.disr_lo < policy.disr_hi < 1
        and 0 < policy.up <= 1
        and 0 < policy.down <= 1
        and 0 < policy.kG_min <= policy.kG_max
        and 0 < policy.kL_min <= policy.kL_max
        and policy.kG_max + policy.kL_max < 1
    ):
        raise ValueError("unsupported default phase policy")
    factors, cuts = cfg.GLYPH_FACTORS, cfg.SELECTOR_THRESHOLDS
    if "ZHIR_theta_shift" in factors:
        raise ValueError("a fixed Mutation shift leaves the declared factor law")
    law = NativeMutationLaw(
        pressure=pressure,
        stability=stability,
        form_rails=(_exact(cfg.EPI_MIN, "EPI_MIN"), _exact(cfg.EPI_MAX, "EPI_MAX")),
        capacity_rails=(_exact(cfg.VF_MIN, "VF_MIN"), _exact(cfg.VF_MAX, "VF_MAX")),
        oz_factor=_exact(factors["OZ_dnfr_factor"], "OZ factor"),
        il_factor=_exact(factors["IL_dnfr_factor"], "IL factor"),
        mutation_shift=_exact(factors["ZHIR_theta_shift_factor"], "ZHIR factor")
        * _PI
        / 4,
        source_tiny=_OZ_TINY,
        xi=_exact(cfg.ZHIR_THRESHOLD_XI, "ZHIR threshold"),
        pressure_cut=_exact(cuts["dnfr_hi"], "pressure cut"),
        si_low=_exact(cuts["si_lo"], "low Si cut"),
        si_high=_exact(cuts["si_hi"], "high Si cut"),
        phase_gains=(
            _exact(cfg.PHASE_K_GLOBAL, "initial global gain"),
            _exact(cfg.PHASE_K_LOCAL, "initial local gain"),
        ),
        phase_policy=policy,
        provenance="fresh CoreDefaults; exact normalization of represented weights",
    )
    if not (
        law.form_rails == (Q(-1), Q(1))
        and law.capacity_rails[0] == 0 < law.capacity_rails[1]
        and law.oz_factor > 0
        and law.mutation_shift > 0
        and 0 < law.il_factor <= 1
        and law.xi > 0
        and 0 < law.pressure_cut < 1
        and 0 < law.si_low < law.si_high < 1
    ):
        raise ValueError("current defaults leave the supported scalar law")

    def codes(values):
        return {FUNCTION_TO_GLYPH[value].value for value in values}

    destab, stabilizers, handlers = map(
        codes, (DESTABILIZERS, STABILIZERS, BIFURCATION_HANDLERS)
    )
    if not (
        {"OZ", "ZHIR"} <= destab
        and "IL" in stabilizers & handlers
        and U2_DEBT_CAPACITY >= 2
        and BIFURCATION_WINDOW >= 1
    ):
        raise ValueError("the fixed grammar context no longer admits these roles")
    initial = (("AL", "IL", "EN"),) * 5
    traces1 = tuple(
        trace + (("OZ" if i == 0 else "IL"),) for i, trace in enumerate(initial)
    )
    traces2 = tuple(
        trace + (("ZHIR" if i == 0 else "IL"),) for i, trace in enumerate(traces1)
    )
    premises = ExecutionPremises(
        dt=_DT,
        times=(Q(1), Q(1) + _DT, Q(1) + 2 * _DT),
        support=tuple(
            tuple(
                2 * block + parity
                for block in ((i // 2 - 1) % 5, (i // 2 + 1) % 5)
                for parity in (0, 1)
            )
            for i in range(10)
        ),
        traces=(initial, traces1, traces2),
        debts=((0,) * 5, (1, 0, 0, 0, 0), (2, 0, 0, 0, 0)),
        node_ordinals=(3, 4, 5),
        runtime_ordinals=(0, 1, 2),
        lags=(0, 1, 2),
        possible_counts=((0,), (0, 1), (0, 1, 2)),
        hypotheses=(
            "paired primitive state, unit edges, nodes 0..9 and whole-pair order",
            "direct default glyph requests; strict grammar raise; supplied initial context",
            "prior IL true; common nonempty kind; initial debt, lags, counts, rates and acceleration zero",
            "initial timestamped EPI histories empty at t=1; later samples are associated ideal rows",
            "Gamma zero; no callbacks, optional mathematics or delayed REMESH evidence",
            "held Euler pressure; no form clipping; capacities fixed since tau exceeds two",
            "ideal circular common lifts; absolute Mutation telemetry and binary64 realization excluded",
            "negative algebraic zero is an outer direction, never an acquired native sign bit",
        ),
    )
    return law, premises


def _constant(value=Q(0)):
    return AffineExpression((Q(0),) * 5, value)


def _add(*expressions):
    return AffineExpression(
        tuple(
            sum(expression.coefficients[i] for expression in expressions)
            for i in range(5)
        ),
        sum(expression.constant for expression in expressions),
    )


def _scale(expression, value):
    return AffineExpression(
        tuple(value * x for x in expression.coefficients), value * expression.constant
    )


def _laplacian(values):
    return tuple(
        (values[(i - 1) % 5] + values[(i + 1) % 5]) / 2 - values[i] for i in range(5)
    )


def _affine_laplacian(values):
    return tuple(
        _add(
            _scale(values[(i - 1) % 5], Q(1, 2)),
            _scale(values[(i + 1) % 5], Q(1, 2)),
            _scale(values[i], -1),
        )
        for i in range(5)
    )


def _gains(law, old, branch):
    p = law.phase_policy
    if branch == "stable":
        targets = p.kG_min, p.kL_min
    elif branch == "dissonant":
        targets = p.kG_max, (p.kL_min + p.kL_max) / 2
    else:
        targets = (p.kG_min + p.kG_max) / 2, (p.kL_min + p.kL_max) / 2
    result = tuple(
        current + (p.up if target > current else p.down) * (target - current)
        for current, target in zip(old, targets)
    )
    if not (p.kG_min <= result[0] <= p.kG_max and p.kL_min <= result[1] <= p.kL_max):
        raise ValueError("gain update leaves the declared rails")
    return result


def _phase(values, gains):
    g, local = gains
    return tuple(
        (1 - g - local) * values[i]
        + local * (values[(i - 1) % 5] + values[(i + 1) % 5]) / 2
        for i in range(5)
    )


def _controller_obligation(law, step, branch, values, traces, old, new):
    disruptive = {getattr(value, "value", value) for value in DISRUPTORS}
    load = Q(
        sum(code in disruptive for trace in traces for code in trace),
        sum(map(len, traces)),
    )
    p = law.phase_policy
    if load > p.disr_lo:
        raise ValueError("the declared traces no longer have low disruptor load")
    return ControllerObligation(
        step=step,
        branch=branch,
        phase_differences=tuple(
            values[i] - values[j] for i, j in combinations(range(5), 2)
        ),
        constant=Q(1, 5),
        cosine_coefficient=Q(2, 25),
        lower_squared=(
            None
            if branch == "dissonant"
            else (p.R_hi if branch == "stable" else p.R_lo) ** 2
        ),
        upper_squared=(
            None
            if branch == "stable"
            else (p.R_lo if branch == "dissonant" else p.R_hi) ** 2
        ),
        lower_strict=branch == "transition",
        upper_strict=branch == "transition",
        disruptor_load=load,
        incoming_gains=old,
        outgoing_gains=new,
    )


def _prepare(spec):
    if not isinstance(spec, NativeMutationSlice):
        raise TypeError("expected primitive NativeMutationSlice")
    law, premises = _owned_law()
    capacities, phases = _five(spec.capacities, "capacities"), _five(
        spec.phases, "phases"
    )
    if any(not 0 < value <= law.capacity_rails[1] for value in capacities):
        raise ValueError("all capacities must be active and within the declared rails")
    for branch in (spec.first_branch, spec.second_branch):
        if type(branch) is not str or branch not in BRANCHES:
            raise ValueError(
                "controller branches must be stable, transition or dissonant"
            )
    # Phase origin is discarded only by the declared ideal quotient. Retain
    # one coherent source lift; do not materialize or wrap its coordinates.
    phases = tuple(value - phases[0] for value in phases)
    normalized = NativeMutationSlice(
        capacities, phases, spec.first_branch, spec.second_branch
    )
    gains1 = _gains(law, law.phase_gains, spec.first_branch)
    gains2 = _gains(law, gains1, spec.second_branch)
    first = _phase(phases, gains1)
    mutated = (first[0] - law.mutation_shift, *first[1:])
    terminal = _phase(mutated, gains2)
    geometry = (
        ParameterCheck("source_lift_width", max(phases) - min(phases), Q(3), "<="),
        ParameterCheck("first_common_arc", max(first) - min(first), _PI, "<"),
        ParameterCheck("mutation_common_arc", max(mutated) - min(mutated), _PI, "<"),
        ParameterCheck(
            "negative_outward_gap", first[0] - (first[1] + first[4]) / 2, Q(0), "<"
        ),
        ParameterCheck(
            "complete_renewal",
            max(first) - min(first),
            max(terminal) - min(terminal),
            "<",
        ),
        ParameterCheck(
            "target_not_capacity_maximum", capacities[0], max(capacities), "<"
        ),
        *(
            ParameterCheck(
                f"terminal_target_minimum_{i}", terminal[0], terminal[i], "<="
            )
            for i in range(1, 5)
        ),
    )
    obligations = (
        _controller_obligation(
            law,
            1,
            spec.first_branch,
            phases,
            premises.traces[1],
            law.phase_gains,
            gains1,
        ),
        _controller_obligation(
            law, 2, spec.second_branch, mutated, premises.traces[2], gains1, gains2
        ),
    )
    pressure = tuple(
        AffineExpression(tuple(Q(i == j) for i in range(5))) for j in range(4)
    )
    pressure += (_scale(_add(*pressure), -1),)
    offset = AffineExpression((Q(0), Q(0), Q(0), Q(0), Q(1)))
    k, b, w = law.pressure
    d = w / _PI
    inverse = tuple(
        _add(
            _scale(pressure[i], Q(-4, 5)),
            _scale(pressure[(i - 2) % 5], Q(2, 5)),
            _scale(pressure[(i + 2) % 5], Q(2, 5)),
        )
        for i in range(5)
    )
    x0 = tuple(
        _add(
            _scale(inverse[i], 1 / k),
            _constant((-b * capacities[i] - d * phases[i]) / k),
            offset,
        )
        for i in range(5)
    )
    flow1 = tuple(
        _scale(
            pressure[i], capacities[i] * (law.oz_factor if i == 0 else law.il_factor)
        )
        for i in range(5)
    )
    x1 = tuple(_add(x0[i], _scale(flow1[i], _DT)) for i in range(5))
    change = _laplacian(tuple(first[i] - phases[i] for i in range(5)))
    flow_pressure = _affine_laplacian(flow1)
    successor = tuple(
        _add(pressure[i], _scale(flow_pressure[i], _DT * k), _constant(d * change[i]))
        for i in range(5)
    )
    x2 = tuple(
        _add(
            x1[i],
            _scale(
                successor[i], _DT * capacities[i] * (1 if i == 0 else law.il_factor)
            ),
        )
        for i in range(5)
    )
    return (
        normalized,
        law,
        premises,
        pressure,
        successor,
        (x0, x1, x2),
        first,
        mutated,
        terminal,
        obligations,
        geometry,
    )


def _key(key):
    if not isinstance(key, CellKey):
        raise TypeError("expected CellKey")

    def peak(value, allowed, name):
        if (
            not isinstance(value, (tuple, list))
            or len(value) != 2
            or any(type(x) is not int for x in value)
        ):
            raise ValueError(f"{name} requires integer block and sign")
        value = tuple(value)
        if value not in allowed:
            raise ValueError(f"{name} is not a canonical signed peak owner")
        return value

    source = peak(key.source_peak, _SOURCE_PEAKS, "source peak")
    successor = (
        None
        if key.successor_peak is None
        else peak(key.successor_peak, _NEXT_PEAKS, "successor peak")
    )
    return CellKey(source, successor)


def _si_bases(spec, law, phases):
    """Shared fresh Si terms before subtracting the pressure ratio."""
    alpha, beta, gamma = law.stability
    maximum = max(spec.capacities)
    gaps = _laplacian(phases)
    return tuple(
        alpha * spec.capacities[i] / maximum + beta * (1 - abs(gaps[i]) / _PI) + gamma
        for i in range(5)
    )


def _cell(prepared, key):
    spec, law, _, pressure, successor, boundaries, first, *_ = prepared
    key = _key(key)
    rows = []

    def row(label, expression, rhs=Q(0), *, strict=False, unit="pressure"):
        rows.append(
            LinearConstraint(
                label, expression.coefficients, rhs - expression.constant, strict, unit
            )
        )

    for time, values in enumerate(boundaries):
        for i, value in enumerate(values):
            row(f"rail.t{time}.b{i}.upper", value, law.form_rails[1], unit="form")
            row(
                f"rail.t{time}.b{i}.lower",
                _scale(value, -1),
                -law.form_rails[0],
                unit="form",
            )

    def peak_rows(prefix, values, owner):
        index, sign = owner
        peak = _scale(values[index], sign)
        row(f"{prefix}.peak.positive", _scale(peak, -1), strict=True)
        for i, value in enumerate(values):
            row(f"{prefix}.peak.b{i}.upper", _add(value, _scale(peak, -1)))
            row(f"{prefix}.peak.b{i}.lower", _add(_scale(value, -1), _scale(peak, -1)))
        return peak

    gamma = law.stability[2]
    s0, s1 = _si_bases(spec, law, spec.phases), _si_bases(spec, law, first)
    peak0 = peak_rows("source", pressure, key.source_peak)
    row(
        "source.target.si_low",
        _add(_scale(peak0, s0[0] - law.si_low), _scale(pressure[0], -gamma)),
    )
    row(
        "source.target.oz_ratio",
        _add(_scale(peak0, law.pressure_cut), _scale(pressure[0], -1)),
        strict=True,
    )
    row("source.target.oz_tiny", _scale(pressure[0], -1), -law.source_tiny, strict=True)
    row(
        "source.target.acquired_rate",
        _scale(pressure[0], -spec.capacities[0] * law.oz_factor),
        -law.xi,
        strict=True,
        unit="form/time",
    )

    def il_rows(prefix, values, bases, peak):
        for i in range(1, 5):
            for sign, name in ((1, "positive"), (-1, "negative")):
                row(
                    f"{prefix}.il.b{i}.{name}",
                    _add(
                        _scale(values[i], sign * gamma),
                        _scale(peak, law.si_high - bases[i]),
                    ),
                )

    il_rows("source", pressure, s0, peak0)
    if key.successor_peak is None:
        for i, value in enumerate(successor):
            row(f"successor.zero.b{i}.upper", value)
            row(f"successor.zero.b{i}.lower", _scale(value, -1))
        row(
            "successor.zero.target.si_low",
            _constant(s1[0] - law.si_low),
            unit="dimensionless",
        )
        for i in range(1, 5):
            row(
                f"successor.zero.il.b{i}",
                _constant(law.si_high - s1[i]),
                unit="dimensionless",
            )
        direction = "negative signed-zero outer direction; no native realization proof"
    else:
        peak1 = peak_rows("successor", successor, key.successor_peak)
        row(
            "successor.target.si_low",
            _add(_scale(peak1, s1[0] - law.si_low), _scale(successor[0], gamma)),
        )
        row("successor.target.nonpositive", successor[0])
        row(
            "successor.target.zhir_ratio",
            _add(_scale(successor[0], -1), _scale(peak1, -law.pressure_cut)),
        )
        il_rows("successor", successor, s1, peak1)
        direction = "negative q0, or unproved negative signed zero at q0=0"
    return CompiledCell(key, tuple(rows), direction)


def _mask_boundary(law, bases, boundary):
    """Eliminate normalized pressure by exact intervals of sums.

    This internal arithmetic helper receives only freshly rebuilt terms from
    ``project_native_mutation_masks``. It is not a report-consuming admission
    API. For a peak j, the other three ratios fill [-sum(B), sum(B)].
    Source pressure is +tP with c<t<=1; successor pressure is -tP with
    0<=t<=c. The zero-peak branch uses Si at rho=0 independently.
    """
    gamma, cut = law.stability[2], law.pressure_cut
    budgets = tuple(min(Q(1), (value - law.si_high) / gamma) for value in bases[1:])
    target = max(Q(0), (bases[0] - law.si_low) / gamma)
    common = tuple(
        ParameterCheck(f"{boundary}.il.b{i}.nonnegative", Q(0), value, "<=")
        for i, value in enumerate(budgets, start=1)
    )
    owners = []
    peaks = _SOURCE_PEAKS if boundary == "source" else (*_NEXT_PEAKS, None)
    for peak in peaks:
        checks = list(common)

        def require(label, lhs, rhs, relation="<="):
            checks.append(ParameterCheck(f"{boundary}.{label}", lhs, rhs, relation))

        if peak is None:
            case = "successor_zero"
            require("zero.target_mask", target, Q(0))
        elif peak[0] == 0:
            case = "source_target"
            require("target_ratio", target, Q(1))
            require("surrounding_sum", Q(1), sum(budgets))
        else:
            block, sign = peak
            rest = sum(value for i, value in enumerate(budgets, start=1) if i != block)
            require(f"peak.b{block}.budget", Q(1), budgets[block - 1])
            if boundary == "source":
                require("target_ratio", target, Q(1))
                if sign > 0:
                    case = "source_positive"
                    require("strict_ratio_sum", Q(1) + cut, rest, "<")
                    require("target_sum", Q(1) + target, rest)
                else:
                    # t=1 is permitted, including a tie with the target peak.
                    case = "source_negative"
            else:
                require("target_ratio", target, cut)
                if sign > 0:
                    case = "successor_positive"
                    require("remaining_sum", Q(1) - cut, rest)
                else:
                    case = "successor_negative"
                    require("remaining_sum", Q(1) + target, rest)
        owners.append(
            MaskPeakProjection(
                peak=peak,
                case=case,
                conditions=tuple(checks),
                status=(
                    "projection_compatible"
                    if all(check.satisfied for check in checks)
                    else "excluded"
                ),
            )
        )
    return MaskBoundaryProjection(
        boundary=boundary,
        stability_bases=tuple(bases),
        surrounding_budgets=budgets,
        target_ratio_lower=target,
        pressure_cut=cut,
        owners=tuple(owners),
        status=(
            "projection_compatible"
            if any(owner.status == "projection_compatible" for owner in owners)
            else "excluded"
        ),
    )


def project_native_mutation_masks(
    spec: NativeMutationSlice,
) -> NativeMutationMaskProjection:
    """Rebuild and project the two diagnostic mask systems, without solving.

    An excluded boundary precludes every pressure scale in that subsystem.
    Passing both boundaries does not associate their pressures through the
    source constitutive row or held-flow transport. The canonical 81 full
    systems are retained unchanged by ``compile_native_mutation_slice``.
    """
    prepared = _prepare(spec)
    normalized, law, premises = prepared[:3]
    source = _mask_boundary(
        law, _si_bases(normalized, law, normalized.phases), "source"
    )
    successor = _mask_boundary(
        law, _si_bases(normalized, law, prepared[6]), "successor"
    )
    return NativeMutationMaskProjection(
        spec=normalized,
        law=law,
        premises=premises,
        source=source,
        successor=successor,
        phase_geometry=prepared[10],
        controller_obligations=prepared[9],
        status=(
            "projection_compatible"
            if source.status == successor.status == "projection_compatible"
            else "excluded"
        ),
    )


def compile_native_mutation_slice(spec: NativeMutationSlice) -> CompiledSlice:
    """Compile all 81 conditional systems; perform no feasibility evaluation."""
    prepared = _prepare(spec)
    (
        normalized,
        law,
        premises,
        pressure,
        successor,
        boundaries,
        first,
        mutated,
        terminal,
        obligations,
        geometry,
    ) = prepared
    cells = tuple(
        _cell(prepared, CellKey(source, following))
        for source in _SOURCE_PEAKS
        for following in (*_NEXT_PEAKS, None)
    )
    histories = tuple(
        tuple((time, boundaries[j][i]) for j, time in enumerate(premises.times))
        for i in range(5)
    )
    status = (
        "conditional"
        if all(check.satisfied for check in geometry)
        else "parameter_incompatible"
    )
    return CompiledSlice(
        normalized,
        law,
        pressure,
        successor,
        boundaries,
        histories,
        first,
        mutated,
        terminal,
        obligations,
        geometry,
        premises,
        cells,
        status,
    )


def check_cell_contradiction(
    spec: NativeMutationSlice,
    key: CellKey,
    multipliers: Iterable[tuple[str, Any]],
) -> CellContradiction:
    """Rebuild one cell and check an exact nonnegative row combination.

    Negative summed rhs, or zero rhs with a positively weighted strict row,
    proves this conditional cell inconsistent after exact coefficient
    cancellation. An invalid certificate raises; an inconclusive one returns
    proven=False and says nothing about feasibility. Compiled rows and cached
    verdicts are deliberately not accepted as inputs.
    """
    prepared, key = _prepare(spec), _key(key)
    rows = {row.label: row for row in _cell(prepared, key).rows}
    supplied = _ordered(multipliers, "labeled multipliers")
    coefficients, rhs, strict, normalized, seen = [Q(0)] * 5, Q(0), False, [], set()
    for entry in supplied:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each multiplier needs exactly one label and scalar")
        label, value = entry
        if type(label) is not str or label not in rows or label in seen:
            raise ValueError("unknown or duplicate canonical row label")
        seen.add(label)
        weight = _exact(value, f"multiplier {label}")
        if weight < 0:
            raise ValueError("row multipliers must be nonnegative")
        row = rows[label]
        coefficients = [a + weight * b for a, b in zip(coefficients, row.coefficients)]
        rhs += weight * row.rhs
        strict |= weight > 0 and row.strict
        normalized.append((label, weight))
    if any(coefficients):
        raise ValueError("certificate coefficients do not cancel exactly")
    return CellContradiction(
        key=key,
        proven=rhs < 0 or (rhs == 0 and strict),
        weighted_rhs=rhs,
        positive_strict_weight=strict,
        coefficients=tuple(coefficients),
        multipliers=tuple(normalized),
        spec=prepared[0],
        law=prepared[1],
        premises=prepared[2],
        controller_obligations=prepared[9],
        phase_geometry=prepared[10],
    )
