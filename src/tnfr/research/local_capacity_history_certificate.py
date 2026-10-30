"""Independent full-state enclosures for the fixed local VAL history gate.

The numerical producer is separate. This adapter re-admits its represented
source and records, then compares them with complete forty-coordinate
Picard/Taylor enclosures. It executes no operator or numerical trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..mathematics._rational_interval import I, cos, pi_interval, sin
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    validated_box_taylor_step,
)
from ..physics._sine_admission import _sine_model_coefficients
from ..physics.relational_observations import (
    _ordered,
    bound_relational_sample_jet_budget,
)
from ..physics.relational_sine_support import _rebuild_sine_support_comparison
from ..physics.relational_sine_symmetry import (
    _support_involution_field,
    assess_sine_support_involution,
)

PARENT = 2
CAPACITY = Q(128)
MEAN = Q(5, 16)
BOOST = PUSH = Q(1, 1024)
SOURCE_EPSILON = Q(1, 2**18)
SAMPLE_STEP = Q(1, 2**20)
TIMES = (Q(7), Q(7) + SAMPLE_STEP, Q(7) + 2 * SAMPLE_STEP)
ORDER = 4
ENDPOINT_BUDGET = Q(1, 2**52)
READER_BUDGET = Q(1, 2**13)
TRUE_ACCELERATION_LOWER = Q(1, 8)


@dataclass(frozen=True)
class SourceAdmission:
    source: object
    radius_squared: I
    storage_excess: I
    max_form_deviation: Q
    max_phase_deviation: Q
    max_weight_deviation: Q
    parent_pressure: I


@dataclass(frozen=True)
class HistoryCertificate:
    admission: SourceAdmission
    times: tuple[Q, ...]
    represented_states: tuple[tuple[Q, ...], ...]
    steps: tuple[ValidatedBoxTaylorStep, ...]
    endpoint_bounds: tuple[tuple[I, ...], ...]
    state_error_bounds: tuple[tuple[Q, ...], ...]
    parent_error_bounds: tuple[Q, ...]
    true_parent_pressure_bounds: tuple[I, ...]
    represented_parent_pressure_bounds: tuple[I, ...]
    sample_error_bound: Q
    reader_error_bound: Q
    trigger_lower_bound: Q
    endpoint_budget_certified: bool


def _represented(value, label):
    """Reject source coercion instead of changing a declared represented value."""
    exact = exact_or_represented_real(value, label)
    _, represented = finite_represented_real(value, label)
    if exact != represented:
        raise ValueError(f"{label} must already be exactly represented in binary64")
    return represented


def _support():
    old = {
        (i, j)
        for i in range(20)
        for j in range(i + 1, 20)
        if i // 10 == j // 10 and ((i % 10 // 2 - j % 10 // 2) % 5) in (1, 4)
    }
    added = {(0, 1), (0, 10), (0, 11), (1, 10), (1, 11)}
    return old, added


def prepare_field(report):
    """Return rebuilt primitives and the full weighted structural-time field.

    The identity permutation retains all forty coordinates, every actual
    conductance and the nonuniform capacities. No twin reduction, positive
    loss or common time change is introduced.
    """
    source = _rebuild_sine_support_comparison(report)
    reduction, flow = _support_involution_field(
        assess_sine_support_involution(
            source, permutation_indices=tuple(range(len(source.nodes)))
        )
    )
    if not reduction.source_trajectory_reduction_certified or any(
        value != 1 for value in reduction.multiplicities
    ):
        raise ValueError("the complete source must belong to the identity family")
    return reduction.source, flow


def admit_source(report):
    """Admit one represented post-VAL source by the direct-output corollary.

    The fixed nominal origin is 5/16 and the fixed target capacity is 128.
    Admission proves the mathematical premises, not ancestry, a completed
    operator word, a supplied clock association or absence of live Gamma.
    Those execution facts remain the numerical producer's obligations.
    """
    source = _rebuild_sine_support_comparison(report)
    if source.nodes != tuple(range(20)):
        raise ValueError("source must retain the ordered twenty-node contact")
    if _sine_model_coefficients(source.reference_model) != (Q(0), Q(1), Q(1)):
        raise ValueError("source requires e=0, w=beta=1 in structural time")
    expected_capacity = tuple(
        CAPACITY if i == PARENT else Q(1, 2) if i < 2 else Q(1) for i in range(20)
    )
    if source.capacity != expected_capacity:
        raise ValueError("source capacity must be the fixed local VAL endpoint")
    old, added = _support()
    actual = dict(zip(source.edges, source.conductances, strict=True))
    if actual.keys() != old | added:
        raise ValueError("source support must be the declared 45-edge contact")
    for label, values in (
        ("form", source.epi),
        ("phase", source.phase),
        ("conductance", source.conductances),
    ):
        for i, value in enumerate(values):
            _represented(value, f"source {label}[{i}]")
    if any(actual[edge] != 1 for edge in old) or any(
        not Q(3, 4) < actual[edge] <= 1 for edge in added
    ):
        raise ValueError("actual weights violate the contact barrier hypotheses")

    alpha = 2 * pi_interval() / 5
    reference = tuple((i % 10 // 2) * alpha for i in range(20))
    nominal_phase = list(reference)
    for i in (2, 3):
        nominal_phase[i] = (1 - PUSH) * alpha
    for i in (8, 9):
        nominal_phase[i] = (4 + PUSH) * alpha
    form_deviation = max(
        abs(value - MEAN - (BOOST if i < 2 else 0))
        for i, value in enumerate(source.epi)
    )
    phase_deviation = max(
        (I(value) - nominal).abs_max
        for value, nominal in zip(source.phase, nominal_phase, strict=True)
    )
    regularizer = Q(1e-9)
    omega = 1 - BOOST / (4 * (2 * MEAN + BOOST + regularizer))
    weight_deviation = max(
        abs(actual[edge] - (1 if edge == (0, 1) else omega)) for edge in added
    )
    if form_deviation >= SOURCE_EPSILON or phase_deviation >= 2 * SOURCE_EPSILON:
        raise ValueError("source coordinates leave the direct-output domain")
    if weight_deviation >= 3 * SOURCE_EPSILON:
        raise ValueError("source weights leave the direct-output domain")

    form_mean = sum(source.epi, Q(0)) / 20
    phase_error = tuple(
        I(value) - base for value, base in zip(source.phase, reference, strict=True)
    )
    phase_mean = sum(phase_error, I(0)) / 20
    radius_squared = I(sum(((x - form_mean) ** 2 for x in source.epi), Q(0)))
    radius_squared += sum(((value - phase_mean) ** 2 for value in phase_error), I(0))
    storage_excess = source.storage - 40 * (1 - cos(alpha))
    if radius_squared.hi >= Q(1, 256):
        raise ValueError("source radius does not admit the retained identity")
    if storage_excess.hi >= Q(1, 8192):
        raise ValueError("source storage does not admit the retained barrier")
    if source.pressure[PARENT].lo <= Q(1, 2**14):
        raise ValueError("fresh source pressure lacks the proved strict margin")
    return SourceAdmission(
        source,
        radius_squared,
        storage_excess,
        form_deviation,
        phase_deviation,
        weight_deviation,
        source.pressure[PARENT],
    )


def _parent_pressure(source, state):
    phases = state[20:]
    adjacent = tuple(
        right if left == PARENT else left
        for left, right in source.edges
        if PARENT in (left, right)
    )
    return sum((sin(phases[j] - phases[PARENT]) for j in adjacent), I(0)) / (
        pi_interval() * source.degrees[PARENT]
    )


def certify_history(report, times, recorded_states):
    """Bound the fixed three numerical records independently of their trigger.

    Two independent direct-from-source order-four steps enclose the true
    solution at the prescribed endpoints. Neither the numerical states nor
    an evaluated acceleration select their order, durations or initial box.
    All forty output errors are retained; only parent-form errors enter the
    three-sample budget. The result certifies no runtime history association.
    """
    raw_times = _ordered(times, "times", limit=4)
    admitted_times = tuple(
        _represented(value, f"times[{i}]") for i, value in enumerate(raw_times)
    )
    if admitted_times != TIMES:
        raise ValueError("history must use the three frozen exact structural times")
    rows = _ordered(recorded_states, "recorded_states", limit=4)
    if len(rows) != 3:
        raise ValueError("history requires exactly three complete recorded states")
    records = []
    for index, row in enumerate(rows):
        values = _ordered(row, f"recorded_states[{index}]", limit=41)
        if len(values) != 40:
            raise ValueError("each record must retain all forty ordered coordinates")
        records.append(
            tuple(
                _represented(value, f"recorded_states[{index}][{i}]")
                for i, value in enumerate(values)
            )
        )
    records = tuple(records)
    admission = admit_source(report)
    source, flow = prepare_field(admission.source)
    initial = source.epi + source.phase
    if records[0] != initial:
        raise ValueError("initial recorded state must equal the admitted source")
    box = tuple(I(value) for value in initial)
    steps = []
    endpoints = [box]
    for time in TIMES[1:]:
        step, _, reason = validated_box_taylor_step(
            box, time - TIMES[0], flow, lambda _: (Q(1),), order=ORDER, time=TIMES[0]
        )
        if step is None:
            raise ValueError(f"fixed full-state enclosure unavailable: {reason}")
        steps.append(step)
        endpoints.append(step.endpoint)
    errors = tuple(
        tuple(
            (I(value) - bound).abs_max for value, bound in zip(row, bounds, strict=True)
        )
        for row, bounds in zip(records, endpoints, strict=True)
    )
    parent_errors = tuple(row[PARENT] for row in errors)
    sample_budget = bound_relational_sample_jet_budget(
        sample_step=SAMPLE_STEP,
        sample_error_bounds=parent_errors,
        third_derivative_bound=CAPACITY**3 / 128,
        observation_time=TIMES[0],
    )
    # The full-window theorem bounds the true sample average. The separate
    # initial-derivative truncation component concerns a different comparison.
    sample_error = sample_budget.acceleration_sample_error_bound
    if any(not Q(1, 8) < row[PARENT] < Q(1, 2) for row in records):
        raise ValueError("represented parent forms leave the reader arithmetic range")
    true_pressure = tuple(_parent_pressure(source, state) for state in endpoints)
    represented_pressure = tuple(
        _parent_pressure(source, tuple(I(value) for value in state))
        for state in records
    )
    return HistoryCertificate(
        admission=admission,
        times=admitted_times,
        represented_states=records,
        steps=tuple(steps),
        endpoint_bounds=tuple(endpoints),
        state_error_bounds=errors,
        parent_error_bounds=parent_errors,
        true_parent_pressure_bounds=true_pressure,
        represented_parent_pressure_bounds=represented_pressure,
        sample_error_bound=sample_error,
        reader_error_bound=READER_BUDGET,
        trigger_lower_bound=TRUE_ACCELERATION_LOWER - sample_error - READER_BUDGET,
        endpoint_budget_certified=max(parent_errors) <= ENDPOINT_BUDGET,
    )
