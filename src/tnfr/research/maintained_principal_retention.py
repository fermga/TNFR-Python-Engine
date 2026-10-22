"""One bounded full-state principal-retention attempt for a prepared successor.

Preparation owners supply the complete held field, original branch labels and
independently rebuilt global energy margins for the same half-unit window.
This shared loop chooses no event, source, velocity target or physical model.
It neither authenticates its supplied premises nor freezes an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from time import perf_counter

from .._exact_time import exact_or_represented_real
from ..mathematics._comparison_flow import _exact, _ordered
from ..mathematics._rational_interval import I, pi_interval
from ..mathematics._validated_taylor import validated_box_taylor_step
from .maintained_successor_chamber import _admit_edges

HORIZON = Q(1, 2)
STEP = Q(1, 512)
STEP_COUNT = 256
ORDER = 12
WALL_BUDGET_SECONDS = 900


@dataclass(frozen=True)
class RetentionAttempt:
    """A method outcome; unresolved bounds and budget stops are not escapes."""

    status: str
    attempted_steps: int
    accepted_steps: int
    certified_retention_time: Q
    minimum_retained_margin: Q
    full_horizon_retention: bool
    elapsed_seconds: float


def principal_margins(state, edges, shifts):
    """Read all 53 original branches on the complete 42-coordinate state.

    The support and shifts retain their supplied order. No wrapping can hide
    a crossing of the initially selected branch. This is a same-family work
    policy, not a general restriction on sine dynamics.
    """
    state = tuple(I.coerce(value) for value in _ordered(state, "state"))
    edges = _admit_edges(21, _ordered(edges, "edges"))
    shifts = _ordered(shifts, "shifts")
    if len(state) != 42 or len(edges) != 53 or len(shifts) != 53:
        raise ValueError("the successor observation requires all 42 rows and 53 edges")
    if any(type(shift) is not int for shift in shifts):
        raise TypeError("original branches require exact integer shifts")
    phase = state[21:]
    pi = pi_interval()
    return tuple(
        pi.lo - (phase[j] - phase[i] - 2 * shift * pi).abs_max
        for (i, j), shift in zip(edges, shifts, strict=True)
    )


def _clock_read(monotonic_clock):
    value = monotonic_clock()
    exact_or_represented_real(value, "resource clock")
    return value


def certify_retention(
    box,
    flow,
    edges,
    shifts,
    energy_margins,
    *,
    record_step,
    monotonic_clock=perf_counter,
    started=None,
    stepper=None,
):
    """Attempt the fixed half-unit window once from fully prepared premises.

    ``energy_margins`` are exact per-edge lower bounds valid throughout the
    entire declared window. The caller must derive them for this same source,
    field, support, capacity and clock; saved verdicts do not supply that proof.
    The maximum with a tube margin combines two independent sufficient bounds.
    The full sine field is smooth globally, including beyond principal cuts.

    An optional ``started`` includes preparation in the resource budget. An
    injected stepper supports orchestration controls; scientific callers use
    the shared validated source-box Taylor step. No relative-velocity readout
    is required or evaluated. Recording and solver exceptions propagate.
    """
    if not all(callable(value) for value in (flow, record_step, monotonic_clock)):
        raise TypeError(
            "field, recording and resource-clock callbacks must be callable"
        )
    if stepper is None:
        stepper = validated_box_taylor_step
    if not callable(stepper):
        raise TypeError("the enclosure stepper must be callable")
    if started is None:
        started = _clock_read(monotonic_clock)
    else:
        exact_or_represented_real(started, "resource start")
    box = tuple(I.coerce(value) for value in _ordered(box, "state"))
    edges = _admit_edges(21, _ordered(edges, "edges"))
    shifts = _ordered(shifts, "shifts")
    energy_margins = tuple(
        _exact(value) for value in _ordered(energy_margins, "energy margins")
    )
    if len(energy_margins) != len(edges):
        raise ValueError("one global energy margin is required per original branch")
    initial_margins = principal_margins(box, edges, shifts)
    if min(initial_margins) <= 0:
        raise ValueError("initial principal identity is unresolved")
    minimum = min(initial_margins)
    accepted = attempted = 0
    status = "completed"

    def elapsed():
        value = _clock_read(monotonic_clock) - started
        if value < 0:
            raise ValueError("resource clock precedes its declared start")
        return value

    for index in range(STEP_COUNT):
        if elapsed() >= WALL_BUDGET_SECONDS:
            status = "resource_budget_exhausted"
            break
        attempted += 1
        time = index * STEP
        step, failed_tube, reason = stepper(
            box, STEP, flow, lambda _: (Q(1),), order=ORDER, time=time
        )
        if step is None:
            record_step(
                {
                    "index": index,
                    "time": time,
                    "step": None,
                    "failed_tube": failed_tube,
                    "reason": reason,
                }
            )
            status = "enclosure_unavailable"
            break
        tube_margins = principal_margins(step.tube, edges, shifts)
        margins = tuple(
            max(a, b) for a, b in zip(tube_margins, energy_margins, strict=True)
        )
        record_step(
            {
                "index": index,
                "time": time,
                "step": step,
                "tube_principal_margins": tube_margins,
                "combined_principal_margins": margins,
            }
        )
        if min(margins) <= 0:
            status = "principal_bound_unresolved"
            break
        minimum = min(minimum, min(margins))
        accepted += 1
        box = step.endpoint
    return RetentionAttempt(
        status,
        attempted,
        accepted,
        accepted * STEP,
        minimum,
        accepted == STEP_COUNT,
        elapsed(),
    )
