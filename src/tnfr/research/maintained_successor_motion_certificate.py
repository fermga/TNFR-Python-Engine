"""One fixed full-state confinement attempt for the hypothetical UM successor.

This detached Picard/Taylor calculation installs no event or runtime law. Its
conditional source family is rebuilt from primitive graph state. The fixed
mesh and observations must be frozen with the calling instrument before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from time import perf_counter

from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import validated_box_taylor_step
from ..physics._sine_flow import _held_sine_neighbor_field
from . import maintained_principal_retention as _retention
from .local_capacity_history import MODEL
from .maintained_successor_motion import predict_successor_motion

# Preserve the historical module constants and public imports for consumers.
HORIZON = _retention.HORIZON
STEP = _retention.STEP
STEP_COUNT = _retention.STEP_COUNT
ORDER = _retention.ORDER
WALL_BUDGET_SECONDS = _retention.WALL_BUDGET_SECONDS
_principal_margins = _retention.principal_margins
PARENT = 2
CHILD = 20


@dataclass(frozen=True)
class MotionAttempt:
    """A method outcome, with unavailable claims kept distinct from escape."""

    status: str
    attempted_steps: int
    accepted_steps: int
    certified_retention_time: Q
    initial_relative_velocity: I
    positive_velocity_times: tuple[Q, ...]
    minimum_retained_margin: Q
    full_horizon_retention: bool
    reversal_certified: bool
    joint_target_certified: bool
    elapsed_seconds: float


def prepare_motion(graph):
    """Rebuild the conditional source and a held full weighted sine field.

    All capacities and weights remain uncertain held constants. Independent
    output allowances forbid imposing the ideal twin synchronization manifold.
    Neither this adapter nor its shared field trusts cached motion verdicts.
    """
    prediction = predict_successor_motion(graph, horizon=HORIZON)
    chamber = prediction.chamber
    rows = [[] for _ in chamber.reset.nodes]
    for (i, j), weight in zip(
        chamber.edges, chamber.conditional_conductances, strict=True
    ):
        rows[i].append((j, 1, weight))
        rows[j].append((i, 1, weight))
    degrees = tuple(map(len, rows))
    flow = _held_sine_neighbor_field(
        MODEL, degrees, chamber.conditional_capacities, tuple(map(tuple, rows))
    )
    box = tuple(I(value) for value in chamber.reset.epi) + chamber.conditional.phases
    if len(box) != 42 or len(chamber.edges) != 53:
        raise ValueError("the fixed motion protocol requires the unchanged successor")
    return prediction, box, flow


def certify_motion(graph, *, record_step, monotonic_clock=perf_counter):
    """Attempt the frozen half-unit window once, without adaptive retries.

    Every accepted or failed enclosure is delivered to the recording callback
    before another step. An unresolved branch bound stops the attempt; it is
    not an observed crossing. Resource checks occur between bounded steps.
    Callback or solver exceptions propagate, preserving any preceding records.
    This function does not itself establish freeze chronology or write assets.
    """
    if not callable(record_step) or not callable(monotonic_clock):
        raise TypeError("recording and resource-clock callbacks must be callable")
    started = monotonic_clock()
    prediction, box, flow = prepare_motion(graph)
    chamber = prediction.chamber
    edges, shifts = chamber.edges, chamber.conditional.shifts
    initial_rates = flow(box)
    initial_velocity = initial_rates[21 + CHILD] - initial_rates[21 + PARENT]
    if initial_velocity.hi >= 0:
        raise ValueError(
            "the frozen reversal target requires strictly negative initial speed"
        )
    positive_times = []
    # Independent energy bounds apply throughout the entire fixed window.
    energy_margins = tuple(
        value.lo for value in prediction.conditional.horizon_remaining_margins
    )

    def record_motion(record):
        step = record["step"]
        if step is not None:
            rates = flow(step.endpoint)
            velocity = rates[21 + CHILD] - rates[21 + PARENT]
            record["endpoint_relative_velocity"] = velocity
        record_step(record)
        if (
            step is not None
            and min(record["combined_principal_margins"]) > 0
            and velocity.lo > 0
        ):
            positive_times.append(record["time"] + STEP)

    retained = _retention.certify_retention(
        box,
        flow,
        edges,
        shifts,
        energy_margins,
        record_step=record_motion,
        monotonic_clock=monotonic_clock,
        started=started,
        stepper=validated_box_taylor_step,
    )
    reversal = bool(positive_times)
    return MotionAttempt(
        retained.status,
        retained.attempted_steps,
        retained.accepted_steps,
        retained.certified_retention_time,
        initial_velocity,
        tuple(positive_times),
        retained.minimum_retained_margin,
        retained.full_horizon_retention,
        reversal,
        retained.full_horizon_retention and reversal,
        retained.elapsed_seconds,
    )
