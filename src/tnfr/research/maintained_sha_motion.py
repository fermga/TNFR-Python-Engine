"""Rebuilt complete-state continuation after the retained SHA closure.

Preparation re-admits primitive event evidence and installs no event or flow.
The separate certificate wrapper requires a prospectively frozen instrument;
it evaluates the actual changed-capacity law once under the shared fixed budget.
Neither grammatical closure nor the earlier UM certificate supplies retention.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from time import perf_counter

from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import validated_box_taylor_step
from ..physics._sine_flow import _held_sine_neighbor_field
from .local_capacity_history import MODEL
from .maintained_parent_thol import _signature
from .maintained_principal_retention import HORIZON, certify_retention
from .maintained_sha_motion_source import validate_live_sha_successor
from .maintained_successor_chamber import _certify_chamber
from .maintained_successor_motion import MotionBounds, _bound_motion


@dataclass(frozen=True)
class ShaMotionPrediction:
    """Actual represented source and fresh bounds for its held complete law.

    Edges index ``nodes`` and use canonical lexicographic order. All coordinate
    and coefficient entries are exact represented rationals; no UM output
    allowance is inherited after SHA. Branch shifts refer to the unchanged
    real phase lifts, rather than independently wrapped endpoint differences.
    """

    nodes: tuple
    edges: tuple[tuple[int, int], ...]
    shifts: tuple[int, ...]
    epi: tuple[Q, ...]
    phases: tuple[Q, ...]
    capacities: tuple[Q, ...]
    conductances: tuple[Q, ...]
    bounds: MotionBounds
    horizon: Q


def prepare_sha_motion(admitted):
    """Rebuild the actual source, static bounds and all 42 held-law rows.

    The source reader verifies retained associations and reconstructs the SHA
    endpoint assessment. Cached predictions and mutable caller reports never
    replace those primitives. Neighbor rows retain the actual graph order;
    the shared field evaluates each unordered sine pair once. The structural
    clock, zero forcing/loss, support and positive represented capacities stay
    fixed. No event, pressure refresh or trajectory is evaluated here.
    """
    admitted = validate_live_sha_successor(admitted)
    graph = admitted.graph
    signature = _signature(graph)
    fields = admitted.assessment["actual_fields"]
    nodes = fields["nodes"]
    positions = {node: index for index, node in enumerate(nodes)}
    weights = {
        tuple(sorted((positions[a], positions[b]))): weight
        for (a, b), weight in zip(fields["edges"], fields["conductances"], strict=True)
    }
    edges = tuple(sorted(weights))
    epi, phases, capacities = (fields[key] for key in ("epi", "phase", "capacity"))
    conductances = tuple(weights[edge] for edge in edges)
    if len(nodes) != 21 or len(edges) != 53:
        raise ValueError("SHA continuation requires all 21 nodes and 53 edges")
    chamber = _certify_chamber(tuple(map(I, phases)), edges)
    bounds = _bound_motion(
        epi=epi,
        phases=phases,
        capacities=capacities,
        edges=edges,
        conductances=conductances,
        shifts=chamber.shifts,
        horizon=HORIZON,
    )
    rows = tuple(
        tuple(
            (
                positions[neighbor],
                1,
                weights[tuple(sorted((index, positions[neighbor])))],
            )
            for neighbor in graph[node]
        )
        for index, node in enumerate(nodes)
    )
    flow = _held_sine_neighbor_field(MODEL, tuple(map(len, rows)), capacities, rows)
    box = tuple(map(I, epi + phases))
    if _signature(graph) != signature:
        raise ValueError("read-only SHA preparation changed persistent state")
    prediction = ShaMotionPrediction(
        nodes,
        edges,
        chamber.shifts,
        epi,
        phases,
        capacities,
        conductances,
        bounds,
        HORIZON,
    )
    return prediction, box, flow


def certify_sha_retention(admitted, *, record_step, monotonic_clock=perf_counter):
    """Attempt only the frozen half-unit retention target, without retry.

    The caller owns prospective source/configuration freezing and durable
    recording. Preparation counts toward the fixed wall budget. This wrapper
    makes no turning-time, autonomous-occurrence or future-event prediction.
    """
    if not callable(record_step) or not callable(monotonic_clock):
        raise TypeError("recording and resource-clock callbacks must be callable")
    started = monotonic_clock()
    prediction, box, flow = prepare_sha_motion(admitted)
    return certify_retention(
        box,
        flow,
        prediction.edges,
        prediction.shifts,
        tuple(value.lo for value in prediction.bounds.horizon_remaining_margins),
        record_step=record_step,
        monotonic_clock=monotonic_clock,
        started=started,
        stepper=validated_box_taylor_step,
    )
