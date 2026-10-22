"""Frozen finite registration on one admitted weighted-star successor.

The full ten-node law closes on five exact nominal blocks. RK4 supplies an
independent numerical trace; retained metric Taylor balls certify whole-time
observations. Neither trace supplies events or an effective collective NFR.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from math import pi, sin
from time import perf_counter

from ..dynamics.integrators import rk4_coupled_step
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ..mathematics._validated_metric import (
    metric_taylor_observation_boxes,
    validated_metric_taylor_step,
)
from ..physics._sine_flow import _held_sine_neighbor_field
from ..physics.relational_sine_support import bound_relational_sine_support_exchange
from ..physics.relational_sine_weighted_partition import (
    assess_sine_weighted_phase_offset_partition,
)
from .artifact_io import encode_exact_tree, exact_record
from .retained_graph_state import record_graph_state, restore_recorded_graph_state
from .weighted_star_contact import (
    COLLECTIVE_BLOCKS,
    MODEL,
    NEW_EDGES,
    OLD_EDGES,
    OLD_WEIGHTS,
    POST_DEGREES,
    PROBES,
    contact_source,
    execute_contact_word,
)

MULTIPLICITIES = tuple(map(len, COLLECTIVE_BLOCKS))


def frozen_protocol():
    """Prospective policy; changing it requires a separately named experiment."""
    return {
        "schema": "tnfr.weighted-star-registration-episode.v1",
        "source": "central_canonical_AL_UM_SHA_successor",
        "horizon": Q(8),
        "integration_step": Q(1, 1024),
        "certificate_step": Q(1, 16),
        "certificate_order": 12,
        "certificate_bins": 16,
        "minimum_residence": Q(1, 16),
        "registration_threshold": Q(1, 4),
        "form_scale": Q(1),
        "phase_scale_radians": Q(1),
        "maximum_metric_radius": Q(1, 4096),
        "maximum_rk4_metric_error": Q(1, 4096),
        "wall_budget_seconds": 900,
        "branches": (
            {"name": "nonlinear", "gamma": Q(1)},
            {"name": "tangent", "gamma": Q(0)},
        ),
        "clock": "original_structural_time; elapsed zero at graph time seven",
        "numerical_method": "shared_classical_RK4_binary64_refreshed_both_rows",
        "certificate_method": "shared_metric_Taylor_dyadic128_whole_bins",
        "stopping": "fixed_horizon_or_first_branch_failure_or_global_wall_budget",
        "selection": "first_certified_run; unresolved_bins_do_not_establish_absence",
        "preparation_uncertainty": "exact_represented_nominal_only",
        "inputs": "none; held_support_conductance_capacity; no_events_or_Gamma",
    }


def _admit_protocol(protocol):
    # Canonical serialization distinguishes True from 1 and 1.0 from an exact
    # integer; Python mapping equality alone is not scalar admission.
    if encode_exact_tree(protocol) != encode_exact_tree(frozen_protocol()):
        raise ValueError("protocol differs from the frozen finite experiment")
    for key, expected in frozen_protocol().items():
        if type(expected) in (int, Q) and type(protocol[key]) is not type(expected):
            raise ValueError("protocol scalar representation differs")
    for supplied, expected in zip(protocol["branches"], frozen_protocol()["branches"]):
        if type(supplied["gamma"]) is not Q or supplied != expected:
            raise ValueError("growth policy differs")


def _primitive_signature(source):
    return {
        "nodes": source.nodes,
        "edges": source.edges,
        "conductances": source.conductances,
        "form": source.epi,
        "phase_lifts": source.phase,
        "capacity": source.capacity,
        "degrees": source.degrees,
        "graph_time": Q(7),
    }


def prepare_source():
    """Execute only the previously admitted static word, with its full context."""
    states, stages, _ = execute_contact_word(contact_source(*PROBES[0]))
    source = bound_relational_sine_support_exchange(states[-1], reference_model=MODEL)
    return {
        "schema": "tnfr.weighted-star-central-preparation.v1",
        "word": ("emission", "coupling", "silence"),
        "target": 4,
        "static_states": tuple(record_graph_state(graph) for graph in states),
        "event_count": len(stages),
        "primitive": _primitive_signature(source),
    }


def admit_preparation(preparation):
    """Re-admit primitive law/state and the nominal reduction hypotheses.

    The immutable preparation and its separately tested static producer own
    the association of conductance bytes with UM. This reader does not replay
    events or authenticate that history from numerical coordinate equality.
    """
    if (
        preparation["schema"] != "tnfr.weighted-star-central-preparation.v1"
        or tuple(preparation["word"]) != ("emission", "coupling", "silence")
        or type(preparation["target"]) is not int
        or preparation["target"] != 4
        or type(preparation["event_count"]) is not int
        or preparation["event_count"] != 3
        or len(preparation["static_states"]) != 4
    ):
        raise ValueError("central preparation association differs")
    graph = restore_recorded_graph_state(preparation["static_states"][-1]).graph
    if graph.graph.get("GAMMA") != {"type": "none"}:
        raise ValueError("source requires the declared absent Gamma")
    if type(graph.graph.get("_t")) is not int or graph.graph["_t"] != 7:
        raise ValueError("source graph clock differs")
    source = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    if encode_exact_tree(_primitive_signature(source)) != encode_exact_tree(
        preparation["primitive"]
    ):
        raise ValueError("primitive source disagrees with the retained graph")
    expected_form = (Q(5, 16),) * 4 + (Q(257, 1024),) + (Q(5, 16),) * 4 + (Q(1, 4),)
    expected_phase = (Q(0),) * 5 + (Q(201, 512),) * 5
    expected_capacity = (
        Q(4),
        Q(2),
        Q(2),
        Q(2),
        Q(117, 32),
        Q(4),
        Q(2),
        Q(2),
        Q(2),
        Q(8),
    )
    weights = dict(zip(source.edges, source.conductances, strict=True))
    if (
        source.nodes != tuple(range(10))
        or set(source.edges) != set(OLD_EDGES + NEW_EDGES)
        or source.degrees != POST_DEGREES
        or source.epi != expected_form
        or source.phase != expected_phase
        or source.capacity != expected_capacity
        or any(weights[edge] != weight for edge, weight in zip(OLD_EDGES, OLD_WEIGHTS))
        or any(not Q(9, 10) < weights[edge] < 1 for edge in NEW_EDGES)
        or len({weights[edge] for edge in NEW_EDGES[:4]}) != 1
        or source.storage.hi >= Q(1, 2)
    ):
        raise ValueError("central nominal source or growth hypotheses differ")
    partition = assess_sine_weighted_phase_offset_partition(
        source, blocks=COLLECTIVE_BLOCKS, phase_offset_turns=(Q(0),) * 10
    )
    state = partition.observe_radian_state()
    if not (state.source_membership_certified and state.full_row_equality_certified):
        raise ValueError("full source does not close in the exact nominal chart")
    return source, partition, state


def _metric(source, partition):
    blocks = partition.node_blocks
    size = len(COLLECTIVE_BLOCKS)
    matrices = [[[Q(0) for _ in range(size)] for _ in range(size)] for _ in range(2)]
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        i, j = blocks[left], blocks[right]
        if i == j:
            continue
        for matrix, value in zip(matrices, (weight, Q(1)), strict=True):
            matrix[i][i] += value
            matrix[j][j] += value
            matrix[i][j] -= value
            matrix[j][i] -= value
    rho = partition.block_mean_weights
    total = sum(rho)
    return tuple(
        tuple(
            matrices[b][i][j] + rho[i] * rho[j] / total if b == bb else Q(0)
            for bb in range(2)
            for j in range(size)
        )
        for b in range(2)
        for i in range(size)
    )


def prepare_fields(source, partition, branch):
    """Retain actual representative degrees, capacities and both full rows."""
    if branch not in ("nonlinear", "tangent"):
        raise ValueError("unknown complete law")
    representatives = tuple(block[0] for block in partition.block_indices)
    degrees = tuple(source.degrees[i] for i in representatives)
    capacity = tuple(source.capacity[i] for i in representatives)
    rows = []
    for index in representatives:
        row = []
        for target in range(len(representatives)):
            neighbors = [
                j if i == index else i
                for i, j in source.edges
                if index in (i, j)
                and partition.node_blocks[j if i == index else i] == target
            ]
            weights = dict(zip(source.edges, source.conductances, strict=True))
            if neighbors:
                row.append(
                    (
                        target,
                        len(neighbors),
                        sum(weights[tuple(sorted((index, j)))] for j in neighbors),
                    )
                )
        rows.append(tuple(row))
    rows = tuple(rows)
    mobility = tuple(nu / degree for nu, degree in zip(capacity, degrees, strict=True))
    nonlinear = _held_sine_neighbor_field(MODEL, degrees, capacity, rows)

    def tangent(values):
        forms, phases = values[:5], values[5:]
        return tuple(
            sum((count * (phases[j] - phases[i]) for j, count, _ in row), 0)
            * mobility[i]
            / pi_interval()
            for i, row in enumerate(rows)
        ) + tuple(
            sum((weight * (forms[i] - forms[j]) for j, _, weight in row), 0)
            * mobility[i]
            / pi_interval()
            for i, row in enumerate(rows)
        )

    float_rows = tuple(
        tuple((j, count, float(weight)) for j, count, weight in row) for row in rows
    )
    float_mobility = tuple(float(value) for value in mobility)

    def numerical(_time, values):
        forms, phases = values[:5], values[5:]
        phase_current = sin if branch == "nonlinear" else lambda value: value
        return tuple(
            float_mobility[i]
            / pi
            * sum(count * phase_current(phases[j] - phases[i]) for j, count, _ in row)
            for i, row in enumerate(float_rows)
        ) + tuple(
            float_mobility[i]
            / pi
            * sum(weight * (forms[i] - forms[j]) for j, _, weight in row)
            for i, row in enumerate(float_rows)
        )

    return (nonlinear if branch == "nonlinear" else tangent), numerical


def registration_bounds(values):
    """Full ten-node Q via the exact multiplicity-weighted pair identity.

    Its equality to the circular observation uses the separately proved hub
    short-arc theorem for these two admitted trajectories, not this function.
    """
    if len(values) != 10:
        raise ValueError("registration retains all five forms and five phase lifts")
    values = tuple(I.coerce(value) for value in values)
    return sum(
        (
            Q(MULTIPLICITIES[i] * MULTIPLICITIES[j], 10)
            * ((values[i] - values[j]) ** 2 + (values[i + 5] - values[j + 5]) ** 2)
            for i in range(5)
            for j in range(i + 1, 5)
        ),
        I(0),
    )


def storage_bounds(values, source, partition, *, tangent=False):
    values = tuple(I.coerce(value) for value in values)
    result = I(0)
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        i, j = partition.node_blocks[left], partition.node_blocks[right]
        gap = values[i + 5] - values[j + 5]
        result += weight * (values[i] - values[j]) ** 2 / 2
        result += gap**2 / 2 if tangent else 1 - cos(gap)
    return result


def _interval_record(value):
    return {"lo": value.lo, "hi": value.hi}


def evaluate_protocol(preparation, protocol, *, progress=None):
    """Evaluate once after external freeze; preserve every carried certificate."""
    _admit_protocol(protocol)
    started = perf_counter()
    source, partition, state = admit_preparation(preparation)
    metric = _metric(source, partition)
    initial = state.collective_form + state.collective_phase_radians
    if storage_bounds(initial, source, partition, tangent=True).hi >= Q(1, 2):
        raise ValueError("tangent source fails the retained global short-arc bound")
    h, dt = protocol["certificate_step"], protocol["integration_step"]
    results = []
    for branch in protocol["branches"]:
        flow, numerical = prepare_fields(source, partition, branch["name"])
        center, radius, trajectory = initial, Q(0), tuple(map(float, initial))
        records, reason = [], None

        def domain(box):
            if branch["name"] == "tangent":
                return (Q(1),)
            return (1 - storage_bounds(box, source, partition).hi,)

        for index in range(int(protocol["horizon"] / h)):
            if perf_counter() - started >= protocol["wall_budget_seconds"]:
                reason = "global_wall_budget_exhausted"
                break
            elapsed = index * h
            step, failed, failure = validated_metric_taylor_step(
                center,
                radius,
                metric,
                h,
                flow,
                domain,
                growth_rate=branch["gamma"],
                order=protocol["certificate_order"],
                time=Q(7) + elapsed,
                domain_failure="whole_tube_storage_growth_domain_unavailable",
            )
            if step is None:
                reason = failure
                break
            if step.endpoint_radius > protocol["maximum_metric_radius"]:
                reason = "frozen_metric_error_allowance_exceeded"
                break
            bins = metric_taylor_observation_boxes(
                step, flow, bin_count=protocol["certificate_bins"]
            )
            for substep in range(int(h / dt)):
                trajectory = rk4_coupled_step(
                    trajectory,
                    numerical,
                    t=float(Q(7) + elapsed + substep * dt),
                    dt=float(dt),
                )
            difference = tuple(
                Q(value) - exact
                for value, exact in zip(trajectory, step.endpoint_center, strict=True)
            )
            squared_error = sum(
                difference[i] * metric[i][j] * difference[j]
                for i in range(10)
                for j in range(10)
            )
            numerical_error = sqrt(I(squared_error)).hi + step.endpoint_radius
            if numerical_error > protocol["maximum_rk4_metric_error"]:
                reason = "frozen_RK4_error_allowance_exceeded"
                break
            record = {
                "index": index,
                "elapsed_start": elapsed,
                "elapsed_end": elapsed + h,
                "structural_start": step.time,
                "initial_center": step.initial_center,
                "initial_radius": step.initial_radius,
                "endpoint_center": step.endpoint_center,
                "endpoint_radius": step.endpoint_radius,
                "tube": tuple(map(_interval_record, step.tube)),
                "domain_lower_bounds": step.domain_lower_bounds,
                "picard_margin": step.picard_interior_margin,
                "growth_factor": step.growth_factor_upper_bound,
                "local_remainder": tuple(
                    map(_interval_record, step.local_remainder_bounds)
                ),
                "local_metric_error": step.local_metric_error_upper_bound,
                "rk4_endpoint": tuple(Q(value) for value in trajectory),
                "rk4_metric_error_upper": numerical_error,
                "bins": tuple(
                    {
                        "start": elapsed + lo,
                        "end": elapsed + hi,
                        "box": tuple(map(_interval_record, box)),
                        "registration": _interval_record(registration_bounds(box)),
                    }
                    for lo, hi, box in bins
                ),
            }
            records.append(record)
            if progress is not None:
                progress(branch["name"], index, deepcopy(record))
            center, radius = step.endpoint_center, step.endpoint_radius
        results.append(
            {"name": branch["name"], "records": tuple(records), "stop_reason": reason}
        )
    return {
        "schema": protocol["schema"],
        "primitive_source": _primitive_signature(source),
        "initial_center": initial,
        "metric": metric,
        "multiplicities": MULTIPLICITIES,
        "branches": tuple(results),
        "elapsed_wall_seconds": perf_counter() - started,
        "initial_registration": _interval_record(registration_bounds(initial)),
        "physical_identification_claimed": False,
        "new_collective_NFR_claimed": False,
    }


def _read_interval(record):
    if type(record) is not dict or set(record) != {"lo", "hi"}:
        raise ValueError("invalid interval record")
    return I(exact_record(record["lo"]), exact_record(record["hi"]))


def summarize_response(response, protocol):
    """Rebuild Q from primitive observation boxes, never a cached verdict.

    This consumes an associated solver response. It does not authenticate its
    provenance or replace verification of the field/Picard/Taylor certificate.
    """
    _admit_protocol(protocol)
    if response["schema"] != protocol["schema"]:
        raise ValueError("response schema differs")
    summaries = []
    for branch, declaration in zip(
        response["branches"], protocol["branches"], strict=True
    ):
        if branch["name"] != declaration["name"]:
            raise ValueError("response law order differs")
        expected, run, episode = Q(0), None, None
        all_outside = True
        minimum_lower, minimum_upper = None, None
        previous = tuple(exact_record(value) for value in response["initial_center"])
        radius = Q(0)
        for index, record in enumerate(branch["records"]):
            if (
                type(record["index"]) is not int
                or record["index"] != index
                or exact_record(record["elapsed_start"]) != expected
                or exact_record(record["elapsed_end"])
                != expected + protocol["certificate_step"]
                or exact_record(record["structural_start"]) != 7 + expected
                or tuple(exact_record(value) for value in record["initial_center"])
                != previous
                or exact_record(record["initial_radius"]) != radius
                or len(record["bins"]) != protocol["certificate_bins"]
            ):
                raise ValueError(
                    "certificate clock/state carry or cell association differs"
                )
            width = protocol["certificate_step"] / protocol["certificate_bins"]
            for bin_record in record["bins"]:
                start, end = (exact_record(bin_record[key]) for key in ("start", "end"))
                if start != expected or end != start + width:
                    raise ValueError(
                        "whole-time bins must be contiguous at the frozen width"
                    )
                box = tuple(_read_interval(value) for value in bin_record["box"])
                value = registration_bounds(box)
                minimum_lower = (
                    value.lo if minimum_lower is None else min(minimum_lower, value.lo)
                )
                minimum_upper = (
                    value.hi if minimum_upper is None else min(minimum_upper, value.hi)
                )
                all_outside &= value.lo > protocol["registration_threshold"]
                if value.hi <= protocol["registration_threshold"]:
                    run = start if run is None else run
                    if episode is None and end - run >= protocol["minimum_residence"]:
                        episode = {"start": run, "end": end}
                else:
                    run = None
                expected = end
            previous = tuple(exact_record(value) for value in record["endpoint_center"])
            radius = exact_record(record["endpoint_radius"])
            if not 0 <= radius <= protocol["maximum_metric_radius"]:
                raise ValueError("retained metric radius exceeds the frozen allowance")
        complete = expected == protocol["horizon"]
        summaries.append(
            {
                "name": branch["name"],
                "status": (
                    "finite_episode_certified"
                    if episode is not None
                    else (
                        "finite_horizon_no_entry_certified"
                        if complete and all_outside
                        else "episode_unavailable_within_frozen_budget"
                    )
                ),
                "first_certified_run": episode,
                "validated_duration": expected,
                "minimum_registration_lower_bound": minimum_lower,
                "minimum_bin_upper_bound": minimum_upper,
                "maximum_endpoint_metric_radius": max(
                    (
                        exact_record(item["endpoint_radius"])
                        for item in branch["records"]
                    ),
                    default=Q(0),
                ),
                "stop_reason": branch["stop_reason"],
            }
        )
    return {
        "branches": tuple(summaries),
        "scope": "central_nominal_only; finite_registration_not_new_NFR_or_physical_identity",
    }
