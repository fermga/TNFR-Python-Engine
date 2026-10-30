"""One fixed complete-law matched-cut response certificate.

The twenty-node host evolves under the shared held sine field. The isolated
child remains an exact primitive constant and is reinserted into every marked
observation. Each generated cell is checked again, including its solution jets
and whole-tube Taylor remainder, before its endpoint can be carried forward.
This owner chooses no new grid, response threshold or retry after evaluation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from time import perf_counter

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..mathematics._rational_interval import I
from ..mathematics._validated_taylor import (
    ValidatedBoxTaylorStep,
    flow_jets,
    reconstruct_box_taylor_arithmetic,
    validated_box_taylor_step,
)
from ..physics._sine_flow import _held_sine_neighbor_field
from ..physics.relational_observations import _ordered
from .local_capacity_history import MODEL, TIMES
from .maintained_parent_thol import _signature
from .maintained_principal_retention import HORIZON, ORDER, STEP, STEP_COUNT
from .maintained_sha_cut_separator import (
    CutSeparatorSource,
    _capture_cut_source,
    _fresh_cut,
)
from .maintained_sha_direct_response import (
    DirectResponseAggregate,
    DirectResponseCell,
    DirectResponseSource,
    _aggregate,
    _bound_cell,
)
from .maintained_sha_motion_source import validate_live_sha_successor

WALL_BUDGET_SECONDS = 2400


@dataclass(frozen=True)
class CutResponsePreparation:
    """Fresh original primitives, supplied cut law and exact host embedding.

    ``source`` retains the connected preparation and the cut host's indices.
    The observation source instead uses cut edges/degrees. Its zero child beta
    is the supplied frozen row, never capacity divided by degree zero. Host
    mean weights are d_cut/nu; the excluded child has weight zero. ``initial``
    orders all host forms followed by all host continuous phase lifts.
    """

    source: CutSeparatorSource
    observation_source: DirectResponseSource
    initial: tuple[I, ...]
    source_receipts: tuple[dict, ...] = ()
    persistent_source_unchanged: bool = True


@dataclass(frozen=True)
class CutStepVerification:
    """Fresh derivative and inclusion checks for one already generated cell."""

    tube_rates: tuple[I, ...]
    picard_interior_margin: Q
    solution_order: int = ORDER
    remainder_order: int = ORDER + 1
    series_verified: bool = True
    remainder_verified: bool = True
    arithmetic_verified: bool = True


@dataclass(frozen=True)
class ShaCutResponseAttempt:
    """A bounded method outcome with only its fully admitted prefix.

    ``completed`` requires the entire original window and an admitted signed
    angular integral. It does not require a clockwise orientation or a passing
    2*pi diagnostic. Failed cells remain in the caller's durable record rather
    than being promoted to accepted motion. No partial prefix supplies a
    full-window response, and failure does not establish escape or equality.
    """

    preparation: CutResponsePreparation
    cells: tuple[DirectResponseCell, ...]
    aggregate: DirectResponseAggregate
    status: str
    reason: str | None
    attempted_steps: int
    accepted_steps: int
    certified_time: Q
    elapsed_seconds: float
    scope: tuple[str, ...] = (
        "fresh_original_source_same_preparation_held_capacities_conductances_and_clock",
        "delete_only_actual_bridge_rebuild_degrees_freeze_both_isolated_child_rows",
        "complete_host_field_all_forms_and_continuous_phase_lifts_no_frozen_environment",
        "fixed_256_steps_of_one_over_512_order12_dyadic128_no_retry_or_adaptation",
        "strict_Picard_and_all_source_jets_whole_tube_remainders_rechecked_per_step",
        "exact_endpoint_carry_before_next_step_and_durable_attempt_record_before_acceptance",
        "both_host_means_checked_without_state_projection_or_interval_narrowing",
        "original_marked_plane_direct_signed_cross_product_and_positive_radius",
        "signed_clockwise_increment_available_independently_of_net_turn_diagnostic",
        "no_connected_rerun_event_pressure_refresh_or_empirical_error_model",
        "no_formation_autonomous_identity_periodicity_or_physical_identification_claim",
    )


def _prepare_cut_response(graph, *, parent, child, reference_model):
    """Reuse primitive/cut admission, without computing either separator."""
    signature = _signature(graph)
    source = _capture_cut_source(
        graph, parent=parent, child=child, reference_model=reference_model
    )
    # The shared owner verifies the nontrivial host and the complete frozen
    # isolate rule. No cached cut assessment or separator output is consumed.
    _fresh_cut(graph, source, reference_model)
    host = source.host_indices
    local = {index: position for position, index in enumerate(host)}
    positions = {node: index for index, node in enumerate(source.nodes)}
    weights = {
        frozenset(edge): weight
        for edge, weight in zip(source.cut_edges, source.cut_conductances, strict=True)
    }
    rows = tuple(
        tuple(
            (
                local[positions[neighbor]],
                1,
                weights[frozenset((index, positions[neighbor]))],
            )
            for neighbor in graph[source.nodes[index]]
            if positions[neighbor] != source.child_index
        )
        for index in host
    )
    degrees = tuple(source.cut_degrees[index] for index in host)
    if tuple(map(len, rows)) != degrees or any(degree <= 0 for degree in degrees):
        raise ValueError("cut host rows must retain their complete positive degrees")
    capacities = tuple(source.capacity[index] for index in host)
    flow = _held_sine_neighbor_field(reference_model, degrees, capacities, rows)

    size = len(source.nodes)
    a, c = source.parent_index, source.child_index
    beta = tuple(
        source.capacity[i] / source.cut_degrees[i] if i != c else Q(0)
        for i in range(size)
    )
    mean_weights = tuple(
        Q(source.cut_degrees[i]) / source.capacity[i] if i != c else Q(0)
        for i in range(size)
    )
    edge_coefficients, phase_coefficients = [], [Q(0)] * size
    for (i, j), weight in zip(source.cut_edges, source.cut_conductances, strict=True):
        # The child row is identically zero. Combine the negative parent row
        # in exact rationals before the shared marked observation evaluates it.
        signed = -beta[a] * ((i == a) - (j == a))
        edge_coefficients.append(signed)
        phase_coefficients[i] += signed * weight
        phase_coefficients[j] -= signed * weight
    observation_source = DirectResponseSource(
        source.nodes,
        source.cut_edges,
        source.cut_conductances,
        source.epi,
        source.phase,
        source.capacity,
        source.cut_degrees,
        beta,
        a,
        c,
        source.reference_conductance,
        source.structural_time,
        tuple(edge_coefficients),
        tuple(phase_coefficients),
        mean_weights,
        source.form_mean,
        source.phase_lift_mean,
    )
    initial = tuple(I(source.epi[i]) for i in host) + tuple(
        I(source.phase[i]) for i in host
    )
    preparation = CutResponsePreparation(source, observation_source, initial)
    original = tuple(map(I, source.epi + source.phase))
    if (
        _embed_host_state(preparation, initial) != original
        or _project_full_state(preparation, original) != initial
    ):
        raise ValueError("cut host embedding changed an original primitive coordinate")
    if _signature(graph) != signature:
        raise ValueError("read-only cut preparation changed persistent source state")
    return preparation, flow


def _embed_host_state(preparation, host_state):
    """Insert the same frozen child into a complete host state box."""
    if type(preparation) is not CutResponsePreparation:
        raise TypeError("fresh cut response preparation required")
    source = preparation.source
    host = source.host_indices
    values = _ordered(host_state, "complete cut host state", limit=2 * len(host) + 1)
    if len(values) != 2 * len(host):
        raise ValueError("cut state must retain every host form and phase lift")
    values = tuple(map(I.coerce, values))
    forms, phases = list(map(I, source.epi)), list(map(I, source.phase))
    for local, index in enumerate(host):
        forms[index], phases[index] = values[local], values[len(host) + local]
    return tuple(forms + phases)


def _project_full_state(preparation, full_state):
    """Remove only the supplied constant child, preserving host coordinate order."""
    if type(preparation) is not CutResponsePreparation:
        raise TypeError("fresh cut response preparation required")
    source = preparation.source
    size = len(source.nodes)
    values = _ordered(full_state, "complete cut state", limit=2 * size + 1)
    if len(values) != 2 * size:
        raise ValueError("full cut state must retain all original coordinates")
    values = tuple(map(I.coerce, values))
    c = source.child_index
    if values[c] != I(source.epi[c]) or values[size + c] != I(source.phase[c]):
        raise ValueError("the isolated child's form and phase must remain fixed")
    return tuple(values[i] for i in source.host_indices) + tuple(
        values[size + i] for i in source.host_indices
    )


def prepare_sha_cut_response(admitted):
    """Re-admit the original fixed source once and prepare its complete cut law."""
    admitted = validate_live_sha_successor(admitted)
    graph = admitted.graph
    before = _signature(graph)
    if len(graph) != 21 or graph.number_of_edges() != 53:
        raise ValueError(
            "SHA cut response requires the retained 21-node 53-edge source"
        )
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if clock != TIMES[-1]:
        raise ValueError("SHA cut response requires the retained structural clock")
    preparation, flow = _prepare_cut_response(
        graph, parent=2, child="2_sub_0", reference_model=MODEL
    )
    if _signature(graph) != before:
        raise ValueError("read-only SHA cut preparation changed persistent state")
    return replace(preparation, source_receipts=deepcopy(admitted.receipts)), flow


def _verify_step(step, initial, flow, *, index):
    """Recompute every consumed derivative and inclusion on the saved boxes.

    This is a second arithmetic verification of the same generated cell,
    not another trajectory. The actual initial box and whole-time tube are
    retained; no smaller tube or alternate Taylor order is substituted.
    """
    if type(step) is not ValidatedBoxTaylorStep:
        raise ValueError("a complete shared source-box Taylor step is required")
    if type(index) is not int or not 0 <= index < STEP_COUNT:
        raise ValueError("step index must belong to the fixed chronological grid")
    if (
        type(step.time) not in (int, Q)
        or step.time != index * STEP
        or type(step.duration) not in (int, Q)
        or step.duration != STEP
        or type(step.order) is not int
        or step.order != ORDER
        or step.method != "direct_source_box_Picard_Taylor_dyadic128_v1"
    ):
        raise ValueError(
            "Taylor step differs from the fixed clock, duration, order or method"
        )
    initial = tuple(map(I.coerce, _ordered(initial, "initial cut host state")))
    if not initial or len(initial) % 2 or len(initial) > 64:
        raise ValueError("complete host forms and phases are required")
    if step.initial_box != initial or len(step.tube) != len(initial):
        raise ValueError("Taylor step changed its exact source carry or dimension")
    if step.domain_lower_bounds != (Q(1),):
        raise ValueError("cut field must retain its declared global smooth domain")
    if any(
        not value.subset_of(bound)
        for value, bound in zip(initial, step.tube, strict=True)
    ):
        raise ValueError("whole-time tube omits an initial coordinate")
    rates = tuple(map(I.coerce, flow(step.tube)))
    if len(rates) != len(initial):
        raise ValueError("complete cut field changed state dimension")
    image = tuple(
        value + rate * I(0, STEP) for value, rate in zip(initial, rates, strict=True)
    )
    interior = min(
        min(value.lo - bound.lo, bound.hi - value.hi)
        for value, bound in zip(image, step.tube, strict=True)
    )
    if interior <= 0 or step.picard_interior_margin != interior:
        raise ValueError(
            "strict Picard inclusion or its recorded margin failed verification"
        )
    series = flow_jets(initial, ORDER, flow)
    if step.series != series:
        raise ValueError(
            "source Taylor coefficients differ from full derivative re-evaluation"
        )
    remainder = tuple(
        row[-1] * STEP ** (ORDER + 1) for row in flow_jets(step.tube, ORDER + 1, flow)
    )
    if step.local_remainder_bounds != remainder:
        raise ValueError(
            "whole-tube Taylor remainder differs from derivative re-evaluation"
        )
    try:
        increment, endpoint = reconstruct_box_taylor_arithmetic(
            initial, step.tube, series, remainder, STEP, order=ORDER
        )
    except (TypeError, ArithmeticError) as error:
        raise ValueError(
            "Taylor arithmetic cannot be independently reconstructed"
        ) from error
    if step.increment != increment or step.endpoint != endpoint:
        raise ValueError(
            "Taylor increment or endpoint differs from reconstructed arithmetic"
        )
    return CutStepVerification(rates, interior)


def _clock_value(clock):
    return exact_or_represented_real(clock(), "resource clock")


def _certify_cut_response(
    preparation, flow, *, record_step, monotonic_clock, started, stepper=None
):
    """Attempt the fixed window once, retaining every failed or accepted attempt.

    The optional stepper is for manufactured orchestration controls. Scientific
    callers use the default shared producer and freeze its exact implementation.
    Resource limits are checked at stage boundaries; an external hard monitor
    owns interruption, resident-memory and durable-output limits.
    """
    if type(preparation) is not CutResponsePreparation:
        raise TypeError("fresh cut response preparation required")
    if not all(callable(item) for item in (flow, record_step, monotonic_clock)):
        raise TypeError("field, recording and resource clock must be callable")
    if stepper is None:
        stepper = validated_box_taylor_step
    if not callable(stepper):
        raise TypeError("the shared enclosure stepper must be callable")
    started = exact_or_represented_real(started, "resource start")
    last_clock = started

    def elapsed():
        nonlocal last_clock
        now = _clock_value(monotonic_clock)
        if now < last_clock:
            raise ValueError("resource clock moved backwards")
        last_clock = now
        return now - started

    def exhausted():
        return elapsed() >= WALL_BUDGET_SECONDS

    box = preparation.initial
    if _project_full_state(preparation, _embed_host_state(preparation, box)) != box:
        raise ValueError("cut host preparation has inconsistent coordinate association")
    cells = []
    attempted = 0
    status, reason = "completed", None
    for index in range(STEP_COUNT):
        if exhausted():
            status, reason = (
                "resource_budget_exhausted",
                "fixed_wall_boundary_budget_exhausted",
            )
            break
        attempted += 1
        time = index * STEP
        step = failed_tube = verification = observation = None
        accepted = False
        try:
            step, failed_tube, failure = stepper(
                box, STEP, flow, lambda _: (Q(1),), order=ORDER, time=time
            )
        except AssertionError:
            raise
        except Exception as error:
            status, reason = "solver_error", f"{type(error).__name__}: {error}"
        else:
            if step is None:
                status, reason = (
                    "enclosure_unavailable",
                    failure or "shared_enclosure_unavailable",
                )
            elif failed_tube is not None or failure is not None:
                status, reason = (
                    "verification_failed",
                    "producer_returned_conflicting_step_and_failure",
                )
            elif exhausted():
                status, reason = (
                    "resource_budget_exhausted",
                    "wall_budget_after_step_production",
                )
            else:
                try:
                    verification = _verify_step(step, box, flow, index=index)
                except AssertionError:
                    raise
                except Exception as error:
                    status, reason = (
                        "verification_failed",
                        f"{type(error).__name__}: {error}",
                    )
                else:
                    if exhausted():
                        status, reason = (
                            "resource_budget_exhausted",
                            "wall_budget_after_step_verification",
                        )
                    else:
                        try:
                            full_tube = _embed_host_state(preparation, step.tube)
                            observation = _bound_cell(
                                preparation.observation_source,
                                full_tube,
                                index=index,
                                time=time,
                                duration=STEP,
                            )
                        except AssertionError:
                            raise
                        except Exception as error:
                            status, reason = (
                                "observation_unavailable",
                                f"{type(error).__name__}: {error}",
                            )
                        else:
                            missing = list(observation.unavailable_reasons)
                            if not all(observation.means_consistent):
                                missing.append("both_conserved_host_means_required")
                            if observation.radius_squared.lo <= 0:
                                missing.append(
                                    "strict_positive_angular_denominator_unavailable"
                                )
                            if (
                                observation.angular_rate is None
                                or observation.angular_increment is None
                            ):
                                missing.append("signed_angular_increment_unavailable")
                            if missing:
                                status, reason = "observation_unavailable", ",".join(
                                    missing
                                )
                            elif exhausted():
                                status, reason = (
                                    "resource_budget_exhausted",
                                    "wall_budget_after_step_observation",
                                )
                            else:
                                accepted = True
        record_step(
            dict(
                index=index,
                time=time,
                step=step,
                failed_tube=failed_tube,
                verification=verification,
                observation=observation,
                accepted=accepted,
                reason=reason,
            )
        )
        if not accepted:
            break
        # Durable recording precedes promotion of the state and observation.
        cells.append(observation)
        box = step.endpoint
        if exhausted():
            status, reason = (
                "resource_budget_exhausted",
                "wall_budget_after_durable_record",
            )
            break
    count = len(cells)
    final_elapsed = elapsed()
    if status == "completed" and final_elapsed >= WALL_BUDGET_SECONDS:
        status, reason = (
            "resource_budget_exhausted",
            "wall_budget_before_final_aggregation",
        )
    aggregate = _aggregate(
        tuple(cells), full_window=count == STEP_COUNT and status == "completed"
    )
    if status == "completed" and (
        count != STEP_COUNT
        or aggregate.radius_gate.status != "pass"
        or aggregate.accepted_duration != HORIZON
        or aggregate.clockwise_travel is None
    ):
        status, reason = (
            "observation_unavailable",
            "fixed_full_window_signed_response_unavailable",
        )
    return ShaCutResponseAttempt(
        preparation,
        tuple(cells),
        aggregate,
        status,
        reason,
        attempted,
        count,
        count * STEP,
        float(final_elapsed),
    )


def certify_sha_cut_response(
    admitted, *, record_step, monotonic_clock=perf_counter, started=None
):
    """Prepare and attempt one prospectively frozen complete matched-cut run.

    Artifact drivers may split public preparation and the private loop to keep
    old historical admission checks separate from new cut-kernel accounting.
    This convenience boundary includes fresh preparation in its wall budget.
    """
    if not callable(record_step) or not callable(monotonic_clock):
        raise TypeError("recording and resource clock must be callable")
    if started is None:
        started = _clock_value(monotonic_clock)
    else:
        started = exact_or_represented_real(started, "resource start")
    preparation, flow = prepare_sha_cut_response(admitted)
    return _certify_cut_response(
        preparation,
        flow,
        record_step=record_step,
        monotonic_clock=monotonic_clock,
        started=started,
    )
