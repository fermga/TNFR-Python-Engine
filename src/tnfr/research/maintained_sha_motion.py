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

from .._exact_time import finite_represented_real
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    cos,
    pi_interval,
    sin,
    sqrt,
)
from ..mathematics._validated_taylor import validated_box_taylor_step
from ..physics._sine_admission import _sine_model_coefficients
from ..physics._sine_flow import _held_sine_neighbor_field
from ..physics.relational_sine_support import (
    SineFrozenIsolateComparison,
    SineSupportExchangeComparison,
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from ..sdk._state import copy_graph_state
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


@dataclass(frozen=True)
class PairPortSnapshot:
    """Initial marked observation of one freshly captured complete law.

    The cut keeps the connected bridge coefficient as an observation only.
    Its child is explicitly frozen, has zero mean weight and no inverse
    positive mean metric. Displayed means concern its nontrivial component.
    These are instantaneous enclosures, never a later port or rotation bound.
    """

    source: SineSupportExchangeComparison | SineFrozenIsolateComparison
    parent_index: int
    child_index: int
    external_indices: tuple[int, ...]
    reference_conductance: Q
    form_difference: Q
    phase_difference: Q
    external_sine_sum: I
    external_form_gradient: Q
    parent_mobility: I
    child_mobility: I
    internal_rotation_coefficient: I
    pair_energy: I
    radial_signal: I
    port_norm: I
    form_difference_rate: I
    phase_difference_rate: I
    pair_row_residuals: tuple[I, I]
    pair_energy_rate: I
    power_residual: I
    phase_portrait_rate: I | None
    angle_unavailable_reason: str | None
    mean_weights: tuple[Q, ...]
    positive_degree_indices: tuple[int, ...]
    form_mean: Q
    phase_lift_mean: Q


@dataclass(frozen=True)
class PairBoundaryBudget:
    """Static uniform-norm certificate diagnostics, without an identity fit.

    ``energy_port_norm_lower_limit`` is the infimum of the connected global
    energy *bound* as E_low approaches initial E from below. It is not a lower
    bound on the physical port norm. At this matched preparation it equals
    the cut's global port bound, because H_connected - E_initial = H_cut.
    In particular, the cut bound never subtracts its marked observation E.

    Necessary margins are threshold minus alpha; ``rotation_margin`` means
    the optimistic necessary margin s-alpha, not an angular rate. These
    diagnostics apply only when ``initial_annulus_status`` is ``eligible``.
    A nonpositive certified margin excludes the specified sufficient
    certificate family only. No margin establishes a whole-window port
    bound, turn or formation verdict.
    """

    connected: PairPortSnapshot
    cut: PairPortSnapshot
    structural_time: Q
    external_degree: int
    external_conductance_sum: Q
    joint_energy_coefficient: Q
    initial_environment_storage: I
    storage_decomposition_residual: I
    joint_ellipse_left: I
    joint_ellipse_residual: I
    energy_port_norm_lower_limit: I
    cut_global_port_norm_bound: I
    initial_norm_alpha: I | None
    energy_bound_alpha: I | None
    necessary_alpha_threshold: I
    initial_norm_necessary_margin: I | None
    energy_bound_necessary_margin: I | None
    initial_norm_rotation_margin: I | None
    energy_bound_rotation_margin: I | None
    uniform_norm_unavailable_reason: str | None
    initial_principal_branch_certified: bool
    initial_positive_subseparatrix_energy_certified: bool
    initial_annulus_status: str
    initial_annulus_reasons: tuple[str, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_primitive_complete_beta1_zero_loss_sine_law",
        "held_positive_conductances_and_capacities_original_structural_clock",
        "matched_cut_deletes_only_pendant_edge_and_freezes_both_child_rows",
        "global_energy_bound_lower_limit_is_not_a_physical_port_lower_bound",
        "necessary_margin_excludes_only_the_uniform_norm_certificate_family",
        "no_annulus_fit_trajectory_event_refresh_or_formation_evaluation",
    )


def _pair_port_snapshot(source, *, parent_index, child_index, external, weight, cut):
    """Derive marked quantities from an internally captured source only."""
    a, c = parent_index, child_index
    r, delta = source.epi[c] - source.epi[a], source.phase[c] - source.phase[a]
    current = sum(
        (sin(I(source.phase[j] - source.phase[a])) for j, _ in external), I(0)
    )
    gradient = sum((w * (source.epi[a] - source.epi[j]) for j, w in external), Q(0))
    pi, root_weight = pi_interval(), sqrt(I(weight))
    ma = I(source.capacity[a]) / (pi * source.degrees[a])
    mc = I(0) if cut else I(source.capacity[c]) / (pi * source.degrees[c])
    k = I(0) if cut else ma + mc
    energy = I(weight * r**2 / 2) + 1 - cos(I(delta))
    signal = sqrt(2 * energy)
    norm = sqrt(I(gradient) ** 2 + weight * current**2)
    form_rate = source.form_rates[c] - source.form_rates[a]
    phase_rate = source.phase_rates[c] - source.phase_rates[a]
    pair_form_rate = -k * sin(I(delta)) - ma * current
    pair_phase_rate = k * weight * r - ma * gradient
    energy_rate = -ma * (weight * r * current + sin(I(delta)) * gradient)
    radius_squared = delta**2 + weight * r**2
    radius = I(radius_squared)
    angle_rate = (
        root_weight * (delta * form_rate - r * phase_rate) / radius
        if radius.lo > 0
        else None
    )
    angle_reason = None
    if angle_rate is None:
        angle_reason = (
            "zero_relative_state_radius"
            if not radius_squared
            else "relative_state_radius_not_separated_from_zero"
        )
    rho = tuple(
        Q(d) / nu for d, nu in zip(source.degrees, source.capacity, strict=True)
    )
    total = sum(rho, Q(0))

    def mean(values):
        return sum((v * q for v, q in zip(values, rho, strict=True)), Q(0)) / total

    return PairPortSnapshot(
        source=source,
        parent_index=a,
        child_index=c,
        external_indices=tuple(j for j, _ in external),
        reference_conductance=weight,
        form_difference=r,
        phase_difference=delta,
        external_sine_sum=current,
        external_form_gradient=gradient,
        parent_mobility=ma,
        child_mobility=mc,
        internal_rotation_coefficient=k * root_weight,
        pair_energy=energy,
        radial_signal=signal,
        port_norm=norm,
        form_difference_rate=form_rate,
        phase_difference_rate=phase_rate,
        pair_row_residuals=(form_rate - pair_form_rate, phase_rate - pair_phase_rate),
        pair_energy_rate=energy_rate,
        power_residual=weight * r * form_rate
        + sin(I(delta)) * phase_rate
        - energy_rate,
        phase_portrait_rate=angle_rate,
        angle_unavailable_reason=angle_reason,
        mean_weights=rho,
        positive_degree_indices=tuple(i for i, d in enumerate(source.degrees) if d),
        form_mean=mean(source.epi),
        phase_lift_mean=mean(source.phase),
    )


def _bound_pair_budget(graph, *, parent, child, reference_model):
    """Capture a positive pendant pair and its static matched-cut control.

    This private research consumer admits graph primitives through shared
    owners. It takes no cached report, fitted identity width or future bound.
    Manufactured controls may use another graph under the same beta=1 law.
    """
    if _sine_model_coefficients(reference_model) != (Q(0), Q(1), Q(1)):
        raise ValueError("pair budget requires the beta=1 unit-exchange zero-loss law")
    _, clock = finite_represented_real(graph.graph.get("_t"), "structural clock")
    if any("weight" not in data for _, _, data in graph.edges(data=True)):
        raise ValueError("pair budget requires stored conductance on every edge")
    source = bound_relational_sine_support_exchange(
        graph, reference_model=reference_model
    )
    if any(v <= 0 for v in source.capacity + source.conductances):
        raise ValueError("pair budget requires positive capacities and conductances")
    positions = {node: i for i, node in enumerate(source.nodes)}
    if parent == child or parent not in positions or child not in positions:
        raise ValueError("distinct parent and child must belong to the source")
    a, c = positions[parent], positions[child]
    if source.degrees[c] != 1 or not graph.has_edge(parent, child):
        raise ValueError("the marked child must be pendant at the marked parent")
    if source.degrees[a] < 2:
        raise ValueError("the marked parent requires at least one external neighbor")
    weights = {
        frozenset(edge): value
        for edge, value in zip(source.edges, source.conductances, strict=True)
    }
    weight = weights[frozenset((parent, child))]
    external = tuple(
        (positions[j], weights[frozenset((parent, j))])
        for j in graph[parent]
        if j != child
    )
    control = copy_graph_state(graph)
    control.remove_edge(parent, child)
    cut_source = bound_relational_sine_frozen_isolates(
        control, reference_model=reference_model
    )
    connected = _pair_port_snapshot(
        source,
        parent_index=a,
        child_index=c,
        external=external,
        weight=weight,
        cut=False,
    )
    cut = _pair_port_snapshot(
        cut_source,
        parent_index=a,
        child_index=c,
        external=external,
        weight=weight,
        cut=True,
    )
    n, strength = len(external), sum((w for _, w in external), Q(0))
    coefficient = max(strength, weight * n)
    # Direct edge deletion computes H-E without subtraction of overlapping
    # enclosures. On the cut this is its ENTIRE H, not H minus observed E.
    environment = cut_source.storage
    energy_norm = sqrt(2 * coefficient * environment)
    A = connected.internal_rotation_coefficient
    alpha0 = connected.parent_mobility * connected.port_norm / A if A.lo > 0 else None
    alpha_energy = connected.parent_mobility * energy_norm / A if A.lo > 0 else None
    threshold = connected.radial_signal / (1 + sqrt(2 * pi_interval())) ** 2
    ellipse = (
        I(connected.external_form_gradient**2 / strength)
        + connected.external_sine_sum**2 / n
    )
    principal = abs(connected.phase_difference) < pi_interval().lo
    positive_energy = connected.pair_energy.lo > 0 and connected.pair_energy.hi < 2
    reasons = []
    excluded = False
    if not principal:
        excluded = abs(connected.phase_difference) >= pi_interval().hi
        reasons.append("initial_zero_principal_branch_not_certified")
    if not positive_energy:
        excluded |= connected.pair_energy.hi <= 0 or connected.pair_energy.lo >= 2
        reasons.append("initial_strict_positive_subseparatrix_energy_not_certified")
    return PairBoundaryBudget(
        connected=connected,
        cut=cut,
        structural_time=clock,
        external_degree=n,
        external_conductance_sum=strength,
        joint_energy_coefficient=coefficient,
        initial_environment_storage=environment,
        storage_decomposition_residual=source.storage
        - connected.pair_energy
        - environment,
        joint_ellipse_left=ellipse,
        joint_ellipse_residual=2 * environment - ellipse,
        energy_port_norm_lower_limit=energy_norm,
        cut_global_port_norm_bound=energy_norm,
        initial_norm_alpha=alpha0,
        energy_bound_alpha=alpha_energy,
        necessary_alpha_threshold=threshold,
        initial_norm_necessary_margin=(
            threshold - alpha0 if alpha0 is not None else None
        ),
        energy_bound_necessary_margin=(
            threshold - alpha_energy if alpha_energy is not None else None
        ),
        initial_norm_rotation_margin=(
            connected.radial_signal - alpha0 if alpha0 is not None else None
        ),
        energy_bound_rotation_margin=(
            connected.radial_signal - alpha_energy if alpha_energy is not None else None
        ),
        uniform_norm_unavailable_reason=(
            None
            if A.lo > 0
            else "internal_rotation_coefficient_not_separated_from_zero"
        ),
        initial_principal_branch_certified=principal,
        initial_positive_subseparatrix_energy_certified=positive_energy,
        initial_annulus_status=(
            "excluded" if excluded else ("unavailable" if reasons else "eligible")
        ),
        initial_annulus_reasons=tuple(reasons),
    )


def prepare_sha_pair_budget(admitted):
    """Re-admit retained SHA primitives and bound only the registered snapshot.

    This installs no law or flow and reads no retained continuation jets.
    Calling it is a static source evaluation requiring its own prospective
    freeze; the report does not choose identity parameters or a response time.
    """
    admitted = validate_live_sha_successor(admitted)
    graph = admitted.graph
    before = _signature(graph)
    if tuple(graph) != tuple(range(20)) + ("2_sub_0",) or graph.number_of_edges() != 53:
        raise ValueError("SHA pair budget requires the retained 21-node 53-edge source")
    result = _bound_pair_budget(graph, parent=2, child="2_sub_0", reference_model=MODEL)
    if result.structural_time != Q(7) + Q(1, 2**19):
        raise ValueError("SHA pair budget requires the retained structural clock")
    if _signature(graph) != before:
        raise ValueError("read-only SHA pair budget changed persistent state")
    return result
