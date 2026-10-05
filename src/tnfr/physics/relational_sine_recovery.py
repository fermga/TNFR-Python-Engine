"""Local geometry, recovery and conservative identity for sine critical targets.

Targets have exact rational turns on the complete supplied support; uniform
cycle twists retain a specialized spectral bound. Criticality follows from
symbolic edge-current cancellation, not from a rounded phase residual.
Uncertain observations and validated endpoints retain different state-set
provenance. Dissipative recovery and conservative identity use the same local
barrier with separate law admission. No trajectory or support event is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sqrt
from .phase_cycle_geometry import PhaseCycleGeometry, PhaseCycleState
from .phase_cycle_geometry import _derive as _derive_phase_geometry
from .phase_cycle_geometry import reconstruct_phase_cycle_state
from .relational_observations import _ordered
from .relational_sine_comparison import (
    SineExchangeComparison,
    _validate_comparison_labels,
)
from .relational_sine_pattern import SineRelativeForecast, SineRelativePattern
from .structural_diffusion import _exact_real_laplacian_gap_lower_bound

__all__ = (
    "SineCycleRecovery",
    "SinePatternRecovery",
    "SineSectorCapture",
    "SineCycleIdentityAssessment",
    "certify_sine_cycle_recovery",
    "certify_sine_pattern_recovery",
    "certify_sine_sector_capture",
    "assess_sine_cycle_identity",
)


@dataclass(frozen=True)
class SineCycleRecovery:
    """Sufficient recovery of every state in the declared uncertainty set.

    An unavailable report means some sufficient condition is unresolved or
    a theorem hypothesis is absent, not that the state is unstable. A forecast
    endpoint is assessed only at its actual validated time, even when the
    requested propagation horizon was not reached.
    """

    source: SineRelativePattern | SineRelativeForecast
    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    winding: int
    phase_turns: tuple[int, ...]
    target_phase_turns: tuple[Q, ...]
    radius: Q
    uncertainty_scope: str
    observation_time: Q | None
    input_forecast_admitted: bool | None
    input_forecast_requested_end_time: Q | None
    capacity_bounds: tuple[I, ...]
    exact_held_capacity: tuple[Q | None, ...]
    spectral_gap_bounds: I
    spectral_gap_lower_bound: Q
    target_edge_angle_bounds: I
    target_phase_storage_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    form_norm_squared_upper_bound: Q
    phase_norm_squared_upper_bound: Q
    norm_squared_upper_bound: Q
    form_edge_gap_bounds: tuple[I, ...]
    phase_deviation_edge_bounds: tuple[I, ...]
    phase_hessian_upper_bounds: tuple[Q, ...]
    form_energy_upper_bound: Q
    phase_excess_upper_bound: Q
    excess_storage_upper_bound: Q
    barrier_lower_bound: Q | None
    norm_margin: Q
    energy_margin: Q | None
    hypothesis_failures: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_simple_unit_Cn_and_declared_integer_winding_with_4_abs_winding_lt_n",
        "exact_target_turns_prove_sine_criticality_by_opposite_edge_cancellation",
        "explicit_integer_turns_added_to_observed_lifts_without_automatic_unwrapping",
        "pairwise_centered_norm_preserves_common_offset_cancellation",
        "phase_linear_energy_terms_cancel_before_quadratic_remainder_bounds",
        "strict_local_radius_and_excess_storage_barrier_for_the_whole_input_set",
        "positive_epi_weight_and_strictly_positive_held_capacity_required",
        "exact_held_capacity_constraints_survive_wider_arithmetic_enclosures",
        "capacity_family_admission_does_not_supply_a_uniform_recovery_rate",
        "conditional_recovery_to_uniform_form_and_target_phase_modulo_common_origins",
        "unavailable_is_not_instability_or_an_existence_claim_for_measurement_data",
        "no_trajectory_support_event_autonomous_formation_or_physical_identity_claim",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        pattern = (
            self.source.pattern
            if isinstance(self.source, SineRelativeForecast)
            else self.source
        )
        _validate_comparison_labels(pattern)
        return {
            "schema": "tnfr.relational-sine-cycle-recovery.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SinePatternRecovery:
    """Full-support recovery about an exactly reconstructed critical target.

    The target geometry contributes only circular reconstruction and symbolic
    sine cancellation. Its separate equal-capacity phase-lock interpretation
    is not consumed. Recovery follows the complete sine law with the supplied
    strictly positive held capacities, including heterogeneous capacities.
    """

    source: SineRelativePattern | SineRelativeForecast
    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    target_geometry: PhaseCycleState
    target_edge_turns: tuple[Q, ...]
    phase_turns: tuple[int, ...]
    target_phase_turns: tuple[Q, ...]
    radius: Q
    uncertainty_scope: str
    observation_time: Q | None
    input_forecast_admitted: bool | None
    input_forecast_requested_end_time: Q | None
    capacity_bounds: tuple[I, ...]
    exact_held_capacity: tuple[Q | None, ...]
    spectral_gap_lower_bound: Q
    spectral_gap_method: str
    target_edge_angle_bounds: tuple[I, ...]
    maximum_target_angle_bounds: I
    target_phase_storage_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    form_norm_squared_upper_bound: Q
    phase_norm_squared_upper_bound: Q
    norm_squared_upper_bound: Q
    form_edge_gap_bounds: tuple[I, ...]
    phase_deviation_edge_bounds: tuple[I, ...]
    phase_hessian_upper_bounds: tuple[Q, ...]
    form_energy_upper_bound: Q
    phase_excess_upper_bound: Q
    excess_storage_upper_bound: Q
    barrier_lower_bound: Q | None
    norm_margin: Q
    energy_margin: Q | None
    hypothesis_failures: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_supplied_connected_simple_unit_support_all_nodes_and_incident_edges",
        "exact_rational_target_turns_with_certified_acute_wrapped_edge_angles",
        "symbolic_odd_sine_cancellation_proves_target_criticality_not_a_small_residual",
        "phase_geometry_reconstruction_only_not_its_separate_equal_capacity_lock_model",
        "target_geometry_retains_its_32_node_50_edge_numerical_evaluation_budget",
        "exact_rational_combinatorial_Laplacian_quotient_gap_not_a_float_spectrum",
        "shared_whole_set_norm_Taylor_remainder_and_coercive_recovery_barrier",
        "regional_relative_offsets_intermediaries_and_hidden_coordinates_are_retained",
        "strictly_positive_held_capacities_and_positive_epi_weight_required",
        "exact_known_capacities_survive_wider_arithmetic_capacity_enclosures",
        "not_a_sum_of_independent_regional_recovery_certificates",
        "no_new_trajectory_support_event_autonomous_formation_or_physical_identification",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        pattern = (
            self.source.pattern
            if isinstance(self.source, SineRelativeForecast)
            else self.source
        )
        _validate_comparison_labels(pattern)
        return {
            "schema": "tnfr.relational-sine-pattern-recovery.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineSectorCapture:
    """Target-free capture inside one supplied complete acute phase sector.

    Every face has a certified lower bound, including faces the relaxation
    cannot prove feasible. A strict total-storage comparison certifies an
    interior equilibrium and convergence without supplying its coordinates.
    Unavailability is not a proof of instability or absence of equilibrium.
    """

    source: SineExchangeComparison | SineRelativePattern | SineRelativeForecast
    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    geometry: PhaseCycleGeometry
    edge_turn_offsets: tuple[int, ...]
    cycle_periods: tuple[int, ...]
    uncertainty_scope: str
    observation_time: Q | None
    input_forecast_admitted: bool | None
    input_forecast_requested_end_time: Q | None
    capacity_bounds: tuple[I, ...]
    exact_held_capacity: tuple[Q | None, ...]
    form_edge_gap_bounds: tuple[I, ...]
    phase_edge_gap_bounds: tuple[I, ...]
    edge_acute_margin_bounds: tuple[I, ...]
    form_storage_bounds: I
    phase_storage_bounds: I
    storage_bounds: I
    storage_upper_bound: Q
    boundary_face_lower_bounds: tuple[Q, ...]
    boundary_phase_storage_lower_bound: Q
    boundary_storage_lower_bound: Q
    energy_margin: Q
    weighted_form_mean: Q | None
    weighted_phase_mean: Q | None
    weighted_mean_scope: str
    hypothesis_failures: tuple[str, ...]
    unresolved_conditions: tuple[str, ...]
    status: str
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_connected_simple_unit_support_and_normalized_sine_two_row_law",
        "positive_held_capacities_positive_form_loss_exchange_and_storage_scale",
        "canonical_edge_offsets_are_independent_integers_not_nodal_target_coordinates",
        "full_acute_cell_modulo_common_phase_with_all_signed_edge_boundary_faces",
        "cycle_periods_derived_by_exact_telescoping_in_the_integer_fundamental_basis",
        "whole_source_set_acute_admission_and_full_form_plus_phase_storage",
        "all_face_supporting_hyperplane_lower_bounds_not_sampled_boundary_values",
        "strict_storage_barrier_proves_interior_existence_uniqueness_and_capture",
        "equilibrium_coordinates_and_recovery_time_are_not_supplied_or_estimated",
        "weighted_form_and_lifted_phase_means_are_conserved_per_complete_state",
        "relative_sources_leave_their_absolute_common_origins_unobserved",
        "forecast_endpoint_set_is_read_at_its_actual_reported_validated_time_only",
        "structural_revalidation_and_public_dataclasses_do_not_authenticate_source_provenance",
        "no_trajectory_event_selection_sector_creation_native_Arg_or_physical_identity_claim",
    )

    @property
    def admitted(self):
        return self.status == "admitted"

    def to_dict(self):
        from ..sdk.relational_reports import _project

        pattern = (
            self.source.pattern
            if isinstance(self.source, SineRelativeForecast)
            else self.source
        )
        _validate_comparison_labels(pattern)
        return {
            "schema": "tnfr.relational-sine-sector-capture.v1",
            "report": _project(self),
        }


def _full_cycle(pattern, cycle):
    size = len(pattern.nodes)
    if size < 3:
        raise ValueError("cycle recovery requires at least three nodes")
    cycle = _ordered(cycle, "cycle", limit=size + 1)
    if (
        len(cycle) != size
        or len(set(cycle)) != size
        or set(cycle) != set(pattern.nodes)
    ):
        raise ValueError("cycle must cover the complete supplied node set exactly once")
    expected = {
        frozenset((left, right)) for left, right in zip(cycle, cycle[1:] + cycle[:1])
    }
    if (
        len(pattern.edges) != size
        or {frozenset(edge) for edge in pattern.edges} != expected
    ):
        raise ValueError(
            "support must be exactly the declared full cycle without chords"
        )
    indices = {node: index for index, node in enumerate(pattern.nodes)}
    return cycle, tuple(indices[node] for node in cycle)


def _source_data(source):
    """Read a source already validated by the shared primitive admission."""
    if isinstance(source, SineExchangeComparison):

        def gap(left, right, *, phase=False):
            values = source.phase if phase else source.epi
            return I(values[right] - values[left])

        return (
            source,
            source.reference_model,
            tuple(I(value) for value in source.capacity),
            source.capacity,
            gap,
            "exact_captured_full_state_no_discarded_common_origin",
            None,
            None,
            None,
        )
    if isinstance(source, SineRelativePattern):
        pattern = source

        def gap(left, right, *, phase=False):
            values = pattern.nominal_phase if phase else pattern.nominal_form
            errors = pattern.phase_error_bounds if phase else pattern.form_error_bounds
            center = values[right] - values[left]
            error = errors[right] + errors[left]
            return I(center - error, center + error)

        return (
            pattern,
            pattern.reference_model,
            tuple(I(value) for value in pattern.capacity),
            pattern.capacity,
            gap,
            "original_correlated_common_offset_plus_residual_observation_set",
            None,
            None,
            None,
        )
    if not isinstance(source, SineRelativeForecast):
        raise TypeError("a SineRelativePattern or SineRelativeForecast is required")
    pattern, full = source.pattern, source.full_forecast
    size = len(pattern.nodes)
    endpoint = tuple(I.coerce(value) for value in full.endpoint)

    def gap(left, right, *, phase=False):
        offset = size if phase else 0
        return endpoint[offset + right] - endpoint[offset + left]

    capacities = tuple(I(value) for value in full.visible_capacity) + (endpoint[-1],)
    return (
        pattern,
        full.model,
        capacities,
        full.visible_capacity + (None,),
        gap,
        "full_validated_endpoint_rectangular_box_not_original_residual_or_projected_box",
        full.validated_end_time,
        full.admitted,
        full.end_time,
    )


def _target_geometry(pattern, target):
    """Reuse the exact circular reconstruction and symbolic sine owner."""
    positions = {node: index for index, node in enumerate(pattern.nodes)}
    indices = tuple(
        (positions[left], positions[right]) for left, right in pattern.edges
    )
    geometry = _derive_phase_geometry(
        pattern.nodes,
        tuple(sorted((min(left, right), max(left, right)) for left, right in indices)),
    )

    def principal_turn(left, right):
        return (target[right] - target[left] + Q(1, 2)) % 1 - Q(1, 2)

    reconstructed = reconstruct_phase_cycle_state(
        geometry,
        edge_turns=tuple(principal_turn(left, right) for left, right in geometry.edges),
    )
    if reconstructed.sine_balance_status != "proved_by_odd_cancellation":
        raise ValueError("target sine balance is not proved by exact odd cancellation")
    return (
        reconstructed,
        indices,
        tuple(principal_turn(left, right) for left, right in indices),
    )


def _target_admission(pattern, *, cycle, winding, target_phase_turns):
    """Select a declared exact target and its independently justified gap."""
    size, pi = len(pattern.nodes), pi_interval()
    if target_phase_turns is None:
        cycle, indices = _full_cycle(pattern, cycle)
        if type(winding) is not int or 4 * abs(winding) >= size:
            raise ValueError(
                "winding must be a nonboolean integer with 4*abs(winding)<n"
            )
        target = [Q(0)] * size
        for position, index in enumerate(indices):
            target[index] = Q(winding * position, size)
        alpha = Q(2 * winding, size) * pi
        spectral_gap = 2 - 2 * cos(Q(2, size) * pi)
        return (
            tuple(target),
            tuple(zip(indices, indices[1:] + indices[:1])),
            (alpha,) * size,
            Q(2 * abs(winding), size) * pi,
            spectral_gap.lo,
            dict(
                cycle=cycle,
                cycle_indices=indices,
                winding=winding,
                spectral_gap_bounds=spectral_gap,
                target_edge_angle_bounds=alpha,
                target_phase_storage_bounds=size * (1 - cos(alpha)),
            ),
        )
    raw = _ordered(target_phase_turns, "target_phase_turns", limit=size + 1)
    if len(raw) != size or any(
        type(value) is not int and not isinstance(value, Q) for value in raw
    ):
        raise ValueError(
            "target_phase_turns must contain one exact integer or Fraction per node"
        )
    target = tuple(map(Q, raw))
    geometry, indices, turns = _target_geometry(pattern, target)
    neighbors = [set() for _ in pattern.nodes]
    for left, right in indices:
        neighbors[left].add(right)
        neighbors[right].add(left)
    laplacian = tuple(
        tuple(
            Q(len(neighbors[i]) if i == j else -int(j in neighbors[i]))
            for j in range(size)
        )
        for i in range(size)
    )
    gap_lower, uniform = _exact_real_laplacian_gap_lower_bound(laplacian)
    if not uniform:
        raise ArithmeticError(
            "the full-support Laplacian failed its uniform-mode check"
        )
    angles = tuple((2 * turn) * pi for turn in turns)
    maximum_angle = (2 * max(map(abs, turns))) * pi
    return (
        target,
        indices,
        angles,
        maximum_angle,
        gap_lower,
        dict(
            target_geometry=geometry,
            target_edge_turns=turns,
            target_edge_angle_bounds=angles,
            maximum_target_angle_bounds=maximum_angle,
            target_phase_storage_bounds=sum((1 - cos(angle) for angle in angles), I(0)),
            spectral_gap_method="exact_rational_combinatorial_Laplacian_quotient_lower_bound",
        ),
    )


def _local_sine_geometry(
    source,
    *,
    radius,
    phase_turns=None,
    cycle=None,
    winding=None,
    target_phase_turns=None,
):
    """Shared source, target, norm and barrier bounds before law admission."""
    admitted_source, _ = _admit_sine_source(source)
    (
        pattern,
        model,
        capacities,
        exact_capacity,
        gap,
        uncertainty,
        at,
        forecast_ok,
        requested,
    ) = _source_data(admitted_source)
    size = len(pattern.nodes)
    target, edge_indices, edge_angles, maximum_angle, gap_lower, target_fields = (
        _target_admission(
            pattern, cycle=cycle, winding=winding, target_phase_turns=target_phase_turns
        )
    )
    radius = exact_or_represented_real(radius, "radius")
    if radius <= 0:
        raise ValueError("radius must be strictly positive")
    turns = (
        (0,) * size
        if phase_turns is None
        else _ordered(phase_turns, "phase_turns", limit=size + 1)
    )
    if len(turns) != size or any(type(value) is not int for value in turns):
        raise ValueError("phase_turns must contain one nonboolean integer per node")
    pi = pi_interval()

    def deviation(left, right):
        # Combine exact turn coefficients before enclosing mathematical pi.
        coefficient = turns[right] - turns[left] - target[right] + target[left]
        return gap(left, right, phase=True) + (2 * coefficient) * pi

    form_norm = phase_norm = Q(0)
    for i in range(size):
        for j in range(i + 1, size):
            form_norm += gap(i, j).abs_max ** 2 / size
            phase_norm += deviation(i, j).abs_max ** 2 / size
    norm = form_norm + phase_norm
    edge_form, edge_phase, hessian = [], [], []
    form_energy = phase_excess = Q(0)
    beta = Q(model.storage_scale)
    for (left, right), alpha in zip(edge_indices, edge_angles):
        form_gap, phase_gap = gap(left, right), deviation(left, right)
        # The linear phase term cancels at the exact critical geometry. Bound
        # the remaining Taylor Hessian on the entire segment to that target.
        upper = cos(alpha + I(0).hull(phase_gap)).hi
        edge_form.append(form_gap)
        edge_phase.append(phase_gap)
        hessian.append(upper)
        form_energy += form_gap.abs_max**2 / 2
        phase_excess += beta * upper * phase_gap.abs_max**2 / 2
    excess = form_energy + phase_excess
    radius_angle = maximum_angle + sqrt(I(2)) * radius
    acute_margin = pi / 2 - radius_angle
    cosine = cos(radius_angle).lo if acute_margin.lo > 0 else None
    coercivity = (
        gap_lower * min(Q(1), beta * cosine) / 2
        if gap_lower > 0 and cosine is not None and cosine > 0
        else None
    )
    barrier = None if coercivity is None else coercivity * radius**2
    norm_margin = radius**2 - norm
    energy_margin = None if barrier is None else barrier - excess
    capacity_failures = tuple(
        reason
        for condition, reason in (
            (
                all(
                    exact > 0 if exact is not None else enclosure.lo > 0
                    for exact, enclosure in zip(exact_capacity, capacities)
                ),
                "strictly_positive_held_capacity_required",
            ),
        )
        if not condition
    )
    geometry_unresolved = tuple(
        reason
        for condition, reason in (
            (
                acute_margin.lo > 0 and cosine is not None and cosine > 0,
                "strict_acute_radius_not_certified",
            ),
            (gap_lower > 0, "positive_spectral_gap_not_certified"),
        )
        if not condition
    )
    source_unresolved = tuple(
        reason
        for condition, reason in (
            (norm_margin > 0, "strict_initial_radius_not_certified"),
            (
                energy_margin is not None and energy_margin > 0,
                "strict_excess_storage_barrier_not_certified",
            ),
        )
        if not condition
    )
    common = dict(
        source=source,
        reference_model=model,
        nodes=pattern.nodes,
        edges=pattern.edges,
        phase_turns=turns,
        target_phase_turns=target,
        radius=radius,
        uncertainty_scope=uncertainty,
        observation_time=at,
        input_forecast_admitted=forecast_ok,
        input_forecast_requested_end_time=requested,
        capacity_bounds=capacities,
        exact_held_capacity=exact_capacity,
        spectral_gap_lower_bound=gap_lower,
        radius_angle_bounds=radius_angle,
        acute_radius_margin_bounds=acute_margin,
        cosine_lower_bound=cosine,
        coercivity_lower_bound=coercivity,
        form_norm_squared_upper_bound=form_norm,
        phase_norm_squared_upper_bound=phase_norm,
        norm_squared_upper_bound=norm,
        form_edge_gap_bounds=tuple(edge_form),
        phase_deviation_edge_bounds=tuple(edge_phase),
        phase_hessian_upper_bounds=tuple(hessian),
        form_energy_upper_bound=form_energy,
        phase_excess_upper_bound=phase_excess,
        excess_storage_upper_bound=excess,
        barrier_lower_bound=barrier,
        norm_margin=norm_margin,
        energy_margin=energy_margin,
    )
    return (
        common,
        target_fields,
        capacity_failures,
        geometry_unresolved,
        source_unresolved,
    )


def _certify_recovery(
    source,
    *,
    radius,
    phase_turns=None,
    cycle=None,
    winding=None,
    target_phase_turns=None,
):
    """Apply dissipative recovery admission to the shared local geometry."""
    if not isinstance(source, (SineRelativePattern, SineRelativeForecast)):
        raise TypeError("a SineRelativePattern or SineRelativeForecast is required")
    common, target_fields, capacity_failures, geometry_unresolved, source_unresolved = (
        _local_sine_geometry(
            source,
            radius=radius,
            phase_turns=phase_turns,
            cycle=cycle,
            winding=winding,
            target_phase_turns=target_phase_turns,
        )
    )
    failures = (
        ()
        if common["reference_model"].effective_weights[0] > 0
        else ("positive_epi_weight_required",)
    ) + capacity_failures
    unresolved = geometry_unresolved + source_unresolved
    report_type = (
        SineCycleRecovery if target_phase_turns is None else SinePatternRecovery
    )
    return report_type(
        **common,
        **target_fields,
        hypothesis_failures=failures,
        unresolved_conditions=unresolved,
        status="unavailable" if failures or unresolved else "admitted",
    )


def certify_sine_cycle_recovery(
    source, *, cycle, winding, radius, phase_turns=None
) -> SineCycleRecovery:
    """Certify a sufficient full-cycle recovery domain without evolving it.

    In the declared cycle order the exact target has turns k*j/n. The optional
    integer sequence is added to observed phases in captured node order;
    neither target nor observation lifts are chosen from a small residual.
    Pattern observations retain original pair errors, whereas forecasts retain
    their entire validated endpoint box and actual capacity enclosure.
    """
    return _certify_recovery(
        source, cycle=cycle, winding=winding, radius=radius, phase_turns=phase_turns
    )


def certify_sine_pattern_recovery(
    source, *, target_phase_turns, radius, phase_turns=None
) -> SinePatternRecovery:
    """Certify one exact acute critical geometry on the complete support.

    Target coordinates are exact integers or Fractions in turns, in captured
    node order. Wrapped target edge turns must be strictly acute and their
    sine currents must cancel by the shared symbolic oddness proof. An
    unsupported target raises before any critical-point energy bound is used.
    This sufficient symbolic test does not decide all trigonometric identities.

    The shared target owner imposes its 32-node/50-edge execution budget. Its
    reconstruction and criticality results do not import another phase law.
    The full unit Laplacian supplies a rational certified spectral-gap lower
    bound. All regional offsets, intermediary coordinates, edges and capacities
    remain in the common recovery norm and energy; isolated certificates are
    not added. No source graph, solver, stored response or controller is read.
    """
    if target_phase_turns is None:
        raise ValueError("an explicit exact target_phase_turns sequence is required")
    return _certify_recovery(
        source,
        target_phase_turns=target_phase_turns,
        radius=radius,
        phase_turns=phase_turns,
    )


def _admit_sine_source(source):
    """Compatibility adapter for the shared budget-neutral source owner."""
    from ._sine_admission import _admit_sine_source as admit

    return admit(source)


def _sector_source_admission(source):
    """Compatibility adapter retaining the sector geometry's work budget."""
    from ._sine_admission import _sector_source_admission as admit

    return admit(source)


def certify_sine_sector_capture(source, *, edge_turn_offsets) -> SineSectorCapture:
    """Certify full acute-sector capture without an equilibrium target.

    Integer offsets follow the canonical edges of the returned geometry:
    ``delta_e=theta_head-theta_tail+2*pi*offset_e``. These independent
    edge integers declare the cell; their fundamental cycle sums give its
    periods. No phase unwrapping, response fit or equilibrium solve is used.

    The entire source set must be strictly acute and its full storage upper
    bound must lie below a certified lower bound on every signed boundary
    face. The shared boundary relaxation is sufficient, not necessarily
    sharp. Unsupported numerical budgets and malformed declarations raise;
    unresolved strict comparisons or missing theorem premises are unavailable.
    """
    admitted_source, geometry = _sector_source_admission(source)
    (
        pattern,
        model,
        capacities,
        exact_capacity,
        gap,
        uncertainty,
        at,
        forecast_ok,
        requested,
    ) = _source_data(admitted_source)
    form_mean = phase_mean = None
    mean_scope = "absolute_common_form_and_phase_origins_unobserved"
    if isinstance(admitted_source, SineExchangeComparison):
        mean_scope = "unavailable_without_strictly_positive_held_capacity"
        if all(value > 0 for value in admitted_source.capacity):
            from .relational_sine_comparison import _sine_form_weights

            weights = _sine_form_weights(admitted_source)
            total = sum(weights, Q(0))
            form_mean = (
                sum(
                    (
                        weight * value
                        for weight, value in zip(weights, admitted_source.epi)
                    ),
                    Q(0),
                )
                / total
            )
            phase_mean = (
                sum(
                    (
                        weight * value
                        for weight, value in zip(weights, admitted_source.phase)
                    ),
                    Q(0),
                )
                / total
            )
            mean_scope = (
                "exact_captured_weighted_form_and_declared_real_phase_lift_means"
            )
    return _certify_sine_sector_set(
        source=source,
        geometry=geometry,
        model=model,
        capacity_bounds=capacities,
        exact_held_capacity=exact_capacity,
        form_edge_gap_bounds=tuple(gap(i, j) for i, j in geometry.edges),
        phase_edge_gap_bounds=tuple(gap(i, j, phase=True) for i, j in geometry.edges),
        edge_turn_offsets=edge_turn_offsets,
        uncertainty_scope=uncertainty,
        observation_time=at,
        input_forecast_admitted=forecast_ok,
        input_forecast_requested_end_time=requested,
        weighted_form_mean=form_mean,
        weighted_phase_mean=phase_mean,
        weighted_mean_scope=mean_scope,
    )


def _certify_sine_sector_set(
    *,
    source,
    geometry,
    model,
    capacity_bounds,
    exact_held_capacity,
    form_edge_gap_bounds,
    phase_edge_gap_bounds,
    edge_turn_offsets,
    uncertainty_scope,
    observation_time=None,
    input_forecast_admitted=None,
    input_forecast_requested_end_time=None,
    weighted_form_mean=None,
    weighted_phase_mean=None,
    weighted_mean_scope="absolute_common_form_and_phase_origins_unobserved",
    correlated_form_storage_upper_bound=None,
    correlated_phase_storage_upper_bound=None,
):
    """Share the all-face capture proof for an admitted nonempty state set.

    Producers own source admission, full-state enclosure validity and its
    provenance. Edge rows retain their correlations through their proved
    bounds; they are not an independently realizable collection of samples.
    In particular an analytic endpoint is not relabeled as an observation
    or a Taylor forecast. Its source can retain the original preparation.
    Optional correlated storage upper bounds require an independent producer
    proof for this same set. The phase bound is the unscaled sine potential.
    They tighten the edgewise intervals without discarding their lower bounds.
    """
    from ._sine_sector_boundary import _sector_boundary_bounds

    edge_count, size = len(geometry.edges), len(geometry.nodes)
    if (
        len(capacity_bounds) != size
        or len(exact_held_capacity) != size
        or len(form_edge_gap_bounds) != edge_count
        or len(phase_edge_gap_bounds) != edge_count
    ):
        raise ValueError("capture state set must retain every node and edge")
    offsets = _ordered(edge_turn_offsets, "edge_turn_offsets", limit=edge_count + 1)
    if len(offsets) != edge_count or any(type(value) is not int for value in offsets):
        raise ValueError(
            "edge_turn_offsets requires one nonboolean integer per canonical edge"
        )
    periods = tuple(
        sum(sign * offset for sign, offset in zip(row, offsets))
        for row in geometry.cycle_rows
    )
    pi, beta = pi_interval(), Q(model.storage_scale)
    form_gaps = tuple(I.coerce(value) for value in form_edge_gap_bounds)
    phase_gaps = tuple(
        I.coerce(value) + (2 * offset) * pi
        for value, offset in zip(phase_edge_gap_bounds, offsets)
    )
    acute_margins = tuple(pi / 2 - abs(value) for value in phase_gaps)
    form_storage = sum((value**2 / 2 for value in form_gaps), I(0))
    phase_storage = sum((1 - cos(value) for value in phase_gaps), I(0))

    def intersect_storage(bounds, supplied, label):
        if supplied is None:
            return bounds
        upper = exact_or_represented_real(supplied, label)
        if upper < 0:
            raise ValueError(f"{label} must be nonnegative")
        if upper < bounds.lo:
            raise ValueError(
                f"{label} is inconsistent with the edgewise storage lower bound"
            )
        return I(bounds.lo, min(bounds.hi, upper))

    form_storage = intersect_storage(
        form_storage,
        correlated_form_storage_upper_bound,
        "correlated_form_storage_upper_bound",
    )
    phase_storage = intersect_storage(
        phase_storage,
        correlated_phase_storage_upper_bound,
        "correlated_phase_storage_upper_bound",
    )
    storage = form_storage + beta * phase_storage
    face_bounds = _sector_boundary_bounds(geometry, periods)
    if len(face_bounds) != 2 * edge_count:
        raise ArithmeticError(
            "sector boundary certificate must cover every signed edge face"
        )
    phase_barrier = min(face_bounds)
    barrier = beta * phase_barrier
    margin = barrier - storage.hi
    failures = tuple(
        reason
        for condition, reason in (
            (model.effective_weights[0] > 0, "positive_epi_weight_required"),
            (
                all(
                    value > 0 if value is not None else bounds.lo > 0
                    for value, bounds in zip(exact_held_capacity, capacity_bounds)
                ),
                "strictly_positive_held_capacity_required",
            ),
        )
        if not condition
    )
    unresolved = tuple(
        reason
        for condition, reason in (
            (
                all(value.lo > 0 for value in acute_margins),
                "whole_source_set_strict_acute_sector_not_certified",
            ),
            (margin > 0, "strict_total_storage_boundary_barrier_not_certified"),
        )
        if not condition
    )
    return SineSectorCapture(
        source=source,
        reference_model=model,
        nodes=geometry.nodes,
        edges=tuple((geometry.nodes[i], geometry.nodes[j]) for i, j in geometry.edges),
        geometry=geometry,
        edge_turn_offsets=offsets,
        cycle_periods=periods,
        uncertainty_scope=uncertainty_scope,
        observation_time=observation_time,
        input_forecast_admitted=input_forecast_admitted,
        input_forecast_requested_end_time=input_forecast_requested_end_time,
        capacity_bounds=capacity_bounds,
        exact_held_capacity=exact_held_capacity,
        form_edge_gap_bounds=form_gaps,
        phase_edge_gap_bounds=phase_gaps,
        edge_acute_margin_bounds=acute_margins,
        form_storage_bounds=form_storage,
        phase_storage_bounds=phase_storage,
        storage_bounds=storage,
        storage_upper_bound=storage.hi,
        boundary_face_lower_bounds=face_bounds,
        boundary_phase_storage_lower_bound=phase_barrier,
        boundary_storage_lower_bound=barrier,
        energy_margin=margin,
        weighted_form_mean=weighted_form_mean,
        weighted_phase_mean=weighted_phase_mean,
        weighted_mean_scope=weighted_mean_scope,
        hypothesis_failures=failures,
        unresolved_conditions=unresolved,
        status="unavailable" if failures or unresolved else "admitted",
    )


@dataclass(frozen=True)
class SineCycleIdentityAssessment:
    """Conservative winding protection and a separate recurrent family.

    Geometry is shared with dissipative recovery, but the admitted law here
    has explicitly zero EPI weight. First-exit trapping preserves the local
    radius and cycle winding without asserting convergence to the target.

    The caller chooses an analysis family with centered radius less than r,
    excess storage less than h and weighted form mean in an open interval.
    If 0<h<barrier, this is a positive finite-volume invariant family, and
    recurrence holds almost everywhere. Source-set trapping and relative
    family membership are separate checks. Relative observations discard an
    arbitrary common form origin, so no finite absolute mean slab contains
    their entire uncertainty set. Choosing a representative does not measure
    that missing mean or certify recurrence of a chosen moving state.
    """

    source: SineRelativePattern | SineRelativeForecast
    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    cycle: tuple[Any, ...]
    cycle_indices: tuple[int, ...]
    winding: int
    phase_turns: tuple[int, ...]
    target_phase_turns: tuple[Q, ...]
    radius: Q
    uncertainty_scope: str
    observation_time: Q | None
    input_forecast_admitted: bool | None
    input_forecast_requested_end_time: Q | None
    capacity_bounds: tuple[I, ...]
    exact_held_capacity: tuple[Q | None, ...]
    spectral_gap_bounds: I
    spectral_gap_lower_bound: Q
    target_edge_angle_bounds: I
    target_phase_storage_bounds: I
    radius_angle_bounds: I
    acute_radius_margin_bounds: I
    cosine_lower_bound: Q | None
    coercivity_lower_bound: Q | None
    form_norm_squared_upper_bound: Q
    phase_norm_squared_upper_bound: Q
    norm_squared_upper_bound: Q
    form_edge_gap_bounds: tuple[I, ...]
    phase_deviation_edge_bounds: tuple[I, ...]
    phase_hessian_upper_bounds: tuple[Q, ...]
    form_energy_upper_bound: Q
    phase_excess_upper_bound: Q
    excess_storage_upper_bound: Q
    barrier_lower_bound: Q | None
    norm_margin: Q
    energy_margin: Q | None
    excess_ceiling: Q
    form_mean_bounds: tuple[Q, Q]
    family_barrier_margin: Q | None
    source_family_excess_margin: Q
    hypothesis_failures: tuple[str, ...]
    family_unresolved_conditions: tuple[str, ...]
    source_trapping_unresolved_conditions: tuple[str, ...]
    family_admitted: bool
    source_set_trapping_certified: bool
    relative_source_family_membership: str
    family_almost_everywhere_recurrence_certified: bool
    absolute_source_family_membership: str = "unavailable_common_form_origin_unobserved"
    individual_recurrence_status: str = "unavailable_for_chosen_state"
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "shared_full_Cn_exact_acute_twist_geometry_and_original_observation_provenance",
        "explicit_zero_epi_weight_and_strictly_positive_held_capacities",
        "no_dissipative_recovery_certificate_or_fabricated_positive_loss_model",
        "same_pairwise_norm_and_cancelled_linear_phase_energy_bounds_as_recovery",
        "first_exit_barrier_preserves_relative_radius_and_declared_integer_winding",
        "source_set_trapping_does_not_require_membership_in_the_smaller_declared_family",
        "excess_ceiling_and_mean_slab_choose_an_analysis_family_not_new_constitutive_parameters",
        "family_is_open_centered_radius_excess_sublevel_and_weighted_mean_slab",
        "positive_finite_ambient_volume_in_real_form_and_circular_phase",
        "family_recurrence_is_almost_everywhere_not_all_states_or_exact_periodicity",
        "probability_one_requires_absolutely_continuous_preparation_in_that_family",
        "source_common_form_origin_is_unobserved_no_absolute_mean_membership_inferred",
        "relative_family_membership_checks_only_radius_and_excess_not_absolute_mean",
        "failed_upper_bounds_mean_unresolved_membership_not_proved_exclusion",
        "forecast_capacity_families_are_separate_fixed_realizations_not_one_mixed_flow",
        "weighted_mean_uses_degree_over_capacity_for_each_fixed_realization",
        "actual_forecast_validated_time_retained_even_if_requested_horizon_unavailable",
        "no_convergence_selected_moving_state_recurrence_or_period_certificate",
        "no_observation_authentication_trajectory_support_event_or_pattern_birth",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        pattern = (
            self.source.pattern
            if isinstance(self.source, SineRelativeForecast)
            else self.source
        )
        _validate_comparison_labels(pattern)
        return {
            "schema": "tnfr.relational-sine-cycle-identity.v1",
            "report": _project(self),
        }


def assess_sine_cycle_identity(
    source,
    *,
    cycle,
    winding,
    radius,
    excess_ceiling,
    form_mean_bounds,
    phase_turns=None,
) -> SineCycleIdentityAssessment:
    """Separate conservative trapping from an almost-everywhere family theorem.

    The complete cycle, target, explicit phase-lift turns and uncertain source
    reuse the existing local geometry owner. A positive excess ceiling and
    positive-width open mean interval select the family to study. Geometry
    and law failures abstain; they do not assert instability. Family validity
    is independent of whether the source-set bounds prove trapping or the
    stricter relative family conditions. Its arbitrary common form offset
    prevents an absolute mean-membership claim for these source interfaces.
    """
    if not isinstance(source, (SineRelativePattern, SineRelativeForecast)):
        raise TypeError("a SineRelativePattern or SineRelativeForecast is required")
    ceiling = exact_or_represented_real(excess_ceiling, "excess_ceiling")
    if ceiling <= 0:
        raise ValueError("excess_ceiling must be strictly positive")
    raw_mean = _ordered(form_mean_bounds, "form_mean_bounds", limit=3)
    if len(raw_mean) != 2:
        raise ValueError("form_mean_bounds must contain exactly two ordered endpoints")
    lower, upper = (
        exact_or_represented_real(value, "form_mean_bounds") for value in raw_mean
    )
    if lower >= upper:
        raise ValueError("form_mean_bounds must have strictly positive width")
    common, target_fields, capacity_failures, geometry_unresolved, source_unresolved = (
        _local_sine_geometry(
            source,
            cycle=cycle,
            winding=winding,
            radius=radius,
            phase_turns=phase_turns,
        )
    )
    failures = (
        ()
        if common["reference_model"].effective_weights[0] == 0
        else ("zero_epi_weight_required",)
    ) + capacity_failures
    barrier = common["barrier_lower_bound"]
    family_margin = None if barrier is None else barrier - ceiling
    family_unresolved = geometry_unresolved + (
        ()
        if family_margin is not None and family_margin > 0
        else ("strict_family_excess_barrier_not_certified",)
    )
    trapping_unresolved = geometry_unresolved + source_unresolved
    relative_margin = ceiling - common["excess_storage_upper_bound"]
    relative_inside = common["norm_margin"] > 0 and relative_margin > 0
    family_admitted = not failures and not family_unresolved
    return SineCycleIdentityAssessment(
        **common,
        **target_fields,
        excess_ceiling=ceiling,
        form_mean_bounds=(lower, upper),
        family_barrier_margin=family_margin,
        source_family_excess_margin=relative_margin,
        hypothesis_failures=failures,
        family_unresolved_conditions=family_unresolved,
        source_trapping_unresolved_conditions=trapping_unresolved,
        family_admitted=family_admitted,
        source_set_trapping_certified=not failures and not trapping_unresolved,
        relative_source_family_membership=(
            "certified_inside" if relative_inside else "unresolved"
        ),
        family_almost_everywhere_recurrence_certified=family_admitted,
    )
