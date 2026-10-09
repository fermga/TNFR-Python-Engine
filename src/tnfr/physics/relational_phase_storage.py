"""Static constitutive comparison of locally identical phase-storage laws.

The supplied smooth cutoff law agrees with sine on a complete open chamber
but raises the antipodal edge cost. This owner certifies a local-flow equality
and a global winding obstruction; it installs no runtime or trajectory law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval
from .relational_sine_comparison import (
    SineExchangeComparison,
    _sine_phase_storage,
    _validate_comparison_labels,
)
from .relational_sine_corridor import SineSaddlePreparation, prepare_sine_saddle_state
from .relational_sine_partition import _private_leaf_support
from .relational_sine_regional import _branches

__all__ = ("SaddleStorageDiscriminator", "assess_saddle_storage_discriminator")


@dataclass(frozen=True)
class SaddleStorageDiscriminator:
    """A supplied alternative on the same freshly constructed initial state.

    The full independent coordinate cube inherits local equality of the two
    complete flows and, when certified, the alternative winding obstruction.
    Signed-family sine corridor gates belong only to the nominal preparation.
    The discriminator proves a global difference for that same nominal state;
    it supplies no new sine passage certificate for every point of the cube.
    """

    source: SineExchangeComparison
    preparation: SineSaddlePreparation
    prepared_state: SineExchangeComparison
    cycle: tuple[object, ...]
    cycle_indices: tuple[int, ...]
    edge_indices: tuple[tuple[int, int], ...]
    epsilon: Q
    initial_coordinate_radius: Q
    initial_box: tuple[I, ...]
    cutoff_phase_bounds: I
    initial_edge_principal_gap_bounds: tuple[I, ...] | None
    initial_agreement_margin_lower_bound: Q | None
    initial_cycle_principal_gap_bounds: tuple[I, ...] | None
    initial_edge_turn_offsets: tuple[int, ...] | None
    initial_cycle_turn_offsets: tuple[int, ...] | None
    initial_winding: int | None
    initial_form_storage_bounds: I
    initial_sine_phase_storage_bounds: I
    sine_initial_storage_bounds: I
    alternative_initial_storage_bounds: I | None
    antipodal_storage: Q
    energy_margin_lower_bound: Q | None
    local_scaled_duration: Q
    local_phase_radius_bound: Q
    local_agreement_margin_lower_bound: Q
    initial_agreement_certified: bool
    whole_local_flow_agreement_certified: bool
    nominal_sine_outer_gate_certified: bool
    nominal_sine_inner_gate_certified: bool
    alternative_winding_invariant_both_time_directions_certified: bool
    alternative_zero_winding_passage_excluded: bool
    discriminator_certified: bool
    status: str
    reasons: tuple[str, ...]
    sine_full_cube_passage_certified: bool = False
    law: str = "smooth_cutoff_phase_storage_reciprocal_exchange"
    regularity: str = "C_infinity_but_not_real_analytic_at_the_cutoff"
    clock: str = "original structural t; local proof tau=t/pi"
    potential_formula: str = (
        "U(delta)=1-cos(delta)+2*h(y); "
        "y=(-cos(delta)-c)/(1-c); c=sqrt(2)/2; "
        "h(y)=0 for y<=0 and exp(1-1/y) for y>0"
    )
    current_formula: str = (
        "j(delta)=sin(delta) for y<=0; "
        "j(delta)=sin(delta)*(1+2*h(y)/((1-c)*y^2)) for y>0"
    )
    complete_rows: tuple[str, ...] = (
        "dx_i/dt=sum_j j(theta_j-theta_i)/(pi*d_i)",
        "dtheta_i/dt=sum_j(x_i-x_j)/(pi*d_i)",
        "fixed complete support; held unit capacities; zero inputs and no events",
    )
    storage_identity: str = (
        "H=sum_edges((x_j-x_i)^2/2+U(theta_j-theta_i)); "
        "dH/dtau=-(Lx)^T*K*grad(V)+grad(V)^T*K*Lx=0"
    )
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "source_anchors_reference_support_and_law_not_the_generated_preparation",
        "prepared_state_retains_a_sine_reference_declaration_not_an_installed_alternative",
        "smooth_even_periodic_nonnegative_potential_is_supplied_not_uniquely_derived",
        "global_first_resultant_sufficiency_and_native_pressure_alignment_do_not_hold",
        "cutoff_and_antipodal_cost_are_counterexample_choices_not_physical_constants",
        "all_full_state_coordinates_contacts_and_form_edge_storage_are_retained",
        "initial_box_outwardly_encloses_the_declared_exact_independent_coordinate_cube",
        "alternative_initial_storage_is_evaluated_only_inside_the_exact_zero_bump_chamber",
        "whole_local_flow_equality_uses_a_uniform_bound_and_smooth_uniqueness_in_both_time_directions",
        "independent_cube_perturbations_do_not_inherit_nominal_signed_family_corridor_gates",
        "discriminator_certified_compares_nominal_global_passage_not_full_cube_sine_passage",
        "strict_full_storage_below_antipodal_cost_prevents_any_cycle_winding_change",
        "winding_invariance_is_not_acute_identity_retention_or_attraction",
        "no_alternative_trajectory_solver_law_installation_or_physical_identification",
        "no_retained_response_is_authenticated_recomputed_or_rewritten",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label_groups

        _validate_comparison_labels(self.source)
        _validate_comparison_labels(self.prepared_state)
        _validate_label_groups(self.cycle)
        self.preparation.to_dict()
        return {
            "schema": "tnfr.saddle-storage-discriminator.v1",
            "report": _project(self),
        }


def assess_saddle_storage_discriminator(
    source, *, cycle, initial_coordinate_radius, epsilon=Q(1, 2**32)
) -> SaddleStorageDiscriminator:
    """Compare a rebuilt sine preparation with one explicit smooth alternative.

    No prepared state or cached verdict is accepted from the caller. The
    source only anchors the existing conservative unit C5/private-leaf law;
    the shared preparation owner constructs the actual rational center.
    Radius is a nonnegative exact or represented real error at every form and
    phase coordinate. Capacity, support and clock stay fixed.

    The alternative has zero added storage/current for principal edge gaps
    at most3*pi/4. Its complete storage is therefore exactly the rebuilt sine
    storage on that chamber. Outside it this static reader leaves alternative
    storage unavailable instead of evaluating an unimplemented runtime.
    The exact antipodal value U(pi)=4 supplies the global boundary proof.
    """
    delta = exact_or_represented_real(
        initial_coordinate_radius, "initial_coordinate_radius"
    )
    if delta < 0:
        raise ValueError("initial_coordinate_radius must be nonnegative")
    source, edges, indices, pairs, _ = _private_leaf_support(source, cycle)
    labels = tuple(source.nodes[index] for index in indices)
    preparation = prepare_sine_saddle_state(source, cycle=labels, epsilon=epsilon)
    prepared = preparation.prepared_state
    initial_box = tuple(
        I(value - delta, value + delta) for value in prepared.epi + prepared.phase
    )
    count = len(prepared.nodes)
    forms, phases = initial_box[:count], initial_box[count:]
    pi = pi_interval()
    cutoff = 3 * pi / 4
    edge_offsets, principal, _, _ = _branches(
        tuple(phases[j] - phases[i] for i, j in edges)
    )
    agreement_margin = (
        None if principal is None else min(cutoff.lo - gap.abs_max for gap in principal)
    )
    initial_agreement = agreement_margin is not None and agreement_margin >= 0
    offsets, cycle_gaps, winding, _ = _branches(
        tuple(phases[j] - phases[i] for i, j in pairs)
    )
    form_storage = sum(((forms[j] - forms[i]) ** 2 / 2 for i, j in edges), I(0))
    phase_storage = _sine_phase_storage(phases, edges)
    sine_storage = form_storage + phase_storage
    alternative_storage = sine_storage if initial_agreement else None
    energy_margin = (
        None if alternative_storage is None else Q(4) - alternative_storage.hi
    )
    local_radius = 729 * (
        3 * preparation.epsilon + preparation.preparation_error_bound + delta
    )
    # Saddle gaps are bounded by2pi/3. Two nodal errors affect an edge, so
    # the extra gap allowance before the3pi/4 cutoff is only pi/12.
    local_margin = pi.lo / 12 - 2 * local_radius
    local_agreement = (
        preparation.formation.correlated_preparation_certified
        and initial_agreement
        and local_margin > 0
    )
    invariant = (
        initial_agreement
        and winding is not None
        and energy_margin is not None
        and energy_margin > 0
    )
    excludes_zero = invariant and winding != 0
    outer = preparation.outer_corridor_gate_certified
    inner = preparation.inner_corridor_gate_certified
    certified = local_agreement and outer and inner and excludes_zero
    reasons = tuple(
        reason
        for passed, reason in (
            (
                initial_agreement,
                "initial_full_cube_not_certified_inside_agreement_chamber",
            ),
            (local_agreement, "whole_local_flow_agreement_not_certified"),
            (winding is not None, "initial_cycle_branch_not_certified"),
            (
                energy_margin is not None and energy_margin > 0,
                "alternative_storage_below_seam_not_certified",
            ),
            (outer and inner, "nominal_sine_corridor_gates_not_certified"),
            (excludes_zero, "alternative_zero_winding_exclusion_not_certified"),
        )
        if not passed
    )
    return SaddleStorageDiscriminator(
        source=source,
        preparation=preparation,
        prepared_state=prepared,
        cycle=labels,
        cycle_indices=indices,
        edge_indices=edges,
        epsilon=preparation.epsilon,
        initial_coordinate_radius=delta,
        initial_box=initial_box,
        cutoff_phase_bounds=cutoff,
        initial_edge_principal_gap_bounds=principal,
        initial_agreement_margin_lower_bound=agreement_margin,
        initial_cycle_principal_gap_bounds=cycle_gaps,
        initial_edge_turn_offsets=edge_offsets,
        initial_cycle_turn_offsets=offsets,
        initial_winding=winding,
        initial_form_storage_bounds=form_storage,
        initial_sine_phase_storage_bounds=phase_storage,
        sine_initial_storage_bounds=sine_storage,
        alternative_initial_storage_bounds=alternative_storage,
        antipodal_storage=Q(4),
        energy_margin_lower_bound=energy_margin,
        local_scaled_duration=Q(3),
        local_phase_radius_bound=local_radius,
        local_agreement_margin_lower_bound=local_margin,
        initial_agreement_certified=initial_agreement,
        whole_local_flow_agreement_certified=local_agreement,
        nominal_sine_outer_gate_certified=outer,
        nominal_sine_inner_gate_certified=inner,
        alternative_winding_invariant_both_time_directions_certified=invariant,
        alternative_zero_winding_passage_excluded=excludes_zero,
        discriminator_certified=certified,
        status="certified" if certified else "unavailable",
        reasons=reasons,
    )
