"""Static enclosures for an explicitly distinct smooth reciprocal comparison.

The reference model supplies only the held coefficients and graph-admission
contract. Replacing Arg pressure by normalized sine current changes both
evolution rows. No runtime dispatch, boundary continuation or law selection
is supplied here, and no native regular-domain field is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction as Q
from typing import Any

from .._exact_time import exact_or_represented_real
from ..constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.relational import RelationalExchangeModel, _stage
from ..mathematics._phase_resultant_chamber import (
    certified_sine_bounds,
    relative_resultant_bounds,
    relative_resultant_rate_bounds,
)
from ..mathematics._rational_interval import INTERVAL_METHOD, I, cos, pi_interval, sin

__all__ = (
    "SineExchangeComparison",
    "SineResultantKinematics",
    "SineMobilityComparison",
    "SineMobilityRelativeBalance",
    "SineRegionalTransfer",
    "SineRegionalStorageBalance",
    "SineRegionalChannelBalance",
    "SineFormIncrementAssessment",
    "bound_relational_sine_exchange",
)


@dataclass(frozen=True)
class SineExchangeComparison:
    """Ideal instantaneous bounds at exact captured represented coordinates.

    ``inverse_phase_metric`` encloses 1/(pi*d_i), without capacity. Work
    residuals are actually computed interval sums; their enclosure of zero
    checks the algebra but is not the proof of its all-state identity.
    ``reference_model`` does not label the returned field as the native law.
    """

    reference_model: RelationalExchangeModel
    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    relative_resultant: tuple[tuple[I, I], ...]
    inverse_phase_metric: tuple[I, ...]
    phase_sources: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    form_storage: Q
    phase_storage: I
    storage: I
    dissipation: tuple[Q, ...]
    continuous_loss: Q
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    node_balance_residual: tuple[I, ...]
    storage_rate: I
    balance_residual: I
    law: str = "normalized_sine_reciprocal_exchange"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "conditional_comparison_with_normalized_pairwise_pressure_superposition_premise",
        "fixed_simple_connected_unit_support_held_nonnegative_capacity_no_input_or_events",
        "same_reference_coefficients_storage_and_nodal_product_but_changed_pressure_and_phase_law",
        "ideal_globally_smooth_continuous_law_not_a_native_boundary_continuation",
        "instantaneous_enclosures_at_exact_captured_represented_coordinates",
        "computed_work_residual_enclosures_are_not_forced_to_zero",
        "no_graph_write_trajectory_solver_or_finite_step_stability_certificate",
        "no_frozen_response_reinterpretation_constitutive_selection_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self)
        return {
            "schema": "tnfr.relational-sine-comparison.v1",
            "report": _project(self),
        }

    def phase_rate_numerators(self) -> tuple[Q, ...]:
        """Return the exact captured phase row before its shared pi divisor.

        The complete comparison law gives theta'_i=N_i/pi. This detached
        rational calculation re-admits primitive state and rebuilds the form
        gradient. It needs neither trigonometric bounds nor a graph reread;
        it does not reconstruct rates from observed responses.
        """
        source, neighbors = _admit_sine_comparison_rows(self)
        return _sine_phase_rate_numerators(
            source.reference_model,
            source.degrees,
            _sine_form_gradient(source.epi, neighbors),
            source.capacity,
        )

    def resultant_kinematics(self):
        """Enclose node-relative neighbor-resultant drift from this capture.

        This is not the normalized phasor of a supplied pair or region.
        The exact comparison rates share one mathematical-pi denominator:
        theta'_i=a_i/pi with a_i=w*nu_i*q_i/(beta*d_i). Applying the existing
        rational-direction chain rule to a and dividing afterwards retains
        this common factor without subtracting independently rounded rates.
        Primitive state and consumed derived fields are re-admitted and rebuilt.
        No graph, native field or temporal sample is read.
        """
        source = _rebuild_sine_comparison(self)
        neighbors = _comparison_neighbors(source)
        numerators = _sine_phase_rate_numerators(
            source.reference_model,
            source.degrees,
            source.form_gradient,
            source.capacity,
        )
        raw_bounds = relative_resultant_rate_bounds(source.phase, neighbors, numerators)
        pi = pi_interval()
        drifts = tuple((I(*real) / pi, I(*imag) / pi) for real, imag in raw_bounds)
        speeds = tuple(
            I(sum((abs(numerators[j] - numerators[i]) for j in row), Q(0))) / pi
            for i, row in enumerate(neighbors)
        )
        association = self if source == self else source
        return SineResultantKinematics(association, numerators, drifts, speeds)

    def asymptotic_equilibria(self, *, cycles):
        """Admit the two-C5 finite critical set and the separate convergence law."""
        from .relational_sine_equilibria import assess_sine_asymptotic_equilibria

        return assess_sine_asymptotic_equilibria(self, cycles=cycles)

    def assess_cycle_symmetry(self, *, permutation_indices, cycle):
        """Check full-state symmetry and reversal of a supplied oriented cycle."""
        from .relational_sine_symmetry import assess_sine_cycle_symmetry

        return assess_sine_cycle_symmetry(
            self, permutation_indices=permutation_indices, cycle=cycle
        )

    def mediated_pressure(self, *, mediator):
        """Derive causal environmental pressure from this unchanged capture.

        Retain the supplied mediator's actual form and phase, including zero
        capacity or a cancelled resultant. The returned instantaneous state,
        port decomposition and Volterra coefficients do not evaluate a memory
        trajectory, impose a stationary replacement or reread a graph.
        """
        from .relational_sine_mediation import _mediated_pressure

        return _mediated_pressure(self, mediator=mediator)

    def regional_transfer(self, *, region):
        """Observe the weighted form current entering one supplied region.

        The fixed weights d_i/nu_i require positive held capacity at every
        node. Region labels must be distinct, known and explicitly ordered;
        the whole graph is allowed, an empty region is not. Boundary currents
        retain both diffusion and reciprocal sine exchange. They belong to
        this complete law, not the native Arg or variable-mobility models.
        No regional midpoint angle or nonzero resultant is required. A zero
        instantaneous cut current does not certify an inactive phase channel
        or silence over a time interval.
        Primitive state and all nested comparison fields are rebuilt before
        retaining an equivalent original report association.
        No source reread, regional closure or trajectory is supplied.
        """
        from .relational_observations import _ordered, _regions

        source, weights = _admit_sine_form_accounting(self)
        region = _ordered(region, "region", limit=len(source.nodes) + 1)
        indices = _regions(source.nodes, (region,))[0]
        source = _rebuild_sine_comparison(source)
        members = set(indices)
        complement = tuple(i for i in range(len(source.nodes)) if i not in members)
        positions = {node: i for i, node in enumerate(source.nodes)}
        boundary = []
        for left, right in source.edges:
            i, j = positions[left], positions[right]
            if (i in members) != (j in members):
                boundary.append((i, j) if i in members else (j, i))
        boundary = tuple(boundary)
        rates = source.form_rates
        e, w = map(Q, source.reference_model.effective_weights)
        pi = pi_interval()
        diffusion = tuple(e * (source.epi[j] - source.epi[i]) for i, j in boundary)
        sine = tuple(
            (w / pi) * I(*certified_sine_bounds(source.phase[j] - source.phase[i]))
            for i, j in boundary
        )
        currents = tuple(I(d) + s for d, s in zip(diffusion, sine))
        regional_rate = sum(currents, I(0))
        direct_region = sum((weights[i] * rates[i] for i in indices), I(0))
        direct_complement = sum((weights[i] * rates[i] for i in complement), I(0))
        return SineRegionalTransfer(
            comparison=self if source == self else source,
            region=region,
            region_indices=indices,
            complement_indices=complement,
            boundary_edge_indices=boundary,
            form_weights=weights,
            regional_weighted_form=sum(
                (weights[i] * source.epi[i] for i in indices), Q(0)
            ),
            complement_weighted_form=sum(
                (weights[i] * source.epi[i] for i in complement), Q(0)
            ),
            boundary_diffusive_currents=diffusion,
            boundary_sine_currents=sine,
            boundary_form_currents=currents,
            regional_form_rate_bounds=regional_rate,
            complement_form_rate_bounds=-regional_rate,
            direct_regional_form_rate_bounds=direct_region,
            direct_complement_form_rate_bounds=direct_complement,
            regional_balance_residual_bounds=direct_region - regional_rate,
            complement_balance_residual_bounds=direct_complement + regional_rate,
            global_weighted_form_rate_bounds=direct_region + direct_complement,
        )

    def regional_storage_balance(self, *, region):
        """Partition storage and its full-law boundary work at this state.

        Regional and complementary storage contain only their internal edges;
        cross-boundary edges form a third, disjoint storage compartment.
        All rates retain full-network degrees and the original structural t
        clock. Zero held capacity is admitted. This energy ledger is distinct
        from ``regional_transfer``'s degree/capacity weighted form flux.

        Primitive state and law are re-admitted and every consumed field is
        rebuilt. The report is instantaneous; it neither authenticates its
        source nor integrates boundary work or selects a regional closure.
        """
        return _regional_storage_balance(self, region=region)

    def assess_form_increment(self, *, increments):
        """Check one complete supplied form endpoint against its invariant.

        ``increments`` must give one signed value per captured node, in that
        order. Rational inputs stay exact; other real inputs use the shared
        represented-real admission. These are declared endpoint differences,
        not derivative data, a runtime event or a proposed time step. A nonzero
        weighted change excludes the endpoint under the same closed complete
        law. A zero change alone certifies neither reachability nor timing.
        The nested comparison is rebuilt from admitted primitives; its rates
        are not used as the supplied increment or as evidence of reachability.
        """
        from .relational_observations import _ordered

        source, weights = _admit_sine_form_accounting(self)
        raw = _ordered(increments, "increments", limit=len(source.nodes) + 1)
        if len(raw) != len(source.nodes):
            raise ValueError("increments must contain one value per captured node")
        values = tuple(
            exact_or_represented_real(value, f"increments[{i}]")
            for i, value in enumerate(raw)
        )
        source = _rebuild_sine_comparison(source)
        before = sum((weight * x for weight, x in zip(weights, source.epi)), Q(0))
        change = sum((weight * dx for weight, dx in zip(weights, values)), Q(0))
        return SineFormIncrementAssessment(
            comparison=self if source == self else source,
            increments=values,
            form_weights=weights,
            weighted_form_before=before,
            weighted_form_after=before + change,
            weighted_form_change=change,
            closed_flow_endpoint_obstructed=change != 0,
        )

    def with_current_squared_mobility(self, *, epsilon):
        """Evaluate a separately declared reciprocal counterfamily at this source.

        The finite nonnegative epsilon gives f_i=1+epsilon*(S_i/d_i)**2.
        Both exchange rows use f_i/(pi*d_i); no new capture, native dispatch
        or installation of a law occurs. Only explicit zero form loss is
        admitted here. At epsilon zero the original full rates are recovered.
        Re-admit the primitive source and rebuild its currents and gradients;
        cached report rates and verdicts are not inputs to this calculation.
        """
        source = _rebuild_sine_comparison(self)
        result = _sine_current_squared_mobility(source, epsilon)
        return replace(result, comparison=self) if source == self else result


@dataclass(frozen=True)
class SineRegionalTransfer:
    """Ideal boundary ledger of one captured region, with inward currents.

    Each ``boundary_edge_indices`` entry is (inside, outside), in captured
    edge order. Its positive current increases the region's degree/capacity
    weighted form total. The complement receives the opposite current.
    Interior cancellation is algebraic; the separately summed fine rates
    retain their computed interval residuals. No regional state closure or
    derivative of a separately assigned regional energy is asserted.
    """

    comparison: SineExchangeComparison
    region: tuple[Any, ...]
    region_indices: tuple[int, ...]
    complement_indices: tuple[int, ...]
    boundary_edge_indices: tuple[tuple[int, int], ...]
    form_weights: tuple[Q, ...]
    regional_weighted_form: Q
    complement_weighted_form: Q
    boundary_diffusive_currents: tuple[Q, ...]
    boundary_sine_currents: tuple[I, ...]
    boundary_form_currents: tuple[I, ...]
    regional_form_rate_bounds: I
    complement_form_rate_bounds: I
    direct_regional_form_rate_bounds: I
    direct_complement_form_rate_bounds: I
    regional_balance_residual_bounds: I
    complement_balance_residual_bounds: I
    global_weighted_form_rate_bounds: I
    global_weighted_form_conserved: bool = True
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_normalized_sine_law_including_its_form_loss_channel",
        "captured_fixed_unit_support_and_strictly_positive_held_capacity",
        "degree_over_capacity_weights_use_original_full_support_degrees",
        "inward_boundary_diffusion_and_sine_currents_cancel_on_internal_edges",
        "algebraic_global_form_invariant_is_distinct_from_interval_rate_residual",
        "complement_compensates_regional_change_without_external_form_injection",
        "positive_regional_rate_is_not_an_AL_event_or_autonomous_event_trigger",
        "no_regional_energy_or_closed_coarse_state_is_inferred",
        "no_transfer_to_native_Arg_or_state_dependent_mobility",
        "no_graph_reread_evolution_finite_step_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for node in self.region:
            _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-regional-transfer.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineRegionalChannelBalance:
    """Internal conversion and boundary inputs to the two regional stores.

    Entries in the gradient, current and nodal-rate rows follow region_indices.
    internal_conversion is positive for phase-to-form internal transfer.
    boundary_form_input and boundary_phase_input feed the separate internal
    stores; they are not the old conjugate boundary-work decomposition.
    signed_form_loss=e*q_R.T*M*q uses the full q and may be negative, unlike
    the nonnegative full-node loss. Phase storage rates include beta.
    """

    region_indices: tuple[int, ...]
    internal_form_gradient: tuple[I, ...]
    internal_sine_current: tuple[I, ...]
    external_form_contrast: tuple[I, ...]
    external_sine_current: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    internal_conversion: I
    boundary_form_input: I
    boundary_phase_input: I
    signed_form_loss: I
    form_storage_rate: I
    weighted_phase_storage_rate: I
    total_storage_rate: I
    form_balance_residual: I
    phase_balance_residual: I
    total_balance_residual: I
    flat_phase_storage_acceleration: I | None
    clock: str = "structural_t"
    scope: tuple[str, ...] = (
        "same_complete_sine_law_full_support_degrees_and_held_capacity",
        "regional_internal_form_and_beta_weighted_phase_stores",
        "internal_conversion_and_boundary_inputs_keep_distinct_causal_channels",
        "signed_regional_form_loss_is_not_nonnegative_full_node_loss",
        "channel_boundary_inputs_are_not_conjugate_boundary_work_components",
        "direct_nodal_storage_derivatives_and_reconstructed_balance_residuals",
        "optional_exact_internal_flat_phase_curvature_is_instantaneous_not_formation",
        "no_trajectory_work_integral_monotonicity_or_formation_verdict",
    )


@dataclass(frozen=True)
class SineRegionalStorageBalance:
    """Three disjoint edge stores and inward work under the complete sine law.

    Boundary edges are oriented (region, complement). Phase storage fields
    contain the unit cosine potential; total stores multiply it by beta.
    Positive boundary work increases the named internal store before its
    full-node loss is subtracted. The cross-edge derivative compensates both
    inward powers, which need not be opposites of one another. All derivatives
    use the original structural t clock, including its shared pi divisor.
    """

    comparison: SineExchangeComparison
    region: tuple[Any, ...]
    region_indices: tuple[int, ...]
    complement_indices: tuple[int, ...]
    regional_edge_indices: tuple[tuple[int, int], ...]
    complement_edge_indices: tuple[tuple[int, int], ...]
    boundary_edge_indices: tuple[tuple[int, int], ...]
    regional_form_storage: Q
    regional_phase_storage: I
    regional_storage: I
    complement_form_storage: Q
    complement_phase_storage: I
    complement_storage: I
    boundary_form_storage: Q
    boundary_phase_storage: I
    boundary_storage: I
    regional_boundary_form_work: I
    regional_boundary_phase_work: I
    regional_boundary_work: I
    complement_boundary_form_work: I
    complement_boundary_phase_work: I
    complement_boundary_work: I
    regional_loss: Q
    complement_loss: Q
    direct_regional_storage_rate: I
    direct_complement_storage_rate: I
    direct_boundary_storage_rate: I
    regional_balance_residual: I
    complement_balance_residual: I
    boundary_balance_residual: I
    storage_partition_residual: I
    global_balance_residual: I
    regional_channels: SineRegionalChannelBalance | None = None
    complement_channels: SineRegionalChannelBalance | None = None
    clock: str = "structural_t"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "complete_normalized_sine_law_original_structural_t_clock",
        "fixed_simple_connected_unit_support_held_nonnegative_capacity_no_input_or_events",
        "primitive_source_readmitted_and_all_consumed_rates_and_stores_rebuilt",
        "internal_region_internal_complement_and_cross_edges_partition_storage_once",
        "all_nodal_rates_gradients_degrees_and_losses_use_full_support",
        "regional_boundary_work_retains_form_and_phase_channels",
        "optional_channel_ledgers_separate_internal_conversion_from_boundary_inputs",
        "inward_regional_powers_are_compensated_by_cross_edge_storage_change",
        "direct_storage_derivatives_and_balance_residuals_are_independently_summed",
        "weighted_form_transfer_is_a_distinct_observation_not_this_energy_ledger",
        "zero_capacity_admitted_without_division_by_capacity",
        "no_work_history_monotonic_regional_storage_closure_or_trajectory_claim",
        "no_native_law_transfer_source_authentication_event_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        for node in self.region:
            _validate_label(node)
        return {
            "schema": "tnfr.relational-sine-regional-storage-balance.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class SineFormIncrementAssessment:
    """Exact full-form invariant change for a supplied endpoint difference.

    A nonzero change rules out the entire stated form endpoint under this
    closed law, independently of any unspecified final phase. Zero change
    passes only this necessary condition. Rational increments can be ideal
    exact inputs rather than binary64 runtime proposals; their origin must be
    supplied separately. The AL adapter uses its actual represented changes.
    This concerns full form in the same chart; a quotient discarding the
    common form origin requires its own endpoint-realizability question.
    """

    comparison: SineExchangeComparison
    increments: tuple[Q, ...]
    form_weights: tuple[Q, ...]
    weighted_form_before: Q
    weighted_form_after: Q
    weighted_form_change: Q
    closed_flow_endpoint_obstructed: bool
    endpoint_reachability_certified: bool = False
    arithmetic_method: str = "exact_rational_increments_and_captured_represented_state"
    scope: tuple[str, ...] = (
        "same_complete_normalized_sine_law_fixed_support_and_positive_held_capacity",
        "full_node_increment_vector_in_captured_order_not_a_partial_regional_endpoint",
        "exact_rational_inputs_preserved_other_real_inputs_use_shared_binary64_admission",
        "nonzero_degree_over_capacity_weighted_change_excludes_closed_flow_endpoint",
        "zero_weighted_change_does_not_certify_reachability_return_time_or_phase_endpoint",
        "form_invariant_is_distinct_from_storage_passivity_and_event_admission",
        "declared_increments_do_not_authenticate_an_executed_event_or_measured_response",
        "no_forcing_capacity_support_mobility_or_clock_change_is_supplied",
        "no_graph_reread_runtime_commit_solver_or_physical_identification",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-form-increment.v1",
            "report": _project(self),
        }


def _admit_sine_form_accounting(comparison):
    """Admit primitive source data and its positive-capacity accounting weights."""
    from ._sine_admission import _admit_sine_source

    if not isinstance(comparison, SineExchangeComparison):
        raise TypeError("a SineExchangeComparison source is required")
    comparison = _admit_sine_source(comparison)[0]
    if any(value <= 0 for value in comparison.capacity):
        raise ValueError(
            "weighted form balance requires strictly positive held capacity"
        )
    weights = tuple(
        Q(degree) / nu for degree, nu in zip(comparison.degrees, comparison.capacity)
    )
    return comparison, weights


def _sine_form_weights(comparison):
    """Share the actual full-degree/held-capacity form accounting weights."""
    return _admit_sine_form_accounting(comparison)[1]


@dataclass(frozen=True)
class SineMobilityComparison:
    """Complete instantaneous reciprocal counterfamily, retaining its source.

    ``inverse_phase_metric`` now encloses the declared mobility f_i/(pi*d_i).
    The storage and clock are those of ``comparison``. Source storage balance
    carries through the two matched rows; Hamiltonian, volume, recurrence,
    pulse and finite-window results are not transferred. Computed work
    residuals remain interval checks, not independent all-state proofs.
    """

    comparison: SineExchangeComparison
    epsilon: Q
    mobility_factors: tuple[I, ...]
    mobility_corrections: tuple[I, ...]
    inverse_phase_metric: tuple[I, ...]
    phase_sources: tuple[I, ...]
    pressure: tuple[I, ...]
    form_rates: tuple[I, ...]
    phase_rates: tuple[I, ...]
    dissipation: tuple[Q, ...]
    continuous_loss: Q
    form_work: tuple[I, ...]
    phase_work: tuple[I, ...]
    node_balance_residual: tuple[I, ...]
    storage_rate: I
    balance_residual: I
    law: str = "current_squared_reciprocal_mobility"
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_captured_fixed_unit_support_held_capacity_storage_and_structural_clock",
        "explicit_zero_form_loss_finite_nonnegative_epsilon_and_phase_only_mobility",
        "both_exchange_rows_share_f_i_over_pi_degree_not_a_one_row_correction",
        "epsilon_zero_recovers_the_normalized_sine_complete_field",
        "storage_cancellation_does_not_select_mobility_or_preserve_constant_Poisson_form",
        "no_transfer_of_volume_recurrence_pulse_or_finite_window_certificates",
        "no_graph_reread_solver_fitted_epsilon_live_law_or_native_dispatch",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-mobility-comparison.v1",
            "report": _project(self),
        }

    def relative_balance(self, *, reference_node):
        """Remove two common origins and evaluate the actual relative field.

        The reference node selects a coordinate chart, not a physical origin
        or a new clock. Phase differences use captured real representatives
        of circular phase; their rate subtracts the moving reference rate.
        The weighted phase mean is likewise a local lifted observable, not
        a globally defined scalar on the phase torus.
        The supplied mobility law, coefficient and complete source are
        re-admitted; all rates, gradients and currents are rebuilt before use.
        """
        if self.law != "current_squared_reciprocal_mobility":
            raise ValueError("relative balance requires the declared mobility law")
        source = _rebuild_sine_comparison(self.comparison)
        mobility = _sine_current_squared_mobility(source, self.epsilon)
        rho = _sine_form_weights(source)
        positions = {node: i for i, node in enumerate(source.nodes)}
        if reference_node not in positions:
            raise ValueError("reference_node must be a captured node label")
        reference = positions[reference_node]
        indices = tuple(i for i in range(len(source.nodes)) if i != reference)
        total = sum(rho, Q(0))
        weights = tuple(value / total for value in rho)
        currents = tuple(imag for _, imag in source.relative_resultant)
        cosines = tuple(real for real, _ in source.relative_resultant)
        w = Q(source.reference_model.effective_weights[1])
        beta = Q(source.reference_model.storage_scale)
        pi = pi_interval()
        form_mean_rate = (
            (w * mobility.epsilon)
            / (pi * total)
            * sum(
                (s**3 / (degree**2) for s, degree in zip(currents, source.degrees)),
                I(0),
            )
        )
        phase_mean_rate = (
            (w * mobility.epsilon)
            / (beta * pi * total)
            * sum(
                (
                    s**2 * q / (degree**2)
                    for s, q, degree in zip(
                        currents, source.form_gradient, source.degrees
                    )
                ),
                I(0),
            )
        )
        direct_form = sum(
            (weight * rate for weight, rate in zip(weights, mobility.form_rates)), I(0)
        )
        direct_phase = sum(
            (weight * rate for weight, rate in zip(weights, mobility.phase_rates)), I(0)
        )
        divergence = tuple(
            (-2 * mobility.epsilon * w * nu * q) / (beta * pi * degree**3) * s * c
            for nu, q, s, c, degree in zip(
                source.capacity, source.form_gradient, currents, cosines, source.degrees
            )
        )
        full_divergence = sum(divergence, I(0))
        return SineMobilityRelativeBalance(
            mobility=self if mobility == self else mobility,
            reference_node=reference_node,
            reference_index=reference,
            relative_indices=indices,
            relative_form=tuple(source.epi[i] - source.epi[reference] for i in indices),
            relative_phase=tuple(
                source.phase[i] - source.phase[reference] for i in indices
            ),
            relative_form_rates=tuple(
                mobility.form_rates[i] - mobility.form_rates[reference] for i in indices
            ),
            relative_phase_rates=tuple(
                mobility.phase_rates[i] - mobility.phase_rates[reference]
                for i in indices
            ),
            normalized_mean_weights=weights,
            weighted_form_mean=sum(
                (weight * x for weight, x in zip(weights, source.epi)), Q(0)
            ),
            weighted_lifted_phase_mean=sum(
                (weight * angle for weight, angle in zip(weights, source.phase)), Q(0)
            ),
            weighted_form_mean_rate_bounds=form_mean_rate,
            weighted_lifted_phase_mean_rate_bounds=phase_mean_rate,
            direct_weighted_form_mean_rate_bounds=direct_form,
            direct_weighted_lifted_phase_mean_rate_bounds=direct_phase,
            weighted_form_mean_rate_residual_bounds=direct_form - form_mean_rate,
            weighted_lifted_phase_mean_rate_residual_bounds=direct_phase
            - phase_mean_rate,
            node_divergence_bounds=divergence,
            divergence_bounds=full_divergence,
            relative_divergence_bounds=full_divergence,
            constant_mobility_mean_and_volume_identities_certified=mobility.epsilon
            == 0,
        )


@dataclass(frozen=True)
class SineMobilityRelativeBalance:
    """Sufficient relative state and balances of the declared complete field.

    Weights are normalized d_i/nu_i, with every held capacity strictly
    positive. Mean-rate formulas cancel the exact unperturbed sums of S and
    L*x before interval evaluation. Source divergence and relative divergence
    agree by the constant-Jacobian change to two removed origin coordinates.
    Their nonzero value refuses the old Euclidean-volume proof, not recurrence
    itself or another invariant density. A zero snapshot proves no identity.
    """

    mobility: SineMobilityComparison
    reference_node: Any
    reference_index: int
    relative_indices: tuple[int, ...]
    relative_form: tuple[Q, ...]
    relative_phase: tuple[Q, ...]
    relative_form_rates: tuple[I, ...]
    relative_phase_rates: tuple[I, ...]
    normalized_mean_weights: tuple[Q, ...]
    weighted_form_mean: Q
    weighted_lifted_phase_mean: Q
    weighted_form_mean_rate_bounds: I
    weighted_lifted_phase_mean_rate_bounds: I
    direct_weighted_form_mean_rate_bounds: I
    direct_weighted_lifted_phase_mean_rate_bounds: I
    weighted_form_mean_rate_residual_bounds: I
    weighted_lifted_phase_mean_rate_residual_bounds: I
    node_divergence_bounds: tuple[I, ...]
    divergence_bounds: I
    relative_divergence_bounds: I
    constant_mobility_mean_and_volume_identities_certified: bool
    relative_field_closure_certified: bool = True
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "same_complete_phase_only_reciprocal_mobility_and_declared_source_clock",
        "strictly_positive_held_capacity_for_degree_over_capacity_mean_weights",
        "two_common_origins_removed_not_internal_continuous_state",
        "anchored_phase_coordinates_are_circular_with_supplied_raw_real_representatives",
        "weighted_phase_mean_is_a_local_lift_observable_not_a_global_torus_scalar",
        "relative_rates_subtract_the_moving_reference_form_and_phase_rates",
        "global_origin_shift_symmetry_makes_the_relative_field_autonomous",
        "full_and_relative_divergence_agree_under_constant_Jacobian_coordinates",
        "nonzero_divergence_blocks_Euclidean_volume_proof_not_other_densities",
        "zero_captured_mean_rate_or_divergence_is_not_an_all_state_identity",
        "no_mean_slab_recurrence_periodicity_or_trajectory_certificate",
        "no_source_recapture_reference_clock_redefinition_or_native_dispatch",
    )

    @property
    def comparison(self):
        return self.mobility.comparison

    def to_dict(self):
        from ..sdk.relational_reports import _project, _validate_label

        _validate_comparison_labels(self.comparison)
        _validate_label(self.reference_node)
        return {
            "schema": "tnfr.relational-sine-mobility-relative-balance.v1",
            "report": _project(self),
        }


def _comparison_neighbors(comparison):
    """Recover incidence in captured node order without consulting a graph."""
    positions = {node: i for i, node in enumerate(comparison.nodes)}
    rows = [[] for _ in comparison.nodes]
    for left, right in comparison.edges:
        i, j = positions[left], positions[right]
        rows[i].append(j)
        rows[j].append(i)
    return tuple(tuple(row) for row in rows)


def _validate_comparison_labels(comparison):
    from ..sdk.relational_reports import _validate_label

    for node in comparison.nodes:
        _validate_label(node)
    for edge in comparison.edges:
        for node in edge:
            _validate_label(node)


@dataclass(frozen=True)
class SineResultantKinematics:
    """Ideal directional bounds with the complete comparison source retained.

    ``speed_upper_bounds`` encloses sum_j |theta'_j-theta'_i|, an upper
    bound on |z'_i|. Its lower endpoint is not a lower bound on actual speed.
    These instantaneous results impose no positive-resultant admission and
    contain no future trajectory or whole-time speed certificate.
    """

    comparison: SineExchangeComparison
    phase_rate_numerators: tuple[Q, ...]
    resultant_rate_bounds: tuple[tuple[I, I], ...]
    speed_upper_bounds: tuple[I, ...]
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "ideal_normalized_sine_exchange_rates_at_exact_captured_coordinates",
        "shared_rational_direction_chain_rule_retains_common_mathematical_pi_factor",
        "supplied_comparison_is_retained_not_authenticated_by_dataclass_projection",
        "no_resultant_division_or_nonzero_resultant_requirement",
        "instantaneous_kinematics_not_whole_time_rate_or_trajectory_bounds",
        "no_graph_reread_native_field_temporal_sample_or_solver",
    )

    def to_dict(self):
        from ..sdk.relational_reports import _project

        _validate_comparison_labels(self.comparison)
        return {
            "schema": "tnfr.relational-sine-resultant-kinematics.v1",
            "report": _project(self),
        }


@dataclass(frozen=True)
class _SineState:
    """Shared detached admission and exact capture for sine-law reports."""

    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    epi: tuple[Q, ...]
    phase: tuple[Q, ...]
    capacity: tuple[Q, ...]
    neighbors: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]
    form_gradient: tuple[Q, ...]
    relative_resultant: tuple[tuple[I, I], ...]


def _admit_sine_comparison_rows(source):
    """Normalize a detached comparison's primitives and its actual incidence."""
    from ._sine_admission import _admit_sine_source

    if not isinstance(source, SineExchangeComparison):
        raise TypeError("a SineExchangeComparison source is required")
    admitted, _ = _admit_sine_source(source)
    return admitted, _comparison_neighbors(admitted)


def _rebuild_sine_comparison(source):
    """Rebuild every derived field after shared primitive and law admission.

    This is a detached consumer boundary, not graph capture or authentication.
    Always return normalized computed data; callers may retain an equivalent
    original association only after their arithmetic uses this rebuilt state.
    """
    admitted, neighbors = _admit_sine_comparison_rows(source)
    state = _sine_state_from_rows(
        admitted.nodes,
        admitted.edges,
        admitted.epi,
        admitted.phase,
        admitted.capacity,
        neighbors,
    )
    return _comparison_from_state(state, admitted.reference_model)


def _capture_sine_state(graph, reference_model):
    from ._sine_admission import _require_regular_sine_model

    _require_regular_sine_model(reference_model)
    # Shared relational staging revalidates the stored coefficients before
    # any state capture. It does not repeat constructor normalization.
    staged = _stage(graph, reference_model)
    nodes = tuple(staged)
    edges = tuple(staged.edges())
    indices = {node: i for i, node in enumerate(nodes)}
    epi = tuple(Q(staged.nodes[node][ALIAS_EPI[0]]) for node in nodes)
    phase = tuple(Q(staged.nodes[node][ALIAS_THETA[0]]) for node in nodes)
    capacity = tuple(Q(staged.nodes[node][ALIAS_VF[0]]) for node in nodes)
    neighbors = tuple(tuple(indices[j] for j in staged[node]) for node in nodes)
    return _sine_state_from_rows(nodes, edges, epi, phase, capacity, neighbors)


def _sine_state_from_rows(nodes, edges, epi, phase, capacity, neighbors):
    """Derive gradients and currents from admitted primitive state and incidence."""
    degrees = tuple(map(len, neighbors))
    gradient = _sine_form_gradient(epi, neighbors)
    resultants = tuple(
        (I(*real), I(*imag))
        for real, imag in relative_resultant_bounds(phase, neighbors)
    )
    return _SineState(
        nodes, edges, epi, phase, capacity, neighbors, degrees, gradient, resultants
    )


def _sine_form_gradient(epi, neighbors):
    """Exact form gradient on already admitted complete neighbor rows."""
    return tuple(
        sum((epi[i] - epi[j] for j in row), Q(0)) for i, row in enumerate(neighbors)
    )


def _sine_phase_rate_numerators(reference_model, degrees, gradient, capacity):
    """Exact phase row for admitted source or explicitly hypothetical gradients.

    Callers retain source/support provenance and supply already admitted exact
    rows. This kernel performs no capture or implicit hypothetical-state commit.
    """
    w = Q(reference_model.effective_weights[1])
    beta = Q(reference_model.storage_scale)
    return tuple(
        w * nu * q / (beta * degree)
        for nu, q, degree in zip(capacity, gradient, degrees)
    )


def _sine_rates(
    reference_model,
    degrees,
    gradient,
    capacity,
    phase_currents,
    *,
    exchange_factors=None,
):
    """Shared ideal rate rows for rational intervals or matched Taylor jets."""
    pi = pi_interval()
    inverse_metric = tuple(1 / (pi * degree) for degree in degrees)
    if exchange_factors is not None:
        inverse_metric = tuple(
            factor * mobility
            for factor, mobility in zip(exchange_factors, inverse_metric)
        )
    sources = tuple(
        imag * mobility for imag, mobility in zip(phase_currents, inverse_metric)
    )
    e, w = map(Q, reference_model.effective_weights)
    beta = Q(reference_model.storage_scale)
    pressure = tuple(
        -e * q / degree + w * source
        for q, degree, source in zip(gradient, degrees, sources)
    )
    form_rates = tuple(nu * value for nu, value in zip(capacity, pressure))
    phase_rates = tuple(
        (w / beta) * nu * q * mobility
        for nu, q, mobility in zip(capacity, gradient, inverse_metric)
    )
    return dict(
        inverse_phase_metric=inverse_metric,
        phase_sources=sources,
        pressure=pressure,
        form_rates=form_rates,
        phase_rates=phase_rates,
    )


def _sine_work(
    reference_model,
    degrees,
    gradient,
    capacity,
    phase_currents,
    *,
    exchange_factors=None,
):
    """Shared ideal rate/work rows for supplied reduced or full gradients."""
    rates = _sine_rates(
        reference_model,
        degrees,
        gradient,
        capacity,
        phase_currents,
        exchange_factors=exchange_factors,
    )
    form_rates, phase_rates = rates["form_rates"], rates["phase_rates"]
    e = Q(reference_model.effective_weights[0])
    beta = Q(reference_model.storage_scale)
    dissipation = tuple(
        e * nu * q**2 / degree for nu, q, degree in zip(capacity, gradient, degrees)
    )
    form_work = tuple(q * rate for q, rate in zip(gradient, form_rates))
    phase_work = tuple(
        -beta * imag * rate for imag, rate in zip(phase_currents, phase_rates)
    )
    residuals = tuple(
        left + right + loss
        for left, right, loss in zip(form_work, phase_work, dissipation)
    )
    loss = sum(dissipation, Q(0))
    storage_rate = sum(form_work, I(0)) + sum(phase_work, I(0))
    return dict(
        **rates,
        dissipation=dissipation,
        continuous_loss=loss,
        form_work=form_work,
        phase_work=phase_work,
        node_balance_residual=residuals,
        storage_rate=storage_rate,
        balance_residual=storage_rate + loss,
    )


def _sine_current_squared_mobility(source, epsilon):
    """Build the counterfamily from a freshly admitted/rebuilt sine source."""
    epsilon = exact_or_represented_real(epsilon, "epsilon")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if Q(source.reference_model.effective_weights[0]) != 0:
        raise ValueError("mobility comparison requires explicit zero form loss")
    currents = tuple(imag for _, imag in source.relative_resultant)
    corrections = tuple(
        epsilon * (current / degree) ** 2
        for current, degree in zip(currents, source.degrees)
    )
    factors = tuple(1 + correction for correction in corrections)
    return SineMobilityComparison(
        comparison=source,
        epsilon=epsilon,
        mobility_factors=factors,
        mobility_corrections=corrections,
        **_sine_work(
            source.reference_model,
            source.degrees,
            source.form_gradient,
            source.capacity,
            currents,
            exchange_factors=factors,
        ),
    )


def _sine_phase_storage(phase, edges):
    """Sum unweighted edge potentials using the shared 128-bit interval policy.

    Rows and index edges must already be admitted. This is a storage reader,
    independent of the precision used to enclose nodal resultants or rates.
    """
    return sum((1 - cos(I.coerce(phase[j] - phase[i])) for i, j in edges), I(0))


def _comparison_from_state(state, reference_model):
    """Rebuild the complete comparison from already admitted primitive rows."""
    indices = {node: i for i, node in enumerate(state.nodes)}
    edges = tuple((indices[left], indices[right]) for left, right in state.edges)
    form_storage = sum(((state.epi[i] - state.epi[j]) ** 2 / 2 for i, j in edges), Q(0))
    phase_storage = _sine_phase_storage(state.phase, edges)
    beta = Q(reference_model.storage_scale)
    return SineExchangeComparison(
        reference_model=reference_model,
        nodes=state.nodes,
        edges=state.edges,
        epi=state.epi,
        phase=state.phase,
        capacity=state.capacity,
        degrees=state.degrees,
        form_gradient=state.form_gradient,
        relative_resultant=state.relative_resultant,
        form_storage=form_storage,
        phase_storage=phase_storage,
        storage=form_storage + beta * phase_storage,
        **_sine_work(
            reference_model,
            state.degrees,
            state.form_gradient,
            state.capacity,
            tuple(imag for _, imag in state.relative_resultant),
        ),
    )


def _sine_regional_channel_balance(
    *, reference_model, edges, degrees, capacity, epi, phase, region_indices
):
    """Decompose a regional store for already admitted full-support rows.

    Callers own support, row dimensions, held-capacity association and region
    admission. Coordinates may be exact/represented signed scalars or rational
    intervals; interval enclosures are retained without midpoint substitution.
    Full rates use the shared sine-law kernel in its original structural t.
    """
    from ._sine_admission import _sine_model_coefficients

    e, w, beta = _sine_model_coefficients(reference_model)

    def interval(value, label):
        if isinstance(value, I):
            return I(
                exact_or_represented_real(value.lo, label),
                exact_or_represented_real(value.hi, label),
            )
        return I(exact_or_represented_real(value, label))

    forms = tuple(interval(value, "epi") for value in epi)
    phases = tuple(interval(value, "phase") for value in phase)
    capacities = tuple(interval(value, "capacity") for value in capacity)
    size = len(forms)
    if len(phases) != size or len(capacities) != size or len(degrees) != size:
        raise ValueError("channel rows must retain all full-support coordinates")
    if any(value.lo < 0 for value in capacities):
        raise ValueError("channel capacities must be nonnegative")
    if any(type(degree) is not int or degree <= 0 for degree in degrees):
        raise ValueError("channel degrees must be positive full-support integers")
    members = set(region_indices)
    if len(members) != len(region_indices) or any(
        type(i) is not int or not 0 <= i < size for i in region_indices
    ):
        raise ValueError(
            "channel region indices must be distinct full-source positions"
        )
    q, currents, qr, sr = ([I(0) for _ in range(size)] for _ in range(4))
    internal, boundary = [], []
    for i, j in edges:
        difference = forms[i] - forms[j]
        angle = phases[j] - phases[i]
        current = (
            I(*certified_sine_bounds(angle.lo)) if angle.lo == angle.hi else sin(angle)
        )
        q[i], q[j] = q[i] + difference, q[j] - difference
        currents[i], currents[j] = currents[i] + current, currents[j] - current
        if i in members and j in members:
            qr[i], qr[j] = qr[i] + difference, qr[j] - difference
            sr[i], sr[j] = sr[i] + current, sr[j] - current
            internal.append((i, j, difference, current))
        elif (i in members) != (j in members):
            boundary.append((i, j, difference, current))
    # Build the two boundary rows directly rather than subtracting dependent
    # interval sums q_R-q and S-S_R, which would discard cancellations.
    external_form, external_sine = ([I(0) for _ in range(size)] for _ in range(2))
    for i, j, difference, current in boundary:
        if i in members:
            external_form[i] -= difference
            external_sine[i] += current
        else:
            external_form[j] += difference
            external_sine[j] -= current
    rates = _sine_rates(reference_model, degrees, q, capacities, currents)
    factor = w / pi_interval()
    conversion, form_input, phase_input, signed_loss = (I(0) for _ in range(4))
    for i in region_indices:
        mobility = capacities[i] / degrees[i]
        conversion += factor * qr[i] * mobility * sr[i]
        form_input += factor * qr[i] * mobility * external_sine[i]
        phase_input += factor * sr[i] * mobility * external_form[i]
        signed_loss += e * qr[i] * mobility * q[i]
    # The derivatives below differentiate each internal edge separately,
    # independently of the channel partition used to reconstruct the balance.
    form_rate, phase_rate = I(0), I(0)
    for i, j, difference, current in internal:
        form_rate += difference * (rates["form_rates"][i] - rates["form_rates"][j])
        phase_rate += (
            beta * current * (rates["phase_rates"][j] - rates["phase_rates"][i])
        )
    flat_acceleration = None
    if all(phases[j] - phases[i] == I(0) for i, j, _, _ in internal):
        # At zero internal phase differences the potential gradient vanishes;
        # its second derivative therefore needs only the actual phase rates.
        flat_acceleration = sum(
            (
                beta * (rates["phase_rates"][j] - rates["phase_rates"][i]) ** 2
                for i, j, _, _ in internal
            ),
            I(0),
        )
    return SineRegionalChannelBalance(
        region_indices=tuple(region_indices),
        internal_form_gradient=tuple(qr[i] for i in region_indices),
        internal_sine_current=tuple(sr[i] for i in region_indices),
        external_form_contrast=tuple(external_form[i] for i in region_indices),
        external_sine_current=tuple(external_sine[i] for i in region_indices),
        form_rates=tuple(rates["form_rates"][i] for i in region_indices),
        phase_rates=tuple(rates["phase_rates"][i] for i in region_indices),
        internal_conversion=conversion,
        boundary_form_input=form_input,
        boundary_phase_input=phase_input,
        signed_form_loss=signed_loss,
        form_storage_rate=form_rate,
        weighted_phase_storage_rate=phase_rate,
        total_storage_rate=form_rate + phase_rate,
        form_balance_residual=form_rate - conversion - form_input + signed_loss,
        phase_balance_residual=phase_rate + conversion - phase_input,
        total_balance_residual=form_rate
        + phase_rate
        - form_input
        - phase_input
        + signed_loss,
        flat_phase_storage_acceleration=flat_acceleration,
    )


def _regional_storage_balance(source, *, region):
    from .relational_observations import _ordered, _regions

    if not isinstance(source, SineExchangeComparison):
        raise TypeError("regional storage requires an exact SineExchangeComparison")
    source = _rebuild_sine_comparison(source)
    positions = {node: i for i, node in enumerate(source.nodes)}
    edges = tuple(
        sorted(
            (
                min(positions[left], positions[right]),
                max(positions[left], positions[right]),
            )
            for left, right in source.edges
        )
    )
    region = _ordered(region, "region", limit=len(source.nodes) + 1)
    indices = _regions(source.nodes, (region,))[0]
    members = set(indices)
    complement = tuple(i for i in range(len(source.nodes)) if i not in members)
    internal, external, boundary = [], [], []
    for i, j in edges:
        if i in members and j in members:
            internal.append((i, j))
        elif i not in members and j not in members:
            external.append((i, j))
        else:
            boundary.append((i, j) if i in members else (j, i))
    internal, external, boundary = tuple(internal), tuple(external), tuple(boundary)
    beta = Q(source.reference_model.storage_scale)

    def edge_store_and_rate(rows):
        form, rate = Q(0), I(0)
        phase = _sine_phase_storage(source.phase, rows)
        for i, j in rows:
            dx, angle = source.epi[j] - source.epi[i], source.phase[j] - source.phase[i]
            current = I(*certified_sine_bounds(angle))
            form += dx**2 / 2
            rate += dx * (source.form_rates[j] - source.form_rates[i]) + (
                beta * current * (source.phase_rates[j] - source.phase_rates[i])
            )
        return form, phase, form + beta * phase, rate

    r_form, r_phase, r_storage, r_rate = edge_store_and_rate(internal)
    c_form, c_phase, c_storage, c_rate = edge_store_and_rate(external)
    b_form, b_phase, b_storage, b_rate = edge_store_and_rate(boundary)
    r_form_work, r_phase_work, c_form_work, c_phase_work = (I(0) for _ in range(4))
    for i, j in boundary:
        dx = source.epi[j] - source.epi[i]
        current = I(*certified_sine_bounds(source.phase[j] - source.phase[i]))
        r_form_work += dx * source.form_rates[i]
        r_phase_work += beta * current * source.phase_rates[i]
        c_form_work -= dx * source.form_rates[j]
        c_phase_work -= beta * current * source.phase_rates[j]
    r_work, c_work = r_form_work + r_phase_work, c_form_work + c_phase_work
    r_loss = sum((source.dissipation[i] for i in indices), Q(0))
    c_loss = sum((source.dissipation[i] for i in complement), Q(0))
    return SineRegionalStorageBalance(
        comparison=source,
        region=region,
        region_indices=indices,
        complement_indices=complement,
        regional_edge_indices=internal,
        complement_edge_indices=external,
        boundary_edge_indices=boundary,
        regional_form_storage=r_form,
        regional_phase_storage=r_phase,
        regional_storage=r_storage,
        complement_form_storage=c_form,
        complement_phase_storage=c_phase,
        complement_storage=c_storage,
        boundary_form_storage=b_form,
        boundary_phase_storage=b_phase,
        boundary_storage=b_storage,
        regional_boundary_form_work=r_form_work,
        regional_boundary_phase_work=r_phase_work,
        regional_boundary_work=r_work,
        complement_boundary_form_work=c_form_work,
        complement_boundary_phase_work=c_phase_work,
        complement_boundary_work=c_work,
        regional_loss=r_loss,
        complement_loss=c_loss,
        direct_regional_storage_rate=r_rate,
        direct_complement_storage_rate=c_rate,
        direct_boundary_storage_rate=b_rate,
        regional_balance_residual=r_rate - r_work + r_loss,
        complement_balance_residual=c_rate - c_work + c_loss,
        boundary_balance_residual=b_rate + r_work + c_work,
        storage_partition_residual=r_storage + c_storage + b_storage - source.storage,
        global_balance_residual=r_rate + c_rate + b_rate + r_loss + c_loss,
        regional_channels=_sine_regional_channel_balance(
            reference_model=source.reference_model,
            edges=edges,
            degrees=source.degrees,
            capacity=source.capacity,
            epi=source.epi,
            phase=source.phase,
            region_indices=indices,
        ),
        complement_channels=_sine_regional_channel_balance(
            reference_model=source.reference_model,
            edges=edges,
            degrees=source.degrees,
            capacity=source.capacity,
            epi=source.epi,
            phase=source.phase,
            region_indices=complement,
        ),
    )


def bound_relational_sine_exchange(graph, *, reference_model):
    """Enclose a separate smooth law without executing the native field.

    With q=B*x, degrees D, and s_i=sum_j sin(theta_j-theta_i)/(pi*d_i),
    the declared comparison is x'=N*(-e*D^-1*q+w*s) and
    theta'=(w/beta)*N*(pi*D)^-1*q. It retains cosine/Dirichlet storage and
    its ideal exact loss. Normalized pairwise pressure superposition and the
    declared storage/work/capacity premises select the constant phase mobility
    within that comparison class, not among all smooth collective laws.

    Shared staging admits aliases, signed form, finite coordinates, held
    nonnegative capacity, unit support and absent Gamma. No phase-resultant
    nonzero or branch admission applies to this different law. All source
    and work values use mathematical trigonometric enclosures, not a native
    floating-point field or derivative inferred from observed responses.
    """
    return _comparison_from_state(
        _capture_sine_state(graph, reference_model), reference_model
    )
