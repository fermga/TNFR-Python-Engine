"""Prospective derivative budgets for an independently bounded sine-law class.

The bounds use the full initial form diameter and a ceiling on every held
capacity, including hidden nodes. They require no samples, fitted derivatives
or trajectory. They neither infer those premises nor close a noisy-state
inverse or establish a physical measurement model.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import INTERVAL_METHOD, pi_interval
from ._sine_admission import _sine_model_coefficients
from .relational_observations import (
    RelationalSampleJetBudget,
    _ordered,
    bound_relational_sample_jet_budget,
)

__all__ = (
    "SineSamplingSmoothness",
    "SineSamplingBudget",
    "bound_sine_sampling_smoothness",
)


@dataclass(frozen=True)
class SineSamplingSmoothness:
    """Uniform full-network rate bounds derived from supplied class premises."""

    reference_model: RelationalExchangeModel
    initial_form_diameter_bound: Q
    capacity_ceiling: Q
    window: tuple[Q, Q]
    sine_exchange_upper_bound: Q
    phase_exchange_upper_bound: Q
    form_diameter_upper_bound: Q
    form_speed_bound: Q
    phase_speed_bound: Q
    form_acceleration_bound: Q
    phase_acceleration_bound: Q
    form_third_derivative_bound: Q
    phase_third_derivative_bound: Q
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "supplied_smooth_sine_law_fixed_simple_unit_support_and_held_capacities",
        "each_node_has_nonzero_degree_no_forcing_events_or_clipping",
        "diameter_and_capacity_ceiling_bound_all_nodes_including_hidden_nodes",
        "initial_diameter_premise_applies_at_the_start_of_the_enlarged_window",
        "maximum_principle_and_edge_chain_rule_not_a_fitted_smoothness_budget",
        "global_form_and_phase_shifts_do_not_change_these_bounds",
        "no_graph_read_samples_trajectory_or_preparation_authentication",
    )

    def sample_budget(
        self,
        *,
        sample_step,
        form_sample_error_bounds,
        phase_sample_error_bounds,
        timestamp_error_bounds=(0, 0, 0),
        observation_time=0,
    ):
        """Rebuild the declared class bounds, then compose temporal budgets."""
        window = _ordered(self.window, "smoothness.window", limit=3)
        if len(window) != 2:
            raise ValueError("smoothness.window must contain two ordered times")
        smoothness = bound_sine_sampling_smoothness(
            reference_model=self.reference_model,
            form_diameter_bound=self.initial_form_diameter_bound,
            capacity_ceiling=self.capacity_ceiling,
            window_start=window[0],
            window_end=window[1],
        )
        common = dict(
            sample_step=sample_step,
            timestamp_error_bounds=timestamp_error_bounds,
            observation_time=observation_time,
        )
        form = bound_relational_sample_jet_budget(
            sample_error_bounds=form_sample_error_bounds,
            first_derivative_bound=smoothness.form_speed_bound,
            third_derivative_bound=smoothness.form_third_derivative_bound,
            **common,
        )
        common = dict(
            sample_step=form.sample_step,
            timestamp_error_bounds=form.timestamp_error_bounds,
            observation_time=form.observation_time,
        )
        phase = bound_relational_sample_jet_budget(
            sample_error_bounds=phase_sample_error_bounds,
            first_derivative_bound=smoothness.phase_speed_bound,
            third_derivative_bound=smoothness.phase_third_derivative_bound,
            **common,
        )
        if not (
            smoothness.window[0] <= form.evidence_window[0]
            and form.evidence_window[1] <= smoothness.window[1]
        ):
            raise ValueError(
                "sampling evidence exceeds the independently bounded window"
            )
        pi_lower = pi_interval().lo
        margins = tuple(
            pi_lower
            - smoothness.phase_speed_bound
            * (
                phase.sample_step
                + phase.timestamp_error_bounds[j]
                + phase.timestamp_error_bounds[j + 1]
            )
            - phase.sample_error_bounds[j]
            - phase.sample_error_bounds[j + 1]
            for j in range(2)
        )
        return SineSamplingBudget(smoothness, form, phase, margins)

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-sampling-smoothness.v1", "report": _project(self)}


@dataclass(frozen=True)
class SineSamplingBudget:
    """A prior error budget, without synthetic sample values or an inverse."""

    smoothness: SineSamplingSmoothness
    form: RelationalSampleJetBudget
    phase: RelationalSampleJetBudget
    phase_increment_margins: tuple[Q, Q]
    scope: tuple[str, ...] = (
        "one_nominal_initial_time_for_both_rates_and_accelerations",
        "noise_clock_and_truncation_contributions_retained_separately",
        "positive_phase_margins_suffice_for_unique_nearest_increment_lifting",
        "a_phase_lift_still_requires_an_initial_reference_and_actual_sample_admission",
        "zero_declared_anchor_error_is_not_an_inferred_exact_preparation",
        "derivative_boxes_do_not_authenticate_a_raw_sample_generating_trajectory",
        "no_acquisition_hidden_state_inference_future_prediction_or_physical_admission",
    )

    @property
    def phase_increments_resolved(self):
        return min(self.phase_increment_margins) > 0

    def to_dict(self):
        from ..sdk.relational_reports import _project

        return {"schema": "tnfr.sine-sampling-budget.v1", "report": _project(self)}


def bound_sine_sampling_smoothness(
    *, reference_model, form_diameter_bound, capacity_ceiling, window_start, window_end
):
    """Derive uniform C3 bounds without knowing a particular hidden state.

    At a maximum of form, dissipative diffusion has nonpositive sign; sine
    forcing is at most N*a. The minimum has the opposite bound, giving
    D(t)<=D0+2*N*a*T. Exact edge differentiation then bounds first through
    third form/phase derivatives on the entire supplied interval. Pi's
    certified lower bound yields rational upper coefficients; no floating
    transcendental value substitutes for mathematical pi.
    """
    e, w, beta = _sine_model_coefficients(reference_model)
    diameter = exact_or_represented_real(form_diameter_bound, "form_diameter_bound")
    ceiling = exact_or_represented_real(capacity_ceiling, "capacity_ceiling")
    start = exact_or_represented_real(window_start, "window_start")
    end = exact_or_represented_real(window_end, "window_end")
    if diameter < 0 or ceiling < 0 or not 0 <= start <= end:
        raise ValueError("require nonnegative bounds and 0<=window_start<=window_end")
    a, b = w / pi_interval().lo, w / (beta * pi_interval().lo)
    whole_diameter = diameter + 2 * ceiling * a * (end - start)
    velocity = ceiling * (e * whole_diameter + a)
    omega = ceiling * b * whole_diameter
    acceleration = ceiling * (2 * e * velocity + 2 * a * omega)
    alpha = 2 * ceiling * b * velocity
    third_form = ceiling * (2 * e * acceleration + a * (4 * omega**2 + 2 * alpha))
    third_phase = 2 * ceiling * b * acceleration
    return SineSamplingSmoothness(
        reference_model,
        diameter,
        ceiling,
        (start, end),
        a,
        b,
        whole_diameter,
        velocity,
        omega,
        acceleration,
        alpha,
        third_form,
        third_phase,
    )
