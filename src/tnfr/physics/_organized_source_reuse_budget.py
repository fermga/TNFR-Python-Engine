"""Detached mean and form budgets for finite ideal organized-source words.

The prescribed word is Reception, Coherence, Silence on targets zero and one.
The budget assumes the separately justified passive primary maps and reciprocal
positive-capacity continuation. It executes no events or flow and does not
establish grammar, the Silence capacity floor, occurrence or backend errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, sqrt
from ..operators.factor_contracts import validate_glyph_factor
from ._organized_source_port import (
    _derive_organized_source_port,
    _source_emission_geometry,
)

_MAX_WORD_COUNT = 4096
_ENERGY_CEILING = Q(1, 100)


@dataclass(frozen=True)
class _OrganizedSourceReuseBudget:
    """Conditional bounds, not a receipt or permission to invoke an operator."""

    reception_mix: Q
    silence_capacity_factor: Q
    initial_mean_bound: Q
    energy_budget: Q
    word_count: int
    initial_pair_capacity: Q
    final_pair_capacity: Q
    support_diameter: int
    target_degree: int
    reception_step: Q
    reception_mean_loss_coefficient: Q
    inverse_mean_denominator_squared_sum_bounds: I
    pair_weight_increment: Q
    form_range_bounds: I
    reception_mean_drift_bounds: I
    silence_mean_drift_bounds: I
    final_mean_absolute_upper_bound: Q
    form_absolute_upper_bound: Q
    reception_pressure_absolute_upper_bound: Q
    unit_form_rails_certified: bool
    scope: tuple[str, ...] = (
        "fixed_45_node_90_unit_edge_support_and_targets_zero_one",
        "exact_positive_pair_capacity_initial_v_times_kappa_power_j_others_one",
        "initial_acute_cell_and_actual_excess_strictly_below_one_hundredth",
        "energy_budget_bounds_initial_excess_and_all_intermediate_excess",
        "reception_exact_unclipped_simultaneous_pair_map_with_declared_mix",
        "coherence_holds_forms_and_is_separately_proved_storage_nonincreasing",
        "silence_holds_forms_phases_support_and_multiplies_pair_capacity",
        "exact_reciprocal_shared_capacity_continuation_between_complete_words",
        "arbitrary_finite_waits_and_final_future_flow_with_capacity_held",
        "finite_word_count_is_a_work_budget_not_capacity_or_grammar_admission",
        "sum_defined_exactly_by_admitted_v_kappa_count_enclosed_on_dyadic_grid",
        "no_events_flows_runtime_roundoff_or_autonomous_selection_are_certified",
    )


def _support_diameter(geometry):
    """Derive distances from the shared support, without a second edge catalog."""
    adjacency = {node: set() for node in geometry.node_order}
    for left, right, _ in geometry.weighted_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    diameter = 0
    for origin in geometry.node_order:
        distances = {origin: 0}
        frontier = [origin]
        for node in frontier:
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    frontier.append(neighbor)
        if len(distances) != len(adjacency):
            raise ValueError("organized-source support must be connected")
        diameter = max(diameter, max(distances.values()))
    return diameter


def _assess_organized_source_reuse_budget(
    *,
    reception_mix,
    silence_capacity_factor,
    initial_mean_bound,
    energy_budget,
    word_count,
    initial_pair_capacity=Q(1),
):
    """Bound mean drift and form rails under the declared finite composition.

    All non-target capacities are one; the two targets start at ``v`` and have
    capacity ``v*kappa**j`` before word ``j``. The exact finite sum is
    ``sum((43*v*kappa**j + 2)**-2 for j in range(word_count))``. Its defining
    primitives are retained; outward dyadic accumulation avoids constructing
    the enormous common denominator of a collapsed rational sum.

    For ``a=beta/5`` and ``Q=q[0]+q[1]``, the Reception mean shift is
    ``-a*Q/(43*v+2)``. Its loss has the exact correlated component
    ``a*(1-2*a)*Q**2/2``. Cauchy's inequality bounds all Reception mean
    shifts using their *joint* loss budget, not a fresh copy per event.
    Silence changes only the weighting of the same forms. Those pair-weight
    increments telescope, bounding total mean drift by their net increment
    times the common form-range bound.

    ``energy_budget`` may equal 1/100 as an upper bound, but actual states
    must satisfy the strict acute-cell excess hypothesis. Geometric retention,
    passivity of Coherence, live policies and represented defects are separate
    obligations. A failed sufficient rail check does not prove a rail exit.
    """
    if type(word_count) is not int or not 0 <= word_count <= _MAX_WORD_COUNT:
        raise ValueError("word_count must be an integer between zero and 4096")
    values = {
        name: _exact(value, name)
        for name, value in (
            ("reception_mix", reception_mix),
            ("silence_capacity_factor", silence_capacity_factor),
            ("initial_mean_bound", initial_mean_bound),
            ("energy_budget", energy_budget),
            ("initial_pair_capacity", initial_pair_capacity),
        )
    }
    beta = values["reception_mix"]
    kappa = values["silence_capacity_factor"]
    mean = values["initial_mean_bound"]
    energy = values["energy_budget"]
    initial_capacity = values["initial_pair_capacity"]
    for name, key in (
        ("reception_mix", "EN_mix"),
        ("silence_capacity_factor", "SHA_vf_factor"),
    ):
        if Q(validate_glyph_factor(key, values[name])) != values[name]:
            raise ValueError(f"{name} must be the exact represented runtime factor")
    if kappa <= 0 or initial_capacity <= 0:
        raise ValueError("continued pair capacities must be strictly positive")
    if mean < 0:
        raise ValueError("initial_mean_bound must be nonnegative")
    if not 0 <= energy <= _ENERGY_CEILING:
        raise ValueError("energy_budget must lie between zero and one hundredth")

    geometry = _derive_organized_source_port()
    valid, *_ = _source_emission_geometry(geometry)
    if not valid:
        raise ValueError("the organized-source support contract is required")
    diameter = _support_diameter(geometry)
    if diameter != 7 or geometry.support_degrees[:2] != (5, 5):
        raise ValueError("the diameter-seven, degree-five target support is required")

    a = beta / 5
    coefficient = 2 * a / (1 - 2 * a)
    denominator_sum = I(0)
    capacity_bounds = I(initial_capacity)
    for _ in range(word_count):
        denominator_sum += 1 / (43 * capacity_bounds + 2) ** 2
        capacity_bounds *= kappa
    final_capacity = initial_capacity * kappa**word_count
    pair_weight_increment = 2 / (43 * final_capacity + 2) - 2 / (
        43 * initial_capacity + 2
    )
    form_range = sqrt(I(2 * diameter * energy))
    reception_drift = sqrt(energy * coefficient * denominator_sum)
    silence_drift = pair_weight_increment * form_range
    mean_bound = mean + reception_drift.hi + silence_drift.hi
    form_bound = mean_bound + form_range.hi
    degree = geometry.support_degrees[0]
    pressure_bound = (sqrt(I(2 * degree * energy)) + degree * geometry.gamma_bounds) / 4
    return _OrganizedSourceReuseBudget(
        reception_mix=beta,
        silence_capacity_factor=kappa,
        initial_mean_bound=mean,
        energy_budget=energy,
        word_count=word_count,
        initial_pair_capacity=initial_capacity,
        final_pair_capacity=final_capacity,
        support_diameter=diameter,
        target_degree=degree,
        reception_step=a,
        reception_mean_loss_coefficient=coefficient,
        inverse_mean_denominator_squared_sum_bounds=denominator_sum,
        pair_weight_increment=pair_weight_increment,
        form_range_bounds=form_range,
        reception_mean_drift_bounds=reception_drift,
        silence_mean_drift_bounds=silence_drift,
        final_mean_absolute_upper_bound=mean_bound,
        form_absolute_upper_bound=form_bound,
        reception_pressure_absolute_upper_bound=pressure_bound.hi,
        unit_form_rails_certified=form_bound < 1,
    )
