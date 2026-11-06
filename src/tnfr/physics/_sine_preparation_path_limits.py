"""Energy-compatible terminal geometries for the unchanged C9 source path.

The source-to-transit calculation is rebuilt from primitive preparation inputs.
Exact odd-cycle closure then restricts the critical limits of those same full
trajectories. Candidate membership is a necessary energy test, not reachability
or basin selection. Saddles remain candidates for exceptional preparations.
Proof: theory/nodal/SINE_PREPARATION_PATH_LIMITS.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos, pi_interval
from ._sine_preparation_path_selection import (
    _assess_preparation_path_selection,
    _PreparationPathSelection,
)
from .phase_cycle_geometry import (
    _odd_cycle_negative_phase_modes,
    _odd_cycle_sine_templates,
)


@dataclass(frozen=True)
class _C9TerminalGeometry:
    """One labeled critical relative phase state, without a chosen common lift.

    Edges use cycle order 0->1->...->8->0, unlike the sorted graph-edge order
    of the transit report. Turns are exact circular representatives. An
    antipodal edge leaves winding unavailable, rather than choosing a cut.
    """

    edge_turns: tuple[Q, ...]
    principal_sine_turn: Q
    supplementary_mask: int
    storage_bounds: I
    phase_hessian_inertia: tuple[int, int, int]
    relative_stable_modes: int
    relative_unstable_modes: int
    winding: int | None
    energy_status: str

    @property
    def locally_attracting(self):
        return self.relative_unstable_modes == 0


def _c9_terminal_geometries(storage_upper_bound):
    """Rebuild exact critical branches and apply a conservative energy filter.

    A bound straddling a critical storage retains that branch as unresolved.
    Equality also retains it: dissipation only proves endpoint nonincrease.
    This standalone calculation supplies no trajectory or convergence verdict.
    """
    budget = _exact(storage_upper_bound, "storage_upper_bound")
    if budget < 0:
        raise ValueError("storage_upper_bound must be nonnegative")
    options, currents, masks = _odd_cycle_sine_templates(9)
    pi = pi_interval()
    storage_by_type = {}
    branches = []
    for turns, current, mask in zip(options, currents, masks):
        count = mask.bit_count()
        key = (count, abs(current))
        if key not in storage_by_type:
            storage_by_type[key] = (
                I(2 * count)
                if current == 0
                else 9 - (9 - 2 * count) * cos(2 * pi * current)
            )
        storage = storage_by_type[key]
        negative = _odd_cycle_negative_phase_modes(9, count)
        principal = tuple((turn + Q(1, 2)) % 1 - Q(1, 2) for turn in turns)
        period = sum(principal, Q(0))
        if period.denominator != 1:
            raise ArithmeticError("the exact critical cycle failed closure")
        winding = (
            None if any(abs(turn) == Q(1, 2) for turn in principal) else int(period)
        )
        status = (
            "excluded_by_storage"
            if storage.lo > budget
            else (
                "not_excluded_by_storage"
                if storage.hi <= budget
                else "unresolved_storage_comparison"
            )
        )
        branches.append(
            _C9TerminalGeometry(
                edge_turns=turns,
                principal_sine_turn=current,
                supplementary_mask=mask,
                storage_bounds=storage,
                phase_hessian_inertia=(8 - negative, negative, 0),
                relative_stable_modes=16 - negative,
                relative_unstable_modes=negative,
                winding=winding,
                energy_status=status,
            )
        )
    return tuple(branches)


@dataclass(frozen=True)
class _PreparationPathLimits:
    """Necessary terminal alternatives for the original complete source family.

    Full lifted convergence retains the conserved means and a trajectory's
    eventual integer lift offsets. Neither those offsets nor its terminal
    branch are selected. Null exceptional sets refer to sixteen-dimensional
    latent-error volume at each fixed parameter, not to a source probability
    or to the unperturbed one-dimensional preparation curve.
    """

    source: _PreparationPathSelection
    critical_geometries: tuple[_C9TerminalGeometry, ...]
    candidate_indices: tuple[int, ...]
    unresolved_energy_indices: tuple[int, ...]
    attracting_candidate_indices: tuple[int, ...]
    saddle_candidate_indices: tuple[int, ...]
    attracting_candidate_windings: tuple[int, ...]
    single_relative_equilibrium_convergence_certified: bool
    full_lifted_state_convergence_certified: bool
    saddle_source_volume_zero_at_each_parameter_certified: bool
    selected_terminal_index: None = None
    selection_status: str = "unavailable_no_source_to_basin_selector"
    scope: tuple[str, ...] = (
        "fixed_isolated_unit_C9_complete_positive_loss_sine_law_and_held_capacities",
        "source_and_transit_rebuilt_from_primitives_without_a_state_reset",
        "same_actual_histories_and_storage_nonincrease_after_the_declared_time",
        "complete_exact_labeled_critical_geometry_modulo_common_circular_origin",
        "energy_compatibility_is_necessary_not_reachability_or_source_selection",
        "finite_critical_set_and_connected_omega_limit_give_one_relative_limit",
        "conserved_phase_lift_mean_reconstructs_full_lifted_convergence",
        "joint_form_phase_stability_not_phase_gradient_flow_substitution",
        "saddle_stable_manifolds_remain_exceptional_source_candidates",
        "null_source_volume_requires_full_dimensional_errors_at_each_fixed_parameter",
        "no_probability_law_or_nominal_curve_selection_follows_from_null_volume",
        "no_uniform_capture_deadline_final_mode_or_terminal_integer_lift_selected",
        "no_trajectory_parameter_sweep_or_frozen_producer_is_evaluated",
    )


def _assess_preparation_path_limits(
    *, parameter_interval, scaled_time, form_error_bound, phase_error_bound
) -> _PreparationPathLimits:
    """Carry the original source into its energy-compatible critical limits.

    Inputs retain the path owner's exact admission and budgets. No incoming
    report, cached certificate or independently initialized state is consumed.
    The complete law has a finite hyperbolic relative critical set regardless
    of source-policy success; energy filtering only narrows its alternatives.
    """
    source = _assess_preparation_path_selection(
        parameter_interval=parameter_interval,
        scaled_time=scaled_time,
        form_error_bound=form_error_bound,
        phase_error_bound=phase_error_bound,
    )
    geometries = _c9_terminal_geometries(source.endpoint_total_storage_upper_bound)
    candidates = tuple(
        index
        for index, branch in enumerate(geometries)
        if branch.energy_status != "excluded_by_storage"
    )
    attracting = tuple(
        index for index in candidates if geometries[index].locally_attracting
    )
    saddles = tuple(
        index for index in candidates if not geometries[index].locally_attracting
    )
    return _PreparationPathLimits(
        source=source,
        critical_geometries=geometries,
        candidate_indices=candidates,
        unresolved_energy_indices=tuple(
            index
            for index in candidates
            if geometries[index].energy_status == "unresolved_storage_comparison"
        ),
        attracting_candidate_indices=attracting,
        saddle_candidate_indices=saddles,
        attracting_candidate_windings=tuple(
            sorted(geometries[index].winding for index in attracting)
        ),
        single_relative_equilibrium_convergence_certified=True,
        full_lifted_state_convergence_certified=True,
        saddle_source_volume_zero_at_each_parameter_certified=source.full_relative_preparation,
    )
