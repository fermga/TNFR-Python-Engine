"""Detached open-domain bounds for Reception, Coherence and Silence.

The three primary maps act successively on the fixed organized-source support.
This reader installs no event, controller or pressure law. Its real-coordinate
domain assumes the declared initialized grammar/metadata fiber. Transfer to a
represented execution additionally requires checked stage defects, held fields
and callback contracts; no uniform platform trigonometry bound is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import bounded_represented_real
from .._exact_time import exact_or_represented_real as _exact
from .._exact_time import finite_represented_real
from ..mathematics._rational_interval import I, arg, cos, pi_interval, sin, sqrt
from ..operators.factor_contracts import validate_glyph_factor
from ._organized_source_capture import (
    _RETENTION_CEILING,
    _admit_organized_source_turns,
    _derive_organized_source_capture_barrier,
)
from ._organized_source_port import (
    _observe_organized_source_phase_state,
    _OrganizedSourcePhaseState,
    _source_emission_geometry,
)
from ._repeated_organized_composition import _vector

_TARGETS = (0, 1)


@dataclass(frozen=True)
class _OrganizedSourceWordDomain:
    """Sufficient domain of a fixed word, not an execution receipt."""

    source: _OrganizedSourcePhaseState
    phase_turns: tuple[int, ...]
    radius: Q
    reception_mix: Q
    phase_coefficient: Q
    pressure_retention: Q
    silence_capacity: Q
    reception_epi_ceiling: Q
    reception_pressure_ceiling: Q
    silence_min_capacity: Q
    reception_form_error: Q
    coherence_phase_error: Q
    pressure_error: Q
    final_capacities: tuple[Q, ...]
    center_form_after_reception: tuple[Q, ...]
    resultant_real_bounds: tuple[I, ...]
    resultant_imaginary_bounds: tuple[I, ...]
    resultant_real_lower_bounds: tuple[Q, ...]
    initial_edge_angle_bounds: tuple[I, ...]
    reception_pressure_upper_bounds: tuple[Q, ...]
    form_radius: Q
    center_weighted_form_mean: Q
    weighted_form_mean_absolute_bound: Q
    spectral_gap_lower_bound: Q | None
    capacity_ratio: Q
    unresolved_conditions: tuple[str, ...]
    admitted: bool = False
    center_phase_lift_after_coherence_bounds: tuple[I, ...] | None = None
    phase_radius: Q | None = None
    final_edge_angle_bounds: tuple[I, ...] | None = None
    center_excess_bounds: I | None = None
    form_energy_variation_bound: Q | None = None
    phase_energy_variation_bound: Q | None = None
    excess_upper_bound: Q | None = None
    centered_form_squared_upper_bound: Q | None = None
    form_absolute_upper_bound: Q | None = None
    retained_windings: tuple[int, ...] | None = None
    status: str = "unavailable"
    scope: tuple[str, ...] = (
        "ninety_dimensional_real_form_and_lift_box_on_fixed_discrete_context",
        "unit_initial_capacities_fixed_support_targets_clock_and_factor_policies",
        "simultaneous_targets_within_each_of_three_separately_atomic_stages",
        "all_nodes_share_one_nonempty_kind_and_supplied_initialized_AL_IL_history",
        "core_grammar_raise_mode_and_explicit_coherence_event_domain_policy",
        "hard_unit_form_rails_no_controllers_or_held_step_nodal_validator",
        "independent_pressure_only_refresh_after_each_stage_no_continuous_steps",
        "defect_budgets_are_hypotheses_not_uniform_libm_or_backend_certificates",
        "silence_holds_form_phase_support_and_realizes_declared_capacity_exactly",
        "continued_positive_held_capacity_reciprocal_phase_law_no_future_events",
        "actual_successor_mean_is_retained_not_projected_to_zero",
        "no_event_occurrence_runtime_receipt_source_acquisition_or_physical_claim",
    )


def _assess_organized_source_word_domain(
    *,
    form,
    phases,
    phase_turns,
    radius,
    reception_mix,
    phase_coefficient,
    pressure_retention,
    silence_capacity,
    reception_epi_ceiling,
    reception_pressure_ceiling,
    silence_min_capacity,
    reception_form_error=Q(0),
    coherence_phase_error=Q(0),
    pressure_error=Q(0),
):
    """Bound a complete primary word on independent form/phase perturbations.

    ``radius`` bounds each input form and declared real lift coordinate.
    Reception is the exact convex unweighted neighbor blend; Coherence uses
    the relative-resultant principal angle; Silence resets the two unit
    capacities. Supplied error budgets bound deviations from those maps at
    their actual inputs. In particular, the phase error includes the reset's
    mathematical-period lift, not just its stored wrapped representative.

    Every primitive is admitted before geometry. Finite centers outside the
    sufficient chamber return unavailable; no branch is silently selected.
    A passing result proves the conditional domain on the fixed runtime
    context in ``scope``. It neither executes nor authenticates that context.
    """
    values = _vector(form, "form", 45)
    angles = _vector(phases, "phases", 45)
    turns = _admit_organized_source_turns(phase_turns)
    parameters = {
        name: _exact(value, name)
        for name, value in (
            ("radius", radius),
            ("reception_mix", reception_mix),
            ("phase_coefficient", phase_coefficient),
            ("pressure_retention", pressure_retention),
            ("silence_capacity", silence_capacity),
            ("reception_epi_ceiling", reception_epi_ceiling),
            ("reception_pressure_ceiling", reception_pressure_ceiling),
            ("silence_min_capacity", silence_min_capacity),
            ("reception_form_error", reception_form_error),
            ("coherence_phase_error", coherence_phase_error),
            ("pressure_error", pressure_error),
        )
    }
    r = parameters["radius"]
    beta = parameters["reception_mix"]
    lam = parameters["phase_coefficient"]
    kappa = parameters["silence_capacity"]
    ex, et, ep = (
        parameters[name]
        for name in ("reception_form_error", "coherence_phase_error", "pressure_error")
    )
    if r <= 0:
        raise ValueError("radius must be strictly positive")
    for name, key in (
        ("reception_mix", "EN_mix"),
        ("pressure_retention", "IL_dnfr_factor"),
        ("silence_capacity", "SHA_vf_factor"),
    ):
        represented = Q(validate_glyph_factor(key, parameters[name]))
        if represented != parameters[name]:
            raise ValueError(f"{name} must be the exact represented runtime factor")
    represented_phase = bounded_represented_real(
        lam, "phase coefficient", minimum=0, maximum=1
    )[1]
    if represented_phase != lam:
        raise ValueError(
            "phase coefficient must be the exact represented runtime value"
        )
    for name in (
        "reception_epi_ceiling",
        "reception_pressure_ceiling",
        "silence_min_capacity",
    ):
        if finite_represented_real(parameters[name], name)[1] != parameters[name]:
            raise ValueError(f"{name} must be the exact represented runtime value")
    if kappa <= 0:
        raise ValueError("Silence continuation requires strictly positive capacity")
    if min(ex, et, ep, parameters["silence_min_capacity"]) < 0:
        raise ValueError("defect budgets and the capacity floor must be nonnegative")

    source = _observe_organized_source_phase_state(form=values, phases=angles)
    geometry = source.geometry
    barrier = _derive_organized_source_capture_barrier()
    geometry_ok, _diameter, cycle_gap, child_gap, _degrees = _source_emission_geometry(
        geometry
    )
    geometry_ok = geometry_ok and geometry == barrier.geometry
    gap = min(cycle_gap, child_gap) if geometry_ok else None
    pi = pi_interval()
    neighbors = tuple(
        tuple(b if a == n else a for a, b, _ in geometry.weighted_edges if n in (a, b))
        for n in _TARGETS
    )
    center_form = list(values)
    real, imaginary, margins = [], [], []
    for n, peers in zip(_TARGETS, neighbors):
        center_form[n] = (1 - beta) * values[n] + beta * sum(
            (values[j] for j in peers), Q(0)
        ) / len(peers)
        gaps = tuple(I(angles[j] - angles[n]) for j in peers)
        real.append(sum((cos(value) for value in gaps), I(0)))
        imaginary.append(sum((sin(value) for value in gaps), I(0)))
        margins.append(real[-1].lo - 2 * len(peers) * r)
    center_form = tuple(center_form)
    capacities = tuple(kappa if n in _TARGETS else Q(1) for n in range(45))
    weights = tuple(1 / nu for nu in capacities)
    mean = sum((w * x for w, x in zip(weights, center_form)), Q(0)) / sum(weights)
    a = r + ex
    mean_bound = abs(mean) + a
    initial_angles = tuple(
        I(angles[j] - angles[i])
        + 2
        * pi
        * (turns[j] - turns[i] + int(i // 9 == j // 9 and i % 9 == 8 and j % 9 == 0))
        + I(-2 * r, 2 * r)
        for i, j, _ in geometry.weighted_edges
    )
    pressure_bounds = tuple(
        source.pressure_tau_bounds[n].hi
        + Q(len(peers), 2) * (r + geometry.gamma_bounds.hi * r)
        + ep
        for n, peers in zip(_TARGETS, neighbors)
    )
    unresolved = tuple(
        name
        for passed, name in (
            (geometry_ok and barrier.barrier_certified, "fixed_geometry_unavailable"),
            (all(m > 0 for m in margins), "resultant_chamber_unavailable"),
            (
                all(edge.abs_max < pi.lo / 2 for edge in initial_angles),
                "initial_acute_cell_unavailable",
            ),
            (
                max(abs(x) for x in values) + r < 1,
                "input_hard_rail_margin_unavailable",
            ),
            (
                max(abs(x) for x in center_form) + a < 1,
                "reception_hard_rail_margin_unavailable",
            ),
            (
                all(
                    values[n] + r < parameters["reception_epi_ceiling"]
                    for n in _TARGETS
                ),
                "reception_signed_form_admission_unavailable",
            ),
            (
                all(
                    p < parameters["reception_pressure_ceiling"]
                    for p in pressure_bounds
                ),
                "reception_signed_pressure_admission_unavailable",
            ),
            (
                parameters["silence_min_capacity"] <= 1,
                "silence_initial_capacity_admission_unavailable",
            ),
        )
        if not passed
    )
    common = dict(
        source=source,
        phase_turns=turns,
        **parameters,
        final_capacities=capacities,
        center_form_after_reception=center_form,
        resultant_real_bounds=tuple(real),
        resultant_imaginary_bounds=tuple(imaginary),
        resultant_real_lower_bounds=tuple(margins),
        initial_edge_angle_bounds=initial_angles,
        reception_pressure_upper_bounds=pressure_bounds,
        form_radius=a,
        center_weighted_form_mean=mean,
        weighted_form_mean_absolute_bound=mean_bound,
        spectral_gap_lower_bound=gap,
        capacity_ratio=1 / kappa,
    )
    if unresolved:
        return _OrganizedSourceWordDomain(**common, unresolved_conditions=unresolved)

    changes = [I(0) for _ in angles]
    for n, re, im in zip(_TARGETS, real, imaginary):
        changes[n] = lam * arg(re, im)
    center_phase = tuple(
        I(theta) + 2 * pi * turn + change
        for theta, turn, change in zip(angles, turns, changes)
    )
    b = (
        max(
            r + lam * 2 * len(peers) * r / margin
            for peers, margin in zip(neighbors, margins)
        )
        + et
    )
    final_angles = tuple(
        I(angles[j] - angles[i])
        + changes[j]
        - changes[i]
        + 2
        * pi
        * (turns[j] - turns[i] + int(i // 9 == j // 9 and i % 9 == 8 and j % 9 == 0))
        + I(-2 * b, 2 * b)
        for i, j, _ in geometry.weighted_edges
    )
    if any(edge.abs_max >= pi.lo / 2 for edge in final_angles):
        return _OrganizedSourceWordDomain(
            **common,
            unresolved_conditions=("successor_acute_cell_unavailable",),
            center_phase_lift_after_coherence_bounds=center_phase,
            phase_radius=b,
            final_edge_angle_bounds=final_angles,
        )
    total_weight = sum((w for _, _, w in geometry.weighted_edges), Q(0))
    variation = sum(
        (
            w * abs(center_form[j] - center_form[i])
            for i, j, w in geometry.weighted_edges
        ),
        Q(0),
    )
    f0 = sum(
        (
            w * (center_form[j] - center_form[i]) ** 2 / 2
            for i, j, w in geometry.weighted_edges
        ),
        Q(0),
    )
    v0 = sum(
        (
            w * (1 - cos(I(angles[j] - angles[i]) + changes[j] - changes[i]))
            for i, j, w in geometry.weighted_edges
        ),
        I(0),
    )
    center_excess = f0 + v0 - barrier.target_storage_bounds
    df, dv = 2 * a * variation + 2 * a**2 * total_weight, 2 * b * total_weight
    upper = center_excess.hi + df + dv
    if upper < 0:
        raise ValueError("negative storage excess bound contradicts the fixed carrier")
    centered_bound = 2 * upper / (kappa * gap)
    rail_bound = mean_bound + sqrt(I(centered_bound)).hi
    unresolved += tuple(
        name
        for passed, name in (
            (
                upper < _RETENTION_CEILING < barrier.boundary_energy_lower_bound,
                "successor_energy_ceiling_unavailable",
            ),
            (rail_bound < 1, "future_form_rail_margin_unavailable"),
        )
        if not passed
    )
    admitted = not unresolved
    return _OrganizedSourceWordDomain(
        **common,
        unresolved_conditions=unresolved,
        admitted=admitted,
        center_phase_lift_after_coherence_bounds=tuple(center_phase),
        phase_radius=b,
        final_edge_angle_bounds=final_angles,
        center_excess_bounds=center_excess,
        form_energy_variation_bound=df,
        phase_energy_variation_bound=dv,
        excess_upper_bound=upper,
        centered_form_squared_upper_bound=centered_bound,
        form_absolute_upper_bound=rail_bound,
        retained_windings=(1,) * 5 if admitted else None,
        status="certified_conditional_word_domain" if admitted else "unavailable",
    )
