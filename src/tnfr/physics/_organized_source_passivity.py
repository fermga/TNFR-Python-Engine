"""Full acute-domain work of the ideal Reception--Coherence--Silence word.

This detached reader reuses the source barrier and arbitrary-mean admission.
It evaluates no runtime event or trajectory. The ideal phase increments are
independent, not projected onto reflection; represented execution retains its
own work, lift, policy and callback obligations.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import bounded_represented_real
from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, arg, cos, sin
from ..operators.factor_contracts import validate_glyph_factor
from ._organized_source_capture import (
    _assess_organized_source_weighted_mean_retention,
    _OrganizedSourceWeightedMeanRetention,
)


@dataclass(frozen=True)
class _OrganizedSourceWordPassivity:
    """Conditional ideal work and mean resets, without a runtime admission."""

    domain: _OrganizedSourceWeightedMeanRetention
    reception_mix: Q
    phase_coefficient: Q
    silence_capacity_factor: Q
    admitted: bool = False
    reception_form: tuple[Q, ...] | None = None
    reception_work: Q | None = None
    reception_work_upper_bound: Q | None = None
    reception_symmetric_loss: Q | None = None
    reception_antisymmetric_loss: Q | None = None
    relative_resultant_real_bounds: tuple[I, ...] | None = None
    relative_resultant_imaginary_bounds: tuple[I, ...] | None = None
    phase_angle_bounds: tuple[I, ...] | None = None
    phase_displacement_bounds: tuple[I, ...] | None = None
    phase_work_bounds: I | None = None
    isolated_phase_work_bounds: tuple[I, ...] | None = None
    shared_edge_correction_bounds: I | None = None
    work_decomposition_residual_bounds: I | None = None
    phase_work_upper_bound: Q | None = None
    word_work_upper_bound: Q | None = None
    silence_work: Q | None = None
    final_capacities: tuple[Q, ...] | None = None
    form_mean_before: Q | None = None
    form_mean_after_reception: Q | None = None
    form_mean_after_silence: Q | None = None
    reception_mean_jump: Q | None = None
    silence_mean_jump: Q | None = None
    strict_reception_loss_certified: bool = False
    strict_phase_loss_certified: bool = False
    all_reset_prefixes_passive: bool = False
    retained_windings: tuple[int, ...] | None = None
    unresolved_conditions: tuple[str, ...] = ()
    status: str = "unavailable"
    scope: tuple[str, ...] = (
        "full_raw_acute_low_energy_domain_without_reflection_or_zero_mean",
        "same_fixed_ninety_unit_edges_and_simultaneous_targets_zero_one",
        "ideal_primary_maps_with_exact_common_reception_and_phase_coefficients",
        "local_resultant_and_joint_shared_edge_work_bounds_reuse_existing_proofs",
        "comparison_reset_prefixes_are_not_elapsed_continuous_flow",
        "positive_held_capacity_reciprocal_continuation_between_admitted_events",
        "mean_resets_are_rebuilt_from_raw_form_and_changed_capacity_weights",
        "passivity_does_not_supply_form_rails_runtime_policies_or_event_occurrence",
        "no_rounded_capacity_product_or_represented_phase_work_certificate",
    )


def _assess_organized_source_word_passivity(
    *,
    form,
    phases,
    phase_turns,
    capacities,
    reception_mix,
    phase_coefficient,
    silence_capacity_factor,
):
    """Rebuild one ideal word's joint work on the complete maintained domain.

    For each input state the mean reader checks the actual acute lift and
    storage ceiling. Its sufficient form-rail flag is not a premise here.
    Every ideal EN/IL reset prefix is passive by the same snapshot formula,
    so the existing acute boundary cannot be crossed during either jump.
    No reflected endpoint, finite sampling or positive tolerance proves this.
    """
    beta = _exact(reception_mix, "reception_mix")
    lam = _exact(phase_coefficient, "phase_coefficient")
    kappa = _exact(silence_capacity_factor, "silence_capacity_factor")
    for name, value in (("EN_mix", beta), ("SHA_vf_factor", kappa)):
        if Q(validate_glyph_factor(name, value)) != value:
            raise ValueError(f"{name} must equal its represented runtime value")
    if (
        bounded_represented_real(lam, "phase_coefficient", minimum=0, maximum=1)[1]
        != lam
    ):
        raise ValueError("phase coefficient must equal its represented runtime value")
    if kappa <= 0:
        raise ValueError("positive continuation excludes zero Silence capacity")
    domain = _assess_organized_source_weighted_mean_retention(
        form=form, phases=phases, phase_turns=phase_turns, capacities=capacities
    )
    common = dict(
        domain=domain,
        reception_mix=beta,
        phase_coefficient=lam,
        silence_capacity_factor=kappa,
    )
    if not domain.geometric_retention_certified:
        return _OrganizedSourceWordPassivity(
            **common, unresolved_conditions=domain.unresolved_conditions
        )

    observation = domain.observation
    reference = observation.unit_capacity_reference
    geometry = reference.geometry
    edges = geometry.weighted_edges
    targets = (0, 1)
    neighbors = tuple(
        tuple(b if a == n else a for a, b, _ in edges if n in (a, b)) for n in targets
    )
    if tuple(map(len, neighbors)) != (5, 5) or not any(
        {a, b} == {0, 1} and weight == 1 for a, b, weight in edges
    ):
        return _OrganizedSourceWordPassivity(
            **common, unresolved_conditions=("target_block_geometry_unavailable",)
        )
    q0, q1 = reference.form_gradient[:2]
    step = beta / 5
    # Exact modes retain the correlation used by the cumulative mean budget.
    symmetric_loss = step * (1 - 2 * step) * (q0 + q1) ** 2 / 2
    antisymmetric_loss = step * (1 - 3 * step) * (q0 - q1) ** 2 / 2
    en_work = -(symmetric_loss + antisymmetric_loss)
    en_upper = -step * (1 - 3 * step) * (q0**2 + q1**2)
    after_form = list(reference.form)
    after_form[0] -= step * q0
    after_form[1] -= step * q1
    after_form = tuple(after_form)
    angles = reference.phases
    real, imaginary = [], []
    for n, peers in zip(targets, neighbors):
        gaps = tuple(I(angles[j] - angles[n]) for j in peers)
        real.append(sum((cos(gap) for gap in gaps), I(0)))
        imaginary.append(sum((sin(gap) for gap in gaps), I(0)))
    if any(value.lo <= 4 for value in real):
        return _OrganizedSourceWordPassivity(
            **common,
            reception_form=after_form,
            reception_work=en_work,
            reception_work_upper_bound=en_upper,
            unresolved_conditions=("full_domain_resultant_bound_unresolved",),
        )
    phase_angles = tuple(arg(re, im) for re, im in zip(real, imaginary))
    if any(value.abs_max >= Q(1, 8) for value in phase_angles):
        return _OrganizedSourceWordPassivity(
            **common, unresolved_conditions=("full_domain_angle_bound_unresolved",)
        )
    displacement = tuple(lam * value for value in phase_angles)
    changes = displacement + (I(0),) * 43
    work = sum(
        (
            w
            * (
                cos(I(angles[b] - angles[a]))
                - cos(I(angles[b] - angles[a]) + changes[b] - changes[a])
            )
            for a, b, w in edges
            if a in targets or b in targets
        ),
        I(0),
    )
    isolated = tuple(
        sum(
            (
                cos(I(angles[j] - angles[n]))
                - cos(I(angles[j] - angles[n]) - displacement[n])
                for j in peers
            ),
            I(0),
        )
        for n, peers in zip(targets, neighbors)
    )
    gap = I(angles[1] - angles[0])
    correction = (
        cos(gap - displacement[0])
        + cos(gap + displacement[1])
        - cos(gap)
        - cos(gap + displacement[1] - displacement[0])
    )
    if lam == 0:
        work = correction = I(0)
        isolated = (I(0), I(0))
    lower_angle_square = sum(
        (
            min(value.lo**2, value.hi**2) if value.lo > 0 or value.hi < 0 else Q(0)
            for value in phase_angles
        ),
        Q(0),
    )
    # This analytic upper bound, not a clipped numeric work box, proves passivity.
    phase_upper = -lam * (3 - 2 * lam) * lower_angle_square
    capacities_before = observation.capacities
    capacities_after = tuple(
        nu * kappa if n in targets else nu for n, nu in enumerate(capacities_before)
    )

    def mean(values, mobilities):
        weights = tuple(1 / nu for nu in mobilities)
        return sum((w * x for w, x in zip(weights, values)), Q(0)) / sum(weights)

    initial_mean = domain.weighted_form_mean
    en_mean = mean(after_form, capacities_before)
    sha_mean = mean(after_form, capacities_after)
    return _OrganizedSourceWordPassivity(
        **common,
        admitted=True,
        reception_form=after_form,
        reception_work=en_work,
        reception_work_upper_bound=en_upper,
        reception_symmetric_loss=symmetric_loss,
        reception_antisymmetric_loss=antisymmetric_loss,
        relative_resultant_real_bounds=tuple(real),
        relative_resultant_imaginary_bounds=tuple(imaginary),
        phase_angle_bounds=phase_angles,
        phase_displacement_bounds=displacement,
        phase_work_bounds=work,
        isolated_phase_work_bounds=isolated,
        shared_edge_correction_bounds=correction,
        work_decomposition_residual_bounds=work - sum(isolated, I(0)) - correction,
        phase_work_upper_bound=phase_upper,
        word_work_upper_bound=en_upper + phase_upper,
        silence_work=Q(0),
        final_capacities=capacities_after,
        form_mean_before=initial_mean,
        form_mean_after_reception=en_mean,
        form_mean_after_silence=sha_mean,
        reception_mean_jump=en_mean - initial_mean,
        silence_mean_jump=sha_mean - en_mean,
        strict_reception_loss_certified=en_work < 0,
        strict_phase_loss_certified=phase_upper < 0,
        all_reset_prefixes_passive=True,
        retained_windings=(1,) * 5,
        status="certified_conditional_ideal_word_passivity",
    )
