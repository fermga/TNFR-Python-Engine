"""Detached complete flow/word tangent at the organized carrier phase.

ORGANIZED_SOURCE_PORT.md owns the complete-law comparison and its initialized
event context. This reader retains both rows of ``DW F_1 - F_kappa W`` on the
shared support, with exact Reception forms and symbolic Coherence derivatives.
It executes neither an event nor a trajectory and supplies no occurrence law.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import bounded_represented_real
from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import I, cos
from ..operators.factor_contracts import validate_glyph_factor
from ._organized_source_port import (
    _derive_organized_source_port,
    _OrganizedSourceGeometry,
    _source_emission_geometry,
)
from ._repeated_organized_composition import _vector
from ._sine_preparation import _prepared_form_costs

_TARGETS = (0, 1)
_ENERGY_CEILING = Q(1, 100)
_MEAN_CEILING = Q(1, 30)


@dataclass(frozen=True)
class _OrganizedSourceCarrierTiming:
    """Exact carrier tangent and phase enclosures, not a runtime receipt.

    With ``c=cos(2*pi/9)``, the normalized phase row is exactly
    ``phase_commutator_base + (phase_commutator_matching +
    c*phase_commutator_cycle)/(3+2*c)``. The reported interval row includes
    gamma; the three rational coefficient rows do not. Primitive phases are
    fixed symbolically at ``psi[9*child+j]=2*pi*(j-4)/9`` throughout this
    reader's input domain. They are not globally equal phases.
    """

    geometry: _OrganizedSourceGeometry
    form: tuple[Q, ...]
    successor_form: tuple[Q, ...]
    form_gradient: tuple[Q, ...]
    successor_form_gradient: tuple[Q, ...]
    successor_capacities: tuple[Q, ...]
    energy_excess: Q
    initial_form_mean: Q
    successor_form_mean: Q
    form_commutator: tuple[Q, ...]
    phase_commutator_base: tuple[Q, ...]
    phase_commutator_matching: tuple[Q, ...]
    phase_commutator_cycle: tuple[Q, ...]
    phase_commutator_bounds: tuple[I, ...]
    gamma_bounds: I
    carrier_cosine_bounds: I
    reception_mix: Q
    phase_coefficient: Q
    silence_capacity_factor: Q
    scope: tuple[str, ...] = (
        "fixed_shared_45_node_90_unit_edge_support_and_targets_zero_one",
        "exact_organized_winding_one_carrier_phases_not_global_phase_consensus",
        "initial_unit_capacities_and_positive_held_silence_successor_capacities",
        "independent_pressure_and_reciprocal_phase_rows_in_the_same_tau_clock",
        "ideal_simultaneous_reception_and_coherence_maps_then_silence",
        "coherence_phase_map_fixes_carrier_but_its_full_derivative_is_retained",
        "raw_forms_and_actual_inverse_capacity_successor_mean_are_retained",
        "stored_pressure_retention_is_separate_from_fresh_constitutive_pressure",
        "initialized_grammar_context_and_live_event_policies_are_separate_premises",
        "exact_form_tangent_symbolic_phase_tangent_and_outward_phase_enclosures",
        "no_events_trajectories_finite_time_error_bounds_or_runtime_admission",
    )


def _assess_organized_source_carrier_timing(
    form, *, reception_mix, phase_coefficient, silence_capacity_factor
):
    """Rebuild the complete local timing difference from admitted forms.

    Reception has ``A=I-(beta/5)*P*L`` and Silence gives ``D`` with target
    capacities kappa. At the exact carrier, ``S(psi)=0`` and ``C(psi)=psi``,
    but ``DC`` acts nontrivially on the incoming phase rate. Consequently the
    form row is ``(D*L*A*x-A*L*x)/4`` and the phase row is
    ``gamma*(DC*L*x-D*L*A*x)/4``. Capacity rates vanish on both sides.

    The phase derivative uses each target's actual two cycle neighbors and
    three matching neighbors. Its symbolic denominator ``3+2*cos(alpha)``
    is the positive carrier resultant; no numerical phase reset is executed.
    Strict excess and initial-mean bounds select the section 23 source domain.
    Context, policy and represented execution remain separate obligations.
    """
    values = _vector(form, "form", 45)
    beta = _exact(reception_mix, "reception_mix")
    lam = _exact(phase_coefficient, "phase_coefficient")
    kappa = _exact(silence_capacity_factor, "silence_capacity_factor")
    for label, key, value in (
        ("reception_mix", "EN_mix", beta),
        ("silence_capacity_factor", "SHA_vf_factor", kappa),
    ):
        if Q(validate_glyph_factor(key, value)) != value:
            raise ValueError(f"{label} must equal its represented runtime value")
    if (
        bounded_represented_real(lam, "phase_coefficient", minimum=0, maximum=1)[1]
        != lam
    ):
        raise ValueError("phase coefficient must equal its represented runtime value")
    if kappa <= 0:
        raise ValueError("positive continuation excludes zero Silence capacity")

    geometry = _derive_organized_source_port()
    valid, *_ = _source_emission_geometry(geometry)
    if not valid or geometry.support_degrees[:2] != (5, 5):
        raise ValueError("the organized-source degree-five target support is required")
    edges = geometry.weighted_edges
    costs, gradient = _prepared_form_costs(form=values, weighted_edges=edges)
    energy = sum(costs, Q(0))
    mean = sum(values, Q(0)) / len(values)
    if energy >= _ENERGY_CEILING:
        raise ValueError("carrier energy excess must be strictly below one hundredth")
    if abs(mean) >= _MEAN_CEILING:
        raise ValueError(
            "absolute initial form mean must be strictly below one thirtieth"
        )

    step = beta / geometry.support_degrees[0]
    successor = tuple(
        value - step * gradient[i] if i in _TARGETS else value
        for i, value in enumerate(values)
    )
    capacities = tuple(kappa if i in _TARGETS else Q(1) for i in range(45))
    _, successor_gradient = _prepared_form_costs(form=successor, weighted_edges=edges)
    _, gradient_laplacian = _prepared_form_costs(form=gradient, weighted_edges=edges)
    reception_gradient = tuple(
        value - step * gradient_laplacian[i] if i in _TARGETS else value
        for i, value in enumerate(gradient)
    )
    form_difference = tuple(
        (capacity * after - transported) / 4
        for capacity, after, transported in zip(
            capacities, successor_gradient, reception_gradient
        )
    )
    phase_base = tuple(
        (before - capacity * after) / 4
        for before, capacity, after in zip(gradient, capacities, successor_gradient)
    )
    matching, cycle = [Q(0)] * 45, [Q(0)] * 45
    for target in _TARGETS:
        for left, right, weight in edges:
            if target not in (left, right):
                continue
            neighbor = right if left == target else left
            channel = cycle if target // 9 == neighbor // 9 else matching
            channel[target] -= (
                lam * weight * (gradient[target] - gradient[neighbor]) / 4
            )
    matching, cycle = tuple(matching), tuple(cycle)
    carrier_cosine = cos(geometry.alpha_bounds)
    resultant = 3 + 2 * carrier_cosine
    if resultant.lo <= 0:
        raise ValueError("the positive organized-carrier resultant is required")
    phase_bounds = tuple(
        geometry.gamma_bounds * (base + (match + cyclic * carrier_cosine) / resultant)
        for base, match, cyclic in zip(phase_base, matching, cycle)
    )
    weights = tuple(1 / capacity for capacity in capacities)
    successor_mean = sum(
        (weight * value for weight, value in zip(weights, successor)), Q(0)
    ) / sum(weights, Q(0))
    return _OrganizedSourceCarrierTiming(
        geometry=geometry,
        form=values,
        successor_form=successor,
        form_gradient=gradient,
        successor_form_gradient=successor_gradient,
        successor_capacities=capacities,
        energy_excess=energy,
        initial_form_mean=mean,
        successor_form_mean=successor_mean,
        form_commutator=form_difference,
        phase_commutator_base=phase_base,
        phase_commutator_matching=matching,
        phase_commutator_cycle=cycle,
        phase_commutator_bounds=phase_bounds,
        gamma_bounds=geometry.gamma_bounds,
        carrier_cosine_bounds=carrier_cosine,
        reception_mix=beta,
        phase_coefficient=lam,
        silence_capacity_factor=kappa,
    )
