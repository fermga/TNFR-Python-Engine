"""Necessary budgets for two fixed whole-window Cartesian-box recipes.

This consumer evaluates no post-Silence box, Picard inclusion or trajectory.
Shared source admission independently recertifies the retained prebirth
history. The fresh initial moving-reference audit compares lower bounds on
chosen enclosure work budgets, not actual displacement or work.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from ..mathematics._rational_interval import INTERVAL_METHOD, I, pi_interval, sqrt
from .maintained_sha_reference import (
    ReferenceGate,
    ShaReferenceAudit,
    _bound_reference,
    _gate,
    prepare_sha_reference,
)


@dataclass(frozen=True)
class BoxCoefficientRows:
    """Exact full signed rows before the common mathematical-pi divisor.

    Rows of ``phase_rate_numerators`` are indexed by ``external_indices``;
    columns follow the complete source node order in the reference audit.
    They are H_ji in h_ji=H_ji/pi, not independently rounded mobility rows.
    """

    beta: tuple[Q, ...]
    center_mobility_numerator: Q
    external_port: tuple[Q, ...]
    weighted_laplacian: tuple[tuple[Q, ...], ...]
    external_indices: tuple[int, ...]
    phase_rate_numerators: tuple[tuple[Q, ...], ...]
    child_column_cancels: bool
    parent_column_absolute_sum_numerator: Q
    parent_column_absolute_sum: I


@dataclass(frozen=True)
class PairSaturationActivation:
    """Non-strict sufficient activation for the explicit row-bound subclass.

    A non-established activation is not a physical non-saturation statement.
    These margins concern radii demanded by that enclosure construction.
    """

    pair_margin: I | None
    parent_margin: I | None
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ShaBoxBudgetAudit:
    reference: ShaReferenceAudit
    coefficients: BoxCoefficientRows
    initial_form_rate_image_coefficient: I
    minimum_turn_time: I | None
    signal_budget_ceiling: I | None
    picard_image_budget_floor: I | None
    picard_image_gate: ReferenceGate
    saturation: PairSaturationActivation
    explicit_row_budget_floor: I | None
    explicit_row_gate: ReferenceGate
    arithmetic_method: str = INTERVAL_METHOD
    scope: tuple[str, ...] = (
        "fresh_initial_source_reference_and_complete_held_law_primitives",
        "same_original_clock_support_capacities_lifts_and_both_conserved_means",
        "single_full_window_Picard_image_with_symmetric_outer_radii_about_initial_state",
        "picard_image_floor_is_not_an_actual_displacement_or_work_lower_bound",
        "explicit_Sbar_Fbar_subclass_saturation_is_a_separate_sufficient_activation",
        "nonpositive_margin_excludes_only_its_registered_centered_box_work_recipe",
        "positive_necessary_margin_does_not_establish_a_feasible_box",
        "no_post_SHA_picard_call_candidate_box_horizon_scan_trajectory_event_or_refresh",
        "shared_source_admission_recertifies_retained_prebirth_history",
        "no_exclusion_of_subdivision_or_sharper_correlated_work_geometry",
    )


def _coefficient_rows(reference):
    """Rebuild exact rows from an internally fresh complete reference only."""
    budget = reference.pair_budget
    pair, source = budget.connected, budget.connected.source
    a, c = pair.parent_index, pair.child_index
    positions = {node: i for i, node in enumerate(source.nodes)}
    size = len(source.nodes)
    beta = tuple(
        nu / degree for nu, degree in zip(source.capacity, source.degrees, strict=True)
    )
    mur = beta[a] * beta[c] / (beta[a] + beta[c])
    laplacian = [[Q(0) for _ in range(size)] for _ in range(size)]
    port = [Q(0) for _ in range(size)]
    port[a] = budget.external_conductance_sum
    external_weights = {}
    for (left, right), weight in zip(source.edges, source.conductances, strict=True):
        i, j = positions[left], positions[right]
        laplacian[i][i] += weight
        laplacian[j][j] += weight
        laplacian[i][j] -= weight
        laplacian[j][i] -= weight
        if i == a and j != c:
            external_weights[j] = weight
        elif j == a and i != c:
            external_weights[i] = weight
    for j in pair.external_indices:
        port[j] = -external_weights[j]
    rows = tuple(
        tuple(beta[j] * laplacian[j][i] - mur * port[i] for i in range(size))
        for j in pair.external_indices
    )
    cancellation = all(row[c] == 0 for row in rows)
    if not cancellation:
        raise ArithmeticError(
            "the admitted pendant child lost its exact row cancellation"
        )
    parent_sum = sum((abs(row[a]) for row in rows), Q(0))
    expected_parent_sum = (
        sum((beta[j] * external_weights[j] for j in pair.external_indices), Q(0))
        + budget.external_degree * mur * budget.external_conductance_sum
    )
    if parent_sum != expected_parent_sum:
        raise ArithmeticError(
            "the parent coefficient sum lost its exact signed identity"
        )
    return BoxCoefficientRows(
        beta=beta,
        center_mobility_numerator=mur,
        external_port=tuple(port),
        weighted_laplacian=tuple(tuple(row) for row in laplacian),
        external_indices=pair.external_indices,
        phase_rate_numerators=rows,
        child_column_cancels=cancellation,
        parent_column_absolute_sum_numerator=parent_sum,
        parent_column_absolute_sum=I(parent_sum) / pi_interval(),
    )


def _saturation(pair_margin, parent_margin):
    if pair_margin is None or parent_margin is None:
        return PairSaturationActivation(
            pair_margin,
            parent_margin,
            "unavailable",
            ("minimum_turn_time_unavailable",),
        )
    if pair_margin.lo >= 0 and parent_margin.lo >= 0:
        return PairSaturationActivation(pair_margin, parent_margin, "established", ())
    failed = pair_margin.hi < 0 or parent_margin.hi < 0
    return PairSaturationActivation(
        pair_margin,
        parent_margin,
        "not_established" if failed else "unavailable",
        ("registered_activation_not_met" if failed else "activation_sign_unresolved",),
    )


def _box_budget_from_reference(reference):
    """Internal fresh-report consumer; both entry points rebuild all premises."""
    rows = _coefficient_rows(reference)
    pair = reference.pair_budget.connected
    source = pair.source
    Kx = (
        sum(
            (
                abs(coefficient) * abs(rate)
                for row in rows.phase_rate_numerators
                for coefficient, rate in zip(row, source.form_rates, strict=True)
            ),
            I(0),
        )
        / pi_interval()
    )
    work = reference.initial_work_budget
    duration = None if work is None else work.minimum_turn_time
    chart, state, rates = reference.chart, reference.state, reference.rates
    if (
        duration is None
        or state is None
        or rates is None
        or chart.status != "available"
    ):
        reason = "admitted_initial_reference_and_minimum_turn_time_required"
        return ShaBoxBudgetAudit(
            reference,
            rows,
            Kx,
            None,
            None,
            None,
            _gate(None, reason),
            _saturation(None, None),
            None,
            _gate(None, reason),
        )
    lower, upper = chart.relative_curvature_lower, chart.relative_curvature_upper
    root_m, root_lower, root_upper = (
        sqrt(I(chart.curvature_lower)),
        sqrt(I(lower)),
        sqrt(I(upper)),
    )
    ceiling = (
        lower * state.radial_excitation - upper * root_upper * state.reference_radius
    ) / (lower + upper * root_upper / root_lower)
    common = rates.work_norm * duration
    image_floor = common + chart.gamma * Kx * duration**2 / root_m
    scaled_time_squared = (pair.internal_rotation_coefficient * duration) ** 2
    saturation = _saturation(
        scaled_time_squared - 1, chart.gamma * scaled_time_squared - 1
    )
    explicit_floor = None
    explicit_gate = _gate(None, "explicit_pair_saturation_not_established")
    if saturation.status == "established":
        explicit_floor = (
            common
            + chart.gamma
            * pair.parent_mobility
            * source.degrees[pair.parent_index]
            * rows.parent_column_absolute_sum
            * duration**2
            / root_m
        )
        explicit_gate = _gate(
            ceiling - explicit_floor, "explicit_budget_margin_unresolved"
        )
    return ShaBoxBudgetAudit(
        reference=reference,
        coefficients=rows,
        initial_form_rate_image_coefficient=Kx,
        minimum_turn_time=duration,
        signal_budget_ceiling=ceiling,
        picard_image_budget_floor=image_floor,
        picard_image_gate=_gate(
            ceiling - image_floor, "picard_image_budget_margin_unresolved"
        ),
        saturation=saturation,
        explicit_row_budget_floor=explicit_floor,
        explicit_row_gate=explicit_gate,
    )


def _bound_box_budget(graph, *, parent, child, reference_model):
    """Rebuild manufactured controls through the existing shared admission."""
    return _box_budget_from_reference(
        _bound_reference(
            graph, parent=parent, child=child, reference_model=reference_model
        )
    )


def prepare_sha_box_budget(admitted):
    """Audit initial necessary budgets without integrating a post-SHA candidate.

    The existing reference wrapper re-admits retained files, primitive state,
    source association and clock, including the old prebirth-history certificate.
    Those bounded historical solver checks must be counted separately. A cached
    reference or box verdict is never a public premise. This static evaluation
    requires its prospective freeze.
    """
    return _box_budget_from_reference(prepare_sha_reference(admitted))
