"""Conditional continuation of the complete mixed-source Coupling image.

The ideal proposal supplies its frozen conductances and phase reset. Uniform
coefficient inclusion proves the following two complete laws retain that image;
it does not authenticate represented endpoints or execute an operator word.
"""

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._interval_taylor import Jet
from ..mathematics._rational_interval import I, cos, pi_interval
from ..operators import _coupling_stage_kernel as coupling_policy
from ._sine_flow import _held_sine_neighbor_field
from ._sine_mixed_capture import (
    _assess_sine_mixed_capture,
    _derive_mixed_constitutive_pair,
    _SineMixedCapture,
)
from .relational_observations import _ordered
from .support_transport import _laplacian

_COMPLETIONS = ("common_sense", "split_sense")
_NORMALIZATIONS = ("bare_degree", "transport_strength")
_SOURCE_LIMIT = Q(1, 10**10)
_EPSILON = Q(1, 10**70)
_ETA = 20 * _EPSILON
_DELTA = Q(1, 1024)
_RADIUS = Q(1, 32)
_LOSS = Q(1023, 1024)
_NEW_WEIGHT_RANGE = (Q(3, 4), Q(1))
_GAMMA_RANGE = (Q(1, 3300), Q(1, 3000))


def _choice(value, name, choices):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


@dataclass(frozen=True)
class _CouplingSuccessorGeometry:
    completion: str
    nodes: tuple[int, ...]
    internal_edges: tuple[tuple[int, int, Q], ...]
    new_edge_pairs: tuple[tuple[int, int], ...]
    edge_pairs: tuple[tuple[int, int], ...]
    degree: int
    degrees: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    target_phase_turns: tuple[Q, ...]
    phase_indices: tuple[int, ...]
    marked_pairs: tuple[tuple[int, int], ...]
    marked_branch_turns: tuple[Q, ...]
    internal_eigenvalue_bounds: tuple[I, ...]
    bare_gap_lower_bound: Q
    weighted_gap_lower_bound: Q
    quotient_coercivity_lower_bound: Q
    support_certified: bool
    reference_stationary_certified: bool
    spectral_gap_certified: bool
    held_capacities: tuple[Q, ...] = (Q(1),) * 18


def _successor_geometry(base, completion):
    """Build support from a freshly derived source geometry, not an input report."""
    internal = tuple(
        sorted((min(a, b), max(a, b), weight) for a, b, weight in base.internal_edges)
    )
    new_pairs = {
        tuple(sorted((9 * ring + j, 9 * ring + (j + step) % 9)))
        for ring, step in ((0, 2), (1, 4))
        for j in range(9)
    }
    common = completion == "common_sense"
    if common:
        for a in range(9):
            for b in range(9, 18):
                gap = (
                    base.target_phase_turns[b] - base.target_phase_turns[a] + Q(1, 2)
                ) % 1 - Q(1, 2)
                if abs(gap) <= Q(2, 9):
                    new_pairs.add((a, b))
    new_pairs = tuple(sorted(new_pairs))
    pairs = tuple((a, b) for a, b, _ in internal) + new_pairs
    degrees = tuple(sum(i in edge for edge in pairs) for i in base.nodes)
    degree = 9 if common else 4
    components = (base.nodes,) if common else (tuple(range(9)), tuple(range(9, 18)))
    phase_indices = tuple(int((9 * turn) % 9) for turn in base.target_phase_turns)
    neighbor_bins = tuple(
        tuple(
            (
                base.target_phase_turns[b if i == a else a]
                - base.target_phase_turns[i]
                + Q(1, 2)
            )
            % 1
            - Q(1, 2)
            for a, b in pairs
            if i in (a, b)
        )
        for i in base.nodes
    )
    stationary = all(
        sorted(row) == sorted(-value for value in row) for row in neighbor_bins
    )
    pi = pi_interval()
    eigenvalues = tuple(
        4 - 2 * cos(2 * pi * m / 9) - 2 * cos(4 * pi * m / 9) for m in range(1, 9)
    )
    support = (
        base.nodes == tuple(range(18))
        and base.class_order == (1, 2)
        and base.held_capacities == (Q(1),) * 18
        and len(internal) == 18
        and all(weight == 1 for _, _, weight in internal)
        and len(new_pairs) == (63 if common else 18)
        and len(set(pairs)) == len(pairs)
        and degrees == (degree,) * 18
        and all(
            tuple(sorted(phase_indices[9 * r : 9 * r + 9])) == tuple(range(9))
            for r in range(2)
        )
        and all(abs(value) <= Q(2, 9) for row in neighbor_bins for value in row)
    )
    return _CouplingSuccessorGeometry(
        completion=completion,
        nodes=base.nodes,
        internal_edges=internal,
        new_edge_pairs=new_pairs,
        edge_pairs=pairs,
        degree=degree,
        degrees=degrees,
        components=components,
        target_phase_turns=base.target_phase_turns,
        phase_indices=phase_indices,
        marked_pairs=tuple((a, b) for a, b, _ in base.contact_edges),
        marked_branch_turns=base.contact_branch_turns,
        internal_eigenvalue_bounds=eigenvalues,
        bare_gap_lower_bound=Q(4 if common else 2),
        weighted_gap_lower_bound=Q(3) if common else Q(3, 2),
        quotient_coercivity_lower_bound=Q(1, 5) if common else Q(1, 10),
        support_certified=support,
        reference_stationary_certified=support and stationary,
        spectral_gap_certified=support and min(value.lo for value in eigenvalues) > 2,
    )


def _derive_coupling_successor_geometry(*, completion):
    """Return fixed bare support and its phase-index spectral certificate."""
    completion = _choice(completion, "completion", _COMPLETIONS)
    return _successor_geometry(_derive_mixed_constitutive_pair(), completion)


def _coupling_successor_field(
    *, completion, new_edge_conductances, loss_normalization="bare_degree"
):
    """Compile all 36 rows with held weights and refreshed constitutive pressure.

    New weights follow ``geometry.new_edge_pairs`` and are exact/represented
    scalars in [3/4,1]. These are members of an enclosing coefficient family,
    not authenticated proposals. Every old edge has weight one. States use
    the shared scalar/interval or common-order Taylor-jet domain, forms first.
    The structural clock and reciprocal phase row are the same for both losses.
    """
    completion = _choice(completion, "completion", _COMPLETIONS)
    loss_normalization = _choice(
        loss_normalization, "loss_normalization", _NORMALIZATIONS
    )
    count = 63 if completion == "common_sense" else 18
    raw = _ordered(new_edge_conductances, "new_edge_conductances", limit=count + 1)
    if len(raw) != count:
        raise ValueError(f"new_edge_conductances requires {count} ordered weights")
    weights = tuple(
        _exact(value, f"new_edge_conductances[{j}]") for j, value in enumerate(raw)
    )
    if any(
        not _NEW_WEIGHT_RANGE[0] <= weight <= _NEW_WEIGHT_RANGE[1] for weight in weights
    ):
        raise ValueError("new edge conductances must lie in [3/4,1]")
    geometry = _derive_coupling_successor_geometry(completion=completion)
    if not geometry.support_certified:
        raise ValueError("successor support is unavailable")
    edges = geometry.internal_edges + tuple(
        (a, b, weight)
        for (a, b), weight in zip(geometry.new_edge_pairs, weights, strict=True)
    )
    rows = tuple(
        tuple((b if i == a else a, 1, weight) for a, b, weight in edges if i in (a, b))
        for i in geometry.nodes
    )
    model = RelationalExchangeModel(
        1, epi_weight=_LOSS, phase_weight=Q(1, 1024), phase_domain="regular"
    )
    evaluate = _held_sine_neighbor_field(
        model, geometry.degrees, geometry.held_capacities, rows
    )
    directed = tuple(
        row for a, b, weight in edges for row in ((a, b, weight), (b, a, weight))
    )
    strengths = tuple(
        sum((w for a, _, w in directed if a == i), Q(0)) for i in geometry.nodes
    )
    correction = tuple(Q(1, geometry.degree) - 1 / strength for strength in strengths)

    def flow(values):
        state = _ordered(values, "state", limit=37)
        rates = tuple(value / _LOSS for value in evaluate(state))
        if loss_normalization == "bare_degree":
            return rates
        forms = state[:18]
        if not isinstance(forms[0], Jet):
            forms = tuple(I.coerce(value) for value in forms)
        gradient = _laplacian(directed, forms)
        return (
            tuple(
                value + factor * q
                for value, factor, q in zip(rates[:18], correction, gradient)
            )
            + rates[18:]
        )

    return flow


@dataclass(frozen=True)
class _CouplingSuccessorCompletion:
    """Both losses share this quotient trap, not their absolute mean form."""

    geometry: _CouplingSuccessorGeometry
    initial_phase_norm_squared_upper_bound: Q
    initial_quotient_norm_squared_upper_bound: Q
    initial_phase_radius_margin: Q
    initial_quotient_radius_margin: Q
    initial_energy_upper_bound: Q
    storage_barrier: Q
    storage_margin: Q
    initial_registration_bounds: tuple[Q, Q]
    registration_limit_bounds: tuple[Q, Q]
    constituent_identity_certified: bool
    asymptotic_convergence_certified: bool
    joint_registration_certified: bool
    disconnected_registration_certified: bool
    degree_form_limit_bounds: tuple[tuple[Q, Q], ...]
    strength_form_limit_bounds: tuple[tuple[Q, Q], ...] | None
    inventory: str = "sum_component(x)+(d/gamma)*sum_component(B*y)"
    target_form: str = "component_inventory/component_size; quotient form tends to zero"


@dataclass(frozen=True)
class _SineCouplingSuccessor:
    """Uniform conditional theorem for the full ideal image and its two losses.

    Candidate numbers do not establish unavailable source premises. The old
    reference supplies only geometry, original recoveries and isolated waiting.
    ``None`` split strength form bounds mean the limit is retained through its
    exact component inventory, not set to zero or inferred from nominal data.
    No finite observation deadline, rate or source-to-runtime authentication is
    supplied. The strength common-form interval concerns the connected image.
    Its negative form limit uses the correlated weights and reset of the ideal
    UM image. It is not asserted for arbitrary weights admitted by the field
    builder; all-unit weights, for example, make its two loss laws identical.
    """

    form_error_bound: Q
    phase_error_bound: Q
    precontact_reference: _SineMixedCapture
    completion_reports: tuple[_CouplingSuccessorCompletion, ...]
    source_certified: bool
    isolated_waiting_certified: bool
    geometry_certified: bool
    conditional_image_certified: bool
    form_similarity_regularizer: Q
    similarity_regularizer_margin: Q
    proposal_regularizer_certified: bool
    phase_curvature_bounds: I
    phase_curvature_margin: Q
    nominal_strength_bounds: tuple[I, I]
    nominal_common_strength_form_limit_bounds: I
    nominal_form_magnitude_bounds: tuple[Q, Q]
    common_form_limit_error_upper_bound: Q
    common_strength_form_limit_bounds: tuple[Q, Q]
    common_form_target_margins: tuple[Q, Q]
    constituent_identity_certified: bool
    connected_joint_registration_certified: bool
    disconnected_nonregistration_certified: bool
    normalization_common_form_separation_certified: bool
    certified: bool
    status: str
    unresolved_conditions: tuple[str, ...]
    source_allowance_limit: Q = _SOURCE_LIMIT
    waiting_radius: Q = _ETA
    phase_reset_error_bound: Q = 3 * _ETA
    relative_phase_gap: Q = _DELTA
    phase_radius: Q = _RADIUS
    phase_cosine_lower_bound: Q = Q(1, 10)
    new_weight_range: tuple[Q, Q] = _NEW_WEIGHT_RANGE
    edge_weight_deviation_upper_bound: Q = 10**9 * _ETA
    strength_deviation_upper_bound: Q = 7 * 10**9 * _ETA
    mobility_deviation_upper_bound: Q = 2 * 10**8 * _ETA
    gauge_phase_deviation_upper_bound: Q = 6 * _ETA
    gamma_range: tuple[Q, Q] = _GAMMA_RANGE
    registered_gap_target: Q = _DELTA / 8
    clock: str = (
        "tau=(1023/1024)*t; bare divisors9/4 in reciprocal phase and sine current"
    )
    scope: tuple[str, ...] = (
        "actual_ideal_proposal_image_inclusion_not_authenticated_represented_endpoint",
        "fresh_source_and_waiting_only_no_old_joined_capture_verdict_transfer",
        "frozen_actual_proposed_weights_not_weighted_sine_or_time_varying_feedback",
        "both_complete_declared_losses_with_refreshed_pressure_and_unit_capacities",
        "component_mean_gauges_and_inventories_retain_reset_mean_changes",
        "all_time_quotient_trap_and_constituent_windings_asymptotic_registration",
        "no_quantitative_deadline_rate_trajectory_or_measured_observation",
        "nonzero_strength_common_form_not_a_shift_of_the_original_form_origin",
        "no_committed_event_autonomous_selector_or_physical_identification",
    )


def _assess_sine_coupling_successor(*, form_error_bound, phase_error_bound):
    """Rebuild the source and prove uniform continuation of its ideal image."""
    rx = _exact(form_error_bound, "form_error_bound")
    rt = _exact(phase_error_bound, "phase_error_bound")
    if min(rx, rt) < 0:
        raise ValueError("source error bounds must be nonnegative")
    regularizer = _exact(
        coupling_policy._EPI_SIMILARITY_EPSILON, "UM form-similarity regularizer"
    )
    regularizer_margin = regularizer - Q(9, 10**10)
    regularizer_ok = regularizer_margin > 0
    base = _derive_mixed_constitutive_pair()
    reference = _assess_sine_mixed_capture(
        form_error_bound=rx, phase_error_bound=rt, observation_error_bound=0
    )
    source_ok = (
        max(rx, rt) <= _SOURCE_LIMIT
        and reference.form_error_bound == rx
        and reference.phase_error_bound == rt
        and reference.source_certified
        and len(reference.recoveries) == 2
        and reference.initial_form_means == (Q(0), Q(0))
        and reference.supplied_phase_origins == (-_DELTA / 2, _DELTA / 2)
        and reference.relative_phase_gap == _DELTA
        and all(
            row.target_winding == winding
            and row.source_certified
            and row.recovery_certified
            for winding, row in zip((1, 2), reference.recoveries, strict=True)
        )
    )
    waiting_ok = (
        source_ok
        and reference.isolated_waiting_certified
        and reference.recovered_radius == _EPSILON
        and reference.waiting_radius == _ETA
    )
    geometries = tuple(_successor_geometry(base, kind) for kind in _COMPLETIONS)
    geometry_ok = (
        reference.geometry == base
        and base.support_certified
        and base.phase_alignment_certified
        and _GAMMA_RANGE[0]
        <= base.gamma_bounds.lo
        <= base.gamma_bounds.hi
        <= _GAMMA_RANGE[1]
        and all(
            row.support_certified
            and row.reference_stationary_certified
            and row.spectral_gap_certified
            for row in geometries
        )
    )
    pi = pi_interval()
    curvature = cos(4 * pi / 9 + 2 * _RADIUS)
    curvature_margin = curvature.lo - Q(1, 10)
    image_ok = waiting_ok and geometry_ok and regularizer_ok
    s1, s2 = I(Q(71, 9)) - _DELTA / (2 * pi), I(Q(73, 9)) - _DELTA / (2 * pi)
    nominal = -_DELTA / (2 * base.gamma_bounds * s1 * s2)
    magnitude = (
        _DELTA / (2 * _GAMMA_RANGE[1] * Q(33, 4) ** 2),
        _DELTA / (2 * _GAMMA_RANGE[0] * Q(31, 4) ** 2),
    )
    limit_error = 10**12 * _ETA
    strength_limit = (nominal.lo - limit_error, nominal.hi + limit_error)
    target_margins = (strength_limit[0] + Q(1, 32), -Q(1, 64) - strength_limit[1])
    form_separated = (
        image_ok
        and all(Q(31, 4) < value.lo <= value.hi < Q(33, 4) for value in (s1, s2))
        and magnitude[0] > Q(5, 256)
        and magnitude[1] < Q(7, 256)
        and limit_error < Q(1, 256)
        and min(target_margins) > 0
    )
    initial_registration = (_DELTA - 6 * _ETA, _DELTA + 6 * _ETA)
    completions = []
    for geometry in geometries:
        common = geometry.completion == "common_sense"
        phase_norm = (
            Q(9, 2) * _DELTA**2 + 54 * _DELTA * _ETA if common else Q(0)
        ) + 162 * _ETA**2
        initial_norm = phase_norm + 2 * _ETA**2
        energy = (
            Q(27, 2) * _DELTA**2 + 486 * _ETA if common else 216 * _ETA
        ) + 18 * _ETA**2
        barrier = geometry.quotient_coercivity_lower_bound * _RADIUS**2
        radius_margin, storage_margin = _RADIUS**2 - phase_norm, barrier - energy
        quotient_radius_margin = _RADIUS**2 - initial_norm
        retained = (
            image_ok
            and curvature_margin > 0
            and radius_margin > 0
            and quotient_radius_margin > 0
            and storage_margin > 0
        )
        absent = initial_registration[0] > 7 * _DELTA / 8
        completions.append(
            _CouplingSuccessorCompletion(
                geometry=geometry,
                initial_phase_norm_squared_upper_bound=phase_norm,
                initial_quotient_norm_squared_upper_bound=initial_norm,
                initial_phase_radius_margin=radius_margin,
                initial_quotient_radius_margin=quotient_radius_margin,
                initial_energy_upper_bound=energy,
                storage_barrier=barrier,
                storage_margin=storage_margin,
                initial_registration_bounds=initial_registration,
                registration_limit_bounds=(
                    (Q(0), Q(0)) if common else initial_registration
                ),
                constituent_identity_certified=retained,
                asymptotic_convergence_certified=retained,
                joint_registration_certified=retained and absent and common,
                disconnected_registration_certified=retained and absent and not common,
                degree_form_limit_bounds=((Q(0), Q(0)),) * len(geometry.components),
                strength_form_limit_bounds=(strength_limit,) if common else None,
            )
        )
    constituents = all(row.constituent_identity_certified for row in completions)
    connected = completions[0].joint_registration_certified
    disconnected = completions[1].disconnected_registration_certified
    unresolved = tuple(
        label
        for passed, label in (
            (source_ok, "original_sources_unavailable"),
            (waiting_ok, "isolated_waiting_unavailable"),
            (geometry_ok, "successor_geometry_unavailable"),
            (regularizer_ok, "proposal_similarity_regularizer_unavailable"),
            (constituents, "full_image_quotient_traps_unavailable"),
            (
                connected and disconnected,
                "asymptotic_registration_contrast_unavailable",
            ),
            (form_separated, "normalization_common_form_separation_unavailable"),
        )
        if not passed
    )
    return _SineCouplingSuccessor(
        form_error_bound=rx,
        phase_error_bound=rt,
        precontact_reference=reference,
        completion_reports=tuple(completions),
        source_certified=source_ok,
        isolated_waiting_certified=waiting_ok,
        geometry_certified=geometry_ok,
        conditional_image_certified=image_ok,
        form_similarity_regularizer=regularizer,
        similarity_regularizer_margin=regularizer_margin,
        proposal_regularizer_certified=regularizer_ok,
        phase_curvature_bounds=curvature,
        phase_curvature_margin=curvature_margin,
        nominal_strength_bounds=(s1, s2),
        nominal_common_strength_form_limit_bounds=nominal,
        nominal_form_magnitude_bounds=magnitude,
        common_form_limit_error_upper_bound=limit_error,
        common_strength_form_limit_bounds=strength_limit,
        common_form_target_margins=target_margins,
        constituent_identity_certified=constituents,
        connected_joint_registration_certified=connected,
        disconnected_nonregistration_certified=disconnected,
        normalization_common_form_separation_certified=form_separated and constituents,
        certified=not unresolved,
        status="certified" if not unresolved else "unavailable",
        unresolved_conditions=unresolved,
    )
