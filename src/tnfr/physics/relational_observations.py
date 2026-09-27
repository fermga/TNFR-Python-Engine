"""Read-only regional observations of the selected relational exchange model.

An explicit reference supplies real phase lifts, not a discovered equilibrium.
The full centered form and phase-error vectors retain information discarded by
regional means. Shared winding and transport observations add no evolution,
identity certificate, reduced closure or automatic selection policy.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import networkx as nx

from .._exact_time import finite_represented_real
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..dynamics.relational import (
    RelationalExchangeField,
    RelationalExchangeModel,
    evaluate_relational_exchange,
)
from .support_transport import (
    RegionalSupportBalance,
    RegionalSupportCut,
    observe_regional_support_balance,
    observe_regional_support_cut,
    observe_support_transport,
)
from .winding_certificates import WindingCertificate, certify_phase_winding

__all__ = (
    "RegionalRelationalWork",
    "RegionalExchangeBalance",
    "RegionalPhaseResponse",
    "RegionalPatternObservation",
    "RelationalPatternObservation",
    "observe_relational_pattern",
)


@dataclass(frozen=True)
class RegionalRelationalWork:
    """Sums of the captured nodal gradient-work contributions in one region.

    Positive exchange transfers phase storage toward form storage. These are
    contributions to the global storage derivative, not derivatives of an
    independently defined regional energy. Overlapping regions double count
    shared nodes; only a partition reproduces the global sums.
    """

    dissipation: Fraction
    exchange: Fraction
    form_work: Fraction
    phase_work: Fraction
    form_residual: Fraction
    phase_residual: Fraction
    balance_residual: Fraction


@dataclass(frozen=True)
class RegionalExchangeBalance:
    """Two actual weighted rates using one outward form cut on full support.

    Form weights are full-graph degree/capacity; phase weights are the captured
    phase metric/capacity. The latter rate is not the derivative of a weighted
    phase total: its metric varies with phase. Exact arithmetic on represented
    rates retains phase and pressure/rounding defects rather than setting them
    to their ideal zero values. A zero capacity inside the region makes the
    divided rates unavailable, while the cut and undivided model terms remain
    defined. No evolution, reduced closure or provenance seal is supplied.
    """

    cut: RegionalSupportCut
    form_weighted_rate: Fraction | None
    form_boundary_rate: Fraction
    form_source_rate: Fraction
    form_pressure_defect_rate: Fraction
    form_rounding_defect_rate: Fraction | None
    form_identity_residual: Fraction | None
    phase_weighted_rate: Fraction | None
    phase_boundary_rate: Fraction
    phase_rate_residual: Fraction | None
    weighted_rate_unavailable_reason: str | None


@dataclass(frozen=True)
class RegionalPhaseResponse:
    """Unweighted phase-rate balance from the captured phase mobility and form.

    Mobility is capacity divided by the positive phase metric. Population
    covariance retains its correlation with the exact form gradient; it need
    not vanish when the outward cut vanishes. The squared covariance-rate
    bound is Cauchy--Schwarz, not a calibrated threshold. Actual model rates
    retain their rounding residual. This instantaneous identity supplies no
    autonomous regional closure, measured derivative or monotone clock.
    """

    mean_mobility: Fraction
    mean_form_gradient: Fraction
    mobility_variance: Fraction
    form_gradient_variance: Fraction
    mobility_gradient_covariance: Fraction
    mean_mobility_boundary_rate: Fraction
    covariance_rate: Fraction
    covariance_rate_squared_bound: Fraction
    model_total_rate: Fraction
    rounding_residual: Fraction
    total_rate: Fraction
    mean_rate: Fraction
    identity_residual: Fraction


@dataclass(frozen=True)
class RegionalPatternObservation:
    """One supplied region's complete centered coordinates and separate offsets.

    Means are arithmetic means, whereas ``transport.mean`` uses the shared
    degree/capacity metric. The two squared norms have their respective form
    and phase units; no combined metric or recovery threshold is introduced.
    """

    nodes: tuple[Any, ...]
    form_mean: Fraction
    phase_error_mean: Fraction
    centered_form: tuple[Fraction, ...]
    centered_phase_error: tuple[Fraction, ...]
    form_norm_squared: Fraction
    phase_norm_squared: Fraction
    transport: RegionalSupportBalance | None
    transport_unavailable_reason: str | None
    work: RegionalRelationalWork | None = None
    boundary: RegionalExchangeBalance | None = None
    phase_response: RegionalPhaseResponse | None = None


@dataclass(frozen=True)
class RelationalPatternObservation:
    """Detached fresh field and observations in caller-supplied regional frames.

    ``reference_phase`` follows ``field.nodes`` and contains materialized real
    lifts. Subtracting them does not wrap angles or infer a common semicircle.
    A common change of phase offset disappears on centering; independently
    changing node representatives by full turns changes the supplied frame.
    """

    field: RelationalExchangeField
    reference_phase: tuple[float, ...]
    regions: tuple[RegionalPatternObservation, ...]
    winding: tuple[WindingCertificate, ...]
    scope: tuple[str, ...] = (
        "one_fresh_detached_relational_field_without_live_graph_writes",
        "supplied_real_reference_lifts_and_ordered_possibly_overlapping_regions",
        "exact_rational_centering_of_materialized_form_and_phase_coordinates",
        "separate_form_and_phase_units_without_a_combined_norm_or_threshold",
        "regional_transport_uses_full_support_and_independent_phase_source",
        "nodal_and_regional_work_retains_exact_represented_gradients_and_rate_defects",
        "paired_weighted_rates_share_one_cut_not_a_conserved_phase_total",
        "unweighted_phase_response_retains_mobility_form_covariance_and_rounding",
        "declared_cycle_winding_is_snapshot_geometry_not_temporal_identity",
        "no_reference_equilibrium_recovery_formation_closed_reduction_or_selection_certificate",
    )


def _ordered(value, label):
    if isinstance(value, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered iterable")
    try:
        return tuple(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered iterable") from exc


def _regions(nodes, regions):
    raw = _ordered(regions, "regions")
    if not raw:
        raise ValueError("regions must contain at least one region")
    lookup = {node: index for index, node in enumerate(nodes)}
    result = []
    for region in raw:
        row = _ordered(region, "region")
        if not row:
            raise ValueError("each region must be nonempty")
        try:
            indices = tuple(lookup[node] for node in row)
        except (TypeError, KeyError) as exc:
            raise ValueError("region nodes must belong to the full support") from exc
        if len(set(indices)) != len(indices):
            raise ValueError("region nodes must be distinct")
        result.append(indices)
    return tuple(result)


def _detached_graph(field):
    """Rebuild only the captured state consumed by the existing observers."""
    graph = nx.Graph()
    for i, node in enumerate(field.nodes):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: field.epi[i],
                ALIAS_VF[0]: field.capacity[i],
                ALIAS_THETA[0]: field.phase[i],
                ALIAS_DNFR[0]: field.pressure[i],
            },
        )
    graph.add_edges_from((a, b, {"weight": 1.0}) for a, b in field.edges)
    return graph


def _regional_work(field, indices):
    work = field.work
    if work is None:
        raise RuntimeError("fresh relational field must contain work accounting")
    return RegionalRelationalWork(
        **{
            name: sum((getattr(work, name)[i] for i in indices), Fraction(0))
            for name in RegionalRelationalWork.__dataclass_fields__
        }
    )


def _regional_boundary(field, cut, degrees):
    indices = cut.region_indices
    current = cut.outward_cut_current
    e = Fraction(field.model.epi_weight)
    w = Fraction(field.model.phase_weight)
    beta = Fraction(field.model.storage_scale)
    source = sum(
        (w * degrees[i] * Fraction(field.phase_source[i]) for i in indices),
        Fraction(0),
    )
    pressure_defect = sum(
        (degrees[i] * field.pressure_split_residual[i] for i in indices),
        Fraction(0),
    )
    reason = (
        "zero_capacity_in_region"
        if any(field.capacity[i] == 0.0 for i in indices)
        else None
    )
    form_rate = phase_rate = rounding = form_residual = phase_residual = None
    if reason is None:
        form_rate = sum(
            (
                degrees[i] * Fraction(field.form_rate[i]) / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        rounding = sum(
            (
                degrees[i]
                * field.nodal_rate_rounding_defect[i]
                / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        phase_rate = sum(
            (
                Fraction(field.phase_metric[i])
                * Fraction(field.phase_rate[i])
                / Fraction(field.capacity[i])
                for i in indices
            ),
            Fraction(0),
        )
        form_residual = form_rate + e * current - source - pressure_defect - rounding
        phase_residual = phase_rate - w * current / beta
        if form_residual:
            raise RuntimeError("exact represented regional form-rate identity failed")
    return RegionalExchangeBalance(
        cut=cut,
        form_weighted_rate=form_rate,
        form_boundary_rate=-e * current,
        form_source_rate=source,
        form_pressure_defect_rate=pressure_defect,
        form_rounding_defect_rate=rounding,
        form_identity_residual=form_residual,
        phase_weighted_rate=phase_rate,
        phase_boundary_rate=w * current / beta,
        phase_rate_residual=phase_residual,
        weighted_rate_unavailable_reason=reason,
    )


def _regional_phase_response(field, cut):
    if (
        field.work is None
        or field.phase_mobility is None
        or field.phase_rate_rounding_defect is None
    ):
        raise RuntimeError(
            "fresh relational field must contain exact phase-rate evidence"
        )
    indices = cut.region_indices
    n = len(indices)
    mobility = tuple(field.phase_mobility[i] for i in indices)
    gradient = tuple(field.work.form_gradient[i] for i in indices)
    mean_mobility = sum(mobility, Fraction(0)) / n
    mean_gradient = sum(gradient, Fraction(0)) / n
    centered_mobility = tuple(a - mean_mobility for a in mobility)
    centered_gradient = tuple(q - mean_gradient for q in gradient)
    mobility_variance = sum((a * a for a in centered_mobility), Fraction(0)) / n
    gradient_variance = sum((q * q for q in centered_gradient), Fraction(0)) / n
    covariance = (
        sum((a * q for a, q in zip(centered_mobility, centered_gradient)), Fraction(0))
        / n
    )
    k = Fraction(field.model.phase_weight) / Fraction(field.model.storage_scale)
    boundary_rate = k * mean_mobility * cut.outward_cut_current
    covariance_rate = k * n * covariance
    squared_bound = k * k * n * n * mobility_variance * gradient_variance
    model_total = k * sum((a * q for a, q in zip(mobility, gradient)), Fraction(0))
    rounding = sum((field.phase_rate_rounding_defect[i] for i in indices), Fraction(0))
    total = sum((Fraction(field.phase_rate[i]) for i in indices), Fraction(0))
    residual = total - boundary_rate - covariance_rate - rounding
    if residual or model_total != boundary_rate + covariance_rate:
        raise RuntimeError("exact represented regional phase-rate identity failed")
    if covariance_rate * covariance_rate > squared_bound:
        raise RuntimeError("exact regional covariance-rate bound failed")
    return RegionalPhaseResponse(
        mean_mobility=mean_mobility,
        mean_form_gradient=mean_gradient,
        mobility_variance=mobility_variance,
        form_gradient_variance=gradient_variance,
        mobility_gradient_covariance=covariance,
        mean_mobility_boundary_rate=boundary_rate,
        covariance_rate=covariance_rate,
        covariance_rate_squared_bound=squared_bound,
        model_total_rate=model_total,
        rounding_residual=rounding,
        total_rate=total,
        mean_rate=total / n,
        identity_residual=residual,
    )


def observe_relational_pattern(
    graph,
    *,
    model: RelationalExchangeModel,
    reference_phase: Mapping,
    regions,
    cycles=(),
) -> RelationalPatternObservation:
    """Observe supplied regions using exactly one fresh relational evaluation.

    Reference phases must map the exact full node support to finite represented
    real lifts. Regions are ordered, nonempty selections without repeated nodes;
    they may overlap and need not partition support. No reference equilibrium,
    lift compatibility over time or automatic regional discovery is inferred.

    Transport is unavailable if any full-support capacity is zero, EPI weight
    is zero, or the region covers all support (in that order of precedence).
    Those are limitations of the existing regional transport contract, not
    failures of the retained geometry. Its forcing is independently supplied
    as ``w*phase_source``; native pressure-split defects are retained.

    Work and paired boundary balances also cover full-support regions and
    zero EPI weight. Their actual divided rates require positive capacity only
    inside the selected region. These more general read-outs do not broaden
    the older transport metric's admission. Each actual report populates
    ``work``, ``boundary`` and ``phase_response``; their optional defaults only
    preserve older manually constructed records. The unweighted phase response
    splits the actual phase-rate sum into mean-mobility cut, mobility/form
    covariance and exact rounding residual. It admits zero capacity and retains
    an exact squared covariance-rate bound without inventing a tolerance.

    Optional ordered cycles delegate geometric availability to the existing
    winding owner. No observation advances a graph or certifies a trajectory.
    """
    field = evaluate_relational_exchange(graph, model=model)
    if not isinstance(reference_phase, Mapping):
        raise TypeError("reference_phase must map the exact full node support")
    if set(reference_phase) != set(field.nodes):
        raise ValueError("reference_phase must match the exact full node support")
    reference = tuple(
        finite_represented_real(reference_phase[node], "reference phase")[0]
        for node in field.nodes
    )
    selections = _regions(field.nodes, regions)
    ordered_cycles = tuple(
        _ordered(cycle, "cycle") for cycle in _ordered(cycles, "cycles")
    )
    detached = _detached_graph(field)
    unavailable = (
        "zero_capacity_in_full_support"
        if any(capacity == 0.0 for capacity in field.capacity)
        else ("zero_epi_weight" if model.epi_weight == 0.0 else None)
    )
    source = observe_support_transport(detached)
    degrees = tuple(len(row) for row in source.support_neighbors)
    forcing = tuple(
        Fraction(model.phase_weight) * Fraction(value) for value in field.phase_source
    )
    observed = []
    for indices in selections:
        nodes = tuple(field.nodes[i] for i in indices)
        form = tuple(Fraction(field.epi[i]) for i in indices)
        error = tuple(
            Fraction(field.phase[i]) - Fraction(reference[i]) for i in indices
        )
        form_mean, phase_mean = sum(form) / len(form), sum(error) / len(error)
        centered_form = tuple(value - form_mean for value in form)
        centered_phase = tuple(value - phase_mean for value in error)
        reason = unavailable or (
            "full_support_region" if len(indices) == len(field.nodes) else None
        )
        transport = (
            observe_regional_support_balance(
                source, nodes, epi_weight=Fraction(model.epi_weight), forcing=forcing
            )
            if reason is None
            else None
        )
        cut = (
            transport.cut
            if transport is not None
            else observe_regional_support_cut(source, nodes)
        )
        observed.append(
            RegionalPatternObservation(
                nodes=nodes,
                form_mean=form_mean,
                phase_error_mean=phase_mean,
                centered_form=centered_form,
                centered_phase_error=centered_phase,
                form_norm_squared=sum(
                    (value**2 for value in centered_form), Fraction(0)
                ),
                phase_norm_squared=sum(
                    (value**2 for value in centered_phase), Fraction(0)
                ),
                transport=transport,
                transport_unavailable_reason=reason,
                work=_regional_work(field, indices),
                boundary=_regional_boundary(field, cut, degrees),
                phase_response=_regional_phase_response(field, cut),
            )
        )
    return RelationalPatternObservation(
        field=field,
        reference_phase=reference,
        regions=tuple(observed),
        winding=tuple(
            certify_phase_winding(detached, cycle) for cycle in ordered_cycles
        ),
    )
