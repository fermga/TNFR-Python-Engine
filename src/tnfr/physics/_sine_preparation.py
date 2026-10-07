"""Shared exact preparation for conditional full-law sine estimates.

This owner revalidates primitive source data and computes the common weighted
geometry, coefficient bounds and initial norm. It neither evolves the source
nor decides a theorem's horizon, phase preparation or endpoint admission.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real
from ..dynamics.relational import RelationalExchangeModel
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ._sine_admission import _sector_source_admission, _sine_model_coefficients
from .phase_cycle_geometry import PhaseCycleGeometry, _rebuild
from .relational_observations import _ordered
from .relational_sine_comparison import SineExchangeComparison
from .relational_sine_pattern import SineRelativePattern
from .structural_diffusion import _exact_flow_gap_from_rationals


@dataclass(frozen=True)
class _SineDomain:
    geometry: PhaseCycleGeometry
    model: RelationalExchangeModel
    capacity: tuple[Q, ...]
    degrees: tuple[int, ...]
    e: Q
    w: Q
    beta: Q
    weights: tuple[Q, ...]
    mobility: tuple[Q, ...]
    normalized_weights: tuple[Q, ...]
    gap: Q
    pi: I
    alpha: I
    inverse_alpha: I
    eta: I
    forcing: I


def _sine_domain(geometry, reference_model, capacity) -> _SineDomain:
    """Admit held coefficients and weighted geometry without a nodal state.

    The support declares unit sine coupling. Rebuilding its geometry retains
    the shared 32-node/50-edge budget and rejects forged derived fields. Model
    coefficients are their authoritative stored values, without a second
    normalization. No form, phase, common origin or representative is supplied.
    """
    geometry = _rebuild(geometry)
    e, w, beta = _sine_model_coefficients(reference_model, positive_loss=True)
    size = len(geometry.nodes)
    raw_capacity = _ordered(capacity, "capacity", limit=size + 1)
    if len(raw_capacity) != size:
        raise ValueError("capacity must contain one value per support node")
    capacity = tuple(
        exact_or_represented_real(value, f"capacity[{i}]")
        for i, value in enumerate(raw_capacity)
    )
    if any(value <= 0 for value in capacity):
        raise ValueError(
            "analytic sine domain requires strictly positive held capacities"
        )
    adjacency = [set() for _ in geometry.nodes]
    for left, right in geometry.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    degrees = tuple(map(len, adjacency))
    weights = tuple(Q(degree) / nu for degree, nu in zip(degrees, capacity))
    mobility = tuple(Q(1) / value for value in weights)
    total_weight = sum(weights, Q(0))
    normalized_weights = tuple(weight / total_weight for weight in weights)
    laplacian = tuple(
        tuple(
            Q(len(adjacency[i]) if i == j else -int(j in adjacency[i]))
            for j in range(size)
        )
        for i in range(size)
    )
    gap, mean_preserved, consensus_preserved, uniform_fixed = (
        _exact_flow_gap_from_rationals(laplacian, mobility, weights)
    )
    if gap <= 0 or not all((mean_preserved, consensus_preserved, uniform_fixed)):
        raise ArithmeticError(
            "the complete weighted diffusion quotient failed admission"
        )
    pi = pi_interval()
    alpha = (w / (beta * e)) / pi
    inverse_alpha = (beta * e / w) * pi
    eta = (w**2 / (beta * e**2)) / pi**2
    forcing = sqrt(I(sum((nu * degree for nu, degree in zip(capacity, degrees)), Q(0))))
    return _SineDomain(
        geometry=geometry,
        model=reference_model,
        capacity=capacity,
        degrees=degrees,
        e=e,
        w=w,
        beta=beta,
        weights=weights,
        mobility=mobility,
        normalized_weights=normalized_weights,
        gap=gap,
        pi=pi,
        alpha=alpha,
        inverse_alpha=inverse_alpha,
        eta=eta,
        forcing=forcing,
    )


@dataclass(frozen=True)
class _SinePreparation:
    """Shared prepared rows; report consumers retain their admitted source.

    ``admitted`` is None only for an internal producer using freshly admitted
    exact rows and domain directly. It is never absent on the report-consuming
    ``_sine_preparation`` path, and it supplies no arithmetic premises.
    """

    admitted: SineExchangeComparison | SineRelativePattern | None
    geometry: PhaseCycleGeometry
    uncertain: bool
    form: tuple[Q, ...]
    phase: tuple[Q, ...]
    form_errors: tuple[Q, ...]
    phase_errors: tuple[Q, ...]
    model: RelationalExchangeModel
    e: Q
    w: Q
    beta: Q
    weights: tuple[Q, ...]
    mobility: tuple[Q, ...]
    normalized_weights: tuple[Q, ...]
    mean: Q
    phase_mean: Q
    centered: tuple[Q, ...]
    centered_form_errors: tuple[Q, ...]
    centered_phase_errors: tuple[Q, ...]
    gap: Q
    pi: I
    alpha: I
    inverse_alpha: I
    eta: I
    nominal_initial_norm: I
    initial_error_norm: Q
    initial_norm: I
    forcing: I
    scaled_nominal: tuple[I, ...]
    scaled_initial: tuple[I, ...]
    initial_form_storage: Q | None
    initial_form_storage_bounds: I
    initial_phase_storage_bounds: I
    initial_storage_bounds: I
    initial_phase_gaps: tuple[I, ...]


def _sine_preparation(source) -> _SinePreparation:
    """Admit a complete exact source or its original relative residual set.

    Exact and nominal phase lifts may be nonuniform. Each consuming theorem
    owns any stronger phase premise. Means here are nominal for relative sources;
    they do not recover their unknown common origins or identify one mean leaf
    for the entire residual family. The shared 32-node/50-edge geometry budget
    and strictly positive held capacities, form loss and exchange apply.
    """
    if not isinstance(source, (SineExchangeComparison, SineRelativePattern)):
        raise TypeError("a SineExchangeComparison or SineRelativePattern is required")
    admitted, geometry = _sector_source_admission(source)
    uncertain = isinstance(admitted, SineRelativePattern)
    if uncertain:
        form, phase = admitted.nominal_form, admitted.nominal_phase
        form_errors, phase_errors = (
            admitted.form_error_bounds,
            admitted.phase_error_bounds,
        )
    else:
        form, phase = admitted.epi, admitted.phase
        form_errors = phase_errors = (Q(0),) * len(admitted.nodes)
    domain = _sine_domain(geometry, admitted.reference_model, admitted.capacity)
    return _sine_preparation_from_rows(
        domain,
        form=form,
        phase=phase,
        form_errors=form_errors,
        phase_errors=phase_errors,
        admitted=admitted,
        uncertain=uncertain,
    )


def _sine_preparation_from_rows(
    domain: _SineDomain,
    *,
    form: tuple[Q, ...],
    phase: tuple[Q, ...],
    form_errors: tuple[Q, ...],
    phase_errors: tuple[Q, ...],
    admitted: SineExchangeComparison | SineRelativePattern | None = None,
    uncertain: bool = True,
) -> _SinePreparation:
    """Compute preparation data from one freshly admitted domain and rows.

    Private callers own prior scalar, row-size and error-sign admission and
    must pass normalized exact rows in the domain's node order. ``domain``
    comes from a fresh ``_sine_domain`` call; a cached domain or incoming
    report cannot substitute for its model/support/capacity admission.

    The report adapter above preserves its normalized source association and
    exact-versus-relative metadata. A fixed internal producer may omit that
    association after directly admitting its primitive rows, sharing one
    domain within an invocation without building discarded observation fields.
    The association and ``uncertain`` metadata do not alter the arithmetic.
    """
    geometry, model = domain.geometry, domain.model
    e, w, beta = domain.e, domain.w, domain.beta
    weights, mobility = domain.weights, domain.mobility
    normalized_weights = domain.normalized_weights
    gap, pi = domain.gap, domain.pi
    alpha, inverse_alpha, eta = domain.alpha, domain.inverse_alpha, domain.eta
    forcing = domain.forcing
    mean = sum(
        (weight * value for weight, value in zip(normalized_weights, form)), Q(0)
    )
    phase_mean = sum(
        (weight * value for weight, value in zip(normalized_weights, phase)), Q(0)
    )
    centered = tuple(value - mean for value in form)

    def centered_errors(errors):
        average = sum(
            (weight * value for weight, value in zip(normalized_weights, errors)), Q(0)
        )
        return tuple(
            radius + average - 2 * weight * radius
            for radius, weight in zip(errors, normalized_weights)
        )

    centered_form_errors = centered_errors(form_errors)
    centered_phase_errors = centered_errors(phase_errors)
    nominal_initial_norm = alpha * sqrt(
        I(sum((weight * value**2 for weight, value in zip(weights, centered)), Q(0)))
    )
    initial_error_norm = (
        alpha
        * sqrt(
            I(
                sum(
                    (
                        weight * radius**2
                        for weight, radius in zip(weights, form_errors)
                    ),
                    Q(0),
                )
            )
        )
    ).hi
    # Weighted centering is an orthogonal projection. Its residual norm is
    # at most the uncentered weighted radius norm; no common offset is added.
    initial_norm = (
        nominal_initial_norm
        if not any(form_errors)
        else I(
            max(Q(0), nominal_initial_norm.lo - initial_error_norm),
            nominal_initial_norm.hi + initial_error_norm,
        )
    )
    scaled_nominal = tuple(alpha * value for value in centered)
    scaled_initial = tuple(
        alpha * I(value - radius, value + radius)
        for value, radius in zip(centered, centered_form_errors)
    )
    initial_form_storage = initial_phase_storage = I(0)
    initial_phase_gaps = []
    for left, right in geometry.edges:
        form_error = form_errors[left] + form_errors[right]
        phase_error = phase_errors[left] + phase_errors[right]
        form_gap = form[right] - form[left]
        phase_gap = phase[right] - phase[left]
        initial_form_storage += I(form_gap - form_error, form_gap + form_error) ** 2 / 2
        phase_interval = I(phase_gap - phase_error, phase_gap + phase_error)
        initial_phase_gaps.append(phase_interval)
        initial_phase_storage += 1 - cos(phase_interval)
    exact_form_storage = sum(
        ((form[right] - form[left]) ** 2 / 2 for left, right in geometry.edges),
        Q(0),
    )
    return _SinePreparation(
        admitted=admitted,
        geometry=geometry,
        uncertain=uncertain,
        form=form,
        phase=phase,
        form_errors=form_errors,
        phase_errors=phase_errors,
        model=model,
        e=e,
        w=w,
        beta=beta,
        weights=weights,
        mobility=mobility,
        normalized_weights=normalized_weights,
        mean=mean,
        phase_mean=phase_mean,
        centered=centered,
        centered_form_errors=centered_form_errors,
        centered_phase_errors=centered_phase_errors,
        gap=gap,
        pi=pi,
        alpha=alpha,
        inverse_alpha=inverse_alpha,
        eta=eta,
        nominal_initial_norm=nominal_initial_norm,
        initial_error_norm=initial_error_norm,
        initial_norm=initial_norm,
        forcing=forcing,
        scaled_nominal=scaled_nominal,
        scaled_initial=scaled_initial,
        initial_form_storage=None if any(form_errors) else exact_form_storage,
        initial_form_storage_bounds=initial_form_storage,
        initial_phase_storage_bounds=initial_phase_storage,
        initial_storage_bounds=initial_form_storage + beta * initial_phase_storage,
        initial_phase_gaps=tuple(initial_phase_gaps),
    )
