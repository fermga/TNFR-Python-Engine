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
from .reversible_eigenmode_reference import _negative_exp_bounds
from .structural_diffusion import _exact_flow_gap_from_rationals


def _prepared_form_costs(*, form, weighted_edges):
    """Return exact edge storage and its nodal gradient for admitted data.

    Callers supply finite rational forms and validated nonnegative weighted
    edges indexed in that form tuple. Costs retain contact edges; grouping,
    source-error allowances and preparation budgets belong to the consumer.
    No source or trajectory verdict is inferred by this algebraic kernel.
    """
    gradient = [Q(0)] * len(form)
    costs = []
    for a, b, weight in weighted_edges:
        difference = form[a] - form[b]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        costs.append(weight * difference**2 / 2)
    return tuple(costs), tuple(gradient)


def _prepared_heat_error_radii(
    *,
    initial_form_error,
    initial_phase_error,
    gamma_upper,
    phase_growth_upper,
    form_convolution_factor,
):
    """Return full form/phase error radii from an admitted heat-integral proof.

    Callers prove the complete reciprocal rows, mean-zero form difference,
    spectral/Lipschitz bounds and the whole-time growth factor. Identical
    prescribed additive form inputs cancel from the compared rows; estimates
    for a common input do not transfer to different input recipes. Identical
    additive jumps preserve the difference only when applied to both compared
    states. No event, source, clock or horizon is selected by this arithmetic.
    Exact rational errors remain exact even below the interval grid.
    """
    form, phase, gamma, growth, convolution = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("initial_form_error", initial_form_error),
            ("initial_phase_error", initial_phase_error),
            ("gamma_upper", gamma_upper),
            ("phase_growth_upper", phase_growth_upper),
            ("form_convolution_factor", form_convolution_factor),
        )
    )
    if min(form, phase, gamma, convolution) < 0 or growth < 1:
        raise ValueError("heat error bounds require nonnegative inputs and growth >= 1")
    phase_radius = (phase + gamma * form) * growth
    return form + gamma * convolution * phase_radius, phase_radius


def _prepared_degree_four_probe_remainder(
    *, gamma_upper, initial_form_upper_bound, phase_jump_upper_bound, elapsed_time
):
    """Bound a finite matched phase-probe increment for the quarter-rate law.

    The caller admits degree-four unit support, both reciprocal smooth-sine
    rows, held unit capacity, an unchanged initial form and an initial edge
    phase jump bounded by ``phase_jump_upper_bound``. Heat contraction and
    the global sine derivative retain background motion and probe feedback.
    The result bounds the coordinate maximum error from the initial slope;
    it neither selects a protocol nor certifies its state, work or identity.
    """
    gamma, initial, jump, time = (
        exact_or_represented_real(value, name)
        for name, value in (
            ("gamma_upper", gamma_upper),
            ("initial_form_upper_bound", initial_form_upper_bound),
            ("phase_jump_upper_bound", phase_jump_upper_bound),
            ("elapsed_time", elapsed_time),
        )
    )
    if min(gamma, initial, jump, time) < 0:
        raise ValueError("finite response bounds require nonnegative inputs")
    if 2 * gamma**2 * time**2 > Q(1, 2):
        raise ValueError("finite response comparison is outside its domain")
    return (
        2 * gamma * jump * time**2
        + 2 * gamma**2 * jump * (initial + gamma * time) * time**2
        + Q(4, 3) * gamma**3 * jump * time**3
    )


def _prepared_transit_radii(
    *,
    time,
    decay_upper,
    gap_lower,
    initial_norm_upper,
    feedback_upper,
    forcing_upper,
):
    """Bound scaled form and the integrated phase remainder of a prepared flow.

    Callers admit the complete positive-loss law, its weighted mean-free
    semigroup gap, nonnegative time and bounds before using this kernel.
    Arguments are exact rational bounds. The phase radius is measured from
    the actual initial scaled form plus initial phase; a nominal-source
    comparison must retain its initial form and phase errors separately.
    These are norm bounds for the actual coupled flow, not endpoint boxes
    or a capture verdict. No source, law or horizon is cached here.
    """
    scaled_form_radius = (
        decay_upper * initial_norm_upper
        + feedback_upper * forcing_upper * min(time, Q(1) / gap_lower)
    )
    phase_radius = scaled_form_radius + feedback_upper * time * forcing_upper
    return scaled_form_radius, phase_radius


def _prepared_velocity_defect(
    *,
    time,
    gap_lower,
    feedback_upper,
    lipschitz_upper,
    forcing_upper,
    initial_defect_upper,
):
    """Bound slow velocity error after an admitted fast relaxation time.

    Callers supply exact rational nonnegative bounds for the complete rows
    a'=-A*a+epsilon*F(theta), theta'=A*a on a preserved mean-zero leaf.
    The generator gap and global derivative bound must satisfy g>epsilon*K.
    The resulting decreasing envelope bounds theta_s-F(theta), s=epsilon*t,
    for every later time. It is not a phase trajectory or capture bound.
    """
    if (
        min(time, feedback_upper, lipschitz_upper, forcing_upper, initial_defect_upper)
        < 0
    ):
        raise ValueError("velocity defect requires nonnegative bounds and time")
    damping = gap_lower - feedback_upper * lipschitz_upper
    if damping <= 0:
        raise ValueError("velocity defect requires gap greater than feedback stiffness")
    decay = I(*_negative_exp_bounds(damping * time))
    floor = feedback_upper * lipschitz_upper * forcing_upper / damping
    return damping, decay, floor, decay.hi * initial_defect_upper + floor


def _prepared_duhamel_bounds(
    *,
    time,
    decay_upper,
    gap_lower,
    rate_upper,
    forcing_upper,
    initial_norm_upper,
    scaled_form_error_upper,
    phase_error_upper,
    feedback_upper,
):
    """Bound a prepared scaled flow around its nominal Poisson profile.

    Callers admit a self-adjoint mean-free generator with the supplied gap,
    upper rate and global forcing/Lipschitz bounds. All arguments are exact
    nonnegative bounds. The returned values enclose the weighted scaled-form
    remainder and phase distance from the nominal initial scaled form.
    This estimate retains the full transient and forcing history.
    """
    initial_error = scaled_form_error_upper + phase_error_upper
    history = feedback_upper * forcing_upper * (time + 1 / gap_lower)
    remainder = decay_upper * (
        initial_norm_upper
        + scaled_form_error_upper
        + feedback_upper * forcing_upper / gap_lower
    ) + feedback_upper * rate_upper * (
        initial_norm_upper * time * decay_upper + (initial_error + history) / gap_lower
    )
    phase_error = decay_upper * initial_norm_upper + initial_error + history
    return remainder, phase_error


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
