"""Fixed finite inputs for the localized full variational comparison.

LOCALIZED_RETENTION_MECHANISMS.md, equations (49) and (83), own this
canonical m=1/2, C1024 construction. The factory prepares immutable finite
coefficients; evaluating its closure encloses the prescribed input rows,
not a correction, nonlinear response, or maintenance certificate. Profile,
sine, and cosine truncation errors remain the separate analytical budgets
of that owner. Arithmetic here uses the shared outward 128-bit intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from dataclasses import dataclass, field
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._dyadic_linear import _dot_encoded_rows, _encode_admitted_row
from ..mathematics._rational_interval import (
    INTERVAL_METHOD,
    I,
    exp_neg_unit,
    pi_interval,
    sqrt,
)
from ._localized_mobile_spectral import _complex_multiply, _reduced_sine_cosine

_NODE_COUNT = 1024
_PROFILE_ORDER = 24
_AGM_STEPS = 8
_ZERO = I(0)
_ONE = I(1)
_SINE_COEFFICIENTS = tuple(
    I(Q((-1) ** j, factorial(2 * j + 1))) for j in range(8, 0, -1)
)
_COSINE_COEFFICIENTS = tuple(
    I(Q((-1) ** (j + 1), factorial(2 * j))) for j in range(8, 0, -1)
)


def _interval(value, label):
    """Re-admit interval endpoints or a primitive real before arithmetic."""
    if isinstance(value, I):
        return I(_exact(value.lo, f"{label} lower"), _exact(value.hi, f"{label} upper"))
    return I(_exact(value, label))


def _admit_vector(values, size, label):
    """Detach a finite indexed sequence; reject unordered or malformed rows."""
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered finite sequence")
    try:
        count = len(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered finite sequence") from exc
    if count != size:
        raise ValueError(f"{label} must have exactly {size} entries")
    try:
        raw = tuple(values[index] for index in range(size))
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(f"{label} must be an indexed finite sequence") from exc
    return tuple(
        _interval(value, f"{label}[{index}]") for index, value in enumerate(raw)
    )


def _agm_elliptic_parameter_half():
    """Enclose K(1/2) by eight monotone AGM steps, retaining interval widths."""
    arithmetic, geometric = _ONE, sqrt(I(Q(1, 2)))
    for _ in range(_AGM_STEPS):
        arithmetic, geometric = (arithmetic + geometric) / 2, sqrt(
            arithmetic * geometric
        )
    limit = I(geometric.lo, arithmetic.hi)
    return pi_interval() / (2 * limit)


@lru_cache(maxsize=1)
def _spatial_roots():
    """Cache only immutable fixed-cycle roots, never times or evaluated rows."""
    sine, cosine = _reduced_sine_cosine(2 * pi_interval() / _NODE_COUNT)
    spatial = [(_ONE, _ZERO)]
    for _ in range(1, _NODE_COUNT):
        spatial.append(_complex_multiply(spatial[-1], (cosine, sine)))
    diagonal = sqrt(I(2)) / 2
    carrier = [(_ONE, _ZERO)]
    for _ in range(1, 8):
        carrier.append(_complex_multiply(carrier[-1], (diagonal, diagonal)))
    return tuple(spatial), tuple(carrier)


def _prepare_profile_basis(spatial_roots, coefficients, spacing):
    """Separate spatial aliases from the original Fourier mode indices.

    Inputs are local normalized intervals. The encoded immutable rows act on
    [1, Re(z), Im(z), ..., Re(z**M), Im(z**M)]. Only spatial root indices
    wrap around the cycle; derivative weights and temporal powers retain n.
    """
    weights = tuple(
        (2 * coefficient, -2 * mode * spacing * coefficient)
        for mode, coefficient in enumerate(coefficients[1:], 1)
    )
    size = len(spatial_roots)
    basis = []
    for site in range(size):
        profile, derivative = [coefficients[0]], [_ZERO]
        for mode, (weight, derivative_weight) in enumerate(weights, 1):
            cosine, sine = spatial_roots[(site * mode) % size]
            profile.extend((weight * cosine, -weight * sine))
            derivative.extend((derivative_weight * sine, derivative_weight * cosine))
        basis.append((_encode_admitted_row(profile), _encode_admitted_row(derivative)))
    return tuple(basis)


@lru_cache(maxsize=1)
def _canonical_profile_basis(coefficients, spacing):
    """Cache one immutable fixed-profile table, never a time or response."""
    return _prepare_profile_basis(_spatial_roots()[0], coefficients, spacing)


def _temporal_profile_vector(root, order):
    """Encode the original temporal modes once for every spatial site."""
    values = [_ONE]
    power = root
    for mode in range(1, order + 1):
        values.extend(power)
        if mode < order:
            power = _complex_multiply(power, root)
    return _encode_admitted_row(values)


def _profile_from_basis(basis, temporal):
    return tuple(_dot_encoded_rows(row, temporal) for row in basis)


def _profile_at_phase(root, coefficients, spacing):
    """Evaluate a Fourier polynomial through the same separated arithmetic."""
    basis = _prepare_profile_basis(((_ONE, _ZERO),), coefficients, spacing)[0]
    return _profile_from_basis(
        basis, _temporal_profile_vector(root, len(coefficients) - 1)
    )


def _reconstruction_constants(velocity, frequency, correctors):
    minus_one, positive_three, negative_three = correctors
    return (
        velocity,
        frequency,
        minus_one,
        positive_three + negative_three,
        positive_three - negative_three,
    )


def _carrier_reconstruction_weights(carrier, constants):
    """Prepare four signed rows for one carrier class, including its cube."""
    velocity, frequency, minus_one, form_three, phase_three = constants
    cosine, sine = carrier
    cosine_three, sine_three = _complex_multiply(
        _complex_multiply(carrier, carrier), carrier
    )
    form_cube = minus_one * cosine + form_three * cosine_three
    phase_cube = -minus_one * sine + phase_three * sine_three
    return tuple(
        _encode_admitted_row(row)
        for row in (
            (cosine, form_cube, _ZERO, _ZERO),
            (sine, phase_cube, _ZERO, _ZERO),
            (
                -frequency * sine,
                -frequency * (minus_one * sine + 3 * form_three * sine_three),
                velocity * cosine,
                velocity * form_cube,
            ),
            (
                frequency * cosine,
                frequency * (-minus_one * cosine + 3 * phase_three * cosine_three),
                velocity * sine,
                velocity * phase_cube,
            ),
        )
    )


def _reconstruct_prepared_row(profile, spatial_derivative, weights):
    """Keep the exact pointwise finite-profile cube and its derivative."""
    square = profile**2
    values = _encode_admitted_row(
        (profile, square * profile, spatial_derivative, 3 * square * spatial_derivative)
    )
    return tuple(_dot_encoded_rows(row, values) for row in weights)


def _reconstruct_row(
    profile, spatial_derivative, carrier, velocity, frequency, correctors
):
    """Both real rows and analytic time derivatives of equation (49)."""
    weights = _carrier_reconstruction_weights(
        carrier, _reconstruction_constants(velocity, frequency, correctors)
    )
    return _reconstruct_prepared_row(profile, spatial_derivative, weights)


def _cycle_difference(values):
    return tuple(
        value - values[(index + 1) % len(values)] for index, value in enumerate(values)
    )


def _cycle_adjoint(values):
    return tuple(value - values[index - 1] for index, value in enumerate(values))


def _cycle_laplacian(values):
    """A=D^T D/2 on the complete cycle, including its constant mode."""
    return tuple(
        value - (values[index - 1] + values[(index + 1) % len(values)]) / 2
        for index, value in enumerate(values)
    )


def _sine_cosine_polynomials(gap):
    """Return sin(a)-a through degree17 and 1-cos(a) through degree16."""
    square, sine, cosine = gap**2, _ZERO, _ZERO
    for sine_coefficient, cosine_coefficient in zip(
        _SINE_COEFFICIENTS, _COSINE_COEFFICIENTS
    ):
        sine = sine * square + sine_coefficient
        cosine = cosine * square + cosine_coefficient
    return gap * square * sine, square * cosine


def _apply_variation(form, phase, coefficient, sigma):
    """Apply J_sigma+V_tilde to normalized interval tuples in O(N) work."""
    form_laplacian, phase_laplacian = _cycle_laplacian(form), _cycle_laplacian(phase)
    edge = tuple(
        value * gap for value, gap in zip(coefficient, _cycle_difference(phase))
    )
    feedback = _cycle_adjoint(edge)
    return (
        tuple(
            -sigma * u - eta + value / 2
            for u, eta, value in zip(form_laplacian, phase_laplacian, feedback)
        ),
        form_laplacian,
    )


@dataclass(frozen=True)
class _VariationalInputRows:
    time_bounds: I
    x: tuple[I, ...]
    theta: tuple[I, ...]
    dx_dt: tuple[I, ...]
    dtheta_dt: tuple[I, ...]
    phase_gap: tuple[I, ...]
    source_x: tuple[I, ...]
    source_theta: tuple[I, ...]
    cosine_coefficient: tuple[I, ...]
    action: Callable = field(repr=False, compare=False)


def _build_rows(time, x, theta, dx_dt, dtheta_dt, sigma):
    """Keep signed source cancellation and capture immutable action inputs."""
    gap = _cycle_difference(theta)
    polynomial_rows = tuple(_sine_cosine_polynomials(value) for value in gap)
    sine, cosine = tuple(zip(*polynomial_rows))
    nonlinear = _cycle_adjoint(sine)
    ax, atheta = _cycle_laplacian(x), _cycle_laplacian(theta)
    source_x = tuple(
        rate + sigma * form + phase + term / 2
        for rate, form, phase, term in zip(dx_dt, ax, atheta, nonlinear)
    )
    source_theta = tuple(rate - form for rate, form in zip(dtheta_dt, ax))

    def action(form, phase):
        admitted_form = _admit_vector(form, len(cosine), "form perturbation")
        admitted_phase = _admit_vector(phase, len(cosine), "phase perturbation")
        return _apply_variation(admitted_form, admitted_phase, cosine, sigma)

    return _VariationalInputRows(
        time, x, theta, dx_dt, dtheta_dt, gap, source_x, source_theta, cosine, action
    )


def _make_input_evaluator(
    basis, carrier_roots, spacing, velocity, frequency, correctors, sigma, time_domain
):
    """Capture locally prepared immutable inputs; admit every clock first."""
    basis = tuple(basis)
    carrier_roots = tuple(tuple(root) for root in carrier_roots)
    order = (len(basis[0][0]) - 1) // 2
    constants = _reconstruction_constants(velocity, frequency, correctors)
    translation_rate = spacing * velocity

    def evaluate(time):
        time = _interval(time, "variational input time")
        if time.lo < 0 or time.hi > time_domain.hi:
            raise ValueError("variational input time must lie in [0,T.hi]")
        sine, cosine = _reduced_sine_cosine(translation_rate * time)
        temporal = _temporal_profile_vector((cosine, sine), order)
        sine, cosine = _reduced_sine_cosine(frequency * time)
        rotation = cosine, sine
        weights = tuple(
            _carrier_reconstruction_weights(
                _complex_multiply(root, rotation), constants
            )
            for root in carrier_roots
        )
        rows = tuple(
            _reconstruct_prepared_row(
                *_profile_from_basis(site_basis, temporal), weights[site % len(weights)]
            )
            for site, site_basis in enumerate(basis)
        )
        x, theta, dx_dt, dtheta_dt = tuple(zip(*rows))
        return _build_rows(time, x, theta, dx_dt, dtheta_dt, sigma)

    return evaluate


@dataclass(frozen=True)
class _VariationalInputMetadata:
    node_count: int
    node_order: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    form_loss: Q
    exchange: Q
    sigma_bounds: I
    carrier_bounds: I
    dispersion_bounds: I
    nonlinear_bounds: I
    velocity_bounds: I
    elliptic_integral_bounds: I
    inverse_width_bounds: I
    peak_bounds: I
    mean_bounds: I
    nome_bounds: I
    slow_frequency_bounds: I
    frequency_bounds: I
    horizon_bounds: I
    evaluation_time_bounds: I
    profile_coefficients: tuple[I, ...]
    profile_cutoff: int = _PROFILE_ORDER
    sine_degree: int = 17
    cosine_degree: int = 16
    agm_steps: int = _AGM_STEPS
    interval_method: str = INTERVAL_METHOD
    method: str = "canonical_localized_variational_inputs_dyadic128_v2"
    temporal_profile_modes: int = _PROFILE_ORDER
    carrier_classes: int = 8
    complex_products_per_evaluation: int = _PROFILE_ORDER - 1 + 3 * 8
    profile_dot_products_per_evaluation: int = 2 * _NODE_COUNT
    reconstruction_dot_products_per_evaluation: int = 4 * _NODE_COUNT
    retained_profile_basis_rows: int = 2 * _NODE_COUNT
    retained_profile_basis_entries: int = 2 * _NODE_COUNT * (2 * _PROFILE_ORDER + 1)
    integers_per_encoded_basis_entry: int = 3
    retained_interval_rows: int = 8
    dense_matrix_entries: int = 0
    scope: tuple[str, ...] = (
        "fixed_C1024_unit_capacity_conductance_length_and_storage_scale",
        "tau_equals_exchange_times_original_time_over_pi_in_both_rows",
        "elliptic_parameter_one_half_carrier_pi_over_four_no_profile_search",
        "finite_profile_exact_cube_signed_two_row_residual_and_cosine_Jacobian",
        "numerical_time_rectangle_0_to_T_upper_includes_physical_phase_horizon",
        "evaluation_O_NM_action_O_N_fixed_basis_storage_O_NM_no_dense_matrix",
        "spatial_aliases_only_temporal_and_derivative_mode_indices_retained",
        "immutable_spatial_basis_and_Horner_coefficients_reused_no_time_cache",
        "analytical_truncation_error_is_separate_from_interval_arithmetic",
        "no_correction_integration_nonlinear_response_or_retention_verdict",
    )


def _prepare_localized_variational_inputs():
    """Prepare fixed finite inputs; the returned closure evaluates their rows.

    Times are exact/represented reals or shared intervals wholly inside
    [0,T.hi]. This numerical rectangle contains the uncertain physical T;
    evaluating its extra sliver does not extend a physical theorem's horizon.
    The closure captures its own immutable inputs, not returned report fields.
    """
    size, pi, root_two = _NODE_COUNT, pi_interval(), sqrt(I(2))
    eigenvalue, dispersion, velocity = 1 - root_two / 2, root_two / 4, root_two / 2
    nonlinear = eigenvalue**2 / 8
    sigma, spacing = pi / (size**2 - 1), 2 * pi / size
    elliptic = _agm_elliptic_parameter_half()
    width = 2 * elliptic / size
    scale = sqrt(2 * dispersion / nonlinear)
    peak, mean = width * scale, pi * scale / size
    nome = exp_neg_unit(pi / 4) ** 4
    slow = Q(3, 2) * dispersion * width**2
    frequency, horizon = eigenvalue - slow, 1 / slow
    time_domain = I(0, horizon.hi)
    coefficients = (mean,) + tuple(
        2 * mean * nome**mode / (1 + nome ** (2 * mode))
        for mode in range(1, _PROFILE_ORDER + 1)
    )
    correctors = (-eigenvalue / 16, I(Q(1, 96)), -(3 * root_two - 2) / 672)
    _, carrier_roots = _spatial_roots()
    basis = _canonical_profile_basis(coefficients, spacing)
    loss = Q(1, 2**20)
    metadata = _VariationalInputMetadata(
        size,
        tuple(range(size)),
        tuple((j, (j + 1) % size) for j in range(size)),
        loss,
        1 - loss,
        sigma,
        pi / 4,
        dispersion,
        nonlinear,
        velocity,
        elliptic,
        width,
        peak,
        mean,
        nome,
        slow,
        frequency,
        horizon,
        time_domain,
        coefficients,
    )

    evaluate = _make_input_evaluator(
        basis,
        carrier_roots,
        spacing,
        velocity,
        frequency,
        correctors,
        sigma,
        time_domain,
    )
    return metadata, evaluate
