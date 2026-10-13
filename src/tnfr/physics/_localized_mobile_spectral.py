"""Validated finite Fourier integrand for the prepared mobile cubic response.

The factory admits fresh primitive preparation and law data. Its opaque
closure evaluates the cubic Duhamel integrand, not a nonlinear trajectory.
Both real state channels use the actual positive-loss two-by-two tangent.
The proof and frozen protocol belong to LOCALIZED_MOBILE_INTERACTION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real as _exact
from ..mathematics._rational_interval import (
    INTERVAL_BITS,
    INTERVAL_METHOD,
    I,
    cos,
    exp_neg_unit,
    pi_interval,
    sin,
    sqrt,
)
from ._localized_identity_transport import _admit_cycle_law, _admit_cycle_window

_MAX_NODE_COUNT = 4096
_MAX_FEJER_ORDER = 32
_GRID = 1 << INTERVAL_BITS
_PRODUCT_GRID = _GRID**2


def _reduced_sine_cosine(angle):
    """Use exact quarter-turn identities before shared small-angle enclosures.

    The integer choice is merely a representation choice. Subtracting the
    outward pi interval retains every angle regardless of that choice and
    avoids repeatedly reducing a large rational argument in the series owner.
    No floating phase reduction or separate trigonometric series is used.
    """
    if angle.lo == angle.hi == 0:
        return I(0), I(1)
    half_pi = pi_interval() / 2
    quarter_turns = (angle.midpoint / half_pi.midpoint + Q(1, 2)) // 1
    reduced = angle - quarter_turns * half_pi
    sine, cosine = sin(reduced), cos(reduced)
    return (
        (sine, cosine),
        (cosine, -sine),
        (-sine, -cosine),
        (-cosine, sine),
    )[quarter_turns % 4]


def _complex_multiply(left, right):
    real, imag = left
    other_real, other_imag = right
    return real * other_real - imag * other_imag, real * other_imag + imag * other_real


def _conjugate(value):
    return value[0], -value[1]


def _signed_mode(mode, node_count):
    return (mode + node_count // 2) % node_count - node_count // 2


def _dyadic_cyclic_convolution(left, right, node_count):
    """Enclose an unnormalized complex cyclic polynomial product exactly.

    Inputs are internal dictionaries of shared I rectangles. Their dyadic
    endpoints become exact integers on the shared grid. Each product uses
    the four endpoint corners; integer sums are accumulated without any
    intermediate rounding. Shared I construction performs the sole outward
    projection of each resulting coefficient. DFT normalization is owned
    by the caller. This is a polynomial reduction, not another interval type.
    """

    def integers(value):
        return tuple(
            endpoint.numerator * _GRID // endpoint.denominator
            for component in value
            for endpoint in (component.lo, component.hi)
        )

    def product(a, b, c, d):
        corners = a * c, a * d, b * c, b * d
        return min(corners), max(corners)

    left_rows = tuple((mode, integers(value)) for mode, value in left.items())
    right_rows = tuple((mode, integers(value)) for mode, value in right.items())
    sums = {}
    for left_mode, (ar0, ar1, ai0, ai1) in left_rows:
        for right_mode, (br0, br1, bi0, bi1) in right_rows:
            rr0, rr1 = product(ar0, ar1, br0, br1)
            ii0, ii1 = product(ai0, ai1, bi0, bi1)
            ri0, ri1 = product(ar0, ar1, bi0, bi1)
            ir0, ir1 = product(ai0, ai1, br0, br1)
            mode = _signed_mode(left_mode + right_mode, node_count)
            bounds = sums.setdefault(mode, [0, 0, 0, 0])
            bounds[0] += rr0 - ii1
            bounds[1] += rr1 - ii0
            bounds[2] += ri0 + ir0
            bounds[3] += ri1 + ir1
    return {
        mode: (
            I(Q(real_lower, _PRODUCT_GRID), Q(real_upper, _PRODUCT_GRID)),
            I(Q(imag_lower, _PRODUCT_GRID), Q(imag_upper, _PRODUCT_GRID)),
        )
        for mode, (real_lower, real_upper, imag_lower, imag_upper) in sums.items()
    }


def _mode_propagator(eigenvalue, sigma, frequency_factor, time):
    """Return actual positive-loss U_k(time), for admitted internal intervals."""
    if time.lo == time.hi == 0:
        return I(1), I(0), I(0), I(1)
    angle = eigenvalue * frequency_factor * time
    decay = exp_neg_unit(sigma * eigenvalue * time / 2)
    sine, cosine = _reduced_sine_cosine(angle)
    scaled_sine = sine / frequency_factor
    loss_sine = sigma * scaled_sine / 2
    return (
        decay * (cosine - loss_sine),
        -decay * scaled_sine,
        decay * scaled_sine,
        decay * (cosine + loss_sine),
    )


@dataclass(frozen=True)
class _MobileSpectralMetadata:
    """Primitive source association and complete retained Fourier support.

    This metadata is not accepted as input to the evaluator factory. Fourier
    coordinates are unitary and keep the form and phase channels separate.
    Reflection makes the resulting integrand even in space, so each channel
    has real equal coefficients at opposite modes; no modes are truncated.
    """

    node_count: int
    fejer_order: int
    amplitude: Q
    horizon: int
    form_loss_coefficient: Q
    exchange_coefficient: Q
    storage_scale: Q
    initial_centers: tuple[int, int]
    node_order: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    phase_modes: tuple[int, ...]
    modes: tuple[int, ...]
    interval_method: str = INTERVAL_METHOD
    method: str = "actual_damped_Fourier_mixed_cubic_dyadic128_v1"
    coefficient_layout: tuple[str, ...] = (
        "form_real",
        "form_imaginary",
        "phase_real",
        "phase_imaginary",
    )
    scope: tuple[str, ...] = (
        "fresh_primitive_complete_law_and_exact_nominal_Fejer_preparation",
        "unitary_DFT_with_all_cubic_output_modes_retained",
        "both_real_channels_use_the_actual_positive_loss_tangent",
        "source_B_is_exact_spatial_reflection_of_source_A",
        "reflection_even_integrand_has_real_equal_opposite_mode_coefficients",
        "time_intervals_enclose_elementary_evaluation_and_source_materialization",
        "integrand_only_no_quadrature_nonlinear_trajectory_or_response_verdict",
    )


def _prepare_mobile_spectral_kernel(
    *, reference_model, node_count, fejer_order, amplitude, horizon
):
    """Build immutable metadata and an opaque admitted-time evaluation closure.

    In addition to the prepared-cycle domain, this finite instrument requires
    N<=4096, Q<=32, even T, separated cubic bands 3(Q-1)<N/4, and sigma*T<=1.
    The latter keeps every decay argument in the shared unit exponential
    domain. These are numerical work bounds, not new physical restrictions.

    The returned callable takes exact/represented time or shared I wholly
    inside [0,T]. Its output follows metadata.modes, with four I coordinates
    per mode: form real/imaginary and phase real/imaginary. It neither accepts
    cached preparation reports nor infers an unobserved primitive state.
    """
    _admit_cycle_window(node_count, fejer_order, horizon)
    if horizon % 2:
        raise ValueError("mobile spectral horizon must be even")
    if node_count > _MAX_NODE_COUNT or fejer_order > _MAX_FEJER_ORDER:
        raise ValueError("mobile spectral work limit requires N<=4096 and Q<=32")
    if 12 * (fejer_order - 1) >= node_count:
        raise ValueError("mobile spectral instrument requires 3(Q-1)<N/4")
    amplitude = _exact(amplitude, "amplitude")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    loss, exchange, beta = _admit_cycle_law(reference_model)
    pi = pi_interval()
    sigma = I(pi.lo * loss / exchange, pi.hi * loss / exchange)
    if sigma.hi * horizon > 1:
        raise ValueError("mobile spectral exponential domain requires sigma*T<=1")
    frequency_factor = sqrt(1 - sigma**2 / 4)
    if frequency_factor.lo <= 0:
        raise ValueError("mobile spectral frequency factor is unresolved")
    center, carrier = horizon // 2, node_count // 4
    positive_phase_modes = tuple(
        range(carrier - fejer_order + 1, carrier + fejer_order)
    )
    phase_modes = tuple(
        sorted(positive_phase_modes + tuple(-k for k in positive_phase_modes))
    )
    radius = 3 * (fejer_order - 1)
    positive_modes = tuple(range(carrier - radius, carrier + radius + 1))
    modes = tuple(sorted(positive_modes + tuple(-k for k in positive_modes)))
    coefficient_square_sum = Q(2 * fejer_order**2 + 1, 3 * fejer_order)
    normalization_root = sqrt(I(coefficient_square_sum))
    normalization = I(
        amplitude / normalization_root.hi, amplitude / normalization_root.lo
    )
    zero = I(0)
    initial_positive = {}
    incidence = {}
    eigenvalues = {}
    for mode in positive_modes:
        angle = 2 * pi * Q(mode, node_count)
        sine, cosine = _reduced_sine_cosine(angle)
        eigenvalues[mode] = 1 - cosine
        incidence[mode] = (1 - cosine, -sine)
        incidence[-mode] = _conjugate(incidence[mode])
    for mode in positive_phase_modes:
        registration = -2 * pi * Q(mode * center, node_count)
        weight = 1 - Q(abs(mode - carrier), fejer_order)
        sine, cosine = _reduced_sine_cosine(registration)
        initial_positive[mode] = (
            normalization * weight * cosine / 2,
            normalization * weight * sine / 2,
        )

    def evaluate(time):
        if not isinstance(time, I):
            time = I(_exact(time, "spectral time"))
        if time.lo < 0 or time.hi > horizon:
            raise ValueError("spectral time interval must lie in [0,T]")
        phase = {}
        for mode in positive_phase_modes:
            _, _, u21, u22 = _mode_propagator(
                eigenvalues[mode], sigma, frequency_factor, time
            )
            phase[mode] = _complex_multiply(initial_positive[mode], (u21, -u22))
            phase[-mode] = _conjugate(phase[mode])
        first = {
            mode: _complex_multiply(incidence[mode], phase[mode])
            for mode in phase_modes
        }
        second = {
            mode: _complex_multiply(incidence[mode], phase[-mode])
            for mode in phase_modes
        }
        combined = {
            mode: (first[mode][0] + second[mode][0], first[mode][1] + second[mode][1])
            for mode in phase_modes
        }
        product = _dyadic_cyclic_convolution(first, second, node_count)
        mixed = _dyadic_cyclic_convolution(product, combined, node_count)
        coefficients = {}
        for mode in positive_modes:
            positive = _complex_multiply(_conjugate(incidence[mode]), mixed[mode])
            negative = _complex_multiply(_conjugate(incidence[-mode]), mixed[-mode])
            if not positive[1].contains(0) or not negative[1].contains(0):
                raise ArithmeticError(
                    "reflection-even cubic source lost zero imaginary enclosure"
                )
            source = I(
                max(positive[0].lo, negative[0].lo), min(positive[0].hi, negative[0].hi)
            )
            source /= 4 * node_count  # Two unitary DFT products contribute 1/N.
            u11, _, u21, _ = _mode_propagator(
                eigenvalues[mode], sigma, frequency_factor, horizon - time
            )
            value = u11 * source, zero, u21 * source, zero
            coefficients[mode] = coefficients[-mode] = value
        return tuple(coefficients[mode] for mode in modes)

    metadata = _MobileSpectralMetadata(
        node_count=node_count,
        fejer_order=fejer_order,
        amplitude=amplitude,
        horizon=horizon,
        form_loss_coefficient=loss,
        exchange_coefficient=exchange,
        storage_scale=beta,
        initial_centers=(center, -center),
        node_order=tuple(range(node_count)),
        edges=tuple((j, (j + 1) % node_count) for j in range(node_count)),
        phase_modes=phase_modes,
        modes=modes,
    )
    return metadata, evaluate
