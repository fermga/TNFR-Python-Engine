"""Exact modified-energy kernels for conditional reciprocal-law recovery.

Public owners admit the full law, mean leaf, acute trapping neighborhood and
primitive values. These private kernels take normalized rational bounds only;
they neither admit a report nor establish those geometric premises. They run
no exponential evaluator, choose no clock or horizon, and cache no evidence.

The common-quadratic helpers allow a time-varying symmetric stiffness under
fixed matrix bounds; their metric does not consume a stiffness derivative.
The separate fixed-potential helpers retain their autonomous gradient and
target-potential premises. Neither theorem transfers those premises to the
other mechanism.
"""

from __future__ import annotations

from fractions import Fraction as Q

from .._exact_time import exact_or_represented_real


def _quadratic_rational(value: Q, name: str) -> Q:
    """Require rational bounds and normalize fixed-width integer payloads."""
    if not isinstance(value, Q):
        raise TypeError(f"{name} must be an exact Fraction bound")
    return exact_or_represented_real(value, name)


def _common_quadratic_gamma(lower: Q, upper: Q) -> None:
    """Validate already normalized common-metric coupling bounds."""
    if not Q(0) <= lower <= upper <= Q(1, 2):
        raise ValueError("gamma bounds must satisfy 0 <= lower <= upper <= 1/2")


def _sine_common_quadratic_coefficients(
    *,
    gamma_lower: Q,
    gamma_upper: Q,
    cosine_lower: Q,
    gap_lower: Q,
    capacity_lower: Q,
    capacity_upper: Q,
) -> tuple[Q, Q, Q]:
    """Return norm metric bounds and a common quadratic decay rate.

    For ``a'=-(V H a+gamma V K b)/4`` and ``b'=gamma V H a/4``, let
    ``W=a^T V^-1 a+b^T V^-1 b+2 gamma a^T V^-1 b``. The caller proves
    symmetric ``H >= gap_lower I``, ``cosine_lower H <= K <= H`` and
    positive fixed diagonal V with the supplied capacity bounds. K may vary
    with time or state; no derivative of K enters this metric's identity.

    The returned ``(lower, upper, rate)`` satisfy
    ``lower ||(a,b)||^2 <= W <= upper ||(a,b)||^2`` and ``W' <= -rate W``
    under those hypotheses. Bounds obey ``0 <= gamma_lower <= gamma_upper
    <= 1/2``, ``0 <= cosine_lower <= 1`` and positive gap and capacities.
    Zero lower coupling or cosine gives rate zero, not positive recovery.
    All bounds must be Fractions; their payloads normalize before algebra.
    Geometry, trapping, clocks and event jumps remain caller obligations.
    """
    gamma_lower = _quadratic_rational(gamma_lower, "gamma_lower")
    gamma_upper = _quadratic_rational(gamma_upper, "gamma_upper")
    cosine_lower = _quadratic_rational(cosine_lower, "cosine_lower")
    gap_lower = _quadratic_rational(gap_lower, "gap_lower")
    capacity_lower = _quadratic_rational(capacity_lower, "capacity_lower")
    capacity_upper = _quadratic_rational(capacity_upper, "capacity_upper")
    _common_quadratic_gamma(gamma_lower, gamma_upper)
    if not Q(0) <= cosine_lower <= Q(1):
        raise ValueError("cosine_lower must lie in [0, 1]")
    if gap_lower <= 0:
        raise ValueError("gap_lower must be positive")
    if not 0 < capacity_lower <= capacity_upper:
        raise ValueError("capacity bounds must satisfy 0 < lower <= upper")
    return (
        (1 - gamma_upper) / capacity_upper,
        (1 + gamma_upper) / capacity_lower,
        capacity_lower
        * gamma_lower**2
        * cosine_lower
        * gap_lower
        / (4 * (1 + gamma_upper)),
    )


def _sine_common_quadratic_initial_upper(
    *,
    gamma_lower: Q,
    gamma_upper: Q,
    weighted_square: Q,
    weighted_cross: Q,
) -> Q:
    """Bound the common metric while retaining the signed mixed product.

    The caller supplies ``weighted_square=sum((a_i^2+b_i^2)/nu_i)`` and
    ``weighted_cross=sum(a_i*b_i/nu_i)`` from the same admitted state and
    positive fixed V. Their necessary inequality is
    ``weighted_square >= 2*abs(weighted_cross)``. A nonnegative mixed sum
    uses gamma_upper; a negative one uses gamma_lower. No source or report
    authentication is inferred from these scalar bounds.
    """
    gamma_lower = _quadratic_rational(gamma_lower, "gamma_lower")
    gamma_upper = _quadratic_rational(gamma_upper, "gamma_upper")
    weighted_square = _quadratic_rational(weighted_square, "weighted_square")
    weighted_cross = _quadratic_rational(weighted_cross, "weighted_cross")
    _common_quadratic_gamma(gamma_lower, gamma_upper)
    if weighted_square < 2 * abs(weighted_cross):
        raise ValueError("weighted_square must be >= 2*abs(weighted_cross)")
    gamma = gamma_upper if weighted_cross >= 0 else gamma_lower
    return weighted_square + 2 * gamma * weighted_cross


def _sine_lyapunov_coefficients(
    *,
    eta_lower: Q,
    eta_upper: Q,
    cosine_lower: Q | None,
    gap_lower: Q,
    rate_upper: Q,
    phase_hessian_upper: Q = Q(1),
) -> tuple[Q | None, Q, Q | None, Q, Q | None]:
    """Return ``(mu, M, a_minus, a_plus, decay_rate)`` for admitted bounds.

    Callers supply ``0 <= eta_lower <= eta_upper``, ``0 < gap_lower <=
    rate_upper`` and an admitted nonnegative acute cosine lower bound. None
    denotes a missing cosine premise, retaining only the upper coefficients.
    A zero lower stiffness supplies no positive decay rate. The spectral and
    nonlinear Hessian hypotheses belong to the caller's complete model.
    ``phase_hessian_upper`` bounds the actual phase Hessian relative to the
    graph Laplacian; its default one retains the original sine calculation.
    A different potential must supply its own bound, even at matched tangents.
    """
    if not isinstance(phase_hessian_upper, Q) or phase_hessian_upper <= 0:
        raise ValueError("phase_hessian_upper must be a positive rational bound")
    if cosine_lower is not None and cosine_lower > phase_hessian_upper:
        raise ValueError("phase Hessian lower bound exceeds its upper bound")
    epsilon = gap_lower / 4
    mu = eta_lower * cosine_lower * gap_lower**2 if cosine_lower is not None else None
    upper_stiffness = eta_upper * phase_hessian_upper * rate_upper**2
    lower_position = mu / 2 + gap_lower**2 / 16 if mu is not None else None
    upper_position = upper_stiffness / 2 + epsilon * rate_upper / 2 + epsilon**2
    decay_rate = (
        min(4 * (gap_lower - epsilon) / 3, epsilon * mu / upper_position)
        if mu is not None and mu > 0
        else None
    )
    return mu, upper_stiffness, lower_position, upper_position, decay_rate


def _sine_lyapunov_initial_upper(
    *,
    eta_upper: Q,
    rate_upper: Q,
    gap_lower: Q,
    position_upper: Q,
    form_norm_upper: Q,
    phase_norm_upper: Q,
) -> Q:
    """Bound modified energy from admitted Euclidean form/target-phase radii.

    Radii and coefficients are nonnegative; ``gap_lower`` is positive. The
    phase radius is about the critical target, not a nominal source proxy.
    No intervention is presumed: the same expression applies to an unprobed
    formed family or a separately admitted post-intervention family.
    """
    return (
        Q(3, 4) * eta_upper * rate_upper * form_norm_upper**2
        + position_upper * phase_norm_upper**2 / gap_lower
    )


def _sine_lyapunov_return_squared(
    *,
    initial_upper: Q,
    decay_upper: Q,
    eta_lower: Q,
    gap_lower: Q,
    rate_upper: Q,
    position_lower: Q,
) -> tuple[Q, Q, Q]:
    """Return energy, Euclidean form norm squared and phase norm squared.

    The caller proves uninterrupted trapping and supplies a nonnegative
    exponential upper bound; positive denominator bounds are already admitted.
    All arithmetic stays rational, including decay bounds below an interval
    grid. Exponential precision, work caps and final verdicts belong to the
    consuming owner. Neither an event nor a half-radius threshold is assumed.
    """
    returned = initial_upper * decay_upper
    form_squared = 4 * returned / (eta_lower * gap_lower)
    phase_squared = rate_upper * returned / position_lower
    return returned, form_squared, phase_squared


def _sine_common_quadratic_exchange_coefficients(
    *,
    gamma_lower: Q,
    gamma_upper: Q,
    cosine_lower: Q,
    gap_lower: Q,
    capacity: Q,
    exchange_norm_upper: Q,
) -> tuple[Q, Q, Q]:
    """Extend the common metric by commuting symmetric form exchange.

    The complete rows are ``a'=-(v H a+gamma v K b)/4-G a`` and
    ``b'=gamma v H a/4`` with one positive scalar v. In addition to the
    base kernel's bounds, the caller proves held symmetric H and G, G>=0,
    ||G||<=exchange_norm_upper, preservation of the declared invariant leaf,
    and HG=GH. H is positive on that leaf; H^-1 is used only there.
    K can vary and need not commute with either matrix.

    Add ``4 b^T G H^-1 b/v^2`` to the base W. Its derivative cancels the
    mixed exchange term, leaving the nonpositive term ``-2 a^T G a/v``.
    If extra=4||G||/(v^2*gap), then Wbase<=Wplus<=
    (1+extra/lower)*Wbase. Returned bounds are consequently
    (lower, upper+extra, rate*lower/(lower+extra)). Zero exchange returns
    exactly the original coefficients. These scalars do not certify any
    matrix, chart, source, event, physical capacity or clock premise.

    Every primitive is normalized before calling the base kernel. That
    owner admits its original domains; the new capacity and exchange
    domains are admitted here. No inverse, trajectory or exponential runs.
    """
    gamma_lower = _quadratic_rational(gamma_lower, "gamma_lower")
    gamma_upper = _quadratic_rational(gamma_upper, "gamma_upper")
    cosine_lower = _quadratic_rational(cosine_lower, "cosine_lower")
    gap_lower = _quadratic_rational(gap_lower, "gap_lower")
    capacity = _quadratic_rational(capacity, "capacity")
    exchange_norm_upper = _quadratic_rational(
        exchange_norm_upper, "exchange_norm_upper"
    )
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if exchange_norm_upper < 0:
        raise ValueError("exchange_norm_upper must be nonnegative")
    lower, upper, rate = _sine_common_quadratic_coefficients(
        gamma_lower=gamma_lower,
        gamma_upper=gamma_upper,
        cosine_lower=cosine_lower,
        gap_lower=gap_lower,
        capacity_lower=capacity,
        capacity_upper=capacity,
    )
    extra = 4 * exchange_norm_upper / (capacity**2 * gap_lower)
    return lower, upper + extra, rate * lower / (lower + extra)


def _sine_noncommuting_port_coefficients(
    *, transport_gap_lower: Q, port_norm_upper: Q, loss_divisor: Q
) -> tuple[Q, Q | None]:
    """Bound total quadratic loss with a noncommuting positive form port.

    The caller proves held symmetric H>=h I>0 on an invariant leaf and
    symmetric G>=0 preserving that leaf, with ||G||<=C. No commutation is
    required. For D=||H x||^2/d+x^T H G x, positive-contraction Jordan
    algebra gives D>=(1/d-C/(8h))*||H x||^2. Return that bracket and its
    product with h^2 only when the bracket is nonnegative; otherwise no
    Euclidean coercivity follows from a lower spectral bound alone.

    These are candidate scalar coefficients, not matrix, source, storage,
    port-only passivity or physical-unit admission. Every primitive must
    be an exact Fraction and is normalized before domain checks or algebra.
    No inverse, field, exponential, trajectory or event is evaluated.
    """
    transport_gap_lower = _quadratic_rational(
        transport_gap_lower, "transport_gap_lower"
    )
    port_norm_upper = _quadratic_rational(port_norm_upper, "port_norm_upper")
    loss_divisor = _quadratic_rational(loss_divisor, "loss_divisor")
    if transport_gap_lower <= 0:
        raise ValueError("transport_gap_lower must be positive")
    if port_norm_upper < 0:
        raise ValueError("port_norm_upper must be nonnegative")
    if loss_divisor <= 0:
        raise ValueError("loss_divisor must be positive")
    bracket = 1 / loss_divisor - port_norm_upper / (8 * transport_gap_lower)
    coercivity = bracket * transport_gap_lower**2 if bracket >= 0 else None
    return bracket, coercivity


def _sine_strict_energy_coefficients(
    *,
    storage_lower: Q,
    storage_upper: Q,
    form_loss_lower: Q,
    transport_norm_upper: Q,
    damping_norm_upper: Q,
    phase_stiffness_lower: Q,
    gamma_lower: Q,
    gamma_upper: Q,
    loss_divisor: Q,
    cross_coefficient: Q,
) -> tuple[Q, Q, Q, Q, Q | None]:
    """Strictify an admitted full-state energy using the complete phase row.

    The caller proves ``L*(||x||^2+||y||^2)<=E<=U*(||x||^2+||y||^2)``
    and ``E'<=-a*||x||^2`` on an invariant domain for the complete rows
    ``x'=-D*x-gamma*g(y)/d`` and ``y'=gamma*H*x/d``. The fixed coupling
    lies in the supplied interval, ``||D||<=Dbar``, ``||H||<=Hbar``, and
    ``y.g(y)>=k*||y||^2`` on the admitted phase leaf. The form leaf can
    retain common modes. These matrix, gradient, storage and domain premises
    are not established by this scalar helper; no commutation is required.

    For ``V=E+eps*gamma*x.y``, return candidate bounds ``(Ls,Us,qx,qy,rate)``.
    When Ls,qx,qy are strictly positive, ``Ls*N^2<=V<=Us*N^2`` and
    ``V'<=-qx*||x||^2-qy*||y||^2<=-rate*V``. Young's inequality uses a/2
    of the original form loss. A nonpositive candidate phase bracket does
    not give a valid loss coefficient by substituting the lower gamma;
    such candidates retain rate None, rather than claiming recovery.

    All ten bounds must be Fractions and normalize before any comparison
    or algebra. Zero cross coefficient, lower coupling or phase stiffness
    gives no positive rate. No horizon, exponential, state or field runs.
    """
    storage_lower = _quadratic_rational(storage_lower, "storage_lower")
    storage_upper = _quadratic_rational(storage_upper, "storage_upper")
    form_loss_lower = _quadratic_rational(form_loss_lower, "form_loss_lower")
    transport_norm_upper = _quadratic_rational(
        transport_norm_upper, "transport_norm_upper"
    )
    damping_norm_upper = _quadratic_rational(damping_norm_upper, "damping_norm_upper")
    phase_stiffness_lower = _quadratic_rational(
        phase_stiffness_lower, "phase_stiffness_lower"
    )
    gamma_lower = _quadratic_rational(gamma_lower, "gamma_lower")
    gamma_upper = _quadratic_rational(gamma_upper, "gamma_upper")
    loss_divisor = _quadratic_rational(loss_divisor, "loss_divisor")
    cross_coefficient = _quadratic_rational(cross_coefficient, "cross_coefficient")
    if not 0 < storage_lower <= storage_upper:
        raise ValueError("storage bounds must satisfy 0 < lower <= upper")
    if form_loss_lower <= 0:
        raise ValueError("form_loss_lower must be positive")
    if min(transport_norm_upper, damping_norm_upper, phase_stiffness_lower) < 0:
        raise ValueError("transport, damping and phase bounds must be nonnegative")
    if not 0 <= gamma_lower <= gamma_upper:
        raise ValueError("gamma bounds must satisfy 0 <= lower <= upper")
    if loss_divisor <= 0:
        raise ValueError("loss_divisor must be positive")
    if cross_coefficient < 0:
        raise ValueError("cross_coefficient must be nonnegative")
    cross = cross_coefficient * gamma_upper / 2
    lower, upper = storage_lower - cross, storage_upper + cross
    form_loss = (
        form_loss_lower / 2
        - cross_coefficient * gamma_upper**2 * transport_norm_upper / loss_divisor
    )
    phase_loss = (
        cross_coefficient
        * gamma_lower**2
        * (
            phase_stiffness_lower / loss_divisor
            - cross_coefficient * damping_norm_upper**2 / (2 * form_loss_lower)
        )
    )
    rate = (
        min(form_loss, phase_loss) / upper
        if lower > 0 and form_loss > 0 and phase_loss > 0
        else None
    )
    return lower, upper, form_loss, phase_loss, rate
