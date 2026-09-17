"""Self-consistent spatial and rounding bounds for the fixed carried C6 map.

The profile, contraction and numeric pressure law have existing owners. This
module closes their joint error inequality exactly, without iterating fitted
error estimates. Arithmetic mean and finite band admission remain separate.
"""

from dataclasses import dataclass
from fractions import Fraction as F
from math import isqrt
import sys

from ..dynamics._euler_kernel import NodalRemainderState
from ._cycle_algebra import Vector, laplacian_action
from .c6_carried_profile import C6CarriedProfile
from .c6_carried_tube import C6CarriedTube, derive_c6_carried_tube

__all__ = ["C6CarriedClosure", "derive_c6_carried_closure"]


def _sqrt_upper(value: F) -> F:
    """Return an exact rational upper square root, with no float conversion.

    For value=n/d, ceil(sqrt(n*d))/d suffices. Perfect rational squares
    are returned exactly. This auxiliary bound is used only for displayed
    row/mean envelopes; the invariant energy proof has no sqrt rounding.
    """
    if type(value) is not F or value < 0:
        raise ValueError("the square-root bound requires a nonnegative Fraction")
    product = value.numerator * value.denominator
    root = isqrt(product)
    return F(root + int(root * root < product), value.denominator)


def _integer_gradient_interval(center, quantum, carry, squared):
    """Sharp integer hull of |quantum*(m-center)|<=sqrt(squared)+2*carry."""
    radius = (_sqrt_upper(squared) + 2 * carry) / quantum
    lower, upper = (center - radius).__ceil__(), (center + radius).__floor__()

    def admitted(index):
        residual = abs(quantum * (index - center)) - 2 * carry
        return residual <= 0 or residual * residual <= squared

    pivot = center.__floor__()
    if not admitted(pivot):
        raise RuntimeError("a valid initial state must admit a nonempty gradient interval")
    left, right = lower, pivot
    while left < right:
        middle = (left + right) // 2
        if admitted(middle):
            right = middle
        else:
            left = middle + 1
    lower = left
    left, right = pivot, upper
    while left < right:
        middle = (left + right + 1) // 2
        if admitted(middle):
            left = middle
        else:
            right = middle - 1
    upper = left
    if admitted(lower - 1) or admitted(upper + 1):
        raise RuntimeError("the integer gradient hull lost its exact endpoint proof")
    return lower, upper


@dataclass(frozen=True, slots=True)
class C6CarriedClosure:
    """Exact invariant spatial envelope conditional on the fixed source and band.

    Every step whose displayed state remains in the admitted band obeys
    the bound. This object does not establish infinite band containment,
    signed-mean cancellation, gradient reachability or live graph admission.
    The gradient intervals are necessary coordinate conditions; their
    Cartesian product is not a class of jointly attainable graph states.
    """

    base_tube: C6CarriedTube
    profile_laplacian: Vector
    rounding_affine_constant: F
    rounding_affine_slope: F
    effective_norm_factor: F
    energy_floor: F
    energy_bound: F
    laplacian_error_squared_bound: F
    laplacian_error_upper_bound: F
    product_error_bound: F
    assembly_error_bound: F
    rounding_bound: F
    mean_increment_lower: F
    mean_increment_upper: F
    profile_gradient_indices: Vector
    gradient_index_lower: tuple[int, ...]
    gradient_index_upper: tuple[int, ...]

    @property
    def infinite_mean_control_certified(self) -> bool:
        return False

    @property
    def gradient_reachability_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def derive_c6_carried_closure(
    profile: C6CarriedProfile, *, state: NodalRemainderState, timestep: float,
) -> C6CarriedClosure:
    """Close the carried spatial/RN feedback with an exact rational energy floor.

    Write E=||y||^2, y=P*(x+r)-z, G=max|L*z|, R=ulp(upper)/2,
    u=2^-53 and t=2^-1075. Since each C6 Laplacian row has squared norm
    3/2, |L*x|<=G+sqrt(3*E/2)+2*R. The two pressure RN operations give
    epsilon(E)<=a+b*sqrt(3*E/2), where
      a=u*(2+u)*w*(G+2*R)+u*max|A|+(2+u)*t,
      b=u*(2+u)*w.
    The existing sharp centered contraction q therefore yields
      sqrt(E_next)<=(q+3*h*b)*sqrt(E)+h*sqrt(6)*(2*w*R+a).
    Set Q=q+3*h*b. When Q<1, the invariant envelope is
      E_bound=max(E_initial,6*h^2*(2*w*R+a)^2/(1-Q)^2).
    The identity sqrt(6)*sqrt(3/2)=3 makes this floor wholly rational;
    no observed maxima, fixed-point iteration or numerical eigensolver
    enters the proof. The separate signed mean interval is
      h*(mean(A)-epsilon_bound) <= mean(X_next-X)
                               <= h*(mean(A)+epsilon_bound).
    Carry feedback has exactly zero mean and cannot cancel mean(A).

    Integer gradient intervals retain exact square comparisons at their
    endpoints. They can sharpen later finite pressure-class observations,
    but do not assert that every integer in the intervals is reachable.
    """
    base = derive_c6_carried_tube(profile, state=state, timestep=timestep)
    contraction = base.contraction
    balance = contraction.profile.forced_balance
    weight, h, carry = balance.epi_weight, F(contraction.timestep), base.carry_bound
    unit, tail = F(1, 2**53), F(1, 2**1075)
    laplacian = laplacian_action(balance.relative_profile)
    source_max = max(map(abs, balance.forcing))
    geometric = max(map(abs, laplacian)) + 2 * carry
    slope = unit * (2 + unit) * weight
    constant = slope * geometric + unit * source_max + (2 + unit) * tail
    effective = contraction.norm_factor + 3 * h * slope
    if effective >= 1:
        raise ValueError("the self-consistent rounding feedback requires q + 3*h*b < 1")
    floor = 6 * h * h * (2 * weight * carry + constant)**2 / (1 - effective)**2
    energy = max(base.initial_energy, floor)
    squared = F(3, 2) * energy
    radius = _sqrt_upper(squared)
    product_argument = weight * (geometric + radius)
    product_error = unit * product_argument + tail
    assembly_argument = source_max + product_argument + product_error
    if max(product_argument, assembly_argument) > F(sys.float_info.max):
        raise ValueError("the closed pressure envelope cannot certify finite binary64 operations")
    assembly_error = unit * assembly_argument + tail
    rounding = constant + slope * radius
    if rounding != product_error + assembly_error:
        raise RuntimeError("the affine rounding envelope lost its two-operation decomposition")
    quantum = contraction.profile.lattice.gradient_quantum
    centers = tuple(-value / quantum for value in laplacian)
    intervals = tuple(_integer_gradient_interval(value, quantum, carry, squared) for value in centers)
    visible = tuple(F(value) for value in state.epi)
    actual = tuple(-value / quantum for value in laplacian_action(visible))
    if any(value.denominator != 1 or not lo <= value <= hi
           for value, (lo, hi) in zip(actual, intervals, strict=True)):
        raise RuntimeError("the primitive initial gradient escaped its derived necessary intervals")
    mean = balance.mean_drift
    return C6CarriedClosure(
        base, laplacian, constant, slope, effective, floor, energy, squared, radius,
        product_error, assembly_error, rounding, h * (mean - rounding), h * (mean + rounding),
        centers, tuple(lo for lo, _ in intervals), tuple(hi for _, hi in intervals),
    )
