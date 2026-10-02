"""Exact spatial contraction and finite band budgets for the carried C6 map.

All bounds concern the fixed numerical source and unit capacities. They
neither prescribe an operator sequence nor certify future graph admission.
The common mean is bounded separately from centered disagreement.
"""

import math
import sys
from dataclasses import dataclass
from fractions import Fraction as F

from ..dynamics._euler_kernel import (
    NodalRemainderState,
    _finite_binary64,
    _require_remainder_rounding,
    _validate_nodal_remainder_state,
)
from ._cycle_algebra import Matrix, Vector, dot, laplacian_action, laplacian_matrix
from .c6_carried_profile import C6CarriedProfile, derive_c6_carried_profile
from .forced_support import observe_forced_support_pattern

__all__ = [
    "C6CarriedContraction",
    "derive_c6_carried_contraction",
    "C6CarriedTube",
    "derive_c6_carried_tube",
    "C6CarriedBandHorizon",
    "derive_c6_carried_band_horizon",
    "C6CarriedCutExclusion",
    "observe_c6_carried_cut_exclusion",
]


def _profile(profile):
    if type(profile) is not C6CarriedProfile:
        raise TypeError("profile must be a C6CarriedProfile")
    return derive_c6_carried_profile(profile.lattice)


@dataclass(frozen=True, slots=True)
class C6CarriedContraction:
    """Sharp Euclidean norm factor on the exact five-dimensional mean-zero subspace."""

    profile: C6CarriedProfile
    timestep: float
    step_factor: F
    laplacian: Matrix
    centering: Matrix
    transition: Matrix
    nonuniform_eigenvalues: Vector
    eigenvectors: tuple[Vector, ...]
    mode_factors: Vector
    norm_factor: F
    energy_factor: F


def derive_c6_carried_contraction(profile, *, timestep: float) -> C6CarriedContraction:
    """Certify T=I-h*w*L exactly, using a complete rational orthogonal basis.

    For 0<s=h*w<1, nonuniform Laplacian eigenvalues are 1/2,1/2,
    3/2,3/2,2, so q=max|1-s*lambda|<1. The squared-norm gain is
    q^2; the uniform vector instead has gain one. No floating eigensolver,
    measured decay rate or rounded transition matrix enters this identity.
    """
    ref = _profile(profile)
    h = _finite_binary64(timestep, "timestep")
    s = F(h) * ref.forced_balance.epi_weight
    if not 0 < s < 1:
        raise ValueError(
            "strict C6 spatial contraction requires 0 < timestep*epi_weight < 1"
        )
    laplacian = laplacian_matrix(6)
    identity = tuple(tuple(F(i == j) for j in range(6)) for i in range(6))
    projection = tuple(tuple(value - F(1, 6) for value in row) for row in identity)
    transition = tuple(
        tuple(value - s * laplacian[i][j] for j, value in enumerate(row))
        for i, row in enumerate(identity)
    )
    eigenvalues = (F(1, 2), F(1, 2), F(3, 2), F(3, 2), F(2))
    vectors = tuple(
        tuple(map(F, row))
        for row in (
            (2, 1, -1, -2, -1, 1),
            (0, 1, 1, 0, -1, -1),
            (2, -1, -1, 2, -1, -1),
            (0, 1, -1, 0, 1, -1),
            (1, -1, 1, -1, 1, -1),
        )
    )
    basis = ((F(1),) * 6,) + vectors
    if any(not dot(v, v) for v in basis) or any(
        dot(v, w) for i, v in enumerate(basis) for w in basis[i + 1 :]
    ):
        raise RuntimeError(
            "the six rational directions lost their complete orthogonal basis"
        )
    for value, vector in zip(eigenvalues, vectors, strict=True):
        if laplacian_action(vector) != tuple(value * entry for entry in vector):
            raise RuntimeError("the rational cycle mode lost its Laplacian identity")
    factors = tuple(1 - s * value for value in eigenvalues)
    q = max(map(abs, factors))
    if not 0 <= q < 1 or any(sum(row) != 1 for row in transition):
        raise RuntimeError("the centered contraction or preserved mean identity failed")
    return C6CarriedContraction(
        ref,
        h,
        s,
        laplacian,
        projection,
        transition,
        eigenvalues,
        vectors,
        factors,
        q,
        q * q,
    )


@dataclass(frozen=True, slots=True)
class C6CarriedTube:
    """Uniform disagreement envelope with a separate absolute mean-increment bound.

    Energy is sum(y_i^2), where y=P*X-z. The affine energy bound uses
    factor q (Young's inequality), while the homogeneous sharp gain is
    q^2. Neither the persistent energy floor nor a per-step mean bound
    implies asymptotic convergence or an infinite signed mean budget.
    """

    contraction: C6CarriedContraction
    state: NodalRemainderState
    carry_bound: F
    product_error_bound: F
    assembly_error_bound: F
    rounding_bound: F
    forcing_component_bound: F
    centered_forcing_norm_squared_bound: F
    initial_error: Vector
    initial_energy: F
    energy_floor: F
    energy_bound: F
    mean_increment_bound: F
    initial_mean: F

    @property
    def infinite_mean_control_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def derive_c6_carried_tube(
    profile, *, state: NodalRemainderState, timestep: float
) -> C6CarriedTube:
    """Derive all numerical defect bounds from the declared band and coefficients.

    Let D=upper-lower, u=2^-53, t=2^-1075 and A be the actual represented
    fixed phase source. The local reducer is RN(w*g), |g|<=D. Its error
    is bounded by e1=u*w*D+t; the subsequent sum has error at most
    e2=u*(max|A|+w*D+e1)+t. Products and sums are checked against the
    largest finite binary64 value. Thus |eta_i|<=e1+e2=epsilon, uniformly.

    R=ulp(upper)/2 bounds every carry in the positive band. Consequently
    |w*(L*r)_i+eta_i|<=2*w*R+epsilon=B and ||P*d||^2<=6*B^2.
    For E=||P*X-z||^2, Young's inequality gives
    E_next<=q*E+h^2*6*B^2/(1-q). The invariant energy envelope is
    max(E_initial,6*h^2*B^2/(1-q)^2). It remains conditional on band/source
    admission until the separate finite-band bootstrap closes that premise.
    The exact zero mean of L*r removes carry from the mean bound:
    |mean(X_next)-mean(X)|<=h*(|mean(A)|+epsilon).
    """
    _require_remainder_rounding()
    contraction = derive_c6_carried_contraction(profile, timestep=timestep)
    ref = contraction.profile
    exact = _validate_nodal_remainder_state(state)
    source = ref.lattice.source
    if (
        len(exact) != 6
        or not source.epi_lower
        <= state.epi_lower
        <= state.epi_upper
        <= source.epi_upper
    ):
        raise ValueError(
            "the six-coordinate state's band must lie inside the reference slab"
        )
    weight, h = ref.forced_balance.epi_weight, F(contraction.timestep)
    width = F(state.epi_upper) - F(state.epi_lower)
    unit, half_subnormal = F(1, 2**53), F(1, 2**1075)
    product = weight * width
    product_error = unit * product + half_subnormal
    assembly_argument = (
        max(abs(value) for value in ref.forced_balance.forcing)
        + product
        + product_error
    )
    if max(product, assembly_argument) > F(sys.float_info.max):
        raise ValueError(
            "the uniform pressure envelope cannot certify finite binary64 operations"
        )
    assembly_error = unit * assembly_argument + half_subnormal
    rounding = product_error + assembly_error
    carry = F(math.ulp(state.epi_upper)) / 2
    forcing = 2 * weight * carry + rounding
    norm_squared = 6 * forcing**2
    pattern = observe_forced_support_pattern(
        ref.forced_balance, nodes=tuple(range(6)), epi=exact
    )
    energy = dot(pattern.relative_error, pattern.relative_error)
    floor = h * h * norm_squared / (1 - contraction.norm_factor) ** 2
    source_mean = sum(ref.forced_balance.forcing, F(0)) / 6
    return C6CarriedTube(
        contraction,
        state,
        carry,
        product_error,
        assembly_error,
        rounding,
        forcing,
        norm_squared,
        pattern.relative_error,
        energy,
        floor,
        max(energy, floor),
        h * (abs(source_mean) + rounding),
        sum(exact, F(0)) / 6,
    )


def _tube(tube):
    if type(tube) is not C6CarriedTube:
        raise TypeError("tube must be a C6CarriedTube")
    return derive_c6_carried_tube(
        tube.contraction.profile, state=tube.state, timestep=tube.contraction.timestep
    )


@dataclass(frozen=True, slots=True)
class C6CarriedBandHorizon:
    """Sufficient numerical-band horizon; failure beyond it is not actual exit.

    All prefixes through maximum_steps retain the fixed numerical-map
    premises by induction. No loop of that length is executed. This does
    not bind live operator admission, events, changing phase or support.
    """

    tube: C6CarriedTube
    centered_coordinate_squared_bound: F
    minimum_initial_margin: F
    tube_initially_admitted: bool
    maximum_steps: int | None
    next_step_margin: F | None
    next_step_passes: bool | None
    unbounded_conditional_prefix: bool

    @property
    def actual_band_exit_certified(self) -> bool:
        return False

    @property
    def future_runtime_certified(self) -> bool:
        return False


def derive_c6_carried_band_horizon(tube) -> C6CarriedBandHorizon:
    """Close the finite band premise with exact margins and a monotone search.

    Centering gives |y_i|^2<=5*E_bound/6. With the per-step mean bound b,
    a=min_i(mean0+z_i-lower,upper-mean0-z_i) suffices through n steps
    whenever a-n*b>=0 and (a-n*b)^2>=5*E_bound/6. Induction first bounds
    the exact next candidate, then admits its RN value and retained carry.
    This is a sufficient certificate, not the maximal actual survival time.
    None distinguishes a rejected initial tube from an unbounded admitted
    one through the explicit admission and unbounded flags.
    """
    bound = _tube(tube)
    lower, upper = F(bound.state.epi_lower), F(bound.state.epi_upper)
    profile = bound.contraction.profile.forced_balance.relative_profile
    margin = min(
        min(bound.initial_mean + value - lower, upper - bound.initial_mean - value)
        for value in profile
    )
    squared = F(5, 6) * bound.energy_bound
    rate = bound.mean_increment_bound

    def admitted(n):
        reserve = margin - n * rate
        return reserve >= 0 and reserve * reserve >= squared

    initial = admitted(0)
    maximum = next_margin = next_passes = None
    unbounded = initial and rate == 0
    if initial and not unbounded:
        low, high = 0, margin // rate
        while low < high:
            middle = (low + high + 1) // 2
            if admitted(middle):
                low = middle
            else:
                high = middle - 1
        maximum = low
        next_margin = margin - (maximum + 1) * rate
        next_passes = admitted(maximum + 1)
        if next_passes:
            raise RuntimeError(
                "the sufficient band horizon is not the maximal admitted integer"
            )
    return C6CarriedBandHorizon(
        bound, squared, margin, initial, maximum, next_margin, next_passes, unbounded
    )


@dataclass(frozen=True, slots=True)
class C6CarriedCutExclusion:
    """One sufficient gradient-cut exclusion; a false flag proves no reachability."""

    tube: C6CarriedTube
    node: int
    nonpositive_cut: int
    profile_gradient_index: F
    remaining_laplacian_distance: F
    centered_laplacian_squared_bound: F
    cut_excluded: bool

    @property
    def cut_reachability_certified(self) -> bool:
        return False


def observe_c6_carried_cut_exclusion(tube, *, node: int) -> C6CarriedCutExclusion:
    """Use a derived centered tube to exclude, or leave open, a nonpositive cut.

    For m*=-2*L*z/delta and m<=k, the visible deviation needs
    delta*(m*-k)/2 <= |L*y|+|L*r| <= sqrt(3*E_bound/2)+2*R.
    Subtract 2*R and square only if the remaining distance is positive.
    Failure of the sufficient exclusion is not a feasible path or sign hit.
    """
    bound = _tube(tube)
    if type(node) is not int or not 0 <= node < 6:
        raise ValueError("node must be an integer C6 index")
    ref = bound.contraction.profile
    quantum = ref.lattice.epi_quantum
    index = -2 * laplacian_action(ref.forced_balance.relative_profile)[node] / quantum
    cut = ref.lattice.rows[node].nonpositive_max_index
    distance = quantum * (index - cut) / 2 - 2 * bound.carry_bound
    squared = F(3, 2) * bound.energy_bound
    return C6CarriedCutExclusion(
        bound,
        node,
        cut,
        index,
        distance,
        squared,
        distance > 0 and distance**2 > squared,
    )
