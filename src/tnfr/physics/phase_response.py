"""Exact conditional phase geometry and joint nodal response.

Cosine Gram data describe unit planar phasors without approximating their
angles. These detached coefficients do not identify live phase gates, a
binary64 derivative, a fixed point or a complete operator trajectory.
Joint pressure/acceleration identities reuse the transport owner and retain
the still-supplied capacity and phase velocities as explicit inputs.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real
from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Matrix, Vector, dot, ordered_vector
from .support_transport import (
    SupportTransportSnapshot,
    _rebuild,
    _support_gradient,
    observe_support_transport_derivative,
)

__all__ = [
    "PhaseResponseReference", "derive_phase_response",
    "PhaseSourceGeometry", "observe_phase_source_geometry",
    "JointNodalResponse", "derive_joint_nodal_response",
]


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered sequence")
    try:
        return tuple(values)
    except TypeError as error:
        raise TypeError(f"{label} must be an ordered sequence") from error


def _unit_planar_gram(values):
    gram = tuple(ordered_vector(row, "cosine Gram row")
                 for row in _ordered(values, "cosine_gram"))
    n = len(gram)
    if not n or any(len(row) != n for row in gram):
        raise ValueError("cosine Gram must be a nonempty square matrix")
    if any(gram[i][i] != 1 for i in range(n)):
        raise ValueError("unit cosine Gram requires diagonal one")
    if any(gram[i][j] != gram[j][i] for i in range(n) for j in range(n)):
        raise ValueError("cosine Gram must be symmetric")
    # Unit first vector fixes one axis. The Schur complement must be a
    # positive semidefinite rank-at-most-one Gram on the remaining axis.
    residual = tuple(tuple(gram[i][j] - gram[i][0] * gram[0][j]
                           for j in range(n)) for i in range(n))
    if any(residual[i][i] < 0 for i in range(n)):
        raise ValueError("cosine Gram must be positive semidefinite and planar")
    pivot = next((i for i in range(n) if residual[i][i] > 0), None)
    if pivot is None:
        valid = not any(value for row in residual for value in row)
    else:
        valid = all(residual[i][j] * residual[pivot][pivot]
                    == residual[i][pivot] * residual[pivot][j]
                    for i in range(n) for j in range(n))
    if not valid:
        raise ValueError("cosine Gram must be positive semidefinite with rank at most two")
    return gram


def _index_rows(values, n, label):
    rows = tuple(_ordered(row, label) for row in _ordered(values, label))
    if len(rows) != n:
        raise ValueError(f"{label} rows must match the Gram dimension")
    for row in rows:
        if (not row or any(type(i) is not int or not 0 <= i < n for i in row)
                or len(set(row)) != len(row)):
            raise ValueError(f"{label} requires nonempty rows of distinct valid indices")
    return rows


@dataclass(frozen=True)
class PhaseResponseReference:
    """One exact phase Jacobian from declared mean and receiving-source rows."""

    cosine_gram: Matrix
    mean_neighbors: tuple
    receiver_sources: tuple
    phase_factor: Fraction
    mean_response: Matrix
    mean_resultant_squared: Vector
    receiver_response: Matrix
    jacobian: Matrix
    row_sum_residuals: Vector
    is_nonnegative: bool


def derive_phase_response(
    *, cosine_gram, mean_neighbors, receiver_sources, phase_factor,
) -> PhaseResponseReference:
    """Differentiate unweighted phasor means and average receiver proposals.

    For a nonzero S_i=sum_{j in N_i} exp(i*theta_j), differentiation gives
    d Arg(S_i)/d theta_j = Re(exp(i*theta_j)/S_i). A supplied unit planar
    cosine Gram G_jk=cos(theta_j-theta_k) realizes this coefficient exactly as
    1[j in N_i]*sum_{k in N_i}G_jk / sum_{l,k in N_i}G_lk. Its rows sum to
    one but may contain negative entries. Zero resultants are rejected.

    Each receiver i averages the phase proposals from its declared sources
    T_i. In any regular fixed shortest-arc chart the Jacobian is
    (1-factor)*I + factor*mean_{s in T_i} R_s. Pointwise IL uses T_i=(i,);
    all-target bidirectional UM must supply its actual contributing sources,
    with the target INCLUDED in each source mean. Nonzero baseline proposal
    displacements need not vanish individually for this derivative to hold.

    Input Gram data must be exactly symmetric, unit, positive semidefinite
    and rank <=2. A rounded cosine table may fail those identities and is
    never repaired into an apparent theorem. Exact rational inputs and other
    supported represented reals use the shared scalar reader. Factors in
    [0,1] are detached algebraic inputs, not live operator admission.

    No phase angles, orientation, fixed point, U3 support or local chart are
    inferred. This is not the derivative of binary64 arithmetic. Nodal EPI
    pressure uses its own factor and support, even when sharing the mean
    derivative. A nonnegative Jacobian alone is not strict contraction.
    """
    gram = _unit_planar_gram(cosine_gram)
    n = len(gram)
    neighbors = _index_rows(mean_neighbors, n, "mean_neighbors")
    receivers = _index_rows(receiver_sources, n, "receiver_sources")
    factor = exact_or_represented_real(phase_factor, "phase_factor")
    if not 0 <= factor <= 1:
        raise ValueError("phase_factor must lie in [0,1]")
    squared = tuple(sum((gram[j][k] for j in row for k in row), Fraction(0))
                    for row in neighbors)
    if any(value <= 0 for value in squared):
        raise ValueError("every declared phasor mean requires a nonzero resultant")
    mean = tuple(tuple(sum((gram[j][k] for k in row), Fraction(0)) / squared[i]
                       if j in row else Fraction(0) for j in range(n))
                 for i, row in enumerate(neighbors))
    received = tuple(tuple(sum((mean[s][j] for s in sources), Fraction(0)) / len(sources)
                           for j in range(n)) for sources in receivers)
    jacobian = tuple(tuple((1 - factor) * int(i == j) + factor * received[i][j]
                           for j in range(n)) for i in range(n))
    residuals = tuple(sum(row, Fraction(0)) - 1 for row in jacobian)
    if any(residuals) or any(sum(row, Fraction(0)) != 1 for row in mean):
        raise RuntimeError("exact phase derivative lost rotation covariance")
    return PhaseResponseReference(
        gram, neighbors, receivers, factor, mean, squared, received, jacobian,
        residuals, all(value >= 0 for row in jacobian for value in row),
    )


@dataclass(frozen=True)
class PhaseSourceGeometry:
    """Conditional exact tangent geometry of the canonical phase source.

    On a regular shortest-arc chart, Dg=(R-I)/pi. The stored matrix is R-I,
    whose kernel and rank equal those of Dg. A one-dimensional kernel allows
    only common rotation; a larger tangent space does not prove a finite
    source-preserving path. These detached data do not verify an actual phase
    chart, U3 gates, a dynamical law, stability or runtime provenance.
    """

    reference: PhaseResponseReference
    scaled_source_jacobian: Matrix
    rank: int
    tangent_dimension: int
    mean_is_nonnegative: bool

    @property
    def only_common_rotation(self) -> bool:
        """Whether the conditional tangent kernel is exactly the rotation line."""
        return self.tangent_dimension == 1


def observe_phase_source_geometry(reference) -> PhaseSourceGeometry:
    """Read exact fixed-source tangent freedom from the shared mean derivative.

    Rebuild the supplied reference from its primitive Gram and incidence data
    before using any cached coefficient. This observes mean_response R, not
    the final merged operator-stage Jacobian: a zero stage factor must not
    turn a rigid source into an apparent identity map. Signed mean responses
    are supported; nonnegativity alone is neither necessary nor sufficient
    for one-dimensional tangent freedom. See FORCED_SUPPORT_BALANCE section 24
    for the nonnegative irreducible theorem and its regular-chart hypotheses.
    """
    if type(reference) is not PhaseResponseReference:
        raise TypeError("phase source geometry requires a PhaseResponseReference")
    rebuilt = derive_phase_response(
        cosine_gram=reference.cosine_gram,
        mean_neighbors=reference.mean_neighbors,
        receiver_sources=reference.receiver_sources,
        phase_factor=reference.phase_factor,
    )
    if rebuilt != reference:
        raise ValueError("phase response reference differs from its rebuilt coefficients")
    mean = rebuilt.mean_response
    size = len(mean)
    jacobian = tuple(tuple(value - int(i == j) for j, value in enumerate(row))
                     for i, row in enumerate(mean))
    rank = exact_rank(jacobian)
    if rank >= size:
        raise RuntimeError("phase source derivative lost common-rotation invariance")
    return PhaseSourceGeometry(
        rebuilt, jacobian, rank, size - rank,
        all(value >= 0 for row in mean for value in row),
    )


@dataclass(frozen=True)
class JointNodalResponse:
    """Conditional exact joint pressure response and nodal acceleration.

    Stored pressure is the declared p in x'=nu*p. Equality to the canonical
    pressure law, the phase/Gram relationship and the regular wrap chart are
    hypotheses, not certified by these detached data. Supplied phase/capacity
    rates are independent inputs; this observer does not derive their laws.
    """

    source: SupportTransportSnapshot
    phase_geometry: PhaseSourceGeometry
    epi_weight: Fraction
    phase_weight: Fraction
    capacity_weight: Fraction
    phase_rate_over_pi: Vector
    capacity_rate: Vector
    epi_pressure_rate: Vector
    phase_pressure_rate: Vector
    capacity_pressure_rate: Vector
    pressure_rate: Vector
    capacity_acceleration: Vector
    pressure_acceleration: Vector
    epi_acceleration: Vector
    scope: tuple[str, ...] = (
        "conditional_exact_real_smooth_response",
        "fixed_symmetric_effective_conductance_and_unique_support",
        "fixed_channel_coefficients_without_renormalization",
        "nonempty_support_at_every_node",
        "declared_stored_pressure_p_and_nodal_rate_nu_times_p",
        "requires_p_equal_canonical_pressure_and_regular_phase_chart",
        "phase_gram_state_and_pressure_compatibility_not_certified",
        "phase_rate_over_pi_and_capacity_rate_are_supplied_not_derived",
        "fixed_topology_channel_has_zero_derivative",
        "no_binary64_derivative_runtime_or_future_admissibility_claim",
    )


def derive_joint_nodal_response(
    snapshot, phase_reference, *, epi_weight, phase_weight, capacity_weight,
    phase_rate_over_pi, capacity_rate,
) -> JointNodalResponse:
    """Differentiate the joint canonical pressure law on declared fixed support.

    Let B_W be the weighted neighbor-difference operator, B_U the unweighted
    unique-support operator, and R the circular-mean response. With fixed
    conductance, support and channel coefficients, the conditional identity is

        p' = w_E*B_W*(nu*p) + w_phi*(R-I)*(theta'/pi) + w_nu*B_U*nu',
        x'' = nu'*p + nu*p'.

    The topology channel has zero derivative on this fixed support. Loops
    count once, parallel edges aggregate only for EPI, and zero-conductance
    support edges still enter phase/capacity means. ``phase_rate_over_pi`` is
    an exact declared coordinate: no rational approximation to pi is inserted.
    Weights are nonnegative exact/represented scalars and are not normalized.

    Rebuild all support caches and validate the phase reference through the
    existing source-geometry owner. Its mean neighborhoods must match the
    snapshot's unique support, independent of iteration order. This first
    domain excludes empty graphs and isolates even when a weight is zero.

    The source's stored pressure is DECLARED p. The identity requires p to
    equal the canonical pressure law along a differentiable path with a
    nonzero-resultant regular shortest-arc chart. This call cannot establish
    that hypothesis, associate the exact Gram with a live phase state, or
    identify a derivative of floating arithmetic. No missing constitutive
    law or offset is supplied, and neither a graph nor a trajectory is changed.
    """
    source = _rebuild(snapshot)
    size = len(source.nodes)
    if not size or any(not row for row in source.support_neighbors):
        raise ValueError("joint nodal response requires nonempty support at every node")
    geometry = observe_phase_source_geometry(phase_reference)
    reference = geometry.reference
    if len(reference.mean_neighbors) != size or any(
        set(mean) != set(support) for mean, support in zip(
            reference.mean_neighbors, source.support_neighbors, strict=True,
        )
    ):
        raise ValueError("phase mean neighborhoods must match the snapshot support")
    weights = tuple(exact_or_represented_real(value, name) for name, value in (
        ("epi_weight", epi_weight), ("phase_weight", phase_weight),
        ("capacity_weight", capacity_weight),
    ))
    if any(value < 0 for value in weights):
        raise ValueError("joint response weights must be nonnegative")
    epi_weight, phase_weight, capacity_weight = weights
    phase_rate = ordered_vector(phase_rate_over_pi, "phase_rate_over_pi")
    capacity_rate = ordered_vector(capacity_rate, "capacity_rate")
    if len(phase_rate) != size or len(capacity_rate) != size:
        raise ValueError("joint response rate vectors must match the snapshot node order")

    transport = observe_support_transport_derivative(
        source, conductance_rates=(Fraction(0),) * len(source.conductance),
    )
    epi_response = tuple(epi_weight * value for value in transport.epi_gradient_rate)
    phase_response = tuple(
        phase_weight * dot(row, phase_rate) for row in geometry.scaled_source_jacobian
    )
    capacity_response = tuple(
        capacity_weight * value
        for value in _support_gradient(source.support_neighbors, capacity_rate)
    )
    pressure_rate = tuple(a + b + c for a, b, c in zip(
        epi_response, phase_response, capacity_response, strict=True,
    ))
    capacity_acceleration = tuple(a * b for a, b in zip(
        capacity_rate, source.stored_pressure, strict=True,
    ))
    pressure_acceleration = tuple(a * b for a, b in zip(
        source.capacity, pressure_rate, strict=True,
    ))
    acceleration = tuple(a + b for a, b in zip(
        capacity_acceleration, pressure_acceleration, strict=True,
    ))
    return JointNodalResponse(
        source, geometry, epi_weight, phase_weight, capacity_weight,
        phase_rate, capacity_rate, epi_response, phase_response, capacity_response,
        pressure_rate, capacity_acceleration, pressure_acceleration, acceleration,
    )
