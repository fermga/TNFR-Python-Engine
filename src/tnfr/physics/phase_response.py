"""Exact local derivatives of declared circular means and phase-stage merges.

Cosine Gram data describe unit planar phasors without approximating their
angles. These detached coefficients do not identify live phase gates, a
binary64 derivative, a fixed point or a complete operator trajectory.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real
from ._cycle_algebra import Matrix, Vector, ordered_vector

__all__ = ["PhaseResponseReference", "derive_phase_response"]


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
