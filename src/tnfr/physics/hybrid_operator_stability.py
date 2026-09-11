r"""Hybrid pure-EPI stability under explicitly bounded affine operator jumps.

The restricted diffusion certificates prove decay of a common disagreement
energy along the pure EPI channel of the nodal equation.  They do not cover the
instantaneous state changes made by an engine operator.  This module supplies
the missing, deliberately conditional, jump theorem without assigning gains
from an operator's U2 role.

Let ``h`` be the positive common metric, ``H = diag(h)``, and

``Q = I - 1 h.T / (h.T 1)``,
``V(x) = 1/2 ||Q x||_H**2``.

For a declared affine EPI reset ``J(x) = A x + b``, a finite multiplicative
bound ``V(J(x)) <= gamma V(x)`` for every ``x`` exists exactly when ``A 1`` and
``b`` are uniform vectors.  Under those conditions the sharp energy gain is

``gamma = ||H**(1/2) Q A Q H**(-1/2)||_2**2``.

Its binary64 SVD evaluation is reported only as a sharp *estimate*.  Exact
rational arithmetic first forms the weighted-Frobenius bound

``Gamma_F = sum_ij (h_i / h_j) (Q A Q)_ij**2 >= gamma``

formed from the represented binary64 coefficients.  Scalar quotient actions
(``Q A Q = c Q``) have the exact gain ``c**2``.  One-dimensional quotients are
also exact; two-dimensional quotients use a rational upper enclosure of their
algebraic largest eigenvalue.  Quotient dimensions three and four use exact
positive-semidefinite tests to bisect a rational upper bound, and larger
quotients retain the safe Frobenius bound.  Thus every precise gain bound is
proved and retained in rational arithmetic, while avoiding the
dimension-dependent ``n-1`` penalty that the Frobenius bound assigns even to
the identity map. Log-space composition first rounds each gain upward to an
exact 32-bit-significand dyadic rational. This keeps each
transcendental-series input bounded without weakening the upper-bound
guarantee.

If either condition fails, a consensus input has zero energy and some consensus
input is sent to positive disagreement, so the global gain is infinite.  This
zero-to-positive counterexample is retained in the certificate.

Between jumps, a fixed or exact-common-metric switching diffusion certificate
gives ``V' <= -r V``.  A finite hybrid word therefore satisfies

``V(T) <= exp(-r T_flow) product_k(Gamma_k) V(0)``.

If the same non-Zeno word is repeated and its multiplier is below one,
disagreement converges exponentially.  The separate conditions
``h.T A = h.T`` and ``h.T b = 0`` for every jump certify convergence specifically
to the *initial h-weighted consensus*.  Without them, disagreement may vanish
while the scalar consensus coordinate drifts; convergence to some other scalar
limit requires a separate analysis of that one-dimensional dynamics.

Scope
-----
The theorem is exact for a declared finite-dimensional affine reset and the
declared pure-EPI flow certificates.  Algebraic hypotheses and the conservative
reset gain are evaluated with rational arithmetic on the represented binary64
coefficients; rational inputs are first coerced to binary64.  Spectral estimates
and residuals are separate
tolerance-conditioned diagnostics.  An operator name contributes contract
metadata only: U2 labels and legacy policy multipliers in
:mod:`tnfr.physics.lyapunov` are not gain bounds.  Nonlinear clipping, phase,
independent pressure changes, topology/history mutation, REMESH, and the
derivation of an affine reset from a runtime operator remain outside the theorem
until separately certified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
from numbers import Real
import sys
from typing import Any, Iterable

from .._exact_time import (
    atanh_log_bounds as _atanh_log_bounds,
    exact_log_bounds as _exact_log_bounds,
    exp_unit_bounds as _exp_unit_bounds,
    exp_upper_float as _exp_upper_float,
    fraction_lower_float as _fraction_lower_float,
    fraction_upper_float as _fraction_upper_float,
    fraction_upper_signed_float as _fraction_upper_signed_float,
    materialize_nonnegative_time_sequence as _materialize_time_sequence,
)
from ..mathematics._weight_normalization import normalize_weights
from ..mathematics.unified_numerical import np
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._exact_linear_algebra import (
    exact_matrix_inverse as _exact_matrix_inverse,
    exact_symmetric_semidefinite as _exact_symmetric_semidefinite,
)
from ._exact_metric import (
    binary64_vectors_exactly_proportional as _exactly_proportional,
)
from ._helpers import finite_real_scalar
from .directed_diffusion import NormKind, induced_operator_norm
from .structural_diffusion import (
    HeterogeneousDiffusionStabilityCertificate,
    SwitchingDiffusionStabilityCertificate,
)

__all__ = [
    "AffineEPIJumpGainCertificate",
    "HybridEPIStabilityCertificate",
    "certify_affine_epi_jump_gain",
    "compose_hybrid_epi_stability",
]


# For a positive gain g in [2**e, 2**(e+1)), upward quantization to p=32
# significand bits has quantum q=2**(e-(p-1)).  Its strict ceiling error is
# below q <= g*2**(-(p-1)), hence g_hat < g*(1 + 2**-31).  Across k positive
# gains the product inflation is therefore below (1 + 2**-31)**k.  This may
# make the contraction test more conservative but cannot make it unsound, and
# it bounds rational growth in repeated log-space composition.  The precise
# quotient gains and their product remain exposed separately in the
# certificate.
_COMPOSITION_GAIN_SIGNIFICAND_BITS = 32
# Exact generalized-eigenvalue bisection is deliberately limited to small
# quotient spaces.  Scalar quotient actions are recognized exactly in every
# dimension; larger non-scalar maps retain the weighted-Frobenius fallback.
_EXACT_QUOTIENT_BISECTION_STEPS = 16
_EXACT_QUOTIENT_REFINEMENT_MAX_DIMENSION = 4
_EXACT_SQRT_UPPER_BITS = 64


_JUMP_SCOPE = (
    "EXACT affine-reset theorem on the declared common pure-EPI disagreement "
    "metric. Exact rational algebra supplies a quotient energy-gain upper "
    "bound, with the weighted-Frobenius value retained as a conservative "
    "fallback; binary64 spectral estimates and residuals are "
    "diagnostic only. The canonical operator name supplies metadata only and "
    "does not determine the gain. "
    "Runtime-map identification, nonlinear clipping, phase, independent "
    "pressure, topology/history changes, and REMESH remain outside scope."
)

_HYBRID_SCOPE = (
    "CONDITIONAL hybrid pure-EPI theorem: certified affine resets are composed "
    "with one existing fixed or exact-common-metric switching diffusion "
    "certificate. A repeated-word conclusion additionally assumes the same "
    "finite word and positive flow duration repeat without Zeno accumulation. "
    "Disagreement convergence and convergence to the initial weighted "
    "consensus are reported separately. Precise rational reset-gain bounds are "
    "retained, while their upward 32-bit dyadic enclosures provide bounded-"
    "complexity exact rational inputs to the logarithm. Exact rational "
    "enclosures of the flow rate, durations, logarithm, and exponential "
    "support the finite and "
    "repeated-word bounds. "
    "Canonical jumps have zero duration in the separate event schedule. "
    "Runtime temporal realization, any nonzero operator duration, other "
    "scalar consensus dynamics, and unsupplied dynamics remain outside scope."
)


def _fraction(value: Any, name: str) -> Fraction:
    """Return the exact rational value of the represented binary64 scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must contain finite real values, not booleans")
    try:
        source_is_nonzero = bool(value != 0)
        floating = float(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must contain finite real values") from exc
    if not math.isfinite(floating) or (floating == 0.0 and source_is_nonzero):
        raise ValueError(f"{name} must contain finite real values")
    return Fraction.from_float(floating)


def _real_array(
    value: Any,
    name: str,
    *,
    ndim: int,
    shape: tuple[int, ...] | None = None,
) -> tuple[Any, tuple[Fraction, ...]]:
    """Validate an array once and retain exact represented scalar values."""
    try:
        objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if objects.ndim != ndim:
        noun = "matrix" if ndim == 2 else "vector"
        raise ValueError(f"{name} must be a {noun}")
    if shape is not None and objects.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    exact = tuple(_fraction(item, name) for item in objects.flat)
    try:
        array = np.asarray([float(item) for item in exact], dtype=float).reshape(
            objects.shape
        )
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return np.array(array, dtype=float, copy=True), exact


def _relative_tolerance(value: Any) -> float:
    try:
        tolerance = finite_real_scalar(value, "tolerance")
    except ValueError as exc:
        raise ValueError(
            "tolerance must be a finite real in the open interval (0, 1)"
        ) from exc
    if not 0.0 < tolerance < 1.0:
        raise ValueError(
            "tolerance must be a finite real in the open interval (0, 1)"
        )
    return tolerance


def _nonnegative_scalar_with_exact(
    value: Any, name: str
) -> tuple[float, Fraction]:
    """Return an upward-rounded float and the exact supplied real value."""
    exact = _fraction(value, name)
    if exact < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return _fraction_upper_float(exact), exact


def _bounded_composition_gain_factor(
    exact_gain: Fraction,
) -> Fraction:
    """Round a gain upward to a bounded-significand dyadic rational.

    For ``2**e <= g < 2**(e+1)`` and ``p`` significand bits, the quantum is
    ``q = 2**(e-(p-1))``.  Exact integer ceiling gives
    ``g <= g_hat < g + q <= g * (1 + 2**(-(p-1)))``.  Thus every factor and
    any finite product remain conservative, while the normalized significand
    has at most ``p`` bits.  The precise gain remains certificate data.
    """
    if exact_gain < 0:
        raise ValueError("an exact composition gain must be nonnegative")
    if exact_gain == 0:
        return Fraction(0)

    exponent = exact_gain.numerator.bit_length() - exact_gain.denominator.bit_length()
    power = (
        Fraction(1 << exponent)
        if exponent >= 0
        else Fraction(1, 1 << -exponent)
    )
    if exact_gain < power:
        exponent -= 1

    scale_exponent = exponent - (_COMPOSITION_GAIN_SIGNIFICAND_BITS - 1)
    scale = (
        Fraction(1 << scale_exponent)
        if scale_exponent >= 0
        else Fraction(1, 1 << -scale_exponent)
    )
    scaled = exact_gain / scale
    units = -(-scaled.numerator // scaled.denominator)
    factor = units * scale
    if factor < exact_gain:
        raise RuntimeError("composition-gain quantization lost its upper bound")
    return factor


def _exact_matrix_product(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Matrix product over exact rationals."""
    rows = len(left)
    inner = len(right)
    columns = len(right[0])
    return tuple(
        tuple(
            sum(
                (left[i][k] * right[k][j] for k in range(inner)),
                Fraction(0),
            )
            for j in range(columns)
        )
        for i in range(rows)
    )


def _exact_quotient_basis(
    weights: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return columns spanning ``h.T y = 0`` with bounded coefficients."""

    dimension = len(weights)
    pivot = max(range(dimension), key=weights.__getitem__)
    free = tuple(index for index in range(dimension) if index != pivot)
    basis = [
        [Fraction(0) for _ in range(dimension - 1)]
        for _ in range(dimension)
    ]
    for column, index in enumerate(free):
        basis[index][column] = Fraction(1)
        basis[pivot][column] = -weights[index] / weights[pivot]
    return tuple(tuple(row) for row in basis)


def _exact_weighted_gram(
    left: tuple[tuple[Fraction, ...], ...],
    right: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Return ``left.T diag(weights) right`` exactly."""

    columns_left = len(left[0])
    columns_right = len(right[0])
    return tuple(
        tuple(
            sum(
                (
                    weights[row] * left[row][i] * right[row][j]
                    for row in range(len(weights))
                ),
                Fraction(0),
            )
            for j in range(columns_right)
        )
        for i in range(columns_left)
    )


def _exact_scalar_quotient_gain(
    quotient_map: tuple[tuple[Fraction, ...], ...],
    projection: tuple[tuple[Fraction, ...], ...],
) -> Fraction | None:
    """Return ``c**2`` when ``Q A Q`` acts as ``c I`` on the quotient."""

    scale: Fraction | None = None
    for mapped_row, projection_row in zip(quotient_map, projection):
        for mapped, projected in zip(mapped_row, projection_row):
            if projected == 0:
                if mapped != 0:
                    return None
                continue
            candidate = mapped / projected
            if scale is None:
                scale = candidate
            elif candidate != scale:
                return None
    if scale is None:  # pragma: no cover - Q is nonzero for dimension >= 2
        raise RuntimeError("internal error: consensus quotient is empty")
    return scale * scale


def _exact_sqrt_upper(value: Fraction) -> Fraction:
    """Return a rational upper enclosure of a nonnegative square root."""

    if value < 0:
        raise ValueError("square-root enclosure requires a nonnegative value")
    if value == 0:
        return Fraction(0)
    scale = 1 << _EXACT_SQRT_UPPER_BITS
    scaled_numerator = value.numerator * scale * scale
    root = math.isqrt(scaled_numerator // value.denominator)
    if root * root * value.denominator < scaled_numerator:
        root += 1
    return Fraction(root, scale)


def _exact_quotient_energy_gain_upper_bound(
    quotient_map: tuple[tuple[Fraction, ...], ...],
    projection: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    frobenius_bound: Fraction,
) -> Fraction:
    r"""Prove an upper bound for ``max V(B y)/V(y)`` on ``h.T y=0``.

    ``B = Q A Q``.  Scalar quotient actions and one-dimensional quotients are
    exact.  Dimension two encloses the algebraic square root rationally.
    Dimensions three and four bisect ``lambda G - K >= 0`` using exact
    positive-semidefinite tests, where ``G = R.T H R`` and
    ``K = (B R).T H (B R)``.  The initial upper endpoint is the smaller of the
    weighted-Frobenius bound and an exact induced infinity-norm bound on
    ``G^-1 K``.  No floating eigensolver participates.
    """

    scalar_gain = _exact_scalar_quotient_gain(quotient_map, projection)
    if scalar_gain is not None:
        return scalar_gain

    quotient_dimension = len(weights) - 1
    if quotient_dimension > _EXACT_QUOTIENT_REFINEMENT_MAX_DIMENSION:
        return frobenius_bound

    basis = _exact_quotient_basis(weights)
    mapped_basis = _exact_matrix_product(quotient_map, basis)
    metric = _exact_weighted_gram(basis, basis, weights)
    output = _exact_weighted_gram(mapped_basis, mapped_basis, weights)
    if quotient_dimension == 1:
        return min(frobenius_bound, output[0][0] / metric[0][0])

    transfer = _exact_matrix_product(_exact_matrix_inverse(metric), output)
    if quotient_dimension == 2:
        trace = transfer[0][0] + transfer[1][1]
        determinant = (
            transfer[0][0] * transfer[1][1]
            - transfer[0][1] * transfer[1][0]
        )
        discriminant = trace * trace - 4 * determinant
        if discriminant < 0:  # pragma: no cover - exact PSD pencil invariant
            raise RuntimeError("internal error: affine quotient has complex energy")
        algebraic_upper = (trace + _exact_sqrt_upper(discriminant)) / 2
        return min(frobenius_bound, algebraic_upper)

    induced_upper = max(
        sum((abs(value) for value in row), Fraction(0))
        for row in transfer
    )
    upper = min(frobenius_bound, induced_upper)
    if upper == 0:
        return Fraction(0)

    def upper_is_valid(candidate: Fraction) -> bool:
        shifted = tuple(
            tuple(
                candidate * metric[i][j] - output[i][j]
                for j in range(quotient_dimension)
            )
            for i in range(quotient_dimension)
        )
        return _exact_symmetric_semidefinite(shifted)

    # Both initial candidates are analytic upper bounds.  Retain the
    # Frobenius endpoint if a future implementation error violates that fact.
    if not upper_is_valid(upper):  # pragma: no cover - defensive invariant
        upper = frobenius_bound
        if not upper_is_valid(upper):
            raise RuntimeError("internal error: invalid affine gain upper bound")

    lower = max(
        output[index][index] / metric[index][index]
        for index in range(quotient_dimension)
    )
    if lower >= upper:
        return upper
    for _ in range(_EXACT_QUOTIENT_BISECTION_STEPS):
        midpoint = (lower + upper) / 2
        if upper_is_valid(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return upper


def _exact_matrix_vector(
    matrix: tuple[tuple[Fraction, ...], ...],
    vector: tuple[Fraction, ...],
) -> tuple[Fraction, ...]:
    """Matrix-vector product over exact rationals."""
    return tuple(
        sum(
            (coefficient * value for coefficient, value in zip(row, vector)),
            Fraction(0),
        )
        for row in matrix
    )


def _exact_projection(
    weights: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    """Exact ``I - 1 h.T/(h.T 1)`` for a positive rational metric."""
    total = sum(weights, Fraction(0))
    dimension = len(weights)
    return tuple(
        tuple(
            (Fraction(1) if i == j else Fraction(0)) - weights[j] / total
            for j in range(dimension)
        )
        for i in range(dimension)
    )


def _exact_weighted_frobenius_energy_bound(
    quotient_map: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
) -> Fraction:
    """Exact squared weighted Frobenius norm of a rational map."""
    return sum(
        (
            weights[i] * coefficient * coefficient / weights[j]
            for i, row in enumerate(quotient_map)
            for j, coefficient in enumerate(row)
        ),
        Fraction(0),
    )


def _exact_normalized_energy(
    vector: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
) -> Fraction:
    """Exact disagreement energy in the normalized version of ``weights``."""
    total = sum(weights, Fraction(0))
    return sum(
        (weight * value * value for weight, value in zip(weights, vector)),
        Fraction(0),
    ) / (2 * total)


def _node_order(values: Iterable[Any] | None, dimension: int) -> tuple[Any, ...]:
    if values is None:
        return tuple(range(dimension))
    if isinstance(values, (str, bytes)):
        raise TypeError("nodes must be an iterable of unique node identifiers")
    try:
        nodes = tuple(values)
    except TypeError as exc:
        raise TypeError("nodes must be an iterable of unique node identifiers") from exc
    if len(nodes) != dimension:
        raise ValueError("nodes length must equal the affine-map dimension")
    try:
        if len(set(nodes)) != len(nodes):
            raise ValueError("nodes must contain unique identifiers")
    except TypeError as exc:
        raise TypeError("node identifiers must be hashable") from exc
    return nodes


def _readonly_array(value: Any) -> Any:
    """Detach one NumPy payload and freeze it with its certificate."""
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_float_signature(value: Any) -> tuple[str, ...]:
    """Encode a finite array exactly enough for an in-memory proof stamp."""

    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("proof data must remain finite")
    return tuple(float(item).hex() for item in array.flat)


def _affine_jump_proof_stamp(
    nodes: tuple[Any, ...],
    linear_map: Any,
    offset: Any,
    metric_weights: Any,
    exact_consensus: bool,
    exact_frobenius_bound: Fraction,
    exact_quotient_bound: Fraction,
    finite_global_gain: bool,
    global_gain_bound: float,
    composition_bound: float,
    exact_mean_preservation: bool,
    counterexample_level: float | None,
    exact_counterexample_energy: Fraction | None,
) -> tuple[Any, ...]:
    """Snapshot every input and result used by affine theorem properties."""

    return (
        "affine_epi_jump_gain_v2",
        structural_proof_signature(tuple(nodes)),
        np.asarray(linear_map).shape,
        _finite_float_signature(linear_map),
        np.asarray(offset).shape,
        _finite_float_signature(offset),
        np.asarray(metric_weights).shape,
        _finite_float_signature(metric_weights),
        bool(exact_consensus),
        Fraction(exact_frobenius_bound),
        Fraction(exact_quotient_bound),
        bool(finite_global_gain),
        float(global_gain_bound).hex(),
        float(composition_bound).hex(),
        bool(exact_mean_preservation),
        counterexample_level,
        (
            None
            if exact_counterexample_energy is None
            else Fraction(exact_counterexample_energy)
        ),
    )


def _weighted_norm(vector: Any, weights: Any, name: str) -> float:
    """Scale-safe norm induced by positive normalized diagonal weights."""
    values = np.asarray(vector, dtype=float)
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normalized = values / scale
            result = scale * math.sqrt(float(np.sum(weights * normalized**2)))
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _weighted_energy(vector: Any, weights: Any, name: str) -> float:
    norm = _weighted_norm(vector, weights, name)
    try:
        energy = 0.5 * norm * norm
    except OverflowError as exc:  # pragma: no cover - Python float multiplication
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not math.isfinite(energy):
        raise ValueError(f"{name} exceeds finite floating-point range")
    if energy == 0.0 and norm != 0.0:
        raise ValueError(f"{name} is below nonzero floating-point range")
    return energy


def _within_relative_tolerance(residual: float, scale: float, tolerance: float) -> bool:
    return bool(residual == 0.0 or residual / max(1.0, scale) <= tolerance)


@dataclass(frozen=True, slots=True)
class AffineEPIJumpGainCertificate:
    """Global common-metric gain of one declared affine EPI reset.

    ``exact_quotient_energy_gain_upper_bound`` is the rational theorem factor
    retained in the precise hybrid product.  It is not passed directly to the
    logarithm: ``HybridEPIStabilityCertificate`` exposes the separate upward
    dyadic factors actually used by log-space composition.
    ``energy_gain_bound_for_composition`` is the backward-compatible binary64
    display of the precise quotient bound when the global theorem applies.
    ``exact_weighted_frobenius_energy_bound`` is the always-valid fallback and
    remains visible for comparison.  A supplied declaration is certified
    against the quotient bound; the separate
    ``declared_bound_certified_by_frobenius`` flag records whether it also
    clears the generally stronger Frobenius test.
    """

    operator_name: str
    glyph: str
    primary_channel: str
    operator_scale: str
    nodes: tuple[Any, ...]
    linear_map: Any
    offset: Any
    metric_weights: Any
    normalized_metric_weights: Any
    consensus_linear_residual: float
    consensus_offset_residual: float
    exact_consensus_subspace_preservation: bool
    consensus_subspace_preservation_within_tolerance: bool
    sharp_quotient_operator_norm_estimate: float | None
    sharp_quotient_energy_gain_estimate: float | None
    exact_weighted_frobenius_energy_bound: Fraction
    weighted_frobenius_energy_bound: float
    exact_quotient_energy_gain_upper_bound: Fraction
    quotient_energy_gain_upper_bound: float
    global_energy_gain_bound: float
    finite_global_energy_gain: bool
    declared_energy_gain_bound: float | None
    exact_declared_energy_gain_bound: Fraction | None
    declared_energy_gain_bound_certified: bool | None
    declared_bound_certified_by_frobenius: bool | None
    declared_bound_within_tolerance: bool | None
    energy_gain_bound_for_composition: float
    weighted_mean_linear_residual: float
    weighted_mean_offset_residual: float
    exact_weighted_mean_preservation: bool
    weighted_mean_preservation_within_tolerance: bool
    consensus_counterexample_level: float | None
    exact_consensus_counterexample_energy_after: Fraction | None
    consensus_counterexample_energy_after: float | None
    tolerance: float
    scope: str
    _proof_stamp: tuple[Any, ...] = field(repr=False, compare=False)

    def _proof_fields_are_intact(self) -> bool:
        """Detect ordinary replacement or mutation of decisive proof fields."""

        try:
            expected = _affine_jump_proof_stamp(
                self.nodes,
                self.linear_map,
                self.offset,
                self.metric_weights,
                self.exact_consensus_subspace_preservation,
                self.exact_weighted_frobenius_energy_bound,
                self.exact_quotient_energy_gain_upper_bound,
                self.finite_global_energy_gain,
                self.global_energy_gain_bound,
                self.energy_gain_bound_for_composition,
                self.exact_weighted_mean_preservation,
                self.consensus_counterexample_level,
                self.exact_consensus_counterexample_energy_after,
            )
            observed = object.__getattribute__(self, "_proof_stamp")
        except BaseException:
            return False
        return proof_stamps_are_identical(observed, expected)

    @property
    def supports_global_gain_theorem(self) -> bool:
        """Whether this reset has a usable finite multiplicative energy gain."""
        return bool(
            self._proof_fields_are_intact()
            and self.finite_global_energy_gain
            and math.isfinite(self.energy_gain_bound_for_composition)
        )

    @property
    def preserves_initial_weighted_consensus(self) -> bool:
        """Whether the reset preserves the initial common-metric consensus."""
        return bool(
            self._proof_fields_are_intact()
            and self.exact_consensus_subspace_preservation
            and self.exact_weighted_mean_preservation
        )


@dataclass(frozen=True, slots=True)
class HybridEPIStabilityCertificate:
    """Composition of pure-EPI flow decay and affine reset gains.

    ``exact_cumulative_jump_energy_gain_bound`` is the product of the precise
    rational quotient-gain bounds.  ``exact_log_composition_gain_factors``
    exposes the distinct upward dyadic factors actually supplied to the
    rational logarithm enclosures.  Their bounded complexity can introduce
    conservative slack but cannot understate the precise product.
    ``log_contraction_decision_margin`` remains zero as a compatibility field
    because contraction uses the exact upper log's sign.
    """

    nodes: tuple[Any, ...]
    flow_certificate_kind: str
    flow_hypotheses_pass: bool
    normalized_metric_weights: Any
    flow_energy_decay_rate: float
    flow_durations: tuple[float, ...]
    total_flow_duration: float
    jumps: tuple[AffineEPIJumpGainCertificate, ...]
    jump_energy_gain_bounds: tuple[float, ...]
    cumulative_jump_log_energy_gain: float
    flow_log_energy_decay_lower_bound: float
    net_log_energy_gain_bound: float
    energy_multiplier_bound: float
    exact_cumulative_jump_energy_gain_bound: Fraction | None
    exact_log_composition_gain_factors: tuple[Fraction, ...] | None
    exact_flow_log_energy_decay_lower_bound: Fraction
    exact_net_log_energy_gain_upper_bound: Fraction | None
    log_contraction_decision_margin: float
    numerical_conditions: tuple[tuple[str, bool], ...]
    finite_horizon_disagreement_bound_certified: bool
    disagreement_contracts_over_declared_horizon: bool
    initial_weighted_mean_preserved: bool
    repeat_schedule_declared: bool
    repeated_schedule_disagreement_convergence_certified: bool | None
    repeated_schedule_initial_weighted_consensus_convergence_certified: bool | None
    asymptotic_disagreement_energy_decay_rate: float | None
    tolerance: float
    scope: str

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        """Names of failed flow or jump hypotheses."""
        return tuple(name for name, passed in self.numerical_conditions if not passed)


def certify_affine_epi_jump_gain(
    operator: str,
    linear_map: Any,
    metric_weights: Any,
    *,
    offset: Any = None,
    nodes: Iterable[Any] | None = None,
    declared_energy_gain_bound: Any = None,
    tolerance: float = 1e-10,
) -> AffineEPIJumpGainCertificate:
    r"""Certify the common-metric gain of ``x+ = A x- + b``.

    ``operator`` is resolved through the canonical contract registry, but its
    contract and U2 role do not constrain ``A``, ``b``, or the returned gain.
    The caller must derive the affine realization independently.  Non-affine
    runtime branches require a different theorem.

    Exact Booleans refer to exact rational values of the represented binary64
    coefficients.  Rational and integer inputs are first coerced to binary64.
    Residual Booleans are separate tolerance-conditioned numerical diagnostics.
    """
    if not isinstance(operator, str):
        raise TypeError("operator must be a canonical name, function name, or glyph")
    # Import lazily so this physics module can be re-exported without creating
    # an operators -> physics -> operators initialization cycle.
    from ..operators.operator_contracts import contract_for

    try:
        contract = contract_for(operator)
    except KeyError as exc:
        raise ValueError(f"unknown canonical operator {operator!r}") from exc
    tol = _relative_tolerance(tolerance)

    matrix, matrix_exact_flat = _real_array(
        linear_map, "linear_map", ndim=2
    )
    if matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("linear_map must be a nonempty square matrix")
    dimension = int(matrix.shape[0])
    if dimension < 2:
        raise ValueError("operator-gain certification requires at least two nodes")
    node_tuple = _node_order(nodes, dimension)

    metric, metric_exact_flat = _real_array(
        metric_weights, "metric_weights", ndim=1, shape=(dimension,)
    )
    metric_exact = tuple(metric_exact_flat)
    if any(value <= 0 for value in metric_exact):
        raise ValueError("metric_weights must be finite and positive")
    normalized_metric, _, _ = normalize_weights(metric)
    if (
        not np.all(np.isfinite(normalized_metric))
        or np.any(normalized_metric <= 0.0)
    ):
        raise ValueError(
            "metric normalization exceeds floating-point dynamic range"
        )

    if offset is None:
        offset_array = np.zeros(dimension, dtype=float)
        offset_exact = tuple(Fraction(0) for _ in range(dimension))
    else:
        offset_array, offset_exact_flat = _real_array(
            offset, "offset", ndim=1, shape=(dimension,)
        )
        offset_exact = tuple(offset_exact_flat)

    if declared_energy_gain_bound is None:
        declared_bound = None
        declared_bound_exact = None
    else:
        declared_bound, declared_bound_exact = _nonnegative_scalar_with_exact(
            declared_energy_gain_bound, "declared_energy_gain_bound"
        )

    matrix_exact = tuple(
        tuple(matrix_exact_flat[i * dimension : (i + 1) * dimension])
        for i in range(dimension)
    )
    exact_projection = _exact_projection(metric_exact)
    exact_ones = tuple(Fraction(1) for _ in range(dimension))
    exact_mapped_consensus = _exact_matrix_vector(matrix_exact, exact_ones)
    exact_linear_defect = _exact_matrix_vector(
        exact_projection, exact_mapped_consensus
    )
    exact_offset_defect = _exact_matrix_vector(exact_projection, offset_exact)
    exact_linear = all(value == 0 for value in exact_linear_defect)
    exact_offset = all(value == 0 for value in exact_offset_defect)
    exact_consensus = exact_linear and exact_offset

    exact_quotient_map = _exact_matrix_product(
        _exact_matrix_product(exact_projection, matrix_exact), exact_projection
    )
    exact_frobenius_bound = _exact_weighted_frobenius_energy_bound(
        exact_quotient_map, metric_exact
    )
    frobenius_bound = _fraction_upper_float(exact_frobenius_bound)
    exact_quotient_bound = _exact_quotient_energy_gain_upper_bound(
        exact_quotient_map,
        exact_projection,
        metric_exact,
        exact_frobenius_bound,
    )
    if exact_quotient_bound > exact_frobenius_bound:
        raise RuntimeError(
            "internal error: quotient gain bound exceeds Frobenius fallback"
        )
    quotient_bound = _fraction_upper_float(exact_quotient_bound)

    ones = np.ones(dimension, dtype=float)
    projection = np.eye(dimension) - np.outer(ones, normalized_metric)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mapped_consensus = matrix @ ones
            linear_defect = projection @ mapped_consensus
            offset_defect = projection @ offset_array
        linear_residual = _weighted_norm(
            linear_defect, normalized_metric, "consensus linear residual"
        )
        offset_residual = _weighted_norm(
            offset_defect, normalized_metric, "consensus offset residual"
        )
        consensus_linear_scale = max(
            1.0,
            _weighted_norm(
                mapped_consensus, normalized_metric, "consensus linear scale"
            ),
        )
        consensus_offset_scale = max(
            1.0,
            _weighted_norm(
                offset_array, normalized_metric, "consensus offset scale"
            ),
        )
        consensus_within = bool(
            _within_relative_tolerance(
                linear_residual, consensus_linear_scale, tol
            )
            and _within_relative_tolerance(
                offset_residual, consensus_offset_scale, tol
            )
        )
    except (FloatingPointError, OverflowError, ValueError):
        linear_residual = float("inf")
        offset_residual = float("inf")
        consensus_within = False

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            quotient_map = projection @ matrix @ projection
        quotient_norm = induced_operator_norm(
            quotient_map, kind=NormKind.STATIONARY, pi=normalized_metric
        )
        quotient_gain = quotient_norm * quotient_norm
        if not math.isfinite(quotient_norm) or not math.isfinite(quotient_gain):
            raise FloatingPointError
        if quotient_gain == 0.0 and quotient_norm != 0.0:
            raise FloatingPointError
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        quotient_norm = None
        quotient_gain = None

    global_gain_bound = quotient_bound if exact_consensus else float("inf")
    if declared_bound_exact is None:
        declared_certified = None
        declared_by_frobenius = None
        declared_within = None
    elif not exact_consensus:
        declared_certified = False
        declared_by_frobenius = False
        declared_within = False
    else:
        declared_certified = declared_bound_exact >= exact_quotient_bound
        declared_by_frobenius = declared_bound_exact >= exact_frobenius_bound
        if declared_certified:
            declared_within = True
        else:
            declared_difference = _fraction_upper_float(
                exact_quotient_bound - declared_bound_exact
            )
            declared_within = _within_relative_tolerance(
                declared_difference,
                max(quotient_bound, declared_bound),
                tol,
            )
    composition_bound = quotient_bound if exact_consensus else float("inf")

    weighted_row = tuple(
        sum(
            (metric_exact[i] * matrix_exact[i][j] for i in range(dimension)),
            Fraction(0),
        )
        for j in range(dimension)
    )
    weighted_offset = sum(
        (weight * value for weight, value in zip(metric_exact, offset_exact)),
        Fraction(0),
    )
    exact_mean = weighted_row == metric_exact and weighted_offset == 0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean_linear_defect = normalized_metric @ matrix - normalized_metric
            mean_offset_defect = float(normalized_metric @ offset_array)
        mean_linear_residual = float(
            np.max(np.abs(mean_linear_defect), initial=0.0)
        )
        mean_offset_residual = abs(mean_offset_defect)
        mean_linear_scale = max(
            1.0,
            float(np.max(np.abs(normalized_metric @ matrix), initial=0.0)),
            float(np.max(np.abs(normalized_metric), initial=0.0)),
        )
        mean_offset_scale = max(
            1.0, float(normalized_metric @ np.abs(offset_array))
        )
        mean_within = bool(
            _within_relative_tolerance(
                mean_linear_residual, mean_linear_scale, tol
            )
            and _within_relative_tolerance(
                mean_offset_residual, mean_offset_scale, tol
            )
        )
    except (FloatingPointError, OverflowError, ValueError):
        mean_linear_residual = float("inf")
        mean_offset_residual = float("inf")
        mean_within = False

    counter_level: float | None = None
    exact_counter_energy: Fraction | None = None
    counter_energy: float | None = None
    if not exact_consensus:
        counter_level = 0.0 if not exact_offset else 1.0
        exact_counter_level = Fraction(int(counter_level))
        exact_counter_input = tuple(
            exact_counter_level for _ in range(dimension)
        )
        exact_counter_output = tuple(
            mapped + shift
            for mapped, shift in zip(
                _exact_matrix_vector(matrix_exact, exact_counter_input),
                offset_exact,
            )
        )
        exact_counter_disagreement = _exact_matrix_vector(
            exact_projection, exact_counter_output
        )
        exact_counter_energy = _exact_normalized_energy(
            exact_counter_disagreement, metric_exact
        )
        if exact_counter_energy <= 0:
            raise RuntimeError(
                "internal error: exact consensus failure lacks a counterexample"
            )
        counter_energy = _fraction_upper_float(exact_counter_energy)

    frozen_matrix = _readonly_array(matrix)
    frozen_offset = _readonly_array(offset_array)
    frozen_metric = _readonly_array(metric)
    proof_stamp = _affine_jump_proof_stamp(
        node_tuple,
        frozen_matrix,
        frozen_offset,
        frozen_metric,
        exact_consensus,
        exact_frobenius_bound,
        exact_quotient_bound,
        exact_consensus,
        global_gain_bound,
        composition_bound,
        exact_mean,
        counter_level,
        exact_counter_energy,
    )

    return AffineEPIJumpGainCertificate(
        operator_name=contract.english_name,
        glyph=contract.glyph,
        primary_channel=contract.primary_channel.value,
        operator_scale=contract.scale.value,
        nodes=node_tuple,
        linear_map=frozen_matrix,
        offset=frozen_offset,
        metric_weights=frozen_metric,
        normalized_metric_weights=_readonly_array(normalized_metric),
        consensus_linear_residual=linear_residual,
        consensus_offset_residual=offset_residual,
        exact_consensus_subspace_preservation=exact_consensus,
        consensus_subspace_preservation_within_tolerance=consensus_within,
        sharp_quotient_operator_norm_estimate=(
            None if quotient_norm is None else float(quotient_norm)
        ),
        sharp_quotient_energy_gain_estimate=(
            None if quotient_gain is None else float(quotient_gain)
        ),
        exact_weighted_frobenius_energy_bound=exact_frobenius_bound,
        weighted_frobenius_energy_bound=float(frobenius_bound),
        exact_quotient_energy_gain_upper_bound=exact_quotient_bound,
        quotient_energy_gain_upper_bound=float(quotient_bound),
        global_energy_gain_bound=float(global_gain_bound),
        finite_global_energy_gain=exact_consensus,
        declared_energy_gain_bound=declared_bound,
        exact_declared_energy_gain_bound=declared_bound_exact,
        declared_energy_gain_bound_certified=declared_certified,
        declared_bound_certified_by_frobenius=declared_by_frobenius,
        declared_bound_within_tolerance=declared_within,
        energy_gain_bound_for_composition=float(composition_bound),
        weighted_mean_linear_residual=mean_linear_residual,
        weighted_mean_offset_residual=mean_offset_residual,
        exact_weighted_mean_preservation=exact_mean,
        weighted_mean_preservation_within_tolerance=mean_within,
        consensus_counterexample_level=counter_level,
        exact_consensus_counterexample_energy_after=exact_counter_energy,
        consensus_counterexample_energy_after=counter_energy,
        tolerance=tol,
        scope=_JUMP_SCOPE,
        _proof_stamp=proof_stamp,
    )


def _flow_data(
    certificate: Any,
) -> tuple[str, tuple[Any, ...], Any, Any, float, bool, bool]:
    """Read the common metric and energy rate from an existing certificate."""
    if isinstance(certificate, HeterogeneousDiffusionStabilityCertificate):
        if not certificate._proof_fields_are_intact():
            raise ValueError(
                "flow certificate proof fields were replaced or mutated"
            )
        weights, _, _ = normalize_weights(certificate.metric_weights)
        return (
            "fixed_heterogeneous_diffusion",
            tuple(certificate.nodes),
            np.asarray(certificate.metric_weights, dtype=float).copy(),
            weights,
            float(certificate.certified_exponential_rate_lower_bound),
            bool(certificate.is_certified),
            bool(certificate.exact_weighted_mean_preservation),
        )
    if isinstance(certificate, SwitchingDiffusionStabilityCertificate):
        if not certificate._proof_fields_are_intact():
            raise ValueError(
                "flow certificate proof fields were replaced or mutated"
            )
        weights = np.asarray(
            certificate.normalized_metric_weights, dtype=float
        ).copy()
        return (
            "exact_common_metric_switching_diffusion",
            tuple(certificate.nodes),
            np.asarray(certificate.reference_metric_weights, dtype=float).copy(),
            weights,
            float(
                certificate.certified_uniform_exponential_rate_lower_bound
            ),
            bool(certificate.supports_exact_switching_theorem),
            bool(certificate.exact_common_weighted_mean_preservation),
        )
    raise TypeError(
        "flow_certificate must be a heterogeneous or switching diffusion "
        "stability certificate"
    )


def _materialize_jumps(
    values: Iterable[AffineEPIJumpGainCertificate],
) -> tuple[AffineEPIJumpGainCertificate, ...]:
    if isinstance(values, (str, bytes, AffineEPIJumpGainCertificate)):
        raise TypeError("jumps must be an iterable of jump-gain certificates")
    try:
        jumps = tuple(values)
    except TypeError as exc:
        raise TypeError("jumps must be an iterable of jump-gain certificates") from exc
    if any(not isinstance(jump, AffineEPIJumpGainCertificate) for jump in jumps):
        raise TypeError("jumps must contain only jump-gain certificates")
    return jumps


def _materialize_durations(values: Iterable[Any], expected: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("flow_durations must be an iterable of nonnegative times")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "flow_durations must be an iterable of nonnegative times"
        ) from exc
    if len(raw) != expected:
        raise ValueError("flow_durations must contain one more entry than jumps")
    try:
        durations, _ = _materialize_time_sequence(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "flow_durations must contain finite nonnegative real values"
        ) from exc
    try:
        total = math.fsum(durations)
    except OverflowError as exc:
        raise ValueError(
            "total flow duration exceeds finite floating-point range"
        ) from exc
    if not math.isfinite(total):
        raise ValueError("total flow duration exceeds finite floating-point range")
    return durations


def compose_hybrid_epi_stability(
    flow_certificate: (
        HeterogeneousDiffusionStabilityCertificate
        | SwitchingDiffusionStabilityCertificate
    ),
    jumps: Iterable[AffineEPIJumpGainCertificate],
    flow_durations: Iterable[Any],
    *,
    repeat_schedule: bool = False,
    tolerance: float = 1e-10,
) -> HybridEPIStabilityCertificate:
    r"""Compose a finite hybrid energy bound from existing certificates.

    ``flow_durations`` has one more entry than ``jumps``: time before the first
    reset, between resets, and after the last reset.  The scalar bound depends
    only on their sum, while retaining the full tuple makes the event timeline
    explicit.  ``repeat_schedule=True`` asks for the corollary obtained by
    repeating that same positive-duration word indefinitely.  The contraction
    decision uses exact rational enclosures of the represented inputs. Each
    precise reset gain is retained in the result and rounded upward to a
    32-bit dyadic significand only for the rational log enclosure;
    ``tolerance`` affects only the accompanying numerical diagnostics.
    """
    tol = _relative_tolerance(tolerance)
    if not isinstance(repeat_schedule, bool):
        raise TypeError("repeat_schedule must be a boolean")
    (
        kind,
        nodes,
        reference_metric,
        metric,
        rate,
        flow_pass,
        flow_mean_preserved,
    ) = _flow_data(flow_certificate)
    if (
        metric.shape != (len(nodes),)
        or not np.all(np.isfinite(metric))
        or np.any(metric <= 0.0)
    ):
        raise ValueError("flow certificate does not expose a positive finite metric")
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("flow certificate must expose a positive finite energy rate")

    supplied_jumps = _materialize_jumps(jumps)
    durations = _materialize_durations(flow_durations, len(supplied_jumps) + 1)
    total_duration = math.fsum(durations)
    if repeat_schedule and total_duration <= 0.0:
        raise ValueError(
            "a repeated hybrid schedule requires positive flow duration to "
            "exclude Zeno accumulation"
        )

    conditions: list[tuple[str, bool]] = [("pure_epi_flow_certificate", flow_pass)]
    gains: list[float] = []
    exact_gains: list[Fraction] = []
    validated_jumps: list[AffineEPIJumpGainCertificate] = []
    for index, supplied_jump in enumerate(supplied_jumps):
        # Public dataclass records are convenient to inspect and can therefore
        # also be copied with replaced result fields.  Rebuild the theorem from
        # its declared affine inputs; never trust a stored Boolean or gain.
        jump = certify_affine_epi_jump_gain(
            supplied_jump.operator_name,
            supplied_jump.linear_map,
            supplied_jump.metric_weights,
            offset=supplied_jump.offset,
            nodes=supplied_jump.nodes,
            declared_energy_gain_bound=supplied_jump.declared_energy_gain_bound,
            tolerance=tol,
        )
        if jump.nodes != nodes:
            raise ValueError(
                f"jump {index} node order does not match the flow certificate"
            )
        if not _exactly_proportional(jump.metric_weights, reference_metric):
            raise ValueError(
                f"jump {index} metric is not exactly proportional to the flow metric"
            )
        jump_pass = jump.supports_global_gain_theorem
        conditions.append((f"jump_{index}_finite_global_energy_gain", jump_pass))
        gain = float(jump.energy_gain_bound_for_composition)
        if math.isnan(gain) or gain < 0.0:
            raise ValueError(
                f"jump {index} exposes an invalid energy-gain bound"
            )
        gains.append(gain)
        exact_gains.append(jump.exact_quotient_energy_gain_upper_bound)
        validated_jumps.append(jump)

    jump_tuple = tuple(validated_jumps)

    all_jump_pass = all(passed for _, passed in conditions[1:])
    exact_decay = sum(
        (
            Fraction.from_float(rate) * Fraction.from_float(duration)
            for duration in durations
        ),
        Fraction(0),
    )
    flow_log_decay = _fraction_lower_float(exact_decay)
    exact_gain_product: Fraction | None = None
    composition_factors: tuple[Fraction, ...] | None = None
    exact_net_log_upper: Fraction | None
    if not all_jump_pass or any(math.isinf(gain) for gain in gains):
        jump_log = float("inf")
        net_log = float("inf")
        multiplier = float("inf")
        exact_net_log_upper = None
    else:
        exact_gain_product = math.prod(exact_gains, start=Fraction(1))
        composition_factors = tuple(
            _bounded_composition_gain_factor(exact_gain)
            for exact_gain in exact_gains
        )
        if exact_gain_product == 0:
            jump_log = float("-inf")
            net_log = float("-inf")
            multiplier = 0.0
            exact_net_log_upper = None
        else:
            exact_jump_log_upper = sum(
                (
                    _exact_log_bounds(factor)[1]
                    for factor in composition_factors
                ),
                Fraction(0),
            )
            exact_net_log_upper = exact_jump_log_upper - exact_decay
            jump_log = _fraction_upper_signed_float(exact_jump_log_upper)
            net_log = _fraction_upper_signed_float(exact_net_log_upper)
            multiplier = _exp_upper_float(exact_net_log_upper)

    finite_horizon = bool(flow_pass and all_jump_pass)
    decision_margin = 0.0
    if net_log == float("-inf"):
        contracts = finite_horizon
    else:
        contracts = bool(
            finite_horizon
            and exact_net_log_upper is not None
            and exact_net_log_upper < 0
        )
    initial_mean_preserved = bool(
        flow_pass
        and flow_mean_preserved
        and all(
            jump.preserves_initial_weighted_consensus for jump in jump_tuple
        )
    )

    if repeat_schedule:
        disagreement_converges: bool | None = contracts
        initial_consensus_converges: bool | None = (
            contracts and initial_mean_preserved
        )
        if contracts:
            if net_log == float("-inf"):
                decay_rate = float("inf")
            else:
                exact_total_duration = sum(
                    (Fraction.from_float(value) for value in durations),
                    Fraction(0),
                )
                decay_rate = _fraction_lower_float(
                    -exact_net_log_upper / exact_total_duration
                )
        else:
            decay_rate = None
    else:
        disagreement_converges = None
        initial_consensus_converges = None
        decay_rate = None

    return HybridEPIStabilityCertificate(
        nodes=nodes,
        flow_certificate_kind=kind,
        flow_hypotheses_pass=flow_pass,
        normalized_metric_weights=_readonly_array(metric),
        flow_energy_decay_rate=rate,
        flow_durations=durations,
        total_flow_duration=total_duration,
        jumps=jump_tuple,
        jump_energy_gain_bounds=tuple(gains),
        cumulative_jump_log_energy_gain=jump_log,
        flow_log_energy_decay_lower_bound=flow_log_decay,
        net_log_energy_gain_bound=net_log,
        energy_multiplier_bound=multiplier,
        exact_cumulative_jump_energy_gain_bound=exact_gain_product,
        exact_log_composition_gain_factors=composition_factors,
        exact_flow_log_energy_decay_lower_bound=exact_decay,
        exact_net_log_energy_gain_upper_bound=exact_net_log_upper,
        log_contraction_decision_margin=decision_margin,
        numerical_conditions=tuple(conditions),
        finite_horizon_disagreement_bound_certified=finite_horizon,
        disagreement_contracts_over_declared_horizon=contracts,
        initial_weighted_mean_preserved=initial_mean_preserved,
        repeat_schedule_declared=repeat_schedule,
        repeated_schedule_disagreement_convergence_certified=disagreement_converges,
        repeated_schedule_initial_weighted_consensus_convergence_certified=(
            initial_consensus_converges
        ),
        asymptotic_disagreement_energy_decay_rate=decay_rate,
        tolerance=tol,
        scope=_HYBRID_SCOPE,
    )
