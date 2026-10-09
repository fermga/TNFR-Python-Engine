"""Exact linear observations and coordinate memory of a supplied generator.

The positive-sign convention is z'=J*z. The invariant row space is a model
property, not a graph, constitutive-law, trajectory or provenance certificate.
Known affine sources can be projected once this linear space is admitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from .._exact_time import exact_or_represented_real
from ._exact_linear_algebra import exact_matrix_inverse, exact_matrix_product
from .krylov import exact_rank

__all__ = [
    "LinearObservationLevel",
    "LinearObservation",
    "derive_linear_observation",
    "LinearCoordinateMemory",
    "derive_coordinate_memory",
    "FormLossObservationBound",
    "bound_form_loss_observation_error",
    "bound_orthogonal_form_loss_observation_error",
]

Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


@dataclass(frozen=True)
class LinearObservationLevel:
    """One completely examined output-Krylov level O*J**power."""

    power: int
    rank_before: int
    rank_after: int
    candidate_rows: int
    selected_row_indices: tuple[int, ...]


@dataclass(frozen=True)
class LinearObservation:
    """Minimal all-state linear realization of y=O*z under z'=J*z.

    With s=C*z, the exact identities C*T=I, C*J=G*C and O=D*C imply
    s'=G*s and y=D*s. The selected coordinates need not be local, physical
    nodes or a nonlinear observation. Source/initial-state admissibility is
    separate. Rationalized floating coefficients certify only that supplied
    rational model, not the exact transcendental matrix they approximate.
    """

    generator: Matrix
    output_rows: Matrix
    output_count: int
    output_rank: int
    dimension: int
    full_state_dimension: int
    extra_coordinates: int
    rank_progression: tuple[int, ...]
    level_records: tuple[LinearObservationLevel, ...]
    selected_row_labels: tuple[tuple[int, int], ...]
    pivot_columns: tuple[int, ...]
    observation: Matrix
    right_inverse: Matrix
    reduced_generator: Matrix
    output_map: Matrix
    exact_identity_checks: tuple[str, ...]
    rank_calls: int
    max_rank_calls: int
    matrix_product_calls: int
    completed_levels: int
    stabilization_power: int
    max_coefficient_bits: int
    scope: str


@dataclass(frozen=True)
class LinearCoordinateMemory:
    """Exact blocks of y'=A*y+B*h, h'=C*y+D*h under z'=J*z.

    Visible coordinates retain the requested order; hidden coordinates keep
    their original order. Eliminating h gives the initial-state contribution
    B*exp(D*t)*h0 and convolution kernel K(t)=B*exp(D*t)*C. Only K(0)=B*C is
    evaluated here. A vanishing K(0) does not imply a vanishing kernel or
    initial-state contribution. Blocks may be signed and D need not be stable.
    """

    generator: Matrix
    visible_indices: tuple[int, ...]
    hidden_indices: tuple[int, ...]
    visible_generator: Matrix
    hidden_to_visible: Matrix
    visible_to_hidden: Matrix
    hidden_generator: Matrix
    kernel_at_zero: Matrix
    scope: str


@dataclass(frozen=True)
class FormLossObservationBound:
    """Conditional lower bound for a conservative two-channel observation.

    The admitted generator is skew. Two orthonormal output rows define the
    transpose preparation. The quadrature method additionally verifies the
    canonical channel rotation and its output intertwiner; the initial-response
    method instead retains a generator-norm upper bound. The bound concerns
    the supremum of the Euclidean full-response error on [0, horizon] for
    the unit visible ball
    with zero hidden preparation. It also holds for enlarged preparation
    families containing that subset. It is not an upper bound, the actual
    error, or an authentication of graph, clock or energy coordinates.
    """

    generator: Matrix
    output_rows: Matrix
    preparation: Matrix
    target_generator: Matrix
    damping: Fraction
    exchange: Fraction
    horizon: Fraction
    witness_time: Fraction
    uniform_error_lower_bound: Fraction
    scope: str
    method: str = "quadrature_covariance"
    generator_norm_upper_bound: Fraction | None = None


def _ordered(values, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered collection")
    try:
        return tuple(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered collection") from exc


def _matrix(values, label):
    rows = _ordered(values, label)
    if not rows:
        raise ValueError(f"{label} must be nonempty")
    return tuple(
        tuple(
            exact_or_represented_real(value, f"{label}[{i}]")
            for value in _ordered(row, f"{label}[{i}]")
        )
        for i, row in enumerate(rows)
    )


def _conservative_observation(generator, output_rows):
    """Admit the common conservative law, observation and transpose lift."""
    admitted = _matrix(generator, "generator")
    size = len(admitted)
    if size % 2 or any(len(row) != size for row in admitted):
        raise ValueError("generator must be an even-dimensional square matrix")
    if any(admitted[i][j] != -admitted[j][i] for i in range(size) for j in range(size)):
        raise ValueError("generator must be exactly skew-symmetric")
    output = _matrix(output_rows, "output_rows")
    if len(output) != 2 or any(len(row) != size for row in output):
        raise ValueError("output_rows must have two rows matching the generator")
    preparation = tuple(zip(*output))
    if exact_matrix_product(output, preparation) != ((1, 0), (0, 1)):
        raise ValueError("output_rows must be exactly orthonormal")
    return admitted, output, preparation


def _form_loss_parameters(damping, exchange, horizon):
    """Admit one supplied two-channel target and comparison horizon."""
    gamma = exact_or_represented_real(damping, "damping")
    omega = exact_or_represented_real(exchange, "exchange")
    time = exact_or_represented_real(horizon, "horizon")
    if gamma < 0 or omega <= 0 or time < 0:
        raise ValueError(
            "damping and horizon must be nonnegative; exchange must be positive"
        )
    return gamma, omega, time


def bound_form_loss_observation_error(
    generator, output_rows, *, damping, exchange, horizon
) -> FormLossObservationBound:
    """Bound unavoidable error against a declared single-pair form-loss law.

    J is an even-dimensional real skew generator ordered as (form, phase),
    with equal-sized channel blocks. It must commute exactly with
    R_n=[[0,-I],[I,0]]. The two rows O must satisfy O O^T=I_2 and O R_n=R_1 O.
    The preparation is E=O^T and hidden preparation is zero; the law and
    observations must already use the same clock and energy-normalized chart.

    The target G=[[-gamma,-omega],[omega,0]] has nonnegative damping gamma
    and positive exchange omega. For T>=0, put M=gamma+omega and
    s=min(T,1/(2M)). The supremum over 0<=t<=T and ||y0||<=1 of
    ||O exp(Jt) E y0 - exp(Gt) y0|| is at least gamma*s*(1-M*s)/2.
    This uses exact covariance, target contractivity and the commutator
    remainder bound; no trajectory, exponential or fitted coefficient is used.

    Matrices and scalars share exact-or-represented admission. An exact
    rational probe does not authenticate an irrational energy chart or an
    ideal TNFR generator. Failed hypotheses raise rather than yielding a
    zero certificate. Zero horizon/damping are valid zero-bound controls.
    """
    admitted, output, preparation = _conservative_observation(generator, output_rows)
    size = len(admitted)
    half = size // 2
    rotation = tuple(
        tuple(Fraction(int(i == j + half) - int(j == i + half)) for j in range(size))
        for i in range(size)
    )
    if exact_matrix_product(admitted, rotation) != exact_matrix_product(
        rotation, admitted
    ):
        raise ValueError("generator must commute with the canonical channel rotation")
    visible_rotation = ((Fraction(0), Fraction(-1)), (Fraction(1), Fraction(0)))
    if exact_matrix_product(output, rotation) != exact_matrix_product(
        visible_rotation, output
    ):
        raise ValueError("output_rows must intertwine the canonical channel rotations")
    gamma, omega, time = _form_loss_parameters(damping, exchange, horizon)
    norm_upper = gamma + omega
    witness = min(time, 1 / (2 * norm_upper))
    lower = gamma * witness * (1 - norm_upper * witness) / 2
    return FormLossObservationBound(
        generator=admitted,
        output_rows=output,
        preparation=preparation,
        target_generator=((-gamma, -omega), (omega, Fraction(0))),
        damping=gamma,
        exchange=omega,
        horizon=time,
        witness_time=witness,
        uniform_error_lower_bound=lower,
        scope=(
            "Exact rational lower bound on uniform full-response error under the "
            "admitted conservative channel covariance and transpose preparation. "
            "The hidden-zero subset is retained; an enlarged independent hidden "
            "family cannot reduce the supremum. No graph, clock, energy-chart, "
            "nonlinear accuracy, effective-coefficient selection or physical "
            "identification is authenticated. Represented matrices certify only "
            "their supplied rational law."
        ),
    )


def bound_orthogonal_form_loss_observation_error(
    generator, output_rows, *, damping, exchange, horizon
) -> FormLossObservationBound:
    """Bound form-loss error without assuming equal-channel covariance.

    J must be even-dimensional and skew, and O must have exactly two
    orthonormal rows; the lift is O^T. All matrices, the target and the
    nonnegative horizon share the supplied clock and Euclidean energy chart.
    Target scalars have the same admission as bound_form_loss_observation_error.
    Unlike that reader, this bound admits channel-asymmetric observations.

    With N=max_i sum_j |J_ij|, M=gamma+omega, Q=N^2+M^2 and
    s=min(T,gamma/Q), the unit-visible-ball response error over [0,T] is at
    least gamma*s-Q*s^2/2. Skewness makes N a spectral norm upper bound.
    The proof uses the skew initial observed derivative and contractive
    Taylor remainders, not rotation covariance. The result does not admit
    correlated hidden preparation, a discarded initial layer, a different
    norm, nonlinear approximation or a graph/clock interpretation by itself.
    """
    admitted, output, preparation = _conservative_observation(generator, output_rows)
    gamma, omega, time = _form_loss_parameters(damping, exchange, horizon)
    norm_upper = max(sum(map(abs, row), Fraction(0)) for row in admitted)
    remainder = norm_upper**2 + (gamma + omega) ** 2
    witness = min(time, gamma / remainder)
    return FormLossObservationBound(
        generator=admitted,
        output_rows=output,
        preparation=preparation,
        target_generator=((-gamma, -omega), (omega, Fraction(0))),
        damping=gamma,
        exchange=omega,
        horizon=time,
        witness_time=witness,
        uniform_error_lower_bound=gamma * witness - remainder * witness**2 / 2,
        method="orthogonal_initial_response",
        generator_norm_upper_bound=norm_upper,
        scope=(
            "Exact rational lower bound from conservative initial-response skewness "
            "and finite generator norm, under orthogonal zero-hidden preparation. "
            "Channel covariance is not required. An enlarged hidden family containing "
            "the zero-hidden subset cannot lower the supremum. No graph, energy "
            "chart, clock, nonlinear accuracy or physical identity is authenticated. "
            "Represented matrices certify only their supplied rational law."
        ),
    )


def derive_coordinate_memory(generator, visible_indices) -> LinearCoordinateMemory:
    """Partition a fixed linear law without discarding its hidden dynamics.

    J is a finite, ordered, nonempty square matrix with the shared exact-or-
    represented scalar admission. Rational coefficients remain exact; other
    admitted real coefficients retain their binary64 values. Visible indices
    must be an ordered, nonempty proper subset of distinct nonboolean integers.

    For z'=J*z and visible/hidden blocks (y,h), the exact elimination identity is

        y'(t) = A*y(t) + B*exp(D*t)*h(0)
                + integral_0^t B*exp(D*(t-s))*C*y(s) ds.

    The returned matrices specify this identity without evaluating an
    exponential, supplying an initial state or evolving a graph. Neither
    memory decay, kernel positivity, autonomous instantaneous closure nor a
    physical connection follows from this decomposition. In particular, a
    zero B*C need not make B*exp(D*t)*C identically zero. No approximation,
    Schur-complement substitution or selected constitutive law is introduced.
    """
    admitted = _matrix(generator, "generator")
    dimension = len(admitted)
    if any(len(row) != dimension for row in admitted):
        raise ValueError("generator must be a nonempty square matrix")
    supplied_indices = _ordered(visible_indices, "visible_indices")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in supplied_indices
    ):
        raise TypeError("visible_indices must contain non-boolean integers")
    visible = tuple(int(index) for index in supplied_indices)
    if not 0 < len(visible) < dimension:
        raise ValueError("visible_indices must be a nonempty proper subset")
    selected = set(visible)
    if len(selected) != len(visible):
        raise ValueError("visible_indices must be distinct")
    if any(index < 0 or index >= dimension for index in visible):
        raise ValueError("visible_indices must lie within the generator dimension")
    hidden = tuple(index for index in range(dimension) if index not in selected)

    def block(rows, columns):
        return tuple(tuple(admitted[i][j] for j in columns) for i in rows)

    a, b = block(visible, visible), block(visible, hidden)
    c, d = block(hidden, visible), block(hidden, hidden)
    return LinearCoordinateMemory(
        generator=admitted,
        visible_indices=visible,
        hidden_indices=hidden,
        visible_generator=a,
        hidden_to_visible=b,
        visible_to_hidden=c,
        hidden_generator=d,
        kernel_at_zero=exact_matrix_product(b, c),
        scope=(
            "Exact coordinate decomposition of one supplied fixed rational generator "
            "with positive sign z'=J*z. General initial-hidden-state influence is retained. "
            "No exponential evaluation, decay/positivity/instantaneous-closure theorem, "
            "graph authentication, autonomous connection law or physical identity "
            "is certified. Represented coefficients do not certify an ideal "
            "transcendental generator."
        ),
    )


def derive_linear_observation(
    generator,
    output_rows,
    *,
    max_rank_calls=4096,
) -> LinearObservation:
    r"""Close the row space of O under a fixed positive-sign generator J.

    Inputs are finite ordered nonempty matrices. J must be square and each
    output row must have the generator's state dimension. Rational scalars remain
    exact; other admitted real scalars use the shared binary64 admission.
    Boolean, nonfinite and nonzero-underflow scalars are rejected. Redundant
    and all-zero output rows are permitted.

    Examine complete levels O, OJ, ... through the first stabilized level,
    selecting independent rows deterministically. All-state minimality follows
    because any invariant row space retaining O contains every OJ**k. A first
    row of rank one can require n+1 complete levels, through power n. A zero
    output requires the power-one check and yields an empty reduced state.

    Rank invocations are bounded by a positive nonboolean max_rank_calls.
    Exhaustion raises without a partial result or approximate fallback. The
    count does not bound input materialization, coefficient bit growth, memory
    or wall time; product counts exclude internal elimination operations.
    Neither stability, reachable-state minimality nor a nonlinear reduction
    follows. No graph, trajectory, affine source or initial state is supplied.
    """
    if type(max_rank_calls) is not int:
        raise TypeError("max_rank_calls must be a positive non-boolean integer")
    if max_rank_calls < 1:
        raise ValueError("max_rank_calls must be positive")
    a = _matrix(generator, "generator")
    if any(len(row) != len(a) for row in a):
        raise ValueError("generator must be a nonempty square matrix")
    r = _matrix(output_rows, "output_rows")
    if any(len(row) != len(a) for row in r):
        raise ValueError("each output row must match the generator dimension")
    n, m = len(a), len(r)
    rank_calls = product_calls = max_bits = 0

    def retain(matrix):
        nonlocal max_bits
        for row in matrix:
            for value in row:
                max_bits = max(
                    max_bits,
                    abs(value.numerator).bit_length(),
                    value.denominator.bit_length(),
                )
        return matrix

    def rank(matrix):
        nonlocal rank_calls
        if rank_calls >= max_rank_calls:
            raise ValueError("exact rank-call budget exhausted; realization incomplete")
        rank_calls += 1
        return exact_rank(matrix)

    def product(left, right):
        nonlocal product_calls
        product_calls += 1
        return retain(exact_matrix_product(left, right))

    retain(a)
    retain(r)
    basis, labels, levels = [], [], []
    level = r
    for power in range(n + 1):
        previous_rank = len(basis)
        selected = []
        for row_index, row in enumerate(level):
            observed_rank = rank((*basis, row))
            if observed_rank not in (len(basis), len(basis) + 1):
                raise RuntimeError(
                    "exact independent-row selection lost rank consistency"
                )
            if observed_rank > len(basis):
                basis.append(row)
                labels.append((power, row_index))
                selected.append(row_index)
        levels.append(
            LinearObservationLevel(
                power,
                previous_rank,
                len(basis),
                m,
                tuple(selected),
            )
        )
        if power > 0 and len(basis) == previous_rank:
            break
        if power < n:
            level = product(level, a)
    else:
        raise RuntimeError(
            "exact row-space closure did not stabilize within its dimension bound"
        )
    c = tuple(basis)
    dimension = len(c)
    output_rank = levels[0].rank_after
    columns, chosen = [], []
    zero, one = Fraction(0), Fraction(1)
    if dimension:
        for j in range(n):
            column = tuple(row[j] for row in c)
            observed_rank = rank((*columns, column))
            if observed_rank not in (len(columns), len(columns) + 1):
                raise RuntimeError("exact pivot-column selection lost rank consistency")
            if observed_rank > len(columns):
                columns.append(column)
                chosen.append(j)
            if len(columns) == dimension:
                break
        if len(columns) != dimension:
            raise RuntimeError(
                "independent observation rows have no invertible column minor"
            )
        minor = tuple(tuple(row[j] for j in chosen) for row in c)
        inverse = retain(exact_matrix_inverse(minor))
        selected_positions = {j: i for i, j in enumerate(chosen)}
        t = retain(
            tuple(
                (
                    inverse[selected_positions[i]]
                    if i in selected_positions
                    else (zero,) * dimension
                )
                for i in range(n)
            )
        )
        ca = product(c, a)
        g, d = product(ca, t), product(r, t)
    else:
        # Empty tuples alone lose their column count. Handle the known
        # 0-by-n, n-by-0, 0-by-0 and m-by-0 shapes explicitly rather than
        # changing the nonempty shared product/inverse contracts.
        t = ((),) * n
        ca, g, d = (), (), ((),) * m
    identity = tuple(
        tuple(one if i == j else zero for j in range(dimension))
        for i in range(dimension)
    )
    checks = {
        "rank(C)=dimension": rank(c) == dimension,
        "C T=I": (product(c, t) if dimension else ()) == identity,
        "C J=G C": ca == (product(g, c) if dimension else ()),
        "O=D C": r == (product(d, c) if dimension else ((zero,) * n,) * m),
        "complete-level stabilization": levels[-1].rank_before == levels[-1].rank_after,
        "initial output rank retained": output_rank <= dimension,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"exact sufficient-observation identities failed: {failed}")
    return LinearObservation(
        generator=a,
        output_rows=r,
        output_count=m,
        output_rank=output_rank,
        dimension=dimension,
        full_state_dimension=n,
        extra_coordinates=dimension - output_rank,
        rank_progression=tuple(level.rank_after for level in levels),
        level_records=tuple(levels),
        selected_row_labels=tuple(labels),
        pivot_columns=tuple(chosen),
        observation=c,
        right_inverse=t,
        reduced_generator=g,
        output_map=d,
        exact_identity_checks=tuple(checks),
        rank_calls=rank_calls,
        max_rank_calls=max_rank_calls,
        matrix_product_calls=product_calls,
        completed_levels=len(levels),
        stabilization_power=levels[-1].power,
        max_coefficient_bits=max_bits,
        scope=(
            "Minimal all-state linear observation of one supplied fixed rational "
            "generator with positive sign z'=J*z. No nonlinear/reachable-state "
            "minimality, graph authentication, constitutive selection, trajectory "
            "or physical identity is certified."
        ),
    )
