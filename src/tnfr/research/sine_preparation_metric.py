"""Conditional graph-metric arithmetic for held sine preparation errors.

The component metric is fixed to ``(L_weighted, L_bare)``. Signed finite
secant cosines retain the reciprocal cancellation; no acute-phase premise,
trajectory, source authentication or family-selection rule is supplied here.
The caller owns the nominal whole-cell and complete-law evidence.

Prepared objects are ephemeral, immutable capabilities, not serializable proof
premises. Public consumers reject copied/replaced capabilities. Each run derives
its own coefficient bounds from admitted primitive phase boxes, retains both
conserved preparation means, and stops at its first unresolved sufficient gate.

Work policies bound component sizes and admitted/post-operation rational sizes.
The shared inverse/product/trigonometric kernels can create larger internal
temporaries before returning. These boundary checks are not a memory or time
proof; an external monitored budget is required for a measured application.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from weakref import WeakValueDictionary

from ..mathematics._comparison_flow import _exact, scalar_affine_upper
from ..mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_matrix_product,
)
from ..mathematics._rational_interval import I, cos, pi_interval, sqrt
from ..physics.relational_observations import _ordered

MAX_METRIC_NODES = 21
MAX_METRIC_EDGES = 53
MAX_METRIC_CELLS = 256
MAX_INPUT_BITS = 4096
MAX_INTERMEDIATE_BITS = 65536
METRIC_METHOD = "bare_graph_signed_secant_trace_degree_scaled_scalar_v1"

_CONTEXTS = WeakValueDictionary()
_FAMILIES = WeakValueDictionary()


class MetricBudgetError(ValueError):
    """An explicit rational-work boundary was exceeded."""

    def __init__(self, message, *, maximum_bits=0):
        super().__init__(message)
        self.maximum_bits = maximum_bits


class _Budget:
    def __init__(self, initial=0):
        self.maximum = initial

    def check(self, value):
        if isinstance(value, Q):
            bits = max(
                abs(value.numerator).bit_length(), value.denominator.bit_length()
            )
            self.maximum = max(self.maximum, bits)
            if bits > MAX_INTERMEDIATE_BITS:
                raise MetricBudgetError(
                    "intermediate rational bit policy exceeded",
                    maximum_bits=self.maximum,
                )
        elif isinstance(value, (tuple, list)):
            for item in value:
                self.check(item)
        else:
            raise TypeError("internal rational-work record has an invalid value")
        return value


def _input(value, label, budget):
    result = _exact(value)
    bits = max(abs(result.numerator).bit_length(), result.denominator.bit_length())
    if bits > MAX_INPUT_BITS:
        raise MetricBudgetError(f"{label} exceeds the input rational bit policy")
    return budget.check(result)


def _items(values, label, maximum):
    result = _ordered(values, label, limit=maximum + 1)
    if len(result) > maximum:
        raise ValueError(f"{label} exceeds its declared work limit")
    return result


def _index(value, size, label):
    if type(value) is not int:
        raise TypeError(f"{label} requires an ordinary integer")
    if not 0 <= value < size:
        raise ValueError(f"{label} is outside the component")
    return value


def _pair(values, label, budget, *, errors=False):
    values = _items(values, label, 2)
    if len(values) != 2:
        raise ValueError(f"{label} requires two endpoints")
    pair = tuple(_input(value, label, budget) for value in values)
    if errors:
        if min(pair) < 0:
            raise ValueError(f"{label} requires nonnegative error magnitudes")
    elif pair[0] > pair[1]:
        raise ValueError(f"{label} endpoints are reversed")
    return pair


def _pairs(values, size, label, budget, *, errors=False):
    values = _items(values, label, size)
    if len(values) != size:
        raise ValueError(f"{label} has the wrong component dimension")
    return tuple(_pair(value, label, budget, errors=errors) for value in values)


def _laplacian(size, edges, weights, budget):
    matrix = [[Q(0) for _ in range(size)] for _ in range(size)]
    for (i, j), weight in zip(edges, weights):
        matrix[i][i] += weight
        matrix[j][j] += weight
        matrix[i][j] -= weight
        matrix[j][i] -= weight
    return budget.check(tuple(tuple(row) for row in matrix))


def _quadratic(matrix, vector, budget):
    return budget.check(
        sum(
            (
                vector[i] * matrix[i][j] * vector[j]
                for i in range(len(vector))
                for j in range(len(vector))
            ),
            Q(0),
        )
    )


def _edge_quadratic(matrix, edge, budget):
    i, j = edge
    return budget.check(matrix[i][i] + matrix[j][j] - 2 * matrix[i][j])


def _sqrt_scaled(square, budget):
    """Return exact scale and a shared outward root of the normalized square."""
    budget.check(square)
    if square < 0:
        raise ValueError("a proved squared quantity must be nonnegative")
    if not square:
        return Q(1), Q(0)
    exponent = (square.numerator.bit_length() - square.denominator.bit_length()) // 2
    scale = Q(1 << exponent) if exponent >= 0 else Q(1, 1 << -exponent)
    normalized = budget.check(square / scale**2)
    enclosed = I(normalized)
    budget.check((enclosed.lo, enclosed.hi))
    upper = sqrt(enclosed).hi
    budget.check((scale, upper, scale * upper))
    return scale, upper


def _sqrt_upper(square, budget):
    scale, normalized = _sqrt_scaled(square, budget)
    return budget.check(scale * normalized)


@dataclass(frozen=True, eq=False)
class MetricContext:
    """Builder-admitted component; field copies are not admitted capabilities."""

    capacities: tuple
    edges: tuple
    conductances: tuple
    degrees: tuple
    beta: tuple
    rho: tuple
    alpha: tuple
    laplacian_weighted: tuple
    laplacian_bare: tuple
    form_metric: tuple
    phase_metric: tuple
    form_inverse: tuple
    phase_inverse: tuple
    form_sandwich: tuple
    edge_phase_duals: tuple
    edge_form_duals: tuple
    edge_sandwich: tuple
    edge_norm_uppers: tuple
    a0: Q
    b0: Q
    degree_norm_upper: Q
    maximum_bits: int


def _require_context(context):
    if type(context) is not MetricContext or _CONTEXTS.get(id(context)) is not context:
        raise ValueError("context must be the live result of prepare_metric_context")
    return context


def prepare_metric_context(*, capacities, edges, conductances, degrees):
    """Admit a connected component and build its two exact quotient inverses.

    Scalars require ordinary ``int`` or normalized ``Fraction``. Indices and
    degrees require ordinary integers. Zero conductances retain bare support;
    the positive-conductance subgraph must itself be connected. Node order is
    the supplied capacity order, and every edge must have ``i < j``.
    """
    budget = _Budget()
    capacities = _items(capacities, "capacities", MAX_METRIC_NODES)
    if len(capacities) < 2:
        raise ValueError("a dynamic component requires between 2 and 21 nodes")
    capacities = tuple(_input(value, "capacity", budget) for value in capacities)
    if min(capacities) <= 0:
        raise ValueError("component capacities must be positive")
    size = len(capacities)
    raw_edges = _items(edges, "edges", MAX_METRIC_EDGES)
    if not raw_edges:
        raise ValueError("the component requires edges")
    admitted_edges = []
    for edge in raw_edges:
        pair = _items(edge, "edge", 2)
        if len(pair) != 2:
            raise ValueError("an edge requires exactly two indices")
        i, j = (_index(value, size, "edge index") for value in pair)
        if i >= j:
            raise ValueError("edges require distinct ascending node indices")
        admitted_edges.append((i, j))
    edges = tuple(admitted_edges)
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate edges are not admitted")
    raw_weights = _items(conductances, "conductances", len(edges))
    if len(raw_weights) != len(edges):
        raise ValueError("conductances must match the edge order")
    conductances = tuple(_input(value, "conductance", budget) for value in raw_weights)
    if min(conductances) < 0:
        raise ValueError("conductances must be nonnegative")
    degrees = _items(degrees, "degrees", size)
    if len(degrees) != size or any(type(value) is not int for value in degrees):
        raise TypeError("degrees require one ordinary integer per node")
    actual = tuple(sum(i in edge for edge in edges) for i in range(size))
    if degrees != actual or min(actual) == 0:
        raise ValueError("degrees must match complete unique bare support")
    reached = {0}
    while True:
        expanded = reached | {
            j if i in reached else i
            for (i, j), weight in zip(edges, conductances)
            if weight > 0 and (i in reached or j in reached)
        }
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != size:
        raise ValueError("positive-conductance support must be connected")
    beta = budget.check(tuple(capacities[i] / degrees[i] for i in range(size)))
    rho = budget.check(tuple(1 / value for value in beta))
    total = budget.check(sum(rho, Q(0)))
    alpha = budget.check(tuple(value / total for value in rho))
    weighted = _laplacian(size, edges, conductances, budget)
    bare = _laplacian(size, edges, (Q(1),) * len(edges), budget)
    completion = budget.check(
        tuple(tuple(rho[i] * rho[j] / total for j in range(size)) for i in range(size))
    )
    matrices = tuple(
        budget.check(
            tuple(
                tuple(matrix[i][j] + completion[i][j] for j in range(size))
                for i in range(size)
            )
        )
        for matrix in (weighted, bare)
    )
    inverses = tuple(budget.check(exact_matrix_inverse(matrix)) for matrix in matrices)
    identity = tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))
    for matrix, inverse in zip(matrices, inverses):
        if budget.check(exact_matrix_product(matrix, inverse)) != identity:
            raise ArithmeticError("shared inverse failed its exact residual")
    sandwich = budget.check(
        tuple(
            tuple(beta[i] * weighted[i][j] * beta[j] for j in range(size))
            for i in range(size)
        )
    )
    phase_duals = tuple(_edge_quadratic(inverses[1], edge, budget) for edge in edges)
    form_duals = tuple(_edge_quadratic(inverses[0], edge, budget) for edge in edges)
    edge_sandwich = tuple(_edge_quadratic(sandwich, edge, budget) for edge in edges)
    if min(phase_duals + form_duals + edge_sandwich) <= 0:
        raise ArithmeticError("connected positive metric has a nonpositive edge dual")
    a0 = budget.check(2 * max(capacities))
    b0 = budget.check(2 * max(beta[i] * weighted[i][i] for i in range(size)))
    edge_norms = tuple(
        _sqrt_upper(budget.check(a * q), budget)
        for a, q in zip(edge_sandwich, phase_duals)
    )
    degree_norm = _sqrt_upper(budget.check(a0 * b0), budget)
    result = MetricContext(
        capacities,
        edges,
        conductances,
        degrees,
        beta,
        rho,
        alpha,
        weighted,
        bare,
        matrices[0],
        matrices[1],
        inverses[0],
        inverses[1],
        sandwich,
        phase_duals,
        form_duals,
        edge_sandwich,
        edge_norms,
        a0,
        b0,
        degree_norm,
        budget.maximum,
    )
    _CONTEXTS[id(result)] = result
    return result


@dataclass(frozen=True, eq=False)
class MetricFamily:
    """Whole coordinate-error box, including its conserved mean offsets."""

    context: MetricContext
    form_errors: tuple
    phase_errors: tuple
    parent: int
    child: int | None
    reference_weight: Q
    initial_norm_squared: Q
    radius_scale: Q
    initial_normalized_radius: Q
    initial_radius_upper: Q
    form_mean_error: tuple
    phase_mean_error: tuple
    form_offset: tuple
    phase_offset: tuple
    offset_norm_upper: Q
    observation_form_dual: Q
    observation_phase_dual: Q
    observation_factor: Q
    observation_root_upper: Q
    maximum_bits: int


def _require_family(family):
    if type(family) is not MetricFamily or _FAMILIES.get(id(family)) is not family:
        raise ValueError("family must be the live result of prepare_metric_family")
    _require_context(family.context)
    return family


def prepare_metric_family(
    context,
    *,
    form_errors,
    phase_errors,
    parent,
    reference_weight,
    child=None,
    child_form_error=None,
    child_phase_error=None,
):
    """Map a whole initial box into a scaled norm and the original portrait.

    Each error pair is ``(lower_magnitude, upper_magnitude)`` for the signed
    interval ``[-lower_magnitude, upper_magnitude]``. With ``child=None`` the
    child is frozen outside this host, and both child error pairs are required.
    Connected portraits require an in-component distinct child and no external
    child pairs. Neither case replaces member-specific mean errors with zero.
    """
    context = _require_context(context)
    budget = _Budget(context.maximum_bits)
    size = len(context.capacities)
    form_errors = _pairs(form_errors, size, "form_errors", budget, errors=True)
    phase_errors = _pairs(phase_errors, size, "phase_errors", budget, errors=True)
    parent = _index(parent, size, "parent")
    weight = _input(reference_weight, "reference_weight", budget)
    if weight <= 0:
        raise ValueError("reference_weight must be positive")
    means = tuple(
        budget.check(
            (
                -sum((a * pair[0] for a, pair in zip(context.alpha, errors)), Q(0)),
                sum((a * pair[1] for a, pair in zip(context.alpha, errors)), Q(0)),
            )
        )
        for errors in (form_errors, phase_errors)
    )
    if child is None:
        child_errors = (
            _pair(child_form_error, "child_form_error", budget, errors=True),
            _pair(child_phase_error, "child_phase_error", budget, errors=True),
        )
        offsets = tuple(
            budget.check((-pair[0] - mean[1], pair[1] - mean[0]))
            for pair, mean in zip(child_errors, means)
        )
        reading = tuple(Q(i == parent) - context.alpha[i] for i in range(size))
    else:
        child = _index(child, size, "child")
        if (
            child == parent
            or child_form_error is not None
            or child_phase_error is not None
        ):
            raise ValueError(
                "connected portraits require a distinct child and no external errors"
            )
        offsets = ((Q(0), Q(0)), (Q(0), Q(0)))
        reading = tuple(Q(i == child) - Q(i == parent) for i in range(size))
    radii_form = tuple(max(pair) for pair in form_errors)
    radii_phase = tuple(max(pair) for pair in phase_errors)
    initial_square = budget.check(
        sum(
            (
                conductance * (radii_form[i] + radii_form[j]) ** 2
                + (radii_phase[i] + radii_phase[j]) ** 2
                for (i, j), conductance in zip(context.edges, context.conductances)
            ),
            Q(0),
        )
    )
    scale, normalized = _sqrt_scaled(initial_square, budget)
    offset_square = budget.check(
        max(map(abs, offsets[1])) ** 2 + weight * max(map(abs, offsets[0])) ** 2
    )
    offset_upper = _sqrt_upper(offset_square, budget)
    form_dual = _quadratic(context.form_inverse, reading, budget)
    phase_dual = _quadratic(context.phase_inverse, reading, budget)
    if min(form_dual, phase_dual) <= 0:
        raise ArithmeticError("the marked reading must have positive dual coefficients")
    factor = budget.check(max(phase_dual, weight * form_dual))
    observation_root = _sqrt_upper(factor, budget)
    result = MetricFamily(
        context,
        form_errors,
        phase_errors,
        parent,
        child,
        weight,
        initial_square,
        scale,
        normalized,
        scale * normalized,
        means[0],
        means[1],
        offsets[0],
        offsets[1],
        offset_upper,
        form_dual,
        phase_dual,
        factor,
        observation_root,
        budget.maximum,
    )
    _FAMILIES[id(result)] = result
    return result


@dataclass(frozen=True)
class MetricCoefficientBound:
    """Two proved norm upper bounds for the same signed secant intervals."""

    secant_intervals: tuple
    midpoints: tuple
    radii: tuple
    trace_square: Q
    trace_residual_upper: Q
    trace_upper: Q
    degree_upper: Q
    norm_upper: Q
    gamma_upper: Q
    maximum_bits: int


def _coefficient_bound(context, secant_intervals, budget):
    midpoints = budget.check(tuple((lo + hi) / 2 for lo, hi in secant_intervals))
    radii = budget.check(tuple((hi - lo) / 2 for lo, hi in secant_intervals))
    difference = _laplacian(
        len(context.capacities),
        context.edges,
        tuple(1 - value for value in midpoints),
        budget,
    )
    left = budget.check(exact_matrix_product(difference, context.form_sandwich))
    product = budget.check(exact_matrix_product(left, difference))
    size = len(context.capacities)
    trace_square = budget.check(
        sum(
            (
                context.phase_inverse[i][j] * product[j][i]
                for i in range(size)
                for j in range(size)
            ),
            Q(0),
        )
    )
    if trace_square < 0:
        raise ArithmeticError("the exact squared Frobenius trace is negative")
    residual = budget.check(
        sum(
            (radius * norm for radius, norm in zip(radii, context.edge_norm_uppers)),
            Q(0),
        )
    )
    trace_upper = budget.check(_sqrt_upper(trace_square, budget) + residual)
    delta = max(1 - lo for lo, _ in secant_intervals)
    degree_upper = budget.check(delta * context.degree_norm_upper)
    norm_upper = min(trace_upper, degree_upper)
    gamma = budget.check(norm_upper / (2 * pi_interval().lo))
    return MetricCoefficientBound(
        secant_intervals,
        midpoints,
        radii,
        trace_square,
        residual,
        trace_upper,
        degree_upper,
        norm_upper,
        gamma,
        budget.maximum,
    )


def bound_secant_metric(context, *, secant_intervals):
    """Bound supplied finite secants; their dynamical association is external.

    This constructed/arithmetic entry point accepts exact endpoint pairs in
    [-1,1]. Whole-cell runs independently derive their own intervals by cosine
    and do not consume this report as a premise.
    """
    context = _require_context(context)
    budget = _Budget(context.maximum_bits)
    intervals = _pairs(secant_intervals, len(context.edges), "secant_intervals", budget)
    if any(lo < -1 or hi > 1 for lo, hi in intervals):
        raise ValueError("secant intervals must lie within [-1,1]")
    return _coefficient_bound(context, intervals, budget)


@dataclass(frozen=True)
class MetricStep:
    """One attempted cell, with scalar carry and strict domain margins."""

    index: int
    time: Q
    duration: Q
    inflated_edges: tuple
    coefficient: MetricCoefficientBound
    normalized_radius_before: Q
    normalized_radius_after: Q
    radius_upper: Q
    edge_margins: tuple
    portrait_error_upper: Q
    domain_passed: bool
    maximum_bits: int


class _ScalarWorkError(ValueError):
    def __init__(self, coefficient, maximum_bits):
        super().__init__("the fixed scalar cell exceeds gamma*duration <= 1")
        self.coefficient = coefficient
        self.maximum_bits = maximum_bits


def _portrait_upper(family, radius, budget):
    return budget.check(
        family.offset_norm_upper + family.observation_root_upper * radius
    )


def _edge_margins(context, allowances, radius, budget):
    return budget.check(
        tuple(
            h**2 - q * radius**2 for h, q in zip(allowances, context.edge_phase_duals)
        )
    )


def _bound_metric_cell(
    family,
    phase_intervals,
    edge_allowances,
    *,
    index,
    time,
    duration,
    normalized_radius,
):
    """Private cell arithmetic; caller has admitted the complete input batch."""
    budget = _Budget(family.maximum_bits)
    context = family.context
    inflated = budget.check(
        tuple(
            (
                phase_intervals[i][0] - phase_intervals[j][1] - h,
                phase_intervals[i][1] - phase_intervals[j][0] + h,
            )
            for (i, j), h in zip(context.edges, edge_allowances)
        )
    )
    secants = []
    for lo, hi in inflated:
        result = cos(I(lo, hi))
        secants.append(budget.check((result.lo, result.hi)))
    coefficient = _coefficient_bound(context, tuple(secants), budget)
    if budget.check(coefficient.gamma_upper * duration) > 1:
        raise _ScalarWorkError(coefficient, budget.maximum)
    after = budget.check(
        scalar_affine_upper(normalized_radius, coefficient.gamma_upper, duration, Q(0))
    )
    if after < normalized_radius:
        raise ArithmeticError("a nonnegative growth bound reduced the carried radius")
    radius = budget.check(family.radius_scale * after)
    margins = _edge_margins(context, edge_allowances, radius, budget)
    portrait = _portrait_upper(family, radius, budget)
    return MetricStep(
        index,
        time,
        duration,
        inflated,
        coefficient,
        normalized_radius,
        after,
        radius,
        margins,
        portrait,
        all(value > 0 for value in margins),
        budget.maximum,
    )


@dataclass(frozen=True)
class MetricRun:
    """Conditional whole-window result, or an explicit incomplete prefix.

    ``completed`` means every supplied cell closed its sufficient first-exit
    gate. It authenticates neither a nominal trajectory nor a response claim.
    With ``unavailable``, the reported radius is only the recorded comparison
    endpoint; it is not a certified whole-window preparation bound.
    """

    status: str
    reason: str | None
    steps: tuple
    initial_edge_margins: tuple
    final_normalized_radius: Q
    final_radius_upper: Q
    final_portrait_error_upper: Q
    maximum_bits: int
    attempted_cells: int
    completed_cells: int
    failed_coefficient: MetricCoefficientBound | None


def bound_metric_cells(family, *, phase_cells, edge_allowances, durations):
    """Attempt a declared, nonadaptive sequence of at most 256 nominal cells.

    Each cell is the full node-ordered phase box; endpoints are exact signed
    pairs. Durations are strictly positive and define contiguous chronology
    starting at zero. Positive edge allowances are fixed for the entire run.
    Input admission completes before arithmetic iteration. A nonpositive
    domain margin or excessive scalar-work cell returns unavailable without
    refinement, retry, rate fitting or a change of family. Monotonicity of the
    nonnegative scalar comparison makes each endpoint bound its entire cell.
    """
    family = _require_family(family)
    context = family.context
    budget = _Budget(family.maximum_bits)
    raw_cells = _items(phase_cells, "phase_cells", MAX_METRIC_CELLS)
    if not raw_cells:
        raise ValueError("at least one phase cell is required")
    cells = tuple(
        _pairs(cell, len(context.capacities), "phase cell", budget)
        for cell in raw_cells
    )
    raw_durations = _items(durations, "durations", MAX_METRIC_CELLS)
    if len(raw_durations) != len(cells):
        raise ValueError("durations must match the complete cell sequence")
    durations = tuple(_input(value, "duration", budget) for value in raw_durations)
    if min(durations) <= 0:
        raise ValueError("cell durations must be positive")
    times = []
    end = Q(0)
    for duration in durations:
        times.append(end)
        end = budget.check(end + duration)
    raw_allowances = _items(edge_allowances, "edge_allowances", len(context.edges))
    if len(raw_allowances) != len(context.edges):
        raise ValueError("allowances must match complete bare support")
    allowances = tuple(
        _input(value, "edge allowance", budget) for value in raw_allowances
    )
    if min(allowances) <= 0:
        raise ValueError("edge allowances must be positive")
    radius = family.initial_radius_upper
    normalized = family.initial_normalized_radius
    initial = _edge_margins(context, allowances, radius, budget)
    portrait = _portrait_upper(family, radius, budget)
    reason = None if all(value > 0 for value in initial) else "initial_domain"
    steps = []
    attempted = 0
    failed_coefficient = None
    if reason is None:
        for index, (cell, duration, time) in enumerate(zip(cells, durations, times)):
            attempted += 1
            try:
                step = _bound_metric_cell(
                    family,
                    cell,
                    allowances,
                    index=index,
                    time=time,
                    duration=duration,
                    normalized_radius=normalized,
                )
            except _ScalarWorkError as exc:
                budget.maximum = max(budget.maximum, exc.maximum_bits)
                failed_coefficient = exc.coefficient
                reason = "scalar_work"
                break
            except MetricBudgetError as exc:
                budget.maximum = max(budget.maximum, exc.maximum_bits)
                reason = "intermediate_budget"
                break
            steps.append(step)
            budget.maximum = max(budget.maximum, step.maximum_bits)
            normalized, radius = step.normalized_radius_after, step.radius_upper
            portrait = step.portrait_error_upper
            if not step.domain_passed:
                reason = "first_exit"
                break
    return MetricRun(
        "completed" if reason is None else "unavailable",
        reason,
        tuple(steps),
        initial,
        normalized,
        radius,
        portrait,
        budget.maximum,
        attempted,
        sum(step.domain_passed for step in steps),
        failed_coefficient,
    )
