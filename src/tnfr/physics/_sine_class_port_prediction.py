"""Causal six-port prediction from grounded-path Volterra coefficients.

This private fixed-model calculator evaluates ordinary time polynomials of
the collective interface. It does not call the full-state variational
recurrence or a nonlinear integrator. Source phase intervals are deviations
``y=theta-Theta_k`` at ``0-``; they are not absolute phase lifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb

from .._exact_time import exact_or_represented_real
from ..mathematics._exact_linear_algebra import exact_matrix_product
from ..mathematics._rational_interval import INTERVAL_METHOD, I
from ._sine_class_collective_interface import (
    _bound_collective_interface,
    _CollectiveInterface,
    _CollectiveInterfaceBound,
    _derive_collective_interface,
)
from .relational_observations import _ordered
from .relational_sine_class_cubic_response import _convolution, _exponential_tail
from .relational_sine_class_mediation import _EDGES

_ORDER = 32
_LINEAR_NORM = Q(201, 100)
_GAMMA_UPPER = Q(1, 3000)
_ZERO = I(0)


def _nonzero(value):
    return value.lo != 0 or value.hi != 0


def _sparse(matrix):
    return tuple(
        tuple((j, value) for j, value in enumerate(row) if _nonzero(value))
        for row in matrix
    )


def _apply(rows, vector):
    return tuple(
        sum((value * vector[j] for j, value in row if _nonzero(vector[j])), _ZERO)
        for row in rows
    )


def _product(left, right):
    columns = tuple(zip(*right))
    return tuple(
        tuple(
            sum(
                (a * b for a, b in zip(row, column) if _nonzero(a) and _nonzero(b)),
                _ZERO,
            )
            for column in columns
        )
        for row in left
    )


def _beta(i, j):
    """Exact integral coefficient for (t-s)**i * s**j."""
    return Q(1, (i + j + 1) * comb(i + j, i))


@dataclass(frozen=True)
class _PortKernelCoefficients:
    """Grounded-path factors, with hidden coordinates grouped by component."""

    port_indices: tuple[int, ...]
    hidden_component_indices: tuple[tuple[int, ...], ...]
    visible_generator: tuple[tuple[I, ...], ...]
    hidden_to_visible: tuple[tuple[tuple[I, ...], ...], ...]
    visible_to_hidden: tuple[tuple[tuple[I, ...], ...], ...]
    grounded_kernel_coefficients: tuple[tuple[tuple[tuple[I, ...], ...], ...], ...]
    hidden_input_kernel_coefficients: tuple[tuple[tuple[tuple[I, ...], ...], ...], ...]
    memory_kernel_coefficients: tuple[tuple[tuple[I, ...], ...], ...]
    order: int


def _kernel_coefficients(descriptor, *, order):
    """Build G_n=T**n tensor(P**n/n!) and K_n=B G_n C.

    The rational spatial factor is divided before interval materialization.
    A prematurely rounded T**n/n! would magnify its absolute grid error when
    multiplied by growing spatial powers.
    """
    if type(order) is not int or not 1 <= order <= _ORDER:
        raise ValueError("causal polynomial order must lie in 1..32")
    parameters = descriptor.parameter_bounds
    gamma = parameters.gamma
    form = descriptor.spatial_partition.generator
    phase_components = tuple(p.generator for p in descriptor.phase_component_partitions)
    cosines = tuple(parameters.edge_cosines[9 * j] for j in range(3)) + (I(1),)

    def entry(i, j):
        if i < 27 and j < 27:
            return I(-form[i][j])
        if i >= 27 and j < 27:
            return gamma * form[i - 27][j]
        if i < 27:
            value = sum(
                (
                    cosine * part[i][j - 27]
                    for cosine, part in zip(cosines, phase_components)
                    if part[i][j - 27]
                ),
                _ZERO,
            )
            return -gamma * value
        return _ZERO

    ports = descriptor.port_indices
    groups = tuple(
        path + tuple(i + 27 for i in path)
        for path in descriptor.hidden_path_node_orders
    )
    visible = tuple(tuple(entry(i, j) for j in ports) for i in ports)
    outgoing = tuple(
        tuple(tuple(entry(i, j) for j in group) for i in ports) for group in groups
    )
    incoming = tuple(
        tuple(tuple(entry(i, j) for j in ports) for i in group) for group in groups
    )
    path = descriptor.normalized_grounded_path_matrix
    spatial = [tuple(tuple(Q(int(i == j)) for j in range(8)) for i in range(8))]
    for n in range(1, order + 1):
        product = exact_matrix_product(spatial[-1], path)
        spatial.append(tuple(tuple(value / n for value in row) for row in product))
    kernels, inputs = [], []
    memory = [[[_ZERO] * 6 for _ in range(6)] for _ in range(order + 1)]
    for component in range(3):
        channel = ((I(-1), -gamma * cosines[component]), (gamma, _ZERO))
        power = ((I(1), _ZERO), (_ZERO, I(1)))
        gs, us = [], []
        for n, spatial_n in enumerate(spatial):
            matrix = tuple(
                tuple(
                    (
                        power[i // 8][j // 8] * spatial_n[i % 8][j % 8]
                        if spatial_n[i % 8][j % 8]
                        else _ZERO
                    )
                    for j in range(16)
                )
                for i in range(16)
            )
            gc = _product(matrix, incoming[component])
            bgc = _product(outgoing[component], gc)
            for i in range(6):
                for j in range(6):
                    memory[n][i][j] += bgc[i][j]
            gs.append(matrix)
            us.append(gc)
            if n < order:
                power = _product(power, channel)
        kernels.append(tuple(gs))
        inputs.append(tuple(us))
    return _PortKernelCoefficients(
        ports,
        groups,
        visible,
        outgoing,
        incoming,
        tuple(kernels),
        tuple(inputs),
        tuple(tuple(tuple(row) for row in matrix) for matrix in memory),
        order,
    )


def _kernel_integral(kernel, forcing, order):
    """A finite exact-beta convolution, with no coefficient outside the budget."""
    size = len(kernel[0])
    result = [[_ZERO] * size for _ in range(order + 1)]
    sparse = tuple(_sparse(matrix) for matrix in kernel)
    for p in range(order):
        for q in range(min(len(forcing), order - p)):
            if not any(_nonzero(value) for value in forcing[q]):
                continue
            values = _apply(sparse[p], forcing[q])
            weight = _beta(p, q)
            for i, value in enumerate(values):
                if _nonzero(value):
                    result[p + q + 1][i] += value * weight
    return tuple(tuple(row) for row in result)


def _join_state(ports, hidden, kernels):
    state = [_ZERO] * 54
    for i, value in zip(kernels.port_indices, ports):
        state[i] = value
    for indices, values in zip(kernels.hidden_component_indices, hidden):
        for i, value in zip(indices, values):
            state[i] = value
    return tuple(state)


def _causal_linear_series(kernels, initial, forcing=None):
    """Solve the six-port Volterra coefficient identity and reconstruct hidden rows."""
    order = kernels.order
    visible_initial = tuple(initial[i] for i in kernels.port_indices)
    hidden_initial = tuple(
        tuple(initial[i] for i in group) for group in kernels.hidden_component_indices
    )
    free_hidden, forced_hidden = [], []
    for part, group in enumerate(kernels.hidden_component_indices):
        free_hidden.append(
            tuple(
                _apply(_sparse(matrix), hidden_initial[part])
                for matrix in kernels.grounded_kernel_coefficients[part]
            )
        )
        if forcing is None:
            forced_hidden.append(((_ZERO,) * 16,) * (order + 1))
        else:
            hidden_force = tuple(tuple(row[i] for i in group) for row in forcing)
            forced_hidden.append(
                _kernel_integral(
                    kernels.grounded_kernel_coefficients[part], hidden_force, order
                )
            )
    source = []
    for n in range(order + 1):
        row = [_ZERO] * 6
        for part in range(3):
            hidden = tuple(
                a + b for a, b in zip(free_hidden[part][n], forced_hidden[part][n])
            )
            contribution = _apply(_sparse(kernels.hidden_to_visible[part]), hidden)
            row = [a + b for a, b in zip(row, contribution)]
        source.append(tuple(row))
    rows = [visible_initial]
    e = _sparse(kernels.visible_generator)
    memory = tuple(_sparse(matrix) for matrix in kernels.memory_kernel_coefficients)
    for n in range(order):
        rate = [a + b for a, b in zip(_apply(e, rows[n]), source[n])]
        if forcing is not None:
            rate = [
                value + forcing[n][i] for i, value in zip(kernels.port_indices, rate)
            ]
        for p in range(n):
            q = n - 1 - p
            contribution = _apply(memory[p], rows[q])
            weight = _beta(p, q)
            rate = [a + b * weight for a, b in zip(rate, contribution)]
        rows.append(tuple(value / (n + 1) for value in rate))
    hidden_rows = []
    for part in range(3):
        coupled = _kernel_integral(
            kernels.hidden_input_kernel_coefficients[part], rows, order
        )
        hidden_rows.append(
            tuple(
                tuple(
                    a + b + c
                    for a, b, c in zip(
                        free_hidden[part][n], forced_hidden[part][n], coupled[n]
                    )
                )
                for n in range(order + 1)
            )
        )
    full = tuple(
        _join_state(rows[n], tuple(part[n] for part in hidden_rows), kernels)
        for n in range(order + 1)
    )
    return full, tuple(source)


def _nonlinear_forcing(first, second, descriptor, *, order):
    """Original edge Q or 2Q+T force, using reconstructed hidden histories."""
    force = [[_ZERO] * 54 for _ in range(order)]
    for index, (left, right) in enumerate(_EDGES):
        p = tuple(row[27 + right] - row[27 + left] for row in first)
        square = _convolution(p, p, order - 1)
        if second is None:
            values = tuple(
                value * descriptor.quadratic_edge_scalar_bounds[index]
                for value in square
            )
        else:
            q = tuple(row[27 + right] - row[27 + left] for row in second)
            mixed = _convolution(p, q, order - 1)
            cube = _convolution(square, p, order - 1)
            values = tuple(
                2 * u * descriptor.quadratic_edge_scalar_bounds[index]
                + v * descriptor.cubic_edge_scalar_bounds[index]
                for u, v in zip(mixed, cube)
            )
        for n, value in enumerate(values):
            force[n][left] += value / descriptor.parameter_bounds.degrees[left]
            force[n][right] -= value / descriptor.parameter_bounds.degrees[right]
    return tuple(tuple(row) for row in force)


def _grounded_series(generator, initial, order):
    rows = [initial]
    matrix = _sparse(generator)
    for n in range(order):
        rows.append(tuple(value / (n + 1) for value in _apply(matrix, rows[-1])))
    return tuple(rows)


@dataclass(frozen=True)
class _CausalPortCoefficients:
    kernels: _PortKernelCoefficients
    nominal_level_coefficients: tuple[tuple[tuple[I, ...], ...], ...]
    quadratic_forcing_coefficients: tuple[tuple[I, ...], ...]
    cubic_forcing_coefficients: tuple[tuple[I, ...], ...]
    linear_source_coefficients: tuple[tuple[I, ...], ...]
    hidden_initialization_port_forcing_coefficients: tuple[tuple[I, ...], ...]
    comparator_nominal_coefficients: tuple[tuple[I, ...], ...]
    comparator_source_coefficients: tuple[tuple[I, ...], ...]


def _causal_coefficients(
    descriptor, port_impulse, initial_state, comparator_initial, *, order
):
    """Finite causal algebra; lower orders are for independent implementation controls."""
    kernels = _kernel_coefficients(descriptor, order=order)
    impulse = tuple(I(port_impulse[i]) if i < 3 else _ZERO for i in range(6))
    zero = (_ZERO,) * 54
    initial = _join_state(impulse, ((_ZERO,) * 16,) * 3, kernels)
    first, _ = _causal_linear_series(kernels, initial)
    quadratic = _nonlinear_forcing(first, None, descriptor, order=order)
    # Reflection makes the nominal second visible level exactly zero. Its
    # hidden odd functional survives and must feed the third-level force.
    hidden_second = []
    for part, indices in enumerate(kernels.hidden_component_indices):
        forcing = tuple(tuple(row[i] for i in indices) for row in quadratic)
        hidden_second.append(
            _kernel_integral(kernels.grounded_kernel_coefficients[part], forcing, order)
        )
    second = tuple(
        _join_state((_ZERO,) * 6, tuple(part[n] for part in hidden_second), kernels)
        for n in range(order + 1)
    )
    cubic = _nonlinear_forcing(first, second, descriptor, order=order)
    third, _ = _causal_linear_series(kernels, zero, cubic)
    source, initialization = _causal_linear_series(kernels, initial_state)
    return _CausalPortCoefficients(
        kernels,
        (first, second, third),
        quadratic,
        cubic,
        source,
        initialization,
        _grounded_series(kernels.visible_generator, impulse, order),
        _grounded_series(kernels.visible_generator, comparator_initial, order),
    )


def _port_time_tails(amplitude, horizon, *, order=_ORDER):
    g, h, rate = _GAMMA_UPPER, horizon, _LINEAR_NORM
    return (
        amplitude * _exponential_tail(rate * h, order + 1),
        2 * g * amplitude**2 * h * _exponential_tail(2 * rate * h, order),
        Q(4, 3) * g * amplitude**3 * h * _exponential_tail(3 * rate * h, order)
        + 4 * g**2 * amplitude**3 * h**2 * _exponential_tail(3 * rate * h, order - 1),
    )


def _endpoint(series, horizon, indices):
    state = tuple(series[-1][i] for i in indices)
    for row in reversed(series[:-1]):
        state = tuple(value * horizon + row[i] for value, i in zip(state, indices))
    return state


def _admit_box(raw, size, label):
    rows = _ordered(raw, label, limit=size + 1)
    if len(rows) != size:
        raise ValueError(f"{label} must have {size} coordinate pairs")
    pairs = []
    for i, row in enumerate(rows):
        pair = _ordered(row, f"{label}[{i}]", limit=3)
        if len(pair) != 2:
            raise ValueError(f"{label}[{i}] must have two endpoints")
        lo, hi = tuple(
            exact_or_represented_real(value, f"{label}[{i}]") for value in pair
        )
        if lo > hi:
            raise ValueError(f"{label}[{i}] lower endpoint exceeds upper endpoint")
        pairs.append((lo, hi))
    return tuple(pairs)


@dataclass(frozen=True)
class _CollectivePortPrediction:
    """Nominal causal prediction and independently retained initialization.

    Each phase source is y=theta-Theta_k at 0- about declared common origins.
    Physical source ranges are not numerical error. Numerical radius fields
    contain nominal polynomial rounding and its time tail only. The grounded
    comparator consumes its own visible source; no equality of residuals is
    imposed. Form fidelity concerns the complete sine model conditionally on
    the declared source cover and supplied event, not pattern acquisition.
    """

    mediator_class: int
    initial_form_bounds: tuple[tuple[Q, Q], ...]
    initial_phase_bounds: tuple[tuple[Q, Q], ...]
    comparator_initial_bounds: tuple[tuple[Q, Q], ...]
    port_impulse: tuple[Q, ...]
    horizon: Q
    source_radius: Q
    comparator_source_radius: Q
    descriptor: _CollectiveInterface
    coefficients: _CausalPortCoefficients
    nominal_level_endpoint_polynomials: tuple[tuple[I, ...], ...]
    nominal_level_time_tail_upper_bounds: tuple[Q, ...]
    nominal_endpoint_polynomials: tuple[I, ...]
    nominal_time_tail_upper_bound: Q
    nominal_endpoint_bounds: tuple[I, ...]
    nominal_numerical_radius_upper_bounds: tuple[Q, ...]
    linear_source_endpoint_polynomials: tuple[I, ...]
    linear_source_time_tail_upper_bound: Q
    linear_source_endpoint_bounds: tuple[I, ...]
    linear_source_uniform_bound: Q
    comparator_nominal_endpoint_polynomials: tuple[I, ...]
    comparator_nominal_time_tail_upper_bound: Q
    comparator_nominal_endpoint_bounds: tuple[I, ...]
    comparator_numerical_radius_upper_bounds: tuple[Q, ...]
    comparator_source_endpoint_polynomials: tuple[I, ...]
    comparator_source_time_tail_upper_bound: Q
    comparator_source_endpoint_bounds: tuple[I, ...]
    comparator_source_uniform_bound: Q
    fidelity: _CollectiveInterfaceBound
    order: int = _ORDER
    method: str = "grounded_path_volterra_beta_time_polynomials_v1"
    arithmetic_method: str = INTERVAL_METHOD
    source_coordinates: str = (
        "pre-input form x and phase deviation y=theta-Theta_k at 0-"
    )
    comparator_model: str = "grounded_tangent_port_truncation_v_prime_equals_E_v"
    scope: tuple[str, ...] = (
        "port_order_x4_x13_x22_y4_y13_y22",
        "full_hidden_source_BG_h0_is_retained_in_the_causal_linear_initialization",
        "nominal_hidden_quadratic_functional_recouples_into_the_cubic_port_force",
        "nominal_parity_does_not_impose_reflection_on_the_actual_initial_source",
        "polynomial_and_time_tail_are_separate_evidence_then_added_once",
        "numerical_radius_excludes_physical_source_and_amplitude_truncation_error",
        "source_series_enclosures_and_heat_comparison_envelopes_are_separate_valid_bounds",
        "comparator_initial_visible_residuals_are_independent_supplied_primitives",
        "original_declared_common_origins_are_retained_externally_not_evolved_by_E",
        "no_full_state_coefficient_recurrence_nonlinear_flow_or_reserved_response_is_used",
        "no_source_acquisition_work_identity_sensor_or_physical_identification_verdict",
    )


def _predict_collective_port_response(
    *,
    mediator_class,
    initial_form_bounds,
    initial_phase_bounds,
    comparator_initial_bounds,
    port_impulse,
    horizon,
) -> _CollectivePortPrediction:
    """Evaluate the fixed degree-32 causal interface for one central form event.

    All source pairs describe deviations before that event. Horizons lie in
    [0,2] and total absolute impulse is at most 7/5000. No supplied source
    box certifies acquisition, centering, charge correlations or identity.
    """
    if type(mediator_class) is not int or mediator_class not in (1, 2):
        raise ValueError("mediator class must be ordinary integer one or two")
    form = _admit_box(initial_form_bounds, 27, "initial_form_bounds")
    phase = _admit_box(initial_phase_bounds, 27, "initial_phase_bounds")
    comparator = _admit_box(comparator_initial_bounds, 6, "comparator_initial_bounds")
    impulses = _ordered(port_impulse, "port_impulse", limit=4)
    if len(impulses) != 3:
        raise ValueError("port_impulse must have three signed scalars")
    impulses = tuple(
        exact_or_represented_real(value, "port_impulse") for value in impulses
    )
    duration = exact_or_represented_real(horizon, "horizon")
    amplitude = sum(abs(value) for value in impulses)
    eps = max(abs(value) for pair in form + phase for value in pair)
    comparator_eps = max(abs(value) for pair in comparator for value in pair)
    fidelity = _bound_collective_interface(
        total_input_variation=amplitude, horizon=duration, endpoint_radius=eps
    )
    descriptor = _derive_collective_interface(mediator_class)
    source = tuple(I(*pair) for pair in form + phase)
    comparator_source = tuple(I(*pair) for pair in comparator)
    coefficients = _causal_coefficients(
        descriptor, impulses, source, comparator_source, order=_ORDER
    )
    ports = descriptor.port_indices
    level_endpoints = tuple(
        _endpoint(rows, duration, ports)
        for rows in coefficients.nominal_level_coefficients
    )
    tails = _port_time_tails(amplitude, duration)
    nominal = tuple(a + b for a, b in zip(level_endpoints[0], level_endpoints[2]))
    nominal_tail = tails[0] + tails[2]
    nominal_bounds = tuple(value + I(-nominal_tail, nominal_tail) for value in nominal)
    source_poly = _endpoint(coefficients.linear_source_coefficients, duration, ports)
    exponential_tail = _exponential_tail(_LINEAR_NORM * duration, _ORDER + 1)
    source_tail = eps * exponential_tail
    source_bounds = tuple(value + I(-source_tail, source_tail) for value in source_poly)
    comparator_poly = _endpoint(
        coefficients.comparator_nominal_coefficients, duration, range(6)
    )
    comparator_tail = tails[0]
    comparator_bounds = tuple(
        value + I(-comparator_tail, comparator_tail) for value in comparator_poly
    )
    comparator_source_poly = _endpoint(
        coefficients.comparator_source_coefficients, duration, range(6)
    )
    comparator_source_tail = comparator_eps * exponential_tail
    comparator_source_bounds = tuple(
        value + I(-comparator_source_tail, comparator_source_tail)
        for value in comparator_source_poly
    )
    return _CollectivePortPrediction(
        mediator_class,
        form,
        phase,
        comparator,
        impulses,
        duration,
        eps,
        comparator_eps,
        descriptor,
        coefficients,
        level_endpoints,
        tails,
        nominal,
        nominal_tail,
        nominal_bounds,
        tuple(value.radius for value in nominal_bounds),
        source_poly,
        source_tail,
        source_bounds,
        eps / fidelity.flow_comparison_margin,
        comparator_poly,
        comparator_tail,
        comparator_bounds,
        tuple(value.radius for value in comparator_bounds),
        comparator_source_poly,
        comparator_source_tail,
        comparator_source_bounds,
        comparator_eps / fidelity.flow_comparison_margin,
        fidelity,
    )
