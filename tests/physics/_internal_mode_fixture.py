"""Shared exact P2 x C3 fixture; constructing inputs is not a trajectory.

The rational generator uses exact conductances and degrees. It is distinct
from a materialized binary64 Laplacian or a fresh binary64 pressure kernel.
Primitive phase, capacities, support and pressure are declared independently.
"""

from dataclasses import replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.physics._cycle_algebra import dot
from tnfr.physics.support_transport import (
    _laplacian,
    observe_regional_support_balance,
    observe_support_transport,
)

P = (Q(1), Q(-1), Q(0))
Q_MODE = (Q(1), Q(1), Q(-2))
GRAM = (Q(2), Q(6))
NODES = tuple((a, i) for a in range(2) for i in range(3))
LIFT = tuple(
    tuple(
        (P[i] if k == 0 else Q_MODE[i]) if a == b else Q(0)
        for b in range(2)
        for k in range(2)
    )
    for a, i in NODES
)
PROJECTION = tuple(
    tuple((P[i] / 2 if k == 0 else Q_MODE[i] / 6) if a == b else Q(0) for a, i in NODES)
    for b in range(2)
    for k in range(2)
)
INDUCED = (
    (Q(-4, 3), Q(0), Q(1, 3), Q(0)),
    (Q(0), Q(-4, 3), Q(0), Q(1, 3)),
    (Q(1, 3), Q(0), Q(-4, 3), Q(0)),
    (Q(0), Q(1, 3), Q(0), Q(-4, 3)),
)


def _apply(matrix, values):
    return tuple(dot(row, values) for row in matrix)


def _inner(left, right):
    return sum((w * a * b for w, a, b in zip(GRAM, left, right, strict=True)), Q(0))


def _graph(
    coefficients=(Q(1, 4), Q(0), Q(1, 4), Q(0)),
    *,
    means=(Q(1, 2), Q(1, 2)),
):
    graph = nx.cartesian_product(nx.path_graph(2), nx.cycle_graph(3))
    assert tuple(graph) == NODES
    graph.graph["DNFR_WEIGHTS"] = {"epi": 1.0, "phase": 0.0, "vf": 0.0, "topo": 0.0}
    for node, deviation in zip(graph, _apply(LIFT, coefficients), strict=True):
        graph.nodes[node].update(
            EPI=means[node[0]] + deviation, nu_f=1.0, theta=0.0, delta_nfr=0.0
        )
    return graph


def _exact_generator(snapshot):
    size = len(snapshot.nodes)
    strengths = tuple(
        sum((w for i, _, w in snapshot.conductance if i == row), Q(0))
        for row in range(size)
    )
    columns = tuple(
        tuple(
            -nu * value / d
            for nu, value, d in zip(
                snapshot.capacity,
                _laplacian(snapshot.conductance, tuple(Q(i == j) for i in range(size))),
                strengths,
                strict=True,
            )
        )
        for j in range(size)
    )
    return tuple(zip(*columns, strict=True))


def _phase_support():
    """Return the same prism with ordered unique-neighbor phase incidence."""
    graph = _graph()
    position = {node: index for index, node in enumerate(NODES)}
    rows = tuple(
        tuple(position[other] for other in graph.neighbors(node)) for node in NODES
    )
    return graph, rows


def _prepared_phase_geometry():
    """Exact symbolic prepared phase data; the returned graph is not evolved."""
    s = pytest.importorskip("sympy")
    graph, rows = _phase_support()
    phases = (0, s.pi / 3, s.pi / 6) * 2
    center = s.pi / 6
    cosine = tuple(s.cos(phase - center) for phase in phases)
    gram = s.Matrix(6, 6, lambda j, k: s.cos(phases[j] - phases[k]))
    response = s.Matrix(
        6, 6, lambda i, j: (cosine[j] / (1 + s.sqrt(3)) if j in rows[i] else 0)
    )
    return s, graph, rows, phases, center, cosine, gram, response


def _prepared_phase_metric():
    """Return sympy, graph, phases, H_phi, g and Dg for the same preparation."""
    s, graph, _, phases, center, _, _, response = _prepared_phase_geometry()
    resultant = 1 + s.sqrt(3)
    displacements = tuple(center - phase for phase in phases)
    metric = s.diag(
        *(
            s.pi * resultant if delta == 0 else s.pi * resultant * s.sin(delta) / delta
            for delta in displacements
        )
    )
    source = s.Matrix([delta / s.pi for delta in displacements])
    jacobian = (response - s.eye(6)) / s.pi
    return s, graph, phases, metric, source, jacobian


def _nonrepeated_phase_geometry():
    """Primitive data of the retained boundary-source preparation, not a flow."""
    s = pytest.importorskip("sympy")
    graph, rows = _phase_support()
    angle = s.pi / 6
    return s, graph, rows, (-angle, angle, 0, -angle, angle, angle)


def _symbolic_phase_response(s, phases, rows):
    """Shared exact phasor derivative for surd fixtures, not binary64 input.

    This uses the production owner's cosine-sum identity. Independent tests
    differentiate the actual Arg expression; irrational cosine data must not
    be rounded into the exact-rational production geometry boundary.
    """
    gram = s.Matrix(len(phases), len(phases), lambda i, j: s.cos(phases[i] - phases[j]))
    squared = tuple(
        s.simplify(sum(gram[j, k] for j in row for k in row)) for row in rows
    )
    response = s.Matrix(
        len(phases),
        len(phases),
        lambda i, j: (
            s.simplify(sum(gram[j, k] for k in rows[i]) / squared[i])
            if j in rows[i]
            else 0
        ),
    )
    return response, squared


def _metric_differential(s, rows, phases):
    """Differentiate both resultant magnitude and angular displacement."""
    response, squared = _symbolic_phase_response(s, phases, rows)
    real = tuple(s.simplify(sum(s.cos(phases[j]) for j in row)) for row in rows)
    imag = tuple(s.simplify(sum(s.sin(phases[j]) for j in row)) for row in rows)
    assert all(value.is_positive for value in real)
    radius = tuple(s.sqrtdenest(s.sqrt(value)) for value in squared)
    displacement = tuple(
        s.simplify(s.atan(y / x) - angle)
        for x, y, angle in zip(real, imag, phases, strict=True)
    )
    sine = tuple(
        s.simplify((y * s.cos(angle) - x * s.sin(angle)) / r)
        for x, y, r, angle in zip(real, imag, radius, phases, strict=True)
    )
    cosine = tuple(
        s.simplify((x * s.cos(angle) + y * s.sin(angle)) / r)
        for x, y, r, angle in zip(real, imag, radius, phases, strict=True)
    )
    # sinc(0)=1 and sinc'(0)=0 are removable values, not tolerances or
    # replacement pressures. They occur at two nodes of the prepared state.
    sinc = tuple(
        1 if delta == 0 else sin / delta
        for sin, delta in zip(sine, displacement, strict=True)
    )
    sinc_derivative = tuple(
        0 if delta == 0 else (delta * cos - sin) / delta**2
        for sin, cos, delta in zip(sine, cosine, displacement, strict=True)
    )
    metric = s.diag(
        *(s.simplify(s.pi * r * value) for r, value in zip(radius, sinc, strict=True))
    )
    magnitude_response = s.Matrix(
        len(phases),
        len(phases),
        lambda i, j: (
            s.simplify(
                (imag[i] * s.cos(phases[j]) - real[i] * s.sin(phases[j])) / radius[i]
            )
            if j in rows[i]
            else 0
        ),
    )
    displacement_response = response - s.eye(len(phases))
    derivative = s.Matrix(
        len(phases),
        len(phases),
        lambda i, j: (
            s.pi * sinc[i] * magnitude_response[i, j]
            + s.pi * radius[i] * sinc_derivative[i] * displacement_response[i, j]
        ),
    )
    source = s.Matrix(displacement) / s.pi
    return metric, derivative, source, displacement_response / s.pi


def _unit_regional_budget(snapshot, forcing, *, e=Q(1, 2)):
    """Exact declared rate budget on unit-capacity prism, never an update."""
    assert snapshot.nodes == NODES and snapshot.capacity == (1,) * 6
    pressure = tuple(
        e * g + f for g, f in zip(snapshot.epi_gradient, forcing, strict=True)
    )
    declared = replace(snapshot, stored_pressure=pressure)
    rows = tuple(
        observe_regional_support_balance(
            declared, region, epi_weight=e, forcing=forcing
        )
        for region in (NODES[:3], NODES[3:])
    )
    assert all(row.stored_pressure_defect == (0,) * 6 for row in rows)
    squared = Q(2, 3) * sum(row.variance for row in rows)
    rate = Q(2, 3) * sum(row.stored_variance_rate for row in rows)
    work = Q(2, 3) * sum(row.variance_forcing_rate for row in rows)
    return squared, rate, work


def _prepared_pressure_coordinates():
    """Existing exact centered chart; it does not select an acceleration law."""
    s, graph, rows, phases, center, cosine, _, response = _prepared_phase_geometry()
    source = observe_support_transport(graph)
    laplacian = -s.Matrix(_exact_generator(source))
    common = s.ones(6) / 6
    projection = s.eye(6) - common

    def normalize(value):
        return s.expand(s.radsimp(value))

    scaled_source = (response - s.eye(6)).applyfunc(normalize)  # pi*Dg.
    projected = (projection * scaled_source).applyfunc(normalize)
    inverse = ((projected + common).inv() * projection).applyfunc(normalize)
    return (
        s,
        graph,
        rows,
        phases,
        center,
        cosine,
        laplacian,
        projection,
        scaled_source,
        inverse,
    )
