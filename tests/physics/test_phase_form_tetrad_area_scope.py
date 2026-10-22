"""Restricted oriented-area readout from the existing potential and curvature.

Exact identities require repeated triples, unit path lengths, constant unit
capacity and fresh ideal pressure. Detached binary64 readers have separately
retained pressure defects. The readout does not reconstruct common EPI/phase,
authenticate arbitrary stored pressure or establish general tetrad closure.
"""

import math
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    Q_MODE,
    P,
    _exact_generator,
    _graph,
    _prepared_phase_geometry,
)
from tests.physics.test_internal_mode_macro_state import _potential_matrix
from tests.physics.test_internal_mode_tetrad import _prepare_pressure
from tnfr.physics.fields import compute_phase_curvature, compute_structural_potential
from tnfr.physics.support_transport import observe_support_transport


def _geometry():
    symbolic = pytest.importorskip("sympy")
    graph = _graph()
    nx.set_edge_attributes(graph, 1.0, "length")
    kernel = symbolic.Matrix(_potential_matrix(graph))
    basis = symbolic.Matrix.hstack(
        symbolic.Matrix(P) / symbolic.sqrt(2),
        symbolic.Matrix(Q_MODE) / symbolic.sqrt(6),
    )
    return symbolic, graph, kernel, basis, basis.col_join(basis)


def test_actual_distance_kernel_recovers_repeated_internal_area_but_not_gauges():
    s, graph, kernel, basis, lift = _geometry()
    # The reused exact kernel uses hop distance; explicit unit lengths make
    # that exactly the physical distance of the public potential reader here.
    metric_distances = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    for i, left in enumerate(NODES):
        for j, right in enumerate(NODES):
            expected = 0 if i == j else Q(1, int(metric_distances[left][right]) ** 2)
            assert kernel[i, j] == expected
    ones = s.ones(6, 1)
    assert kernel * ones == s.Rational(7, 2) * ones
    assert s.simplify(kernel * lift + lift / 4) == s.zeros(6, 2)
    assert s.simplify(basis.T * basis) == s.eye(2)
    e, w = s.symbols("e w", positive=True)
    x, y, a, b, mu, beta, common = s.symbols("x y a b mu beta common", real=True)
    form, phase = s.Matrix([x, y]), s.Matrix([a, b])
    generator = s.Matrix(_exact_generator(observe_support_transport(graph)))
    epi = mu * ones + lift * form
    theta = beta * ones + lift * phase
    curvature = theta - (beta + s.pi * common) * ones
    pressure = e * generator * epi + w * (common * ones - lift * phase / s.pi)
    potential = kernel * pressure
    projected_phi = s.simplify(basis.T * potential[:3, :])
    projected_k = s.simplify(basis.T * curvature[:3, :])
    assert s.simplify(projected_phi - (e * form + (w / s.pi) * phase) / 4) == s.zeros(
        2, 1
    )
    assert projected_k == phase
    assert (
        s.simplify(4 * s.det(s.Matrix.hstack(projected_phi, projected_k)) / e)
        == x * b - y * a
    )
    assert potential.diff(mu) == curvature.diff(beta) == s.zeros(6, 1)
    # Bind the curvature formula to the already prepared strict phase chart.
    _, _, _, phases, center, _, _, _ = _prepared_phase_geometry()
    prepared_k = s.Matrix([value - center for value in phases])
    assert s.simplify(lift * basis.T * prepared_k[:3, :] - prepared_k) == s.zeros(6, 1)
    assert max(prepared_k) - min(prepared_k) == s.pi / 3


def test_detached_fresh_pressure_and_public_tetrad_readers_resolve_nonzero_area():
    s, _, kernel, basis, _ = _geometry()
    graph = _graph((Q(1, 8), Q(1, 16), Q(1, 8), Q(1, 16)))
    graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0.0)
    nx.set_edge_attributes(graph, 1.0, "length")
    for fiber, i in NODES:
        graph.nodes[fiber, i]["theta"] = (0.0, math.pi / 3, math.pi / 6)[i]
    captured = _prepare_pressure(
        graph
    )  # Fresh values populate detached fixture data only.
    assert dict(captured.normalized_weights) == {
        "phase": Q(1, 4),
        "epi": Q(1, 2),
        "vf": Q(1, 4),
        "topo": Q(0),
    }
    assert (
        captured.snapshot.capacity_gradient
        == captured.snapshot.topology_gradient
        == (0,) * 6
    )
    assert captured.stored_pressure_residual == (0,) * 6
    assert captured.full_kernel_pressure == tuple(
        captured.epi_weight * gradient + source + defect
        for gradient, source, defect in zip(
            captured.snapshot.epi_gradient,
            captured.forcing,
            captured.kernel_pressure_defect,
            strict=True,
        )
    )
    numeric_basis = np.asarray(basis, dtype=float)
    form = numeric_basis.T @ np.array(list(map(float, captured.snapshot.epi[:3])))
    phase = numeric_basis.T @ np.array(list(map(float, captured.phase[:3])))
    phi = compute_structural_potential(graph, alpha=2.0)
    curvature = compute_phase_curvature(graph)
    projected_phi = numeric_basis.T @ np.array([phi[node] for node in NODES[:3]])
    projected_k = numeric_basis.T @ np.array([curvature[node] for node in NODES[:3]])
    expected_potential = kernel * s.Matrix(captured.full_kernel_pressure)
    np.testing.assert_allclose(
        [phi[node] for node in NODES],
        list(map(float, expected_potential)),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(projected_k, phase, rtol=0, atol=1e-15)
    area = float(np.linalg.det(np.column_stack((form, phase))))
    observed = (
        4 * float(np.linalg.det(np.column_stack((projected_phi, projected_k)))) / 0.5
    )
    assert area == pytest.approx(math.sqrt(3) * math.pi / 48, rel=0, abs=1e-15)
    assert observed == pytest.approx(area, rel=0, abs=1e-14)
    # This tolerance controls floating readouts; it does not erase the exact
    # kernel-pressure defects above or certify an exact binary64 identity.


def test_repeated_stored_pressure_defect_has_an_exact_scoped_area_correction():
    s, _, kernel, basis, lift = _geometry()
    e, k = s.symbols("e k", positive=True)
    x, y, a, b, d0, d1, q = s.symbols("x y a b d0 d1 q", real=True)
    form, phase, defect = s.Matrix([x, y]), s.Matrix([a, b]), s.Matrix([d0, d1])
    pressure = q * s.ones(6, 1) + lift * (-e * form - k * phase)
    stored = pressure + lift * defect
    projected_phi = s.simplify(basis.T * (kernel * stored)[:3, :])
    area = s.det(s.Matrix.hstack(form, phase))
    defect_area = s.det(s.Matrix.hstack(defect, phase))
    read_area = 4 * s.det(s.Matrix.hstack(projected_phi, phase)) / e
    assert s.simplify(read_area - area + defect_area / e) == 0
    assert (
        s.simplify(
            (read_area - area).subs({e: s.Rational(1, 2), d0: 1, d1: 0, a: 0, b: 1})
        )
        == -2
    )
    # The simple correction requires the defect to repeat across fibers.
    # An opposite-fiber defect sees eigenvalue -7/4, not the repeated -1/4.
    opposite = (basis * defect).col_join(-basis * defect)
    opposite_projection = s.simplify(basis.T * (kernel * opposite)[:3, :])
    assert opposite_projection == -s.Rational(7, 4) * defect
    opposite_area_error = 4 * s.det(s.Matrix.hstack(opposite_projection, phase)) / e
    assert s.simplify(opposite_area_error + 7 * defect_area / e) == 0
    # Delta is relative to the declared ideal pressure here. Runtime stale
    # pressure and fresh-kernel error must both be retained when applying it.
