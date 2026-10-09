"""Exact regional form-zero passage under the existing closed EPI diffusion.

This is a detached analytic continuation, not a native engine run, a new
phase law or a modification of the frozen selector experiment. A regional
polar observation loses its domain while the complete Cartesian nodal state
remains regular. Neighboring form can regenerate regional disagreement even
as total internal disagreement decreases.
"""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from math import isfinite, pi

import pytest

from tests.physics._internal_mode_fixture import (
    INDUCED,
    NODES,
    PROJECTION,
    _apply,
    _exact_generator,
    _graph,
    _inner,
    _unit_regional_budget,
)
from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)
from tnfr.physics.support_transport import observe_regional_support_balance


def _coefficients(q):
    return (q**3 * (1 - 4 * q**2) / 32, 0, q**3 * (1 + 4 * q**2) / 32, 0)


def _reference(q):
    graph = _graph(_coefficients(q))
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5,
        "phase": 0.25,
        "vf": 0.25,
        "topo": 0.0,
    }
    return graph, capture_non_epi_forcing(graph)


def test_exact_existing_diffusion_reaches_and_crosses_a_regional_form_zero():
    s = pytest.importorskip("sympy")
    q, e, time = s.symbols("q e time", positive=True)
    coefficients = s.Matrix(_coefficients(q))
    induced = s.Matrix(INDUCED)
    rate = -e * q * coefficients.diff(q) / 3
    assert s.simplify(rate - e * induced * coefficients) == s.zeros(4, 1)
    assert induced.rank() == 4
    # q decreases from one, with the clock inherited from the two eigenrates
    # e and 5e/3. It is not a supplied primitive-phase frequency or event cut.
    assert s.expand(coefficients[0] / q**3 + (2 * q - 1) * (2 * q + 1) / 32) == 0
    assert s.solve(coefficients[0] / q**3, q) == [s.Rational(1, 2)]
    crossing = 3 * s.log(2) / e
    assert s.exp(-e * crossing / 3) == s.Rational(1, 2)
    assert coefficients.subs(q, s.Rational(1, 2)) == s.Matrix(
        [0, 0, s.Rational(1, 128), 0]
    )
    assert rate.subs(q, s.Rational(1, 2)) == s.Matrix([e / 384, 0, -e / 96, 0])
    # The entire finite-time curve is analytic. For 0<q<=1, |u_a|<=5/32 by
    # the triangle inequality, so all six scalar EPIs lie in [11/32,21/32].
    curve = coefficients.subs(q, s.exp(-e * time / 3))
    assert s.simplify(curve.diff(time) - e * induced * curve) == s.zeros(4, 1)
    assert tuple(_coefficients(Q(1))) == (Q(-3, 32), 0, Q(5, 32), 0)


def test_polar_domain_loss_leaves_fine_source_and_primitive_phase_regular():
    s = pytest.importorskip("sympy")
    graph, observed = _reference(Q(1, 2))
    before = deepcopy(graph)
    snapshot = observed.snapshot
    assert _apply(PROJECTION, snapshot.epi) == (0, 0, Q(1, 128), 0)
    fine_generator = _exact_generator(snapshot)
    fine_rate = tuple(value / 2 for value in _apply(fine_generator, snapshot.epi))
    assert _apply(PROJECTION, fine_rate) == (Q(1, 768), 0, Q(-1, 192), 0)
    assert s.Matrix(fine_generator).rank() == 5
    assert fine_rate[:3] == (Q(1, 768), Q(-1, 768), 0)
    assert observed.phase_gradient == observed.forcing == (0,) * 6
    assert all(isfinite(float(value)) for value in observed.full_kernel_pressure)
    assert observed.full_kernel_pressure[0] > 0 > observed.full_kernel_pressure[1]
    # Native division by three remains a separately captured arithmetic defect.
    assert observed.kernel_pressure_defect != (0,) * 6
    assert (
        tuple(
            actual - ideal
            for actual, ideal in zip(
                observed.full_kernel_pressure, fine_rate, strict=True
            )
        )
        == observed.kernel_pressure_defect
    )
    index = {node: i for i, node in enumerate(NODES)}
    response = derive_phase_response(
        cosine_gram=((1,) * 6,) * 6,
        mean_neighbors=tuple(
            tuple(index[neighbor] for neighbor in graph.neighbors(node))
            for node in NODES
        ),
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=1,
    )
    geometry = observe_phase_source_geometry(response)
    assert response.mean_resultant_squared == (9,) * 6
    assert geometry.rank == 5 and geometry.only_common_rotation
    # These are three exact points on the analytic curve, not time-stepped
    # samples. Multiplication by sqrt(2) would not change their real-axis ray.
    directions = tuple(
        reduce_phasor_components(((float(_coefficients(q)[0]), 0.0),))
        for q in (Q(3, 4), Q(1, 2), Q(1, 4))
    )
    assert directions[0].angle == pi and not directions[0].joint_zero
    assert directions[1].angle is None and directions[1].joint_zero
    assert directions[2].angle == 0 and not directions[2].joint_zero
    assert graph.graph == before.graph
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(graph.edges) == dict(before.edges)


def test_regional_regrowth_is_boundary_transfer_during_global_decay():
    # This exact point lies just after q=1/2; no new phase or capacity term is
    # installed. The rational reference pressure is explicitly detached.
    graph, observed = _reference(Q(7, 16))
    source = observed.snapshot
    fine_rate = tuple(
        value / 2 for value in _apply(_exact_generator(source), source.epi)
    )
    declared = replace(source, stored_pressure=fine_rate)
    regional = observe_regional_support_balance(
        declared, NODES[:3], epi_weight=Q(1, 2), forcing=(0,) * 6
    )
    coefficients = _apply(PROJECTION, source.epi)
    rate = _apply(PROJECTION, fine_rate)
    assert coefficients[0] > 0 and rate[0] > 0
    local_rate = 2 * _inner(coefficients[:2], rate[:2])
    assert local_rate == Q(2, 3) * regional.stored_variance_rate > 0
    assert regional.variance_forcing_rate == regional.variance_defect_rate == 0
    assert regional.variance_boundary_rate > regional.internal_dissipation
    norm, total_rate, source_work = _unit_regional_budget(source, (0,) * 6)
    assert norm > 0 and total_rate < 0 and source_work == 0
    assert total_rate == 2 * (
        _inner(coefficients[:2], rate[:2]) + _inner(coefficients[2:], rate[2:])
    )
    assert tuple(graph.nodes) == NODES and graph.number_of_edges() == 9
