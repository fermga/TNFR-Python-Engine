"""Scope controls for observing phase support from the passive unit prism.

No feedback or trajectory is executed. Equivariance and continuity at the
consensus state are additional observation hypotheses; polar normalization
does not satisfy the latter. A diagnostic-work bound on passive motion is
not transferred to a different, source-coupled evolution.
"""

from fractions import Fraction as Q
from math import pi

import pytest

from tests.physics._internal_mode_fixture import (
    INDUCED,
    LIFT,
    NODES,
    PROJECTION,
    _apply,
    _graph,
    _inner,
)
from tests.physics._internal_mode_fixture import _unit_regional_budget as _budget
from tnfr.mathematics.krylov import exact_rank
from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.support_transport import observe_support_transport

ZERO = (Q(0),) * 6


def _phase_source():
    graph = _graph((Q(1, 16), 0, Q(1, 16), 0))
    graph.graph["DNFR_WEIGHTS"] = {
        "epi": 0.5,
        "phase": 0.25,
        "vf": 0.25,
        "topo": 0.0,
    }
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = (0, pi / 3, pi / 6)[i]
    return graph


def test_prism_equivariance_forces_circle_observations_to_be_consensus_there():
    graph = _graph()
    generators = (
        tuple((a, (i + 1) % 3) for a, i in NODES),
        tuple((a, (1, 0, 2)[i]) for a, i in NODES),
        tuple((1 - a, i) for a, i in NODES),
    )
    edge_set = {frozenset(edge) for edge in graph.edges}
    constraints = []
    consensus = (Q(1, 2),) * 6
    for images in generators:
        permutation = dict(zip(NODES, images, strict=True))
        assert {
            frozenset((permutation[i], permutation[j])) for i, j in graph.edges
        } == edge_set
        matrix = tuple(tuple(Q(target == node) for node in NODES) for target in images)
        assert _apply(matrix, consensus) == consensus
        constraints.extend(
            tuple(value - Q(i == j) for j, value in enumerate(row))
            for i, row in enumerate(matrix)
        )
    assert exact_rank(tuple(constraints)) == 5
    assert _apply(tuple(constraints), (Q(1),) * 6) == (0,) * 18
    # An equivariant observation at this invariant input must lie in this
    # fixed space. Apply the rank result to both real components of a circle
    # field: all six unit vectors agree. No linearity of the observation
    # itself, arbitrary phase representative, or runtime gauge claim is used.


def test_smooth_constant_observation_need_not_be_consensus_compatible():
    graph = _phase_source()
    for node in graph:
        graph.nodes[node]["EPI"] = Q(1, 2)
    observed = capture_non_epi_forcing(graph)
    assert observed.snapshot.epi_gradient == ZERO
    assert any(observed.forcing)
    assert _apply(PROJECTION, observed.forcing) != (0,) * 4
    # Theta(x)=this fixed prepared phase profile is smooth in x. Its nonzero
    # source is supplied at consensus, not an emergent residual of EPI.
    # The cycle automorphism does not fix the prepared circle values.
    phase_turns = (Q(0), Q(1, 3), Q(1, 6)) * 2
    cycled = tuple(phase_turns[NODES.index((a, (i + 1) % 3))] for a, i in NODES)
    assert cycled != phase_turns
    assert all(0 <= value < 2 for value in phase_turns)


def test_polar_observation_has_distinct_limits_along_exact_decaying_eigenrays():
    symbolic = pytest.importorskip("sympy")
    amplitude = symbolic.Symbol("amplitude", positive=True)
    first, second = (Q(1), Q(0), Q(1), Q(0)), (Q(0), Q(1), Q(0), Q(1))
    for ray in (first, second):
        assert _apply(INDUCED, ray) == tuple(-value for value in ray)
        fine = _apply(LIFT, ray)
        assert symbolic.Matrix(fine) * amplitude != symbolic.zeros(6, 1)
        assert (symbolic.Matrix(fine) * amplitude).limit(
            amplitude, 0
        ) == symbolic.zeros(6, 1)
    # In the inherited orthonormal complex chart the two rays point along
    # distinct axes for every amplitude>0; both Cartesian states tend to0.
    first_unit = symbolic.Matrix([symbolic.sqrt(2) * amplitude, 0]) / (
        symbolic.sqrt(2) * amplitude
    )
    second_unit = symbolic.Matrix([0, symbolic.sqrt(6) * amplitude]) / (
        symbolic.sqrt(6) * amplitude
    )
    assert first_unit == symbolic.Matrix([1, 0])
    assert second_unit == symbolic.Matrix([0, 1])
    assert first_unit != second_unit
    zero = reduce_phasor_components(((0.0, 0.0),))
    assert zero.joint_zero and zero.angle is None
    # No continuous common extension or assigned phase at the origin follows.


def test_bounded_diagnostic_phase_work_is_integrable_along_the_passive_ray():
    symbolic = pytest.importorskip("sympy")
    observed = capture_non_epi_forcing(_phase_source())
    e = observed.epi_weight
    squared, passive_rate, passive_work = _budget(observed.snapshot, ZERO, e=e)
    assert squared == Q(1, 64) and passive_rate == -2 * e * squared
    assert passive_work == 0
    _, coupled_rate, diagnostic_work = _budget(observed.snapshot, observed.forcing, e=e)
    assert coupled_rate > 0  # This is a different declared rate, not passive motion.
    forcing_squared = sum(value**2 for value in observed.forcing)
    weight = dict(observed.normalized_weights)["phase"]
    assert forcing_squared <= 6 * weight**2
    assert diagnostic_work**2 <= 4 * squared * forcing_squared
    time = symbolic.Symbol("time", nonnegative=True)
    work_on_passive_ray = symbolic.Rational(diagnostic_work) * symbolic.exp(-e * time)
    integral = symbolic.integrate(work_on_passive_ray, (time, 0, symbolic.oo))
    assert integral == diagnostic_work / e
    # The universal bounded-source envelope does not require the prepared
    # phase profile to decay: |work| <=2*w_phi*sqrt(6*S0)*exp(-e*t).
    envelope = 2 * weight * symbolic.sqrt(6 * squared) * symbolic.exp(-e * time)
    envelope_integral = symbolic.integrate(envelope, (time, 0, symbolic.oo))
    assert envelope_integral == 2 * weight * symbolic.sqrt(6 * squared) / e
    assert (envelope_integral - integral).is_positive
    # The passive exponential cannot be used after feeding this source into
    # the fine dynamics, as the opposite initial rate signs already show.


def test_sharp_support_gate_requires_alignment_not_just_a_large_source_norm():
    e = Q(1, 2)
    for sign, threshold in ((1, e), (-1, Q(5, 3) * e)):
        source = observe_support_transport(_graph((Q(1, 16), 0, sign * Q(1, 16), 0)))
        c = _apply(PROJECTION, source.epi)
        internal = _apply(LIFT, c)
        squared = _inner(c[:2], c[:2]) + _inner(c[2:], c[2:])
        gap = tuple(a - b for a, b in zip(c[:2], c[2:], strict=True))
        disagreement = _inner(gap, gap)
        assert threshold == e * (1 + disagreement / (3 * squared))
        aligned = tuple(threshold * value for value in internal)
        norm, rate, work = _budget(source, aligned, e=e)
        assert norm == squared and rate == 0
        assert work == 2 * threshold * squared
        assert sum(value**2 for value in aligned) == threshold**2 * squared
        wrong_sign = tuple(-value for value in aligned)
        assert _budget(source, wrong_sign, e=e)[1] == -4 * threshold * squared
        transverse = _apply(LIFT, (0, Q(1, 4), 0, Q(1, 4)))
        assert sum(a * b for a, b in zip(internal, transverse, strict=True)) == 0
        assert sum(value**2 for value in transverse) > threshold**2 * squared
        _, transverse_rate, transverse_work = _budget(source, transverse, e=e)
        assert transverse_work == 0
        assert transverse_rate == -2 * threshold * squared < 0
    # These are conditional source vectors, not fitted phase profiles or a
    # proposed constitutive law. Norm size is necessary but not sufficient.
