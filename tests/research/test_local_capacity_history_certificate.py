"""Independent local-history admission and full-field controls.

These fixtures are synthetic. No reserved numerical history is generated and
no source producer or operator is executed by this module.
"""

from dataclasses import replace
from fractions import Fraction as Q
from math import pi

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.research import local_capacity_history_certificate as certificate

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _report(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


@pytest.fixture(scope="module")
def synthetic_contact():
    """Perturb three independent coordinates away from the reserved nominal."""
    graph = nx.Graph()
    alpha = 2 * pi / 5
    epsilon = 2**-18
    for i in range(20):
        phase = (i % 10 // 2) * alpha
        if i in (2, 3):
            phase = (1 - 2**-10) * alpha
        elif i in (8, 9):
            phase = (4 + 2**-10) * alpha
        graph.add_node(
            i,
            EPI=5 / 16 + (2**-10 if i < 2 else 0) + (epsilon / 4 if i == 4 else 0),
            theta=phase + (epsilon / 4 if i == 13 else 0),
            nu_f=128.0 if i == 2 else 0.5 if i < 2 else 1.0,
        )
    old, added = certificate._support()
    graph.add_edges_from((i, j, {"weight": 1.0}) for i, j in sorted(old))
    omega = 1 - 2**-10 / (4 * (2 * (5 / 16) + 2**-10 + 1e-9))
    graph.add_edges_from(
        (i, j, {"weight": 1.0 if (i, j) == (0, 1) else omega}) for i, j in sorted(added)
    )
    graph.edges[0, 10]["weight"] -= epsilon / 4
    return _report(graph)


def test_direct_source_admission_rebuilds_storage_pressure_and_degrees(
    synthetic_contact,
):
    admitted = certificate.admit_source(synthetic_contact)
    poisoned = replace(
        synthetic_contact,
        storage=I(10**9),
        pressure=(I(-(10**9)),) * 20,
        degrees=(1,) * 20,
        form_gradient=(Q(10**9),) * 20,
    )
    rebuilt = certificate.admit_source(poisoned)
    assert rebuilt == admitted
    assert admitted.radius_squared.hi < Q(1, 256)
    assert admitted.storage_excess.hi < Q(1, 8192)
    assert admitted.max_form_deviation == Q(1, 2**20)
    assert admitted.max_weight_deviation > 0
    assert admitted.parent_pressure.lo > Q(1, 2**14)


@pytest.mark.parametrize(
    "change", ("capacity", "weight", "phase_lift", "form", "rational")
)
def test_direct_source_rejects_primitive_departures(synthetic_contact, change):
    source = synthetic_contact
    if change == "capacity":
        values = list(source.capacity)
        values[2] = Q(256)
        source = replace(source, capacity=tuple(values))
    elif change == "weight":
        values = list(source.conductances)
        values[source.edges.index((0, 10))] = Q(3, 4)
        source = replace(source, conductances=tuple(values))
    elif change == "phase_lift":
        values = list(source.phase)
        values[2] = Q(float(values[2]) + 2 * pi)
        source = replace(source, phase=tuple(values))
    else:
        values = list(source.epi)
        values[4] = (
            Q(5, 16) + Q(1, 2**100) if change == "rational" else Q(5, 16) + Q(1, 2**18)
        )
        source = replace(source, epi=tuple(values))
    with pytest.raises(ValueError):
        certificate.admit_source(source)


def test_identity_field_retains_weighted_rows_and_nonuniform_capacity():
    graph = nx.Graph()
    for i, (form, phase, capacity) in enumerate(
        ((0.3, -0.2, 128.0), (-0.125, 0.1, 0.5), (0.75, 0.7, 2.0))
    ):
        graph.add_node(i, EPI=form, theta=phase, nu_f=capacity)
    graph.add_edge(0, 1, weight=0.75)
    graph.add_edge(1, 2, weight=0.0)
    graph.add_edge(0, 2, weight=0.875)
    source, field = certificate.prepare_field(_report(graph))
    state = source.epi + source.phase
    bounds = field(tuple(I(value) for value in state))
    with mpmath.workdps(100):

        def mp(value):
            return mpmath.mpf(value.numerator) / value.denominator

        x, theta, nu = (
            tuple(map(mp, source.epi)),
            tuple(map(mp, source.phase)),
            tuple(map(mp, source.capacity)),
        )
        expected_form, expected_phase = [], []
        for i in range(3):
            neighbors = tuple(graph[i])
            weighted_gradient = sum(
                mp(Q(graph.edges[i, j]["weight"])) * (x[i] - x[j]) for j in neighbors
            )
            sine_current = sum(mpmath.sin(theta[j] - theta[i]) for j in neighbors)
            mobility = nu[i] / (mpmath.pi * len(neighbors))
            expected_form.append(mobility * sine_current)
            expected_phase.append(mobility * weighted_gradient)
        expected = tuple(expected_form + expected_phase)
        for bound, value in zip(bounds, expected, strict=True):
            assert mp(bound.lo) <= value <= mp(bound.hi)
        # Independently differentiate the actual field along its own velocity.
        jets = tuple(
            Jet((I(value), rate)) for value, rate in zip(state, bounds, strict=True)
        )
        acceleration = field(jets)
        for i in range(3):
            expected_acceleration = (
                nu[i]
                / (2 * mpmath.pi)
                * sum(
                    mpmath.cos(theta[j] - theta[i])
                    * (expected_phase[j] - expected_phase[i])
                    for j in graph[i]
                )
            )
            bound = acceleration[i].coeffs[1]
            assert mp(bound.lo) <= expected_acceleration <= mp(bound.hi)


def test_history_rejects_unassociated_initial_record_before_enclosure(
    synthetic_contact, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid history must reject before a continuous enclosure")

    monkeypatch.setattr(certificate, "validated_box_taylor_step", forbidden)
    initial = synthetic_contact.epi + synthetic_contact.phase
    changed = list(initial)
    changed[2] += Q(1, 2**20)
    with pytest.raises(ValueError, match="initial recorded state"):
        certificate.certify_history(
            synthetic_contact, certificate.TIMES, (tuple(changed), initial, initial)
        )


@pytest.mark.parametrize("bad", (True, float("nan"), Q(7) + Q(1, 2**100)))
def test_history_clock_rejects_invalid_or_unrepresented_values(synthetic_contact, bad):
    initial = synthetic_contact.epi + synthetic_contact.phase
    with pytest.raises((TypeError, ValueError)):
        certificate.certify_history(
            synthetic_contact, (bad,) + certificate.TIMES[1:], (initial,) * 3
        )


def test_independent_whole_solution_derivative_budget():
    # Sup-norm multilinear chain rule bounds, independent of any endpoint.
    speed, jacobian = Q(4), Q(256, 3)
    second, third, fourth = Q(512, 3), Q(1024, 3), Q(2048, 3)
    a = jacobian * speed
    j = second * speed**2 + jacobian * a
    d4 = third * speed**3 + 3 * second * speed * a + jacobian * j
    d5 = (
        fourth * speed**4
        + 6 * third * speed**2 * a
        + 3 * second * a**2
        + 4 * second * speed * j
        + jacobian * d4
    )
    assert d5 == Q(36571709440, 81)
    assert d5 * Q(1, 2**19) ** 5 / 120 < Q(1, 2**73)
