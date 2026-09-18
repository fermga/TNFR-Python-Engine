"""Conditional parameter covariance and local EPI-pressure characterizations.

These are exact detached model controls, not selected laws or trajectories.
The pressure owner materializes binary64 phase values; algebra below treats
those retained coefficients as exact, without a transcendental claim.
"""

from fractions import Fraction as Q

import networkx as nx

from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.support_transport import _from_data, observe_support_transport


_X = (Q(1, 4), Q(1), Q(3, 4))
_NU = (Q(1, 2), Q(1), Q(3, 2))
_WEIGHTS = {"phase": Q(1, 4), "epi": Q(1, 4), "vf": Q(1, 4), "topo": Q(1, 4)}


def _capture(*, epi=_X, capacity=_NU, weights=None, normalize=False):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["weight"] = 1.0
    graph.edges[1, 2]["weight"] = 2.0
    for node, x, nu, theta in zip(graph, epi, capacity, (0.0, 0.25, -0.5), strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=theta)
    supplied = {key: float(value) for key, value in (weights or _WEIGHTS).items()}
    graph.graph["DNFR_WEIGHTS"] = supplied
    if not normalize:
        # The existing owner reads already materialized coefficients verbatim.
        # This deliberately declares an unrenormalized model; it does not claim
        # that normal public configuration leaves these coefficients unchanged.
        graph.graph["_dnfr_weights"] = supplied.copy()
    return capture_non_epi_forcing(graph)


def _pressure(observation):
    decompose_non_epi_forcing(observation)  # validate against the shared owner
    return tuple(
        observation.epi_weight * gradient + forcing
        for gradient, forcing in zip(
            observation.snapshot.epi_gradient, observation.forcing, strict=True,
        )
    )


def _rate(observation):
    return tuple(
        nu * pressure
        for nu, pressure in zip(observation.snapshot.capacity, _pressure(observation), strict=True)
    )


def test_joint_form_and_time_change_requires_transformed_channel_coefficients():
    original = _capture()
    # y=a*x+b; tau=c*t; nu_tau=nu/c. All supplied values are exact dyadics.
    a, b, c = Q(7, 4), Q(1, 2), Q(2)
    weights = {
        "epi": _WEIGHTS["epi"],
        "phase": a * _WEIGHTS["phase"],
        "vf": a * c * _WEIGHTS["vf"],
        "topo": a * _WEIGHTS["topo"],
    }
    transformed = _capture(
        epi=tuple(a*x+b for x in _X),
        capacity=tuple(nu/c for nu in _NU),
        weights=weights,
    )
    assert dict(transformed.normalized_weights) == weights
    assert sum(weights.values()) == 2  # retained coefficients, no normalization
    assert transformed.phase_gradient == original.phase_gradient
    assert transformed.snapshot.topology_gradient == original.snapshot.topology_gradient
    assert _pressure(transformed) == tuple(a*p for p in _pressure(original))
    assert _rate(transformed) == tuple(a*r/c for r in _rate(original))


def test_fixed_capacity_channel_coefficient_breaks_time_unit_covariance():
    original = _capture()
    c = Q(2)
    incorrectly_fixed = _capture(capacity=tuple(nu/c for nu in _NU))
    difference = tuple(
        after-before for before, after in zip(_pressure(original), _pressure(incorrectly_fixed))
    )
    assert difference == tuple(
        _WEIGHTS["vf"] * (1/c-1) * gradient
        for gradient in original.snapshot.capacity_gradient
    )
    assert any(difference)
    assert _rate(incorrectly_fixed) != tuple(r/c for r in _rate(original))


def test_normalizing_transformed_weights_changes_the_required_pressure_scale():
    original = _capture()
    a, b, c = Q(7, 4), Q(1, 2), Q(2)
    weights = {"epi": Q(1, 4), "phase": Q(7, 16), "vf": Q(7, 8), "topo": Q(7, 16)}
    transformed = _capture(
        epi=tuple(a*x+b for x in _X),
        capacity=tuple(nu/c for nu in _NU),
        weights=weights,
        normalize=True,
    )
    normalization = sum(weights.values())
    assert normalization == 2
    assert dict(transformed.normalized_weights) == {
        key: value/normalization for key, value in weights.items()
    }
    assert _pressure(transformed) == tuple(a*p/normalization for p in _pressure(original))
    assert _rate(transformed) == tuple(a*r/(c*normalization) for r in _rate(original))
    assert _rate(transformed) != tuple(a*r/c for r in _rate(original))


def test_pure_epi_clock_change_needs_no_capacity_source_compensation():
    weights = {"epi": 1, "phase": 0, "vf": 0, "topo": 0}
    original = _capture(weights=weights)
    rescaled = _capture(capacity=tuple(nu/2 for nu in _NU), weights=weights)
    assert _pressure(rescaled) == _pressure(original)
    assert _rate(rescaled) == tuple(r/2 for r in _rate(original))


def test_shared_epi_owner_has_the_local_linear_difference_representation():
    source = _capture().snapshot
    columns = []
    for column in range(3):
        basis = tuple(Q(index == column) for index in range(3))
        basis_source = _from_data(
            source.nodes, source.conductance, source.support_neighbors,
            basis, source.capacity, source.stored_pressure,
        )
        columns.append(basis_source.epi_gradient)
    matrix = tuple(tuple(columns[j][i] for j in range(3)) for i in range(3))
    assert matrix == ((-1, 1, 0), (Q(1, 3), -1, Q(2, 3)), (0, 1, -1))
    assert all(sum(row) == 0 for row in matrix)
    assert all(matrix[i][j] >= 0 for i in range(3) for j in range(3) if i != j)
    represented = tuple(
        sum(matrix[i][j] * (source.epi[j]-source.epi[i]) for j in range(3) if j != i)
        for i in range(3)
    )
    assert represented == source.epi_gradient
    # This finite owner control does not prove the characterization for all
    # laws; locality, linearity and a maximum principle are separate premises.


def test_symmetric_support_does_not_imply_detailed_balance():
    # Every directed triangle edge exists in reverse and has positive weight.
    # The rows define a local, shift-invariant, maximum-principle pressure.
    matrix = (
        (-1, Q(2, 3), Q(1, 3)),
        (Q(1, 3), -1, Q(2, 3)),
        (Q(2, 3), Q(1, 3), -1),
    )
    assert all(sum(row) == 0 for row in matrix)
    assert all(matrix[i][j] > 0 for i in range(3) for j in range(3) if i != j)
    clockwise = matrix[0][1] * matrix[1][2] * matrix[2][0]
    reverse = matrix[0][2] * matrix[2][1] * matrix[1][0]
    assert clockwise == Q(8, 27)
    assert reverse == Q(1, 27)
    assert clockwise != reverse
    # Multiplying h_i*a_ij=h_j*a_ji around the cycle would cancel all
    # positive h_i and require equality, so no reversible metric exists.


def test_edge_transitive_cycle_uniform_weights_reduce_to_neighbor_averages():
    fields = []
    for weight in (Q(1), Q(7, 4)):
        graph = nx.cycle_graph(4)
        for left, right in graph.edges:
            graph.edges[left, right]["weight"] = float(weight)
        for node, epi in zip(graph, (0, 1, 3, -1), strict=True):
            graph.nodes[node].update(EPI=epi, nu_f=1, theta=0)
        snapshot = observe_support_transport(graph)
        fields.append(snapshot.epi_gradient)
        assert all(value == weight for _, _, value in snapshot.conductance)
    assert fields[0] == fields[1] == (0, Q(1, 2), -3, Q(5, 2))
    # Cycle rotations carry each undirected edge to every other edge; a
    # support-only reciprocal assignment invariant under those rotations
    # therefore has exactly the common-weight freedom tested here.
