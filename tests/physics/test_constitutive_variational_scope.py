"""Exact conditional variational constraints from canonical pressure owners."""

from fractions import Fraction as Q

import networkx as nx

from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    decompose_non_epi_forcing,
)
from tnfr.physics.phase_response import derive_phase_response
from tnfr.physics.support_transport import _from_data

_EPI = (Q(1, 4), Q(1), Q(3, 4))
_CAPACITY = (Q(1, 2), Q(1), Q(3, 2))


def _source(epi=_EPI, capacity=_CAPACITY, edge_weights=(1, 2)):
    graph = nx.path_graph(3)
    for (left, right), weight in zip(graph.edges, edge_weights, strict=True):
        graph.edges[left, right]["weight"] = float(weight)
    for node, x, nu in zip(graph, epi, capacity, strict=True):
        graph.nodes[node].update(EPI=float(x), nu_f=float(nu), theta=0.0)
    graph.graph["DNFR_WEIGHTS"] = {
        "phase": 0.0,
        "epi": 0.5,
        "vf": 0.25,
        "topo": 0.25,
    }
    observation = capture_non_epi_forcing(graph)
    reference = derive_forced_support_balance(
        observation.snapshot,
        epi_weight=observation.epi_weight,
        forcing=observation.forcing,
    )
    return observation, reference


def _replace_epi(snapshot, epi):
    return _from_data(
        snapshot.nodes,
        snapshot.conductance,
        snapshot.support_neighbors,
        epi,
        snapshot.capacity,
        snapshot.stored_pressure,
    )


def _forced_potential(reference, snapshot):
    # Conditional Psi=0 slice, used only to check the forced x dependence.
    # Any x-independent Psi cancels in all mixed differences below.
    return reference.epi_weight * snapshot.dirichlet_energy - sum(
        d * x * forcing
        for d, x, forcing in zip(
            reference.strengths,
            snapshot.epi,
            reference.forcing,
            strict=True,
        )
    )


def test_forced_potential_realizes_the_weighted_multichannel_epi_equation():
    observation, reference = _source()
    snapshot = observation.snapshot
    channels = dict(decompose_non_epi_forcing(observation))
    assert reference.strengths == (1, 3, 2)
    assert channels == {
        "phase": (0, 0, 0),
        "vf": (Q(1, 8), 0, Q(-1, 8)),
        "topo": (Q(1, 4), Q(-1, 4), Q(1, 4)),
    }
    gradient = tuple(
        reference.epi_weight * bx - d * forcing
        for bx, d, forcing in zip(
            snapshot.dirichlet_gradient,
            reference.strengths,
            reference.forcing,
            strict=True,
        )
    )
    rate_from_gradient = tuple(
        -nu * derivative / d
        for nu, derivative, d in zip(
            snapshot.capacity,
            gradient,
            reference.strengths,
            strict=True,
        )
    )
    rate_from_pressure = tuple(
        nu * (reference.epi_weight * epi_gradient + forcing)
        for nu, epi_gradient, forcing in zip(
            snapshot.capacity,
            snapshot.epi_gradient,
            reference.forcing,
            strict=True,
        )
    )
    assert rate_from_gradient == rate_from_pressure

    direction = (1, -2, 3)
    positive = _replace_epi(
        snapshot, tuple(x + h for x, h in zip(snapshot.epi, direction))
    )
    negative = _replace_epi(
        snapshot, tuple(x - h for x, h in zip(snapshot.epi, direction))
    )
    # A centered finite difference is exact for this quadratic polynomial;
    # no numerical derivative, epsilon or trajectory is involved.
    difference = (
        _forced_potential(reference, positive) - _forced_potential(reference, negative)
    ) / 2
    assert difference == sum(g * h for g, h in zip(gradient, direction))


def test_capacity_pressure_forces_nonzero_reciprocal_mixed_partial():
    # Vary x_0 and nu_1 independently. The exact mixed rectangle measures
    # partial_nu1 partial_x0 E = v*d_0*(L_U)_01 = -1/4.
    epi_shifted = (_EPI[0] + 1, *_EPI[1:])
    capacity_shifted = (_CAPACITY[0], _CAPACITY[1] + 1, _CAPACITY[2])
    values = []
    for epi, capacity in (
        (epi_shifted, capacity_shifted),
        (epi_shifted, _CAPACITY),
        (_EPI, capacity_shifted),
        (_EPI, _CAPACITY),
    ):
        observation, reference = _source(epi, capacity)
        values.append(_forced_potential(reference, observation.snapshot))
    assert values[0] - values[1] - values[2] + values[3] == Q(-1, 4)
    # An x-independent capacity force under an x-independent invertible
    # block mobility has zero opposite mixed derivative, so cannot equal it.


def test_incompatible_source_gives_exact_unbounded_common_offset_tilt():
    observation, reference = _source()
    assert reference.forcing == (Q(3, 8), Q(-1, 4), Q(1, 8))
    assert reference.compatibility_residual == Q(-1, 8)
    assert not reference.has_zero_pressure_equilibrium
    baseline = _forced_potential(reference, observation.snapshot)
    for offset in (-2, 1, 4):
        shifted = _replace_epi(
            observation.snapshot,
            tuple(x + offset for x in observation.snapshot.epi),
        )
        assert shifted.dirichlet_energy == observation.snapshot.dirichlet_energy
        assert (
            _forced_potential(reference, shifted) - baseline
            == -offset * reference.compatibility_residual
        )
    # The formula holds for arbitrary real offsets; the unboundedness proof
    # is the nonzero linear coefficient, not extrapolation from these values.


def test_compatible_source_uses_existing_poisson_profile_to_complete_square():
    observation, reference = _source(edge_weights=(1, 1))
    assert reference.compatibility_residual == 0
    assert reference.has_zero_pressure_equilibrium
    snapshot = observation.snapshot
    profile = _replace_epi(snapshot, reference.relative_profile)
    centered = _replace_epi(
        snapshot,
        tuple(x - z for x, z in zip(snapshot.epi, reference.relative_profile)),
    )
    completed = reference.epi_weight * (
        centered.dirichlet_energy - profile.dirichlet_energy
    )
    assert _forced_potential(reference, snapshot) == completed
    assert centered.dirichlet_energy >= 0


def _star_phase_response(cosine):
    gram = tuple(
        tuple(Q(1) if (i == 3) == (j == 3) else cosine for j in range(4))
        for i in range(4)
    )
    return derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=((1, 2, 3), (0,), (0,), (0,)),
        receiver_sources=((0,), (1,), (2,), (3,)),
        phase_factor=1,
    )


def test_strict_u3_star_refutes_constant_diagonal_phase_gradient_metric():
    consensus = _star_phase_response(Q(1))
    changed = _star_phase_response(Q(3, 5))
    assert consensus.mean_response[0] == (0, Q(1, 3), Q(1, 3), Q(1, 3))
    assert changed.mean_response[0] == (0, Q(13, 37), Q(13, 37), Q(11, 37))
    assert changed.mean_resultant_squared == (Q(37, 5), 1, 1, 1)
    assert all(changed.cosine_gram[0][j] > 0 for j in (1, 2, 3))
    for reference in (consensus, changed):
        assert tuple(reference.mean_response[j][0] for j in (1, 2, 3)) == (1, 1, 1)

    # Consensus reciprocity uniquely fixes the positive diagonal metric ray.
    metric = (3, 1, 1, 1)
    assert all(
        metric[i] * consensus.mean_response[i][j]
        == metric[j] * consensus.mean_response[j][i]
        for i in range(4)
        for j in range(4)
    )
    defects = tuple(
        metric[0] * changed.mean_response[0][j]
        - metric[j] * changed.mean_response[j][0]
        for j in (1, 2, 3)
    )
    assert defects == (Q(2, 37), Q(2, 37), Q(-4, 37))
    # The analytic identity (1-c)/(5+4*c)>0 for every c in (0,1)
    # gives the open-neighborhood obstruction; this is its exact owner control.
    assert defects[0] == (1 - Q(3, 5)) / (5 + 4 * Q(3, 5))
