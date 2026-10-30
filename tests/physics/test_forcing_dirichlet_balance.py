"""Exact heterogeneous-capacity source work from detached pressure captures."""

import math
from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.physics import forcing_realization as realization
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    observe_forcing_dirichlet_balance,
)

F = Fraction


def _pair(weights, phases=(0.0, 0.0)):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    for node, epi, capacity, pressure, phase in zip(
        graph,
        (0.75, -0.25),
        (0.5, 2.0),
        (0.5, -0.25),
        phases,
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi, nu_f=capacity, delta_nfr=pressure, theta=phase
        )
    graph.graph["DNFR_WEIGHTS"] = weights
    return capture_non_epi_forcing(graph)


@pytest.fixture(scope="module")
def pure_epi():
    return _pair({"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0})


@pytest.fixture(scope="module")
def combined_channels():
    return _pair(
        {"phase": 1.0, "epi": 1.0, "vf": 1.0, "topo": 1.0},
        (0.0, math.pi / 2),
    )


def test_pure_epi_uses_heterogeneous_mobility_and_distinguishes_stored_pressure(
    pure_epi,
):
    balance = observe_forcing_dirichlet_balance(pure_epi)
    # E=(x0-x1)^2 for edge weight two, so grad(E)=(2,-2).
    # Fresh velocity=(-1/2,2), while stored velocity=(1/4,-1/2).
    assert balance.source.dirichlet_energy == 1
    assert balance.source.dirichlet_gradient == (2, -2)
    assert balance.diffusion_rate == -5
    assert balance.channel_rates == (("phase", 0), ("vf", 0), ("topo", 0))
    assert balance.source_rate == 0
    assert balance.modeled_rate == balance.fresh_rate == -5
    assert balance.kernel_defect_rate == 0
    assert balance.stored_residual_rate == F(13, 2)
    assert balance.stored_rate == F(3, 2)
    assert balance.identity_residual == 0


def test_phase_and_capacity_work_use_capacity_once_and_preserve_channel_signs(
    combined_channels,
):
    balance = observe_forcing_dirichlet_balance(combined_channels)
    # Unit phase gradient=(1/2,-1/2), capacity gradient=(3/2,-3/2).
    # Multiplication by weights 1/4 gives F=(1/2,-1/2).
    # The work covector grad(E)*nu=(1,-4) is not the unit-capacity (2,-2).
    assert balance.diffusion_rate == F(-5, 4)
    assert balance.channel_rates == (
        ("phase", F(5, 8)),
        ("vf", F(15, 8)),
        ("topo", 0),
    )
    assert balance.source_rate == F(5, 2)
    assert balance.modeled_rate == balance.fresh_rate == F(5, 4)
    assert balance.kernel_defect_rate == 0
    assert balance.stored_residual_rate == F(1, 4)
    assert balance.stored_rate == F(3, 2)
    assert balance.identity_residual == 0


def test_detached_observer_rebuilds_all_pressure_and_transport_caches(pure_epi):
    expected = observe_forcing_dirichlet_balance(pure_epi)
    forged = replace(
        pure_epi,
        snapshot=replace(
            pure_epi.snapshot,
            epi_gradient=(F(999),) * 2,
            capacity_gradient=(F(999),) * 2,
            topology_gradient=(F(999),) * 2,
            dirichlet_gradient=(F(999),) * 2,
            rate=(F(999),) * 2,
            dirichlet_energy=F(999),
            energy_rate=F(999),
        ),
        kernel_pressure_defect=(F(999),) * 2,
        stored_pressure_residual=(F(999),) * 2,
    )
    assert observe_forcing_dirichlet_balance(forged) == expected

    # A detached full-pressure primitive remains distinct from its cache.
    # This is an arithmetic input control, not a new live kernel observation.
    altered = replace(forged, full_kernel_pressure=(F(-7, 8), F(1)))
    balance = observe_forcing_dirichlet_balance(altered)
    assert balance.modeled_rate == -5
    assert balance.kernel_defect_rate == F(1, 8)
    assert balance.fresh_rate == F(-39, 8)
    assert balance.stored_residual_rate == F(51, 8)
    assert balance.stored_rate == F(3, 2)
    assert balance.identity_residual == 0


def test_detached_rate_observation_does_not_recapture_graph_or_run_kernel(
    combined_channels, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("detached rate arithmetic cannot recapture or execute")

    monkeypatch.setattr(realization, "capture_non_epi_forcing", forbidden)
    monkeypatch.setattr(realization, "observe_support_transport", forbidden)
    monkeypatch.setattr(
        realization.fused_dnfr, "compute_fused_gradients_symmetric", forbidden
    )
    balance = observe_forcing_dirichlet_balance(combined_channels)
    assert balance.source_rate == F(5, 2)
    assert balance.stored_rate == F(3, 2)


@pytest.mark.parametrize("pressure", [(), (F(0),), (F(0),) * 3])
def test_full_pressure_must_match_the_complete_node_space(pure_epi, pressure):
    with pytest.raises(ValueError):
        observe_forcing_dirichlet_balance(
            replace(pure_epi, full_kernel_pressure=pressure)
        )


@pytest.mark.parametrize(
    "pressure,exception",
    [
        ({0: 0, 1: 0}, TypeError),
        ((True, 0), TypeError),
        ((float("nan"), 0), ValueError),
        ((float("inf"), 0), ValueError),
    ],
)
def test_full_pressure_rejects_unordered_or_nonfinite_payloads(
    pure_epi, pressure, exception
):
    with pytest.raises(exception):
        observe_forcing_dirichlet_balance(
            replace(pure_epi, full_kernel_pressure=pressure)
        )


def test_rate_observer_requires_an_explicit_forcing_observation():
    with pytest.raises(TypeError):
        observe_forcing_dirichlet_balance({"forcing": (0, 0)})
