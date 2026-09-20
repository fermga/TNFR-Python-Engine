"""Fixed-strength mean accounting, distinct from capacity-dependent H means."""

from dataclasses import replace
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics import forcing_realization as realization
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing,
    observe_forcing_mean_balance,
)


def _capture(capacity=(1, 2), *, weight=3):
    graph = nx.Graph()
    graph.add_edge("left", 9, weight=float(weight))
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    for node, epi, nu in zip(graph, (0, 1), capacity, strict=True):
        graph.nodes[node].update(
            EPI=float(epi), nu_f=float(nu), theta=0.0, delta_nfr=0.0
        )
    default_compute_delta_nfr(graph)
    return capture_non_epi_forcing(graph)


@pytest.mark.parametrize(
    "capacity,expected",
    [((1, 2), -F(1, 2)), ((2, 1), F(1, 2)), ((1, 1), F(0)), ((0, 2), -F(1))],
)
def test_heterogeneous_diffusion_can_move_the_fixed_strength_mean(capacity, expected):
    report = observe_forcing_mean_balance(_capture(capacity))
    assert report.strengths == (3, 3)
    assert report.total_strength == 6
    assert report.mean == F(1, 2)
    assert report.diffusion_rate == expected
    assert report.source_rate == 0
    assert report.modeled_rate == report.fresh_rate == report.stored_rate == expected
    assert report.kernel_defect_rate == report.stored_residual_rate == 0
    assert report.identity_residual == 0
    if min(capacity) > 0:
        # The different fixed-capacity H=d/nu total is conserved by pure
        # diffusion. Its zero derivative cannot be assigned to the D mean.
        assert (
            sum(
                d * rate / nu
                for d, rate, nu in zip(
                    report.strengths,
                    report.source.rate,
                    report.source.capacity,
                    strict=True,
                )
            )
            == 0
        )


def test_mean_separates_signed_kernel_and_stored_defects_without_recapture(monkeypatch):
    captured = _capture()
    # Declared detached arithmetic inputs, not claimed production-kernel
    # defects: choose independent signed perturbations with exact answers.
    full = (F(9, 8), -F(5, 4))
    stored = (F(13, 8), -F(5, 4))
    changed = replace(
        captured,
        full_kernel_pressure=full,
        snapshot=replace(captured.snapshot, stored_pressure=stored, rate=(F(99),) * 2),
        kernel_pressure_defect=(F(99),) * 2,
        stored_pressure_residual=(F(99),) * 2,
    )

    def forbidden(**kwargs):
        raise AssertionError("detached mean accounting must not call a pressure kernel")

    monkeypatch.setattr(
        realization.fused_dnfr, "compute_fused_gradients_symmetric", forbidden
    )
    report = observe_forcing_mean_balance(changed)
    assert report.modeled_rate == -F(1, 2)
    assert report.kernel_defect_rate == -F(3, 16)
    assert report.fresh_rate == -F(11, 16)
    assert report.stored_residual_rate == F(1, 4)
    assert report.stored_rate == -F(7, 16)
    assert report.identity_residual == 0


def test_capacity_only_change_does_not_reweight_the_fixed_strength_mean():
    captured = _capture()
    following = replace(
        captured, snapshot=replace(captured.snapshot, capacity=(F(5), F(1)))
    )
    before, after = map(observe_forcing_mean_balance, (captured, following))
    assert after.source.epi == before.source.epi
    assert after.mean == before.mean == F(1, 2)
    assert before.diffusion_rate == -F(1, 2)
    assert after.diffusion_rate == 2
    h_means = []
    for report in (before, after):
        h = tuple(
            d / nu
            for d, nu in zip(report.strengths, report.source.capacity, strict=True)
        )
        h_means.append(
            sum(w * x for w, x in zip(h, report.source.epi, strict=True)) / sum(h)
        )
    assert h_means == [F(1, 3), F(5, 6)]


def test_zero_total_transport_strength_is_unavailable():
    captured = _capture(weight=0)
    with pytest.raises(ValueError, match="strength"):
        observe_forcing_mean_balance(captured)


def test_public_mean_owner_admits_disconnected_transport_and_a_zero_strength_isolate():
    from tnfr.physics import ForcingMeanBalance
    from tnfr.physics import observe_forcing_mean_balance as public_observe

    assert public_observe is observe_forcing_mean_balance
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from(((0, 1, 3.0), (2, 3, 2.0)))
    graph.graph["DNFR_WEIGHTS"] = {"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0}
    for node, epi, nu in zip(graph, (0, 1, 2, 4, 1000), (1, 2, 1, 1, 0), strict=True):
        graph.nodes[node].update(
            EPI=float(epi), nu_f=float(nu), theta=0.0, delta_nfr=0.0
        )
    default_compute_delta_nfr(graph)
    report = public_observe(capture_non_epi_forcing(graph))
    assert isinstance(report, ForcingMeanBalance)
    assert report.strengths == (3, 3, 2, 2, 0)
    assert report.total_strength == 10 and report.mean == F(3, 2)
    assert report.diffusion_rate == report.stored_rate == -F(3, 10)
    assert report.source.rate[-1] == 0
    assert report.identity_residual == 0


def test_invalid_detached_type_and_pressure_dimension_are_rejected():
    with pytest.raises(TypeError):
        observe_forcing_mean_balance({"mean": 0})
    with pytest.raises(ValueError, match="pressure"):
        observe_forcing_mean_balance(replace(_capture(), full_kernel_pressure=(F(0),)))
