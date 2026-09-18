"""Exact regional cut, source and centered-variance balances."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.physics.support_transport import (
    _from_data, observe_regional_support_balance, observe_support_transport,
)


F = Fraction


def _path(*, epi=(0, 1, 5), capacity=(1, 2, 1), pressure=(0, 0, 0)):
    return _from_data(("a", "b", "outside"),
                      ((0, 1, 1), (1, 0, 1), (1, 2, 1), (2, 1, 1)),
                      ((1,), (0, 2), (1,)), epi, capacity, pressure)


def _observe(source=None, region=("a", "b"), *, e=1, forcing=(0, 0, 0)):
    return observe_regional_support_balance(_path() if source is None else source, region,
                                            epi_weight=e, forcing=forcing)


def test_three_node_hand_balance_keeps_forcing_and_stored_defect_separate():
    result = _observe(forcing=(1, -1, 2))
    assert result.source.nodes == ("a", "b", "outside")
    assert result.region == ("a", "b") and result.environment == ("outside",)
    assert result.region_indices == (0, 1) and result.cut_edges == ((1, 2, F(1)),)
    assert result.strengths == (1, 2, 1) and result.metric_weights == (1, 1, 1)
    assert result.regional_weight == 2 and result.weighted_total == 1
    assert result.mean == F(1, 2) and result.centered_epi == (F(-1, 2), F(1, 2))
    assert result.variance == F(1, 4)
    assert result.outward_cut_current == -4 and result.mass_boundary_rate == 4
    assert result.mass_forcing_rate == -1 and result.model_mass_rate == 3
    assert result.model_pressure == (2, F(1, 2), -2)
    assert result.stored_pressure_defect == (-2, F(-1, 2), 2)
    assert result.mass_defect_rate == -3 and result.stored_mass_rate == 0
    assert result.internal_dissipation == 1 and result.variance_boundary_rate == 2
    assert result.variance_forcing_rate == F(-3, 2)
    assert result.model_variance_rate == F(-1, 2)
    assert result.variance_defect_rate == F(1, 2) and result.stored_variance_rate == 0
    assert (result.model_mass_identity_residual, result.mass_identity_residual,
            result.model_variance_identity_residual, result.variance_identity_residual) == (0, 0, 0, 0)


def test_boundary_injection_can_grow_regional_variance_despite_internal_dissipation():
    source = _path(pressure=(1, F(3, 2), -4))
    result = _observe(source)
    assert result.stored_pressure_defect == (0, 0, 0)
    assert result.internal_dissipation == 1
    assert result.variance_boundary_rate == 2
    assert result.model_variance_rate == result.stored_variance_rate == 1 > 0
    # Independent two-coordinate calculation: E=(x_b-x_a)^2/4.
    rate = source.rate
    assert result.stored_variance_rate == (source.epi[1]-source.epi[0])*(rate[1]-rate[0])/2


def test_full_graph_normalization_and_environment_are_not_replaced_by_induced_subgraph():
    source = _path()
    observed = _observe(source)
    assert observed.strengths[1] == 2 and observed.metric_weights[1] == 1
    induced_mean = (F(1, 2)*source.epi[1])/(1+F(1, 2))
    assert observed.mean != induced_mean
    altered = _observe(replace(source, epi=(F(0), F(1), F(9))))
    assert altered.mean == observed.mean and altered.variance == observed.variance
    assert altered.model_mass_rate != observed.model_mass_rate
    assert altered.model_variance_rate != observed.model_variance_rate
    assert altered.source.nodes == observed.source.nodes


def test_complementary_cut_totals_cancel_and_recover_the_full_source_balance():
    source = _path(pressure=(1, -2, 3))
    left = _observe(source, forcing=(1, -1, 2))
    right = _observe(source, region=("outside",), forcing=(1, -1, 2))
    assert left.outward_cut_current == -right.outward_cut_current
    assert left.mass_boundary_rate+right.mass_boundary_rate == 0
    assert left.model_mass_rate+right.model_mass_rate == 1
    assert left.stored_mass_rate+right.stored_mass_rate == sum(
        h*r for h, r in zip(left.metric_weights, source.rate, strict=True))
    assert right.variance == right.model_variance_rate == right.stored_variance_rate == 0
    assert right.internal_dissipation == right.variance_boundary_rate == 0


def test_region_order_is_preserved_without_affecting_scalar_balances():
    forward, reverse = _observe(), _observe(region=("b", "a"))
    assert reverse.region_indices == (1, 0)
    assert reverse.centered_epi == tuple(reversed(forward.centered_epi))
    for name in ("mean", "variance", "model_mass_rate", "stored_mass_rate",
                 "internal_dissipation", "variance_boundary_rate", "model_variance_rate"):
        assert getattr(forward, name) == getattr(reverse, name)


def test_self_loops_enter_full_metric_but_have_zero_cut_or_internal_flux():
    source = _path()
    looped = replace(source, conductance=((0, 0, F(3)),)+source.conductance,
                     support_neighbors=((0, 1), (0, 2), (1,)))
    result = _observe(looped)
    assert result.strengths == (4, 2, 1) and result.metric_weights == (4, 1, 1)
    assert result.mean == F(1, 5)
    assert result.internal_dissipation == 1 and result.outward_cut_current == -4
    assert result.variance_boundary_rate == F(16, 5)
    assert result.model_variance_identity_residual == 0


def test_disconnected_full_support_and_disconnected_region_need_no_profile_solve():
    source = _from_data((0, 1, 2, 3), ((0, 1, 1), (1, 0, 1), (2, 3, 2), (3, 2, 2)),
                        ((1,), (0,), (3,), (2,)), (0, 1, 3, 4), (1, 1, 2, 2), (0, 0, 0, 0))
    result = _observe(source, region=(0, 2), forcing=(0, 0, 0, 0))
    assert result.internal_dissipation == 0
    assert result.environment == (1, 3)
    assert result.outward_cut_current == -3 and result.model_mass_rate == 3


def test_cached_derived_fields_are_rebuilt_and_invalid_primitives_remain_rejected():
    source = _path()
    forged = replace(source, epi_gradient=(F(99),)*3, rate=(F(99),)*3,
                     dirichlet_gradient=(F(99),)*3, dirichlet_energy=F(99), energy_rate=F(99))
    assert _observe(forged) == _observe(source)
    bad = replace(forged, capacity=(F(1), F(0), F(1)))
    with pytest.raises(ValueError, match="positive"):
        _observe(bad)
    with pytest.raises(TypeError, match="SupportTransportSnapshot"):
        _observe({"nodes": source.nodes})


@pytest.mark.parametrize("region", ((), ("a", "b", "outside"), ("a",)*2, ("missing",),
                                    ([],), ("a", "b", "outside", "extra")))
def test_invalid_region_membership_is_rejected(region):
    with pytest.raises(ValueError):
        _observe(region=region)


@pytest.mark.parametrize("region", ("a", b"a", {"a"}, {"a": 1}, None))
def test_unordered_or_scalar_regions_are_rejected(region):
    with pytest.raises(TypeError):
        _observe(region=region)


def test_region_materialization_is_bounded_by_source_size():
    consumed = []

    def endless():
        while True:
            consumed.append("a")
            yield "a"

    with pytest.raises(ValueError, match="proper"):
        _observe(region=endless())
    assert len(consumed) == 3


@pytest.mark.parametrize("weight", (0, -1, float("nan"), float("inf"), True))
def test_invalid_epi_weight_is_rejected(weight):
    with pytest.raises((TypeError, ValueError)):
        _observe(e=weight)


@pytest.mark.parametrize("forcing", ((0, 0), (0, float("nan"), 0), (0, float("inf"), 0),
                                     (0, True, 0), {0, 1, 2}))
def test_forcing_must_be_explicit_finite_and_full_graph_aligned(forcing):
    with pytest.raises((TypeError, ValueError)):
        _observe(forcing=forcing)


def test_zero_strength_anywhere_is_outside_the_declared_positive_metric_domain():
    source = _path()
    isolated = replace(source, conductance=source.conductance[:2], support_neighbors=((1,), (0,), ()))
    with pytest.raises(ValueError, match="positive full strengths"):
        _observe(isolated)


def test_input_state_and_graph_are_unchanged_and_output_is_frozen():
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update({ALIAS_EPI[0]: float(node), ALIAS_VF[0]: 1.0, ALIAS_DNFR[0]: .1})
    graph.graph["marker"] = {"values": [1, 2]}
    saved = deepcopy((graph.graph, dict(graph.nodes(data=True)), dict(graph.edges)))
    source = observe_support_transport(graph)
    region, forcing = [0, 1], [0, 0, 0]
    result = _observe(source, region, forcing=forcing)
    assert saved == (graph.graph, dict(graph.nodes(data=True)), dict(graph.edges))
    region.reverse()
    forcing[0] = 9
    assert result.region == (0, 1) and result.forcing == (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        result.mean = F(99)
    assert "No causal execution" in result.scope


def test_rational_finite_difference_of_regional_variance_matches_the_declared_rate():
    source = _path(epi=(F(-2, 3), F(7, 5), F(11, 7)), capacity=(2, 3, 5),
                   pressure=(F(1, 3), F(-2, 5), F(4, 7)))
    result = _observe(source, e=F(3, 7), forcing=(F(1, 5), F(-2, 3), F(4, 9)))
    # A centered difference of this quadratic along the supplied stored rate
    # is exact at any rational h; no numerical solver or small-step claim.

    def energy(step):
        x = tuple(value+step*rate for value, rate in zip(source.epi, source.rate, strict=True))
        h = result.metric_weights
        mean = (h[0]*x[0]+h[1]*x[1])/(h[0]+h[1])
        return (h[0]*(x[0]-mean)**2+h[1]*(x[1]-mean)**2)/2

    step = F(7, 11)
    assert (energy(step)-energy(-step))/(2*step) == result.stored_variance_rate
