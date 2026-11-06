"""Shared exact cut admission, independent of a positive regional metric."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest

from tnfr.physics.support_transport import (
    _from_data,
    observe_regional_support_balance,
    observe_regional_support_cut,
)

F = Fraction


def _source(*, capacity=(1, 2, 1)):
    return _from_data(
        ("a", "b", "outside"),
        ((0, 1, 2), (1, 0, 2), (1, 2, 3), (2, 1, 3)),
        ((1,), (0, 2), (1,)),
        (F(-1, 2), F(3, 2), 4),
        capacity,
        (0, 0, 0),
    )


def test_cut_uses_full_indices_and_weighted_outward_convention():
    source = _source()
    region = ["b", "a"]
    cut = observe_regional_support_cut(source, iter(region))
    region.reverse()
    assert cut.nodes == source.nodes
    assert cut.region == ("b", "a") and cut.region_indices == (1, 0)
    assert cut.environment == ("outside",)
    assert cut.cut_edges == ((1, 2, F(3)),)
    assert cut.outward_cut_current == 3 * (F(3, 2) - 4)
    complement = observe_regional_support_cut(source, cut.environment)
    assert complement.cut_edges == ((2, 1, F(3)),)
    assert complement.outward_cut_current == -cut.outward_cut_current
    with pytest.raises(FrozenInstanceError):
        cut.outward_cut_current = F(1)


def test_existing_balance_projects_the_same_cut_without_new_metric_semantics():
    source = _source()
    region = ("b", "a")
    balance = observe_regional_support_balance(
        source, region, epi_weight=F(2, 3), forcing=(0, 0, 0)
    )
    cut = observe_regional_support_cut(source, region)
    assert balance.cut == cut
    assert balance.mass_boundary_rate == -F(2, 3) * cut.outward_cut_current
    assert balance.mass_identity_residual == balance.variance_identity_residual == 0
    with pytest.raises(ValueError, match="proper"):
        observe_regional_support_balance(
            source, source.nodes, epi_weight=1, forcing=(0, 0, 0)
        )
    with pytest.raises(ValueError, match="positive"):
        observe_regional_support_balance(
            source, region, epi_weight=0, forcing=(0, 0, 0)
        )


def test_full_support_and_zero_capacity_need_no_division():
    source = _source(capacity=(0, 0, 0))
    regional = observe_regional_support_cut(source, ("a", "b"))
    assert regional.outward_cut_current == F(-15, 2)
    full = observe_regional_support_cut(source, ("outside", "a", "b"))
    assert full.region_indices == (2, 0, 1)
    assert full.environment == full.cut_edges == ()
    assert full.outward_cut_current == 0


def test_isolates_and_zero_conductance_support_do_not_invent_cut_flux():
    source = _from_data(
        ("a", "b", "isolate"),
        ((0, 0, 2),),
        ((0, 1), (0,), ()),
        (1, 9, 100),
        (0, 1, 0),
        (0, 0, 0),
    )
    # a-b is support only, a-a is a loop, and the remaining node is isolated.
    for region in (("a",), ("isolate",), source.nodes):
        cut = observe_regional_support_cut(source, region)
        assert cut.cut_edges == () and cut.outward_cut_current == 0


def test_cut_rebuilds_derived_caches_but_rejects_invalid_primitive_data():
    source = _source()
    forged = replace(
        source,
        epi_gradient=(F(99),) * 3,
        dirichlet_gradient=(F(99),) * 3,
        dirichlet_energy=F(99),
        rate=(F(99),) * 3,
        energy_rate=F(99),
    )
    region = ("b", "a")
    assert observe_regional_support_cut(forged, region) == observe_regional_support_cut(
        source, region
    )
    with pytest.raises(ValueError, match="symmetric"):
        observe_regional_support_cut(
            replace(forged, conductance=forged.conductance[:-1]), region
        )
    with pytest.raises(ValueError, match="nonnegative"):
        observe_regional_support_cut(replace(forged, capacity=(-1, 1, 1)), region)
    with pytest.raises(TypeError, match="SupportTransportSnapshot"):
        observe_regional_support_cut({"nodes": source.nodes}, region)


@pytest.mark.parametrize(
    "region", ((), ("a", "a"), ("absent",), ([],), ("a", "b", "outside", "a"))
)
def test_invalid_region_membership_is_rejected(region):
    with pytest.raises(ValueError):
        observe_regional_support_cut(_source(), region)


@pytest.mark.parametrize("region", ("a", b"a", {"a"}, {"a": 1}, None))
def test_unordered_or_scalar_region_is_rejected(region):
    with pytest.raises(TypeError):
        observe_regional_support_cut(_source(), region)


def test_full_support_admission_checks_one_extra_element_then_stops():
    consumed = []

    def endless():
        while True:
            consumed.append("a")
            yield "a"

    with pytest.raises(ValueError, match="subset"):
        observe_regional_support_cut(_source(), endless())
    assert len(consumed) == 4
    empty = _from_data((), (), (), (), (), ())
    with pytest.raises(ValueError, match="nonempty"):
        observe_regional_support_cut(empty, ())
