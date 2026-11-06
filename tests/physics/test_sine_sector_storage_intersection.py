"""Correlated producer bounds meet the common sector storage enclosure."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import (
    _certify_sine_sector_set,
    _sector_source_admission,
    _source_data,
    certify_sine_sector_capture,
)


@pytest.fixture(scope="module")
def correlated_family():
    graph = nx.path_graph(3)
    for node, form, phase in zip(graph, (0, Q(1, 2), 1), (0, Q(1, 8), Q(1, 4))):
        graph.nodes[node].update(EPI=float(form), theta=float(phase), nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_pattern(
        graph,
        reference_model=RelationalExchangeModel(1, phase_domain="regular"),
        reference_node=0,
        form_error_bounds=(0, Q(1, 2), 0),
        phase_error_bounds=(0, Q(1, 8), 0),
    )
    admitted, geometry = _sector_source_admission(source)
    _, model, capacities, exact_capacity, gap, scope, _, _, _ = _source_data(admitted)
    return dict(
        source=source,
        geometry=geometry,
        model=model,
        capacity_bounds=capacities,
        exact_held_capacity=exact_capacity,
        form_edge_gap_bounds=tuple(gap(i, j) for i, j in geometry.edges),
        phase_edge_gap_bounds=tuple(gap(i, j, phase=True) for i, j in geometry.edges),
        edge_turn_offsets=(0, 0),
        uncertainty_scope=scope,
    )


def test_default_and_looser_storage_bounds_preserve_the_existing_report(
    correlated_family,
):
    original = certify_sine_sector_capture(
        correlated_family["source"], edge_turn_offsets=(0, 0)
    )
    assert _certify_sine_sector_set(**correlated_family) == original
    assert (
        _certify_sine_sector_set(
            **correlated_family,
            correlated_form_storage_upper_bound=10,
            correlated_phase_storage_upper_bound=10,
        )
        == original
    )


def test_independent_correlations_tighten_full_storage_and_capture(correlated_family):
    # Up to common origins, x=(0,t,1), 0<=t<=1, and theta=(0,s,delta),
    # 0<=s<=delta=1/4. The exact identity t²+(1-t)²=1-2t(1-t)<=1
    # proves form storage<=1/2. Likewise s²+(delta-s)²<=delta² and
    # 1-cos(a)<=a²/2 prove phase storage<=delta²/2=1/32 for the whole set.
    original = _certify_sine_sector_set(**correlated_family)
    tightened = _certify_sine_sector_set(
        **correlated_family,
        correlated_form_storage_upper_bound=Q(1, 2),
        correlated_phase_storage_upper_bound=Q(1, 32),
    )
    assert not original.admitted and tightened.admitted
    assert original.form_storage_bounds.hi == 1
    assert tightened.form_storage_bounds.hi == Q(1, 2)
    assert tightened.phase_storage_bounds.hi == Q(1, 32)
    assert tightened.storage_upper_bound == Q(17, 32)
    assert tightened.energy_margin == Q(15, 32)
    assert tightened.form_storage_bounds.lo == original.form_storage_bounds.lo
    assert tightened.phase_storage_bounds.lo == original.phase_storage_bounds.lo
    assert tightened.form_edge_gap_bounds == original.form_edge_gap_bounds
    assert tightened.phase_edge_gap_bounds == original.phase_edge_gap_bounds
    assert tightened.source is correlated_family["source"]


@pytest.mark.parametrize("kind", ("form", "phase"))
@pytest.mark.parametrize("value", (True, -1, float("nan"), float("inf")))
def test_invalid_correlated_storage_bounds_reject(correlated_family, kind, value):
    with pytest.raises((TypeError, ValueError)):
        _certify_sine_sector_set(
            **correlated_family,
            **{f"correlated_{kind}_storage_upper_bound": value},
        )


@pytest.mark.parametrize("kind", ("form", "phase"))
def test_correlated_upper_bound_cannot_discard_edgewise_lower_bound(
    correlated_family, kind
):
    arguments = {
        **correlated_family,
        f"{kind}_edge_gap_bounds": (I(Q(1, 4)),) * 2,
        f"correlated_{kind}_storage_upper_bound": Q(0),
    }
    with pytest.raises(ValueError, match="inconsistent.*lower bound"):
        _certify_sine_sector_set(**arguments)
