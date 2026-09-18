"""An EPI chart, magnitude observation and directed pressure are distinct."""

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.mathematics import BanachSpaceEPI, BEPIElement
from tnfr.physics.support_transport import observe_support_transport
from tnfr.types import ensure_bepi, real_scalar_epi, scalarize_epi


def _pressure(field):
    graph = nx.path_graph(2)
    graph.graph["DNFR_WEIGHTS"] = dict(epi=1.0, phase=0.0, vf=0.0, topo=0.0)
    for node, value in zip(graph, field):
        graph.nodes[node].update(EPI=value, nu_f=1.0, theta=0.0)
    default_compute_delta_nfr(graph)
    return observe_support_transport(graph)


def test_nonlinear_epi_relabeling_requires_a_pushforward_pressure():
    original = _pressure((1.0, 2.0))
    renamed = _pressure((1.0, 4.0))  # y=x^2, an invertible chart on x>0
    pushed_velocity = tuple(2*x*v for x, v in zip(original.epi, original.rate))
    assert original.rate == (1, -1)
    assert pushed_velocity == (2, -4)
    assert renamed.rate == (3, -3)
    assert pushed_velocity != renamed.rate
    # A new metric/state representation cannot retain the old pressure rule
    # merely by retaining the text of x'=nu*p.


@pytest.mark.parametrize("scale,offset", [(2.0, 3.0), (-2.0, 3.0), (0.5, -1.0)])
def test_pure_epi_gradient_respects_common_affine_chart_changes(scale, offset):
    original = _pressure((1.0, 2.0))
    renamed = _pressure(tuple(scale*x+offset for x in (1.0, 2.0)))
    assert renamed.rate == tuple(scale*v for v in original.rate)


def test_equal_metric_change_magnitudes_do_not_identify_directed_pressure():
    first, second = _pressure((0.0, 1.0)), _pressure((1.0, 0.0))
    assert tuple(abs(p) for p in first.stored_pressure) == (1, 1)
    assert tuple(abs(p) for p in second.stored_pressure) == (1, 1)
    assert first.rate == (1, -1)
    assert second.rate == (-1, 1)


def test_rich_epi_magnitude_is_not_an_injective_or_linear_state_chart():
    left = BEPIElement((1.0, -1.0), (0.0, 0.0), (0.0, 1.0))
    right = BEPIElement((-1.0, 1.0), (0.0, 0.0), (0.0, 1.0))
    assert real_scalar_epi(left) is real_scalar_epi(right) is None
    assert scalarize_epi(left) == scalarize_epi(right) == 1.0
    assert not np.array_equal(left.f_continuous, right.f_continuous)
    assert scalarize_epi(left + right) == 0.0
    assert scalarize_epi(left) + scalarize_epi(right) == 2.0


def test_uniform_real_bepi_storage_is_the_same_signed_scalar_chart():
    for value in (-2.0, 0.0, 2.0):
        stored = ensure_bepi(value)
        assert real_scalar_epi(stored) == scalarize_epi(stored) == value
        assert abs(stored) == abs(value)


def test_historical_paired_basis_does_not_span_the_full_direct_sum():
    space = BanachSpaceEPI()
    columns = []
    for continuous in range(2):
        for discrete in range(2):
            value = space.canonical_basis(
                continuous_size=2, discrete_size=2,
                continuous_index=continuous, discrete_index=discrete,
            )
            columns.append(tuple(value.f_continuous.real) + tuple(value.a_discrete.real))
    # Every generated column lies in the proper hyperplane sum(f)=sum(a).
    assert all(column[0]+column[1]-column[2]-column[3] == 0 for column in columns)
    assert all(column != (1, 0, 0, 0) for column in columns)
