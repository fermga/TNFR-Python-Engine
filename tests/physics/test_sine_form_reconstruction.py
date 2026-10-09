"""Form information and original-clock phase reconstruction at zero loss."""

from fractions import Fraction as Q
from functools import partial

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_forecast import _sine_flow


def _source(model, *, forms, phases, capacities):
    graph = nx.path_graph(2)
    graph.graph["GAMMA"] = {"type": "none"}
    for node, x, theta, nu in zip(graph, forms, phases, capacities):
        graph.nodes[node].update(EPI=x, theta=theta, nu_f=nu)
    return bound_relational_sine_exchange(graph, reference_model=model)


def _zero(bound, tolerance=Q(1, 10**25)):
    assert bound.contains(0)
    assert bound.abs_max < tolerance


def test_identical_pressure_phase_and_capacity_do_not_reconstruct_form_or_phase_rate():
    model = RelationalExchangeModel(
        2, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    first, second = (
        _source(model, forms=forms, phases=(0, Q(1, 3)), capacities=(1, 2))
        for forms in ((1, -2), (2, -4))
    )
    assert first.epi != second.epi
    for name in ("nodes", "edges", "phase", "capacity", "pressure", "form_rates"):
        assert getattr(first, name) == getattr(second, name)
    assert first.pressure[0].lo > 0
    for source in (first, second):
        assert sum((x / nu for x, nu in zip(source.epi, source.capacity)), Q(0)) == 0
    assert second.phase_rate_numerators() == tuple(
        2 * value for value in first.phase_rate_numerators()
    )
    assert first.phase_rates[0].hi < second.phase_rates[0].lo


@pytest.mark.parametrize("epi_weight", (Q(0), Q(2)))
def test_original_clock_velocity_reconstructs_form_and_second_order_phase(epi_weight):
    model = RelationalExchangeModel(
        2, epi_weight=epi_weight, phase_weight=3, phase_domain="regular"
    )
    source = _source(
        model,
        forms=(Q(5, 4), Q(-1, 2)),
        phases=(Q(1, 5), Q(-2, 5)),
        capacities=(1, Q(3, 2)),
    )
    e, w = map(Q, model.effective_weights)
    beta = Q(model.storage_scale)
    nu0, nu1 = source.capacity
    sigma = nu0 + nu1
    mean = (nu1 * source.epi[0] + nu0 * source.epi[1]) / sigma
    numerator0, numerator1 = source.phase_rate_numerators()
    # theta_dot_i=numerator_i/pi in the original clock, including e=0.
    contrast = beta * (numerator0 - numerator1) / (w * sigma)
    assert (mean + nu0 * contrast / sigma, mean - nu1 * contrast / sigma) == source.epi
    assert numerator0 / nu0 + numerator1 / nu1 == 0

    jets = flow_jets(
        tuple(I(v) for v in source.epi + source.phase + (nu1,)),
        2,
        partial(
            _sine_flow,
            neighbors=((1,), (0,)),
            visible_capacity=(nu0,),
            model=model,
        ),
    )
    delta_rate = jets[3][1] - jets[2][1]
    acceleration = 2 * (jets[3][2] - jets[2][2])
    restoring = (
        (w**2 * sigma**2 / beta)
        * sin(I(source.phase[1] - source.phase[0]))
        / pi_interval() ** 2
    )
    _zero(acceleration + e * sigma * delta_rate + restoring)
    assert acceleration.abs_max > Q(1, 1000)
    for i in range(2):
        # The detached trigonometric report and flow jets use different
        # enclosure precisions; agreement still requires actual overlap.
        _zero(jets[i][1] - source.form_rates[i], Q(1, 10**17))
        _zero(jets[2 + i][1] - source.phase_rates[i])
