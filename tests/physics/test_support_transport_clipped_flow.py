"""Exact hard-clipped held-rate references, separate from runtime execution."""

from dataclasses import replace
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.physics.support_transport import (
    observe_support_transport,
    observe_support_transport_clipped_flow,
)

F = Fraction


def _snapshot(epi=(0.0, 0.0), capacity=(0.5, 2.0), pressure=(4.0, -1.0)):
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    for node, value, nu, p in zip(graph, epi, capacity, pressure, strict=True):
        graph.nodes[node].update(EPI=value, nu_f=nu, delta_nfr=p)
    return observe_support_transport(graph)


def _observe(before, after, dt=F(1), *, lower=F(-1), upper=F(1)):
    return observe_support_transport_clipped_flow(
        before, after, dt, lower=lower, upper=upper
    )


def test_clipping_and_remaining_endpoint_work_are_separate_exact_terms():
    before = _snapshot()
    # This supplied endpoint tests arithmetic only; it is not a runtime receipt.
    after = replace(before, epi=(F(1), F(-3, 4)))
    result = _observe(before, after)
    # Weight two gives E=(x0-x1)^2. Held velocity=(2,-2), so the
    # unconstrained, clipped and supplied energies are 16, 4 and 49/16.
    assert result.held.expected_epi == (2, -2)
    assert result.held.drift_term == 0
    assert result.held.quadratic_term == 16
    assert result.unclipped_energy_change == 16
    assert result.clipped_epi == (1, -1)
    assert result.clipping_term == -12
    assert result.clipped_energy_change == 4
    assert result.implementation_state_defect == (0, F(1, 4))
    assert result.implementation_term == F(-15, 16)
    assert result.held.energy_change == F(49, 16)
    assert result.clipping_term + result.implementation_term == result.held.defect_term
    assert result.identity_residual == result.held.identity_residual == 0


def test_interior_reference_has_no_clipping_or_implementation_defect():
    before = _snapshot((0.25, -0.25), (1.0, 1.0), (0.5, -0.5))
    after = replace(before, epi=(F(1, 2), F(-1, 2)))
    result = _observe(before, after, F(1, 2))
    assert result.clipped_epi == (F(1, 2), F(-1, 2))
    assert result.unclipped_energy_change == result.clipped_energy_change == F(3, 4)
    assert result.clipping_term == result.implementation_term == 0
    assert result.implementation_state_defect == (0, 0)
    assert result.identity_residual == 0


@pytest.mark.parametrize(
    "epi,velocity",
    [
        ((F(0), F(1, 4)), (F(2), F(-3))),
        ((F(0), F(1, 4)), (F(1, 8), F(-1, 4))),
        ((F(1), F(-1)), (F(2), F(-3))),
        ((F(1), F(-1)), (F(-1, 4), F(1, 4))),
        ((F(1), F(-1)), (F(0), F(0))),
    ],
)
def test_exact_constant_rate_reference_composes_across_a_rational_partition(
    epi, velocity
):
    # Compose detached model references, with no extra engine/phase calls.
    before = replace(
        _snapshot(), epi=epi, capacity=(F(1), F(1)), stored_pressure=velocity
    )
    whole = _observe(before, before, F(1))
    first = _observe(before, before, F(1, 3))
    intermediate = replace(before, epi=first.clipped_epi)
    second = _observe(intermediate, intermediate, F(2, 3))
    assert second.clipped_epi == whole.clipped_epi


def test_changing_rate_does_not_inherit_the_constant_rate_semigroup():
    before = _snapshot((0.0, 0.0), (1.0, 1.0), (2.0, -2.0))
    first = _observe(before, before)
    reversed_rate = replace(
        before, epi=first.clipped_epi, stored_pressure=(F(-2), F(2))
    )
    second = _observe(reversed_rate, reversed_rate)
    # Opposite areas cancel before clipping, but saturated sequential motion
    # loses that cancellation. Fixed pressure is an essential hypothesis.
    assert second.clipped_epi == (-1, 1)
    assert second.clipped_epi != before.epi


def test_zero_duration_and_a_degenerate_common_interval_are_admitted():
    before = _snapshot((0.5, 0.5))
    zero = _observe(before, before, F(0), lower=F(1, 2), upper=F(1, 2))
    positive = _observe(before, before, F(1), lower=F(1, 2), upper=F(1, 2))
    assert zero.clipped_epi == positive.clipped_epi == (F(1, 2), F(1, 2))
    assert zero.unclipped_energy_change == zero.clipping_term == 0
    assert positive.unclipped_energy_change == 16
    assert positive.clipping_term == -16
    assert zero.clipped_energy_change == positive.clipped_energy_change == 0
    assert zero.identity_residual == positive.identity_residual == 0


def test_public_transport_caches_are_rebuilt_before_clipped_accounting():
    before = _snapshot()
    after = replace(before, epi=(F(1), F(-1)))
    expected = _observe(before, after)

    def forge(value):
        return replace(
            value,
            epi_gradient=(F(999), F(999)),
            dirichlet_gradient=(F(999), F(999)),
            rate=(F(999), F(999)),
            dirichlet_energy=F(999),
            energy_rate=F(999),
        )

    assert _observe(forge(before), forge(after)) == expected


@pytest.mark.parametrize(
    "field,value",
    [("conductance", ()), ("capacity", (F(1), F(2)))],
)
def test_changed_transport_or_capacity_is_not_a_held_flow(field, value):
    before = _snapshot()
    with pytest.raises(ValueError):
        _observe(before, replace(before, **{field: value}))


@pytest.mark.parametrize("epi", [(F(5, 4), F(0)), (F(0), F(-5, 4))])
def test_initial_state_outside_the_rails_does_not_enter_the_semigroup_class(epi):
    before = replace(_snapshot(), epi=epi)
    with pytest.raises(ValueError):
        _observe(before, before, F(0))


@pytest.mark.parametrize(
    "lower,upper,exception",
    [
        (F(1), F(-1), ValueError),
        (float("nan"), F(1), ValueError),
        (F(-1), float("inf"), ValueError),
        (False, F(1), TypeError),
        ("-1", F(1), TypeError),
        (F(-1), complex(1, 0), TypeError),
    ],
)
def test_rails_require_finite_ordered_real_scalars(lower, upper, exception):
    before = _snapshot()
    with pytest.raises(exception):
        _observe(before, before, lower=lower, upper=upper)


def test_negative_duration_does_not_define_forward_saturation():
    before = _snapshot()
    with pytest.raises(ValueError):
        _observe(before, before, F(-1))
