"""Isolated birth is invisible to edge energy, but not a fixed-support reset."""

import pytest

from benchmarks.thol_birth_transport import prepare_birth_selection_source
from tnfr.operators.definitions import SelfOrganization
from tnfr.physics.support_transport import (
    observe_support_transport, observe_support_transport_reset,
)


def test_actual_isolated_birth_extends_the_dirichlet_form_by_a_zero_block():
    graph, _ = prepare_birth_selection_source()
    before = observe_support_transport(graph)
    SelfOrganization()(graph, 0)
    after = observe_support_transport(graph)

    assert after.nodes[:-1] == before.nodes
    assert after.epi[:-1] == before.epi
    assert after.conductance == before.conductance
    assert after.support_neighbors[:-1] == before.support_neighbors
    assert after.support_neighbors[-1] == ()
    assert after.dirichlet_gradient == (*before.dirichlet_gradient, 0)
    assert after.dirichlet_energy == before.dirichlet_energy
    assert after.epi[-1] != 0 and after.capacity[-1] > 0
    # Positive capacity does not supply degree: H_child=d_child/nu_child=0.
    child_index = len(after.nodes) - 1
    assert sum(weight for i, _, weight in after.conductance if i == child_index) == 0

    # Do not feed the dimensional jump to the fixed-support reset identity.
    with pytest.raises(ValueError, match="identical node order"):
        observe_support_transport_reset(before, after)
