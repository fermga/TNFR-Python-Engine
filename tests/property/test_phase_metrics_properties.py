"""Property-based tests covering neighbour phase metrics stability."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st

from tnfr.metrics.trig import neighbor_phase_mean_bulk, neighbor_phase_mean_list

from .strategies import (
    PROPERTY_TEST_SETTINGS,
    PhaseBulkScenario,
    PhaseNeighbourhood,
    phase_bulk_scenarios,
    phase_neighbourhoods,
)


def _wrap_angle(angle: float) -> float:
    """Normalise ``angle`` to the principal value in ``[-π, π]``.

    Circular comparisons require an explicit normalisation so tests remain
    stable under the 2π discontinuity of arctangent outputs. This helper is
    referenced in assertions whenever an equality is expected modulo 2π.
    """

    return math.atan2(math.sin(angle), math.cos(angle))


@PROPERTY_TEST_SETTINGS
@given(
    neighbourhood=phase_neighbourhoods(min_neighbours=1),
    rotation=st.floats(
        min_value=-2.0 * math.pi,
        max_value=2.0 * math.pi,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_neighbor_phase_mean_list_rotates_by_constant(
    neighbourhood: PhaseNeighbourhood, rotation: float
) -> None:
    """Rotating all angles by a constant should shift the mean identically."""

    base_mean = neighbor_phase_mean_list(
        neighbourhood.neighbours,
        neighbourhood.cos_map,
        neighbourhood.sin_map,
        fallback=neighbourhood.fallback,
    )

    rotated_cos = {
        key: math.cos(angle + rotation) for key, angle in neighbourhood.angles.items()
    }
    rotated_sin = {
        key: math.sin(angle + rotation) for key, angle in neighbourhood.angles.items()
    }

    rotated_mean = neighbor_phase_mean_list(
        neighbourhood.neighbours,
        rotated_cos,
        rotated_sin,
        fallback=neighbourhood.fallback + rotation,
    )

    delta = _wrap_angle(rotated_mean - base_mean - rotation)
    assert math.isclose(delta, 0.0, abs_tol=1e-9)


@PROPERTY_TEST_SETTINGS
@given(
    data=st.data(),
    neighbourhood=phase_neighbourhoods(min_neighbours=1),
)
def test_neighbor_phase_mean_list_invariant_to_permutation(
    data: st.DataObject, neighbourhood: PhaseNeighbourhood
) -> None:
    """Neighbour order must not affect the circular mean."""

    base_mean = neighbor_phase_mean_list(
        neighbourhood.neighbours,
        neighbourhood.cos_map,
        neighbourhood.sin_map,
        fallback=neighbourhood.fallback,
    )

    permuted: Sequence[str] = data.draw(
        st.permutations(neighbourhood.neighbours),
        label="permuted_neighbours",
    )
    permuted_mean = neighbor_phase_mean_list(
        permuted,
        neighbourhood.cos_map,
        neighbourhood.sin_map,
        fallback=neighbourhood.fallback,
    )

    assert math.isclose(_wrap_angle(permuted_mean - base_mean), 0.0, abs_tol=1e-9)


@PROPERTY_TEST_SETTINGS
@given(neighbourhood=phase_neighbourhoods(min_neighbours=0, max_neighbours=0))
def test_neighbor_phase_mean_list_respects_fallback(
    neighbourhood: PhaseNeighbourhood,
) -> None:
    """Empty neighbourhoods must reuse the fallback phase consistently."""

    first = neighbor_phase_mean_list(
        neighbourhood.neighbours,
        neighbourhood.cos_map,
        neighbourhood.sin_map,
        fallback=neighbourhood.fallback,
    )
    second = neighbor_phase_mean_list(
        neighbourhood.neighbours,
        neighbourhood.cos_map,
        neighbourhood.sin_map,
        fallback=neighbourhood.fallback,
    )

    assert math.isclose(first, neighbourhood.fallback, abs_tol=0.0)
    assert math.isclose(second, neighbourhood.fallback, abs_tol=0.0)


@PROPERTY_TEST_SETTINGS
@given(scenario=phase_bulk_scenarios())
def test_neighbor_phase_mean_bulk_masks_nodes_without_neighbours(
    scenario: PhaseBulkScenario,
) -> None:
    """The boolean mask must align with the explicit neighbour counts."""

    mean_theta, has_neighbours = neighbor_phase_mean_bulk(
        scenario.edge_src,
        scenario.edge_dst,
        cos_values=scenario.cos_values,
        sin_values=scenario.sin_values,
        theta_values=scenario.theta_values,
        node_count=scenario.node_count,
        np=np,
    )

    expected_counts = np.asarray(scenario.neighbour_counts, dtype=float)
    assert has_neighbours.shape == expected_counts.shape
    assert np.array_equal(has_neighbours, expected_counts > 0.0)

    if scenario.node_count:
        theta_values = np.asarray(scenario.theta_values, dtype=float)
        isolated = expected_counts == 0.0
        assert np.allclose(mean_theta[isolated], theta_values[isolated])
        assert has_neighbours.dtype == bool
