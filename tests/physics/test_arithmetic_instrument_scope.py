"""Finite symmetry instruments require supplied groups and complete evidence."""

from pathlib import Path
from runpy import run_path

import networkx as nx
import numpy as np
import pytest


@pytest.fixture(scope="module")
def inverse_demo():
    return run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "benchmarks/inverse_spectrum_to_symmetry.py"
        )
    )


def test_group_comparison_requires_a_supplied_group_and_does_not_infer(inverse_demo):
    compare = inverse_demo["compare_supplied_group"]
    with pytest.raises(TypeError, match="group"):
        compare([1, 3, 5])
    icosahedral = compare([1, 5], group="I (icosahedral)")
    octahedral = compare([1, 5], group="O (octahedral)")
    assert icosahedral.supplied_group != octahedral.supplied_group
    assert not icosahedral.group_inference_available
    assert not octahedral.group_inference_available
    assert "infer_group" not in inverse_demo


def test_allowed_irrep_dimension_need_not_occur_in_the_selected_graph(inverse_demo):
    observed = inverse_demo["full_modes"](nx.icosahedral_graph())
    comparison = inverse_demo["compare_supplied_group"](
        observed, group="I (icosahedral)"
    )
    assert 4 in comparison.unseen_irrep_dimensions
    assert 4 not in observed
    assert 4 in inverse_demo["full_modes"](nx.dodecahedral_graph())


def test_complete_enumeration_accepts_its_boundary_and_rejects_truncation(inverse_demo):
    enumerate_group = inverse_demo["automorphism_matrices"]
    graph = nx.path_graph(3)  # Identity and reversal are its two automorphisms.
    assert len(enumerate_group(graph, limit=2)) == 2
    with pytest.raises(ValueError, match="incomplete automorphism enumeration"):
        enumerate_group(graph, limit=1)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_enumeration_limit_is_an_actual_positive_integer(inverse_demo, limit):
    with pytest.raises(ValueError, match="positive integer"):
        inverse_demo["automorphism_matrices"](nx.path_graph(3), limit=limit)


def test_missing_fivefold_modes_cannot_pass_the_ambiguity_control(
    inverse_demo, monkeypatch
):
    control = inverse_demo["control_irreducibility"]
    monkeypatch.setitem(
        control.__globals__, "eigenspace_irreducibility", lambda graph: [(0, 1, 1)]
    )
    assert control() is False


def test_missing_counterexample_constructor_cannot_pass(inverse_demo, monkeypatch):
    monkeypatch.delattr(nx, "truncated_cube_graph")
    with pytest.raises(AttributeError, match="truncated_cube_graph"):
        inverse_demo["control_irreducibility"]()


@pytest.fixture(scope="module")
def directed_paley_demo():
    return run_path(
        str(Path(__file__).resolve().parents[2] / "benchmarks/directed_paley_bridge.py")
    )


def test_real_axis_distance_uses_the_circle_at_zero_and_pi_seams(directed_paley_demo):
    phases = np.array([-0.1, 0.1, np.pi - 0.1, -np.pi + 0.1, np.pi + 0.1])
    distances = directed_paley_demo["real_axis_distances"](phases)
    np.testing.assert_allclose(distances, np.full(phases.size, 0.1), atol=1e-14)
    # In particular, -0.1 must not become an off-axis point near 2*pi.
    assert not np.any(distances > 0.3)


def test_real_axis_distance_matches_independent_geometry_and_periodicity(
    directed_paley_demo,
):
    phases = np.array([-2.4, -1.2, -0.4, 0.0, 0.6, 1.5, 2.8])
    distance = directed_paley_demo["real_axis_distances"]
    # Distance to either real ray is arcsin of the unit circle's vertical height.
    expected = np.arcsin(np.abs(np.sin(phases)))
    np.testing.assert_allclose(distance(phases), expected, atol=1e-14)
    np.testing.assert_allclose(distance(phases + 6 * np.pi), expected, atol=1e-14)


def test_phase_sample_report_cannot_count_near_zero_negative_angles_as_off_axis(
    directed_paley_demo,
    monkeypatch,
    capsys,
):
    control = directed_paley_demo["test_finite_spectra_and_phase_samples"]
    monkeypatch.setitem(
        control.__globals__, "riemann_s_phase", lambda *args: -0.1 / np.pi
    )
    assert control(limit=31) is False
    assert "samples beyond chosen angular cut: 0 / 18" in capsys.readouterr().out
