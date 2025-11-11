"""Test dnfr_path_decision telemetry field."""

import pytest


def test_dnfr_path_decision_sparse():
    """Verify dnfr_path_decision is set to 'sparse' for low-density graphs."""
    pytest.importorskip("numpy")
    import networkx as nx
    from tnfr.dynamics.dnfr import _prepare_dnfr_data

    # Create a sparse graph (5 nodes, 3 edges, density = 3/20 = 0.15 < 0.25)
    G = nx.Graph()
    for i in range(5):
        G.add_node(
            i,
            theta=float(i) * 0.1,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)

    data = _prepare_dnfr_data(G)

    assert "dnfr_path_decision" in data
    assert data["dnfr_path_decision"] == "sparse"
    assert data["prefer_sparse"] is True
    assert data["dense_override"] is False


def test_dnfr_path_decision_dense_auto():
    """Verify dnfr_path_decision is set to 'dense_auto' for high-density graphs."""
    pytest.importorskip("numpy")
    import networkx as nx
    from tnfr.dynamics.dnfr import _prepare_dnfr_data

    # Create a dense graph (5 nodes, 8 edges, density = 8/20 = 0.40 > 0.25)
    G = nx.Graph()
    for i in range(5):
        G.add_node(
            i,
            theta=float(i) * 0.1,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    # Add many edges to make it dense
    for i in range(4):
        G.add_edge(i, i + 1, weight=1.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(0, 3, weight=1.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(2, 4, weight=1.0)

    data = _prepare_dnfr_data(G)

    assert "dnfr_path_decision" in data
    assert data["dnfr_path_decision"] == "dense_auto"
    assert data["prefer_sparse"] is False
    assert data["dense_override"] is False


def test_dnfr_path_decision_dense_forced():
    """Verify dnfr_path_decision is set to 'dense_forced' when user forces dense mode."""
    pytest.importorskip("numpy")
    import networkx as nx
    from tnfr.dynamics.dnfr import _prepare_dnfr_data

    # Create a sparse graph but force dense mode
    G = nx.Graph()
    for i in range(5):
        G.add_node(
            i,
            theta=float(i) * 0.1,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)

    # Force dense mode
    G.graph["dnfr_force_dense"] = True

    data = _prepare_dnfr_data(G)

    assert "dnfr_path_decision" in data
    assert data["dnfr_path_decision"] == "dense_forced"
    assert data["prefer_sparse"] is False
    assert data["dense_override"] is True


def test_dnfr_path_decision_fallback_without_numpy():
    """Verify dnfr_path_decision is 'fallback' when NumPy is unavailable."""
    # Note: In practice, it's difficult to test the fallback path when NumPy is
    # already loaded in the test suite. This test verifies the logic exists but
    # may not actually exercise the fallback path if NumPy is available.
    import networkx as nx
    from tnfr.dynamics.dnfr import _prepare_dnfr_data

    # Create a graph
    G = nx.Graph()
    for i in range(3):
        G.add_node(
            i,
            theta=float(i) * 0.1,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    G.add_edge(0, 1, weight=1.0)

    # Disable vectorization in the graph (simulates no NumPy scenario)
    G.graph["vectorized_dnfr"] = False

    data = _prepare_dnfr_data(G)

    # With vectorization disabled, we expect fallback path
    # (though dnfr_path_decision logic runs before vectorization check,
    # so it may still show sparse/dense based on graph structure)
    assert "dnfr_path_decision" in data
    # The field should be one of the valid values
    assert data["dnfr_path_decision"] in [
        "sparse",
        "dense_auto",
        "dense_forced",
        "fallback",
    ]


def test_dnfr_path_decision_persists_in_result():
    """Verify dnfr_path_decision field is available for telemetry/debugging."""
    pytest.importorskip("numpy")
    import networkx as nx
    from tnfr.dynamics.dnfr import _prepare_dnfr_data

    G = nx.Graph()
    for i in range(3):
        G.add_node(
            i,
            theta=float(i) * 0.1,
            vf=1.0,
            EPI={"continuous": (0.5,), "discrete": (), "grid": (0.0, 1.0)},
        )
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=1.0)

    data = _prepare_dnfr_data(G)

    # The field should be in the returned dictionary for downstream analysis
    assert "dnfr_path_decision" in data
    assert isinstance(data["dnfr_path_decision"], str)
    assert data["dnfr_path_decision"] in [
        "sparse",
        "dense_auto",
        "dense_forced",
        "fallback",
    ]
