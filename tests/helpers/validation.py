"""Shared validation test utilities for structural assertions.

This module centralizes common validation patterns used across
integration, mathematics, property, and stress tests to follow
DRY principles while maintaining TNFR structural invariants.
"""

from __future__ import annotations

import math

import networkx as nx

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY


def assert_dnfr_balanced(
    graph: nx.Graph,
    *,
    abs_tol: float = 1e-9,
) -> None:
    """Assert that total ΔNFR across nodes sums to zero (structural conservation).

    Parameters
    ----------
    graph:
        Network with ΔNFR values on nodes.
    abs_tol:
        Absolute tolerance for zero-sum check.

    Raises
    ------
    AssertionError:
        If total ΔNFR violates conservation law.
    """
    total_dnfr = sum(float(data.get(DNFR_PRIMARY, 0.0)) for _, data in graph.nodes(data=True))
    assert math.isclose(
        total_dnfr, 0.0, abs_tol=abs_tol
    ), f"ΔNFR not conserved: total={total_dnfr}, tolerance={abs_tol}"


def assert_dnfr_homogeneous_stable(
    graph: nx.Graph,
    *,
    abs_tol: float = 1e-9,
) -> None:
    """Assert that homogeneous graphs remain stable (ΔNFR ≈ 0 everywhere).

    Parameters
    ----------
    graph:
        Network with homogeneous EPI and νf values.
    abs_tol:
        Absolute tolerance for zero ΔNFR.

    Raises
    ------
    AssertionError:
        If any node has non-zero ΔNFR beyond tolerance.
    """
    for node, data in graph.nodes(data=True):
        delta = float(data.get(DNFR_PRIMARY, 0.0))
        assert math.isclose(
            delta, 0.0, abs_tol=abs_tol
        ), f"Node {node} ΔNFR={delta} violates homogeneous stability"


def assert_epi_vf_in_bounds(
    graph: nx.Graph,
    *,
    epi_min: float | None = None,
    epi_max: float | None = None,
    vf_min: float | None = None,
    vf_max: float | None = None,
) -> None:
    """Assert that EPI and νf values remain within structural bounds.

    Parameters
    ----------
    graph:
        Network with EPI and νf attributes.
    epi_min, epi_max:
        Optional bounds for EPI values.
    vf_min, vf_max:
        Optional bounds for νf values.

    Raises
    ------
    AssertionError:
        If any node violates the specified bounds.
    """
    for node, data in graph.nodes(data=True):
        epi = float(data.get(EPI_PRIMARY, 0.0))
        vf = float(data.get(VF_PRIMARY, 0.0))

        if epi_min is not None:
            assert epi >= epi_min, f"Node {node} EPI={epi} below minimum {epi_min}"
        if epi_max is not None:
            assert epi <= epi_max, f"Node {node} EPI={epi} above maximum {epi_max}"
        if vf_min is not None:
            assert vf >= vf_min, f"Node {node} νf={vf} below minimum {vf_min}"
        if vf_max is not None:
            assert vf <= vf_max, f"Node {node} νf={vf} above maximum {vf_max}"


def assert_graph_has_tnfr_defaults(graph: nx.Graph) -> None:
    """Assert that graph has required TNFR structural defaults.

    Parameters
    ----------
    graph:
        Network to validate.

    Raises
    ------
    AssertionError:
        If required TNFR defaults are missing.
    """
    from tnfr.constants import DEFAULTS

    # Verify presence of key default structures
    required_keys = list(DEFAULTS.keys())[:3]  # Sample of required keys
    for key in required_keys:
        assert key in graph.graph, f"Missing required TNFR default: {key}"


def get_dnfr_values(graph: nx.Graph) -> list[float]:
    """Extract all ΔNFR values from graph nodes.

    Parameters
    ----------
    graph:
        Network with ΔNFR attributes.

    Returns
    -------
    list[float]:
        Sorted list of ΔNFR values.
    """
    return sorted(float(data.get(DNFR_PRIMARY, 0.0)) for _, data in graph.nodes(data=True))


def assert_dnfr_lists_close(
    left: list[float],
    right: list[float],
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> None:
    """Assert two ΔNFR value lists are element-wise close.

    Parameters
    ----------
    left, right:
        Lists of ΔNFR values to compare.
    rel_tol:
        Relative tolerance.
    abs_tol:
        Absolute tolerance.

    Raises
    ------
    AssertionError:
        If lists differ in length or values.
    """
    assert len(left) == len(right), f"Length mismatch: {len(left)} vs {len(right)}"
    for i, (l_val, r_val) in enumerate(zip(left, right)):
        assert math.isclose(
            l_val, r_val, rel_tol=rel_tol, abs_tol=abs_tol
        ), f"Index {i}: {l_val} ≠ {r_val}"
