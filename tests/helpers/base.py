"""Shared base classes and parametrized fixtures for test optimization.

This module provides reusable test base classes to eliminate redundancy
across integration, mathematics, property, and stress test suites.
Following DRY principles while maintaining TNFR structural fidelity.
"""

from __future__ import annotations

import abc
from typing import Any, Callable

import networkx as nx
import pytest

from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY, inject_defaults
from tests.helpers.validation import (
    assert_dnfr_balanced,
    assert_dnfr_homogeneous_stable,
    assert_epi_vf_in_bounds,
)


class BaseStructuralTest(abc.ABC):
    """Base class for structural validation tests.
    
    Provides common test patterns for ΔNFR conservation, homogeneity,
    and structural bounds checking that can be reused across test suites.
    """

    @abc.abstractmethod
    def create_test_graph(self, **kwargs: Any) -> nx.Graph:
        """Create a test graph with specified parameters.
        
        Subclasses must implement this to provide domain-specific graphs.
        """
        pass

    def test_dnfr_conservation(self, **graph_params: Any) -> None:
        """Verify ΔNFR conservation holds for the test graph."""
        graph = self.create_test_graph(**graph_params)
        from tnfr.dynamics import dnfr_epi_vf_mixed
        dnfr_epi_vf_mixed(graph)
        assert_dnfr_balanced(graph)

    def test_homogeneous_stability(self, **graph_params: Any) -> None:
        """Verify homogeneous graphs remain stable."""
        graph = self.create_test_graph(**graph_params)
        # Ensure homogeneity
        epi_val = graph_params.get("epi_value", 0.5)
        vf_val = graph_params.get("vf_value", 1.0)
        for _, data in graph.nodes(data=True):
            data[EPI_PRIMARY] = epi_val
            data[VF_PRIMARY] = vf_val
            data[DNFR_PRIMARY] = 0.0
        
        from tnfr.dynamics import dnfr_epi_vf_mixed
        dnfr_epi_vf_mixed(graph)
        assert_dnfr_homogeneous_stable(graph)

    def test_structural_bounds(self, **graph_params: Any) -> None:
        """Verify EPI and νf values remain within bounds."""
        graph = self.create_test_graph(**graph_params)
        epi_min = graph_params.get("epi_min", -2.0)
        epi_max = graph_params.get("epi_max", 2.0)
        vf_min = graph_params.get("vf_min", 0.0)
        vf_max = graph_params.get("vf_max", 5.0)
        assert_epi_vf_in_bounds(
            graph,
            epi_min=epi_min,
            epi_max=epi_max,
            vf_min=vf_min,
            vf_max=vf_max,
        )


class BaseOperatorTest(abc.ABC):
    """Base class for operator generation and validation tests.
    
    Provides common patterns for testing operator properties including
    hermiticity, spectral properties, and parameter validation.
    """

    @abc.abstractmethod
    def create_operator(self, **kwargs: Any) -> Any:
        """Create an operator with specified parameters.
        
        Subclasses must implement this to provide domain-specific operators.
        """
        pass

    def assert_hermitian(self, operator: Any, atol: float = 1e-9) -> None:
        """Assert operator is Hermitian (self-adjoint)."""
        import numpy as np
        if hasattr(operator, "matrix"):
            matrix = operator.matrix
        else:
            matrix = operator
        assert np.allclose(matrix, matrix.conj().T, atol=atol), "Operator not Hermitian"

    def assert_finite_values(self, operator: Any) -> None:
        """Assert operator contains only finite values."""
        import numpy as np
        if hasattr(operator, "matrix"):
            matrix = operator.matrix
        else:
            matrix = operator
        assert np.all(np.isfinite(matrix)), "Operator contains non-finite values"

    def assert_real_eigenvalues(self, operator: Any, atol: float = 1e-9) -> None:
        """Assert operator eigenvalues are real (characteristic of Hermitian)."""
        import numpy as np
        if hasattr(operator, "eigenvalues"):
            eigenvalues = operator.eigenvalues
        else:
            if hasattr(operator, "matrix"):
                matrix = operator.matrix
            else:
                matrix = operator
            eigenvalues = np.linalg.eigvalsh(matrix)
        assert np.all(np.isreal(eigenvalues)), "Eigenvalues not real"


class BaseValidatorTest(abc.ABC):
    """Base class for nodal and network validator tests.
    
    Provides common patterns for testing validation rules including
    attribute presence, bounds checking, and structural constraints.
    """

    def assert_required_attributes(
        self,
        graph: nx.Graph,
        node: Any,
        required_attrs: list[str],
    ) -> None:
        """Assert node has all required TNFR attributes."""
        for attr in required_attrs:
            assert attr in graph.nodes[node], f"Missing required attribute: {attr}"

    def assert_finite_attributes(
        self,
        graph: nx.Graph,
        node: Any,
        attrs_to_check: list[str],
    ) -> None:
        """Assert node attributes are finite."""
        import math
        for attr in attrs_to_check:
            value = graph.nodes[node].get(attr, 0.0)
            assert math.isfinite(value), f"Attribute {attr} not finite: {value}"

    def assert_phase_in_range(
        self,
        graph: nx.Graph,
        node: Any,
        phase_key: str = "theta",
    ) -> None:
        """Assert phase value is in [-π, π]."""
        import math
        phase = graph.nodes[node].get(phase_key, 0.0)
        assert -math.pi <= phase <= math.pi, f"Phase {phase} out of range"


# Parametrized fixtures for common test scenarios
@pytest.fixture(params=[
    {"num_nodes": 5, "edge_probability": 0.3, "seed": 42},
    {"num_nodes": 10, "edge_probability": 0.4, "seed": 123},
    {"num_nodes": 20, "edge_probability": 0.2, "seed": 456},
])
def parametrized_graph_config(request):
    """Parametrized graph configurations for reducing test redundancy."""
    return request.param


@pytest.fixture(params=[
    {"epi_value": 0.0, "vf_value": 1.0},
    {"epi_value": 0.5, "vf_value": 1.5},
    {"epi_value": -0.3, "vf_value": 0.8},
])
def parametrized_homogeneous_config(request):
    """Parametrized homogeneous configurations for reducing test redundancy."""
    return request.param


@pytest.fixture(params=[2, 3, 4, 5, 8])
def parametrized_operator_dimension(request):
    """Parametrized operator dimensions for reducing test redundancy."""
    return request.param


@pytest.fixture(params=[
    {"topology": "laplacian", "nu_f": 1.0, "scale": 1.0},
    {"topology": "adjacency", "nu_f": 2.0, "scale": 0.5},
    {"topology": "laplacian", "nu_f": 0.5, "scale": 2.0},
])
def parametrized_operator_config(request):
    """Parametrized operator configurations for reducing test redundancy."""
    return request.param
