"""Tests verifying @runtime_checkable on TNFR type protocols.

This module validates that core TNFR type protocols support isinstance checks
through the @runtime_checkable decorator, ensuring structural validation at runtime.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import pytest

from tnfr.types import (
    BEPIProtocol,
    GraphLike,
    IntegratorProtocol,
)


class TestBEPIProtocol:
    """Test suite for BEPIProtocol runtime checking."""

    def test_bepi_protocol_is_runtime_checkable(self) -> None:
        """Verify BEPIProtocol supports isinstance checks."""

        class MockBEPI:
            """Mock BEPI implementation."""

            def __init__(self):
                self.f_continuous = [1.0, 2.0]
                self.a_discrete = [0.5, 0.5]
                self.x_grid = [0.0, 1.0]

            def direct_sum(self, other: Any) -> Any:
                """Mock direct sum operation."""
                return self

            def tensor(self, vector) -> Any:
                """Mock tensor product."""
                return vector

            def adjoint(self) -> Any:
                """Mock adjoint operation."""
                return self

            def compose(self, transform, *, spectral_transform=None) -> Any:
                """Mock composition."""
                return self

        mock_bepi = MockBEPI()
        assert isinstance(
            mock_bepi, BEPIProtocol
        ), "MockBEPI should satisfy BEPIProtocol"

    def test_bepi_protocol_rejects_incomplete(self) -> None:
        """Verify incomplete BEPI implementations are rejected."""

        class IncompleteBEPI:
            """Missing required methods."""

            def __init__(self):
                self.f_continuous = []
                self.a_discrete = []
                self.x_grid = []

        incomplete = IncompleteBEPI()
        assert not isinstance(
            incomplete, BEPIProtocol
        ), "IncompleteBEPI should not satisfy BEPIProtocol"


class TestGraphLikeProtocol:
    """Test suite for GraphLike protocol runtime checking."""

    def test_graphlike_protocol_is_runtime_checkable(self) -> None:
        """Verify GraphLike protocol supports isinstance checks."""

        class MockNodeView:
            """Mock node view."""

            def __iter__(self):
                return iter([1, 2, 3])

            def __call__(self, data=False):
                return iter([1, 2, 3])

            def __getitem__(self, node):
                return {"data": "value"}

        class MockEdgeView:
            """Mock edge view."""

            def __iter__(self):
                return iter([(1, 2), (2, 3)])

            def __call__(self, data=False):
                return iter([(1, 2), (2, 3)])

        class MockGraph:
            """Mock graph implementation."""

            def __init__(self):
                self.graph: MutableMapping[str, Any] = {}
                self.nodes = MockNodeView()
                self.edges = MockEdgeView()

            def number_of_nodes(self) -> int:
                """Return node count."""
                return 3

            def neighbors(self, n: Any) -> Iterable[Any]:
                """Return neighbors."""
                return iter([])

            def __getitem__(self, node: Any) -> MutableMapping[Any, Any]:
                """Return adjacency dict."""
                return {}

            def __iter__(self) -> Iterable[Any]:
                """Iterate nodes."""
                return iter([1, 2, 3])

        mock_graph = MockGraph()
        assert isinstance(
            mock_graph, GraphLike
        ), "MockGraph should satisfy GraphLike protocol"

    def test_graphlike_protocol_checks_methods(self) -> None:
        """Verify GraphLike requires all protocol methods."""

        class PartialGraph:
            """Missing some required methods."""

            def __init__(self):
                self.graph: MutableMapping[str, Any] = {}

            # Missing nodes, edges, number_of_nodes, etc.

        partial = PartialGraph()
        assert not isinstance(
            partial, GraphLike
        ), "PartialGraph should not satisfy GraphLike protocol"


class TestIntegratorProtocol:
    """Test suite for IntegratorProtocol runtime checking."""

    def test_integrator_protocol_is_runtime_checkable(self) -> None:
        """Verify IntegratorProtocol supports isinstance checks."""

        class MockIntegrator:
            """Mock integrator implementation."""

            def integrate(
                self,
                graph,
                *,
                dt=None,
                t=None,
                method=None,
                n_jobs=None,
            ) -> None:
                """Mock integration step."""
                pass

        mock_integrator = MockIntegrator()
        assert isinstance(
            mock_integrator, IntegratorProtocol
        ), "MockIntegrator should satisfy IntegratorProtocol"

    def test_integrator_protocol_requires_integrate(self) -> None:
        """Verify IntegratorProtocol requires integrate method."""

        class NotAnIntegrator:
            """Missing integrate method."""

            def step(self, graph) -> None:
                """Wrong method name."""
                pass

        not_integrator = NotAnIntegrator()
        assert not isinstance(
            not_integrator, IntegratorProtocol
        ), "NotAnIntegrator should not satisfy IntegratorProtocol"


def test_all_protocols_support_isinstance() -> None:
    """Verify all tested protocols support isinstance at runtime.

    This serves as a regression test to ensure @runtime_checkable
    remains applied to all critical TNFR protocols.
    """
    from typing import Protocol

    protocols = [BEPIProtocol, GraphLike, IntegratorProtocol]

    for protocol in protocols:
        # Verify it's actually a Protocol
        assert issubclass(
            protocol, Protocol
        ), f"{protocol.__name__} should be a Protocol"

        # The most reliable test: verify isinstance works with a mock
        class MockImplementation:
            """Generic mock that will fail isinstance for incomplete protocols."""

            pass

        mock = MockImplementation()

        # This should not raise TypeError (which would happen without @runtime_checkable)
        try:
            result = isinstance(mock, protocol)
            # We don't care if it's True or False, just that it doesn't raise
            assert isinstance(
                result, bool
            ), f"isinstance check for {protocol.__name__} should return bool"
        except TypeError as e:
            pytest.fail(
                f"{protocol.__name__} does not support isinstance checks. "
                f"Likely missing @runtime_checkable decorator. Error: {e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
