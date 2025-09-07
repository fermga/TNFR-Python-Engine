"""Utilidades de pruebas."""

import pytest
import networkx as nx

from tnfr.constants import inject_defaults


@pytest.fixture
def graph_canon():
    """Return a new graph with default attributes attached."""

    def _factory():
        G = nx.Graph()
        inject_defaults(G)
        return G

    return _factory
