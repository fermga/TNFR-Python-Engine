"""Test utilities."""

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.utils import cached_import, prune_failed_imports


@pytest.fixture
def graph_canon():
    """Return a new graph with default attributes attached."""

    def _factory():
        G = nx.Graph()
        inject_defaults(G)
        return G

    return _factory


@pytest.fixture(scope="module")
def reset_cached_import():
    """Provide a helper to reset cached import state for tests."""

    def _reset() -> None:
        cached_import.cache_clear()
        prune_failed_imports()

    _reset()
    yield _reset
    _reset()
