"""Test utilities."""

from __future__ import annotations

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import inject_defaults
from tnfr.utils import cached_import, prune_failed_imports

STRUCTURAL_ATOL = 1e-12
STRUCTURAL_RTOL = 1e-10


@pytest.fixture(scope="session")
def structural_tolerances() -> dict[str, float]:
    """Return the canonical absolute/relative tolerances used in tests."""

    return {"atol": STRUCTURAL_ATOL, "rtol": STRUCTURAL_RTOL}


@pytest.fixture
def structural_rng() -> np.random.Generator:
    """Provide a reproducible RNG aligned with TNFR structural conventions."""

    return np.random.default_rng(seed=0)


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
