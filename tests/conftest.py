"""Test utilities."""

from __future__ import annotations

import os

import networkx as nx
import pytest

np = pytest.importorskip("numpy")

from tnfr.constants import inject_defaults
from tnfr.utils import cached_import, prune_failed_imports

STRUCTURAL_ATOL = 1e-12
STRUCTURAL_RTOL = 1e-10

def pytest_addoption(parser: pytest.Parser) -> None:
    """Expose CLI flag to force a specific mathematics backend."""

    parser.addoption(
        "--math-backend",
        action="store",
        default=None,
        help="Force TNFR_MATH_BACKEND during the session (numpy, jax, torch).",
    )

def pytest_configure(config: pytest.Config) -> None:
    """Propagate backend selection from CLI or environment before tests import."""

    requested = config.getoption("math_backend")
    env_override = os.getenv("TNFR_TEST_MATH_BACKEND")
    choice = (requested or env_override)
    if not choice:
        return

    os.environ["TNFR_MATH_BACKEND"] = choice

    # Ensure stale caches do not override the requested backend.
    from tnfr.mathematics import backend as backend_module  # imported lazily

    backend_module._BACKEND_CACHE.clear()

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
