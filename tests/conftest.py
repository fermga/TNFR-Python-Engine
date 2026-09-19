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
    choice = requested or env_override
    if not choice:
        return

    os.environ["TNFR_MATH_BACKEND"] = choice

    # Ensure stale caches do not override the requested backend.
    from tnfr.mathematics import backend as backend_module  # imported lazily

    backend_module._BACKEND_CACHE.clear()


@pytest.fixture(scope="session")
def structural_tolerances() -> dict[str, float]:
    """Return the shared numerical comparison tolerances used in tests."""

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


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset the selected mutable process caches used by these tests.

    Logging flags, callback limits, backend/cache managers, immutable-value
    checks, selector thresholds and RNG caches have explicit resets below.
    Import caches are intentionally retained; reset_cached_import owns opt-in
    clearing. Graph-owned callbacks, observers and integrators require fresh
    graph fixtures. This is not a claim to reset every possible global object.
    """
    # Reset state before test
    _reset_all_state()

    yield

    # Reset state after test
    _reset_all_state()


def _reset_all_state() -> None:
    """Apply the supported cache resets without importing retired subsystems."""

    # Reset logging configured flag (but don't call _reset_logging_state as it may cause issues)
    try:
        from tnfr.utils import init as init_module

        init_module._LOGGING_CONFIGURED = False
        init_module._NP_MISSING_LOGGED = False
        # Clear IMPORT_LOG to avoid test interference
        if hasattr(init_module, "IMPORT_LOG"):
            init_module.IMPORT_LOG.clear()
    except (ImportError, AttributeError):
        pass

    # Reset callback manager
    try:
        from tnfr.utils.callbacks import callback_manager

        # Reset error limit to default
        if hasattr(callback_manager, "_error_limit"):
            callback_manager._error_limit = 100
            callback_manager._error_limit_cache = 100
        # Note: Callbacks are stored in graph.graph['callbacks'], not in the manager
        # So tests creating fresh graphs will have clean callback state
    except ImportError:
        pass

    # Reset backend cache
    try:
        from tnfr.mathematics import backend as backend_module

        backend_module._BACKEND_CACHE.clear()
    except ImportError:
        pass

    # NOTE: We skip resetting import caches here because it can trigger
    # module re-imports that affect logging state, causing test isolation issues.
    # The reset_cached_import fixture provides this functionality for tests that need it.

    # Reset global cache managers
    try:
        from tnfr.utils import cache as cache_module

        # Reset global cache manager if it exists
        cache_module._GLOBAL_CACHE_MANAGER = None
        cache_module._GLOBAL_CACHE_LAYER_CONFIG.clear()
    except (ImportError, AttributeError):
        pass

    # Reset immutable cache
    try:
        from tnfr import immutable as immutable_module

        immutable_module._IMMUTABLE_CACHE.clear()
    except (ImportError, AttributeError):
        pass

    # Reset selector threshold cache
    try:
        from tnfr import selector as selector_module

        selector_module._SELECTOR_THRESHOLD_CACHE.clear()
    except (ImportError, AttributeError):
        pass

    # Reset RNG cache (seed_hash cache)
    try:
        from tnfr import rng as rng_module

        if hasattr(rng_module, "seed_hash") and hasattr(
            rng_module.seed_hash, "cache_clear"
        ):
            rng_module.seed_hash.cache_clear()
        # Reset RNG cache lock flag and cache
        rng_module._CACHE_LOCKED = False
        if hasattr(rng_module, "_seed_hash_cache"):
            rng_module._seed_hash_cache.clear()
        if hasattr(rng_module, "_RNG_CACHE_MANAGER"):
            # Clear the cache manager layers
            manager = rng_module._RNG_CACHE_MANAGER
            if hasattr(manager, "clear_all"):
                manager.clear_all()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def boundary_test_cases() -> dict[str, list[float]]:
    """Provide standard test cases for boundary testing.

    Returns standard EPI values near boundaries and in safe ranges
    for testing operator behavior at extremes.

    Returns
    -------
    dict[str, list[float]]
        Dictionary with 'upper_boundary', 'lower_boundary', and 'safe_values' keys
    """
    return {
        "upper_boundary": [0.95, 0.99, 0.999, 1.0 - 1e-10],
        "lower_boundary": [-0.95, -0.99, -0.999, -1.0 + 1e-10],
        "safe_values": [0.0, 0.5, -0.5, 0.8, -0.8],
    }


def assert_epi_in_bounds(
    epi_value: float, tolerance: float = 1e-9, abs_tol: float = 1e-12
) -> None:
    """Check the configured [-1, 1] EPI interval with numerical tolerance.

    This helper uses math.isclose to handle floating-point precision issues
    that may occur near boundaries.

    Parameters
    ----------
    epi_value : float
        The EPI value to check
    tolerance : float, default 1e-9
        Relative tolerance for boundary comparisons
    abs_tol : float, default 1e-12
        Absolute tolerance for boundary comparisons

    Raises
    ------
    AssertionError
        If EPI value is outside [-1.0, 1.0] beyond tolerance
    """
    import math

    # Check primary bounds
    if -1.0 <= epi_value <= 1.0:
        return

    # Check with tolerance for floating point precision
    if epi_value > 1.0:
        assert math.isclose(
            epi_value, 1.0, rel_tol=tolerance, abs_tol=abs_tol
        ), f"EPI {epi_value} exceeds upper boundary 1.0 beyond tolerance"
    elif epi_value < -1.0:
        assert math.isclose(
            epi_value, -1.0, rel_tol=tolerance, abs_tol=abs_tol
        ), f"EPI {epi_value} falls below lower boundary -1.0 beyond tolerance"
