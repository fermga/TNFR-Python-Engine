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

@pytest.fixture(autouse=True)
def reset_global_state(request):
    """Reset all global state between tests to ensure test isolation.
    
    This fixture resets:
    - Callback manager state
    - Backend caches  
    - Import caches
    - Global cache managers
    - Other module-level state
    
    Maintains TNFR canonical invariants (§3.8 - controlled determinism).
    """
    # Skip for specific tests that explicitly manage logging state
    skip_patterns = [
        'test_logging_utils_proxy_state',
        'test_configure_logging',
        'test_reset_logging_state',
    ]
    if any(pattern in request.node.name for pattern in skip_patterns):
        yield
        return
    
    # Reset state before test
    _reset_all_state()
    
    yield
    
    # Reset state after test
    _reset_all_state()

def _reset_all_state() -> None:
    """Helper to reset all global state."""
    
    # Reset logging configured flag (but don't call _reset_logging_state as it may cause issues)
    try:
        from tnfr.utils import init as init_module
        init_module._LOGGING_CONFIGURED = False
        init_module._NP_MISSING_LOGGED = False
    except (ImportError, AttributeError):
        pass
    
    # Reset callback manager
    try:
        from tnfr.utils.callbacks import callback_manager
        # Clear any registered callbacks on graphs by resetting the manager's internal state
        # The callback_manager is a singleton, so we need to be careful
        # We'll rely on tests to create fresh graphs
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
        if hasattr(rng_module, 'seed_hash') and hasattr(rng_module.seed_hash, 'cache_clear'):
            rng_module.seed_hash.cache_clear()
        # Reset RNG cache lock flag
        rng_module._CACHE_LOCKED = False
    except (ImportError, AttributeError):
        pass
    
    # Reset functools lru_caches
    try:
        from tnfr.utils import cache as cache_module
        if hasattr(cache_module, '_lru_cache_wrapper') and hasattr(cache_module._lru_cache_wrapper, 'cache_clear'):
            cache_module._lru_cache_wrapper.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset alias mapping cache
    try:
        from tnfr import alias as alias_module
        if hasattr(alias_module, '_to_canonical_epi') and hasattr(alias_module._to_canonical_epi, 'cache_clear'):
            alias_module._to_canonical_epi.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset gamma cache
    try:
        from tnfr import gamma as gamma_module
        if hasattr(gamma_module, '_get_builtin_gamma') and hasattr(gamma_module._get_builtin_gamma, 'cache_clear'):
            gamma_module._get_builtin_gamma.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset validation rules caches
    try:
        from tnfr.validation import rules as rules_module
        if hasattr(rules_module, '_get_glyph_name_lookup') and hasattr(rules_module._get_glyph_name_lookup, 'cache_clear'):
            rules_module._get_glyph_name_lookup.cache_clear()
        if hasattr(rules_module, '_get_glyph_function_map') and hasattr(rules_module._get_glyph_function_map, 'cache_clear'):
            rules_module._get_glyph_function_map.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset operator grammar cache (remesh cooldown cache)
    try:
        from tnfr.operators import remesh as remesh_module
        if hasattr(remesh_module, '_get_remesh_cooldown_default') and hasattr(remesh_module._get_remesh_cooldown_default, 'cache_clear'):
            remesh_module._get_remesh_cooldown_default.cache_clear()
    except (ImportError, AttributeError):
        pass
