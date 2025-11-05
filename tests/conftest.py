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
        # Clear IMPORT_LOG to avoid test interference
        if hasattr(init_module, 'IMPORT_LOG'):
            init_module.IMPORT_LOG.clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset callback manager
    try:
        from tnfr.utils.callbacks import callback_manager
        # Reset error limit to default
        if hasattr(callback_manager, '_error_limit'):
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
        if hasattr(rng_module, 'seed_hash') and hasattr(rng_module.seed_hash, 'cache_clear'):
            rng_module.seed_hash.cache_clear()
        # Reset RNG cache lock flag and cache
        rng_module._CACHE_LOCKED = False
        if hasattr(rng_module, '_seed_hash_cache'):
            rng_module._seed_hash_cache.clear()
        if hasattr(rng_module, '_RNG_CACHE_MANAGER'):
            # Clear the cache manager layers
            manager = rng_module._RNG_CACHE_MANAGER
            if hasattr(manager, 'clear_all'):
                manager.clear_all()
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
        # NOTE: Don't clear _functional_translators and _structural_tables caches
        # as they contain static lookup tables that should persist across tests
    except (ImportError, AttributeError):
        pass
    
    # Reset operator grammar cache (remesh cooldown cache)
    try:
        from tnfr.operators import remesh as remesh_module
        if hasattr(remesh_module, '_get_remesh_cooldown_default') and hasattr(remesh_module._get_remesh_cooldown_default, 'cache_clear'):
            remesh_module._get_remesh_cooldown_default.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset dynamics runtime cache
    try:
        from tnfr.dynamics import runtime as runtime_module
        # Clear any cached integrators or state
        if hasattr(runtime_module, '_INTEGRATOR_CACHE'):
            runtime_module._INTEGRATOR_CACHE.clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset metrics module caches
    try:
        from tnfr import metrics as metrics_module
        # Clear any metrics computation caches
        if hasattr(metrics_module, 'compute_sense_index') and hasattr(metrics_module.compute_sense_index, 'cache_clear'):
            metrics_module.compute_sense_index.cache_clear()
        if hasattr(metrics_module, 'compute_coherence') and hasattr(metrics_module.compute_coherence, 'cache_clear'):
            metrics_module.compute_coherence.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset observers state
    try:
        from tnfr import observers as observers_module
        # Observers register themselves on graphs, so we can't easily clear them globally
        # Tests should create fresh graphs for proper isolation
    except (ImportError, AttributeError):
        pass
    
    # Reset parallel execution state
    try:
        from tnfr.dynamics import parallel as parallel_module
        # Clear any executor caches or state
        if hasattr(parallel_module, '_EXECUTOR_CACHE'):
            parallel_module._EXECUTOR_CACHE.clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset glyph selector caches
    try:
        from tnfr import selector as selector_module
        if hasattr(selector_module, 'select_glyph') and hasattr(selector_module.select_glyph, 'cache_clear'):
            selector_module.select_glyph.cache_clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset configuration state
    try:
        from tnfr import secure_config as config_module
        # Reset any loaded configurations
        if hasattr(config_module, '_LOADED_CONFIGS'):
            config_module._LOADED_CONFIGS.clear()
    except (ImportError, AttributeError):
        pass
    
    # Reset validation service state
    try:
        from tnfr.validation import validator as validator_module
        # Clear any validator caches
        if hasattr(validator_module, '_VALIDATOR_CACHE'):
            validator_module._VALIDATOR_CACHE.clear()
    except (ImportError, AttributeError):
        pass
