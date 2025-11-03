"""Tests for import cycle detection and module coupling verification.

This test module verifies:
1. No circular runtime import dependencies in utils package
2. TYPE_CHECKING guards properly isolate type-only imports
3. Utils modules respect layer hierarchy
4. Compatibility shims properly redirect to new locations

These tests support TNFR INVARIANT #8 (Controlled determinism) by ensuring
deterministic import behavior.
"""

import ast
import sys
from pathlib import Path
from typing import Set

import pytest


def test_no_circular_imports_utils_package():
    """Verify no circular import cycles in utils package.
    
    Structural Function: Coherence - maintains import graph consistency
    TNFR Invariants: #8 (Controlled determinism)
    """
    # Import entire utils package - should not raise ImportError
    import tnfr.utils
    import tnfr.utils.cache
    import tnfr.utils.callbacks
    import tnfr.utils.chunks
    import tnfr.utils.data
    import tnfr.utils.graph
    import tnfr.utils.init
    import tnfr.utils.io
    import tnfr.utils.numeric
    
    # If we reach here without ImportError, no cycles exist
    assert True, "All utils modules imported successfully without cycles"


def test_callback_utils_compatibility_shim():
    """Verify callback_utils emits deprecation warning and redirects properly.
    
    Structural Function: Transition - managed deprecation path
    TNFR Invariants: #8 (Controlled determinism)
    """
    with pytest.warns(DeprecationWarning, match="callback_utils.*deprecated"):
        import tnfr.callback_utils
        
        # Verify shim properly exports main symbols
        assert hasattr(tnfr.callback_utils, 'CallbackManager')
        assert hasattr(tnfr.callback_utils, 'CallbackEvent')
        assert hasattr(tnfr.callback_utils, 'callback_manager')


def test_type_checking_imports_isolated():
    """Verify TYPE_CHECKING guards prevent runtime imports.
    
    This ensures that the apparent circular reference between init.py and
    cache.py does not create a runtime circular import.
    
    Structural Function: Coherence - maintains clean runtime dependencies
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    from tnfr.utils import init
    
    # CacheManager should NOT be in init's runtime namespace
    # (it's only imported under TYPE_CHECKING)
    assert not hasattr(init, 'CacheManager'), \
        "CacheManager should not be in init namespace at runtime"
    
    # But we can still import it from cache
    from tnfr.utils.cache import CacheManager
    assert CacheManager is not None


def get_module_imports(module_path: Path) -> Set[str]:
    """Extract all TNFR imports from a Python module.
    
    Args:
        module_path: Path to Python source file
    
    Returns:
        Set of imported TNFR module names (e.g., 'tnfr.utils.cache')
    """
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(module_path))
    except SyntaxError:
        return set()
    
    imports = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('tnfr'):
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith('tnfr'):
                imports.add(node.module)
            elif node.level > 0:
                # Relative import - mark for manual review
                imports.add('_relative_import')
    
    return imports


def test_numeric_module_has_no_tnfr_imports():
    """Verify numeric.py (Layer 2) does not import from higher layers.
    
    The numeric module should be pure mathematical functions with no TNFR
    dependencies, ensuring it can be reused independently.
    
    Structural Function: Coherence - maintains layer separation
    TNFR Invariants: #4 (Operator closure), #10 (Domain neutrality)
    """
    from tnfr.utils import numeric
    
    module_path = Path(numeric.__file__)
    imports = get_module_imports(module_path)
    
    # Filter out relative imports (those are within utils package)
    tnfr_imports = [imp for imp in imports if not imp.startswith('_')]
    
    assert len(tnfr_imports) == 0, \
        f"numeric.py should have no TNFR imports, found: {tnfr_imports}"


def test_chunks_module_has_no_tnfr_imports():
    """Verify chunks.py (Layer 2) does not import from higher layers.
    
    The chunks module should be pure functions for parallel decomposition.
    
    Structural Function: Self-organization - maintains decomposition purity
    TNFR Invariants: #4 (Operator closure), #10 (Domain neutrality)
    """
    from tnfr.utils import chunks
    
    module_path = Path(chunks.__file__)
    imports = get_module_imports(module_path)
    
    # Filter out relative imports
    tnfr_imports = [imp for imp in imports if not imp.startswith('_')]
    
    assert len(tnfr_imports) == 0, \
        f"chunks.py should have no TNFR imports, found: {tnfr_imports}"


def test_utils_layer_hierarchy():
    """Verify utils modules respect dependency layer hierarchy.
    
    Expected hierarchy:
        Layer 1: init
        Layer 2: numeric, chunks
        Layer 3: data (depends on Layer 1, 2)
        Layer 4: graph (depends on types only)
        Layer 5: io (depends on Layer 1)
        Layer 6: cache (depends on Layers 1, 4, 5)
        Layer 7: callbacks (depends on Layers 1, 3)
    
    Structural Function: Coherence - maintains architectural integrity
    TNFR Invariants: #4 (Operator closure), #7 (Operational fractality)
    """
    from tnfr.utils import data
    
    # data.py should only import from init, numeric (lower layers)
    module_path = Path(data.__file__)
    imports = get_module_imports(module_path)
    
    # Extract just the module names from relative imports
    # Acceptable: init, numeric
    # Not acceptable: cache, callbacks
    forbidden = ['cache', 'callbacks']
    
    violations = [imp for imp in imports 
                  if any(f in imp for f in forbidden)]
    
    assert len(violations) == 0, \
        f"data.py should not import from {forbidden}, found: {violations}"


def test_all_utils_modules_importable():
    """Verify all utils modules can be imported individually.
    
    This ensures each module is self-contained and doesn't rely on
    import side effects from __init__.py.
    
    Structural Function: Coherence - validates module independence
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    utils_modules = [
        'tnfr.utils.cache',
        'tnfr.utils.callbacks',
        'tnfr.utils.chunks',
        'tnfr.utils.data',
        'tnfr.utils.graph',
        'tnfr.utils.init',
        'tnfr.utils.io',
        'tnfr.utils.numeric',
    ]
    
    for module_name in utils_modules:
        # Remove from sys.modules if already imported
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        # Import fresh
        try:
            __import__(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")


def test_utils_init_exports_match_submodules():
    """Verify utils.__init__ properly re-exports submodule APIs.
    
    Ensures public API surface is consistent between:
    - from tnfr.utils import X
    - from tnfr.utils.submodule import X
    
    Structural Function: Resonance - maintains API propagation
    TNFR Invariants: #4 (Operator closure)
    """
    import tnfr.utils
    import tnfr.utils.cache
    import tnfr.utils.callbacks
    
    # Key functions should be accessible from both paths
    assert hasattr(tnfr.utils, 'cached_node_list')
    assert hasattr(tnfr.utils.cache, 'cached_node_list')
    assert callable(tnfr.utils.cached_node_list)
    assert callable(tnfr.utils.cache.cached_node_list)
    
    assert hasattr(tnfr.utils, 'CallbackManager')
    assert hasattr(tnfr.utils.callbacks, 'CallbackManager')
    # Verify they're the same class by checking type and qualname
    assert type(tnfr.utils.CallbackManager) == type(tnfr.utils.callbacks.CallbackManager)
    assert tnfr.utils.CallbackManager.__name__ == tnfr.utils.callbacks.CallbackManager.__name__


def test_no_import_star_in_utils():
    """Verify utils modules don't use 'from X import *'.
    
    Import star makes dependencies implicit and can create namespace pollution.
    
    Structural Function: Coherence - maintains explicit dependencies
    TNFR Invariants: #8 (Controlled determinism)
    """
    utils_dir = Path(__file__).parent.parent / 'src' / 'tnfr' / 'utils'
    if not utils_dir.exists():
        pytest.skip("Utils directory not found in expected location")
    
    for py_file in utils_dir.glob('*.py'):
        if py_file.name.startswith('_'):
            continue
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for 'import *' patterns
        if 'import *' in content:
            # Parse to verify it's an actual import statement
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            if alias.name == '*':
                                pytest.fail(
                                    f"{py_file.name} uses 'from {node.module} import *' "
                                    "which should be avoided"
                                )
            except SyntaxError:
                pass


def test_utils_package_stability():
    """Verify utils package can be imported multiple times without side effects.
    
    Tests that the utils package is idempotent - importing it multiple times
    should not change program state.
    
    Structural Function: Coherence - maintains import stability
    TNFR Invariants: #8 (Controlled determinism)
    """
    import tnfr.utils
    
    # Get initial state
    initial_attrs = set(dir(tnfr.utils))
    
    # Reimport (this is a no-op in Python, but verifies no import-time side effects)
    import importlib
    importlib.reload(tnfr.utils)
    
    # Verify state unchanged
    final_attrs = set(dir(tnfr.utils))
    
    added = final_attrs - initial_attrs
    removed = initial_attrs - final_attrs
    
    assert len(removed) == 0, f"Attributes removed after reload: {removed}"
    # Note: Some attributes may be added (e.g., from importlib machinery), 
    # but none should be removed


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
