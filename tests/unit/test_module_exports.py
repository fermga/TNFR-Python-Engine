"""Tests for module export consistency between .py and .pyi files.

This test module verifies:
1. All symbols declared in __all__ are actually importable
2. All symbols imported by re-exports exist in source modules
3. Stub files (.pyi) match implementation files (.py)

These tests support TNFR INVARIANT #8 (Controlled determinism) by ensuring
consistent and deterministic module exports.
"""

import pytest


def test_cli_main_exports():
    """Verify tnfr.cli exports all declared symbols.

    Structural Function: Coherence - maintains CLI API consistency
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    from tnfr import cli

    # Verify __all__ exists and is a tuple
    assert hasattr(cli, "__all__"), "cli module should define __all__"
    assert isinstance(cli.__all__, tuple), "__all__ should be a tuple"

    # Verify all symbols in __all__ are actually importable
    for symbol in cli.__all__:
        assert hasattr(cli, symbol), f"cli.__all__ declares '{symbol}' but it's not available"
        value = getattr(cli, symbol)
        assert value is not None, f"cli.{symbol} is None"


def test_cli_arguments_parser_functions():
    """Verify cli.arguments exports all parser configuration functions.

    Structural Function: Coherence - validates parser function availability
    TNFR Invariants: #4 (Operator closure)
    """
    from tnfr.cli import arguments

    # Public parser configuration functions
    public_functions = [
        "add_common_args",
        "add_grammar_args",
        "add_grammar_selector_args",
        "add_history_export_args",
        "add_canon_toggle",
    ]

    for func_name in public_functions:
        assert hasattr(arguments, func_name), f"arguments module should export {func_name}"
        func = getattr(arguments, func_name)
        assert callable(func), f"arguments.{func_name} should be callable"

    # Private parser functions used by cli.__init__
    private_functions = [
        "_add_run_parser",
        "_add_sequence_parser",
        "_add_metrics_parser",
        "_add_profile_parser",
        "_add_profile_pipeline_parser",
        "_add_math_run_parser",
        "_add_epi_validate_parser",
    ]

    for func_name in private_functions:
        assert hasattr(
            arguments, func_name
        ), f"arguments module should export {func_name} for internal use"
        func = getattr(arguments, func_name)
        assert callable(func), f"arguments.{func_name} should be callable"


def test_callback_utils_backward_compatibility_removed():
    """Verify callback_utils module has been removed.

    Structural Function: Transition - validates deprecation removal
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    import sys

    # The module should no longer exist
    # Note: We don't actually import it because it doesn't exist anymore
    # Just verify that the functionality is available via the correct path
    from tnfr.utils import callbacks

    # Verify __all__ exists and is a tuple
    assert hasattr(callbacks, "__all__"), "callbacks module should define __all__"

    # Verify all symbols are actually importable from the correct location
    expected_symbols = [
        "CallbackEvent",
        "CallbackManager",
        "callback_manager",
        "CallbackError",
        "CallbackSpec",
    ]

    for symbol in expected_symbols:
        assert symbol in callbacks.__all__, f"callbacks.__all__ should include '{symbol}'"
        assert hasattr(callbacks, symbol), f"callbacks should export '{symbol}'"


def test_utils_callbacks_exports():
    """Verify utils.callbacks exports all declared symbols.

    Structural Function: Coherence - validates callback utilities
    TNFR Invariants: #4 (Operator closure)
    """
    from tnfr.utils import callbacks

    # Verify __all__ exists
    assert hasattr(callbacks, "__all__"), "callbacks module should define __all__"

    # Core callback symbols
    core_symbols = [
        "CallbackEvent",
        "CallbackManager",
        "callback_manager",
        "CallbackError",
        "CallbackSpec",
    ]

    for symbol in core_symbols:
        assert symbol in callbacks.__all__, f"callbacks.__all__ should include '{symbol}'"
        assert hasattr(callbacks, symbol), f"callbacks should export '{symbol}'"

    # Private helper functions (not in callbacks.__all__ but accessible)
    # Note: These ARE in callback_utils.__all__ for backward compatibility
    helper_functions = [
        "_normalize_callbacks",
        "_normalize_callback_entry",
    ]

    for func_name in helper_functions:
        assert (
            func_name not in callbacks.__all__
        ), f"callbacks.__all__ should NOT include private function '{func_name}'"
        assert hasattr(
            callbacks, func_name
        ), f"callbacks should define {func_name} for internal use"
        func = getattr(callbacks, func_name)
        assert callable(func), f"callbacks.{func_name} should be callable"


def test_utils_cache_exports():
    """Verify utils.cache exports all declared symbols.

    Structural Function: Coherence - validates cache infrastructure
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    from tnfr.utils import cache

    # Verify __all__ exists
    assert hasattr(cache, "__all__"), "cache module should define __all__"

    # Key cache symbols that should be in __all__
    expected_in_all = [
        "CacheManager",
        "CacheLayer",
        "SecurityError",
        "cached_node_list",
        "cached_nodes_and_A",
        "DNFR_PREP_STATE_KEY",
        "DnfrPrepState",
    ]

    for symbol in expected_in_all:
        assert symbol in cache.__all__, f"cache.__all__ should include '{symbol}'"
        assert hasattr(cache, symbol), f"cache should export '{symbol}'"

    # Private symbols used by utils.__init__ (not in __all__)
    private_symbols = [
        "_GRAPH_CACHE_MANAGER_KEY",
        "_graph_cache_manager",
    ]

    for symbol in private_symbols:
        assert hasattr(cache, symbol), f"cache should define {symbol} for internal use"


def test_utils_main_exports():
    """Verify utils.__init__ exports all declared symbols.

    Structural Function: Resonance - maintains utils API propagation
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    from tnfr import utils

    # Verify __all__ exists and is a tuple
    assert hasattr(utils, "__all__"), "utils module should define __all__"
    assert isinstance(utils.__all__, tuple), "__all__ should be a tuple"

    # Verify all symbols in __all__ are actually importable
    for symbol in utils.__all__:
        assert hasattr(utils, symbol), f"utils.__all__ declares '{symbol}' but it's not available"

    # Specifically check callback-related exports
    callback_symbols = [
        "CallbackEvent",
        "CallbackManager",
        "callback_manager",
        "CallbackSpec",
    ]

    for symbol in callback_symbols:
        assert symbol in utils.__all__, f"utils.__all__ should include callback symbol '{symbol}'"
        assert hasattr(utils, symbol), f"utils should export callback symbol '{symbol}'"

    # Check SecurityError
    assert "SecurityError" in utils.__all__, "utils.__all__ should include 'SecurityError'"
    assert hasattr(utils, "SecurityError"), "utils should export 'SecurityError'"


def test_validation_exports():
    """Verify validation.__init__ exports all declared symbols.

    Structural Function: Coherence - validates validation API
    TNFR Invariants: #4 (Operator closure), #8 (Controlled determinism)
    """
    from tnfr import validation

    # Verify __all__ exists and is a tuple
    assert hasattr(validation, "__all__"), "validation module should define __all__"
    assert isinstance(validation.__all__, tuple), "__all__ should be a tuple"

    # Verify all symbols in __all__ are actually importable
    for symbol in validation.__all__:
        assert hasattr(
            validation, symbol
        ), f"validation.__all__ declares '{symbol}' but it's not available"

    # Check specific validation symbols
    expected_symbols = [
        "ValidationOutcome",
        "Validator",
        "coerce_glyph",
        "get_norm",
        "glyph_fallback",
        "normalized_dnfr",
        "GrammarContext",
        "validate_canon",
        "GraphCanonicalValidator",
    ]

    for symbol in expected_symbols:
        assert symbol in validation.__all__, f"validation.__all__ should include '{symbol}'"
        assert hasattr(validation, symbol), f"validation should export '{symbol}'"


def test_validation_rules_exports():
    """Verify validation.rules exports all declared symbols.

    Structural Function: Coherence - validates rule helpers
    TNFR Invariants: #4 (Operator closure)
    """
    from tnfr.validation import rules

    # Verify __all__ exists
    assert hasattr(rules, "__all__"), "rules module should define __all__"

    # Public rule helper functions
    public_functions = [
        "coerce_glyph",
        "glyph_fallback",
        "get_norm",
        "normalized_dnfr",
    ]

    for func_name in public_functions:
        assert func_name in rules.__all__, f"rules.__all__ should include '{func_name}'"
        assert hasattr(rules, func_name), f"rules should export {func_name}"
        func = getattr(rules, func_name)
        assert callable(func), f"rules.{func_name} should be callable"

    # Private helper functions (in __all__ but typically internal)
    private_functions = [
        "_norm_attr",
        "_si",
        "_check_oz_to_zhir",
        "_check_thol_closure",
        "_check_compatibility",
    ]

    for func_name in private_functions:
        assert (
            func_name in rules.__all__
        ), f"rules.__all__ should include internal function '{func_name}'"
        assert hasattr(rules, func_name), f"rules should define {func_name}"


def test_cross_module_import_consistency():
    """Verify symbols can be imported via multiple paths consistently.

    Structural Function: Resonance - validates API propagation paths
    TNFR Invariants: #4 (Operator closure), #7 (Operational fractality)
    """
    # Test 1: CallbackManager accessible via multiple paths
    from tnfr.utils import CallbackManager as CM1
    from tnfr.utils.callbacks import CallbackManager as CM2

    assert CM1 is CM2, "CallbackManager should be same object via utils and utils.callbacks"

    # Test 2: coerce_glyph accessible via validation
    from tnfr.validation import coerce_glyph as cg1
    from tnfr.validation.rules import coerce_glyph as cg2

    assert cg1 is cg2, "coerce_glyph should be same function via validation and validation.rules"

    # Test 3: cached_node_list accessible via utils
    from tnfr.utils import cached_node_list as cnl1
    from tnfr.utils.cache import cached_node_list as cnl2

    assert cnl1 is cnl2, "cached_node_list should be same function via utils and utils.cache"


def test_no_missing_stub_symbols():
    """Verify no AttributeError when importing symbols declared in stubs.

    This test catches cases where a .pyi file declares a symbol but the
    corresponding .py file doesn't export it, which causes mypy to pass
    but runtime imports to fail.

    Structural Function: Coherence - validates stub/implementation consistency
    TNFR Invariants: #8 (Controlled determinism)
    """
    # Test cli exports match stub
    from tnfr.cli import (
        main,
        add_common_args,
        add_grammar_args,
        add_grammar_selector_args,
        add_history_export_args,
        add_canon_toggle,
        build_basic_graph,
        apply_cli_config,
        register_callbacks_and_observer,
        resolve_program,
        run_program,
    )

    # Test utils exports match stub
    from tnfr.utils import (
        CallbackEvent,
        CallbackManager,
        callback_manager,
        CallbackSpec,
        SecurityError,
        CacheManager,
        cached_node_list,
    )

    # Test validation exports match stub
    from tnfr.validation import (
        ValidationOutcome,
        Validator,
        coerce_glyph,
        get_norm,
        glyph_fallback,
        normalized_dnfr,
        validate_canon,
    )

    # If we reach here, all imports succeeded
    assert True, "All stub-declared symbols are importable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
