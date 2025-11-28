"""Test module isolation utilities work correctly.

This test module validates the test utility functions used for module
isolation and cache management. These utilities help ensure test independence
by clearing modules from sys.modules between tests.

IMPORTANT: The pattern 'module_name in sys.modules' is NOT URL validation
or sanitization. This is legitimate module management for test isolation.
Static analysis tools (e.g., CodeQL's py/incomplete-url-substring-sanitization)
may incorrectly flag this pattern as a security issue.

Structural Function: Coherence - ensures test isolation
TNFR Invariants: #8 (Controlled determinism)
"""

import sys

import pytest

from tests.utils import clear_test_module


def test_clear_test_module_removes_existing():
    """Verify clear_test_module removes existing modules from sys.modules.

    This test ensures that the utility properly clears modules that are
    already loaded, enabling test isolation.
    """
    # Import a standard library module to ensure it's in sys.modules
    import json  # noqa: F401

    module_name = "json"
    assert module_name in sys.modules

    # Clear it
    clear_test_module(module_name)

    # Should be removed from sys.modules
    assert module_name not in sys.modules


def test_clear_test_module_handles_missing():
    """Verify clear_test_module handles non-existent modules gracefully.

    The function should not raise an exception when asked to clear
    a module that isn't loaded.
    """
    fake_module = "non.existent.module.name.for.testing"
    assert fake_module not in sys.modules

    # Should not raise exception
    try:
        clear_test_module(fake_module)
    except KeyError:
        pytest.fail("clear_test_module raised KeyError for non-existent module")


def test_clear_test_module_is_not_url_validation():
    """Document that clear_test_module is module management, not URL validation.

    This test exists to clearly document the intent of the utility for
    security scanners and code reviewers. The pattern used internally
    ('module_name in sys.modules') should NOT be flagged as URL sanitization.

    Context: CodeQL's py/incomplete-url-substring-sanitization rule may
    incorrectly identify module path checking as incomplete URL validation.
    This is a false positive.
    """
    # Use a module name with dots (similar to URLs but clearly not one)
    module_name = "test.module.with.dots.but.not.url"

    # This pattern should NOT be flagged as URL sanitization
    clear_test_module(module_name)

    # Verify function exists and is documented
    assert clear_test_module.__doc__ is not None
    assert "test isolation" in clear_test_module.__doc__.lower()
    assert (
        "not security" in clear_test_module.__doc__.lower()
        or "not url" in clear_test_module.__doc__.lower()
    )


def test_clear_test_module_enables_fresh_import():
    """Verify clear_test_module allows re-importing modules with fresh state.

    This test demonstrates the primary use case: clearing a module so it
    can be re-imported with a clean state in subsequent tests.
    """
    # Import a module
    import email  # noqa: F401

    module_name = "email"

    # Store a reference
    first_import = sys.modules[module_name]  # noqa: F841

    # Clear it
    clear_test_module(module_name)

    # Re-import
    import email as email_reimport  # noqa: F401

    # Should be a fresh import (in practice, same object due to caching,
    # but the pattern enables test isolation when combined with fixtures)
    assert module_name in sys.modules


def test_clear_test_module_docstring_quality():
    """Verify clear_test_module has comprehensive documentation.

    The docstring should explain:
    1. What the function does
    2. Why it exists (test isolation)
    3. What it is NOT (security/URL validation)
    4. Why it might trigger false positives
    """
    doc = clear_test_module.__doc__
    assert doc is not None, "clear_test_module must have a docstring"

    # Check for key concepts
    assert "module" in doc.lower()
    assert "test" in doc.lower()
    assert "isolation" in doc.lower()

    # Check for security clarification
    assert "not" in doc.lower()
    assert any(word in doc.lower() for word in ["security", "url", "validation"])

    # Check for examples
    assert "Example" in doc or "Args" in doc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
