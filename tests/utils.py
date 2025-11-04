import sys

import tnfr.utils.io as json_utils


def clear_orjson_cache() -> None:
    """Clear cached :mod:`orjson` module."""
    cache_clear = getattr(json_utils.cached_import, "cache_clear", None)
    if cache_clear:
        cache_clear()
    clear_warns = getattr(json_utils, "clear_orjson_param_warnings", None)
    if clear_warns:
        clear_warns()


def clear_test_module(module_name: str) -> None:
    """Clear module from sys.modules for test isolation.

    This function removes a module from the Python module cache to enable
    test isolation and force re-import. This is NOT URL validation or
    sanitization - it is legitimate module management for testing purposes.

    Args:
        module_name: Full module path (e.g., 'tnfr.utils.io')

    Note:
        This is for test isolation, not security validation. The pattern
        'module_name in sys.modules' may trigger false positives in static
        analysis tools (like CodeQL's py/incomplete-url-substring-sanitization)
        that mistake module path checking for URL validation.

    Example:
        >>> clear_test_module('tnfr.utils.io')
        >>> import tnfr.utils.io  # Fresh import
    """
    if module_name in sys.modules:
        del sys.modules[module_name]
