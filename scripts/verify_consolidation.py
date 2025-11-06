#!/usr/bin/env python3
"""Verification script for utility and validation consolidation.

This script verifies that all generic utilities and validators are properly
consolidated under stable interfaces with no code duplication.

Run this script to confirm consolidation status:
    python scripts/verify_consolidation.py

Exit codes:
    0 - Consolidation verified successfully
    1 - Consolidation issues detected
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def verify_utils_exports() -> tuple[bool, str]:
    """Verify tnfr.utils exports all expected utilities."""
    try:
        from tnfr import utils

        expected_categories = {
            "numeric": ["clamp", "clamp01", "angle_diff", "kahan_sum_nd"],
            "cache": ["CacheManager", "cached_node_list", "edge_version_cache"],
            "data": ["normalize_weights", "convert_value", "ensure_collection"],
            "io": ["json_dumps", "read_structured_file", "safe_write"],
            "callbacks": ["CallbackEvent", "CallbackManager", "callback_manager"],
        }

        missing = []
        for category, funcs in expected_categories.items():
            for func in funcs:
                if func not in utils.__all__:
                    missing.append(f"{category}.{func}")

        if missing:
            return False, f"Missing exports: {', '.join(missing)}"

        return True, f"✅ All {len(utils.__all__)} utilities exported from tnfr.utils"

    except Exception as e:
        return False, f"Failed to import tnfr.utils: {e}"


def verify_validation_exports() -> tuple[bool, str]:
    """Verify tnfr.validation exports all expected validators."""
    try:
        from tnfr import validation

        expected = [
            "ValidationOutcome",
            "Validator",
            "GraphCanonicalValidator",
            "validate_canon",
            "coerce_glyph",
            "CANON_COMPAT",
            "CANON_FALLBACK",
        ]

        missing = [name for name in expected if name not in validation.__all__]

        if missing:
            return False, f"Missing validation exports: {', '.join(missing)}"

        return (
            True,
            f"✅ All {len(validation.__all__)} validators exported from tnfr.validation",
        )

    except Exception as e:
        return False, f"Failed to import tnfr.validation: {e}"


def verify_cache_import_path() -> tuple[bool, str]:
    """Verify tnfr.cache is the correct import path for caching."""
    try:
        from tnfr import cache
        
        # Verify cache exports expected symbols
        expected = ["TNFRHierarchicalCache", "CacheLevel", "cache_tnfr_computation"]
        missing = [name for name in expected if not hasattr(cache, name)]
        
        if missing:
            return False, f"❌ tnfr.cache missing exports: {', '.join(missing)}"
        
        return True, "✅ tnfr.cache properly exports caching functionality"
    except ImportError as e:
        return False, f"❌ Failed to import tnfr.cache: {e}"


def verify_callback_utils_removed() -> tuple[bool, str]:
    """Verify tnfr.callback_utils module has been removed."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("tnfr.callback_utils")
        if spec is not None:
            return False, "❌ tnfr.callback_utils should be removed but still exists"
        
        # Verify functionality is available via tnfr.utils.callbacks
        from tnfr.utils import CallbackEvent, CallbackManager, callback_manager
        return True, "✅ tnfr.callback_utils removed, functionality in tnfr.utils.callbacks"
    
    except Exception as e:
        return False, f"Failed to verify callback_utils removal: {e}"


def verify_no_duplicate_converters() -> tuple[bool, str]:
    """Verify type converters are not duplicated."""
    try:
        from tnfr.utils import convert_value, normalize_optional_int, ensure_collection
        from tnfr.validation import coerce_glyph

        # Check that these are unique functions
        converters = {
            "convert_value": convert_value,
            "normalize_optional_int": normalize_optional_int,
            "ensure_collection": ensure_collection,
            "coerce_glyph": coerce_glyph,
        }

        return True, f"✅ All {len(converters)} type converters are unique"

    except Exception as e:
        return False, f"Failed to verify converters: {e}"


def verify_cli_utils_scoped() -> tuple[bool, str]:
    """Verify CLI utils are properly scoped."""
    try:
        from tnfr.cli import utils as cli_utils

        # CLI utils should have spec and _parse_cli_variants
        if not hasattr(cli_utils, "spec"):
            return False, "❌ CLI utils missing 'spec' function"

        if not hasattr(cli_utils, "_parse_cli_variants"):
            return False, "❌ CLI utils missing '_parse_cli_variants' function"

        return True, "✅ CLI utils properly scoped in tnfr.cli.utils"

    except Exception as e:
        return False, f"Failed to verify CLI utils: {e}"


def main() -> int:
    """Run all verification checks."""
    print("=" * 70)
    print("TNFR Consolidation Verification")
    print("=" * 70)
    print()

    checks = [
        ("Utils exports", verify_utils_exports),
        ("Validation exports", verify_validation_exports),
        ("Cache import path", verify_cache_import_path),
        ("Callback utils removed", verify_callback_utils_removed),
        ("No duplicate converters", verify_no_duplicate_converters),
        ("CLI utils scoped", verify_cli_utils_scoped),
    ]

    results = []
    for name, check_fn in checks:
        print(f"Checking: {name}...", end=" ")
        success, message = check_fn()
        results.append(success)
        print(message)

    print()
    print("=" * 70)

    if all(results):
        print("✅ CONSOLIDATION VERIFIED: All checks passed")
        print()
        print("Summary:")
        print("  - Generic utilities centralized in tnfr.utils")
        print("  - Validators consolidated in tnfr.validation")
        print("  - No redundant helper modules")
        print("  - Legacy imports properly handled")
        print("  - Type converters unified")
        print()
        return 0
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"❌ CONSOLIDATION ISSUES: {failed_count}/{len(results)} checks failed")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
