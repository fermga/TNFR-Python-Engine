"""Test script to verify circular import fix in validation module.

This test reproduces the exact import chain that was causing the circular
import error in CI/CD.

Expected behavior: All imports should succeed without ImportError.
"""

import sys
import traceback


def test_import_chain():
    """Test the complete import chain that was failing."""
    print("Testing TNFR import chain...")
    print("=" * 60)
    
    try:
        # Step 1: Import main package
        print("1. Importing tnfr...")
        import tnfr
        print(f"   ✅ tnfr v{tnfr.__version__}")
        
        # Step 2: Import dynamics (which imports metrics)
        print("2. Importing tnfr.dynamics...")
        from tnfr.dynamics import step
        print(f"   ✅ tnfr.dynamics.step: {step.__name__}")
        
        # Step 3: Import metrics (which imports observers)
        print("3. Importing tnfr.metrics...")
        from tnfr.metrics.sense_index import compute_Si
        print(f"   ✅ tnfr.metrics.sense_index.compute_Si: {compute_Si.__name__}")
        
        # Step 4: Import observers (which imports validation)
        print("4. Importing tnfr.observers...")
        from tnfr import observers
        print(f"   ✅ tnfr.observers")
        
        # Step 5: Import validation (which imports runtime)
        print("5. Importing tnfr.validation...")
        from tnfr.validation import validate_window
        print(f"   ✅ tnfr.validation.validate_window: {validate_window.__name__}")
        
        # Step 6: Import runtime (previously caused circular import)
        print("6. Importing tnfr.validation.runtime...")
        from tnfr.validation.runtime import (
            GraphCanonicalValidator,
            apply_canonical_clamps,
            validate_canon,
        )
        print(f"   ✅ GraphCanonicalValidator: {GraphCanonicalValidator.__name__}")
        print(f"   ✅ apply_canonical_clamps: {apply_canonical_clamps.__name__}")
        print(f"   ✅ validate_canon: {validate_canon.__name__}")
        
        # Step 7: Verify base module
        print("7. Importing tnfr.validation.base...")
        from tnfr.validation.base import ValidationOutcome, Validator
        print(f"   ✅ ValidationOutcome: {ValidationOutcome.__name__}")
        print(f"   ✅ Validator: {Validator.__name__}")
        
        print("=" * 60)
        print("🎉 SUCCESS: All imports completed without circular dependency!")
        print()
        print("The circular import issue has been resolved by:")
        print("  - Moving ValidationOutcome and Validator to base.py")
        print("  - Updating runtime.py to import from .base instead of .")
        print("  - Updating __init__.py to re-export from .base")
        return True
        
    except ImportError as e:
        print("=" * 60)
        print("❌ FAILURE: Circular import detected!")
        print()
        print(f"Error: {e}")
        print()
        print("Traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_import_chain()
    sys.exit(0 if success else 1)
