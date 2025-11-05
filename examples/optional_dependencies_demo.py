#!/usr/bin/env python3
"""Example demonstrating optional dependency handling in TNFR.

This example shows how TNFR gracefully handles missing optional dependencies
while providing informative error messages and type checking compatibility.

Installation
------------
# Minimal installation (core includes NumPy, NetworkX, Cachetools)
pip install tnfr

# With visualization support
pip install tnfr[viz-basic]

# With JAX backend
pip install tnfr[compute-jax]

# With PyTorch backend  
pip install tnfr[compute-torch]
"""

from __future__ import annotations

import sys


def demo_compat_layer() -> None:
    """Demonstrate the compatibility layer for optional dependencies."""
    print("=== TNFR Optional Dependency Compatibility Demo ===\n")

    # Check what's installed
    print("Checking installed packages:")
    try:
        import numpy

        print("✓ NumPy is installed:", numpy.__version__)
        numpy_available = True
    except ImportError:
        print("✗ NumPy is not installed")
        numpy_available = False

    try:
        import matplotlib

        print("✓ Matplotlib is installed:", matplotlib.__version__)
        mpl_available = True
    except ImportError:
        print("✗ Matplotlib is not installed")
        mpl_available = False

    try:
        import jsonschema

        print("✓ jsonschema is installed:", jsonschema.__version__)
        js_available = True
    except ImportError:
        print("✗ jsonschema is not installed")
        js_available = False

    print("\n" + "=" * 60 + "\n")

    # Demonstrate compat layer
    print("Using TNFR compatibility layer:")
    from tnfr.compat import (
        get_numpy_or_stub,
        get_matplotlib_or_stub,
        get_jsonschema_or_stub,
    )

    np = get_numpy_or_stub()
    print(f"get_numpy_or_stub() returned: {type(np).__name__}")

    mpl = get_matplotlib_or_stub()
    print(f"get_matplotlib_or_stub() returned: {type(mpl).__name__}")

    js = get_jsonschema_or_stub()
    print(f"get_jsonschema_or_stub() returned: {type(js).__name__}")

    print("\n" + "=" * 60 + "\n")

    # Demonstrate viz module fallback
    print("Testing viz module:")
    from tnfr import viz

    print(f"viz module exports: {viz.__all__}")

    if numpy_available and mpl_available:
        print("✓ Visualization functions are available")
        print("  Can call: viz.plot_coherence_matrix()")
        print("  Can call: viz.plot_phase_sync()")
        print("  Can call: viz.plot_spectrum_path()")
    else:
        print("✗ Visualization functions will raise ImportError when called")
        print("  To enable: pip install tnfr[viz]")

    print("\n" + "=" * 60 + "\n")

    # Demonstrate type checking compatibility
    print("Type checking compatibility:")
    print(
        "The compat layer allows code to type-check even without optional packages."
    )
    print("Example:")
    print("  from typing import TYPE_CHECKING")
    print("  if TYPE_CHECKING:")
    print("      import numpy as np")
    print("  else:")
    print("      from tnfr.compat import numpy_stub as np")
    print("")
    print("This pattern lets type checkers see numpy types while providing")
    print("runtime stubs that raise informative errors if used without numpy.")

    print("\n" + "=" * 60 + "\n")

    # Demonstrate graceful error handling
    print("Graceful error handling:")
    if not numpy_available:
        print("\nTrying to use numpy stub (will raise informative error):")
        from tnfr.compat import numpy_stub

        try:
            numpy_stub.array([1, 2, 3])
        except RuntimeError as e:
            print(f"✓ Caught expected error: {e}")

    if not mpl_available:
        print("\nTrying to use matplotlib stub (will raise informative error):")
        from tnfr.compat import matplotlib_stub

        try:
            matplotlib_stub.pyplot.subplots()
        except RuntimeError as e:
            print(f"✓ Caught expected error: {e}")

    print("\n" + "=" * 60 + "\n")
    print("Demo complete!")
    print(
        "\nTo install optional dependencies: pip install tnfr[numpy,viz,yaml,orjson]"
    )


if __name__ == "__main__":
    demo_compat_layer()
