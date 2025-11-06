"""Test dependency management and graceful degradation.

This module validates:
- Core dependencies are available
- Optional backends degrade gracefully when missing
- Backend selection works correctly
- Version constraints are appropriate
"""

from __future__ import annotations

import sys
import pytest


def parse_version(version_string: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers for comparison."""
    return tuple(int(x) for x in version_string.split(".")[:3])


class TestCoreDependencies:
    """Test that core dependencies are installed and importable."""

    def test_numpy_is_core_dependency(self):
        """NumPy must be available as a core dependency."""
        try:
            import numpy

            assert numpy is not None
        except ImportError:
            pytest.fail("NumPy should be available as core dependency")

    def test_networkx_is_available(self):
        """NetworkX must be available as a core dependency."""
        try:
            import networkx

            assert networkx is not None
        except ImportError:
            pytest.fail("NetworkX should be available as core dependency")

    def test_cachetools_is_available(self):
        """Cachetools must be available as a core dependency."""
        try:
            import cachetools

            assert cachetools is not None
        except ImportError:
            pytest.fail("Cachetools should be available as core dependency")

    def test_core_module_imports_work(self):
        """Core TNFR modules should import without optional dependencies."""
        # These should work with just core dependencies
        import tnfr
        from tnfr import constants
        from tnfr import utils
        from tnfr.backends import get_backend, available_backends

        assert tnfr is not None
        assert constants is not None
        assert utils is not None
        assert get_backend is not None
        assert available_backends is not None


class TestVersionConstraints:
    """Test that dependencies have appropriate version constraints."""

    def test_numpy_version_in_supported_range(self):
        """NumPy version should be >= 1.24 and < 3.0."""
        try:
            import numpy

            np_version = parse_version(numpy.__version__)
            assert np_version >= (1, 24, 0)
            assert np_version < (3, 0, 0)
        except ImportError:
            pytest.skip("NumPy not installed")

    def test_networkx_version_in_supported_range(self):
        """NetworkX version should be >= 2.6 and < 4.0."""
        try:
            import networkx

            nx_version = parse_version(networkx.__version__)
            assert nx_version >= (2, 6, 0)
            assert nx_version < (4, 0, 0)
        except ImportError:
            pytest.skip("NetworkX not installed")

    def test_cachetools_version_in_supported_range(self):
        """Cachetools version should be >= 5.0 and < 7.0."""
        try:
            import cachetools

            ct_version = parse_version(cachetools.__version__)
            assert ct_version >= (5, 0, 0)
            assert ct_version < (7, 0, 0)
        except ImportError:
            pytest.skip("Cachetools not installed")


class TestBackendSystem:
    """Test backend detection and graceful degradation."""

    def test_numpy_backend_always_available(self):
        """NumPy backend must be available since numpy is now core."""
        from tnfr.backends import available_backends, get_backend

        backends = available_backends()
        assert "numpy" in backends

        backend = get_backend("numpy")
        assert backend.name == "numpy"

    def test_optional_backends_handle_missing_gracefully(self):
        """Optional backends (JAX, PyTorch) should not break if missing."""
        from tnfr.backends import available_backends

        backends = available_backends()
        # These are optional, should not cause import errors
        # Just check that the registry doesn't crash
        assert isinstance(backends, dict)
        assert len(backends) >= 1  # At least numpy should be there

    def test_backend_selection_defaults_to_numpy(self):
        """Default backend should be numpy when no preference set."""
        import os
        from tnfr.backends import get_backend

        # Clear any environment variable
        old_env = os.environ.pop("TNFR_BACKEND", None)
        try:
            backend = get_backend()
            assert backend.name == "numpy"
        finally:
            if old_env:
                os.environ["TNFR_BACKEND"] = old_env

    def test_get_backend_raises_for_unavailable_backend(self):
        """Requesting unavailable backend should raise clear error."""
        from tnfr.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend_xyz")


class TestOptionalDependencies:
    """Test optional dependency handling."""

    def test_jax_backend_availability(self):
        """JAX backend should only be available if JAX is installed."""
        from tnfr.backends import available_backends

        backends = available_backends()

        try:
            import jax

            # If JAX is installed, backend should be registered
            assert "jax" in backends
        except ImportError:
            # If JAX is not installed, backend should not crash import
            # but may or may not be in registry (depends on import attempt)
            pass

    def test_torch_backend_availability(self):
        """PyTorch backend should only be available if torch is installed."""
        from tnfr.backends import available_backends

        backends = available_backends()

        try:
            import torch

            # If torch is installed, backend should be registered
            assert "torch" in backends
        except ImportError:
            # If torch is not installed, backend should not crash import
            pass

    def test_matplotlib_optional_for_viz(self):
        """Visualization should work if matplotlib is installed."""
        try:
            import matplotlib

            # If matplotlib is available, viz should import
            from tnfr import viz

            assert viz is not None
        except ImportError:
            # If matplotlib is not installed, viz imports should still work
            # but operations may raise informative errors
            from tnfr import viz

            assert viz is not None


class TestCompatibilityLayer:
    """Test the compatibility layer for optional dependencies."""

    def test_get_numpy_returns_real_numpy(self):
        """get_numpy_or_stub should return real numpy when available."""
        from tnfr.compat import get_numpy_or_stub

        np = get_numpy_or_stub()
        # Since numpy is now core, should always get real numpy
        import numpy

        assert np is numpy

    def test_compat_layer_provides_stubs(self):
        """Compat layer should provide stubs for type checking."""
        from tnfr.compat import numpy_stub, matplotlib_stub, jsonschema_stub

        assert numpy_stub is not None
        assert matplotlib_stub is not None
        assert jsonschema_stub is not None
