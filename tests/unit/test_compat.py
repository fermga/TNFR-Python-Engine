"""Tests for optional dependency compatibility layer.

These tests validate that the compat module provides appropriate stubs and
fallbacks for optional dependencies, maintaining TNFR semantic clarity when
packages are not installed.
"""

import pytest


class TestCompatModule:
    """Test the compat module structure and exports."""

    def test_compat_module_exists(self) -> None:
        """Verify compat module is importable."""
        from tnfr import compat

        assert compat is not None

    def test_compat_exports(self) -> None:
        """Verify compat module exports expected helpers."""
        from tnfr import compat

        assert hasattr(compat, "get_numpy_or_stub")
        assert hasattr(compat, "get_matplotlib_or_stub")
        assert hasattr(compat, "get_jsonschema_or_stub")
        assert hasattr(compat, "numpy_stub")
        assert hasattr(compat, "matplotlib_stub")
        assert hasattr(compat, "jsonschema_stub")


class TestNumpyStub:
    """Test numpy stub provides appropriate fallbacks."""

    def test_numpy_stub_importable(self) -> None:
        """Verify numpy_stub module is importable."""
        from tnfr.compat import numpy_stub

        assert numpy_stub is not None

    def test_numpy_stub_has_types(self) -> None:
        """Verify numpy_stub provides type compatibility."""
        from tnfr.compat import numpy_stub

        # These should exist for type checking
        assert hasattr(numpy_stub, "ndarray")
        assert hasattr(numpy_stub, "float64")
        assert hasattr(numpy_stub, "complex128")
        assert hasattr(numpy_stub, "dtype")

    def test_numpy_stub_operations_raise(self) -> None:
        """Verify numpy_stub operations raise informative errors."""
        from tnfr.compat import numpy_stub

        with pytest.raises(RuntimeError, match="numpy is not installed"):
            numpy_stub.array([1, 2, 3])

        with pytest.raises(RuntimeError, match="numpy is not installed"):
            numpy_stub.zeros(10)

    def test_numpy_stub_constants(self) -> None:
        """Verify numpy_stub provides mathematical constants."""
        from tnfr.compat import numpy_stub

        assert hasattr(numpy_stub, "pi")
        assert isinstance(numpy_stub.pi, float)


class TestMatplotlibStub:
    """Test matplotlib stub provides appropriate fallbacks."""

    def test_matplotlib_stub_importable(self) -> None:
        """Verify matplotlib_stub module is importable."""
        from tnfr.compat import matplotlib_stub

        assert matplotlib_stub is not None

    def test_matplotlib_stub_has_modules(self) -> None:
        """Verify matplotlib_stub provides module structure."""
        from tnfr.compat import matplotlib_stub

        assert hasattr(matplotlib_stub, "pyplot")
        assert hasattr(matplotlib_stub, "axes")
        assert hasattr(matplotlib_stub, "figure")

    def test_matplotlib_stub_has_types(self) -> None:
        """Verify matplotlib_stub provides type compatibility."""
        from tnfr.compat import matplotlib_stub

        assert hasattr(matplotlib_stub, "Axes")
        assert hasattr(matplotlib_stub, "Figure")

    def test_matplotlib_stub_operations_raise(self) -> None:
        """Verify matplotlib_stub operations raise informative errors."""
        from tnfr.compat import matplotlib_stub

        with pytest.raises(RuntimeError, match="matplotlib is not installed"):
            matplotlib_stub.pyplot.subplots()

        with pytest.raises(RuntimeError, match="matplotlib is not installed"):
            matplotlib_stub.Figure()


class TestJsonschemaStub:
    """Test jsonschema stub provides appropriate fallbacks."""

    def test_jsonschema_stub_importable(self) -> None:
        """Verify jsonschema_stub module is importable."""
        from tnfr.compat import jsonschema_stub

        assert jsonschema_stub is not None

    def test_jsonschema_stub_has_validator(self) -> None:
        """Verify jsonschema_stub provides validator types."""
        from tnfr.compat import jsonschema_stub

        assert hasattr(jsonschema_stub, "Draft7Validator")
        assert hasattr(jsonschema_stub, "exceptions")

    def test_jsonschema_stub_has_exceptions(self) -> None:
        """Verify jsonschema_stub provides exception types."""
        from tnfr.compat import jsonschema_stub

        assert hasattr(jsonschema_stub.exceptions, "SchemaError")
        assert hasattr(jsonschema_stub.exceptions, "ValidationError")

    def test_jsonschema_stub_operations_raise(self) -> None:
        """Verify jsonschema_stub operations raise informative errors."""
        from tnfr.compat import jsonschema_stub

        with pytest.raises(RuntimeError, match="jsonschema is not installed"):
            jsonschema_stub.Draft7Validator({})


class TestCompatHelpers:
    """Test compat helper functions."""

    def test_get_numpy_or_stub_with_numpy(self) -> None:
        """Verify get_numpy_or_stub returns numpy when available."""
        from tnfr.compat import get_numpy_or_stub

        np = get_numpy_or_stub()
        # Should return either real numpy or stub depending on availability
        # We just verify it returns something and doesn't raise
        assert np is not None
        # Check if it's the real numpy by checking for a unique attribute
        has_real_numpy = hasattr(np, "__version__") and hasattr(np, "ndarray")
        # It's either the real module or the stub
        assert has_real_numpy or hasattr(np, "_NotInstalledError")

    def test_get_matplotlib_or_stub(self) -> None:
        """Verify get_matplotlib_or_stub returns appropriate module."""
        from tnfr.compat import get_matplotlib_or_stub

        mpl = get_matplotlib_or_stub()
        # Should return either real matplotlib or stub
        assert mpl is not None

    def test_get_jsonschema_or_stub(self) -> None:
        """Verify get_jsonschema_or_stub returns appropriate module."""
        from tnfr.compat import get_jsonschema_or_stub

        js = get_jsonschema_or_stub()
        # Should return either real jsonschema or stub
        assert js is not None


class TestVizFallback:
    """Test viz module fallback behavior."""

    def test_viz_imports_with_deps(self) -> None:
        """Verify viz module imports when dependencies are available."""
        # This test only makes sense if matplotlib is installed
        pytest.importorskip("matplotlib")
        pytest.importorskip("numpy")

        from tnfr import viz

        assert hasattr(viz, "plot_coherence_matrix")
        assert hasattr(viz, "plot_phase_sync")
        assert hasattr(viz, "plot_spectrum_path")

    def test_viz_has_plot_functions(self) -> None:
        """Verify viz module exports plotting functions."""
        from tnfr import viz

        # Functions should be defined
        assert hasattr(viz, "plot_coherence_matrix")
        assert hasattr(viz, "plot_phase_sync")
        assert hasattr(viz, "plot_spectrum_path")

        # If matplotlib is not available, functions will raise ImportError when called
        # but they should still be defined in the module


class TestOperatorGrammarFallback:
    """Test operator grammar module handles missing jsonschema."""

    def test_grammar_module_imports(self) -> None:
        """Verify grammar module imports without jsonschema."""
        # The grammar module should import successfully whether or not
        # jsonschema is installed, thanks to try/except handling
        from tnfr.operators import grammar

        assert grammar is not None

    def test_grammar_has_validator_types(self) -> None:
        """Verify grammar module defines validator interfaces."""
        from tnfr.operators import grammar

        # These should be available for type checking
        assert hasattr(grammar, "validate_sequence")
        assert hasattr(grammar, "parse_sequence")


class TestTypesModuleWithoutNumpy:
    """Test types module handles missing numpy gracefully."""

    def test_types_module_imports(self) -> None:
        """Verify types module imports without numpy."""
        from tnfr import types

        assert types is not None

    def test_types_module_has_protocols(self) -> None:
        """Verify types module provides TNFR protocols."""
        from tnfr import types

        # Core TNFR types should always be available
        assert hasattr(types, "TNFRGraph")
        assert hasattr(types, "NodeId")
