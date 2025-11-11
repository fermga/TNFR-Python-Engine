"""
Test pytest and plugin compatibility.

This test suite verifies that pytest 8.x and all plugins work correctly together.
It serves as both a compatibility check and documentation of our testing setup.
"""

import sys

import pytest


class TestPytestVersion:
    """Verify pytest version is 8.x or higher."""

    def test_pytest_version_is_8_or_higher(self):
        """Verify we're using pytest 8.x or higher."""
        assert pytest.__version__.startswith("8."), (
            f"Expected pytest 8.x, but got {pytest.__version__}. "
            "This project requires pytest 8.x for compatibility."
        )

    def test_pytest_version_is_latest_8x(self):
        """Document the current pytest version."""
        major, minor, patch = pytest.__version__.split(".")[:3]
        assert int(major) == 8, f"Expected pytest 8.x, got {major}.x"
        # This test documents that we're on 8.4.x or higher
        assert int(minor) >= 4, (
            f"Expected pytest 8.4.x or higher for best compatibility, "
            f"got {major}.{minor}.{patch}"
        )


class TestPytestPluginsAvailable:
    """Verify all required pytest plugins are available."""

    def test_pytest_cov_available(self):
        """Verify pytest-cov plugin is loaded."""
        import pytest_cov

        assert pytest_cov is not None
        # pytest-cov 7.0+ is compatible with pytest 8.x
        assert hasattr(pytest_cov, "plugin")

    def test_pytest_timeout_available(self):
        """Verify pytest-timeout plugin is loaded."""
        import pytest_timeout

        assert pytest_timeout is not None

    def test_pytest_xdist_available(self):
        """Verify pytest-xdist plugin is loaded."""
        import xdist

        assert xdist is not None

    def test_pytest_benchmark_available(self):
        """Verify pytest-benchmark plugin is loaded."""
        import pytest_benchmark

        assert pytest_benchmark is not None
        # Verify version is 5.x or higher for pytest 8.x compatibility
        version = pytest_benchmark.__version__
        major = int(version.split(".")[0])
        assert major >= 5, f"Expected pytest-benchmark 5.x or higher, got {version}"

    def test_hypothesis_available(self):
        """Verify hypothesis is available."""
        import hypothesis

        assert hypothesis is not None
        # Hypothesis 6.x is compatible with pytest 8.x
        version = hypothesis.__version__
        major = int(version.split(".")[0])
        assert major >= 6, f"Expected hypothesis 6.x or higher, got {version}"

    def test_hypothesis_networkx_available(self):
        """Verify hypothesis-networkx is available."""
        import hypothesis_networkx

        assert hypothesis_networkx is not None


class TestPytestPluginsCompatibility:
    """Test that pytest plugins work correctly with pytest 8.x."""

    def test_pytest_cov_integration(self, tmp_path):
        """Test pytest-cov works with pytest 8.x."""
        # Create a simple test file
        test_file = tmp_path / "test_simple.py"
        test_file.write_text("def test_pass():\n    assert True\n", encoding="utf-8")

        # Run pytest with coverage
        result = pytest.main(
            [
                str(test_file),
                "--cov=tnfr",
                "--cov-report=term-missing",
                "-v",
            ]
        )
        # Should pass without errors
        assert result == 0

    @pytest.mark.timeout(5)
    def test_pytest_timeout_integration(self):
        """Test pytest-timeout works with pytest 8.x."""
        # This test should complete quickly and not timeout
        import time

        time.sleep(0.1)
        assert True

    def test_hypothesis_integration(self):
        """Test hypothesis works with pytest 8.x."""
        from hypothesis import given
        from hypothesis import strategies as st

        @given(st.integers())
        def test_hypothesis_example(x):
            assert isinstance(x, int)

        # Run the hypothesis test
        test_hypothesis_example()


class TestPytestConfiguration:
    """Test pytest configuration is correctly set up."""

    def test_pytest_config_loaded(self, pytestconfig):
        """Verify pytest configuration is loaded from pyproject.toml."""
        # Check that our custom markers are registered
        markers = pytestconfig.getini("markers")
        # Markers are returned as strings in pytest 8.x
        markers_str = " ".join(str(m) for m in markers)
        assert "slow" in markers_str
        assert "benchmarks" in markers_str
        assert "stress" in markers_str

    def test_pythonpath_includes_src(self):
        """Verify src directory is in Python path."""
        # This is configured in pyproject.toml
        import tnfr

        assert tnfr is not None
        # Verify we can import from the src directory
        assert hasattr(tnfr, "__version__")


class TestDeprecationWarnings:
    """Test that we're not seeing pytest-specific deprecation warnings."""

    def test_no_pytest_deprecation_warnings(self, recwarn):
        """Verify no pytest deprecation warnings during test execution."""
        # Run a simple assertion
        assert True

        # Check for pytest-specific deprecation warnings
        pytest_warnings = [
            w
            for w in recwarn.list
            if "pytest" in str(w.category).lower() and "deprecat" in str(w.message).lower()
        ]

        assert len(pytest_warnings) == 0, (
            f"Found pytest deprecation warnings: " f"{[str(w.message) for w in pytest_warnings]}"
        )


class TestPytestFeatures:
    """Test that pytest 8.x features work as expected."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, True),
            (2, True),
            (0, False),
            (-1, False),
        ],
    )
    def test_parametrize_works(self, value, expected):
        """Test that parametrize decorator works correctly."""
        result = value > 0
        assert result == expected

    def test_fixtures_work(self, tmp_path):
        """Test that pytest fixtures work correctly."""
        assert tmp_path.exists()
        assert tmp_path.is_dir()

    def test_approx_works(self):
        """Test that pytest.approx works correctly."""
        assert 0.1 + 0.2 == pytest.approx(0.3)

    @pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires Python 3.9+")
    def test_skipif_works(self):
        """Test that skipif marker works correctly."""
        assert sys.version_info >= (3, 9)

    @pytest.mark.xfail(reason="Example of expected failure")
    def test_xfail_works(self):
        """Test that xfail marker works correctly."""
        assert False  # This is expected to fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
