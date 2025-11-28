"""Tests for migration examples to ensure they run without errors."""

import subprocess
import sys
from pathlib import Path


class TestMigrationExamples:
    """Test that all migration examples run successfully."""

    EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "migration"

    def test_before_after_comparison_runs(self):
        """Test before_after_comparison.py runs without errors."""
        example_path = self.EXAMPLES_DIR / "before_after_comparison.py"
        assert example_path.exists(), f"Example not found: {example_path}"

        result = subprocess.run(
            [sys.executable, str(example_path)], capture_output=True, text=True, timeout=30
        )

        # Should complete successfully
        assert result.returncode == 0, f"Example failed:\n{result.stderr}"

        # Should contain expected output
        assert "GRAMMAR 2.0 MIGRATION" in result.stdout
        assert "BEFORE" in result.stdout
        assert "AFTER" in result.stdout

    def test_health_optimization_tutorial_runs(self):
        """Test health_optimization_tutorial.py runs without errors."""
        example_path = self.EXAMPLES_DIR / "health_optimization_tutorial.py"
        assert example_path.exists(), f"Example not found: {example_path}"

        result = subprocess.run(
            [sys.executable, str(example_path)], capture_output=True, text=True, timeout=30
        )

        assert result.returncode == 0, f"Example failed:\n{result.stderr}"
        assert "HEALTH OPTIMIZATION TUTORIAL" in result.stdout
        assert "LESSON" in result.stdout

    def test_pattern_upgrade_examples_runs(self):
        """Test pattern_upgrade_examples.py runs without errors."""
        example_path = self.EXAMPLES_DIR / "pattern_upgrade_examples.py"
        assert example_path.exists(), f"Example not found: {example_path}"

        result = subprocess.run(
            [sys.executable, str(example_path)], capture_output=True, text=True, timeout=30
        )

        assert result.returncode == 0, f"Example failed:\n{result.stderr}"
        assert "PATTERN UPGRADE EXAMPLES" in result.stdout
        assert "Evolution:" in result.stdout or "PATTERN COMPARISON" in result.stdout

    def test_regenerative_cycles_intro_runs(self):
        """Test regenerative_cycles_intro.py runs without errors."""
        example_path = self.EXAMPLES_DIR / "regenerative_cycles_intro.py"
        assert example_path.exists(), f"Example not found: {example_path}"

        result = subprocess.run(
            [sys.executable, str(example_path)], capture_output=True, text=True, timeout=30
        )

        assert result.returncode == 0, f"Example failed:\n{result.stderr}"
        assert "REGENERATIVE CYCLES" in result.stdout
        assert "Example" in result.stdout

    def test_all_examples_have_main(self):
        """Test that all example files have a main function."""
        for example_file in self.EXAMPLES_DIR.glob("*.py"):
            if example_file.name.startswith("__"):
                continue

            content = example_file.read_text()
            assert "def main():" in content, f"{example_file.name} missing main()"
            assert (
                'if __name__ == "__main__":' in content
            ), f"{example_file.name} missing __main__ guard"
