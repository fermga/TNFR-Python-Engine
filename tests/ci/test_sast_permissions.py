"""Tests for CI infrastructure - virtual environment permission fixes.

This test suite validates that the scripts/fix_venv_perms.sh script correctly
handles permission issues with virtual environment binaries in CI environments.

These tests ensure operational invariants for CI tooling without touching
core TNFR structural code.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

class TestVenvPermissionFix:
    """Test suite for virtual environment permission fixing script."""

    @pytest.fixture
    def temp_venv_structure(self, tmp_path: Path) -> Path:
        """Create a temporary venv-like directory structure.

        Args:
            tmp_path: pytest tmp_path fixture

        Returns:
            Path to the temporary venv directory
        """
        venv_dir = tmp_path / "test_venv"
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir(parents=True)

        # Create mock executables
        mock_executables = ["python", "python3", "pip", "pip3", "semgrep", "bandit"]
        for exe_name in mock_executables:
            exe_path = bin_dir / exe_name
            exe_path.write_text(f"#!/bin/sh\n# Mock {exe_name}\necho 'mock'\n")

            # Remove execute permissions to simulate artifact download issue
            exe_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # rw- --- ---

        return venv_dir

    @pytest.fixture
    def fix_script_path(self) -> Path:
        """Get path to the fix_venv_perms.sh script.

        Returns:
            Path to the script
        """
        # Repository root is 2 levels up from tests/ci/
        repo_root = Path(__file__).parent.parent.parent
        script_path = repo_root / "scripts" / "fix_venv_perms.sh"

        if not script_path.exists():
            pytest.skip(f"Script not found: {script_path}")

        return script_path

    def test_script_exists_and_executable(self, fix_script_path: Path) -> None:
        """Verify the fix script exists and is executable."""
        assert fix_script_path.exists(), "fix_venv_perms.sh must exist"
        assert os.access(
            fix_script_path, os.X_OK
        ), "fix_venv_perms.sh must be executable"

    def test_fix_permissions_on_mock_executables(
        self, temp_venv_structure: Path, fix_script_path: Path
    ) -> None:
        """Test that the script fixes permissions on non-executable files.

        This test:
        1. Creates mock executables without execute permissions
        2. Runs the fix script
        3. Asserts all files become executable
        """
        bin_dir = temp_venv_structure / "bin"

        # Verify files are not executable before fix
        python_path = bin_dir / "python"
        semgrep_path = bin_dir / "semgrep"

        assert python_path.exists(), "Mock python must exist"
        assert not os.access(python_path, os.X_OK), "Mock python should not be executable initially"
        assert semgrep_path.exists(), "Mock semgrep must exist"
        assert not os.access(semgrep_path, os.X_OK), "Mock semgrep should not be executable initially"

        # Run the fix script
        result = subprocess.run(
            [str(fix_script_path), str(temp_venv_structure)],
            capture_output=True,
            text=True,
            check=False,
        )

        # Verify script succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "Done" in result.stdout, "Script should report completion"

        # Verify files are now executable
        assert os.access(python_path, os.X_OK), "Mock python should be executable after fix"
        assert os.access(semgrep_path, os.X_OK), "Mock semgrep should be executable after fix"

        # Verify all files in bin are executable
        for exe_path in bin_dir.iterdir():
            if exe_path.is_file():
                assert os.access(
                    exe_path, os.X_OK
                ), f"{exe_path.name} should be executable"

    def test_script_idempotent(
        self, temp_venv_structure: Path, fix_script_path: Path
    ) -> None:
        """Test that running the script multiple times is safe (idempotent).

        This test:
        1. Runs the fix script once
        2. Runs it again
        3. Verifies no errors and files remain executable
        """
        # Run twice
        result1 = subprocess.run(
            [str(fix_script_path), str(temp_venv_structure)],
            capture_output=True,
            text=True,
            check=False,
        )
        result2 = subprocess.run(
            [str(fix_script_path), str(temp_venv_structure)],
            capture_output=True,
            text=True,
            check=False,
        )

        # Both runs should succeed
        assert result1.returncode == 0, "First run should succeed"
        assert result2.returncode == 0, "Second run should succeed (idempotent)"

        # Verify files are still executable
        bin_dir = temp_venv_structure / "bin"
        python_path = bin_dir / "python"
        assert os.access(python_path, os.X_OK), "Files should remain executable"

    def test_script_handles_missing_venv(self, tmp_path: Path, fix_script_path: Path) -> None:
        """Test that the script handles missing venv directory gracefully.

        This test:
        1. Passes a non-existent directory to the script
        2. Verifies it returns an error code
        3. Verifies it doesn't crash
        """
        nonexistent_venv = tmp_path / "nonexistent"

        result = subprocess.run(
            [str(fix_script_path), str(nonexistent_venv)],
            capture_output=True,
            text=True,
            check=False,
        )

        # Should fail but not crash
        assert result.returncode != 0, "Script should fail for missing venv"
        output_lower = result.stdout.lower()
        assert "error" in output_lower or "not found" in output_lower, "Should report error"

    def test_script_logs_actions(
        self, temp_venv_structure: Path, fix_script_path: Path
    ) -> None:
        """Test that the script logs its actions for debugging.

        This test:
        1. Runs the fix script
        2. Verifies it produces informative log output
        """
        result = subprocess.run(
            [str(fix_script_path), str(temp_venv_structure)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, "Script should succeed"

        # Check for expected log messages
        output = result.stdout
        assert "Checking virtual environment" in output, "Should log checking message"
        assert "file(s) needing execute permissions" in output, "Should report count"
        assert "python is executable" in output or "python3 is executable" in output, "Should verify key executables"
        assert "Done" in output, "Should report completion"

    def test_script_verifies_key_executables(
        self, temp_venv_structure: Path, fix_script_path: Path
    ) -> None:
        """Test that the script verifies key executables after fixing permissions.

        This test:
        1. Creates a venv with mock executables
        2. Runs the fix script
        3. Verifies the script checks for python, pip, etc.
        """
        result = subprocess.run(
            [str(fix_script_path), str(temp_venv_structure)],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, "Script should succeed"

        output = result.stdout
        # Should verify key executables
        assert "python" in output.lower(), "Should check python"
        assert "executable" in output.lower(), "Should report executable status"
