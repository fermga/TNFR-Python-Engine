"""Tests for path traversal vulnerability prevention.

This module tests the security utilities that prevent path traversal attacks
while maintaining TNFR structural coherence.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tnfr.security import (
    PathTraversalError,
    resolve_safe_path,
    validate_file_path,
    validate_path_safe,
)


class TestValidateFilePath:
    """Test the validate_file_path function."""

    def test_valid_relative_path(self):
        """Test validation of simple relative paths."""
        result = validate_file_path("config.json")
        assert isinstance(result, Path)
        assert result.name == "config.json"

    def test_valid_relative_path_with_subdirs(self):
        """Test validation of relative paths with subdirectories."""
        result = validate_file_path("data/export/metrics.csv")
        assert isinstance(result, Path)
        assert result.name == "metrics.csv"

    def test_reject_path_traversal_dotdot(self):
        """Test rejection of path traversal with .. components."""
        with pytest.raises(PathTraversalError, match="Path traversal detected"):
            validate_file_path("../../../etc/passwd")

    def test_reject_path_traversal_in_middle(self):
        """Test rejection of path traversal in the middle of path."""
        with pytest.raises(PathTraversalError, match="Path traversal detected"):
            validate_file_path("data/../../../etc/passwd")

    def test_reject_null_bytes(self):
        """Test rejection of null bytes in path."""
        with pytest.raises(ValueError, match="Null byte detected"):
            validate_file_path("config.json\x00malicious")

    def test_reject_newline(self):
        """Test rejection of newline characters."""
        with pytest.raises(ValueError, match="Newline character"):
            validate_file_path("config\n.json")

    def test_reject_carriage_return(self):
        """Test rejection of carriage return characters."""
        with pytest.raises(ValueError, match="Carriage return"):
            validate_file_path("config\r.json")

    def test_reject_tilde_expansion(self):
        """Test rejection of tilde (home directory expansion)."""
        with pytest.raises(ValueError, match="Home directory expansion"):
            validate_file_path("~/config.json")

    def test_empty_path(self):
        """Test rejection of empty path."""
        with pytest.raises(ValueError, match="Path cannot be empty"):
            validate_file_path("")

    def test_absolute_path_not_allowed_by_default(self):
        """Test that absolute paths are rejected by default."""
        with pytest.raises(ValueError, match="Absolute paths not allowed"):
            validate_file_path("/etc/passwd")

    def test_absolute_path_allowed_when_enabled(self):
        """Test that absolute paths work when explicitly allowed."""
        result = validate_file_path("/tmp/test.json", allow_absolute=True)
        assert isinstance(result, Path)
        assert result.is_absolute()

    def test_allowed_extensions_json(self):
        """Test file extension validation."""
        result = validate_file_path(
            "config.json", allowed_extensions=[".json", ".yaml"]
        )
        assert result.suffix == ".json"

    def test_reject_disallowed_extension(self):
        """Test rejection of disallowed file extensions."""
        with pytest.raises(ValueError, match="File extension.*not allowed"):
            validate_file_path(
                "malicious.exe", allowed_extensions=[".json", ".yaml", ".toml"]
            )

    def test_case_insensitive_extensions(self):
        """Test that extension matching is case-insensitive."""
        result = validate_file_path("config.JSON", allowed_extensions=[".json"])
        assert result.suffix.lower() == ".json"

    def test_path_object_input(self):
        """Test that Path objects are accepted as input."""
        path_obj = Path("config.json")
        result = validate_file_path(path_obj)
        assert isinstance(result, Path)
        assert result.name == "config.json"


class TestResolveSafePath:
    """Test the resolve_safe_path function."""

    def test_resolve_relative_path_within_base(self, tmp_path):
        """Test resolving a relative path within base directory."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()

        result = resolve_safe_path("settings.json", base_dir)
        assert result.is_absolute()
        assert result.parent == base_dir

    def test_resolve_with_subdirectories(self, tmp_path):
        """Test resolving paths with subdirectories."""
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        result = resolve_safe_path("exports/metrics.csv", base_dir)
        assert result.is_absolute()
        assert result.parent.name == "exports"
        # Ensure the path is still within base_dir
        result.relative_to(base_dir)  # Should not raise

    def test_reject_path_escaping_base(self, tmp_path):
        """Test rejection of paths that escape base directory."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()

        # Should raise PathTraversalError (either during validate or resolve check)
        with pytest.raises(
            PathTraversalError, match="(escapes base directory|Path traversal detected)"
        ):
            resolve_safe_path("../../../etc/passwd", base_dir)

    def test_reject_absolute_path_outside_base(self, tmp_path):
        """Test rejection of absolute paths outside base directory."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()

        with pytest.raises(PathTraversalError, match="escapes base directory"):
            resolve_safe_path("/etc/passwd", base_dir)

    def test_must_exist_flag(self, tmp_path):
        """Test the must_exist flag."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()

        # Should fail when file doesn't exist
        with pytest.raises(ValueError, match="Path does not exist"):
            resolve_safe_path("nonexistent.json", base_dir, must_exist=True)

        # Should succeed when file exists
        test_file = base_dir / "existing.json"
        test_file.write_text("{}")

        result = resolve_safe_path("existing.json", base_dir, must_exist=True)
        assert result.exists()

    def test_empty_path(self, tmp_path):
        """Test rejection of empty path."""
        base_dir = tmp_path / "configs"
        base_dir.mkdir()

        with pytest.raises(ValueError, match="Path cannot be empty"):
            resolve_safe_path("", base_dir)

    def test_empty_base_dir(self):
        """Test rejection of empty base directory."""
        with pytest.raises(ValueError, match="Base directory cannot be empty"):
            resolve_safe_path("config.json", "")

    def test_allowed_extensions(self, tmp_path):
        """Test file extension restrictions."""
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        result = resolve_safe_path(
            "export.csv", base_dir, allowed_extensions=[".csv", ".json"]
        )
        assert result.suffix == ".csv"

        with pytest.raises(ValueError, match="File extension.*not allowed"):
            resolve_safe_path(
                "malicious.exe", base_dir, allowed_extensions=[".csv", ".json"]
            )

    def test_symlink_within_base(self, tmp_path):
        """Test that symlinks within base directory are allowed."""
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        # Create a file and symlink to it within base_dir
        real_file = base_dir / "real.json"
        real_file.write_text("{}")

        link_file = base_dir / "link.json"
        link_file.symlink_to(real_file)

        result = resolve_safe_path("link.json", base_dir)
        assert result.is_absolute()
        # The resolved path should still be within base_dir
        result.relative_to(base_dir)  # Should not raise


class TestBackwardCompatibility:
    """Test backward compatibility with validate_path_safe."""

    def test_validate_path_safe_still_works(self):
        """Test that old validate_path_safe function still works."""
        result = validate_path_safe("src/tnfr/core.py")
        assert isinstance(result, Path)

    def test_validate_path_safe_rejects_traversal(self):
        """Test that old function still rejects path traversal."""
        from tnfr.security import CommandValidationError

        with pytest.raises(CommandValidationError, match="Path traversal"):
            validate_path_safe("../../../etc/passwd")


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_config_file_loading_scenario(self, tmp_path):
        """Test typical config file loading scenario."""
        # Create config directory structure
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Create a valid config file
        config_file = config_dir / "settings.yaml"
        config_file.write_text("key: value\n")

        # Resolve with base directory restriction
        result = resolve_safe_path(
            "settings.yaml",
            config_dir,
            must_exist=True,
            allowed_extensions=[".yaml", ".json", ".toml"],
        )

        assert result.exists()
        assert result.parent == config_dir

    def test_data_export_scenario(self, tmp_path):
        """Test typical data export scenario."""
        # Create export directory
        export_dir = tmp_path / "exports"
        export_dir.mkdir()

        # Validate export path with subdirectory
        result = resolve_safe_path(
            "metrics/glyphogram.csv",
            export_dir,
            must_exist=False,
            allowed_extensions=[".csv", ".json"],
        )

        assert result.is_absolute()
        assert result.name == "glyphogram.csv"
        # Verify it's within export_dir
        result.relative_to(export_dir)  # Should not raise

    def test_cache_file_path_scenario(self, tmp_path):
        """Test cache file path validation."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Cache files can have various extensions
        result = resolve_safe_path(
            "coherence.db",
            cache_dir,
            must_exist=False,
        )

        assert result.is_absolute()
        assert result.parent == cache_dir

    def test_visualization_save_scenario(self, tmp_path):
        """Test visualization save path validation."""
        viz_dir = tmp_path / "visualizations"
        viz_dir.mkdir()

        # Visualizations can be various image formats
        result = resolve_safe_path(
            "phase_sync.png",
            viz_dir,
            must_exist=False,
        )

        assert result.is_absolute()
        assert result.parent == viz_dir

    def test_nested_directory_creation(self, tmp_path):
        """Test that nested directories can be safely validated."""
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        # Validate a deeply nested path
        result = resolve_safe_path(
            "exports/2024/01/metrics.json",
            base_dir,
            must_exist=False,
        )

        assert result.is_absolute()
        # Verify all parts are within base_dir
        result.relative_to(base_dir)  # Should not raise


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_dot_in_filename(self):
        """Test that dots in filenames (not path traversal) are allowed."""
        result = validate_file_path("my.config.json")
        assert result.name == "my.config.json"

    def test_leading_dot_hidden_file(self):
        """Test that hidden files (leading dot) are handled correctly."""
        result = validate_file_path(".config")
        assert result.name == ".config"

    def test_trailing_slash_removed(self):
        """Test handling of trailing slashes."""
        result = validate_file_path("directory/")
        assert isinstance(result, Path)

    def test_windows_style_paths_on_posix(self):
        """Test handling of Windows-style paths on POSIX systems."""
        # This should work as a filename, not a path separator
        result = validate_file_path("file:name.txt", allow_absolute=False)
        assert isinstance(result, Path)

    def test_unicode_characters_in_path(self):
        """Test that Unicode characters in paths are handled."""
        result = validate_file_path("données_français.json")
        assert isinstance(result, Path)
        assert "français" in str(result)

    def test_very_long_path(self, tmp_path):
        """Test handling of very long paths."""
        base_dir = tmp_path / "data"
        base_dir.mkdir()

        # Create a path with many nested directories
        long_path = "/".join([f"level{i}" for i in range(20)]) + "/file.txt"

        result = resolve_safe_path(long_path, base_dir, must_exist=False)
        assert result.is_absolute()
        # Should still be within base_dir
        result.relative_to(base_dir)  # Should not raise
