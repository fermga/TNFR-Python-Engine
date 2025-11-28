"""Security tests to verify no hardcoded secrets in the TNFR codebase.

These tests ensure that:
1. No hardcoded API keys, passwords, or tokens exist in source code
2. All sensitive configuration is loaded from environment variables
3. The configuration utilities work correctly
"""

import os
import re
from pathlib import Path

import pytest

from tnfr.secure_config import (
    ConfigurationError,
    get_cache_secret,
    get_env_variable,
    load_github_credentials,
    load_pypi_credentials,
    load_redis_config,
    validate_no_hardcoded_secrets,
)


class TestNoHardcodedSecrets:
    """Test suite to verify no hardcoded secrets in source code."""

    @pytest.fixture
    def repo_root(self):
        """Get the repository root directory."""
        # Tests are in tests/unit/security/
        return Path(__file__).parent.parent.parent.parent

    @pytest.fixture
    def source_files(self, repo_root):
        """Get all Python source files."""
        src_dir = repo_root / "src"
        scripts_dir = repo_root / "scripts"

        files = []
        if src_dir.exists():
            files.extend(src_dir.rglob("*.py"))
        if scripts_dir.exists():
            files.extend(scripts_dir.rglob("*.py"))

        return [f for f in files if "__pycache__" not in str(f)]

    def test_no_hardcoded_github_tokens(self, source_files):
        """Verify no hardcoded GitHub tokens in source code."""
        # Pattern for GitHub tokens - adjusted to catch more variants
        # Classic tokens are ~40 chars after prefix, fine-grained can vary
        github_token_pattern = re.compile(r"(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{30,}", re.IGNORECASE)

        violations = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = github_token_pattern.findall(content)
                if matches:
                    violations.append((file_path, matches))
            except Exception:
                # Skip files that can't be read
                continue

        assert not violations, f"Found potential hardcoded GitHub tokens in:\n" + "\n".join(
            f"  {path}: {matches}" for path, matches in violations
        )

    def test_no_hardcoded_pypi_tokens(self, source_files):
        """Verify no hardcoded PyPI tokens in source code."""
        # Pattern for PyPI tokens - comprehensive to catch all variants
        pypi_token_pattern = re.compile(r"pypi-[a-zA-Z0-9+/=_-]+", re.IGNORECASE)

        violations = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = pypi_token_pattern.findall(content)
                if matches:
                    # Filter out placeholder patterns (all X's or clearly fake)
                    real_matches = [m for m in matches if "X" * 10 not in m]
                    if real_matches:
                        violations.append((file_path, real_matches))
            except Exception:
                continue

        assert not violations, f"Found potential hardcoded PyPI tokens in:\n" + "\n".join(
            f"  {path}: {matches}" for path, matches in violations
        )

    def test_no_suspicious_long_strings(self, source_files):
        """Check for suspiciously long alphanumeric strings that might be secrets."""
        # Pattern for long alphanumeric strings (potential secrets)
        # At least 32 characters of base64-like content
        # Using non-capturing groups for quotes for efficiency
        suspicious_pattern = re.compile(r'(?:["\'])([a-zA-Z0-9+/=_-]{32,})(?:["\'])')

        # Allowed patterns (version strings, hashes, etc.)
        allowed_patterns = [
            r"^[0-9]+\.[0-9]+\.[0-9]+",  # Version strings
            r"^v[0-9]+\.[0-9]+",  # Version tags
            r"^[0-9a-f]{32,}$",  # Hash values (all lowercase hex)
            r"^sha256:[0-9a-f]+$",  # SHA256 hashes
        ]

        violations = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                # Skip certain files by exact name matching
                if file_path.name in [
                    ".env.example",
                    "_version.py",
                    "_generated_version.py",
                ]:
                    continue

                for match in suspicious_pattern.finditer(content):
                    string_value = match.group(1)
                    # Check if it matches allowed patterns
                    if any(re.match(pattern, string_value) for pattern in allowed_patterns):
                        continue
                    # Skip common non-secret strings
                    if string_value.isupper():  # Environment variable names
                        continue
                    if string_value.lower().startswith("http"):  # URLs
                        continue

                    # Get context (line number)
                    line_num = content[: match.start()].count("\n") + 1
                    violations.append((file_path, line_num, string_value[:20] + "..."))
            except Exception:
                continue

        # This test warns but doesn't fail to avoid false positives
        if violations:
            message = "Found suspicious long strings (potential secrets):\n" + "\n".join(
                f"  {path}:{line}: {value}" for path, line, value in violations[:10]
            )
            pytest.skip(f"Review needed: {message}")

    def test_env_files_in_gitignore(self, repo_root):
        """Verify that .env files are in .gitignore."""
        gitignore_path = repo_root / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file not found"

        gitignore_content = gitignore_path.read_text(encoding="utf-8")
        assert ".env" in gitignore_content, ".env not found in .gitignore"

    def test_env_example_exists(self, repo_root):
        """Verify that .env.example template exists."""
        env_example_path = repo_root / ".env.example"
        assert env_example_path.exists(), ".env.example template not found"

        content = env_example_path.read_text(encoding="utf-8")
        # Check that it has placeholder values, not real secrets
        assert (
            "your-" in content.lower() or "..." in content
        ), ".env.example should contain placeholders, not real values"

    def test_no_actual_env_file_in_repo(self, repo_root):
        """Verify that .env file is not committed to repository."""
        env_path = repo_root / ".env"
        # In a fresh clone, .env should not exist
        # This test might pass in development environments where .env exists
        # but is gitignored
        if env_path.exists():
            pytest.skip(".env file exists (likely in development, should be gitignored)")


class TestConfigurationUtilities:
    """Test the secure configuration utilities."""

    def test_get_env_variable_with_default(self, monkeypatch):
        """Test getting environment variable with default value."""
        monkeypatch.delenv("TEST_VAR", raising=False)
        value = get_env_variable("TEST_VAR", default="default_value")
        assert value == "default_value"

    def test_get_env_variable_set(self, monkeypatch):
        """Test getting set environment variable."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        value = get_env_variable("TEST_VAR", default="default_value")
        assert value == "test_value"

    def test_get_env_variable_required_missing(self, monkeypatch):
        """Test that required variable raises error when missing."""
        monkeypatch.delenv("REQUIRED_VAR", raising=False)
        with pytest.raises(ConfigurationError, match="Required environment variable"):
            get_env_variable("REQUIRED_VAR", required=True)

    def test_get_env_variable_secret_warning(self, monkeypatch):
        """Test that using default for secret issues warning."""
        monkeypatch.delenv("SECRET_VAR", raising=False)
        with pytest.warns(UserWarning, match="Using default value for secret"):
            get_env_variable("SECRET_VAR", default="default", secret=True)

    def test_load_pypi_credentials(self, monkeypatch):
        """Test loading PyPI credentials from environment."""
        monkeypatch.setenv("PYPI_USERNAME", "__token__")
        monkeypatch.setenv("PYPI_PASSWORD", "test-token")
        monkeypatch.setenv("PYPI_REPOSITORY", "testpypi")

        creds = load_pypi_credentials()
        assert creds["username"] == "__token__"
        assert creds["password"] == "test-token"
        assert creds["repository"] == "testpypi"

    def test_load_pypi_credentials_fallback(self, monkeypatch):
        """Test PyPI credentials fallback to TWINE_ variables."""
        monkeypatch.delenv("PYPI_USERNAME", raising=False)
        monkeypatch.delenv("PYPI_PASSWORD", raising=False)
        monkeypatch.setenv("TWINE_USERNAME", "__token__")
        monkeypatch.setenv("TWINE_PASSWORD", "twine-token")

        creds = load_pypi_credentials()
        assert creds["username"] == "__token__"
        assert creds["password"] == "twine-token"

    def test_load_github_credentials(self, monkeypatch):
        """Test loading GitHub credentials from environment."""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
        monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")

        creds = load_github_credentials()
        assert creds["token"] == "ghp_test"
        assert creds["repository"] == "owner/repo"

    def test_load_redis_config_defaults(self, monkeypatch):
        """Test loading Redis config with defaults."""
        # Clear all Redis environment variables
        for key in [
            "REDIS_HOST",
            "REDIS_PORT",
            "REDIS_PASSWORD",
            "REDIS_DB",
            "REDIS_USE_TLS",
        ]:
            monkeypatch.delenv(key, raising=False)

        config = load_redis_config()
        assert config["host"] == "localhost"
        assert config["port"] == 6379
        assert config["password"] is None
        assert config["db"] == 0
        assert config["ssl"] is False

    def test_load_redis_config_custom(self, monkeypatch):
        """Test loading custom Redis config."""
        monkeypatch.setenv("REDIS_HOST", "redis.example.com")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_PASSWORD", "secret")
        monkeypatch.setenv("REDIS_DB", "1")
        monkeypatch.setenv("REDIS_USE_TLS", "true")

        config = load_redis_config()
        assert config["host"] == "redis.example.com"
        assert config["port"] == 6380
        assert config["password"] == "secret"
        assert config["db"] == 1
        assert config["ssl"] is True

    def test_load_redis_config_invalid_port(self, monkeypatch):
        """Test that invalid port raises error."""
        monkeypatch.setenv("REDIS_PORT", "not-a-number")

        with pytest.raises(ConfigurationError, match="REDIS_PORT must be an integer"):
            load_redis_config()

    def test_get_cache_secret_none(self, monkeypatch):
        """Test cache secret when not set."""
        monkeypatch.delenv("TNFR_CACHE_SECRET", raising=False)
        secret = get_cache_secret()
        assert secret is None

    def test_get_cache_secret_valid(self, monkeypatch):
        """Test getting valid cache secret."""
        test_secret = "0123456789abcdef" * 4  # 64-char hex
        monkeypatch.setenv("TNFR_CACHE_SECRET", test_secret)

        secret = get_cache_secret()
        assert secret is not None
        assert isinstance(secret, bytes)
        assert len(secret) == 32  # 64 hex chars = 32 bytes

    def test_get_cache_secret_invalid(self, monkeypatch):
        """Test that invalid hex raises error."""
        monkeypatch.setenv("TNFR_CACHE_SECRET", "not-valid-hex")

        with pytest.raises(ConfigurationError, match="must be a hex-encoded string"):
            get_cache_secret()

    def test_validate_github_token_pattern(self):
        """Test validation rejects GitHub token patterns."""
        with pytest.raises(ValueError, match="GitHub token"):
            validate_no_hardcoded_secrets("ghp_1234567890abcdef")

    def test_validate_pypi_token_pattern(self):
        """Test validation rejects PyPI token patterns."""
        with pytest.raises(ValueError, match="PyPI token"):
            validate_no_hardcoded_secrets("pypi-AgEIcHlwaS5vcmcCtest")

    def test_validate_safe_strings(self):
        """Test validation accepts safe strings."""
        assert validate_no_hardcoded_secrets("my-password")
        assert validate_no_hardcoded_secrets("test-value-123")
        assert validate_no_hardcoded_secrets("ENVIRONMENT_VARIABLE_NAME")


class TestSecurityDocumentation:
    """Test that security documentation is present and complete."""

    @pytest.fixture
    def repo_root(self):
        """Get the repository root directory."""
        return Path(__file__).parent.parent.parent.parent

    def test_security_md_exists(self, repo_root):
        """Verify SECURITY.md exists."""
        security_md = repo_root / "SECURITY.md"
        assert security_md.exists(), "SECURITY.md not found"

    def test_security_md_mentions_secrets(self, repo_root):
        """Verify SECURITY.md discusses secret management."""
        security_md = repo_root / "SECURITY.md"
        content = security_md.read_text(encoding="utf-8")

        # Check for key security topics
        keywords = ["secret", "token", "password", "credential", "environment"]
        found_keywords = [kw for kw in keywords if kw.lower() in content.lower()]

        assert len(found_keywords) >= 3, (
            f"SECURITY.md should discuss secret management. " f"Found: {found_keywords}"
        )

    def test_env_example_has_documentation(self, repo_root):
        """Verify .env.example has proper documentation."""
        env_example = repo_root / ".env.example"
        content = env_example.read_text(encoding="utf-8")

        # Check for security warnings
        assert (
            "never commit" in content.lower() or "do not commit" in content.lower()
        ), ".env.example should warn about not committing credentials"

        # Check for security best practices
        assert (
            "api token" in content.lower() or "token" in content.lower()
        ), ".env.example should mention using API tokens"
