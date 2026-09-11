"""Synthetic, offline Redis configuration parsing regressions."""

import pytest

from tnfr.config.security import ConfigurationError, SecurityAuditor, load_redis_config


@pytest.fixture(autouse=True)
def isolated_redis_environment(monkeypatch):
    for key in ("REDIS_URL", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD",
                "REDIS_DB", "REDIS_USE_TLS"):
        monkeypatch.delenv(key, raising=False)


def test_url_credentials_are_decoded_for_connection_parameters(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "rediss://:synthetic%40pass%2Fword@localhost:6380/0")
    config = load_redis_config()
    assert config["password"] == "synthetic@pass/word"
    assert (config["host"], config["port"], config["db"], config["ssl"]) == (
        "localhost", 6380, 0, True,
    )


def test_individual_reserved_password_does_not_break_url_validation(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic/pass#word?value")
    assert load_redis_config()["password"] == "synthetic/pass#word?value"


@pytest.mark.parametrize("raw,expected", [(" true ", True), (" FALSE ", False)])
def test_tls_flag_ignores_surrounding_whitespace(monkeypatch, raw, expected):
    monkeypatch.setenv("REDIS_USE_TLS", raw)
    assert load_redis_config()["ssl"] is expected


@pytest.mark.parametrize("from_url", [True, False])
@pytest.mark.parametrize("database", ["-1", "wrong"])
def test_invalid_databases_fail_consistently(monkeypatch, from_url, database):
    key, value = ("REDIS_URL", f"redis://localhost/{database}") if from_url else (
        "REDIS_DB", database,
    )
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigurationError, match="REDIS_DB"):
        load_redis_config()


def test_individual_ipv6_host_can_be_validated(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "::1")
    config = load_redis_config()
    assert config["host"] == "::1"
    assert config["db"] == 0


def test_security_audit_reads_effective_url_configuration(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "rediss://:synthetic-password@localhost/0")
    assert SecurityAuditor().check_redis_config_security() == []


def test_security_audit_obeys_url_precedence_over_individual_settings(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost/0")
    monkeypatch.setenv("REDIS_USE_TLS", "true")
    monkeypatch.setenv("REDIS_PASSWORD", "synthetic-password")
    issues = SecurityAuditor().check_redis_config_security()
    assert len(issues) == 2


def test_invalid_tls_flag_is_rejected(monkeypatch):
    monkeypatch.setenv("REDIS_USE_TLS", "unrecognized")
    with pytest.raises(ConfigurationError, match="REDIS_USE_TLS"):
        load_redis_config()
