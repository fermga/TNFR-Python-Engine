"""Tests for secure cache layer helpers and security warnings."""

from __future__ import annotations

import os
import pytest

from tnfr.utils import (
    SecurityError,
    SecurityWarning,
    create_hmac_signer,
    create_hmac_validator,
    create_secure_shelve_layer,
    create_secure_redis_layer,
    ShelveCacheLayer,
    RedisCacheLayer,
)


class _FakeRedis:
    """Minimal Redis client stub for testing."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def delete(self, *keys: str) -> None:
        for key in keys:
            self._data.pop(key, None)

    def scan_iter(self, match: str | None = None):
        for key in list(self._data):
            yield key

    def keys(self, pattern: str) -> list[str]:
        return list(self._data.keys())


def test_create_hmac_signer_with_bytes():
    """Test HMAC signer creation with bytes secret."""
    secret = b"test-secret-key"
    signer = create_hmac_signer(secret)

    payload = b"test payload"
    signature = signer(payload)

    assert isinstance(signature, bytes)
    assert len(signature) == 32  # SHA256 produces 32 bytes

    # Same payload should produce same signature
    signature2 = signer(payload)
    assert signature == signature2


def test_create_hmac_signer_with_str():
    """Test HMAC signer creation with string secret."""
    secret = "test-secret-key"
    signer = create_hmac_signer(secret)

    payload = b"test payload"
    signature = signer(payload)

    assert isinstance(signature, bytes)
    assert len(signature) == 32


def test_create_hmac_validator_accepts_valid_signature():
    """Test HMAC validator accepts valid signatures."""
    secret = b"validation-secret"
    signer = create_hmac_signer(secret)
    validator = create_hmac_validator(secret)

    payload = b"important data"
    signature = signer(payload)

    assert validator(payload, signature) is True


def test_create_hmac_validator_rejects_invalid_signature():
    """Test HMAC validator rejects tampered signatures."""
    secret = b"validation-secret"
    signer = create_hmac_signer(secret)
    validator = create_hmac_validator(secret)

    payload = b"important data"
    signature = signer(payload)

    # Tamper with signature
    tampered = bytearray(signature)
    tampered[0] ^= 0xFF

    assert validator(payload, bytes(tampered)) is False


def test_create_hmac_validator_rejects_wrong_secret():
    """Test HMAC validator rejects signatures from different secret."""
    secret1 = b"secret-one"
    secret2 = b"secret-two"

    signer = create_hmac_signer(secret1)
    validator = create_hmac_validator(secret2)

    payload = b"data"
    signature = signer(payload)

    assert validator(payload, signature) is False


def test_create_secure_shelve_layer_with_secret(tmp_path):
    """Test creating secure shelve layer with explicit secret."""
    secret = b"my-secure-secret"
    layer = create_secure_shelve_layer(str(tmp_path / "secure.db"), secret=secret)

    try:
        # Verify it's configured for signature validation
        assert layer._require_signature is True
        assert layer._signer is not None
        assert layer._validator is not None

        # Test round-trip
        layer.store("test", {"value": 42, "epi": [1.0, 2.0, 3.0]})
        result = layer.load("test")
        assert result == {"value": 42, "epi": [1.0, 2.0, 3.0]}
    finally:
        layer.close()


def test_create_secure_shelve_layer_from_env(tmp_path, monkeypatch):
    """Test creating secure shelve layer from environment variable."""
    monkeypatch.setenv("TNFR_CACHE_SECRET", "env-secret-key")

    layer = create_secure_shelve_layer(str(tmp_path / "env-secure.db"))

    try:
        assert layer._require_signature is True

        # Test it works
        layer.store("data", [1, 2, 3])
        assert layer.load("data") == [1, 2, 3]
    finally:
        layer.close()


def test_create_secure_shelve_layer_requires_secret(tmp_path, monkeypatch):
    """Test secure shelve layer raises if no secret provided."""
    monkeypatch.delenv("TNFR_CACHE_SECRET", raising=False)

    with pytest.raises(ValueError, match="Secret required"):
        create_secure_shelve_layer(str(tmp_path / "nosecret.db"))


def test_create_secure_redis_layer_with_secret():
    """Test creating secure Redis layer with explicit secret."""
    fake_client = _FakeRedis()
    secret = b"redis-secret"

    layer = create_secure_redis_layer(
        client=fake_client, secret=secret, namespace="test:cache"
    )

    assert layer._require_signature is True
    assert layer._signer is not None
    assert layer._validator is not None

    # Test round-trip
    layer.store("key1", {"nfr": "data"})
    result = layer.load("key1")
    assert result == {"nfr": "data"}


def test_create_secure_redis_layer_from_env(monkeypatch):
    """Test creating secure Redis layer from environment."""
    monkeypatch.setenv("TNFR_CACHE_SECRET", "redis-env-key")
    fake_client = _FakeRedis()

    layer = create_secure_redis_layer(client=fake_client)

    assert layer._require_signature is True

    layer.store("test", {"value": 99})
    assert layer.load("test") == {"value": 99}


def test_create_secure_redis_layer_requires_secret(monkeypatch):
    """Test secure Redis layer raises if no secret provided."""
    monkeypatch.delenv("TNFR_CACHE_SECRET", raising=False)

    with pytest.raises(ValueError, match="Secret required"):
        create_secure_redis_layer(client=_FakeRedis())


def test_unsigned_shelve_layer_emits_warning(tmp_path, monkeypatch):
    """Test ShelveCacheLayer emits warning when used without signatures."""
    monkeypatch.delenv("TNFR_ALLOW_UNSIGNED_PICKLE", raising=False)

    with pytest.warns(SecurityWarning, match="pickle without signature validation"):
        layer = ShelveCacheLayer(str(tmp_path / "unsigned.db"))
        layer.close()


def test_unsigned_shelve_layer_suppressed_by_env(tmp_path, monkeypatch):
    """Test warning can be suppressed via environment variable."""
    monkeypatch.setenv("TNFR_ALLOW_UNSIGNED_PICKLE", "1")

    # Should not emit warning
    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        layer = ShelveCacheLayer(str(tmp_path / "suppressed.db"))
        layer.close()

    # Check no SecurityWarning was issued
    security_warnings = [
        w for w in warning_list if issubclass(w.category, SecurityWarning)
    ]
    assert len(security_warnings) == 0


def test_unsigned_redis_layer_emits_warning(monkeypatch):
    """Test RedisCacheLayer emits warning when used without signatures."""
    monkeypatch.delenv("TNFR_ALLOW_UNSIGNED_PICKLE", raising=False)

    with pytest.warns(SecurityWarning, match="pickle without signature validation"):
        _layer = RedisCacheLayer(client=_FakeRedis())


def test_unsigned_redis_layer_suppressed_by_env(monkeypatch):
    """Test Redis warning can be suppressed via environment variable."""
    monkeypatch.setenv("TNFR_ALLOW_UNSIGNED_PICKLE", "1")

    import warnings

    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        _layer = RedisCacheLayer(client=_FakeRedis())

    security_warnings = [
        w for w in warning_list if issubclass(w.category, SecurityWarning)
    ]
    assert len(security_warnings) == 0


def test_secure_shelve_layer_rejects_tampering(tmp_path):
    """Test secure shelve layer detects and rejects tampered data."""
    secret = b"tamper-test-secret"
    layer = create_secure_shelve_layer(str(tmp_path / "tamper.db"), secret=secret)

    try:
        layer.store("data", {"value": 100})

        # Directly tamper with the underlying shelf
        with layer._lock:
            blob = layer._shelf["data"]
            assert isinstance(blob, (bytes, bytearray))
            # Flip last byte
            mutated = bytearray(blob)
            mutated[-1] ^= 0xFF
            layer._shelf["data"] = bytes(mutated)
            layer._shelf.sync()

        # Should raise SecurityError and delete the entry
        with pytest.raises(SecurityError, match="signature validation failed"):
            layer.load("data")

        # Entry should be deleted
        with layer._lock:
            assert "data" not in layer._shelf
    finally:
        layer.close()


def test_secure_redis_layer_rejects_tampering():
    """Test secure Redis layer detects and rejects tampered data."""
    fake_client = _FakeRedis()
    secret = b"redis-tamper-test"
    layer = create_secure_redis_layer(client=fake_client, secret=secret)

    layer.store("key", {"important": "data"})

    # Tamper with Redis data
    redis_key = "tnfr:cache:key"
    blob = fake_client._data[redis_key]
    mutated = bytearray(blob)
    mutated[-1] ^= 0x0F
    fake_client._data[redis_key] = bytes(mutated)

    # Should raise SecurityError and delete the entry
    with pytest.raises(SecurityError, match="signature validation failed"):
        layer.load("key")

    # Entry should be deleted
    assert redis_key not in fake_client._data


def test_secure_layers_preserve_tnfr_structures(tmp_path):
    """Test secure layers correctly serialize TNFR data structures."""
    import networkx as nx

    secret = b"tnfr-structures-test"
    layer = create_secure_shelve_layer(str(tmp_path / "tnfr.db"), secret=secret)

    try:
        # Create TNFR-like data structures
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3])
        G.add_edges_from([(1, 2), (2, 3)])

        # Store complex nested structure
        tnfr_data = {
            "graph": G,
            "epi": [1.5, 2.3, 3.7],
            "theta": [0.1, 0.2, 0.3],
            "coherence": 0.85,
            "metadata": {
                "vf": 2.5,
                "phase": "resonant",
            },
        }

        layer.store("tnfr_state", tnfr_data)
        restored = layer.load("tnfr_state")

        # Verify structure preservation
        assert isinstance(restored, dict)
        assert isinstance(restored["graph"], nx.Graph)
        assert list(restored["graph"].nodes()) == [1, 2, 3]
        assert list(restored["graph"].edges()) == [(1, 2), (2, 3)]
        assert restored["epi"] == [1.5, 2.3, 3.7]
        assert restored["theta"] == [0.1, 0.2, 0.3]
        assert restored["coherence"] == 0.85
        assert restored["metadata"]["vf"] == 2.5
    finally:
        layer.close()
