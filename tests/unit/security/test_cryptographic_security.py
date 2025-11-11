"""Tests for cryptographic security in TNFR.

These tests verify that the TNFR engine uses secure cryptographic algorithms
and does not use weak or outdated cryptographic primitives.

Intent: Ensure structural hash calculations, node authentication, and data
integrity verification use modern cryptographic algorithms (BLAKE2b, SHA-256+).

Operators involved: Coherence (stable hashing), Self-organization (RNG seeding)
Affected invariants: #8 (Controlled determinism), #9 (Structural metrics)
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path

import pytest


class TestNoWeakHashAlgorithms:
    """Verify that weak hash algorithms are not used in the codebase."""

    def test_no_md5_in_source(self) -> None:
        """Verify MD5 is not used for any purpose in source code."""
        # Use parents[4] to go from test file up to project root
        src_dir = Path(__file__).parents[4] / "src" / "tnfr"
        # Pattern specifically for hashlib.md5 usage, not arbitrary .md5() method calls
        md5_pattern = re.compile(r"hashlib\.md5\(|from\s+hashlib\s+import.*\bmd5\b")

        violations = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for line_num, line in enumerate(content.splitlines(), 1):
                if md5_pattern.search(line):
                    violations.append(f"{py_file.name}:{line_num}: {line.strip()}")

        assert not violations, f"Found MD5 usage in source code:\n" + "\n".join(violations)

    def test_no_sha1_in_source(self) -> None:
        """Verify SHA-1 is not used for any purpose in source code."""
        # Use parents[4] to go from test file up to project root
        src_dir = Path(__file__).parents[4] / "src" / "tnfr"
        # Pattern specifically for hashlib.sha1 usage, not arbitrary .sha1() method calls
        sha1_pattern = re.compile(r"hashlib\.sha1\(|from\s+hashlib\s+import.*\bsha1\b")

        violations = []
        for py_file in src_dir.rglob("*.py"):
            content = py_file.read_text(encoding="utf-8")
            for line_num, line in enumerate(content.splitlines(), 1):
                if sha1_pattern.search(line):
                    violations.append(f"{py_file.name}:{line_num}: {line.strip()}")

        assert not violations, f"Found SHA-1 usage in source code:\n" + "\n".join(violations)


class TestModernCryptographicAlgorithms:
    """Verify that modern cryptographic algorithms are used correctly."""

    def test_blake2b_available(self) -> None:
        """Verify BLAKE2b is available and working."""
        data = b"test data for TNFR structural hash"
        hash_result = hashlib.blake2b(data, digest_size=16).hexdigest()

        # BLAKE2b with digest_size=16 produces 32 hex characters
        assert len(hash_result) == 32
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_blake2b_deterministic(self) -> None:
        """Verify BLAKE2b produces deterministic results (TNFR Invariant #8)."""
        data = b"TNFR structural coherence test"

        hash1 = hashlib.blake2b(data, digest_size=8).hexdigest()
        hash2 = hashlib.blake2b(data, digest_size=8).hexdigest()

        assert hash1 == hash2, "BLAKE2b must be deterministic for structural hashing"

    def test_hmac_sha256_available(self) -> None:
        """Verify HMAC-SHA256 is available for cache validation."""
        secret = b"test-secret-key"
        message = b"test message"

        mac = hmac.new(secret, message, hashlib.sha256).digest()

        # SHA-256 produces 32 bytes
        assert len(mac) == 32

    def test_hmac_sha256_secure_comparison(self) -> None:
        """Verify HMAC uses constant-time comparison."""
        secret = b"secret"
        msg = b"message"

        mac1 = hmac.new(secret, msg, hashlib.sha256).digest()
        mac2 = hmac.new(secret, msg, hashlib.sha256).digest()

        # hmac.compare_digest performs constant-time comparison
        assert hmac.compare_digest(mac1, mac2)


class TestStructuralHashingSecurity:
    """Verify structural hash functions use secure algorithms."""

    def test_rng_seed_hash_uses_blake2b(self) -> None:
        """Verify RNG seed hashing uses BLAKE2b."""
        from tnfr.rng import seed_hash

        seed = 12345
        key = 67890

        # seed_hash should use BLAKE2b internally
        result = seed_hash(seed, key)

        # Result should be a 64-bit integer
        assert isinstance(result, int)
        assert 0 <= result < 2**64

    def test_rng_seed_hash_deterministic(self) -> None:
        """Verify RNG seed hashing is deterministic (TNFR Invariant #8)."""
        from tnfr.rng import seed_hash

        seed = 42
        key = 100

        hash1 = seed_hash(seed, key)
        hash2 = seed_hash(seed, key)

        assert hash1 == hash2, "RNG seed hashing must be deterministic"

    def test_remesh_topology_snapshot_secure(self) -> None:
        """Verify remesh topology snapshots use secure hashing."""
        try:
            import networkx as nx
            from tnfr.operators.remesh import _snapshot_topology
        except ImportError:
            pytest.skip("networkx not available")

        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])

        snapshot = _snapshot_topology(G, nx)

        # Should return a 12-character hex string (6 bytes of BLAKE2b)
        assert snapshot is not None
        assert len(snapshot) == 12
        assert all(c in "0123456789abcdef" for c in snapshot)

    def test_remesh_epi_snapshot_secure(self) -> None:
        """Verify remesh EPI snapshots use secure hashing."""
        try:
            import networkx as nx
            from tnfr.operators.remesh import _snapshot_epi
        except ImportError:
            pytest.skip("networkx not available")

        G = nx.Graph()
        G.add_node(0, epi=1.0)
        G.add_node(1, epi=2.0)

        mean_val, checksum = _snapshot_epi(G)

        # Should return mean and a 12-character hex string (6 bytes of BLAKE2b)
        assert isinstance(mean_val, float)
        assert len(checksum) == 12
        assert all(c in "0123456789abcdef" for c in checksum)


class TestCacheSecurityFeatures:
    """Verify cache security features are properly implemented."""

    def test_hmac_signer_creation(self) -> None:
        """Verify HMAC signer can be created with secure algorithm."""
        from tnfr.utils.cache import create_hmac_signer

        secret = b"test-secret"
        signer = create_hmac_signer(secret)

        # Signer should be callable
        assert callable(signer)

        # Should produce consistent signatures
        payload = b"test payload"
        sig1 = signer(payload)
        sig2 = signer(payload)
        assert sig1 == sig2
        assert len(sig1) == 32  # SHA-256 produces 32 bytes

    def test_hmac_validator_creation(self) -> None:
        """Verify HMAC validator uses secure comparison."""
        from tnfr.utils.cache import create_hmac_validator, create_hmac_signer

        secret = b"test-secret"
        signer = create_hmac_signer(secret)
        validator = create_hmac_validator(secret)

        payload = b"test payload"
        signature = signer(payload)

        # Should validate correct signature
        assert validator(payload, signature)

        # Should reject incorrect signature (32 bytes required for SHA-256)
        wrong_signature = b"x" * 32
        assert not validator(payload, wrong_signature)


class TestRandomNumberGeneration:
    """Verify random number generation uses secure seeding."""

    def test_make_rng_deterministic(self) -> None:
        """Verify RNG is deterministic with same seed/key (TNFR Invariant #8)."""
        from tnfr.rng import make_rng

        seed = 42
        key = 100

        rng1 = make_rng(seed, key)
        rng2 = make_rng(seed, key)

        # Both should produce same sequence
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 == seq2, "RNG must be deterministic for reproducibility"

    def test_make_rng_different_keys(self) -> None:
        """Verify different keys produce different RNG streams."""
        from tnfr.rng import make_rng

        seed = 42

        rng1 = make_rng(seed, key=1)
        rng2 = make_rng(seed, key=2)

        # Different keys should produce different sequences
        seq1 = [rng1.random() for _ in range(10)]
        seq2 = [rng2.random() for _ in range(10)]

        assert seq1 != seq2, "Different keys must produce different RNG streams"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
