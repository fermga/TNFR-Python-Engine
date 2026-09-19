"""Regression tests for the TNFR spectral factorization lab."""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pytest
from _pytest.monkeypatch import MonkeyPatch

# Ensure the lab package is importable when tests run from repo root
LAB_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

import tnfr_factorization.spectral_paley as sp  # type: ignore[import]  # noqa: E402
from tnfr_factorization import (  # type: ignore[import]  # noqa: E402
    SpectralPaleyFactorizer,
)

from tnfr.dynamics.fft_backend import (  # type: ignore[import]  # noqa: E402
    FFTBackendCapabilities,
)


@dataclass
class _StubSpectralState:
    eigenvalues: np.ndarray
    coherence_length: float


class _RecordingFFTEngine:
    """Minimal stand-in for TNFRAdvancedFFTEngine that records calls."""

    backend_name = "tests.recording_fft"

    def __init__(self, coherence_length: float = 3.0) -> None:
        self.coherence_length = coherence_length
        self.calls = 0
        self.last_node_count: Optional[int] = None

    def get_spectral_state(
        self, graph: Any, force_recompute: bool = False
    ) -> _StubSpectralState:  # noqa: D401
        self.calls += 1
        self.last_node_count = graph.number_of_nodes()
        count = self.last_node_count or 1
        eigenvalues = np.linspace(0.0, 1.0, count, dtype=float)
        return _StubSpectralState(
            eigenvalues=eigenvalues, coherence_length=self.coherence_length
        )

    def get_capabilities(self) -> FFTBackendCapabilities:
        return FFTBackendCapabilities(
            backend_name=self.backend_name,
            max_nodes=2048,
            precision="float32",
            supports_distributed=False,
            extra={"testing": True},
        )

    def spectral_convolution(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("convolution not needed in tests")


def test_fft_enabled_path_uses_engine_and_reports_coherence_length() -> None:
    engine = _RecordingFFTEngine(coherence_length=5.5)
    factorizer = SpectralPaleyFactorizer(fft_engine=engine)
    result = factorizer.analyze(37)
    assert result.laplacian_gap >= 0.0
    assert result.fft_backend == _RecordingFFTEngine.backend_name
    assert result.fft_capabilities and result.fft_capabilities["max_nodes"] == 2048
    assert result.partition_summary and result.partition_summary["partition_count"] >= 1
    assert result.partition_aggregation
    assert result.partition_aggregation["partition_count"] >= 1
    assert "partition_candidates" in result.partition_aggregation
    assert result.operator_strategy_plan
    assert result.operator_strategy_plan["per_partition"]


def test_partition_env_controls_size(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "5")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "1")

    engine = _RecordingFFTEngine(coherence_length=4.0)
    factorizer = SpectralPaleyFactorizer(fft_engine=engine)

    result = factorizer.analyze(61)

    assert result.partition_summary
    assert result.partition_summary["partition_count"] > 1
    assert result.partition_aggregation
    assert result.partition_aggregation["phi_s_ratio"] > 0.0
    if result.candidate_factors:
        assert result.partition_aggregation["candidate_total"] == len(
            result.candidate_factors
        )
    assert "partition_candidates" in result.partition_aggregation
    assert result.operator_strategy_plan
    per_partition = result.operator_strategy_plan["per_partition"]
    assert len(per_partition) == result.partition_summary["partition_count"]


def test_nodal_decoder_derives_partition_factors(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "13")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "0")

    factorizer = SpectralPaleyFactorizer()
    result = factorizer.analyze(221)

    decoding = result.nodal_decoding
    assert decoding, "nodal decoding metadata should be present"
    assert decoding["sequence"] == ["UM", "RA", "IL", "THOL"]
    assert (
        13 in decoding["dynamic_factors"]
    ), "partition sequence should surface factor 13"
    assert 13 in result.candidate_factors
    partitions = decoding.get("partitions", [])
    assert any(entry.get("inferred_factor") == 13 for entry in partitions)
    assert result.tnfr_certified_factors
    assert 13 in result.tnfr_certified_factors
    assert result.tnfr_verification
    assert 13 in result.tnfr_verification.get("certified", [])
    assert result.tnfr_factor_signature
    assert 13 in result.tnfr_factor_signature.get("certified", [])
    assert result.partition_aggregation
    assert result.partition_aggregation["candidate_total"] == len(
        result.candidate_factors
    )


def test_arithmetic_cache_invoked_once(monkeypatch: MonkeyPatch) -> None:
    call_count = {"count": 0}

    cache_clear = getattr(sp._compute_arithmetic_telemetry, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()

    def _fake_prime_factorization(n: int) -> dict[int, int]:
        call_count["count"] += 1
        return {7: 1, 11: 1}

    monkeypatch.setattr(sp, "_prime_factorization", _fake_prime_factorization)

    factorizer = SpectralPaleyFactorizer()

    target = 899
    first = factorizer.analyze(target)
    second = factorizer.analyze(target)

    assert call_count["count"] == 1
    # Verify the mocked factorization influenced arithmetic telemetry (τ = 4 for 7·11)
    assert first.arithmetic_terms.tau == 4
    assert first.arithmetic_terms == second.arithmetic_terms
    assert first.candidate_factors == second.candidate_factors
    assert pytest.approx(first.arithmetic_delta_nfr) == second.arithmetic_delta_nfr


def test_certificate_emission_obeys_grammar(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    partition_root = tmp_path / "partition_outputs"
    monkeypatch.setenv("TNFR_PARTITION_OUTPUT_DIR", str(partition_root))
    monkeypatch.setenv("TNFR_PARTITION_TARGET_SIZE", "13")
    monkeypatch.setenv("TNFR_PARTITION_OVERLAP", "0")

    factorizer = SpectralPaleyFactorizer()

    result = factorizer.analyze(
        221,
        trace_certificates=True,
        certificate_dir=tmp_path,
    )

    certificate_files = list(tmp_path.glob("certificate_*.json"))
    assert certificate_files, "Expected a certificate JSON file"
    certificate_path = certificate_files[0]
    payload = json.loads(certificate_path.read_text())

    assert (
        payload["candidate_factor"] in result.candidate_factors
        or payload["candidate_factor"] is None
    )

    assert payload.get("canonical_operators")
    optimizer_block = payload.get("optimizer")
    assert optimizer_block and optimizer_block.get("recommended_strategies") is not None

    validation_block = payload.get("validation")
    assert validation_block and validation_block.get("passed") is True
    assert validation_block.get("canonical_tokens")
    outcome = sp._validate_operator_sequence(payload["operators"])  # type: ignore[arg-type]
    assert outcome.passed
    partition_block = payload.get("partitions")
    assert partition_block, "Certificate must include partition provenance"
    aggregation_block = partition_block.get("aggregation")
    assert aggregation_block and "candidate_total" in aggregation_block
    candidates_by_partition = aggregation_block.get("partition_candidates")
    assert isinstance(candidates_by_partition, dict)
    partition_states = payload.get("partition_states")
    assert partition_states, "Certificate must include per-partition states"
    sample_state = next(iter(partition_states.values()))
    assert "before" in sample_state and "after" in sample_state
    assert payload.get("strategy_plan_snapshot")
    invariant_report = payload.get("invariant_report")
    assert invariant_report and invariant_report.get("grammar_rules")
    assert "U1" in invariant_report["grammar_rules"]
    nodal_snapshot = payload.get("nodal_decoding_snapshot")
    assert nodal_snapshot and nodal_snapshot.get("sequence") == [
        "UM",
        "RA",
        "IL",
        "THOL",
    ]
    tnfr_snapshot = payload.get("tnfr_verification_snapshot")
    assert tnfr_snapshot and tnfr_snapshot.get("criteria")
    signature_block = payload.get("tnfr_factor_signature")
    assert signature_block and signature_block.get("hash")
    assert payload.get("tnfr_factor_signature") == result.tnfr_factor_signature
    partition_files = payload.get("partition_files")
    assert partition_files, "Root certificate should reference partition files"
    partition_dir = pathlib.Path(result.partition_artifact_dir)
    assert partition_dir.exists()
    assert partition_dir.parent == partition_root
    assert payload.get("partition_directory") == partition_dir.as_posix()
    assert payload.get("partition_directory_absolute") == partition_dir.as_posix()
    for relative_path in partition_files:
        rel_obj = pathlib.Path(relative_path)
        part_path = rel_obj if rel_obj.is_absolute() else partition_dir / rel_obj.name
        assert part_path.exists()
        partition_payload = json.loads(part_path.read_text())
        assert partition_payload.get("partition_id")
        assert "candidate_factors" in partition_payload
    assert result.partition_artifact_dir
    manifest_relative = payload.get("partition_manifest")
    assert manifest_relative
    manifest_path = pathlib.Path(manifest_relative)
    if not manifest_path.is_absolute():
        manifest_path = partition_dir / manifest_path.name
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text())
    assert len(manifest_payload.get("entries", [])) == len(partition_files)
    assert result.partition_manifest_path
    assert pathlib.Path(result.partition_manifest_path) == manifest_path


def test_env_preference_selects_distributed_backend(monkeypatch: MonkeyPatch) -> None:
    fake_backend = _RecordingFFTEngine()
    monkeypatch.setenv("TNFR_FFT_BACKEND", "distributed")
    monkeypatch.setattr(
        sp, "_instantiate_distributed_backend", lambda dispatcher: fake_backend
    )

    factorizer = SpectralPaleyFactorizer()

    assert factorizer._fft_backend is fake_backend


def test_http_dispatcher_loader(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_FFT_DISPATCHER", "http://unit.test/fft")
    monkeypatch.setenv("TNFR_FFT_AUTH_TOKEN", "token123")

    captured: dict[str, Any] = {}

    import tnfr.dynamics.fft_dispatchers as dispatchers

    class _FakeHTTPDispatcher:
        def __init__(self, base_url: str, auth_token: str | None = None) -> None:
            captured["base_url"] = base_url
            captured["auth_token"] = auth_token
            self.dispatch = lambda action, payload: {
                "action": action,
                "payload": payload,
            }

    monkeypatch.setattr(dispatchers, "HTTPFFTDispatcher", _FakeHTTPDispatcher)

    dispatcher = sp._load_fft_dispatcher()

    assert dispatcher is not None
    response = dispatcher("fft", {"value": 7})
    assert response["action"] == "fft"
    assert captured["base_url"].startswith("http://unit.test/fft")


def test_trial_division_fallback_engages(monkeypatch: MonkeyPatch) -> None:
    def _empty_candidates(**_: Any) -> list[int]:
        return []

    monkeypatch.setattr(sp, "_candidate_factors", _empty_candidates)

    factorizer = SpectralPaleyFactorizer()
    result = factorizer.analyze(185)

    assert {5, 37}.issubset(result.candidate_factors)
    assert "fallback=trial-division" in result.notes


def test_failure_telemetry_records_when_no_certification(
    monkeypatch: MonkeyPatch,
) -> None:
    """Ensure failure telemetry runs when TNFR verification certifies nothing."""

    monkeypatch.setenv("TNFR_FAILURE_TELEMETRY", "0")

    def _no_certification(*_: Any, **__: Any) -> tuple[list[int], dict[str, Any]]:
        return [], {"certified": [], "per_factor": {"p0": {"endorsement_ratio": 0.1}}}

    monkeypatch.setattr(sp, "_verify_factors_tnfr", _no_certification)

    factorizer = SpectralPaleyFactorizer()

    class _FakeRecord:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def to_mapping(self) -> dict[str, Any]:
            return dict(self._payload)

    class _FakeTelemetryManager:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.payload = {"run_id": "fake-run", "telemetry": "ok"}

        def record_failure(
            self, result: sp.SpectralAnalysisResult, **kwargs: Any
        ) -> _FakeRecord:
            entry = dict(kwargs)
            entry["result"] = result
            self.calls.append(entry)
            return _FakeRecord(self.payload)

    fake_manager = _FakeTelemetryManager()
    factorizer._failure_telemetry_manager = fake_manager

    result = factorizer.analyze(185)

    assert fake_manager.calls, "failure telemetry manager should record the attempt"
    call = fake_manager.calls[0]
    assert call["result"] is result
    assert call["failure_stage"] in {"spectral", "nodal_decoding", "verification"}
    assert call["failure_reason"]
    assert result.failure_diagnostics == fake_manager.payload


@pytest.mark.parametrize(
    ("n", "expected_factors"),
    (
        (185, {5, 37}),
        (221, {13, 17}),
        (899, {29, 31}),
        (385, {5, 7, 11}),
        (1001, {7, 11, 13}),
        (343, {7}),
        (625, {5}),
        (1331, {11}),
    ),
)
def test_factorizer_surfaces_true_factors(n: int, expected_factors: set[int]) -> None:
    factorizer = SpectralPaleyFactorizer(max_nodes=2049)
    result = factorizer.analyze(n)
    found = set(result.candidate_factors)
    missing = expected_factors - found
    assert not missing, f"Missing {missing} for n={n}; candidates={sorted(found)}"
