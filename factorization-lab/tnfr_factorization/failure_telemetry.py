"""Failure telemetry instrumentation for TNFR factorization attempts."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List
import json
import math
import time
import uuid

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from .spectral_paley import SpectralAnalysisResult

__all__ = [
    "BottleneckSignal",
    "FailureTelemetryRecord",
    "FailureTelemetryManager",
]


@dataclass
class BottleneckSignal:
    """Structured description of a detected failure bottleneck."""

    code: str
    severity: str
    metric: str
    value: float
    threshold: float
    description: str

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureTelemetryRecord:
    """Full telemetry snapshot for a failed factorization attempt."""

    run_id: str
    artifact_path: str
    timestamp: float
    n: int
    modulus: int
    failure_reason: str
    failure_stage: str
    metrics: Dict[str, Any]
    bottlenecks: List[BottleneckSignal]
    recommendations: List[str]
    convergence_profile: Dict[str, Any] | None
    partition_summary: Dict[str, Any] | None
    partition_aggregation: Dict[str, Any] | None
    nodal_decoding_snapshot: Dict[str, Any] | None
    verification_report: Dict[str, Any] | None
    replay_metadata: Dict[str, Any] | None
    seed_state_path: str | None
    extra_context: Dict[str, Any] | None

    def to_mapping(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["bottlenecks"] = [signal.to_mapping() for signal in self.bottlenecks]
        return payload


class FailureTelemetryManager:
    """Manages persistence and analysis of failed factorization attempts."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        manifest_limit: int = 200,
    ) -> None:
        self.root = Path(root or Path("results") / "failure_telemetry").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_limit = max(1, manifest_limit)
        self.manifest_path = self.root / "failure_manifest.json"

    def record_failure(
        self,
        result: "SpectralAnalysisResult",
        *,
        failure_stage: str,
        failure_reason: str,
        snapshot_analysis: Dict[str, Any] | None = None,
        replay_metadata: Dict[str, Any] | None = None,
        extra_context: Dict[str, Any] | None = None,
    ) -> FailureTelemetryRecord:
        """Persist telemetry for a failed attempt and derive diagnostics."""

        run_id = self._make_run_id(result.n)
        day_dir = self.root / time.strftime("%Y-%m-%d", time.gmtime())
        day_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = day_dir / f"{run_id}.json"

        metrics = self._collect_metrics(result)
        convergence = self._build_convergence_profile(result, snapshot_analysis)
        bottlenecks = self._detect_bottlenecks(metrics, result, failure_stage, failure_reason)
        recommendations = self._recommendations_for(bottlenecks, failure_reason)

        record = FailureTelemetryRecord(
            run_id=run_id,
            artifact_path=str(artifact_path),
            timestamp=time.time(),
            n=result.n,
            modulus=result.modulus,
            failure_reason=failure_reason,
            failure_stage=failure_stage,
            metrics=metrics,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            convergence_profile=convergence,
            partition_summary=self._json_like(result.partition_summary),
            partition_aggregation=self._json_like(result.partition_aggregation),
            nodal_decoding_snapshot=self._json_like(result.nodal_decoding),
            verification_report=self._json_like(result.tnfr_verification),
            replay_metadata=self._json_like(replay_metadata),
            seed_state_path=None,
            extra_context=extra_context or {},
        )

        artifact_path.write_text(json.dumps(record.to_mapping(), indent=2))
        self._update_manifest(record)
        return record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_metrics(self, result: "SpectralAnalysisResult") -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "laplacian_gap": float(result.laplacian_gap),
            "coherence_score": float(result.coherence_score),
            "phi_s": float(result.phi_s),
            "phase_gradient": float(result.phase_gradient),
            "phase_curvature": float(result.phase_curvature),
            "coherence_length": float(result.coherence_length),
            "arithmetic_delta_nfr": float(result.arithmetic_delta_nfr),
            "candidate_count": len(result.candidate_factors or []),
            "tnfr_certified_count": len(result.tnfr_certified_factors or []),
        }

        summary = self._json_like(result.partition_summary) or {}
        aggregation = self._json_like(result.partition_aggregation) or {}
        metrics.update(
            {
                "partition_count": summary.get("partition_count"),
                "partition_coherence_ratio": aggregation.get("coherence_ratio"),
                "partition_candidate_ratio": aggregation.get("candidate_ratio"),
                "partition_coverage_ratio": aggregation.get("coverage_ratio"),
                "boundary_fraction": aggregation.get("boundary_fraction"),
                "empty_partition_count": len(aggregation.get("empty_partitions", []) or []),
            }
        )
        return metrics

    def _build_convergence_profile(
        self,
        result: "SpectralAnalysisResult",
        snapshot_analysis: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if snapshot_analysis:
            payload = dict(snapshot_analysis)
            payload.setdefault("source", "snapshot")
            return payload

        stages: List[Dict[str, Any]] = []
        stages.append(
            {
                "stage": "spectral",
                "coherence": float(result.coherence_score),
                "phi_s": float(result.phi_s),
                "phase_gradient": float(result.phase_gradient),
            }
        )
        aggregation = self._json_like(result.partition_aggregation) or {}
        stages.append(
            {
                "stage": "partitioning",
                "coherence": aggregation.get("coherence_ratio"),
                "coverage": aggregation.get("coverage_ratio"),
                "candidate_ratio": aggregation.get("candidate_ratio"),
            }
        )
        verification = self._json_like(result.tnfr_verification) or {}
        stages.append(
            {
                "stage": "verification",
                "coherence": self._average_verification_coherence(verification),
                "endorsement_rate": self._average_verification_endorsement(verification),
            }
        )

        coherence_values: List[float] = [
            float(stage.get("coherence", 0.0))
            for stage in stages
            if isinstance(stage.get("coherence"), (int, float))
        ]
        trend = None
        delta = None
        if len(coherence_values) >= 2:
            delta = coherence_values[-1] - coherence_values[0]
            trend = delta / max(len(coherence_values) - 1, 1)

        return {
            "source": "synthetic",
            "stages": stages,
            "trend": trend,
            "delta": delta,
        }

    def _detect_bottlenecks(
        self,
        metrics: Dict[str, Any],
        result: "SpectralAnalysisResult",
        failure_stage: str,
        failure_reason: str,
    ) -> List[BottleneckSignal]:
        signals: List[BottleneckSignal] = []

        def add(code: str, severity: str, metric: str, value: float, threshold: float, description: str) -> None:
            signals.append(
                BottleneckSignal(
                    code=code,
                    severity=severity,
                    metric=metric,
                    value=float(value),
                    threshold=float(threshold),
                    description=description,
                )
            )

        coherence = metrics.get("coherence_score")
        if isinstance(coherence, (int, float)) and not math.isnan(coherence):
            if coherence < 0.55:
                add("low_global_coherence", "high", "coherence_score", coherence, 0.65, "Parent state coherence collapsed below safe range")
            elif coherence < 0.65:
                add("low_global_coherence", "medium", "coherence_score", coherence, 0.65, "Parent state coherence trending low")

        gradient = metrics.get("phase_gradient")
        if isinstance(gradient, (int, float)) and not math.isnan(gradient) and gradient > 0.32:
            severity = "high" if gradient > 0.4 else "medium"
            add("phase_gradient_instability", severity, "phase_gradient", gradient, 0.32, "Phase gradient exceeded harmonic stability band")

        curvature = metrics.get("phase_curvature")
        if isinstance(curvature, (int, float)) and not math.isnan(curvature) and curvature > 2.9:
            severity = "high" if curvature > 3.2 else "medium"
            add("phase_curvature_instability", severity, "phase_curvature", curvature, 2.9, "Phase curvature drift indicates torsion bottleneck")

        coherence_length = metrics.get("coherence_length")
        if isinstance(coherence_length, (int, float)) and not math.isnan(coherence_length) and coherence_length < 1.0:
            add("coherence_length_collapse", "medium", "coherence_length", coherence_length, 1.0, "Correlation length fell below single partition span")

        delta_nfr = metrics.get("arithmetic_delta_nfr")
        if isinstance(delta_nfr, (int, float)) and abs(delta_nfr) < 1e-6:
            add("delta_nfr_flat", "medium", "arithmetic_delta_nfr", delta_nfr, 1e-6, "Arithmetic ΔNFR pressure vanished, indicating weak structural drive")

        partition_coherence = metrics.get("partition_coherence_ratio")
        if isinstance(partition_coherence, (int, float)) and partition_coherence < 0.75:
            add("partition_coherence_loss", "medium", "partition_coherence_ratio", partition_coherence, 0.75, "Partitions lost coherence relative to parent state")

        coverage_ratio = metrics.get("partition_coverage_ratio")
        if isinstance(coverage_ratio, (int, float)) and coverage_ratio < 0.65:
            add("partition_coverage_gap", "medium", "partition_coverage_ratio", coverage_ratio, 0.65, "Partitions did not cover enough of the graph")

        candidate_ratio = metrics.get("partition_candidate_ratio")
        if isinstance(candidate_ratio, (int, float)) and candidate_ratio < 0.5:
            add("candidate_signal_low", "medium", "partition_candidate_ratio", candidate_ratio, 0.5, "Too few partitions surfaced candidate factors")

        partition_count = metrics.get("partition_count") or 0
        empty_partition_count = metrics.get("empty_partition_count") or 0
        if partition_count and empty_partition_count / partition_count > 0.25:
            add(
                "empty_partition_density",
                "medium",
                "empty_partition_count",
                empty_partition_count,
                partition_count * 0.25,
                "More than 25% of partitions yielded no signal",
            )

        verification = self._json_like(result.tnfr_verification) or {}
        per_factor_raw = verification.get("per_factor")
        per_factor: Dict[str, Any] = per_factor_raw if isinstance(per_factor_raw, dict) else {}
        certified = verification.get("certified") or []
        if per_factor and not certified:
            endorsement_rates = [
                block.get("endorsement_ratio", 0.0)
                for block in per_factor.values()
                if isinstance(block, dict)
            ]
            if endorsement_rates and max(endorsement_rates) < 0.5:
                add(
                    "verification_filters_strict",
                    "medium",
                    "endorsement_ratio",
                    max(endorsement_rates),
                    0.5,
                    "All candidate partitions failed endorsement thresholds",
                )

        if failure_stage == "spectral" and metrics.get("candidate_count", 0) == 0:
            add(
                "no_candidate_clusters",
                "high",
                "candidate_count",
                0.0,
                1.0,
                "Spectral analysis failed to generate any candidate factors",
            )

        if failure_stage == "nodal_decoding" and not result.nodal_decoding:
            add(
                "nodal_decoder_inactive",
                "high",
                "nodal_decoding",
                0.0,
                1.0,
                "Nodal decoder did not emit partitions for verification",
            )

        if not signals:
            add("unknown_failure", "low", "coherence_score", metrics.get("coherence_score", 0.0), 0.0, failure_reason)
        return signals

    def _recommendations_for(
        self,
        bottlenecks: List[BottleneckSignal],
        failure_reason: str,
    ) -> List[str]:
        mapping = {
            "low_global_coherence": "Insert IL/THOL stabilization sweep before partitioning or increase partition overlap",
            "phase_gradient_instability": "Throttle destabilizers and verify |∇φ| bounds before coupling",
            "phase_curvature_instability": "Use NAV/REMESH to redistribute curvature before verification",
            "coherence_length_collapse": "Increase modulus or reduce target partition size to regain ξ_C",
            "delta_nfr_flat": "Inject OZ/ZHIR exploration before IL to create ΔNFR pressure",
            "partition_coherence_loss": "Re-plan partitions with higher overlap or run auto_optimize() on weak blocks",
            "partition_coverage_gap": "Increase target partition size or ensure candidate assignment covers entire modulus",
            "candidate_signal_low": "Enable fallback candidate sources (spectral cascades or arithmetic hints)",
            "empty_partition_density": "Review partition telemetry; prune partitions with low Φ_s before verification",
            "verification_filters_strict": "Inspect tnfr_verification report and adjust criteria or add stabilizers",
            "no_candidate_clusters": "Check Laplacian gap assumptions; consider alternative modulus or dissonance probes",
            "nodal_decoder_inactive": "Ensure nodal decoder received operator strategy plan and dynamic factors",
        }
        recs: List[str] = []
        for signal in bottlenecks:
            suggestion = mapping.get(signal.code)
            if suggestion and suggestion not in recs:
                recs.append(suggestion)
        if not recs:
            recs.append(f"Review failure reason: {failure_reason}")
        return recs

    def _average_verification_coherence(self, verification: Dict[str, Any]) -> float | None:
        per_factor_raw = verification.get("per_factor")
        per_factor: Dict[str, Any] = per_factor_raw if isinstance(per_factor_raw, dict) else {}
        samples: List[float] = []
        for block in per_factor.values():
            if not isinstance(block, dict):
                continue
            span = block.get("coherence_span")
            if isinstance(span, list) and len(span) == 2:
                samples.append(sum(float(value) for value in span) / 2.0)
        if not samples:
            return None
        return float(sum(samples) / len(samples))

    def _average_verification_endorsement(self, verification: Dict[str, Any]) -> float | None:
        per_factor_raw = verification.get("per_factor")
        per_factor: Dict[str, Any] = per_factor_raw if isinstance(per_factor_raw, dict) else {}
        ratios: List[float] = []
        for block in per_factor.values():
            if not isinstance(block, dict):
                continue
            ratio = block.get("endorsement_ratio")
            if isinstance(ratio, (int, float)):
                ratios.append(float(ratio))
        if not ratios:
            return None
        return float(sum(ratios) / len(ratios))

    def _update_manifest(self, record: FailureTelemetryRecord) -> None:
        manifest: Dict[str, Any] = {"version": "1.0", "records": []}
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
            except Exception:
                manifest = {"version": "1.0", "records": []}
        entries = manifest.setdefault("records", [])
        entries.append(
            {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "n": record.n,
                "modulus": record.modulus,
                "failure_reason": record.failure_reason,
                "failure_stage": record.failure_stage,
                "bottlenecks": [signal.code for signal in record.bottlenecks],
                "artifact_path": record.artifact_path,
            }
        )
        manifest["records"] = entries[-self.manifest_limit :]
        self.manifest_path.write_text(json.dumps(manifest, indent=2))

    def _json_like(self, payload: Any) -> Dict[str, Any] | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            return json.loads(json.dumps(payload))
        return None

    def _make_run_id(self, n: int) -> str:
        suffix = uuid.uuid4().hex[:12]
        return f"failure_{n}_{suffix}"
