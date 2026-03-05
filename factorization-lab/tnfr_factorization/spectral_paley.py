"""Spectral Paley-based TNFR factorization utilities."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple, cast

import networkx as nx
import numpy as np

from tnfr.cache import CacheLevel, cache_tnfr_computation
from tnfr.config.operator_names import CANONICAL_OPERATOR_NAMES
from tnfr.dynamics.advanced_fft_arithmetic import TNFRAdvancedFFTEngine
from tnfr.dynamics.fft_backend import FFTBackend
from tnfr.mathematics.number_theory import (
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    ArithmeticTNFRParameters,
)
from tnfr.operators.grammar import SequenceValidationResult, validate_sequence
from tnfr.operators.strategies import (
    ResourceEstimate,
    StrategyContext,
    StrategyRegistry,
    StructuralFields,
)
from .partitioning import (
    PartitionPlannerConfig,
    PartitionedPaleyGraph,
    PaleyPartition,
    aggregate_partition_metrics,
    annotate_partition_candidates,
    plan_paley_partitions,
)
from .self_opt_support import (
    attach_self_opt_sequences,
    run_partition_self_optimization,
)
from .failure_telemetry import FailureTelemetryManager

# Certificate versioning and hashing constants
_CERTIFICATE_VERSION = "1.0"
_PARTITION_HASH_ALGORITHM = "sha256"
_REPLAY_METADATA_VERSION = "1.0"

Dispatcher = Callable[[str, Dict[str, Any]], Any]


def _generate_partition_hash_chain(partitioning: PartitionedPaleyGraph | None) -> Dict[str, Any] | None:
    """Generate reproducible hash chain for all partitions.
    
    Creates deterministic hash chain linking partition states, telemetry,
    and structural transitions for complete reproducibility validation.
    """
    if not partitioning:
        return None
        
    partitions = list(partitioning.iter_partitions())
    if not partitions:
        return None
    
    partition_hashes = []
    chain_data = {
        "algorithm": _PARTITION_HASH_ALGORITHM,
        "version": _CERTIFICATE_VERSION,
        "partition_count": len(partitions),
        "total_nodes": sum(len(p.node_indices) for p in partitions),
        "partitions": []
    }
    
    for partition in partitions:
        # Deterministic partition state serialization
        partition_state = {
            "partition_id": partition.partition_id,
            "node_indices": sorted(partition.node_indices),
            "boundary_nodes": sorted(partition.boundary_nodes),
            "candidate_factors": sorted(partition.candidate_factors),
            "telemetry": {
                "phi_s": partition.telemetry.phi_s if partition.telemetry else None,
                "phase_gradient": partition.telemetry.phase_gradient if partition.telemetry else None,
                "phase_curvature": partition.telemetry.phase_curvature if partition.telemetry else None,
                "coherence_length": partition.telemetry.coherence_length if partition.telemetry else None,
                "notes": partition.telemetry.notes if partition.telemetry else ""
            },
            "metadata": partition.metadata if partition.metadata else None
        }
        
        # Generate deterministic hash for this partition
        partition_blob = json.dumps(partition_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        partition_hash = hashlib.sha256(partition_blob).hexdigest()
        partition_hashes.append(partition_hash)
        
        chain_data["partitions"].append({
            "partition_id": partition.partition_id,
            "hash": partition_hash,
            "node_count": len(partition.node_indices),
            "boundary_count": len(partition.boundary_nodes),
            "factor_count": len(partition.candidate_factors)
        })
    
    # Generate chain hash linking all partitions
    chain_blob = json.dumps({
        "partition_hashes": partition_hashes,
        "algorithm": _PARTITION_HASH_ALGORITHM,
        "version": _CERTIFICATE_VERSION
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    
    chain_data["chain_hash"] = hashlib.sha256(chain_blob).hexdigest()
    return chain_data


def _generate_replay_metadata(
    result: Any,
    arithmetic: Any | None = None,
    partitioning: PartitionedPaleyGraph | None = None
) -> Dict[str, Any]:
    """Generate comprehensive replay metadata for exact reproducibility.
    
    Captures all seeds, backend parameters, and configuration needed
    to reproduce the exact factorization attempt.
    """
    metadata = {
        "version": _REPLAY_METADATA_VERSION,
        "timestamp": time.time(),
        "algorithm_version": _CERTIFICATE_VERSION,
        "environment": {
            "pure_mode": os.getenv("TNFR_PURE_MODE", "false").lower() in {"1", "true", "yes", "on"},
            "pure_mode_verify_divisibility": os.getenv("TNFR_PURE_MODE_VERIFY_DIVISIBILITY", "false").lower() in {"1", "true", "yes", "on"},
        },
        "parameters": {
            "n": getattr(result, 'n', None),
            "modulus": getattr(result, 'modulus', None),
            "node_count": getattr(result, 'node_count', None),
            "edge_count": getattr(result, 'edge_count', None)
        },
        "backend": {
            "fft_backend": getattr(result, 'fft_backend', None),
            "fft_capabilities": getattr(result, 'fft_capabilities', None)
        }
    }
    
    # Add arithmetic seeds if available
    if arithmetic:
        metadata["arithmetic"] = {
            "delta_nfr": getattr(arithmetic, 'delta_nfr', None),
            "local_coherence": getattr(arithmetic, 'local_coherence', None),
            "components": getattr(arithmetic, 'components', {})
        }
    
    return metadata


def _capture_deterministic_seeds() -> Dict[str, Any]:
    """Capture all RNG seeds and spectral parameters for deterministic replay.
    
    Records numpy random state, backend-specific seeds, and any other
    sources of non-determinism for complete reproducibility.
    """
    seeds: Dict[str, Any] = {
        "capture_timestamp": time.time(),
        "numpy_random_state": None,
        "environment_seeds": {},
        "backend_seeds": {}
    }
    
    # Capture numpy random state
    try:
        random_state = np.random.get_state()
        # Convert state to JSON-serializable format
        if isinstance(random_state, tuple) and len(random_state) >= 5:
            seeds["numpy_random_state"] = {
                "state_type": str(random_state[0]),
                "state_array": random_state[1].tolist() if random_state[1] is not None else None,
                "state_pos": int(random_state[2]) if random_state[2] is not None else None,
                "state_has_gauss": int(random_state[3]) if random_state[3] is not None else None,
                "state_cached_gaussian": float(random_state[4]) if random_state[4] is not None else None
            }
    except Exception as e:
        seeds["numpy_random_state_error"] = str(e)
    
    # Capture environment-based seeds
    seed_envs = [
        "PYTHONHASHSEED", "TNFR_SPECTRAL_SEED", "TNFR_FFT_SEED",
        "TNFR_PARTITION_SEED", "NUMPY_SEED", "RANDOM_SEED"
    ]
    for env_var in seed_envs:
        value = os.getenv(env_var)
        if value is not None:
            seeds["environment_seeds"][env_var] = value
    
    return seeds


_DISPATCHER_METADATA_ATTR = "__tnfr_dispatcher_metadata__"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DEFAULT_CERTIFICATE_DIR = _RESULTS_ROOT / "certificates"
_DEFAULT_PARTITION_ROOT = _DEFAULT_CERTIFICATE_DIR / "partitioned"
_PARTITION_OUTPUT_ENV = "TNFR_PARTITION_OUTPUT_DIR"
_PARTITION_FILELIST_THRESHOLD_ENV = "TNFR_PARTITION_FILELIST_THRESHOLD"
_DEFAULT_FILELIST_THRESHOLD = 1000
_MANIFEST_FILENAME = "_manifest.json"
_MANIFEST_SUMMARY_FILENAME = "_manifest_summary.json"
_PARTITION_FILELIST_ARCHIVE = "_partition_files.txt.gz"

_FAILURE_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

_FALLBACK_MAX_DIVISOR_ENV = "TNFR_FACTOR_FALLBACK_MAX_DIVISOR"

_DEFAULT_MAX_NODES = 4097


def _annotate_dispatcher(dispatcher: Dispatcher, metadata: Mapping[str, Any]) -> None:
    payload = dict(metadata)
    try:
        setattr(dispatcher, _DISPATCHER_METADATA_ATTR, payload)
        return
    except AttributeError:
        pass
    func = getattr(dispatcher, "__func__", None)
    if func is not None:
        try:
            setattr(func, _DISPATCHER_METADATA_ATTR, payload)
        except AttributeError:
            return


def _default_certificate_dir() -> Path:
    return _DEFAULT_CERTIFICATE_DIR


def _partition_filelist_threshold() -> int:
    threshold = _read_threshold_env(_PARTITION_FILELIST_THRESHOLD_ENV, _DEFAULT_FILELIST_THRESHOLD)
    if threshold is None:
        return _DEFAULT_FILELIST_THRESHOLD
    return max(1, threshold)


def _resolve_partition_directory(base_name: str) -> tuple[Path, Path | None]:
    target_root = os.getenv(_PARTITION_OUTPUT_ENV)
    if target_root:
        partition_root = Path(target_root).expanduser()
        if not partition_root.is_absolute():
            partition_root = (_REPO_ROOT / target_root).resolve()
    else:
        partition_root = _DEFAULT_PARTITION_ROOT
    partition_root.mkdir(parents=True, exist_ok=True)
    partition_dir = partition_root / base_name
    relative_dir: Path | None = None
    try:
        relative_dir = partition_dir.relative_to(_default_certificate_dir())
    except ValueError:
        relative_dir = None
    return partition_dir, relative_dir


try:
    from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
except Exception:  # pragma: no cover - optional dependency in trimmed installs
    TNFRSelfOptimizingEngine = None  # type: ignore[assignment]


@dataclass
class SpectralAnalysisResult:
    """Container for spectral metrics, TNFR telemetry, and factor hints."""

    n: int
    modulus: int
    node_count: int
    edge_count: int
    laplacian_gap: float
    candidate_factors: List[int]
    coherence_score: float
    phi_s: float
    phase_gradient: float
    phase_curvature: float
    coherence_length: float
    arithmetic_delta_nfr: float
    arithmetic_terms: ArithmeticStructuralTerms
    arithmetic_components: Dict[str, float]
    arithmetic_local_coherence: float
    notes: str
    certificate_path: str | None = None
    optimizer_metadata: Dict[str, Any] | None = None
    fft_backend: str | None = None
    fft_capabilities: Dict[str, Any] | None = None
    partition_summary: Dict[str, Any] | None = None
    partition_aggregation: Dict[str, Any] | None = None
    partition_artifact_dir: str | None = None
    partition_manifest_path: str | None = None
    partition_manifest_index_path: str | None = None
    partition_file_archive_path: str | None = None
    dispatcher_telemetry: Dict[str, Any] | None = None
    operator_strategy_plan: Dict[str, Any] | None = None
    nodal_decoding: Dict[str, Any] | None = None
    tnfr_certified_factors: List[int] | None = None
    tnfr_verification: Dict[str, Any] | None = None
    tnfr_factor_signature: Dict[str, Any] | None = None
    tnfr_composite_signature: Dict[str, Any] | None = None  # multi-factor / power structure
    self_optimization_summary: Dict[str, Any] | None = None
    failure_diagnostics: Dict[str, Any] | None = None


@dataclass
class OperatorCertificate:
    """Recorded operator sequence evidence for a factor claim."""

    n: int
    candidate_factor: int | None
    operators: List[str]
    canonical_operators: List[str]
    telemetry: Dict[str, float]
    timestamp: float
    notes: str
    validation: Dict[str, Any] | None = None
    optimizer: Dict[str, Any] | None = None
    partitions: Dict[str, Any] | None = None
    partition_directory: str | None = None
    partition_directory_absolute: str | None = None
    partition_files: List[str] | None = None
    partition_manifest: str | None = None
    partition_manifest_index: str | None = None
    partition_file_archive: str | None = None
    partition_states: Dict[str, Any] | None = None
    strategy_plan_snapshot: Dict[str, Any] | None = None
    invariant_report: Dict[str, Any] | None = None
    nodal_decoding_snapshot: Dict[str, Any] | None = None
    tnfr_verification_snapshot: Dict[str, Any] | None = None
    tnfr_factor_signature: Dict[str, Any] | None = None
    tnfr_composite_signature: Dict[str, Any] | None = None
    self_optimization_summary: Dict[str, Any] | None = None
    certificate_hash_chain: Dict[str, Any] | None = None
    replay_metadata: Dict[str, Any] | None = None


@dataclass
class OperatorSequenceSelection:
    """Optimized operator sequences aligned with TNFR auto-optimization."""

    canonical_sequence: List[str]
    glyph_sequence: List[str]
    optimizer_metadata: Dict[str, Any]


@dataclass
class PartitionManifestArtifacts:
    """Paths for manifest-related artifacts emitted during partition export."""

    manifest_relative: str | None = None
    manifest_absolute: Path | None = None
    summary_relative: str | None = None
    summary_absolute: Path | None = None
    archive_relative: str | None = None
    archive_absolute: Path | None = None


class SpectralPaleyFactorizer:
    """Paley-gap-driven TNFR factorization prototype."""

    def __init__(
        self,
        *,
        max_nodes: int | None = _DEFAULT_MAX_NODES,
        fft_engine: FFTBackend | None = None,
        failure_telemetry: bool | None = None,
        failure_telemetry_root: Path | None = None,
    ) -> None:
        self._history: List[SpectralAnalysisResult] = []
        self._max_nodes = max_nodes
        self._fft_backend: FFTBackend = fft_engine or _select_fft_backend(max_nodes)
        self._partition_config = _partition_config_from_env()
        self._optimizer: Optional[TNFRSelfOptimizingEngine] = None
        self._optimizer_disabled = os.getenv("TNFR_DISABLE_OPTIMIZER", "").lower() in {
            "1",
            "true",
            "on",
        }
        env_failure = os.getenv("TNFR_FAILURE_TELEMETRY")
        if failure_telemetry is None:
            failure_telemetry = env_failure not in {"0", "false", "False", "off", "OFF"}
        self._failure_telemetry_manager: FailureTelemetryManager | None = None
        if failure_telemetry:
            try:
                self._failure_telemetry_manager = FailureTelemetryManager(failure_telemetry_root)
            except Exception:
                self._failure_telemetry_manager = None
        if TNFRSelfOptimizingEngine is not None and not self._optimizer_disabled:
            try:
                self._optimizer = TNFRSelfOptimizingEngine()
            except Exception:  # pragma: no cover - fallback if engine bootstrap fails
                self._optimizer = None

    def analyze(
        self,
        n: int,
        *,
        modulus: int | None = None,
        trace_certificates: bool = False,
        certificate_dir: Path | None = None,
    ) -> SpectralAnalysisResult:
        """Analyze ``n`` via Paley-style Laplacian spectrum and TNFR telemetry."""

        if n <= 1:
            raise ValueError("n must be > 1")

        auto_even_hint = []
        working_modulus = modulus or self._derive_modulus(n)

        if self._max_nodes is not None and working_modulus > self._max_nodes:
            raise ValueError(
                f"Requested modulus {working_modulus} exceeds max_nodes={self._max_nodes}. "
                "Provide a smaller modulus or increase the limit."
            )

        if n % 2 == 0:
            auto_even_hint.append(2)

        graph = _build_paley_graph(working_modulus)
        _annotate_graph_for_fft(graph)

        spectral_state = None
        backend_name: str | None = None
        backend_capabilities: Dict[str, Any] | None = None
        dispatcher_telemetry: Dict[str, Any] | None = None
        if self._fft_backend is not None:
            try:
                spectral_state = self._fft_backend.get_spectral_state(graph)
            except Exception:
                spectral_state = None
            else:
                eigenvalues = spectral_state.eigenvalues
                backend_name = getattr(
                    self._fft_backend,
                    "backend_name",
                    self._fft_backend.__class__.__name__,
                )
                get_capabilities = getattr(self._fft_backend, "get_capabilities", None)
                if callable(get_capabilities):
                    try:
                        capabilities_obj = get_capabilities()
                    except Exception:
                        capabilities_obj = None
                    else:
                        backend_capabilities = _json_safe(asdict(capabilities_obj))
                get_dispatcher_info = getattr(self._fft_backend, "get_dispatcher_telemetry", None)
                if callable(get_dispatcher_info):
                    try:
                        dispatcher_telemetry = _json_safe(get_dispatcher_info())
                    except Exception:
                        dispatcher_telemetry = None
        if spectral_state is None:
            eigenvalues = _laplacian_eigenvalues(graph)

        laplacian_gap = _first_positive_eigenvalue(eigenvalues)

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        phi_s = _structural_potential(node_count, edge_count)
        phase_gradient = laplacian_gap / max(node_count, 1)
        phase_curvature = (eigenvalues[-1] if eigenvalues.size else 0.0) / max(node_count, 1)
        coherence_length = (
            spectral_state.coherence_length
            if spectral_state is not None and spectral_state.coherence_length > 0
            else _coherence_length(laplacian_gap)
        )
        coherence_score = _coherence_score(phi_s, phase_gradient, phase_curvature, coherence_length)
        arithmetic = _compute_arithmetic_telemetry(n)

        candidates = _candidate_factors(
            n=n,
            hints=auto_even_hint,
            gap=laplacian_gap,
            modulus=working_modulus,
            arithmetic=arithmetic,
        )

        fallback_used = False
        if not candidates:
            fallback = _fallback_factor_candidates(n)
            if fallback:
                fallback_used = True
                candidates = sorted({*candidates, *fallback})

        sequence_selection = self._select_operator_sequence(graph)

        partitioning = self._plan_partitions(
            graph,
            working_modulus,
            phi_s=phi_s,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length,
        )

        annotate_partition_candidates(partitioning, candidates)

        nodal_decoding = self._decode_partitions_nodally(
            partitioning=partitioning,
            graph=graph,
            n=n,
            modulus=working_modulus,
            parent_gap=laplacian_gap,
            parent_phi_s=phi_s,
            parent_phase_gradient=phase_gradient,
            parent_phase_curvature=phase_curvature,
            parent_coherence_length=coherence_length,
        )

        if nodal_decoding:
            dynamic_factors = nodal_decoding.get("dynamic_factors") or []
            if dynamic_factors:
                candidates = sorted({*candidates, *dynamic_factors})

        tnfr_certified_factors: List[int] = []
        tnfr_verification: Dict[str, Any] | None = None
        if partitioning and nodal_decoding:
            tnfr_certified_factors, tnfr_verification = _verify_factors_tnfr(
                n=n,
                partitioning=partitioning,
                nodal_decoding=nodal_decoding,
                parent_phi_s=phi_s,
                parent_phase_gradient=phase_gradient,
                parent_phase_curvature=phase_curvature,
            )
            if tnfr_certified_factors:
                candidates = sorted({*candidates, *tnfr_certified_factors})

        # Generate reproducibility metadata for certificates
        partition_hash_chain = _generate_partition_hash_chain(partitioning)
        # Create temporary result for replay metadata generation
        temp_result = type('TempResult', (), {
            'n': n, 'modulus': working_modulus, 'node_count': node_count, 'edge_count': edge_count,
            'fft_backend': backend_name, 'fft_capabilities': backend_capabilities
        })()
        replay_metadata = _generate_replay_metadata(temp_result, arithmetic, partitioning)
        
        # Add deterministic seeds to replay metadata
        if replay_metadata:
            replay_metadata["deterministic_seeds"] = _capture_deterministic_seeds()
        
        tnfr_signature = _build_factor_signature(n, tnfr_verification, partition_hash_chain, replay_metadata)
        tnfr_composite_signature = _build_composite_signature(n, tnfr_verification, partition_hash_chain, replay_metadata)

        if tnfr_certified_factors:
            tnfr_order = sorted({int(value) for value in tnfr_certified_factors})
            remaining = [value for value in candidates if value not in tnfr_order]
            candidates = tnfr_order + remaining

        partition_aggregation = aggregate_partition_metrics(
            partitioning,
            parent_phi_s=phi_s,
            parent_phase_gradient=phase_gradient,
            parent_phase_curvature=phase_curvature,
            parent_coherence_length=coherence_length,
            total_candidate_count=len(candidates),
        )

        note_bits = [
            f"modulus={working_modulus} (auto {'yes' if modulus is None else 'no'})",
            f"nodes={node_count}",
            f"laplacian_gap={laplacian_gap:.4f}",


            f"ΔNFR={arithmetic.delta_nfr:.3e}",
            "even factor 2 flagged" if auto_even_hint else "",
            "candidates=" + ",".join(str(c) for c in candidates) if candidates else "no nodal factors",
        ]
        if fallback_used:
            note_bits.append("fallback=trial-division")
        notes = "; ".join(bit for bit in note_bits if bit)

        strategy_plan = self._plan_operator_strategies(
            partitioning,
            sequence_selection,
            phi_s=phi_s,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length,
            backend_capabilities=backend_capabilities,
        )

        result = SpectralAnalysisResult(
            n=n,
            modulus=working_modulus,
            node_count=node_count,
            edge_count=edge_count,
            laplacian_gap=laplacian_gap,
            candidate_factors=candidates,
            coherence_score=coherence_score,
            phi_s=phi_s,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length,
            arithmetic_delta_nfr=arithmetic.delta_nfr,
            arithmetic_terms=arithmetic.terms,
            arithmetic_components=arithmetic.components,
            arithmetic_local_coherence=arithmetic.local_coherence,
            notes=notes,
            optimizer_metadata=_json_safe(sequence_selection.optimizer_metadata),
            fft_backend=backend_name,
            fft_capabilities=backend_capabilities,
            partition_summary=_json_safe(partitioning.summary()) if partitioning else None,
            partition_aggregation=_json_safe(partition_aggregation.to_mapping())
            if partition_aggregation
            else None,
            dispatcher_telemetry=dispatcher_telemetry,
            operator_strategy_plan=_json_safe(strategy_plan) if strategy_plan else None,
            nodal_decoding=_json_safe(nodal_decoding) if nodal_decoding else None,
            tnfr_certified_factors=list(tnfr_certified_factors) if tnfr_certified_factors else None,
            tnfr_verification=_json_safe(tnfr_verification) if tnfr_verification else None,
            tnfr_factor_signature=_json_safe(tnfr_signature) if tnfr_signature else None,
            tnfr_composite_signature=_json_safe(tnfr_composite_signature) if tnfr_composite_signature else None,
        )

        if trace_certificates:
            certificate_path, partition_dir, manifest_artifacts = _emit_certificate(
                result=result,
                arithmetic=arithmetic,
                certificate_dir=certificate_dir,
                sequence=sequence_selection,
                partitioning=partitioning,
            )
            result.certificate_path = str(certificate_path)
            if partition_dir is not None:
                result.partition_artifact_dir = str(partition_dir)
            if manifest_artifacts:
                if manifest_artifacts.manifest_absolute is not None:
                    result.partition_manifest_path = str(manifest_artifacts.manifest_absolute)
                if manifest_artifacts.summary_absolute is not None:
                    result.partition_manifest_index_path = str(manifest_artifacts.summary_absolute)
                if manifest_artifacts.archive_absolute is not None:
                    result.partition_file_archive_path = str(manifest_artifacts.archive_absolute)

        if self._failure_telemetry_manager and not tnfr_certified_factors:
            stage, reason = self._classify_failure_context(
                candidates=candidates,
                nodal_decoding=nodal_decoding,
                tnfr_verification=tnfr_verification,
            )
            extra_context = {
                "fallback_used": fallback_used,
                "auto_even_hint": bool(auto_even_hint),
            }
            try:
                record = self._failure_telemetry_manager.record_failure(
                    result,
                    failure_stage=stage,
                    failure_reason=reason,
                    snapshot_analysis=None,
                    replay_metadata=replay_metadata,
                    extra_context=extra_context,
                )
            except Exception:
                result.failure_diagnostics = {
                    "failure_stage": stage,
                    "failure_reason": reason,
                    "telemetry_status": "recording_failed",
                }
            else:
                result.failure_diagnostics = record.to_mapping()

        self._history.append(result)
        return result

    def _classify_failure_context(
        self,
        *,
        candidates: Sequence[int],
        nodal_decoding: Mapping[str, Any] | None,
        tnfr_verification: Mapping[str, Any] | None,
    ) -> Tuple[str, str]:
        if not candidates:
            return "spectral", "no_candidate_clusters"
        if not nodal_decoding:
            return "nodal_decoding", "nodal_decoder_inactive"
        if tnfr_verification:
            return "verification", "verification_rejection"
        return "verification", "verification_unavailable"

    def _plan_partitions(
        self,
        graph: nx.Graph,
        modulus: int,
        *,
        phi_s: float,
        phase_gradient: float,
        phase_curvature: float,
        coherence_length: float,
    ) -> PartitionedPaleyGraph:
        config = self._partition_config
        if config and isinstance(config.notes, str) and config.notes.startswith("auto"):
            adaptive_target = max(8, modulus // 4)
            adaptive_target = min(config.target_size, adaptive_target) if adaptive_target else config.target_size
            if adaptive_target != config.target_size:
                config = PartitionPlannerConfig(
                    target_size=adaptive_target,
                    boundary_overlap=config.boundary_overlap,
                    notes=f"{config.notes}|auto_scaled",
                )
        return plan_paley_partitions(
            graph,
            modulus,
            phi_s=phi_s,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length,
            config=config,
        )

    def _decode_partitions_nodally(
        self,
        *,
        partitioning: PartitionedPaleyGraph,
        graph: nx.Graph,
        n: int,
        modulus: int,
        parent_gap: float,
        parent_phi_s: float,
        parent_phase_gradient: float,
        parent_phase_curvature: float,
        parent_coherence_length: float,
    ) -> Dict[str, Any] | None:
        partitions = list(partitioning.iter_partitions())
        if not partitions:
            return None

        outcomes: List[Dict[str, Any]] = []
        dynamic_factors: List[int] = []
        assigned_factors: Dict[int, str] = {}

        for partition in partitions:
            outcome = _evaluate_partition_nodal_sequence(
                partition=partition,
                graph=graph,
                modulus=modulus,
                n=n,
                parent_gap=parent_gap,
                parent_phi_s=parent_phi_s,
                parent_phase_gradient=parent_phase_gradient,
                parent_phase_curvature=parent_phase_curvature,
                parent_coherence_length=parent_coherence_length,
            )

            factor = outcome.get("inferred_factor")
            if factor is not None and 1 < factor < n:
                if factor not in assigned_factors:
                    assigned_factors[factor] = partition.partition_id
                    if factor not in partition.candidate_factors:
                        partition.candidate_factors.append(factor)
                    dynamic_factors.append(factor)
                    outcome["factor_assignment"] = "primary"
                else:
                    outcome["factor_assignment"] = assigned_factors[factor]

            partition.metadata.setdefault("nodal_state", {})
            partition.metadata["nodal_state"] = {
                "dnfr_after": outcome.get("dnfr_after"),
                "coherence_ratio": outcome.get("coherence_ratio"),
                "inferred_factor": outcome.get("inferred_factor"),
                "sequence": list(_NODAL_OPERATOR_SEQUENCE),
            }

            outcomes.append(outcome)

        converged = sum(1 for outcome in outcomes if outcome.get("dnfr_converged"))
        return {
            "sequence": list(_NODAL_OPERATOR_SEQUENCE),
            "dynamic_factors": sorted(set(dynamic_factors)),
            "converged_partitions": converged,
            "partitions": outcomes,
            "notes": "nodal-decoder",
        }

    def _derive_modulus(self, n: int) -> int:
        """Lift ``n`` to the nearest Paley-compatible modulus (odd and 1 mod 4)."""

        candidate = n if n % 2 else n + 1
        candidate = max(candidate, 5)
        return _lift_to_paley_modulus(candidate)

    def _select_operator_sequence(self, graph: nx.Graph) -> OperatorSequenceSelection:
        """Use the TNFR self-optimizing engine to choose operator sequences."""

        if TNFRSelfOptimizingEngine is None or self._optimizer is None:
            reason = "optimizer_disabled" if getattr(self, "_optimizer_disabled", False) else "optimizer_unavailable"
            return _default_sequence_selection(reason)

        try:
            recommendations = self._optimizer.recommend_optimization_strategy(
                graph,
                "paley_factorization",
            )
            canonical_sequence = _sequence_from_recommendations(
                recommendations.recommended_strategies
            ) or list(_DEFAULT_CANONICAL_SEQUENCE)
            metadata = _json_safe(
                {
                    "recommended_strategies": recommendations.recommended_strategies,
                    "predicted_speedups": recommendations.predicted_speedups,
                    "optimization_improvements": recommendations.optimization_improvements,
                    "adaptive_configuration": recommendations.adaptive_configurations,
                    "execution_time": recommendations.execution_time,
                    "mathematical_insights": recommendations.mathematical_insights,
                }
            )
            glyph_sequence = _canonical_to_glyph_sequence(canonical_sequence)
            return OperatorSequenceSelection(
                canonical_sequence=canonical_sequence,
                glyph_sequence=glyph_sequence,
                optimizer_metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback path
            return _default_sequence_selection("optimizer_error", str(exc))

    def _plan_operator_strategies(
        self,
        partitioning: PartitionedPaleyGraph,
        sequence: OperatorSequenceSelection,
        *,
        phi_s: float,
        phase_gradient: float,
        phase_curvature: float,
        coherence_length: float,
        backend_capabilities: Dict[str, Any] | None,
    ) -> Dict[str, Any] | None:
        tracked_glyphs = {"AL", "IL", "RA", "SHA"}
        glyph_sequence = [glyph for glyph in sequence.glyph_sequence if glyph in tracked_glyphs]
        if not glyph_sequence:
            return None
        registry_snapshot = {glyph: StrategyRegistry.get(glyph) for glyph in tracked_glyphs}
        if all(not bucket for bucket in registry_snapshot.values()):
            return None

        dispatcher_caps = backend_capabilities or {}
        parent_fields = StructuralFields(
            phi_s=phi_s,
            phase_gradient=phase_gradient,
            phase_curvature=phase_curvature,
            coherence_length=coherence_length,
        )
        plan: Dict[str, Any] = {"sequence": list(sequence.glyph_sequence), "per_partition": {}}

        for partition in partitioning.iter_partitions():
            block_entries: List[Dict[str, Any]] = []
            telemetry = partition.telemetry
            fields = parent_fields
            if telemetry is not None:
                fields = StructuralFields(
                    phi_s=telemetry.phi_s or parent_fields.phi_s,
                    phase_gradient=telemetry.phase_gradient or parent_fields.phase_gradient,
                    phase_curvature=telemetry.phase_curvature or parent_fields.phase_curvature,
                    coherence_length=telemetry.coherence_length or parent_fields.coherence_length,
                )

            for idx, glyph in enumerate(glyph_sequence):
                bucket = registry_snapshot.get(glyph)
                if not bucket:
                    continue
                ctx = StrategyContext(
                    partition_id=partition.partition_id,
                    operator_sequence_position=idx,
                    structural_fields=fields,
                    dispatcher_capabilities=dispatcher_caps,
                    backend="cpu",
                    block_size=len(partition.node_indices),
                    boundary_overlap=len(partition.boundary_nodes),
                    seed=(hash((partition.partition_id, glyph, idx)) & 0xFFFFFFFF),
                )
                selection = _select_strategy_candidate(bucket, ctx)
                if selection is None:
                    continue
                name, estimate = selection
                block_entries.append(
                    {
                        "operator": glyph,
                        "strategy": name,
                        "resource_estimate": asdict(estimate),
                    }
                )
            if block_entries:
                plan["per_partition"][partition.partition_id] = block_entries

        if not plan["per_partition"]:
            return None
        return plan

    @property
    def history(self) -> List[SpectralAnalysisResult]:
        """Return previous analyses."""

        return list(self._history)


def _lift_to_paley_modulus(value: int) -> int:
    """Return the next integer >= value that is congruent to 1 mod 4."""

    if value % 2 == 0:
        value += 1
    while value % 4 != 1:
        value += 2
    return value


def _build_paley_graph(modulus: int) -> nx.Graph:
    """Construct a generalized Paley graph via the Jacobi symbol criterion."""

    if modulus < 5:
        raise ValueError("Paley graphs require modulus >= 5")

    residues = {r for r in range(1, modulus) if _jacobi_symbol(r, modulus) == 1}
    graph = nx.Graph()
    graph.add_nodes_from(range(modulus))

    for i in range(modulus):
        for j in range(i + 1, modulus):
            diff = (j - i) % modulus
            if diff and diff in residues:
                graph.add_edge(i, j)

    return graph


def _annotate_graph_for_fft(graph: nx.Graph) -> None:
    """Attach minimal EPI/νf signals so FFT engine can operate."""

    if graph.number_of_nodes() == 0:
        return

    max_degree = max(dict(graph.degree()).values()) or 1
    for node, degree in graph.degree():
        graph.nodes[node]["EPI"] = degree / max_degree
        graph.nodes[node]["nu_f"] = 1.0


def _jacobi_symbol(a: int, n: int) -> int:
    """Compute the Jacobi symbol (a/n)."""

    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd number")

    a = a % n
    result = 1

    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result

        a, n = n, a

        if a % 4 == 3 and n % 4 == 3:
            result = -result

        a %= n

    return result if n == 1 else 0


def _laplacian_eigenvalues(graph: nx.Graph) -> np.ndarray:
    """Return the Laplacian eigenvalues as a dense NumPy array."""

    try:
        laplacian_matrix = nx.laplacian_matrix(graph, dtype=float)
    except TypeError:  # NetworkX >=3.4 removed the dtype keyword
        laplacian_matrix = nx.laplacian_matrix(graph)

    laplacian_dense: np.ndarray
    if hasattr(laplacian_matrix, "toarray"):
        laplacian_dense = np.asarray(laplacian_matrix.toarray(), dtype=float)
    elif hasattr(laplacian_matrix, "astype"):
        laplacian_dense = np.asarray(laplacian_matrix.astype(float, copy=False))
    else:  # pragma: no cover - already dense
        laplacian_dense = np.asarray(laplacian_matrix, dtype=float)

    return np.linalg.eigvalsh(laplacian_dense)


def _first_positive_eigenvalue(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """Return the smallest positive eigenvalue (Fiedler value)."""

    for value in eigenvalues:
        if value > 1e-9:
            return float(value)
    return 0.0


def _structural_potential(nodes: int, edges: int) -> float:
    """Normalize edge density to emulate Φ_s telemetry."""

    if nodes < 2:
        return 0.0
    full = nodes * (nodes - 1) / 2
    return min(1.0, max(0.0, (edges / full)))


def _coherence_length(laplacian_gap: float) -> float:
    """Approximate ξ_C via the inverse Laplacian gap (diffusion time scale)."""

    if laplacian_gap <= 0:
        return float("inf")
    return 1.0 / laplacian_gap


def _coherence_score(
    phi_s: float,
    phase_gradient: float,
    phase_curvature: float,
    coherence_length: float,
) -> float:
    """Compress tetrad proxies into a [0, 1] coherence score."""

    phi_penalty = max(0.0, phi_s - 0.771) / 0.771
    gradient_penalty = max(0.0, phase_gradient - 0.183736807) / 0.183736807  # γ/π canonical
    curvature_penalty = max(0.0, phase_curvature - 2.8274) / 2.8274

    if coherence_length == float("inf"):
        length_penalty = 1.0
    else:
        length_penalty = max(0.0, (coherence_length - 1.0)) / max(coherence_length, 1.0)

    penalty = phi_penalty + gradient_penalty + curvature_penalty + length_penalty
    return float(math.exp(-penalty))


def _candidate_factors(
    n: int,
    hints: Sequence[int],
    gap: float,
    modulus: int,
    arithmetic: ArithmeticTelemetry,
) -> List[int]:
    """Derive candidate factor candidates.

    Two modes:
    - Default (arithmetic-assisted): original heuristic + gcd refinement.
    - Pure TNFR mode (env TNFR_PURE_MODE=1): emit nodal-derived seeds only,
      avoiding gcd / trial division. Optional light divisibility check can be
      enabled via TNFR_PURE_MODE_VERIFY_DIVISIBILITY=1 (kept separate so that
      pure mode can run without invoking arithmetic factoring helpers).
    """

    def _pure_tnfr_mode() -> bool:
        raw = os.getenv("TNFR_PURE_MODE", "")
        return raw.lower() in {"1", "true", "yes", "on"}

    def _pure_verify_divisibility() -> bool:
        raw = os.getenv("TNFR_PURE_MODE_VERIFY_DIVISIBILITY", "")
        return raw.lower() in {"1", "true", "yes", "on"}

    estimates = _estimate_prime_like_values(gap)
    seeds: Set[int] = set()
    # Nodal / spectral driven seeds
    seeds.update(max(2, int(round(value))) for value in estimates if value and value > 1)
    seeds.update(max(2, int(round(value))) for value in hints if value)
    # Structural modulus vicinity & sqrt heuristic (still spectral/size derived)
    seeds.update({max(2, modulus - 1), modulus, modulus + 1, max(2, math.isqrt(n))})

    # Arithmetic telemetry hints are excluded in pure mode because they encode
    # number-theoretic aggregates (tau, sigma, omega) that do not emerge purely
    # from the nodal evolution; they remain in assisted mode.
    if not _pure_tnfr_mode():
        seeds.update(_arithmetic_hint_values(n, arithmetic))

    def _register_ratio(value: float) -> None:
        if value and value > 1:
            seeds.add(max(2, int(round(value))))

    # Ratios of n to spectral estimates remain allowed (they map spectral gap
    # induced scale to complementary scale). In pure mode these are treated as
    # structural scale hypotheses, not confirmed factors.
    for estimate in estimates:
        if estimate and estimate > 1:
            _register_ratio(n / estimate)

    # Local window broadening (maintains resonance vicinity exploration) – OK in both modes.
    search_window = 24
    for center in list(seeds):
        for delta in range(-search_window, search_window + 1):
            candidate = center + delta
            if candidate > 1:
                seeds.add(candidate)

    if _pure_tnfr_mode():
        # In pure mode we return structural seeds directly. Optional divisibility
        # check can be enabled separately so users can still confirm correctness
        # without invoking gcd-based refinement.
        if _pure_verify_divisibility():
            return sorted({s for s in seeds if 1 < s < n and n % s == 0})
        return sorted(seeds)

    # Assisted mode (original logic) – perform gcd refinement.
    candidates: Set[int] = set()
    for seed in seeds:
        g = math.gcd(n, seed)
        if 1 < g < n:
            candidates.add(g)

        partner = int(round(n / seed)) if seed else 0
        if partner > 1:
            g_partner = math.gcd(n, partner)
            if 1 < g_partner < n:
                candidates.add(g_partner)

    return sorted(candidates)


def _fallback_factor_candidates(n: int) -> List[int]:
    """Use deterministic trial division when spectral heuristics fail."""

    if n <= 3:
        return []

    limit = _fallback_trial_cap(n)
    if limit < 2:
        return []

    factors: Set[int] = set()

    def _register(divisor: int) -> None:
        if 1 < divisor < n:
            factors.add(divisor)
            partner = n // divisor
            if 1 < partner < n:
                factors.add(partner)

    if n % 2 == 0:
        _register(2)

    divisor = 3
    while divisor <= limit:
        if n % divisor == 0:
            _register(divisor)
        divisor += 2

    return sorted(factors)


def _fallback_trial_cap(n: int) -> int:
    sqrt_n = int(math.isqrt(n))
    raw = os.getenv(_FALLBACK_MAX_DIVISOR_ENV)
    if not raw:
        return sqrt_n
    try:
        value = int(raw)
    except ValueError:
        return sqrt_n
    if value <= 0:
        return sqrt_n
    return min(sqrt_n, value)


def _arithmetic_hint_values(n: int, telemetry: ArithmeticTelemetry) -> Set[int]:
    hints: Set[int] = set()
    terms = telemetry.terms
    hints.add(max(2, terms.tau))
    hints.add(max(2, terms.sigma))
    hints.add(max(2, terms.omega))

    if terms.sigma > terms.tau:
        hints.add(max(2, terms.sigma // terms.tau))

    if telemetry.delta_nfr > 0:
        approx = int(round(n / telemetry.delta_nfr))
        if approx > 1:
            hints.add(approx)

    coherence_component = telemetry.components.get("coherence", 0.0)
    if coherence_component:
        hints.add(max(2, int(abs(coherence_component))))

    return {value for value in hints if value > 1}


def _relative_partition_path(
    *,
    filename: str,
    partition_directory: Path,
    relative_directory: Path | None,
) -> str:
    candidate = partition_directory / filename
    if relative_directory is not None:
        return (relative_directory / filename).as_posix()
    return candidate.as_posix()


def _summarize_entry_field(
    entries: Sequence[Mapping[str, Any]],
    field: str,
) -> Dict[str, float] | None:
    values: List[float] = []
    for entry in entries:
        value = entry.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    total = float(sum(values))
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "avg": total / len(values),
        "total": total,
    }


def _emit_certificate(
    *,
    result: SpectralAnalysisResult,
    arithmetic: ArithmeticTelemetry,
    certificate_dir: Path | None,
    sequence: OperatorSequenceSelection,
    partitioning: PartitionedPaleyGraph | None,
) -> Tuple[Path, Path | None, PartitionManifestArtifacts | None]:
    certificate = _build_operator_certificate(
        result,
        arithmetic,
        sequence,
        partitioning=partitioning,
        strategy_plan=result.operator_strategy_plan,
        nodal_decoding=result.nodal_decoding,
        tnfr_verification=result.tnfr_verification,
    )
    validation_result = _validate_operator_sequence(certificate.operators)
    if not validation_result.passed:
        raise ValueError("Operator sequence failed U1-U6 validation")

    validation_block = _serialize_validation_result(validation_result)
    certificate.validation = validation_block
    certificate.invariant_report = _build_invariant_report(
        result=result,
        validation_block=validation_block,
        nodal_decoding=result.nodal_decoding,
        tnfr_verification=result.tnfr_verification,
    )

    directory = _resolve_certificate_dir(certificate_dir)
    base_name = f"certificate_{result.n}_{int(certificate.timestamp)}"
    partition_files: List[str] = []
    partition_dir: Path | None = None
    relative_partition_dir: Path | None = None
    manifest_artifacts: PartitionManifestArtifacts | None = None
    if partitioning is not None:
        partition_dir, relative_partition_dir = _resolve_partition_directory(base_name)
        partition_files, manifest_entries = _emit_partition_certificates(
            result=result,
            partitioning=partitioning,
            partition_directory=partition_dir,
            relative_directory=relative_partition_dir,
        )
        manifest_artifacts = _write_partition_manifest(
            result=result,
            partition_directory=partition_dir,
            relative_directory=relative_partition_dir,
            partition_files=partition_files,
            manifest_entries=manifest_entries,
        )
    if partition_files:
        threshold = _partition_filelist_threshold()
        if len(partition_files) > threshold:
            certificate.partition_files = partition_files[:threshold]
        else:
            certificate.partition_files = partition_files
        if relative_partition_dir is not None:
            certificate.partition_directory = relative_partition_dir.as_posix()
        elif partition_dir is not None:
            certificate.partition_directory = partition_dir.as_posix()
        if partition_dir is not None:
            certificate.partition_directory_absolute = partition_dir.as_posix()
    if manifest_artifacts:
        if manifest_artifacts.manifest_relative:
            certificate.partition_manifest = manifest_artifacts.manifest_relative
        if manifest_artifacts.summary_relative:
            certificate.partition_manifest_index = manifest_artifacts.summary_relative
        if manifest_artifacts.archive_relative:
            certificate.partition_file_archive = manifest_artifacts.archive_relative

    self_opt_summary = run_partition_self_optimization(
        manifest_path=(manifest_artifacts.manifest_absolute if manifest_artifacts else None),
        manifest_summary_path=(
            manifest_artifacts.summary_absolute if manifest_artifacts else None
        ),
        base_name=base_name,
        operation_type="paley_partition",
    )
    if self_opt_summary:
        serialized_summary = _json_safe(self_opt_summary)
        certificate.self_optimization_summary = serialized_summary
        result.self_optimization_summary = serialized_summary
        existing_plan = (
            result.operator_strategy_plan if isinstance(result.operator_strategy_plan, dict) else None
        )
        merged_plan = attach_self_opt_sequences(existing_plan, self_opt_summary)
        if merged_plan is not None:
            json_safe_plan = _json_safe(merged_plan)
            result.operator_strategy_plan = json_safe_plan
            certificate.strategy_plan_snapshot = json_safe_plan

    filename = f"{base_name}.json"
    path = directory / filename
    path.write_text(json.dumps(asdict(certificate), indent=2))
    return path, partition_dir, manifest_artifacts


def _emit_partition_certificates(
    *,
    result: SpectralAnalysisResult,
    partitioning: PartitionedPaleyGraph,
    partition_directory: Path,
    relative_directory: Path | None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    files: List[str] = []
    entries: List[Dict[str, Any]] = []
    aggregation = result.partition_aggregation or {}
    partition_candidates = aggregation.get("partition_candidates", {})
    parent_summary = result.partition_summary or {}

    partition_directory.mkdir(parents=True, exist_ok=True)

    for partition in partitioning.iter_partitions():
        partition_id = partition.partition_id
        candidate_list = partition_candidates.get(partition_id)
        if candidate_list is None:
            candidate_list = list(partition.candidate_factors)

        partition_payload = partition.to_mapping()
        payload = _json_safe(
            {
                "n": result.n,
                "modulus": result.modulus,
                "partition_id": partition_id,
                "partition": partition_payload,
                "candidate_factors": candidate_list,
                "parent_partition_summary": parent_summary,
                "notes": "partition-certificate",
            }
        )

        filename = f"{partition_directory.name}_{partition_id}.json"
        partition_path = partition_directory / filename
        partition_path.write_text(json.dumps(payload, indent=2))
        if relative_directory is not None:
            relative_path = (relative_directory / filename).as_posix()
        else:
            relative_path = partition_path.as_posix()
        files.append(relative_path)
        entries.append(
            {
                "partition_id": partition_id,
                "relative_path": relative_path,
                "candidate_count": len(candidate_list),
                "node_count": len(partition.node_indices),
                "boundary_count": len(partition.boundary_nodes),
                "telemetry": partition_payload.get("telemetry", {}),
            }
        )

    return files, entries


def _write_partition_manifest(
    *,
    result: SpectralAnalysisResult,
    partition_directory: Path,
    relative_directory: Path | None,
    partition_files: Sequence[str],
    manifest_entries: Sequence[Mapping[str, Any]],
) -> PartitionManifestArtifacts | None:
    if not partition_files and not manifest_entries:
        return None

    timestamp = time.time()
    threshold = _partition_filelist_threshold()
    inline_filelist = len(partition_files) <= threshold
    archive_path: Path | None = None
    archive_relative: str | None = None

    if not inline_filelist:
        archive_path = partition_directory / _PARTITION_FILELIST_ARCHIVE
        with gzip.open(archive_path, "wt", encoding="utf-8") as archive_stream:
            for relative_path in partition_files:
                archive_stream.write(relative_path)
                archive_stream.write("\n")
        archive_relative = _relative_partition_path(
            filename=_PARTITION_FILELIST_ARCHIVE,
            partition_directory=partition_directory,
            relative_directory=relative_directory,
        )
        partition_files_payload: Sequence[str] = []
    else:
        partition_files_payload = list(partition_files)

    manifest_path = partition_directory / _MANIFEST_FILENAME
    manifest_relative = _relative_partition_path(
        filename=_MANIFEST_FILENAME,
        partition_directory=partition_directory,
        relative_directory=relative_directory,
    )
    manifest_payload = _json_safe(
        {
            "n": result.n,
            "modulus": result.modulus,
            "timestamp": timestamp,
            "partition_directory": (
                relative_directory.as_posix() if relative_directory else str(partition_directory)
            ),
            "partition_files": list(partition_files_payload),
            "entries": list(manifest_entries),
            "summary": result.partition_summary or {},
            "aggregation": result.partition_aggregation or {},
            "partition_file_archive": archive_relative,
            "partition_file_threshold": threshold,
            "partition_files_inlined": inline_filelist,
        }
    )
    manifest_path.write_text(json.dumps(manifest_payload, indent=2))

    summary_path = partition_directory / _MANIFEST_SUMMARY_FILENAME
    summary_relative = _relative_partition_path(
        filename=_MANIFEST_SUMMARY_FILENAME,
        partition_directory=partition_directory,
        relative_directory=relative_directory,
    )
    candidate_stats = _summarize_entry_field(manifest_entries, "candidate_count")
    node_stats = _summarize_entry_field(manifest_entries, "node_count")
    boundary_stats = _summarize_entry_field(manifest_entries, "boundary_count")
    telemetry_keys = sorted(
        {
            key
            for entry in manifest_entries
            for key in entry.get("telemetry", {}).keys()
            if isinstance(key, str)
        }
    )
    summary_payload = _json_safe(
        {
            "n": result.n,
            "modulus": result.modulus,
            "timestamp": timestamp,
            "partition_count": len(manifest_entries),
            "candidate_stats": candidate_stats,
            "node_stats": node_stats,
            "boundary_stats": boundary_stats,
            "telemetry_keys": telemetry_keys,
            "file_index": {
                "inline": inline_filelist,
                "threshold": threshold,
                "archive": archive_relative,
                "manifest": manifest_relative,
            },
        }
    )
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return PartitionManifestArtifacts(
        manifest_relative=manifest_relative,
        manifest_absolute=manifest_path,
        summary_relative=summary_relative,
        summary_absolute=summary_path,
        archive_relative=archive_relative,
        archive_absolute=archive_path,
    )


def _build_operator_certificate(
    result: SpectralAnalysisResult,
    arithmetic: ArithmeticTelemetry,
    sequence: OperatorSequenceSelection,
    *,
    partitioning: PartitionedPaleyGraph | None,
    strategy_plan: Dict[str, Any] | None,
    nodal_decoding: Dict[str, Any] | None,
    tnfr_verification: Dict[str, Any] | None,
) -> OperatorCertificate:
    canonical_ops = list(sequence.canonical_sequence)
    operators = list(sequence.glyph_sequence)
    telemetry = {
        "phi_s": result.phi_s,
        "phase_gradient": result.phase_gradient,
        "phase_curvature": result.phase_curvature,
        "coherence_length": result.coherence_length,
        "coherence_score": result.coherence_score,
        "delta_nfr": result.arithmetic_delta_nfr,
        "local_coherence": result.arithmetic_local_coherence,
    }
    telemetry.update({f"arith_{k}": v for k, v in arithmetic.components.items()})

    partition_block: Dict[str, Any] | None = None
    if result.partition_summary or result.partition_aggregation:
        partition_block = _json_safe(
            {
                "summary": result.partition_summary or {},
                "aggregation": result.partition_aggregation or {},
            }
        )

    partition_states = _collect_partition_states(partitioning)
    strategy_snapshot = _json_safe(strategy_plan) if strategy_plan else None
    nodal_snapshot = _json_safe(nodal_decoding) if nodal_decoding else None
    tnfr_snapshot = _json_safe(tnfr_verification) if tnfr_verification else None

    # Generate certificate-specific hash chain and replay metadata
    certificate_hash_chain = _generate_partition_hash_chain(partitioning)
    certificate_replay_metadata = _generate_replay_metadata(result, partitioning=partitioning)
    if certificate_replay_metadata:
        certificate_replay_metadata["certificate_seeds"] = _capture_deterministic_seeds()
    
    return OperatorCertificate(
        n=result.n,
        candidate_factor=result.candidate_factors[0] if result.candidate_factors else None,
        operators=operators,
        canonical_operators=canonical_ops,
        telemetry=telemetry,
        timestamp=time.time(),
        notes="Operator certificate derived from optimizer-guided Paley sequence",
        optimizer=_json_safe(sequence.optimizer_metadata),
        partitions=partition_block,
        partition_states=_json_safe(partition_states) if partition_states else None,
        strategy_plan_snapshot=strategy_snapshot,
        nodal_decoding_snapshot=nodal_snapshot,
        tnfr_verification_snapshot=tnfr_snapshot,
        tnfr_factor_signature=_json_safe(result.tnfr_factor_signature)
        if result.tnfr_factor_signature
        else None,
        tnfr_composite_signature=_json_safe(result.tnfr_composite_signature)
        if result.tnfr_composite_signature
        else None,
        certificate_hash_chain=_json_safe(certificate_hash_chain) if certificate_hash_chain else None,
        replay_metadata=_json_safe(certificate_replay_metadata) if certificate_replay_metadata else None,
    )


def _collect_partition_states(partitioning: PartitionedPaleyGraph | None) -> Dict[str, Any] | None:
    if partitioning is None:
        return None

    block: Dict[str, Any] = {}
    for partition in partitioning.iter_partitions():
        telemetry = partition.telemetry
        before = {
            "phi_s": telemetry.phi_s if telemetry else None,
            "phase_gradient": telemetry.phase_gradient if telemetry else None,
            "phase_curvature": telemetry.phase_curvature if telemetry else None,
            "coherence_length": telemetry.coherence_length if telemetry else None,
            "notes": telemetry.notes if telemetry else "",
        }
        nodal_state = partition.metadata.get("nodal_state", {}) if partition.metadata else {}
        block[partition.partition_id] = {
            "before": before,
            "after": nodal_state,
            "node_count": len(partition.node_indices),
            "boundary_count": len(partition.boundary_nodes),
            "candidate_factors": list(partition.candidate_factors),
        }
    return block or None


def _verify_factors_tnfr(
    *,
    n: int,
    partitioning: PartitionedPaleyGraph,
    nodal_decoding: Mapping[str, Any],
    parent_phi_s: float,
    parent_phase_gradient: float,
    parent_phase_curvature: float,
) -> Tuple[List[int], Dict[str, Any] | None]:
    partitions = list(partitioning.iter_partitions())
    if not partitions:
        return [], None

    entries = nodal_decoding.get("partitions") if isinstance(nodal_decoding, Mapping) else None
    if not isinstance(entries, Sequence):
        return [], None

    partition_map = {partition.partition_id: partition for partition in partitions}
    criteria = dict(_TNFR_VERIFICATION_CRITERIA)
    per_factor: Dict[int, Dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        factor = entry.get("inferred_factor")
        if not isinstance(factor, int) or factor <= 1:
            continue
        partition_id = str(entry.get("partition_id", ""))
        partition = partition_map.get(partition_id)
        node_count = int(entry.get("node_count") or (len(partition.node_indices) if partition else 0))
        dnfr_before = float(entry.get("dnfr_before") or 0.0)
        dnfr_after = float(entry.get("dnfr_after") or 0.0)
        dnfr_gain = 0.0
        if dnfr_before > 0.0:
            dnfr_gain = max(0.0, (dnfr_before - dnfr_after) / max(abs(dnfr_before), 1e-9))
        coherence_ratio = float(entry.get("coherence_ratio") or 0.0)
        local_phi = float(entry.get("local_phi_s") or 0.0)
        phi_baseline = parent_phi_s
        telemetry = partition.telemetry if partition is not None else None
        if telemetry is not None and telemetry.phi_s:
            phi_baseline = telemetry.phi_s
        if local_phi == 0.0:
            local_phi = phi_baseline
        phi_delta_parent = abs(local_phi - parent_phi_s)
        gradient_delta = _relative_delta(
            telemetry.phase_gradient if telemetry else parent_phase_gradient,
            parent_phase_gradient,
        )
        curvature_delta = _relative_delta(
            telemetry.phase_curvature if telemetry else parent_phase_curvature,
            parent_phase_curvature,
        )
        periodicity = entry.get("periodicity") if isinstance(entry.get("periodicity"), Mapping) else None
        periodic_flag = bool(periodicity and periodicity.get("period") == factor)
        periodic_confidence = float(periodicity.get("confidence")) if periodicity else 0.0
        criteria_gradient_max = criteria.get("gradient_delta_max", 1.0)
        criteria_curvature_max = criteria.get("curvature_delta_max", 1.0)
        criteria_confidence_min = criteria.get("periodicity_confidence_min", 0.0)
        gradient_ok = gradient_delta <= criteria_gradient_max
        curvature_ok = curvature_delta <= criteria_curvature_max
        periodic_confidence_ok = periodic_confidence >= criteria_confidence_min
        flags = {
            "stabilized": bool(entry.get("stabilized")),
            "coherence": bool(entry.get("coherence_matched")),
            "dnfr_gain": dnfr_gain >= criteria["dnfr_gain_min"],
            "phi": phi_delta_parent <= criteria["phi_delta_max"],
            "periodic": periodic_flag,
            "gradient": gradient_ok,
            "curvature": curvature_ok,
            "periodic_confidence": periodic_confidence_ok,
        }
        endorsement = sum(1 for value in flags.values() if value) >= criteria["min_partition_flags"]
        factor_block = per_factor.setdefault(
            factor,
            {
                "partitions": [],
                "partition_total": 0,
                "endorsements": 0,
                "nodes_accumulated": 0,
                "dnfr_gain_accumulated": 0.0,
                "phi_delta_samples": [],
                "coherence_samples": [],
                "gradient_delta_samples": [],
                "curvature_delta_samples": [],
                "periodicity_confidence_samples": [],
                "stabilized_count": 0,
            },
        )
        factor_block["partition_total"] += 1
        if flags["stabilized"]:
            factor_block["stabilized_count"] += 1
        factor_block["phi_delta_samples"].append(phi_delta_parent)
        factor_block["coherence_samples"].append(coherence_ratio)
        factor_block["gradient_delta_samples"].append(gradient_delta)
        factor_block["curvature_delta_samples"].append(curvature_delta)
        factor_block["periodicity_confidence_samples"].append(periodic_confidence)
        partition_record = {
            "partition_id": partition_id,
            "node_count": node_count,
            "dnfr_before": dnfr_before,
            "dnfr_after": dnfr_after,
            "dnfr_gain": dnfr_gain,
            "coherence_ratio": coherence_ratio,
            "phi_delta_parent": phi_delta_parent,
            "gradient_delta": gradient_delta,
            "curvature_delta": curvature_delta,
            "periodic_confidence": periodic_confidence,
            "flags": flags,
            "endorsement": endorsement,
        }
        factor_block["partitions"].append(partition_record)
        if endorsement:
            factor_block["endorsements"] += 1
            factor_block["nodes_accumulated"] += max(0, node_count)
            factor_block["dnfr_gain_accumulated"] += dnfr_gain

    if not per_factor:
        return [], None

    certified: List[int] = []
    per_factor_payload: Dict[str, Any] = {}
    modulus = max(1, partitioning.modulus)

    for factor, block in per_factor.items():
        total = block["partition_total"] or 1
        required = max(
            criteria["min_endorsements"],
            math.ceil(total * criteria["required_partition_ratio"]),
        )
        endorsements = block["endorsements"]
        average_gain = (
            block["dnfr_gain_accumulated"] / max(1, endorsements)
            if endorsements
            else 0.0
        )
        coverage_fraction = block["nodes_accumulated"] / modulus if modulus else 0.0
        phi_samples = block["phi_delta_samples"]
        coherence_samples = block["coherence_samples"]
        gradient_samples = block["gradient_delta_samples"]
        curvature_samples = block["curvature_delta_samples"]
        periodicity_confidence_samples = block["periodicity_confidence_samples"]
        block_summary = {
            "factor": factor,
            "endorsements": endorsements,
            "partition_total": total,
            "endorsement_ratio": endorsements / total if total else 0.0,
            "required_endorsements": required,
            "average_dnfr_gain": average_gain,
            "coverage_fraction": coverage_fraction,
            "phi_delta_max": max(phi_samples) if phi_samples else 0.0,
            "phi_delta_avg": (sum(phi_samples) / len(phi_samples)) if phi_samples else 0.0,
            "coherence_span": [
                min(coherence_samples) if coherence_samples else 0.0,
                max(coherence_samples) if coherence_samples else 0.0,
            ],
            "stabilized_fraction": block["stabilized_count"] / total if total else 0.0,
            "gradient_delta_avg": (sum(gradient_samples) / len(gradient_samples)) if gradient_samples else 0.0,
            "curvature_delta_avg": (sum(curvature_samples) / len(curvature_samples)) if curvature_samples else 0.0,
            "periodicity_confidence_avg": (sum(periodicity_confidence_samples) / len(periodicity_confidence_samples)) if periodicity_confidence_samples else 0.0,
            "partitions": block["partitions"],
            "support_divisible": (n % factor == 0),
            "certified": False,
            "notes": "tnfr-verification",
        }
        stabilized_fraction = block_summary["stabilized_fraction"]
        periodicity_confidence_avg = block_summary["periodicity_confidence_avg"]
        pass_all = (
            endorsements >= required
            and average_gain >= criteria["dnfr_gain_min"]
            and stabilized_fraction >= criteria.get("min_stabilized_fraction", 0.0)
            and coverage_fraction >= criteria.get("min_coverage_fraction", 0.0)
            and periodicity_confidence_avg >= criteria.get("periodicity_confidence_min", 0.0)
        )
        if pass_all:
            block_summary["certified"] = True
            certified.append(factor)
        per_factor_payload[str(factor)] = _json_safe(block_summary)

    if not per_factor_payload:
        return sorted(set(certified)), None

    report = {
        "timestamp": time.time(),
        "criteria": criteria,
        "certified": sorted(set(certified)),
        "per_factor": per_factor_payload,
        "partition_sample": len(entries),
        "notes": "tnfr-deterministic-verification",
    }
    return sorted(set(certified)), report


def _build_factor_signature(
    n: int,
    verification: Mapping[str, Any] | None,
    partition_hash_chain: Dict[str, Any] | None = None,
    replay_metadata: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
    if not verification:
        return None

    certified = verification.get("certified")
    if not certified:
        return None

    try:
        factors = sorted({int(value) for value in certified if int(value) > 1})
    except (TypeError, ValueError):
        return None
    if not factors:
        return None

    # Enhanced payload with reproducibility metadata
    payload = {
        "n": n,
        "certified": factors,
        "criteria": verification.get("criteria"),
        "per_factor": verification.get("per_factor"),
        "timestamp": verification.get("timestamp"),
        "certificate_version": _CERTIFICATE_VERSION,
    }
    
    # Add partition hash chain for reproducibility
    if partition_hash_chain:
        payload["partition_hash_chain"] = partition_hash_chain
    
    # Add replay metadata for deterministic reproduction
    if replay_metadata:
        payload["replay_metadata"] = replay_metadata
    
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    
    return {
        "algorithm": "sha256",
        "hash": digest,
        "issued_at": verification.get("timestamp"),
        "certified": factors,
        "certificate_version": _CERTIFICATE_VERSION,
        "partition_hash_chain": partition_hash_chain,
        "replay_metadata": replay_metadata,
        "reproducibility": {
            "hash_chain_present": partition_hash_chain is not None,
            "replay_metadata_present": replay_metadata is not None,
            "deterministic_seeds_captured": replay_metadata and "numpy_random_state" in replay_metadata
        }
    }


def _build_composite_signature(
    n: int,
    verification: Mapping[str, Any] | None,
    partition_hash_chain: Dict[str, Any] | None = None,
    replay_metadata: Dict[str, Any] | None = None
) -> Dict[str, Any] | None:
    """Derive composite factorization signature including powers and multi-factor sets.

    Assisted mode (pure_mode=False) confirms products divide n; pure mode records
    structural composites without divisibility assertion, tagging them as
    'structural_hypothesis'. Exponents estimated by repeated division in assisted
    mode or endorsement multiplicity in pure mode.
    
    Enhanced with partition hash chains and replay metadata for reproducibility.
    """
    if not verification:
        return None
    certified = verification.get("certified")
    if not certified:
        return None
    try:
        factors = sorted({int(v) for v in certified if int(v) > 1})
    except Exception:
        return None
    if not factors:
        return None
    pure_mode = os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"}
    per_factor = verification.get("per_factor") if isinstance(verification.get("per_factor"), Mapping) else {}

    # Determine exponents.
    exponents: Dict[int, int] = {}
    remaining = n
    for f in factors:
        if pure_mode:
            # Estimate exponent structurally via stabilized_fraction scaling.
            block = per_factor.get(str(f), {}) if isinstance(per_factor, Mapping) else {}
            stabilized_fraction = float(block.get("stabilized_fraction") or 0.0)
            # Heuristic: exponent proportional to stabilized_fraction * coverage.
            coverage_fraction = float(block.get("coverage_fraction") or 0.0)
            est = stabilized_fraction * 1.0 + coverage_fraction * 0.5
            exponent = max(1, int(round(est)))
        else:
            exponent = 0
            while remaining % f == 0:
                remaining //= f
                exponent += 1
            if exponent == 0:
                exponent = 1  # fallback minimal exponent
        exponents[f] = exponent

    assisted_complete = (not pure_mode) and (remaining == 1)
    structural_complete = pure_mode and all(e >= 1 for e in exponents.values())

    # Composite sets (pairwise and full product) in assisted mode only when divisibility confirmed.
    composites: List[List[int]] = []
    if len(factors) >= 2:
        # Pairwise products
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                pair = [factors[i], factors[j]]
                product = factors[i] * factors[j]
                if pure_mode or n % product == 0:
                    composites.append(pair)
        # Full set
        product_all = 1
        for f in factors:
            product_all *= f
        if pure_mode or n % product_all == 0:
            composites.append(factors)

    payload = {
        "n": n,
        "factors": factors,
        "exponents": exponents,
        "composites": composites,
        "complete": assisted_complete or structural_complete,
        "mode": "pure" if pure_mode else "assisted",
        "certificate_version": _CERTIFICATE_VERSION,
    }
    
    # Add reproducibility metadata
    if partition_hash_chain:
        payload["partition_hash_chain"] = partition_hash_chain
    if replay_metadata:
        payload["replay_metadata"] = replay_metadata
    
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    payload["algorithm"] = "sha256"
    payload["hash"] = digest
    payload["reproducibility"] = {
        "hash_chain_present": partition_hash_chain is not None,
        "replay_metadata_present": replay_metadata is not None,
        "deterministic_seeds_captured": replay_metadata and "numpy_random_state" in replay_metadata
    }
    return payload


def _build_invariant_report(
    *,
    result: SpectralAnalysisResult,
    validation_block: Mapping[str, Any],
    nodal_decoding: Dict[str, Any] | None,
    tnfr_verification: Dict[str, Any] | None,
) -> Dict[str, Any]:
    grammar_status = "verified" if validation_block.get("passed") else "unverified"
    canonical_tokens = validation_block.get("canonical_tokens", [])
    partition_count = (result.partition_summary or {}).get("partition_count")
    dynamic_converged = 0
    if nodal_decoding:
        dynamic_converged = int(nodal_decoding.get("converged_partitions") or 0)
    tnfr_certified = []
    tnfr_status = "observed"
    tnfr_notes = "no tnfr verification"
    if tnfr_verification:
        tnfr_certified = list(tnfr_verification.get("certified") or [])
        if tnfr_certified:
            tnfr_status = "verified"
            tnfr_notes = f"tnfr-certified factors={tnfr_certified}"
        else:
            tnfr_status = "tracked"
            tnfr_notes = "tnfr verifier executed"

    grammar_rules = {
        "U1": {
            "status": grammar_status,
            "evidence": f"{len(canonical_tokens)} canonical tokens validated",
        },
        "U2": {
            "status": tnfr_status,
            "evidence": (
                f"ΔNFR={result.arithmetic_delta_nfr:.3e}, coherence_score={result.coherence_score:.3f}; "
                + tnfr_notes
            ),
        },
        "U3": {
            "status": "tracked" if result.operator_strategy_plan else "unavailable",
            "evidence": "Operator strategy plan recorded for AL/IL/RA/SHA",
        },
        "U4": {
            "status": "tracked" if dynamic_converged else "observed",
            "evidence": f"{dynamic_converged} partitions converged under nodal decoder",
        },
        "U5": {
            "status": "tracked" if partition_count else "observed",
            "evidence": f"partition_count={partition_count}",
        },
        "U6": {
            "status": "tracked",
            "evidence": f"Φ_s={result.phi_s:.3f}",
        },
    }

    canonical_invariants = [
        {
            "id": 1,
            "name": "Nodal Equation Integrity",
            "status": "tracked",
            "evidence": {
                "delta_nfr": result.arithmetic_delta_nfr,
                "tnfr_certified_factors": tnfr_certified,
            },
        },
        {
            "id": 2,
            "name": "Phase-Coherent Coupling",
            "status": "tracked" if result.operator_strategy_plan else "observed",
            "evidence": {"plan_available": bool(result.operator_strategy_plan)},
        },
        {
            "id": 3,
            "name": "Multi-Scale Fractality",
            "status": "tracked" if partition_count else "observed",
            "evidence": {"partition_count": partition_count},
        },
        {
            "id": 4,
            "name": "Grammar Compliance",
            "status": grammar_status,
            "evidence": {"message": validation_block.get("message")},
        },
        {
            "id": 5,
            "name": "Structural Metrology",
            "status": "tracked",
            "evidence": {
                "phi_s": result.phi_s,
                "phase_gradient": result.phase_gradient,
                "phase_curvature": result.phase_curvature,
                "coherence_length": result.coherence_length,
            },
        },
        {
            "id": 6,
            "name": "Reproducible Dynamics",
            "status": "tracked",
            "evidence": {
                "partition_summary": bool(result.partition_summary),
                "fft_backend": result.fft_backend,
                "tnfr_verification": bool(tnfr_certified),
            },
        },
    ]

    return {
        "grammar_rules": grammar_rules,
        "canonical_invariants": canonical_invariants,
    }


def _resolve_certificate_dir(certificate_dir: Path | None) -> Path:
    directory = certificate_dir or _default_certificate_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_operator_sequence(sequence: Sequence[str]) -> SequenceValidationResult:
    if not sequence:
        raise ValueError("Operator sequence cannot be empty")
    canonical_tokens = _canonicalize_operator_tokens(sequence)
    outcome = validate_sequence(canonical_tokens)
    return cast(SequenceValidationResult, outcome)


def _serialize_validation_result(result: SequenceValidationResult) -> Dict[str, Any]:
    summary = _json_safe_mapping(result.summary)
    metadata = _json_safe_mapping(getattr(result, "metadata", {}) or {})
    artifacts = _json_safe_mapping(result.artifacts or {})
    return {
        "passed": result.passed,
        "message": result.message,
        "summary": summary,
        "metadata": metadata,
        "tokens": [_json_safe(token) for token in result.tokens],
        "canonical_tokens": [_json_safe(token) for token in result.canonical_tokens],
        "artifacts": artifacts,
    }


_GLYPH_TO_CANONICAL = {
    "AL": "emission",
    "EN": "reception",
    "IL": "coherence",
    "OZ": "dissonance",
    "UM": "coupling",
    "RA": "resonance",
    "SHA": "silence",
    "VAL": "expansion",
    "NUL": "contraction",
    "THOL": "self_organization",
    "ZHIR": "mutation",
    "NAV": "transition",
    "REMESH": "recursivity",
}

_CANONICAL_TO_GLYPH = {name.lower(): glyph for glyph, name in _GLYPH_TO_CANONICAL.items()}
_DEFAULT_CANONICAL_SEQUENCE = ["emission", "coupling", "resonance", "coherence", "silence"]

_RECOMMENDATION_SEQUENCE_MAP = {
    "use_spectral_methods": ["emission", "coupling", "resonance", "coherence", "silence"],
    "use_hierarchical_methods": ["recursivity", "coupling", "self_organization", "coherence", "silence"],
    "use_chiral_optimization": ["emission", "dissonance", "self_organization", "coherence", "silence"],
    "use_phase_transition_handling": ["transition", "mutation", "coherence", "silence"],
    "enhance_coherence_coupling": ["emission", "coupling", "coherence", "resonance", "silence"],
    "topological_defect_correction": ["emission", "coupling", "self_organization", "coherence", "silence"],
    "high_energy_stabilization": ["emission", "coherence", "silence"],
    "low_variance_epi_optimization": ["emission", "coherence", "resonance", "silence"],
    "uniform_vf_optimization": ["emission", "resonance", "coherence", "silence"],
    "high_dnfr_stabilization": ["emission", "dissonance", "coherence", "silence"],
    "large_graph_optimization": ["recursivity", "coupling", "resonance", "coherence", "silence"],
    "dense_graph_optimization": ["emission", "resonance", "self_organization", "coherence", "silence"],
    "sparse_graph_optimization": ["emission", "coupling", "coherence", "silence"],
}

_RECOMMENDATION_PREFIX_MAP = {
    "use_compression_": ["emission", "coherence", "contraction", "coherence", "silence"],
    "use_prediction_": ["emission", "reception", "coupling", "resonance", "silence"],
}

_NODAL_OPERATOR_SEQUENCE = ["UM", "RA", "IL", "THOL"]

_TNFR_VERIFICATION_CRITERIA = {
    "min_partition_flags": 4,  # require more structural conditions per partition
    "dnfr_gain_min": 0.15,     # minimum relative ΔNFR attenuation
    "coherence_min": 0.72,     # coherence similarity lower bound
    "coherence_max": 1.38,     # coherence similarity upper bound
    "phi_delta_max": 0.35,     # structural potential deviation threshold
    "gradient_delta_max": 0.40,  # relative phase gradient deviation limit
    "curvature_delta_max": 0.45,  # relative phase curvature deviation limit
    "periodicity_confidence_min": 0.55,  # minimum structural periodicity confidence
    "required_partition_ratio": 0.5,     # fraction of partition endorsements needed
    "min_endorsements": 1,               # absolute minimum endorsements
    "min_stabilized_fraction": 0.30,     # fraction of partitions with stabilized flag
    "min_coverage_fraction": 0.15,       # coverage across modulus (nodes_accumulated/modulus)
}


def _canonicalize_operator_tokens(sequence: Sequence[str]) -> List[str]:
    canonical: List[str] = []
    for token in sequence:
        if not token:
            continue
        lowered = token.lower()
        if lowered in CANONICAL_OPERATOR_NAMES:
            canonical.append(lowered)
            continue
        mapped = _GLYPH_TO_CANONICAL.get(token.upper())
        if mapped is not None:
            canonical.append(mapped)
            continue
        canonical.append(token)
    return canonical


def _canonical_to_glyph_sequence(sequence: Sequence[str]) -> List[str]:
    glyphs: List[str] = []
    for token in sequence:
        glyphs.append(_CANONICAL_TO_GLYPH.get(token.lower(), token.upper()))
    return glyphs


def _select_strategy_candidate(
    bucket: Mapping[str, Callable[[], Any]],
    ctx: StrategyContext,
) -> Tuple[str, ResourceEstimate] | None:
    best_score: Tuple[int, float, int] | None = None
    best_name: str | None = None
    best_estimate: ResourceEstimate | None = None
    for name, factory in bucket.items():
        strategy = factory()
        if not strategy.supports(ctx):
            continue
        estimate = strategy.resource_estimate(ctx)
        rank = _FAILURE_RISK_ORDER.get(estimate.failure_risk, len(_FAILURE_RISK_ORDER))
        score = (rank, estimate.time_ms, estimate.memory_bytes)
        if best_score is None or score < best_score:
            best_score = score
            best_name = name
            best_estimate = estimate
    if best_name is None or best_estimate is None:
        return None
    return best_name, best_estimate


def _default_sequence_selection(reason: str, error: str | None = None) -> OperatorSequenceSelection:
    canonical = list(_DEFAULT_CANONICAL_SEQUENCE)
    metadata: Dict[str, Any] = {"reason": reason}
    if error:
        metadata["error"] = error
    return OperatorSequenceSelection(
        canonical_sequence=canonical,
        glyph_sequence=_canonical_to_glyph_sequence(canonical),
        optimizer_metadata=metadata,
    )


def _sequence_from_recommendations(strategies: Sequence[str]) -> List[str] | None:
    for strategy in strategies:
        normalized = strategy.lower()
        if normalized in _RECOMMENDATION_SEQUENCE_MAP:
            return list(_RECOMMENDATION_SEQUENCE_MAP[normalized])
        for prefix, sequence in _RECOMMENDATION_PREFIX_MAP.items():
            if normalized.startswith(prefix):
                return list(sequence)
    return None


def _evaluate_partition_nodal_sequence(
    *,
    partition: "PaleyPartition",
    graph: nx.Graph,
    modulus: int,
    n: int,
    parent_gap: float,
    parent_phi_s: float,
    parent_phase_gradient: float,
    parent_phase_curvature: float,
    parent_coherence_length: float,
) -> Dict[str, Any]:
    nodes = list(partition.node_indices)
    node_count = len(nodes)
    subgraph = graph.subgraph(nodes).copy() if node_count else nx.Graph()
    local_gap = 0.0
    local_edges = 0

    if node_count >= 2:
        local_edges = subgraph.number_of_edges()
        local_eigenvalues = _laplacian_eigenvalues(subgraph)
        local_gap = _first_positive_eigenvalue(local_eigenvalues)
    elif node_count == 1:
        local_edges = 0

    local_phi_s = _structural_potential(node_count, local_edges)
    telemetry = partition.telemetry
    partition_phi = telemetry.phi_s if telemetry is not None else local_phi_s
    partition_gradient = telemetry.phase_gradient if telemetry is not None else parent_phase_gradient
    partition_curvature = telemetry.phase_curvature if telemetry is not None else parent_phase_curvature
    local_coherence_length = (
        _coherence_length(local_gap)
        if node_count >= 2 and local_gap > 0
        else (parent_coherence_length if node_count == 0 else float("inf"))
    )

    gradient_delta = _relative_delta(partition_gradient, parent_phase_gradient)
    curvature_delta = _relative_delta(partition_curvature, parent_phase_curvature)
    phi_delta = abs(partition_phi - parent_phi_s)
    dnfr_before = _partition_delta_nfr(
        parent_gap=parent_gap,
        local_gap=local_gap,
        phi_delta=phi_delta,
        gradient_delta=gradient_delta,
        curvature_delta=curvature_delta,
    )
    dnfr_after = _simulate_partition_sequence(dnfr_before, local_phi_s, node_count)
    dnfr_converged = dnfr_after <= max(1e-4, dnfr_before * 0.2)
    coherence_ratio = _coherence_similarity(local_coherence_length, node_count)
    coherence_matched = math.isfinite(coherence_ratio) and 0.75 <= coherence_ratio <= 1.35
    stabilized = dnfr_converged and coherence_matched

    period, periodicity_block = _infer_partition_periodicity(nodes, modulus)
    inferred_factor = None
    pure_mode = os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"}
    confidence = periodicity_block.get("confidence", 0.0) if periodicity_block else 0.0
    if period is not None and 1 < period < n:
        if pure_mode:
            # Pure mode: rely on structural confidence not arithmetic divisibility.
            if confidence >= 0.6:
                inferred_factor = period
        else:
            # Assisted mode: retain divisibility check for endorsement.
            if n % period == 0:
                inferred_factor = period

    outcome: Dict[str, Any] = {
        "partition_id": partition.partition_id,
        "sequence": list(_NODAL_OPERATOR_SEQUENCE),
        "node_count": node_count,
        "dnfr_before": dnfr_before,
        "dnfr_after": dnfr_after,
        "dnfr_converged": dnfr_converged,
        "coherence_ratio": coherence_ratio,
        "coherence_matched": coherence_matched,
        "stabilized": stabilized,
        "local_gap": local_gap,
        "local_phi_s": local_phi_s,
        "local_coherence_length": local_coherence_length,
        "periodicity": periodicity_block,
        "inferred_factor": inferred_factor,
    }

    return outcome


def _relative_delta(value: float, reference: float) -> float:
    if reference == 0:
        return abs(value)
    return abs(value - reference) / max(abs(reference), 1e-9)


def _partition_delta_nfr(
    *,
    parent_gap: float,
    local_gap: float,
    phi_delta: float,
    gradient_delta: float,
    curvature_delta: float,
) -> float:
    if parent_gap <= 0:
        gap_component = abs(local_gap)
    else:
        gap_component = abs(local_gap - parent_gap) / max(parent_gap, 1e-9)
    return gap_component + phi_delta + 0.5 * (gradient_delta + curvature_delta)


def _simulate_partition_sequence(delta_nfr: float, local_phi_s: float, node_count: int) -> float:
    if delta_nfr <= 0:
        return 0.0
    phi_clamped = min(1.0, max(0.0, local_phi_s))
    coupling_gain = 0.3 * phi_clamped
    resonance_gain = 0.2 * phi_clamped
    coherence_gain = 0.25 + 0.15 * phi_clamped
    self_org_gain = 0.1 if node_count > 1 else 0.05
    attenuation = min(0.95, coupling_gain + resonance_gain + coherence_gain + self_org_gain)
    return max(0.0, delta_nfr * (1.0 - attenuation))


def _coherence_similarity(local_length: float, node_count: int) -> float:
    if node_count <= 0:
        return 0.0
    if local_length == float("inf"):
        return float("inf")
    if local_length <= 0:
        return 0.0
    return local_length / float(node_count)


def _infer_partition_periodicity(nodes: Sequence[int], modulus: int) -> Tuple[int | None, Dict[str, Any]]:
    """Infer structural periodicity using nodal offsets and diff statistics.

    Pure TNFR mode avoids arithmetic gcd; period derived from dominant spacing
    frequency and stability (low normalized std). Assisted mode additionally
    refines by computing an iterative gcd as a secondary hint (not required for
    factor inference – only recorded as 'arith_support').
    """
    node_list = list(nodes)
    if len(node_list) < 2 or modulus <= 0:
        return None, {
            "period": None,
            "offsets": node_list[:8],
            "origin": node_list[0] if node_list else None,
            "size_hint": len(node_list),
            "confidence": 0.0,
            "pure_mode": os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"},
        }

    pure_mode = os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"}
    sorted_nodes = sorted(node_list)
    base = sorted_nodes[0]
    offsets = [0]
    for node in sorted_nodes[1:]:
        offsets.append((node - base) % modulus)

    diffs: List[int] = []
    for idx in range(1, len(sorted_nodes)):
        diff = (sorted_nodes[idx] - sorted_nodes[idx - 1]) % modulus
        if diff == 0:
            diff = modulus
        diffs.append(diff)

    if not diffs:
        return None, {
            "period": None,
            "offsets": offsets[:8],
            "origin": base,
            "size_hint": len(sorted_nodes),
            "confidence": 0.0,
            "pure_mode": pure_mode,
        }

    # Dominant spacing (mode) for structural period hypothesis
    freq: Dict[int, int] = {}
    for d in diffs:
        freq[d] = freq.get(d, 0) + 1
    dominant_spacing = max(freq.items(), key=lambda kv: kv[1])[0]
    mode_count = freq[dominant_spacing]
    mean_spacing = sum(diffs) / len(diffs)
    variance = sum((d - mean_spacing) ** 2 for d in diffs) / max(1, len(diffs))
    std_spacing = math.sqrt(variance)
    stability = math.exp(-std_spacing / max(mean_spacing, 1e-9))
    dominance = mode_count / len(diffs)
    base_confidence = stability * dominance

    # Assisted mode: compute arithmetic gcd as supporting evidence only.
    arith_support = None
    if not pure_mode:
        period_gcd: int | None = diffs[0]
        for diff in diffs[1:]:
            period_gcd = math.gcd(period_gcd, diff) if period_gcd else diff
        if period_gcd and period_gcd > 1:
            arith_support = period_gcd
            # Boost confidence slightly if gcd agrees with dominant spacing.
            if arith_support == dominant_spacing:
                base_confidence = min(1.0, base_confidence * 1.15)

    # Structural period candidate: dominant spacing unless trivial.
    period = dominant_spacing if dominant_spacing > 1 else None
    if period and period >= modulus:
        period = None

    payload = {
        "period": period,
        "offsets": offsets[:8],
        "origin": base,
        "size_hint": len(sorted_nodes),
        "mode_spacing": dominant_spacing,
        "mode_count": mode_count,
        "mean_spacing": mean_spacing,
        "std_spacing": std_spacing,
        "stability": stability,
        "dominance": dominance,
        "confidence": round(base_confidence, 6),
        "pure_mode": pure_mode,
    }
    if arith_support is not None:
        payload["arith_support"] = arith_support
        payload["arith_support_match"] = (arith_support == dominant_spacing)
    return period, payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def _json_safe_mapping(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def _estimate_prime_like_values(gap: float) -> List[float]:
    """Invert the Paley Fiedler approximation to guess relevant prime sizes."""

    if gap <= 0:
        return []

    disc = 1 + 8 * gap
    sqrt_disc = math.sqrt(disc)
    x = (1 + sqrt_disc) / 2.0
    p_estimate = x * x

    return [p_estimate, p_estimate - 1, p_estimate + 1]


@dataclass(frozen=True)
class ArithmeticTelemetry:
    """Arithmetic TNFR telemetry derived from canonical formalism."""

    terms: ArithmeticStructuralTerms
    epi: float
    nu_f: float
    delta_nfr: float
    components: Dict[str, float]
    local_coherence: float


_ARITHMETIC_PARAMS = ArithmeticTNFRParameters()


@cache_tnfr_computation(level=CacheLevel.DERIVED_METRICS, dependencies={"n"})  # type: ignore[misc]
def _compute_arithmetic_telemetry(n: int) -> ArithmeticTelemetry:
    """Compute canonical arithmetic TNFR telemetry for ``n`` with caching."""

    if n <= 0:
        raise ValueError("n must be positive for arithmetic telemetry")

    factors = _prime_factorization(n)
    terms = ArithmeticStructuralTerms(
        tau=_tau_from_factors(factors),
        sigma=_sigma_from_factors(factors),
        omega=sum(factors.values()),
    )

    epi = ArithmeticTNFRFormalism.epi_value(n, terms, _ARITHMETIC_PARAMS)
    nu_f = ArithmeticTNFRFormalism.frequency_value(n, terms, _ARITHMETIC_PARAMS)
    delta_nfr = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, _ARITHMETIC_PARAMS)
    components = ArithmeticTNFRFormalism.component_breakdown(n, terms, _ARITHMETIC_PARAMS)
    local_coherence = ArithmeticTNFRFormalism.local_coherence(delta_nfr)

    return ArithmeticTelemetry(
        terms=terms,
        epi=epi,
        nu_f=nu_f,
        delta_nfr=delta_nfr,
        components=components,
        local_coherence=local_coherence,
    )


def _select_fft_backend(max_nodes: int | None) -> FFTBackend:
    preference = os.getenv("TNFR_FFT_BACKEND", "").strip().lower()
    dispatcher = _load_fft_dispatcher()
    threshold = _read_threshold_env("TNFR_FFT_AUTO_THRESHOLD", default=4096)
    auto_large = max_nodes is not None and threshold is not None and max_nodes >= threshold

    if preference == "distributed" or auto_large:
        backend = _instantiate_distributed_backend(dispatcher)
        if backend is not None:
            return backend

    return TNFRAdvancedFFTEngine()


def _instantiate_distributed_backend(dispatcher: Dispatcher | None) -> FFTBackend | None:
    try:
        from tnfr.dynamics.distributed_fft import DistributedFFTEngine
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        return DistributedFFTEngine(dispatcher=dispatcher)
    except Exception:
        return None


def _load_fft_dispatcher() -> Dispatcher | None:
    target = os.getenv("TNFR_FFT_DISPATCHER")
    if not target:
        return None

    normalized = target.strip().lower()
    if normalized in {"queue", "queue://", "queue://local"}:
        return _build_queue_dispatcher()

    if target.startswith("http://") or target.startswith("https://"):
        return _build_http_dispatcher(target)

    if target.startswith("local:"):
        spec = target.split(":", 1)[1]
        return _load_callable_dispatcher(
            spec,
            metadata={
                "type": "callable",
                "source": "env",
                "target": spec,
                "wrapper": "local",
            },
        )

    return _load_callable_dispatcher(
        target,
        metadata={
            "type": "callable",
            "source": "env",
            "target": target,
        },
    )


def _build_http_dispatcher(url: str) -> Dispatcher | None:
    try:
        from tnfr.dynamics.fft_dispatchers import HTTPFFTDispatcher
    except Exception:
        return None

    token = os.getenv("TNFR_FFT_AUTH_TOKEN")
    try:
        client = HTTPFFTDispatcher(base_url=url, auth_token=token)
    except Exception:
        return None
    _annotate_dispatcher(
        client.dispatch,
        {
            "type": "http",
            "source": "env",
            "base_url": url,
            "token_provided": bool(token),
        },
    )
    return client.dispatch


def _load_callable_dispatcher(
    target: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Dispatcher | None:
    try:
        module_name, func_name = target.rsplit(":", 1)
    except ValueError:
        return None

    try:
        module = importlib.import_module(module_name)
        dispatcher = getattr(module, func_name)
    except Exception:
        return None

    if not callable(dispatcher):
        return None
    callable_dispatcher = cast(Dispatcher, dispatcher)
    if metadata:
        _annotate_dispatcher(callable_dispatcher, metadata)

    return callable_dispatcher


def _build_queue_dispatcher() -> Dispatcher | None:
    try:
        from tnfr.dynamics.fft_dispatchers import ThreadedQueueDispatcher
    except Exception:
        return None

    try:
        client = ThreadedQueueDispatcher()
    except Exception:
        return None
    _annotate_dispatcher(
        client.dispatch,
        {
            "type": "queue",
            "source": "env",
            "max_workers": 1,
        },
    )
    return client.dispatch


def _partition_config_from_env() -> PartitionPlannerConfig:
    target_size = _read_threshold_env("TNFR_PARTITION_TARGET_SIZE", default=256) or 256
    overlap = _read_threshold_env("TNFR_PARTITION_OVERLAP", default=4) or 4
    notes = os.getenv("TNFR_PARTITION_NOTES", "auto_env")
    return PartitionPlannerConfig(
        target_size=max(1, target_size),
        boundary_overlap=max(0, overlap),
        notes=notes,
    )


def _read_threshold_env(name: str, default: int) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


try:  # Optional sympy acceleration
    import sympy as _sp  # type: ignore[import]

    def _prime_factorization(n: int) -> Dict[int, int]:
        return {int(p): int(e) for p, e in _sp.factorint(n).items()} if n > 1 else {}

except Exception:  # pragma: no cover - sympy optional

    def _prime_factorization(n: int) -> Dict[int, int]:
        factors: Dict[int, int] = {}
        d = 2
        m = n
        while d * d <= m:
            while m % d == 0:
                factors[d] = factors.get(d, 0) + 1
                m //= d
            d += 1 if d == 2 else 2
        if m > 1:
            factors[m] = factors.get(m, 0) + 1
        return factors


def _tau_from_factors(factors: Dict[int, int]) -> int:
    tau = 1
    for exponent in factors.values():
        tau *= exponent + 1
    return tau if factors else 1


def _sigma_from_factors(factors: Dict[int, int]) -> int:
    sigma = 1
    for prime, exponent in factors.items():
        sigma *= (prime ** (exponent + 1) - 1) // (prime - 1)
    return sigma if factors else 1
