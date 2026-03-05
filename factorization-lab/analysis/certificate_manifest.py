"""Certificate telemetry manifest and optional TNFR pattern discovery."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAB_ROOT = _REPO_ROOT / "factorization-lab"
_SRC_ROOT = _REPO_ROOT / "src"

for candidate in (_LAB_ROOT, _SRC_ROOT):
    candidate_str = candidate.as_posix()
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_DEFAULT_CERT_DIR = _REPO_ROOT / "results" / "certificates"
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "analysis" / "certificate_manifest.json"

try:  # Factorization helpers (Paley graphs live in the lab package).
    from tnfr_factorization.spectral_paley import (  # type: ignore[attr-defined,import-not-found]  # noqa: E402
        _build_paley_graph,
        _annotate_graph_for_fft,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - propagates with actionable hint
    raise RuntimeError(
        "Unable to import tnfr_factorization; run from the repository root so the lab package is on PYTHONPATH."
    ) from exc

try:  # Pattern engine for emergent mathematical structure detection.
    from tnfr.engines.pattern_discovery.mathematical_patterns import (  # noqa: E402
        EmergentPattern,
        TNFREmergentPatternEngine,
    )
    _PATTERN_ENGINE_ERROR: str | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    EmergentPattern = Any  # type: ignore[assignment]
    TNFREmergentPatternEngine = None  # type: ignore[assignment]
    _PATTERN_ENGINE_ERROR = str(exc)

DetectorFn = Any

_DETECTORS: Dict[str, DetectorFn] = {
    "eigenmode": lambda engine, G: engine.discover_eigenmode_resonances(G),
    "spectral": lambda engine, G: engine.discover_spectral_cascades(G),
    "entropy": lambda engine, G: engine.discover_entropy_flow_patterns(G),
    "topology": lambda engine, G: engine.discover_topological_invariants(G),
}
_DETECTOR_DEPENDENCIES: Dict[str, Sequence[str]] = {
    "eigenmode": ("spectral", "scipy"),
    "spectral": ("spectral", "physics_fields", "scipy"),
    "topology": ("spectral",),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate certificate telemetry into a manifest and optionally run "
            "TNFR emergent pattern discovery over the associated Paley graphs."
        )
    )
    parser.add_argument(
        "--cert-dir",
        default=str(_DEFAULT_CERT_DIR),
        help="Directory containing certificate_*.json files"
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Path for the manifest JSON output"
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="Enable emergent pattern discovery pass"
    )
    parser.add_argument(
        "--pattern-depth",
        choices=("shallow", "medium", "deep"),
        default="medium",
        help="Discovery depth for TNFREmergentPatternEngine (default: medium)"
    )
    parser.add_argument(
        "--pattern-limit",
        type=int,
        default=128,
        help="Maximum number of certificates to feed into pattern discovery (default: 128)."
    )
    parser.add_argument(
        "--detectors",
        default="eigenmode,spectral",
        help="Comma-separated list of detectors (eigenmode,spectral,entropy,topology)."
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _aggregate_partitions(states: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(states, Mapping):
        return {
            "partition_count": 0,
            "candidate_partitions": 0,
            "coherence_ratio_min": None,
            "coherence_ratio_max": None,
            "coherence_ratio_finite": 0,
            "inferred_factors": [],
        }

    coherence_values: List[float] = []
    inferred: List[int] = []
    candidate_partitions = 0

    for entry in states.values():
        if not isinstance(entry, Mapping):
            continue
        after = entry.get("after") or {}
        ratio = after.get("coherence_ratio")
        if isinstance(ratio, (int, float)):
            if math.isfinite(ratio):
                coherence_values.append(float(ratio))
        inferred_factor = after.get("inferred_factor")
        if inferred_factor is not None:
            inferred_factor = _safe_int(inferred_factor)
            if inferred_factor:
                inferred.append(inferred_factor)
        candidates = entry.get("candidate_factors") or []
        if isinstance(candidates, Iterable) and any(_safe_int(c) for c in candidates):
            candidate_partitions += 1

    coherence_values.sort()
    return {
        "partition_count": len(states),
        "candidate_partitions": candidate_partitions,
        "coherence_ratio_min": coherence_values[0] if coherence_values else None,
        "coherence_ratio_max": coherence_values[-1] if coherence_values else None,
        "coherence_ratio_finite": len(coherence_values),
        "inferred_factors": inferred,
    }


def _manifest_entry(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    telemetry = data.get("telemetry", {})
    partitions = _aggregate_partitions(data.get("partition_states"))
    entry = {
        "certificate_path": str(path.relative_to(_REPO_ROOT)),
        "n": _safe_int(data.get("n")),
        "candidate_factor": _safe_int(data.get("candidate_factor")),
        "phi_s": _safe_float(telemetry.get("phi_s")),
        "phase_gradient": _safe_float(telemetry.get("phase_gradient")),
        "phase_curvature": _safe_float(telemetry.get("phase_curvature")),
        "coherence_length": _safe_float(telemetry.get("coherence_length")),
        "coherence_score": _safe_float(telemetry.get("coherence_score")),
        "delta_nfr": _safe_float(telemetry.get("delta_nfr")),
        "local_coherence": _safe_float(telemetry.get("local_coherence")),
        "arith_factorization_pressure": _safe_float(telemetry.get("arith_factorization_pressure")),
        "arith_divisor_pressure": _safe_float(telemetry.get("arith_divisor_pressure")),
        "arith_sigma_pressure": _safe_float(telemetry.get("arith_sigma_pressure")),
        "modulus": _safe_int(data.get("partitions", {}).get("summary", {}).get("modulus")),
        "operator_count": len(data.get("operators", [])),
        **partitions,
    }
    tnfr_verification = data.get("tnfr_verification_snapshot") or data.get("tnfr_verification")
    if isinstance(tnfr_verification, Mapping):
        entry["tnfr_verification_passed"] = bool(tnfr_verification.get("passed", True))
        entry["tnfr_verification_hash"] = tnfr_verification.get("verification_hash")
    else:
        entry["tnfr_verification_passed"] = None
        entry["tnfr_verification_hash"] = None
    return entry


def build_manifest(cert_dir: Path) -> List[Dict[str, Any]]:
    cert_paths = sorted(cert_dir.glob("certificate_*.json"))
    manifest = []
    for path in cert_paths:
        try:
            manifest.append(_manifest_entry(path))
        except Exception as exc:  # telemetry is best-effort, log in entry
            manifest.append(
                {
                    "certificate_path": str(path.relative_to(_REPO_ROOT)),
                    "error": str(exc),
                }
            )
    return manifest


def _pattern_to_dict(pattern: EmergentPattern) -> Dict[str, Any]:
    return {
        "type": pattern.pattern_type.value,
        "confidence": pattern.discovery_confidence,
        "temporal_scale": pattern.temporal_scale,
        "spatial_scale": pattern.spatial_scale,
        "prediction_horizon": pattern.prediction_horizon,
        "compression_ratio": pattern.compression_ratio,
        "physical_interpretation": pattern.physical_interpretation,
        "applications": list(pattern.applications),
        "mathematical_signature": pattern.mathematical_signature,
    }


def discover_patterns(
    manifest: Sequence[Mapping[str, Any]],
    *,
    depth: str,
    limit: int,
    detector_names: Sequence[str],
) -> Dict[str, Any]:
    if TNFREmergentPatternEngine is None:
        raise RuntimeError(
            "Pattern discovery requested but TNFREmergentPatternEngine is unavailable: "
            f"{_PATTERN_ENGINE_ERROR or 'unknown import error'}"
        )
    engine = TNFREmergentPatternEngine(analysis_depth=depth)
    engine_metadata = engine.get_discovery_statistics()
    module_availability = engine_metadata.get("available_modules", {})

    def _module_ready(name: str) -> bool:
        value = module_availability.get(name)
        return True if value is None else bool(value)

    detectors = [name.strip().lower() for name in detector_names if name.strip()]
    detectors = [name for name in detectors if name in _DETECTORS]
    if not detectors:
        detectors = ["eigenmode", "spectral"]

    detector_warnings: Dict[str, str] = {}
    runnable_detectors: List[str] = []
    for detector_name in detectors:
        missing = [dep for dep in _DETECTOR_DEPENDENCIES.get(detector_name, ()) if not _module_ready(dep)]
        if missing:
            detector_warnings[detector_name] = (
                f"skipped (missing modules: {', '.join(sorted(missing))})"
            )
            continue
        runnable_detectors.append(detector_name)

    if not runnable_detectors:
        detector_warnings["__global__"] = "no detectors runnable with current engine configuration"

    analyzed = 0
    pattern_results: List[Dict[str, Any]] = []

    for entry in manifest:
        if limit is not None and analyzed >= limit:
            break
        modulus = entry.get("modulus")
        if not isinstance(modulus, int) or modulus < 5:
            continue
        try:
            graph = _build_paley_graph(modulus)
            _annotate_graph_for_fft(graph)
        except Exception as exc:
            entry_block = {
                "certificate_path": entry.get("certificate_path"),
                "error": f"graph_build_failed: {exc}",
            }
            pattern_results.append(entry_block)
            continue

        analyzed += 1
        detector_payload: Dict[str, List[Dict[str, Any]]] = {name: [] for name in detectors}

        for detector_name in runnable_detectors:
            detector_fn = _DETECTORS[detector_name]
            try:
                patterns = detector_fn(engine, graph) or []
            except Exception as exc:  # continue collecting other detectors
                detector_payload[detector_name] = [
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                ]
                continue
            detector_payload[detector_name] = [
                _pattern_to_dict(pattern) for pattern in patterns
            ]

        pattern_results.append(
            {
                "certificate_path": entry.get("certificate_path"),
                "n": entry.get("n"),
                "modulus": modulus,
                "detectors": detector_payload,
            }
        )

    return {
        "detectors": detectors,
        "analyzed_certificates": analyzed,
        "engine_metadata": engine_metadata,
        "detector_warnings": detector_warnings,
        "results": pattern_results,
    }


def save_manifest(
    *,
    manifest: Sequence[Mapping[str, Any]],
    output: Path,
    cert_dir: Path,
    pattern_report: Mapping[str, Any] | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.time(),
        "certificate_dir": str(cert_dir.relative_to(_REPO_ROOT)),
        "certificate_count": len(manifest),
        "manifest": list(manifest),
    }
    if pattern_report is not None:
        payload["pattern_analysis"] = pattern_report
    output.write_text(json.dumps(payload, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cert_dir = Path(args.cert_dir).expanduser()
    output = Path(args.output).expanduser()
    manifest = build_manifest(cert_dir)

    pattern_report = None
    if args.patterns:
        detector_names = args.detectors.split(",") if args.detectors else []
        try:
            pattern_report = discover_patterns(
                manifest,
                depth=args.pattern_depth,
                limit=args.pattern_limit,
                detector_names=detector_names,
            )
        except RuntimeError as exc:
            print("Pattern discovery skipped:", exc, file=sys.stderr)

    save_manifest(
        manifest=manifest,
        output=output,
        cert_dir=cert_dir,
        pattern_report=pattern_report,
    )
    print(
        "Certificate manifest saved to",
        output,
        "covering",
        len(manifest),
        "certificates",
        "with pattern analysis" if pattern_report else "",
    )


if __name__ == "__main__":
    main()
