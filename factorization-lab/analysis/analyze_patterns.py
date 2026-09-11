"""Pattern analysis toolkit for TNFR certificate manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "results" / "analysis" / "certificate_manifest.json"
_DEFAULT_OUTPUT = _REPO_ROOT / "results" / "patterns" / "pattern_summary.json"
_DEFAULT_CSV = _REPO_ROOT / "results" / "patterns" / "pattern_manifest.csv"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze TNFR certificate manifests, correlate detector patterns, and emit reusable signatures."
    )
    parser.add_argument(
        "--manifest",
        default=str(_DEFAULT_MANIFEST),
        help="Path to certificate_manifest.json (default: results/analysis/certificate_manifest.json)",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help="Where to write the aggregated JSON report (default: results/patterns/pattern_summary.json)",
    )
    parser.add_argument(
        "--export-csv",
        default=str(_DEFAULT_CSV),
        help="Optional CSV path for the flattened manifest entries",
    )
    parser.add_argument(
        "--min-signature-support",
        type=int,
        default=4,
        help="Minimum sample count required before emitting a reusable pattern signature (default: 4)",
    )
    return parser.parse_args(argv)


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_manifest(doc: Dict[str, Any]) -> pd.DataFrame:
    manifest_entries = doc.get("manifest", [])
    frame = pd.DataFrame(manifest_entries)
    if frame.empty:
        raise ValueError(
            "Manifest contains no entries; run certificate_manifest.py first."
        )
    numeric_cols = [
        "n",
        "candidate_factor",
        "phi_s",
        "phase_gradient",
        "phase_curvature",
        "coherence_length",
        "coherence_score",
        "delta_nfr",
        "local_coherence",
        "arith_factorization_pressure",
        "arith_divisor_pressure",
        "arith_sigma_pressure",
        "modulus",
        "partition_count",
        "candidate_partitions",
        "coherence_ratio_min",
        "coherence_ratio_max",
        "coherence_ratio_finite",
    ]
    for column in numeric_cols:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["tnfr_verification_passed"] = frame["tnfr_verification_passed"].astype(
        "float"
    )
    return frame


def _histogram(series: pd.Series, bins: int = 24) -> Dict[str, Any]:
    cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
    if cleaned.empty:
        return {"bins": [], "edges": []}
    counts, bin_edges = np.histogram(cleaned, bins=bins)
    return {"bins": counts.astype(int).tolist(), "edges": bin_edges.tolist()}


def _scatter_stats(frame: pd.DataFrame) -> Dict[str, Any]:
    trimmed = frame.dropna(subset=["phase_gradient", "arith_factorization_pressure"])
    if trimmed.empty:
        return {}
    corr = trimmed["phase_gradient"].corr(trimmed["arith_factorization_pressure"])
    return {
        "pearson": float(corr) if not math.isnan(corr) else None,
        "samples": int(trimmed.shape[0]),
        "phase_gradient": {
            "min": float(trimmed["phase_gradient"].min()),
            "max": float(trimmed["phase_gradient"].max()),
        },
        "arith_factorization_pressure": {
            "min": float(trimmed["arith_factorization_pressure"].min()),
            "max": float(trimmed["arith_factorization_pressure"].max()),
        },
    }


def _bucket_coherence(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf]
    labels = ["<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">=2.0"]
    return pd.cut(series, bins=bins, labels=labels)


def _expand_patterns(doc: Dict[str, Any]) -> pd.DataFrame:
    pattern_report = doc.get("pattern_analysis", {})
    results = pattern_report.get("results", [])
    rows: List[Dict[str, Any]] = []
    for result in results:
        detectors = result.get("detectors") or {}
        for detector_name, patterns in detectors.items():
            if not isinstance(patterns, list):
                continue
            for pattern in patterns:
                if not isinstance(pattern, dict):
                    continue
                if pattern.get("type") == "error":
                    rows.append(
                        {
                            "certificate_path": result.get("certificate_path"),
                            "n": result.get("n"),
                            "modulus": result.get("modulus"),
                            "detector": detector_name,
                            "pattern_type": "error",
                            "error": pattern.get("message"),
                        }
                    )
                    continue
                rows.append(
                    {
                        "certificate_path": result.get("certificate_path"),
                        "n": result.get("n"),
                        "modulus": result.get("modulus"),
                        "detector": detector_name,
                        "pattern_type": pattern.get("type"),
                        "confidence": pattern.get("confidence"),
                        "temporal_scale": pattern.get("temporal_scale"),
                        "spatial_scale": pattern.get("spatial_scale"),
                        "prediction_horizon": pattern.get("prediction_horizon"),
                        "compression_ratio": pattern.get("compression_ratio"),
                    }
                )
    return pd.DataFrame(rows)


def _pattern_pivot(patterns: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    if patterns.empty:
        return {}
    pivot = patterns.pivot_table(
        index="pattern_type",
        columns="coherence_bucket",
        values="certificate_path",
        aggfunc="count",
        fill_value=0,
    )
    return {idx: row.dropna().astype(int).to_dict() for idx, row in pivot.iterrows()}


def _recommend_sequence(combo: List[str]) -> List[str] | None:
    combo_set = set(combo)
    if {"spectral_cascade", "entropy_flow"}.issubset(combo_set):
        return ["UM", "RA", "IL", "THOL"]
    if {"eigenmode_resonance", "topological_invariant"}.issubset(combo_set):
        return ["AL", "IL", "RA", "SHA"]
    if "fractal_scaling" in combo_set:
        return ["UM", "RA", "REMESH", "IL"]
    return None


def _derive_signatures(
    patterns: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    min_support: int,
) -> List[Dict[str, Any]]:
    if patterns.empty:
        return []
    combos = (
        patterns.groupby("certificate_path")["pattern_type"]
        .apply(
            lambda values: tuple(sorted(set(v for v in values if v and v != "error")))
        )
        .reset_index(name="combo")
    )
    combos = combos[combos["combo"].map(len) > 0]
    if combos.empty:
        return []
    combos = combos.merge(
        manifest[
            [
                "certificate_path",
                "candidate_partitions",
                "partition_count",
                "modulus",
                "tnfr_verification_passed",
            ]
        ],
        on="certificate_path",
        how="left",
    )
    grouped = combos.groupby("combo").agg(
        sample_size=("certificate_path", "count"),
        avg_candidate_partitions=("candidate_partitions", "mean"),
        avg_partition_count=("partition_count", "mean"),
        avg_modulus=("modulus", "mean"),
        tnfr_success_rate=("tnfr_verification_passed", "mean"),
    )
    grouped = grouped[grouped["sample_size"] >= max(1, min_support)]
    signatures: List[Dict[str, Any]] = []
    for combo, stats in grouped.sort_values("sample_size", ascending=False).iterrows():
        combo_list = list(combo)
        signatures.append(
            {
                "pattern_combo": combo_list,
                "sample_size": int(stats["sample_size"]),
                "avg_candidate_partitions": float(stats["avg_candidate_partitions"]),
                "avg_partition_count": float(stats["avg_partition_count"]),
                "avg_modulus": float(stats["avg_modulus"]),
                "tnfr_success_rate": (
                    float(stats["tnfr_success_rate"])
                    if not math.isnan(stats["tnfr_success_rate"])
                    else None
                ),
                "recommended_sequence": _recommend_sequence(combo_list),
            }
        )
    return signatures


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser()
    output_path = Path(args.output).expanduser()
    csv_path = Path(args.export_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    doc = _load_manifest(manifest_path)
    pattern_block = doc.get("pattern_analysis") or {}
    pattern_metadata = {
        "engine_metadata": pattern_block.get("engine_metadata"),
        "detector_warnings": pattern_block.get("detector_warnings"),
    }
    manifest_df = _normalize_manifest(doc)
    manifest_df["coherence_bucket"] = _bucket_coherence(
        manifest_df["coherence_ratio_max"]
    )

    patterns_df = _expand_patterns(doc)
    if not patterns_df.empty:
        patterns_df = patterns_df.merge(
            manifest_df[
                ["certificate_path", "coherence_bucket", "tnfr_verification_passed"]
            ],
            on="certificate_path",
            how="left",
        )

    telemetry_summary = {
        "phi_s_hist": _histogram(manifest_df["phi_s"]),
        "phase_gradient_hist": _histogram(manifest_df["phase_gradient"]),
        "phase_curvature_hist": _histogram(manifest_df["phase_curvature"]),
        "coherence_length_hist": _histogram(manifest_df["coherence_length"]),
        "phase_gradient_vs_arith_factorization_pressure": _scatter_stats(manifest_df),
    }

    pattern_counts = (
        patterns_df["pattern_type"].value_counts().astype(int).to_dict()
        if not patterns_df.empty
        else {}
    )
    coherence_pivot = _pattern_pivot(patterns_df) if not patterns_df.empty else {}
    signatures = _derive_signatures(
        patterns_df,
        manifest_df,
        min_support=args.min_signature_support,
    )

    summary = {
        "timestamp": time.time(),
        "manifest_path": str(manifest_path.relative_to(_REPO_ROOT)),
        "entry_count": int(manifest_df.shape[0]),
        "pattern_entry_count": int(
            patterns_df.shape[0] if not patterns_df.empty else 0
        ),
        "telemetry_summary": telemetry_summary,
        "pattern_counts": pattern_counts,
        "coherence_pattern_pivot": coherence_pivot,
        "signatures": signatures,
        "pattern_metadata": pattern_metadata,
    }

    output_path.write_text(json.dumps(summary, indent=2))
    manifest_df.to_csv(csv_path, index=False)
    if summary["pattern_entry_count"] == 0:
        warnings = pattern_metadata.get("detector_warnings")
        if warnings:
            print("Pattern detectors were skipped:", warnings)
    print(
        "Pattern analysis saved to",
        output_path,
        "with",
        summary["entry_count"],
        "entries and",
        summary["pattern_entry_count"],
        "pattern rows",
    )


def generate_optimization_manifest(
    certificate_manifest_path: Path,
    output_dir: Path,
    batch_id: str,
    certificate_filter: Dict[str, Any] | None = None,
) -> Dict[str, Path]:
    """Generate self-optimization manifest for batch certificate processing.

    Parameters
    ----------
    certificate_manifest_path : Path
        Path to the certificate_manifest.json to analyze.
    output_dir : Path
        Directory where optimization manifests will be written.
    batch_id : str
        Unique identifier for this batch optimization operation.
    certificate_filter : Dict[str, Any], optional
        Filter criteria for selecting certificates (e.g., {"coherence_ratio_max": {"min": 0.7}}).

    Returns
    -------
    Dict[str, Path]
        Dictionary with keys 'manifest_absolute' and 'summary_absolute'
        pointing to the generated manifest files.

    Notes
    -----
    Manifest format compatible with self_opt_support pipeline:
    - operation_type: 'batch_certificate_optimization'
    - batch_id: unique identifier
    - certificates: list of certificate paths with telemetry
    - filter_criteria: applied selection filters
    - aggregate_telemetry: summary statistics across batch
    """
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalize manifest
    doc = _load_manifest(certificate_manifest_path)
    manifest_df = _normalize_manifest(doc)

    # Apply filters if provided
    if certificate_filter:
        for key, criteria in certificate_filter.items():
            if key in manifest_df.columns:
                if "min" in criteria:
                    manifest_df = manifest_df[manifest_df[key] >= criteria["min"]]
                if "max" in criteria:
                    manifest_df = manifest_df[manifest_df[key] <= criteria["max"]]

    # Serialize certificates with telemetry
    certificates_serialized = []
    for _, row in manifest_df.iterrows():
        cert_data = {
            "certificate_path": str(row["certificate_path"]),
            "modulus": int(row["modulus"]) if not pd.isna(row["modulus"]) else None,
            "coherence_ratio_max": (
                float(row["coherence_ratio_max"])
                if not pd.isna(row["coherence_ratio_max"])
                else None
            ),
            "phi_s": float(row["phi_s"]) if not pd.isna(row["phi_s"]) else None,
            "phase_gradient": (
                float(row["phase_gradient"])
                if not pd.isna(row["phase_gradient"])
                else None
            ),
            "phase_curvature": (
                float(row["phase_curvature"])
                if not pd.isna(row["phase_curvature"])
                else None
            ),
            "coherence_length": (
                float(row["coherence_length"])
                if not pd.isna(row["coherence_length"])
                else None
            ),
            "tnfr_verification_passed": (
                bool(row["tnfr_verification_passed"])
                if not pd.isna(row["tnfr_verification_passed"])
                else None
            ),
        }
        certificates_serialized.append(cert_data)

    # Compute aggregate telemetry
    aggregate_telemetry = {
        "certificate_count": len(certificates_serialized),
        "avg_coherence_ratio_max": (
            float(manifest_df["coherence_ratio_max"].mean())
            if "coherence_ratio_max" in manifest_df.columns
            else None
        ),
        "avg_phi_s": (
            float(manifest_df["phi_s"].mean())
            if "phi_s" in manifest_df.columns
            else None
        ),
        "avg_phase_gradient": (
            float(manifest_df["phase_gradient"].mean())
            if "phase_gradient" in manifest_df.columns
            else None
        ),
        "verification_success_rate": (
            float(manifest_df["tnfr_verification_passed"].mean())
            if "tnfr_verification_passed" in manifest_df.columns
            else None
        ),
    }

    # Build manifest
    manifest = {
        "operation_type": "batch_certificate_optimization",
        "batch_id": batch_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(certificate_manifest_path),
        "filter_criteria": certificate_filter or {},
        "certificates": certificates_serialized,
        "aggregate_telemetry": aggregate_telemetry,
    }

    # Write manifest
    manifest_path = output_dir / "batch_optimization_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write summary
    summary = {
        "operation_type": "batch_certificate_optimization",
        "batch_id": batch_id,
        "certificate_count": len(certificates_serialized),
        "avg_coherence_ratio_max": aggregate_telemetry["avg_coherence_ratio_max"],
        "verification_success_rate": aggregate_telemetry["verification_success_rate"],
    }
    summary_path = output_dir / "batch_optimization_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "manifest_absolute": manifest_path.resolve(),
        "summary_absolute": summary_path.resolve(),
    }


if __name__ == "__main__":
    main()
