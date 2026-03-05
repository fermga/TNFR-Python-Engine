"""Shared helpers for TNFR self-optimization workflows.

This module centralizes the logic that coordinates the CLI-based
self-optimization runner/validator so any factorization workflow
can trigger the pipeline without duplicating boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "run_partition_self_optimization",
    "attach_self_opt_sequences",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_DEFAULT_SELF_OPT_ROOT = _RESULTS_ROOT / "self_optimization"


def run_partition_self_optimization(
    *,
    manifest_path: Path | None,
    manifest_summary_path: Path | None,
    base_name: str,
    operation_type: str,
    output_root: Path | None = None,
) -> Dict[str, Any] | None:
    """Execute the CLI self-optimization workflow for a manifest.

    Parameters
    ----------
    manifest_path:
        Absolute path to the partition manifest (required).
    manifest_summary_path:
        Absolute path to the manifest summary file (optional).
    base_name:
        Base identifier used to create the payload directory.
    operation_type:
        Operation tag forwarded to the runner CLI (e.g. "paley_partition").
    output_root:
        Optional override for the root directory that stores the
        self-optimization payloads. Defaults to ``results/self_optimization``.
    """

    if manifest_path is None or not manifest_path.exists():
        return None

    try:
        from scripts import run_self_optimization as self_opt_runner
        from scripts import run_self_opt_validation as self_opt_validator
    except Exception:
        return None

    payload_root = ((output_root or _DEFAULT_SELF_OPT_ROOT) / base_name).resolve()
    payload_root.mkdir(parents=True, exist_ok=True)

    summary_path = payload_root / "_summary.json"
    runner_args = [
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(payload_root),
        "--operation-type",
        operation_type,
        "--capture-snapshots",
        "--summary",
        str(summary_path),
        "--quiet",
    ]
    if manifest_summary_path is not None:
        runner_args.extend(["--manifest-summary", str(manifest_summary_path)])

    parsed_runner_args = self_opt_runner.parse_args(runner_args)
    runner_summary = self_opt_runner.run(parsed_runner_args)

    validation_report = payload_root / "_validation.json"
    validator_args = self_opt_validator.parse_args(
        [
            "--payload-root",
            str(payload_root),
            "--report",
            str(validation_report),
            "--quiet",
        ]
    )
    validation_summary = self_opt_validator.run(validator_args)

    promotable = _extract_promotable_partitions(runner_summary, validation_summary)

    return {
        "payload_root": str(payload_root),
        "summary_path": str(summary_path),
        "validation_report": str(validation_report),
        "promotable": promotable,
        "runner_summary": runner_summary,
        "validation_summary": validation_summary,
    }


def attach_self_opt_sequences(
    plan: Optional[Dict[str, Any]],
    self_opt_summary: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge validated self-optimization sequences into a plan snapshot."""

    if not self_opt_summary:
        return plan

    promotable = self_opt_summary.get("promotable") or {}
    if not promotable:
        return plan

    base_plan: Dict[str, Any] = dict(plan) if isinstance(plan, dict) else {}
    seq_block: Dict[str, Any] = {}
    for partition_id, payload in promotable.items():
        telemetry = payload.get("telemetry") or {}
        engine_block = payload.get("engine") or {}
        validation_block = engine_block.get("validation") or {}
        seq_block[partition_id] = {
            "canonical_tokens": validation_block.get("canonical_tokens"),
            "tokens": validation_block.get("tokens"),
            "delta_c": telemetry.get("delta_c"),
            "delta_phi_s": telemetry.get("delta_phi_s"),
            "delta_si": telemetry.get("delta_si"),
            "signature": engine_block.get("signature"),
            "snapshot_path": engine_block.get("snapshot_path"),
        }

    if seq_block:
        self_opt_plan = base_plan.setdefault("self_optimization", {})
        self_opt_plan["validated_sequences"] = seq_block

    return base_plan


def _index_validation_status(validation_summary: Mapping[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for record in validation_summary.get("results", []):
        metadata = record.get("metadata") or {}
        partition_id = metadata.get("partition_id")
        if partition_id:
            mapping[str(partition_id)] = record.get("status") or "pending"
    return mapping


def _extract_promotable_partitions(
    runner_summary: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    validation_map = _index_validation_status(validation_summary)
    promotable: Dict[str, Any] = {}
    for entry in runner_summary.get("partition_results", []):
        if not entry.get("success"):
            continue
        partition_id = entry.get("partition_id")
        if not partition_id:
            continue
        telemetry = entry.get("telemetry") or {}
        delta_c = telemetry.get("delta_c")
        if delta_c is None or float(delta_c) <= 0.0:
            continue
        engine_block = entry.get("engine") or {}
        validation_block = engine_block.get("validation") or {}
        if not validation_block.get("passed"):
            continue
        if validation_map.get(str(partition_id)) != "validated":
            continue
        promotable[str(partition_id)] = {
            "telemetry": telemetry,
            "engine": engine_block,
            "candidate_factors": entry.get("candidate_factors"),
            "summary_entry": entry,
        }
    return promotable
