"""SDK helpers for TNFR self-optimization workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "run_partition_self_optimization",
    "run_pattern_discovery_optimization",
    "run_fractal_partition_optimization",
    "run_batch_certificate_optimization",
]


def run_partition_self_optimization(
    manifest_path: Path | str,
    *,
    manifest_summary_path: Path | str | None = None,
    base_name: Optional[str] = None,
    operation_type: str = "paley_partition",
    output_root: Path | str | None = None,
) -> Dict[str, Any] | None:
    """Run the CLI-driven self-optimization pipeline for a partition manifest.

    Parameters
    ----------
    manifest_path:
        Path to the partition manifest JSON produced by a factorization workflow.
    manifest_summary_path:
        Optional path to the manifest summary emitted alongside the manifest.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    operation_type:
        Operation label passed to the CLI runner (defaults to ``"paley_partition"``).
    output_root:
        Optional directory under which all self-optimization payloads should be stored.
    """

    manifest_path_obj = Path(manifest_path).resolve()
    summary_path_obj = Path(manifest_summary_path).resolve() if manifest_summary_path else None
    payload_root = Path(output_root).resolve() if output_root else None
    effective_base_name = base_name or manifest_path_obj.stem

    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

    try:
        from tnfr_factorization.self_opt_support import run_partition_self_optimization as _runner
    except ImportError as exc:  # pragma: no cover - optional dependency in minimal installs
        raise RuntimeError(
            "Self-optimization helpers require the 'tnfr_factorization' package to be available"
        ) from exc

    return _runner(
        manifest_path=manifest_path_obj,
        manifest_summary_path=summary_path_obj,
        base_name=effective_base_name,
        operation_type=operation_type,
        output_root=payload_root,
    )


def run_pattern_discovery_optimization(
    manifest_path: Path | str,
    *,
    manifest_summary_path: Path | str | None = None,
    base_name: Optional[str] = None,
    output_root: Path | str | None = None,
) -> Dict[str, Any] | None:
    """Run self-optimization pipeline for pattern discovery results.

    Parameters
    ----------
    manifest_path:
        Path to the pattern_manifest.json from TNFREmergentPatternEngine.export_pattern_manifest().
    manifest_summary_path:
        Optional path to the pattern_summary.json file.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    output_root:
        Optional directory under which all self-optimization payloads should be stored.

    Returns
    -------
    Dict[str, Any] | None
        Summary of the self-optimization run, or None if unsuccessful.

    Examples
    --------
    >>> from tnfr.sdk import run_pattern_discovery_optimization
    >>> result = run_pattern_discovery_optimization(
    ...     "results/patterns/pattern_manifest.json",
    ...     base_name="experiment_42"
    ... )
    """
    return run_partition_self_optimization(
        manifest_path=manifest_path,
        manifest_summary_path=manifest_summary_path,
        base_name=base_name,
        operation_type="pattern_discovery",
        output_root=output_root,
    )


def run_fractal_partition_optimization(
    manifest_path: Path | str,
    *,
    manifest_summary_path: Path | str | None = None,
    base_name: Optional[str] = None,
    output_root: Path | str | None = None,
) -> Dict[str, Any] | None:
    """Run self-optimization pipeline for fractal partition results.

    Parameters
    ----------
    manifest_path:
        Path to the fractal_partition_manifest.json from FractalPartitioner.partition_with_manifest().
    manifest_summary_path:
        Optional path to the fractal_partition_summary.json file.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    output_root:
        Optional directory under which all self-optimization payloads should be stored.

    Returns
    -------
    Dict[str, Any] | None
        Summary of the self-optimization run, or None if unsuccessful.

    Examples
    --------
    >>> from tnfr.sdk import run_fractal_partition_optimization
    >>> result = run_fractal_partition_optimization(
    ...     "results/partitions/fractal_partition_manifest.json",
    ...     base_name="network_42"
    ... )
    """
    return run_partition_self_optimization(
        manifest_path=manifest_path,
        manifest_summary_path=manifest_summary_path,
        base_name=base_name,
        operation_type="fractal_partition",
        output_root=output_root,
    )


def run_batch_certificate_optimization(
    manifest_path: Path | str,
    *,
    manifest_summary_path: Path | str | None = None,
    base_name: Optional[str] = None,
    output_root: Path | str | None = None,
) -> Dict[str, Any] | None:
    """Run self-optimization pipeline for batch certificate processing.

    Parameters
    ----------
    manifest_path:
        Path to the batch_optimization_manifest.json from analyze_patterns.generate_optimization_manifest().
    manifest_summary_path:
        Optional path to the batch_optimization_summary.json file.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    output_root:
        Optional directory under which all self-optimization payloads should be stored.

    Returns
    -------
    Dict[str, Any] | None
        Summary of the self-optimization run, or None if unsuccessful.

    Examples
    --------
    >>> from tnfr.sdk import run_batch_certificate_optimization
    >>> result = run_batch_certificate_optimization(
    ...     "results/batch/batch_optimization_manifest.json",
    ...     base_name="batch_42"
    ... )
    """
    return run_partition_self_optimization(
        manifest_path=manifest_path,
        manifest_summary_path=manifest_summary_path,
        base_name=base_name,
        operation_type="batch_certificate_optimization",
        output_root=output_root,
    )
