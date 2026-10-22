"""SDK helpers for TNFR self-optimization workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    base_name: str | None = None,
    operation_type: str = "paley_partition",
    output_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Collect dry-run recommendations for explicit graph partitions.

    This checkout-only workflow uses the repository's ``scripts`` package.
    It neither executes recommended operator words nor promotes them. Mapped
    regression suites assess current code, not recommendation effects; an
    operation without a mapped suite remains ``pending``.

    Parameters
    ----------
    manifest_path:
        Path to a manifest whose partition files contain serialized graphs from
        ``tnfr.engines.manifest.encode_graph`` or ``write_manifest_bundle``.
    manifest_summary_path:
        Optional path to the manifest summary emitted alongside the manifest.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    operation_type:
        Operation label passed to the CLI runner. The historical default
        ``"paley_partition"`` is only a label; it does not reconstruct a graph.
    output_root:
        Optional directory under which all self-optimization payloads should be stored.

    Returns
    -------
    dict[str, Any]
        Payload/report paths, runner and validation summaries, and an empty
        ``promotable`` mapping. Errors propagate rather than returning ``None``.
    """

    manifest_path_obj = Path(manifest_path).resolve()
    summary_path_obj = (
        Path(manifest_summary_path).resolve() if manifest_summary_path else None
    )
    effective_base_name = base_name or manifest_path_obj.stem

    if not manifest_path_obj.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path_obj}")

    try:
        from scripts import run_self_opt_validation as validator
        from scripts import run_self_optimization as runner
    except ImportError as exc:
        raise RuntimeError(
            "Self-optimization orchestration requires a TNFR repository checkout "
            "with scripts.run_self_optimization and scripts.run_self_opt_validation "
            "importable; these drivers are not included in the installed wheel."
        ) from exc

    default_root = Path(__file__).resolve().parents[3] / "results" / "self_optimization"
    payload_root = (
        (Path(output_root) if output_root else default_root) / effective_base_name
    ).resolve()
    summary_path = payload_root / "_summary.json"
    runner_args = [
        "--manifest",
        str(manifest_path_obj),
        "--output-dir",
        str(payload_root),
        "--operation-type",
        operation_type,
        "--capture-snapshots",
        "--summary",
        str(summary_path),
        "--quiet",
    ]
    if summary_path_obj is not None:
        runner_args.extend(["--manifest-summary", str(summary_path_obj)])
    runner_summary = runner.run(runner.parse_args(runner_args))

    validation_report = payload_root / "_validation.json"
    validation_summary = validator.run(
        validator.parse_args(
            [
                "--payload-root",
                str(payload_root),
                "--report",
                str(validation_report),
                "--quiet",
            ]
        )
    )
    return {
        "payload_root": str(payload_root),
        "summary_path": str(summary_path),
        "validation_report": str(validation_report),
        "promotable": {},
        "runner_summary": runner_summary,
        "validation_summary": validation_summary,
    }


def run_pattern_discovery_optimization(
    manifest_path: Path | str,
    *,
    manifest_summary_path: Path | str | None = None,
    base_name: str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Collect checkout-only dry-run recommendations for pattern discovery.

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
    dict[str, Any] | None
        Dry-run summary; errors propagate. See ``run_partition_self_optimization``.

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
    base_name: str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Collect checkout-only dry-run recommendations for fractal partitions.

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
    dict[str, Any] | None
        Dry-run summary; errors propagate. See ``run_partition_self_optimization``.

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
    base_name: str | None = None,
    output_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Collect checkout-only dry-run recommendations for a graph-manifest batch.

    Parameters
    ----------
    manifest_path:
        Path to an explicit graph-partition manifest for the batch.
    manifest_summary_path:
        Optional path to the batch_optimization_summary.json file.
    base_name:
        Optional override for the payload directory name. Defaults to the manifest stem.
    output_root:
        Optional directory under which all self-optimization payloads should be stored.

    Returns
    -------
    dict[str, Any] | None
        Dry-run summary; errors propagate. See ``run_partition_self_optimization``.

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
