"""Batch runner that forces failure telemetry artifacts for inspection."""

# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Sequence

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))


def _load_factorizer(max_nodes: int, telemetry_root: Path) -> Any:
    try:
        module = importlib.import_module("tnfr_factorization")
    except ModuleNotFoundError as exc:  # pragma: no cover - direct script usage guard
        raise SystemExit(
            "tnfr_factorization package not found. Run this script from the repo root with "
            "PYTHONPATH set to 'src;factorization-lab'."
        ) from exc

    factorizer_cls = getattr(module, "SpectralPaleyFactorizer")
    return factorizer_cls(
        max_nodes=max_nodes,
        failure_telemetry=True,
        failure_telemetry_root=telemetry_root,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "numbers",
        nargs="*",
        type=int,
        help="Explicit integers to analyze. If omitted, provide --range-start/--range-stop.",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=None,
        help="Inclusive start for automatic sweep (used when numbers are omitted).",
    )
    parser.add_argument(
        "--range-stop",
        type=int,
        default=None,
        help="Inclusive stop for automatic sweep (used when numbers are omitted).",
    )
    parser.add_argument(
        "--range-step",
        type=int,
        default=1,
        help="Step for automatic sweep (default 1).",
    )
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=Path("results") / "failure_telemetry" / "batch_demo",
        help="Directory where failure telemetry artifacts should be written.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1025,
        help="Optional override for the Paley graph size.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=25,
        help="Print progress every N numbers when running large ranges.",
    )
    return parser.parse_args()


def _summarize_result(result: Any) -> dict:
    return {
        "n": result.n,
        "candidates": len(result.candidate_factors or []),
        "tnfr_certified": list(result.tnfr_certified_factors or []),
        "failure_diagnostics": result.failure_diagnostics,
        "notes": result.notes,
    }


def _resolve_targets(args: argparse.Namespace) -> list[int]:
    explicit = [value for value in (args.numbers or []) if value >= 2]
    if explicit:
        # Preserve insertion order while removing duplicates
        return list(dict.fromkeys(explicit))

    if args.range_start is None or args.range_stop is None:
        raise SystemExit(
            "Provide --range-start/--range-stop or explicit numbers to analyze."
        )
    if args.range_start > args.range_stop:
        raise SystemExit("--range-start must be <= --range-stop")

    start = max(2, args.range_start)
    stop = args.range_stop
    step = max(1, args.range_step)
    return list(range(start, stop + 1, step))


def run_batch(
    numbers: Sequence[int],
    telemetry_root: Path,
    max_nodes: int,
    *,
    status_interval: int = 25,
) -> list[dict]:
    telemetry_root = telemetry_root.expanduser()
    telemetry_root.mkdir(parents=True, exist_ok=True)

    factorizer = _load_factorizer(
        max_nodes=max_nodes,
        telemetry_root=telemetry_root,
    )

    interval = max(1, status_interval)
    total = len(numbers)
    summaries: list[dict] = []
    for idx, n in enumerate(numbers, start=1):
        prefix = f"[tnfr {idx}/{total}]" if total > 1 else "[tnfr]"
        result = factorizer.analyze(n)
        summary = _summarize_result(result)
        summaries.append(summary)

        should_log = interval == 1 or idx == 1 or idx == total or (idx % interval == 0)
        if not should_log:
            continue

        if summary["tnfr_certified"]:
            print(
                f"{prefix} n={n} certified factors {summary['tnfr_certified']} (telemetry skipped)"
            )
        else:
            diag = summary["failure_diagnostics"] or {}
            stage = diag.get("failure_stage") if isinstance(diag, dict) else None
            reason = diag.get("failure_reason") if isinstance(diag, dict) else None
            print(
                f"{prefix} n={n} failure telemetry captured: stage={stage} reason={reason}"
            )

    return summaries


def _find_manifest(root: Path) -> Path:
    manifest = root / "failure_manifest.json"
    if manifest.exists():
        return manifest
    raise FileNotFoundError(f"No failure manifest at {manifest}")


def _print_manifest_hint(manifest: Path) -> None:
    print("\n--- Telemetry manifest ---")
    print(manifest)
    print("Inspect this file to validate bottleneck codes and recommendations.")


if __name__ == "__main__":
    args = _parse_args()
    targets = _resolve_targets(args)
    summaries = run_batch(
        targets,
        args.telemetry_root,
        args.max_nodes,
        status_interval=args.status_interval,
    )
    for entry in summaries:
        if entry["tnfr_certified"]:
            continue
        diag = entry["failure_diagnostics"] or {}
        recommendation = diag.get("recommendations") if isinstance(diag, dict) else None
        print(f"n={entry['n']} recommendations={recommendation}")
    manifest_path = _find_manifest(args.telemetry_root)
    _print_manifest_hint(manifest_path)
