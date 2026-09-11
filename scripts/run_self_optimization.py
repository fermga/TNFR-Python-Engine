"""Command-line runner for TNFR self-optimization over partition manifests."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Optional, Sequence

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
FACTOR_LAB_ROOT = REPO_ROOT / "factorization-lab"
if (
    FACTOR_LAB_ROOT.exists()
):  # Ensure tnfr_factorization is importable without installation
    sys.path.insert(0, str(FACTOR_LAB_ROOT))

from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine  # noqa: E402
from tnfr.engines.manifest import decode_graph  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "self_optimization"
DEFAULT_OPERATION = "paley_partition"
DEFAULT_MAX_WORKERS = 1


@dataclass
class PartitionWorkItem:
    partition_id: str
    path: Path
    manifest_entry: Dict[str, Any]
    source_index: int = 0


class PaleyGraphCache:
    """Caches annotated Paley graphs keyed by modulus."""

    def __init__(self) -> None:
        self._cache: Dict[int, nx.Graph] = {}
        self._lock = threading.Lock()

    def get(self, modulus: int) -> nx.Graph:
        from tnfr_factorization.spectral_paley import (
            _annotate_graph_for_fft, _build_paley_graph,
        )

        with self._lock:
            cached = self._cache.get(modulus)
            if cached is not None:
                return cached
            graph = _build_paley_graph(modulus)
            _annotate_graph_for_fft(graph)
            self._cache[modulus] = graph
            return graph


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, type=Path, help="Path to partition _manifest.json"
    )
    parser.add_argument(
        "--manifest-summary", type=Path, help="Optional manifest summary path"
    )
    parser.add_argument(
        "--partition-dir",
        type=Path,
        help="Override partition directory (defaults to manifest parent or embedded path)",
    )
    parser.add_argument(
        "--partitions",
        nargs="*",
        help="Optional list of partition IDs to process (default processes all entries)",
    )
    parser.add_argument(
        "--max-partitions", type=int, help="Maximum number of partitions to process"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Recorded seed label plus original entry index; does not reseed engine execution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for dry-run payloads and telemetry snapshots",
    )
    parser.add_argument(
        "--operation-type",
        default=None,
        help="Operation label passed to TNFRSelfOptimizingEngine",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum worker threads for partition processing",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply optimization instead of dry-run mode",
    )
    parser.add_argument(
        "--capture-snapshots",
        action="store_true",
        help="Force telemetry snapshot capture even when not in dry-run mode",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional path to write aggregated summary JSON (defaults to stdout only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-partition status logs (errors are still reported)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.perf_counter()
    manifest = _load_json(args.manifest)
    args.operation_type = args.operation_type or manifest.get("operation_type") or DEFAULT_OPERATION
    manifest_summary = (
        _load_json(args.manifest_summary) if args.manifest_summary else None
    )
    items = _collect_partition_entries(manifest, args.manifest, args.partition_dir)

    if args.partitions:
        allowed = {pid.strip() for pid in args.partitions if pid}
        items = [item for item in items if item.partition_id in allowed]
        if not items:
            raise ValueError("No manifest entries matched the requested partition IDs")

    if args.max_partitions is not None:
        if args.max_partitions <= 0:
            raise ValueError("--max-partitions must be positive when provided")
        items = items[: args.max_partitions]

    if not items:
        raise ValueError("Manifest did not provide any partition entries")

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_cache = PaleyGraphCache()
    engine = TNFRSelfOptimizingEngine()
    processor = PartitionProcessor(
        engine=engine,
        graph_cache=graph_cache,
        args=args,
    )

    results = processor.process(items)
    duration = time.perf_counter() - start

    summary = _build_summary(
        manifest=manifest,
        manifest_summary=manifest_summary,
        args=args,
        results=results,
        duration=duration,
    )

    if args.summary:
        _write_json(args.summary, summary)

    return summary


class PartitionProcessor:
    """Handles TNFR self-optimization execution for manifest entries."""

    def __init__(
        self,
        engine: TNFRSelfOptimizingEngine,
        graph_cache: PaleyGraphCache,
        args: argparse.Namespace,
    ) -> None:
        self._engine = engine
        self._graph_cache = graph_cache
        self._args = args

    def process(self, items: Sequence[PartitionWorkItem]) -> List[Dict[str, Any]]:
        max_workers = max(1, int(self._args.max_workers or DEFAULT_MAX_WORKERS))
        results: List[tuple[int, Dict[str, Any]]] = []
        if max_workers == 1:
            for index, item in enumerate(items):
                result = self._process_single(index, item)
                results.append((index, result))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map: Dict[
                    Future[Dict[str, Any]], tuple[int, PartitionWorkItem]
                ] = {}
                for index, item in enumerate(items):
                    future = executor.submit(self._process_single, index, item)
                    future_map[future] = (index, item)
                for future in as_completed(future_map):
                    index, _ = future_map[future]
                    try:
                        result = future.result()
                    except (
                        Exception
                    ) as exc:  # pragma: no cover - logged via _process_single
                        result = {
                            "partition_id": future_map[future][1].partition_id,
                            "error": str(exc),
                            "success": False,
                        }
                    results.append((index, result))
        results.sort(key=lambda pair: pair[0])
        return [result for _, result in results]

    def _process_single(self, index: int, item: PartitionWorkItem) -> Dict[str, Any]:
        try:
            partition_payload = _load_json(item.path)
            partition_data = partition_payload.get("partition") or {}
            if "graph" in partition_payload:
                subgraph = decode_graph(partition_payload["graph"])
            else:
                modulus_value = partition_payload.get("modulus")
                if modulus_value is None:
                    raise ValueError("Partition file requires a graph payload or Paley modulus")
                node_indices = partition_data.get("node_indices") or []
                if not node_indices:
                    raise ValueError("Partition file is missing node indices")
                base_graph = self._graph_cache.get(int(modulus_value))
                if any(node not in base_graph for node in node_indices):
                    raise ValueError("Partition node indices are outside the Paley graph")
                subgraph = base_graph.subgraph(node_indices).copy()
            operator_sequence = _extract_operator_sequence(partition_data)
            seed_value = None if self._args.seed is None else self._args.seed + item.source_index
            dry_run = not bool(self._args.apply)
            capture_snapshots = self._args.capture_snapshots or dry_run
            result = self._run_optimizer(
                subgraph=subgraph,
                partition_id=item.partition_id,
                dry_run=dry_run,
                capture_snapshots=capture_snapshots,
                seed_value=seed_value,
                operator_sequence=operator_sequence,
            )
            success = "error" not in result
            telemetry = _extract_telemetry(item.manifest_entry, partition_payload)
            telemetry_deltas = _compute_telemetry_deltas(
                telemetry,
                result.get("telemetry_snapshots"),
            )
            for field in ("delta_c", "delta_phi_s", "delta_si", "delta_sense_index"):
                if field in telemetry:
                    telemetry[f"manifest_{field}"] = telemetry.pop(field)
            telemetry.update(telemetry_deltas)
            engine_payload = {
                k: v for k, v in result.items() if k not in {"telemetry_snapshots"}
            }
            summary = {
                "partition_id": item.partition_id,
                "path": str(item.path),
                "success": success,
                "seed": seed_value,
                "seed_scope": "recorded_label",
                "engine": _json_safe(engine_payload),
                "telemetry_snapshots": result.get("telemetry_snapshots"),
                "telemetry": telemetry,
                "telemetry_deltas": telemetry_deltas,
                "candidate_factors": partition_payload.get("candidate_factors"),
            }
            if not self._args.quiet:
                status = "ok" if success else "error"
                print(f"[self-opt] partition={item.partition_id} status={status}")
            return summary
        except Exception as exc:
            if not self._args.quiet:
                print(f"[self-opt] partition={item.partition_id} error={exc}")
            return {
                "partition_id": item.partition_id,
                "path": str(item.path),
                "success": False,
                "error": str(exc),
            }

    def _run_optimizer(
        self,
        *,
        subgraph: nx.Graph,
        partition_id: str,
        dry_run: bool,
        capture_snapshots: bool,
        seed_value: Optional[int],
        operator_sequence: Optional[List[str]],
    ) -> Dict[str, Any]:
        return self._engine.optimize_automatically(
            subgraph,
            self._args.operation_type or DEFAULT_OPERATION,
            dry_run=dry_run,
            seed=seed_value,
            partition_id=partition_id,
            operator_sequence=operator_sequence,
            output_dir=self._args.output_dir or DEFAULT_OUTPUT_DIR,
            capture_snapshots=capture_snapshots,
        )


def _collect_partition_entries(
    manifest: Dict[str, Any],
    manifest_path: Path,
    override_partition_dir: Optional[Path],
) -> List[PartitionWorkItem]:
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest JSON is missing 'entries'")
    manifest_dir = manifest_path.resolve().parent
    partition_dir = (
        override_partition_dir.resolve() if override_partition_dir is not None
        else manifest_dir / manifest.get("partition_directory", "")
    )
    resolved_items: List[PartitionWorkItem] = []
    seen_ids = set()
    for source_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("Manifest entries must be objects")
        partition_id = entry.get("partition_id") or entry.get("id")
        if not partition_id:
            raise ValueError("Manifest entry is missing partition_id")
        partition_id = str(partition_id)
        if partition_id in seen_ids:
            raise ValueError(f"Duplicate manifest partition_id: {partition_id}")
        seen_ids.add(partition_id)
        relative_path = entry.get("relative_path") or entry.get("path")
        candidate_paths = _candidate_paths(
            relative_path=relative_path,
            manifest_dir=manifest_dir,
            partition_dir=partition_dir,
            partition_id=partition_id,
        )
        partition_path = _select_existing(candidate_paths)
        if partition_path is None:
            raise FileNotFoundError(
                f"Unable to resolve partition file for {partition_id}; tried: "
                + ", ".join(str(path) for path in candidate_paths)
            )
        resolved_items.append(
            PartitionWorkItem(
                partition_id=partition_id, path=partition_path, manifest_entry=entry,
                source_index=source_index,
            )
        )
    return resolved_items


def _candidate_paths(
    *,
    relative_path: Optional[str],
    manifest_dir: Path,
    partition_dir: Optional[Path],
    partition_id: str,
) -> List[Path]:
    candidates: List[Path] = []
    potential = []
    if relative_path:
        rel_path = Path(relative_path)
        potential.append(rel_path)
    if partition_dir:
        potential.append(partition_dir / f"{partition_dir.name}_{partition_id}.json")
    potential.append(manifest_dir / f"{manifest_dir.name}_{partition_id}.json")
    for path in potential:
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append((manifest_dir / path).resolve())
            candidates.append((REPO_ROOT / path).resolve())
    if relative_path:
        candidates.append((manifest_dir / relative_path).resolve())
    unique: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _select_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.is_file():
            return path
    return None


def _extract_operator_sequence(partition_data: Dict[str, Any]) -> Optional[List[str]]:
    metadata = partition_data.get("metadata") or {}
    nodal_state = metadata.get("nodal_state") or {}
    sequence = nodal_state.get("sequence") or metadata.get("sequence")
    if not sequence:
        return None
    if isinstance(sequence, str):
        return [sequence]
    return list(sequence)


def _extract_telemetry(
    manifest_entry: Dict[str, Any],
    partition_payload: Dict[str, Any],
) -> Dict[str, Any]:
    telemetry: Dict[str, Any] = {}
    manifest_tel = manifest_entry.get("telemetry") or {}
    partition_tel = (partition_payload.get("partition") or {}).get("telemetry") or {}
    telemetry.update(manifest_tel)
    for key, value in partition_tel.items():
        telemetry.setdefault(key, value)
    return telemetry


def _mean_numeric(values: Iterable[float]) -> Optional[float]:
    data = [float(v) for v in values if isinstance(v, (int, float))]
    if not data:
        return None
    try:
        return fmean(data)
    except Exception:
        return float(sum(data) / len(data))


def _snapshot_field(snapshot: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not snapshot:
        return None
    value = snapshot.get(key)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _snapshot_phi_mean(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    if not snapshot:
        return None
    telemetry = snapshot.get("telemetry") or {}
    canonical = telemetry.get("canonical") or {}
    phi_map = canonical.get("phi_s")
    if isinstance(phi_map, dict) and phi_map:
        return _mean_numeric(phi_map.values())
    return None


def _compute_telemetry_deltas(
    telemetry: Dict[str, Any],
    snapshots: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not snapshots:
        return {}
    before = snapshots.get("before") or {}
    after = snapshots.get("after") or {}
    deltas: Dict[str, float] = {}

    snapshot_phi = _snapshot_phi_mean(before)
    snapshot_coherence = _snapshot_field(before, "coherence")
    snapshot_si = _snapshot_field(before, "sense_index")

    # Archived-source drift is not an optimization gain. A dry run has the
    # same before/after snapshot and must report zero actual improvement.
    for field, value in (
        ("phi_s", snapshot_phi), ("coherence", snapshot_coherence),
        ("sense_index", snapshot_si),
    ):
        baseline = telemetry.get(field)
        if (
            value is not None and isinstance(baseline, (int, float))
            and math.isfinite(baseline) and math.isfinite(value)
        ):
            deltas[f"manifest_{field}_drift"] = value - float(baseline)

    phi_after = _snapshot_phi_mean(after)
    if snapshot_phi is not None and phi_after is not None:
        deltas["delta_phi_s"] = float(phi_after) - float(snapshot_phi)
        deltas["delta_phi_s_snapshot"] = deltas["delta_phi_s"]

    coherence_after = _snapshot_field(after, "coherence")
    if snapshot_coherence is not None and coherence_after is not None:
        deltas["delta_c"] = float(coherence_after) - float(snapshot_coherence)
        deltas["delta_c_snapshot"] = deltas["delta_c"]

    sense_after = _snapshot_field(after, "sense_index")
    if snapshot_si is not None and sense_after is not None:
        deltas["delta_si"] = float(sense_after) - float(snapshot_si)
        deltas["delta_sense_index"] = deltas["delta_si"]
        deltas["delta_sense_snapshot"] = deltas["delta_si"]

    return deltas


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _aggregate_telemetry(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    fields = ["phi_s", "phase_gradient", "phase_curvature", "coherence_length"]
    aggregates: Dict[str, float] = {}
    for field in fields:
        values = [
            res.get("telemetry", {}).get(field)
            for res in results
            if res.get("telemetry", {}).get(field) is not None
        ]
        if values:
            aggregates[f"{field}_mean"] = sum(values) / len(values)
            aggregates[f"{field}_min"] = min(values)
            aggregates[f"{field}_max"] = max(values)
    return aggregates


def _build_summary(
    *,
    manifest: Dict[str, Any],
    manifest_summary: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    results: Sequence[Dict[str, Any]],
    duration: float,
) -> Dict[str, Any]:
    success_count = sum(1 for result in results if result.get("success"))
    failure_count = len(results) - success_count
    summary = {
        "manifest": str(args.manifest),
        "manifest_summary": (
            str(args.manifest_summary) if args.manifest_summary else None
        ),
        "operation_type": args.operation_type or DEFAULT_OPERATION,
        "dry_run": not bool(args.apply),
        "apply": bool(args.apply),
        "partitions_requested": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "duration_seconds": duration,
        "partition_results": results,
        "manifest_aggregation": manifest.get("aggregation"),
        "manifest_summary_payload": manifest_summary,
        "telemetry_summary": _aggregate_telemetry(results),
    }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
