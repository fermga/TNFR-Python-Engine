"""Full-spectrum TNFR factorization benchmark with certificate capture."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

# Ensure both the factorization lab and core TNFR sources are importable when
# the benchmark is executed from the repository root.
BENCHMARK_PATH = Path(__file__).resolve()
LAB_ROOT = BENCHMARK_PATH.parents[1]
REPO_ROOT = LAB_ROOT.parent
SRC_DIR = REPO_ROOT / "src"

for candidate in (LAB_ROOT, SRC_DIR):
    candidate_str = candidate.as_posix()
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from tnfr_factorization import SpectralPaleyFactorizer  # noqa: E402

RESULTS_DIR = LAB_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FULL_SPECTRUM_TARGETS: Mapping[str, Sequence[int]] = {
    "semiprimes": (185, 221, 899),
    "triprimes": (385, 1001),
    "powers": (343, 625, 1331),
    "highly_composite": (360, 840, 1260),
    "structured_range": tuple(range(150, 171)),
}


@dataclass
class PartitionTrace:
    partition_id: str
    node_count: int
    boundary_count: int
    phi_s: float | None
    phase_gradient: float | None
    phase_curvature: float | None
    coherence_length: float | None
    dnfr_after: float | None
    coherence_ratio: float | None
    inferred_factor: int | None
    candidate_factors: List[int]


@dataclass
class FullSpectrumRecord:
    n: int
    category: str
    modulus: int
    phi_s: float
    phase_gradient: float
    phase_curvature: float
    coherence_length: float
    candidate_factors: List[int]
    dynamic_factors: List[int]
    tnfr_certified_factors: List[int]
    certificate_path: str | None
    strategy_plan: Dict[str, object] | None
    nodal_decoding: Dict[str, object] | None
    tnfr_verification: Dict[str, object] | None
    partition_traces: List[PartitionTrace]
    runtime_ms: float


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


def _relative_path(path: str | None) -> str | None:
    if not path:
        return None
    target = Path(path).resolve()
    try:
        return target.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return target.as_posix()


def _load_certificate_payload(certificate_path: str | None) -> MutableMapping[str, object]:
    if not certificate_path:
        return {}
    candidate = Path(certificate_path)
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text())
    except json.JSONDecodeError:
        return {}


def _load_partition_traces(certificate_path: str | None) -> List[PartitionTrace]:
    payload = _load_certificate_payload(certificate_path)
    states = payload.get("partition_states") or {}
    traces: List[PartitionTrace] = []
    if not isinstance(states, Mapping):
        return traces

    for partition_id, block in states.items():
        if not isinstance(block, Mapping):
            continue
        before = block.get("before") or {}
        after = block.get("after") or {}
        candidate_factors = block.get("candidate_factors") or []
        traces.append(
            PartitionTrace(
                partition_id=str(partition_id),
                node_count=_safe_int(block.get("node_count")) or 0,
                boundary_count=_safe_int(block.get("boundary_count")) or 0,
                phi_s=_safe_float(before.get("phi_s")),
                phase_gradient=_safe_float(before.get("phase_gradient")),
                phase_curvature=_safe_float(before.get("phase_curvature")),
                coherence_length=_safe_float(before.get("coherence_length")),
                dnfr_after=_safe_float(after.get("dnfr_after")),
                coherence_ratio=_safe_float(after.get("coherence_ratio")),
                inferred_factor=_safe_int(after.get("inferred_factor")),
                candidate_factors=list(candidate_factors) if isinstance(candidate_factors, Iterable) else [],
            )
        )
    return traces


def _build_category_summary(records: Sequence[FullSpectrumRecord]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        bucket = buckets.setdefault(
            record.category,
            {"phi_s": [], "phase_gradient": [], "phase_curvature": [], "coherence_length": []},
        )
        bucket["phi_s"].append(record.phi_s)
        bucket["phase_gradient"].append(record.phase_gradient)
        bucket["phase_curvature"].append(record.phase_curvature)
        bucket["coherence_length"].append(record.coherence_length)

    for category, bucket in buckets.items():
        if not bucket["phi_s"]:
            continue
        summary[category] = {
            "avg_phi_s": sum(bucket["phi_s"]) / len(bucket["phi_s"]),
            "avg_phase_gradient": sum(bucket["phase_gradient"]) / len(bucket["phase_gradient"]),
            "avg_phase_curvature": sum(bucket["phase_curvature"]) / len(bucket["phase_curvature"]),
            "avg_coherence_length": sum(bucket["coherence_length"]) / len(bucket["coherence_length"]),
            "sample_count": float(len(bucket["phi_s"])),
        }
    return summary


def run_full_spectrum(
    targets: Mapping[str, Sequence[int]],
    *,
    max_nodes: int | None,
) -> List[FullSpectrumRecord]:
    factorizer = SpectralPaleyFactorizer(max_nodes=max_nodes)
    records: List[FullSpectrumRecord] = []

    for category, numbers in targets.items():
        for n in numbers:
            start = time.perf_counter()
            result = factorizer.analyze(n, trace_certificates=True)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            partition_traces = _load_partition_traces(result.certificate_path)
            record = FullSpectrumRecord(
                n=n,
                category=category,
                modulus=result.modulus,
                phi_s=result.phi_s,
                phase_gradient=result.phase_gradient,
                phase_curvature=result.phase_curvature,
                coherence_length=result.coherence_length,
                candidate_factors=list(result.candidate_factors),
                dynamic_factors=list((result.nodal_decoding or {}).get("dynamic_factors", [])),
                tnfr_certified_factors=list(result.tnfr_certified_factors or []),
                certificate_path=_relative_path(result.certificate_path),
                strategy_plan=result.operator_strategy_plan or {},
                nodal_decoding=result.nodal_decoding or {},
                tnfr_verification=result.tnfr_verification or {},
                partition_traces=partition_traces,
                runtime_ms=runtime_ms,
            )
            records.append(record)
    return records


def save_results(
    records: Sequence[FullSpectrumRecord],
    *,
    filename: str,
    targets: Mapping[str, Sequence[int]],
) -> Path:
    payload = {
        "timestamp": time.time(),
        "categories": {name: list(values) for name, values in targets.items()},
        "records": [asdict(record) for record in records],
        "category_summary": _build_category_summary(records),
    }
    output_path = RESULTS_DIR / filename
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TNFR full-spectrum factorization benchmark")
    parser.add_argument(
        "--output",
        default="full_spectrum_factorization.json",
        help="Filename (under factorization-lab/results/benchmarks) for benchmark dump",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Optional override for SpectralPaleyFactorizer max_nodes",
    )
    parser.add_argument(
        "--numbers",
        type=int,
        nargs="+",
        help="Additional custom n values to analyze",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        help="Inclusive start for an additional contiguous range of integers",
    )
    parser.add_argument(
        "--range-stop",
        type=int,
        help="Inclusive stop for an additional contiguous range of integers",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI wrapper
    args = parse_args(argv)
    targets: Dict[str, Sequence[int]] = {name: tuple(values) for name, values in FULL_SPECTRUM_TARGETS.items()}
    if args.numbers:
        targets["custom"] = tuple(args.numbers)
    if args.range_start is not None and args.range_stop is not None:
        start, stop = sorted((args.range_start, args.range_stop))
        if start <= stop:
            targets["contiguous_range"] = tuple(range(start, stop + 1))

    records = run_full_spectrum(targets, max_nodes=args.max_nodes)
    output_path = save_results(records, filename=args.output, targets=targets)
    print(
        "Saved TNFR full-spectrum factorization benchmark to",
        output_path,
        "covering",
        len(records),
        "analyses",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
