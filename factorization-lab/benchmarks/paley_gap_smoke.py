"""Benchmark script that records Paley spectral gap telemetry for reference n values."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from tnfr_factorization import SpectralPaleyFactorizer

DEFAULT_TARGETS = (185, 221, 289, 299, 323)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkRecord:
    n: int
    modulus: int
    laplacian_gap: float
    phase_gradient: float
    coherence_score: float
    candidate_factors: List[int]
    arithmetic_delta_nfr: float
    runtime_ms: float


def run_benchmark(targets: Sequence[int]) -> List[BenchmarkRecord]:
    factorizer = SpectralPaleyFactorizer()
    records: List[BenchmarkRecord] = []

    for n in targets:
        start = time.perf_counter()
        result = factorizer.analyze(n)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        records.append(
            BenchmarkRecord(
                n=n,
                modulus=result.modulus,
                laplacian_gap=result.laplacian_gap,
                phase_gradient=result.phase_gradient,
                coherence_score=result.coherence_score,
                candidate_factors=result.candidate_factors,
                arithmetic_delta_nfr=result.arithmetic_delta_nfr,
                runtime_ms=runtime_ms,
            )
        )

    return records


def save_results(records: Sequence[BenchmarkRecord], *, filename: str = "paley_gap_smoke.json") -> Path:
    payload = {
        "timestamp": time.time(),
        "targets": [record.n for record in records],
        "records": [asdict(record) for record in records],
    }
    output_path = RESULTS_DIR / filename
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def main(args: Iterable[str] | None = None) -> None:  # pragma: no cover
    del args  # unused for now
    records = run_benchmark(DEFAULT_TARGETS)
    output_path = save_results(records)
    print(f"Saved benchmark snapshot to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
