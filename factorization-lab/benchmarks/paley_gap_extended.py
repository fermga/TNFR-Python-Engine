"""Benchmark extended Paley spectral factorization with larger moduli."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from tnfr_factorization import SpectralPaleyFactorizer

EXTENDED_TARGETS = (493, 697, 899, 1189, 1387)
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ExtendedBenchmarkRecord:
    n: int
    modulus: int
    laplacian_gap: float
    coherence_score: float
    phi_s: float
    phase_gradient: float
    phase_curvature: float
    coherence_length: float
    arithmetic_delta_nfr: float
    candidate_factors: List[int]
    true_factors: List[int]
    matched_factors: List[int]
    match_ratio: float
    runtime_ms: float


def _prime_factors(n: int) -> Dict[int, int]:
    factors: Dict[int, int] = {}
    m = n
    divisor = 2
    while divisor * divisor <= m:
        while m % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            m //= divisor
        divisor += 1 if divisor == 2 else 2
    if m > 1:
        factors[m] = factors.get(m, 0) + 1
    return factors


def _flatten_factors(factors: Dict[int, int]) -> List[int]:
    expanded: List[int] = []
    for prime, exponent in factors.items():
        expanded.extend([prime] * exponent)
    return expanded or [1]


def run_benchmark(targets: Sequence[int]) -> List[ExtendedBenchmarkRecord]:
    factorizer = SpectralPaleyFactorizer(max_nodes=4097)
    records: List[ExtendedBenchmarkRecord] = []

    for n in targets:
        start = time.perf_counter()
        result = factorizer.analyze(n)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        true_factor_dict = _prime_factors(n)
        true_factors = _flatten_factors(true_factor_dict)
        matched = sorted(set(true_factors) & set(result.candidate_factors))
        match_ratio = len(matched) / len(set(true_factors)) if true_factors else 0.0

        records.append(
            ExtendedBenchmarkRecord(
                n=n,
                modulus=result.modulus,
                laplacian_gap=result.laplacian_gap,
                coherence_score=result.coherence_score,
                phi_s=result.phi_s,
                phase_gradient=result.phase_gradient,
                phase_curvature=result.phase_curvature,
                coherence_length=result.coherence_length,
                arithmetic_delta_nfr=result.arithmetic_delta_nfr,
                candidate_factors=result.candidate_factors,
                true_factors=true_factors,
                matched_factors=matched,
                match_ratio=match_ratio,
                runtime_ms=runtime_ms,
            )
        )

    return records


def save_results(records: Sequence[ExtendedBenchmarkRecord], *, filename: str = "paley_gap_extended.json") -> Path:
    payload = {
        "timestamp": time.time(),
        "targets": [record.n for record in records],
        "records": [asdict(record) for record in records],
    }
    output_path = RESULTS_DIR / filename
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def main() -> None:  # pragma: no cover
    records = run_benchmark(EXTENDED_TARGETS)
    output = save_results(records)
    print(f"Saved extended benchmark snapshot to {output}")


if __name__ == "__main__":  # pragma: no cover
    main()
