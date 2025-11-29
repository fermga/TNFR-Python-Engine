#!/usr/bin/env python3
"""
TNFR New Zero Search Engine
===========================

Systematically scans unexplored height ranges on the Riemann critical line
using the refined discriminant
    F(s) = ΔNFR(s) + λ |ζ(s)|²
with λ ≈ 0.05462277.  The scan identifies candidate locations where F(s)
falls below a configurable threshold, refines the location, validates via
|ζ(s)|, and records telemetry for downstream confirmation.

Usage (default scan 250 ≤ t ≤ 500):
    python search_new_zeros.py

Usage (custom range):
    python search_new_zeros.py --t-min 500 --t-max 800 --step 0.25

Output artifacts:
- Console summary of detected candidates
- JSON report with detailed telemetry (new_zero_candidates.json)
- Optional CSV export for integration with further tooling
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from tnfr.mathematics.zeta import zeta_function, mp

from refined_zero_discriminant import TNFRRefinedZeroDiscriminant


@dataclass
class ZeroSearchConfig:
    """Configuration for systematic zero search."""

    t_min: float = 250.0
    t_max: float = 500.0
    step: float = 0.25
    refinement_window: float = 1.0
    refinement_iterations: int = 6
    discriminant_threshold: float = 1e-6
    zeta_threshold: float = 1e-8
    lambda_coeff: float = 0.05462277217684343
    results_path: Path = Path("new_zero_candidates.json")
    csv_path: Optional[Path] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None


@dataclass
class ZeroCandidate:
    """Detected candidate for a new non-trivial zero."""

    height_estimate: float
    discriminant_value: float
    zeta_magnitude: float
    delta_nfr: float
    confidence: float
    refinement_history: List[float]
    validation_success: bool
    validation_details: Dict[str, Any]


@dataclass
class ScanTelemetry:
    """Telemetry describing a completed scan."""

    total_samples: int
    threshold_hits: int
    refinement_calls: int
    evaluation_time_s: float
    min_discriminant: Optional[float]
    max_discriminant: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "threshold_hits": self.threshold_hits,
            "refinement_calls": self.refinement_calls,
            "evaluation_time_s": self.evaluation_time_s,
            "min_discriminant": self.min_discriminant,
            "max_discriminant": self.max_discriminant,
        }


class TNFRZeroSearcher:
    """Searches for new RH zeros using the refined TNFR discriminant."""

    def __init__(self, config: ZeroSearchConfig):
        self.config = config
        self.discriminant = TNFRRefinedZeroDiscriminant(config.lambda_coeff)

        if hasattr(mp, "dps"):
            mp.dps = 50

    def evaluate_discriminant(self, t_value: float) -> Dict[str, Any]:
        """Evaluate F(s) on the critical line at height t_value."""
        s = complex(0.5, t_value)
        result = self.discriminant.compute_refined_discriminant(s)
        return {
            "t": t_value,
            "discriminant": float(result.discriminant_value),
            "zeta_magnitude": float(result.zeta_magnitude),
            "delta_nfr": float(result.delta_nfr),
            "confidence": float(result.confidence),
        }

    def refine_candidate(self, t_seed: float) -> Dict[str, Any]:
        """Refine candidate location by minimizing F(s) locally."""
        window = self.config.refinement_window
        left = t_seed - window / 2
        right = t_seed + window / 2
        history: List[float] = []
        best_eval = self.evaluate_discriminant(t_seed)
        best_point = t_seed

        for _ in range(self.config.refinement_iterations):
            segment = [
                left,
                left + (right - left) / 3,
                left + 2 * (right - left) / 3,
                right,
            ]
            evaluations = [
                (pt, self.evaluate_discriminant(pt))
                for pt in segment
            ]
            evaluations.sort(key=lambda item: item[1]["discriminant"])
            best_point, best_eval = evaluations[0]
            history.append(best_point)

            best_index = segment.index(best_point)
            if best_index == 0:
                left, right = segment[0], segment[1]
            elif best_index == len(segment) - 1:
                left, right = segment[-2], segment[-1]
            else:
                left, right = segment[best_index - 1], segment[best_index + 1]

        best_eval["t"] = best_point
        best_eval["history"] = history
        return best_eval

    def validate_candidate(self, t_value: float) -> Dict[str, Any]:
        """Validate candidate by computing |ζ(0.5 + it)| with mp precision."""
        s = mp.mpf("0.5") + mp.mpf(t_value) * mp.j
        zeta_val = zeta_function(s)
        magnitude = abs(zeta_val)
        validation = {
            "zeta_value": complex(zeta_val),
            "zeta_magnitude": float(magnitude),
            "passes_threshold": magnitude < self.config.zeta_threshold,
        }
        return validation

    def scan_range(self) -> Tuple[List[ZeroCandidate], ScanTelemetry]:
        """Scan configured range and return candidate zeros with telemetry."""
        candidates: List[ZeroCandidate] = []
        t_values = np.arange(self.config.t_min, self.config.t_max + self.config.step,
                             self.config.step)
        threshold_hits = 0
        refinement_calls = 0
        min_disc = float("inf")
        max_disc = float("-inf")
        start_time = time.perf_counter()

        for t in t_values:
            evaluation = self.evaluate_discriminant(float(t))
            disc_value = evaluation["discriminant"]
            min_disc = min(min_disc, disc_value)
            max_disc = max(max_disc, disc_value)
            if disc_value <= self.config.discriminant_threshold:
                threshold_hits += 1
                refined = self.refine_candidate(evaluation["t"])
                refinement_calls += 1
                validation = self.validate_candidate(refined["t"])
                candidate = ZeroCandidate(
                    height_estimate=refined["t"],
                    discriminant_value=refined["discriminant"],
                    zeta_magnitude=refined["zeta_magnitude"],
                    delta_nfr=refined["delta_nfr"],
                    confidence=refined["confidence"],
                    refinement_history=refined["history"],
                    validation_success=validation["passes_threshold"],
                    validation_details=validation,
                )
                candidates.append(candidate)

        elapsed = time.perf_counter() - start_time
        telemetry = ScanTelemetry(
            total_samples=len(t_values),
            threshold_hits=threshold_hits,
            refinement_calls=refinement_calls,
            evaluation_time_s=elapsed,
            min_discriminant=min_disc if min_disc != float("inf") else None,
            max_discriminant=max_disc if max_disc != float("-inf") else None,
        )
        return candidates, telemetry

    def summarize_candidates(self, candidates: List[ZeroCandidate]) -> Dict[str, Any]:
        """Generate descriptive statistics about detected candidates."""
        if not candidates:
            return {
                "count": 0,
                "message": "No candidates detected in the scanned range.",
            }

        heights = [c.height_estimate for c in candidates]
        discriminants = [c.discriminant_value for c in candidates]

        summary = {
            "count": len(candidates),
            "height_min": min(heights),
            "height_max": max(heights),
            "median_height": float(statistics.median(heights)),
            "mean_discriminant": float(statistics.mean(discriminants)),
            "min_discriminant": min(discriminants),
            "validated": sum(1 for c in candidates if c.validation_success),
        }
        return summary

    def save_results(self, candidates: List[ZeroCandidate], summary: Dict[str, Any],
                     telemetry: ScanTelemetry) -> None:
        """Persist results to JSON file."""

        def to_json_ready(value: Any) -> Any:
            """Convert payload values into JSON-serializable structures."""
            if isinstance(value, dict):
                return {k: to_json_ready(v) for k, v in value.items()}
            if isinstance(value, list):
                return [to_json_ready(v) for v in value]
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, complex):
                return {"real": value.real, "imag": value.imag}
            return value

        payload = {
            "config": asdict(self.config),
            "summary": summary,
            "telemetry": telemetry.to_dict(),
            "candidates": [asdict(c) for c in candidates],
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        payload = to_json_ready(payload)
        with self.config.results_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        print(f"💾 Results saved to {self.config.results_path}")

    def save_csv(self, candidates: List[ZeroCandidate]) -> None:
        """Optionally persist results to CSV for downstream tooling."""
        if not self.config.csv_path:
            return

        field_names = [
            "height_estimate",
            "discriminant_value",
            "zeta_magnitude",
            "delta_nfr",
            "confidence",
            "validation_success",
        ]

        with self.config.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_names)
            writer.writeheader()
            for candidate in candidates:
                writer.writerow({
                    "height_estimate": candidate.height_estimate,
                    "discriminant_value": candidate.discriminant_value,
                    "zeta_magnitude": candidate.zeta_magnitude,
                    "delta_nfr": candidate.delta_nfr,
                    "confidence": candidate.confidence,
                    "validation_success": candidate.validation_success,
                })

        print(f"📄 CSV export saved to {self.config.csv_path}")


def parse_args() -> ZeroSearchConfig:
    """Parse CLI arguments into ZeroSearchConfig."""
    parser = argparse.ArgumentParser(description="TNFR new zero search")
    parser.add_argument("--t-min", type=float, default=250.0,
                        help="Lower bound for imaginary part scan")
    parser.add_argument("--t-max", type=float, default=500.0,
                        help="Upper bound for imaginary part scan")
    parser.add_argument("--step", type=float, default=0.25,
                        help="Sampling step size")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Discriminant threshold for candidate detection")
    parser.add_argument("--output", type=str, default="new_zero_candidates.json",
                        help="Output JSON path")
    parser.add_argument("--csv-output", type=str, default=None,
                        help="Optional CSV export path")
    parser.add_argument("--refinement-window", type=float, default=1.0,
                        help="Window size used during candidate refinement")
    parser.add_argument("--refinement-iterations", type=int, default=6,
                        help="Number of refinement iterations")
    parser.add_argument("--zeta-threshold", type=float, default=1e-8,
                        help="Validation threshold for |ζ(0.5+it)|")
    parser.add_argument("--lambda-coeff", type=float, default=0.05462277217684343,
                        help="λ coefficient used by the discriminant")
    parser.add_argument("--chunk-count", type=int, default=None,
                        help="Split the scan range into N chunks")
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="0-based index of the chunk to process")

    args = parser.parse_args()

    chunk_t_min = args.t_min
    chunk_t_max = args.t_max
    chunk_suffix = None
    if args.chunk_count is not None:
        if args.chunk_count <= 0:
            parser.error("--chunk-count must be positive")
        if args.chunk_index is None:
            parser.error("--chunk-index is required when using --chunk-count")
        if not (0 <= args.chunk_index < args.chunk_count):
            parser.error("--chunk-index must satisfy 0 ≤ index < chunk_count")
        span = args.t_max - args.t_min
        if span <= 0:
            parser.error("t_max must be greater than t_min")
        chunk_size = span / args.chunk_count
        chunk_t_min = args.t_min + chunk_size * args.chunk_index
        chunk_t_max = args.t_min + chunk_size * (args.chunk_index + 1)
        if args.chunk_index == args.chunk_count - 1:
            chunk_t_max = args.t_max
        chunk_suffix = f"_chunk-{args.chunk_index + 1}-of-{args.chunk_count}"
    elif args.chunk_index is not None:
        parser.error("--chunk-count is required when specifying --chunk-index")

    def with_suffix(path: Path, suffix: str) -> Path:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")

    results_path = Path(args.output)
    if chunk_suffix and args.output == "new_zero_candidates.json":
        results_path = with_suffix(results_path, chunk_suffix)

    csv_path = Path(args.csv_output) if args.csv_output else None
    if chunk_suffix:
        if csv_path:
            csv_path = with_suffix(csv_path, chunk_suffix)
        else:
            csv_path = results_path.with_suffix(".csv")

    return ZeroSearchConfig(
        t_min=chunk_t_min,
        t_max=chunk_t_max,
        step=args.step,
        discriminant_threshold=args.threshold,
        refinement_window=args.refinement_window,
        refinement_iterations=args.refinement_iterations,
        zeta_threshold=args.zeta_threshold,
        lambda_coeff=args.lambda_coeff,
        results_path=results_path,
        csv_path=csv_path,
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
    )


def main() -> None:
    config = parse_args()
    searcher = TNFRZeroSearcher(config)

    print(f"🔍 Scanning {config.t_min} ≤ t ≤ {config.t_max} with step {config.step}")
    if config.chunk_count:
        assert config.chunk_index is not None
        print(f"🧩 Chunk {config.chunk_index + 1}/{config.chunk_count}")
    candidates, telemetry = searcher.scan_range()
    summary = searcher.summarize_candidates(candidates)

    print("\n📊 Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n📈 Telemetry:")
    for key, value in telemetry.to_dict().items():
        print(f"  {key}: {value}")

    searcher.save_results(candidates, summary, telemetry)
    searcher.save_csv(candidates)


if __name__ == "__main__":
    main()
