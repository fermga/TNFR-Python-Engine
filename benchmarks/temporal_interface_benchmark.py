#!/usr/bin/env python3
"""TNFR Temporal Structural-Interface Benchmark (real phase-native data).

This benchmark exercises the *temporal* extension of the TNFR Structural
Interface Theory on **real measured data**.  It downloads a month of power-grid
frequency measurements (TransnetBW open data, CC-BY), reconstructs the
instantaneous phase with the Hilbert transform, embeds the series into a
delay-coordinate proximity graph, and tracks the TNFR Structural Field Tetrad
(``|∇φ|``, ``K_φ``, ``ξ_C``, ``Φ_s``) window by window.  The tetrad trends are
compared against the canonical early-warning baselines from the
critical-slowing-down (CSD) literature: rolling variance and lag-1
autocorrelation.

Honest scope
------------
This is a *phase-native structural telemetry* demonstration, **not** a
blackout-prediction claim.  Power-grid frequency is dominated by fast
stochastic fluctuation around the nominal set-point; it is not a slow
bifurcation approach, so a flat or baseline-favourable result is a correct,
honest outcome and is reported as such.

A key, deliberately-reported finding from this pipeline: for a *single scalar
series* approaching a fold bifurcation, the classical CSD indicators (variance,
lag-1 autocorrelation) are the appropriate and typically superior tool.  The
TNFR tetrad's differential value lives in *multi-channel, phase-coupled* networks
(e.g. simultaneously-measured oscillators), where ``ξ_C`` is a genuine spatial
coherence length.  The single-series setting here is the *weakest* case for the
tetrad, and we report it honestly rather than selecting a favourable fixture.

The ``synthetic`` source is a **test fixture only** (a series approaching a fold
bifurcation, where CSD is guaranteed).  It validates pipeline mechanics and
calibrates expectations; it is never presented as evidence for the thesis.

Usage (PowerShell)::

    $env:PYTHONPATH=(Resolve-Path -Path ./src).Path
    # Real grid data (downloaded + cached, bounded size):
    python benchmarks/temporal_interface_benchmark.py --source grid \
        --year 2020 --month 1 --output results/reports
    # Offline synthetic sanity fixture:
    python benchmarks/temporal_interface_benchmark.py --source synthetic
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Ensure local src is importable ------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from tnfr.validation.temporal_interface import (  # noqa: E402
    TemporalInterfaceConfig,
    evaluate_early_warning,
    window_tetrad_series,
)

# TransnetBW publishes monthly grid-frequency archives (CC-BY).  The dataset is
# documented on Zenodo (record 15784548); the lighter per-month source is the
# TransnetBW webservice.  Replace year/month in the URL to fetch another month.
TRANSNETBW_FREQUENCY_URL_TEMPLATE = (
    "https://webservices.transnetbw.de/files/bis/netzfrequenz/" "{yyyymm}_Frequenz.zip"
)

DEFAULT_MAX_BYTES = 80_000_000  # bounded download guard (~80 MB)
DEFAULT_MAX_POINTS = 6_000  # subsample target to keep ξ_C tractable
NOMINAL_FREQUENCY_HZ = 50.0


# ---------------------------------------------------------------------------
# Real data acquisition (bounded, cached, graceful-skip)
# ---------------------------------------------------------------------------
def download_grid_frequency_month(
    year: int,
    month: int,
    *,
    cache_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 60.0,
) -> Path | None:
    """Download one month of grid-frequency data with a hard size bound.

    Returns the cached zip path, or ``None`` on any failure (no network, HTTP
    error, oversized payload).  The download is read in chunks and aborted if
    the cumulative size exceeds ``max_bytes`` so a mistaken URL cannot pull a
    multi-gigabyte archive.
    """
    yyyymm = f"{year:04d}{month:02d}"
    url = TRANSNETBW_FREQUENCY_URL_TEMPLATE.format(yyyymm=yyyymm)
    path = cache_path or _ROOT / "results" / "data" / f"{yyyymm}_Frequenz.zip"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = Request(url, headers={"User-Agent": "tnfr-benchmark/1.0"})
        buffer = io.BytesIO()
        total = 0
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    print(
                        f"  [skip] download exceeded {max_bytes} bytes; aborting",
                        file=sys.stderr,
                    )
                    return None
                buffer.write(chunk)
    except Exception as exc:  # noqa: BLE001 - graceful offline skip
        print(f"  [skip] grid download failed: {exc}", file=sys.stderr)
        return None
    path.write_bytes(buffer.getvalue())
    return path


def _parse_frequency_csv(text: str) -> np.ndarray:
    """Parse a TransnetBW frequency CSV into a float array (NaN for gaps).

    Robust to ``;``/``,``/tab delimiters and to decimal commas.  The frequency
    is taken as the last numeric field of each row; unparseable rows (headers,
    blanks, NaN markers) are dropped.
    """
    values: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ";" in line:
            delimiter = ";"
        elif "\t" in line:
            delimiter = "\t"
        else:
            delimiter = ","
        fields = [f.strip() for f in line.split(delimiter) if f.strip()]
        if len(fields) < 2:
            continue
        token = fields[-1]
        # German decimal comma reconstruction for comma-delimited rows that
        # split a value like "50,012" into ["50", "012"].
        if (
            delimiter == ","
            and len(fields) >= 3
            and fields[-2].isdigit()
            and token.isdigit()
        ):
            token = f"{fields[-2]}.{token}"
        else:
            token = token.replace(",", ".")
        try:
            freq = float(token)
        except ValueError:
            continue
        # Plausibility band: discard obvious non-frequency tokens.
        if not math.isfinite(freq) or freq < 40.0 or freq > 60.0:
            values.append(float("nan"))
            continue
        values.append(freq)
    return np.asarray(values, dtype=float)


def load_grid_frequency_series(
    zip_path: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> np.ndarray | None:
    """Load and subsample a grid-frequency series from a downloaded zip.

    Returns a 1-D float array with NaN gaps removed, subsampled by a uniform
    stride to at most ``max_points`` samples, or ``None`` if parsing yields too
    little usable data.
    """
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                names = archive.namelist()
            if not names:
                return None
            raw = archive.read(names[0])
    except Exception as exc:  # noqa: BLE001 - corrupt/partial archive
        print(f"  [skip] could not read zip: {exc}", file=sys.stderr)
        return None

    text = raw.decode("utf-8", errors="ignore")
    series = _parse_frequency_csv(text)
    series = series[np.isfinite(series)]
    if series.size < 512:
        print(f"  [skip] parsed only {series.size} usable samples", file=sys.stderr)
        return None
    if series.size > max_points:
        stride = int(math.ceil(series.size / max_points))
        series = series[::stride]
    return series


# ---------------------------------------------------------------------------
# Synthetic test fixture (mechanics only; never presented as evidence)
# ---------------------------------------------------------------------------
def synthetic_fold_transition(
    n: int = 2400,
    *,
    transition_at: int = 1800,
    noise: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic series approaching a fold bifurcation (CSD guaranteed).

    This is a **test fixture only**: the autoregressive recovery rate decays to
    zero as ``t`` approaches ``transition_at``, so rolling variance and lag-1
    autocorrelation rise by construction.  It validates pipeline mechanics and
    is never used as evidence for the TNFR thesis.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    for t in range(1, n):
        if t < transition_at:
            ar = min(0.98, 0.2 + 0.78 * (t / transition_at))
        else:
            ar = 0.99
        oscillation = 0.02 * math.sin(2.0 * math.pi * t / 12.0)
        x[t] = ar * x[t - 1] + oscillation + noise * rng.standard_normal()
    return x


def detect_excursion_event(series: np.ndarray, *, margin: float = 0.1) -> int | None:
    """Locate the largest frequency excursion as an exploratory event index.

    Restricts the search to the interior ``[margin, 1 - margin]`` of the record
    so that there is room for pre-event windows.  Returns ``None`` if the series
    is too short.
    """
    n = series.size
    if n < 64:
        return None
    lo = int(n * margin)
    hi = int(n * (1.0 - margin))
    if hi - lo < 16:
        return None
    deviation = np.abs(series[lo:hi] - NOMINAL_FREQUENCY_HZ)
    return int(lo + int(np.argmax(deviation)))


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
def run_temporal_benchmark(
    *,
    source: str,
    year: int,
    month: int,
    config: TemporalInterfaceConfig,
    max_points: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Run the temporal structural-interface benchmark on the chosen source."""
    report: dict[str, Any] = {
        "source": source,
        "config": asdict(config),
        "honest_scope": (
            "Phase-native structural telemetry on real measured phase. "
            "Single-series CSD favours classical indicators; the TNFR tetrad's "
            "differential value is in multi-channel phase-coupled networks."
        ),
    }

    if source == "grid":
        zip_path = download_grid_frequency_month(year, month, max_bytes=max_bytes)
        if zip_path is None:
            report["status"] = "skipped"
            report["reason"] = (
                "Grid-frequency source unreachable in this environment. "
                "Re-run with network access to fetch "
                f"{year:04d}-{month:02d}; the pipeline mechanics are validated "
                "offline via --source synthetic."
            )
            return report
        series = load_grid_frequency_series(zip_path, max_points=max_points)
        if series is None:
            report["status"] = "skipped"
            report["reason"] = "Downloaded archive could not be parsed."
            return report
        report["data"] = {
            "samples": int(series.size),
            "mean_hz": float(np.mean(series)),
            "std_hz": float(np.std(series)),
            "min_hz": float(np.min(series)),
            "max_hz": float(np.max(series)),
            "cache": str(zip_path.relative_to(_ROOT)),
        }
        transition_index = detect_excursion_event(series)
        report["event_kind"] = "largest-frequency-excursion (exploratory)"
    elif source == "synthetic":
        series = synthetic_fold_transition()
        transition_index = 1800
        report["data"] = {
            "samples": int(series.size),
            "note": "Synthetic fold fixture; mechanics only, not evidence.",
        }
        report["event_kind"] = "synthetic fold bifurcation (fixture)"
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown source: {source}")

    if series.size < config.window:
        report["status"] = "skipped"
        report["reason"] = (
            f"Series ({series.size}) shorter than window ({config.window})."
        )
        return report

    comparison = evaluate_early_warning(
        series, transition_index=transition_index, config=config
    )
    series_obj = window_tetrad_series(series, config=config)

    report["status"] = "ok"
    report["transition_index"] = transition_index
    report["n_windows"] = int(series_obj.window_end.size)
    report["n_pre_transition_windows"] = comparison.n_pre_transition_windows
    report["trends"] = {
        k: (None if math.isnan(v) else round(float(v), 4))
        for k, v in comparison.trends.items()
    }
    report["best_tnfr_channel"] = comparison.best_tnfr[0]
    report["best_tnfr_tau"] = (
        None
        if math.isnan(comparison.best_tnfr[1])
        else round(float(comparison.best_tnfr[1]), 4)
    )
    report["best_baseline_channel"] = comparison.best_baseline[0]
    report["best_baseline_tau"] = (
        None
        if math.isnan(comparison.best_baseline[1])
        else round(float(comparison.best_baseline[1]), 4)
    )
    report["interpretation"] = comparison.interpretation
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TNFR temporal structural-interface benchmark."
    )
    parser.add_argument(
        "--source",
        choices=("grid", "synthetic"),
        default="grid",
        help="Data source: real grid frequency or synthetic fixture.",
    )
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=3)
    parser.add_argument("--embedding-tau", type=int, default=2)
    parser.add_argument("--k-neighbours", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "reports",
        help="Directory for the JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    config = TemporalInterfaceConfig(
        embedding_dim=args.embedding_dim,
        embedding_tau=args.embedding_tau,
        k_neighbours=args.k_neighbours,
        window=args.window,
        step=args.step,
    )
    report = run_temporal_benchmark(
        source=args.source,
        year=args.year,
        month=args.month,
        config=config,
        max_points=args.max_points,
        max_bytes=args.max_bytes,
    )

    print(json.dumps(report, indent=2))

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{args.year:04d}{args.month:02d}" if args.source == "grid" else "synthetic"
    )
    out_path = output_dir / f"temporal_interface_{args.source}_{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        display = out_path.relative_to(_ROOT)
    except ValueError:
        display = out_path
    print(f"\nReport written to {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
