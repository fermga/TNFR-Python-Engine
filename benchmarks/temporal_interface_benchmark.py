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
import csv
import hashlib
import io
import json
import math
import re
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
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
)

# TransnetBW publishes monthly grid-frequency archives (CC-BY).  The dataset is
# documented on Zenodo (record 15784548); the lighter per-month source is the
# TransnetBW webservice.  Replace year/month in the URL to fetch another month.
TRANSNETBW_FREQUENCY_URL_TEMPLATE = (
    "https://webservices.transnetbw.de/files/bis/netzfrequenz/" "{yyyymm}_Frequenz.zip"
)

DEFAULT_MAX_BYTES = 80_000_000  # bounded download guard (~80 MB)
DEFAULT_MAX_POINTS = 6_000  # subsample target to keep ξ_C tractable
DEFAULT_MAX_EXPANDED_BYTES = 128_000_000
DEFAULT_MAX_MEMBER_BYTES = 64_000_000
DEFAULT_MAX_MEMBERS = 64
DEFAULT_MAX_ROWS = 1_000_000
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
    _positive_limit(max_bytes, "max_bytes")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    yyyymm = f"{year:04d}{month:02d}"
    url = TRANSNETBW_FREQUENCY_URL_TEMPLATE.format(yyyymm=yyyymm)
    path = cache_path or _ROOT / "results" / "data" / f"{yyyymm}_Frequenz.zip"
    if path.exists():
        if not path.is_file() or path.stat().st_size > max_bytes:
            print(
                "  [skip] cached archive exceeds the compressed-byte limit",
                file=sys.stderr,
            )
            return None
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


def _positive_limit(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class GridFrequencyRecord:
    """Bounded raw rows plus an analysis selection; no gap is compressed away.

    Timestamps, values and masks retain *every* parsed data row. The stride
    changes only ``selected_indices``. Naive ISO timestamps have no verified
    timezone; relative seconds are descriptive and never admit Hz_str.
    ``None`` values/times explicitly preserve unavailable observations.
    """

    timestamps: tuple[str, ...]
    elapsed_seconds: tuple[float | None, ...]
    values_hz: tuple[float | None, ...]
    missing: tuple[bool, ...]
    source_line_numbers: tuple[int, ...]
    selected_indices: tuple[int, ...]
    stride: int
    time_status: str
    archive_sha256: str
    member_sha256: str
    member_name: str


def _parse_frequency_rows(text: str, *, max_rows: int):
    _positive_limit(max_rows, "max_rows")
    timestamps, values, lines = [], [], []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        delimiter = ";" if ";" in line else ("\t" if "\t" in line else ",")
        fields = [f.strip() for f in next(csv.reader([line], delimiter=delimiter))]
        if len(fields) < 2:
            raise ValueError(f"row {line_number} has no timestamp/value separation")
        if not timestamps and fields[0].lower().lstrip("\ufeff") in (
            "timestamp",
            "time",
            "datetime",
            "zeitstempel",
            "datum",
        ):
            continue
        if len(values) >= max_rows:
            raise ValueError("CSV row limit exceeded")
        token = fields[-1].replace(",", ".")
        if (
            delimiter == ","
            and len(fields) == 3
            and fields[-2].isdigit()
            and fields[-1].isdigit()
        ):
            token = f"{fields[-2]}.{fields[-1]}"
        elif len(fields) != 2:
            raise ValueError(
                "ambiguous CSV columns; supply one timestamp and one frequency"
            )
        try:
            value = float(token)
        except ValueError:
            value = float("nan")
        values.append(value if math.isfinite(value) and 40 <= value <= 60 else None)
        timestamps.append(fields[0])
        lines.append(line_number)
    return tuple(timestamps), tuple(values), tuple(lines)


def _relative_times(timestamps: tuple[str, ...]):
    """Read relative ISO times with consistent microsecond precision.

    Python 3.10 requires ``+00:00`` instead of ``Z`` for UTC-aware timestamps,
    and three or six fractional second digits. Normalize the terminal ``Z`` and
    represent second fractions at datetime's six-digit precision (padding or
    truncating, never rounding).
    Stored timestamps, missing rows and ordering policies remain unchanged.
    A date alone followed by ``Z`` is not a UTC timestamp.
    """
    parsed = []
    for value in timestamps:
        try:
            terminal_utc = value.endswith("Z")
            iso_value = value[:-1] + "+00:00" if terminal_utc else value
            iso_value = re.sub(
                r"(\d{2}:\d{2}:\d{2})[.,](\d+)",
                lambda match: match[1] + "." + match[2][:6].ljust(6, "0"),
                iso_value,
            )
            instant = datetime.fromisoformat(iso_value)
            if terminal_utc and instant.utcoffset() is None:
                raise ValueError("UTC designator requires a timezone-aware time")
            parsed.append(instant)
        except ValueError:
            parsed.append(None)
    if not parsed:
        return (), "unavailable"
    if any(value is None for value in parsed):
        return tuple(None for _ in parsed), "unavailable_timestamp"
    try:
        elapsed = tuple((value - parsed[0]).total_seconds() for value in parsed)
    except TypeError:
        return tuple(None for _ in parsed), "mixed_timezone_unavailable"
    if any(right <= left for left, right in zip(elapsed, elapsed[1:])):
        return elapsed, "nonmonotone_timestamp"
    status = (
        "relative_seconds_timezone_unverified"
        if parsed[0].tzinfo is None
        else "relative_seconds"
    )
    return elapsed, status


def _parse_frequency_csv(text: str) -> np.ndarray:
    """Legacy descriptive values view; malformed/missing values remain NaN."""
    _, values, _ = _parse_frequency_rows(text, max_rows=DEFAULT_MAX_ROWS)
    return np.asarray([float("nan") if x is None else x for x in values])


def load_grid_frequency_record(
    zip_path: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_expanded_bytes: int = DEFAULT_MAX_EXPANDED_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_rows: int = DEFAULT_MAX_ROWS,
    member_name: str | None = None,
    expected_archive_sha256: str | None = None,
) -> GridFrequencyRecord:
    """Read one bounded CSV without losing row times, gaps or their alignment.

    Admission errors raise ``ValueError``; no acquisition occurs here. Bounds
    apply to cached archives too, all ZIP entries and their declared expanded
    sizes. The selected member is also read with an actual byte ceiling.
    A supplied archive digest binds the bounded bytes before ZIP parsing.
    Multiple CSV members require an explicit member name.
    """
    for name, value in (
        ("max_points", max_points),
        ("max_bytes", max_bytes),
        ("max_expanded_bytes", max_expanded_bytes),
        ("max_member_bytes", max_member_bytes),
        ("max_members", max_members),
        ("max_rows", max_rows),
    ):
        _positive_limit(value, name)
    expected_digest = None
    if expected_archive_sha256 is not None:
        if (
            type(expected_archive_sha256) is not str
            or re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", expected_archive_sha256)
            is None
        ):
            raise ValueError("expected_archive_sha256 must be a SHA-256 digest")
        expected_digest = expected_archive_sha256.removeprefix("sha256:").lower()
    with Path(zip_path).open("rb") as handle:
        raw_archive = handle.read(max_bytes + 1)
    if len(raw_archive) > max_bytes:
        raise ValueError("archive exceeds compressed-byte limit")
    archive_digest = hashlib.sha256(raw_archive).hexdigest()
    if expected_digest is not None and archive_digest != expected_digest:
        raise ValueError("archive SHA-256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        entries = archive.infolist()
        if len(entries) > max_members:
            raise ValueError("archive exceeds member-count limit")
        if sum(item.file_size for item in entries) > max_expanded_bytes:
            raise ValueError("archive exceeds expanded-byte limit")
        if any(item.file_size > max_member_bytes for item in entries):
            raise ValueError("archive member exceeds expanded-byte limit")
        if len({item.filename for item in entries}) != len(entries):
            raise ValueError("duplicate archive member names are ambiguous")
        candidates = [
            item
            for item in entries
            if not item.is_dir() and item.filename.lower().endswith(".csv")
        ]
        if member_name is not None:
            candidates = [item for item in candidates if item.filename == member_name]
        if not candidates:
            raise ValueError("archive contains no selected CSV member")
        if len(candidates) != 1:
            raise ValueError("multiple CSV members require explicit member_name")
        selected = candidates[0]
        with archive.open(selected) as handle:
            raw = handle.read(max_member_bytes + 1)
        if len(raw) > max_member_bytes or len(raw) != selected.file_size:
            raise ValueError("expanded member failed its byte bound")
    timestamps, values, lines = _parse_frequency_rows(
        raw.decode("utf-8-sig"), max_rows=max_rows
    )
    elapsed, time_status = _relative_times(timestamps)
    stride = max(1, math.ceil(len(values) / max_points))
    return GridFrequencyRecord(
        timestamps=timestamps,
        elapsed_seconds=elapsed,
        values_hz=values,
        missing=tuple(value is None for value in values),
        source_line_numbers=lines,
        selected_indices=tuple(range(0, len(values), stride)),
        stride=stride,
        time_status=time_status,
        archive_sha256=archive_digest,
        member_sha256=hashlib.sha256(raw).hexdigest(),
        member_name=selected.filename,
    )


def load_grid_frequency_series(
    zip_path: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
) -> np.ndarray | None:
    """Legacy descriptive values view; use the record API for physical time.

    Gaps are retained as NaN; this view cannot support derivative/rate claims.
    Returns ``None`` on unavailable input or fewer than 512 raw data rows.
    """
    warnings.warn(
        "values-only grid loading is descriptive; use load_grid_frequency_record "
        "to retain time and gap provenance",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        record = load_grid_frequency_record(zip_path, max_points=max_points)
    except Exception as exc:  # noqa: BLE001 - corrupt/partial archive
        print(f"  [skip] could not read zip: {exc}", file=sys.stderr)
        return None

    if len(record.values_hz) < 512:
        print(
            f"  [skip] parsed only {len(record.values_hz)} data rows", file=sys.stderr
        )
        return None
    return np.asarray(
        [
            float("nan") if record.values_hz[i] is None else record.values_hz[i]
            for i in record.selected_indices
        ],
        dtype=float,
    )


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
        "processing_mode": "retrospective_descriptive",
        "prospective_prediction": False,
        "honest_scope": (
            "Retrospective signal descriptors with same-record channel selection. "
            "Hilbert phase is not a calibrated physical oscillator phase; "
            "these results do not certify prediction or nodal dynamics."
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
        try:
            record = load_grid_frequency_record(
                zip_path, max_points=max_points, max_bytes=max_bytes
            )
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            report["status"] = "unavailable"
            report["reason"] = f"Archive admission failed: {exc}"
            return report
        report["ingestion"] = asdict(record)
        if any(record.missing) or record.time_status not in (
            "relative_seconds",
            "relative_seconds_timezone_unverified",
        ):
            report["status"] = "unavailable"
            report["reason"] = (
                "Missing values or unavailable/nonmonotone timestamps; no gap compaction."
            )
            return report
        series = np.asarray(
            [record.values_hz[i] for i in record.selected_indices], dtype=float
        )
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
    report["status"] = "ok"
    report["transition_index"] = transition_index
    report["n_windows"] = comparison.metadata["n_windows"]
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
