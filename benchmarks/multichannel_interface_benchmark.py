#!/usr/bin/env python3
"""Multichannel signal-derived graph observations and finite score comparisons.

Per-channel Hilbert phase/amplitude feed a supplied phase-locking graph. Shared
TNFR fields are compared with the Kuramoto order parameter, mean phase-locking
value and phase dispersion. This observational graph does not identify measured
canonical wiring, capacity or a physical pressure law.

Real source: UCI "EEG Eye State"
---------------------------------
14 EEG channels sampled at 128 Hz (Emotiv headset), with a binary label per
sample: eyes open (0) vs eyes closed (1). The selected alpha-band preprocessing
and label discrimination do not themselves establish a physical transition or
an independently validated TNFR measurement bridge.

Honest scope
------------
- Local phase stress and global phase order can correlate. Different formulas
  and finite univariate AUCs do not prove independent predictive information.
- ``ξ_C`` can come from a static product fit or a separate spectral fallback.
  The current adapter does not retain that provenance, so this report cannot
  identify its value as a measured correlation length.
- The ``synthetic`` source concatenates separately initialized low/high-coupling
  Kuramoto blocks. It validates pipeline mechanics; their join is not one
  continuously evolved switch or evidence for a TNFR law.

Usage (PowerShell)::

    $env:PYTHONPATH=(Resolve-Path -Path ./src).Path
    # Real EEG data (downloaded + cached, bounded size):
    python benchmarks/multichannel_interface_benchmark.py --source eeg \
        --output results/reports
    # Offline synthetic Kuramoto fixture:
    python benchmarks/multichannel_interface_benchmark.py --source synthetic
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from decimal import Decimal, InvalidOperation
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Ensure local src is importable ------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from tnfr.utils.io import json_dumps, safe_write  # noqa: E402
from tnfr.validation.multichannel_interface import (  # noqa: E402
    MultichannelConfig,
    _evaluate_synchrony_series,
    _window_labels,
    multichannel_window_series,
)

# UCI EEG Eye State (dataset 264).  The post-2023 UCI layout serves a zip; the
# classic mirror serves the raw ARFF.  Both are tried in order; either yields
# the same 14-channel + label table.
EEG_EYE_STATE_URLS = (
    "https://archive.ics.uci.edu/static/public/264/eeg+eye+state.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00264/"
    "EEG%20Eye%20State.arff",
)

DEFAULT_MAX_BYTES = 20_000_000  # bounded download guard (~20 MB)
EEG_SAMPLING_RATE_HZ = 128.0
ALPHA_BAND_HZ = (8.0, 12.0)
N_EEG_CHANNELS = 14
# Known acquisition spikes (sensor pops) are clipped at this robust z-threshold.
ROBUST_CLIP_SIGMA = 6.0


# ---------------------------------------------------------------------------
# Real data acquisition (bounded, cached, graceful-skip)
# ---------------------------------------------------------------------------
def _byte_limit(max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral):
        raise ValueError("max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    return int(max_bytes)


def download_eeg_eye_state(
    *,
    cache_path: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = 60.0,
) -> Path | None:
    """Download the EEG Eye State dataset with a hard size bound.

    Tries each candidate URL in turn; the raw payload (zip or ARFF) is cached
    under ``results/data``.  Returns the cached path, or ``None`` on any failure
    (no network, HTTP error, oversized payload) so the caller can skip
    gracefully offline.
    """
    max_bytes = _byte_limit(max_bytes)
    path = cache_path or _ROOT / "results" / "data" / "eeg_eye_state.raw"
    # Treat the cache as valid only if it holds a non-trivial payload; a stale
    # empty/truncated file from an aborted run must not short-circuit the fetch.
    if path.exists():
        size = path.stat().st_size
        if size > max_bytes:
            print(f"  [skip] cached EEG exceeds {max_bytes} bytes", file=sys.stderr)
            return None
        if size > 1024:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for url in EEG_EYE_STATE_URLS:
        try:
            request = Request(url, headers={"User-Agent": "tnfr-benchmark/1.0"})
            buffer = BytesIO()
            total = 0
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                while True:
                    chunk = response.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        print(
                            f"  [skip] download exceeded {max_bytes} bytes; "
                            "aborting",
                            file=sys.stderr,
                        )
                        buffer = None  # type: ignore[assignment]
                        break
                    buffer.write(chunk)
                if buffer is None:
                    continue
        except Exception as exc:  # noqa: BLE001 - graceful offline skip
            print(f"  [skip] EEG download failed ({url}): {exc}", file=sys.stderr)
            continue
        if buffer is not None and total > 0:
            safe_write(path, lambda stream: stream.write(buffer.getvalue()), mode="wb")
            return path
    return None


def _extract_arff_text(raw: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES) -> str | None:
    """Decode one bounded ARFF member without silently dropping corrupt bytes."""
    max_bytes = _byte_limit(max_bytes)
    if len(raw) > max_bytes:
        print(f"  [skip] EEG payload exceeds {max_bytes} bytes", file=sys.stderr)
        return None
    try:
        if raw[:2] == b"PK":  # zip magic
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                members = [
                    member
                    for member in archive.infolist()
                    if not member.is_dir() and member.filename.lower().endswith(".arff")
                ]
                if len(members) != 1:
                    raise ValueError("EEG archive must contain exactly one ARFF member")
                if members[0].file_size > max_bytes:
                    raise ValueError("expanded EEG member exceeds byte limit")
                with archive.open(members[0]) as stream:
                    raw = stream.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    raise ValueError("expanded EEG member exceeds byte limit")
        return raw.decode("utf-8-sig")
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"  [skip] could not decode EEG payload: {exc}", file=sys.stderr)
        return None


def parse_arff(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse EEG Eye State ARFF text into ``(signals, labels)``.

    ``signals`` has shape ``(n_channels, n_samples)`` (channels first); ``labels``
    is the per-sample binary eye-state. Metadata precedes ``@DATA`` and ``%``
    comments are skipped. Malformed, nonfinite or nonbinary rows reject the
    payload instead of compacting its sample clock. Returns ``None`` if the
    complete valid record contains fewer than 1024 samples.
    """
    rows: list[list[float]] = []
    data_started = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered == "@data" and not data_started:
            data_started = True
            continue
        if line.startswith("%"):
            continue
        if not data_started:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != N_EEG_CHANNELS + 1:
            raise ValueError(
                f"EEG row {line_number} must contain 14 channels and a label"
            )
        try:
            channels = [float(p) for p in parts[:N_EEG_CHANNELS]]
            label = Decimal(parts[N_EEG_CHANNELS])
        except (ValueError, InvalidOperation) as exc:
            raise ValueError(
                f"EEG row {line_number} contains a nonnumeric value"
            ) from exc
        if (
            not all(np.isfinite(channels))
            or not label.is_finite()
            or label not in (0, 1)
        ):
            raise ValueError(
                f"EEG row {line_number} requires finite channels and a binary label"
            )
        rows.append(channels + [float(label)])
    if len(rows) < 1024:
        print(f"  [skip] parsed only {len(rows)} usable rows", file=sys.stderr)
        return None
    table = np.asarray(rows, dtype=float)
    signals = table[:, :N_EEG_CHANNELS].T  # (channels, samples)
    labels = table[:, N_EEG_CHANNELS].astype(int)
    return signals, labels


def _robust_clip(
    signals: np.ndarray, *, sigma: float = ROBUST_CLIP_SIGMA
) -> np.ndarray:
    """Clip per-channel sensor spikes to a robust ``median ± σ·MAD`` band."""
    cleaned = np.array(signals, dtype=float, copy=True)
    for j in range(cleaned.shape[0]):
        row = cleaned[j]
        med = float(np.median(row))
        mad = float(np.median(np.abs(row - med)))
        if mad <= 0.0:
            continue
        scale = 1.4826 * mad  # MAD -> approx std for normal data
        lo = med - sigma * scale
        hi = med + sigma * scale
        cleaned[j] = np.clip(row, lo, hi)
    return cleaned


def load_eeg_eye_state(
    path: Path, *, max_bytes: int = DEFAULT_MAX_BYTES
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a bounded complete EEG record and clean its supplied channel values."""
    max_bytes = _byte_limit(max_bytes)
    with path.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    text = _extract_arff_text(raw, max_bytes=max_bytes)
    if text is None:
        return None
    try:
        parsed = parse_arff(text)
    except ValueError as exc:
        print(f"  [skip] invalid EEG record: {exc}", file=sys.stderr)
        return None
    if parsed is None:
        return None
    signals, labels = parsed
    return _robust_clip(signals), labels


# ---------------------------------------------------------------------------
# Synthetic test fixture (mechanics only; never presented as evidence)
# ---------------------------------------------------------------------------
def kuramoto_simulate(
    n_oscillators: int,
    coupling: float,
    steps: int,
    *,
    dt: float = 0.05,
    mean_omega: float = 1.0,
    omega_spread: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Euler-integrate a Kuramoto network; observable is ``sin(θ_j(t))``.

    Returns shape ``(n_oscillators, steps)``.  Below the critical coupling the
    network stays incoherent; well above it the oscillators phase-lock.  The
    observable is the *signal* ``sin θ`` (not the latent phase), so the pipeline
    must recover phase via the Hilbert transform, exactly as for real data.
    """
    rng = np.random.default_rng(seed)
    omega = rng.normal(mean_omega, omega_spread, n_oscillators)
    theta = rng.uniform(-np.pi, np.pi, n_oscillators)
    out = np.empty((n_oscillators, steps), dtype=float)
    for t in range(steps):
        z = np.mean(np.exp(1j * theta))
        order = np.abs(z)
        psi = np.angle(z)
        theta = theta + dt * (omega + coupling * order * np.sin(psi - theta))
        out[:, t] = np.sin(theta)
    return out


def synthetic_kuramoto_regime_switch(
    *,
    n_oscillators: int = 14,
    block: int = 4096,
    coupling_low: float = 0.05,
    coupling_high: float = 2.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate separately initialized low/high-coupling Kuramoto fixtures.

    Concatenates a low-coupling (incoherent) block and a high-coupling
    (synchronised) block.  ``labels`` is 0 on the incoherent block and 1 on the
    coherent block. Labels name the supplied coupling preparation; the join
    is not one continuously evolved switch or evidence for the TNFR thesis.
    """
    incoherent = kuramoto_simulate(n_oscillators, coupling_low, block, seed=seed + 1)
    coherent = kuramoto_simulate(n_oscillators, coupling_high, block, seed=seed + 2)
    signals = np.concatenate([incoherent, coherent], axis=1)
    labels = np.concatenate([np.zeros(block), np.ones(block)]).astype(int)
    return signals, labels


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
def _block_means(values: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    """Finite class means; absent classes/values are unavailable, encoded as null."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    pos = values[finite & labels]
    neg = values[finite & ~labels]
    return {
        "label0": float(np.mean(neg)) if neg.size else None,
        "label1": float(np.mean(pos)) if pos.size else None,
    }


def run_multichannel_benchmark(
    *,
    source: str,
    config: MultichannelConfig,
    max_bytes: int = DEFAULT_MAX_BYTES,
    synthetic_block: int = 4096,
) -> dict[str, Any]:
    """Run the multi-channel structural-interface benchmark on the chosen source."""
    report: dict[str, Any] = {
        "source": source,
        "config": {
            "window": config.window,
            "step": config.step,
            "k_neighbours": config.k_neighbours,
            "sampling_rate": config.sampling_rate,
            "bandpass": list(config.bandpass) if config.bandpass else None,
        },
        "honest_scope": (
            "Signal-derived graph observations with supplied Hilbert phases "
            "and an amplitude-pressure proxy. Finite score rankings do not "
            "establish measured canonical wiring, a physical nodal law or "
            "independent predictive information. The adapter omits xi_C "
            "fit/fallback provenance; its numeric value alone cannot identify "
            "a measured correlation length."
        ),
    }

    if source == "eeg":
        cache = download_eeg_eye_state(max_bytes=max_bytes)
        if cache is None:
            report["status"] = "skipped"
            report["reason"] = (
                "EEG Eye State source unreachable in this environment. Re-run "
                "with network access; pipeline mechanics are validated offline "
                "via --source synthetic."
            )
            return report
        loaded = load_eeg_eye_state(cache, max_bytes=max_bytes)
        if loaded is None:
            report["status"] = "skipped"
            report["reason"] = "Downloaded EEG payload could not be parsed."
            return report
        signals, labels = loaded
        report["data"] = {
            "n_channels": int(signals.shape[0]),
            "n_samples": int(signals.shape[1]),
            "label_balance": float(np.mean(labels)),
            "cache": str(cache.relative_to(_ROOT)),
            "label_semantics": "0 = eyes open, 1 = eyes closed",
        }
        report["event_kind"] = "eyes-open versus eyes-closed labels"
    elif source == "synthetic":
        signals, labels = synthetic_kuramoto_regime_switch(
            n_oscillators=max(N_EEG_CHANNELS, 3), block=synthetic_block
        )
        report["data"] = {
            "n_channels": int(signals.shape[0]),
            "n_samples": int(signals.shape[1]),
            "label_balance": float(np.mean(labels)),
            "note": (
                "Concatenated independently initialized Kuramoto blocks; "
                "pipeline mechanics only, not evidence for a TNFR law."
            ),
        }
        report["event_kind"] = "concatenated low/high coupling fixtures"
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown source: {source}")

    if signals.shape[1] < config.window:
        report["status"] = "skipped"
        report["reason"] = (
            f"Signal length ({signals.shape[1]}) shorter than window "
            f"({config.window})."
        )
        return report

    series = multichannel_window_series(signals, config=config)
    discrimination = _evaluate_synchrony_series(
        series,
        labels,
        config=config,
        n_channels=signals.shape[0],
        n_samples=signals.shape[1],
    )

    report["status"] = "ok"
    report["n_windows"] = int(series.window_end.size)
    report["n_positive_windows"] = discrimination.n_positive_windows
    report["auc_available"] = discrimination.metadata["auc_available"]
    report["auc_unavailable_reason"] = discrimination.metadata["auc_unavailable_reason"]
    report["auc"] = {k: round(float(v), 4) for k, v in discrimination.auc.items()}
    report["best_tnfr"] = {
        "channel": discrimination.best_tnfr[0],
        "auc": round(float(discrimination.best_tnfr[1]), 4),
    }
    report["best_baseline"] = {
        "channel": discrimination.best_baseline[0],
        "auc": round(float(discrimination.best_baseline[1]), 4),
    }
    window_labels = _window_labels(labels, series, config.window)
    report["block_means"] = {
        name: _block_means(getattr(series, name), window_labels)
        for name in (
            "grad_phi",
            "k_phi",
            "xi_c",
            "phi_s",
            "order_parameter",
            "mean_plv",
            "phase_dispersion",
        )
    }
    report["interpretation"] = discrimination.interpretation
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TNFR multi-channel structural-interface benchmark."
    )
    parser.add_argument(
        "--source",
        choices=("eeg", "synthetic"),
        default="eeg",
        help="Data source: real EEG Eye State or synthetic Kuramoto fixture.",
    )
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--k-neighbours", type=int, default=4)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--synthetic-block", type=int, default=4096)
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "reports",
        help="Directory for the JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.source == "eeg":
        config = MultichannelConfig(
            window=args.window,
            step=args.step,
            k_neighbours=args.k_neighbours,
            sampling_rate=EEG_SAMPLING_RATE_HZ,
            bandpass=ALPHA_BAND_HZ,
        )
    else:
        config = MultichannelConfig(
            window=args.window,
            step=args.step,
            k_neighbours=args.k_neighbours,
        )

    report = run_multichannel_benchmark(
        source=args.source,
        config=config,
        max_bytes=args.max_bytes,
        synthetic_block=args.synthetic_block,
    )

    serialized = json_dumps(report, indent=2, allow_nan=False)
    print(serialized)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    out_path = output_dir / f"multichannel_interface_{args.source}.json"
    safe_write(out_path, lambda stream: stream.write(serialized + "\n"))
    try:
        display = out_path.relative_to(_ROOT)
    except ValueError:
        display = out_path
    print(f"\nReport written to {display}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
