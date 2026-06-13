#!/usr/bin/env python3
"""TNFR Multi-channel Structural-Interface Benchmark (coupled-oscillator networks).

This benchmark exercises the *multi-channel* extension of the TNFR Structural
Interface Theory — the setting where the Structural Field Tetrad is genuinely
native.  Where the temporal benchmark embeds a single scalar series, here a set
of simultaneously-measured channels is treated as a network of coupled
oscillators: per-channel Hilbert phase/amplitude feed a phase-locking coupling
graph, and the TNFR spatial tetrad (``|∇φ|``, ``K_φ``, ``ξ_C``, ``Φ_s``) is
tracked window by window and compared against the recognised synchronisation
baselines (Kuramoto order parameter ``R``, mean phase-locking value, spatial
phase dispersion).

Real source: UCI "EEG Eye State"
---------------------------------
14 EEG channels sampled at 128 Hz (Emotiv headset), with a binary label per
sample: eyes open (0) vs eyes closed (1).  Eye closure increases occipital
alpha-band (8–12 Hz) synchronisation, so the two label states differ in
multi-channel phase-coupling structure — a genuine regime change for which the
spatial tetrad is the right tool.  We band-pass to the alpha band before phase
extraction.

Honest scope
------------
- The Kuramoto order parameter ``R`` is the gold-standard global synchrony
  measure and is included as a *fair* baseline, not a strawman.
- ``|∇φ|`` is partially redundant with ``1 − R`` (both fall as the network
  synchronises).  The genuinely distinct TNFR fields are ``ξ_C`` (a spatial
  coherence *length*) and ``K_φ`` (phase-field curvature), which have no
  order-parameter analogue.  The reported question is whether they *add*
  discriminative power; comparable or baseline-favourable outcomes are reported
  honestly.
- The ``synthetic`` source is a **test fixture only** (a Kuramoto network
  switched from incoherent to coherent coupling).  It validates pipeline
  mechanics and is never presented as evidence for the thesis.

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
import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

# Ensure local src is importable ------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from tnfr.validation.multichannel_interface import (  # noqa: E402
    MultichannelConfig,
    evaluate_synchrony_discrimination,
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
    path = cache_path or _ROOT / "results" / "data" / "eeg_eye_state.raw"
    # Treat the cache as valid only if it holds a non-trivial payload; a stale
    # empty/truncated file from an aborted run must not short-circuit the fetch.
    if path.exists() and path.stat().st_size > 1024:
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
            path.write_bytes(buffer.getvalue())
            return path
    return None


def _extract_arff_text(raw: bytes) -> str | None:
    """Return ARFF text from a raw payload that may be a zip or plain ARFF."""
    if raw[:2] == b"PK":  # zip magic
        try:
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                names = [
                    n for n in archive.namelist() if n.lower().endswith(".arff")
                ]
                if not names:
                    names = [
                        n
                        for n in archive.namelist()
                        if not n.endswith("/")
                    ]
                if not names:
                    return None
                return archive.read(names[0]).decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001 - corrupt/partial archive
            print(f"  [skip] could not read zip: {exc}", file=sys.stderr)
            return None
    return raw.decode("utf-8", errors="ignore")


def parse_arff(text: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse EEG Eye State ARFF text into ``(signals, labels)``.

    ``signals`` has shape ``(n_channels, n_samples)`` (channels first); ``labels``
    is the per-sample binary eye-state.  ``@``/``%`` metadata lines are skipped;
    rows are read after ``@DATA``.  Returns ``None`` if too little data parses.
    """
    rows: list[list[float]] = []
    data_started = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            data_started = True
            continue
        if line.startswith("@") or line.startswith("%"):
            continue
        if not data_started:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < N_EEG_CHANNELS + 1:
            continue
        try:
            channels = [float(p) for p in parts[:N_EEG_CHANNELS]]
            label = int(float(parts[N_EEG_CHANNELS]))
        except ValueError:
            continue
        rows.append(channels + [float(label)])
    if len(rows) < 1024:
        print(f"  [skip] parsed only {len(rows)} usable rows", file=sys.stderr)
        return None
    table = np.asarray(rows, dtype=float)
    signals = table[:, :N_EEG_CHANNELS].T  # (channels, samples)
    labels = table[:, N_EEG_CHANNELS].astype(int)
    return signals, labels


def _robust_clip(signals: np.ndarray, *, sigma: float = ROBUST_CLIP_SIGMA) -> np.ndarray:
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


def load_eeg_eye_state(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and clean the cached EEG Eye State payload."""
    raw = path.read_bytes()
    text = _extract_arff_text(raw)
    if text is None:
        return None
    parsed = parse_arff(text)
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
    """Incoherent → coherent Kuramoto regime switch (mechanics fixture only).

    Concatenates a low-coupling (incoherent) block and a high-coupling
    (synchronised) block.  ``labels`` is 0 on the incoherent block and 1 on the
    coherent block.  Used to validate that the tetrad and the baselines both
    track synchronisation; never presented as evidence for the thesis.
    """
    incoherent = kuramoto_simulate(n_oscillators, coupling_low, block, seed=seed + 1)
    coherent = kuramoto_simulate(n_oscillators, coupling_high, block, seed=seed + 2)
    signals = np.concatenate([incoherent, coherent], axis=1)
    labels = np.concatenate([np.zeros(block), np.ones(block)]).astype(int)
    return signals, labels


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------
def _block_means(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Mean indicator value within each label class (for a quick sanity view)."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    finite = np.isfinite(values)
    pos = values[finite & labels]
    neg = values[finite & ~labels]
    return {
        "label0": float(np.mean(neg)) if neg.size else float("nan"),
        "label1": float(np.mean(pos)) if pos.size else float("nan"),
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
            "Multi-channel phase-coupled network: the native setting for the "
            "TNFR spatial tetrad. The Kuramoto order parameter R is the "
            "gold-standard synchrony baseline (fair, not a strawman). |∇φ| is "
            "partially redundant with 1−R; the genuinely distinct TNFR fields "
            "are ξ_C (spatial coherence length) and K_φ (phase-field curvature), "
            "which have no order-parameter analogue. Comparable or "
            "baseline-favourable outcomes are reported honestly."
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
        loaded = load_eeg_eye_state(cache)
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
        report["event_kind"] = "eyes open vs eyes closed (alpha synchronisation)"
    elif source == "synthetic":
        signals, labels = synthetic_kuramoto_regime_switch(
            n_oscillators=max(N_EEG_CHANNELS, 3), block=synthetic_block
        )
        report["data"] = {
            "n_channels": int(signals.shape[0]),
            "n_samples": int(signals.shape[1]),
            "label_balance": float(np.mean(labels)),
            "note": "Synthetic Kuramoto regime switch; mechanics only, not evidence.",
        }
        report["event_kind"] = "incoherent → coherent coupling (fixture)"
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"unknown source: {source}")

    if signals.shape[1] < config.window:
        report["status"] = "skipped"
        report["reason"] = (
            f"Signal length ({signals.shape[1]}) shorter than window "
            f"({config.window})."
        )
        return report

    discrimination = evaluate_synchrony_discrimination(
        signals, labels, config=config
    )
    series = multichannel_window_series(signals, config=config)

    report["status"] = "ok"
    report["n_windows"] = int(series.window_end.size)
    report["n_positive_windows"] = discrimination.n_positive_windows
    report["auc"] = {k: round(float(v), 4) for k, v in discrimination.auc.items()}
    report["best_tnfr"] = {
        "channel": discrimination.best_tnfr[0],
        "auc": round(float(discrimination.best_tnfr[1]), 4),
    }
    report["best_baseline"] = {
        "channel": discrimination.best_baseline[0],
        "auc": round(float(discrimination.best_baseline[1]), 4),
    }
    report["block_means"] = {
        name: _block_means(getattr(series, name), _series_labels(series, labels, config))
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


def _series_labels(
    series: Any, labels: np.ndarray, config: MultichannelConfig
) -> np.ndarray:
    """Majority window labels aligned with the tetrad series (for block means)."""
    arr = np.asarray(labels, dtype=float)
    out = np.empty(series.window_end.size, dtype=bool)
    for i, end in enumerate(series.window_end):
        start = int(end) - config.window + 1
        segment = arr[max(0, start) : int(end) + 1]
        out[i] = bool(np.mean(segment) >= 0.5) if segment.size else False
    return out


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

    print(json.dumps(report, indent=2))

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"multichannel_interface_{args.source}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        display = out_path.relative_to(_ROOT)
    except ValueError:
        display = out_path
    print(f"\nReport written to {display}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
