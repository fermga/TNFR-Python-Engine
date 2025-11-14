"""Unified telemetry emitter for TNFR Phase 3.

This module provides a lightweight, unified interface for exporting
structural metrics and canonical field measurements during simulations.

Design Goals (Phase 3):
-----------------------
1. Physics fidelity: All metrics trace directly to TNFR invariants or
   canonical structural fields (Φ_s, |∇φ|, K_φ, ξ_C plus extended suite).
2. Zero mutation: Telemetry collection MUST NOT mutate EPI or ΔNFR.
3. Low overhead: Target <5% added wall time per sampling interval.
4. Fractality aware: Works for nested EPIs (operational fractality).
5. Reproducibility: Includes seed + run id for trajectory replay.
6. Grammar alignment: Does not interfere with operator sequencing
   (U1-U4); U6 confinement data is read-only.

Core Concepts:
--------------
TelemetryEvent: Immutable snapshot of structural metrics.
TelemetryEmitter: Context-managed collector writing JSON Lines and/or
human-readable summaries. Batching is optional; immediate flush by
default for reliability on long runs.

Minimal Public API:
-------------------
TelemetryEmitter(path).record(G, step=..., operator=..., extra=...)
TelemetryEmitter(path).flush()

Extension Points:
-----------------
 - Add selective sampling policies
 - Integrate performance guardrails (duration stats)
 - Attach operator introspection metadata (to be added in Phase 3 task)

Invariants Preserved:
---------------------
1. EPI changes only via operators (no mutation here)
2. νf units preserved (Hz_str not altered)
3. ΔNFR semantics retained (never reframed as loss)
4. Operator closure untouched
5. Phase verification external (we only read phase values)
6. Lifecycle unaffected
7. Fractality supported through recursive traversal utilities (future)
8. Determinism: seed included if provided
9. Structural metrics exported (C(t), Si, phase, νf + fields)
10. Domain neutrality: No domain-specific assumptions

NOTE: This initial implementation focuses on correctness & clarity.
Performance guardrails and operator introspection will hook into this
emitter in subsequent Phase 3 steps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping
import json
import time

try:  # Physics field computations (canonical tetrad + extended suite)
    from ..physics.fields import (
        compute_extended_canonical_suite,  # returns dict
        compute_structural_potential,
        compute_phase_gradient,
        compute_phase_curvature,
        estimate_coherence_length,
    )
except Exception:  # pragma: no cover - graceful degradation
    compute_extended_canonical_suite = None  # type: ignore
    compute_structural_potential = None  # type: ignore
    compute_phase_gradient = None  # type: ignore
    compute_phase_curvature = None  # type: ignore
    estimate_coherence_length = None  # type: ignore

try:  # Existing metrics
    from .sense_index import sense_index  # type: ignore
except Exception:  # pragma: no cover
    sense_index = None  # type: ignore

try:
    from .coherence import compute_coherence  # type: ignore
except Exception:  # pragma: no cover
    compute_coherence = None  # type: ignore

__all__ = ["TelemetryEmitter", "TelemetryEvent"]


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Immutable telemetry snapshot.

    Fields
    ------
    t_iso : str
        ISO8601 timestamp for wall-clock time.
    t_epoch : float
        Seconds since UNIX epoch.
    step : int | None
        Simulation step / operator index (if provided).
    operator : str | None
        Last applied operator mnemonic (AL, IL, OZ, etc.).
    metrics : Mapping[str, Any]
        Structural metrics dictionary.
    extra : Mapping[str, Any] | None
        User-supplied contextual additions (seed, run_id, notes, ...).
    """

    t_iso: str
    t_epoch: float
    step: int | None
    operator: str | None
    metrics: Mapping[str, Any]
    extra: Mapping[str, Any] | None = None


class TelemetryEmitter:
    """Unified telemetry collector for TNFR simulations.

    Parameters
    ----------
    path : str | Path
        Output file path (JSON Lines). Parent directories are created.
    flush_interval : int, default=1
        Number of events to batch before auto-flush. 1 = flush each event.
    include_extended : bool, default=True
        If True, compute extended canonical suite when available for
        efficiency; otherwise compute tetrad fields individually.
    safe : bool, default=True
        If True, wraps metric computations in try/except returning partial
        results on failure (never raises during record).
    human_mirror : bool, default=False
        If True, writes a sibling *.log file with concise summaries.

    Notes
    -----
    The emitter never mutates graph state; it only reads node attributes.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        flush_interval: int = 1,
        include_extended: bool = True,
        safe: bool = True,
        human_mirror: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = max(1, int(flush_interval))
        self.include_extended = bool(include_extended)
        self.safe = bool(safe)
        self.human_mirror = bool(human_mirror)
        self._buffer: list[TelemetryEvent] = []
        self._start_time = time.perf_counter()
        self._human_path = (
            self.path.with_suffix(".log") if self.human_mirror else None
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    def __enter__(self) -> "TelemetryEmitter":  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        try:
            self.flush()
        finally:
            # No open handles to close (using append mode on demand)
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(
        self,
        G: Any,
        *,
        step: int | None = None,
        operator: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Capture a telemetry snapshot.

        Parameters
        ----------
        G : Any
            TNFR graph-like object with node attributes.
        step : int | None
            Simulation step index.
        operator : str | None
            Last operator mnemonic for sequencing context.
        extra : Mapping[str, Any] | None
            Additional context (seed, run_id, grammar_state, etc.).
        """

        metrics: MutableMapping[str, Any] = {}

        def _compute() -> None:
            # Core structural metrics
            if compute_coherence is not None:
                try:
                    metrics["coherence_total"] = float(compute_coherence(G))
                except Exception:
                    if not self.safe:
                        raise
            if sense_index is not None:
                try:
                    metrics["sense_index"] = float(sense_index(G))
                except Exception:
                    if not self.safe:
                        raise

            # Canonical field tetrad (plus extended suite if available)
            if (
                self.include_extended
                and compute_extended_canonical_suite is not None
            ):
                try:
                    suite = compute_extended_canonical_suite(G)
                    if isinstance(suite, Mapping):
                        for k, v in suite.items():
                            metrics[k] = v
                except Exception:
                    if not self.safe:
                        raise
            else:
                # Tetrad individually
                if compute_structural_potential is not None:
                    try:
                        metrics["phi_s"] = compute_structural_potential(G)
                    except Exception:
                        if not self.safe:
                            raise
                if compute_phase_gradient is not None:
                    try:
                        metrics["phase_grad"] = compute_phase_gradient(G)
                    except Exception:
                        if not self.safe:
                            raise
                if compute_phase_curvature is not None:
                    try:
                        metrics["phase_curv"] = compute_phase_curvature(G)
                    except Exception:
                        if not self.safe:
                            raise
                if estimate_coherence_length is not None:
                    try:
                        metrics["xi_c"] = estimate_coherence_length(G)
                    except Exception:
                        if not self.safe:
                            raise

        if self.safe:
            try:
                _compute()
            except Exception:
                # Swallow and proceed with partial metrics
                pass
        else:
            _compute()

        # Use timezone-aware UTC to avoid deprecation of datetime.utcnow()
        event = TelemetryEvent(
            t_iso=datetime.now(UTC).isoformat(timespec="seconds"),
            t_epoch=time.time(),
            step=step,
            operator=operator,
            metrics=dict(metrics),
            extra=dict(extra) if extra else None,
        )
        self._buffer.append(event)
        if len(self._buffer) >= self.flush_interval:
            self.flush()
        return event

    def flush(self) -> None:
        """Flush buffered telemetry events to disk."""
        if not self._buffer:
            return
        # JSON Lines write
        with self.path.open("a", encoding="utf-8") as fh:
            for ev in self._buffer:
                fh.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        if self._human_path is not None:
            with self._human_path.open("a", encoding="utf-8") as hf:
                for ev in self._buffer:
                    coh = ev.metrics.get("coherence_total")
                    si = ev.metrics.get("sense_index")
                    phi = (
                        ev.metrics.get("phi_s")
                        or ev.metrics.get("structural_potential")
                    )
                    hf.write(
                        (
                            f"[{ev.step}] op={ev.operator} C={coh:.3f} "
                            f"Si={si:.3f} Φ_s={phi} t={ev.t_iso}\n"
                        )
                    )
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Introspection / diagnostics
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return emitter internal statistics (buffer + runtime)."""
        return {
            "buffer_len": len(self._buffer),
            "flush_interval": self.flush_interval,
            "include_extended": self.include_extended,
            "uptime_sec": time.perf_counter() - self._start_time,
            "path": str(self.path),
        }


# Convenience helper -------------------------------------------------------
def stream_telemetry(
    G: Any,
    *,
    emitter: TelemetryEmitter,
    steps: Iterable[int],
    operator_sequence: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> list[TelemetryEvent]:
    """Record telemetry across a sequence of steps.

    Parameters
    ----------
    G : Any
        TNFR graph instance.
    emitter : TelemetryEmitter
        Active telemetry emitter.
    steps : Iterable[int]
        Step indices to record.
    operator_sequence : Iterable[str] | None
        Optional operator mnemonics aligned with steps.
    extra : Mapping[str, Any] | None
        Additional context (seed/run id).
    """

    events: list[TelemetryEvent] = []
    ops_iter = (
        iter(operator_sequence) if operator_sequence is not None else None
    )
    for s in steps:
        op_name = next(ops_iter) if ops_iter is not None else None
        events.append(emitter.record(G, step=s, operator=op_name, extra=extra))
    emitter.flush()
    return events
