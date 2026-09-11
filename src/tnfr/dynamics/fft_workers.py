"""Background FFT worker utilities for distributed dispatchers."""

from __future__ import annotations

from typing import Any

from ..errors import TNFRValueError
from .advanced_fft_arithmetic import TNFRAdvancedFFTEngine


class _DefaultFFTWorker:
    """Singleton worker that owns a TNFRAdvancedFFTEngine instance."""

    def __init__(self) -> None:
        self._engine = TNFRAdvancedFFTEngine()

    def __call__(self, action: str, payload: dict[str, Any]) -> Any:
        if action == "get_spectral_state":
            graph = payload["graph"]
            force_recompute = payload.get("force_recompute", False)
            return self._engine.get_spectral_state(
                graph, force_recompute=force_recompute
            )
        if action == "spectral_convolution":
            graph = payload["graph"]
            signal1 = payload.get("signal1")
            signal2 = payload.get("signal2")
            operation = payload.get("operation", "multiply")
            return self._engine.spectral_convolution(
                graph,
                signal1=signal1,
                signal2=signal2,
                operation=operation,
            )
        msg = f"Unsupported FFT action '{action}'"
        raise TNFRValueError(
            msg,
            context={
                "action": action,
                "allowed": ["get_spectral_state", "spectral_convolution"],
            },
            suggestion="Use 'get_spectral_state' or 'spectral_convolution'.",
        )


_default_worker = _DefaultFFTWorker()


def default_fft_worker(action: str, payload: dict[str, Any]) -> Any:
    """Return spectral results using the shared TNFRAdvancedFFTEngine."""

    return _default_worker(action, payload)


__all__ = ["default_fft_worker"]
