"""Distributed FFT engine stub for large-scale TNFR workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from .advanced_fft_arithmetic import FFTArithmeticResult, SpectralState, TNFRAdvancedFFTEngine
from .fft_backend import FFTBackend, FFTBackendCapabilities

Dispatcher = Callable[[str, Dict[str, Any]], Any]

DISPATCHER_METADATA_ATTR = "__tnfr_dispatcher_metadata__"


def annotate_dispatcher(dispatcher: Dispatcher, metadata: Mapping[str, Any]) -> None:
    payload = dict(metadata)
    try:
        setattr(dispatcher, DISPATCHER_METADATA_ATTR, payload)
        return
    except AttributeError:
        pass

    func = getattr(dispatcher, "__func__", None)
    if func is not None:
        try:
            setattr(func, DISPATCHER_METADATA_ATTR, payload)
        except AttributeError:
            return


def _get_dispatcher_metadata(dispatcher: Dispatcher | None, fallback: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if dispatcher is None:
        return dict(fallback) if fallback is not None else None
    metadata = getattr(dispatcher, DISPATCHER_METADATA_ATTR, None)
    if metadata is None:
        return dict(fallback) if fallback is not None else None
    if isinstance(metadata, Mapping):
        return {str(k): metadata[k] for k in metadata}
    return None


@dataclass
class DistributedFFTConfig:
    """Configuration for the distributed FFT dispatcher."""

    description: str = ""
    endpoint: str | None = None
    auth_token: str | None = None


class DistributedFFTEngine(FFTBackend):
    """Prototype FFT backend that delegates to remote workers when available."""

    backend_name = "tnfr.distributed_fft"

    def __init__(
        self,
        dispatcher: Dispatcher | None = None,
        *,
        config: DistributedFFTConfig | None = None,
        fallback_engine: FFTBackend | None = None,
        telemetry: Dict[str, Any] | None = None,
    ) -> None:
        self._dispatcher = dispatcher
        self._config = config or DistributedFFTConfig()
        self._fallback = fallback_engine or TNFRAdvancedFFTEngine()
        self._dispatcher_metadata = telemetry or _get_dispatcher_metadata(dispatcher)
        self._capabilities = FFTBackendCapabilities(
            backend_name=self.backend_name,
            max_nodes=None,
            precision="float64",
            supports_distributed=True,
            extra={
                "endpoint": self._config.endpoint,
                "description": self._config.description,
                "has_dispatcher": dispatcher is not None,
                "dispatcher": self._dispatcher_metadata,
            },
        )

    def get_capabilities(self) -> FFTBackendCapabilities:  # noqa: D401
        return self._capabilities

    def _dispatch(self, action: str, payload: Dict[str, Any]) -> Any:
        if self._dispatcher is None:
            raise RuntimeError("No dispatcher configured for distributed FFT")
        return self._dispatcher(action, payload)

    def get_dispatcher_telemetry(self) -> Dict[str, Any] | None:  # noqa: D401
        if self._dispatcher_metadata is None:
            return None
        return dict(self._dispatcher_metadata)

    def get_spectral_state(self, G: Any, force_recompute: bool = False) -> SpectralState:
        if self._dispatcher is not None:
            try:
                response = self._dispatch(
                    "get_spectral_state",
                    {"graph": G, "force_recompute": force_recompute},
                )
                if isinstance(response, SpectralState):
                    return response
            except Exception:
                # Fall back to local engine if remote execution fails
                pass
        return self._fallback.get_spectral_state(G, force_recompute=force_recompute)

    def spectral_convolution(
        self,
        G: Any,
        signal1: Any | None = None,
        signal2: Any | None = None,
        operation: str = "multiply",
    ) -> FFTArithmeticResult:
        if self._dispatcher is not None:
            try:
                response = self._dispatch(
                    "spectral_convolution",
                    {
                        "graph": G,
                        "signal1": signal1,
                        "signal2": signal2,
                        "operation": operation,
                    },
                )
                if isinstance(response, FFTArithmeticResult):
                    return response
            except Exception:
                pass
        return self._fallback.spectral_convolution(
            G,
            signal1=signal1,
            signal2=signal2,
            operation=operation,
        )
