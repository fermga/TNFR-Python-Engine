"""Utility FFT dispatcher implementations for distributed engines."""

from __future__ import annotations

import base64
import json
import pickle
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:  # Optional dependency
    import requests  # type: ignore[import]
except Exception:  # pragma: no cover - requests not available
    requests = None  # type: ignore[assignment]

from .fft_workers import default_fft_worker

Dispatcher = Callable[[str, dict[str, Any]], Any]


@dataclass
class HTTPFFTDispatcher:
    """Simple HTTP dispatcher that exchanges payloads via base64 pickles."""

    base_url: str
    auth_token: str | None = None
    timeout: float = 30.0
    session_factory: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HTTP dispatcher requires a base URL")
        self._session = self._build_session()

    def _build_session(self) -> Any:  # type: ignore[override]
        if requests is None:
            raise RuntimeError("requests is required for HTTPFFTDispatcher")
        if self.session_factory is not None:
            return self.session_factory()
        return requests.Session()

    def dispatch(self, action: str, payload: dict[str, Any]) -> Any:
        if requests is None:
            raise RuntimeError("requests is required for HTTPFFTDispatcher")
        encoded_payload = _encode_payload(payload)
        headers = {"Content-type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        url = f"{self.base_url.rstrip('/')}/{action}"
        response = self._session.post(
            url,
            data=json.dumps({"payload": encoded_payload}),
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        encoded_result = data.get("payload")
        if encoded_result is None:
            return data
        return _decode_payload(encoded_result)


@dataclass
class LocalWorkerDispatcher:
    """Dispatcher that routes actions to in-process callables."""

    registry: Mapping[str, Dispatcher]

    def dispatch(self, action: str, payload: dict[str, Any]) -> Any:
        if action not in self.registry:
            raise KeyError(f"No registered worker for action '{action}'")
        worker = self.registry[action]
        return worker(action, payload)


RequestEncoder = Callable[[dict[str, Any]], Any]
RequestDecoder = Callable[[Any], dict[str, Any]]
ResponseEncoder = Callable[[Any], Any]
ResponseDecoder = Callable[[Any], Any]


class ThreadedQueueDispatcher:
    """Dispatcher backed by one or more background worker threads and queues."""

    def __init__(
        self,
        worker: Dispatcher | None = None,
        *,
        timeout: float = 60.0,
        queue_maxsize: int = 0,
        max_workers: int = 1,
        request_serializer: RequestEncoder | None = None,
        request_deserializer: RequestDecoder | None = None,
        response_serializer: ResponseEncoder | None = None,
        response_deserializer: ResponseDecoder | None = None,
    ) -> None:
        self._worker = worker or default_fft_worker
        self._timeout = timeout
        self._requests: "queue.Queue[tuple[str, str, Any]]" = queue.Queue(
            maxsize=queue_maxsize
        )
        self._responses: dict[str, dict[str, Any]] = {}
        self._response_lock = threading.Lock()
        self._request_serializer = request_serializer or (lambda payload: payload)
        self._request_deserializer = request_deserializer or (lambda payload: payload)
        self._response_serializer = response_serializer or (lambda payload: payload)
        self._response_deserializer = response_deserializer or (lambda payload: payload)
        self._threads: list[threading.Thread] = []
        for index in range(max(1, max_workers)):
            thread = threading.Thread(
                target=self._loop,
                name=f"tnfr-fft-dispatcher-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def dispatch(self, action: str, payload: dict[str, Any]) -> Any:
        request_id = uuid.uuid4().hex
        event = threading.Event()
        serialized = self._request_serializer(payload)
        with self._response_lock:
            self._responses[request_id] = {"event": event}
        self._requests.put((request_id, action, serialized))
        if not event.wait(self._timeout):
            with self._response_lock:
                self._responses.pop(request_id, None)
            raise TimeoutError(
                f"FFT dispatcher timed out waiting for action '{action}'"
            )
        with self._response_lock:
            envelope = self._responses.pop(request_id, None) or {}
        if "error" in envelope:
            raise envelope["error"]
        result_payload = envelope.get("result")
        return self._response_deserializer(result_payload)

    def _loop(self) -> None:
        while True:
            request_id, action, serialized_payload = self._requests.get()
            with self._response_lock:
                handle = self._responses.get(request_id)
            if handle is None:
                continue
            try:
                payload = self._request_deserializer(serialized_payload)
                result = self._worker(action, payload)
                encoded_result = self._response_serializer(result)
            except Exception as exc:  # pragma: no cover - defensive path
                handle["error"] = exc
            else:
                handle["result"] = encoded_result
            finally:
                event = handle.get("event")
                if isinstance(event, threading.Event):
                    event.set()


def _encode_payload(payload: Mapping[str, Any]) -> str:
    blob = pickle.dumps(payload)
    return base64.b64encode(blob).decode("ascii")


def _decode_payload(blob: str) -> Any:
    data = base64.b64decode(blob.encode("ascii"))
    return pickle.loads(data)


__all__ = [
    "Dispatcher",
    "HTTPFFTDispatcher",
    "LocalWorkerDispatcher",
    "ThreadedQueueDispatcher",
]
