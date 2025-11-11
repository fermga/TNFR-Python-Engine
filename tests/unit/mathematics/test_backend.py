from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from tnfr.mathematics import backend as backend_module


@pytest.fixture(autouse=True)
def reset_backend_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset backend caches and configuration between tests."""

    backend_module._BACKEND_CACHE.clear()
    monkeypatch.delenv("TNFR_MATH_BACKEND", raising=False)

    feature_flags = importlib.import_module("tnfr.config.feature_flags")
    feature_flags._BASE_FLAGS = None
    feature_flags._FLAGS_STACK.clear()
    yield
    backend_module._BACKEND_CACHE.clear()
    monkeypatch.delenv("TNFR_MATH_BACKEND", raising=False)
    feature_flags._BASE_FLAGS = None
    feature_flags._FLAGS_STACK.clear()


@contextmanager
def _mock_torch(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Provide minimal torch stubs so the backend can be instantiated."""

    class DummyTensor:
        def __init__(self, value: np.ndarray | list[float] | float):
            self._value = np.asarray(value)

        def detach(self) -> "DummyTensor":
            return self

        def cpu(self) -> "DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return self._value

        def conj(self) -> "DummyTensor":
            return DummyTensor(np.conjugate(self._value))

        def transpose(self, *axes: int) -> "DummyTensor":
            return DummyTensor(np.transpose(self._value, axes=axes or None))

        @property
        def mH(self) -> "DummyTensor":
            return DummyTensor(np.conjugate(self._value.T))

        def __matmul__(self, other: object) -> "DummyTensor":
            return DummyTensor(self._value @ _to_array(other))

        def __array__(self) -> np.ndarray:  # pragma: no cover - numpy interop only
            return self._value

    def _to_array(value: object) -> np.ndarray:
        if isinstance(value, DummyTensor):
            return value._value
        return np.asarray(value)

    class DummyLinalg:
        def eig(self, matrix: object) -> tuple[DummyTensor, DummyTensor]:
            vals, vecs = np.linalg.eig(_to_array(matrix))
            return DummyTensor(vals), DummyTensor(vecs)

        def eigh(self, matrix: object) -> tuple[DummyTensor, DummyTensor]:
            vals, vecs = np.linalg.eigh(_to_array(matrix))
            return DummyTensor(vals), DummyTensor(vecs)

        def matrix_exp(self, matrix: object) -> DummyTensor:
            vals, vecs = np.linalg.eig(_to_array(matrix))
            inv = np.linalg.inv(vecs)
            return DummyTensor(vecs @ np.diag(np.exp(vals)) @ inv)

        def norm(
            self, value: object, ord: object = None, dim: object = None
        ) -> DummyTensor:
            arr = _to_array(value)
            if dim is None:
                return DummyTensor(np.linalg.norm(arr, ord=ord))
            return DummyTensor(np.linalg.norm(arr, ord=ord, axis=dim))

    class DummyTorch:
        def as_tensor(self, value: object) -> DummyTensor:
            return DummyTensor(np.asarray(value))

        def einsum(
            self, pattern: str, *operands: object, **kwargs: object
        ) -> DummyTensor:
            return DummyTensor(
                np.einsum(pattern, *(_to_array(o) for o in operands), **kwargs)
            )

        def matmul(self, a: object, b: object) -> DummyTensor:
            return DummyTensor(_to_array(a) @ _to_array(b))

        def stack(self, arrays: object, dim: int = 0) -> DummyTensor:
            stacked = np.stack([_to_array(x) for x in arrays], axis=dim)
            return DummyTensor(stacked)

    dummy_torch = DummyTorch()
    dummy_linalg = DummyLinalg()

    original_cached_import = backend_module.cached_import

    def fake_cached_import(
        name: str, attr: str | None = None, **kwargs: object
    ) -> object:
        if name == "torch" and attr is None:
            return dummy_torch
        if name == "torch.linalg" and attr is None:
            return dummy_linalg
        if name == "numpy" and attr is None:
            return np
        return original_cached_import(name, attr=attr, **kwargs)

    monkeypatch.setattr(backend_module, "cached_import", fake_cached_import)
    try:
        yield
    finally:
        monkeypatch.setattr(backend_module, "cached_import", original_cached_import)


def test_default_backend_is_numpy() -> None:
    backend = backend_module.get_backend()
    assert backend.name == "numpy"


def test_alias_resolution_returns_numpy() -> None:
    backend = backend_module.get_backend("np")
    assert backend.name == "numpy"


def test_environment_prefers_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TNFR_MATH_BACKEND", "torch")
    with _mock_torch(monkeypatch):
        backend = backend_module.get_backend()
    assert backend.name == "torch"


def test_context_flags_override(monkeypatch: pytest.MonkeyPatch) -> None:
    with _mock_torch(monkeypatch):
        from tnfr.config import context_flags

        with context_flags(math_backend="torch"):
            backend = backend_module.get_backend()
            assert backend.name == "torch"

    backend_module._BACKEND_CACHE.clear()
    assert backend_module.get_backend().name == "numpy"


def test_register_backend_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        backend_module.register_backend(
            "numpy", lambda: backend_module.get_backend("numpy")
        )
