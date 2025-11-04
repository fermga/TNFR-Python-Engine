from __future__ import annotations

from typing import Dict

import pytest

from benchmarks import evolution_backend_speedup as benchmark_module

class _DummyBackend:
    def __init__(self, name: str) -> None:
        self.name = name

@pytest.mark.parametrize(
    "requested, expected_names, expected_numpy_calls",
    [
        ("np", ["numpy"], 0),
        ("pytorch", ["numpy", "torch"], 1),
    ],
)
def test_resolve_backends_accepts_aliases_without_skip(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    requested: str,
    expected_names: list[str],
    expected_numpy_calls: int,
) -> None:
    calls: list[str] = []
    backend_names: Dict[str, str] = {
        "numpy": "numpy",
        "np": "numpy",
        "torch": "torch",
        "pytorch": "torch",
        "jax": "numpy",  # Simulate fallback behaviour for unavailable backend
    }

    def _fake_get_backend(name: str) -> _DummyBackend:
        calls.append(name)
        try:
            canonical = backend_names[name]
        except KeyError as exc:  # pragma: no cover - guard for unexpected names
            raise AssertionError(f"Unexpected backend request: {name}") from exc
        return _DummyBackend(canonical)

    monkeypatch.setattr(benchmark_module, "get_backend", _fake_get_backend)

    resolved = benchmark_module._resolve_backends([requested, "jax"])
    out = capsys.readouterr().out

    assert [entry[0] for entry in resolved] == expected_names
    assert calls.count("numpy") == expected_numpy_calls
    assert f"[skip] backend '{requested}'" not in out
    assert "[skip] backend 'jax'" in out
