"""Consumer typing contracts for backend inspection and canonical failures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def test_backend_stub_supports_typed_consumers(tmp_path: Path) -> None:
    pytest.importorskip("mypy", reason="Backend typing checks require the dev extra")
    root = Path(__file__).resolve().parents[2]
    config = tmp_path / "mypy.ini"
    config.write_text(
        "[mypy]\n"
        f"mypy_path = {root / 'src'}\n"
        "follow_imports = skip\n"
        "ignore_missing_imports = True\n"
        "\n[mypy-tnfr.core.exceptions]\n"
        "follow_imports = normal\n",
        encoding="utf-8",
    )
    probe = tmp_path / "backend_consumer.py"
    probe.write_text(
        """from typing import Any, Mapping

from tnfr.core.exceptions import BackendError
from tnfr.core.exceptions import BackendUnavailableError as CanonicalUnavailable
from tnfr.mathematics.backend import (
    BackendUnavailableError,
    MathematicsBackend,
    _JaxBackend,
    _NumpyBackend,
    _TorchBackend,
    get_backend,
)


def inspect_backend(backend: MathematicsBackend) -> tuple[bool, str, Mapping[str, Any]]:
    return (
        backend.is_gpu_available(),
        backend.get_device_name(),
        backend.get_backend_info(),
    )


def inspect_candidate(candidate: object) -> str | None:
    if isinstance(candidate, MathematicsBackend):
        return candidate.get_device_name()
    return None


backend: MathematicsBackend = get_backend("numpy")
inspection: tuple[bool, str, Mapping[str, Any]] = inspect_backend(backend)
canonical: CanonicalUnavailable = BackendUnavailableError("unavailable")
reexported: BackendUnavailableError = CanonicalUnavailable("unavailable")
family: BackendError = reexported

numpy = _NumpyBackend(_np=object(), _scipy_linalg=None)
jax = _JaxBackend(_jnp=object(), _jax_linalg=object(), _jax=object())
torch = _TorchBackend(
    _torch=object(), _torch_linalg=object(), _device=object(), _use_cuda=False
)
numpy_device: str = numpy.get_device_name()
jax_accelerated: bool = jax.is_gpu_available()
torch_info: Mapping[str, Any] = torch.get_backend_info()
torch_device_info: dict[str, Any] = torch.get_device_info()

# These controls must remain errors; strict unused-ignore checking catches
# an unresolved Any, a false exception superclass, or an empty constructor.
wrong_device: int = backend.get_device_name()  # type: ignore[assignment]
wrong_gpu: str = backend.is_gpu_available()  # type: ignore[assignment]
wrong_info: int = backend.get_backend_info()  # type: ignore[assignment]
wrong_exception: RuntimeError = reexported  # type: ignore[assignment]
missing_numpy_arguments = _NumpyBackend()  # type: ignore[call-arg]
missing_jax_arguments = _JaxBackend()  # type: ignore[call-arg]
missing_torch_arguments = _TorchBackend()  # type: ignore[call-arg]
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(config),
            "--strict",
            "--python-version",
            "3.10",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(probe),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
