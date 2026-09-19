"""Optional process telemetry must not block CPU sequence transforms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnfr.engines.computation import unified_gpu_system as gpu_module

ROOT = Path(__file__).resolve().parents[2]
COLD_IMPORT = """
import importlib.abc
import json
import sys

sys.path.insert(0, sys.argv[1])
assert "tnfr" not in sys.modules
assert "psutil" not in sys.modules

class MissingPsutil(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "psutil" or fullname.startswith("psutil."):
            raise ModuleNotFoundError("psutil unavailable in this test", name=fullname)
        return None

sys.meta_path.insert(0, MissingPsutil())

from tnfr.engines.computation.unified_fft_engine import (
    TNFRUnifiedFFTEngine,
    UnifiedFFTConfig,
)

engine = TNFRUnifiedFFTEngine(UnifiedFFTConfig(
    preferred_backend="basic",
    auto_backend_selection=False,
    enable_gpu_acceleration=False,
    log_backend_selection=False,
))
result = engine.compute_fft([1.0, 2.0, 3.0, 4.0])
memory = engine.gpu_manager.get_memory_info()
assert "psutil" not in sys.modules
print(json.dumps({
    "system_memory_mb": memory["system_memory_mb"],
    "backend_used": result.backend_used,
    "real": result.spectral_data.real.tolist(),
    "imag": result.spectral_data.imag.tolist(),
}))
"""


def _telemetry_owner():
    owner = object.__new__(gpu_module.TNFRUnifiedGPUSystem)
    owner._available_devices = []
    owner._current_device = None
    owner._active_allocations = {"declared": 1.25}
    return owner


def test_cold_import_and_cpu_fft_work_without_psutil():
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", COLD_IMPORT, str(ROOT / "src")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["system_memory_mb"] is None
    assert report["backend_used"] == "basic"
    assert report["real"] == [10.0, -2.0, -2.0, -2.0]
    assert report["imag"] == [0.0, 2.0, 0.0, -2.0]


def test_available_psutil_preserves_reported_memory_and_detached_allocations(
    monkeypatch,
):
    module = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=13 * 1024 * 1024)
    )
    monkeypatch.setattr(gpu_module, "cached_import", lambda *args, **kwargs: module)
    owner = _telemetry_owner()

    report = owner.get_memory_info()

    assert report["system_memory_mb"] == 13.0
    assert report["active_allocations"] == {"declared": 1.25}
    report["active_allocations"]["declared"] = 9.0
    assert owner._active_allocations == {"declared": 1.25}


def test_available_memory_reader_errors_are_not_hidden(monkeypatch):
    def fail():
        raise RuntimeError("memory reader failed")

    module = SimpleNamespace(virtual_memory=fail)
    monkeypatch.setattr(gpu_module, "cached_import", lambda *args, **kwargs: module)

    with pytest.raises(RuntimeError, match="memory reader failed"):
        _telemetry_owner().get_memory_info()
