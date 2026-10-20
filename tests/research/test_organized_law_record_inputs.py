"""Read-only moment provenance and final instrument wiring controls."""

import importlib.util
import sys
from pathlib import Path

import pytest

from tnfr.physics import _organized_law_contrast_kernels as producer
from tnfr.research import _organized_law_record_inputs as inputs

ROOT = Path(__file__).resolve().parents[2]


def test_reader_rebuilds_primitives_without_replaying_the_kernel_producer(monkeypatch):
    def forbidden():
        raise AssertionError("frozen moment producer was replayed")

    monkeypatch.setattr(producer, "_assess_organized_law_contrast_kernels", forbidden)
    moments, receipts = inputs._read_organized_law_record_moments(ROOT)
    assert tuple(item.harmonic for item in moments) == (1, 2, 9, 18, 36)
    assert all(item.mode == 2 and 0 < item.mu.lo < item.mu.hi < 1 for item in moments)
    assert all(item.sine_hyperbolic.lo > 0 for item in moments)
    assert tuple(item["path"] for item in receipts) == ("attempt.json", "result.json")


@pytest.mark.parametrize("filename", ("attempt.json", "result.json"))
def test_modified_archive_is_rejected_before_decoding(tmp_path, filename):
    destination = tmp_path / "docs/assets/organized_law_contrast_kernels"
    destination.mkdir(parents=True)
    for name in ("attempt.json", "result.json"):
        raw = (ROOT / "docs/assets/organized_law_contrast_kernels" / name).read_bytes()
        (destination / name).write_bytes(raw + (b" " if name == filename else b""))
    with pytest.raises(ValueError, match="byte association"):
        inputs._read_organized_law_record_moments(tmp_path)


def test_record_instrument_uses_exclusive_runner_without_scientific_execution(
    monkeypatch, tmp_path
):
    name = "organized_record_instrument_wiring"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "benchmarks/organized_law_record_separation.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    seen = {}

    def runner(**kwargs):
        seen.update(kwargs)
        return "synthetic_wiring_verdict"

    monkeypatch.setattr(module, "_run_single_analytic_assessment", runner)
    assert module.main(["--output", str(tmp_path / "result")]) == 0
    assert seen["schema_name"] == "organized_law_record_separation"
    assert seen["assess"] is module._assess
    assert seen["declaration_path"] == "theory/nodal/ORGANIZED_LAW_OBSERVATION.md"
    assert not (tmp_path / "result").exists()
