from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.figure import Figure
import numpy as np

from tnfr.viz import matplotlib as tnfr_matplotlib


def _patch_savefig(monkeypatch):
    calls: dict[str, object] = {}

    def fake_savefig(self, path, *args, **kwargs):  # type: ignore[override]
        calls["self"] = self
        calls["path"] = Path(path)
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(Figure, "savefig", fake_savefig, raising=True)
    return calls


def test_plot_coherence_matrix_exports_metadata(tmp_path, monkeypatch):
    save_spy = _patch_savefig(monkeypatch)
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])

    fig, ax = tnfr_matplotlib.plot_coherence_matrix(
        matrix,
        channels=["EPI-A", "EPI-B"],
        save_path=tmp_path / "coherence.png",
        dpi=200,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert save_spy["path"] == (tmp_path / "coherence.png").resolve()
    kwargs = save_spy["kwargs"]
    assert kwargs["dpi"] == 200
    assert kwargs["bbox_inches"] == "tight"
    assert kwargs["metadata"]["tnfr_plot"] == "coherence_matrix"
    assert kwargs["metadata"]["engine"] == "TNFR"
    fig.clf()


def test_plot_phase_sync_exports_metadata(tmp_path, monkeypatch):
    save_spy = _patch_savefig(monkeypatch)
    phase_paths = np.array([[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]])
    time_axis = np.array([0.0, 1.0, 2.0])

    fig, ax = tnfr_matplotlib.plot_phase_sync(
        phase_paths,
        time_axis,
        structural_frequency=2.5,
        node_labels=["Emitter", "Receiver"],
        save_path=tmp_path / "phase.png",
        dpi=150,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    kwargs = save_spy["kwargs"]
    assert kwargs["dpi"] == 150
    assert kwargs["bbox_inches"] == "tight"
    assert kwargs["metadata"]["tnfr_plot"] == "phase_sync"
    assert float(kwargs["metadata"]["nu_f_hz_str"]) == 2.5
    fig.clf()


def test_plot_spectrum_path_exports_metadata(tmp_path, monkeypatch):
    save_spy = _patch_savefig(monkeypatch)
    frequencies = np.array([0.5, 1.0, 1.5])
    spectrum = np.array([0.2, 0.5, 0.3])

    fig, ax = tnfr_matplotlib.plot_spectrum_path(
        frequencies,
        spectrum,
        label="C(t) path",
        save_path=tmp_path / "spectrum.png",
        dpi=180,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    kwargs = save_spy["kwargs"]
    assert kwargs["dpi"] == 180
    assert kwargs["bbox_inches"] == "tight"
    assert kwargs["metadata"]["tnfr_plot"] == "spectrum_path"
    assert float(kwargs["metadata"]["nu_f_max"]) == 1.5
    fig.clf()
