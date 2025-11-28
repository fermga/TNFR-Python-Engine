from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from typing import Sequence

PathLike = str | Path

def plot_coherence_matrix(
    coherence_matrix: np.ndarray,
    *,
    channels: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    dpi: int = 300,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]: ...
def plot_phase_sync(
    phase_paths: np.ndarray,
    time_axis: np.ndarray,
    *,
    structural_frequency: float,
    node_labels: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]: ...
def plot_spectrum_path(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    *,
    label: str = "C(t) spectral density",
    save_path: PathLike | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]: ...
