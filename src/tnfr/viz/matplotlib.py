"""Matplotlib plots for TNFR telemetry channels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

PathLike = str | Path


def _normalise_path(save_path: PathLike | None) -> Path | None:
    """Normalize and validate a save path for visualization exports.

    Parameters
    ----------
    save_path : str | Path | None
        Path where the visualization should be saved, or None.

    Returns
    -------
    Path | None
        Validated and resolved path, or None if save_path is None.

    Raises
    ------
    ValueError
        If the path contains unsafe patterns or path traversal attempts.
    """
    if save_path is None:
        return None

    # Import security utilities
    from ..security import validate_file_path, PathTraversalError

    # Validate the path (allow absolute paths for save operations)
    try:
        validated = validate_file_path(
            save_path,
            allow_absolute=True,
            allowed_extensions=None,  # Allow various image formats
        )
        # Expand user home directory and resolve to absolute path
        return validated.expanduser().resolve()
    except (ValueError, PathTraversalError) as e:
        raise ValueError(f"Invalid save path {save_path!r}: {e}") from e


def _prepare_metadata(
    base: Mapping[str, str] | None = None, **entries: float | str
) -> MutableMapping[str, str]:
    metadata: MutableMapping[str, str] = {"engine": "TNFR"}
    if base is not None:
        metadata.update(base)
    for key, value in entries.items():
        metadata[key] = str(value)
    return metadata


def plot_coherence_matrix(
    coherence_matrix: np.ndarray,
    *,
    channels: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    dpi: int = 300,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """Plot the coherence matrix :math:`C(t)` describing nodal coupling.

    Parameters
    ----------
    coherence_matrix:
        Square matrix reporting pairwise TNFR coherence (0-1). Each entry
        encodes how two nodes sustain a mutual resonance while the total coherence
        :math:`C(t)` evolves.
    channels:
        Optional channel names aligned with the matrix axes.
    save_path:
        Optional filesystem location. When provided the figure is exported with
        explicit metadata so structural logs can capture how :math:`C(t)` was
        rendered.
    dpi:
        Resolution of the exported artifact in dots per inch.
    cmap:
        Matplotlib colormap name used for the coherence heatmap.

    Returns
    -------
    (Figure, Axes)
        The Matplotlib figure and heatmap axis.
    """

    matrix = np.asarray(coherence_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("coherence_matrix must be a square 2D array")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title("TNFR Coherence Matrix C(t)")
    ax.set_xlabel("Emission nodes (νf order)")
    ax.set_ylabel("Reception nodes (νf order)")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Structural coherence C(t)")

    size = matrix.shape[0]
    ticks = np.arange(size)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if channels is not None and len(channels) == size:
        ax.set_xticklabels(channels, rotation=45, ha="right")
        ax.set_yticklabels(channels)

    mean_coherence = float(matrix.mean())
    metadata = _prepare_metadata(
        {"tnfr_plot": "coherence_matrix"},
        c_t_mean=mean_coherence,
        phase_reference="synchrony-check",
    )

    resolved_path = _normalise_path(save_path)
    if resolved_path is not None:
        fig.savefig(
            resolved_path,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )

    return fig, ax


def plot_phase_sync(
    phase_paths: np.ndarray,
    time_axis: np.ndarray,
    *,
    structural_frequency: float,
    node_labels: Sequence[str] | None = None,
    save_path: PathLike | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Plot phase synchrony φ(t) trajectories for TNFR nodes.

    Parameters
    ----------
    phase_paths:
        Array with shape ``(nodes, samples)`` describing the phase of each node
        in radians. Synchronised paths map how the coupling operator preserves
        phase locking.
    time_axis:
        Monotonic timestamps aligned with the samples describing the evolution of
        :math:`C(t)`.
    structural_frequency:
        Global structural frequency :math:`ν_f` (Hz_str) used as the reference
        rate for the displayed phases.
    node_labels:
        Optional labels describing the emitting nodes. When omitted generic
        indices are used.
    save_path:
        Optional filesystem location to export the figure with TNFR metadata.
    dpi:
        Resolution used for exported figures.

    Returns
    -------
    (Figure, Axes)
        The Matplotlib figure and axis holding the phase trajectories.
    """

    phases = np.asarray(phase_paths, dtype=float)
    times = np.asarray(time_axis, dtype=float)
    if phases.ndim != 2:
        raise ValueError("phase_paths must be a 2D array")
    if times.ndim != 1:
        raise ValueError("time_axis must be a 1D array")
    if phases.shape[1] != times.shape[0]:
        raise ValueError("phase_paths samples must align with time_axis")

    fig, ax = plt.subplots(figsize=(7, 4))
    labels: Iterable[str]
    if node_labels is not None and len(node_labels) == phases.shape[0]:
        labels = node_labels
    else:
        labels = (f"node {idx}" for idx in range(phases.shape[0]))

    for path, label in zip(phases, labels):
        ax.plot(times, path, label=label)

    ax.set_title("TNFR Phase Synchrony φ(t)")
    ax.set_xlabel("Time (structural cycles)")
    ax.set_ylabel("Phase (rad)")
    ax.legend(loc="best")

    metadata = _prepare_metadata(
        {"tnfr_plot": "phase_sync"},
        nu_f_hz_str=structural_frequency,
        phase_span=float(np.ptp(phases)),
    )

    resolved_path = _normalise_path(save_path)
    if resolved_path is not None:
        fig.savefig(
            resolved_path,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )

    return fig, ax


def plot_spectrum_path(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    *,
    label: str = "C(t) spectral density",
    save_path: PathLike | None = None,
    dpi: int = 300,
) -> tuple[Figure, Axes]:
    """Plot the spectral path of coherence intensity over structural frequency.

    Parameters
    ----------
    frequencies:
        Frequency samples (Hz_str) that describe the spectrum of ΔNFR driven
        reorganisations.
    spectrum:
        Intensity values tracking how coherence redistributes along the
        structural frequency axis.
    label:
        Legend label identifying the traced path of :math:`C(t)`.
    save_path:
        Optional filesystem location to persist the figure with TNFR metadata.
    dpi:
        Resolution used for exported figures.

    Returns
    -------
    (Figure, Axes)
        The Matplotlib figure and axis holding the spectrum path.
    """

    freq = np.asarray(frequencies, dtype=float)
    spec = np.asarray(spectrum, dtype=float)
    if freq.ndim != 1:
        raise ValueError("frequencies must be a 1D array")
    if spec.ndim != 1:
        raise ValueError("spectrum must be a 1D array")
    if freq.shape[0] != spec.shape[0]:
        raise ValueError("frequencies and spectrum must share the same length")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq, spec, marker="o", label=label)
    ax.fill_between(freq, spec, alpha=0.2)
    ax.set_title("TNFR Structural Spectrum")
    ax.set_xlabel("Structural frequency ν_f (Hz_str)")
    ax.set_ylabel("Coherence intensity C(t)")
    ax.legend(loc="best")

    metadata = _prepare_metadata(
        {"tnfr_plot": "spectrum_path"},
        nu_f_min=float(freq.min()),
        nu_f_max=float(freq.max()),
    )

    resolved_path = _normalise_path(save_path)
    if resolved_path is not None:
        fig.savefig(
            resolved_path,
            dpi=dpi,
            bbox_inches="tight",
            metadata=metadata,
        )

    return fig, ax
