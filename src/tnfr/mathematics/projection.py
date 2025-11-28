"""Projection helpers constructing TNFR state vectors."""

from __future__ import annotations

from ..compat.dataclass import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing hook when numpy.typing is available
    import numpy.typing as npt

    ComplexVector = npt.NDArray[np.complexfloating[np.float64, np.float64]]
else:  # pragma: no cover - runtime fallback without numpy.typing
    ComplexVector = np.ndarray  # type: ignore[assignment]

__all__ = ["StateProjector", "BasicStateProjector"]


@runtime_checkable
class StateProjector(Protocol):
    """Protocol describing state projection callables.

    Notes
    -----
    Marked with @runtime_checkable to enable isinstance() checks for validating
    state projector implementations conform to the expected callable interface.
    """

    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector:
        """Return a normalised TNFR state vector for the provided parameters."""


@dataclass(slots=True)
class BasicStateProjector:
    """Canonical projector building deterministic TNFR state vectors.

    The projector maps the structural scalars of a node—its EPI magnitude,
    structural frequency ``νf`` and phase ``θ``—onto the canonical Hilbert
    basis.  The resulting vector encodes a coherent amplitude envelope derived
    from the structural intensity while the complex exponential captures the
    phase progression across the local modes.  Optional stochastic excitation is
    injected via a :class:`numpy.random.Generator` to model controlled
    dissonance while preserving determinism when a seed is provided.
    """

    dtype: np.dtype[np.complexfloating[np.float64, np.float64]] = np.dtype(np.complex128)
    atol: float = 1e-12

    def __call__(
        self,
        epi: float,
        nu_f: float,
        theta: float,
        dim: int,
        rng: np.random.Generator | None = None,
    ) -> ComplexVector:
        if dim <= 0:
            raise ValueError("State dimension must be a positive integer.")

        indices = np.arange(1, dim + 1, dtype=float)
        phase_progression = theta + (nu_f + 1.0) * indices / max(dim, 1)
        envelope = np.abs(epi) + 0.5 * indices / dim + 1.0
        base_vector = envelope * np.exp(1j * phase_progression)

        if rng is not None:
            noise_scale = (np.abs(epi) + np.abs(nu_f) + 1.0) * 0.05
            real_noise = rng.standard_normal(dim)
            imag_noise = rng.standard_normal(dim)
            stochastic = noise_scale * (real_noise + 1j * imag_noise)
            base_vector = base_vector + stochastic

        norm = np.linalg.norm(base_vector)
        if np.isclose(norm, 0.0, atol=self.atol):
            raise ValueError("Cannot normalise a null state vector.")

        normalised = base_vector / norm
        return np.asarray(normalised, dtype=self.dtype)
