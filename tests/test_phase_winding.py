import math
import numpy as np


def compute_winding_number(phases: np.ndarray) -> int:
    """Compute integer winding number Q from a closed loop of phases.

    Parameters
    ----------
    phases : np.ndarray
        1D array of phase angles in radians in [0, 2π) sampled
        along a closed loop.

    Returns
    -------
    int
        Integer winding number Q = round(sum Δφ / (2π)).

    Notes
    -----
    We accumulate "principal" angle differences across the closed loop,
    mapping each incremental phase difference into (−π, π] to avoid
    artificial ±2π jumps introduced by branch cuts. This yields Q≈0 for
    plane-wave-like loops with no defect and Q≈±m for m-winding vortices.
    """
    phases = np.asarray(phases, dtype=float)
    # Pairwise principal differences including the closure (last→first)
    d = np.empty_like(phases)
    d[:-1] = np.angle(np.exp(1j * (phases[1:] - phases[:-1])))
    d[-1] = np.angle(np.exp(1j * (phases[0] - phases[-1])))
    total = float(np.sum(d))
    q = int(round(total / (2.0 * math.pi)))
    return q


def test_winding_number_vortex_ring():
    # Generate a synthetic ring with m=+1 winding
    n = 256
    thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    m = 1
    phases = (m * thetas) % (2.0 * math.pi)
    q = compute_winding_number(phases)
    assert q == 1


def test_winding_number_no_defect():
    # Plane-wave along loop should produce net Q≈0
    n = 512
    thetas = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    # small gradient that integrates to near zero around closed loop
    phases = (0.05 * np.sin(thetas)) % (2.0 * math.pi)
    q = compute_winding_number(phases)
    assert q == 0
