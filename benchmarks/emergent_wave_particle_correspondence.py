"""Sampled ring phase modes: spectral and branch-aware winding diagnostics.

For a unit-weight n-cycle, the phase phasor z_i = exp(i*theta_i) with
theta_i = 2*pi*k*i/n is an eigenvector of the symmetric normalized Laplacian:
L_sym z = (1 - cos(2*pi*k/n)) z. The input index k is periodic modulo n.
Away from the even-n Nyquist boundary, its shortest-arc winding is the signed
sampled representative k_star in (-n/2, n/2), not an unrestricted W = k.
The production winding certificate abstains at the wrap branch and reports
U3 admissibility separately. Defined winding does not imply admissible coupling.

The phase phasor z is not the geometric field Psi = K_phi + i*J_phi. A uniform
zero-phase ring has z = 1 and Psi = 0. This benchmark does not identify their
trajectories or derive the auxiliary graph-wave law from the nodal equation.
The ring and phases are declared fixtures, not dynamically emergent entities.

Measurements:
  M1: original n=60 low-mode fixtures plus n=12 alias/branch/U3 controls;
      floating-point eigenvector residual and production winding certificate.
  M2: the five declared amplitudes of one seed-0 real-ripple fixture on n=60.
      Its zero windings do not establish a theorem for arbitrary real ripples.
  M3: a production geometric-field readout demonstrating z != Psi at zero phase.

Scope: finite static classical phase diagnostics. Neither integer winding nor
the spectral identity proves localized particles, a probability amplitude,
detector statistics, physical angular momentum, or quantum wave-particle duality.
The numerical residual tolerance below is a fixture check, not a physical input.

Run:
    python benchmarks/emergent_wave_particle_correspondence.py

Owners: tnfr.physics.structural_diffusion, winding_certificates, and unified.
Theoretical boundary: theory/EMERGENT_ONTOLOGY.md sections 7.1 and 9.2.
Status: RESEARCH (finite spectral/winding controls; no dynamical bridge).
"""

from __future__ import annotations

import math
from numbers import Integral
import pathlib
import sys
from typing import Any

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import winding_ring  # noqa: E402
from tnfr.physics.structural_diffusion import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from tnfr.physics.unified import compute_complex_geometric_field  # noqa: E402
from tnfr.physics.winding_certificates import (  # noqa: E402
    WindingCertificate,
    certify_phase_winding,
)


def _ring_mode_observation(n: int, k: int) -> dict[str, Any]:
    """Read one declared integer phase mode without evolving the graph."""
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise TypeError("k must be an integer mode index")
    # Validate support through the production fixture before reducing the index.
    winding_ring(n, 0)
    sampled_index = int(k) % n
    graph = winding_ring(n, sampled_index)
    nodes, laplacian = symmetric_normalized_laplacian(graph)
    phase_phasor = np.exp(1j * np.array([graph.nodes[i]["theta"] for i in nodes]))
    laplacian_phasor = np.asarray(laplacian) @ phase_phasor
    predicted = 1.0 - math.cos(math.tau * sampled_index / n)
    measured = float(
        np.real(
            np.vdot(phase_phasor, laplacian_phasor)
            / np.vdot(phase_phasor, phase_phasor)
        )
    )
    residual = float(np.max(np.abs(laplacian_phasor - predicted * phase_phasor)))
    representative = (
        None
        if 2 * sampled_index == n
        else sampled_index if 2 * sampled_index < n else sampled_index - n
    )
    return {
        "mode_index": int(k),
        "sampled_representative": representative,
        "phase_phasor": phase_phasor,
        "eigenvalue_measured": measured,
        "eigenvalue_predicted": predicted,
        "eigenvector_residual": residual,
        "certificate": certify_phase_winding(graph, range(n)),
    }


def _real_ripple_observations() -> tuple[tuple[float, WindingCertificate], ...]:
    """Read only the declared seed-0, n=60, three-cosine amplitude fixture."""
    n = 60
    modes = (1, 2, 3)
    coefficients = np.random.default_rng(0).uniform(-1, 1, size=len(modes))
    indices = np.arange(n)
    observations = []
    for amplitude in (0.5, 2.0, 5.0, 10.0, 20.0):
        phases = np.zeros(n)
        for coefficient, mode in zip(coefficients, modes):
            phases += amplitude * coefficient * np.cos(math.tau * mode * indices / n)
        phases = np.mod(phases, math.tau)
        graph = winding_ring(n, 0)
        for node in range(n):
            # Declared initial phase fixture; this is not an evolution step.
            graph.nodes[node]["theta"] = float(phases[node])
            graph.nodes[node]["phase"] = float(phases[node])
        observations.append((amplitude, certify_phase_winding(graph, range(n))))
    return tuple(observations)


def _uniform_phase_field_control(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return distinct phase-phasor and geometric readouts of one zero-phase ring."""
    graph = winding_ring(n, 0)
    geometric_field = compute_complex_geometric_field(graph)
    return (
        np.exp(1j * np.array([graph.nodes[i]["theta"] for i in range(n)])),
        np.array([geometric_field[i] for i in range(n)]),
    )


def main() -> None:
    print("=" * 82)
    print("SAMPLED RING MODES: PHASE PHASOR, WINDING AND DISTINCT GEOMETRIC FIELD")
    print("=" * 82)

    print("\n[M1] Static ring fixtures; k_star is the signed sampled representative.")
    print(
        f"     {'n':>3} {'k':>3} {'k_star':>9} {'lambda':>10} "
        f"{'residual':>10} {'winding':>10} {'U3':>6} {'branch margin':>14}"
    )
    fixtures = [(60, k) for k in (1, 2, 3, 5, 10)]
    fixtures += [(12, k) for k in (1, 5, 6, 7, 13)]
    for n, k in fixtures:
        observation = _ring_mode_observation(n, k)
        certificate = observation["certificate"]
        representative = observation["sampled_representative"]
        winding = certificate.winding if certificate.is_defined else "undefined"
        print(
            f"     {n:>3} {k:>3} {str(representative):>9} "
            f"{observation['eigenvalue_measured']:>10.6f} "
            f"{observation['eigenvector_residual']:>10.2e} {winding:>10} "
            f"{str(certificate.u3_admissible):>6} "
            f"{certificate.minimum_branch_margin:>14.6f}"
        )
        assert observation["eigenvector_residual"] < 1e-9
        assert (
            abs(
                observation["eigenvalue_measured"] - observation["eigenvalue_predicted"]
            )
            < 1e-9
        )
        if representative is None:
            assert not certificate.is_defined, "Nyquist winding must be undefined"
        else:
            assert certificate.is_defined
            assert certificate.winding == representative
            assert abs(certificate.raw_winding - representative) < 1e-9
    print("     -> PASS: W=k_star where defined; alias and Nyquist controls included.")
    print("        U3 is independent telemetry; a defined winding can fail its gate.")

    print("\n[M2] Finite real-ripple fixture: n=60, seed=0, modes=(1,2,3).")
    print(f"     {'amplitude':>10} {'winding':>10} {'U3':>6} {'branch margin':>14}")
    for amplitude, certificate in _real_ripple_observations():
        winding = certificate.winding if certificate.is_defined else "undefined"
        print(
            f"     {amplitude:>10.1f} {winding:>10} "
            f"{str(certificate.u3_admissible):>6} "
            f"{certificate.minimum_branch_margin:>14.6f}"
        )
        assert certificate.is_defined and certificate.winding == 0
    print(
        "     -> PASS: these five initial fixtures have W=0; "
        "no universal ripple claim."
    )

    print(
        "\n[M3] Uniform zero-phase control using the production geometric-field reader."
    )
    phase_phasor, geometric_field = _uniform_phase_field_control()
    assert np.array_equal(phase_phasor, np.ones(12, dtype=complex))
    assert np.array_equal(geometric_field, np.zeros(12, dtype=complex))
    print("     -> PASS: z=exp(i*theta)=1, while Psi=K_phi+i*J_phi=0 on every node.")
    print("\nScope: static sampled phase modes; no emergent particle, detector model,")
    print(
        "       quantum duality, or nodal-to-auxiliary trajectory bridge established."
    )
    print("=" * 82)


if __name__ == "__main__":
    main()
