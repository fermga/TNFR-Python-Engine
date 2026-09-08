"""Structural initialization, evolution, and winding controls.

This benchmark contains four reproducible observations. They are independent
controls rather than a demonstrated causal genesis:

* M1 reads the SDK initialization EPI = 0, nu_f = 1 Hz_str, and phase = 0.
  Grammar U1 requires a generator at the start of a standalone operator word;
  that is an operator-history contract, not a theorem that the nodal
  derivative is undefined or that physical matter is created from nothing.
* M2 executes the registered basic_activation word for five cycles and prints
  its live stage list. Its seeded fixture shows increasing mean absolute EPI.
  The change cannot be attributed to Emission alone because the complete mixed
  word is executed.
* M3 reads canonical C(t) and winding after those cycles. A single state with
  C(t) = 1 and winding zero does not establish an attractor, a vacuum phase, or
  a symmetry-breaking transition.
* M4 constructs winding_ring(n, 1) directly and checks its integer winding and
  compatibility classifier. It is not produced from M3 by an engine
  trajectory. No Kibble dynamics, bifurcation, particle creation, or winding
  conservation law is executed.

The ring, graph distance, cycle index, and operator order are declared model
coordinates. This script does not derive physical space, physical time, a
causal cone, or cosmological initial conditions. It provides finite graph and
operator diagnostics that can serve as inputs to a future transition study.

Measured claims are limited to the printed seeded values and the exact winding
of the constructed fixture. The name classify_particle is retained from the
public research API; its result is reported here as a topological defect class,
not as evidence of a physical particle.

Status: RESEARCH DIAGNOSTIC. No cosmological or particle-formation claim.

Run:
    python benchmarks/emergent_structural_genesis.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.constants import VF_KEY  # noqa: E402
from tnfr.physics.emergent_particles import (  # noqa: E402
    classify_particle,
    winding_number,
    winding_ring,
)
from tnfr.sdk import TNFR  # noqa: E402
from tnfr.sdk.fluent import NAMED_SEQUENCES  # noqa: E402


def epi_magnitude(net) -> float:
    """Return mean absolute scalar EPI over the network."""
    values = []
    for node in net.G.nodes():
        epi = net.G.nodes[node].get("EPI")
        if isinstance(epi, dict):
            continuous = epi.get("continuous")
            if continuous:
                values.append(abs(complex(continuous[0])))
        elif epi is not None:
            values.append(abs(complex(epi)))
    return float(np.mean(values)) if values else 0.0


def ring_winding(net) -> int:
    """Return integer phase winding in the declared node order."""
    for node in sorted(net.G.nodes()):
        net.G.nodes[node].setdefault(
            "phase",
            net.G.nodes[node].get("theta", 0.0),
        )
    winding, _ = winding_number(net.G, order=sorted(net.G.nodes()))
    return winding


def main() -> None:
    print("=" * 74)
    print("STRUCTURAL INITIALIZATION, EVOLUTION, AND WINDING CONTROLS")
    print("=" * 74)

    n = 12

    print("\n[M1] SDK zero-form initialization.")
    net = TNFR.create(n, seed=0).ring()
    initial_epi = epi_magnitude(net)
    initial_vf = {
        float(net.G.nodes[node][VF_KEY])
        for node in net.G.nodes()
    }
    print(f"     mean |EPI| = {initial_epi:.4f}")
    print(f"     initialized nu_f values = {sorted(initial_vf)} Hz_str")
    print("     U1 requires a generator to open a standalone operator word.")
    assert initial_epi < 1e-6
    assert initial_vf == {1.0}
    print("     -> PASS: this SDK fixture starts at EPI = 0 with active capacity.")

    print("\n[M2] Five cycles of the declared basic_activation word.")
    word = NAMED_SEQUENCES["basic_activation"]
    print(f"     word = {' -> '.join(word)}")
    print(f"     {'cycle':>6} {'mean |EPI|':>12} {'C(t)':>14}")
    print(f"     {'start':>6} {initial_epi:>12.4f} {net.coherence():>14.4f}")
    epi_series = [initial_epi]
    for cycle in range(1, 6):
        net.evolve(1, sequence="basic_activation")
        current_epi = epi_magnitude(net)
        epi_series.append(current_epi)
        print(
            f"     {cycle:>6} {current_epi:>12.4f} "
            f"{net.coherence():>14.4f}"
        )
    assert epi_series[-1] > epi_series[0]
    assert all(
        epi_series[index + 1] >= epi_series[index] - 1e-9
        for index in range(len(epi_series) - 1)
    )
    print("     -> PASS: mean |EPI| increases in this seeded mixed-word fixture.")
    print("        The benchmark does not assign that net change to one stage.")

    print("\n[M3] Read-only diagnostics on the evolved snapshot.")
    final_coherence = net.coherence()
    zero_winding = ring_winding(net)
    print(f"     canonical C(t) = {final_coherence:.4f}")
    print(f"     phase winding W = {zero_winding:+d}")
    assert zero_winding == 0
    print("     -> PASS: the selected snapshot has C(t) = 1 and W = 0.")
    print("        One snapshot is not an attractor or phase-transition proof.")

    print("\n[M4] Directly constructed unit-winding control.")
    constructed = winding_ring(n, 1)
    unit_winding, raw_winding = winding_number(constructed)
    classification = classify_particle(constructed)
    print(
        f"     constructed W = {unit_winding:+d} "
        f"(raw {raw_winding:.6f})"
    )
    print(
        "     compatibility classifier label: "
        f"{classification.particle_class}"
    )
    assert unit_winding == 1
    assert abs(raw_winding - 1.0) < 1e-9
    print("     -> PASS: the constructed graph has the requested winding.")
    print("        It is an input fixture, not the output of M1-M3 dynamics.")

    print("\n" + "=" * 74)
    print("RESULT: four finite controls were verified. The default operator")
    print("word evolves the seeded zero-EPI graph, while the unit-winding graph")
    print("is constructed independently. No Kibble transition, physical")
    print("particle genesis, emergent spacetime, or causal cone is inferred.")
    print("=" * 74)


if __name__ == "__main__":
    main()
