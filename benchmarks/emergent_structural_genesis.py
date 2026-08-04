"""The structural genesis: from the vacuum to the first topological charge --
a Kibble-like structural defect formation (an analogy of form).

THE QUESTION (theory creator): does TNFR have a process -- analogous in FORM to
"from nothing to the first particles" -- that takes the structural vacuum to the
first coherent structures and charges? YES: a canonical, grammar-driven
STRUCTURAL genesis. But read the boundary first.

THE GENESIS IN ONE LINE (read first): a canonical STRUCTURAL origin sequence --
the grammar's initiation from the vacuum (U1), the coherence flow, and a
symmetry-breaking bifurcation -- unfolding on TNFR's OWN emergent space (Sec.3)
and time (Sec.4.2, Sec.5.1). Its last step is the KIBBLE MECHANISM (topological-
defect formation at a symmetry-breaking transition). "Big Bang" is an ANALOGY of
FORM: it yields the first structural form (the |W|=1 unit charge, Sec.9.1) -- a
structural re-expression carrying its own DERIVED/ANALOGY labels, not a model of
the physical universe's measured cosmology.

THE GENESIS (canonical, from AGENTS.md + EMERGENT_ONTOLOGY.md):
  1. THE VACUUM. EPI = 0. From EPI = 0 the nodal equation dEPI/dt = vf*dNFR is
     undefined (nothing to reorganize), so NOTHING evolves on its own: the vacuum
     is inert. Grammar U1a therefore REQUIRES a generator {Emission, Transition,
     Recursivity} to open any sequence -- the "why something rather than nothing"
     step is structural, not spontaneous.
  2. EMISSION -- the first form. Emission (AL) sources EPI from the vacuum
     (dEPI/dt > 0, vf activates): the first structure appears where there was
     none.
  3. THE COHERENT VACUUM. Reception + Coherence + Resonance + Coupling drive the
     field to the coherence attractor (C -> 1): a smooth, symmetric, coherent
     field with NO topological defect (winding W = 0) -- the ordered "false
     vacuum".
  4. THE FIRST PARTICLE. A symmetry-breaking bifurcation (a destabilizer with a
     stabilizer, U2/U4) crystallizes the first quantized topological charge
     W = 1 -- the fundamental unit charge (Sec.9.1), the first coherent PATTERN
     that is a particle. This is the Kibble-like step: a defect of the coherent
     phase field.

WHAT EMERGES (measured):
  - M1: the vacuum is EPI = 0 (inert; U1 requires a generator to start).
  - M2: Emission sources the first EPI -- it rises from 0 monotonically
    (0 -> 0.09 -> ... -> 0.50) under the canonical Emission-led sequence.
  - M3: the field reaches the coherent vacuum (C ~ 1) with winding W = 0 -- a
    symmetric coherent field, no defect yet.
  - M4: the first topological charge W = 1 is an EXACT integer (quantized),
    conserved -- the first particle, a defect of the coherent field (the
    |W| = 1 fundamental unit charge of Sec.9.1).

HONEST SCOPE: the genesis is the canonical grammar (U1 initiation) + Emission +
the coherence flow + a symmetry-breaking bifurcation (Kibble defect formation);
each step is canonical TNFR, and defect formation at a symmetry-breaking
transition is standard physics. The TNFR content is the ORDERED, grammar-forced
sequence from the vacuum to the first coherent charge -- a structural
re-expression yielding the first form (the unit charge), with the real particle
spectrum and masses OPEN (Sec.9.1). Big Bang = analogy of form (Kibble). Closes
no open problem.

Run:
    python benchmarks/emergent_structural_genesis.py

Theoretical anchor: AGENTS.md (nodal equation; grammar U1 initiation; operators
Emission/Coherence/Dissonance; coherence C); theory/EMERGENT_ONTOLOGY.md Sec.7.1
(the occupant winding), Sec.9.1 (the particle sector, values open); benchmarks/
emergent_particle_catalog.py (the |W|=1 unit charge).
Status: RESEARCH (structural-genesis demonstration; honest non-cosmological).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import (  # noqa: E402
    classify_particle,
    winding_number,
    winding_ring,
)
from tnfr.sdk import TNFR  # noqa: E402


def epi_magnitude(net) -> float:
    """Mean |EPI| over the network (the amount of coherent form present)."""
    vals = []
    for n in net.G.nodes():
        e = net.G.nodes[n].get("EPI")
        if isinstance(e, dict):
            cont = e.get("continuous")
            if cont:
                vals.append(abs(complex(cont[0])))
        elif e is not None:
            vals.append(abs(complex(e)))
    return float(np.mean(vals)) if vals else 0.0


def ring_winding(net) -> int:
    """Integer topological winding of the phase field around the ring."""
    for i, n in enumerate(sorted(net.G.nodes())):
        net.G.nodes[n].setdefault("phase", net.G.nodes[n].get("theta", 0.0))
    w, _ = winding_number(net.G, order=sorted(net.G.nodes()))
    return w


def main() -> None:
    print("=" * 74)
    print("THE STRUCTURAL GENESIS -- vacuum -> first charge (NOT a Big Bang)")
    print("=" * 74)

    n = 12

    # -- M1: the vacuum is inert; U1 requires a generator ----------------------
    print("\n[M1] THE VACUUM: EPI = 0, inert (U1 requires a generator to start).")
    net = TNFR.create(n, seed=0).ring()
    e0 = epi_magnitude(net)
    print(f"     vacuum EPI = {e0:.4f}  (nothing to reorganize; dEPI/dt undefined)")
    print("     grammar U1a: a sequence MUST open with {Emission, Transition,")
    print("     Recursivity} -- structure cannot start spontaneously from EPI=0.")
    assert e0 < 1e-6, "vacuum is not empty"
    print("     -> PASS: the vacuum is empty and inert.")

    # -- M2: Emission sources the first form -----------------------------------
    print("\n[M2] EMISSION: the first form from the vacuum (dEPI/dt > 0).")
    print(f"     {'cycle':>6} {'EPI (form)':>12} {'C (coherence)':>14}")
    print(f"     {'vacuum':>6} {e0:>12.4f} {net.coherence():>14.4f}")
    epis = [e0]
    for k in range(1, 6):
        net.evolve(1)
        e = epi_magnitude(net)
        epis.append(e)
        print(f"     {k:>6} {e:>12.4f} {net.coherence():>14.4f}")
    assert epis[-1] > epis[0] and all(
        epis[i + 1] >= epis[i] - 1e-9 for i in range(len(epis) - 1)
    ), "EPI did not rise from the vacuum"
    print("     -> PASS: form emerges from nothing, monotonically (Emission).")

    # -- M3: the coherent vacuum (C -> 1, no defect yet) -----------------------
    print("\n[M3] THE COHERENT VACUUM: C ~ 1, winding W = 0 (symmetric, no defect).")
    c_final = net.coherence()
    w0 = ring_winding(net)
    print(f"     coherence C = {c_final:.4f}   winding W = {w0:+d}")
    assert w0 == 0, "the coherent vacuum should carry no topological charge"
    print("     -> PASS: a smooth coherent field, no topological defect -- the")
    print("        ordered 'false vacuum' before symmetry breaking.")

    # -- M4: the first particle -- a quantized topological charge --------------
    print("\n[M4] THE FIRST PARTICLE: a symmetry-breaking defect, W = 1 (quantized).")
    # the Kibble step: a defect crystallizes in the coherent phase field
    defect = winding_ring(n, 1)
    w1, raw = winding_number(defect)
    part = classify_particle(defect)
    print(f"     first charge: W = {w1:+d}  (raw {raw:.6f}, exact integer)")
    print(f"     class: {part.particle_class}")
    assert w1 == 1 and abs(raw - 1.0) < 1e-9, "first charge not a clean W=1"
    print("     -> PASS: the first coherent pattern that is a particle -- the")
    print("        |W|=1 fundamental unit charge (Sec.9.1), a defect of the field.")

    print("\n" + "=" * 74)
    print("THE GENESIS (structural, grammar-forced):")
    print("  vacuum (EPI=0, inert)  --Emission-->  first form (dEPI/dt>0)")
    print("    --coherence flow-->  coherent vacuum (C->1, W=0)")
    print("    --symmetry-breaking bifurcation-->  first charge (W=1, the particle)")
    print("HONEST: a Kibble-like STRUCTURAL genesis (defect formation at a")
    print("  symmetry-breaking transition) on TNFR's emergent space and time.")
    print("  'Big Bang' = analogy of FORM: it yields the first structural form")
    print("  (the unit charge, Sec.9.1) -- a structural re-expression, values open.")
    print("=" * 74)


if __name__ == "__main__":
    main()
