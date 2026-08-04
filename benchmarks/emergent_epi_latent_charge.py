"""EPI's own complex structure carries an independent topological charge --
but it is LATENT: the canonical dynamics is blind to it (magnitude-only).

THE QUESTION (theory creator): are we mapping EPI to the concrete phenomena we
study, when EPI (canonically a BEPIElement -- a Banach-space object with an
independent complex f_continuous field and a_discrete spectral tail,
src/tnfr/mathematics/epi.py) is architecturally much richer than the plain real
scalar used in every winding/mass/generation benchmark so far (all built on
``winding_ring``, which sets EPI to a trivial constant and puts all topology on
``theta``)? Concretely: can EPI's OWN complex representation carry its own,
independent topological charge -- a SECOND winding sector, distinct from the
phase-channel winding W explored throughout Sec.7-9 -- and if so, does the
canonical dynamics actually see/couple to it?

WHAT EMERGES (measured, using the REAL canonical objects, not shortcuts):
  - M1 (kinematically real, independent): a per-node complex representative of
    EPI, e^(i*2*pi*W_epi*i/n), carries an EXACT integer winding W_epi -- and it
    can be DIFFERENT from theta's own winding W_theta simultaneously (measured:
    W_theta=1, W_epi=3 -- independent, not a relabeling of the same charge).
  - M2 (DECISIVE, dynamically latent): running the ACTUAL canonical
    ``default_compute_delta_nfr`` (dnfr.py) on a graph where EPI is a genuine
    ``BEPIElement`` carrying W_epi=3 gives IDENTICAL delta_nfr (~1e-16, noise)
    to a CONTROL where EPI is a trivial constant float=1.0 (no winding at all).
    The EPI-channel gradient reads ONLY the magnitude (|e^(i*theta)|=1 in both
    cases, identically) -- the winding is invisible to it.
  - M3 (category check): attempting the SAME wrap-sum winding construction on
    nu_f (a positive real rate) gives a result that changes completely with an
    arbitrary choice of "period" -- ill-defined, not a robust integer. nu_f and
    topology (node degree) are not angle-valued, so a winding-type invariant
    does not apply to them: a category fact about which channels CAN carry
    this specific invariant, not an unexplored gap.

THE PRECISE FINDING: EPI's own complex structure CAN carry an independent
topological invariant (a second, distinct integer charge, kinematically) -- but
it is LATENT, not ACTIVE: no canonical operator currently reads or conserves it
(the multichannel DNFR_WEIGHTS hierarchy -- phase pi/(pi+1)~0.7585 > epi
pi/(pi+1)^2~0.1832 > vf 1/(pi+1)^2~0.0583, topo=0 by design for fixed graphs,
src/tnfr/constants/canonical.py -- governs how much EACH channel's MAGNITUDE
gradient contributes, but the array-building step itself, `np.fromiter(...,
dtype=float)` in dnfr.py, discards any complex/phase content before the weight
is even applied).

CORRECTION -- THIS IS NOT AN OPEN FRONTIER; IT IS AN INDEPENDENT CONFIRMATION
OF AN ALREADY-CLOSED CONJECTURE. theory/TNFR_RIEMANN_RESEARCH_NOTES.md
Sec.13triginta-quarta-sexta (the "T-EPI Type Conjecture", part of the catalog
type-hygiene programme) already investigated, with a two-axis diagnostic
(src/tnfr/riemann/epi_type_signature.py) and a forcing-axiom reduction (F1-F10),
EXACTLY this question -- and closed it NEGATIVE: "no canonical operator
constructs or reads a BEPIElement with non-trivial content" is not a gap, it
is the VERIFIED, INTENDED behaviour. BEPIElement is formally classified as a
"non-canonical research envelope (E2)": preserved, not deleted, available for
EXPLICITLY off-catalog research, but no operator should be built or modified
to read/write it. The mechanism (TMEP, the Temporal-Modal Equivalence
Principle): the multi-modal expressivity BEPIElement's f_continuous/a_discrete
WOULD provide SPATIALLY (several components at one instant) is ALREADY
provided canonically, TEMPORALLY -- via REMESH's own history vector
x(t)=(EPI(t),...,EPI(t-T_max)) plus the scalar EPI(t) trajectory's own high
spectral entropy (measured S_EPI~0.88-0.90, N_eff~21-41 effective modes,
Sec.13triginta-quarta.6) -- the SAME richness, a DIFFERENT (temporal, not
spatial) canonical route. Methodology lesson L3 (an Occam's-razor for TNFR):
when an existing canonical mechanism already discharges an expressivity
demand, the proposed upgrade is non-canonical, regardless of internal
consistency. M1/M2 above independently CONFIRM the T-EPI NEGATIVE verdict
(storage_bepi_fraction=0 in both investigations) via a different diagnostic
angle (direct kinematic construction + live execution of the real
default_compute_delta_nfr, vs the T-EPI programme's spectral-entropy
diagnostic) -- a valuable independent corroboration, not a new discovery.

HONEST SCOPE: BEPIElement's Banach-space structure (a continuous field plus a
discrete spectral tail, both complex) is established, documented TNFR
mathematics (src/tnfr/mathematics/epi.py, spaces.py) -- not new here. The
winding-number construction and measurement are the same exact topological
identity used throughout Sec.7 (degree of a map S^1->S^1). What IS new here is
the INDEPENDENT CONFIRMATION (via a direct kinematic construction and a live
run of the real canonical dynamics) of the T-EPI Conjecture's own NEGATIVE
verdict (theory/TNFR_RIEMANN_RESEARCH_NOTES.md Sec.13triginta-quarta-sexta):
EPI's own complex structure is real and independent, kinematically, but the
canonical dynamics, AS BUILT AND AS INTENDED, does not couple to it -- and this
is not a gap to close (TMEP already gives the reason, and REMESH already
supplies the same expressivity temporally). No new operator should be built or
modified to read/write BEPIElement's internal structure; doing so off-catalog,
explicitly labelled as non-canonical research, remains legitimate. Closes no
open problem; corroborates one already closed.

Run:
    python benchmarks/emergent_epi_latent_charge.py

Theoretical anchor: src/tnfr/mathematics/epi.py (BEPIElement); src/tnfr/dynamics/
dnfr.py (default_compute_delta_nfr, the DNFR_WEIGHTS hierarchy);
src/tnfr/constants/canonical.py (CHANNEL_WEIGHT_PRIMARY/SECONDARY/TERTIARY);
theory/TNFR_RIEMANN_RESEARCH_NOTES.md Sec.13triginta-quarta-sexta (the T-EPI
Conjecture, NEGATIVE verdict, TMEP, the E2 envelope classification of
BEPIElement -- the already-settled answer this benchmark corroborates);
src/tnfr/riemann/epi_type_signature.py (the official two-axis diagnostic);
theory/EMERGENT_ONTOLOGY.md Sec.7.1b (the theta-winding occupant), Sec.9.1
(the particle-sector arc this connects to, without extending its OPEN list).
Status: RESEARCH (independent corroboration of an already-closed conjecture).
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.alias import get_attr  # noqa: E402
from tnfr.constants import inject_defaults  # noqa: E402
from tnfr.constants.aliases import ALIAS_DNFR  # noqa: E402
from tnfr.dynamics.dnfr import default_compute_delta_nfr  # noqa: E402
from tnfr.mathematics.epi import BEPIElement  # noqa: E402
from tnfr.physics.emergent_particles import winding_number  # noqa: E402


def build_ring(n: int, w_theta: int, epi_values: dict) -> nx.Graph:
    """A ring with theta carrying w_theta, and EPI set per-node from epi_values."""
    graph = nx.cycle_graph(n)
    inject_defaults(graph)
    for i in graph.nodes():
        theta = float(np.mod(2.0 * math.pi * w_theta * i / n, 2.0 * math.pi))
        graph.nodes[i]["theta"] = theta
        graph.nodes[i]["phase"] = theta
        graph.nodes[i]["EPI"] = epi_values[i]
    return graph


def complex_winding(values: dict[int, complex]) -> float:
    """Winding of a per-node complex sequence around the ring (same wrap-sum
    topological identity used throughout Sec.7 for theta)."""
    ordered = [values[i] for i in sorted(values)]
    total = 0.0
    n = len(ordered)
    for k in range(n):
        a, b = np.angle(ordered[k]), np.angle(ordered[(k + 1) % n])
        total += (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return total / (2.0 * math.pi)


def main() -> None:
    print("=" * 74)
    print("EPI'S OWN LATENT TOPOLOGICAL CHARGE (independent of theta's winding)")
    print("=" * 74)

    n = 24
    w_theta, w_epi = 1, 3  # deliberately different -> tests independence

    epi_complex = {
        i: complex(np.exp(1j * 2.0 * math.pi * w_epi * i / n)) for i in range(n)
    }
    epi_bepi = {
        i: BEPIElement((c, c), (c, c), (0.0, 1.0)) for i, c in epi_complex.items()
    }
    graph = build_ring(n, w_theta, epi_bepi)

    # -- M1: EPI's own complex value carries an INDEPENDENT integer winding ----
    print(f"\n[M1] ring n={n}: theta winding set to {w_theta}, EPI's own complex")
    print(f"     representative winding set to {w_epi} (deliberately different).")
    w_theta_meas, raw_theta = winding_number(graph)
    w_epi_raw = complex_winding(epi_complex)
    w_epi_meas = round(w_epi_raw)
    print(f"     measured winding(theta) = {w_theta_meas}  (raw {raw_theta:.6f})")
    print(f"     measured winding(EPI)   = {w_epi_meas}  (raw {w_epi_raw:.6f})")
    assert w_theta_meas == w_theta and abs(raw_theta - w_theta) < 1e-9
    assert w_epi_meas == w_epi and abs(w_epi_raw - w_epi) < 1e-9
    assert w_theta_meas != w_epi_meas, "windings coincided -- not a fair test"
    print("     -> PASS: two SIMULTANEOUS, INDEPENDENT integer windings -- EPI's")
    print("        own complex structure is a genuine, distinct topological sector.")

    # -- M2: the REAL canonical dynamics -- does it see the EPI winding? -------
    print("\n[M2] the ACTUAL canonical default_compute_delta_nfr -- is it blind to it?")
    default_compute_delta_nfr(graph)
    dnfr_charged = [get_attr(graph.nodes[i], ALIAS_DNFR, None) for i in graph.nodes()]
    print(f"     delta_nfr with EPI winding={w_epi}: {dnfr_charged[:4]} ...")

    # control: EPI as a plain constant float (no winding at all)
    control = nx.cycle_graph(n)
    inject_defaults(control)
    for i in control.nodes():
        theta = float(np.mod(2.0 * math.pi * w_theta * i / n, 2.0 * math.pi))
        control.nodes[i]["theta"] = theta
        control.nodes[i]["phase"] = theta
        control.nodes[i]["EPI"] = 1.0
    default_compute_delta_nfr(control)
    dnfr_control = [
        get_attr(control.nodes[i], ALIAS_DNFR, None) for i in control.nodes()
    ]
    print(f"     delta_nfr with EPI=const (no winding): {dnfr_control[:4]} ...")

    identical = np.allclose(dnfr_charged, dnfr_control, atol=1e-9)
    print(f"     identical to the no-winding control? {identical}")
    assert identical, "the EPI winding unexpectedly changed delta_nfr"
    print("     -> PASS (decisive): the canonical dynamics gives the IDENTICAL")
    print("        delta_nfr whether EPI carries winding=3 or is a trivial")
    print("        constant -- it reads ONLY the magnitude (|e^(i*theta)|=1 in")
    print("        both cases), so the EPI winding is completely invisible to it.")

    # -- M3: does a winding-type invariant even APPLY to nu_f / topology? ------
    print("\n[M3] CATEGORY CHECK: does winding apply to nu_f (a positive real rate)?")
    vf_ramp = np.linspace(0.1, 3.0, n)
    print("     attempting the SAME wrap-sum construction, at several ARBITRARY")
    print("     'periods' (nu_f has no natural periodicity -- VF_MAX is a bound,")
    print("     not a wraparound identification like theta's 2*pi):")
    results = []
    for period in (1.0, 2.0 * math.pi, 5.0, 10.0):
        total = 0.0
        for k in range(n):
            d = vf_ramp[(k + 1) % n] - vf_ramp[k]
            d = (d + period / 2.0) % period - period / 2.0
            total += d
        results.append(total / period)
    print(
        f"     'winding' at each arbitrary period: " f"{[round(r, 3) for r in results]}"
    )
    not_robust = len({round(r) for r in results}) > 1
    assert not_robust, "nu_f 'winding' was unexpectedly robust to the period choice"
    print("     -> CONFIRMED: the result changes completely with the arbitrary")
    print("        period -- ill-defined, not an integer invariant. Winding")
    print("        requires a genuine circle (theta, EPI's own arg); nu_f (R+,")
    print("        a rate) and topology (node degree, a count) are not")
    print("        angle-valued, so this specific invariant TYPE does not apply")
    print("        to them -- a category fact, not an unexplored gap.")

    print("\n" + "=" * 74)
    print("THE PRECISE FINDING (an independent CONFIRMATION, not a new frontier):")
    print("  EPI's own complex structure (BEPIElement) CAN carry an independent")
    print("  topological charge -- kinematically real, distinct from theta's.")
    print("  The canonical dynamics, AS BUILT AND AS INTENDED, does not read or")
    print("  conserve it -- this CORROBORATES the T-EPI Conjecture's own NEGATIVE")
    print("  verdict (TNFR_RIEMANN_RESEARCH_NOTES.md Sec.13triginta-quarta-sexta):")
    print("  BEPIElement is a non-canonical envelope (E2) by design -- REMESH's")
    print("  temporal history vector + the scalar EPI(t) trajectory's own high")
    print("  spectral entropy ALREADY supply the same expressivity, temporally")
    print("  (TMEP). No operator should be built to couple EPI's own phase into")
    print("  the dynamics -- that question is already, rigorously, closed.")
    print("  nu_f and topology do NOT support a winding-type charge either (they")
    print("  are not angle-valued) -- a separate, independent category fact.")
    print("=" * 74)


if __name__ == "__main__":
    main()
