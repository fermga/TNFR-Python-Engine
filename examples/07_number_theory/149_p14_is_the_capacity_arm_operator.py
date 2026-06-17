#!/usr/bin/env python3
"""
Example 149 — The Riemann Hamiltonian P14 Is the Capacity-Arm Operator of the
Dual-Lever: Why It Sees the Primes and the Substrate Cannot
==============================================================================

Example 148 located the Riemann difficulty on the CAPACITY arm (νf = log) of the
dual-lever and showed the per-node substrate is structurally blind to it because
the substrate encodes the PRESSURE arm (ΔNFR ← Ω). This example closes that loop
by identifying the canonical TNFR-Riemann Hamiltonian P14 as EXACTLY the
capacity-arm operator — which EXPLAINS, structurally, why P14 reproduces the
prime / von Mangoldt data (the zeros) while the pressure substrate is blind. It
unifies three threads under one structure: the dual-lever (physics, examples
37/130), the free monoid on primes (number theory, examples 146-148), and the
prime-ladder Hamiltonian (the TNFR-Riemann program, P14).

The dual-lever, restated
------------------------
Every operator acts on form through the single nodal rule ∂EPI/∂t = νf · ΔNFR,
moving the CAPACITY lever νf (how fast the node reorganizes) or the PRESSURE
lever ΔNFR (the structural forcing) — the dual-lever (examples 37/130). On
arithmetic nodes (examples 147-148) these are the two canonical additive
gradings of the free monoid on primes: SIZE log n (the capacity arm, carrying
von Mangoldt and the Riemann zeros, ex 148) and COUNT Ω (the pressure arm,
carrying the smooth primality field ΔNFR that the substrate encodes).

The identification
------------------
The P14 prime-ladder Hamiltonian (src/tnfr/riemann/prime_ladder_hamiltonian.py)
assigns every node (p,k) the structural frequency νf = k·log p and leaves the
pressure neutral (ΔNFR = 0). So P14 places ALL of its structural information on
the CAPACITY arm — it IS the capacity-arm operator. This is the structural
reason P14 can expose the primes and the Riemann zeros (as the poles of its
weighted trace −ζ'/ζ), while the per-node symplectic substrate — which encodes
the PRESSURE arm Φ_s ← ΔNFR ← Ω — is blind to them (the Fix(G)^⊥ blindness of
examples 103/116/120). The two operators live on the two arms of the dual-lever:
P14 on capacity (sees the zeros), the substrate on pressure (smooth, blind).

Doctrine compliance
-------------------
P14, the weighted trace, and the certificate are canonical
(src/tnfr/riemann/prime_ladder_hamiltonian.py, von_mangoldt.py). The dual-lever
is the canonical operator classification (AGENTS.md "Operator-Tetrad Synergies").
Nothing is imposed; the νf = k·log p assignment and ΔNFR = 0 are read off the
canonical construction and measured.

Three measured results
----------------------
M1 P14 IS THE CAPACITY-ARM OPERATOR. Every P14 node carries νf = k·log p (the
   capacity arm, exact 20/20) and ΔNFR = 0 (the pressure arm is neutral, 20/20).
   P14 places all structural information on the capacity lever — the same axis
   (log = νf) that carries von Mangoldt and the zeros (ex 148).

M2 INTER-PRIME ORTHOGONALITY = THE FREE-MONOID FREEDOM. The prime ladders are
   structurally disconnected — n_primes independent components, each a single
   prime's ladder — so distinct primes are independent invariant subspaces of
   P14. This IS the Euler product at the operator level, and it IS the freedom
   of the free commutative monoid on primes (ex 147): the prime generators do
   not couple.

M3 THE CAPACITY OPERATOR REPRODUCES von MANGOLDT (the zeros' side). P14's
   weighted spectral trace Tr(Ŵ e^{−sĤ}) equals the TNFR von Mangoldt trace
   Z_vM(s) = Σ Λ(n) n^{−s} = −ζ'/ζ(s) (P12), reproduced to machine precision
   (certificate: spectrum error 0, trace rel-error ~1e-16). The zeros are the
   poles of this trace (ex 148 M2). P14 reaches the capacity/zeros side BECAUSE
   it is built on νf = k·log p.

Honest scope
------------
P14 already exists and already reproduces the von Mangoldt trace; this example
adds no new operator and proves nothing new about ζ. The NEW content is the
unifying reading: P14 = the capacity-arm operator of the dual-lever, which
EXPLAINS the capacity-sees / pressure-blind dichotomy of example 148 (P14 lives
on capacity, the substrate on pressure). It does NOT advance RH: G4 remains open,
the oscillatory residue S(T) ∈ ker(R∞) ∩ Fix(S_n)^⊥ — the capacity arm's
oscillatory half — is still the obstruction, and the program stays PAUSED at
T-HP. The value is the synergy it fixes: physics νf-capacity ↔ free-monoid
size-grading ↔ the prime-ladder Hamiltonian, one structure across three modules.

References
----------
- src/tnfr/riemann/prime_ladder_hamiltonian.py (P14: νf = k log p, ΔNFR = 0)
- src/tnfr/riemann/von_mangoldt.py (P12: the von Mangoldt trace = −ζ'/ζ)
- examples/07_number_theory/146_primality_grammatical_inertness.py (the kernel)
- examples/07_number_theory/147_numbers_as_free_monoid_words.py (the dual-lever gradings)
- examples/07_number_theory/148_capacity_arm_carries_von_mangoldt.py (zeros on capacity)
- examples/08_emergent_geometry/130_operators_break_substrate_charges.py (dual-lever)
- theory/TNFR_RIEMANN_RESEARCH_NOTES.md §8-§9 (P12-P14), §13septies (T-HP)
- AGENTS.md "Operator-Tetrad Synergies" (dual-lever), "REMESH-∞ Closure" (range/ker R∞)
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import networkx as nx

from tnfr.riemann.prime_ladder_hamiltonian import (
    build_prime_ladder_hamiltonian,
    verify_hamiltonian_reproduces_prime_ladder,
)
from tnfr.riemann.von_mangoldt import tnfr_log_zeta_derivative

# Dual-lever classification (AGENTS.md "Operator-Tetrad Synergies", example 37):
# the capacity arm is nu_f, the pressure arm is dNFR. P14 lives on capacity.
CAPACITY_ARM = "nu_f"
PRESSURE_ARM = "dNFR"


def experiment_1_p14_is_capacity():
    print("=" * 72)
    print("M1: P14 is the CAPACITY-arm operator (nu_f = k log p, dNFR = 0)")
    print("=" * 72)
    bundle = build_prime_ladder_hamiltonian(5, max_power=4)
    G = bundle.graph
    ok_nu = ok_dnfr = total = 0
    for (p, k), data in G.nodes(data=True):
        total += 1
        if abs(data.get("nu_f", -1) - k * math.log(p)) <= 1e-12:
            ok_nu += 1
        if abs(data.get("dnfr", data.get("delta_nfr", 1.0))) <= 1e-12:
            ok_dnfr += 1
    print(f"  prime ladders for primes {sorted({p for p, k in G.nodes()})}, "
          f"echo depth K=4 -> {total} nodes")
    print(f"  nodes with nu_f = k log p (CAPACITY arm):  {ok_nu}/{total}")
    print(f"  nodes with dNFR = 0 (PRESSURE arm neutral): {ok_dnfr}/{total}")
    print("  sample (p,k) -> nu_f = k log p:")
    for (p, k), data in list(G.nodes(data=True))[:5]:
        print(f"    ({p},{k}) -> nu_f = {data['nu_f']:.4f} = {k}*log({p})")
    print("  -> P14 places ALL structural information on the capacity lever; the")
    print("     same axis (log = nu_f) that carries von Mangoldt + the zeros (ex 148).")
    return bundle


def experiment_2_orthogonality_is_free_monoid(bundle):
    print()
    print("=" * 72)
    print("M2: inter-prime orthogonality = the free-monoid freedom (ex 147)")
    print("=" * 72)
    G = bundle.graph
    components = list(nx.connected_components(G))
    primes = sorted({p for p, k in G.nodes()})
    each_single = all(len({p for p, k in comp}) == 1 for comp in components)
    print(f"  primes: {primes}")
    print(f"  connected components (independent prime ladders): {len(components)}"
          f"  == n_primes {len(primes)}: {len(components) == len(primes)}")
    print(f"  each component is ONE prime's ladder (orthogonal subspace): "
          f"{each_single}")
    print("  -> distinct primes = independent invariant subspaces of P14 = the")
    print("     Euler product at the operator level = the free-monoid generators")
    print("     (ex 147). The prime 'letters' do not couple.")


def experiment_3_capacity_reproduces_von_mangoldt(bundle):
    print()
    print("=" * 72)
    print("M3: the capacity operator reproduces von Mangoldt (the zeros' side)")
    print("=" * 72)
    spectrum = bundle.spectrum
    print("  P14 weighted trace Z_vM(s) = sum Lambda(n) n^-s = -zeta'/zeta(s) (P12):")
    for s in (2.0, 3.0, 4.0):
        z = tnfr_log_zeta_derivative(spectrum, s)
        print(f"    s={s}: Z_vM = {z:.6f}")
    cert = verify_hamiltonian_reproduces_prime_ladder(bundle)
    print(f"  certificate: spectrum reproduced {cert.spectrum_reproduced} "
          f"(max abs error {cert.spectrum_max_abs_error:.1e})")
    print(f"               trace reproduced    {cert.trace_reproduced} "
          f"(max rel error {cert.trace_max_rel_error:.1e})")
    print(f"               overall_ok = {cert.overall_ok}")
    print("  -> the capacity-arm operator's weighted trace IS von Mangoldt; the")
    print("     Riemann zeros are its poles (ex 148 M2). P14 reaches the")
    print("     capacity/zeros side BECAUSE it is built on nu_f = k log p.")


def main():
    print()
    print("#" * 72)
    print("# Example 149 - The Riemann Hamiltonian P14 Is the Capacity-Arm Operator")
    print("#" * 72)
    print()
    bundle = experiment_1_p14_is_capacity()
    experiment_2_orthogonality_is_free_monoid(bundle)
    experiment_3_capacity_reproduces_von_mangoldt(bundle)
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("  The canonical TNFR-Riemann Hamiltonian P14 puts all structural")
    print("  information on the CAPACITY arm of the dual-lever (nu_f = k log p,")
    print("  dNFR = 0). That is exactly the axis carrying von Mangoldt and the")
    print("  Riemann zeros (ex 148), so P14 sees the primes while the per-node")
    print("  substrate -- which encodes the PRESSURE arm (Phi_s <- dNFR <- Omega)")
    print("  -- is blind. Inter-prime orthogonality is the free-monoid freedom")
    print("  (ex 147), the Euler product at the operator level. One structure")
    print("  unifies physics (nu_f), number theory (the monoid), and the Riemann")
    print("  Hamiltonian. No new operator, no RH advance; the program stays")
    print("  paused at T-HP with the obstruction S(T) on the capacity arm's")
    print("  oscillatory half.")
    print()


if __name__ == "__main__":
    main()
