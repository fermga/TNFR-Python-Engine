"""
Emergent Chemistry & Particles from TNFR Nodal Dynamics
=======================================================

Demonstration that chemistry and particle classes can be DERIVED from the TNFR
nodal dynamics (∂EPI/∂t = νf·ΔNFR), following the same canonical template that the
number-theory layer uses for primality — instead of being POSTULATED (random-seeded
atoms, hand-placed "electron" vortices) as in the legacy ``tnfr.physics.patterns``
analogy layer.

Unifying principle across all three domains — the SAME structural-equilibrium /
quantization criterion:

    Number theory : n is prime          ⟺  ΔNFR(n) = 0      (intrinsic: Ω, τ, σ)
    Chemistry     : Z is noble (inert)  ⟺  ΔNFR_chem(Z) = 0 (emergent: shell filling)
    Particles     : topological charge  ∈  ℤ (quantized)    (emergent: phase winding)

The quantum regime is NOT imported: a bounded structural manifold admits only
discrete resonant eigenmodes (TNFR discrete-mode regime), and a closed manifold
admits only integer topological charges. Both facts already emerge from the nodal
equation. Atomic number Z is an emergent count of filled eigenmodes; particle class
is an emergent reading of the quantized winding number.

Run:
    python examples/07_number_theory/emergent_chemistry_particles_demo.py
"""

from __future__ import annotations

from tnfr.physics.emergent_chemistry import (
    fibonacci_sphere_graph,
    structural_eigenmodes,
    emergent_magic_numbers,
    classify_element,
)
from tnfr.physics.emergent_particles import (
    winding_ring,
    winding_number,
    classify_particle,
)

try:
    from tnfr.mathematics.number_theory import ArithmeticTNFRFormalism as _NT
    from tnfr.mathematics.number_theory import (
        ArithmeticStructuralTerms,
        ArithmeticTNFRParameters,
    )
    import sympy as _sp

    _HAS_NT = True
except Exception:  # pragma: no cover - optional cross-check
    _HAS_NT = False


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_number_theory_anchor() -> None:
    """Canonical anchor: primality as ΔNFR = 0 (the template we follow)."""
    _rule("ANCHOR — Number theory: primality as ΔNFR = 0 (canonical template)")
    if not _HAS_NT:
        print("  (sympy/number_theory unavailable; skipping anchor)")
        return
    params = ArithmeticTNFRParameters()
    print("  Coefficients derive from (φ, γ, π, e): "
          f"ζ={params.zeta:.4f}  η={params.eta:.4f}  θ={params.theta:.4f}")
    print("  n   ΔNFR(n)   prime?")
    for n in (17, 15, 8, 30):
        terms = ArithmeticStructuralTerms(
            tau=int(_sp.divisor_count(n)),
            sigma=int(_sp.divisor_sigma(n)),
            omega=sum(e for _p, e in _sp.factorint(n).items()),
        )
        dnfr = _NT.delta_nfr_value(n, terms, params)
        print(f"  {n:<3d} {dnfr:7.4f}   {bool(_sp.isprime(n))}")
    print("  → ΔNFR = 0 exactly on primes. Triad derived from intrinsic Ω, τ, σ.")


def demo_chemistry() -> None:
    """Chemistry derived from structural eigenmodes."""
    _rule("CHEMISTRY — atomic structure from structural eigenmodes")

    print("\nSTEP 1 — eigenmode degeneracy EMERGES (structural Laplacian on S²)")
    G = fibonacci_sphere_graph(n_points=400, k_neighbors=6)
    shells = structural_eigenmodes(G)
    for s in shells:
        print(f"  l={s.angular_index}  multiplicity (2l+1) = {s.multiplicity}"
              f"   eigenvalue = {s.eigenvalue:.4f}")
    print(f"  → emergent multiplicities {[s.multiplicity for s in shells]} "
          "= angular eigenmodes 1,3,5,7 (NOT postulated)")

    print("\nSTEP 2 — magic numbers EMERGE from eigenmode filling (νf ∝ n+l)")
    print(f"  emergent noble-gas Z : {emergent_magic_numbers()}")
    print("  empirical noble gases: [2, 10, 18, 36, 54, 86]  (118 = predicted next)")

    print("\nSTEP 3 — octet rule as ΔNFR = 0 (same criterion as primality)")
    samples = {1: "H", 2: "He", 3: "Li", 6: "C", 8: "O", 9: "F",
               10: "Ne", 11: "Na", 17: "Cl", 18: "Ar", 36: "Kr", 54: "Xe"}
    print("   Z  el  valence  ΔNFR_chem   status")
    for Z, name in samples.items():
        e = classify_element(Z)
        status = "CLOSED (noble)" if e.closed_shell else "reactive"
        print(f"  {Z:>3d} {name:<3s}   {e.valence_electrons:>2d}     "
              f"{e.delta_nfr:7.4f}   {status}")
    nobles_ok = all(classify_element(z).closed_shell for z in (2, 10, 18, 36, 54))
    print(f"  → every noble gas has ΔNFR_chem = 0: {nobles_ok}")
    print("  → halogens & alkali share |ΔNFR| = 1/φ ≈ 0.618 (one step from closure)")


def demo_particles() -> None:
    """Particle classes derived from the quantized topological charge."""
    _rule("PARTICLES — coherent modes classified by quantized topological charge")

    print("\nTopological charge QUANTIZES (integer winding, any manifold size)")
    print("  target   n=37   n=128   n=501   (measured winding W)")
    for target in (0.0, 1.0, 2.0, 3.0, -1.0, 1.5):
        ws = [winding_number(winding_ring(n, target))[0] for n in (37, 128, 501)]
        note = "  ← half-integer forbidden → snaps to ℤ" if target == 1.5 else ""
        print(f"  {target:+.1f}     {ws[0]:+d}      {ws[1]:+d}      {ws[2]:+d}{note}")

    print("\nEmergent classification (output of measured invariants, not a label)")
    print("   W   chirality   class")
    for target in (0.0, 1.0, -1.0, 2.0, 3.0):
        p = classify_particle(winding_ring(240, target))
        chir = {1: "matter", -1: "antimatter", 0: "neutral"}[p.chirality]
        print(f"  {p.winding:+d}   {chir:<10s}  {p.particle_class}")
    print("  → |W|=0 boson-like, |W|=1 fermion-like, |W|≥2 composite; sign = chirality")


def main() -> None:
    print(__doc__)
    demo_number_theory_anchor()
    demo_chemistry()
    demo_particles()
    _rule("SUMMARY")
    print("  One criterion, three domains:")
    print("    primality   : ΔNFR(n) = 0")
    print("    noble gas   : ΔNFR_chem(Z) = 0")
    print("    particle    : topological charge ∈ ℤ")
    print("  All derived from ∂EPI/∂t = νf·ΔNFR — none postulated.")


if __name__ == "__main__":
    main()
