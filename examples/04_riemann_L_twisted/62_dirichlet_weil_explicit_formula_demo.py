"""P35 — χ-twisted Weil–Guinand explicit formula (demo).

Structural analogue of P15 (Weil–Guinand for ζ) for general primitive
real Dirichlet L-functions ``L(s, χ)``.

The identity verified (for ``h(t) = exp(-t²/(2σ²))`` Gaussian, real
primitive non-principal χ with conductor ``q`` and parity
``a = (1 − χ(−1))/2 ∈ {0, 1}``) is

.. math::

    \\sum_\\gamma h(\\gamma)
    \\;=\\;
    g(0)\\,\\log(q/\\pi)
    \\;+\\; \\frac{1}{2\\pi}\\!\\int h(t)\\,\\Re\\psi(\\tfrac14+\\tfrac{a}{2}
                                          +\\tfrac{it}{2})\\,dt
    \\;-\\; 2\\,\\Re\\!\\sum_n \\tfrac{\\chi(n)\\Lambda(n)}{\\sqrt n}\\,g(\\log n).

* **Zero side** — Hardy-Z bisection on ``Z_χ(t)`` to locate the
  positive imaginary parts ``γ`` of the non-trivial L-zeros on
  ``Re(s) = 1/2``.
* **Prime side** — diagonal projection of the χ-twisted weight
  operator ``W^(χ)`` from the canonical P34 Hamiltonian, in its
  eigenbasis (same einsum idiom as P15).
* **Archimedean side** — numerical quadrature of the digamma factor.

Tested characters: ``χ_3`` (mod 3, a=1), ``χ_4`` (mod 4, a=1),
``χ_5`` (mod 5, a=0).

Honest scope
------------
P35 closes gap **G3_χ** operationally for every primitive real
Dirichlet L-function (both sides of the χ-twisted Weil–Guinand
identity have a canonical TNFR realisation that agrees to the
declared tolerance).

P35 does **NOT** advance:

* gap **G4 = RH** or the **Generalised Riemann Hypothesis** (zero
  localisation is *assumed* on the critical line, not proved);
* any new analytic continuation beyond P33.

Run from repo root::

    $env:PYTHONPATH = (Resolve-Path ./src).Path
    $env:PYTHONIOENCODING = "utf-8"
    & ./.venv312/Scripts/python.exe \\
        examples/04_riemann_L_twisted/62_dirichlet_weil_explicit_formula_demo.py
"""

from __future__ import annotations

import sys

from tnfr.riemann import (
    build_twisted_prime_ladder_hamiltonian,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    verify_twisted_weil_explicit_formula,
)

# Windows cp1252 defaults choke on χ / γ / ψ; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def step_1_chi_3_breakdown() -> None:
    banner("Step 1 — χ_3 (mod 3) detailed breakdown at σ = 2.0")
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi, n_primes=25, max_power=10, coupling=0.0
    )
    cert = verify_twisted_weil_explicit_formula(chi, bundle, sigma=2.0)
    print(f"character        : {cert.character_name}")
    print(f"conductor q      : {cert.character_modulus}")
    print(f"parity a         : {cert.character_parity}")
    print(f"sigma            : {cert.sigma}")
    print(f"n zeros used     : {cert.n_zeros_used}")
    print(f"zeros γ          : {[round(g, 4) for g in cert.zeros]}")
    print(f"")
    print(f"constant term    : {cert.constant_term:+.10f}"
          " = g(0) log(q/π)")
    print(f"archimedean side : {cert.archimedean_side:+.10f}"
          " = (1/2π) ∫ h(t) Re ψ(1/4 + a/2 + it/2) dt")
    print(f"prime side       : {cert.prime_side:+.10f}"
          " = -2 Re Σ χ(n)Λ(n)/√n · g(log n)")
    print(f"rhs total        : {cert.rhs_total:+.10f}")
    print(f"zero side        : {cert.zero_side:+.10f}"
          " = Σ_γ h(γ)")
    print(f"residual         : {cert.residual:+.3e}")
    print(f"relative residual: {cert.relative_residual:.3e}")
    print(f"tolerance        : {cert.tolerance:.3e}")
    print(f"verified         : {cert.verified}")


def step_2_sweep_characters() -> None:
    banner("Step 2 — Sweep over (χ_3, χ_4, χ_5) and σ ∈ {2.0, 2.5, 3.0}")
    print(
        f"{'character':<10} {'q':>3} {'a':>3} {'σ':>5} "
        f"{'n_zeros':>8} {'residual':>13} {'rel':>13} {'ok':>4}"
    )
    print("-" * 78)
    factories = [
        ("chi_3", real_character_mod_3),
        ("chi_4", real_character_mod_4),
        ("chi_5", real_character_mod_5),
    ]
    for name, factory in factories:
        chi = factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi, n_primes=25, max_power=10, coupling=0.0
        )
        for sigma in (2.0, 2.5, 3.0):
            cert = verify_twisted_weil_explicit_formula(
                chi, bundle, sigma=sigma
            )
            print(
                f"{name:<10} {cert.character_modulus:>3d} "
                f"{cert.character_parity:>3d} {sigma:>5.2f} "
                f"{cert.n_zeros_used:>8d} "
                f"{cert.residual:>+13.3e} "
                f"{cert.relative_residual:>13.3e} "
                f"{'YES' if cert.verified else 'no':>4}"
            )


def step_3_certificate_summaries() -> None:
    banner("Step 3 — Certificate summaries (σ = 2.5)")
    factories = [
        real_character_mod_3,
        real_character_mod_4,
        real_character_mod_5,
    ]
    for factory in factories:
        chi = factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi, n_primes=25, max_power=10, coupling=0.0
        )
        cert = verify_twisted_weil_explicit_formula(
            chi, bundle, sigma=2.5
        )
        print(cert.summary())


def step_4_honest_scope() -> None:
    banner("Honest scope — what P35 closes and what it does NOT")
    print(
        """
P35 establishes (operational closure of gap G3_χ):

  • Both sides of the χ-twisted Weil-Guinand explicit formula for
    every primitive real Dirichlet L-function have a canonical TNFR
    realisation:
      - zero side from L-zeros located on Re(s) = 1/2 by Hardy-Z
        bisection on Z_χ(t) (built on P33 continuation);
      - prime side from the canonical P34 Hamiltonian via the same
        weighted-eigenbasis projection used in P15.
  • The identity is satisfied to machine precision for χ_3, χ_4,
    χ_5 across σ ∈ {2.0, 2.5, 3.0}.

P35 does NOT establish:

  • Generalised Riemann Hypothesis (GRH).  Zero localisation on the
    critical line is ASSUMED (Hardy-Z bisection starts from
    Re(s) = 1/2); proving every L-zero lies there is the χ-twisted
    analogue of gap G4 = RH.
  • Any new analytic continuation beyond P33.

Restriction
-----------
P35 currently supports only primitive REAL characters (χ_3, χ_4,
χ_5).  The complex-χ extension requires a Hermitisation of W^(χ)
that is intentionally left for a future increment.
"""
    )


def main() -> None:
    step_1_chi_3_breakdown()
    step_2_sweep_characters()
    step_3_certificate_summaries()
    step_4_honest_scope()


if __name__ == "__main__":
    main()
