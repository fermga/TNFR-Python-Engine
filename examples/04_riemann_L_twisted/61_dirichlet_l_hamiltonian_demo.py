"""P34 — Canonical TNFR Hamiltonian for χ-twisted prime ladder (demo).

Demonstrates the structural analogue of P14 for general Dirichlet
L-functions: build the canonical TNFR ``InternalHamiltonian`` on the
χ-twisted prime-ladder graph (primes ``p ∤ q``, REMESH echoes up to
``K``), expose the diagonal χ-twisted weight operator
``W^(χ)[(p,k),(p,k)] = χ(p)^k log p``, and verify that

* the decoupled Hamiltonian spectrum reproduces the P32 χ-twisted
  prime-ladder eigenvalues exactly, and
* the χ-twisted weighted spectral trace
  ``Tr(W^(χ) exp(-s H_freq))`` reproduces the P32 reference trace
  ``Z_TNFR(s, χ)`` (= ``-L'(s,χ)/L(s,χ)`` for ``Re(s) > 1``) to
  machine precision.

Tested characters: ``χ_3`` (mod 3), ``χ_4`` (mod 4, = Dirichlet β),
``χ_5`` (mod 5).

Honest scope
------------
P34 closes gap **G1_χ** operationally at the P14 layer (a canonical
self-adjoint operator exists whose decoupled spectrum and χ-twisted
weighted trace match the P32 reference data to machine precision).

P34 does **NOT** advance:

* gap **G4 = RH** or the **Generalised Riemann Hypothesis** (same
  arithmetic obstruction inherited from RH);
* gap **G3_χ** — the χ-twisted Weil–Guinand explicit formula (that
  is the future **P35**).

Run from repo root::

    $env:PYTHONPATH = (Resolve-Path ./src).Path
    & ./.venv312/Scripts/python.exe examples/04_riemann_L_twisted/61_dirichlet_l_hamiltonian_demo.py
"""

from __future__ import annotations

import numpy as np

from tnfr.riemann import (
    build_twisted_prime_ladder_hamiltonian,
    dirichlet_log_l_derivative_continued,
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    tnfr_log_l_derivative,
    twisted_weighted_spectral_trace,
    verify_twisted_hamiltonian_reproduces_prime_ladder,
)


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def step_1_build_chi_3() -> None:
    banner("Step 1 — Build the canonical Hamiltonian for χ_3 (mod 3)")
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi, n_primes=20, max_power=8, coupling=0.0
    )
    H = bundle.hamiltonian
    print(f"character             : {bundle.character_name} (mod {bundle.character_modulus})")
    print(f"primes excluded (p|q) : {bundle.graph.graph['primes_excluded']}")
    print(f"primes active (p∤q)   : {bundle.spectrum.n_active}")
    print(f"max_power K           : {bundle.spectrum.max_power}")
    print(f"Hilbert dimension N   : {H.N}")
    print(f"coupling J_0          : {bundle.coupling}")
    print(f"W^(χ) dtype           : {bundle.weight_operator.dtype}")
    print(f"W^(χ) Hermitian?      : "
          f"{np.allclose(bundle.weight_operator, bundle.weight_operator.conj().T)}"
          " (real χ ⇒ real-diagonal weight)")


def step_2_verify_all_characters() -> None:
    banner("Step 2 — Spectrum + χ-twisted trace reproduction (χ_3, χ_4, χ_5)")
    print(f"{'character':<12} {'N':>5} {'n_active':>9} {'spec_err':>14} "
          f"{'trace_rel_err':>16} {'ok':>4}")
    print("-" * 78)
    s_values = (2.0, 3.0, 2.0 + 1.0j, 3.0 + 2.0j, 5.0, 10.0)
    for chi_factory in (
        real_character_mod_3,
        real_character_mod_4,
        real_character_mod_5,
    ):
        chi = chi_factory()
        bundle = build_twisted_prime_ladder_hamiltonian(
            chi, n_primes=20, max_power=8, coupling=0.0
        )
        cert = verify_twisted_hamiltonian_reproduces_prime_ladder(
            bundle, s_values=s_values, n_primes_seed=20
        )
        print(
            f"{cert.character_name:<12} {cert.hilbert_dim:>5d} "
            f"{cert.n_active:>9d} {cert.spectrum_max_abs_error:>14.3e} "
            f"{cert.trace_max_rel_error:>16.3e} "
            f"{'YES' if cert.overall_ok else 'no':>4}"
        )


def step_3_triple_agreement_chi_3() -> None:
    banner("Step 3 — Triple agreement for χ_3 on Re(s) > 1:\n"
           "         (a) P34 Hamiltonian trace\n"
           "         (b) P32 χ-twisted prime-ladder trace\n"
           "         (c) P33 high-precision continuation of -L'/L")
    chi = real_character_mod_3()
    bundle = build_twisted_prime_ladder_hamiltonian(
        chi, n_primes=30, max_power=10, coupling=0.0
    )
    H = bundle.hamiltonian
    W = bundle.weight_operator
    spectrum = bundle.spectrum

    s_values = [2.0, 2.5, 3.0, 4.0, 2.0 + 1.0j, 3.0 + 2.0j]
    print(f"{'s':<14} {'P34 Tr(W·e^-sH)':>26} {'P32 Z_TNFR(s,χ)':>26} "
          f"{'P33 -L′/L (mpmath)':>26}")
    print("-" * 96)
    for s in s_values:
        z_p34 = twisted_weighted_spectral_trace(H.H_int, W, complex(s))
        z_p32 = complex(tnfr_log_l_derivative(spectrum, complex(s)))
        z_p33 = complex(dirichlet_log_l_derivative_continued(chi, complex(s), dps=40))
        s_str = f"{s.real:+.2f}{s.imag:+.2f}j" if isinstance(s, complex) else f"{s:+.2f}"
        print(
            f"{s_str:<14} "
            f"{z_p34.real:+.6e}{z_p34.imag:+.3e}j   "
            f"{z_p32.real:+.6e}{z_p32.imag:+.3e}j   "
            f"{z_p33.real:+.6e}{z_p33.imag:+.3e}j"
        )

    print("\nNote: P34 ≡ P32 to machine precision by construction;")
    print("      P32 → P33 with relative error O(p_max^{-Re(s)}) as Re(s) > 1.")


def step_4_honest_scope() -> None:
    banner("Honest scope — what P34 closes and what it does NOT")
    print(
        """
P34 establishes (operational closure of gap G1_χ at the P14 layer):

  • A canonical self-adjoint TNFR ``InternalHamiltonian`` on the
    χ-twisted prime-ladder graph (primes coprime to q, REMESH echoes
    up to K) exists for every Dirichlet character χ.
  • Its decoupled spectrum equals the χ-twisted prime-ladder
    spectrum {k log p : p ∤ q, k = 1..K} from P32 EXACTLY.
  • Its χ-twisted weighted spectral trace
    Tr(W^(χ) exp(-s H_freq)) reproduces Z_TNFR(s, χ) ≡ -L'(s,χ)/L(s,χ)
    (on Re(s) > 1) to machine precision.

P34 does NOT establish:

  • Generalised Riemann Hypothesis (GRH) — RH-equivalent in every
    L-function; the arithmetic obstruction (gap G4) is inherited
    unchanged from ζ.
  • The χ-twisted Weil–Guinand explicit formula for L(s, χ) — that
    is the future P35 (gap G3_χ).
  • Any new analytic continuation beyond P33; P34 is purely a
    canonical operator-theoretic representation of the existing
    χ-twisted ladder data.
"""
    )


def main() -> None:
    step_1_build_chi_3()
    step_2_verify_all_characters()
    step_3_triple_agreement_chi_3()
    step_4_honest_scope()


if __name__ == "__main__":
    main()
