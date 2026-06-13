r"""P36 demo: chi-twisted Li-Keiper positivity criterion.

Structural analogue of the P16 Li-Keiper demo for primitive real
Dirichlet L-functions. Computes the chi-twisted Li-Keiper coefficients

    lambda_n(chi) = sum_k 2 Re[ 1 - (1 - 1/rho_k)^n ],
                    rho_k = 1/2 + i t_k,

where (t_k) are the non-trivial zeros of L(s, chi) enumerated on the
critical line by the P35 Hardy-Z bisection routine
(find_dirichlet_l_zeros). Verifies the Lagarias generalisation of
Li 1997: GRH_chi <=> lambda_n(chi) > 0 for every n >= 1.

Tested on primitive real characters chi_3 (mod 3, parity 1),
chi_4 (mod 4, parity 1; Dirichlet beta), chi_5 (mod 5, parity 0).

Honest scope
------------
A finite positivity check of lambda_1, ..., lambda_n_max is a
necessary condition for GRH, NOT sufficient. P36 provides a
TNFR-native finite diagnostic witness; it does NOT prove GRH for any
L(s, chi) and does NOT advance G4 = RH or the arithmetic
obstruction of GRH.

Run from repo root::

    $env:PYTHONPATH = (Resolve-Path ./src).Path
    $env:PYTHONIOENCODING = "utf-8"
    & ./.venv312/Scripts/python.exe \
        examples/04_riemann_L_twisted/63_dirichlet_li_keiper_demo.py
"""

from __future__ import annotations

import sys

from tnfr.riemann import (
    real_character_mod_3,
    real_character_mod_4,
    real_character_mod_5,
    verify_twisted_li_keiper_criterion,
)

# Windows cp1252 defaults choke on chi / gamma / lambda; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def step_1_chi_3_breakdown() -> None:
    banner("Step 1 — χ_3 (mod 3) detailed Li-Keiper breakdown, n_max=20")
    chi = real_character_mod_3()
    cert = verify_twisted_li_keiper_criterion(
        chi, n_max=20, t_max=80.0, character_label="chi_3"
    )
    print(cert.summary())
    print()
    print("  full lambda_n table:")
    for n, lam in enumerate(cert.lambda_coefficients, start=1):
        marker = "  " if lam > 0 else "!!"
        print(f"    {marker} lambda_{n:>2d} = {lam:+.6e}")
    print()
    print(f"  positivity for n = 1..{cert.n_max}: {cert.positivity}")
    print(f"  min lambda_n            : {cert.min_lambda:+.6e}")


def step_2_sweep_three_characters() -> None:
    banner(
        "Step 2 — Positivity sweep: χ_3, χ_4, χ_5 × n_max ∈ {20, 30, 50}"
    )

    characters = [
        ("chi_3", real_character_mod_3()),
        ("chi_4", real_character_mod_4()),
        ("chi_5", real_character_mod_5()),
    ]
    n_max_values = [20, 30, 50]

    print(
        f"  {'character':<10} {'q':>3} {'parity':>6} "
        f"{'n_max':>6} {'#zeros':>7} {'min λ_n':>15} "
        f"{'positive?':>10}"
    )
    print("  " + "-" * 70)

    for name, chi in characters:
        for n_max in n_max_values:
            cert = verify_twisted_li_keiper_criterion(
                chi,
                n_max=n_max,
                t_max=80.0,
                character_label=name,
            )
            ok = "yes" if cert.positivity else "NO"
            print(
                f"  {cert.character_name:<10} {cert.character_modulus:>3}"
                f" {cert.character_parity:>6} {cert.n_max:>6}"
                f" {cert.n_zeros_used:>7}"
                f" {cert.min_lambda:>+15.6e} {ok:>10}"
            )


def step_3_certificate_summaries() -> None:
    banner(
        "Step 3 — Certificate summaries at n_max = 30 across χ_3, χ_4, χ_5"
    )
    characters = [
        ("chi_3", real_character_mod_3()),
        ("chi_4", real_character_mod_4()),
        ("chi_5", real_character_mod_5()),
    ]
    for name, chi in characters:
        cert = verify_twisted_li_keiper_criterion(
            chi, n_max=30, t_max=80.0, character_label=name
        )
        print(cert.summary())
        print()


def step_4_honest_scope() -> None:
    banner("Step 4 — Honest scope of P36")
    print(
        "Lagarias generalisation of Li 1997 states:\n"
        "    GRH for L(s, χ)  ⇔  λ_n(χ) > 0 for every n ≥ 1.\n"
        "\n"
        "The verification above only checks λ_n > 0 for n = 1..n_max\n"
        "from a finite truncation of the zero list (every zero with\n"
        "0 < t < t_max). This is necessary but NOT sufficient for GRH:\n"
        "rigorous bounds on the truncation tail are required to upgrade\n"
        "the finite check to a proof, which P36 does NOT provide.\n"
        "\n"
        "What P36 DOES advance (structural):\n"
        "  • TNFR-native finite diagnostic witness for GRH on L(s, χ)\n"
        "    via the canonical P35 Hardy-Z zero enumerator.\n"
        "  • Structural analogue of P16 (li_keiper) at the L-function\n"
        "    level: the same canonical formula\n"
        "        λ_n = Σ_k 2 Re[1 - (1 - 1/ρ_k)^n]\n"
        "    applies because the sum over zeros is L-function agnostic.\n"
        "\n"
        "What P36 does NOT advance:\n"
        "  • G4 = RH (the arithmetic obstruction on Re(s) = 1/2\n"
        "    localisation persists; we ASSUME the critical-line\n"
        "    location via Hardy-Z bisection on Z_χ).\n"
        "  • GRH for any specific L(s, χ) -- only a diagnostic, not\n"
        "    a proof.\n"
        "  • The structural derivation of the smooth zero density\n"
        "    for L(s, χ) (analogue of P28 for L-functions).\n"
    )


if __name__ == "__main__":
    step_1_chi_3_breakdown()
    step_2_sweep_three_characters()
    step_3_certificate_summaries()
    step_4_honest_scope()
