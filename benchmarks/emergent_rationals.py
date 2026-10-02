"""Supplied graph-spectral encodings of integers and rational arithmetic.

Bipartite adjacency spectra have sign symmetry. Selected integral graphs
therefore provide signed integer eigenvalues. Complete bipartite K_{a,b} has
Laplacian spectrum {0, a^(b-1), b^(a-1), a+b}; ratios of selected nonzero
integer modes encode rational numbers. For every positive p/q, taking
(a,b)=(2p,2q) ensures both eigenvalues have nonzero multiplicity, including
when p or q is one. Negative signs and zero require the separate arithmetic
construction; positive Laplacian ratios alone do not give all of Q.

The finite arithmetic controls use explicit integer inputs, outer_sum,
outer_prod and Python Fraction. The field-of-fractions construction Frac(Z)=Q
is supplied mathematics, not an autonomous physical production of arithmetic.
Stern-Brocot traversal separately applies a prescribed mediant rule.

A separate two-oscillator sine-coupling ODE is integrated by explicit Euler
as a comparison. It tests a finite-time ratio near a 1:1 lock for selected
frequencies and coupling; it does not generate Stern-Brocot mediants or all
rational locks. Grammar U3 is a circular compatibility gate and does not
derive this phase law, its frequency inputs or physical realization.

The helpers remain for graph-symmetry consumers. Scope owners are
``theory/TNFR_NUMBER_THEORY.md`` and ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md``.
No TNFR operator trajectory, physical measurement or RH result follows.
Run ``python benchmarks/emergent_rationals.py`` for the configured controls.
"""

from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from composition_arithmetic import (  # noqa: E402
    adj_spectrum,
    lap_spectrum,
    multiset_close,
    outer_prod,
    outer_sum,
)

TOL = 1e-9


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def integer_spectrum(spec, tol=1e-6):
    """Round a spectrum to ints if it is integral; else raise."""
    rounded = np.round(spec)
    if not np.allclose(spec, rounded, atol=tol):
        raise ValueError("spectrum is not integral")
    return rounded.astype(int)


def is_pm_symmetric(spec, tol=1e-8):
    """True if the multiset {spec} equals {-spec} (chiral / bipartite symmetry)."""
    return multiset_close(spec, -np.asarray(spec, dtype=float), tol)


def kab_laplacian_eigenvalues(a: int, b: int) -> set[int]:
    """Distinct Laplacian eigenvalues of the complete bipartite graph K_{a,b}.

    Spectrum: {0, a (mult b-1), b (mult a-1), a+b} for positive a,b.
    This numerical helper rounds the nonzero eigenvalues to integers under
    its configured tolerance; it does not observe physical modes.
    """
    G = nx.complete_bipartite_graph(a, b)
    spec = integer_spectrum(lap_spectrum(G))
    return set(int(v) for v in spec if v != 0)


def stern_brocot_path(target: Fraction, max_steps: int = 10000):
    """Navigate the Stern-Brocot tree to `target` by mediants from 0/1 and 1/0.

    Returns (steps, reached, mediants) where `mediants` is the list of mediant
    fractions visited. Each mediant explicitly applies (a+c)/(b+d) to the two
    bounding fractions; no phase dynamics generates that rule here.
    """
    left = (0, 1)  # 0/1
    right = (1, 0)  # 1/0 (infinity sentinel)
    mediants: list[Fraction] = []
    for step in range(1, max_steps + 1):
        med = (left[0] + right[0], left[1] + right[1])  # mediant
        med_frac = Fraction(med[0], med[1])
        mediants.append(med_frac)
        if med_frac == target:
            return step, True, mediants
        if target < med_frac:
            right = med
        else:
            left = med
    return max_steps, False, mediants


def kuramoto_two_rotation_number(omega1, omega2, K, steps=40000, dt=0.005):
    """Finite-horizon mean-rate ratio for a supplied sine-coupled phase law.

    Explicit Euler integrates dtheta_i/dt = omega_i + K sin(theta_j-theta_i)
    from zero phases. For K>0 and |omega1-omega2|<2K, the continuous phase-
    difference equation admits a stable locked branch. If reached and its
    common mean rate is nonzero, the long-time ratio is one. The finite return
    does not certify that limit, the boundary case, or the engine U3 policy.
    """
    t1 = 0.0
    t2 = 0.0
    for _ in range(steps):
        d1 = omega1 + K * math.sin(t2 - t1)
        d2 = omega2 + K * math.sin(t1 - t2)
        t1 += dt * d1
        t2 += dt * d2
    f1 = t1 / (steps * dt)
    f2 = t2 / (steps * dt)
    return f1 / f2


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_additive_inverse_from_bipartite_symmetry():
    print("=" * 78)
    print("(1) Bipartite adjacency spectra: paired eigenvalue signs")
    print("=" * 78)
    bipartite = [
        ("C6", nx.cycle_graph(6)),
        ("K_{3,3}", nx.complete_bipartite_graph(3, 3)),
        ("Q3 (hypercube)", nx.hypercube_graph(3)),
        ("P4", nx.path_graph(4)),
    ]
    non_bipartite = [("C5", nx.cycle_graph(5)), ("K4", nx.complete_graph(4))]

    all_sym = True
    for name, G in bipartite:
        spec = adj_spectrum(G)
        sym = is_pm_symmetric(spec)
        all_sym &= sym
        print(
            f"  {name:<16} spec(A) +/- symmetric? {sym}    "
            f"spec = {np.round(spec, 3)}"
        )
    none_sym = True
    for name, G in non_bipartite:
        spec = adj_spectrum(G)
        sym = is_pm_symmetric(spec)
        none_sym &= not sym
        print(
            f"  {name:<16} spec(A) +/- symmetric? {sym}  (non-bipartite, " "contrast)"
        )

    # Selected integral bipartite graph gives a finite set of signed integers.
    q3 = integer_spectrum(adj_spectrum(nx.hypercube_graph(3)))
    signed = sorted(set(int(v) for v in q3))
    print(f"  Q3 gives the selected signed integer eigenvalues: {signed}")

    ok = all_sym and none_sym and (-min(signed) == max(signed))
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- -n is forced by the "
        "bipartite adjacency symmetry on the selected graphs"
    )
    return ok


def test_division_from_integral_eigenvalue_ratios():
    print()
    print("=" * 78)
    print("(2) Explicit ratios of selected integral Laplacian eigenvalues")
    print("=" * 78)
    targets = [Fraction(3, 2), Fraction(5, 3), Fraction(7, 4), Fraction(5, 2)]
    all_ok = True
    for r in targets:
        a, b = r.numerator, r.denominator
        eig = kab_laplacian_eigenvalues(a, b)
        # K_{b,a} has Laplacian eigenvalues a (mult b-1) and b (mult a-1)
        have = a in eig and b in eig
        ratio = Fraction(a, b)
        ok = have and ratio == r
        all_ok &= ok
        print(
            f"  {r}  realised by K_{{{a},{b}}}: eigenvalues {{a,b}}={{{a},{b}}} "
            f"present? {have};  ratio = {ratio}  matches? {ratio == r}"
        )
    print("  Any positive p/q can be encoded using K_{2p,2q}; division is supplied.")
    print(
        f"  VERDICT: {'PASS' if all_ok else 'FAIL'} -- division = ratio of "
        "two selected nonzero integer eigenvalues"
    )
    return all_ok


def test_field_closure_is_Q():
    print()
    print("=" * 78)
    print("(3) Supplied rational arithmetic: finite +,-,x,/ controls")
    print("=" * 78)
    # Supplied integers also occurring in the indicated Laplacian spectra.
    print("  integer spectra K_{2,3} -> {2,3,5}, K_{2,4} -> {2,4,6}")
    a, b, c, d = 2, 3, 4, 6
    r1 = Fraction(a, b)  # 2/3
    r2 = Fraction(c, d)  # 4/6 = 2/3
    # use two genuinely different ratios
    r1 = Fraction(2, 3)
    r2 = Fraction(5, 2)
    a, b = r1.numerator, r1.denominator
    c, d = r2.numerator, r2.denominator
    print(
        f"  r1 = {r1} (= {a}/{b}),  r2 = {r2} (= {c}/{d})  "
        "[supplied integer values in the selected spectra]"
    )

    checks = []

    # x : numerator ac and denominator bd via the TENSOR product (spectra multiply)
    num_mul = outer_prod([a], [c])[0]  # ac
    den_mul = outer_prod([b], [d])[0]  # bd
    prod = Fraction(int(round(num_mul)), int(round(den_mul)))
    checks.append(("x", prod, r1 * r2))

    # + : ad, bc via x ; ad+bc via [] (outer_sum) ; bd via x
    ad = outer_prod([a], [d])[0]
    bc = outer_prod([b], [c])[0]
    num_add = outer_sum([ad], [bc])[0]  # ad + bc
    den_add = outer_prod([b], [d])[0]  # bd
    summ = Fraction(int(round(num_add)), int(round(den_add)))
    checks.append(("+", summ, r1 + r2))

    # - : ad - bc via the bipartite additive inverse (outer_sum with -bc)
    num_sub = outer_sum([ad], [-bc])[0]  # ad - bc
    diff = Fraction(int(round(num_sub)), int(round(den_add)))
    checks.append(("-", diff, r1 - r2))

    # / : (a/b)/(c/d) = ad/bc, with nonzero divisor c/d.
    quot = Fraction(int(round(ad)), int(round(bc)))
    checks.append(("/", quot, r1 / r2))

    all_ok = True
    for op, got, expect in checks:
        ok = got == expect
        all_ok &= ok
        print(
            f"  r1 {op} r2 = {got}   (exact {expect})   "
            f"agrees with Fraction arithmetic? {ok}"
        )
    print("  numerators/denominators all built with outer_sum / outer_prod")
    print("  Frac(Z)=Q is the supplied algebraic construction; this tests examples.")
    print(
        f"  VERDICT: {'PASS' if all_ok else 'FAIL'} -- four arithmetic identities "
        "on the selected rational inputs"
    )
    return all_ok


def test_resonance_generates_Q():
    print()
    print("=" * 78)
    print("(4) Separate supplied models: sine-coupled pair and Stern-Brocot traversal")
    print("=" * 78)
    # 4a) two-oscillator Kuramoto 1:1 lock -> rational rotation number 1
    inside = kuramoto_two_rotation_number(1.0, 1.3, 0.5)  # |dw|=0.3 <= 2K=1.0
    outside = kuramoto_two_rotation_number(1.0, 3.0, 0.2)  # |dw|=2.0 >  2K=0.4
    locked = abs(inside - 1.0) < 1e-2
    unlocked = abs(outside - 1.0) > 5e-2
    print(
        f"  Kuramoto 1:1 inside Arnold tongue:  rotation number = {inside:.4f} "
        f"-> finite ratio within tolerance of 1? {locked}"
    )
    print(
        f"  Kuramoto outside tongue:            rotation number = {outside:.4f} "
        f"-> not 1 (drifting)?       {unlocked}"
    )

    # 4b) prescribed Stern-Brocot mediants, independent of the phase integration.
    targets = [Fraction(3, 2), Fraction(5, 3), Fraction(22, 7), Fraction(1, 4)]
    gen_ok = True
    for r in targets:
        steps, reached, _ = stern_brocot_path(r)
        gen_ok &= reached
        print(
            f"  Stern-Brocot reaches {str(r):>5} in {steps:>3} mediants "
            f"(prescribed arithmetic rule)?  {reached}"
        )
    # One exact mediant example; no oscillator state enters this calculation.
    med_demo = Fraction(1 + 1, 2 + 3)  # mediant of 1/2 and 1/3 = 2/5
    med_ok = med_demo == Fraction(2, 5)
    print(
        f"  mediant(1/2, 1/3) = (1+1)/(2+3) = {med_demo}  "
        f"(exact arithmetic identity)?  {med_ok}"
    )

    ok = locked and unlocked and gen_ok and med_ok
    print(
        f"  VERDICT: {'PASS' if ok else 'FAIL'} -- finite oscillator-ratio and "
        "independent mediant controls; no dynamical derivation of Q"
    )
    return ok


def main():
    print(__doc__)
    results = [
        (
            "(1) bipartite adjacency sign symmetry",
            test_additive_inverse_from_bipartite_symmetry(),
        ),
        (
            "(2) selected integral eigenvalue ratios",
            test_division_from_integral_eigenvalue_ratios(),
        ),
        ("(3) finite rational arithmetic identities", test_field_closure_is_Q()),
        (
            "(4) independent oscillator and mediant controls",
            test_resonance_generates_Q(),
        ),
    ]
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for name, ok in results:
        print(f"  {name:<50}: {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    print()
    print(f"  OVERALL: {'ALL PASS' if overall else 'SOME FAIL'}")
    print()
    print("  Reading: selected graph spectra encode integer values; explicit")
    print("  arithmetic forms their ratios. The independent sine-coupled ODE")
    print("  does not generate the mediant rule or derive all rational locks.")
    print("  No physical emergence, engine U3 phase law or RH result is tested.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
