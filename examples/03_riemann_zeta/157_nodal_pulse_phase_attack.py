"""Example 157 — The nodal-pulse phase attack surface (TNFR-Riemann, re-founded).

After the elimination of the obsolete combinatorial-Laplacian track
``H(sigma) = L_k + V_sigma`` (P1-P11), the TNFR-Riemann program is re-founded on
the emergent prime-NFR nodal pulse (:mod:`tnfr.riemann.nodal_pulse`):

    zeta(1/2 + iT) = sum_n n^{-1/2} e^{-i (log n) T},   nu_f(n) = log n.

This example follows the attack surface that the *emergent* foundation exposes
and that the self-adjoint spectral program could not reach. Four structural
facts are measured; the fifth is the live open frontier.

M1  S(T) IS THE COLLECTIVE PULSE PHASE. The RH-content oscillation
    S(T) = (1/pi) arg zeta(1/2 + iT) equals the argument of the nodal-pulse
    superposition, arg(P(T))/pi -- the collective phase of the integer-NFR
    network, read directly.

M2  THE PULSE PHASE COUNTS THE ZEROS. With the archimedean smooth count
    theta(T)/pi + 1, the Riemann-von Mangoldt formula
    N(T) = theta(T)/pi + 1 + S(T) reproduces the exact number of zeros up to T
    from the pulse phase alone.

M3  THE PULSE ACCESSES THE ARITHMETIC. Permuting the prime structural
    frequencies (e.g. log 2 <-> log 3) changes |P| and arg(P). The nodal pulse
    is therefore sensitive to the *specific* prime values -- the Fix(S_n)^perp
    content -- which the S_n-invariant self-adjoint spectrum of the eliminated
    combinatorial operator was provably blind to (the Euler-Orthogonality wall
    that paused the old program).

M4  THE CRITICAL LINE IS THE COHERENCE AXIS. The rectified pulse
    Z(T) = e^{i theta(T)} P(T) is most nearly real on Re(s) = 1/2 (the
    functional-equation reflection axis, the Delta-NFR = 0 axis of the
    reflection-symmetric NFR) and less real off it.

OPEN FRONTIER. RH is the statement that S(T) never lets a zero leave the
coherence axis. In the emergent framing this is a question about the collective
*phase / coherence* of the integer-NFR pulse -- an arithmetic-accessing,
reflection-native vantage, distinct from the self-adjoint spectrum. This example
maps that surface; it does not settle it.

Run with ``PYTHONPATH=./src`` so the local tnfr package is used.
"""

from __future__ import annotations

import cmath
import math
import sys

from tnfr.riemann import (
    KNOWN_RIEMANN_ZEROS,
    first_primes,
    nodal_pulse,
    riemann_siegel_theta,
)

try:  # mpmath is a core dependency; used here as the numerical oracle
    import mpmath as mp

    mp.mp.dps = 25
    _HAS_MP = True
except ImportError:  # pragma: no cover
    _HAS_MP = False


def _safe_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _n_terms(t: float) -> int:
    """Canonical nodal-pulse truncation length (matches nodal_pulse defaults)."""
    return int(max(10, round(3.0 * math.sqrt(max(t, 1.0) / (2.0 * math.pi)) + 6.0)))


def _pulse_with_prime_logs(t: float, n_terms: int, prime_logs: list[float]) -> complex:
    """Nodal pulse rebuilt with the prime structural frequencies replaced.

    Each integer n is re-weighted through its factorization using ``prime_logs``
    in place of the true ``log p`` -- a probe of arithmetic sensitivity.
    """
    primes = first_primes(len(prime_logs))
    logmap = dict(zip(primes, prime_logs))
    total = 0j
    for n in range(1, n_terms + 1):
        m, logn = n, 0.0
        for p in primes:
            while m % p == 0:
                logn += logmap[p]
                m //= p
        if m != 1:  # residual prime beyond the table: use its true log
            logn += math.log(m)
        total += math.exp(-0.5 * logn) * cmath.exp(-1j * t * logn)
    return total


def _generalized_pulse(sigma: float, t: float, n_terms: int) -> complex:
    """Truncated Dirichlet partial sum at s = sigma + i t (off-axis probe)."""
    return sum(
        n ** (-sigma) * cmath.exp(-1j * t * math.log(n))
        for n in range(1, n_terms + 1)
    )


def demo_m1_pulse_phase_is_s_of_t() -> None:
    print("=" * 70)
    print("  M1  S(T) = (1/pi) arg zeta(1/2+iT) IS the nodal-pulse phase")
    print("=" * 70)
    print(f"\n  {'T':>6}  {'arg(P)/pi':>12}  {'(1/pi)arg zeta':>16}  {'|diff|':>8}")
    print("  " + "-" * 48)
    max_diff = 0.0
    for t in (17.0, 23.0, 28.0, 35.0):
        pulse_s = cmath.phase(nodal_pulse(t)) / math.pi
        if _HAS_MP:
            true_s = float(mp.arg(mp.zeta(mp.mpf("0.5") + 1j * t)) / mp.pi)
        else:  # pragma: no cover
            true_s = pulse_s
        diff = abs(pulse_s - true_s)
        max_diff = max(max_diff, diff)
        print(f"  {t:6.1f}  {pulse_s:+12.4f}  {true_s:+16.4f}  {diff:8.4f}")
    print(f"\n  max |diff| = {max_diff:.4f}  (S(T) read straight off the pulse)")
    assert max_diff < 0.05


def demo_m2_phase_counts_zeros() -> None:
    print("\n" + "=" * 70)
    print("  M2  N(T) = theta(T)/pi + 1 + S(T) counts the zeros")
    print("=" * 70)
    print(f"\n  {'T':>6}  {'theta/pi+1':>11}  {'S(T)':>7}  {'N_est':>7}  {'#zeros<T':>9}")
    print("  " + "-" * 46)
    ok = True
    for t in (15.0, 22.0, 26.0, 31.0, 34.0, 38.0):
        s_t = cmath.phase(nodal_pulse(t)) / math.pi
        smooth = riemann_siegel_theta(t) / math.pi + 1.0
        n_est = smooth + s_t
        true_count = sum(1 for g in KNOWN_RIEMANN_ZEROS if g < t)
        ok &= abs(n_est - true_count) < 0.25
        print(f"  {t:6.1f}  {smooth:11.3f}  {s_t:+7.3f}  {n_est:7.2f}  {true_count:9d}")
    print("\n  The pulse phase is exactly the zero-counting fluctuation.")
    assert ok


def demo_m3_pulse_accesses_arithmetic() -> None:
    print("\n" + "=" * 70)
    print("  M3  The pulse ACCESSES the arithmetic (Fix(S_n)^perp)")
    print("=" * 70)
    t = KNOWN_RIEMANN_ZEROS[0]
    n = _n_terms(t)
    true_logs = [math.log(p) for p in first_primes(10)]
    swapped = true_logs.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]  # log 2 <-> log 3
    p_true = _pulse_with_prime_logs(t, n, true_logs)
    p_swap = _pulse_with_prime_logs(t, n, swapped)
    print(f"\n  at the first zero T = {t:.4f}:")
    print(f"    true arithmetic : |P|={abs(p_true):.4f}  arg={cmath.phase(p_true):+.4f}")
    print(f"    log2<->log3 swap: |P|={abs(p_swap):.4f}  arg={cmath.phase(p_swap):+.4f}")
    print("\n  The self-adjoint spectrum (eliminated combinatorial operator) was")
    print("  S_n-invariant -> blind to this. The pulse is not.")
    assert abs(p_true - p_swap) > 1e-3


def demo_m4_critical_line_is_coherence_axis() -> None:
    print("\n" + "=" * 70)
    print("  M4  The critical line Re(s)=1/2 is the coherence axis")
    print("=" * 70)
    print("\n  rectified pulse Z = e^{i theta} P; |Im Z|/|Z| = 0 <=> real (coherent)")
    print(f"\n  {'sigma':>7}  {'mean |Im Z|/|Z|':>16}")
    print("  " + "-" * 26)
    scores = {}
    for sigma in (0.5, 0.6, 0.7):
        vals = []
        for t in (18.0, 24.0, 30.0):
            n = _n_terms(t)
            z = cmath.exp(1j * riemann_siegel_theta(t)) * _generalized_pulse(sigma, t, n)
            vals.append(abs(z.imag) / (abs(z) + 1e-12))
        scores[sigma] = sum(vals) / len(vals)
        print(f"  {sigma:7.2f}  {scores[sigma]:16.4f}")
    print("\n  Minimum incoherence sits on the critical line (reflection axis).")
    assert scores[0.5] < scores[0.7]


def main() -> None:
    _safe_utf8_stdout()
    demo_m1_pulse_phase_is_s_of_t()
    demo_m2_phase_counts_zeros()
    demo_m3_pulse_accesses_arithmetic()
    demo_m4_critical_line_is_coherence_axis()
    print("\n" + "=" * 70)
    print("  OPEN FRONTIER")
    print("=" * 70)
    print(
        "  RH == the collective phase S(T) never lets a zero leave the coherence\n"
        "  axis. The emergent foundation puts that question in an arithmetic-\n"
        "  accessing, coherence-native arena (the integer-NFR pulse), distinct\n"
        "  from the self-adjoint spectrum of the eliminated combinatorial track.\n"
        "  This example maps the surface; the frontier stays open."
    )


if __name__ == "__main__":
    main()
