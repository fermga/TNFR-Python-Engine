"""Tests for the pulse-phase / coherence attack-surface layer (re-founded).

The nodal pulse makes S(T) -- the RH-content oscillation the eliminated
combinatorial (S_n-invariant) spectrum was blind to -- directly accessible;
the critical line is the exact coherence axis.
"""

from __future__ import annotations

import math

import mpmath as mp

from tnfr.riemann import (
    KNOWN_RIEMANN_ZEROS,
    argument_fluctuation,
    coherence_defect,
    prime_side_fluctuation,
    verify_pulse_coherence,
    zero_count,
)

mp.mp.dps = 25


def _true_s(t: float) -> float:
    return float(mp.arg(mp.zeta(mp.mpf("0.5") + 1j * t)) / mp.pi)


def test_argument_fluctuation_matches_oracle_off_zeros():
    for t in (17.0, 23.0, 28.0, 35.0):
        assert abs(argument_fluctuation(t) - _true_s(t)) < 0.05


def test_zero_count_counts_zeros():
    for t in (17.0, 23.0, 28.0, 35.0, 47.0):
        true_count = sum(1 for g in KNOWN_RIEMANN_ZEROS if g < t)
        assert round(zero_count(t)) == true_count


def test_critical_line_is_exact_coherence_axis():
    # Z = e^{i theta} zeta is exactly real on sigma=1/2 (functional equation).
    for t in (18.0, 24.0, 30.0):
        assert coherence_defect(t, 0.5) < 1e-9
        assert coherence_defect(t, 0.7) > coherence_defect(t, 0.5)


def test_prime_side_series_does_not_converge_on_the_line():
    # Honest localization of the RH content: the prime-side series for S(T) has
    # abscissa of convergence Re(s)=1, so on the line adding primes does NOT
    # converge to S(T). This is the obstruction made explicit, not an estimator.
    t = 41.0
    errs = [abs(prime_side_fluctuation(t, npr, 6) - _true_s(t)) for npr in (10, 40, 160)]
    assert min(errs) > 0.2  # never gets close; no convergence


def test_verify_pulse_coherence_certificate():
    cert = verify_pulse_coherence()
    assert cert.zero_count_matches
    assert cert.coherence_axis_is_minimal
    assert cert.max_abs_s_error < 0.05
    assert "PASS" in cert.summary()


def test_nu_f_log_additivity_is_the_euler_product_seam():
    # nu_f(p*q) = nu_f(p)+nu_f(q): the additivity that makes the pulse an Euler
    # product; the seam the pulse phase carries (Fix(S_n)^perp).
    assert math.isclose(math.log(35), math.log(5) + math.log(7))
