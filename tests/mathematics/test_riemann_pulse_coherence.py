"""Tests for the pulse-phase / coherence attack-surface layer (re-founded).

The nodal pulse makes S(T) -- the RH-content oscillation the eliminated
combinatorial (S_n-invariant) spectrum was blind to -- directly accessible;
the critical line is the exact coherence axis.
"""

from __future__ import annotations

import builtins
import math
import subprocess
import sys
import textwrap

import mpmath as mp
import pytest

from tnfr.riemann import (
    KNOWN_RIEMANN_ZEROS,
    argument_fluctuation,
    coherence_defect,
    prime_side_fluctuation,
    verify_pulse_coherence,
    zero_count,
)


def _true_s(t: float) -> float:
    with mp.workdps(25):
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
    errs = [
        abs(prime_side_fluctuation(t, npr, 6) - _true_s(t)) for npr in (10, 40, 160)
    ]
    assert min(errs) > 0.2  # never gets close; no convergence


def test_verify_pulse_coherence_certificate():
    with mp.workdps(60):
        cert = verify_pulse_coherence()
        assert mp.mp.dps == 60
    assert cert.zero_count_matches
    assert cert.coherence_axis_is_minimal
    assert cert.max_abs_s_error < 0.05
    assert "PASS" in cert.summary()


def test_verification_restores_precision_after_failure(monkeypatch):
    from tnfr.riemann import pulse_coherence

    def fail(_):
        assert mp.mp.dps == 25
        raise RuntimeError("injected observation failure")

    monkeypatch.setattr(pulse_coherence, "argument_fluctuation", fail)
    with mp.workdps(60):
        with pytest.raises(RuntimeError, match="injected observation failure"):
            verify_pulse_coherence()
        assert mp.mp.dps == 60


@pytest.mark.parametrize(
    "tolerance, passed", [(0.2, True), (0.125, True), (0.1, False), (0.0, False)]
)
def test_requested_tolerance_controls_summary(monkeypatch, tolerance, passed):
    from tnfr.riemann import pulse_coherence

    # Independent oracle phase is zero while the observed approximation is 1/8.
    # Count and axis checks stay successful, isolating the formerly ignored cut.
    monkeypatch.setattr(mp, "zeta", lambda _: mp.mpc(1))
    monkeypatch.setattr(pulse_coherence, "argument_fluctuation", lambda _: 0.125)
    monkeypatch.setattr(pulse_coherence, "zero_count", lambda _: 1.0)
    monkeypatch.setattr(
        pulse_coherence, "coherence_defect", lambda _, sigma: abs(sigma - 0.5)
    )
    certificate = verify_pulse_coherence((17.0,), s_tol=tolerance)

    assert certificate.max_abs_s_error == 0.125
    assert certificate.zero_count_matches and certificate.coherence_axis_is_minimal
    assert certificate.s_tolerance == tolerance
    assert certificate.s_tolerance_satisfied is passed
    assert ("[PASS]" in certificate.summary()) is passed


@pytest.mark.parametrize("tolerance", [-1.0, math.nan, math.inf, True, False, None])
def test_verification_rejects_invalid_tolerances(tolerance):
    with pytest.raises(ValueError, match="s_tol must be finite"):
        verify_pulse_coherence(s_tol=tolerance)


def test_missing_oracle_cannot_certify_self_comparison(monkeypatch):
    from tnfr.riemann import pulse_coherence

    original_import = builtins.__import__

    def without_mpmath(name, *args, **kwargs):
        if name == "mpmath":
            raise ImportError("oracle unavailable")
        return original_import(name, *args, **kwargs)

    def unsupported_observation(_):
        raise AssertionError("must not substitute the observation for its oracle")

    monkeypatch.setattr(builtins, "__import__", without_mpmath)
    monkeypatch.setattr(
        pulse_coherence, "argument_fluctuation", unsupported_observation
    )
    certificate = verify_pulse_coherence((17.0,))

    assert certificate.max_abs_s_error is None
    assert not certificate.independent_oracle_available
    assert not certificate.s_tolerance_satisfied
    assert "[PARTIAL]" in certificate.summary()
    assert "unavailable" in certificate.summary()


def test_empty_comparison_cannot_pass():
    certificate = verify_pulse_coherence(())

    assert not certificate.s_tolerance_satisfied
    assert "[PARTIAL]" in certificate.summary()


def test_zeta_import_and_fixed_precision_leave_caller_context_unchanged():
    # A fresh interpreter catches import-time precision mutation and the former
    # invalid mp.mp attribute access independently of pytest's import order.
    script = textwrap.dedent(
        """
        import mpmath as mp
        mp.mp.dps = 67
        from tnfr.mathematics import zeta

        assert mp.mp.dps == 67
        assert zeta.mp.dps == 25
        observed = zeta.zeta_function(2)
        assert abs(mp.mpf(observed) - mp.zeta(2)) < mp.mpf('1e-24')
        assert mp.mp.dps == 67
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_nu_f_log_additivity_is_the_euler_product_seam():
    # nu_f(p*q) = nu_f(p)+nu_f(q): the additivity that makes the pulse an Euler
    # product; the seam the pulse phase carries (Fix(S_n)^perp).
    assert math.isclose(math.log(35), math.log(5) + math.log(7))
