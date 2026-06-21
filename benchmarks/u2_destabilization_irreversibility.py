"""U2 destabilization-irreversibility test -- validated instrument + pre-registered
falsification record.

WHAT THIS IS
------------
A TNFR-MOTIVATED empirical test and its honest outcome. TNFR grammar rule U2
("a destabilizer requires a stabilizer"), derived from the convergence requirement
of the nodal equation

    d EPI/dt = vf * dNFR,   coherence persists  <=>  integral(vf*dNFR) dt < inf,

predicts that in a persistently coherent signal, *sustained destabilization*
(a run of consecutive drops in a coherence proxy c(t)) should be SUPPRESSED
relative to a baseline that preserves all linear structure. This module tests
whether that signature is a universal EXTERNAL empirical regularity.

IMPORTANT SCOPE. This is TNFR-*motivated* (the statistic derives from U2) but NOT
TNFR-*mechanistic*: earthquakes, markets, rivers, heart-rate are NOT TNFR networks;
c(t) is a proxy. U2 remains canonically true INSIDE TNFR regardless of the outcome
here (it is a theorem of the nodal equation). This module only asks whether U2's
qualitative signature generalizes outward.

STATISTIC (pre-registered): overload_D(L=3, W=4) = number of length-4 windows of the
increment-sign sequence of c(t) containing >= 3 drops. Uses only the SIGN of
increments, hence invariant to any monotone rescaling of c (only the DIRECTION of
coherence matters).

NULL (pre-registered): IAAFT surrogates (preserve the power spectrum AND the marginal
distribution => preserve all linear autocorrelation; destroy only nonlinear /
higher-order / time-irreversible structure). This is exactly what standard linear
tools (ARMA/GARCH/spectral) already capture, so a significant result = structure those
tools do NOT predict. Two-sided: p_lower => SUPPRESSION (U2-like), p_upper =>
AMPLIFICATION (anti-U2).

INSTRUMENT VALIDATION (reproducible below, no network needed)
------------------------------------------------------------
1. CALIBRATION: false-positive rate ~ alpha on linear / time-reversible processes
   (iid, AR(1), AR(2), ARMA) -- the test must NOT fire on these.
2. POWER: an injected, tunable U2 constraint (rho) is detected with rising power;
   rho=0 -> ~alpha, rho>=0.25 -> ~1.0. This proves a U2 constraint leaves a signature
   BEYOND linear autocorrelation (it is not linearly absorbable) and the test sees it.

PRE-REGISTERED REAL-DATA RESULT (recorded; needs network + pyedflib/wfdb/yfinance)
---------------------------------------------------------------------------------
Six domains, directions fixed from domain physics BEFORE running, IAAFT null, 500
surrogates, Bonferroni alpha = 0.05/6 = 0.0083:

    domain         field        pred           observed        z       match
    EQ-magnitude   geophysics   SUPPRESSION    SUPPRESSION    -3.79    yes
    RR-interval    physiology   SUPPRESSION    AMPLIFICATION  +2.69    no
    EQ-interevent  geophysics   AMPLIFICATION  none (ns)      -1.70    --
    FIN-equity     markets      AMPLIFICATION  AMPLIFICATION  +2.71    yes
    FIN-alt        markets      AMPLIFICATION  SUPPRESSION    -4.16    no
    STREAMFLOW     hydrology    AMPLIFICATION  AMPLIFICATION  +23.9    yes

VERDICT: the structural sign hypothesis is FALSIFIED. 3/6 ~ chance, and -- decisively --
FIN-equity (stock indices) AMPLIFIES while FIN-alt (crypto/FX/commodity) SUPPRESSES under
the *same* operationalization (c = -|log-return|): markets contradict themselves, so the
sign is NOT a clean dynamical-character property (it is idiosyncratic / data-microstructure
sensitive, e.g. equity weekend/overnight gaps vs 24/7 crypto). EQ-interevent failed because
Omori clustering lives in the LINEAR autocorrelation, which IAAFT preserves.

CONCLUSION: U2's suppression is NOT a universal external regularity; there is no
demonstrated TNFR-specific cross-domain empirical law here. What survives is (i) a
calibrated + powered nonlinearity / time-irreversibility instrument (NOT TNFR-exclusive --
it is a standard surrogate-data test) and (ii) a clean pre-registered negative that
prevents overclaiming. EQ-magnitude suppression and streamflow amplification are genuine
but are known seismology / hydrology, not TNFR.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- apparatus
def iaaft(x, rng, iters=40):
    """Iterative amplitude-adjusted Fourier transform surrogate: preserves power
    spectrum (all linear autocorrelation) AND marginal distribution."""
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(np.fft.rfft(x))
    srt = np.sort(x)
    s = rng.permutation(x)
    for _ in range(iters):
        S = np.fft.rfft(s)
        s = np.fft.irfft(amp * np.exp(1j * np.angle(S)), n=n)
        s = srt[np.argsort(np.argsort(s))]
    return s


def overload_drop(c, L=3, W=4):
    """#length-W windows of the increment-sign sequence with >= L drops (c decreasing)."""
    d = (np.diff(np.asarray(c, float)) < 0).astype(int)
    m = len(d)
    if m < W:
        return 0
    cs = np.cumsum(d)
    wsum = cs[W - 1:] - np.concatenate([[0], cs[:m - W]])
    return int(np.sum(wsum >= L))


def iaaft_test(signals, L=3, W=4, n_sur=500, seed=0, iters=40):
    """Two-sided IAAFT surrogate test pooled over a list of signals."""
    rng = np.random.default_rng(seed)
    real = sum(overload_drop(c, L, W) for c in signals)
    null = np.empty(n_sur)
    for i in range(n_sur):
        null[i] = sum(overload_drop(iaaft(c, rng, iters), L, W) for c in signals)
    mu, sd = float(null.mean()), float(null.std())
    return {"real": int(real), "null_mean": round(mu, 1),
            "z": round((real - mu) / (sd + 1e-12), 2),
            "p_lower": round(float(np.mean(null <= real)), 4),
            "p_upper": round(float(np.mean(null >= real)), 4)}


# --------------------------------------------------------------------- generators
def _ar(rng, n, coeffs, sd):
    x = rng.standard_normal(n)
    p = len(coeffs)
    for t in range(p, n):
        x[t] = sum(c * x[t - 1 - j] for j, c in enumerate(coeffs)) + sd * rng.standard_normal()
    return x


def gen(kind, n, rng):
    if kind == "iid":
        return rng.standard_normal(n)
    if kind == "ar1_0.5":
        return _ar(rng, n, [0.5], np.sqrt(1 - 0.25))
    if kind == "ar1_0.9":
        return _ar(rng, n, [0.9], np.sqrt(1 - 0.81))
    if kind == "ar2":
        return _ar(rng, n, [0.6, 0.3], 0.5)
    if kind == "arma":
        e = rng.standard_normal(n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.7 * x[t - 1] + e[t] + 0.4 * e[t - 1]
        return x
    raise ValueError(kind)


def gen_u2(n, rng, phi=0.9, rho=0.5, L=3):
    """AR(1) base with an injected U2 constraint: a run of L-1 consecutive drops is,
    with probability rho, prevented from extending. Nonlinear and time-irreversible."""
    x = np.empty(n)
    x[0] = 0.0
    sd = np.sqrt(1 - phi * phi)
    drops = 0
    for t in range(1, n):
        nx = phi * x[t - 1] + sd * rng.standard_normal()
        if nx < x[t - 1]:
            if drops >= L - 1 and rng.random() < rho:
                nx = x[t - 1] + (x[t - 1] - nx)
                drops = 0
            else:
                drops += 1
        else:
            drops = 0
        x[t] = nx
    return x


def power(rho, n=4000, reps=20, n_sur=150, phi=0.9, alpha=0.05, n_sig=6, iters=18):
    hits = 0
    for r in range(reps):
        rng = np.random.default_rng(1000 + r)
        sigs = [gen_u2(n, rng, phi=phi, rho=rho) for _ in range(n_sig)]
        hits += iaaft_test(sigs, 3, 4, n_sur, seed=7000 + r, iters=iters)["p_lower"] < alpha
    return hits / reps


def main():
    print("U2 irreversibility instrument -- self-contained validation (synthetic).\n")
    print("(1) CALIBRATION (false-positive ~ alpha; p_lower in [.025,.975] = ok)")
    for kind in ("iid", "ar1_0.5", "ar1_0.9", "ar2", "arma"):
        rng = np.random.default_rng(0)
        sigs = [gen(kind, 3000, rng) for _ in range(6)]
        r = iaaft_test(sigs, 3, 4, 250, seed=0, iters=20)
        ok = "ok" if 0.025 < r["p_lower"] < 0.975 else "FIRES"
        print(f"    {kind:9} z={r['z']:>6} p_lower={r['p_lower']:<6} {ok}")
    print("\n(2) POWER to detect an injected U2 constraint (rho):")
    for rho in (0.0, 0.5, 1.0):
        print(f"    rho={rho:>3}  power(p_lower<0.05) = {power(rho):.2f}")
    print("\nSee module docstring for the pre-registered real-data result (FALSIFIED).")


if __name__ == "__main__":
    main()
