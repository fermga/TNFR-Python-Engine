"""Finite winding-sector energy and relaxation study.

The module prepares phase rings with target windings ``W = 0, ..., 5`` and
measures four restricted statements:

* wrapped circulation returns an integer on those declared closed loops;
* the XY energy and the implemented structural excitation-energy read-out are
  approximately quadratic in ``W`` over this finite family;
* a separately defined wrapped-consensus flow preserves ``W`` on the tested
  trajectories and relaxes toward a uniform-gradient representative;
* the explicitly defined proxy ``1/(1+mean|grad phi|)`` decreases with ``|W|``.

The exact integrality of a single-valued loop is a topological fact. The energy
and relaxation conclusions are conditional on the prepared ring, energy
functional, time step and finite winding range below. The inequality
``E(2) > 2 E(1)`` gives an energetic comparison; this script does not simulate
fission, prove stability or attraction, show that TNFR dynamics creates a
winding, or identify any winding sector with a physical particle species.

Run:
    python benchmarks/emergent_particle_catalog.py

Theoretical anchor: theory/EMERGENT_ONTOLOGY.md Sec.9.1 (open physical map) and
emergent_particles.py (declared-cycle winding certificate).
Status: RESEARCH (finite winding-sector diagnostic).
"""

from __future__ import annotations

import math
import numbers
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import (  # noqa: E402
    classify_winding_sector,
    winding_number,
    winding_ring,
)
from tnfr.physics.unified import compute_energy_density  # noqa: E402

_TWO_PI = 2.0 * math.pi


def _phase_vector(phases: np.ndarray) -> np.ndarray:
    """Return a finite one-dimensional phase vector for a closed ring."""
    raw = np.asarray(phases)
    if raw.ndim != 1 or raw.size < 3:
        raise ValueError("phases must be a one-dimensional ring with at least 3 nodes")
    if raw.dtype.kind not in "fiu":
        raise TypeError("phases must contain real numeric values")
    result = raw.astype(float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("phases must be finite")
    return result


def _finite_control(value: object, *, name: str, positive: bool) -> float:
    """Validate a finite real benchmark control."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (positive and result <= 0.0) or (not positive and result < 0.0):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {relation}")
    return result


def _wrap_pi_array(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (x + math.pi) % _TWO_PI - math.pi


def ring_phase_gradient_mean(phases: np.ndarray) -> float:
    """Mean |wrap(neighbour phase difference)| on a closed ring (tetrad |grad phi|)."""
    phases = _phase_vector(phases)
    diffs = _wrap_pi_array(np.roll(phases, -1) - phases)
    return float(np.mean(np.abs(diffs)))


def relax_phase_ring(
    phases: np.ndarray,
    *,
    nu_f: float = 1.0,
    dt: float = 0.1,
    steps: int = 600,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Wrapped nearest-neighbour consensus flow on a ring.

    Integrates d phi_i/dt = nu_f * mean_j wrap(phi_j - phi_i), the smooth-limit
    analogue of graph diffusion on a circular phase variable. Winding can be
    preserved while increments stay away from the wrap discontinuity; the
    finite protocol below checks that condition empirically.
    """
    p = _phase_vector(phases)
    nu_f = _finite_control(nu_f, name="nu_f", positive=True)
    dt = _finite_control(dt, name="dt", positive=True)
    noise = _finite_control(noise, name="noise", positive=False)
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, numbers.Integral):
        raise TypeError("steps must be an integer")
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        p = p + noise * rng.standard_normal(p.shape[0])
    initial_signed = _wrap_pi_array(np.roll(p, -1) - p)
    initial_margin = float(np.min(math.pi - np.abs(initial_signed)))
    if initial_margin <= 0.0:
        raise ValueError("initial phase state lies on the wrap branch boundary")
    path_windings = [int(round(float(np.sum(initial_signed) / _TWO_PI)))]
    minimum_branch_margin = initial_margin
    for _ in range(steps):
        left = _wrap_pi_array(np.roll(p, 1) - p)
        right = _wrap_pi_array(np.roll(p, -1) - p)
        p = p + dt * nu_f * 0.5 * (left + right)
        signed = _wrap_pi_array(np.roll(p, -1) - p)
        path_windings.append(int(round(float(np.sum(signed) / _TWO_PI))))
        minimum_branch_margin = min(
            minimum_branch_margin,
            float(np.min(math.pi - np.abs(signed))),
        )
    final_signed = _wrap_pi_array(np.roll(p, -1) - p)
    return p, {
        "path_windings": tuple(path_windings),
        "minimum_branch_margin": minimum_branch_margin,
        "final_signed_gradient_std": float(np.std(final_signed)),
    }


def _phases_of(G) -> np.ndarray:
    nodes = sorted(G.nodes())
    return np.array([G.nodes[i]["phase"] for i in nodes], dtype=float)


def _structural_snapshot_energy(G) -> float:
    ed = compute_energy_density(G)
    return float(sum(ed[n] for n in G.nodes()))


def measure_catalog(n: int = 60, windings: tuple[int, ...] = (0, 1, 2, 3, 4, 5)):
    """Measure winding, snapshot energy and gradient proxy per prepared sector."""
    rows = []
    for w in windings:
        G = winding_ring(n, w)
        phases = _phases_of(G)
        w_meas, raw = winding_number(G)
        sector = classify_winding_sector(G)
        grad = ring_phase_gradient_mean(phases)
        e_xy = float(np.sum(1.0 - np.cos(_wrap_pi_array(np.roll(phases, -1) - phases))))
        e_can = _structural_snapshot_energy(G)
        phase_gradient_proxy = 1.0 / (1.0 + grad)
        rows.append(
            {
                "W_in": w,
                "W_meas": w_meas,
                "raw": raw,
                "class": sector.winding_class,
                "grad_phi": grad,
                "grad_pred": _TWO_PI * abs(w) / n,
                "E_xy": e_xy,
                "E_snapshot": e_can,
                "phase_gradient_proxy": phase_gradient_proxy,
            }
        )
    return rows


def loglog_slope(ws, energies) -> float:
    """Slope of log(E) vs log(W) over the nonzero-W points (expected ~ 2)."""
    ws = tuple(ws)
    energies = tuple(energies)
    if len(ws) != len(energies):
        raise ValueError("ws and energies must have equal lengths")
    xs, ys = [], []
    for w, e in zip(ws, energies):
        if w > 0 and e > 0:
            xs.append(math.log(w))
            ys.append(math.log(e))
    if len(xs) < 2:
        return float("nan")
    a = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(a[0])


def main() -> None:
    print("=" * 72)
    print("FINITE WINDING-SECTOR ENERGY AND RELAXATION STUDY")
    print("=" * 72)

    n = 60
    windings = (0, 1, 2, 3, 4, 5)
    rows = measure_catalog(n=n, windings=windings)

    # -- M1: loop winding on the prepared finite family ------------------------
    print("\n[M1] LOOP WINDING on the prepared finite family.")
    print(f"     {'W_in':>5} {'W_meas':>7} {'raw':>10} {'class':>42}")
    all_int = True
    for r in rows:
        print(f"     {r['W_in']:>5} {r['W_meas']:>7} {r['raw']:>10.6f}   {r['class']}")
        if abs(r["raw"] - round(r["raw"])) > 1e-9:
            all_int = False
    assert all_int, "winding not integer -- quantization failed"
    assert [r["W_meas"] for r in rows] == list(windings), "winding mismatch"
    print("     -> PASS: every prepared loop returns its target integer winding.")

    # -- M2: finite-family energy hierarchy E(W) ~ W^2 -------------------------
    # The implemented snapshot energy carries a constant W=0 baseline.
    # (the Phi_s / K_phi offset that does not scale with the winding); the
    # Excitation above the baseline, E_exc(W) = E_snapshot(W) - E_snapshot(0), is
    # a structural snapshot-energy difference for this prepared family.
    print("\n[M2] ENERGY HIERARCHY (S2): E(W) ~ W^2 (super-linear).")
    print(
        f"     {'W':>3} {'E_xy':>10} {'E_exc(>base)':>12} "
        f"{'E_xy/E(1)':>10} {'E_exc/E(1)':>11}"
    )
    e1_xy = next(r["E_xy"] for r in rows if r["W_in"] == 1)
    e_vac = next(r["E_snapshot"] for r in rows if r["W_in"] == 0)
    e1_exc = next(r["E_snapshot"] for r in rows if r["W_in"] == 1) - e_vac
    e_exc = {}
    for r in rows:
        exc = r["E_snapshot"] - e_vac
        e_exc[r["W_in"]] = exc
        ratio_xy = r["E_xy"] / e1_xy if e1_xy > 0 else float("nan")
        ratio_exc = exc / e1_exc if e1_exc > 0 else float("nan")
        print(
            f"     {r['W_in']:>3} {r['E_xy']:>10.5f} {exc:>12.5f} "
            f"{ratio_xy:>10.3f} {ratio_exc:>11.3f}"
        )
    slope_xy = loglog_slope([r["W_in"] for r in rows], [r["E_xy"] for r in rows])
    slope_exc = loglog_slope(list(e_exc.keys()), list(e_exc.values()))
    print(f"     log-log slope  E_xy(W)       : {slope_xy:.3f}  (expected ~ 2)")
    print(f"     log-log slope  E_exc(W)      : {slope_exc:.3f}  (expected ~ 2)")
    ratios_match = all(
        math.isclose(e_exc[w] / e1_exc, float(w * w), rel_tol=1e-10, abs_tol=1e-10)
        for w in windings
        if w > 0
    )
    print(f"     E_exc ratios match W^2 within 1e-10: {ratios_match}.")
    assert 1.7 < slope_xy < 2.3, f"E_xy slope {slope_xy} not ~2"
    assert 1.9 < slope_exc < 2.1, f"E_exc slope {slope_exc} not ~2"
    assert ratios_match, "snapshot excitation ratios do not match W^2"
    print("     -> PASS: this implemented excitation read-out is quadratic in W")
    print("        over the sampled uniform-ring family.")

    # -- M3: sampled path winding and final gradient uniformity ----------------
    print("\n[M3] FINITE RELAXATION: wrapped consensus approaches 2pi|W|/n.")
    print(
        f"     {'W':>3} {'|grad|*meas':>12} {'2pi|W|/n':>10} "
        f"{'P_grad':>8} {'W path':>8} {'std grad':>10}"
    )
    prev_c = None
    monotone = True
    for w in windings:
        G = winding_ring(n, w)
        p0 = _phases_of(G)
        # Check a noise-perturbed prepared state under this finite protocol.
        p_relaxed, diagnostics = relax_phase_ring(
            p0, noise=0.15, seed=100 + w, steps=1600
        )
        for i, node in enumerate(sorted(G.nodes())):
            G.nodes[node]["phase"] = float(p_relaxed[i])
            G.nodes[node]["theta"] = float(p_relaxed[i])
        w_after, _ = winding_number(G)
        grad_after = ring_phase_gradient_mean(p_relaxed)
        phase_proxy = 1.0 / (1.0 + grad_after)
        path_windings = diagnostics["path_windings"]
        path_kept = bool(path_windings) and all(
            measured == w for measured in path_windings
        )
        gradient_std = float(diagnostics["final_signed_gradient_std"])
        print(
            f"     {w:>3} {grad_after:>12.5f} {_TWO_PI * abs(w) / n:>10.5f} "
            f"{phase_proxy:>8.4f} {str(path_kept):>8} {gradient_std:>10.3e}"
        )
        assert w_after == w and path_kept, (
            f"winding {w} changed in the sampled consensus trajectory"
        )
        assert float(diagnostics["minimum_branch_margin"]) > 1e-6
        assert gradient_std < 2e-3
        if prev_c is not None and phase_proxy > prev_c + 1e-9:
            monotone = False
        prev_c = phase_proxy
    assert monotone, "phase-gradient proxy is not monotone in |W|"
    print("     -> PASS: sampled path windings stayed fixed, the final signed")
    print("        gradients are nearly uniform, and P_grad decreases with |W|.")

    # -- M4: energetic comparison; no fission dynamics are simulated -----------
    e2_exc = e_exc[2]
    print("\n[M4] ENERGETIC COMPARISON: W=2 versus two W=1 values.")
    print(f"     E_exc(W=2)         = {e2_exc:.5f}")
    print(f"     2 x E_exc(W=1)     = {2 * e1_exc:.5f}")
    print(f"     E(2) / [2E(1)]     = {e2_exc / (2 * e1_exc):.3f}")
    assert e2_exc > 2 * e1_exc, "sampled W=2 energy is not above two W=1 values"
    print("     -> The sampled energy is superadditive at W=2. Stability and")
    print("        fission require a multi-defect dynamics that is not tested here.")

    print("\n" + "=" * 72)
    print("SCOPED RESULT:")
    print("  W = 0     : zero-winding sector")
    print("  |W| = 1   : unit-winding sector")
    print("  |W| >= 2  : higher-winding sectors")
    print("  sign(W)   : loop orientation")
    print("Measured here: finite-family energy ordering and consensus relaxation.")
    print("Open: formation, stability, fission and any physical interpretation.")
    print("=" * 72)


if __name__ == "__main__":
    main()
