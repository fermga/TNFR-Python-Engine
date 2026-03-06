"""TNFR Dissipative Systems — Lindblad Decoherence and Grammar Violations.

Demonstrates how TNFR structural conservation extends to dissipative
(open-system) regimes via the Lindblad master equation, and reveals
WHY grammar rules U1-U6 exist: violations act as collapse operators
that drive decoherence and purity loss.

Key results shown:
1. Density operator snapshots: trace, purity, von Neumann entropy
2. Lindblad dissipator action D[rho] from collapse operators
3. Dissipation bound: |D[rho]| <= Sum_k ||L_k||^2 * (1 - Tr(rho^2))
4. Purity decay and entropy production tracking
5. Dissipative balance verification (contractivity, charge leak rate)
6. Regime classification: weak / moderate / strong / decoherence
7. Grammar violations as collapse operators (U2 -> decoherence mapping)
8. Analytical predictions: amplitude damping and dephasing channels

See: theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md for the full treatment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from tnfr.physics.dissipative_conservation import (
    capture_dissipative_snapshot,
    compute_dissipation_bound,
    compute_dissipator_action,
    compute_purity_decay_bound,
    verify_dissipative_balance,
    predict_amplitude_damping_purity,
    predict_dephasing_purity,
    classify_dissipative_regime,
    DissipativeTimeSeries,
)

SEED = 42


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_pure_state(dim: int = 4) -> np.ndarray:
    """Create a random pure-state density operator |psi><psi|."""
    rng = np.random.default_rng(SEED)
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def _make_mixed_state(dim: int = 4, purity_target: float = 0.5) -> np.ndarray:
    """Create a mixed-state density operator with approximate purity."""
    rng = np.random.default_rng(SEED + 1)
    # Blend pure state with maximally mixed
    pure = _make_pure_state(dim)
    identity = np.eye(dim, dtype=np.complex128) / dim
    # alpha * pure + (1-alpha) * I/d  gives purity ~ alpha^2 + (1-alpha)^2/d
    alpha = min(1.0, max(0.0, purity_target))
    rho = alpha * pure + (1 - alpha) * identity
    return rho / np.trace(rho).real


def _make_collapse_operators(dim: int = 4, strength: float = 0.1) -> list[np.ndarray]:
    """Create collapse operators for dephasing and amplitude damping."""
    ops: list[np.ndarray] = []

    # Dephasing: L_deph = sqrt(gamma) * |k><k| (diagonal projectors)
    for k in range(dim):
        L = np.zeros((dim, dim), dtype=np.complex128)
        L[k, k] = np.sqrt(strength)
        ops.append(L)

    # Amplitude damping: L_ad = sqrt(gamma) * |k><k+1| (lowering operators)
    for k in range(dim - 1):
        L = np.zeros((dim, dim), dtype=np.complex128)
        L[k, k + 1] = np.sqrt(strength * 0.5)
        ops.append(L)

    return ops


def _lindblad_step(
    rho: np.ndarray,
    collapse_operators: list[np.ndarray],
    hamiltonian: np.ndarray | None = None,
    dt: float = 0.1,
) -> np.ndarray:
    """One Euler step of the Lindblad master equation.

    d rho/dt = -i[H, rho] + D[rho]
    """
    # Hamiltonian contribution
    if hamiltonian is not None:
        drho = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    else:
        drho = np.zeros_like(rho)

    # Dissipator D[rho]
    drho += compute_dissipator_action(rho, collapse_operators)

    rho_new = rho + dt * drho

    # Enforce Hermiticity and trace preservation
    rho_new = 0.5 * (rho_new + rho_new.conj().T)
    rho_new /= np.trace(rho_new).real

    return rho_new


# ------------------------------------------------------------------
# 1. Density operator snapshots
# ------------------------------------------------------------------

def demo_snapshots() -> None:
    """Capture and compare structural invariants of pure vs mixed states."""
    print("=" * 60)
    print("1. DENSITY OPERATOR SNAPSHOTS — pure vs mixed states")
    print("=" * 60)

    dim = 4

    pure = _make_pure_state(dim)
    mixed = _make_mixed_state(dim, purity_target=0.4)
    maximally_mixed = np.eye(dim, dtype=np.complex128) / dim

    for label, rho in [("Pure state", pure), ("Mixed (P~0.4)", mixed),
                        ("Maximally mixed", maximally_mixed)]:
        snap = capture_dissipative_snapshot(rho)
        print(f"\n  {label}:")
        print(f"    Trace   = {snap.trace:.6f}  (should be 1)")
        print(f"    Purity  = {snap.purity:.6f}  (1=pure, 1/{dim}={1/dim:.4f}=max mixed)")
        print(f"    Entropy = {snap.von_neumann_entropy:.6f}  (0=pure, ln({dim})={np.log(dim):.4f}=max)")
    print()


# ------------------------------------------------------------------
# 2. Dissipation bound
# ------------------------------------------------------------------

def demo_dissipation_bound() -> None:
    """Verify the theoretical bound |D[rho]| <= Sum ||L_k||^2 * (1 - P)."""
    print("=" * 60)
    print("2. DISSIPATION BOUND — |D[rho]| <= Sum ||L_k||^2 * (1 - Tr(rho^2))")
    print("=" * 60)

    dim = 4
    collapse_ops = _make_collapse_operators(dim, strength=0.1)

    for label, rho in [("Pure", _make_pure_state(dim)),
                        ("Mixed", _make_mixed_state(dim, 0.5)),
                        ("Max mixed", np.eye(dim, dtype=np.complex128) / dim)]:
        snap = capture_dissipative_snapshot(rho)
        bound = compute_dissipation_bound(collapse_ops, snap.purity)
        D_action = compute_dissipator_action(rho, collapse_ops)
        actual = float(np.linalg.norm(D_action, ord='fro'))
        satisfied = actual <= bound + 1e-10

        print(f"\n  {label}:")
        print(f"    Purity P = {snap.purity:.6f}")
        print(f"    |D[rho]| actual = {actual:.6f}")
        print(f"    |D[rho]| bound  = {bound:.6f}")
        print(f"    Bound satisfied = {satisfied}")
    print()


# ------------------------------------------------------------------
# 3. Purity decay and entropy production tracking
# ------------------------------------------------------------------

def demo_purity_tracking() -> None:
    """Track purity decay and entropy growth under Lindblad evolution."""
    print("=" * 60)
    print("3. PURITY DECAY & ENTROPY PRODUCTION — Lindblad trajectory")
    print("=" * 60)

    dim = 4
    rho = _make_pure_state(dim)
    collapse_ops = _make_collapse_operators(dim, strength=0.08)
    dt = 0.1
    steps = 30

    series = DissipativeTimeSeries()
    current = rho.copy()

    for step in range(steps + 1):
        snap = capture_dissipative_snapshot(current)
        series.times.append(step * dt)
        series.purity.append(snap.purity)
        series.entropy.append(snap.von_neumann_entropy)

        if step < steps:
            current = _lindblad_step(current, collapse_ops, dt=dt)

    print(f"\n  Evolution: {steps} steps, dt = {dt}")
    print(f"  Purity:  {series.purity[0]:.4f} → {series.purity[-1]:.4f}  "
          f"(change: {series.purity[-1] - series.purity[0]:+.4f})")
    print(f"  Entropy: {series.entropy[0]:.4f} → {series.entropy[-1]:.4f}  "
          f"(change: {series.entropy[-1] - series.entropy[0]:+.4f})")
    print(f"  Total entropy produced: {series.entropy[-1] - series.entropy[0]:.4f}")

    # Show a few milestones
    milestones = [0, 5, 10, 20, steps]
    print("\n  Step  |  t    |  Purity  |  Entropy")
    print("  ------|-------|----------|----------")
    for m in milestones:
        if m <= steps:
            print(f"  {m:4d}  | {series.times[m]:5.2f} | {series.purity[m]:.4f}   | {series.entropy[m]:.4f}")
    print()


# ------------------------------------------------------------------
# 4. Dissipative balance verification
# ------------------------------------------------------------------

def demo_dissipative_balance() -> None:
    """Verify the dissipative continuity equation between snapshots."""
    print("=" * 60)
    print("4. DISSIPATIVE BALANCE — continuity equation verification")
    print("=" * 60)

    dim = 4
    rho = _make_mixed_state(dim, purity_target=0.7)
    collapse_ops = _make_collapse_operators(dim, strength=0.05)
    dt = 0.1

    snap_before = capture_dissipative_snapshot(rho)
    rho_after = _lindblad_step(rho, collapse_ops, dt=dt)
    snap_after = capture_dissipative_snapshot(rho_after)

    balance = verify_dissipative_balance(
        snap_before, snap_after,
        dt=dt,
        collapse_operators=collapse_ops,
    )

    print(f"\n  Purity:  {balance.purity_before:.6f} → {balance.purity_after:.6f}")
    print(f"  Purity decay rate:       {balance.purity_decay_rate:.6f} (should be <= 0)")
    print(f"  Entropy production rate: {balance.entropy_production_rate:.6f} (should be >= 0)")
    print(f"  Trace drift:             {balance.trace_drift:.2e}")
    print(f"  Dissipation bound:       {balance.dissipation_bound:.6f}")
    print(f"  Actual dissipation:      {balance.actual_dissipation:.6f}")
    print(f"  Charge leak rate:        {balance.charge_leak_rate:.6f}")
    print(f"  Contractivity gap:       {balance.contractivity_gap:.6f}")
    print(f"  Is contractive?          {balance.is_contractive}")
    print()


# ------------------------------------------------------------------
# 5. Regime classification
# ------------------------------------------------------------------

def demo_regime_classification() -> None:
    """Classify dissipative strength into TNFR grammar tiers."""
    print("=" * 60)
    print("5. REGIME CLASSIFICATION — four dissipative tiers")
    print("=" * 60)

    dim = 4
    strengths = [0.001, 0.02, 0.1, 0.5]
    dt = 0.5  # Larger dt amplifies dissipation

    for strength in strengths:
        rho = _make_mixed_state(dim, purity_target=0.8)
        collapse_ops = _make_collapse_operators(dim, strength=strength)

        snap_before = capture_dissipative_snapshot(rho)
        rho_after = _lindblad_step(rho, collapse_ops, dt=dt)
        snap_after = capture_dissipative_snapshot(rho_after)

        balance = verify_dissipative_balance(snap_before, snap_after, dt=dt,
                                              collapse_operators=collapse_ops)
        classification = classify_dissipative_regime(balance)

        print(f"\n  Collapse strength = {strength}")
        print(f"    Regime:        {classification['regime']}")
        print(f"    Quality:       {classification['conservation_quality']:.3f}")
        print(f"    Grammar:       {classification['grammar_analog']}")
    print()


# ------------------------------------------------------------------
# 6. Grammar violations as collapse operators
# ------------------------------------------------------------------

def demo_grammar_violations() -> None:
    """Show how grammar U2 violation magnitude maps to dissipation."""
    print("=" * 60)
    print("6. GRAMMAR VIOLATIONS → COLLAPSE OPERATORS")
    print("=" * 60)

    dim = 4
    rho = _make_pure_state(dim)

    print("\n  U2 compliance level  |  After 20 steps")
    print("  --------------------|-------------------")

    for label, strength in [("Full compliance (weak)", 0.001),
                             ("Partial violation (moderate)", 0.03),
                             ("Strong violation", 0.15),
                             ("No stabilizers (critical)", 0.5)]:
        collapse_ops = _make_collapse_operators(dim, strength=strength)
        current = rho.copy()
        for _ in range(20):
            current = _lindblad_step(current, collapse_ops, dt=0.1)

        snap = capture_dissipative_snapshot(current)
        print(f"  {label:35s}  P={snap.purity:.4f}  S={snap.von_neumann_entropy:.4f}")

    print("\n  Interpretation: Stronger grammar violations (U2 destabilizers")
    print("  without stabilizers) map to stronger collapse operators,")
    print("  causing faster purity loss and entropy production.")
    print()


# ------------------------------------------------------------------
# 7. Analytical predictions
# ------------------------------------------------------------------

def demo_analytical_predictions() -> None:
    """Compare Lindblad evolution with analytical predictions."""
    print("=" * 60)
    print("7. ANALYTICAL PREDICTIONS — amplitude damping & dephasing")
    print("=" * 60)

    dim = 4
    gamma = 0.1
    t_values = [0.0, 1.0, 3.0, 5.0, 10.0]

    # Amplitude damping prediction
    print("\n  Amplitude damping (gamma = 0.1):")
    print("    t     |  P_predicted  |  P_actual")
    print("  --------|---------------|----------")

    rho0 = _make_mixed_state(dim, purity_target=0.6)
    snap0 = capture_dissipative_snapshot(rho0)
    P0 = snap0.purity

    # Build amplitude damping collapse operators
    ad_ops: list[np.ndarray] = []
    for k in range(dim - 1):
        L = np.zeros((dim, dim), dtype=np.complex128)
        L[k, k + 1] = np.sqrt(gamma)
        ad_ops.append(L)

    for t in t_values:
        steps = max(1, int(t / 0.05))
        dt = t / steps if steps > 0 else 0.05
        current = rho0.copy()
        for _ in range(steps):
            current = _lindblad_step(current, ad_ops, dt=dt)
        snap = capture_dissipative_snapshot(current)
        predicted = predict_amplitude_damping_purity(P0, gamma, t, dim)
        print(f"  {t:7.1f} | {predicted:13.6f} | {snap.purity:.6f}")

    # Dephasing prediction
    print("\n  Pure dephasing (gamma = 0.1):")
    print("    t     |  P_predicted  |  P_actual")
    print("  --------|---------------|----------")

    deph_ops: list[np.ndarray] = []
    for k in range(dim):
        L = np.zeros((dim, dim), dtype=np.complex128)
        L[k, k] = np.sqrt(gamma)
        deph_ops.append(L)

    for t in t_values:
        steps = max(1, int(t / 0.05))
        dt = t / steps if steps > 0 else 0.05
        current = rho0.copy()
        for _ in range(steps):
            current = _lindblad_step(current, deph_ops, dt=dt)
        snap = capture_dissipative_snapshot(current)
        predicted = predict_dephasing_purity(rho0, gamma, t)
        print(f"  {t:7.1f} | {predicted:13.6f} | {snap.purity:.6f}")
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print()
    print("TNFR DISSIPATIVE SYSTEMS — LINDBLAD DECOHERENCE & GRAMMAR")
    print("Grammar violations mapping to Lindblad collapse operators.")
    print("Conservation → Dissipation → Decoherence continuum.")
    print()

    demo_snapshots()
    demo_dissipation_bound()
    demo_purity_tracking()
    demo_dissipative_balance()
    demo_regime_classification()
    demo_grammar_violations()
    demo_analytical_predictions()

    print("=" * 60)
    print("CONCLUSION: Grammar rules U1-U6 ensure conservative dynamics.")
    print("Violations act as Lindblad collapse operators, causing")
    print("purity loss, entropy production, and structural decoherence.")
    print("See: theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
