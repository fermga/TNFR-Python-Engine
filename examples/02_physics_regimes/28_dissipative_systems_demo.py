"""Scoped Lindblad diagnostics: unital mixing and non-unital purification.

This example verifies exact finite-dimensional GKSL identities used by TNFR's
open-system diagnostic module. The density operator is an auxiliary model; its
collapse operators do not identify violations of grammar U1--U6.

Demonstrated results:

1. A universal dissipator-action bound remains nonzero for a generic pure state.
2. Pure dephasing is unital and monotonically reduces purity.
3. Amplitude damping is non-unital: an excited state first mixes, then purifies.
4. Fixed-point contractivity is evaluated in trace distance.
5. Empirical change tiers report magnitude and direction without a grammar verdict.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from tnfr.physics.dissipative_conservation import (
    capture_dissipative_snapshot,
    classify_dissipative_regime,
    compute_dissipation_bound,
    compute_dissipator_action,
    compute_instantaneous_purity_rate,
    compute_purity_decay_bound,
    is_unital_dissipator,
    predict_amplitude_damping_purity,
    predict_dephasing_purity,
    verify_dissipative_balance,
)


def _pure_state(index: int) -> np.ndarray:
    vector = np.zeros(2, dtype=np.complex128)
    vector[index] = 1.0
    return np.outer(vector, vector.conj())


def _plus_state() -> np.ndarray:
    vector = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    return np.outer(vector, vector.conj())


def _amplitude_damping_operator(gamma: float) -> np.ndarray:
    operator = np.zeros((2, 2), dtype=np.complex128)
    operator[0, 1] = math.sqrt(gamma)
    return operator


def _dephasing_operator(gamma: float) -> np.ndarray:
    return math.sqrt(gamma / 2.0) * np.diag([1.0, -1.0]).astype(np.complex128)


def _amplitude_damping_channel(
    density: np.ndarray, gamma: float, time: float
) -> np.ndarray:
    eta = math.exp(-gamma * time)
    excited_population = float(density[1, 1].real)
    return np.array(
        [
            [
                density[0, 0] + (1.0 - eta) * excited_population,
                math.sqrt(eta) * density[0, 1],
            ],
            [math.sqrt(eta) * density[1, 0], eta * density[1, 1]],
        ],
        dtype=np.complex128,
    )


def _dephasing_channel(
    density: np.ndarray, gamma: float, time: float
) -> np.ndarray:
    result = density.copy()
    decay = math.exp(-gamma * time)
    result[0, 1] *= decay
    result[1, 0] *= decay
    return result


def demo_state_validation() -> None:
    print("=" * 72)
    print("1. VALIDATED DENSITY-STATE SNAPSHOTS")
    print("=" * 72)
    states = {
        "ground pure": _pure_state(0),
        "plus pure": _plus_state(),
        "maximally mixed": np.eye(2, dtype=np.complex128) / 2.0,
    }
    for label, density in states.items():
        snapshot = capture_dissipative_snapshot(density)
        print(
            f"  {label:18s} trace={snapshot.trace:.6f}  "
            f"purity={snapshot.purity:.6f}  "
            f"entropy={snapshot.von_neumann_entropy:.6f}"
        )
    print()


def demo_universal_bounds() -> None:
    print("=" * 72)
    print("2. UNIVERSAL NORMS — PURE DOES NOT MEAN STATIONARY")
    print("=" * 72)
    gamma = 0.4
    excited = _pure_state(1)
    collapse = [_amplitude_damping_operator(gamma)]
    action = compute_dissipator_action(excited, collapse)
    action_norm = float(np.linalg.norm(action, ord="fro"))
    action_bound = compute_dissipation_bound(collapse, purity=1.0)
    purity_rate = compute_instantaneous_purity_rate(collapse, excited)
    purity_bound = compute_purity_decay_bound(collapse, excited)

    assert action_norm <= action_bound + 1e-12
    assert abs(purity_rate) <= purity_bound + 1e-12
    assert action_norm > 0.0
    print(f"  ||D[|1><1|]||_F                 = {action_norm:.6f}")
    print(f"  universal dissipator bound      = {action_bound:.6f}")
    print(f"  instantaneous dP/dt             = {purity_rate:+.6f}")
    print(f"  universal |dP/dt| bound         = {purity_bound:.6f}")
    print("  Result: a pure excited state has nonzero dissipative motion.")
    print()


def demo_channel_hypotheses() -> None:
    print("=" * 72)
    print("3. CHANNEL HYPOTHESES — UNITALITY CONTROLS MONOTONE MIXING")
    print("=" * 72)
    gamma = 0.5
    amplitude = [_amplitude_damping_operator(gamma)]
    dephasing = [_dephasing_operator(gamma)]

    mostly_ground = np.diag([0.9, 0.1]).astype(np.complex128)
    amplitude_rate = compute_instantaneous_purity_rate(amplitude, mostly_ground)
    dephasing_rate = compute_instantaneous_purity_rate(dephasing, _plus_state())

    assert not is_unital_dissipator(amplitude)
    assert is_unital_dissipator(dephasing)
    assert amplitude_rate > 0.0
    assert dephasing_rate < 0.0
    print(
        f"  amplitude damping: unital={is_unital_dissipator(amplitude)!s:5s}, "
        f"dP/dt={amplitude_rate:+.6f}"
    )
    print(
        f"  pure dephasing:    unital={is_unital_dissipator(dephasing)!s:5s}, "
        f"dP/dt={dephasing_rate:+.6f}"
    )
    print("  Result: general GKSL dynamics does not fix the sign of dP/dt.")
    print()


def demo_exact_trajectories() -> None:
    print("=" * 72)
    print("4. EXACT CHANNEL TRAJECTORIES")
    print("=" * 72)
    gamma = 0.5
    excited = _pure_state(1)
    plus = _plus_state()
    times = [0.0, 0.5, math.log(2.0) / gamma, 3.0, 8.0]

    amplitude_purities: list[float] = []
    dephasing_purities: list[float] = []
    print("  t        amplitude P   amplitude S   dephasing P   dephasing S")
    print("  -------  ------------  ------------  ------------  ------------")
    for time in times:
        amplitude_state = _amplitude_damping_channel(excited, gamma, time)
        dephasing_state = _dephasing_channel(plus, gamma, time)
        amplitude_snapshot = capture_dissipative_snapshot(amplitude_state)
        dephasing_snapshot = capture_dissipative_snapshot(dephasing_state)
        amplitude_purities.append(amplitude_snapshot.purity)
        dephasing_purities.append(dephasing_snapshot.purity)
        print(
            f"  {time:7.3f}  {amplitude_snapshot.purity:12.6f}  "
            f"{amplitude_snapshot.von_neumann_entropy:12.6f}  "
            f"{dephasing_snapshot.purity:12.6f}  "
            f"{dephasing_snapshot.von_neumann_entropy:12.6f}"
        )

    assert math.isclose(amplitude_purities[2], 0.5, abs_tol=1e-12)
    assert amplitude_purities[-1] > amplitude_purities[2]
    assert all(
        later <= earlier + 1e-12
        for earlier, later in zip(dephasing_purities, dephasing_purities[1:])
    )
    print("  Result: amplitude damping mixes then purifies; dephasing only mixes.")
    print()
def demo_predictions_and_balance() -> None:
    print("=" * 72)
    print("5. ANALYTICAL PREDICTIONS AND TRACE-DISTANCE BALANCE")
    print("=" * 72)
    gamma = 0.3
    time = 1.2
    initial = np.array(
        [[0.35, 0.12 + 0.08j], [0.12 - 0.08j, 0.65]], dtype=np.complex128
    )
    amplitude_state = _amplitude_damping_channel(initial, gamma, time)
    predicted_amplitude = predict_amplitude_damping_purity(initial, gamma, time)
    actual_amplitude = capture_dissipative_snapshot(amplitude_state).purity
    predicted_dephasing = predict_dephasing_purity(initial, gamma, time)
    dephasing_state = _dephasing_channel(initial, gamma, time)
    actual_dephasing = capture_dissipative_snapshot(dephasing_state).purity

    assert math.isclose(predicted_amplitude, actual_amplitude, abs_tol=1e-12)
    assert math.isclose(predicted_dephasing, actual_dephasing, abs_tol=1e-12)
    print(
        f"  amplitude damping: predicted={predicted_amplitude:.9f}, "
        f"actual={actual_amplitude:.9f}"
    )
    print(
        f"  dephasing:         predicted={predicted_dephasing:.9f}, "
        f"actual={actual_dephasing:.9f}"
    )

    before = capture_dissipative_snapshot(_pure_state(1))
    after_state = _amplitude_damping_channel(_pure_state(1), gamma, time)
    after = capture_dissipative_snapshot(after_state)
    balance = verify_dissipative_balance(
        before,
        after,
        dt=time,
        collapse_operators=[_amplitude_damping_operator(gamma)],
        steady_state=_pure_state(0),
    )
    classification = classify_dissipative_regime(balance)
    assert balance.contractivity_evaluated and balance.is_contractive
    assert not classification["grammar_status_inferred"]
    print(f"  fixed-point trace-distance ratio = {balance.contractivity_gap:.6f}")
    print(
        f"  change tier={classification['change_tier']}, "
        f"purity={classification['purity_direction']}, "
        f"entropy={classification['entropy_direction']}"
    )
    print(f"  grammar inference: {classification['grammar_analog']}")
    print()


def main() -> None:
    print()
    print("TNFR OPEN-SYSTEM DIAGNOSTICS — SCOPED GKSL IDENTITIES")
    print()
    demo_state_validation()
    demo_universal_bounds()
    demo_channel_hypotheses()
    demo_exact_trajectories()
    demo_predictions_and_balance()
    print("All algebraic bounds and analytical channel checks passed.")
    print("See: theory/DISSIPATIVE_AND_OPEN_SYSTEMS.md")


if __name__ == "__main__":
    main()
