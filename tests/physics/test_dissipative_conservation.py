"""Tests for scoped finite-dimensional Lindblad diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from tnfr.physics.dissipative_conservation import (
    DissipativeBalance,
    DissipativeConservationTracker,
    DissipativeTimeSeries,
    analyze_dissipation_rates,
    capture_dissipative_snapshot,
    classify_dissipative_regime,
    compute_dissipation_bound,
    compute_dissipator_action,
    compute_instantaneous_purity_rate,
    compute_purity_decay_bound,
    is_unital_dissipator,
    predict_amplitude_damping_purity,
    predict_dephasing_purity,
    steady_state_from_generator,
    verify_dissipative_balance,
)

try:
    from tnfr.mathematics.dynamics import ContractiveDynamicsEngine
    from tnfr.mathematics.generators import build_lindblad_delta_nfr
    from tnfr.mathematics.spaces import HilbertSpace

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False


def _pure_state(index: int = 0) -> np.ndarray:
    vector = np.zeros(2, dtype=np.complex128)
    vector[index] = 1.0
    return np.outer(vector, vector.conj())


def _maximally_mixed(dim: int = 2) -> np.ndarray:
    return np.eye(dim, dtype=np.complex128) / dim


def _plus_state() -> np.ndarray:
    vector = np.array([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
    return np.outer(vector, vector.conj())


def _amplitude_damping_ops(gamma: float = 0.1) -> list[np.ndarray]:
    operator = np.zeros((2, 2), dtype=np.complex128)
    operator[0, 1] = math.sqrt(gamma)
    return [operator]


def _dephasing_ops(gamma: float = 0.1) -> list[np.ndarray]:
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    return [math.sqrt(gamma / 2.0) * sigma_z]


def _amplitude_damping_state(
    density: np.ndarray, gamma: float, time: float
) -> np.ndarray:
    eta = math.exp(-gamma * time)
    excited = float(density[1, 1].real)
    return np.array(
        [
            [density[0, 0] + (1.0 - eta) * excited, math.sqrt(eta) * density[0, 1]],
            [math.sqrt(eta) * density[1, 0], eta * density[1, 1]],
        ],
        dtype=np.complex128,
    )


def _dephased_state(density: np.ndarray, gamma: float, time: float) -> np.ndarray:
    result = density.copy()
    decay = math.exp(-gamma * time)
    result[0, 1] *= decay
    result[1, 0] *= decay
    return result


def _build_qubit_engine(collapse_ops: list[np.ndarray]):
    generator = build_lindblad_delta_nfr(
        hamiltonian=np.zeros((2, 2), dtype=np.complex128),
        collapse_operators=collapse_ops,
        dim=2,
    )
    return ContractiveDynamicsEngine(generator, HilbertSpace(2))


class TestSnapshotValidation:
    def test_pure_and_mixed_state_invariants(self):
        pure = capture_dissipative_snapshot(_pure_state())
        mixed = capture_dissipative_snapshot(_maximally_mixed(3))

        assert pure.trace == pytest.approx(1.0)
        assert pure.purity == pytest.approx(1.0)
        assert pure.von_neumann_entropy == pytest.approx(0.0)
        assert mixed.purity == pytest.approx(1.0 / 3.0)
        assert mixed.von_neumann_entropy == pytest.approx(math.log(3.0))

    @pytest.mark.parametrize(
        "density",
        [
            np.ones((2, 3)),
            np.array([[1.0, 0.2], [0.0, 0.0]]),
            2.0 * np.eye(2),
            np.diag([1.1, -0.1]),
            np.array([[np.nan, 0.0], [0.0, 1.0]]),
        ],
    )
    def test_invalid_density_is_rejected(self, density):
        with pytest.raises(ValueError):
            capture_dissipative_snapshot(density)

    @pytest.mark.parametrize("atol", [-1.0, float("nan")])
    def test_invalid_snapshot_tolerance_is_rejected(self, atol):
        with pytest.raises(ValueError):
            capture_dissipative_snapshot(_pure_state(), atol=atol)


class TestDissipatorAlgebra:
    def test_amplitude_damping_excited_state_action(self):
        gamma = 0.3
        action = compute_dissipator_action(
            _pure_state(1), _amplitude_damping_ops(gamma)
        )
        assert np.allclose(action, np.diag([gamma, -gamma]))
        assert np.trace(action) == pytest.approx(0.0)

    def test_pure_state_bound_is_nonzero_and_valid(self):
        gamma = 0.3
        density = _pure_state(1)
        action_norm = np.linalg.norm(
            compute_dissipator_action(density, _amplitude_damping_ops(gamma)),
            ord="fro",
        )
        bound = compute_dissipation_bound(
            _amplitude_damping_ops(gamma), purity=1.0
        )
        assert bound > 0.0
        assert action_norm <= bound

    def test_ground_state_is_stationary_but_bound_need_not_be_tight(self):
        operators = _amplitude_damping_ops(0.5)
        assert np.allclose(compute_dissipator_action(_pure_state(0), operators), 0.0)
        assert compute_dissipation_bound(operators, 1.0) > 0.0

    def test_unitality_distinguishes_channels(self):
        assert is_unital_dissipator(_dephasing_ops(0.2))
        assert not is_unital_dissipator(_amplitude_damping_ops(0.2))
        assert is_unital_dissipator([])

    def test_purity_rate_sign_is_not_universal(self):
        gamma = 0.4
        operators = _amplitude_damping_ops(gamma)
        excited_rate = compute_instantaneous_purity_rate(operators, _pure_state(1))
        mostly_ground = np.diag([0.9, 0.1]).astype(np.complex128)
        purifying_rate = compute_instantaneous_purity_rate(operators, mostly_ground)

        assert excited_rate == pytest.approx(-2.0 * gamma)
        assert purifying_rate > 0.0

    def test_unital_dephasing_decreases_plus_state_purity(self):
        rate = compute_instantaneous_purity_rate(
            _dephasing_ops(0.25), _plus_state()
        )
        assert rate == pytest.approx(-0.25)

    @pytest.mark.parametrize(
        "density,operators",
        [
            (_pure_state(1), _amplitude_damping_ops(0.7)),
            (_plus_state(), _dephasing_ops(0.4)),
            (np.diag([0.8, 0.2]).astype(np.complex128), _amplitude_damping_ops(0.3)),
        ],
    )
    def test_absolute_purity_rate_bound(self, density, operators):
        actual = abs(compute_instantaneous_purity_rate(operators, density))
        bound = compute_purity_decay_bound(operators, density)
        assert actual <= bound + 1e-12

    @pytest.mark.parametrize(
        "operators",
        [
            [np.ones((2, 3))],
            [np.eye(2), np.eye(3)],
            [np.array([[np.inf, 0.0], [0.0, 0.0]])],
            [np.array([[1e308, 0.0], [0.0, 0.0]])],
        ],
    )
    def test_invalid_collapse_operators_are_rejected(self, operators):
        with pytest.raises(ValueError):
            compute_dissipator_action(_pure_state(), operators)

    @pytest.mark.parametrize("purity", [-0.1, 1.1, float("nan")])
    def test_invalid_purity_is_rejected(self, purity):
        with pytest.raises(ValueError):
            compute_dissipation_bound(_amplitude_damping_ops(), purity)


class TestBalance:
    def test_historical_dataclass_schema_is_constructor_and_payload_stable(self):
        balance = DissipativeBalance(
            0.8,
            0.7,
            -0.1,
            0.2,
            0.3,
            0.1,
            0.0,
            1.0,
            0.25,
            0.05,
            0.9,
            True,
        )
        payload = asdict(balance)
        assert list(payload)[:12] == [
            "purity_before",
            "purity_after",
            "purity_decay_rate",
            "entropy_before",
            "entropy_after",
            "entropy_production_rate",
            "trace_drift",
            "dissipation_bound",
            "actual_dissipation",
            "charge_leak_rate",
            "contractivity_gap",
            "is_contractive",
        ]
        assert "purity_change_rate" not in payload
        assert "dissipator_action_norm" not in payload
        assert balance.purity_change_rate == pytest.approx(-0.1)
        assert balance.entropy_change_rate == pytest.approx(0.1)
        assert balance.dissipator_action_norm == pytest.approx(0.25)
        assert balance.frobenius_norm_loss_rate == pytest.approx(0.05)

        changed = replace(
            balance,
            purity_decay_rate=-0.2,
            actual_dissipation=0.4,
        )
        assert changed.purity_change_rate == pytest.approx(-0.2)
        assert changed.dissipator_action_norm == pytest.approx(0.4)

    def test_without_fixed_point_contractivity_is_unknown(self):
        snapshot = capture_dissipative_snapshot(_pure_state())
        balance = verify_dissipative_balance(snapshot, snapshot)
        assert not balance.contractivity_evaluated
        assert not balance.is_contractive
        assert math.isnan(balance.contractivity_gap)

    def test_trace_distance_to_fixed_point_is_used(self):
        before = capture_dissipative_snapshot(_pure_state(1))
        after = capture_dissipative_snapshot(np.diag([0.7, 0.3]))
        balance = verify_dissipative_balance(
            before, after, steady_state=_pure_state(0)
        )
        assert balance.contractivity_gap == pytest.approx(0.3)
        assert balance.contractivity_evaluated
        assert balance.is_contractive

    def test_balance_separates_state_change_and_dissipator_action(self):
        before_density = np.diag([0.8, 0.2]).astype(np.complex128)
        after_density = _amplitude_damping_state(before_density, 0.4, 0.1)
        balance = verify_dissipative_balance(
            capture_dissipative_snapshot(before_density),
            capture_dissipative_snapshot(after_density),
            dt=0.1,
            collapse_operators=_amplitude_damping_ops(0.4),
        )
        assert balance.state_change_rate > 0.0
        assert balance.dissipator_action_norm > 0.0
        assert balance.dissipation_bound_satisfied
        assert balance.unital_dissipator is False
        assert balance.purity_change_rate > 0.0
        assert balance.entropy_change_rate < 0.0
        assert balance.actual_dissipation == balance.dissipator_action_norm
        assert balance.charge_leak_rate == balance.frobenius_norm_loss_rate

    @pytest.mark.parametrize("dt", [0.0, -1.0, float("nan")])
    def test_invalid_time_step_is_rejected(self, dt):
        snapshot = capture_dissipative_snapshot(_pure_state())
        with pytest.raises(ValueError):
            verify_dissipative_balance(snapshot, snapshot, dt=dt)


class TestAnalyticalPredictions:
    @pytest.mark.parametrize("time", [0.0, 0.2, 1.0, 4.0])
    def test_exact_qubit_amplitude_damping_prediction(self, time):
        initial = np.array(
            [[0.35, 0.12 + 0.08j], [0.12 - 0.08j, 0.65]],
            dtype=np.complex128,
        )
        predicted = predict_amplitude_damping_purity(initial, 0.4, time)
        evolved = _amplitude_damping_state(initial, 0.4, time)
        actual = float(np.trace(evolved @ evolved).real)
        assert predicted == pytest.approx(actual)

    def test_equal_initial_purity_does_not_identify_amplitude_damping(self):
        ground = predict_amplitude_damping_purity(_pure_state(0), 0.5, 1.0)
        excited = predict_amplitude_damping_purity(_pure_state(1), 0.5, 1.0)
        assert ground == pytest.approx(1.0)
        assert excited < 1.0

    def test_scalar_amplitude_input_is_explicitly_deprecated(self):
        with pytest.warns(DeprecationWarning):
            result = predict_amplitude_damping_purity(0.5, 1.0, 0.0)
        assert result == pytest.approx(0.5)

    def test_historical_initial_purity_keyword_is_preserved(self):
        with pytest.warns(DeprecationWarning):
            result = predict_amplitude_damping_purity(
                initial_purity=0.5,
                gamma=1.0,
                time=0.0,
            )
        assert result == pytest.approx(0.5)

    def test_density_keyword_and_legacy_keyword_are_unambiguous(self):
        assert predict_amplitude_damping_purity(
            initial_density=_pure_state(1), gamma=0.5, time=0.0
        ) == pytest.approx(1.0)
        with pytest.raises(TypeError, match="only one"):
            predict_amplitude_damping_purity(
                initial_density=_pure_state(1),
                initial_purity=1.0,
                gamma=0.5,
                time=0.0,
            )
        with pytest.raises(ValueError, match="must be scalar"):
            predict_amplitude_damping_purity(
                initial_purity=_pure_state(1), gamma=0.5, time=0.0
            )

    @pytest.mark.parametrize("time", [0.0, 0.3, 2.0])
    def test_exact_dephasing_prediction(self, time):
        initial = _plus_state()
        predicted = predict_dephasing_purity(initial, 0.7, time)
        evolved = _dephased_state(initial, 0.7, time)
        actual = float(np.trace(evolved @ evolved).real)
        assert predicted == pytest.approx(actual)


class TestSpectralAnalysis:
    @pytest.mark.skipif(not HAS_ENGINE, reason="Mathematics backend not available")
    def test_amplitude_damping_has_unique_stationary_state(self):
        operators = _amplitude_damping_ops(0.3)
        engine = _build_qubit_engine(operators)
        result = analyze_dissipation_rates(engine.generator, dim=2)

        assert result["is_trace_preserving"]
        assert not result["has_unstable_modes"]
        assert result["n_steady_modes"] == 1
        assert result["relaxes_to_unique_state"]
        assert result["spectral_gap"] > 0.0

    @pytest.mark.skipif(not HAS_ENGINE, reason="Mathematics backend not available")
    def test_dephasing_has_stationary_manifold(self):
        engine = _build_qubit_engine(_dephasing_ops(0.5))
        result = analyze_dissipation_rates(engine.generator, dim=2)
        assert result["n_steady_modes"] == 2
        assert not result["relaxes_to_unique_state"]
        assert result["spectral_gap"] > 0.0

    def test_neutral_nonstationary_modes_preclude_relaxation_certificate(self):
        generator = np.diag([0.0, 1.0j, -1.0j, -1.0]).astype(np.complex128)
        result = analyze_dissipation_rates(generator, dim=2)
        assert result["n_steady_modes"] == 1
        assert result["has_neutral_nonstationary_modes"]
        assert result["spectral_gap"] == 0.0
        assert math.isinf(result["relaxation_time"])
        assert not result["relaxes_to_unique_state"]

    def test_steady_state_solver_rejects_non_trace_preserving_generator(self):
        with pytest.raises(ValueError, match="not trace preserving"):
            steady_state_from_generator(-np.eye(4), dim=2)

    @pytest.mark.parametrize(
        "generator,dim",
        [(np.eye(3), 2), (np.eye(4), 0), (np.full((4, 4), np.nan), 2)],
    )
    def test_invalid_generator_is_rejected(self, generator, dim):
        with pytest.raises(ValueError):
            analyze_dissipation_rates(generator, dim)


class TestClassification:
    def test_classification_is_a_change_tier_not_a_grammar_verdict(self):
        before = capture_dissipative_snapshot(_pure_state(1))
        after = capture_dissipative_snapshot(_maximally_mixed())
        balance = verify_dissipative_balance(before, after)
        result = classify_dissipative_regime(balance)

        assert result["change_tier"] == "large"
        assert result["regime"] == "decoherence"  # compatibility label
        assert result["purity_direction"] == "decreasing"
        assert not result["grammar_status_inferred"]
        assert "No U1--U6 status" in result["grammar_analog"]

    def test_purification_is_classified_by_magnitude_and_direction(self):
        before_density = np.diag([0.9, 0.1]).astype(np.complex128)
        after_density = _amplitude_damping_state(before_density, 1.0, 1.0)
        balance = verify_dissipative_balance(
            capture_dissipative_snapshot(before_density),
            capture_dissipative_snapshot(after_density),
        )
        result = classify_dissipative_regime(balance)
        assert result["purity_direction"] == "increasing"
        assert result["entropy_direction"] == "decreasing"


class TestTimeSeriesCompatibility:
    def test_signed_names_and_legacy_aliases_agree(self):
        series = DissipativeTimeSeries(
            times=[0.0, 1.0],
            purity=[0.6, 0.7],
            entropy=[0.5, 0.4],
            purity_decay_rate=[0.0, 0.1],
            entropy_production_rate=[0.0, -0.1],
            contractivity_gap=[float("nan"), 0.8],
        )
        assert series.is_contractive
        assert series.mean_purity_decay == series.mean_purity_change
        assert series.total_entropy_produced == series.total_entropy_change
        series.purity_decay_rate = [-0.2]
        assert series.purity_change_rate == [-0.2]

        payload = asdict(series)
        assert list(payload) == [
            "times",
            "purity",
            "entropy",
            "trace_drift",
            "purity_decay_rate",
            "entropy_production_rate",
            "dissipation_bound",
            "contractivity_gap",
        ]
        assert "purity_change_rate" not in payload
        changed = replace(series, entropy_production_rate=[0.2])
        assert changed.entropy_change_rate == [0.2]

    def test_infinite_contractivity_ratio_is_a_violation(self):
        series = DissipativeTimeSeries(contractivity_gap=[float("nan"), float("inf")])
        assert not series.is_contractive


@pytest.mark.skipif(not HAS_ENGINE, reason="Mathematics backend not available")
class TestTracker:
    def test_amplitude_damping_can_first_mix_then_purify(self):
        operators = _amplitude_damping_ops(0.5)
        tracker = DissipativeConservationTracker(
            _build_qubit_engine(operators),
            collapse_operators=operators,
            steady_state=_pure_state(0),
        )
        report = tracker.evolve_and_track(_pure_state(1), steps=200, dt=0.05)
        minimum_index = int(np.argmin(report.purity))

        assert 0 < minimum_index < len(report.purity) - 1
        assert report.purity[minimum_index] == pytest.approx(0.5, abs=2e-3)
        assert report.purity[-1] > report.purity[minimum_index]
        assert report.entropy[-1] < max(report.entropy)
        assert report.is_contractive
        assert max(report.trace_drift) < 1e-8

    def test_unital_dephasing_has_monotone_purity_loss(self):
        operators = _dephasing_ops(0.5)
        tracker = DissipativeConservationTracker(
            _build_qubit_engine(operators), collapse_operators=operators
        )
        report = tracker.evolve_and_track(_plus_state(), steps=30, dt=0.1)
        assert all(
            later <= earlier + 1e-10
            for earlier, later in zip(report.purity, report.purity[1:])
        )
        assert report.total_entropy_change > 0.0
        assert not report.is_contractive  # no fixed point was selected

    def test_compute_steady_state_and_tracker_reset(self):
        operators = _amplitude_damping_ops(0.3)
        engine = _build_qubit_engine(operators)
        tracker = DissipativeConservationTracker(engine, collapse_operators=operators)
        stationary = tracker.compute_steady_state()
        assert np.allclose(stationary, _pure_state(0), atol=1e-8)
        assert np.allclose(
            steady_state_from_generator(engine.generator, 2), _pure_state(0), atol=1e-8
        )

        first = tracker.evolve_and_track(_pure_state(1), steps=2, dt=0.1)
        second = tracker.evolve_and_track(_pure_state(1), steps=1, dt=0.1)
        assert len(first.times) == 3
        assert len(second.times) == 2
        assert tracker.latest_balance is not None

    def test_nonstationary_reference_state_is_rejected(self):
        operators = _amplitude_damping_ops(0.3)
        engine = _build_qubit_engine(operators)
        with pytest.raises(ValueError, match="not stationary"):
            DissipativeConservationTracker(
                engine,
                collapse_operators=operators,
                steady_state=_pure_state(1),
            )

    @pytest.mark.parametrize("steps,dt", [(-1, 0.1), (1, 0.0), (True, 0.1)])
    def test_invalid_evolution_arguments(self, steps, dt):
        operators = _amplitude_damping_ops(0.2)
        tracker = DissipativeConservationTracker(
            _build_qubit_engine(operators), collapse_operators=operators
        )
        with pytest.raises(ValueError):
            tracker.evolve_and_track(_pure_state(1), steps=steps, dt=dt)
