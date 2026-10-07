"""Original-value admission and invocation-local dissipative diagnostics."""

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from tnfr.physics import dissipative_conservation as dc

_INVALID_SCALARS = [
    True,
    np.bool_(False),
    "0",
    0j,
    float("nan"),
    float("inf"),
    Fraction(1, 10**400),
]


@pytest.mark.parametrize("bad", _INVALID_SCALARS)
@pytest.mark.parametrize("consumer", ["density", "collapse", "generator"])
def test_matrix_consumers_admit_original_components(bad, consumer):
    # A genuine complex component is valid in matrices, unlike real scalars.
    if isinstance(bad, complex):
        bad = complex(0, float("inf"))
    matrix = [[1, bad], [bad, 0]]
    with pytest.raises(ValueError, match="finite representable"):
        if consumer == "density":
            dc.capture_dissipative_snapshot(matrix)
        elif consumer == "collapse":
            dc.compute_dissipator_action(np.diag([1.0, 0.0]), [matrix])
        else:
            dc.analyze_dissipation_rates([[bad]], dim=1)


@pytest.mark.parametrize("bad", _INVALID_SCALARS)
@pytest.mark.parametrize(
    "consumer", ["atol", "tolerance", "dt", "purity", "gamma", "time"]
)
def test_real_parameters_reject_loss_and_coercible_nonreal_values(bad, consumer):
    density = np.diag([1.0, 0.0])
    with pytest.raises(ValueError):
        if consumer == "atol":
            dc.capture_dissipative_snapshot(density, atol=bad)
        elif consumer == "tolerance":
            dc.is_unital_dissipator([], tolerance=bad)
        elif consumer == "dt":
            snapshot = dc.capture_dissipative_snapshot(density)
            dc.verify_dissipative_balance(snapshot, snapshot, dt=bad)
        elif consumer == "purity":
            dc.compute_dissipation_bound([], bad)
        elif consumer == "gamma":
            dc.predict_amplitude_damping_purity(density, bad, 0.1)
        else:
            dc.predict_dephasing_purity(density, 0.1, bad)


@pytest.mark.parametrize("bad", _INVALID_SCALARS)
def test_tracker_timestamp_admission_precedes_mutation(bad):
    engine = SimpleNamespace(
        generator=np.zeros((4, 4)), hilbert_space=SimpleNamespace(dimension=2)
    )
    tracker = dc.DissipativeConservationTracker(engine)
    with pytest.raises(ValueError):
        tracker.record(np.diag([1.0, 0.0]), t=bad)
    assert tracker.report().times == []
    assert tracker.latest_balance is None
    tracker.record(np.diag([1.0, 0.0]), t=Fraction(-1, 2))
    assert tracker.report().times == [-0.5]


def test_exact_components_and_noncontiguous_density_keep_owned_snapshot():
    exact = [[Fraction(3, 4), Fraction(1, 8)], [Fraction(1, 8), Fraction(1, 4)]]
    expected = np.asarray(exact, dtype=np.complex128)
    buffer = np.zeros((4, 4), dtype=np.complex128)
    view = buffer[::2, ::2]
    view[:] = expected
    snapshot = dc.capture_dissipative_snapshot(view)
    exact_snapshot = dc.capture_dissipative_snapshot(exact)
    assert np.array_equal(snapshot.density, exact_snapshot.density)
    assert snapshot.purity == pytest.approx(21 / 32)
    assert not np.shares_memory(snapshot.density, view)
    assert snapshot.density.flags.writeable
    view[:] = 0
    np.testing.assert_array_equal(snapshot.density, expected)


def test_balance_readmits_mutated_densities_and_ignores_cached_diagnostics():
    before = dc.capture_dissipative_snapshot(np.eye(2) / 2)
    after = dc.capture_dissipative_snapshot(np.diag([0.4, 0.6]))
    before.density[:] = np.diag([0.8, 0.2])
    before.eigenvalues[:] = np.nan
    before = replace(before, purity=float("nan"), von_neumann_entropy=-100.0)
    result = dc.verify_dissipative_balance(before, after, dt=0.5)
    assert result.purity_before == pytest.approx(0.68)
    assert result.purity_change_rate == pytest.approx(-0.32)
    before.density[:] = np.eye(2)
    with pytest.raises(ValueError, match="unit trace"):
        dc.verify_dissipative_balance(before, after)


def test_balance_reuses_only_current_call_spectra_and_operator_norms(monkeypatch):
    before = dc.capture_dissipative_snapshot(np.diag([0.0, 1.0]))
    after = dc.capture_dissipative_snapshot(np.diag([0.1, 0.9]))
    operator = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.complex128)
    eigenvalues = np.linalg.eigvalsh
    norm = np.linalg.norm
    calls = {"spectrum": 0, "spectral_norm": 0}

    def counted_spectrum(*args, **kwargs):
        calls["spectrum"] += 1
        return eigenvalues(*args, **kwargs)

    def counted_norm(*args, **kwargs):
        if kwargs.get("ord") == 2:
            calls["spectral_norm"] += 1
        return norm(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", counted_spectrum)
    monkeypatch.setattr(np.linalg, "norm", counted_norm)
    first = dc.verify_dissipative_balance(before, after, collapse_operators=[operator])
    assert calls == {"spectrum": 2, "spectral_norm": 1}
    assert first.dissipator_action_norm == pytest.approx(np.sqrt(2) / 4)
    assert first.dissipation_bound == pytest.approx(0.5)
    assert first.instantaneous_purity_change_rate == pytest.approx(-0.5)
    assert first.unital_dissipator is False

    operator[0, 1] = 1.0
    second = dc.verify_dissipative_balance(before, after, collapse_operators=[operator])
    assert calls == {"spectrum": 4, "spectral_norm": 2}
    assert second.dissipation_bound == pytest.approx(2.0)
    assert second.dissipator_action_norm == pytest.approx(np.sqrt(2))
    operator[0, 1] = np.nan
    with pytest.raises(ValueError):
        dc.verify_dissipative_balance(before, after, collapse_operators=[operator])


def test_bound_uses_spectral_operator_norm_instead_of_frobenius_norm():
    # ||I||_2=1 while ||I||_F=sqrt(2); the identity dissipator vanishes.
    operator = np.eye(2)
    snapshot = dc.capture_dissipative_snapshot(np.diag([1.0, 0.0]))
    result = dc.verify_dissipative_balance(
        snapshot, snapshot, collapse_operators=[operator]
    )
    assert result.dissipation_bound == pytest.approx(2.0)
    assert result.dissipator_action_norm == pytest.approx(0.0)
    assert result.unital_dissipator is True


def test_custom_public_hooks_retain_order_and_live_operator_mutations(monkeypatch):
    before = dc.capture_dissipative_snapshot(np.diag([0.0, 1.0]))
    after = dc.capture_dissipative_snapshot(np.diag([0.1, 0.9]))
    operator = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.complex128)
    original_capture = dc.capture_dissipative_snapshot
    original_action = dc.compute_dissipator_action
    original_bound = dc.compute_dissipation_bound
    original_unital = dc.is_unital_dissipator
    calls = []

    def capture(*args, **kwargs):
        calls.append("capture")
        return original_capture(*args, **kwargs)

    def action(density, operators):
        calls.append("action")
        operators[0][0, 1] = 1.0
        return original_action(density, operators)

    def bound(*args, **kwargs):
        calls.append("bound")
        return original_bound(*args, **kwargs)

    def unital(*args, **kwargs):
        calls.append("unital")
        return original_unital(*args, **kwargs)

    monkeypatch.setattr(dc, "capture_dissipative_snapshot", capture)
    monkeypatch.setattr(dc, "compute_dissipator_action", action)
    monkeypatch.setattr(dc, "compute_dissipation_bound", bound)
    monkeypatch.setattr(dc, "is_unital_dissipator", unital)
    result = dc.verify_dissipative_balance(before, after, collapse_operators=[operator])
    assert calls == ["capture", "capture", "action", "bound", "unital"]
    assert result.dissipation_bound == pytest.approx(2.0)
    assert result.dissipator_action_norm == pytest.approx(np.sqrt(2))
    assert operator[0, 1] == 1.0


def test_density_and_purity_errors_precede_collapse_materialization():
    class MustNotIterate:
        def __iter__(self):
            raise AssertionError(
                "collapse sequence was consumed before state admission"
            )

    with pytest.raises(ValueError):
        dc.compute_dissipation_bound(MustNotIterate(), True)
    with pytest.raises(ValueError, match="unit trace"):
        dc.compute_dissipator_action(np.eye(2), MustNotIterate())


@pytest.mark.parametrize("consumer", ["bound", "purity_bound", "unital", "balance"])
def test_later_operator_conversion_does_not_leave_stale_retained_norm(consumer):
    first = np.array([[0.0, 0.5], [0.0, 0.0]], dtype=np.complex128)

    class MutatingArray:
        def __array__(self, dtype=None, copy=None):
            first[0, 1] = 1.0
            return np.zeros((2, 2), dtype=dtype)

    operators = [first, MutatingArray()]
    density = np.diag([0.0, 1.0])
    if consumer == "bound":
        assert dc.compute_dissipation_bound(operators, 1.0) == pytest.approx(2.0)
    elif consumer == "purity_bound":
        assert dc.compute_purity_decay_bound(operators, density) == pytest.approx(4.0)
    elif consumer == "unital":
        # Numerical residual sqrt(2), tolerance scale 0.8*(1+||L||_2**2)=1.6.
        # The obsolete pre-conversion norm would instead set the threshold to 1.
        assert dc.is_unital_dissipator(operators, tolerance=0.8)
    else:
        snapshot = dc.capture_dissipative_snapshot(density)
        result = dc.verify_dissipative_balance(
            snapshot, snapshot, collapse_operators=operators
        )
        assert result.dissipation_bound == pytest.approx(2.0)
        assert result.dissipator_action_norm == pytest.approx(np.sqrt(2))
