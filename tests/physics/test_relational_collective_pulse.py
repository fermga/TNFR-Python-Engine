"""Static causal poles, quotient defects and explicit null controls."""

import hashlib
import math
from fractions import Fraction as Q

import numpy as np
import pytest

from benchmarks import relational_collective_pulse as study
from tnfr.utils.io import json_loads


def _complex(pairs):
    return np.array([complex(*pair) for pair in pairs])


def test_damped_oscillator_has_independent_analytic_transfer_and_attenuation():
    damping, omega = 0.25, 2.0
    matrix = np.array([[-damping, -omega], [omega, -damping]])
    report = study._modal_data(matrix, (1, 0), ((0, 1),))
    s = study.RESOLVENT_POINT
    expected = omega / ((s + damping) ** 2 + omega**2)
    assert _complex(report["direct_resolvent"])[0] == pytest.approx(expected, abs=1e-15)
    assert _complex(report["modal_resolvent"])[0] == pytest.approx(expected, abs=1e-15)
    for row in report["poles"]:
        value = complex(*row["pole"])
        assert value.real == pytest.approx(-damping)
        assert abs(value.imag) == pytest.approx(omega)
        assert row["decay_time"] == pytest.approx(1 / damping)
        assert row["period"] == pytest.approx(2 * math.pi / omega)
        assert row["cycles_per_decay_time"] == pytest.approx(
            omega / (2 * math.pi * damping)
        )
        assert _complex(row["residues"])[0] == pytest.approx(
            -0.5j if value.imag > 0 else 0.5j
        )


def test_degenerate_disconnected_oscillators_can_have_extended_modes_but_no_transfer():
    block = np.array([[-0.25, -2.0], [2.0, -0.25]])
    matrix = np.kron(np.eye(2), block)
    initial, output = np.array([1, 0, 0, 0]), np.array([[0, 0, 1, 0]])
    report = study._modal_data(matrix, initial, output)
    assert _complex(report["direct_resolvent"]) == pytest.approx([0j], abs=1e-15)
    assert _complex(report["modal_resolvent"]) == pytest.approx([0j], abs=1e-15)
    assert all(not row["individually_separated"] for row in report["poles"])
    poles, local = np.linalg.eig(block)
    extended = np.kron(np.array([[1, 1], [1, -1]]) / math.sqrt(2), local)
    coefficients = (output @ extended) * (np.linalg.inv(extended) @ initial)[None, :]
    assert np.min(np.abs(extended)) > 0
    assert np.max(np.abs(coefficients)) > 0.2
    transfer = np.sum(
        coefficients / (study.RESOLVENT_POINT - np.tile(poles, 2)), axis=1
    )
    assert transfer == pytest.approx([0j], abs=1e-15)


@pytest.fixture(scope="module")
def report():
    def forbidden(*args, **kwargs):
        pytest.fail("static modal instrument attempted a time step")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study.tangent_owner, "step_relational_exchange", forbidden)
        return study.analyze_collective_pulse()


def test_difference_maps_remove_declared_offsets_without_selecting_zero_eigenvalues(
    report,
):
    d, lift = np.array(report["difference_map"], float), np.array(report["lift"], float)
    np.testing.assert_array_equal(d @ lift, np.eye(20))
    shifts = np.zeros((22, 2))
    shifts[:11, 0] = 1
    shifts[11:, 1] = 1
    np.testing.assert_array_equal(d @ shifts, np.zeros((20, 2)))
    matrix = np.array(report["generator"], float)
    assert np.array(report["quotient_generator"], float) == pytest.approx(
        d @ matrix @ lift, abs=2e-16
    )
    assert len(report["modal"]["poles"]) == len(report["quotient_generator"]) == 20
    assert report["quotient_identity_residual"] != ((Q(0),) * 22,) * 20
    assert report["full_vs_quotient_resolvent_error_linf"] < 1e-12


def test_native_causal_resolvent_is_reassembled_from_all_nonnormal_poles(report):
    matrix = np.array(report["quotient_generator"], float)
    initial = np.array(report["quotient_input"], float)
    output = np.array(report["quotient_observations"], float)
    expected = output @ np.linalg.solve(
        study.RESOLVENT_POINT * np.eye(20) - matrix, initial
    )
    reconstructed = sum(
        (
            _complex(row["residues"]) / (study.RESOLVENT_POINT - complex(*row["pole"]))
            for row in report["modal"]["poles"]
        ),
        np.zeros(6, complex),
    )
    assert reconstructed == pytest.approx(expected, rel=0, abs=1e-12)
    assert report["modal"]["resolvent_reassembly_error_linf"] < 1e-12
    tolerance = report["modal"]["residue_zero_tolerance"]
    shared = [
        row
        for row in report["modal"]["poles"]
        if row["pole"][1] > 0
        and row["individually_separated"]
        and row["numerically_oscillatory"]
        and max(abs(_complex(row["residues"])[:2])) > tolerance
        and max(abs(_complex(row["residues"])[4:])) > tolerance
    ]
    assert shared
    assert all(row["pole"][0] < 0 for row in shared)


def test_full_state_null_blocks_every_route_but_frozen_mediator_does_not(report):
    control = report["controls"]
    null, groups = control["block_diagonal_generator"], control["groups"]
    assert control["cross_blocks_exactly_zero"]
    assert all(
        null[i][j] == 0 for i in range(22) for j in range(22) if groups[i] != groups[j]
    )
    assert _complex(control["null_receiver_resolvent"]) == pytest.approx(
        [0j, 0j], abs=1e-15
    )
    assert any(control["frozen_mediator_receiver_second_moment"])
    vector = report["input"]
    for _ in range(4):
        assert all(vector[i] == 0 for i in range(22) if groups[i] != 0)
        vector = tuple(
            sum((value * x for value, x in zip(row, vector)), Q(0)) for row in null
        )


def test_ideal_positive_cube_trace_is_separate_from_materialized_spectrum(report):
    certificate = report["ideal_trace_cube_certificate"]
    lower, upper = map(float, certificate["trace_cube"])
    assert certificate["positive_lower_bound"] and 0 < lower < upper
    matrix = np.array(report["generator"], float)
    observed = float(np.trace(matrix @ matrix @ matrix))
    assert lower - 3e-12 <= observed <= upper + 3e-12
    assert "no_causal_residue" in certificate["scope"]


def test_snapshot_and_partial_source_fingerprints_remain_explicit(report):
    assert (
        report["snapshot"]["geometry_status"]
        == "rational_midpoint_not_an_exact_equilibrium"
    )
    assert "partial" in report["source_scope"]
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((study.ROOT / path).read_bytes()).hexdigest() == digest
    encoded = json_loads(study._encoded(report))
    assert encoded["snapshot"]["root_bracket"]["lower"]["denominator"] > 0


def test_cli_uses_exclusive_shared_json_output(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "analyze_collective_pulse", lambda: {"exact": Q(1, 7)})
    destination = tmp_path / "nested" / "result.json"
    assert study.main(["--output", str(destination)]) == 0
    before = destination.read_bytes()
    assert json_loads(before)["exact"] == {"numerator": 1, "denominator": 7}
    with pytest.raises(FileExistsError, match="replace"):
        study.main(["--output", str(destination)])
    assert destination.read_bytes() == before
