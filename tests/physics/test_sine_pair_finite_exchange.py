"""Finite pair exchange checked against exact original-node Cartesian jets.

The jets below are local algebra, not trajectory samples or numerical
certificates. The finite enclosure belongs to the separate analytic theorem.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._sine_pair_oracles import (
    EDGES,
    ONE,
    ORIENTATIONS,
    PAIRS,
    ZERO,
    _conj,
    _current_coefficient,
    _fine_jets,
    _integrated_current_remainder,
    _mul,
    _preparation,
)
from tnfr.physics.relational_sine_scale import assess_sine_pair_finite_exchange
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils import io

ROTATION = (Q(399, 401), Q(40, 401))
HORIZON = Q(1, 20)


def _storage_coefficients(x, z, degree):
    form = sum(
        (
            (x[i][r] - x[j][r]) * (x[i][degree - r] - x[j][degree - r]) / 2
            for i, j in EDGES
            for r in range(degree + 1)
        ),
        Q(0),
    )
    phase = Q(len(EDGES) if degree == 0 else 0) - sum(
        (
            _mul(_conj(z[i][r]), z[j][degree - r])[0]
            for i, j in EDGES
            for r in range(degree + 1)
        ),
        Q(0),
    )
    return form, phase


@pytest.fixture(scope="module")
def frozen_report():
    return assess_sine_pair_finite_exchange(
        phase_rotation=ROTATION, horizon_tau=HORIZON
    )


@pytest.mark.parametrize("rotation", (ROTATION, (Q(3, 5), Q(4, 5)), (Q(0), Q(1))))
def test_integrated_leading_term_is_the_actual_fine_edge_current_jet(rotation):
    report = assess_sine_pair_finite_exchange(
        phase_rotation=rotation, horizon_tau=HORIZON
    )
    coefficients = []
    for orientation, reported_phasors in zip(
        ORIENTATIONS, report.comparison_initial_phasors
    ):
        phasors = _preparation(rotation, orientation)
        assert reported_phasors == phasors
        _, z = _fine_jets(phasors)
        assert _current_coefficient(z, 0, 0, 1) == 0
        assert _current_coefficient(z, 1, 0, 1) == 0
        assert _current_coefficient(z, 3, 0, 1) == 0
        coefficients.append(_current_coefficient(z, 2, 0, 1))
    # Integrate the actual second-order current coefficient in tau. This
    # also checks the clock conversion: dt=pi*d_tau cancels I_t's 1/pi.
    expected = (coefficients[0] - coefficients[1]) * HORIZON**3 / 3
    assert expected != 0
    assert report.integrated_current_difference_leading_term == expected
    assert report.directed_block_edge == (0, 1)
    assert report.orientation_order == ("real_antipodal", "imaginary_antipodal")
    assert report.clock == "tau=t/pi"


def test_frozen_preparations_have_identical_interfaces_and_balanced_fine_storage(
    frozen_report,
):
    reports = frozen_report.comparison_initial_states
    for field in (
        "form_means",
        "resultants",
        "internal_form_squared",
        "form_phase_moments",
    ):
        assert getattr(reports[0], field) == getattr(reports[1], field)
    assert reports[0].phase_products[0] != reports[1].phase_products[0]
    for orientation, state in zip(ORIENTATIONS, reports):
        x, z = _fine_jets(_preparation(ROTATION, orientation))
        form_zero, phase_zero = _storage_coefficients(x, z, 0)
        assert state.form_storage == form_zero == 0
        assert state.phase_storage == phase_zero
        assert state.storage == phase_zero
        assert _storage_coefficients(x, z, 2)[0] > 0
        for degree in range(1, 5):
            # Nontrivial O(tau^2) exchange of full fine form/phase storage
            # cancels, rather than only their zero instantaneous rates.
            assert sum(_storage_coefficients(x, z, degree)) == 0
            assert sum((node[degree] for node in x), Q(0)) == 0
        for receiver in range(5):
            for degree in range(4):
                observed = (
                    (degree + 1)
                    * sum((x[i][degree + 1] for i in PAIRS[receiver]), Q(0))
                    / 2
                )
                incoming = sum(
                    (
                        _current_coefficient(z, degree, receiver, source)
                        for source in ((receiver - 1) % 5, (receiver + 1) % 5)
                    ),
                    Q(0),
                )
                assert observed == incoming
                # Degree/capacity weighted cut uses weight four per node.
                fine_cut = (
                    4
                    * (degree + 1)
                    * sum((x[i][degree + 1] for i in PAIRS[receiver]), Q(0))
                )
                assert fine_cut == 8 * incoming
        for degree in range(5):
            assert _current_coefficient(z, degree, 0, 1) == -_current_coefficient(
                z, degree, 1, 0
            )
    assert reports[0].storage == reports[1].storage


@pytest.mark.parametrize("orientation", (*ORIENTATIONS, (Q(3, 5), Q(4, 5))))
def test_symmetry_preserving_control_is_an_exact_fine_equilibrium(
    orientation, frozen_report
):
    phasors = _preparation(ROTATION, orientation, control=True)
    x, z = _fine_jets(phasors, order=1)
    # Both full fine rows vanish, so uniqueness supplies all-time silence;
    # a finite zero jet of an otherwise evolving state is not used as proof.
    assert all(node[1] == 0 for node in x)
    assert all(node[1] == ZERO for node in z)
    for receiver in range(5):
        for source in ((receiver - 1) % 5, (receiver + 1) % 5):
            assert _current_coefficient(z, 0, receiver, source) == 0
    assert frozen_report.control_integrated_currents == (Q(0), Q(0))
    assert frozen_report.control_integrated_current_difference == 0
    for direction, retained in zip(ORIENTATIONS, frozen_report.control_initial_phasors):
        assert retained == _preparation(ROTATION, direction, control=True)


def test_frozen_finite_bound_excludes_zero_and_encloses_the_analytic_budget(
    frozen_report,
):
    lead = frozen_report.integrated_current_difference_leading_term
    bound = frozen_report.integrated_current_difference_bounds
    remainder = frozen_report.integrated_current_difference_remainder_upper_bound
    assert remainder > 0
    assert bound.lo <= lead - remainder < lead + remainder <= bound.hi < 0
    assert frozen_report.status == "certified_negative"
    assert frozen_report.unavailable_reason is None
    assert frozen_report.horizon_tau == HORIZON
    assert frozen_report.phase_rotation == ROTATION


@pytest.mark.parametrize("horizon", (HORIZON, Q(3)))
def test_remainder_integrates_the_full_product_rule_bound(horizon):
    # Independent arithmetic of the theorem's fine-row derivative bounds.
    # Coefficients are in increasing powers of tau. Do not copy the final
    # 8/15 and 8/105 constants from the report-producing implementation.
    # Taylor's integral kernel (tau-s)^2/2 followed by current integration
    # supplies m!/(m+4)! for each s^m. Add the two orientation errors.
    expected = 2 * _integrated_current_remainder(horizon)
    report = assess_sine_pair_finite_exchange(
        phase_rotation=ROTATION, horizon_tau=horizon
    )
    assert report.integrated_current_difference_remainder_upper_bound == expected


def test_complex_conjugation_reverses_the_finite_exchange_orientation(frozen_report):
    reflected = assess_sine_pair_finite_exchange(
        phase_rotation=_conj(ROTATION), horizon_tau=HORIZON
    )
    assert reflected.status == "certified_positive"
    assert reflected.integrated_current_difference_bounds.lo > 0
    assert (
        reflected.integrated_current_difference_bounds
        == -frozen_report.integrated_current_difference_bounds
    )
    assert (
        reflected.integrated_current_difference_remainder_upper_bound
        == frozen_report.integrated_current_difference_remainder_upper_bound
    )


@pytest.mark.parametrize("rotation,horizon", ((ONE, HORIZON), (ROTATION, Q(1))))
def test_nonseparating_bounds_abstain_without_changing_preparation_or_horizon(
    rotation, horizon
):
    report = assess_sine_pair_finite_exchange(
        phase_rotation=rotation, horizon_tau=horizon
    )
    assert 0 in report.integrated_current_difference_bounds
    assert report.status == "unavailable"
    assert report.unavailable_reason == "finite_exchange_interval_contains_zero"
    assert report.phase_rotation == rotation
    assert report.horizon_tau == horizon


def test_tiny_horizon_retains_exact_response_but_zero_touching_rounded_bound_abstains():
    tiny = Q(1, 2**1100)
    report = assess_sine_pair_finite_exchange(phase_rotation=ROTATION, horizon_tau=tiny)
    assert report.horizon_tau == tiny > 0
    assert report.integrated_current_difference_leading_term < 0
    assert report.integrated_current_difference_remainder_upper_bound > 0
    assert report.integrated_current_difference_bounds.hi == 0
    assert report.status == "unavailable"


def test_tiny_exact_phase_rotation_is_not_normalized_into_the_invisible_family():
    slope = Q(1, 2**1100)
    rotation = ((1 - slope**2) / (1 + slope**2), 2 * slope / (1 + slope**2))
    report = assess_sine_pair_finite_exchange(
        phase_rotation=rotation, horizon_tau=HORIZON
    )
    assert report.phase_rotation == rotation != ONE
    assert report.integrated_current_difference_leading_term < 0
    assert report.status == "unavailable"


@pytest.mark.parametrize(
    "bad", (True, False, float("nan"), float("inf"), -float("inf"), "1", 1j)
)
def test_nonphysical_scalars_reject_before_any_bound_arithmetic(bad):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_pair_finite_exchange(phase_rotation=ROTATION, horizon_tau=bad)
    with pytest.raises((TypeError, ValueError)):
        assess_sine_pair_finite_exchange(phase_rotation=(bad, 0), horizon_tau=HORIZON)


@pytest.mark.parametrize("horizon", (Q(0), Q(-1, 20), Q(-1, 2**1100)))
def test_horizon_requires_strictly_positive_admitted_time(horizon):
    with pytest.raises(ValueError):
        assess_sine_pair_finite_exchange(phase_rotation=ROTATION, horizon_tau=horizon)


@pytest.mark.parametrize(
    "rotation",
    ((), (1,), (1, 0, 0), {0, 1}, (0, 0), (2, 0), (0.6, 0.8), (1, Q(1, 2**1100))),
)
def test_rotation_requires_ordered_exact_unit_phase_data(rotation):
    with pytest.raises((TypeError, ValueError)):
        assess_sine_pair_finite_exchange(phase_rotation=rotation, horizon_tau=HORIZON)


def test_ordered_phase_generator_is_captured_once(frozen_report):
    assert (
        assess_sine_pair_finite_exchange(
            phase_rotation=iter(ROTATION), horizon_tau=HORIZON
        )
        == frozen_report
    )


def test_sdk_and_direct_projection_keep_exact_finite_evidence(tmp_path, frozen_report):
    direct = frozen_report.to_dict()
    exported = relational_report_to_dict(frozen_report)
    assert direct["schema"] == "tnfr.sine-pair-finite-exchange.v1"
    assert exported["report"] == direct["report"]
    assert exported["report"]["phase_rotation"] == [
        {"numerator": value.numerator, "denominator": value.denominator}
        for value in ROTATION
    ]
    for name in (
        "integrated_current_difference_leading_term",
        "integrated_current_difference_remainder_upper_bound",
        "horizon_tau",
    ):
        value = getattr(frozen_report, name)
        assert exported["report"][name] == {
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    target = tmp_path / "finite-exchange.json"
    export_to_json(exported, target)
    assert io.json_loads(target.read_text(encoding="utf-8")) == exported


def test_export_uses_atomic_replace_and_preserves_prior_evidence_on_failure(
    tmp_path, monkeypatch, frozen_report
):
    target = tmp_path / "finite-exchange.json"
    original = b'{"previous_evidence": true}\n'
    target.write_bytes(original)

    def fail_replace(source, destination):
        assert source.parent == target.parent
        assert destination == target
        assert (
            io.json_loads(source.read_text(encoding="utf-8"))["report"]["status"]
            == "certified_negative"
        )
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        export_to_json(relational_report_to_dict(frozen_report), target)
    assert target.read_bytes() == original
    assert tuple(tmp_path.iterdir()) == (target,)
