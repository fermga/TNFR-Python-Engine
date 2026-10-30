"""Heat-filter, noncommuting and preserved-protocol form tracking controls.

Pure kernel tests use synthetic supplied linear comparisons only. The actual
family fixture is evaluated after archival and the first reserved assessment.
"""

from fractions import Fraction as Q
from inspect import signature
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tests.physics.test_sine_formed_class_contact import _mp
from tests.physics.test_sine_port_relaxation import INPUTS
from tnfr.physics import relational_sine_port_form_tracking as owner
from tnfr.physics import relational_sine_port_relaxation as baseline_owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads


def _assess(**changes):
    return owner.assess_sine_port_form_tracking(**(INPUTS | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 90
    return context


@pytest.fixture(scope="module", autouse=True)
def no_probe_trajectory_or_previous_composition():
    from tnfr.dynamics import relational
    from tnfr.physics import (
        relational_sine_forecast,
        relational_sine_formed_classes,
        relational_sine_port_composition,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "form tracking must reconstruct unprobed primitives without a trajectory"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (relational, "step_relational_exchange"),
            (relational, "_advance"),
            (relational_sine_forecast, "bound_sine_flow"),
            (relational_sine_formed_classes, "assess_sine_formed_class_response"),
            (relational_sine_port_composition, "assess_sine_port_composition"),
        ):
            patch.setattr(module, name, forbidden)
        yield


class TestStaticHeatKernel:
    @pytest.mark.parametrize("gap", [Q(7, 30), Q(3, 100), Q(1, 1000), Q(12)])
    def test_exact_rational_gain_dominates_continuous_spectral_envelope(self, gap, mp):
        report = owner._heat_form_bounds(
            gap=gap, forcing=Q(1), initial_norm=Q(0), gamma_upper=Q(1, 1000)
        )
        mu, maximum = _mp(mp, gap) / 6, mp.mpf(2)
        split1, split2 = 1 / maximum, 1 / mu
        first = mp.quad(lambda t: maximum * mp.exp(-maximum * t), [0, split1])
        middle = mp.log(split2 / split1) / mp.e
        last = mp.exp(-mu * split2)
        exact_envelope = first + middle + last
        assert mp.almosteq(exact_envelope, 1 + mp.log(maximum / mu) / mp.e)
        assert exact_envelope <= _mp(mp, report.heat_integral_upper_bound)
        assert (
            report.derivative_filter_gain_upper_bound
            == 1 + report.heat_integral_upper_bound
        )
        assert mp.log(2) < mp.mpf(7) / 10 and 1 / mp.e < mp.mpf(3) / 8

    @pytest.mark.parametrize("bands", [1, 6, 9, 100])
    @pytest.mark.parametrize("direction,extra", [(-1, 1), (0, 0), (1, 0)])
    def test_dyadic_band_count_preserves_exact_boundary(self, bands, direction, extra):
        gap = Q(12, 2**bands) * (1 + direction * Q(1, 2**500))
        report = owner._heat_form_bounds(
            gap=gap, forcing=Q(0), initial_norm=Q(0), gamma_upper=Q(1, 1000)
        )
        assert report.dyadic_band_count == bands + extra
        assert report.spectral_ratio_upper_bound <= 2**report.dyadic_band_count
        assert report.spectral_ratio_upper_bound > Q(2) ** (
            report.dyadic_band_count - 1
        )

    def test_small_gain_closure_retains_initial_exchange(self):
        gamma, gap, norm, forcing = Q(1, 100), Q(1, 10), Q(1, 500), Q(1, 70)
        report = owner._heat_form_bounds(
            gap=gap, forcing=forcing, initial_norm=norm, gamma_upper=gamma
        )
        assert report.certified and report.loop_margin > 0
        expected_rhs = (
            norm
            + gamma * 2 * (1 + gamma) * norm / gap
            + gamma * report.derivative_filter_gain_upper_bound * forcing / gap
        )
        assert report.form_norm_upper_bound * report.loop_margin == expected_rhs
        assert report.initial_form_contribution_upper_bound == norm
        assert report.initial_joint_contribution_upper_bound > 0
        zero_force = owner._heat_form_bounds(
            gap=gap, forcing=Q(0), initial_norm=norm, gamma_upper=gamma
        )
        assert zero_force.form_norm_upper_bound > norm

    def test_invalid_loop_is_unavailable_without_division(self):
        report = owner._heat_form_bounds(
            gap=Q(1, 1000), forcing=Q(1), initial_norm=Q(1), gamma_upper=Q(1, 10)
        )
        assert report.loop_gain_upper_bound > 1 and not report.certified
        assert report.form_norm_upper_bound is None

    def test_scalar_step_and_initial_form_have_closed_form_control(self, mp):
        lam, stiffness, gamma, force = (
            mp.mpf(1),
            mp.mpf(3) / 4,
            mp.mpf(1) / 100,
            mp.mpf(1) / 10,
        )
        roots = (
            (-lam + mp.sqrt(lam**2 - 4 * gamma**2 * lam * stiffness)) / 2,
            (-lam - mp.sqrt(lam**2 - 4 * gamma**2 * lam * stiffness)) / 2,
        )
        report = owner._heat_form_bounds(
            gap=Q(1), forcing=Q(1, 10), initial_norm=Q(0), gamma_upper=Q(1, 100)
        )
        for time in (0, 1, 10, 1000, 10000):
            exact_x = (
                gamma
                * force
                * (mp.exp(roots[0] * time) - mp.exp(roots[1] * time))
                / (roots[0] - roots[1])
            )
            assert abs(exact_x) <= _mp(mp, report.form_norm_upper_bound)
        initial = owner._heat_form_bounds(
            gap=Q(1), forcing=Q(0), initial_norm=Q(1, 1000), gamma_upper=Q(1, 100)
        )
        generator = mp.matrix([[-lam, -gamma * stiffness], [gamma * lam, 0]])
        for state in (mp.matrix([mp.mpf("0.001"), 0]), mp.matrix([0, mp.mpf("0.001")])):
            for time in (0, 1, 10000):
                assert abs((mp.expm(generator * time) * state)[0]) <= _mp(
                    mp, initial.form_norm_upper_bound
                )

    def test_noncommuting_operators_and_ordered_heat_identity(self, mp):
        a = mp.matrix([[1, 0], [0, 2]])
        b = mp.matrix([[1, mp.mpf(1) / 4], [mp.mpf(1) / 4, mp.mpf(3) / 2]])
        assert mp.norm(a * b - b * a) > 0
        gamma, eta = mp.mpf("0.01"), mp.mpf("0.0001")
        generator = mp.matrix(4)
        for i in range(2):
            for j in range(2):
                generator[i, j] = -a[i, j]
                generator[i, j + 2] = -gamma * b[i, j]
                generator[i + 2, j] = gamma * a[i, j]
        report = owner._heat_form_bounds(
            gap=Q(1), forcing=Q(1, 10), initial_norm=Q(1, 1000), gamma_upper=Q(1, 100)
        )
        state = mp.matrix([mp.mpf("0.001"), 0, 0, mp.mpf("0.001")])
        for duration, drive in (
            (mp.mpf("0.2"), (1, 0)),
            (mp.mpf(3), (0, -1)),
            (mp.mpf(1000), (-1, 0)),
        ):
            augmented = mp.matrix(5)
            for i in range(4):
                for j in range(4):
                    augmented[i, j] = generator[i, j]
            for i in range(2):
                augmented[i, 4] = gamma * mp.mpf("0.1") * drive[i]
            advanced = mp.expm(augmented * duration) * mp.matrix([*state, 1])
            state = mp.matrix(advanced[:4])
            assert mp.sqrt(state[0] ** 2 + state[1] ** 2) <= _mp(
                mp, report.form_norm_upper_bound
            )

        # The initial joint-coordinate heat term keeps the operator order.
        time = mp.mpf("0.7")
        y0 = mp.matrix([mp.mpf("0.2"), mp.mpf("-0.1")])
        ordered = mp.matrix(
            [
                mp.quad(
                    lambda s: (
                        mp.expm(-a * (time - s)) * b * mp.expm(-eta * b * s) * y0
                    )[i],
                    [0, time],
                )
                for i in range(2)
            ]
        )
        wrong = mp.matrix(
            [
                mp.quad(
                    lambda s: (
                        b * mp.expm(-a * (time - s)) * mp.expm(-eta * b * s) * y0
                    )[i],
                    [0, time],
                )
                for i in range(2)
            ]
        )
        assert mp.norm(ordered - wrong) > mp.mpf("1e-4")
        mixed = mp.matrix(4)
        for i in range(2):
            for j in range(2):
                mixed[i, j] = -eta * b[i, j]
                mixed[i + 2, j] = -eta * b[i, j]
                mixed[i + 2, j + 2] = -a[i, j]
        answer = mp.expm(mixed * time) * mp.matrix([*y0, 0, 0])
        assert mp.norm(mp.matrix(answer[2:]) - (-eta * ordered)) < mp.mpf("1e-70")


class TestActualFormTracking:
    def test_same_primitives_and_original_phase_certificate(self, frozen):
        assert frozen.status == "full" and frozen.joint_resolution_certified
        assert frozen.all_time_envelopes_certified
        assert frozen.form_resolution_certified and frozen.phase_resolution_certified
        assert frozen.unavailable_reasons == frozen.resolution_limitations == ()
        assert (
            signature(owner.assess_sine_port_form_tracking).parameters.keys()
            == signature(baseline_owner.assess_sine_port_relaxation).parameters.keys()
        )
        old = frozen.baseline_certificate
        assert old.status == "phase_only" and not old.form_resolution_certified
        assert (
            frozen.all_time_phase_error_upper_bound
            == old.all_time_phase_error_upper_bound
        )
        assert (
            frozen.phase_resolution_margin_bounds == old.phase_resolution_margin_bounds
        )
        assert frozen.phase_resolution_certified == old.phase_resolution_certified
        assert frozen.form_mean_error_floor == old.form_mean_error_floor
        assert frozen.phase_mean_error_floor == old.phase_mean_error_floor
        artifact = (
            Path(__file__).parents[2]
            / "docs/assets/sine_formed_classes/port-relaxation-v1.json"
        )
        assert json_loads(artifact.read_bytes())["report"] == old.to_dict()["report"]

    def test_bridge_variation_retained_only_in_even_forcing(self, frozen):
        old = frozen.baseline_certificate
        assert (
            frozen.bridge_hessian_variation_upper_bound
            == old.edge_disagreement_norm_squared_upper_bound
        )
        assert (
            frozen.even_bridge_forcing_upper_bound
            == old.edge_disagreement_norm_squared_upper_bound
            * old.even_bounds.phase_norm_upper_bound
        )
        assert (
            frozen.odd_heat_bounds.forcing_upper_bound
            == old.odd_bounds.forcing_upper_bound
        )
        assert (
            frozen.even_heat_bounds.forcing_upper_bound
            == old.even_bounds.forcing_upper_bound
            + frozen.even_bridge_forcing_upper_bound
        )
        assert frozen.even_bridge_forcing_upper_bound > 0

    def test_independent_direct_form_bound_and_channel_margins(self, frozen, mp):
        gamma = 1 / (1023 * mp.pi)
        expected = []
        for bounds in (frozen.odd_heat_bounds, frozen.even_heat_bounds):
            lam = _mp(mp, bounds.gap_lower_bound)
            gain = _mp(mp, bounds.derivative_filter_gain_upper_bound)
            norm, force = _mp(mp, bounds.initial_norm_upper_bound), _mp(
                mp, bounds.forcing_upper_bound
            )
            solved = (
                norm + gamma * 2 * (1 + gamma) * norm / lam + gamma * gain * force / lam
            ) / (1 - gamma**2 * 2 * gain / lam)
            assert _mp(mp, bounds.form_norm_upper_bound) >= solved
            expected.append(solved)
            assert bounds.certified and bounds.loop_margin > 0
        final = mp.sqrt(sum(x * x for x in expected)) / mp.sqrt(2) + _mp(
            mp, frozen.form_mean_error_floor
        )
        assert _mp(mp, frozen.all_time_form_error_upper_bound) >= final
        assert frozen.form_resolution_certified == (
            frozen.form_resolution_margin_bounds.lo > 0
        )
        assert frozen.joint_resolution_certified == (
            frozen.form_resolution_certified and frozen.phase_resolution_certified
        )

    @pytest.mark.parametrize(
        "changes",
        [
            {"formation_time": 0},
            {"relaxation_duration": 0},
            {"endpoint_radius": Q(1, 2**500)},
            {"phase_origins": (0, 1, 0)},
            {"normalized_gap_lower_bound": Q(1, 20)},
            {"normalized_gap_lower_bound": Q(1, 10**9)},
            {"contacts": ()},
        ],
    )
    def test_missing_baseline_premises_remain_unavailable(self, changes):
        report = _assess(**changes)
        assert report.status == "unavailable"
        assert not report.all_time_envelopes_certified
        assert report.all_time_form_error_upper_bound is None
        assert report.odd_heat_bounds is report.even_heat_bounds is None

    def test_work_policy_is_separate_from_mathematical_envelopes(self):
        report = _assess(work_allowance=0)
        assert report.status == "unavailable"
        assert (
            "supplied_contact_work_allowance_not_certified"
            in report.unavailable_reasons
        )
        assert report.all_time_envelopes_certified

    @pytest.mark.parametrize(
        "phase_fraction,form_fraction,status",
        [
            (Q(1, 2), Q(1, 10**9), "phase_only"),
            (Q(1, 10**9), Q(1, 2), "form_only"),
            (Q(1, 10**9), Q(1, 10**9), "envelopes_only"),
        ],
    )
    def test_resolution_policies_do_not_modify_envelopes(
        self, frozen, phase_fraction, form_fraction, status
    ):
        report = _assess(
            phase_resolution_fraction=phase_fraction,
            form_resolution_fraction=form_fraction,
        )
        assert report.status == status
        assert (
            report.all_time_form_error_upper_bound
            == frozen.all_time_form_error_upper_bound
        )
        assert (
            report.all_time_phase_error_upper_bound
            == frozen.all_time_phase_error_upper_bound
        )

    @pytest.mark.parametrize(
        "field",
        [
            "formation_time",
            "relaxation_duration",
            "form_error_bound",
            "phase_error_bound",
            "endpoint_radius",
            "radius",
            "work_allowance",
            "normalized_gap_lower_bound",
            "phase_resolution_fraction",
            "form_resolution_fraction",
        ],
    )
    @pytest.mark.parametrize("bad", [True, "0", 0j, float("nan"), float("inf")])
    def test_original_admission_precedes_new_heat_arithmetic(
        self, monkeypatch, field, bad
    ):
        def forbidden(**kwargs):
            pytest.fail("invalid original primitive reached the heat gain")

        monkeypatch.setattr(owner, "_heat_form_bounds", forbidden)
        with pytest.raises((TypeError, ValueError)):
            _assess(**{field: bad})

    def test_fresh_baseline_no_supplied_report_and_tiny_primitive_preservation(
        self, monkeypatch
    ):
        calls = []
        original = owner.assess_sine_port_relaxation

        def observe(**kwargs):
            report = original(**kwargs)
            calls.append((kwargs, report))
            return report

        monkeypatch.setattr(owner, "assess_sine_port_relaxation", observe)
        a, b = _assess(form_error_bound=0), _assess(form_error_bound=Q(1, 2**300))
        assert len(calls) == 2 and a.baseline_certificate is not b.baseline_certificate
        assert b.baseline_certificate.form_error_bound == Q(1, 2**300)
        with pytest.raises(TypeError):
            owner.assess_sine_port_form_tracking(
                baseline_certificate=a.baseline_certificate
            )
        with pytest.raises(TypeError):
            owner.assess_sine_port_form_tracking(**INPUTS, contact_duration=1)

    def test_sdk_and_saved_evaluation(self, frozen, tmp_path):
        direct = frozen.to_dict()
        assert direct["schema"] == "tnfr.sine-port-form-tracking.v1"
        assert relational_report_to_dict(frozen)["report"] == direct["report"]
        path = tmp_path / "form.json"
        export_to_json(frozen, path)
        assert json_loads(path.read_bytes())["report"] == direct["report"]
        artifact = (
            Path(__file__).parents[2]
            / "docs/assets/sine_formed_classes/port-form-tracking-v1.json"
        )
        retained = json_loads(artifact.read_bytes())
        assert (
            retained["schema"] == direct["schema"]
            and retained["report"] == direct["report"]
        )
        assert retained["frozen_stopping_rule_passed"] is True
        assert (
            get_type_hints(owner.assess_sine_port_form_tracking)["return"]
            is owner.SinePortFormTracking
        )
        get_type_hints(owner.SinePortFormTracking)
