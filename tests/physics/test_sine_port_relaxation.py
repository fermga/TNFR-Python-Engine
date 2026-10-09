"""Independent spectral and nonlinear controls for all-time port tracking.

The source assessor is invoked only after protocol/source archival and the
first reserved evaluation. Spectral tests concern the supplied support alone.
"""

from fractions import Fraction as Q
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tests.physics.test_sine_formed_class_contact import _mp
from tests.physics.test_sine_port_composition import _fine, _lifts, _transpose
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.physics import relational_sine_port_relaxation as owner
from tnfr.physics._sine_port_geometry import _central_port_geometry
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

INPUTS = dict(
    classes=(1, 2, 1),
    contacts=((0, 1), (1, 2)),
    phase_origins=(Q(0), Q(1, 1000), Q(0)),
    formation_time=Q(100),
    relaxation_duration=Q(10**13),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    endpoint_radius=Q(1, 10**32),
    radius=Q(1, 12),
    decay_power=512,
    work_allowance=Q(2, 10**6),
    normalized_gap_lower_bound=Q(3, 100),
    phase_resolution_fraction=Q(1, 2),
    form_resolution_fraction=Q(1, 2),
)


def _assess(**changes):
    return owner.assess_sine_port_relaxation(**(INPUTS | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


@pytest.fixture(scope="module", autouse=True)
def no_old_certificate_or_trajectory():
    from tnfr.dynamics import relational
    from tnfr.physics import (
        relational_sine_forecast,
        relational_sine_formed_classes,
        relational_sine_port_composition,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "all-time tracking requires fresh sources, not a prior verdict or trajectory"
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


class TestStaticSpectralControls:
    @pytest.mark.parametrize(
        "gap,expected",
        [(Q(3, 100), True), (Q(150217, 5000000), True), (Q(60087, 2000000), False)],
    )
    def test_normalized_full_quotient_gap_has_an_exact_witness(self, gap, expected):
        _, laplacian, degrees = _fine(3, ((0, 1), (1, 2)))
        assert sum(degrees) == 58
        assert (degrees[4], degrees[13], degrees[22]) == (3, 4, 3)
        witness = tuple(
            tuple(
                laplacian[i][j]
                - gap * (degrees[i] * Q(i == j) - Q(degrees[i] * degrees[j], 58))
                for j in range(27)
            )
            for i in range(27)
        )
        assert exact_symmetric_semidefinite(witness) is expected

    def test_even_witness_is_the_fine_weighted_quotient_projection(self):
        _, laplacian, degrees = _fine(3, ((0, 1), (1, 2)))
        lift, _ = _lifts(3)
        geometry = _central_port_geometry(3, ((0, 1), (1, 2)))
        gap, masses = Q(3, 100), geometry.layer_masses
        fine = tuple(
            tuple(
                laplacian[i][j]
                - gap * (degrees[i] * Q(i == j) - Q(degrees[i] * degrees[j], 58))
                for j in range(27)
            )
            for i in range(27)
        )
        reduced = tuple(
            tuple(
                geometry.joined_laplacian[i][j]
                - gap * (masses[i] * Q(i == j) - masses[i] * masses[j] / 58)
                for j in range(15)
            )
            for i in range(15)
        )
        assert (
            exact_matrix_product(_transpose(lift), exact_matrix_product(fine, lift))
            == reduced
        )
        assert exact_symmetric_semidefinite(reduced)

    @pytest.mark.parametrize("component", range(3))
    def test_odd_block_survives_actual_contact_degree_changes(self, component):
        _, laplacian, degrees = _fine(3, ((0, 1), (1, 2)))
        generator = tuple(
            tuple(v / degrees[i] for v in row) for i, row in enumerate(laplacian)
        )
        lift = [[Q(0)] * 4 for _ in range(27)]
        for layer in range(4):
            lift[9 * component + 3 - layer][layer] = Q(1)
            lift[9 * component + 5 + layer][layer] = Q(-1)
        lift = tuple(map(tuple, lift))
        odd = tuple(
            tuple(
                (
                    Q(3, 2)
                    if i == j == 3
                    else Q(1) if i == j else Q(-1, 2) if abs(i - j) == 1 else Q(0)
                )
                for j in range(4)
            )
            for i in range(4)
        )
        assert exact_matrix_product(generator, lift) == exact_matrix_product(lift, odd)
        assert all(lift[port] == (0, 0, 0, 0) for port in (4, 13, 22))
        assert all(
            sum(degrees[i] * lift[i][j] for i in range(27)) == 0 for j in range(4)
        )
        witness = tuple(
            tuple(odd[i][j] - Q(7, 30) * (i == j) for j in range(4)) for i in range(4)
        )
        assert exact_symmetric_semidefinite(witness, strict=True)
        # Sylvester's test independently supplies compact rational minors.
        previous, current = Q(1), witness[0][0]
        minors = [current]
        for i in range(1, 4):
            previous, current = (
                current,
                witness[i][i] * current - witness[i][i - 1] ** 2 * previous,
            )
            minors.append(current)
        assert minors == [Q(23, 30), Q(76, 225), Q(1817, 27000), Q(323, 405000)]


class TestStaticRectangles:
    @pytest.mark.parametrize(
        "forcing,initial",
        [(Q(0), Q(0)), (Q(0), Q(1, 100)), (Q(1, 700), Q(0)), (Q(1, 700), Q(1, 100))],
    )
    def test_exact_metzler_corner_invariance_and_initial_exchange(
        self, forcing, initial
    ):
        bound = owner._parity_tracking_bounds(
            gap=Q(1, 10), forcing=forcing, initial_norm=initial, gamma_upper=Q(1, 100)
        )
        assert bound.certified
        a, b = bound.phase_damping, bound.scaled_form_damping
        y, z = bound.joint_norm_upper_bound, bound.scaled_form_norm_upper_bound
        assert y >= bound.initial_joint_norm_upper_bound
        assert z >= bound.initial_scaled_form_norm_upper_bound
        assert -a * y + 2 * z + forcing <= 0
        assert 2 * y - b * z + forcing <= 0
        assert -a * 0 + 2 * z + forcing >= 0
        assert 2 * y - b * 0 + forcing >= 0
        if initial:
            # This admitted toy has an outward initial y row. Ignoring its
            # initial exchange correction would invalidate that rectangle.
            assert bound.joint_initial_correction > 0
        if not initial:
            assert a * y - 2 * z == forcing
            assert -2 * y + b * z == forcing

    def test_nonpositive_determinant_has_no_claimed_rectangle(self):
        report = owner._parity_tracking_bounds(
            gap=Q(1, 10**9), forcing=Q(1), initial_norm=Q(1), gamma_upper=Q(1, 100)
        )
        assert report.determinant < 0 and not report.certified
        assert report.phase_norm_upper_bound is None
        assert report.joint_norm_upper_bound is None
        assert report.scaled_form_norm_upper_bound is None

    def test_closed_comparison_solution_stays_within_the_rectangle(self, mp):
        report = owner._parity_tracking_bounds(
            gap=Q(1, 10),
            forcing=Q(1, 700),
            initial_norm=Q(1, 100),
            gamma_upper=Q(1, 100),
        )
        matrix = mp.matrix(
            [
                [-_mp(mp, report.phase_damping), 2],
                [2, -_mp(mp, report.scaled_form_damping)],
            ]
        )
        drive = mp.matrix([_mp(mp, report.forcing_upper_bound)] * 2)
        equilibrium = mp.lu_solve(-matrix, drive)
        initial = mp.matrix(
            [
                _mp(mp, report.initial_joint_norm_upper_bound),
                _mp(mp, report.initial_scaled_form_norm_upper_bound),
            ]
        )
        for time in (0, 1, 10, 1000):
            point = equilibrium + mp.expm(matrix * time) * (initial - equilibrium)
            assert 0 <= point[0] <= _mp(mp, report.joint_norm_upper_bound)
            assert 0 <= point[1] <= _mp(mp, report.scaled_form_norm_upper_bound)

    @pytest.mark.parametrize("odd_amplitude", [Q(0), Q(1, 1000)])
    def test_full_internal_sine_remainder_obeys_parity_forcing_bounds(
        self, mp, odd_amplitude
    ):
        edges, _, degrees = _fine(3, INPUTS["contacts"])
        y = tuple(
            Q((abs(i % 9 - 4) * 7 + i // 9) % 11 - 5, 1000)
            + odd_amplitude * (i % 9 - 4)
            for i in range(27)
        )
        reflected = tuple(9 * (i // 9) + 8 - i % 9 for i in range(27))
        target = tuple(
            2 * INPUTS["classes"][i // 9] * mp.pi * (i % 9 - 4) / 9 for i in range(27)
        )
        residual = [mp.mpf(0) for _ in range(27)]
        edge_norm_squared = Q(0)
        for i, j in edges:
            gap = y[j] - y[i]
            edge_norm_squared += gap**2
            if i // 9 != j // 9:
                continue  # Exact bridge currents are retained in both laws.
            base = target[j] - target[i]
            remainder = (
                mp.sin(base + _mp(mp, gap)) - mp.sin(base) - mp.cos(base) * _mp(mp, gap)
            )
            residual[i] += remainder
            residual[j] -= remainder
        normalized = tuple(value / degree for value, degree in zip(residual, degrees))
        even = tuple((normalized[i] + normalized[reflected[i]]) / 2 for i in range(27))
        odd = tuple((normalized[i] - normalized[reflected[i]]) / 2 for i in range(27))
        even_norm = mp.sqrt(sum(degrees[i] * even[i] ** 2 for i in range(27)))
        odd_norm = mp.sqrt(sum(degrees[i] * odd[i] ** 2 for i in range(27)))
        hidden_phase_norm = mp.sqrt(
            _mp(
                mp,
                sum(
                    (
                        degrees[i] * ((y[i] - y[reflected[i]]) / 2) ** 2
                        for i in range(27)
                    ),
                    Q(0),
                ),
            )
        )
        b = mp.sqrt(_mp(mp, edge_norm_squared))
        assert odd_norm <= mp.sqrt(2) * b**2 / 2
        assert even_norm <= 2 * b * hidden_phase_norm + mp.sqrt(2) * b**3 / 6
        if not odd_amplitude:
            assert hidden_phase_norm == 0
            assert odd_norm > 0  # Even preparation generates omitted odd modes.


class TestActualTracking:
    def test_frozen_report_preserves_separate_channel_resolution(self, frozen):
        assert frozen.status == "phase_only"
        assert (
            frozen.phase_resolution_certified and not frozen.form_resolution_certified
        )
        assert frozen.resolution_limitations == ("form_resolution_not_certified",)
        assert frozen.normalized_gap_certified and frozen.refined_chart_certified
        assert frozen.joined_bounds.identity_certified
        assert frozen.joined_bounds.work_within_allowance
        assert frozen.all_time_envelopes_certified
        assert frozen.phase_resolution_certified == (
            frozen.phase_resolution_margin_bounds.lo > 0
        )
        assert frozen.form_resolution_certified == (
            frozen.form_resolution_margin_bounds.lo > 0
        )
        assert frozen.joint_resolution_certified == (
            frozen.phase_resolution_certified and frozen.form_resolution_certified
        )
        assert frozen.unavailable_reasons == ()
        assert frozen.origin_span == Q(1, 1000)
        assert frozen.phase_allowance == Q(1, 2000)
        assert frozen.form_allowance_bounds == frozen.gamma_bounds * Q(1, 2000)

    def test_independent_high_precision_gain_and_norm_oracle(self, frozen, mp):
        gamma = 1 / (1023 * mp.pi)
        energy = _mp(mp, frozen.joined_bounds.joined_excess_storage_upper_bound)
        coarse = mp.cos(4 * mp.pi / 9 + mp.sqrt(2) / 12)
        edge = mp.sqrt(2 * energy / coarse)
        assert _mp(mp, frozen.edge_deviation_bounds.hi) >= edge
        assert _mp(mp, frozen.refined_cosine_bounds.lo) <= mp.cos(4 * mp.pi / 9 + edge)
        squared = 12 * energy
        bnorm = mp.sqrt(squared)

        def solve(gap, force, norm):
            a, b = gap / 6, gap / gamma**2 - 2
            initial = mp.matrix([(1 + gamma) * norm, gamma * norm])
            generator = mp.matrix([[-a, 2], [2, -b]])
            initial_derivative = generator * initial
            correction = mp.matrix([max(0, initial_derivative[i]) for i in range(2)])
            extra = mp.lu_solve(-generator, mp.matrix([force, force]) + correction)
            return initial + extra

        odd_force = mp.sqrt(2) * squared / 2
        odd = solve(
            mp.mpf(7) / 30, odd_force, mp.sqrt(6) * _mp(mp, INPUTS["endpoint_radius"])
        )
        even_force = 2 * bnorm * sum(odd) + mp.sqrt(2) * bnorm**3 / 6
        even = solve(
            mp.mpf(3) / 100,
            even_force,
            mp.sqrt(12) * _mp(mp, INPUTS["endpoint_radius"]),
        )
        for actual, expected in ((frozen.odd_bounds, odd), (frozen.even_bounds, even)):
            assert _mp(mp, actual.joint_norm_upper_bound) >= expected[0]
            assert _mp(mp, actual.scaled_form_norm_upper_bound) >= expected[1]
        mean = _mp(mp, Q(2, 29) * INPUTS["endpoint_radius"])
        phase = mp.sqrt(sum(odd) ** 2 + sum(even) ** 2) / mp.sqrt(2) + mean
        form = mp.sqrt(odd[1] ** 2 + even[1] ** 2) / (gamma * mp.sqrt(2)) + mean
        assert _mp(mp, frozen.all_time_phase_error_upper_bound) >= phase
        assert _mp(mp, frozen.all_time_form_error_upper_bound) >= form

    def test_actual_conserved_mean_floor_survives_zero_sum_source_error(self, frozen):
        eps = INPUTS["endpoint_radius"]
        _, _, degree = _fine(3, INPUTS["contacts"])
        residual = [Q(0)] * 27
        # Each component has zero ordinary mean and Euclidean norm below eps,
        # yet the added port degrees change its conserved joined mean.
        for component in range(3):
            residual[9 * component + 4] = eps / 2
            residual[9 * component + 3] = -eps / 2
        mean = sum((d * x for d, x in zip(degree, residual)), Q(0)) / 58
        assert mean > 0
        assert (
            frozen.form_mean_error_floor
            == frozen.phase_mean_error_floor
            == Q(2, 29) * eps
        )
        assert mean <= frozen.form_mean_error_floor
        assert frozen.all_time_form_error_upper_bound >= frozen.form_mean_error_floor
        assert frozen.all_time_phase_error_upper_bound >= frozen.phase_mean_error_floor

    @pytest.mark.parametrize(
        "changes,reason",
        [
            ({"formation_time": 0}, "formation_unavailable"),
            ({"relaxation_duration": 0}, "unprobed_endpoint_budget_not_certified"),
            (
                {"endpoint_radius": Q(1, 2**500)},
                "unprobed_endpoint_budget_not_certified",
            ),
            ({"phase_origins": (0, 1, 0)}, "whole_network_identity_not_certified"),
            ({"phase_origins": (0, Q(1, 800), 0)}, "refined_acute_chart_not_certified"),
            ({"normalized_gap_lower_bound": Q(1, 20)}, "normalized_gap_not_certified"),
            (
                {"normalized_gap_lower_bound": Q(1, 10**9)},
                "parity_tracking_envelopes_not_certified",
            ),
            ({"contacts": ()}, "connected_multi_component_identity_not_supported"),
        ],
    )
    def test_missing_hypotheses_do_not_supply_tracking_envelopes(self, changes, reason):
        report = _assess(**changes)
        assert report.status == "unavailable" and reason in report.unavailable_reasons
        assert not report.all_time_envelopes_certified
        assert report.all_time_form_error_upper_bound is None
        assert report.all_time_phase_error_upper_bound is None

    def test_work_failure_remains_separate_from_valid_trajectory_bounds(self):
        report = _assess(work_allowance=0)
        assert report.status == "unavailable"
        assert (
            "supplied_contact_work_allowance_not_certified"
            in report.unavailable_reasons
        )
        assert report.all_time_envelopes_certified
        assert report.joined_bounds.identity_certified

    def test_resolution_change_does_not_change_underlying_envelopes(self, frozen):
        report = _assess(
            phase_resolution_fraction=Q(1, 10**9), form_resolution_fraction=Q(1, 10**9)
        )
        assert report.status == "envelopes_only"
        assert (
            not report.phase_resolution_certified
            and not report.form_resolution_certified
        )
        assert (
            report.all_time_phase_error_upper_bound
            == frozen.all_time_phase_error_upper_bound
        )
        assert (
            report.all_time_form_error_upper_bound
            == frozen.all_time_form_error_upper_bound
        )

    @pytest.mark.parametrize(
        "phase_fraction,status", [(Q(1, 2), "full"), (Q(1, 10**12), "form_only")]
    )
    def test_independent_resolution_policies_have_symmetric_statuses(
        self, phase_fraction, status
    ):
        # One fixed smaller-origin boundary control exercises policy routing;
        # it neither changes nor replaces the reserved three-component result.
        report = _assess(
            phase_origins=(0, Q(1, 10**6), 0), phase_resolution_fraction=phase_fraction
        )
        assert report.status == status
        assert report.form_resolution_certified
        assert report.phase_resolution_certified is (status == "full")
        assert report.all_time_envelopes_certified

    def test_zero_origin_span_is_valid_without_positive_resolution(self):
        report = _assess(phase_origins=(0, 0, 0))
        assert report.status == "envelopes_only"
        assert report.origin_span == report.phase_allowance == 0
        assert report.form_allowance_bounds.lo == report.form_allowance_bounds.hi == 0
        assert report.all_time_envelopes_certified
        assert (
            not report.phase_resolution_certified
            and not report.form_resolution_certified
        )

    def test_touching_phase_margin_is_not_resolution(self, frozen):
        fraction = frozen.all_time_phase_error_upper_bound / frozen.origin_span
        assert 0 < fraction <= 1
        report = _assess(phase_resolution_fraction=fraction)
        assert (
            report.phase_resolution_margin_bounds.lo
            == report.phase_resolution_margin_bounds.hi
            == 0
        )
        assert not report.phase_resolution_certified
        assert report.all_time_envelopes_certified

    def test_common_origin_shift_preserves_tracking_bounds(self, frozen):
        report = _assess(
            phase_origins=tuple(o + Q(7, 3) for o in INPUTS["phase_origins"])
        )
        assert report.origin_span == frozen.origin_span
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
    @pytest.mark.parametrize("bad", [True, "0", 0j, float("inf"), float("nan")])
    def test_original_scalar_admission_precedes_any_certificate(
        self, monkeypatch, field, bad
    ):
        def forbidden(*args, **kwargs):
            pytest.fail("invalid primitive reached scientific arithmetic")

        monkeypatch.setattr(owner, "_central_port_geometry", forbidden)
        monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
        with pytest.raises((ValueError, TypeError)):
            _assess(**{field: bad})

    @pytest.mark.parametrize(
        "changes",
        [
            {"formation_time": -1},
            {"relaxation_duration": -1},
            {"form_error_bound": -1},
            {"phase_error_bound": -1},
            {"endpoint_radius": 0},
            {"radius": 0},
            {"radius": Q(1, 12) + Q(1, 2**200)},
            {"work_allowance": -1},
            {"normalized_gap_lower_bound": 0},
            {"normalized_gap_lower_bound": Q(7, 30) + Q(1, 2**200)},
            {"phase_resolution_fraction": 0},
            {"phase_resolution_fraction": Q(1) + Q(1, 2**200)},
            {"form_resolution_fraction": 0},
            {"form_resolution_fraction": 2},
            {"decay_power": True},
            {"decay_power": Q(512)},
            {"decay_power": -1},
            {"decay_power": 4097},
        ],
    )
    def test_invalid_domains_precede_reconstruction(self, monkeypatch, changes):
        def forbidden(*args, **kwargs):
            pytest.fail("invalid domain reached scientific arithmetic")

        monkeypatch.setattr(owner, "_central_port_geometry", forbidden)
        with pytest.raises(ValueError):
            _assess(**changes)

    def test_fresh_primitive_sources_and_clock_work_caps(self, monkeypatch, frozen):
        calls = []
        original = owner._unprobed_handoff

        def observe(**kwargs):
            result = original(**kwargs)
            calls.append((kwargs, result))
            return result

        monkeypatch.setattr(owner, "_unprobed_handoff", observe)
        a = _assess(form_error_bound=0)
        b = _assess(form_error_bound=Q(1, 2**300))
        assert a.unprobed_handoff is not b.unprobed_handoff and len(calls) == 2
        assert calls[0][0]["form_error_bound"] == 0
        assert calls[1][0]["form_error_bound"] == Q(1, 2**300)
        with pytest.raises(TypeError):
            owner.assess_sine_port_relaxation(unprobed_handoff=a.unprobed_handoff)
        with pytest.raises(TypeError):
            owner.assess_sine_port_relaxation(**INPUTS, contact_duration=1)
        with pytest.raises(ValueError, match="scaled_time"):
            _assess(formation_time=20481)
        with pytest.raises(ValueError, match="lyapunov_decay_rate"):
            _assess(
                relaxation_duration=Q(4097)
                / frozen.unprobed_handoff.lyapunov_decay_rate
            )

    def test_sdk_and_original_retained_report(self, frozen, tmp_path):
        direct = frozen.to_dict()
        assert direct["schema"] == "tnfr.sine-port-relaxation.v1"
        assert relational_report_to_dict(frozen)["report"] == direct["report"]
        path = tmp_path / "relaxation.json"
        export_to_json(frozen, path)
        assert json_loads(path.read_bytes())["report"] == direct["report"]
        artifact = (
            Path(__file__).parents[2]
            / "docs/assets/sine_formed_classes/port-relaxation-v1.json"
        )
        retained = json_loads(artifact.read_bytes())
        assert (
            retained["schema"] == direct["schema"]
            and retained["report"] == direct["report"]
        )
        assert retained["frozen_stopping_rule_passed"] is False
        assert (
            get_type_hints(owner.assess_sine_port_relaxation)["return"]
            is owner.SinePortRelaxation
        )
        get_type_hints(owner.SinePortRelaxation)
