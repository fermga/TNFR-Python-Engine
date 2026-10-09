"""Fine-graph, balance and actual-family controls for reduced C9 networks.

Static controls use no source assessor. The actual-family fixture is evaluated
only after the prospective protocol/source archive and first reserved result.
No trajectory, inferred contact event or ideal-state reset is used.
"""

import random
from fractions import Fraction as Q
from pathlib import Path
from typing import get_type_hints

import mpmath
import pytest

from tests.physics.test_sine_formed_class_contact import _contains, _mp, _mv
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.physics import relational_sine_port_composition as owner
from tnfr.sdk import export_to_json, relational_report_to_dict
from tnfr.utils.io import json_loads

INPUTS = dict(
    classes=(1, 2, 1),
    contacts=((0, 1), (1, 2)),
    phase_origins=(Q(0), Q(1, 1000), Q(0)),
    formation_time=Q(100),
    relaxation_duration=Q(10**13),
    contact_duration=Q(1, 100),
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    endpoint_radius=Q(1, 10**32),
    radius=Q(1, 12),
    decay_power=512,
    work_allowance=Q(2, 10**6),
    approximation_allowance=Q(2, 10**32),
)
SUPPORTS = (
    (1, ()),
    (2, ((0, 1),)),
    (3, ((0, 1), (1, 2))),
    (3, ((0, 1), (0, 2), (1, 2))),
    (4, ((0, 1), (0, 2), (0, 3))),
    (4, ((0, 1), (2, 3))),
)


def _evaluate(**changes):
    values = dict(
        classes=(1, 2, 1),
        contacts=((0, 1), (1, 2)),
        phase_origins=(Q(0), Q(1, 1000), Q(0)),
        forms=(Q(0),) * 15,
        phase_deviations=(Q(0),) * 15,
    )
    return owner.evaluate_sine_port_composition(**(values | changes))


def _assess(**changes):
    return owner.assess_sine_port_composition(**(INPUTS | changes))


@pytest.fixture(scope="module")
def frozen():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 120
    return context


@pytest.fixture(scope="module", autouse=True)
def no_old_verdict_probe_or_trajectory():
    from tnfr.dynamics import relational
    from tnfr.physics import (
        relational_sine_forecast,
        relational_sine_formed_class_contact,
        relational_sine_formed_class_maintenance,
        relational_sine_formed_classes,
        relational_sine_reduced_class_ports,
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "composition must retain fresh unprobed sources and analytic bounds"
        )

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (relational, "step_relational_exchange"),
            (relational, "_advance"),
            (relational_sine_forecast, "bound_sine_flow"),
            (relational_sine_formed_class_contact, "assess_sine_formed_class_contact"),
            (relational_sine_formed_classes, "assess_sine_formed_class_response"),
            (
                relational_sine_formed_class_maintenance,
                "assess_sine_formed_class_maintenance",
            ),
            (relational_sine_reduced_class_ports, "assess_sine_reduced_class_ports"),
        ):
            patch.setattr(module, name, forbidden)
        yield


def _fine(count, contacts):
    """Assemble the original nine-node cycles, before any radial reduction."""
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(count) for j in range(9)
    ) + tuple((9 * i + 4, 9 * j + 4) for i, j in contacts)
    size = 9 * count
    matrix = [[Q(0) for _ in range(size)] for _ in range(size)]
    for i, j in edges:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return (
        edges,
        tuple(tuple(row) for row in matrix),
        tuple(sum(i in edge for edge in edges) for i in range(size)),
    )


def _lifts(count):
    orbit = tuple(5 * c + abs(j - 4) for c in range(count) for j in range(9))
    lift = tuple(
        tuple(Q(j == orbit[i]) for j in range(5 * count)) for i in range(9 * count)
    )
    project = tuple(
        tuple(Q(orbit[i] == j, orbit.count(j)) for i in range(9 * count))
        for j in range(5 * count)
    )
    return lift, project


def _transpose(matrix):
    return tuple(zip(*matrix))


class TestStaticComposition:
    @pytest.mark.parametrize("count,contacts", SUPPORTS)
    def test_exact_fine_lift_and_degree_normalization(self, count, contacts):
        row = _evaluate(
            classes=tuple(1 + i % 2 for i in range(count)),
            contacts=contacts,
            phase_origins=(Q(0),) * count,
            forms=(Q(0),) * (5 * count),
            phase_deviations=(Q(0),) * (5 * count),
        )
        _, fine_l, degrees = _fine(count, contacts)
        lift, project = _lifts(count)
        identity = tuple(
            tuple(Q(i == j) for j in range(5 * count)) for i in range(5 * count)
        )
        assert exact_matrix_product(project, lift) == identity
        assert (
            exact_matrix_product(_transpose(lift), exact_matrix_product(fine_l, lift))
            == row.geometry.joined_laplacian
        )
        fine_a = tuple(
            tuple(x / degrees[i] for x in values) for i, values in enumerate(fine_l)
        )
        assert exact_matrix_product(fine_a, lift) == exact_matrix_product(
            lift, row.geometry.normalized_form_matrix
        )
        assert row.geometry.layer_masses == tuple(
            sum(degrees[i] * lift[i][j] for i in range(9 * count))
            for j in range(5 * count)
        )
        assert len(row.forms) + len(row.phase_deviations) == 10 * count

    def test_changed_middle_degree_balance_counterexample(self, mp):
        x = tuple(Q(i == 5) for i in range(15))
        report = _evaluate(forms=x, phase_origins=(0, 0, 0))
        assert report.geometry.contact_degrees == (1, 2, 1)
        assert report.form_storage == 2
        assert report.dissipation == Q(17, 3)
        assert report.storage_rate == -Q(17, 3)
        assert report.form_rate_bounds[5].contains(-1)
        assert report.network_form_charge_rate == report.network_phase_charge_rate == 0
        edges, lap, degrees = _fine(3, INPUTS["contacts"])
        fine_x = tuple(Q(i == 13) for i in range(27))
        lx = _mv(lap, fine_x)
        wrong_degrees = tuple(
            3 if i % 9 == 4 else degree for i, degree in enumerate(degrees)
        )
        wrong_rows = tuple(-value / degree for value, degree in zip(lx, wrong_degrees))
        assert wrong_rows[13] == -Q(4, 3)
        assert sum(degree * rate for degree, rate in zip(degrees, wrong_rows)) == -Q(
            4, 3
        )
        assert sum(lx[i] * wrong_rows[i] for i in range(27)) == -7
        assert len(edges) == 29
        _contains(mp, report.storage_rate_from_rows_bounds, -mp.mpf(17) / 3)

    @pytest.mark.parametrize("count,contacts", SUPPORTS[1:])
    def test_complete_fine_rows_storage_and_conjugate_supply(self, mp, count, contacts):
        classes = tuple(1 + i % 2 for i in range(count))
        x = tuple(Q((7 * i) % 17 - 8, 19) for i in range(5 * count))
        y = tuple(Q((11 * i) % 13 - 6, 71) for i in range(5 * count))
        origins = tuple(Q(i * i - 2, 29) for i in range(count))
        report = _evaluate(
            classes=classes,
            contacts=contacts,
            phase_origins=origins,
            forms=x,
            phase_deviations=y,
        )
        edges, lap, degrees = _fine(count, contacts)
        lift, _ = _lifts(count)
        fine_x, fine_y = _mv(lift, x), _mv(lift, y)
        lx = _mv(lap, fine_x)
        gamma = 1 / (1023 * mp.pi)
        cosine = tuple(mp.cos(2 * k * mp.pi / 9) for k in classes)
        gradient = [mp.mpf(0) for _ in fine_y]
        phase_energy = mp.mpf(0)
        ix = [Q(0) for _ in fine_x]
        iy = [Q(0) for _ in fine_y]
        for i, j in edges:
            if i // 9 == j // 9:
                ix[i] += fine_x[i] - fine_x[j]
                ix[j] -= fine_x[i] - fine_x[j]
                iy[i] += fine_y[i] - fine_y[j]
                iy[j] -= fine_y[i] - fine_y[j]
                current = cosine[i // 9] * _mp(mp, fine_y[j] - fine_y[i])
                phase_energy += (
                    cosine[i // 9] * _mp(mp, (fine_y[j] - fine_y[i]) ** 2) / 2
                )
            else:
                gap = _mp(mp, origins[j // 9] - origins[i // 9] + fine_y[j] - fine_y[i])
                current = mp.sin(gap)
                phase_energy += 1 - mp.cos(gap)
            gradient[i] -= current
            gradient[j] += current
        dx = tuple(
            (-_mp(mp, lx[i]) - gamma * gradient[i]) / degrees[i]
            for i in range(9 * count)
        )
        dy = tuple(gamma * _mp(mp, lx[i]) / degrees[i] for i in range(9 * count))
        for component in range(count):
            for layer in range(5):
                fine_index, reduced_index = (
                    9 * component + 4 - layer,
                    5 * component + layer,
                )
                _contains(mp, report.form_rate_bounds[reduced_index], dx[fine_index])
                _contains(mp, report.phase_rate_bounds[reduced_index], dy[fine_index])
            nodes = range(9 * component, 9 * component + 9)
            internal_rate = sum(
                _mp(mp, ix[i]) * dx[i] + cosine[component] * _mp(mp, iy[i]) * dy[i]
                for i in nodes
            )
            loss = sum((ix[i] ** 2 / degrees[i] for i in nodes), Q(0))
            assert report.internal_dissipation[component] == loss
            _contains(
                mp,
                report.internal_supply_rate_bounds[component],
                internal_rate + _mp(mp, loss),
            )
            _contains(
                mp,
                report.component_form_charge_rate_bounds[component],
                sum(degrees[i] * dx[i] for i in nodes),
            )
            # Sum the rational fine Laplacian coefficients before multiplying
            # by gamma; finite-precision cancellation cannot refute exact zero.
            _contains(
                mp,
                report.component_phase_charge_rate_bounds[component],
                gamma * _mp(mp, sum((lx[i] for i in nodes), Q(0))),
            )
        assert report.form_storage == sum(
            ((fine_x[j] - fine_x[i]) ** 2 / 2 for i, j in edges), Q(0)
        )
        assert report.dissipation == sum(
            (lx[i] ** 2 / degrees[i] for i in range(9 * count)), Q(0)
        )
        _contains(mp, report.phase_storage_bounds, phase_energy)
        _contains(
            mp, report.storage_rate_from_rows_bounds, -_mp(mp, report.dissipation)
        )

    def test_origin_representations_and_closed_cycle_are_correlated(self):
        x = tuple(Q(i - 7, 17) for i in range(15))
        y = tuple(Q(i * i - 11, 101) for i in range(15))
        contacts = ((1, 2), (2, 0), (1, 0))
        origins = (Q(1, 5), Q(-2, 7), Q(3, 11))
        base = _evaluate(
            contacts=contacts, phase_origins=origins, forms=x, phase_deviations=y
        )
        shifts = (Q(2), Q(-7, 9), Q(5, 13))
        shifted = _evaluate(
            contacts=contacts,
            phase_origins=tuple(o + s for o, s in zip(origins, shifts)),
            forms=x,
            phase_deviations=tuple(v - shifts[i // 5] for i, v in enumerate(y)),
        )
        for field in (
            "form_rate_bounds",
            "phase_rate_bounds",
            "weighted_phase_charges",
            "storage_bounds",
            "bridge_sine_bounds",
        ):
            assert getattr(base, field) == getattr(shifted, field)
        assert base.contacts == ((0, 1), (0, 2), (1, 2))
        d01, d02, d12 = base.bridge_phase_differences
        assert d01 + d12 - d02 == 0
        with pytest.raises(TypeError):
            owner.evaluate_sine_port_composition(edge_phase_offsets=(0, 0, 0))

    def test_component_permutation_preserves_geometry_and_rows(self):
        classes, origins = (1, 2, 1), (Q(2, 13), Q(-3, 17), Q(1, 7))
        x = tuple(Q(i - 8, 31) for i in range(15))
        y = tuple(Q((i * 3) % 7, 37) for i in range(15))
        original = _evaluate(
            classes=classes, phase_origins=origins, forms=x, phase_deviations=y
        )
        permutation = (2, 0, 1)
        inverse = {old: new for new, old in enumerate(permutation)}
        reordered = _evaluate(
            classes=tuple(classes[i] for i in permutation),
            phase_origins=tuple(origins[i] for i in permutation),
            contacts=tuple((inverse[j], inverse[i]) for i, j in INPUTS["contacts"]),
            forms=tuple(x[5 * i + j] for i in permutation for j in range(5)),
            phase_deviations=tuple(y[5 * i + j] for i in permutation for j in range(5)),
        )
        assert reordered.form_storage == original.form_storage
        assert reordered.dissipation == original.dissipation
        for old, new in inverse.items():
            for j in range(5):
                # Incidence summation order can widen intervals by a last ulp.
                for field in ("form_rate_bounds", "phase_rate_bounds"):
                    a, b = (
                        getattr(original, field)[5 * old + j],
                        getattr(reordered, field)[5 * new + j],
                    )
                    assert max(a.lo, b.lo) <= min(a.hi, b.hi)

    def test_generated_odd_nonlinear_defect_is_not_an_exact_quotient(self, mp):
        y = tuple(Q(i == 5, 10) for i in range(15))
        row = _evaluate(phase_deviations=y, phase_origins=(0, 0, 0))
        edges, _, degrees = _fine(3, INPUTS["contacts"])
        lift, _ = _lifts(3)
        fy = _mv(lift, y)
        alpha = 2 * mp.pi / 9
        target = tuple(
            INPUTS["classes"][i // 9] * alpha * (i % 9 - 4) for i in range(27)
        )
        pressure = [mp.mpf(0) for _ in range(27)]
        for i, j in edges:
            current = mp.sin(target[j] - target[i] + _mp(mp, fy[j] - fy[i]))
            pressure[i] += current
            pressure[j] -= current
        gamma = 1 / (1023 * mp.pi)
        full_rates = tuple(
            gamma * value / degree for value, degree in zip(pressure, degrees)
        )
        assert abs(full_rates[12] - full_rates[14]) > mp.mpf("1e-7")
        _contains(mp, row.form_rate_bounds[6], gamma * mp.cos(2 * alpha) / 20)
        # The first-order remainder bound is normalized per fine row; bridge
        # currents are already exact and contribute no discarded remainder.
        bound = 2 * gamma * mp.mpf("0.1") ** 2
        for i in range(27):
            interval = row.form_rate_bounds[5 * (i // 9) + abs(i % 9 - 4)]
            assert abs(full_rates[i] - _mp(mp, interval.midpoint)) < bound

    def test_singleton_disconnected_and_component_cap(self):
        for count in (1, 16):
            report = _evaluate(
                classes=(1,) * count,
                contacts=(),
                phase_origins=(0,) * count,
                forms=(0,) * (5 * count),
                phase_deviations=(0,) * (5 * count),
            )
            assert report.storage_rate == 0
            assert report.geometry.contact_degrees == (0,) * count
            assert report.geometry.connected is (count == 1)

    @pytest.mark.parametrize("field", ["forms", "phase_deviations", "phase_origins"])
    @pytest.mark.parametrize("bad", [True, "0", 0j, float("inf"), float("nan")])
    def test_original_scalar_admission_precedes_geometry(self, monkeypatch, field, bad):
        def forbidden(*args):
            pytest.fail("invalid primitive reached matrix construction")

        monkeypatch.setattr(owner, "_central_port_geometry", forbidden)
        values = [0] * (3 if field == "phase_origins" else 15)
        values[-1] = bad
        with pytest.raises((TypeError, ValueError)):
            _evaluate(**{field: values})

    @pytest.mark.parametrize(
        "changes",
        [
            {"classes": ()},
            {"classes": (1,) * 17},
            {"classes": (True, 2, 1)},
            {"classes": (Q(1), 2, 1)},
            {"classes": (0, 2, 1)},
            {"classes": (3, 2, 1)},
            {"contacts": ((0, 0),)},
            {"contacts": ((0, 3),)},
            {"contacts": ((-1, 0),)},
            {"contacts": ((0, True),)},
            {"contacts": ((0, Q(1)),)},
            {"contacts": ((0, 1), (1, 0))},
            {"contacts": ((0, 1, 2),)},
            {"contacts": "01"},
            {"contacts": {0: 1}},
            {"forms": (0,) * 14},
            {"phase_deviations": (0,) * 16},
            {"phase_origins": (0, 0)},
            {"phase_origins": {0, 1, 2}},
        ],
    )
    def test_support_and_shape_admission(self, monkeypatch, changes):
        def forbidden(*args):
            pytest.fail("invalid support reached matrix construction")

        monkeypatch.setattr(owner, "_central_port_geometry", forbidden)
        with pytest.raises((TypeError, ValueError)):
            _evaluate(**changes)

    def test_exact_tiny_inputs_and_sdk_projection(self, tmp_path):
        tiny = Q(1, 2**1200)
        report = _evaluate(forms=(tiny,) + (0,) * 14)
        assert report.forms[0] == tiny and report.form_storage > 0
        direct = report.to_dict()
        assert direct["schema"] == "tnfr.sine-port-composition-state.v1"
        assert relational_report_to_dict(report)["report"] == direct["report"]
        path = tmp_path / "rows.json"
        export_to_json(report, path)
        assert json_loads(path.read_bytes())["report"] == direct["report"]
        assert (
            get_type_hints(owner.evaluate_sine_port_composition)["return"]
            is owner.SinePortCompositionState
        )
        get_type_hints(owner.SinePortCompositionState)


class TestActualComposition:
    def test_frozen_complete_conditions_and_independent_bounds(self, frozen, mp):
        assert frozen.status == "certified_sine_port_composition"
        assert frozen.unavailable_reasons == ()
        assert (
            frozen.approximation_certified
            and frozen.identity_certified
            and frozen.work_within_allowance
        )
        assert frozen.unprobed_handoff.handoff_certified_by_class == (True, True)
        gamma, h = 1 / (1023 * mp.pi), _mp(mp, INPUTS["contact_duration"])
        sigma = mp.sin(mp.mpf(1) / 1000) / 2
        expected = (
            2
            * gamma**5
            * sigma**2
            * h**5
            / (5 * (1 - 2 * gamma**2 * h**2 / 3) ** 2 * (1 - 3 * h))
        )
        actual = _mp(mp, frozen.ideal_surrogate_discrepancy_upper_bound)
        assert actual >= expected
        assert actual - expected < mp.mpf("1e-60")
        eps, phi = INPUTS["endpoint_radius"], INPUTS["phase_origins"][1]
        assert frozen.joined_gap_lower_bound == Q(2, 135)
        assert frozen.joined_radius_squared_upper_bound == 6 * eps**2 + 6 * phi**2
        assert (
            frozen.joined_excess_storage_upper_bound
            == 18 * eps**2 + (phi + 2 * eps) ** 2
        )
        assert frozen.contact_work_upper_bound == 4 * eps**2 + (phi + 2 * eps) ** 2
        assert frozen.preparation_error_upper_bound == eps / (
            1 - 3 * INPUTS["contact_duration"]
        )
        assert (
            frozen.total_approximation_error_upper_bound
            == frozen.preparation_error_upper_bound
            + frozen.ideal_surrogate_discrepancy_upper_bound
        )
        _contains(
            mp, frozen.joined_cosine_bounds, mp.cos(4 * mp.pi / 9 + mp.sqrt(2) / 12)
        )
        assert frozen.joined_form_mean_bounds.contains(2 * eps / 29)
        assert frozen.joined_form_mean_bounds.contains(-2 * eps / 29)
        assert frozen.joined_phase_mean_bounds.contains(10 * phi / 29)

    def test_full_fine_gap_exact_semidefinite(self, frozen):
        _, matrix, _ = _fine(3, INPUTS["contacts"])
        gap = frozen.joined_gap_lower_bound
        residual = tuple(
            tuple(matrix[i][j] - gap * (Q(i == j) - Q(1, 27)) for j in range(27))
            for i in range(27)
        )
        assert exact_symmetric_semidefinite(residual)
        assert all(sum(row) == 0 for row in residual)

    def test_actual_zero_sum_endpoint_errors_energy_work_and_means(self, frozen, mp):
        rng = random.Random(12937)
        eps = INPUTS["endpoint_radius"]
        edges, _, degrees = _fine(3, INPUTS["contacts"])
        for _ in range(8):
            x, y = [], []
            for _component in range(3):
                for target in (x, y):
                    integers = [rng.randint(-8, 8) for _ in range(8)]
                    integers.append(-sum(integers))
                    scale = sum(abs(v) for v in integers) or 1
                    target.extend(eps * Q(v, scale) for v in integers)
            target = tuple(
                2 * INPUTS["classes"][i // 9] * mp.pi * (i % 9 - 4) / 9
                for i in range(27)
            )
            theta = tuple(
                target[i] + _mp(mp, INPUTS["phase_origins"][i // 9] + y[i])
                for i in range(27)
            )
            excess = mp.mpf(0)
            work = mp.mpf(0)
            for i, j in edges:
                term = _mp(mp, (x[i] - x[j]) ** 2) / 2 + 1 - mp.cos(theta[j] - theta[i])
                excess += term - (1 - mp.cos(target[j] - target[i]))
                if i // 9 != j // 9:
                    work += term
            assert excess <= _mp(mp, frozen.joined_excess_storage_upper_bound)
            assert work <= _mp(mp, frozen.contact_work_upper_bound)
            mean_x = sum((degrees[i] * x[i] for i in range(27)), Q(0)) / sum(degrees)
            mean_y = sum(
                (
                    degrees[i] * (INPUTS["phase_origins"][i // 9] + y[i])
                    for i in range(27)
                ),
                Q(0),
            ) / sum(degrees)
            assert frozen.joined_form_mean_bounds.contains(mean_x)
            assert frozen.joined_phase_mean_bounds.contains(mean_y)

    @pytest.mark.parametrize(
        "changes",
        [
            {"formation_time": 0},
            {"relaxation_duration": 0},
            {"endpoint_radius": Q(1, 2**500)},
        ],
    )
    def test_failed_actual_handoff_does_not_become_ideal_reset(self, changes):
        result = _assess(**changes)
        assert result.status == "unavailable"
        assert not all(result.unprobed_handoff.handoff_certified_by_class)
        assert result.total_approximation_error_upper_bound is None
        assert result.contact_work_upper_bound is None
        assert result.joined_excess_storage_upper_bound is None
        assert result.joined_form_mean_bounds is None

    @pytest.mark.parametrize(
        "changes,reason",
        [
            ({"work_allowance": 0}, "supplied_contact_work_allowance_not_certified"),
            (
                {"approximation_allowance": 0},
                "uniform_approximation_allowance_not_certified",
            ),
            ({"phase_origins": (0, 1, 0)}, "whole_network_identity_not_certified"),
            ({"contacts": ()}, "connected_multi_component_identity_not_supported"),
            (
                {"classes": (1,), "contacts": (), "phase_origins": (0,)},
                "connected_multi_component_identity_not_supported",
            ),
        ],
    )
    def test_unavailable_is_separate_from_other_diagnostics(self, changes, reason):
        result = _assess(**changes)
        assert result.status == "unavailable" and reason in result.unavailable_reasons
        assert result.total_approximation_error_upper_bound is not None
        if "contacts" in changes:
            assert result.approximation_certified and result.work_within_allowance
            assert result.joined_barrier_lower_bound is None

    def test_strict_accuracy_touching_and_closed_work_allowance(self, frozen):
        touching = _assess(
            approximation_allowance=frozen.total_approximation_error_upper_bound
        )
        assert (
            touching.approximation_margin_bounds.lo
            == touching.approximation_margin_bounds.hi
            == 0
        )
        assert not touching.approximation_certified
        work_touching = _assess(work_allowance=frozen.contact_work_upper_bound)
        assert (
            work_touching.work_margin_bounds.lo
            == work_touching.work_margin_bounds.hi
            == 0
        )
        assert work_touching.work_within_allowance
        assert work_touching.status == "certified_sine_port_composition"

    def test_zero_and_maximum_window_preserve_full_state_error(self):
        initial = _assess(contact_duration=0)
        assert initial.ideal_surrogate_discrepancy_upper_bound == 0
        assert (
            initial.total_approximation_error_upper_bound == INPUTS["endpoint_radius"]
        )
        endpoint = _assess(contact_duration=Q(1, 4), approximation_allowance=1)
        assert endpoint.preparation_error_upper_bound == 4 * INPUTS["endpoint_radius"]
        assert endpoint.approximation_certified

    @pytest.mark.parametrize("h", [Q(0), Q(1, 100), Q(1, 4)])
    def test_comparison_bounds_dominate_closed_linear_majorants(self, mp, h):
        # Exact solutions of the positive comparison systems provide a check
        # independent of the report's geometric-denominator implementation.
        t, gamma = _mp(mp, h), 1 / (1023 * mp.pi)
        eta, sigma = gamma**2, mp.sin(mp.mpf("0.001")) / 2
        denominator = 1 - 2 * eta * t**2 / 3
        v = sigma * mp.sinh(2 * gamma * t) / (2 * gamma)
        y = sigma * (mp.cosh(2 * gamma * t) - 1) / 2
        assert v <= sigma * t / denominator
        assert y <= eta * sigma * t**2 / denominator
        report = _assess(contact_duration=h, approximation_allowance=1)
        forcing = 2 * gamma**5 * sigma**2 / denominator**2
        exact_defect_comparison = forcing * mp.quad(
            lambda s: mp.exp(3 * (t - s)) * s**4, [0, t]
        )
        assert exact_defect_comparison <= _mp(
            mp, report.ideal_surrogate_discrepancy_upper_bound
        )
        assert _mp(mp, INPUTS["endpoint_radius"]) * mp.exp(3 * t) <= _mp(
            mp, report.preparation_error_upper_bound
        )

    @pytest.mark.parametrize(
        "field",
        [
            "formation_time",
            "relaxation_duration",
            "contact_duration",
            "form_error_bound",
            "phase_error_bound",
            "endpoint_radius",
            "radius",
            "work_allowance",
            "approximation_allowance",
        ],
    )
    @pytest.mark.parametrize("bad", [True, "0", 0j, float("nan"), float("inf")])
    def test_original_scalars_before_handoff(self, monkeypatch, field, bad):
        def forbidden(**kwargs):
            pytest.fail("invalid primitive reached source reconstruction")

        monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
        with pytest.raises((TypeError, ValueError)):
            _assess(**{field: bad})

    @pytest.mark.parametrize(
        "changes",
        [
            {"formation_time": -1},
            {"relaxation_duration": -1},
            {"contact_duration": -1},
            {"contact_duration": Q(1, 4) + Q(1, 2**200)},
            {"form_error_bound": -1},
            {"phase_error_bound": -1},
            {"endpoint_radius": 0},
            {"radius": 0},
            {"radius": Q(1, 12) + Q(1, 2**200)},
            {"work_allowance": -1},
            {"approximation_allowance": -1},
            {"decay_power": True},
            {"decay_power": Q(512)},
            {"decay_power": -1},
            {"decay_power": 4097},
        ],
    )
    def test_domain_rejection_before_handoff(self, monkeypatch, changes):
        def forbidden(**kwargs):
            pytest.fail("invalid domain reached source reconstruction")

        monkeypatch.setattr(owner, "_unprobed_handoff", forbidden)
        with pytest.raises(ValueError):
            _assess(**changes)

    def test_fresh_source_primitives_and_exponential_work_caps(
        self, monkeypatch, frozen
    ):
        calls = []
        original = owner._unprobed_handoff

        def observe(**kwargs):
            result = original(**kwargs)
            calls.append((kwargs, result))
            return result

        monkeypatch.setattr(owner, "_unprobed_handoff", observe)
        a = _assess(form_error_bound=0)
        b = _assess(form_error_bound=Q(1, 2**300))
        assert len(calls) == 2 and a.unprobed_handoff is not b.unprobed_handoff
        assert calls[0][0]["form_error_bound"] == 0
        assert calls[1][0]["form_error_bound"] == Q(1, 2**300)
        with pytest.raises(TypeError):
            owner.assess_sine_port_composition(unprobed_handoff=a.unprobed_handoff)
        with pytest.raises(ValueError, match="scaled_time"):
            _assess(formation_time=20481)
        with pytest.raises(ValueError, match="lyapunov_decay_rate"):
            _assess(
                relaxation_duration=Q(4097)
                / frozen.unprobed_handoff.lyapunov_decay_rate
            )

    def test_sdk_projection_and_retained_response(self, frozen, tmp_path):
        direct = frozen.to_dict()
        assert direct["schema"] == "tnfr.sine-port-composition.v1"
        assert relational_report_to_dict(frozen)["report"] == direct["report"]
        path = tmp_path / "composition.json"
        export_to_json(frozen, path)
        assert json_loads(path.read_bytes())["report"] == direct["report"]
        artifact = (
            Path(__file__).parents[2]
            / "docs/assets/sine_formed_classes/port-composition-v1.json"
        )
        retained = json_loads(artifact.read_bytes())
        assert retained["schema"] == direct["schema"]
        assert retained["report"] == direct["report"]
        assert retained["algebraic_control"]["passed"]
        assert retained["frozen_stopping_rule_passed"]
        assert (
            get_type_hints(owner.assess_sine_port_composition)["return"]
            is owner.SinePortComposition
        )
        get_type_hints(owner.SinePortComposition)
