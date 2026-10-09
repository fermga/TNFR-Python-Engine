"""Independent cubic-law balances and retained same-source obstruction.

No producer or trajectory is replayed. The known historical response is
loaded once by the module fixture; algebraic controls use separate states.
"""

from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import pytest

EDGES = tuple((i, (i + 1) % 5) for i in range(5)) + tuple((i, i + 5) for i in range(5))
ETA = Q(1, 100)


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _potential(gap, eta):
    cosine = mp.cos(gap)
    return 1 - cosine + eta * (1 - cosine) ** 2 * (cosine + 2) / 3


def _full_storage(form, phase, eta):
    return sum(
        (form[j] - form[i]) ** 2 / 2 + _potential(phase[j] - phase[i], eta)
        for i, j in EDGES
    )


def test_cubic_current_is_the_full_storage_gradient_and_preserves_joint_work():
    with mp.workdps(100):
        eta = _mp(ETA)
        form = tuple(_mp(Q((i * 7) % 11 - 5, 13)) for i in range(10))
        phase = tuple(_mp(Q((i * 5) % 13 - 6, 7)) for i in range(10))
        current, contrast, degree = [mp.mpf(0)] * 10, [mp.mpf(0)] * 10, [0] * 10
        for i, j in EDGES:
            gap = phase[j] - phase[i]
            sine = mp.sin(gap)
            edge_current = sine + eta * sine**3
            assert abs(
                mp.diff(lambda t: _potential(t, eta), gap) - edge_current
            ) < mp.mpf("1e-95")
            current[i] += edge_current
            current[j] -= edge_current
            contrast[i] += form[i] - form[j]
            contrast[j] += form[j] - form[i]
            degree[i] += 1
            degree[j] += 1
        form_rate = tuple(current[i] / degree[i] for i in range(10))
        phase_rate = tuple(contrast[i] / degree[i] for i in range(10))
        energy_rate = mp.mpf(0)
        for i in range(10):
            form_gradient = mp.diff(
                lambda value: _full_storage(
                    form[:i] + (value,) + form[i + 1 :], phase, eta
                ),
                form[i],
            )
            phase_gradient = mp.diff(
                lambda value: _full_storage(
                    form, phase[:i] + (value,) + phase[i + 1 :], eta
                ),
                phase[i],
            )
            assert abs(form_gradient - contrast[i]) < mp.mpf("1e-95")
            assert abs(phase_gradient + current[i]) < mp.mpf("1e-95")
            energy_rate += form_gradient * form_rate[i] + phase_gradient * phase_rate[i]
        assert abs(energy_rate) < mp.mpf("1e-94")
        assert abs(sum(degree[i] * form_rate[i] for i in range(10))) < mp.mpf("1e-95")
        assert abs(sum(degree[i] * phase_rate[i] for i in range(10))) < mp.mpf("1e-95")
        assert _full_storage(form, phase, eta) > sum(
            (form[j] - form[i]) ** 2 / 2 + _potential(phase[j] - phase[i], eta)
            for i, j in EDGES[:5]
        )


@pytest.mark.parametrize("orientation", (-1, 1))
def test_sector_boundary_cost_uses_four_path_edges_and_the_opposite_face(orientation):
    # On the wider unit-winding sector face the oriented gaps are
    # (2pi/3, pi/3, pi/3, pi/3, pi/3), up to simultaneous reversal.
    def exact_potential_from_cosine(cosine):
        return 1 - cosine + ETA * (1 - cosine) ** 2 * (cosine + 2) / 3

    positive_face = exact_potential_from_cosine(
        Q(-1, 2)
    ) + 4 * exact_potential_from_cosine(Q(1, 2))
    opposite_face = 5 * exact_potential_from_cosine(Q(-1, 2))
    assert positive_face == Q(7, 2) + Q(47, 24) * ETA
    assert opposite_face > positive_face
    with mp.workdps(100):
        tangent_point = mp.pi / 3
        slope = mp.sin(tangent_point) * (1 + _mp(ETA) * mp.sin(tangent_point) ** 2)
        for numerator in range(-16, 17):
            gap = mp.pi * numerator / 24
            residual = (
                _potential(gap, _mp(ETA))
                - _potential(tangent_point, _mp(ETA))
                - slope * (gap - tangent_point)
            )
            assert residual >= -mp.mpf("1e-95")
        gaps = (orientation * 2 * mp.pi / 3,) + (orientation * mp.pi / 3,) * 4
        assert abs(sum(gaps) - orientation * 2 * mp.pi) < mp.mpf("1e-95")
        assert abs(
            sum(_potential(gap, _mp(ETA)) for gap in gaps) - _mp(positive_face)
        ) < mp.mpf("1e-95")


@pytest.mark.parametrize("geometry", ("saddle", "uniform_twist"))
def test_critical_skeleton_keeps_inertia_by_leaf_congruence_not_uniform_full_scaling(
    geometry,
):
    with mp.workdps(100):
        angle = mp.pi / 3 if geometry == "saddle" else 2 * mp.pi / 5
        receiver = tuple((i - 2) * angle for i in range(5))
        contact_signs = (1, -1, 1, -1, 1)
        phase = receiver + tuple(
            receiver[i] + (mp.pi if contact_signs[i] < 0 else 0) for i in range(5)
        )
        matrices = []
        for eta in (mp.mpf(0), _mp(ETA)):
            hessian = mp.matrix(10)
            current = [mp.mpf(0)] * 10
            for i, j in EDGES:
                gap = phase[j] - phase[i]
                sine, cosine = mp.sin(gap), mp.cos(gap)
                force = sine + eta * sine**3
                current[i] += force
                current[j] -= force
                weight = mp.diff(lambda value: _potential(value, eta), gap, 2)
                assert abs(weight - cosine * (1 + 3 * eta * sine**2)) < mp.mpf("1e-95")
                hessian[i, i] += weight
                hessian[j, j] += weight
                hessian[i, j] -= weight
                hessian[j, i] -= weight
            assert max(map(abs, current)) < mp.mpf("1e-94")
            matrices.append(hessian)
        # A leaf increment = its receiver increment + an independent contact
        # increment. Congruence then splits cycle and contact quadratic forms.
        change = mp.eye(10)
        for i in range(5):
            change[5 + i, i] = 1
        factor = 1 + 3 * _mp(ETA) * mp.sin(angle) ** 2
        reduced = tuple(change.T * matrix * change for matrix in matrices)
        for i in range(10):
            for j in range(10):
                if i < 5 and j < 5:
                    assert abs(reduced[1][i, j] - factor * reduced[0][i, j]) < mp.mpf(
                        "1e-94"
                    )
                elif i >= 5 and j >= 5:
                    expected = contact_signs[i - 5] if i == j else 0
                    assert abs(reduced[1][i, j] - expected) < mp.mpf("1e-94")
                else:
                    assert abs(reduced[1][i, j]) < mp.mpf("1e-94")
        assert abs(matrices[1][5, 5] - factor * matrices[0][5, 5]) > mp.mpf("0.001")
        inertias = []
        for matrix in matrices:
            eigenvalues = tuple(mp.eigsy(matrix, eigvals_only=True))
            inertias.append(
                (
                    sum(value < -mp.mpf("1e-80") for value in eigenvalues),
                    sum(abs(value) <= mp.mpf("1e-80") for value in eigenvalues),
                )
            )
        assert inertias[0] == inertias[1] == (3 if geometry == "saddle" else 2, 1)


@pytest.fixture(scope="module")
def retained():
    from tnfr.research.sine_constitutive_robustness import (
        assess_sine_constitutive_robustness,
    )

    return assess_sine_constitutive_robustness()


def test_retained_source_keeps_full_form_and_contact_storage_before_changing_law(
    retained,
):
    assert len(retained.source.nodes) == 10
    assert {frozenset(edge) for edge in retained.source.edges} == {
        frozenset(edge) for edge in EDGES
    }
    assert len(retained.initial_box) == len(retained.source_image_box) == 20
    for value, bound in zip(
        retained.source.epi + retained.source.phase, retained.initial_box
    ):
        assert bound.lo <= value - retained.declared_initial_coordinate_radius
        assert value + retained.declared_initial_coordinate_radius <= bound.hi
    with mp.workdps(110):
        form = tuple(map(_mp, retained.source.epi))
        phase = tuple(map(_mp, retained.source.phase))
        initial_energy = _full_storage(form, phase, mp.mpf(0))
        assert (
            _mp(retained.initial_sine_storage_bounds.lo)
            <= initial_energy
            <= _mp(retained.initial_sine_storage_bounds.hi)
        )
        image_phase = tuple(
            _mp(value.midpoint) for value in retained.source_image_box[10:]
        )
        added = tuple(
            (1 - mp.cos(image_phase[j] - image_phase[i])) ** 2
            * (2 + mp.cos(image_phase[j] - image_phase[i]))
            / 3
            for i, j in EDGES
        )
        assert sum(added[5:]) > 0
        assert (
            _mp(retained.added_phase_storage_bounds.lo)
            <= sum(added)
            <= _mp(retained.added_phase_storage_bounds.hi)
        )
        assert initial_energy + _mp(retained.eta) * sum(added) < _mp(
            retained.sector_barrier
        )
    # The conserved old-law energy comes from the original cube. No energy
    # equality is assigned to every independent point in the outer image box.
    assert retained.perturbed_source_storage_bounds.hi >= (
        retained.initial_sine_storage_bounds.hi
        + retained.eta * retained.added_phase_storage_bounds.hi
    )
    assert retained.sector_barrier == Q(7, 2) + retained.eta * Q(47, 24)
    assert retained.eta == ETA
    assert (
        retained.energy_margin_lower_bound
        == retained.sector_barrier - retained.perturbed_source_storage_bounds.hi
        > 0
    )
    assert retained.same_source_acute_formation_excluded


def test_actual_source_branch_geometry_is_zero_winding_and_sdk_preserves_scope(
    retained,
):
    from tnfr.sdk import relational_report_to_dict

    with mp.workdps(100):
        total = mp.mpf(0)
        for (i, j), offset in zip(EDGES[:5], retained.source_turn_offsets):
            raw_lower = _mp(
                retained.source_image_box[10 + j].lo
                - retained.source_image_box[10 + i].hi
            )
            raw_upper = _mp(
                retained.source_image_box[10 + j].hi
                - retained.source_image_box[10 + i].lo
            )
            lower, upper = (
                raw_lower - 2 * mp.pi * offset,
                raw_upper - 2 * mp.pi * offset,
            )
            assert -mp.pi < lower <= upper < mp.pi
            total += (lower + upper) / 2
        assert abs(total) < mp.mpf("1e-90")
    assert sum(retained.source_turn_offsets) == 0
    assert retained.source_time == 237
    assert retained.reference_target_window == (467, 471)
    assert not retained.independent_numerical_source_ball_certified
    assert not retained.retained_protocol_passed
    exported = relational_report_to_dict(retained)
    assert exported["schema"] == "tnfr.relational-report.v1"
    assert exported["report_type"] == "SineConstitutiveRobustness"
    assert exported["report"] == retained.to_dict()["report"]
    assert retained.to_dict()["schema"] == "tnfr.sine-constitutive-robustness.v1"
    assert exported["report"]["independent_numerical_source_ball_certified"] is False
    assert exported["report"]["retained_protocol_passed"] is False


def test_altered_bundle_rejects_before_archive_or_json_parsing(tmp_path, monkeypatch):
    from tnfr.research import sine_constitutive_robustness as owner

    (tmp_path / "response-v1.evidence.zip").write_bytes(b"not the frozen archive")

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported bytes must reject before archive or JSON parsing")

    monkeypatch.setattr(owner.zipfile, "ZipFile", unexpected)
    monkeypatch.setattr(owner, "json_loads", unexpected)
    with pytest.raises(ValueError, match="changed frozen evidence"):
        owner.assess_sine_constitutive_robustness(tmp_path)


def test_missing_bundle_is_not_replaced_by_unrelated_local_evidence(tmp_path):
    from tnfr.research import sine_constitutive_robustness as owner

    with pytest.raises(FileNotFoundError):
        owner.assess_sine_constitutive_robustness(tmp_path)


@pytest.mark.parametrize(
    "value", (True, float("nan"), float("inf"), {"numerator": True, "denominator": 1})
)
def test_projected_interval_admission_rejects_ambiguous_scalars(value):
    from tnfr.research import sine_constitutive_robustness as owner

    with pytest.raises((TypeError, ValueError)):
        owner._interval({"lo": value, "hi": 1})


def test_literal_declaration_and_rebuilt_state_cannot_substitute_a_changed_law(
    retained,
):
    from tnfr.research import sine_constitutive_robustness as owner
    from tnfr.utils.io import json_loads

    declaration = json_loads(
        (Path(owner.DEFAULT_EVIDENCE_DIRECTORY) / "declaration.json").read_bytes()
    )
    primitive = retained.source.to_dict()["report"]
    report = {"source": primitive, "cycle_indices": list(retained.cycle)}
    rebuilt, _, _, _ = owner._rebuild_source(report, declaration)
    assert rebuilt.epi == retained.source.epi
    # Metadata that happens to compare equal numerically is not admitted as
    # a physical scalar, even when the copied source still looks canonical.
    primitive["reference_model"]["phase_weight"] = True
    with pytest.raises((TypeError, ValueError)):
        owner._rebuild_source(report, declaration)
