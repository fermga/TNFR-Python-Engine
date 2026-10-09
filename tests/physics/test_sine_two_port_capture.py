"""Static independent controls for the frozen two-port capture producer.

No test in this owner reruns the reserved long reference trajectory. The
folded field is checked directly against every original node at synthetic
states; artifact audits separately own retained evaluated evidence.
"""

from fractions import Fraction as Q
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_two_port_capture as owner
from tnfr.physics.relational_sine_two_port_transit import assess_sine_two_port_transit
from tnfr.utils.io import json_loads


@pytest.fixture(scope="module")
def preparation():
    return assess_sine_two_port_transit(
        form_error_radius=Q(1, 65536), phase_error_radius=Q(1, 65536)
    )


@pytest.fixture(scope="module")
def folded(preparation):
    return owner._folded_reference(preparation)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 85
    return context


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _inside(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def test_exact_folded_geometry_retains_complete_weighted_norm_and_common_gauge(
    preparation, folded
):
    p, e = folded.permutation, folded.reconstruction_matrix
    assert tuple(i for i, j in enumerate(p) if i == j) == (5, 14)
    assert all(p[p[i]] == i for i in range(18))
    assert folded.representatives == (0, 2, 3, 4, 9, 11, 12, 13)
    edges = set(preparation.geometry.edges)
    assert {tuple(sorted((p[i], p[j]))) for i, j in edges} == edges
    for i in range(8):
        assert sum(preparation.degrees[node] * e[node][i] for node in range(18)) == 0
        for j in range(8):
            actual = sum(
                preparation.degrees[node] * e[node][i] * e[node][j]
                for node in range(18)
            )
            assert folded.metric[i][j] == actual
    assert tuple(folded.metric[i][i] for i in range(8)) == (6, 4, 4, 4, 6, 4, 4, 4)
    # The source has an affine circular reflection, not a homogeneous real-lift
    # symmetry. Its two fixed-node phases differ by pi, which must be retained.
    turns = preparation.nominal_phase_turns
    origin_twice = turns[0] + turns[1]
    assert turns[5] - turns[14] == Q(1, 2)
    assert all(
        (turns[i] + turns[p[i]] - origin_twice).denominator == 1 for i in range(18)
    )


def test_canonical_affine_angles_cover_every_edge_without_redundant_sines(
    preparation, folded
):
    assert len(folded.angle_turns) == 11
    for edge, ((i, j), turn) in enumerate(
        zip(preparation.geometry.edges, preparation.nominal_edge_turns)
    ):
        index, sign = folded.edge_angle_indices[edge], folded.edge_angle_signs[edge]
        assert sign * folded.angle_turns[index] == turn
        assert tuple(
            sign * value for value in folded.angle_coefficients[index]
        ) == tuple(
            b - a
            for a, b in zip(
                folded.reconstruction_matrix[i], folded.reconstruction_matrix[j]
            )
        )
    for row, node in zip(folded.current_coefficients, folded.representatives):
        independent = [Q(0)] * len(folded.angle_turns)
        for (i, j), index, sign in zip(
            preparation.geometry.edges,
            folded.edge_angle_indices,
            folded.edge_angle_signs,
        ):
            independent[index] += Q(
                (int(node == i) - int(node == j)) * sign, preparation.degrees[node]
            )
        assert row == tuple(independent)


@pytest.mark.parametrize("scale", (Q(0), Q(1, 2000), -Q(1, 3000)))
def test_sparse_interval_field_equals_every_full_node_and_retains_acute_margins(
    preparation, folded, mp, scale
):
    h = tuple(scale * (i - 3) for i in range(8))
    phases = tuple(
        2 * mp.pi * _mp(mp, turn)
        + sum(_mp(mp, coefficient * value) for coefficient, value in zip(row, h))
        for turn, row in zip(
            preparation.nominal_phase_turns, folded.reconstruction_matrix
        )
    )
    fine = [mp.mpf(0)] * 18
    for i, j in preparation.geometry.edges:
        current = mp.sin(phases[j] - phases[i])
        fine[i] += current / preparation.degrees[i]
        fine[j] -= current / preparation.degrees[j]
    flow, domain = owner._reference_field(folded)
    reduced = flow(tuple(I(value) for value in h))
    for row, node in zip(reduced, folded.representatives):
        _inside(mp, row, fine[node])
    for node, row in enumerate(folded.reconstruction_matrix):
        expected = sum(
            _mp(mp, coefficient) * fine[index]
            for coefficient, index in zip(row, folded.representatives)
        )
        assert abs(fine[node] - expected) < mp.mpf("1e-77")
    margins = domain(tuple(I(value) for value in h))
    assert len(margins) == 20 and min(margins) > 0
    for margin, (i, j), offset in zip(
        margins, preparation.geometry.edges, preparation.edge_integer_offsets
    ):
        actual = (
            mp.pi / 2
            - abs(phases[j] - phases[i] - 2 * mp.pi * offset)
            - _mp(mp, Q(1, 2048))
        )
        assert _mp(mp, margin) <= actual


def test_jet_jacobian_retains_gradient_metric_nonexpansion(preparation, folded, mp):
    h = tuple(Q(i - 3, 2500) for i in range(8))
    flow, _ = owner._reference_field(folded)
    columns = []
    for column in range(8):
        values = tuple(
            Jet((I(value), I(int(i == column)))) for i, value in enumerate(h)
        )
        columns.append(tuple(row.coeffs[1] for row in flow(values)))
    phases = tuple(
        2 * mp.pi * _mp(mp, turn)
        + sum(_mp(mp, coefficient * value) for coefficient, value in zip(row, h))
        for turn, row in zip(
            preparation.nominal_phase_turns, folded.reconstruction_matrix
        )
    )
    hessian = mp.zeros(18)
    for i, j in preparation.geometry.edges:
        cosine = mp.cos(phases[j] - phases[i])
        assert cosine > 0
        hessian[i, i] += cosine
        hessian[j, j] += cosine
        hessian[i, j] -= cosine
        hessian[j, i] -= cosine
    reconstruction = mp.matrix(
        [[_mp(mp, value) for value in row] for row in folded.reconstruction_matrix]
    )
    pulled = reconstruction.T * hessian * reconstruction
    for i in range(8):
        for j in range(8):
            expected = -pulled[i, j] / _mp(mp, folded.metric[i][i])
            _inside(mp, columns[j][i], expected)
    assert min(mp.eigsy(pulled, eigvals_only=True)) > 0


def test_compact_step_projects_retained_evidence_without_changing_radius_or_tube():
    source = SimpleNamespace(
        time=Q(2),
        duration=Q(1, 4),
        initial_center=(Q(3), Q(5)),
        initial_radius=Q(1, 100),
        tube=(I(2, 4), I(4, 6)),
        picard_interior_margin=Q(1, 10),
        domain_lower_bounds=(Q(1, 32),),
        local_metric_error_upper_bound=Q(1, 1000),
        endpoint_center=(Q(13, 4), Q(21, 4)),
        endpoint_radius=Q(11, 1000),
    )
    compact = owner._compact_step(source)
    for name in compact.__dataclass_fields__:
        assert getattr(compact, name) == getattr(source, name)
    assert (
        compact.endpoint_radius
        == compact.initial_radius + compact.local_metric_error_upper_bound
    )


def _arguments():
    return dict(
        form_error_radius=Q(1, 65536),
        phase_error_radius=Q(1, 65536),
        reference_duration=1024,
        time_step=Q(1, 4),
        order=8,
        max_steps=4096,
    )


@pytest.mark.parametrize(
    "field,value",
    (
        ("form_error_radius", True),
        ("phase_error_radius", np.bool_(True)),
        ("form_error_radius", -1),
        ("phase_error_radius", float("nan")),
        ("reference_duration", float("inf")),
        ("reference_duration", 0),
        ("reference_duration", 1025),
        ("time_step", 0),
        ("time_step", Q(1, 2)),
        ("order", True),
        ("order", 0),
        ("order", 17),
        ("order", 8.0),
        ("max_steps", False),
        ("max_steps", 0),
        ("max_steps", 4097),
        ("max_steps", 1),
    ),
)
def test_invalid_declared_domains_reject_before_any_source_or_target_evaluation(
    field, value, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input reached a scientific producer")

    monkeypatch.setattr(owner, "assess_sine_two_port_transit", forbidden)
    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(owner, "validated_metric_taylor_step", forbidden)
    arguments = _arguments()
    arguments[field] = value
    with pytest.raises((TypeError, ValueError)):
        owner.assess_sine_two_port_capture(**arguments)


def test_all_preparation_and_numerical_primitives_are_required():
    parameters = signature(owner.assess_sine_two_port_capture).parameters
    assert tuple(parameters) == tuple(_arguments())
    assert all(value.default is value.empty for value in parameters.values())


@pytest.fixture(scope="module")
def retained_target():
    """Read existing target evidence for interface tests without any producer.

    Only the fields consumed at this wiring boundary are projected. This is
    not a new scientific target admission or provenance authentication.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/sine_formed_classes/two-port-compatibility-v1.json"
    )
    saved = json_loads(path.read_text(encoding="utf-8"))["report"]

    def interval(value):
        return I(
            *(
                Q(value[key]["numerator"], value[key]["denominator"])
                for key in ("lo", "hi")
            )
        )

    return SimpleNamespace(
        acute_margin_turns_bounds=interval(saved["acute_margin_turns_bounds"]),
        local_attraction_certified=saved["local_attraction_certified"],
        target_phase_bounds=tuple(map(interval, saved["target_phase_bounds"])),
    )


def _target_wiring(monkeypatch, target):
    calls = []

    def replacement(**kwargs):
        assert kwargs == dict(
            classes=(2, 1), outer_refinements=32, inner_refinements=64
        )
        calls.append(kwargs)
        return target

    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", replacement)
    return calls


def _controlled_step(
    center, radius, duration, flow, domain, *, endpoint, endpoint_radius, **kwargs
):
    """Deliberate interface fixture; it establishes no scientific flow claim."""
    assert kwargs["growth_rate"] == 0
    assert kwargs["order"] == 8
    assert callable(flow) and callable(domain)
    return SimpleNamespace(
        time=kwargs["time"],
        duration=duration,
        initial_center=center,
        initial_radius=radius,
        tube=tuple(I(-1, 1) for _ in center),
        picard_interior_margin=Q(1, 100),
        domain_lower_bounds=(Q(1, 64),) * 20,
        local_metric_error_upper_bound=max(Q(0), endpoint_radius - radius),
        endpoint_center=endpoint,
        endpoint_radius=endpoint_radius,
    )


def test_energy_precheck_abstains_before_target_or_reference_evaluation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("failed preparation must not evaluate target or reference")

    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(owner, "validated_metric_taylor_step", forbidden)
    result = owner.assess_sine_two_port_capture(
        **(_arguments() | dict(form_error_radius=Q(1, 100), phase_error_radius=0))
    )
    assert not result.preparation_admitted
    assert result.comparison_margin > 0
    assert result.unavailable_reasons == (
        "strict_initial_excess_storage_budget_not_certified",
    )
    assert result.target is None and result.reference_steps == ()
    assert result.validated_reference_duration == 0
    assert not result.capture_certified and result.status == "unavailable"


def test_exact_comparison_equality_abstains_even_with_admitted_energy(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("zero comparison margin cannot start target or reference work")

    monkeypatch.setattr(owner, "assess_sine_two_port_compatibility", forbidden)
    monkeypatch.setattr(owner, "validated_metric_taylor_step", forbidden)
    # Solve the documented linear comparison inequality exactly at Q=1/2048.
    ratio = Q(1, 100000)
    radius = (Q(1, 2048) * (1 - 2 * ratio) - 24 * ratio - ratio * Q(5, 12)) / 7
    result = owner.assess_sine_two_port_capture(
        **(_arguments() | dict(form_error_radius=0, phase_error_radius=radius))
    )
    assert result.preparation.energy_budget_admitted
    assert result.phase_error_candidate == Q(1, 2048)
    assert result.comparison_margin == 0
    assert result.unavailable_reasons == (
        "strict_full_horizon_comparison_margin_not_certified",
    )
    assert result.target is None and result.reference_steps == ()
    assert not result.preparation_admitted and not result.capture_certified


def test_unavailable_target_leaves_reference_work_unstarted(
    retained_target, monkeypatch
):
    target = SimpleNamespace(
        **(vars(retained_target) | dict(local_attraction_certified=False))
    )
    calls = _target_wiring(monkeypatch, target)
    monkeypatch.setattr(
        owner,
        "validated_metric_taylor_step",
        lambda *args, **kwargs: pytest.fail(
            "unavailable target reached reference integration"
        ),
    )
    result = owner.assess_sine_two_port_capture(**_arguments())
    assert len(calls) == 1
    assert result.preparation_admitted and not result.target_admitted
    assert result.unavailable_reasons == (
        "required_implicit_target_geometry_not_certified",
    )
    assert result.reference_steps == () and result.validated_reference_duration == 0
    assert result.reference_target_distance_upper_bound is None


def test_failed_step_preserves_the_last_valid_prefix_without_retry(
    retained_target, monkeypatch
):
    _target_wiring(monkeypatch, retained_target)
    endpoint = tuple(Q(i + 1, 1000) for i in range(8))
    failed = (I(-2, 2),) * 8
    calls = []

    def step(center, radius, metric, duration, flow, domain, **kwargs):
        calls.append((center, radius, kwargs["time"]))
        if len(calls) == 1:
            result = _controlled_step(
                center,
                radius,
                duration,
                flow,
                domain,
                endpoint=endpoint,
                endpoint_radius=Q(1, 10**6),
                **kwargs,
            )
            return result, None, None
        assert len(calls) == 2
        return None, failed, "controlled_metric_failure"

    monkeypatch.setattr(owner, "validated_metric_taylor_step", step)
    result = owner.assess_sine_two_port_capture(
        **(_arguments() | dict(reference_duration=Q(1, 2), max_steps=2))
    )
    assert calls == [((Q(0),) * 8, Q(0), Q(0)), (endpoint, Q(1, 10**6), Q(1, 4))]
    assert len(result.reference_steps) == 1
    assert result.validated_reference_duration == Q(1, 4)
    assert result.reference_endpoint_center == endpoint
    assert result.reference_endpoint_radius == Q(1, 10**6)
    assert result.reference_minimum_acute_margin == Q(1, 64) + Q(1, 2048)
    assert result.failed_tube == failed
    assert result.unavailable_reasons == ("controlled_metric_failure",)
    assert not result.reference_validated and not result.reference_endpoint_certified
    assert not result.full_comparison_certified and not result.capture_certified
    assert result.endpoint_excess_storage_upper_bound is None


def test_reference_endpoint_outside_target_ball_cannot_become_capture(
    retained_target, monkeypatch
):
    _target_wiring(monkeypatch, retained_target)

    def step(center, radius, metric, duration, flow, domain, **kwargs):
        result = _controlled_step(
            center,
            radius,
            duration,
            flow,
            domain,
            endpoint=center,
            endpoint_radius=Q(1, 10**7),
            **kwargs,
        )
        return result, None, None

    monkeypatch.setattr(owner, "validated_metric_taylor_step", step)
    result = owner.assess_sine_two_port_capture(
        **(_arguments() | dict(reference_duration=Q(1, 4), max_steps=1))
    )
    assert result.reference_validated
    assert result.reference_target_distance_upper_bound > Q(1, 2048)
    assert not result.reference_endpoint_certified
    assert not result.full_comparison_certified and not result.capture_certified
    assert result.endpoint_phase_distance_upper_bound is None
    assert result.unavailable_reasons == (
        "reference_endpoint_target_distance_not_certified",
    )


def test_controlled_reference_endpoint_connects_to_analytic_tail_and_capture(
    retained_target, preparation, folded, monkeypatch
):
    calls = _target_wiring(monkeypatch, retained_target)
    pi = owner.pi_interval()
    endpoint = tuple(
        (
            retained_target.target_phase_bounds[i]
            - 2 * pi * preparation.nominal_phase_turns[i]
        ).midpoint
        for i in folded.representatives
    )
    endpoint_radius = Q(1, 10**7)

    def step(center, radius, metric, duration, flow, domain, **kwargs):
        result = _controlled_step(
            center,
            radius,
            duration,
            flow,
            domain,
            endpoint=endpoint,
            endpoint_radius=endpoint_radius,
            **kwargs,
        )
        return result, None, None

    monkeypatch.setattr(owner, "validated_metric_taylor_step", step)
    result = owner.assess_sine_two_port_capture(
        **(_arguments() | dict(reference_duration=Q(1, 4), max_steps=1))
    )
    assert len(calls) == 1
    assert result.reference_endpoint_certified and result.full_comparison_certified
    assert result.reference_target_distance_upper_bound >= endpoint_radius
    assert result.reference_target_distance_upper_bound < Q(1, 2048)
    assert result.full_slow_horizon == Q(5, 4)
    assert result.full_horizon_pi_squared_coefficient == Q(1023 * 1024 * 5, 4)
    assert (
        result.endpoint_phase_distance_upper_bound
        == result.reference_target_distance_upper_bound + result.phase_error_candidate
    )
    assert result.endpoint_relative_form_norm_upper_bound < Q(1, 8192)
    assert result.endpoint_phase_distance_upper_bound < Q(1, 1024)
    assert (
        result.endpoint_excess_storage_upper_bound
        == result.endpoint_phase_distance_upper_bound**2
        + result.endpoint_relative_form_norm_upper_bound**2
    )
    assert (
        result.capture_storage_margin
        == Q(1, 648000) - result.endpoint_excess_storage_upper_bound
        > 0
    )
    assert result.capture_certified and result.status == "certified_capture"
    assert result.unavailable_reasons == ()
