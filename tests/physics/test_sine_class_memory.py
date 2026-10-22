"""Conditional memory budgets, source retention and shared-reference controls."""

import json
import subprocess
from decimal import Decimal
from fractions import Fraction as Q
from inspect import Parameter, signature
from pathlib import Path

import numpy as np
import pytest

from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_memory as owner
from tnfr.sdk.relational_reports import relational_report_to_dict
from tnfr.utils.io import json_loads


def _arguments():
    return dict(
        probe_amplitude=Q(1, 4096),
        contact_duration=Q(1, 16384),
        endpoint_radius=Q(1, 2**108),
        readout_error_bound=Q(1, 2**110),
    )


@pytest.fixture(scope="module", autouse=True)
def no_source_or_response_acquisition():
    from tnfr.mathematics import _validated_taylor
    from tnfr.physics import (
        _sine_formed_contact,
        relational_sine_class_mediation,
        relational_sine_formed_classes,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("conditional memory must not acquire a source or replay a producer")

    with pytest.MonkeyPatch.context() as patch:
        for module, names in (
            (
                _validated_taylor,
                ("flow_jets", "picard_tube", "validated_box_taylor_step"),
            ),
            (
                _sine_formed_contact,
                ("_unprobed_handoff", "assess_sine_formed_class_pair"),
            ),
            (
                relational_sine_class_mediation,
                ("_unprobed_handoff", "assess_sine_class_mediation"),
            ),
            (relational_sine_formed_classes, ("assess_sine_formed_class_pair",)),
            (subprocess, ("run", "Popen")),
        ):
            for name in names:
                patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def memory():
    return owner.derive_sine_class_mediated_memory()


@pytest.fixture(scope="module")
def bound():
    return owner.bound_sine_class_mediated_memory(**_arguments())


def test_fixed_structure_and_four_primitive_budget_have_no_report_admission():
    assert not signature(owner.derive_sine_class_mediated_memory).parameters
    parameters = signature(owner.bound_sine_class_mediated_memory).parameters
    assert set(parameters) == set(_arguments())
    assert all(
        p.kind == Parameter.KEYWORD_ONLY and p.default == Parameter.empty
        for p in parameters.values()
    )
    for name in (
        "source_handoff",
        "classes",
        "kernel",
        "visible_state",
        "hidden_state",
        "formation_time",
        "radius",
        "work_allowance",
    ):
        with pytest.raises(TypeError):
            owner.bound_sine_class_mediated_memory(**_arguments(), **{name: None})


@pytest.mark.parametrize("name", tuple(_arguments()))
@pytest.mark.parametrize(
    "value", (True, np.bool_(False), float("nan"), float("inf"), -1, Decimal("1e-400"))
)
def test_scalar_admission_precedes_structure(name, value, monkeypatch):
    def forbidden():
        pytest.fail("invalid scalar must be rejected before building the model")

    monkeypatch.setattr(owner, "derive_sine_class_mediated_memory", forbidden)
    values = _arguments()
    values[name] = value
    with pytest.raises((TypeError, ValueError)):
        owner.bound_sine_class_mediated_memory(**values)


def test_exact_horizon_cap_precedes_structure(monkeypatch):
    monkeypatch.setattr(
        owner, "derive_sine_class_mediated_memory", lambda: pytest.fail("cap bypass")
    )
    values = _arguments()
    values["contact_duration"] = Q(1, 4) + Q(1, 2**400)
    with pytest.raises(ValueError):
        owner.bound_sine_class_mediated_memory(**values)


def test_shared_partitioner_is_applied_once_to_exact_full_spatial_matrix(
    monkeypatch, memory
):
    original = owner.derive_coordinate_memory
    calls = []

    def partition(matrix, visible):
        calls.append((matrix, visible))
        return original(matrix, visible)

    monkeypatch.setattr(owner, "derive_coordinate_memory", partition)
    assert owner.derive_sine_class_mediated_memory() == memory
    assert len(calls) == 1
    matrix, visible = calls[0]
    assert len(matrix) == 27 and all(len(row) == 27 for row in matrix)
    assert all(type(value) is Q for row in matrix for value in row)
    assert visible == tuple(range(9)) + tuple(range(18, 27))
    assert memory.visible_indices == visible + tuple(node + 27 for node in visible)
    assert len(set(memory.visible_indices) | set(memory.hidden_indices)) == 54
    assert not set(memory.visible_indices) & set(memory.hidden_indices)


def test_unscaled_hidden_source_and_kernel_have_finite_nonzero_envelopes(bound):
    m, h, eps = bound.memory, bound.contact_duration, bound.endpoint_radius
    g = m.gamma_bounds.hi
    assert m.hidden_to_visible_norm_upper_bound == (1 + g) / 3
    assert m.visible_to_hidden_norm_upper_bound == (1 + g) / 2
    assert bound.whole_window_kernel_norm_upper_bound == (1 + g) ** 2 / (
        6 * (1 - 3 * h)
    )
    assert (
        bound.hidden_initial_source_norm_upper_bound
        == (1 + g) * eps / (3 * (1 - 3 * h))
        > 0
    )
    # Original phase units avoid introducing an artificial eps/gamma factor.
    assert bound.hidden_initial_source_norm_upper_bound < eps


def test_source_matched_error_is_integral_of_the_full_nonlinear_taylor_defect(bound):
    a, eps, h = bound.probe_amplitude, bound.endpoint_radius, bound.contact_duration
    g, d = bound.memory.gamma_bounds.hi, bound.bootstrap_margin
    assert d == 1 - 2 * g**2 * h**2
    for amplitude, form, error in (
        (a, bound.probe_form_norm_upper_bound, bound.probe_tangent_error_upper_bound),
        (
            Q(0),
            bound.baseline_form_norm_upper_bound,
            bound.baseline_tangent_error_upper_bound,
        ),
    ):
        assert d * form == amplitude + eps + 2 * g * eps * h
        # Integrate 2*g*(eps+2*g*form*t)^2 coefficient by coefficient.
        defect_coefficients = (
            2 * g * eps**2,
            8 * g**2 * eps * form,
            8 * g**3 * form**2,
        )
        integral = sum(
            coefficient * h ** (power + 1) / (power + 1)
            for power, coefficient in enumerate(defect_coefficients)
        )
        assert error * (1 - 3 * h) == integral
    assert bound.paired_class_reduction_error_upper_bound == 2 * (
        bound.probe_tangent_error_upper_bound + bound.baseline_tangent_error_upper_bound
    )
    assert (
        bound.nonlinear_class_contrast_bounds
        == bound.tangent_class_contrast_bounds
        + I(
            -bound.paired_class_reduction_error_upper_bound,
            bound.paired_class_reduction_error_upper_bound,
        )
    )
    assert bound.status == "certified_conditional_contrast"


def test_zero_radius_recovers_existing_ideal_remainder_without_a_handoff():
    values = _arguments()
    values["endpoint_radius"] = 0
    report = owner.bound_sine_class_mediated_memory(**values)
    reference = owner._mediation_reference_bounds(
        owner._central_port_geometry(3, owner._CONTACTS),
        report.probe_amplitude,
        report.contact_duration,
    )
    assert (
        report.baseline_form_norm_upper_bound
        == report.baseline_tangent_error_upper_bound
        == 0
    )
    assert report.hidden_initial_source_norm_upper_bound == 0
    assert report.paired_class_reduction_error_upper_bound <= reference.nonlinear


def test_paired_cancellation_does_not_add_linear_source_error_to_reduction():
    values = _arguments()
    values["endpoint_radius"] = Q(1, 2**40)
    report = owner.bound_sine_class_mediated_memory(**values)
    old_initial_error = 4 * report.endpoint_radius / (1 - 3 * report.contact_duration)
    assert report.paired_class_reduction_error_upper_bound < old_initial_error / 10**12
    assert report.response_certified
    assert report.hidden_initial_source_norm_upper_bound > 0
    # The nonlinear stationary comparator still retains independent visible
    # initial errors; paired LINEAR cancellation cannot erase those errors.
    assert report.quasistatic_nonlinear_recorded_contrast_bounds.hi >= old_initial_error
    assert not report.quasistatic_comparator_excluded
    assert report.status == "bounds_only"


@pytest.mark.parametrize("name", ("probe_amplitude", "contact_duration"))
def test_zero_probe_or_time_cannot_separate_the_classes(name):
    values = _arguments()
    values[name] = 0
    report = owner.bound_sine_class_mediated_memory(**values)
    assert report.leading_contrast_bounds == I(0)
    assert report.recorded_class_contrast_bounds.contains(0)
    assert not report.response_certified and not report.quasistatic_comparator_excluded


def test_strict_response_threshold_and_distinct_stationary_bands():
    values = _arguments()
    values["readout_error_bound"] = 0
    reference = owner.bound_sine_class_mediated_memory(**values)
    values["readout_error_bound"] = reference.nonlinear_class_contrast_bounds.lo / 4
    report = owner.bound_sine_class_mediated_memory(**values)
    assert report.recorded_class_contrast_bounds.lo == 0
    assert not report.response_certified
    assert report.quasistatic_tangent_recorded_contrast_bounds == I(
        -4 * report.readout_error_bound, 4 * report.readout_error_bound
    )
    assert (
        report.quasistatic_nonlinear_recorded_contrast_bounds.hi
        > report.quasistatic_tangent_recorded_contrast_bounds.hi
    )
    assert (
        report.quasistatic_exclusion_margin_bounds
        == report.recorded_class_contrast_bounds
        - report.quasistatic_nonlinear_recorded_contrast_bounds.hi
    )


def test_finite_wide_horizon_retains_bounds_without_separation():
    values = _arguments()
    values["contact_duration"] = Q(1, 4)
    report = owner.bound_sine_class_mediated_memory(**values)
    assert report.bootstrap_margin > 0
    assert report.recorded_class_contrast_bounds.lo < 0
    assert report.status == "bounds_only"


def test_exact_subgrid_source_and_represented_time_are_not_coerced():
    values = _arguments()
    values.update(endpoint_radius=Q(1, 10**400), contact_duration=0.125)
    report = owner.bound_sine_class_mediated_memory(**values)
    assert report.endpoint_radius == Q(1, 10**400) > 0
    assert report.hidden_initial_source_norm_upper_bound > 0
    assert report.contact_duration == Q(1, 8)


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"numerator", "denominator"}:
            return Q(value["numerator"], value["denominator"])
        if set(value) == {"lo", "hi"}:
            return I(_decode(value["lo"]), _decode(value["hi"]))
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    return value


def test_shared_reference_extraction_preserves_retained_arithmetic_exactly():
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/assets/sine_formed_classes/class-mediation-v1.json"
    )
    retained = _decode(
        json_loads(path.read_text(encoding="utf-8"))["response"]["report"]
    )
    reference = owner._mediation_reference_bounds(
        owner._central_port_geometry(3, owner._CONTACTS),
        retained["probe_amplitude"],
        retained["contact_duration"],
    )
    for new, old in (
        ("gamma", "gamma_bounds"),
        ("eta", "eta_bounds"),
        ("cosines", "mediator_cosine_bounds"),
        ("difference", "mediator_cosine_difference_bounds"),
        ("common", "common_diffusion_walk_coefficient"),
        ("walk", "mediator_walk_coefficient"),
        ("leading", "leading_contrast_bounds"),
        ("tail", "linear_tail_upper_bound"),
        ("denominator", "nonlinear_bootstrap_margin"),
        ("nonlinear", "nonlinear_contrast_error_upper_bound"),
        ("ideal", "ideal_nonlinear_contrast_bounds"),
    ):
        assert getattr(reference, new) == retained[old]


@pytest.mark.parametrize("fixture_name", ("memory", "bound"))
def test_actual_report_sdk_and_json_projection_is_exact(request, fixture_name):
    report = request.getfixturevalue(fixture_name)
    direct = report.to_dict()
    projected = relational_report_to_dict(report)
    assert projected["report"] == direct["report"]
    assert projected["report_type"] == type(report).__name__
    assert json_loads(json.dumps(projected, allow_nan=False)) == projected
    assert "formation_certified" not in projected["report"]
    assert "identity_certified" not in projected["report"]
    assert "work_within_allowance" not in projected["report"]
