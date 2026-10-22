"""Read-only reconstruction of the retained mobile endpoint assessment.

The spectral boxes are retained numerical evidence, not regenerated here.
Their mathematical reductions are checked conditionally on those boxes;
archive receipts associate declared bytes and do not authenticate execution
or independently prove every integrand evaluation. No frozen producer runs.
"""

import importlib.util
from dataclasses import asdict
from fractions import Fraction as Q
from math import factorial
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import INTERVAL_BITS, I, pi_interval, sqrt
from tnfr.physics import _localized_mobile_response as response_owner
from tnfr.physics import _localized_mobile_spectral as spectral_owner
from tnfr.physics._localized_mobile_interaction import (
    _bound_localized_mobile_interaction,
)
from tnfr.research.artifact_io import (
    decode_exact_tree,
    encode_exact_tree,
    exact_record,
    file_receipt,
    read_bytes_bounded,
)
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/localized_mobile_interaction"


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    endpoints = exact_record(value["lo"]), exact_record(value["hi"])
    enclosed = I(*endpoints)
    assert (enclosed.lo, enclosed.hi) == endpoints
    return enclosed


def _ordinary_int(value):
    assert type(value) is int
    return value


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(EVIDENCE / name, max_bytes=32 * 1024**2))
    )


@pytest.fixture(scope="module")
def retained():
    specification = importlib.util.spec_from_file_location(
        "_retained_mobile_response_driver",
        ROOT / "scripts/run_localized_mobile_response.py",
    )
    runner = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(runner)

    def forbidden(*args, **kwargs):
        pytest.fail("retained mobile evidence audit attempted scientific execution")

    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    before = tuple(file_receipt(path) for path in paths)
    with pytest.MonkeyPatch.context() as patch:
        for owner, names in (
            (runner, ("freeze", "evaluate", "_evaluate_response")),
            (
                response_owner,
                (
                    "_evaluate_localized_mobile_response",
                    "_prepare_mobile_spectral_kernel",
                ),
            ),
            (spectral_owner, ("_prepare_mobile_spectral_kernel", "_mode_propagator")),
        ):
            for name in names:
                patch.setattr(owner, name, forbidden)
        # This audit permits pinned Git/ZIP reads, but does not require the
        # current source tree to remain byte-identical to the frozen archive.
        association = runner.audit(ROOT, require_current=False)
        records = {
            name: _read(name)
            for name in (
                "protocol.json",
                "freeze.json",
                "source-manifest.json",
                "attempt.json",
                "report.json",
            )
        }
        with forbid_organized_response_evaluation():
            yield records, association
    assert tuple(file_receipt(path) for path in paths) == before
    assert tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file())) == paths


@pytest.fixture(scope="module")
def primitives_and_budget(retained):
    policy = retained[0]["protocol.json"]["policy"]
    model = RelationalExchangeModel(**policy["model"])
    arguments = dict(policy["arguments"])
    order = _ordinary_int(arguments.pop("quadrature_order"))
    panel_limit = _ordinary_int(arguments.pop("max_panels"))
    budget = _bound_localized_mobile_interaction(reference_model=model, **arguments)
    return policy, arguments, order, panel_limit, budget


def _reconstruct(report, arguments, order, panel_limit, budget):
    """Rebuild consumed reductions; stored bounds and verdicts are not inputs."""
    a = exact_record(arguments["amplitude"])
    horizon = _ordinary_int(arguments["horizon"])
    size = _ordinary_int(arguments["node_count"])
    fejer = _ordinary_int(arguments["fejer_order"])
    target = a**3 / 1000
    sigma = (
        pi_interval().hi * budget.form_loss_coefficient / budget.exchange_coefficient
    )
    qmax = 2 * pi_interval().hi * Q(fejer - 1, size)
    growth = (1 + sigma) * (4 + 6 * qmax)
    derivative = 8 * a**3 * growth ** (2 * order)
    gauss_constant = Q(
        factorial(order) ** 4, (2 * order + 1) * factorial(2 * order) ** 3
    )
    panels = 1
    while True:
        quadrature = (
            derivative
            * horizon ** (2 * order + 1)
            * gauss_constant
            / panels ** (2 * order)
        )
        if quadrature <= target / 10 or 2 * panels > panel_limit:
            break
        panels *= 2
    carrier, radius = size // 4, 3 * (fejer - 1)
    positives = tuple(range(carrier - radius, carrier + radius + 1))
    modes = tuple(sorted(positives + tuple(-mode for mode in positives)))
    assert tuple(_ordinary_int(mode) for mode in report["spectral_modes"]) == modes
    raw_boxes = report["spectral_quadrature"]
    assert type(raw_boxes) is tuple and len(raw_boxes) == len(modes)
    assert all(type(row) is tuple and len(row) == 4 for row in raw_boxes)
    boxes = tuple(tuple(_interval(value) for value in row) for row in raw_boxes)
    arithmetic = sqrt(
        I(sum((value.radius**2 for row in boxes for value in row), Q(0)))
    ).hi
    norm = sqrt(sum((value**2 for row in boxes for value in row), I(0)))
    numerical_error = quadrature + max(arithmetic, norm.radius)

    def expand(value, allowance):
        return I(max(Q(0), value.lo - allowance), value.hi + allowance)

    cubic = expand(norm, quadrature)
    nominal = expand(cubic, budget.nominal_cubic_remainder_upper_bound)
    uniform = expand(nominal, budget.shared_preparation_sensitivity_upper_bound)
    numerical = numerical_error <= target
    identity = budget.prepared_identity_certified
    reasons = []
    if not numerical:
        reasons.append("interval_arithmetic_budget_exhausted")
    if not identity:
        reasons.append("prepared_identity_not_certified")
    return (
        {
            "quadrature_order": order,
            "panel_limit": panel_limit,
            "panels": panels,
            "numerical_error_target": target,
            "derivative_growth_upper_bound": growth,
            "integrand_derivative_upper_bound": derivative,
            "quadrature_error_upper_bound": quadrature,
            "arithmetic_error_upper_bound": arithmetic,
            "quadrature_norm_bounds": norm,
            "numerical_error_upper_bound": numerical_error,
            "cubic_norm_bounds": cubic,
            "nominal_defect_norm_bounds": nominal,
            "uniform_defect_norm_bounds": uniform,
            "integrand_evaluations": panels * order,
            "numerical_certified": numerical,
            "nominal_interaction_certified": numerical and identity and nominal.lo > 0,
            "uniform_interaction_certified": numerical and identity and uniform.lo > 0,
            "status": "evaluated" if numerical else "unavailable",
            "unavailable_reasons": tuple(reasons),
        },
        modes,
        boxes,
    )


@pytest.fixture(scope="module")
def reconstructed(retained, primitives_and_budget):
    _, arguments, order, panel_limit, budget = primitives_and_budget
    report = retained[0]["report.json"]["report"]
    return _reconstruct(report, arguments, order, panel_limit, budget)


def test_retained_attempt_archive_and_protocol_associations_are_read_only(retained):
    records, association = retained
    assert association["status"] == "byte_associations_verified"
    assert association["source_or_execution_authenticated"] is False
    assert association["existing_outcomes"] == ("attempt.json", "report.json")
    assert records["freeze.json"]["association_only_not_authentication"] is True
    assert records["report.json"]["source_or_execution_authenticated"] is False
    assert records["protocol.json"]["evaluation_status_at_freeze"] == "not_evaluated"
    assert records["attempt.json"]["evaluation_status_at_start"] == "not_evaluated"
    assert records["attempt.json"]["evaluation_count"] == 1
    assert (
        records["report.json"]["schema"] == "tnfr.localized-mobile-response-outcome.v1"
    )
    assert records["protocol.json"]["runtime"]["interval_bits"] == INTERVAL_BITS == 128
    archived = {item["path"] for item in records["source-manifest.json"]["files"]}
    imported = records["report.json"]["imported_sources_at_completion"]
    assert imported and set(imported.values()) <= archived
    assert all(
        (name == "tnfr" or name.startswith("tnfr.")) and path.startswith("src/tnfr/")
        for name, path in imported.items()
    )


def test_primitive_preparation_and_budgets_are_readmitted(
    retained, primitives_and_budget
):
    policy, arguments, order, panel_limit, budget = primitives_and_budget
    assert arguments == {
        "node_count": 1024,
        "fejer_order": 16,
        "amplitude": Q(1, 4096),
        "initial_state_error_bound": Q(1, 4096000),
        "localization_radius": 96,
        "horizon": 256,
    }
    assert order == 12 and panel_limit == 65536
    assert policy["numerical_target"] == budget.amplitude**3 / 1000
    assert policy["node_order"] == tuple(range(1024))
    assert policy["maximum_evaluations"] == 1 and policy["seed"] is None
    assert budget.form_loss_coefficient == Q(1, 2**20)
    assert budget.exchange_coefficient == 1 - Q(1, 2**20)
    assert budget.prepared_identity_certified and budget.cubic_remainder_certified
    expected = decode_exact_tree(encode_exact_tree(asdict(budget)))
    assert retained[0]["report.json"]["report"]["budget"] == expected


def test_nonlinear_and_shared_source_allowances_follow_from_primitive_geometry(
    primitives_and_budget,
):
    _, arguments, _, _, budget = primitives_and_budget
    a, eps, horizon = budget.amplitude, budget.initial_state_error_bound, budget.horizon
    size, fejer = arguments["node_count"], arguments["fejer_order"]
    pi_hi = pi_interval().hi
    linear = (
        2 * pi_hi * budget.form_loss_coefficient * horizon / budget.exchange_coefficient
    )
    linear += Q(horizon, 6) * (2 * pi_hi * Q(fejer - 1, size)) ** 3
    nonlinear = Q(32, 3) * horizon * a**3
    single_error = eps + a * linear + nonlinear
    paired_error = 2 * eps + sqrt(I(2)).hi * a * linear + nonlinear
    square_sum = sum((1 - Q(abs(mode), fejer)) ** 2 for mode in range(1 - fejer, fejer))
    peak = a * fejer / sqrt(I(size * square_sum)).lo
    single_peak, paired_peak = peak + single_error, 2 * peak + paired_error
    single_radius, paired_radius = Q(9, 8) * a, Q(3, 2) * a
    assert a + eps + Q(4, 3) * horizon * single_radius**3 < single_radius
    assert (
        sqrt(I(2)).hi * a + 2 * eps + Q(4, 3) * horizon * paired_radius**3
        < paired_radius
    )
    fifth = (Q(4, 15) * horizon + Q(8, 3) * horizon**2) * (
        paired_peak**4 * paired_radius + 2 * single_peak**4 * single_radius
    )
    exponents = (4 * single_peak**2 * horizon, 4 * paired_peak**2 * horizon)
    assert all(0 < exponent < 1 for exponent in exponents)
    shared = 2 * eps * sum(exponent / (1 - exponent) for exponent in exponents)
    assert budget.nominal_cubic_remainder_upper_bound == fifth
    assert budget.shared_preparation_sensitivity_upper_bound == shared
    assert budget.interaction_proxy_error_upper_bound == fifth + shared


def test_stored_spectral_boxes_retain_both_channels_all_modes_and_exact_symmetry(
    reconstructed,
):
    _, modes, boxes = reconstructed
    assert len(modes) == 182 and 0 not in modes
    coordinates = dict(zip(modes, boxes))
    for mode, row in coordinates.items():
        assert row == coordinates[-mode]
        assert row[1] == row[3] == I(0)
    # This shape/symmetry check is conditional on the retained boxes. It is
    # not a replay or an independent proof of each archived integrand value.


def test_gauss_norm_error_and_both_verdicts_are_reconstructed(retained, reconstructed):
    expected, _, boxes = reconstructed
    report = retained[0]["report.json"]["report"]
    for name, value in expected.items():
        if isinstance(value, I):
            assert _interval(report[name]) == value
        elif type(value) is bool:
            assert type(report[name]) is bool and report[name] is value
        elif type(value) is Q:
            assert exact_record(report[name]) == value
        else:
            assert type(report[name]) is type(value) and report[name] == value
    assert expected["panels"] == 64 and expected["integrand_evaluations"] == 768
    assert (
        expected["quadrature_error_upper_bound"]
        <= expected["numerical_error_target"] / 10
    )
    # Independently sum the exact box extrema before any dyadic reduction.
    square_lo = sum(
        (
            0 if value.contains(0) else min(value.lo**2, value.hi**2)
            for row in boxes
            for value in row
        ),
        Q(0),
    )
    square_hi = sum(
        (max(value.lo**2, value.hi**2) for row in boxes for value in row), Q(0)
    )
    independent_norm = sqrt(I(square_lo, square_hi))
    assert expected["quadrature_norm_bounds"].contains(independent_norm)


def test_cached_verdicts_and_bounds_are_not_reconstruction_premises(
    retained, primitives_and_budget, reconstructed
):
    _, arguments, order, panel_limit, budget = primitives_and_budget
    report = dict(retained[0]["report.json"]["report"])
    report.update(
        numerical_certified=None,
        nominal_interaction_certified=None,
        uniform_interaction_certified=None,
        numerical_error_upper_bound="untrusted cached field",
        cubic_norm_bounds=None,
        budget={"prepared_identity_certified": False},
    )
    assert _reconstruct(report, arguments, order, panel_limit, budget) == reconstructed


@pytest.mark.parametrize(
    "record",
    [
        {"lo": True, "hi": 1},
        {"lo": 0.0, "hi": 1},
        {"lo": Q(1, 3), "hi": 1},
        {"lo": 1, "hi": 0},
        {"lo": 0, "hi": 1, "cached": True},
    ],
)
def test_retained_interval_coordinates_require_exact_ordered_grid_endpoints(record):
    with pytest.raises((AssertionError, TypeError, ValueError)):
        _interval(record)
