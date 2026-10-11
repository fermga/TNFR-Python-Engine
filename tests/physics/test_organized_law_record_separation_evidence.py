"""Read-only audit of both declarations and the retained separation certificate.

No moment, scientific producer or full certificate consumer is replayed.
Receipts associate exact bytes and source; they do not authenticate execution
or turn this conditional mathematical distinction into a physical observation.
"""

from fractions import Fraction as Q
import hashlib
import importlib
from pathlib import Path
import subprocess

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr._exact_time import exp_unit_bounds
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research.artifact_io import decode_exact_tree, exact_record, file_receipt
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
DECLARATION = "theory/nodal/ORGANIZED_LAW_OBSERVATION.md"
SPECS = (
    ("organized_law_contrast_kernels", "0b21a040270c7c7653cd09e9f3076fdb1ca9e08f"),
    ("organized_law_record_separation", "b9b2e49a8146d6544407eed55a395377d91504c4"),
)
FROZEN_PARENTS = (
    "src/tnfr/physics/_organized_computable_prediction.py",
    "src/tnfr/physics/_organized_observed_prediction.py",
    "src/tnfr/physics/_organized_observed_forecast.py",
    "src/tnfr/physics/_organized_finite_observation.py",
    "src/tnfr/physics/_organized_finite_observation_source.py",
    "src/tnfr/physics/_organized_robust_current.py",
    "benchmarks/organized_computable_prediction.py",
    "benchmarks/organized_observed_prediction.py",
    "benchmarks/organized_observed_forecast.py",
    "benchmarks/organized_finite_observation.py",
    "benchmarks/organized_robust_current.py",
    "docs/assets/organized_computable_prediction",
    "docs/assets/organized_observed_prediction",
    "docs/assets/organized_observed_forecast",
    "docs/assets/organized_finite_observation",
    "docs/assets/organized_robust_current",
)


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


def _interval(value):
    assert type(value) is dict and set(value) == {"lo", "hi"}
    endpoints = exact_record(value["lo"]), exact_record(value["hi"])
    result = I(*endpoints)
    assert (result.lo, result.hi) == endpoints
    return result


@pytest.fixture(scope="module")
def records():
    result = {}
    for name, _ in SPECS:
        directory = ROOT / "docs/assets" / name
        raw = tuple(
            (directory / filename).read_bytes()
            for filename in ("attempt.json", "result.json")
        )
        decoded = tuple(
            decode_exact_tree(json_loads(item.decode("utf-8"))) for item in raw
        )
        result[name] = (*decoded, *raw)
    return result


@pytest.fixture(scope="module")
def source_snapshots(records):
    # Resolve all Git reads before the execution guard excludes subprocesses.
    snapshots = {}
    try:
        for name, revision in SPECS:
            attempt = records[name][0]
            declaration = _git("show", revision + ":" + DECLARATION)
            snapshots[name] = {
                "listing": _git(
                    "ls-tree",
                    "-r",
                    "-z",
                    "--full-tree",
                    revision,
                    "--",
                    *attempt["source_scope"],
                ),
                "src": _git("rev-parse", revision + ":src").decode().strip(),
                "declaration": declaration,
            }
        snapshots["parent_changes"] = _git(
            "diff", "--name-only", "cd0f8f30", SPECS[1][1], "--", *FROZEN_PARENTS
        )
        snapshots["uncommitted_parent_changes"] = _git(
            "diff", "--name-only", "--", *FROZEN_PARENTS
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return snapshots


@pytest.fixture(scope="module", autouse=True)
def exclude_execution_and_preserve_files(records, source_snapshots):
    targets = (
        (
            "tnfr.physics._organized_law_contrast_kernels",
            (
                "_assess_organized_law_contrast_kernels",
                "_enclose_moment",
                "_ramp_monomial_moment",
                "_three_ramp_contrast",
            ),
        ),
        (
            "tnfr.physics._organized_law_record_separation",
            (
                "_certify_organized_law_record_separation",
                "_bound_late_history",
                "_three_ramp_contrast",
            ),
        ),
        (
            "tnfr.physics._organized_law_joint_bounds",
            ("_bound_organized_law_joint_phases",),
        ),
        (
            "tnfr.physics._organized_law_observation",
            ("_certify_organized_phase_circle",),
        ),
        (
            "tnfr.research._organized_law_record_inputs",
            ("_read_organized_law_record_moments",),
        ),
        (
            "tnfr.research._single_analytic_assessment",
            ("_run_single_analytic_assessment",),
        ),
    )
    imported = tuple(
        (importlib.import_module(name), functions) for name, functions in targets
    )
    paths = tuple(
        ROOT / "docs/assets" / name / filename
        for name, _ in SPECS
        for filename in ("attempt.json", "result.json")
    )
    paths += tuple(ROOT / name for name in FROZEN_PARENTS if (ROOT / name).is_file())
    before = tuple(file_receipt(path) for path in paths)

    def forbidden(*args, **kwargs):
        pytest.fail("read-only evidence audit attempted scientific evaluation")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, functions in imported:
            for function in functions:
                patch.setattr(module, function, forbidden)
        yield
    assert tuple(file_receipt(path) for path in paths) == before


@pytest.mark.parametrize("name,revision", SPECS)
def test_both_committed_declarations_source_inventories_and_attempts(
    name, revision, records, source_snapshots
):
    attempt, result, raw_attempt, _ = records[name]
    evaluator = "benchmarks/" + name + ".py"
    assert attempt["source_revision"] == result["source_revision"] == revision
    assert attempt["schema"] == name + "_attempt_v1"
    assert result["schema"] == name + "_result_v1"
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert (
        attempt["arithmetic_method"]
        == "rational_outward_dyadic128_machin_trigonometry_v1"
    )
    assert attempt["declaration"] == DECLARATION and attempt["evaluator"] == evaluator
    assert attempt["source_scope"] == (
        "src/tnfr",
        evaluator,
        DECLARATION,
        "pyproject.toml",
    )
    assert result["attempt"] == {
        "path": "attempt.json",
        "bytes": len(raw_attempt),
        "sha256": hashlib.sha256(raw_attempt).hexdigest(),
    }
    directory = ROOT / "docs/assets" / name
    assert {path.name for path in directory.iterdir()} == {
        "attempt.json",
        "result.json",
    }
    if source_snapshots is None:
        pytest.skip("retained source revisions unavailable in this checkout")
    snapshot = source_snapshots[name]
    inventory = {}
    for entry in filter(None, snapshot["listing"].split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = {"mode": mode, "blob": blob}
    assert inventory == attempt["source_inventory"]
    assert (
        hashlib.sha256(snapshot["listing"]).hexdigest()
        == attempt["source_inventory_sha256"]
    )
    assert snapshot["src"] == attempt["source_tree"]
    declaration = snapshot["declaration"]
    git_payload = b"blob " + str(len(declaration)).encode("ascii") + b"\0" + declaration
    assert hashlib.sha1(git_payload).hexdigest() == inventory[DECLARATION]["blob"]
    text = declaration.decode("utf-8")
    assert "### Prospective scalar-moment evaluation" in text
    assert "Mode 2 is selected before evaluation" in text
    assert "radius at most `1e-28`" in text
    if name == SPECS[1][0]:
        assert "### Frozen final assessment declaration" in text
        assert "shared reference s3=9.00009" in text
        assert "STRICT squared" in text
    assert {DECLARATION, evaluator, "src/tnfr/physics/" + "_" + name + ".py"} <= set(
        inventory
    )
    for record in (attempt, result):
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert not source_snapshots["parent_changes"]
    assert not source_snapshots["uncommitted_parent_changes"]


def test_cross_record_receipts_and_preselected_primitive_inventory(records):
    kernel, final = (records[name][1] for name, _ in SPECS)
    receipts = tuple(
        {"path": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in zip(("attempt.json", "result.json"), records[SPECS[0][0]][2:])
    )
    assert final["report"]["kernel_source_receipts"] == receipts
    p = kernel["report"]
    assert p["terms"] == 20 and p["harmonics"] == (1, 2, 9, 18, 36)
    assert p["selected_mode"] == 2 and p["interval_bits"] == 128
    assert p["kernel_radius_budget"] == Q(1, 10**28)
    assert tuple((item["mode"], item["harmonic"]) for item in p["kernels"]) == tuple(
        (mode, harmonic) for mode in (1, 2) for harmonic in (1, 2, 9, 18, 36)
    )
    for item in p["kernels"]:
        for field in ("sine_hyperbolic", "cosine_hyperbolic"):
            assert _interval(item[field]).radius <= Q(1, 10**28)
    assert p["maximum_kernel_radius"] == max(
        _interval(item[field]).radius
        for item in p["kernels"]
        for field in ("sine_hyperbolic", "cosine_hyperbolic")
    )
    selected = tuple(item for item in p["kernels"] if item["mode"] == 2)
    consumed = final["report"]["certificate"]["moments"]
    assert len(selected) == len(consumed) == 5
    for old, new in zip(selected, consumed):
        for field in ("mode", "harmonic", "mu", "sine_hyperbolic", "cosine_hyperbolic"):
            assert old[field] == new[field]
        assert new["sine_truncation"] == new["cosine_truncation"] == 0
    assert p["scope"] == "moment_kernel_arithmetic_only_not_a_record_verdict"


@pytest.fixture(scope="module")
def reconstructed_contrast(records):
    p = records[SPECS[1][0]][1]["report"]["certificate"]
    moments = {item["harmonic"]: item for item in p["moments"]}
    C = {n: _interval(item["sine_hyperbolic"]) for n, item in moments.items()}
    D = {n: _interval(item["cosine_hyperbolic"]) for n, item in moments.items()}
    mu = (5 + sqrt(5)) / 8
    assert all(_interval(item["mu"]) == mu for item in moments.values())
    s, c = sin(Q(1, 20)), cos(Q(1, 20))
    alpha_e, alpha_o = s * C[1] / (c * D[1] - D[2]), s * D[1] / (c * C[1] + C[2])
    even = tuple(
        sin(Q(n, 20)) * C[n] - alpha_e * (cos(Q(n, 20)) * D[n] - D[2 * n])
        for n in (9, 18)
    )
    odd = tuple(
        sin(Q(n, 20)) * D[n] - alpha_o * (cos(Q(n, 20)) * C[n] + C[2 * n])
        for n in (9, 18)
    )
    return p, mu, alpha_e, alpha_o, even, odd


def test_contrast_algebra_uses_primitive_moments_not_archived_coefficients(
    reconstructed_contrast,
):
    p, _, alpha_e, alpha_o, even, odd = reconstructed_contrast
    contrast = p["contrast"]
    assert _interval(contrast["alpha_even"]) == alpha_e
    assert _interval(contrast["alpha_odd"]) == alpha_o
    assert tuple(map(_interval, contrast["even_coefficients"])) == even
    assert tuple(map(_interval, contrast["odd_coefficients"])) == odd
    assert 0 < alpha_e.lo <= alpha_e.hi < 1 < alpha_o.lo
    assert even[1].hi < 0 < odd[1].lo
    # Both parities share one harmonic phase; absolute signs are retained.
    assert max((2 * even[0] / even[1]).abs_max, (2 * odd[0] / odd[1]).abs_max) < Q(
        29, 100
    )


@pytest.fixture(scope="module")
def primitive_joint_bounds():
    g, T, h, X = Q(1, 3000), Q(16), Q(1, 100000), Q(1, 100)
    laws = []
    for coefficient in (0, 1):
        J, L, B = (
            1 + Q(coefficient, 1458),
            1 + Q(2 * coefficient, 243),
            1 + Q(coefficient, 9),
        )
        s, a, b = 2 * g * T, Q(31, 10**9), Q(601, 10**10)
        hidden_phase = (b + s * L * a) / (1 - (s * L) ** 2)
        source = T * (g * B * hidden_phase**2 / 9) / (1 - L * (g * T) ** 2)
        form = X + g * J * T / 2
        drift = 4 * g * X + 2 * g**2 * J * T
        second = form + g * J / 2 + g**2 * L * form
        laws.append((J, L, source, form, drift, second))
    delta = Q(2, 10**14) + sum((law[2] for law in laws), Q(0))
    return g, T, h, X, tuple(laws), delta


def test_joint_bounds_retain_both_laws_sources_and_prove_winding_phasor_limits(
    records, primitive_joint_bounds
):
    g, _, h, _, laws, delta = primitive_joint_bounds
    p = records[SPECS[1][0]][1]["report"]["certificate"]["joint_bounds"]
    assert p["sample_difference_upper"] == delta < Q(201, 10**16)
    assert p["carrier_form_bounds"] == tuple(law[3] for law in laws)
    assert p["carrier_phase_drift_bounds"] == tuple(law[4] for law in laws)
    assert p["plateau_second_derivative_bounds"] == tuple(law[5] for law in laws)
    reading = (2 + h / 2) * delta / h
    trapezoid = sum((law[5] for law in laws), Q(0)) * h**2 / 12
    drift_error = g * sum((law[1] * law[4] for law in laws), Q(0)) / 2
    kappa = 4 * 3300 * (reading + trapezoid + drift_error)
    angle, epsilon = Q(7, 5), Q(1, 10)
    cosine = 1 - angle**2 / 2 + angle**4 / 24 - angle**6 / 720
    command_cosine_defect = epsilon**2 / 2 - epsilon**4 / 24
    sine_error = 2 * kappa / command_cosine_defect
    deformation = 3 * (9 * epsilon) ** 2 / (2 * 2187 * command_cosine_defect)
    gap = (sine_error + deformation) / cosine
    assert p["initial_gap_difference_upper"] == gap
    assert p["same_winding_margin"] == 6 - 5 * gap > 0
    zeta = reading + trapezoid + g**2 * h * sum((law[1] * law[3] for law in laws), Q(0))
    assert p["initial_current_divergence_upper"] == zeta < Q(41, 10**10)
    sine_radius = 3 + 2 * zeta * (3300 * 4 * 2187)
    initial_radius = sine_radius / cosine
    reference_radius = initial_radius + 2187 * sum((law[4] for law in laws), Q(0))
    assert p["fundamental_sine_radius_ratio"] == sine_radius
    assert p["initial_fundamental_radius_ratio"] == initial_radius
    assert p["reference_fundamental_radius_ratio"] == reference_radius < 20


def _exponential(value):
    lower, _ = exp_unit_bounds(value.lo)
    _, upper = exp_unit_bounds(value.hi)
    return I(lower, upper)


def _decay_for_audit(mu, time):
    return (1 / _exponential(mu * time / 16)) ** 16


def test_late_grid_and_independent_coarse_phase_and_difference_bounds(
    records, primitive_joint_bounds
):
    _, _, h, X, laws, delta = primitive_joint_bounds
    p = records[SPECS[1][0]][1]["report"]["certificate"]["late_bounds"]
    readings = tuple(
        3 * child * (1 + h) + stage * (1 + h) + endpoint
        for child in range(5)
        for stage in range(3)
        for endpoint in (Q(0), h)
    )
    endpoints = tuple(
        tuple(
            (
                3 * child * (1 + h) + stage * (1 + h) + h,
                3 * child * (1 + h) + stage * (1 + h) + h + 1,
            )
            for stage in range(3)
        )
        for child in (2, 3)
    )
    assert p["observation_times"] == readings
    assert p["ramp_endpoints"] == endpoints
    assert all(
        value in readings for block in endpoints for pair in block for value in pair
    )
    start, reference, end = (3 * child * (1 + h) for child in (2, 3, 4))
    assert (p["interval_start"], p["reference_time"], p["interval_end"]) == (
        start,
        reference,
        end,
    )
    L = reference - start
    assert p["maximum_reference_distance"] == L == end - reference
    root = sqrt(5)
    modes = (5 - root) / 8, (5 + root) / 8
    rows = 1 + 1 / root, 1 + 3 / root
    for endpoint in (start, end):
        row_bound = sum(
            (
                row.hi
                * (
                    _decay_for_audit(mu, endpoint) - _decay_for_audit(mu, reference)
                ).abs_max
                for row, mu in zip(rows, modes)
            ),
            Q(0),
        )
        assert row_bound < Q(14, 100)
    gamma = Q(1, 3200)
    assert (Q(1, 1023) / pi_interval()).hi < gamma
    assert (2 + 2 * root).hi / 5 < Q(13, 10)
    coarse_heat = Q(14, 100) * gamma * X
    forced = Q(13, 8) * gamma**2 * laws[1][0] * L
    assert p["heat_phase_upper"] < coarse_heat
    assert p["forced_phase_upper"] == forced
    assert coarse_heat + forced < Q(1, 10**6)
    assert p["phase_motion_upper"] == p["heat_phase_upper"] + forced
    assert p["phase_motion_cap"] == Q(1, 10**6)
    # This scalar bound divides the existing gamma endpoint exactly; the
    # circle kernel separately materializes its radius as an interval.
    radius_lower = (Q(1, 1023) / pi_interval()).lo / (4 * 2187)
    assert p["form_difference_radius_ratio"] == 43 + delta / radius_lower < 44
    relative = 22 * gamma**2 * L
    assert p["relative_phase_current_radius_ratio"] == relative
    assert p["current_remainder_radius_ratio"] == Q(38, 10**6) + relative < Q(46, 10**6)


def test_strict_exclusion_rebuilt_with_coarser_independent_error_budgets(
    reconstructed_contrast, primitive_joint_bounds
):
    p, mu, alpha_e, alpha_o, even, odd = reconstructed_contrast
    delta = primitive_joint_bounds[-1]
    exp_half = _exponential(mu / 2)
    cosh_half = (exp_half + 1 / exp_half) / 2
    common = exp_half * (1 - _decay_for_audit(mu, 1)) / mu * (3 + sqrt(5)) / 4
    weights = (
        abs(1 + alpha_e) + 2 * abs(alpha_e) + abs(alpha_e - 1),
        abs(1 - alpha_o) + 2 * abs(alpha_o) + abs(1 + alpha_o),
    )
    assert (common * weights[0]).hi < 5
    assert (common * weights[1]).hi < 11
    noise = tuple(
        sqrt(10) * cosh_half * (1 + 2 * abs(alpha)) for alpha in (alpha_e, alpha_o)
    )
    radius = (Q(1, 1023) / pi_interval()) / (4 * 2187)
    coarse_errors = tuple(
        multiplier.hi * delta + transport * Q(46, 10**6) * radius.hi
        for multiplier, transport in zip(noise, (5, 11))
    )
    assert 0 < p["even_error"] < coarse_errors[0]
    assert 0 < p["odd_error"] < coarse_errors[1]
    ratios = 2 * even[0] / even[1], 2 * odd[0] / odd[1]
    margin = 1 - max(value.abs_max for value in ratios)
    errors = tuple(
        2 * error / (radius.lo * min(abs(pair[1].lo), abs(pair[1].hi)))
        for error, pair in zip(coarse_errors, (even, odd))
    )
    assert margin > Q(71, 100)
    assert sum(error**2 for error in errors) < margin**2


def test_recorded_strict_circle_arithmetic_and_scoped_verdict(
    records, reconstructed_contrast
):
    p, _, _, _, even, odd = reconstructed_contrast
    circle = p["circle"]
    radius = (Q(1, 1023) / pi_interval()) / (4 * 2187)
    assert _interval(circle["ninth_radius"]) == radius
    assert tuple(map(_interval, circle["odd_coefficients"])) == odd
    assert tuple(map(_interval, circle["even_coefficients"])) == even
    assert circle["raw_error_bounds"] == (p["odd_error"], p["even_error"])
    ratios = tuple(2 * pair[0] / pair[1] for pair in (odd, even))
    assert tuple(map(_interval, circle["normalized_ratios"])) == ratios
    ratio_upper = max(value.abs_max for value in ratios)
    assert circle["ratio_upper"] == ratio_upper
    margin = 1 - ratio_upper
    errors = tuple(
        2 * error / (radius.lo * min(abs(pair[1].lo), abs(pair[1].hi)))
        for error, pair in zip((p["odd_error"], p["even_error"]), (odd, even))
    )
    assert circle["normalized_error_bounds"] == errors
    assert circle["margin_lower"] == margin
    assert circle["error_square_upper"] == sum(error**2 for error in errors) < margin**2
    assert circle["available"] is circle["exclusion_certified"] is True
    assert p["available"] is p["separation_certified"] is True
    result = records[SPECS[1][0]][1]
    assert result["verdict"] == "conditional_two_law_record_separation_certified"
    assert (
        result["report"]["scope"]
        == "conditional_two_law_record_separation_not_physical_law_selection"
    )
