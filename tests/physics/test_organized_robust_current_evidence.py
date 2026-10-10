"""Read-only reconstruction of the frozen common-current decision.

The source receipt is a provenance association, not execution authentication
or physical evidence. No bound, assessor, driver or trajectory is imported.
"""

from fractions import Fraction as Q
import hashlib
from math import prod
from pathlib import Path
import subprocess

import pytest

from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research.artifact_io import decode_exact_tree
from tnfr.utils.io import json_loads


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/organized_robust_current"
REVISION = "98ccd8686592eea15ad22151802d731d683d70e2"
BASE = "689736a7"
DECLARATION = "theory/nodal/ORGANIZED_ROBUST_CURRENT.md"
EVALUATOR = "benchmarks/organized_robust_current.py"
PURE = "src/tnfr/physics/_organized_robust_current.py"


@pytest.fixture(scope="module")
def records():
    return tuple(
        decode_exact_tree(json_loads((EVIDENCE / name).read_text(encoding="utf-8")))
        for name in ("attempt.json", "result.json")
    )


def _git(*arguments):
    return subprocess.check_output(
        ("git", *arguments), cwd=ROOT, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="module")
def declaration():
    try:
        return _git("show", REVISION + ":" + DECLARATION).decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("retained source revision unavailable in this checkout")


@pytest.fixture(scope="module")
def reconstructed(records):
    p = records[1]["report"]
    C, G, A, D = (
        I(**p[name + "_ratio"])
        for name in ("capacitance", "conductance", "josephson", "clock")
    )
    coefficients = {
        "lambda": I(G.lo / (C.hi * D.hi), G.hi / (C.lo * D.lo)),
        "rho": I(1 / (C.hi * D.hi), 1 / (C.lo * D.lo)),
        "eta": I(1 / (A.hi * D.hi), 1 / (A.lo * D.lo)),
        "beta": I(
            A.lo * G.lo**2 / (C.hi**2 * D.hi),
            A.hi * G.hi**2 / (C.lo**2 * D.lo),
        ),
        "eta_rho": I(
            1 / (A.hi * C.hi * D.hi**2),
            1 / (A.lo * C.lo * D.lo**2),
        ),
    }
    pi = pi_interval()
    gamma = I(1 / (1023 * pi.hi), 1 / (1023 * pi.lo))
    lam, eta_rho = coefficients["lambda"], coefficients["eta_rho"]
    damping = max(abs(1 - lam.lo), abs(1 - lam.hi))
    phase_mismatch = max(abs(1 - eta_rho.lo), abs(1 - eta_rho.hi))
    # Integral of the nonnegative cubic derivative is one on every ramp.
    increments = (Q(1, 10), Q(-1, 5), Q(1, 10)) * 5
    offset_integral = 4 * Q(4, 5) * sum(map(abs, increments)) / gamma.lo
    forcing = coefficients["rho"].hi * damping * offset_integral / 4
    forcing += p["total_tracking"]
    return coefficients, gamma, damping, phase_mismatch, offset_integral, forcing


def test_single_attempt_receipt_and_preserved_frozen_producers(records, declaration):
    attempt, result = records
    assert attempt["source_revision"] == result["source_revision"] == REVISION
    assert attempt["schema"] == "organized_robust_current_attempt_v1"
    assert result["schema"] == "organized_robust_current_result_v1"
    assert attempt["maximum_assessments"] == result["assessment_count"] == 1
    assert attempt["evaluation_status_at_start"] == "not_evaluated"
    assert attempt["parameters"] == {} and attempt["seed"] is None
    assert attempt["interval_bits"] == 128
    assert attempt["declaration"] == DECLARATION and attempt["evaluator"] == EVALUATOR
    assert attempt["source_scope"] == (
        "src/tnfr",
        EVALUATOR,
        DECLARATION,
        "pyproject.toml",
    )
    raw = (EVIDENCE / "attempt.json").read_bytes()
    assert result["attempt"] == dict(
        path="attempt.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    assert {path.name for path in EVIDENCE.glob("*.json")} == {
        "attempt.json",
        "result.json",
    }
    listing = _git(
        "ls-tree", "-r", "-z", "--full-tree", REVISION, "--", *attempt["source_scope"]
    )
    inventory = {}
    for entry in filter(None, listing.split(b"\0")):
        metadata, path = entry.split(b"\t", 1)
        mode, kind, blob = metadata.decode("ascii").split()
        assert kind == "blob"
        inventory[path.decode("utf-8")] = dict(mode=mode, blob=blob)
    assert inventory == attempt["source_inventory"]
    assert hashlib.sha256(listing).hexdigest() == attempt["source_inventory_sha256"]
    assert (
        _git("rev-parse", REVISION + ":src").decode().strip() == attempt["source_tree"]
    )
    assert {DECLARATION, EVALUATOR, PURE}.issubset(inventory)
    for record in records:
        assert set(record["imported_sources"].values()) <= set(inventory)
    assert result["imported_sources"]["tnfr.physics._organized_robust_current"] == PURE
    unchanged = (
        "src/tnfr/physics/_organized_finite_observation.py",
        "src/tnfr/physics/_organized_finite_observation_source.py",
        "src/tnfr/physics/_organized_electrical_protocol.py",
        "src/tnfr/physics/_organized_electrical_readout.py",
        "benchmarks/organized_finite_observation.py",
        "docs/assets/organized_finite_observation",
        "docs/assets/organized_electrical_actuation",
    )
    assert not _git("diff", "--name-only", BASE, REVISION, "--", *unchanged)
    assert "before one analytical assessment" in declaration


def test_same_preparation_support_word_and_nonzero_error_primitives(
    records, declaration
):
    p = records[1]["report"]
    assert p["reading_shape"] == (5, 3, 2, 5) and prod(p["reading_shape"]) == 150
    assert p["probe_amplitude"] == Q(1, 10)
    assert p["plateau_duration"] == Q(1, 10**5)
    assert p["total_elapsed_scaled_time"] == 15 * (1 + p["plateau_duration"])
    for name in ("capacitance", "conductance", "josephson", "clock"):
        assert I(**p[name + "_ratio"]) == I(1 - Q(1, 10**8), 1 + Q(1, 10**8))
    assert p["total_tracking"] == Q(1, 10**4)
    assert p["plateau_tracking"] == Q(1, 10**15)
    assert p["raw_readout_error"] == Q(1, 10**14)
    assert p["aperture"] == Q(1, 10**13) < p["plateau_duration"]
    assert tuple(model["multiplier"] for model in p["model_bounds"]) == (1, 2)
    # These are declared premises, not reconstructed physical preparation.
    for premise in (
        "Full C9-by-C5 laws (3) with mu equal to 1 or 2",
        "45 forms and phases, 90 unit edges, held unit capacities",
        "zero global form mean",
        "abs(x_i)<=X=.01",
        "abs(z_cj)<=zeta=1e-6",
        "Aphase=7/5",
        "winding in `{-1,0,1}`",
        "Same nominal cubic word for both laws",
        "marked increments `.1,-.2,.1`",
        "positive normalized average within its own inward endpoint aperture",
        "no amplitude-dependent redefinition of the observed statistic",
        "does not establish acquisition under mu=2 or perturbed laws",
    ):
        assert premise in declaration
    assert (
        "same_initial_prior_fixed_support_clock_word_and_150_form_readings"
        in p["scope"]
    )


def test_calibration_and_whole_history_bounds_are_rebuilt_for_both_laws(
    records, reconstructed
):
    p = records[1]["report"]
    coeff, gamma, damping, mismatch, offset, forcing = reconstructed
    for name, interval in coeff.items():
        assert I(**p[name + "_bounds"]) == interval
    assert I(**p["gamma_bounds"]) == gamma
    assert p["damping_mismatch_upper"] == damping
    assert p["phase_command_mismatch_upper"] == mismatch
    assert p["offset_laplacian_integral_upper"] == offset
    lam, eta, beta = coeff["lambda"], coeff["eta"], coeff["beta"]
    for model in p["model_bounds"]:
        mu = model["multiplier"]
        sine_drift = mu * gamma.hi * beta.hi
        accumulated = sine_drift * p["total_elapsed_scaled_time"] + forcing
        form = Q(1, 100) + accumulated
        phase = gamma.hi * eta.hi / lam.lo * (Q(1, 100) + form + accumulated)
        phase += mismatch * Q(4, 5) * p["probe_amplitude"]
        second = 4 * lam.hi * (lam.hi * form + sine_drift / 2)
        second += 4 * sine_drift * gamma.hi * eta.hi * form
        sample = p["raw_readout_error"] + p["plateau_tracking"]
        sample += (lam.hi * form + sine_drift / 2) * p["aperture"]
        assert model["forcing_residual_integral_upper"] == forcing
        assert model["observation_form_upper"] == form
        assert model["phase_residual_upper"] == phase
        assert model["plateau_second_derivative_upper"] == second
        assert model["sample_error_upper"] == sample
    first, second = p["model_bounds"]
    assert second["observation_form_upper"] > first["observation_form_upper"]
    assert second["phase_residual_upper"] > first["phase_residual_upper"]


def test_corrected_reading_errors_and_unnormalized_radial_allowances(
    records, reconstructed
):
    p = records[1]["report"]
    coeff, gamma, damping, _, _, _ = reconstructed
    h, epsilon = p["plateau_duration"], p["probe_amplitude"]
    endpoint_coefficients = (-1 + h / 4, 1 + h / 4) + (-h / 8,) * 4
    sine, cosine_defect = sin(I(epsilon)), 1 - cos(I(epsilon))
    for model in p["model_bounds"]:
        transport = model["multiplier"] * gamma.hi * coeff["beta"].hi
        transport *= h * model["phase_residual_upper"]
        diffusion = damping * h * model["observation_form_upper"]
        trapezoid = model["plateau_second_derivative_upper"] * h**3 / 12
        tracking = p["plateau_tracking"] * (1 + h / 2 + coeff["lambda"].hi * h**2 / 4)
        reading = sum(map(abs, endpoint_coefficients)) * model["sample_error_upper"]
        corrected = transport + diffusion + trapezoid + tracking + reading
        assert model["corrected_increment_error_upper"] == corrected
        # Three-plateau odd/even coefficients have absolute sums two/four.
        odd, even = 2 * corrected, 4 * corrected
        e_a = 2 * odd / (gamma.lo * h * sine.lo)
        e_b = 2 * even / (gamma.lo * h * cosine_defect.lo)
        assert model["cosine_quadrature_error_upper"] == e_a
        assert model["sine_quadrature_error_upper"] == e_b
        radial = sqrt(I(e_a**2 + e_b**2)).hi
        assert model["radial_error_upper"] == radial
        assert radial**2 >= e_a**2 + e_b**2


def test_both_identity_certificates_and_strict_current_separation(records):
    result, p = records[1], records[1]["report"]
    pi, zeta, epsilon = pi_interval(), Q(1, 10**6), p["probe_amplitude"]
    beta = I(**p["beta_bounds"])
    carrier_modulus = cos(I(2 * zeta)).lo
    for model in p["model_bounds"]:
        phase = model["phase_residual_upper"]
        margins = dict(
            internal_positive_gap=2 * pi.lo / 9 - 2 * zeta - 2 * phase,
            internal_acute_gap=pi.lo / 2 - 2 * pi.hi / 9 - 2 * zeta - 2 * phase,
            outer_acute_gap=pi.lo / 2 - Q(7, 5) - epsilon - 2 * zeta - 2 * phase,
            circular_collar=pi.lo / 2 - 2 * zeta - 4 * phase,
        )
        assert len(model["identity_margins"]) == 4
        assert dict(model["identity_margins"]) == margins
        assert all(value > 0 for value in margins.values())
        assert model["finite_identity_certified"] is True
        mu, error = model["multiplier"], model["radial_error_upper"]
        assert model["amplitude_lower"] == max(
            Q(0), mu * beta.lo * carrier_modulus - error
        )
        assert model["amplitude_upper"] == mu * beta.hi + error
    first, second = p["model_bounds"]
    assert (
        p["separation_margin"]
        == second["amplitude_lower"] - first["amplitude_upper"]
        > 0
    )
    assert p["admitted"] is True and p["unresolved_conditions"] == ()
    assert p["status"] == "conditionally_separated_current_envelopes"
    assert result["verdict"] == "conditional_current_discrimination_admitted"
    assert records[0]["arithmetic_method"] == p["interval_method"]
