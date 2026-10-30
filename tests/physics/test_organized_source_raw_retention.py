"""Source-free maintenance on the complete zero-total-form leaf.

The fixed nonsymmetric control has x0=-x9=1/64 and one phase-offset increment
1/100 at node0 above the rendered winding-one carrier with origin1/40.
Countercontrols alter a declared integer lift, the form total or the energy.
Two retained IL endpoints are read as primitive observations, never replayed.
All conclusions concern the unchanged unit-capacity continuous law; there are
no subsequent events, source acquisition, trajectories or symmetry repair.
"""

from copy import deepcopy
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr._exact_time import exact_or_represented_real
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import network_stage, word_execution
from tnfr.physics import _organized_source_capture as owner
from tnfr.physics import _organized_source_port as port
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
)
from tnfr.utils.io import json_loads

BASE = (
    Path(__file__).resolve().parents[2]
    / "docs/assets/organized_source_coherence_handoff"
)
CEILING, GAP = Q(1, 100), Q(4, 15)
FORM = tuple(Q(1, 64) if n == 0 else -Q(1, 64) if n == 9 else Q(0) for n in range(45))
TURNS = (0,) * 45


@pytest.fixture(scope="module", autouse=True)
def no_events_sources_or_flow():
    def forbidden(*args, **kwargs):
        pytest.fail("Raw retention is a static primitive-state certificate")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (owner, "_assess_organized_source_capture"),
            (owner, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return port._derive_organized_source_port()


@pytest.fixture(scope="module")
def phases(geometry):
    return tuple(
        Q(
            float(
                geometry.alpha_bounds.midpoint * (n % 9 - 4)
                + Q(1, 40)
                + (Q(1, 100) if n == 0 else 0)
            )
        )
        for n in range(45)
    )


@pytest.fixture(scope="module")
def report(phases):
    return owner._assess_organized_source_raw_retention(
        form=FORM, phases=phases, phase_turns=TURNS
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def test_actual_full_fine_zero_sum_space_has_exact_positive_gap(geometry):
    """44-dimensional congruence certificate, independent of cached gap flags."""
    laplacian = sp.zeros(45)
    for a, b, weight in geometry.weighted_edges:
        assert weight == 1
        laplacian[a, a] += 1
        laplacian[b, b] += 1
        laplacian[a, b] -= 1
        laplacian[b, a] -= 1
    assert laplacian * sp.ones(45, 1) == sp.zeros(45, 1)
    # Columns e_i-e_44 span the WHOLE zero-sum fine space. Its Gram matrix
    # has diagonal2 and off-diagonal1, so the norm term cannot be omitted.
    restricted = sp.Matrix(
        44,
        44,
        lambda i, j: laplacian[i, j]
        - laplacian[i, 44]
        - laplacian[44, j]
        + laplacian[44, 44]
        - sp.Rational(4, 15) * (1 + int(i == j)),
    )
    triangular, diagonal = restricted.LDLdecomposition(hermitian=False)
    assert all(value > 0 for value in diagonal.diagonal())
    assert triangular * diagonal * triangular.T == restricted
    assert 2 * CEILING / GAP == Q(3, 40) < Q(3, 10) ** 2 < 1


def test_nonsymmetric_state_uses_global_leaf_and_new_phase_origin(report, phases, mp):
    assert report.observation.form == FORM and report.observation.phases == phases
    assert sum(FORM) == 0 and FORM[0] != -FORM[1]
    assert report.phase_turns == TURNS and report.phase_turn_sum == 0
    assert report.raw_phase_sum == sum(phases)
    _contains(mp, report.lifted_phase_mean_bounds, _mp(mp, sum(phases) / 45))
    assert report.lifted_phase_mean_bounds.lo > Q(1, 40)
    assert report.geometry_certified and report.in_acute_cell
    assert report.below_energy_ceiling and report.admitted
    assert report.energy_ceiling == CEILING and report.spectral_gap_lower_bound == GAP
    assert report.form_squared_upper_bound == Q(3, 40)
    assert report.form_absolute_upper_bound == Q(3, 10)
    assert report.retained_windings == (1,) * 5
    assert not report.unresolved_conditions
    assert report.excess_bounds.lo > 0 and report.excess_bounds.hi < CEILING
    energy_form = sum(
        w * (FORM[a] - FORM[b]) ** 2 / 2
        for a, b, w in report.observation.geometry.weighted_edges
    )
    assert GAP * sum(v * v for v in FORM) <= 2 * energy_form
    assert energy_form < report.excess_bounds.lo
    gradient = report.observation.form_gradient
    assert sum(gradient) == 0
    # Only the full global mean is retained: childwise phase means can move.
    child_sums = tuple(sum(gradient[9 * c : 9 * c + 9]) for c in range(5))
    assert child_sums == (Q(1, 16), -Q(3, 64), -Q(1, 64), Q(1, 64), -Q(1, 64))
    assert all(value != 0 for value in child_sums)
    mean = _mp(mp, sum(phases) / 45)
    for n, bounds in enumerate(report.asymptotic_phase_lift_bounds):
        _contains(mp, bounds, 2 * mp.pi / 9 * (n % 9 - 4) + mean)


def test_independent_energy_and_complete_acute_boundary(report, phases, mp):
    geometry = report.observation.geometry
    energy = mp.mpf(0)
    for a, b, w in geometry.weighted_edges:
        angle = _mp(mp, phases[b] - phases[a])
        energy += _mp(mp, w) * (_mp(mp, FORM[a] - FORM[b]) ** 2 / 2 + 1 - mp.cos(angle))
    baseline = 45 * (1 - mp.cos(2 * mp.pi / 9))
    _contains(mp, report.excess_bounds, energy - baseline)
    positive_face = 1 + 8 * (1 - mp.cos(3 * mp.pi / 16))
    negative_face = 1 + 8 * (1 - mp.cos(5 * mp.pi / 16))
    cycle_minimum = 9 * (1 - mp.cos(2 * mp.pi / 9))
    _contains(
        mp, report.barrier.cycle_boundary_excess_bounds, positive_face - cycle_minimum
    )
    assert (
        negative_face > positive_face and positive_face - cycle_minimum > mp.mpf(6) / 25
    )
    assert report.domain_boundary_margin > Q(6, 25) - CEILING
    assert report.barrier.signed_internal_boundary_faces == 90
    assert report.barrier.signed_matching_boundary_faces == 90
    for bounds, (a, b, _) in zip(report.edge_angle_bounds, geometry.weighted_edges):
        angle = _mp(mp, phases[b] - phases[a])
        if a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0:
            angle += 2 * mp.pi
        _contains(mp, bounds, angle)
        assert bounds.abs_max < pi_interval().lo / 2


def test_common_turns_change_only_lift_origin_not_physical_rows(report, phases, mp):
    shifted = owner._assess_organized_source_raw_retention(
        form=FORM, phases=phases, phase_turns=(3,) * 45
    )
    assert shifted.admitted and shifted.phase_turn_sum == 135
    assert shifted.observation == report.observation
    assert shifted.edge_angle_bounds == report.edge_angle_bounds
    assert shifted.excess_bounds == report.excess_bounds
    expected_mean = _mp(mp, sum(phases) / 45) + 6 * mp.pi
    _contains(mp, shifted.lifted_phase_mean_bounds, expected_mean)
    for n, bounds in enumerate(shifted.asymptotic_phase_lift_bounds):
        _contains(mp, bounds, 2 * mp.pi / 9 * (n % 9 - 4) + expected_mean)


def test_local_wrong_turn_keeps_physical_energy_but_loses_chart(report, phases):
    wrong = owner._assess_organized_source_raw_retention(
        form=FORM, phases=phases, phase_turns=(1,) + (0,) * 44
    )
    assert wrong.observation == report.observation
    assert wrong.below_energy_ceiling and wrong.excess_bounds == report.excess_bounds
    assert not wrong.in_acute_cell and not wrong.admitted
    assert wrong.form_squared_upper_bound is None
    assert wrong.form_absolute_upper_bound is None
    assert wrong.retained_windings is None
    assert wrong.asymptotic_phase_lift_bounds is None


def test_low_raw_energy_does_not_establish_winding_one():
    wrong = owner._assess_organized_source_raw_retention(
        form=(Q(0),) * 45, phases=(Q(0),) * 45, phase_turns=TURNS
    )
    assert wrong.excess_bounds.hi < 0 and wrong.below_energy_ceiling
    assert not wrong.in_acute_cell and not wrong.admitted
    assert (
        wrong.retained_windings is None and wrong.asymptotic_phase_lift_bounds is None
    )


def test_high_energy_has_no_maintenance_flags_even_inside_cell(phases):
    large = tuple(16 * v for v in FORM)
    report = owner._assess_organized_source_raw_retention(
        form=large, phases=phases, phase_turns=TURNS
    )
    assert report.in_acute_cell and not report.below_energy_ceiling
    assert not report.admitted and report.excess_bounds.lo > CEILING
    assert report.form_absolute_upper_bound is None and report.retained_windings is None
    assert report.asymptotic_phase_lift_bounds is None


def test_constant_form_storage_cannot_replace_zero_total_admission(phases, monkeypatch):
    zero = port._observe_organized_source_phase_state(form=(Q(0),) * 45, phases=phases)
    constant = port._observe_organized_source_phase_state(
        form=(Q(1, 2),) * 45, phases=phases
    )
    assert zero.joint_storage_bounds == constant.joint_storage_bounds
    assert zero.form_gradient == constant.form_gradient == (0,) * 45

    def forbidden():
        pytest.fail("nonzero-total leaf reached geometric assessment")

    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(owner, "_observe_organized_source_phase_state", forbidden)
    with pytest.raises(ValueError):
        owner._assess_organized_source_raw_retention(
            form=constant.form, phases=phases, phase_turns=TURNS
        )


@pytest.mark.parametrize("field", ("form", "phases"))
@pytest.mark.parametrize("invalid", (True, float("nan"), 1j, "wrong_length"))
def test_invalid_vectors_reject_before_geometry(monkeypatch, field, invalid):
    def forbidden():
        pytest.fail("invalid primitives reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(owner, "_observe_organized_source_phase_state", forbidden)
    values = dict(form=(Q(0),) * 45, phases=(Q(0),) * 45, phase_turns=TURNS)
    values[field] = (
        (Q(0),) * 44 if invalid == "wrong_length" else (invalid,) + (Q(0),) * 44
    )
    with pytest.raises((ValueError, TypeError)):
        owner._assess_organized_source_raw_retention(**values)


@pytest.mark.parametrize("invalid", (True, 0.0, Q(0), "wrong_length"))
def test_turns_require_exact_ordinary_integer_branch_data(monkeypatch, invalid):
    def forbidden():
        pytest.fail("invalid branch data reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(owner, "_observe_organized_source_phase_state", forbidden)
    turns = (0,) * 44 if invalid == "wrong_length" else (invalid,) + (0,) * 44
    with pytest.raises((ValueError, TypeError)):
        owner._assess_organized_source_raw_retention(
            form=(Q(0),) * 45, phases=(Q(0),) * 45, phase_turns=turns
        )


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(BASE / name, max_bytes=2**27))
    )


@pytest.fixture(scope="module")
def evidence():
    paths = tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    yield _read("protocol.json"), _read("response.json")
    assert paths == tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    assert receipts == tuple(file_receipt(path) for path in paths)


def _admit_saved_snapshot(snapshot, protocol):
    """Validate runtime/model association before invoking the fixed-law owner."""
    rows = snapshot["nodes"]
    if len(rows) != 45 or any(type(row["node"]) is not int for row in rows):
        raise ValueError("invalid node identity")
    if tuple(row["node"] for row in rows) != tuple(range(45)):
        raise ValueError("invalid node order")
    edges = []
    for a, b, weight in snapshot["weighted_edges"]:
        if type(a) is not int or type(b) is not int:
            raise ValueError("invalid edge identity")
        edges.append((a, b, exact_or_represented_real(weight, "conductance")))
    capacities = tuple(
        exact_or_represented_real(row["capacity"], "capacity") for row in rows
    )
    clock = exact_or_represented_real(snapshot["time"], "tau timestamp")
    form = tuple(exact_or_represented_real(row["form"], "form") for row in rows)
    phases = tuple(exact_or_represented_real(row["phase"], "phase") for row in rows)
    if (
        capacities != (Q(1),) * 45
        or clock != protocol["configuration"]["structural_time_tau"]
    ):
        raise ValueError("capacity or clock differs from the selected model")
    geometry = port._derive_organized_source_port()
    if tuple(edges) != geometry.weighted_edges:
        raise ValueError("fixed support/order differs")
    return form, phases


@pytest.mark.parametrize("index", (1, 2))
def test_saved_represented_il_endpoints_receive_independent_raw_maintenance(
    evidence, index, mp
):
    protocol, response = evidence
    stage = response["stages"][index]
    assert stage["glyph"] == "IL"
    form, phases = _admit_saved_snapshot(stage["after"], protocol)
    turns = protocol["phase_representation"]["output_turns"]
    report = owner._assess_organized_source_raw_retention(
        form=form, phases=phases, phase_turns=turns
    )
    assert report.admitted and report.geometry_certified
    assert report.retained_windings == (1,) * 5
    assert report.form_squared_upper_bound == Q(
        3, 40
    ) and report.form_absolute_upper_bound == Q(3, 10)
    assert report.observation.form == form and report.observation.phases == phases
    assert report.observation.geometry.held_capacities == (1,) * 45
    assert report.observation.geometry.clock_factor == Q(1, 4)
    assert report.phase_turn_sum == -2 and report.raw_phase_sum == sum(phases)
    mean = _mp(mp, sum(phases) / 45) + 2 * mp.pi * sum(turns) / 45
    _contains(mp, report.lifted_phase_mean_bounds, mean)
    assert report.lifted_phase_mean_bounds.hi < Q(
        1, 40
    ) or report.lifted_phase_mean_bounds.lo > Q(1, 40)
    for n, bounds in enumerate(report.asymptotic_phase_lift_bounds):
        _contains(mp, bounds, mean + 2 * mp.pi / 9 * (n % 9 - 4))
    assert report.in_acute_cell and report.excess_bounds.hi < CEILING
    assert not stage["analysis"]["exact_reflection_certified"]
    assert not stage["analysis"]["future_retention_certified"]
    # New certificate, unchanged rows: frozen old flags are never rewritten.
    frozen = stage["callback"][0]["observation"]
    for clock in ("tau", "slow", "original"):
        for channel in ("form", "phase"):
            field = f"{channel}_{clock}_rate_bounds"
            for actual, old in zip(getattr(report.observation, field), frozen[field]):
                assert actual == I(old["lo"], old["hi"])


@pytest.mark.parametrize("field", ("node", "weight", "capacity", "time"))
def test_saved_runtime_association_does_not_equate_booleans_with_physical_one(
    evidence, field
):
    protocol, response = evidence
    snapshot = deepcopy(response["stages"][1]["after"])
    if field == "node":
        snapshot["nodes"][1]["node"] = True
    elif field == "weight":
        edges = list(snapshot["weighted_edges"])
        a, b, _ = edges[0]
        edges[0] = (a, b, True)
        snapshot["weighted_edges"] = tuple(edges)
    elif field == "capacity":
        snapshot["nodes"][0]["capacity"] = True
    else:
        snapshot["time"] = True
    with pytest.raises((ValueError, TypeError)):
        _admit_saved_snapshot(snapshot, protocol)
