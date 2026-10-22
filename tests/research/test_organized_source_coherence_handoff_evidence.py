"""Read-only evidence and raw-phase observation controls for the frozen handoff.

This owner never imports the benchmark or executes an event. The retained
binary64 coordinates are distinct from the exact source lift. Immediate acute
geometry and energy are checked without promoting tolerance to reflection or
claiming a reflected future-retention theorem for the represented endpoint.
"""

from dataclasses import asdict
from fractions import Fraction as Q
from pathlib import Path

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, pi_interval
from tnfr.operators import word_execution
from tnfr.physics import _organized_source_port as owner
from tnfr.research.artifact_io import (
    decode_exact_tree,
    file_receipt,
    read_bytes_bounded,
    verify_archive_members,
)
from tnfr.research.phase_reset_certificate import bound_phase_reset
from tnfr.utils.io import json_loads

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/assets/organized_source_coherence_handoff"
TARGETS = (0, 1)


@pytest.fixture(scope="module", autouse=True)
def no_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("This owner only reads retained responses and static fields")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _read(name):
    return decode_exact_tree(
        json_loads(read_bytes_bounded(BASE / name, max_bytes=2**27))
    )


def _interval(record):
    return I(record["lo"], record["hi"])


def _field(snapshot, name):
    return tuple(row[name] for row in snapshot["nodes"])


def _observe(snapshot):
    return owner._observe_organized_source_phase_state(
        form=_field(snapshot, "form"), phases=_field(snapshot, "phase")
    )


def _mp_oracle(form, phases, edges):
    """Independent high-precision incidence sums, never a numerical trajectory."""
    mp = mpmath.mp.clone()
    mp.dps = 80

    def value(q):
        return mp.mpf(q.numerator) / q.denominator

    x, theta = tuple(map(value, form)), tuple(map(value, phases))
    q, current = [mp.mpf(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for a, b, weight in edges:
        w = value(weight)
        difference, gap = x[a] - x[b], theta[b] - theta[a]
        q[a] += w * difference
        q[b] -= w * difference
        flux = w * mp.sin(gap)
        current[a] += flux
        current[b] -= flux
        storage += w * (difference**2 / 2 + 1 - mp.cos(gap))
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 4 for a, b in zip(q, current))
    dt = tuple(gamma * a / 4 for a in q)
    return mp, value, tuple(q), tuple(current), storage, gamma, dx, dt


def _contains(value, interval, exact):
    assert value(interval.lo) <= exact <= value(interval.hi)


def _check_oracle(report):
    mp, value, q, current, storage, gamma, dx, dt = _mp_oracle(
        report.form, report.phases, report.geometry.weighted_edges
    )
    assert tuple(map(value, report.form_gradient)) == q
    for bounds, expected in zip(report.phase_current_bounds, current):
        _contains(value, bounds, expected)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for channel, expected in (("form", dx), ("phase", dt)):
            for bounds, rate in zip(
                getattr(report, f"{channel}_{clock}_rate_bounds"), expected
            ):
                _contains(value, bounds, factor * rate)
    _contains(value, report.joint_storage_bounds, storage)
    loss = sum(v**2 for v in q) / 4
    assert abs(value(report.joint_dissipation_tau) - loss) < mp.mpf("1e-75")
    assert report.joint_storage_tau_rate == -report.joint_dissipation_tau
    assert report.balance_residual_bounds.lo <= 0 <= report.balance_residual_bounds.hi


@pytest.fixture(scope="module")
def evidence():
    paths = tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    records = {
        name: _read(name)
        for name in ("protocol.json", "freeze.json", "attempt.json", "response.json")
    }
    yield records
    assert paths == tuple(sorted(path for path in BASE.iterdir() if path.is_file()))
    assert receipts == tuple(file_receipt(path) for path in paths)


def test_retained_response_counts_preparation_and_freeze_receipts(evidence):
    protocol, freeze = evidence["protocol.json"], evidence["freeze.json"]
    attempt, response = evidence["attempt.json"], evidence["response.json"]
    assert attempt["status"] == "not_evaluated"
    assert protocol["status_at_freeze"] == "not_evaluated"
    for name in ("protocol", "freeze"):
        assert attempt[name] == file_receipt(
            BASE / f"{name}.json", label=f"{name}.json"
        )
    archive = freeze["archive"]
    assert archive == file_receipt(BASE / archive["path"], label=archive["path"])
    verify_archive_members(BASE / archive["path"], freeze["members"])
    assert (
        attempt["driver"]["path"] == "benchmarks/organized_source_coherence_handoff.py"
    )
    assert attempt["driver"]["sha256"] == freeze["members"][attempt["driver"]["path"]]
    protocol_path = "docs/assets/organized_source_coherence_handoff/protocol.json"
    assert attempt["protocol"]["sha256"] == freeze["members"][protocol_path]
    assert response["status"] == "evaluated"
    assert tuple(row["glyph"] for row in response["stages"]) == ("EN", "IL", "IL")
    assert tuple(row["preparation"] for row in response["stages"]) == (
        "reception_handoff",
        "reception_handoff",
        "moving_control",
    )
    assert attempt["execution_counts"] == protocol["execution_counts"]
    for row in response["stages"]:
        assert row["stage"]["schedule"] == "two_phase_jacobi"
        assert row["stage"]["nodes_processed"] == 2
        assert len(row["callback"]) == 1


@pytest.mark.parametrize("name", ("reception_handoff", "moving_control"))
def test_exact_source_prediction_is_independently_rebuilt(evidence, name):
    protocol = evidence["protocol.json"]
    prediction = evidence["attempt.json"]["predictions"][name]
    source = protocol["preparations"][name]
    edges = protocol["support"]["weighted_edges"]
    mp = mpmath.mp.clone()
    mp.dps = 80

    def value(q):
        return mp.mpf(q.numerator) / q.denominator

    alpha = 2 * mp.pi / 9
    phases = tuple(
        alpha * (n % 9 - 4) + value(offset)
        for n, offset in enumerate(source["phase_offsets"])
    )
    coefficient = value(protocol["configuration"]["phase_locking_coefficient"])
    displacements = [mp.mpf(0)] * 45
    for index, n in enumerate(TARGETS):
        neighbors = tuple(b if a == n else a for a, b, _ in edges if n in (a, b))
        real = sum(mp.cos(phases[j] - phases[n]) for j in neighbors)
        imaginary = sum(mp.sin(phases[j] - phases[n]) for j in neighbors)
        assert real > 4
        displacements[n] = coefficient * mp.atan2(imaginary, real)
        _contains(value, _interval(prediction["resultants"][index][0]), real)
        _contains(value, _interval(prediction["resultants"][index][1]), imaginary)
    endpoints = tuple(a + b for a, b in zip(phases, displacements))
    for bounds, exact in zip(prediction["after"], endpoints):
        _contains(value, _interval(bounds), exact)
    work = sum(
        value(w) * (mp.cos(phases[b] - phases[a]) - mp.cos(endpoints[b] - endpoints[a]))
        for a, b, w in edges
    )
    _contains(value, _interval(prediction["work"]), work)
    response = next(
        row
        for row in evidence["response.json"]["stages"]
        if row["glyph"] == "IL" and row["preparation"] == name
    )
    actual = _field(response["after"], "phase")
    turns = protocol["phase_representation"]["output_turns"]
    for raw, turn, ideal in zip(actual, turns, endpoints):
        assert abs(value(raw) + 2 * mp.pi * turn - ideal) <= value(
            protocol["budgets"]["exact_source_endpoint"]
        )
    if name == "moving_control":
        assert work < 0 and displacements[0] < 0 < displacements[1]
    else:
        assert abs(work) < mp.mpf("1e-75")


@pytest.mark.parametrize("name", ("reception_handoff", "moving_control"))
def test_initial_rendering_preserves_declared_source_association(evidence, name):
    protocol, response = evidence["protocol.json"], evidence["response.json"]
    source = protocol["preparations"][name]
    snapshot = response["initial"][name]["after"]
    report = _observe(snapshot)
    assert report.form == source["form"]
    assert report.geometry.weighted_edges == protocol["support"]["weighted_edges"]
    for node, actual, offset in zip(range(45), report.phases, source["phase_offsets"]):
        ideal = report.geometry.alpha_bounds * (node % 9 - 4) + offset
        assert (I(actual) - ideal).abs_max <= protocol["budgets"]["rendering"]
        assert Q(float.fromhex(snapshot["nodes"][node]["phase_hex"])) == actual
    assert _field(snapshot, "capacity") == (1,) * 45
    assert _field(snapshot, "kind") == ("organized-source",) * 45
    assert not any(_field(snapshot, "temporal_history_present"))


def test_actual_reception_output_is_the_main_coherence_input(evidence):
    response, attempt = evidence["response.json"], evidence["attempt.json"]
    en, il, _ = response["stages"]
    assert en["after"] == il["before"]
    expected = attempt["predictions"]["reception"]
    actual = _field(en["after"], "form")
    assert all(
        abs(a - b) <= expected["form_error_budget"]
        for a, b in zip(actual, expected["form_after"])
    )
    assert actual[0] == -actual[1] and actual[0] > 0
    assert _field(en["before"], "phase") == _field(en["after"], "phase")
    edges = en["before"]["weighted_edges"]
    old = _field(en["before"], "form")
    work = sum(
        w * ((actual[a] - actual[b]) ** 2 - (old[a] - old[b]) ** 2) / 2
        for a, b, w in edges
    )
    assert work < 0
    assert abs(work - expected["work"]) < Q(1, 10**17)
    for observation, n in zip(en["stage"]["reception_observations"], TARGETS):
        assert observation["node"] == n
        assert observation["read_boundary"] == "pre_en_snapshot"
        assert observation["neighbors"] == en["before"]["neighbors"][n]


@pytest.mark.parametrize("index", (1, 2))
def test_coherence_held_fields_and_live_grammar_are_actual(evidence, index):
    row = evidence["response.json"]["stages"][index]
    before, after = row["before"], row["after"]
    assert before["weighted_edges"] == after["weighted_edges"]
    assert before["neighbors"] == after["neighbors"]
    assert before["time"] == after["time"] == 7
    for field in (
        "form",
        "capacity",
        "kind",
        "stored_dEPI",
        "temporal_history_present",
        "reception_sources",
        "prior_coherence",
    ):
        assert _field(before, field) == _field(after, field)
    assert all(_field(after, "prior_coherence"))
    for n, (old, new) in enumerate(zip(before["nodes"], after["nodes"])):
        if n in TARGETS:
            assert old["history"] == ("AL", "IL", "EN") and old["ordinal"] == 3
            assert new["history"] == ("AL", "IL", "EN", "IL") and new["ordinal"] == 4
            assert new["source_glyph"] == "IL"
            assert new["debt"] == old["debt"] == 0
        else:
            assert new["history"] == old["history"] == ("AL", "IL")
            assert new["ordinal"] == old["ordinal"] == 2
            assert new["phase"] == old["phase"]


@pytest.mark.parametrize("index", (1, 2))
def test_reset_fixed_lift_and_exact_source_error_are_separate(evidence, index):
    protocol, attempt = evidence["protocol.json"], evidence["attempt.json"]
    row = evidence["response.json"]["stages"][index]
    old, new = _field(row["before"], "phase"), _field(row["after"], "phase")
    certificates = bound_phase_reset(
        old,
        new,
        row["before"]["neighbors"],
        TARGETS,
        coefficient=protocol["configuration"]["phase_locking_coefficient"],
        budget=protocol["budgets"]["runtime_reset"],
    )
    ideal = attempt["predictions"][row["preparation"]]
    lifted = tuple(
        I(v) + 2 * pi_interval() * turn
        for v, turn in zip(new, protocol["phase_representation"]["output_turns"])
    )
    for n, certificate in zip(TARGETS, certificates):
        assert certificate.nearest_lift_certified
        assert certificate.classification == "certified_within_budget"
        assert (
            lifted[n] - old[n] - certificate.ideal_displacement
        ).abs_max <= certificate.budget
        assert (
            I(new[n]) - old[n] - certificate.ideal_displacement
        ).lo > pi_interval().hi
    for actual, expected in zip(lifted, ideal["after"]):
        assert (actual - _interval(expected)).abs_max <= protocol["budgets"][
            "exact_source_endpoint"
        ]
    # Rational represented increments minus 4*pi cannot be exact zero.
    pair = lifted[0] + lifted[1] - old[0] - old[1]
    assert pair.lo > 0 or pair.hi < 0
    assert not row["analysis"]["exact_reflection_certified"]
    assert not row["analysis"]["future_retention_certified"]


@pytest.mark.parametrize("index", (1, 2))
def test_independent_nine_edge_work_and_immediate_geometry(evidence, index):
    protocol, attempt = evidence["protocol.json"], evidence["attempt.json"]
    row = evidence["response.json"]["stages"][index]
    before, after = row["before"], row["after"]
    old, new = _field(before, "phase"), _field(after, "phase")
    edges = before["weighted_edges"]
    touched = tuple(edge for edge in edges if edge[0] in TARGETS or edge[1] in TARGETS)
    assert len(touched) == 9
    work = sum(
        (
            w * (cos(I(old[b] - old[a])) - cos(I(new[b] - new[a])))
            for a, b, w in touched
        ),
        I(0),
    )
    old_report, new_report = _observe(before), _observe(after)
    assert (
        work - (new_report.joint_storage_bounds - old_report.joint_storage_bounds)
    ).abs_max < Q(1, 2**110)
    ideal = _interval(attempt["predictions"][row["preparation"]]["work"])
    assert (work - ideal).abs_max <= protocol["budgets"]["work_vs_exact_source"]
    certificates = bound_phase_reset(
        old,
        new,
        before["neighbors"],
        TARGETS,
        coefficient=protocol["configuration"]["phase_locking_coefficient"],
        budget=protocol["budgets"]["runtime_reset"],
    )
    ideal_new = list(map(I, old))
    for n, c in zip(TARGETS, certificates):
        ideal_new[n] += c.ideal_displacement
    input_work = sum(
        (
            w * (cos(I(old[b] - old[a])) - cos(ideal_new[b] - ideal_new[a]))
            for a, b, w in touched
        ),
        I(0),
    )
    assert (work - input_work).abs_max <= protocol["budgets"][
        "work_vs_actual_input_ideal"
    ]
    # Adjacent resets share one edge; isolated work is not additive.
    isolated = I(0)
    for n in TARGETS:
        isolated += sum(
            (
                cos(I(old[j] - old[n])) - cos(I(old[j] - new[n]))
                for j in before["neighbors"][n]
            ),
            I(0),
        )
    correction = work - isolated
    if row["preparation"] == "moving_control":
        assert work.hi < 0 and correction.lo > 0
    else:
        assert work.abs_max <= protocol["budgets"]["work_vs_exact_source"]
    lifted = tuple(
        I(v) + 2 * pi_interval() * t
        for v, t in zip(new, protocol["phase_representation"]["output_turns"])
    )
    for a, b, _ in edges:
        gap = lifted[b] - lifted[a]
        if a // 9 == b // 9 and a % 9 == 8 and b % 9 == 0:
            gap += 2 * pi_interval()
        assert gap.abs_max < pi_interval().lo / 2
    for c in range(5):
        winding = sum(
            (
                lifted[9 * c + (j + 1) % 9]
                - lifted[9 * c + j]
                + (2 * pi_interval() if j == 8 else 0)
                for j in range(9)
            ),
            I(0),
        )
        assert (winding - 2 * pi_interval()).abs_max < Q(1, 2**110)
    assert (
        new_report.joint_storage_bounds
        - 45 * (1 - cos(new_report.geometry.alpha_bounds))
    ).hi < Q(1, 100)
    assert all(-1 < v < 1 for v in new_report.form)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_callback_refresh_retains_actual_pressure_and_all_complete_rows(
    evidence, index
):
    protocol = evidence["protocol.json"]
    row = evidence["response.json"]["stages"][index]
    callback = row["callback"][0]
    fresh = _observe(callback["before"])
    _check_oracle(fresh)
    assert asdict(fresh) == callback["observation"]
    assert callback["after"] == row["after"]
    for n, (stored, bounds) in enumerate(
        zip(_field(callback["after"], "pressure"), fresh.pressure_tau_bounds)
    ):
        assert (I(stored) - bounds).abs_max <= protocol["budgets"][
            "pressure_materialization_tau"
        ]
        if row["glyph"] == "IL":
            incoming = _field(row["before"], "pressure")[n]
            expected = (
                Q(
                    float(incoming)
                    * float(protocol["configuration"]["pressure_retention"])
                )
                if n in TARGETS
                else incoming
            )
            assert callback["before"]["nodes"][n]["pressure"] == expected
    if row["glyph"] == "IL":
        metrics = callback["metrics_before_refresh"][-2:]
        assert len(metrics) == 2
        for n, metric in zip(TARGETS, metrics):
            assert metric["glyph"] == "IL"
            assert Q(metric["dnfr_after"]) == callback["before"]["nodes"][n]["pressure"]
        for key in ("IL_phase_locking", "IL_dnfr_reductions", "IL_coherence_tracking"):
            assert tuple(item["node"] for item in row["telemetry"][key]) == TARGETS
        old = _observe(row["before"])
        assert old.form_gradient == fresh.form_gradient
        for clock in ("tau", "slow", "original"):
            assert getattr(old, f"phase_{clock}_rate_bounds") == getattr(
                fresh, f"phase_{clock}_rate_bounds"
            )
        gamma = fresh.geometry.gamma_bounds
        for a, b, sa, sb in zip(
            old.form_tau_rate_bounds,
            fresh.form_tau_rate_bounds,
            old.phase_current_bounds,
            fresh.phase_current_bounds,
        ):
            assert ((b - a) - gamma * (sb - sa) / 4).abs_max < Q(1, 2**110)


def test_raw_observer_independent_signed_rows_and_common_phase_shift():
    # Generic field contract, not another organized-source response/preparation.
    form = tuple(Q((n % 7) - 3, 32) for n in range(45))
    phases = tuple(Q((n % 5) - 2, 16) for n in range(45))
    report = owner._observe_organized_source_phase_state(form=form, phases=phases)
    shifted = owner._observe_organized_source_phase_state(
        form=form, phases=tuple(v + Q(3, 8) for v in phases)
    )
    _check_oracle(report)
    assert report.phases == phases and report.form == form
    for field in (
        "form_gradient",
        "phase_current_bounds",
        "edge_storage_bounds",
        "form_tau_rate_bounds",
        "phase_tau_rate_bounds",
        "joint_storage_bounds",
    ):
        assert getattr(report, field) == getattr(shifted, field)


@pytest.mark.parametrize("field", ("form", "phases"))
@pytest.mark.parametrize("invalid", (True, float("inf"), 1j, "wrong_length"))
def test_raw_observer_rejects_invalid_primitives_before_geometry(
    monkeypatch, field, invalid
):
    def forbidden():
        pytest.fail("invalid primitive reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    values = dict(form=(Q(0),) * 45, phases=(Q(0),) * 45)
    values[field] = (
        (Q(0),) * 44 if invalid == "wrong_length" else (invalid,) + (Q(0),) * 44
    )
    with pytest.raises((ValueError, TypeError)):
        owner._observe_organized_source_phase_state(**values)
