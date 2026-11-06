"""Ideal paired Coherence work and the separate strict-runtime obstruction.

Only the two previously frozen profiles are consumed: the Reception source
with opposed child-1/2 offsets and the capacity discriminator with pair0,1
offsets +/-1/100. The shared default phase coefficient and boundaries0,1 are
fixed; no operator, trajectory, acquisition or coefficient search is run.
Ideal phase admission does not override the current signed EPI>0 IL policy.
"""

from copy import deepcopy
from fractions import Fraction as Q

import mpmath
import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, cos, sqrt
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.preconditions.coherence import validate_coherence_strict
from tnfr.physics import _organized_source_capture as owner
from tnfr.physics import _organized_source_port as port

ORIGIN, EPSILON = Q(1, 40), Q(1, 100)
DEFAULT_COEFFICIENT = Q(DEFAULT_PHASE_LOCKING_COEFFICIENT)
COEFFICIENTS = (Q(0), DEFAULT_COEFFICIENT, Q(1))
QUIET_FORM = (Q(1, 64), -Q(1, 64)) + (Q(0),) * 43
QUIET_OFFSETS = tuple(
    ORIGIN
    + (EPSILON if c == 1 else -EPSILON if c == 2 else 0)
    * (1 if j == 0 else -1 if j == 1 else 0)
    for c in range(5)
    for j in range(9)
)
MOVING_FORM = (Q(1, 10**7), -Q(1, 10**7)) + (Q(0),) * 43
MOVING_OFFSETS = (ORIGIN + EPSILON, ORIGIN - EPSILON) + (ORIGIN,) * 43
PROFILES = {
    "quiet": (QUIET_FORM, QUIET_OFFSETS),
    "moving": (MOVING_FORM, MOVING_OFFSETS),
}


@pytest.fixture(scope="module", autouse=True)
def no_response_event_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Coherence assessment cannot execute flow, events or sources")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (word_execution, "execute_network_operator_stage"),
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
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)


def _prediction(geometry, offsets, coefficient, mp):
    alpha = 2 * mp.pi / 9
    theta = tuple(alpha * (n % 9 - 4) + _mp(mp, offsets[n]) for n in range(45))
    neighbors = tuple(
        tuple(b if a == n else a for a, b, _ in geometry.weighted_edges if n in (a, b))
        for n in (0, 1)
    )
    resultants = tuple(
        sum(mp.exp(mp.j * (theta[j] - theta[n])) for j in neighbors[n]) for n in (0, 1)
    )
    errors = tuple(mp.arg(value) for value in resultants)
    displacements = tuple(_mp(mp, coefficient) * value for value in errors)
    changes = displacements + (mp.mpf(0),) * 43
    after = tuple(value + change for value, change in zip(theta, changes))
    touched = tuple(
        (a, b, w) for a, b, w in geometry.weighted_edges if a in (0, 1) or b in (0, 1)
    )
    assert tuple(map(len, neighbors)) == (5, 5) and len(touched) == 9
    work = sum(
        _mp(mp, w) * (mp.cos(theta[b] - theta[a]) - mp.cos(after[b] - after[a]))
        for a, b, w in touched
    )
    isolated = tuple(
        sum(
            mp.cos(theta[j] - theta[n]) - mp.cos(theta[j] - theta[n] - changes[n])
            for j in neighbors[n]
        )
        for n in (0, 1)
    )
    gap = theta[1] - theta[0]
    correction = (
        mp.cos(gap - changes[0])
        + mp.cos(gap + changes[1])
        - mp.cos(gap)
        - mp.cos(gap + changes[1] - changes[0])
    )
    current_jump = [mp.mpf(0)] * 45
    for a, b, w in touched:
        value = _mp(mp, w) * (mp.sin(after[b] - after[a]) - mp.sin(theta[b] - theta[a]))
        current_jump[a] += value
        current_jump[b] -= value
    return dict(
        theta=theta,
        after=after,
        resultants=resultants,
        errors=errors,
        changes=changes,
        work=work,
        isolated=isolated,
        correction=correction,
        current_jump=tuple(current_jump),
    )


@pytest.fixture(scope="module")
def predictions(geometry, mp):
    # Freeze independent exact-support calculations before consuming the API.
    return {
        (name, coefficient): _prediction(geometry, offsets, coefficient, mp)
        for name, (_, offsets) in PROFILES.items()
        for coefficient in COEFFICIENTS
    }


@pytest.fixture(scope="module")
def reports(predictions):
    return {
        (name, coefficient): owner._assess_organized_source_reflection_coherence(
            form=form, phase_offsets=offsets, phase_locking_coefficient=coefficient
        )
        for name, (form, offsets) in PROFILES.items()
        for coefficient in COEFFICIENTS
    }


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_resultants_and_ideal_endpoint_keep_the_full_lift_and_shared_snapshot(
    name, coefficient, geometry, mp, predictions, reports
):
    report, expected = reports[name, coefficient], predictions[name, coefficient]
    offsets = PROFILES[name][1]
    assert report.domain.admitted and report.phase_locking_coefficient == coefficient
    assert report.ideal_retention_certified and report.retained_windings == (1,) * 5
    assert report.storage_jump_upper_bound == 0
    for n in (0, 1):
        _contains(mp, report.resultants_real_bounds[n], expected["resultants"][n].real)
        _contains(
            mp, report.resultants_imaginary_bounds[n], expected["resultants"][n].imag
        )
        _contains(mp, report.phase_error_bounds[n], expected["errors"][n])
        _contains(mp, report.phase_displacement_bounds[n], expected["changes"][n])
        assert report.resultants_real_bounds[n].lo > 4
        assert report.phase_error_bounds[n].abs_max < Q(1, 8)
    assert report.phase_displacement_bounds[1] == -report.phase_displacement_bounds[0]
    for n, bounds in enumerate(report.phase_offset_after_bounds):
        _contains(mp, bounds, _mp(mp, offsets[n]) + expected["changes"][n])
    for child in range(5):
        increments = []
        for j in range(9):
            a, b = 9 * child + j, 9 * child + (j + 1) % 9
            gap = expected["after"][b] - expected["after"][a]
            increments.append(mp.atan2(mp.sin(gap), mp.cos(gap)))
        assert all(abs(angle) < mp.pi / 2 for angle in increments)
        assert abs(sum(increments) / (2 * mp.pi) - 1) < mp.mpf("1e-70")
    # Actual degree5 and the full-domain edge-displacement budget yield the
    # conservative positive-resultant/argument margins, not a fitted cutoff.
    baseline = 3 + 2 * cos(geometry.alpha_bounds)
    perturbation = sqrt(I(5 * 4 * Q(1, 100)))
    assert baseline.lo > Q(9, 2) and perturbation.hi < Q(1, 2)
    assert (baseline - perturbation).lo > 4


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("coefficient", COEFFICIENTS)
def test_joint_touched_edge_work_and_fresh_rows_match_independent_calculus(
    name, coefficient, mp, predictions, reports
):
    report, expected = reports[name, coefficient], predictions[name, coefficient]
    _contains(mp, report.storage_jump_bounds, expected["work"])
    for bounds, value in zip(report.isolated_storage_jump_bounds, expected["isolated"]):
        _contains(mp, bounds, value)
    _contains(mp, report.shared_edge_correction_bounds, expected["correction"])
    assert abs(
        expected["work"] - sum(expected["isolated"]) - expected["correction"]
    ) < mp.mpf("1e-70")
    assert report.work_decomposition_residual_bounds.contains(0)
    gamma = 1 / (1023 * mp.pi)
    for index, expected_current in enumerate(expected["current_jump"]):
        _contains(mp, report.phase_current_jump_bounds[index], expected_current)
        for clock, factor in (
            ("tau", mp.mpf(1)),
            ("original", mp.mpf(1023) / 1024),
            ("slow", 2 / gamma**2),
        ):
            _contains(
                mp,
                getattr(report, f"form_{clock}_rate_jump_bounds")[index],
                factor * gamma * expected_current / 4,
            )
            assert getattr(report, f"phase_{clock}_rate_jump_bounds")[index] == I(0)
    if name == "moving" and coefficient > 0:
        assert report.strict_work_decrease_certified
        assert (
            report.storage_jump_bounds.hi < 0 < report.shared_edge_correction_bounds.lo
        )
        assert _mp(mp, report.storage_jump_bounds.lo) > sum(expected["isolated"])
        norm2 = sum(change**2 for change in expected["changes"][:2])
        assert expected["work"] <= -(3 / _mp(mp, coefficient) - 2) * norm2
        assert any(
            not value.contains(0) for value in report.form_tau_rate_jump_bounds[2:]
        )
    else:
        assert not report.strict_work_decrease_certified
        assert report.storage_jump_bounds.contains(0)


def test_quiet_profile_has_real_resultants_and_zero_coefficient_is_exact(mp, reports):
    real = 2 * mp.cos(2 * mp.pi / 9) + 2 * mp.cos(_mp(mp, EPSILON)) + 1
    quiet = reports["quiet", DEFAULT_COEFFICIENT]
    for n in (0, 1):
        _contains(mp, quiet.resultants_real_bounds[n], real)
        assert quiet.resultants_imaginary_bounds[n].contains(0)
        assert quiet.phase_error_bounds[n].contains(0)
    for name in PROFILES:
        disabled = reports[name, Q(0)]
        assert disabled.phase_displacement_bounds == (I(0), I(0))
        assert (
            disabled.storage_jump_bounds
            == disabled.shared_edge_correction_bounds
            == I(0)
        )
        assert disabled.phase_current_jump_bounds == (I(0),) * 45
        assert disabled.form_tau_rate_jump_bounds == (I(0),) * 45


@pytest.mark.parametrize("amplitude", (Q(1, 64), -Q(1, 64), Q(0)))
def test_default_strict_coherence_cannot_admit_a_complete_reflected_pair(
    amplitude, geometry
):
    graph = nx.Graph()
    for n in geometry.node_order:
        form = amplitude if n == 0 else -amplitude if n == 1 else Q(0)
        graph.add_node(n, EPI=float(form), nu_f=1.0, delta_nfr=0.0)
    for a, b, w in geometry.weighted_edges:
        graph.add_edge(a, b, weight=float(w))
    before_nodes, before_graph = deepcopy(dict(graph.nodes(data=True))), deepcopy(
        graph.graph
    )
    before_edges = tuple(graph.edges(data=True))
    accepted = []
    for node in (0, 1):
        if graph.nodes[node]["EPI"] > 0:
            assert validate_coherence_strict(graph, node, emit_warnings=False) is None
            accepted.append(True)
        else:
            with pytest.raises(ValueError, match="IL precondition failed: EPI"):
                validate_coherence_strict(graph, node, emit_warnings=False)
            accepted.append(False)
    assert not all(accepted)
    assert sum(accepted) == (0 if amplitude == 0 else 1)
    assert dict(graph.nodes(data=True)) == before_nodes and graph.graph == before_graph
    assert tuple(graph.edges(data=True)) == before_edges


def test_valid_high_energy_state_has_no_endpoint_or_future_evidence():
    report = owner._assess_organized_source_reflection_coherence(
        form=(Q(1, 4), -Q(1, 4)) + (Q(0),) * 43,
        phase_offsets=QUIET_OFFSETS,
        phase_locking_coefficient=DEFAULT_COEFFICIENT,
    )
    assert report.domain.in_acute_cell and not report.domain.admitted
    assert not report.ideal_retention_certified
    assert report.retained_windings is report.storage_jump_upper_bound is None
    assert report.phase_offset_after_bounds is report.storage_jump_bounds is None
    assert report.phase_error_bounds is report.form_tau_rate_jump_bounds is None
    assert report.unresolved_conditions


@pytest.mark.parametrize(
    "bad", (True, None, float("inf"), float("nan"), -Q(1, 10), Q(11, 10))
)
def test_coefficient_rejects_before_domain_or_geometry(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid Coherence coefficient reached domain assessment")

    monkeypatch.setattr(
        owner, "_assess_organized_source_reflection_retention", forbidden
    )
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_reflection_coherence(
            form=QUIET_FORM, phase_offsets=QUIET_OFFSETS, phase_locking_coefficient=bad
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
@pytest.mark.parametrize("bad", (True, float("inf"), "short"))
def test_primitive_state_rejects_before_geometry(field, bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid Coherence source reached geometry")

    arguments = dict(
        form=QUIET_FORM,
        phase_offsets=QUIET_OFFSETS,
        phase_locking_coefficient=DEFAULT_COEFFICIENT,
    )
    values = arguments[field]
    arguments[field] = values[:-1] if bad == "short" else (bad,) + values[1:]
    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_reflection_coherence(**arguments)


def test_off_sector_state_is_rejected_without_projecting_an_ideal_endpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Off-sector Coherence source reached geometry")

    form = list(QUIET_FORM)
    form[2], form[3] = Q(1, 1024), -Q(1, 1024)
    monkeypatch.setattr(owner, "_derive_organized_source_capture_barrier", forbidden)
    with pytest.raises(ValueError, match="reflection"):
        owner._assess_organized_source_reflection_coherence(
            form=tuple(form),
            phase_offsets=QUIET_OFFSETS,
            phase_locking_coefficient=DEFAULT_COEFFICIENT,
        )
