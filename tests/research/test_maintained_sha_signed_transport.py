"""Constructed-state controls for the bounded retained-transport reader.

These controls use no retained source, trajectory, historical recertification
or candidate tube. They check smooth arithmetic, complete environmental rows
and report admission separately from the prospectively registered audit.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics import _validated_taylor as taylor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.research import maintained_sha_motion as motion
from tnfr.research import maintained_sha_reference as reference
from tnfr.research import maintained_sha_signed_transport as owner

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _overlap(actual, expected):
    assert (actual - expected).contains(0)


@pytest.fixture(autouse=True)
def no_retained_source_or_evolution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("constructed transport controls cannot read or evolve a source")

    monkeypatch.setattr(owner, "load_sha_retention", forbidden)
    monkeypatch.setattr(reference, "prepare_sha_pair_budget", forbidden)
    monkeypatch.setattr(taylor, "picard_tube", forbidden)
    for name in (
        "validate_live_sha_successor",
        "validated_box_taylor_step",
        "certify_retention",
        "prepare_sha_motion",
    ):
        monkeypatch.setattr(motion, name, forbidden)


def _differences(delta, root=I(0), *, lower=Q(1), upper=Q(2)):
    return owner._smooth_divided_differences(
        delta,
        root,
        gamma=Q(1, 2),
        C=I(1),
        S=I(0),
        Cdot=I(0),
        Sdot=I(3),
        lower=lower,
        upper=upper,
    )


def test_zero_error_uses_the_continuous_divided_differences():
    result = _differences(I(0))
    assert result.h.contains(Q(5, 4))
    assert result.a.contains(Q(3, 2))
    assert result.b.contains(Q(6, 5))
    assert result.h.lo > 0
    with pytest.raises(FrozenInstanceError):
        result.b = I(0)


def test_a_zero_crossing_error_box_encloses_the_same_continuous_limit():
    radius = Q(1, 2**20)
    result = _differences(I(-radius, radius))
    assert result.h.contains(Q(5, 4))
    assert result.a.contains(Q(3, 2))
    assert result.b.contains(Q(6, 5))
    assert result.h.lo >= 1
    assert result.h.hi <= 2


def test_away_from_zero_the_smooth_values_enclose_independent_quotients():
    result = _differences(I(Q(1, 2)))
    # F(delta)-F(0) and W_t(delta)-W_t(0), divided by the nonzero delta.
    h = 2 * sin(I(Q(1, 2))) + sin(I(Q(1, 4)))
    a = 6 * sin(I(Q(1, 4)))
    _overlap(result.h, h)
    _overlap(result.a, a)
    _overlap(result.b, a / h)


def test_disjoint_curvature_or_unadmitted_sinc_domain_is_not_coerced():
    with pytest.raises((ArithmeticError, ValueError)):
        _differences(I(0), lower=Q(2), upper=Q(3))
    with pytest.raises((ArithmeticError, ValueError)):
        _differences(I(3))


def test_signed_negative_initial_correction_storage_keeps_an_upper_bound():
    result = owner._excitation_envelope(
        I(1), I(Q(3, 4)), epsilon=Q(1), remainder_integral=Q(0)
    )
    assert result.C0 == I(Q(-1, 2))
    assert result.radicand.contains(Q(1, 2))
    _overlap(result.upper, 1 + sqrt(I(Q(1, 2))))
    assert result.lower == I(0)
    assert result.floor_gate.status == "excluded"


def test_zero_radicand_is_admitted_without_differentiating_sqrt_at_zero():
    result = owner._excitation_envelope(
        I(1), I(1), epsilon=Q(1), remainder_integral=Q(0)
    )
    assert result.C0 == I(-1)
    assert result.radicand.contains(0)
    assert result.upper.contains(1)
    assert result.lower == I(0)


def test_wrapped_initial_storage_and_correction_use_opposite_safe_endpoints():
    squared = I(Q(99, 100), Q(101, 100))
    correction = I(Q(9, 100), Q(11, 100))
    epsilon, remainder = Q(1, 4), Q(1, 20)
    result = owner._excitation_envelope(
        squared, correction, epsilon=epsilon, remainder_integral=remainder
    )
    assert result.C0.contains(Q(77, 100))
    assert result.C0.contains(Q(83, 100))
    assert result.C0.width <= Q(6, 100) + Q(1, 2**100)
    upper, lower = result.upper.hi, result.lower.lo
    # These rational polynomial inequalities independently certify the
    # direction of rounding, without duplicating the square-root algorithm.
    assert upper**2 - 2 * (epsilon + remainder) * upper - result.C0.hi >= 0
    floor_margin = result.C0.lo - 2 * remainder * upper
    assert floor_margin > 0
    assert lower > 0
    assert lower**2 + 2 * epsilon * lower <= floor_margin
    assert result.D0.lo <= floor_margin
    assert result.floor_gate.status == "pass"


def test_unresolved_initial_signed_storage_does_not_become_a_positive_floor():
    result = owner._excitation_envelope(
        I(1),
        I(Q(49, 100), Q(51, 100)),
        epsilon=Q(1),
        remainder_integral=Q(0),
    )
    assert result.C0.lo < 0 < result.C0.hi
    assert result.lower == I(0)
    assert result.floor_gate.status == "unavailable"


def test_public_wrapper_reads_fresh_retention_instead_of_accepting_a_cached_report(
    monkeypatch,
):
    fresh, expected = object(), object()
    calls = []

    def load(path, *, repository_root):
        assert path == "constructed-evidence"
        assert repository_root == "constructed-repository"
        calls.append("readmit_full_retention")
        return fresh

    def consume(value):
        assert value is fresh
        calls.append("consume_fresh_primitives")
        return expected

    monkeypatch.setattr(owner, "load_sha_retention", load)
    monkeypatch.setattr(owner, "_audit_from_fresh_retention", consume)
    assert (
        owner.audit_sha_signed_transport(
            "constructed-evidence", repository_root="constructed-repository"
        )
        is expected
    )
    assert calls == ["readmit_full_retention", "consume_fresh_primitives"]


def _four_node_reference(sign):
    graph = nx.Graph()
    u, alpha = Q(1, 5), Q(1, 4)
    names = ("parent", "child", "rest", "remote")
    phases = tuple(value - sign * alpha / 4 for value in (0, 0, 0, sign * alpha))
    for node, form, phase, capacity in zip(
        names, (-u / 2, u / 2, 0, 0), phases, (2, 1, 2, 1), strict=True
    ):
        graph.add_node(node, EPI=form, theta=phase, nu_f=capacity, delta_nfr=99)
    graph.add_weighted_edges_from(
        (("parent", "child", 2), ("parent", "rest", 3), ("rest", "remote", 5))
    )
    graph.graph.update(_t=0, GAMMA={"type": "none"})
    return reference._bound_reference(
        graph, parent="parent", child="child", reference_model=MODEL
    )


def _admitted_contrast(report):
    pair = report.pair_budget.connected
    # Shared primitive admission materializes these non-dyadic graph inputs.
    # The independent law formulas must use that represented contrast.
    return pair.source.epi[pair.child_index] - pair.source.epi[pair.parent_index]


@pytest.fixture(scope="module", params=(-1, 1))
def complete_path_case(request):
    report = _four_node_reference(request.param)
    source = report.pair_budget.connected.source
    tube = tuple(map(I, source.epi + source.phase))
    cell = owner._bound_cell(report, owner._coefficient_rows(report), tube)
    return request.param, report, cell


def test_complete_four_node_rows_retain_remote_acceleration(complete_path_case):
    sign, report, cell = complete_path_case
    u = _admitted_contrast(report)
    alpha, w, v, remote = Q(1, 4), Q(2), Q(3), Q(5)
    pi = pi_interval()
    angular_rate = 3 * v * u / (4 * pi)
    angular_acceleration = sign * (3 * v / 2 + 2 * remote) * sin(I(alpha)) / pi**2
    assert report.chart.gamma == Q(1, 2)
    assert report.chart.kappa == Q(11, 4)
    assert report.chart.root.bracket.contains(0)
    assert cell.V0.contains(0)
    assert cell.C.contains(1)
    assert cell.S.contains(0)
    _overlap(cell.Vdot, -sign * v * sin(I(alpha)) / pi)
    _overlap(cell.Vddot, v * u * (3 * (w + v) + v * cos(I(alpha))) / (2 * pi**2))
    _overlap(cell.alpha_rates[0], angular_rate)
    _overlap(cell.alpha_accelerations[0], angular_acceleration)
    _overlap(cell.Cdot, I(0))
    _overlap(cell.Sdot, angular_rate)
    _overlap(cell.Cddot, -(angular_rate**2))
    _overlap(cell.Sddot, angular_acceleration)
    assert all(cell.means_consistent)
    assert cell.form_mean.contains(0)
    assert cell.phase_mean.contains(0)
    assert cell.actual_phase_in_chart


def test_cell_constructive_bounds_cover_complete_point_values(complete_path_case):
    sign, report, cell = complete_path_case
    u = _admitted_contrast(report)
    assert cell.V1 >= cell.Vdot.abs_max
    assert cell.V2 >= cell.Vddot.abs_max
    assert cell.R0 == 1
    assert cell.R1 >= sqrt(cell.Cdot**2 + cell.Sdot**2).hi
    assert cell.R2 >= sqrt(cell.Cddot**2 + cell.Sddot**2).hi
    assert cell.Pdom >= u
    assert cell.epsilon.lo > 0
    assert cell.eta_R.lo > 0
    assert cell.unavailable_reasons == ()
    # At delta=delta*=0 the correction remains nonzero when the reference
    # moves. Its independent complete-law value has the same sign in both
    # opposite-work states: P0=-3*v*u^2/20.
    smooth = owner._smooth_divided_differences(
        I(0),
        report.chart.root.bracket,
        gamma=report.chart.gamma,
        C=cell.C,
        S=cell.S,
        Cdot=cell.Cdot,
        Sdot=cell.Sdot,
        lower=report.chart.curvature_lower,
        upper=report.chart.curvature_upper,
    )
    correction = -u * smooth.b / (2 / pi_interval())
    assert correction.contains(-9 * u**2 / 20)
    excitation = sqrt(I(11 * u**2 / 4))
    assert (cell.epsilon.hi * excitation - 9 * u**2 / 20).lo > 0
    # At this point q=0, hdot=0 and bdot=(2/5)*Sddot. The remote
    # neighbor therefore remains in the true R=p*bdot/k term.
    remainder = sign * u * Q(29, 2) * sin(I(Q(1, 4))) / (5 * pi_interval())
    assert (cell.eta_R.hi * excitation - abs(remainder)).lo > 0


def test_positive_excitation_alone_does_not_supply_offset_or_a_turn():
    report = _four_node_reference(1)
    envelope = owner._excitation_envelope(
        2 * report.state.relative_storage,
        I(0),
        epsilon=Q(0),
        remainder_integral=Q(0),
    )
    quiet = owner._response_bounds(
        report,
        envelope,
        form_displacement=Q(0),
        phase_displacement=Q(0),
        full_window=True,
    )
    assert quiet.signal_gate.status == "pass"
    assert quiet.offset_gate.status == "pass"
    assert quiet.turn_gate.status == "excluded"
    shifted = owner._response_bounds(
        report,
        envelope,
        form_displacement=Q(0),
        phase_displacement=Q(1),
        full_window=True,
    )
    assert shifted.offset_gate.status == "excluded"
    assert shifted.turn_gate.status == "unavailable"


def test_missing_whole_window_never_borrows_a_turn_from_an_available_prefix():
    report = _four_node_reference(1)
    envelope = owner._excitation_envelope(
        2 * report.state.relative_storage,
        I(0),
        epsilon=Q(0),
        remainder_integral=Q(0),
    )
    result = owner._response_bounds(
        report,
        envelope,
        form_displacement=Q(0),
        phase_displacement=Q(0),
        full_window=False,
    )
    assert result.turn_gate.status == "unavailable"


def test_initial_correction_keeps_its_sign_and_the_independent_hidden_storage(
    complete_path_case,
):
    _, report, _ = complete_path_case
    u = _admitted_contrast(report)
    initial = owner._initial_transport(report, owner._coefficient_rows(report))
    assert initial.P0.contains(-9 * u**2 / 20)
    assert initial.P0.hi < 0
    assert initial.C0.contains(73 * u**2 / 20)
    assert initial.corrected_storage_gate.status == "pass"
    _overlap(initial.hidden_storage, 1 - cos(I(Q(1, 4))))
    assert initial.storage_identity_residual.contains(0)
    assert initial.storage_identity_consistent
    assert initial.unavailable_reasons == ()


def test_partial_cell_collection_does_not_trust_a_complete_flag(complete_path_case):
    _, report, cell = complete_path_case
    initial = owner._initial_transport(report, owner._coefficient_rows(report))
    result = owner._aggregate(report, initial, (cell,), full_window=True)
    assert not result.full_window
    assert result.accepted_cells == 1
    assert result.envelope is None
    assert result.response is None
    assert "fixed_256_cell_half_unit_window_unavailable" in result.unavailable_reasons


def test_internal_consumer_rejects_an_unadmitted_cached_report():
    with pytest.raises(TypeError, match="fresh shared retained SHA evidence required"):
        owner._audit_from_fresh_retention(object())
