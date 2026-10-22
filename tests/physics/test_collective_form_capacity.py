"""Synthetic controls for complete VAL/NUL form-capacity reset identities.

Pure proposals act only on small synthetic states. The final controls re-admit
retained contact inputs and evaluate analytical bounds, without proposing or
executing a contact event. No word, grammar admission or trajectory is run.
Exact edge arithmetic and the shared weighted-form/bare-phase reader check
different parts of the argument; these controls do not certify maintenance.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.factor_contracts import resolve_operator_factors
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.types import Glyph

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right, strict=True)), Q(0))


def _gradient(graph, values):
    """Differentiate the edge form storage without the report's matrix code."""
    result = [Q(0)] * len(graph)
    for left, right, data in graph.edges(data=True):
        flux = Q(data["weight"]) * (values[left] - values[right])
        result[left] += flux
        result[right] -= flux
    return tuple(result)


def _storage(graph, values):
    return sum(
        (
            Q(data["weight"]) * (values[left] - values[right]) ** 2 / 2
            for left, right, data in graph.edges(data=True)
        ),
        Q(0),
    )


def _rows(graph, name):
    return tuple(Q(graph.nodes[node][name]) for node in graph)


def _weights(graph):
    return tuple(Q(graph.degree[node]) / Q(graph.nodes[node]["nu_f"]) for node in graph)


def _mean(weights, values):
    return _dot(weights, values) / sum(weights)


def _field(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _reset(graph, glyph, factor, targets=(0, 1), *, edge_aware=True):
    """Build proposals from one source, then construct a detached output graph."""
    proposals = {
        node: propose_scale_operator(
            glyph=glyph,
            factor=factor,
            vf_before=graph.nodes[node]["nu_f"],
            dnfr_before=graph.nodes[node]["delta_nfr"],
            edge_aware_enabled=edge_aware,
            epi_before=graph.nodes[node]["EPI"],
        )
        for node in targets
    }
    after = graph.copy()
    for node, proposal in proposals.items():
        after.nodes[node]["nu_f"] = proposal.vf_after
        if proposal.write_epi:
            after.nodes[node]["EPI"] = proposal.epi_after
        if glyph is Glyph.NUL:
            after.nodes[node]["delta_nfr"] = proposal.dnfr_after
    return after, proposals


@pytest.fixture
def graph():
    result = nx.Graph()
    result.add_nodes_from(range(4))
    result.add_weighted_edges_from(((0, 1, 2), (1, 2, 0.5), (2, 3, 1.5), (0, 2, 0)))
    for node, form, phase, capacity in zip(
        result,
        (0.75, -0.5, 0.25, 0.125),
        (0, 0.0625, -0.125, 0.125),
        (0.5, 2, 0.75, 1.25),
        strict=True,
    ):
        result.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=0.25)
    result.graph["GAMMA"] = {"type": "none"}
    return result


@pytest.mark.parametrize("glyph,factor", ((Glyph.VAL, 2), (Glyph.NUL, 0.5)))
def test_complete_reset_has_exact_edge_work_including_shared_target_edge(
    graph, glyph, factor
):
    after, proposals = _reset(graph, glyph, factor)
    before_x, after_x = _rows(graph, "EPI"), _rows(after, "EPI")
    change = tuple(b - a for a, b in zip(before_x, after_x, strict=True))
    direct = _storage(after, after_x) - _storage(graph, before_x)
    gradient = _gradient(graph, before_x)
    assert direct == _dot(gradient, change) + _dot(change, _gradient(graph, change)) / 2
    assert _field(after).form_storage - _field(graph).form_storage == direct
    assert _field(after).phase_storage == _field(graph).phase_storage

    separate = Q(0)
    for target in proposals:
        alone = list(before_x)
        alone[target] = after_x[target]
        separate += _storage(graph, alone) - _storage(graph, before_x)
    cross = -Q(graph.edges[0, 1]["weight"]) * change[0] * change[1]
    assert cross != 0 and direct == separate + cross
    assert graph.nodes[0]["EPI"] == 0.75  # Proposal evaluation preserves its source.


@pytest.mark.parametrize("glyph,factor", ((Glyph.VAL, 2), (Glyph.NUL, 0.5)))
def test_capacity_metric_changes_both_declared_lifted_mean_leaves(graph, glyph, factor):
    after, _ = _reset(graph, glyph, factor)
    rho, rho_after = _weights(graph), _weights(after)
    before_x, after_x, phase = (
        _rows(graph, "EPI"),
        _rows(after, "EPI"),
        _rows(graph, "theta"),
    )
    change = tuple(b - a for a, b in zip(before_x, after_x, strict=True))
    metric_change = tuple(b - a for a, b in zip(rho, rho_after, strict=True))
    old_form_mean, old_phase_mean = _mean(rho, before_x), _mean(rho, phase)
    form_jump = _mean(rho_after, after_x) - old_form_mean
    phase_jump = _mean(rho_after, phase) - old_phase_mean
    assert form_jump == (
        _dot(rho_after, change)
        + _dot(metric_change, tuple(x - old_form_mean for x in before_x))
    ) / sum(rho_after)
    assert phase_jump == _dot(
        metric_change, tuple(angle - old_phase_mean for angle in phase)
    ) / sum(rho_after)
    assert form_jump != 0 and phase_jump != 0
    assert _rows(after, "theta") == phase
    for state in (graph, after):
        report, weights = _field(state), _weights(state)
        for rates in (report.form_rates, report.phase_rates):
            balance = sum((weight * rate for weight, rate in zip(weights, rates)), I(0))
            assert balance.contains(0) and balance.width < Q(1, 10**11)


def test_unclipped_joint_scaling_preserves_charge_but_not_its_normalized_mean(graph):
    after, _ = _reset(graph, Glyph.NUL, 0.5)
    rho, rho_after = _weights(graph), _weights(after)
    before_x, after_x = _rows(graph, "EPI"), _rows(after, "EPI")
    assert _dot(rho_after, after_x) == _dot(rho, before_x)
    assert sum(rho_after) != sum(rho)
    assert _mean(rho_after, after_x) != _mean(rho, before_x)
    clipped, _ = _reset(graph, Glyph.VAL, 2)
    assert _dot(_weights(clipped), _rows(clipped, "EPI")) != _dot(rho, before_x)


@pytest.mark.parametrize("glyph,factor", ((Glyph.VAL, 2), (Glyph.NUL, 0.5)))
def test_fresh_complete_rows_keep_form_change_and_untargeted_phase_response(
    graph, glyph, factor
):
    after, _ = _reset(graph, glyph, factor)
    before_field, after_field = _field(graph), _field(after)
    change = tuple(
        b - a for a, b in zip(before_field.epi, after_field.epi, strict=True)
    )
    gradient_change = _gradient(graph, change)
    ratios = tuple(b / a for a, b in zip(before_field.capacity, after_field.capacity))
    assert after_field.pressure == before_field.pressure
    old_phase = before_field.phase_rate_numerators()
    new_phase = after_field.phase_rate_numerators()
    for node, ratio in enumerate(ratios):
        residual = after_field.form_rates[node] - ratio * before_field.form_rates[node]
        assert residual.contains(0) and residual.width < Q(1, 10**11)
        mobility = before_field.capacity[node] / before_field.degrees[node]
        assert new_phase[node] - old_phase[node] == (
            (ratio - 1) * old_phase[node] + ratio * mobility * gradient_change[node]
        )
    assert ratios[2] == 1 and gradient_change[2] != 0
    assert after_field.form_rates[2] == before_field.form_rates[2]
    assert new_phase[2] != old_phase[2]
    assert (
        before_field.degrees[0] == 2
    )  # The zero-conductance edge is still bare support.


def test_nul_stored_product_does_not_preserve_the_refreshed_form_row(graph):
    after, proposals = _reset(graph, Glyph.NUL, 0.5)
    old, new = _field(graph), _field(after)
    for node, proposal in proposals.items():
        assert Q(proposal.vf_after) * Q(proposal.dnfr_after) == (
            Q(proposal.vf_before) * Q(proposal.dnfr_before)
        )
        assert old.pressure[node] == new.pressure[node]
        assert not old.form_rates[node].contains(0)
        assert not (new.form_rates[node] - old.form_rates[node]).contains(0)
        residual = new.form_rates[node] - old.form_rates[node] / 2
        assert residual.contains(0) and residual.width < Q(1, 10**11)


def test_local_nul_can_raise_whole_edge_storage():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1)
    for node, form in enumerate((0.5, 0.75)):
        graph.nodes[node].update(EPI=form, theta=0, nu_f=1, delta_nfr=0)
    after, _ = _reset(graph, Glyph.NUL, 0.5, (0,))
    assert _storage(after, _rows(after, "EPI")) - _storage(
        graph, _rows(graph, "EPI")
    ) == Q(3, 32)


def test_uniform_capacity_control_scales_both_rows_but_full_reset_does_not(graph):
    targets = tuple(graph)
    pure, _ = _reset(graph, Glyph.NUL, 0.5, targets, edge_aware=False)
    complete, _ = _reset(graph, Glyph.NUL, 0.5, targets)
    old, clock, changed = _field(graph), _field(pure), _field(complete)
    for left, right in zip(old.form_rates, clock.form_rates):
        assert (right - left / 2).contains(0)
    old_phase = old.phase_rate_numerators()
    assert clock.phase_rate_numerators() == tuple(value / 2 for value in old_phase)
    assert changed.phase_rate_numerators() == tuple(value / 4 for value in old_phase)
    assert changed.form_rates == clock.form_rates
    assert not old.form_rates[0].contains(0) and old_phase[0] != 0
    # At this state the form row requires clock factor 1/2, phase requires 1/4.
    assert not (changed.phase_rates[0] - old.phase_rates[0] / 2).contains(0)


def test_local_capacity_changes_do_not_supply_a_common_clock(graph):
    after, _ = _reset(graph, Glyph.NUL, 0.5, (0,), edge_aware=False)
    old, new = _field(graph), _field(after)
    assert not old.form_rates[0].contains(0) and not old.form_rates[1].contains(0)
    assert (new.form_rates[0] - old.form_rates[0] / 2).contains(0)
    assert new.form_rates[1] == old.form_rates[1]
    assert not (new.form_rates[1] - old.form_rates[1] / 2).contains(0)


def test_form_identity_change_uses_the_complete_centered_reset(graph):
    after, _ = _reset(graph, Glyph.VAL, 2)
    before_x, after_x = _rows(graph, "EPI"), _rows(after, "EPI")
    before_center = sum(before_x) / len(before_x)
    after_center = sum(after_x) / len(after_x)
    before_shape = tuple(value - before_center for value in before_x)
    after_shape = tuple(value - after_center for value in after_x)
    shape_change = tuple(b - a for a, b in zip(before_shape, after_shape, strict=True))
    direct = _dot(after_shape, after_shape) - _dot(before_shape, before_shape)
    assert direct == 2 * _dot(before_shape, shape_change) + _dot(
        shape_change, shape_change
    )
    assert direct > 0
    # Primitive phases have not changed, so their contribution to Z squared cancels.
    assert _rows(after, "theta") == _rows(graph, "theta")


def test_local_complete_scale_needs_the_retained_common_form_origin(graph):
    shifted = graph.copy()
    for node in shifted:
        shifted.nodes[node]["EPI"] += 0.125
    assert _gradient(graph, _rows(graph, "EPI")) == _gradient(
        shifted, _rows(shifted, "EPI")
    )
    original_output, _ = _reset(graph, Glyph.NUL, 0.5, (0,))
    shifted_output, _ = _reset(shifted, Glyph.NUL, 0.5, (0,))
    difference = tuple(
        b - a
        for a, b in zip(_rows(original_output, "EPI"), _rows(shifted_output, "EPI"))
    )
    assert difference == (Q(1, 16), Q(1, 8), Q(1, 8), Q(1, 8))
    assert _gradient(original_output, difference) != (Q(0),) * 4
    # The two inputs had identical relative geometry; the outputs do not.
    assert (
        _field(original_output).phase_rate_numerators()
        != _field(shifted_output).phase_rate_numerators()
    )


def test_default_factors_and_even_reciprocal_factors_do_not_invert_full_state(graph):
    factors = CORE_DEFAULTS["GLYPH_FACTORS"]
    default_val, _ = _reset(graph, Glyph.VAL, factors["VAL_scale"])
    default_pair, _ = _reset(default_val, Glyph.NUL, factors["NUL_scale"])
    assert Q(factors["VAL_scale"]) * Q(factors["NUL_scale"]) != 1
    assert _rows(default_pair, "nu_f") != _rows(graph, "nu_f")

    expanded, _ = _reset(graph, Glyph.VAL, 2)
    returned, _ = _reset(expanded, Glyph.NUL, 0.5)
    assert _rows(returned, "nu_f") == _rows(graph, "nu_f")
    assert returned.nodes[0]["EPI"] == 0.5 != graph.nodes[0]["EPI"]
    assert returned.nodes[1]["EPI"] == graph.nodes[1]["EPI"]
    assert returned.nodes[1]["delta_nfr"] != graph.nodes[1]["delta_nfr"]


@pytest.mark.parametrize("form,expected", ((0.75, 1), (-0.75, -1), (-0.125, -0.25)))
def test_signed_default_projection_uses_final_form_and_keeps_capacity_scale(
    form, expected
):
    proposal = propose_scale_operator(
        glyph=Glyph.VAL,
        factor=2,
        vf_before=0.5,
        epi_before=form,
        edge_aware_enabled=True,
    )
    assert proposal.epi_after == expected
    assert proposal.vf_after == 1
    assert Q(proposal.epi_after) * Q(form) > 0
    if abs(form) > 0.5:
        assert proposal.effective_epi_scale != proposal.requested_scale


def test_positive_lower_rail_can_reverse_the_nominal_nul_form_change():
    proposal = propose_scale_operator(
        glyph=Glyph.NUL,
        factor=0.5,
        vf_before=1,
        dnfr_before=0.25,
        epi_before=0.25,
        edge_aware_enabled=True,
        epi_min=0.5,
        epi_max=1,
    )
    assert proposal.raw_epi_after == 0.125 < proposal.epi_before < proposal.epi_after
    assert proposal.epi_after == 0.5 and proposal.clip_delta == 0.375
    # This supplied scalar starts outside that rail: admission must be assessed separately.
    assert proposal.vf_after == 0.5 and proposal.dnfr_after == 0.5


@pytest.fixture(scope="module")
def retained_contact():
    """Reuse strict input readers, never import or call the frozen producer."""
    from tests.research import test_collective_pressure_handoff_evidence as evidence

    paths = tuple(evidence.EVIDENCE / name for name in ("response.json", "report.json"))
    receipts = tuple(evidence.file_receipt(path) for path in paths)
    response, report = evidence._read("response.json"), evidence._read("report.json")
    assert response["status"] == "completed"
    assert report["all_checks_passed"] is False
    assert len(report["checks"]) == 9
    assert sum(check["passed"] is True for check in report["checks"]) == 8
    capture = response["captures"]["OZ_completed"]
    source = evidence._rebuild(capture)
    encoded = evidence._mapping(evidence._mapping(capture["state"])["graph"])
    metadata = {
        name: evidence._value(encoded[name])
        for name in ("EPI_MIN", "EPI_MAX", "CLIP_MODE", "GLYPH_FACTORS")
    }
    assert "EDGE_AWARE_ENABLED" not in encoded  # The declared default is active.

    alpha = 2 * pi_interval() / 5
    errors = tuple(
        arg(
            cos(I(phase) - (node % 10 // 2) * alpha),
            sin(I(phase) - (node % 10 // 2) * alpha),
        )
        for node, phase in zip(source.nodes, source.phase, strict=True)
    )
    phase_center, form_center = sum(errors, I(0)) / 20, sum(source.epi) / 20
    norm_squared = sum(((x - form_center) ** 2 for x in source.epi), Q(0)) + sum(
        ((error - phase_center) ** 2 for error in errors), I(0)
    )
    yield source, metadata, norm_squared
    assert receipts == tuple(evidence.file_receipt(path) for path in paths)


def test_retained_input_admits_the_exact_scale_bound_premises(retained_contact):
    source, metadata, norm_squared = retained_contact
    common_form = Q(5, 16)
    assert source.nodes == tuple(range(20))
    assert source.epi == (
        common_form + Q(9, 8192),
        common_form + Q(7, 8192),
        *((common_form,) * 18),
    )
    assert metadata["EPI_MIN"] == -1 and metadata["EPI_MAX"] == 1
    assert metadata["CLIP_MODE"] == "hard"
    assert all(nu > 0 for nu in source.capacity)
    boundary_edges = {
        (left, right): weight
        for (left, right), weight in zip(source.edges, source.conductances)
        if left in (0, 1) and right in (2, 3, 8, 9)
    }
    assert set(boundary_edges) == {(i, j) for i in (0, 1) for j in (2, 3, 8, 9)}
    assert set(boundary_edges.values()) == {Q(1)}
    assert norm_squared.hi < Q(1, 256**2) < Q(1, 256)
    # Phase is unchanged by these maps, and its retained excess is nonnegative.
    alpha = 2 * pi_interval() / 5
    assert (source.phase_storage - 40 * (1 - cos(alpha))).lo > 0


@pytest.mark.parametrize("glyph", (Glyph.VAL, Glyph.NUL))
def test_default_local_scale_has_an_inside_endpoint_bound_above_the_energy_gate(
    retained_contact, glyph
):
    source, metadata, norm_squared = retained_contact
    factor_key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
    assert factor_key not in metadata["GLYPH_FACTORS"]
    factor = Q(resolve_operator_factors(metadata["GLYPH_FACTORS"], glyph)[factor_key])
    deviation = factor - 1 if glyph is Glyph.VAL else 1 - factor
    assert Q(1, 13) < deviation < Q(1, 12)

    # Bound the whole admitted factor interval, not one evaluated proposal.
    common_form, rounding = Q(5, 16), Q(1, 2**50)
    maximum_form = max(source.epi[:2])
    assert 0 < maximum_form * (1 + Q(1, 12)) < 1
    gap_floor = common_form / 13 - rounding
    if glyph is Glyph.NUL:
        gap_floor -= Q(9, 8192)
    assert gap_floor > 0
    form_storage_floor = 8 * gap_floor**2 / 2
    assert form_storage_floor > Q(1, 6400)

    # Triangle inequality uses Z, while the reconstructed quantity is Z squared.
    assert norm_squared.hi < Q(1, 256**2)
    displacement_bound = maximum_form / 12 + rounding
    norm_upper = Q(1, 256) + Q(3, 2) * displacement_bound
    assert norm_upper < Q(1, 16)
    # An endpoint inside the ball but outside a sufficient energy sublevel
    # neither certifies its future maintenance nor proves that it will escape.


def test_retained_topology_has_a_cheaper_phase_boundary_than_both_form_floors(
    retained_contact,
):
    source, _, _ = retained_contact
    signs = tuple(Q(1 if node < 10 else -1) for node in source.nodes)
    reference_phase_pi_coefficients = tuple(
        Q(2 * (node % 10 // 2), 5) for node in source.nodes
    )
    assert sum(signs) == 0 and _dot(signs, signs) == 20
    crossing = tuple(
        (left, right) for left, right in source.edges if signs[left] != signs[right]
    )
    assert set(crossing) == {(0, 10), (0, 11), (1, 10), (1, 11)}
    assert all(
        reference_phase_pi_coefficients[left] == reference_phase_pi_coefficients[right]
        for left, right in crossing
    )
    squared_gap_coefficient = sum(
        (signs[left] - signs[right]) ** 2 for left, right in source.edges
    )
    assert squared_gap_coefficient / 2 == 8
    radius_squared = Q(1, 256)
    amplitude_squared = radius_squared / _dot(signs, signs)
    # The global inequality 1-cos(z) <= z^2/2 supplies this upper bound;
    # no square root, trigonometric evaluation or boundary trajectory is needed.
    boundary_storage_upper = squared_gap_coefficient * amplitude_squared / 2
    assert boundary_storage_upper == Q(2, 5) * radius_squared == Q(1, 640)
    rounding, common_form = Q(1, 2**50), Q(5, 16)
    form_floors = (
        4 * (common_form / 13 - rounding) ** 2,
        4 * (common_form / 13 - Q(9, 8192) - rounding) ** 2,
    )
    assert all(floor > boundary_storage_upper for floor in form_floors)
    # Any scalar-H sublevel containing either endpoint also contains
    # an identity-boundary state; this is not a proof of dynamical escape.


def test_positive_mean_leaves_do_not_remove_the_phase_boundary_control(
    retained_contact,
):
    source, _, _ = retained_contact
    # An arbitrary positive metric different from the source metric. The
    # theorem allows any positive metric; this supplies an exact witness.
    weights = tuple(
        Q(degree) / capacity + Q(node + 1, 20)
        for node, degree, capacity in zip(source.nodes, source.degrees, source.capacity)
    )
    assert all(weight > 0 for weight in weights)
    # Keep the constant form, reference phase's pi coefficient and perturbing
    # phase's s coefficient separate; no approximation of pi or s is made.
    shapes = (
        (Q(0),) * 20,
        tuple(Q(2 * (node % 10 // 2), 5) for node in source.nodes),
        tuple(Q(1 if node < 10 else -1) for node in source.nodes),
    )
    for shape, desired_mean in zip(shapes, (Q(7, 13), Q(2, 7), Q(-5, 17))):
        shift = desired_mean - _mean(weights, shape)
        shifted = tuple(value + shift for value in shape)
        assert _mean(weights, shifted) == desired_mean
        center = sum(shifted) / len(shifted)
        old_center = sum(shape) / len(shape)
        assert tuple(value - center for value in shifted) == tuple(
            value - old_center for value in shape
        )
        assert all(
            shifted[left] - shifted[right] == shape[left] - shape[right]
            for left, right in source.edges
        )
    # Form and lifted phase means can therefore be assigned by common shifts
    # without changing the centered identity or either edge-based storage.
