"""Actual enabled/disabled NUL form-policy comparison on the fixed source.

Both configurations use read-only stage admission and the shared pure proposal
kernel. The disabled write flag retains admitted source form. Complete sine
rows and analytic finite bounds are checked without runtime event execution,
trajectory integration, retained evidence replay or a changed horizon.
"""

from copy import deepcopy
from fractions import Fraction as Q

import pytest

from tests.research import test_nul_cross_law_observation as cross_law
from tests.research import test_nul_transverse_response as transverse
from tests.research.test_nul_causal_class_exit import (
    CRITICAL_EDGES,
    HORIZON,
    MODEL,
    _source_graph,
    _state,
)
from tnfr.constants import DEFAULTS
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage
from tnfr.operators._epi_domain import require_real_scalar_epi
from tnfr.operators._scale_operator_kernel import propose_scale_operator
from tnfr.operators.definitions import Coherence, Contraction, Silence
from tnfr.operators.factor_contracts import resolve_operator_factors
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.types import Glyph

full_tangent = transverse.full_tangent


@pytest.fixture(scope="module", autouse=True)
def no_event_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("policy controls must not execute or prepare a runtime event")

    with pytest.MonkeyPatch.context() as patch:
        for name in (
            "execute_pointwise_stage",
            "execute_operator_major_stage",
            "_propose_pointwise",
        ):
            patch.setattr(network_stage, name, forbidden)
        yield


@pytest.fixture(scope="module")
def policies(no_event_execution):
    enabled = _source_graph()
    disabled = deepcopy(enabled)
    disabled.graph["EDGE_AWARE_ENABLED"] = False
    factors = resolve_operator_factors(None, Glyph.NUL)
    operator = Contraction()
    context = ValidatedSequence(
        (operator, Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    ).step(0)
    records = []
    for source in (enabled, disabled):
        original = _state(source)
        preflight = network_stage._preflight_two_phase_stage(
            source,
            operator,
            (0,),
            sequence_context=context,
            execution_kwargs={},
        )
        admitted_form = require_real_scalar_epi(
            source.nodes[0]["EPI"], operator="Contraction", label="source form"
        )
        edge_aware = source.graph["EDGE_AWARE_ENABLED"]
        proposal = propose_scale_operator(
            glyph=Glyph.NUL,
            factor=factors["NUL_scale"],
            vf_before=source.nodes[0]["nu_f"],
            dnfr_before=source.nodes[0]["delta_nfr"],
            configured_densification_factor=factors.get("NUL_densification_factor"),
            edge_aware_enabled=edge_aware,
            epi_before=admitted_form if edge_aware else None,
            epi_min=source.graph["EPI_MIN"],
            epi_max=source.graph["EPI_MAX"],
            epsilon=source.graph.get(
                "EDGE_AWARE_EPSILON", DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12)
            ),
            clip_mode=source.graph["CLIP_MODE"],
            clip_soft_k=source.graph.get("CLIP_SOFT_K", DEFAULTS["CLIP_SOFT_K"]),
        )
        successor = deepcopy(source)
        successor.nodes[0].update(
            nu_f=proposal.vf_after,
            delta_nfr=proposal.dnfr_after,
            EPI=proposal.epi_after if proposal.write_epi else admitted_form,
        )
        assert _state(source) == original
        assert _state(preflight.snapshot) == original
        records.append((source, successor, proposal, preflight))
    return tuple(records)


@pytest.fixture(scope="module")
def branches(policies):
    # Reuse only the preceding formal full-Jacobian fixture. Its support is
    # rebuilt from this actual prepared source, not a cached symmetry verdict.
    source, successor, proposal, _ = policies[0]
    return source, successor, proposal


@pytest.mark.parametrize("policy", (0, 1))
def test_both_real_configurations_admit_the_same_prefix_without_source_writes(
    policies, policy
):
    source, successor, proposal, preflight = policies[policy]
    enabled, disabled = policies[0][0], policies[1][0]
    expected_config = dict(enabled.graph)
    expected_config["EDGE_AWARE_ENABLED"] = False
    assert disabled.graph == expected_config
    assert tuple(enabled.nodes(data=True)) == tuple(disabled.nodes(data=True))
    assert tuple(enabled.edges(data=True)) == tuple(disabled.edges(data=True))
    assert preflight.exact_stage and preflight.snapshot is not source
    assert _state(preflight.snapshot) == _state(source)
    assert source.nodes[0]["_grammar_u2_debt"] == 1
    assert tuple(source.nodes[0]["glyph_history"]) == ("AL", "UM", "SHA", "IL", "OZ")
    assert source.graph == successor.graph
    assert tuple(source.edges(data=True)) == tuple(successor.edges(data=True))
    assert all(source.nodes[i] == successor.nodes[i] for i in range(1, 20))
    assert all(
        source.nodes[0][key] == successor.nodes[0][key]
        for key in source.nodes[0]
        if key not in ("EPI", "nu_f", "delta_nfr")
    )
    assert proposal.glyph is Glyph.NUL
    assert proposal.write_epi is (policy == 0)
    # These detached primary-stage images do not commit context or execute
    # Coherence/Silence from the supplied continuation word.


def test_shared_kernel_off_flag_preserves_admitted_form_instead_of_none(policies):
    source, enabled, on, _ = policies[0]
    _, disabled, off, _ = policies[1]
    factor = Q(resolve_operator_factors(None, Glyph.NUL)["NUL_scale"])
    assert on.requested_scale == off.requested_scale == float(factor)
    assert Q(on.vf_after) == Q(off.vf_after) == factor
    assert on.dnfr_after == off.dnfr_after == 0.0
    assert on.densification_factor == off.densification_factor == 1.0 / float(factor)
    assert on.binary64_inverse_product_residual == off.binary64_inverse_product_residual
    assert on.write_epi and on.clip_delta == 0.0
    assert Q(on.epi_after) == factor / 2
    assert not off.write_epi
    assert off.epi_before is off.raw_epi_after is off.epi_after is None
    assert off.effective_epi_scale is off.clip_delta is None
    assert disabled.nodes[0]["EPI"] == source.nodes[0]["EPI"] == 0.5
    assert enabled.nodes[0]["EPI"] != disabled.nodes[0]["EPI"]
    assert tuple(disabled.nodes[i]["EPI"] for i in disabled) == (0.5,) * 20
    assert tuple(enabled.nodes[i]["nu_f"] for i in enabled) == tuple(
        disabled.nodes[i]["nu_f"] for i in disabled
    )
    assert (source.graph["EPI_MIN"], source.graph["EPI_MAX"]) == (-1.0, 1.0)
    assert source.graph["CLIP_MODE"] == "hard"


def test_complete_refresh_distinguishes_raw_pressure_capacity_and_form_actions(
    policies,
):
    source = policies[0][0]
    enabled, disabled = policies[0][1], policies[1][1]
    before = cross_law._primitive_rows(source, Q(0))
    on = cross_law._primitive_rows(enabled, Q(0))
    off = cross_law._primitive_rows(disabled, Q(0))
    assert before[1] == on[1] == off[1]  # same bare sine currents
    assert on[3] == off[3]  # same initial refreshed form rates
    assert off[0] == off[2] == (Q(0),) * 20
    assert any(value != 0 for value in on[0])
    assert any(not value.contains(0) for value in on[4])
    assert any(not value.contains(0) for value in off[3])
    assert on[4] != off[4]
    for graph, independent in ((enabled, on), (disabled, off)):
        assert all(graph.nodes[i]["delta_nfr"] == 0.0 for i in graph)
        shared = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
        assert independent[0] == shared.form_gradient
        assert independent[2] == shared.phase_rate_numerators()
        for direct, admitted in zip(
            independent[3] + independent[4], shared.form_rates + shared.phase_rates
        ):
            assert (direct - admitted).contains(0)
        # The refreshed pressure is S/(pi*d), not the supplied raw stored zero.
        refreshed_pressure = independent[1][0] / (pi_interval() * graph.degree[0])
        assert not refreshed_pressure.contains(0)
    # An unchanged nonzero untargeted form row forces a common clock factor
    # to one, whereas the nonzero node-0 form row is scaled by f != 1.
    factor = Q(policies[1][2].requested_scale)
    assert not before[3][0].contains(0) and not before[3][2].contains(0)
    assert off[3][2] == before[3][2]
    assert (off[3][0] - factor * before[3][0]).contains(0)
    assert (off[3][0] - before[3][0]).lo > 0
    assert factor != 1


def test_actual_event_work_and_new_mean_leaves_follow_each_complete_policy(policies):
    source = policies[0][0]
    enabled, disabled = policies[0][1], policies[1][1]
    factor = Q(policies[0][2].requested_scale)
    amplitude = (1 - factor) / 2

    def storage(graph):
        return sum(
            (
                Q(data["weight"])
                * (Q(graph.nodes[i]["EPI"]) - Q(graph.nodes[j]["EPI"])) ** 2
                / 2
                for i, j, data in graph.edges(data=True)
            ),
            Q(0),
        )

    def means(graph):
        rho = tuple(Q(graph.degree[i]) / Q(graph.nodes[i]["nu_f"]) for i in graph)
        total = sum(rho)
        form_charge = sum(rho[i] * Q(graph.nodes[i]["EPI"]) for i in graph)
        phase_charge = sum(rho[i] * Q(graph.nodes[i]["theta"]) for i in graph)
        return (
            rho,
            total,
            form_charge,
            phase_charge,
            form_charge / total,
            phase_charge / total,
        )

    original, on, off = means(source), means(enabled), means(disabled)
    assert storage(disabled) - storage(source) == 0
    assert storage(enabled) - storage(source) == Q(27, 8) * amplitude**2 > 0
    assert all(
        source.nodes[i]["theta"]
        == enabled.nodes[i]["theta"]
        == disabled.nodes[i]["theta"]
        for i in source
    )  # no phase-potential contribution to either jump
    assert on[0] == off[0] and on[1] == off[1] == 83 + 7 / factor
    assert off[4] == Q(1, 2)
    assert on[4] == Q(1, 2) - on[0][0] * amplitude / on[1] < off[4]
    assert on[2] == original[2] == 45 < off[2]
    assert on[3] == off[3] == original[3] > 0
    assert on[5] == off[5] < original[5]
    for graph in (enabled, disabled):
        gradient = cross_law._primitive_rows(graph, Q(0))[0]
        assert sum(gradient) == 0
        # Reciprocal bare edges cancel the two sine terms, while each actual
        # weighted edge contributes opposite form gradients. Multiplication
        # by rho cancels the node mobility in both rows: new leaves persist.
        assert all(graph.has_edge(j, i) for i, j in graph.edges)


def test_capacity_only_retains_all_faces_while_enabled_crosses_at_the_same_time(
    policies,
):
    enabled, disabled = policies[0][1], policies[1][1]
    pi = pi_interval()
    remainder = 2 * HORIZON**2 / pi.lo**2
    assert remainder < Q(1, 2**30)
    enabled_rates = cross_law._primitive_rows(enabled, Q(0))[2]
    disabled_rates = cross_law._primitive_rows(disabled, Q(0))[2]
    assert disabled_rates == (Q(0),) * 20
    crossed = set()
    for edge in disabled.edges:
        initial = cross_law._gap(disabled, edge)
        assert pi.lo / 2 - initial.abs_max - remainder > Q(7, 2**30)
        assert initial.abs_max + remainder < pi.lo
        left, right = edge
        rate = I(enabled_rates[right] - enabled_rates[left]) / pi
        displacement = rate.abs_max * HORIZON + remainder
        assert initial.abs_max + displacement < pi.lo
        if edge in CRITICAL_EDGES:
            assert initial.lo + rate.lo * HORIZON - remainder - pi.hi / 2 > Q(15, 2**30)
            crossed.add(edge)
        else:
            assert pi.lo / 2 - initial.abs_max - displacement > 0
    assert disabled.number_of_edges() == 45 and crossed == CRITICAL_EDGES
    # Bounds hold throughout [0,T]. The critical endpoint lower bounds prove
    # enabled exit; the disabled branch has no initial linear phase term.


def test_disabled_successor_has_the_same_exact_full_transverse_sector(
    policies, full_tangent
):
    s, phases, current_jacobian, form_jacobian, injection = full_tangent
    enabled, disabled = policies[0][1], policies[1][1]
    assert tuple(enabled.edges(data="weight")) == tuple(disabled.edges(data="weight"))
    assert (
        tuple(sorted(disabled[8])) == tuple(sorted(disabled[9])) == transverse.NEIGHBORS
    )
    assert not disabled.has_edge(8, 9)
    assert all(
        disabled.nodes[8][key] == disabled.nodes[9][key]
        for key in ("EPI", "theta", "nu_f")
    )
    assert disabled.nodes[8]["nu_f"] == 1.0
    mobility = s.diag(
        *(s.Rational(disabled.nodes[i]["nu_f"]) / disabled.degree[i] for i in disabled)
    )
    upper = s.zeros(20).row_join(mobility * current_jacobian / s.pi)
    lower = (mobility * form_jacobian / s.pi).row_join(s.zeros(20))
    tangent = upper.col_join(lower).subs(phases[9], phases[8])
    stiffness = sum(s.cos(phases[8] - phases[j]) for j in transverse.NEIGHBORS)
    sector = s.Matrix(((0, -stiffness / (4 * s.pi)), (1 / s.pi, 0)))
    assert (tangent * injection - injection * sector).applyfunc(s.trigsimp) == s.zeros(
        40, 2
    )
    assert len(stiffness.free_symbols) == 5


def test_each_policy_readmits_the_whole_window_tangent_remainder_hypotheses(policies):
    pi = pi_interval()
    for _, graph, _, _ in policies:
        capacity = tuple(Q(graph.nodes[i]["nu_f"]) for i in graph)
        forms = tuple(Q(graph.nodes[i]["EPI"]) for i in graph)
        diameter = max(forms) - min(forms) + 2 * HORIZON / pi.lo
        assert diameter < Q(1, 8)
        assert all(0 < value <= 1 for value in capacity)
        for i in graph:
            velocity = (
                capacity[i]
                * sum(Q(data["weight"]) for data in graph[i].values())
                / graph.degree[i]
            )
            acceleration = (
                capacity[i]
                * sum(
                    Q(data["weight"]) * (capacity[i] + capacity[j])
                    for j, data in graph[i].items()
                )
                / graph.degree[i]
            )
            assert velocity <= 1 and acceleration <= 2
        # Four actual pair neighbors and the full-row bounds re-admit (44):
        # |h|<=4, |h'|<=1/pi, |h''|<=65/(4pi^2)<2 on each complete branch.
        assert len(graph[8]) == 4
        assert Q(65, 4) / pi.lo**2 < 2
    assert 1 / (1 - HORIZON**2 / (2 * pi.lo**2)) < 2
    assert Q(1, 18) + 5 * HORIZON / 81 < Q(1, 16)
    derivative_remainder = Q(17, 144) + HORIZON / 48
    assert derivative_remainder < Q(1, 8)
    assert derivative_remainder / 4 < Q(1, 32)


def test_identical_fixed_reading_has_a_strict_finite_enabled_over_disabled_gap(
    policies,
):
    enabled, disabled = policies[0][1], policies[1][1]
    pi = pi_interval()
    phase = tuple(Q(enabled.nodes[i]["theta"]) for i in enabled)
    assert phase == tuple(Q(disabled.nodes[i]["theta"]) for i in disabled)
    h0 = sum((cos(I(phase[8] - phase[j])) for j in transverse.NEIGHBORS), I(0))
    assert h0.lo > Q(2, 3)
    derivatives = []
    for graph in (enabled, disabled):
        rates = cross_law._primitive_rows(graph, Q(0))[2]
        derivatives.append(
            -sum(
                (
                    sin(I(phase[8] - phase[j])) * (rates[8] - rates[j]) / pi
                    for j in transverse.NEIGHBORS
                ),
                I(0),
            )
        )
    assert derivatives[0].lo > Q(205, 65536)
    assert derivatives[1] == I(0)
    cubic_lower = Q(2, 3) * Q(205, 65536) / (12 * 4**2)
    finite_lower = cubic_lower * HORIZON**3 - 2 * HORIZON**4 / 32
    assert finite_lower == Q(133, 9 * 2**63) > Q(1, 2**60)
    # Same h0, tangent (u,v)=(0,1), clock and T; both independent remainder
    # costs are retained. Zero disabled cubic coefficient does not compare its
    # higher-order response to no event, and does not make it stationary.
