"""Finite capacity budgets and grammar controls without operator execution."""

import math
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.errors import TNFRValueError
from tnfr.operators import factor_contracts, network_stage, word_execution
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.operators.definitions_base import Operator
from tnfr.operators.grammar_debt import (
    advance_debt,
    advance_prior_coherence,
    read_grammar_state,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.preconditions import OperatorPreconditionError, validate_silence
from tnfr.physics._organized_source_reuse_capacity import (
    _assess_organized_source_reuse_capacity as assess,
)


@pytest.fixture(scope="module", autouse=True)
def no_event_or_flow():
    def forbidden(*args, **kwargs):
        pytest.fail("A detached capacity budget must not execute an event or flow")

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
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def default_report():
    return assess(minimum_capacity=0.01, maximum_calls=256)


def test_default_exact_capacity_limit_is_56_with_inclusive_floor(default_report):
    report = default_report
    factor = report.silence_capacity_factor
    floor = Q(0.01)
    # Independent power inequalities, not the observer's loop or verdict.
    assert factor**55 >= floor > factor**56
    assert report.exact.status == "floor_blocked"
    assert report.exact.last_admissible_call == 56
    assert report.exact.first_blocked_call == 57
    assert report.exact.last_accepted_pre_capacity == factor**55
    assert report.exact.first_blocked_pre_capacity == factor**56


def test_represented_budget_matches_sequential_hardware_products(default_report):
    report = default_report
    factor = float(report.silence_capacity_factor)
    capacity = 1.0
    independent_path = [Q(capacity)]
    while capacity >= 0.01:
        capacity *= factor
        independent_path.append(Q(capacity))
    assert report.represented_capacities == tuple(independent_path)
    assert report.represented.status == "floor_blocked"
    assert report.represented.last_admissible_call == 56
    assert report.represented.first_blocked_call == 57
    assert report.represented.first_blocked_pre_capacity == Q(capacity)
    assert (
        report.exact_capacity_at_represented_count == report.silence_capacity_factor**56
    )
    assert report.represented_minus_exact_at_represented_count != 0
    # Standard product bound is independent of the exact rounding algorithm;
    # all intermediate products in this control are normal binary64 values.
    u = Q(1, 2**53)
    gamma = 56 * u / (1 - 56 * u)
    assert abs(report.represented_minus_exact_at_represented_count) <= (
        gamma * report.exact_capacity_at_represented_count
    )


@pytest.mark.parametrize("initial,floor,count", [(1.0, 0.125, 4), (0.25, 0.25, 1)])
def test_exact_dyadic_floor_is_inclusive(initial, floor, count):
    report = assess(
        initial_capacity=initial,
        minimum_capacity=floor,
        silence_capacity_factor=0.5,
        maximum_calls=16,
    )
    for branch in (report.exact, report.represented):
        assert branch.last_admissible_call == count
        assert branch.last_accepted_pre_capacity == Q(floor)
        assert branch.first_blocked_call == count + 1
        assert branch.first_blocked_pre_capacity == Q(floor) / 2
    assert report.represented_minus_exact_at_represented_count == 0


def test_sequential_rounding_can_change_the_floor_call_count():
    # 0.1**2 as exact represented parameters is less than RN(0.1 * 0.1).
    floor = 0.1 * 0.1
    report = assess(
        minimum_capacity=floor, silence_capacity_factor=0.1, maximum_calls=16
    )
    assert Q(0.1) ** 2 < Q(floor)
    assert report.exact.last_admissible_call == 2
    assert report.represented.last_admissible_call == 3
    assert report.represented.last_accepted_pre_capacity == Q(floor)


def test_initial_capacity_below_floor_has_no_accepted_prefix():
    report = assess(initial_capacity=0.125, minimum_capacity=0.25, maximum_calls=1)
    for branch in (report.exact, report.represented):
        assert branch.status == "floor_blocked"
        assert branch.last_admissible_call == 0
        assert branch.first_blocked_call == 1
        assert branch.first_blocked_pre_capacity == Q(1, 8)
        assert branch.last_accepted_pre_capacity is None
    assert report.represented_capacities == (Q(1, 8),)


def test_explicit_budget_does_not_become_a_terminal_admission_count():
    report = assess(
        minimum_capacity=0.125, silence_capacity_factor=0.5, maximum_calls=3
    )
    for branch in (report.exact, report.represented):
        assert branch.status == "budget_exhausted"
        assert branch.accepted_calls == 3
        assert branch.current_capacity == Q(1, 8)
        assert branch.first_blocked_call is None
        assert branch.last_admissible_call is None
    resolved = assess(
        minimum_capacity=0.125, silence_capacity_factor=0.5, maximum_calls=4
    )
    assert resolved.exact.first_blocked_call == 5
    assert resolved.represented.first_blocked_call == 5


def test_factor_near_one_returns_a_bounded_unresolved_prefix():
    report = assess(
        minimum_capacity=0.01,
        silence_capacity_factor=math.nextafter(1.0, 0.0),
        maximum_calls=7,
    )
    for branch in (report.exact, report.represented):
        assert branch.status == "budget_exhausted"
        assert branch.accepted_calls == 7
        assert branch.last_admissible_call is None
    assert len(report.represented_capacities) == 8


def test_positive_product_underflow_is_rejected_before_counting_the_call():
    tiny = math.ulp(0.0)
    report = assess(
        initial_capacity=tiny,
        minimum_capacity=tiny,
        silence_capacity_factor=0.5,
        maximum_calls=8,
    )
    assert report.exact.last_admissible_call == 1
    assert report.exact.current_capacity == Q(1, 2**1075)
    branch = report.represented
    assert branch.status == "product_underflow"
    assert branch.accepted_calls == branch.last_admissible_call == 0
    assert branch.first_blocked_call == 1
    assert branch.first_blocked_pre_capacity == Q(tiny)
    assert branch.blocked_rounded_product == 0
    assert report.represented_capacities == (Q(tiny),)


def test_subnormal_rounding_stagnation_is_not_a_floor_or_unlimited_certificate():
    tiny = math.ulp(0.0)
    report = assess(
        initial_capacity=tiny,
        minimum_capacity=tiny,
        silence_capacity_factor=0.75,
        maximum_calls=8,
    )
    assert report.exact.last_admissible_call == 1
    assert report.represented.status == "represented_stagnation"
    assert report.represented.accepted_calls == 1
    assert report.represented.last_admissible_call is None
    assert report.represented.first_blocked_call is None
    assert report.represented_capacities == (Q(tiny), Q(tiny))
    assert report.represented_minus_exact_at_represented_count == Q(tiny) / 4


def test_exact_subnormal_products_do_not_trigger_underflow_or_stagnation():
    report = assess(
        initial_capacity=math.ldexp(1.0, -1022),
        minimum_capacity=math.ldexp(1.0, -1024),
        silence_capacity_factor=0.5,
        maximum_calls=8,
    )
    for branch in (report.exact, report.represented):
        assert branch.status == "floor_blocked"
        assert branch.last_admissible_call == 3
        assert branch.current_capacity == Q(1, 2**1025)
    assert report.represented_minus_exact_at_represented_count == 0


def test_registry_default_is_resolved_live_without_a_duplicate_constant(monkeypatch):
    defaults = factor_contracts.canonical_glyph_factor_defaults()
    defaults["SHA_vf_factor"] = 0.5
    monkeypatch.setattr(
        factor_contracts, "canonical_glyph_factor_defaults", lambda: dict(defaults)
    )
    report = assess(minimum_capacity=0.125, maximum_calls=16)
    assert report.factor_source == "canonical_factor_registry"
    assert report.silence_capacity_factor == Q(1, 2)
    assert report.exact.last_admissible_call == 4


@pytest.mark.parametrize(
    "field", ["initial_capacity", "minimum_capacity", "silence_capacity_factor"]
)
@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), None, "0.5", float("nan"), float("inf"), -0.1, 0.0],
)
def test_invalid_physical_scalars_reject_before_arithmetic(field, value):
    if field == "silence_capacity_factor" and value is None:
        # None is the documented live-registry selector, not a physical value.
        report = assess(
            minimum_capacity=0.01, maximum_calls=16, silence_capacity_factor=None
        )
        assert report.factor_source == "canonical_factor_registry"
        return
    args = dict(minimum_capacity=0.01, maximum_calls=16)
    args[field] = value
    with pytest.raises((TypeError, ValueError)):
        assess(**args)


@pytest.mark.parametrize("value", [1.0, 1.1, Q(1) + Q(1, 2**1000)])
def test_factor_domain_does_not_round_an_invalid_source_into_admission(value):
    with pytest.raises(ValueError):
        assess(
            minimum_capacity=0.01,
            maximum_calls=16,
            silence_capacity_factor=value,
        )


@pytest.mark.parametrize("field", ["initial_capacity", "minimum_capacity"])
@pytest.mark.parametrize("value", [Q(1, 2**1075), -Q(1, 2**1075), Q(2) ** 1024])
def test_unrepresentable_source_is_not_zero_or_an_infinite_budget(field, value):
    args = dict(minimum_capacity=0.01, maximum_calls=16)
    args[field] = value
    with pytest.raises(ValueError):
        assess(**args)


@pytest.mark.parametrize("value", [True, 3.0, -1, 0, 4097, float("inf"), "16"])
def test_work_budget_is_an_explicit_bounded_integer(value):
    with pytest.raises((TypeError, ValueError)):
        assess(minimum_capacity=0.01, maximum_calls=value)


def test_integral_budget_materializes_before_arithmetic():
    report = assess(minimum_capacity=0.01, maximum_calls=np.int64(2))
    assert type(report.maximum_calls) is int
    assert report.represented.status == "budget_exhausted"


def test_56_repeated_contextual_words_pass_static_grammar_without_execution():
    sequence = ValidatedSequence(
        tuple(op for _ in range(56) for op in (Reception(), Coherence(), Silence())),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    assert sequence.names == ("reception", "coherence", "silence") * 56
    # This finite-word check has no graph, capacity, live pressure or U6 trace.
    with pytest.raises(TNFRValueError, match="Invalid canonical sequence"):
        ValidatedSequence(
            (Reception(), Coherence(), Silence()),
            context={"initial_epi_nonzero": False, "compatibility_profile": "core"},
        )


def test_live_sha_to_en_context_preserves_history_and_lifetime_coherence():
    history = ("AL", "IL") + ("EN", "IL", "SHA") * 56
    graph = nx.Graph()
    graph.add_node(
        0,
        EPI=0.0,
        glyph_history=history,
        _grammar_u2_debt=0,
        _grammar_prior_coherence=True,
        _operator_step=len(history),
        latent=True,
    )
    before = deepcopy(dict(graph.nodes[0]))
    result = validate_candidate(graph, 0, "EN")
    state = read_grammar_state(graph.nodes[0])
    assert result.allowed
    assert state.history == history
    assert state.u2_debt == 0
    assert state.prior_coherence
    assert state.operator_step == len(history)
    assert dict(graph.nodes[0]) == before
    debt, prior = 0, False
    for glyph in history:
        debt = advance_debt(debt, glyph)
        prior = advance_prior_coherence(prior, glyph)
    assert debt == 0 and prior


def test_sha_floor_remains_a_separate_read_only_live_gate(default_report):
    graph = nx.Graph()
    graph.graph["SHA_MIN_VF"] = 0.01
    graph.add_node(0, nu_f=0.01)
    validate_silence(graph, 0)
    graph.nodes[0]["nu_f"] = float(
        default_report.represented.first_blocked_pre_capacity
    )
    with pytest.raises(OperatorPreconditionError):
        validate_silence(graph, 0)
