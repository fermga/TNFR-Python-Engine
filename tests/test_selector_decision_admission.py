"""Configured selector decisions use admitted observations and explicit batches."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.constants import inject_defaults
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_SI
from tnfr.dynamics import selectors
from tnfr.selector import _selector_thresholds
from tnfr.types import Glyph


def _graph():
    graph = nx.path_graph(2)
    inject_defaults(graph)
    graph.graph.update(
        SELECTOR_THRESHOLDS={"si_hi": 0.8, "si_lo": 0.2, "dnfr_hi": 0.5},
        GLYPH_SELECTOR_MARGIN=0.0,
        GRAMMAR_CANON={"enabled": False},
        AL_MAX_LAG=100,
        EN_MAX_LAG=100,
    )
    for node, pressure in enumerate((0.1, 1.0)):
        graph.nodes[node].update({ALIAS_SI[0]: 0.5, ALIAS_DNFR[0]: pressure})
    return graph


@pytest.mark.parametrize(
    "selector_type,prepared,value",
    [
        (selectors.DefaultGlyphSelector, False, True),
        (selectors.DefaultGlyphSelector, True, "0.9"),
        (selectors.ParametricGlyphSelector, False, float("nan")),
        (selectors.ParametricGlyphSelector, True, Fraction(1, 2**2000)),
    ],
)
def test_invalid_authoritative_sense_cannot_become_an_operator_decision(
    selector_type, prepared, value
):
    graph = _graph()
    graph.nodes[0][ALIAS_SI[0]] = value
    graph.nodes[0][ALIAS_SI[1]] = 0.9
    selector = selector_type()
    with pytest.raises((TypeError, ValueError)):
        if prepared:
            selector.prepare(graph, list(graph))
        selector(graph, 0)


@pytest.mark.parametrize("prepared", [False, True])
def test_each_new_decision_or_prepare_refreshes_pressure_normalization(prepared):
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    if prepared:
        selector.prepare(graph, list(graph))
    assert selector(graph, 0) == "RA"  # |p0| / max|p| = 0.1
    graph.nodes[1][ALIAS_DNFR[0]] = 0.1
    if prepared:
        selector.prepare(graph, list(graph))
    assert selector(graph, 0) == "NAV"  # The refreshed ratio is 1.


def test_scalar_and_array_preselection_share_raw_admission(monkeypatch):
    graph = _graph()
    graph.nodes[1][ALIAS_SI[0]] = "0.9"
    monkeypatch.setattr(selectors, "np", None)
    with pytest.raises(TypeError):
        selectors.DefaultGlyphSelector().prepare(graph, list(graph))


def test_invalid_late_metric_stops_batch_before_any_operator_or_lag_write(monkeypatch):
    graph = _graph()
    graph.nodes[1][ALIAS_SI[0]] = True
    before = deepcopy(dict(graph.nodes(data=True)))
    history = {}

    def unexpected(*args, **kwargs):
        pytest.fail("invalid selection input reached operator execution")

    monkeypatch.setattr(selectors, "apply_glyph", unexpected)
    selector = selectors.DefaultGlyphSelector()
    with pytest.raises(TypeError):
        selectors._apply_glyphs(graph, selector, history)
    assert dict(graph.nodes(data=True)) == before
    assert history == {}
    assert selector._preselection is None


@pytest.mark.parametrize("fail", [False, True])
def test_batch_snapshot_is_released_after_application_or_failure(monkeypatch, fail):
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    selected = []

    def apply(graph, node, glyph, **kwargs):
        selected.append((node, glyph))
        for data in graph.nodes.values():
            data[ALIAS_SI[0]] = 0.9
        if fail:
            raise RuntimeError("application failed")

    monkeypatch.setattr(selectors, "apply_glyph", apply)
    if fail:
        with pytest.raises(RuntimeError, match="application failed"):
            selectors._apply_glyphs(graph, selector, {})
    else:
        selectors._apply_glyphs(graph, selector, {})
        assert selected == [(0, "RA"), (1, "NAV")]
    assert selector._preselection is None
    assert selector(graph, 0) == "IL"  # New call reads current Si, not the batch.


def test_failed_manual_prepare_does_not_retain_an_older_successful_snapshot():
    graph = _graph()
    selector = selectors.DefaultGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.nodes[0][ALIAS_SI[0]] = True
    with pytest.raises(TypeError):
        selector.prepare(graph, list(graph))
    assert selector._preselection is None


def test_score_weights_follow_current_configuration_between_unprepared_calls():
    graph = _graph()
    weights = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    graph.graph["SELECTOR_WEIGHTS"] = weights
    score = selectors._compute_selector_score
    assert score(graph, graph.nodes[0], 0.8, 0.2, 0.9, "RA") == 0.8
    weights.update(w_si=0.0, w_accel=1.0)
    assert score(graph, graph.nodes[0], 0.8, 0.2, 0.9, "RA") == pytest.approx(0.1)


def test_prepared_score_weights_are_one_detached_batch_policy():
    graph = _graph()
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    selector = selectors.ParametricGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.graph["SELECTOR_WEIGHTS"]["w_si"] = 0.0
    assert selector._preselection.weights == {
        "w_si": 1.0,
        "w_dnfr": 0.0,
        "w_accel": 0.0,
    }


def test_explicitly_disabled_hysteresis_remains_disabled_in_prepared_batch(monkeypatch):
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = None
    selector = selectors.ParametricGlyphSelector()
    selector.prepare(graph, list(graph))
    graph.graph["GLYPH_SELECTOR_MARGIN"] = 1.0
    consumed = []
    real_hysteresis = selectors._apply_selector_hysteresis

    def observe(*args):
        consumed.append(args[-1])
        return real_hysteresis(*args)

    monkeypatch.setattr(selectors, "_apply_selector_hysteresis", observe)
    selector(graph, 0)
    assert consumed == [None]


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_threshold_cannot_be_coerced_to_a_zero_policy(sign):
    graph = _graph()
    graph.graph["SELECTOR_THRESHOLDS"]["si_hi"] = sign * Fraction(1, 2**2000)
    with pytest.raises(ValueError, match="si_hi.*underflows"):
        _selector_thresholds(graph)


@pytest.mark.parametrize("prepared", [False, True])
def test_invalid_score_policy_cannot_be_hidden_by_hysteresis(prepared):
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = 1.0
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": "bad", "w_dnfr": -5.0, "w_accel": 1.0}
    graph.nodes[0]["glyph_history"] = ["IL"]
    selector = selectors.ParametricGlyphSelector()
    with pytest.raises((TypeError, ValueError), match="SELECTOR_WEIGHTS"):
        if prepared:
            selector.prepare(graph, list(graph))
        selector(graph, 0)


def test_worker_preselection_matches_array_decisions_for_one_admitted_snapshot(
    monkeypatch,
):
    graph = _graph()
    reference = selectors.DefaultGlyphSelector()
    reference.prepare(graph, list(graph))
    expected = [reference(graph, node) for node in graph]
    graph.graph["GLYPH_SELECTOR_N_JOBS"] = 2
    monkeypatch.setattr(selectors, "np", None)
    parallel = selectors.DefaultGlyphSelector()
    parallel.prepare(graph, list(graph))
    assert [parallel(graph, node) for node in graph] == expected


def test_invalid_hysteresis_margin_cannot_silently_disable_the_policy():
    graph = _graph()
    graph.graph["GLYPH_SELECTOR_MARGIN"] = float("nan")
    with pytest.raises(ValueError, match="GLYPH_SELECTOR_MARGIN"):
        selectors.ParametricGlyphSelector()(graph, 0)


@pytest.mark.parametrize("recorded", ["IL", "Coherence", "coherence", "Glyph.IL"])
def test_repetition_feedback_validates_consumed_samples_and_compares_without_overflow(
    recorded,
):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = [recorded]
    graph.graph["SELECTOR_WEIGHTS"] = {"w_si": 1.0, "w_dnfr": 0.0, "w_accel": 0.0}
    graph.graph["history"] = {"sense_sigma_mag": [1e308, -1e308]}
    score = selectors._compute_selector_score
    assert score(graph, graph.nodes[0], 0.9, 0.2, 0.1, "IL") == pytest.approx(0.85)
    graph.graph["history"]["sense_sigma_mag"][-1] = True
    with pytest.raises(TypeError):
        score(graph, graph.nodes[0], 0.9, 0.2, 0.1, "IL")


def test_malformed_selector_configuration_does_not_select_the_default_controller():
    graph = _graph()
    graph.graph["glyph_selector"] = "parametric"
    with pytest.raises(TypeError, match="glyph_selector"):
        selectors._apply_selector(graph)


def test_missing_observations_keep_the_declared_selector_baseline():
    graph = _graph()
    for data in graph.nodes.values():
        data.clear()
    assert selectors.DefaultGlyphSelector()(graph, 0) == "RA"


@pytest.mark.parametrize("key", ["GLYPH_HYSTERESIS_WINDOW", "AL_MAX_LAG", "EN_MAX_LAG"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", None, -1])
def test_batch_count_configuration_rejects_before_lag_or_operator_writes(
    monkeypatch, key, value
):
    graph = _graph()
    graph.graph[key] = value
    history = {}
    before = deepcopy(dict(graph.nodes(data=True)))
    monkeypatch.setattr(
        selectors, "apply_glyph", lambda *a, **k: pytest.fail("applied")
    )
    with pytest.raises((TypeError, ValueError)):
        selectors._apply_glyphs(graph, lambda *_: "IL", history)
    assert history == {}
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("key", ["since_AL", "since_EN"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1.5, "2", None, -1])
def test_invalid_later_lag_counter_does_not_increment_an_earlier_counter(
    monkeypatch, key, value
):
    graph = _graph()
    history = {"since_AL": {0: 3, 1: 2}, "since_EN": {0: 4, 1: 1}}
    history[key][1] = value
    before = deepcopy(history)
    monkeypatch.setattr(
        selectors, "apply_glyph", lambda *a, **k: pytest.fail("applied")
    )
    with pytest.raises((TypeError, ValueError)):
        selectors._apply_glyphs(graph, lambda *_: "IL", history)
    assert history == before


@pytest.mark.parametrize(
    "flag,enabled", [("false", False), ("off", False), ("true", True)]
)
def test_incremental_grammar_switch_uses_shared_boolean_parsing(
    monkeypatch, flag, enabled
):
    graph = _graph()
    graph.graph["GRAMMAR_CANON"] = {"enabled": flag}
    checked = []
    monkeypatch.setattr(
        selectors, "enforce_canonical_grammar", lambda G, n, g: checked.append(n) or g
    )
    monkeypatch.setattr(selectors, "apply_glyph", lambda *a, **k: None)
    selectors._apply_glyphs(graph, lambda *_: "IL", {})
    assert checked == (list(graph) if enabled else [])


@pytest.mark.parametrize("config", [None, [], {"enabled": "sometimes"}])
def test_invalid_incremental_grammar_configuration_does_not_write_lags(config):
    graph = _graph()
    graph.graph["GRAMMAR_CANON"] = config
    history = {}
    with pytest.raises((TypeError, ValueError)):
        selectors._apply_glyphs(graph, lambda *_: "IL", history)
    assert history == {}


@pytest.mark.parametrize("clock", [True, float("nan"), Fraction(1, 2**2000)])
def test_invalid_selector_clock_cannot_enter_abstention_evidence(monkeypatch, clock):
    graph = _graph()
    graph.graph["_t"] = clock
    history = {}
    monkeypatch.setattr(
        selectors, "apply_glyph", lambda *a, **k: pytest.fail("applied")
    )
    with pytest.raises((TypeError, ValueError), match="selector event clock"):
        selectors._apply_glyphs(graph, lambda *_: "ZHIR", history)
    assert history == {}


def _live_graph():
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(AL_MAX_LAG=100, EN_MAX_LAG=100, GRAMMAR_CANON={"enabled": False})
    graph.nodes[0].update(EPI=0.3, nu_f=1.0, theta=0.0, delta_nfr=0.4)
    return graph


def test_forced_al_replaced_by_il_does_not_reset_the_unrealized_emission_lag():
    from tnfr.operators.grammar_debt import U2_DEBT_KEY
    from tnfr.operators.grammar_dynamics import U2_DEBT_CAPACITY

    graph = _live_graph()
    graph.graph.update(AL_MAX_LAG=0, GRAMMAR_CANON={"enabled": True})
    graph.nodes[0][U2_DEBT_KEY] = U2_DEBT_CAPACITY + 1
    history = {}
    selectors._apply_glyphs(graph, lambda *_: "RA", history)
    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]
    assert graph.nodes[0][U2_DEBT_KEY] == U2_DEBT_CAPACITY
    assert history["since_AL"][0] == 1
    assert history["since_EN"][0] == 1


def test_read_only_chooser_applies_grammar_to_forced_candidates_without_incrementing():
    from tnfr.operators.grammar_debt import U2_DEBT_KEY
    from tnfr.operators.grammar_dynamics import U2_DEBT_CAPACITY
    from tnfr.types import Glyph

    graph = _live_graph()
    graph.nodes[0][U2_DEBT_KEY] = U2_DEBT_CAPACITY + 1
    al_lags, en_lags = {0: 1}, {0: 1}
    before = deepcopy(dict(graph.nodes(data=True)))
    choose = selectors._choose_glyph
    assert choose(graph, 0, lambda *_: "RA", True, al_lags, en_lags, 0, 100) == "IL"
    assert (
        choose(graph, 0, lambda *_: "RA", False, al_lags, en_lags, 0, 100) is Glyph.AL
    )
    assert al_lags == en_lags == {0: 1}
    assert dict(graph.nodes(data=True)) == before


def test_strict_native_late_rejection_preserves_the_already_applied_prefix():
    from tnfr.operators.grammar_types import StructuralGrammarError

    graph = _live_graph()
    graph.nodes[0]["EPI"] = 0.0
    graph.add_node(1, **deepcopy(graph.nodes[0]))
    graph.graph.update(GRAMMAR_CANON={"enabled": True}, GRAMMAR_REJECTION_MODE="raise")
    history = {}
    with pytest.raises(StructuralGrammarError):
        selectors._apply_glyphs(graph, lambda G, n: "AL" if n == 0 else "SHA", history)
    assert list(graph.nodes[0]["glyph_history"]) == ["AL"]
    assert "glyph_history" not in graph.nodes[1]
    assert history["since_AL"] == {0: 0, 1: 1}


@pytest.mark.parametrize("requested", ["AL", "EN"])
def test_successfully_applied_forced_glyph_resets_only_its_actual_lag(requested):
    graph = _live_graph()
    graph.graph[f"{requested}_MAX_LAG"] = 0
    history = {"since_AL": {0: 3}, "since_EN": {0: 4}}
    selectors._apply_glyphs(graph, lambda *_: "RA", history)
    assert list(graph.nodes[0]["glyph_history"]) == [requested]
    assert history[f"since_{requested}"][0] == 0
    other = "EN" if requested == "AL" else "AL"
    assert history[f"since_{other}"][0] == (5 if other == "EN" else 4)


@pytest.mark.parametrize("use_canon", [False, True])
@pytest.mark.parametrize(
    "code,requested",
    [
        ("AL", Glyph.AL),
        ("AL", "Emission"),
        ("AL", "emission"),
        ("AL", "al"),
        ("AL", "Glyph.AL"),
        ("EN", Glyph.EN),
        ("EN", "Reception"),
        ("EN", "reception"),
        ("EN", "en"),
        ("EN", "Glyph.EN"),
    ],
)
def test_native_aliases_preserve_applied_state_and_actual_lag_reset(
    use_canon, code, requested
):
    graph = _live_graph()
    graph.graph.update(
        GRAMMAR_CANON={"enabled": use_canon}, GRAMMAR_REJECTION_MODE="raise"
    )
    reference = deepcopy(graph)
    history = {"since_AL": {0: 3}, "since_EN": {0: 4}}
    reference_history = deepcopy(history)

    selectors._apply_glyphs(reference, lambda *_: code, reference_history)
    selectors._apply_glyphs(graph, lambda *_: requested, history)

    assert dict(graph.nodes[0]) == dict(reference.nodes[0])
    assert history == reference_history
    assert list(graph.nodes[0]["glyph_history"]) == [code]
    assert history[f"since_{code}"][0] == 0
    other = "EN" if code == "AL" else "AL"
    assert history[f"since_{other}"][0] == (5 if other == "EN" else 4)


@pytest.mark.parametrize("use_canon", [False, True])
@pytest.mark.parametrize(
    "requested", [Glyph.ZHIR, "Mutation", "mutation", "zhir", "Glyph.ZHIR"]
)
@pytest.mark.parametrize("evidence", ["missing", "below_threshold", "crossed"])
def test_native_mutation_aliases_preserve_evidence_gate_and_abstention(
    use_canon, requested, evidence
):
    graph = _live_graph()
    graph.graph.update(
        _t=1.0,
        GRAMMAR_CANON={"enabled": use_canon},
        GRAMMAR_REJECTION_MODE="raise",
    )
    graph.nodes[0].update(EPI=0.2, glyph_history=["IL", "OZ"])
    if evidence != "missing":
        earlier_epi = 0.19 if evidence == "below_threshold" else 0.0
        graph.nodes[0]["epi_time_history"] = [(0.0, earlier_epi), (1.0, 0.2)]
    reference = deepcopy(graph)
    history, reference_history = {}, {}

    selectors._apply_glyphs(reference, lambda *_: "ZHIR", reference_history)
    selectors._apply_glyphs(graph, lambda *_: requested, history)

    assert dict(graph.nodes[0]) == dict(reference.nodes[0])
    assert history == reference_history
    if evidence == "crossed":
        assert list(graph.nodes[0]["glyph_history"])[-1] == "ZHIR"
        assert graph.nodes[0]["theta"] != 0.0
        assert "mutation_abstentions" not in history
    else:
        assert list(graph.nodes[0]["glyph_history"])[-1] == "IL"
        assert graph.nodes[0]["theta"] == 0.0
        abstention = history["mutation_abstentions"][0]
        assert abstention["requested_glyph"] == "ZHIR"
        assert abstention["applied_glyph"] == "IL"
        assert abstention["reason"] == (
            "missing_history"
            if evidence == "missing"
            else "observed_threshold_not_crossed"
        )


@pytest.mark.parametrize("use_canon", [False, True])
def test_invalid_later_proposal_keeps_the_executed_prefix_and_rejection_owner(
    use_canon,
):
    from tnfr.errors import TNFRValueError
    from tnfr.operators.grammar_types import StructuralGrammarError

    graph = _live_graph()
    graph.add_node(1, **deepcopy(graph.nodes[0]))
    graph.graph.update(
        GRAMMAR_CANON={"enabled": use_canon}, GRAMMAR_REJECTION_MODE="raise"
    )
    history = {}
    exception = StructuralGrammarError if use_canon else TNFRValueError
    with pytest.raises(exception):
        selectors._apply_glyphs(
            graph, lambda G, n: "Emission" if n == 0 else "unknown", history
        )
    assert list(graph.nodes[0]["glyph_history"]) == ["AL"]
    assert "glyph_history" not in graph.nodes[1]
    assert history["since_AL"] == {0: 0, 1: 1}
    assert history["since_EN"] == {0: 1, 1: 1}


@pytest.mark.parametrize("channel", ["EPI", "nu_f", "delta_nfr"])
@pytest.mark.parametrize("value", [True, None, "bad", float("nan")])
def test_mutation_selector_cannot_abstain_using_a_coerced_authoritative_channel(
    channel, value
):
    from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
    from tnfr.operators.preconditions import OperatorPreconditionError

    aliases = {"EPI": ALIAS_EPI, "nu_f": ALIAS_VF, "delta_nfr": ALIAS_DNFR}[channel]
    graph = _live_graph()
    graph.nodes[0][aliases[1]] = graph.nodes[0][channel]
    graph.nodes[0][aliases[0]] = value
    history = {}
    with pytest.raises(OperatorPreconditionError):
        selectors._apply_glyphs(graph, lambda *_: "ZHIR", history)
    assert "glyph_history" not in graph.nodes[0]
    assert "mutation_abstentions" not in history


@pytest.mark.parametrize("serialized", [False, True])
def test_shared_mutation_certificate_preserves_signed_uniform_bepi(serialized):
    from tnfr.mathematics import BEPIElement
    from tnfr.operators._mutation_gate import mutation_trigger_certificate
    from tnfr.types import serialize_bepi

    graph = _live_graph()
    form = BEPIElement((-0.3, -0.3), (-0.3, -0.3), (0.0, 1.0))
    graph.nodes[0]["EPI"] = serialize_bepi(form) if serialized else form
    certificate = mutation_trigger_certificate(graph.nodes[0], graph.graph)
    assert certificate.current_epi == -0.3
    assert certificate.reason == "missing_history"
    history = {}
    selectors._apply_glyphs(graph, lambda *_: "ZHIR", history)
    assert list(graph.nodes[0]["glyph_history"]) == ["IL"]
    assert history["mutation_abstentions"][0]["reason"] == "missing_history"


@pytest.mark.parametrize(
    "field,value",
    [
        ("glyph_history", "AL"),
        ("glyph_history", None),
        ("_grammar_u2_debt", True),
        ("_grammar_prior_coherence", "false"),
    ],
)
def test_invalid_later_grammar_state_is_rejected_before_native_batch_writes(
    field, value
):
    graph = _graph()
    graph.nodes[1][field] = value
    before = deepcopy(dict(graph.nodes(data=True)))
    history = {}
    with pytest.raises((TypeError, ValueError)):
        selectors._apply_glyphs(graph, lambda *_: "AL", history)
    assert history == {}
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("validate_context", [False, True])
@pytest.mark.parametrize(
    "window", [True, np.bool_(False), 1.5, Fraction(3, 2), float("nan")]
)
def test_soft_filter_rejects_original_invalid_window_without_coercion(
    monkeypatch, validate_context, window
):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["AL", "OZ"]
    graph.graph["GRAMMAR"] = {
        "window": window,
        "avoid_repeats": ["OZ"],
        "fallbacks": {"OZ": "RA"},
    }
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1" if validate_context else "0")
    before = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((TypeError, ValueError), match="window"):
        selectors._soft_grammar_prefilter(graph, 0, "OZ")
    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("validate_context", [False, True])
@pytest.mark.parametrize("window,expected", [(0, "OZ"), (np.int64(2), "RA")])
def test_soft_filter_shared_integer_window_preserves_existing_policy(
    monkeypatch, validate_context, window, expected
):
    graph = _graph()
    graph.nodes[0]["glyph_history"] = ["AL", "OZ"]
    graph.graph["GRAMMAR"] = {
        "window": window,
        "avoid_repeats": ["OZ"],
        "fallbacks": {"OZ": "RA"},
        "force_dnfr": 0.6,
        "force_accel": 0.6,
    }
    monkeypatch.setenv("TNFR_GRAMMAR_VALIDATE", "1" if validate_context else "0")
    assert selectors._soft_grammar_prefilter(graph, 0, "OZ") == expected
