"""Declared flags cannot become verified structural evidence."""

import networkx as nx
import pytest

from tnfr.operators import Coherence, Emission, Silence
from tnfr.operators.grammar_observations import observe_grammar


def _graph():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=1.0, nu_f=1.0, theta=0.0)
    return graph


@pytest.mark.parametrize(
    "word",
    [
        ["emission", "coherence", "silence"],
        ["AL", "IL", "SHA"],
        [Emission(), Coherence(), Silence()],
    ],
)
def test_word_observation_shares_one_core_for_names_glyphs_and_instances(word):
    observation = observe_grammar(_graph(), 0, word)
    assert observation.sequence_valid
    assert observation.as_dict()["word_validation_scope"] == "canonical_word_policy"


def test_caller_flags_remain_declared_and_cannot_supply_structural_proof():
    report = observe_grammar(
        _graph(),
        0,
        ["IL", "SHA"],
        contract_satisfied=True,
        u6_checked=True,
    ).as_dict()
    assert report["contract_satisfied"] is True
    assert report["u6_checked"] is True
    assert report["contract_evidence_source"] == "caller_declaration"
    assert report["u6_evidence_source"] == "caller_declaration"
    assert not report["verified_contract_postconditions"]
    assert not report["verified_u6"]
    assert report["structural_evidence"] is None


def test_raw_dictionary_is_not_execution_evidence():
    with pytest.raises(TypeError):
        observe_grammar(
            _graph(),
            0,
            ["IL", "SHA"],
            execution_evidence={"exact_energy_gain_upper_bound": 0},
        )
