"""Four finite controls separating proposals, admission and ordered return.

The protocol in artifacts/operator-viability-20261103 fixes seven fresh
two-node sources. Existing shared stages supply every actual operation.
Reference-domain failure is not NFR disappearance or global inviability.
"""

import math

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.config import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SI,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.selectors import DefaultGlyphSelector
from tnfr.errors import TNFRValueError
from tnfr.operators import Coherence, Contraction, Coupling, Resonance
from tnfr.operators.grammar_debt import read_grammar_state
from tnfr.operators.grammar_dynamics import _check_violations
from tnfr.operators.grammar_types import glyph_function_name
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    execute_coupling_stage,
    execute_neighbor_stage,
    execute_pointwise_stage,
)
from tnfr.operators.registry import get_operator_class
from tnfr.types import require_finite_real_scalar_epi
from tnfr.utils import angle_diff
from tnfr.utils._structural_signature import structural_proof_signature

C, D = "receiver", "participant"
_ABS = 2e-14


def _raw(graph, node, aliases):
    return get_attr(graph.nodes[node], aliases, strict=True, conv=lambda value: value)


def _state(graph, node):
    return (
        require_finite_real_scalar_epi(_raw(graph, node, ALIAS_EPI)),
        _raw(graph, node, ALIAS_VF),
        _raw(graph, node, ALIAS_DNFR),
        _raw(graph, node, ALIAS_THETA),
    )


def _checkpoint(graph):
    return structural_proof_signature(
        (
            graph.graph,
            tuple(graph.nodes(data=True)),
            tuple(graph.edges(data=True)),
            hasattr(graph, "_last_operator_applied"),
            getattr(graph, "_last_operator_applied", None),
        ),
        opaque_references=(graph,),
    )


def _assert_stage(result, glyph, targets):
    assert result.glyph == glyph
    assert result.schedule == TWO_PHASE_JACOBI
    assert result.nodes_processed == len(targets)


def _pointwise(graph, operator, targets=(C,), **options):
    result = execute_pointwise_stage(
        graph, operator, targets, require_exact_operator=True, **options
    )
    _assert_stage(result, operator.glyph.value, targets)


def _resonate(graph, operator=None):
    operator = Resonance() if operator is None else operator
    result = execute_neighbor_stage(graph, operator, (C,))
    _assert_stage(result, "RA", (C,))


def _couple(graph):
    result = execute_coupling_stage(graph, Coupling(), (C,), preserve_relations=True)
    _assert_stage(result, "UM", (C,))


def _context(graph, node, expected):
    state = read_grammar_state(graph.nodes[node])
    window = graph.graph["GLYPH_HYSTERESIS_WINDOW"]
    assert window > 0
    assert state.history == tuple(expected)[-window:]
    assert state.operator_step == len(expected)
    assert state.u2_debt == 0
    assert state.prior_coherence is True
    assert _raw(graph, node, ALIAS_SOURCE_GLYPH) == expected[-1]


def _boundary(graph):
    assert tuple(graph) == (C, D)
    assert tuple(graph.edges) == ((C, D),)
    assert graph.edges[C, D] == {"weight": 1.0, "origin": "supplied"}
    assert graph.graph["_t"] == 0.0
    for node in (C, D):
        assert _raw(graph, node, ALIAS_EPI_KIND) == "form"
        assert "epi_time_history" not in graph.nodes[node]
        assert math.isfinite(_raw(graph, node, ALIAS_D2EPI))


def _in_domain(graph, *, prefix=False):
    """The declared finite-test obligation, not an engine admission oracle."""
    _boundary(graph)
    x, capacity, pressure, phase = _state(graph, C)
    u, donor_capacity, donor_pressure, donor_phase = _state(graph, D)
    x_lo, x_hi = (1 / 8, 3 / 4) if prefix else (1 / 4, 1 / 2)
    c_lo, c_hi = (1 / 8, 3 / 2) if prefix else (1 / 4, 1)
    pressure_bound = 3 / 8 if prefix else 3 / 16
    return (
        x_lo <= x <= x_hi
        and c_lo <= capacity <= c_hi
        and abs(pressure) <= pressure_bound
        and abs(angle_diff(phase, donor_phase)) <= 1 / 4
        and 3 / 8 <= u <= 3 / 4
        and 1 / 2 <= donor_capacity <= 1
        and abs(donor_pressure) <= 1 / 8
        and all(
            read_grammar_state(graph.nodes[node]).u2_debt == 0
            and read_grammar_state(graph.nodes[node]).prior_coherence
            for node in (C, D)
        )
    )


def _source(
    *,
    form=1 / 4,
    capacity=1 / 2,
    pressure=1 / 64,
    donor_form=3 / 4,
    donor_capacity=1.0,
    donor_pressure=1 / 8,
):
    graph = nx.Graph()
    for node, x, c, p, phase, si in (
        (C, form, capacity, pressure, 1 / 4, 3 / 8),
        (D, donor_form, donor_capacity, donor_pressure, 0.0, 3 / 4),
    ):
        graph.add_node(
            node,
            **{
                ALIAS_EPI[0]: x,
                ALIAS_VF[0]: c,
                ALIAS_DNFR[0]: 2 * p,
                ALIAS_THETA[0]: phase,
                ALIAS_SI[0]: si,
                ALIAS_D2EPI[0]: 0.0,  # Supplied diagnostic, not sampled curvature.
                ALIAS_EPI_KIND[0]: "form",
            },
        )
    graph.add_edge(C, D, weight=1.0, origin="supplied")
    inject_defaults(graph)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=79,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        IL_PRECONDITIONS={"warn_zero_dnfr": False, "warn_isolated": False},
        EDGE_AWARE_ENABLED=True,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        UM_FUNCTIONAL_LINKS=False,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
    )
    factors = graph.graph["GLYPH_FACTORS"]
    factors.pop("NUL_densification_factor", None)
    factors.update(
        RA_epi_diff=0.5,
        RA_vf_amplification=0.125,
        RA_phase_coupling=0.5,
        NUL_scale=0.5,
        UM_theta_push=0.5,
        UM_vf_sync=0.25,
        UM_dnfr_reduction=0.25,
        IL_dnfr_factor=0.5,
    )
    _pointwise(graph, Coherence(), (C, D), phase_locking_coefficient=0.0)
    assert _state(graph, C) == (form, capacity, pressure, 1 / 4)
    assert _state(graph, D) == (
        donor_form,
        donor_capacity,
        donor_pressure,
        0.0,
    )
    _context(graph, C, ("IL",))
    _context(graph, D, ("IL",))
    _boundary(graph)
    return graph


def test_equal_default_proposal_inputs_do_not_identify_domain_preservation():
    projections, observations, outcomes = [], [], {}
    for capacity in (1 / 2, 1.0):
        graph = _source(capacity=capacity)
        assert _in_domain(graph)
        own_context = read_grammar_state(graph.nodes[C])
        projections.append(
            (
                _raw(graph, C, ALIAS_SI),
                _raw(graph, C, ALIAS_DNFR),
                _raw(graph, C, ALIAS_D2EPI),
                own_context,
                _state(graph, D),
            )
        )
        selector = DefaultGlyphSelector()
        details = {}
        try:
            selector.prepare(graph, tuple(graph))
            selected = selector.select(graph, C, _selection_details=details)
        finally:
            selector.clear()
        assert selected == "RA"
        assert details["normalized_metrics"] == (3 / 8, 1 / 8, 0.0)
        observations.append(details)
        assert read_grammar_state(graph.nodes[C]) == own_context
        # An explicit selector-to-stage composition, not a native batch.
        operator = get_operator_class(glyph_function_name(selected))()
        assert type(operator) is Resonance
        _resonate(graph, operator)
        assert _state(graph, C)[:3] == (1 / 2, 9 * capacity / 8, 1 / 64)
        assert _state(graph, C)[3] == pytest.approx(1 / 8, rel=0, abs=_ABS)
        assert _state(graph, D) == (3 / 4, 1.0, 1 / 8, 0.0)
        _context(graph, C, ("IL", "RA"))
        _context(graph, D, ("IL",))
        outcomes[capacity] = _in_domain(graph)
    assert projections[0] == projections[1]
    assert observations[0] == observations[1]
    assert outcomes == {1 / 2: True, 1.0: False}


def test_grammar_allows_ra_but_real_il_is_needed_for_pressure_admission():
    graph = _source(pressure=3 / 4)
    allowed, violations = _check_violations(graph, C, "RA")
    assert allowed is True
    assert violations == []
    assert not _in_domain(graph)
    before = _checkpoint(graph)
    with pytest.raises(TNFRValueError, match="controlled dissonance"):
        _resonate(graph)
    assert _checkpoint(graph) == before
    _context(graph, C, ("IL",))
    _pointwise(graph, Coherence(), phase_locking_coefficient=0.5)
    assert _state(graph, C)[:3] == (1 / 4, 1 / 2, 3 / 8)
    assert _state(graph, C)[3] == pytest.approx(1 / 8, rel=0, abs=_ABS)
    _context(graph, C, ("IL", "IL"))
    _resonate(graph)
    assert _state(graph, C)[:3] == (1 / 2, 9 / 16, 3 / 8)
    assert _state(graph, C)[3] == pytest.approx(1 / 16, rel=0, abs=_ABS)
    assert _state(graph, D) == (3 / 4, 1.0, 1 / 8, 0.0)
    _context(graph, C, ("IL", "IL", "RA"))
    _context(graph, D, ("IL",))
    assert not _in_domain(graph)  # Repaired admission has not restored S.


@pytest.mark.parametrize(
    "donor_form,donor_capacity,forward_form,reverse_form,final_capacity",
    ((3 / 8, 1 / 2, 3 / 16, 9 / 32, 43 / 128), (3 / 4, 1.0, 9 / 32, 15 / 32, 59 / 128)),
)
def test_ordered_admitted_words_have_distinct_return_domains(
    donor_form, donor_capacity, forward_form, reverse_form, final_capacity
):
    results, contexts, memberships, prefixes = {}, {}, {}, {}
    words = {
        "forward": ("RA", "NUL", "UM", "IL"),
        "reverse": ("NUL", "RA", "UM", "IL"),
    }
    for label, word in words.items():
        graph = _source(
            form=3 / 8,
            pressure=3 / 16,
            donor_form=donor_form,
            donor_capacity=donor_capacity,
            donor_pressure=0.0,
        )
        assert _in_domain(graph)
        rows = []
        for glyph in word:
            if glyph == "RA":
                _resonate(graph)
            elif glyph == "NUL":
                _pointwise(graph, Contraction())
            elif glyph == "UM":
                _couple(graph)
            else:
                assert glyph == "IL"
                _pointwise(graph, Coherence(), phase_locking_coefficient=0.5)
            assert _in_domain(graph, prefix=True)
            rows.append((_state(graph, C), _in_domain(graph)))
        prefixes[label] = rows
        results[label] = _state(graph, C)
        memberships[label] = _in_domain(graph)
        contexts[label] = read_grammar_state(graph.nodes[C])
        _context(graph, C, ("IL", *word))
        _context(graph, D, ("IL",))
        assert _state(graph, D)[:3] == (donor_form, donor_capacity, 0.0)
        assert _state(graph, D)[3] == pytest.approx(1 / 32, rel=0, abs=_ABS)
        assert results[label][1] == final_capacity
        assert results[label][2] == pytest.approx(
            (3 / 16) * (3 / 4 + 1 / (64 * math.pi)), rel=0, abs=_ABS
        )
        assert results[label][3] == pytest.approx(1 / 16, rel=0, abs=_ABS)

    assert results["forward"][0] == forward_form
    assert results["reverse"][0] == reverse_form
    assert reverse_form - forward_form == donor_form / 4
    assert results["forward"][1:] == results["reverse"][1:]
    assert contexts["forward"].history != contexts["reverse"].history
    assert contexts["forward"].operator_step == contexts["reverse"].operator_step == 5
    assert memberships == {"forward": donor_form == 3 / 4, "reverse": True}
    # NUL-first crosses the lower return-form boundary before an admitted RA.
    assert prefixes["reverse"][0][0][:3] == (3 / 16, 1 / 4, 3 / 8)
    assert prefixes["reverse"][0][1] is False
    if donor_form == 3 / 4:
        assert prefixes["forward"][0][0][0] == 9 / 16
        assert prefixes["forward"][0][1] is False
        assert memberships["forward"] is True
