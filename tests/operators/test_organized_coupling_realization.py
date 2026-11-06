"""Detached single-stage Coupling support audit, without event execution.

The acquired source has an analytic uncertainty certificate. The represented
carrier fixtures below are separate constructed controls, never reconstructed
or authenticated acquired endpoints. Every observed numerical phase difference
is compared with the exact difference of its represented operands.
"""

import math
from copy import deepcopy
from fractions import Fraction as Q
from functools import lru_cache

import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics import integrators, runtime
from tnfr.errors import TNFRValueError
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.metrics import phase_compatibility
from tnfr.operators import _coupling_stage_kernel as coupling
from tnfr.operators import _phase_gate as gate
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import (
    GlyphFactorValidationError,
    resolve_runtime_operator_factors,
)
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics.relational_sine_regional import _branches
from tnfr.types import Glyph
from tnfr.utils import angle_diff

ERROR = Q(1, 10**10)
DONOR_ERROR = Q(1, 10**70)
ANGLE_ERROR = Q(1, 2**40)


@pytest.fixture(scope="module", autouse=True)
def no_events_trajectories_or_retained_response():
    def forbidden(*args, **kwargs):
        pytest.fail("contact realization audit cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@lru_cache(maxsize=None)
def _principal(first, second):
    # Subtract exact stored operands, not their rounded binary64 difference.
    _, principal, _, _ = _branches((I(Q(first) - Q(second)),))
    assert principal is not None, "the finite control needs a certified regular branch"
    return principal[0]


@pytest.fixture
def angle_calls(monkeypatch, record_property):
    calls = []

    def checked(first, second):
        value = angle_diff(first, second)
        exact = _principal(first, second)
        error_bound = (I(Q(value)) - exact).abs_max
        assert error_bound < ANGLE_ERROR
        calls.append((first, second, value, error_bound))
        return value

    for module in (coupling, gate, phase_compatibility):
        monkeypatch.setattr(module, "angle_diff", checked)
    yield calls
    record_property("certified_angle_calls", len(calls))
    record_property(
        "maximum_gap_error_bound", str(max((item[3] for item in calls), default=Q(0)))
    )
    record_property("allowed_gap_error", str(ANGLE_ERROR))


def _graph(phases, edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(phases)))
    graph.add_edges_from(edges, weight=1.0, length=1.0)
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        UM_FUNCTIONAL_LINKS=True,
        UM_CANDIDATE_COUNT=0,
        UM_COMPAT_THRESHOLD=0.0,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
    )
    for node, phase in enumerate(phases):
        graph.nodes[node].update(
            {
                ALIAS_THETA[0]: phase,
                ALIAS_EPI[0]: 1.0,
                ALIAS_SI[0]: 1.0,
                ALIAS_VF[0]: 1.0,
                ALIAS_DNFR[0]: 0.0,
            }
        )
    return graph


def _factors(graph):
    return resolve_runtime_operator_factors(
        graph.graph["GLYPH_FACTORS"], Glyph.UM, graph.graph
    )


def _proposal(graph, targets=None, factors=None):
    before = deepcopy(graph)
    try:
        return coupling.propose_coupling_stage(
            graph,
            tuple(graph) if targets is None else targets,
            _factors(graph) if factors is None else factors,
            resolved_seed=17,
            node_offsets={node: node for node in graph},
        )
    finally:
        # Includes graph-owned policies/caches, nodal fields and edge attributes.
        assert nx.utils.graphs_equal(graph, before)
        assert graph.__dict__.keys() == before.__dict__.keys()


def _phases(graph, proposal=None):
    phases = {node: graph.nodes[node][ALIAS_THETA[0]] for node in graph}
    if proposal is not None:
        for update in proposal.node_updates:
            if update.theta_after is not None:
                phases[update.node] = update.theta_after
            assert update.vf_after is None and update.dnfr_after is None
    return phases


def _gate_decision(first, second, limit):
    exact = abs(_principal(first, second))
    actual = abs(angle_diff(first, second)) <= limit
    # The fixed controls have strict margins; unresolved comparisons cannot pass.
    if actual:
        assert exact.hi < Q(limit)
    else:
        assert exact.lo > Q(limit)
    return actual


def _rim_identities(phases, geometry, expected=1):
    for cycle in geometry.outer_cycles:
        raw = tuple(
            I(Q(phases[right]) - Q(phases[left]))
            for left, right in zip(cycle, cycle[1:] + cycle[:1])
        )
        assert sum((gap.lo for gap in raw), Q(0)) == 0
        turns, principal, winding, margin = _branches(raw)
        assert principal is not None and winding == expected and margin > 0
        assert winding == -sum(turns)


@pytest.fixture(scope="module")
def geometry():
    return external._derive_organized_external_interaction()


@pytest.fixture(scope="module")
def source():
    return external._assess_organized_external_interaction(
        form_error_bounds=(ERROR,) * 6, phase_error_bounds=(ERROR,) * 6
    )


@pytest.mark.parametrize("origin", (0, 1))
def test_acquired_source_family_has_nine_strictly_forbidden_initial_spokes(
    source, geometry, origin, record_property
):
    """All original directions are bounded, without inventing an endpoint."""
    assert all(source.source_admission_flags)
    pi = pi_interval()
    host = source.host_acquisition
    residual = (
        host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    h0 = host.primitive_entry.endpoint_phase_bounds[0] + I(-residual, residual)
    assert h0 == source.outer_one_host_phase_offset_bounds[0]
    assert (h0 + 8 / pi).abs_max + DONOR_ERROR < Q(1, 40000)
    beta = I(Q(3, 5)) if origin == 0 else 2 * pi / 5 - Q(3, 5)
    # The prior decoder's whole +/-0.001 origin envelopes remain harmless.
    gap = beta + I(-Q(1, 1000), Q(1, 1000)) + I(-DONOR_ERROR, DONOR_ERROR) - h0
    assert gap.lo > pi.hi and gap.hi < Q(13, 4)
    assert (2 * pi - gap).lo > Q(303, 100) > 3 * pi.hi / 4
    assert Q(1, 385) - Q(1, 1000) - Q(1, 40000) > 0
    assert Q(23, 35) + Q(400, 157) + Q(1, 1000) + Q(1, 40000) < Q(13, 4)
    assert tuple((j, 45 + j) for j in range(9)) == tuple(
        (left, right) for left, right, _ in geometry.contact_edges if left < 9
    )
    record_property("source_residual_upper_bound", str(Q(1, 40000)))
    record_property("all_source_forbidden_spokes", 9)
    record_property("source_spoke_magnitude_lower_bound", str(Q(303, 100)))
    # The carrier cancels at matching slots. These are exact source-family
    # exclusions conditional on a gate/error bridge, not nine runtime calls.


@pytest.mark.parametrize("origin", (0, 1))
@pytest.mark.parametrize("zero_push", (False, True))
def test_constructed_carrier_cannot_acquire_all_spokes_in_one_proposal(
    geometry, angle_calls, origin, zero_push, record_property
):
    beta = 0.6 if origin == 0 else 2 * math.pi / 5 - 0.6
    phases = tuple(
        2 * math.pi * j / 9 + (-4 * math.pi / 5 + 2 * math.pi * c / 5)
        for c in range(5)
        for j in range(9)
    ) + tuple(2 * math.pi * j / 9 + beta for j in range(9))
    graph = _graph(
        phases,
        tuple((i, j) for i, j, _ in geometry.internal_edges + geometry.outer_edges),
    )
    required = {frozenset((i, j)) for i, j, _ in geometry.contact_edges}
    assert len(graph) == 54 and graph.number_of_edges() == 99 and len(required) == 45
    assert all(not graph.has_edge(*edge) for edge in required)
    factors = _factors(graph)
    assert graph.graph.get("UM_BIDIRECTIONAL", True) is True
    if zero_push:
        # The pure proposal admits its closed numerical boundary; the registered
        # factor contract rejects zero. No registered-stage claim is made here.
        factors = {**factors, "UM_theta_push": 0.0}
    before = _phases(graph)
    _rim_identities(before, geometry)
    stage = _proposal(graph, factors=factors)
    after = _phases(graph, stage)
    _rim_identities(after, geometry)
    proposed = {frozenset((edge.left, edge.right)) for edge in stage.edges}
    target_map = {proposal.node: proposal for proposal in stage.target_proposals}
    _, limit = gate.resolve_u3_phase_limits(graph.graph, operator_code="UM")
    assert Q(limit) + ANGLE_ERROR < 3 * pi_interval().lo / 4
    assert len(required - proposed) >= 9
    for node, target in target_map.items():
        assert target.effective_phase_limit == limit
        expected = {
            other
            for other in graph
            if other != node
            and not graph.has_edge(node, other)
            and _gate_decision(before[node], before[other], limit)
        }
        assert {candidate.target for candidate in target.link_candidates} == expected
        for other in target.compatible_neighbors:
            assert _gate_decision(before[node], before[other], limit)
            assert _gate_decision(after[node], after[other], limit)
        for candidate in target.link_candidates:
            pair = frozenset((node, candidate.target))
            final_admitted = _gate_decision(after[node], after[candidate.target], limit)
            assert (pair in proposed) == final_admitted
    for state in (before, after):
        for j in range(9):
            spokes = tuple(
                _principal(state[45 + j], state[9 * c + j]) for c in range(5)
            )
            assert max(abs(gap).lo for gap in spokes) > 3 * pi_interval().hi / 4
    # Both endpoint orientations were proposed, so no missing candidate can
    # be recovered by an unchecked reverse proposal for the undirected edge.
    initially_forbidden = {
        edge
        for edge in required
        if all(
            all(
                candidate.target != right
                for candidate in target_map[left].link_candidates
            )
            for left, right in (tuple(edge), tuple(reversed(tuple(edge))))
        )
    }
    assert len(initially_forbidden) >= 9 and initially_forbidden.isdisjoint(proposed)
    record_property("proposed_required_spokes", len(required & proposed))
    record_property("missing_required_spokes", len(required - proposed))
    record_property("initially_excluded_required_spokes", len(initially_forbidden))
    record_property("strict_acute_winding_one_old_rims_before_and_after", 9)
    record_property("resolved_phase_factor", factors["UM_theta_push"])
    assert angle_calls


def test_winding_zero_positive_control_can_add_spokes_but_not_weak_weights(
    geometry, angle_calls, record_property
):
    graph = _graph(
        (0.0,) * 54,
        tuple((i, j) for i, j, _ in geometry.internal_edges + geometry.outer_edges),
    )
    _rim_identities(_phases(graph), geometry, expected=0)
    stage = _proposal(graph)
    _rim_identities(_phases(graph, stage), geometry, expected=0)
    proposed = {frozenset((edge.left, edge.right)): edge.weight for edge in stage.edges}
    assert all(proposed[frozenset((i, j))] == 1.0 for i, j, _ in geometry.contact_edges)
    assert len(stage.edges) == 54 * 53 // 2 - 99
    assert all(weight != Q(1, 100000) for weight in proposed.values())
    record_property("proposed_required_spokes", 45)
    record_property("missing_required_spokes", 0)
    record_property("old_rim_winding", 0)
    record_property("new_edge_conductance", 1)
    assert angle_calls


def test_final_new_edge_can_be_omitted_after_snapshot_candidate_admission(angle_calls):
    graph = _graph((0.0, 1.4, 0.0, -1.4), ((0, 1), (2, 3)))
    graph.graph["UM_BIDIRECTIONAL"] = False
    stage = _proposal(
        graph, targets=(0, 2), factors={**_factors(graph), "UM_theta_push": 1.0}
    )
    candidates = {
        proposal.node: {item.target for item in proposal.link_candidates}
        for proposal in stage.target_proposals
    }
    assert 2 in candidates[0] and 0 in candidates[2]
    assert not any({edge.left, edge.right} == {0, 2} for edge in stage.edges)
    final = _phases(graph, stage)
    _, limit = gate.resolve_u3_phase_limits(graph.graph, operator_code="UM")
    assert not _gate_decision(final[0], final[2], limit)
    assert not _gate_decision(final[2], final[0], limit)
    assert angle_calls


def test_final_old_edge_gate_rejects_whole_stage_without_any_write(angle_calls):
    graph = _graph(
        (0.0, 1.56, 3.12, -1.56, -1.56, 4.68, 4.68),
        ((0, 1), (1, 2), (0, 3), (0, 4), (2, 5), (2, 6)),
    )
    phases = _phases(graph)
    _, limit = gate.resolve_u3_phase_limits(graph.graph, operator_code="UM")
    assert all(_gate_decision(phases[i], phases[j], limit) for i, j in graph.edges)
    with pytest.raises(
        TNFRValueError, match="merged phase would violate the final U3 gate"
    ):
        _proposal(graph, targets=(0, 2))
    assert angle_calls


def test_zero_phase_push_is_only_a_pure_kernel_boundary():
    graph = _graph((0.0, 0.1), ((0, 1),))
    raw = {**graph.graph["GLYPH_FACTORS"], "UM_theta_push": 0.0}
    with pytest.raises(GlyphFactorValidationError):
        resolve_runtime_operator_factors(raw, Glyph.UM, graph.graph)


@pytest.mark.parametrize("wider", (math.pi, 100.0))
def test_operator_specific_limit_cannot_weaken_the_canonical_hard_gate(wider):
    graph = _graph((0.0, 0.1), ((0, 1),))
    hard, baseline = gate.resolve_u3_phase_limits(graph.graph, operator_code="UM")
    graph.graph["UM_MAX_PHASE_DIFF"] = wider
    assert gate.resolve_u3_phase_limits(graph.graph, operator_code="UM") == (
        hard,
        baseline,
    )
    assert Q(hard) + ANGLE_ERROR < 3 * pi_interval().lo / 4


def test_above_ceiling_hard_gate_rejects_before_proposal_writes():
    graph = _graph((0.0, 0.1), ((0, 1),))
    hard, _ = gate.resolve_u3_phase_limits(graph.graph, operator_code="UM")
    graph.graph["DELTA_PHI_MAX"] = math.nextafter(hard, math.inf)
    with pytest.raises(TNFRValueError, match="DELTA_PHI_MAX"):
        _proposal(graph)


def test_ambiguous_true_pi_branch_is_unavailable_not_selected():
    offsets, principal, winding, margin = _branches((pi_interval(),))
    assert (offsets, principal, winding, margin) == (None, None, None, None)
