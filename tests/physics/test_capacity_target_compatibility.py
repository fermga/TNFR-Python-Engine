"""Canonical capacity-supported profiles and one finite admitted P2 word.

Prepared coefficient controls are not simulated trajectories. The public
VAL/IL/UM/SHA word below has no physical flow or repeated-policy claim.
"""

from fractions import Fraction
import math

import networkx as nx
import pytest

from tnfr.config import inject_defaults
from tnfr.constants.aliases import (
    ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF,
)
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.operators.definitions import Coherence, Coupling, Expansion, Silence
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.physics.forced_support import (
    derive_forced_support_balance, observe_forced_support_target,
)
from tnfr.physics.forcing_realization import (
    capture_non_epi_forcing, decompose_non_epi_forcing,
)
from tnfr.validation import validate_sequence


F = Fraction


def _prepare(capacity, *, phase=None, edge_weights=None):
    """Declare initial data once; later runtime state writes use operators only."""
    graph = nx.path_graph(len(capacity))
    inject_defaults(graph)
    graph.graph.update(RANDOM_SEED=17, _t=0.0)
    phases = (0.0,) * len(capacity) if phase is None else phase
    weights = (1.0,) * len(graph.edges) if edge_weights is None else edge_weights
    for edge, weight in zip(graph.edges, weights, strict=True):
        graph.edges[edge]["weight"] = weight
    for node, nu, theta in zip(graph, capacity, phases, strict=True):
        graph.nodes[node].update({
            ALIAS_EPI[0]: 0.5, ALIAS_VF[0]: float(nu), ALIAS_THETA[0]: theta,
            ALIAS_DNFR[0]: 0.0, ALIAS_DEPI[0]: 0.0, ALIAS_SI[0]: 0.5,
            "glyph_history": [],
        })
    default_compute_delta_nfr(graph)
    return graph


def _reference(capture):
    return derive_forced_support_balance(
        capture.snapshot, epi_weight=capture.epi_weight, forcing=capture.forcing,
    )


def _observation(target, capture):
    return observe_forced_support_target(
        target, _reference(capture), capture.snapshot,
        forcing_components=decompose_non_epi_forcing(capture),
    )


def _ratio(capture):
    return dict(capture.normalized_weights)["vf"] / capture.epi_weight


@pytest.mark.parametrize(("capacity", "centered_profile"), (
    ((1, 2), (F(1, 3), F(-2, 3))),
    ((1, 2, 4), (F(7, 9), F(-2, 9), F(-20, 9))),
))
@pytest.mark.parametrize("edge_weight", (1.0, 2.0))
def test_captured_default_channels_give_the_hand_capacity_supported_profile(
    capacity, centered_profile, edge_weight,
):
    graph = _prepare(capacity, edge_weights=(edge_weight,) * (len(capacity) - 1))
    capture = capture_non_epi_forcing(graph)
    reference = _reference(capture)
    ratio = _ratio(capture)
    assert capture.phase_gradient == (0,) * len(capacity)
    assert dict(capture.normalized_weights)["topo"] == 0
    assert reference.mean_drift == reference.compatibility_residual == 0
    assert reference.relative_profile == tuple(ratio * value for value in centered_profile)
    components = dict(decompose_non_epi_forcing(capture))
    assert components["phase"] == components["topo"] == (0,) * len(capacity)
    assert components["vf"] == capture.forcing
    result = _observation(reference, capture)
    assert result.target_compatible
    assert result.compatibility_residual == result.profile_identity_residual == (0,) * len(capacity)
    assert result.channel_energy_identity_residual == 0


def test_uniform_capacity_increment_preserves_the_original_nonuniform_target():
    old_capture = capture_non_epi_forcing(_prepare((1, 2, 4)))
    current_capture = capture_non_epi_forcing(_prepare((2, 3, 5)))
    target = _reference(old_capture)
    result = _observation(target, current_capture)
    assert old_capture.normalized_weights == current_capture.normalized_weights
    assert result.reference.relative_profile != target.relative_profile
    assert result.target_compatible
    assert result.limiting_pattern.relative_error == (0, 0, 0)
    assert result.compatibility_residual == (0, 0, 0)
    assert result.metric_proportionality is None


def test_single_node_capacity_increment_has_the_hand_projected_target_mismatch():
    old_capture = capture_non_epi_forcing(_prepare((1, 2, 4)))
    current_capture = capture_non_epi_forcing(_prepare((2, 2, 4)))
    target = _reference(old_capture)
    result = _observation(target, current_capture)
    ratio = _ratio(current_capture)
    assert target.metric_weights == (1, 1, F(1, 4))
    assert result.limiting_pattern.relative_error == (
        -5 * ratio / 9, 4 * ratio / 9, 4 * ratio / 9,
    )
    assert not result.target_compatible
    assert any(result.compatibility_residual)
    assert result.profile_identity_residual == (0, 0, 0)


@pytest.fixture(scope="module")
def admitted_p2_word():
    graph = _prepare((1, 1))
    initial = capture_non_epi_forcing(graph)
    target = _reference(initial)
    operators = (Expansion(), Coherence(), Coupling(), Silence())
    context = {"initial_epi_nonzero": graph.nodes[0][ALIAS_EPI[0]] > 0.0}
    names = tuple(operator.name for operator in operators)
    assert validate_sequence(list(names), context=context).passed
    sequence = ValidatedSequence(operators, context=context)
    events = []
    for index, operator in enumerate(operators):
        before = capture_non_epi_forcing(graph)
        step = sequence.step(index)
        admission = validate_candidate(graph, 0, operator.glyph, sequence_context=step)
        assert admission.allowed, admission.violations
        operator(graph, 0, collect_metrics=True, sequence_context=step)
        raw = capture_non_epi_forcing(graph)
        actual_history = tuple(graph.nodes[0]["glyph_history"])
        assert actual_history == tuple(item.glyph.value for item in operators[:index + 1])
        default_compute_delta_nfr(graph)
        refreshed = capture_non_epi_forcing(graph)
        assert refreshed.stored_pressure_residual == (0, 0)
        events.append((before, raw, refreshed, admission))
    return graph, initial, target, tuple(events)


def test_actual_default_word_is_admitted_without_coefficient_or_topology_tuning(admitted_p2_word):
    graph, initial, _, events = admitted_p2_word
    defaults = canonical_glyph_factor_defaults()
    for name in ("VAL_scale", "IL_dnfr_factor", "UM_vf_sync", "SHA_vf_factor"):
        assert graph.graph["GLYPH_FACTORS"][name] == defaults[name]
    assert tuple(graph.nodes[0]["glyph_history"]) == ("VAL", "IL", "UM", "SHA")
    assert tuple(graph.nodes[1]["glyph_history"]) == ()
    assert graph.graph["_t"] == 0.0
    assert tuple(graph.edges) == ((0, 1),)
    for _, raw, refreshed, admission in events:
        assert admission.allowed
        assert raw.phase == refreshed.phase == (0, 0)
        assert refreshed.normalized_weights == initial.normalized_weights
        for name in ("epi", "capacity", "conductance"):
            assert getattr(raw.snapshot, name) == getattr(refreshed.snapshot, name)


def test_actual_val_breaks_compatibility_and_has_the_predicted_capacity_gap_floor(admitted_p2_word):
    _, initial, target, events = admitted_p2_word
    before, raw, refreshed, _ = events[0]
    result = _observation(target, refreshed)
    assert initial.snapshot.capacity == (1, 1)
    assert target.relative_profile == (0, 0)
    gap = refreshed.snapshot.capacity[0] - refreshed.snapshot.capacity[1]
    assert gap > 0
    assert raw.snapshot.epi[0] > before.snapshot.epi[0]
    assert raw.snapshot.epi[1] == before.snapshot.epi[1]
    ratio = _ratio(refreshed)
    vf_weight = dict(refreshed.normalized_weights)["vf"]
    assert not result.target_compatible
    assert result.limiting_pattern.relative_error == (-ratio * gap / 2, ratio * gap / 2)
    assert result.limiting_pattern.error_variance == ratio**2 * gap**2 / 4
    lifted = tuple(x + ratio * nu for x, nu in zip(
        refreshed.snapshot.epi, refreshed.snapshot.capacity, strict=True,
    ))
    metric = result.reference.metric_weights
    lifted_mean = sum(h * y for h, y in zip(metric, lifted, strict=True)) / sum(metric)
    assert result.state.relative_error == tuple(y - lifted_mean for y in lifted)
    amplitude = vf_weight * gap * (2 + gap) / 2
    assert result.compatibility_residual == (-amplitude, amplitude)


def test_actual_il_pressure_reduction_does_not_repair_the_refreshed_target(admitted_p2_word):
    _, _, target, events = admitted_p2_word
    before, raw, refreshed, _ = events[1]
    assert raw.snapshot.epi == refreshed.snapshot.epi == before.snapshot.epi
    assert raw.snapshot.capacity == before.snapshot.capacity
    assert raw.forcing == refreshed.forcing == before.forcing
    assert abs(raw.snapshot.stored_pressure[0]) < abs(before.snapshot.stored_pressure[0])
    assert raw.stored_pressure_residual[0] != 0
    assert refreshed.snapshot.stored_pressure == before.snapshot.stored_pressure
    old_result, new_result = _observation(target, before), _observation(target, refreshed)
    assert not new_result.target_compatible
    assert new_result.compatibility_residual == old_result.compatibility_residual
    assert new_result.limiting_pattern == old_result.limiting_pattern


def test_actual_um_reduces_but_does_not_close_the_observed_capacity_gap(admitted_p2_word):
    _, _, target, events = admitted_p2_word
    before, raw, refreshed, _ = events[2]
    before_gap = before.snapshot.capacity[0] - before.snapshot.capacity[1]
    after_gap = refreshed.snapshot.capacity[0] - refreshed.snapshot.capacity[1]
    assert 0 < after_gap < before_gap
    assert raw.snapshot.capacity[1] == before.snapshot.capacity[1] == 1
    assert raw.snapshot.epi == refreshed.snapshot.epi == before.snapshot.epi
    assert raw.snapshot.conductance == before.snapshot.conductance
    old_result, new_result = _observation(target, before), _observation(target, refreshed)
    assert not new_result.target_compatible
    ratio = _ratio(refreshed)
    assert new_result.limiting_pattern.relative_error == (-ratio * after_gap / 2, ratio * after_gap / 2)
    assert new_result.limiting_pattern.error_variance == ratio**2 * after_gap**2 / 4
    assert new_result.limiting_pattern.error_variance < old_result.limiting_pattern.error_variance
    # This ratio uses actual represented endpoints, not an ideal repeated map.
    assert (new_result.limiting_pattern.error_variance
            / old_result.limiting_pattern.error_variance) == (after_gap / before_gap)**2


def test_actual_sha_closes_the_word_without_an_epi_recovery_claim(admitted_p2_word):
    _, _, _, events = admitted_p2_word
    before, raw, refreshed, _ = events[3]
    assert raw.snapshot.epi == refreshed.snapshot.epi == before.snapshot.epi
    assert raw.snapshot.capacity[0] < before.snapshot.capacity[0]
    assert raw.snapshot.capacity[1] == before.snapshot.capacity[1]


def test_heterogeneous_conductance_invalidates_the_unweighted_capacity_profile_formula():
    capture = capture_non_epi_forcing(_prepare((1, 2, 4), edge_weights=(1.0, 2.0)))
    reference = _reference(capture)
    ratio = _ratio(capture)
    vf_weight = dict(capture.normalized_weights)["vf"]
    assert capture.snapshot.capacity_gradient == (1, F(1, 2), -2)
    assert reference.metric_weights == (1, F(3, 2), F(1, 2))
    assert reference.compatibility_residual == -3 * vf_weight / 2
    assert reference.mean_drift == -vf_weight / 2
    # The current H-mean capacity is 2; this tempting formula is false here.
    assert reference.relative_profile != (ratio, 0, -2 * ratio)


def test_nonzero_phase_adds_a_profile_not_explained_by_uniform_capacity():
    original = capture_non_epi_forcing(_prepare((1, 1)))
    capture = capture_non_epi_forcing(_prepare((1, 1), phase=(0.0, math.pi / 2)))
    reference = _reference(capture)
    phase_weight = dict(capture.normalized_weights)["phase"]
    assert capture.phase_gradient == (F(1, 2), F(-1, 2))
    assert capture.snapshot.capacity_gradient == (0, 0)
    assert reference.relative_profile == (
        phase_weight / (4 * capture.epi_weight), -phase_weight / (4 * capture.epi_weight),
    )
    assert reference.relative_profile != (0, 0)
    result = _observation(_reference(original), capture)
    assert not result.target_compatible
    assert result.compatibility_residual == (phase_weight / 2, -phase_weight / 2)
