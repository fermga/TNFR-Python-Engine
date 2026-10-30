"""Independent high-precision checks of the detached default-UM admission.

Only retained THOL primitives are read. No UM proposal, operator, trajectory,
or earlier producer is permitted; the calculation has no observed UM response.
The high-precision oracle checks rational enclosures, not implementation error.
"""

from fractions import Fraction as Q
from math import pi
from pathlib import Path

import mpmath
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import (
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_SI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.constants.canonical import (
    COUPLING_GENTLE,
    COUPLING_MODERATE,
    UM_COMPAT_THRESHOLD,
    UM_THETA_PUSH,
)
from tnfr.dynamics import integrators, relational
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions import Coupling, SelfOrganization
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research.artifact_io import file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor
from tnfr.research.maintained_parent_um_work import predict_parent_um_work

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"
PRIOR = ROOT / "docs/assets/local_capacity_history"
PARENT, CHILD = 2, "2_sub_0"


def _mp(mp, value):
    exact = Q(value)
    return mp.mpf(exact.numerator) / exact.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _row_contains(mp, intervals, values):
    assert len(intervals) == len(values)
    for interval, value in zip(intervals, values, strict=True):
        _contains(mp, interval, value)


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("analytical UM evidence must not execute an operator or flow")

    with pytest.MonkeyPatch.context() as patch:
        for owner, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (maintained_parent_thol_execution, "execute_network_operator_stage"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
            (Coupling, "_execute"),
            (SelfOrganization, "_prepare_execution"),
            (SelfOrganization, "_prepare_bifurcation"),
        ):
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def study():
    paths = tuple(
        sorted(
            path
            for directory in (EVIDENCE, PRIOR)
            for path in directory.iterdir()
            if path.is_file()
        )
    )
    receipts = tuple(file_receipt(path) for path in paths)
    source = load_thol_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(source.graph)
    report = predict_parent_um_work(source.graph)
    yield source, report
    assert _signature(source.graph) == signature
    assert receipts == tuple(file_receipt(path) for path in paths)


@pytest.fixture(scope="module")
def oracle(study):
    """Evaluate declared formulas independently in a 90-digit context."""
    source, _ = study
    graph = source.graph
    mp = mpmath.mp.clone()
    mp.dps = 90
    nodes = tuple(graph)
    rank = {node: index for index, node in enumerate(nodes)}

    def values(alias, default=None):
        return tuple(
            _mp(mp, get_attr(graph.nodes[node], alias, default)) for node in nodes
        )

    epi, phase, capacity, pressure, sense = (
        values(alias, 0) if alias == ALIAS_SI else values(alias)
        for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR, ALIAS_SI)
    )

    def wrap(value):
        return mp.atan2(mp.sin(value), mp.cos(value))

    limit = _mp(mp, pi / 2)
    neighbors = tuple(
        rank[node]
        for node in graph.neighbors(PARENT)
        if abs(wrap(phase[rank[node]] - phase[PARENT])) <= limit
    )
    resultant = mp.fsum(mp.exp(1j * phase[index]) for index in (PARENT, *neighbors))
    consensus = mp.arg(resultant)
    after = list(phase)
    for index in (PARENT, *neighbors):
        after[index] += _mp(mp, UM_THETA_PUSH) * wrap(consensus - phase[index])
    capacity_after = list(capacity)
    capacity_after[PARENT] += _mp(mp, COUPLING_GENTLE) * (
        mp.fsum(capacity[index] for index in neighbors) / len(neighbors)
        - capacity[PARENT]
    )
    alignment = mp.fsum(
        1 - abs(wrap(after[PARENT] - after[index])) / _mp(mp, pi) for index in neighbors
    ) / len(neighbors)
    pressure_after = list(pressure)
    pressure_after[PARENT] *= 1 - _mp(mp, COUPLING_MODERATE) * alignment
    inventory, new_edges, new_weights = [], [], []
    for index, node in enumerate(nodes):
        if node == PARENT or graph.has_edge(PARENT, node):
            continue
        before_distance = abs(wrap(phase[index] - phase[PARENT]))
        pre = before_distance <= limit
        after_distance = score = None
        selected = False
        if pre:
            after_distance = abs(wrap(after[index] - after[PARENT]))
            if after_distance <= limit:
                score = (1 - after_distance / _mp(mp, pi)) / 2 + (
                    1
                    - abs(epi[PARENT] - epi[index])
                    / (epi[PARENT] + epi[index] + _mp(mp, 1e-9))
                    + 1
                    - abs(sense[PARENT] - sense[index])
                ) / 4
                selected = score >= _mp(mp, UM_COMPAT_THRESHOLD)
                if selected:
                    new_edges.append((PARENT, index))
                    new_weights.append(score)
        inventory.append((index, before_distance, pre, after_distance, score, selected))
    old_edges = tuple((rank[left], rank[right]) for left, right in graph.edges)
    old_weights = tuple(_mp(mp, data["weight"]) for *_, data in graph.edges(data=True))
    return {
        "mp": mp,
        "nodes": nodes,
        "epi": epi,
        "phase": phase,
        "after": tuple(after),
        "capacity": capacity,
        "capacity_after": tuple(capacity_after),
        "pressure_after": tuple(pressure_after),
        "sense": sense,
        "neighbors": neighbors,
        "resultant": resultant,
        "consensus": consensus,
        "alignment": alignment,
        "inventory": tuple(inventory),
        "old_edges": old_edges,
        "old_weights": old_weights,
        "new_edges": tuple(new_edges),
        "new_weights": tuple(new_weights),
    }


def test_full_inventory_precedes_post_merge_selection(study, oracle):
    source, report = study
    reset, mp = report.reset, oracle["mp"]
    assert reset.nodes == oracle["nodes"]
    assert reset.compatible_indices == oracle["neighbors"]
    assert len(reset.candidates) == len(oracle["inventory"]) == 16
    for item, expected in zip(reset.candidates, oracle["inventory"], strict=True):
        index, before, admitted, after, score, selected = expected
        assert item.candidate_index == index
        assert item.pre_admitted is admitted
        assert item.selected is selected
        _contains(mp, item.pre_distance, before)
        _contains(mp, item.prefilter_margin, _mp(mp, pi / 2) - before)
        if not admitted:
            assert item.post_distance is item.postfilter_margin is None
            assert item.score is item.score_margin is None
            assert item.prefilter_margin.hi < -reset.numerical_allowance
        else:
            _contains(mp, item.post_distance, after)
            _contains(mp, item.score, score)
            assert (
                min(
                    item.prefilter_margin.lo,
                    item.postfilter_margin.lo,
                    item.score_margin.lo,
                )
                > reset.numerical_allowance
            )
    assert reset.new_edges == oracle["new_edges"]
    assert tuple(reset.nodes[right] for _, right in reset.new_edges) == (
        3,
        10,
        11,
        12,
        13,
        14,
        15,
        CHILD,
    )
    _row_contains(mp, reset.new_conductances, oracle["new_weights"])
    # Child Si is an absent field consumed as zero, not a measured observation.
    assert all(alias not in source.graph.nodes[CHILD] for alias in ALIAS_SI)
    assert reset.sense_indices == (Q(1, 2),) * 20 + (Q(0),)


def test_complete_reset_has_independent_consensus_capacity_and_relief(study, oracle):
    _, report = study
    reset, mp = report.reset, oracle["mp"]
    _contains(mp, reset.consensus, oracle["consensus"])
    _contains(mp, reset.consensus_real, oracle["resultant"].real)
    _contains(mp, reset.consensus_imaginary, oracle["resultant"].imag)
    _row_contains(mp, reset.phase_after, oracle["after"])
    _row_contains(mp, reset.capacity_after, oracle["capacity_after"])
    _row_contains(mp, reset.raw_pressure_after, oracle["pressure_after"])
    _contains(mp, reset.parent_alignment, oracle["alignment"])
    assert reset.phase_changed_indices == (0, 1, 2, 4, 5)
    assert reset.capacity_changed_indices == (2,)
    assert reset.capacity_after[20].lo == reset.capacity_after[20].hi == Q(121.6)
    assert reset.phase_after[10].lo == reset.phase_after[10].hi < 0
    assert reset.phase_after[11].lo == reset.phase_after[11].hi < 0
    for index in reset.phase_changed_indices:
        assert 0 < reset.phase_after[index].lo < reset.phase_after[index].hi < 2 * Q(pi)
    assert report.numerical_allowance == Q(1, 2**40)


def _state_oracle(oracle, after):
    mp, epi = oracle["mp"], oracle["epi"]
    phase = oracle["after" if after else "phase"]
    capacity = oracle["capacity_after" if after else "capacity"]
    edges = oracle["old_edges"] + (oracle["new_edges"] if after else ())
    weights = oracle["old_weights"] + (oracle["new_weights"] if after else ())
    degrees, current, gradient = [0] * 21, [mp.mpf(0)] * 21, [mp.mpf(0)] * 21
    for (left, right), weight in zip(edges, weights, strict=True):
        degrees[left] += 1
        degrees[right] += 1
        flux = mp.sin(phase[right] - phase[left])
        current[left] += flux
        current[right] -= flux
        flux = weight * (epi[left] - epi[right])
        gradient[left] += flux
        gradient[right] -= flux
    pressure = tuple(
        current[i] / (mp.pi * degrees[i]) if degrees[i] else mp.mpf(0)
        for i in range(21)
    )
    form_rate = tuple(capacity[i] * pressure[i] for i in range(21))
    phase_rate = tuple(
        capacity[i] * gradient[i] / (mp.pi * degrees[i]) if degrees[i] else mp.mpf(0)
        for i in range(21)
    )
    mean_weights = tuple(degrees[i] / capacity[i] for i in range(21))

    def mean(values, count):
        return mp.fsum(values[i] * mean_weights[i] for i in range(count)) / mp.fsum(
            mean_weights[:count]
        )

    return {
        "degrees": tuple(degrees),
        "pressure": pressure,
        "form_rate": form_rate,
        "phase_rate": phase_rate,
        "mean_weights": mean_weights,
        "total_mean_weight": mp.fsum(mean_weights),
        "form_mean": mean(epi, 21),
        "phase_mean": mean(phase, 21),
        "old_marked_form_mean": mean(epi, 20),
        "old_marked_phase_mean": mean(phase, 20),
    }


def test_fresh_complete_rows_and_lifted_means_use_every_successor_edge(study, oracle):
    _, report = study
    mp = oracle["mp"]
    for result, after in ((report.before, False), (report.ideal.state, True)):
        expected = _state_oracle(oracle, after)
        assert result.degrees == expected.pop("degrees")
        for name in ("pressure", "form_rate", "phase_rate", "mean_weights"):
            _row_contains(mp, getattr(result, name), expected.pop(name))
        for name, value in expected.items():
            _contains(mp, getattr(result, name), value)
    assert report.before.positive_mean_metric is False
    assert report.ideal.state.positive_mean_metric is True
    assert report.ideal.state.degrees[2] == 12
    assert report.ideal.state.degrees[20] == 1
    assert report.ideal.state.form_rate[20].lo > 0
    assert report.ideal.state.phase_rate[20].hi < 0
    assert report.ideal.state.pressure[2].lo > report.reset.raw_pressure_after[2].hi
    # Full-support positive means differ from old-marked observations.
    assert report.ideal.state.form_mean.hi < report.ideal.state.old_marked_form_mean.lo


def test_event_work_includes_old_phase_changes_and_all_added_edges(study, oracle):
    _, report = study
    mp, phase, after, epi = (oracle[key] for key in ("mp", "phase", "after", "epi"))
    old_phase = tuple(
        mp.cos(phase[j] - phase[i]) - mp.cos(after[j] - after[i])
        for i, j in oracle["old_edges"]
    )
    new_form = tuple(
        weight * (epi[i] - epi[j]) ** 2 / 2
        for (i, j), weight in zip(
            oracle["new_edges"], oracle["new_weights"], strict=True
        )
    )
    new_phase = tuple(1 - mp.cos(after[j] - after[i]) for i, j in oracle["new_edges"])
    work = report.ideal.work
    assert len(work.old_edge_phase) == 45
    assert len(work.new_edge_phase) == len(work.new_edge_form) == 8
    _row_contains(mp, work.old_edge_phase, old_phase)
    _row_contains(mp, work.new_edge_form, new_form)
    _row_contains(mp, work.new_edge_phase, new_phase)
    _contains(mp, work.old_phase_total, mp.fsum(old_phase))
    _contains(mp, work.new_form_total, mp.fsum(new_form))
    _contains(mp, work.new_phase_total, mp.fsum(new_phase))
    _contains(mp, work.total, mp.fsum(old_phase + new_form + new_phase))
    assert work.total.lo > 3
    assert work.old_phase_total.lo > 0
    assert work.new_phase_total.lo > 0
    assert work.new_form_total.lo > 0
    # A child-only work calculation omits most of this supplied event work.
    assert work.total.lo > work.new_edge_form[-1].hi + work.new_edge_phase[-1].hi


def test_original_identity_is_excluded_without_claiming_loss_of_all_organization(
    study, oracle
):
    _, report = study
    mp, epi = oracle["mp"], oracle["epi"]
    form_mean = mp.fsum(epi[:20]) / 20

    def radius(phases):
        errors = tuple(phases[i] - (i % 10 // 2) * 2 * mp.pi / 5 for i in range(20))
        mean = mp.fsum(errors) / 20
        return mp.fsum(
            (epi[i] - form_mean) ** 2 + (errors[i] - mean) ** 2 for i in range(20)
        )

    _contains(mp, report.before_old_radius_squared, radius(oracle["phase"]))
    _contains(mp, report.ideal.old_radius_squared, radius(oracle["after"]))
    assert report.before_old_radius_squared.hi < Q(1, 256)
    assert report.ideal.old_radius_squared.lo > Q(1, 256)
    assert report.ideal.old_radius_excluded is True
    assert report.conditional.old_radius_excluded is True
    assert report.conditional.old_radius_squared.lo > Q(1, 256)
    # Acute initial geometry does not restore the previous small-radius identity.
    gaps = tuple(
        abs(
            mp.atan2(
                mp.sin(oracle["after"][j] - oracle["after"][i]),
                mp.cos(oracle["after"][j] - oracle["after"][i]),
            )
        )
        for i, j in oracle["old_edges"] + oracle["new_edges"]
    )
    assert max(gaps) < mp.pi / 2


def test_conditional_budget_is_separate_from_exact_reset_and_never_an_observation(
    study,
):
    source, report = study
    assert report.conditional.work.total.lo < report.ideal.work.total.lo
    assert report.conditional.work.total.hi > report.ideal.work.total.hi
    assert report.conditional.old_radius_squared.lo < report.ideal.old_radius_squared.lo
    assert report.conditional.old_radius_squared.hi > report.ideal.old_radius_squared.hi
    assert tuple(source.graph.nodes[PARENT]["glyph_history"]) == ("VAL", "THOL")
    assert source.graph.degree(CHILD) == 0
    assert source.graph.number_of_edges() == 45
    assert source.graph.nodes[CHILD]["glyph_history"] == []
