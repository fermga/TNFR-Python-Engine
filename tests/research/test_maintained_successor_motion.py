"""Static weighted-law derivatives and storage bounds without temporal execution."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.dynamics import integrators, relational
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.research import local_capacity_history, maintained_parent_thol_execution
from tnfr.research import maintained_successor_motion as owner
from tnfr.research.artifact_io import file_receipt
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_parent_um_source import load_thol_successor

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_thol"


def _small(**changes):
    arguments = dict(
        epi=(0, 1),
        phases=(I(0), pi_interval() / 6),
        capacities=(2, 3),
        edges=((0, 1),),
        conductances=(2,),
        shifts=(0,),
        horizon=Q(1, 10),
    )
    arguments.update(changes)
    return owner._bound_motion(**arguments)


def _same_enclosed_value(actual, expected):
    defect = actual - expected
    assert defect.contains(0)
    assert defect.abs_max < Q(1, 10**30)


def test_two_node_law_has_independent_signed_rates_and_acceleration():
    report = _small()
    pi = pi_interval()
    # W=2, nu=(2,3), theta gap=pi/6: S=(1/2,-1/2).
    _same_enclosed_value(report.form_rate[0], 1 / pi)
    _same_enclosed_value(report.form_rate[1], -3 / (2 * pi))
    _same_enclosed_value(report.edge_phase_velocity[0], 10 / pi)
    _same_enclosed_value(report.phase_acceleration[0], 10 / pi**2)
    _same_enclosed_value(report.phase_acceleration[1], -15 / pi**2)
    _same_enclosed_value(report.edge_phase_acceleration[0], -25 / pi**2)
    _same_enclosed_value(report.edge_acceleration_magnitudes[0], 50 / pi**2)
    assert report.storage_rate.contains(0)


def test_energy_speed_bound_uses_actual_weights_and_capacity_cross_term():
    report = _small()
    pi = pi_interval()
    energy = 2 - sqrt(I(3)) / 2
    _same_enclosed_value(report.form_storage, I(1))
    _same_enclosed_value(report.storage, energy)
    _same_enclosed_value(report.edge_energy_coefficients[0], 50 / pi**2)
    _same_enclosed_value(report.edge_speed_bounds[0], sqrt(100 * energy / pi**2))
    assert report.retained_edge_indices == (0,)
    assert report.horizon_remaining_margins[0].lo > 0


def test_zero_phase_two_node_source_saturates_energy_cauchy_bound():
    report = _small(phases=(0, 0))
    # This state has V=0 and the only relative form direction saturates Cauchy.
    _same_enclosed_value(
        report.edge_speed_bounds[0], abs(report.edge_phase_velocity[0])
    )
    assert report.phase_storage.contains(0)
    assert report.edge_phase_acceleration[0].contains(0)


def test_reflection_preserves_phase_storage_and_means_but_reverses_motion():
    original = _small()
    # rho=(1/2,1/3), x=(0,1) give weighted form mean 2/5.
    reflected = _small(epi=(Q(4, 5), Q(-1, 5)))
    assert reflected.phases == original.phases
    _same_enclosed_value(reflected.storage, original.storage)
    _same_enclosed_value(
        reflected.epi[0] / 2 + reflected.epi[1] / 3,
        original.epi[0] / 2 + original.epi[1] / 3,
    )
    for before, after in zip(original.phase_rate, reflected.phase_rate, strict=True):
        _same_enclosed_value(after, -before)
    assert original.edge_phase_velocity[0].lo > 0
    assert reflected.edge_phase_velocity[0].hi < 0


def test_sparse_coefficient_equals_incidence_psd_quadratic_form():
    edges, weights = ((0, 1), (0, 2), (1, 2)), (2, 3, 5)
    report = owner._bound_motion(
        epi=(0, 1, -1),
        phases=(0, 0, 0),
        capacities=(1, 2, 4),
        edges=edges,
        conductances=weights,
        shifts=(0, 0, 0),
        horizon=0,
    )
    for index, (left, right) in enumerate(edges):
        # The node incidence column is -e_left+e_right; evaluate W differences
        # independently instead of reconstructing the diagonal formula.
        vector = [I(0) for _ in range(3)]
        vector[left] = -report.mobility[left]
        vector[right] = report.mobility[right]
        quadratic = sum(
            (
                weight * (vector[b] - vector[a]) ** 2
                for (a, b), weight in zip(edges, weights)
            ),
            I(0),
        )
        _same_enclosed_value(report.edge_energy_coefficients[index], quadratic)
        assert report.edge_energy_coefficients[index].lo > 0


def test_failed_margin_is_an_abstention_not_a_crossing_verdict():
    report = _small(horizon=100)
    assert report.retained_edge_indices == ()
    assert report.horizon_remaining_margins[0].hi < 0
    assert not hasattr(report, "crossing_certified")
    assert not hasattr(report, "trajectory")
    with pytest.raises(FrozenInstanceError):
        report.storage = I(0)


@pytest.mark.parametrize("horizon", (True, False, -1, float("inf"), float("nan")))
def test_invalid_horizon_is_rejected_before_source_access(horizon):
    with pytest.raises((TypeError, ValueError)):
        owner.predict_successor_motion(None, horizon=horizon)


@pytest.mark.parametrize(
    "changes",
    (
        {"capacities": (True, 3)},
        {"capacities": (0, 3)},
        {"conductances": (0,)},
        {"epi": (False, 1)},
        {"phases": (I(0),)},
        {"conductances": ()},
        {"shifts": (False,)},
        {"shifts": (1,)},
        {"edges": ((0, 1), (0, 1)), "conductances": (2, 2), "shifts": (0, 0)},
    ),
)
def test_invalid_primitive_law_or_branch_is_not_repaired(changes):
    with pytest.raises((TypeError, ValueError)):
        _small(**changes)


@pytest.fixture(scope="module", autouse=True)
def no_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("static motion admission must not execute an event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (_coupling_stage_kernel, "propose_coupling_target"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (network_stage, "execute_coupling_stage"),
            (word_execution, "execute_network_operator_stage"),
            (maintained_parent_thol_execution, "evaluate_thol_event"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "rk4_coupled_step"),
            (integrators, "rk4_coupled_step"),
            (relational, "step_relational_exchange"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def study():
    paths = tuple(sorted(path for path in EVIDENCE.iterdir() if path.is_file()))
    receipts = tuple(file_receipt(path) for path in paths)
    source = load_thol_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(source.graph)
    report = owner.predict_successor_motion(source.graph)
    yield source, report
    assert _signature(source.graph) == signature
    assert tuple(file_receipt(path) for path in paths) == receipts


def test_complete_energy_bound_retains_all_41_slow_edges_through_half(study):
    _, report = study
    edges = report.chamber.edges
    slow = tuple(
        index for index, edge in enumerate(edges) if 2 not in edge and 20 not in edge
    )
    assert len(slow) == 41 and report.horizon == Q(1, 2)
    for state in (report.ideal, report.conditional):
        assert state.retained_edge_indices == slow
        assert min(state.horizon_remaining_margins[index].lo for index in slow) > Q(
            59, 100
        )
        assert state.form_storage.lo > Q(105, 10000)
        assert state.storage_rate.contains(0)
        assert all(value.lo > 0 for value in state.edge_energy_coefficients)
        assert not hasattr(state, "all_time_principal_retention")


def test_leaf_initial_motion_is_signed_without_a_frozen_velocity_prediction(study):
    _, report = study
    leaf = report.chamber.edges.index((2, 20))
    for state in (report.ideal, report.conditional):
        velocity, acceleration = (
            state.edge_phase_velocity[leaf],
            state.edge_phase_acceleration[leaf],
        )
        assert -Q(538, 100) < velocity.lo < velocity.hi < -Q(537, 100)
        assert -Q(77, 100) < acceleration.lo < acceleration.hi < -Q(76, 100)
        assert state.horizon_remaining_margins[leaf].hi < 0
        assert state.edge_acceleration_magnitudes[leaf].lo > 3700


def test_conditional_boxes_keep_every_ideal_source_coefficient(study):
    _, report = study
    for name in (
        "epi",
        "phases",
        "capacities",
        "conductances",
        "mobility",
        "form_rate",
        "phase_rate",
        "phase_acceleration",
        "edge_phase_velocity",
        "edge_phase_acceleration",
        "edge_energy_coefficients",
        "edge_speed_bounds",
    ):
        for outer, inner in zip(
            getattr(report.conditional, name), getattr(report.ideal, name), strict=True
        ):
            assert outer.contains(inner)
    assert report.conditional.storage.contains(report.ideal.storage)


def test_cached_motion_report_cannot_override_invalid_live_primitive(study):
    source, report = study
    changed = deepcopy(source.graph)
    changed.graph["_successor_motion"] = report
    set_attr(changed.nodes[2], ALIAS_VF, True)
    with pytest.raises((TypeError, ValueError)):
        owner.predict_successor_motion(changed)


def test_report_is_not_an_admitted_graph(study):
    _, report = study
    with pytest.raises((AttributeError, TypeError, ValueError)):
        owner.predict_successor_motion(report)
