"""Actual SHA-law preparation and wrapper controls, without reserved motion."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.research import (
    maintained_parent_sha_execution,
    maintained_parent_um_execution,
)
from tnfr.research import maintained_sha_motion as owner
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_sha_motion_source import load_sha_successor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def no_event_refresh_or_taylor():
    def forbidden(*args, **kwargs):
        pytest.fail("SHA preparation controls cannot execute an event or trajectory")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (owner, "validated_box_taylor_step"),
            (maintained_parent_sha_execution, "evaluate_sha_event"),
            (maintained_parent_sha_execution, "execute_network_operator_stage"),
            (maintained_parent_sha_execution, "refresh_pressure"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "execute_network_operator_stage"),
            (maintained_parent_um_execution, "refresh_pressure"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def prepared():
    admitted = load_sha_successor(
        ROOT / "docs/assets/maintained_parent_sha", repository_root=ROOT
    )
    signature = _signature(admitted.graph)
    result = owner.prepare_sha_motion(admitted)
    yield admitted, result
    assert _signature(admitted.graph) == signature


def _tight_overlap(left, right):
    assert max(left.lo, right.lo) <= min(left.hi, right.hi)
    assert max(left.hi, right.hi) - min(left.lo, right.lo) < Q(1, 2**80)


def test_actual_point_source_retains_all_coordinates_lifts_and_changed_capacity(
    prepared,
):
    admitted, (prediction, box, _) = prepared
    reset = admitted.prediction
    assert prediction.nodes == tuple(range(20)) + ("2_sub_0",)
    assert len(prediction.edges) == len(prediction.shifts) == 53
    assert len(box) == 42 and all(value.lo == value.hi for value in box)
    assert box == tuple(map(I, prediction.epi + prediction.phases))
    assert prediction.epi == reset.epi and prediction.phases == reset.phases
    assert prediction.capacities == reset.represented_capacity_after
    assert prediction.capacities[2] != reset.capacity_before[2]
    assert prediction.capacities[2] < reset.source_um_parent_capacity.lo
    assert prediction.phases[10] < 0 and prediction.phases[11] < 0
    assert prediction.horizon == Q(1, 2)
    assert all(
        type(value) is Q
        for values in (
            prediction.epi,
            prediction.phases,
            prediction.capacities,
            prediction.conductances,
        )
        for value in values
    )
    for (i, j), shift in zip(prediction.edges, prediction.shifts, strict=True):
        gap = I(prediction.phases[j] - prediction.phases[i]) - 2 * shift * pi_interval()
        assert gap.abs_max < pi_interval().lo / 2


def test_complete_field_matches_independently_summed_actual_law(prepared):
    _, (prediction, box, flow) = prepared
    rates = flow(box)
    assert len(rates) == 42
    for i in range(21):
        neighbors = [
            (b if a == i else a, weight)
            for (a, b), weight in zip(
                prediction.edges, prediction.conductances, strict=True
            )
            if i in (a, b)
        ]
        mobility = I(prediction.capacities[i]) / (pi_interval() * len(neighbors))
        current = sum(
            (sin(I(prediction.phases[j] - prediction.phases[i])) for j, _ in neighbors),
            I(0),
        )
        gradient = sum(
            (
                I(weight * (prediction.epi[i] - prediction.epi[j]))
                for j, weight in neighbors
            ),
            I(0),
        )
        _tight_overlap(rates[i], mobility * current)
        _tight_overlap(rates[21 + i], mobility * gradient)
        _tight_overlap(rates[i], prediction.bounds.form_rate[i])
        _tight_overlap(rates[21 + i], prediction.bounds.phase_rate[i])
    with pytest.raises(ValueError, match="every reduced/full"):
        flow(box[:-1])


def test_static_bounds_retain_41_edges_and_leave_actual_parent_edges_unresolved(
    prepared,
):
    _, (prediction, _, _) = prepared
    bounds = prediction.bounds
    retained = set(bounds.retained_edge_indices)
    assert len(retained) == 41
    assert {edge for k, edge in enumerate(prediction.edges) if k not in retained} == {
        edge for edge in prediction.edges if 2 in edge
    }
    assert all(bounds.horizon_remaining_margins[k].lo > Q(593, 1000) for k in retained)
    child = prediction.edges.index((2, 20))
    assert bounds.edge_phase_velocity[child].hi < Q(-534, 100)
    assert bounds.edge_phase_acceleration[child].hi < Q(-64, 100)
    assert bounds.horizon_remaining_margins[child].hi < 0
    # A failed sufficient bound is not an exit. Fresh global acceleration
    # and initial speed give a strictly shorter analytical continuity window.
    time = Q(1, 25)
    assert all(
        margin.lo - velocity.abs_max * time - acceleration.hi * time**2 / 2 > 0
        for margin, velocity, acceleration in zip(
            bounds.principal_margins,
            bounds.edge_phase_velocity,
            bounds.edge_acceleration_magnitudes,
            strict=True,
        )
    )


def test_cached_prediction_and_assessment_cannot_supply_new_premises(prepared):
    admitted, (prediction, box, flow) = prepared
    forged = replace(
        admitted,
        prediction=object(),
        assessment={"actual_fields": {"capacity": (False,) * 21}},
        sequence=None,
        receipts=(),
    )
    rebuilt, source, field = owner.prepare_sha_motion(forged)
    assert rebuilt == prediction and source == box
    assert field(source) == flow(box)


@pytest.mark.parametrize("alias,value", ((ALIAS_EPI, 0.25), (ALIAS_VF, False)))
def test_mutated_live_endpoint_is_rejected_before_field_preparation(
    prepared, alias, value
):
    admitted, _ = prepared
    graph = deepcopy(admitted.graph)
    set_attr(graph.nodes[2], alias, value)
    with pytest.raises(ValueError, match="live SHA endpoint differs"):
        owner.prepare_sha_motion(replace(admitted, graph=graph))


def test_unassociated_source_cannot_become_a_motion_preparation(prepared):
    admitted, _ = prepared
    with pytest.raises(ValueError, match="file-backed"):
        owner.prepare_sha_motion(replace(admitted, evidence_dir=None))


def test_wrapper_passes_complete_new_law_and_includes_preparation_time(monkeypatch):
    calls = []
    initial, field, source, stepper = (object() for _ in range(4))
    prediction = SimpleNamespace(
        edges=((0, 1),),
        shifts=(3,),
        bounds=SimpleNamespace(horizon_remaining_margins=(I(Q(1, 2), Q(3, 4)),)),
    )

    def prepare(value):
        assert calls == ["clock"] and value is source
        calls.append("prepare")
        return prediction, initial, field

    def clock():
        calls.append("clock")
        return 17.25

    def certificate(box, flow, edges, shifts, energy, **options):
        assert calls == ["clock", "prepare"]
        assert box is initial and flow is field
        assert edges == ((0, 1),) and shifts == (3,) and energy == (Q(1, 2),)
        assert options == {
            "record_step": record,
            "monotonic_clock": clock,
            "started": 17.25,
            "stepper": stepper,
        }
        return "synthetic retained result"

    def record(_):
        pass

    monkeypatch.setattr(owner, "prepare_sha_motion", prepare)
    monkeypatch.setattr(owner, "certify_retention", certificate)
    monkeypatch.setattr(owner, "validated_box_taylor_step", stepper)
    assert (
        owner.certify_sha_retention(source, record_step=record, monotonic_clock=clock)
        == "synthetic retained result"
    )


def test_invalid_callbacks_reject_before_source_read(monkeypatch):
    def forbidden(_):
        pytest.fail("invalid callback must reject before source admission")

    monkeypatch.setattr(owner, "prepare_sha_motion", forbidden)
    with pytest.raises(TypeError, match="callbacks"):
        owner.certify_sha_retention(None, record_step=None)
    with pytest.raises(TypeError, match="callbacks"):
        owner.certify_sha_retention(None, record_step=lambda _: None, monotonic_clock=0)
