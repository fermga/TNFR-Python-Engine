"""Prior-only sine forecast controls, never the reserved research preparation.

Synthetic prior evidence comes from independent full-edge differentiation in
the capacity-observation tests. The distinct dyadic fixture below is used only
for API admission and a short numerical control, with no frozen producer.
"""

import json
import pickle
from copy import copy
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tests.physics.test_relational_sine_capacity_observation import _mp, _oracle
from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_comparison
from tnfr.physics import relational_sine_forecast as owner
from tnfr.physics import relational_sine_mediation
from tnfr.physics.relational_sine_observation import infer_relational_sine_hidden_state

MODEL = RelationalExchangeModel(2, phase_domain="regular")
PORTS = (1, 2)
OBSERVATION_TIME = Q(3, 2)
START = OBSERVATION_TIME + Q(1, 64)
END = OBSERVATION_TIME + Q(1, 32)
STEP = Q(1, 64)
NEIGHBORS = ((2,), (2,), (0, 1))


def _evidence(*, hidden_phase=Q(-1, 4)):
    visible = nx.empty_graph(PORTS)
    for node, x, phase, capacity in ((1, 1, 0, 1), (2, -0.5, 0.75, 2)):
        visible.nodes[node].update(EPI=x, theta=phase, nu_f=capacity)
    oracle = _oracle(
        visible,
        PORTS,
        hidden_capacity=Q(3, 2),
        hidden_form=Q(1, 4),
        hidden_phase=hidden_phase,
        model=MODEL,
    )
    for field in (
        "form_rates",
        "phase_rates",
        "form_accelerations",
        "phase_accelerations",
    ):
        oracle[field] = {
            port: ((lo + hi) / 2 - Q(1, 2**24), (lo + hi) / 2 + Q(1, 2**24))
            for port, (lo, hi) in oracle[field].items()
        }
    state = infer_relational_sine_hidden_state(
        visible,
        ports=PORTS,
        form_rate_bounds=oracle["form_rates"],
        phase_rate_bounds=oracle["phase_rates"],
        reference_model=MODEL,
        source_id="nonprotocol-independent-first-derivatives",
        clock_id="declared-test-clock",
        observation_time=OBSERVATION_TIME,
        evidence_window=(OBSERVATION_TIME, OBSERVATION_TIME),
        forecast_start=START,
    )
    capacity = state.infer_capacity(
        form_acceleration_bounds=oracle["form_accelerations"],
        phase_acceleration_bounds=oracle["phase_accelerations"],
        source_id="nonprotocol-independent-second-derivatives",
        clock_id=state.clock_id,
        observation_time=OBSERVATION_TIME,
        evidence_window=(OBSERVATION_TIME, OBSERVATION_TIME),
    )
    return oracle, capacity


@pytest.fixture(scope="module")
def prior():
    oracle, capacity = _evidence()
    admission = owner.admit_sine_prior(capacity)
    assert admission.admitted, admission.reasons
    return oracle, capacity, admission


def _forecast(admission, **changes):
    arguments = dict(end_time=END, time_step=STEP, order=6)
    arguments.update(changes)
    return owner.forecast_sine_prior(admission, **arguments)


def _inside(interval, expected):
    with mp.workdps(90):
        assert _mp(interval.lo) <= expected <= _mp(interval.hi)


@pytest.mark.parametrize("loss", (Q(0), Q(7, 19), Q(1, 2**1100)))
def test_forecast_admits_exact_stored_law_without_renormalizing_or_solving(
    loss, monkeypatch
):
    model = copy(MODEL)
    for field, value in (
        ("epi_weight", loss),
        ("phase_weight", Q(5, 17)),
        ("storage_scale", Q(11, 13)),
    ):
        object.__setattr__(model, field, value)
    calls = []

    def probe(initial, duration, flow, domain, **kwargs):
        calls.append(flow(initial))
        return None, None, "static_admission_probe"

    monkeypatch.setattr(owner, "validated_taylor_step", probe)
    report = owner.bound_sine_flow(
        (0, 1, 0, 0, 3),
        neighbors=((1,), (0,)),
        visible_capacity=(2,),
        model=model,
        observation_time=0,
        end_time=Q(1, 100),
        time_step=Q(1, 100),
    )
    assert report.model is model
    assert report.model.epi_weight == loss
    assert report.reasons == ("static_admission_probe",)
    assert len(calls) == 1
    form0, form1, phase0, phase1, capacity = calls[0]
    assert form0.contains(2 * loss) and form1.contains(-3 * loss)
    assert max(form0.width, form1.width) < Q(1, 10**35)
    assert capacity == I(0)
    with mp.workdps(90):
        _inside(phase0, -mp.mpf(130) / (187 * mp.pi))
        _inside(phase1, mp.mpf(195) / (187 * mp.pi))


def test_joint_witness_is_derived_from_prior_and_checks_every_observed_row(prior):
    oracle, capacity, admission = prior
    assert admission.capacity_inference is capacity
    assert admission.joint_witness == (
        Q(1),
        Q(-1, 2),
        Q(1, 4),
        Q(0),
        Q(3, 4),
        Q(-1, 4),
        Q(3, 2),
    )
    assert len(admission.initial_box) == 7
    for interval, point in zip(admission.initial_box, admission.joint_witness):
        assert interval.contains(point)
    assert any(interval.width > 0 for interval in admission.initial_box)
    for index, node in enumerate(PORTS):
        for offset, kind in ((0, "form"), (3, "phase")):
            row = index + offset
            first = admission.witness_first_rate_bounds[row]
            acceleration = admission.witness_acceleration_bounds[row]
            _inside(first, oracle[f"exact_{kind}_rate"][node])
            _inside(acceleration, oracle[f"exact_{kind}_acceleration"][node])
            assert first.subset_of(I(*oracle[f"{kind}_rates"][node]))
            assert acceleration.subset_of(I(*oracle[f"{kind}_accelerations"][node]))
    assert admission.witness_first_rate_bounds[-1] == I(0)
    assert admission.witness_acceleration_bounds[-1] == I(0)


def test_overlap_alone_does_not_certify_a_joint_witness(prior):
    _, capacity, admission = prior
    original = admission.witness_first_rate_bounds[0]
    assert original.width > 0
    partial = I(original.hi)
    assert partial.subset_of(original) and not original.subset_of(partial)
    state = capacity.state_inference
    changed = replace(state, form_rate_bounds=(partial,) + state.form_rate_bounds[1:])
    rejected = owner.admit_sine_prior(replace(capacity, state_inference=changed))
    assert not rejected.admitted
    assert rejected.reasons


def test_prior_admission_rebuilds_inference_from_retained_evidence(prior):
    _, capacity, expected = prior
    stale_state = replace(
        capacity.state_inference,
        hidden_form_bounds=I(99),
        status="inconsistent",
    )
    stale = replace(
        capacity,
        state_inference=stale_state,
        capacity_bounds=I(0),
        status="inconsistent",
    )
    rebuilt = owner.admit_sine_prior(stale)
    assert rebuilt.admitted
    assert rebuilt.initial_box == expected.initial_box
    assert rebuilt.joint_witness == expected.joint_witness
    assert rebuilt.capacity_inference == capacity
    assert stale.capacity_bounds == I(0)


@pytest.mark.parametrize(
    "changes",
    ({"clock_id": "another-clock"}, {"observation_time": OBSERVATION_TIME + 1}),
)
def test_prior_admission_rechecks_evidence_association(prior, changes):
    _, capacity, _ = prior
    with pytest.raises(ValueError):
        owner.admit_sine_prior(replace(capacity, **changes))


def test_equivalent_original_association_does_not_replace_normalized_inputs(
    prior, monkeypatch
):
    _, capacity, expected = prior
    state = capacity.state_inference
    represented = replace(
        capacity,
        observation_time=float(capacity.observation_time),
        state_inference=replace(
            state, visible_capacity=tuple(map(float, state.visible_capacity))
        ),
    )
    admission = owner.admit_sine_prior(represented)
    assert admission.capacity_inference is represented
    assert admission.initial_box == expected.initial_box

    def stop_at_initial(box, duration, *args, **kwargs):
        assert kwargs["time"] == OBSERVATION_TIME
        assert type(kwargs["time"]) is Q
        return None, box, "fixture_budget_unavailable"

    monkeypatch.setattr(owner, "validated_taylor_step", stop_at_initial)
    result = _forecast(admission)
    assert result.prior_admission is admission
    assert result.visible_capacity == (Q(1), Q(2))
    assert all(type(value) is Q for value in result.visible_capacity)
    assert result.initial_box == expected.initial_box


def test_unresolved_relative_phase_chart_is_explicitly_unavailable():
    # Represented float(pi) is near the negative real axis, not mathematical pi.
    _, capacity = _evidence(hidden_phase=Q(float(mp.pi)))
    assert capacity.state_inference.status == "bounded_candidate"
    admission = owner.admit_sine_prior(capacity)
    assert not admission.admitted
    assert admission.reasons


def test_sine_field_directional_derivative_matches_independent_full_edge_oracle(prior):
    oracle, _, admission = prior
    initial = tuple(I(value) for value in admission.joint_witness)
    field = owner._sine_flow(
        initial, neighbors=NEIGHBORS, visible_capacity=(Q(1), Q(2)), model=MODEL
    )
    jets = owner._sine_flow(
        tuple(Jet((value, rate)) for value, rate in zip(initial, field)),
        neighbors=NEIGHBORS,
        visible_capacity=(Q(1), Q(2)),
        model=MODEL,
    )
    nodes = PORTS + (oracle["hidden"],)
    for index, node in enumerate(nodes):
        for offset, kind in ((0, "form"), (3, "phase")):
            _inside(field[index + offset], oracle[f"exact_{kind}_rate"][node])
            _inside(
                jets[index + offset].coeffs[1],
                oracle[f"exact_{kind}_acceleration"][node],
            )
    assert field[-1] == I(0) and jets[-1].coeffs == (I(0), I(0))


def test_exact_frozen_state_is_enclosed_for_the_complete_nonzero_time_gap():
    initial = tuple(map(I, (Q(1), Q(-1, 2), Q(1, 4), Q(0), Q(3, 4), Q(-1, 4), Q(0))))
    result = owner.bound_sine_flow(
        initial,
        neighbors=NEIGHBORS,
        visible_capacity=(Q(0), Q(0)),
        model=MODEL,
        observation_time=OBSERVATION_TIME,
        end_time=END,
        time_step=STEP,
        order=6,
    )
    assert result.admitted, result.reasons
    assert result.endpoint == initial
    assert result.observation_time == OBSERVATION_TIME
    assert result.end_time == result.validated_end_time == END
    assert result.steps[0].time == OBSERVATION_TIME
    assert sum((step.duration for step in result.steps), Q(0)) == END - OBSERVATION_TIME


def test_eleven_node_complete_field_and_one_step_retain_all_coordinates():
    # An unrelated path control exercises the enlarged numerical work domain;
    # it is not a reserved research preparation or a trajectory campaign.
    size = 11
    neighbors = tuple(
        tuple(j for j in (i - 1, i + 1) if 0 <= j < size) for i in range(size)
    )
    forms = tuple(Q(i - 5, 16) for i in range(size))
    phases = tuple(Q((-1) ** i, 32) for i in range(size))
    capacities = tuple(1 + Q(i % 3, 4) for i in range(size))
    initial = tuple(map(I, forms + phases + (capacities[-1],)))
    rates = owner._sine_flow(
        initial, neighbors=neighbors, visible_capacity=capacities[:-1], model=MODEL
    )
    with mp.workdps(90):
        for i, adjacent in enumerate(neighbors):
            degree = len(adjacent)
            gradient = sum((_mp(forms[i] - forms[j]) for j in adjacent), mp.mpf(0))
            current = sum(
                (mp.sin(_mp(phases[j] - phases[i])) for j in adjacent), mp.mpf(0)
            )
            capacity = _mp(capacities[i])
            _inside(
                rates[i], capacity * (-gradient / 2 + current / (2 * mp.pi)) / degree
            )
            _inside(rates[size + i], capacity * gradient / (4 * mp.pi * degree))
    assert rates[-1] == I(0)

    duration = Q(1, 128)
    result = owner.bound_sine_flow(
        initial,
        neighbors=neighbors,
        visible_capacity=capacities[:-1],
        model=MODEL,
        observation_time=Q(0),
        end_time=duration,
        time_step=duration,
        order=4,
    )
    assert result.admitted, result.reasons
    assert len(result.steps) == 1
    assert len(result.endpoint) == 23
    assert result.validated_end_time == duration
    assert result.endpoint[-1] == initial[-1]
    step = result.steps[0]
    assert step.picard_interior_margin > 0
    assert all(value.subset_of(tube) for value, tube in zip(step.endpoint, step.tube))


def test_twelve_node_state_exceeds_shared_work_limit_before_solver(monkeypatch):
    size = 12
    neighbors = tuple(
        tuple(j for j in (i - 1, i + 1) if 0 <= j < size) for i in range(size)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("the state work limit must reject before solver execution")

    monkeypatch.setattr(owner, "validated_taylor_step", forbidden)
    with pytest.raises(ValueError, match="24 state coordinates"):
        owner.bound_sine_flow(
            (I(0),) * (2 * size) + (I(1),),
            neighbors=neighbors,
            visible_capacity=(Q(1),) * (size - 1),
            model=MODEL,
            observation_time=Q(0),
            end_time=Q(1, 128),
            time_step=Q(1, 128),
            order=4,
        )


def test_prior_forecast_retains_capacity_uncertainty_and_uses_observation_time(prior):
    _, capacity, admission = prior
    result = _forecast(admission)
    assert result.admitted, result.reasons
    assert result.initial_box == admission.initial_box
    assert result.initial_box[-1].width > 0
    assert result.initial_box[-1].lo <= capacity.capacity_bounds.lo
    assert capacity.capacity_bounds.hi <= result.initial_box[-1].hi
    assert result.endpoint[-1] == result.initial_box[-1]
    assert result.steps[0].time == OBSERVATION_TIME < START
    assert result.validated_end_time == END
    assert sum((step.duration for step in result.steps), Q(0)) == END - OBSERVATION_TIME
    assert all(step.picard_interior_margin > 0 for step in result.steps)
    assert any(step.propagated_initial_radii[0] > 0 for step in result.steps)


@pytest.mark.parametrize(
    "changes",
    (
        {"initial_box": (I(0),) * 7},
        {"neighbors": ((1, 2), (0, 2), (0, 1))},
        {"visible_capacity": (Q(5), Q(5))},
        {"status": "unavailable", "joint_witness": None},
    ),
)
def test_forecast_rebuilds_preparation_instead_of_trusting_admission_cache(
    prior, monkeypatch, changes
):
    _, _, admission = prior
    stale = replace(admission, **changes)
    calls = []

    def stop_at_initial(box, duration, *args, **kwargs):
        calls.append(box)
        return None, box, "fixture_budget_unavailable"

    monkeypatch.setattr(owner, "validated_taylor_step", stop_at_initial)
    result = _forecast(stale)
    assert calls == [admission.initial_box]
    assert result.initial_box == admission.initial_box
    assert result.neighbors == admission.neighbors
    assert result.visible_capacity == admission.visible_capacity
    assert result.prior_admission == admission
    assert result.prior_admission.admitted
    assert result.prior_admission is not stale
    assert result.freeze_hidden is False
    assert result.status == "unavailable"


def test_capacity_uncertainty_alone_reaches_hidden_and_visible_state(prior):
    _, _, admission = prior
    initial = tuple(I(value) for value in admission.joint_witness[:-1]) + (
        I(Q(23, 16), Q(25, 16)),
    )
    result = owner.bound_sine_flow(
        initial,
        neighbors=NEIGHBORS,
        visible_capacity=(Q(1), Q(2)),
        model=MODEL,
        observation_time=OBSERVATION_TIME,
        end_time=START,
        time_step=STEP,
        order=6,
    )
    assert result.admitted, result.reasons
    assert all(value.width == 0 for value in result.initial_box[:-1])
    assert result.endpoint[-1] == initial[-1]
    radii = result.steps[0].propagated_initial_radii
    assert radii[2] > 0 and radii[5] > 0  # hidden form and phase
    assert radii[0] > 0 and radii[3] > 0  # transmitted visible uncertainty
    assert result.endpoint[0].width > Q(1, 10**8)


def test_frozen_mediator_is_a_declared_capacity_control_with_original_prior(prior):
    _, capacity, admission = prior
    original = pickle.dumps(admission, protocol=5)
    result = _forecast(admission, freeze_hidden=True)
    assert result.admitted, result.reasons
    assert result.freeze_hidden is True
    assert result.prior_admission is admission
    assert result.prior_admission.capacity_inference is capacity
    assert result.initial_box[:-1] == admission.initial_box[:-1]
    assert result.initial_box[-1] == I(0)
    assert admission.initial_box[-1].lo > 0
    for index in (2, 5):
        held = result.initial_box[index]
        endpoint = result.endpoint[index]
        assert endpoint == held
    assert result.neighbors == NEIGHBORS
    assert pickle.dumps(admission, protocol=5) == original


def test_failed_fixed_step_retains_unavailability_without_retry(prior, monkeypatch):
    _, _, admission = prior
    calls = []

    def failed_step(box, duration, *args, **kwargs):
        calls.append((box, duration, kwargs["time"]))
        return None, box, "independent_injected_budget_failure"

    monkeypatch.setattr(owner, "validated_taylor_step", failed_step)
    result = _forecast(admission)
    assert not result.admitted
    assert result.reasons == ("independent_injected_budget_failure",)
    assert result.validated_end_time == OBSERVATION_TIME
    assert result.endpoint == result.failed_tube == admission.initial_box
    assert result.end_time == END and result.steps == ()
    assert calls == [(admission.initial_box, STEP, OBSERVATION_TIME)]


@pytest.mark.parametrize(
    "changes",
    (
        {"end_time": START - Q(1, 128)},
        {"time_step": 0.01},
        {"freeze_hidden": 1},
    ),
)
def test_invalid_clock_or_control_rejects_before_solver(prior, monkeypatch, changes):
    _, _, admission = prior

    def forbidden(*args, **kwargs):
        pytest.fail("invalid forecast inputs must reject before numerical execution")

    monkeypatch.setattr(owner, "validated_taylor_step", forbidden)
    with pytest.raises((ValueError, TypeError)):
        _forecast(admission, **changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"neighbors": ((2, 2), (2,), (0, 1))},
        {"neighbors": ((2,), (2,), (0,))},
        {"neighbors": ((1,), (0,), (3,), (2,))},
        {"initial": (I(0),) * 6},
        {"initial": (I(0),) * 6 + (I(-1),)},
        {"time_step": STEP / 1000},
    ),
)
def test_invalid_support_state_or_work_budget_rejects_before_solver(
    prior, monkeypatch, changes
):
    _, _, admission = prior

    def forbidden(*args, **kwargs):
        pytest.fail("invalid sine-domain data must reject before numerical execution")

    monkeypatch.setattr(owner, "validated_taylor_step", forbidden)
    arguments = dict(
        initial=admission.initial_box,
        neighbors=NEIGHBORS,
        visible_capacity=(Q(1), Q(2)),
        model=MODEL,
        observation_time=OBSERVATION_TIME,
        end_time=END,
        time_step=STEP,
    )
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        owner.bound_sine_flow(**arguments)


def test_prior_admission_and_forecast_never_recapture_hidden_graph(prior, monkeypatch):
    _, capacity, _ = prior
    before = pickle.dumps(capacity, protocol=5)

    def forbidden(*args, **kwargs):
        pytest.fail("prior forecast must not read the full source state or native law")

    for name in (
        "_stage",
        "_field",
        "evaluate_relational_exchange",
        "step_relational_exchange",
    ):
        monkeypatch.setattr(relational, name, forbidden)
    for module in (relational_sine_comparison, relational_sine_mediation):
        monkeypatch.setattr(module, "_capture_sine_state", forbidden)
    admission = owner.admit_sine_prior(capacity)
    result = _forecast(admission)
    assert result.admitted
    assert pickle.dumps(capacity, protocol=5) == before


def test_export_preserves_joint_admission_time_and_uncertainty(prior):
    _, _, admission = prior
    result = _forecast(admission)
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    report = payload["report"]
    assert report["prior_admission"]["status"] == "admitted"
    assert report["freeze_hidden"] is False
    assert report["observation_time"] == {"numerator": 3, "denominator": 2}
    assert report["forecast_start"] == {
        "numerator": START.numerator,
        "denominator": START.denominator,
    }
    assert report["initial_box"][-1]["lo"] != report["initial_box"][-1]["hi"]
