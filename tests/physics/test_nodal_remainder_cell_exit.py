"""Budgeted first-cell replay preserves carry and rejects work beforehand."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
import math

import pytest

from tnfr.dynamics._euler_kernel import (
    NodalRemainderState, advance_nodal_remainder, initialize_nodal_remainder,
)
from tnfr.physics import nodal_remainder as owner


def _source():
    return initialize_nodal_remainder((.5,))


def _observe(**changes):
    inputs = dict(state=_source(), timestep=.0625, capacity=(1.,),
                  pressure=(2.**-50,), step_budget=2)
    inputs.update(changes)
    return owner.observe_nodal_remainder_cell_exit(**inputs)


def _forbid_replay(monkeypatch):
    calls = []

    def forbidden(**kwargs):
        calls.append(kwargs)
        raise AssertionError("replay must not begin")

    monkeypatch.setattr(owner, "observe_nodal_remainder_sequence", forbidden)
    return calls


def test_first_exit_matches_independent_shared_kernel_steps_and_all_prefixes():
    observed = _observe()
    horizon, sequence = observed.horizon, observed.sequence
    first = advance_nodal_remainder(_source(), timestep=.0625, capacity=(1.,), pressure=(2.**-50,))
    second = advance_nodal_remainder(first.after, timestep=.0625, capacity=(1.,), pressure=(2.**-50,))
    assert horizon.max_unchanged_steps == 1 and horizon.first_exit_step == 2
    assert sequence.steps == (first, second)
    assert sequence.steps[0].after.epi == (.5,)
    assert sequence.endpoint.epi == (math.nextafter(.5, math.inf),)
    assert sequence.steps[-1].before.exact_epi == horizon.unchanged_endpoint
    assert sequence.endpoint.exact_epi == horizon.first_exit_exact
    assert all(prefix.identity_residual == (0,) for prefix in sequence.prefixes)
    assert sequence.prefixes[-1].cumulative_nodal_area == (F(1, 2**53),)
    with pytest.raises(FrozenInstanceError):
        observed.sequence = None


def test_nonzero_inherited_carry_changes_the_exit_without_resetting_it():
    state = NodalRemainderState((.5,), (F(1, 2**54),), .05, 1.)
    observed = _observe(state=state, step_budget=1)
    assert observed.horizon.first_exit_step == 1
    assert observed.sequence.initial is state
    assert observed.sequence.steps[0].before.remainder == state.remainder
    assert observed.sequence.endpoint.epi == (math.nextafter(.5, math.inf),)
    with pytest.raises(ValueError, match="step_budget"):
        _observe(state=_source(), step_budget=1)
    assert state.exact_epi == (F(.5) + F(1, 2**54),)


def test_one_step_exit_with_zero_unchanged_prefix_is_supported():
    result = _observe(pressure=(.25,), step_budget=1)
    assert result.horizon.max_unchanged_steps == 0
    assert result.horizon.unchanged_endpoint == result.sequence.initial.exact_epi
    assert len(result.sequence.steps) == 1
    assert result.sequence.endpoint.epi == (.515625,)


def test_exact_budget_is_consumed_by_one_shared_sequence_call(monkeypatch):
    original = owner.observe_nodal_remainder_sequence
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(owner, "observe_nodal_remainder_sequence", record)
    result = _observe()
    assert len(calls) == 1
    assert calls[0]["initial"] is result.horizon.state
    assert calls[0]["timesteps"] == (.0625, .0625)
    assert calls[0]["capacities"] == ((1.,), (1.,))
    assert calls[0]["pressures"] == ((2.**-50,),) * 2


def test_budget_rejection_precedes_even_an_astronomical_input_allocation(monkeypatch):
    calls = _forbid_replay(monkeypatch)
    minimum = math.ulp(0.)
    with pytest.raises(ValueError, match="step_budget"):
        _observe(timestep=minimum, capacity=(minimum,), pressure=(minimum,), step_budget=256)
    assert not calls


@pytest.mark.parametrize("changes", ({"timestep": 0.}, {"capacity": (0.,)}, {"pressure": (0.,)}))
def test_stationary_horizon_is_rejected_before_replay(changes, monkeypatch):
    calls = _forbid_replay(monkeypatch)
    with pytest.raises(ValueError, match="no finite cell exit"):
        _observe(**changes)
    assert not calls


def test_hidden_exact_band_exit_is_rejected_before_replay(monkeypatch):
    calls = _forbid_replay(monkeypatch)
    state = initialize_nodal_remainder((1.,))
    horizon = owner.derive_nodal_remainder_cell_horizon(
        state=state, timestep=.0625, capacity=(1.,), pressure=(2.**-50,),
    )
    assert horizon.first_exit_step == 1
    assert horizon.first_exit_leaves_band == (True,)
    assert horizon.first_exit_leaves_cell == (False,)
    with pytest.raises(ValueError, match="EPI band"):
        _observe(state=state, step_budget=1)
    assert not calls
    assert state.exact_epi == (F(1),)


@pytest.mark.parametrize("budget,error", (
    (0, ValueError), (-1, ValueError), (True, TypeError), (False, TypeError),
    (1., TypeError), (F(1), TypeError), ("2", TypeError),
))
def test_budget_must_be_an_actual_positive_integer_before_any_replay(budget, error, monkeypatch):
    calls = _forbid_replay(monkeypatch)
    with pytest.raises(error, match="step_budget"):
        _observe(step_budget=budget)
    assert not calls


@pytest.mark.parametrize("changes,error", (
    ({"state": None}, TypeError), ({"timestep": -.0625}, ValueError),
    ({"capacity": (-1.,)}, ValueError), ({"capacity": (1., 1.)}, ValueError),
    ({"pressure": (math.nan,)}, ValueError), ({"pressure": [1.]}, TypeError),
))
def test_invalid_nodal_inputs_fail_before_replay(changes, error, monkeypatch):
    calls = _forbid_replay(monkeypatch)
    with pytest.raises(error):
        _observe(**changes)
    assert not calls


@pytest.mark.parametrize("fault", ("before_visible", "intermediate_visible", "last_before_exact", "final_exact"))
def test_horizon_binding_rejects_an_inconsistent_returned_prefix(fault, monkeypatch):
    sequence = _observe().sequence
    steps = list(sequence.steps)
    changed = initialize_nodal_remainder((.6,))
    if fault == "before_visible":
        steps[0] = replace(steps[0], before=changed)
    elif fault == "intermediate_visible":
        steps[0] = replace(steps[0], after=changed)
    elif fault == "last_before_exact":
        steps[-1] = replace(steps[-1], before=sequence.initial)
    else:
        steps[-1] = replace(steps[-1], after=changed)
    forged = replace(sequence, steps=tuple(steps))
    monkeypatch.setattr(owner, "observe_nodal_remainder_sequence", lambda **kwargs: forged)
    with pytest.raises(RuntimeError, match="analytic first cell exit"):
        _observe()
