"""Snapshot matching cannot invent missing observations or certify invalid data."""

import copy
import sys
from fractions import Fraction

import pytest

from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.operators.remesh import StructuralIdentity
from tnfr.types import ensure_bepi, serialize_bepi_json


def _state():
    return {ALIAS_EPI[0]: -0.25, ALIAS_VF[0]: 1.0, ALIAS_THETA[0]: 0.0}


@pytest.mark.parametrize("aliases", (ALIAS_EPI, ALIAS_VF, ALIAS_THETA))
def test_missing_consumed_channel_cannot_match(aliases):
    state = _state()
    del state[aliases[0]]
    identity = StructuralIdentity(-0.25, (0.5, 1.5), phase_pattern=0.0)
    assert not identity.matches(state)
    if aliases == ALIAS_THETA:
        captured = StructuralIdentity.capture_from_node(state)
        assert captured.phase_pattern is None
        assert captured.matches(state)
    else:
        with pytest.raises(TNFRValueError, match="requires EPI and capacity"):
            StructuralIdentity.capture_from_node(state)


@pytest.mark.parametrize("aliases", (ALIAS_EPI, ALIAS_VF, ALIAS_THETA))
@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), Fraction(1, 10**400))
)
def test_invalid_authoritative_observation_cannot_fall_through(aliases, bad):
    state = _state()
    valid = state[aliases[0]]
    state[aliases[0]] = bad
    if len(aliases) > 1:
        state[aliases[1]] = valid
    identity = StructuralIdentity(-0.25, (0.5, 1.5), phase_pattern=0.0)
    before_keys = tuple(state)
    with pytest.raises(TNFRValueError):
        identity.matches(state)
    with pytest.raises(TNFRValueError):
        StructuralIdentity.capture_from_node(state)
    assert tuple(state) == before_keys
    assert state[aliases[0]] is bad


@pytest.mark.parametrize(
    "changes",
    (
        {"epi_signature": float("nan")},
        {"epi_signature": True},
        {"phase_pattern": float("inf")},
        {"phase_pattern": False},
        {"tolerance": float("nan")},
        {"tolerance": float("inf")},
        {"tolerance": -0.1},
        {"tolerance": True},
        {"vf_range": (-1.0, 2.0)},
        {"vf_range": (0.0, float("inf"))},
        {"vf_range": (True, 2.0)},
        {"vf_range": (2.0, 1.0)},
        {"vf_range": (1 + Fraction(1, 2**60), 1)},
        {"vf_range": (0.0,)},
        {"frozen_by_sha": "unknown"},
    ),
)
def test_invalid_signature_rejects_on_construction_and_after_mutation(changes):
    options = {"epi_signature": -0.25, "vf_range": (0.5, 1.5)}
    with pytest.raises(TNFRValueError):
        StructuralIdentity(**(options | changes))
    identity = StructuralIdentity(**options)
    for key, value in changes.items():
        setattr(identity, key, value)
    with pytest.raises(TNFRValueError):
        identity.matches(_state())


@pytest.mark.parametrize("bad", (True, -1, float("nan"), float("inf")))
def test_tolerance_override_cannot_certify_invalid_threshold(bad):
    identity = StructuralIdentity(-0.25, (0.5, 1.5))
    with pytest.raises(TNFRValueError):
        identity.matches(_state(), tolerance=bad)


def test_signed_bepi_capture_and_matching_preserve_form_and_inputs():
    state = _state()
    state[ALIAS_EPI[0]] = serialize_bepi_json(ensure_bepi(-0.25))
    before = copy.deepcopy(state)
    captured = StructuralIdentity.capture_from_node(state)
    assert captured.epi_signature == -0.25
    assert captured.matches(state)
    assert state == before
    state[ALIAS_EPI[0]] = ensure_bepi(0.25)
    assert not captured.matches(state)
    state[ALIAS_EPI[0]] = BEPIElement((0.25, -0.25), (0.0, 0.0), (0.0, 1.0))
    with pytest.raises(TNFRValueError):
        captured.matches(state)


def test_unconsumed_phase_is_optional_but_capture_rejects_invalid_present_phase():
    state = _state()
    state[ALIAS_THETA[0]] = float("nan")
    assert StructuralIdentity(-0.25, (0.5, 1.5)).matches(state)
    with pytest.raises(TNFRValueError):
        StructuralIdentity.capture_from_node(state)


def test_low_capacity_label_is_not_provenance_or_permission_for_negative_capacity():
    identity = StructuralIdentity(-0.25, (0.5, 1.5), frozen_by_sha=True)
    state = _state()
    state[ALIAS_VF[0]] = 0.0
    assert identity.matches(state)
    identity.frozen_by_sha = "false"
    assert not identity.matches(state)
    identity.frozen_by_sha = True
    state[ALIAS_VF[0]] = -0.01
    with pytest.raises(TNFRValueError):
        identity.matches(state)


def test_large_finite_differences_and_circular_representatives_are_admitted():
    maximum = sys.float_info.max
    identity = StructuralIdentity(maximum, (0.0, maximum), tolerance=maximum)
    state = {ALIAS_EPI[0]: -maximum, ALIAS_VF[0]: maximum}
    assert not identity.matches(state)
    state[ALIAS_EPI[0]] = maximum
    state[ALIAS_THETA[0]] = maximum
    identity.phase_pattern = maximum
    identity.tolerance = 0.0
    assert identity.matches(state)
    state[ALIAS_THETA[0]] = -maximum
    with pytest.raises(TNFRValueError, match="phase differences must be finite"):
        identity.matches(state)


def test_tiny_negative_phase_difference_cannot_be_erased_by_separate_modulo():
    identity = StructuralIdentity(-0.25, (0.5, 1.5), phase_pattern=-2e-16, tolerance=0)
    state = _state()
    state[ALIAS_THETA[0]] = -1e-16
    assert not identity.matches(state)
    assert identity.matches(state, tolerance=2e-16)


def test_snapshot_match_is_read_only_and_does_not_authenticate_sha_history():
    state = _state()
    state["glyph_history"] = ["AL"]
    before = copy.deepcopy(state)
    captured = StructuralIdentity.capture_from_node(state, is_sha_frozen="false")
    signature = copy.deepcopy(captured)
    assert not captured.frozen_by_sha
    assert captured.lineage == []
    assert captured.matches(state)
    assert state == before
    assert captured == signature
