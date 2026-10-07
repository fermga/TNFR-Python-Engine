"""Public import and report identity contracts after pair-owner extraction."""

import pickle
from fractions import Fraction
from typing import get_type_hints

import pytest

from tnfr.physics import relational_sine_pair as pair
from tnfr.physics import relational_sine_scale as scale

PUBLIC_NAMES = (
    "SineGlobalPairState",
    "SinePairCancellationObservation",
    "SinePairFiniteExchange",
    "SinePairReceiverReadout",
    "SinePairReceiverConfounding",
    "derive_sine_global_pair_state",
    "evaluate_sine_global_pair_state",
    "observe_sine_pair_cancellation",
    "assess_sine_pair_finite_exchange",
    "assess_sine_pair_receiver_readout",
    "assess_sine_pair_receiver_confounding",
)
REPORT_NAMES = PUBLIC_NAMES[:5]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_scale_imports_are_identical_canonical_pair_objects(name):
    assert name in pair.__all__
    assert name in scale.__all__
    assert getattr(scale, name) is getattr(pair, name)
    assert getattr(pair, name).__module__ == pair.__name__


@pytest.mark.parametrize("name", REPORT_NAMES)
def test_historical_pickle_global_class_lookup_resolves_through_scale(name):
    # A protocol-zero GLOBAL encodes the original module and class lookup.
    # These local bytes exercise compatibility without an untrusted pickle.
    old_lookup = f"ctnfr.physics.relational_sine_scale\n{name}\n.".encode("ascii")
    assert pickle.loads(old_lookup) is getattr(pair, name)


def test_new_report_pickle_uses_the_canonical_owner_and_round_trips():
    report = pair.derive_sine_global_pair_state((0,) * 10, ((1, 0),) * 10)
    payload = pickle.dumps(report, protocol=0)
    assert b"ctnfr.physics.relational_sine_pair\nSineGlobalPairState\n" in payload
    restored = pickle.loads(payload)
    assert type(restored) is pair.SineGlobalPairState
    assert restored == report


def test_future_annotations_resolve_in_the_canonical_owner():
    for name in PUBLIC_NAMES:
        get_type_hints(getattr(pair, name))
    state_hints = get_type_hints(pair.SineGlobalPairState)
    assert state_hints["form_means"] == tuple[Fraction, ...]
    response_hints = get_type_hints(pair.SinePairFiniteExchange)
    assert (
        response_hints["comparison_initial_states"]
        == tuple[pair.SineGlobalPairState, ...]
    )
    assert (
        get_type_hints(pair.derive_sine_global_pair_state)["return"]
        is pair.SineGlobalPairState
    )
