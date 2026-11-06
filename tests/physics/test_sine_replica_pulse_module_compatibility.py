"""Stable public imports and report identity after pulse-owner extraction."""

import pickle
import subprocess
import sys
from typing import get_type_hints

import pytest

from tnfr.physics import relational_sine_replica_pulse as pulse
from tnfr.physics import relational_sine_scale as scale

REPORT_NAMES = (
    "SineReplicaPulseAssessment",
    "SineReplicaPulseVariation",
    "SineReplicaPulseWorkResponse",
    "SineReplicaPulseFiniteWorkResponse",
    "SineReplicaStiffnessTraceCurve",
    "SineReplicaPulseSplitting",
)
PUBLIC_NAMES = REPORT_NAMES + (
    "assess_sine_replica_pulse",
    "assess_sine_replica_pulse_variation",
    "assess_sine_replica_pulse_work_response",
    "assess_sine_replica_pulse_finite_work_response",
    "assess_sine_replica_stiffness_trace_curve",
    "assess_sine_replica_pulse_splitting",
)


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_scale_reexports_the_identical_canonical_pulse_object(name):
    assert name in pulse.__all__
    assert name in scale.__all__
    assert getattr(scale, name) is getattr(pulse, name)
    assert getattr(pulse, name).__module__ == pulse.__name__


@pytest.mark.parametrize("name", REPORT_NAMES)
def test_historical_pickle_class_lookup_resolves_through_scale(name):
    # Locally constructed protocol-zero GLOBAL bytes exercise the old lookup.
    old_lookup = f"ctnfr.physics.relational_sine_scale\n{name}\n.".encode("ascii")
    assert pickle.loads(old_lookup) is getattr(pulse, name)


def test_new_report_pickle_uses_the_canonical_owner_and_round_trips():
    report = pulse.assess_sine_replica_stiffness_trace_curve(
        trace_bounds=(1, 2, 3), determinant_bounds=(1, 4, 9)
    )
    payload = pickle.dumps(report, protocol=0)
    assert (
        b"ctnfr.physics.relational_sine_replica_pulse\nSineReplicaStiffnessTraceCurve\n"
        in payload
    )
    restored = pickle.loads(payload)
    assert type(restored) is pulse.SineReplicaStiffnessTraceCurve
    assert restored == report


def test_future_annotations_resolve_in_the_canonical_owner():
    for name in PUBLIC_NAMES:
        get_type_hints(getattr(pulse, name))
    assert (
        get_type_hints(pulse.SineReplicaPulseVariation)["pulse"]
        is pulse.SineReplicaPulseAssessment
    )
    assert (
        get_type_hints(pulse.SineReplicaPulseFiniteWorkResponse)["tangent"]
        is pulse.SineReplicaPulseWorkResponse
    )


def test_canonical_pulse_import_does_not_load_scale_facade(source_tree_environment):
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tnfr.physics.relational_sine_replica_pulse; "
            "assert 'tnfr.physics.relational_sine_scale' not in sys.modules",
        ],
        env=source_tree_environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
