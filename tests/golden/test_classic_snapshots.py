"""Golden snapshots validating the classical runtime remains stable."""
from __future__ import annotations

import pytest

from tnfr.operators.definitions import Coherence, Emission, Reception, Resonance, Transition

from tests.helpers.compare_classical import classical_operator_snapshot

def test_classic_runtime_sequence_matches_golden_snapshot() -> None:
    ops = [Emission(), Reception(), Coherence(), Resonance(), Transition()]
    snapshot = classical_operator_snapshot(ops)

    seed = snapshot["classic-seed"]
    partner = snapshot["classic-partner"]

    assert seed["EPI"] == pytest.approx(0.723125)
    assert seed["vf"] == pytest.approx(1.2)
    assert seed["theta"] == pytest.approx(0.1)
    assert seed["dnfr"] == pytest.approx(-0.2615625)

    assert partner["EPI"] == pytest.approx(0.5)
    assert partner["vf"] == pytest.approx(0.9)
    assert partner["theta"] == pytest.approx(0.0)
    assert partner["dnfr"] == pytest.approx(0.2615625)
