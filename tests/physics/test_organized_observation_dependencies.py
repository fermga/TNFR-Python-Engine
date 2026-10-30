"""Cold observation paths admit exact records without an acquisition oracle."""

import subprocess
import sys
from decimal import Decimal
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.physics._sine_preparation import _prepared_coordinate_vector


@pytest.mark.parametrize("path", ("matched", "sequential", "invalid_source"))
def test_cold_observation_does_not_import_acquisition(path, source_tree_environment):
    # A fresh process makes the boundary independent of pytest collection and
    # fails on an attempted import, even if a caller would swallow its error.
    script = """
import importlib.abc
import sys
from fractions import Fraction as Q

attempted = []
forbidden = (
    'tnfr.physics._organized_outer_winding',
    'tnfr.physics._repeated_organized_composition',
    'tnfr.physics._sine_constitutive_',
    'tnfr.physics._sine_class_constitutive_',
)

class WithoutAcquisition(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(forbidden):
            attempted.append(fullname)
            raise AssertionError('observation imported acquisition: ' + fullname)

sys.meta_path.insert(0, WithoutAcquisition())
assert not any(name.startswith(forbidden) for name in sys.modules)
mode = sys.argv[1]
if mode == 'sequential':
    from tnfr.physics import _organized_sequential_observation as owner
    assert owner._assess_organized_sequential_observation_protocol().admitted
    result = owner._infer_sequential_outer_winding(readings=((0,) * 5,) * 5)
    form, phase = owner._apply_sequential_observation_event(
        form=(0,) * 45, phase_offsets=(0,) * 45,
        marked_child=2, event_index=1,
    )
    assert phase[18] == -Q(4, 25) and sum(phase) == 0
else:
    from tnfr.physics import _organized_outer_observation as owner
    if mode == 'matched':
        assert owner._assess_organized_outer_observation_protocol().admitted
        result = owner._infer_organized_outer_winding(
            unprobed_means=(0,) * 5, positive_probe_means=(0,) * 5,
            negative_probe_means=(0,) * 5,
        )
        form, phase = owner._apply_outer_observation_probe(
            form=(0,) * 45, phase_offsets=(0,) * 45, marked_child=2, sign=1,
        )
        assert phase[18] == Q(2, 25) and sum(phase) == 0
    else:
        for bad in ((False,) * 5, (0,) * 6, (Q(1, 10**9),) * 5):
            try:
                owner._assess_outer_observation_source_inclusion(
                    form_error_bounds=bad, phase_error_bounds=(0,) * 5,
                )
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError('invalid source admitted')
if mode != 'invalid_source':
    assert result.identified_winding is None
    assert result.status == 'incompatible_records_or_premises'
    assert form == (0,) * 45
assert not attempted
assert not any(name.startswith(forbidden) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, path],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_registered_vector_retains_exact_and_represented_values():
    tiny, huge = Q(1, 10**500), Q(10**500) + Q(1, 7)
    raw = [tiny, huge, Q(1, 8), 0.1, -3]
    assert _prepared_coordinate_vector(iter(raw), "readings", 5) == (
        tiny,
        huge,
        Q(1, 8),
        Q.from_float(0.1),
        Q(-3),
    )
    assert raw == [tiny, huge, Q(1, 8), 0.1, -3]


@pytest.mark.parametrize(
    "path", ("preparation", "prepared", "pulse", "window", "recovery")
)
def test_cold_collective_proofs_keep_acquisition_and_observation_boundaries(
    path, source_tree_environment
):
    # Run outside any evidence fixture: preloaded research modules would hide
    # accidental dependencies, and a generic guard deliberately imports them.
    script = """
import importlib.abc
import sys
from fractions import Fraction as Q

mode = sys.argv[1]
attempted = []
forbidden = (
    'tnfr.research',
    'tnfr.mathematics._validated',
    'tnfr.physics._sine_flow',
    'tnfr.physics._organized_finite_observation_source',
    'tnfr.physics._organized_robust_current',
    'tnfr.physics._organized_outer_winding',
)
detached_owners = {
    'prepared': 'tnfr.physics._organized_effective_law',
    'pulse': 'tnfr.physics._organized_loaded_pulse',
    'window': 'tnfr.physics._organized_window_observation',
    'recovery': 'tnfr.physics._organized_recovery_observation',
}
allowed = {'preparation': (), 'prepared': ('prepared',), 'pulse': ('pulse',),
           'window': ('pulse', 'window'), 'recovery': ('pulse', 'recovery')}[mode]
forbidden += tuple(module for name, module in detached_owners.items() if name not in allowed)

class WithoutScientificDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(forbidden):
            attempted.append(fullname)
            raise AssertionError('detached proof imported an unrelated owner: ' + fullname)

sys.meta_path.insert(0, WithoutScientificDependencies())
assert not any(name.startswith(forbidden) for name in sys.modules)
from tnfr import _exact_time
if mode != 'window':
    def no_exponential(*args, **kwargs):
        raise AssertionError('energy or raw-area proof evaluated an exponential bound')
    _exact_time.exp_unit_bounds = no_exponential

if mode == 'preparation':
    from tnfr.physics._organized_c5_preparation import prepared_c5_initial_energy_upper
    assert prepared_c5_initial_energy_upper(response_upper=Q(1, 8), preparation_radius=Q(1, 4)) == Q(1, 2)
elif mode == 'prepared':
    from tnfr.physics._organized_effective_law import bound_prepared_matrix_response
    result = bound_prepared_matrix_response()
    assert result.identity_certified and result.response_separated
elif mode == 'pulse':
    from tnfr.physics._organized_loaded_pulse import bound_loaded_c5_pulse
    result = bound_loaded_c5_pulse(
        response_lower=Q(1, 100), response_upper=Q(1, 10),
        amplitude=Q(1, 1024), horizon=Q(16), preparation_radius=0,
        sensor_preparation_radius=0, sensor_coupling=Q(1, 64),
    )
    assert result.identity_certified and result.energy_upper == Q(1, 32768)
elif mode == 'window':
    from tnfr.physics._organized_window_observation import bound_loaded_window_observation
    result = bound_loaded_window_observation()
    assert result.identity_certified and result.response_separated
    assert result.phase_blind_records_overlap
else:
    from tnfr.physics._organized_recovery_observation import bound_recovery_area_observation
    result = bound_recovery_area_observation()
    assert result.identity_certified and result.response_gap > Q(1, 2**14)
assert not attempted
assert not any(name.startswith(forbidden) for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, path],
        cwd=Path(__file__).resolve().parents[2],
        env=source_tree_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "bad", (True, "0", None, float("nan"), float("inf"), complex(0), Decimal("0.125"))
)
def test_registered_vector_rejects_invalid_primitive(bad):
    with pytest.raises((TypeError, ValueError)):
        _prepared_coordinate_vector((Q(1, 7), bad), "readings", 2)


@pytest.mark.parametrize("bad", ("01", b"01", {0, 1}, {0: 0, 1: 1}, None))
def test_registered_vector_rejects_unordered_or_noniterable_input(bad):
    with pytest.raises(TypeError, match="readings must be an ordered iterable"):
        _prepared_coordinate_vector(bad, "readings", 2)


def test_registered_vector_detects_excess_without_exhausting_a_stream():
    consumed = []

    def stream():
        for value in range(3):
            consumed.append(value)
            yield value
        pytest.fail("registered vector consumed beyond its one-item excess bound")

    with pytest.raises(ValueError, match="readings requires 2 registered coordinates"):
        _prepared_coordinate_vector(stream(), "readings", 2)
    assert consumed == [0, 1, 2]


def test_registered_vector_rejects_missing_coordinates():
    with pytest.raises(ValueError, match="readings requires 2 registered coordinates"):
        _prepared_coordinate_vector((1,), "readings", 2)
