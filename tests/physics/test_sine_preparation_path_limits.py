"""Independent terminal geometry, full-law stability and source-scope controls."""

from collections import Counter
from fractions import Fraction as Q
from itertools import combinations

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_preparation_path_limits as owner
from tnfr.physics import _sine_preparation_path_selection as path_owner


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    inputs = dict(
        parameter_interval=(Q(3, 4), Q(4, 5)),
        scaled_time=200,
        form_error_bound=Q(1, 10**10),
        phase_error_bound=Q(1, 10**10),
    )
    return owner._assess_preparation_path_limits(**(inputs | changes))


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _canonical(turns):
    return tuple(value % 1 for value in turns)


def _expected_candidates():
    states = {(Q(0),) * 9, (Q(1, 9),) * 9, (Q(8, 9),) * 9}
    for sign in (-1, 1):
        for special in range(9):
            states.add(
                _canonical(
                    tuple(
                        sign * (Q(3, 7) if edge == special else Q(1, 14))
                        for edge in range(9)
                    )
                )
            )
    for special in combinations(range(9), 2):
        states.add(tuple(Q(1, 2) if edge in special else Q(0) for edge in range(9)))
    return states


def test_exact_energy_filter_matches_independent_complete_candidate_families(
    report, mp
):
    assert len(report.critical_geometries) == 630
    actual = {
        _canonical(report.critical_geometries[i].edge_turns)
        for i in report.candidate_indices
    }
    assert actual == _expected_candidates()
    assert len(actual) == 57
    assert not report.unresolved_energy_indices
    counts = Counter(
        report.critical_geometries[i].relative_unstable_modes
        for i in report.candidate_indices
    )
    assert counts == {0: 3, 1: 18, 2: 36}
    assert len(report.attracting_candidate_indices) == 3
    assert len(report.saddle_candidate_indices) == 54
    assert report.attracting_candidate_windings == (-1, 0, 1)
    upper = _mp(mp, report.source.endpoint_total_storage_upper_bound)
    for branch in report.critical_geometries:
        energy = sum(
            1 - mp.cos(2 * mp.pi * _mp(mp, turn)) for turn in branch.edge_turns
        )
        assert (
            _mp(mp, branch.storage_bounds.lo) - mp.mpf("1e-75")
            <= energy
            <= _mp(mp, branch.storage_bounds.hi) + mp.mpf("1e-75")
        )
        assert (branch.energy_status == "excluded_by_storage") == (energy > upper)
        assert branch.phase_hessian_inertia[2] == 0
        assert branch.relative_stable_modes + branch.relative_unstable_modes == 16


def test_antipodal_critical_states_do_not_acquire_a_cut_dependent_winding(report):
    boundary = [
        report.critical_geometries[i]
        for i in report.saddle_candidate_indices
        if report.critical_geometries[i].principal_sine_turn == 0
    ]
    assert len(boundary) == 36
    for branch in boundary:
        assert branch.winding is None
        assert branch.storage_bounds.lo == branch.storage_bounds.hi == 4
        assert sum(turn == Q(1, 2) for turn in branch.edge_turns) == 2
        assert branch.phase_hessian_inertia == (6, 2, 0)


def test_direct_joint_jacobians_agree_with_inertia_for_both_attractors_and_saddles(
    report, mp
):
    # Independent full two-row linearization in coordinates with node8=-sum0..7.
    representatives = []
    for winding, negative in ((0, 0), (1, 0), (1, 1), (None, 2)):
        representatives.append(
            next(
                branch
                for branch in report.critical_geometries
                if branch.winding == winding
                and branch.relative_unstable_modes == negative
            )
        )
    gamma = 1 / (1023 * mp.pi)
    laplacian = mp.matrix(9)
    for i in range(9):
        laplacian[i, i] = 2
        laplacian[i, (i + 1) % 9] = laplacian[i, (i - 1) % 9] = -1
    for branch in representatives:
        hessian = mp.matrix(9)
        for i, turn in enumerate(branch.edge_turns):
            j = (i + 1) % 9
            weight = mp.cos(2 * mp.pi * _mp(mp, turn))
            hessian[i, i] += weight
            hessian[j, j] += weight
            hessian[i, j] -= weight
            hessian[j, i] -= weight
        jacobian = mp.matrix(16)
        for i in range(8):
            for j in range(8):
                a = (laplacian[i, j] - laplacian[i, 8]) / 2
                h = hessian[i, j] - hessian[i, 8]
                jacobian[i, j] = -a
                jacobian[i, j + 8] = -gamma * h / 2
                jacobian[i + 8, j] = gamma * a
        eigenvalues = mp.eig(jacobian, left=False, right=False)
        assert (
            sum(root.real > mp.mpf("1e-60") for root in eigenvalues)
            == branch.relative_unstable_modes
        )
        assert (
            sum(root.real < -mp.mpf("1e-60") for root in eigenvalues)
            == branch.relative_stable_modes
        )


def test_source_is_rebuilt_and_neither_candidate_count_nor_null_volume_selects_a_mode(
    report,
):
    assert report.source.parameter_interval == (Q(3, 4), Q(4, 5))
    assert report.source.scaled_time == 200
    assert report.source.winding_at_time == 2
    assert report.source.forced_winding_two_exit_certified
    assert report.source.captured_winding is None
    assert report.single_relative_equilibrium_convergence_certified
    assert report.full_lifted_state_convergence_certified
    assert report.saddle_source_volume_zero_at_each_parameter_certified
    assert report.selected_terminal_index is None
    assert report.selection_status == "unavailable_no_source_to_basin_selector"


@pytest.mark.parametrize("errors", [(0, 0), (0, Q(1, 10**10)), (Q(1, 10**10), 0)])
def test_lower_dimensional_preparations_do_not_inherit_null_exceptional_volume(errors):
    reduced = _assess(form_error_bound=errors[0], phase_error_bound=errors[1])
    assert reduced.single_relative_equilibrium_convergence_certified
    assert not reduced.saddle_source_volume_zero_at_each_parameter_certified
    assert len(reduced.saddle_candidate_indices) == 54
    assert reduced.selected_terminal_index is None


def test_zero_time_rebuilds_the_original_large_budget_without_later_exclusions():
    initial = _assess(scaled_time=0)
    assert len(initial.candidate_indices) == len(initial.critical_geometries) == 630
    assert initial.attracting_candidate_windings == (-2, -1, 0, 1, 2)
    assert not initial.source.forced_winding_two_exit_certified
    assert initial.single_relative_equilibrium_convergence_certified


def test_exact_threshold_equality_and_unresolved_intervals_retain_candidates():
    equal = owner._c9_terminal_geometries(4)
    below = owner._c9_terminal_geometries(4 - Q(1, 2**300))
    for a, b in zip(equal, below):
        if a.principal_sine_turn == 0 and a.supplementary_mask.bit_count() == 2:
            assert a.energy_status == "not_excluded_by_storage"
            assert b.energy_status == "excluded_by_storage"
    saddle = next(branch for branch in equal if branch.principal_sine_turn == Q(1, 14))
    midpoint = (saddle.storage_bounds.lo + saddle.storage_bounds.hi) / 2
    uncertain = owner._c9_terminal_geometries(midpoint)
    same = next(
        branch for branch in uncertain if branch.edge_turns == saddle.edge_turns
    )
    assert same.energy_status == "unresolved_storage_comparison"


@pytest.mark.parametrize("bad", [True, "4", 1j, float("inf"), float("nan"), -1])
def test_storage_admission_precedes_template_construction(monkeypatch, bad):
    def forbidden(*args):
        pytest.fail("invalid energy budget reached the exact catalog")

    monkeypatch.setattr(owner, "_odd_cycle_sine_templates", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._c9_terminal_geometries(bad)


@pytest.mark.parametrize(
    "change",
    [
        {"scaled_time": True},
        {"form_error_bound": float("nan")},
        {"phase_error_bound": -1},
        {"parameter_interval": (0, 1 + Q(1, 2**300))},
    ],
)
def test_primitive_source_admission_precedes_any_terminal_classification(
    monkeypatch, change
):
    def forbidden(*args):
        pytest.fail("invalid preparation reached the terminal catalog")

    monkeypatch.setattr(owner, "_c9_terminal_geometries", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**change)


def test_terminal_reader_recomputes_consumed_transit_instead_of_accepting_a_report(
    monkeypatch,
):
    original = owner._assess_preparation_path_selection
    calls = []

    def observed(**kwargs):
        calls.append(kwargs.copy())
        return original(**kwargs)

    monkeypatch.setattr(owner, "_assess_preparation_path_selection", observed)
    result = _assess()
    assert len(calls) == 1
    assert calls[0]["parameter_interval"] == result.source.parameter_interval
    with pytest.raises(TypeError):
        owner._assess_preparation_path_limits(source=result.source)
    assert isinstance(result.source, path_owner._PreparationPathSelection)
