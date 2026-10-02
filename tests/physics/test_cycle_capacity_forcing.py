"""Independent arithmetic controls for the conditional capacity tube.

Small static graphs exercise the report; the retained post-UM execution and
actual eligibility evidence belong to test_cycle_capacity_synergies.
"""

import copy
from fractions import Fraction as F

import mpmath as mp
import networkx as nx
import pytest

from tests.joint_phase_helpers import configure, triangle
from tnfr.physics.cycle_relaxation import (
    bound_cycle_capacity_forcing,
    bound_cycle_relaxation,
)
from tnfr.physics.forcing_realization import observe_forcing_dirichlet_balance


def _mp(value):
    value = F(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def template():
    graph = nx.cycle_graph(5)
    configure(graph, weights={"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0.0})
    nx.set_edge_attributes(graph, 1.0, "weight")
    graph.edges[0, 4]["weight"] = 0.75
    for node, phase in enumerate((0.1, 0.125, 0.15, 0.1, 0.075)):
        graph.nodes[node].update(
            EPI=(node - 2) / 16, theta=phase, nu_f=1.0, delta_nfr=0.0
        )
    return bound_cycle_relaxation(
        graph, range(5), coupling_strength=F(1, 2), times=(0,)
    )


def test_rational_tube_encloses_independent_continuous_comparison(template):
    before = copy.deepcopy(template)
    report = bound_cycle_capacity_forcing(
        template,
        capacity_bounds=(1, 1 + F(1, 4096)),
        phase_radius=F(1, 2),
        times=(0, F(1, 8), 4),
    )
    assert report.admitted and report.admission_failure is None
    assert report.phase_tube_radius_upper < report.phase_radius
    assert template == before
    assert (
        report.initial_epi_dirichlet_energy
        == template.capture.snapshot.dirichlet_energy
    )
    assert (
        report.samples[0].epi_dirichlet_energy_upper
        == report.initial_epi_dirichlet_energy
    )
    assert (
        report.samples[0].gap_deviation_norm_upper
        == report.initial_gap_deviation_norm_upper
    )
    assert report.samples[0].non_epi_source_integral_upper == 0
    assert report.samples[0].epi_interval == (
        min(template.capture.snapshot.epi),
        max(template.capture.snapshot.epi),
    )
    with mp.workdps(100):
        width = mp.mpf(1) / 4096
        rho = _mp(report.phase_radius)
        # Direct spectrum of unweighted C5, independent of the exact owner.
        gamma = mp.cos(rho) * (2 - 2 * mp.cos(2 * mp.pi / 5)) / 4
        assert 0 < _mp(report.phase_decay_rate_lower_bound) <= gamma
        assert (
            _mp(report.reference_gap_deviation_upper)
            >= mp.sqrt(mp.mpf(5) / 2) * width / gamma
        )
        for sample in report.samples:
            time = _mp(sample.time)
            decay = mp.exp(-_mp(report.phase_decay_rate_lower_bound) * time)
            q = (
                _mp(report.phase_gap_tail_upper)
                + (
                    _mp(report.initial_gap_deviation_norm_upper)
                    - _mp(report.phase_gap_tail_upper)
                )
                * decay
            )
            form_decay = mp.exp(-_mp(report.epi_energy_decay_rate_lower_bound) * time)
            energy = (
                _mp(report.epi_energy_tail_upper)
                + (
                    _mp(report.initial_epi_dirichlet_energy)
                    - _mp(report.epi_energy_tail_upper)
                )
                * form_decay
            )
            assert _mp(sample.gap_deviation_norm_upper) + mp.mpf("1e-90") >= q
            assert _mp(sample.epi_dirichlet_energy_upper) + mp.mpf("1e-90") >= energy
            assert (
                sample.capacity_exposure_upper
                == report.capacity_difference_norm_upper * sample.time
            )
            assert _mp(sample.phase_source_norm_upper) >= q / mp.pi
            integrated_q = _mp(report.phase_gap_tail_upper) * time + (
                _mp(report.initial_gap_deviation_norm_upper)
                - _mp(report.phase_gap_tail_upper)
            ) * (1 - decay) / _mp(report.phase_decay_rate_lower_bound)
            assert (
                _mp(sample.gap_deviation_integral_upper) + mp.mpf("1e-90")
                >= integrated_q
            )
            weights = dict(template.capture.normalized_weights)
            source_integral = (
                _mp(weights["phase"]) * integrated_q / mp.pi
                + _mp(weights["vf"]) * _mp(report.capacity_difference_norm_upper) * time
            )
            assert (
                _mp(sample.non_epi_source_integral_upper) + mp.mpf("1e-90")
                >= source_integral
            )
            offset = report.capacity_bounds[1] * sample.non_epi_source_integral_upper
            assert sample.epi_interval == (
                min(template.capture.snapshot.epi) - offset,
                max(template.capture.snapshot.epi) + offset,
            )
    for previous, following in zip(report.samples, report.samples[1:]):
        assert following.epi_interval[0] <= previous.epi_interval[0]
        assert following.epi_interval[1] >= previous.epi_interval[1]
    # The existing signed ledger retains source cancellation. Its diffusion
    # term uses actual transport weights; no unit-cycle replacement occurs.
    ledger = observe_forcing_dirichlet_balance(template.capture)
    assert ledger.identity_residual == 0
    assert ledger.diffusion_rate < 0


def test_zero_spread_has_no_capacity_floor_and_does_not_claim_a_mean(template):
    report = bound_cycle_capacity_forcing(
        template, capacity_bounds=(1, 1), phase_radius=F(1, 2), times=(0, 4)
    )
    assert report.admitted
    assert report.capacity_difference_norm_upper == report.phase_gap_tail_upper == 0
    assert report.reference_gap_deviation_upper == 0
    assert (
        report.samples[-1].gap_deviation_norm_upper
        < report.samples[0].gap_deviation_norm_upper
    )
    assert all(sample.capacity_exposure_upper == 0 for sample in report.samples)


def test_increasing_envelopes_use_the_other_exponential_endpoint():
    graph = triangle(
        phase=(0.25, 0.25),
        epi=(0, 0, 0),
        weights={"phase": 0.5, "epi": 0.25, "vf": 0.25, "topo": 0},
    )
    reference = bound_cycle_relaxation(
        graph, range(3), coupling_strength=F(1, 2), times=(0,)
    )
    report = bound_cycle_capacity_forcing(
        reference,
        capacity_bounds=(1, 1 + F(1, 4096)),
        phase_radius=F(1, 2),
        times=(0, 4),
    )
    assert report.admitted
    assert (
        report.initial_gap_deviation_norm_upper
        == report.initial_epi_dirichlet_energy
        == 0
    )
    sample = report.samples[-1]
    assert sample.gap_deviation_norm_upper > 0
    assert sample.epi_dirichlet_energy_upper > 0
    with mp.workdps(100):
        for upper, tail, rate in (
            (
                sample.gap_deviation_norm_upper,
                report.phase_gap_tail_upper,
                report.phase_decay_rate_lower_bound,
            ),
            (
                sample.epi_dirichlet_energy_upper,
                report.epi_energy_tail_upper,
                report.epi_energy_decay_rate_lower_bound,
            ),
        ):
            exact = _mp(tail) * (1 - mp.exp(-4 * _mp(rate)))
            assert _mp(upper) >= exact
        gamma = _mp(report.phase_decay_rate_lower_bound)
        integrated_q = _mp(report.phase_gap_tail_upper) * (
            4 - (1 - mp.exp(-4 * gamma)) / gamma
        )
        assert _mp(sample.gap_deviation_integral_upper) >= integrated_q > 0


def test_zero_source_preserves_the_initial_extrema_without_a_new_solver():
    graph = triangle(phase=(0.25, 0.25), epi=(-0.25, 0.125, 0.5))
    reference = bound_cycle_relaxation(
        graph, range(3), coupling_strength=F(1, 2), times=(0,)
    )
    report = bound_cycle_capacity_forcing(
        reference, capacity_bounds=(1, 1), phase_radius=F(1, 2), times=(0, 4, 32)
    )
    assert report.admitted
    for sample in report.samples:
        assert sample.non_epi_source_integral_upper == 0
        assert sample.epi_interval == (F(-1, 4), F(1, 2))


def test_insufficient_margin_is_an_explicit_abstention(template):
    report = bound_cycle_capacity_forcing(
        template, capacity_bounds=(1, 2), phase_radius=F(1, 2), times=(0, 4)
    )
    assert not report.admitted
    assert report.admission_failure
    assert report.samples == ()
    assert report.phase_tube_radius_upper >= report.phase_radius


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity_bounds": (0, 1)},
        {"capacity_bounds": (2, 1)},
        {"capacity_bounds": (True, 2)},
        {"capacity_bounds": (1, float("inf"))},
        {"capacity_bounds": (1, 2, 3)},
        {"phase_radius": 0},
        {"phase_radius": 2},
        {"phase_radius": float("nan")},
        {"times": ()},
        {"times": (1, 0)},
        {"times": (-1,)},
    ],
)
def test_invalid_budget_domain_is_rejected(template, kwargs):
    arguments = dict(
        capacity_bounds=(1, 1 + F(1, 4096)), phase_radius=F(1, 2), times=(0, 4)
    )
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        bound_cycle_capacity_forcing(template, **arguments)
