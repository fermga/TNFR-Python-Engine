"""Admission of primitive phase as a regional form angle, not a selected law.

The existing directed-triangle generator and pressure/phase owners are reused.
All comparisons are instantaneous on fixed support and positive amplitudes;
no trajectory, default-law change or global polar continuation is asserted.
"""

import math

import networkx as nx
import numpy as np
import pytest

from tests.physics import test_coupled_directed_form_phase as directed_reference
from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import default_compute_delta_nfr, dnfr_phase_only
from tnfr.dynamics.phase_evolution import propose_u3_gated_phase_step

directed_model = directed_reference.model


@pytest.fixture(scope="module")
def regional_preparations(directed_model):
    s, adjacency, _, _, lift, _ = directed_model
    delta = s.pi / 6
    phases = (0.0,) * 3 + (float(delta),) * 3
    preparations = []
    for ratio in (1, 2):
        state = s.Matrix((1, 0, ratio * s.cos(delta), ratio * s.sin(delta))) / 16
        form = s.ones(6, 1) / 2 + lift * state
        graph = nx.from_numpy_array(
            np.asarray(adjacency, dtype=float), create_using=nx.DiGraph
        )
        graph.graph.update(
            DNFR_WEIGHTS={"epi": 0.5, "phase": 0.5, "vf": 0.0, "topo": 0.0},
            DELTA_PHI_MAX=math.pi / 2,
            UM_MAX_PHASE_DIFF=math.pi / 2,
        )
        for node in graph:
            graph.nodes[node].update(
                EPI=float(form[node]), theta=phases[node], nu_f=1.0
            )
        phase_graph = graph.copy()
        dnfr_phase_only(phase_graph)
        phase_gradient = tuple(
            get_attr(phase_graph.nodes[node], ALIAS_DNFR) for node in phase_graph
        )
        default_compute_delta_nfr(graph)
        pressure = tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph)
        preparations.append((graph, state, phase_gradient, pressure))
    return phases, tuple(preparations)


def test_nonzero_native_phase_source_changes_means_and_retains_form_angle_row(
    directed_model, regional_preparations
):
    s, _, generator, _, lift, reduced = directed_model
    e, weight, nu = s.symbols("e w nu", positive=True)
    delta = s.Symbol("delta", real=True)
    means = s.Matrix(s.symbols("mu0:2", real=True))
    state = s.Matrix(s.symbols("u0 v0 u1 v1", real=True))
    mean_lift = s.diag(s.ones(3, 1), s.ones(3, 1))
    # Each outgoing neighborhood contains one own-region and one other-region
    # phase. For |delta|<pi the rotated resultant is positive: its Arg is
    # exactly the midpoint, independently of the form amplitudes.
    centered_resultant = s.exp(-s.I * delta / 2) * (1 + s.exp(s.I * delta))
    assert s.simplify(s.expand_complex(centered_resultant) - 2 * s.cos(delta / 2)) == 0
    phase_source = mean_lift * s.Matrix((delta, -delta)) / (2 * s.pi)
    fine = mean_lift * means + lift * state
    fine_rate = nu * (e * generator * fine + weight * phase_source)
    contrast_rate = (lift.T * fine_rate).applyfunc(s.simplify)
    assert (contrast_rate - e * nu * reduced * state).applyfunc(s.simplify) == s.zeros(
        4, 1
    )
    observed_means = (mean_lift.T * fine_rate / 3).applyfunc(s.simplify)
    required_means = nu * (
        e * s.Matrix((means[1] - means[0], means[0] - means[1])) / 2
        + weight * s.Matrix((delta, -delta)) / (2 * s.pi)
    )
    assert (observed_means - required_means).applyfunc(s.simplify) == s.zeros(2, 1)

    r0, r1 = s.symbols("r0 r1", positive=True)
    psi0, psi1 = s.symbols("psi0 psi1", real=True)
    polar = s.Matrix(
        (r0 * s.cos(psi0), r0 * s.sin(psi0), r1 * s.cos(psi1), r1 * s.sin(psi1))
    )
    polar_rate = contrast_rate.subs(dict(zip(state, polar, strict=True)))
    required_phase_rows = (
        -s.sqrt(3) * e * nu / 4 + e * nu * r1 * s.sin(psi1 - psi0) / (2 * r0),
        -s.sqrt(3) * e * nu / 4 + e * nu * r0 * s.sin(psi0 - psi1) / (2 * r1),
    )
    for region, radius in enumerate((r0, r1)):
        start = 2 * region
        vector = polar[start : start + 2, 0]
        velocity = polar_rate[start : start + 2, 0]
        observed_angle_rate = s.det(s.Matrix.hstack(vector, velocity)) / radius**2
        assert s.trigsimp(observed_angle_rate - required_phase_rows[region]) == 0
    # This is the necessary tangency row for theta=arg(z), not a proof that
    # an independently supplied primitive phase model satisfies it.

    _, preparations = regional_preparations
    for graph, _, phase_gradient, pressure in preparations:
        assert tuple(graph.nodes[node]["nu_f"] for node in graph) == (1,) * 6
        assert phase_gradient == pytest.approx(
            (1 / 12,) * 3 + (-1 / 12,) * 3, abs=3e-16
        )
        source = s.Matrix([s.Rational(value) for value in phase_gradient]) / 2
        assert source != s.zeros(6, 1)
        assert (lift.T * source).applyfunc(s.simplify) == s.zeros(4, 1)
        source_means = mean_lift.T * source / 3
        assert tuple(map(float, source_means)) == pytest.approx(
            (1 / 24, -1 / 24), abs=3e-16
        )
        # The reversible forcing observer is not admitted on this directed
        # graph. Compare the actual graph-facing pressure with the independent
        # directed generator and retain materialization/assembly residuals.
        retained_form = s.Matrix(
            [s.Rational(graph.nodes[node]["EPI"]) for node in graph]
        )
        fresh = s.Matrix([s.Rational(value) for value in pressure])
        inherited = generator * retained_form / 2
        residual = fresh - inherited - source
        assert max(abs(float(value)) for value in residual) < 1e-15
        assert tuple(map(float, lift.T * (fresh - inherited))) == pytest.approx(
            (0.0,) * 4, abs=1e-15
        )
        assert tuple(
            map(float, mean_lift.T * (fresh - inherited) / 3)
        ) == pytest.approx(tuple(map(float, source_means)), abs=1e-15)


def test_phase_only_writer_cannot_preserve_identification_at_two_amplitude_ratios(
    directed_model, regional_preparations
):
    s, _, _, _, _, reduced = directed_model
    phases, preparations = regional_preparations
    required_rates, proposals = [], []
    for graph, state, _, _ in preparations:
        # The first control proves the nonzero phase source has no contrast
        # projection, so this is the angle rate of the complete nodal row.
        velocity = reduced * state / 2  # fixed e=1/2 and nu=1
        angles = []
        for start in (0, 2):
            vector, rate = state[start : start + 2, 0], velocity[start : start + 2, 0]
            assert vector.dot(vector) > 0
            angles.append(s.det(s.Matrix.hstack(vector, rate)) / vector.dot(vector))
        required_rates.append(s.simplify(angles[1] - angles[0]))
        proposals.append(
            propose_u3_gated_phase_step(
                graph,
                tuple(graph),
                phases,
                (1.0,) * 6,
                dt=1 / 64,
                coupling_strength=0.5,
            )
        )
    assert required_rates == [-s.Rational(1, 4), -s.Rational(5, 16)]
    assert required_rates[1] - required_rates[0] == -s.Rational(1, 16)
    np.testing.assert_array_equal(proposals[0], proposals[1])
    # The declared strength happens to match the first state's relative row;
    # the impossibility for BOTH states is independent of that chosen value.
    realized_rate = ((proposals[0][3] - proposals[0][0]) - (phases[3] - phases[0])) * 64
    assert realized_rate == pytest.approx(-0.25, abs=8e-15)
    assert abs(realized_rate - float(required_rates[1])) > 0.06
    assert all(0 <= phase < math.pi / 2 for phase in phases)
    # No graph evolution occurred: this API returns a detached proposal.
    for graph, _, _, _ in preparations:
        assert tuple(graph.nodes[node]["theta"] for node in graph) == phases

    # A form angle is unavailable at zero regional amplitude although the
    # inherited Cartesian coupling still supplies a finite nonzero response.
    zero_recipient = s.Matrix((0, 0, s.Rational(1, 16), 0))
    assert zero_recipient[:2, 0].dot(zero_recipient[:2, 0]) == 0
    assert (reduced * zero_recipient / 2)[:2, 0] == s.Matrix((s.Rational(1, 64), 0))
