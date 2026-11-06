"""Scope of the optional pressure/phase completion on the retained prism.

No integration is executed. Exact fresh-pressure derivatives are compared
with the existing optional scalar/flux pipeline; represented square roots
and trigonometric evaluations retain their ordinary numerical tolerances.
The final fresh-pressure substitution is a hypothetical comparison, not the
implemented optional runtime, a selected law, or an invariant-domain claim.
"""

import math
from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    P,
    _apply,
    _exact_generator,
    _graph,
)
from tnfr.dynamics.canonical import compute_extended_nodal_system
from tnfr.dynamics.integrators import (
    _estimate_local_coupling_strength,
    compute_flux_divergence_vectorized,
)
from tnfr.physics.extended import compute_dnfr_flux, compute_phase_current
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import (
    derive_joint_nodal_response,
    derive_phase_response,
)


def test_optional_pressure_contrast_grows_against_the_fresh_epi_chain_rule():
    graph = _graph((Q(-1, 4), 0, Q(-1, 4), 0))
    captured = capture_non_epi_forcing(graph)
    for node, value in zip(NODES, captured.full_kernel_pressure, strict=True):
        graph.nodes[node]["delta_nfr"] = float(value)
    captured = capture_non_epi_forcing(graph)
    source = captured.snapshot
    assert all(0 < value < 1 for value in source.epi)
    assert source.stored_pressure == tuple(value / 4 for value in P * 2)
    assert (
        captured.kernel_pressure_defect == captured.stored_pressure_residual == (0,) * 6
    )
    generator = _exact_generator(source)
    assert _apply(generator, source.stored_pressure) == tuple(
        -p for p in source.stored_pressure
    )
    assert (
        _apply(generator, _apply(generator, source.stored_pressure))
        == source.stored_pressure
    )
    reference = derive_phase_response(
        cosine_gram=((Q(1),) * 6,) * 6,
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    fresh = derive_joint_nodal_response(
        source,
        reference,
        epi_weight=1,
        phase_weight=0,
        capacity_weight=0,
        phase_rate_over_pi=(0,) * 6,
        capacity_rate=(0,) * 6,
    )
    # The phase channel is disabled, so the optional phase velocity cannot
    # change this fresh-pressure derivative, regardless of its value.
    assert fresh.pressure_rate == tuple(-p for p in source.stored_pressure)
    current = compute_phase_current(graph)
    flux = compute_dnfr_flux(graph)
    divergence = compute_flux_divergence_vectorized(graph, flux)
    multiplier = -compute_extended_nodal_system(1, 0, 0, 0, 1).dnfr_derivative
    assert multiplier > 1
    slopes = []
    for i, node in enumerate(NODES):
        p = source.stored_pressure[i]
        assert current[node] == 0
        assert flux[node] == -p
        assert divergence[node] == pytest.approx(-math.sqrt(3) * float(p))
        result = compute_extended_nodal_system(
            1,
            float(p),
            0,
            current[node],
            divergence[node],
            _estimate_local_coupling_strength(graph, node),
        )
        assert result.classical_derivative == p
        assert result.dnfr_derivative == pytest.approx(
            multiplier * math.sqrt(3) * float(p)
        )
        slopes.append(result.dnfr_derivative)
        if p:
            assert p * result.dnfr_derivative > 0
            assert p * fresh.pressure_rate[i] < 0
        else:
            assert result.dnfr_derivative == fresh.pressure_rate[i] == 0
    assert (
        sum(
            float(p) * slope
            for p, slope in zip(source.stored_pressure, slopes, strict=True)
        )
        > 0
    )
    # The exact model is p'=(1+lambda)*sqrt(3)*L^2*p on this graph.
    # The code uses represented sqrt(3) and arithmetic, not that exact real.


def test_optional_phase_formula_contains_pressure_feedback_and_alignment_damping():
    graph = _graph()
    angle = math.pi / 6
    for a, i in NODES:
        graph.nodes[a, i]["theta"] = (-angle, angle, 0)[i]
    current = compute_phase_current(graph)
    j = (math.sin(2 * angle) + math.sin(angle)) / 3
    coupling = _estimate_local_coupling_strength(graph, NODES[0])
    assert coupling > 0
    assert all(
        _estimate_local_coupling_strength(graph, node) == coupling for node in NODES
    )
    for eta in (0.0, 0.125):
        response = 0.5 * math.sin(math.pi * eta) + 0.15 * eta
        expected_a_rate = response - 0.135 * coupling * j
        for node, sign in zip(NODES, P * 2, strict=True):
            assert current[node] == pytest.approx(float(sign) * j, abs=1e-15)
            result = compute_extended_nodal_system(
                1,
                -eta * float(sign),
                graph.nodes[node]["theta"],
                current[node],
                0,
                coupling,
            )
            assert result.phase_derivative == pytest.approx(
                -expected_a_rate * float(sign), abs=1e-15
            )
        if eta == 0:
            assert expected_a_rate < 0  # Positive phase contrast relaxes at p=0.
        else:
            assert expected_a_rate > 0  # Pressure can reverse its instantaneous rate.
    # These are independently supplied scalar inputs. In the full optional
    # runtime pressure follows its own incompatible equation tested above.


def test_hypothetical_fresh_pressure_substitution_has_a_strict_local_lyapunov_balance():
    s = pytest.importorskip("sympy")
    eta, angle = s.symbols("eta angle", real=True)
    alpha, beta, e, c, k = s.symbols("alpha beta e c k", positive=True)
    response = alpha * s.sin(s.pi * eta) + beta * eta
    current = (s.sin(2 * angle) + s.sin(angle)) / 3
    angle_rate = response - k * current
    eta_rate = -e * eta - c * response + c * k * current
    potential = (
        alpha * (1 - s.cos(s.pi * eta)) / s.pi
        + beta * eta**2 / 2
        + c * k * ((1 - s.cos(2 * angle)) / 6 + (1 - s.cos(angle)) / 3)
    )
    assert s.simplify(s.diff(potential, eta) - response) == 0
    assert s.simplify(s.diff(potential, angle) - c * k * current) == 0
    rate = s.diff(potential, eta) * eta_rate + s.diff(potential, angle) * angle_rate
    expected = -e * eta * response - c * (response - k * current) ** 2
    assert s.simplify(rate - expected) == 0
    # Exact interval facts supply the signs, rather than a phase/pressure scan.
    assert s.calculus.util.function_range(
        s.sin(s.pi * eta), eta, s.Interval.open(0, 1)
    ) == s.Interval.Lopen(0, 1)
    assert s.simplify(response.subs(eta, -eta) + response) == 0
    assert s.simplify(current.subs(angle, -angle) + current) == 0
    assert s.solveset(
        current, angle, domain=s.Interval.open(-s.pi / 4, s.pi / 4)
    ) == s.FiniteSet(0)
    assert s.simplify(expected.subs(eta, 0) + c * k**2 * current**2) == 0
    # For 0<|eta|<1, eta*f(eta)>0; at eta=0 the remaining term vanishes
    # only at angle=0 inside strict U3. Hence V'<0 off the origin there.
    # This pairs the existing phase formula with refreshed pressure instead
    # of the implemented independent pressure update. No such substitution
    # is installed, and invariance/attraction of this domain is not asserted.
