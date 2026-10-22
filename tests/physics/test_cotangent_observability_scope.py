"""State loss and chart covariance under the conditional cotangent law.

The snapshot witness fixes the graph, clock, capacities and constitutive law;
the positive chart control transforms the full law rather than resetting it.
Pressure is never reconstructed from observed motion. Neither control excludes
temporal identification or establishes a privileged physical form origin.
"""

import networkx as nx
import pytest

from tests.physics.test_cotangent_phase_exchange_scope import _p2_field
from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.canonical import compute_canonical_nodal_derivative
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.sdk.simple import Network
from tnfr.sdk.study import diagnose_network


def test_identical_tetrad_and_pressure_hide_different_relative_form_acceleration():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, _, _, field = _p2_field(s)
    x0, x1, delta = coordinates
    mean, contrast = s.symbols("m a", real=True)
    # The regular consensus jet is taken from the FULL field before imposing
    # either preparation; its first derivatives retain the geometric K*x term.
    jet = field.applyfunc(lambda value: s.series(value, delta, 0, 2).removeO())
    consensus_field = jet.subs(delta, 0)
    acceleration = jet.jacobian(coordinates).subs(delta, 0) * consensus_field
    contrast_acceleration = s.expand((acceleration[0] - acceleration[1]) / 2)
    shifted = {x0: mean + contrast, x1: mean - contrast}
    centered = {x0: contrast, x1: -contrast}
    assert s.simplify(
        consensus_field.subs(shifted) - consensus_field.subs(centered)
    ) == s.zeros(3, 1)
    acceleration_gap = s.simplify(
        contrast_acceleration.subs(shifted) - contrast_acceleration.subs(centered)
    )
    assert acceleration_gap == -4 * contrast * mean**2 / (3 * s.pi**2)
    # On unit-length P2, Phi_s=(p1,p0). At unit capacity its first time
    # derivative is therefore the reversed form acceleration under this law.
    potential_rate = s.Matrix((acceleration[1], acceleration[0]))
    potential_rate_gap = (
        potential_rate.subs(shifted) - potential_rate.subs(centered)
    ).applyfunc(s.simplify)
    assert (
        potential_rate_gap.subs({contrast: s.Rational(1, 8), mean: s.Rational(1, 4)})
        == s.Matrix((s.Rational(1, 64), -s.Rational(1, 192))) / s.pi**2
    )

    reports = []
    for prepared_mean in (s.Integer(0), s.Rational(1, 4)):
        preparation = {
            x0: prepared_mean + s.Rational(1, 8),
            x1: prepared_mean - s.Rational(1, 8),
            e: s.Rational(1, 2),
            weight: s.Rational(1, 2),
        }
        candidate_pressure = tuple(
            float(value.subs(preparation)) for value in consensus_field[:2]
        )
        graph = nx.path_graph(2)
        graph.edges[0, 1].update(weight=1.0, length=1.0)
        graph.graph["DNFR_WEIGHTS"] = {
            "epi": 0.5,
            "phase": 0.5,
            "vf": 0.0,
            "topo": 0.0,
        }
        for node, symbol in enumerate((x0, x1)):
            set_attr(graph.nodes[node], ALIAS_EPI, float(preparation[symbol]))
            set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
            set_attr(graph.nodes[node], ALIAS_VF, 1.0)
        default_compute_delta_nfr(graph)
        for node, pressure in enumerate(candidate_pressure):
            # K*x vanishes at consensus, so the current pressure owner agrees
            # here. This does not install the candidate's subsequent evolution.
            assert get_attr(graph.nodes[node], ALIAS_DNFR) == pressure
            set_attr(graph.nodes[node], ALIAS_DNFR, pressure)
            product = compute_canonical_nodal_derivative(1.0, pressure)
            set_attr(graph.nodes[node], ALIAS_DEPI, product.derivative)
        reports.append(diagnose_network(Network(graph)))

    baseline, shifted_report = reports
    assert baseline["state"] != shifted_report["state"]
    assert baseline["tetrad"] == shifted_report["tetrad"]
    assert all(field["available"] for field in baseline["tetrad"].values())
    assert baseline["tetrad"]["phi_s"]["value"] == [0.125, -0.125]
    assert baseline["tetrad"]["grad_phi"]["value"] == [0.0, 0.0]
    assert baseline["tetrad"]["k_phi"]["value"] == [0.0, 0.0]
    xi = baseline["tetrad"]["xi_c"]
    assert xi["value"] == pytest.approx(2.0**-0.5)
    assert xi["provenance"]["method"] == "spectral_gap"
    assert not xi["provenance"]["fit_available"]
    assert baseline["metrics"]["coherence"] == shifted_report["metrics"]["coherence"]
    for report in reports:
        nodes = [row["value"] for row in report["nodal"]]
        assert [node["expected_depi_dt"] for node in nodes] == [-0.125, 0.125]
        assert all(not node["acceleration_available"] for node in nodes)
        assert all(node["observed_d2epi_dt2"] is None for node in nodes)
    # The difference is in contrast acceleration, hence cannot be removed by
    # a common phase rotation. No trajectory or empirical derivative is used.
    assert acceleration_gap.subs(
        {contrast: s.Rational(1, 8), mean: s.Rational(1, 4)}
    ) == -1 / (96 * s.pi**2)


def test_passive_form_translation_inherits_joint_law_momentum_and_storage():
    s = pytest.importorskip("sympy")
    coordinates, e, weight, h, old_phase_rate, old_field = _p2_field(s)
    x0, x1, delta = coordinates
    theta = s.symbols("theta0:2", real=True)
    momentum = s.Matrix(s.symbols("p0:2", real=True))
    form = s.Matrix(s.symbols("y0:2", real=True))
    offset = s.Symbol("beta", real=True)
    lifted_h = h.subs(delta, theta[1] - theta[0])
    # Independently push the canonical bracket through y=p/h+beta. The
    # geometry is differentiated before replacing momentum by its new chart.
    observation = s.Matrix(
        [momentum[i] / lifted_h + offset for i in range(2)] + list(theta)
    )
    tangent = observation.jacobian((*theta, *momentum))
    canonical = s.zeros(2).row_join(s.eye(2)).col_join((-s.eye(2)).row_join(s.zeros(2)))
    inverse = {momentum[i]: lifted_h * (form[i] - offset) for i in range(2)}
    phase_chart = {theta[0]: 0, theta[1]: delta}
    bracket = (tangent * canonical * tangent.T).subs(inverse).subs(phase_chart)

    canonical_storage = sum((value / lifted_h) ** 2 for value in momentum) / 2
    canonical_storage += weight * (1 - s.cos(theta[1] - theta[0]))
    storage = s.simplify(canonical_storage.subs(inverse).subs(phase_chart))
    total_momentum = s.simplify(sum(momentum).subs(inverse).subs(phase_chart))
    old_coordinates = {x0: form[0] - offset, x1: form[1] - offset}
    old_storage = (x0**2 + x1**2) / 2 + weight * (1 - s.cos(delta))
    assert s.simplify(storage - old_storage.subs(old_coordinates)) == 0
    assert s.simplify(total_momentum - h * (x0 + x1).subs(old_coordinates)) == 0

    gradient = s.Matrix(
        (
            s.diff(storage, form[0]),
            s.diff(storage, form[1]),
            -s.diff(storage, delta),
            s.diff(storage, delta),
        )
    )
    laplacian = s.Matrix(((1, -1), (-1, 1)))
    translated_field = bracket * gradient
    translated_field += (-e * laplacian * form).col_join(s.zeros(2, 1))
    # Compare BOTH absolute phase rows as well as form. Equality is stronger
    # than coincident relative phase or a common time reparameterization.
    expected = s.Matrix((old_field[0], old_field[1], *old_phase_rate))
    assert (translated_field - expected.subs(old_coordinates)).applyfunc(
        s.simplify
    ) == s.zeros(4, 1)

    relative_field = s.Matrix(
        (
            translated_field[0],
            translated_field[1],
            translated_field[3] - translated_field[2],
        )
    )
    retained = (*form, delta)
    storage_rate = (s.Matrix([storage]).jacobian(retained) * relative_field)[0]
    momentum_rate = (s.Matrix([total_momentum]).jacobian(retained) * relative_field)[0]
    assert s.simplify(storage_rate + e * (form[0] - form[1]) ** 2) == 0
    assert s.simplify(momentum_rate) == 0
