"""Inherited C4 joint law on an imposed synchronized block manifold.

The counted pressure observer is production code. The cotangent completion is
the conditional research model, not the default engine or an off-manifold
closure claim. All controls are exact state/derivative calculations, not flows.
"""

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import _metric_differential
from tests.physics.test_cotangent_phase_exchange_scope import _connection
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.joint_quotient import observe_joint_nodal_quotient


def _prepared_graph(graph, phases, form):
    graph.graph["DNFR_WEIGHTS"] = {"epi": 0.5, "phase": 0.5, "vf": 0, "topo": 0}
    for node, phase, value in zip(graph, phases, form, strict=True):
        graph.nodes[node].update(
            EPI=float(value), theta=float(phase), nu_f=1.0, delta_nfr=0.0
        )
    return graph


@pytest.fixture(scope="module")
def c4_quotient():
    s = pytest.importorskip("sympy")
    blocks = ((0,), (1, 3), (2,))
    phases = (0, s.pi / 6, s.pi / 2)
    supplied_form = (s.Rational(1, 4), -s.Rational(1, 8), s.Rational(1, 2))
    graph = _prepared_graph(
        nx.cycle_graph(4),
        (phases[0], phases[1], phases[2], phases[1]),
        (supplied_form[0], supplied_form[1], supplied_form[2], supplied_form[1]),
    )
    observed = observe_joint_nodal_quotient(graph, blocks)
    structure = observed.structure
    lift = s.Matrix(4, 3, lambda i, a: int(structure.node_blocks[i] == a))
    mass = lift.T * lift
    laplacian = s.eye(3) - s.Matrix(structure.multiplicity) / 2
    fine_rows = tuple(tuple(graph.neighbors(i)) for i in graph)
    fine_phases = tuple(phases[a] for a in structure.node_blocks)
    fine_metric, fine_differential, fine_source, _ = _metric_differential(
        s, fine_rows, fine_phases
    )

    # Independent macro construction differentiates the explicit two-neighbor
    # expressions BEFORE evaluating the prepared state. None of u, v, a is
    # zero here, so the removable sinc limits need no numerical tolerance.
    theta = s.symbols("Theta0:3", real=True)
    u, v = theta[1] - theta[0], theta[1] - theta[2]
    a, b = -(u + v) / 2, (u - v) / 2
    diagonal = s.Matrix(
        (
            2 * s.pi * s.sin(u) / u,
            2 * s.pi * s.cos(b) * s.sin(a) / a,
            2 * s.pi * s.sin(v) / v,
        )
    )
    substitution = dict(zip(theta, phases, strict=True))
    evaluated = diagonal.subs(substitution).applyfunc(s.simplify)
    metric = s.diag(*evaluated)
    differential = diagonal.jacobian(theta).subs(substitution).applyfunc(s.simplify)
    source = (s.Matrix((u, a, v)) / s.pi).subs(substitution)
    potential = 2 * (1 - s.cos(u)) + 2 * (1 - s.cos(v))
    gradient = s.Matrix([s.diff(potential, value) for value in theta])
    gradient = gradient.subs(substitution).applyfunc(s.simplify)
    return {
        "s": s,
        "graph": graph,
        "observed": observed,
        "lift": lift,
        "mass": mass,
        "laplacian": laplacian,
        "phases": phases,
        "supplied_form": supplied_form,
        "fine_rows": fine_rows,
        "fine_phases": fine_phases,
        "fine_metric": fine_metric,
        "fine_differential": fine_differential,
        "fine_source": fine_source,
        "metric": metric,
        "differential": differential,
        "source": source,
        "potential": potential.subs(substitution),
        "gradient": gradient,
    }


def test_unequal_c4_blocks_inherit_mass_metric_joint_rows_and_budgets(c4_quotient):
    data = c4_quotient
    s, observed = data["s"], data["observed"]
    structure = observed.structure
    lift, mass, laplacian = data["lift"], data["mass"], data["laplacian"]
    metric, differential, source = data["metric"], data["differential"], data["source"]
    fine_metric, fine_differential = data["fine_metric"], data["fine_differential"]
    assert structure.multiplicity == ((0, 2, 0), (1, 0, 1), (0, 2, 0))
    assert structure.block_degree == (2, 2, 2)
    assert structure.block_capacity == structure.effective_capacity == (1, 1, 1)
    assert structure.macro_metric_weights == (2, 4, 2)
    assert structure.macro_conductance == ((0, 2, 0), (2, 0, 2), (0, 2, 0))
    assert mass == s.diag(1, 2, 1)
    assert metric == s.diag(6, 6 * (s.sqrt(3) - 1), 3 * s.sqrt(3))
    assert (fine_metric * lift - lift * metric).applyfunc(s.simplify) == s.zeros(4, 3)
    assert (fine_differential * lift - lift * differential).applyfunc(
        s.simplify
    ) == s.zeros(4, 3)
    assert (data["fine_source"] - lift * source).applyfunc(s.simplify) == s.zeros(4, 1)
    assert (mass * metric * source + data["gradient"]).applyfunc(s.simplify) == s.zeros(
        3, 1
    )
    assert tuple(map(float, observed.counted_phase_gradient)) == pytest.approx(
        tuple(map(float, source)), abs=3e-15
    )

    form = s.Matrix(s.symbols("X0:3", real=True))
    fine_form = lift * form
    fine_laplacian = s.eye(4) - s.Matrix(
        4, 4, lambda i, j: s.Rational(int(j in data["fine_rows"][i]), 2)
    )
    assert fine_laplacian * lift == lift * laplacian
    fine_connection = _connection(s, fine_metric, fine_differential, fine_form)
    # Momentum is S*H*X. Its connection acts on grad_X E=S*X, not on X.
    inherited_connection = _connection(s, mass * metric, mass * differential, form)
    correction = inherited_connection * mass * form
    assert correction.applyfunc(s.simplify) != s.zeros(3, 1)
    assert (fine_connection * fine_form - lift * correction).applyfunc(
        s.simplify
    ) == s.zeros(4, 1)
    e, weight = s.symbols("e w", positive=True)
    form_rate = -e * laplacian * form + weight * source + correction
    phase_rate = metric.inv() * form
    fine_form_rate = -e * fine_laplacian * fine_form + weight * data["fine_source"]
    fine_form_rate += fine_connection * fine_form
    assert (fine_form_rate - lift * form_rate).applyfunc(s.simplify) == s.zeros(4, 1)
    assert (fine_metric.inv() * fine_form - lift * phase_rate).applyfunc(
        s.simplify
    ) == s.zeros(4, 1)
    assert (lift.T * fine_metric * fine_form - mass * metric * form).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)

    fine_potential = sum(
        1 - s.cos(data["fine_phases"][j] - data["fine_phases"][i])
        for i, j in data["graph"].edges
    )
    assert s.simplify(fine_potential - data["potential"]) == 0
    assert fine_form.dot(fine_form) == form.dot(mass * form)
    storage_rate = (mass * form).dot(form_rate) + weight * data["gradient"].dot(
        phase_rate
    )
    dissipation = e * ((form[0] - form[1]) ** 2 + (form[2] - form[1]) ** 2)
    assert s.simplify(storage_rate + dissipation) == 0
    assert s.simplify((mass * form).dot(correction)) == 0
    # Common phase rotation supplies total inherited canonical momentum.
    momentum_rate = sum(mass * metric * form_rate)
    momentum_rate += (mass * form).dot(differential * phase_rate)
    assert s.simplify(momentum_rate + e * sum(mass * metric * laplacian * form)) == 0


def test_bare_p3_recomputation_keeps_pressure_but_loses_inherited_joint_rate(
    c4_quotient,
):
    data = c4_quotient
    s = data["s"]
    rows = ((1,), (0, 2), (1,))
    bare_metric, bare_differential, bare_source, _ = _metric_differential(
        s, rows, data["phases"]
    )
    assert (
        bare_metric - s.diag(s.Rational(1, 2), 1, s.Rational(1, 2)) * data["metric"]
    ).applyfunc(s.simplify) == s.zeros(3)
    assert (bare_source - data["source"]).applyfunc(s.simplify) == s.zeros(3, 1)
    bare_laplacian = s.eye(3) - s.Matrix(
        3, 3, lambda i, j: s.Rational(int(j in rows[i]), len(rows[i]))
    )
    assert bare_laplacian == data["laplacian"]
    bare_graph = _prepared_graph(
        nx.path_graph(3), data["phases"], data["supplied_form"]
    )
    bare_capture = capture_non_epi_forcing(bare_graph)
    old_pressure = -data["laplacian"] * s.Matrix(data["supplied_form"]) / 2
    old_pressure += data["source"] / 2
    assert tuple(map(float, bare_capture.full_kernel_pressure)) == pytest.approx(
        tuple(map(float, old_pressure)), abs=3e-15
    )
    assert tuple(map(float, data["observed"].effective_pressure)) == pytest.approx(
        tuple(map(float, old_pressure)), abs=3e-15
    )

    form = s.Matrix(data["supplied_form"])
    inherited_velocity = data["metric"].inv() * form
    bare_velocity = bare_metric.inv() * form
    assert (bare_velocity - s.diag(2, 1, 2) * inherited_velocity).applyfunc(
        s.simplify
    ) == s.zeros(3, 1)
    assert bare_velocity != inherited_velocity
    required_scales = {
        s.simplify(bare_velocity[i] / inherited_velocity[i]) for i in range(3)
    }
    assert len(required_scales) > 1  # No common exchange scale repairs all rows.
    inherited_connection = _connection(
        s, data["mass"] * data["metric"], data["mass"] * data["differential"], form
    )
    bare_connection = _connection(s, bare_metric, bare_differential, form)
    defect = (
        bare_connection * form - inherited_connection * data["mass"] * form
    ).applyfunc(s.simplify)
    assert defect != s.zeros(3, 1)
    # This comparison exposes discarded multiplicity. It does not extend the
    # regular-support unweighted-storage theorem to the irregular bare P3.
