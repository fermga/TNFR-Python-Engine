"""Routine contracts for the opt-in, conditional relational joint solver.

Analytic small graphs check the actual shared owner. No operator policy,
autonomous-law selection, trajectory persistence or physical bridge is inferred.
"""

import math
import pickle
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_theta_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import relational
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    step_relational_exchange,
)
from tnfr.errors.contextual import TNFRUserError
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics import BEPIElement
from tnfr.metrics.trig_cache import get_trig_cache
from tnfr.sdk import Network
from tnfr.types import (
    ensure_bepi,
    require_finite_real_scalar_epi,
    serialize_bepi,
    serialize_bepi_json,
)

ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


@pytest.mark.parametrize(
    "field,value",
    (
        ("storage_scale", True),
        ("storage_scale", 0),
        ("storage_scale", float("nan")),
        ("epi_weight", True),
        ("epi_weight", -0.5),
        ("phase_weight", True),
        ("phase_weight", 0),
        ("phase_domain", "unrecognized"),
    ),
)
def test_stored_model_admission_precedes_field_or_step_work(monkeypatch, field, value):
    graph = _graph()
    before = pickle.dumps(graph)
    model = copy(RelationalExchangeModel(1))
    object.__setattr__(model, field, value)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid stored model must reject before field arithmetic")

    monkeypatch.setattr(relational, "_field", forbidden)
    with pytest.raises((TypeError, ValueError)):
        evaluate_relational_exchange(graph, model=model)
    with pytest.raises((TypeError, ValueError)):
        step_relational_exchange(graph, model=model, dt=Q(1, 128))
    assert pickle.dumps(graph) == before


def _graph():
    graph = nx.path_graph(3)
    graph.graph.update(
        GAMMA={"type": "none"},
        DNFR_WEIGHTS={"epi": 0.1, "phase": 0.2, "vf": 0.3, "topo": 0.4},
        _t=2.0,
        untouched={"evidence": [1, 2]},
    )
    for node, form, phase, capacity in zip(
        graph,
        (0.75, 0.25, 0.5),
        (-math.pi / 6, 0.0, math.pi / 6),
        (1.0, 0.0, 2.0),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity, delta_nfr=999.0)
    return graph


def _pair(*, form=(1.0, -1.0), phase=(0.0, 0.0)):
    graph = nx.path_graph(2)
    graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=phase[node], nu_f=1.0, delta_nfr=999.0
        )
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_analytic_heterogeneous_path_field_is_fresh_detached_and_read_only():
    graph = _graph()
    before = _snapshot(graph)
    report = evaluate_relational_exchange(graph, model=RelationalExchangeModel(2.0))
    assert _snapshot(graph) == before
    assert report.nodes == (0, 1, 2)
    assert report.form_gradient == pytest.approx((0.5, -0.75, 0.25))
    assert report.phase_source == pytest.approx((1 / 6, 0, -1 / 6))
    assert report.phase_metric == pytest.approx((3, math.pi * math.sqrt(3), 3))
    assert report.pressure == pytest.approx((-1 / 6, 3 / 16, -5 / 24))
    assert report.form_rate == pytest.approx((-1 / 6, 0, -5 / 12))
    assert report.phase_rate == pytest.approx((1 / 24, 0, 1 / 24))
    assert float(report.continuous_loss) == pytest.approx(3 / 16)
    assert float(report.storage_rate) == pytest.approx(-3 / 16)
    assert abs(float(report.balance_residual)) < 1e-14
    with pytest.raises(FrozenInstanceError):
        report.epi = (0.0, 0.0, 0.0)
    graph.nodes[0]["EPI"] = 10.0
    assert report.epi == (0.75, 0.25, 0.5)
    assert isinstance(report.pressure, tuple)
    assert isinstance(report.edges, tuple)


def test_consensus_metric_limit_and_receiver_capacity_separation():
    graph = _graph()
    for node in graph:
        graph.nodes[node]["theta"] = 0.0
    model = RelationalExchangeModel(2.0)
    original = evaluate_relational_exchange(graph, model=model)
    assert original.phase_source == (0.0, 0.0, 0.0)
    assert original.phase_metric == pytest.approx((math.pi, 2 * math.pi, math.pi))
    assert original.phase_rate == pytest.approx(
        (1 / (8 * math.pi), 0, 1 / (8 * math.pi))
    )
    graph.nodes[1]["nu_f"] = 0.5
    changed = evaluate_relational_exchange(graph, model=model)
    assert changed.phase_rate[0] == original.phase_rate[0]
    assert changed.phase_rate[2] == original.phase_rate[2]
    assert changed.phase_rate[1] != original.phase_rate[1]


@pytest.mark.parametrize("phase_domain", ("acute", "positive_resultant"))
def test_primitive_star_row_survives_remote_changes_and_neighbor_degree_changes(
    phase_domain,
):
    graph = nx.Graph(((0, 1), (0, 2), (0, 3), (1, 4)))
    for node, form, phase, capacity in zip(
        graph,
        (0.75, 0.25, -0.25, 0.5, -0.125),
        (0.0625, 0.125, -0.25, 0.375, 0.25),
        (1.5, 0.5, 2.0, 0.75, 1.0),
        strict=True,
    ):
        graph.nodes[node].update(EPI=form, theta=phase, nu_f=capacity)
    extended = deepcopy(graph)
    extended.nodes[4].update(EPI=2.0, theta=-0.375, nu_f=4.0)
    extended.add_node(5, EPI=-1.0, theta=0.125, nu_f=0.0)
    extended.add_node(6, EPI=0.0, theta=0.25, nu_f=3.0)
    extended.add_edges_from(((1, 2), (2, 5), (5, 6)))
    assert tuple(graph[0]) == tuple(extended[0]) == (1, 2, 3)
    assert all(graph.nodes[node] == extended.nodes[node] for node in (0, 1, 2, 3))
    assert graph.degree[1] != extended.degree[1]
    assert graph.degree[2] != extended.degree[2]
    snapshots = _snapshot(graph), _snapshot(extended)
    model = RelationalExchangeModel(2.0, phase_domain=phase_domain)
    before = evaluate_relational_exchange(graph, model=model)
    after = evaluate_relational_exchange(extended, model=model)

    # Neither neighboring degrees nor edges among neighbors are inputs to the
    # primitive row. Keep the same neighbor order and small native backend;
    # locality of the ideal law is not a cross-backend bitwise guarantee.
    assert before.pressure_path == after.pressure_path
    assert before.form_gradient[0] == after.form_gradient[0] == 1.75
    for name in (
        "phase_source",
        "phase_metric",
        "phase_gradient",
        "phase_rate",
        "phase_mobility",
        "relative_resultant",
        "phase_rate_rounding_defect",
    ):
        assert getattr(before, name)[0] == getattr(after, name)[0]
    for name in ("pressure", "form_rate"):
        assert getattr(before, name)[0] == pytest.approx(
            getattr(after, name)[0], rel=2e-15, abs=2e-15
        )
    for field in (before, after):
        split = -Q(model.epi_weight) * Q(7, 4) / 3 + Q(model.phase_weight) * Q(
            field.phase_source[0]
        )
        assert field.pressure_split_residual[0] == Q(field.pressure[0]) - split
        assert abs(field.pressure_split_residual[0]) < Q(1, 10**14)
    assert before.storage != after.storage
    assert (_snapshot(graph), _snapshot(extended)) == snapshots


@pytest.mark.parametrize("phase_domain", ("acute", "positive_resultant"))
def test_primitive_row_locality_does_not_bypass_remote_domain_admission(phase_domain):
    graph = nx.path_graph(3)
    for node in graph:
        graph.nodes[node].update(EPI=float(node), theta=0.0, nu_f=1.0)
    model = RelationalExchangeModel(2.0, phase_domain=phase_domain)
    admitted = evaluate_relational_exchange(graph, model=model)
    assert admitted.phase_rate[0] != 0.0
    local_state = tuple(dict(graph.nodes[node]) for node in (0, 1))
    # Node 2 is outside node 0's primitive star. The complete API still admits
    # the whole graph, so its remote invalid phase chart rejects evaluation.
    graph.nodes[2]["theta"] = math.pi
    assert tuple(dict(graph.nodes[node]) for node in (0, 1)) == local_state
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="acute|positive real part"):
        evaluate_relational_exchange(graph, model=model)
    assert _snapshot(graph) == before


def test_pressure_uses_explicit_model_and_matches_the_native_owner():
    graph = _graph()
    before = _snapshot(graph)
    model = RelationalExchangeModel(2.0, epi_weight=3.0, phase_weight=1.0)
    report = evaluate_relational_exchange(graph, model=model)
    native = deepcopy(graph)
    native.graph["DNFR_WEIGHTS"] = {"epi": 0.75, "phase": 0.25, "vf": 0.0, "topo": 0.0}
    default_compute_delta_nfr(native)
    expected = tuple(get_attr(native.nodes[node], ALIAS_DNFR) for node in native)
    assert report.pressure == pytest.approx(expected, abs=1e-15)
    assert _snapshot(graph) == before


def test_acute_phase_admission_does_not_use_binary64_tau_as_a_trigonometric_period():
    raw = math.tau * 2**55
    # A floating-tau modulo falsely reports consensus. The phasor is instead
    # obtuse, so it cannot be a source for the acute-only execution contract.
    assert math.remainder(raw, math.tau) == 0
    assert math.cos(raw) < 0
    graph = _pair(form=(0.0, 1.0), phase=(0.0, raw))
    before = _snapshot(graph)
    for operation in (
        lambda: evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0)),
        lambda: step_relational_exchange(
            graph, model=RelationalExchangeModel(1.0), dt=1 / 32
        ),
    ):
        with pytest.raises(ValueError, match="strictly acute"):
            operation()
        assert _snapshot(graph) == before


@pytest.mark.parametrize("domain", ("acute", "positive_resultant", "regular"))
def test_large_raw_lift_uses_one_phase_source_for_pressure_storage_and_rates(domain):
    raw = math.tau * 2**56
    sine, cosine = math.sin(raw), math.cos(raw)
    gap = math.atan2(sine, cosine)
    assert math.remainder(raw, math.tau) == 0
    assert 0 < gap < math.pi / 2 and cosine > 0
    graph = _pair(form=(0.0, 1.0), phase=(0.0, raw))
    before = _snapshot(graph)
    report = evaluate_relational_exchange(
        graph, model=RelationalExchangeModel(1.0, phase_domain=domain)
    )
    assert _snapshot(graph) == before
    assert report.phase == (0.0, raw)
    assert report.phase_source == pytest.approx(
        (gap / math.pi, -gap / math.pi), abs=1e-15
    )
    assert report.phase_metric == pytest.approx((math.pi * sine / gap,) * 2, rel=2e-15)
    expected_pressure = 0.5 + gap / (2 * math.pi)
    assert report.pressure == pytest.approx(
        (expected_pressure, -expected_pressure), abs=1e-15
    )
    assert float(report.storage) == pytest.approx(0.5 + 1 - cosine, abs=1e-15)
    assert abs(float(report.balance_residual)) < 1e-14
    assert all(abs(float(value)) < 1e-15 for value in report.pressure_split_residual)
    assert report.pressure_path == "relative_resultant_canonical"


@pytest.mark.parametrize("domain", ("acute", "positive_resultant", "regular"))
def test_every_relational_mode_reuses_its_captured_pressure_source_and_provenance(
    monkeypatch, domain
):
    from tnfr.dynamics import dnfr

    def forbidden(*args, **kwargs):
        pytest.fail("a relational field must not reconstruct a second phase source")

    monkeypatch.setattr(dnfr, "default_compute_delta_nfr", forbidden)
    graph = _graph()
    model = RelationalExchangeModel(2.0, phase_domain=domain)
    report = step_relational_exchange(graph, model=model, dt=1 / 64)
    assert (
        report.before.pressure_path
        == report.after.pressure_path
        == "relative_resultant_canonical"
    )
    assert graph.graph["_DNFR_META"]["hook"] == "relative_resultant_canonical"
    assert graph.graph["_dnfr_hook_name"] == "relative_resultant_canonical"


def test_large_acute_lift_keeps_the_real_initial_gap_in_whole_segment_admission():
    graph = _pair(form=(0.0, 1.0), phase=(0.0, math.tau * 2**56))
    model = RelationalExchangeModel(1.0)
    initial = evaluate_relational_exchange(graph, model=model)
    assert 0 < initial.phase_source[0] < 0.5
    # The right raw lift cannot materialize this small increment, but the
    # left node moves. Its proposal crosses the actual initial acute face;
    # treating the initial gap as a binary64-tau remainder would miss it.
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="initial acute lift"):
        step_relational_exchange(graph, model=model, dt=2.0)
    assert _snapshot(graph) == before


def test_single_step_is_simultaneous_unclipped_and_refreshes_endpoint_pressure():
    graph = _graph()
    graph.graph.update(CLIP_MODE="hard", EPI_MIN=-0.1, EPI_MAX=0.1)
    ambient_weights = deepcopy(graph.graph["DNFR_WEIGHTS"])
    model = RelationalExchangeModel(2.0)
    dt = 1 / 16
    report = step_relational_exchange(graph, model=model, dt=dt)
    expected_form = (0.75 - dt / 6, 0.25, 0.5 - 5 * dt / 12)
    expected_phase = (-math.pi / 6 + dt / 24, 0, math.pi / 6 + dt / 24)
    assert tuple(
        get_attr(graph.nodes[node], ALIAS_EPI) for node in graph
    ) == pytest.approx(expected_form)
    assert tuple(get_theta_attr(graph.nodes[node]) for node in graph) == pytest.approx(
        expected_phase
    )
    assert graph.graph["_t"] == 2.0 + dt
    assert report.t_before == 2.0
    assert report.t_after == 2.0 + dt
    assert tuple(get_attr(graph.nodes[node], ALIAS_VF) for node in graph) == (
        1.0,
        0.0,
        2.0,
    )
    assert graph.graph["DNFR_WEIGHTS"] == ambient_weights
    assert graph.graph["CLIP_MODE"] == "hard"
    assert graph.graph["untouched"] == {"evidence": [1, 2]}
    endpoint = evaluate_relational_exchange(graph, model=model)
    assert (
        tuple(get_attr(graph.nodes[node], ALIAS_DNFR) for node in graph)
        == endpoint.pressure
    )
    assert report.after.pressure == endpoint.pressure
    assert report.after.pressure != report.before.pressure


@pytest.mark.parametrize(
    "arguments",
    (
        {"storage_scale": 0.0},
        {"storage_scale": 1.0, "phase_weight": 0.0},
        {"storage_scale": True},
    ),
)
def test_model_rejects_unsupported_scales_without_boolean_coercion(arguments):
    with pytest.raises(ADMISSION_ERRORS):
        RelationalExchangeModel(**arguments)


@pytest.mark.parametrize(
    "attribute,value",
    (
        ("theta", True),
        ("nu_f", Q(1, 2**2000)),
        ("EPI", "0.5"),
        ("EPI", True),
        ("EPI", Q(1, 2**2000)),
    ),
)
def test_raw_last_node_rejection_is_atomic(attribute, value):
    graph = _graph()
    graph.nodes[2][attribute] = value
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_rich_bepi_is_not_replaced_by_a_magnitude_even_at_zero_capacity():
    graph = _graph()
    graph.nodes[2]["EPI"] = BEPIElement((0.5, -0.5), (0.5, -0.5), (0.0, 1.0))
    graph.nodes[2]["nu_f"] = 0.0
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_uniform_negative_bepi_and_legacy_aliases_keep_the_signed_chart():
    graph = _pair(form=(-0.5, -0.25))
    for node in graph:
        attributes = graph.nodes[node]
        original = attributes.pop("EPI")
        attributes[ALIAS_EPI[-1]] = ensure_bepi(original)
        attributes["phase"] = attributes.pop("theta")
        attributes["EPI_kind"] = "wave"
    get_trig_cache(graph)
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(1.0), dt=0.125
    )
    observed = tuple(
        require_finite_real_scalar_epi(
            get_attr(graph.nodes[node], ALIAS_EPI, conv=lambda raw: raw)
        )
        for node in graph
    )
    assert observed == pytest.approx((-0.484375, -0.265625))
    assert all(value < 0 for value in observed)
    assert all(ALIAS_EPI[-1] in graph.nodes[node] for node in graph)
    assert all(graph.nodes[node]["EPI_kind"] == "wave" for node in graph)
    cached_phase = get_trig_cache(graph).theta
    assert tuple(cached_phase[node] for node in graph) == pytest.approx(
        report.after.phase
    )


@pytest.mark.parametrize("serialize", (serialize_bepi, serialize_bepi_json))
def test_serialized_uniform_real_epi_uses_the_same_field_and_atomic_step(serialize):
    from tnfr.physics.relational_observations import observe_relational_reset

    scalar = _pair(form=(-0.5, -0.25))
    serialized = _pair(form=(-0.5, -0.25))
    for node in serialized:
        serialized.nodes[node]["EPI"] = serialize(serialized.nodes[node]["EPI"])
    model = RelationalExchangeModel(1.0)
    before = _snapshot(serialized)
    field = evaluate_relational_exchange(serialized, model=model)
    assert field == evaluate_relational_exchange(scalar, model=model)
    assert field.epi == (-0.5, -0.25)
    # The reset/transport observer already admits this canonical representation.
    reset = observe_relational_reset(serialized, serialized, storage_scale=1.0)
    assert reset.before.epi == (Q(-0.5), Q(-0.25))
    assert reset.storage_before == field.storage
    assert reset.storage_change == reset.identity_residual == 0
    assert _snapshot(serialized) == before
    actual = step_relational_exchange(serialized, model=model, dt=1 / 8)
    expected = step_relational_exchange(scalar, model=model, dt=1 / 8)
    assert actual == expected
    assert actual.after.epi == (-0.484375, -0.265625)


@pytest.mark.parametrize(
    "invalid",
    (
        {"not_epi": 0.5},
        {"continuous": (0.5, 0.25), "discrete": (0.5, 0.5), "grid": (0, 1)},
        {"continuous": (0.5j, 0.5j), "discrete": (0.5j, 0.5j), "grid": (0, 1)},
        {"continuous": (True, True), "discrete": (1.0, 1.0), "grid": (0, 1)},
        {
            "continuous": (Q(1, 2**2000),) * 2,
            "discrete": (0.0, 0.0),
            "grid": (0, 1),
        },
        {
            "continuous": ({"real": 0.5, "imag": Q(1, 2**2000)},) * 2,
            "discrete": (0.5, 0.5),
            "grid": (0, 1),
        },
    ),
)
def test_serialized_epi_admission_rejects_rich_or_lost_state_atomically(invalid):
    graph = _pair()
    graph.nodes[1]["EPI"] = invalid
    graph.nodes[1]["nu_f"] = 0.0
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(1), dt=1 / 8)
    assert _snapshot(graph) == before


def test_nonzero_underflowed_nodal_rate_is_not_certified_as_equilibrium():
    graph = _pair(form=(0.25, 0.0))
    graph.nodes[0]["nu_f"] = math.ulp(0.0)
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    assert _snapshot(graph) == before


def test_exact_nonzero_storage_survives_below_binary64_energy_resolution():
    small = 1e-200
    graph = _pair(form=(small, 0.0))
    report = evaluate_relational_exchange(graph, model=RelationalExchangeModel(1.0))
    assert report.form_storage == Q(small) ** 2 / 2
    assert report.storage > 0
    assert report.continuous_loss > 0
    assert float(report.storage) == 0.0
    assert float(report.continuous_loss) == 0.0
    assert all(rate != 0.0 for rate in report.form_rate)


@pytest.mark.parametrize("kind", ("directed", "multigraph", "disconnected", "weighted"))
def test_unsupported_graph_is_rejected_before_any_owned_write(kind):
    graph = _graph()
    if kind == "directed":
        graph = nx.DiGraph(graph)
    elif kind == "multigraph":
        graph = nx.MultiGraph(graph)
    elif kind == "disconnected":
        graph.remove_edge(1, 2)
    else:
        graph.edges[0, 1]["weight"] = 2.0
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


@pytest.mark.parametrize(
    "clock_arguments",
    ({"dt": True}, {"dt": Q(1, 2**2000)}, {"dt": 0.1, "t": True}),
)
def test_solver_clock_rejects_raw_invalid_values_atomically(clock_arguments):
    graph = _graph()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(
            graph, model=RelationalExchangeModel(2.0), **clock_arguments
        )
    assert _snapshot(graph) == before


def test_overridden_none_gamma_cannot_execute_or_hide_declared_forcing(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an unforced exchange must never invoke an external Gamma callback")

    graph = _graph()
    monkeypatch.setitem(GAMMA_REGISTRY, "none", GammaEntry(forbidden, False))
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def test_active_gamma_declaration_is_rejected_even_if_a_parameter_is_zero():
    graph = _graph()
    graph.graph["GAMMA"] = {"type": "harmonic", "beta": 0.0}
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        evaluate_relational_exchange(graph, model=RelationalExchangeModel(2.0))
    assert _snapshot(graph) == before


def test_nonacute_candidate_endpoint_rolls_back_form_phase_clock_and_metadata():
    graph = _pair()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(
            graph,
            model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1),
            dt=2.0,
        )
    assert _snapshot(graph) == before


def test_failed_cache_hook_cannot_commit_a_valid_candidate_partially():
    graph = _graph()
    graph.graph["_trig_version"] = "malformed existing cache metadata"
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        step_relational_exchange(graph, model=RelationalExchangeModel(2.0), dt=0.1)
    assert _snapshot(graph) == before


def _p2_coupled_jacobian(model, *, nu=1.0):
    """Axis probes of the ideal consensus derivative, with represented rates."""

    def antisym(xi, phi):
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=1.0)
        graph.graph.update(GAMMA={"type": "none"}, _t=0.0)
        for node, (x, th) in zip((0, 1), ((xi / 2, phi / 2), (-xi / 2, -phi / 2))):
            graph.nodes[node].update(EPI=x, theta=th, nu_f=nu, delta_nfr=0.0)
        field = evaluate_relational_exchange(graph, model=model)
        return (
            field.form_rate[0] - field.form_rate[1],
            field.phase_rate[0] - field.phase_rate[1],
        )

    fx, px = antisym(0.25, 0.0)
    fp, pp = antisym(0.0, 0.25)
    return ((fx / 0.25, fp / 0.25), (px / 0.25, pp / 0.25))


@pytest.mark.parametrize(
    "epi_w,beta,nu",
    ((0.25, 1.0, 1.0), (0.5, 1.0, 2.0), (0.25, 4.0, 1.0)),
)
def test_coupled_form_phase_relaxation_jacobian_and_underdamping_threshold(
    epi_w, beta, nu
):
    model = RelationalExchangeModel(beta, epi_weight=epi_w, phase_weight=1.0 - epi_w)
    e, w = model.epi_weight, model.phase_weight
    (a, b), (c, d) = _p2_coupled_jacobian(model, nu=nu)
    # P2 axis probes recover the consensus derivative, not a full nonlinear flow.
    assert a == pytest.approx(-2 * nu * e, rel=0, abs=1e-15)
    assert b == pytest.approx(-2 * nu * w / math.pi, rel=0, abs=1e-15)
    assert c == pytest.approx(2 * nu * w / (beta * math.pi), rel=0, abs=1e-15)
    assert d == pytest.approx(0.0, rel=0, abs=1e-15)
    # The discriminant classifies poles, not monotonicity of an observation.
    discriminant = e**2 - 4 * w**2 / (beta * math.pi**2)
    threshold = 2 / (math.pi * math.sqrt(beta))
    assert (discriminant < 0) == (e / w < threshold)
    trace, determinant = a + d, a * d - b * c
    assert trace == pytest.approx(-2 * nu * e, rel=0, abs=1e-15)
    assert determinant == pytest.approx(
        4 * nu**2 * w**2 / (beta * math.pi**2), rel=0, abs=1e-15
    )
    assert trace**2 - 4 * determinant == pytest.approx(
        4 * nu**2 * discriminant, rel=0, abs=1e-14
    )


@pytest.mark.parametrize("beta", (1.0, 4.0, 0.25))
def test_coupled_relaxation_critical_damping_at_pi_sqrt_beta(beta):
    # e/w = 2/(pi sqrt(beta)) with e + w = 1 gives a vanishing discriminant.
    ratio = 2 / (math.pi * math.sqrt(beta))
    epi_w = ratio / (1 + ratio)
    model = RelationalExchangeModel(beta, epi_weight=epi_w, phase_weight=1 - epi_w)
    e, w = model.epi_weight, model.phase_weight
    assert e / w == pytest.approx(ratio, rel=0, abs=1e-15)
    assert e**2 - 4 * w**2 / (beta * math.pi**2) == pytest.approx(0.0, abs=1e-15)
    (a, b), (c, d) = _p2_coupled_jacobian(model)
    assert (a + d) ** 2 - 4 * (a * d - b * c) == pytest.approx(0.0, abs=2e-15)


@pytest.mark.parametrize(
    "epi_w,phase_w,beta,oscillatory",
    (
        (1, 3, 1.0, True),  # (e/w)^2 beta = 1/9 < 4/pi^2
        (1, 1, 1.0, False),  # 1 > 4/pi^2
        (2, 3, 0.5, True),  # (2/3)^2 * 1/2 = 2/9 < 4/pi^2
        (1, 2, 2.0, False),  # 1/2 > 4/pi^2
    ),
)
def test_coupled_relaxation_character_is_the_dimensionless_invariant(
    epi_w, phase_w, beta, oscillatory
):
    model = RelationalExchangeModel(beta, epi_weight=epi_w, phase_weight=phase_w)
    e, w = model.epi_weight, model.phase_weight
    # This boundary belongs to the declared consensus law and phase normalization.
    invariant = (e / w) ** 2 * beta
    assert (invariant < 4 / math.pi**2) == oscillatory
    (a, b), (c, d) = _p2_coupled_jacobian(model)
    assert ((a + d) ** 2 - 4 * (a * d - b * c) < 0) == oscillatory


def test_coupled_underdamping_threshold_is_mode_independent_on_a_regular_ring():
    np = pytest.importorskip("numpy")
    n, beta, nu = 5, 1.0, 1.0
    ring = nx.cycle_graph(n)
    for node in ring:
        ring.add_edge(node, (node + 1) % n, weight=1.0)
    nodes = tuple(ring)
    threshold = 2 / (math.pi * math.sqrt(beta))

    def jacobian(model):
        def rates(vector):
            for i, node in enumerate(nodes):
                ring.nodes[node].update(
                    EPI=vector[i], theta=vector[n + i], nu_f=nu, delta_nfr=0.0
                )
            field = evaluate_relational_exchange(ring, model=model)
            assert field.nodes == nodes
            return [*field.form_rate, *field.phase_rate]

        h, columns = 1e-6, []
        for k in range(2 * n):
            plus, minus = [0.0] * (2 * n), [0.0] * (2 * n)
            plus[k], minus[k] = h, -h
            rp, rm = rates(plus), rates(minus)
            columns.append([(rp[i] - rm[i]) / (2 * h) for i in range(2 * n)])
        return np.array(columns).T

    for epi_w, oscillatory in ((0.25, True), (0.5, False)):
        model = RelationalExchangeModel(beta, epi_weight=epi_w, phase_weight=1 - epi_w)
        e, w = model.epi_weight, model.phase_weight
        assert (e / w < threshold) == oscillatory
        eigenvalues = np.linalg.eigvals(jacobian(model))
        complex_modes = int(np.sum(np.abs(eigenvalues.imag) > 1e-6))
        # At consensus the spatial modes share a pole class, not monotone readings.
        assert complex_modes == (2 * (n - 1) if oscillatory else 0)


def test_wrapped_acute_endpoint_cannot_hide_nonacute_euler_segment():
    graph = _pair()
    before = _snapshot(graph)
    # The two opposite phase rates produce a relative turn of approximately
    # -2*pi: its endpoint wraps near zero but its segment crosses pi/2.
    with pytest.raises(ADMISSION_ERRORS, match="initial acute lift"):
        step_relational_exchange(
            graph,
            model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1),
            dt=math.pi**2 / 2,
        )
    assert _snapshot(graph) == before


def test_fresh_pressure_rebuilds_a_stale_maximum_with_an_invalid_node():
    graph = _graph()
    graph.graph.update(_dnfrmax=1e100, _dnfrmax_node="not_a_live_node")
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(2.0), dt=0.0625
    )
    expected = max(abs(value) for value in report.after.pressure)
    assert graph.graph["_dnfrmax"] == expected
    maximum_node = graph.graph["_dnfrmax_node"]
    assert maximum_node in graph
    assert abs(get_attr(graph.nodes[maximum_node], ALIAS_DNFR)) == expected


def test_euler_energy_increase_is_reported_without_false_continuous_monotonicity():
    graph = _pair()
    report = step_relational_exchange(
        graph, model=RelationalExchangeModel(1.0, epi_weight=0, phase_weight=1), dt=0.1
    )
    assert float(report.before.continuous_loss) == 0.0
    assert float(report.energy_change) > 0.0
    assert float(report.energy_step_defect) > 0.0
    assert report.after.epi == report.before.epi
    assert report.after.phase != report.before.phase


def test_sdk_methods_delegate_to_the_same_model_owner(monkeypatch):
    graph = _graph()
    network = Network(graph)
    model = RelationalExchangeModel(2.0)
    captured = []
    marker = object()

    def evaluate(target, *, model):
        captured.append(("evaluate", target, model))
        return marker

    def step(target, *, model, dt, t=None):
        captured.append(("step", target, model, dt, t))
        return marker

    monkeypatch.setattr(relational, "evaluate_relational_exchange", evaluate)
    monkeypatch.setattr(relational, "step_relational_exchange", step)
    assert network.relational_exchange(model) is marker
    assert network.step_relational(model, dt=0.25, t=3.0) is marker
    assert captured == [("evaluate", graph, model), ("step", graph, model, 0.25, 3.0)]
