"""Complete runtime handoffs retain the selected clock and live producers."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY, inject_defaults
from tnfr.dynamics import runtime
from tnfr.gamma import GAMMA_REGISTRY, GammaEntry
from tnfr.mathematics.backend import get_backend
from tnfr.mathematics.dynamics import MathematicalDynamicsEngine
from tnfr.metrics import diagnosis
from tnfr.metrics.core import _metrics_step, register_metrics_callbacks
from tnfr.observers import attach_standard_observer
from tnfr.structural import create_math_nfr
from tnfr.utils import CallbackEvent, callback_manager


def _graph():
    graph = nx.empty_graph(1)
    inject_defaults(graph)
    graph.graph.update(
        DT=0.125, DT_MIN=0.0, CALLBACKS_STRICT=True, PHASE_ADAPT={"enabled": False}
    )
    graph.nodes[0].update(EPI=0.1, nu_f=0.0, theta=0.0, delta_nfr=0.0, Si=0.5)
    return graph


@pytest.mark.parametrize("entry", ["step", "run"])
@pytest.mark.parametrize("explicit_dt", [None, 0.25])
@pytest.mark.parametrize("replacement", [9.0, True])
def test_source_side_effect_cannot_change_the_already_selected_auxiliary_clock(
    monkeypatch, entry, explicit_dt, replacement
):
    graph = _graph()
    calls = []

    def source(live_graph, node, time, config):
        calls.append((node, time))
        live_graph.graph["DT"] = replacement
        return 0.2

    monkeypatch.setitem(GAMMA_REGISTRY, "boundary_clock", GammaEntry(source, False))
    graph.graph["GAMMA"] = {"type": "boundary_clock"}
    seen = []
    monkeypatch.setattr(
        runtime,
        "_advance_math_engine",
        lambda _graph, *, dt, step_idx, hist: seen.append((dt, step_idx)),
    )
    if entry == "step":
        runtime.step(graph, dt=explicit_dt, use_Si=False, apply_glyphs=False)
    else:
        runtime.run(graph, 1, dt=explicit_dt, use_Si=False, apply_glyphs=False)
    consumed = 0.125 if explicit_dt is None else explicit_dt
    assert seen == [(consumed, 0)]
    assert graph.graph["_t"] == consumed
    # The supplied source remains active even at zero nodal capacity.
    assert graph.nodes[0]["EPI"] == pytest.approx(0.1 + 0.2 * consumed)
    assert calls == [(0, 0.0)]
    assert graph.graph["DT"] is replacement


@pytest.mark.parametrize("verbosity", ["basic", "detailed", "debug"])
@pytest.mark.parametrize("observer_first", [False, True])
def test_metrics_registration_is_idempotent_for_all_tiers_and_callback_orders(
    verbosity, observer_first
):
    graph = _graph()
    graph.graph["METRICS"]["verbosity"] = verbosity
    if observer_first:
        attach_standard_observer(graph)
    register_metrics_callbacks(graph)
    if not observer_first:
        attach_standard_observer(graph)
    before = dict(graph.graph["callbacks"]["after_step"])
    for _ in range(3):
        register_metrics_callbacks(graph)
    assert graph.graph["callbacks"]["after_step"] == before
    runtime.step(graph, dt=0.125, use_Si=False, apply_glyphs=False)
    assert graph.graph["_stability_observation_revision"] == 1
    assert graph.graph["_std_observer_reorg"]["status"] == "recorded"
    if verbosity == "debug":
        assert len(graph.graph["history"]["nodal_diag"]) == 1


def test_registered_diagnosis_reads_live_jobs_and_keeps_explicit_override(monkeypatch):
    graph = _graph()
    graph.graph["DIAGNOSIS"]["enabled"] = False
    graph.graph["DIAGNOSIS_N_JOBS"] = 2
    diagnosis.register_diagnosis_callbacks(graph)
    callback = graph.graph["callbacks"]["after_step"]["diagnosis_step"].func
    observed = []
    original = diagnosis._coerce_jobs

    def capture(raw):
        observed.append(raw)
        return original(raw)

    monkeypatch.setattr(diagnosis, "_coerce_jobs", capture)
    graph.graph["DIAGNOSIS_N_JOBS"] = 3
    callback(graph, {})
    diagnosis._diagnosis_step(graph, {}, n_jobs=4)
    assert observed == [3, 4]


def test_metrics_registration_does_not_weaken_conflicting_callback_rejection():
    graph = _graph()
    graph.graph["METRICS"]["verbosity"] = "debug"
    callback_manager.register_callback(
        graph, CallbackEvent.AFTER_STEP, lambda *_: None, name="diagnosis_step"
    )
    original = deepcopy(graph.graph["callbacks"]["after_step"]["diagnosis_step"])
    with pytest.raises(ValueError, match="already registered"):
        register_metrics_callbacks(graph)
    assert graph.graph["callbacks"]["after_step"]["diagnosis_step"] == original


@pytest.mark.parametrize(
    "invalid", [True, "0.1", Fraction(1, 2**2000), -0.1, float("nan")]
)
def test_auxiliary_tolerance_rejects_before_advancing_or_publishing(
    monkeypatch, invalid
):
    graph, _ = create_math_nfr(
        "runtime-node",
        dimension=2,
        spectral_spectrum=[4.0, 4.0],
        spectral_expectation_threshold=3.0,
    )
    config = graph.graph["MATH_ENGINE"]
    config["atol"] = invalid
    original_keys = tuple(config)
    state = config.get("_state_vector")
    history = {}

    def forbidden_step(*args, **kwargs):
        pytest.fail("invalid tolerance must reject before auxiliary evolution")

    monkeypatch.setattr(runtime.MathematicalDynamicsEngine, "step", forbidden_step)
    with pytest.raises((TypeError, ValueError)):
        runtime._advance_math_engine(graph, dt=0.1, step_idx=0, hist=history)
    assert tuple(config) == original_keys
    assert config.get("_state_vector") is state
    assert history == {}


def test_auxiliary_spectral_summary_uses_current_threshold_not_cached_validator():
    graph, _ = create_math_nfr(
        "runtime-node",
        dimension=2,
        spectral_spectrum=[4.0, 4.0],
        spectral_expectation_threshold=3.0,
    )
    config = graph.graph["MATH_ENGINE"]
    history = {}
    runtime._advance_math_engine(graph, dt=0.1, step_idx=0, hist=history)
    assert config["last_summary"]["spectral_operator_expectation"]["passed"] is True
    config["spectral_expectation_threshold"] = config["coherence_threshold"] = 5.0
    runtime._advance_math_engine(graph, dt=0.1, step_idx=1, hist=history)
    assert config["last_summary"]["spectral_operator_expectation"]["passed"] is False
    assert config["last_summary"]["spectral_operator_expectation"]["threshold"] == 5.0


def test_auxiliary_continuation_admits_native_torch_state_without_implicit_numpy():
    torch = pytest.importorskip("torch")
    graph, _ = create_math_nfr(
        "runtime-node",
        dimension=2,
        spectral_spectrum=[4.0, 4.0],
        spectral_expectation_threshold=3.0,
    )
    config = graph.graph["MATH_ENGINE"]
    backend = get_backend("torch")
    config["dynamics_engine"] = MathematicalDynamicsEngine(
        [[0.0, 0.0], [0.0, 0.0]], hilbert_space=config["hilbert_space"], backend=backend
    )
    native = backend.as_array([1.0 + 0.0j, 0.0 + 0.0j]).requires_grad_()
    config["_state_vector"] = native
    with pytest.raises((RuntimeError, TypeError)):
        runtime.np.asarray(native)
    history = {}
    for index in range(2):
        runtime._advance_math_engine(graph, dt=0.125, step_idx=index, hist=history)
    assert isinstance(config["_state_vector"], torch.Tensor)
    assert config["_state_vector"].requires_grad
    assert len(history["math_engine_summary"]) == 2
    assert history["math_engine_norm"] == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("channel", [EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY])
@pytest.mark.parametrize("invalid", [True, "0.5", None, Fraction(1, 2**2000)])
def test_auxiliary_initialization_never_coerces_or_omits_an_invalid_node(
    channel, invalid
):
    graph, node = create_math_nfr("runtime-node", dimension=2)
    config = graph.graph["MATH_ENGINE"]
    graph.nodes[node][channel] = invalid
    # Leave another well-formed constituent: silent filtering would otherwise
    # build and evolve a normalized auxiliary state from this node alone.
    graph.add_node("valid", EPI=0.5, nu_f=1.0, theta=0.0)
    config.pop("_state_vector", None)
    history = {}
    with pytest.raises((TypeError, ValueError)):
        runtime._advance_math_engine(graph, dt=0.125, step_idx=0, hist=history)
    assert "_state_vector" not in config
    assert history == {}


@pytest.mark.parametrize(
    "invalid", [True, "0.125", Fraction(1, 2**2000), -0.1, float("nan")]
)
def test_advanced_metrics_admit_their_configured_clock_before_observation_writes(
    invalid,
):
    graph = _graph()
    graph.graph["METRICS"]["verbosity"] = "debug"
    graph.graph["DT"] = invalid
    before_graph = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    with pytest.raises((TypeError, ValueError)):
        _metrics_step(graph)
    assert graph.graph == before_graph
    assert dict(graph.nodes(data=True)) == before_nodes


@pytest.mark.parametrize("verbosity", ["basic", "detailed"])
def test_metrics_without_duration_diagnostics_do_not_consume_configured_clock(
    verbosity,
):
    graph = _graph()
    graph.graph["METRICS"]["verbosity"] = verbosity
    graph.graph["DT"] = "unused duration input"
    _metrics_step(graph)
    assert graph.graph["_stability_observation_revision"] == 1
