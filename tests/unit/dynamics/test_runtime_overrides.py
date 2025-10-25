"""Tests for runtime override helpers and integrator resolution."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.dynamics import integrators, runtime
from tnfr.dynamics.types import TNFRGraph


class DummyIntegrator(integrators.AbstractIntegrator):
    """Minimal integrator used to satisfy runtime contracts in tests."""

    def integrate(
        self,
        graph: TNFRGraph,
        *,
        dt: float | None,
        t: float | None,
        method: str | None,
        n_jobs: int | None,
    ) -> None:
        return None


def test_normalize_job_overrides_handles_none_and_suffixes():
    assert runtime._normalize_job_overrides(None) == {}

    overrides = runtime._normalize_job_overrides(
        {
            "dnfr_n_jobs": 2,
            "VF_adapt": "3",
            None: 4,
            "phase": 0,
        }
    )

    assert overrides == {
        "DNFR": 2,
        "VF_ADAPT": "3",
        "PHASE": 0,
    }


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("4", 4),
        (5, 5),
        (None, None),
        (object(), None),
        ("not-a-number", None),
    ],
)
def test_coerce_jobs_value_covers_strings_and_invalid_objects(raw, expected):
    assert runtime._coerce_jobs_value(raw) == expected


@pytest.mark.parametrize(
    "value, allow_non_positive, expected",
    [
        (5, False, 5),
        (0, False, None),
        (0, True, 0),
        (-3, False, None),
        (-3, True, -3),
        (None, False, None),
    ],
)
def test_sanitize_jobs_enforces_policy(value, allow_non_positive, expected):
    assert (
        runtime._sanitize_jobs(value, allow_non_positive=allow_non_positive)
        == expected
    )


def test_resolve_jobs_override_prefers_normalised_override_over_graph_default():
    overrides = runtime._normalize_job_overrides({"dnfr_n_jobs": "8"})

    resolved = runtime._resolve_jobs_override(
        overrides,
        "dnfr",
        2,
        allow_non_positive=False,
    )

    assert resolved == 8


def test_resolve_jobs_override_falls_back_to_graph_and_handles_bad_values():
    overrides = runtime._normalize_job_overrides({"vf_adapt": object()})

    resolved_with_override = runtime._resolve_jobs_override(
        overrides,
        "vf_adapt",
        6,
        allow_non_positive=False,
    )

    # Invalid override is ignored and returns ``None`` instead of raising.
    assert resolved_with_override is None

    resolved_from_graph = runtime._resolve_jobs_override(
        overrides,
        "phase",
        "4",
        allow_non_positive=True,
    )

    assert resolved_from_graph == 4


def test_call_integrator_factory_supports_zero_or_one_positional_argument():
    G = nx.Graph()
    zero_called = False

    def zero_factory():
        nonlocal zero_called
        zero_called = True
        return "zero"

    assert runtime._call_integrator_factory(zero_factory, G) == "zero"
    assert zero_called

    received = None

    def one_factory(graph):
        nonlocal received
        received = graph
        return "one"

    assert runtime._call_integrator_factory(one_factory, G) == "one"
    assert received is G


def test_call_integrator_factory_rejects_multiple_positionals():
    G = nx.Graph()

    def bad_factory(graph, extra):
        raise AssertionError("should not be called")

    with pytest.raises(TypeError, match="at most one positional"):
        runtime._call_integrator_factory(bad_factory, G)


def test_call_integrator_factory_handles_non_introspectable_callable(monkeypatch):
    G = nx.Graph()
    calls = []

    class NonIntrospectable:
        def __call__(self):
            calls.append("called")
            return "sentinel"

    factory = NonIntrospectable()

    def boom(*_args, **_kwargs):
        raise TypeError("cannot inspect")

    monkeypatch.setattr(runtime.inspect, "signature", boom)

    assert runtime._call_integrator_factory(factory, G) == "sentinel"
    assert calls == ["called"]


def test_call_integrator_factory_rejects_keyword_only_requirements():
    G = nx.Graph()

    def kw_only_factory(*, graph):
        return graph

    with pytest.raises(TypeError):
        runtime._call_integrator_factory(kw_only_factory, G)


def test_resolve_integrator_instance_invokes_callable_factories(monkeypatch):
    G = nx.Graph()

    factory = lambda graph: DummyIntegrator()  # noqa: E731
    calls: list[tuple[object, object]] = []

    def fake_call(factory_arg, graph_arg):
        calls.append((factory_arg, graph_arg))
        return DummyIntegrator()

    monkeypatch.setattr(runtime, "_call_integrator_factory", fake_call)
    G.graph["integrator"] = factory

    instance = runtime._resolve_integrator_instance(G)

    assert isinstance(instance, DummyIntegrator)
    assert calls == [(factory, G)]


def test_resolve_integrator_instance_rejects_non_integrator_returns():
    G = nx.Graph()

    def bad_factory():
        return object()

    G.graph["integrator"] = bad_factory

    with pytest.raises(TypeError):
        runtime._resolve_integrator_instance(G)


def test_resolve_integrator_instance_rejects_non_callable_integrator():
    G = nx.Graph()
    G.graph["integrator"] = "not-a-callable"

    with pytest.raises(
        TypeError,
        match="Graph integrator must be an AbstractIntegrator, subclass or callable",
    ):
        runtime._resolve_integrator_instance(G)

    assert runtime._INTEGRATOR_CACHE_KEY not in G.graph
    assert G.graph.pop(runtime._INTEGRATOR_CACHE_KEY, None) is None


def test_resolve_integrator_instance_uses_cache(monkeypatch):
    G = nx.Graph()

    factory = lambda graph: DummyIntegrator()  # noqa: E731
    calls: list[tuple[object, object]] = []

    def fake_call(factory_arg, graph_arg):
        calls.append((factory_arg, graph_arg))
        return DummyIntegrator()

    monkeypatch.setattr(runtime, "_call_integrator_factory", fake_call)
    G.graph["integrator"] = factory

    first = runtime._resolve_integrator_instance(G)
    second = runtime._resolve_integrator_instance(G)

    assert isinstance(first, DummyIntegrator)
    assert first is second
    assert calls == [(factory, G)]
