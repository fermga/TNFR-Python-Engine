"""Direct rollback coverage for mutable graph runtime state."""

from __future__ import annotations

import copyreg
import io
import logging
import re
import threading
from collections import defaultdict, deque
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from functools import partial
from random import Random
from types import MappingProxyType

import networkx as nx
import pytest

from tnfr.errors import TNFRValueError
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.operators.network_stage import GraphTransactionSnapshot
from tnfr.utils import CallbackEvent, CallbackSpec, callback_manager


@pytest.mark.parametrize(
    "value",
    (
        date(2026, 1, 2),
        time(3, 4, 5),
        time(3, 4, 5, tzinfo=timezone.utc),
        datetime(2026, 1, 2, 3, 4, 5),
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        timedelta(days=2, seconds=3),
        timezone.utc,
        timezone(timedelta(hours=2)),
        Decimal("1.25"),
        re.compile(r"^tnfr$"),
    ),
)
def test_snapshot_accepts_common_immutable_c_metadata_and_keeps_identity(
    value: object,
) -> None:
    graph = nx.Graph()
    graph.graph["metadata"] = value
    snapshot = GraphTransactionSnapshot(graph)

    graph.graph["metadata"] = "changed"
    snapshot.restore(graph)

    assert graph.graph["metadata"] is value


def test_snapshot_rejects_primitive_subclass_copy_hooks_before_they_run() -> None:
    graph = nx.Graph()

    class TrapInt(int):
        def __new__(cls, value: int) -> "TrapInt":
            instance = super().__new__(cls, value)
            instance.state = {"value": "before"}
            return instance

        def __deepcopy__(self, _memo: dict[int, object]) -> "TrapInt":
            graph.graph["copy_hook_ran"] = True
            self.state["value"] = "after"
            return self

    trapped = TrapInt(1)
    graph.graph["nested"] = [trapped]

    with pytest.raises(TNFRValueError, match="custom __deepcopy__"):
        GraphTransactionSnapshot(graph)

    assert "copy_hook_ran" not in graph.graph
    assert trapped.state == {"value": "before"}


def test_snapshot_rejects_stateful_primitive_subclass_nested_in_tuple() -> None:
    class StatefulInt(int):
        pass

    value = StatefulInt(1)
    value.state = {"value": "before"}
    graph = nx.Graph()
    graph.graph["nested"] = (value,)

    with pytest.raises(TNFRValueError, match="opaque interpreter state"):
        GraphTransactionSnapshot(graph)


def test_snapshot_rejects_nested_enum_copy_hooks_before_they_run() -> None:
    graph = nx.Graph()

    class TrapEnum(Enum):
        MEMBER = "member"

        def __deepcopy__(self, _memo: dict[int, object]) -> "TrapEnum":
            graph.graph["copy_hook_ran"] = True
            self.state.append("after")
            return self

    TrapEnum.MEMBER.state = ["before"]
    graph.graph["nested"] = (TrapEnum.MEMBER,)

    with pytest.raises(TNFRValueError, match="custom __deepcopy__"):
        GraphTransactionSnapshot(graph)

    assert "copy_hook_ran" not in graph.graph
    assert TrapEnum.MEMBER.state == ["before"]


def test_snapshot_rejects_structured_numpy_void_metadata() -> None:
    np = pytest.importorskip("numpy")
    payload = ["before"]
    base = np.empty(1, dtype=[("payload", object), ("number", "i4")])
    base[0] = (payload, 1)
    structured_scalar = base[0]
    graph = nx.Graph()
    graph.add_node(0, marker=structured_scalar)
    graph.graph["metadata"] = structured_scalar

    with pytest.raises(TNFRValueError, match="cannot be snapshotted"):
        GraphTransactionSnapshot(graph)

    assert structured_scalar["number"] == 1
    assert payload == ["before"]


def test_snapshot_rejects_numpy_record_owned_by_callback() -> None:
    np = pytest.importorskip("numpy")
    record = np.rec.array([(1,)], dtype=[("number", "i4")])[0]

    def refresh(_graph: nx.Graph, default=record) -> None:
        default.number = 7

    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["compute_delta_nfr"] = refresh

    with pytest.raises(TNFRValueError, match="NumPy void/record"):
        GraphTransactionSnapshot(graph)

    assert record.number == 1


def test_snapshot_restores_state_owned_by_numpy_scalar_subclass_callback() -> None:
    np = pytest.importorskip("numpy")

    class StatefulFloat(np.float64):
        pass

    scalar = StatefulFloat(1.0)
    scalar.state = ["before"]

    def refresh(_graph: nx.Graph, default=scalar) -> None:
        default.state.append("mutated")

    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["compute_delta_nfr"] = refresh
    snapshot = GraphTransactionSnapshot(graph)

    refresh(graph)
    snapshot.restore(graph)

    assert scalar.state == ["before"]


def test_snapshot_keeps_safe_exact_numpy_scalar_identity() -> None:
    np = pytest.importorskip("numpy")
    values = (
        np.int64(3),
        np.float64(1.25),
        np.str_("tnfr"),
        np.datetime64("2026-09-10"),
    )
    graph = nx.Graph()
    graph.graph["metadata"] = values
    snapshot = GraphTransactionSnapshot(graph)

    graph.graph["metadata"] = ()
    snapshot.restore(graph)

    assert graph.graph["metadata"] is values


@pytest.mark.parametrize("location", ("runtime", "attribute", "slot"))
def test_snapshot_rejects_mutable_object_array_referents_observationally(
    location: str,
) -> None:
    np = pytest.importorskip("numpy")

    class SlotGraph(nx.Graph):
        __slots__ = ("object_array_slot",)

    graph = SlotGraph()
    graph.add_node(0)
    graph.graph["marker"] = "before"

    class HostilePayload:
        def __deepcopy__(self, _memo: dict[int, object]) -> "HostilePayload":
            graph.graph["marker"] = "copy-hook-ran"
            return self

    payload = HostilePayload()
    array = np.empty(1, dtype=object)
    array[0] = payload
    if location == "runtime":
        graph.graph["_object_cache"] = array
    elif location == "attribute":
        graph.object_array_attribute = array
    else:
        graph.object_array_slot = array

    with pytest.raises(TNFRValueError, match="object array contains mutable"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"
    assert array[0] is payload


def test_snapshot_rejects_mutable_object_array_owned_by_callback_preflight() -> None:
    np = pytest.importorskip("numpy")
    payload = ["before"]
    array = np.empty(1, dtype=object)
    array[0] = payload

    class Refresh:
        def __init__(self) -> None:
            self.calls = 0
            self.cache = array

        def __call__(self, _graph: nx.Graph) -> None:
            self.calls += 1
            payload.append("after")
            raise RuntimeError("callback failed")

    graph = nx.Graph()
    graph.add_node(0)
    callback = Refresh()
    graph.graph["compute_delta_nfr"] = callback

    with pytest.raises(TNFRValueError, match="object array contains mutable"):
        GraphTransactionSnapshot(graph)

    assert callback.calls == 0
    assert payload == ["before"]
    assert array[0] is payload


def test_snapshot_restores_object_array_of_transitively_immutable_values() -> None:
    np = pytest.importorskip("numpy")
    nested = ("tnfr", frozenset((1, Decimal("1.25"))))
    array = np.empty(2, dtype=object)
    array[:] = (nested, None)
    graph = nx.Graph()
    graph.graph["_object_cache"] = array
    snapshot = GraphTransactionSnapshot(graph)

    array[:] = ("changed", "changed")
    snapshot.restore(graph)

    assert graph.graph["_object_cache"] is array
    assert array[0] is nested
    assert array[1] is None


def test_snapshot_restores_object_array_referents_only_for_epi_history() -> None:
    np = pytest.importorskip("numpy")
    delayed = {0: 0.0, 1: 2.0}
    history = np.asarray([delayed], dtype=object)
    graph = nx.Graph()
    graph.graph["_epi_hist"] = history
    graph.graph["delayed_alias"] = delayed
    snapshot = GraphTransactionSnapshot(graph)

    delayed[0] = 9.0
    history[0] = {0: -1.0, 1: -1.0}
    graph.graph["_epi_hist"] = deque(({0: 7.0},), maxlen=2)
    graph.graph["delayed_alias"] = {}
    snapshot.restore(graph)

    assert graph.graph["_epi_hist"] is history
    assert history.shape == (1,)
    assert history.dtype == np.dtype(object)
    assert history[0] is delayed
    assert graph.graph["delayed_alias"] is delayed
    assert delayed == {0: 0.0, 1: 2.0}


def test_snapshot_gives_invalid_epi_history_domain_error_before_opaque_state(
) -> None:
    history = iter(({0: 0.0}, {0: 1.0}))
    graph = nx.Graph()
    graph.graph["_epi_hist"] = history

    with pytest.raises(TNFRValueError, match="replayable indexed history"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["_epi_hist"] is history
    assert next(history) == {0: 0.0}


def test_snapshot_restores_ordinary_callable_bindings_and_owned_state() -> None:
    class SlotGraph(nx.Graph):
        __slots__ = ("function_slot",)

    class Worker:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def record(self, label: str) -> None:
            self.calls.append(label)

    def build_function():
        closure_state: list[str] = []

        def function(default_state: list[str] = []) -> None:
            closure_state.append("called")
            default_state.append("called")
            function.audit.append("called")

        function.audit = []
        return function, closure_state

    graph = SlotGraph()
    worker = Worker()
    python_method = worker.record
    builtin_receiver: list[str] = []
    builtin_method = builtin_receiver.append
    function, closure_state = build_function()
    function_defaults = function.__defaults__
    function_default_state = function_defaults[0]
    function_audit = function.audit
    partial_state: list[str] = []
    partial_callback = partial(worker.record, "partial")
    partial_callback.audit = partial_state

    graph.python_method = python_method
    graph.function_slot = function
    graph.graph["ordinary_builtin"] = builtin_method
    graph.add_node(0, ordinary_partial=partial_callback)
    snapshot = GraphTransactionSnapshot(graph)

    python_method("method")
    builtin_method("builtin")
    function()
    partial_callback()
    partial_state.append("changed")
    function.__defaults__ = (["replacement"],)
    function.audit = ["replacement"]
    partial_callback.audit = ["replacement"]
    snapshot.restore(graph)

    assert graph.python_method is python_method
    assert graph.function_slot is function
    assert graph.graph["ordinary_builtin"] is builtin_method
    assert graph.nodes[0]["ordinary_partial"] is partial_callback
    assert worker.calls == []
    assert builtin_receiver == []
    assert closure_state == []
    assert function.__defaults__ is function_defaults
    assert function_default_state == []
    assert function.audit is function_audit
    assert function_audit == []
    assert partial_callback.audit is partial_state
    assert partial_state == []


def test_snapshot_accepts_canonical_callback_spec_and_restores_callback_state(
) -> None:
    graph = nx.Graph()
    observations: list[str] = []

    def observe(_graph: nx.Graph, _context: object) -> None:
        observations.append("called")

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        observe,
        name="transaction-observer",
    )
    callbacks = graph.graph["callbacks"]
    event_callbacks = callbacks[CallbackEvent.ON_REMESH.value]
    spec = event_callbacks["transaction-observer"]
    assert type(spec) is CallbackSpec

    snapshot = GraphTransactionSnapshot(graph)
    observe(graph, {})
    event_callbacks.clear()
    graph.graph["callbacks"] = {}
    snapshot.restore(graph)

    assert graph.graph["callbacks"] is callbacks
    assert callbacks[CallbackEvent.ON_REMESH.value] is event_callbacks
    assert event_callbacks["transaction-observer"] is spec
    assert spec.func is observe
    assert observations == []


def test_snapshot_rejects_modified_callback_spec_copy_hook_before_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    def observe(_graph: nx.Graph, _context: object) -> None:
        return None

    callback_manager.register_callback(
        graph,
        CallbackEvent.ON_REMESH,
        observe,
        name="hostile-spec",
    )

    def hostile_copy(self, _memo):
        graph.graph["marker"] = "copy-hook-ran"
        return self

    monkeypatch.setattr(CallbackSpec, "__deepcopy__", hostile_copy, raising=False)

    with pytest.raises(TNFRValueError, match="modified copy-protocol hooks"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


def test_snapshot_restores_python_function_owned_state_and_bindings() -> None:
    graph = nx.Graph()
    graph.add_node(0)
    closure_calls = 0
    closure_log: list[str] = []

    def refresh(
        _graph: nx.Graph,
        default_log: list[str] = [],
        *,
        settings: dict[str, str] = {"mode": "before"},
    ) -> None:
        """Refresh test pressure."""

        nonlocal closure_calls
        closure_calls += 1
        closure_log.append("called")
        default_log.append("called")
        settings["mode"] = "after"
        refresh.audit["calls"] += 1
        refresh.__annotations__["mutated"] = True
        refresh.__annotations__ = {"replacement": True}
        refresh.__defaults__ = (["replacement"],)
        refresh.__doc__ = "changed"
        refresh.__kwdefaults__ = {"settings": {"mode": "replacement"}}
        refresh.__module__ = "changed"
        refresh.__name__ = "changed"
        refresh.__qualname__ = "changed"

    def observed_closure_calls() -> int:
        return closure_calls

    refresh.audit = {"calls": 0}
    defaults = refresh.__defaults__
    keyword_defaults = refresh.__kwdefaults__
    audit = refresh.audit
    annotations = refresh.__annotations__
    annotations_before = dict(annotations)
    code = refresh.__code__
    documentation = refresh.__doc__
    module = refresh.__module__
    name = refresh.__name__
    qualified_name = refresh.__qualname__
    type_parameters = getattr(refresh, "__type_params__", None)
    graph.graph["compute_delta_nfr"] = refresh
    snapshot = GraphTransactionSnapshot(graph)

    refresh(graph)
    if hasattr(refresh, "__type_params__"):
        refresh.__type_params__ = ("changed",)
    assert observed_closure_calls() == 1

    snapshot.restore(graph)

    assert graph.graph["compute_delta_nfr"] is refresh
    assert refresh.__defaults__ is defaults
    assert refresh.__kwdefaults__ is keyword_defaults
    assert refresh.__annotations__ is annotations
    assert annotations == annotations_before
    assert refresh.__code__ is code
    assert refresh.__doc__ == documentation
    assert refresh.__module__ == module
    assert refresh.__name__ == name
    assert refresh.__qualname__ == qualified_name
    if hasattr(refresh, "__type_params__"):
        assert refresh.__type_params__ is type_parameters
    assert defaults == ([],)
    assert keyword_defaults == {"settings": {"mode": "before"}}
    assert refresh.audit is audit
    assert audit == {"calls": 0}
    assert closure_log == []
    assert observed_closure_calls() == 0


def test_snapshot_restores_bound_method_receiver_and_live_graph_alias() -> None:
    graph = nx.Graph()
    graph.add_node(0, delta_nfr=0.0)

    class Worker:
        def __init__(self, live_graph: nx.Graph) -> None:
            self.graph = live_graph
            self.calls = 0

        def refresh(self, live_graph: nx.Graph) -> None:
            assert live_graph is self.graph
            self.calls += 1
            live_graph.nodes[0]["delta_nfr"] = 1.0

    worker = Worker(graph)
    callback = worker.refresh
    graph.graph["compute_delta_nfr"] = callback
    snapshot = GraphTransactionSnapshot(graph)

    callback(graph)
    snapshot.restore(graph)

    assert graph.graph["compute_delta_nfr"] is callback
    assert callback.__self__ is worker
    assert worker.calls == 0
    assert worker.graph is graph
    assert graph.nodes[0]["delta_nfr"] == 0.0


def test_snapshot_rolls_back_failing_partial_callable_graph() -> None:
    graph = nx.Graph()
    graph.add_node(0)
    payload: list[str] = []
    settings = {"mode": "before"}

    class Worker:
        def __init__(self) -> None:
            self.calls = 0
            self.callback = None

        def refresh(
            self,
            bound_payload: list[str],
            _graph: nx.Graph,
            *,
            settings: dict[str, str],
        ) -> None:
            self.calls += 1
            bound_payload.append("called")
            settings["mode"] = "after"
            self.callback.keywords["injected"] = True
            self.callback.audit["calls"] += 1
            raise RuntimeError("refresh failed")

    worker = Worker()
    callback = partial(worker.refresh, payload, settings=settings)
    callback.audit = {"calls": 0}
    worker.callback = callback
    keywords = callback.keywords
    audit = callback.audit
    graph.graph["compute_delta_nfr"] = callback
    snapshot = GraphTransactionSnapshot(graph)

    with pytest.raises(RuntimeError, match="refresh failed"):
        callback(graph)
    snapshot.restore(graph)

    assert graph.graph["compute_delta_nfr"] is callback
    assert callback.func.__self__ is worker
    assert callback.args[0] is payload
    assert callback.keywords is keywords
    assert callback.audit is audit
    assert worker.callback is callback
    assert worker.calls == 0
    assert payload == []
    assert settings == {"mode": "before"}
    assert keywords == {"settings": settings}
    assert audit == {"calls": 0}


def test_snapshot_restores_callable_class_and_namespace_identity() -> None:
    class OriginalCallback:
        def __init__(self) -> None:
            self.value = "before"

        def __call__(self, _graph: nx.Graph) -> None:
            pass

    class ReplacementCallback:
        def __call__(self, _graph: nx.Graph) -> None:
            pass

    graph = nx.Graph()
    callback = OriginalCallback()
    namespace = callback.__dict__
    graph.graph["compute_delta_nfr"] = callback
    snapshot = GraphTransactionSnapshot(graph)

    callback.__class__ = ReplacementCallback
    callback.__dict__ = {"value": "before"}

    snapshot.restore(graph)

    assert type(callback) is OriginalCallback
    assert callback.__dict__ is namespace
    assert namespace == {"value": "before"}
    assert graph.graph["compute_delta_nfr"] is callback


def test_snapshot_restores_mutable_node_and_multigraph_key_state() -> None:
    class IdentityKey:
        __hash__ = object.__hash__
        __eq__ = object.__eq__

        def __init__(self, label: str) -> None:
            self.label = label

    left = IdentityKey("left")
    right = IdentityKey("right")
    edge_key = IdentityKey("edge")
    graph = nx.MultiGraph()
    graph.add_edge(left, right, key=edge_key)
    snapshot = GraphTransactionSnapshot(graph)

    left.label = "changed-left"
    right.__dict__ = {"label": "changed-right"}
    edge_key.label = "changed-edge"
    snapshot.restore(graph)

    assert left.label == "left"
    assert right.label == "right"
    assert edge_key.label == "edge"
    assert graph.has_edge(left, right, edge_key)


def test_snapshot_rejects_mutable_structural_hash_keys() -> None:
    class StructuralKey:
        def __init__(self, label: str) -> None:
            self.label = label

        def __hash__(self) -> int:
            return hash(self.label)

        def __eq__(self, other: object) -> bool:
            return self is other

    graph = nx.Graph()
    graph.add_node(StructuralKey("node"))

    with pytest.raises(TNFRValueError, match="object-identity hash"):
        GraphTransactionSnapshot(graph)


def test_snapshot_rejects_opaque_state_behind_builtin_factory() -> None:
    iterator = iter((1, 2, 3))
    graph = nx.Graph()
    graph.add_node(0, payload=defaultdict(iterator.__next__))

    with pytest.raises(TNFRValueError, match="opaque interpreter state"):
        GraphTransactionSnapshot(graph)

    assert next(iterator) == 1


@pytest.mark.parametrize(
    "payload",
    (io.StringIO("a"), io.BytesIO(b"a")),
)
def test_snapshot_rejects_objects_with_unmodeled_c_state(payload) -> None:
    graph = nx.Graph()
    graph.add_node(0, payload=payload)

    with pytest.raises(TNFRValueError, match="opaque interpreter state"):
        GraphTransactionSnapshot(graph)

    assert payload.tell() == 0


def test_snapshot_cannot_restore_a_different_graph() -> None:
    source = nx.Graph()
    source.add_node("source", value=1)
    target = nx.Graph()
    target.add_node("target", value=2)
    source_namespace = source.__dict__
    target_namespace = target.__dict__
    snapshot = GraphTransactionSnapshot(source)

    with pytest.raises(TNFRValueError, match="belongs to a different graph"):
        snapshot.restore(target)

    assert source.__dict__ is source_namespace
    assert target.__dict__ is target_namespace
    assert tuple(source) == ("source",)
    assert tuple(target) == ("target",)


def test_snapshot_is_observational_for_virtual_graph_and_container_reads() -> None:
    class HostileGraph(nx.Graph):
        def is_directed(self) -> bool:
            self.graph["marker"] = "virtual-graph-read"
            return False

        def is_multigraph(self) -> bool:
            self.graph["marker"] = "virtual-graph-read"
            return False

    class HostileDict(dict):
        def items(self):
            graph.graph["marker"] = "virtual-items-read"
            return super().items()

    class HostileDeque(deque):
        def __getattribute__(self, name: str):
            if name == "maxlen":
                graph.graph["marker"] = "virtual-maxlen-read"
            return super().__getattribute__(name)

    graph = HostileGraph()
    graph.add_node(0)
    graph.graph["marker"] = "before"
    callback_state = HostileDict({"queue": HostileDeque((1,), maxlen=2)})

    def refresh(_graph: nx.Graph) -> None:
        del _graph
        callback_state["queue"].append(2)

    refresh.state = callback_state
    graph.graph["compute_delta_nfr"] = refresh

    GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


def test_snapshot_restore_bypasses_graph_attribute_overrides() -> None:
    class HostileGraph(nx.DiGraph):
        def __setattr__(self, name: str, value) -> None:
            if name == "custom":
                raise RuntimeError("virtual setattr")
            super().__setattr__(name, value)

        def __delattr__(self, name: str) -> None:
            raise RuntimeError(f"virtual delattr: {name}")

    graph = HostileGraph()
    object.__setattr__(graph, "custom", "before")
    graph.add_edge(0, 1)
    snapshot = GraphTransactionSnapshot(graph)

    graph.__dict__["custom"] = "after"
    graph.__dict__["transient"] = True
    graph.__dict__["_node"] = {}
    graph.__dict__["_adj"] = {}
    graph.__dict__["_succ"] = {}
    graph.__dict__["_pred"] = {}
    graph.__dict__["graph"] = {"changed": True}

    snapshot.restore(graph)

    assert graph.custom == "before"
    assert "transient" not in graph.__dict__
    assert graph.has_edge(0, 1)
    assert graph._adj is graph._succ
    assert graph.graph == {}


def test_snapshot_restores_random_and_defaultdict_internal_state() -> None:
    graph = nx.Graph()
    generator = Random(1234)
    storage = defaultdict(list, before=[1])

    class Worker:
        def __init__(self) -> None:
            self.generator = generator
            self.storage = storage

        def refresh(self, _graph: nx.Graph) -> None:
            self.generator.random()
            self.storage.default_factory = dict
            self.storage["after"] = [2]

    callback = Worker().refresh
    graph.graph["compute_delta_nfr"] = callback
    generator_state = generator.getstate()
    snapshot = GraphTransactionSnapshot(graph)

    callback(graph)
    snapshot.restore(graph)

    assert generator.getstate() == generator_state
    assert storage.default_factory is list
    assert storage == {"before": [1]}


def test_snapshot_restores_function_factory_defaults_and_closure() -> None:
    closure_state: list[str] = []

    def node_factory(default_state: list[str] = []) -> dict[str, object]:
        closure_state.append("called")
        default_state.append("called")
        return {}

    node_factory.audit = {"calls": 0}
    defaults = node_factory.__defaults__
    audit = node_factory.audit
    graph = nx.Graph()
    graph.node_attr_dict_factory = node_factory
    snapshot = GraphTransactionSnapshot(graph)

    closure_state.append("changed")
    defaults[0].append("changed")
    audit["calls"] = 1
    node_factory.__defaults__ = (["replacement"],)
    snapshot.restore(graph)

    assert graph.node_attr_dict_factory is node_factory
    assert node_factory.__defaults__ is defaults
    assert defaults == ([],)
    assert closure_state == []
    assert node_factory.audit is audit
    assert audit == {"calls": 0}


def test_snapshot_restores_partial_factory_receiver_and_bindings() -> None:
    payload: list[str] = []

    class Factory:
        def __init__(self) -> None:
            self.calls = 0

        def build(self, bound_payload: list[str]) -> dict[str, object]:
            self.calls += 1
            bound_payload.append("called")
            return {}

    receiver = Factory()
    factory = partial(receiver.build, payload)
    factory.audit = {"calls": 0}
    audit = factory.audit
    graph = nx.Graph()
    graph.edge_attr_dict_factory = factory
    snapshot = GraphTransactionSnapshot(graph)

    receiver.calls = 2
    payload.append("changed")
    audit["calls"] = 1
    snapshot.restore(graph)

    assert graph.edge_attr_dict_factory is factory
    assert factory.func.__self__ is receiver
    assert factory.args[0] is payload
    assert receiver.calls == 0
    assert payload == []
    assert factory.audit is audit
    assert audit == {"calls": 0}


def test_snapshot_rejects_unmodelled_opaque_callable_state() -> None:
    graph = nx.Graph()
    cursor = iter((1, 2))

    def refresh(_graph: nx.Graph) -> None:
        next(cursor)

    graph.graph["compute_delta_nfr"] = refresh

    with pytest.raises(TNFRValueError, match="unsupported mutable state"):
        GraphTransactionSnapshot(graph)


def test_snapshot_rejects_custom_deepcopy_without_invoking_it() -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    class HostileCopy:
        def __deepcopy__(self, _memo):
            graph.graph["marker"] = "copy-hook-ran"
            return self

    graph.add_node(0, payload=HostileCopy())

    with pytest.raises(TNFRValueError, match="custom __deepcopy__"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


def test_snapshot_rejects_custom_deepcopy_nested_in_runtime_object() -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    class HostilePayload:
        def __deepcopy__(self, _memo):
            graph.graph["marker"] = "copy-hook-ran"
            return self

    class Integrator:
        def __init__(self) -> None:
            self.payload = HostilePayload()

    graph.graph["integrator"] = Integrator()

    with pytest.raises(TNFRValueError, match="cannot be snapshotted atomically"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


@pytest.mark.parametrize(
    "hook_name",
    (
        "__reduce_ex__",
        "__reduce__",
        "__getstate__",
        "__setstate__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__new__",
    ),
)
def test_snapshot_rejects_copy_protocol_hooks_before_invocation(
    hook_name: str,
) -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    class Trap:
        pass

    trapped = Trap()

    def hook(*_args, **_kwargs):
        graph.graph["marker"] = "copy-hook-ran"
        return (Trap, ())

    setattr(Trap, hook_name, staticmethod(hook) if hook_name == "__new__" else hook)
    graph.graph["nested"] = {"trap": trapped}

    with pytest.raises(TNFRValueError, match="copy-protocol"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


def test_snapshot_rejects_registered_copyreg_reducer_before_invocation() -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    class Trap:
        pass

    trapped = Trap()

    def reducer(_value: Trap):
        graph.graph["marker"] = "copyreg-ran"
        return Trap, ()

    copyreg.pickle(Trap, reducer)
    try:
        graph.add_node(0, trapped=trapped)
        with pytest.raises(TNFRValueError, match="copyreg reducer"):
            GraphTransactionSnapshot(graph)
    finally:
        copyreg.dispatch_table.pop(Trap, None)

    assert graph.graph["marker"] == "before"


def test_snapshot_manually_captures_custom_attribute_lookup_without_calling_it(
) -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"

    class Trap:
        def __init__(self) -> None:
            self.state: list[str] = []

        def __getattribute__(self, name: str):
            graph.graph["marker"] = "attribute-hook-ran"
            return object.__getattribute__(self, name)

    trapped = Trap()
    state = object.__getattribute__(trapped, "state")
    graph.graph["nested"] = trapped
    snapshot = GraphTransactionSnapshot(graph)
    assert graph.graph["marker"] == "before"

    state.append("changed")
    snapshot.restore(graph)

    assert graph.graph["marker"] == "before"
    assert graph.graph["nested"] is trapped
    assert object.__getattribute__(trapped, "state") is state
    assert state == []


def test_snapshot_manually_captures_callable_lookup_without_calling_it() -> None:
    graph = nx.Graph()
    graph.add_node(0)
    graph.graph["marker"] = "before"

    class Trap:
        def __init__(self) -> None:
            object.__setattr__(self, "state", [])

        def __getattribute__(self, name: str):
            graph.graph["marker"] = "attribute-hook-ran"
            return object.__getattribute__(self, name)

        def __call__(self, _graph: nx.Graph) -> None:
            object.__getattribute__(self, "state").append("called")

    callback = Trap()
    state = object.__getattribute__(callback, "state")
    graph.graph["compute_delta_nfr"] = callback
    snapshot = GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"
    state.append("changed")
    snapshot.restore(graph)

    assert graph.graph["marker"] == "before"
    assert graph.graph["compute_delta_nfr"] is callback
    assert object.__getattribute__(callback, "state") is state
    assert state == []


def test_snapshot_rejects_nested_mapping_key_with_nonidentity_hash() -> None:
    graph = nx.Graph()
    graph.graph["marker"] = "before"
    armed = False

    class Key:
        def __hash__(self) -> int:
            if armed:
                graph.graph["marker"] = "hash-hook-ran"
            return 1

        def __eq__(self, other: object) -> bool:
            return self is other

    key = Key()
    nested = {key: "value"}
    graph.graph["nested"] = nested
    armed = True

    with pytest.raises(TNFRValueError, match="object-identity hash"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"
    assert tuple(dict.values(nested)) == ("value",)


def test_snapshot_mapping_proxy_policy_preserves_safe_values_and_rejects_mutable(
) -> None:
    safe = MappingProxyType({"type": "none", "parameters": (0.0, None)})
    safe_graph = nx.Graph()
    safe_graph.graph["_gamma_spec"] = safe
    snapshot = GraphTransactionSnapshot(safe_graph)
    safe_graph.graph["_gamma_spec"] = MappingProxyType({})
    snapshot.restore(safe_graph)
    assert safe_graph.graph["_gamma_spec"] is safe

    payload: list[str] = []
    unsafe = MappingProxyType({"payload": payload})
    unsafe_graph = nx.Graph()
    unsafe_graph.graph["ordinary_proxy"] = unsafe
    with pytest.raises(TNFRValueError, match="exposes mutable referents"):
        GraphTransactionSnapshot(unsafe_graph)
    assert payload == []


def test_snapshot_restores_shared_ordinary_graph_container_identity() -> None:
    sink: list[dict[str, object]] = []
    graph = nx.Graph()
    graph.graph["hybrid_event_log"] = sink
    graph.graph["settings"] = sink
    snapshot = GraphTransactionSnapshot(graph)

    sink.append({"event": "changed"})
    snapshot.restore(graph)

    assert graph.graph["hybrid_event_log"] is sink
    assert graph.graph["settings"] is sink
    assert sink == []


def test_snapshot_rejects_ndarray_finalize_hook_before_capture() -> None:
    np = pytest.importorskip("numpy")
    graph = nx.Graph()
    graph.graph["marker"] = "before"
    armed = False

    class FinalizeTrap(np.ndarray):
        def __array_finalize__(self, _source) -> None:
            if armed:
                graph.graph["marker"] = "array-finalize-ran"

    array = np.arange(2.0).view(FinalizeTrap)
    armed = True
    graph.graph["_array_cache"] = array

    with pytest.raises(TNFRValueError, match="custom array/copy hooks"):
        GraphTransactionSnapshot(graph)

    assert graph.graph["marker"] == "before"


def test_snapshot_bypasses_virtual_builtin_container_mutators() -> None:
    class TrapList(list):
        def clear(self) -> None:
            raise RuntimeError("virtual list clear")

        def extend(self, _values) -> None:
            raise RuntimeError("virtual list extend")

    class TrapDict(dict):
        def clear(self) -> None:
            raise RuntimeError("virtual dict clear")

        def update(self, *_args, **_kwargs) -> None:
            raise RuntimeError("virtual dict update")

    class TrapSet(set):
        def clear(self) -> None:
            raise RuntimeError("virtual set clear")

        def update(self, *_values) -> None:
            raise RuntimeError("virtual set update")

    class TrapDeque(deque):
        def clear(self) -> None:
            raise RuntimeError("virtual deque clear")

        def extend(self, _values) -> None:
            raise RuntimeError("virtual deque extend")

    list_cache = TrapList([1])
    dict_cache = TrapDict({"before": 1})
    set_cache = TrapSet({1})
    deque_cache = TrapDeque((1,), maxlen=3)
    graph = nx.Graph()
    graph.graph.update(
        list_cache=list_cache,
        dict_cache=dict_cache,
        set_cache=set_cache,
        deque_cache=deque_cache,
    )
    snapshot = GraphTransactionSnapshot(graph)

    list.append(list_cache, 2)
    dict.__setitem__(dict_cache, "after", 2)
    set.add(set_cache, 2)
    deque.append(deque_cache, 2)

    snapshot.restore(graph)

    assert list_cache == [1]
    assert dict_cache == {"before": 1}
    assert set_cache == {1}
    assert tuple(deque_cache) == (1,)


@pytest.mark.parametrize("location", ("graph", "node", "edge"))
def test_snapshot_restores_ordinary_lock_metadata_by_identity(location: str) -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1)
    user_lock = threading.Lock()
    if location == "graph":
        metadata = graph.graph
    elif location == "node":
        metadata = graph.nodes[0]
    else:
        metadata = graph.edges[0, 1]
    metadata["user_lock"] = user_lock
    metadata["marker"] = "before"
    snapshot = GraphTransactionSnapshot(graph)

    metadata["user_lock"] = threading.Lock()
    metadata["marker"] = "after"

    snapshot.restore(graph)

    if location == "graph":
        restored = graph.graph
    elif location == "node":
        restored = graph.nodes[0]
    else:
        restored = graph.edges[0, 1]
    assert restored["user_lock"] is user_lock
    assert restored["marker"] == "before"


def test_snapshot_treats_standard_logger_as_external_callable_resource() -> None:
    class Worker:
        def __init__(self) -> None:
            self.logger = logging.getLogger("tnfr.tests.transaction-resource")
            self.calls: list[str] = []

        def __call__(self, _graph: nx.Graph) -> None:
            self.calls.append("called")

    graph = nx.Graph()
    worker = Worker()
    logger = worker.logger
    graph.graph["compute_delta_nfr"] = worker
    snapshot = GraphTransactionSnapshot(graph)

    worker.calls.append("changed")
    worker.logger = logging.getLogger("tnfr.tests.transaction-replacement")

    snapshot.restore(graph)

    assert graph.graph["compute_delta_nfr"] is worker
    assert worker.calls == []
    assert worker.logger is logger


@pytest.mark.parametrize(
    "container_factory",
    (
        pytest.param(lambda logger: [logger], id="list"),
        pytest.param(lambda logger: (logger,), id="tuple"),
        pytest.param(lambda logger: {"logger": logger}, id="dict"),
        pytest.param(lambda logger: {logger}, id="set"),
        pytest.param(lambda logger: frozenset({logger}), id="frozenset"),
    ),
)
def test_snapshot_never_copies_or_restores_nested_standard_logger_state(
    monkeypatch: pytest.MonkeyPatch,
    container_factory,
) -> None:
    logger = logging.Logger("tnfr.tests.nested-transaction-resource")
    logger.setLevel(logging.WARNING)
    reducer_calls: list[logging.Logger] = []

    def trap_reduce(self: logging.Logger):
        reducer_calls.append(self)
        raise AssertionError("Logger.__reduce__ must not run during snapshot")

    monkeypatch.setattr(logging.Logger, "__reduce__", trap_reduce)
    graph = nx.Graph()
    graph.graph["nested_logger"] = container_factory(logger)
    graph.graph["mutable_state"] = ["before"]
    snapshot = GraphTransactionSnapshot(graph)

    logger.setLevel(logging.ERROR)
    graph.graph["mutable_state"].append("changed")
    snapshot.restore(graph)

    nested = graph.graph["nested_logger"]
    members = nested.values() if type(nested) is dict else nested
    assert any(member is logger for member in members)
    assert reducer_calls == []
    assert logger.level == logging.ERROR
    assert graph.graph["mutable_state"] == ["before"]


def test_snapshot_does_not_trust_custom_logger_subclasses_as_resources() -> None:
    class CustomLogger(logging.Logger):
        pass

    graph = nx.Graph()
    logger = CustomLogger("tnfr.tests.custom-logger")
    graph.graph["logger"] = logger

    with pytest.raises(TNFRValueError, match="copy-protocol"):
        GraphTransactionSnapshot(graph)


def test_snapshot_does_not_trust_foreign_networkx_views_as_resources() -> None:
    foreign_graph = nx.path_graph(2)

    class Worker:
        def __init__(self) -> None:
            self.foreign_nodes = foreign_graph.nodes

        def __call__(self, _graph: nx.Graph) -> None:
            return None

    graph = nx.Graph()
    graph.graph["compute_delta_nfr"] = Worker()

    with pytest.raises(TNFRValueError, match="copy-protocol"):
        GraphTransactionSnapshot(graph)


def test_snapshot_restores_array_shapes_dtypes_and_graph_node_aliases() -> None:
    np = pytest.importorskip("numpy")
    graph = nx.Graph()
    direct = np.array([1.0, 2.0])
    tuple_member = np.array([3.0, 4.0])
    tuple_cache = ("marker", tuple_member)
    node_only = np.array([5.0, 6.0])
    graph.add_node(
        0,
        direct_cache_alias=direct,
        tuple_cache_alias=tuple_member,
        node_only_array=node_only,
        marker="before",
    )
    graph.graph.update(
        _direct_cache=direct,
        _tuple_cache=tuple_cache,
        ordinary_marker="before",
    )
    snapshot = GraphTransactionSnapshot(graph)

    direct.dtype = np.int32
    direct.resize((5,), refcheck=False)
    direct[:] = 9
    direct.flags.writeable = False
    tuple_member.resize((3,), refcheck=False)
    tuple_member[:] = 8.0
    node_only.resize((3,), refcheck=False)
    node_only[:] = 7.0
    graph.nodes[0]["marker"] = "after"
    graph.graph["ordinary_marker"] = "after"

    snapshot.restore(graph)

    assert graph.graph["_direct_cache"] is direct
    assert graph.nodes[0]["direct_cache_alias"] is direct
    assert direct.shape == (2,)
    assert direct.dtype == np.dtype(float)
    assert direct.flags.writeable
    assert np.array_equal(direct, np.array([1.0, 2.0]))
    assert graph.graph["_tuple_cache"] is tuple_cache
    assert graph.graph["_tuple_cache"][1] is tuple_member
    assert graph.nodes[0]["tuple_cache_alias"] is tuple_member
    assert tuple_member.shape == (2,)
    assert np.array_equal(tuple_member, np.array([3.0, 4.0]))
    restored_node_only = graph.nodes[0]["node_only_array"]
    assert restored_node_only.shape == (2,)
    assert np.array_equal(restored_node_only, np.array([5.0, 6.0]))
    assert graph.nodes[0]["marker"] == "before"
    assert graph.graph["ordinary_marker"] == "before"


def test_snapshot_rebinds_array_when_tuple_member_refuses_resize() -> None:
    np = pytest.importorskip("numpy")

    class RefusesResize(np.ndarray):
        def resize(self, *args, **kwargs) -> None:
            raise RuntimeError("in-place resize refused")

    array = np.ndarray.__new__(RefusesResize, shape=(2,), dtype=float)
    array[:] = (1.0, 2.0)
    tuple_cache = ("marker", array)
    graph = nx.Graph()
    graph.add_node(0, array_alias=array, marker="before")
    graph.graph.update(
        _tuple_cache=tuple_cache,
        ordinary_marker="before",
    )
    snapshot = GraphTransactionSnapshot(graph)

    np.ndarray.resize(array, (3,), refcheck=False)
    array[:] = 9.0
    graph.nodes[0]["marker"] = "after"
    graph.graph["ordinary_marker"] = "after"

    snapshot.restore(graph)

    restored_tuple = graph.graph["_tuple_cache"]
    restored_array = restored_tuple[1]
    assert restored_tuple is not tuple_cache
    assert restored_array is not array
    assert isinstance(restored_array, RefusesResize)
    assert restored_array.shape == (2,)
    assert np.array_equal(restored_array, np.array([1.0, 2.0]))
    assert graph.nodes[0]["array_alias"] is restored_array
    assert graph.nodes[0]["marker"] == "before"
    assert graph.graph["ordinary_marker"] == "before"


def test_snapshot_restores_standard_container_size_and_type_changes() -> None:
    list_cache = [1, 2]
    dict_cache = {"a": 1}
    set_cache = {1, 2}
    deque_cache = deque((1, 2), maxlen=3)
    graph = nx.Graph()
    graph.graph.update(
        list_cache=list_cache,
        dict_cache=dict_cache,
        set_cache=set_cache,
        deque_cache=deque_cache,
    )
    snapshot = GraphTransactionSnapshot(graph)

    list_cache[:] = ["changed"]
    dict_cache.clear()
    dict_cache["changed"] = 9
    set_cache.clear()
    set_cache.add(9)
    deque_cache.clear()
    deque_cache.append(9)
    graph.graph["list_cache"] = {"replacement": True}
    graph.graph["dict_cache"] = ["replacement"]
    graph.graph["set_cache"] = ("replacement",)
    graph.graph["deque_cache"] = deque((9,), maxlen=1)

    snapshot.restore(graph)

    assert graph.graph["list_cache"] is list_cache
    assert graph.graph["dict_cache"] is dict_cache
    assert graph.graph["set_cache"] is set_cache
    assert graph.graph["deque_cache"] is deque_cache
    assert list_cache == [1, 2]
    assert dict_cache == {"a": 1}
    assert set_cache == {1, 2}
    assert tuple(deque_cache) == (1, 2)
    assert deque_cache.maxlen == 3


def _physical_test_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    for node, epi in zip(graph, (1.0, -1.0), strict=True):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    return graph


def _physical_test_schedule():
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.5,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        (0.25, 0.25),
    )
    return schedule, partition


def test_public_physical_guard_rolls_back_ordinary_bound_method_state() -> None:
    class Helper:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def mutate(self, default_state: list[str] = []) -> None:
            self.calls.append("called")
            default_state.append("called")

    graph = _physical_test_graph()
    helper = Helper()
    helper_method = helper.mutate
    defaults = helper_method.__func__.__defaults__
    default_state = defaults[0]
    graph.helper = helper_method

    def refresh(live_graph: nx.Graph) -> None:
        live_graph.helper()
        for node in live_graph:
            live_graph.nodes[node]["delta_nfr"] = 1.0

    graph.graph["compute_delta_nfr"] = refresh
    schedule, partition = _physical_test_schedule()

    with pytest.raises(TNFRValueError, match="changed non-pressure graph state"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
            suppress_birth_warnings=True,
        )

    assert graph.helper is helper_method
    assert helper.calls == []
    assert helper_method.__func__.__defaults__ is defaults
    assert default_state == []
    assert graph.graph["_t"] == 0.0
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == (1.0, -1.0)
    assert tuple(graph.nodes[node]["delta_nfr"] for node in graph) == (0.0, 0.0)


def test_public_schedule_rejects_copy_protocol_before_any_user_hook() -> None:
    graph = nx.Graph()
    graph.graph.update(_t=0.0, marker="before")

    class Trap:
        pass

    trapped = Trap()

    def reduce_ex(_self, _protocol):
        graph.graph["marker"] = "copy-hook-ran"
        return Trap, ()

    Trap.__reduce_ex__ = reduce_ex
    graph.graph["nested"] = trapped
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.0,),
    )

    with pytest.raises(TNFRValueError, match="copy-protocol"):
        execute_operator_event_schedule(graph, schedule)

    assert graph.graph["marker"] == "before"
    assert graph.graph["nested"] is trapped


def test_public_schedule_rejects_ndarray_finalize_before_any_user_hook() -> None:
    np = pytest.importorskip("numpy")
    graph = nx.Graph()
    graph.graph.update(_t=0.0, marker="before")
    armed = False

    class FinalizeTrap(np.ndarray):
        def __array_finalize__(self, _source) -> None:
            if armed:
                graph.graph["marker"] = "array-finalize-ran"

    array = np.arange(2.0).view(FinalizeTrap)
    armed = True
    graph.graph["ordinary_array"] = array
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(0.0,),
    )

    with pytest.raises(TNFRValueError, match="custom array/copy hooks"):
        execute_operator_event_schedule(graph, schedule)

    assert graph.graph["marker"] == "before"
    assert graph.graph["ordinary_array"] is array


def test_public_physical_schedule_rejects_callback_object_array_preflight() -> None:
    np = pytest.importorskip("numpy")
    graph = _physical_test_graph()
    payload: list[str] = []
    array = np.empty(1, dtype=object)
    array[0] = payload

    class Refresh:
        def __init__(self) -> None:
            self.cache = array
            self.calls = 0

        def __call__(self, _graph: nx.Graph) -> None:
            self.calls += 1
            payload.append("called")

    refresh = Refresh()
    graph.graph["compute_delta_nfr"] = refresh
    schedule, partition = _physical_test_schedule()

    with pytest.raises(TNFRValueError, match="object array contains mutable"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
            suppress_birth_warnings=True,
        )

    assert refresh.calls == 0
    assert payload == []
    assert array[0] is payload
    assert graph.graph["_t"] == 0.0


def test_public_physical_schedule_rejects_mutable_mapping_proxy_preflight() -> None:
    graph = _physical_test_graph()
    payload: list[str] = []
    graph.graph["ordinary_proxy"] = MappingProxyType({"payload": payload})
    callback_calls = 0

    def refresh(live_graph: nx.Graph) -> None:
        nonlocal callback_calls
        callback_calls += 1
        live_graph.graph["ordinary_proxy"]["payload"].append("called")

    graph.graph["compute_delta_nfr"] = refresh
    schedule, partition = _physical_test_schedule()

    with pytest.raises(TNFRValueError, match="exposes mutable referents"):
        execute_operator_event_schedule(
            graph,
            schedule,
            physical_flow_partitions=(partition,),
            suppress_birth_warnings=True,
        )

    assert callback_calls == 0
    assert payload == []
    assert graph.graph["_t"] == 0.0
