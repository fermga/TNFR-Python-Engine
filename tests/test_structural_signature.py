"""Adversarial tests for internal proof-value signatures."""

from __future__ import annotations

import struct
from collections import defaultdict
from functools import partial
from random import Random

import pytest

from tnfr.mathematics.unified_numerical import np
from tnfr.utils._structural_signature import (
    StructuralSignatureError,
    proof_stamps_are_identical,
    structural_object_state_signature,
    structural_proof_signature,
)


def _assert_immutable_tree(value: object) -> None:
    assert type(value) in {bool, bytes, int, str, tuple}
    if type(value) is tuple:
        for item in value:
            _assert_immutable_tree(item)


def test_proof_stamp_comparison_is_bit_faithful_and_requires_exact_tuples() -> None:
    positive_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    same_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    other_nan = struct.unpack(">d", bytes.fromhex("7ff8000000000002"))[0]

    assert proof_stamps_are_identical(("v1", 1), ("v1", 1))
    assert not proof_stamps_are_identical(["v1", 1], ("v1", 1))
    assert not proof_stamps_are_identical(("v1", -0.0), ("v1", 0.0))
    assert proof_stamps_are_identical(("v1", positive_nan), ("v1", same_nan))
    assert not proof_stamps_are_identical(
        ("v1", positive_nan),
        ("v1", other_nan),
    )


def test_proof_stamp_comparison_never_dispatches_injected_equality() -> None:
    calls: list[str] = []

    class HostileEquality:
        def __eq__(self, other: object) -> bool:
            del other
            calls.append("equality dispatched")
            raise SystemExit("proof comparison must fail closed")

    observed = ("v1", HostileEquality())

    assert not proof_stamps_are_identical(observed, ("v1", ("safe",)))
    assert calls == []


def test_custom_object_signature_never_dispatches_class_lookup() -> None:
    calls: list[str] = []

    class HostileLookup:
        def __getattribute__(self, name: str) -> object:
            calls.append(name)
            raise SystemExit("signature capture must bypass custom lookup")

    signature = structural_proof_signature(HostileLookup())

    _assert_immutable_tree(signature)
    assert calls == []


def test_signature_is_immutable_and_preserves_binary_bits_aliases_and_cycles():
    negative_nan = struct.unpack(">d", bytes.fromhex("fff8000000000001"))[0]
    shared = [complex(-0.0, negative_nan)]
    value = [shared, shared]
    value.append(value)

    signature = structural_proof_signature(value)

    _assert_immutable_tree(signature)
    assert signature == structural_proof_signature(value)
    assert signature != structural_proof_signature(
        [[complex(0.0, negative_nan)], [complex(0.0, negative_nan)]]
    )
    shared.append(1)
    assert signature != structural_proof_signature(value)

    shared_bytes = bytearray(b"x")
    assert structural_proof_signature(
        (shared_bytes, shared_bytes)
    ) != structural_proof_signature((bytearray(b"x"), bytearray(b"x")))

    shared_view = memoryview(b"x")
    assert structural_proof_signature(
        (shared_view, shared_view)
    ) != structural_proof_signature((memoryview(b"x"), memoryview(b"x")))


def test_mapping_and_set_signatures_ignore_insertion_order_and_accept_tuple_keys():
    left = {("b", 2): {3, 1}, ("a", 1): {4, 2}}
    right = {("a", 1): {2, 4}, ("b", 2): {1, 3}}

    assert structural_proof_signature(left) == structural_proof_signature(right)


def test_hostile_methods_are_not_called_and_object_state_is_observed():
    class HostileNode:
        __slots__ = ("payload",)

        def __repr__(self):
            raise AssertionError("repr must not run")

        def __hash__(self):
            raise AssertionError("hash must not run")

        def __eq__(self, other):
            raise AssertionError("equality must not run")

    node = HostileNode()
    node.payload = 1
    before = structural_proof_signature(node)
    node.payload = 2

    assert before != structural_proof_signature(node)


def test_container_subclasses_bypass_overrides_and_include_custom_state():
    class HostileList(list):
        def __iter__(self):
            raise AssertionError("custom iterator must not run")

    class HostileDict(dict):
        def __iter__(self):
            raise AssertionError("custom iterator must not run")

        def items(self):
            raise AssertionError("custom items must not run")

    sequence = HostileList([1, 2])
    sequence.marker = "left"
    mapping = HostileDict({"a": 1})
    mapping.marker = "left"

    sequence_signature = structural_proof_signature(sequence)
    mapping_signature = structural_proof_signature(mapping)
    sequence.marker = "right"
    mapping.marker = "right"

    assert sequence_signature != structural_proof_signature(sequence)
    assert mapping_signature != structural_proof_signature(mapping)


def test_custom_namespace_and_slot_descriptors_fail_closed_without_running():
    calls: list[str] = []

    class CustomNamespace:
        @property
        def __dict__(self):
            calls.append("namespace")
            raise AssertionError("custom descriptor must not run")

    class Slotted:
        __slots__ = ("value",)

    slotted = Slotted()
    slotted.value = 1

    def read_slot(_value):
        calls.append("slot")
        raise AssertionError("custom descriptor must not run")

    Slotted.value = property(read_slot)

    with pytest.raises(StructuralSignatureError):
        structural_proof_signature(CustomNamespace())
    with pytest.raises(StructuralSignatureError):
        structural_proof_signature(slotted)
    assert calls == []


def test_custom_slot_iterable_and_class_metadata_fail_closed_without_running():
    calls: list[str] = []

    class SlotNames:
        def __iter__(self):
            calls.append("slots")
            raise AssertionError("custom iterator must not run")

    class Value:
        __slots__ = ("payload",)

    value = Value()
    value.payload = 1
    Value.__slots__ = SlotNames()
    with pytest.raises(StructuralSignatureError):
        structural_proof_signature(value)

    class InvalidModule:
        pass

    InvalidModule.__module__ = object()
    with pytest.raises(StructuralSignatureError):
        structural_proof_signature(InvalidModule())

    assert calls == []


def test_custom_metaclass_metadata_descriptor_is_not_invoked():
    calls: list[str] = []

    class Metadata(type):
        @property
        def __module__(cls):
            calls.append("metadata")
            raise AssertionError("custom descriptor must not run")

    class Value(metaclass=Metadata):
        pass

    with pytest.raises(StructuralSignatureError):
        structural_proof_signature(Value())
    assert calls == []


def test_class_metadata_mutation_changes_signature():
    class Value:
        pass

    value = Value()
    before = structural_proof_signature(value)
    Value.__module__ = "moved.module"

    assert before != structural_proof_signature(value)


def test_explicit_opaque_reference_preserves_alias_without_hiding_owner_state():
    external = {"pressure": 0.0}

    class Holder:
        def __init__(self) -> None:
            self.external = external
            self.calls: list[float] = []

    holder = Holder()
    before = structural_object_state_signature(
        holder,
        opaque_references=(external,),
    )

    external["pressure"] = 1.0
    assert before == structural_object_state_signature(
        holder,
        opaque_references=(external,),
    )

    holder.calls.append(0.0)
    assert before != structural_object_state_signature(
        holder,
        opaque_references=(external,),
    )

    proof_before = structural_proof_signature(
        (holder, external),
        opaque_references=(external,),
    )
    external["pressure"] = 2.0
    assert proof_before == structural_proof_signature(
        (holder, external),
        opaque_references=(external,),
    )


def test_identity_sensitive_state_detects_equal_value_rebinding() -> None:
    class Holder:
        pass

    holder = Holder()
    original = []
    holder.value = original
    value_signature = structural_object_state_signature(holder)
    identity_signature = structural_object_state_signature(
        holder,
        identity_sensitive=True,
    )

    holder.value = []
    assert value_signature == structural_object_state_signature(holder)
    assert identity_signature != structural_object_state_signature(
        holder,
        identity_sensitive=True,
    )

    namespace = holder.__dict__
    namespace_signature = structural_object_state_signature(holder)
    holder.__dict__ = dict(namespace)
    assert namespace_signature != structural_object_state_signature(holder)


@pytest.mark.skipif(np is None, reason="NumPy is unavailable")
def test_numpy_array_signature_observes_numeric_and_object_state() -> None:
    numeric = np.array([1.0, 2.0])
    numeric_before = structural_proof_signature(numeric)
    numeric[0] = 3.0
    assert numeric_before != structural_proof_signature(numeric)

    nested = [1]
    objects = np.empty(1, dtype=object)
    objects[0] = nested
    objects_before = structural_proof_signature(objects)
    nested.append(2)
    assert objects_before != structural_proof_signature(objects)

    payload = tuple(np.array([float(index)]) for index in range(32))
    assert structural_proof_signature(payload) == structural_proof_signature(
        payload
    )

    cyclic = np.empty(1, dtype=object)
    cyclic[0] = cyclic
    cyclic_signature = structural_proof_signature(cyclic)
    assert cyclic_signature == structural_proof_signature(cyclic)

    typed = np.array([1], dtype=np.dtype("i4", metadata={"unit": "a"}))
    typed_before = structural_proof_signature(typed, identity_sensitive=True)
    typed.dtype = np.dtype("i4", metadata={"unit": "b"})
    assert typed_before != structural_proof_signature(
        typed,
        identity_sensitive=True,
    )

    metadata_holder: list[object] = []
    metadata_cycle = np.array(
        [1],
        dtype=np.dtype("i4", metadata={"holder": metadata_holder}),
    )
    metadata_holder.append(metadata_cycle)
    assert structural_proof_signature(
        metadata_cycle
    ) == structural_proof_signature(metadata_cycle)


def test_mapping_signature_canonically_orders_bit_identical_nan_keys() -> None:
    first = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    second = struct.unpack(">d", bytes.fromhex("7ff8000000000001"))[0]
    mapping = {first: "first", second: "second"}
    before = structural_proof_signature(mapping)

    first_value = mapping.pop(first)
    mapping[first] = first_value

    assert structural_proof_signature(mapping) == before


def test_random_generator_signature_observes_internal_state() -> None:
    generator = Random(123)
    generator.audit = []
    before = structural_proof_signature(generator)
    generator.random()

    assert before != structural_proof_signature(generator)

    generator = Random(123)
    generator.audit = []
    before = structural_proof_signature(generator)
    generator.audit.append("changed")
    assert before != structural_proof_signature(generator)

    generator = Random(123)
    random_cycle: list[object] = [generator]
    generator.gauss_next = random_cycle
    assert structural_proof_signature(generator) == structural_proof_signature(
        generator
    )


def test_defaultdict_signature_observes_default_factory_binding() -> None:
    mapping = defaultdict(list, {"value": [1]})
    before = structural_proof_signature(mapping, identity_sensitive=True)

    mapping.default_factory = set

    assert before != structural_proof_signature(
        mapping,
        identity_sensitive=True,
    )


def test_defaultdict_signature_observes_factory_owned_behavior_state() -> None:
    closure = ["a"]

    def from_closure() -> str:
        return closure[0]

    closure_mapping = defaultdict(from_closure)
    closure_before = structural_proof_signature(
        closure_mapping,
        identity_sensitive=True,
    )
    closure[0] = "b"
    assert closure_before != structural_proof_signature(
        closure_mapping,
        identity_sensitive=True,
    )

    def from_keyword(*, value: str) -> str:
        return value

    factory = partial(from_keyword, value="a")
    partial_mapping = defaultdict(factory)
    partial_before = structural_proof_signature(
        partial_mapping,
        identity_sensitive=True,
    )
    factory.keywords["value"] = "b"
    assert partial_before != structural_proof_signature(
        partial_mapping,
        identity_sensitive=True,
    )


def test_builtin_callable_signature_observes_definition_and_module_binding() -> None:
    receiver = [1]
    mapping = defaultdict(receiver.copy)
    before = structural_proof_signature(mapping, identity_sensitive=True)

    mapping.default_factory = receiver.clear

    assert before != structural_proof_signature(
        mapping,
        identity_sensitive=True,
    )
    assert structural_proof_signature(len) == structural_proof_signature(len)
    assert structural_proof_signature(len) != structural_proof_signature(sum)


def test_ephemeral_opaque_references_remain_live_during_traversal() -> None:
    def ephemeral_references():
        values = [[] for _ in range(200)]
        yield from values
        values.clear()

    generator = Random(1)
    expected = structural_proof_signature(generator)

    for _ in range(20):
        assert structural_proof_signature(
            generator,
            opaque_references=ephemeral_references(),
        ) == expected


def test_retained_references_prevent_identity_reuse_across_comparison() -> None:
    class Holder:
        pass

    holder = Holder()
    holder.value = {}
    retained: list[object] = []
    before = structural_object_state_signature(
        holder,
        identity_sensitive=True,
        _retained_references=retained,
    )
    original_identity = id(holder.value)

    del holder.value
    holder.value = {}

    assert retained
    assert id(holder.value) != original_identity
    assert before != structural_object_state_signature(
        holder,
        identity_sensitive=True,
    )


@pytest.mark.skipif(np is None, reason="NumPy is unavailable")
def test_numpy_dtype_root_signature_observes_metadata_state() -> None:
    holder: list[object] = []
    dtype = np.dtype("i4", metadata={"holder": holder})
    before = structural_proof_signature(dtype)

    holder.append(1)

    assert before != structural_proof_signature(dtype)

    nested: dict[str, object] = {"value": {}}
    typed = np.dtype("i4", metadata={"nested": nested})
    retained: list[object] = []
    identity_before = structural_proof_signature(
        typed,
        identity_sensitive=True,
        _retained_references=retained,
    )
    assert identity_before == structural_proof_signature(
        typed,
        identity_sensitive=True,
    )
    nested["value"] = {}
    assert identity_before != structural_proof_signature(
        typed,
        identity_sensitive=True,
    )
